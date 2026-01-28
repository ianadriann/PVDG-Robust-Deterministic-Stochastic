"""
catatan: saya menggunkan PV-only: supaya kenaikan “robustness” langsung terlihat sebagai kenaikan kapasitas PV.
nanti di paper bisa jelaskan dengan kalimat sederhana seperti: 
“In the robust formulation, we require the average PV output plus a slack reserve 
to cover a µ+κσ estimate of the system load, so that PV installations themselves provide 
robustness against demand uncertainty, rather than relying on the upstream grid.”
"""
#kode sebelum perubahan (23 Februari 2026)



from gurobipy import Model, GRB, quicksum

def build_robust_pv_model(
    name, pv_buses, all_buses, hours, lines, df_pv, robust_load_bh, L_0, edges_by_to, edges_by_from, slack_bus,
    x_max, x_min, n_max, V2_min, V2_max, pf_min, alpha_pv, beta_grid,
    total_pv_cap_max=60000,
    solve=False
):

    # 0. Buat model
    model_rob = Model(f"Robust_{name}")  # name = 'Low', 'Base', 'High'
    model_rob.setParam("OutputFlag", 0)

    # 1. Variabel keputusan
    x_rob              = model_rob.addVars(pv_buses, vtype=GRB.CONTINUOUS, name="PV_Capacity")  # kW
    y_rob              = model_rob.addVars(pv_buses, vtype=GRB.BINARY,    name="PV_Install")
    P_grid_rob         = model_rob.addVars(hours, vtype=GRB.CONTINUOUS, name="Grid_Import")  # MW
    V2_rob             = model_rob.addVars(all_buses, hours, lb=V2_min, ub=V2_max, vtype=GRB.CONTINUOUS, name="V2")
    P_line_rob         = model_rob.addVars(len(lines), hours, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="P_line")
    P_pv_rob           = model_rob.addVars(pv_buses, hours, vtype=GRB.CONTINUOUS, name="PV_Output")


    # 2. Fungsi objektif: kapasitas PV + impor grid + reserve penalty
    model_rob.setObjective(
        alpha_pv * quicksum(x_rob[i] for i in pv_buses) +
        beta_grid * quicksum(P_grid_rob[h] for h in hours),
        GRB.MINIMIZE
        )
    
    df_pv_rob = df_pv.groupby(['Hour','Bus'])['PV Output Factor'].mean().reset_index()
    pv_profile_rob = {(int(r['Hour']), int(r['Bus'])): float(r['PV Output Factor'])
                    for _, r in df_pv_rob.iterrows()}


    # 3. Kendala PV (C1)
    for i in pv_buses:
        for h in hours:
            profile = pv_profile_rob.get((h, i), 0.0)
            model_rob.addConstr(P_pv_rob[i, h] <= x_rob[i] * profile / 1000.0,
                                name=f"pv_output_{i}_{h}")


    # (Cgrid-rob) Grid import tidak boleh negatif
    for h in hours:
        model_rob.addConstr(P_grid_rob[h] >= 0.0, name=f"grid_import_nonneg_{h}")


    # 4. Kendala kapasitas & siting PV (C3, C4)
    for i in pv_buses:
        model_rob.addConstr(x_rob[i] <= x_max * y_rob[i], name=f"x_le_ymax_{i}")
        model_rob.addConstr(x_rob[i] >= x_min * y_rob[i], name=f"x_ge_ymin_{i}")

    model_rob.addConstr(quicksum(y_rob[i] for i in pv_buses) <= n_max, name="max_selected_sites")


    # 5. Batas total & per bus (60 MW, 3×L0)  ← tetap
    model_rob.addConstr(quicksum(x_rob[i] for i in pv_buses) <= total_pv_cap_max, name="total_pv_cap_le_60MW")
    for i in pv_buses:
        max_capacity_i = L_0[i] * 1000 * 3  # 3x beban dasar (MW→kW)
        model_rob.addConstr(x_rob[i] <= max_capacity_i, name=f"cap_per_bus_{i}")

    # 6. LinDistFlow-Lite constraints
    # (1) Slack voltage reference
    for h in hours:
        model_rob.addConstr(V2_rob[slack_bus, h] == 1.0, name=f"slack_V2_{h}")

    # (2) Voltage drop (linear, lossless-ish)
    for e, (u, v, R, S_MVA) in enumerate(lines):
        for h in hours:
            model_rob.addConstr(V2_rob[v, h] == V2_rob[u, h] - 2.0 * R * P_line_rob[e, h],
                                name=f"vdrop_e{e}_h{h}")

    # (3) Thermal limits (pakai S_MVA dan pf_min → batas P aktif)
    for e, (u, v, R, S_MVA) in enumerate(lines):
        P_lim = pf_min * S_MVA  # MW
        for h in hours:
                model_rob.addConstr(P_line_rob[e, h] <=  P_lim, name=f"thermal_up_e{e}_h{h}")
                model_rob.addConstr(P_line_rob[e, h] >= -P_lim, name=f"thermal_lo_e{e}_h{h}")

    # (4) Nodal power balance (aktif, lossless) - ROBUST LOAD
    for i in all_buses:
        for h in hours:
            inflow  = quicksum(P_line_rob[e, h] for e in edges_by_to[i])
            outflow = quicksum(P_line_rob[e, h] for e in edges_by_from[i])
            load_i  = robust_load_bh.get((h, i), 0.0)

            grid_term = P_grid_rob[h] if i == slack_bus else 0.0
            pv_term   = P_pv_rob[i, h] if i in pv_buses else 0.0

            model_rob.addConstr(
                pv_term + inflow - outflow + grid_term == load_i,
                name=f"nodal_balance_i{i}_h{h}"
            )

        
    # (5) s tegangan (Vmin–Vmax)
    for i in all_buses:
        for h in hours:
            model_rob.addConstr(V2_rob[i, h] >= V2_min, name=f"vmin_i{i}_h{h}")
            model_rob.addConstr(V2_rob[i, h] <= V2_max, name=f"vmax_i{i}_h{h}")



    # 8. Solve + ambil hasil (x, y, tegangan, loading, adeq, dst.)
    if solve:
        model_rob.optimize()
    

    # Kembalikan model + semua variabel biar mudah diakses
    vars_rob = {
        "x_rob": x_rob,
        "y_rob": y_rob,
        "P_grid_rob": P_grid_rob,
        "V2_rob": V2_rob,
        "P_line_rob": P_line_rob,
        "P_pv_rob": P_pv_rob,
    }


    return model_rob, vars_rob



#======
"""
# 0. Buat model
model_rob = Model(f"Robust_{name}")  # name = 'Low', 'Base', 'High'
model_rob.setParam("OutputFlag", 0)

# 1. Variabel keputusan
x_rob              = model_rob.addVars(pv_buses, vtype=GRB.CONTINUOUS, name="PV_Capacity")  # kW
y_rob              = model_rob.addVars(pv_buses, vtype=GRB.BINARY,    name="PV_Install")
P_grid_rob         = model_rob.addVars(hours, scenarios, vtype=GRB.CONTINUOUS, name="Grid_Import")  # MW
V2_rob             = model_rob.addVars(all_buses, hours, scenarios, lb=V2_min, ub=V2_max, vtype=GRB.CONTINUOUS, name="V2")
P_line_rob         = model_rob.addVars(len(lines), hours, scenarios, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="P_line_rob")
R_res              = model_rob.addVars(hours, vtype=GRB.CONTINUOUS, lb=0.0, name="Reserve")
P_pv_rob           = model_rob.addVars(pv_buses, hours, scenarios, vtype=GRB.CONTINUOUS, name="PV_Output")

#   P_line_rob[e,h,s], V2_rob[i,h,s], (dan R_res[h] khusus robust)


# 2. Fungsi objektif
model_rob.setObjective(quicksum(x_rob[i] for i in pv_buses) +
                   c_res * quicksum(R_res[h] for h in hours),
                   GRB.MINIMIZE)


# 3. Kendala PV (C1)
for i in pv_buses:
    for h in hours:
        for s in scenarios:
            profile = df_pv.query(
                "Scenario == @s and Hour == @h and Bus == @i"
            )["PV Output Factor"].values[0]
            model_rob.addConstr(P_pv_rob[i, h, s] == x_rob[i] * profile / 1000.0,  # kW→MW
                            name=f"pv_output_{i}_{h}_{s}")


# 4. Kendala kapasitas & siting PV (C3, C4)
for i in pv_buses:
    model_rob.addConstr(x_rob[i] <= x_max * y_rob[i], name=f"x_le_ymax_{i}")
    model_rob.addConstr(x_rob[i] >= x_min * y_rob[i], name=f"x_ge_ymin_{i}")

model_rob.addConstr(quicksum(y_rob[i] for i in pv_buses) <= n_max, name="max_selected_sites")


# 5. Batas total & per bus (60 MW, 3×L0)  ← tetap
model_rob.addConstr(quicksum(x_rob[i] for i in pv_buses) <= 60000, name="total_pv_cap_le_60MW")
for i in pv_buses:
    max_capacity_i = L_0[i] * 1000 * 3  # 3x beban dasar (MW→kW)
    model_rob.addConstr(x_rob[i] <= max_capacity_i, name=f"cap_per_bus_{i}")

# 6. LinDistFlow-Lite constraints
# (1) Slack voltage reference
for h in hours:
    for s in scenarios:
        model_rob.addConstr(V2_rob[slack_bus, h, s] == 1.0, name=f"slack_V2_{h}_{s}")

# (2) Voltage drop (linear, lossless-ish)
for e, (u, v, R, S_MVA) in enumerate(lines):
    for h in hours:
        for s in scenarios:
            model_rob.addConstr(V2_rob[v, h, s] == V2_rob[u, h, s] - 2.0 * R * P_line_rob[e, h, s],
                            name=f"vdrop_e{e}_h{h}_s{s}")

# (3) Thermal limits (pakai S_MVA dan pf_min → batas P aktif)
for e, (u, v, R, S_MVA) in enumerate(lines):
    P_lim = pf_min * S_MVA  # MW
    for h in hours:
        for s in scenarios:
            model_rob.addConstr(P_line_rob[e, h, s] <=  P_lim, name=f"thermal_up_e{e}_h{h}_s{s}")
            model_rob.addConstr(P_line_rob[e, h, s] >= -P_lim, name=f"thermal_lo_e{e}_h{h}_s{s}")

# (4) Nodal power balance (aktif, lossless)
for i in all_buses:  # termasuk slack_bus
    for h in hours:
        for s in scenarios:
            inflow  = quicksum(P_line_rob[e, h, s] for e in edges_by_to[i])
            outflow = quicksum(P_line_rob[e, h, s] for e in edges_by_from[i])
            load_i  = df_load.query("Scenario == @s and Hour == @h and Bus == @i")["Load (MW)"].sum()
            
            # Grid injection hanya di slack bus
            grid_term = P_grid_rob[h, s] if i == slack_bus else 0.0
            
            # PV hanya di pv_buses
            pv_term = P_pv_rob[i, h, s] if i in pv_buses else 0.0
            
            model_rob.addConstr(
                pv_term + inflow - outflow + grid_term == load_i,
                name=f"nodal_balance_i{i}_h{h}_s{s}"
            )
    
# (5) s tegangan (Vmin–Vmax)
for i in all_buses:
    for h in hours:
        for s in scenarios:
            model_rob.addConstr(V2_rob[i, h, s] >= V2_min, name=f"vmin_i{i}_h{h}_s{s}")
            model_rob.addConstr(V2_rob[i, h, s] <= V2_max, name=f"vmax_i{i}_h{h}_s{s}")


# 7. Kendala robust adequacy (hanya untuk model robust)
for h in hours:
    avg_pv = (1.0 / len(scenarios)) * quicksum(
        P_pv_rob[i, h, s] for i in pv_buses for s in scenarios
    )
    model_rob.addConstr(
        avg_pv + R_res[h] >= mu_load[h] + kappa * sigma_load[h],
        name=f"robust_margin_hour{h}"
    )



# 8. Solve + ambil hasil (x, y, tegangan, loading, adeq, dst.)
model_rob.optimize()
"""
