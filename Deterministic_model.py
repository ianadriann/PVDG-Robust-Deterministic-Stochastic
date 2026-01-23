#kode sebelum perubahan (23 Februari 2026)
from gurobipy import Model, GRB, quicksum

def build_deterministic_pv_model(
    name, pv_buses, all_buses, hours, lines, df_pv, df_load,
    L_0, edges_by_to, edges_by_from, slack_bus, x_max, x_min,
    n_max, V2_min, V2_max, pf_min, alpha_pv, beta_grid,
    total_pv_cap_max=60000,  # batas total kapasitas (kW), default 60 MW
    solve=False               # kalau True: langsung optimize
):



    # 0. Buat model
    model_det = Model(f"Deterministic_{name}")
    model_det.setParam("OutputFlag", 0)

    # 1. Variabel keputusan
    x_det      = model_det.addVars(pv_buses, vtype=GRB.CONTINUOUS, name="PV_Capacity")   # kW
    y_det      = model_det.addVars(pv_buses, vtype=GRB.BINARY,    name="PV_Install")
    P_grid_det = model_det.addVars(hours,    vtype=GRB.CONTINUOUS, name="Grid_Import")   # MW
    V2_det     = model_det.addVars(all_buses, hours, lb=V2_min, ub=V2_max,
                                   vtype=GRB.CONTINUOUS, name="V2_det")
    P_line_det = model_det.addVars(len(lines), hours, lb=-GRB.INFINITY,
                                   vtype=GRB.CONTINUOUS, name="P_line")
    P_pv_det   = model_det.addVars(pv_buses, hours, vtype=GRB.CONTINUOUS, name="PV_Output")

    # 2. Fungsi objektif: minimasi total kapasitas PV terpasang
    # 2. Fungsi objektif: kombinasi kapasitas PV + impor grid
    model_det.setObjective(
        alpha_pv * quicksum(x_det[i] for i in pv_buses) +
        beta_grid * quicksum(P_grid_det[h] for h in hours),
        GRB.MINIMIZE
    )


    # 3. Kendala PV (C1): rata-rata PV Output Factor per Hour–Bus
    df_pv_det = df_pv.groupby(['Hour', 'Bus'])['PV Output Factor'].mean().reset_index()
    pv_profile_det = {
        (int(row['Hour']), int(row['Bus'])): row['PV Output Factor']
        for _, row in df_pv_det.iterrows()
    }

    for i in pv_buses:
        for h in hours:
            profile = pv_profile_det.get((h, i), 0.0)
            model_det.addConstr(
                P_pv_det[i, h] == x_det[i] * profile / 1000.0,  # kW → MW
                name=f"pv_output_{i}_{h}"
            )

    # (Cgrid-det) Grid import tidak boleh negatif
    for h in hours:
        model_det.addConstr(
            P_grid_det[h] >= 0.0,
            name=f"grid_import_nonneg_{h}"
        )


    # 4. Kendala kapasitas & siting PV (C3, C4)
    for i in pv_buses:
        model_det.addConstr(x_det[i] <= x_max * y_det[i], name=f"x_le_ymax_{i}")
        model_det.addConstr(x_det[i] >= x_min * y_det[i], name=f"x_ge_ymin_{i}")

    model_det.addConstr(quicksum(y_det[i] for i in pv_buses) <= n_max,
                        name="max_selected_sites")

    # Batas total & per bus (kW)
    model_det.addConstr(quicksum(x_det[i] for i in pv_buses) <= total_pv_cap_max,
                        name="total_pv_cap_limit")

    for i in pv_buses:
        max_capacity_i = L_0[i] * 1000 * 3  # 3x beban dasar (MW→kW)
        model_det.addConstr(x_det[i] <= max_capacity_i, name=f"cap_per_bus_{i}")

    # 5. LinDistFlow-Lite constraints
    # Beban deterministic: rata-rata per Hour–Bus dari semua skenario
    df_load_det = df_load.groupby(['Hour', 'Bus'])['Load (MW)'].mean().reset_index()
    load_det = {
        (int(r['Hour']), int(r['Bus'])): r['Load (MW)']
        for _, r in df_load_det.iterrows()
    }

    # (1) Slack voltage reference
    for h in hours:
        model_det.addConstr(V2_det[slack_bus, h] == 1.0, name=f"slack_V2_{h}")

    # (2) Voltage drop
    for e, (u, v, R, S_MVA) in enumerate(lines):
        for h in hours:
            model_det.addConstr(
                V2_det[v, h] == V2_det[u, h] - 2.0 * R * P_line_det[e, h],
                name=f"vdrop_e{e}_h{h}"
            )

    # (3) Thermal limits
    for e, (u, v, R, S_MVA) in enumerate(lines):
        P_lim = pf_min * S_MVA  # MW
        for h in hours:
            model_det.addConstr(
                P_line_det[e, h] <= P_lim,
                name=f"thermal_up_e{e}_h{h}"
            )
            model_det.addConstr(
                P_line_det[e, h] >= -P_lim,
                name=f"thermal_lo_e{e}_h{h}"
            )

    # (4) Nodal power balance
    for i in all_buses:  # termasuk slack_bus
        for h in hours:
            inflow  = quicksum(P_line_det[e, h] for e in edges_by_to[i])
            outflow = quicksum(P_line_det[e, h] for e in edges_by_from[i])
            load_i  = load_det.get((h, i), 0.0)

            grid_term = P_grid_det[h] if i == slack_bus else 0.0
            pv_term   = P_pv_det[i, h] if i in pv_buses else 0.0

            model_det.addConstr(
                pv_term + inflow - outflow + grid_term == load_i,
                name=f"nodal_balance_i{i}_h{h}"
            )

    # (5) Batas tegangan (Vmin–Vmax) - sebenarnya sudah di definisi var, tapi eksplisit juga tidak apa
    for i in all_buses:
        for h in hours:
            model_det.addConstr(V2_det[i, h] >= V2_min, name=f"vmin_i{i}_h{h}")
            model_det.addConstr(V2_det[i, h] <= V2_max, name=f"vmax_i{i}_h{h}")

    # 6. Solve (opsional)
    if solve:
        model_det.optimize()

    # Kembalikan model + semua variabel biar mudah diakses
    vars_det = {
        "x_det": x_det,
        "y_det": y_det,
        "P_grid_det": P_grid_det,
        "V2_det": V2_det,
        "P_line_det": P_line_det,
        "P_pv_det": P_pv_det,
    }

    return model_det, vars_det

"""
# 0. Buat model
model_det = Model(f"Deterministic_{name}")  # name = 'Low', 'Base', 'High'
model_det.setParam("OutputFlag", 0)

# Variabel keputusan
x_det              = model_det.addVars(pv_buses, vtype=GRB.CONTINUOUS, name="PV_Capacity")  # kW
y_det              = model_det.addVars(pv_buses, vtype=GRB.BINARY,    name="PV_Install")
P_grid_det         = model_det.addVars(hours, vtype=GRB.CONTINUOUS, name="Grid_Import")  # MW
V2_det             = model_det.addVars(all_buses, hours, lb=V2_min, ub=V2_max, vtype=GRB.CONTINUOUS, name="V2_det")
P_line_det         = model_det.addVars(len(lines), hours, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="P_line")
P_pv_det           = model_det.addVars(pv_buses, hours, vtype=GRB.CONTINUOUS, name="PV_Output")


# 2. Fungsi objektif
model_det.setObjective(quicksum(x_det[i] for i in pv_buses),GRB.MINIMIZE)

# 3. Kendala PV (C1)
# Rata-rata PV Output Factor per Hour–Bus (agregat semua skenario)
df_pv_det = df_pv.groupby(['Hour', 'Bus'])['PV Output Factor'].mean().reset_index()
pv_profile_det = {
    (int(row['Hour']), int(row['Bus'])): row['PV Output Factor']
    for _, row in df_pv_det.iterrows()
}

for i in pv_buses:
    for h in hours:
        profile = pv_profile_det.get((h, i), 0.0)
        model_det.addConstr(
            P_pv_det[i, h] == x_det[i] * profile / 1000.0,  # kW → MW
            name=f"pv_output_{i}_{h}"
        )



# 4. Kendala kapasitas & siting PV (C3, C4)
for i in pv_buses:
    model_det.addConstr(x_det[i] <= x_max * y_det[i], name=f"x_le_ymax_{i}")
    model_det.addConstr(x_det[i] >= x_min * y_det[i], name=f"x_ge_ymin_{i}")

model_det.addConstr(quicksum(y_det[i] for i in pv_buses) <= n_max, name="max_selected_sites")


# Batas total & per bus (kW)
model_det.addConstr(quicksum(x_det[i] for i in pv_buses) <= 60000, name="total_pv_cap_le_60MW")
for i in pv_buses:
    max_capacity_i = L_0[i] * 1000 * 3  # 3x beban dasar (MW→kW)
    model_det.addConstr(x_det[i] <= max_capacity_i, name=f"cap_per_bus_{i}")

# 6. LinDistFlow-Lite constraints
# ------------------------
# LinDistFlow-Lite constraints
# ------------------------
# Beban deterministic: rata-rata per Hour–Bus dari semua skenario
df_load_det = df_load.groupby(['Hour', 'Bus'])['Load (MW)'].mean().reset_index()
load_det = {(int(r['Hour']), int(r['Bus'])): r['Load (MW)'] for _, r in df_load_det.iterrows()}


# (1) Slack voltage reference
for h in hours:
    model_det.addConstr(V2_det[slack_bus, h] == 1.0, name=f"slack_V2_{h}")

# (2) Voltage drop (linear, lossless-ish)
for e, (u, v, R, S_MVA) in enumerate(lines):
    for h in hours:
        model_det.addConstr(V2_det[v, h] == V2_det[u, h] - 2.0 * R * P_line_det[e, h],
                        name=f"vdrop_e{e}_h{h}")

# (3) Thermal limits (pakai S_MVA dan pf_min → batas P aktif)
for e, (u, v, R, S_MVA) in enumerate(lines):
    P_lim = pf_min * S_MVA  # MW
    for h in hours:
        model_det.addConstr(
            P_line_det[e, h] <= P_lim,
            name=f"thermal_up_e{e}_h{h}"
        )
        model_det.addConstr(
            P_line_det[e, h] >= -P_lim,
            name=f"thermal_lo_e{e}_h{h}"
        )


# (4) Nodal power balance (aktif, lossless)
for i in all_buses:  # termasuk slack_bus
    for h in hours:
        inflow  = quicksum(P_line_det[e, h] for e in edges_by_to[i])
        outflow = quicksum(P_line_det[e, h] for e in edges_by_from[i])
        load_i  = load_det.get((h, i), 0.0)  # rata-rata beban jam h di bus i

        grid_term = P_grid_det[h] if i == slack_bus else 0.0
        pv_term   = P_pv_det[i, h] if i in pv_buses else 0.0

        model_det.addConstr(
            pv_term + inflow - outflow + grid_term == load_i,
            name=f"nodal_balance_i{i}_h{h}"
        )

    
# (5) s tegangan (Vmin–Vmax)
for i in all_buses:
    for h in hours:
        model_det.addConstr(V2_det[i, h] >= V2_min, name=f"vmin_i{i}_h{h}")
        model_det.addConstr(V2_det[i, h] <= V2_max, name=f"vmax_i{i}_h{h}")




# 8. Solve + ambil hasil (x, y, tegangan, loading, adeq, dst.)
model_det.optimize()"""