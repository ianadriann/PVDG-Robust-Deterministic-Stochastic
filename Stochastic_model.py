#kode sebelum perubahan (23 Februari 2026)
from gurobipy import Model, GRB, quicksum
import numpy as np

def build_stochastic_pv_model(
    name, pv_buses, all_buses, hours, scenarios, lines, df_pv, df_load,
    L_0, edges_by_to, edges_by_from, slack_bus, x_max, x_min, n_max,
    V2_min, V2_max, pf_min, tanphi, alpha_pv, beta_grid,
    total_pv_cap_max=60000,  # batas total kapasitas (kW), default 60 MW
    solve=False               # kalau True: langsung optimize
):

    # 0. Buat model
    model_stoc = Model(f"Stochastic_{name}")  # name = 'Low', 'Base', 'High'
    model_stoc.setParam("OutputFlag", 0)

    # 1. Variabel keputusan
    x_stoc              = model_stoc.addVars(pv_buses, vtype=GRB.CONTINUOUS, name="PV_Capacity")  # kW
    y_stoc              = model_stoc.addVars(pv_buses, vtype=GRB.BINARY,    name="PV_Install")
    P_grid_stoc         = model_stoc.addVars(hours, scenarios, vtype=GRB.CONTINUOUS, name="Grid_Import")  # MW
    V2_stoc             = model_stoc.addVars(all_buses, hours, scenarios, lb=V2_min, ub=V2_max, vtype=GRB.CONTINUOUS, name="V2")
    P_line_stoc         = model_stoc.addVars(len(lines), hours, scenarios, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="P_line")
    P_pv_stoc = model_stoc.addVars(pv_buses, hours, scenarios, lb=0.0, vtype=GRB.CONTINUOUS, name="PV_Output")


    Q_grid_stoc = model_stoc.addVars(hours, scenarios, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="Q_grid")
    Q_line_stoc = model_stoc.addVars(len(lines), hours, scenarios, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="Q_line")
    Q_pv_stoc   = model_stoc.addVars(pv_buses, hours, scenarios, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="Q_pv")



    # --- Inverter reactive capability (PF constraint) ---
    pf_inv_min = 0.90
    qmax_ratio = np.tan(np.arccos(pf_inv_min))  # ~0.484

    # 2. Fungsi objektif: kapasitas PV + rata-rata impor grid
    model_stoc.setObjective(
        alpha_pv * quicksum(x_stoc[i] for i in pv_buses) +
        beta_grid * (1.0 / len(scenarios)) * quicksum(
            P_grid_stoc[h, s] for h in hours for s in scenarios
        ),
        GRB.MINIMIZE
    )

    for i in pv_buses:
        for h in hours:
            for s in scenarios:
                model_stoc.addConstr(Q_pv_stoc[i, h, s] <=  qmax_ratio * x_stoc[i] / 1000.0,
                                    name=f"qcap_up_{i}_{h}_{s}")
                model_stoc.addConstr(Q_pv_stoc[i, h, s] >= -qmax_ratio * x_stoc[i] / 1000.0,
                                    name=f"qcap_lo_{i}_{h}_{s}")


    # 3. Kendala PV (C1)
    for i in pv_buses:
        for h in hours:
            for s in scenarios:
                profile = df_pv.query(
                    "Scenario == @s and Hour == @h and Bus == @i"
                )["PV Output Factor"].values[0]
                model_stoc.addConstr(P_pv_stoc[i, h, s] <= x_stoc[i] * profile / 1000.0,  # kW→MW
                                name=f"pv_output_{i}_{h}_{s}")

    # (Cgrid-stoch) Grid import tidak boleh negatif
    for h in hours:
        for s in scenarios:
            model_stoc.addConstr(
                P_grid_stoc[h, s] >= 0.0,
                name=f"grid_import_nonneg_h{h}_s{s}"
            )


    # 4. Kendala kapasitas & siting PV (C3, C4)
    for i in pv_buses:
        model_stoc.addConstr(x_stoc[i] <= x_max * y_stoc[i], name=f"x_le_ymax_{i}")
        model_stoc.addConstr(x_stoc[i] >= x_min * y_stoc[i], name=f"x_ge_ymin_{i}")

    model_stoc.addConstr(quicksum(y_stoc[i] for i in pv_buses) <= n_max, name="max_selected_sites")

    # 5. Batas total & per bus (60 MW, 3×L0)  ← tetap
    model_stoc.addConstr(quicksum(x_stoc[i] for i in pv_buses) <= total_pv_cap_max, name="total_pv_cap_le_60MW")
    for i in pv_buses:
        max_capacity_i = L_0[i] * 1000 * 3  # 3x beban dasar (MW→kW)
        model_stoc.addConstr(x_stoc[i] <= max_capacity_i, name=f"cap_per_bus_{i}")


    # 6. LinDistFlow-Lite constraints
    #   (1) Slack voltage reference
    for h in hours:
        for s in scenarios:
            model_stoc.addConstr(V2_stoc[slack_bus, h, s] == 1.05**2, name=f"slack_V2_{h}_{s}")

    #   (2) Voltage drop
    for e, (u, v, R, X, S_MVA) in enumerate(lines):
        for h in hours:
            for s in scenarios:
                model_stoc.addConstr(V2_stoc[v, h, s] == V2_stoc[u, h, s] - 2.0 * (R * P_line_stoc[e, h, s] + X*Q_line_stoc[e, h, s]),
                                name=f"vdrop_e{e}_h{h}_s{s}")
                    
    #   (3) Thermal limits
    for e, (u, v, R, X, S_MVA) in enumerate(lines):
        P_lim = pf_min * S_MVA  # MW
        Q_lim = ((1.0 - pf_min**2) ** 0.5) * S_MVA  # MVAr
        for h in hours:
            for s in scenarios:
                model_stoc.addConstr(P_line_stoc[e, h, s] <=  P_lim, name=f"thermal_up_e{e}_h{h}_s{s}")
                model_stoc.addConstr(P_line_stoc[e, h, s] >= -P_lim, name=f"thermal_lo_e{e}_h{h}_s{s}")

                # Q limits
                model_stoc.addConstr(Q_line_stoc[e, h, s] <=  Q_lim, name=f"thermalQ_up_e{e}_h{h}_s{s}")
                model_stoc.addConstr(Q_line_stoc[e, h, s] >= -Q_lim, name=f"thermalQ_lo_e{e}_h{h}_s{s}")

    #   (4) Nodal power balance (aktif, lossless)
    for i in all_buses:  # termasuk slack_bus
        for h in hours:
            for s in scenarios:
                inflow  = quicksum(P_line_stoc[e, h, s] for e in edges_by_to[i])
                outflow = quicksum(P_line_stoc[e, h, s] for e in edges_by_from[i])
                load_i  = df_load.query("Scenario == @s and Hour == @h and Bus == @i")["Load (MW)"].sum()

                q_load_i = load_i * tanphi
                
                # Grid injection hanya di slack bus
                grid_term = P_grid_stoc[h, s] if i == slack_bus else 0.0
                
                # PV hanya di pv_buses
                pv_term = P_pv_stoc[i, h, s] if i in pv_buses else 0.0
                
                model_stoc.addConstr(
                    pv_term + inflow - outflow + grid_term == load_i,
                    name=f"nodal_balance_i{i}_h{h}_s{s}"
                )

                qin  = quicksum(Q_line_stoc[e, h, s] for e in edges_by_to[i])
                qout = quicksum(Q_line_stoc[e, h, s] for e in edges_by_from[i])
                qgrid_term = Q_grid_stoc[h, s] if i == slack_bus else 0.0
                qpv_term   = Q_pv_stoc[i, h, s] if i in pv_buses else 0.0

                model_stoc.addConstr(
                    qpv_term + qin - qout + qgrid_term == q_load_i,
                    name=f"q_balance_i{i}_h{h}_s{s}"
                )



    #   (5) Batas tegangan (Vmin–Vmax)
    for i in all_buses:
        for h in hours:
            for s in scenarios:
                model_stoc.addConstr(V2_stoc[i, h, s] >= V2_min, name=f"vmin_i{i}_h{h}_s{s}")
                model_stoc.addConstr(V2_stoc[i, h, s] <= V2_max, name=f"vmax_i{i}_h{h}_s{s}")




    # 6. Solve + ambil hasil (x, y, tegangan, loading, adeq, dst.)
    if solve:
        model_stoc.optimize()
        if model_stoc.Status == GRB.INFEASIBLE:
            model_stoc.computeIIS()
            model_stoc.write(f"iis_{model_stoc.ModelName}.ilp")
            print("IIS written:", f"iis_{model_stoc.ModelName}.ilp")

    # Kembalikan model + semua variabel biar mudah diakses
    vars_stoc = {
        "x_stoc": x_stoc,
        "y_stoc": y_stoc,
        "P_grid_stoc": P_grid_stoc,
        "V2_stoc": V2_stoc,
        "P_line_stoc": P_line_stoc,
        "P_pv_stoc": P_pv_stoc,
        "Q_grid_stoc": Q_grid_stoc,
        "Q_line_stoc": Q_line_stoc,
        "Q_pv_stoc": Q_pv_stoc,

    }

    return model_stoc, vars_stoc