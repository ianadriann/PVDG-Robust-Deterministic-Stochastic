import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gurobipy import Model, GRB, quicksum
import os
from pathlib import Path

from Deterministic_model import build_deterministic_pv_model
from Stochastic_model import build_stochastic_pv_model
from Robust_model import build_robust_pv_model


# =========================
# Seed untuk reproducibility
# =========================
np.random.seed(42)

# =========================
# Parameter umum & skenario
# =========================
hours = list(range(24))
scenarios = list(range(1, 11))     # 10 skenario Monte Carlo
planning_years = 15
rho = 2.0                          # tingkat konservatif robust (≈95%:1.96; 99%:2.33)
max_sun = 1000

# Sub-skenario load growth
growth_scenarios = {'Low': 0.02, 'Base': 0.03, 'High': 0.05}

# Profil beban harian (dinormalisasi)
load_profile = np.array([
    0.6, 0.5, 0.4, 0.4, 0.5, 0.6, 0.7, 0.8,
    0.9, 1.0, 1.0, 0.95, 0.9, 0.85, 0.9, 0.95,
    1.0, 1.0, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5
])
load_profile = load_profile / load_profile.max()

# ===========================================
# Network data for LinDistFlow-Lite (radial)
# ===========================================
slack_bus = 2
Vmin, Vmax = 0.95, 1.05
Vmin2, Vmax2 = Vmin**2, Vmax**2

R_default = 1e-5   # p.u. linearized resistance (kecil agar drop tidak over-restrictive)
Sseed = 0.0        # placeholder; S_MVA aktual dihitung per growth

# Topologi radial (2..33) dengan cabang
lines_base = [
    (2, 3, R_default, Sseed),
    (3, 4, R_default, Sseed),
    (4, 5, R_default, Sseed),
    (5, 6, R_default, Sseed),
    (6, 7, R_default, Sseed),
    (7, 8, R_default, Sseed),
    (8, 9, R_default, Sseed),
    (9, 10, R_default, Sseed),
    (10, 11, R_default, Sseed),
    (11, 12, R_default, Sseed),
    (12, 13, R_default, Sseed),
    (13, 14, R_default, Sseed),
    (14, 15, R_default, Sseed),
    (15, 16, R_default, Sseed),
    (16, 17, R_default, Sseed),
    (17, 18, R_default, Sseed),

    (2, 19, R_default, Sseed),
    (19, 20, R_default, Sseed),
    (20, 21, R_default, Sseed),
    (21, 22, R_default, Sseed),

    (3, 23, R_default, Sseed),
    (23, 24, R_default, Sseed),
    (24, 25, R_default, Sseed),

    (6, 26, R_default, Sseed),
    (26, 27, R_default, Sseed),
    (27, 28, R_default, Sseed),
    (28, 29, R_default, Sseed),
    (29, 30, R_default, Sseed),
    (30, 31, R_default, Sseed),
    (31, 32, R_default, Sseed),
    (32, 33, R_default, Sseed),
]

# Turunkan set bus dari topology dasar
buses = sorted({u for (u, _, _, _) in lines_base} | {v for (_, v, _, _) in lines_base})
all_buses = buses[:]                              # alias bila dibutuhkan
pv_buses = [b for b in buses if b != slack_bus]   # PV tidak ditempatkan di slack

# Build children adjacency (untuk hitung downstream set)
children = {i: [] for i in buses}
for (u, v, R, S) in lines_base:
    children[u].append(v)

# Precompute downstream node-set untuk tiap edge (sekali saja)
edge_downstream = []  # list of set(node) untuk setiap edge index pada lines_base
for (u, v, R, S) in lines_base:
    stack = [v]
    ds = set()
    while stack:
        w = stack.pop()
        if w in ds:
            continue
        ds.add(w)
        stack.extend(children.get(w, []))
    edge_downstream.append(ds)

# =====================
# Data beban baseline
# =====================
# L_0 untuk semua bus (MW)
L_0 = {
    2: 8.5, 3: 12.0, 4: 7.0, 5: 9.5, 6: 10.0, 7: 6.5,
    8: 11.0, 9: 5.0, 10: 7.5, 11: 8.0, 12: 9.0,
    13: 6.5, 14: 7.5, 15: 10.5, 16: 8.0, 17: 11.0,
    18: 5.5, 19: 12.5, 20: 9.5, 21: 7.0, 22: 11.5,
    23: 10.5, 24: 11.5, 25: 6.5, 26: 5.0, 27: 8.0, 28: 9.5,
    29: 7.5, 30: 8.5, 31: 7.0, 32: 9.5, 33: 11.0,
}
# Pastikan semua bus punya entri
for b in buses:
    L_0.setdefault(b, 0.0)

# ==========================
# Simulasi Output PV (profil)
# ==========================
k = 2.0           # Weibull shape
lam = 0.8         # Weibull scale
efficiency = 0.18
area_per_kw = 5   # m^2/kW

pv_data = []
for s in scenarios:
    for h in hours:
        for i in pv_buses:
            if 5 <= h <= 17:
                irr = np.random.weibull(k) * lam * max_sun
                irr = min(max(irr, 300), max_sun)
                pv_factor = irr * area_per_kw * efficiency / 1000  # kW output per kW capacity
            else:
                irr = 0
                pv_factor = 0
            pv_data.append([s, h, i, irr, round(pv_factor, 4)])

df_pv = pd.DataFrame(pv_data, columns=['Scenario', 'Hour', 'Bus', 'Irradiance (W/m²)', 'PV Output Factor'])

# ===========================
# Parameter /Optimasi
# ===========================

x_min   = 300                 # kW
x_max   = 40000 #20000               # kW
c_res   = 0.1               # contoh nilai kecil, silakan disesuaikan
n_max   = 5                   # Batas Jumlah PV
V2_min  = 0.95**2            # Batas Minimal Tegangan
V2_max  = 1.05**2            # Batas maksimal Tegangan
kappa   = 1.645
# Bobot objektif teknis
alpha_pv  = 0.005 # bobot kapasitas PV (per kW)
beta_grid = 1.0    # bobot impor grid (per MW rata-rata)



# === Penampung hasil lintas growth (dipakai untuk gambar & tabel) ===
demand_stats_by_growth = {}   # simpan μ_h & σ_h per growth

# ==============================================
# Opsi B + S_MVA: headroom & batas fisik rating
# ==============================================
HEADROOM = 1.10            # 10% di atas robust-peak
pf_min = 0.90              # faktor daya minimum
Smin_phys_MVA = 20.0       # MVA batas fisik minimum (lateral kecil)
Smax_phys_MVA = 600.0      # MVA batas fisik maksimum (trunk besar)

# =========================================================
# RATING LINE / TRAFO EKSISTING BERDASARKAN SKENARIO "High"
# (designed for worst-case growth, lalu DIPAKAI TETAP untuk
#  semua growth: Low, Base, High → no reinforcement)
# =========================================================

g_design = growth_scenarios['High']  # skenario growth paling berat

# Simulasi beban untuk skenario desain "High"
np.random.seed(42)  # konsisten dengan simulasi lain
load_data_high = []
for s in scenarios:
    for h in hours:
        for i in all_buses:
            mean = L_0[i] * (1 + g_design) ** planning_years * load_profile[h]
            std_dev = 0.05 * mean
            value = np.random.normal(mean, std_dev)
            load_data_high.append([s, h, i, max(value, 0.0)])

df_load_high = pd.DataFrame(load_data_high,
                            columns=['Scenario', 'Hour', 'Bus', 'Load (MW)'])

# μ & σ per-bus per-jam untuk skenario "High"
df_bus_stats = df_load_high.groupby(['Hour', 'Bus'])['Load (MW)'] \
                           .agg(['mean', 'std']).reset_index()
df_bus_stats['std'] = df_bus_stats['std'].fillna(0.0)

# Robust load = μ + ρσ
robust_load_high = {
    (int(row['Hour']), int(row['Bus'])): row['mean'] + rho * row['std']
    for _, row in df_bus_stats.iterrows()
}

# Hitung S_edge_MVA dari robust downstream peak + headroom
S_edge_MVA = []
for e, (u, v, R, _) in enumerate(lines_base):
    ds_nodes = edge_downstream[e]
    ds_hour_peaks = []
    for h in hours:
        total_ds = sum(robust_load_high.get((h, b), 0.0) for b in ds_nodes)  # MW
        ds_hour_peaks.append(total_ds)

    robust_peak_MW = HEADROOM * (max(ds_hour_peaks) if ds_hour_peaks else 0.0)  # MW
    S_req_MVA = robust_peak_MW / pf_min   # konversi ke MVA
    S_edge_MVA.append(max(Smin_phys_MVA, min(S_req_MVA, Smax_phys_MVA)))

# Bentuk lines eksisting (R, S_MVA tetap untuk SEMUA growth)
lines = []
for e, (u, v, R, _) in enumerate(lines_base):
    lines.append((u, v, R, S_edge_MVA[e]))

# Index bantu edges_by_from / edges_by_to (topologi sama)
edges_by_from = {i: [] for i in all_buses}
edges_by_to   = {i: [] for i in all_buses}
for e, (u, v, R, S_MVA) in enumerate(lines):
    edges_by_from[u].append(e)
    edges_by_to[v].append(e)




if __name__ == "__main__":
    # Penampung hasil untuk SEMUA growth & model
    summary_rows = []      # untuk tabel ringkasan utama
    siting_by_growth = {}  # simpan df siting per growth & model
    
    for name, g in growth_scenarios.items():
        # ----------------------------
        # Simulasi beban (growth ini)
        # ----------------------------
        np.random.seed(42)  # konsisten antar growth
        load_data = []
        for s in scenarios:
            for h in hours:
                for i in all_buses:
                    mean = L_0[i] * (1 + g) ** planning_years * load_profile[h]
                    std_dev = 0.05 * mean
                    value = np.random.normal(mean, std_dev)
                    load_data.append([s, h, i, max(value, 0)])
        df_load = pd.DataFrame(load_data, columns=['Scenario', 'Hour', 'Bus', 'Load (MW)'])


        # ----------------
        # μ & σ sistem per jam (untuk robust adequacy sistem)
        # ----------------
        df_total = df_load.groupby(['Scenario', 'Hour'])['Load (MW)'].sum().reset_index()
        df_stats = df_total.groupby('Hour').agg(
            mu_load=('Load (MW)', 'mean'),
            sigma_load=('Load (MW)', 'std')
        ).reset_index()
        df_stats['sigma_load'] = df_stats['sigma_load'].fillna(0.0)
        mu_load = dict(zip(df_stats['Hour'], df_stats['mu_load']))
        sigma_load = dict(zip(df_stats['Hour'], df_stats['sigma_load']))

        demand_stats_by_growth[name] = df_stats[['Hour', 'mu_load', 'sigma_load']].copy()


        # ---------------
        # Model Gurobi (3 model)
        # ---------------

        print(f"[{name}] Menjalankan model deterministic...")
        model_det, vars_det = build_deterministic_pv_model(
                                    name, pv_buses, all_buses, hours, lines, df_pv, df_load,
                                    L_0, edges_by_to, edges_by_from, slack_bus, x_max=x_max, x_min=x_min,
                                    n_max=n_max, V2_min=V2_min, V2_max=V2_max, pf_min=pf_min, alpha_pv=alpha_pv, beta_grid=beta_grid,
                                    total_pv_cap_max=160000,  # batas total kapasitas (kW), default 60 MW
                                    solve=True               # kalau True: langsung optimize
                                )

        print(f"[{name}] Menjalankan model stochastic...")
        model_stoc, vars_stoc = build_stochastic_pv_model(
                                    name, pv_buses, all_buses, hours, scenarios, lines, df_pv, df_load,
                                    L_0, edges_by_to, edges_by_from, slack_bus, x_max=x_max, x_min=x_min, n_max=n_max,
                                    V2_min=V2_min, V2_max=V2_max, pf_min=pf_min, alpha_pv=alpha_pv, beta_grid=beta_grid,
                                    total_pv_cap_max=160000,  # batas total kapasitas (kW), default 60 MW
                                    solve=True               # kalau True: langsung optimize
                                )
        
        print(f"[{name}] Menjalankan model robust...")
        model_rob, vars_rob = build_robust_pv_model(
                                    name, pv_buses, all_buses, hours, scenarios, lines, df_pv, df_load, L_0, edges_by_to,
                                    edges_by_from, slack_bus, x_max=x_max, x_min=x_min, n_max=n_max, V2_min=V2_min, V2_max=V2_max,
                                    pf_min=pf_min, mu_load=mu_load, sigma_load=sigma_load, kappa=kappa, c_res=c_res, alpha_pv=alpha_pv, beta_grid=beta_grid,
                                    total_pv_cap_max=160000,  # batas total kapasitas (kW), default 60 MW
                                    solve=True               # kalau True: langsung optimize
                                )
        
        # =======================
        # Ringkasan MODEL DETERMINISTIC
        # =======================
        x_det      = vars_det["x_det"]
        y_det      = vars_det["y_det"]
        V2_det     = vars_det["V2_det"]
        P_line_det = vars_det["P_line_det"]

        if model_det.Status == GRB.OPTIMAL:
            # Kapasitas & lokasi PV
            det_results = []
            for i in pv_buses:
                if y_det[i].X > 0.5:
                    det_results.append([i, x_det[i].X / 1000.0])  # kW→MW
            df_det_siting = pd.DataFrame(det_results, columns=["Bus", "PV Capacity (MW)"])

            # Total PV
            total_pv_det_mw = sum(x_det[i].X for i in pv_buses) / 1000.0
            num_sites_det   = sum(1 for i in pv_buses if y_det[i].X > 0.5)

            # Metrik jaringan sederhana
            vmin_det = min((V2_det[i, h].X)**0.5 for i in all_buses for h in hours)
            vmax_det = max((V2_det[i, h].X)**0.5 for i in all_buses for h in hours)

            max_loading_det = 0.0
            for e, (u, v, R, S_MVA) in enumerate(lines):
                P_lim = pf_min * S_MVA
                if P_lim <= 0:
                    continue
                for h in hours:
                    ratio = abs(P_line_det[e, h].X) / P_lim
                    max_loading_det = max(max_loading_det, ratio)
            max_loading_det_pct = 100.0 * max_loading_det

            # simpan ke ringkasan
            summary_rows.append({
                "Growth": name,
                "Model": "DET",
                "Total PV (MW)": total_pv_det_mw,
                "#Sites": num_sites_det,
                "Vmin (p.u.)": vmin_det,
                "Vmax (p.u.)": vmax_det,
                "Max line loading (%)": max_loading_det_pct,
            })

            # kalau mau simpan siting detail:
            siting_by_growth[(name, "DET")] = df_det_siting.copy()
        else:
            print(f"[{name}] Model deterministic tidak optimal, status =", model_det.Status)

        # =======================
        # Ringkasan MODEL STOCHASTIC
        # =======================
        x_stoc      = vars_stoc["x_stoc"]
        y_stoc      = vars_stoc["y_stoc"]
        V2_stoc     = vars_stoc["V2_stoc"]
        P_line_stoc = vars_stoc["P_line_stoc"]
        P_pv_stoc   = vars_stoc["P_pv_stoc"]
        P_grid_stoc = vars_stoc["P_grid_stoc"]

        if model_stoc.Status == GRB.OPTIMAL:
            stoc_results = []
            for i in pv_buses:
                if y_stoc[i].X > 0.5:
                    stoc_results.append([i, x_stoc[i].X / 1000.0])
            df_stoc_siting = pd.DataFrame(stoc_results, columns=["Bus", "PV Capacity (MW)"])

            total_pv_stoc_mw = sum(x_stoc[i].X for i in pv_buses) / 1000.0
            num_sites_stoc   = sum(1 for i in pv_buses if y_stoc[i].X > 0.5)

            # Voltage & loading: worst case across semua s
            vmin_stoc = float("inf")
            vmax_stoc = 0.0
            for i in all_buses:
                for h in hours:
                    for s in scenarios:
                        v = (V2_stoc[i, h, s].X)**0.5
                        vmin_stoc = min(vmin_stoc, v)
                        vmax_stoc = max(vmax_stoc, v)

            max_loading_stoc = 0.0
            for e, (u, v, R, S_MVA) in enumerate(lines):
                P_lim = pf_min * S_MVA
                if P_lim <= 0:
                    continue
                for h in hours:
                    for s in scenarios:
                        ratio = abs(P_line_stoc[e, h, s].X) / P_lim
                        max_loading_stoc = max(max_loading_stoc, ratio)
            max_loading_stoc_pct = 100.0 * max_loading_stoc

            # OPTIONAL: share energi PV vs grid (sehari representatif)
            pv_energy_day    = 0.0
            grid_energy_day  = 0.0
            for s in scenarios:
                for h in hours:
                    pv_tot   = sum(P_pv_stoc[i, h, s].X for i in pv_buses)
                    grid_tot = P_grid_stoc[h, s].X
                    pv_energy_day   += pv_tot
                    grid_energy_day += grid_tot
            pv_energy_day   /= len(scenarios)  # MWh/hari kalau ∆t = 1 jam
            grid_energy_day /= len(scenarios)

            summary_rows.append({
                "Growth": name,
                "Model": "STOCH",
                "Total PV (MW)": total_pv_stoc_mw,
                "#Sites": num_sites_stoc,
                "Vmin (p.u.)": vmin_stoc,
                "Vmax (p.u.)": vmax_stoc,
                "Max line loading (%)": max_loading_stoc_pct,
                "PV energy (MWh/day)": pv_energy_day,
                "Grid energy (MWh/day)": grid_energy_day,
            })

            siting_by_growth[(name, "STOCH")] = df_stoc_siting.copy()
        else:
            print(f"[{name}] Model stochastic tidak optimal, status =", model_stoc.Status)
        
        # =======================
        # Ringkasan MODEL ROBUST
        # =======================
        x_rob      = vars_rob["x_rob"]
        y_rob      = vars_rob["y_rob"]
        V2_rob     = vars_rob["V2_rob"]
        P_line_rob = vars_rob["P_line_rob"]
        P_pv_rob   = vars_rob["P_pv_rob"]
        P_grid_rob = vars_rob["P_grid_rob"]
        R_res      = vars_rob["R_res"]

        if model_rob.Status == GRB.OPTIMAL:
            rob_results = []
            for i in pv_buses:
                if y_rob[i].X > 0.5:
                    rob_results.append([i, x_rob[i].X / 1000.0])
            df_rob_siting = pd.DataFrame(rob_results, columns=["Bus", "PV Capacity (MW)"])

            total_pv_rob_mw = sum(x_rob[i].X for i in pv_buses) / 1000.0
            num_sites_rob   = sum(1 for i in pv_buses if y_rob[i].X > 0.5)

            # Voltage & loading: worst case across semua s
            vmin_rob = float("inf")
            vmax_rob = 0.0
            for i in all_buses:
                for h in hours:
                    for s in scenarios:
                        v = (V2_rob[i, h, s].X)**0.5
                        vmin_rob = min(vmin_rob, v)
                        vmax_rob = max(vmax_rob, v)

            max_loading_rob = 0.0
            for e, (u, v, R, S_MVA) in enumerate(lines):
                P_lim = pf_min * S_MVA
                if P_lim <= 0:
                    continue
                for h in hours:
                    for s in scenarios:
                        ratio = abs(P_line_rob[e, h, s].X) / P_lim
                        max_loading_rob = max(max_loading_rob, ratio)
            max_loading_rob_pct = 100.0 * max_loading_rob

            # Energi PV & grid per "hari representatif"
            pv_energy_day_rob   = 0.0
            grid_energy_day_rob = 0.0
            for s in scenarios:
                for h in hours:
                    pv_tot   = sum(P_pv_rob[i, h, s].X for i in pv_buses)
                    grid_tot = P_grid_rob[h, s].X
                    pv_energy_day_rob   += pv_tot
                    grid_energy_day_rob += grid_tot
            pv_energy_day_rob   /= len(scenarios)
            grid_energy_day_rob /= len(scenarios)

            # Robust reserve rata-rata
            avg_reserve = sum(R_res[h].X for h in hours) / len(hours)

            summary_rows.append({
                "Growth": name,
                "Model": "ROBUST",
                "Total PV (MW)": total_pv_rob_mw,
                "#Sites": num_sites_rob,
                "Vmin (p.u.)": vmin_rob,
                "Vmax (p.u.)": vmax_rob,
                "Max line loading (%)": max_loading_rob_pct,
                "PV energy (MWh/day)": pv_energy_day_rob,
                "Grid energy (MWh/day)": grid_energy_day_rob,
                "Avg reserve (MW)": avg_reserve,
            })

            siting_by_growth[(name, "ROBUST")] = df_rob_siting.copy()
        else:
            print(f"[{name}] Model robust tidak optimal, status =", model_rob.Status)


    print("\n=== Detail lokasi PV per growth & model ===")
    for (growth, model), df_siting in siting_by_growth.items():
        print(f"\n[{growth} - {model}]")
        if df_siting.empty:
            print("  (Tidak ada PV dipasang)")
        else:
            print(df_siting)


df_summary = pd.DataFrame(summary_rows)
print("\n=== Ringkasan semua growth & model ===")
print(df_summary)
df_summary.to_excel("summary_det_stoch_robust.xlsx", index=False)



# =========================================
# 0. Baca / siapkan df_summary
# =========================================
# Kalau df_summary belum ada di memory, bisa load dari Excel:
# df_summary = pd.read_excel("summary_det_stoch_robust.xlsx")

# Pastikan urutan Growth rapi
growth_order = ['Low', 'Base', 'High']
df_summary['Growth'] = pd.Categorical(
    df_summary['Growth'],
    categories=growth_order,
    ordered=True
)
df_summary = df_summary.sort_values(['Growth', 'Model'])

# =========================================
# 1. Max line loading (%) vs Growth & Model
#    (pivot ke bentuk Growth × Model)
# =========================================
pivot_loading = df_summary.pivot(
    index='Growth',
    columns='Model',
    values='Max line loading (%)'
)

# Pastikan hanya pakai model yang memang ada di kolom
models = [m for m in ['DET', 'STOCH', 'ROBUST'] if m in pivot_loading.columns]

# -----------------------------------------
# 1A. BAR PLOT – Maximum line loading
# -----------------------------------------
plt.figure(figsize=(8, 4))
ax = pivot_loading[models].plot(kind='bar', figsize=(8, 4))

ax.set_xlabel("Load growth scenario")
ax.set_ylabel("Max line loading (%)")
ax.set_title("Figure 1. Maximum line loading vs growth and model")
ax.legend(title="Model", loc="upper left")
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# -----------------------------------------
# 1B. LINE PLOT – Maximum line loading
# -----------------------------------------
plt.figure(figsize=(8, 4))
for model in models:
    plt.plot(
        pivot_loading.index,
        pivot_loading[model],
        marker='o',
        label=model
    )

plt.xlabel("Load growth scenario")
plt.ylabel("Max line loading (%)")
plt.title("Figure 2. Maximum line loading vs growth and model")
plt.grid(True, alpha=0.3)
plt.legend(title="Model")
plt.tight_layout()
plt.show()

# =========================================
# 2. Avg reserve (MW) – hanya model ROBUST
# =========================================
df_rob = df_summary[df_summary['Model'] == 'ROBUST'].copy()
df_rob = df_rob.sort_values('Growth')

plt.figure(figsize=(6, 4))
plt.bar(df_rob['Growth'], df_rob['Avg reserve (MW)'])

plt.xlabel("Load growth scenario")
plt.ylabel("Average reserve R_h (MW)")
plt.title("Figure 3. Robust model – average reserve vs growth")
plt.grid(axis='y', alpha=0.3)

# (opsional) tulis nilai di atas masing-masing bar
for x, y in zip(df_rob['Growth'], df_rob['Avg reserve (MW)']):
    plt.text(x, y, f"{y:.1f}",
             ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

