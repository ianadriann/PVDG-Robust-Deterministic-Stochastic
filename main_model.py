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
x_max   = 20000               # kW
c_res   = 0.001               # contoh nilai kecil, silakan disesuaikan
n_max   = 5                   # Batas Jumlah PV
V2_min  = 0.95**2            # Batas Minimal Tegangan
V2_max  = 1.05**2            # Batas maksimal Tegangan
kappa   = 1.645



# Prioritas site (berbasis L0)
L0_vals = [v for v in L_0.values() if v > 0]
L0_max = max(L0_vals) if L0_vals else 1.0

# === Penampung hasil lintas growth (dipakai untuk gambar & tabel) ===
demand_stats_by_growth = {}   # simpan μ_h & σ_h per growth
supply_by_growth = {}         # simpan df_supply (per (s,h)) per growth
siting_by_growth = {}         # simpan df_result kapasitas PV terpasang per growth
adequacy_stats_by_growth = {} # simpan ringkasan adequacy (mean, std, CV) per jam per growth
tight_hours_list = []         # akumulasi jam paling ketat lintas growth (untuk Table V.2)

tight_hours_min_s_list = []   # 3–5 jam dengan min gap terburuk lintas skenario
tight_hours_cv_list = []      # 3–5 jam dengan CV(adequacy) tertinggi

network_metrics_by_growth = {}   # name -> dict berisi vmin, line loading, binding flags, dll.



# ==============================================
# Opsi B + S_MVA: headroom & batas fisik rating
# ==============================================
HEADROOM = 1.10            # 10% di atas robust-peak
pf_min = 0.90              # faktor daya minimum
Smin_phys_MVA = 20.0       # MVA batas fisik minimum (lateral kecil)
Smax_phys_MVA = 600.0      # MVA batas fisik maksimum (trunk besar)

# ======================
# Loop sub-skenario growth
# ======================
summary_rows = []
avg_by_growth = {}   # untuk overlay lintas growth
comp_rows = []  # for Table V.7 — model size & solve stats

if __name__ == "__main__":
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

        # -------------------------------
        # μ & σ per-bus per-jam → robust
        # -------------------------------
        df_bus_stats = df_load.groupby(['Hour', 'Bus'])['Load (MW)'].agg(['mean', 'std']).reset_index()
        df_bus_stats['std'] = df_bus_stats['std'].fillna(0.0)
        robust_load = {(int(row['Hour']), int(row['Bus'])): row['mean'] + rho * row['std']
                    for _, row in df_bus_stats.iterrows()}

        # --------------------------------------------------------
        # Hitung S_MVA per edge dari robust downstream peak + headroom
        # --------------------------------------------------------
        S_edge_MVA = []
        for e, (u, v, R, _) in enumerate(lines_base):
            ds_nodes = edge_downstream[e]
            # akumulasi robust downstream per jam → ambil puncak
            ds_hour_peaks = []
            for h in hours:
                total_ds = sum(robust_load.get((h, b), 0.0) for b in ds_nodes)  # MW
                ds_hour_peaks.append(total_ds)
            robust_peak_MW = HEADROOM * (max(ds_hour_peaks) if ds_hour_peaks else 0.0)  # MW with headroom
            # konversi kebutuhan P ke rating apparent (MVA) via pf_min
            S_req_MVA = robust_peak_MW / pf_min
            # jepit ke rentang fisik
            S_edge_MVA.append(max(Smin_phys_MVA, min(S_req_MVA, Smax_phys_MVA)))

        # bentuk ulang lines efektif untuk growth ini (pakai S_MVA)
        lines = []
        for (e, (u, v, R, _)) in enumerate(lines_base):
            lines.append((u, v, R, S_edge_MVA[e]))  # terakhir sekarang S_MVA


        # rebuild index bantu (edges_by_from/to) sesuai lines (topologi sama)
        edges_by_from = {i: [] for i in all_buses}  # bukan pv_buses
        edges_by_to   = {i: [] for i in all_buses}
        for e, (u, v, R, S_MVA) in enumerate(lines):
            edges_by_from[u].append(e)
            edges_by_to[v].append(e)


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
                                    n_max=n_max, V2_min=V2_min, V2_max=V2_max, pf_min=pf_min,
                                    total_pv_cap_max=60000,  # batas total kapasitas (kW), default 60 MW
                                    solve=True               # kalau True: langsung optimize
                                )

        print(f"[{name}] Menjalankan model stochastic...")
        model_stoc, vars_stoc = build_stochastic_pv_model(
                                    name, pv_buses, all_buses, hours, scenarios, lines, df_pv, df_load,
                                    L_0, edges_by_to, edges_by_from, slack_bus, x_max=x_max, x_min=x_min, n_max=n_max,
                                    V2_min=V2_min, V2_max=V2_max, pf_min=pf_min,
                                    total_pv_cap_max=60000,  # batas total kapasitas (kW), default 60 MW
                                    solve=True               # kalau True: langsung optimize
                                )
        
        print(f"[{name}] Menjalankan model robust...")
        model_rob, vars_rob = build_robust_pv_model(
                                    name, pv_buses, all_buses, hours, scenarios, lines, df_pv, df_load, L_0, edges_by_to,
                                    edges_by_from, slack_bus, x_max=x_max, x_min=x_min, n_max=n_max, V2_min=V2_min, V2_max=V2_max,
                                    pf_min=pf_min, mu_load=mu_load, sigma_load=sigma_load, kappa=kappa, c_res=c_res,
                                    total_pv_cap_max=60000,  # batas total kapasitas (kW), default 60 MW
                                    solve=True               # kalau True: langsung optimize
                                )

print(model_det.Status, model_stoc.Status, model_rob.Status)


