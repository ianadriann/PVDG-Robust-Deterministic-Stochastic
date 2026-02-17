
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gurobipy import Model, GRB, quicksum
import os
from pathlib import Path

from Deterministic_model import build_deterministic_pv_model
from Stochastic_model import build_stochastic_pv_model
from Robust_model import build_robust_pv_model


# =========================================================
# EXPORT FIGURE/TABLE FOR PAPER (Daily load profile & Growth table)
# Letakkan tepat setelah import, sebelum np.random.seed(42)
# =========================================================

def export_daily_load_profile(lambda_h, outdir="paper_outputs", fname="Fig2_load_profile_lambda.png"):
    """
    Figure: Daily load profile λ(h) untuk 24 jam.
    lambda_h: array-like panjang 24 (sudah dinormalisasi p.u.)
    """
    os.makedirs(outdir, exist_ok=True)

    lam = np.asarray(lambda_h, dtype=float).flatten()
    if lam.size != 24:
        raise ValueError(f"load_profile harus panjang 24, tetapi sekarang {lam.size}")

    hours_plot = np.arange(1, 25)  # 1..24 agar rapi untuk paper

    plt.figure(figsize=(7.5, 3.8))
    plt.plot(hours_plot, lam, marker="o")
    plt.xlabel("Hour (h)")
    plt.ylabel("Load multiplier, λ(h) (p.u.)")
    plt.xticks(hours_plot)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.8)
    plt.tight_layout()

    outpath = os.path.join(outdir, fname)
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"[Paper] Saved daily load profile figure to: {outpath}")

def export_daily_load_profile_table(lambda_h,
                                    outdir="paper_outputs",
                                    fname_xlsx="Fig2_load_profile_lambda.xlsx",
                                    fname_csv="Fig2_load_profile_lambda.csv"):
    """
    Export data Figure 2 (λ(h) 24 jam) ke Excel/CSV.
    Output kolom:
      - Hour (1..24)
      - lambda_h (p.u.)
    """
    os.makedirs(outdir, exist_ok=True)

    lam = np.asarray(lambda_h, dtype=float).flatten()
    if lam.size != 24:
        raise ValueError(f"load_profile harus panjang 24, tetapi sekarang {lam.size}")

    df = pd.DataFrame({
        "Hour": np.arange(1, 25),
        "lambda_h (p.u.)": lam
    })

    xlsx_path = os.path.join(outdir, fname_xlsx)
    csv_path  = os.path.join(outdir, fname_csv)

    df.to_excel(xlsx_path, index=False)
    df.to_csv(csv_path, index=False)

    print(f"[Paper] Saved Fig2 λ(h) table XLSX to: {xlsx_path}")
    print(f"[Paper] Saved Fig2 λ(h) table CSV  to: {csv_path}")
    print("\n[Paper] Preview Fig2 λ(h):\n", df.to_string(index=False))


def export_growth_table(growth_rates_dict, planning_years=15, stage_years=(0, 5, 10, 15),
                        outdir="paper_outputs",
                        fname_csv="Table1_growth_multipliers.csv",
                        fname_xlsx="Table1_growth_multipliers.xlsx"):
    """
    Table: Load growth scenarios and growth multipliers (T=planning_years).
    growth_rates_dict: dict annual growth rate, contoh: {'Low':0.02,'Base':0.03,'High':0.05}
    multiplier(t) = (1+r)^t
    """
    os.makedirs(outdir, exist_ok=True)

    rows = []
    for scen, r in growth_rates_dict.items():
        r = float(r)
        row = {"Scenario": scen, "Annual growth rate r": r}
        for t in stage_years:
            row[f"Multiplier at year {t}"] = (1.0 + r) ** int(t)
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("Scenario")

    csv_path = os.path.join(outdir, fname_csv)
    xlsx_path = os.path.join(outdir, fname_xlsx)

    df.to_csv(csv_path, index=False)
    df.to_excel(xlsx_path, index=False)

    print(f"[Paper] Saved growth table CSV  to: {csv_path}")
    print(f"[Paper] Saved growth table XLSX to: {xlsx_path}")
    print("\n[Paper] Preview Table 1:\n", df.to_string(index=False))



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
S_base = 100.0  # MVA

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
#R_default = 1e-5
Sseed = 0.0

excel_file = "data-program/lines_base.xlsx"

# =================
# Sheet 1: topologi 
# =================
df_lines = pd.read_excel(excel_file, sheet_name="topologi") 

# Buang baris kosong kalau ada
df_lines = pd.read_excel(excel_file, sheet_name="topologi")
df_lines = df_lines.dropna(subset=["From", "To", "R_pu", "X_pu"])

lines_base = []
for _, row in df_lines.iterrows():
    u = int(row["From"])
    v = int(row["To"])
    R = float(row["R_pu"]) / S_base
    X = float(row["X_pu"]) / S_base
    lines_base.append((u, v, R, X, Sseed))   # <-- BENAR (5 elemen)


Rs = [R for (_, _, R, _, _) in lines_base]
Xs = [X for (_, _, _, X, _) in lines_base]
print("R min/max:", min(Rs), max(Rs))
print("X min/max:", min(Xs), max(Xs))



# Turunkan set bus dari topology dasar (tetap sama seperti kode lama)
buses = sorted({u for (u, _, _, _, _) in lines_base} | {v for (_, v, _, _, _) in lines_base})
all_buses = buses[:]
slack_bus = 2
pv_buses = [b for b in buses if b != slack_bus]

# ========================
# Sheet 2: beban dasar L_0 
# ========================
df_L0 = pd.read_excel(excel_file, sheet_name="Loads")  # atau sheet_name=1

# Buang baris yang Bus / L0_MW-nya kosong
df_L0 = df_L0.dropna(subset=["Bus", "L0_MW"])

# Bentuk dictionary L_0 dari Excel
L_0 = {}
for _, row in df_L0.iterrows():
    bus = int(row["Bus"])
    load = float(row["L0_MW"])
    L_0[bus] = load

print("Total base load sum(L0) [MW] =", sum(L_0.values()))
print("Max single-bus L0 [MW] =", max(L_0.values()))


# Pastikan semua bus punya entri (kalau ada bus yang tidak tertulis di Excel, diisi 0.0)
for b in buses:
    L_0.setdefault(b, 0.0)


# Turunkan set bus dari topology dasar
buses = sorted({u for (u, _, _, _, _) in lines_base} | {v for (_, v, _, _, _) in lines_base})
all_buses = buses[:]                              # alias bila dibutuhkan
pv_buses = [b for b in buses if b != slack_bus]   # PV tidak ditempatkan di slack

# Build children adjacency (untuk hitung downstream set)
children = {i: [] for i in buses}
for (u, v, R, X, S) in lines_base:
    children[u].append(v)

# Precompute downstream node-set untuk tiap edge (sekali saja)
edge_downstream = []  # list of set(node) untuk setiap edge index pada lines_base
for (u, v, R, X, S) in lines_base:
    stack = [v]
    ds = set()
    while stack:
        w = stack.pop()
        if w in ds:
            continue
        ds.add(w)
        stack.extend(children.get(w, []))
    edge_downstream.append(ds)


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

# =========================================================
# FIG. X — PV output factor profile (24 hours) + Excel export
# =========================================================

# 1) Rata-rata PV output factor per (Scenario, Hour) di seluruh PV buses
df_pv_hour_s = (
    df_pv.groupby(["Scenario", "Hour"])["PV Output Factor"]
         .mean()
         .reset_index()
)

# 2) Statistik lintas skenario per jam: mean/min/max (opsional untuk band)
df_pv_profile = (
    df_pv_hour_s.groupby("Hour")["PV Output Factor"]
                .agg(["mean", "min", "max"])
                .reset_index()
)

# Ubah Hour dari 0–23 menjadi 1–24 (lebih enak untuk paper)
df_pv_profile["Hour"] = df_pv_profile["Hour"] + 1

# Rapikan nama kolom
df_pv_profile = df_pv_profile.rename(columns={
    "mean": "PV_Factor_Mean",
    "min":  "PV_Factor_Min",
    "max":  "PV_Factor_Max"
})

print("\n=== PV output factor profile (hourly) ===")
print(df_pv_profile)

# 3) Simpan ke Excel
# 3) Simpan ke Excel (masuk folder paper_outputs)
os.makedirs("paper_outputs", exist_ok=True)
out_xlsx = os.path.join("paper_outputs", "FigX_pv_output_factor_profile.xlsx")

with pd.ExcelWriter(out_xlsx) as writer:
    df_pv_profile.to_excel(writer, sheet_name="HourlyProfile", index=False)
    df_pv_hour_s.assign(Hour=df_pv_hour_s["Hour"] + 1).to_excel(writer, sheet_name="ScenarioHourly", index=False)

print(f"\n[Saved] Excel PV profile -> {out_xlsx}")

# 4) Plot Figure + simpan PNG (masuk folder paper_outputs)
plt.figure(figsize=(8, 4))
plt.plot(df_pv_profile["Hour"], df_pv_profile["PV_Factor_Mean"],
         marker="o", linestyle="-", label="Mean PV output factor")

plt.fill_between(df_pv_profile["Hour"],
                 df_pv_profile["PV_Factor_Min"],
                 df_pv_profile["PV_Factor_Max"],
                 alpha=0.15, label="Min–Max across scenarios")

plt.xlabel("Hour (h)")
plt.ylabel("PV output factor (kW/kW)")
plt.title("Fig. X. PV output factor profile over a representative day")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

out_png = os.path.join("paper_outputs", "FigX_pv_output_factor_profile.png")
plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.show()

print(f"[Saved] Figure PV profile -> {out_png}")



# ===========================
# Parameter /Optimasi
# ===========================

x_min   = 10   # kW (atau 0 kalau mau lebih fleksibel)
x_max   = 40000 #20000               # kW
c_res   = 0.1               # contoh nilai kecil, silakan disesuaikan
n_max   = 6                   # Batas Jumlah PV
V2_min  = 0.95**2            # Batas Minimal Tegangan
V2_max  = 1.05**2            # Batas maksimal Tegangan
kappa   = 1.645
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
for e, (u, v, R, X, _) in enumerate(lines_base):
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
for e, (u, v, R, X, _) in enumerate(lines_base):
    lines.append((u, v, R, X, S_edge_MVA[e]))

# Index bantu edges_by_from / edges_by_to (topologi sama)
edges_by_from = {i: [] for i in all_buses}
edges_by_to   = {i: [] for i in all_buses}
for e, (u, v, R, X, S_MVA) in enumerate(lines):
    edges_by_from[u].append(e)
    edges_by_to[v].append(e)


if __name__ == "__main__":
    # Penampung hasil untuk SEMUA growth & model
    summary_rows = []       # untuk tabel ringkasan utama
    siting_by_growth = {}   # simpan df siting per growth & model
    adequacy_rows = []      #Penampung metrik adequacy (PV vs μ, μ+κσ)
    v_envelope = {}         # (growth, model) → DataFrame Hour, Vmin, Vmax

    # ============================
    # (Tambahan untuk PAPER)
    # ============================
    # Figure: profil beban harian λ(h) (pakai load_profile Anda)
    export_daily_load_profile(load_profile, outdir="paper_outputs", fname="Fig3_load_profile_lambda.png")

    # Table: growth scenarios & multipliers (pakai growth_scenarios Anda)
    export_growth_table(growth_scenarios, planning_years=planning_years, stage_years=(0, 5, 10, 15),
                        outdir="paper_outputs",
                        fname_csv="Table1_growth_multipliers.csv",
                        fname_xlsx="Table1_growth_multipliers.xlsx")
    
    # >>> Tambahan: export data Fig2 ke Excel/CSV
    export_daily_load_profile_table(load_profile,
                                    outdir="paper_outputs",
                                    fname_xlsx="Fig2_load_profile_lambda.xlsx",
                                    fname_csv="Fig2_load_profile_lambda.csv")


    # ============================
    # Loop utama 
    # ============================
    for name, g in growth_scenarios.items():
        growth_factor = (1 + g) ** planning_years   
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

        # --- Tambahkan Q-load (MVAr) dari PF tetap ---
        pf_load = 0.95
        tanphi = np.tan(np.arccos(pf_load))
        df_load["Q (MVAr)"] = df_load["Load (MW)"] * tanphi


        # --- Robust load per (Hour,Bus): mu + kappa*sigma ---
        df_bus = df_load.groupby(['Hour','Bus'])['Load (MW)'].agg(['mean','std']).reset_index()
        df_bus['std'] = df_bus['std'].fillna(0.0)
        robust_load_bh = {(int(r['Hour']), int(r['Bus'])): float(r['mean'] + kappa * r['std'])
                        for _, r in df_bus.iterrows()}


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
                                    n_max=n_max, V2_min=V2_min, V2_max=V2_max, pf_min=pf_min, tanphi=tanphi, growth_factor=growth_factor, alpha_pv=alpha_pv, beta_grid=beta_grid,
                                    total_pv_cap_max=160000,  # batas total kapasitas (kW), default 60 MW
                                    solve=True               # kalau True: langsung optimize
                                )

        print(f"[{name}] Menjalankan model stochastic...")
        model_stoc, vars_stoc = build_stochastic_pv_model(
                                    name, pv_buses, all_buses, hours, scenarios, lines, df_pv, df_load,
                                    L_0, edges_by_to, edges_by_from, slack_bus, x_max=x_max, x_min=x_min, n_max=n_max,
                                    V2_min=V2_min, V2_max=V2_max, pf_min=pf_min, tanphi=tanphi, growth_factor=growth_factor, alpha_pv=alpha_pv, beta_grid=beta_grid,
                                    total_pv_cap_max=160000,  # batas total kapasitas (kW), default 60 MW
                                    solve=True               # kalau True: langsung optimize
                                )
        
        print(f"[{name}] Menjalankan model robust...")
        model_rob, vars_rob = build_robust_pv_model(
                                    name=name, pv_buses=pv_buses, all_buses=all_buses, hours=hours, lines=lines, df_pv=df_pv,
                                    robust_load_bh=robust_load_bh, L_0=L_0, edges_by_to=edges_by_to, edges_by_from=edges_by_from,
                                    slack_bus=slack_bus, x_max=x_max, x_min=x_min, n_max=n_max, V2_min=V2_min, V2_max=V2_max,
                                    pf_min=pf_min, alpha_pv=alpha_pv, tanphi=tanphi, growth_factor=growth_factor, beta_grid=beta_grid,
                                    total_pv_cap_max=160000,
                                    solve=True
                                )
        
        # =======================
        # Ringkasan MODEL DETERMINISTIC
        # =======================
        x_det      = vars_det["x_det"]
        y_det      = vars_det["y_det"]
        V2_det     = vars_det["V2_det"]
        P_line_det = vars_det["P_line_det"]
        P_pv_det   = vars_det["P_pv_det"] 

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
            for e, (u, v, R, X, S_MVA) in enumerate(lines):
                P_lim = pf_min * S_MVA
                if P_lim <= 0:
                    continue
                for h in hours:
                    ratio = abs(P_line_det[e, h].X) / P_lim
                    max_loading_det = max(max_loading_det, ratio)
            max_loading_det_pct = 100.0 * max_loading_det

            # --- Envelope tegangan per jam (DET) ---
            vmin_by_h_det = []
            vmax_by_h_det = []
            for h in hours:
                vs = [(V2_det[i, h].X)**0.5 for i in all_buses]
                vmin_by_h_det.append(min(vs))
                vmax_by_h_det.append(max(vs))

            df_env_det = pd.DataFrame({
                "Hour": hours,
                "Vmin": vmin_by_h_det,
                "Vmax": vmax_by_h_det,
            })
            v_envelope[(name, "DET")] = df_env_det


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
            for e, (u, v, R, X, S_MVA) in enumerate(lines):
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

            # --- Envelope tegangan per jam (STOCH) ---
            vmin_by_h_stoc = []
            vmax_by_h_stoc = []
            for h in hours:
                vmin_h = float("inf")
                vmax_h = 0.0
                for s in scenarios:
                    for i in all_buses:
                        v = (V2_stoc[i, h, s].X)**0.5
                        vmin_h = min(vmin_h, v)
                        vmax_h = max(vmax_h, v)
                vmin_by_h_stoc.append(vmin_h)
                vmax_by_h_stoc.append(vmax_h)

            df_env_stoc = pd.DataFrame({
                "Hour": hours,
                "Vmin": vmin_by_h_stoc,
                "Vmax": vmax_by_h_stoc,
            })
            v_envelope[(name, "STOCH")] = df_env_stoc


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

        if model_rob.Status == GRB.OPTIMAL:
            rob_results = []
            for i in pv_buses:
                if y_rob[i].X > 0.5:
                    rob_results.append([i, x_rob[i].X / 1000.0])
            df_rob_siting = pd.DataFrame(rob_results, columns=["Bus", "PV Capacity (MW)"])

            total_pv_rob_mw = sum(x_rob[i].X for i in pv_buses) / 1000.0
            num_sites_rob   = sum(1 for i in pv_buses if y_rob[i].X > 0.5)

            # Voltage & loading: worst case across semua s
            vmin_rob = min((V2_rob[i,h].X)**0.5 for i in all_buses for h in hours)
            vmax_rob = max((V2_rob[i,h].X)**0.5 for i in all_buses for h in hours)

            max_loading_rob = 0.0
            for e, (u, v, R, X, S_MVA) in enumerate(lines):
                P_lim = pf_min * S_MVA
                if P_lim <= 0:
                    continue
                for h in hours:
                    ratio = abs(P_line_rob[e, h].X) / P_lim
                    max_loading_rob = max(max_loading_rob, ratio)
            max_loading_rob_pct = 100.0 * max_loading_rob

            # Energi PV & grid per "hari representatif"
            pv_energy_day_rob   = sum(sum(P_pv_rob[i, h].X for i in pv_buses) for h in hours)
            grid_energy_day_rob = sum(P_grid_rob[h].X for h in hours)



            # --- Envelope tegangan per jam (ROBUST) ---
            # --- Envelope tegangan per jam (ROBUST) ---
            vmin_by_h_rob = []
            vmax_by_h_rob = []

            for h in hours:
                vs = [(V2_rob[i, h].X)**0.5 for i in all_buses]  # tidak ada indeks skenario
                vmin_by_h_rob.append(min(vs))
                vmax_by_h_rob.append(max(vs))

            df_env_rob = pd.DataFrame({
                "Hour": hours,
                "Vmin": vmin_by_h_rob,
                "Vmax": vmax_by_h_rob,
            })
            v_envelope[(name, "ROBUST")] = df_env_rob



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
            })

            siting_by_growth[(name, "ROBUST")] = df_rob_siting.copy()
        else:
            print(f"[{name}] Model robust tidak optimal, status =", model_rob.Status)

        # -------------------------------
        # ADEQUACY: PV vs μ dan μ + κσ
        # -------------------------------
        # Kita bandingkan di level sistem (agregat semua bus)
        # - DET  : pakai PV deterministik (sudah merepresentasikan rata-rata skenario)
        # - STOCH: pakai rata-rata PV across scenarios
        # - ROB  : rata-rata PV saja, dan PV + R_res (robust reserve)

        # Pastikan semua model optimal dulu
        if (model_det.Status == GRB.OPTIMAL and
            model_stoc.Status == GRB.OPTIMAL and
            model_rob.Status == GRB.OPTIMAL):

            # --- Helper: mu_h dan mu_kappa_sigma_h per jam ---
            # mu_load dan sigma_load sudah dihitung di awal loop growth
            # mu_load[h] : rata-rata beban sistem per jam (MW)
            # sigma_load[h] : std dev beban sistem per jam (MW)
            # mu+kappaσ : batas adequacy robust
            # (satuan semua MW)

            # 1) MODEL DETERMINISTIC: PV_det vs μ dan μ+κσ
            min_gap_mu_det      = float("inf")
            min_gap_muk_det     = float("inf")

            for h in hours:
                # total PV expected output di jam h (MW)
                pv_det_h = sum(P_pv_det[i, h].X for i in pv_buses)

                mu_h      = mu_load[h]
                mu_k_h    = mu_load[h] + kappa * sigma_load[h]

                gap_mu    = pv_det_h - mu_h
                gap_muk   = pv_det_h - mu_k_h

                min_gap_mu_det  = min(min_gap_mu_det,  gap_mu)
                min_gap_muk_det = min(min_gap_muk_det, gap_muk)

            adequacy_rows.append({
                "Growth": name,
                "Model": "DET",
                "Min PV - μ (MW)": min_gap_mu_det,
                "Min PV - (μ+κσ) (MW)": min_gap_muk_det,
            })

            # 2) MODEL STOCHASTIC: rata-rata PV_stoc vs μ dan μ+κσ
            min_gap_mu_stoc  = float("inf")
            min_gap_muk_stoc = float("inf")

            for h in hours:
                # rata-rata across scenarios: (1/|S|) Σ_s Σ_i P_pv_stoc[i,h,s]
                pv_sum = 0.0
                for s in scenarios:
                    pv_sum += sum(P_pv_stoc[i, h, s].X for i in pv_buses)
                pv_avg_h = pv_sum / len(scenarios)  # MW

                mu_h    = mu_load[h]
                mu_k_h  = mu_load[h] + kappa * sigma_load[h]

                gap_mu   = pv_avg_h - mu_h
                gap_muk  = pv_avg_h - mu_k_h

                min_gap_mu_stoc  = min(min_gap_mu_stoc,  gap_mu)
                min_gap_muk_stoc = min(min_gap_muk_stoc, gap_muk)

            adequacy_rows.append({
                "Growth": name,
                "Model": "STOCH",
                "Min PV - μ (MW)": min_gap_mu_stoc,
                "Min PV - (μ+κσ) (MW)": min_gap_muk_stoc,
            })

            # 3) MODEL ROBUST (Opsi A1): tanpa skenario, tanpa reserve
            min_gap_mu_rob  = float("inf")
            min_gap_muk_rob = float("inf")

            for h in hours:
                # total PV output robust di jam h (MW) - tanpa skenario
                pv_rob_h = sum(P_pv_rob[i, h].X for i in pv_buses)

                mu_h   = mu_load[h]
                mu_k_h = mu_load[h] + kappa * sigma_load[h]

                min_gap_mu_rob  = min(min_gap_mu_rob,  pv_rob_h - mu_h)
                min_gap_muk_rob = min(min_gap_muk_rob, pv_rob_h - mu_k_h)

            adequacy_rows.append({
                "Growth": name,
                "Model": "ROBUST",
                "Min PV - μ (MW)": min_gap_mu_rob,
                "Min PV - (μ+κσ) (MW)": min_gap_muk_rob,
            })

        else:
            print(f"[{name}] Adequacy tidak dihitung karena ada model yang tidak optimal.")



    print("\n=== Detail lokasi PV per growth & model ===")
    for (growth, model), df_siting in siting_by_growth.items():
        print(f"\n[{growth} - {model}]")
        if df_siting.empty:
            print("  (Tidak ada PV dipasang)")
        else:
            print(df_siting)
    
    # =========================================================
    # TABLE A — PV siting & sizing summary (export to Excel/CSV)
    # Letakkan setelah print "Detail lokasi PV..." dan sebelum df_summary dibuat
    # =========================================================
    os.makedirs("paper_outputs", exist_ok=True)

    # Long table: Growth, Model, Bus, PV Capacity (MW)
    tableA_long_rows = []
    for (growth, model), df_siting in siting_by_growth.items():
        if df_siting is None or df_siting.empty:
            continue
        for _, r in df_siting.iterrows():
            tableA_long_rows.append({
                "Growth": growth,
                "Model": model,
                "Bus": int(r["Bus"]),
                "PV Capacity (MW)": float(r["PV Capacity (MW)"])
            })

    df_tableA_long = pd.DataFrame(tableA_long_rows)

    # Summary per (Growth, Model): Total PV + #Sites
    tableA_sum_rows = []
    for (growth, model), df_siting in siting_by_growth.items():
        if df_siting is None or df_siting.empty:
            total_pv = 0.0
            nsite = 0
        else:
            total_pv = float(df_siting["PV Capacity (MW)"].sum())
            nsite = int(df_siting.shape[0])

        tableA_sum_rows.append({
            "Growth": growth,
            "Model": model,
            "Total PV (MW)": total_pv,
            "#Sites": nsite
        })

    df_tableA_summary = pd.DataFrame(tableA_sum_rows).sort_values(["Growth", "Model"])

    # Save
    tableA_path = os.path.join("paper_outputs", "TableA_PV_siting_sizing.xlsx")
    with pd.ExcelWriter(tableA_path) as writer:
        df_tableA_summary.to_excel(writer, sheet_name="Summary", index=False)
        df_tableA_long.to_excel(writer, sheet_name="SitingDetail", index=False)

    df_tableA_summary.to_csv(os.path.join("paper_outputs", "TableA_PV_siting_sizing_summary.csv"), index=False)
    df_tableA_long.to_csv(os.path.join("paper_outputs", "TableA_PV_siting_sizing_detail.csv"), index=False)

    print(f"[Paper] Saved Table A to: {tableA_path}")



df_summary = pd.DataFrame(summary_rows)
print("\n=== Ringkasan semua growth & model ===")
print(df_summary)
df_summary.to_excel("summary_det_stoch_robust.xlsx", index=False)

# =========================================================
# TABLE B — Comparative metrics (paper-ready subset)
# Letakkan setelah df_summary disimpan
# =========================================================
os.makedirs("paper_outputs", exist_ok=True)

df_tableB = df_summary[[
    "Growth", "Model",
    "Total PV (MW)", "#Sites",
    "Vmin (p.u.)", "Vmax (p.u.)",
    "Max line loading (%)"
]].copy()

tableB_path = os.path.join("paper_outputs", "TableB_Comparative_Metrics.xlsx")
df_tableB.to_excel(tableB_path, index=False)
df_tableB.to_csv(os.path.join("paper_outputs", "TableB_Comparative_Metrics.csv"), index=False)

print(f"[Paper] Saved Table B to: {tableB_path}")


# ---- Tabel adequacy tambahan ----
df_adequacy = pd.DataFrame(adequacy_rows)
print("\n=== Ringkasan adequacy (PV vs μ, μ+κσ) ===")
print(df_adequacy)
df_adequacy.to_excel("adequacy_det_stoch_robust.xlsx", index=False)

# =========================================
# Table X. Summary of optimization formulations (DET vs STOCH vs ROBUST)
# (tabel deskriptif untuk paper)
# =========================================
tableX_rows = [
    {
        "Formulation": "Deterministic (DET)",
        "Uncertainty handling": "Single representative operating point (mean/expected inputs)",
        "Indices": "h ∈ {0..23}",
        "Load input": "df_load (uses mean profile implicitly in formulation)",
        "PV input": "df_pv (PV Output Factor, without scenario index in constraints)",
        "Objective (typical in this code)": "min α_pv·Σ_i x_i  +  β_grid·Σ_h P_grid(h) (no scenario index)",
        "Backbone constraints": "LinDistFlow-Lite nodal balance + voltage drop, voltage limits, thermal limits",
        "Siting/sizing constraints": "x_min ≤ x_i ≤ x_max·y_i ; Σ_i y_i ≤ n_max ; cap_per_bus(i)",
        "Inverter VAR constraints": "Q_pv bounded by inverter capability as function of PV capacity/output",
    },
    {
        "Formulation": "Scenario-based stochastic (STOCH)",
        "Uncertainty handling": "Monte Carlo scenarios (sampled load & PV factors), expected-value objective",
        "Indices": "h ∈ {0..23}, s ∈ {1..N_MC}",
        "Load input": "df_load (Scenario, Hour, Bus)",
        "PV input": "df_pv (Scenario, Hour, Bus) → PV Output Factor",
        "Objective (typical in this code)": "min α_pv·Σ_i x_i  +  β_grid·(1/|S|)·Σ_s Σ_h P_grid(h,s)",
        "Backbone constraints": "LinDistFlow-Lite nodal balance + voltage drop for each (h,s), voltage limits, thermal limits",
        "Siting/sizing constraints": "x_min ≤ x_i ≤ x_max·y_i ; Σ_i y_i ≤ n_max ; cap_per_bus(i)",
        "Inverter VAR constraints": "Q_pv bounded by inverter capability for each (h,s)",
    },
    {
        "Formulation": "Robust feeder-aware (ROBUST)",
        "Uncertainty handling": "Robust load per bus-hour: μ + κσ (no scenario index in network constraints)",
        "Indices": "h ∈ {0..23}",
        "Load input": "robust_load_bh[(h,i)] = μ_{i}(h) + κ·σ_{i}(h)",
        "PV input": "df_pv (PV Output Factor, used without scenario index in constraints)",
        "Objective (typical in this code)": "min α_pv·Σ_i x_i  +  β_grid·Σ_h P_grid(h) (robust-feasible network)",
        "Backbone constraints": "LinDistFlow-Lite nodal balance + voltage drop under robust load, voltage limits, thermal limits",
        "Siting/sizing constraints": "x_min ≤ x_i ≤ x_max·y_i ; Σ_i y_i ≤ n_max ; cap_per_bus(i)",
        "Inverter VAR constraints": "Q_pv bounded by inverter capability as function of PV capacity/output",
    },
]

df_tableX = pd.DataFrame(tableX_rows)

print("\n=== Table X. Summary of optimization formulations (DET vs STOCH vs ROBUST) ===")
print(df_tableX)

# Simpan ke Excel (bisa langsung dipakai untuk copy ke Word)
df_tableX.to_excel("TableX_Summary_Formulations.xlsx", index=False)


# =========================================
# 3. Voltage envelope: tabel + Excel
# =========================================

env_rows = []
for (growth, model), df_env in v_envelope.items():
    for _, row in df_env.iterrows():
        env_rows.append({
            "Growth": growth,
            "Model": model,
            "Hour": int(row["Hour"]),
            "Vmin (p.u.)": float(row["Vmin"]),
            "Vmax (p.u.)": float(row["Vmax"]),
        })

if env_rows:
    df_env_all = pd.DataFrame(env_rows)
    df_env_all['Growth'] = pd.Categorical(
        df_env_all['Growth'],
        categories=['Low', 'Base', 'High'],
        ordered=True
    )
    df_env_all = df_env_all.sort_values(['Growth', 'Model', 'Hour'])

    print("\n=== Voltage envelope – semua growth & model ===")
    # Kalau mau lihat semua, hapus .head(60)
    print(df_env_all.head(60))

    # Simpan ke Excel:
    #  - Sheet "All"   : semua growth+model
    #  - Sheet per kombinasi (Low_DET, Base_STOCH, dst.)
    os.makedirs("paper_outputs", exist_ok=True)
    env_all_path = os.path.join("paper_outputs", "voltage_envelope_all.xlsx")
    with pd.ExcelWriter("voltage_envelope_all.xlsx") as writer:
        df_env_all.to_excel(writer, sheet_name="All", index=False)
        for (growth, model), df_env in v_envelope.items():
            sheet_name = f"{growth}_{model}"
            # sheet name max 31 chars, aman di sini
            df_env.to_excel(writer, sheet_name=sheet_name, index=False)
else:
    print("\n[Tidak ada data voltage envelope di v_envelope]")


# =========================================
# 4. Voltage envelope per jam (contoh: Base growth)
# =========================================
growth_to_plot = "High"

# Pastikan data envelope untuk growth & model tersedia
models_env = []
for m in ["DET", "STOCH", "ROBUST"]:
    key = (growth_to_plot, m)
    if key in v_envelope:
        models_env.append(m)

if models_env:
    plt.figure(figsize=(8, 4))

    # Plot Vmin per model
    for m in models_env:
        df_env = v_envelope[(growth_to_plot, m)]
        plt.plot(df_env["Hour"] + 1, df_env["Vmin"],
                 marker="o",
                 linestyle="-",
                 label=f"{m} - Vmin")

    # Plot Vmax per model
    for m in models_env:
        df_env = v_envelope[(growth_to_plot, m)]
        plt.plot(df_env["Hour"] + 1, df_env["Vmax"],
                 marker="",
                 linestyle="--",
                 label=f"{m} - Vmax")

    # Batas tegangan
    plt.axhline(0.95, linestyle=":", label="Vmin limit (0.95 p.u.)")
    plt.axhline(1.05, linestyle=":", label="Vmax limit (1.05 p.u.)")

    plt.xlabel("Hour of day")
    plt.ylabel("Voltage magnitude (p.u.)")
    plt.title(f"Voltage envelope – {growth_to_plot} growth scenario")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    # --- Save Fig. A + export data (paper outputs) ---
    os.makedirs("paper_outputs", exist_ok=True)

    # Ubah hour 0..23 -> 1..24 agar konsisten dengan λ(h)
    df_env_save = pd.DataFrame()
    for m in models_env:
        tmp = v_envelope[(growth_to_plot, m)].copy()
        tmp["Hour"] = tmp["Hour"] + 1
        tmp["Growth"] = growth_to_plot
        tmp["Model"] = m
        df_env_save = pd.concat([df_env_save, tmp], ignore_index=True)

    env_xlsx = os.path.join("paper_outputs", f"FigA_VoltageEnvelope_{growth_to_plot}.xlsx")
    df_env_save.to_excel(env_xlsx, index=False)
    df_env_save.to_csv(os.path.join("paper_outputs", f"FigA_VoltageEnvelope_{growth_to_plot}.csv"), index=False)

    figA_path = os.path.join("paper_outputs", f"FigA_VoltageEnvelope_{growth_to_plot}.png")
    plt.savefig(figA_path, dpi=300, bbox_inches="tight")

    print(f"[Paper] Saved Fig. A PNG to: {figA_path}")
    print(f"[Paper] Saved Fig. A data to: {env_xlsx}")

    plt.show()
else:
    print(f"Tidak ada data envelope untuk growth = {growth_to_plot}")


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

# =========================================================
# FIG. B — Export max line loading data to Excel/CSV
# =========================================================
os.makedirs("paper_outputs", exist_ok=True)

df_figB_data = pivot_loading.reset_index()
figB_data_path = os.path.join("paper_outputs", "FigB_MaxLineLoading_Data.xlsx")
df_figB_data.to_excel(figB_data_path, index=False)
df_figB_data.to_csv(os.path.join("paper_outputs", "FigB_MaxLineLoading_Data.csv"), index=False)
print(f"[Paper] Saved Fig. B data to: {figB_data_path}")


# Pastikan hanya pakai model yang memang ada di kolom
models = [m for m in ['DET', 'STOCH', 'ROBUST'] if m in pivot_loading.columns]

# -----------------------------------------
# 1A. BAR PLOT – Maximum line loading
# -----------------------------------------
plt.figure(figsize=(8, 4))
ax = pivot_loading[models].plot(kind='bar', figsize=(8, 4))

ax.set_xlabel("Load growth scenario")
ax.set_ylabel("Max line loading (%)")
ax.set_title("Fig. B1. Maximum line loading vs growth and model (bar)")
ax.legend(title="Model", loc="upper left")
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()

figB_bar_path = os.path.join("paper_outputs", "FigB_MaxLineLoading_Bar.png")
ax.get_figure().savefig(figB_bar_path, dpi=300, bbox_inches="tight")
print(f"[Paper] Saved Fig. B bar plot to: {figB_bar_path}")

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

figB_line_path = os.path.join("paper_outputs", "FigB_MaxLineLoading_Line.png")
plt.savefig(figB_line_path, dpi=300, bbox_inches="tight")
print(f"[Paper] Saved Fig. B line plot to: {figB_line_path}")

plt.show()

