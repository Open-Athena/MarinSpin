"""Re-render the data-backed poster figures from saved .npz with the OA blog style (Lato + cream bg).

No model/rollout compute --- this is the cheap-reformat path enabled by saving plot data."""
import numpy as np
import poster_style
poster_style.apply()
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

D = "scratch/data"
FIG = "../poster/figures"
OACMAP = ListedColormap(["#7B4A2A", "#E8C18A"])  # OA-brand 2-spin map
BLUE, RED = "#0F4C81", "#C1440E"

# ---- equilibrium xi(T) ------------------------------------------------------
d = np.load(f"{D}/xi_vs_temp.npz")
x, mm, ms_, km, ks_ = d["temps"], d["model_mean"], d["model_std"], d["kmc_mean"], d["kmc_std"]
fig, ax = plt.subplots(figsize=(7.4, 4.4))
ax.axvspan(2.0, 2.8, color="gray", alpha=0.12)
ax.text(2.4, max(km.max(), mm.max()) * 0.96, "held out\n($T_c$)", ha="center", va="top", fontsize=12, color="gray")
ax.errorbar(x, km, yerr=ks_, fmt="s-", color=RED, lw=2, ms=7, capsize=3, label="KMC (equilibrium)")
ax.errorbar(x, mm, yerr=ms_, fmt="o-", color=BLUE, lw=2, ms=7, capsize=3, label="MarinSpin")
ax.set_xlabel("temperature $T$", fontsize=15)
ax.set_ylabel(r"equilibrium correlation length  $\xi=L/\sqrt{n_c}$", fontsize=15)
ax.set_title("Equilibrium correlation length vs temperature", fontsize=15)
ax.legend(fontsize=14); ax.grid(True, alpha=0.3)
fig.tight_layout(); fig.savefig(f"{FIG}/xi_ensemble.png", dpi=150); plt.close(fig)

# ---- error breakdown (% of oracle floor) -----------------------------------
d = np.load(f"{D}/error_breakdown.npz")
x, fp, fd = d["temps"], d["flip_site_pct"], d["dt_pct"]
fig, ax = plt.subplots(figsize=(7.4, 4.0))
ax.plot(x, fp, "o-", color=BLUE, lw=2.5, ms=7, label="flip-site (position)")
ax.plot(x, fd, "s-", color=RED, lw=2.5, ms=7, label=r"waiting time $\Delta t$")
ax.axvspan(2.0, 2.8, color="gray", alpha=0.12)
ax.text(2.4, max(fp.max(), fd.max()) * 0.9, "held out\n($T_c$)", ha="center", va="top", fontsize=12, color="gray")
ax.set_xlabel("temperature $T$", fontsize=15)
ax.set_ylabel(r"$D_{\mathrm{KL}}\,/\,S(p_{\mathrm{KMC}})$  (%)", fontsize=15)
ax.set_title(r"Residual $D_{\mathrm{KL}}$ as a fraction of the oracle floor $S$", fontsize=15)
ax.legend(fontsize=14); ax.grid(True, alpha=0.3)
fig.tight_layout(); fig.savefig(f"{FIG}/error_breakdown.png", dpi=150); plt.close(fig)

# ---- anisotropy (merged, equilibrium T=2.8) --------------------------------
d = np.load(f"{D}/anisotropy_T2.8.npz")
r = d["r"]
series = {"MarinSpin": (d["model_Cv"], d["model_Ch"], float(d["model_xi_v"]), float(d["model_xi_h"]), BLUE),
          "KMC": (d["kmc_Cv"], d["kmc_Ch"], float(d["kmc_xi_v"]), float(d["kmc_xi_h"]), RED)}
fig, a = plt.subplots(figsize=(7.2, 4.6))
for name, (Cv, Ch, xv, xh, c) in series.items():
    a.plot(r, Cv, "-o", ms=4, color=c, label=rf"{name} vertical ($\xi$={xv:.2f})")
    a.plot(r, Ch, "--s", ms=4, color=c, mfc=poster_style.BOX, label=rf"{name} horizontal ($\xi$={xh:.2f})")
a.axhline(0, color="gray", lw=0.6)
a.set_xlabel("separation $r$"); a.set_ylabel("$C(r)$")
ratios = "   ".join(rf"{n}: $\xi_v/\xi_h$={v[2] / v[3]:.3f}" for n, v in series.items())
a.set_title(f"Directional spin correlation: vertical vs horizontal  (T=2.8)\n{ratios}", fontsize=14)
a.legend(fontsize=12, ncol=2); a.grid(True, alpha=0.3)
fig.tight_layout(); fig.savefig(f"{FIG}/anisotropy.png", dpi=140); plt.close(fig)

# ---- quench montage (brand colors, time + xi + domain) ---------------------
d = np.load(f"{D}/quench_montage.npz")
mc, kc = d["model_configs"], d["kmc_configs"]
mx, kx, md, kd, mt, kt = d["model_xi"], d["kmc_xi"], d["model_domain"], d["kmc_domain"], d["model_time"], d["kmc_time"]
NCOL = mc.shape[0]
fig, axes = plt.subplots(2, NCOL, figsize=(3.0 * NCOL, 6.8), constrained_layout=True)
for c in range(NCOL):
    axes[0, c].imshow(mc[c], cmap=OACMAP, vmin=-1, vmax=1, interpolation="nearest")
    axes[0, c].set_title(r"$\xi$=" + f"{mx[c]:.1f}" + f"\n$t$={mt[c]:.1f}", fontsize=18)
    axes[1, c].imshow(kc[c], cmap=OACMAP, vmin=-1, vmax=1, interpolation="nearest")
    axes[1, c].set_title(r"$\xi$=" + f"{kx[c]:.1f}" + f"\n$t$={kt[c]:.1f}", fontsize=18)
    for rr in (0, 1):
        axes[rr, c].set_xticks([]); axes[rr, c].set_yticks([])
axes[0, 0].set_ylabel("MarinSpin", fontsize=23, fontweight="bold")
axes[1, 0].set_ylabel("KMC", fontsize=23, fontweight="bold")
fig.savefig(f"{FIG}/quench_montage.png", dpi=300); plt.close(fig)
print("replotted xi, error, anisotropy, montage from saved data (Lato + cream bg)")
