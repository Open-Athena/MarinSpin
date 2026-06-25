"""Combine the three quantitative line plots into one triptych (from saved .npz, no recompute)."""
import numpy as np
import poster_style
poster_style.apply()
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 25, "axes.titlesize": 23, "axes.labelsize": 25,
                     "legend.fontsize": 21, "xtick.labelsize": 22, "ytick.labelsize": 22})

D = "scratch/data"
BLUE, RED = "#0F4C81", "#C1440E"
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(19, 5.0), constrained_layout=True)

# (1) equilibrium correlation length vs T
d = np.load(f"{D}/xi_vs_temp.npz")
x, mm, ms_, km, ks_ = d["temps"], d["model_mean"], d["model_std"], d["kmc_mean"], d["kmc_std"]
ax1.axvspan(2.0, 2.8, color="gray", alpha=0.12)
ax1.text(2.4, max(km.max(), mm.max()) * 0.97, "held out\n($T_c$)", ha="center", va="top", fontsize=17, color="gray")
ax1.errorbar(x, km, yerr=ks_, fmt="s-", color=RED, lw=2, ms=6, capsize=3, label="KMC")
ax1.errorbar(x, mm, yerr=ms_, fmt="o-", color=BLUE, lw=2, ms=6, capsize=3, label="MarinSpin")
ax1.set_xlabel("temperature $T$"); ax1.set_ylabel(r"correlation length $\xi$")
ax1.set_title("Equilibrium correlation length"); ax1.legend(); ax1.grid(True, alpha=0.3)

# (2) residual D_KL / H, flip-site vs dt
d = np.load(f"{D}/error_breakdown.npz")
x, fp, fd = d["temps"], d["flip_site_pct"], d["dt_pct"]
ax2.axvspan(2.0, 2.8, color="gray", alpha=0.12)
ax2.plot(x, fp, "o-", color=BLUE, lw=2.5, ms=6, label="flip-site")
ax2.plot(x, fd, "s-", color=RED, lw=2.5, ms=6, label=r"waiting time $\Delta t$")
ax2.set_xlabel("temperature $T$"); ax2.set_ylabel(r"$D_{\mathrm{KL}}\,/\,S(p_{\mathrm{KMC}})$  (%)")
ax2.set_title("Residual loss by token type"); ax2.legend(); ax2.grid(True, alpha=0.3)

# (3) directional correlation (cluster-shape isotropy) at T=2.8
d = np.load(f"{D}/anisotropy_T2.8.npz")
r = d["r"]
for nm, Cv, Ch, xv, xh, c in [("MarinSpin", d["model_Cv"], d["model_Ch"], float(d["model_xi_v"]), float(d["model_xi_h"]), BLUE),
                              ("KMC", d["kmc_Cv"], d["kmc_Ch"], float(d["kmc_xi_v"]), float(d["kmc_xi_h"]), RED)]:
    ax3.plot(r, Cv, "-o", ms=4, color=c, label=f"{nm} vertical")
    ax3.plot(r, Ch, "--s", ms=4, color=c, mfc=poster_style.BOX, label=f"{nm} horizontal")
ax3.axhline(0, color="gray", lw=0.6)
ax3.set_xlabel("separation $r$"); ax3.set_ylabel("$C(r)$")
ax3.set_title("Correlation function by direction")
ax3.legend(fontsize=16, ncol=1, loc="upper right", title=r"$T{=}2.8$", title_fontsize=17)
ax3.grid(True, alpha=0.3)

fig.savefig("../poster/figures/triptych.png", dpi=300)
print("wrote triptych.png")
