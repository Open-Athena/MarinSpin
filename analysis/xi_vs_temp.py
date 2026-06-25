"""Equilibrium correlation length vs temperature: model vs KMC, with error bars.

The out-of-equilibrium quench xi(t) is fluctuation-dominated (a single coarsening realization is not
meaningful). The stationary observable is instead the TYPICAL correlation length of EQUILIBRATED
configs at each temperature. We compare:
  - KMC ground truth: xi of the eval-set equilibrium configs (real equilibrated samples).
  - model: xi of configs the model produces when started from equilibrium at T and evolved
           in-distribution at the SAME T (so it must hold the correct equilibrium structure).
xi is the true correlation length from the connected correlation G(r)=<s0 sr>-<s>^2 ~ e^{-r/xi}
(direction-averaged integrated length) --- the SAME estimator as the anisotropy plot, not L/sqrt(n_c).
Error bars = +/- 1 std (the typical spread)."""
import numpy as np
from haliax.partitioning import set_mesh
from levanter.grug.sharding import compact_grug_mesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from marin_spin.compare_bkl import equilibrium_configs
from marin_spin.corr_anisotropy import corr_dir, xi_integrated, model_rollout_snaps
from marin_spin.rollout_quench import build_tokenizer, load_model
from marin_spin.tokenize_ising import LATTICE_L, WINDOW_EVENTS

TEMPS = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5]
H5 = "/Users/yaelelmatad/Downloads"
CKPT = "scratch/ckpt/step-49400"   # OG hero baseline = best dynamics model
N_KMC, N_CH, N_WIN, BURN = 200, 16, 2, 0   # KV-cache makes the rollout cheap; use a real 2-window test
L, W = LATTICE_L, WINDOW_EVENTS


def _xi_set(s):
    """Raw two-point correlation length over configs s [N,L,L]: C(r)=<s0 sr> (no mean subtraction),
    so it stays high in the ordered (cold) phase and decays when hot. Normalized (C(0)=1), integrated
    to the first zero, direction-averaged."""
    vals = []
    for axis in (1, 2):
        C = np.array([(s * np.roll(s, -r, axis=axis)).mean() for r in range(L // 2 + 1)])
        C = C / C[0]
        vals.append(xi_integrated(C))
    return 0.5 * (vals[0] + vals[1])


def xi_ensemble(configs, rng, nboot=200):
    """Ensemble correlation length + bootstrap std."""
    s = configs.astype(np.float64)
    boots = np.array([_xi_set(s[rng.integers(0, len(s), len(s))]) for _ in range(nboot)])
    return _xi_set(s), boots.std()


def main():
    tok = build_tokenizer()
    res = {}
    with set_mesh(compact_grug_mesh()):
        model = load_model(CKPT)
        for T in TEMPS:
            spins0 = equilibrium_configs(T, N_CH, H5).astype(np.int8)
            ms = model_rollout_snaps(model, tok, spins0, T, N_WIN, W, 1.0,
                                     np.random.default_rng(7), BURN)  # (N_CH, n_keep, L, L)
            mcfg = ms.reshape(-1, L, L)
            kcfg = equilibrium_configs(T, N_KMC, H5)
            mmean, mstd = xi_ensemble(mcfg, np.random.default_rng(11))
            kmean, kstd = xi_ensemble(kcfg, np.random.default_rng(12))
            res[T] = (mmean, mstd, kmean, kstd)
            print(f"T={T:.1f}  model {res[T][0]:.2f}±{res[T][1]:.2f}   KMC {res[T][2]:.2f}±{res[T][3]:.2f}", flush=True)

    x = np.array(TEMPS)
    mm = np.array([res[T][0] for T in TEMPS]); ms_ = np.array([res[T][1] for T in TEMPS])
    km = np.array([res[T][2] for T in TEMPS]); ks_ = np.array([res[T][3] for T in TEMPS])
    np.savez("scratch/data/xi_vs_temp.npz", temps=x, model_mean=mm, model_std=ms_, kmc_mean=km, kmc_std=ks_)
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.axvspan(2.0, 2.8, color="gray", alpha=0.12)
    ax.text(2.4, max(km.max(), mm.max()) * 0.96, "held out\n($T_c$)", ha="center", va="top",
            fontsize=9, color="gray")
    ax.errorbar(x, km, yerr=ks_, fmt="s-", color="#C1440E", lw=2, ms=7, capsize=3, label="KMC (equilibrium)")
    ax.errorbar(x, mm, yerr=ms_, fmt="o-", color="#0F4C81", lw=2, ms=7, capsize=3, label="model")
    ax.set_xlabel("temperature $T$", fontsize=12)
    ax.set_ylabel(r"correlation length  $\xi$  (from $C(r){=}\langle s_0 s_r\rangle$)", fontsize=12)
    ax.set_title("Equilibrium correlation length vs temperature", fontsize=12)
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("scratch/xi_vs_temp.png", dpi=150)
    print("wrote scratch/xi_vs_temp.png")


if __name__ == "__main__":
    main()
