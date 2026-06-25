"""Clean ensemble coarsening figure: correlation length xi(t), model vs KMC (mean +/- std band)."""
import numpy as np
from haliax.partitioning import set_mesh
from levanter.grug.sharding import compact_grug_mesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from marin_spin.compare_bkl import checkerboard, transformer_ensemble_grug, bkl_ensemble
from marin_spin.rollout_quench import build_tokenizer, load_model
from marin_spin.tokenize_ising import LATTICE_L, WINDOW_EVENTS

L, T, W = LATTICE_L, 1.5, WINDOW_EVENTS
NW, SNAP, NCH, NBKL, ST = 24, 8, 8, 48, 0.85

tok = build_tokenizer()
with set_mesh(compact_grug_mesh()):
    model = load_model("scratch/ckpt-enriched-d256/step-100000")
    _, tr = transformer_ensemble_grug(model, tok, checkerboard(L), T, NW, W,
                                      n_chains=NCH, sample_temp=ST, snapshot_every=SNAP,
                                      rng=np.random.default_rng(2))
_, bk = bkl_ensemble(checkerboard(L), T, NW, W, n_chains=NBKL, snapshot_every=SNAP,
                     rng=np.random.default_rng(1))

xm, xb = tr["xi"], bk["xi"]          # (chains, snaps)
x = np.arange(xm.shape[1])
fig, ax = plt.subplots(figsize=(7.2, 4.0))
for arr, c, lab in [(xb, "#C1440E", f"KMC (exact, n={NBKL})"), (xm, "#0F4C81", f"model (n={NCH})")]:
    m, s = arr.mean(0), arr.std(0)
    ax.plot(x, m, "-", color=c, lw=2.5, label=lab)
    ax.fill_between(x, m - s, m + s, color=c, alpha=0.20)
ax.set_xlabel("snapshot (coarsening time)", fontsize=12)
ax.set_ylabel(r"correlation length  $\xi = L/\sqrt{n_c}$", fontsize=12)
ax.set_title(f"Coarsening: model tracks KMC  (T={T}, sample-temp {ST})", fontsize=12)
ax.legend(fontsize=11, loc="upper left"); ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("scratch/xi_clean.png", dpi=150)
print(f"wrote scratch/xi_clean.png  | final xi: model {xm[:,-1].mean():.2f}+/-{xm[:,-1].std():.2f}  KMC {xb[:,-1].mean():.2f}+/-{xb[:,-1].std():.2f}")
