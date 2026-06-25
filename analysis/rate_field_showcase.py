"""Rate-field showcase for the poster: deliberate, clean configs (not random trajectory snapshots).
Each row: [spin config] -> [exact BKL p(flip)] -> [model p(flip)], log color scale."""
import numpy as np
from haliax.partitioning import set_mesh
from levanter.grug.sharding import compact_grug_mesh
from matplotlib.colors import BoundaryNorm, ListedColormap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OACMAP = ListedColormap(["#7B4A2A", "#E8C18A"])  # OA-brand 2-spin map: deep brown / light wheat
import poster_style
poster_style.apply()  # Lato + cream block-colored background

from marin_spin.compare_bkl import checkerboard, ac
from marin_spin.probs_vs_bkl import bkl_pos_prob, model_pos_probs, _kl, _tv
from marin_spin.rollout_quench import build_tokenizer, load_model
from marin_spin.tokenize_ising import LATTICE_L

L, T = LATTICE_L, 1.5


def droplet(r):
    i, j = np.indices((L, L)); c = (L - 1) / 2.0
    s = np.ones((L, L), np.int8)
    s[(i - c) ** 2 + (j - c) ** 2 <= r * r] = -1
    return s


def stripe():
    s = np.ones((L, L), np.int8); s[:, : L // 2] = -1
    return s


# one clean, real coarsening config (large domains) from a quench trajectory
cfgs, *_ = ac.bkl_rollout(checkerboard(L), T, 18, 50, np.random.default_rng(3), snapshot_every=1)
real = cfgs[min(750, len(cfgs) - 1)]

configs = [droplet(4), stripe(), real]
labels = ["droplet (curved wall)", "flat domain wall", "coarsening (real)"]

tok = build_tokenizer()
p_bkl = [bkl_pos_prob(c, T) for c in configs]
with set_mesh(compact_grug_mesh()):
    model = load_model("scratch/ckpt/step-49400")  # OG hero baseline
    p_model = model_pos_probs(model, tok, configs, T)

nlev = 14
VMIN, VMAX = 1e-4, 0.20
levels = np.logspace(np.log10(VMIN), np.log10(VMAX), nlev + 1)
cmap = plt.get_cmap("magma", nlev).copy()
cmap.set_under("black"); cmap.set_over(cmap(cmap.N - 1))
norm = BoundaryNorm(levels, cmap.N)

n = len(configs)
fig, axes = plt.subplots(n, 3, figsize=(10.5, 3.4 * n), constrained_layout=True)
im = None
for i, lab in enumerate(labels):
    acfg, a0, a1 = axes[i]
    acfg.imshow(configs[i], cmap=OACMAP, vmin=-1, vmax=1, interpolation="nearest")
    acfg.set_title(lab, fontsize=13); acfg.axis("off")
    im = a0.imshow(p_bkl[i].reshape(L, L), cmap=cmap, norm=norm)
    a0.set_title("KMC  p(flip site)", fontsize=13); a0.axis("off")
    a1.imshow(p_model[i].reshape(L, L), cmap=cmap, norm=norm)
    kl, tv = _kl(p_bkl[i], p_model[i]), _tv(p_bkl[i], p_model[i])
    a1.set_title(f"MarinSpin  p(flip site)\nKL={kl:.3f}  TV={tv:.3f}", fontsize=13); a1.axis("off")
cbar = fig.colorbar(im, ax=axes[:, 1:].ravel().tolist(), fraction=0.04, pad=0.02, extend="both")
cbar.set_label("p(flip site)", fontsize=13); cbar.ax.tick_params(labelsize=11)
fig.savefig("scratch/rate_field_showcase.png", dpi=300)
print("wrote scratch/rate_field_showcase.png")
for lab, c, pb, pm in zip(labels, configs, p_bkl, p_model):
    print(f"{lab:>22}: KL={_kl(pb, pm):.3f}")
