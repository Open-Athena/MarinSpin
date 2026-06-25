import numpy as np
from haliax.partitioning import set_mesh
from levanter.grug.sharding import compact_grug_mesh
from marin_spin.compare_bkl import equilibrium_configs, checkerboard, ac
from marin_spin.probs_vs_bkl import bkl_pos_prob, model_pos_probs, _kl
from marin_spin.rollout_quench import build_tokenizer, load_model
from marin_spin.tokenize_ising import LATTICE_L


def entropy(p, eps=1e-12):
    p = np.asarray(p, float) + eps
    p = p / p.sum()
    return float(-(p * np.log(p)).sum())


tok = build_tokenizer()
L = LATTICE_L
T = 1.5

# config sets: (a) equilibrium = in-distribution; (b) coarsening = OOD configs the rollout traverses
eq = equilibrium_configs(T, 64, "/Users/yaelelmatad/Downloads")
co200, co600 = [], []
for s in range(32):
    cfgs, *_ = ac.bkl_rollout(checkerboard(L), T, 13, 50, np.random.default_rng(1000 + s), snapshot_every=1)
    co200.append(cfgs[min(200, len(cfgs) - 1)])
    co600.append(cfgs[min(600, len(cfgs) - 1)])
sets = {"equilibrium": np.array(eq), "coarsen-200ev": np.array(co200), "coarsen-600ev": np.array(co600)}

bkl = {}
for nm, cfgs in sets.items():
    pb = [bkl_pos_prob(c, T) for c in cfgs]
    bkl[nm] = (float(np.mean([entropy(p) for p in pb])), pb)
print("BKL entropy (nats):", {nm: round(bkl[nm][0], 3) for nm in sets}, flush=True)
print("(higher entropy = blurrier/less sharp; peak<1 = model peak below BKL = blurrier)\n", flush=True)

ckpts = [("baseline-49k", "scratch/ckpt/step-49400"),
         ("aug-49k", "scratch/ckpt-aug/step-49400"),
         ("aug-100k", "scratch/ckpt-aug-long/step-100000")]
with set_mesh(compact_grug_mesh()):
    for name, ck in ckpts:
        m = load_model(ck)
        cells = []
        for nm, cfgs in sets.items():
            pm = model_pos_probs(m, tok, list(cfgs), T)
            pb = bkl[nm][1]
            Hm = float(np.mean([entropy(p) for p in pm]))
            dH = Hm - bkl[nm][0]
            peak = float(np.mean([pm[i].max() / pb[i].max() for i in range(len(cfgs))]))
            kl = float(np.mean([_kl(pb[i], pm[i]) for i in range(len(cfgs))]))
            cells.append(f"{nm}: H={Hm:.2f} dH={dH:+.2f} peak={peak:.2f} KL={kl:.3f}")
        print(f"{name:>13} | " + "  ||  ".join(cells), flush=True)
