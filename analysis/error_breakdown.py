"""Per-temperature residual / oracle-floor, split into flip-site (position) vs waiting-time (dt) error.

For each temperature, over equilibrium configs, the model's per-token cross-entropy decomposes as
floor (= entropy of the exact KMC distribution) + residual KL. We plot residual/floor separately for
the position tokens and the dt tokens, so it shows (a) how close to optimal per temp and (b) whether
the error lives in the flip-site choice or the waiting time."""
import numpy as np
from haliax.partitioning import set_mesh
from levanter.grug.sharding import compact_grug_mesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from marin_spin.compare_bkl import equilibrium_configs
from marin_spin.probs_vs_bkl import (
    bkl_pos_prob, model_pos_probs, bkl_dt_binned, model_dt_probs, _kl)
from marin_spin.rollout_quench import build_tokenizer, load_model

TEMPS = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5]
NCFG = 16
H5 = "/Users/yaelelmatad/Downloads"


def ent(p, eps=1e-12):
    p = np.asarray(p, float) + eps; p = p / p.sum()
    return float(-(p * np.log(p)).sum())


tok = build_tokenizer()
cfgs = {T: equilibrium_configs(T, NCFG, H5) for T in TEMPS}
pos_floor, pos_res, dt_floor, dt_res = {}, {}, {}, {}
with set_mesh(compact_grug_mesh()):
    model = load_model("scratch/ckpt/step-49400")  # OG hero baseline
    for T in TEMPS:
        cs = list(cfgs[T])
        pb = [bkl_pos_prob(c, T) for c in cs]
        pm = model_pos_probs(model, tok, cs, T)
        db = [bkl_dt_binned(c, T, tok)[0] for c in cs]
        sites = [int(b.argmax()) for b in pb]
        dm = model_dt_probs(model, tok, cs, T, sites)
        pos_floor[T] = np.mean([ent(b) for b in pb]); pos_res[T] = np.mean([_kl(pb[i], pm[i]) for i in range(len(cs))])
        dt_floor[T] = np.mean([ent(b) for b in db]);  dt_res[T] = np.mean([_kl(db[i], dm[i]) for i in range(len(cs))])
        print(f"T={T:.1f}  pos KL/floor={pos_res[T]/pos_floor[T]:.3f}  dt KL/floor={dt_res[T]/dt_floor[T]:.3f}", flush=True)

x = np.array(TEMPS)
fp = np.array([100 * pos_res[T] / pos_floor[T] for T in TEMPS])
fd = np.array([100 * dt_res[T] / dt_floor[T] for T in TEMPS])
np.savez("scratch/data/error_breakdown.npz", temps=x, flip_site_pct=fp, dt_pct=fd,
         pos_floor=np.array([pos_floor[T] for T in TEMPS]), pos_res=np.array([pos_res[T] for T in TEMPS]),
         dt_floor=np.array([dt_floor[T] for T in TEMPS]), dt_res=np.array([dt_res[T] for T in TEMPS]))
fig, ax = plt.subplots(figsize=(7.4, 4.0))
ax.plot(x, fp, "o-", color="#0F4C81", lw=2.5, ms=7, label="flip-site (position)")
ax.plot(x, fd, "s-", color="#C1440E", lw=2.5, ms=7, label=r"waiting time $\Delta t$")
ax.axvspan(2.0, 2.8, color="gray", alpha=0.12)
ax.text(2.4, max(fp.max(), fd.max()) * 0.9, "held out\n($T_c$)",
        ha="center", va="top", fontsize=9, color="gray")
ax.set_xlabel("temperature $T$", fontsize=12)
ax.set_ylabel(r"$D_{\mathrm{KL}}\,/\,H(p_{\mathrm{KMC}})$  (%)", fontsize=12)
ax.set_title("Where is the error? flip-site vs waiting-time, per temperature", fontsize=12)
ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("scratch/error_breakdown.png", dpi=150)
print("wrote scratch/error_breakdown.png")
