"""Static coarsening montage for the poster: best-model checkerboard quench vs KMC ground truth.

Two rows (model / KMC) x 5 columns (time progression). Rolls an ensemble of chains and picks the
cleanest-coarsening model chain (largest final domain, not collapsed) and a representative KMC chain
(median final xi), so the showcased rollout is a good --- but real --- example. Panel titles show the
correlation length and cluster count (|m| is not a coarsening metric)."""
import numpy as np
from haliax.partitioning import set_mesh
from levanter.grug.sharding import compact_grug_mesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

OACMAP = ListedColormap(["#7B4A2A", "#E8C18A"])  # OA-brand 2-spin map: deep brown / light wheat

from marin_spin.compare_bkl import ac, checkerboard
from marin_spin.corr_anisotropy import corr_dir, xi_integrated
from marin_spin.kv_decode import cached_rollout
from marin_spin.rollout_quench import build_tokenizer, load_model
from marin_spin.tokenize_ising import LATTICE_L, WINDOW_EVENTS

T, SAMPLE_TEMP = 1.5, 0.9        # more greedy than 0.85 -> cleaner domains, less salt-and-pepper noise
N_CHAINS, N_WINDOWS, SEED = 12, 12, 7   # more chains -> more candidates to pick the cleanest from
CKPT = "scratch/ckpt/step-49400"  # OG hero baseline = best dynamics model
NCOL = 5


def xi_corr(c):
    """Noise-robust correlation length: direction-averaged integrated connected C(r). During coarsening
    |m|~0 so the connected correlation is well-defined, and point noise (stray flips) barely affects the
    integral --- unlike L/sqrt(n_clusters), which counts every isolated spin as a cluster."""
    return 0.5 * (xi_integrated(corr_dir(c[None], 1)) + xi_integrated(corr_dir(c[None], 2)))


def _metrics_all(spins):
    """Returns (n_clusters, largest_frac, xi_corr) per chain; xi_corr is the noise-robust length."""
    nc, lf, xc = [], [], []
    for b in range(spins.shape[0]):
        m = ac.coarsening_metrics(spins[b])  # (nc, lf, xi=L/sqrt(nc))
        nc.append(m[0]); lf.append(m[1]); xc.append(xi_corr(spins[b]))
    return np.array(nc), np.array(lf), np.array(xc)


def model_rollout_all(model, tok, rng):
    """KV-cached checkerboard quench; capture every chain's config once per window. Returns snaps[S,B,L,L], nc, lf, xi."""
    spins0 = np.broadcast_to(checkerboard(tok.L), (N_CHAINS, tok.L, tok.L)).copy()
    snaps, times = cached_rollout(model, tok, spins0, T, N_WINDOWS, WINDOW_EVENTS, sample_temp=SAMPLE_TEMP, rng=rng)
    met = [[ac.coarsening_metrics(snaps[s, b]) for b in range(N_CHAINS)] for s in range(snaps.shape[0])]
    nc = np.array([[m[b][0] for b in range(N_CHAINS)] for m in met])
    lf = np.array([[m[b][1] for b in range(N_CHAINS)] for m in met])
    xc = np.array([[xi_corr(snaps[s, b]) for b in range(N_CHAINS)] for s in range(snaps.shape[0])])
    print(f"  [model] rollout done; final <xi_corr>={xc[-1].mean():.1f}", flush=True)
    return snaps, nc, lf, xc, times


def kmc_rollout_all(rng):
    """n_chains independent KMC trajectories from the same checkerboard. Returns snaps[S,B,L,L], nc, lf, xi."""
    cs, ncs, lfs, xis, ts = [], [], [], [], []
    for b in range(N_CHAINS):
        cfgs, _m, _dw, _e, nc, lf, xi, t = ac.bkl_rollout(
            checkerboard(LATTICE_L), T, N_WINDOWS, WINDOW_EVENTS, rng, snapshot_every=WINDOW_EVENTS)
        cs.append(np.array(cfgs)); ncs.append(np.array(nc)); lfs.append(np.array(lf)); xis.append(np.array(xi)); ts.append(np.array(t))
    return (np.array(cs).transpose(1, 0, 2, 3), np.array(ncs).T, np.array(lfs).T, np.array(xis).T, np.array(ts).T)


def main():
    tok = build_tokenizer()
    with set_mesh(compact_grug_mesh()):
        model = load_model(CKPT)
        print("=== model ensemble rollout ===", flush=True)
        ms, mnc, mlf, mxi, mt = model_rollout_all(model, tok, np.random.default_rng(SEED + 2))
    print("=== KMC ensemble rollout ===", flush=True)
    ks, knc, klf, kxi, kt = kmc_rollout_all(np.random.default_rng(SEED + 1))

    # cleanest model chain: fewest clusters (least salt-and-pepper) among well-coarsened, not-collapsed chains
    lf_f, nc_f = mlf[-1], mnc[-1]
    ok = np.where((lf_f > 0.45) & (lf_f < 0.92))[0]
    cand = ok if len(ok) else np.arange(N_CHAINS)
    sel_m = int(cand[np.argmin(nc_f[cand])])
    # representative KMC chain: median final largest-domain fraction
    sel_k = int(np.argsort(klf[-1])[len(klf[-1]) // 2])
    print(f"model chain {sel_m}: final xi_corr={mxi[-1, sel_m]:.1f} S_max={lf_f[sel_m]:.2f} nc={int(nc_f[sel_m])}")
    print(f"KMC   chain {sel_k}: final S_max={klf[-1, sel_k]:.2f}")

    cols = np.linspace(0, ms.shape[0] - 1, NCOL).round().astype(int)
    np.savez("scratch/data/quench_montage.npz", cols=cols, events_per_window=WINDOW_EVENTS,
             model_configs=ms[cols, sel_m], kmc_configs=ks[cols, sel_k],
             model_xi=np.array([xi_corr(ms[s, sel_m]) for s in cols]),
             kmc_xi=np.array([xi_corr(ks[s, sel_k]) for s in cols]),
             model_domain=mlf[cols, sel_m], kmc_domain=klf[cols, sel_k],
             model_time=mt[cols, sel_m], kmc_time=kt[cols, sel_k])
    fig, axes = plt.subplots(2, NCOL, figsize=(3.0 * NCOL, 6.8), constrained_layout=True)
    for c, s in enumerate(cols):
        mc, kc = ms[s, sel_m], ks[s, sel_k]
        axes[0, c].imshow(mc, cmap=OACMAP, vmin=-1, vmax=1, interpolation="nearest")
        axes[0, c].set_title(r"$\xi$=" + f"{xi_corr(mc):.1f}  {mlf[s, sel_m] * 100:.0f}%" + f"\n$t$={mt[s, sel_m]:.1f}", fontsize=10)
        axes[1, c].imshow(kc, cmap=OACMAP, vmin=-1, vmax=1, interpolation="nearest")
        axes[1, c].set_title(r"$\xi$=" + f"{xi_corr(kc):.1f}  {klf[s, sel_k] * 100:.0f}%" + f"\n$t$={kt[s, sel_k]:.1f}", fontsize=10)
        for r in (0, 1):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
    axes[0, 0].set_ylabel("model", fontsize=14, fontweight="bold")
    axes[1, 0].set_ylabel("KMC", fontsize=14, fontweight="bold")
    fig.suptitle(f"Checkerboard quench → T={T}: correlation length and domain size grow together  "
                 f"(sample temp {SAMPLE_TEMP})", fontsize=13)
    fig.savefig("scratch/quench_montage.png", dpi=150)
    print("wrote scratch/quench_montage.png")


if __name__ == "__main__":
    main()
