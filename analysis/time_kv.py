import time, numpy as np
from haliax.partitioning import set_mesh
from levanter.grug.sharding import compact_grug_mesh
from marin_spin.compare_bkl import checkerboard
from marin_spin.rollout_quench import build_tokenizer, load_model
from marin_spin.tokenize_ising import LATTICE_L, WINDOW_EVENTS
from marin_spin.kv_decode import cached_rollout
tok = build_tokenizer()
with set_mesh(compact_grug_mesh()):
    model = load_model("scratch/ckpt/step-49400")
    spins0 = np.broadcast_to(checkerboard(LATTICE_L), (8, LATTICE_L, LATTICE_L)).copy()
    t0 = time.time()
    snaps = cached_rollout(model, tok, spins0, 1.5, 1, WINDOW_EVENTS, sample_temp=0.9, rng=np.random.default_rng(0))
    print(f"window 1 (incl compile): {time.time()-t0:.1f}s")
    t0 = time.time()
    snaps = cached_rollout(model, tok, spins0, 1.5, 3, WINDOW_EVENTS, sample_temp=0.9, rng=np.random.default_rng(0))
    print(f"3 more windows (warm): {time.time()-t0:.1f}s -> {(time.time()-t0)/3:.1f}s/window, 8 chains")
    print("final config domain frac:", max((snaps[-1,0]==1).mean(), (snaps[-1,0]==-1).mean()))
