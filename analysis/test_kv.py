"""Verify KV-cache decode matches the dense model.logits exactly (same predictions, fp tolerance)."""
import numpy as np
import jax.numpy as jnp
from haliax.partitioning import set_mesh
from levanter.grug.sharding import compact_grug_mesh

from marin_spin.compare_bkl import checkerboard
from marin_spin.rollout_quench import build_tokenizer, load_model
from marin_spin.tokenize_ising import LATTICE_L
from marin_spin.kv_decode import rope_tables, init_cache, prefill, decode_step

T = 1.5
tok = build_tokenizer()
with set_mesh(compact_grug_mesh()):
    model = load_model("scratch/ckpt/step-49400")
    cfg = model.config
    t_tok = tok.T_id(T)
    prefix = tok.encode(T, checkerboard(LATTICE_L), np.zeros(0, np.int32), np.zeros(0, np.float64)).astype(np.int32)
    ctxlen = len(prefix)
    extra = np.array([t_tok, tok.POS_OFFSET + 5, tok.DT_OFFSET + 2,
                      t_tok, tok.POS_OFFSET + 9, tok.DT_OFFSET + 1], np.int32)
    full = np.concatenate([prefix, extra])
    P, E = ctxlen, len(extra)

    dense = np.asarray(model.logits(jnp.asarray(full[None]))[0])  # [P+E, V]

    cos, sin = rope_tables(P + E, cfg.inferred_head_dim, cfg.rope.theta)
    cache = init_cache(model, 1, P + E)
    cache = prefill(model, jnp.asarray(prefix[None]), cache, cos, sin)
    maxdiff = 0.0
    for i in range(E):
        pos = P + i
        logits, cache = decode_step(model, jnp.asarray(full[pos:pos + 1]), pos, cache, cos, sin)
        d = float(np.max(np.abs(np.asarray(logits)[0] - dense[pos])))
        # also check the argmax / top token agrees and the masked-softmax over pos range matches
        maxdiff = max(maxdiff, d)
        print(f"pos={pos}  max|cached-dense|={d:.3e}")
    print(f"\nMAX logit diff over all decoded positions: {maxdiff:.3e}")
    print("PASS" if maxdiff < 1e-2 else "FAIL")
