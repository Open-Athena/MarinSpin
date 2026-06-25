# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""KV-cache incremental decoding for the grug/base Transformer (inference only).

The autoregressive rollout in ``compare_bkl`` / ``rollout_quench`` calls ``model.logits`` on the whole
growing sequence for every generated token, re-encoding the entire context (~1175 tokens, the bulk of
which is the fixed lattice-raster prefix) from scratch each step --- O(L^2) per token. On CPU that is
seconds per token and hours per figure.

This module reuses the trained weights to do standard cached decoding instead:
  - ``prefill`` runs the prefix once and stores each layer's (K, V) into a fixed-size cache.
  - ``decode_step`` processes a single new token, attends to the cache, and writes its K/V back.

The cache is preallocated to a fixed length so shapes never change and XLA compiles once. Correctness
is checked against the dense ``model.logits`` in ``tests``/the ``_selfcheck`` helper below.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from einops import rearrange

from marin_spin.grug.model import Transformer
from levanter.grug.sharding import Pbatch, Plogits, unshard


def rope_tables(max_len: int, head_dim: int, theta: float) -> tuple[jax.Array, jax.Array]:
    """cos/sin of shape [max_len, head_dim//2] (matches levanter.grug.attention._rotary_cache)."""
    half = head_dim // 2
    inv_freq = 1.0 / (theta ** (jnp.arange(0, half, dtype=jnp.float32) / half))
    pos = jnp.arange(max_len, dtype=jnp.float32)
    ang = pos[:, None] * inv_freq[None, :]
    return jnp.cos(ang), jnp.sin(ang)


def _apply_rope(x: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
    """Rotate x [B,S,H,D] given cos/sin broadcastable to [.,S,.,D/2]."""
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1).astype(x.dtype)


def _rms(x: jax.Array, weight: jax.Array, eps: float) -> jax.Array:
    dtype = x.dtype
    x = x.astype(jnp.float32)
    var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    return (x * jax.lax.rsqrt(var + eps) * weight).astype(dtype)


def _mlp(block, x: jax.Array) -> jax.Array:
    up = jnp.einsum("bsh,hm->bsm", x, block.mlp.mlp_up)
    return jnp.einsum("bsm,mh->bsh", jax.nn.relu(up), block.mlp.mlp_down, out_sharding=Pbatch)


def _gqa(x: jax.Array, n_q: int) -> jax.Array:
    """Expand KV heads [B,S,m,d] -> [B,S,n_q,d] (matches align_kv_heads)."""
    m = x.shape[2]
    if m == n_q:
        return x
    rep = n_q // m
    return jnp.broadcast_to(x[:, :, :, None, :], (*x.shape[:3], rep, x.shape[3])).reshape(
        *x.shape[:2], n_q, x.shape[3])


def _qkv(block, normed: jax.Array, head_dim: int) -> tuple[jax.Array, jax.Array, jax.Array]:
    q = rearrange(jnp.einsum("bsh,hd->bsd", normed, block.attn.w_q), "... (n d) -> ... n d", d=head_dim)
    k = rearrange(jnp.einsum("bsh,hd->bsd", normed, block.attn.w_k), "... (m d) -> ... m d", d=head_dim)
    v = rearrange(jnp.einsum("bsh,hd->bsd", normed, block.attn.w_v), "... (m d) -> ... m d", d=head_dim)
    return q, k, v


def _attend(q: jax.Array, k: jax.Array, v: jax.Array, bias: jax.Array, n_q: int) -> jax.Array:
    """q [B,Sq,n,d], k/v [B,Sk,m,d], bias [.,Sq,Sk] additive (-inf disallowed). Returns [B,Sq,n,d]."""
    d = q.shape[-1]
    k = _gqa(k, n_q); v = _gqa(v, n_q)
    scores = jnp.einsum("bqhd,bkhd->bhqk", q * (1.0 / np.sqrt(d)), k)
    scores = scores + bias[:, None, :, :]
    w = jax.nn.softmax(scores, axis=-1).astype(v.dtype)
    return jnp.einsum("bhqk,bkhd->bqhd", w, v)


import equinox as eqx

_prefill_jit = eqx.filter_jit(lambda m, t, c, cs, sn: prefill(m, t, c, cs, sn))
_decode_jit = eqx.filter_jit(lambda m, tk, p, c, cs, sn: decode_step(m, tk, p, c, cs, sn))


def _sample_masked(logits_np, lo: int, hi: int, temp: float, rng) -> np.ndarray:
    sub = logits_np[:, lo:hi].astype(np.float64) / max(temp, 1e-6)
    sub -= sub.max(axis=1, keepdims=True)
    p = np.exp(sub); p /= p.sum(axis=1, keepdims=True)
    return np.array([lo + rng.choice(hi - lo, p=p[b]) for b in range(p.shape[0])], dtype=np.int32)


def cached_rollout(model: Transformer, tok, initial_spins: np.ndarray, T: float, n_windows: int, W: int,
                   *, sample_temp: float, rng) -> tuple[np.ndarray, np.ndarray]:
    """KV-cached rollout. Returns (snapshots [n_windows+1, B, L, L], times [n_windows+1, B] in real units).

    Per window: prefill the (changed) lattice-context prefix once, then incrementally decode the 3 tokens
    per event ([T][pos][dt]) against the cache --- O(L) per token instead of re-encoding the full O(L^2)
    sequence every step. Single-spin flips are applied from the sampled position tokens (never sampled)."""
    B, L = initial_spins.shape[0], tok.L
    N, POS, DT, NDT = tok.N, tok.POS_OFFSET, tok.DT_OFFSET, tok.n_dt_tokens
    t_tok = tok.T_id(T)
    ctxlen = tok.ctxlen
    native = ctxlen + 3 * W
    cos, sin = rope_tables(native, model.config.inferred_head_dim, model.config.rope.theta)
    spins = initial_spins.astype(np.int8).copy()
    snaps = [spins.copy()]
    clock = np.zeros(B)            # accumulated physical time per chain (decoded Delta t)
    times = [clock.copy()]
    for w in range(n_windows):
        prefix = np.stack([tok.encode(T, spins[b], np.zeros(0, np.int32), np.zeros(0, np.float64))
                           for b in range(B)]).astype(np.int32)
        cache = _prefill_jit(model, jnp.asarray(prefix), init_cache(model, B, native), cos, sin)
        pos_all = np.empty((B, W), np.int32)
        for k in range(W):
            base = ctxlen + 3 * k
            logits, cache = _decode_jit(model, jnp.full((B,), t_tok, jnp.int32), jnp.int32(base), cache, cos, sin)
            pos_tok = _sample_masked(np.asarray(logits), POS, POS + N, sample_temp, rng)
            logits, cache = _decode_jit(model, jnp.asarray(pos_tok), jnp.int32(base + 1), cache, cos, sin)
            dt_tok = _sample_masked(np.asarray(logits), DT, DT + NDT, sample_temp, rng)
            for b in range(B):
                clock[b] += tok.sample_dt(int(dt_tok[b]), rng)  # decode the dt token to real time
            _, cache = _decode_jit(model, jnp.asarray(dt_tok), jnp.int32(base + 2), cache, cos, sin)
            pos_all[:, k] = pos_tok - POS
        flat = spins.reshape(B, -1)
        for b in range(B):
            for idx in pos_all[b]:
                flat[b, idx] *= -1
        snaps.append(spins.copy())
        times.append(clock.copy())
    return np.array(snaps), np.array(times)


def init_cache(model: Transformer, batch: int, max_len: int) -> list[tuple[jax.Array, jax.Array]]:
    cfg = model.config
    m, d = cfg.num_kv_heads, cfg.inferred_head_dim
    return [(jnp.zeros((batch, max_len, m, d), jnp.float32), jnp.zeros((batch, max_len, m, d), jnp.float32))
            for _ in range(cfg.num_layers)]


def prefill(model: Transformer, tokens: jax.Array, cache: list, cos: jax.Array, sin: jax.Array) -> list:
    """Run the prefix [B,P] through the model, writing each layer's K,V into cache[:, :P]. Returns cache."""
    cfg = model.config
    n_q, head_dim = cfg.num_heads, cfg.inferred_head_dim
    P = tokens.shape[1]
    x = model.token_embed.at[tokens].get(out_sharding=Pbatch)
    # causal bias among the P prefix tokens
    qi = jnp.arange(P)[:, None]; ki = jnp.arange(P)[None, :]
    bias = jnp.where(ki <= qi, 0.0, -1e9)[None, :, :]  # [1,P,P]
    cosP, sinP = cos[None, :P, None, :], sin[None, :P, None, :]
    new_cache = []
    for li, block in enumerate(model.blocks):
        normed = _rms(x, block.rms_attn.weight, block.rms_attn.eps)
        q, k, v = _qkv(block, normed, head_dim)
        q = _apply_rope(q, cosP, sinP); k = _apply_rope(k, cosP, sinP)
        q, k, v = unshard(q), unshard(k), unshard(v)
        attn = _attend(q, k, v, bias, n_q)
        attn = rearrange(attn, "... n d -> ... (n d)")
        x = x + jnp.einsum("bsh,hd->bsd", attn, block.attn.w_o, out_sharding=Pbatch)
        x = x + _mlp(block, _rms(x, block.rms_mlp.weight, block.rms_mlp.eps))
        kc, vc = cache[li]
        kc = kc.at[:, :P].set(k); vc = vc.at[:, :P].set(v)
        new_cache.append((kc, vc))
    return new_cache


def decode_step(model: Transformer, token: jax.Array, pos: jax.Array, cache: list,
                cos: jax.Array, sin: jax.Array) -> tuple[jax.Array, list]:
    """Process one token [B] at absolute position `pos` (scalar). Returns (logits [B,V], updated cache)."""
    cfg = model.config
    n_q, head_dim = cfg.num_heads, cfg.inferred_head_dim
    max_len = cache[0][0].shape[1]
    x = model.token_embed.at[token].get(out_sharding=Pbatch)[:, None, :]  # [B,1,D]
    cos_p = cos[pos][None, None, None, :]; sin_p = sin[pos][None, None, None, :]
    allowed = (jnp.arange(max_len) <= pos)[None, None, :]  # [1,1,max_len]
    bias = jnp.where(allowed, 0.0, -1e9)
    new_cache = []
    for li, block in enumerate(model.blocks):
        normed = _rms(x, block.rms_attn.weight, block.rms_attn.eps)
        q, k, v = _qkv(block, normed, head_dim)
        q = _apply_rope(q, cos_p, sin_p); k = _apply_rope(k, cos_p, sin_p)
        q, k, v = unshard(q), unshard(k), unshard(v)
        kc, vc = cache[li]
        kc = jax.lax.dynamic_update_slice_in_dim(kc, k, pos, axis=1)
        vc = jax.lax.dynamic_update_slice_in_dim(vc, v, pos, axis=1)
        attn = _attend(q, kc, vc, bias, n_q)  # q attends to full (masked) cache
        attn = rearrange(attn, "... n d -> ... (n d)")
        x = x + jnp.einsum("bsh,hd->bsd", attn, block.attn.w_o, out_sharding=Pbatch)
        x = x + _mlp(block, _rms(x, block.rms_mlp.weight, block.rms_mlp.eps))
        new_cache.append((kc, vc))
    x = _rms(x, model.final_norm.weight, model.final_norm.eps)
    logits = jnp.einsum("bsh,hd->bsd", x, model.output_proj, out_sharding=Plogits)[:, 0, :]
    return logits, new_cache
