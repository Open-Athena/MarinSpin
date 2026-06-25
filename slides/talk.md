---
theme: default
title: "Learning Ising Kinetics with a Transformer"
highlighter: shiki
lineNumbers: false
drawings:
  persist: false
transition: slide-left
mdc: true
---

# Learning Ising Kinetics with a Transformer

**Can a language model learn the rules of statistical mechanics?**

<br>

Next-token prediction on spin flip events

---

# The Ising Model

A canonical model of magnetism — spins on a 2D square lattice, each ±1

$$H = -J \sum_{\langle ij \rangle} s_i s_j \qquad (J=1,\ k_B=1)$$

<br>

**Two phases separated by a phase transition at $T_c \approx 2.269$:**

|  | Ordered ($T < T_c$) | Disordered ($T > T_c$) |
|---|---|---|
| Spins | Mostly aligned | Random |
| Magnetization | $\|m\| \approx 1$ | $\|m\| \approx 0$ |
| Dynamics | Rare boundary flips | Frequent, spatially uniform |

<br>

**Why interesting:** exactly solvable, rich physics, clear ground truth for model evaluation

---

# The Goal

<br>

> **Train a transformer to generate statistically valid Ising spin flip trajectories, conditioned on temperature.**

<br>

**Concretely:**
- Given: current spin configuration + temperature
- Predict: which spin flips next, and when

**Why NTP?**
- No physics-specific architecture — pure sequence modeling
- If it works, the model has implicitly learned the Metropolis rates
- Testable against exact ground truth (BKL algorithm)

---

# Generating Ground Truth: BKL Algorithm

**Bortz-Kalos-Lebowitz** — rejection-free kinetic Monte Carlo

At each step, every spin $i$ has a flip rate:

$$w_i = \min\!\left(1,\ e^{-\Delta E_i / T}\right) \qquad \text{(Metropolis)}$$

Total rate $R = \sum_i w_i$. Then:

1. Draw time increment $\Delta t \sim \text{Exp}(R)$
2. Select spin $i$ with probability $w_i / R$
3. Flip spin $i$, update rates for $i$ and its 4 neighbors

**No rejected moves** → continuous-time event sequence: a list of (position, $\Delta t$) pairs

<br>

Dataset: **42,000 trajectories** × 500 events at 3 sizes (L=16, 32, 64) × 14 temperatures

---

# The Grammar

Each trajectory window of $W=50$ events becomes one training sequence:

```
[T_bin]
[pos_0][spin_0] [pos_1][spin_1] ... [pos_255][spin_255]   ← spin config (copy 1)
[pos_0][spin_0] [pos_1][spin_1] ... [pos_255][spin_255]   ← spin config (copy 2)
[T_bin][pos_k][dt_k]  [T_bin][pos_{k+1}][dt_{k+1}]  ...  ← 50 events
```

**Vocabulary:**
- `T_bin` — one token per training temperature (14 total)
- `spin` — 2 tokens (↑, ↓)
- `pos` — 256 tokens (flat index of the flipped spin)
- `dt` — 66 tokens (log-uniform time bins)

Total: **~342 tokens** in vocabulary, **1175 tokens** per sequence (L=16, W=50)

**Loss computed only on `pos` and `dt` tokens** — config and temperature are pure context

---

# Why This Grammar?

<br>

**Config duplication:** copy 1 is fully in causal context before the first event token, so the model can attend to the full spin state before predicting anything

**T_bin before every event:** temperature signal is always ≤2 positions from any event token — not 1000+ tokens away

**Windowing:** BKL rates depend only on *current* neighbor spins — re-injecting the config every 50 events is physically correct and gives 10× more training windows per trajectory

<br>

**What the model must learn:**

Given the spin config at step $k$ and temperature $T$, predict the Metropolis rate distribution — i.e., which sites are on domain boundaries and how energetically costly each flip is

---

# Architecture & Training

**Model:** decoder-only causal transformer (vanilla GPT-style)

| | |
|---|---|
| Parameters | 4.8M |
| Layers | 6 |
| Heads | 6 |
| d_model | 384 |
| Context | 1175 tokens |

<br>

**Training:**
- AdamW, cosine LR schedule, teacher forcing
- 80/10/10 train/val/test split (deterministic, seed=0)
- L=16 only (one size for now)
- ~37 epochs so far, still training

---

# How Close to Optimal?

The **oracle model** is BKL itself — it knows the exact Metropolis rates at every step.

$$p_\text{oracle}(\text{pos} = j) = \frac{w_j}{R} \qquad p_\text{oracle}(\Delta t \in \text{bin } k) = e^{-R \cdot t_\text{lo}} - e^{-R \cdot t_\text{hi}}$$

<br>

| | NLL (nats) |
|---|---|
| Oracle floor | 3.9609 |
| Our model (epoch 35) | 3.9729 |
| **Gap** | **0.012 nats (0.30%)** |

<br>

**Where is the gap?**
- **96% in `pos` tokens** — which spin flips
- **4% in `dt` tokens** — timing is essentially solved

The model has learned the *when* but still has room on the *where*

---

# NLL Discriminator: Can You Tell Them Apart?

Use the model's own NLL as a two-sample test: AUC = P(NLL$_\text{BKL}$ > NLL$_\text{TR}$)

AUC = 0.5 → indistinguishable. AUC > 0.5 → model assigns higher probability to its own outputs.

<br>

| Phase | T range | AUC | Interpretation |
|---|---|---|---|
| Ordered | 1.5–1.7 | **0.65–0.70** | Model too sharp — exposure bias |
| Ordered | 1.8–2.0 | 0.50–0.56 | Nearly indistinguishable |
| Disordered | 2.8–3.5 | 0.44–0.53 | Indistinguishable ✓ |

<br>

**Ordered phase:** teacher-forced training at low T is too easy (near-frozen spins) → model learns a slightly tighter distribution than the true one. Accumulates during rollout.

**Disordered phase:** matched well — high entropy makes training harder but the model has kept up.

---

# Physical Observables

Running transformer rollout at each training temperature and comparing to BKL:

| Observable | Ordered phase | Disordered phase |
|---|---|---|
| $\|m\|$ | ✓ within ~1% | ✓ roughly correct |
| $\xi$ (correlation length) | ~10% too small | ~15% too small |
| $E/N$ (energy/spin) | ✓ within 3–4% | ~20% too hot |

<br>

**Interpretation:** transformer generates slightly too many small clusters (ξ low) and slightly too energetic configurations at high T. Stable across checkpoints — intrinsic bias, not a training depth issue.

**Likely cause:** the 96% pos-token gap. Getting which spin flips slightly wrong systematically biases the spatial structure.

---

# Where Next?

**The spatial routing problem:** the model must attend from an event token back to its 4 neighbors scattered through the 512-token config section. Current raster-scan order provides no locality.

**Proposed tokenization redesign (future work):**

```
[pos_0][pos_1]  [pos_0][pos_16]  ...  ← all 512 lattice edges (fixed, 1024 tokens)
[config_start] [pos_5] [pos_21] ... [config_end]   ← up-spin positions only
[T_bin][pos_7][dt_3]  [T_bin][pos_21][dt_8]  ...   ← events (unchanged)
```

- Edge list: neighbor relationships injected explicitly, once, reusing existing pos tokens
- Sparse config: encodes only up-spin positions (~128–230 tokens vs 1024 currently)
- No new architecture — standard causal mask, same vocabulary
- Worst-case context: 1432 tokens (+22% vs current 1175)

**Goal:** eliminate the spatial routing burden so the 0.012 nat gap closes

---

# Summary

<br>

- Trained a **vanilla decoder-only transformer** on Ising kinetic Monte Carlo trajectories
- Tokenized as next-event prediction: given spin config + T, predict (pos, Δt)
- **0.30% above the thermodynamic oracle floor** after ~37 epochs
- **Timing essentially solved** (4% of gap); **spatial routing** is the remaining challenge
- Model **matches BKL in the disordered phase** (AUC ≈ 0.5); detectable exposure bias in ordered phase
- Physical observables (|m|, ξ, E/N) broadly correct, with systematic ~10–20% biases in cluster structure

<br>

**Open question:** can explicit neighbor injection in the grammar close the spatial gap without any architecture changes?
