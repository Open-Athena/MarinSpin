# Data schema — what the viewer expects

> **These files are a compiled artifact, not the source of truth.** They are
> built from the Parquet star-schema warehouse by `scripts/build-data.mjs`
> (`npm run build:data`). To add or change a dataset, edit the warehouse — see
> [`data/warehouse/README.md`](../../data/warehouse/README.md) — not these files.
> This document is the contract the build step must produce.

The Residual Stream Viewer is a pure function of the files in this directory.
It generates **no** data itself: `src/data/source.js` fetches these JSON files
and hands the UI a `DataSource`.

All vectors are plain JSON arrays of numbers. "length `dModel`" means the array
must have exactly `dModel` entries. Numbers may be rounded for size; the UI does
not require full precision.

---

## File layout

```
public/data/
  models.json                                              # model list + default (loaded once)
  <model_id>/
    index.json                                             # per-model config + vocab + unembed + sentences + checkpoints
    checkpoints/
      <c>/                                                 # training checkpoint (0..nCheckpoints-1)
        sentences/
          <id>/
            trace-0.json                                   # one ModelTrace per token position
            ...
            trace-<T-1>.json                               # T === that sentence's token count
```

`loadModels()` reads `models.json`; `loadIndex(modelId)` reads that model's
`index.json`. When a (checkpoint, sentence, token) is in view,
`makeSource(index, modelId, checkpoint, sentenceId)` fetches the one
`<model>/checkpoints/<c>/sentences/<id>/trace-<i>.json` **lazily** and caches it
(the cache is shared, so scrubbing checkpoints/tokens is instant). There must be
one trace file per token position, per checkpoint, per sentence, per model.

---

## `models.json`

```ts
type Models = {
  models: {
    id: string;          // model_id (folder name under public/data)
    label: string;       // shown in the model-size picker, e.g. "d_model 512"
    dModel: number;
    nLayers: number;
    nCheckpoints: number;
  }[];
  default: string;       // model_id selected on first load
};
```

## `<model_id>/index.json`

```ts
type Index = {
  seed: number;          // integer; seeds the procedural MLP weights (see "Weights" below)
  dModel: number;        // residual width — length of every activation/write/unembed vector
  nLayers: number;       // transformer layers; each contributes one attn + one mlp stage
  nHeads: number;        // attention heads per attn stage
  nExperts: number;      // MoE router pool size (== moe.poolSize in traces)
  nActive: number;       // routed experts selected per token (top-k; == moe.routed.length)

  vocab: string[];       // candidate next-token vocabulary (shared across sentences/checkpoints)

  // the selectable sentences, in menu order
  sentences: {
    id: string;          // folder name (e.g. "s03")
    label: string;       // short preview shown in the picker
    tokens: string[];    // the input tokens, in order; length === number of trace files
  }[];

  // training checkpoints, in training order (drives the checkpoint slider)
  checkpoints: {
    id: number;          // folder name under checkpoints/ (0..nCheckpoints-1)
    label: string;       // "init" / "50%" / "final"
    trainFrac: number;   // 0 = init, 1 = final

    // lm_head row per vocabulary token for THIS checkpoint, used for the
    // "lm_head basis" projection. Must contain an entry for every token in
    // `vocab` AND every token that appears in any sentence (so any token is
    // projectable). Each vector has length dModel. Stored per checkpoint so
    // the projection of `stages[-1].activation` (post-final-norm) onto
    // `unembed[token]` reflects this checkpoint's actual logit direction.
    unembed: Record<string, number[]>;
  }[];
};
```

---

## `<model>/checkpoints/<c>/sentences/<id>/trace-<i>.json` — `ModelTrace`

One per token position `i` (`0 … tokens.length-1`) of sentence `<id>`, at
checkpoint `<c>`. The shape is identical across checkpoints and models (only the
numbers differ — and `dModel`).

```ts
type ModelTrace = {
  dModel: number;        // must equal index.dModel
  tokenIdx: number;      // == i (the position this trace describes)
  token: string;         // == this sentence's tokens[i]

  stages: Stage[];       // length === 2 + 2*nLayers; see ordering below
  predictions: Pred[];   // next-token distribution over `vocab`, sorted desc by prob
  correctNext: string|null;  // gold next token; null at the last position
};
```

### `stages` ordering

`stages[0]` is the token embedding and `stages[1]` is the RMS norm applied after
it. After that, each layer `L` contributes two stages in order: its attention
stage, then its MLP stage. The final stage is the post-final-RMS-norm
residual — projecting *that* onto `unembed[token]` reproduces the
checkpoint's actual logit direction (subject to per-checkpoint `unembed`). So
the array is:

```
[ embed, rmsnorm, attn(L0), mlp(L0), attn(L1), mlp(L1), …, attn(L_{n-1}), mlp(L_{n-1}), final_rmsnorm ]
```

length `3 + 2*nLayers` (e.g. 6 layers → 15 stages). The first and last entries
both have `kind: "norm"`; the viewer distinguishes them by `short` (`"rms"` vs
`"fin"`) and by `label` (`"RMS norm"` vs `"Final RMS norm"`).

### `Stage`

```ts
type Stage = {
  kind: "embed" | "norm" | "attn" | "mlp";
  layer: number | null;       // 0-based layer index; null for the embed and norm stages
  label: string;              // long label, e.g. "Layer 2 · Attention"
  short: string;              // compact tag, e.g. "emb" / "rms" / "2a" / "2m"

  activation: number[];       // length dModel — the residual stream AFTER this stage
  write: number[] | null;     // length dModel — what this stage ADDED; null for embed
                              //   (for the norm stage this is the Δ from normalization)

  heads?: Head[];             // present iff kind === "attn"
  moe?:   Moe;                // present iff kind === "mlp"
};
```

### `Head` (attention stages)

```ts
type Head = {
  write: number[];     // length dModel — this head's contribution to the residual
  pattern: number[];   // attention weights over key positions 0..tokenIdx
                       //   length === tokenIdx + 1, entries >= 0, sums to 1
};
```

### `Moe` + `Expert` (MLP stages)

```ts
type Moe = {
  shared: Expert;      // always-on shared expert (gate fixed at 1)
  routed: Expert[];    // the top-k routed experts for this token (length === nActive)
  poolSize: number;    // total router pool size (== index.nExperts)
  dFf: number;         // routed expert hidden width (d_model / 2). NOTE: the shared
                       //   expert is wider — its acts length is d_model. `acts.length`
                       //   is authoritative per expert.
};

type Expert = {
  write: number[];     // length dModel — this expert's contribution to the residual
  acts: number[];      // per-neuron activations (sparse, relu-ish, >= 0).
                       //   length === d_model for the shared expert, d_model/2 for routed
  gate: number;        // routing weight; the shared expert uses gate === 1
  eid: number;         // expert id; convention: shared === 99, routed === its pool index
  id?: number;         // routed experts only: the pool index shown in the UI (== eid)

  // the few neurons (default 5) that contribute most to `write`, each with its
  // real w_out direction. Powers the neuron drill-down WITHOUT shipping the full
  // weight matrix. Rank by |gate * acts[i]| * ‖wOut‖ (descending). Optional, but
  // required for a faithful neuron view — omit it and the drill-down is empty.
  topNeurons?: {
    i: number;         // index into this expert's `acts`
    wOut: number[];    // length dModel — that neuron's w_out column (output direction)
  }[];
};
```

### `Pred` (next-token predictions)

```ts
type Pred = {
  token: string;       // a token from `vocab` (or correctNext)
  prob: number;        // probability in [0,1]; the full list sums to 1
  rank: number;        // 1-based rank by prob (1 === most likely)
};
```

`predictions` must be sorted by descending `prob`, with `rank` assigned `1..n`.

---

## Invariants the UI relies on

These are not validated at runtime, but the visuals assume them:

- **Residual accumulation:** `stages[k].activation ≈ stages[k-1].activation + stages[k].write`
  for every `k ≥ 1`.
- **Attention write:** `attn.write ≈ Σ heads[h].write`.
- **MLP write:** `mlp.write ≈ moe.shared.write + Σ moe.routed[r].write`.
- **Per-expert write:** `expert.write ≈ Σ_i gate * acts[i] * w_out_i` — i.e. the
  expert's output is the gated sum of its neuron directions. For each neuron `n` in
  `topNeurons`, `n.wOut * acts[n.i] * gate` is its contribution.
- **Pattern normalization:** each `head.pattern` is a probability distribution over
  key positions `0..tokenIdx` (length `tokenIdx+1`, sums to 1).
- **Prediction normalization:** `Σ predictions[*].prob === 1`, sorted desc, ranks `1..n`.

---

## Neuron directions (`topNeurons`)

The neuron drill-down needs each neuron's **output direction** (its `w_out`
column, length `dModel`). The full weight matrix (`dFf × dModel` per expert, per
layer, per checkpoint) is far too large to ship — so each expert carries only the
**top-K neurons by contribution** (`|gate * acts[i]| * ‖w_out‖`), each with its
real `wOut` vector, in `Expert.topNeurons` (above). The UI ranks/inspects only
those K; this is faithful for the dominant neurons but cannot surface a neuron
outside the top-K (raise K if you need more headroom).

`wOut` is **token-independent** (it's a weight), so a real pipeline may dedupe it
into a per-(checkpoint, layer, expert, neuron) table and reference by index. The
reference mock inlines it per token because its synthetic directions happen to
depend on position. Either way the JSON contract above is what the UI reads.
