> **Repo note.** Verbatim working record from the rig workspace
> (`/home/deni/Claude/bonsai_prep_2026_07_20/README.md`), preserved unedited as a citation
> target. Reader-facing synthesis: [`BONSAI_2026_07.md`](BONSAI_2026_07.md).

---

# Bonsai on the PIM rig — host-side preparation (task B3, 2026-07-20)

Everything in this directory was produced CPU-only. The FPGA was never
touched. Weights live in `/home/deni/bonsai_weights/`; a dedicated venv is
at `/home/deni/bonsai_venv` (the pinned BitNet env — SYSTEM python user-site,
torch 2.7.1+cpu / transformers 4.52.0 — was not modified).

---

## 1. Model facts (with citations)

### 1.1 IMPORTANT correction: two unrelated "Bonsai" families exist

* **PrismML Bonsai (the target of this task)** — a family of **1-bit and
  ternary extreme quantizations of Qwen3 models**, published by Prism ML, Inc.
  (HF org `prism-ml`, site prismml.com, GitHub `PrismML-Eng`) starting
  April 2026, flagship 27B released 2026-07-14. This matches the user's
  "PrismML, 1 bit and ternary" exactly.
* **deepgrove/Bonsai** — an older, unrelated 0.5B ternary LLM
  (Llama-architecture, Mistral tokenizer, trained-from-scratch on 3.8B tokens,
  March 2025, technical report in its GitHub repo; per-CHANNEL scales).
  It merely shares the name. The task hint "~0.5B ternary LLM, arXiv paper
  likely" describes this one — it is NOT what PrismML publishes.
  No arXiv id was found for either family; PrismML publishes whitepaper PDFs
  in its GitHub repo, deepgrove a technical report in theirs.

### 1.2 PrismML Bonsai family (July 2026 state)

33 HF repos under `prism-ml`. Text-model sizes: **1.7B, 4B, 8B, 27B**, each in
two weight variants and three packagings (plus 4B image-diffusion models):

| Variant | Weights | GGUF type | HF naming |
|---|---|---|---|
| Bonsai (1-bit) | {−1, +1}, no zeros | `Q1_0` | `Bonsai-<size>-{gguf,mlx-1bit,unpacked}` |
| Ternary Bonsai (1.58-bit) | {−1, 0, +1} | `Q2_0` | `Ternary-Bonsai-<size>-{gguf,mlx-2bit,unpacked}` |

* Base models: Qwen3-1.7B / Qwen3-4B / Qwen3-8B (ternary whitepaper Table 1);
  27B is built from "Qwen3.6-27B" (July 2026 release, post-dates this rig's
  knowledge of Qwen releases).
* Weight format (both variants): **w_i = s_g · t_i with one shared FP16 scale
  per group of 128 consecutive input-dim weights** ("Ternary g128"; the 1-bit
  release uses the same group-wise format with t ∈ {−1,+1}).
* Quantized layers: **embeddings, all attention projections, all MLP
  projections, and the LM head** — "no higher-precision escape hatches".
  Norms (RMSNorm weights, q_norm/k_norm) and scales stay FP16.
* Quantization method: proprietary ("mathematically grounded", "proprietary
  Caltech intellectual property"); no training-code release.
* License: Apache-2.0 (NOTICE: "built from Qwen3-1.7B, © Alibaba Cloud").
* Benchmarks (whitepapers): Ternary 8B avg 75.5 (Qwen3-8B FP16: 79.3);
  Ternary 1.7B avg 57.5; 27B: ternary retains 94.6% of FP16, 1-bit 89.5%.
* No 0.5B exists from PrismML — **1.7B is the smallest**, hence the size
  prepared here.

**Bonsai-1.7B architecture** (config.json, both variants identical):
`Qwen3ForCausalLM`, 28 layers, hidden 2048, 16 Q heads / 8 KV heads,
head_dim 128, intermediate 6144 (SiLU/SwiGLU), RMSNorm eps 1e-6,
vocab 151,669 (Qwen2 tokenizer), tied embeddings (no separate lm_head
tensor), YaRN ×4 → 32k context, no biases anywhere. 1.720B params.

### 1.3 Citations

* HF org listing (33 models): https://huggingface.co/api/models?author=prism-ml
* Downloaded repos: https://huggingface.co/prism-ml/Bonsai-1.7B-unpacked
  (lastModified 2026-04-16T16:38Z) and
  https://huggingface.co/prism-ml/Ternary-Bonsai-1.7B-unpacked (2026-04-16T11:45Z)
* Ternary format/announcement: https://prismml.com/news/ternary-bonsai
  ("{−s, 0, +s} … encoded (−1,0,+1) … shared FP16 scale factor (s) for each
  group of 128 weights"; "no higher-precision escape hatches")
* Docs: https://docs.prismml.com/llms.txt ;
  https://docs.prismml.com/download/formats.md (Q1_0 = {−1,+1} 1 bit each,
  Q2_0 = {−1,0,+1} 2 bits); https://docs.prismml.com/models/bonsai-1-7b.md
* Whitepapers (PDFs in https://github.com/PrismML-Eng/Bonsai-demo):
  `ternary-bonsai-8b-whitepaper.pdf` (2026-04-16; Qwen3 bases, g128 format,
  quantized-layer list), `1-bit-bonsai-8b-whitepaper.pdf` (end-to-end 1-bit,
  Caltech IP), `bonsai-27b-whitepaper.pdf`
* 27B release coverage (2026-07-14): https://www.marktechpost.com/2026/07/14/prismml-releases-bonsai-27b-1-bit-and-ternary-builds-of-qwen3-6-27b-that-run-on-laptops-and-phones/
* deepgrove disambiguation: https://huggingface.co/deepgrove/Bonsai ,
  https://github.com/deepgrove-ai/Bonsai , per-channel-scale note:
  https://github.com/microsoft/BitNet/issues/169

---

## 2. What was downloaded

Both `-unpacked` repos (plain HF safetensors with the quantized VALUES
dequantized to fp16 — the form our pipeline can consume; the gguf/mlx repos
are packed for llama.cpp/MLX kernels and were not needed):

| Variant | Path | Size on disk |
|---|---|---|
| 1-bit    | `/home/deni/bonsai_weights/1bit/`    | 3.5 GB (model.safetensors 3.44 GB) |
| ternary  | `/home/deni/bonsai_weights/ternary/` | 3.5 GB (model.safetensors 3.44 GB) |

310 tensors each: 1 embed + 28×(7 projections + input/post norms +
q_norm/k_norm) + final norm. **No bias tensors, no lm_head (tied).**
Downloaded with `huggingface_hub.snapshot_download` (venv), ~106 s each.

## 3. Verified weight representation (measured, not assumed)

For every projection tensor `W [d_out, d_in]` (fp16): each row's consecutive
group of 128 inputs is symmetric two/three-level:

* 1-bit: `W[r, 128g:128g+128] ∈ {−s[r,g], +s[r,g]}` — **0 zeros anywhere**
  (all 196 projection tensors, zero_frac = 0.0000).
* ternary: `∈ {−s[r,g], 0, +s[r,g]}` — zero_frac 35.0–44.0% (mean 39.5%).
* Embeddings follow the same structure (kept host-side, like BitNet).
* Exception: **~1e-5 of weights** (14,584 of 1.409B in 1-bit; 8,503 in
  ternary; concentrated in MLP tensors) sit a few fp16 ulps off their group
  level (≤9.8e-4 absolute over the full set, ≲0.6% relative). Cause unknown (likely conversion
  double-rounding in PrismML's unpack step; they persist at any hypothesized
  group size 32/64/128). Handled exactly via sparse residuals (below).

## 4. Golden CPU reference (fresh venv, stock transformers)

`trust_remote_code` is **NOT needed** — both load as stock `Qwen3ForCausalLM`
(no custom code in the repos; the custom-code caveat in the task is moot).
venv: torch 2.13.0+cpu, transformers 5.14.1, numpy 2.x, compute f32.

Prompt `"What is the capital of France?"`, greedy, 16 new tokens, two modes:
`raw` (no chat template — matches `PIM_NO_CHAT_TEMPLATE=1`) and `chat`
(same system message the BitNet pipeline uses).

| Variant/mode | Greedy 16-token output |
|---|---|
| 1bit/raw | ` What is the capital of France? What is the capital of France? What is` (echo — base-style) |
| 1bit/chat | `The capital of France is Paris.<|im_end|>\n\nParis is the largest city in France` |
| ternary/raw | ` The capital of France is Paris. Paris is a city located in the north of` |
| ternary/chat | `The capital of France is Paris. It is a major city located in the north` |

Files: `golden_{1bit,ternary}.json` (config, exact input_ids, generated ids +
text, top-16 logits for first 4 steps, both modes) and
`golden_{1bit,ternary}_logits.npy` (float32 [4, 151669] full-vocab logits of
the raw mode). First-step top-1: 1bit/chat id 785 ("The") logit 24.578;
ternary/chat id 785 logit 24.343.

## 5. Extraction (`extract_bonsai.py`)

Output: `/home/deni/bonsai_weights/extracted/<variant>/L<LL>.<proj>.npz`,
196 files per variant (28 layers × q/k/v/o/gate/up/down), 1.4 GB per variant,
plus `manifest.json` (shapes, zero fractions, residual counts per tensor).

NPZ keys (per projection):

| key | dtype/shape | meaning |
|---|---|---|
| `codes` | int8 `[d_out, d_in]`, {−1,0,+1} | exactly `PimBitLinear._w_int` (pim_linear.py:251-257 consumes this to build pos/neg masks) |
| `group_scales` | f32 `[d_out, d_in/128]` | s[r,g]; replaces the scalar `base.weight_scale` |
| `residual_idx` / `residual_val` | int64 `[n,2]` / f32 `[n]` | sparse exact deltas (w_true − s·code) for the ~1e-5 stray weights; host-side correction `y += Δ·x[col]` |
| `d_in, d_out, group_size` | int64 | group_size = 128 |
| `weight_scale` | f32 scalar | mean(group_scales) — BitNet-API-shape compatibility ONLY; approximate. Exact math must use `group_scales`. |

Reconstruction identity (verified bit-exact):
`W = codes * repeat(group_scales, 128, axis=1)` + residuals.

Projection shapes per layer: q,o `[2048,2048]`; k,v `[1024,2048]`;
gate,up `[6144,2048]`; down `[2048,6144]`. All d_in % 32 == 0 (pim_linear's
assert) and % 128 == 0 → **each scale group = exactly 4 of the client's
32-input chunks; groups never straddle chunk or sub-handle boundaries.**

## 6. Verification (`verify_extraction.py`, `verify_report.json`)

Sampled 21 tensors/variant (layers 0/13/27 × all 7 projections), 4 random
activations each; numpy popcount path mirrors the server math bit-for-bit:

* **A. Reconstruction: bit-exact (max dev 0.0) — confirmed on the sampled 42
  AND on a full sweep of all 392 tensors** with residuals; without residuals
  max dev 9.77e-4 (the stray weights only).
* **B. Dual-track popcount identity == int64 `codes @ x`: exact (max err 0)**
  — pos/neg mask packing + 8 bitplanes with factors [1,2,4,8,16,32,64,−128],
  i.e. the extraction is consumable by the existing machinery unmodified.
* **B1. 1-bit single-track identity exact (max err 0)**: with zero-free ±1
  weights, `neg_mask = ~pos_mask` within d_in, so per bitplane
  `pc_neg = pc(x_b) − pc_pos`, giving `y = Σ_b f_b·(2·pc_pos,b − pc(x_b))`
  — equivalently **y = 2·(pos-track-only result) − Σ_i x_i** where Σx is a
  free host-side scalar. See §8.
* **C. Group-rescale matvec** `y = Σ_g s[r,g]·(codes_g·x_g)` vs f64 `W @ x`:
  with residual correction **exactly 0.0**; without: ≤0.60 absolute on
  int8-range activations (bounded by the stray weights; negligible vs y RMS
  of thousands).
* **D. int8-activation path** (BitNet per-token absmax→[−128,127] quant, which
  the PIM run will impose) vs the model's own fp matmul on float activations:
  max abs deviation 0.05–0.16 per output (unit-normal activations). This is
  activation-quantization cost, NOT extraction error — the unpacked Bonsai
  computes fp16 matmuls with no activation quant. Their deployed llama.cpp
  kernels quantize activations to 8-bit blocks, so an int8-activation PIM run
  is deployment-faithful in spirit. Expect Bonsai-PIM outputs to deviate
  slightly from `golden_*` logits for this reason (unlike BitNet, whose
  BitLinear quantizes activations natively → bit-faithful was achievable).

## 7. THE structural difference vs BitNet: per-group scales

BitNet: ONE bf16 `weight_scale` per projection; client rescales once
(pim_linear.py:258 reads it; :440 applies `y_int * weight_scale / x_scale`).

Bonsai: **s[r,g] per (output row, 128-input group)**. The integer popcount
matvec is unchanged, but partial sums must be rescaled per group before
summation over d_in:

```
y[r] = Σ_g s[r,g] · ( Σ_{c∈group g} popcount-matvec(chunk c) )   / x_scale
```

Exact client adaptation (all in a Bonsai-variant of pim_linear.py; server
UNCHANGED for correctness):

1. Load `codes` + `group_scales` from npz instead of `base.weight` /
   `base.weight_scale` (pim_linear.py:251-258). Slice `group_scales[a:b, :]`
   alongside the d_out slices (D_OUT_SLICE=2048, :289-293).
2. LOAD path: set env `PIM_MAX_CHUNKS_PER_SUB=4` (pim_linear.py:25 already
   reads it) → each sub-handle = 4 chunks = exactly 1 scale group. In
   `_one_calib` (:581-597), replace `y_acc += partial` with
   `y_f32_acc += s[:, g_of(sub)] * partial` (f32 accumulate).
3. V2 path (the >95% of traffic once pools fill): reuse the existing
   `v2_parts` chunk-range-body machinery (:337-357 build, :634-675 dispatch)
   with per-GROUP parts (c_a..c_b = 4-chunk windows) and the same per-part
   rescale at accumulation (:673-675). The single full-body V2 request
   (:676-683) cannot carry per-group scales — do not use it for Bonsai.
4. Rescale line :440 becomes `/ x_scale` only (group scales already applied);
   then add the sparse residual correction
   `y[r] += Σ_residual Δ·x_int8[col] / x_scale` (~40-350 terms/projection,
   host, negligible).
5. Voting: 3-way calib voting (:686-689) and n_copies in-row voting
   (:718-732) vote on int32 popcount outputs — vote each per-group partial
   (int) BEFORE applying s[r,g], then scale+sum. k/v replication (n_real=1024,
   n_copies=2) shares rows→shares scales, so vote-then-scale stays valid.

Cost note: group-granular requests are 16 (d_in 2048) or 48 (d_in 6144)
round-trips per d_out-slice where BitNet used 1-4 → per-token server-request
overhead grows ~4-16×. Optional later server change to win it back: return
per-4-chunk partial vectors per request (one new magic; response 4×8 KB), or
extend LOAD/MM3D with a server-side f32 scale table per handle. Not required
for a first silicon run.

A quick-and-dirty smoke alternative (NOT exact, don't use for claims):
per-tensor `weight_scale` from the npz gives a BitNet-identical zero-code-change
run whose outputs are wrong by the scale dispersion (typically ±30-50% per
group) — only useful to exercise plumbing.

## 8. 1-bit variant mapping note (task item 4)

Mapping chosen: same dual-track {pos_mask, neg_mask} convention with an
**empty zero-set** — codes are strictly ±1, so for every input chunk
`neg_mask = pos_mask XOR 0xFFFFFFFF` (within real d_in). This runs on the
existing client + server with **zero changes** (beyond §7's group scales):
masks are just denser (~50% ones vs ~30% for ternary/BitNet).

Popcount identity: per bitplane b, `pc(neg&x_b) = pc(x_b) − pc(pos&x_b)`,
hence `y = Σ_b f_b·(2·pc_pos,b − pc(x_b)) = 2·y_postrack − Σ_i x_i`
(y per column; Σ_i x_i is one scalar per token, free on host at pack time).
Verified exact in check B1. Consequence: the neg track is REDUNDANT for
1-bit — a "single-track" mode would halve pool rows (§9) and halve MAJ3
DRAM work per matmul. Two implementation options: (a) client sends
neg_mask=0 and host computes `2·y − Σx` — no server change, but no DRAM-op
savings because the server processes every (chunk,sign) unit unconditionally
(test_bitnet_server.cpp:1362: `n_units = n_chunks * 2` — all-zero masks are
NOT skipped, they still get pool rows and per-column writes + MAJ3 bodies);
(b) a server single-track flag that emits only sign-0 units — small change,
real 2× on rows and ops. Recommend (b) only after a correctness run with (the
unchanged) dual-track.

Zero-assumption audit of the existing client (places checked, none break):
* pim_linear.py:295-302 — pos/neg masks from `==1`/`==−1` independently;
  a weight in NEITHER set (ternary zero) is simply absent; a weight in
  exactly one set (1-bit) is the normal case. No code path requires zeros.
* pim_linear.py:315-319 — partial-slice replication copies masks verbatim.
* pim_linear.py:446-447 — bias add skipped (Bonsai has no biases).
* test_bitnet_server.cpp:1362 (+ MM3D unit loops ~:1717) — units iterate
  (chunk, sign) unconditionally; density of the mask is irrelevant to
  control flow. Popcounts stay in [0,32] per word — no range issue.

## 9. Sizing / fit under the pool geometry

Client slicing (D_OUT_SLICE=2048, chunks of 32 inputs; LOAD sub-handles of
`PIM_MAX_CHUNKS_PER_SUB` chunks; each sub-handle consumes
`ceil(n_chunks_sub × 2 tracks / 4 banks)` rows from EVERY bank's pool —
test_bitnet_server.cpp:1362-1363 with N=4):

Per Bonsai-1.7B layer, 16-chunk subs (today's default):

| proj | shape | slices | subs | pool rows/bank |
|---|---|---|---|---|
| q_proj | 2048×2048 | 1 | 4 | 32 |
| k_proj | 1024×2048 | 1 (n_copies=2) | 4 | 32 |
| v_proj | 1024×2048 | 1 (n_copies=2) | 4 | 32 |
| o_proj | 2048×2048 | 1 | 4 | 32 |
| gate_proj | 6144×2048 | 3 | 12 | 96 |
| up_proj | 6144×2048 | 3 | 12 | 96 |
| down_proj | 2048×6144 | 1 | 12 | 96 |
| **per layer** | | | **52** | **416** |
| **×28 layers** | | | **1,456** | **11,648** |

(Group-aligned 4-chunk subs quadruple the sub count to 5,824 but each takes
2 rows/bank → the total is the same 11,648 rows/bank. BitNet-2B for
comparison: 2,940 subs, 23,520 rows/bank — Bonsai-1.7B is half the footprint.)

Pool capacity today (clone-ok layouts, minus the PIM_V2_SCRATCH=16 tail;
a handle needs rows on ALL 4 banks, so the smallest bank pool gates):

| DIMM | pool rows/bank | usable | 16-chunk subs resident | 4-chunk subs resident |
|---|---|---|---|---|
| 0 | 168 (bank1: 223) | 152 | 19 | 76 |
| 2 | 197 | 181 | 22 | 90 |
| **0+2** | | | **41 of 1,456 (2.8%)** | **166 of 5,824 (2.9%)** |

**Verdict: Bonsai-1.7B does NOT fit resident in the current pools — same
operating regime as BitNet-2B production: first ~3% of sub-handles go
LOAD-resident, the rest stream via the V2 split path.** That regime is
proven (BitNet runs the full model this way at 21 s/tok single-DIMM /
faster multi-DIMM). `PIM_LOAD_OVERFLOW_SUBS=1` (default on) additionally
overflows into any screened extra subarray pools. Full residency would need
11,648 rows/bank ≈ 55–75 screened subarray windows per bank at the observed
~25-33% clone-ok yield — a screening campaign, not a config change. The
1-bit single-track option (§8b) halves the requirement to 5,824 rows/bank.

No NEW pool planning is required to run 1.7B the way BitNet runs today.
(For a future Bonsai-4B/8B: d_in grows to 2560/4096-class → same regime,
larger V2 fraction; 27B is out of scope for this rig's host RAM.)

## 10. What the silicon phase needs (no rediscovery)

Inputs ready:
* Weights/codes: `/home/deni/bonsai_weights/extracted/{1bit,ternary}/L*.npz`
  (+ `manifest.json`). Tokenizer/model for the host wrapper:
  `/home/deni/bonsai_weights/{1bit,ternary}/` (stock Qwen3, no
  trust_remote_code).
* Golden refs: `golden_{1bit,ternary}.json` + `_logits.npy` here.
* Harnesses: `extract_bonsai.py`, `verify_extraction.py` (CPU re-checks).

Client work (one new file recommended, e.g. `pim_linear_bonsai.py`, keeping
the BitNet path untouched): the §7 changes (npz loading, group-scale
accumulation, group-granular V2 parts, residual correction, vote ordering) +
a Bonsai run script cloned from run_bitnet_pim.py that instantiates
Qwen3ForCausalLM from the LOCAL variant dir and substitutes the 196
projections (model.model.layers[i].{self_attn,mlp}.*_proj — same attribute
paths as BitNet).

Server invocation: UNCHANGED —
`bitnet-proj-server <bender> <calib> <banks>` with the production env per
DIMM (from run_bitnet_pim.py DIMM_SPECS):
* DIMM 0: calib_dimm0.txt, `PIM_POOL_LIST_FILE=pool_layout_dimm0_cloneok_bank{bank}.txt`,
  `PIM_SUB_START=38400 PIM_SUB_END=39040`, optional fused colmask files.
* DIMM 2: calib_dimm2.txt, `pool_layout_dimm2_cloneok_bank{bank}.txt`,
  `PIM_SUB_START=45312 PIM_SUB_END=45952`.
* Always: `BITSTREAM_IMEM=8192`, `PIM_V2_SCRATCH=16`, `PIM_USE_LOAD_WEIGHTS=1`,
  `PIM_MAX_CHUNKS_PER_SUB=4` (group alignment), banks `0,1,2,3`,
  multi-DIMM `--dimms 0,2` supported as-is.
* All the usual rig rules apply (never SIGKILL during DMA, no arbitrary
  timeouts, bring-up per RUNBOOK_TOWER.md).

Suggested silicon sequence: (1) single projection L00.q_proj ternary,
`PIM_INT_DIFF=1`, expect int-exact vs `codes @ x` per group; (2) same for
1bit (dual-track unchanged); (3) layer-0 all-7; (4) full-model ternary chat
prompt vs `golden_ternary.json` (expect small logit drift from activation
quant — §6D — so compare argmax token agreement, not bit-exactness);
(5) full-model 1bit. Remember the pool-collision lesson: layer-0-only
validation cannot catch pool-scale bugs — do one full-model run before
calling it validated.

Open items / risks:
* Activation-quant deviation (§6D) means token-level divergence from the fp16
  golden is possible on long generations; the chat-mode first tokens ("The
  capital of France is Paris.") should be robust.
* The per-group request amplification (§7 cost note) will slow tokens
  relative to BitNet at equal size until the optional server-side partial
  extension lands.
* The ~1e-5 stray-weight residuals are host-corrected; if skipped, worst-case
  per-output error ≤0.6 int units before rescale — likely invisible, but the
  correction is cheap enough to keep on.
