> **Repo note.** Verbatim working record from the rig workspace
> (`/home/deni/Claude/bonsai_client_2026_07_20/README.md`), preserved unedited as a citation
> target. Reader-facing synthesis: [`BONSAI_2026_07.md`](BONSAI_2026_07.md).

---

# Bonsai client-side integration + SIM validation (task B4, 2026-07-20)

Host-only session. The FPGA was NEVER touched: every server run in this
directory went through `runtime/sim-server`, a wrapper that hard-forces
`PIM_BACKEND=sim` around a byte-verified COPY of the prebuilt
`bitnet-proj-server` (sha256 `f1d21ee641131538…`, copied 2026-07-20 so the
concurrent SiMRA-tree rebuild cannot race us; `cmp` verified). Nothing under
`SiMRA-DRAM-main/` was modified or rebuilt. Every validation script asserts
(a) the server printed the `PIM_BACKEND=sim — using in-process SimDramModel`
banner and (b) the server process holds zero `/dev/xdma*` fds.

Companion prep report (model facts, extraction, identities, fit):
`/home/deni/Claude/bonsai_prep_2026_07_20/README.md`.

---

## 1. What changed (production files)

### 1.1 `/home/deni/bitnet_weights/pim_linear.py` (844 → 1103 lines)

Pristine pre-change copy: `baseline_src/pim_linear.py.orig`
(sha256 `ef47c380c279e0ff…`; post sha256 `2a8b3bf408852fee…`).
All changes are **default-off**: with `weight_spec=None` (the default) every
code path is the legacy one — proven byte-identical in §2.

| lines (new file) | change |
|---|---|
| :201-203 | `PimBitLinear.__init__` gains optional `weight_spec=None` kwarg |
| :250-326 | weight-source switch: `None` → legacy `base.weight`/`base.weight_scale` extraction (verbatim); dict → external `codes` int8 + optional `group_scales` f32 [out, in/128] + optional sparse residuals (`residual_idx`/`residual_val`) + scalar `weight_scale` fallback. Group mode validates shapes, requires the server backend, d_in % group_size == 0, group_size % 32 == 0 |
| :401-430 | per-slice V2 parts: group mode builds **per-GROUP part bodies** (one 4-chunk window per 128-input scale group, round-robin'd across servers); the legacy `_n_srv_cfg > 1` multi-DIMM split is untouched (elif). Group mode never uses the single full-body V2 request (it cannot carry per-group scales) |
| :454-462 | LOAD sub sizing: group mode clamps `sub_size` to the group width and rounds down to a divisor of it, so **no sub-handle straddles a scale-group boundary**. With group 128 (4 chunks) the default `PIM_MAX_CHUNKS_PER_SUB=16` self-aligns to 4 — the env override in the prep report is not required (setting it is harmless) |
| :493-506 | each load-sub gains a `group` tag (asserted single-group); each slice stashes `group_scales[a:b, :]` (real rows only — copy rows never need scales, see vote order) |
| :549-556 | `forward()`: group mode divides by `x_scale` only (scales+residuals already applied); legacy line byte-identical in the else |
| :621-624 | f32 output buffer for group mode |
| :695-798 | **the group dispatch** in `_pim_matmul_one_token`: `_one_calib_groups(cal_idx)` fetches int32 per-group partials `[n_groups, 2048]` (LOAD → one MM3D per sub routed into its group slot; V2 → one request per group part; per-server accumulation buffers merged after joins — thread-safe multi-DIMM). **Vote-then-scale** exactly as prep §7.5: 3-way calib vote and the n_copies in-row vote both run on the INT partials (median / mean-of-2 semantics identical to legacy), only then `y[a:b] = Σ_g s[r,g]·G_voted[g]` in f32. `PIM_INT_DIFF=1` prints per-slice `groups-sum` int exactness (validated: 100% in sim). Ends with `continue` — the legacy slice flow below is untouched |
| :985-1004 | sparse-residual host correction after the slice loop: `y[r] += Δ·x_int8[col]` (np.add.at, npz order, f32) + telemetry (`_resid_terms_applied`, `_resid_abs_sum`, `_resid_abs_max`) |
| :1017-1041 | `pim_substitute(..., weight_spec_fn=None)`: optional `(layer_idx, proj_path) → weight_spec` callable |
| :1085-1096 | `print_pim_timing_summary`: residual-telemetry line (prints nothing for BitNet) |

Design note vs the prep report: §7.2 sketched applying `s[:, g]` inside
`_one_calib`; §7.5 required voting on int partials BEFORE scaling. Both
cannot hold in one accumulator, so the implementation follows §7.5
(vote-then-scale) and keeps per-group INT partials until after votes — on
sim they are equivalent (deterministic server); on silicon vote-then-scale
is the correct order. No other deviation from the prep design was needed.

### 1.2 `/home/deni/bitnet_weights/run_bitnet_pim.py` (195 → 272 lines)

Pristine copy: `baseline_src/run_bitnet_pim.py.orig`. Default invocation
(`--model bitnet`, the default) is the original flow line-for-line.
No separate bonsai_run.py fork — per the "prefer NOT forking logic"
guidance the production script gained a model switch.

| lines | change |
|---|---|
| :25-29 | `SERVER` now honors `PIM_SERVER_PATH` env override (default = production path, unchanged) — lets sim runs point at the copied wrapper without touching production defaults |
| :31-49 | `BONSAI_SPECS`: `bonsai_1bit` / `bonsai_ternary` → model dir `/home/deni/bonsai_weights/<variant>/` + extract dir `…/extracted/<variant>/`; header comment records the 1-bit dual-track mapping note |
| :52-69 | `make_bonsai_spec_fn(extract_dir)`: loads `L<LL>.<proj>.npz` on demand → weight_spec dict (codes, group_scales, group_size, residual_idx/val, weight_scale) |
| :95-104 | `--model {bitnet,bonsai_1bit,bonsai_ternary}` (default bitnet) and `--dtype {bfloat16,float32}` (default: bf16 for bitnet — unchanged; f32 for bonsai, matching the golden reference) |
| :181-218 | model-loading branch: bitnet branch is the original code verbatim; bonsai branch loads stock `Qwen3ForCausalLM` from the LOCAL dir (28 layers, hidden 2048, GQA 16/8, head_dim 128, intermediate 6144, tied embeddings, no biases — all from the checkpoint's config.json; no trust_remote_code) and builds `weight_spec_fn`. Attention/norms/embeddings/LM-head/sampling stay on CPU exactly like BitNet |
| :239 | `pim_substitute(..., weight_spec_fn=weight_spec_fn)` |

### 1.3 The 1-bit variant mapping (explicit)

`bonsai_1bit` runs as **dual-track with an empty zero-set**: codes are
strictly ±1, so the packed `neg_mask` is the bitwise complement of
`pos_mask` on every real output column (asserted at unit-test time on
re-packed chunks). This runs on the UNCHANGED client machinery and the
UNCHANGED server — masks are just denser (~50% ones). The neg track is
therefore **redundant**: `y = 2·y_postrack − Σ_i x_i` (prep §8, identity
check B1) would halve pool rows and MAJ3 work, but needs a server
single-track flag — OUT OF SCOPE here, deliberately not touched. The run
script prints this note when `--model bonsai_1bit` is selected.

---

## 2. Regression proof (Bonsai flags off ⇒ byte-identical)

`smoke_bitnet_l0.py`: BitNet-2B, layer-0 all-7 projections substituted,
SIM server (production-shaped env: `PIM_USE_LOAD_WEIGHTS=1`, dimm0 pool
layouts, banks 0,1,2,3), fixed-seed 3-token activations through every
projection. Captured BEFORE the edits (`regression/pre/`) and re-run
AFTER (`regression/post/`):

| check | result |
|---|---|
| 7 per-projection y outputs (.npy bytes) | **IDENTICAL** (cmp, all 7) |
| full wire stream: every request body incl. length prefixes, in order | **IDENTICAL** — sha256 `9c9e06cfd8fc…`, 255 requests, 130,087,716 B both runs |
| ENOSPC/LOAD→V2 fallback point | identical (at gate_proj) |

The wire-stream hash makes this stronger than output equality: the edited
client sends bit-for-bit the same protocol bytes in the same order.

Re-run once more against the FINAL shipped files after all edits
(`regression/final/`): outputs and wire stream again identical to `pre/`.
Shipped hashes: `pim_linear.py` sha256 `49c014936d0a1582…`,
`run_bitnet_pim.py` sha256 `05ebef6bc886c802…`.

## 3. Unit exactness (layer 0, all 7 projections, both variants)

`unit_bonsai_l0.py`: random int8 activations (2 per projection) through
`PimBitLinear._pim_matmul_one_token` on the SIM server vs a mirror numpy
reference computed from the extracted `codes` + `group_scales` +
`residuals` (identical f32 op order). Six configurations:

| variant | mode | dispatch exercised | result |
|---|---|---|---|
| ternary | v2 | per-group V2 parts, calib vote ON (0/1/2 trips) | **bitwise exact**, max abs diff = 0.0, all 7 |
| ternary | load | LOAD/MM3D 4-chunk subs; ENOSPC hit mid-run → V2 fallback also covered | **bitwise exact**, all 7 |
| ternary | multi | TWO sim servers (production DIMM_SPECS shape: dimm0+dimm2 cloneok pools, sub windows 38400/39040 + 45312/45952), threaded round-robin group parts | **bitwise exact**, all 7 |
| 1bit | v2 / load / multi | same three + empty-zero-set mask assertion (`neg == ~pos` on re-packed chunks; 0 zero codes) | **bitwise exact**, all 7 × 3 |

Per-projection request counts (2 tokens, vote ON): q/o 96, k/v 32 (in-row
copy vote, no calib vote — n_copies=2), gate/up/down 288. Sparse residuals
were live in every run (ternary L0: k 20, o 55, gate 48, up 43, down 94
terms; 1bit L0: q 3, k 1, v 15, o 7, gate 355, up 263, down 219) and the
results are STILL bitwise exact — plumbing plus residual correction proven
end-to-end. `PIM_INT_DIFF=1` additionally reports 100.0000% per-slice int
exactness (groups-sum vs `codes @ x`) in sim.

## 4. E2E-lite (full 28-layer forward, canonical prompt, SIM)

`e2e_bonsai.py`: ONE prefill forward of the raw canonical prompt
("What is the capital of France?", 7 tokens = the golden raw mode), ALL
28 layers x 7 projections = 196 PimBitLinears on the SIM server,
`PIM_USE_LOAD_WEIGHTS=1` (LOAD until ENOSPC then per-group V2 — the
production streaming regime). Step-0 last-position logits compared against
`golden_<variant>_logits.npy[0]` (verified equal to a fresh venv forward).
Vote was OFF for wall time — in sim the 3-trip vote is numerically a no-op
(deterministic server; §3 ran vote ON, bitwise exact); `--vote` restores it.

**Attribution baseline first**: the SAME checkpoint forwarded on CPU in the
client env (torch 2.7.1 / transformers 4.52) reproduces the golden venv
(torch 2.13 / transformers 5.14) logits with **max|Δ| = 0 — bit-exact, both
variants**. Framework drift is therefore ZERO and every PIM deviation below
is attributable to the PIM path itself, whose only lossy element is the
BitNet-style per-token int8 activation quantization (the prep README §6D
prediction; the weight path is exact per §3).

| metric (step-0 logits, full vocab 151,669) | ternary | 1bit |
|---|---|---|
| argmax golden / client-CPU / PIM | 576 / 576 / **576** ("The") | 3555 / 3555 / **3555** ("What", the base-style echo — matches golden) |
| top-16 id-set overlap | **16/16** | 15/16 |
| top-16 exact-order prefix | 6 | 4 |
| max abs dev, full vocab | 0.4693 | 0.3446 |
| mean abs dev, full vocab | 0.0703 | 0.0586 |
| max abs dev on golden top-16 ids | 0.3140 | 0.1295 |
| golden top-1 logit → PIM | 15.005 → 14.813 | 13.667 → 13.710 |
| server requests (7-token prefill) | 40,899 | 40,899 |
| wall (sim) | 1103 s (39.4 s/layer) | 1102 s |
| residual terms applied / sum abs dy_int / max single | 59,521 / 224.0 / 0.071 | 102,088 / 459.4 / 0.062 |

Deviations of ~0.05-0.47 on logits of magnitude ~12-15 (≤ ~2.5% of the
top-1 logit, argmax and near-full top-16 agreement) are consistent with
int8-activation quantization accumulating over 28 layers — the same class
of deviation the deployed Bonsai llama.cpp kernels accept (8-bit activation
blocks). The sparse-residual host correction is measured and tiny, as
predicted: its largest single contribution to any y_int is 0.071 (ternary)
/ 0.062 (1bit) pre-x_scale units — kept ON (negligible cost, exactness at
the unit level).

Per-layer hidden-state drift vs the golden per-layer states
(`analyze_e2e.py`; max over 7 positions x 2048 dims, relative to the
layer's max |h|):

* ternary: ≤ 0.31% of layer magnitude through L15, peaking 0.86% (L19),
  0.79% at L26; post-final-norm state (L27n) 0.9 abs.
* 1bit: ≤ 0.55% through L25, 0.76% at L26; post-final-norm 0.7 abs.
* Note: the raw hook capture at the LAST layer must be final-RMSNorm'd
  before comparing — HF `output_hidden_states` returns the post-norm state
  as its last entry. Before that correction the L27 row looks like a
  5e3/1.8e4 "blowup"; it is an index-semantics artifact, not drift
  (`analyze_e2e.py` handles it; logits are the honest last-layer check).

Production entry-point smoke (the actual `run_bitnet_pim.py`, not the
harness): `--model bonsai_ternary --layers 0,1 --max-tokens 3`, raw
prompt, sim → generated exactly ` The capital of` (golden ternary/raw
continuation), 0.03 tok/s in sim, residual telemetry printed. And the
LITERAL silicon command shape `--model bonsai_ternary --dimms 2 --bank
0,1,2,3` (vote ON, `BITSTREAM_IMEM=8192`) ran end-to-end in sim: the
DIMM-2 trio (calib_dimm2 + cloneok pool layout + `PIM_SUB_START=45312
PIM_SUB_END=45952`) was wired automatically into the server env
(visible in the PimServer key), first generated token " The" — correct.
Measured vote-ON traffic: 4,011 requests for 8 token-passes over one
layer ≈ 500/layer-token (theoretical 560 minus ENOSPC-latched LOAD
probes).

Artifacts: `e2e/{ternary,1bit}_report.json`, `*_pim_logits.npy`,
`*_cpu_logits.npy`, `*_pim_hidden.npy`, `golden_hidden_*.npy`, run logs.

## 5. Silicon-run recipe (config change, not a discovery process)

All host-side pieces are in place; the silicon phase is env + flags.
**Rig rules apply as always**: bring-up per `RUNBOOK_TOWER.md`, locate the
card via `lspci -nn -d 10ee:` (never by BDF), `BITSTREAM_IMEM=8192` always,
never SIGKILL during DMA, no arbitrary timeouts (leave `PIM_RECV_TIMEOUT_MS`
unset), post-JTAG bring-up = remove+rescan or warm reboot.

### 5.1 Production invocation (bender 2 = DIMM 2, the production die)

```bash
cd /home/deni/bitnet_weights
BITSTREAM_IMEM=8192 \
python3 run_bitnet_pim.py --model bonsai_ternary --dimms 2 --bank 0,1,2,3 \
    --prompt "What is the capital of France?" --max-tokens 16
```

`--dimms 2` pulls `DIMM_SPECS[2]` and hands the server the MANDATORY
DIMM-2 trio automatically (per-server env, same as BitNet production):
`calib_dimm2.txt` + `PIM_POOL_LIST_FILE=pool_layout_dimm2_cloneok_bank{bank}.txt`
+ `PIM_SUB_START=45312 PIM_SUB_END=45952`. Multi-DIMM: `--dimms 0,2`
(group parts and load-subs round-robin across benders — the exact shape
validated bitwise in sim, §3 multi). Single-`--bender` invocations work
too but then calib/pool/sub-window env is on you — prefer `--dimms`.

Notes:
* `PIM_USE_LOAD_WEIGHTS=1` — set by the script (default). First ~2.9% of
  the 5,824 4-chunk sub-handles go LOAD-resident (prep §9: pools gate at
  ~166 subs on 0+2), the rest stream per-group V2 — the SAME regime
  BitNet-2B production runs in, no new pool planning required.
* `PIM_MAX_CHUNKS_PER_SUB` does NOT need to be set: group mode self-clamps
  sub-handles to the 4-chunk scale-group width (setting =4 is harmless).
* `PIM_V2_SCRATCH` defaults to 16 in the server — no env needed.
* 1-bit: `--model bonsai_1bit` — identical flags; dual-track with empty
  zero-set, no server change (see §1.3; ~50%-dense masks).
* `PIM_SERVER_PATH` unset → the production server binary in the SiMRA
  tree, as always. (Sim reruns: point it at
  `bonsai_client_2026_07_20/runtime/sim-server`.)
* `PIM_FUSED_COSET=1` operates at MAJ3-emission level (below the request
  protocol) so it should compose with group-granular bodies, but it was
  NOT exercised with Bonsai in sim — A/B it with `PIM_INT_DIFF=1` before
  trusting it in a Bonsai production run.

### 5.2 Suggested silicon sequence (prep §10, with the new tooling)

1. Single projection, int-level: `--model bonsai_ternary --layers 0
   --projs self_attn.q_proj --max-tokens 1` with `PIM_INT_DIFF=1` — the
   group-mode diff line prints per-slice `groups-sum exact=` vs
   `codes @ x` (sim showed 100.0000%; silicon should look like BitNet's
   int-diff profile, cell-noise-limited).
2. Same for `--model bonsai_1bit` (denser masks, otherwise identical).
3. Layer-0 all 7 (`--layers 0`), still `PIM_INT_DIFF=1`.
4. Full-model ternary, chat prompt, vs `bonsai_prep_2026_07_20/
   golden_ternary.json` chat mode — compare TOKEN AGREEMENT, not logits
   bit-exactness (int8-activation quant is a real, quantified deviation:
   §4 below; silicon adds cell noise on top).
5. Full-model 1bit. **Pool-collision lesson applies: layer-0-only
   validation cannot catch pool-scale bugs — do one full-model run before
   calling silicon validated.**

### 5.3 Expected request amplification (measured, sim)

Group-granular rescale sends one request per 128-input group instead of
1-4 per d_out-slice (prep §7 cost note):

| | per layer-token | full model per token |
|---|---|---|
| Bonsai vote ON (production `PIM_VOTE_FULL=1`) | 560 theoretical; **~500 measured** (dimms-2 sim run, ENOSPC-latched) | **~15,680** |
| Bonsai vote OFF | 208 | 5,824 (**measured 5,843**/token incl. LOAD bodies — 40,899 req / 7 tokens, full-model e2e) |
| BitNet-2B measured same sim harness (vote ON, LOAD+V2 mix) | ~85 | ~2,550 (30 layers) |

≈ 6.2× more round-trips per token than BitNet-2B at vote parity — inside
the prep's 4-16× band. Total in-DRAM MAJ3 work per token is roughly
UNCHANGED (same (chunk,sign) unit count, split 4× finer: each 4-chunk
request runs 2 rounds instead of 8), so the growth is round-trip/pipe
overhead, not DRAM time. Expect slower tokens than an equal-size BitNet
until the prep report's optional server-side extension lands (per-4-chunk
partial response vectors, or a server-side f32 scale table per handle —
one new magic either way; NOT required for correctness).

### 5.4 Inputs inventory (all ready)

* Weights/codes: `/home/deni/bonsai_weights/extracted/{1bit,ternary}/L*.npz`
  (+ manifest.json). Checkpoints/tokenizer: `/home/deni/bonsai_weights/{1bit,ternary}/`.
* Golden refs: `bonsai_prep_2026_07_20/golden_{1bit,ternary}.json` +
  `_logits.npy`; per-layer golden hidden states:
  `bonsai_client_2026_07_20/e2e/golden_hidden_{1bit,ternary}.npy`.
* Sim harnesses for re-verification: `smoke_bitnet_l0.py` (BitNet
  regression), `unit_bonsai_l0.py` (exactness), `e2e_bonsai.py`
  (full-model sim), all in this directory; sim server wrapper in
  `runtime/`.
