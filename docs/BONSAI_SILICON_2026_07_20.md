> **Repo note.** Verbatim working record from the rig workspace
> (`/home/deni/Claude/bonsai_silicon_2026_07_20/README.md`), preserved unedited as a citation
> target. Reader-facing synthesis: [`BONSAI_2026_07.md`](BONSAI_2026_07.md).

---

# Bonsai-1.7B (PrismML) — first silicon runs, both variants (2026-07-20)

First on-silicon execution of PrismML Bonsai-1.7B (Qwen3 architecture, 28
layers, hidden 2048, g128 group scales + sparse residuals) through the PIM
matmul path — full model, all 196 BitLinears substituted, single-DIMM
**bender 2** (DIMM 2, the production die), banks 0,1,2,3, LOAD→V2 streaming
regime, vote OFF (`PIM_VOTE_FULL=0`).

Client machinery validated host-side the same day:
`/home/deni/Claude/bonsai_client_2026_07_20/README.md` (§5 = the recipe used
verbatim here). Weights: `/home/deni/bonsai_weights/extracted/{ternary,1bit}/`.
Goldens (fp32 CPU greedy, torch 2.13/transformers 5.14 venv):
`/home/deni/Claude/bonsai_prep_2026_07_20/golden_{ternary,1bit}.json`.

## Verdict table

| run | variant | mode (prompt) | new tokens | generated text (verbatim) | vs golden greedy | s/generated-token | server calls | vote |
|---|---|---|---|---|---|---|---|---|
| 1 | ternary | raw, 7 input ids | 8 | ` The capital of France is Paris. Paris` | **8/8 exact-id prefix** of `modes.raw` — no divergence | 100.0 (799.7 s / 8) | 81,627 | OFF |
| 2 | 1bit | raw, 7 input ids | 8 | ` What is the capital of France? What` | **8/8 exact-id prefix** of `modes.raw` — no divergence | 100.8 (806.5 s / 8) | 81,627 | OFF |
| 3 | 1bit | chat, 31 input ids | 8 (eos stop; max was 10) | `The capital of France is Paris.` | **8/8 exact-id** of `modes.chat` incl. the `<|im_end|>` eos stop | 275.0 (2199.8 s / 8; 31-pos prefill dominates) | 221,403 | OFF |

All three runs are **token-exact reproductions of the fp32 golden greedy
continuation** — silicon ≥ the sim e2e expectation (sim predicted argmax
agreement at step 0 with 16/16 / 15/16 top-16 overlap; silicon delivered
the full 8-token greedy chain in every run). The 1-bit raw echo (" What is
the capital of France? What…") is the model's own base-style raw-mode
behavior — the golden does exactly the same — not a machinery defect; run 3
closes the loop by showing the same 1-bit model **answer correctly in chat
mode** on silicon. Headline: **both variants produce the correct answer on
silicon** (ternary in raw mode, 1-bit in chat mode), and the machinery is
transparent enough that the ternary-vs-1bit capability difference (answer
vs echo on the same raw prompt) is directly visible on silicon.

No degraded outputs → the vote-ON contingency arm was not needed. No
failures, no stalls, no recoveries — zero FPGA incidents across all runs.

## Environment / commands (verbatim)

All runs: `cd /home/deni/bitnet_weights`, production entry point
`run_bitnet_pim.py`, production server binary
`SiMRA-DRAM-main/.../BitNet/bitnet-proj-server` (sha256 `f1d21ee641131538…`
— byte-identical to the copy validated in sim), `PIM_SERVER_PATH` unset.
`--dimms 2` auto-wired the mandatory DIMM-2 trio (visible in the PimServer
key in each log): `calib_dimm2.txt` +
`PIM_POOL_LIST_FILE=pool_layout_dimm2_cloneok_bank{bank}.txt` +
`PIM_SUB_START=45312 PIM_SUB_END=45952`.

```bash
# run 1
BITSTREAM_IMEM=8192 PIM_NO_CHAT_TEMPLATE=1 PIM_VOTE_FULL=0 PIM_RECV_TIMEOUT_MS=15000 \
python3 -u run_bitnet_pim.py --model bonsai_ternary --dimms 2 --bank 0,1,2,3 \
    --prompt "What is the capital of France?" --max-tokens 8
# run 2: same with --model bonsai_1bit
# run 3: same as run 2 but PIM_NO_CHAT_TEMPLATE omitted (chat template ON) and --max-tokens 10
```

`PIM_NO_CHAT_TEMPLATE=1` (runs 1-2) reproduces the golden **raw** mode
(7-token prompt) — the same mode every sim validation used.
`PIM_RECV_TIMEOUT_MS=15000` armed the opt-in receive-stall guard (never
fired). `PIM_VOTE_FULL=0` = vote OFF. `PIM_FUSED_COSET` deliberately NOT
set (unexercised with Bonsai per client README §5.1). Nothing under
`SiMRA-DRAM-main/` or `/home/deni/bitnet_weights/` was modified.

Full teed logs in this directory:
`run1_ternary_voteoff.log`, `run2_1bit_voteoff.log`, `run3_1bit_chat.log`,
`preflight_rowclone_smoke.log`.

## Pre-flight (all green, 20:23-20:25)

- `lspci -nn -d 10ee:` → `0000:01:00.0 … Xilinx [10ee:9038]` (card
  enumerated; located by vendor, not BDF).
- `/dev/xdma0_*` device nodes present (driver loaded since 19:31).
- RowClone smoke, bender 2 bank 0 (60000→60016), `BITSTREAM_IMEM=8192`:
  **PERFECT_CLONE 8192/8192 at t_23 = 1,2,3,4** (`preflight_rowclone_smoke.log`).
- `calib_dimm2.txt` + `pool_layout_dimm2_cloneok_bank{0..3}.txt` all present.
- Server binary sha256 = the sim-validated copy; no lingering
  `bitnet-proj-server` before/between/after runs (pgrep-verified each time);
  no concurrent builds.

## Run 1 — ternary, raw mode (20:26:11 → 20:39:39)

- Output ids `[576, 6722, 315, 9625, 374, 12095, 13, 12095]` = golden
  `modes.raw.generated_ids[:8]` exactly (tokenizer round-trip verified
  exact). Text: ` The capital of France is Paris. Paris`.
- 8 tokens in **799.7 s** (99.96 s/generated-token); total wall incl. model
  load + substitution: 808 s.
- **81,627 server calls**, 5,291.4 MB sent, over 14 token-positions
  (7-position prefill + 7 incremental) in 8 forward calls →
  **5,830.5 requests/token-position** (sim e2e measured 5,843 — match,
  minus ENOSPC-latched LOAD probes) = **10,203 requests/generated-token**
  in this 7+8 shape.
- Round-trip bound as predicted: pipe-read 787.8 s of 792.9 s
  server-request time; server-time-implied only 3.1 s.
- LOAD pool exhausted (ENOSPC) early → per-group V2 streaming, the
  production regime; fallback-slice milestone prints at 1/100/200/300.
- Residual correction: 119,042 terms (= exactly 2× the sim e2e 7-position
  count of 59,521 — deterministic per-position), sum|Δy_int| 462.2, max
  single 0.069.

## Run 2 — 1bit, raw mode (21:41:06 → 21:54:41)

- Output ids `[3555, 374, 279, 6722, 315, 9625, 30, 3555]` = golden
  `modes.raw.generated_ids[:8]` exactly. Text:
  ` What is the capital of France? What` (the model's own raw-mode echo —
  identical in the golden).
- 8 tokens in **806.5 s** (100.8 s/generated-token); wall 815 s.
- **81,627 server calls** (identical to ternary — the request protocol is
  weight-value-independent), 5,291.4 MB, pipe-read 794.8 s. Dual-track with
  empty zero-set (`neg = ~pos`, ~50%-dense masks) on the unchanged
  client+server, as designed.
- Residual correction: 204,176 terms (= 2× sim's 102,088), sum 956.6, max
  single 0.062.

## Run 3 — 1bit, chat mode (21:57:58 → 22:34:46)

Purpose: complete the headline "both variants produce the correct answer on
silicon" — in raw mode the 1-bit base-style model echoes; in chat mode its
golden answers `The capital of France is Paris.<|im_end|>…`.

- Pre-launch check: client chat-template ids (system "You are a helpful AI
  assistant." + user prompt, add_generation_prompt) = **exactly the 31 ids
  stored in** `golden_1bit.json modes.chat.input_ids` — same template path,
  apples to apples.
- Golden chat continuation `[785, 6722, 315, 9625, 374, 12095, 13, 151645, …]`;
  position 7 is `<|im_end|>` = the model's `generation_config.eos_token_id`
  (151645), so a faithful greedy run stops at 8 new tokens by construction
  (`--max-tokens 10` cannot be reached).

Result:

- Printed response: `The capital of France is Paris.` — 8 new tokens in
  **2199.8 s**, generation stopped at 8 of 10 (eos hit). The 7 visible ids
  encode to `[785, 6722, 315, 9625, 374, 12095, 13]` =
  `modes.chat.generated_ids[:7]` exactly; the 8th emitted token is the eos
  `<|im_end|>` (151645) itself — invisible under `skip_special_tokens` but
  proven by the stop (n_new=8 < max-tokens=10 requires eos, and eos =
  151645 = golden position 7). **All 8 emitted tokens match golden: 8/8.**
  Golden positions 9-10 are unreachable by construction (the golden harness
  forced 16 tokens past eos; `model.generate` faithfully stops).
- **221,403 server calls**, 14,352.2 MB sent, 38 token-positions
  (31-position prefill + 7 incremental) → 5,826.4 requests/position.
  Pipe-read 2,166.1 s of 2,180.4 s server-request time; server-implied
  8.7 s.
- Residual correction: 554,192 terms = **exactly** 38 × 14,584 (the sim
  per-position 1bit count, 102,088/7) — third clean determinism
  cross-check of the campaign.
- 275.0 s/generated-token headline is prefill-dominated; per token-position
  it is 57.9 s — in line with runs 1-2 (57.1 / 57.6).

## Throughput observations (single-DIMM bender 2, vote OFF)

- **~100 s/generated-token** for both variants in the 7-input/8-output raw
  shape; **~57-58 s per token-position** across all three runs (57.1 /
  57.6 / 57.9 — the position-normalized number is the shape-independent
  one; run 3's 275 s/generated-token is just the 31-token prefill
  amortized over 8 outputs). Requests/token-position **~5,826-5,831
  measured** — the ~6× request amplification vs BitNet-2B at vote parity
  that the client README §5.3 predicts (group-granular g128 rescale sends
  one request per 128-input group; in-DRAM MAJ3 work per token is roughly
  unchanged — the cost is round-trip/pipe overhead, and the logs confirm:
  >99% of wall is pipe-read; server-implied compute 3-9 s per run).
- These are FIRST-RUN numbers on ONE DIMM with the correctness-first
  protocol. Future optimization arms (not exercised here, deliberately):
  dual-DIMM `--dimms 0,2` split, `PIM_FUSED_COSET` (needs a Bonsai
  `PIM_INT_DIFF=1` A/B first), and the server-side group-rescale extension
  (per-4-chunk partial response vectors / server-side scale table) sketched
  in the prep report — the lever that removes the 6× round-trip
  amplification itself.

## Honest deviations / notes

- Wall-clock s/token here includes the client-side python path (bitplane
  build, body concat ~250 s aggregate) overlapped with round-trips; the
  server-implied compute time is ~3 s — the pipe is the wall.
- The exact LOAD→V2 fallback-slice total is not printed (milestone prints
  only; last milestone 300 in runs 1-2).
- `PIM_RECV_TIMEOUT_MS=15000` was set on every run per campaign protocol
  (opt-in stall guard, fixed 2026-07-17); it never triggered.
- Runs 1-2 compare against golden **raw** mode (the mode all sim validation
  used); run 3 against golden **chat** mode. Golden = fp32 CPU greedy of the
  SAME checkpoints in the pinned venv; client-env CPU forward reproduces
  golden logits bit-exactly (client README §4), so any silicon deviation
  would be attributable to the PIM path — none was observed at the token
  level.
