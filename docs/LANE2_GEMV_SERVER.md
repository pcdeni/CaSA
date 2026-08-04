# Lane-2 GeMV server + llama.cpp binding design (B1 phase 2)

2026-07-18. Lane 2 only: their models, their conventions, Road-A in-DRAM
accumulation. Never blended with the BitNet production server.

2026-07-20 (task O7) — the three phase-1 fidelity deviations are CLOSED as
env-gated server modes (defaults preserve the phase-1 behavior exactly):
- `LANE2_ENCODE=clone` — §V-C RowClone-encoded on-the-fly products (the
  activation bit selects the RowCopy source; products physically created by
  clones; fastpath fused-gate machinery; all-MAJ3 dual-rail FA). 3.0× faster
  than write-load at 4096², 94.7% unvoted (commodity-DIMM MAJ noise
  compounds over the 9-gate FA; their error-free rests on Frac+calib [48] +
  module screening).
- `LANE2_DUALTRACK=1` — Fig-15 dual-track: inverted matrix planes at
  LOAD_MATRIX, ~carry/~sum formed IN-DRAM at every tree level (De-Morgan
  MAJ3 / complement-rail MAJ5); host transports rails but never negates.
  2× wall; 99.49% unvoted / 99.83% vote3 at 4096².
- `GEMV_PARTIALS` (magic 0x4D563003) — exact per-32-block integer partial
  sums (M × K/32 i32); host applies q4_0/q8_0 per-block scales → the first
  EXACT FP32 the reproduction produces (bit-exact vs the CPU fp32 reference
  on real Llama-2-7B tensors; driver lane2_partials_fp32.py).
Full A/B tables + row-budget math: REPRODUCTION.md 2026-07-20 section.

## Scope decision (honesty convention)
Their §VIII-A e2e = every mulmat_op through MVDRAM, 256 tok × 10 runs × 4
models × Q2/Q4. At our per-op rate (host-round-trip rig, no §V-E command
streaming) a single 4096×4096 4-bit GeMV is minutes — full-fidelity e2e is
multi-day per model. Reproduction shape we run instead:
- **Sampled interception**: the hook (MVDRAM_PIM=1) routes every Nth
  eligible GeMV (batch=1, q4_0/q2_K weight) to the PIM server; CPU
  computes the rest AND the same op as reference. Over a 256-token run
  every layer/projection gets PIM-sampled repeatedly; per-op outputs are
  verified against the CPU int reference each time.
- **Per-op table (B2)** at their exact dims (4096×4096, 4096↔11008,
  32000×4096) under their conventions (1000-iter avg, 50% bit-sparsity
  inputs) — measured, not sampled.
- Deviation statement in the writeup: sampled e2e + full per-op
  verification + their protocol semantics; not their wall-clock e2e
  (structural gap = §V-E streaming, documented in `RELATED_SYSTEMS.md` §2).

## Server (lane2-gemv-server)
- Location: mvdram-repro tree (NOT BitNet/). Wraps the validated Lane-2
  kernels: safe-placement computation-rows dataflow
  (test_mvdram_compute_rows_safe machinery), on-the-fly encoding
  (mvdram-gemv-exe), dual-track adder accumulation (mvdram-adder /
  popcount-indram machinery), screened columns from the Table-I-analog
  colmasks.
- Protocol (stdin/stdout, length-prefixed, mirrors the BitNet server
  style but separate magics):
  - LOAD_MATRIX 0x4D563001: {handle, q_bits, K, M, bitplane data
    row-major per §VI horizontal layout} → ack. Matrix bitplanes become
    resident weight rows sharded N≤128 outputs per subarray across
    subarrays × banks (their §VII partitioning); complements stored for
    the dual-track (their doubled-storage convention, Fig 15 later).
  - GEMV 0x4D563002: {handle, r_bits, activation bitplanes} → y[M] i32.
    Activations are NEVER written: encoded as RowCopy-or-skip per §V,
    zero-skip per §V-D. Accumulate via dual-track FA tree (Road A),
    read result planes only.
- Bit-serial math: y = Σ_i Σ_j 2^(i+j) · popcount-plane(W_i ∧ a_j) with
  sign handling per quant format (q4_0 zero-offset 8; q2_K per-block
  scales handled HOST-side after integer GeMV — document exactly which
  part is in-DRAM: the integer bitplane GeMV, matching the paper).
- Q4_0x8 repack: the shim de-repacks to plain q4_0 lanes before LOAD
  (one-time per tensor), so the server sees canonical bitplanes.

## Shim (mode 1 in mvdram-pim.c)
- Eligibility: op->src0 type in {q4_0, q2_K(later)}, ne11==1, dims in the
  census table. Env MVDRAM_PIM_SAMPLE=N (default 64): route every Nth
  eligible call; others fall through. Routed call: also compute CPU path,
  compare int results, log per-op verdict + timing to MVDRAM_PIM_LOG.
- Thread contract: intercept decides on ith==0 and BARRIERS the op via
  returning true only when the PIM result is fully written to dst (other
  threads return true immediately — ggml threads all call the op; the
  shim must make thread 0 do the work and others no-op: use the existing
  params->ith gating + ggml barrier semantics of returning without
  writing for ith != 0).

## Order of work
1. Server skeleton + LOAD/GEMV for q4_0 at 4096×4096 on bender 2 (the
   B2 dims double as the bring-up shape). Verify vs numpy on random
   data first (their P2 methodology), then real tensor slices.
2. Shim mode-1 marshaling + de-repack + sampled e2e smoke (Llama-2-7B,
   few tokens, sample rate high).
3. B2 table runs (1000-iter, 50% sparsity) on the same server.
   [2026-07-19: driver staged = b2_gemv_table.py (direct protocol, no
   llama.cpp; paper dims, qb 1–4, 50% + measured-density arms, --iters 5
   default with variance vs their infeasible 1000, --full mode, numpy CPU
   column, B2_CONFIRM gating; sim-validated, NOT yet run — ~1–1.5 h
   projected for the default table at rb=1).]
4. 256-tok sampled e2e per model as wall-time allows; then the writeup
   with the deviation statement.
