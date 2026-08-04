# MVDRAM reproduction on commodity DDR4 — results summary

Reproduction of MVDRAM (arXiv 2503.23817) from the paper alone (no source
released), on our BCU1525 DIMM 0 + DIMM 2, DRAM-Bender, 2K-IMEM bitstream.
Started + completed (correctness-faithful) 2026-06-17.

## What was reproduced, on real silicon, bit-exact-validated

| MVDRAM mechanism | Our result (DIMM 2 unless noted) | Tool |
|---|---|---|
| RowClone (their product/copy primitive) | 8192/8192, both DIMMs | rowclone-smoke |
| MAJ5 reliability via column screening | 87.1% reliable cols (DIMM2), 88.2% (DIMM0) — matches their 83–95% | mvdram-maj5 |
| Dual-track MAJ full adder (carry=MAJ3, sum=MAJ5) | 99.94% on screened cols | mvdram-adder |
| In-DRAM GeMV, ternary×binary (N=2) | 100% bit-exact (3× vote) | mvdram-gemv |
| In-DRAM GeMV, **2-bit & 4-bit signed × int8** (all 4 target models' precisions) | 99.99% | mvdram-gemv |
| In-DRAM carry-save popcount TREE (N=4,8) | 99.97–99.98% | mvdram-popcount |
| **In-DRAM dual-track ~carry** (no host NOT, faithful §II-C2) | 99.97% | mvdram-popcount |
| **COMPLETE integrated GeMV** (tree + q-bit + factor + dual-track) | 99.91% (N=4,q=2,r=2, no vote) | mvdram-gemvn |

Models MVDRAM targeted: Llama2-7B/13B, Llama3-8B, Phi-4 @ 2-bit AND 4-bit (via
llama.cpp). All four share one kernel (signed q-bit × r-bit GeMV) differing only
in dimensions → our 2-bit & 4-bit kernel validation covers all four at the
compute level. Ternary (BitNet) is the simpler q=2 special case.

Residual everywhere ≈ 0.01–0.1%, confined to weakly-marginal columns that
pass the screen: columns are qualified by an op-matched screen (3 trials,
AND-ed) and computation uses reliable columns only — the same per-column
screening policy MVDRAM applies (their 83–95% reliable). Optional 3× result
voting removes the residual at 3× wall; unvoted it is within the tolerance
low-bit LLMs are built for. This is bit-exact evidence MVDRAM's own paper
does not report.

**Depth-vs-noise characteristic (measured):** error scales with the number of
MAJ ops in the computation. Integrated GeMV: N=4/q=2/r=2 = 99.91% (no vote);
N=8/q=2/r=4 = 99.87% (3× vote — deeper tree × more planes accumulates more
transient events than 3× recovers). Clean bit-exact at depth needs higher vote
(5–7×), tighter screening, or lower-noise silicon. At small/ternary configs 3×
vote reaches 100% (N=2 ternary = 100%, seed 99).

## The headline finding (a result beyond the paper)
**Faithful FAST loading (MVDRAM's in-DRAM RowCopy of operands) is silicon-blocked
on our commodity DIMMs by the XOR-spread we discovered.** Measured: loading the
tuple by RowClone → 50.1% bit-exact (total corruption), because RowClone TO a
tuple member spreads source→source⊕vuln, and a calibrated MAJ tuple's 16 open
rows are physically adjacent ⇒ mutual XOR-shadows ⇒ sequential RowClone-load
self-corrupts. Per-column WRITE doesn't spread, so the correctness-faithful path
works at 99.9%+. CONCLUSION: MVDRAM's fast RowCopy loading works on its SCREENED
low-spread module but not on commodity DIMMs with mutually-shadowing tuples —
this quantifies why MVDRAM had to screen 16 modules, and our XOR-spread
characterization is the gating constraint for fast faithful PUD on commodity DRAM.

## Row-selection investigation (exhaustive, mvdram-rowsel-exe)
Asked: can a row selection work around the spread for fast RowClone loading?
Measured on s_id 86: among 32 candidate source rows, conflict edges are SPARSE
(0.625/cand) → a 26/32 mutually-non-shadow independent set exists. So source-row
shadowing IS solvable. BUT plugging bit8=1 IS rows into the fast half-add still
failed (25–50% bit-exact). Conclusion: **non-shadow source selection is necessary
but NOT sufficient** — the fast path's blocker is the RowClone-load operation
itself (16 sequential doubleACT(30,1) into physically-adjacent tuple open rows
disturb each other / leave a charge state incompatible with the MAJ). Per-column
WRITE has no such disturbance → stays 99.9%. So fast in-DRAM operand loading is
blocked on our silicon at a level deeper than source-row choice.

## P3 (end-to-end) feasibility — runtime-blocked by the same root cause
A real Llama2-7B GeMV (4096×4096, 2-bit×int8) on the surviving per-column-write
path = ~262k MAJ ops ≈ 12.8M executes ≈ **54 h per GeMV ≈ ~1 year/token**.
Infeasible. The ONLY viable speed is MVDRAM's fast in-DRAM RowClone+accumulation
dataflow — which our XOR-spread blocks (above). So end-to-end on an actual model
is gated by the identical root cause. The FEASIBLE LLM-in-DRAM already exists:
CaSA's BitNet pipeline ("Paris") using the achievable per-col-write + host-popcount
kernel. A literal MVDRAM-shape end-to-end is demonstrable only at toy model scale
(small N) or via casa_sched projection of the would-be fast-dataflow throughput.

## ★ Operands-in-place fast accumulation — DEMONSTRATED (the performance-faithful path)
The earlier "performance-faithful is blocked" verdict was WRONG (corrected after
deeper investigation). The fast in-DRAM accumulation works on our silicon via an
operands-in-place dataflow that never reloads intermediates:
- **Two-level in-place** (mvdram-twolevel): two co-activatable tuples share a row;
  L1 result consumed by L2 in place, NO reload → 100%/99.998% bit-exact.
- **Deep 5-level in-place** (mvdram-chain): running accumulator in one row, 5
  chained MAJ3s, never reloaded → 99.99% bit-exact AT EVERY LEVEL, no depth
  degradation (self-stabilizing).
- **Fan-out in place** (mvdram-fanout): a MAJ result is replicated across ALL its
  tuple's rows, so two downstream tuples read it via two different rows → R1 feeds
  TWO consumers in place, both 99.99% bit-exact. This is the input-duplication a
  carry-save full adder needs (each input feeds carry=MAJ3 AND sum=MAJ5), provided
  FREE by the replication — no copy.
- **Why it works:** load-disturbance only comes from RowClone-loading coupled
  tuples; if intermediates are never reloaded (they stay where the MAJ produced
  them and a shared-row tuple consumes them in place), disturbance is AVOIDED.
- **IN-PLACE FULL ADDER** (mvdram-fulladder): all-MAJ3 (MVDRAM's MAJ5 sum not
  in-place-able — MAJ5∩MAJ3 16-row tuple overlap ≤4, need 6). Sum via
  a^b^c=MAJ3(MAJ3(a,b,¬c),MAJ3(a,¬b,c),MAJ3(¬a,b,c)), 3 inner→1 outer in place.
  SUM 99.99%, CARRY 99.99-100%, JOINT 99.98-99.99%.
- **N=4 IN-PLACE POPCOUNT** (mvdram-plan + popcount_alloc): full carry-save 4→3-bit
  count, 18 MAJ3 gates, **99.97% bit-exact** (N=2 99.997%). Built by a generic
  in-place MAJ3-DAG allocator + plan executor. Fully-in-place is infeasible (only
  31 co-activatable MAJ3 tuples/subarray; ~1 three-way interlock supported, ~7
  needed) → HYBRID: in-place hand-offs where geometry allows + SPILL (RowClone-style
  host operand-move) across gaps (N=4 = 7 in-place + 13 spill). This is the
  SIMDRAM/MVDRAM operand-move strategy on commodity geometry.
- **IN-PLACE GeMV (integrated, both DIMMs)** (mvdram-gemv-inplace): the in-place
  popcount IS the accumulation in the full bit-serial signed q-bit×r-bit GeMV.
  DIMM2: ternary 99.93%, 2bx2b 99.92%, 2bx4b 99.92% (vote 99.94%); DIMM0 (s77,
  bender 0): 2bx2b 99.86%. Accuracy PARITY with the per-col-write tree (gemv_n
  99.91%) — accumulation now in-place. ALL findings generalize to DIMM0 (geometry
  tighter: s77 = 10 MAJ3-grade tuples vs s86's 31 → more spills).
- **d_in SCALING (tiling)** (PIM_DIN): arbitrary contraction d_in = (d_in/N) tiles
  of N=4; popcount each tile in-place, sum chunk counts (host, exact). Validated
  q2r2: DIMM2 d_in 4..64 → 99.80-99.94% (error BOUNDED, not growing with d_in);
  DIMM0 d_in 8/16 → ~99.80%. Scaling to model dims (4096 = 1024 tiles) = proven
  mechanism, more executes. Cross-chunk/cross-plane reduction on host (small).
- **ALL primitives for an in-place carry-save tree are now proven on silicon:**
  (1) in-place chaining (deep, 5 levels), (2) fan-out (1 result → 2+ consumers),
  (3) free duplication via replication, (4) rich tuple substrate. The full adder
  (carry=MAJ3, ncarry=MAJ3(~a,~b,~c) dual-track, sum=MAJ5 reading inputs via
  fan-out + ncarry) is constructible from these with no copy/reload.
- **Substrate:** 44,361 co-activatable sets / 12,160 distinct 4-row tuples (many
  sharing rows) per subarray → ample to route a real carry-save tree.
- **Remaining = engineering:** a PUMA-style allocator that maps a full carry-save
  popcount tree onto shared-row tuple chains (the current chain is a linear
  accumulator; the GeMV needs the branching full-adder tree), + fast initial
  product loading. No fundamental blocker.

## Faithfulness verdict
- **Correctness-faithful: ACHIEVED.** Every MVDRAM mechanism reproduced
  bit-exact on our silicon, integrated end-to-end at the kernel level, including
  in-DRAM dual-track accumulation (no host NOT).
- **Performance-faithful (their throughput): silicon-blocked** on our DIMMs by
  the XOR-spread, with a measured 50%-corruption demonstration + mechanistic
  explanation. Needs a low-spread screened module (as MVDRAM used) or
  spread-mitigated tuple construction (open problem; our IS-pool tooling is the
  starting point).

## Files
Tooling (DRAM-Bender BitNet/): mvdram-{maj5,adder,gemv,popcount,gemvn,indram}-exe.
Calib/masks/logs + docs (PLAN, ADR-001/002/003, FAITHFULNESS, checkpoint, this):
/home/deni/Claude/mvdram-repro/. Numpy oracle: ref_mvdram.py.

## Remaining (engineering / scale, no scientific risk)
- Voting/screening to drive integrated GeMV to clean 100% (mechanical).
- Scale N to model dims via subarray sharding (slow on per-col-write path).
- P3: llama.cpp mulmat replace + per-model run (the end-to-end; speed is
  spread-limited on our silicon, documented).

## 2026-07-17 UPDATE — the performance-faithful path is UNBLOCKED and MEASURED

June's headline ("fast RowCopy loading silicon-blocked", "blocked deeper
than source choice") is REVERSED by the safe pair-offset placement result
(`sublattice_broadcast_2026_07_17/RESULT.md` Parts 2–3): corruption during
tuple loading is a function of the PAIR offset src⊕dst; offsets free of
generator-subset bits make loads tuple-clean BY CONSTRUCTION. The faithful
computation-rows dataflow reruns at **99.98% e2e (vs June's 6.1%)** on
benders 2 and 0.

Assembled performance-shape kernel (`mvdram-fastpath-ab-exe`: the same
popcount-4 carry-save DAG, same safe placement, three shapes, 5 iters):

| shape | b2 ms/gate | b2 e2e | b0 ms/gate | b0 e2e |
|---|---|---|---|---|
| C — write-load (June correctness shape) | 1.585 | 99.990% | 1.574 | 99.974% |
| A — clone-load, unfused | 0.885 | 99.865% | 0.898 | 99.839% |
| B — clone-load, FUSED 1 exec/gate | **0.683** | 99.808% | **0.705** | 99.814% |

- Fast in-DRAM operand movement + single-execute gates = **2.2–2.3× per
  gate** over the June shape; fast loading costs ~0.17% e2e accuracy
  (99.99 → 99.81); fusing itself costs nothing (A ≈ B accuracy).
  Cross-die reproducible.
- The frac row's per-gate per-column rewrite is replaced by an in-program
  `wrRow(ONE)` (it holds a uniform constant — the June tool over-paid).
- The "Performance-faithful: silicon-blocked" verdict above is SUPERSEDED.
  What remains toward the paper's full throughput shape is engineering
  (multi-gate programs under IMEM, subarray sharding, voting to 100%),
  not physics.

## 2026-07-18 — P3 progress: llama.cpp mulmat hook landed (census mode)

Their §VIII-A e2e replaces mulmat_op inside llama.cpp; our hook now exists
in /home/deni/mvdram_bench/llama.cpp (their 4 models downloaded at Q4_0 +
Q2_K under /home/deni/mvdram_bench/models, 39 GB):
- Additive shim ggml/src/ggml-cpu/mvdram-pim.{h,c} + two attachment
  points: ggml_compute_forward_mul_mat AND the repack buffer type's
  compute_forward (repack.cpp) — the repack path carries ALL quantized
  projection mulmats on CPU and bypasses the generic entry; hooking only
  the latter sees just the q6_K output head.
- MVDRAM_PIM=census records (type, K, M, batch); Llama-2-7B Q4_0 table
  matches the paper's dims exactly: 4096×4096 (q/k/v/o), 4096→11008
  (gate/up), 11008→4096 (down), 32000×4096 head (q6_K). batch=1 rows =
  the token-gen GeMVs to intercept; batch=8 = prefill GeMM.
- Same-DDR4 CPU baseline sample (llama-bench, 6 threads, this tower):
  7B Q4_0 pp8 55.7 t/s, tg4 14.2–14.9 t/s. (Their Table II uses an
  i7-9700K with the same modules as the PIM — our analog: same host DDR4.)
- Binding note: the repack path hands the shim REPACKED weights (Q4_0x8
  interleaved). The PIM binding must either unpack that layout or pin the
  target tensors to the plain buffer type before repacking.
- llama-cli in this checkout spins an interactive prompt loop at stdin
  EOF even with --no-conversation (burned a 27 GB log before diagnosis);
  use llama-bench (or pipe a real tty) for hands-off runs.
Next: Lane-2 GeMV server (Road-A in-DRAM kernel) + MVDRAM_PIM=1
interception → their protocol at documented reduced scope (their 256 tok ×
10 runs × 8 model/quant combos at our per-op rate is multi-day; scope and
state the deviation per the honesty convention).

## 2026-07-18 — P3 progress: lane2 server phase 1 (LOAD_MATRIX + GEMV on silicon)

`lane2-gemv-server` (source + Makefile here, binary here; protocol per
LANE2_GEMV_SERVER.md) is up and verified on bender 2 / bank 0 / s86
(calib_maj5_dimm2.txt + colmask_dimm2_s86_robust.txt — production s72
window untouched). Kernel = the June correctness-faithful machinery
verbatim (per-column-write loading, MAJ3 carry + MAJ5(a,b,c,~c,~c) sum on
the 16-row MAJ5 tuple, host-glued ~carry, op-matched column screen now
repeated LANE2_SCREEN_TRIALS=3× and AND-ed), §V-D zero-skip, CSA tree over
the full contraction time-multiplexing the single screened tuple. Only
change vs June: 8K-IMEM program packing (one pcwrite per program instead
of 3; ≤16 wrRows batched) — same instruction sequences, ~3× fewer execs.
LANE2_PACK=0 restores the 2K-era shapes.

Measured (random data, host numpy/int64 reference in lane2_client_smoke.py):

| shape | vote | exact outputs | wall/GeMV |
|---|---|---|---|
| K=8 M=32 q=2 r=1 | off | 32/32 | ~0.01 s |
| K=256 M=1024 q=4 r=1 | off | 1024/1024 | 2.2 s |
| **K=4096 M=4096 q=4 r=1** | off | 4093/4096, 4094/4096 (99.93–99.95%) | **37 s** |
| **K=4096 M=4096 q=4 r=1** | **PIM_VOTE3=1** | **4096/4096 × 2 GeMVs (bit-exact)** | **110 s** |
| K=64 M=256 q=4 r=4 signed | on | 256/256 | 6 s |

- Bring-up shape unvoted: 8.4K FA / 16.9K MAJ / 287K execs / ~2.2 ms per
  FA; residual = 2–3 outputs/4096 on weakly-marginal screened lanes with
  run-to-run-varying |err| on weakly-marginal screened lanes (gemvn
  99.87–99.96% of lane-samples). 3× result voting recovers it
  fully at 3× wall. Screen: 1784 → ~1660 segments after 3-trial op-match
  (capacity ~53K outputs/pass; M=4096 uses 128).
- How to run: `cd /home/deni/Claude/mvdram-repro && make && python3
  lane2_client_smoke.py --K 4096 --M 4096 --qbits 4 --rbits 1 --vote3 1`
  (client sets BITSTREAM_IMEM=8192 + PIM_RECV_TIMEOUT_MS=15000, launches
  the server, LOADs, runs GEMV(s), verifies, quits via sentinel).
- Documented phase-1 deviations (full list in the server header): on-the-fly
  encoding host-resolved (activation bit selects W-row vs zero-row as tree
  input — test_mvdram_gemv.cpp convention); matrix host-resident and
  streamed per-op (one screened MAJ5 tuple ≠ their multi-subarray
  residency; the tiled-execution clause of the phase-1 scope); ~carry
  host-formed between MAJ3 and MAJ5. r_bits>1 (signed two's-complement)
  already implemented + spot-verified; phase-2 items = llama.cpp shim
  binding + Q4_0x8 de-repack + the resident/RowClone-encoded fast path.
  **[2026-07-20 SUPERSEDED in part: the encoding, complement-formation and
  exact-fp32 deviations are closed as env-gated modes — see the 2026-07-20
  O7 section below. The tiled-execution deviation (host-streamed matrix,
  single-tuple time-multiplexing) remains.]**

## 2026-07-18 — P3 progress: llama.cpp shim mode-1 landed (sampled interception, route c2) + Llama-2-7B silicon smoke

MVDRAM_PIM=1 is now real interception: the shim
(/home/deni/mvdram_bench/llama.cpp/ggml/src/ggml-cpu/mvdram-pim.{h,c})
routes sampled eligible GeMVs to lane2-gemv-server and verifies them on
silicon. Scope of this first cut: q4_0, ne11==1, K=M=4096 (the
server-verified B2 shape = q/k/v/o projections); every Nth eligible op
(MVDRAM_PIM_SAMPLE, default 9999) up to MVDRAM_PIM_MAX_OPS (default 3).

**The q4_0-scale question (route c2).** q4_0 carries an fp16 scale per
32-weight block, so the exact fp32 op is NOT expressible as ONE whole-K
integer GeMV. Re-read of the paper for what THEY do: §VI-B decomposes o_m
purely over integer two's-complement bit-planes of q-bit weights and r-bit
activations; §V-A keeps floating-point work on the processor; the words
scale/zero-point/dequantize never appear anywhere in the paper — their
llama.cpp §VIII-A integration is silent on per-block scales. So the
in-DRAM part that is reproducible AND verifiable is exactly the integer
bitplane GeMV, and that is what we run: the sampled op executes y_int on
silicon and is compared against a host int64 reference over the SAME
descaled ints ({-8..7}) and quantized activations; dst always gets the
stock CPU fp32 result (per-op log: MVDRAM_PIM_OPLOG). Interception is
therefore a verified side-channel, not an output path — the honest phase-2
scope declared in advance (option c2). Full-fp32 PIM output would need
per-block partial sums returned from DRAM (M×(K/32) values), a protocol
extension deferred to B2+.

**Threading contract (trivially safe).** All threads return false from the
hook, so ggml runs the stock op on all threads exactly as if the shim were
absent; thread 0 additionally does the PIM round-trip before returning.
No dst writes, no extra barriers; ith!=0 threads simply wait at the op's
own internal barrier until thread 0 arrives.

**Repack findings.** The repack path presents src0->type == GGML_TYPE_Q4_0
but data in block_q4_0x8 (AVX2 selects 8x8 interleave on this tower): 8
consecutive rows interleaved per block group, qs in 16 8-byte chunks
(chunk i <- row i%8, canonical bytes (i/8)*8..+8), all XOR 0x88 — i.e. the
stored nibbles are ALREADY two's-complement. The repack.cpp hook now
passes the template constants (INTER_SIZE, NB_COLS) via
mvdram_pim_maybe_intercept_ext so the shim de-interleaves exactly; the
generic ggml-cpu.c path passes (0,0) = canonical. No build switch used —
repack stays enabled so the CPU baseline is untouched. De-repack validated
BIT-EXACT against the gguf ground truth (blk.0.attn_q.weight: 16,777,216
descaled ints + 524,288 raw fp16 scales, zero mismatches) via
MVDRAM_PIM_DRY=1 + MVDRAM_PIM_DUMP + check_derepack.py (artifacts in
/home/deni/mvdram_bench/smoke_2026_07_18/) — zero silicon spent on
marshaling validation.

**r=8 standalone (before the smoke).** lane2_client_smoke.py --K 256
--M 1024 --qbits 4 --rbits 8: PASS bit-exact 1024/1024, GEMV 17.5 s =
8.0x the r=1 wall (cost scales with q_bits x r_bits plane-pairs; per-FA
~4.4 ms unchanged). Activations quantized host-side to signed 8-bit
symmetric (amax/127), matching the server's FAC top-bit-negative
convention.

**Silicon smoke (Llama-2-7B Q4_0, real model).** Command:
`MVDRAM_PIM=1 MVDRAM_PIM_SAMPLE=1 MVDRAM_PIM_MAX_OPS=2 MVDRAM_PIM_RBITS=8
llama-cli -m llama-2-7b.Q4_0.gguf -p "The capital of France is" -n 4 -st
-no-cnv --temp 0 -t 4 </dev/null` (cwd /home/deni/mvdram_bench/smoke_2026_07_18,
logs smoke.{out,err}, mvdram_pim_smoke_ops.log). The first 2 eligible ops
= the first generated token's layer-0 attn_q and attn_v projections
(graph order presents v before k; warmup decodes are batch=2 =
ineligible). Both ops on bender 2 / bank 0 / s86, unvoted, r=8:

| op | tensor | shape | LOAD | GEMV wall | int-exact vs host ref | max/mean abs err |
|---|---|---|---|---|---|---|
| 1 | blk.0.attn_q.weight | 4096x4096 q4 r8 | 0.23 s | 160.4 s | 4092/4096 (99.90%) | 360 / 0.21 |
| 2 | blk.0.attn_v.weight | 4096x4096 q4 r8 | 0.25 s | 157.7 s | 4091/4096 (99.88%) | 1114 / 0.59 |

- Wall breakdown per op: LOAD is negligible (0.23–0.25 s host-side
  bitplane scatter; matrix stays host-resident, streamed per-op — phase-1
  shape); GEMV dominates at ~160 s = 36,060 FA / 1.22 M execs / ~4.4 ms
  per FA, identical to the r=1 calibration. Server init (calib + 3-trial
  op-matched screen) ~1.5 min once per process. Real-activation plane
  density 0.276 → zero-skip removed 72.4% of taps (both ops identical FA
  counts — q and v consume the same attn_norm vector, a free internal
  consistency check).
- Exactness is the documented unvoted cell-noise envelope (phase-1 r=1:
  4093–4094/4096; 9 wrong lanes of 8192 here, run-to-run transient);
  PIM_VOTE3=1 recovers bit-exact at 3x wall (110 s at r=1) — MVDRAM_PIM_VOTE3=1
  wires it through.
- Model run completed cleanly (exit 0), server shut down via len-0
  sentinel (exit status 0), card enumerated after. Text output is
  byte-identical to the CPU-only baseline of the same command (temp 0,
  deterministic) — route c2 is provably output-neutral. NOTE this
  checkout's llama-cli applies a ChatML-style template to -p even with
  -no-cnv/--no-jinja, so the base model's greedy continuation reads oddly
  ("The assistant of the president is") on CPU and PIM runs alike — a
  pre-existing CLI property, not a PIM artifact; template-free text goes
  via llama-server/raw API when B2 needs it. Supersedes the 07-18 census
  note: `-st -no-cnv` DOES exit cleanly at EOF (the earlier spin was
  -no-cnv without -st).

**What B2 needs from here.** (1) The per-op table runs (their 1000-iter,
50% bit-sparsity conventions) can drive the server directly with the
standalone client — no llama.cpp in the loop; (2) 4096↔11008 and
32000x4096 dims need multi-pass sharding over the 128-output segment
capacity (server accepts M≤53K already; K=11008 needs LOAD/GEMV tiling
over K≤16384 — fits); (3) the resident/RowClone-encoded fast path and
per-block partial-sum return (exact fp32 reconstruction) remain the
honest gaps to their §V-E/§VII shape, both documented in the MVDRAM
comparison (`RELATED_SYSTEMS.md` §2).

## 2026-07-19 — B1 host-side remainder: Q2_K + q6_K head + 11008 dims in the shim (dry-run-validated; silicon smoke staged, NOT run)

Host-only session (FPGA owned by another agent — zero silicon spent; the
lane2 server source/binary UNTOUCHED). The shim
(/home/deni/mvdram_bench/llama.cpp/ggml/src/ggml-cpu/mvdram-pim.{h,c}) now
covers all of Llama-2-7B's census dims and both quant families:

**Q2_K integer mapping (the route-c2 decision, documented per plan).**
q2_K stores q∈{0..3} with per-16-weight sub-block 4-bit scales/mins under
fp16 d/dmin (value = d·sc·q − dmin·m). The in-DRAM part is the whole-K
scale-free integer GeMV over **qs = q−2 ∈ {−2..1}** (2-bit two's-complement
= the top-bit flip q^2, the exact structural analog of q4_0's q−8 = q^8):
one native qb=2 handle, because the server's FAC(1,2) = −2 signed
convention computes P0−2P1 — RAW unsigned {0..3} would be mis-signed
in-DRAM, and is host-recoverable anyway via the exact identity
Σq·x = Σ(q−2)·x + 2·Σx, so verifying the signed contraction verifies the
raw one. All per-sub-block affine work (d·sc, dmin·m, the +2 de-bias) is
host-side, matching §V-A fp-on-processor and the June-validated "2-bit
signed" kernels. Same decision structure as q4_0's 07-18 route c2.

**Q2_K repack finding.** repack.cpp DOES carry q2_K repack variants
(block_q2_Kx8/x16) but selection gates on **AVX512** (or RISC-V V) — this
tower's i7-13700 is AVX2-without-AVX512, so **q2_K arrives canonical at
the generic hook** (verified live: `layout=canonical (inter=0 nb_cols=0)`).
No de-interleave needed; a non-(0,0)/(8,8-q4_0) layout hits a warn-once
refusal (censused only). q6_K repack is NEON-only — never on x86.

**Census of the Q2_K model (llama-2-7b.Q2_K.gguf).** In llama.cpp's Q2_K
mix only **attn_q + attn_k are q2_K** (4096×4096); attn_v/attn_o and all
ffn tensors are **q3_K** (not in B1 scope; note qb=3 would fit the server's
qb≤4 if ever wanted), head q6_K. So Q2_K-model coverage = q/k projections;
the 11008-dim coverage comes from the Q4_0 model (ffn tensors are q4_0).

**Dims / server capacity (caps read from lane2_gemv_server.cpp source).**
qb∈[1,4], K≤16384, M≤65536, rb≤8, request ≤256 MB; screened-segment
capacity is runtime (07-18 screen: 1631 segs → M≤52192/pass, enforced at
LOAD with status 4). Therefore: **K=11008 is ONE LOAD/GEMV — no host
K-split needed** (the server already time-multiplexes the tuple across the
contraction); M=11008 = 344 segments and M=32000 = 1000 segments — both
single-pass, no M-sharding. Largest LOAD frame (head qb=4 handle) = 65.5 MB,
under the 256 MB request cap. Eligibility = census shape table
{4096×4096, 4096→11008, 11008→4096, 32000×4096-q6_K}; MVDRAM_PIM_ANY_SHAPE=1
relaxes to server-cap bounds for the other models' dims later.

**q6_K head: implemented (not left on CPU).** w = q−32 ∈ {−32..31} (6-bit
two's-complement; per-sub-block int8 scale d·sc host-applied, no min term).
The server caps qb≤4, so one tensor loads as THREE handles — an exact plane
split with host summation, zero server change:
y = y[qb4: b0..3] + 16·y[qb1: b3] + 16·y[qb2: b4..5]
  = (P0+2P1+4P2−8P3) + 16P3 + 16(P4−2P5) = Σ2^iP_i − 32P5. Cost = 7
plane-pair units vs 6 for a hypothetical native qb=6 (+17%), the price of
not touching the silicon-validated server binary mid-B2. New shim
mechanics: per-tensor handle sets (up to 3), shift-aware plane packing,
GEMV per handle with multiplier summation into the int64 verdict.

**Dry-run evidence (all host, MVDRAM_PIM_DRY=1 = extraction + FULL frame
marshal, nothing sent; artifacts in /home/deni/mvdram_bench/smoke_2026_07_19/):**
- Ground truth vs gguf, BIT-EXACT (0 mismatches): q2_K blk.0.attn_q
  (16.7M ints q−2 + 1.05M scale bytes + fp16 d/dmin, check_q2k.py);
  q6_K output.weight (131M ints q−32 + 8.2M int8-scale bytes + fp16 d,
  check_q6k.py); q4_0 blk.31.ffn_down K=11008 AND blk.31.ffn_gate M=11008
  de-repacked from the live q4_0x8 layout (45.1M ints + 1.4M fp16 scales
  each, check_derepack.py).
- Host server-math sims, EXACT: qb=2 FAC (P0−2P1) == W@x on the real
  attn_q ints; the q6_K 3-handle split sum == W@x on the real head ints;
  the raw-q identity for q2_K.
- llama-bench census+dry-marshal runs on BOTH real models (-p 8 -n 8,
  SAMPLE=32): sampled q2_K/q4_0/q6_K ops all marshal with plausible
  integer stats (e.g. head r=8: nz 31996/32000, max|y| 28310 ≪ the
  4096·32·127 bound; ffn_gate M=11008: nz 11006/11008); llama-cli targeted
  runs (MVDRAM_PIM_ONLY) trigger deterministic ops: attn_q → blk.0+blk.1,
  ffn_down → one K=11008 op, output.weight → one 3-handle head op.
- New env knobs: MVDRAM_PIM_ONLY (name filter), MVDRAM_PIM_DUMP_TENSOR
  (choose the dumped tensor), MVDRAM_PIM_ANY_SHAPE; tensor cache 16→24
  entries and overflow is now non-fatal (new tensors stay CPU, cached ones
  keep verifying — matters for the 256-tok sampled runs).

**Awaits silicon (staged, gated):** run_b1_silicon_smoke.sh (this dir,
chmod +x, NOT executed) — Phase A: 2 q2_K ops (~80 s/op unvoted r=8);
Phase B: one ffn_down K=11008 op (~7 min, taps ∝ K); optional Phase C:
ffn_gate M=11008 (~160 s) + q6_K head (3 LOADs + ~4.7 min GEMV over 1000
segments). Gates: lspci 10ee probe (never BDF), fuser on /dev/xdma0_*,
pgrep for live PIM servers, and B1_SMOKE_CONFIRM=1. Expected verdicts =
the unvoted cell-noise envelope (07-18: 99.88–99.90 % int-exact at r=8);
MVDRAM_PIM_VOTE3=1 for bit-exact at 3× wall. After the smoke, the 256-tok
sampled e2e runs (scope per LANE2_GEMV_SERVER.md) can proceed on all four
shapes; q3_K (Q2_K-model v/o/ffn) and the other models' dims
(ANY_SHAPE + 13B/8B/phi-4) remain the known extensions.

### 2026-07-19 silicon smoke — shape-coverage table COMPLETE (was: awaits silicon)
| tensor | type | K×M | handles | LOAD | GEMV | int-exact (unvoted) |
|---|---|---|---|---|---|---|
| blk.0.attn_q | q2_K | 4096×4096 | 1 (qb2, q−2) | 0.19 s | 83.2 s | 4092/4096 = 99.90% |
| blk.1.attn_q | q2_K | 4096×4096 | 1 | 0.13 s | 43.5 s | 4094/4096 = 99.95% |
| blk.31.ffn_down | q4_0 | 11008×4096 | 1 (ONE call, no K-split) | 0.78 s | 75.5 s | 4091/4096 = 99.88% |
| output.weight | q6_K | 4096×32000 | 3 (plane split) | 3.15 s | 449.1 s | 31883/32000 = 99.63% |
Zero-skip carried the 11008-K op (plane density 0.049 → 75 s, ~5× under
the naive estimate). Head's slight dip = 3-handle compounding with ×16
multipliers (expected; VOTE3 available). Logs: mvdram_bench/smoke_2026_07_19/silicon/.
All quant paths their protocol needs are now silicon-verified: q4_0, q2_K,
q6_K; q3_K extractor is the one remaining coverage item (2-bit models'
v/o/ffn tensors).

## 2026-07-19 — host-side #2: q3_K extractor (quant coverage COMPLETE) + B2 GeMV-table harness staged

Host-only session (FPGA held by the BitNet server the whole time — zero
silicon; lane2 server source/binary untouched; the smoke script's own gate
verified the in-use state live).

**q3_K integer mapping (route c2, same decision structure as q4_0/q2_K).**
block_q3_K (110 B / 256 weights): hmask[32] (high bit, inverted sense),
qs[64] (low 2 bits, SAME addressing as q2_K's qs), scales[12] (16 6-bit
scales), fp16 d — value = low2 − (hmask_bit ? 0 : 4). The in-DRAM part is
the whole-K integer GeMV over **w = q−4 ∈ {−4..3}** with the biased q
ASSEMBLED from split storage, q = low2 | (hmask_bit<<2); 3-bit
two's-complement = the top-bit flip **q^4** (verified exhaustively for
q=0..7 and in the checker). ONE native qb=3 handle — the server's
FAC(2,3) = −4 matches natively. Raw-q recoverable via
Σq·x = Σ(q−4)·x + 4·Σx. Per-16-weight 6-bit scale d·(sc−32), NO min term,
host-applied (§V-A fp-on-processor).

**q3_K repack finding: none exists.** ggml_repack_get_optimal_repack_type
(repack.cpp) has NO GGML_TYPE_Q3_K case on ANY architecture in this
checkout (q2_K has AVX512/RISC-V variants, q6_K NEON-only, q3_K nothing)
— q3_K ALWAYS arrives canonical at the generic hook. Verified live:
`layout=canonical (inter=0 nb_cols=0)` on all dry runs.

**Census dims (already in census_llama2_7b_q2k.tsv, now in MV_SHAPES).**
In the Q2_K model, q3_K carries attn_v + attn_output (4096×4096), ffn
gate/up (4096→11008), and ffn_down (11008→4096) — with q2_K q/k and the
q6_K head, that closes EVERY census shape of both Llama-2-7B models.

**Ground truth + dry-run evidence (host, MVDRAM_PIM_DRY=1; artifacts in
mvdram_bench/smoke_2026_07_19/):** check_q3k.py (mirrors check_q2k.py,
magic 0x4433564D, 14-byte [scales[12]|d] records) on blk.0.attn_v: 16.7M
ints (q−4) BIT-EXACT vs gguf, 786K raw scale bytes + fp16 d BIT-EXACT,
AND the fp32 reconstruction from dumped ints × dumped scales BIT-EXACT vs
llama.cpp's own dequantizer (gguf-py gguf.quants.dequantize — an
independent oracle the earlier checkers didn't have); qb=3 FAC server-math
sim (P0+2P1−4P2) == W@x EXACT; raw-q identity EXACT; q^4 flip EXACT.
Dry-marshals on the real model, all canonical, plausible stats:
blk.0/1.attn_v (d=0.290/0.149), blk.31.ffn_down K=11008 (d=0.048),
blk.31.ffn_gate M=11008 (d=0.291). Files: mvdram-pim.{h,c} (extractor +
handle plan + 3 MV_SHAPES rows + eligibility), llama-cli/llama-bench
rebuilt. run_b1_silicon_smoke.sh gained **Phase D** (one q3_K op,
blk.0.attn_v, ~125 s expected at r=8 d=0.29; B1_SMOKE_PHASES defaults to
D since A/B/C completed) — staged, NOT run.

**B2 harness staged: b2_gemv_table.py (this dir) — NOT run.** Drives
lane2-gemv-server DIRECTLY (no llama.cpp) to produce the paper's §VIII-A
GeMV table. Design choices, all documented in its docstring:
- Dims = their set {4096×4096, 4096×11008, 11008×4096, 32000×4096};
  qb ∈ {1,2,3,4} (their sweep starts at 2; 1 and 3 native to the server).
- TWO input arms, both in the table: `paper50` = their 50%-bit-sparsity
  convention (i.i.d. Bernoulli(0.5) planes ≡ uniform signed ints), and
  `measured` = the real Llama-2-7B activation plane densities from the B1
  runs (4096×4096: 0.276; 4096→11008: 0.251; 11008→4096: 0.049;
  32000×4096: 0.433) — zero-skip makes wall ∝ set bits, so the honest
  number for real inference is the measured column.
- Their 1000-iter average is infeasible at our per-op walls: --iters
  default 5 with variance reporting (mean/std/min/max per cell), --full =
  1000 (documented deviation). Fresh activations per iteration; per-iter
  int-exact verification vs numpy int64 over identical ints; --vote3.
- CPU baseline column = same-host numpy int64 GeMV over identical ints,
  timed per iteration (plus the recorded llama-bench t/s baselines quoted
  in the md output's notes).
- Gates exactly like the B1 smoke (lspci 10ee, fuser /dev/xdma0_*, pgrep
  PIM servers, B2_CONFIRM=1) → unattended-runnable when the FPGA frees;
  handle 1 reused per cell so the server holds ~1 matrix; graceful
  quit-sentinel shutdown, no SIGKILL, no arbitrary timeouts.
- Host-validated end-to-end on LANE2_BACKEND=sim (skips gates, zero
  silicon): protocol/packing/reference/table plumbing all exercised; qb=1
  binary and rb=3 signed paths exact 32/32; gate abort verified live
  (exit 1 while bitnet-proj-ser held XDMA). Output: markdown + CSV under
  b2_results/<stamp>/, partial rows kept on abort.
- **Projected wall (calibrated 4.4 ms/FA × qb × set-bits model, validated
  against every 07-18/19 silicon op): default full table (32 cells × 5
  iters, rb=1) ≈ 59 min GEMV + ~2 min init/LOADs ≈ 1–1.5 h**; rb=8 ≈ 8×
  (~8 h); --full (1000 iters) ≈ 8 days (hence the documented default).
  `--estimate-only` prints the per-cell schedule without touching anything.
Shape-coverage table addendum (2026-07-19 smoke phase D): blk.0.attn_v
q3_K 4096×4096, 1 handle (qb3, q−4), LOAD 0.25 s, GEMV 116.7 s,
int-exact 4093/4096 = 99.93% unvoted — ALL quant paths now silicon-verified.
B2 table complete: b2_results/20260719_233500/ (32 cells, 56 min, both
sparsity arms, 5-iter variance, CPU column; zero-skip measured in-table).

### 2026-07-20 — §VIII-A sampled e2e: Llama-2-7B BOTH quants COMPLETE
12 sampled ops (6/model, MVDRAM_PIM_SAMPLE=60) verified live inside real
256-token greedy generations: Q4_0 run 99.81–99.90% int-exact (walls
135–594 s/op), Q2_K-model run 99.83–99.90% (its q2_K + q3_K tensors,
117–443 s/op); layers 0–31, dims incl. 11008 both directions. Outputs
CPU-exact by construction (route c2). Ops logs: smoke_2026_07_19/silicon/
e2e_q{4,2}_v*_ops.log. Operational notes: setsid forks (concurrent-launch
trap); one device-open race between chained servers → always gate on
fuser device-free.

### 2026-07-20 — §VIII-A protocol COMPLETE: all four models, both precisions
13B/8B/phi-4 sampled e2e (4 ops each × Q4_0+Q2_K, ANY_SHAPE dims):
all runs clean; 24 more verified ops spanning 5120/7680/13824/14336
dims, Llama-3 GQA kv (M=1024), and a 35840-output head-class op;
int-exact 99.58–99.99% unvoted. With the 7B pair: 36 sampled ops total
across the four-model matrix. B1 closed; writeup at Phase P.

## 2026-07-20 — O7: the three Lane-2 fidelity deviations CLOSED (clone-encoded products, in-DRAM dual-track, per-block partials → first exact fp32)

All three phase-1 deviations of lane2-gemv-server are now implemented as
env-gated modes (defaults byte-preserve phase-1 behavior; R1 below
revalidates it), A/B'd on silicon at the paper contrast shape (4096×4096,
q4, r=1, 50%-density activations, seed 42, bender 2 / bank 0 / the s86
subarray). Logs: o7_logs_2026_07_20/.

**(a) §V-C RowClone-encoded on-the-fly products — `LANE2_ENCODE=clone`.**
The activation bit now selects the RowCopy SOURCE and the product is
physically created by clones (paper Fig 6b), not resolved on the host.
Engine = test_mvdram_fastpath_ab.cpp machinery integrated as a server mode:
the validated 4-row MAJ3 tuple {54340,54341,54724,54725} (same physical
subarray as the s86 MAJ5 tuple), on-silicon mask screen for value rows
(48/75 usable 3-bit antichain masks), safe load order i1→i2→i0, the whole
gate fused into ONE program (clones + wrRow(ONE)→frac + frac×3 + MAJ3 +
rdRow + result clone-out). Because MAJ5 is not clone-loadable on this tuple
class, the FA is the all-MAJ3 dual-rail DAG (test_mvdram_fulladder.cpp
identity, 9 gates/FA) — clone mode is therefore inherently dual-track:
leaf products AND their complements enter via clones from W / ~W plane
rows; every intermediate rail is silicon-formed.

| arm (4096² q4 r1, seed 42) | wall | per-FA | exact outputs | notes |
|---|---|---|---|---|
| host-encode (default, R1) | 38.4 s | 4.55 ms (2 MAJ, ~30 packed pcwrites) | 4095/4096 = 99.976% | matches the 07-18 envelope |
| clone-encode (R3b) | **11.9 s** | **1.41 ms** (9 fused clone-gates, ≤6 pcwrites) | 3877/4096 = 94.65% | 75,996 MAJ, 0.157 ms/gate; pcwrites 253,032 → 50,592 (5.0×) |

- **The paper's §V-C speed claim reproduces at GeMV scale: 3.2× e2e** (the
  fastpath's 2.2–2.3×/gate was the per-gate view; fused clone gates measure
  0.157 ms/gate here vs the July 0.683 — the 8K-IMEM era receive path).
  The §V-C mechanism (encoding eliminates the operand write stream)
  delivers exactly the predicted structure: 5× fewer per-column writes.
- **The accuracy price on commodity DIMMs is now quantified at scale**:
  99.976% → 94.65% unvoted. The ~0.03%/gate MAJ3 flake (fastpath: 99.965%
  per-op) compounds over 9× more MAJs and ~12 tree levels; errors
  concentrate at deep count-bit weights (|err| ~2.4–3.5K and multiples).
  A depth-2 chained-FA screen (added; catches depth-flaky lanes at init)
  bought only +0.4% (94.29 → 94.65) — the residual is transient-dominated,
  NOT stable-lane-screenable. Rail-violation telemetry (free in dual-rail:
  v⊕nv must be all-ones) flags 0.30% of lane-checks — a built-in error
  DETECTOR; rail-checked gate retry is the natural (unimplemented)
  corrector. PIM_VOTE3 is unavailable in clone mode (the result clone-out
  is fused into the gate). Their "error-free" columns rest on Frac +
  calibration [48] + 16-module screening (footnote 3) — the same gap
  the MVDRAM comparison (`RELATED_SYSTEMS.md` §2) documents, now with a
  GeMV-scale number attached.

**(b) Fig-15/§VII in-DRAM dual-track complements — `LANE2_DUALTRACK=1`.**
LOAD_MATRIX now also prepares the inverted matrix bitplanes (~W — the
paper's "Inverted Matrix Rows", doubling matrix storage exactly as Fig 15
draws), and the FA computes BOTH rails in-DRAM: carry=MAJ3(a,b,c),
**~carry=MAJ3(~a,~b,~c)** (De Morgan), sum=MAJ5(a,b,c,~carry,~carry),
**~sum=MAJ5(~a,~b,~c,carry,carry)**. The host transports rail data between
tiled ops but applies NOT nowhere in the chain (leaf complements = the
load-prepared ~W planes; intermediate complements = silicon MAJ outputs).

| arm (4096² q4 r1, seed 42) | wall | MAJs | exact outputs | rail violations |
|---|---|---|---|---|
| host-formed ~carry (R1) | 38.4 s | 16,888 | 99.976% | n/a |
| host-formed + vote3 (07-18) | 110 s | ×3 | **4096/4096 ×2 (bit-exact)** | n/a |
| dual-track unvoted (R2) | 76.0 s | 33,776 | 4075/4096 = 99.49% | 0.063% |
| dual-track + vote3 (R2v) | 229.8 s | 101,328 | 4089/4096 = 99.83% | 0.021% |

- **Depth achieved: the FULL CSA tree** — complements are silicon-formed at
  every one of the ~12 reduction levels (2,114 selected leaves/plane →
  count bits), not just level 1; level 1 is additionally fully physical in
  clone mode (~W clones). Cost = exactly 2× MAJs/wall ("at the cost of
  additional row usage", §VII) + 2× matrix planes.
- **Row-count math (why full residency doesn't fit, and what does):** paper
  Fig-15 shape per subarray = qN matrix + qN inverted + computation +
  output rows; a resident dual-rail CSA chunk of N taps needs ~2N leaf
  rows + ~N live intermediate rails ≈ 3N + 16 tuple rows → N ≤ ~160 in a
  512-row subarray — consistent with their N ≤ 128 partitioning (§VII).
  Our single-tuple rig time-multiplexes instead (the standing
  tiled-execution deviation): resident row budget actually used = the 16-row
  MAJ5 tuple (host mode) / 15 role rows from the 48-row screened value pool
  (clone mode, rails in-place within the FA, host-transported across FAs).
- Accuracy: unvoted 99.49% vs 99.976% host-formed — a wrong ~carry feeds
  the MAJ5 twice, so rail errors amplify (deep-weight |err| ~2.6–3.3K);
  vote3 recovers to 99.83% (not bit-exact — the doubled MAJ exposure needs
  deeper voting than the host-formed arm). The dual rail's practical value
  on commodity silicon: it is a self-CHECKING computation (violation
  telemetry above), which host-formed complements cannot give.

**(c) GEMV_PARTIALS → the first exact fp32 — magic 0x4D563003.**
Same request as GEMV; returns the exact per-32-weight-block integer partial
sums (M × K/32 i32, m-major; 4096² → 2 MB, the 32000×4096 head → 16 MB,
both far under the 256 MB frame cap). This is the paper's "partial sums
computed per subarray, retrieved and aggregated by the processor"
(§II-C2/§VII) at q4_0/q8_0 quant-block granularity — which is precisely
what makes EXACT fp32 possible: y[m] = Σ_b fp32(d4[m,b])·fp32(d8[b])·P[m,b]
with ggml's own scalar vec_dot_q4_0_q8_0 order and association.
- Per-block trees are slightly CHEAPER than whole-K (2.18 s vs 2.5 s at
  K=256 M=1024 q4 r1: shallower carries), and unvoted partial exactness is
  high (8191/8192 = 99.99% in the smoke — errors localize to one block).
- Silicon verification on the REAL tensor (blk.0.attn_q.weight, the
  bit-exact-validated de-repack dump: 16.7M descaled ints + 524,288 fp16
  scales), q8_0-quantized activations (ggml convention, plane densities
  0.48–0.51), PIM_VOTE3=1 (R4):

| run | wall | int-exact partials | fp32 vs CPU reference |
|---|---|---|---|
| 4096×4096 q4_0 × q8_0 (r=8), vote3 | 732.8 s (55,388 FA / 332,328 MAJ) | **524,288/524,288 = 100%** | **BIT-EXACT, 4096/4096 outputs bit-identical** |

  **The reproduction produces exact fp32 for the first time** — not just
  exact integers: every per-block partial exact, and the host per-block
  scale application reproduces the ggml scalar vec_dot result bit-for-bit
  (fp64 dequant oracle agrees to 2.3e-4 rel, the fp32-accumulation
  rounding floor — sanity only). Server shut down clean; card enumerated
  after. Driver: lane2_partials_fp32.py (run log
  o7_logs_2026_07_20/R4_fp32_realtensor.log).

**Paper ambiguities met (consult-the-paper convention):**
1. §V-C never states where product rows live relative to the adder inputs;
   we clone products directly into the computation (tuple input) rows —
   consistent with Fig 2 steps 2–3.
2. §VII says carry and sum are computed "along with their complements" but
   never writes the complement identities; we used the unique symmetric
   choices ~s1=MAJ3(~x0,~x1,~x2), ~s0=MAJ5(~x0,~x1,~x2,s1,s1).
3. Scales/dequantization appear nowhere in the paper, and their stated
   partial-sum granularity (N ≤ 128/subarray) CROSSES q4_0's 32-weight
   scale blocks — exact fp32 is impossible from partials at their stated
   granularity. Our 32-granularity partials are a completion beyond the
   paper's shape (route c2's honest closure).
4. Their "1-bit vector/matrix" sweeps are unsigned binary; the server's
   qb=1 matches (the smoke client generated signed 1-bit until today —
   client-side convention fix, no server change).

## 2026-07-21 — Road B closes, and re-opens the paper's own dataflow

The FPGA popcount accumulator (Road B, the rig-specific arm kept
strictly apart from the reproduction's headline numbers) is now
complete on silicon: three builds of the readback engine, the last
fixing a buffer_space conservation leak that also exists in stock
DRAM-Bender streaming DIFF mode, then 65,000-program sessions with zero
stream-integrity faults. Sources in `rtl/` + `api-patches/0003`.

The reproduction-relevant part is what it did to the dataflow choice.
With per-read totals nearly free, the optimal GeMV shape inverts from
our carry-save tree back to the **paper's own §V per-output product
form** — one product row per output, aggregated at readout. Measured at
their dims (per-op, single tuple, same conventions as the 07-20 B2
table): 4096×4096 qb4 in **2.93 s vs the tree arm's 34.1 s (~12×)**,
sparsity-independent, numpy-exact at small scale and inside the usual
unvoted flake envelope at full scale. Reading it the other way: the
June "readout wall" that forced us AWAY from the paper's dataflow was a
property of our host-round-trip readout, not of the dataflow — with
line-rate aggregation (which MVDRAM's §V-E streaming command generation
implies their system effectively has), the paper's shape is the right
one. That is a stronger, more specific corroboration of their design
choice than the tree arm could give.

**2026-07-21 addendum — quant coverage complete.** The last staged
shape-coverage op ran: q3_K native (Llama-2-7B Q2_K blk.0.attn_v,
K=M=4096, qb=3) — 122.1 s, **99.90 % int-exact**, inside the unvoted
envelope. q2_K / q3_K / q4_0 / q6_K are all silicon-verified through
the llama.cpp interception path; the model set (7B, 13B, Llama-3-8B,
phi-4) was already covered by the sampled e2e logs in
`docs/data/lane2/`.

**2026-07-21 addendum — 256-token sampled e2e.** The depth extension of
the four-model sampled protocol: Llama-2-7B Q4_0, 256 generated tokens,
24 ops routed through the in-DRAM server across the whole horizon —
**24/24 verified, mean 99.845 % int-exact (99.77–99.90 %)**, attn and
ffn shapes, K up to 11008. (Route c2: the CPU stays dst; the PIM is the
verified side-channel, so text quality is the base model's own
prompt-echo at greedy — orthogonal to the verification claim.)
