# MVDRAM Reproduction Explainer — Claim Ledger

> ⚠ **RETIRED 2026-08-04.** The interactive MVDRAM deck (`mvdram.html`) has been
> folded into one comparison home and now redirects. The MVDRAM comparison lives
> in **[RELATED_SYSTEMS.md](../RELATED_SYSTEMS.md)** (§2), with citations inline;
> the deep hardware-reproduction study is
> **[MVDRAM_REPRODUCTION.md](../MVDRAM_REPRODUCTION.md)**. This file is kept only
> as prior provenance — do not treat it as a live publish gate.

Every factual claim in `mvdram.html` must trace to one of:
- **[paper §SEC / Fig N / Table N / fn. N]** — MVDRAM, Kubo et al., arXiv:2503.23817 (v2, 23 Sep 2025)
- **[study §N]** — [`docs/MVDRAM_REPRODUCTION.md`](../MVDRAM_REPRODUCTION.md) (updated 2026-07-17)
- **[code:PATH]** — a reproducer in this repository
- **[data:PATH]** — a measurement file in `docs/data/`
- **[issue:URL]** — a linked GitHub issue
- **[example]** — chip-specific value shown only as a labelled example
- **[editorial]** — framing/interpretation, flagged as such in the deck and grounded in the cited rows around it

Claims that fit none of those tiers are **REMOVED** before publication.

Ledger created 2026-07-17, same date as the deck. Companion ledgers:
[`pim_explainer_ledger.md`](pim_explainer_ledger.md) (main deck),
[`xor_spread_ledger.md`](xor_spread_ledger.md) (row-spread deck, July-update scenes).

---

## Scene 1 — What MVDRAM claims

| Claim | Source |
|---|---|
| MVDRAM executes GeMV inside unmodified DDR4 via two PUD primitives (RowCopy, MAJX) chained so operands move in-DRAM; matrix preloaded once; small result readout | [study §1] (derived from the paper; the deck presents it as the paper's claim) |
| The performance claim rests on that chain being executable in-DRAM (its latency profile) | [paper Fig 3] (the profile assumes fast in-DRAM operand movement); [study §1] |
| 32000×4096 GeMV (llama2-7B dims), 1-bit vector × 2-bit matrix: CPU 1.44 ms, GPU 1.70 ms, MVDRAM 0.14 ms in-DRAM + 0.05 ms aggregation = 0.19 ms → 7.29× / 8.55× | [paper §VIII-B] |
| Scaling to 32768×32768 (2-bit): 3.38× / 3.74× | [paper §VIII-B] |
| GeMV benchmark convention: 1,000-iteration averages, inputs at 50% bit sparsity ("typical LLM distribution") | [paper §VIII-A] |
| End-to-end: Llama2-7B, Llama2-13B, Llama3-8B, Phi-4 on llama.cpp with mulmat_op replaced; 256 generated tokens, averaged over 10 runs | [paper §VIII-A] |
| End-to-end result: 2.18× vs CPU (2-bit Llama2-13B; 3.33× vs GPU), 1.31× vs CPU (4-bit) | [paper Fig 16] |
| This is a real measured run through DRAM, not a projection | [paper §VIII-A] (method as described); [study §1] |
| CPU baseline: i7-9700K with the same DDR4-2400 modules (77 GB/s) | [paper Table II] |
| GPU baseline: NVIDIA Jetson Orin Nano (LPDDR5, 68 GB/s) — an edge part | [paper Table II]; "edge part, not a datacenter GPU" framing is [editorial] grounded in the part's identity |
| Energy: 30.5× GeMV / 3.04× per-token vs CPU; MVDRAM power CACTI-estimated [their ref 56], baselines measured (RAPL/tegrastats) | [paper Fig 14, Fig 17] |
| Temperature/voltage robustness (§IX) cited from prior work [their ref 36], not measured by them | [paper §IX] |

## Scene 2 — Their two techniques

| Claim | Source |
|---|---|
| On-the-fly encoding: activation bit selects the RowCopy source — a=1 → copy W row, a=0 → copy zero row into the computation region | [paper §V, Fig 6] |
| Bit-sparsity skip: when a=0 the operation can be skipped | [paper §V-D] |
| Their inputs follow a 50% bit-sparsity convention | [paper §VIII-A] |
| Horizontal layout: bitline position = output element; partial-product rows reduced by MAJ-based adders down each bitline; row-major readout | [paper §VI, Fig 7–10] |
| Partitioning: N ≤ 128 rows per subarray, spread across subarrays and 4 modules | [paper §VII] |
| Dual-track: both polarities stored; matrix rows + inverted matrix rows ≈ 2× matrix storage | [paper §VII, Fig 15] |
| Full adder: carry s₁ = MAJ3(x₀,x₁,x₂); sum s₀ = MAJ5(x₀,x₁,x₂,¬s₁,¬s₁) — sum bit needs reliable MAJ5 | [paper §II-C1] |
| "Error-free" MAJX rests on reliable-column screening + Frac [their ref 34] + calibration [their ref 48] | [paper §VII] |

## Scene 3 — What "reproduce" means here

| Claim | Source |
|---|---|
| The paper released no source code, no artifact, no module date codes | [study §6.1–6.2] ("the paper released no code"; date codes requested) |
| Our reproducers are public and run on any DRAM-Bender setup | [study §6.2]; [code:app/] |
| Their rig: host PC + DRAM Bender on Xilinx Alveo U200, 4× SK Hynix DDR4-2400 HMA851U6CJR6N-UHN0 | [paper §VII, Fig 11] |
| Our rig: DRAM-Bender on Xilinx BCU1525, quad-controller bitstream, same doubleACT timing-envelope primitives; 6 modules = 2× new units of the named part + 4 commodity DDR4 UDIMMs; ambient temperature | [study §2] |
| Heated retest planned | [study §6.4] |
| Footnote 3: they characterized 16 SK Hynix modules; the named part selected as "the most reliable one that supports both strict RowCopy and MAJX operations" (up to MAJ15) | [paper fn. 3] (quotation < 15 words, attributed) |
| Table I: 54,365–61,727 of 65,536 reliable columns per module = 82.9–95.1% | [paper Table I] |
| Compute restricted to consecutive runs of q reliable columns | [paper §VII] |
| "15 of 16 modules did not make the cut, and nothing tells a reader which purchasable modules would" | [editorial] grounded in [paper fn. 3] (1 selected of 16 characterized) |
| Scope statement (what the study answers vs. defers) | [editorial]; deferred items enumerated with sources in Scene 7 |

## Scene 4 — Result A: the named part performs no PUD

| Claim | Source |
|---|---|
| Two brand-new units of SK Hynix HMA851U6CJR6N-UHN0 purchased June 2026 | [study §2, §3] |
| Both work perfectly as memory; 25/25 constellation rows verify in the spread-test baseline; full row write/readback used throughout | [study §3] |
| RowClone fixed pairs, t₁₂ swept 5→150 (14 values), t₂₃ 1→4: best 41/8192 bytes (noise floor) at every timing, both units | [study §3, table]; [code:app/test_rowclone_smoke.cpp] (PIM_T12 override) |
| 30,000 random pairs per module (rows uniform in [0, 65536)): 0 clones, 0 partials (>1000 B), best 45/8192 and 47/8192 | [study §3, table]; [code:app/test_rowclone_random.cpp]; [data:docs/data/mvdram-repro/rcrand_b0.log], [data:docs/data/mvdram-repro/rcrand_b3.log] |
| SiMRA characterization sweep (RowClone stage): 9.3M attempts, max match 0, zero co-activatable row groups (all "subarrays" singletons) | [study §3, table] |
| Control (commodity PUD-capable module, same tool): 6 full clones in 500 pairs (1.2%); 640-row subarray structure found within minutes | [study §3]; [data:docs/data/mvdram-repro/README.md] |
| Random pairs land same-physical-subarray at ~1% (per the control), so 0/60,000 rules out address scrambling, timing choice, row choice, dead modules | [study §3] (the argument as published) |
| A part that performs no PUD cannot execute any part of MVDRAM's method, as purchased today | [study §3, §5] |
| Not a claim that their measurements are wrong; consistent with severe inter-module variance within one part number given their 16-module screening | [study §5]; [paper fn. 3]; variance interpretation [editorial] |
| Date codes / SPD dumps requested from the authors | [study §6.1] |
| Result A (no PUD on the named part) is independent of the chained-dataflow result (Result B) | [study §8] |

## Scene 5 — Result B: the chained dataflow, and the placement it requires

| Claim | Source |
|---|---|
| Naive chain: faithful computation-rows dataflow (their Fig 2) 6.1% end-to-end; per-op MAJ 50.3% (coin flip) | [study §4, table]; [code:app/test_mvdram_compute_rows.cpp] (the un-placed reproducer) |
| Naive chain: on-the-fly encoding 11.3% vs 99.9% for the identical GeMV with host-written products | [study §4, table] |
| Naive chain: RowClone-loading a calibrated 16-row tuple 50.1%; minimal 4-row tuple 75%; mitigations (full-restore, non-shadow source) no improvement | [study §4, table] |
| Every MVDRAM mechanism works in isolation; a naive chain corrupts | [study §4] |
| Cause: the XOR-spread (deposit into lattice siblings of the source); real, deterministic, unmentioned by the paper | [study §4]; [code:app/test_spread.cpp], [code:app/test_fault_sweep.cpp] |
| Corruption is a function of the pair offset, not the source row, so it is fixed by placement rather than by mitigation; an earlier public verdict ("general copy-based dataflow not achievable") is superseded (study §8) | [study §8]; [code:app/test_safe_load.cpp] |
| Mechanism settled as Multi-RowCopy's co-activation lattice; selection law | [issue:https://github.com/CMU-SAFARI/DRAM-Bender/issues/12]; [issue:https://github.com/CMU-SAFARI/SiMRA-DRAM/issues/1]; [code:app/test_selection_timing.cpp] + [data:docs/data/selection-law/] |
| Spread-aware: corruption during a RowClone into a co-activated group is a function of the pair offset src⊕dst; generator-subset-free offsets are clean by construction | [study §8]; [code:app/test_safe_load.cpp] |
| Measured: 20/20 safe external loads clean; unsafe-offset controls corrupt exactly the predicted rows; full 16-row tuple loaded with 8 doubleACTs, 16/16 | [code:app/test_safe_load.cpp] (phases 1, 2, 4) |
| Same DAG (popcount-4 carry-save, 18 MAJ3 gates), placement-only change: write-load 1.59 ms/gate @ 99.99% → safe unfused 0.89 @ 99.87% → fused 0.68 @ 99.81%; faithful dataflow 99.98% e2e on the safe-placement reproducer | [study §8, table]; [code:app/test_mvdram_compute_rows_safe.cpp] |
| The 6.1% naive-chain figure is an addressing artifact of un-placed loading; the corrected result is in the study with dated supersession (§8) | [study §8] |
| Two-die fastpath A/B: write-load 1.585/1.574 ms/gate @ 99.990%/99.974% e2e; fused clone-load 0.683/0.705 @ 99.808%/99.814% = 2.32×/2.23× (die A = bender 2, die B = bender 0) | [code:app/test_mvdram_fastpath_ab.cpp]; [data:docs/data/mvdram-repro/fastpath_ab_b2.log] (die A), [data:docs/data/mvdram-repro/fastpath_ab_b0.log] (die B) |
| Fast in-DRAM loading costs ~0.17% end-to-end; fusing costs nothing | [study §8] (accuracy-cost statement); arithmetic from the fastpath logs (99.990−99.808 / 99.974−99.814 ≈ 0.17 pp) |

## Scene 6 — The mechanism scoreboard

One row per scoreboard entry; "paper" column then "ours" column.

| Scoreboard row | Paper source | Our source |
|---|---|---|
| Reliable-column fraction: 83–95% vs 94–95% strict all 8 banks (ZERO reference policy) | [paper Table I] | RESULT.md addendum 10 §C + colmasks_maj5_zero_2026_07_17/ (per-bank winners 1919–1954/2048 — verified against majx select-mode logs) |
| RowCopy: strict on screened part vs 8192/8192 deterministic (commodity) | [paper fn. 3] | [study §4, table]; [code:app/test_rowclone_smoke.cpp] |
| Dual-track adder: error-free after Frac[34]+calib[48] vs mechanism-exact — strictly error-free on 94.8% of cols (ZERO+2), 16-bit chain carry 0.0000% | [paper §VII] | RESULT.md addendum 8b (1942/2048 = 94.8% adder-strict on screen) + addendum 13 ([data:sublattice_broadcast_2026_07_17/chain_b2_s78_16bit.csv]: carry 0.0000% all 16 positions, sum 0.009–0.068%) ; the 99.94% single-op row retained in [study §4] |
| — all-MAJ3 variant 99.98% | n/a (their adder uses MAJ5) [paper §II-C1] | [code:app/test_mvdram_compute_rows_safe.cpp]; [study §4 table ("in-DRAM carry-save popcount tree 99.97–99.98%") + §8] |
| On-the-fly encoding: PHYSICAL — clone-created products, 3.0× faster, 94.7% unvoted at 4096² | [paper §V, Fig 6] | [code:mvdram-repro/lane2_gemv_server.cpp] (LANE2_ENCODE=clone); [data:mvdram-repro/o7_logs_2026_07_20/R3b_clone_4096_depthscreen.log] (11.9 s, 3877/4096 = 94.65% — verified 2026-07-20) vs R1 (38.4 s) |
| Bit-sparsity skip: measured — commands ∝ density (9.0→5.1/pass at 100→50%); zero-skip in all server modes; B2 table's measured-density arm quantifies it at model dims | [paper §V-D] | RESULT.md addendum 17 §4 (gemv-encoded); [data:mvdram-repro/b2_results/20260719_233500/] (paper50 vs measured arms) |
| Horizontal layout + row-wise aggregation: reproduced | [paper §VI, Fig 7–10] | [study §4, table]; [code:app/test_mvdram_gemv_inplace.cpp] |
| Computation-rows dataflow: 99.98% with safe placement (vs 6.1% naive chain) | [paper Fig 2] (as characterized in [study §4]) | [code:app/test_mvdram_compute_rows_safe.cpp]; [study §8] |
| Fast in-DRAM operand movement: 2.2–2.3×/gate vs host-write shape; 3.0×/GeMV in the fused clone server mode | [paper Fig 3] (profile assumption) | [code:app/test_mvdram_fastpath_ab.cpp]; [data:docs/data/mvdram-repro/fastpath_ab_b*.log] (1.585/0.885/0.683 b2; 1.574/0.898/0.705 b0 — re-verified 2026-07-20); lane2 clone A/B (o7 logs) |
| Partial sums to processor: per-32-block integer partials → FIRST EXACT FP32, bit-exact vs CPU on the real Llama-2-7B blk.0.attn_q tensor | [paper §II-C2, §VII] | [code:mvdram-repro/lane2_gemv_server.cpp] (GEMV_PARTIALS 0x4D563003) + [code:mvdram-repro/lane2_partials_fp32.py]; [data:mvdram-repro/o7_logs_2026_07_20/R4_fp32_realtensor.log] (524,288/524,288 int-exact; 4096/4096 fp32 bit-identical — verified 2026-07-20) |
| Their e2e protocol: sampled interception, 36 verified ops across 4 models × both precisions, 99.58–99.99% int-exact unvoted, outputs CPU-exact by construction | [paper §VIII-A] | [code:mvdram_bench/llama.cpp shim mvdram-pim.{h,c}]; [data:mvdram_bench/smoke_2026_07_19/silicon/e2e_*_ops.log] (36 ops counted, range re-derived 2026-07-20: min 99.5759%, max 99.9928%) |

## Scene 7 — The three gaps: two closed, one remains

Gap definitions retained below; the closure rows follow.

| Claim (07-20) | Source |
|---|---|
| Gap 1 CLOSED: MAJ5 is conditioning-sensitive; the dominant knob is the reference row's INIT, not pulse count | RESULT.md addendum 8 (frac-maj5-exe sweeps) |
| ZERO-init 963 strict cols vs ~200 frac'd-ONE on the marginal subarray (≈5×); ZERO+2 on the good subarray: 1915 strict / 2010 soft95 of 2048 (600-run confirm) | [data:sublattice_broadcast_2026_07_17/frac_maj5_b0_s72.csv, frac_maj5_b2_s86_confirm.log] (confirm rows re-read 2026-07-20: n_frac=2 t_frac=0 ZERO → strict=1915 soft95=2010) |
| Per-bank MAJ5 colmasks 94–95% strict on all 8 banks of both compute dies | RESULT.md addendum 10 §C; colmasks_maj5_zero_2026_07_17/maj5_best_b{0,2}_bank{0-3}.txt |
| Exact dual-track adder strictly error-free on 1942/2048 = 94.8% of s86 under ZERO+2; the earlier chain shortfall is attributed to frac'd-ONE on unscreened cols | RESULT.md addenda 8b + 13 |
| MAJ3 is conditioning-INSENSITIVE (flat 78/78 configs); frac does NOT rescue the geometry-limited die (corr −0.44) — two dead ends closed | RESULT.md addendum 7 ([data:frac_sweep_b2_s72.csv, frac_sweep_b1_s24.csv]) |
| Gap 2 CLOSED at declared sampled scope: llama.cpp hook + lane2 server + 36 verified ops + B2 table | Scene 8 rows (primary) |
| Gap 3 REMAINS with a measured size: 32000×4096 qb2 paper50 cell 17.04 s vs their 0.19 ms ≈ 9×10⁴ | [data:mvdram-repro/b2_results/20260719_233500/b2_gemv_table.md] (17.04±0.31 — verified 2026-07-20) ÷ [paper §VIII-B] |
| Fetch-side direction: seq_engine 100% command-bus utilization in Verilator; 8K-IMEM multi-body packing measured ~2.3× | memory record dram_bender_seq_engine; RESULT.md addendum 15 |

### Original 07-17 gap definitions (retained)

| Claim | Source |
|---|---|
| Their error-free MAJX rests on Frac [34] + calibration [48] on top of screening | [paper §VII] |
| We never implemented Frac[34]/calibration[48] on our rig (negative scope statement) | [editorial: scope]; supporting detail — our `frac_template` is a single ACT–PRE discharge primitive, not FracDRAM fractional-charge conditioning: [code:app/test_bitnet_server.cpp:97-113] |
| It might rescue MAJ5-chain columns on our silicon → concrete next experiment | [editorial: open work] (until run, our MAJ5 results and theirs are not directly comparable) |
| Their benchmark protocol: 4 models on llama.cpp, mulmat_op replaced, Q2/Q4, 256 tokens × 10 runs | [paper §VIII-A] |
| We ran our own model instead: BitNet b1.58-2B via transformers (all 30 layers' projections in DRAM) | [study §7]; [code:python/run_bitnet_pim.py] |
| We never executed their exact benchmark; a kernel-level equivalence argument covers the compute but it is a different end-to-end | [editorial: scope], grounded in the two rows above |
| Unmatched conventions: 50%-sparsity inputs, 1,000-iteration averages, N≤128/subarray, per-module Table-I-style reliable-column tables | conventions from [paper §VIII-A, §VII, Table I]; "unmatched" is a scope statement |
| §V-E: a single-threaded CPU generates commands faster than DDR4-2400's ~1.5 ns/command processing — command generation overlaps execution | [paper §V-E] |
| Their 0.14 ms full GeMV implies on the order of 10⁵ commands as one continuous stream | arithmetic from [paper §VIII-B] (0.14 ms) ÷ [paper §V-E] (~1.5 ns/command) ≈ 9×10⁴ |
| Ours: per-program host round-trips, ≤2048-instruction IMEM; a fused gate is 154 instructions | [data:docs/data/mvdram-repro/fastpath_ab_b0.log, fastpath_ab_b2.log] ("fused gate program: max 154 insts (IMEM 2048)") |
| This is the structural gap between our per-gate numbers and their kernel numbers — engineering, not physics | [editorial], grounded in the two rows above |

## Scene 8 — Their protocol, run (NEW 2026-07-20)

| Claim | Source |
|---|---|
| Shim at BOTH mulmat attachment points (generic + repack); the repack path carries all quantized projection mulmats on x86 | REPRODUCTION.md 2026-07-18 (census section); [code:mvdram_bench/llama.cpp ggml/src/ggml-cpu/mvdram-pim.{h,c}] |
| Sampled interception: every Nth eligible GeMV runs on silicon AND CPU with a per-op verdict; generation text CPU-exact by construction (route c2, declared in advance) | LANE2_GEMV_SERVER.md (scope decision); REPRODUCTION.md 2026-07-18 (threading contract + route c2) |
| Full-fidelity 256-tok e2e is multi-day per model at our per-op walls — the reason for sampled scope, stated | LANE2_GEMV_SERVER.md (scope decision) |
| 36 verified ops: 7B 12 (Q4_0 99.81–99.90%, Q2_K 99.83–99.90%), 13B 8, 8B 8, Phi-4 8; overall range 99.58–99.99% | [data:mvdram_bench/smoke_2026_07_19/silicon/e2e_*_ops.log] (counted 36; per-file min/max re-derived 2026-07-20) |
| Dims covered: 4096/5120/7680/11008/13824/14336/32000/35840 incl. Llama-3 GQA kv (M=1024) and a 35,840-output head | REPRODUCTION.md 2026-07-20 (§VIII-A protocol COMPLETE) + ops logs |
| Quant mappings validated bit-exact vs gguf ground truth before silicon: q4_0 (q−8), q2_K (q−2), q3_K (split-storage q−4), q6_K (q−32, 3-handle split) | REPRODUCTION.md 2026-07-18/19 (check_derepack/check_q2k/check_q3k/check_q6k, zero mismatches) |
| B2 table: 32 cells × 5 iters, both input arms (paper50 + measured density), CPU column, on silicon | [data:mvdram-repro/b2_results/20260719_233500/b2_gemv_table.{md,csv}] |
| Cells quoted in the deck: 4096² qb4 paper50 34.09±0.64 s / measured 18.88±0.39; 32000×4096 qb2 17.04±0.31 / 14.83±0.33; qb1 8.56±0.20 / 7.51±0.12 | same table (rows re-read 2026-07-20) |
| Deviations stated: 5 iters vs 1,000; single module/tuple vs 4-module N≤128; host round-trips vs §V-E | table header (the deviations are printed in the artifact itself) |
| Exact fp32: 524,288/524,288 partials int-exact; 4096/4096 outputs bit-identical to CPU fp32 reference; 732.8 s, 55,388 FA, 332,328 MAJ, vote3 | [data:mvdram-repro/o7_logs_2026_07_20/R4_fp32_realtensor.log] (verified 2026-07-20) |
| Paper-side finding: N≤128 partial granularity crosses q4_0's 32-weight scale blocks → exact fp32 impossible from partials at the paper's stated shape; scales/dequantization absent from the paper | REPRODUCTION.md 2026-07-18 (route c2 paper re-read: "the words scale/zero-point/dequantize never appear") + 2026-07-20 O7 ambiguity 3 [editorial: arithmetic consequence] |

## Scene 9 — The last fidelity deviations, closed (NEW 2026-07-20)

| Claim | Source |
|---|---|
| Phase-1 deviations declared 07-18: host-resolved encoding; host-formed ¬carry; no exact-fp32 path; host-streamed matrix | REPRODUCTION.md 2026-07-18 phase-1 section (the deviations list, written before the closures) |
| All three closed as env-gated modes; defaults byte-preserve phase-1 (R1 revalidates) | LANE2_GEMV_SERVER.md 2026-07-20 header; REPRODUCTION.md O7 section |
| Clone-encode: 11.9 s vs 38.4 s = 3.2×; pcwrites 253,032 → 50,592 (5.0×); 94.65% unvoted | [data:o7_logs R3b vs R1] (verified 2026-07-20) |
| Accuracy price mechanism: ~0.03%/gate MAJ3 flake × 9× more MAJs × ~12 levels; transient-dominated; depth-2 screen bought +0.4% only | REPRODUCTION.md O7 (a) |
| Rail telemetry: v⊕¬v violations 0.30% of lane-checks (clone mode); a built-in error detector | REPRODUCTION.md O7 (a); R2/R2v logs show 0.063%/0.021% in dual-track mode (verified) |
| Dual-track: 76.0 s / 99.49% unvoted; vote3 229.8 s / 99.83%; complements silicon-formed at every ~12 levels; exactly 2× MAJs/wall + 2× matrix planes | [data:o7_logs R2, R2v] (verified 2026-07-20); [paper §VII, Fig 15] for the stated trade |
| Row-budget math: resident dual-rail chunk ≈ 3N+16 rows → N ≤ ~160 per 512-row subarray, consistent with their N≤128 | REPRODUCTION.md O7 (b) [editorial: arithmetic]; [paper §VII] |
| Standing deviation: matrix residency (single-tuple time-multiplexing), stated wherever numbers appear | REPRODUCTION.md 2026-07-20 supersession note |
| Four paper ambiguities + conventions chosen (product-row placement; complement identities; partial granularity; unsigned 1-bit) | REPRODUCTION.md O7 "Paper ambiguities met" (items 1–4) |

## Scene 10 (was Scene 8) — What our silicon adds beyond the paper

| Claim | Source |
|---|---|
| Selection law: member A⊕S fires iff every predecoder unit contributes ∅ or its whole share of d; units {1,2},{3,4},{5,6},{7,8}, bits 0/9 singleton | [code:app/test_selection_timing.cpp]; [data:docs/data/selection-law/] |
| 1691/1691 member-observations explained on each of two dies; zero exceptions; zero mixed rows | [data:docs/data/selection-law/selection_timing_b0.csv], [data:...selection_timing_b2.csv] (1691 complete member rows each; `fired` column never 2) |
| Firing set identical at all 8 timing combos (t₁₂∈{5,10,20,30} × t₂₃∈{1,2}) | [data:docs/data/selection-law/selection_timing_b2.log] (per-(k,t₁₂,t₂₃) summary constant in t) |
| Screening becomes a computed predicate | [editorial: consequence]; operationalized in [code:app/test_safe_load.cpp] |
| Cross-die determinism: the per-member CSVs are byte-identical across the two dies (only the device-path preamble differs) | [data:docs/data/selection-law/selection_timing_b0.csv vs selection_timing_b2.csv] (diff = 2 preamble lines) |
| Hence a placement discipline computed on one die transfers to another of the same design | [editorial: direct consequence of the byte-identical law data] |
| Copy-timing operand deposit: at copy timing (or drifted timing) the co-activation deposits the first-activated row into other tuple members, so a subsequent vote runs over a substituted operand set; on all three of our dies the clean vote at the operating point does not deposit — the deposit is placed by a copy-timing preparation/load or drift, never by the vote itself | [study §4, provenance note] |
| Column-static error models (incl. reliable-column screening) cannot represent the effect — it depends on which rows are co-activated, not the column | [study §4, closing paragraph] |
| All-MAJ3 accumulation: carry-save popcount from MAJ3 only, 99.98% e2e with safe placement; no MAJ5 → no Frac/calibration dependency for the adder | [code:app/test_mvdram_compute_rows_safe.cpp] (18-MAJ3-gate DAG); [study §4 table, §8]; MAJ5 dependency of their adder: [paper §II-C1] |

### Scene 10 additions (cards 5–6, NEW 2026-07-20)

| Claim | Source |
|---|---|
| The reference-policy law (card 5): rows as in Scene 7 gap-1 closure | Scene 7 closure rows (primary) |
| The clone-dead law (card 6): closed form; held-out 2,496/2,496 pre-committed; die 2: 2,494/2,496 zero false-dead; converts screening into computation | xor_spread_explainer_ledger.md Scene 6 rows (primary); [code:app/clone_law.py] |

## Scene 11 (was Scene 9) — Verdict

| Claim | Source |
|---|---|
| Completions line: mechanism-exact adder (94.8%), four-model protocol at sampled scope (36 ops), first exact fp32 | Scenes 7–9 rows above (primary) |
| Energy caveat retained verbatim: 30.5×/3.04× CACTI-estimated MVDRAM side vs measured baselines (RAPL/tegrastats) — asymmetric method the paper itself states | [paper Fig 14, Fig 17] |
| The measured streaming gap ≈9×10⁴ quoted in the verdict | Scene 7 gap-3 row (primary) |

### Original Scene 9 (Verdict) rows, retained

| Claim | Source |
|---|---|
| Verdict: MVDRAM's method is achievable on commodity DDR4 — but only with spread-aware addressing the paper does not discuss, on silicon its own screening would have rejected; an earlier "not achievable" verdict is superseded (study §8) | [study §8] |
| Our spread characterization is the requirement specification, not a refutation | [study §8] |
| Part-number lottery: named part 0/60,000 ×2 new units; their own screening kept 1 of 16 | [study §3]; [paper fn. 3] |
| "Unmodified DRAM in practice means unmodified, hand-screened DRAM" | [editorial], grounded in the row above |
| Recommendation: budget for screening; run the 1-minute spread test on survivors | [study §6.5] |
| Their end-to-end 2.18×/1.31× is a real measured llama.cpp run — the paper's strongest evidence | [paper §VIII-A, Fig 16]; "strongest evidence" is [editorial] |
| Their energy figures are CACTI-estimated on the MVDRAM side vs measured baselines | [paper Fig 14, Fig 17] |
| Their GPU baseline is a Jetson Orin Nano (edge part) | [paper Table II] |
| Our performance numbers are kernel-level (per-gate/per-matmul) on a round-trip-bound rig, not their benchmark | [editorial: scope]; the gap quantified in Scene 7 rows |

---

## Claims deliberately left out (no in-repo source)

Recorded so they are not silently re-added:

- A measured **4.37× command-stream reduction** for the §V-D bit-sparsity
  skip (host-side analog) exists in internal notes but has no committed
  reproducer log in this repository — the deck says "implemented
  (PIM_SKIP); not benchmarked at scale" instead.
- The **1.37% survival of columns through the chained MAJ5 adder** (vs
  87–88% single-op MAJ5) exists in internal notes only; the deck's
  Scene 7 states the Frac/calibration gap without that number.
