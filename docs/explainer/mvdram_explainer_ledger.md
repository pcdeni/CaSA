# MVDRAM Reproduction Explainer — Claim Ledger

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
| Result A unaffected by the Result B reversal | [study §8] |

## Scene 5 — Result B: June's collapse, July's reversal

| Claim | Source |
|---|---|
| June: faithful computation-rows dataflow (their Fig 2) 6.1% end-to-end; per-op MAJ 50.3% (coin flip) | [study §4, table]; [code:app/test_mvdram_compute_rows.cpp] (the June-shape reproducer) |
| June: on-the-fly encoding 11.3% vs 99.9% for the identical GeMV with host-written products | [study §4, table] |
| June: RowClone-loading a calibrated 16-row tuple 50.1%; minimal 4-row tuple 75%; mitigations (full-restore, non-shadow source) no improvement | [study §4, table] |
| Every MVDRAM mechanism worked in isolation; chaining did not (June state) | [study §4] |
| Cause: the XOR-spread (deposit into lattice siblings of the source); real, deterministic, unmentioned by the paper | [study §4]; [code:app/test_spread.cpp], [code:app/test_fault_sweep.cpp] |
| June's model ("source-specific vulnerability") was wrong; the published June verdict ("general copy-based dataflow not achievable") is withdrawn | [study §8]; June model description [code:app/test_safe_load.cpp] (header recaps the June post-mortem) |
| Mechanism settled as Multi-RowCopy's co-activation lattice; selection law | [issue:https://github.com/CMU-SAFARI/DRAM-Bender/issues/12]; [issue:https://github.com/CMU-SAFARI/SiMRA-DRAM/issues/1]; [code:app/test_selection_timing.cpp] + [data:docs/data/selection-law/] |
| July: corruption during a RowClone into a co-activated group is a function of the pair offset src⊕dst; generator-subset-free offsets are clean by construction | [study §8]; [code:app/test_safe_load.cpp] |
| Measured: 20/20 safe external loads clean; unsafe-offset controls corrupt exactly the predicted rows; full 16-row tuple loaded with 8 doubleACTs, 16/16 (June: 5/16) | [code:app/test_safe_load.cpp] (phases 1, 2, 4) |
| Same DAG (popcount-4 carry-save, 18 MAJ3 gates), placement-only change: write-load 1.59 ms/gate @ 99.99% → safe unfused 0.89 @ 99.87% → fused 0.68 @ 99.81%; faithful dataflow 99.98% e2e on the safe-placement reproducer | [study §8, table]; [code:app/test_mvdram_compute_rows_safe.cpp] |
| June's 6.1% was our own addressing artifact — corrected in public with dated supersession banners | [study §8] |
| Two-die fastpath A/B: write-load 1.585/1.574 ms/gate @ 99.990%/99.974% e2e; fused clone-load 0.683/0.705 @ 99.808%/99.814% = 2.32×/2.23× (die A = bender 2, die B = bender 0) | [code:app/test_mvdram_fastpath_ab.cpp]; [data:docs/data/mvdram-repro/fastpath_ab_b2.log] (die A), [data:docs/data/mvdram-repro/fastpath_ab_b0.log] (die B) |
| Fast in-DRAM loading costs ~0.17% end-to-end; fusing costs nothing | [study §8] (accuracy-cost statement); arithmetic from the fastpath logs (99.990−99.808 / 99.974−99.814 ≈ 0.17 pp) |

## Scene 6 — The mechanism scoreboard

One row per scoreboard entry; "paper" column then "ours" column.

| Scoreboard row | Paper source | Our source |
|---|---|---|
| Reliable-column fraction: 83–95% vs 87–88% MAJ5-reliable | [paper Table I] | [study §4, table] |
| RowCopy: strict on screened part vs 8192/8192 deterministic (commodity) | [paper fn. 3] | [study §4, table]; [code:app/test_rowclone_smoke.cpp] |
| Dual-track adder: error-free after Frac[34]+calib[48] vs 99.94% (MAJ5 sum, screened cols) | [paper §VII] | [study §4, table] |
| — all-MAJ3 variant 99.98% | n/a (their adder uses MAJ5) [paper §II-C1] | [code:app/test_mvdram_compute_rows_safe.cpp]; [study §4 table ("in-DRAM carry-save popcount tree 99.97–99.98%") + §8] |
| On-the-fly encoding: reproduced (select-source RowClone) | [paper §V, Fig 6] | [code:app/test_mvdram_gemv_inplace.cpp] (`select`/`srcrow` product literals) |
| Bit-sparsity skip: implemented, not benchmarked at their scale | [paper §V-D] | [code:app/test_mvdram_gemv_inplace.cpp] (`PIM_SKIP` env); "not benchmarked at scale" is a scope statement, see Scene 7 |
| Horizontal layout + row-wise aggregation: reproduced | [paper §VI, Fig 7–10] | [study §4, table]; [code:app/test_mvdram_gemv_inplace.cpp] |
| Computation-rows dataflow: 99.98% with safe placement (June 6.1% reversed) | [paper Fig 2] (as characterized in [study §4]) | [code:app/test_mvdram_compute_rows_safe.cpp]; [study §8] |
| Fast in-DRAM operand movement: 2.2–2.3×/gate vs host-write shape | [paper Fig 3] (profile assumption) | [code:app/test_mvdram_fastpath_ab.cpp]; [data:docs/data/mvdram-repro/fastpath_ab_b*.log] |

## Scene 7 — What we have NOT done

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

## Scene 8 — What our silicon adds beyond the paper

| Claim | Source |
|---|---|
| Selection law: member A⊕S fires iff every predecoder unit contributes ∅ or its whole share of d; units {1,2},{3,4},{5,6},{7,8}, bits 0/9 singleton | [code:app/test_selection_timing.cpp]; [data:docs/data/selection-law/] |
| 1691/1691 member-observations explained on each of two dies; zero exceptions; zero mixed rows | [data:docs/data/selection-law/selection_timing_b0.csv], [data:...selection_timing_b2.csv] (1691 complete member rows each; `fired` column never 2) |
| Firing set identical at all 8 timing combos (t₁₂∈{5,10,20,30} × t₂₃∈{1,2}) | [data:docs/data/selection-law/selection_timing_b2.log] (per-(k,t₁₂,t₂₃) summary constant in t) |
| Screening becomes a computed predicate | [editorial: consequence]; operationalized in [code:app/test_safe_load.cpp] |
| Cross-die determinism: the per-member CSVs are byte-identical across the two dies (only the device-path preamble differs) | [data:docs/data/selection-law/selection_timing_b0.csv vs selection_timing_b2.csv] (diff = 2 preamble lines) |
| Hence a placement discipline computed on one die transfers to another of the same design | [editorial: direct consequence of the byte-identical law data] |
| MAJ3 self-pollution: on one module, a tuple rated 100% reliable has 2 of 16 open rows silently overwritten during every MAJ, absorbed by a 14-vs-2 majority | [study §4, provenance note] |
| Column-static error models (incl. reliable-column screening) cannot represent the effect — it depends on which rows are co-activated, not the column | [study §4, closing paragraph] |
| All-MAJ3 accumulation: carry-save popcount from MAJ3 only, 99.98% e2e with safe placement; no MAJ5 → no Frac/calibration dependency for the adder | [code:app/test_mvdram_compute_rows_safe.cpp] (18-MAJ3-gate DAG); [study §4 table, §8]; MAJ5 dependency of their adder: [paper §II-C1] |

## Scene 9 — Verdict

| Claim | Source |
|---|---|
| Revised verdict: MVDRAM's method is achievable on commodity DDR4 — but only with spread-aware addressing the paper does not discuss, on silicon its own screening would have rejected | [study §8] |
| June's "not achievable" is withdrawn | [study §8] |
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
