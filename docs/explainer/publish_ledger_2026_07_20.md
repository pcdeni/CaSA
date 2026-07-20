# Publish-phase verification ledger — 2026-07-20

Scope: every load-bearing quantitative claim in the four explainers as of
this pass —

- `pim_explainer.html` (13 scenes; Scenes 10–11 NEW today; base = the
  PR-#1 `july-2026-update` branch copy, which superseded the May-20 local
  file — May file kept as `pim_explainer.html.pre_2026_07_20.bak`),
- `xor_spread_explainer.html` (11 scenes; Scenes 6–7 + Scene-5 step 5 NEW
  today; same base note; May file kept as `.pre_2026_07_20.bak`),
- `mvdram_explainer.html` (11 scenes; Scenes 8–9 NEW today, Scenes 6/7/10/11
  updated; NEW as a local file — base = the PR-branch `mvdram.html`),
- `system_explainer.html` (NEW narrative; quantitative claims are the same
  ones as the decks' and are grounded through the deck ledgers + this file).

Method: each claim traced to its PRIMARY artifact (campaign log/CSV/table
file) and re-read today where the artifact is on this machine; arithmetic
recomputed where the claim is derived (s/tok, ratios, ranges). Claims whose
primary log is not retained in the workspace are marked **SECONDARY** (the
durable record is RESULT.md / a memory file) — they are NOT softened in the
decks; they are listed here per the no-silent-deletion rule.

Status codes: **V** = verified against the primary artifact today ·
**V(a)** = verified arithmetic/derivation from verified inputs ·
**S** = secondary record only (primary log scratchpad-resident or
truncated) · **F** = FLAGGED discrepancy (listed at the end).

## Stats

- Claims checked: **74**
- **V / V(a): 61** (direct artifact or recomputed derivation)
- **S: 9** (secondary/durable-record-only — none softened)
- **F: 4** (discrepancies — all listed, none silently fixed in sources)

---

## A. pim_explainer.html

| # | Claim (scene) | Primary source | Status |
|---|---|---|---|
| A1 | Fused tool A/B iters=100: A 0.357 → B2 0.089 ms/MAJ = 3.99×, 0/204,800 (die A) (S9/S11) | sublattice_broadcast_2026_07_17/fused_final_b2.log lines 312-315 | **V** (re-read today) |
| A2 | Die B: 0.556 → 0.085 = 6.53×, 0/204,800 (S9/S11) | fused_final_b0.log lines 312-315 | **V** |
| A3 | B1 unbalanced 7W/4x/4z wrong: 204,140/204,800 (99.7%) b2; 200,450/204,800 (97.9%) b0 (S9) | fused_final_b{2,0}.log B1 rows | **V** |
| A4 | Server fused 1.45–1.6×/matmul, bit-exact both arms (S9) | RESULT.md addendum 2 table; ab_server logs in BitNet dir | **V** (ab_4bank/ab_server logs present; numbers as recorded) |
| A5 | Real-model layer-0 A/B: server-request 112.5 → 70.2 s = 1.60× (S9) | token_ab_baseline.log / token_ab_fused.log totals lines (112531 ms / 70233 ms) | **V** |
| A6 | Real-model wall 117.2 → 71.8 s = 1.63×, both arms "Paris" (S9) | addendum 6 (client wall); logs show forward=113.0/70.7 s + startup | **V(a)** — wall = forward + load/startup; see note N1 |
| A7 | Road A in-DRAM popcount K=16/32/64: lanes 99.47/98.98/98.64% exact; bit-err 0.182/0.341/0.488%; reduction 3.2/5.3/9.1× (S10) | popcount_indram.log "===" summary lines | **V** (exact complements of logged 0.5288/1.0171/1.3613% wrong) |
| A8 | K=8 row: 99.73% lanes / 0.111% / 2.0× (S10) | RESULT.md addendum 14 (log retained on disk starts mid-K=16 — K=8 summary not in the surviving file) | **F1** |
| A9 | 213× at K=2560 (S10) | arithmetic K/ceil(log2(K+1)) = 2560/12 = 213.3 | **V(a)** |
| A10 | Non-accumulating: 16-bit chain carry 0.0000% all positions, sum 0.009–0.068% (S10) | chain_b2_s78_16bit.csv (bit12-15 rows re-read: sum 0.0085–0.0682%, carry 0.0000%) | **V** |
| A11 | Host-marshalled in-DRAM ~320× slower at K=64 (0.706 s vs ≈2.2 ms) (S10) | RESULT.md addendum 14b (measured correction section) | **S** (addendum-recorded measurement; no separate log file) |
| A12 | Packed primitive rates 34.2 µs/read, 2.6 µs/MAJ; crossover margin 4.4× (S10) | RESULT.md addendum 14b table (packed-maj-exe M=29 run) | **S** (tool output recorded in addendum only) |
| A13 | Resident tile 58 ops/product; 4–6× slower single-bank; sparsity ops 522→94, err 2.99→1.15% (S10) | RESULT.md addendum 17b tables | **S** |
| A14 | Multi-body packing ~2.3×; M=8: 2.58×; M=29 at 7688 insts (S10) | RESULT.md addendum 15 table | **S** |
| A15 | Road B 2048× (8 KiB→4 B), Verilator bit-exact 5/5 (S10) | ADR-005 + memory bitnet_bus_bound_hdl_staged | **V** (ADR-005 re-read) |
| A16 | pop_count4 0xE undercount found+fixed; fix observed firing on silicon (0xE → 49152) (S10) | RESULT.md addenda 20/20c | **V** (addendum 20c: "including 0xE → 49152 twice") |
| A17 | build4 53/53 Verilator checks; build3 40/40 incl. 3 bug demos; Vivado build in flight (S10) | RESULT.md addendum 20d | **V** (addendum table re-read; build log poll command recorded) |
| A18 | Arc: 632 → 360.8 → 137.1 → 80.5 → 47.5 = 13.3× (S11) | 632/438/360.8: addendum 19b; 137.1: o8 log (1096.7 s/8); 80.5: memory bitnet_optimization_state O2 close; 47.5: addendum 25 (1140.6 s/24) | **V** for 137.1 + 47.5 arithmetic; **S** for 632/360.8 walls (addendum tables) and 80.5 (see F3) |
| A19 | Four host bugs (pool path / xbp cache / V2-LOAD collision / mis-scoped extras); silicon clean; full-model rule (S11) | RESULT.md addenda 18–19; memories bitnet_xbp_cache_bug, bitnet_pool_collision_and_extras | **V** (records re-read; mechanisms + fixes as stated) |
| A20 | Voting economics rows: 360.8 voted / 134.8 unvoted-garbage (May pool); 137.1 unvoted-Paris / 372.5 voted byte-identical (clone-ok) (S11) | o8_fullmodel_cloneok_unvoted.log (8 tok in 1096.7 s) + _voted.log (2979.9 s), identical response strings | **V** (walls + identical text re-read today); 134.8 = T5 record (memory) → **S** |
| A21 | 48-tok steady state 80.5 s/tok, text stable, voting no benefit (S11) | memory bitnet_optimization_state "O2 close" | **F3** (primary log scratchpad-resident) |
| A22 | Dual-DIMM defect: D0 15,510 calls/23.5 GB vs D2 628/51 MB; 90.3 s/tok; fix = split-V2; 47.5 s/tok; balance 0.01%; 1.91×/token-matmul at 96% of bound (S11) | RESULT.md addendum 25 (all tables); arithmetic re-checked (2167.0/24=90.3; 1140.6/24=47.5; 70.2/36.8=1.91) | **V(a)** on arithmetic; **S** on the raw logs (F4) |
| A23 | Drift predicted the consts regression; fixed run byte-identical to control (S11) | BitNet/o4_constsfix_silicon_2026_07_19/fullmodel_constsfix.out ('The capital of France is Paris. Paris', 6060.9 s) vs control 6064.3 s (addendum 23) | **V** (decisive-arm output re-read today) |
| A24 | POPCNT3 27–348× vs A100; SiMRA/FracDRAM/FCDRAM rows (S1–S9, unchanged) | paper ledger rows (pim_explainer_ledger.md, adversarial review 2026-05) | **V** (carried; re-reviewed 05-20 pass) |

## B. xor_spread_explainer.html

| # | Claim (scene) | Primary source | Status |
|---|---|---|---|
| B1 | Selection law 1691/1691 ×2 dies, zero exceptions, timing-invariant (S5) | selection_timing_b{0,2}.csv/.log | **V** (CSVs present; b2 log tail shows constant per-timing summaries) |
| B2 | Cross-die byte-identical CSVs (only device banner differs) (S5) | diff selection_timing_b0.csv selection_timing_b2.csv → exactly the 2 banner lines | **V** (diff run today) |
| B3 | Full-set% by k = 100/100/75/25/0 (S5) | selection_timing_b2.log summaries | **V** |
| B4 | 1024-block scope: same-block cross-640 cosets 8/8 + 16/16 exact; cross-block no coset (S5 step 5) | fcdram_b0_v2.csv + RESULT.md addendum 11 | **V** (CSV present; addendum hand-verified memberships) |
| B5 | Bit 9 dead deposit generator: d=512/513 → 0.00% vs 100.00% for d=1/16/17 (S5 step 5) | RESULT.md addendum 17 §2 (gemv-encoded probe) | **S** (probe output recorded in addendum; tool in repo) |
| B6 | Partial dies: best partition fits 61% / 42%, weakly timing-dependent (S5 step 5) | seltiming_b1.csv/.log, seltiming_b3.csv/.log + addendum 9 | **V** (files present; addendum table) |
| B7 | Clone-dead: ~17% of rows; median 1,140/2,048, min ~400; deterministic per (tuple,d_low) (S6) | dimm2_fault_sweep_subs_2026_07_18 sweep logs + addendum 22(b) | **V** (campaign files present; census restated in 3 addenda consistently) |
| B8 | Closed form (u≥5; u==4 bit0∧bit9∧{G1,G2,G3}); u≤3 zero exceptions 6,128/6,128; 99.980% on even-Rf; zero false-dead (S6) | o8_clone_law_and_cloneok_pool.py + clone_law.py + addendum 22(b) | **V** (script + predicate present; acc line in script) |
| B9 | Held-out 2,496/2,496 with pre-committed predictions (S6) | o8_sub85_clone_predictions.txt (627 lines = 3 header + 624 predictions, header states pre-silicon commitment) + sub85_bank{0-3}.log | **V** (predictions file + sweep logs re-read today) |
| B10 | Cross-die 2,494/2,496, zero false-dead; banks 2/3 perfect (S6) | dimm0_fault_sweep_2026_07_20/o5_dimm0_clone_predictions.txt + dimm0_bank{0-3}.log + addendum 24 §1 | **V** (files present) |
| B11 | Anti-selection: 108/294 production pool, 9/16 tail; D0 pools 32.8–39.1% dead (S6) | addendum 22(b) audit + addendum 24 §1 | **V** (both audits re-read; mechanism consistent with degree-0 observation) |
| B12 | Rf-parity caveat (sub77 odd anchor, 97.9%) (S6) | addendum 22(b) | **V** (stated as caveat in deck, matching record) |
| B13 | Drift arms R1–R7: 52/80 baseline; R4 idle 72/80 + y-bad 80/80; saturation ordered u≤3→~84% vs u5→~36%; byte-lane masks (S7) | o8_arm_R{1-7}*.log + o8_drift_correlation.py + addendum 22(c) | **V** (arm logs present; numbers from the addendum's table over those logs) |
| B14 | Refresh exonerated twice; label-collision bug era covariate (S7) | addendum 22 part 1 + 22(c) R2≡R3 | **V** |
| B15 | Consts regression + per-request rewrite fix; pre-fix babble 6021.1 s; fixed = byte-identical Paris 6060.9 vs control 6064.3 s; residual ~0.3% protocol tail (S7) | o4_constsfix_silicon_2026_07_19/ (fullmodel_constsfix.out re-read today; ab_long300_* logs present: 300/298/299 per addendum 23) | **V** |
| B16 | Fused/coset + safe-load rows carried from 07-17 (S11): 20/20 clean; 16/16 fast load; 99.98% dataflow | safeload_b2.log (FAST_LOAD_OK 16/16 re-read); cr_safe_screened_b2.log (99.9756% e2e re-read) | **V** |
| B17 | Original scenes 1–4, 8–10 chip-example figures | May adversarial pass (pim_explainer_review.md) + xor ledger | **V** (carried) |

## C. mvdram_explainer.html

| # | Claim (scene) | Primary source | Status |
|---|---|---|---|
| C1 | Paper rows (Table I 82.9–95.1%; 0.19 ms headline; 2.18×/1.31×; CACTI-estimated energy; Orin Nano baseline; fn.3 16-module screening; §V-E ~1.5 ns/cmd) (S1/S2/S11) | mvdram_2503.23817.pdf via PAPER_CONTRAST.md §1 (re-read 07-17 against v2 PDF) | **V** (as extracted in PAPER_CONTRAST; quotations <15 words) |
| C2 | Result A: 0/60,000 pairs on the named part (S4) | rcrand_b0.log, rcrand_b3.log (mvdram-repro/) | **V** (files present; carried from PR ledger pass) |
| C3 | June 6.14% → safe placement 99.98%/99.96% e2e; 54/54 loads; per-op 99.99% (S5/S6) | cr_orig_b2.log (6.14471%) + cr_safe_screened_b{2,0}.log (99.9756/99.9588%) | **V** (re-read today) |
| C4 | Fastpath C/A/B: 1.585/0.885/0.683 ms/gate b2; 1.574/0.898/0.705 b0; e2e 99.990→99.808 / 99.974→99.814 (S5/S6) | fastpath_ab_b2.log, fastpath_ab_b0.log | **V** (re-read today) |
| C5 | MAJ5 colmasks 94–95% strict ×8 banks; winners 1919–1954/2048 (S6/S7) | colmasks_maj5_zero_2026_07_17/ + addendum 10 §C table | **V** (files present; b2/s86 anchors cross-checked: m5_Z2 1958 in majx_b2_s86maj5.log) |
| C6 | Reference-policy law: ZERO 963 vs frac'd-ONE 157–206 (s72/b0); ZERO+2 1915 strict / 2010 soft95 (s86 600-run) (S7) | frac_maj5_b0_s72.log ("ZERO no frac 963" family) + frac_maj5_b2_s86_confirm.log (n_frac=2 tf=0 ZERO → 1915/2010 re-read today) | **V** |
| C7 | MAJ3 frac-flat 78/78 configs; corr −0.44 on DIMM 1; no rescue (S7) | frac_sweep_b2_s72.csv, frac_sweep_b1_s24.csv + addendum 7 | **V** (files present) |
| C8 | Exact adder strictly error-free on 1942/2048 = 94.8% (ZERO+2); 16-bit chain carry 0.0000% (S6/S7) | addendum 8b table + chain_b2_s78_16bit.csv | **V** (chain CSV re-read; 8b table consistent with majx/confirm anchors) |
| C9 | MAJX menu rows: m7_ZZ 737 (b2) / 693 (b0); MAJ9+ dead ≤22; 32-row law-built MAJ3 2048/2048; their-timing transfer = 0 (S7 context) | majx_b2_s86maj5.log, majx_b0_s72.log, majx32_b0_s77.log (all re-read today) | **V** |
| C10 | Gap-3 size: 17.04±0.31 s vs 0.19 ms ≈ 9×10⁴ (S7/S8) | b2_results/20260719_233500/b2_gemv_table.md (row re-read) ÷ paper §VIII-B | **V(a)** (17.04 s / 0.19 ms = 8.97×10⁴) |
| C11 | B2 table: 32 cells, 5 iters, both arms, CPU column, silicon; quoted cells 34.09±0.64 / 18.88±0.39 / 17.04±0.31 / 14.83±0.33 / 8.56±0.20 / 7.51±0.12 (S8) | same table file | **V** (all six cells re-read today) |
| C12 | 36 verified e2e ops; per-model ranges (7B q4 99.81–99.90, 7B q2 99.83–99.90; overall 99.58–99.99) (S8) | smoke_2026_07_19/silicon/e2e_*_ops.log — counted 36 today; per-file min/max recomputed | **V** (and see **F2** for the REPRODUCTION.md range statement) |
| C13 | Quant mappings bit-exact vs gguf pre-silicon (q4_0/q2_K/q3_K/q6_K) (S8) | REPRODUCTION.md 07-18/19 dry-run evidence sections (check_* artifacts in mvdram_bench/smoke dirs) | **V** (artifact dirs present) |
| C14 | 07-18 smoke ops: attn_q 160.4 s 99.90%, attn_v 157.7 s 99.88% (S8 context) | mvdram_pim_smoke_ops.log | **V** (re-read today) |
| C15 | O7 closure numbers: R1 38.4 s/99.976%; R3b 11.9 s/94.65% (3.2×, 5.0× fewer pcwrites); R2 76.0 s/99.49% (viol 0.063%); R2v 229.8 s/99.83% (0.021%); R4 732.8 s, 524,288/524,288, fp32 bit-exact 4096/4096 (S9/S6) | o7_logs_2026_07_20/R1,R2,R2v,R3b,R4 logs | **V** (all five re-read today) |
| C16 | Row-budget N ≤ ~160 vs their N≤128; four paper ambiguities (S9) | REPRODUCTION.md O7 section ([editorial: arithmetic] as tagged) | **V** (as editorial-tagged derivation) |
| C17 | Lane2 phase-1: 4096² unvoted 99.93–99.95% at 37 s; vote3 bit-exact ×2 at 110 s (S8 context) | REPRODUCTION.md 07-18 phase-1 table | **S** (client smoke stdout not retained as a file; recorded in REPRODUCTION.md at write time) |
| C18 | Clone-dead + reference-policy "beyond the paper" cards (S10) | B8–B10 + C6 rows above | **V** (primary) |

## D. system_explainer.html

All quantitative claims in the narrative are restatements of rows above
(arc numbers → A18–A22; laws → B1–B15; reproduction → C1–C17; refresh
semantics → B14/B15 mechanism rows; four bugs → A19). No new numbers were
introduced. Two editorial simplifications to note: "a few hundred thousand
MAJ operations per token" is an order-of-magnitude statement derived from
the per-projection MAJ-count formula (2·8·(d_in/32)·d_out across 210
projections, sliced and batched in production) — labelled approximate; the
"210 weight-matrix multiplications" count = 30 layers × 7 projections
(model architecture, config.json). Status: **V(a)**.

---

## FLAGGED items (discrepancies + ungroundable claims — none silently altered)

- **F1 — Road-A K=8 row (A8).** The addendum-14 table's K=8 line
  (99.73% lanes / 0.111% bit-err / 2.0×) is not present in the surviving
  `popcount_indram.log`, which begins mid-K=16 (the log was evidently
  overwritten/truncated at 16 lines). K=16/32/64 verify exactly. The K=8
  row stays in the deck sourced to RESULT.md addendum 14 (the durable
  record), with this flag. A cheap re-run would close it; not run today
  (FPGA owned by another agent).
- **F2 — REPRODUCTION.md "99.58–99.98%" (C12).** Recomputing over the 24
  non-7B ops gives min 99.5759% and **max 99.9928%** (13,823/13,824,
  Llama-2-13B Q2_K) — the doc's upper bound understates by 0.01 pp. The
  new explainer states **99.58–99.99%**. Proposed one-character fix to
  REPRODUCTION.md (and to the PR-branch copy) is listed in
  `repo_sync_plan_2026_07_20.md`; the source doc was NOT silently edited.
- **F3 — the 80.5 s/tok O2 point (A18/A21).** The 48-token run's primary
  client log lived in a sibling session's scratchpad and is not retained
  in the workspace. Durable records: memory `bitnet_optimization_state`
  ("2026-07-20: steady-state headline 80.5 (O2 close)") and RESULT.md
  addendum 25's three independent cross-references (80.5 named as the
  single-DIMM baseline; o2 traffic profile 27,778 calls / 41.9 GB; the
  70.2 s/token-matmul figure 3862.7/55 used in the balanced-run
  normalization). Internally consistent; primary artifact absent — carried
  as SECONDARY everywhere it appears.
- **F4 — addendum-25/o5 primary logs (A22).** `o5fix_dualdimm_24tok_
  balanced.log`, the o5 defective-run log, and the o4/o5 layer-0 A/B logs
  are listed in RESULT.md as scratchpad-resident and are no longer on
  disk (the per-session scratchpads were cleaned). RESULT.md addendum 25
  is the durable record; its internal arithmetic re-verifies exactly
  (90.3 = 2167.0/24; 47.5 = 1140.6/24; 1.91 = 70.2/36.8; bound 45.6 =
  1093.7/24). Carried as SECONDARY. Recommendation (in the sync plan):
  future campaign logs go to workspace campaign dirs, never scratchpads —
  the O8/O9/O10/o4 campaigns already did this correctly.

## Minor notes (not flags)

- **N1 (A6):** the 117.2/71.8 s walls are client run-walls; the retained
  logs' `forward=` totals are 113.0/70.7 s (delta = model load + prefill
  bookkeeping). The server-request times (112.5/70.2 s) verify exactly;
  the 1.63×/1.60× ratios hold under either accounting.
- **N2:** addendum 5 says "1,693-line CSVs"; today's `wc -l` gives 1,695
  (2 device-banner lines + header + 1,691 data + trailing summary). The
  load-bearing claims (1,691 member observations; byte-identical minus
  banner) verify directly; the line-count aside in the addendum is off by
  two and is not quoted in any deck.
- **N3:** addendum 14's per-tile seconds column (0.08–0.59 s/tile) is a
  derived per-tile figure; the log prints per-K totals (2.20–3.53 s).
  The decks quote only the verified lane/bit-err/reduction numbers.
- **N4:** the fcdram "no NOT on any tested die" verdict has no single
  verdict line in the CSVs (they are per-config grids); the deck sources
  it to addendum 11 + the four CSV/log files, which are present.
- **N5:** o9 pilot walls re-derived: 1094.5/8 = 136.8, 1102.4/8 = 137.8 —
  match the deck's Scene-11 context row and addendum 27.
