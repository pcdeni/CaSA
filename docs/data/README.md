# `docs/data/` — the measurement files the explainer ledgers cite

Every load-bearing number in the
[explainer pages](https://pcdeni.github.io/CaSA/explainer/) traces to a primary
artifact. This folder holds those artifacts. The claim IDs below (A#/B#/C#/N#)
are the entries the explainer claim ledgers cite —
[`../explainer/index_ledger_2026_08_03.md`](../explainer/index_ledger_2026_08_03.md)
and
[`../explainer/xor-spread_ledger_2026_08_03.md`](../explainer/xor-spread_ledger_2026_08_03.md).

Logs are raw, unedited tool output. CSVs are per-config grids. `*_server.log`
is the PIM server's side of an A/B run. Four claims are carried as **SECONDARY**
(primary log not retained) and are flagged F1–F4 in the publish ledger; only
F1 touches a file present here (noted below).

## Carried from the 07-17 PR (unchanged)

- `selection-law/selection_timing_b{0,2}.csv/.log` — the selection law,
  1691/1691 on each of two dies, timing-invariant (B1–B3).
- `mvdram-repro/` — Result A `rcrand_b{0,3}.log` (0/60,000 pairs, C2) and the
  fastpath A/B logs `fastpath_ab_b{0,2}.log` (C4).

## merge/ — Road A / in-DRAM accumulation (pim deck Scene 10)

| File | What it is | Cites |
|---|---|---|
| `popcount_indram.log` | Road-A in-DRAM popcount, K=16/32/64 lane%/bit-err/reduction | A7 — **K=8 row's summary lost, F1** (durable record: RESULT.md addendum 14) |
| `chain_b2_s78_16bit.csv` | 16-bit dual-track ripple chain: carry 0.0000% all positions, sum 0.009–0.068% | A10, C8 |
| `chain_b2_s78_Z2.csv`, `chain_b0_s77_Z2.csv` | ZERO+2-policy chain, both dies (non-accumulation proof) | A10 |

## majx-policy/ — MAJX × reference-policy screen (mvdram deck Scenes 6–7)

| File | What it is | Cites |
|---|---|---|
| `majx_b2_s86maj5.{csv,log}` | MAJ5 colmask + MAJX menu on die b2 / s86 (m5_Z2 1958; winners 1919–1954) | C5, C9 |
| `majx_b0_s72.{csv,log}` | MAJX menu on die b0 / s72 | C9 |
| `majx32_b0_s77.{csv,log}` | 32-row law-built cosets: MAJ3 2048/2048; their-timing transfer = 0 | C9 |
| `frac_maj5_b0_s72.{csv,log}` | Reference-policy law: ZERO 963 vs frac'd-ONE 157–206 | C6 |
| `frac_maj5_b2_s86_confirm.{csv,log}` | ZERO+2 confirm (600-run): 1915 strict / 2010 soft95 | C6, C8 |
| `frac_sweep_b2_s72.csv`, `frac_sweep_b1_s24.csv` | MAJ3 frac-flat 78/78; corr −0.44 on DIMM 1 (no rescue) | C7 |

## fcdram-not/ — FCDRAM NOT negative + 1024-block scope (xor deck Scene 5 step 5)

| File | What it is | Cites |
|---|---|---|
| `fcdram_b{0,1,3}.csv` | Per-config NOT grids: no NOT on any tested die | N4 (verdict = addendum 11 + these grids) |
| `fcdram_b0_v2.csv` | 1024-block scope: same-block cross-640 cosets 8/8 + 16/16; cross-block none | B4 |

## selection-law/ (additions) — partial dies

| File | What it is | Cites |
|---|---|---|
| `seltiming_b{1,3}.{csv,log}` | H-class partial dies: best partition fits 61% / 42%, weakly timing-dependent | B6 |

## clone-law/ — the clone-dead closed form (xor deck Scene 6)

| File | What it is | Cites |
|---|---|---|
| `o8_clone_law_and_cloneok_pool.py` | The closed-form predicate + clone-ok pool builder (accuracy line in-script) | B8 |
| `o8_sub85_clone_predictions.txt` | Pre-committed held-out predictions (header states pre-silicon commitment) | B9 |
| `sub85_bank{0-3}.log` | The held-out sweep: 2,496/2,496 match | B9 |
| `o5_dimm0_clone_predictions.txt` | Cross-die pre-committed predictions (DIMM 0) | B10 |
| `dimm0_bank{0-3}.log` | Cross-die sweep: 2,494/2,496, zero false-dead; banks 2/3 perfect | B10 |
| `o8_drift_arms.py`, `o8_drift_correlation.py` | Drift-arm driver + correlation analysis | B13 (see drift/) |

## drift/ — drift arms R1–R7 (xor deck Scene 7)

`o8_arm_R{1..7}*.log` (14 files; each arm has a client log + a `_server.log`).
R1 prim-only (baseline 52/80), R2/R3 overflow refresh on/off (R2≡R3 exonerates
refresh), R4 idle-300 (72/80 + y-bad 80/80), R5 v2-sub71, R6 mm3d-fused,
R7 v2-prim. Saturation ordered u≤3 → ~84% vs u5 → ~36%. Cites **B13, B14**.

## fullmodel/ — full-model runs (pim deck Scene 11 / xor Scene 7)

| File | What it is | Cites |
|---|---|---|
| `o8_fullmodel_cloneok_unvoted.log` | 8 tok in 1096.7 s → 137.1 s/tok "Paris" (clone-ok pool) | A18, A20 |
| `o8_fullmodel_cloneok_voted.log` | Voted arm 2979.9 s, byte-identical response text | A20 |
| `fullmodel_o1_fused.log` | Fused full-model arm | A18 (arc) |
| `fullmodel_constsfix.out` | Consts-fix decisive arm: "…France is Paris. Paris", 6060.9 s vs control 6064.3 s | A23, B15 |
| `fullmodel_constsfix_run1_KILLED.out` | The killed first attempt (kept as the record that run 1 was aborted) | B15 context |
| `o9_pilot_{control,pilot}.log` | O9 fresh-subarray pilot: walls 1094.5/8 = 136.8, 1102.4/8 = 137.8 | N5, addendum 27 |

## lane2/ — Lane-2 in-DRAM GeMV (mvdram deck Scene 8)

| File | What it is | Cites |
|---|---|---|
| `b2_gemv_table.{md,csv}` | The B2 GeMV table (32 cells, 5 iters, both arms + CPU column) from the 20260719_233500 silicon run; quoted cells + gap-3 size 17.04 s vs 0.19 ms ≈ 9×10⁴ | C10, C11 |
| `o7_logs_2026_07_20/R1_host_4096.log` | R1 host arm: 38.4 s / 99.976% | C15 |
| `o7_logs_2026_07_20/R2_dualtrack_4096.log` | R2 dual-track: 76.0 s / 99.49% (viol 0.063%) | C15 |
| `o7_logs_2026_07_20/R2v_dualtrack_vote3_4096.log` | R2v vote-3: 229.8 s / 99.83% (0.021%) | C15 |
| `o7_logs_2026_07_20/R3b_clone_4096_depthscreen.log` | R3b clone + depth screen: 11.9 s / 94.65% (3.2×, 5.0× fewer pcwrites) | C15 |
| `o7_logs_2026_07_20/R3_clone_4096.log` | R3 clone (un-screened companion to R3b) | C15 context |
| `o7_logs_2026_07_20/R4_fp32_realtensor.log` | R4 exact fp32: 732.8 s, 524,288/524,288, bit-exact 4096/4096 | C15 |
| `o7_logs_2026_07_20/smoke{A,B,C,D,D2}_*.log` | The gated smoke arms (default / dual-track / partials / clone / clone-q2) | C17 context |
| `e2e_*_ops.log` (8) | The 07-19 e2e op logs; 36 verified ops, per-model ranges, overall int-exact 99.58–99.99% | C12, C13 |
| `mvdram_pim_smoke_ops.log` | 07-18 smoke: attn_q 160.4 s 99.90%, attn_v 157.7 s 99.88% | C14 |
