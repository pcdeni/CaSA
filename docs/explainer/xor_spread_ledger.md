# XOR-Spread Explainer — Claim Ledger (July 2026 update; extended 2026-07-20)

> ⚠ **RETIRED 2026-08-04.** This ledger gated an earlier version of the
> mechanism deck. The live mechanism explainer is `xor-spread.html`, and its
> publish gate is
> **[xor-spread_ledger_2026_08_03.md](xor-spread_ledger_2026_08_03.md)**. This
> file is kept only as prior provenance — do not treat its scope or claims as
> current.

**Scope:** this ledger covers the scenes **added or changed in the July
2026 updates** of `xor-spread.html` (workspace copy:
`xor_spread_explainer.html`):

- Scene 3 (reframed 07-17: settled provenance),
- Scene 5 (NEW 07-17: the selection law; step 5 added 07-20: scope/limits),
- Scene 6 (NEW 07-20: the clone-dead law),
- Scene 7 (NEW 07-20: traffic-induced drift),
- Scene 11 (was Scene 9; steps 1, 3–4 changed 07-17: shipped "use it"
  results; step 2 amended 07-20: clone-law pool filter),
- the header banner (unchanged; its claims are re-listed under Scene 3
  because the scene now carries them too).

Scene renumbering 2026-07-20 (two scenes inserted after Scene 5): old
6→8 (the copy-timing operand deposit), 7→9 (success-rate signature), 8→10 (why it
matters), 9→11 (two responses). Scenes 1–2, 4, 8–10 are otherwise
unchanged from the original deck (their chip-specific figures are
labelled as examples or as this project's own sweep measurements in the
deck itself; adversarial-review record in
[`pim_explainer_review.md`](pim_explainer_review.md) covers the original
publication pass).

Workspace-data tier note (2026-07-20): rows sourced to July-18/20
campaign directories cite the tower workspace paths; the files are
staged for `docs/data/` in `repo_sync_plan_2026_07_20.md` and the tags
will flip to `[data:docs/data/...]` when the sync lands.

Source tiers (same convention as [`pim_explainer_ledger.md`](pim_explainer_ledger.md)):
- **[paper:SiMRA §SEC]** — Yüksel et al., arXiv:2405.06081
- **[code:PATH]** — reproducer in this repository
- **[data:PATH]** — measurement file in `docs/data/`
- **[issue:URL]** — linked GitHub issue
- **[study §N]** — [`docs/MVDRAM_REPRODUCTION.md`](../MVDRAM_REPRODUCTION.md)
- **[example]** — chip-specific value labelled as example
- **[editorial]** — framing, grounded in adjacent cited rows

---

## Scene 3 — The mechanism: Multi-RowCopy seen from the source side (REFRAMED 2026-07-17)

The scene no longer presents the mechanism as unexplained. Provenance is
settled: the spread **is** SiMRA's Multi-RowCopy co-activation, seen from
the source side of the APA pair.

| Claim | Source |
|---|---|
| The row decoder is hierarchical (global decoder → local predecoder banks latching predecoded signals) | [paper:SiMRA §7.1] |
| Short PRE→ACT gap lets a predecoder latch the next address without releasing the previous one → many rows open at once (up to 2⁵ = 32); this is the documented Multi-RowCopy / MAJX substrate | [paper:SiMRA §3.4, §7.1] |
| The co-activated set is the XOR lattice `{src ⊕ S : S ⊆ bits(src⊕dst)}`; the calibrated tuple is this lattice used deliberately from the destination side | [issue:https://github.com/CMU-SAFARI/DRAM-Bender/issues/12] (SiMRA co-author's confirmation of the mechanism) |
| The spread is the same lattice seen from the source side — provenance settled July 2026 | [issue:https://github.com/CMU-SAFARI/DRAM-Bender/issues/12] |
| Verified on our own data: all 696 production tuples are exact (R_F, R_S) lattices; 3,941/3,941 recorded fault rows lie inside the pair lattice | [issue:https://github.com/CMU-SAFARI/SiMRA-DRAM/issues/1] (our verification post); restated in [study §4, provenance note] |
| While sibling wordlines are open, the sense amp drives them to the source's value (deposit mechanism) | [paper:SiMRA §3.4] (sense amp overwrites all activated rows with the source row's data); [issue:DRAM-Bender#12] |
| Supporting evidence lines: bank-invariance (decoder, not cells), timing-independence (wordline, not charge), per-die gap bit | this deck's Scenes 2 and 4 (unchanged claims, project sweep measurements); [study §4] |

## Scene 5 — The selection law (NEW 2026-07-17)

| Claim | Source |
|---|---|
| For pair (A, A⊕d), the lattice predicts 2^k candidates `{A⊕S : S⊆bits(d)}`; only a subset fires | [code:app/test_selection_timing.cpp] (the question the tool answers); [data:docs/data/selection-law/] |
| Subarray-local address bits fall into predecoder units: pairs {1,2}, {3,4}, {5,6}, {7,8}; bits 0 and 9 singleton (on our modules) | [data:docs/data/selection-law/selection_timing_b0.csv, selection_timing_b2.csv] (the grouping that explains every row); consistent with the safe-offset generator rules in [code:app/test_safe_load.cpp] |
| The law: member A⊕S fires iff for every unit g, S∩g ∈ {∅, d∩g} | [code:app/test_selection_timing.cpp] + [data:docs/data/selection-law/] (zero-exception fit) |
| Worked example d={1,2,5}: fired members exactly A⊕{5}, A⊕{1,2}, A⊕{1,2,5} | [example] — an application of the law, labelled EXAMPLE in the caption |
| Method: marker → A; zeros → every other predicted member; doubleACT(t₁,t₂,A,A⊕d); read every member; fired(S) := content == marker | [code:app/test_selection_timing.cpp] (header, steps 1–4) |
| 1691/1691 member-observations explained on EACH of two dies; zero exceptions; zero mixed (partial-copy) rows | [data:selection_timing_b0.csv], [data:selection_timing_b2.csv] — 1691 complete member rows per die; `fired` column ∈ {0,1} throughout (never 2 = mixed) |
| Firing set identical at all 8 timing combos: t₁₂∈{5,10,20,30} × t₂₃∈{1,2} | [data:docs/data/selection-law/selection_timing_b2.log] (per-(k,t₁₂,t₂₃) summaries constant across timings) |
| Per-member CSVs byte-identical across the two dies (only the device-path preamble differs) | [data:selection_timing_b0.csv vs selection_timing_b2.csv] (diff = the two `/dev/xdma0_*` lines) |
| Fired count = 2^(number of units d touches); full 2^k set fires iff d contains no complete pair-unit | [data:docs/data/selection-law/] (counting law fitted with zero exceptions); [code:app/test_selection_timing.cpp] |
| Measured full-set rate by k (4 pairs each, this module): 100 / 100 / 75 / 25 / 0% for k=1..5 | [data:docs/data/selection-law/selection_timing_b2.log] (`full-set%` column) |
| k-decay is combinatorial, not stochastic; screening becomes a predicate computed from d | [editorial: consequence of the two rows above]; operationalized in [code:app/test_safe_load.cpp] |

### Scene 5, step 5 (added 2026-07-20) — scope and limits

| Claim | Source |
|---|---|
| "Subarray-local" = low-10-bit-local within a **1024-row predecoder block**; 640-row units are sense-amp segments only | [data:sublattice_broadcast_2026_07_17/fcdram_b0_v2.csv] + RESULT.md addendum 11 (same-block cross-640 pairs: d=248 coset 8/8 exact both directions; d=444 coset 16/16 exact; hand-verified memberships) |
| Cross-1024-block pairs deposit NO lattice coset (n_full 0–3, non-coset) and degrade the destination's own write | RESULT.md addendum 11 (fcdram_b0_v2 run grid) |
| Allocator rule: plan placement block-scoped; a coset crossing a 640 boundary WILL deposit into the neighboring segment | [editorial: consequence]; operationalized in [code:app/lattice_alloc.h] (1024-block containment check) |
| Bit 9 is a DEAD deposit generator on this tuple class: d=1/16/17 → 100.00% deposit; d=512/513 → 0.00% | RESULT.md addendum 17 §2 (gemv-encoded clone probe, s78) |
| Consequence: only bits {0,4} give clean single-bit scratch regions on this tuple class; the production pair's 16-row (not 32-row) tuple is retro-explained | RESULT.md addendum 17 §2; [editorial: consequence] |
| On the two partial dies, k=1 deposits are reliable but k≥2 selection fits NO clean partition (best 61% / 42%) and weakly depends on timing (t₂₃=2 suppresses) | [data:sublattice_broadcast_2026_07_17/seltiming_b1.csv, seltiming_b3.csv] + RESULT.md addendum 9 |
| This is a third independent corroboration that the partial dies are structurally different (after yield maps and MAJ3 limits) | [editorial], grounded in addendum 9 + the dimm13 verdict (RESULT.md Part 7) |

## Scene 6 — The clone-dead law (NEW 2026-07-20)

| Claim | Source |
|---|---|
| ~17% of every window's rows cannot RowClone into their tuple's Rfirst; partial clones, median 1,140/2,048 words, min ~400 | [data:dimm2_fault_sweep_subs_2026_07_18/] (5 sweeps × 4 banks, PIM_CHECK_CLONE=1) + RESULT.md addendum 21/22(b) |
| 12,480 clone observations; verdict deterministic per (tuple, d_low): 3,106/3,120 rows bank-consistent (14 bank-marginal), zero rows in both ok- and fail-sets per tuple | RESULT.md addendum 22(b) |
| Closed form (even-Rfirst): u = [bit0] + #touched pair-units; DEAD iff u≥5, OR u==4 ∧ bit0 ∧ bit9 ∧ touched units exactly {G1,G2,G3} (all 27 classes dead, all 81 siblings ok) | [code:app/clone_law.py] (the predicate); RESULT.md addendum 22(b) |
| u≤3 clone-OK with zero exceptions: 6,128/6,128 observations incl. the odd-Rf tuple | RESULT.md addendum 22(b) |
| Accuracy 99.980% on the 9,984 even-Rf observations; only misses = 2 bank-marginal rows; zero false-dead | RESULT.md addendum 22(b); regression run of clone_law.py = 12,478/12,480 |
| Two failure families: u5 median match 1200/2048; u4-triple-group family 620/2048 (harder despite smaller coset) | RESULT.md addendum 22(b) |
| Rf-parity caveat: the one odd-Rfirst tuple (sub77) deviates both directions; closed form scores 97.9% on it; per-tuple determinism still perfect | RESULT.md addendum 22(b) (stated in the deck caption as a caveat) |
| Held-out validation: 624 sub85 row-predictions committed pre-silicon (o8_sub85_clone_predictions.txt); silicon returned 2,496/2,496 across 4 banks incl. 40/40 never-observed d classes | [data:dimm2_fault_sweep_subs_2026_07_18/o8_sub85_clone_predictions.txt + run_sub85.sh + sub85_bank{0-3}.log] + RESULT.md addendum 22(b) |
| Cross-die: DIMM-0 predictions committed pre-silicon; 2,494/2,496 (99.92%), banks 2/3 perfect 624/624, zero false-dead | [data:dimm0_fault_sweep_2026_07_20/o5_dimm0_clone_predictions.txt + dimm0_bank{0-3}.log] + RESULT.md addendum 24 §1 |
| Fresh subarrays: same law, same 108/640 (~17%) census — subarray-invariant | RESULT.md addendum 27 (o9 pilot: law_dead = 108/640 on both fresh windows) |
| Anti-selection: clone-dead rows have fault-sweep degree ≈ 0 → degree-ascending greedy IS preferentially selected them: 108/294 production pool, 9/16 V2 scratch tail; DIMM-0 pools 32.8–39.1% dead | RESULT.md addendum 22(b) (production pool audit) + addendum 24 §1 (D0) |
| Voting retro-explanation: unvoted-garbage (134.8 s/tok arm) was computing on clone-dead rows; clone-ok pool → unvoted 137.1 s/tok Paris; voted arm byte-identical text at 372.5 | [data:dimm2_fault_sweep_subs_2026_07_18/o8_fullmodel_cloneok_{unvoted,voted}.log] (8 tok in 1096.7 s / 2979.9 s, same text) + RESULT.md addendum 22(a) |
| Pools now built by predicate: pool_layout_dimm2_cloneok_bank{0-3}.txt (181 LOAD + 16 tail, 197 rows all clone-ok); clone screening removed from new-subarray characterization | [code/data: BitNet pool files] + RESULT.md addendum 22(a), addendum 27 (chain uses law instead of probe) |

## Scene 7 — Traffic-induced drift (NEW 2026-07-20)

| Claim | Source |
|---|---|
| LOAD-resident rows drift from load-time content under production traffic; first flagged at full-model scale in the O1 run (70/71 handles), known-issue comment in the client since May | RESULT.md addendum 21 (instrumented observation); [code:python/pim_linear.py] (the May comment) |
| One-shot transition then saturated-flat (e.g. 0.180→0.180 over 7 verifies / 1,000 s); every handle verifies clean at first use | RESULT.md addendum 22(c) (full-model observations) |
| Isolation arms R1–R7: baseline verify-traffic transitions 52/80; overflow, refresh-off, streamed-request, fused-hammer, scratch-request arms all ≈52-53/80 — the effect saturates; the verify pass itself is the treatment | [data:dimm2_fault_sweep_subs_2026_07_18/o8_arm_R{1-7}*.log + o8_drift_arms.py] + RESULT.md addendum 22(c) |
| Refresh exonerated twice: R2 ≡ R3 (refresh on/off identical), and pre-fix runs (extra windows silently unrefreshed due to the label-collision bug) show the same transitions | RESULT.md addendum 22(c) + addendum 22 part 1 (the label bug) |
| Refresh restores CHARGE of current content, not the loaded CONTENT — it preserves drifted data perfectly | [editorial: mechanism statement], grounded in the refresh-exoneration rows + addendum 23 root cause |
| Saturation level ordered by selection-law coupling class: u≤3 rows → median 6,913/8,192 changed (~84%); u5 rows → 2,917 (~36%) | RESULT.md addendum 22(c) (o8_drift_correlation.py over O1's 1,488 verify observations + arm data) |
| "One mechanism, two faces": clone-source quality and deposit-receiver strength are the same lattice coupling | [editorial], grounded in the (b)+(c) correlation (dense signature ↔ clone-OK rows, partial ↔ clone-dead) |
| Flipped bits byte-lane-localized: new-1 bits confined to 0xff000000 in 1,512/1,540 unvoted verify-bits observations (one x8 device lane) | RESULT.md addendum 22(c) |
| Retention is a separate additive term: R4 (300 s idle, auto-refresh off) → 72/80 transitioned, y-bad 80/80 | [data:o8_arm_R4_idle300.log] + RESULT.md addendum 22(c) |
| Compute impact: real at exact-check granularity (60/80 handles y-mismatch post-saturation at d_in=64) while the full model absorbs 18–33% drifted rows (Paris, unvoted included) | RESULT.md addendum 22(c) |
| No allocator dodge (same-subarray physics); exactness-grade mitigation = per-round row rewrite (PIM_LOAD_REWRITE_ON_MM3D=1) | RESULT.md addendum 22(c); [code:app/test_bitnet_server.cpp] |
| The predicted regression: resident const rows (ONE/ZERO) were deposit-safe = strong-coupling class; written once → drifted → consts-only full model babbled while protocol tests passed | RESULT.md addendum 23 (root cause) + task O4 record |
| Fix: rewrite const-row CONTENT at every request start after the entry refresh (~0.52 ms, wrRow_immediate program); decisive arm = full model byte-identical to features-off control ('The capital of France is Paris. Paris'), 33,368 requests, rewrites firing every MM3D | RESULT.md addendum 23 (pre-fix 6021.1 s babble; fixed 6060.9 s vs control 6064.3 s); [code:app/test_bitnet_server.cpp] (rewrite_resident_const_rows, PIM_CONSTS_REWRITE_EVERY) |
| Honest residual: per-request rewrite narrows exposure to within-request; ~0.3% of protocol matmuls still show ±1-count tail | RESULT.md addendum 23 (LONG 300-matmul arms: 298/300 pre-fix vs 299/300 fixed vs 300/300 no-consts control) |

## Scene 11 (was Scene 9) — Two responses: avoid it, or use it (steps 1, 3–4 UPDATED 2026-07-17; step 2 amended 2026-07-20)

Additional row (2026-07-20 amendment):

| Claim | Source |
|---|---|
| Fault-freedom alone anti-selects clone health; current pools apply BOTH filters (IS discipline + clone-law predicate): 197 clone-ok rows on the production die | Scene 6 rows above; [code/data: pool_layout_dimm2_cloneok_bank{0-3}.txt] + RESULT.md addendum 22(a) |

Step 2 ("engineer around": conflict graph → max independent set; 276/624
and 294/624 pool rows, labelled EXAMPLE) is carried over unchanged from
the original deck.

| Claim | Source |
|---|---|
| Corruption during a RowClone into a co-activated group is a function of the pair offset src⊕dst; offsets free of generator subsets are clean BY CONSTRUCTION | [code:app/test_safe_load.cpp] (model in header; phases verify it); [study §8] |
| 20/20 clean safe-offset loads; unsafe-offset controls corrupt exactly the predicted rows | [code:app/test_safe_load.cpp] (phases 1–2) |
| Full 16-row tuple loaded with 8 doubleACTs: 16/16 correct (June's source-model attempt: 5/16) | [code:app/test_safe_load.cpp] (phase 4; header records the June baseline) |
| Faithful MVDRAM computation-rows dataflow: June 6.1% → July 99.98% end-to-end, same DAG, placement-only change | [code:app/test_mvdram_compute_rows_safe.cpp]; [study §8] |
| Fused per-MAJ tool A/B (iters=100, one process): 0.357 → 0.089 ms/MAJ = 3.99× (die A); 0.556 → 0.085 = 6.53× (die B); bit-exact 0/204,800 bad segments per die | [code:app/test_fused_maj.cpp] (run 2026-07-17 on benders 2 and 0; verified `result == W & x` every iteration) |
| Production server `PIM_FUSED_COSET=1`: 9.7–9.8 → 6.4–6.9 ms/matmul (bank 0), 9.5–9.7 → 6.1 (4-bank) = 1.45–1.6×, bit-exact vs host reference | [code:app/test_bitnet_server.cpp] (the env-gated coset path); [code:python/ab_fused_server.py] (A/B driver: same seed → byte-identical requests; exact host-side reference check) |
| Real model: BitNet b1.58-2B, layer 0 × 7 projections in DRAM, 4-bank, 8 tokens: 117.2 → 71.8 s wall = 1.63× (PIM request time 112.5 → 70.2 s); both arms answer "Paris" | [code:python/run_bitnet_pim.py] + [code:app/test_bitnet_server.cpp] (A/B run 2026-07-17, identical prompt/config except the env flag) |
| MVDRAM fastpath kernel (18-MAJ3-gate popcount-4 carry-save): 1.585/1.574 → 0.683/0.705 ms/gate = 2.32×/2.23× (die A/B); fast loading costs ~0.17% end-to-end | [code:app/test_mvdram_fastpath_ab.cpp]; [data:docs/data/mvdram-repro/fastpath_ab_b2.log] (die A), [data:docs/data/mvdram-repro/fastpath_ab_b0.log] (die B) |
| Closing quote ("Using something in the way it was not intended…") | project's own maxim (also the closing scene of the main deck) — not a factual claim |
