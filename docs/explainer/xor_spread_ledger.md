# XOR-Spread Explainer — Claim Ledger (July 2026 update)

**Scope:** this ledger covers only the scenes **added or changed in the
July 2026 update** of `xor-spread.html`:

- Scene 3 (reframed: settled provenance),
- Scene 5 (NEW: the selection law),
- Scene 9 (steps 1, 3–4 changed: shipped "use it" results),
- the header banner (unchanged in this update; its claims are re-listed
  under Scene 3 because the scene now carries them too).

Scenes 1–2, 4, 6–8 are unchanged from the original deck (their
chip-specific figures are labelled as examples or as this project's own
sweep measurements in the deck itself; adversarial-review record in
[`pim_explainer_review.md`](pim_explainer_review.md) covers the original
publication pass).

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

## Scene 9 — Two responses: avoid it, or use it (steps 1, 3–4 UPDATED 2026-07-17)

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
