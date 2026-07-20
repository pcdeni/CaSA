# BitNet PIM Explainer

An interactive, step-by-step explainer of how a ternary LLM (BitNet b1.58) is computed inside commercial DRAM using SiMRA-style multi-row activation.

**▶ View the explainer:** [https://pcdeni.github.io/CaSA/explainer/](https://pcdeni.github.io/CaSA/explainer/)

## What it covers

13 scenes, navigate with Prev/Next buttons (or arrow keys), Play auto-advances:

1. DRAM hierarchy (cell → row → subarray → bank → DIMM)
2. One cell: ACT / READ / PRE — the destructive-read cycle
3. ACT / READ / PRE on a whole row
4. The APA primitive: `ACT — wait t₁ — PRE — wait t₂ — ACT`
5. Three operations from one envelope: RowCopy / Multi-RowCopy / Multi-Row Activation
6. MAJ-of-K via K-row charge share, plus replication for stability
7. Composing one ternary × int8 multiply from MAJ-based ANDs
8. Inference loop: prompt → next token
9. Bottleneck analysis: where does the wall time go?
10. The merge: in-DRAM accumulation — Road A (faithful dual-track adder) and Road B (popcount HDL) attack the readout wall
11. The arc: 632 → 47.5 s/tok, hop by hop, with what fixed each hop
12. Wishlist: what AI-specific DRAM would change
13. Closing thought: using something in the way it was not intended

## Companion: the doubleACT row-spread

A second, standalone explainer characterizes a side-effect we found while calibrating the hardware — the SiMRA `doubleACT` primitive deposits a bit-exact copy of its source row into address-XOR sibling rows, and that same effect silently pollutes the operands of a MAJ3 during multi-row-init characterization.

**▶ View it:** [https://pcdeni.github.io/CaSA/explainer/xor-spread.html](https://pcdeni.github.io/CaSA/explainer/xor-spread.html)

11 scenes: the observation (bit-exact spread to `R ⊕ (1<<b)`), its five properties, the mechanism (settled July 2026: Multi-RowCopy's co-activation lattice seen from the source side — confirmed by a SiMRA co-author in [DRAM-Bender#12](https://github.com/CMU-SAFARI/DRAM-Bender/issues/12), verified on our data in [SiMRA-DRAM#1](https://github.com/CMU-SAFARI/SiMRA-DRAM/issues/1)), cross-DIMM universality (each die has a different "gap bit"), the selection law (which lattice members fire — 1691/1691 member-observations explained on each of two dies, zero exceptions; `app/test_selection_timing.cpp` + `docs/data/selection-law/`), the clone-dead law (a closed-form predicate for which rows silently fail to clone — held out 2,496/2,496 on one die, 2,494/2,496 cross-die; `app/clone_law.py` + `docs/data/clone-law/`), the drift arms that predicted the resident-const regression (`docs/data/drift/`), the 1024-block predecoder scope the law lives in, the MAJ3 self-pollution it causes, the signature it leaves in MAJ success-rate data (a collapse to 0.7% at the K/2 tie boundary), why this confounds PuD characterization, and two responses (engineer around it with an independent-set pool, or use the lattice as an addressing system — safe-by-construction loads, the fused coset production path, and the MVDRAM fastpath kernel). Chip-specific numbers are labelled as examples; the full-sweep figures (1066 single-bit edges, the overlap cross-tab) are this project's own measurements.

## Companion: the MVDRAM reproduction

A third deck presents the [MVDRAM reproduction study](../MVDRAM_REPRODUCTION.md) interactively.

**▶ View it:** [https://pcdeni.github.io/CaSA/explainer/mvdram.html](https://pcdeni.github.io/CaSA/explainer/mvdram.html)

11 scenes: what MVDRAM claims (headline numbers and what its baselines actually are), its two techniques, what "reproduce" means for a paper with no released source, Result A (the paper's named DRAM part performs no PUD in our hands — 0 charge-share events in 60,000 random pairs on two new units), the Result B journey (June's 6.1% collapse honestly presented as our own corrected error → July's 99.98% with spread-aware placement → the 2.2–2.3×/gate fused fast path), the Lane-2 in-DRAM GeMV protocol run and its O7 closures (R1–R4 host/dual-track/clone/fp32 arms on `docs/data/lane2/`), an updated mechanism-by-mechanism scoreboard and gap decomposition, what we have NOT done (Frac/calibration, their llama.cpp benchmark, streaming-scale execution), what our silicon adds beyond the paper, and the updated verdict with the paper's own caveats stated plainly. Claim-to-source ledger: [`mvdram_explainer_ledger.md`](mvdram_explainer_ledger.md); the July-update scenes of the row-spread deck are covered by [`xor_spread_ledger.md`](xor_spread_ledger.md).

## Companion: the full-system tour

A fourth deck stitches the other three into one narrative — a plain-language
guided tour from "a prompt arrives" to "the answer comes back out of DRAM,"
pointing into the technical decks at each step.

**▶ View it:** [https://pcdeni.github.io/CaSA/explainer/system.html](https://pcdeni.github.io/CaSA/explainer/system.html)

## Reading order (four decks)

Start with the tour, then drop into whichever technical deck a step made you
curious about:

1. **[system.html](system.html)** — the guided tour (start here).
2. **[index.html](index.html)** — how a ternary LLM is computed inside DRAM (13 scenes).
3. **[xor-spread.html](xor-spread.html)** — the row-spread characterization, the selection law, and the clone-dead law (11 scenes).
4. **[mvdram.html](mvdram.html)** — the MVDRAM reproduction: what held, what reversed (11 scenes).

## How it was built

This is meant to read at research-conference quality. Every factual claim was sourced and adversarially reviewed before publication. The supporting documents in this folder are the evidence trail:

- **[paper_mechanism_notes.md](paper_mechanism_notes.md)** — extraction of the bitline/sense-amp mechanism from the source papers (SiMRA, FracDRAM, FCDRAM). Each claim cites paper section/figure.
- **[pim_explainer_ledger.md](pim_explainer_ledger.md)** — claim-to-source ledger. Every factual statement in the HTML traces to one of: a paper, the production server code, or labelled-as-example chip-specific data.
- **[pim_explainer_review.md](pim_explainer_review.md)** — adversarial review record. Lists every claim the reviewer found, severity-tagged (BLOCKER / MAJOR / MINOR / OK), with the fix applied to each one.
- **[publish_ledger_2026_07_20.md](publish_ledger_2026_07_20.md)** — the July-20 publish-phase verification pass across all four decks: 74 load-bearing claims traced to their primary artifact (re-read where retained), each tagged V / V(a) / S / F. Publishing it *is* the honesty convention — the four flagged items (F1–F4) are listed openly, none silently altered.

## Sources

- **SiMRA** — Yüksel et al., *Simultaneous Many-Row Activation in Off-the-Shelf DRAM Chips*, arXiv:2405.06081 (2024). https://arxiv.org/abs/2405.06081
- **FracDRAM** — Gao, Tziantzioulis, Wentzlaff, *FracDRAM: Fractional Values in Off-the-Shelf DRAM*, MICRO 2022.
- **FCDRAM** — Yüksel et al., *Functionally-Complete Boolean Logic in Real DRAM Chips*, arXiv:2402.18736 (2024).
- **POPCNT3** — Kubo et al., *Bulk Bitwise Accumulation in Commercial DRAM*, NeurIPS MLNCP 2024.
- **BitNet b1.58** — `microsoft/bitnet-b1.58-2B-4T` on HuggingFace.

Related systems (context, not mechanism sources for the scenes):

- **MVDRAM** — Kubo et al., *MVDRAM: Enabling GeMV Execution in Unmodified DRAM for Low-Bit LLM Acceleration*, arXiv:2503.23817 (2025). https://arxiv.org/abs/2503.23817 — the closest peer system: throughput-oriented GeMV in unmodified DDR4 on the same testbed family. Mechanics comparison with this project: [`../MVDRAM_COMPARISON.md`](../MVDRAM_COMPARISON.md).
- **SiTe CiM** — Thakuria et al., *SiTe CiM: Signed Ternary Computing-in-Memory for Ultra-Low Precision DNNs*, arXiv:2408.13617 (2024) — custom-cell signed-ternary CiM (simulation); the modified-silicon end of the design space.
- **PARBOR** — Khan, Lee, Mutlu, *PARBOR: An Efficient System-Level Technique to Detect Data-Dependent Failures in DRAM*, DSN 2016 — chip-specific bitline-neighbor coupling characterization at JEDEC timing; the bitline-side sibling of our [XOR-spread findings](xor-spread.html) on the row-decoder side.

## Repo this lives in

This explainer is part of the [CaSA project](https://github.com/pcdeni/CaSA) — DRAM-as-compute experimental work building toward in-memory inference of ternary LLMs on commodity DDR4 hardware.

## Local viewing

```bash
git clone https://github.com/pcdeni/CaSA.git
cd CaSA/docs/explainer
python3 -m http.server 8000
# open http://localhost:8000/
```

Or just open `index.html` directly in a browser — no build step required.
