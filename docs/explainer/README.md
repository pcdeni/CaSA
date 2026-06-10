# BitNet PIM Explainer

An interactive, step-by-step explainer of how a ternary LLM (BitNet b1.58) is computed inside commercial DRAM using SiMRA-style multi-row activation.

**▶ View the explainer:** [https://pcdeni.github.io/CaSA/explainer/](https://pcdeni.github.io/CaSA/explainer/)

## What it covers

11 scenes, navigate with Prev/Next buttons (or arrow keys), Play auto-advances:

1. DRAM hierarchy (cell → row → subarray → bank → DIMM)
2. One cell: ACT / READ / PRE — the destructive-read cycle
3. ACT / READ / PRE on a whole row
4. The APA primitive: `ACT — wait t₁ — PRE — wait t₂ — ACT`
5. Three operations from one envelope: RowCopy / Multi-RowCopy / Multi-Row Activation
6. MAJ-of-K via K-row charge share, plus replication for stability
7. Composing one ternary × int8 multiply from MAJ-based ANDs
8. Inference loop: prompt → next token
9. Bottleneck analysis: where does the wall time go?
10. Wishlist: what AI-specific DRAM would change
11. Closing thought: using something in the way it was not intended

## Companion: the doubleACT row-spread

A second, standalone explainer characterizes a side-effect we found while calibrating the hardware — the SiMRA `doubleACT` primitive deposits a bit-exact copy of its source row into address-XOR sibling rows, and that same effect silently pollutes the operands of a MAJ3 during multi-row-init characterization.

**▶ View it:** [https://pcdeni.github.io/CaSA/explainer/xor-spread.html](https://pcdeni.github.io/CaSA/explainer/xor-spread.html)

8 scenes: the observation (bit-exact spread to `R ⊕ (1<<b)`), its five properties, the row-decoder predecoder-coupling mechanism, cross-DIMM universality (each die has a different "gap bit"), the MAJ3 self-pollution it causes, the signature it leaves in MAJ success-rate data (a collapse to 0.7% at the K/2 tie boundary), why this confounds PuD characterization, and two responses (engineer around it with an independent-set pool, or exploit it as a free fan-out primitive). Chip-specific numbers are labelled as examples; the full-sweep figures (1066 single-bit edges, the overlap cross-tab) are this project's own measurements.

## How it was built

This is meant to read at research-conference quality. Every factual claim was sourced and adversarially reviewed before publication. The supporting documents in this folder are the evidence trail:

- **[paper_mechanism_notes.md](paper_mechanism_notes.md)** — extraction of the bitline/sense-amp mechanism from the source papers (SiMRA, FracDRAM, FCDRAM). Each claim cites paper section/figure.
- **[pim_explainer_ledger.md](pim_explainer_ledger.md)** — claim-to-source ledger. Every factual statement in the HTML traces to one of: a paper, the production server code, or labelled-as-example chip-specific data.
- **[pim_explainer_review.md](pim_explainer_review.md)** — adversarial review record. Lists every claim the reviewer found, severity-tagged (BLOCKER / MAJOR / MINOR / OK), with the fix applied to each one.

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
