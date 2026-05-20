# BitNet PIM Explainer

An interactive, step-by-step explainer of how a ternary LLM (BitNet b1.58) is computed inside commercial DRAM using SiMRA-style multi-row activation.

**▶ View the explainer:** [https://pcdeni.github.io/CaSA/explainer/](https://pcdeni.github.io/CaSA/explainer/)

## What it covers

10 scenes, navigate with Prev/Next buttons (or arrow keys), Play auto-advances:

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
