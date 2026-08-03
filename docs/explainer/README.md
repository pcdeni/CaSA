# CaSA explainers

Two self-contained, interactive explainers of how a low-bit LLM is computed
inside unmodified commodity DRAM, plus the peer-facing companion that carries
the comparison numbers and the methodology.

Both explainers are single HTML files with no external requests — they render
identically from a local `file://` and from GitHub Pages, in light or dark.

## Read them in this order

1. **[index.html](index.html) — *How an LLM runs inside a DRAM chip*.**
   The plain-language walkthrough. No prior knowledge assumed: a DRAM cell as a
   bucket of charge, the row/subarray/bank layout, the majority-vote trick, and
   a step-by-step **instruction walk** (write → clone → vote → read → iterate)
   over real rows, ending at "the correct word comes back out of the memory."

   **▶ [pcdeni.github.io/CaSA/explainer/](https://pcdeni.github.io/CaSA/explainer/)**

2. **[xor-spread.html](xor-spread.html) — *One command pair, two physics*.**
   The mechanism. One double-activation command pair does one of two completely
   different things depending on its timing alone — a clean **majority vote** at
   the operating point, or a **multi-row copy** a few command slots away. The
   central object is an interactive **timing dial** (vote / tie / copy); it
   drives the coset + selection law, the good/bad-case row tables, and the
   single binary vote-derivation figure. Chip-specific numbers are labelled as
   measured examples on named silicon.

   **▶ [pcdeni.github.io/CaSA/explainer/xor-spread.html](https://pcdeni.github.io/CaSA/explainer/xor-spread.html)**

For throughput numbers, the wall model, the MVDRAM contrast, and the
verification discipline behind every number, both explainers link out to the
peer doc, **[Related systems and methodology](../RELATED_SYSTEMS.md)** — the
single home for those. Live lever status is in **[the roadmap](../ROADMAP.md)**.

## Companion: the MVDRAM reproduction study

A separate deck presents the [MVDRAM reproduction study](../MVDRAM_REPRODUCTION.md)
interactively — what MVDRAM claims, what reproduces on the commodity silicon we
own, and what does not (the named part performs no PUD in our hands; the chained
MAJ5 adder does not reach bit-exactness on any module we own). It owns the
*reproduction*; the peer doc owns the *comparison*.

**▶ [pcdeni.github.io/CaSA/explainer/mvdram.html](https://pcdeni.github.io/CaSA/explainer/mvdram.html)**

Claim-to-source ledger: [`mvdram_explainer_ledger.md`](mvdram_explainer_ledger.md).

## How these were built

Research-conference quality is the bar: every factual claim is sourced and
adversarially reviewed before publication. The supporting files in this folder
are the evidence trail.

- **[xor-spread_ledger_2026_08_03.md](xor-spread_ledger_2026_08_03.md)** — the
  publish-gate claim ledger for the mechanism explainer: every factual statement
  maps to a paper, production code, a measured log, or a claim-register entry.
  (Supersedes the retired deck's `xor_spread_ledger.md`, kept as prior
  provenance.)
- **[index_ledger_2026_08_03.md](index_ledger_2026_08_03.md)** — the publish-gate
  claim ledger for the system explainer. (Supersedes `pim_explainer_ledger.md`
  for the rebuilt content; that file is kept as prior provenance.)
- **[paper_mechanism_notes.md](paper_mechanism_notes.md)** — extraction of the
  bitline / sense-amp mechanism from the source papers (SiMRA, FracDRAM,
  FCDRAM); each claim cites paper section / figure.
- **[pim_explainer_review.md](pim_explainer_review.md)** — adversarial-review
  record, severity-tagged.
- **[publish_ledger_2026_07_20.md](publish_ledger_2026_07_20.md)** — a dated
  publish-phase verification pass across the decks.

## Sources

- **SiMRA** — Yüksel et al., *Simultaneous Many-Row Activation in Off-the-Shelf
  DRAM Chips*, arXiv:2405.06081 (2024). https://arxiv.org/abs/2405.06081
- **FracDRAM** — Gao, Tziantzioulis, Wentzlaff, *FracDRAM: Fractional Values in
  Off-the-Shelf DRAM*, MICRO 2022.
- **FCDRAM** — Yüksel et al., *Functionally-Complete Boolean Logic in Real DRAM
  Chips*, arXiv:2402.18736 (2024).
- **POPCNT3** — Kubo et al., *Bulk Bitwise Accumulation in Commercial DRAM*,
  NeurIPS MLNCP 2024.
- **BitNet b1.58** — `microsoft/bitnet-b1.58-2B-4T` on HuggingFace.

Related systems (context, treated in full in
[`../RELATED_SYSTEMS.md`](../RELATED_SYSTEMS.md)):

- **MVDRAM** — Kubo et al., *MVDRAM: Enabling GeMV Execution in Unmodified DRAM
  for Low-Bit LLM Acceleration*, arXiv:2503.23817 (2025). The closest published
  peer: general matrix-vector multiply in unmodified DDR4 on the same testbed
  family.
- **SiTe CiM** — Thakuria et al., arXiv:2408.13617 (2024) — custom-cell
  signed-ternary CiM (simulation); the modified-silicon end of the design space.
- **PARBOR** — Khan, Lee, Mutlu, DSN 2016 — data-dependent bitline-neighbor
  coupling; the bitline-side sibling of our row-decoder-side coset finding.

## Repo this lives in

Part of the [CaSA project](https://github.com/pcdeni/CaSA) — DRAM-as-compute
experimental work building toward in-memory inference of low-bit LLMs on
commodity DDR4.

## Local viewing

```bash
git clone https://github.com/pcdeni/CaSA.git
cd CaSA/docs/explainer
python3 -m http.server 8000
# open http://localhost:8000/
```

Or open `index.html` directly in a browser — no build step required.
