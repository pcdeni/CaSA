# CaSA explainers

Two self-contained, interactive explainers of how a low-bit LLM is computed
inside unmodified commodity DRAM, plus the peer-facing companion that carries
the comparison numbers and the methodology.

## Read them in this order

1. **[pcdeni.github.io/CaSA/explainer/](https://pcdeni.github.io/CaSA/explainer/) — *How an LLM runs inside a DRAM chip*.**
   The plain-language walkthrough. No prior knowledge assumed: a DRAM cell as a
   bucket of charge, the row/subarray/bank layout, the majority-vote trick, and
   a step-by-step **instruction walk** (write → clone → vote → read → iterate)
   over real rows, ending at "the correct word comes back out of the memory."

2. **[pcdeni.github.io/CaSA/explainer/xor-spread.html](https://pcdeni.github.io/CaSA/explainer/xor-spread.html) — *One command pair, two physics*.**
   The mechanism. One double-activation command pair does one of two completely
   different things depending on its timing alone — a clean **majority vote** at
   the operating point, or a **multi-row copy** a few command slots away. The
   central object is an interactive **timing dial** (vote / tie / copy); it
   drives the coset + selection law, the good/bad-case row tables, and the
   single binary vote-derivation figure. Chip-specific numbers are labelled as
   measured examples on named silicon.

For throughput numbers, the wall model, the MVDRAM contrast, and the
verification discipline behind every number, both explainers link out to the
peer doc, **[Related systems and methodology](../RELATED_SYSTEMS.md)** — the
single home for those.

## MVDRAM: the closest published peer

MVDRAM has one home here, not two. The **comparison** — MVDRAM mechanism by
mechanism, what the paper claims and what reproduces on the commodity silicon we
own (the named part performs no PUD in our hands) — lives in the peer doc,
[Related systems and methodology](../RELATED_SYSTEMS.md) (§2). The full
hardware-reproduction **study** is [MVDRAM_REPRODUCTION.md](../MVDRAM_REPRODUCTION.md).
The former interactive deck now redirects to those two.

## How these were built

Every factual claim is sourced and adversarially reviewed.
The supporting files in this folder are the evidence trail.

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
