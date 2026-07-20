# Programmable co-activation lattice: fast loading, sub-lattice broadcast, and an MVDRAM fast-path reproduction

**Status: DRAFT — internal, not yet published. Results reproduced on silicon
2026-07-16/17 (BCU1525, two full-PUD SK Hynix M-die modules, benders 0 & 2).**

This note turns the SiMRA co-activation lattice (see
[XOR-spread explainer](https://pcdeni.github.io/CaSA/explainer/xor-spread.html)
and CMU-SAFARI/SiMRA-DRAM#1) from a hazard we route around into an **addressing
primitive we program**. Three results follow, each verified bit-exact.

## 1. Sub-lattice broadcast
A `doubleACT` between two members of a calibrated open-rows tuple, at
k-generator distance, deposits the sensed-first row's data into **exactly** the
2^k sub-coset `{X ⊕ S : S ⊆ bits(local(X)⊕local(Y))}` — a targeted broadcast.
Verified for all 15 partners of a 16-row tuple × 3 repeats × 2 dies: predicted
row-set == measured, zero external leak. Design rule: t₁₂ ≥ 10 (t₁₂=0 deposits
nothing); t₂₃ free. Tool: `test_sublattice_bcast.cpp`.

## 2. Corruption-free fast loading (by construction)
RowClone-into-tuple corruption is exactly the pair coset of the (src,dst)
address offset. Choosing offsets whose bits contain no tuple generator makes
loads tuple-clean **by construction** — no reliance on empirical source
screening. 20/20 safe loads clean; unsafe offsets corrupt exactly the predicted
rows. A full 16-row tuple loads correctly with 8 `doubleACT`s. Tool:
`test_safe_load.cpp`. (This corrects an earlier "blocked deeper than source
choice" reading based on a source-specific vulnerability model.)

## 3. MVDRAM faithful dataflow — reproduced on commodity silicon
MVDRAM (arXiv:2503.23817) released no source. Its performance premise is a fast
in-DRAM RowClone→MAJ→RowClone dataflow with weights resident. Reproducing it
faithfully on our commodity, spread-afflicted DIMMs previously yielded **6.1%
end-to-end** (the RowClone-into-compute-tuple was self-corrupting). With the
lattice-aware safe placement above — identical MAJ and DAG, only addressing
changed — the 18-gate popcount dataflow runs at **99.98% end-to-end, 99.99%
per-op, on two dies**. Tool: `test_mvdram_compute_rows_safe.cpp`.

This reproduces MVDRAM's performance *shape* without a golden "strict RowCopy"
module. It does not rehabilitate the paper's specific part claim (our units of
the paper's exact SK Hynix part exhibited no PUD at all); it is a broader
result — the fast path on ordinary silicon, steered by address algebra.

## Consequences for in-DRAM LLM (BitNet-class ternary)
Measured on silicon, software-only, no bitstream change:
- **Activation update** (≈80% of per-MAJ cost): 5 per-column writes → 1 write +
  1 coset `doubleACT` = **4.98× fewer SoftMC instructions, 2.81× wall-clock**.
- **Persistent weights**: keep W in a backup row at a safe offset, refresh the
  active tuple with one clone per token instead of a per-column rewrite =
  **2.6× per-MAJ, 100% bit-exact** (`test_matvec_persist_ab.cpp`).
- These compose (different lines of the per-MAJ program) → combined ~4–5×
  target, pending a fused per-MAJ program and a full-layer throughput run.

## Module-geometry finding
MAJ3 reliability needs **operand separation**, not just clean co-activation: a
tuple whose generators are all small (adjacent rows, e.g. {1,2}) self-pollutes
on mixed votes; a separated generator (e.g. {1,384}) protects the vote. Two of
our modules ("partial-PUD") have clean co-activation ONLY at the adjacent scale
— every wider pair is a dirty lattice — so they are MAJ3-limited by decoder
structure, usable for storage/non-MAJ roles. The two M-die modules decompose
704/704 wide tuples and carry the compute.

---
*Reproducibility: all tools under `DRAM-Bender/sources/apps/DSN_AE_APPS/BitNet/`;
raw logs + enumeration scripts in the project's `sublattice_broadcast_2026_07_17/`.*
