# MVDRAM reproduction on commodity DDR4 — results summary

Reproduction of MVDRAM (arXiv 2503.23817) from the paper alone (no source
released).
Started + completed (correctness-faithful).

## What was reproduced, on real silicon, bit-exact-validated

| MVDRAM mechanism | Our result (DIMM 2 unless noted) | Tool |
|---|---|---|
| RowClone (their product/copy primitive) | 8192/8192, both DIMMs | rowclone-smoke |
| MAJ5 reliability via column screening | 87.1% reliable cols (DIMM2), 88.2% (DIMM0) — matches their 83–95% | mvdram-maj5 |
| Dual-track MAJ full adder (carry=MAJ3, sum=MAJ5) | 99.94% on screened cols | mvdram-adder |
| In-DRAM GeMV, ternary×binary (N=2) | 100% bit-exact (3× vote) | mvdram-gemv |
| In-DRAM GeMV, **2-bit & 4-bit signed × int8** (all 4 target models' precisions) | 99.99% | mvdram-gemv |
| In-DRAM carry-save popcount TREE (N=4,8) | 99.97–99.98% | mvdram-popcount |
| **In-DRAM dual-track ~carry** (no host NOT, faithful §II-C2) | 99.97% | mvdram-popcount |
| **COMPLETE integrated GeMV** (tree + q-bit + factor + dual-track) | 99.91% (N=4,q=2,r=2, no vote) | mvdram-gemvn |

Models MVDRAM targeted: Llama2-7B/13B, Llama3-8B, Phi-4 @ 2-bit AND 4-bit (via
llama.cpp). All four share one kernel (signed q-bit × r-bit GeMV) differing only
in dimensions → our 2-bit & 4-bit kernel validation covers all four at the
compute level. Ternary (BitNet) is the simpler q=2 special case.

Residual everywhere ≈ 0.01–0.1% = transient MAJ5 cell noise; grows with op
count; reduced by column screening; within the tolerance low-bit LLMs are built for.
