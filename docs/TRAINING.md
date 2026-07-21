# Can commodity-DRAM PIM train, or only infer?

Short answer: **the boundary runs through the optimizer state, not
through "training" as a category.** Full end-to-end backbone training in
DDR4 is blocked by physics we have measured ourselves; several
training-adjacent workloads are not blocked at all, and one of them is
buildable from this repo today.

## Why the core of training resists this substrate

Training's essential object is not the big matmul — it is a
**high-precision accumulator**: a latent weight integrating millions of
~1e-4 updates. Ternary-forward models (BitNet, Bonsai) are themselves
trained that way — full-precision master weights underneath, the ternary
value just a snapshot (straight-through estimation). A DRAM cell is a
one-bit charge bucket with destructive reads, and our own
characterization sharpened the objection: under PUD traffic, resident
content **drifts** (refresh restores charge, not content — see the
content-drift results). DRAM is a poor long-lived integrator even at
1 bit, never mind 20. This is why in-memory-*training* literature lives
on analog non-volatile cells (ReRAM/PCM), which physically integrate
small updates. The missing piece is a physical accumulator, not a
smarter multiply.

## What this project measured that moves the boundary

1. **Precision is not the wall for the compute.** The reproduction's
   GEMV_PARTIALS path produces *exact* integer per-block partials and
   reconstructs bit-exact fp32 with host-side scales
   ([`MVDRAM_REPRODUCTION.md`](MVDRAM_REPRODUCTION.md)). Bit-serial
   arithmetic extends to any integer width at linear cost; what dies is
   floating-point exponent handling. Block-scaled integer formats — the
   g128 machinery built for Bonsai — are exactly the format family
   quantized-*training* schemes (int8/block-float gradients) use.
   A quantized-gradient experiment is format-compatible with this
   server; backward needs the weights resident in the transposed
   orientation too (2× storage — the same move the dual-track already
   makes for complements).

2. **The frozen-backbone loophole is real.** LoRA/adapter fine-tuning
   never updates the backbone: it needs the backbone forward and the
   backward input-gradient matmul (dY·Wᵀ), both with W frozen —
   inference-grade operations this pipeline has validated — while the
   trainable part is tiny and stays fp16 on the host. The FLOP-dominant
   ~99 % of a fine-tuning step is frozen-backbone matmuls.
   **"Train a LoRA on a model whose weights live and compute inside a
   DDR4 DIMM"** is buildable from `python/pim_linear.py` (wrap the
   module in an autograd Function; keep Wᵀ resident) and is queued as a
   post-publication demonstration.

3. **A creative idea we rate against ourselves: physical stochastic
   rounding.** Low-precision training theory permits dropping the
   latent weights if updates apply *stochastically* (flip a ternary
   weight with probability ∝ gradient). This silicon contains a biased
   coin — the MAJ tie boundary, tunable by Frac conditioning (the
   reference-policy results). But this project's recurring lesson is
   that the noise here is *dirty*: content-conditional, spatially
   structured, partly deterministic (the fused-marginal floor). An
   optimizer built on it inherits a calibration burden host-side
   stochastic rounding gets for free. Interesting; not our lane.

4. **Training-relevant without touching precision: gradient
   collectives.** Distributed training spends enormous bandwidth
   summing gradients — and summation is what this substrate is provably
   good at (Road A in-DRAM adders; the Road B readout accumulator).
   Block-quantized gradient reduction in memory sidesteps the
   accumulator problem entirely.

## The honest one-liner

Inference and frozen-weight compute are commodity-DRAM PIM's native
lane; trainable state must live in a substrate that can integrate.
Don't write "training is impossible in memory" — a LoRA-over-DRAM run
would place the boundary exactly where it belongs: at the optimizer,
not at the matmul.
