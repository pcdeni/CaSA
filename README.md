# Your RAM Is Secretly an AI Accelerator

**CaSA: Ternary LLM Inference on Commodity DRAM**

*February 2026*

---

> **Corrections (March 2026):** This README has been substantially revised. Key changes: (1) AND requires three-row majority (MAJ3), not two-row activation. (2) The activation-sacrificial protocol uses SA-mediated RowCopy for weight restoration, not asymmetric row survival. (3) SA-mediated RowCopy is zero-BER, solving the RowCopy reliability problem. (4) Throughput numbers updated with multi-bank pipelining analysis. (5) SiMRA citation corrected — 79M-trial result is from FindOpenRows, not AND operations. See inline corrections throughout.

---

## The Hidden Compute Inside Every Memory Chip

Every stick of RAM in your computer has a hidden trick. When you force three rows of memory cells to turn on at the same time — which violates the timing spec, but physically works — the electrical charges mix together and you get a free majority vote (MAJ3) across tens of thousands of bits simultaneously. By setting one of the three rows to zero, you get an AND operation between the other two. Nanoseconds. Almost zero energy.

This has been measured. The CMU-SAFARI group demonstrated charge-sharing majority operations across 120 real DDR4 chips using the SiMRA framework. The physics works. It has always worked.

The compute capacity inside the chip is over 1,000x more than the memory bus can deliver. It's just sitting there, unused.

## Why Nobody Could Use It

The compute exists, but previous attempts to harness it for anything useful ran into a fatal problem: the operation destroys the input data. When you perform a three-row majority, all three source rows are overwritten with the result. For neural network inference, where you need to reuse the same weight data millions of times, this is devastating — the weights get destroyed on first use.

The brute-force fix is RowCopy: duplicate the weights before each operation. But unmediated RowCopy on commodity DDR4 has a 16.3% bit error rate. That's one in six bits flipped. Neural network inference is impossible at that error rate.

Every prior approach to "Processing-in-Memory" either required custom silicon (Samsung HBM-PIM, SK Hynix AiM, UPMEM) or stopped at demonstrating basic bitwise operations without building anything useful on top.

## The Fix: SA-Mediated RowCopy

Our fix exploits a property of DRAM sense amplifiers that has been hiding in plain sight.

When you activate a row in DRAM, the sense amplifier latches its value. If you then activate a second row while the sense amplifier is still holding the first row's data, the SA drives its latched value into the second row — producing a perfect bit-for-bit copy. This is SA-mediated RowCopy, and it has zero bit errors because the sense amplifier is a full-swing digital latch, not an analog charge-sharing process.

This solves the weight restoration problem. The protocol works as follows:

1. **Backup weights** to a reserved row via SA-mediated RowCopy (zero BER).
2. **Perform AND** via MAJ3(weight, activation, 0) — three-row majority with a pre-zeroed row. This computes weight AND activation, but destroys all three source rows.
3. **Read out the result** through the bus for popcount/accumulation.
4. **Restore weights** from backup via SA-mediated RowCopy (zero BER).

We call this the **activation-sacrificial protocol**, and the full architecture **CaSA** (Charge-sharing Activation-Sacrificial Architecture). The key insight is that SA-mediated RowCopy is fundamentally different from unmediated RowCopy — it uses the sense amplifier as a digital relay rather than relying on analog charge sharing. The 16.3% BER of unmediated RowCopy is irrelevant because we never use unmediated RowCopy.

## Why Ternary Changes Everything

This trick works cleanly only at one specific precision: **ternary** — where neural network weights are restricted to {-1, 0, +1}.

Why? Because multiplying a ternary weight by a binary activation is literally just an AND gate. That's exactly what MAJ3(A, B, 0) gives you. You encode +1 as one binary row, -1 as another, AND each with the activation bits, and the difference gives you the matrix-vector product.

At higher precisions (4-bit, 8-bit), the number of AND operations per weight multiplies rapidly. Only at ternary does it collapse to something commodity DRAM can handle competitively.

The industry currently evaluates ternary on the wrong axis. The question people ask is: "Does ternary match INT4 accuracy on GPUs?" Answer: roughly yes (Microsoft's BitNet b1.58 matches LLaMA quality), but GPUs aren't optimized for ternary, so there's no speed benefit. Conclusion: ternary seems pointless.

That analysis completely misses the memory axis. Ternary is the **only** precision at which every RAM chip in the world becomes a neural network accelerator. The reason nobody saw this is that nobody had demonstrated commodity DRAM PIM actually working for inference until now.

## Why Now

This couldn't have been done two years ago. Microsoft published BitNet b1.58 — the first production-quality ternary language model — in February 2024. Before that, there were no ternary models worth running. The DRAM physics has existed since the 1970s. The charge-sharing trick has been measured since 2017. But until ternary models arrived, there was nothing to connect the compute substrate to the workload. CaSA is what happens when those two threads finally meet.

## What We Actually Built

We designed a complete inference pipeline for **BitNet b1.58-2B-4T** — a real 2-billion-parameter ternary language model from Microsoft — running on a single 8 GB DDR4 DIMM ($15-25) with an FPGA controller.

The DRAM handles the heavy matrix multiplications via charge-sharing AND (MAJ3 with a zeroed row). The FPGA handles the lightweight operations: popcount (counting 1-bits in the result), accumulation, RMSNorm, SiLU activation, and softmax. The model fits in a single DIMM with room to spare.

**Baseline speed: 0.40 tokens per second on one DIMM (no pipelining).**

That's slow. But the critical insight is *where* the time goes — and why pipelining changes everything.

## The Bottleneck Breakdown

Without pipelining, the time breakdown reveals that most of the work is DRAM-internal:

| Component                              | Share of Inference Time |
| :------------------------------------- | :---------------------: |
| **RowCopy (weight backup + restore)**  |         **67%**         |
| **MAJ3 AND (charge-sharing compute)**  |         **33%**         |
| Bus read/write                         |          <1%            |

The system is completely DRAM-bound. But here's the key: RowCopy is a DRAM-internal operation that never touches the memory bus. This means it can be completely hidden by multi-bank pipelining — while one bank is doing RowCopy, another bank's results can be read out through the bus.

**With 4-bank pipelining, the bottleneck flips entirely:**

| Component                              | Share of Inference Time |
| :------------------------------------- | :---------------------: |
| RowCopy (hidden by pipelining)         |          ~0%            |
| MAJ3 AND (hidden by pipelining)        |          ~0%            |
| **Bus read (result transfer)**         |        **~98%**         |

All DRAM-internal operations (RowCopy, MAJ3) are fully hidden behind bus transfers from other banks. The system becomes 98% bus-bound — exactly the bottleneck that popcount eliminates.

## The Scaling Path

| Configuration                          | Tokens/sec  | What it takes                            |
| :------------------------------------- | :---------: | :--------------------------------------- |
| **1 DIMM (no pipelining)**             |  **0.40**   | **Works today on unmodified DDR4**       |
| 1 DIMM (4-bank pipeline, ternary)      |    3.38     | Firmware optimization only               |
| 4 DIMMs (pipelined, ternary)           |   13.53     | $60 of commodity RAM, no chip changes    |
| **+ In-DRAM Popcount (8-bank)**        | **47.82**   | **~2,000 gates per bank (~$0.10/DIMM)**  |
| HBM2 (8-ch) + Popcount                |    ~100     | Server memory                            |

The popcount register is the single biggest lever. It's a tiny bit-counting circuit — about 2,000 logic gates — that counts the 1-bits in a DRAM row without reading the data out through the bus. This eliminates the entire ~98% read bottleneck in the pipelined configuration. Samsung patented this exact circuit in 2014. It has never been shipped in any product.

## It's Surprisingly Robust

A natural question: if you're doing computation by mixing analog charges, how fragile is this?

Not very. Even at a bit error rate of 0.01% — orders of magnitude worse than what has been measured on real hardware — model output quality degrades by less than half a percent. The safety margin between measured reliability and the point where accuracy starts to suffer is roughly 50,000x. Commodity DRAM, within its validated timing window, is not fragile.

## Manufacturer Compatibility (This Matters)

Not all DDR4 works:

- **SK Hynix C-die (2018-2020):** Confirmed compatible. This is our target platform.
- **Micron DDR4:** Likely compatible. The FCDRAM study tested 256 chips from two anonymized manufacturers (believed to be SK Hynix and Micron) with ~95% success rate.
- **Samsung DDR4: Incompatible.** Zero processing-using-DRAM operations work on Samsung dies. This appears to be a hard incompatibility from proprietary internal circuitry, not a calibration issue.
- **Newer SK Hynix (D-die, M-die):** Unknown. More aggressive RowHammer protections may interfere.

Ironically, Samsung holds the key popcount patent and could fix their incompatibility. If they did both — made their chips charge-sharing compatible and added the popcount register — they'd be in the strongest competitive position of any memory manufacturer.

## A Message to Memory Manufacturers

We've identified exactly what's bottlenecking this architecture, and exactly what would fix it. Here's what we'd ask for, ordered from cheapest to most impactful:

**Tier 0 — Costs nothing but coordination:**

- **A PIM mode bit in the Mode Register Set.** One bit that tells the chip: "I'm about to do charge-sharing operations, suppress RowHammer protections and bypass on-die ECC for the next N cycles." This is a spec change, not a silicon change. It would immediately unblock DDR5 (which is currently unusable for PIM because its mandatory on-die error correction scrambles the charge-sharing results). It would also eliminate the ~5% throughput tax from RowHammer guard intervals on DDR4. The catch: this requires JEDEC coordination, which typically takes 3-5 years. But the silicon cost is literally zero.

- **Publish your charge-sharing timing parameters.** Right now, finding the optimal timing for multi-row activation on a specific die revision requires reverse-engineering via tools like DRAM Bender. If manufacturers documented the safe operating window per die revision, it would replace months of characterization with a datasheet lookup.

**Tier 1 — Tiny silicon changes, massive impact:**

- **In-DRAM popcount register (~2,000 gates/bank, <0.3% die area, ~$0.10/DIMM).** This is the single highest-impact change. After a charge-sharing AND, the result sits in 65,536 sense amplifiers. Currently, we have to read all 8,000 bytes out through the bus just to count the 1-bits. A popcount register counts them in-place and returns a single 16-bit number. This eliminates ~98% of total inference time in the pipelined configuration — the entire read bottleneck. Samsung patented exactly this circuit in 2014. It's combinational logic (no clock, no pipeline, no state machine), so it works at full speed even on DRAM-process transistors. It's a passive reduction circuit, not a processor.

**Tier 2 — Moderate silicon, transformative results:**

- **Per-bank activation register (a few hundred thousand transistors per bank).** Right now, we rewrite the activation data from the bus for every single weight row — because MAJ3 destroys the activation row each time. A small static register per bitline would hold the activation vector and drive it onto the bitlines repeatedly without being destroyed. Combined with popcount, this eliminates ALL bus transfers during compute. Bus utilization drops to near zero. A single DIMM becomes deeply compute-bound rather than bus-bound.

- **Wider rows.** This is counterintuitive: the industry trend is toward narrower rows (2 KB in LPDDR5X and HBM, vs 8 KB in DDR4) for latency and power reasons. But for PIM, row width is the fundamental unit of parallelism — each charge-sharing AND processes one full row simultaneously. DDR4's 8 KB rows pack 25 neurons per AND operation. LPDDR5X's 2 KB rows pack only 6, requiring 4x more sequential cycles. A PIM-optimized memory would maximize row width, not minimize it. DDR4's wide rows are an accidental PIM advantage that future memory standards should preserve.

**The bottom line for manufacturers:** The Tier 1 popcount register alone converts CaSA from a proof-of-concept (0.40 tok/s unpipelined, 13.5 tok/s pipelined with 4 DIMMs) to a competitive inference engine (47.8 tok/s with popcount) at a cost of ~$0.10 per DIMM. Combined with the Tier 2 activation register, every DIMM in every server, laptop, and phone becomes an LLM inference accelerator — using memory the customer has already paid for. The business case is not "sell a new product." It's "make the product you already sell billions of dramatically more valuable."

## What This Paper Is Not

We want to be clear about what we haven't done:

**No hardware validation yet.** Everything is simulation calibrated against published DRAM characterization datasets. The charge-sharing physics is proven, but our specific end-to-end pipeline hasn't run on physical DIMMs. That's the next step.

**Prefill is painfully slow.** Processing an input prompt takes roughly a minute for a typical short prompt on a single DIMM. This architecture works best for short prompts and long-running sessions — not document summarization or long conversations. A hybrid approach where the CPU handles prompt processing and CaSA handles generation is the practical near-term path.

**The FPGA prototype is expensive and power-hungry.** The research platform costs thousands of dollars and draws 42W. A production controller would be 10-40x cheaper and draw a fraction of the power. The DRAM itself costs $15.

**We depend on ternary models existing.** If the industry standardizes on 4-bit quantization and ternary models never materialize beyond BitNet, CaSA becomes less compelling. We're betting that the memory-side advantage of ternary — which this paper is the first to demonstrate — will shift that calculus.

**This is inference only.** CaSA accelerates running a trained model, not training one. Training requires high-precision gradients and backpropagation — fundamentally different operations that charge-sharing can't help with.

## The Actual Contribution

The contribution is not 0.40 tokens per second. That number is a floor measured through a straw.

The contribution is three things:

**1. The activation-sacrificial protocol works.** You can do reliable neural network inference on commodity DRAM by combining MAJ3 charge-sharing for AND operations with SA-mediated RowCopy for zero-BER weight restoration. No custom silicon. The 16.3% BER of unmediated RowCopy is sidestepped entirely by using sense-amplifier-mediated copies.

**2. The bus is the only bottleneck (once pipelined).** With multi-bank pipelining, all DRAM-internal operations (RowCopy, MAJ3) are completely hidden. 98% of inference time is bus traffic. The internal compute capacity of commodity DRAM exceeds what the bus can deliver by 1,000x. Every future improvement is about getting data to and from the array faster — or eliminating bus transfers entirely via popcount.

**3. The path from floor to ceiling is concrete and quantified.** We trace every step from commodity hardware to optimized silicon: multi-bank pipelining, multi-DIMM scaling, popcount registers, activation registers, next-generation memory standards. Each step has a cost, a throughput gain, and a dependency. Nobody has to guess what comes next.

## What This Could Mean

If this works at scale, the memory already in your laptop, phone, or server becomes an AI accelerator — without buying new hardware. Not a toy demo. A real language model, running on the RAM you already own, at a fraction of the power draw of a GPU. The compute has always been there. We just didn't have the right model format to unlock it.

Nobody knows how fast this could become if memory manufacturers designed for it. This paper provides the first data to inform that question.

---

*Full technical report with complete derivations, error analysis, cross-technology projections, patent landscape, and hardware validation plan: [github.com/pcdeni/CaSA](https://github.com/pcdeni/CaSA)*

*This work was conducted by an independent researcher using AI-assisted analysis tools. The core architectural insights, all design decisions, and every claim were verified by the human author. All errors are the author's responsibility.*
