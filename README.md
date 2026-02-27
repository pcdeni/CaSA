# Your RAM Is Secretly an AI Accelerator

**CaSA: Ternary LLM Inference on Commodity DRAM**

*February 2026*

---

## The Hidden Compute Inside Every Memory Chip

Every stick of RAM in your computer has a hidden trick. When you force two rows of memory cells to turn on at the same time — which violates the timing spec, but physically works — the electrical charges mix together and you get a free AND operation across tens of thousands of bits simultaneously. Nanoseconds. Almost zero energy.

This has been measured. The CMU-SAFARI group tested it 79 million times across 120 real DDR4 chips. Zero failures in the reliable operating window. The physics works. It has always worked. Every DRAM chip ever manufactured can do this.

The compute capacity inside the chip is over 1,000x more than the memory bus can deliver. It's just sitting there, unused.

## Why Nobody Could Use It

The compute exists, but previous attempts to harness it for anything useful ran into a fatal problem: to set up the operation, you need to copy data around inside the chip (called RowCopy). On commodity DDR4, RowCopy has a 16.3% bit error rate. That's not a rounding error — that's one in six bits flipped. Neural network inference is impossible at that error rate.

Every prior approach to "Processing-in-Memory" either required custom silicon (Samsung HBM-PIM, SK Hynix AiM, UPMEM) or stopped at demonstrating basic bitwise operations without building anything useful on top.

## The Fix: Stop Copying, Start Sacrificing

Our fix is embarrassingly simple.

In a neural network, there are two kinds of data:
- **Weights** — the model's learned knowledge. Permanent. Written once, read millions of times.
- **Activations** — the intermediate values flowing through the network. Temporary. Freshly computed every single step, then thrown away.

The charge-sharing trick has an asymmetry: the first row you activate survives intact. The second row gets overwritten with the AND result.

So: activate the weight row first (it survives), then the activation row second (it gets consumed). The weights are preserved. The activations were going to be discarded anyway. You get the AND result with essentially zero errors — no RowCopy needed.

Error rate drops from 16.3% to less than 0.000004%. Four orders of magnitude. That's the entire paper in one paragraph.

We call this the **activation-sacrificial protocol**, and the full architecture **CaSA** (Charge-sharing Activation-Sacrificial Architecture).

## Why Ternary Changes Everything

This trick works cleanly only at one specific precision: **ternary** — where neural network weights are restricted to {-1, 0, +1}.

Why? Because multiplying a ternary weight by a binary activation is literally just an AND gate. That's exactly what charge-sharing gives you for free. You encode +1 as one binary row, -1 as another, AND each with the activation bits, and the difference gives you the matrix-vector product.

At higher precisions (4-bit, 8-bit), the number of AND operations per weight multiplies rapidly. Only at ternary does it collapse to something commodity DRAM can handle competitively.

The industry currently evaluates ternary on the wrong axis. The question people ask is: "Does ternary match INT4 accuracy on GPUs?" Answer: roughly yes (Microsoft's BitNet b1.58 matches LLaMA quality), but GPUs aren't optimized for ternary, so there's no speed benefit. Conclusion: ternary seems pointless.

That analysis completely misses the memory axis. Ternary is the **only** precision at which every RAM chip in the world becomes a neural network accelerator. The reason nobody saw this is that nobody had demonstrated commodity DRAM PIM actually working for inference until now.

## Why Now

This couldn't have been done two years ago. Microsoft published BitNet b1.58 — the first production-quality ternary language model — in February 2024. Before that, there were no ternary models worth running. The DRAM physics has existed since the 1970s. The charge-sharing trick has been measured since 2017. But until ternary models arrived, there was nothing to connect the compute substrate to the workload. CaSA is what happens when those two threads finally meet.

## What We Actually Built

We designed a complete inference pipeline for **BitNet b1.58-2B-4T** — a real 2-billion-parameter ternary language model from Microsoft — running on a single 8 GB DDR4 DIMM ($15-25) with an FPGA controller.

The DRAM handles the heavy matrix multiplications via charge-sharing AND. The FPGA handles the lightweight operations: popcount (counting 1-bits in the result), accumulation, RMSNorm, SiLU activation, and softmax. The model fits in a single DIMM with room to spare.

**Current speed: 1.8 tokens per second on one DIMM.**

That's slow. A CPU running llama.cpp does 15-30 tok/s on the same hardware. We know. Here's why it doesn't matter:

## The Bus Bottleneck (and Why 1.8 Is a Floor, Not a Ceiling)

The 1.8 tok/s is almost entirely bus overhead. Here's where the time goes:

| Component                              | Share of Inference Time |
| :------------------------------------- | :---------------------: |
| **Writing activations to DRAM (Bus)**  |         **44%**         |
| **Reading results from DRAM (Bus)**    |         **44%**         |
| Charge-sharing AND (Compute)           |           6%            |
| FPGA overhead                          |           6%            |

The in-DRAM compute takes 6% of total time. The other 88% is just moving data through the 64-bit DDR4 bus. The chip can compute 1,000x faster than the bus can deliver data. You're looking at a thousand-lane highway feeding through a single-lane toll booth.

This means every improvement that reduces bus traffic produces dramatic speedups:

## The Scaling Path

| Configuration                          | Tokens/sec  | What it takes                            |
| :------------------------------------- | :---------: | :--------------------------------------- |
| **1 DIMM (Baseline)**                  |   **1.8**   | **Works today on unmodified DDR4**       |
| 4 DIMMs                               |     7.6     | $60 of commodity RAM, no chip changes    |
| 4 DIMMs + Batching                    |     ~35     | Firmware optimization only               |
| **+ In-DRAM Popcount**                 | **60–166**  | **~2,000 gates per bank (~$0.10/DIMM)**  |
| LPDDR5X (16-ch) + Popcount            |     169     | Phone/laptop memory, single package      |
| HBM2 (8-ch) + Popcount                |     229     | Server memory                            |

The popcount register is the single biggest lever. It's a tiny bit-counting circuit — about 2,000 logic gates — that counts the 1-bits in a DRAM row without reading the data out through the bus. This eliminates the entire 44% read bottleneck. Samsung patented this exact circuit in 2014. It has never been shipped in any product.

## It's Surprisingly Robust

A natural question: if you're doing computation by mixing analog charges, how fragile is this?

Not very. Even at a bit error rate of 0.01% — ten thousand times worse than what was measured on real hardware — model output quality degrades by less than half a percent. The safety margin between measured reliability and the point where accuracy starts to suffer is roughly 50,000x. Commodity DRAM, within its validated timing window, is not fragile.

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

- **Publish your charge-sharing timing parameters.** Right now, finding the optimal timing for dual-row activation on a specific die revision requires reverse-engineering via tools like DRAM Bender. If manufacturers documented the safe operating window per die revision, it would replace months of characterization with a datasheet lookup.

**Tier 1 — Tiny silicon changes, massive impact:**

- **In-DRAM popcount register (~2,000 gates/bank, <0.3% die area, ~$0.10/DIMM).** This is the single highest-impact change. After a charge-sharing AND, the result sits in 65,536 sense amplifiers. Currently, we have to read all 8,000 bytes out through the bus just to count the 1-bits. A popcount register counts them in-place and returns a single 16-bit number. This eliminates 44% of total inference time — the entire read bottleneck. Samsung patented exactly this circuit in 2014. It's combinational logic (no clock, no pipeline, no state machine), so it works at full speed even on DRAM-process transistors. It's a passive reduction circuit, not a processor.

- **Reliable RowCopy.** Our activation-sacrificial protocol exists because RowCopy is broken at 16.3% BER. If manufacturer calibration (like PUDTune's sense amplifier offset compensation) brought RowCopy BER below 0.01%, two things happen: (1) we can distribute activation data inside the chip without touching the bus, roughly doubling throughput even without popcount, and (2) we can build a "software-defined popcount" — an adder tree constructed entirely from sequences of charge-sharing AND/OR/NOT operations inside the chip, using the SIMDRAM approach. This would break the bus bottleneck on completely unmodified DRAM with zero additional circuitry. It would be slower than a dedicated popcount register (~100-200 charge-sharing steps per accumulation vs. one cycle), but it would work today if RowCopy were reliable.

**Tier 2 — Moderate silicon, transformative results:**

- **Per-bank activation register (a few hundred thousand transistors per bank).** Right now, we rewrite the activation data from the bus for every single weight row — because charge-sharing destroys the activation row each time. A small static register per bitline would hold the activation vector and drive it onto the bitlines repeatedly without being destroyed. Combined with popcount, this eliminates ALL bus transfers during compute. Bus utilization drops from 88% to under 5%. A single DIMM becomes deeply compute-bound rather than bus-bound.

- **Wider rows.** This is counterintuitive: the industry trend is toward narrower rows (2 KB in LPDDR5X and HBM, vs 8 KB in DDR4) for latency and power reasons. But for PIM, row width is the fundamental unit of parallelism — each charge-sharing AND processes one full row simultaneously. DDR4's 8 KB rows pack 25 neurons per AND operation. LPDDR5X's 2 KB rows pack only 6, requiring 4x more sequential cycles. A PIM-optimized memory would maximize row width, not minimize it. DDR4's wide rows are an accidental PIM advantage that future memory standards should preserve.

**The bottom line for manufacturers:** The Tier 1 popcount register alone converts CaSA from a proof-of-concept (1.8 tok/s) to a competitive inference engine (60-166 tok/s) at a cost of ~$0.10 per DIMM. Combined with the Tier 2 activation register, every DIMM in every server, laptop, and phone becomes an LLM inference accelerator — using memory the customer has already paid for. The business case is not "sell a new product." It's "make the product you already sell billions of dramatically more valuable."

## What This Paper Is Not

We want to be clear about what we haven't done:

**No hardware validation yet.** Everything is simulation calibrated against the SiMRA measurement dataset. The physics is proven (79M trials), but our specific end-to-end pipeline hasn't run on physical DIMMs. That's the next step.

**Prefill is painfully slow.** Processing an input prompt takes roughly a minute for a typical short prompt on a single DIMM. This architecture works best for short prompts and long-running sessions — not document summarization or long conversations. A hybrid approach where the CPU handles prompt processing and CaSA handles generation is the practical near-term path.

**The FPGA prototype is expensive and power-hungry.** The research platform costs thousands of dollars and draws 42W. A production controller would be 10-40x cheaper and draw a fraction of the power. The DRAM itself costs $15.

**We depend on ternary models existing.** If the industry standardizes on 4-bit quantization and ternary models never materialize beyond BitNet, CaSA becomes less compelling. We're betting that the memory-side advantage of ternary — which this paper is the first to demonstrate — will shift that calculus.

**This is inference only.** CaSA accelerates running a trained model, not training one. Training requires high-precision gradients and backpropagation — fundamentally different operations that charge-sharing can't help with.

## The Actual Contribution

The contribution is not 1.8 tokens per second. That number is a floor measured through a straw.

The contribution is three things:

**1. The activation-sacrificial protocol works.** You can do reliable neural network inference on commodity DRAM by exploiting the asymmetric survival property of charge-sharing. No RowCopy. No custom silicon. Four orders of magnitude better reliability than any prior approach.

**2. The bus is the only bottleneck.** 88% of inference time is bus traffic, 6% is compute. The internal compute capacity of commodity DRAM is not the limiting factor — it exceeds what the bus can deliver by 1,000x. Every future improvement is about getting data to and from the array faster.

**3. The path from floor to ceiling is concrete and quantified.** We trace every step from commodity hardware to optimized silicon: multi-DIMM scaling, batch processing, popcount registers, activation registers, next-generation memory standards. Each step has a cost, a throughput gain, and a dependency. Nobody has to guess what comes next.

## What This Could Mean

If this works at scale, the memory already in your laptop, phone, or server becomes an AI accelerator — without buying new hardware. Not a toy demo. A real language model, running on the RAM you already own, at a fraction of the power draw of a GPU. The compute has always been there. We just didn't have the right model format to unlock it.

Nobody knows how fast this could become if memory manufacturers designed for it. This paper provides the first data to inform that question.

---

*Full technical report with complete derivations, error analysis, cross-technology projections, patent landscape, and hardware validation plan: [github.com/pcdeni/CaSA](https://github.com/pcdeni/CaSA)*

*This work was conducted by an independent researcher using AI-assisted analysis tools. The core architectural insights, all design decisions, and every claim were verified by the human author. All errors are the author's responsibility.*
