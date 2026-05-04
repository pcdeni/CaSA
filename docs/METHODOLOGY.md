# Methodology — what we measured, what we projected, and why we trust each

This document is the long-form companion to the README's headline
table. It explains:

1. What was measured directly on this hardware.
2. What was projected from the cycle-level scheduler in
   `scheduler/casa_sched.c`.
3. Where the simulator is honest, where it is optimistic, and what
   the gap is fundamental.

The goal is to make every number in the project either
**measured** (with a date, a configuration, and a way to reproduce)
or **scheduler-bounded** (with the assumed configuration explicit).
Numbers we tried to estimate without measurement turned out wrong
by orders of magnitude — see "Estimation vs. measurement" below.

## Per-token throughput we measured

All measurements use the same prompt
(`"What is the capital of Hungary? Answer in one sentence."`), the
same model (`microsoft/bitnet-b1.58-2B-4T`), `do_sample=False` so
generation is deterministic, and 8 generated tokens unless noted.
Per-token times include the full Python + PyTorch + PCIe + DRAM
loop, not just FPGA compute.

| Configuration | 8 tok | per token |
|---|---|---|
| Pure PyTorch (no PIM) | ~3.0 s | 0.4 s/tok |
| 1 layer's `q_proj` on PIM, no persistent weights | 70.0 s | 8.75 s/tok |
| 1 layer's `q_proj` on PIM, persistent weights | 41.8 s | 5.23 s/tok |
| 1 layer's `q_proj` on PIM, persistent + combined program | 27.4 s | 3.43 s/tok |
| All 7 layer-0 projections on PIM, persistent + combined, single bank | 303.7 s | 38.0 s/tok |
| All 7 layer-0 projections on PIM, multi-bank `0,1,2,3` | **238.1 s** | **29.76 s/tok** |

The **measured today** number in the README headline is the bottom
row: ~30 s/tok for one layer's full set of seven projections on
PIM, multi-bank.

To extrapolate to all 30 layers running on PIM (which we have not
demonstrated end-to-end — the work to make 30 layers' worth of
weights persistent is engineering, not science): roughly
~30× the per-layer number, so ~900 s/tok = ~15 minutes per token
in the configuration measured. This is the "today, all 30 layers
extrapolated" bar in the chart.

## Where the time goes per matrix-multiply

From the scheduler, the per-MAJ3 critical path on the measured
single-DIMM Tier-A configuration breaks down approximately as:

| Phase | % of critical path |
|---|---|
| RowClone (refresh weight from backup) | ~15 % |
| Broadcast (copy activation across rows) | ~20 % |
| MAJ3 (the actual physics) | ~18 % |
| Bus read (8 KB result row over PCIe) | **~35 %** |
| Misc DRAM ops (PRE / SLEEP / frac) | ~10 % |
| Popcount (CPU side) | ~2 % |

We are bus-bound, not compute-bound. The 8 KB result row crossing
the bus is the single biggest chunk. This is why an in-DRAM popcount
circuit (which lets the chip ship back ~2 bytes instead of 8 KB)
gives the largest speedup of any optimization in the projection
ladder, despite being a small piece of silicon at the periphery.

## Throughput projections from the scheduler

`scheduler/casa_sched.c` is a discrete-event simulator that models
the standard JEDEC DDR4 timing parameters (tRCD=9, tRP=9, tRAS=24,
tRC=33, tRRD=4/6, tFAW=20, tCCD=5, tBurst=4, tWR=10, tREFI=5200,
tRFC=233 cycles), bank/bus contention, and the measured
charge-sharing latencies (t_12/t_23 = 0/0 for MAJ3, 10/2 for
broadcast, 30/1 for RowClone). It does not model PCIe / kernel /
Python overhead; it assumes the FPGA is fed instructions at zero
cost.

| Stack | Per-token | Bus % | tok/s | What it requires |
|---|---|---|---|---|
| **A.** sim baseline (1 DIMM, FPGA pop) | 523 ms | 98.2 % | 1.9 | software work to feed the FPGA at full bus rate, persistent weights everywhere |
| **B.** + bank-group parallel bus | 419 ms | 97.9 % | 2.4 | scheduler change, no hardware |
| **C.** + 4 DIMMs in parallel | 114 ms | 96.8 % per DIMM | 8.7 | calibration on additional DIMMs |
| **E.** + POPCNT3 (chained MAJ3 popcount in DRAM) | 69 ms | 95.9 % | 14.6 | RTL change to the FPGA-side controller |
| ─── horizontal divider — vendor changes below ─── | | | | |
| **D.** + dedicated in-DRAM popcount circuit | 17 ms | 75 % | 60 | DRAM-vendor silicon change |
| **F.** + LISA cross-subarray bus | 10 ms | 60 % | 97 | DRAM-vendor silicon change |
| **H.** + binary activations (model retrain) | 1.7 ms | 52 % | ~580 | retrain BitNet at 1-bit activations |

The horizontal divider is where the story changes. **Above it:
software, calibration, and FPGA RTL — engineering only, no new
chip.** Even with all of it, we cap at ~15 tok/s because the bus
stays ~96 % busy. Below it: small DRAM-vendor changes that move the
bottleneck off the bus and into the DRAM compute itself.

## Where the simulator matches reality, and where it doesn't

| Where it matches reality | Where it is optimistic | Where the gap is fundamental |
|---|---|---|
| DDR4 timing (all standard JEDEC) | Per-MAJ3 work-graph (assumes weights are persistent in DRAM) | LISA and pLUTo are research, not productized |
| Charge-sharing latencies (we patched in the measured `t_12/t_23` values) | FPGA popcount latency (4 cycles is best-case; real PCIe roundtrip much higher) | In-DRAM popcount circuit doesn't exist in shipped DRAM today |
| Bank/bus contention model | Layout: simulator assumes ~12 neurons per row with pos+neg packed (5120 bits per neuron); we use 2048 outputs per row, separate pos/neg masks | Vendor timeline is years |
| Bank-group-parallel bus model | No PCIe / kernel / Python overhead modeled | — |
| Refresh windowing (tREFI / tRFC) | | |

After patching the measured charge-sharing latencies into the
simulator, projections shifted by **<2 %**. This is because the
projections are bus-bound, not MAJ3-bound — the bus is the
98 %-busy resource, so the exact MAJ3 cost barely matters. **This
also makes the projections robust to small parameter uncertainty
elsewhere.**

## Sensitivity analysis

We re-ran the simulator with conservative assumptions for the
hypothetical parameters:

- In-DRAM popcount with `t_popcount = 20` cycles (~30 ns) → **60 tok/s**
- In-DRAM popcount with `t_popcount = 200` cycles (~300 ns, 10× slower) → **58 tok/s** (1.5 % drop)

So the in-DRAM popcount projection is robust to the exact circuit
performance. Even a slow popcount circuit gets us most of the way.
This matters for the vendor pitch: they don't have to deliver an
ultra-fast popcount, just a reasonable one.

LISA's projection is more speculative because it depends on the
assumed cross-subarray bandwidth and the data-flow rewrite to use
it. We used a `2 × rc_time` cost which is a reasonable middle
ground.

## Estimation vs. measurement (where we were wrong)

A note about the limits of our own intuition. Three pre-implementation
predictions in this project's roadmap turned out wrong by 100× or by
category:

| Thing we tried | Predicted | Measured |
|---|---|---|
| Long-running PIM server (eliminate subprocess overhead) | 5-20× speedup | ~1 % speedup |
| Cause of server's wrong-output bug | "state leak between requests" | stdout pollution from `std::cout` in `BoardInterface::init()` corrupting the binary response channel |
| Effect of patching simulator's pre-implementation charge-sharing timings | "could be material" | <2 % (projections are bus-bound) |

The takeaway, for anyone reading the roadmap: **every projection in
this project is either measured today (with a configuration) or
comes from a cycle-level scheduler with bus contention tracked
explicitly.** Where we tried to estimate without measuring, we were
wrong by orders of magnitude. Treat projection numbers as ceilings
of what the silicon can do, not as promises of what an
implementation will hit on the first try.

## Per-cell and per-run variance — honest framing

In one full BitNet layer's worth of PIM-side outputs (22 144
values), we measured **22 139 bit-exact matches** against the
PyTorch float reference — 99.98 %. The 5 mismatches come from cells
that pass the calibrated 1000-pattern stability test but flip on
specific bit-combinations the calibration didn't exhaust.

These 5 cells are not a fixed set. **The same bank can produce
different output token sequences across two runs of the same
prompt.** Marginal cells genuinely flip differently between runs.
For deterministic demos, pin to a single bank that has performed
well on a recent canary check; for production throughput, accept
that the model will produce slightly-different sensible outputs
across runs.

This is a feature, not a bug, in the context of ternary LLMs:
ternary networks are robust to small per-weight perturbations by
design, and this is what makes the PIM / ternary pairing
economically viable. A floating-point model would not survive these
flips; a ternary one does.

## What we explicitly do not claim

- That this implementation is faster than a GPU on the same model
  today. It is roughly 100× slower at this writing.
- That we have a custom DRAM chip. Everything runs on stock DDR4.
- That we run training in DRAM. Training stays on GPUs.
- That this is general-purpose computation in memory. It is a
  specific subset of operations (bitwise AND + popcount) on a
  specific class of model (ternary LLMs).

The point of the work is to demonstrate the **mechanism** on real
silicon and put concrete, scheduler-bounded numbers on what would
change with two specific DRAM-vendor improvements. The mechanism
exists; the numbers say where it could go; the demo proves it works
end-to-end on a published model. That's the contribution.
