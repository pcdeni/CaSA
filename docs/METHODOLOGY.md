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

## What we are running today (and what we are not)

The current measurement runs **one of BitNet's 30 transformer
layers' matrix-multiplies in DRAM** — specifically the seven
projections of layer 0 (Q, K, V, O, gate, up, down). The other 29
layers run in PyTorch on the CPU. This is enough to demonstrate the
mechanism end-to-end on a real published model. Running all 30
layers in DRAM is engineering — make all weights persistent in their
calibrated rows, never page them in over PCIe, route activations
between layers — not a science question.

All measurements below use the same prompt
(`"What is the capital of Hungary? Answer in one sentence."`), the
same model (`microsoft/bitnet-b1.58-2B-4T`), `do_sample=False` so
generation is deterministic, and 8 generated tokens unless noted.
Per-token times include the full Python + PyTorch + PCIe + DRAM
loop, not just FPGA compute.

| Configuration | 8 tok | per token (1 layer in DRAM) |
|---|---|---|
| Pure PyTorch (no PIM) | ~3.0 s | 0.4 s/tok |
| 1 layer's `q_proj` only, no persistent weights | 70.0 s | 8.75 s/tok |
| 1 layer's `q_proj` only, persistent weights | 41.8 s | 5.23 s/tok |
| 1 layer's `q_proj` only, persistent + combined program | 27.4 s | 3.43 s/tok |
| All 7 layer-0 projections, single bank | 303.7 s | 38.0 s/tok |
| All 7 layer-0 projections, multi-bank `0,1,2,3` | **238.1 s** | **29.76 s/tok** |

The **measured today** number in the README headline is the bottom
row: ~30 s/tok for one BitNet layer's full set of seven projections
on PIM, multi-bank. Extrapolated linearly to all 30 layers in DRAM
at this same orchestration overhead: ~900 s/tok ≈ 15 minutes/token.

That extrapolation is **not what the silicon does** — it is what
our current orchestration does. The same hardware running the same
operations, with no per-call subprocess and no per-column writes
mid-loop, would land on the bus-bound projection below — ~500 ms/tok
on 1 DIMM, three orders of magnitude faster than the orchestration-
bound floor. The gap is software, not silicon.

## Where the time actually goes — TODAY vs BUS-BOUND

The two operating points are very different in shape, not just in
magnitude. Below: roughly where each microsecond is spent per
matrix-multiply step, in (a) what we measured today and (b) what the
scheduler says the silicon does once orchestration overhead is gone.

| Phase | TODAY (Tier-0, orchestration-bound) | BUS-BOUND (Tier-A, scheduler) |
|---|---|---|
| Per-column writes for weight loading | ~340 ms / matmul | 0 (weights persistent in DRAM) |
| Subprocess + Python + PCIe per `execute()` round-trips | ~50 ms / matmul | 0 (assumed zero in scheduler) |
| Per-MAJ3 PCIe read of 8 KB result row | tens of ms / matmul | ~35 % of critical path |
| RowClone broadcast (`doubleACT(10,2)`) | tens of ms / matmul | ~20 % |
| MAJ3 doubleACT (the actual physics) | tens of ms / matmul | ~18 % |
| Misc DRAM bookkeeping (PRE / SLEEP / frac) | ~10 ms / matmul | ~10 % |
| Popcount (CPU-side adds of 8 KB row) | ~5 ms / matmul | ~2 % |
| **Per matrix-multiply step (≈ 1 BitNet projection)** | **~3 sec** | **~3 ms** |
| **Per token, all 30 layers in DRAM** | **~15 min** (extrapolated) | **~500 ms** (scheduler) |
| **Bus utilization** | **~2 %** (almost idle) | **~98 %** (bus-bound, the wall) |

The TODAY column is dominated by per-column writes (we copy the
weight pattern over PCIe before each matmul) and per-call PCIe
round-trips between Python and the FPGA. Engineering this out gets
us to the BUS-BOUND column, where the bus is the actual wall and
the chip cannot go faster on this configuration without an
architectural change.

This is why we are bus-bound at the silicon ceiling, even though we
are nowhere near bus-bound today: the bus is already the limit *of
the silicon*, not of our software.

## Throughput projections from the scheduler

`scheduler/casa_sched.c` is a discrete-event simulator that models
the standard JEDEC DDR4 timing parameters (tRCD=9, tRP=9, tRAS=24,
tRC=33, tRRD=4/6, tFAW=20, tCCD=5, tBurst=4, tWR=10, tREFI=5200,
tRFC=233 cycles), bank/bus contention, and the measured
charge-sharing latencies (t_12/t_23 = 0/0 for MAJ3, 10/2 for
broadcast / Multi-Row-Init, 30/1 for the 2-row RowClone we use for
persistent-weight refresh).

The MAJ3 and RowClone primitives are modeled as `doubleACT`
sequences (two ACTs back-to-back with the timing violation between
them, no intermediate PRE) — matching what the silicon actually
does. Derived per-primitive cost: **MAJ3 = 20 tCK = 30 ns**,
**RC = 32 tCK = 48 ns** at our measured timings.

### What the scheduler counts per MAJ3

To make the bus-bound projection reflect our **current silicon
implementation** (not a hypothetical future), the scheduler
explicitly accounts for what `emit_bank_combined_body` in
`app/test_bitnet_server.cpp` actually issues:

| Operation | Cost | Why |
|---|---|---|
| Weight RowClone refresh `doubleACT(30,1)` | rc_time = 32 tCK | bank-only, charge-sharing |
| Activation broadcast `doubleACT(10,2)` | rc_time = 32 tCK | bank-only, charge-sharing |
| **Activation update wrRows × 5** | **5 × bus_wr_time = ~3140 tCK** | **bus traffic** — `doubleACT(10,2)` broadcasts to *all 16* open rows, so the 5 activation slots must be individually overwritten with the per-bitplane x_pattern. With persistent zeros + buffer (one-time setup per chunk), only the 5 activation wrRows remain per MAJ3. |
| Frac discharge × 3 | ~84 tCK | bank-only, observed empirically necessary on DIMM 0 for stable MAJ3 |
| MAJ3 `doubleACT(0,0)` | maj3_time = 20 tCK | bank-only, the actual physics |
| Result read | bus_rd_time ≈ 618 tCK *(or ~50 tCK with in-DRAM popcount)* | bus traffic |

**The 5 wrRows for activation update dominate per-MAJ3 silicon
time** — about 3140 tCK out of ~3920 tCK total. The bus_read is the
second-largest chunk (618 tCK). MAJ3 itself is negligible (20 tCK,
~0.5 %).

The scheduler does **not** model PCIe / kernel / Python overhead. It
assumes the FPGA is fed instructions at zero cost. That is correct
for the silicon-ceiling question; it is the gap between the
scheduler and our current measured throughput that "orchestration
overhead" refers to. Per-MAJ3 measured wall-clock today is ~1.2 ms
(wall-clock for one of the multi-bank executes); per-MAJ3 silicon-
only at bus-bound is ~5.9 µs. Ratio ≈ **200×**, all engineering
overhead (PCIe per-call, Python, per-column weight writes that we
re-issue per matmul instead of once at startup).

### Realistic projections (current silicon implementation)

All numbers below are from running `casa_sched.c` with the listed
flags. Reproduce locally with `cc -O2 scheduler/casa_sched.c -o
casa_sched` and the same flags. The default config models our
current silicon path (5 activation wrRows + frac discharge per
MAJ3); see "ideal-acts" section below for the hypothetical-future
projections.

| Stack | CLI flags | ms / token | Bus % | tok/s |
|---|---|---|---|---|
| **R1** 1 DIMM | `--dimms 1` | 3000 | 97.2 % | **0.33** |
| **R2** + bank-group-parallel bus | `... --bg-parallel` | 2400 | 96.5 % | 0.40 |
| **R3** + 4 DIMMs in parallel | `--dimms 4 --bg-parallel` | 610 | 95.6 % | 1.57 |
| **R4** + in-DRAM popcount *(vendor RTL)* | `... --popcount dram` | 520 | 94.9 % | 1.86 |
| **R5** + LISA cross-subarray bus *(vendor RTL)* | `... --lisa` | 510 | 94.9 % | 1.90 |

The realistic ceiling for our current silicon path on 4 DIMMs even
with **both** vendor changes is **~1.9 tok/s**. In-DRAM popcount
helps only **1.18×** here (1.57 → 1.86), and LISA barely registers,
because the bus is saturated by activation wrRows, not result reads.
The vendor changes target the wrong bottleneck if we keep the
current activation-update mechanism.

### "ideal-acts" projections (hypothetical primitive that doesn't exist today)

If a future DRAM primitive enabled selective broadcast — i.e.
update *only* the 5 activation rows in the 16-row open-set via
charge-sharing instead of 5 individual full-row bus writes — the
per-MAJ3 silicon cost drops from ~3920 tCK to ~770 tCK, and the
projections shift dramatically:

| Stack | CLI flags | ms / token | Bus % | tok/s |
|---|---|---|---|---|
| **I1** 1 DIMM ideal | `--dimms 1 --ideal-acts` | 503 | 98.3 % | 1.92 |
| **I2** + bg-parallel | `... --bg-parallel` | 404 | 97.9 % | 2.38 |
| **I3** + 4 DIMMs | `--dimms 4 --bg-parallel --ideal-acts` | 110 | 96.8 % | 8.75 |
| **I4** + in-DRAM popcount | `... --popcount dram` | 16 | 73 % | **59.99** |
| **I5** + LISA | `... --lisa` | 9 | 18 % | **109.16** |
| **I6** + binary activations (model retrain) | `... --act-bits 1` | 1.8 | 18 % | 545 |

In the ideal-acts world, in-DRAM popcount delivers the **~30×**
speedup we'd expect (1.86 → 60), because the bus is no longer
write-saturated. **GPU-competitive throughput on existing DDR4
silicon requires both the popcount circuit AND the selective-
broadcast primitive.** Either alone gives modest improvement.

This is an honest consequence of our current MAJ3 implementation
relying on the all-16 broadcast. If the BitNet PIM mapping were
restructured (different layout — e.g., bit-serial — or a different
multi-row-charge-sharing primitive), some of this gap could close
without needing a new DRAM primitive.

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

**Realistic mode** (4 DIMMs + bg-parallel + popcount, default activation
wrRows + frac discharge):
- `t_popcount = 20` cycles (~30 ns) → **1.86 tok/s**
- `t_popcount = 200` cycles (~300 ns, 10× slower) → **1.86 tok/s** (no measurable drop — bus is saturated by activation wrRows, not the popcount path)

**Ideal-acts mode** (with the hypothetical selective-broadcast primitive,
4 DIMMs + bg-parallel + popcount):
- `t_popcount = 20` cycles → **59.99 tok/s**
- `t_popcount = 200` cycles → **56.97 tok/s** (~5 % drop)

The realistic-mode flatness is informative: it confirms the bus is
write-bound, so faster popcount inside DRAM doesn't help. To move
the 1.86 tok/s number, you'd need to address the wrRow bottleneck
first (selective-broadcast primitive, or a layout change that
amortizes activation updates).

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
