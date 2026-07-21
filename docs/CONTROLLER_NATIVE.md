# Could we be controller-native too? (investigation, 2026-07-21)

Our wall analysis (`METHOD_MVDRAM_LENS.md`) ends at: *we are
host-orchestrated over PCIe; MVDRAM is controller-native; that is why
our organizing constraint is the host↔DRAM round-trip.* The obvious
follow-up: **could we be controller-native as well — and what would it
take?** This document is that investigation. Short answer: yes, in
rungs — and the first two rungs run on the hardware we already own,
because the load-bearing fact is about the *execution model*, not the
silicon.

## First, an honesty correction about what "controller-native" means

MVDRAM's testbed is **DRAM Bender on a Xilinx Alveo U200** — the same
FPGA soft-memory-controller harness class as our BCU1525. They did not
run on a processor-integrated memory controller either; nobody has (see
Rung 4). What they have that we lack is the **streaming execution
regime** (their §V-E): command *generation* runs faster than DDR4's
~1.5 ns/command *consumption*, so generation overlaps execution and the
command bus never waits. Their end-to-end numbers are built on that
regime; a processor-integrated MC is the premise it models. Our rig
instead executes a program, waits for the host round-trip, then
executes the next — ~150–200 µs of dead bus per trip, ~20 trips per
slice. "Controller-native", operationally, = **the DRAM command bus
never waits for a host**. That is an achievable property of our
existing card.

## The ladder

### Rung 0 — pipelined program issue (software only; days)
Post program N+1 over h2c while N executes and its results drain over
c2h. The XDMA h2c/c2h channels are independent, and the platform already
runs an asynchronous drain thread; what serializes today is the
execute→receive→execute call pattern in the server, not the hardware.
Double-buffering hides most of the per-trip latency under the previous
program's execution. Bounded win (~1.5–2×: IMEM load and execute still
serialize per program) and it touches the delicate execute/receive
machinery — the same machinery whose failure modes we have already
root-caused twice (the 8-then-wedge, the transition drains) — so it
gets the full stream-integrity validation treatment, not a casual edit.

### Rung 1 — streaming/queued execution (RTL; the real one)
The FPGA fetches program K+1 from a queue while K executes: a ping-pong
IMEM pair (we already widened IMEM to 8K) plus a fetch stage that loads
the idle bank during execution, back-pressured by `buffer_space`
(exists since build-6). The host becomes a pure producer streaming
programs at PCIe *bandwidth* — which is free for us (~0.3 % utilized;
our problem was only ever *latency*) — and round-trip latency vanishes
from the steady state. The DDR command bus becomes the only clock.
This IS MVDRAM's §V-E regime on our card, and it is exactly what
`PAPER_CONTRAST.md` gap 2 called "the structural performance gap …
where the last 2–3 orders of magnitude live." `rtl/SEQ_ENGINE.md`'s
sequencer (100 % command-bus utilization in Verilator) is the
proof-of-concept for the execute side; the missing piece is the
double-buffered fetch. This rung subsumes the per-program round-trip
levers entirely.

### Rung 2 — on-fabric orchestration (RTL + soft core)
Move the *server loop itself* into the VU9P: a small soft CPU
(MicroBlaze / PicoRV32) or a microcoded sequencer that, given one
activation vector, runs the whole projection — select pool rows, emit
wcol + MAJ3 bodies per round, collect SEG_POP bytes, accumulate with
bit-plane weights (composing with the planned cross-bit-plane
accumulator, `ROADMAP.md` §A.0) — and returns one y vector. Host
round-trips per projection: **1**. The VU9P has abundant free fabric
(the die-utilization table in `UTILIZATION.md`), and every datapath
piece this needs already exists or is on the roadmap; what's new is a
control-flow engine instead of a program stream. This is full
controller-native *inference* on hardware we own: PCIe carries x in and
y out, nothing else.

### Rung 3 — CPU and memory controller on one die (platform port)
PiDRAM-class systems (SAFARI: Zynq SoC — ARM cores + a custom MC on one
die, MC reachable over on-chip AXI at ~100 ns) demonstrate the true
"processor-integrated" shape: the orchestrator is a real CPU a hundred
nanoseconds from the command generator. A port would replace our
150–200 µs trips with ~0.1 µs ones — but Zynq-class boards drive one
DIMM with modest fabric, against our 4-DIMM VU9P with the whole Road-B
datapath. Rung 2 gets the same latency class on better hardware, so
Rung 3 is mainly valuable as the *story* rung (it is the existence
proof that the premise MVDRAM models is buildable) — not our next move.

### Rung 4 — commodity CPU memory controllers (the honest boundary)
Not possible today, and worth saying precisely why: x86/ARM MCs expose
no command-level interface; BIOS timing registers are coarse, init-time
knobs; the MC scheduler reorders commands and injects refresh at will —
you cannot guarantee the cycle-precise ACT→PRE→ACT triplets PUD needs.
This is why *every* published unmodified-DRAM PUD result (ComputeDRAM
MICRO'19, SiMRA, MVDRAM, PiDRAM, ours) drives a soft or custom
controller. What could change it: vendors exposing an MC command mode
(firmware), CXL.mem devices — which contain their *own* programmable
controllers a vendor could open, arguably the most realistic
commercial path to controller-native PUD — or JEDEC adopting
multi-row activation as a documented command (the die-changes list in
`UTILIZATION.md`). Until one of those happens, "controller-native"
means Rungs 1–3.

## What this changes in the plan

1. **Rung 1 is the destination for the round-trip lever family.**
   Request batching (done), cross-round packing (next), and the
   cross-bit-plane accumulator all reduce trips; Rung 1 eliminates
   waiting on them. They are not wasted — packed programs and in-fabric
   accumulation are exactly what a streaming fetch wants to consume —
   but the roadmap should name Rung 1 as where they converge.
2. **Rung 0 is a cheap bridge** worth taking only with full
   stream-integrity validation (sentinels, oversize-skip checks).
3. **Rung 2 is the repo's end-state demo**: "hand the card an
   activation vector, get the projection back" — the strongest possible
   form of the go-to-repo claim, on hardware we already own.

Sequenced accordingly in `ROADMAP.md` (§A) and the internal ledger.
