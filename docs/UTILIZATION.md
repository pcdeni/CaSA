# Utilization: cycles vs resources — how far from optimal is in-DRAM LLM inference?

The pipeline-occupancy question, answered with this project's own
measurements: which independent resources (rows, banks, DDR PHY, PCIe,
host) are busy, which are idle, what binds today, and what a memory die
would need to change. Numbers below are measured on this rig unless
marked as projections.

## The utilization pyramid

| Level | Resource | Utilization today | Evidence |
|---|---|---|---|
| 0 | The op itself: one doubleACT MAJ3 = a 65,536-bitline majority in tens of ns | ~at physics | t12/t23 dwells are real charge-sharing time (calibrated per DIMM); not shortenable |
| 1 | DDR command bus during command issue | ~9.4 % slot util | seq_engine Verilator A/B: the stock fetch/decode/execute pipeline emits 1 op/cycle; 3 of 4 bus slots idle. seq_engine reaches 100 % (8–10.7×), module validated (`rtl/SEQ_ENGINE.md`) |
| 2 | DDR bus across a whole program | **< 1 %** | of the measured 5.9 ms/program (`wcol 1.3 + exec 1.0 + recv 3.1 + pop 0.2 + other 0.3`), actual DDR-active time is tens of µs; the rest is transfer + turnaround |
| 3 | Banks / DIMMs | 4 of 16 banks per die; 2 of 4 DIMMs | dual-DIMM measured 1.91–1.95× (96–98 % of ideal halving); D1/D3 excluded on measured silicon grounds; 16-bank issue never attempted (tFAW/tRRD were a measured constraint in the pack4 root-cause) |
| 4 | Subarrays / rows as compute elements | ~0.2 % of rows touched, tiny duty cycle | one 16-row calibrated tuple + a few-hundred-row pool per bank, of ~10⁵ rows/bank |
| 5 | PCIe bandwidth | **~0.3 %** | 2.6 GB per 8 tokens over ~270 s ≈ 17 MB/s vs multi-GB/s XDMA |
| 6 | Host orchestrator (python, syscalls, pipes) | ~free | the V2G result: collapsing 81,689 → 10,541 requests/token bought ~1 % of wall |

Multiply the levels and the honest headline is: **the die is ~99.99 %
idle**. That is not an indictment — it is the room. It is why this
project's measured ladder keeps producing integer-factor wins
(632 → 47.5 s/tok BitNet; 100 → 18.7 s/tok Bonsai) without touching
the physics once.

## What binds: not the bus, not PCIe — the program round-trip

Today's binding term is the **per-program turnaround at the FPGA/host
interface**: first the readback drain (`recv` 3.1 ms — XDMA
small-transfer serialization, *not* PCIe bandwidth), then operand
loading (`wcol`) and fetch-limited command issue (`exec`). The lever
ladder maps one-to-one:

1. **SEG_POP** (readback reduction in the FPGA datapath,
   `PRODUCTION_ROADB_DESIGN.md`) kills `recv` → projected ~25.6 s/tok
   BitNet upper bound.
2. **seq_engine** (`rtl/`) kills `exec` — sequenced second because it
   accelerates the smaller term.
3. Packed/streamed execution kills per-program turnaround (the T2
   packing measurement already showed dispatch amortizing until
   readback dominated — same ordering).
4. Only then does the **DDR bus** bind — the `casa_sched` bus-bound
   floor (tens of tokens/s class for BitNet, a projection, labelled as
   such). MVDRAM's §V-E is the existence proof of that regime: a host
   generating commands faster than DDR4's ~1.5 ns/command consumption,
   i.e. a saturated PHY. We are ~2 orders below it; the staged path is
   the climb.

## What a memory die would need (ranked by which measured wall it kills)

None of these require new cell technology — every one is periphery or
interface, consistent with this project's premise of exploiting the
cell exactly as manufactured:

1. **A reduction unit on the row buffer** (per-segment or whole-row
   popcount at the sense amps). Kills the readout wall — our largest
   term. The entire Road-A (in-DRAM adder trees) / Road-B (FPGA
   accumulator) / SEG_POP arc is an emulation of this one missing die
   feature, one hop too far from the cells. Micron (US10068652) and
   Samsung (US9836277) hold patents of exactly this shape.
2. **Multi-row activation as a documented primitive** with deliberate
   decode isolation and guaranteed margins. Kills the correctness tax:
   the co-activation selection law, XOR-spread-safe placement, MAJ
   self-pollution screening, and the per-module lottery all exist only
   because we drive the row decoder out of spec.
3. **In-die row-copy / broadcast as first-class commands** (RowClone,
   JEDEC-blessed). Kills operand movement (`wcol`) — our second wall;
   the coset-broadcast technique is the bootleg version.
4. **Subarray-parallel issue** (SALP-class). The die already contains
   16 banks × dozens of independently-sensable subarrays; the
   interface serializes them. This is the 2–3 orders of spatial
   parallelism idle in the table — what MVDRAM's subarray×module
   partitioning and our dual-DIMM 1.95× both sample the edge of.
5. **Compute-region refresh semantics**. Refresh restores charge, not
   content (the measured drift result); compute rows want maskable
   refresh or latched staging.
6. **A "PUD-capable" datasheet bin**. Result A — the exact named part
   performing zero PUD across 60,000 pairs while working perfectly as
   memory — is a part-lottery a vendor could end with one binning test.

## One-line summary

The primitive is near physics; the die is ~99.99 % idle; the walls are,
in order, readout → operand movement → command issue → and only then
the bus — and every one of them is a periphery fix, not a cell fix.

## Addendum (2026-07-21): are banks copies of each other?

Yes in structure, no in margins — and our own data shows the split
directly. DRAM is step-and-repeat at every level (cell → row →
mat/subarray with its sense-amp stripe → bank → bank group → die), and
the periphery replicates with the arrays. Measured consequences:

- **DIMM 2's XOR-spread profile is byte-identical across its banks**
  (couples R⊕{1,2,4,…,512}, never ⊕256, same signature every bank) —
  the spread is decoder wiring, and every bank carries the same decoder.
- The selection law's predecoder groups and the two overlapping
  granularities (640-row sense-amp segments vs 1024-row predecoder
  blocks) are **design constants**, not per-bank accidents — different
  periphery layers repeat at different pitches, and their misalignment
  is observable.
- What varies per bank is **yield**: on DIMM 0, banks 0/2/3 share one
  calibrated tuple while bank 1 needs its own; per-bank column masks and
  flake patterns differ. Law replicates; margins are per-instance.
- One level up the same pattern holds: fault sets and calibration
  transfer byte-identically across same-model **dies**.

Practical consequence: **characterization transfers**. Scaling from 4
banks to 16 (and to more subarrays — the idle spatial parallelism in
the table above) needs no re-derivation, only a cheap margin re-screen
per new bank against the known laws. The 16-bank bank-similarity audit
is on the roadmap (`ROADMAP.md` §C) as the enabling experiment.
