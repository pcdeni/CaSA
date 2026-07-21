# The MVDRAM design method — and our system audited through it

MVDRAM (arXiv:2503.23817) is both a result and a *method*. The result
(GeMV on unmodified DDR4) we reproduce and contrast elsewhere
(`MVDRAM_REPRODUCTION.md`, `PAPER_CONTRAST.md`). This document is about
the **method** — how the authors reasoned from "an LLM" and "a DRAM
chip" to a working system — and then applies that same reasoning to
*our* system to find where the leverage actually is. It doubles as the
rationale behind the roadmap: every open lever is scored here against the
one constraint the method says organizes everything.

## Part 1 — What MVDRAM actually did (the method, not the trick)

Read past the two named innovations (horizontal layout, on-the-fly
vector encoding) and the paper is a six-step mapping of a workload onto
an unmodified substrate. Each step has a *why* that generalizes.

1. **Profile the workload down to its dominant, substrate-shaped
   kernel.** LLM *decode* (autoregressive, batch-1) is
   memory-bandwidth-bound and dominated by GeMV (weight matrix ×
   activation vector). They did not attack "LLM inference"; they
   attacked the one kernel that is both the bottleneck *and* a shape the
   substrate can help with. *Why:* PIM only pays if the moved kernel is
   simultaneously on the critical path and substrate-amenable — anything
   else is work spent off the critical path.

2. **Enumerate the substrate ISA and its cost asymmetries.** Unmodified
   DDR4 under multi-row activation offers bitwise AND/OR across
   co-activated rows (charge sharing) and RowClone (in-subarray copy).
   The load-bearing fact is a *cost asymmetry*: reduction *along*
   bitlines is nearly free; moving data *across* columns is expensive —
   the "column-to-column data movement" limitation. *Why:* you cannot
   optimize against a substrate you have not cost-modelled, and the
   asymmetry is the whole game.

3. **Make the fundamental limitation the organizing axis.**
   Column-to-column movement is THE constraint, so both innovations
   exist only to avoid it: horizontal matrix layout aligns the reduction
   with the cheap (bitline) direction; on-the-fly vector encoding
   presents the activation in a form the bitwise op consumes with no
   cross-column shuffle. *Why:* one dominant constraint becomes a design
   compass — every choice reduces to "does this avoid the expensive
   direction?"

4. **Name the degrees of freedom and spend them on the constraint.** The
   free variables are (a) data layout — which axis is "along bitlines",
   (b) bit-plane decomposition of low-bit values (a low-bit multiply =
   Σ_i 2^i·popcount(AND of bit-planes)), (c) the work-partition between
   DRAM and the host/controller, (d) operand encoding. They chose
   horizontal layout, bit-serial planes, host-side place-value sum +
   sign, DRAM-side parallel AND. *Why:* layout + encoding + partition
   *are* the design; fixing them well is the whole method.

5. **Co-design host and memory around the partition.** DRAM does the
   embarrassingly-parallel part (AND across thousands of columns); the
   controller does the inherently-serial part (accumulate popcounts with
   2^i weights, apply sign). *Why:* PIM is not "everything in memory" —
   it is putting the parallel part where parallelism is free and keeping
   the serial part where control is cheap.

6. **Bound the regime and quantify per-regime, honestly.** Wins are for
   *low-bit* GeMV (INT2/4/8): up to 7.29× on the kernel, 2.18×
   end-to-end; high-precision floating point destroys the bit-serial
   economics, and they say so. *Why:* a method claiming universal wins
   is untrustworthy; bounding the regime is what makes the numbers
   credible and tells the reader when to reach for it.

The transferable core: **profile → enumerate the ISA cost model →
choose the one organizing constraint → spend layout/encoding/partition
on it → co-design host+memory → bound the regime.**

## Part 2 — Our system, audited step by step

Running the same six steps against our stack (BitNet / Bonsai on
DRAM-Bender + BCU1525) is clarifying, because at step 2 it exposes a
**structural difference that reframes everything we do**.

**1. Workload decomposition — we are correctly focused, with one honest
inversion.** We substitute 196/196 linear layers (the GeMV bulk); LM
head, embeddings, attention, RMSNorm stay in PyTorch. On a GPU those
"leftovers" (attention/KV) can dominate — but for *us* the linears are
the slow part (they run in DRAM-PIM), so the leftovers are a tiny
fraction of *our* wall. Moving more of them into DRAM would *lower* the
in-DRAM fraction and not help the wall. So the correct target is making
the linears cheaper, not moving more work in. The one genuine
decomposition gap is **prefill vs decode**: we optimize decode (batch-1
GeMV); prefill is GeMM (many activation vectors × one weight matrix),
where a resident weight row can be reused across the whole prompt — a
lever we have not spent (see Part 3, R2).

**2. ISA cost model — richer primitives, and a DIFFERENT organizing
constraint.** Our substrate ISA is a superset of MVDRAM's: MAJ3 (3-input,
not just 2-input AND), RowClone, the XOR-spread *co-activation lattice*
(a value lands in address-sibling rows for free — a cheap quasi-horizontal
movement they do not have), coset broadcast, and the APA selection law
(which coset activates). But the decisive difference is the **cost
model**, which we measured (`UTILIZATION.md`, the wall work):

> MVDRAM's expensive operation is *column-to-column movement*, because
> their operations are DDR commands issued by the real memory
> controller — the issue cost itself is ~free. OUR expensive operation
> is the **host↔DRAM round-trip**, because we drive the array over PCIe /
> XDMA from a host, and each round-trip carries a ~1.5 ms fixed latency.

That is the whole reason our wall is *request-count-bound* while theirs
is *column-movement-bound*. Same method, different substrate cost model,
different top constraint. It is also why SEG_POP (a 4× byte cut) was
wall-neutral: it reduced bytes-per-round-trip, not round-trips.

**3. The organizing constraint — the host↔DRAM round-trip.** Every lever
should be scored on one axis: *does it reduce round-trips (count ×
fixed latency), or only the work inside a round-trip?* This single axis
orders the entire roadmap (Part 3). It is the honest statement of where
we are: our bottleneck is an artifact of host orchestration, not DRAM
physics — and the bus-bound floor `casa_sched.c` projects is what a
memory-controller-native design (MVDRAM's premise) would see instead.

**4. Degrees of freedom — several unspent.**
- *Layout* (spent): vertical for production, horizontal for the lane2
  repro; SEG_POP lets vertical keep readout parity.
- *Encoding* (partly spent): 1-bit single-track (V2S) computes only the
  positive track — half the DRAM work — and the host reconstructs
  y = 2·y_pos − Σ pc(x). Ternary still pays a dual-track tax; whether a
  cheaper ternary encoding exists is open (Part 3, M1).
- *Work-partition* (actively moving): SEG_POP pushed popcount into the
  FPGA; Road-B pushes accumulation in. **Still on the host:** the
  place-value weighted sum across bit-planes. Folding that into the
  fabric (an accumulator that weights each plane by 2^i and sums
  in-fabric) turns n_bitplanes readback vectors into ONE — a round-trip
  cut at the RTL level (Part 3, RTL1). This is the standout RTL lever.
- *The lattice as a DoF* (unspent): the XOR-spread deposits operands in
  sibling rows for free — our analog of their horizontal layout. Using
  it to place operands where the reduction is cheap kills the
  host-marshalled per-column writes (`wcol`) — coset-broadcast operand
  fan-out (Part 3, RTL3 / roadmap #6).

**5. Host/memory partition — push the serial-but-fabric-cheap work in.**
The method says keep only genuinely-serial control on the host. Our host
still does the bit-plane place-value sum and the group-scale
accumulation; both are cheap in fabric and would *also* cut round-trips.
The cross-bit-plane accumulator (DoF 4c) is exactly this move.

**6. Regime, honestly.** We win where MVDRAM cannot be independently
checked (their source is unpublished): a *reproducible* in-DRAM LLM on
unmodified DDR4, a richer primitive (MAJ3 → exact adders), bit-exact or
near-exact fidelity, and the readout collapse. We lose on absolute
s/tok, because host round-trip latency dominates our rig — a property of
the *harness*, not the physics. Stating that plainly is what makes the
levers legible: they are all about closing the gap to the
controller-native floor.

## Part 3 — The lever taxonomy this produces (scored by round-trips)

Every lever tagged by *how it attacks the organizing constraint*:
**[×N]** cuts round-trip count ~N-fold; **[B]** cuts bytes/work per
round-trip (helps only once transfers are big); **[F]** raises fidelity
or generality (value, not speed). Cross-referenced to `ROADMAP.md`.

| id | lever | attack | status |
|---|---|---|---|
| **now** | request batching (V2GS): all scale-groups of a slice in one request | **[×g]** (g = groups/slice, ~16 for Bonsai) | implementing |
| R2 | prefill weight-row reuse across prompt tokens (GeMM regime) | **[×T]** (T = prompt tokens) | idea |
| R3 | dual-subarray / multi-bank residency → fewer LOAD round-trips, less V2 fallback | **[×]** + capacity | roadmap #8/#13 |
| RTL1 | cross-bit-plane accumulator: weight each plane by 2^i in-fabric, return ONE vector | **[×n_bitplanes]** | NEW — Part 2.4c |
| RTL2 | seq_engine: host feeds compound programs; DDR consumes without a round-trip each | **[×]** on issue | designed (`rtl/SEQ_ENGINE.md`) |
| RTL3 | coset-broadcast operand fan-out: load operands via the lattice, not per-column host writes | kills `wcol` **[B]** + **[×]** | roadmap #6 |
| B1 | SEG_POP (shipped): per-segment popcount readback | **[B]** 4× bytes | done |
| M1 | cheaper ternary encoding / weight-zero sparsity skip (BitNet dual-track tax) | **[F]/[×]** per-model | idea — Part 2.4 |
| M2 | 1-bit models already single-track; batching+residency is their whole win | — | 1-bit optimal |

The re-scoring changes priorities. Before this audit, `seq_engine` was
"sequenced after readout" on a *bytes* argument; the audit says it
attacks *round-trip count*, which IS the wall — so its priority rises
once request batching is banked. Conversely SEG_POP's byte win is real
but only cashes in once batching makes transfers large (they compose).
The cross-bit-plane accumulator (RTL1) is new and high-value: it is the
partition move step 5 demands, and it is a round-trip cut, not a byte
cut.

## Part 4 — Direct answers to "what else, where, how"

- **Rig experiments:** request batching (now); prefill weight-reuse
  (R2); dual-subarray residency (R3); the 16-bank similarity audit
  (roadmap §C) that makes residency scale-out cheap.
- **RTL:** the cross-bit-plane accumulator (RTL1) is the highest-value
  next bitstream — it is the partition move and a round-trip cut in one;
  then seq_engine (RTL2, re-prioritized up); coset-broadcast operand
  fan-out (RTL3) for the `wcol` term.
- **Per model:** 1-bit (Bonsai) is already encoding-optimal
  (single-track) — its wins are batching + residency, pure round-trip
  levers. Ternary (BitNet) carries a dual-track tax — its lever is an
  *encoding* one (cheaper ternary representation, or skipping
  weight-zero columns), the DoF MVDRAM's method points straight at.
  Generality (more g128 families) is an [F] lever: value, not speed.

The through-line: MVDRAM's method says find the one constraint and spend
layout/encoding/partition on it. For us the constraint is the host
round-trip, and the biggest unspent moves are (1) fewer, larger requests
and (2) pushing the place-value sum into the fabric. Both are underway
or specified above.
