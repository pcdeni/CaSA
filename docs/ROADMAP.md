# Roadmap — open levers and investigations

The public mirror of the project's improvements ledger, trimmed of internal
items. Statuses: **RUN** / **READY** / **DESIGNED** / **IDEA** / **PARKED**;
finished levers move to DONE with their measured result. Every lever cites
the measurement or design that motivates it — the roadmap is itself
evidence-first.

**The scoring rule, and why it inverted.** The goal is a prompt in and a
response out at the speed the DDR bus allows, with nothing but the prompt and
the response crossing PCIe and the weights already resident. Against *today's*
wall — which is host orchestration — host-side levers score high and
device-cycle levers score near zero. Against the *goal*, that reverses
exactly: host-loop levers score zero at the end state because the host loop is
gone, and device-cycle levers become everything. Levers below are marked with
which axis they pay on. A "wall-neutral" verdict expires with the wall it was
measured on.

---

## A. Fabric-side (each needs an FPGA build and a flash)

1. **Direct emission — the keystone.** ⇑ *device axis; the main open lever.*
   Instead of the host assembling and sending a program per operation, the
   fabric emits the DDR command stream for a whole projection from a compact
   descriptor. Proven at projection scale: scaled emission to 1080 bodies with
   the delta inside the certified configuration's own replay envelope, a
   **3.37× body-lane wall**, flat to 1080 bodies, at a cadence of ~8 clock
   cycles per column access. This is the rung the remaining orders of
   magnitude live on; everything else in section A is either its dependency or
   its consequence.
   ⚠ Known bounded defect: an emission that outlives its maintenance deadline
   can strand mid-record. Zero occurrences at production shapes; fenced in the
   design.

2. **Cross-bit-plane accumulator (ACCUM_XBP).** ⇑ *device axis.* Sums the K
   activation bit-planes in fabric, so a group drains once instead of once per
   plane. Silicon-exact RTL, validated bit-exact against a CPU reference.
   **Production net-negative today** — it removes drains that the host cadence
   was already hiding — and that measurement is the cost model working. It is
   a **hard dependency of direct emission**, where the drains it removes stop
   being hidden. Default off.

3. **Per-segment popcount readback (SEG_POP).** Shipped and silicon-validated:
   2048 B/row of per-segment popcount bytes instead of the 8 KB raw row,
   bit-identical to the plain readback path. **Wall-neutral, default off** —
   re-measured twice, 0 % delta, because the c2h bytes it saves are hidden
   inside the request cadence. It becomes live again on the device axis, where
   nothing hides them.
   ⚠ Its long-standing "SEG_POP is permuted" mystery was a *simulation* defect,
   not silicon: the behavioural readback FIFO model emitted the low half of
   each 512-bit write first, where the IP presents the high half first. Two
   compensating swaps cancelled for a plain read and could not cancel for
   SEG_POP. Fixed in `rtl/` and `sim/e2e/`; correlation 0.037 → 0.998.

4. **Inter-bender fabric link.** Router RTL authored and Verilator-gated (idle
   bit-identity, cross-clock delivery, wedge-free backpressure, frame-atomic
   reroute), synthesized and inert by default — a star topology until a route
   is enabled. It also carries a popcount BIST register so every future image
   proves the popcount fix is synthesized. Missing: the consumer side and the
   send primitive. Turns a host-mediated cross-DIMM hop into a fabric one,
   which is what an in-fabric attention stage needs.

5. **Recovering the second rank.** ⇑ *capacity, not speed.* The modules are
   dual-rank and the board family routes chip-select for both ranks, but the
   command encoding's rank bit is dropped before the pins, so half of every
   module is unreachable — 4 GiB addressable where 8 is installed. Recovering
   it doubles residency and dissolves the subarray aliasing in section C.
   Needs a controller change and a memory-interface regeneration.

6. **Nonlinears in fabric** — DESIGNED, and re-framed by measurement. Area was
   never the blocker: the layer-norm / activation / rope datapaths *fit*. The
   blocker is the clock. Routed at the fabric's period they close at 11–38 MHz
   (rope is ~384 logic levels deep), so no nonlinear can be a producer at any
   area without a 15–20-stage pipelining workstream. That workstream is the
   item, not the area budget.

7. **Streaming instruction fetch (Rung-1).** CLOSED. The producer loop is
   correct and the RTL works; the wall moved **−4.0 %**, which is not worth a
   default. Off. Superseded on its own axis by direct emission.

8. **Fabric-resident projection loop (Rung-2 probe).** A closed-loop top wired
   the sequencer to a deterministic DRAM model and the exact popcount /
   accumulate datapath, and returned **27/27 byte-exact** integer partials for
   a projection's chunk/bitplane loop with no host command in the loop.
   Feasibility established. Superseded as a *build* by direct emission, which
   reaches the same place with less new hardware.

---

## B. Host and software levers (no new bitstream)

9. **Dual-subarray LOAD pools** — IDEA. Server helpers exist. Doubles
   residency, which shifts traffic toward the packed path. Needs a second
   calibrated subarray and pool layouts per bank — now a translation rather
   than a sweep (section C), so the cost is minutes per window.
   ⚠ Audit every candidate window modulo 2^15 first (section C).

10. **Lane roles must match the population.** ⇈ *free, and it was costing
    1.43×.* The server generates resident constants only on lanes it believes
    are compute lanes; a lane it believes is storage re-reads the constant
    material over the bus on every operation. A compiled-in role table left
    from an earlier population made half the channels 1.43× slower *per DRAM
    operation*, uniformly across prefill and decode, and it hid for weeks
    because nothing about it looks like a performance bug. Proved causal in
    both directions on the same channel, minutes apart, with one environment
    variable. Roles now come from `calibration/DIMM_POPULATION.conf`; the
    remaining work is removing the compiled-in default entirely.

11. **Pipelined dual-DIMM join** — DESIGNED, bridge only. Splitting a matmul
    across two channels and waiting for both costs `E[max] − E[mean]`. Measured
    honestly, that join is **0.362 s/tok = 3.27 %**, not the 16 % a first
    reading suggested: 99.7 % of the spread is common-mode jitter, the join
    already recovers 76 % of the independent-jitter cost, and neither die is
    reliably faster (a 49.5 % coin flip). A one-deep asynchronous join is worth
    a realistic 2–3 %. Ranked below the keystone, and it disappears at the end
    state.

12. **xrefresh / accumulator knob tuning** — minor; only if a measurement says
    so.

---

## C. Characterization and capacity

13. **Calibration transfer** — DONE, and it is the result that makes the rest
    affordable. Banks and same-family dies transfer byte-identically with a
    margin re-screen only; subarrays transfer by *translation* of the relative
    tuple geometry plus a pool re-derivation; different parts need the lattice
    re-derived first. Recipe and evidence: `docs/CALIBRATION_TRANSFER.md`.
    Adopting a subarray went from ~28 minutes to ~2.5 minutes per window.

14. **16-bank scale-out** — DONE as configuration. All 16 banks carry
    validated pool fixtures from a single transfer, verified byte-identical,
    and the bank-similarity audit found classification-identical spread tables
    on never-calibrated banks (350/350 rows × 14 primitive cases) with zero
    off-lattice deposits on the null control.

15. **⚠ Subarray windows alias modulo 2^15** — MEASURED, and it invalidates
    naive capacity claims. These parts decode 15 row bits, so rows `r` and
    `r + 32768` are the same silicon. Of 44 characterized windows on the
    production die, **19 pairs were aliases — only 25 were physically
    distinct**, and one of them turned out to be the production window under a
    different name. Aliased windows pass every screen independently and then
    overwrite each other, and a byte-verify cannot catch it because each write
    verifies against what it just wrote. Every pool list must be audited mod
    2^15. Production and every active configuration are clean; the affected
    historical accuracy figures are withdrawn.

16. **Cross-die determinism, now n=4 across manufacturing dates** — MEASURED.
    Four modules of one part number: identical co-activation fault set on all
    four channels, identical read/write screen, `PERFECT_CLONE` on every
    `t_23`. And the whole production trio — calibration, pool, window — ran on
    a sibling die with no fresh calibration, no fresh screen, no new fixture,
    token-exact on first contact. The earlier cross-*part-number* result stands
    alongside it: the invariance follows the die design.

17. **Certification is not one test, and the numerics oracle is blind to one
    failure** — MEASURED. A channel passed RowClone, byte-lane, read/write and
    the matmul oracle, then latched a byte lane *during* its own full-model run
    while the oracle's correlation stayed at 1.0. Clearance therefore means
    RowClone across all `t_23`, a byte-lane map, a read/write screen — **and the
    same checks again after sustained traffic**. The repair path costs
    essentially nothing in wall terms (a channel repairing hundreds of millions
    of column substitutions ran within 0.2 % of a clean one, and returned the
    exact reference tokens).

18. **Boundary atlas** — DONE for the working windows: the long-range
    co-activation offsets are predecoder-block-relative, shrinking toward a
    ~512-row block midpoint, vanishing, then flipping sign. This is why bank
    transfer is free and subarray transfer is not.

19. **Storage-role channels** — the role is a deployment choice now, not a
    part limitation. Every installed module is compute-grade; assigning one the
    storage role is a configuration line. The design work that assumed a
    permanently MAJ3-dead tier survives unchanged, because it was written
    against the *role*.

---

## D. Model and application levers

20. **Coarser activation quantization (`PIM_ACT_K`)** — SHIPPED and SETTLED.
    The activation is decomposed into K bit-planes = K MAJ3 bodies = K
    round-trips, so dropping K cuts the binding wall proportionally, with no
    accuracy loss down to a model-specific floor that tracks the training
    recipe rather than the weight width. Measured full-model, each against its
    own K=8: **Bonsai-1bit K=6 −21.7 %, Bonsai-ternary K=6 −22.2 %, BitNet-2B
    K=5 −32.2 %**, all token-identical, numerics correlation 0.99995. K=4
    collapses on Bonsai; BitNet tolerates it but K=5 is the safe floor.
    Defaults are set per model and are not to be re-tuned. Client-side only.

21. **Batched-token shapes** — IDEA; the grouped-response protocol is ready as
    the carrier.

22. **More group-scaled model families** — READY anytime; the weight-spec path
    is generic. Value is generality, not throughput.

23. **KV cache in DRAM** — DESIGNED (`docs/KV_ALLOCATOR_DESIGN.md`). Pages are
    segment-aligned, prefix sharing is by reference with a single
    RowClone at the fork boundary, and placement constraints appear only where
    a charge-sharing operation touches the page's block. Capacity sizing is
    against 4 GiB per DIMM, which is what is addressable.

24. **Prefetch conveyor** — DONE as a host scheduler, dry-tested 9/9 card-free
    (degenerate small models, a valid large-model schedule, the three residency
    properties, the bandwidth crossover, and the configuration wire round-trip).
    `python/pim_conveyor.py`; design in `docs/CONVEYOR_DESIGN.md`. The
    review-gated server twin is in `app/experimental/conveyor/`.

25. **LoRA over DRAM** — PARKED. Sketch in `docs/TRAINING.md`.

26. **In-orbit DRAM scrub** — PARKED. In-DRAM majority/RowClone self-scrubbing
    of commodity memory against radiation upsets.

---

## E. Repository and explainers

27. Explainers and their claim ledgers live under `docs/explainer/`. Every
    quantitative claim in a published page carries its evidence.

---

## DONE — with the measured result

- **Slice-split dual-DIMM dispatch** (`PIM_DUAL_SPLIT=slice`) — **1.78×**.
  Dispatches every (token, output-slice) request whole, round-robin across
  servers, so each server sees half the request *count* at full size; the
  older grouped-byte split only halved bytes and was worth 1.53×. The promoted
  default. In `python/pim_linear.py`.
- **Vectorised bit-plane pack** (`PIM_XBP_VEC`) — **−7.79 % on the token wall**,
  95 % CI [−10.33, −5.25]. The pack was the largest pure-Python item on the
  client; the vectorised kernel is byte-exact by construction, not by
  approximation, and the per-token counter went 996 → 9 ms. Default on.
- **Resident constants** — **1.293×**. Constant operand rows generated once on
  a compute lane instead of re-read per operation. See lever 10 for what
  happens when a lane's role is wrong.
- **Cross-round program packing** — each round's write programs packed into a
  few IMEM-bounded ones; write-only, so immune to the receive-wake tax that
  made naive batching lose. Token-identical, per-request write cost 10.8 → 6.4 ms.
- **Request batching (grouped + single-track)** — one request per server per
  slice instead of one per scale group, token-identical.
- **1-bit single-track protocol** — the server computes one track and the
  client reconstructs the other, halving per-request DRAM work. 1.81× on the
  1-bit lane.
- **Bank-set configuration plumbing** — resident-weight pool and scheduler
  generalized past banks 0–3 to any host-configured bank set, seeded at
  power-up and mutable at run time, default-inert.
- **Clone-resident products (lane 2)** — 59 µs/gate resident vs 150–180 for the
  per-column write path, ~2.7× per product; fidelity trade and capacity limits
  characterized.
- **Plane-packed multi-read totals** — the multi-read accumulate regime is
  exact and byte-identical to the reference. It also surfaced a silent-skip
  integrity hazard, now observable: any application whose results map one-to-one
  onto executed programs must check that the platform's oversize-skip counter
  did not advance across the batch.
- **Output-numerics gate** — the project had no full-coverage numerics gate;
  token identity is insensitive (it passed at 87 % wrong masks). Built as a
  correlation gate, because raw per-operation output is not bit-stable across
  processes while correlation is — and correlation collapses immediately on
  wrong weights or stale masks. Gate mask, weight and pool changes on numerics,
  never on token identity. `python/v2_oracle.py`.
- **Fixture-set refusal** — the same class of defect one level up: a die driven
  with another die's calibration returns a fraction of cells wrong and passes a
  correlation gate. Measured on one channel minutes apart: foreign trio 265/512
  bit-exact at 13.6 % worst relative error, PASS; correct trio 512/512. The
  population is one config file now and the resolver refuses fixtures from
  outside it. `python/dimm_population.py`.
- **Bank-parallel issue (`PIM_PARALLEL_BANKS`)** — engaged, provably, and worth
  ~3 % ≈ variance. That is the ceiling for a compute-issue lever on a
  readout-bound wall, and it is why readout came first. Gotcha recorded:
  batching bit-planes disables it structurally.
- **Coset-broadcast operand fan-out** — ⚠ NEGATIVE in production. One
  charge-sharing operation loads a scratch row byte-exactly from a
  pool-resident source, 265.8× fewer instructions than the per-column write,
  and it contributes nothing: the body recomputes its scratch row, so the
  deposit is never read. The standalone primitive demonstration stands; the
  production integration does not. Default off.
- **Activation-side residency and clone (X-master)** — ⚠ CLOSED NEGATIVE. A
  RowClone establishes the operand at charge-shared, not write-driven, levels,
  so the in-DRAM value is not the value written. Corrupts at production mask
  density on both tracks (correlation 0.94–0.97, 0/2048 bit-exact) where the
  same shape without it is bit-exact. Hard-gated off. Revivable only if the
  seed can be made write-driven-equivalent, which charge-sharing physics argues
  against.
- **Streamed packed matmul** — correctness certified (zero framing, poison or
  fatal events across 26 arms), wall **+0.67 %** once a teardown artifact was
  removed: neutral. Off. Three independent brackets — bytes, program count and
  drain discipline — each moved the wall by nothing, which is what named the
  host cadence as the binder rather than any of them.
- **Descriptor-serve engine** — honest negative as a wall lever: execution
  alone floors at the whole wall by construction. Retained as a certification
  accelerator, where sharing one session across walks is 3.4× faster *and* more
  exact.
- **MAJ5 chain policy A/B** — honest negative (99.51 vs 99.90); the published
  reference policy stands.
- **q3_K quantization coverage** — 99.90 %; quant coverage complete.
