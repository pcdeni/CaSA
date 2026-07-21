# Roadmap — open levers and investigations

The public mirror of the project's living improvements ledger
(trimmed of internal/user-gated items). Statuses: RUN / READY /
DESIGNED / IDEA / PARKED. Dated entries; finished items move to
DONE with their measured result. Every lever cites the measurement or
design doc that motivates it — the roadmap is itself evidence-first.

---

## A. Bitstream-gated (each = a Vivado build + a JTAG flash)

1. **SEG_POP readback mode** — **SHIPPED 2026-07-21 (build 7)**, silicon-
   validated: 2048 B/row per-segment popcount bytes (byte-aligned 4×,
   chosen over the sketched 1536 B packing for trivial host unpack),
   READ/DIFF bit-identical to build 6, WNS +0.118 ns. Production wiring
   `PIM_SEGPOP=1` in `app/test_bitnet_server.cpp`; full-model A/B in
   `docs/ROADB_2026_07.md` §7. Design history:
   `docs/PRODUCTION_ROADB_DESIGN.md`, `rtl/README.md` (build 7).
2. **seq_engine pipeline integration** — DESIGNED (`rtl/SEQ_ENGINE.md`),
   deliberately sequenced AFTER SEG_POP (recv 3.1ms > exec 1.0ms).
   Mixed-stream Verilator non-regression A/B is the flash gate.
3. **Streaming/queued execution — "controller-native on our card"** —
   UPGRADED (see `docs/CONTROLLER_NATIVE.md`, the full investigation).
   MVDRAM's testbed was DRAM Bender on an Alveo U200 — the same
   soft-MC class as ours; "controller-native" is their §V-E *execution
   regime* (the DDR command bus never waits for the host), and it is
   achievable here: ping-pong IMEM pair + a fetch stage that loads the
   idle bank during EXECUTE, back-pressured by `buffer_space`. The
   host becomes a pure producer at PCIe bandwidth (~0.3 % used — our
   problem was only ever latency). This is where the round-trip lever
   family converges, and the last 2–3 orders per `PAPER_CONTRAST.md`
   gap 2 live. Ladder above/below it: pipelined issue (software
   bridge), on-fabric orchestrator (soft core; 1 round-trip per
   projection — the end-state demo), and the honest Rung-4 boundary
   (commodity MCs expose no command-level control; every published
   unmodified-DRAM PUD result runs a soft/custom controller).

0. **ACCUM_XBP (build-8): cross-bit-plane accumulator** — DESIGNED
   (`docs/ACCUM_XBP_DESIGN.md`). In-fabric place-value sum: one 8 KB
   drain per group instead of 8 per-plane drains (recv wakes ÷8,
   ~1.6× projected on the measured wake-dominated recv). Verification
   gate = the build-7 discipline verbatim. Note: the driver-side
   attack on the same term (xdma poll_mode) measured EIO on this
   build — ladder caught it before any timing claim; rolled back
   clean. The fabric cut does not depend on driver behavior.

## B. Host/software levers (no bitstream needed)

3b. **V2GS request batching** — DONE 2026-07-21: `MAGIC_V2GS` composes
   the grouped response (V2G) with single-track (V2S) — one request per
   server per slice instead of one per scale-group, token-identical,
   268.8 → 260.3 s /8 tok (+3.2 %, the pipe-framing share). Default on
   (`PIM_REQ_BATCH=1`). The batched profile refines the cost model: a
   slice = ~20 XDMA round-trips (~12 write programs + ~8 exec/recv) ×
   ~150–200 µs — which sets up the next lever precisely.
3c. **V2 cross-round program packing** — DONE 2026-07-21: each round's
   12 write programs packed into ~3 IMEM-bounded ones (write-only, no
   c2h — immune to the recv-wake tax that made K-batching lose).
   Token-identical; wcol 10.8 → 6.4 ms/request; 8-tok wall
   260.3 → 233.0 s. Default on (`PIM_V2_PACK=0` restores the legacy
   cadence byte-for-byte). Tonight's stack: 267.1 → 233.0 s (−12.7 %).
   Originally specified as: Fuse a slice's ~20 programs into few: one program
   interleaving [write(round r); 4-bank MAJ3 bodies(round r)] across
   rounds within the 8K-IMEM envelope — preserving write-then-use
   locality (the upfront-batched-writes attempt of 2026-05-04 is the
   documented anti-pattern; the MM3D packed path is the proven
   pattern). Estimated 2–4× on the handler → ~2–3× wall. Gate:
   layer-0 exact, then full-model token-identity.
   Method context for both: `docs/METHOD_MVDRAM_LENS.md`.

4. **LANE2_WRES clone-resident products** — DONE 2026-07-21: 59 µs/gate
   resident vs 150–180 pcwrite (~2.7×/product); fidelity trade and
   capacity limits documented in `docs/ROADB_2026_07.md` §5.
5. **Plane-packed multi-read totals** — DONE 2026-07-21: the multi-read
   accum regime validated EXACT (all-resident numpy-exact,
   byte-identical); per-plane-gate 32–53 µs vs 59–65 at moderate M;
   wall-neutral until residency capacity grows. The bring-up also
   surfaced a silent-skip integrity hazard now fixed with
   `oversize_skips()` observability — read `docs/ROADB_2026_07.md` §6
   before building accum-total systems on this stack.
6. **M3 coset-broadcast operand fan-out** — IDEA→DESIGN next. The
   production wcol killer AND the unlock for V2-path packing (PACK_ROUNDS
   is MM3D-only because V2 needs write-then-use locality — broadcast
   loading changes the locality story). Needs pool-layout co-design with
   the validated sub-lattice broadcast mechanism
   (`docs/LATTICE_ADDRESSING_2026_07.md`).
7. **PIM_PARALLEL_BANKS=1 probe** — DONE 2026-07-21: pack4 provably
   ENGAGED (program-dump signature) yet wall effect 3.3 % ≈ variance —
   the ceiling math for a compute-issue lever on a readout-bound wall.
   Confirms readout-first sequencing. Gotcha recorded: any
   `PIM_INLINE_BITPLANES>1` batch disables pack4 structurally
   (duplicate-bank serial fallback). `docs/UTILIZATION.md` addendum.
8. **Dual-subarray LOAD pools** — IDEA. Server helpers exist (bc_pool_idx
   dual mode). Doubles residency ⇒ shifts V2→MM3D traffic (where
   PACK_ROUNDS works). Needs a second calibrated subarray + pool layouts
   per bank (D0 s77 robust 88% is a candidate).
9. **V2G protocol for streaming/batched shapes** — DONE as protocol
   (wall-neutral), READY as the carrier for any future batched regime.
10. **xrefresh / accum-knob tuning** — minor; only if a measurement says.

## C. Characterization / science (learn + enable)

11. **Bank-similarity audit** — TRANCHE 1 CONFIRMED 2026-07-21
    (`docs/BANK_AUDIT_2026_07.md`): four never-calibrated banks under a
    verbatim-transferred calib produce classification-identical spread
    tables (350/350 rows × 14 primitive cases); even the flake fringe is
    deterministic and bank-invariant; the null control shows zero
    deposits off-lattice on every bank. Calibration transfer = margin
    re-screen only, demonstrated on zero-characterization silicon —
    the 16-bank scale-out (#13, ~4× residency/parallelism headroom
    with no new sweeps) is real. Remaining: margin maps, banks 8-15,
    the selection-law probe on one new bank.
12. **Calib-transfer procedure** — formalize: apply bank-0 calib to a new
    bank with margin re-screen only (already half-exploited: D0 banks
    0/2/3 share calib; cross-die transfer evidenced). Deliverable: a
    documented recipe + transfer-success table.
13. **16-bank / multi-subarray scale-out** — IDEA, after 11. The idle
    spatial parallelism from docs/UTILIZATION.md (die ~99.99% idle).
    Constraints known: tFAW/tRRD scheduling (pack4 machinery), per-bank
    pools, c2h contention.
14. **640/1024 boundary atlas** — IDEA. Map sense-amp-segment vs
    predecoder-block boundaries per bank; structural confirmation of the
    replicated-block hierarchy (the two-granularity finding).
15. **D1/D3 storage roles** — PARKED-ish. MAJ3-limited dies as weight
    parking + RowClone shuttle. Revisit only if capacity binds.

## D. Model / application levers

16. **Bonsai/BitNet batched-token shapes** — IDEA; V2G-ready carrier.
17. **More g128 model families** — READY anytime (weight-spec path is
    generic); value = generality story, not throughput.
18. **LoRA-over-DRAM demo** — PARKED for later. Design sketch in
    docs/TRAINING.md.
19. **LEO in-orbit DRAM scrub** — PARKED for later (in-DRAM MAJ/RowClone
    self-scrubbing of COTS memory against radiation upsets).

## E. Publication / story (repo = go-to for in-memory LLM)

20. **Explainer HTML scenes for the 07-21 material** — ledger rows staged
    (`docs/explainer/pim_explainer_ledger.md` 07-21 block); scenes need
    their own careful pass.
21. **256-token sampled e2e writeup** — RUN (in flight);
    fold the verdict table into MVDRAM_REPRODUCTION when done.

## DONE (move rows here with the measured result)

- 2026-07-21: Road-B lane2 integration (product dataflow 1.4–28×; accum
  crossover ~50K products; 65K totals 0 faults) — `docs/ROADB_2026_07.md`.
- 2026-07-21: 1-bit single-track V2S (18.7 s/tok, 1.81×, ladder 5.36×).
- 2026-07-21: MAJ5 ZERO+2 chain A/B — honest negative (99.51 vs 99.90;
  SiMRA ONE+3 stands).
- 2026-07-21: q3_K Phase D (99.90%) — quant coverage complete.
- 2026-07-21: PACK_ROUNDS triage — MM3D-only, ~3% lever in production;
  the real unlock is broadcast/clone loading (→ lever 6).
