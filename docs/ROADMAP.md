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
3. **Streaming/queued execution** — IDEA. The §V-E regime (host feeds
   commands faster than DDR consumes): program-queue in fetch, or seq
   compound programs. The last 2–3 orders per PAPER_CONTRAST gap 2.
   Design after 1+2 land.

## B. Host/software levers (no bitstream needed)

3b. **Client-side request batching (fewer, larger requests)** — NEW
   2026-07-21, MEASURED-IN, top wall lever. `req-prof` decomposition:
   the client keeps the pipe full (gap 0.2 ms), each request ≈ one
   program ≈ 3.2 ms, wall = ~5,400 requests/forward × 3.2 ms; `recv`
   is XDMA-latency-dominated (~1.5 ms fixed/read — why SEG_POP's 4×
   byte cut was wall-neutral); `PIM_INLINE_BITPLANES=4` measured a net
   LOSS (requests already carry ~1 program). Fix: batch the ~28
   per-chunk requests per projection into few multi-unit requests (the
   V2S wire format already allows it) → fewer, larger recv windows —
   which is also when the SEG_POP byte collapse starts paying.
   Validation gate: bit-exact y per op, then full-model token-identity
   (`docs/ROADB_2026_07.md` §7).

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

11. **Bank-similarity audit (16 banks)** — READY (posed 2026-07-21).
    Run the spread-profile + selection-law probe + margin
    screen on all 16 banks of D2 (we use 4). Prediction from existing
    data: identical lattice/law (spread profile already byte-identical
    across the 4 measured banks; predecoder groups are design constants),
    per-bank margin maps only. Confirms characterization-transfer ⇒ makes
    16-bank scale-out cheap.
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
