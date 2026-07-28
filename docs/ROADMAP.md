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
   ⚠ **CORRECTION 2026-07-27: streaming alone does NOT invert the
   regime.** With `PIM_STREAM` on vs off, recv (~64% of the request
   wall) is UNCHANGED, because the receive is synchronous — streaming
   the SEND alone is **wall-neutral** (the −4.0% production A/B was
   phase-1 magnitude, not a regime change). Instruction-count levers
   (M3, K-batching) therefore do NOT become top post-streaming levers.
   The recv term only moves when the send is pipelined PAST the recv:
   that is **phase-2 send-ahead**, and it is the real recv attack ↓.
3a. **Phase-2 send-ahead (`PIM_STREAM_PIPE`)** — ✅ **VALIDATED ON
   SILICON 2026-07-27 (build-26, no flash), −26.3% wall.** Full-model
   A/B: pipe-on 1863.1 s vs pipe-off control 2529.1 s; per-request recv
   **112 → 57 ms (halved)**, total 176 → 130 ms; token-exact (`'The'`).
   The `~1/4k` lost-record stall that kept this default-OFF does NOT
   reproduce: **0 stalls / 0 decay / 0 errors over 11,500 requests**
   (would expect ~3 if present) — the RTL blocker is gone. Root cause of
   the stale "waiting on build-11": build-11's fix never reached fabric
   (Synth 8-7071, `fetch_restart` unconnected in `softmc_core.v` through
   builds 11–13); the true fix landed in build-14 (magic `0x0D`) and the
   running build-26 (`0x15`) is later still. UNBLOCKED for production
   enable — flipping the `PIM_STREAM_PIPE` default is a
   production-behavior change and is USER-GATED. See
   `docs/SESSION_2026_07_27.md` §1, `docs/PHASE2_PIPE_2026_07_24.md`.
3a3. **On-fabric orchestrator (Rung-2) — first probe PASSES.** ✅
   **Verilator 27/27 byte-exact 2026-07-27** (`rung2_probe_2026_07_27/`,
   design in `rung2_sequencer_design_2026_07_27.md`): a closed-loop top
   wires the existing `seq_engine` → a deterministic SiMRA DRAM model →
   the existing exact popcount+accumulate datapath, driven by a
   hard-coded sequencer walking one projection's chunk/bitplane loop.
   TB sends `x` once, reads integer partials once, byte-compares to a CPU
   oracle (3 edge + {1,8,64}×8 seeds). Feasibility established: the fabric
   CAN drive a host-command-free projection loop and return exact integer
   partials. Not yet a soft core / ISA / allocator — those remain the
   Rung-2 build. See `docs/SESSION_2026_07_27.md` §4.

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
6. **M3 coset-broadcast operand fan-out** —
   ⚠ **SUPERSEDED 2026-07-27: production-negative.** The server
   `PIM_BCAST_LOAD` integration was measured on silicon and contributes
   nothing — the MAJ3 body recomputes its scratch row, so the deposit is
   never read; the `NODEP` A/B is token-identical; the ceiling is <9% of the
   request wall regardless. Kept at default 0, slated for removal. The gate-1
   numbers below stand only as a *standalone-harness primitive demonstration*
   and do **not** reach the request wall.
   **FIRST GATE PASSED 2026-07-22, both dies** (`docs/M3_COSET_FANOUT_DESIGN.md` §First
   gate; tool `app/test_m3_scratch_ab.cpp`, logs `docs/data/m3/`). One
   coset `doubleACT` loads the scratch row(s) byte-exactly from a
   pool-resident source: 20/20 (k=1) + 20/20 (k=2: 1 op → 3 rows) per
   die, zero leak, all timings, **265.8× fewer instructions / ~4× wall**
   vs the 3-chunk per-column write. Design finding: the legacy pool is
   an independent set over the coupling graph — deposits must target
   the pool's coupled *shadow* rows; allocator pairs resident↔shadow.
   ~~Next: server `PIM_BCAST_LOAD` + shadow allocator~~ — **done 2026-07-26,
   resolved negative** (see the banner above).
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
28. **X-master-clone (activation-side residency + clone)** — ❌
    **CLOSED NEGATIVE for production (2026-07-28, root-caused): the
    RowClone'd x seed sits at charge-shared — not write-driven — levels
    (in-DRAM x′ ≠ x); the corruption is present at ALL request depths
    at production mask density on BOTH tracks. The server now
    hard-gates X-master off on the single track. Default-off
    everywhere.**
    On the fused-coset path the 5 activation `wrRow`s (~1,280 slots each)
    rewrite THE SAME `x` plane every round (MAJ3 charge-sharing destroys
    its operand rows each execution). X-master writes each plane's `x`
    master row once per request (8 planes × banks, outside the tuple),
    then RowClones master→tuple (~40 slots vs ~1,280) per body. Server
    path behind `PIM_XMASTER` (default 0) + twin `PIM_XMASTER_ALTERNATE`;
    adversarial verify confirmed default-off byte-identical and caught +
    fixed a deposit-screening gap for the `PIM_XMASTER`+`PIM_RC_V2`
    pairing. Engages 8/8 masters all 4 banks; production-shape numerically
    correct (d_in=512 corr 0.998). **Wall is request-shape-dependent:**
    single-op d_in=2560 87.8 → 69.8 ms (−20.5%, exec −25%/wcol −15%) but
    small ops (d_in=512, 16 chunks) go net-negative (master-fill overhead
    > savings). Full-model A/B: OFF 1541.0 s → ON **1387.7 s (−9.9%)**,
    token-identical, 0 stalls/decay. **Requires fused-coset**
    (`PIM_FUSED_COSET=1`, server default 0), so the −9.9% only cashes in
    if fused is adopted. Stays default-off pending a user decision.
    `docs/SESSION_2026_07_27.md` §3.
    **2026-07-28 RESOLUTION (session audit → root cause → fix):** the
    −9.9% A/B above was BitNet dual-track V2 at max-tokens **1** (both
    arms '1', the legitimate list-start) — a gate too thin to carry a
    "verified". On the production Bonsai **V2S single-track** path,
    X-master produced degenerate output in every config (K=6/K=8,
    streaming on/off) on the same binary. Root cause (code-trace +
    oracle-proven): the row roles and orderings are CORRECT on both
    tracks — the defect is that the master→body RowClone establishes
    the x seed at **charge-shared, not write-driven, levels** (x′ ≠ x).
    The dual track computes both popcount terms in-DRAM against the
    same x′ — and a controlled depth sweep (07-28, four depths 8–80
    rounds) shows the corruption is **DEPTH-INDEPENDENT** (masters are
    refilled every round, so per-round seed errors never accumulate):
    at production mask density (all 2048 output rows real) dual-track
    corr ≈ **0.94–0.97 at every depth**, 0/2048 bit-exact, vs bit-exact
    without X-master at all depths to 80 rounds. The 07-27
    "corr 0.998 shallow" reading reproduces only at its low-density
    harness config (512 of 2048 rows real) — a density artifact, not a
    depth effect. The single track reconstructs
    `y = 2·pc_pos(x′) − Σx` with a host-exact Σx — the seed error
    doubles, nothing cancels, output decorrelates to garbage. Proven
    with a new V2S mode of the numerics oracle (live-validated with a
    negative control): same binary/shape/seed, XM=0 **bit-exact
    2048/2048**, XM=1 corr 0.951 (4/2048). **Fix shipped:** both
    server X-master gates now include `&& !single`, so V2S keeps the
    exact write-driven wrRow seeds — post-fix V2S is bit-exact with
    `PIM_XMASTER=1` (inert). The −9.9% claim is retracted: at
    production depth X-master fails the ≥0.98 correlation discipline
    on both tracks. Only revivable if the seed can be made
    write-driven-equivalent (charge-sharing physics argues no).

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
12. **Calib-transfer procedure** — DONE 2026-07-22
    (`docs/CALIBRATION_TRANSFER.md`): the recipe + transfer-success
    table. Banks (all 16) and same-model dies transfer byte-identically
    (margin re-screen only); subarrays partial (pool re-derivation, the
    long offset is block-relative); parts need lattice re-derivation.
    16-bank scale-out is now a config exercise.
13. **16-bank / multi-subarray scale-out** — IDEA, after 11. The idle
    spatial parallelism from docs/UTILIZATION.md (die ~99.99% idle).
    Constraints known: tFAW/tRRD scheduling (pack4 machinery), per-bank
    pools, c2h contention.
14. **640/1024 boundary atlas** — IDEA. Map sense-amp-segment vs
    predecoder-block boundaries per bank; structural confirmation of the
    replicated-block hierarchy (the two-granularity finding).
15. **D1/D3 storage roles** — PARKED-ish. MAJ3-limited dies as weight
    parking + RowClone shuttle. Revisit only if capacity binds.
29. **V2 output-numerics gate** — ✅ **BUILT + VALIDATED 2026-07-27.**
    The project had no full-coverage output-numerics gate (the old
    per-projection oracle was deleted; `ab_fused_server.py` fails silently
    on the disabled handle path; token-identity is insensitive — passed at
    87% wrong PIM masks). Two halves built: (a) `mm3d-verify` widened past
    round-0 via `PIM_VERIFY_ROUNDS` (default 1 = old byte-identical
    behaviour; raised = strided across all rounds), rebuilt + proven on
    silicon; (b) `numerics_gate/v2_oracle.py` drives the REAL `MAGIC_V2`
    production path, builds a CPU reference from the same masks/bitplane
    factors, refuses to fail silently (asserts response length, `y≠0`,
    `exec>0`), and carries a working `--inject-fault` negative control —
    `PIM_BACKEND=sim` 8/8 bit-exact, card-free. **KEY finding: the gate
    must be CORRELATION-based, not bit-exact** — raw per-op V2 silicon is
    NOT bit-exact (same op across processes: bit-exact count swung
    28/459/463/283 of 512, per-boot operating point) yet **corr =
    0.997–0.9998 STABLE**; wrong weights/stale masks collapse the
    correlation. Threshold (corr ≥ 0.98) provisional pending a real
    corruption-run calibration. RULE recorded: gate mask/weight/pool
    changes on numerics, never on token identity.
    `docs/SESSION_2026_07_27.md` §2.

## D. Model / application levers

30. **Coarser activation quant (`PIM_ACT_K`)** — ✅ **SHIPPED + SILICON-
    VALIDATED 2026-07-28, across the whole model zoo, token-identical.**
    The activation is decomposed into K bit-planes = K MAJ3 bodies = K
    `platform.execute` round-trips (1/plane at the production
    `PIM_INLINE_BITPLANES=1`). Dropping K cuts the *binding recv wall*
    proportionally, with no accuracy loss down to a model-specific floor
    (which tracks the training recipe, not weight bits). Client-only, NO
    bitstream. Measured full-model A/B (V2 path + phase-2), each vs its own
    K=8: **Bonsai-1bit K=6 −21.7%, Bonsai-ternary K=6 −22.2%, BitNet-2B
    K=5 −32.2%** — all token-identical, numerics gate corr 0.99995. K=4
    (int4) collapses on all Bonsai; BitNet-2B tolerates K=4 (QAT native-A8)
    but K=5 is the safe floor. Production defaults set per model
    (1bit/ternary=6, bitnet=5); `setdefault` so explicit `PIM_ACT_K` wins.
    Composes orthogonally with fused/streaming (acts on the execute COUNT,
    they act on per-execute body time).
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

- 2026-07-27: Phase-2 send-ahead (`PIM_STREAM_PIPE`) VALIDATED on silicon
  (build-26, no flash) — full-model −26.3% (2529.1 → 1863.1 s), recv
  112 → 57 ms, token-exact, 0 stalls/decay/errors over 11,500 requests;
  RTL blocker gone (fix reached fabric in build-14). UNBLOCKED, USER-gated
  to enable. `docs/SESSION_2026_07_27.md` §1. (Correction: streaming ALONE
  is wall-neutral; phase-2 is the recv attack — see lever 3/3a.)
- 2026-07-27: V2 output-numerics gate BUILT (`numerics_gate/v2_oracle.py`
  + `PIM_VERIFY_ROUNDS`) — real `MAGIC_V2` path, sim 8/8 bit-exact, working
  negative control; gate is CORRELATION-based (corr 0.997–0.9998 stable
  where raw per-op bit-exactness is process-dependent).
  `docs/SESSION_2026_07_27.md` §2. (Lever #29.)
- 2026-07-27: X-master-clone IMPLEMENTED (default-off, requires fused) —
  full-model −9.9% (1541.0 → 1387.7 s), token-identical, 0 stalls/decay;
  small-op net-negative, single-op d_in=2560 −20.5%.
  `docs/SESSION_2026_07_27.md` §3. (Lever #28.)
- 2026-07-27: Rung-2a on-fabric orchestrator probe — Verilator 27/27
  byte-exact (fabric-driven projection loop returns exact integer partials,
  no host command). Feasibility of a host-command-free loop established.
  `docs/SESSION_2026_07_27.md` §4.
- 2026-07-21: Road-B lane2 integration (product dataflow 1.4–28×; accum
  crossover ~50K products; 65K totals 0 faults) — `docs/ROADB_2026_07.md`.
- 2026-07-21: 1-bit single-track V2S (18.7 s/tok, 1.81×, ladder 5.36×).
- 2026-07-21: MAJ5 ZERO+2 chain A/B — honest negative (99.51 vs 99.90;
  SiMRA ONE+3 stands).
- 2026-07-21: q3_K Phase D (99.90%) — quant coverage complete.
- 2026-07-21: PACK_ROUNDS triage — MM3D-only, ~3% lever in production;
  the real unlock is broadcast/clone loading (→ lever 6).
