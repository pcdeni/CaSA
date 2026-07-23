# build-9 (Rung-1 streaming fetch) RTL verification — Verilator-PROVEN

The ping-pong IMEM streaming fetch (`docs/STREAMING_FETCH_DESIGN.md`,
`CONTROLLER_NATIVE.md` Rung 1): hide the ~150–200 µs per-program
fetch-idle by loading program N+1 into an idle IMEM bank while N
executes. This is the only readout-side lever left that attacks the
*binding* term — the per-execute round-trip (reconfirmed net-negative by
the ACCUM_XBP A/B, which only collapsed drains).

## The change (frontend.v only; fetch_stage / softmc_core / readback untouched)

- **Ping-pong IMEM pair** (`imem0`/`imem1`, two 8K instances). `exec_bank_r`
  = the bank fetch reads; `load_bank = ~exec_bank_r` is the only write
  target. Per-bank address mux: the exec bank takes `addr_in` (fetch),
  the load bank takes `xfer_ctr` (h2c write) — both in the same cycle.
- **`h2c_tready` extended into EXECUTE** when `stream_en && !loaded[load_bank]`:
  the host streams N+1 into the idle bank during N's execution, XDMA
  back-pressuring the producer exactly as it did in INIT_MEM.
- **`loaded[2]` flags**: set on a streamed program's h2c tlast, cleared
  when the swap consumes that bank.
- **Swap on `frontend_ready`** (softmc_fin + 32) when the next bank is
  loaded — else degrade to the legacy IDLE path (a slow producer gets
  exactly build8 behavior, never worse).
- **`STREAM_EN` control word** (bit INSTR_WIDTH+11, idempotent, payload
  bit0 = on/off), default OFF → byte-identical build8 FSM. SET words
  (+5..+10) and reset also decode mid-EXECUTE so control still works while
  programs stream. Trailer magic → 0xDBC0DE08.

## The bug the gate caught (this is why the gate exists)

First Verilator run: scenarios A/B (legacy) and D (slow-producer degrade)
passed, but C (eager ping-pong) and E (SET mid-stream) failed — the new
program's **word 0 was skipped**, fetch resumed at word 1. The cycle
trace pinned it: at the swap, `exec_bank_r` flips and `fetch_hold`
releases in the *same* cycle, but the per-bank address mux keys on
`exec_bank_r` and the IMEM has 1-cycle read latency — so the first read
after the swap comes from the old bank / wrong address and word 0 is
lost. **Fix**: a `swap_settle` counter holds fetch for 3 more cycles
after the bank flip so the read pipeline settles on the new bank before
fetch re-issues pc=0 (pc stays 0 while fetch is stalled). A pure
same-cycle swap is a real silicon hazard the sim caught before any flash.

## Simulation (box Verilator, real frontend + fetch_stage + pre_decode + behavioral IMEM)

`Vfrontend_fetch_top` drives the XDMA producer and models execute
completion (softmc_fin a fixed delay after fetch_stage's softmc_end),
checking the dispatched instruction stream == the streamed programs, in
order:

| scenario | result |
|---|---|
| A stream-off, single program | PASS |
| B stream-off, three programs (legacy back-to-back) | PASS |
| **C stream-on, eager producer (the ping-pong)** | **PASS** |
| D stream-on, slow producer (degrade to legacy) | PASS |
| E stream-on, SET word between programs | PASS |

**ALL_PASS (0 fails).** Legacy paths are behavior-identical; the eager
ping-pong runs three back-to-back programs across two bank swaps with
zero corruption; the degrade path matches build8; control words survive
mid-stream.

## Gate discipline / scope

- v1 streaming is READ/SEG_POP only (per-program trailers are the
  in-order result delimiters). Accum-family modes keep the
  execute→receive cadence — their capture logic assumes read-quiet
  windows that overlapped programs would destroy. The host contract
  enforces this (server sets STREAM_EN only on the streamable path).
- Lint clean (production warning set). Two full 8K IMEM instances — a
  BRAM bump, negligible on the VU9P.
- Next: Vivado build-9 (running), then user JTAG flash, then a streaming
  silicon tool (back-to-back program stream, results in order) + the
  server producer loop (Rung-0 host side: execute-without-wait + in-order
  result matching + sentinel/oversize discipline). One bitstream change
  at a time; the flash is user-gated on this sim passing — which it now
  does.

## Host streaming API (platform, 2026-07-22) — the Rung-0/producer side

Written and validated pre-silicon (patch-0006 material) so the flash is
the only remaining gate:

- **`set_stream_en(bool)`** — the STREAM_EN control word (INSTR_WIDTH+11 =
  byte[9] bit 3 = 0x08, payload bit0 = on/off). Exact idempotent-SET
  pattern of the +5..+10 words. HAZARD: never on a pre-build9 image.
- **Streaming producer/consumer** (`stream_start(payload_bytes)` /
  `stream_send` / `stream_recv` / `stream_stop`): unlike `execute()`
  (one receiver per program, JOINED before the next send — which
  serializes on the HOST even though the FPGA pipelines), streaming runs
  ONE persistent drain thread (`consumeDataStream`) that reads the
  concatenated c2h stream and splits it into `(payload+32)` records,
  pushing each program's payload into `api_recv_buf` in order.
  `stream_send` only sends h2c (XDMA back-pressures via the frontend's
  tready when the idle bank is full); `stream_recv` pops one program's
  result. READ/SEG_POP only (host contract mirrors the RTL v1 scope).

### The sim caught a framing assumption (valuable)

First `consumeDataStream` stripped the 32-B trailer only on a per-program
short read — which holds on silicon (per-program TLAST) but breaks when
the backend coalesces programs into one kernel read (trailers embedded
mid-chunk → corruption). The in-process sim exposed this immediately.
Fix: a **payload-size-aware record parser** that tracks a byte offset
within the current `(payload+32)` record and survives arbitrary chunk
boundaries — correct under BOTH framings.

### Record-parser unit test (record_parser_test.cpp): ALL_PASS 12/12

The parse loop, exercised in isolation over N=5 concatenated programs,
payloads {8192, 2048} × chunkings {1, 4, 100, payload+32, 8224, 32768}:
extracts exactly the N payloads in order every time; nonzero sentinel
trailer bytes never leak into payload. The parser is correct independent
of kernel read framing.

### Sim limitation (honest)

`PIM_BACKEND=sim` cannot end-to-end validate streaming: `send_program`
runs synchronously and CLEARS the response queue on each send (it assumes
the execute→drain→execute cadence), and models control words as
trailer-emitting programs. A gated `PIM_SIM_STREAM=1` hook suppresses the
per-send clear, but full streaming validation is silicon-only (the
pre-staged `stream-hw-exe` runs the legacy-vs-streaming A/B + wall the
moment build-9 is flashed). RTL is Verilator-proven; host code is
compile- + unit-validated; only the silicon wall number is pending.

## SILICON VALIDATED (2026-07-22 evening, build-9b image)

First flash was the magic-07 staging incident (see RUN_AFTER_FLASH.md
banner); build-9b (readback line 829 = 0xDBC0DE08, bit md5 5261a88b…)
flashed clean:

- Ladder: rowclone PERFECT_CLONE, segpop ALL_PASS (3 EXACT cases),
  popcount ALL_PASS — on b2; rowclone+segpop green on b0. Trailer magic
  **0xDBC0DE08 in-band** (16/16 trailers) on both dies. Stream-off FSM
  is regression-clean, as the first-flash image already proved.
- **Streaming A/B (stream-hw-exe), byte-identical in-order, 0 bad:**
  | die | N | legacy ms/row | stream ms/row | speedup |
  |---|---|---|---|---|
  | b2 | 64 | 0.139 | 0.046 | **3.02×** |
  | b0 | 64 | 0.047 | 0.017 | **2.68×** |
  | b2 | 256 | 0.097 | 0.039 | **2.49×** |
  N=256 = sustained ping-pong across 256 back-to-back programs / >128
  bank swaps, zero corruption. The ~150–200 µs per-program fetch-idle
  is gone from the steady state — the Rung-1 claim, measured.
- Logs: stream_ab_b2.log, stream_ab_b0.log, stream_ab_b2_n256.log,
  ladder9b_*.log (this dir).

Next: the server producer loop (PIM_STREAM in test_bitnet_server.cpp,
READ/SEG_POP-only scope, in-order matching + sentinel/oversize
discipline) — the production wall number.

## Producer-loop gate: full-model FAILED → root cause = build-7 SEG_POP
## packer desync under streaming (2026-07-22 night)

Phase-1 server integration (PIM_STREAM=1, per-request sized sessions):
layer-0 token-identical (62.8 vs 64.8 s legacy — and the first streamed
run exposed a real session-teardown tax, fixed via pthread_kill tick;
also proved the harness) — but the MANDATORY full-model run diverged
('1. The capital of France is the' vs '1. 2. 2.'), silently (zero
platform errors). The layer-0 "pass" was CPU-masked (only layer 0 on
PIM): per-op popcount corruption existed there too.

Primitive isolation (stream-hw arms C–G, silicon, b2):
- D pure-SEGPOP streams: byte-identical to legacy (0/32).
- E/E2/E4/E6 MIXED write+segpop streams: ~31/32 rows garbage —
  independent of recv interleave, row spacing (stride-16), and write
  size (1-chunk = 3-chunk).
- E5 the SAME mixed stream in READ mode: 0/32 — writes, ordering,
  framing all exonerated.
- F single group in a fresh session: clean (short sessions dodge it).
- G [write][read][read][read]: r1=r2=r3 ≈ 1530/2048 bytes wrong —
  STICKY, and the sample shows EVERY 4TH BYTE CORRECT (≈¾ wrong = the
  4-counts-per-32-bit-word packer emitting 3 lanes from a desynced
  phase).

**Verdict: the build-7 SEG_POP output packer's lane counter desyncs
when a zero-beat (write-only) program's trailer passes through it in a
back-to-back streamed session, and stays desynced (no IDLE dwell to
reset it). Legacy cadence masked it via per-program IDLE. No host
workaround exists (sticky ⇒ discard-reads don't help; per-round mode
resets forfeit the streaming win).**

Consequences:
- PIM_STREAM stays default-OFF and is NOT production-safe for SEG_POP
  paths until the RTL fix (build-10: reset the segpop packer lane/state
  at every program start; Verilator TB must reproduce the ¾-lane
  garbage on pre-fix RTL — the 8a→8b discipline verbatim).
- READ-mode streaming remains fully validated (PR #12 claims stand).
- The teardown fix (stream_stop tick-signal) and the sized-record
  parser are correct and stay.
- Open loose end (recorded honestly): E3's clean-then-dirty flip across
  tool revisions — consistent with sticky state inherited from prior
  arms, not yet independently pinned.

## BUILD-10 SILICON VALIDATED (2026-07-23 early): the producer loop is CORRECT

Flash #3 (magic 0xDBC0DE09 in-band 16/16, ladder green, DIFF/popcount
suite ALL_PASS — the wdata mask preserved DIFF semantics on silicon):

- **stream-hw suite ALL_PASS**: every build-9-failing mixed arm now
  0/32 clean — E (interleaved), E2 (no interleave), E4 (stride-16),
  E6 (small writes) — alongside D/E5/E3/A/B/F. Arm G retired (probe
  design artifact; its claim = arm E's comparison, clean 32/32; kept
  behind STREAM_HW_G=1 with a note).
- **Full-model gate: TOKEN_IDENTICAL** on the all-V2 stress config
  (every slice streamed): both arms '1. The capital of France is ',
  stream 1900.2 s vs legacy 1988.1 s (−4.4% — the phase-1 magnitude:
  writes no longer join before execs; each exec still awaits its own
  recv; deeper pipelining is phase 2).
- Production-stack A/B (PIM_USE_LOAD_WEIGHTS=1 + fused + pack4) running
  as of this note; numbers land in the PR #12 closing commit.

The 2026-07-22 arc in one line: producer loop → silent full-model
divergence → 7-arm silicon isolation → wiring-level root cause
(persistent wide_reg vs streamed no-INIT_MEM swaps) → one-line semantic
mask → Verilator repro+fix gate → build-10 → all clean, token-identical.

## LOAD-mix × streaming: OPEN residual, quarantined (2026-07-23 ~03:00)

Production-stack A/B (LOAD+fused): TOKEN_DIVERGED both with pack4
(catastrophic '1.1.1.1.') and without (subtle: 'the' vs 'Paris' tail —
a near-tie token flip). Isolation so far:
- pack4 REFUTED as sole trigger (diverges with it off).
- execute-in-session guard (new, platform-level): ZERO hits — no
  unconverted call sites.
- M1/M2 (ab_fused_server): LOAD+MM3D all_exact under BOTH arms —
  stream ARMING is innocent; sessions are required for the damage.
- mixed_probe.py (LOAD→MM3D→V2→MM3D): INVALID AS BUILT — its synthetic
  V2 requests are inexact in the LEGACY control arm too (40-1349/2048)
  and degrade subsequent MM3D even under legacy; the fused-V2 path
  semantics need the real client's per-slice construction. The
  isolation therefore needs a production-faithful single-op harness
  (client-side y-dump compare, first diverging op) — NEXT SESSION.

Quarantine state: PIM_STREAM stays default-OFF; VALIDATED scope =
READ-mode streams + pure-SEGPOP streams + mixed write/segpop streams
(stream-hw ALL_PASS) + the ALL-V2 full model (token-identical, −4.4%).
NOT-validated scope = PIM_STREAM under PIM_USE_LOAD_WEIGHTS mixes.
The all-V2 shape is not the production default, so production is
unaffected either way until the residual is pinned.

## Isolation ledger addendum (2026-07-23): maintenance hypothesis REFUTED

User lead: DRAM Bender issues maintenance ops between user programs
(manually triggerable via SMC_REF/SMC_ZQ — both exist in instruction.h).
Investigated:
- Mechanism EXISTS in principle: maintenance timers reload while
  program_process is active; ops fire only in quiet gaps.
- Periodic REF: aref-gated (maintenance_controller pr_ref_request needs
  aref_switch_r) — production runs aref OFF ⇒ REF dead in BOTH arms;
  confirmed by cnt_ref_init == 0 in every trailer.
- Periodic reads (tPRDI) + ZQ: MEASURED in-process via trailer counters
  (server's own program trailers, PIM_RECV_DEBUG): legacy arm cnt_rd
  delta 279,121 vs streamed arm 289,115 over comparable probe runs —
  **same rate; NOT starved under streamed sessions**.
- Note for future counter work: reset_fpga() zeroes the counters, so
  cross-process sampling is invalid; measure within one server run.

Suspect list now: production-shaped V2-session traffic interacting with
resident rows / fused paths. Next instrument: first-diverging-op y-dump
via the REAL client (per-op output compare, legacy vs stream, LOAD mix;
classify op #1 by path class). Quarantine unchanged (PIM_STREAM
default-OFF; validated scope explicit).

## FIRST-DIVERGING-OP RESULT (2026-07-23): streamed V2 on REPLICATED slices

y-dump instrument (PIM_YDUMP in pim_linear.py, binary per-op records):
- FIRST diverging op = **#0** (immediate, not cumulative), and ALL 6510
  ops diverge — but op #0 differs ONLY in slice 1 (251/512 partial-slice
  elements, ODD indices in the sample), slice 0 EXACT.
- Decode: slice 1 = the REPLICATED partial slice (512 real outputs ×4
  copies, client vote-aggregates); with ENOSPC at ~20 sub-handles such
  slices run V2 (= streamed sessions), while clean slice 0 is
  LOAD-resident MM3D (legacy path). Values differ by
  plausible-magnitude amounts (e.g. 248→146, −382→−376) — copy-vote
  flips, not garbage.
- CLASSIFICATION: resident/MM3D = clean; **streamed V2 on
  replicated-mask shapes = the diverging class.** Explains why the
  synthetic probe failed its legacy control (production V2 =
  replicated+voted shapes) and why stream-hw arm E (random masks) is
  clean.
- NEXT PROBE (primitive-level repro attempt): stream-hw E-arm variant
  with REPLICATED masks (512-wide pattern ×4 across the row — the
  client's exact replication) streamed vs legacy. If it reproduces,
  the mechanism hunt continues from a seconds-scale silicon repro;
  suspects: pattern-regularity × back-to-back write/exec timing.

## E7 + suspect ranking (2026-07-23, continued)

- E7 (replicated 512-period masks, E-cadence): **CLEAN 0 bytes** —
  replication alone does not reproduce at the primitive level.
- Constraint inventory: fused-V2-streamed = clean at full-model scale
  (the all-V2 gate ran PIM_FUSED_COSET=1, token-identical); replicated
  masks clean (E7); resident MM3D clean (slice-0 exact + ab_fused);
  maintenance refuted; framing/writes/order all proven.
- **Top suspect by elimination: the ALTERNATION** — legacy-MM3D
  requests interleaving with streamed-V2 sessions (per-request
  stream_start/stop + per-program receivers alternating; swap-path vs
  IDLE-path execution alternating) — the one element present ONLY in
  the LOAD-mix and never isolated. Two next probes, either decisive:
  (1) E8 primitive: rapidly alternate legacy exec/recv reads with
  streamed write+read sessions, exactness both sides; (2) y-dump A/B
  with PIM_USE_LOAD_WEIGHTS=1 but sessions forced ALWAYS (stream LOAD
  handler too) or NEVER mid-mix — splitting alternation from
  coexistence. Also worth one look: op #0 slice-1 ODD-index-only
  pattern (copy-vote structure) against the V2 scratch row parity.
