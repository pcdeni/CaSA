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

## E8 (2026-07-23): alternation primitive CLEAN — residual is compute-in-the-loop

E8 (16× interleave: legacy write+segpop-read "MM3D-analog" ↔ per-request
streamed session "V2-analog", incl. stream_start/stop churn; both sides
legacy-re-verified): **clean**. The read/write primitive space is now
EXHAUSTED — framing, order, writes, replication, maintenance,
alternation all clean on silicon. The divergence therefore requires the
MAJ3 COMPUTE in the loop (fused V2 bodies on scratch, streamed cadence,
LOAD-mix context) — invisible to read/write primitives by construction.
Next bisect (server-level, ~65 min): y-dump A/B with PIM_FUSED_COSET=0
under the LOAD mix — splits fused-body×mix from plain-V2×mix; then the
op-#0 odd-index/copy-parity analysis on whichever arm still diverges.

## Fused-off bisect (2026-07-23): fused REFUTED; E9 is the next primitive

- PIM_FUSED_COSET=0 LOAD-mix y-dump A/B: STILL diverges — op #0,
  254/2560 elements, slices [0,1] (one slice-0 element @487 + the same
  slice-1 ODD-index cluster 2049,2051,2053…), all 6510 ops.
- Elimination matrix now: pack4 ✗, call sites ✗ (guard), arming ✗,
  maintenance ✗ (counters), replication ✗ (E7), alternation ✗ (E8),
  fused ✗ (this bisect). Streamed V2 in the LOAD mix still diverges;
  streamed V2 in the all-V2 run was token-identical.
- **The last untested primitive shape: write→ACT→read WITHIN ONE
  program, streamed back-to-back** — the production V2 exec program's
  internal structure (x-writes + doubleACTs + rdRow in one program);
  every E-arm used separate write-programs and read-programs. E9 =
  single program [per-column writes + PRE + segpop rdRow], streamed
  vs legacy, odd-lane-sensitive verification. If E9 reproduces: chase
  digitally in the Verilator e2e (within-program wdata/lane paths);
  if clean: the remaining delta vs the clean all-V2 run is the
  RESIDENT-ROWS-PHYSICALLY-PRESENT context (charge environment), i.e.
  physics-class — silicon-only territory.
- Also decode pending: the ODD-index signature (odd segments ↔ segpop
  lane pairing / 256b half-swap structure) — check against the E9
  read's lane map.

## E9 (2026-07-23): CLEAN — digital primitive space exhausted

E9 (write→PRE→segpop-read WITHIN one program — the production exec
shape — streamed back-to-back from identical DRAM state vs legacy):
**0/16 programs differ, zero bytes, both parities.** Swap-path program
starts (1..15) identical to the IDLE-path start (0).

Full primitive/bisect matrix, all silicon: framing ✓ order ✓ writes ✓
pure-segpop ✓ mixed ✓ replication ✓ alternation+churn ✓
within-program write→read ✓ — all EXACT under streaming; pack4/fused/
arming/maintenance/call-sites refuted as triggers. Yet the LOAD-mix
full model diverges from op #0 (slice-1 odd-index cluster) while the
all-V2 full model is token-identical.

Remaining candidates, in test order:
1. **E10 (designed, last digital-ish shape):** E8's alternation but
   with the MM3D-entry REFRESH ACT-train (~640-row charge-restore
   program) as the legacy interlude — the one production program class
   never present in any primitive; it runs between streamed sessions
   only in the LOAD mix (MM3D requests exist only there).
2. **Resident-charge environment (physics-class):** the mere physical
   presence of LOAD-resident rows changing the margin context of
   streamed V2 MAJ3 compute — silicon-only territory; would be probed
   via the odd-index signature decode (odd segments vs segpop lane /
   half-swap structure) + a residents-present-but-unused control run
   (LOAD handles uploaded, client forced all-V2: PIM_USE_LOAD_WEIGHTS=1
   with MM3D disabled client-side if a flag exists — separates
   "residents present" from "MM3D requests interleaved").

## E10 (2026-07-23): refresh-train alternation CLEAN — digital space closed

E10 (E8's alternation with the PRODUCTION MM3D-entry refresh ACT-train
— a hardware-looped label/branch program over the full 640-row window —
as the legacy interlude, ×12): **clean** (0 rows, 0 bytes, recv clean).
Every production program class is now silicon-proven exact under and
alongside streaming: straight-line writes/reads, mixed sized sessions,
within-program write→read, and looped refresh trains.

RUNNING: the residents-present-but-unused control (PIM_LOAD_NO_USE=1 —
client one-liner: uploads happen, MM3D never used, all slices V2):
- control DIVERGES ⇒ "residents physically present" is the ingredient
  (charge-environment class; next = odd-index lane decode + charge
  probes; also re-examine the LOAD phase under armed streaming).
- control CLEAN ⇒ divergence requires MM3D actually USED — real
  MAJ3-on-resident programs between sessions (compute×compute) — next
  = E11 with genuine MAJ3 bodies as interludes / server forced-session
  bisects.

## CONTROL RESULT (2026-07-23): residents-PRESENT-but-UNUSED DIVERGES — class = resident-charge-environment

PIM_LOAD_NO_USE=1 (residents uploaded, MM3D never used, ALL slices V2):
**DIVERGES 2536/2560 at op #0, both slices, all indices.** Contrast:
- all-V2, NO LOAD uploads (the full-model gate): TOKEN-IDENTICAL.
- all-V2 + LOAD uploads present-but-unused (this control): near-total
  divergence.
The ONLY delta between them is that the LOAD upload programs RAN
(resident rows physically written into the pool). MM3D use is NOT
required. VERDICT: the divergence is triggered by the PHYSICAL PRESENCE
of freshly-written resident rows, acting on streamed V2 compute — a
charge/analog-margin class effect, NOT a digital datapath bug (10
primitive arms + every program class are silicon-clean under
streaming). Consistent with the whole theme: streaming removes the
inter-program idle that let charge settle; residents raise the stakes.
(Slice-0 now diverges too because PIM_LOAD_NO_USE forces it to V2; the
earlier odd-only slice-1 pattern was the subset where slice-0 still ran
resident MM3D.)

NEXT: E11 — reproduce as a self-contained primitive (pre-charge a big
block of pool rows like LOAD does, THEN the E4-style mixed streamed
V2-analog on other rows vs legacy). E4/E6 were clean WITHOUT the
pre-charge, so pre-charge is the single added variable. If E11
reproduces → seconds-scale silicon repro of the production effect,
class nailed; the fix menu (settle interlude when residents present /
keep streaming only for no-resident configs / M3 where residents
become the operands) is then an engineering choice, not more hunting.

## E11 CLEAN — reframe: the untested cell is streamed MAJ3 COMPUTE per-op

E11 (pre-charge 256 pool rows LOAD-style, THEN E4-style mixed streamed
write+read on base rows vs legacy): **clean 0/16**. Simple spatial
charge-coupling from charged neighbors is REFUTED. This exposes the real
gap: EVERY E-arm (and E9/E10) tests write→read FRAMING; NONE runs the
actual MAJ3 doubleACT/frac COMPUTE under streaming. So the reachable
suspects narrow to a genuine untested cell:
- streamed V2 MAJ3 compute exactness has only ever been checked at
  TOKEN granularity (the all-V2 gate), never PER-OP.
RUNNING (decisive): all-V2, NO LOAD, PIM_YDUMP legacy-vs-stream per-op.
- DIVERGES ⇒ streamed V2 MAJ3 compute is intrinsically (slightly)
  non-exact, token-masked in the all-V2 gate; residents merely amplify
  it into token flips. Fix = per-op, not residency-specific.
- EXACT ⇒ residents are truly necessary → the interaction is
  residents-present × streamed-MAJ3 specifically (pool-cursor row
  selection or a resident-write side effect on the compute path), and
  the next probe recreates THAT exact pair.

## 2026-07-23 (cont): mislabeled cell + THE STRUCTURAL FIND — odd-byte-only corruption

**Correction first**: the "decisive all-V2 NO-LOAD" run above was
MISLABELED. The runner omitted PIM_USE_LOAD_WEIGHTS and the client's
`setdefault('PIM_USE_LOAD_WEIGHTS','1')` silently re-enabled the LOAD
path — yd_v2_*.log shows the same "LOAD pool exhausted / LOAD→V2
fallback" arc as the fus run. That cell was a LOAD-mix REPLICA (which
its op-0 set confirms: 249/256 index overlap with fus). The true
no-LOAD cell has still never been y-dumped; it is RUN B below.

**The structural find** (offline analysis of all four dump pairs,
`ydiff_clean.py`): restricting to hardware-clean comparisons — the 72
layer-0 prefill ops (q=0..23, k=24..47, v=48..71), whose inputs are
prompt embeddings and therefore identical across arms in every run —
the streamed-arm error is NOT noise:

- **Only ODD-indexed y elements ever diverge; even elements are exact**
  (union odd = 1280/1280 for q, 319/319 k, 320/320 v; in v2/fus/nf).
- y[j] maps 1:1 to BYTE j of the 2048-B SEG_POP row image
  (`row_pc: out[s]=row[s]`, then `y[j] += weight*pc[j]`), so this is:
  **odd bytes of every SEG_POP row image wrong, even bytes exact** —
  i.e. the popcounts of odd 32-bit segments (odd lanes of the RTL
  packer, `seg g = beat*16 + lane`).
- Phase-locked onset: the FIRST streamed request of the process is
  exact (op-0 slice-0); everything after — across per-request session
  close/open — is corrupt. Persistent state, same CLASS as the
  build-10 wide_reg bug, different register/path.
- Deltas are small and varied (neighboring-count-like), same-op index
  sets are ~98% identical across fused/non-fused configs, and
  ~unchanged counts (254/251/254) — deterministic digital artifact,
  NOT analog margin. The earlier "resident-charge-environment" verdict
  is WITHDRAWN: residents amplify (ctl: 2536/2560, both parities —
  all-slices-V2 shape), they do not trigger.
- Why every E-arm missed it: E-arms verified streamed SEG_POP reads of
  test-authored programs; the corruption engages only in the
  production request flow (first request clean, later ones dirty) —
  the E binaries re-launch per arm and never got past "first requests".

Elimination matrix update: analog/charge class OUT; digital persistent
state in the SEG_POP count path (RTL) vs raw-data/transport — split by
RUN A.

RUNNING (chained, ydump_ab_runner.sh):
- RUN A: PIM_SEGPOP=0 both arms (raw 8 KB rows, host popcount),
  LOAD-mix config. CLEAN ⇒ bug lives in the SEG_POP count/pack path
  under streaming (→ box TB scenario, build-11 fix). DIVERGES ⇒ the
  row DATA or generic transport is wrong (parity structure of the raw
  dumps then localizes it).
- RUN B: PIM_SEGPOP=1, PIM_USE_LOAD_WEIGHTS=0 explicit — the REAL
  no-LOAD cell, for the record and for the residents-amplifier decode.

### Interim corrections + prepared instruments (2026-07-23, card busy on RUN A/B)

Two corrections to the addendum above, from code reading:
1. "First streamed request exact / phase-locked onset" — WITHDRAWN as an
   inference. Op-0 slice-0 was clean because it ran resident-MM3D
   (LEGACY path — MM3D never streams in phase 1), not because early
   session state is clean. The remaining op-0 anomaly (later q ops
   diverge in slice 0 too, op 0 does not) is plausibly position-0
   x-sparsity (zero-plane skips suppress the V2 sub-handle programs) —
   RUN B (all-V2 everywhere) will show op 0 directly.
2. The LI→LDWD read-after-write hazard theory (streamed paired fetch)
   is REFUTED by inspection of arm E: it already IS the production
   per-column write shape (LI+LDWD per slot per column, 43/43/42,
   distinct data, sized session, per-round recv) and it is CLEAN on
   build-10. The write class is exonerated at primitive level.

What has NEVER streamed at primitive level = the EXEC class:
(a) multi-bank M-row readout (one program, BAR reloads mid-program —
    D-arm was single-bank), (b) branch-looped wrRow bodies (all E-arm
    writes were unrolled; production compute bodies loop), (c) the
    coset doubleACTs themselves.

Prepared while the card is busy (all compile-clean, none yet run):
- Server PIM_STREAM_SCOPE=wcol|exec (DIAGNOSTIC): per-burst sessions
  stream only one program class on the REAL flow; payload-0 identifies
  the wcol class; legacy execute never runs inside an open session
  (guard preserved); MM3D/LOAD handlers untouched (no session object).
  Scope y-dumps split the guilty class wholesale.
- stream-hw E13: multi-bank interleaved segpop readout streamed ×32
  sends across 8 sessions vs identical-program legacy oracle; odd-byte
  tally built in.
- stream-hw E14: branch-looped write body (LDWD prologue + 128-iter
  WRITE loop, per-iteration slot-0 rotation makes columns distinct) —
  streamed, RAW legacy verify; catches loop/fetch corruption and
  in-loop LDWD staleness.

Run order when the card frees: RUN A/B verdicts → E13/E14 (fast) →
scope y-dumps as needed. Decision tree:
- RUN A clean + E13 or E14 reproduces → mechanism found at primitive
  level; fix follows the arm.
- RUN A clean + E13/E14 clean → scope runs isolate the class on the
  real flow (suspect (c) coset doubleACTs or a composition effect).
- RUN A diverges (odd-32b-segment in raw y) → data-level corruption;
  raw row images localize which rows/segments; E14 verdict then
  separates write-content vs compute.
