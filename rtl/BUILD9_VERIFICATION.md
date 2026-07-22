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
