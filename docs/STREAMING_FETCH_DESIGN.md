# Rung 1: streaming program fetch — "controller-native on our card"

The convergence point for the round-trip lever family
(`CONTROLLER_NATIVE.md` Rung 1). Request batching, cross-round packing,
and the cross-bit-plane accumulator all *reduce* host↔DRAM round-trips;
this eliminates *waiting* on them, by making the DDR command bus the only
clock — MVDRAM's §V-E execution regime on the hardware we already own.

## The serialization point (from the RTL)

`sources/hdl/verilog/fetch_stage.v` fetches IMEM linearly until `is_end`,
then resets `pc` to 0 and idles. The next program cannot start until the
host has streamed it into IMEM over h2c — so today every program pays a
full host round-trip of *fetch-idle* latency between execute bursts. The
per-program profile's ~150–200 µs "gap" between the DDR bursts is exactly
this idle. `buffer_space` back-pressure (build-6) already gates fetch
against the c2h drain; what's missing is overlap of *the next program's
load* with *this program's execution*.

## Design: ping-pong IMEM + prefetch

1. **Two IMEM banks** (the 8K-deep instr_blk_mem already exists; instance
   a second, or split the existing depth 8K → 2×4K). A 1-bit `active`
   selects which bank `fetch_stage` reads and which the h2c loader writes.
2. **Load-into-idle**: the h2c instruction-load path (frontend
   `imem_wr_data`) targets the *inactive* bank. The host may stream
   program N+1 while `fetch_stage` executes N from the active bank.
3. **Swap on END**: `fetch_stage`'s `is_end` (which today zeroes `pc`)
   instead flips `active` and restarts `pc=0` on the freshly-loaded bank —
   *if* it is marked ready (a per-bank `loaded` flag set on the h2c
   tlast of a full program, cleared when fetch starts consuming it). If
   the next bank is not yet ready, fall back to today's behavior (idle
   until load) — so a slow producer degrades to current performance, never
   worse.
4. **Host = pure producer**: the client streams programs continuously at
   PCIe *bandwidth* (≈0.3 % utilized — latency was always the constraint,
   never bytes). c2h drains concurrently as it does now. Steady state:
   the command bus never waits; round-trip latency leaves the critical
   path.

## What already exists vs what's new

- **Exists**: 8K IMEM (`BITSTREAM_IMEM=8192`), `buffer_space`
  back-pressure (build-6), the async c2h drain thread (platform), the
  idempotent SET-word control plane (build-4/7/8), and — critically —
  `rtl/seq_engine.v`, which reaches 100 % DDR command-bus utilization in
  Verilator (`rtl/SEQ_ENGINE.md`). seq_engine is the *execute-side*
  proof; this is the *fetch-side* complement.
- **New**: the second IMEM bank + `active`/`loaded` handshake in
  `fetch_stage`, and a frontend tweak to route `imem_wr_data` to the
  inactive bank. No change to decode/execute/ddr_pipeline or the readback
  engine — it composes with SEG_POP / ACCUM_XBP unchanged.

## Server/client consequences

- Server: emit programs back-to-back without waiting for each c2h
  completion (the receive thread already decouples this). The per-request
  `handler` loop becomes a producer loop; results are matched to programs
  by the existing in-order c2h stream + sentinel discipline
  (`ROADB_2026_07.md`).
- Client: unchanged wire protocol; the win is entirely in the server↔FPGA
  cadence. Composes with V2GS batching (a batched request already carries
  a whole slice's worth of programs to stream).

## Projected effect and gate

The gap term is ~half of a program's wall today (~150–200 µs of a
~350–400 µs program at K=1). Hiding it under execution approaches the
DDR-bus-bound floor `casa_sched.c` projects — the "last 2–3 orders"
`PAPER_CONTRAST.md` gap 2 attributes to the streaming shape. Realistic
first-cut target: **1.5–2× wall**, stacking multiplicatively with the
ACCUM_XBP recv-wake cut (they attack different terms — fetch-idle vs
receive).

Gate (the build-7/8 discipline): a mixed-stream Verilator A/B
(back-to-back programs, ping-pong vs single-bank) proving identical
result streams and zero desync, then Vivado timing closure, then the
silicon suite. Sequenced AFTER ACCUM_XBP lands (one bitstream change at a
time; each is independently validated and independently useful).
