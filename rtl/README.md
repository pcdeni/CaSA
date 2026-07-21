# rtl/ — the DIFF-accum readback engine (Road B) and its validation harness

The FPGA-side half of Road B (ADR-005): `readback_engine.v` for the
DRAM-Bender BCU1525 QUAD bitstream, extended with **POPCOUNT_ACCUM_MODE**
— in DIFF mode the readback path no longer streams row data; it
accumulates `popcount(rd_data XOR ddr_wdata)` across a whole program and
drains ONE 32-bit total per program (64 B chunk + 32 B trailer, vs 8 KB
per row read). READ mode is bit-identical to stock.

## Files

- `readback_engine_build6.v` — the current engine (trailer magic
  `0xDBC0DE04`). Relative to stock it carries three fixes, each found on
  silicon:
  1. **pop_count4 0xE missing case** — nibble 0xE undercounted.
  2. **FWFT trailer framing** (build 5) — trailer beats mis-framed under
     FIFO fill latency.
  3. **buffer_space conservation** (build 6) — THE 8-programs-then-wedge
     root cause: DIFF-mode beats are consumed by the popcount path but
     the c2h budget only credited actual c2h transfers, leaking
     2·reads−2 units per program (254 at 128 reads); the 2048-unit
     budget starved at program 9 and `need_flush` looped forever. Fix:
     credit consumed DIFF beats, stop crediting DIFF c2h transfers.
     **Stock DRAM-Bender's streaming DIFF mode leaks identically** —
     upstream-relevant.
- `readback_engine_build5.v` — the pre-conservation engine, kept for
  diffing the fix.
- `tb_readback.cpp` + `rdback_fifo_sim.v` / `rdback_fifo_sim_filllatency.v`
  — Verilator harness; the fill-latency FIFO stub is what discriminates
  build-4's framing bug. Scenario `j` reproduces the leak (buffer_space
  1024→8 after 8 programs on ≤build5; conserved on build6).
- `validate_build5.sh` / `validate_build6.sh` — the engine × FIFO
  validation matrices. `build_and_run.sh` — harness build wrapper.

## Silicon state (2026-07-21)

Build 6 flashed and validated on the BCU1525: 9/9 accum suite exact +
toggle-back READ sanity, 60/60-program endurance at the configuration
that starved at 9 on build 5, and **65,000 totals delivered in order
with zero sentinel mismatches** in the lane2 GeMV integration
(`docs/ROADB_2026_07.md`). Data: `docs/data/roadb/`.

## Host-side counterpart

`api-patches/0003-platform-accum-receiver-and-diff-mode.patch` — the
platform additions the accum mode needs: `set_readback_mode` (idempotent
SET words), the bounded `consumeDataAccum` receiver (per-execute windows,
tick-paced kernel reads, quiet-window exit; `PIM_ACCUM_QUIET_MS` /
`PIM_ACCUM_TICK_MS`), `receiveDataTry` / `drain_stray`, and the
post-DIFF transition drain. Consumption pattern reference:
`app/test_popcount_hw.cpp` (suite) and the lane2 server's batched
drain (`lane2/lane2_gemv_server.cpp`, `LANE2_ACCUM=2`).

Timing on the shipped image: WNS +0.064 ns, 0 failing endpoints.
Build recipe: standard DRAM-Bender BCU1525_QUAD project with this
engine dropped into `verilog/` (single-variable flow — only
readback_engine.v differs from stock + the pop_count4 fix).
