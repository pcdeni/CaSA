# rtl/ — the DIFF-accum readback engine (Road B) and its validation harness

The FPGA-side half of Road B: `readback_engine.v` for the
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
with zero sentinel mismatches** in the lane2 GeMV integration.
Data: `docs/data/roadb/`.

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

## Build 7 — SEG_POP readback mode (2026-07-21, silicon-validated)

Third readback mode alongside READ and DIFF-accum: **per-32-bit-segment
popcount readout**. Each read beat's sixteen 6-bit segment popcounts
(tapped at `pc_out_l4`, a level already present in the popcount tree)
are packed one-per-byte; four beats form one 512-bit FIFO word, so a
full row drains as **2048 B of popcount bytes instead of the 8 KB raw
row** — a 4× c2h collapse that also deletes the host's per-segment
popcount pass. Vertical layouts keep working unchanged; READ and
DIFF-accum are bit-identical to build 6.

- Files: `readback_engine_build7.v`, `frontend_build7.v` (SET word,
  control byte `0x80` — idempotent, same pattern as READ/DIFF SETs),
  `softmc_core_build7.v` (wiring). Trailer magic `0xDBC0DE05`.
- `buffer_space` conservation follows build 6's DIFF fix: +2 credit per
  consumed SEG_POP beat, no credit on SEG_POP c2h (the build-6 READ-mode
  gate on the c2h credit makes this compose automatically).
- Constraint: read counts must be a multiple of 4 beats (full-row reads
  always are). Legacy toggle hardened to a 2-state READ↔DIFF flip.
- Verification: `BUILD7_VERIFICATION.md` — Verilator failure-set diff vs
  build 6 (zero new failures), SEG_POP scenario 128/128 byte-exact,
  buffer_space conserved, plus an independent line-by-line re-review.
- Silicon (`app/test_segpop_hw.cpp`): 3 pattern cases × 2048/2048
  segment-bytes EXACT, READ toggle-back clean, build-6 suite 9/9 on the
  same image. Timing: WNS +0.118 ns, 0 failing / 285,499 endpoints.
- Production consumer: `app/test_bitnet_server.cpp` `PIM_SEGPOP=1`
  (matvec reads in SEG_POP; raw-byte verify paths auto-switch to READ).
- **Flash-order hazard**: on a pre-build7 image the `0x80` word falls
  through frontend decode into instruction-load — flash first, then run.

## Build 8 — ACCUM_XBP cross-bit-plane accumulator (2026-07-22, Verilator-proven)

Fourth readback mode: an in-fabric **place-value accumulator** that folds
the host's Σ_b 2^i·popcount_b across bit-plane programs. A 128×512-bit
accumulator (16 signed-int32 lanes/word) reads-modify-writes on each
consumed beat — `acc[word].lane += (±pc_out_l4[lane]) << shift` — where
the weight (sign + power-of-two shift = the bit-plane factor) is latched
out-of-band per program. FLUSH_ACC drains the 128 words as one message
and zeroes the accumulator. A group's n_bitplanes per-plane readbacks
become **one** drained vector — a round-trip cut (recv wakes ÷
n_bitplanes) the same shape SEG_POP was for bytes, and the step-5
partition move of the MVDRAM design method.

- Files: `readback_engine_build8.v`, `frontend_build8.v` (control words
  +8 SET mode / +9 SET_ACC_WEIGHT payload `{neg,shift}` / +10 FLUSH_ACC),
  `softmc_core_build8.v`. Trailer magic `0xDBC0DE06`.
- Mode-entry cost: a 128-cycle accumulator zero sweep (the BRAM has no
  reset); the host idles ≥128 cycles before the first read.
- Verification (`BUILD8_VERIFICATION.md`): failure-set diff vs build 7
  (zero new failures on shared scenarios), the ACCUM_XBP scenario
  **128/128 lane-exact** (Σ over 4 planes incl. the −2³ top plane),
  buffer_space conserved, back-to-back no-desync, accumulator zeroed
  between drains, clean READ transition. A real FWFT ordering bug
  (proc_flush must rise on the first pushed word, not on arming) was
  found and fixed in sim.
- Composes with SEG_POP (per-segment popcount path unchanged) and READ/
  DIFF (bit-identical). Flash-order hazard identical to build 7.

## build-8b (2026-07-22)

The 8a image validated everything except the accumulate word index:
`read_seq_incoming` is a level/multi-pulse overlapping the returning
beats, and the level-priority realign starved the per-beat increment
(acc word0 = Σ beats 0..126, word1 = beat 127 — dump-proven,
deterministic). 8b realigns on announcement-edge + quiet read path
(`rd_outstanding == 0 && ~rd_valid`); trailer magic bumps to
`0xDBC0DE07`. The TB gains scenario (l2) — silicon-faithful overlapped
announcements + tail bubble — which reproduces the 8a failure on the
pre-fix RTL (3/128 lanes) and passes 128/128 on the fix. Silicon:
`accxbp-hw-exe` EXACT (0/2048, both passes) on both dies.
