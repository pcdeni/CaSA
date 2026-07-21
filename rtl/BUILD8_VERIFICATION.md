# build-8 (ACCUM_XBP) RTL verification — verdict: CORRECT, Verilator-proven

Task: the cross-bit-plane accumulator (`docs/ACCUM_XBP_DESIGN.md`) — fold
the host's place-value sum Σ_b 2^i·pc_b into the readback engine, so a
group's n_bitplanes per-plane readbacks collapse to ONE drained vector.
The step-5 partition move from `METHOD_MVDRAM_LENS.md`; a round-trip cut
at the RTL level (recv wakes ÷ n_bitplanes), composing with SEG_POP.

## Files changed (vs build-7)
- `readback_engine_build8.v`: 4th mode ACCUM_XBP (2'd3, free in the 2-bit
  mode reg); a 128×512b accumulator BRAM (16 signed-i32 lanes/word); the
  RMW pipeline (read acc word at rd_valid, add `±pc_out_l4<<shift` one
  cycle later when the beat's popcounts are valid, write back — consecutive
  beats hit consecutive words so no same-address hazard); per-program word
  realignment (reset word index on `read_seq_incoming`); clear-on-mode-
  entry (128-cycle zero sweep — acc BRAM has no reset); the drain FSM
  (2-phase read→push+zero, 128 words on FLUSH_ACC); buffer_space
  conservation (SEG_POP pattern); trailer magic 0xDBC0DE06.
- `frontend_build8.v`: three control words — SET ACCUM_XBP (bit
  INSTR_WIDTH+8), SET_ACC_WEIGHT (+9, payload {neg,shift} in tdata[3:0]),
  FLUSH_ACC (+10). Exact mirror of the proven +5/+6/+7 SET pattern.
- `softmc_core_build8.v`: wire the four new engine ports through.

## Static review findings
- **Control-word bits free & in range**: frontend uses +1..+7; +8/+9/+10
  are next. h2c_tdata is 256b, INSTR_WIDTH=64 ⇒ bits 72/73/74 valid. The
  weight payload occupies tdata[3:0], which are instruction bits only when
  the word is NOT a control word (set_acc_weight low), so no aliasing.
- **BRAM port discipline**: one sync read port (RMW and drain never read
  simultaneously — drain runs only in a no-read flush window) and one
  write port with a documented exclusive priority (clear ▷ RMW-writeback
  ▷ drain-zero; the three are mutually exclusive by protocol).
- **Weight width**: pc∈[0,32] (6b) << shift≤7 = ≤4096; accumulated over
  ~8 planes × 128 reads × 32 ≈ 32K — comfortably inside int32.
- **Mode-entry cost**: entering ACCUM_XBP zeroes 128 words over 128
  cycles; the host must idle ≥128 cycles before the first read (a
  once-per-mode cost, negligible vs a projection). Documented constraint.

## Simulation (box Verilator 4.028, POPCOUNT_ACCUM_MODE, extended TB)
New scenario (l) ACCUM_XBP + the full build-6/7 suite, `-CFLAGS -DTB_BUILD8`:

| check | result |
|---|---|
| accxbp one tlast message | PASS |
| accxbp trailer magic (0xDBC0DE06) | PASS |
| accxbp data beat count == 256 (128 words × 2) | PASS |
| **accxbp lane == Σ_planes w·pc (4 planes, incl. −2³ top)** | **128/128 exact** |
| accxbp buffer_space conserved (1024→1024) | PASS |
| accxbp 2nd back-to-back program, no desync | PASS |
| accxbp 2nd drain: accumulator zeroed by 1st flush | PASS |
| accxbp→READ transition clean | PASS |

**Non-regression, proven by failure-set diff**: build-7 and build-8 run
the *identical* TB and produce the *same* baseline failure set (the
inverted "B3 BUG" checks that only pass on the broken engine); build-8
introduces **zero** new failures on the shared scenarios a–j + SEG_POP.

## A bug worth recording (found and fixed in sim)
The drain's `proc_flush` initially rose when the drain was merely *armed*,
before any word was in the FIFO — so the FWFT read path surfaced 2 words
of stale `mem` (a garbage prefix) at the empty→filling boundary. SEG_POP
never hit this because its flush asserts only after data is buffered. Fix:
`proc_flush` rises exactly on the first pushed word (`acc_dword_valid`),
and the existing proc_flush_r hold carries it to tlast. This is a real
hardware-relevant ordering constraint, not just a sim artifact.

Verdict: **the RTL edits are correct.** Cleared for the Vivado build.
The residual risk (frontend control-word decode) is covered by the static
audit above + the proven +5/+6/+7 precedent; the flash-order hazard is
identical to build-7 (never send the SET words on a pre-build8 image).
