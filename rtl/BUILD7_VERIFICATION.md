# build-7 (SEG_POP) RTL verification — verdict: CORRECT, Verilator-proven

Task: SEG_POP readback mode (per-32b-segment popcount → 2048 B/row, 4×
collapse, keeps the vertical layout).
This doc is the correctness review + simulation gate BEFORE any bitstream.

## Files changed (vs build-6, authored against the box's authoritative source)
- `readback_engine_build7.v`: mode reg 1→2 bit (READ/DIFF/SEG_POP);
  `set_mode_segpop` input; SEG_POP datapath (tap `pc_out_l4[15:0]` — the
  per-32b-segment popcounts already in the tree — pack 4 read-beats ×16
  bytes into one 512b FIFO word); 3-way FIFO wr_en/din mux; buffer_space
  conservation (credit +2 per consumed SEG_POP beat, mirroring build-6's
  DIFF fix — the c2h-granularity leak the design doc had wrongly called
  absent); legacy toggle hardened to a 2-state READ↔DIFF flip; trailer
  magic 0xDBC0DE04→**0xDBC0DE05** (image fingerprint).
- `frontend_build7.v`: `rbe_set_segpop` output, decode of control bit
  INSTR_WIDTH+7 (byte[8]=0x80) — exact mirror of the proven +5/+6 SET.
- `softmc_core_build7.v`: wire `rbe_set_segpop` → engine `set_mode_segpop`.

## Static review findings
- Mode width change clean: every `mode_r` use is an explicit `==`/`!=`
  (no stale `~mode_r` bool), verified by grep.
- Byte order: SEG_POP applies the *same* 256b half-swap as READ, so the
  fixed c2h path de-swap yields natural order `byte g = popcount(seg g)`.
- DIFF accum/capture machinery stays dormant in SEG_POP (all its gates
  are `mode==DIFF_MODE`), so framing = READ framing.
- **Documented constraint**: SEG_POP requires read counts that are a
  multiple of 4 beats (a partial final group is not emitted). Full-row
  (128-beat) reads always satisfy this; the server/tool must not issue
  non-4-multiple SEG_POP reads.

## Simulation (box Verilator 4.028, POPCOUNT_ACCUM_MODE, extended TB)
New scenario (k) SEG_POP + the full build-6 suite (a–j), `-CFLAGS -DTB_BUILD7`:

| check | result |
|---|---|
| segpop one tlast message | PASS |
| segpop trailer magic (0xDBC0DE05) | PASS |
| segpop data beat count (8 beats→2 words→4 c2h beats) | PASS |
| **segpop byte[g]==popcount(segment g)** | **128/128 exact** |
| segpop buffer_space conserved (1024→1024) | PASS |
| segpop 2nd back-to-back program, no desync | PASS |
| segpop→READ transition clean | PASS |

**Non-regression, proven by failure-set diff**: build-6 and build-7 run
through the *identical* TB and produce the *same 7 failures* — every one
an inverted `"B3 BUG"`/`"LEAK DOCUMENTED"` baseline check that only passes
on the broken build-3/5 engine and correctly fails on any fixed engine.
build-7 introduces **zero** new failures; scenarios a–e (READ/DIFF
correctness + trailer framing + conservation) pass identically to
build-6. 45 PASS total on build-7.

Verdict: **the RTL edits are correct.** The residual risk flagged in
review (trailer/framing desync under the inherited READ path) is closed
by the no-desync + clean-transition checks. Cleared for the Vivado build.

## Independent re-review (Fable, 2026-07-21 PM) — verdict CONFIRMED

Line-by-line re-derivation from the diffs, not from this doc:
- buffer_space conservation re-proven from code: debit 2/announced read;
  +2 per consumed SEG_POP beat; the build6 c2h credit at the `+12'd1`
  line is gated `mode_r == READ_MODE` (not `!= DIFF_MODE`), so SEG_POP
  c2h beats are automatically uncredited — exact net zero. Had build6
  written `!= DIFF_MODE` there, build7 would inflate ~2 per 4 beats.
- seg_beat_valid timing: derived by the identical registered expression
  as silicon-proven diff_valid; pc tree l1..l4 combinational from
  read_diff (leaves eat 4-bit slices ⇒ l4 = 16 × 32b segments, 6b wide).
- Packer: registered seg_word/seg_word_valid pair (no comb race);
  LSB-beat-first ⇒ natural byte g = popcount(seg g); same half-swap
  idiom as READ ⇒ inherits READ's silicon-proven end-to-end ordering
  regardless of sim-FIFO fidelity.
- Control bit audit (NOT covered by the engine-level TB): frontend chain
  +1 toggle, +2 dll, +3 aref, +4 HBM temp, +5 SET-READ 0x20, +6 SET-DIFF
  0x40 ⇒ **+7/0x80 was free**; platform byte[8]=0x80 encoding consistent.
- FIFO occupancy: 4096×256b FIFO; SEG_POP writes 32×512b words per
  128-read program vs the platform's mandatory concurrent c2h drain —
  no overflow under the protocol (credit-on-consume means buffer_space
  no longer tracks FIFO occupancy in SEG_POP, same as DIFF; bounded by
  per-program receive cadence).
- TB nit (cosmetic): seg_pattern's "scramble" guard is computed then
  discarded (dead var); the ramp still distinguishes value-vs-popcount
  for nb≥6, and the silicon tool's pseudorandom case covers the rest.

**Flash-order hazard (procedural)**: on any pre-build7 image the 0x80
word falls through the frontend decode into instruction-load (clobbers
IMEM word 0, arms EXECUTE on tlast) — run `segpop-hw-exe` ONLY after the
build7 JTAG flash is confirmed (magic 0xDBC0DE05).

## BUILD SUCCEEDED (2026-07-21 16:21 UTC) — flashed same day

- Artifact: BCU1525_QUAD project `impl_1/bcu1525_quad_top.bit`
  (standard DRAM-Bender BCU1525_QUAD flow with the build7 engine +
  frontend + core dropped into `verilog/` — same single-variable recipe
  as builds 4–6, see `rtl/README.md`).
- 34,163,343 bytes, **md5 `c63beb71a946f3d2928339ce001f6984`**
- Timing CLOSED: **WNS +0.118 ns, 0 failing / 285,499 endpoints**
  (hold WHS +0.010, 0 failing) — build5 +0.104, build6 +0.064.
- 0 `ERROR:` lines in the full Vivado log.

## Post-flash plan (user does the JTAG step)

1. JTAG program; then per RUNBOOK: remove+rescan or WARM reboot — never
   pci-reset post-JTAG, **never cold-cycle** (erases the image).
2. `fpga-helper load` + full_reset on ALL FOUR channels + RowClone smoke.
3. `segpop-hw-exe 2 0 60000` — expect 3/3 pattern cases EXACT (2048
   segment-bytes each) + READ toggle-back 0 wrong words.
4. Build-6 non-regression on the new image: `popcount-hw-exe 2 0 60000`
   (9/9 + toggle-back) — DIFF/READ paths must be bit-identical to build6.
5. Then the production SEG_POP server integration (readback collapse
   ~5.3×, projected ~25.6 s/tok BitNet).
