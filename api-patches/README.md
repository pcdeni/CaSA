# SoftMC API patches (apply to SiMRA-DRAM / DRAM-Bender `sources/api/`)

Unified diffs against the pristine SiMRA-DRAM artifact. Apply from the
DRAM-Bender root with `patch -p1 < 000X-*.patch`. Rebuild everything that
includes the touched headers afterwards (the app Makefiles have no header
dependency tracking — `rm -f *.o` first).

## 0001 — Program finalization must run exactly once  ⚠ upstream-relevant

`Program::get_inst_array()` re-ran finalization on every call:
`linear_analysis()` records SMC_INFO read-burst packets in `warnings`
(never cleared), `insert_generated()` re-inserts all of them each call (the
program grows every execution), and `preprocess_branches()` ORs a shifted
branch target over the previously baked bits — garbage branch destinations
once any insertion lands before a label. Net effect: **executing a
pre-built `Program` object more than once silently corrupts it** — reads
stop streaming (c2h stall), or worse. This presented for weeks as a
"probabilistic XDMA stall" and once wedged our PCIe link hard enough to
need a cold power cycle.

The fix finalizes once (a `finalized` flag) and clears `warnings`; later
calls only re-serialize. Upstream `CMU-SAFARI/DRAM-Bender` carries the
identical code path (`sources/api/prog.cpp`, `get_inst_array` /
`preprocess_branches`) and thus the same latent bug.

## 0002 — Platform: c2h stall guard + robustness fixes

Cumulative delta of `SoftMCPlatform` (platform.{h,cpp}):
- `PIM_RECV_TIMEOUT_MS=<ms>` (opt-in): `receiveData` returns short once no
  NEW c2h data arrives for that long, poisons the platform
  (`recv_stalled()`), refuses further `execute()` (whose thread-join was a
  second unguarded hang point), and detaches instead of joining in the
  destructor — a stalled process exits cleanly instead of hanging until an
  external kill (kills mid-transfer are what wedge PCIe links). Unset env
  keeps the pristine block-forever semantics, bit-identical.
- `PIM_RECV_DEBUG=1`: log every c2h read size + trailer-strip decision.
- `BITSTREAM_IMEM` env gate: refuse programs larger than the deployed
  bitstream's IMEM up front (oversize send = silent FPGA deadlock).
- consume-thread drain architecture + partial-write handling retained from
  our earlier fixes (see comments in the diff).

Two hardening layers added in the 07-20 regeneration:
- `receiveDataTry(buf, size, timeout_ms)`: a bounded, **non-poisoning**
  receive probe — waits up to `timeout_ms`, then returns whatever arrived
  *without* setting the poisoned flag. For mode-/drain-probe logic where
  "no data yet" is an expected, recoverable outcome (used by
  `app/test_popcount_hw.cpp`), as opposed to `receiveData`'s timeout which
  poisons because a real stall there is unrecoverable.
- Send-failure poison: if `sendData` fails (h2c wedge), `execute()` now
  poisons the platform and detaches the drain thread instead of joining it
  forever — the second half of the 2026-07-18 deadlock class (the first
  half being the receive-side hang).

Readback-mode control for the Road-B popcount path (`set_readback_mode` is
**build4-only**):
- `toggle_readback_mode()` — flips the readback engine READ_MODE ↔ DIFF_MODE
  by a *stateful* toggle. Fragile: a lost or duplicated toggle flips parity,
  so on build3 and older images it is paired with the verified-toggle probe
  pattern (`app/test_popcount_hw.cpp`).
- `set_readback_mode(bool diff)` — **build4+ images ONLY** (trailer magic
  `0xDBC0DE02`): idempotent SET words, no toggle parity to lose. On
  pre-build4 images the words are ignored; do NOT call it unless the flashed
  bitstream is build4+.

`0001` is unchanged since 07-17 (`prog.{cpp,h}` did not move); only `0002`
was regenerated against the same pristine base.

## 0003 — accum-mode receiver + DIFF-mode plumbing (Road B host side)

Everything the POPCOUNT_ACCUM readback engine (`rtl/`) needs from the
platform: idempotent `set_readback_mode` SET words (decode-state safe,
unlike the legacy parity toggle), the bounded `consumeDataAccum`
receiver (per-execute windows; tick-paced interruptible kernel reads;
quiet-window exit — `PIM_ACCUM_QUIET_MS`, `PIM_ACCUM_TICK_MS`, with the
post-DIFF transition drain floored at 500 ms), `receiveDataTry` /
`drain_stray` consumers, and poisoned-state fail-fast on h2c errors
(no more SIGABRT mid-stream with a live drain thread). Applies on top
of 0001+0002. Consumption-pattern references:
`app/test_popcount_hw.cpp`, `lane2/lane2_gemv_server.cpp`
(`LANE2_ACCUM=2`), `docs/ROADB_2026_07.md`.

## 0004 — build7 SEG_POP SET word + oversize-skip observability

Two small platform additions, applied on top of 0001+0002+0003:

1. `set_readback_mode_segpop()` — the build7 image's third readback
   mode (trailer magic `0xDBC0DE05`): per-32-bit-segment popcount
   readout, 2048 B/row instead of the 8 KB raw row, READ-style framing.
   Idempotent SET (control byte `0x80`), same decode pattern as 0003's
   READ/DIFF SET words. **Hazard:** on pre-build7 images the word falls
   through the frontend decode chain into the instruction-load path —
   never call it unless the flashed image is build7+.

2. `oversize_skips()` — a counter of programs the platform's IMEM-size
   gate refused (printed + skipped, nothing sent). ⚠ **Integrity note,
   learned the hard way** (`docs/ROADB_2026_07.md`, PLANE_PACK spill
   incident): in READ-mode flows a skipped program hangs the next
   `receiveData` (loud), but in accum-total flows the stream just
   shortens and end-of-stream kicker programs can silently backfill the
   missing totals — deterministic wrong numbers with clean sentinels.
   Any app whose result maps 1:1 onto executed programs must snapshot
   this counter around a batch and refuse the batch's results if it
   advanced. Consumers: `lane2/lane2_gemv_server.cpp` (accum GEMV),
   `app/test_bitnet_server.cpp` (`PIM_SEGPOP`),
   `app/test_segpop_hw.cpp` (silicon validation tool).

## 0005 — build8b ACCUM_XBP control words + flush drain plumbing

Three idempotent SET methods for the build8 cross-bit-plane accumulator
plus the host-side drain plumbing the first silicon round proved
necessary (image trailer `0xDBC0DE07`; the interim 8a image `0xDBC0DE06`
carried a word-index realign bug — full story in
`rtl/BUILD8_VERIFICATION.md`), applied on top of 0001–0004:

- `set_readback_mode_accxbp()` (INSTR_WIDTH+8), `set_acc_weight(neg,
  shift)` (+9, payload in tdata[3:0]), `flush_acc()` (+10) — the same
  decode pattern as the +5/+6/+7 SET words. **Hazard:** never call on a
  pre-build8 image (the words fall through into instruction-load).
- `flush_acc()` spawns its own bounded c2h reader BEFORE sending the
  word — the drain has no `execute()` behind it, so nothing else moves
  it from the kernel ring into the receive queue (first silicon run:
  a clean 0/8192 stall, caught by the 0002 timeout guard).
- `receiver_flush_wait`: the XDMA surface lag can glue a stranded 32-B
  program trailer onto the FRONT of the flush payload (silicon: segment
  0 read back the trailer magic as an int32). The flush receiver strips
  leading `0xDBC0DExx` words — safe because no accum-mode payload word
  can take that value (DIFF/ACCXBP magnitude-bounded, SEG_POP bytes
  6-bit) — and only a payload-bearing read ends the wait.
- ACCXBP-mode accumulate/write programs emit NO c2h: the per-execute
  accum receiver would idle its full quiet window per program, so the
  server integration shrinks `PIM_ACCUM_QUIET_MS`/`PIM_ACCUM_TICK_MS`
  to 8/4 ms when `PIM_ACCUM_XBP=1` (user env overrides win).

Consumers: `app/test_accxbp_hw.cpp` (silicon validation — EXACT 0/2048,
both passes, BOTH dies, 2026-07-22) and `PIM_ACCUM_XBP` in
`app/test_bitnet_server.cpp` (single-track grouped matvecs: per-plane
weight latch, one 8 KB int32 drain per round, SEG_POP fallback for
dual-track / K>1 / fused-repair requests).
