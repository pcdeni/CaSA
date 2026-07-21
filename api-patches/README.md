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
