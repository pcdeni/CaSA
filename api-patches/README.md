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
