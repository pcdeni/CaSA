# `lane2/` — MVDRAM reproduction: the in-DRAM GeMV server (Lane 2)

## The two-lane rule (read first)

This project runs two deliberately **separate** lanes, and this directory is
Lane 2 only:

- **Lane 1 — our BitNet PIM** (`app/`, `python/`, `scheduler/`). Our own
  system: ternary BitNet b1.58 computed in DRAM. It may borrow any idea and
  take "Road B" (rig-specific HDL such as the popcount engine).
- **Lane 2 — the MVDRAM reproduction** (this dir + `docs/MVDRAM_REPRODUCTION.md`,
  `docs/LANE2_GEMV_SERVER.md`). MVDRAM
  (arXiv:2503.23817) reproduced **on its own terms**: their target models via
  llama.cpp, their conventions, and **Road A (in-DRAM) only** — no rig-specific
  HDL shortcuts.

The two lanes are **never blended**. Numbers, tables, and claims from one lane
are never mixed into the other's. Keep it that way.

## Files

| File | What |
|---|---|
| `lane2_gemv_server.cpp` | The in-DRAM GeMV server: `LOAD_MATRIX` / `GEMV` / `GEMV_PARTIALS`, in `host`, `dualtrack`, and `clone` modes. |
| `Makefile` | Builds `lane2-gemv-server` against the prebuilt DSN_AE_APPS/api objects (header-only boost). |
| `lane2_client_smoke.py` | Standalone verifier client (drives LOAD/GEMV, checks vs a host reference). |
| `lane2_partials_fp32.py` | The exact-fp32 partials driver (R4 arm). |
| `b2_gemv_table.py` | The B2 table harness — timing cells across arms, gates + estimate-only mode. |
| `run_b1_silicon_smoke.sh` | The gated silicon smoke runner; documents the safety gates. |

## Wire protocol (magics)

All requests/responses are little-endian u32 words; the `0x4D5630..` prefix is
`"MV0"` (MVDRAM lane). See the header of `lane2_gemv_server.cpp` for the full
body layout.

| Message | Request magic | Response magic |
|---|---|---|
| `LOAD_MATRIX` `{handle,q_bits,K,M,bitplanes}` | `0x4D563001` | `0x4D5630F1` (`{handle,status}`) |
| `GEMV` `{handle,r_bits,activation bitplanes}` → `y[M]` i32 | `0x4D563002` | `0x4D5630F2` (`{handle,status,M,…}`) |
| `GEMV_PARTIALS` (2026-07-20; same body as GEMV, returns partials) | `0x4D563003` | `0x4D5630F3` |

Modes: `host` and `dualtrack` encode activations on the host; `clone` mode
loads by RowClone and ignores the host encode path.

## Building

The Makefile links against the DRAM-Bender api/util objects the DSN_AE_APPS
builds already produce, so it points at that tree by absolute path:

```make
BENDER := /home/deni/Claude/SiMRA-DRAM-main/DRAM-Bender/sources   # <-- EDIT THIS
```

**Edit `BENDER`** to your DRAM-Bender checkout before `make`. If the shared
objects are missing, build them once from the apps tree
(`make -C $(APPS)/BitNet maj3-smoke-exe` produces `util.o` + `api/*.o`).

### Other machine-local paths to adjust

These shipped as the exact working-tree values and must be pointed at your own
environment:

- `run_b1_silicon_smoke.sh`: `BENCH=/home/deni/mvdram_bench`,
  `REPRO=/home/deni/Claude/mvdram-repro`.
- `lane2_partials_fp32.py`: the `--dump` default
  (`/home/deni/mvdram_bench/smoke_2026_07_18/dump_first_tensor.bin`).

## Scope

This is the **phase-1** Lane-2 server: correctness-faithful in-DRAM GeMV with
the LOAD/GEMV/PARTIALS protocol and the host/dualtrack/clone arms, driven
offline by the smoke clients. It is **not** wired into llama.cpp's streaming
execution (that integration lives as the `shim-patches/`), and it does not
claim MVDRAM's streaming-scale throughput. The scope-decision record and the
per-arm results are in [`docs/LANE2_GEMV_SERVER.md`](../docs/LANE2_GEMV_SERVER.md);
the measurement logs are under [`docs/data/lane2/`](../docs/data/lane2/).
