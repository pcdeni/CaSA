# `app/` — BitNet PIM applications for DRAM-Bender

C++ apps that issue charge-sharing primitives on real DRAM-Bender
silicon and run the matrix-multiplications of Microsoft's BitNet
b1.58-2B-4T using them.

These files are **drop-in additions** to a DRAM-Bender source tree.
They depend on DRAM-Bender's SoftMC API (`prog.h`, `instruction.h`,
`platform.h`) and the SAFARI `util.h` helpers, neither of which we
re-host.

## Integration

```bash
# 1. Clone DRAM-Bender per the upstream README.
git clone https://github.com/CMU-SAFARI/DRAM-Bender

# 2. Bring up the BCU1525 (or compatible) bitstream and verify the
#    /dev/xdma* devices show up. See docs/HARDWARE.md in this
#    repository for our specific setup.

# 3. Copy these C++ apps into the DRAM-Bender apps tree:
DBROOT=path/to/DRAM-Bender
DEST=$DBROOT/sources/apps/DSN_AE_APPS/BitNet
mkdir -p $DEST
cp *.cpp Makefile $DEST/
cp ../calibration/calib_dimm0.txt $DEST/   # or your own

# 4. Build.
cd $DEST
make
```

The Makefile expects to be in
`DRAM-Bender/sources/apps/DSN_AE_APPS/BitNet/` because it references
`../util.cpp` and `../../../api/*.cpp` relative to that location. If
you want to build the apps elsewhere, adapt the SHARED_SRCS /
INCLUDE_DIRS lines at the top of the Makefile to point at the right
DRAM-Bender paths.

## Binaries built by the Makefile

| Binary | Purpose |
|---|---|
| `maj3-smoke-exe` | Sanity-check `doubleACT(0,0)` MAJ3 on a calibrated tuple. |
| `matvec-smoke-exe` | Single-MAJ3 partial dot product. |
| `columnwise-smoke-exe` | Per-column non-uniform writes work as expected. |
| `dense-matvec-smoke-exe` | Dense matrix-vector at full row width. |
| `dense-matvec-bcast-exe` | Same with `doubleACT(10,2)` broadcast. |
| `broadcast-verify-exe` | Verify broadcast pattern preservation. |
| `multibank-bcast-exe` | Path C: 4 banks per execute. |
| `bitnet-e2e-v{1,2,3}-exe` | End-to-end BitNet matmul harness, three iterations. |
| `bitnet-real-exe` | Ternary × int8 matmul against a PyTorch reference. |
| `bitnet-proj-exe` | Per-call subprocess runner (one matmul per process). |
| `bitnet-proj-server` | **Long-running PIM server** — accepts matmul requests over stdin/stdout. The Python orchestrator talks to this. |
| `rowclone-smoke-exe` | RowClone reliability probe (sweeps `t_23 ∈ {1..4}`). |
| `persistent-smoke-exe` | End-to-end persistent-weight MAJ3 vs direct per-col write. |

## Notable files

- **`test_bitnet_server.cpp`** — the PIM server used by the Python
  orchestrator. Uses persistent weights (per-col write the mask once
  to a backup row, RowClone-refresh on every MAJ3) and multi-bank
  parallelism (Path C). Accepts `--bank "0,1,2,3"` or `--bank "1"`.
- **`test_persistent_weights_smoke.cpp`** — end-to-end check that
  persistent weights produce bit-exact output vs direct per-col write.
- **`test_rowclone_smoke.cpp`** — verify RowClone is reliable on a
  given (backup, target) row pair, sweeping the short
  charge-sharing delay `t_23`.
- **`test_multibank_bcast.cpp`** — Path C standalone: 4 banks, 4
  MAJ3s per `platform.execute()`, byte-exact verification.
- **`Makefile`** — single source of truth for the binary list.

## Runtime environment flags (production server)

The server reads these once at startup (`init_debug_flags`). Defaults
preserve older behavior; each flag names the doc that measured it.

| flag | default | effect |
|---|---|---|
| `BITSTREAM_IMEM` | 2048 | IMEM instruction ceiling of the FLASHED bitstream. Set `8192` on the 8K-IMEM image; programs above it are refused (loud on READ flows, see the accum caveat in `api-patches/README.md` §0004). |
| `PIM_USE_LOAD_WEIGHTS` | client sets 1 | Weights resident in DRAM backup rows (RowClone-refreshed) instead of streamed per request. |
| `PIM_FUSED_COSET` | 0 | Coset-broadcast fused MAJ3 bodies — 1.45–1.6×/matmul, 1.63× real-model (`docs/BONSAI_2026_07.md`). Requires the primary calibrated tuple. |
| `PIM_1BIT_SINGLE` | 0 | 1-bit models: compute only the positive track; the client reconstructs y = 2·y_pos − Σ fac·pc(x). ~1.8× (`docs/BONSAI_2026_07.md`). |
| `PIM_SEGPOP` | 0 | **build7 image only**: SEG_POP readback — 2048 B/row of per-segment popcount bytes, host popcount eliminated (`docs/ROADB_2026_07.md` §7). Never set on pre-build7 images. |
| `PIM_INLINE_BITPLANES` | 1 | K bitplane-chunks batched per program (needs the 8K IMEM for K>1). NOTE: K>1 structurally disables `PIM_PARALLEL_BANKS` (duplicate-bank fallback). |
| `PIM_PARALLEL_BANKS` | 0 | pack4 4-bank interleaved doubleACTs. Engages only at K=1 with 4 distinct banks; wall-neutral on the readout-bound wall (`docs/UTILIZATION.md` addendum). |
| `PIM_PACK_ROUNDS` | 1 | MM3D-only: pack up to N rounds' bodies per program within the IMEM envelope. |
| `PIM_RECV_TIMEOUT_MS` | unset | Opt-in stall guard on `receiveData` (unset = block forever, the pristine behavior). |
| `PIM_VOTE_FULL` | model-dep | Full voting extras; off in current production ladders. |

Per-DIMM setup (subarray window `PIM_SUB_START`/`PIM_SUB_END`, calib
file, pool layouts) is applied automatically by the Python client's
per-bender preset table (`python/run_bitnet_pim.py`).
