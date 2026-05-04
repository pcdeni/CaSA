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
