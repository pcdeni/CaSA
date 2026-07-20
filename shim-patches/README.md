# `shim-patches/` — the llama.cpp interception shim (Lane 2)

`llama.cpp` is a third-party tree, so its integration ships here as **files +
an attachment diff** rather than a vendored copy.

- `mvdram-pim.h`, `mvdram-pim.c` — the census/interception shim. It hooks
  `llama.cpp`'s CPU-backend GeMV path, recognizes the quantized weight tensors
  of the four target models, extracts their bitplanes (**all four quant
  extractors**: `q4_0` incl. the `q4_0x8`-repacked layout, `q2_K`, `q3_K`,
  `q6_K`), and routes matching ops to the Lane-2 in-DRAM GeMV server
  ([`../lane2/`](../lane2/)).
- `0001-attach-points.patch` — the generic mul_mat interception hook +
  the repack call site + the CMake wiring. This is the *only* edit to the
  upstream tree; the shim logic lives entirely in the two files above.

## Applying (base commit `6bdd77f`)

The patch was generated against `llama.cpp` at
`6bdd77f13cf11b264b4231d320afc404f48d576e`. From a checkout at (or near) that
commit:

```bash
cp shim-patches/mvdram-pim.h shim-patches/mvdram-pim.c \
   llama.cpp/ggml/src/ggml-cpu/
cd llama.cpp
git apply /path/to/shim-patches/0001-attach-points.patch   # CMakeLists.txt, ggml-cpu.c, repack.cpp
# then build as usual (the CMakeLists hunk compiles mvdram-pim.c into ggml-cpu)
```

The three attach points: `ggml-cpu.c` (the mul_mat interception), `repack.cpp`
(the repack call site), `CMakeLists.txt` (build the shim). The shim is inert
unless `MVDRAM_PIM=1`.

## Env knobs

| Var | Default | Purpose |
|---|---|---|
| `MVDRAM_PIM` | off | Master enable. Unset = pristine llama.cpp. |
| `MVDRAM_PIM_DRY` | `0` | **Census / dry-run**: recognize + log the ops, do NOT touch silicon. Use this first to see what would be intercepted. |
| `MVDRAM_PIM_ONLY` | — | Substring filter on the tensor/op name (intercept only matches). |
| `MVDRAM_PIM_ANY_SHAPE` | `0` | Relax the shape gate from the known census shapes to the server's cap. |
| `MVDRAM_PIM_MAX_OPS` | `3` | Cap the number of ops actually offloaded (safety fuse). |
| `MVDRAM_PIM_SAMPLE` | `9999` | How many output rows to sample-verify against the CPU reference. |
| `MVDRAM_PIM_VOTE` | — | Replica-vote the in-DRAM result (majority over N runs). |
| `MVDRAM_PIM_SERVER` / `_BENDER` / `_BANK` / `_SID` | — | Lane-2 server socket + which DRAM-Bender / bank / subarray to target. |
| `MVDRAM_PIM_CALIB` / `_COLMASK` / `_RBITS` | — | Calibration file, column mask, and r-bit precision for the offloaded GeMV. |
| `MVDRAM_PIM_LOG` / `_OPLOG` / `_DUMP` / `_DUMP_TENSOR` | — | Diagnostics: per-op log, op-summary log, payload dump, tensor dump. |
| `MVDRAM_PIM_RESP_TIMEOUT_S` | — | Client-side response timeout for the server round-trip. |

Recommended first run: `MVDRAM_PIM=1 MVDRAM_PIM_DRY=1 …` to census, then drop
`_DRY` and raise `_MAX_OPS` once the shapes/route look right.

## The repack de-interleave finding

llama.cpp's CPU backend may store `q4_0` weights **repacked** as
`block_q4_0x8` (144 B, `repack.cpp make_block_q4_0x8`, interleave = 8): eight
logical q4_0 blocks are interleaved, so the extractor must **de-interleave**
them to recover the canonical per-column nibbles before bitplane encoding — a
naive canonical read gets scrambled weights. `q3_K` has **no** repack variant
in this checkout (`ggml_repack_get_optimal_repack_type` has no Q3_K case), so
its extractor always takes the canonical path. The extractors handle both
layouts; see the header comments in `mvdram-pim.c`.
