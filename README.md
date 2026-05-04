# CaSA — running an LLM inside a DRAM chip

A real, published 2.4-billion-parameter language model answers a
question, and the matrix-multiplications that produce the answer
happen *inside the DRAM cells* of a regular DDR4 memory module — via
charge-sharing, on real silicon.

This repository contains the software side of that demonstration:

- **`scheduler/`** — `casa_sched.c`, a discrete-event scheduler that
  models a DDR4 channel running charge-sharing PIM primitives, with
  bus contention tracked explicitly. Used to project per-token
  throughput under different optimization stacks.
- **`app/`** — C++ apps that issue charge-sharing primitives
  (RowClone, MAJ3 via `doubleACT`, multi-row broadcast) on real
  DRAM-Bender silicon and run the matmuls of Microsoft's
  BitNet b1.58-2B-4T using them. Drop these into a DRAM-Bender
  checkout and `make`.
- **`python/`** — orchestrator that patches Hugging Face's
  `transformers` library to route specific BitNet projection layers to
  the FPGA-side server while the rest of the model runs on the CPU.
- **`calibration/`** — calibrated MAJ3-perfect open-row tuples for one
  of our test DIMMs. Format documented; you produce your own for new
  DIMMs.
- **`docs/`** — hardware requirements, calibration protocol,
  scheduler-projection methodology.

This builds directly on prior research from the
[CMU SAFARI group](https://safari.ethz.ch/) — RowClone, Ambit,
SiMRA-DRAM, Multi-Row-Init, LISA, pLUTo — and the open-source
[DRAM-Bender](https://github.com/CMU-SAFARI/DRAM-Bender) FPGA platform.
We don't re-host either; you clone them yourself and place the C++
apps from `app/` into the right path. See `app/README.md`.

## Headline result — three regimes

**What we are running today** is BitNet b1.58-2B-4T with **1 of its
30 transformer layers' matrix-multiplies executing in DRAM** (the
seven projections of layer 0). The other 29 layers run in PyTorch
on the CPU. This is enough to demonstrate the mechanism end-to-end
on a real published model — running all 30 layers in DRAM is
straightforward engineering, not a science question, and the
scheduler projects what that would look like.

The current measurement is **dominated by orchestration overhead**
(per-call PCIe round-trips, per-column weight writes, Python +
subprocess). It is not what the silicon can do — it is what our
software currently lets the silicon do.

The cycle-level scheduler `casa_sched.c` exists to project the
**bus-bound** silicon ceiling: what happens once orchestration
overhead is engineered out and the only real wall is the DDR bus.
Every number below the "MEASURED" row comes from running the
scheduler with the listed flags; the bus utilization the scheduler
reports is printed alongside.

All projections come from `casa_sched.c` configured to match what our
silicon actually issues per MAJ3: an activation broadcast plus 5
full-row bus_writes (the activation update — `doubleACT(10,2)`
broadcasts to all 16 open rows, so the activation slots have to be
overwritten individually), a 3-cycle frac discharge, the MAJ3 itself,
and a result read. The scheduler bookkeeping respects every standard
DDR4 timing parameter (tRCD, tRP, tFAW, tCCD, tBurst, tWR, tREFI,
tRFC) and tracks bus and bank utilization explicitly. Bus-bound
projections quoted below are what our **current silicon
implementation** would achieve at full bus utilization — i.e. with
all the orchestration overhead engineered out. They are not
hypothetical-future projections.

| Regime | Per-token | tok/s | Bus % | Source |
|---|---|---|---|---|
| **MEASURED today** — 1 of 30 BitNet layers in DRAM, multi-bank, ~200× orchestration overhead per MAJ3 | ~30 s/tok per layer in DRAM | ~0.001 *(if extrapolated to all 30 layers)* | ~2 % | this hardware, today |
| **Bus-bound ceiling** — all 30 layers in DRAM, 1 DIMM, current silicon path | 3.0 s | **0.33** | 97.2 % | `casa_sched --dimms 1` |
| + bank-group-parallel bus | 2.4 s | 0.40 | 96.5 % | `... --bg-parallel` |
| + 4 DIMMs in parallel | 0.61 s | 1.57 | 95.6 % | `--dimms 4 --bg-parallel` |
| + in-DRAM popcount *(vendor RTL change)* | 0.52 s | 1.86 | 94.9 % | `... --popcount dram` |
| + LISA cross-subarray bus *(vendor RTL change)* | 0.51 s | 1.90 | 94.9 % | `... --lisa` |
| ─── *with a hypothetical selective-broadcast DRAM primitive (`--ideal-acts`) that would eliminate the per-MAJ3 activation wrRows* ─── | | | | |
| 1 DIMM ideal | 503 ms | 1.92 | 98.3 % | `casa_sched --dimms 1 --ideal-acts` |
| 4 DIMMs ideal + popcount | 16 ms | **59.99** | 73 % | `... --popcount dram --ideal-acts` |
| 4 DIMMs ideal + popcount + LISA | 9 ms | **109.16** | 18 % | `... --lisa --ideal-acts` |
| + binary activations (model retrain) | 1.8 ms | 545 | 18 % | `... --act-bits 1 --ideal-acts` |

Three sentences for the story:

1. **Today's measurement is ~200× slower per MAJ3 than the
   silicon's bus-bound ceiling**, and the entire gap is software —
   eliminating per-call PCIe round-trips, batching MAJ3s into
   SoftMC outer loops, pre-loading weights into DRAM at startup
   so the runtime never per-column-writes weights again.

2. **The bus-bound ceiling on existing DRAM is modest** — ~2 tok/s
   with 4 DIMMs even after every realistic optimization, because
   updating activation rows takes 5 full-row bus_writes per MAJ3
   (the `doubleACT` broadcast primitive distributes to all 16
   open rows, so the 5 activation slots must be individually
   re-written via wrRow). At that point the bus is genuinely full;
   adding popcount or LISA helps only marginally because they
   target the bus_read, not the bus_writes that dominate.

3. **Beyond ~2 tok/s requires a new DRAM primitive** — specifically
   a selective subset-broadcast that updates the 5 activation rows
   in the open-set via charge-sharing instead of bus_writes. The
   scheduler models this hypothetically as `--ideal-acts`; with it,
   in-DRAM popcount unlocks 60 tok/s and LISA brings it to 109 —
   GPU-competitive on this exact model. None of those numbers are
   reachable on existing DRAM without that new primitive.

Output is **bit-exact correct on most cells** (~22 139 of 22 144 in
one full BitNet layer = 99.98 %). The 5 stray flips come from cells
that pass the calibrated 1000-pattern stability test but flip on
uncalibrated bit-combinations. Ternary models are robust to this by
construction. See `docs/METHODOLOGY.md`.

The point of the work is not to beat a GPU on speed. The point is to
**demonstrate the mechanism** on real silicon and put concrete,
scheduler-bounded numbers on what would change with two specific
DRAM-vendor improvements (in-DRAM popcount, LISA).

## Quick start (assuming you already have DRAM-Bender silicon)

```bash
# 1. Clone DRAM-Bender (the FPGA controller + bitstream)
git clone https://github.com/CMU-SAFARI/DRAM-Bender
# … bring up the BCU1525 bitstream per DRAM-Bender's docs.

# 2. Drop our C++ apps into DRAM-Bender's apps tree and build.
cp app/*.cpp app/Makefile DRAM-Bender/sources/apps/DSN_AE_APPS/BitNet/
cp calibration/calib_dimm0.txt   DRAM-Bender/sources/apps/DSN_AE_APPS/BitNet/
cd DRAM-Bender/sources/apps/DSN_AE_APPS/BitNet
make

# 3. Calibrate a DIMM (only needed once per chip — see docs/CALIBRATION.md).
#    The shipped calib_dimm0.txt is for one of our test DIMMs;
#    your hardware may need its own characterization.

# 4. Run an end-to-end smoke test:
./bitnet-real-exe 0 calib_dimm0.txt 1
# Expected: bit-exact match on a small ternary x int8 matrix multiply.

# 5. Hook the long-running PIM server into BitNet inference:
export CASA_RUNNER=$PWD/bitnet-proj-exe
export CASA_SERVER=$PWD/bitnet-proj-server
export BITNET_CACHE=~/bitnet_weights         # any HF cache dir
cd <repo>/python
pip install transformers==4.52 torch
python3 run_bitnet_pim.py \
    --max-tokens 8 --projs all --bank "0,1,2,3" \
    --prompt "What is the capital of Hungary? Answer in one sentence."
# Expected output ends with "Budapest" after ~4 minutes.
```

See `python/README.md` for argument details and how the
`pim_substitute` swap works internally.

## Hardware requirements (summary)

- A Xilinx Alveo U200 / BCU1525 (or compatible) FPGA card flashed
  with the DRAM-Bender bitstream.
- One or more DDR4 1333 MT/s DIMMs in the FPGA's DIMM slots
  (you'll need to characterize them).
- A host with PCIe-attached FPGA, the Xilinx XDMA driver loaded,
  and the SoftMC API available.

Full details in `docs/HARDWARE.md`.

## Honest caveats

- **Per-cell yield**: a small fraction of cells (we measured
  ~5/22 144 in one full layer = 0.02 %) flip on input bit-patterns
  the calibration didn't exhaust. BitNet is robust enough to absorb
  this, and most outputs land bit-exact. The per-bank yield is
  run-to-run nondeterministic — `docs/METHODOLOGY.md` discusses.
- **Multi-bank divergence**: parallelizing across multiple banks
  uses multiple calibrated tuples, each with its own flaky-cell
  pattern. Output stays sensible but is not bit-exact across runs.
  For deterministic demos, pin to one bank.
- **Multi-DIMM scaling** is not yet integrated: characterization on
  the additional DIMMs (1, 2, 3) is in progress at the time of
  release. The scheduler projections that include 4-DIMM parallelism
  assume the calibration completes; the pure single-DIMM numbers do
  not.
- **The simulator was written before the silicon implementation**.
  Its hardcoded charge-sharing latencies were patched against
  measured DIMM 0 values; numbers shift by <2% because the
  projections are bus-bound, not MAJ3-bound. See
  `docs/METHODOLOGY.md`.

## Acknowledgments

- **CMU SAFARI Group** (Onur Mutlu et al.) — RowClone, Ambit,
  SiMRA-DRAM, Multi-Row-Init, LISA, pLUTo. Without their decade of
  characterization papers and open-source toolkits, none of this is
  possible on existing silicon.
- **Microsoft Research** — BitNet b1.58-2B-4T, an open-weight
  2.4-billion-parameter ternary language model.
- **Hugging Face** — `transformers` library (we test against
  v4.52).
- The communities behind `Manim`, `Piper TTS`, `matplotlib`, and
  `ffmpeg` for the video-production tooling used in the
  presentation (sources for those live in the private prototype
  repository, not here).

## License

MIT — see [LICENSE](LICENSE). Upstream components remain under their
own licenses (DRAM-Bender, SiMRA-DRAM, BitNet, transformers, …); we
don't ship them.
