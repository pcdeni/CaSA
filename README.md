# CaSA: running an LLM inside a DRAM chip (software side)

A published, multi-billion-parameter language model answers a question,
and the heaviest matrix-multiplies that produce the answer run **inside the
charge of ordinary DDR4 cells** — on real silicon, in unmodified memory.

**The name.** *CaSA* is **cha**rge **s**h**a**ring: open several DRAM rows
close enough together and the value that survives on the shared wire is the
majority of what they held — and that majority is the computation.

<p align="center">
  <img src="docs/assets/architecture.svg" width="820"
       alt="A host running PyTorch talks over PCIe to an FPGA controller (DRAM-Bender on a Xilinx BCU1525 card), which drives a DDR4 DIMM. The weights live resident in the DIMM's cells and the projection matrix-multiplies run there as a majority vote over cell charge. The host-to-DRAM round-trip, repeated once per weight chunk, is the wall; only cutting the request count moves it.">
</p>

The host runs attention softmax, norms, and sampling in PyTorch, exactly as
every PUD system does. The projection matrix-multiplies (the model's linear
layers) run in the DIMM, where the weights stay resident. The one slow part is
the round-trip to ask the chip and read an answer back — **the wall**.

## The ladder

BitNet b1.58-2B-4T runs today at **45 s/token** on real silicon, down from
**632 s/token** at the start of the campaign. The **DDR-PHY-bound floor** —
the speed once orchestration is engineered out and only the memory bus is
left — is a sim-validated target.

| Per-token, on this hardware | |
|---|---|
| Start of campaign | **632 s** |
| Today | **45 s** |
| Target: DDR-PHY-bound floor | **0.02–0.04 s** (sim-validated) |

## Seven flagship models, on unmodified DDR4

The same silicon path runs seven LLM families: **BitNet b1.58-2B-4T**,
**Bonsai-1.7B** (ternary), **Bonsai-1.7B** (1-bit), **Llama2-7B**,
**Llama2-13B**, **Llama3-8B**, and **Phi-4**.

## Where to go next

| If you want to… | Go to |
|---|---|
| **Understand it** — the plain-language walkthrough, cell to inference loop | [Interactive explainer](https://pcdeni.github.io/CaSA/explainer/) |
| **See the physics** — one command pair, two behaviours, the timing that picks between them | [Mechanism explainer](https://pcdeni.github.io/CaSA/explainer/xor-spread.html) |
| **Compare it / see how we measure** — related systems, the wall model, verification discipline | [Related systems + methodology](docs/RELATED_SYSTEMS.md) |
| **MVDRAM reproduction** — reproduction of MVDRAM (arXiv 2503.23817) from the paper | [MVDRAM reproduction study](docs/MVDRAM_REPRODUCTION.md) |

## What's in this repository

The software side of the demonstration:

- **`app/`** — C++ apps that issue charge-sharing primitives on real
  DRAM-Bender silicon and run the projection matmuls of Microsoft's BitNet
  b1.58-2B-4T. Drop into a DRAM-Bender checkout and `make`.
- **`python/`** — orchestrator that patches Hugging Face `transformers` to
  route projection layers to the FPGA-side server (per-tensor ternary and g128 group-scaled specs).
- **`lane2/`** — the MVDRAM-reproduction GeMV (general matrix-vector multiply)
  server and drivers (LOAD/GEMV/PARTIALS protocol, dual-track adder, per-block
  exact partials, sampled end-to-end llama.cpp runners).
- **`rtl/`** — the DIFF-accum readback engine with its Verilator harness and
  the silicon-found fixes.
- **`scheduler/`** — `casa_sched.c`, a discrete-event DDR4 scheduler that
  projects the bus-bound floor beneath the measured arc.
- **`calibration/`** — calibrated open-row tuples for one of our test DIMMs
  (format documented; you produce your own for new DIMMs).
- **`api-patches/`** / **`shim-patches/`** — unified diffs for the
  SiMRA/DRAM-Bender API and the llama.cpp mulmat shim.
- **`docs/`** — hardware, calibration, methodology, related systems, and the
  [explainers](https://pcdeni.github.io/CaSA/explainer/).

This builds directly on prior research from the
[SAFARI Research Group](https://safari.ethz.ch/) — RowClone, Ambit, SiMRA-DRAM,
Multi-Row-Init and the open-source
[DRAM-Bender](https://github.com/CMU-SAFARI/DRAM-Bender) FPGA platform. We
don't re-host either; you clone them and place the `app/` C++ apps into the
right path (see `app/README.md`).

## Quick start

*(assuming you already have DRAM-Bender silicon)*

```bash
# 1. Clone DRAM-Bender (the FPGA controller + bitstream) and bring up
#    the BCU1525 bitstream per DRAM-Bender's docs.
git clone https://github.com/CMU-SAFARI/DRAM-Bender

# 2. Drop our C++ apps into DRAM-Bender's apps tree and build.
cp app/*.cpp app/Makefile DRAM-Bender/sources/apps/DSN_AE_APPS/BitNet/
cp calibration/calib_dimm0.txt   DRAM-Bender/sources/apps/DSN_AE_APPS/BitNet/
cd DRAM-Bender/sources/apps/DSN_AE_APPS/BitNet
make

# 3. Calibrate a DIMM (once per chip — see docs/CALIBRATION.md).
#    The shipped calib_dimm0.txt is for one of our test DIMMs;
#    your hardware may need its own characterization.

# 4. End-to-end smoke test (expect a bit-exact ternary x int8 matmul):
./bitnet-real-exe 0 calib_dimm0.txt 1

# 5. Hook the long-running PIM server into BitNet inference.
export PIM_RUNNER=$PWD/bitnet-proj-exe
export PIM_SERVER=$PWD/bitnet-proj-server
export BITNET_CACHE=~/bitnet_weights         # any HF cache dir
cd <repo>/python
pip install transformers==4.52 torch
python3 run_bitnet_pim.py \
    --max-tokens 8 --projs all --bank "0,1,2,3" \
    --prompt "What is the capital of Hungary? Answer in one sentence."
# Expected output ends with "Budapest".
```

See `python/README.md` for argument details and how the `pim_substitute`
swap works internally.

## Hardware requirements (summary)

- A Xilinx Alveo U200 / VCU1525 / BCU1525 (or compatible) FPGA card flashed with the
  DRAM-Bender bitstream.
- One or more DDR4 1333 MT/s DIMMs in the FPGA's DIMM slots (you characterize
  them yourself).
- A host with a PCIe-attached FPGA, the Xilinx XDMA driver loaded, and the
  SoftMC API available.

Full details in `docs/HARDWARE.md`.

## Acknowledgments

- **SAFARI Research Group** (Onur Mutlu et al.) — RowClone, Ambit, SiMRA-DRAM,
  Multi-Row-Init. Without their decade of characterization
  papers and open-source toolkits, none of this is possible on existing
  silicon.
- **MVDRAM** (Tatsuya Kubo, Daichi Tokuda, Tomoya Nagatani, Masayuki Usui,
  Lei Qu, Ting Cao, et al.) — Enabling GeMV Execution in Unmodified DRAM for
  Low-Bit LLM Acceleration. Special thanks to Tatsuya Kubo!
- **Microsoft Research** — BitNet b1.58-2B-4T, an open-weight
  2.4-billion-parameter ternary language model.
- **Prism ML, Inc.** — Bonsai-1.7B, open-weight 1-bit and ternary
  group-scaled quantizations of Alibaba's Qwen3-1.7B (Apache-2.0); the second
  base-model family brought up on this pipeline.
- **Hugging Face** — `transformers` (we test against v4.52).

## License

MIT — see [LICENSE](LICENSE). Upstream components remain under their own
licenses (DRAM-Bender, SiMRA-DRAM, BitNet, transformers, …); we don't ship
them.
