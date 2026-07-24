# `python/` — BitNet PIM orchestrator

Patches Hugging Face's `transformers` library so specific BitNet
projection layers route their matrix-multiplications to the
FPGA-side server (or per-call runner) while the rest of the model
runs on the CPU.

## Files

- **`pim_linear.py`** — `PimBitLinear`, a drop-in replacement for
  BitNet's `BitLinear` that calls the FPGA-side binary, plus
  `pim_substitute(model, layer_idx, projs, …)` which patches a
  loaded model in place. Maintains a long-running `PimServer`
  subprocess shared across calls (see the inline docstring for
  protocol details).
- **`run_bitnet_pim.py`** — CLI wrapper. Loads the model via
  `transformers.AutoModelForCausalLM`, calls `pim_substitute`,
  then runs `model.generate()` on a prompt.

## Setup

```bash
# Pin to transformers 4.52 — that's what we tested against.
# Newer versions may have moved BitNet internals; older ones lack support.
pip install transformers==4.52 torch

# Download the model. ~1.18 GB, one-time.
huggingface-cli download microsoft/bitnet-b1.58-2B-4T \
    --local-dir ~/bitnet_weights/microsoft__bitnet-b1.58-2B-4T
```

## Configuration via environment variables

`run_bitnet_pim.py` reads:

| Variable | Default | Purpose |
|---|---|---|
| `BITNET_CACHE` | `~/bitnet_weights` | Hugging Face cache directory containing the BitNet snapshot. |
| `CASA_CALIB`   | `../calibration/calib_dimm0.txt` (relative to script) | Calibrated MAJ3-perfect tuples for the target DIMM. |
| `CASA_RUNNER`  | (no default — must be set) | Path to `bitnet-proj-exe` (per-call runner). |
| `CASA_SERVER`  | (no default — must be set) | Path to `bitnet-proj-server` (long-running daemon). |

`run_bitnet_pim.py` requires either `CASA_RUNNER` or `CASA_SERVER` to
be set; it errors out with a helpful message otherwise.

## Run

```bash
cd python
export CASA_RUNNER=/path/to/DRAM-Bender/.../BitNet/bitnet-proj-exe
export CASA_SERVER=/path/to/DRAM-Bender/.../BitNet/bitnet-proj-server

# Single bank, BitNet layer-0 q_proj only on PIM (fastest smoke test).
python3 run_bitnet_pim.py --max-tokens 4 --bank "0" \
    --prompt "What is the capital of Hungary?"

# All 7 layer-0 projections on PIM, multi-bank Path C (1.28× faster
# than single-bank, with mild output drift — see docs/METHODOLOGY.md).
python3 run_bitnet_pim.py --max-tokens 8 --projs all --bank "0,1,2,3" \
    --prompt "What is the capital of Hungary? Answer in one sentence."
```

## What `pim_substitute` does

1. Walks the loaded model and finds `BitLinear` modules at the
   specified `(layer, proj)` paths.
2. Replaces each with `PimBitLinear`, which intercepts the forward
   pass, packs the activation into the protocol the C++ server
   expects, and reads back the result row.
3. The PIM-side computation is bit-exact (or close, see
   methodology) for ternary × int8 matmul; PyTorch handles
   everything else (RMSNorm, attention, sampling).

The substitution is idempotent and reversible — you can swap a layer
in and out of PIM without reloading the model.

## Known limitations

- Tested only against `transformers==4.52` and BitNet
  `b1.58-2B-4T`. Other BitNet variants would need protocol
  adjustments.
- Output diverges between single-bank and multi-bank runs because
  different calibrated tuples have different per-cell flaky
  patterns. For exact reproducibility (e.g., the demo punchline),
  pin to one bank. See `docs/METHODOLOGY.md`.
