#!/usr/bin/env python3
"""BitNet generation with one (or more) layer projections running on
real DRAM-Bender PIM silicon.

By default, swaps layer 0's q_proj with PIM. Use --layers / --projs to
expand. The rest of the model runs in PyTorch.

Configuration via environment variables (with sensible defaults):
  BITNET_CACHE   Hugging Face cache for the BitNet weights.
                 Default: ~/bitnet_weights
  CASA_CALIB     Calibrated MAJ3 tuples for your DIMM.
                 Default: ../calibration/calib_dimm0.txt (relative to this script)
  CASA_RUNNER    Path to the per-call subprocess runner binary
                 (built from app/test_bitnet_proj.cpp; binary name `bitnet-proj-exe`).
  CASA_SERVER    Path to the long-running PIM server binary
                 (built from app/test_bitnet_server.cpp; binary name `bitnet-proj-server`).

Both CASA_RUNNER and CASA_SERVER must be built from the C++ apps in this
project's `app/` directory, which in turn depend on a checkout of
DRAM-Bender (https://github.com/CMU-SAFARI/DRAM-Bender). See app/README.md.
"""
import argparse, os, sys, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Allow `import pim_linear` whether the user runs this script from the
# python/ dir or from elsewhere.
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from pim_linear import pim_substitute

MODEL  = "microsoft/bitnet-b1.58-2B-4T"
CACHE  = os.environ.get("BITNET_CACHE",
                        os.path.expanduser("~/bitnet_weights"))
CALIB  = os.environ.get("CASA_CALIB",
                        os.path.normpath(os.path.join(
                            HERE, "..", "calibration", "calib_dimm0.txt")))
RUNNER = os.environ.get("CASA_RUNNER")
SERVER = os.environ.get("CASA_SERVER")
if not RUNNER and not SERVER:
    sys.stderr.write(
        "error: set CASA_RUNNER and/or CASA_SERVER to point at the built "
        "bitnet-proj-exe / bitnet-proj-server binaries (see app/README.md).\n")
    sys.exit(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="0",
                    help="Comma-separated layer indices to PIM-substitute "
                         "(default: 0).")
    ap.add_argument("--projs", default="self_attn.q_proj",
                    help="Comma-separated projection paths to swap. "
                         "Examples: self_attn.q_proj | "
                         "self_attn.q_proj,self_attn.k_proj,self_attn.v_proj | "
                         "all (= all 7).")
    ap.add_argument("--bender", type=int, default=0)
    ap.add_argument("--bank", type=str, default="1",
                    help="Bank arg passed through to server: '1' (single "
                         "bank) or '0,1,2,3' (multi-bank Path C).")
    ap.add_argument("--prompt", default="What is the capital of France?")
    ap.add_argument("--max-tokens", type=int, default=20)
    args = ap.parse_args()

    layer_idx = [int(x) for x in args.layers.split(",")]
    if args.projs == "all":
        projs = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                 "self_attn.o_proj",
                 "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
    else:
        projs = args.projs.split(",")

    print(f"[bnet] loading {MODEL} (cache={CACHE})", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL, cache_dir=CACHE)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, cache_dir=CACHE)
    model.eval()
    print(f"[bnet] loaded in {time.time()-t0:.1f}s "
          f"({sum(p.numel() for p in model.parameters())/1e9:.2f}B params)",
          flush=True)

    # Long-running PIM server (default). Verified bit-exact match against
    # subprocess backend. Per-token speedup is ~1-3 % in this config —
    # subprocess startup wasn't the dominant cost (per-MAJ3 weight reload
    # is). Set use_server=False to use subprocess-per-call path.
    pim_substitute(model, layer_idx, projs,
                   bender_id=args.bender, calib_file=CALIB,
                   bank_id=args.bank, runner_path=RUNNER,
                   server_path=SERVER, use_server=True, verbose=True)

    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user",   "content": args.prompt},
    ]
    prompt_text = tok.apply_chat_template(messages, tokenize=False,
                                           add_generation_prompt=True)
    inputs = tok(prompt_text, return_tensors="pt")
    n_in = inputs.input_ids.shape[1]
    print(f"\n[bnet] user: {args.prompt!r}")
    print(f"[bnet] {n_in} input tokens, generating up to {args.max_tokens}...",
          flush=True)
    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=args.max_tokens,
                             do_sample=False, pad_token_id=tok.eos_token_id)
    dt = time.time() - t0
    n_new = out.shape[1] - n_in
    response = tok.decode(out[0][n_in:], skip_special_tokens=True)
    print(f"\n[bnet] response ({n_new} tok in {dt:.1f}s, "
          f"{n_new/max(dt,1e-3):.2f} tok/s):")
    print(f"   {response!r}")


if __name__ == "__main__":
    main()
