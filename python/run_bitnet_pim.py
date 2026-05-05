#!/usr/bin/env python3
"""BitNet generation with as many BitLinear projections as possible
running on REAL DRAM-Bender PIM silicon.

Default: ALL transformer layers × ALL 7 BitLinears (= 210 matmuls per
token, ~95% of model parameters). Only the LM head (regular Linear,
not BitLinear) and the embedding lookup stay in PyTorch — those are
not PIM-eligible since the MAJ3 primitive only does ternary × int8.

This is correctness-focused, not speed-focused: per-token wall time
scales with the number of substituted projections. Use --layers /
--projs to subset for shorter test runs.
"""
import argparse, os, sys, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/home/deni/bitnet_weights")
from pim_linear import pim_substitute, print_pim_timing_summary

MODEL = "microsoft/bitnet-b1.58-2B-4T"
CACHE = "/home/deni/bitnet_weights"
CALIB = "/home/deni/Claude/SiMRA-DRAM-main/DRAM-Bender/sources/apps/DSN_AE_APPS/BitNet/calib_dimm0.txt"
RUNNER = "/home/deni/Claude/SiMRA-DRAM-main/DRAM-Bender/sources/apps/DSN_AE_APPS/BitNet/bitnet-proj-exe"
SERVER = "/home/deni/Claude/SiMRA-DRAM-main/DRAM-Bender/sources/apps/DSN_AE_APPS/BitNet/bitnet-proj-server"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="all",
                    help="Comma-separated layer indices to PIM-substitute, "
                         "or 'all' for every transformer layer (default).")
    ap.add_argument("--projs", default="all",
                    help="Comma-separated projection paths to swap, or "
                         "'all' for every BitLinear in the layer (default). "
                         "Example subset: self_attn.q_proj,mlp.gate_proj")
    ap.add_argument("--bender", type=int, default=0)
    ap.add_argument("--bank", type=str, default="1",
                    help="Bank arg passed through to server: '1' (single "
                         "bank) or '0,1,2,3' (multi-bank Path C).")
    ap.add_argument("--prompt", default="What is the capital of France?")
    ap.add_argument("--max-tokens", type=int, default=20)
    args = ap.parse_args()

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

    # Resolve --layers all into the full transformer-layer index range.
    if args.layers == "all":
        layer_idx = list(range(model.config.num_hidden_layers))
    else:
        layer_idx = [int(x) for x in args.layers.split(",")]
    n_total_bitlinears = model.config.num_hidden_layers * 7
    n_to_substitute = len(layer_idx) * len(projs)
    print(f"[bnet] PIM-substituting {n_to_substitute}/{n_total_bitlinears} "
          f"BitLinears = {100*n_to_substitute/n_total_bitlinears:.1f}% of "
          f"model BitLinears (LM head + embeddings stay in PyTorch — "
          f"that's the irreducible non-BitLinear ~5%)", flush=True)

    # Long-running PIM server (default). Verified bit-exact match against
    # subprocess backend. Per-token speedup is ~1-3 % in this config —
    # subprocess startup wasn't the dominant cost (per-MAJ3 weight reload
    # is). Set use_server=False to use subprocess-per-call path.
    pim_substitute(model, layer_idx, projs,
                   bender_id=args.bender, calib_file=CALIB,
                   bank_id=args.bank, runner_path=RUNNER,
                   server_path=SERVER, use_server=True, verbose=True)

    if os.environ.get("PIM_NO_CHAT_TEMPLATE"):
        # Demo / minimum-prefill mode: skip the chat template wrapper so
        # the prompt tokenises into just a few tokens. Useful for verifying
        # the full LLM runs end-to-end on PIM in tight wall-time budgets.
        prompt_text = args.prompt
    else:
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
    print_pim_timing_summary(model)


if __name__ == "__main__":
    main()
