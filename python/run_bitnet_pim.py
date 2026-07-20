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

_BN = "/home/deni/Claude/SiMRA-DRAM-main/DRAM-Bender/sources/apps/DSN_AE_APPS/BitNet"
# Per-DIMM specs for multi-DIMM. Only the validated DIMMs (0, 2) are here.
# sub_start/sub_end = None means the server's default 640-aligned math is
# correct (DIMM 0). DIMM 2's s_id 72 is not 640-aligned → explicit range.
DIMM_SPECS = {
    # 2026-07-20: clone-ok pools on both DIMMs (O8/O5 hygiene — the May
    # pools were ~33-39% clone-dead by the anti-selection corollary).
    # dimm0 sub window explicit per the O5 ready-state spec.
    0: {'bender': 0, 'calib': f"{_BN}/calib_dimm0.txt",
        'pool_layout': f"{_BN}/pool_layout_dimm0_cloneok_bank{{bank}}.txt",
        'sub_start': 38400, 'sub_end': 39040,
        # O10 2026-07-20: fused-layout colmask (host-repairs the fused
        # OPERAND-LAYOUT-marginal columns of this die; ~9% of columns).
        # Gated by PIM_D0_FUSED_COLMASK=0 for A/B against the o5fix shape.
        **({} if os.environ.get('PIM_D0_FUSED_COLMASK', '1') == '0' else
           {'fused_colmask': f"{_BN}/fused_colmask_dimm0_bank{{bank}}.txt"})},
    2: {'bender': 2, 'calib': f"{_BN}/calib_dimm2.txt",
        'pool_layout': f"{_BN}/pool_layout_dimm2_cloneok_bank{{bank}}.txt",
        'sub_start': 45312, 'sub_end': 45952},
}


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
    # Default banks=0,1,2,3 (full 4-bank parallel).  Each bank has its
    # own per-bank fault-free pool layout selected via the {bank} token
    # in PIM_POOL_LIST_FILE.  Banks 0/2/3 share calib (Rfirst=38424) and
    # use the same layout file; bank 1 uses its own (Rfirst=38446).
    # See simra_xor8_spread.md.
    ap.add_argument("--bank", type=str, default="0,1,2,3",
                    help="Bank arg passed through to server: '0' (single "
                         "bank) or '0,1,2,3' (4-bank Path C with per-bank "
                         "fault-free pool layouts).")
    ap.add_argument("--prompt", default="What is the capital of France?")
    ap.add_argument("--max-tokens", type=int, default=20)
    ap.add_argument("--calib", default=CALIB,
                    help="Calibration file (default: DIMM 0). Use "
                         "calib_dimm2.txt for DIMM 2 (bender 2).")
    ap.add_argument("--pool-layout", default=None,
                    help="PIM_POOL_LIST_FILE pattern with {bank} token. "
                         "Default: DIMM 0 layout. For DIMM 2 use "
                         "pool_layout_dimm2_bank{bank}.txt.")
    ap.add_argument("--dimms", default=None,
                    help="Comma-separated DIMM ids for MULTI-DIMM parallel "
                         "(e.g. '0,2'). d_in sub-handles are round-robin'd "
                         "across the DIMMs and run concurrently. Overrides "
                         "--bender/--calib/--pool-layout. Validated DIMMs: 0, 2.")
    args = ap.parse_args()

    # Build multi-DIMM config if --dimms given.
    dimm_configs = None
    if args.dimms:
        dimm_ids = [int(x) for x in args.dimms.split(',') if x.strip()]
        dimm_configs = []
        for d in dimm_ids:
            if d not in DIMM_SPECS:
                sys.exit(f"DIMM {d} not in DIMM_SPECS (validated: "
                         f"{sorted(DIMM_SPECS)}). Add a spec first.")
            spec = dict(DIMM_SPECS[d])
            spec['bank'] = args.bank   # pool_layout keeps its {bank} token; server substitutes
            dimm_configs.append(spec)
        print(f"[bnet] MULTI-DIMM: {dimm_ids} — d_in sub-handles split "
              f"across {len(dimm_configs)} DIMMs, concurrent + summed", flush=True)

    # Production-recommended PIM config: persistent-weight LOAD path with
    # per-bank fault-free pool layouts.  Bank N's layout loaded from
    # pool_layout_dimm0_bank<N>.txt via the {bank} token substitution.
    # ~1.45× speedup over V2 baseline at d_in=2048; max_err vs numpy
    # ~2394 (cleaner than V2's ~4828, because each bank's pool footprint
    # shrinks 4×). Override: PIM_USE_LOAD_WEIGHTS=0 → V2 fallback.
    os.environ.setdefault('PIM_USE_LOAD_WEIGHTS', '1')
    # Multi-DIMM passes pool layout per-server (extra_env); don't set a
    # global default that would shadow it. Single-DIMM keeps the env path.
    if not dimm_configs:
        if args.pool_layout:
            os.environ['PIM_POOL_LIST_FILE'] = args.pool_layout
        else:
            os.environ.setdefault('PIM_POOL_LIST_FILE',
                '/home/deni/Claude/SiMRA-DRAM-main/DRAM-Bender/sources/apps/'
                'DSN_AE_APPS/BitNet/pool_layout_dimm0_bank{bank}.txt')
    _layout_desc = (f"per-DIMM ({[d for d in (args.dimms or '').split(',') if d]})"
                    if dimm_configs
                    else os.path.basename(os.environ.get('PIM_POOL_LIST_FILE', '(none)')))
    print(f"[bnet] PIM config: PIM_USE_LOAD_WEIGHTS="
          f"{os.environ['PIM_USE_LOAD_WEIGHTS']}, "
          f"banks={args.bank}, "
          f"layout={_layout_desc}",
          flush=True)

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
                   bender_id=args.bender, calib_file=args.calib,
                   bank_id=args.bank, runner_path=RUNNER,
                   server_path=SERVER, use_server=True, verbose=True,
                   dimm_configs=dimm_configs)

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
