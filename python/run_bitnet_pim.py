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
# PIM_SERVER_PATH overrides the server binary (e.g. a copied binary run
# through a PIM_BACKEND=sim wrapper). Default = production path, unchanged.
SERVER = os.environ.get(
    "PIM_SERVER_PATH",
    "/home/deni/Claude/SiMRA-DRAM-main/DRAM-Bender/sources/apps/DSN_AE_APPS/BitNet/bitnet-proj-server")

# ---- PrismML Bonsai-1.7B (Qwen3-1.7B based; 2026-07-20) ----
# Extracted per-projection npz: codes int8 {-1,0,+1} + group_scales f32
# per (row, 128-input group) + sparse exact residuals. See
# /home/deni/Claude/bonsai_prep_2026_07_20/README.md and
# /home/deni/Claude/bonsai_client_2026_07_20/README.md.
# 1-bit runs DUAL-TRACK by default (empty zero-set, neg_mask = ~pos_mask
# within d_in). Since 2026-07-21, PIM_1BIT_SINGLE=1 activates the V2S
# single-track protocol: the server computes only the pos track and the
# client reconstructs y = 2·y_pos − Σx — halves the per-request DRAM work
# (pim_linear.py / MAGIC_V2S).
BONSAI_SPECS = {
    "bonsai_1bit": {
        "model_dir": "/home/deni/bonsai_weights/1bit",
        "extract_dir": "/home/deni/bonsai_weights/extracted/1bit",
    },
    "bonsai_ternary": {
        "model_dir": "/home/deni/bonsai_weights/ternary",
        "extract_dir": "/home/deni/bonsai_weights/extracted/ternary",
    },
}


def make_bonsai_spec_fn(extract_dir):
    """weight_spec_fn for pim_substitute: load L<LL>.<proj>.npz on demand."""
    import numpy as np

    def spec_fn(li, proj_path):
        name = proj_path.split(".")[-1]
        z = np.load(os.path.join(extract_dir, f"L{li:02d}.{name}.npz"))
        return {
            "codes": z["codes"],
            "group_scales": z["group_scales"],
            "group_size": int(z["group_size"]),
            "residual_idx": z["residual_idx"],
            "residual_val": z["residual_val"],
            # weight_scale (mean of group scales) kept for API-shape
            # compat only; unused when group_scales is present.
            "weight_scale": float(z["weight_scale"]),
        }
    return spec_fn

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
    ap.add_argument("--model", default="bitnet",
                    choices=["bitnet"] + sorted(BONSAI_SPECS),
                    help="bitnet (default; unchanged behavior) or a "
                         "PrismML Bonsai-1.7B variant (per-group g128 "
                         "scales via the pim_linear weight_spec path).")
    ap.add_argument("--dtype", default=None, choices=["bfloat16", "float32"],
                    help="Host-side model dtype. Default: bfloat16 for "
                         "bitnet (unchanged), float32 for bonsai (matches "
                         "the golden CPU reference).")
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
    # BITSTREAM_IMEM is guarded authoritatively in pim_linear.PimServer
    # (proc_env.setdefault) so EVERY runner gets it, not just this one.
    os.environ.setdefault('PIM_USE_LOAD_WEIGHTS', '1')
    # PIM_STREAM default ON (2026-07-28): phase-2 send-ahead pipeline is
    # silicon-VALIDATED on build-26 (magic 0x15): full-model −26.3% wall,
    # recv halved, 0 stalls/decay/errors over 11,500 requests, token-exact
    # (LEVERS ⚑ 07-27). The old build-11/12 branch-loop wedge (E14) was
    # fixed by the build-14 fetch_restart wiring; the flashed tower build is
    # streaming-capable (confirmed 07-28: PIM_STREAM=1 v2_oracle bit-exact
    # on b2). setdefault means an explicit env value still wins (clean
    # A/B baselines can force PIM_STREAM=0). Only unsafe on pre-build-9
    # flashes (magic < 0x08) — not this tower since 07-22.
    os.environ.setdefault('PIM_STREAM', '1')
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

    weight_spec_fn = None
    if args.model == "bitnet":
        # K=5 activation quant is the BitNet-2B default (CPU + silicon
        # validated 2026-07-28, token-identical, −32.2% wall; LEVERS #24).
        # BitNet-2B was QAT-trained with native int8 acts, so its safe floor
        # is K=5 — one bit lower than the Bonsai family (K=6). setdefault →
        # explicit PIM_ACT_K still wins.
        os.environ.setdefault("PIM_ACT_K", "5")
        print(f"[bnet] loading {MODEL} (cache={CACHE})", flush=True)
        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(MODEL, cache_dir=CACHE)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, torch_dtype=torch.bfloat16, cache_dir=CACHE)
        model.eval()
        print(f"[bnet] loaded in {time.time()-t0:.1f}s "
              f"({sum(p.numel() for p in model.parameters())/1e9:.2f}B params)",
              flush=True)
    else:
        spec = BONSAI_SPECS[args.model]
        dtype = (torch.bfloat16 if args.dtype == "bfloat16"
                 else torch.float32)
        print(f"[bnet] loading {args.model} from {spec['model_dir']} "
              f"(dtype={dtype})", flush=True)
        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(spec["model_dir"])
        model = AutoModelForCausalLM.from_pretrained(
            spec["model_dir"], torch_dtype=dtype)
        model.eval()
        print(f"[bnet] loaded in {time.time()-t0:.1f}s "
              f"({sum(p.numel() for p in model.parameters())/1e9:.2f}B params, "
              f"arch={model.config.architectures})", flush=True)
        weight_spec_fn = make_bonsai_spec_fn(spec["extract_dir"])
        print(f"[bnet] bonsai weight specs from {spec['extract_dir']} "
              f"(codes + g128 group_scales + sparse residuals)", flush=True)
        # K=6 activation quant is the default for BOTH Bonsai families
        # (CPU + silicon validated 2026-07-28, token-identical: 1bit −21.7%,
        # ternary −22.2%; LEVERS #24). The safe floor tracks the training
        # recipe, not weight bits — both Bonsai (post-quantized Qwen3) models
        # cliff at K5→K4, so K=6. setdefault → explicit PIM_ACT_K still wins.
        os.environ.setdefault("PIM_ACT_K", "6")
        if args.model == "bonsai_1bit":
            single_on = os.environ.get("PIM_1BIT_SINGLE", "0") == "1"
            print("[bnet] bonsai_1bit maps to DUAL-TRACK with an empty "
                  "zero-set (neg_mask = complement of pos_mask); "
                  "PIM_1BIT_SINGLE=" + ("1: V2S single-track ACTIVE — the "
                  "server computes only the pos track, client reconstructs "
                  "y = 2*y_pos - sum(x)" if single_on else "0: dual-track "
                  "(set PIM_1BIT_SINGLE=1 to halve per-request DRAM work)"),
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
                   dimm_configs=dimm_configs, weight_spec_fn=weight_spec_fn)

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
