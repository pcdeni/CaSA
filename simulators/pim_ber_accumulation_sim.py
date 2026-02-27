#!/usr/bin/env python3
# ===========================================================================
# BER Accumulation vs Network Depth — Standalone Simulation
#
# Question answered: "Our 4-layer MNIST model tolerates BER=0.1% perfectly,
# but a 30-layer transformer has more layers for errors to accumulate.
# How does tolerance degrade with depth?"
#
# Approach:
#   - Create synthetic ternary weight chains of varying depth & width
#   - Run PIM bit-serial matmul with BER injection at each layer
#   - Measure cosine similarity between clean and noisy outputs
#   - Sweep: depth x BER x width
#
# Reuses the v6 bit-serial PIM matmul approach:
#   MaxAbs INT8 quantization, uint64-packed ternary weights,
#   byte-LUT popcount, per-layer alpha scaling, BN affine, ReLU
# ===========================================================================

import os
import sys
import time
import itertools
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
SEED = 42

# ===========================================================================
# Popcount LUT (byte-level, same approach as v6)
# ===========================================================================

_POPCOUNT_TABLE = np.zeros(256, dtype=np.uint8)
for _i in range(256):
    _POPCOUNT_TABLE[_i] = bin(_i).count("1")


def popcount_uint64(arr):
    """Vectorized popcount via byte LUT on uint64 arrays."""
    x = arr.astype(np.uint64)
    result = np.zeros(arr.shape, dtype=np.int64)
    for byte_idx in range(8):
        byte_val = ((x >> np.uint64(byte_idx * 8)) & np.uint64(0xFF)).astype(np.uint8)
        result += _POPCOUNT_TABLE[byte_val].astype(np.int64)
    return result


# ===========================================================================
# Ternary weight encoding (packed uint64)
# ===========================================================================

def encode_ternary(W_int8):
    """Encode ternary weight matrix into packed uint64 (W_pos, W_neg).

    W_int8: shape (out_dim, in_dim), values in {-1, 0, +1}
    Returns: (W_pos_packed, W_neg_packed, orig_in_dim)
        packed along input dimension into uint64 words
    """
    out_dim, in_dim = W_int8.shape
    W_pos = (W_int8 == 1).astype(np.uint8)
    W_neg = (W_int8 == -1).astype(np.uint8)
    n_words = (in_dim + 63) // 64
    pad_len = n_words * 64 - in_dim
    if pad_len > 0:
        W_pos = np.pad(W_pos, ((0, 0), (0, pad_len)), constant_values=0)
        W_neg = np.pad(W_neg, ((0, 0), (0, pad_len)), constant_values=0)
    W_pos_packed = np.zeros((out_dim, n_words), dtype=np.uint64)
    W_neg_packed = np.zeros((out_dim, n_words), dtype=np.uint64)
    for w in range(n_words):
        for b in range(64):
            col = w * 64 + b
            W_pos_packed[:, w] |= W_pos[:, col].astype(np.uint64) << np.uint64(b)
            W_neg_packed[:, w] |= W_neg[:, col].astype(np.uint64) << np.uint64(b)
    return W_pos_packed, W_neg_packed, in_dim


# ===========================================================================
# Pack activation bitvector into uint64
# ===========================================================================

def pack_bitvector(x_bits, in_dim):
    """Pack a binary vector into uint64 words."""
    n_words = (in_dim + 63) // 64
    pad_len = n_words * 64 - in_dim
    if pad_len > 0:
        x_bits = np.pad(x_bits, (0, pad_len), constant_values=0)
    x_packed = np.zeros(n_words, dtype=np.uint64)
    for w in range(n_words):
        for b in range(64):
            x_packed[w] |= np.uint64(x_bits[w * 64 + b]) << np.uint64(b)
    return x_packed


# ===========================================================================
# Vectorized pack — much faster for large dimensions
# ===========================================================================

def pack_bitvector_fast(x_bits, in_dim):
    """Pack a binary vector into uint64 words using vectorized operations."""
    n_words = (in_dim + 63) // 64
    total_bits = n_words * 64
    if len(x_bits) < total_bits:
        x_padded = np.zeros(total_bits, dtype=np.uint8)
        x_padded[:len(x_bits)] = x_bits
    else:
        x_padded = x_bits[:total_bits]
    # Reshape to (n_words, 64) and compute packed via dot with bit shifts
    x_reshaped = x_padded.reshape(n_words, 64).astype(np.uint64)
    shifts = np.arange(64, dtype=np.uint64)
    # For each word: sum of bit[b] << b
    x_packed = np.zeros(n_words, dtype=np.uint64)
    for b in range(64):
        x_packed |= x_reshaped[:, b] << np.uint64(b)
    return x_packed


# ===========================================================================
# PIM bit-serial matmul with BER injection
# ===========================================================================

def pim_matmul_with_ber(W_pos_packed, W_neg_packed, x_uint8, in_dim,
                        inject_ber=0.0, rng=None):
    """Bit-serial PIM ternary matmul with optional BER injection.

    Args:
        W_pos_packed: (out_dim, n_words) uint64 — packed positive weight bits
        W_neg_packed: (out_dim, n_words) uint64 — packed negative weight bits
        x_uint8: (in_dim,) uint8 — quantized activation (unsigned part)
        in_dim: original input dimension
        inject_ber: BER in percent (0.0 = no errors)
        rng: numpy random generator for reproducible BER injection

    Returns:
        y: (out_dim,) int64 — raw matmul result
    """
    out_dim, n_words = W_pos_packed.shape
    total_padded = n_words * 64
    x_padded = np.zeros(total_padded, dtype=np.uint8)
    x_padded[:in_dim] = x_uint8

    y = np.zeros(out_dim, dtype=np.int64)

    for bit_plane in range(8):
        x_bit = ((x_padded >> bit_plane) & 1).astype(np.uint8)
        x_packed = pack_bitvector_fast(x_bit, total_padded)

        and_pos = W_pos_packed & x_packed[np.newaxis, :]
        and_neg = W_neg_packed & x_packed[np.newaxis, :]

        # BER injection: flip each bit independently with probability ber
        if inject_ber > 0.0 and rng is not None:
            ber_fraction = inject_ber / 100.0
            total_bits = out_dim * n_words * 64
            for and_arr in [and_pos, and_neg]:
                n_flips = rng.binomial(total_bits, ber_fraction)
                if n_flips > 0:
                    n_flips = min(n_flips, total_bits)
                    positions = rng.choice(total_bits, size=n_flips, replace=False)
                    rows, rem = np.divmod(positions, n_words * 64)
                    words, bits = np.divmod(rem, 64)
                    # Vectorized XOR for all flip positions
                    for idx in range(n_flips):
                        and_arr[rows[idx], words[idx]] ^= np.uint64(1) << np.uint64(bits[idx])

        pc_pos = popcount_uint64(and_pos).sum(axis=1)
        pc_neg = popcount_uint64(and_neg).sum(axis=1)
        y += (pc_pos - pc_neg) << bit_plane

    return y


# ===========================================================================
# Single-layer PIM forward pass
# ===========================================================================

def pim_layer_forward(x, W_pos_packed, W_neg_packed, in_dim, alpha,
                      bn_scale, bn_offset, apply_relu, inject_ber, rng):
    """Forward pass through one PIM layer.

    Pipeline: MaxAbs INT8 quant -> bit-serial matmul -> dequant with alpha
              -> BN affine -> ReLU (optional)
    """
    # MaxAbs INT8 quantization
    Qp = 127
    gamma = np.abs(x).max() + 1e-8
    x_int = np.clip(np.round(x / gamma * Qp), -128, 127).astype(np.int16)
    x_pos = np.clip(x_int, 0, 127).astype(np.uint8)
    x_neg = np.clip(-x_int, 0, 128).astype(np.uint8)

    # Bit-serial ternary matmul (positive and negative parts)
    y_pos = pim_matmul_with_ber(W_pos_packed, W_neg_packed, x_pos, in_dim,
                                inject_ber, rng)
    y_neg = pim_matmul_with_ber(W_pos_packed, W_neg_packed, x_neg, in_dim,
                                inject_ber, rng)
    y_int = y_pos - y_neg

    # Dequantize with alpha
    y_float = y_int.astype(np.float64) * gamma / Qp * alpha

    # RMSNorm — prevents activation explosion across deep chains
    # (real transformers use LayerNorm/RMSNorm after every layer)
    rms = np.sqrt(np.mean(y_float ** 2) + 1e-8)
    y_float = y_float / rms

    # Affine rescale (learned scale in real networks)
    y_float = y_float * bn_scale + bn_offset

    # ReLU
    if apply_relu:
        y_float = np.maximum(y_float, 0.0)

    return y_float


# ===========================================================================
# Multi-layer chain inference
# ===========================================================================

def chain_inference(x, chain_layers, inject_ber=0.0, rng=None):
    """Run inference through a chain of PIM layers.

    chain_layers: list of dicts, each with keys:
        W_pos_packed, W_neg_packed, in_dim, alpha, bn_scale, bn_offset, apply_relu
    """
    for layer in chain_layers:
        x = pim_layer_forward(
            x,
            layer["W_pos_packed"],
            layer["W_neg_packed"],
            layer["in_dim"],
            layer["alpha"],
            layer["bn_scale"],
            layer["bn_offset"],
            layer["apply_relu"],
            inject_ber,
            rng
        )
    return x


# ===========================================================================
# Synthetic layer chain construction
# ===========================================================================

def create_synthetic_chain(dim, depth, rng_weights):
    """Create a chain of synthetic ternary PIM layers.

    Args:
        dim: width of all layers (square weight matrices: dim x dim)
        depth: number of layers
        rng_weights: numpy random generator for weight generation

    Returns:
        list of layer dicts ready for chain_inference
    """
    chain = []
    for layer_idx in range(depth):
        # Generate ternary weights with realistic distribution
        # ~30% for -1, ~40% for 0, ~30% for +1
        r = rng_weights.random((dim, dim))
        W = np.zeros((dim, dim), dtype=np.int8)
        W[r < 0.30] = -1
        W[r >= 0.60] = 1
        # Middle 30% stays 0

        # Encode into packed uint64
        W_pos_packed, W_neg_packed, orig_dim = encode_ternary(W)

        # Per-layer alpha: random in realistic range [0.15, 0.35]
        alpha = rng_weights.uniform(0.15, 0.35)

        # Neutral BN affine: scale=1, offset=0
        bn_scale = np.ones(dim, dtype=np.float64)
        bn_offset = np.zeros(dim, dtype=np.float64)

        # ReLU on all layers except the last
        apply_relu = (layer_idx < depth - 1)

        chain.append({
            "W_pos_packed": W_pos_packed,
            "W_neg_packed": W_neg_packed,
            "in_dim": orig_dim,
            "alpha": alpha,
            "bn_scale": bn_scale,
            "bn_offset": bn_offset,
            "apply_relu": apply_relu,
        })

    return chain


# ===========================================================================
# Cosine similarity utility
# ===========================================================================

def cosine_similarity(a, b):
    """Cosine similarity between two vectors."""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a) + 1e-12
    norm_b = np.linalg.norm(b) + 1e-12
    return dot / (norm_a * norm_b)


# ===========================================================================
# Main simulation
# ===========================================================================

def main():
    print("=" * 72)
    print("BER ACCUMULATION vs NETWORK DEPTH — PIM Simulation")
    print("=" * 72)
    print()
    print("Question: How does BER tolerance degrade as network depth increases?")
    print("Method:   Synthetic ternary chains, bit-serial PIM matmul, BER sweep")
    print()

    # -----------------------------------------------------------------------
    # Configuration
    # -----------------------------------------------------------------------

    DEPTHS = [4, 8, 16, 30, 60]
    BER_VALUES = [0.0, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0]  # percent
    WIDTHS = [256, 512, 2560]
    N_SAMPLES = 200

    print(f"Depths   : {DEPTHS}")
    print(f"BER (%)  : {BER_VALUES}")
    print(f"Widths   : {WIDTHS}")
    print(f"Samples  : {N_SAMPLES} per configuration")
    print()

    # -----------------------------------------------------------------------
    # Pre-build all chains (deterministic weights per dim/depth pair)
    # -----------------------------------------------------------------------

    print("Building synthetic ternary layer chains...")
    t_build_start = time.time()

    chains = {}  # (dim, depth) -> chain
    for dim in WIDTHS:
        for depth in DEPTHS:
            rng_w = np.random.default_rng(seed=SEED + dim * 1000 + depth)
            chains[(dim, depth)] = create_synthetic_chain(dim, depth, rng_w)
            n_neg = sum(
                np.sum(encode_ternary(
                    np.zeros((1, 1), dtype=np.int8)  # dummy — we count from chain
                )[0] == 0) for _ in [0]  # placeholder
            )
            print(f"  dim={dim:>5d}, depth={depth:>3d}: "
                  f"{depth} layers, alpha range "
                  f"[{min(l['alpha'] for l in chains[(dim, depth)]):.3f}, "
                  f"{max(l['alpha'] for l in chains[(dim, depth)]):.3f}]")

    print(f"Chain construction: {time.time() - t_build_start:.1f}s")
    print()

    # -----------------------------------------------------------------------
    # Generate random input vectors (one set per dim, reused across configs)
    # -----------------------------------------------------------------------

    print("Generating random input vectors...")
    input_vectors = {}  # dim -> (N_SAMPLES, dim) float64 array
    rng_inputs = np.random.default_rng(seed=SEED + 9999)
    for dim in WIDTHS:
        # Random inputs with realistic post-ReLU distribution:
        # mostly positive (like after a ReLU), some structure
        raw = rng_inputs.standard_normal((N_SAMPLES, dim))
        # Simulate post-ReLU by shifting and clipping
        raw = raw * 0.5 + 0.3
        raw = np.maximum(raw, 0.0)  # ReLU-like
        input_vectors[dim] = raw.astype(np.float64)
    print(f"  Generated {N_SAMPLES} vectors for each of {WIDTHS}")
    print()

    # -----------------------------------------------------------------------
    # Main sweep: for each (width, depth, ber), run N_SAMPLES and collect
    # cosine similarity between clean and noisy outputs
    # -----------------------------------------------------------------------

    # Results storage: results[(dim, depth, ber)] = (mean_cos_sim, std_cos_sim)
    results = {}

    total_configs = len(WIDTHS) * len(DEPTHS) * len(BER_VALUES)
    config_idx = 0
    t_sweep_start = time.time()

    print("=" * 72)
    print("RUNNING BER SWEEP")
    print("=" * 72)

    for dim in WIDTHS:
        print(f"\n--- Width = {dim} ---")

        for depth in DEPTHS:
            chain = chains[(dim, depth)]
            inputs = input_vectors[dim]

            # First, compute clean outputs (BER=0) for this chain
            clean_outputs = []
            t0 = time.time()
            for s in range(N_SAMPLES):
                out = chain_inference(inputs[s], chain, inject_ber=0.0, rng=None)
                clean_outputs.append(out)
            clean_outputs = np.array(clean_outputs)
            t_clean = time.time() - t0

            config_idx += 1
            results[(dim, depth, 0.0)] = (1.0, 0.0)  # BER=0 is trivially cos_sim=1.0
            print(f"  depth={depth:>3d}, BER=0.000%: cos_sim=1.000000+/-0.000000 "
                  f"({t_clean:.1f}s) [{config_idx}/{total_configs}]")

            for ber in BER_VALUES:
                if ber == 0.0:
                    continue  # already handled

                config_idx += 1
                t0 = time.time()
                rng_ber = np.random.default_rng(seed=SEED + int(ber * 100000) + depth * 100 + dim)

                cos_sims = []
                for s in range(N_SAMPLES):
                    noisy_out = chain_inference(inputs[s], chain,
                                               inject_ber=ber, rng=rng_ber)
                    cs = cosine_similarity(clean_outputs[s], noisy_out)
                    cos_sims.append(cs)

                mean_cs = np.mean(cos_sims)
                std_cs = np.std(cos_sims)
                results[(dim, depth, ber)] = (mean_cs, std_cs)
                elapsed = time.time() - t0
                print(f"  depth={depth:>3d}, BER={ber:.3f}%: "
                      f"cos_sim={mean_cs:.6f}+/-{std_cs:.6f} "
                      f"({elapsed:.1f}s) [{config_idx}/{total_configs}]")

    total_sweep_time = time.time() - t_sweep_start
    print(f"\nTotal sweep time: {total_sweep_time:.1f}s")
    print()

    # -----------------------------------------------------------------------
    # Results tables
    # -----------------------------------------------------------------------

    for dim in WIDTHS:
        print("=" * 72)
        print(f"RESULTS TABLE: Width = {dim}")
        print("=" * 72)

        # Header
        header = f"{'Depth':>8s}"
        for ber in BER_VALUES:
            header += f" | BER={ber:.3f}%"
        print(header)
        print("-" * len(header))

        for depth in DEPTHS:
            row = f"{depth:>8d}"
            for ber in BER_VALUES:
                mean_cs, std_cs = results[(dim, depth, ber)]
                row += f" |   {mean_cs:.4f}  "
            print(row)
        print()

    # -----------------------------------------------------------------------
    # Critical BER table: for each depth/width, what BER causes cos_sim
    # to drop below 0.95, 0.90, 0.80
    # -----------------------------------------------------------------------

    print("=" * 72)
    print("CRITICAL BER TABLE")
    print("  For each (width, depth): BER (%) at which cos_sim drops below threshold")
    print("=" * 72)

    thresholds = [0.95, 0.90, 0.80]

    for dim in WIDTHS:
        print(f"\n  Width = {dim}:")
        header = f"  {'Depth':>6s}"
        for th in thresholds:
            header += f" | cos<{th:.2f}"
        print(header)
        print("  " + "-" * (len(header) - 2))

        for depth in DEPTHS:
            row = f"  {depth:>6d}"
            for th in thresholds:
                critical_ber = None
                for ber in BER_VALUES:
                    if ber == 0.0:
                        continue
                    mean_cs, _ = results[(dim, depth, ber)]
                    if mean_cs < th:
                        critical_ber = ber
                        break
                if critical_ber is not None:
                    row += f" | {critical_ber:>7.3f}%"
                else:
                    row += f" |    >1.0%"
            print(row)
    print()

    # -----------------------------------------------------------------------
    # Plot 1: Cosine similarity vs BER for each depth (one subplot per width)
    # -----------------------------------------------------------------------

    print("Generating plots...")

    n_widths = len(WIDTHS)
    fig1, axes1 = plt.subplots(1, n_widths, figsize=(6 * n_widths, 5), squeeze=False)
    fig1.suptitle("BER Tolerance vs Network Depth — Cosine Similarity", fontsize=14)

    depth_colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(DEPTHS)))

    for w_idx, dim in enumerate(WIDTHS):
        ax = axes1[0, w_idx]
        for d_idx, depth in enumerate(DEPTHS):
            ber_list = []
            cs_mean_list = []
            cs_std_list = []
            for ber in BER_VALUES:
                mean_cs, std_cs = results[(dim, depth, ber)]
                ber_list.append(ber if ber > 0 else 1e-5)  # small offset for log scale
                cs_mean_list.append(mean_cs)
                cs_std_list.append(std_cs)

            ber_arr = np.array(ber_list)
            cs_arr = np.array(cs_mean_list)
            std_arr = np.array(cs_std_list)

            ax.plot(ber_arr, cs_arr, "o-", color=depth_colors[d_idx],
                    label=f"depth={depth}", markersize=4, linewidth=1.5)
            ax.fill_between(ber_arr, cs_arr - std_arr, np.minimum(cs_arr + std_arr, 1.0),
                            color=depth_colors[d_idx], alpha=0.15)

        ax.set_xscale("log")
        ax.set_xlabel("BER (%)", fontsize=11)
        ax.set_ylabel("Cosine Similarity", fontsize=11)
        ax.set_title(f"Width = {dim}", fontsize=12)
        ax.axhline(y=0.95, color="red", linestyle="--", alpha=0.4, linewidth=0.8)
        ax.axhline(y=0.90, color="orange", linestyle="--", alpha=0.4, linewidth=0.8)
        ax.axhline(y=0.80, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="lower left")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plot1_path = os.path.join(OUTPUT_DIR, "pim_ber_accumulation.png")
    fig1.savefig(plot1_path, dpi=150)
    print(f"  Saved: {plot1_path}")

    # -----------------------------------------------------------------------
    # Plot 2: Cosine similarity vs depth for selected BER values
    # -----------------------------------------------------------------------

    selected_bers = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]
    ber_colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(selected_bers)))

    fig2, axes2 = plt.subplots(1, n_widths, figsize=(6 * n_widths, 5), squeeze=False)
    fig2.suptitle("BER Tolerance vs Network Depth — Depth Scaling", fontsize=14)

    for w_idx, dim in enumerate(WIDTHS):
        ax = axes2[0, w_idx]
        for b_idx, ber in enumerate(selected_bers):
            depth_list = []
            cs_mean_list = []
            cs_std_list = []
            for depth in DEPTHS:
                mean_cs, std_cs = results[(dim, depth, ber)]
                depth_list.append(depth)
                cs_mean_list.append(mean_cs)
                cs_std_list.append(std_cs)

            depth_arr = np.array(depth_list)
            cs_arr = np.array(cs_mean_list)
            std_arr = np.array(cs_std_list)

            ax.plot(depth_arr, cs_arr, "s-", color=ber_colors[b_idx],
                    label=f"BER={ber}%", markersize=5, linewidth=1.5)
            ax.fill_between(depth_arr, cs_arr - std_arr, np.minimum(cs_arr + std_arr, 1.0),
                            color=ber_colors[b_idx], alpha=0.12)

        ax.set_xlabel("Network Depth (layers)", fontsize=11)
        ax.set_ylabel("Cosine Similarity", fontsize=11)
        ax.set_title(f"Width = {dim}", fontsize=12)
        ax.axhline(y=0.95, color="red", linestyle="--", alpha=0.4, linewidth=0.8)
        ax.axhline(y=0.90, color="orange", linestyle="--", alpha=0.4, linewidth=0.8)
        ax.axhline(y=0.80, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks(DEPTHS)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="lower left")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plot2_path = os.path.join(OUTPUT_DIR, "pim_ber_vs_depth.png")
    fig2.savefig(plot2_path, dpi=150)
    print(f"  Saved: {plot2_path}")

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------

    print()
    print("=" * 72)
    print("SIMULATION COMPLETE")
    print("=" * 72)
    print()
    print("Key findings to look for:")
    print("  1. At what BER does cos_sim start degrading for shallow (4L) vs deep (60L)?")
    print("  2. Does wider network (2560) tolerate more BER than narrow (256)?")
    print("  3. Is the degradation gradual or cliff-like?")
    print("  4. What is the maximum depth that can tolerate BER=0.1% (our MNIST sweet spot)?")
    print()
    print(f"Output plots:")
    print(f"  {plot1_path}")
    print(f"  {plot2_path}")
    print()


if __name__ == "__main__":
    main()
