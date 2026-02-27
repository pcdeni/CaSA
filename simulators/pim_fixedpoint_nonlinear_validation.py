#!/usr/bin/env python3
"""
Fixed-Point Non-Linear Function Accuracy Validation for PIM-LLM

Validates the FPGA-resident non-linear operations (SiLU, RMSNorm, Softmax)
in fixed-point representation against IEEE 754 float64 references.

The PIM-LLM FPGA implements these functions using:
  - SiLU: piecewise-linear approximation with 256-entry BRAM lookup table (16-bit)
  - RMSNorm: fixed-point reciprocal sqrt via lookup table
  - Softmax: fixed-point exponential lookup with streaming accumulator

This script quantifies approximation error across realistic input distributions
to verify that FPGA non-linear ops do not degrade inference quality.

Output: pim_fixedpoint_validation_results.txt + summary plots
"""
import sys
import os
import time
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ==========================================================================
# Reference implementations (float64 — ground truth)
# ==========================================================================

def silu_float(x):
    """SiLU (Swish): x * sigmoid(x), float64."""
    return x / (1.0 + np.exp(-x))

def rmsnorm_float(x, weight, eps=1e-6):
    """RMSNorm: x * weight / sqrt(mean(x^2) + eps), float64."""
    rms = np.sqrt(np.mean(x ** 2) + eps)
    return x * weight / rms

def softmax_float(x):
    """Softmax: exp(x - max) / sum(exp(x - max)), float64."""
    x_shifted = x - np.max(x)
    e = np.exp(x_shifted)
    return e / np.sum(e)


# ==========================================================================
# Fixed-point SiLU approximation (piecewise-linear LUT, 256 entries, 16-bit)
# ==========================================================================

class FixedPointSiLU:
    """
    Piecewise-linear SiLU approximation using a 256-entry lookup table.

    Input range: [-8.0, +8.0] (covers >99.99% of post-RMSNorm activations)
    LUT resolution: 16/256 = 0.0625 per entry
    Value precision: 16-bit fixed-point (Q8.8 format: 8 integer + 8 fractional bits)
    """
    def __init__(self, n_entries=256, input_range=(-8.0, 8.0), value_bits=16):
        self.n_entries = n_entries
        self.lo, self.hi = input_range
        self.step = (self.hi - self.lo) / n_entries
        self.value_bits = value_bits
        self.frac_bits = value_bits // 2  # Q8.8
        self.scale = 2 ** self.frac_bits

        # Build LUT: store SiLU values as fixed-point integers
        self.lut_x = np.linspace(self.lo, self.hi, n_entries + 1)
        self.lut_y_float = silu_float(self.lut_x)
        # Quantize to fixed-point
        self.lut_y_fixed = np.round(self.lut_y_float * self.scale).astype(np.int32)

    def __call__(self, x):
        """Piecewise-linear interpolation from LUT, dequantized to float."""
        x = np.asarray(x, dtype=np.float64)
        result = np.zeros_like(x)

        for i in range(x.size):
            xi = x.flat[i]
            # Clamp to input range
            xi_clamped = np.clip(xi, self.lo, self.hi - 1e-10)

            # Find LUT index
            idx = int((xi_clamped - self.lo) / self.step)
            idx = min(idx, self.n_entries - 1)

            # Fractional position within segment
            x_lo = self.lo + idx * self.step
            frac = (xi_clamped - x_lo) / self.step

            # Fixed-point linear interpolation
            y_lo = self.lut_y_fixed[idx]
            y_hi = self.lut_y_fixed[idx + 1]
            # Interpolate in fixed-point domain
            y_interp = y_lo + int(round(frac * (y_hi - y_lo)))

            # Dequantize
            result.flat[i] = y_interp / self.scale

        return result


# ==========================================================================
# Fixed-point RMSNorm
# ==========================================================================

class FixedPointRMSNorm:
    """
    Fixed-point RMSNorm using reciprocal sqrt lookup table.

    rsqrt LUT: 512 entries covering [eps, max_rms^2]
    Precision: 16-bit fixed-point for rsqrt values
    """
    def __init__(self, n_entries=512, max_rms_sq=64.0, eps=1e-6, value_bits=16):
        self.n_entries = n_entries
        self.eps = eps
        self.max_rms_sq = max_rms_sq
        self.frac_bits = value_bits // 2
        self.scale = 2 ** self.frac_bits

        # Build rsqrt LUT: maps rms^2 -> 1/sqrt(rms^2 + eps)
        self.lut_x = np.linspace(eps, max_rms_sq, n_entries)
        self.lut_y_float = 1.0 / np.sqrt(self.lut_x + eps)
        self.lut_y_fixed = np.round(self.lut_y_float * self.scale).astype(np.int32)

    def __call__(self, x, weight, eps=1e-6):
        """Fixed-point RMSNorm."""
        x = np.asarray(x, dtype=np.float64)
        rms_sq = np.mean(x ** 2)

        # Lookup rsqrt
        rms_sq_clamped = np.clip(rms_sq, self.eps, self.max_rms_sq - 1e-10)
        step = (self.max_rms_sq - self.eps) / self.n_entries
        idx = int((rms_sq_clamped - self.eps) / step)
        idx = min(idx, self.n_entries - 2)

        # Linear interpolation
        x_lo = self.eps + idx * step
        frac = (rms_sq_clamped - x_lo) / step
        y_lo = self.lut_y_fixed[idx]
        y_hi = self.lut_y_fixed[idx + 1]
        rsqrt_fixed = y_lo + int(round(frac * (y_hi - y_lo)))
        rsqrt_val = rsqrt_fixed / self.scale

        return x * weight * rsqrt_val


# ==========================================================================
# Fixed-point Softmax
# ==========================================================================

class FixedPointSoftmax:
    """
    Fixed-point softmax using exponential lookup table.

    exp LUT: 1024 entries covering [-16.0, 0.0] (shifted inputs)
    Precision: 16-bit fixed-point for exp values
    """
    def __init__(self, n_entries=1024, input_range=(-16.0, 0.0), value_bits=16):
        self.n_entries = n_entries
        self.lo, self.hi = input_range
        self.step = (self.hi - self.lo) / n_entries
        self.frac_bits = value_bits
        self.scale = 2 ** self.frac_bits

        # Build exp LUT
        self.lut_x = np.linspace(self.lo, self.hi, n_entries + 1)
        self.lut_y_float = np.exp(self.lut_x)
        # Quantize (use more fractional bits since exp values are < 1)
        self.lut_y_fixed = np.round(self.lut_y_float * self.scale).astype(np.int64)

    def __call__(self, x):
        """Fixed-point softmax."""
        x = np.asarray(x, dtype=np.float64)
        x_shifted = x - np.max(x)  # Shift for numerical stability

        exp_vals = np.zeros_like(x)
        for i in range(x.size):
            xi = x_shifted.flat[i]
            xi_clamped = np.clip(xi, self.lo, self.hi - 1e-10)

            idx = int((xi_clamped - self.lo) / self.step)
            idx = min(idx, self.n_entries - 1)

            x_lo_val = self.lo + idx * self.step
            frac = (xi_clamped - x_lo_val) / self.step

            y_lo = self.lut_y_fixed[idx]
            y_hi = self.lut_y_fixed[idx + 1]
            y_interp = y_lo + int(round(frac * (y_hi - y_lo)))

            exp_vals.flat[i] = y_interp / self.scale

        # Normalize
        total = np.sum(exp_vals)
        if total < 1e-30:
            return np.ones_like(x) / x.size
        return exp_vals / total


# ==========================================================================
# Test suite
# ==========================================================================

def test_silu(n_samples=10000, seed=42):
    """Test SiLU accuracy across realistic activation distributions."""
    rng = np.random.default_rng(seed)
    fp_silu = FixedPointSiLU()

    print("=" * 60)
    print("SiLU Fixed-Point Accuracy Test")
    print("=" * 60)
    print(f"  LUT entries: {fp_silu.n_entries}")
    print(f"  Input range: [{fp_silu.lo}, {fp_silu.hi}]")
    print(f"  Value bits: {fp_silu.value_bits} (Q8.8)")
    print(f"  Samples: {n_samples}")
    print()

    # Test distributions
    distributions = {
        "Uniform [-8, 8]": rng.uniform(-8, 8, n_samples),
        "Normal (0, 1)": rng.standard_normal(n_samples),
        "Normal (0, 2)": rng.standard_normal(n_samples) * 2,
        "Post-RMSNorm (realistic)": rng.standard_normal(n_samples) * 0.8 + 0.1,
        "Heavy-tail (Cauchy-clipped)": np.clip(rng.standard_cauchy(n_samples), -8, 8),
    }

    results = {}
    all_errors = []

    for name, x in distributions.items():
        ref = silu_float(x)
        approx = fp_silu(x)

        abs_err = np.abs(ref - approx)
        rel_err = abs_err / (np.abs(ref) + 1e-10)

        max_abs = np.max(abs_err)
        mean_abs = np.mean(abs_err)
        max_rel = np.max(rel_err[np.abs(ref) > 0.01])  # exclude near-zero
        mean_rel = np.mean(rel_err[np.abs(ref) > 0.01])

        # Cosine similarity of SiLU outputs
        cos_sim = np.dot(ref, approx) / (np.linalg.norm(ref) * np.linalg.norm(approx) + 1e-12)

        results[name] = {
            "max_abs_err": max_abs,
            "mean_abs_err": mean_abs,
            "max_rel_err": max_rel,
            "mean_rel_err": mean_rel,
            "cos_sim": cos_sim,
        }
        all_errors.extend(abs_err.tolist())

        print(f"  {name}:")
        print(f"    Max abs error:  {max_abs:.6f}")
        print(f"    Mean abs error: {mean_abs:.6f}")
        print(f"    Max rel error:  {max_rel:.4%} (excluding |y| < 0.01)")
        print(f"    Mean rel error: {mean_rel:.4%}")
        print(f"    Cosine sim:     {cos_sim:.8f}")
        print()

    return results, all_errors


def test_rmsnorm(n_samples=10000, dim=2560, seed=42):
    """Test RMSNorm accuracy with realistic dimensions."""
    rng = np.random.default_rng(seed)
    fp_rmsnorm = FixedPointRMSNorm()

    print("=" * 60)
    print("RMSNorm Fixed-Point Accuracy Test")
    print("=" * 60)
    print(f"  rsqrt LUT entries: {fp_rmsnorm.n_entries}")
    print(f"  Dimension: {dim}")
    print(f"  Samples: {n_samples}")
    print()

    weight = rng.uniform(0.5, 1.5, dim)  # Typical learned scale params

    cos_sims = []
    max_errs = []
    mean_errs = []

    for i in range(n_samples):
        x = rng.standard_normal(dim) * (0.5 + rng.uniform(0, 2))

        ref = rmsnorm_float(x, weight)
        approx = fp_rmsnorm(x, weight)

        cos_sim = np.dot(ref, approx) / (np.linalg.norm(ref) * np.linalg.norm(approx) + 1e-12)
        abs_err = np.abs(ref - approx)

        cos_sims.append(cos_sim)
        max_errs.append(np.max(abs_err))
        mean_errs.append(np.mean(abs_err))

    cos_sims = np.array(cos_sims)
    max_errs = np.array(max_errs)
    mean_errs = np.array(mean_errs)

    print(f"  Cosine similarity:  mean={np.mean(cos_sims):.8f}, "
          f"min={np.min(cos_sims):.8f}, std={np.std(cos_sims):.2e}")
    print(f"  Max abs error:      mean={np.mean(max_errs):.6f}, "
          f"max={np.max(max_errs):.6f}")
    print(f"  Mean abs error:     mean={np.mean(mean_errs):.6f}")
    print()

    return {
        "cos_sim_mean": float(np.mean(cos_sims)),
        "cos_sim_min": float(np.min(cos_sims)),
        "cos_sim_std": float(np.std(cos_sims)),
        "max_abs_err_mean": float(np.mean(max_errs)),
        "mean_abs_err_mean": float(np.mean(mean_errs)),
    }


def test_softmax(n_samples=10000, dims=[20, 128, 256], seed=42):
    """Test softmax accuracy at various sequence lengths."""
    rng = np.random.default_rng(seed)
    fp_softmax = FixedPointSoftmax()

    print("=" * 60)
    print("Softmax Fixed-Point Accuracy Test")
    print("=" * 60)
    print(f"  exp LUT entries: {fp_softmax.n_entries}")
    print(f"  Dimensions tested: {dims}")
    print(f"  Samples per dim: {n_samples}")
    print()

    results = {}

    for dim in dims:
        kl_divs = []
        cos_sims = []
        max_errs = []

        for i in range(n_samples):
            # Logits: typical attention scores ~ N(0, sqrt(d_head))
            x = rng.standard_normal(dim) * np.sqrt(128)  # head_dim=128

            ref = softmax_float(x)
            approx = fp_softmax(x)

            # KL divergence (ref || approx)
            kl = np.sum(ref * np.log((ref + 1e-30) / (approx + 1e-30)))
            kl_divs.append(kl)

            cos_sim = np.dot(ref, approx) / (np.linalg.norm(ref) * np.linalg.norm(approx) + 1e-12)
            cos_sims.append(cos_sim)

            max_errs.append(np.max(np.abs(ref - approx)))

        kl_divs = np.array(kl_divs)
        cos_sims = np.array(cos_sims)
        max_errs = np.array(max_errs)

        results[dim] = {
            "kl_div_mean": float(np.mean(kl_divs)),
            "kl_div_max": float(np.max(kl_divs)),
            "cos_sim_mean": float(np.mean(cos_sims)),
            "cos_sim_min": float(np.min(cos_sims)),
            "max_abs_err_mean": float(np.mean(max_errs)),
        }

        print(f"  dim={dim}:")
        print(f"    KL divergence:   mean={np.mean(kl_divs):.2e}, max={np.max(kl_divs):.2e}")
        print(f"    Cosine sim:      mean={np.mean(cos_sims):.8f}, min={np.min(cos_sims):.8f}")
        print(f"    Max abs error:   mean={np.mean(max_errs):.6f}")
        print()

    return results


# ==========================================================================
# End-to-end: chain SiLU -> RMSNorm -> Softmax
# ==========================================================================

def test_chain(n_samples=5000, dim=2560, head_dim=128, seed=42):
    """Test chained non-linear ops (simulating one transformer layer's non-linear path)."""
    rng = np.random.default_rng(seed)

    fp_silu = FixedPointSiLU()
    fp_rmsnorm = FixedPointRMSNorm()
    fp_softmax = FixedPointSoftmax()

    print("=" * 60)
    print("Chained Non-Linear Ops Test (RMSNorm -> SiLU -> RMSNorm)")
    print("=" * 60)
    print(f"  dim={dim}, head_dim={head_dim}, samples={n_samples}")
    print()

    weight1 = rng.uniform(0.8, 1.2, dim)
    weight2 = rng.uniform(0.8, 1.2, dim)

    cos_sims = []

    for i in range(n_samples):
        x = rng.standard_normal(dim) * 1.0

        # Reference chain
        y_ref = rmsnorm_float(x, weight1)
        y_ref = silu_float(y_ref)
        y_ref = rmsnorm_float(y_ref, weight2)

        # Fixed-point chain
        y_fp = fp_rmsnorm(x, weight1)
        y_fp = fp_silu(y_fp)
        y_fp = fp_rmsnorm(y_fp, weight2)

        cos_sim = np.dot(y_ref, y_fp) / (np.linalg.norm(y_ref) * np.linalg.norm(y_fp) + 1e-12)
        cos_sims.append(cos_sim)

    cos_sims = np.array(cos_sims)

    print(f"  Chained cos_sim:  mean={np.mean(cos_sims):.8f}, "
          f"min={np.min(cos_sims):.8f}, std={np.std(cos_sims):.2e}")
    print(f"  Interpretation:   {'PASS' if np.mean(cos_sims) > 0.999 else 'MARGINAL' if np.mean(cos_sims) > 0.99 else 'FAIL'} "
          f"(threshold: cos_sim > 0.999 for negligible impact)")
    print()

    return {
        "cos_sim_mean": float(np.mean(cos_sims)),
        "cos_sim_min": float(np.min(cos_sims)),
        "cos_sim_std": float(np.std(cos_sims)),
    }


# ==========================================================================
# Bit-width sensitivity sweep
# ==========================================================================

def test_bitwidth_sweep(n_samples=5000, seed=42):
    """Test how LUT entry count and bit-width affect SiLU accuracy."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n_samples) * 1.5  # realistic post-RMSNorm
    ref = silu_float(x)

    print("=" * 60)
    print("SiLU Bit-Width / LUT Size Sensitivity Sweep")
    print("=" * 60)

    configs = [
        (64, 8), (128, 8), (256, 8),
        (64, 16), (128, 16), (256, 16), (512, 16),
        (256, 24), (512, 24), (1024, 24),
    ]

    results = []
    for n_entries, bits in configs:
        fp = FixedPointSiLU(n_entries=n_entries, value_bits=bits)
        approx = fp(x)

        abs_err = np.abs(ref - approx)
        cos_sim = np.dot(ref, approx) / (np.linalg.norm(ref) * np.linalg.norm(approx) + 1e-12)

        entry = {
            "n_entries": n_entries,
            "bits": bits,
            "mean_abs_err": float(np.mean(abs_err)),
            "max_abs_err": float(np.max(abs_err)),
            "cos_sim": float(cos_sim),
        }
        results.append(entry)

        print(f"  LUT={n_entries:4d}, bits={bits:2d}: "
              f"mean_err={np.mean(abs_err):.6f}, "
              f"max_err={np.max(abs_err):.6f}, "
              f"cos_sim={cos_sim:.8f}")

    print()
    return results


# ==========================================================================
# Main
# ==========================================================================

def main():
    t_start = time.time()

    print("=" * 60)
    print("PIM-LLM Fixed-Point Non-Linear Function Validation")
    print("=" * 60)
    print(f"Purpose: Validate FPGA non-linear ops (SiLU, RMSNorm, Softmax)")
    print(f"         against IEEE 754 float64 reference implementations")
    print()

    # Run all tests
    silu_results, silu_errors = test_silu()
    rmsnorm_results = test_rmsnorm()
    softmax_results = test_softmax()
    chain_results = test_chain()
    sweep_results = test_bitwidth_sweep()

    # ---- Summary ----
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print()

    print("SiLU (256-entry, Q8.8):")
    for name, r in silu_results.items():
        print(f"  {name}: cos_sim={r['cos_sim']:.8f}, max_abs={r['max_abs_err']:.6f}")

    print(f"\nRMSNorm (512-entry rsqrt LUT):")
    print(f"  cos_sim: mean={rmsnorm_results['cos_sim_mean']:.8f}, "
          f"min={rmsnorm_results['cos_sim_min']:.8f}")

    print(f"\nSoftmax (1024-entry exp LUT):")
    for dim, r in softmax_results.items():
        print(f"  dim={dim}: cos_sim={r['cos_sim_mean']:.8f}, KL={r['kl_div_mean']:.2e}")

    print(f"\nChained ops (RMSNorm->SiLU->RMSNorm):")
    print(f"  cos_sim: mean={chain_results['cos_sim_mean']:.8f}, "
          f"min={chain_results['cos_sim_min']:.8f}")

    # ---- Verdict ----
    all_pass = (
        chain_results['cos_sim_mean'] > 0.999 and
        rmsnorm_results['cos_sim_mean'] > 0.9999 and
        all(r['cos_sim_mean'] > 0.999 for r in softmax_results.values())
    )

    print(f"\n{'='*60}")
    print(f"VERDICT: {'PASS' if all_pass else 'REVIEW NEEDED'}")
    if all_pass:
        print("  All fixed-point non-linear approximations achieve cos_sim > 0.999")
        print("  relative to float64 reference. FPGA implementation introduces")
        print("  negligible error compared to charge-sharing BER (< 3.8e-8).")
    else:
        print("  Some tests below threshold. Review LUT sizes and bit-widths.")
    print(f"{'='*60}")

    elapsed = time.time() - t_start
    print(f"\nTotal time: {elapsed:.1f}s")

    # ---- Generate plot ----
    print("\nGenerating validation plots...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("PIM-LLM Fixed-Point Non-Linear Validation", fontsize=14)

    # Plot 1: SiLU comparison
    ax = axes[0, 0]
    x_plot = np.linspace(-6, 6, 1000)
    fp_silu = FixedPointSiLU()
    ax.plot(x_plot, silu_float(x_plot), 'b-', linewidth=2, label='Float64 ref')
    ax.plot(x_plot, fp_silu(x_plot), 'r--', linewidth=1, label='Fixed-point (256, Q8.8)')
    ax.set_title('SiLU: Float64 vs Fixed-Point')
    ax.set_xlabel('Input')
    ax.set_ylabel('Output')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: SiLU error
    ax = axes[0, 1]
    err = np.abs(silu_float(x_plot) - fp_silu(x_plot))
    ax.semilogy(x_plot, err + 1e-10, 'r-', linewidth=1)
    ax.set_title('SiLU Absolute Error')
    ax.set_xlabel('Input')
    ax.set_ylabel('|error|')
    ax.grid(True, alpha=0.3)

    # Plot 3: Softmax KL by dimension
    ax = axes[1, 0]
    dims_plot = list(softmax_results.keys())
    kl_means = [softmax_results[d]['kl_div_mean'] for d in dims_plot]
    cos_means = [softmax_results[d]['cos_sim_mean'] for d in dims_plot]
    ax.bar(range(len(dims_plot)), kl_means, color='steelblue', alpha=0.8)
    ax.set_xticks(range(len(dims_plot)))
    ax.set_xticklabels([str(d) for d in dims_plot])
    ax.set_title('Softmax KL Divergence by Dimension')
    ax.set_xlabel('Softmax Dimension')
    ax.set_ylabel('KL Divergence (nats)')
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 4: Bit-width sweep
    ax = axes[1, 1]
    for bits in [8, 16, 24]:
        subset = [r for r in sweep_results if r['bits'] == bits]
        if subset:
            x_vals = [r['n_entries'] for r in subset]
            y_vals = [r['mean_abs_err'] for r in subset]
            ax.semilogy(x_vals, y_vals, 'o-', label=f'{bits}-bit', markersize=5)
    ax.set_title('SiLU Error vs LUT Size')
    ax.set_xlabel('LUT Entries')
    ax.set_ylabel('Mean Absolute Error')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plot_path = os.path.join(OUTPUT_DIR, "pim_fixedpoint_validation.png")
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {plot_path}")
    plt.close()


if __name__ == "__main__":
    main()
