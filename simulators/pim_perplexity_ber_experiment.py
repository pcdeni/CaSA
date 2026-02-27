"""
PIM-LLM: BitNet Perplexity Under BER Injection Experiment
==========================================================

This experiment measures how charge-sharing bit errors (BER) affect
the actual text quality of BitNet b1.58-2B-4T, the target LLM for PIM-LLM.

Methodology:
  1. Load the real BitNet model from HuggingFace cache
  2. Evaluate baseline perplexity on WikiText-2 (standard LLM benchmark)
  3. Hook into each linear layer's weight-activation matmul
  4. Inject random bit flips at various BER levels into the binary AND
     results (simulating charge-sharing errors in DRAM)
  5. Measure perplexity degradation vs BER

The bit flip injection simulates what happens in PIM hardware:
  - Ternary weights are encoded as W_pos/W_neg binary rows
  - INT8 activations are decomposed into 8 bit-planes
  - The AND of weight-row and activation-bit-plane produces a result
  - BER = probability that any result bit is flipped

We inject errors into the matmul OUTPUT (post-accumulation) rather than
at the individual bit-plane level, using the mathematical equivalence:
  - N independent AND operations each with BER p
  - The accumulated popcount result has variance ~ N*p*(1-p)
  - We model this as additive Gaussian noise on the matmul output,
    calibrated to match the per-bit BER's effect on the final sum

Author: PIM-LLM Project
Date: February 2026
"""

import torch
import torch.nn as nn
import numpy as np
import time
import sys
import os
import json
import math
from collections import defaultdict

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_NAME = "microsoft/BitNet-b1.58-2B-4T"
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bitnet_cache")

# BER levels to test (probability of a single bit flip in AND result)
BER_LEVELS = [0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]

# How many tokens of WikiText-2 to evaluate (full test set is ~240K tokens,
# we use a subset for CPU feasibility). 1024 tokens ≈ 4 sequences of 256.
MAX_EVAL_TOKENS = 1024
SEQ_LEN = 256  # context window for perplexity calculation

# Number of runs per BER level for confidence intervals
NUM_RUNS = 1  # single run for CPU speed; increase to 3 for tighter CI

# Seed for reproducibility
SEED = 42

# Output
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# BER → matmul noise model
# ---------------------------------------------------------------------------
def ber_to_matmul_noise_std(ber, input_dim, num_bitplanes=8):
    """
    Convert per-bit BER to standard deviation of noise on the matmul output.

    In PIM-LLM's bit-serial protocol:
      - Each output neuron computes: sum_i(w_i * x_i) where w_i in {-1,0,+1}
      - This is decomposed as: sum over 8 bitplanes of popcount(W_pos AND act_b) - popcount(W_neg AND act_b)
      - Each AND result bit has probability `ber` of being flipped

    For a single AND between a weight row and activation bitplane:
      - The popcount of a d-dimensional AND result with BER p has:
        - Expected popcount change: each '1' bit has prob p of flipping to 0,
          each '0' bit has prob p of flipping to 1
        - Net effect on popcount: ~ Binomial noise with variance d * p * (1-p)
        - Std dev of popcount error ≈ sqrt(d * p * (1-p))

    Over 8 bitplanes with power-of-2 weighting (1, 2, 4, ..., 128):
      - Total variance = sum_{b=0}^{7} (2^b)^2 * d * p * (1-p)
                       = d * p * (1-p) * sum(4^b for b=0..7)
                       = d * p * (1-p) * (4^8 - 1) / 3
                       = d * p * (1-p) * 21845

    For both W_pos and W_neg (independent):
      - Total variance doubles: 2 * d * p * (1-p) * 21845

    Std dev of matmul output noise per neuron:
      std = sqrt(2 * input_dim * ber * (1-ber) * 21845)

    But we need to scale relative to the typical matmul output magnitude.
    With ternary weights (~58% nonzero) and INT8 activations (mean ~64):
      typical |output| ≈ sqrt(d * 0.58) * 64 ≈ sqrt(d) * 49

    We inject noise = N(0, std) directly onto the matmul output.
    """
    if ber == 0:
        return 0.0

    # Variance from bit-serial accumulation with BER
    bitplane_weight_sum = sum(4**b for b in range(num_bitplanes))  # 21845 for 8 bits
    variance_per_neuron = 2 * input_dim * ber * (1 - ber) * bitplane_weight_sum

    return math.sqrt(variance_per_neuron)


# ---------------------------------------------------------------------------
# Hook-based noise injection
# ---------------------------------------------------------------------------
class BERInjector:
    """Injects BER-calibrated noise into linear layer outputs."""

    def __init__(self, ber=0.0):
        self.ber = ber
        self.hooks = []
        self.noise_stats = defaultdict(list)

    def _hook_fn(self, module, input, output, name=""):
        if self.ber == 0:
            return output

        # Get input dimension (fan-in of this linear layer)
        if hasattr(module, 'in_features'):
            input_dim = module.in_features
        elif hasattr(module, 'weight'):
            input_dim = module.weight.shape[1]
        else:
            input_dim = output.shape[-1]  # fallback

        noise_std = ber_to_matmul_noise_std(self.ber, input_dim)

        # Scale noise relative to output magnitude
        # The noise_std above is in "popcount units"; we need to relate
        # to the actual output scale of this layer
        #
        # In PIM, the matmul output = alpha * (popcount_pos - popcount_neg)
        # where alpha = mean(|weights|). For ternary weights with 58% nonzero,
        # alpha ≈ 0.58. The popcount values are integers in [0, input_dim].
        #
        # The HuggingFace model computes the matmul in float, so outputs
        # are already scaled. We need noise_std in the same units.
        #
        # Approach: estimate the effective alpha from the output statistics,
        # then scale noise accordingly.
        #
        # Simpler: use the ratio noise_std / typical_popcount_magnitude
        # typical |popcount_pos - popcount_neg| ≈ sqrt(input_dim * 0.58) * mean(|act|)
        # For INT8 acts quantized to [0, 255], mean ≈ ~64
        # But the model uses float activations internally, so we use the
        # actual output RMS to calibrate.

        with torch.no_grad():
            output_rms = output.float().pow(2).mean().sqrt().item()
            if output_rms < 1e-8:
                return output

            # Expected signal magnitude from popcount accumulation:
            # Each output = alpha * sum_b(2^b * (pc_pos_b - pc_neg_b))
            # RMS of this ≈ alpha * sqrt(input_dim * 0.58) * mean_act_magnitude * sqrt(21845)
            # ≈ alpha * sqrt(input_dim * 0.58 * 21845) * mean_act
            #
            # noise_std = sqrt(2 * input_dim * ber * (1-ber) * 21845)
            #
            # SNR = signal / noise = sqrt(0.58 * mean_act^2 / (2 * ber))
            #
            # For ber=1e-4: SNR ≈ sqrt(0.58 * 64^2 / 2e-4) ≈ sqrt(11862400) ≈ 3443
            # For ber=1e-2: SNR ≈ sqrt(0.58 * 64^2 / 2e-2) ≈ sqrt(118624) ≈ 344
            #
            # The relative noise level = noise_std / expected_signal_rms
            # = sqrt(2 * ber / 0.58) (the input_dim and 21845 cancel!)

            relative_noise = math.sqrt(2 * self.ber / 0.58)

            # Apply noise proportional to output magnitude
            noise = torch.randn_like(output) * output_rms * relative_noise

            # Track stats
            snr = output_rms / (output_rms * relative_noise + 1e-10)
            self.noise_stats[name].append({
                'output_rms': output_rms,
                'noise_rms': (output_rms * relative_noise),
                'snr': snr,
                'relative_noise': relative_noise
            })

        return output + noise

    def attach(self, model):
        """Attach hooks to all linear layers in the model."""
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                hook = module.register_forward_hook(
                    lambda mod, inp, out, n=name: self._hook_fn(mod, inp, out, n)
                )
                self.hooks.append(hook)
        print(f"  Attached {len(self.hooks)} hooks to linear layers")
        return self

    def detach(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.noise_stats.clear()

    def set_ber(self, ber):
        self.ber = ber
        self.noise_stats.clear()


# ---------------------------------------------------------------------------
# Perplexity evaluation
# ---------------------------------------------------------------------------
def evaluate_perplexity(model, tokenizer, dataset_text, seq_len=256,
                        max_tokens=2048, device='cpu'):
    """
    Evaluate perplexity on a text dataset using sliding window.

    Returns: (perplexity, num_tokens_evaluated)
    """
    model.eval()

    # Tokenize the full text
    encodings = tokenizer(dataset_text, return_tensors='pt', truncation=False)
    input_ids = encodings['input_ids'][0]

    # Limit to max_tokens
    if len(input_ids) > max_tokens:
        input_ids = input_ids[:max_tokens]

    total_log_likelihood = 0.0
    total_tokens = 0
    num_windows = max(1, (len(input_ids) - 1) // seq_len)

    print(f"  Evaluating {len(input_ids)} tokens in {num_windows} windows of {seq_len}...")

    for i in range(0, len(input_ids) - 1, seq_len):
        end = min(i + seq_len, len(input_ids))
        chunk = input_ids[i:end].unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(chunk)
            logits = outputs.logits  # (1, seq_len, vocab_size)

        # Shift: predict token[t+1] from logits[t]
        shift_logits = logits[:, :-1, :].float()  # (1, seq_len-1, vocab)
        shift_labels = chunk[:, 1:]  # (1, seq_len-1)

        # Cross-entropy per token
        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(
            dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        total_log_likelihood += token_log_probs.sum().item()
        total_tokens += shift_labels.numel()

        if (i // seq_len + 1) % 2 == 0 or end >= len(input_ids) - 1:
            current_ppl = math.exp(-total_log_likelihood / total_tokens)
            print(f"    Window {i // seq_len + 1}/{num_windows}: "
                  f"running ppl = {current_ppl:.2f} "
                  f"({total_tokens} tokens)")

    perplexity = math.exp(-total_log_likelihood / total_tokens)
    return perplexity, total_tokens


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("PIM-LLM: BitNet Perplexity Under BER Injection")
    print("=" * 70)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = 'cpu'
    print(f"\nDevice: {device}")
    print(f"Model: {MODEL_NAME}")
    print(f"BER levels: {BER_LEVELS}")
    print(f"Max eval tokens: {MAX_EVAL_TOKENS}")
    print(f"Runs per BER: {NUM_RUNS}")

    # -----------------------------------------------------------------------
    # Step 1: Load model
    # -----------------------------------------------------------------------
    print("\n--- Step 1: Loading BitNet model ---")
    t0 = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    # Load with float32 for CPU
    print(f"  Loading from cache: {CACHE_DIR}")

    config = AutoConfig.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    print(f"  Config: {config.hidden_size}d, {config.num_hidden_layers} layers, "
          f"{config.intermediate_size} FFN, {config.vocab_size} vocab")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)

    # Load model — this will be ~2GB on CPU
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_DIR,
        torch_dtype=torch.float32,
        device_map=device,
        low_cpu_mem_usage=True,
    )
    model.eval()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded: {total_params / 1e9:.2f}B parameters in {time.time() - t0:.1f}s")

    # Verify ternary weights
    ternary_count = 0
    total_weight_elements = 0
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() == 2:
            unique_vals = param.unique()
            if len(unique_vals) <= 5:  # ternary: {-1, 0, +1} possibly with small noise
                ternary_count += 1
            total_weight_elements += param.numel()
    print(f"  Ternary weight matrices: {ternary_count}")

    # Count linear layers (for hook verification)
    num_linear = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
    print(f"  Linear layers: {num_linear}")

    # -----------------------------------------------------------------------
    # Step 2: Load WikiText-2 test set
    # -----------------------------------------------------------------------
    print("\n--- Step 2: Loading evaluation dataset ---")

    try:
        from datasets import load_dataset
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        # Concatenate all non-empty lines
        text = '\n'.join([t for t in dataset['text'] if t.strip()])
        print(f"  WikiText-2 test: {len(text)} characters")
    except Exception as e:
        print(f"  WikiText-2 download failed: {e}")
        print("  Falling back to a built-in text sample...")
        # Use a reasonable English text as fallback
        text = """The history of artificial intelligence began in antiquity, with myths,
stories and rumors of artificial beings endowed with intelligence or consciousness
by master craftsmen. Classical philosophers who attempted to describe the process
of human thinking as the mechanical manipulation of symbols. This work culminated
in the invention of the programmable digital computer in the 1940s, a machine based
on the abstract essence of mathematical reasoning. This device and the ideas behind
it inspired a handful of scientists to begin seriously discussing the possibility of
building an electronic brain. The field of AI research was founded at a workshop held
on the campus of Dartmouth College during the summer of 1956.""" * 20
        print(f"  Fallback text: {len(text)} characters")

    # -----------------------------------------------------------------------
    # Step 3: Baseline perplexity (no errors)
    # -----------------------------------------------------------------------
    print("\n--- Step 3: Baseline perplexity (BER = 0) ---")
    t0 = time.time()

    baseline_ppl, num_tokens = evaluate_perplexity(
        model, tokenizer, text,
        seq_len=SEQ_LEN, max_tokens=MAX_EVAL_TOKENS,
        device=device
    )
    baseline_time = time.time() - t0

    print(f"\n  *** BASELINE PERPLEXITY: {baseline_ppl:.2f} ***")
    print(f"  Tokens evaluated: {num_tokens}")
    print(f"  Time: {baseline_time:.1f}s ({num_tokens / baseline_time:.1f} tok/s)")

    # -----------------------------------------------------------------------
    # Step 4: Evaluate at each BER level
    # -----------------------------------------------------------------------
    print("\n--- Step 4: BER injection sweep ---")

    results = {
        'model': MODEL_NAME,
        'baseline_ppl': baseline_ppl,
        'num_tokens': num_tokens,
        'seq_len': SEQ_LEN,
        'num_runs': NUM_RUNS,
        'seed': SEED,
        'ber_results': {}
    }

    injector = BERInjector()
    injector.attach(model)

    for ber in BER_LEVELS:
        print(f"\n  === BER = {ber:.1e} ===")

        ppls = []
        for run in range(NUM_RUNS):
            # Set new seed for each run (but deterministic per run/ber combo)
            torch.manual_seed(SEED + run * 1000 + int(ber * 1e10))

            injector.set_ber(ber)

            t0 = time.time()
            ppl, _ = evaluate_perplexity(
                model, tokenizer, text,
                seq_len=SEQ_LEN, max_tokens=MAX_EVAL_TOKENS,
                device=device
            )
            elapsed = time.time() - t0
            ppls.append(ppl)

            ppl_change = ((ppl - baseline_ppl) / baseline_ppl) * 100
            print(f"    Run {run+1}/{NUM_RUNS}: ppl = {ppl:.2f} "
                  f"({ppl_change:+.2f}%) [{elapsed:.1f}s]")

        mean_ppl = np.mean(ppls)
        std_ppl = np.std(ppls) if len(ppls) > 1 else 0
        ppl_change_pct = ((mean_ppl - baseline_ppl) / baseline_ppl) * 100

        # Collect noise stats from last run
        avg_snr = 0
        if injector.noise_stats:
            all_snrs = [s['snr'] for stats in injector.noise_stats.values()
                       for s in stats]
            avg_snr = np.mean(all_snrs) if all_snrs else 0

        results['ber_results'][str(ber)] = {
            'perplexities': ppls,
            'mean_ppl': float(mean_ppl),
            'std_ppl': float(std_ppl),
            'ppl_change_pct': float(ppl_change_pct),
            'avg_snr': float(avg_snr),
        }

        print(f"    Mean: {mean_ppl:.2f} ± {std_ppl:.2f} "
              f"({ppl_change_pct:+.2f}% vs baseline)")
        if avg_snr > 0:
            print(f"    Avg SNR: {avg_snr:.1f}")

    injector.detach()

    # -----------------------------------------------------------------------
    # Step 5: Summary table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"\nModel: {MODEL_NAME}")
    print(f"Baseline perplexity: {baseline_ppl:.2f}")
    print(f"Tokens evaluated: {num_tokens}")
    print(f"Runs per BER level: {NUM_RUNS}")
    print()

    header = f"{'BER':>12} | {'Perplexity':>12} | {'Δ ppl':>10} | {'Δ %':>8} | {'Assessment':>15}"
    print(header)
    print("-" * len(header))

    for ber in BER_LEVELS:
        r = results['ber_results'][str(ber)]
        delta = r['mean_ppl'] - baseline_ppl
        pct = r['ppl_change_pct']

        if abs(pct) < 0.5:
            assessment = "Imperceptible"
        elif abs(pct) < 2:
            assessment = "Negligible"
        elif abs(pct) < 5:
            assessment = "Minor"
        elif abs(pct) < 15:
            assessment = "Moderate"
        elif abs(pct) < 50:
            assessment = "Significant"
        else:
            assessment = "SEVERE"

        print(f"{ber:>12.1e} | {r['mean_ppl']:>10.2f}±{r['std_ppl']:<4.2f}| "
              f"{delta:>+10.2f} | {pct:>+7.2f}% | {assessment:>15}")

    # -----------------------------------------------------------------------
    # Step 6: Save results
    # -----------------------------------------------------------------------
    output_file = os.path.join(OUTPUT_DIR, "perplexity_ber_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    # -----------------------------------------------------------------------
    # Step 7: Generate figure
    # -----------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        bers = [b for b in BER_LEVELS if b > 0]
        ppls_mean = [results['ber_results'][str(b)]['mean_ppl'] for b in bers]
        ppls_std = [results['ber_results'][str(b)]['std_ppl'] for b in bers]
        ppl_pcts = [results['ber_results'][str(b)]['ppl_change_pct'] for b in bers]

        # Left: Absolute perplexity
        ax1.axhline(y=baseline_ppl, color='green', linestyle='--',
                    label=f'Baseline: {baseline_ppl:.2f}', linewidth=2)
        ax1.errorbar(bers, ppls_mean, yerr=ppls_std,
                    fmt='o-', color='red', linewidth=2, markersize=8,
                    capsize=5, label='With BER injection')

        # Mark the SiMRA operating point
        ax1.axvline(x=3.8e-8, color='blue', linestyle=':', alpha=0.7,
                    label='SiMRA BER (<3.8×10⁻⁸)')
        # Mark the error budget
        ax1.axvline(x=1e-4, color='orange', linestyle=':', alpha=0.7,
                    label='Error budget (0.01%)')

        ax1.set_xscale('log')
        ax1.set_xlabel('Bit Error Rate (BER)', fontsize=12)
        ax1.set_ylabel('Perplexity', fontsize=12)
        ax1.set_title('BitNet b1.58-2B-4T Perplexity vs BER', fontsize=13)
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

        # Right: Percentage change
        ax2.bar(range(len(bers)), ppl_pcts,
               color=['green' if abs(p) < 2 else 'orange' if abs(p) < 10
                      else 'red' for p in ppl_pcts])
        ax2.set_xticks(range(len(bers)))
        ax2.set_xticklabels([f'{b:.0e}' for b in bers], rotation=45)
        ax2.set_xlabel('BER', fontsize=12)
        ax2.set_ylabel('Perplexity Change (%)', fontsize=12)
        ax2.set_title('Perplexity Degradation vs BER', fontsize=13)
        ax2.axhline(y=0, color='black', linewidth=0.5)
        ax2.axhline(y=1, color='green', linestyle='--', alpha=0.5, label='1% threshold')
        ax2.axhline(y=5, color='orange', linestyle='--', alpha=0.5, label='5% threshold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        fig_path = os.path.join(OUTPUT_DIR, "figures", "fig5_perplexity_ber.png")
        os.makedirs(os.path.dirname(fig_path), exist_ok=True)
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {fig_path}")
        plt.close()

    except Exception as e:
        print(f"Figure generation failed: {e}")

    # -----------------------------------------------------------------------
    # Key findings for paper
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("KEY FINDINGS FOR PAPER")
    print("=" * 70)

    # Find the BER where ppl increases by >1%
    threshold_ber = None
    for ber in sorted(BER_LEVELS):
        if ber == 0:
            continue
        r = results['ber_results'][str(ber)]
        if r['ppl_change_pct'] > 1.0:
            threshold_ber = ber
            break

    if threshold_ber:
        print(f"\n  Perplexity increase exceeds 1% at BER = {threshold_ber:.1e}")
        margin = threshold_ber / 3.8e-8
        print(f"  Safety margin from SiMRA BER: {margin:.0f}×")
    else:
        print(f"\n  Perplexity increase stays below 1% across ALL tested BER levels")
        print(f"  (up to BER = {max(BER_LEVELS):.1e})")

    operating_r = results['ber_results'].get(str(1e-4))
    if operating_r:
        print(f"\n  At error budget (BER = 0.01%):")
        print(f"    Perplexity: {operating_r['mean_ppl']:.2f} "
              f"({operating_r['ppl_change_pct']:+.2f}%)")

    print(f"\n  Conclusion: {'PASS' if (operating_r and operating_r['ppl_change_pct'] < 5) else 'NEEDS INVESTIGATION'}")
    print(f"  — Charge-sharing errors at PIM-LLM's operating BER have "
          f"{'negligible' if (operating_r and operating_r['ppl_change_pct'] < 2) else 'measurable'} "
          f"impact on actual LLM text quality.")

    return results


if __name__ == '__main__':
    results = main()
