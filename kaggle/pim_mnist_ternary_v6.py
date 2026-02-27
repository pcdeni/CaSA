#!/usr/bin/env python3
# ===========================================================================
# Ternary PIM MNIST Demonstration — v6
#
# v6 fixes from v5:
# - v5 achieved 99.49% PyTorch accuracy but PIM inference was broken (16%)
# - ROOT CAUSE 1: Missing alpha scaling in PIM inference path
#   PyTorch computes: output = (w_ternary * alpha) @ x
#   PIM v5 computed:  output = w_ternary @ x  (alpha missing!)
# - ROOT CAUSE 2: AbsMean activation quantization is destructive for sparse
#   MNIST inputs — gamma=mean(|x|) is very small, causing all non-zero pixels
#   to clip at 127, destroying gradient information.
# - FIX 1: Extract alpha per layer, apply in PIM dequantization
# - FIX 2: Use max-abs quantization (no clipping) instead of mean-abs
# - FIX 3: Add float PIM inference (no act quant) as upper bound diagnostic
# - Loads cached teacher & student from v5 — no retraining needed
#
# Architecture: 784 -> 1024 -> 512 -> 256 -> 10 (ternary weights, BN, ReLU)
# ===========================================================================

import math
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch {torch.__version__} | Device: {DEVICE}")
print()

# ===========================================================================
# Ternary Quantization — STE (same as v5)
# ===========================================================================

class TernaryQuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w):
        alpha = w.abs().mean() + 1e-8
        w_ternary = (w / alpha).round().clamp(-1.0, 1.0)
        ctx.save_for_backward(w, alpha.unsqueeze(0))
        return w_ternary, alpha

    @staticmethod
    def backward(ctx, grad_ternary, grad_alpha):
        w, alpha_t = ctx.saved_tensors
        alpha = alpha_t.squeeze()
        mask = (w.abs() / alpha <= 1.5).float()
        return grad_ternary * mask


class TernaryLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_normal_(self.weight, nonlinearity="relu")
        self.register_buffer("_ternary_weight", torch.zeros(out_features, in_features))

    def forward(self, x):
        w_ternary, alpha = TernaryQuantizeSTE.apply(self.weight)
        self._ternary_weight = w_ternary.detach()
        return F.linear(x, w_ternary * alpha)

    def get_ternary_weights(self):
        alpha = self.weight.abs().mean() + 1e-8
        w_t = (self.weight / alpha).round().clamp(-1, 1)
        return w_t.detach().cpu().numpy().astype(np.int8)

    def get_alpha(self):
        """Return the BitNet alpha scaling factor for this layer."""
        return (self.weight.abs().mean() + 1e-8).item()


# ===========================================================================
# Models (same as v5)
# ===========================================================================

class TeacherMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 1024, bias=False)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512, bias=False)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 256, bias=False)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, 10, bias=False)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        return self.fc4(x)


class TernaryMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = TernaryLinear(784, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = TernaryLinear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = TernaryLinear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc4 = TernaryLinear(256, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        return self.fc4(x)

    def get_bn_params(self):
        params = []
        for bn in [self.bn1, self.bn2, self.bn3]:
            scale = bn.weight.detach() / torch.sqrt(bn.running_var.detach() + bn.eps)
            offset = (bn.bias.detach()
                      - bn.weight.detach() * bn.running_mean.detach()
                      / torch.sqrt(bn.running_var.detach() + bn.eps))
            params.append((scale.cpu().numpy(), offset.cpu().numpy()))
        return params

    def get_alphas(self):
        """Return alpha scaling factors for all 4 layers."""
        return [layer.get_alpha() for layer in [self.fc1, self.fc2, self.fc3, self.fc4]]


# ===========================================================================
# Data
# ===========================================================================

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

data_dir = os.path.join(OUTPUT_DIR, "data")
test_dataset = torchvision.datasets.MNIST(root=data_dir, train=False, download=True, transform=transform_test)
test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=0)
print(f"Test: {len(test_dataset):,}")
print()

# ===========================================================================
# Load cached models from v5
# ===========================================================================

teacher_path = os.path.join(OUTPUT_DIR, "best_teacher_v5.pth")
student_path = os.path.join(OUTPUT_DIR, "best_ternary_student_v5.pth")

assert os.path.exists(teacher_path), f"Teacher checkpoint not found: {teacher_path}"
assert os.path.exists(student_path), f"Student checkpoint not found: {student_path}"

teacher = TeacherMLP().to(DEVICE)
teacher.load_state_dict(torch.load(teacher_path, map_location=DEVICE, weights_only=True))
teacher.eval()

student = TernaryMLP().to(DEVICE)
student.load_state_dict(torch.load(student_path, map_location=DEVICE, weights_only=True))
student.eval()

# Verify loaded model accuracy
correct = total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        _, predicted = student(images).max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
pytorch_acc = 100. * correct / total
print(f"Loaded student model: {pytorch_acc:.2f}% test accuracy")
print()

# ===========================================================================
# Weight Extraction — now includes alpha
# ===========================================================================

model = student
layer_names = ["fc1", "fc2", "fc3", "fc4"]
layers = [model.fc1, model.fc2, model.fc3, model.fc4]
ternary_weights = {}
alphas = model.get_alphas()

print("=" * 60)
print("TERNARY WEIGHT EXTRACTION (with alpha)")
print("=" * 60)

for i, (name, layer) in enumerate(zip(layer_names, layers)):
    W = layer.get_ternary_weights()
    unique_vals = set(np.unique(W).tolist())
    assert unique_vals.issubset({-1, 0, 1}), f"{name}: non-ternary!"
    n_neg = int(np.sum(W == -1))
    n_zero = int(np.sum(W == 0))
    n_pos = int(np.sum(W == 1))
    total_el = W.size
    ternary_weights[name] = W
    print(f"  {name}: shape={W.shape} | alpha={alphas[i]:.6f} | "
          f"-1:{n_neg/total_el*100:.1f}% 0:{n_zero/total_el*100:.1f}% +1:{n_pos/total_el*100:.1f}%")

bn_params = model.get_bn_params()
print(f"\nAlpha values: {[f'{a:.6f}' for a in alphas]}")
print(f"All {len(layer_names)} layers verified ternary.")
print()

# ===========================================================================
# PIM Reference (bit-serial) — same as v5
# ===========================================================================

def encode_ternary(W_int8):
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


def pack_bitvector(x_bits, in_dim):
    n_words = (in_dim + 63) // 64
    pad_len = n_words * 64 - in_dim
    if pad_len > 0:
        x_bits = np.pad(x_bits, (0, pad_len), constant_values=0)
    x_packed = np.zeros(n_words, dtype=np.uint64)
    for w in range(n_words):
        for b in range(64):
            x_packed[w] |= np.uint64(x_bits[w * 64 + b]) << np.uint64(b)
    return x_packed


_POPCOUNT_TABLE = np.zeros(256, dtype=np.uint8)
for _i in range(256):
    _POPCOUNT_TABLE[_i] = bin(_i).count("1")


def popcount_uint64(arr):
    x = arr.astype(np.uint64)
    result = np.zeros(arr.shape, dtype=np.int64)
    for byte_idx in range(8):
        byte_val = ((x >> np.uint64(byte_idx * 8)) & np.uint64(0xFF)).astype(np.uint8)
        result += _POPCOUNT_TABLE[byte_val].astype(np.int64)
    return result


def pim_reference_matmul(W_pos_packed, W_neg_packed, x_uint8, in_dim,
                         inject_ber=0.0, rng=None):
    out_dim, n_words = W_pos_packed.shape
    x_padded = np.zeros(n_words * 64, dtype=np.uint8)
    x_padded[:in_dim] = x_uint8
    y = np.zeros(out_dim, dtype=np.int64)
    for bit_plane in range(8):
        x_bit = ((x_padded >> bit_plane) & 1).astype(np.uint8)
        x_packed = pack_bitvector(x_bit, n_words * 64)
        and_pos = W_pos_packed & x_packed[np.newaxis, :]
        and_neg = W_neg_packed & x_packed[np.newaxis, :]
        if inject_ber > 0.0 and rng is not None:
            ber_fraction = inject_ber / 100.0
            total_bits = out_dim * n_words * 64
            for and_arr in [and_pos, and_neg]:
                n_flips = rng.binomial(total_bits, ber_fraction)
                if n_flips > 0:
                    positions = rng.choice(total_bits, size=min(n_flips, total_bits), replace=False)
                    for pos in positions:
                        row, rem = divmod(pos, n_words * 64)
                        word, bit = divmod(rem, 64)
                        and_arr[row, word] ^= np.uint64(1) << np.uint64(bit)
        pc_pos = popcount_uint64(and_pos).sum(axis=1)
        pc_neg = popcount_uint64(and_neg).sum(axis=1)
        y += (pc_pos - pc_neg) << bit_plane
    return y


# Bit-exact check
print("PIM Reference — Bit-exact Verification:")
encoded_layers = []
ternary_ws_list = []
for name in layer_names:
    W = ternary_weights[name]
    out_dim, in_dim = W.shape
    W_pos_packed, W_neg_packed, orig_dim = encode_ternary(W)
    x_test = np.random.randint(0, 256, size=in_dim, dtype=np.uint8)
    y_pim = pim_reference_matmul(W_pos_packed, W_neg_packed, x_test, in_dim)
    y_ref = W.astype(np.int64) @ x_test.astype(np.int64)
    match = np.array_equal(y_pim, y_ref)
    print(f"  {name}: {'PASS' if match else 'FAIL'}")
    assert match, f"Bit-exact FAILED for {name}!"
    encoded_layers.append((W_pos_packed, W_neg_packed, orig_dim))
    ternary_ws_list.append(W)
print("All layers BIT-EXACT!\n")


# ===========================================================================
# PIM Inference — v6: THREE modes to diagnose quantization impact
# ===========================================================================

def pim_inference_single(x_flat, encoded_layers, ternary_ws, bn_folded_params,
                         layer_alphas, quant_mode="maxabs",
                         inject_ber=0.0, rng=None):
    """PIM inference with alpha scaling and configurable activation quantization.

    quant_mode:
      "absmean" — gamma = mean(|x|), original BitNet approach (clips sparse inputs)
      "maxabs"  — gamma = max(|x|), no clipping, preserves full dynamic range
      "float"   — no INT8 quantization, float ternary matmul (upper bound diagnostic)
    """
    x = x_flat.astype(np.float64)
    for layer_idx, ((W_pos, W_neg, in_dim), W_int8) in enumerate(
            zip(encoded_layers, ternary_ws)):
        alpha = layer_alphas[layer_idx]

        if quant_mode == "float":
            # Float matmul with ternary weights — no INT8 quantization at all
            y = W_int8.astype(np.float64) @ x[:in_dim]
            x_new = y * alpha
        else:
            Qp = 127
            if quant_mode == "maxabs":
                gamma = np.abs(x).max() + 1e-8
            else:  # absmean
                gamma = np.abs(x).mean() + 1e-8
            x_int = np.clip(np.round(x / gamma * Qp), -128, 127).astype(np.int16)
            x_pos = np.clip(x_int, 0, 127).astype(np.uint8)
            x_neg = np.clip(-x_int, 0, 128).astype(np.uint8)
            y_pos = pim_reference_matmul(W_pos, W_neg, x_pos, in_dim, inject_ber, rng)
            y_neg = pim_reference_matmul(W_pos, W_neg, x_neg, in_dim, inject_ber, rng)
            y_int = y_pos - y_neg
            x_new = y_int.astype(np.float64) * gamma / Qp * alpha

        if layer_idx < len(bn_folded_params) and bn_folded_params[layer_idx] is not None:
            scale, offset = bn_folded_params[layer_idx]
            x_new = x_new * scale.astype(np.float64) + offset.astype(np.float64)
        if layer_idx < len(encoded_layers) - 1:
            x_new = np.maximum(x_new, 0.0)
        x = x_new
    return x


bn_folded = [bn_params[0], bn_params[1], bn_params[2], None]

# ===========================================================================
# Diagnostic: compare quantization modes on 200 samples
# ===========================================================================

print("=" * 60)
print("DIAGNOSTIC: Quantization mode comparison (200 samples)")
print("=" * 60)

diag_samples = 200
diag_images, diag_labels = [], []
count = 0
for images, labels in test_loader:
    for i in range(len(labels)):
        diag_images.append(images[i].numpy().flatten())
        diag_labels.append(labels[i].item())
        count += 1
        if count >= diag_samples:
            break
    if count >= diag_samples:
        break

for mode_name in ["float", "maxabs", "absmean"]:
    correct = 0
    for i in range(diag_samples):
        logits = pim_inference_single(diag_images[i], encoded_layers, ternary_ws_list,
                                       bn_folded, alphas, quant_mode=mode_name)
        if np.argmax(logits) == diag_labels[i]:
            correct += 1
    acc = 100. * correct / diag_samples
    print(f"  {mode_name:8s}: {acc:.1f}% ({correct}/{diag_samples})")

# Single sample trace for detailed comparison
sample_img = diag_images[0]
sample_label = diag_labels[0]
with torch.no_grad():
    pt_logits = model(torch.tensor(sample_img, dtype=torch.float32).view(1, 1, 28, 28).to(DEVICE))
    pt_logits = pt_logits.cpu().numpy().flatten()

print(f"\nSingle sample trace (label={sample_label}):")
print(f"  PyTorch  : pred={np.argmax(pt_logits)} | |logits|={np.abs(pt_logits).mean():.3f}")
for mode_name in ["float", "maxabs", "absmean"]:
    logits = pim_inference_single(sample_img, encoded_layers, ternary_ws_list,
                                   bn_folded, alphas, quant_mode=mode_name)
    print(f"  {mode_name:8s} : pred={np.argmax(logits)} | |logits|={np.abs(logits).mean():.3f}")
print()

# ===========================================================================
# PIM Inference — 1000 samples with maxabs quantization
# ===========================================================================

QUANT_MODE = "maxabs"  # Best mode from diagnostics (no clipping)
N_PIM_SAMPLES = 1000
print(f"PIM Inference v6 ({QUANT_MODE}) on {N_PIM_SAMPLES} samples (PyTorch acc: {pytorch_acc:.2f}%)")

pim_correct = pim_total = 0
pim_logits_all = []
t0 = time.time()

for batch_idx, (images, labels) in enumerate(test_loader):
    for i in range(len(labels)):
        if pim_total >= N_PIM_SAMPLES:
            break
        logits = pim_inference_single(images[i].numpy().flatten(),
                                       encoded_layers, ternary_ws_list, bn_folded,
                                       alphas, quant_mode=QUANT_MODE)
        if np.argmax(logits) == labels[i].item():
            pim_correct += 1
        pim_logits_all.append(logits)
        pim_total += 1
    if pim_total >= N_PIM_SAMPLES:
        break
    elapsed = time.time() - t0
    if pim_total > 0:
        print(f"  {pim_total}/{N_PIM_SAMPLES} ({100*pim_total/N_PIM_SAMPLES:.0f}%) "
              f"[{pim_total/elapsed:.0f} samp/s]")

pim_acc = 100. * pim_correct / pim_total
print(f"\nPIM v6 accuracy ({QUANT_MODE}): {pim_acc:.2f}% | PyTorch: {pytorch_acc:.2f}% | "
      f"Diff: {abs(pim_acc-pytorch_acc):.2f}%")
pim_logits_all = np.array(pim_logits_all)
print()

# ===========================================================================
# Error Injection (200 samples x 8 BER levels)
# ===========================================================================

BER_VALUES = [0.0, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
N_ERROR_SAMPLES = 200

print(f"Error Injection: {len(BER_VALUES)} BER levels x {N_ERROR_SAMPLES} samples")

clean_logits = pim_logits_all[:N_ERROR_SAMPLES]
clean_labels, test_images_flat = [], []
count = 0
for images, labels in test_loader:
    for i in range(len(labels)):
        clean_labels.append(labels[i].item())
        test_images_flat.append(images[i].numpy().flatten())
        count += 1
        if count >= N_ERROR_SAMPLES:
            break
    if count >= N_ERROR_SAMPLES:
        break
clean_labels = np.array(clean_labels[:N_ERROR_SAMPLES])
test_images_flat = np.array(test_images_flat[:N_ERROR_SAMPLES])

error_results = []
for ber in BER_VALUES:
    t0 = time.time()
    rng = np.random.default_rng(seed=SEED)
    noisy_correct = 0
    cosine_sims = []
    for i in range(N_ERROR_SAMPLES):
        logits = pim_inference_single(test_images_flat[i], encoded_layers,
                                       ternary_ws_list, bn_folded, alphas,
                                       quant_mode=QUANT_MODE,
                                       inject_ber=ber, rng=rng)
        if np.argmax(logits) == clean_labels[i]:
            noisy_correct += 1
        clean = clean_logits[i]
        cs = np.dot(logits, clean) / ((np.linalg.norm(logits) + 1e-12) * (np.linalg.norm(clean) + 1e-12))
        cosine_sims.append(cs)
    acc = 100. * noisy_correct / N_ERROR_SAMPLES
    error_results.append({"ber": ber, "accuracy": acc, "cosine_sim": np.mean(cosine_sims)})
    print(f"  BER={ber:8.4f}% | Acc={acc:6.2f}% | CosSim={np.mean(cosine_sims):.6f} | {time.time()-t0:.1f}s")

clean_acc = error_results[0]["accuracy"]
print(f"\n{'BER (%)':>10} | {'Accuracy':>10} | {'CosSim':>10}")
for r in error_results:
    print(f"{r['ber']:10.4f} | {r['accuracy']:9.2f}% | {r['cosine_sim']:10.6f}")

# Error injection plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
bers = [r["ber"] for r in error_results]
accs = [r["accuracy"] for r in error_results]
coss = [r["cosine_sim"] for r in error_results]

ax1.plot(bers, accs, "bo-")
ax1.axhline(y=clean_acc, color="g", linestyle="--", alpha=0.5, label=f"Clean: {clean_acc:.1f}%")
ax1.set_xlabel("BER (%)")
ax1.set_ylabel("Accuracy (%)")
ax1.set_title("PIM Accuracy vs Bit Error Rate (v6)")
ax1.grid(True, alpha=0.3)
ax1.set_xscale("symlog", linthresh=0.001)
ax1.legend()

ax2.plot(bers, coss, "rs-")
ax2.set_xlabel("BER (%)")
ax2.set_ylabel("Cosine Similarity")
ax2.set_title("Output Cosine Similarity vs BER (v6)")
ax2.grid(True, alpha=0.3)
ax2.set_xscale("symlog", linthresh=0.001)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "error_injection_v6.png"), dpi=150)
print("Saved error_injection_v6.png\n")

# ===========================================================================
# Summary
# ===========================================================================

print("=" * 60)
print("SUMMARY (v6)")
print("=" * 60)
print(f"Student (PyTorch): {pytorch_acc:.2f}%")
print(f"Quant mode       : {QUANT_MODE}")
print(f"PIM v6 accuracy  : {pim_acc:.2f}% (on {N_PIM_SAMPLES} samples)")
print(f"PIM-PyTorch diff : {abs(pim_acc - pytorch_acc):.2f}%")
print(f"Clean BER=0 acc  : {clean_acc:.2f}%")
print(f"\nv6 FIXES:")
print(f"  1. Added alpha scaling in PIM dequantization")
print(f"  2. Using {QUANT_MODE} quantization (avoids clipping)")
print(f"Alpha values: fc1={alphas[0]:.6f}, fc2={alphas[1]:.6f}, fc3={alphas[2]:.6f}, fc4={alphas[3]:.6f}")
print()

# BER tolerance summary
for r in error_results:
    if r["ber"] > 0 and r["accuracy"] < clean_acc * 0.9:
        print(f"BER tolerance: accuracy drops below 90% of clean at BER={r['ber']:.4f}%")
        break

print()
print("=" * 60)
print("SCRIPT COMPLETE (v6)")
print("=" * 60)
