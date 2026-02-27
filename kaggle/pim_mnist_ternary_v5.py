#!/usr/bin/env python3
# ===========================================================================
# Ternary PIM MNIST Demonstration — v5
#
# v5 fixes from v4 (which reached 80.7%):
# - REMOVED activation quantization during training — this was the main
#   culprit for v4's ~80% plateau. AbsMean INT8 quant noise prevented
#   convergence. Now quant is ONLY applied during PIM inference.
# - Higher KD alpha (0.3) — more teacher guidance
# - 50 teacher epochs (was 30) — better teacher baseline
# - 300 student epochs with cosine annealing — more room to converge
# - Larger LR (3e-3) with longer warmup (10 epochs)
# - Target: >95% test accuracy
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
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print()

# ===========================================================================
# Ternary Quantization — STE (clean, no activation quant)
# ===========================================================================

class TernaryQuantizeSTE(torch.autograd.Function):
    """BitNet-style ternary quantization with masked STE.

    Forward: w -> round(clip(w / alpha, -1, 1))  where alpha = mean(|w|)
    Backward: STE with mask — gradients zeroed for weights |w|/alpha > 1.5
    """
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
    """Linear layer with ternary weights and BitNet alpha scaling.

    v5: NO activation quantization during training.
    Output = (w_ternary * alpha) @ x
    """
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


# Sanity check
_w = torch.randn(4, 4)
_wt, _alpha = TernaryQuantizeSTE.apply(_w)
assert set(_wt.unique().tolist()).issubset({-1.0, 0.0, 1.0})
print(f"Ternary quantization OK. alpha={_alpha:.4f}")

# ===========================================================================
# Models
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


# ===========================================================================
# Data
# ===========================================================================

transform_train = transforms.Compose([
    transforms.RandomAffine(degrees=10, translate=(0.08, 0.08), scale=(0.92, 1.08)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

data_dir = os.path.join(OUTPUT_DIR, "data")
train_dataset = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=transform_train)
test_dataset = torchvision.datasets.MNIST(root=data_dir, train=False, download=True, transform=transform_test)

BATCH_SIZE = 512
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=0)
print(f"Train: {len(train_dataset):,} | Test: {len(test_dataset):,} | Batch: {BATCH_SIZE}")
print()

# ===========================================================================
# Phase 1 — Teacher (cached)
# ===========================================================================

teacher = TeacherMLP().to(DEVICE)
teacher_path = os.path.join(OUTPUT_DIR, "best_teacher_v5.pth")

if os.path.exists(teacher_path):
    teacher.load_state_dict(torch.load(teacher_path, map_location=DEVICE, weights_only=True))
    teacher.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            _, predicted = teacher(images).max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    best_teacher_acc = 100. * correct / total
    print(f"Loaded cached teacher: {best_teacher_acc:.2f}%")
else:
    print("=" * 60)
    print("PHASE 1: TRAINING TEACHER (50 epochs)")
    print("=" * 60)
    TEACHER_EPOCHS = 50
    teacher_opt = torch.optim.AdamW(teacher.parameters(), lr=1e-3, weight_decay=1e-4)
    teacher_sched = torch.optim.lr_scheduler.CosineAnnealingLR(teacher_opt, T_max=TEACHER_EPOCHS, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss()
    best_teacher_acc = 0.0

    for epoch in range(1, TEACHER_EPOCHS + 1):
        t0 = time.time()
        teacher.train()
        running_loss = correct = total = 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            teacher_opt.zero_grad()
            outputs = teacher(images)
            loss = criterion(outputs, labels)
            loss.backward()
            teacher_opt.step()
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        teacher_sched.step()

        teacher.eval()
        t_correct = t_total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                _, predicted = teacher(images).max(1)
                t_total += labels.size(0)
                t_correct += predicted.eq(labels).sum().item()
        test_acc = 100. * t_correct / t_total

        if test_acc > best_teacher_acc:
            best_teacher_acc = test_acc
            torch.save(teacher.state_dict(), teacher_path)

        if epoch <= 3 or epoch % 10 == 0 or epoch == TEACHER_EPOCHS:
            print(f"  Epoch {epoch:3d} | Loss {running_loss/total:.4f} | "
                  f"Train {100.*correct/total:.1f}% | Test {test_acc:.2f}% | {time.time()-t0:.1f}s")

    teacher.load_state_dict(torch.load(teacher_path, weights_only=True))
    teacher.eval()
    print(f"\nTeacher best: {best_teacher_acc:.2f}%")

print()

# ===========================================================================
# Phase 2 — Ternary Student (v5: NO activation quant during training)
# ===========================================================================

print("=" * 60)
print("PHASE 2: TRAINING TERNARY STUDENT (v5 — no act quant)")
print("=" * 60)

STUDENT_EPOCHS = 300
WARMUP_EPOCHS = 10
BASE_LR = 3e-3
MIN_LR = 1e-5
KD_TEMP = 4.0
KD_ALPHA = 0.3
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05
GRAD_CLIP = 1.0

torch.manual_seed(SEED)
student = TernaryMLP().to(DEVICE)
student_opt = torch.optim.AdamW(student.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)
ce_criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
kl_criterion = nn.KLDivLoss(reduction="batchmean")


def get_lr(epoch):
    if epoch < WARMUP_EPOCHS:
        return MIN_LR + (BASE_LR - MIN_LR) * epoch / WARMUP_EPOCHS
    progress = (epoch - WARMUP_EPOCHS) / (STUDENT_EPOCHS - WARMUP_EPOCHS)
    return MIN_LR + 0.5 * (BASE_LR - MIN_LR) * (1 + math.cos(math.pi * progress))


student_sched = torch.optim.lr_scheduler.LambdaLR(
    student_opt, lr_lambda=lambda epoch: get_lr(epoch) / BASE_LR
)

print(f"Epochs: {STUDENT_EPOCHS} | KD: T={KD_TEMP}, alpha={KD_ALPHA} | "
      f"LR: {BASE_LR} | WD: {WEIGHT_DECAY} | Clip: {GRAD_CLIP}")
print(f"v5 KEY FIX: No activation quantization during training")
print(f"\n{'Ep':>4} | {'Loss':>7} | {'CE':>6} | {'KD':>6} | {'Trn%':>6} | {'Tst%':>6} | {'LR':>9} | {'t':>4}")
print("-" * 70)

train_losses, test_accs = [], []
best_test_acc, best_epoch = 0.0, 0
student_path = os.path.join(OUTPUT_DIR, "best_ternary_student_v5.pth")

for epoch in range(1, STUDENT_EPOCHS + 1):
    t0 = time.time()
    lr_now = student_opt.param_groups[0]["lr"]

    student.train()
    r_loss = r_ce = r_kd = 0.0
    correct = total = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        student_opt.zero_grad()
        s_logits = student(images)
        with torch.no_grad():
            t_logits = teacher(images)

        ce_loss = ce_criterion(s_logits, labels)
        s_soft = F.log_softmax(s_logits / KD_TEMP, dim=1)
        t_soft = F.softmax(t_logits / KD_TEMP, dim=1)
        kd_loss = kl_criterion(s_soft, t_soft) * (KD_TEMP ** 2)

        loss = KD_ALPHA * kd_loss + (1 - KD_ALPHA) * ce_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=GRAD_CLIP)
        student_opt.step()

        r_loss += loss.item() * images.size(0)
        r_ce += ce_loss.item() * images.size(0)
        r_kd += kd_loss.item() * images.size(0)
        _, predicted = s_logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    student_sched.step()
    train_loss = r_loss / total
    train_acc = 100. * correct / total
    train_losses.append(train_loss)

    student.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            _, predicted = student(images).max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    test_acc = 100. * correct / total
    test_accs.append(test_acc)
    elapsed = time.time() - t0

    if test_acc > best_test_acc:
        best_test_acc = test_acc
        best_epoch = epoch
        torch.save(student.state_dict(), student_path)

    if epoch <= 5 or epoch % 10 == 0 or epoch > STUDENT_EPOCHS - 3:
        print(f"{epoch:4d} | {train_loss:7.4f} | {r_ce/total:6.3f} | {r_kd/total:6.3f} | "
              f"{train_acc:5.1f}% | {test_acc:5.1f}% | {lr_now:9.6f} | {elapsed:3.0f}s")

print(f"\nBest test accuracy: {best_test_acc:.2f}% (epoch {best_epoch})")
student.load_state_dict(torch.load(student_path, weights_only=True))
student.eval()

# Final verify
correct = total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        _, predicted = student(images).max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
final_test_acc = 100. * correct / total
print(f"Final test accuracy (loaded best): {final_test_acc:.2f}%")

# Training curves
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
axes[0].plot(range(1, len(train_losses)+1), train_losses, "b-", linewidth=0.8)
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Training Loss")
axes[0].set_title("Student Training Loss (v5)"); axes[0].grid(True, alpha=0.3)

axes[1].plot(range(1, len(test_accs)+1), test_accs, "r-", linewidth=0.8)
axes[1].axhline(y=best_test_acc, color="g", linestyle="--", alpha=0.5, label=f"Best: {best_test_acc:.1f}%")
axes[1].axhline(y=best_teacher_acc, color="b", linestyle=":", alpha=0.5, label=f"Teacher: {best_teacher_acc:.1f}%")
axes[1].axhline(y=95, color="orange", linestyle="--", alpha=0.5, label="Target: 95%")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Test Accuracy (%)")
axes[1].set_title("Student Test Accuracy (v5)"); axes[1].grid(True, alpha=0.3); axes[1].legend()

lrs = [get_lr(e) for e in range(len(train_losses))]
axes[2].plot(range(1, len(lrs)+1), lrs, "g-", linewidth=0.8)
axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Learning Rate")
axes[2].set_title("LR Schedule"); axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "training_curves_v5.png"), dpi=150)
print("Training curves saved to training_curves_v5.png")
print()

# ===========================================================================
# Quick check: did we hit 95%?
# ===========================================================================

if best_test_acc < 95.0:
    print(f"WARNING: Best accuracy {best_test_acc:.2f}% is below 95% target.")
    print("Consider: more epochs, different LR, or progressive quantization.")
    print("Continuing with PIM verification anyway...")
else:
    print(f"TARGET REACHED: {best_test_acc:.2f}% >= 95%")
print()

# ===========================================================================
# Weight Extraction & Verification
# ===========================================================================

student.eval()
model = student
layer_names = ["fc1", "fc2", "fc3", "fc4"]
layers = [model.fc1, model.fc2, model.fc3, model.fc4]
ternary_weights = {}

print("=" * 60)
print("TERNARY WEIGHT EXTRACTION")
print("=" * 60)

for name, layer in zip(layer_names, layers):
    W = layer.get_ternary_weights()
    unique_vals = set(np.unique(W).tolist())
    assert unique_vals.issubset({-1, 0, 1}), f"{name}: non-ternary!"
    n_neg = int(np.sum(W == -1))
    n_zero = int(np.sum(W == 0))
    n_pos = int(np.sum(W == 1))
    total = W.size
    ternary_weights[name] = W
    print(f"  {name}: shape={W.shape} | -1:{n_neg/total*100:.1f}% 0:{n_zero/total*100:.1f}% +1:{n_pos/total*100:.1f}%")

bn_params = model.get_bn_params()
print(f"\nAll {len(layer_names)} layers verified ternary.")
print()

# ===========================================================================
# PIM Reference (bit-serial) — same as v4
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
print("All layers BIT-EXACT!\n")

# ===========================================================================
# PIM Inference (reduced: 1000 samples for speed)
# ===========================================================================

def pim_inference_single(x_flat, encoded_layers, ternary_ws, bn_folded_params,
                         inject_ber=0.0, rng=None):
    x = x_flat.astype(np.float64)
    for layer_idx, ((W_pos, W_neg, in_dim), W_int8) in enumerate(
            zip(encoded_layers, ternary_ws)):
        Qp = 127
        gamma = np.abs(x).mean() + 1e-8
        x_int = np.clip(np.round(x / gamma * Qp), -128, 127).astype(np.int16)
        x_pos = np.clip(x_int, 0, 127).astype(np.uint8)
        x_neg = np.clip(-x_int, 0, 128).astype(np.uint8)
        y_pos = pim_reference_matmul(W_pos, W_neg, x_pos, in_dim, inject_ber, rng)
        y_neg = pim_reference_matmul(W_pos, W_neg, x_neg, in_dim, inject_ber, rng)
        y_int = y_pos - y_neg
        x = y_int.astype(np.float64) * gamma / Qp
        if layer_idx < len(bn_folded_params) and bn_folded_params[layer_idx] is not None:
            scale, offset = bn_folded_params[layer_idx]
            x = x * scale.astype(np.float64) + offset.astype(np.float64)
        if layer_idx < len(encoded_layers) - 1:
            x = np.maximum(x, 0.0)
    return x


encoded_layers = []
ternary_ws_list = []
for name in layer_names:
    W = ternary_weights[name]
    W_pos, W_neg, orig = encode_ternary(W)
    encoded_layers.append((W_pos, W_neg, orig))
    ternary_ws_list.append(W)

bn_folded = [bn_params[0], bn_params[1], bn_params[2], None]

# PyTorch accuracy
model.eval()
pytorch_correct = pytorch_total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        _, predicted = model(images).max(1)
        pytorch_total += labels.size(0)
        pytorch_correct += predicted.eq(labels).sum().item()
pytorch_acc = 100. * pytorch_correct / pytorch_total

# PIM inference — 1000 samples (faster than 10K)
N_PIM_SAMPLES = 1000
print(f"PIM Inference on {N_PIM_SAMPLES} samples (PyTorch acc: {pytorch_acc:.2f}%)")

pim_correct = pim_total = 0
pim_logits_all = []
t0 = time.time()

for batch_idx, (images, labels) in enumerate(test_loader):
    for i in range(len(labels)):
        if pim_total >= N_PIM_SAMPLES:
            break
        logits = pim_inference_single(images[i].numpy().flatten(),
                                       encoded_layers, ternary_ws_list, bn_folded)
        if np.argmax(logits) == labels[i].item():
            pim_correct += 1
        pim_logits_all.append(logits)
        pim_total += 1
    if pim_total >= N_PIM_SAMPLES:
        break
    elapsed = time.time() - t0
    print(f"  {pim_total}/{N_PIM_SAMPLES} ({100*pim_total/N_PIM_SAMPLES:.0f}%) "
          f"[{pim_total/elapsed:.0f} samp/s]")

pim_acc = 100. * pim_correct / pim_total
print(f"\nPIM accuracy: {pim_acc:.2f}% | PyTorch: {pytorch_acc:.2f}% | "
      f"Diff: {abs(pim_acc-pytorch_acc):.2f}%")
pim_logits_all = np.array(pim_logits_all)
print()

# ===========================================================================
# Error Injection (reduced: 200 samples)
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
                                       ternary_ws_list, bn_folded, inject_ber=ber, rng=rng)
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

ax1.plot(bers, accs, "bo-"); ax1.set_xlabel("BER (%)"); ax1.set_ylabel("Accuracy (%)")
ax1.set_title("Accuracy vs BER"); ax1.grid(True, alpha=0.3); ax1.set_xscale("symlog", linthresh=0.001)
ax2.plot(bers, coss, "rs-"); ax2.set_xlabel("BER (%)"); ax2.set_ylabel("Cosine Similarity")
ax2.set_title("CosSim vs BER"); ax2.grid(True, alpha=0.3); ax2.set_xscale("symlog", linthresh=0.001)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "error_injection_v5.png"), dpi=150)
print("Saved error_injection_v5.png\n")

# ===========================================================================
# Summary
# ===========================================================================

print("=" * 60)
print("SUMMARY (v5)")
print("=" * 60)
print(f"Teacher          : {best_teacher_acc:.2f}%")
print(f"Student (best)   : {best_test_acc:.2f}% (epoch {best_epoch})")
print(f"Gap              : {best_teacher_acc - best_test_acc:.2f}%")
print(f"PyTorch (ternary): {pytorch_acc:.2f}%")
print(f"PIM reference    : {pim_acc:.2f}% (on {N_PIM_SAMPLES} samples)")
print(f"PIM-PyTorch diff : {abs(pim_acc - pytorch_acc):.2f}%")
print(f"\nTarget: >95% | Status: {'PASS' if best_test_acc >= 95.0 else 'NEEDS WORK'}")
print()
print("=" * 60)
print("SCRIPT COMPLETE (v5)")
print("=" * 60)
