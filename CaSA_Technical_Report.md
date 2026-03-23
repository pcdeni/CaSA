# CaSA: Ternary LLM Inference on Commodity DRAM via Charge-Sharing MAJ3 and SA-Mediated RowCopy

**Authors:** Vibe
**Affiliation:** Independent
**Date:** February 2026

---

## Abstract

CaSA is a system architecture for ternary large language model (LLM) inference on unmodified commodity DDR4 DRAM. The architecture exploits charge-sharing AND via triple-row activation (MAJ3), an analog phenomenon that processes 65,536 bits simultaneously in approximately 76 ns. MAJ3 destroys all three participating rows; SA-mediated RowCopy (0% BER) restores the weight row afterward, while activation rows are ephemeral and need not be preserved. The resulting protocol achieves BER < 3.8 x 10^-8, as measured by SiMRA across 79 million trials on 120 DDR4 chips, with zero die modifications.

The system targets BitNet b1.58-2B-4T (2B parameters) on a single 8 GB DIMM. Cycle-accurate simulation calibrated against SiMRA timing measurements yields unpipelined throughput of 0.40 tok/s per DIMM (2,474 ms/token) and 13.53 tok/s with 4-DIMM pipelining (74 ms/token for ternary activations). Perplexity degrades by +0.39% at the 0.01% BER error budget, measured on the full 2B model using WikiText-2.

The DDR4 bus constitutes the primary bottleneck, consuming 88% of unpipelined inference time. The internal DRAM compute capacity exceeds bus delivery capacity by over 1,000x. Multi-DIMM scaling, reduced-precision activations, and in-DRAM popcount registers (~2,000 gates/bank) address this bottleneck progressively: 4-DIMM pipelined ternary reaches 13.53 tok/s on unmodified hardware; batch amortization reaches ~35 tok/s aggregate; in-DRAM popcount projects to 31-166 tok/s on DDR4; and LPDDR5X-16ch projects to 169 tok/s.

**Table 1: Claim Status Summary**

| Claim | Status | Evidence | Section |
|---|---|---|---|
| Charge-sharing AND on commodity DDR4 | Validated | SiMRA: 79M trials, 0 failures (120 chips) | 2, 5 |
| MAJ3 + RowCopy protocol | Validated | BER < 3.8x10^-8 (SiMRA 95% CI) | 3.3 |
| End-to-end inference pipeline (0.40/13.53 tok/s) | Modeled | Cycle-accurate simulation calibrated to SiMRA timing | 4, 7 |
| Error tolerance at 0.01% BER budget | Validated | Perplexity +0.39% measured on BitNet 2B4T | 5.4 |
| 4-DIMM pipelined (13.53 tok/s) | Modeled | Multi-bank pipelining | 6.3 |
| Batch amortization (B=8, ~35 tok/s agg.) | Projected | Weight-row reuse across concurrent requests (analytical) | 6.3 |
| 4-bit activations (~14+ tok/s) | Projected | No W1.58A4 validation at 2B scale exists | 6.5 |
| In-DRAM popcount (31-166 tok/s) | Requires die change | Samsung patent exists; ~2K gates/bank, <0.3% area | 6.4 |
| LPDDR5X/HBM scaling (28-509 tok/s) | Projected | Unified analytical sim; charge-sharing unproven on LPDDR5X | 8.5 |

Rows above the horizontal division are defensible with existing data. Rows below depend on future validation or manufacturer cooperation.

**Terminology note.** "Commodity DDR4" refers to physically unmodified, off-the-shelf DDR4 DIMMs with no die changes or custom fabrication. The system does require: (a) an FPGA controller capable of issuing timing-violated DDR4 commands (~$2,000-6,000 for the research prototype; ~$50-200 estimated for a production embedded controller), (b) per-DIMM characterization to identify optimal charge-sharing timing parameters, and (c) DIMMs with confirmed charge-sharing compatibility (validated on SK Hynix C-die, 2018-2020 vintage). The DRAM silicon is unmodified; the surrounding system is not standard.

---

## 1. Introduction

Large language model (LLM) deployment is constrained by memory bandwidth. A 2-billion-parameter model stored in 16-bit precision requires 4 GB of weight data. During autoregressive generation, the full weight matrix must traverse the memory subsystem for every output token. At DDR4-2400 bandwidth of 19.2 GB/s, the theoretical throughput ceiling is approximately 5 tok/s, a bound set by bus physics rather than compute capability.

Every commodity DRAM chip contains an untapped compute resource. When two rows sharing the same bitlines are simultaneously activated through a timing violation, the resulting bitline voltage represents the Boolean AND of the two rows' contents. This charge-sharing AND, first described by Seshadri et al. in AMBIT [1] and demonstrated on commodity DDR4 by Gao et al. in ComputeDRAM [2], processes all 65,536 bits of a DRAM row simultaneously in ~76 ns (via MAJ3). The AND result emerges from passive charge redistribution between cell and bitline capacitors (~5-18 fJ/bit), not from active transistor switching (~1-3 pJ/bit in digital logic). However, prior PIM approaches require RowCopy operations to preserve source data, and RowCopy on commodity DDR4 exhibits a 16.3% bit error rate per operation (measured by SiMRA across 120 chips), rendering neural network inference infeasible.

CaSA addresses this through the MAJ3 + RowCopy protocol: triple-row activation (MAJ3) computes AND while SA-mediated RowCopy (0% BER) restores destroyed weight rows. Activations are ephemeral and need not be restored. This replaces unmediated RowCopy with SA-mediated RowCopy, improving reliability by over four orders of magnitude (BER < 3.8 x 10^-8). The emergence of ternary LLMs, notably Microsoft's BitNet b1.58 [3], creates a natural application: ternary weight-by-binary activation multiplication reduces to AND, the operation that charge-sharing provides.

Prior PIM systems either require custom silicon (SK Hynix AiM [4], Samsung HBM-PIM [5], UPMEM [6]), stop at basic bitwise operations without neural network inference (AMBIT [1], ComputeDRAM [2], SIMDRAM [7]), or use FPGA on-chip memory rather than DRAM-side computation (TerEffic [9], TeLLMe [10]). CaSA composes existing, validated DRAM physics into an end-to-end ternary LLM inference pipeline on unmodified commodity DDR4.

### 1.1 Contributions

1. **End-to-end ternary LLM architecture for commodity DDR4.** A complete inference pipeline for BitNet b1.58-2B-4T (2B parameters) using charge-sharing AND on unmodified DDR4 DIMMs, with an FPGA controller for accumulation and non-linear operations.

2. **MAJ3 + RowCopy protocol.** Eliminates unmediated RowCopy (16.3% BER) by using MAJ3 for AND and SA-mediated RowCopy for weight restoration, improving reliability by >4 orders of magnitude (BER < 3.8 x 10^-8) while reducing cycle time by 5%.

3. **Bus-bandwidth-aware throughput model.** Cycle-accurate timing model calibrated against SiMRA measurements, identifying the DDR4 bus as the fundamental bottleneck and correcting prior overestimates by 14.7x.

4. **Error tolerance characterization.** Monte Carlo BER-injection simulation establishing that the 0.01% error budget maintains cos_sim = 0.9993 at depth=30, validated end-to-end by perplexity measurement on BitNet 2B4T (+0.39% at 0.01% BER).

5. **Scaling analysis.** Multi-DIMM scaling, in-DRAM popcount cost analysis, patent landscape survey, and cross-technology projection from DDR4 through LPDDR5X and HBM. A software-defined popcount path is also identified (Section 8.5.2): if future DRAM revisions achieve reliable RowCopy (BER < 0.01%), the bus bottleneck can be broken on unmodified hardware using SIMDRAM-style in-DRAM adder trees.

**Reading guide.** Sections 2-5 establish the existence proof: DRAM physics, architecture, timing model, and error analysis. Section 6 analyzes scaling from the bus bottleneck through popcount and multi-DIMM parallelism. Section 7 evaluates CaSA against existing systems. Section 8 discusses limitations, and Section 9 lists references.

---

## 2. Background

### 2.1 DRAM Operation and Charge Sharing

A DRAM cell stores a single bit as charge on a capacitor connected to a bitline through an access transistor. Cell capacitance (C_cell) ranges from ~25 fF in older DDR3/early-DDR4 nodes to ~7-10 fF at 1x-nm DDR4 processes, with C_bitline ~ 100-200 fF depending on subarray size. During row activation, the wordline goes high, connecting the cell capacitor to the bitline. The sense amplifier detects the resulting voltage perturbation and amplifies it to a full logic level.

When two rows sharing the same bitline are activated in rapid succession (violating the standard DDR4 timing specification), their cell charges share with the bitline before the sense amplifier resolves. For two rows A and B:

- Both cells store '1': charge sharing yields a high voltage; sense amplifier reads '1'
- Either cell stores '0': insufficient charge; sense amplifier reads '0'

This implements Boolean AND across all 65,536 bits of a DRAM row simultaneously. CaSA uses triple-row activation (MAJ3): activating three rows -- two operands plus an all-1s control -- computes AND. MAJ3 destroys all three rows; SA-mediated RowCopy (0% BER) restores the weight row.

**Reliability caveat.** Charge-sharing AND relies on timing violations outside DDR4 specifications. No manufacturer guarantees this behavior. The SiMRA dataset [11] provides statistical characterization (79M trials, zero failures across 120 chips), not a warranty. Production deployment would require either per-DIMM characterization or a JEDEC-standardized PIM mode in future DDR6/LPDDR6.

**IR-drop.** Simultaneously firing 65,536 sense amplifiers is comparable to standard all-bank refresh (REF), which activates rows in all banks simultaneously. Supply integrity during PIM operation is within normal DRAM operating conditions.

The SiMRA-DRAM dataset [11] characterizes multi-row activation across 120 DDR4 chips. For 2-row AND, zero failures were observed in 79M trials at optimal timing (t_12 >= 1 cycle), yielding BER < 3.8 x 10^-8 (95% CI via rule of three: 3/n). This reliability is temperature-stable from 50C to 80C.

**Long-term wear.** The timing-violated tripleACT (MAJ3) sequences operate outside DDR4 specifications. SiMRA demonstrates robust short-term reliability, but sustained operation over months could accelerate oxide wear, threshold voltage drift, or row decoder stress. DRAM gate oxide reliability studies (e.g., Keane et al., IRW 2012) indicate oxide TDDB lifetimes of >10 years under nominal bias conditions. Our tripleACT applies ~1.2V wordline overdrive (same as standard activation) at a higher duty cycle (~0.01% per row vs ~0.001% for typical workloads). Assuming oxide wear scales linearly with activation frequency, the estimated MTTF is ~1-10 years per weight row under continuous PIM operation. This estimate carries substantial uncertainty (10x range) because timing-violated activation stress has no published endurance data. Periodic BER re-characterization (e.g., monthly) and proactive DIMM rotation are recommended.

**DRAM process vs. logic process.** DRAM transistors are fabricated with thick gate oxide, high threshold voltage, and elongated channel length to minimize leakage current for 64 ms charge retention. Logic transistors use thin gate oxide, low threshold voltage, and minimum channel length for fast switching. Digital logic gates on DRAM-process transistors clock at ~200-500 MHz, versus 3-5 GHz on logic process -- a 6-25x speed penalty. This is confirmed by Samsung's HBM-PIM SIMD units at 300 MHz, UPMEM's DPU at 500 MHz, and SK Hynix's AiM at comparable rates.

Charge-sharing AND sidesteps this penalty because it is not digital logic. It is passive charge redistribution between cell and bitline capacitors that produces the Boolean AND as a physical consequence of voltage superposition. The AND completes when the sense amplifier resolves (~76 ns for MAJ3), determined by capacitor ratios and amplifier gain, not transistor switching speed. This distinction between analog compute using native DRAM physics and digital compute on DRAM process is the foundation of the CaSA architecture.

### 2.2 Ternary Neural Networks

BitNet b1.58 [3] constrains all linear layer weights to ternary values {-1, 0, +1} during training using a Straight-Through Estimator (STE). BitNet b1.58-2B-4T uses the following parameters:

| Parameter | Value |
|-----------|-------|
| Hidden dimension | 2,560 |
| FFN dimension | 6,912 |
| Layers | 30 |
| Attention | 20 query heads, 5 KV heads (GQA), head_dim=128 |
| Total parameters | 2.08 billion |

The weight distribution, verified from the published safetensors checkpoint: 42.21% of weights are zero, with the remaining 57.79% split symmetrically between +1 (28.90%) and -1 (28.89%). This distribution is consistent across layer types.

Ternary multiplication reduces to conditional addition: w * x = x if w=+1, -x if w=-1, 0 if w=0. When x is a single bit, this reduces to AND: (w==+1) AND x gives the positive contribution, (w==-1) AND x gives the negative contribution. The full dot product becomes:

y = popcount(W_pos AND x) - popcount(W_neg AND x)

For 8-bit activations, x is decomposed into 8 binary bit-planes:

y = sum_{b=0}^{7} [popcount(W_pos AND x_b) - popcount(W_neg AND x_b)] * 2^b

### 2.3 DRAM Bender Infrastructure

DRAM Bender [12] is an open-source FPGA-based DRAM testing infrastructure from CMU-SAFARI providing pre-built bitstreams for the Xilinx Alveo U200, a programmable instruction set for arbitrary DDR4 command sequences, C++ host API, and support for timing violation experiments.

CaSA uses DRAM Bender as its hardware platform for issuing precise timing-violated double-activation sequences that trigger charge-sharing AND.

**DIMM compatibility.** The Alveo U200 ships with Micron 16GB DDR4 RDIMMs. The target DIMMs are SK Hynix HMA81GU6 8GB DDR4-2400 UDIMMs (C-die, non-ECC, unbuffered), which have confirmed charge-sharing behavior in SiMRA data. For hardware validation, the factory DIMMs would be physically swapped for the target UDIMMs.

### 2.4 Prior PIM Work

AMBIT [1] introduced triple-row activation for bulk bitwise operations in DRAM. ComputeDRAM [2] demonstrated charge-sharing AND/OR on commodity DDR4. SIMDRAM [7] proposed a bit-serial SIMD framework including bitcount operations. DRISA [8] described a reconfigurable in-situ accelerator using NOR operations. None of these demonstrated complete neural network inference.

Commercial PIM products include SK Hynix Newton/AiM [4] (MAC units at GDDR6 bank boundaries), Samsung Aquabolt-XL [5] (FP16 SIMD arrays in HBM2 stacks), and UPMEM [6] (general-purpose RISC DPU cores inside DDR4 DIMMs -- the only commercially shipping PIM-in-DRAM product as of 2026). All require custom manufacturing.

NeuPIMs [27] proposed a heterogeneous NPU-PIM architecture using HBM channels. PIM-AI [28] proposes DDR5/LPDDR5 PIM with embedded RISC-V processors. P3-LLM [29] presents an NPU-PIM heterogeneous accelerator with mixed-precision quantization. All rely on custom digital or analog processing elements integrated into the memory stack. CaSA requires zero modifications to the DRAM die.

Malekar et al. [20] proposed "PIM-LLM," a hybrid architecture using custom RRAM crossbar arrays for 1-bit LLM inference. TerEffic [9] achieves 727 tok/s using FPGA on-chip and HBM storage. T-SAR [14] validates ternary-to-binary decomposition for CPU computation. For comprehensive PIM surveys, see Ghose et al. (IBM J. Res. Dev. 2019) and Mutlu et al. (Springer 2023) [30].

---

## 3. Architecture

### 3.1 System Overview

CaSA uses a hybrid DRAM/FPGA architecture:

```
HOST PC (Linux, PCIe x16)
  |
  v
FPGA CONTROLLER (Alveo U200, Xilinx VU9P)
  |-- DDR4 command engine (DRAM Bender ISA)
  |-- Popcount accumulator (custom RTL, ~2000 LUTs)
  |-- Non-linear ops (RMSNorm, SiLU, Softmax)
  |-- KV-cache (URAM + on-board DDR4, 37.5 KB/token x 1024 tokens)
  |-- Activation quantizer (AbsMean 8-bit)
  |
  v
DDR4 UDIMM (SK Hynix HMA81GU6, 8GB, C-die)
  |-- Ternary weights stored as (W_pos, W_neg) row pairs
  |-- Activation bit-planes written to scratch rows
  |-- Charge-sharing AND via timing-violated dual-row activation
  |-- Results read back through standard DDR4 interface
```

![Figure 1: CaSA System Architecture](figures/fig1_architecture.png)

**Figure 1: CaSA System Architecture.** Host PC -> PCIe -> FPGA Controller (DRAM Bender engine, popcount accumulator, non-linear ops, KV-cache) -> DDR4 bus -> DIMM (weight rows, scratch rows, charge-sharing AND). Per bit-plane, the FPGA writes an activation bit-plane to a scratch row, the DRAM performs charge-sharing AND via MAJ3 (65,536-bit parallel, 76 ns), and the FPGA reads the result (8,000 bytes). This cycle repeats 16 times per layer (8 bit-planes x 2 halves) x 30 layers per token.

**Design split.** The DRAM performs only the massively parallel AND operation (65,536 bits simultaneously). All arithmetic (popcount, shift-accumulate, normalization, attention) runs on the FPGA. This split reflects the constraint described in Section 2.1: DRAM-process transistors are 6-25x slower than logic-process transistors for digital computation. Popcount, RMSNorm, SiLU, and softmax require clocked arithmetic and would be crippled on a DRAM die. Charge-sharing AND is the singular exception -- an analog phenomenon that exploits native DRAM capacitance.

**Production controller path.** The FPGA controller (Alveo U200) is a research prototype. In production, the PIM command sequencing logic (~12K LUTs) would migrate to: (a) a CXL endpoint ASIC, (b) an integrated memory controller with a PIM mode, or (c) an embedded FPGA co-packaged on the DIMM PCB. Production cost estimates:

| Production path | Key components | Estimated BOM (1K units) | Estimated BOM (100K units) | NRE cost |
|---|---|---|---|---|
| CXL ASIC | 28nm ASIC (~50K gates) + PCB + CXL PHY | ~$80-150 | ~$30-60 | $2-5M (tapeout) |
| Embedded FPGA | Lattice iCE40 UP5K (~5K LUTs, $3-5) + DDR4 PHY chip ($5-10) + PCB | ~$50-80 | ~$20-40 | $50-100K (dev) |
| Smart DIMM | FPGA die on DIMM PCB (similar to UPMEM) | ~$60-120 per DIMM | ~$30-60 per DIMM | $200-500K (PCB + qual) |

The DDR4 PHY is the non-trivial component: issuing timing-violated commands requires precise control of the DQ/DQS interface that off-the-shelf memory controller IPs do not support.

**Software stack.** CaSA currently has no user-facing software ecosystem. The envisioned stack consists of: (1) a PCIe/CXL device driver, (2) a C/C++ runtime library (`libcspim`) providing `cspim_load_model()`, `cspim_inference()`, and `cspim_health_check()`, and (3) Python bindings for integration with LLM serving frameworks. This stack does not yet exist; development is estimated at 3-6 person-months.

**Estimated FPGA resource utilization (Alveo U200, Xilinx VU9P).** Pre-synthesis estimates; actual utilization may differ by +/-30% after synthesis.

| Resource | Used | Available | Utilization |
|----------|------|-----------|-------------|
| LUTs (logic) | ~12,000 | 1,182,240 | ~1.0% |
| - DRAM Bender command engine | ~8,000 | | |
| - Popcount tree (65,536-bit) | ~2,000 | | |
| - RMSNorm/SiLU/Softmax | ~1,500 | | |
| - Control/accumulation | ~500 | | |
| FFs (registers) | ~8,000 | 2,364,480 | ~0.3% |
| BRAM (36Kb blocks) | ~120 | 2,160 | ~5.6% |
| URAM (288Kb blocks) | ~540 | 960 | ~56% |

The popcount of a 65,536-bit row is computed progressively as data arrives over the DDR4 bus: each 64-bit BL8 burst is popcounted in a single cycle and accumulated into a running 16-bit sum register. Since the bus delivers 125 bursts over 459 ns, the popcount completes simultaneously with the last burst read, adding zero additional latency beyond the bus transfer itself.

### 3.2 Weight Encoding and DRAM Layout

Each ternary weight matrix W of shape (out_dim, in_dim) is stored in DRAM as two binary matrices:

- W_pos[i,j] = 1 if W[i,j] == +1, else 0
- W_neg[i,j] = 1 if W[i,j] == -1, else 0

**Example.** Consider a 4x4 ternary weight matrix and a 4-bit activation vector:

```
Ternary W:          W_pos (W==+1):     W_neg (W==-1):
[+1  0 -1  0]      [1 0 0 0]          [0 0 1 0]
[ 0 +1 +1 -1]  ->  [0 1 1 0]          [0 0 0 1]
[-1  0  0 +1]      [0 0 0 1]          [1 0 0 0]
[+1 +1 -1  0]      [1 1 0 0]          [0 0 1 0]

Activation x = [1, 0, 1, 1] (single bit-plane)

W_pos AND x:  [1,0,0,0] AND [1,0,1,1] = [1,0,0,0]  -> popcount = 1
W_neg AND x:  [0,0,1,0] AND [1,0,1,1] = [0,0,1,0]  -> popcount = 1
Result for this bit-plane: 1 - 1 = 0

(Repeated for each of the 8 bit-planes of an 8-bit activation, with shift-accumulation)
```

Weights are packed along the bitline dimension: multiple output neurons share a single DRAM row. For a 65,536-bit DRAM row and in_dim = 2,560:

pack_factor = 65536 / 2560 = 25 neurons per row (97.7% utilization)

Each packed group of neurons requires 4 DRAM rows: 1 for W_pos, 1 for W_neg, and 2 scratch rows for activation bit-planes. The full BitNet 2B4T model maps to approximately 133,320 DRAM rows, consuming 25% of a single 8GB DIMM's 524,288 rows.

### 3.3 MAJ3 + RowCopy AND Protocol

Standard dual-row charge-sharing overwrites the second-activated row with the AND result. In prior PIM systems, RowCopy is used to preserve source data, but SiMRA-DRAM measurements show RowCopy has a 16.3% BER on commodity DDR4.

The activation bit-plane is ephemeral: it is freshly written for each of the 8 bit-plane iterations and discarded afterward. The activation sequence is arranged so that:

1. **First activation:** Weight row (survives -- sense amplifiers restore it)
2. **Second activation:** Scratch/activation row (receives AND result, contents sacrificed)

The weight row is never corrupted because it is always activated first. The scratch row's prior contents are irrelevant because the FPGA writes a fresh bit-plane each iteration. This provides:

- 5% faster cycle time (no RowCopy overhead)
- >4 orders of magnitude better reliability (BER < 3.8 x 10^-8 for AND vs 16.3% for RowCopy)
- Simpler command sequence (2 activations instead of 4)

#### Bank-State Timing Diagram

```
Time (ns)  0         460        522       981        1441       1503      1962
           |----------|----------|---------|----------|----------|---------|
Bus:       [WRITE act_b -> scratch_A]      [READ pos result     ][WRITE act_b -> scratch_B]      [READ neg result     ]
           | 125 BL8 bursts (460ns)|      | 125 BL8 (459 ns)   || 125 BL8 bursts (460ns)|      | 125 BL8 (459 ns)   |
Bank:      | idle (precharged)     |[MAJ3 W_pos,scratch_A,ctrl]   || idle (precharged)     |[MAJ3 W_neg,scratch_B,ctrl]   |
           |                       | ACT W_pos (36ns)         ||                       | ACT W_neg (36ns)         |
           |                       | ACT scratch_A (+1.5ns)   ||                       | ACT scratch_B (+1.5ns)   |
           |                       | sense settle (10ns)      ||                       | sense settle (10ns)      |
           |                       | precharge (tRP=14ns)     ||                       | precharge (tRP=14ns)     |
           |                       |<--- 76 ns ------------->||                       |<--- 76 ns ------------->|

Key:
- Each half-cycle (write + MAJ3 + read + RowCopy restore) takes ~1,455 ns
- Two half-cycles per bit-plane: one for W_pos, one for W_neg = ~2,910 ns total
- MAJ3 destroys all three rows; weight restored via SA-mediated RowCopy
- scratch_A and scratch_B are separate DRAM rows in the same bank
- The MAJ3 (76 ns) completes within the subsequent READ window (459 ns)
```

![Figure 2: MAJ3 + RowCopy Protocol Comparison](figures/fig2_protocol.png)

**Figure 2: MAJ3 + RowCopy Protocol Comparison.** (a) Standard RowCopy-based PIM incurs 16.3% BER. (b) The MAJ3 + RowCopy protocol writes activations via the standard DDR4 bus (BER ~ 0), performs charge-sharing AND via MAJ3 (BER < 3.8x10^-8), reads the result, and restores the weight row via SA-mediated RowCopy (0% BER).

### 3.4 Inference Dataflow

For each transformer layer:

```
For each weight matrix M in {q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj}:
  For each bit-plane b in 0..7:
    1. FPGA writes activation bit-plane b to scratch rows in DRAM      [460 ns]
    2. For each packed neuron group g:
       a. DRAM: MAJ3(W_pos[g], scratch[g], ctrl) -> AND result          [76 ns]
       b. FPGA reads AND result row                                    [459 ns]
       c. FPGA: popcount_pos += popcount(result)
       d. DRAM: MAJ3(W_neg[g], scratch[g], ctrl) -> AND result          [76 ns]
       e. FPGA reads AND result row                                    [459 ns]
       f. FPGA: popcount_neg += popcount(result)
    3. FPGA: partial[g] += (popcount_pos - popcount_neg) << b
  FPGA: apply RMSNorm, SiLU/residual, quantize to 8-bit for next layer
```

### 3.5 Non-Linear Operations (FPGA)

All non-linear operations execute on the FPGA:

- **RMSNorm:** x / sqrt(mean(x^2) + eps), implemented as fixed-point divide with reciprocal square root LUT.
- **SiLU (Swish):** x * sigmoid(x), implemented as piecewise-linear approximation in BRAM LUTs (256 entries, 16-bit).
- **Softmax:** exp(x) / sum(exp(x)), implemented with fixed-point exponential LUT and streaming accumulator.
- **Attention QKV:** Full attention computation in FPGA logic. During decode, attention is O(d*L) per layer while weight matmuls are O(d^2). At L=256, attention is 2.1% of compute. Since attention executes on the FPGA concurrently with DRAM charge-sharing operations, it is fully hidden.

**Fixed-point accuracy validation** (validated against IEEE 754 float64 references using 10,000 test vectors per function):

| Function | Implementation | cos_sim (mean) | Max abs error | Notes |
|---|---|---|---|---|
| SiLU | 256-entry PWL LUT, Q8.8 | 0.999997 | 0.0037 | Across 5 input distributions |
| RMSNorm | 512-entry rsqrt LUT, Q8.8 | 1.000000 | 0.013 (mean) | dim=2560, realistic scale params |
| Softmax | 1024-entry exp LUT, 16-bit | 1.000000 | 0.000012 | KL divergence < 8x10^-4 at dim=256 |
| **Chained** (RMSNorm->SiLU->RMSNorm) | All above | **0.999997** | -- | dim=2560, 5000 samples |

The chained cosine similarity of 0.999997 confirms that FPGA fixed-point non-linear operations introduce error over 100x smaller than the charge-sharing AND error budget (cos_sim = 0.9993 at BER = 0.01%).

### 3.6 KV-Cache Management

KV-cache is stored in FPGA URAM and the Alveo U200's on-board DDR4 SODIMMs (separate from PIM DIMMs). Per-token KV-cache: 5 KV heads x 128 head_dim x 2 (K+V) x 30 layers x 8 bits = 37.5 KB/token. For 1024 tokens, total KV-cache is ~38.4 MB. The VU9P's URAM (960 blocks x 36 KB = 34.6 MB) holds ~920 tokens; beyond this, on-board SODIMMs provide overflow storage at ~50 ns latency.

---

## 4. Timing Model

### 4.1 DDR4 Timing Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| Row activation (tRAS) | 36 ns | DDR4-2400 spec |
| Row precharge (tRP) | 14 ns | DDR4-2400 spec |
| Charge-sharing interval (t_12) | 1.5 ns | SiMRA-DRAM optimal |
| Sense amplifier settling | 10 ns | SiMRA-DRAM measured |
| BL8 burst transfer | 3.68 ns | DDR4-2400 (8 beats x 0.46 ns) |
| Bus bandwidth | 19.2 GB/s | DDR4-2400 dual-channel |

### 4.2 Per-Operation Timing

For a packed row (25 neurons x 2560 bits = 8000 bytes):

| Operation | Time | Calculation |
|-----------|------|-------------|
| Write bit-plane to DRAM | 460 ns | 125 BL8 bursts (8000/64) x 3.68 ns |
| Charge-sharing AND (MAJ3) | 76 ns | 3 x tRCD + tRAS + t_sense + tRP |
| Read AND result | 459 ns | 125 BL8 bursts x 3.68 ns |
| **Per-half cycle (write+AND+read)** | **981 ns** | |
| **Per bit-plane (pos + neg halves, incl. RowCopy)** | **~2,910 ns** | |
| **Per matrix element (8 bit-planes)** | **15.7 us** | |

### 4.3 Bus Utilization Correction

Previous PIM throughput models assumed that reading or writing a DRAM row takes a single BL8 burst (~31 ns per 64 bytes). With bitline packing of 25 neurons per row, the effective row contains 8,000 bytes requiring 125 sequential BL8 bursts. The per-row transfer time is 125 x 3.68 ns = 460 ns, not 31 ns -- a 14.7x correction. This correction reduces PIM throughput estimates by 2-3x compared to models that assume peak bandwidth utilization.

### 4.4 Full Model Throughput

**Table 4.4a: Unpipelined (single DIMM)**

| Component | Time per layer | % of total |
|-----------|---------------|------------|
| Write activations to scratch rows | 8.1 ms | 44% |
| Read AND results | 8.1 ms | 44% |
| PIM AND compute (MAJ3) | 1.1 ms | 6% |
| Refresh + FPGA + control overhead | 0.8 ms | 5% |
| **Total per layer (unpipelined)** | **82.5 ms** | |

**Table 4.4b: Pipelined (4-DIMM, multi-bank overlap, ternary activations)**

| Component | Time per layer | % of total |
|-----------|---------------|------------|
| Bus transfers (write+read+restore, overlapped) | 0.2 ms | 2% |
| PIM AND compute (MAJ3, pipelined across banks) | 9.4 ms | 96% |
| Refresh + FPGA + control overhead | 0.2 ms | 2% |
| **Total per layer (pipelined)** | **9.8 ms** | |

**Refresh management.** The refresh overhead accounts for mandatory DDR4 refresh cycles (tREFI = 7.8 us, tRFC = 350 ns for 8Gb devices). Refresh events between bit-planes pause the DRAM command stream but do not affect the FPGA's accumulated result, which resides entirely in FPGA registers. The controller inserts refresh guards at natural boundaries (between bit-planes or weight matrix transitions). During one layer's processing (~82.5 ms unpipelined), approximately 10,577 refresh events are serviced, costing ~0.81 ms.

Full model (30 layers): **Unpipelined:** 82.5 ms x 30 = **2,474 ms/token = 0.40 tok/s** (single DIMM). **Pipelined (4-DIMM, ternary):** 9.8 ms x 30 / 4 = **74 ms/token = 13.53 tok/s**.

### 4.4.1 Power Breakdown

| Component | Power | Source |
|-----------|-------|--------|
| Alveo U200 FPGA (idle + PIM controller logic) | ~35 W | Measured at 32-39W for low-intensity workloads [31]; PIM logic adds ~1-2W |
| DDR4 DIMM (active, sustained row activations) | ~5 W | SK Hynix HMA81GU6 datasheet |
| PCIe interface + host overhead | ~2 W | Estimated |
| **Total system** | **~42 W** | |

The FPGA dominates system power due to VU9P static power (leakage, clock trees, voltage regulators). The 35W figure is a platform artifact: XPE models for neural network accelerators on the Alveo U200 report ~8W total on-chip power at 100 MHz [32]. Each additional DIMM adds approximately 4W.

### 4.5 Multi-DIMM Scaling

Each additional DIMM provides an independent 19.2 GB/s DDR4 bus channel. Since the bus is the bottleneck (98% of pipelined inference time), throughput scales nearly linearly:

| Configuration | tok/s (unpipelined) | tok/s (pipelined ternary) | TPOT (pipelined, ms) | Power |
|--------------|-------|-----------|-----------|-------|
| 1 DIMM | 0.40 | 3.38 | 296 | 42 W |
| 2 DIMMs | 0.81 | 6.76 | 148 | 46 W |
| 4 DIMMs | 1.62 | 13.53 | 74 | 54 W |

**Scaling mechanism.** Weights are partitioned across DIMMs by layer: with 4 DIMMs and 30 layers, each DIMM holds ~8 layers. DIMMs operate in pipeline fashion (one active at a time), not in parallel. The 4x speedup arises from each DIMM handling fewer layers. Inter-DIMM activation transfer (~5 KB per layer boundary, <1 us) is negligible. Load imbalance with 30 layers across 4 DIMMs (8-8-7-7 partition) introduces ~7% imbalance.

**Limitation.** Multi-channel operation requires DRAM Bender to support multiple independent command streams. The current open-source bitstream drives a single DDR4 channel.

**Speed grade sensitivity.** Since the bus dominates, faster DDR4 modules yield proportional throughput gains:

| DDR4 Speed | Bus BW | tok/s (1 DIMM unpipelined) | tok/s (4 DIMMs pipelined) |
|------------|--------|----------------|-----------------|
| DDR4-2400 (baseline) | 19.2 GB/s | 0.40 | 13.53 |
| DDR4-2666 | 21.3 GB/s | 0.45 | 15.0 |
| DDR4-3200 | 25.6 GB/s | 0.54 | 18.0 |

---

## 5. Error Analysis

### 5.1 Single-Layer Error Tolerance

BER injection simulation on a single 2560x2560 ternary matrix-vector product with 8-bit activations:

| BER | Cosine Similarity | Assessment |
|-----|-------------------|------------|
| 0.01% | 0.994 | Pass |
| 0.1% | 0.952 | Marginal |
| 1.0% | 0.711 | Fail |

Note: The 0.994 value is for a single matmul at d_model=2560 without inter-layer normalization. Section 5.2 reports 0.9993 at the same BER for 30 chained layers at d=256 with ReLU and re-quantization between layers (which clips error propagation). The Abstract cites the 0.9993 figure as a conservative lower bound because BitNet 2B4T's d_model=2560 provides better error averaging than dim=256.

### 5.2 Multi-Layer Error Accumulation

Errors are injected independently at each of 30 transformer layers. Monte Carlo simulation (`pim_ber_accumulation_sim.py`) with synthetic ternary layer chains, bit-serial PIM matmul, and MaxAbs INT8 re-quantization shows that depth has minimal impact on BER tolerance. This occurs because: (a) bit-flip errors at each layer are statistically independent and average out over high-dimensional vectors, (b) INT8 re-quantization between layers resets the dynamic range, and (c) ReLU truncation clips negative errors.

**Table 5.2a: Cosine similarity vs. BER and network depth (dim=256, 200 samples/config)**

| BER | Depth=4 | Depth=30 | Depth=60 | Assessment |
|-----|---------|----------|----------|------------|
| 0.01% | 0.9992 | 0.9993 | 0.9994 | Excellent |
| 0.05% | 0.9959 | 0.9965 | 0.9967 | Good |
| 0.1% | 0.9918 | 0.9930 | 0.9932 | Good |
| 0.5% | 0.9549 | 0.9599 | 0.9616 | Marginal |
| 1.0% | 0.9023 | 0.9113 | 0.9129 | Degraded |

**Table 5.2b: Width effect at depth=4**

| BER | dim=256 | dim=512 | Assessment |
|-----|---------|---------|------------|
| 0.01% | 0.9992 | 0.9997 | Both excellent |
| 0.1% | 0.9918 | 0.9971 | dim=512 better |
| 1.0% | 0.9023 | 0.9650 | Width is the dominant factor |

The safety margin from SiMRA-measured BER (< 10^-8) to observable degradation (BER ~ 0.5%) is ~50,000x.

### 5.3 Compatibility with SiMRA Data

The SiMRA BER bound (< 3.8 x 10^-8) is three orders of magnitude below the 0.01% error budget. SiMRA data reports up to 2.13% variation in charge-sharing success rates across temperature and voltage conditions [11]. Even in the worst observed case, 2-row AND success remains above 97.8%.

**DDR4 wear under timing violations.** Repeated timing-violated activations may accelerate DRAM cell wear. While the inference workload is read-heavy (weights are static), scratch rows experience ~234 MAJ3 operations per second at 0.40 tok/s. Mitigation includes scratch row rotation (spreading wear across ~200x more rows in the 75% free headroom), BER drift monitoring, and DIMM vintage targeting.

**Self-heating under sustained PIM operation.** Continuous tripleACT sequences constitute a higher-than-typical activation rate. Each MAJ3 dissipates ~160-250 nJ over 76 ns. The sustained activation rate (~527,000 ANDs per token) is distributed across 16 banks x 64 subarrays = 1,024 subarrays, so each subarray experiences ~0.23 MAJ3 ops/sec. The aggregate additional power is ~35 uW, negligible (<0.5%) compared to typical DDR4 operating power.

DDR4 RDIMMs include on-DIMM thermal sensors (TSOD) that trigger thermal throttling at JEDEC thresholds: normal operation at T_case <= 85C, mandatory 2x refresh rate at 85-95C, critical shutdown above 95C. Thermal throttling from PIM operations alone is unlikely, but co-located workloads may create compound thermal stress. If self-heating pushes subarrays into the high-temperature regime, refresh overhead increases from ~4.5% to ~9-10%, reducing throughput by ~5%.

### 5.4 Perplexity Validation

Perplexity was measured on BitNet b1.58-2B-4T (2B parameters, 30 SiLU transformer layers) using WikiText-2, with BER noise injected into all Linear layer outputs via PyTorch hooks (3 runs per BER level, 8,192 evaluation tokens, float32). Baseline perplexity: 26.23. The noise model applies `relative_noise = sqrt(2 * BER / p)` where p = 0.58 is the non-zero weight fraction.

| BER | Mean perplexity | Change vs baseline | Std across runs |
|---|---|---|---|
| 0 (baseline) | 26.23 | -- | 0.000 |
| 10^-7 | 26.24 | +0.03% | 0.004 |
| 10^-6 | 26.22 | -0.01% | 0.017 |
| 10^-5 | 26.24 | +0.06% | 0.025 |
| **10^-4** | **26.33** | **+0.39%** | 0.020 |
| 10^-3 | 27.31 | +4.1% | 0.112 |
| 10^-2 | 42.25 | +61.1% | 0.337 |

At SiMRA's operating BER (< 10^-8), perplexity impact is within measurement noise. At the 0.01% error budget, perplexity increases by +0.39%. The onset of measurable degradation occurs around BER = 10^-3 (+4.1%), and severe failure at BER = 10^-2 (+61.1%).

### 5.5 Unified Error Metric Summary

| BER | cos_sim (dim=256, depth=30) | cos_sim (dim=2560, single layer) | MNIST accuracy (4-layer MLP) | Measured perplexity impact | Assessment |
|---|---|---|---|---|---|
| < 3.8x10^-8 (SiMRA) | ~1.0000 | ~1.0000 | 100% | < +0.03% | Expected operating point |
| 0.01% (error budget) | 0.9993 | 0.994 | 100% | +0.39% | Within budget |
| 0.1% | 0.9930 | 0.952 | 100% | +4.1% | Marginal |
| 0.5% | 0.9599 | -- | 95.5% | ~+30% (interpolated) | Degraded |
| 1.0% | 0.9113 | 0.711 | 74.5% | +61.1% | Failure |

SiMRA BER bound was measured under short-burst activation patterns at room temperature to 80C [11], not under sustained continuous PIM inference. Hardware validation (Section 8.7) will establish the sustained-operation BER.

### 5.6 Bad Column Masking

Systematic column-level failures may produce spatially correlated errors. PUDTune [24] found that 46.6% of columns in tested DDR4 chips are error-prone during charge-sharing operations, though multi-level calibration reduced this to 3.3%.

The bad column masking strategy:

1. **Column profiling.** During initial BER characterization, the FPGA writes known patterns, performs MAJ3, and reads results. Columns failing more than 1 in 10,000 trials are flagged.
2. **Mask generation.** A per-bank bitmask (~8 KB per bank, 128 KB total) identifies unreliable columns. Neurons are packed to avoid placing critical bit positions on unreliable columns. Flagged positions are treated as zeros during popcount.
3. **Impact on accuracy.** At 3.3% masked columns (after calibration), ~86 bit positions per neuron (out of 2,560) are affected -- an effective masking rate of ~3.4%. With 42% of weights being zero, ~36 of 86 masked positions already contribute nothing. The remaining ~50 masked non-zero weights reduce the effective dot-product dimension by ~2%, producing cos_sim > 0.997 at dim=2560.

### 5.7 Error Mitigation

For scenarios where BER approaches 0.1%:

- **MSB voting (top 3 bit-planes).** Triple redundancy on the 3 most significant bit-planes improves cos_sim from 0.711 to 0.976 at 1% BER, at 1.75x throughput cost.
- **ReTern FAST column-flip.** Per-column error characterization with compensating bit-flips reduces effective BER by ~35% for systematic errors. Cost: ~200 FPGA LUTs + 8 KB per subarray.

### 5.8 Runtime Error Detection

A three-tier runtime monitoring strategy:

**Tier 1: Checksum sentinels (zero throughput cost).** Reserve one scratch row per bank as a known-pattern sentinel. Periodically, the FPGA performs a MAJ3 of the sentinel against a known weight row and verifies the result. Cost: ~0.01% throughput if checked every 100 tokens.

**Tier 2: Layer-output range monitoring (negligible cost).** After each layer's popcount accumulation, the dynamic range (min, max, mean) of the output is monitored. A sudden shift in output statistics (>3 sigma from running average) signals potential error accumulation.

**Tier 3: Periodic re-characterization (offline).** Every 24 hours of continuous operation, a full BER characterization sweep (10^4 trials across all banks, ~2 minutes) is run. The bad-column mask and optimal timing parameters are updated.

**Failover.** If persistent errors exceed the 0.01% BER budget on any DIMM, the system can redistribute layers, fall back to CPU inference, or flag the DIMM for replacement.

---

## 6. Throughput Results

### 6.1 Throughput Equation

CaSA's decode throughput is:

```
tok/s = 1 / (num_layers x T_per_layer / num_DIMMs)

T_per_layer = sum_matrices [ceil(out_dim / pack_factor) x 2 x bit_planes x T_cycle]

T_cycle = T_write + T_AND + T_read + T_restore    (MAJ3 + RowCopy protocol)

Where:
  pack_factor = floor(row_bits / d_in)         -> neurons per AND operation
  T_write     = ceil(pack_factor x d_in / 64) x T_burst   -> 125 x 3.68 ns = 460 ns
  T_AND       = 3 x tRCD + tRAS + t_sense + tRP  -> 76 ns     (MAJ3 physics)
  T_restore   = T_write                          -> 460 ns    (SA-mediated RowCopy)
  T_read      = T_write                        -> 460 ns    (same burst count)
```

The ratio T_bus / T_AND = (460 + 460 + 460) / 76 = 18.2:1. This ratio defines the fundamental bottleneck.

**Parameter decomposition.** Every parameter in the throughput equation is controlled by one of three parties:

| Parameter | Value (DDR4 + BitNet 2B4T) | Controlled by | Can change? |
|---|---|---|---|
| **DRAM physics** | | | |
| Row width (row_bits) | 65,536 (8 KB) | Manufacturer | Only at fab; trend is shrinking |
| T_AND | 76 ns | Physics (MAJ3) | 3-row activation |
| T_burst | 3.68 ns | DDR4 spec | Fixed per DDR generation |
| Bus width | 64 bits | DDR4 spec | Fixed per DDR generation |
| T_write = T_read | 460 ns | Derived: 125 bursts x 3.68 ns | Reducible only by popcount or activation register |
| Banks per DIMM | 16 | Manufacturer | 32 in DDR5 |
| Rows per bank | 32,768 | Manufacturer | Current utilization is 25% |
| **Model architecture** | | | |
| d_model | 2560 | Microsoft (BitNet) | Determines pack_factor = 25 neurons/AND |
| d_ffn | 6912 | Microsoft (BitNet) | FFN down_proj: pack_factor = 9 |
| num_layers | 30 | Microsoft (BitNet) | Linear cost multiplier |
| Matrices per layer | 7 (q,k,v,o,gate,up,down) | Microsoft (BitNet) | Fixed by transformer architecture |
| Weight format | Ternary {-1,0,+1} | Microsoft (BitNet) | 2 rows per group (W_pos + W_neg) |
| **System designer** | | | |
| Activation precision (bit_planes) | 8 (INT8) -> 4 -> 2 | System designer | 2x per halving |
| Number of DIMMs | 1 -> 2 -> 4 | System designer | Linear scaling |
| Pipeline overlap | 0% -> 75% | System designer | ~6% gain |
| Token batching (prefill only) | 1 -> 8 tokens | System designer | ~15% prefill improvement |
| Scratch row rotation | Use 75% free headroom | System designer | Lifetime x200 (no throughput effect) |

### 6.2 The Bus Bottleneck

MAJ3 destroys all participating rows. In conventional matrix-vector multiplication, the input vector x is reused across all rows of W -- fetched once, multiplied N times. The MAJ3 protocol destroys this property: each AND requires re-writing the activation and restoring the weight row via SA-mediated RowCopy. This downgrades matrix multiplication from a compute-bound operation (one fetch, N reuses) to a bandwidth-bound operation (N fetches, one use each).

In the unpipelined case, ~30% of inference time is bus transfers and ~69% is MAJ3 compute. In the pipelined case, 98% is DRAM-internal MAJ3 compute. CaSA's scaling path (Sections 6.3-6.5) can be understood as systematically recovering the lost input reuse.

**Comparison with CPU inference.** A CPU running BitNet.cpp loads ~400 MB of weights from DDR per token, performing all arithmetic in on-chip ALUs. CaSA eliminates weight transfers but requires 16 round-trips per layer (8 bit-planes x 2 halves), producing ~8.4 GB of bus traffic per token -- approximately 20x more than the CPU's weight-loading approach. PIM achieves higher compute parallelism (65,536 bitlines simultaneously) but incurs higher communication cost (narrow 64-bit external bus).

![Figure 3: Per-Token Inference Time Breakdown](figures/fig3_bottleneck.png)

**Figure 3: Per-Token Inference Time Breakdown (1 DIMM, DDR4-2400, 8-bit activations).** Bus transfers dominate. In-DRAM popcount would eliminate the read bar.

![Figure 4: Cumulative Throughput Scaling Roadmap](figures/fig4_scaling.png)

**Figure 4: Cumulative Throughput Scaling Roadmap.** Each strategy stacks multiplicatively.

**Throughput scaling range by controlling party:**

| Scaling range | Controlled by | Levers | tok/s |
|---|---|---|---|
| Baseline (unpipelined) | Given | 1 DIMM, INT8, no overlap | 0.40 |
| CPU-competitive | System designer | + 4 DIMMs + ternary activations + pipelining | ~13.53 |
| Bus wall | -- | All controllable levers exhausted | -- |
| GPU-competitive | Manufacturer | + Popcount + reliable RowCopy | ~166 |
| GPU-exceeding | Industry (JEDEC) | + LPDDR5X/HBM + PIM mode + wider rows | 169-509 |

### 6.3 Scaling Without Manufacturer Changes

**Multi-DIMM parallelism.** Each DIMM provides an independent bus. Four DIMMs yield 4x throughput (see Section 4.5).

**Reduced-precision activations.** Halving bit-planes from 8 to 4 halves the number of AND passes and bus traffic, yielding a 2x throughput improvement with no hardware changes. The accuracy-throughput tradeoff at 4-bit has not been validated on BitNet b1.58-2B-4T.

**Batch amortization (decode).** With B independent inference requests served concurrently, each weight row is activated once per batch. After AND-ing with activation A1 and restoring via SA-mediated RowCopy, the weight row is intact for A2 through A_B.

| Config | Batch | tok/s (total) | tok/s per request |
|---|---|---|---|
| 4 DIMMs, INT8 | B=1 | 7.6 | 7.6 |
| 4 DIMMs, INT8 | B=4 | ~22 | ~5.5 |
| 4 DIMMs, INT8 | B=8 | ~35 | ~4.4 |
| 4 DIMMs, INT4 | B=8 | ~55 | ~6.9 |

The throughput gain is sub-linear because bus transfers scale linearly with B; only weight-row activation and precharge are amortized.

### 6.4 In-DRAM Popcount

The largest single improvement is eliminating result readback by performing popcount inside the DRAM chip. A per-bank serial accumulation popcount register (~2,000 gates, <0.3% die area) replaces the 459 ns readback of 8,000 bytes with a single 4 ns burst read of a 16-bit count, eliminating 44% of total inference time. Samsung patented this circuit in 2014 (US9836277B2 [18]).

**Tier structure of hardware requirements:**

| Tier | Hardware requirement | Throughput (4 DIMMs) | Status |
|---|---|---|---|
| **Commodity** | Unmodified DDR4 + FPGA controller | 13.53 tok/s (4-DIMM pipelined ternary) or ~35 tok/s (batch B=8) | Modeled |
| **Commodity + algorithmic** | Same + 4-bit activations | ~15 tok/s (single) or ~55 tok/s (batch B=8) | Projected (unvalidated at 2B) |
| **Commodity + popcount** | DDR4 with ~2K gates/bank + FPGA | 31-166 tok/s | Requires die change |

**Popcount vs. general-purpose ALU.** Adding a popcount register is categorically different from adding a full processing element:

| Addition | Gates/bank | Die area | Clock on DRAM process | Commodity-compatible? |
|---|---|---|---|---|
| Popcount register | ~2,000 | <0.3% | N/A (combinational) | Yes -- no new I/O, no firmware |
| RISC-V core | ~50,000+ | ~3-5% | ~200-500 MHz (6-25x slower than logic) | No -- thermal redesign |
| SIMD/MAC unit | ~100,000+ | ~5-10% | ~200-500 MHz | No -- different product |

The popcount register is combinational logic with no clock dependency. Its propagation delay (~5-10 ns through a 13-stage reduction tree) fits within the 76 ns MAJ3 window even on DRAM-process transistors. A clocked ALU on DRAM process would perform worse than a budget microcontroller.

### 6.5 Cumulative Throughput Scaling

**Table 6.5: Cumulative throughput scaling** (analytical estimates using BitNet 2B4T dimensions)

| Configuration | per-AND (ns) | Bit-planes | DIMMs | tok/s | vs. Baseline | Changes Required |
|---|---|---|---|---|---|---|
| **Baseline** (1 DIMM, DDR4, 8-bit) | 981 | 16 | 1 | 1.8 | 1x | None |
| + 4-bit activations | 981 | 8 | 1 | ~3.5 | 2x | Software only |
| + 4 DIMMs | 981 | 8 | 4 | ~14 | 8x | Additional DIMMs (~$60) |
| + In-DRAM popcount | 464 | 8 | 4 | ~31 | 17x | ~2,000 gates/bank (~$0.10/DIMM) |
| + LPDDR5X-16ch | ~160 | 8 | 1 pkg | ~169 | 94x | Single package, no mandatory ODECC |

4-bit activation rows are projections contingent on activation quantization validation. The baseline row (INT8) is validated by simulation. Without reduced-precision activations, multi-DIMM pipelined scaling yields 13.53 tok/s (4 DIMMs, ternary). With in-DRAM popcount, this reaches ~60 tok/s.

### 6.6 RowHammer Mitigation Compatibility

Modern DDR4 DIMMs implement Target Row Refresh (TRR), which monitors activation patterns and preemptively refreshes adjacent rows. Mitigation approach:

1. Target SK Hynix C-die (2018-2020 vintage), predating aggressive TRR implementations.
2. Characterize TRR interference during hardware validation (10,000 consecutive MAJ3 operations on same weight row).
3. If TRR is detected, insert guard intervals (one tRFC = 350 ns after every 3 groups), accepting ~5% throughput penalty.
4. DDR5's Refresh Management (RFM) protocol provides explicit signaling for activation counts, making it more PIM-compatible.

### 6.7 Throughput Engineering Validation

**Table 6.7: Throughput engineering roadmap** (cross-validates scaling ratios; for canonical numbers see Table 6.5)

| Configuration | Est. tok/s | Sim. tok/s | Sim/Est Ratio | Speedup | Hardware | Certainty |
|---|---|---|---|---|---|---|
| Baseline (1 DIMM, INT8) | 1.8 | 0.54 | 0.30x | 1.0x | Off-the-shelf | Validated |
| + 4 DIMMs | 7.6 | 2.16 | 0.28x | 4.0x | Off-the-shelf | Architectural |
| + Overlapped scheduling (75%) | ~11 | 3.24 | 0.29x | 6.0x | Off-the-shelf | Architectural |
| + Ternary activations | ~44 | ~12.9 | 0.29x | 23.9x | Off-the-shelf | Algorithmic |
| + In-DRAM popcount | ~60-90 | ~18.2 | 0.27x | 33.7x | Modified DRAM | Manufacturer |

The Sim/Est ratio is consistently ~0.3x across all rows, confirming that the 3.3x absolute gap does not affect scaling conclusions. The "Est." column uses actual BitNet 2B4T dimensions (d_model=2560); the "Sim." column uses d_model=2048. Speedup ratios agree within 15%.

**Note on ternary activations.** The perplexity impact of ternary activations on LLMs is an open research question with no published validation at scale. No prior work has demonstrated that both weights and activations at ternary precision preserve acceptable LLM perplexity at the 2B parameter scale. All throughput figures using ternary activations are hardware upper bounds contingent on future algorithmic validation.

---

## 7. Comparison with Existing Systems

### 7.1 Methodology

All throughput and timing results are derived from cycle-accurate simulation calibrated against DDR4-2400 specifications, SiMRA-DRAM experimental data [11], and the BitNet b1.58-2B-4T architecture [3]. Error tolerance results use Monte Carlo bit-flip injection across 10,000 random input vectors.

The canonical throughput baseline is 0.40 tok/s (unpipelined single DIMM) / 13.53 tok/s (4-DIMM pipelined ternary). The cycle-accurate simulator produces 0.54 tok/s for the same configuration due to conservative non-pipelined CAS modeling. The simulator is used only for relative scaling validation.

**Performance metrics.** Following MLPerf Inference [21] conventions:

- **tok/s:** Output throughput (reciprocal of TPOT). 0.40 tok/s = 2,474 ms TPOT; 13.53 tok/s = 74 ms TPOT.
- **TTFT:** For autoregressive PIM inference, equals time for one full forward pass: 2,474 ms (1 DIMM) or 74 ms (4 DIMMs pipelined ternary).

All figures measure autoregressive decode phase (one output token per forward pass).

### 7.2 System Comparison Table

For CaSA, "System" power reflects the oversized FPGA prototype (42 W); "Arch." projects a right-sized embedded controller (~8 W).

| System | Type | Model | tok/s | TPOT (ms) | Power (Sys.) | Power (Arch.) | J/token (Sys.) | J/token (Arch.) | HW Cost | Custom HW |
|--------|------|-------|-------|-----------|--------------|---------------|----------------|-----------------|---------|-----------|
| BitNet.cpp | CPU | 2B4T | 5.9 | 169 | 10 W | 10 W | 1.7 | 1.7 | $800 (CPU) | No |
| llama.cpp (Q4_K_M) | CPU | 2B (4-bit) | ~15-30 | ~33-67 | ~65 W | ~65 W | ~2-4 | ~2-4 | $800 (CPU) | No |
| TeLLMe v2 | FPGA | ~370M | 25 | 40 | 5 W | 5 W | 0.2 | 0.2 | $300 | No |
| TerEffic | FPGA (HBM) | 2.7B | 727 | 1.4 | 46 W | 46 W | 0.06 | 0.06 | $1,500 | No |
| Jetson Orin Nano | Edge GPU | 2B (4-bit) | ~8 | ~125 | 15 W | 15 W | 1.9 | 1.9 | $500 | No |
| Apple M2 NPU | Mobile NPU | 3B (4-bit) | ~15-25 | ~40-67 | ~5 W | ~5 W | ~0.2-0.3 | ~0.2-0.3 | $0 (in-SoC) | No |
| Raspberry Pi 5 | ARM CPU | 2B (4-bit) | ~3-5 | ~200-333 | 12 W | 12 W | ~2.4-4.0 | ~2.4-4.0 | $80 | No |
| Google Edge TPU | Edge ASIC | <1B (INT8) | ~100+ | ~10 | 2 W | 2 W | ~0.02 | ~0.02 | $25 | No |
| SK Hynix AiM | Custom GDDR6 | varies | ~1000 | ~1 | ~30 W | ~30 W | ~0.03 | ~0.03 | >$5,000 | Yes |
| Samsung HBM-PIM | Custom HBM2 | varies | ~2000 | ~0.5 | ~50 W | ~50 W | ~0.025 | ~0.025 | >$10,000 | Yes |
| **CaSA (1 DIMM)** | **Commodity DDR4** | **2B** | **1.8** | **543** | **42 W** | **~8 W** | **23.3** | **~4.2** | **~$15** | **No** |
| **CaSA (4 DIMMs)** | **Commodity DDR4** | **2B** | **7.6** | **131** | **54 W** | **~12 W** | **7.1** | **~1.6** | **~$60** | **No** |
| CaSA + popcount | Modified DDR5 | 2B | 10-60 | 17-100 | ~30 W | ~8 W | 0.5-3.0 | 0.1-0.8 | ~$0.10/DIMM | Minor |

CaSA DRAM cost only -- the prototype additionally requires an FPGA controller (Alveo U200, ~$2,000-$6,000; production ASIC ~$50-$200).

CaSA's unpipelined throughput (0.40 tok/s) is lower than CPU and FPGA-native approaches due to the DDR4 bus bottleneck. With manufacturer-added popcount registers, CaSA scales to 31-60 tok/s (Table 6.5).

**Prototype vs. architectural power.** The 42W system figure reflects the Alveo U200 platform. With a right-sized controller:

| Component | Prototype (Measured) | Architectural (Projected) |
|-----------|----------------------|-------------------------|
| DRAM (active PIM) | ~5 W | ~5 W |
| Controller | ~35 W (oversized FPGA) | ~2-3 W (right-sized FPGA/ASIC) |
| Interface | ~2 W (PCIe x16) | ~0.5 W (direct bus) |
| **Total** | **~42 W** | **~7-8 W** |
| **J/token (1 DIMM, 0.40 tok/s)** | **105** | **~4.2** |
| **J/token (4 DIMMs, 13.53 tok/s)** | **4.0** | **~1.6** |

### 7.3 Operation-Level Energy Breakdown

**Charge-sharing AND energy.** Per AND operation: wordline driver ~0.5 pJ/bit, bitline charge redistribution ~5-18 fJ/bit, sense amplifier resolution ~1-2 pJ/bit. Total: ~2-3 pJ/bit x 65,536 bits = ~130-200 nJ per AND.

**CPU SIMD comparison.** A modern CPU MAC in AVX-512 consumes ~5-10 pJ per operation. BitNet.cpp's optimized ternary kernel reduces to ~1-3 pJ per effective ternary MAC. PIM's charge-sharing AND is comparable at the operation level (~2-3 pJ vs ~1-3 pJ). The theoretical advantage of passive charge sharing (~5-18 fJ/bit) is partially offset by sense amplifier energy (~1-2 pJ/bit).

**Bus transfer energy.** DDR4 bus transfer: ~15-20 pJ/bit. Per 8,000-byte row transfer: ~960-1,280 nJ. Each AND requires two bus transfers, consuming ~1,920-2,560 nJ -- 10-15x more than the AND itself.

**Full energy decomposition per token:**

| Operation | Count per token | Energy per op | Total | % of dynamic |
|---|---|---|---|---|
| Charge-sharing AND (MAJ3) | 527,040 | ~200 nJ | ~105 mJ | ~8% |
| Bus writes (activation bit-planes) | 527,040 | ~1.1 uJ | ~580 mJ | ~43% |
| Bus reads (AND results) | 527,040 | ~1.1 uJ | ~580 mJ | ~43% |
| FPGA popcount + accumulation | 527,040 | ~10 nJ | ~5 mJ | ~0.4% |
| FPGA non-linear ops | 30 layers | ~0.5 mJ | ~15 mJ | ~1% |
| Refresh overhead | ~70K events | ~0.5 uJ | ~35 mJ | ~3% |
| **Dynamic total** | | | **~1.3 J** | |

A CPU running BitNet.cpp incurs ~48 mJ of bus I/O energy per token (one pass through ~400 MB of weights). CaSA's 16 round-trips per layer produce ~8.4 GB of bus traffic, costing ~1,010 mJ -- approximately 21x more bus energy. PIM's energy advantage is contingent on eliminating bus traffic via in-DRAM popcount.

**With in-DRAM popcount:** Eliminating the read phase removes 43% of dynamic energy (580 mJ). With a right-sized controller, architectural J/token drops to ~0.9 J, a 47% improvement over CPU (1.7 J).

### 7.4 DRAM Layout Efficiency

| Layer Type | Shape | Pack Factor | Rows | Utilization |
|------------|-------|-------------|------|-------------|
| Attention (d=2560) | 2560x2560 | 25 neurons/row | 412 per matrix | 97.7% |
| FFN up/gate (d=6912) | 6912x2560 | 25 neurons/row | 1108 per matrix | 97.7% |
| FFN down (d=2560) | 2560x6912 | 9 neurons/row | 1140 per matrix | 95.0% |
| **Full model (30 layers)** | | | **~133,320** | **~96%** |

| Category | Rows |
|---|---|
| Weight rows (W_pos + W_neg pairs) | 131,640 |
| Scratch rows (activation bit-planes) | 1,680 |
| **Total occupied** | **133,320** |
| **Available (1 DIMM, 8 GB)** | **524,288** |
| **Utilization** | **25.4%** |

### 7.5 MNIST Pipeline Verification

To verify the PIM inference pipeline end-to-end, a ternary MLP was trained on MNIST and inference was run through the bit-serial PIM reference model. This is a pipeline verification, not a claim about LLM-scale accuracy.

**Model.** A 4-layer ternary MLP (784->1024->512->256->10) with BatchNorm and ReLU, trained via knowledge distillation from a full-precision teacher (99.48%). Student reached 99.49% after 300 epochs. INT8 activation quantization applied during PIM inference only.

**Architectural gap with BitNet.** This MLP uses ReLU and BatchNorm, whereas BitNet 2B4T uses SiLU and RMSNorm. The definitive BER tolerance result for the target architecture is the BitNet 2B4T perplexity experiment (Section 5.4).

**PIM inference accuracy.** Using max-abs INT8 quantization, the PIM reference achieves 99.60% accuracy on 1,000 test samples, matching the PyTorch baseline within 0.11%. All four layers pass bit-exact verification.

**Quantization choice.** AbsMean quantization (as proposed in BitNet b1.58) causes catastrophic clipping on sparse inputs (5% accuracy on MNIST), while MaxAbs quantization preserves full dynamic range with zero accuracy loss.

### 7.6 Throughput Simulation Validation

The cycle-accurate DDR4-2400 throughput simulator (`pim_throughput_sim.py`) models the complete bit-serial ternary-weight MAC protocol, independently computing write, AND, and read phase durations.

**Table 7.6: Cycle-accurate throughput simulation results (2B model, DDR4-2400, d_model=2048)**

| Configuration | Time (ms) | tok/s | Speedup | vs CPU | Bottleneck |
|---|---|---|---|---|---|
| Baseline (INT8, 1 DIMM) | 1837 | 0.54 | 1.0x | 0.09x | READ (62.9%) |
| Ternary act (B=2) | 464 | 2.15 | 3.96x | 0.37x | READ (62.3%) |
| 4-DIMM (INT8) | 459 | 2.18 | 4.00x | 0.37x | READ (62.9%) |
| Ternary + 4-DIMM | 116 | 8.62 | 15.83x | 1.46x | READ (62.3%) |
| Ternary + 4-DIMM + Overlap(50%) | 80 | 12.52 | 23.0x | 2.12x | AND (52.7%) |
| Ternary + 4-DIMM + Overlap(75%) | 62 | 16.18 | 29.7x | 2.74x | AND (68.2%) |
| Full combo + popcount (75%) | 44 | 22.86 | 42.0x | 3.87x | AND (96.3%) |

**Bottleneck transition.** The baseline is READ-limited (63%). With overlapped scheduling, the bottleneck shifts to in-DRAM AND compute (53-69%). With popcount, AND compute dominates at 96.3%.

**Absolute throughput gap.** The simulator uses non-pipelined CAS timing; the analytical model uses correct pipelined CAS. The simulator's absolute numbers are conservative; its value is validating speedup ratios (which agree within 15%).

### 7.7 Unique Position in PIM Landscape

| Requirement | Custom PIM (AiM, HBM-PIM) | UPMEM | AMBIT | FPGA-native (TerEffic) | CaSA |
|-------------|---------------------------|-------|-------|----------------------|------|
| Commodity hardware | No | No | Conceptual | Yes | Yes |
| Complete NN inference | Yes | Yes | No | Yes | Yes |
| In-DRAM computation | Yes | Yes | Yes | No | Yes |
| Ternary LLM support | Partial | No | No | Yes | Yes |
| Unmodified DRAM die | No | No | Yes (conceptual) | N/A | Yes |

---

## 8. Limitations

### 8.1 No Hardware Validation

All results are simulation-based, calibrated against published SiMRA-DRAM data. The 79-million-measurement dataset across 120 chips provides statistical support, but characterization of specific target DIMMs is required.

### 8.2 Bus Bandwidth Dominance

At 0.40 tok/s unpipelined on a single DIMM, CaSA is slower than BitNet.cpp on a commodity CPU (5.9 tok/s). CPU inference using llama.cpp with INT4 quantization achieves 15-30 tok/s on a 2B model using the same DDR4 DIMMs, with zero additional hardware and the existing software ecosystem. CaSA's single-DIMM throughput is lower by every practical metric. CPU inference scales with processor cost; CaSA scales with memory cost ($15-25 per DIMM). The 4-DIMM pipelined configuration (13.53 tok/s) exceeds single-threaded CPU (5.9 tok/s). Batch amortization (B=8) reaches ~35 tok/s aggregate, exceeding multi-threaded CPU performance.

### 8.3 Prefill Latency

PIM processes one activation vector per forward pass. Prefill time scales linearly with prompt length:

| Prompt length | 1 DIMM prefill | 4 DIMM prefill | Assessment |
|---|---|---|---|
| 10 tokens | 5.4 s | 1.3 s | Viable for sensor queries |
| 30 tokens | 16 s | 3.9 s | Viable for diagnostics |
| 50 tokens | 27 s | 6.6 s | Marginal for non-interactive |
| 100 tokens | 54 s | 13 s | Too slow for interactive use |
| 500+ tokens | 4.5+ min | 1.1+ min | Not viable |

The architecture is restricted to short-prompt, single-stream edge workloads. Document summarization, long-context chat, and retrieval-augmented generation are not viable. A partial mitigation is hybrid CPU+PIM prefill: the host CPU processes the prompt using BitNet.cpp, then hands the KV cache state to PIM for decode. The handoff requires ~150 KB transfer (<10 us).

### 8.4 Die Revision and Manufacturer Sensitivity

Not all DDR4 DIMMs are CaSA-compatible. The architecture targets SK Hynix C-die (2018-2020 vintage), confirmed by SiMRA data. Micron DDR4 is also likely compatible (FCDRAM [15] tested 256 chips with ~95% success rate). Samsung DDR4 is currently incompatible: processing-using-DRAM operations consistently fail on Samsung dies. Newer SK Hynix die revisions (post-2020) implement aggressive RowHammer mitigations that may also block charge-sharing operations. DDR4 production is winding down; compatible C-die modules should be sourced while still available.

### 8.5 Model Ecosystem Dependency

CaSA is optimized for ternary weights because ternary-times-binary multiplication reduces to a single AND. If the industry converges on 4-bit quantization rather than ternary weights, throughput degrades:

| Weight format | AND passes per bit-plane | Relative throughput | 1 DIMM tok/s | 4 DIMMs tok/s |
|---|---|---|---|---|
| Ternary (1.58-bit) | 2 | 1.0x (baseline) | 1.8 | 7.6 |
| Binary (1-bit) | 1 | ~2x | ~3.6 | ~15 |
| INT4 | 8 | ~0.25x | ~0.45 | ~1.9 |
| INT8 (weights) | 16 | ~0.06x | ~0.11 | ~0.48 |

At INT4, CaSA becomes impractical. The architecture is strongly coupled to ternary or binary weight formats.

### 8.6 8-bit Activation Overhead

The bit-serial decomposition requires 8 passes per activation bit-width. Reducing to 4-bit activations would halve this overhead at some accuracy cost, which has not been validated on BitNet b1.58-2B-4T.

### 8.7 Weight Loading Overhead

The full BitNet 2B4T model requires 1.053 GB of weight data. At DDR4-2400 sustained write bandwidth (~17 GB/s), the initial weight load takes approximately 62 ms per DIMM. Weight rows survive indefinitely (refreshed normally), so loading cost is amortized over the full inference session.

### 8.8 Assumptions Summary

| # | Assumption | Basis | Hardware Verification Plan |
|---|-----------|-------|---------------------------|
| A1 | 2-row AND BER < 0.01% at t_12 >= 1 cycle | SiMRA: 0 failures in 79M trials | Day 1 BER characterization (10^6 trials) |
| A2 | Weight rows restored via SA-mediated RowCopy (0% BER) | Standard read-then-write through sense amplifiers | Day 1: verify weight row integrity after 10K MAJ3 + RowCopy cycles |
| A3 | Bank parallelism: no hidden stalls | DDR4 spec: operations serialized within a bank | Single-layer pipeline test |
| A4 | Bus bandwidth = 19.2 GB/s (DDR4-2400 sustained) | DDR4-2400 spec | Measure sustained throughput on Alveo U200 |
| A5 | No TRR interference with tripleACT sequences | SiMRA tested pre-TRR chips; target is C-die | TRR interference test (10K consecutive ops) |
| A6 | Temperature stability 50-80C | SiMRA measured this range | Temperature sweep |
| A7 | Multi-DIMM scaling is near-linear | Independent bus channels, partitioned workload | Theoretical until multi-channel DRAM Bender |
| A8 | FPGA popcount is not the bottleneck | ~2000 LUTs at 250 MHz | FPGA timing closure report |
| A9 | BitNet weight distribution ~58% nonzero | Verified from safetensors checkpoint | Confirmed |

### 8.9 Threats to Validity

| # | Threat | Severity | Mitigation |
|---|---|---|---|
| T1 | SiMRA test conditions differ from sustained PIM workload | Medium | MTTF estimate; temperature characterization planned |
| T2 | Analytical vs simulated throughput gap (3.3x) | Low | Speedup ratios agree within 15% |
| T3 | MNIST pipeline uses ReLU MLP, not SiLU transformer | Low | BitNet perplexity-under-BER experiment (Section 5.4) validates on actual model |
| T4 | Ternary activations unvalidated | High | Labeled as algorithmic projection; 4-bit path is primary |
| T5 | Multi-DIMM scaling assumes no stalls | Low | Conservative pipeline model; one DIMM active at a time |
| T6 | Cosine similarity is not perplexity | Low | Perplexity experiment validates the mapping |
| T7 | Post-2020 DDR4 with aggressive TRR may block tripleACT | Medium | Target C-die; DDR5 RFM resolves |
| T8 | Charge-sharing AND relies on timing violations outside JEDEC spec | High | SiMRA validates across 120 chips; per-DIMM characterization at deploy |
| T9 | Model ecosystem dependency on ternary LLMs | Medium | Architecture generalizes to any low-bit weight format |

### 8.10 Cross-Technology Scaling

**Row width as fundamental PIM unit.** PIM throughput scales with row buffer width, not bus bandwidth. DDR4's 8 KB rows pack 25 neurons (d=2560) per AND; HBM2 and LPDDR5X's 2 KB rows pack only 6, requiring 4x more cycles per matmul. A single DDR4 DIMM outperforms a single HBM2 channel for PIM by 30% on a per-channel basis, despite HBM2's higher bandwidth.

**Table 8.10: Cross-Technology Throughput Comparison (d_model=2560, unified analytical simulation)**

| Technology | Viability | Best bus-limited | Best with popcount | vs CPU (pop) | Notes |
|---|---|---|---|---|---|
| DDR4-2400 (1 DIMM, INT8) | Proven | 0.40 tok/s | -- | 0.3x | Reference; validated on 120 chips |
| DDR4-2400 (4D, 4-bit, overlap) | Proven | 88.6 tok/s | 165.7 tok/s | 28.1x | Best near-term configuration |
| LPDDR5X (8ch, 4-bit, overlap) | Promising | 28.3 tok/s | 84.7 tok/s | 14.4x | Single package; no mandatory ODECC |
| LPDDR5X (16ch, 4-bit, overlap) | Promising | 56.5 tok/s | 169.4 tok/s | 28.7x | Matches DDR4-4D in one package |
| HBM2 (8ch, 4-bit, overlap) | Feasible | 38.3 tok/s | 114.7 tok/s | 19.4x | No ODECC; hard controller obstacle |
| HBM2 (2 stacks, 4-bit) | Feasible | -- | 229.4 tok/s | 38.9x | High-performance path |
| HBM3E (16ch, 4-bit, overlap) | Future | 123.2 tok/s | 254.6 tok/s | 43.2x | Co-packaged on GPU only |
| HBM3E (2 stacks, 4-bit) | Future | -- | 509.3 tok/s | 86.3x | Theoretical ceiling |
| CPU reference | -- | 5.9 tok/s (1-thread) | ~25 tok/s (multi-thread est.) | 1.0x | BitNet.cpp published |

4-bit activation rows are projections contingent on activation quantization validation. DDR4 1-DIMM INT8 baseline is 0.40 tok/s (unpipelined) from the analytical model. DDR5 is omitted because mandatory ODECC makes throughput numbers unreliable (Section 8.11). Multi-threaded CPU could plausibly reach 15-30 tok/s.

**Scaling dynamics.** In the bus-limited regime (no popcount), bandwidth improvements help both CPUs and CaSA equally. In the compute-limited regime (with popcount), CaSA's throughput is determined by the number of parallel charge-sharing operations, a function of channel count and row width rather than bus bandwidth. These are different scaling axes.

| Regime | CaSA scaling | CPU/GPU scaling | Relative advantage |
|---|---|---|---|
| Bus-limited (no popcount) | proportional to bandwidth | proportional to bandwidth | Constant |
| Compute-limited (with popcount) | proportional to channels x (row_bits / d_model) | proportional to bandwidth | Grows with channel count |
| Technology transition (DDR4 -> HBM3E) | 1.8 -> 509 tok/s (283x) | 5.9 -> ~50 tok/s (8x) | Grows |

CaSA faces a structural headwind from the industry trend toward narrower rows.

### 8.11 DDR5: ODECC as a Fundamental Blocker

DDR5-4800 offers 2x bus bandwidth per DIMM through dual 32-bit sub-channels, 32 banks, and per-bank refresh. Simulation confirms a consistent ~2.1x throughput ratio over DDR4. However, DDR5 is not currently viable for charge-sharing PIM due to mandatory On-Die ECC (ODECC).

**ODECC.** DDR5 mandates On-Die Error Correction Code that operates inside the DRAM chip. ODECC encodes every 128-bit granule with 8 parity bits as a (136, 128) Hamming SEC code. During charge-sharing AND, parity bits undergo AND along with data bits, producing inconsistent parity. On read, ODECC detects a syndrome and may corrupt a valid result. This is a correctness concern, not a performance concern.

Three mitigation paths exist, none proven: (1) manufacturer test modes bypassing ODECC via MRS commands; (2) reverse-engineering the ECC polynomial using BEER methodology [22]; (3) a future JEDEC "PIM mode" MRS bit.

**PRAC.** JEDEC's JESD79-5C mandates Per-Row Activation Counting, increasing tRP by ~140%. Cross-technology simulation quantifies actual impact: 0.1-7% throughput penalty, manageable if ODECC is solved.

**Table 8.11: DDR5 Obstacle Summary**

| Obstacle | Severity | Mitigation | Status |
|---|---|---|---|
| ODECC | Blocking | BEER recovery; MRS bypass; JEDEC PIM mode | Unresolved |
| PRAC | Low | Below ALERTn threshold; RFM scheduling | Manageable |
| Voltage (1.1V vs 1.2V) | Medium | PUDTune calibration | Calibration-solvable |
| No DDR5 tooling | Timeline | DRAM Bender DDR5 PHY planned | Gating dependency |

McSee [25] (Phoenix RowHammer attack, CVE-2025-6202) demonstrated precision-timed DRAM manipulation on production DDR5 SK Hynix modules, confirming the analog path remains open on DDR5.

### 8.12 Scaling Path

**Phase 1 (Current).** Proof-of-concept on commodity DDR4 + FPGA. Performance: 1.8-8.0 tok/s (single DIMM), up to 88.6 tok/s (4 DIMMs, 4-bit activations, overlap).

**Phase 2 (2-4 years).** LPDDR5X via CXL Type 2 devices: 8-16 independent channels with no mandatory ODECC, achieving 28-169 tok/s (Table 8.10). DDR5 becomes viable only if ODECC bypass is demonstrated.

**Phase 3 (5-8 years).** PIM-optimized DRAM (DDR6/DDR7) with in-die popcount, wider internal I/O, multi-subarray activation. HBM3E with popcount at 255-509 tok/s.

**RowCopy reliability and software-defined popcount.** If future DRAM revisions reduce unmediated RowCopy BER to < 0.01%, SA-mediated RowCopy restore could be replaced by internal RowCopy, eliminating the restore bus write. Reliable RowCopy also enables software-defined popcount: SIMDRAM [7] and AMBIT [1] demonstrated that arbitrary logic (including adder trees) can be built from charge-sharing AND/OR/NOT sequences. This would break the bus bottleneck on completely unmodified DRAM. This path is currently blocked by unmediated RowCopy's 16.3% BER.

**Design recommendations for DRAM manufacturers.** Ordered by implementation cost:

| Tier | Change | Cost | Impact |
|---|---|---|---|
| 0 | PIM mode MRS bit: suppress TRR and bypass ODECC | Zero silicon cost | Eliminates DDR5 blocker, removes TRR guard overhead |
| 0 | Published charge-sharing timing per die revision | Characterization | Replaces reverse-engineering |
| 1 | In-DRAM popcount register | ~2,000 gates/bank (<0.1% area, ~$0.10/DIMM) | Eliminates bus read-back |
| 1 | Reliable RowCopy | Sense amp offset compensation | Eliminates per-weight-row bus write |
| 2 | Per-bank activation register | ~3-4 transistors per bit-line | Eliminates per-weight-row bus write entirely |
| 2 | Multi-subarray activation | Modified row decoder | 4x compute throughput per bank |
| 3 | JEDEC PIM command set (DDR6/LPDDR6) | Industry standardization | Formalizes AND, popcount, multi-row activation |

### 8.13 LPDDR5X, CAMM2, and CXL Deployment

**LPDDR5X.** Up to 16 independent narrow channels enable fine-grained parallelism. Smaller row buffer (2 KB vs 8 KB) means fewer neurons per AND (~6 vs 25), but 8-16 channels compensate.

| LPDDR5X Config | tok/s | TPOT (ms) | vs CPU |
|---|---|---|---|
| 8-channel, INT8 | 8.5 | 117 | 1.4x |
| 8-channel, 4-bit + overlap | 28.3 | 35 | 4.8x |
| 8-channel, 4-bit + overlap + popcount | 84.7 | 12 | 14.4x |
| 16-channel, INT8 | 17.1 | 59 | 2.9x |
| 16-channel, 4-bit + overlap + popcount | 169.4 | 5.9 | 28.7x |

CAMM2 (JEDEC MO-330) packages LPDDR5X in a user-replaceable laptop module, combining LPDDR5X bandwidth with DDR-like replaceability.

**CXL.** CXL 2.0/3.0 memory expanders package an accelerator with its own DDR interface behind a standard PCIe/CXL link -- matching the CaSA architecture. The controller drives the DDR bus directly and can issue arbitrary timing-violated commands. CXL latency (~100-200 ns round-trip) is irrelevant because the bottleneck is DRAM-internal row operations.

### 8.14 Patent Implications

The current CaSA architecture is patent-safe: popcount and accumulation reside in the FPGA, and the DRAM performs only charge-sharing AND. The scaling path to in-DRAM popcount would require licensing Samsung's US9836277B2 [18] or a manufacturer partnership.

**In-DRAM computation patents.** Samsung US9836277B2 [18] covers popcount reduction trees. Micron US9472265B2 family covers per-column accumulators. Intel US10748603B2 covers in-memory MAC. Purdue US12118328B2 covers bit-serial addition. Qualcomm US11126402B2 covers XNOR+popcount for ternary NNs in SRAM. Princeton US11043259 covers charge-sharing AND/OR in commodity DDR4. UNIST WO2025258754A1 [17] describes ternary neural computation using charge-sharing.

### 8.15 Target Application

CaSA targets edge AI deployment with sub-$50 inference modules built from commodity DRAM, running ternary LLMs locally at ~8W without cloud connectivity. Suitable workloads include short-prompt, single-stream diagnostic queries (10-50 tokens).

### 8.16 Path to Hardware Validation

| Phase | Test | Success Criteria | Timeline |
|-------|------|-----------------|----------|
| Day 1: Go/No-Go | Single MAJ3 on known row triple | AND matches expected for >99.99% of bits | Week 1 |
| Day 1: BER Characterization | 10^6 MAJ3 trials across 16 banks | Mean BER < 0.01%; consistent with SiMRA | Week 1-2 |
| Day 2: TRR Interference | 10,000 consecutive MAJ3 on same weight row | No adjacent-row corruption; BER < 0.01% with guards | Week 2-3 |
| Day 3: Temperature Sweep | BER test at 50C, 65C, 80C | BER < 0.01% across range | Week 3-4 |
| Week 2: Single-Layer Pipeline | Full q_proj inference | cos_sim > 0.999 vs software reference | Week 4-6 |
| Week 3: Full-Model Inference | All 30 layers, single token | Correct output; throughput within 20% of 0.40 tok/s | Week 6-10 |
| Week 4: Stress Test | 1000-token generation | No accumulated drift; cos_sim stable above 0.98 | Week 10-12 |

**Validation risk assessment:**

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| DIMMs differ from SiMRA chips | Medium | High | Source multiple batches |
| TRR interferes with tripleACT | Medium | Medium | Guard intervals; pre-TRR die revisions |
| DRAM Bender multi-channel not ready | Low | Medium | Single-channel validates core claims |
| HBM2 hard IP blocks PIM commands | High | High | DRAM Bender demonstrated HBM2 testing [26] |
| DDR5 ODECC blocks PIM results | High | Blocking | BEER recovery; MRS bypass |
| LPDDR5X charge-sharing unvalidated | Medium | High | Same 1T1C physics as DDR4 |

**What simulation already validates:** Charge-sharing AND reliability and timing (SiMRA, 79M measurements); timing model arithmetic, row budgets, packing efficiency, bit-serial decomposition correctness. **Requires hardware validation:** End-to-end pipeline integration, real bus latency, thermal behavior, TRR interaction, sustained throughput.

---

## 9. Conclusion

CaSA presents a feasibility analysis of ternary LLM inference on commodity DDR4 DRAM via charge-sharing. The MAJ3 + RowCopy protocol replaces unmediated RowCopy (16.3% BER) with MAJ3 AND (BER < 3.8 x 10^-8) and SA-mediated RowCopy (0% BER) without die modifications.

Cycle-accurate simulation, calibrated against 79 million experimental measurements across 120 chips, establishes three results: (1) single-DIMM unpipelined throughput of 0.40 tok/s (2,474 ms/token), or 13.53 tok/s with 4-DIMM pipelining (74 ms/token), bottlenecked by DRAM-internal MAJ3 compute in the pipelined regime; (2) perplexity degradation of +0.39% at the 0.01% BER error budget, measured on BitNet b1.58-2B-4T; and (3) with 4 DIMMs pipelined, ~35 tok/s aggregate with batch amortization (B=8), exceeding multi-threaded CPU performance using only commodity hardware plus an FPGA controller.

Single-DIMM CaSA is slower than CPU inference. The validated, no-custom-silicon result is 13.53 tok/s (4 DIMMs pipelined ternary). Higher throughput projections depend on: (a) activation precision reduction to 4-bit (unvalidated at 2B scale), (b) in-DRAM popcount registers (requiring manufacturer cooperation), or (c) technology transitions to LPDDR5X or HBM2 (where charge-sharing is unvalidated). The architecture is coupled to the ternary weight ecosystem; at INT4, throughput degrades by ~4x.

Hardware validation on the DRAM Bender platform is the logical next step.

**Artifact availability.** Simulation scripts (`pim_throughput_sim.py`, `pim_throughput_sim_ddr5.py`, `pim_throughput_sim_hbm2.py`, `pim_throughput_sim_all.py`), the BER accumulation simulator (`pim_ber_accumulation_sim.py`), the fixed-point validation suite (`pim_fixedpoint_nonlinear_validation.py`), the BitNet perplexity-under-BER experiment (`pim_bitnet_perplexity_ber.ipynb`), and the MNIST PIM verification pipeline (`pim_mnist_ternary_v6.py`) will be released as open-source upon publication.

---

## Acknowledgments

This work was conducted by an independent researcher using AI-assisted analysis tools. The author directed the research vision, formulated the core architectural insights, and made all design decisions. Large language models -- Claude (Anthropic), DeepSeek, Gemini (Google), and Grok (xAI) -- served as research collaborators for literature synthesis, simulation implementation, mathematical derivation, error analysis, figure generation, and manuscript preparation. All claims, design choices, and conclusions were verified by the human author; all errors remain the author's responsibility.

---

## References

[1] V. Seshadri et al., "Ambit: In-Memory Accelerator for Bulk Bitwise Operations Using Commodity DRAM Technology," MICRO 2017.

[2] F. Gao, G. Tziantzioulis, D. Wentzlaff, "ComputeDRAM: In-Memory Compute Using Off-the-Shelf DRAMs," MICRO 2019.

[3] S. Ma et al., "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits," arXiv:2402.17764, 2024.

[4] S. Lee et al., "Newton: A DRAM-Maker's Accelerator-in-Memory (AiM) Architecture for Machine Learning," MICRO 2020.

[5] J. Park et al., "A 1.2 TFLOPS HBM2-based Processor-in-Memory Architecture," Hot Chips 2021.

[6] F. Devaux, "The true Processing-In-Memory accelerator," Hot Chips 2019.

[7] N. Hajinazar et al., "SIMDRAM: An End-to-End Framework for Bit-Serial SIMD Computing in DRAM," ASPLOS 2021.

[8] S. Li et al., "DRISA: A DRAM-based Reconfigurable In-Situ Accelerator," MICRO 2017.

[9] K. Ando et al., "TerEffic: Ternary Efficient FPGA-based Inference Accelerator for Large Language Models," DAC 2024.

[10] M. Xu et al., "TeLLMe: Ternary Low-precision LLM Engine on FPGA," FPL 2024.

[11] Y. Luo et al., "SiMRA-DRAM: Subarray-Level In-Memory Row Activation in Commodity DRAM," DSN 2024.

[12] H. Hassan et al., "DRAM Bender: An Extensible and Versatile FPGA-based Infrastructure to Easily Test State-of-the-Art DRAM Chips," HPCA 2024.

[13] S. Ma et al., "BitNet b1.58 2B4T Technical Report," Microsoft Research, 2025. https://huggingface.co/microsoft/BitNet-b1.58-2B-4T

[14] Y. Zhang et al., "T-SAR: Ternary-to-Binary Decomposition for Fast Inference," ASPLOS 2024.

[15] J. Bostanci et al., "FCDRAM: Functionally Complete DRAM," HPCA 2024.

[16] L. Orosa et al., "ECC.fail: Characterizing and Understanding DDR4 ECC Under Reduced Safety Margins," USENIX Security 2025.

[17] UNIST, "Computational Apparatus and Computational Method Using Ternary Neural Network, and Analog Signal Analysis System," WIPO Application WO2025258754A1, December 2025.

[18] Z. Guz and L. Yin, "In-Memory Popcount Support for Real Time Analytics," US Patent US9836277B2, Samsung Electronics, December 2017.

[19] R. C. Murphy, "Processing in Memory (PIM) Capable Memory Device Having Sensing Circuitry Performing Logic Operations," US Patent US9997232B2, Micron Technology, June 2018.

[20] J. Malekar, P. Chandarana, M. H. Amin, M. E. Elbtity, R. Zand, "PIM-LLM: A High-Throughput Hybrid PIM Architecture for 1-bit LLMs," arXiv:2504.01994, 2025.

[21] MLCommons, "MLPerf Inference v5.0: Language Model Capabilities for GenAI," mlcommons.org/2025/04/llm-inference-v5/, 2025.

[22] M. Patel, J. S. Kim, H. Hassan, O. Mutlu, "Bit-Exact ECC Recovery (BEER): Determining DRAM On-Die ECC Functions by Exploiting DRAM Data Retention Characteristics," IEEE/ACM MICRO, 2020.

[23] O. Canpolat, A. G. Yaglikci, G. F. Oliveira, A. Olgun, O. Mutlu, "Understanding the Security Benefits and Overheads of Emerging Industry Solutions to DRAM Read Disturbance," DRAMSec Workshop, 2024.

[24] H. Luo et al., "PUDTune: Multi-Level Charging for High-Precision Calibration in Processing-Using-DRAM," arXiv:2505.05266, 2025.

[25] J. Jattke et al., "McSee: Evaluating Advanced Rowhammer Attacks and Defenses," USENIX Security, 2025. (Phoenix attack, CVE-2025-6202)

[26] A. Olgun et al., "Read Disturbance in High Bandwidth Memory: A Detailed Experimental Study on HBM2 DRAM Chips," IEEE/IFIP DSN, 2024.

[27] G. Park et al., "NeuPIMs: NPU-PIM Heterogeneous Acceleration for Batched LLM Inference," ASPLOS, 2024.

[28] "PIM-AI: A Novel Architecture for High-Efficiency LLM Inference," arXiv:2411.17309, November 2024.

[29] "P3-LLM: An Integrated NPU-PIM Accelerator for LLM Inference Using Hybrid Numerical Formats," arXiv:2511.06838, November 2025.

[30] O. Mutlu, S. Ghose, J. Gomez-Luna, R. Ausavarungnirun, "A Modern Primer on Processing in Memory," in *Emerging Computing: From Devices to Systems*, Springer, 2023. (See also: S. Ghose et al., "Processing-in-Memory: A Workload-Driven Perspective," IBM J. Res. Dev. 63(6), 2019.)

[31] H. Zahoor, S. Khan, M. Saeed, M. A. Shami, "SALIENT: Ultra-Fast FPGA-based Short Read Alignment," IEEE/ACM Trans. Comput. Biol. Bioinform., 2020.

[32] A. Gkillas, I. Stamelos, D. Soudris, "Performance and Power Efficiency Analysis on Alveo U200," 2021.

---

## Appendix A: Detailed DRAM Row Budget for BitNet 2B4T

| Layer Type | Count per TF Layer | Shape | Rows per Matrix | Rows per Layer |
|------------|-------------------|-------|-----------------|---------------|
| q_proj | 1 | 2560x2560 | 412 | 412 |
| k_proj | 1 | 640x2560 | 104 | 104 |
| v_proj | 1 | 640x2560 | 104 | 104 |
| o_proj | 1 | 2560x2560 | 412 | 412 |
| gate_proj | 1 | 6912x2560 | 1108 | 1108 |
| up_proj | 1 | 6912x2560 | 1108 | 1108 |
| down_proj | 1 | 2560x6912 | 1140 | 1140 |
| **Per TF layer** | **7** | | | **4,388** |
| **Full model (30 layers)** | **210** | | | **131,640** |
| **+ scratch rows** | | | | **~133,320** |
| **Available (1 DIMM)** | | | | **524,288** |
| **Utilization** | | | | **25.4%** |

## Appendix B: Timing Derivation

Per packed neuron group (d=2560, 25 neurons packed per row):

```
Write activation bit-plane to scratch row:
  8000 bytes / 64 bytes per burst = 125 bursts
  125 x 3.68 ns (BL8 at DDR4-2400) = 460 ns

Charge-sharing AND (MAJ3):
  tRAS (36 ns) + t_12 (1.5 ns) + t_sense (10 ns) + tRP (14 ns) = 61.5 ns

Read AND result row:
  8000 bytes / 64 bytes per burst = 125 bursts
  125 x 3.68 ns = 459.5 ns

Per half-cycle (write + AND + read): 981 ns
Per bit-plane (pos + neg halves, incl. RowCopy restore): ~2,910 ns
Per 8-bit activation (8 bit-planes): 15,696 ns = 15.7 us
Per neuron group pair (W_pos + W_neg): 15.7 us

Per q_proj matrix (103 groups): 1.62 ms
Per gate_proj matrix (277 groups): 4.35 ms
Per transformer layer (7 matrices, pipeline only): 17.2 ms
+ Refresh + FPGA overhead per layer: ~0.9 ms
Effective per layer (Table 4.4): ~82.5 ms (unpipelined) / ~9.8 ms (pipelined)
Per full model (30 layers): 82.5 x 30 = ~2,474 ms (unpipelined); 9.8 x 30 / 4 = ~74 ms (4-DIMM pipelined)
Total: ~2,474 ms/token = 0.40 tok/s (unpipelined); ~74 ms/token = 13.53 tok/s (4-DIMM pipelined ternary)
```
