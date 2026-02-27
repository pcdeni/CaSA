#!/usr/bin/env python3
"""
PIM-LLM Throughput Simulator
=============================
Cycle-accurate DDR4 timing simulator that computes tokens/second for a
2B-parameter BitNet b1.58 model running on a Processing-In-Memory DRAM system.

Models the bit-serial ternary-weight MAC protocol under various engineering
strategies: reduced activation precision, multi-DIMM parallelism, overlapped
bank scheduling, and in-DRAM popcount.

Dependencies: numpy, matplotlib (standard scientific Python stack).

Author: PIM-LLM project
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import math
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless rendering
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# ============================================================================
# DDR4-2400 Timing Parameters
# ============================================================================

@dataclass
class DDR4Timing:
    """DDR4-2400 timing parameters for a single DIMM."""
    # Clock / bus
    bus_freq_mhz: float = 1200.0          # Bus clock frequency (MHz)
    data_rate_mts: float = 2400.0          # Megatransfers per second (DDR)
    bus_width_bytes: int = 8               # 64-bit bus = 8 bytes per transfer
    burst_length: int = 8                  # BL8
    cache_line_bytes: int = 64             # 8 bytes * 8 beats = 64 bytes

    # Core timing (nanoseconds)
    tRCD_ns: float = 13.75                 # RAS-to-CAS delay
    tRAS_ns: float = 35.0                  # Row active time
    tRP_ns: float = 13.75                  # Row precharge
    tCL_ns: float = 13.75                  # CAS latency
    tBurst_ns: float = 3.33               # Burst transfer time (BL8 @ 2400 MT/s)
    tREFI_ns: float = 7800.0              # Refresh interval (7.8 us)
    tRFC_ns: float = 350.0                # Refresh cycle time (8 Gb chips)

    # Row geometry
    row_size_bytes: int = 8192             # 8 KB row buffer

    @property
    def peak_bw_GBs(self) -> float:
        """Peak bandwidth in GB/s."""
        return self.data_rate_mts * self.bus_width_bytes / 1000.0  # 19.2 GB/s


# ============================================================================
# LLM Model Parameters (2B BitNet b1.58)
# ============================================================================

@dataclass
class ModelParams:
    """Architecture parameters for a 2B-parameter BitNet b1.58 model."""
    num_layers: int = 30
    hidden_dim: int = 2048
    ffn_dim: int = 5632

    # Attention projections per layer: Q, K, V (2048->2048), O (2048->2048)
    # FFN projections per layer: gate (2048->5632), up (2048->5632), down (5632->2048)
    # Total matvecs per layer: 7

    def get_matvecs(self) -> List[Tuple[str, int, int]]:
        """Return list of (name, N_in, N_out) for all matvecs in one layer."""
        h = self.hidden_dim
        f = self.ffn_dim
        return [
            ("Q_proj",    h, h),
            ("K_proj",    h, h),
            ("V_proj",    h, h),
            ("O_proj",    h, h),
            ("FFN_gate",  h, f),
            ("FFN_up",    h, f),
            ("FFN_down",  f, h),
        ]

    @property
    def total_weight_rows_per_layer(self) -> int:
        """Approximate total output rows across all 7 matvecs."""
        return sum(n_out for _, _, n_out in self.get_matvecs())


# ============================================================================
# PIM Protocol Timing Model
# ============================================================================

def compute_bus_transfer_time(num_bytes: int, timing: DDR4Timing) -> float:
    """
    Time to transfer `num_bytes` over the DDR4 bus.

    Each cache-line (64 B) burst takes tCL + tBurst. We also pay tRCD to open
    the row at the start and tRP to precharge at the end.

    Returns time in nanoseconds.
    """
    num_bursts = math.ceil(num_bytes / timing.cache_line_bytes)
    return (timing.tRCD_ns
            + num_bursts * (timing.tCL_ns + timing.tBurst_ns)
            + timing.tRP_ns)


def compute_write_time(n_in: int, timing: DDR4Timing) -> float:
    """
    WRITE phase: write one activation bit-plane row (N_in bits) into DRAM.
    Transfer size = ceil(N_in / 8) bytes.
    """
    transfer_bytes = math.ceil(n_in / 8)
    return compute_bus_transfer_time(transfer_bytes, timing)


def compute_and_time(timing: DDR4Timing) -> float:
    """
    AND phase (charge-sharing double-activation).

    The PIM protocol issues two ACT commands:
      1. ACT on the activation row  (tRCD to sense)
      2. ACT on the weight row      (tRCD to sense, then charge-sharing AND)
    Plus tRAS to allow the row to fully sense.

    Time = 2 * tRCD + tRAS
    """
    return 2.0 * timing.tRCD_ns + timing.tRAS_ns


def compute_read_time(n_in: int, timing: DDR4Timing) -> float:
    """
    READ phase: read the AND result row (N_in bits) back to the controller.
    Transfer size = ceil(N_in / 8) bytes.
    """
    transfer_bytes = math.ceil(n_in / 8)
    return compute_bus_transfer_time(transfer_bytes, timing)


def compute_refresh_overhead_fraction(timing: DDR4Timing) -> float:
    """
    Fraction of time lost to periodic DRAM refresh.
    Every tREFI ns, the memory is unavailable for tRFC ns.
    """
    return timing.tRFC_ns / timing.tREFI_ns


# ============================================================================
# Per-Layer Simulation
# ============================================================================

@dataclass
class LayerTiming:
    """Breakdown of time components for one transformer layer (nanoseconds)."""
    write_ns: float = 0.0
    and_ns: float = 0.0
    read_ns: float = 0.0
    fpga_ns: float = 0.0
    transfer_ns: float = 0.0   # Inter-DIMM activation transfer

    @property
    def total_ns(self) -> float:
        return self.write_ns + self.and_ns + self.read_ns + self.fpga_ns + self.transfer_ns


def simulate_one_matvec(
    n_in: int,
    n_out: int,
    act_bits: int,
    timing: DDR4Timing,
    overlap_factor: float = 0.0,
    in_dram_popcount: bool = False,
) -> LayerTiming:
    """
    Simulate one matrix-vector multiply (N_in x N_out ternary weights).

    Parameters
    ----------
    n_in : int
        Input dimension (activation vector length).
    n_out : int
        Output dimension (number of weight rows to process).
    act_bits : int
        Activation precision in bits (B). Each bit-plane is processed twice
        (positive weight half, negative weight half) => 2*B passes.
    timing : DDR4Timing
        DDR4 timing parameters.
    overlap_factor : float
        Fraction of (write + read) time that can be overlapped via multi-bank
        scheduling. 0.0 = no overlap, 0.5 = conservative, 0.75 = aggressive.
    in_dram_popcount : bool
        If True, the popcount is done in DRAM, eliminating the read phase.

    Returns
    -------
    LayerTiming with per-component nanosecond breakdown.
    """
    num_passes = act_bits * 2   # B bit-planes * 2 halves (pos/neg weights)

    # --- Per-pass timing ---
    write_per_pass = compute_write_time(n_in, timing)
    and_per_pass = compute_and_time(timing)

    if in_dram_popcount:
        # Read phase eliminated: popcount happens in the sense amplifiers
        read_per_pass = 0.0
    else:
        read_per_pass = compute_read_time(n_in, timing)

    # --- Per weight-row timing (one pass processes all N_out rows) ---
    # Actually: for each pass, we iterate over all N_out weight rows.
    # The WRITE of the activation bit-plane happens once per pass (shared).
    # The AND + READ happen once per weight row per pass.

    # Corrected model:
    # Per pass:
    #   1x WRITE (activation bit-plane)
    #   N_out x (AND + READ) for each weight row
    # Total passes = 2 * B

    write_total = num_passes * write_per_pass
    and_total = num_passes * n_out * and_per_pass
    read_total = num_passes * n_out * read_per_pass

    # Apply overlap: write and read can overlap if using different banks
    bus_time = write_total + read_total
    overlap_savings = bus_time * overlap_factor
    write_total_adj = write_total * (1.0 - overlap_factor)
    read_total_adj = read_total * (1.0 - overlap_factor)
    # The AND time is compute-bound in DRAM, not overlappable with bus
    # But the savings come from bus contention reduction
    # We distribute the savings proportionally
    if bus_time > 0:
        write_total_eff = write_total - overlap_savings * (write_total / bus_time)
        read_total_eff = read_total - overlap_savings * (read_total / bus_time)
    else:
        write_total_eff = 0.0
        read_total_eff = 0.0

    # FPGA post-processing: popcount + shift-accumulate + batch-norm + activation
    # Estimate: ~10 ns per output element (pipelined, very fast relative to DRAM)
    # This includes popcount of N_in-bit vectors, weighted accumulation across
    # bit-planes, batch normalization, and SiLU/ReLU activation.
    fpga_cycles_per_output = 10.0  # ns, conservative for pipelined FPGA @ 200 MHz
    fpga_total = n_out * fpga_cycles_per_output

    return LayerTiming(
        write_ns=write_total_eff,
        and_ns=and_total,
        read_ns=read_total_eff,
        fpga_ns=fpga_total,
        transfer_ns=0.0,
    )


def simulate_one_layer(
    model: ModelParams,
    act_bits: int,
    timing: DDR4Timing,
    overlap_factor: float = 0.0,
    in_dram_popcount: bool = False,
) -> LayerTiming:
    """
    Simulate all 7 matvecs in one transformer layer.
    """
    layer = LayerTiming()
    for name, n_in, n_out in model.get_matvecs():
        mv = simulate_one_matvec(
            n_in, n_out, act_bits, timing, overlap_factor, in_dram_popcount
        )
        layer.write_ns += mv.write_ns
        layer.and_ns += mv.and_ns
        layer.read_ns += mv.read_ns
        layer.fpga_ns += mv.fpga_ns
    return layer


# ============================================================================
# Full-Model Simulation
# ============================================================================

@dataclass
class SimConfig:
    """Configuration for one simulation run."""
    name: str
    act_bits: int = 8
    num_dimms: int = 1
    overlap_factor: float = 0.0
    overlap_label: str = "none"
    in_dram_popcount: bool = False


@dataclass
class SimResult:
    """Results from one simulation run."""
    config: SimConfig
    per_token_ms: float
    throughput_toks: float
    speedup_vs_baseline: float
    vs_cpu_factor: float
    breakdown: LayerTiming   # Aggregated across all layers (nanoseconds)


def simulate_full_model(
    config: SimConfig,
    model: ModelParams,
    timing: DDR4Timing,
    baseline_ms: Optional[float] = None,
    cpu_toks: float = 5.9,
) -> SimResult:
    """
    Simulate the full LLM inference for one token.

    For multi-DIMM configurations, layers are distributed evenly across DIMMs.
    Each DIMM processes its subset of layers independently, so wall-clock time
    is total_layer_time / num_dimms + inter-DIMM transfer overhead.
    """
    # --- Compute per-layer timing ---
    layer_timing = simulate_one_layer(
        model, config.act_bits, timing, config.overlap_factor, config.in_dram_popcount
    )

    # --- Aggregate across all layers ---
    total = LayerTiming(
        write_ns=layer_timing.write_ns * model.num_layers,
        and_ns=layer_timing.and_ns * model.num_layers,
        read_ns=layer_timing.read_ns * model.num_layers,
        fpga_ns=layer_timing.fpga_ns * model.num_layers,
    )

    # --- Multi-DIMM parallelism ---
    # Layers are distributed across DIMMs. Each DIMM handles num_layers/num_dimms.
    # The compute time is divided by num_dimms.
    if config.num_dimms > 1:
        total.write_ns /= config.num_dimms
        total.and_ns /= config.num_dimms
        total.read_ns /= config.num_dimms
        total.fpga_ns /= config.num_dimms

        # Inter-DIMM transfer overhead:
        # Between each pair of consecutive DIMM boundaries, we must transfer
        # the activation vector. There are (num_dimms - 1) such transfers.
        # Transfer size: hidden_dim * act_bits / 8 bytes
        # Transfer time: size / peak_bandwidth
        transfer_bytes = model.hidden_dim * config.act_bits / 8.0
        transfer_time_per_hop_ns = transfer_bytes / (timing.peak_bw_GBs * 1e9 / 1e9)
        # peak_bw_GBs is in GB/s; transfer_bytes is in bytes
        # time = bytes / (GB/s * 1e9 bytes/GB) in seconds, * 1e9 for ns
        transfer_time_per_hop_ns = transfer_bytes / (timing.peak_bw_GBs)  # ns
        # Actually: GB/s = 1e9 bytes/s. time_s = bytes / (GBs * 1e9). time_ns = bytes / GBs
        # 19.2 GB/s => transfer_bytes / 19.2 ns? No.
        # Let's be precise:
        #   time_s = transfer_bytes / (peak_bw_GBs * 1e9)
        #   time_ns = transfer_bytes / (peak_bw_GBs * 1e9) * 1e9 = transfer_bytes / peak_bw_GBs
        # For 2048 bytes: 2048 / 19.2 = 106.67 ns. That's correct.
        transfer_time_per_hop_ns = transfer_bytes / timing.peak_bw_GBs  # ns

        num_transfers = config.num_dimms - 1
        total.transfer_ns = num_transfers * transfer_time_per_hop_ns
    else:
        total.transfer_ns = 0.0

    # --- Refresh overhead ---
    refresh_fraction = compute_refresh_overhead_fraction(timing)
    # Stretch the DRAM-bound portions (write, AND, read) by the refresh penalty
    dram_time = total.write_ns + total.and_ns + total.read_ns
    refresh_penalty = dram_time * refresh_fraction
    # Distribute proportionally
    if dram_time > 0:
        total.write_ns *= (1.0 + refresh_fraction)
        total.and_ns *= (1.0 + refresh_fraction)
        total.read_ns *= (1.0 + refresh_fraction)

    # --- Final per-token time ---
    total_ns = total.total_ns
    per_token_ms = total_ns / 1e6

    throughput = 1000.0 / per_token_ms if per_token_ms > 0 else float('inf')

    speedup = (baseline_ms / per_token_ms) if baseline_ms else 1.0
    vs_cpu = throughput / cpu_toks

    return SimResult(
        config=config,
        per_token_ms=per_token_ms,
        throughput_toks=throughput,
        speedup_vs_baseline=speedup,
        vs_cpu_factor=vs_cpu,
        breakdown=total,
    )


# ============================================================================
# Configuration Definitions
# ============================================================================

def build_configurations() -> List[SimConfig]:
    """Define all engineering strategy configurations to simulate."""
    configs = []

    # Baseline: INT8 activations, single DIMM, no overlap
    configs.append(SimConfig(
        name="Baseline (INT8, 1 DIMM)",
        act_bits=8, num_dimms=1, overlap_factor=0.0, overlap_label="none",
    ))

    # --- Strategy 1: Ternary activations ---
    configs.append(SimConfig(
        name="S1: Ternary act (B=2)",
        act_bits=2, num_dimms=1, overlap_factor=0.0, overlap_label="none",
    ))

    # --- Strategy 3: Multi-DIMM ---
    configs.append(SimConfig(
        name="S3: 2-DIMM",
        act_bits=8, num_dimms=2, overlap_factor=0.0, overlap_label="none",
    ))
    configs.append(SimConfig(
        name="S3: 4-DIMM",
        act_bits=8, num_dimms=4, overlap_factor=0.0, overlap_label="none",
    ))

    # --- Strategy 4: Overlapped scheduling ---
    configs.append(SimConfig(
        name="S4: Overlap (conservative)",
        act_bits=8, num_dimms=1, overlap_factor=0.5, overlap_label="conservative",
    ))
    configs.append(SimConfig(
        name="S4: Overlap (aggressive)",
        act_bits=8, num_dimms=1, overlap_factor=0.75, overlap_label="aggressive",
    ))

    # --- Combinations ---
    configs.append(SimConfig(
        name="S1+S3: Ternary + 2-DIMM",
        act_bits=2, num_dimms=2, overlap_factor=0.0, overlap_label="none",
    ))
    configs.append(SimConfig(
        name="S1+S3: Ternary + 4-DIMM",
        act_bits=2, num_dimms=4, overlap_factor=0.0, overlap_label="none",
    ))
    configs.append(SimConfig(
        name="S1+S4: Ternary + Overlap (cons)",
        act_bits=2, num_dimms=1, overlap_factor=0.5, overlap_label="conservative",
    ))
    configs.append(SimConfig(
        name="S1+S4: Ternary + Overlap (aggr)",
        act_bits=2, num_dimms=1, overlap_factor=0.75, overlap_label="aggressive",
    ))
    configs.append(SimConfig(
        name="S3+S4: 2-DIMM + Overlap (cons)",
        act_bits=8, num_dimms=2, overlap_factor=0.5, overlap_label="conservative",
    ))
    configs.append(SimConfig(
        name="S3+S4: 4-DIMM + Overlap (cons)",
        act_bits=8, num_dimms=4, overlap_factor=0.5, overlap_label="conservative",
    ))
    configs.append(SimConfig(
        name="S1+S3+S4: Ternary+4D+Overlap(cons)",
        act_bits=2, num_dimms=4, overlap_factor=0.5, overlap_label="conservative",
    ))
    configs.append(SimConfig(
        name="S1+S3+S4: Ternary+4D+Overlap(aggr)",
        act_bits=2, num_dimms=4, overlap_factor=0.75, overlap_label="aggressive",
    ))

    # --- With in-DRAM popcount (eliminates read phase) ---
    configs.append(SimConfig(
        name="S1+S3+S4+Pop: Full combo (cons)",
        act_bits=2, num_dimms=4, overlap_factor=0.5, overlap_label="conservative",
        in_dram_popcount=True,
    ))
    configs.append(SimConfig(
        name="S1+S3+S4+Pop: Full combo (aggr)",
        act_bits=2, num_dimms=4, overlap_factor=0.75, overlap_label="aggressive",
        in_dram_popcount=True,
    ))

    return configs


# ============================================================================
# Display Helpers
# ============================================================================

def print_separator(char: str = "=", width: int = 140):
    print(char * width)


def print_main_table(results: List[SimResult]):
    """Print the main results table."""
    print()
    print_separator()
    print("PIM-LLM THROUGHPUT SIMULATION RESULTS")
    print("Model: 2B BitNet b1.58 | DDR4-2400 | Autoregressive token generation")
    print_separator()

    hdr = (
        f"{'Configuration':<42s} "
        f"{'Bits':>4s} "
        f"{'DIMMs':>5s} "
        f"{'Overlap':<13s} "
        f"{'PopCnt':>6s} "
        f"{'Time(ms)':>10s} "
        f"{'tok/s':>10s} "
        f"{'Speedup':>8s} "
        f"{'vs CPU':>8s}"
    )
    print(hdr)
    print_separator("-")

    for r in results:
        c = r.config
        popcount_str = "Yes" if c.in_dram_popcount else "No"
        print(
            f"{c.name:<42s} "
            f"{c.act_bits:>4d} "
            f"{c.num_dimms:>5d} "
            f"{c.overlap_label:<13s} "
            f"{popcount_str:>6s} "
            f"{r.per_token_ms:>10.2f} "
            f"{r.throughput_toks:>10.2f} "
            f"{r.speedup_vs_baseline:>7.2f}x "
            f"{r.vs_cpu_factor:>7.2f}x"
        )

    print_separator()
    print(f"CPU reference: BitNet.cpp = 5.9 tok/s (single-threaded)")
    print()


def print_breakdown_table(results: List[SimResult]):
    """Print the time-component breakdown for each configuration."""
    print()
    print_separator()
    print("TIME BREAKDOWN PER TOKEN (milliseconds)")
    print_separator()

    hdr = (
        f"{'Configuration':<42s} "
        f"{'Write':>10s} "
        f"{'AND':>10s} "
        f"{'Read':>10s} "
        f"{'FPGA':>10s} "
        f"{'Transfer':>10s} "
        f"{'TOTAL':>10s} "
        f"{'AND %':>7s}"
    )
    print(hdr)
    print_separator("-")

    for r in results:
        b = r.breakdown
        total = b.total_ns / 1e6  # convert ns to ms
        w = b.write_ns / 1e6
        a = b.and_ns / 1e6
        rd = b.read_ns / 1e6
        fp = b.fpga_ns / 1e6
        tr = b.transfer_ns / 1e6
        and_pct = (b.and_ns / b.total_ns * 100) if b.total_ns > 0 else 0.0

        print(
            f"{r.config.name:<42s} "
            f"{w:>10.2f} "
            f"{a:>10.2f} "
            f"{rd:>10.2f} "
            f"{fp:>10.2f} "
            f"{tr:>10.2f} "
            f"{total:>10.2f} "
            f"{and_pct:>6.1f}%"
        )

    print_separator()
    print("Note: AND% shows the fraction of time spent on in-DRAM charge-sharing compute.")
    print("      High AND% means the DRAM compute is the bottleneck (good -- bus is not).")
    print()


def print_ddr4_parameters(timing: DDR4Timing):
    """Print DDR4 timing parameters for reference."""
    print()
    print_separator()
    print("DDR4-2400 TIMING PARAMETERS")
    print_separator()
    print(f"  Bus frequency:       {timing.bus_freq_mhz:.0f} MHz (DDR = {timing.data_rate_mts:.0f} MT/s)")
    print(f"  Bus width:           {timing.bus_width_bytes * 8} bits ({timing.bus_width_bytes} bytes)")
    print(f"  Burst length:        {timing.burst_length} ({timing.cache_line_bytes} bytes per burst)")
    print(f"  Peak bandwidth:      {timing.peak_bw_GBs:.1f} GB/s")
    print(f"  tRCD:                {timing.tRCD_ns:.2f} ns")
    print(f"  tRAS:                {timing.tRAS_ns:.2f} ns")
    print(f"  tRP:                 {timing.tRP_ns:.2f} ns")
    print(f"  tCL:                 {timing.tCL_ns:.2f} ns")
    print(f"  tBurst:              {timing.tBurst_ns:.2f} ns")
    print(f"  tREFI:               {timing.tREFI_ns:.1f} ns ({timing.tREFI_ns/1000:.1f} us)")
    print(f"  tRFC:                {timing.tRFC_ns:.1f} ns")
    print(f"  Refresh overhead:    {compute_refresh_overhead_fraction(timing)*100:.2f}%")
    print(f"  Row size:            {timing.row_size_bytes} bytes ({timing.row_size_bytes * 8} bits)")
    print()


def print_model_parameters(model: ModelParams):
    """Print model architecture summary."""
    print()
    print_separator()
    print("MODEL PARAMETERS (2B BitNet b1.58)")
    print_separator()
    print(f"  Transformer layers:  {model.num_layers}")
    print(f"  Hidden dimension:    {model.hidden_dim}")
    print(f"  FFN dimension:       {model.ffn_dim}")
    print(f"  Matvecs per layer:   {len(model.get_matvecs())}")
    print(f"  Weight rows/layer:   {model.total_weight_rows_per_layer}")
    print()
    print("  Per-layer matvecs:")
    for name, n_in, n_out in model.get_matvecs():
        print(f"    {name:<15s}  {n_in:>5d} x {n_out:<5d}  ({n_out} weight rows)")
    print()


# ============================================================================
# Bar Chart Generation
# ============================================================================

def generate_bar_chart(results: List[SimResult], output_path: str):
    """
    Generate a grouped bar chart showing throughput (tok/s) for all
    configurations, with a horizontal reference line for the CPU baseline.
    """
    names = [r.config.name for r in results]
    throughputs = [r.throughput_toks for r in results]
    cpu_ref = 5.9

    # Color coding by strategy type
    colors = []
    for r in results:
        c = r.config
        if c.in_dram_popcount:
            colors.append("#2ca02c")    # Green: full combo with popcount
        elif c.num_dimms > 1 and c.act_bits < 8 and c.overlap_factor > 0:
            colors.append("#17becf")    # Cyan: triple combo
        elif c.act_bits < 8 and c.num_dimms > 1:
            colors.append("#ff7f0e")    # Orange: ternary + multi-DIMM
        elif c.act_bits < 8 and c.overlap_factor > 0:
            colors.append("#9467bd")    # Purple: ternary + overlap
        elif c.num_dimms > 1 and c.overlap_factor > 0:
            colors.append("#8c564b")    # Brown: multi-DIMM + overlap
        elif c.act_bits < 8:
            colors.append("#d62728")    # Red: ternary only
        elif c.num_dimms > 1:
            colors.append("#1f77b4")    # Blue: multi-DIMM only
        elif c.overlap_factor > 0:
            colors.append("#e377c2")    # Pink: overlap only
        else:
            colors.append("#7f7f7f")    # Gray: baseline

    fig, ax = plt.subplots(figsize=(18, 9))

    x = np.arange(len(names))
    bars = ax.bar(x, throughputs, color=colors, edgecolor="black", linewidth=0.5)

    # Reference line for CPU
    ax.axhline(y=cpu_ref, color="red", linestyle="--", linewidth=1.5, label=f"CPU (BitNet.cpp): {cpu_ref} tok/s")

    # Labels on bars
    for bar, val in zip(bars, throughputs):
        ypos = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            ypos + max(throughputs) * 0.01,
            f"{val:.1f}",
            ha="center", va="bottom", fontsize=7, fontweight="bold",
        )

    ax.set_xlabel("Configuration", fontsize=12)
    ax.set_ylabel("Throughput (tokens/second)", fontsize=12)
    ax.set_title(
        "PIM-LLM Throughput: 2B BitNet b1.58 on DDR4-2400\n"
        "Cycle-Accurate Simulation of Engineering Strategies",
        fontsize=14, fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Bar chart saved to: {output_path}")
    plt.close(fig)


# ============================================================================
# Stacked Bar Chart (Time Breakdown)
# ============================================================================

def generate_breakdown_chart(results: List[SimResult], output_path: str):
    """
    Generate a stacked bar chart showing the time breakdown per token
    for each configuration.
    """
    names = [r.config.name for r in results]
    write_ms = [r.breakdown.write_ns / 1e6 for r in results]
    and_ms = [r.breakdown.and_ns / 1e6 for r in results]
    read_ms = [r.breakdown.read_ns / 1e6 for r in results]
    fpga_ms = [r.breakdown.fpga_ns / 1e6 for r in results]
    transfer_ms = [r.breakdown.transfer_ns / 1e6 for r in results]

    fig, ax = plt.subplots(figsize=(18, 9))
    x = np.arange(len(names))
    width = 0.6

    bottom = np.zeros(len(names))

    components = [
        (write_ms, "WRITE (activation to DRAM)", "#1f77b4"),
        (and_ms, "AND (charge-sharing compute)", "#ff7f0e"),
        (read_ms, "READ (result from DRAM)", "#2ca02c"),
        (fpga_ms, "FPGA (popcount + BN + act)", "#d62728"),
        (transfer_ms, "Inter-DIMM transfer", "#9467bd"),
    ]

    for vals, label, color in components:
        arr = np.array(vals)
        ax.bar(x, arr, width, bottom=bottom, label=label, color=color, edgecolor="black", linewidth=0.3)
        bottom += arr

    ax.set_xlabel("Configuration", fontsize=12)
    ax.set_ylabel("Per-Token Time (ms)", fontsize=12)
    ax.set_title(
        "PIM-LLM Time Breakdown Per Token\n"
        "2B BitNet b1.58 on DDR4-2400",
        fontsize=14, fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    # Save alongside the throughput chart
    breakdown_path = output_path.replace(".png", "_breakdown.png")
    plt.savefig(breakdown_path, dpi=150, bbox_inches="tight")
    print(f"Breakdown chart saved to: {breakdown_path}")
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 80)
    print("  PIM-LLM CYCLE-ACCURATE DDR4 THROUGHPUT SIMULATOR")
    print("  Modeling 2B BitNet b1.58 on DDR4-2400 PIM Architecture")
    print("=" * 80)

    # --- Initialize parameters ---
    timing = DDR4Timing()
    model = ModelParams()

    # --- Print parameter summaries ---
    print_ddr4_parameters(timing)
    print_model_parameters(model)

    # --- Sanity checks ---
    print_separator()
    print("SANITY CHECKS")
    print_separator()
    print(f"  Peak BW:                  {timing.peak_bw_GBs:.1f} GB/s (expected 19.2)")
    print(f"  Refresh overhead:         {compute_refresh_overhead_fraction(timing)*100:.2f}%")
    w = compute_write_time(2048, timing)
    print(f"  WRITE 2048-bit row:       {w:.2f} ns  ({math.ceil(2048/8/64)} burst(s))")
    a = compute_and_time(timing)
    print(f"  AND (double-ACT):         {a:.2f} ns")
    r = compute_read_time(2048, timing)
    print(f"  READ 2048-bit row:        {r:.2f} ns  ({math.ceil(2048/8/64)} burst(s))")
    print(f"  Total weight rows/layer:  {model.total_weight_rows_per_layer}")
    print(f"  Total weight rows (30L):  {model.total_weight_rows_per_layer * model.num_layers}")
    print()

    # --- Build configurations ---
    configs = build_configurations()

    # --- Run baseline first to get reference time ---
    baseline_result = simulate_full_model(configs[0], model, timing)
    baseline_ms = baseline_result.per_token_ms

    # --- Run all configurations ---
    results = []
    for cfg in configs:
        result = simulate_full_model(cfg, model, timing, baseline_ms=baseline_ms)
        results.append(result)

    # --- Print results ---
    print_main_table(results)
    print_breakdown_table(results)

    # --- Additional analysis ---
    print_separator()
    print("KEY INSIGHTS")
    print_separator()

    best = max(results, key=lambda r: r.throughput_toks)
    print(f"  Best configuration:       {best.config.name}")
    print(f"  Best throughput:          {best.throughput_toks:.2f} tok/s")
    print(f"  Speedup vs baseline:      {best.speedup_vs_baseline:.2f}x")
    print(f"  Speedup vs CPU:           {best.vs_cpu_factor:.2f}x")
    print()

    # Find first config that beats CPU
    cpu_beaters = [r for r in results if r.throughput_toks > 5.9]
    if cpu_beaters:
        first_beater = cpu_beaters[0]
        print(f"  First config > CPU:       {first_beater.config.name} ({first_beater.throughput_toks:.2f} tok/s)")
    else:
        print("  No configuration exceeds CPU throughput (5.9 tok/s).")
        closest = max(results, key=lambda r: r.throughput_toks)
        print(f"  Closest to CPU:           {closest.config.name} ({closest.throughput_toks:.2f} tok/s)")

    print()

    # Bottleneck analysis
    print_separator()
    print("BOTTLENECK ANALYSIS")
    print_separator()
    for r in results:
        b = r.breakdown
        total = b.total_ns
        if total == 0:
            continue
        components = {
            "WRITE": b.write_ns,
            "AND":   b.and_ns,
            "READ":  b.read_ns,
            "FPGA":  b.fpga_ns,
            "XFER":  b.transfer_ns,
        }
        bottleneck = max(components, key=components.get)
        pct = components[bottleneck] / total * 100
        print(f"  {r.config.name:<42s}  bottleneck: {bottleneck:<5s} ({pct:.1f}%)")

    print()

    # --- Generate charts ---
    chart_path = r"C:\Users\Udja\Documents\Deni\PIM\pim_throughput_sim.png"
    generate_bar_chart(results, chart_path)
    generate_breakdown_chart(results, chart_path)

    print()
    print("Simulation complete.")


if __name__ == "__main__":
    main()
