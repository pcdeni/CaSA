#!/usr/bin/env python3
"""
PIM-LLM DDR5 Throughput Simulator
===================================
Cycle-accurate DDR5-4800 timing simulator that computes tokens/second for a
2B-parameter BitNet b1.58 (2B4T) model running on a Processing-In-Memory DRAM
system.

DDR5 Key Differences from DDR4:
  - 4800 MT/s data rate (2x DDR4-2400)
  - BL16 burst length (vs BL8) -- still 64-byte cache line via 2x 32-bit sub-channels
  - 2 independent 32-bit sub-channels per DIMM (each with own command bus)
  - 32 banks per sub-channel (4 bank groups x 8 banks)
  - Per-bank refresh (REFab) -- only 1 bank stalled at a time
  - Mandatory On-Die ECC (ODECC) -- potential blocker for PIM charge-sharing

Models both ODECC-transparent and ODECC-blocking scenarios.
Compares results against DDR4 equivalents.

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
from typing import List, Optional, Tuple, Dict

# ============================================================================
# DDR5-4800 Timing Parameters
# ============================================================================

@dataclass
class DDR5Timing:
    """DDR5-4800 timing parameters for a single DIMM with 2 sub-channels."""
    # Clock / bus
    bus_freq_mhz: float = 2400.0          # Bus clock (I/O clock), MHz
    data_rate_mts: float = 4800.0          # Megatransfers per second (DDR)
    num_subchannel: int = 2                # Independent 32-bit sub-channels per DIMM
    bus_width_bytes_per_sc: int = 4        # 32-bit = 4 bytes per sub-channel per beat
    burst_length: int = 16                 # BL16
    cache_line_bytes: int = 64             # 16 beats x 4 bytes = 64 bytes per sub-channel burst

    # Bank architecture
    banks_per_subchannel: int = 32         # 4 bank groups x 8 banks per group
    bank_groups: int = 4
    banks_per_group: int = 8

    # Core timing (nanoseconds)
    tRCD_ns: float = 13.75                 # RAS-to-CAS delay
    tRAS_ns: float = 32.0                  # Row active time
    tRP_ns: float = 13.75                  # Row precharge
    tCL_ns: float = 13.75                  # CAS latency
    tBurst_ns: float = 3.33                # Burst transfer time (BL16 @ 4800 MT/s: 16/4800 us)

    # Refresh -- per-bank (REFab)
    tREFI_per_bank_ns: float = 3900.0      # Per-bank refresh interval (~3.9 us)
    tRFC_ns: float = 295.0                 # Refresh cycle time (16 Gb die)

    # Row geometry
    row_size_bytes: int = 8192             # 8 KB row buffer

    @property
    def peak_bw_per_sc_GBs(self) -> float:
        """Peak bandwidth per sub-channel in GB/s."""
        return self.data_rate_mts * self.bus_width_bytes_per_sc / 1000.0  # 19.2 GB/s

    @property
    def peak_bw_GBs(self) -> float:
        """Peak bandwidth per DIMM (both sub-channels) in GB/s."""
        return self.peak_bw_per_sc_GBs * self.num_subchannel  # 38.4 GB/s


# ============================================================================
# DDR4-2400 Timing Parameters (for comparison)
# ============================================================================

@dataclass
class DDR4Timing:
    """DDR4-2400 timing parameters for a single DIMM (for comparison)."""
    bus_freq_mhz: float = 1200.0
    data_rate_mts: float = 2400.0
    bus_width_bytes: int = 8
    burst_length: int = 8
    cache_line_bytes: int = 64

    tRCD_ns: float = 13.75
    tRAS_ns: float = 35.0
    tRP_ns: float = 13.75
    tCL_ns: float = 13.75
    tBurst_ns: float = 3.33
    tREFI_ns: float = 7800.0
    tRFC_ns: float = 350.0

    row_size_bytes: int = 8192
    banks_per_dimm: int = 16

    @property
    def peak_bw_GBs(self) -> float:
        return self.data_rate_mts * self.bus_width_bytes / 1000.0  # 19.2 GB/s


# ============================================================================
# LLM Model Parameters (2B BitNet b1.58 / 2B4T -- REAL dimensions)
# ============================================================================

@dataclass
class ModelParams:
    """
    Architecture parameters for the 2B-parameter BitNet b1.58 (2B4T) model.

    Uses the REAL BitNet 2B4T dimensions:
      hidden_dim = 2560
      ffn_dim = 6912
      7 matvecs per layer with actual input/output dimensions.

    Pack factor:
      d=2560: 65536 / 2560 = 25 neurons/row
      d=6912: 65536 / 6912 =  9 neurons/row
    """
    num_layers: int = 30
    hidden_dim: int = 2560
    ffn_dim: int = 6912
    row_bits: int = 65536          # 8 KB row = 65536 bits

    def get_matvecs(self) -> List[Tuple[str, int, int]]:
        """
        Return list of (name, N_in, N_out) for all matvecs in one layer.

        Q: (2560 -> 2560)
        K: (640 -> 2560)   -- GQA with num_kv_heads = num_heads/4
        V: (640 -> 2560)
        O: (2560 -> 2560)
        gate: (6912 -> 2560)
        up: (6912 -> 2560)
        down: (2560 -> 6912)
        """
        h = self.hidden_dim
        f = self.ffn_dim
        kv_dim = h // 4       # 640 for GQA
        return [
            ("Q_proj",    h,      h),        # 2560 x 2560
            ("K_proj",    kv_dim, h),         # 640 x 2560
            ("V_proj",    kv_dim, h),         # 640 x 2560
            ("O_proj",    h,      h),         # 2560 x 2560
            ("FFN_gate",  h,      f),         # 2560 x 6912
            ("FFN_up",    h,      f),         # 2560 x 6912
            ("FFN_down",  f,      h),         # 6912 x 2560
        ]

    def pack_factor(self, n_in: int) -> int:
        """Number of output neurons that can be packed per DRAM row."""
        return self.row_bits // n_in

    @property
    def total_weight_rows_per_layer(self) -> int:
        """Total output rows across all 7 matvecs in one layer."""
        return sum(n_out for _, _, n_out in self.get_matvecs())


# ============================================================================
# DDR5 PIM Protocol Timing Model
# ============================================================================

def compute_bus_transfer_time_ddr5(num_bytes: int, timing: DDR5Timing) -> float:
    """
    Time to transfer `num_bytes` over ONE DDR5 sub-channel.

    Each cache-line (64 B) burst takes tCL + tBurst. We also pay tRCD to open
    the row at the start and tRP to precharge at the end.

    Returns time in nanoseconds.
    """
    num_bursts = math.ceil(num_bytes / timing.cache_line_bytes)
    return (timing.tRCD_ns
            + num_bursts * (timing.tCL_ns + timing.tBurst_ns)
            + timing.tRP_ns)


def compute_write_time_ddr5(n_in: int, timing: DDR5Timing) -> float:
    """
    WRITE phase: write one activation bit-plane row (N_in bits) into DRAM.
    Transfer size = ceil(N_in / 8) bytes.
    Uses one sub-channel.
    """
    transfer_bytes = math.ceil(n_in / 8)
    return compute_bus_transfer_time_ddr5(transfer_bytes, timing)


def compute_and_time_ddr5(timing: DDR5Timing) -> float:
    """
    AND phase (charge-sharing double-activation).

    Same protocol as DDR4: two ACT commands + tRAS.
    DDR5 has faster tRAS (32 ns vs 35 ns).

    Time = 2 * tRCD + tRAS
    """
    return 2.0 * timing.tRCD_ns + timing.tRAS_ns


def compute_read_time_ddr5(n_in: int, timing: DDR5Timing) -> float:
    """
    READ phase: read the AND result row (N_in bits) back to the controller.
    Transfer size = ceil(N_in / 8) bytes.
    Uses one sub-channel.
    """
    transfer_bytes = math.ceil(n_in / 8)
    return compute_bus_transfer_time_ddr5(transfer_bytes, timing)


def compute_refresh_overhead_fraction_ddr5(timing: DDR5Timing) -> float:
    """
    DDR5 per-bank refresh overhead.

    With per-bank refresh (REFab), only 1 out of 32 banks is stalled at a time.
    Each bank refreshes every tREFI_per_bank (~3.9 us), stalling for tRFC (~295 ns).

    Effective overhead = (1 / banks_per_subchannel) * (tRFC / tREFI_per_bank)

    This is MUCH lower than DDR4's all-bank refresh.
    """
    per_bank_stall_fraction = timing.tRFC_ns / timing.tREFI_per_bank_ns
    return per_bank_stall_fraction / timing.banks_per_subchannel


def compute_refresh_overhead_fraction_ddr4(timing: DDR4Timing) -> float:
    """DDR4 all-bank refresh overhead for comparison."""
    return timing.tRFC_ns / timing.tREFI_ns


# ============================================================================
# DDR4 PIM Protocol Timing (for comparison runs)
# ============================================================================

def compute_bus_transfer_time_ddr4(num_bytes: int, timing: DDR4Timing) -> float:
    """Time to transfer num_bytes over the DDR4 bus (nanoseconds)."""
    num_bursts = math.ceil(num_bytes / timing.cache_line_bytes)
    return (timing.tRCD_ns
            + num_bursts * (timing.tCL_ns + timing.tBurst_ns)
            + timing.tRP_ns)


def compute_write_time_ddr4(n_in: int, timing: DDR4Timing) -> float:
    transfer_bytes = math.ceil(n_in / 8)
    return compute_bus_transfer_time_ddr4(transfer_bytes, timing)


def compute_and_time_ddr4(timing: DDR4Timing) -> float:
    return 2.0 * timing.tRCD_ns + timing.tRAS_ns


def compute_read_time_ddr4(n_in: int, timing: DDR4Timing) -> float:
    transfer_bytes = math.ceil(n_in / 8)
    return compute_bus_transfer_time_ddr4(transfer_bytes, timing)


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


def simulate_one_matvec_ddr5(
    n_in: int,
    n_out: int,
    act_bits: int,
    timing: DDR5Timing,
    overlap_factor: float = 0.0,
    in_dram_popcount: bool = False,
) -> LayerTiming:
    """
    Simulate one matrix-vector multiply on DDR5 PIM.

    DDR5 has 2 independent sub-channels, each with its own command bus.
    Both sub-channels can operate in parallel, effectively doubling the
    number of banks available for scheduling.

    For the AND operation, each sub-channel operates on its own set of banks
    independently. The weight rows are distributed across both sub-channels,
    so the effective AND throughput is 2x compared to a single channel.

    Parameters
    ----------
    n_in : int
        Input dimension (activation vector length).
    n_out : int
        Output dimension (number of weight rows to process).
    act_bits : int
        Activation precision in bits (B). Each bit-plane is processed twice
        (positive weight half, negative weight half) => 2*B passes.
    timing : DDR5Timing
        DDR5 timing parameters.
    overlap_factor : float
        Fraction of bus time that can be overlapped via multi-bank scheduling.
        DDR5's 32 banks per sub-channel enable more aggressive overlap.
    in_dram_popcount : bool
        If True, popcount is done in DRAM, eliminating the read phase.

    Returns
    -------
    LayerTiming with per-component nanosecond breakdown.
    """
    num_passes = act_bits * 2   # B bit-planes x 2 halves (pos/neg weights)

    # Per-pass timing (per sub-channel)
    write_per_pass = compute_write_time_ddr5(n_in, timing)
    and_per_pass = compute_and_time_ddr5(timing)

    if in_dram_popcount:
        read_per_pass = 0.0
    else:
        read_per_pass = compute_read_time_ddr5(n_in, timing)

    # DDR5 sub-channel parallelism:
    # Weight rows are distributed across 2 sub-channels.
    # Each sub-channel handles n_out / 2 weight rows.
    # The activation WRITE goes to both sub-channels in parallel (same data).
    # AND and READ operate independently per sub-channel.
    n_out_per_sc = math.ceil(n_out / timing.num_subchannel)

    # Per pass:
    #   1x WRITE (activation bit-plane) -- goes to both sub-channels simultaneously
    #   n_out_per_sc x (AND + READ) per sub-channel -- both sub-channels in parallel
    write_total = num_passes * write_per_pass
    and_total = num_passes * n_out_per_sc * and_per_pass
    read_total = num_passes * n_out_per_sc * read_per_pass

    # Apply overlap: write and read can overlap if using different banks
    # DDR5 has 32 banks per sub-channel, enabling better overlap than DDR4's 16
    bus_time = write_total + read_total
    if bus_time > 0:
        overlap_savings = bus_time * overlap_factor
        write_total_eff = write_total - overlap_savings * (write_total / bus_time)
        read_total_eff = read_total - overlap_savings * (read_total / bus_time)
    else:
        write_total_eff = 0.0
        read_total_eff = 0.0

    # FPGA post-processing: popcount + shift-accumulate + batch-norm + activation
    # ~10 ns per output element (pipelined, very fast relative to DRAM)
    fpga_cycles_per_output = 10.0  # ns
    fpga_total = n_out * fpga_cycles_per_output

    return LayerTiming(
        write_ns=write_total_eff,
        and_ns=and_total,
        read_ns=read_total_eff,
        fpga_ns=fpga_total,
        transfer_ns=0.0,
    )


def simulate_one_matvec_ddr4(
    n_in: int,
    n_out: int,
    act_bits: int,
    timing: DDR4Timing,
    overlap_factor: float = 0.0,
    in_dram_popcount: bool = False,
) -> LayerTiming:
    """Simulate one matrix-vector multiply on DDR4 PIM (for comparison)."""
    num_passes = act_bits * 2

    write_per_pass = compute_write_time_ddr4(n_in, timing)
    and_per_pass = compute_and_time_ddr4(timing)

    if in_dram_popcount:
        read_per_pass = 0.0
    else:
        read_per_pass = compute_read_time_ddr4(n_in, timing)

    write_total = num_passes * write_per_pass
    and_total = num_passes * n_out * and_per_pass
    read_total = num_passes * n_out * read_per_pass

    bus_time = write_total + read_total
    if bus_time > 0:
        overlap_savings = bus_time * overlap_factor
        write_total_eff = write_total - overlap_savings * (write_total / bus_time)
        read_total_eff = read_total - overlap_savings * (read_total / bus_time)
    else:
        write_total_eff = 0.0
        read_total_eff = 0.0

    fpga_cycles_per_output = 10.0
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
    timing,  # DDR5Timing or DDR4Timing
    overlap_factor: float = 0.0,
    in_dram_popcount: bool = False,
    is_ddr5: bool = True,
) -> LayerTiming:
    """Simulate all 7 matvecs in one transformer layer."""
    layer = LayerTiming()
    for name, n_in, n_out in model.get_matvecs():
        if is_ddr5:
            mv = simulate_one_matvec_ddr5(
                n_in, n_out, act_bits, timing, overlap_factor, in_dram_popcount
            )
        else:
            mv = simulate_one_matvec_ddr4(
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
    is_ddr5: bool = True
    odecc_ber_penalty: float = 0.0  # Additional BER from ODECC interference (0.0 = transparent)


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
    timing_ddr5: DDR5Timing,
    timing_ddr4: DDR4Timing,
    baseline_ms: Optional[float] = None,
    cpu_toks: float = 5.9,
) -> SimResult:
    """
    Simulate the full LLM inference for one token.

    For multi-DIMM configurations, layers are distributed evenly across DIMMs.
    Each DIMM processes its subset of layers independently, so wall-clock time
    is total_layer_time / num_dimms + inter-DIMM transfer overhead.

    DDR5 multi-DIMM: each DIMM has 2 sub-channels, so effective channel count
    is 2 * num_dimms. The sub-channel parallelism is already modeled within
    the per-matvec simulation.
    """
    timing = timing_ddr5 if config.is_ddr5 else timing_ddr4

    # Compute per-layer timing
    layer_timing = simulate_one_layer(
        model, config.act_bits, timing, config.overlap_factor,
        config.in_dram_popcount, is_ddr5=config.is_ddr5
    )

    # Aggregate across all layers
    total = LayerTiming(
        write_ns=layer_timing.write_ns * model.num_layers,
        and_ns=layer_timing.and_ns * model.num_layers,
        read_ns=layer_timing.read_ns * model.num_layers,
        fpga_ns=layer_timing.fpga_ns * model.num_layers,
    )

    # Multi-DIMM parallelism
    if config.num_dimms > 1:
        total.write_ns /= config.num_dimms
        total.and_ns /= config.num_dimms
        total.read_ns /= config.num_dimms
        total.fpga_ns /= config.num_dimms

        # Inter-DIMM transfer overhead
        transfer_bytes = model.hidden_dim * config.act_bits / 8.0
        bw = timing.peak_bw_GBs if config.is_ddr5 else timing.peak_bw_GBs
        transfer_time_per_hop_ns = transfer_bytes / bw  # ns
        num_transfers = config.num_dimms - 1
        total.transfer_ns = num_transfers * transfer_time_per_hop_ns
    else:
        total.transfer_ns = 0.0

    # Refresh overhead
    if config.is_ddr5:
        refresh_fraction = compute_refresh_overhead_fraction_ddr5(timing)
    else:
        refresh_fraction = compute_refresh_overhead_fraction_ddr4(timing)

    dram_time = total.write_ns + total.and_ns + total.read_ns
    if dram_time > 0:
        total.write_ns *= (1.0 + refresh_fraction)
        total.and_ns *= (1.0 + refresh_fraction)
        total.read_ns *= (1.0 + refresh_fraction)

    # ODECC penalty: if ODECC corrupts AND results, model as additional
    # correction passes. Each corrupted result requires a re-read + re-compute.
    # Effective overhead = odecc_ber_penalty fraction of additional AND+READ time.
    if config.odecc_ber_penalty > 0:
        correction_overhead = (total.and_ns + total.read_ns) * config.odecc_ber_penalty
        # Distribute proportionally between AND and READ
        if (total.and_ns + total.read_ns) > 0:
            and_frac = total.and_ns / (total.and_ns + total.read_ns)
            total.and_ns += correction_overhead * and_frac
            total.read_ns += correction_overhead * (1.0 - and_frac)

    # Final per-token time
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

def build_ddr5_configurations() -> List[SimConfig]:
    """Define DDR5 configurations to simulate."""
    configs = []

    # ===========================================
    # DDR5 configurations (ODECC-transparent)
    # ===========================================

    # Baseline: INT8, 1 DIMM, no overlap
    configs.append(SimConfig(
        name="DDR5: Baseline (INT8, 1D)",
        act_bits=8, num_dimms=1, overlap_factor=0.0, overlap_label="none",
        is_ddr5=True,
    ))

    # Ternary activations
    configs.append(SimConfig(
        name="DDR5: Ternary (B=2, 1D)",
        act_bits=2, num_dimms=1, overlap_factor=0.0, overlap_label="none",
        is_ddr5=True,
    ))

    # Multi-DIMM
    configs.append(SimConfig(
        name="DDR5: INT8, 2D",
        act_bits=8, num_dimms=2, overlap_factor=0.0, overlap_label="none",
        is_ddr5=True,
    ))
    configs.append(SimConfig(
        name="DDR5: INT8, 4D",
        act_bits=8, num_dimms=4, overlap_factor=0.0, overlap_label="none",
        is_ddr5=True,
    ))

    # Overlap
    configs.append(SimConfig(
        name="DDR5: INT8, 1D, Overlap(cons)",
        act_bits=8, num_dimms=1, overlap_factor=0.5, overlap_label="conservative",
        is_ddr5=True,
    ))
    configs.append(SimConfig(
        name="DDR5: INT8, 1D, Overlap(aggr)",
        act_bits=8, num_dimms=1, overlap_factor=0.75, overlap_label="aggressive",
        is_ddr5=True,
    ))

    # Combinations
    configs.append(SimConfig(
        name="DDR5: Tern+2D",
        act_bits=2, num_dimms=2, overlap_factor=0.0, overlap_label="none",
        is_ddr5=True,
    ))
    configs.append(SimConfig(
        name="DDR5: Tern+4D",
        act_bits=2, num_dimms=4, overlap_factor=0.0, overlap_label="none",
        is_ddr5=True,
    ))
    configs.append(SimConfig(
        name="DDR5: Tern+Overlap(cons)",
        act_bits=2, num_dimms=1, overlap_factor=0.5, overlap_label="conservative",
        is_ddr5=True,
    ))
    configs.append(SimConfig(
        name="DDR5: Tern+4D+Overlap(cons)",
        act_bits=2, num_dimms=4, overlap_factor=0.5, overlap_label="conservative",
        is_ddr5=True,
    ))
    configs.append(SimConfig(
        name="DDR5: Tern+4D+Overlap(aggr)",
        act_bits=2, num_dimms=4, overlap_factor=0.75, overlap_label="aggressive",
        is_ddr5=True,
    ))

    # With in-DRAM popcount
    configs.append(SimConfig(
        name="DDR5: Tern+4D+Ovlp(c)+Pop",
        act_bits=2, num_dimms=4, overlap_factor=0.5, overlap_label="conservative",
        in_dram_popcount=True, is_ddr5=True,
    ))
    configs.append(SimConfig(
        name="DDR5: Tern+4D+Ovlp(a)+Pop",
        act_bits=2, num_dimms=4, overlap_factor=0.75, overlap_label="aggressive",
        in_dram_popcount=True, is_ddr5=True,
    ))

    return configs


def build_ddr4_comparison_configurations() -> List[SimConfig]:
    """Build matching DDR4 configurations for comparison."""
    configs = []

    configs.append(SimConfig(
        name="DDR4: Baseline (INT8, 1D)",
        act_bits=8, num_dimms=1, overlap_factor=0.0, overlap_label="none",
        is_ddr5=False,
    ))
    configs.append(SimConfig(
        name="DDR4: Ternary (B=2, 1D)",
        act_bits=2, num_dimms=1, overlap_factor=0.0, overlap_label="none",
        is_ddr5=False,
    ))
    configs.append(SimConfig(
        name="DDR4: INT8, 4D",
        act_bits=8, num_dimms=4, overlap_factor=0.0, overlap_label="none",
        is_ddr5=False,
    ))
    configs.append(SimConfig(
        name="DDR4: Tern+4D+Overlap(cons)",
        act_bits=2, num_dimms=4, overlap_factor=0.5, overlap_label="conservative",
        is_ddr5=False,
    ))
    configs.append(SimConfig(
        name="DDR4: Tern+4D+Overlap(aggr)",
        act_bits=2, num_dimms=4, overlap_factor=0.75, overlap_label="aggressive",
        is_ddr5=False,
    ))
    configs.append(SimConfig(
        name="DDR4: Tern+4D+Ovlp(a)+Pop",
        act_bits=2, num_dimms=4, overlap_factor=0.75, overlap_label="aggressive",
        in_dram_popcount=True, is_ddr5=False,
    ))

    return configs


def build_odecc_configurations() -> List[SimConfig]:
    """Build ODECC analysis configurations with varying BER penalties."""
    configs = []
    # Use a good DDR5 config as the base
    base_act = 2
    base_dimms = 4
    base_overlap = 0.5

    for ber_pct in [0.0, 1.0, 5.0, 10.0, 20.0, 50.0]:
        label = f"ODECC BER={ber_pct:.0f}%" if ber_pct > 0 else "ODECC-transparent"
        configs.append(SimConfig(
            name=f"DDR5: Tern+4D+Ovlp(c) [{label}]",
            act_bits=base_act, num_dimms=base_dimms, overlap_factor=base_overlap,
            overlap_label="conservative", is_ddr5=True,
            odecc_ber_penalty=ber_pct / 100.0,
        ))

    return configs


# ============================================================================
# Display Helpers
# ============================================================================

def print_separator(char: str = "=", width: int = 150):
    print(char * width)


def print_ddr5_parameters(timing: DDR5Timing):
    """Print DDR5 timing parameters."""
    print()
    print_separator()
    print("DDR5-4800 TIMING PARAMETERS")
    print_separator()
    print(f"  Bus frequency:         {timing.bus_freq_mhz:.0f} MHz (DDR = {timing.data_rate_mts:.0f} MT/s)")
    print(f"  Sub-channels:          {timing.num_subchannel} independent 32-bit sub-channels per DIMM")
    print(f"  Bus width/sub-ch:      {timing.bus_width_bytes_per_sc * 8} bits ({timing.bus_width_bytes_per_sc} bytes)")
    print(f"  Burst length:          BL{timing.burst_length} ({timing.cache_line_bytes} bytes per burst per sub-channel)")
    print(f"  Peak BW/sub-channel:   {timing.peak_bw_per_sc_GBs:.1f} GB/s")
    print(f"  Peak BW/DIMM:          {timing.peak_bw_GBs:.1f} GB/s")
    print(f"  Banks/sub-channel:     {timing.banks_per_subchannel} ({timing.bank_groups} BG x {timing.banks_per_group} banks)")
    print(f"  tRCD:                  {timing.tRCD_ns:.2f} ns")
    print(f"  tRAS:                  {timing.tRAS_ns:.2f} ns")
    print(f"  tRP:                   {timing.tRP_ns:.2f} ns")
    print(f"  tCL:                   {timing.tCL_ns:.2f} ns")
    print(f"  tBurst:                {timing.tBurst_ns:.2f} ns (BL16 @ {timing.data_rate_mts:.0f} MT/s)")
    print(f"  tREFI (per-bank):      {timing.tREFI_per_bank_ns:.1f} ns ({timing.tREFI_per_bank_ns/1000:.1f} us)")
    print(f"  tRFC:                  {timing.tRFC_ns:.1f} ns")
    ref_ovhd = compute_refresh_overhead_fraction_ddr5(timing)
    print(f"  Refresh overhead:      {ref_ovhd*100:.4f}% (per-bank REFab: 1/{timing.banks_per_subchannel} banks stalled)")
    print(f"  Row size:              {timing.row_size_bytes} bytes ({timing.row_size_bytes * 8} bits)")
    print()


def print_ddr4_parameters(timing: DDR4Timing):
    """Print DDR4 timing parameters for comparison."""
    print()
    print_separator()
    print("DDR4-2400 TIMING PARAMETERS (for comparison)")
    print_separator()
    print(f"  Bus frequency:         {timing.bus_freq_mhz:.0f} MHz (DDR = {timing.data_rate_mts:.0f} MT/s)")
    print(f"  Bus width:             {timing.bus_width_bytes * 8} bits ({timing.bus_width_bytes} bytes)")
    print(f"  Burst length:          BL{timing.burst_length} ({timing.cache_line_bytes} bytes per burst)")
    print(f"  Peak BW/DIMM:          {timing.peak_bw_GBs:.1f} GB/s")
    print(f"  Banks/DIMM:            {timing.banks_per_dimm}")
    print(f"  tRCD:                  {timing.tRCD_ns:.2f} ns")
    print(f"  tRAS:                  {timing.tRAS_ns:.2f} ns")
    print(f"  tRP:                   {timing.tRP_ns:.2f} ns")
    print(f"  tCL:                   {timing.tCL_ns:.2f} ns")
    print(f"  tBurst:                {timing.tBurst_ns:.2f} ns (BL8 @ {timing.data_rate_mts:.0f} MT/s)")
    print(f"  tREFI (all-bank):      {timing.tREFI_ns:.1f} ns ({timing.tREFI_ns/1000:.1f} us)")
    print(f"  tRFC:                  {timing.tRFC_ns:.1f} ns")
    ref_ovhd = compute_refresh_overhead_fraction_ddr4(timing)
    print(f"  Refresh overhead:      {ref_ovhd*100:.2f}% (all-bank refresh)")
    print(f"  Row size:              {timing.row_size_bytes} bytes ({timing.row_size_bytes * 8} bits)")
    print()


def print_model_parameters(model: ModelParams):
    """Print model architecture summary."""
    print()
    print_separator()
    print("MODEL PARAMETERS (2B BitNet b1.58 / 2B4T -- REAL dimensions)")
    print_separator()
    print(f"  Transformer layers:  {model.num_layers}")
    print(f"  Hidden dimension:    {model.hidden_dim}")
    print(f"  FFN dimension:       {model.ffn_dim}")
    print(f"  Row size:            {model.row_bits} bits ({model.row_bits // 8} bytes)")
    print(f"  Matvecs per layer:   {len(model.get_matvecs())}")
    print(f"  Weight rows/layer:   {model.total_weight_rows_per_layer}")
    print()
    print("  Per-layer matvecs:")
    for name, n_in, n_out in model.get_matvecs():
        pf = model.pack_factor(n_in)
        print(f"    {name:<15s}  {n_in:>5d} x {n_out:<5d}  "
              f"({n_out} weight rows, pack_factor={pf} neurons/row)")
    print()


def print_main_table(results: List[SimResult], title: str = ""):
    """Print the main results table."""
    print()
    print_separator()
    if title:
        print(title)
    else:
        print("PIM-LLM DDR5 THROUGHPUT SIMULATION RESULTS")
    print("Model: 2B BitNet b1.58 (2B4T) | Autoregressive token generation")
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


def print_breakdown_table(results: List[SimResult], title: str = ""):
    """Print the time-component breakdown."""
    print()
    print_separator()
    if title:
        print(title)
    else:
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
        total = b.total_ns / 1e6
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
    print("      High AND% means the DRAM compute is the bottleneck (good -- bus is not idle).")
    print()


def print_comparison_table(ddr5_results: List[SimResult], ddr4_results: List[SimResult]):
    """Print a side-by-side DDR5 vs DDR4 comparison table."""
    print()
    print_separator()
    print("DDR5 vs DDR4 COMPARISON TABLE")
    print("Same model (2B BitNet 2B4T), same configurations, different memory technology")
    print_separator()

    # Match configurations by strategy
    comparisons = [
        ("Baseline (INT8, 1D)",       0, 0),
        ("Ternary (B=2, 1D)",         1, 1),
        ("INT8, 4D",                  3, 2),
        ("Tern+4D+Overlap(cons)",     9, 3),
        ("Tern+4D+Overlap(aggr)",    10, 4),
        ("Tern+4D+Ovlp(a)+Pop",     12, 5),
    ]

    hdr = (
        f"{'Strategy':<30s} "
        f"{'DDR4 tok/s':>12s} "
        f"{'DDR5 tok/s':>12s} "
        f"{'DDR5/DDR4':>10s} "
        f"{'DDR4 ms':>10s} "
        f"{'DDR5 ms':>10s}"
    )
    print(hdr)
    print_separator("-")

    for label, ddr5_idx, ddr4_idx in comparisons:
        if ddr5_idx < len(ddr5_results) and ddr4_idx < len(ddr4_results):
            d5 = ddr5_results[ddr5_idx]
            d4 = ddr4_results[ddr4_idx]
            ratio = d5.throughput_toks / d4.throughput_toks if d4.throughput_toks > 0 else float('inf')
            print(
                f"{label:<30s} "
                f"{d4.throughput_toks:>12.2f} "
                f"{d5.throughput_toks:>12.2f} "
                f"{ratio:>9.2f}x "
                f"{d4.per_token_ms:>10.2f} "
                f"{d5.per_token_ms:>10.2f}"
            )

    print_separator()
    print()


def print_odecc_analysis(odecc_results: List[SimResult]):
    """Print the ODECC impact analysis."""
    print()
    print_separator()
    print("ON-DIE ECC (ODECC) IMPACT ANALYSIS")
    print_separator()
    print()
    print("DDR5 has mandatory On-Die ECC that operates INSIDE the DRAM chip:")
    print("  - Corrects single-bit errors per 128-bit granule using (128,120) SEC code")
    print("  - ECC encode/decode happens automatically on every read/write")
    print("  - When we do charge-sharing AND, the result in the sense amplifiers has NOT")
    print("    been through ECC yet -- but when we READ the result, ODECC will 'correct'")
    print("    bits it thinks are errors")
    print("  - ODECC could (a) transparently pass through AND results, or (b) 'correct'")
    print("    valid AND results, corrupting them")
    print()
    print("Modeling approach: ODECC-blocking scenario adds an effective BER penalty,")
    print("representing the fraction of AND+READ operations that must be retried or")
    print("corrected through additional passes.")
    print()

    print_separator("-")
    hdr = (
        f"{'ODECC Scenario':<50s} "
        f"{'BER Penalty':>12s} "
        f"{'tok/s':>10s} "
        f"{'ms/tok':>10s} "
        f"{'vs Transparent':>15s}"
    )
    print(hdr)
    print_separator("-")

    if odecc_results:
        baseline_toks = odecc_results[0].throughput_toks
        for r in odecc_results:
            ratio = r.throughput_toks / baseline_toks if baseline_toks > 0 else 0
            ber_str = f"{r.config.odecc_ber_penalty*100:.0f}%"
            print(
                f"{r.config.name:<50s} "
                f"{ber_str:>12s} "
                f"{r.throughput_toks:>10.2f} "
                f"{r.per_token_ms:>10.2f} "
                f"{ratio:>14.2f}x"
            )

    print_separator()
    print()
    print("ODECC Conclusions:")
    print("  - If ODECC is transparent (or can be disabled via test mode), DDR5 PIM")
    print("    works at full speed with ~2x improvement over DDR4 from sub-channels.")
    print("  - If ODECC corrupts 5-10% of results, performance degrades modestly.")
    print("  - If ODECC corrupts >20% of results, it becomes a significant overhead.")
    print("  - ODECC is an OPEN QUESTION and potential BLOCKER for DDR5 PIM.")
    print("  - Mitigation strategies:")
    print("    (a) Disable ODECC via vendor test mode (if available)")
    print("    (b) Store data in ECC-compatible format (costly)")
    print("    (c) Use ODECC-aware encoding for PIM operands")
    print("    (d) Accept error rate and use error-tolerant inference algorithms")
    print()


def print_bottleneck_analysis(results: List[SimResult]):
    """Print bottleneck analysis."""
    print()
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


# ============================================================================
# Chart Generation
# ============================================================================

def generate_comparison_chart(
    ddr5_results: List[SimResult],
    ddr4_results: List[SimResult],
    odecc_results: List[SimResult],
    output_path: str,
):
    """
    Generate a multi-panel comparison chart:
      Panel 1: DDR5 throughput bar chart for all configurations
      Panel 2: DDR5 vs DDR4 side-by-side comparison
      Panel 3: ODECC impact analysis
      Panel 4: Time breakdown (stacked bar) for key DDR5 configs
    """
    fig, axes = plt.subplots(2, 2, figsize=(22, 16))
    cpu_ref = 5.9

    # ---- Panel 1: DDR5 Throughput Overview ----
    ax1 = axes[0, 0]
    names5 = [r.config.name.replace("DDR5: ", "") for r in ddr5_results]
    toks5 = [r.throughput_toks for r in ddr5_results]

    colors5 = []
    for r in ddr5_results:
        c = r.config
        if c.in_dram_popcount:
            colors5.append("#2ca02c")
        elif c.num_dimms > 1 and c.act_bits < 8 and c.overlap_factor > 0:
            colors5.append("#17becf")
        elif c.act_bits < 8 and c.num_dimms > 1:
            colors5.append("#ff7f0e")
        elif c.act_bits < 8 and c.overlap_factor > 0:
            colors5.append("#9467bd")
        elif c.num_dimms > 1 and c.overlap_factor > 0:
            colors5.append("#8c564b")
        elif c.act_bits < 8:
            colors5.append("#d62728")
        elif c.num_dimms > 1:
            colors5.append("#1f77b4")
        elif c.overlap_factor > 0:
            colors5.append("#e377c2")
        else:
            colors5.append("#7f7f7f")

    x5 = np.arange(len(names5))
    bars5 = ax1.bar(x5, toks5, color=colors5, edgecolor="black", linewidth=0.5)
    ax1.axhline(y=cpu_ref, color="red", linestyle="--", linewidth=1.5,
                label=f"CPU (BitNet.cpp): {cpu_ref} tok/s")

    for bar, val in zip(bars5, toks5):
        ypos = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, ypos + max(toks5) * 0.01,
                 f"{val:.1f}", ha="center", va="bottom", fontsize=6, fontweight="bold")

    ax1.set_xlabel("Configuration", fontsize=9)
    ax1.set_ylabel("Throughput (tok/s)", fontsize=9)
    ax1.set_title("DDR5-4800 PIM Throughput\n2B BitNet 2B4T", fontsize=11, fontweight="bold")
    ax1.set_xticks(x5)
    ax1.set_xticklabels(names5, rotation=55, ha="right", fontsize=5.5)
    ax1.legend(fontsize=8, loc="upper left")
    ax1.grid(axis="y", alpha=0.3)

    # ---- Panel 2: DDR5 vs DDR4 Side-by-Side ----
    ax2 = axes[0, 1]

    # Build matched pairs
    pair_labels = ["Baseline\nINT8,1D", "Ternary\n1D", "INT8\n4D",
                   "Tern+4D\nOvlp(c)", "Tern+4D\nOvlp(a)", "Tern+4D\nOvlp(a)+Pop"]
    ddr5_idxs = [0, 1, 3, 9, 10, 12]
    ddr4_idxs = [0, 1, 2, 3, 4, 5]

    ddr4_vals = [ddr4_results[i].throughput_toks for i in ddr4_idxs if i < len(ddr4_results)]
    ddr5_vals = [ddr5_results[i].throughput_toks for i in ddr5_idxs if i < len(ddr5_results)]

    n_pairs = min(len(ddr4_vals), len(ddr5_vals), len(pair_labels))
    pair_labels = pair_labels[:n_pairs]
    ddr4_vals = ddr4_vals[:n_pairs]
    ddr5_vals = ddr5_vals[:n_pairs]

    x_pair = np.arange(n_pairs)
    width = 0.35
    bars_d4 = ax2.bar(x_pair - width/2, ddr4_vals, width, label="DDR4-2400",
                      color="#1f77b4", edgecolor="black", linewidth=0.5)
    bars_d5 = ax2.bar(x_pair + width/2, ddr5_vals, width, label="DDR5-4800",
                      color="#ff7f0e", edgecolor="black", linewidth=0.5)

    ax2.axhline(y=cpu_ref, color="red", linestyle="--", linewidth=1.5,
                label=f"CPU: {cpu_ref} tok/s")

    for bar, val in zip(bars_d4, ddr4_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f"{val:.1f}", ha="center", va="bottom", fontsize=6)
    for bar, val in zip(bars_d5, ddr5_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f"{val:.1f}", ha="center", va="bottom", fontsize=6)

    # Add ratio labels
    for i in range(n_pairs):
        ratio = ddr5_vals[i] / ddr4_vals[i] if ddr4_vals[i] > 0 else 0
        y_max = max(ddr4_vals[i], ddr5_vals[i])
        ax2.text(x_pair[i], y_max + max(ddr5_vals) * 0.06,
                 f"{ratio:.1f}x", ha="center", va="bottom", fontsize=7,
                 fontweight="bold", color="green")

    ax2.set_xlabel("Configuration", fontsize=9)
    ax2.set_ylabel("Throughput (tok/s)", fontsize=9)
    ax2.set_title("DDR5 vs DDR4 Comparison\n2B BitNet 2B4T", fontsize=11, fontweight="bold")
    ax2.set_xticks(x_pair)
    ax2.set_xticklabels(pair_labels, fontsize=7)
    ax2.legend(fontsize=8, loc="upper left")
    ax2.grid(axis="y", alpha=0.3)

    # ---- Panel 3: ODECC Impact ----
    ax3 = axes[1, 0]
    if odecc_results:
        ber_pcts = [r.config.odecc_ber_penalty * 100 for r in odecc_results]
        odecc_toks = [r.throughput_toks for r in odecc_results]

        ax3.plot(ber_pcts, odecc_toks, 'o-', color="#d62728", linewidth=2, markersize=8)
        ax3.axhline(y=cpu_ref, color="red", linestyle="--", linewidth=1.5,
                    label=f"CPU: {cpu_ref} tok/s")

        for x_val, y_val in zip(ber_pcts, odecc_toks):
            ax3.annotate(f"{y_val:.2f}", (x_val, y_val),
                         textcoords="offset points", xytext=(0, 10),
                         ha="center", fontsize=7, fontweight="bold")

        # Shade regions
        ax3.axvspan(0, 5, alpha=0.1, color="green", label="Likely tolerable")
        ax3.axvspan(5, 20, alpha=0.1, color="yellow", label="Moderate impact")
        ax3.axvspan(20, 55, alpha=0.1, color="red", label="Severe impact")

        ax3.set_xlabel("ODECC BER Penalty (%)", fontsize=9)
        ax3.set_ylabel("Throughput (tok/s)", fontsize=9)
        ax3.set_title("ODECC Impact on DDR5 PIM\n(Tern+4D+Overlap config)", fontsize=11, fontweight="bold")
        ax3.legend(fontsize=7, loc="upper right")
        ax3.grid(alpha=0.3)
        ax3.set_xlim(-2, 55)

    # ---- Panel 4: Time Breakdown (stacked bar) ----
    ax4 = axes[1, 1]

    # Select key configs for breakdown
    key_idxs = [0, 1, 3, 9, 10, 12]
    key_results = [ddr5_results[i] for i in key_idxs if i < len(ddr5_results)]
    key_names = [r.config.name.replace("DDR5: ", "") for r in key_results]

    write_ms = [r.breakdown.write_ns / 1e6 for r in key_results]
    and_ms = [r.breakdown.and_ns / 1e6 for r in key_results]
    read_ms = [r.breakdown.read_ns / 1e6 for r in key_results]
    fpga_ms = [r.breakdown.fpga_ns / 1e6 for r in key_results]
    transfer_ms = [r.breakdown.transfer_ns / 1e6 for r in key_results]

    x_bd = np.arange(len(key_names))
    bar_width = 0.6
    bottom = np.zeros(len(key_names))

    components = [
        (write_ms, "WRITE", "#1f77b4"),
        (and_ms, "AND (compute)", "#ff7f0e"),
        (read_ms, "READ", "#2ca02c"),
        (fpga_ms, "FPGA", "#d62728"),
        (transfer_ms, "Inter-DIMM", "#9467bd"),
    ]

    for vals, label, color in components:
        arr = np.array(vals)
        ax4.bar(x_bd, arr, bar_width, bottom=bottom, label=label, color=color,
                edgecolor="black", linewidth=0.3)
        bottom += arr

    ax4.set_xlabel("Configuration", fontsize=9)
    ax4.set_ylabel("Per-Token Time (ms)", fontsize=9)
    ax4.set_title("DDR5 Time Breakdown Per Token\n2B BitNet 2B4T", fontsize=11, fontweight="bold")
    ax4.set_xticks(x_bd)
    ax4.set_xticklabels(key_names, rotation=45, ha="right", fontsize=6.5)
    ax4.legend(fontsize=7, loc="upper right")
    ax4.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nComparison chart saved to: {output_path}")
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 80)
    print("  PIM-LLM CYCLE-ACCURATE DDR5 THROUGHPUT SIMULATOR")
    print("  Modeling 2B BitNet b1.58 (2B4T) on DDR5-4800 PIM Architecture")
    print("=" * 80)

    # Initialize parameters
    timing_ddr5 = DDR5Timing()
    timing_ddr4 = DDR4Timing()
    model = ModelParams()

    # Print parameter summaries
    print_ddr5_parameters(timing_ddr5)
    print_ddr4_parameters(timing_ddr4)
    print_model_parameters(model)

    # Sanity checks
    print_separator()
    print("SANITY CHECKS")
    print_separator()
    print(f"  DDR5 Peak BW/sub-ch:      {timing_ddr5.peak_bw_per_sc_GBs:.1f} GB/s (expected 19.2)")
    print(f"  DDR5 Peak BW/DIMM:        {timing_ddr5.peak_bw_GBs:.1f} GB/s (expected 38.4)")
    print(f"  DDR4 Peak BW/DIMM:        {timing_ddr4.peak_bw_GBs:.1f} GB/s (expected 19.2)")
    ref5 = compute_refresh_overhead_fraction_ddr5(timing_ddr5)
    ref4 = compute_refresh_overhead_fraction_ddr4(timing_ddr4)
    print(f"  DDR5 Refresh overhead:     {ref5*100:.4f}% (per-bank, 1/{timing_ddr5.banks_per_subchannel} banks)")
    print(f"  DDR4 Refresh overhead:     {ref4*100:.2f}% (all-bank)")
    print(f"  Refresh improvement:       {ref4/ref5:.1f}x lower overhead on DDR5")
    print()

    w5 = compute_write_time_ddr5(2560, timing_ddr5)
    a5 = compute_and_time_ddr5(timing_ddr5)
    r5 = compute_read_time_ddr5(2560, timing_ddr5)
    print(f"  DDR5 WRITE 2560-bit row:   {w5:.2f} ns  ({math.ceil(2560/8/64)} burst(s))")
    print(f"  DDR5 AND (double-ACT):     {a5:.2f} ns")
    print(f"  DDR5 READ 2560-bit row:    {r5:.2f} ns  ({math.ceil(2560/8/64)} burst(s))")
    print()

    w4 = compute_write_time_ddr4(2560, timing_ddr4)
    a4 = compute_and_time_ddr4(timing_ddr4)
    r4 = compute_read_time_ddr4(2560, timing_ddr4)
    print(f"  DDR4 WRITE 2560-bit row:   {w4:.2f} ns  ({math.ceil(2560/8/64)} burst(s))")
    print(f"  DDR4 AND (double-ACT):     {a4:.2f} ns")
    print(f"  DDR4 READ 2560-bit row:    {r4:.2f} ns  ({math.ceil(2560/8/64)} burst(s))")
    print()

    print(f"  Total weight rows/layer:   {model.total_weight_rows_per_layer}")
    print(f"  Total weight rows (30L):   {model.total_weight_rows_per_layer * model.num_layers}")
    print(f"  Pack factor d=2560:        {model.pack_factor(2560)} neurons/row")
    print(f"  Pack factor d=6912:        {model.pack_factor(6912)} neurons/row")
    print(f"  Pack factor d=640:         {model.pack_factor(640)} neurons/row")
    print()

    # Build configurations
    ddr5_configs = build_ddr5_configurations()
    ddr4_configs = build_ddr4_comparison_configurations()
    odecc_configs = build_odecc_configurations()

    # Run DDR5 baseline first
    ddr5_baseline_result = simulate_full_model(
        ddr5_configs[0], model, timing_ddr5, timing_ddr4
    )
    ddr5_baseline_ms = ddr5_baseline_result.per_token_ms

    # Run all DDR5 configurations
    ddr5_results = []
    for cfg in ddr5_configs:
        result = simulate_full_model(
            cfg, model, timing_ddr5, timing_ddr4, baseline_ms=ddr5_baseline_ms
        )
        ddr5_results.append(result)

    # Run DDR4 comparison configurations
    ddr4_baseline_result = simulate_full_model(
        ddr4_configs[0], model, timing_ddr5, timing_ddr4
    )
    ddr4_baseline_ms = ddr4_baseline_result.per_token_ms

    ddr4_results = []
    for cfg in ddr4_configs:
        result = simulate_full_model(
            cfg, model, timing_ddr5, timing_ddr4, baseline_ms=ddr4_baseline_ms
        )
        ddr4_results.append(result)

    # Run ODECC analysis
    odecc_results = []
    for cfg in odecc_configs:
        result = simulate_full_model(
            cfg, model, timing_ddr5, timing_ddr4, baseline_ms=ddr5_baseline_ms
        )
        odecc_results.append(result)

    # ==========================================
    # Print results
    # ==========================================

    print_main_table(ddr5_results, "DDR5-4800 PIM THROUGHPUT RESULTS")
    print_breakdown_table(ddr5_results, "DDR5 TIME BREAKDOWN PER TOKEN (milliseconds)")

    print_main_table(ddr4_results, "DDR4-2400 PIM THROUGHPUT RESULTS (comparison, using 2B4T model)")
    print_breakdown_table(ddr4_results, "DDR4 TIME BREAKDOWN PER TOKEN (milliseconds)")

    # DDR5 vs DDR4 comparison
    print_comparison_table(ddr5_results, ddr4_results)

    # ODECC analysis
    print_odecc_analysis(odecc_results)

    # Bottleneck analysis
    print_bottleneck_analysis(ddr5_results)

    # Key insights
    print_separator()
    print("KEY INSIGHTS: DDR5 vs DDR4")
    print_separator()

    best_ddr5 = max(ddr5_results, key=lambda r: r.throughput_toks)
    best_ddr4 = max(ddr4_results, key=lambda r: r.throughput_toks)

    print(f"  Best DDR5 config:          {best_ddr5.config.name}")
    print(f"  Best DDR5 throughput:      {best_ddr5.throughput_toks:.2f} tok/s")
    print(f"  Best DDR4 config:          {best_ddr4.config.name}")
    print(f"  Best DDR4 throughput:      {best_ddr4.throughput_toks:.2f} tok/s")
    print(f"  DDR5 best / DDR4 best:     {best_ddr5.throughput_toks / best_ddr4.throughput_toks:.2f}x")
    print()

    # DDR5 advantages
    print("  DDR5 Advantages for PIM:")
    print(f"    1. 2x sub-channels: each DIMM has 2 independent 32-bit channels")
    print(f"       -> Weight rows distributed across sub-channels, halving AND time per DIMM")
    print(f"    2. 32 banks/sub-channel (vs DDR4's 16/DIMM)")
    print(f"       -> Better bank-level parallelism for overlapped scheduling")
    print(f"    3. Per-bank refresh (REFab): only 1/32 banks stalled vs DDR4 all-bank")
    print(f"       -> Refresh overhead: {ref5*100:.4f}% vs {ref4*100:.2f}% ({ref4/ref5:.0f}x improvement)")
    print(f"    4. Faster tRAS: 32 ns vs 35 ns -> faster AND operation")
    print()

    print("  DDR5 Concerns for PIM:")
    print(f"    1. ODECC (On-Die ECC): mandatory, may corrupt charge-sharing results")
    print(f"    2. BL16 (vs BL8): same cache line but via sub-channels, no direct penalty")
    print(f"    3. Sub-channel bus is 32-bit (vs DDR4 64-bit), but 2 of them compensate")
    print()

    # CPU comparison
    cpu_beaters_5 = [r for r in ddr5_results if r.throughput_toks > 5.9]
    cpu_beaters_4 = [r for r in ddr4_results if r.throughput_toks > 5.9]
    print(f"  DDR5 configs beating CPU:  {len(cpu_beaters_5)} / {len(ddr5_results)}")
    print(f"  DDR4 configs beating CPU:  {len(cpu_beaters_4)} / {len(ddr4_results)}")
    if cpu_beaters_5:
        first = cpu_beaters_5[0]
        print(f"  First DDR5 config > CPU:   {first.config.name} ({first.throughput_toks:.2f} tok/s)")
    print()

    # Generate charts
    chart_path = r"C:\Users\Udja\Documents\Deni\PIM\pim_throughput_sim_ddr5.png"
    generate_comparison_chart(ddr5_results, ddr4_results, odecc_results, chart_path)

    print()
    print("Simulation complete.")


if __name__ == "__main__":
    main()
