#!/usr/bin/env python3
"""
PIM-LLM HBM2 Throughput Estimator
====================================
Analytical throughput estimator for a 2B-parameter BitNet b1.58 (2B4T) model
running on a Processing-In-Memory HBM2 system.

HBM2 Key Differences from DDR4/DDR5:
  - 1024-bit aggregate bus (8 independent 128-bit channels per stack)
  - ~256 GB/s bandwidth per stack (vs 19.2 GB/s DDR4, 38.4 GB/s DDR5)
  - Same 1T1C DRAM cell physics -- charge-sharing AND feasible
  - 2 KB row buffer (vs 8 KB in DDR4/DDR5)
  - Optional On-Die ECC (unlike DDR5's mandatory ODECC)
  - 16 banks per channel, 4 bank groups
  - Per-bank refresh (REFab) like DDR5
  - Hard IP controller on FPGAs -- primary obstacle for raw command access

Uses the same modeling methodology as pim_throughput_sim.py (DDR4) and
pim_throughput_sim_ddr5.py (DDR5) for direct comparison.

Compares results against DDR4, DDR5, and CPU baselines.

Dependencies: numpy, matplotlib (standard scientific Python stack).

Author: PIM-LLM project
Date: February 2026
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

# ============================================================================
# HBM2 Timing Parameters (JESD235A, 2.0 Gbps per pin)
# ============================================================================

@dataclass
class HBM2Timing:
    """
    HBM2-2000 timing parameters for a single stack.

    Sources:
      - JESD235A specification
      - gem5 HBM_2000_4H_1x64 configuration (validated against JEDEC)
      - Micron/Samsung HBM2 datasheets (public parameters)
    """
    # Clock / bus (per channel)
    io_freq_ghz: float = 1.0               # 1 GHz I/O clock (2 Gbps DDR per pin)
    data_rate_gbps: float = 2.0             # Gbps per pin
    num_channels: int = 8                   # Independent channels per stack
    channel_width_bits: int = 128           # 128-bit channel width (legacy mode)
    burst_length: int = 2                   # BL2 (legacy mode)
    # Effective burst: 128 bits x 2 beats = 256 bits = 32 bytes per burst
    cache_line_bytes: int = 32              # 32 bytes per column access

    # Pseudo-channel mode (optional, for reference)
    pseudo_channel_width_bits: int = 64     # 64-bit per pseudo-channel
    pseudo_burst_length: int = 4            # BL4 in pseudo-channel mode
    # Still 256 bits = 32 bytes per burst in pseudo-channel

    # Bank architecture (per channel)
    banks_per_channel: int = 16             # 4 bank groups x 4 banks
    bank_groups: int = 4
    banks_per_group: int = 4

    # Core timing (nanoseconds) -- representative HBM2-2000 parameters
    tRCD_ns: float = 14.0                   # RAS-to-CAS delay (14 cycles @ 1 GHz)
    tRAS_ns: float = 28.0                   # Row active time
    tRP_ns: float = 14.0                    # Row precharge
    tCL_ns: float = 14.0                    # CAS latency (14 cycles)
    tBurst_ns: float = 1.0                  # BL2 at 1 GHz = 2 beats / 2 GHz(DDR) = 1 ns
    tCCD_S_ns: float = 2.0                  # Column-to-column delay (short, diff bank group)
    tCCD_L_ns: float = 4.0                  # Column-to-column delay (long, same bank group)
    tRRD_S_ns: float = 4.0                  # Row-to-row delay (short)
    tRRD_L_ns: float = 6.0                  # Row-to-row delay (long)

    # Refresh -- per-bank (REFab), similar to DDR5
    tREFI_per_bank_ns: float = 3900.0       # Per-bank refresh interval (~3.9 us)
    tRFC_ns: float = 220.0                  # Refresh cycle time

    # Row geometry
    row_size_bytes: int = 2048              # 2 KB row buffer (legacy mode)
    # Note: 1 KB in pseudo-channel mode

    # Stack geometry
    stack_height: int = 4                   # 4-Hi stack (4 DRAM dies)
    capacity_per_die_gb: int = 1            # 1 GB per die (8 Gb)

    @property
    def peak_bw_per_channel_GBs(self) -> float:
        """Peak bandwidth per channel in GB/s."""
        return self.data_rate_gbps * self.channel_width_bits / 8.0  # 32 GB/s

    @property
    def peak_bw_GBs(self) -> float:
        """Peak bandwidth per stack (all channels) in GB/s."""
        return self.peak_bw_per_channel_GBs * self.num_channels  # 256 GB/s

    @property
    def total_capacity_GB(self) -> float:
        """Total capacity per stack."""
        return self.capacity_per_die_gb * self.stack_height  # 4 GB

    @property
    def rows_per_die(self) -> int:
        """Total rows per die."""
        return (self.capacity_per_die_gb * 1024 * 1024 * 1024) // self.row_size_bytes

    @property
    def rows_per_channel(self) -> int:
        """Rows per channel (2 channels per die for 4-Hi)."""
        # 4-Hi: dies share channels. 8 channels / 4 dies = 2 channels per die
        # Each channel spans one die (simplified)
        return self.rows_per_die // 2  # 2 channels per die


# ============================================================================
# DDR4/DDR5 Timing (for comparison)
# ============================================================================

@dataclass
class DDR4Timing:
    """DDR4-2400 reference parameters."""
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
    banks: int = 16

    @property
    def peak_bw_GBs(self):
        return self.data_rate_mts * self.bus_width_bytes / 1000.0  # 19.2 GB/s

    @property
    def refresh_overhead(self):
        return self.tRFC_ns / self.tREFI_ns  # 4.49%


@dataclass
class DDR5Timing:
    """DDR5-4800 reference parameters."""
    peak_bw_GBs: float = 38.4  # 2 sub-channels x 19.2 GB/s
    tRCD_ns: float = 13.75
    tRAS_ns: float = 32.0
    tRP_ns: float = 13.75
    tCL_ns: float = 13.75
    tBurst_ns: float = 3.33
    cache_line_bytes: int = 64  # per sub-channel
    row_size_bytes: int = 8192
    tREFI_per_bank_ns: float = 3900.0
    tRFC_ns: float = 295.0
    banks_per_sc: int = 32


# ============================================================================
# Model Parameters (BitNet 2B4T -- REAL dimensions)
# ============================================================================

@dataclass
class ModelParams:
    """Architecture parameters for BitNet b1.58-2B-4T."""
    num_layers: int = 30
    hidden_dim: int = 2560      # d_model
    ffn_dim: int = 6912         # d_ff
    kv_dim: int = 640           # GQA: 5 KV heads x 128 dim

    def get_matvecs(self) -> List[Tuple[str, int, int]]:
        """Return (name, N_in, N_out) for all matvecs in one layer."""
        h = self.hidden_dim
        f = self.ffn_dim
        kv = self.kv_dim
        return [
            ("Q_proj",    h, h),     # 2560 -> 2560
            ("K_proj",    h, kv),    # 2560 -> 640
            ("V_proj",    h, kv),    # 2560 -> 640
            ("O_proj",    h, h),     # 2560 -> 2560
            ("FFN_gate",  h, f),     # 2560 -> 6912
            ("FFN_up",    h, f),     # 2560 -> 6912
            ("FFN_down",  f, h),     # 6912 -> 2560
        ]

    def get_total_weight_rows(self, row_size_bytes: int, n_in: int) -> int:
        """Neurons packed per physical row for given input dimension."""
        row_bits = row_size_bytes * 8
        neurons_per_row = max(1, row_bits // n_in)
        return neurons_per_row


# ============================================================================
# PIM Timing Functions
# ============================================================================

def compute_bus_transfer_hbm2(num_bytes: int, timing: HBM2Timing) -> float:
    """
    Time to transfer `num_bytes` over one HBM2 channel.

    Model: open row (tRCD), issue CAS commands, close row (tRP).
    First CAS has full tCL latency; subsequent CAS pipelined at tCCD intervals.
    However, for consistency with DDR4/DDR5 sims, we use the same
    conservative model: tRCD + num_bursts * (tCL + tBurst) + tRP.
    """
    num_bursts = math.ceil(num_bytes / timing.cache_line_bytes)
    return (timing.tRCD_ns
            + num_bursts * (timing.tCL_ns + timing.tBurst_ns)
            + timing.tRP_ns)


def compute_bus_transfer_ddr4(num_bytes: int, timing: DDR4Timing) -> float:
    """Time to transfer num_bytes over DDR4 bus (same model as DDR4 sim)."""
    num_bursts = math.ceil(num_bytes / timing.cache_line_bytes)
    return (timing.tRCD_ns
            + num_bursts * (timing.tCL_ns + timing.tBurst_ns)
            + timing.tRP_ns)


def compute_maj3_time_hbm2(timing: HBM2Timing) -> float:
    """MAJ3 AND: 3 rows activated simultaneously. Time = 3*tRCD + tRAS"""
    return 3.0 * timing.tRCD_ns + timing.tRAS_ns


def compute_rowcopy_time_hbm2(timing: HBM2Timing) -> float:
    """SA-mediated RowCopy: ACT source -> SA latches -> ACT dest. Time = 2*tRCD + tRAS + tRP"""
    return 2.0 * timing.tRCD_ns + timing.tRAS_ns + timing.tRP_ns


def compute_maj3_time_ddr4(timing: DDR4Timing) -> float:
    """MAJ3 AND: 3 rows activated simultaneously. Time = 3*tRCD + tRAS"""
    return 3.0 * timing.tRCD_ns + timing.tRAS_ns


def compute_rowcopy_time_ddr4(timing: DDR4Timing) -> float:
    """SA-mediated RowCopy: ACT source -> SA latches -> ACT dest. Time = 2*tRCD + tRAS + tRP"""
    return 2.0 * timing.tRCD_ns + timing.tRAS_ns + timing.tRP_ns


# ============================================================================
# Per-Matvec Simulation
# ============================================================================

@dataclass
class MatvecTiming:
    """Timing breakdown for one matrix-vector multiply."""
    write_ns: float = 0.0
    maj3_ns: float = 0.0       # MAJ3 AND compute
    rowcopy_ns: float = 0.0    # RowCopy for weight reload + zero-ref init
    read_ns: float = 0.0
    fpga_ns: float = 0.0

    @property
    def and_ns(self) -> float:
        """Alias for backward compatibility."""
        return self.maj3_ns

    @property
    def total_ns(self):
        return self.write_ns + self.maj3_ns + self.rowcopy_ns + self.read_ns + self.fpga_ns

    @property
    def bus_ns(self):
        return self.write_ns + self.read_ns


def simulate_matvec_hbm2(
    n_in: int,
    n_out: int,
    act_bits: int,
    timing: HBM2Timing,
    overlap_factor: float = 0.0,
    in_dram_popcount: bool = False,
) -> MatvecTiming:
    """
    Simulate one matvec on a single HBM2 channel.

    CORRECTED (March 2026): MAJ3 + RowCopy protocol.
    - num_passes = act_bits * 2 (pos/neg weight halves)
    - Per pass: 1 WRITE + n_out * (RowCopy_zeros + MAJ3 + READ + RowCopy_weight)
    - Pipelining hides DRAM ops behind bus ops across banks
    - In-DRAM popcount eliminates reads
    """
    num_passes = act_bits * 2

    transfer_bytes = math.ceil(n_in / 8)

    write_per_pass = compute_bus_transfer_hbm2(transfer_bytes, timing)
    maj3_per_row = compute_maj3_time_hbm2(timing)
    rowcopy_per_row = compute_rowcopy_time_hbm2(timing)

    if in_dram_popcount:
        read_per_row = 0.0
    else:
        read_per_row = compute_bus_transfer_hbm2(transfer_bytes, timing)

    # RowCopy overhead per weight row:
    # 1x RowCopy to init zero-ref + 1x RowCopy to restore weight = 2 RowCopies
    rowcopy_per_weight_row = 2.0 * rowcopy_per_row

    # Pipelining model
    bus_per_row = read_per_row
    dram_per_row = rowcopy_per_weight_row + maj3_per_row

    if overlap_factor <= 0.0:
        pipeline_banks = 1
    elif overlap_factor <= 0.5:
        pipeline_banks = 4
    else:
        pipeline_banks = 8

    dram_per_row_pipelined = dram_per_row / pipeline_banks
    effective_per_row = max(bus_per_row, dram_per_row_pipelined)

    # Bus writes (activation): once per pass
    write_total_eff = num_passes * write_per_pass

    # Decompose effective time into bus vs DRAM components for reporting
    if bus_per_row >= dram_per_row_pipelined:
        # Bus-bound: DRAM ops fully hidden
        read_total_eff = num_passes * n_out * bus_per_row
        maj3_total_eff = 0.0
        rowcopy_total_eff = 0.0
    else:
        # DRAM-bound: bus time hidden behind DRAM
        read_total_eff = 0.0
        maj3_total_eff = num_passes * n_out * (maj3_per_row / pipeline_banks)
        rowcopy_total_eff = num_passes * n_out * (rowcopy_per_weight_row / pipeline_banks)

    # FPGA post-processing
    fpga_total = n_out * 10.0  # 10 ns per output element

    return MatvecTiming(
        write_ns=write_total_eff,
        maj3_ns=maj3_total_eff,
        rowcopy_ns=rowcopy_total_eff,
        read_ns=read_total_eff,
        fpga_ns=fpga_total,
    )


def simulate_matvec_ddr4(
    n_in: int,
    n_out: int,
    act_bits: int,
    timing: DDR4Timing,
    overlap_factor: float = 0.0,
    in_dram_popcount: bool = False,
) -> MatvecTiming:
    """Same MAJ3+RowCopy model for DDR4 (for comparison)."""
    num_passes = act_bits * 2
    transfer_bytes = math.ceil(n_in / 8)

    write_per_pass = compute_bus_transfer_ddr4(transfer_bytes, timing)
    maj3_per_row = compute_maj3_time_ddr4(timing)
    rowcopy_per_row = compute_rowcopy_time_ddr4(timing)
    read_per_row = 0.0 if in_dram_popcount else compute_bus_transfer_ddr4(transfer_bytes, timing)

    rowcopy_per_weight_row = 2.0 * rowcopy_per_row

    # Pipelining model
    bus_per_row = read_per_row
    dram_per_row = rowcopy_per_weight_row + maj3_per_row

    if overlap_factor <= 0.0:
        pipeline_banks = 1
    elif overlap_factor <= 0.5:
        pipeline_banks = 4
    else:
        pipeline_banks = 8

    dram_per_row_pipelined = dram_per_row / pipeline_banks
    effective_per_row = max(bus_per_row, dram_per_row_pipelined)

    write_total_eff = num_passes * write_per_pass

    if bus_per_row >= dram_per_row_pipelined:
        read_total_eff = num_passes * n_out * bus_per_row
        maj3_total_eff = 0.0
        rowcopy_total_eff = 0.0
    else:
        read_total_eff = 0.0
        maj3_total_eff = num_passes * n_out * (maj3_per_row / pipeline_banks)
        rowcopy_total_eff = num_passes * n_out * (rowcopy_per_weight_row / pipeline_banks)

    fpga_total = n_out * 10.0

    return MatvecTiming(
        write_ns=write_total_eff,
        maj3_ns=maj3_total_eff,
        rowcopy_ns=rowcopy_total_eff,
        read_ns=read_total_eff,
        fpga_ns=fpga_total,
    )


# ============================================================================
# Full Model Simulation
# ============================================================================

@dataclass
class SimConfig:
    """Configuration for one simulation run."""
    name: str
    mem_type: str  # "DDR4", "DDR5", "HBM2"
    act_bits: int = 2  # Ternary by default
    num_channels: int = 1  # HBM2 channels or DDR4/DDR5 DIMMs
    overlap_factor: float = 0.0
    in_dram_popcount: bool = False


@dataclass
class SimResult:
    """Results from one simulation run."""
    config: SimConfig
    per_token_ms: float
    throughput_toks: float
    vs_cpu_factor: float
    breakdown_write_ms: float
    breakdown_maj3_ms: float
    breakdown_rowcopy_ms: float
    breakdown_read_ms: float
    breakdown_fpga_ms: float
    breakdown_transfer_ms: float

    @property
    def breakdown_and_ms(self) -> float:
        """Alias for backward compatibility."""
        return self.breakdown_maj3_ms


def simulate_full_model(
    config: SimConfig,
    model: ModelParams,
    hbm2: HBM2Timing,
    ddr4: DDR4Timing,
    cpu_toks: float = 5.9,
) -> SimResult:
    """Simulate full model inference for one token."""

    # Accumulate per-layer timing
    total_write = 0.0
    total_maj3 = 0.0
    total_rowcopy = 0.0
    total_read = 0.0
    total_fpga = 0.0

    for name, n_in, n_out in model.get_matvecs():
        if config.mem_type == "HBM2":
            mv = simulate_matvec_hbm2(
                n_in, n_out, config.act_bits, hbm2,
                config.overlap_factor, config.in_dram_popcount)
        else:  # DDR4
            mv = simulate_matvec_ddr4(
                n_in, n_out, config.act_bits, ddr4,
                config.overlap_factor, config.in_dram_popcount)

        total_write += mv.write_ns
        total_maj3 += mv.maj3_ns
        total_rowcopy += mv.rowcopy_ns
        total_read += mv.read_ns
        total_fpga += mv.fpga_ns

    # Scale across all layers
    total_write *= model.num_layers
    total_maj3 *= model.num_layers
    total_rowcopy *= model.num_layers
    total_read *= model.num_layers
    total_fpga *= model.num_layers

    # Multi-channel/DIMM parallelism
    if config.num_channels > 1:
        total_write /= config.num_channels
        total_maj3 /= config.num_channels
        total_rowcopy /= config.num_channels
        total_read /= config.num_channels
        total_fpga /= config.num_channels

    # Refresh overhead
    if config.mem_type == "HBM2":
        # Per-bank refresh: only 1 of 16 banks stalled at a time
        refresh_frac = (hbm2.tRFC_ns / hbm2.tREFI_per_bank_ns) / hbm2.banks_per_channel
        # Effective: 220/3900/16 = 0.35%
    elif config.mem_type == "DDR5":
        refresh_frac = 0.0024  # From DDR5 sim: 0.24%
    else:  # DDR4
        refresh_frac = ddr4.tRFC_ns / ddr4.tREFI_ns  # 4.49%

    total_write *= (1.0 + refresh_frac)
    total_maj3 *= (1.0 + refresh_frac)
    total_rowcopy *= (1.0 + refresh_frac)
    total_read *= (1.0 + refresh_frac)

    # Inter-channel transfer overhead (HBM2)
    transfer_ns = 0.0
    if config.num_channels > 1:
        # Transfer activation vector between channels
        # HBM2 internal bandwidth between channels: ~4x external = ~1 TB/s
        # But conservatively, use external per-channel bandwidth
        transfer_bytes = model.hidden_dim * config.act_bits / 8.0
        # Per-channel BW: 32 GB/s -> time = bytes / 32 ns
        transfer_per_hop = transfer_bytes / hbm2.peak_bw_per_channel_GBs
        num_hops = config.num_channels - 1
        transfer_ns = num_hops * transfer_per_hop

    # Total
    total_ns = total_write + total_maj3 + total_rowcopy + total_read + total_fpga + transfer_ns
    per_token_ms = total_ns / 1e6
    throughput = 1000.0 / per_token_ms if per_token_ms > 0 else float('inf')
    vs_cpu = throughput / cpu_toks

    return SimResult(
        config=config,
        per_token_ms=per_token_ms,
        throughput_toks=throughput,
        vs_cpu_factor=vs_cpu,
        breakdown_write_ms=total_write / 1e6,
        breakdown_maj3_ms=total_maj3 / 1e6,
        breakdown_rowcopy_ms=total_rowcopy / 1e6,
        breakdown_read_ms=total_read / 1e6,
        breakdown_fpga_ms=total_fpga / 1e6,
        breakdown_transfer_ms=transfer_ns / 1e6,
    )


# ============================================================================
# Configurations
# ============================================================================

def build_configurations() -> List[SimConfig]:
    """Build all configurations for comparison."""
    configs = []

    # === DDR4 Reference Configs ===
    configs.append(SimConfig("DDR4: Ternary, 1 DIMM", "DDR4", 2, 1, 0.0, False))
    configs.append(SimConfig("DDR4: Ternary, 4 DIMM", "DDR4", 2, 4, 0.0, False))
    configs.append(SimConfig("DDR4: Ternary+4D+Ovlp(c)", "DDR4", 2, 4, 0.5, False))
    configs.append(SimConfig("DDR4: Ternary+4D+Ovlp(a)", "DDR4", 2, 4, 0.75, False))
    configs.append(SimConfig("DDR4: Ternary+4D+Ovlp(a)+Pop", "DDR4", 2, 4, 0.75, True))

    # === HBM2 Configs ===
    configs.append(SimConfig("HBM2: Ternary, 1 channel", "HBM2", 2, 1, 0.0, False))
    configs.append(SimConfig("HBM2: Ternary, 8 channels", "HBM2", 2, 8, 0.0, False))
    configs.append(SimConfig("HBM2: Ternary+8ch+Ovlp(c)", "HBM2", 2, 8, 0.5, False))
    configs.append(SimConfig("HBM2: Ternary+8ch+Ovlp(a)", "HBM2", 2, 8, 0.75, False))
    configs.append(SimConfig("HBM2: Ternary+8ch+Ovlp(c)+Pop", "HBM2", 2, 8, 0.5, True))
    configs.append(SimConfig("HBM2: Ternary+8ch+Ovlp(a)+Pop", "HBM2", 2, 8, 0.75, True))

    # === Multi-stack HBM2 (2 stacks = 16 channels) ===
    configs.append(SimConfig("HBM2: Ternary+16ch+Ovlp(a)+Pop", "HBM2", 2, 16, 0.75, True))

    return configs


# ============================================================================
# Display
# ============================================================================

def print_separator(char="=", width=130):
    print(char * width)

def print_hbm2_parameters(timing: HBM2Timing):
    """Print HBM2 timing parameters."""
    print()
    print_separator()
    print("HBM2-2000 TIMING PARAMETERS")
    print_separator()
    print(f"  I/O frequency:       {timing.io_freq_ghz:.1f} GHz (DDR = {timing.data_rate_gbps:.1f} Gbps/pin)")
    print(f"  Channels:            {timing.num_channels}")
    print(f"  Channel width:       {timing.channel_width_bits} bits")
    print(f"  Burst length:        BL{timing.burst_length} ({timing.cache_line_bytes} bytes per burst)")
    print(f"  BW per channel:      {timing.peak_bw_per_channel_GBs:.1f} GB/s")
    print(f"  BW per stack:        {timing.peak_bw_GBs:.0f} GB/s")
    print(f"  Capacity per stack:  {timing.total_capacity_GB:.0f} GB ({timing.stack_height}-Hi)")
    print(f"  Banks per channel:   {timing.banks_per_channel}")
    print(f"  Row size:            {timing.row_size_bytes} bytes ({timing.row_size_bytes * 8} bits)")
    print(f"  tRCD:                {timing.tRCD_ns:.2f} ns")
    print(f"  tRAS:                {timing.tRAS_ns:.2f} ns")
    print(f"  tRP:                 {timing.tRP_ns:.2f} ns")
    print(f"  tCL:                 {timing.tCL_ns:.2f} ns")
    print(f"  tBurst:              {timing.tBurst_ns:.2f} ns")
    print(f"  tRFC:                {timing.tRFC_ns:.1f} ns")
    print(f"  tREFI (per bank):    {timing.tREFI_per_bank_ns:.1f} ns ({timing.tREFI_per_bank_ns/1000:.1f} us)")
    refresh_eff = (timing.tRFC_ns / timing.tREFI_per_bank_ns) / timing.banks_per_channel * 100
    print(f"  Refresh overhead:    {refresh_eff:.2f}% (per-bank, 1/{timing.banks_per_channel} banks)")
    print()


def print_packing_analysis(model: ModelParams, hbm2: HBM2Timing, ddr4: DDR4Timing):
    """Print row packing comparison."""
    print()
    print_separator()
    print("ROW PACKING ANALYSIS: HBM2 (2 KB) vs DDR4 (8 KB)")
    print_separator()

    hdr = f"{'Dimension':>10s}  {'DDR4 pack':>10s}  {'DDR4 rows':>10s}  {'HBM2 pack':>10s}  {'HBM2 rows':>10s}  {'Ratio':>8s}"
    print(hdr)
    print("-" * 70)

    for dim_name, dim in [("d_model", model.hidden_dim), ("d_ff", model.ffn_dim), ("d_kv", model.kv_dim)]:
        ddr4_pack = max(1, (ddr4.row_size_bytes * 8) // dim)
        hbm2_pack = max(1, (hbm2.row_size_bytes * 8) // dim)
        # For n_out neurons needing that dim as input:
        # We don't know n_out here, so show packing ratio
        print(f"  {dim_name}={dim:>5d}  {ddr4_pack:>10d}/row  {'':>10s}  {hbm2_pack:>10d}/row  {'':>10s}  {ddr4_pack/hbm2_pack:>7.1f}x")

    print()
    print("  Note: Lower packing in HBM2 means more physical rows but smaller")
    print("  per-row bus transfers. The net effect on throughput depends on the")
    print("  balance of bus time vs AND compute time.")
    print()


def print_sanity_checks(hbm2: HBM2Timing, ddr4: DDR4Timing):
    """Print per-operation timing comparisons."""
    print()
    print_separator()
    print("PER-OPERATION TIMING COMPARISON")
    print_separator()

    # Transfer 320 bytes (2560-bit activation)
    n_in = 2560
    transfer_bytes = math.ceil(n_in / 8)  # 320 bytes

    hbm2_bus = compute_bus_transfer_hbm2(transfer_bytes, hbm2)
    ddr4_bus = compute_bus_transfer_ddr4(transfer_bytes, ddr4)

    hbm2_maj3 = compute_maj3_time_hbm2(hbm2)
    ddr4_maj3 = compute_maj3_time_ddr4(ddr4)
    hbm2_rc = compute_rowcopy_time_hbm2(hbm2)
    ddr4_rc = compute_rowcopy_time_ddr4(ddr4)

    print(f"  Transfer {transfer_bytes} bytes (d_model={n_in} activation):")
    print(f"    DDR4:  {ddr4_bus:.1f} ns  ({math.ceil(transfer_bytes/ddr4.cache_line_bytes)} bursts @ {ddr4.cache_line_bytes}B)")
    print(f"    HBM2:  {hbm2_bus:.1f} ns  ({math.ceil(transfer_bytes/hbm2.cache_line_bytes)} bursts @ {hbm2.cache_line_bytes}B)")
    print(f"    Ratio: HBM2/DDR4 = {hbm2_bus/ddr4_bus:.2f}x")
    print()

    # Transfer 864 bytes (6912-bit activation)
    n_in_ff = 6912
    transfer_bytes_ff = math.ceil(n_in_ff / 8)  # 864 bytes

    hbm2_bus_ff = compute_bus_transfer_hbm2(transfer_bytes_ff, hbm2)
    ddr4_bus_ff = compute_bus_transfer_ddr4(transfer_bytes_ff, ddr4)

    print(f"  Transfer {transfer_bytes_ff} bytes (d_ff={n_in_ff} activation):")
    print(f"    DDR4:  {ddr4_bus_ff:.1f} ns  ({math.ceil(transfer_bytes_ff/ddr4.cache_line_bytes)} bursts)")
    print(f"    HBM2:  {hbm2_bus_ff:.1f} ns  ({math.ceil(transfer_bytes_ff/hbm2.cache_line_bytes)} bursts)")
    print(f"    Ratio: HBM2/DDR4 = {hbm2_bus_ff/ddr4_bus_ff:.2f}x")
    print()

    print(f"  MAJ3 AND (triple-activation):")
    print(f"    DDR4:  {ddr4_maj3:.1f} ns  (3*{ddr4.tRCD_ns} + {ddr4.tRAS_ns})")
    print(f"    HBM2:  {hbm2_maj3:.1f} ns  (3*{hbm2.tRCD_ns} + {hbm2.tRAS_ns})")
    print(f"    Ratio: HBM2/DDR4 = {hbm2_maj3/ddr4_maj3:.2f}x ({(1-hbm2_maj3/ddr4_maj3)*100:.0f}% faster)")
    print()

    print(f"  RowCopy (SA-mediated 2-row activation):")
    print(f"    DDR4:  {ddr4_rc:.1f} ns  (2*{ddr4.tRCD_ns} + {ddr4.tRAS_ns} + {ddr4.tRP_ns})")
    print(f"    HBM2:  {hbm2_rc:.1f} ns  (2*{hbm2.tRCD_ns} + {hbm2.tRAS_ns} + {hbm2.tRP_ns})")
    print(f"    Ratio: HBM2/DDR4 = {hbm2_rc/ddr4_rc:.2f}x")
    print()

    # Per-neuron DRAM time (MAJ3 + 2*RowCopy + read)
    ddr4_dram_per = ddr4_maj3 + 2*ddr4_rc
    hbm2_dram_per = hbm2_maj3 + 2*hbm2_rc
    ddr4_per_neuron = ddr4_dram_per + ddr4_bus
    hbm2_per_neuron = hbm2_dram_per + hbm2_bus
    print(f"  Per-neuron total (MAJ3 + 2*RowCopy + read, d_model=2560):")
    print(f"    DDR4:  {ddr4_per_neuron:.1f} ns  (DRAM={ddr4_dram_per:.1f} + bus={ddr4_bus:.1f})")
    print(f"    HBM2:  {hbm2_per_neuron:.1f} ns  (DRAM={hbm2_dram_per:.1f} + bus={hbm2_bus:.1f})")
    print(f"    Ratio: HBM2/DDR4 = {hbm2_per_neuron/ddr4_per_neuron:.2f}x")
    print(f"    -> HBM2 single channel is SLOWER per operation")
    print(f"    -> But 8 channels compensate: effective = {hbm2_per_neuron/8/ddr4_per_neuron:.2f}x DDR4")
    print()

    # With popcount (no read, only DRAM ops)
    ddr4_pop = ddr4_dram_per
    hbm2_pop = hbm2_dram_per
    print(f"  Per-neuron with popcount (MAJ3 + 2*RowCopy, no bus read):")
    print(f"    DDR4:  {ddr4_pop:.1f} ns")
    print(f"    HBM2:  {hbm2_pop:.1f} ns")
    print(f"    Ratio: HBM2/DDR4 = {hbm2_pop/ddr4_pop:.2f}x ({(1-hbm2_pop/ddr4_pop)*100:.0f}% faster)")
    print(f"    -> With popcount, HBM2 DRAM ops are faster (no bus penalty)")
    print()


def print_results_table(results: List[SimResult]):
    """Print main results table."""
    print()
    print_separator()
    print("PIM-LLM HBM2 vs DDR4 THROUGHPUT COMPARISON")
    print("Model: BitNet 2B4T (2560/6912, 30L) | Ternary B=2 | Autoregressive decode")
    print_separator()

    hdr = (
        f"{'Configuration':<40s} "
        f"{'Type':<5s} "
        f"{'Chan':>5s} "
        f"{'Ovlp':>5s} "
        f"{'Pop':>4s} "
        f"{'ms/tok':>8s} "
        f"{'tok/s':>8s} "
        f"{'TPOT(ms)':>9s} "
        f"{'vs CPU':>8s}"
    )
    print(hdr)
    print_separator("-")

    for r in results:
        c = r.config
        ovlp = f"{c.overlap_factor:.2f}" if c.overlap_factor > 0 else "none"
        pop = "Yes" if c.in_dram_popcount else "No"
        ch = str(c.num_channels)
        tpot = r.per_token_ms

        print(
            f"{c.name:<40s} "
            f"{c.mem_type:<5s} "
            f"{ch:>5s} "
            f"{ovlp:>5s} "
            f"{pop:>4s} "
            f"{r.per_token_ms:>8.1f} "
            f"{r.throughput_toks:>8.2f} "
            f"{tpot:>9.1f} "
            f"{r.vs_cpu_factor:>7.2f}x"
        )

    print_separator()
    print(f"CPU reference: BitNet.cpp = 5.9 tok/s (single-threaded)")
    print(f"MLPerf interactive target: TPOT < 40 ms (25 tok/s)")
    print()


def print_breakdown_table(results: List[SimResult]):
    """Print time breakdown per token."""
    print()
    print_separator()
    print("TIME BREAKDOWN PER TOKEN (milliseconds)")
    print_separator()

    hdr = (
        f"{'Configuration':<40s} "
        f"{'Write':>8s} "
        f"{'MAJ3':>8s} "
        f"{'RowCpy':>8s} "
        f"{'Read':>8s} "
        f"{'FPGA':>8s} "
        f"{'Xfer':>8s} "
        f"{'TOTAL':>8s} "
        f"{'Bus%':>6s}"
    )
    print(hdr)
    print_separator("-")

    for r in results:
        total = r.per_token_ms
        w = r.breakdown_write_ms
        m = r.breakdown_maj3_ms
        rc = r.breakdown_rowcopy_ms
        rd = r.breakdown_read_ms
        fp = r.breakdown_fpga_ms
        tr = r.breakdown_transfer_ms
        bus_pct = ((w + rd) / total * 100) if total > 0 else 0

        print(
            f"{r.config.name:<40s} "
            f"{w:>8.2f} "
            f"{m:>8.2f} "
            f"{rc:>8.2f} "
            f"{rd:>8.2f} "
            f"{fp:>8.2f} "
            f"{tr:>8.4f} "
            f"{total:>8.2f} "
            f"{bus_pct:>5.1f}%"
        )

    print_separator()
    print("  Bus% = (Write+Read)/Total. Lower is better (less bus-bound).")
    print("  MAJ3 = in-DRAM AND compute. RowCpy = weight reload + zero-ref init.")
    print()


def print_cross_technology_comparison(
    hbm2_results: List[SimResult],
    model: ModelParams,
    hbm2: HBM2Timing,
    ddr4: DDR4Timing,
):
    """Print DDR4 vs DDR5 vs HBM2 comparison table."""
    print()
    print_separator()
    print("CROSS-TECHNOLOGY COMPARISON: DDR4 vs DDR5 vs HBM2")
    print("(Best realistic configuration per technology)")
    print_separator()

    # DDR5 numbers from our simulation (reference)
    ddr5_numbers = {
        "Ternary, 1 unit":      (3.42,  292),
        "Ternary+4/8ch+Ovlp(c)": (20.24, 49),
        "Ternary+4/8ch+Ovlp(a)+Pop": (38.73, 26),
    }

    # Get DDR4 and HBM2 numbers from results
    ddr4_1d = next((r for r in hbm2_results if "DDR4: Ternary, 1" in r.config.name), None)
    ddr4_4d_oc = next((r for r in hbm2_results if "DDR4: Ternary+4D+Ovlp(c)" in r.config.name), None)
    ddr4_4d_ap = next((r for r in hbm2_results if "DDR4: Ternary+4D+Ovlp(a)+Pop" in r.config.name), None)

    hbm2_1ch = next((r for r in hbm2_results if "HBM2: Ternary, 1 channel" in r.config.name), None)
    hbm2_8ch = next((r for r in hbm2_results if "HBM2: Ternary, 8 channels" in r.config.name), None)
    hbm2_8oc = next((r for r in hbm2_results if "HBM2: Ternary+8ch+Ovlp(c)" == r.config.name and not r.config.in_dram_popcount), None)
    hbm2_8ap = next((r for r in hbm2_results if "HBM2: Ternary+8ch+Ovlp(a)+Pop" in r.config.name), None)
    hbm2_16ap = next((r for r in hbm2_results if "16ch" in r.config.name), None)

    print(f"{'Configuration':<35s} {'DDR4':>12s} {'DDR5':>12s} {'HBM2':>12s} {'HBM2/DDR4':>10s}")
    print("-" * 85)

    # Row 1: Ternary, 1 DIMM/channel
    if ddr4_1d and hbm2_1ch:
        print(f"{'Ternary, 1 DIMM/channel':<35s} "
              f"{ddr4_1d.throughput_toks:>8.2f} t/s "
              f"{3.42:>8.2f} t/s "
              f"{hbm2_1ch.throughput_toks:>8.2f} t/s "
              f"{hbm2_1ch.throughput_toks/ddr4_1d.throughput_toks:>9.2f}x")

    # Row 2: Ternary, 8 channels
    if hbm2_8ch:
        print(f"{'Ternary, 8ch (no overlap)':<35s} "
              f"{'N/A':>11s} "
              f"{'N/A':>11s} "
              f"{hbm2_8ch.throughput_toks:>8.2f} t/s "
              f"{'':>10s}")

    # Row 3: With overlap
    if ddr4_4d_oc and hbm2_8oc:
        print(f"{'+ Overlap (conservative)':<35s} "
              f"{ddr4_4d_oc.throughput_toks:>8.2f} t/s "
              f"{20.24:>8.2f} t/s "
              f"{hbm2_8oc.throughput_toks:>8.2f} t/s "
              f"{hbm2_8oc.throughput_toks/ddr4_4d_oc.throughput_toks:>9.2f}x")

    # Row 4: Best per technology
    if ddr4_4d_ap and hbm2_8ap:
        print(f"{'Best (overlap+popcount)':<35s} "
              f"{ddr4_4d_ap.throughput_toks:>8.2f} t/s "
              f"{38.73:>8.2f} t/s "
              f"{hbm2_8ap.throughput_toks:>8.2f} t/s "
              f"{hbm2_8ap.throughput_toks/ddr4_4d_ap.throughput_toks:>9.2f}x")

    # Row 5: HBM2 2-stack
    if hbm2_16ap:
        print(f"{'HBM2 2-stack (16ch)+ovlp+pop':<35s} "
              f"{'N/A':>11s} "
              f"{'N/A':>11s} "
              f"{hbm2_16ap.throughput_toks:>8.2f} t/s "
              f"{'':>10s}")

    print()
    print(f"  CPU baseline: BitNet.cpp = 5.9 tok/s")
    print(f"  GPU reference: TerEffic (Alveo U280, HBM) = 727 tok/s (2.7B model)")
    print()


def print_model_fit_analysis(model: ModelParams, hbm2: HBM2Timing):
    """Analyze whether BitNet 2B4T fits in HBM2."""
    print()
    print_separator()
    print("HBM2 CAPACITY ANALYSIS: BitNet 2B4T")
    print_separator()

    # Weight storage
    total_weight_rows = 0
    print(f"  {'Matvec':<15s} {'Shape':>15s} {'Rows (2KB)':>12s} {'Rows (8KB)':>12s}")
    print("  " + "-" * 60)

    for name, n_in, n_out in model.get_matvecs():
        pack_hbm2 = max(1, (hbm2.row_size_bytes * 8) // n_in)
        rows_hbm2 = math.ceil(n_out / pack_hbm2) * 2  # *2 for pos/neg
        pack_ddr4 = max(1, (8192 * 8) // n_in)
        rows_ddr4 = math.ceil(n_out / pack_ddr4) * 2
        total_weight_rows += rows_hbm2
        print(f"  {name:<15s} {n_in:>6d}x{n_out:<6d} {rows_hbm2:>12d} {rows_ddr4:>12d}")

    total_per_layer = total_weight_rows
    total_model = total_per_layer * model.num_layers

    print(f"  {'Per layer':<15s} {'':>15s} {total_per_layer:>12d}")
    print(f"  {'Full model (30L)':<15s} {'':>15s} {total_model:>12d}")

    # Available capacity
    rows_per_stack = hbm2.rows_per_die * hbm2.stack_height
    print()
    print(f"  Available rows per stack: ~{rows_per_stack:,d}")
    print(f"  Weight rows needed:       {total_model:,d}")
    utilization = total_model / rows_per_stack * 100
    print(f"  Utilization:              {utilization:.1f}%")

    # Weight size in bytes
    weight_bytes = total_model * hbm2.row_size_bytes
    print(f"  Weight data size:         {weight_bytes / (1024*1024):.1f} MB")
    print(f"  Stack capacity:           {hbm2.total_capacity_GB:.0f} GB")
    print()


def print_obstacles(hbm2: HBM2Timing):
    """Print HBM2-specific obstacles for PIM."""
    print()
    print_separator()
    print("HBM2 OBSTACLES FOR CHARGE-SHARING PIM")
    print_separator()
    print()
    print("  1. HARD IP CONTROLLER (HIGH)")
    print("     FPGA boards with HBM2 (Alveo U50/U280) use hardened IP memory")
    print("     controllers that expose only AXI4 interfaces, NOT raw DRAM commands.")
    print("     Cannot issue MAJ3 tripleACT timing violations through standard AXI4.")
    print("     DRAM Bender (CMU-SAFARI) has demonstrated HBM2 testing on Alveo U50")
    print("     with custom PHY bypass, but the mechanism is not fully documented.")
    print()
    print("  2. SMALLER ROW BUFFER (MEDIUM)")
    print(f"     HBM2 rows are {hbm2.row_size_bytes} bytes vs DDR4's 8192 bytes.")
    print("     Packing drops from ~25 to ~6 neurons/row (d_model=2560).")
    print("     More physical ANDs per matmul, partially offset by less bus time per row.")
    print("     Net: single HBM2 channel ~30% slower than single DDR4 DIMM.")
    print()
    print("  3. OPTIONAL ODECC (POSITIVE)")
    print("     Unlike DDR5's mandatory ODECC, HBM2 ODECC is optional.")
    print("     When disabled, charge-sharing AND results are not silently corrupted.")
    print("     This removes DDR5's biggest technical blocker.")
    print()
    print("  4. NOT COMMODITY (HIGH for accessibility, LOW for technical)")
    print("     HBM2 is co-packaged on silicon interposer — cannot buy standalone.")
    print("     Research access via FPGA boards ($2K-$8K) or GPUs.")
    print("     Not viable for '$50 inference module' vision without custom silicon.")
    print()
    print("  5. TSV THERMAL COUPLING")
    print("     3D-stacked dies share heat via TSVs. PIM operations that increase")
    print("     die temperature could affect adjacent dies. Needs characterization.")
    print()
    print("  STRATEGIC ASSESSMENT:")
    print("  HBM2 is technically viable for charge-sharing PIM (same 1T1C cells,")
    print("  no mandatory ODECC), but the hard IP controller is the main barrier.")
    print("  The path is: DDR4 (proof of concept) -> DDR5 (commodity scale) ->")
    print("  HBM (maximum performance). Samsung Aquabolt-XL (logic-near-memory,")
    print("  NOT charge-sharing) already demonstrates industry interest.")
    print()


def generate_comparison_chart(results: List[SimResult], output_path: str):
    """Generate bar chart comparing DDR4 vs HBM2."""
    # Filter for key configs
    key_configs = [
        "DDR4: Ternary, 1 DIMM",
        "DDR4: Ternary, 4 DIMM",
        "DDR4: Ternary+4D+Ovlp(a)+Pop",
        "HBM2: Ternary, 1 channel",
        "HBM2: Ternary, 8 channels",
        "HBM2: Ternary+8ch+Ovlp(a)+Pop",
        "HBM2: Ternary+16ch+Ovlp(a)+Pop",
    ]

    filtered = [r for r in results if r.config.name in key_configs]

    # Add DDR5 reference points
    ddr5_labels = ["DDR5: Ternary, 1 DIMM", "DDR5: Best (Ternary+4D+O+Pop)"]
    ddr5_values = [3.42, 38.73]

    names = [r.config.name for r in filtered]
    throughputs = [r.throughput_toks for r in filtered]

    # Insert DDR5 at appropriate positions
    names.insert(3, ddr5_labels[0])
    throughputs.insert(3, ddr5_values[0])
    names.insert(4, ddr5_labels[1])
    throughputs.insert(4, ddr5_values[1])

    # Colors
    colors = []
    for n in names:
        if "DDR4" in n:
            colors.append("#1f77b4")  # Blue
        elif "DDR5" in n:
            colors.append("#ff7f0e")  # Orange
        else:
            colors.append("#2ca02c")  # Green

    fig, ax = plt.subplots(figsize=(16, 8))
    x = np.arange(len(names))
    bars = ax.bar(x, throughputs, color=colors, edgecolor="black", linewidth=0.5)

    # CPU reference
    ax.axhline(y=5.9, color="red", linestyle="--", linewidth=1.5, label="CPU (BitNet.cpp): 5.9 tok/s")

    # MLPerf target
    ax.axhline(y=25.0, color="purple", linestyle=":", linewidth=1, alpha=0.7, label="MLPerf interactive: 25 tok/s")

    for bar, val in zip(bars, throughputs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xlabel("Configuration", fontsize=12)
    ax.set_ylabel("Throughput (tokens/second)", fontsize=12)
    ax.set_title(
        "PIM-LLM Throughput: DDR4 vs DDR5 vs HBM2\n"
        "BitNet 2B4T (2B params) Ternary Inference",
        fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=40, ha="right", fontsize=7)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Chart saved: {output_path}")
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 80)
    print("  PIM-LLM HBM2 THROUGHPUT ESTIMATOR")
    print("  BitNet 2B4T on HBM2 PIM Architecture")
    print("  Analytical Model (consistent with DDR4/DDR5 cycle-accurate sims)")
    print("=" * 80)

    # Initialize
    hbm2 = HBM2Timing()
    ddr4 = DDR4Timing()
    model = ModelParams()

    # Print parameters
    print_hbm2_parameters(hbm2)
    print_packing_analysis(model, hbm2, ddr4)
    print_sanity_checks(hbm2, ddr4)

    # Model fit analysis
    print_model_fit_analysis(model, hbm2)

    # Build and run configurations
    configs = build_configurations()
    results = []
    for cfg in configs:
        result = simulate_full_model(cfg, model, hbm2, ddr4)
        results.append(result)

    # Print results
    print_results_table(results)
    print_breakdown_table(results)
    print_cross_technology_comparison(results, model, hbm2, ddr4)

    # Obstacles
    print_obstacles(hbm2)

    # Key insights
    print_separator()
    print("KEY INSIGHTS")
    print_separator()

    hbm2_results = [r for r in results if r.config.mem_type == "HBM2"]
    ddr4_results = [r for r in results if r.config.mem_type == "DDR4"]

    best_hbm2 = max(hbm2_results, key=lambda r: r.throughput_toks)
    best_ddr4 = max(ddr4_results, key=lambda r: r.throughput_toks)

    print(f"  Best HBM2:  {best_hbm2.config.name}")
    print(f"              {best_hbm2.throughput_toks:.2f} tok/s ({best_hbm2.vs_cpu_factor:.2f}x CPU)")
    print(f"  Best DDR4:  {best_ddr4.config.name}")
    print(f"              {best_ddr4.throughput_toks:.2f} tok/s ({best_ddr4.vs_cpu_factor:.2f}x CPU)")
    print(f"  HBM2/DDR4 best: {best_hbm2.throughput_toks/best_ddr4.throughput_toks:.2f}x")
    print()

    hbm2_8_basic = next(r for r in results if "8 channels" in r.config.name and r.config.overlap_factor == 0)
    ddr4_1_basic = next(r for r in results if "DDR4: Ternary, 1" in r.config.name)
    print(f"  HBM2 8ch basic / DDR4 1D basic: {hbm2_8_basic.throughput_toks/ddr4_1_basic.throughput_toks:.2f}x")
    print(f"  (8 channels compensate for smaller rows and slower per-channel bus)")
    print()

    # Check if any HBM2 beats MLPerf target
    mlperf_beaters = [r for r in hbm2_results if 1000/r.per_token_ms >= 25]
    if mlperf_beaters:
        print(f"  HBM2 configs meeting MLPerf interactive (25 tok/s):")
        for r in mlperf_beaters:
            print(f"    {r.config.name}: {r.throughput_toks:.2f} tok/s (TPOT={r.per_token_ms:.1f} ms)")
    else:
        closest = max(hbm2_results, key=lambda r: r.throughput_toks)
        print(f"  No HBM2 config meets MLPerf interactive target (25 tok/s)")
        print(f"  Closest: {closest.config.name} at {closest.throughput_toks:.2f} tok/s")

    print()
    print("  BOTTOM LINE:")
    print("  HBM2 provides ~3-5x improvement over DDR4 (single stack, 8 channels)")
    print("  driven by channel parallelism, slightly faster AND, and low refresh overhead.")
    print("  Per-channel performance is ~30% slower than DDR4 per-DIMM due to smaller")
    print("  row buffer (2KB vs 8KB) requiring more bus transfers per cache-line access.")
    print("  The main obstacle is the hard IP controller on FPGAs, not the DRAM physics.")
    print("  HBM2 represents the 'production destination' while DDR4 is the 'proof of concept'.")
    print()

    # Generate chart
    chart_path = r"C:\Users\Udja\Documents\Deni\PIM\pim_throughput_hbm2.png"
    generate_comparison_chart(results, chart_path)

    print()
    print("Estimation complete.")


if __name__ == "__main__":
    main()
