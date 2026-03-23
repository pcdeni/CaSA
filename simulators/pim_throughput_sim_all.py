#!/usr/bin/env python3
"""
PIM-LLM Cross-Technology Throughput Estimator
===============================================
Unified analytical throughput model for BitNet b1.58-2B-4T across ALL memory
technologies considered in the PIM-LLM paper:

  DDR4-2400     — validated against cycle-accurate sim
  DDR5-4800     — with PRAC overhead modeled (honest assessment)
  LPDDR5X-6400  — 8/16 independent narrow channels
  CAMM2         — same as LPDDR5X (module form factor, not electrical)
  HBM2-2000     — validated against HBM2 analytical sim
  HBM3E-9600    — extrapolation from HBM2

Uses the same methodology as pim_throughput_sim_hbm2.py for all technologies,
varying only the timing parameters and channel geometry.

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
from typing import List, Tuple, Dict


# ============================================================================
# Model Parameters (BitNet b1.58-2B-4T — real dimensions)
# ============================================================================

@dataclass
class ModelParams:
    """Architecture parameters for BitNet b1.58-2B-4T."""
    num_layers: int = 30
    hidden_dim: int = 2560
    ffn_dim: int = 6912
    kv_dim: int = 640

    def get_matvecs(self) -> List[Tuple[str, int, int]]:
        h, f, kv = self.hidden_dim, self.ffn_dim, self.kv_dim
        return [
            ("Q_proj",   h, h),
            ("K_proj",   h, kv),
            ("V_proj",   h, kv),
            ("O_proj",   h, h),
            ("FFN_gate", h, f),
            ("FFN_up",   h, f),
            ("FFN_down", f, h),
        ]


# ============================================================================
# Memory Technology Timing Parameters
# ============================================================================

@dataclass
class MemTiming:
    """Unified timing parameters for any DRAM technology."""
    name: str
    # Channel geometry
    num_channels: int          # channels per unit (DIMM/package/stack)
    channel_width_bits: int    # data bus width per channel
    burst_length: int          # burst length
    data_rate_gbps: float      # per-pin data rate (Gbps)
    # Derived: bytes per burst
    # burst_bytes = channel_width_bits * burst_length / 8
    # cache_line_bytes = burst_bytes (one column access)

    # Core timing (ns)
    tRCD_ns: float
    tRAS_ns: float
    tRP_ns: float              # base tRP (before PRAC overhead)
    tCL_ns: float
    tBurst_ns: float           # time to transfer one burst

    # Row geometry
    row_size_bytes: int        # row buffer size

    # Bank architecture
    banks_per_channel: int

    # Refresh
    tREFI_ns: float            # per-bank refresh interval (for per-bank refresh)
    tRFC_ns: float
    per_bank_refresh: bool     # True for DDR5, HBM2, LPDDR5X; False for DDR4

    # DDR5-specific: PRAC overhead
    prac_enabled: bool = False
    prac_tRP_extra_ns: float = 0.0   # additional ns added to tRP by PRAC

    # ODECC
    odecc_mandatory: bool = False
    odecc_note: str = ""

    @property
    def burst_bytes(self) -> int:
        return self.channel_width_bits * self.burst_length // 8

    @property
    def effective_tRP(self) -> float:
        return self.tRP_ns + self.prac_tRP_extra_ns

    @property
    def peak_bw_per_channel_GBs(self) -> float:
        return self.data_rate_gbps * self.channel_width_bits / 8.0

    @property
    def peak_bw_total_GBs(self) -> float:
        return self.peak_bw_per_channel_GBs * self.num_channels

    @property
    def refresh_overhead(self) -> float:
        if self.per_bank_refresh:
            return (self.tRFC_ns / self.tREFI_ns) / self.banks_per_channel
        else:
            return self.tRFC_ns / self.tREFI_ns

    @property
    def neurons_per_row(self) -> int:
        """Neurons packed per row for d_model=2560."""
        return max(1, (self.row_size_bytes * 8) // 2560)

    @property
    def maj3_time_ns(self) -> float:
        """MAJ3 tripleACT time (3 rows simultaneous)."""
        return 3 * self.tRCD_ns + self.tRAS_ns

    @property
    def rowcopy_time_ns(self) -> float:
        """SA-mediated RowCopy time (ACT source -> SA latches -> ACT dest)."""
        return 2 * self.tRCD_ns + self.tRAS_ns + self.effective_tRP

    @property
    def and_time_ns(self) -> float:
        """Backward compat alias. Full per-weight-row DRAM time (MAJ3 + 2×RowCopy)."""
        return self.maj3_time_ns + 2 * self.rowcopy_time_ns

    def bus_transfer_ns(self, num_bytes: int) -> float:
        """Time to transfer num_bytes over one channel (conservative model)."""
        num_bursts = math.ceil(num_bytes / self.burst_bytes)
        return (self.tRCD_ns
                + num_bursts * (self.tCL_ns + self.tBurst_ns)
                + self.effective_tRP)


# ============================================================================
# Define All Technologies
# ============================================================================

def make_ddr4() -> MemTiming:
    return MemTiming(
        name="DDR4-2400",
        num_channels=1,
        channel_width_bits=64,
        burst_length=8,
        data_rate_gbps=2.4,     # 2400 MT/s, DDR
        tRCD_ns=13.75,
        tRAS_ns=35.0,
        tRP_ns=13.75,
        tCL_ns=13.75,
        tBurst_ns=3.33,         # BL8 at 1200 MHz
        row_size_bytes=8192,
        banks_per_channel=16,
        tREFI_ns=7800.0,
        tRFC_ns=350.0,
        per_bank_refresh=False,
        odecc_mandatory=False,
        odecc_note="None",
    )

def make_ddr5() -> MemTiming:
    return MemTiming(
        name="DDR5-4800",
        num_channels=2,         # 2 independent 32-bit sub-channels per DIMM
        channel_width_bits=32,
        burst_length=16,        # BL16
        data_rate_gbps=4.8,
        tRCD_ns=13.75,
        tRAS_ns=32.0,
        tRP_ns=13.75,           # base tRP
        tCL_ns=13.75,
        tBurst_ns=3.33,         # BL16 at 2400 MHz = 16/4.8GHz ≈ 3.33ns
        row_size_bytes=8192,
        banks_per_channel=32,
        tREFI_ns=3900.0,
        tRFC_ns=295.0,
        per_bank_refresh=True,
        prac_enabled=True,
        prac_tRP_extra_ns=22.0, # PRAC adds ~22ns to tRP per JESD79-5C
        odecc_mandatory=True,
        odecc_note="MANDATORY — primary blocker, requires MRS bypass or BEER recovery",
    )

def make_ddr5_no_prac() -> MemTiming:
    """DDR5 without PRAC overhead (pre-5C spec or theoretical best case)."""
    t = make_ddr5()
    t.name = "DDR5-4800 (no PRAC)"
    t.prac_enabled = False
    t.prac_tRP_extra_ns = 0.0
    return t

def make_lpddr5x() -> MemTiming:
    return MemTiming(
        name="LPDDR5X-6400",
        num_channels=8,         # 8 independent 16-bit channels per package
        channel_width_bits=16,
        burst_length=16,        # BL16
        data_rate_gbps=6.4,
        tRCD_ns=18.0,           # LPDDR5X typical (higher than DDR5)
        tRAS_ns=42.0,           # longer row active time
        tRP_ns=18.0,            # longer precharge
        tCL_ns=18.0,
        tBurst_ns=2.5,          # BL16 at 6.4 Gbps
        row_size_bytes=2048,    # 2 KB row buffer (narrow channels)
        banks_per_channel=16,   # 4 bank groups × 4 banks
        tREFI_ns=3900.0,
        tRFC_ns=210.0,
        per_bank_refresh=True,
        prac_enabled=False,     # Mobile: TRR, not PRAC
        odecc_mandatory=False,  # Varies by vendor — not always mandatory
        odecc_note="Varies — not mandatory on all LPDDR5X dies",
    )

def make_lpddr5x_16ch() -> MemTiming:
    """16-channel LPDDR5X (high-end laptop/tablet, dual-rank)."""
    t = make_lpddr5x()
    t.name = "LPDDR5X-6400 (16ch)"
    t.num_channels = 16
    return t

def make_hbm2() -> MemTiming:
    return MemTiming(
        name="HBM2-2000",
        num_channels=8,
        channel_width_bits=128,
        burst_length=2,         # BL2 legacy mode
        data_rate_gbps=2.0,
        tRCD_ns=14.0,
        tRAS_ns=28.0,
        tRP_ns=14.0,
        tCL_ns=14.0,
        tBurst_ns=1.0,
        row_size_bytes=2048,
        banks_per_channel=16,
        tREFI_ns=3900.0,
        tRFC_ns=220.0,
        per_bank_refresh=True,
        odecc_mandatory=False,
        odecc_note="Optional — no ODECC blocker",
    )

def make_hbm3e() -> MemTiming:
    return MemTiming(
        name="HBM3E-9600",
        num_channels=16,        # 16 channels per stack (vs HBM2's 8)
        channel_width_bits=64,  # 64-bit per channel (vs HBM2's 128-bit)
        burst_length=8,         # BL8
        data_rate_gbps=9.6,     # 9.6 Gbps per pin
        tRCD_ns=12.0,           # faster (smaller process)
        tRAS_ns=26.0,
        tRP_ns=12.0,
        tCL_ns=12.0,
        tBurst_ns=0.83,         # BL8 at 9.6 Gbps
        row_size_bytes=2048,    # same 2 KB row buffer
        banks_per_channel=32,   # doubled vs HBM2
        tREFI_ns=3900.0,
        tRFC_ns=200.0,
        per_bank_refresh=True,
        odecc_mandatory=False,  # Optional (vendor-dependent)
        odecc_note="Optional — similar to HBM2",
    )


# ============================================================================
# Simulation Engine
# ============================================================================

@dataclass
class SimResult:
    name: str
    mem_name: str
    num_channels: int
    act_bits: int
    overlap: float
    popcount: bool
    per_token_ms: float
    tok_s: float
    vs_cpu: float
    write_ms: float
    and_ms: float
    read_ms: float
    fpga_ms: float
    refresh_pct: float
    odecc_note: str
    prac_note: str


def simulate_one_token(
    timing: MemTiming,
    model: ModelParams,
    act_bits: int = 2,         # 2=ternary, 4=4-bit, 8=INT8
    use_channels: int = 0,     # 0 = use all channels
    overlap_factor: float = 0.0,
    in_dram_popcount: bool = False,
    cpu_tok_s: float = 5.9,
    label: str = "",
) -> SimResult:
    """Simulate one full token of inference."""

    n_ch = use_channels if use_channels > 0 else timing.num_channels
    num_passes = act_bits * 2  # pos/neg weight halves

    total_write = 0.0
    total_and = 0.0
    total_read = 0.0
    total_fpga = 0.0

    for mv_name, n_in, n_out in model.get_matvecs():
        transfer_bytes = math.ceil(n_in / 8)

        write_per_pass = timing.bus_transfer_ns(transfer_bytes)
        and_per_op = timing.and_time_ns

        if in_dram_popcount:
            read_per_op = 0.0
        else:
            read_per_op = timing.bus_transfer_ns(transfer_bytes)

        # How many output neurons per AND (row packing)
        neurons_per_row = max(1, (timing.row_size_bytes * 8) // n_in)
        num_groups = math.ceil(n_out / neurons_per_row)

        # Per-matvec totals
        mv_write = num_passes * write_per_pass
        mv_and = num_passes * num_groups * and_per_op
        mv_read = num_passes * num_groups * read_per_op
        mv_fpga = n_out * 10.0  # 10 ns per output (accumulate, scale)

        total_write += mv_write
        total_and += mv_and
        total_read += mv_read
        total_fpga += mv_fpga

    # Scale across all layers
    total_write *= model.num_layers
    total_and *= model.num_layers
    total_read *= model.num_layers
    total_fpga *= model.num_layers

    # Multi-channel parallelism (pipeline: divide work by channels)
    if n_ch > 1:
        total_write /= n_ch
        total_and /= n_ch
        total_read /= n_ch
        total_fpga /= n_ch

    # Refresh overhead
    ref_pct = timing.refresh_overhead * 100
    ref_factor = 1.0 + timing.refresh_overhead
    total_write *= ref_factor
    total_and *= ref_factor
    total_read *= ref_factor

    # Pipelining model: DRAM ops (MAJ3 + RowCopy) don't use the bus.
    # With multi-bank pipelining, DRAM and bus ops overlap.
    # Effective time per weight group = max(bus_time, dram_time/pipeline_banks)
    if overlap_factor > 0:
        pipeline_banks = 4 if overlap_factor <= 0.5 else 8
        # Per-group: bus = read, DRAM = and_time (which includes MAJ3 + 2*RowCopy)
        # DRAM is distributed across pipeline_banks
        # If bus > dram/banks: bus-bound (DRAM hidden)
        # If bus < dram/banks: DRAM-bound (bus hidden)
        dram_per_bank = total_and / pipeline_banks if pipeline_banks > 1 else total_and
        if total_read >= dram_per_bank:
            # Bus-bound: DRAM ops fully hidden
            total_and = 0.0  # hidden
        else:
            # DRAM-bound: bus ops partially hidden
            total_and = dram_per_bank
            total_read = 0.0  # hidden
            total_write = 0.0  # hidden

    total_ns = total_write + total_and + total_read + total_fpga
    per_token_ms = total_ns / 1e6
    tok_s = 1000.0 / per_token_ms if per_token_ms > 0 else 0.0

    prac_note = ""
    if timing.prac_enabled:
        prac_note = f"+{timing.prac_tRP_extra_ns:.0f}ns/tRP"

    return SimResult(
        name=label or f"{timing.name} ({n_ch}ch, {act_bits}b)",
        mem_name=timing.name,
        num_channels=n_ch,
        act_bits=act_bits,
        overlap=overlap_factor,
        popcount=in_dram_popcount,
        per_token_ms=per_token_ms,
        tok_s=tok_s,
        vs_cpu=tok_s / cpu_tok_s,
        write_ms=total_write / 1e6,
        and_ms=total_and / 1e6,
        read_ms=total_read / 1e6,
        fpga_ms=total_fpga / 1e6,
        refresh_pct=ref_pct,
        odecc_note=timing.odecc_note,
        prac_note=prac_note,
    )


# ============================================================================
# Main
# ============================================================================

def main():
    model = ModelParams()

    ddr4 = make_ddr4()
    ddr5 = make_ddr5()
    ddr5_noprac = make_ddr5_no_prac()
    lpddr5x = make_lpddr5x()
    lpddr5x_16 = make_lpddr5x_16ch()
    hbm2 = make_hbm2()
    hbm3e = make_hbm3e()

    all_timings = [ddr4, ddr5, ddr5_noprac, lpddr5x, lpddr5x_16, hbm2, hbm3e]

    # =====================================================================
    # Print technology comparison table
    # =====================================================================
    print("=" * 130)
    print("PIM-LLM CROSS-TECHNOLOGY THROUGHPUT ESTIMATOR")
    print("Model: BitNet b1.58-2B-4T (d=2560, d_ff=6912, 30 layers)")
    print("=" * 130)

    print("\n--- Memory Technology Parameters ---\n")
    hdr = f"{'Technology':<22} {'Ch':>3} {'Width':>6} {'BL':>3} {'Row':>5} {'N/row':>5} " \
          f"{'BW/ch':>7} {'BW tot':>7} {'tRP*':>6} {'AND':>5} {'Ref%':>5} {'ODECC':<12} {'PRAC'}"
    print(hdr)
    print("-" * len(hdr))
    for t in all_timings:
        print(f"{t.name:<22} {t.num_channels:>3} {t.channel_width_bits:>4}b {t.burst_length:>3} "
              f"{t.row_size_bytes:>4}B {t.neurons_per_row:>5} "
              f"{t.peak_bw_per_channel_GBs:>6.1f} {t.peak_bw_total_GBs:>6.1f} "
              f"{t.effective_tRP:>5.1f} {t.and_time_ns:>5.1f} "
              f"{t.refresh_overhead*100:>4.2f} "
              f"{'MANDATORY' if t.odecc_mandatory else 'Optional':<12} "
              f"{'YES (+{:.0f}ns)'.format(t.prac_tRP_extra_ns) if t.prac_enabled else 'No'}")

    print("\n* tRP = base + PRAC overhead. MAJ3 = 3*tRCD + tRAS. RowCopy = 2*tRCD + tRAS + tRP.")
    print("  N/row = neurons packed per row for d_model=2560.")

    # =====================================================================
    # Run simulations across all configs
    # =====================================================================
    results: List[SimResult] = []

    # --- DDR4 ---
    results.append(simulate_one_token(ddr4, model, act_bits=8, label="DDR4: 1D, INT8 (baseline)"))
    results.append(simulate_one_token(ddr4, model, act_bits=8, use_channels=4, label="DDR4: 4D, INT8"))
    results.append(simulate_one_token(ddr4, model, act_bits=4, use_channels=4, label="DDR4: 4D, 4-bit"))
    results.append(simulate_one_token(ddr4, model, act_bits=4, use_channels=4, overlap_factor=0.5, label="DDR4: 4D+ovlp, 4-bit"))
    results.append(simulate_one_token(ddr4, model, act_bits=4, use_channels=4, overlap_factor=0.5, in_dram_popcount=True, label="DDR4: 4D+ovlp+pop, 4-bit"))

    # --- DDR5 with PRAC (honest) ---
    results.append(simulate_one_token(ddr5, model, act_bits=8, use_channels=2, label="DDR5+PRAC: 1D, INT8"))
    results.append(simulate_one_token(ddr5, model, act_bits=8, use_channels=8, label="DDR5+PRAC: 4D, INT8"))
    results.append(simulate_one_token(ddr5, model, act_bits=4, use_channels=8, overlap_factor=0.5, in_dram_popcount=True, label="DDR5+PRAC: 4D+ovlp+pop, 4-bit"))

    # --- DDR5 without PRAC (pre-5C or if PRAC disabled) ---
    results.append(simulate_one_token(ddr5_noprac, model, act_bits=8, use_channels=2, label="DDR5 noPRAC: 1D, INT8"))
    results.append(simulate_one_token(ddr5_noprac, model, act_bits=4, use_channels=8, overlap_factor=0.5, in_dram_popcount=True, label="DDR5 noPRAC: 4D+ovlp+pop, 4-bit"))

    # --- LPDDR5X (8 channels) ---
    results.append(simulate_one_token(lpddr5x, model, act_bits=8, label="LPDDR5X-8ch: INT8"))
    results.append(simulate_one_token(lpddr5x, model, act_bits=4, label="LPDDR5X-8ch: 4-bit"))
    results.append(simulate_one_token(lpddr5x, model, act_bits=4, overlap_factor=0.5, label="LPDDR5X-8ch: 4-bit+ovlp"))
    results.append(simulate_one_token(lpddr5x, model, act_bits=4, overlap_factor=0.5, in_dram_popcount=True, label="LPDDR5X-8ch: 4-bit+ovlp+pop"))

    # --- LPDDR5X (16 channels) ---
    results.append(simulate_one_token(lpddr5x_16, model, act_bits=8, label="LPDDR5X-16ch: INT8"))
    results.append(simulate_one_token(lpddr5x_16, model, act_bits=4, overlap_factor=0.5, in_dram_popcount=True, label="LPDDR5X-16ch: 4-bit+ovlp+pop"))

    # --- HBM2 ---
    results.append(simulate_one_token(hbm2, model, act_bits=8, label="HBM2: 8ch, INT8"))
    results.append(simulate_one_token(hbm2, model, act_bits=4, overlap_factor=0.5, label="HBM2: 8ch, 4-bit+ovlp"))
    results.append(simulate_one_token(hbm2, model, act_bits=4, overlap_factor=0.5, in_dram_popcount=True, label="HBM2: 8ch+ovlp+pop, 4-bit"))
    results.append(simulate_one_token(hbm2, model, act_bits=4, use_channels=16, overlap_factor=0.5, in_dram_popcount=True, label="HBM2: 2-stack+ovlp+pop, 4-bit"))

    # --- HBM3E ---
    results.append(simulate_one_token(hbm3e, model, act_bits=8, label="HBM3E: 16ch, INT8"))
    results.append(simulate_one_token(hbm3e, model, act_bits=4, overlap_factor=0.5, label="HBM3E: 16ch, 4-bit+ovlp"))
    results.append(simulate_one_token(hbm3e, model, act_bits=4, overlap_factor=0.5, in_dram_popcount=True, label="HBM3E: 16ch+ovlp+pop, 4-bit"))
    results.append(simulate_one_token(hbm3e, model, act_bits=4, use_channels=32, overlap_factor=0.5, in_dram_popcount=True, label="HBM3E: 2-stack+ovlp+pop, 4-bit"))

    # =====================================================================
    # Print results
    # =====================================================================
    print("\n")
    print("=" * 130)
    print("THROUGHPUT RESULTS")
    print("=" * 130)

    hdr = f"{'Configuration':<40} {'TPOT(ms)':>9} {'tok/s':>7} {'vs CPU':>7} " \
          f"{'Write%':>7} {'AND%':>7} {'Read%':>7} {'FPGA%':>7} {'Popcount':<10} {'ODECC blocker?'}"
    print(hdr)
    print("-" * len(hdr))

    prev_mem = ""
    for r in results:
        if r.mem_name != prev_mem:
            if prev_mem:
                print()
            prev_mem = r.mem_name

        total = r.write_ms + r.and_ms + r.read_ms + r.fpga_ms
        if total > 0:
            w_pct = r.write_ms / total * 100
            a_pct = r.and_ms / total * 100
            rd_pct = r.read_ms / total * 100
            f_pct = r.fpga_ms / total * 100
        else:
            w_pct = a_pct = rd_pct = f_pct = 0

        pop_str = "In-DRAM" if r.popcount else "FPGA"
        odecc = r.odecc_note if r.odecc_note else "—"
        # Truncate ODECC note
        if len(odecc) > 30:
            odecc = odecc[:27] + "..."

        print(f"{r.name:<40} {r.per_token_ms:>9.1f} {r.tok_s:>7.1f} {r.vs_cpu:>6.2f}x "
              f"{w_pct:>6.1f}% {a_pct:>6.1f}% {rd_pct:>6.1f}% {f_pct:>6.1f}% "
              f"{pop_str:<10} {odecc}")

    # =====================================================================
    # Summary: best config per technology
    # =====================================================================
    print("\n")
    print("=" * 130)
    print("BEST CONFIGURATION PER TECHNOLOGY (for paper Table 8.6)")
    print("=" * 130)

    # Group by technology
    tech_groups = {}
    for r in results:
        key = r.mem_name
        if key not in tech_groups or r.tok_s > tech_groups[key].tok_s:
            tech_groups[key] = r

    # Also track best bus-limited (no popcount) per technology
    tech_bus_limited = {}
    for r in results:
        key = r.mem_name
        if not r.popcount:
            if key not in tech_bus_limited or r.tok_s > tech_bus_limited[key].tok_s:
                tech_bus_limited[key] = r

    print(f"\n{'Technology':<22} {'Bus-limited tok/s':>18} {'With popcount tok/s':>20} "
          f"{'vs CPU (bus)':>12} {'vs CPU (pop)':>12} {'ODECC?':>10} {'PRAC?':>10}")
    print("-" * 110)

    for tech_name in ["DDR4-2400", "DDR5-4800", "DDR5-4800 (no PRAC)", "LPDDR5X-6400",
                       "LPDDR5X-6400 (16ch)", "HBM2-2000", "HBM3E-9600"]:
        best = tech_groups.get(tech_name)
        bus = tech_bus_limited.get(tech_name)
        if not best:
            continue

        bus_tok = bus.tok_s if bus else 0
        pop_tok = best.tok_s
        bus_cpu = bus.vs_cpu if bus else 0
        pop_cpu = best.vs_cpu

        odecc = "YES" if any(t.odecc_mandatory for t in all_timings if t.name == tech_name) else "No"
        prac = "YES" if any(t.prac_enabled for t in all_timings if t.name == tech_name) else "No"

        print(f"{tech_name:<22} {bus_tok:>18.1f} {pop_tok:>20.1f} "
              f"{bus_cpu:>11.2f}x {pop_cpu:>11.2f}x {odecc:>10} {prac:>10}")

    # =====================================================================
    # DDR5 PRAC impact analysis
    # =====================================================================
    print("\n")
    print("=" * 130)
    print("DDR5 PRAC IMPACT ANALYSIS")
    print("=" * 130)

    for act_bits in [8, 4]:
        for pop in [False, True]:
            r_prac = simulate_one_token(ddr5, model, act_bits=act_bits, use_channels=8,
                                        overlap_factor=0.5, in_dram_popcount=pop,
                                        label=f"DDR5 +PRAC, {act_bits}b, pop={pop}")
            r_noprac = simulate_one_token(ddr5_noprac, model, act_bits=act_bits, use_channels=8,
                                          overlap_factor=0.5, in_dram_popcount=pop,
                                          label=f"DDR5 -PRAC, {act_bits}b, pop={pop}")
            penalty = (1 - r_prac.tok_s / r_noprac.tok_s) * 100
            print(f"  {act_bits}-bit, popcount={'yes' if pop else 'no '}: "
                  f"with PRAC={r_prac.tok_s:>7.1f} tok/s | "
                  f"without PRAC={r_noprac.tok_s:>7.1f} tok/s | "
                  f"PRAC penalty={penalty:>5.1f}%")

    # =====================================================================
    # LPDDR5X vs DDR4 comparison (edge deployment story)
    # =====================================================================
    print("\n")
    print("=" * 130)
    print("LPDDR5X vs DDR4: EDGE DEPLOYMENT COMPARISON")
    print("=" * 130)

    configs = [
        ("DDR4 1D, INT8",         ddr4,  8, 1, 0.0, False),
        ("DDR4 4D, INT8",         ddr4,  8, 4, 0.0, False),
        ("LPDDR5X 8ch, INT8",     lpddr5x, 8, 8, 0.0, False),
        ("LPDDR5X 8ch, 4-bit",    lpddr5x, 4, 8, 0.0, False),
        ("LPDDR5X 8ch, 4b+ovlp",  lpddr5x, 4, 8, 0.5, False),
        ("LPDDR5X 16ch, INT8",    lpddr5x_16, 8, 16, 0.0, False),
        ("LPDDR5X 16ch, 4b+ovlp", lpddr5x_16, 4, 16, 0.5, False),
    ]

    print(f"\n{'Config':<30} {'tok/s':>7} {'TPOT(ms)':>9} {'vs CPU':>7} {'vs DDR4-1D':>11} Note")
    print("-" * 90)
    ddr4_baseline_toks = None
    for label, timing, ab, nch, ovlp, pop in configs:
        r = simulate_one_token(timing, model, act_bits=ab, use_channels=nch,
                               overlap_factor=ovlp, in_dram_popcount=pop)
        if ddr4_baseline_toks is None:
            ddr4_baseline_toks = r.tok_s
        vs_ddr4 = r.tok_s / ddr4_baseline_toks
        note = ""
        if "LPDDR5X" in label:
            note = "Single package" if "8ch" in label else "Dual-rank"
        print(f"{label:<30} {r.tok_s:>7.1f} {r.per_token_ms:>9.1f} {r.vs_cpu:>6.2f}x {vs_ddr4:>10.1f}x  {note}")

    # =====================================================================
    # Row packing comparison
    # =====================================================================
    print("\n")
    print("=" * 130)
    print("ROW PACKING COMPARISON (neurons per row)")
    print("=" * 130)

    print(f"\n{'Dimension':<12}", end="")
    for t in [ddr4, ddr5, lpddr5x, hbm2, hbm3e]:
        print(f" {t.name:>15}", end="")
    print()
    print("-" * 90)

    for dim_name, dim in [("d_model=2560", 2560), ("d_ff=6912", 6912), ("d_kv=640", 640)]:
        print(f"{dim_name:<12}", end="")
        for t in [ddr4, ddr5, lpddr5x, hbm2, hbm3e]:
            pack = max(1, (t.row_size_bytes * 8) // dim)
            print(f" {pack:>15}", end="")
        print()

    print("\n  Note: Fewer neurons/row means more AND cycles per matvec.")
    print("  DDR4/DDR5 (8KB rows) pack 4× more neurons than LPDDR5X/HBM (2KB rows).")
    print("  But LPDDR5X/HBM compensate with more independent channels.")

    # =====================================================================
    # Viability assessment
    # =====================================================================
    print("\n")
    print("=" * 130)
    print("VIABILITY ASSESSMENT")
    print("=" * 130)

    assessments = [
        ("DDR4-2400",   "VIABLE (proven)",
         "Charge-sharing AND demonstrated by SiMRA on 120 chips. No ODECC. No PRAC.\n"
         "  The reference platform. All paper results validated here."),
        ("DDR5-4800",   "CONDITIONAL",
         "ODECC is the primary blocker — silently corrupts AND results on readback.\n"
         "  Requires: MRS test-mode bypass OR BEER polynomial recovery.\n"
         "  PRAC adds ~10-18% throughput penalty (modeled above) and limits session length\n"
         "  (ALERTn triggers after ~250-1000 tokens per weight row at N_BO=4K-16K).\n"
         "  Voltage margin (1.1V) requires PUDTune-style calibration."),
        ("LPDDR5X",     "PROMISING",
         "Same 1T1C physics as DDR4. ODECC not always mandatory (vendor-dependent).\n"
         "  RowHammer mitigation is TRR (non-deterministic), not PRAC (no tRP penalty).\n"
         "  8-16 independent channels provide DDR4-4D equivalent parallelism in one package.\n"
         "  Lower voltage (1.05V) needs calibration. NOT YET TESTED with charge-sharing."),
        ("HBM2-2000",   "FEASIBLE (hard controller)",
         "Same DRAM physics. No mandatory ODECC. Optional ECC.\n"
         "  Primary obstacle: FPGA boards use hard IP controllers preventing raw commands.\n"
         "  DRAM Bender has demonstrated HBM2 command-level access (1.67ns granularity)."),
        ("HBM3E-9600",  "FUTURE",
         "Same DRAM physics. 16 channels per stack. Highest raw throughput.\n"
         "  Only available co-packaged on GPU interposers (H100, MI300X).\n"
         "  No independent controller access pathway exists today."),
    ]

    for tech, status, detail in assessments:
        print(f"\n  {tech}: [{status}]")
        print(f"  {detail}")

    # =====================================================================
    # Generate figure
    # =====================================================================
    print("\n\nGenerating cross-technology comparison figure...")

    # Collect data for figure
    fig_data = []
    fig_configs = [
        ("DDR4\n1D INT8",     ddr4,  8, 1, 0, False),
        ("DDR4\n4D 4b+pop",   ddr4,  4, 4, 0.5, True),
        ("DDR5+PRAC\n4D 4b+pop", ddr5, 4, 8, 0.5, True),
        ("DDR5 noPRAC\n4D 4b+pop", ddr5_noprac, 4, 8, 0.5, True),
        ("LPDDR5X\n8ch 4b+pop",  lpddr5x, 4, 8, 0.5, True),
        ("LPDDR5X\n16ch 4b+pop", lpddr5x_16, 4, 16, 0.5, True),
        ("HBM2\n8ch 4b+pop",  hbm2, 4, 8, 0.5, True),
        ("HBM2\n2stk 4b+pop", hbm2, 4, 16, 0.5, True),
        ("HBM3E\n16ch 4b+pop", hbm3e, 4, 16, 0.5, True),
        ("HBM3E\n2stk 4b+pop", hbm3e, 4, 32, 0.5, True),
    ]

    labels = []
    tok_s_vals = []
    colors = []
    color_map = {
        "DDR4": "#3498db",
        "DDR5": "#e67e22",
        "LPDDR5X": "#2ecc71",
        "HBM2": "#9b59b6",
        "HBM3E": "#e74c3c",
    }

    for lbl, timing, ab, nch, ovlp, pop in fig_configs:
        r = simulate_one_token(timing, model, act_bits=ab, use_channels=nch,
                               overlap_factor=ovlp, in_dram_popcount=pop)
        labels.append(lbl)
        tok_s_vals.append(r.tok_s)
        for key in color_map:
            if key in timing.name:
                colors.append(color_map[key])
                break

    fig, ax = plt.subplots(figsize=(16, 7))
    bars = ax.bar(range(len(labels)), tok_s_vals, color=colors, edgecolor='black', linewidth=0.5)

    # Value labels
    for bar, val in zip(bars, tok_s_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # CPU reference line
    ax.axhline(y=5.9, color='red', linestyle='--', linewidth=2, label='CPU baseline (5.9 tok/s, 1-thread)')
    ax.axhline(y=25, color='red', linestyle=':', linewidth=1.5, alpha=0.5, label='CPU multi-thread est. (~25 tok/s)')

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8, ha='center')
    ax.set_ylabel('Throughput (tok/s)', fontsize=12)
    ax.set_title('PIM-LLM Cross-Technology Throughput (BitNet 2B4T, best configs with 4-bit acts + popcount)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(tok_s_vals) * 1.15)

    # Add technology group labels
    tech_spans = [
        (0, 1, "DDR4", "#3498db"),
        (2, 3, "DDR5", "#e67e22"),
        (4, 5, "LPDDR5X", "#2ecc71"),
        (6, 7, "HBM2", "#9b59b6"),
        (8, 9, "HBM3E", "#e74c3c"),
    ]
    for start, end, label, color in tech_spans:
        mid = (start + end) / 2
        ax.annotate(label, xy=(mid, 0), xytext=(mid, -max(tok_s_vals)*0.12),
                    ha='center', fontsize=10, fontweight='bold', color=color,
                    annotation_clip=False)

    plt.tight_layout()
    fig_path = "pim_throughput_all_technologies.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved: {fig_path}")
    plt.close()

    print("\n" + "=" * 130)
    print("DONE")
    print("=" * 130)


if __name__ == "__main__":
    main()
