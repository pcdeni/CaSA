# CaSA SPICE Verification Models

Transistor-level simulations verifying the Charge-Sharing Accumulate (CaSA) primitives.
Two implementations: discrete components (bench-testable) and Sky130 130nm PDK (silicon-realistic).

## Directory Structure

```
spice/
  discrete/          Discrete-component models (BSS138/BSS84, 5V, 100nF cells)
  sky130/            Sky130 PDK models (130nm, 1.8V, 30fF cells)
  CORRECTION_MAJ3.md Why 2-row activation is COPY, not AND (and the fix)
  CURRENT_UNDERSTANDING.md  Summary of proven CaSA primitives
  SIMULATION_PLAN.md Simulation methodology and progression
```

## Simulation Files

Each simulation exists in both `discrete/` and `sky130/` unless noted.

| File | Description |
|------|-------------|
| `sim1_cell.cir` | Single DRAM cell charge-sharing read. Validates deltaV = Ccell/(Ccell+Cbl) * (Vcell - Vref). |
| `sim2_sa.cir` | Cross-coupled CMOS sense amplifier with full subarray model (precharge, isolation, SA enable sequencing). |
| `sim3_maj3_and.cir` | MAJ3 AND operation: all 4 truth table entries verified (00->0, 01->0, 10->0, 11->1) using three-row simultaneous activation with one row as constant-0 anchor. |
| `sim4_popcount.cir` | 8-cell popcount accumulation. Produces linear staircase voltage (98.8 mV/step on Sky130). |
| `sim5_rowcopy.cir` | SA-mediated RowCopy: 0% BER across all 4 directions (0->0, 0->1, 1->0, 1->1). Sky130 only. |
| `sim5_copy_*.cir` | Individual RowCopy direction tests (sky130 only): `sim5_copy_0to0.cir`, `sim5_copy_0to1.cir`, `sim5_copy_1to0.cir`, `sim5_copy_1to1.cir`. |
| `sim6_full_cycle.cir` | Full CaSA inference cycle: MAJ3 AND + RowCopy weight reload. Proves the complete primitive sequence. Sky130 only. |

## Results

Detailed measurement tables are in:
- `discrete/discrete_results.md`
- `sky130/sky130_results.md`

## Sky130 PDK Setup

The Sky130 simulations require the SkyWater PDK transistor models. These are NOT included in this repo (they are large and maintained separately).

To set up:

```bash
# Clone the PDK models into the sky130/ directory
cd spice/sky130/
git clone https://github.com/google/skywater-pdk-libs-sky130_fd_pr.git sky130_fd_pr
```

The `sky130_minimal.lib` wrapper file included here points LTspice to the correct model files within `sky130_fd_pr/`.

## Simulator

All simulations were run with **LTspice 26.0.1** on Windows. They should work with any SPICE-compatible simulator (ngspice, etc.) with minor syntax adjustments.

## Key Findings

1. **MAJ3 AND works** -- requires 3-row activation (two data rows + one constant-0 anchor), not 2-row
2. **2-row activation = COPY, not AND** -- this is a common misconception; see `CORRECTION_MAJ3.md`
3. **SA-mediated RowCopy = 0% BER** -- the sense amplifier restores full logic levels before writeback
4. **Popcount is linear** -- staircase voltage scales linearly with hamming weight (R^2 > 0.999)
5. **Full CaSA cycle verified** -- MAJ3 followed by RowCopy weight reload completes without data corruption
