# Discrete DRAM PIM Simulation Results

**Date:** 2026-03-23
**Simulator:** LTspice 26.0.1
**Components:** BSS138 (NFET), BSS84 (PFET), discrete capacitors
**Parameters:** Vdd=5V, Vref=2.5V, Ccell=100nF, Cbl=1uF (10:1 ratio), WL=7V (boosted)

---

## Sim 1: Single Cell Read (sim1_cell.cir)

Single cell storing '1' (5V) read onto precharged bitline (2.5V).

| Measurement       | Value      |
|-------------------|------------|
| BL before WL open | 2.671 V    |
| BL after sharing  | 2.727 V    |
| Cell after sharing| 2.727 V    |
| **Signal (BL - Vref)** | **+227.3 mV** |
| BL settled (45us) | 2.727 V    |

**Note:** BL starts at 2.671V (not 2.5V) because the BSS138 access transistor has slight subthreshold leakage during the initial condition settling. The charge-sharing result matches the analytical prediction: deltaV = Ccell/(Ccell+Cbl) * (Vcell - Vref) = 100n/1.1u * 2.5 = 227.3mV.

---

## Sim 2: Sense Amplifier (sim2_sa.cir)

Cross-coupled CMOS latch amplifying charge-sharing signal from a '1' cell.

| Measurement            | Value      |
|------------------------|------------|
| BL after charge share  | 2.727 V    |
| /BL (reference)        | 2.500 V    |
| Differential input     | +227.3 mV  |
| **BL final (after SA)** | **4.989 V** |
| **/BL final**          | **0.007 V** |
| Cell restored to       | 4.988 V    |
| SA resolution time     | 9.64 us    |

**Analysis:** The sense amplifier correctly resolves 227mV differential to full-swing rails (0V to 5V). Resolution time of ~10us is consistent with the large parasitic capacitances of discrete components. The cell is fully restored to ~5V after sensing.

---

## Sim 3: MAJ3 AND Truth Table (sim3_maj3_and.cir)

Three cells on BL (A, B, R=0) implementing AND(A,B) = MAJ3(A, B, 0).

### Charge-Sharing Phase (before SA)

| A | B | R | BL Voltage | /BL (Ref) | Differential | Expected |
|---|---|---|------------|-----------|-------------|----------|
| 0 | 0 | 0 | 1.923 V    | 2.500 V   | -577 mV     | Negative |
| 5 | 0 | 0 | 2.308 V    | 2.500 V   | -192 mV     | Negative |
| 0 | 5 | 0 | 2.308 V    | 2.500 V   | -192 mV     | Negative |
| 5 | 5 | 0 | 2.692 V    | 2.500 V   | +192 mV     | Positive |

### After Sense Amplification

| A | B | AND(A,B) | BL Final | /BL Final | Cell B Final | Result |
|---|---|----------|----------|-----------|-------------|--------|
| 0 | 0 | **0**    | 0.017 V  | 4.995 V   | 0.017 V     | 0      |
| 1 | 0 | **0**    | 0.028 V  | 4.992 V   | 0.029 V     | 0      |
| 0 | 1 | **0**    | 0.028 V  | 4.992 V   | 0.029 V     | 0      |
| 1 | 1 | **1**    | 4.971 V  | 0.008 V   | 4.968 V     | 5      |

**Analysis:** All four input combinations produce the correct AND result. The MAJ3 gate works because with R=0, at least 2 of 3 cells must be '1' for BL to exceed Vref -- which only happens when both A=1 and B=1.

The charge-sharing formula: BL = (3*Ccell*Vavg_cells + Cbl*Vref) / (3*Ccell + Cbl) where:
- 3 cells of 100nF sharing with 1uF bitline
- Each cell contributes Ccell/(3*Ccell + Cbl) = 100n/1.3u = 76.9mV per '1' cell
- Two '1' cells: +153.8mV above the zero-cells baseline

**Cell B is overwritten** with the AND result (Cell B Final shows the SA-driven value), which is the standard destructive-read writeback behavior of DRAM sense amps.

---

## Sim 4: Popcount (sim4_popcount.cir)

8 cells on one bitline, BL precharged to 0V, sweep number of '1' cells.

| # of '1' Cells | BL Voltage (V) | Step Size (mV) |
|-----------------|-----------------|-----------------|
| 0               | 0.000           | --              |
| 1               | 0.278           | 277.8           |
| 2               | 0.556           | 277.8           |
| 3               | 0.833           | 277.5           |
| 4               | 1.111           | 277.8           |
| 5               | 1.389           | 277.8           |
| 6               | 1.667           | 277.8           |
| 7               | 1.944           | 277.8           |
| 8               | 2.222           | 277.8           |

**Step size:** 277.8 mV per '1' cell (constant)
**Linearity:** Essentially perfect (<0.1% deviation)
**Full range:** 0V to 2.222V across 9 levels
**Theoretical:** deltaV = Ccell*Vdd / (8*Ccell + Cbl) = 100n*5 / 1.8u = 277.8mV (exact match)

**Analysis:** The popcount produces perfectly linear voltage levels. With an 8-bit ADC (19.5mV resolution on 0-5V range), each step of 277.8mV provides ~14x margin over a single LSB. Even a 4-bit ADC (312.5mV steps) could distinguish adjacent counts. This validates the multi-cell charge-sharing principle for analog popcount computation.

---

## Summary

| Simulation | Key Result | Status |
|------------|-----------|--------|
| Single Cell Read | 227.3mV signal, matches theory | PASS |
| Sense Amplifier | Full-swing resolution in 9.6us, cell restored | PASS |
| MAJ3 AND Gate | 4/4 truth table entries correct | PASS |
| Popcount (8-bit) | 277.8mV/step, perfect linearity | PASS |

All four PIM primitives validated with discrete BSS138/BSS84 components at 5V/100nF scale.
