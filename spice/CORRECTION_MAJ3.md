# Critical Correction: CaSA Requires MAJ3, Not 2-Row AND

Date: 2026-03-23

## The Error

The CaSA paper claims that 2-row charge-sharing activation produces Boolean AND,
citing SiMRA's 79M-trial dataset as evidence. Both claims are wrong:

1. **SiMRA's 2-row test (FindOpenRows) does NOT test Boolean AND.** It tests
   whether two rows can be simultaneously activated and subsequently written/read.
   The 100% success rate at t_12≥1 means "both rows opened" — not "AND was computed."

2. **SiMRA's actual AND/logic tests use MAJ3+ (3 or more rows).** The MajOperations
   test suite computes expected majority in software and compares to DRAM output.
   AND(A,B) is implemented as MAJ3(A, B, 0) — three-row majority with a zero reference.

3. **2-row AND is mathematically impossible** with symmetric precharge. Proven by
   SPICE simulation (both discrete BSS138 and Sky130 130nm) and confirmed by
   algebraic analysis of the precharge timing constraints.

## Proof: 2-Row AND is Impossible

For the precharge-based 2-row protocol (ACT-PRE-ACT with violated tRP):

After Row A is sensed and precharge begins, BL decays exponentially toward Vref:
- From A=1 (BL at Vdd): V(t) = Vref + (Vdd-Vref) × exp(-t/τ)
- From A=0 (BL at 0V):  V(t) = Vref - Vref × exp(-t/τ)

For AND(1,0)=0: need V_final < Vref after Cell B=0 shares.
  Requires precharge time t > 2.3τ (BL must be close to Vref).

For AND(0,1)=0: need V_final < Vref after Cell B=1 shares.
  Requires precharge time t < 2.3τ (BL must still be far below Vref).

These constraints are contradictory. No single precharge time satisfies both.

For the SA-mediated 2-row protocol (ACT Row A, SA fires, ACT Row B):
The sense amp drives BL to full rail and overwrites Cell B regardless of B's value.
Result: COPY(A→B), not AND(A,B). Confirmed in simulation at all timing delays.

## Proof: MAJ3 AND Works

Three cells (A, B, R=0) sharing onto one bitline before SA fires:

V_bl = (Vref×Cbl + VA×Cc + VB×Cc + 0×Cc) / (Cbl + 3×Cc)

| A | B | R | V_bl (mV) | vs Vref | SA reads | AND? |
|---|---|---|-----------|---------|----------|------|
| 1 | 1 | 0 | 969       | +69     | 1        | ✓    |
| 1 | 0 | 0 | 831       | -69     | 0        | ✓    |
| 0 | 1 | 0 | 831       | -69     | 0        | ✓    |
| 0 | 0 | 0 | 692       | -208    | 0        | ✓    |

Verified in SPICE (Sky130, realistic subarray with ISO transistors, bitline RC,
wordline RC, SA enable RC). Signal margins: ±69mV for the critical cases,
±208mV for the easy case. All four cases correct.

## Impact on CaSA Architecture

### What Changes

1. **AND requires 3 rows, not 2.** Each AND operation needs: weight row + activation
   row + zero-reference row, all in the same subarray.

2. **Weight is NOT preserved.** In MAJ3, all three rows get overwritten with the
   majority result. The weight row is destroyed after each AND operation.

3. **Weight backup needed.** Each weight row must have a backup copy in another
   subarray (or in a reserved area). After each AND, the weight must be reloaded.
   This can be done via normal WRITE (from bus) or via RowCopy (if reliable enough).

4. **Zero-reference row must be reinitialized.** The reference row (all-0s) is also
   destroyed by each AND. It must be rewritten before the next operation.

### What Doesn't Change

1. **The dot product computation is still valid.** AND(W, x) via MAJ3 produces the
   correct per-column result. Popcount of the AND row gives the dot product term.

2. **The charge-sharing physics is still valid.** Multi-row activation works as
   characterized by SiMRA. The BER data for multi-row operations applies.

3. **The throughput model structure is still valid.** The bus bottleneck analysis,
   bank pipelining, and popcount optimization all still apply — just with different
   cycle counts per operation.

### Throughput Impact

The corrected protocol per bit-plane:

Old (2-row, WRONG):
  1. WRITE activation to scratch row [460ns]
  2. doubleACT weight + scratch [62ns]
  3. READ result [459ns]
  Total: 981ns per half-cycle

Corrected (3-row MAJ3):
  1. WRITE activation to row [460ns]
  2. WRITE zeros to reference row [460ns]  ← NEW
  3. tripleACT weight + activation + reference [~65ns]
  4. READ result [459ns]
  5. WRITE weight back from backup [460ns]  ← NEW (weight destroyed)
  Total: ~1904ns per half-cycle

Approximate throughput hit: ~2x slower than the paper claims.

However, optimizations exist:
- Keep a permanent bank of zero-rows; only reinitialize periodically
- Pipeline weight reload with result readout (different banks)
- Use RowCopy for weight reload if BER is acceptable for weights
  (weights are static, can use ECC/voting across multiple copies)

## SiMRA Data Reinterpretation

| SiMRA Test | What It Actually Measures | CaSA Relevance |
|------------|-------------------------|----------------|
| FindOpenRows (2-row) | Row accessibility after double activation | Confirms timing-violated activation works |
| MajOperations (3-row) | MAJ3 correctness | **THIS is the AND test CaSA needs** |
| MajOperations (5,7,9-row) | Higher-order majority | Useful for multi-bit popcount |
| MultiRowCopy | RowCopy reliability | Weight backup/reload path |

The MajOperations data should be cited for AND reliability, not FindOpenRows.

## Files

### Simulations proving the finding:
- `spice/sky130/sky130_casa_and.cir` — 2-row AND truth table (fails AND(1,0))
- `spice/sky130/sky130_casa_timing_sweep.cir` — timing sweep (fails at all delays)
- `spice/sky130/sky130_pure_charge_share.cir` — simultaneous 2-row (produces MAJORITY)
- `spice/sky130/sky130_casa_simra_protocol.cir` — ACT-PRE-ACT protocol (partial success)
- `spice/sky130/sky130_and_debug.cir` — detailed node tracing of SA behavior
- `spice/sky130/sky130_maj3_and.cir` — **MAJ3 AND: all 4 cases correct**

### SiMRA source code confirming test methodology:
- `dsn_artifact/.../FindOpenRows/test_find_open_rows.cpp` — tests row accessibility, NOT AND
- `dsn_artifact/.../MajOperations/test.cpp` — tests majority (actual AND computation)
