# CaSA: Current Understanding (as of March 23, 2026)

## What Works — Verified in SPICE (Sky130 + discrete)

### 1. MAJ3 AND: AND(A,B) = MAJ(A, B, 0)
- Three rows activated simultaneously before SA fires
- All 4 truth table entries correct
- Signal margins: ±69mV (Sky130), ±192mV (discrete)
- Requires: 3 rows in same subarray (weight, activation, zero-reference)
- Consequence: ALL three rows are overwritten with the majority result

### 2. 2-Row RowCopy: COPY(A → B)
- Activate Row A → SA latches → Activate Row B → Row B overwritten with A's value
- All 4 copy directions verified (1→0, 0→1, 1→1, 0→0)
- Source row (A) is PRESERVED
- Zero BER in simulation (SA-mediated, not charge-sharing-based)

### 3. Popcount via multi-row activation
- N cells sharing simultaneously → BL voltage proportional to count of '1's
- 98.8mV/step at Sky130, 277.8mV/step discrete
- Linear to within 0.2mV across 9 levels
- Validates analog popcount concept

## What Does NOT Work

### 2-Row AND
- 2-row activation produces COPY, not AND
- Mathematically impossible with symmetric precharge
- The CaSA paper's claim of 2-row AND is WRONG
- SiMRA's 2-row "FindOpenRows" tests row accessibility, not AND computation

## Corrected CaSA Protocol

### Per AND operation:
1. Write activation bit-plane to DRAM row (bus transfer)
2. RowCopy: copy zeros from persistent zero row to reference row (~40ns DRAM)
3. MAJ3: activate weight + activation + reference simultaneously (~15ns DRAM)
4. Read AND result row (bus transfer)
5. RowCopy: copy weight from backup row to working row (~40ns DRAM)

### Timing per weight row (pipelined across 4+ banks):
- BUS: read result [96ns, uses bus — BOTTLENECK]
- DRAM: RowCopy zeros + MAJ3 + RowCopy weight [228ns, NO bus — HIDDEN by pipelining]
- Effective per row: 96ns (bus-bound)

### Timing per weight row (no pipelining, single bank):
- DRAM: RowCopy zeros + MAJ3 + RowCopy weight [228ns — BOTTLENECK]
- BUS: read result [96ns — hidden behind DRAM]
- Effective per row: 228ns (DRAM-bound)

### Key insight: RowCopy doesn't use the bus
- RowCopy is DRAM-internal (SA-mediated 2-row activation)
- With multi-bank pipelining, RowCopy runs in parallel with bus transfers
- At 4+ active banks, RowCopy overhead is COMPLETELY HIDDEN
- Pipelined throughput is actually BETTER than original paper predicted

## Row Budget per Neuron Group

Old (paper): 1 weight row + 1 scratch row = 2 rows
New (corrected): 1 weight row + 1 weight backup + 1 activation scratch +
                 1 zero-reference + 1 zero-reference backup = 5 rows

This 2.5x increase in row usage reduces the number of neuron groups per bank.
With 65,536 rows per bank and ~2048 weight rows per layer, there's still
ample space (even at 5 rows per weight row = 10,240 rows, leaving 55K rows free).

## Key Insight: 2-Row "Failure" Enables RowCopy

The same property that prevents 2-row AND (SA overwrites Row B with Row A's value)
is exactly what makes RowCopy work reliably. This turns the protocol correction
from a ~2x throughput penalty into a ~3% penalty by replacing bus-based reloads
with in-DRAM RowCopy.

## What Still Needs Work

1. **Throughput simulator update** — recalculate tok/s with corrected protocol
2. **Paper update** — correct 2-row AND claim, add MAJ3 + RowCopy protocol
3. **Multi-bank pipelining analysis** — quantify overlap of bus and DRAM ops
4. **FPGA prototype** — test MAJ3 on real Micron chips (KRC-4700)
5. **SiMRA data reanalysis** — cite MajOperations data for AND, not FindOpenRows
