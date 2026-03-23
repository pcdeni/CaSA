# DRAM PIM Simulation Plan v2

All simulations use the correct MAJ3 protocol for AND.
Both discrete (BSS138/BSS84, 5V, 100nF/1µF) and Sky130 (130nm CMOS, 1.8V, 30fF/300fF).

## Sim 1: Single Cell Read
Baseline: one cell shares charge onto precharged bitline.
Measure: signal development (mV above Vref), settling time.

## Sim 2: Sense Amp Resolution
Cross-coupled latch resolves cell signal.
Measure: resolution time, final voltages, minimum resolvable signal.

## Sim 3: MAJ3 AND — Full Truth Table
Three cells (A, B, R=0) activated simultaneously. SA fires on combined charge.
Measure: BL voltage for all 4 input combos, signal margin, cell final states.
Expected: AND(A,B) = MAJ(A, B, 0).

## Sim 4: MAJ3 AND — Timing Sweep
Sweep delay between row activations (0 to 5ns stagger).
Verify AND works with simultaneous and staggered activation.

## Sim 5: Popcount Staircase
N cells (0 to 8) sharing simultaneously. Measure BL voltage vs count.
Verify linearity and step size.

## Sim 6: RowCopy Monte Carlo
Two cells sharing, SA resolves. Sweep Vth mismatch.
Find BER vs mismatch curve. Map to SiMRA's 16.3% BER.

## Sim 7: Weight Reload Timing
After MAJ3 destroys the weight, how long to WRITE it back?
Measure: bus write time for one row.

## Sim 8: Full CaSA Cycle
Complete cycle: write activation + write zeros + MAJ3 + read result + reload weight.
Measure: total cycle time. Compare to paper's estimates.
