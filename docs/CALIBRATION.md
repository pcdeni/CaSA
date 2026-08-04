# Calibrating a new DIMM

Each new DIMM you intend to use needs three things characterized:
**MAJ3 timings**, **broadcast timings**, and **RowClone timings**. The
shipped `calibration/calib_dimm0.txt` only fits our reference DIMM —
different silicon batches need their own calibration.

## What you are looking for

Three charge-sharing operations are used by the BitNet PIM apps:

| Operation | Primitive | Default timing on DIMM 0 |
|---|---|---|
| MAJ3 (the actual gate) | `doubleACT(t_12, t_23)` over 16 open rows | `t_12 = 0, t_23 = 0` |
| Broadcast (Multi-Row-Init): copy `Rfirst` to all 16 open rows | `doubleACT(t_12, t_23)` (different rows) | `t_12 = 10, t_23 = 2` |
| RowClone: charge-share 2-row copy | `doubleACT(t_12, t_23, src, dst)` | `t_12 = 30, t_23 = 1` |

A "calibrated tuple" for MAJ3 is a `(s_id, bank, Rfirst, Rsecond,
[r0..r15])` configuration where `doubleACT(0,0,Rfirst,Rsecond)` over
the 16 open rows produces the correct majority output for every
input pattern your characterization tests, on every cell.

## Protocol

### 1. Find candidate tuples (MAJ3 region)

Run DRAM-Bender's `FindOpenRows` application against the DIMM. The
exact invocation depends on your bitstream and the DIMM channel
mapping:

```bash
cd DRAM-Bender/sources/apps/DSN_AE_APPS/FindOpenRows
./find-open-rows-exe <bender_id> <selected_subarrays.txt> <temp> <output_dir>
```

This sweeps `(t_12, t_23) ∈ {0..3}^2 × subarray × row-pair` and
produces a CSV listing every combination that activates cleanly. At
~1-2 hours per (subarray, timing) pair, expect 1-2 days per DIMM for
3-4 subarrays.

### 2. Score MAJ3 stability across input patterns

Run DRAM-Bender's `MajOperations` over the candidate tuples from
step 1. This drives a wide set of input patterns through each
candidate and counts:

- `full_stable_cells` (% of cells that always output the correct
  majority on every tested pattern)
- `full_coverage_cells` (% of cells that flipped at least once but
  always within the same direction the majority computes)

For our reference DIMM we kept tuples with **`majX == 3, t_12 == 0,
t_23 == 0, full_stable_cells == 100, full_coverage_cells == 100`**.
That gave 312 perfect tuples across 4 banks of subarray 61.

### 3. Verify broadcast and RowClone independently

The `app/` C++ smoke tests are designed to verify these
independently of MAJ3:

```bash
# Per-bank RowClone reliability sweep:
for bk in 0 1 2 3; do
  rfirst=$(awk -v b=$bk '$1!~/^#/ && $2==b {print $3; exit}' calib_dimmN.txt)
  ./rowclone-smoke-exe <bender_id> $bk <some_backup_row> $rfirst
done
# Expect "match=8192/8192 (100%) PERFECT_CLONE" for every (bank, t_23) pair
# at t_23 = 1.

# End-to-end persistent weights (uses RowClone + broadcast + MAJ3
# in one combined program):
./persistent-smoke-exe <bender_id> calib_dimmN.txt <bank> <s_id> <backup_row>
# Expect 8192/8192 byte-exact match between the persistent path
# (per-col write to backup ONCE, then RowClone-refresh) and the
# direct per-col write path.
```

If the smoke tests pass with the same `(t_12, t_23)` defaults as
DIMM 0, you can use those defaults in production. If not, **the
defaults are not transferable** — sweep `t_23` in
`rowclone-smoke-exe` (already sweeps {1, 2, 3, 4}) to find the
working value, and adjust the `doubleACT(t_12, t_23, …)` calls in
`test_bitnet_server.cpp::emit_bank_combined_body` accordingly.

### 4. Format the calib file

See `calibration/README.md` for the exact format. The downstream C++
apps read this file and pick one tuple per `(bank, …)` they're
asked to use.

## What can go wrong

- **0 % yield on a DIMM.** Some samples give us zero perfect
  16-open tuples (notably some Crucial Ballistix). The chip may
  still be usable for smaller configurations (8 open rows,
  fewer-bit MAJ3, etc.) — but not for the full BitNet
  configuration. Move to a different DIMM.
- **Run-to-run drift on marginal tuples.** A tuple that scored
  100 % last week may have ~5/22144 cells flip differently this
  week on uncalibrated input bit-patterns. The scoring is robust
  to this within the calibrated input set; outside it, expect
  small per-run variance (~0.02 % of outputs). Ternary models
  absorb it.
- **Per-bank divergence.** Different banks use different physical
  cells with different characteristics. Verify each bank you
  intend to use independently.

## Time budget

For our reference DIMM, the full pipeline (sweep → score →
broadcast verify → RowClone verify) took roughly 30-40 hours of
FPGA wall-time. The other three DIMMs have since finished (May
2026): the timing constants generalized on the two full-PUD
modules; two partial modules turned out MAJ3-limited entirely
(zero separated-generator tuples — a part/binning outcome, not a
calibration failure).

## Calibration transfer (do NOT re-run the 30-40 h pipeline per bank)

What we have measured about how characterization generalizes
(the "replicated blocks" finding + the cross-die results):

- **Across banks of one die**: the co-activation spread profile is
  byte-identical on every bank measured, and the predecoder selection
  law is a design constant. Per-bank differences are *margin* only
  (which columns are strong). Transfer = copy the source bank's calib
  (tuple rows, t_12/t_23, open-row set) verbatim, then re-run ONLY the
  column margin screen on the target bank (minutes-to-hours), then the
  standard RowClone/broadcast/MAJ3 smoke. Our production die runs three
  banks on one calib this way.
- **Across dies of the same part**: fault sets and calibration
  transferred byte-identically between our two same-model modules —
  same recipe as above, margin re-screen per die.
- **Across different parts**: the spread lattice is chip-specific (our
  two part types couple different XOR offsets — one includes ⊕256, one
  does not). The *method* transfers; the lattice does not. Re-derive
  the fault-sweep/lattice first — pool layouts depend on it — then
  calibrate timings (those have been the stable part in our data).

The 16-bank audit on the roadmap (`ROADMAP.md` §C) will turn this into
a quantified transfer-success table.
