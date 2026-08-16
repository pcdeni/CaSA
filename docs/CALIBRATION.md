# Calibrating a DIMM

A DIMM you intend to compute on needs three things characterized: **MAJ3
timings**, **broadcast timings** and **RowClone timings**. A calibration
belongs to the die it was measured on. Different silicon needs its own —
though far less of it than the first one cost, which is what
`docs/CALIBRATION_TRANSFER.md` is about.

## What ships here

`calibration/calib_dimm2.txt` is the live calibration: 384 MAJ3-perfect
tuples across banks 0–3 of subarrays 72, 78 and 86, on the SK hynix part
described in `docs/HARDWARE.md`. With
`calibration/pool_layout_dimm2_cloneok_bank{0-3}.txt` and the row window
`[45312, 45952)` it forms the *trio* every tool resolves through
`calibration/DIMM_POPULATION.conf`.

`calibration/calib_dimm0.txt` and the `pool_layout_dimm0_*`,
`fused_colmask_dimm0_*` files are the characterization record of a module
that is no longer installed — a different part number. They are kept
because they are data, and `python/dimm_population.py` refuses to serve
them as a live default: pointing one die at another die's calibration does
not fail loudly, it computes a fraction of cells wrong.

## What you are looking for

Three charge-sharing operations are used by the PIM apps:

| Operation | Primitive | Timing on our parts |
|---|---|---|
| MAJ3 (the gate itself) | `doubleACT(t_12, t_23)` over 16 open rows | `t_12 = 0, t_23 = 0` |
| Broadcast (Multi-Row-Init): copy `Rfirst` into all 16 open rows | `doubleACT(t_12, t_23)`, different rows | `t_12 = 10, t_23 = 2` |
| RowClone: charge-shared 2-row copy | `doubleACT(t_12, t_23, src, dst)` | `t_12 = 30, t_23 = 1` |

A *calibrated tuple* for MAJ3 is an `(s_id, bank, Rfirst, Rsecond,
[r0..r15])` configuration where `doubleACT(0,0,Rfirst,Rsecond)` over the 16
open rows produces the correct majority for every input pattern you test,
on every cell.

## Protocol

### 1. Find candidate tuples

Run DRAM-Bender's `FindOpenRows` against the DIMM. The exact invocation
depends on your bitstream and channel mapping:

```bash
cd DRAM-Bender/sources/apps/DSN_AE_APPS/FindOpenRows
./find-open-rows-exe <bender_id> <selected_subarrays.txt> <temp> <output_dir>
```

It sweeps `(t_12, t_23) ∈ {0..3}^2 × subarray × row-pair` and produces a CSV
of every combination that activates cleanly. Budget 1–2 hours per
(subarray, timing) pair.

### 2. Score MAJ3 stability across input patterns

Run DRAM-Bender's `MajOperations` over the step-1 candidates. It drives a
wide set of input patterns through each candidate and counts:

- `full_stable_cells` — cells that always output the correct majority on
  every tested pattern;
- `full_coverage_cells` — cells that flipped at least once, but always in
  the direction the majority computes.

We keep tuples with `majX == 3, t_12 == 0, t_23 == 0,
full_stable_cells == 100, full_coverage_cells == 100`.

### 3. Verify broadcast and RowClone independently

The `app/` smoke tests check these without going through MAJ3:

```bash
# Per-bank RowClone reliability sweep:
for bk in 0 1 2 3; do
  rfirst=$(awk -v b=$bk '$1!~/^#/ && $2==b {print $3; exit}' calib_dimmN.txt)
  ./rowclone-smoke-exe <bender_id> $bk <some_backup_row> $rfirst
done
# Expect "match=8192/8192 (100%) PERFECT_CLONE" for every (bank, t_23) pair.

# End-to-end persistent weights (RowClone + broadcast + MAJ3 in one program):
./persistent-smoke-exe <bender_id> calib_dimmN.txt <bank> <s_id> <backup_row>
# Expect 8192/8192 byte-exact between the persistent path (per-column write
# to backup once, then RowClone-refresh) and the direct per-column path.
```

If the smoke tests pass at the same `(t_12, t_23)` defaults as ours, use
them. If not, **the defaults are not transferable** — sweep `t_23` in
`rowclone-smoke-exe` (it already sweeps {1, 2, 3, 4}) for the working value
and adjust the `doubleACT(t_12, t_23, …)` calls in
`test_bitnet_server.cpp::emit_bank_combined_body` to match.

### 4. Format the calib file

See `calibration/README.md` for the format. The C++ apps read it and pick
one tuple per `(bank, …)` they are asked to use.

### 5. Record the population

Put the new module and its trio in `calibration/DIMM_POPULATION.conf` and
check it with `python3 python/dimm_population.py`. That file is the only
place a fixture name belongs; the tools take it from there.

## What can go wrong

- **Zero yield.** Some samples give zero perfect 16-open tuples — we have
  seen it on Crucial Ballistix. The chip may still be usable for smaller
  configurations (8 open rows, fewer-input MAJ3) but not for the full
  BitNet configuration. Yield is a property of the part: we have measured
  anywhere from 0 % to ~38 % candidate rate.
- **Run-to-run drift on marginal tuples.** A tuple that scored 100 % last
  week may see a handful of cells out of ~22,000 flip differently this week
  on *uncalibrated* input patterns. Scoring is robust to this inside the
  calibrated input set; outside it, expect ~0.02 % per-run variance.
  Ternary models absorb it. This is also why the numerics gate for a
  calibration change is a correlation threshold and not bit-exactness —
  raw per-operation output is not bit-stable across processes, while
  correlation is, and it collapses immediately on a genuinely wrong
  weight, mask or pool.
- **Per-bank divergence.** Different banks are different physical cells.
  Verify each bank you intend to use — but see
  `docs/CALIBRATION_TRANSFER.md` first: on a characterized die that is a
  margin re-screen, not a re-sweep.
- **A clean screen is not a certified channel.** A channel can pass
  RowClone, byte-lane and read/write screens and the numerics oracle, and
  then latch a byte lane under sustained model traffic. Re-run the lane and
  clone checks *after* traffic — `docs/HARDWARE.md` has the order.

## Time budget

The first subarray of a new *part* is the expensive one: sweep → score →
broadcast verify → RowClone verify was roughly 30–40 hours of FPGA
wall-time for us. Nothing after that costs anything like it. Adopting
another bank, or another die of the same part, is a copy plus a margin
re-screen; adopting another subarray of a characterized die is minutes.
`docs/CALIBRATION_TRANSFER.md` is the recipe, with the measurements behind
each step.
