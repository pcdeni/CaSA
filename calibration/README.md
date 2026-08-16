# `calibration/` — pre-characterized MAJ3-perfect open-row tuples

Each line of a `calib_*.txt` is one calibrated tuple on one die. Tuples
are the unit of work for the BitNet PIM applications: every MAJ3 reads
from one calibrated tuple's 16 open rows.

Which file is live is not decided here. `DIMM_POPULATION.conf` describes
the installed modules and names the trio (calibration + pool layout + row
window) that serves them, and `python/dimm_population.py` resolves it. Run
`python3 python/dimm_population.py` to see what it picks.

## File format

```
# Comments start with #. Blank lines ignored.
# Format: s_id bank Rfirst Rsecond r0 r1 r2 ... r15
61 0 38424 38988 38442 38446 38447 38450 ...
61 0 38424 38988 38442 38446 38447 38453 ...
...
```

Columns:

| Column | Meaning |
|---|---|
| `s_id` | Subarray id. All 16 open rows belong to the same subarray. |
| `bank` | DDR4 bank index (0..3 on a typical configuration). |
| `Rfirst`, `Rsecond` | The two rows used by the `doubleACT(0,0)` MAJ3 firing. |
| `r0 .. r15` | The 16 open rows that simultaneously hold weight / activation / zero / buffer copies. |

`calib_dimm2.txt` is the **live** calibration: 384 tuples across banks 0–3
of subarrays 72, 78 and 86, on the SK hynix part described in
`docs/HARDWARE.md`. Each tuple was verified at `t_12 == 0, t_23 == 0` and
passed `full_stable_cells == 100`, `full_coverage_cells == 100` on a
1000-pattern test. See `docs/CALIBRATION.md` for the full protocol.

`calib_dimm0.txt` (312 tuples, subarray 61) and the `pool_layout_dimm0_*`,
`fused_colmask_dimm0_*` and `colmasks_maj5_zero_*/maj5_best_b0_*` files
belong to a module of a **different part number** that is no longer
installed. They are kept because they are measurements. They are not a
live default, and `dimm_population.py` refuses to serve them as one
(`PIM_TRIO_ALLOW_RETIRED=1` to replay them deliberately) — a die driven
with another die's calibration does not fail loudly, it returns a fraction
of cells wrong.

## Producing `calib_dimmN.txt` for your own hardware

Different DRAM dies have different per-row-pair sweet spots, so
**you must characterize each chip you intend to use**. The protocol:

1. Run DRAM-Bender's `FindOpenRows` app on the target DIMM at a
   stable temperature. Output: a CSV of all (s_id, t_12, t_23,
   row-pair) combinations that activate cleanly.
2. Run `MajOperations` (also in DRAM-Bender's `DSN_AE_APPS/`) over
   the open-row candidates to score MAJ3 stability across input
   patterns.
3. Filter for `majX == 3, t_12 == 0, t_23 == 0,
   full_stable_cells == 100, full_coverage_cells == 100`. The
   surviving 16-open-row tuples are your calibrated set.
4. Format as above and ship.

Allow ~1-2 days of FPGA wall-time for the first subarray of a new part;
we have observed yields between 0 % (some Crucial Ballistix samples gave
nothing) and 38 % (some SK hynix samples are very productive). See
`docs/CALIBRATION.md` for what to do when yield is low — and
`docs/CALIBRATION_TRANSFER.md` before assuming the *next* die costs the
same, because it does not.

## A note on per-bank consistency

On our parts the `doubleACT` timings are uniform across banks: MAJ3 at
(0,0), broadcast at (10,2) and RowClone at (30,1) all give 100 %
byte-exact results regardless of bank, and the deposit lattice is
byte-identical across all 16. Across a *different* part this is open, and
different silicon may need per-bank timing characterization. Run
`rowclone-smoke-exe` and `persistent-smoke-exe` on each bank of a new part
before assuming the timings carry over.

## Pool families

`pool_layout_dimm{0,2}_bank*.txt` are the unfiltered pools. The screened
families below are what production actually uses:

| Family | Files | What it is |
|---|---|---|
| Clone-ok primary | `pool_layout_dimm{0,2}_cloneok_bank{0-3}.txt` | The production pool re-derived through a **double filter**: (1) the *clone-law* predicate (`app/clone_law.py`) drops every "clone-dead" row — one whose calibrated `doubleACT` clone silently fails — and (2) a greedy **ascending-degree independent set** over the survivors (the construction that reproduces the 294-row DIMM-2 s72 production pool byte-for-byte). Held out 2,496/2,496 on the production die's sub85, and 2,494/2,496 cross-die. |
| Per-subarray extras | `pool_layout_dimm2_sub{71,73,76,77,79,84,85}_bank{0-3}.txt` | Small screened pools for voting extras + LOAD-overflow, one per extra calib subarray; |
| Fused colmask (retired module) | `fused_colmask_dimm0_bank{0-3}.txt` | Column-repair set for the fused layout of the retired module (`PIM_FUSED_COLMASK_FILE`); one good-column index per line. |
| MAJ5 reference-policy | `colmasks_maj5_zero_2026_07_17/maj5_best_b{0,2}_bank{0-3}.txt` (+ `_ranking.csv`) | The reference-policy MAJ5 colmasks (per-tuple ZERO vs ZERO+2 winner in the CSV); full tuple provenance in each `.txt` header. The `_b0_` set belongs to the retired module. |

### Pool-file header rule (`fgets(64)`)

The server's sub-pool loader (`app/test_bitnet_server.cpp`, the
`PIM_POOL_LIST_FILE_SUB` path) reads each line into a fixed `char line[64]`
with `fgets` and skips a line only when its first non-blank byte is `#`.
**A comment/header line longer than 63 bytes is split by `fgets`; the
second chunk does not start with `#`, so if it begins with a digit it is
`atoi`'d into a phantom pool row.** In the shipped files this is harmless
*only* because the loader's subarray-window filter (`PIM_SUB_START` /
`PIM_SUB_END`, else the `(any_open/640)*640` default) discards the
low-valued phantoms — they fall far outside every real subarray window.
When authoring your own pool files: keep every `#` header line ≤ 63 bytes,
and always set the subarray window explicitly.
