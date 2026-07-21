# Calibration transfer: calibrate once, run the die (and its siblings)

Calibrating a MAJ3-perfect tuple is expensive (`CALIBRATION.md`: a
30–40 h sweep per subarray). This document is the payoff: **what transfers
without re-sweeping, and the exact recipe to do it.** All claims are
silicon-measured (2026-07; `BANK_AUDIT_2026_07.md` for the raw probes).

## The transfer hierarchy

The DRAM is a step-and-repeat of identical blocks, but the *scope* over
which a calibration transfers depends on which structural level you move
across:

| move across | co-activation lattice | what transfers | what you re-do |
|---|---|---|---|
| **cells** (within a tuple) | — | nothing; this IS the margin | full per-cell screen (the lottery) |
| **banks** (same die) | **byte-identical, all 16** | tuple rows, timings, open-rows, deposit lattice | margin re-screen only |
| **dies** (same part number) | **byte-identical** | everything as for banks | margin re-screen only |
| **subarrays** (same bank) | **PARTIAL** | short-range coupling + MAJ3 tuple geometry | pool layout re-derivation **+** margin screen |
| **parts** (different model) | chip-specific | the *method* only | fault-sweep + lattice re-derivation, then timings |

The load-bearing measurement: bank 0's tuple, transferred verbatim to all
15 other banks, produces a **byte-identical deposit structure on every
bank** (350/350 constellation rows × 14 primitive cases); only per-bank
cell margins differ (banks 8–11 carry a ~4-cell flake). Across
same-model dies the fault sets and calibration are byte-identical too.

Subarrays are the exception, and we know why: the co-activation lattice's
**long-range** offsets are *predecoder-block-relative*, not fixed —
they shrink toward a ~512-row block midpoint, vanish, then flip sign
(`BANK_AUDIT_2026_07.md`, boundary atlas). Banks preserve the intra-block
position (free transfer); a subarray shift changes it (the ±256/±128
coupling moves, so spread-collision-safe pool layouts must be
re-derived). The short-range coupling and the tuple's own MAJ3 geometry
are position-invariant, so the *tuple still computes* — only the
allocator's spread-avoidance changes.

## Recipe: adopt a new bank (minutes, not 40 hours)

1. **Copy the calib line** for the target bank: take a working bank's
   line from `calib_dimm*.txt`, change only the `bank` field
   (`calib_dimm2_x16.txt` is the worked example — bank 0's lines
   replicated to banks 4–7/8–15).
2. **Copy the pool layout** verbatim (`pool_layout_..._bank{bank}.txt`) —
   the lattice is identical, so the cloneok offsets are valid.
3. **Margin re-screen only**: the server's LOAD-time byte-verify already
   *is* the per-bank margin screen — one full-model run surfaces any
   bank whose margins need column exclusion. (Banks 4–7 and 8–15 ran the
   full model token-identical on first try; banks 8–11's extra flakes
   stayed within the voting/screen envelope.)
4. Done — the bank joins the residency pool and the round-parallelism.

## Recipe: adopt a new die (same part)

Identical to the bank recipe, one margin re-screen for the whole die
(fault sets transfer byte-identically across our two same-model modules).
Use the existing calib file with the die's bender id.

## Recipe: adopt a new subarray (the hard one)

1. Transfer the tuple **geometry** (row offsets relative to the new
   subarray's base) — MAJ3 compute is position-invariant.
2. **Re-derive the pool layout**: run the spread probe
   (`spread-test-exe`) at the new subarray to map its block-relative
   long-range offsets, then screen cloneok offsets against them. This is
   the only re-derivation; it is minutes on the rig, not a 40-h sweep.
3. Margin screen as usual.
Note: on DIMM 2 only 3 subarrays (s72/78/86) were ever swept, so
subarray expansion also needs the target subarray's MAJ3 screen if it was
never characterized — cheaper than the full campaign but not free.

## Consequence

16-bank scale-out (4× residency + spatial parallelism) is a **config
exercise** on a characterized die — the single most expensive experiment
(the calibration sweep) is amortized across the whole die and its
same-model siblings. This is what makes the in-DRAM approach practical to
scale, and it is the empirical backing for `UTILIZATION.md`'s "the die is
~99.99 % idle — the parallelism is there for free."
