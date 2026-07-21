# Bank-similarity audit, tranche 1 (2026-07-21): banks 4-7 vs bank 0

**Claim tested** (LEVERS #11 / ROADMAP §C): the co-activation lattice is
decoder wiring — a design constant — so an unmeasured bank should behave
identically to a calibrated one under a verbatim-transferred calib, with
only cell-level margins differing.

**Setup**: `spread-test-exe` constellation probe (25 XOR offsets × 14
primitive cases: single-ACT, doubleACT(0,0)/(10,2)/(30,1) × far/tuple
dst, MAJ3-like pairs, full MM3D body), bender 2, R=45341 (an s72 open
row), DST_FAR=60000, **bank 0's s72 primary tuple transferred verbatim**
(PIM_CALIB_LINE) to banks 4, 5, 6, 7 — none of which had ever been
calibrated or touched by PIM before this run. Second config (R=45640,
non-tuple center) run first as a null control.

## Results

1. **Classification-identical 350/350 rows on every bank** (4, 5, 6, 7
   vs 0): the same offsets deposit ("here": −1 and +256 on
   doubleACT(10,2)/(30,1)→far; −1 on tuple-directed (30,1)/(10,2)), the
   same offsets flake, the same offsets stay untouched — for all 14
   cases. The transferred tuple computes/behaves on never-calibrated
   banks exactly as on the calibrated one.
2. **The flake fringe is deterministic too**: ±1/±2 partial-corruption
   offsets repeat bank-for-bank, case-for-case — sub-threshold lattice
   coupling, not random cell noise. (New sharpening: the profile
   *including its fringe* is bank-invariant.)
3. **Null control**: with R outside tuple-lattice conditions, zero
   deposits on any bank (only scattered flakes) — reconfirming
   dst/lattice-conditionality uniformly.
4. **Margins are the per-bank part**: exact own-match counts differ by a
   few cells per row across banks (the strict table diff), as predicted.

## Consequences

- **Calib transfer to new banks = margin re-screen only**, now shown at
  the primitive level on silicon that had zero prior characterization.
- The 16-bank scale-out (#13) is real: 12 uncharacterized banks × ~5
  dense clusters each ≈ 4× residency/parallelism headroom without a
  single new calibration sweep — the actual fix for the LOAD-residency
  ceiling (#8: only 5 calibrated clusters/bank exist in the calib file;
  dual-subarray compute itself re-validated safe the same night, May's
  task-#75 failure did not reproduce).
- Remaining audit work: margin maps (own-match statistics on untouched
  rows per bank), banks 8-15, and the selection-law probe replicated on
  one new bank for completeness.

Logs: `audit_bank{0,4,5,6,7}.log` (null config), `audit2_bank*.log`
(tuple config), this dir.
