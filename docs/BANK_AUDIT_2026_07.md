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

## Capstone (same night): full-model inference on the virgin banks

`--bank "4,5,6,7"` with bank-0's calib lines transferred verbatim
(calib_dimm2_x16.txt) and bank-0's cloneok pool layout copied per bank:
bonsai_1bit 2-tok = **token-identical (' What is') at 131.6 s** — the
same wall as the calibrated banks 0-3 (132.8 s). The server's LOAD-time
byte-verify diagnostics on banks 4-7 are the same shape and magnitude
class as the 0-3 baseline (margins in-family; the built-in verify IS the
margin re-screen). The transfer chain is now demonstrated at every
level: primitive constellation (350/350) → full-model inference —
**calibrate one bank, run on any bank of the die.** Next: the 8-bank
run (0-7) — 2× residency AND half the rounds per request.

## 8-bank run (0-7): correctness + capacity mechanics proven, wall-neutral

`--bank "0,1,2,3,4,5,6,7"`: token-identical, 137.4 s /2tok. Rounds per
request halved (exec 128→64 programs ✓) but recv stayed ~24.7 ms — at
16 KB/drain the XDMA size-sublinear cost curve trades wake count for
transfer time almost exactly (effective c2h ~1.2 GB/s vs PCIe's ~7:
per-transfer overhead dominates at every size we use). Residency
doubled but remains a minority of the model (~46 vs ~360 slices; 5
dense clusters/bank is the binding constant — 16 banks ⇒ ~4×, still
partial). Verdict: bank scale-out is a CONFIG EXERCISE now (correct on
first try, virgin banks included); its wall payoff arrives with the
recv-side levers (ACCUM_XBP byte collapse, Rung-1 streaming) and
cluster mining from the existing sweep CSVs.
