# Calibration transfer: calibrate once, run the die (and its siblings)

Calibrating a MAJ3-perfect tuple is expensive (`CALIBRATION.md`: a 30–40 h
sweep for the first subarray of a new part). This document is the payoff:
**what transfers without re-sweeping, and the exact recipe to do it.** All
claims are silicon-measured.

## The transfer hierarchy

The DRAM is a step-and-repeat of identical blocks, but the *scope* over
which a calibration transfers depends on which structural level you move
across:

| move across | co-activation lattice | what transfers | what you re-do |
|---|---|---|---|
| **cells** (within a tuple) | — | nothing; this IS the margin | full per-cell screen (the lottery) |
| **banks** (same die) | **byte-identical, all 16** | tuple rows, timings, open-rows, deposit lattice | margin re-screen only |
| **dies** (same die family) | **byte-identical** | everything as for banks | margin re-screen only |
| **subarrays** (same bank) | **PARTIAL** | short-range coupling + MAJ3 tuple geometry | pool layout re-derivation **+** margin screen |
| **parts** (different model) | chip-specific | the *method* only | fault-sweep + lattice re-derivation, then timings |

The load-bearing measurement across banks: bank 0's tuple, transferred
verbatim to all 15 other banks, produces a **byte-identical deposit
structure on every bank** (350/350 constellation rows × 14 primitive
cases); only per-bank cell margins differ.

Across dies, the fault sets and the calibration are byte-identical too. We
have now measured that twice over, in two different senses:

- across **two different part numbers** of the same die family — the
  invariance follows the die design, not the module SKU;
- across **four modules of one part number** from different manufacturing
  weeks, our present population: same 3,969-edge fault set with the same
  checksum on all four channels, and the *whole production trio* —
  calibration, pool layout and row window together — ran on a sibling die
  with no fresh calibration, no fresh screen and no new fixture, producing
  a token-exact model run on first contact.

Subarrays are the exception, and we know why: the co-activation lattice's
**long-range** offsets are *predecoder-block-relative*, not fixed — they
shrink toward a ~512-row block midpoint, vanish, then flip sign (the
boundary atlas). Banks preserve the intra-block position (free transfer); a
subarray shift changes it (the ±256/±128 coupling moves, so
spread-collision-safe pool layouts must be re-derived). The short-range
coupling and the tuple's own MAJ3 geometry are position-invariant, so the
*tuple still computes* — only the allocator's spread-avoidance changes.

## Recipe: adopt a new bank (minutes, not 40 hours)

1. **Copy the calib line** for the target bank: take a working bank's line
   from your `calib_*.txt` and change only the `bank` field. Nothing else
   in the line moves.
2. **Copy the pool layout** verbatim (`pool_layout_..._bank{bank}.txt`) —
   the lattice is identical, so the clone-ok offsets are valid.
3. **Margin re-screen only**: the server's LOAD-time byte-verify already
   *is* the per-bank margin screen — one full-model run surfaces any bank
   whose margins need column exclusion. (Banks 4–7 and 8–15 ran the full
   model token-identical on first try; the extra flakes on a few banks
   stayed inside the voting/screen envelope.)
4. Done — the bank joins the residency pool and the round-parallelism.

## Recipe: adopt a new die (same die family)

Identical to the bank recipe, with one margin re-screen for the whole die.
Use the existing trio with the new channel's bender id — that is exactly
what `calibration/DIMM_POPULATION.conf` expresses, and it is why one trio
serves all four of our channels.

## Recipe: adopt a new subarray (the hard one)

1. Transfer the tuple **geometry** — the row offsets *relative* to the new
   subarray's base. MAJ3 compute is position-invariant, and the relative
   open-row sets come out byte-identical across the window family, so this
   is a translation, not a re-derivation. It is what took subarray adoption
   from ~28 minutes to ~2.5 minutes per window.
2. **Re-derive the pool layout**: run the spread probe (`spread-test-exe`)
   at the new subarray to map its block-relative long-range offsets, then
   screen clone-ok offsets against them. This is the only re-derivation.
3. **Margin screen as usual** — and enforce every criterion per window on
   silicon (MajOps stability, sweep, clone check). Windows are not uniformly
   good: one of the 44 we characterized turned out genuinely bad, failing
   606 of 624 rows on the read/write screen while all the others failed
   zero. Screening catches it; assuming would not.
4. ⚠ **Audit the window modulo 2^15 before you trust it.** These parts
   decode 15 row bits, so `r` and `r + 32768` are the same silicon. Of 44
   characterized subarray windows on our production die, 19 pairs were
   aliases — only 25 were physically distinct, and one of them was the
   production window under another name. Two "different" windows that alias
   will pass every screen independently and then clobber each other, and a
   byte-verify cannot see it because each write verifies against what it
   just wrote.

## Consequence

16-bank scale-out (more residency, more spatial parallelism) is a **config
exercise** on a characterized die — the single most expensive experiment,
the calibration sweep, is amortized across the whole die and its
die-family siblings. That is what makes this approach practical to scale,
and it is the empirical backing for the observation that the die is
~99.99 % idle: the parallelism is there for free, and reaching it costs
copying a file, not a sweep.
