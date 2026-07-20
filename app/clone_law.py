#!/usr/bin/env python3
"""RowClone-into-Rfirst viability law for DIMM 2 M-die (task O8b, 2026-07-19).

Empirical closed form over 12,480 clone observations (5 tuples x 4 banks,
dimm2_fault_sweep_subs_2026_07_18) + held-out validation on the sub85
tuple: 2496/2496 row verdicts correct (incl. 40/40 in never-observed d
classes). Scope: EVEN-Rfirst tuples on the full-PUD M-die (DIMM 0/2
class). The one odd-Rfirst tuple observed (sub77) deviates in both
directions (97.9% law accuracy) -> for odd anchors, screen on silicon.

Law: with d_low = (R ^ Rfirst) & 1023, predecoder groups
G1..G4 = {1,2},{3,4},{5,6},{7,8} (selection-law addendum 5) and
u = [bit0 in d] + #{groups touched by d}:
  clone R -> Rfirst is DEAD iff
    u >= 5                                  (bit0 + all four groups), or
    u == 4 and bit0 and bit9 and the touched groups are exactly
    {G1,G2,G3}                              (G4 = bits 7,8 untouched).
Everything else is clone-OK; u <= 3 was 6,128/6,128 OK across ALL
tuples including the odd-Rf one. Cross-1024-block source rows follow
the same law (no block term). Bank-invariant (14/12,480 marginal
observations). Mechanism reading: the doubleACT co-activates 2^u rows
(bit9 latches conditionally), diluting the deposit below the
sense threshold.

Corollary (why May's pool was 37% clone-dead): clone-dead rows have
median fault-sweep degree 0 (vs 11 for clone-ok) — the same dilution
hides them from the fault sweep, so degree-ascending greedy IS
constructions preferentially select clone-dead rows unless the
candidate set is pre-filtered with this predicate.
"""

GROUPS = ((1, 2), (3, 4), (5, 6), (7, 8))


def clone_dead(row: int, rfirst: int) -> bool:
    """True if `row` cannot RowClone (doubleACT 30/1) into `rfirst`.

    Valid for even `rfirst` anchors on the DIMM 0/2 M-die class; for odd
    anchors the law is advisory only (~98%) — screen on silicon.
    """
    d = (row ^ rfirst) & 1023
    touched = [bool(d & ((1 << a) | (1 << b))) for a, b in GROUPS]
    u = (d & 1) + sum(touched)
    if u >= 5:
        return True
    return (u == 4 and bool(d & 1) and bool(d & 512)
            and touched == [True, True, True, False])


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        sys.exit("usage: clone_law.py <row> <rfirst>")
    r, rf = int(sys.argv[1]), int(sys.argv[2])
    print(f"clone {r} -> {rf}: {'DEAD' if clone_dead(r, rf) else 'ok'}")
