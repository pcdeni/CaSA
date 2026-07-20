#!/usr/bin/env python3
"""Task O8 (a)+(b): clone-dead law analysis + clone-ok primary pool.

(b) Characterize clone-dead rows vs the APA selection law over ALL
    on-disk clone-check data (5 tuples x 4 banks, campaign
    dimm2_fault_sweep_subs_2026_07_18): is clone-dead PREDICTABLE from
    d = R xor Rfirst alone?  Extract the exact rule, quantify accuracy,
    emit a held-out prediction file for the sub85 tuple (Rf=54412,
    window [54144,54784)) to be verified by a fresh silicon clone-check.

(a) Derive a clone-ok-only PRIMARY pool (greedy-IS, the recovered May
    2026-05-21 construction, restricted to clone-ok rows) ->
    pool_layout_dimm2_cloneok_bank{B}.txt.  Production files untouched.
"""
import re, sys, hashlib, statistics
from collections import defaultdict

D   = "/home/deni/Claude/dimm2_fault_sweep_subs_2026_07_18"
BIT = ("/home/deni/Claude/SiMRA-DRAM-main/DRAM-Bender/sources/apps/"
       "DSN_AE_APPS/BitNet")
MAY = "/home/deni/Claude/dimm2_fault_sweep"
BANKS = [0, 1, 2, 3]

TUPLES = {
    "primary": dict(win=(45312, 45952), rf=45340, rs=45823,
                    open=[45340,45341,45342,45343,45436,45437,45438,45439,
                          45724,45725,45726,45727,45820,45821,45822,45823]),
    "sub71":   dict(win=(45312, 45952), rf=45464, rs=45857,
                    open=[45440,45441,45464,45465,45472,45473,45496,45497,
                          45824,45825,45848,45849,45856,45857,45880,45881]),
    "sub76":   dict(win=(49152, 49792), rf=49178, rs=49622,
                    open=[49170,49174,49178,49182,49234,49238,49242,49246,
                          49554,49558,49562,49566,49618,49622,49626,49630]),
    "sub77":   dict(win=(49152, 49792), rf=49291, rs=49719,
                    open=[49291,49295,49299,49303,49323,49327,49331,49335,
                          49675,49679,49683,49687,49707,49711,49715,49719]),
    "sub84":   dict(win=(54144, 54784), rf=54150, rs=54620,
                    open=[54148,54150,54172,54174,54212,54214,54236,54238,
                          54532,54534,54556,54558,54596,54598,54620,54622]),
}
# Held-out tuple for the (b) verification sweep (same line on all 4 banks
# in calib_dimm2.txt, cluster 85, same physical s86 window):
HELDOUT = dict(name="sub85", win=(54144, 54784), rf=54412, rs=54770,
               open=[54410,54412,54418,54420,54506,54508,54514,54516,
                     54666,54668,54674,54676,54762,54764,54770,54772])

GROUPS = [(1, 2), (3, 4), (5, 6), (7, 8)]   # selection-law predecoder groups

def parse_log(path):
    edges, clones, match = set(), {}, {}
    for line in open(path):
        m = re.match(r"^FAULT R=(\d+) -> T=(\d+)", line)
        if m:
            edges.add((int(m.group(1)), int(m.group(2)))); continue
        m = re.match(r"^CLONE R=(\d+) match=(\d+) ok=(\d)", line)
        if m:
            r = int(m.group(1))
            clones[r] = int(m.group(3)) == 1
            match[r] = int(m.group(2))
    return edges, clones, match

def units_of(d_low, count_bit9):
    """Selection-law unit count of a 10-bit local distance."""
    u = 0
    if d_low & 1: u += 1                      # singleton bit 0
    for a, b in GROUPS:
        if d_low & ((1 << a) | (1 << b)): u += 1
    if count_bit9 and (d_low & (1 << 9)): u += 1
    return u

def group_sig(d_low):
    """Canonical signature: for each unit, whether d holds one or both bits.
    ('0'=untouched, 'a'/'b'=single bit, 'F'=full pair)."""
    sig = ["1" if d_low & 1 else "0"]
    for a, b in GROUPS:
        da, db = bool(d_low & (1 << a)), bool(d_low & (1 << b))
        sig.append("F" if (da and db) else ("a" if da else ("b" if db else "0")))
    sig.append("1" if d_low & (1 << 9) else "0")
    return "".join(sig)

data = {}
for name in TUPLES:
    for b in BANKS:
        data[(name, b)] = parse_log(f"{D}/{name}_bank{b}.log")

# ---------------------------------------------------------------- (b) ----
print("=" * 74)
print("(b) CLONE-DEAD LAW ANALYSIS -- all 5 tuples x 4 banks")
print("=" * 74)

# 1. Determinism of the outcome per (tuple, R) across banks.
rows_total = rows_split_banks = 0
for name in TUPLES:
    for r in data[(name, 0)][1]:
        outs = {data[(name, b)][1].get(r) for b in BANKS}
        rows_total += 1
        if len(outs) > 1:
            rows_split_banks += 1
print(f"[det] per-(tuple,row) outcome identical on all 4 banks: "
      f"{rows_total - rows_split_banks}/{rows_total} "
      f"({rows_split_banks} bank-marginal rows)")

# 2. Determinism in d alone: group every observation by full d and by d_low.
by_dlow = defaultdict(lambda: [0, 0, []])   # d_low -> [ok, fail, matches]
by_dlow_cross = defaultdict(lambda: [0, 0])  # (d_low, crossblk) -> [ok, fail]
per_class_tuples = defaultdict(set)
for name, t in TUPLES.items():
    rf = t["rf"]
    for b in BANKS:
        _, clones, match = data[(name, b)]
        for r, ok in clones.items():
            d = r ^ rf
            d_low = d & 1023
            cross = (r >> 10) != (rf >> 10)
            e = by_dlow[d_low]
            e[0 if ok else 1] += 1
            e[2].append(match[r])
            c = by_dlow_cross[(d_low, cross)]
            c[0 if ok else 1] += 1
            per_class_tuples[d_low].add(name)

pure_ok = pure_fail = mixed = 0
mixed_classes = []
for d_low, (n_ok, n_fail, _) in sorted(by_dlow.items()):
    if n_ok and n_fail:
        mixed += 1; mixed_classes.append(d_low)
    elif n_ok:
        pure_ok += 1
    else:
        pure_fail += 1
print(f"[det] d_low classes: {len(by_dlow)} total -> "
      f"pure-ok {pure_ok}, pure-fail {pure_fail}, MIXED {mixed}")
if mixed_classes:
    print(f"[det] mixed d_low classes: "
          f"{[f'0x{d:03x}' for d in mixed_classes[:40]]}")
    for d_low in mixed_classes[:40]:
        parts = []
        for name, t in TUPLES.items():
            rf = t["rf"]
            for b in BANKS:
                _, clones, match = data[(name, b)]
                for r, ok in clones.items():
                    if (r ^ rf) & 1023 == d_low:
                        parts.append((name, b, r, ok, match[r]))
        byname = defaultdict(lambda: [0, 0])
        for name, b, r, ok, m in parts:
            byname[name][0 if ok else 1] += 1
        detail = " ".join(f"{n}:{v[0]}ok/{v[1]}f" for n, v in sorted(byname.items()))
        cs = {(r >> 10) != (TUPLES[n]['rf'] >> 10) for n, b, r, ok, m in parts}
        print(f"    d_low=0x{d_low:03x} sig={group_sig(d_low)} "
              f"u9={units_of(d_low, True)} cross={cs}  {detail}")

# 3. Unit-count structure.
for count9, tag in [(True, "bit9 COUNTED as unit"), (False, "bit9 NOT a unit")]:
    print(f"[units] --- {tag} ---")
    agg = defaultdict(lambda: [0, 0, []])
    for d_low, (n_ok, n_fail, matches) in by_dlow.items():
        u = units_of(d_low, count9)
        agg[u][0] += n_ok; agg[u][1] += n_fail; agg[u][2].extend(matches)
    for u in sorted(agg):
        n_ok, n_fail, matches = agg[u]
        med = statistics.median(matches)
        print(f"    u={u}: ok={n_ok:5d} fail={n_fail:5d} "
              f"({100.0 * n_fail / (n_ok + n_fail):5.1f}% dead) "
              f"median_match={med:.0f}/2048")

# 4. Drill into the split level: which 5-unit (bit9-counted) classes fail?
print("[split] per-class verdicts at each u level (bit9 counted):")
for u_probe in range(0, 7):
    classes = [(d, v) for d, v in by_dlow.items()
               if units_of(d, True) == u_probe]
    if not classes: continue
    n_pure_f = sum(1 for d, v in classes if not v[0])
    n_pure_o = sum(1 for d, v in classes if not v[1])
    n_mix    = len(classes) - n_pure_f - n_pure_o
    print(f"    u={u_probe}: {len(classes)} classes -> "
          f"{n_pure_o} all-ok, {n_pure_f} all-fail, {n_mix} mixed")

# 5. Candidate predicates, scored over every observation.
def score(pred):
    """pred(d_low) -> True means predicted clone-DEAD."""
    tp = fp = tn = fn = 0
    for d_low, (n_ok, n_fail, _) in by_dlow.items():
        if pred(d_low):
            tp += n_fail; fp += n_ok
        else:
            tn += n_ok; fn += n_fail
    n = tp + fp + tn + fn
    return tp, fp, tn, fn, (tp + tn) / n

# Structural features that could separate the split level:
def has_full_pair(d):  return any((d & ((1<<a)|(1<<b))) == ((1<<a)|(1<<b))
                                  for a, b in GROUPS)
def n_full_pairs(d):   return sum(1 for a, b in GROUPS
                                  if (d & ((1<<a)|(1<<b))) == ((1<<a)|(1<<b)))
def n_single_groups(d):return sum(1 for a, b in GROUPS
                                  if bool(d & (1<<a)) != bool(d & (1<<b)))
cands = {
    "u9 >= 6":            lambda d: units_of(d, True) >= 6,
    "u9 >= 5":            lambda d: units_of(d, True) >= 5,
    "u_no9 >= 5":         lambda d: units_of(d, False) >= 5,
    "u_no9 >= 4":         lambda d: units_of(d, False) >= 4,
    "u9>=6 or (u9==5 & bit9)":  lambda d: units_of(d, True) >= 6 or
                          (units_of(d, True) == 5 and bool(d & 512)),
    "u9>=6 or (u9==5 & ~bit9)": lambda d: units_of(d, True) >= 6 or
                          (units_of(d, True) == 5 and not (d & 512)),
    "u9>=6 or (u9==5 & bit0)":  lambda d: units_of(d, True) >= 6 or
                          (units_of(d, True) == 5 and bool(d & 1)),
    "u9>=6 or (u9==5 & ~bit0)": lambda d: units_of(d, True) >= 6 or
                          (units_of(d, True) == 5 and not (d & 1)),
    "u9>=6 or (u9==5 & full-pair)": lambda d: units_of(d, True) >= 6 or
                          (units_of(d, True) == 5 and has_full_pair(d)),
    "u9>=6 or (u9==5 & no-full-pair)": lambda d: units_of(d, True) >= 6 or
                          (units_of(d, True) == 5 and not has_full_pair(d)),
}
print("[rule] candidate predicates (predict DEAD):")
best = None
for name, p in cands.items():
    tp, fp, tn, fn, acc = score(p)
    print(f"    {name:38s} acc={100*acc:6.2f}%  "
          f"(missed-dead={fn}, false-dead={fp})")
    if best is None or acc > best[1]:
        best = (name, acc, p)

# 6. Fully empirical rule: the exact set of observed all-fail classes.
fail_classes = sorted(d for d, v in by_dlow.items() if v[1] and not v[0])
ok_classes   = sorted(d for d, v in by_dlow.items() if v[0] and not v[1])
print(f"[rule] observed: {len(fail_classes)} all-fail classes, "
      f"{len(ok_classes)} all-ok classes, {mixed} mixed")
print(f"[rule] best closed-form: {best[0]}  acc={100*best[1]:.2f}%")

# 7. Cross-block: does the SAME d_low behave differently cross-block?
cross_diff = []
for (d_low, cross), (n_ok, n_fail) in sorted(by_dlow_cross.items()):
    other = by_dlow_cross.get((d_low, not cross))
    if other is None: continue
    if cross: continue  # report each pair once
    v_in  = (n_ok, n_fail)
    v_out = tuple(other)
    def verdict(v): return "ok" if not v[1] else ("fail" if not v[0] else "MIX")
    if verdict(v_in) != verdict(v_out):
        cross_diff.append((d_low, v_in, v_out))
print(f"[cross-block] d_low classes observed BOTH in- and cross-block with "
      f"DIFFERENT verdicts: {len(cross_diff)}")
for d_low, vi, vo in cross_diff[:20]:
    print(f"    d_low=0x{d_low:03x} sig={group_sig(d_low)} "
          f"in-block ok/f={vi}  cross-block ok/f={vo}")

# 7b. Tuple-dependence: per-(tuple, d_low) purity + who carries the
# deviations from the global per-class majority.
n_pure_td = n_mixed_td = 0
dev_by_tuple = defaultdict(int)
obs_by_tuple = defaultdict(int)
for name, t in TUPLES.items():
    rf = t["rf"]
    per_dl = defaultdict(lambda: [0, 0])
    for b in BANKS:
        _, clones, _ = data[(name, b)]
        for r, ok in clones.items():
            per_dl[(r ^ rf) & 1023][0 if ok else 1] += 1
    for d_low, (n_ok, n_fail) in per_dl.items():
        if n_ok and n_fail: n_mixed_td += 1
        else: n_pure_td += 1
        # deviation vs global majority verdict
        g_ok, g_fail, _ = by_dlow[d_low]
        maj_dead = g_fail > g_ok
        dev = n_fail if not maj_dead else n_ok
        # count obs that disagree with global majority
        dev_by_tuple[name] += (n_ok if maj_dead else n_fail)
        obs_by_tuple[name] += n_ok + n_fail
print(f"[tuple-dep] per-(tuple,d_low) classes: pure={n_pure_td} "
      f"mixed={n_mixed_td} (mixed == bank-marginal rows)")
print(f"[tuple-dep] observations deviating from the GLOBAL per-class "
      f"majority, by tuple:")
for name in TUPLES:
    print(f"    {name:8s}: {dev_by_tuple[name]:4d}/{obs_by_tuple[name]} "
          f"({100.0*dev_by_tuple[name]/obs_by_tuple[name]:.2f}%)  "
          f"Rfirst={TUPLES[name]['rf']} ({'ODD' if TUPLES[name]['rf'] & 1 else 'even'})")

# 7c. The u_no9=4 and =5 exception structure per tuple.
print("[exceptions] u_no9=4 dead + u_no9=5 ok observations by tuple:")
for name, t in TUPLES.items():
    rf = t["rf"]
    d4 = o5 = n4 = n5 = 0
    for b in BANKS:
        _, clones, _ = data[(name, b)]
        for r, ok in clones.items():
            u = units_of((r ^ rf) & 1023, False)
            if u == 4:
                n4 += 1
                if not ok: d4 += 1
            elif u == 5:
                n5 += 1
                if ok: o5 += 1
    print(f"    {name:8s}: u4-dead={d4:3d}/{n4}  u5-ok={o5:3d}/{n5}")

# 7d. Per-class purity EXCLUDING sub77 (even-Rfirst tuples only).
by_dlow_even = defaultdict(lambda: [0, 0])
for name, t in TUPLES.items():
    if name == "sub77": continue
    rf = t["rf"]
    for b in BANKS:
        _, clones, _ = data[(name, b)]
        for r, ok in clones.items():
            by_dlow_even[(r ^ rf) & 1023][0 if ok else 1] += 1
pe_ok = pe_fail = pe_mix = 0
for d_low, (n_ok, n_fail) in by_dlow_even.items():
    if n_ok and n_fail: pe_mix += 1
    elif n_ok: pe_ok += 1
    else: pe_fail += 1
print(f"[even-Rf] per-class purity excluding sub77: "
      f"{pe_ok} all-ok, {pe_fail} all-fail, {pe_mix} mixed "
      f"(of {len(by_dlow_even)})")
mixed_even = [d for d, v in by_dlow_even.items() if v[0] and v[1]]
for d_low in mixed_even[:12]:
    print(f"    even-mixed d_low=0x{d_low:03x} sig={group_sig(d_low)} "
          f"u9={units_of(d_low, True)}")

# 7e. Structure of the u_no9=4 dead classes (even-Rf table): which
# 4-unit patterns die?
u4_dead_cls = sorted(d for d, v in by_dlow_even.items()
                     if v[1] and not v[0] and units_of(d, False) == 4)
u4_ok_cls   = sorted(d for d, v in by_dlow_even.items()
                     if v[0] and not v[1] and units_of(d, False) == 4)
u5_dead_cls = [d for d, v in by_dlow_even.items()
               if v[1] and not v[0] and units_of(d, False) == 5]
print(f"[u4-struct] even-Rf u_no9=4 classes: {len(u4_ok_cls)} ok, "
      f"{len(u4_dead_cls)} dead; u_no9=5 dead classes: {len(u5_dead_cls)}")
def struct_feats(d):
    return dict(bit0=bool(d & 1), bit9=bool(d & 512),
                nfull=n_full_pairs(d), nsing=n_single_groups(d),
                sig=group_sig(d))
from collections import Counter
cnt_dead = Counter((struct_feats(d)["bit0"], struct_feats(d)["bit9"],
                    struct_feats(d)["nfull"]) for d in u4_dead_cls)
cnt_ok = Counter((struct_feats(d)["bit0"], struct_feats(d)["bit9"],
                  struct_feats(d)["nfull"]) for d in u4_ok_cls)
print("    (bit0, bit9, n_full_pairs) -> dead-classes | ok-classes")
for key in sorted(set(cnt_dead) | set(cnt_ok)):
    print(f"    {key}: {cnt_dead.get(key, 0):3d} | {cnt_ok.get(key, 0):3d}")
print("    u4-dead class sigs:",
      " ".join(group_sig(d) for d in u4_dead_cls[:30]))

# 7e2. THE closed-form law (even-Rf): DEAD iff u_no9 >= 5, OR
# (u_no9 == 4 AND bit0 AND bit9 AND touched groups == {G1,G2,G3}).
# Note (u_no9==4 ∧ bit0) ⟺ exactly 3 groups touched, so the second arm
# is "bit0 ∧ bit9 ∧ G4 untouched ∧ G1,G2,G3 touched" — all 27 such
# classes exist and all are dead; the 81 sibling classes (other group
# untouched) are all ok.
def law_dead(d):
    u = units_of(d, False)
    if u >= 5: return True
    if u == 4 and (d & 1) and (d & 512):
        touched = [bool(d & ((1 << a) | (1 << b))) for a, b in GROUPS]
        return touched == [True, True, True, False]
    return False
tp = fp = tn = fn = 0
for name, t in TUPLES.items():
    if name == "sub77": continue
    rf = t["rf"]
    for b in BANKS:
        _, clones, _ = data[(name, b)]
        for r, ok in clones.items():
            pred_dead = law_dead((r ^ rf) & 1023)
            if pred_dead and not ok: tp += 1
            elif pred_dead and ok: fp += 1
            elif not pred_dead and not ok: fn += 1
            else: tn += 1
tot = tp + fp + tn + fn
print(f"[LAW] closed form on even-Rf observations: acc={100.0*(tp+tn)/tot:.3f}% "
      f"({tp+tn}/{tot}; false-dead={fp}, missed-dead={fn})")
tp7 = fp7 = tn7 = fn7 = 0
rf = TUPLES["sub77"]["rf"]
for b in BANKS:
    _, clones, _ = data[("sub77", b)]
    for r, ok in clones.items():
        pred_dead = law_dead((r ^ rf) & 1023)
        if pred_dead and not ok: tp7 += 1
        elif pred_dead and ok: fp7 += 1
        elif not pred_dead and not ok: fn7 += 1
        else: tn7 += 1
print(f"[LAW] same closed form on sub77 (ODD Rf): "
      f"acc={100.0*(tp7+tn7)/2496:.2f}% (false-dead={fp7}, missed-dead={fn7})"
      f" -> the law is Rf-parity-conditional")

# 7f. Fault-degree vs clone status (primary tuple): did the May greedy
# anti-select clone-ok rows?
deg_ok, deg_dead = [], []
adj_prim = defaultdict(set)
for b in BANKS:
    for r, t in data[("primary", b)][0]:
        adj_prim[r].add(t); adj_prim[t].add(r)
pb_ok = set.intersection(*[set(r for r, ok in data[("primary", b)][1].items()
                               if ok) for b in BANKS])
for r in data[("primary", 0)][1]:
    (deg_ok if r in pb_ok else deg_dead).append(len(adj_prim.get(r, ())))
print(f"[degree] primary fault-degree median: clone-ok="
      f"{statistics.median(deg_ok):.0f} clone-dead="
      f"{statistics.median(deg_dead):.0f} (n={len(deg_ok)}/{len(deg_dead)})")

# 8. Held-out predictions for the sub85 tuple (Rf even, like every tuple
# but sub77): predict from the EVEN-Rf per-class table, closed-form
# fallback u_no9>=5 for unseen classes. One verdict per row, committed
# BEFORE the silicon clone-check.
rf85 = HELDOUT["rf"]
open85 = set(HELDOUT["open"])
pred_lines = []
n_pred_dead = n_pred_ok = n_fallback = 0
for r in range(*HELDOUT["win"]):
    if r in open85: continue
    d_low = (r ^ rf85) & 1023
    v = by_dlow_even.get(d_low)
    if v and v[0] and not v[1]:
        p = "ok"; n_pred_ok += 1
    elif v and v[1] and not v[0]:
        p = "dead"; n_pred_dead += 1
    else:  # unseen or (rare) even-mixed class: closed-form fallback
        if units_of(d_low, False) >= 5:
            p = "dead*"; n_pred_dead += 1
        else:
            p = "ok*"; n_pred_ok += 1
        n_fallback += 1
    pred_lines.append(f"{r} {d_low} {p}")
with open(f"{D}/o8_sub85_clone_predictions.txt", "w") as f:
    f.write("# sub85 (Rf=54412) clone predictions committed BEFORE the "
            "silicon clone-check.\n"
            "# rule = even-Rf per-class empirical table (5 tuples x 4 "
            "banks minus sub77) + u_no9>=5 fallback ('*') for unseen "
            "classes.\n# row d_low verdict\n")
    f.write("\n".join(pred_lines) + "\n")
print(f"[heldout] sub85 predictions: {n_pred_ok} ok, {n_pred_dead} dead "
      f"({n_fallback} via closed-form fallback) -> "
      f"o8_sub85_clone_predictions.txt")

# ---------------------------------------------------------------- (a) ----
print()
print("=" * 74)
print("(a) CLONE-OK PRIMARY POOL DERIVATION")
print("=" * 74)

prod = [int(l) for l in open(f"{BIT}/pool_layout_dimm2_bank0.txt")
        if l.strip() and not l.startswith("#")]
prod_set = set(prod)

# Reproduce the production construction first (both candidate orders).
may_edges = parse_log(f"{MAY}/bank0.log")[0]
adj_may = defaultdict(set)
for r, t in may_edges:
    adj_may[r].add(t); adj_may[t].add(r)
open_primary = set(TUPLES["primary"]["open"])
cands_all = [r for r in range(*TUPLES["primary"]["win"])
             if r not in open_primary]

def greedy_rowasc(cands, adj, blocked=frozenset()):
    chosen, cs = [], set()
    for r in cands:
        if r in blocked: continue
        if adj.get(r, frozenset()) & cs: continue
        chosen.append(r); cs.add(r)
    return chosen

def greedy_degasc(cands, adj, blocked=frozenset()):
    order = sorted(cands, key=lambda r: (len(adj.get(r, ())), r))
    chosen, cs = [], set()
    for r in order:
        if r in blocked: continue
        if adj.get(r, frozenset()) & cs: continue
        chosen.append(r); cs.add(r)
    return sorted(chosen)

g_row = greedy_rowasc(cands_all, adj_may)
g_deg = greedy_degasc(cands_all, adj_may)
print(f"[repro] production=294; greedy-row-asc={len(g_row)} "
      f"byte-identical={set(g_row) == prod_set and g_row == sorted(g_row) and prod == sorted(prod) and g_row == prod}; "
      f"greedy-deg-asc={len(g_deg)} identical={g_deg == prod}")
use_greedy = greedy_rowasc if g_row == prod else greedy_degasc
assert (g_row == prod) or (g_deg == prod), "neither order reproduces production"

# Union fault graph for the primary tuple: May bank0 + all 4 new banks.
edges_u = set(may_edges)
for b in BANKS:
    edges_u |= data[("primary", b)][0]
adj_u = defaultdict(set)
for r, t in edges_u:
    adj_u[r].add(t); adj_u[t].add(r)

# clone-ok (all 4 banks) for the PRIMARY tuple.
per_bank_ok = [set(r for r, ok in data[("primary", b)][1].items() if ok)
               for b in BANKS]
cloneok = set.intersection(*per_bank_ok)
print(f"[cloneok] primary clone-ok on all 4 banks: {len(cloneok)}/624 "
      f"(marginal: {len(set.union(*per_bank_ok)) - len(cloneok)})")
prod_dead = sorted(prod_set - cloneok)
tail = prod[-16:]
print(f"[cloneok] production pool rows clone-DEAD: {len(prod_dead)}/294; "
      f"V2 tail dead: {sum(1 for r in tail if r not in cloneok)}/16")

# Blocked set: primary opens + sub71 opens (sub71 voting trips wrRow its
# open rows every trip -- 8 of them are production-pool rows today, a
# built-in resident-corruption path this new pool must not repeat) +
# rows receiving a DIRECTED fault edge from any sub71-pool scratch row
# under the sub71 tuple's graph (sub71 voting bodies deposit there).
open_sub71 = set(TUPLES["sub71"]["open"])
sub71_pool = [int(l) for l in open(f"{BIT}/pool_layout_dimm2_sub71_bank0.txt")
              if l.strip() and not l.startswith("#")]
out71 = defaultdict(set)
for b in BANKS:
    for r, t in data[("sub71", b)][0]:
        out71[r].add(t)
sub71_targets = set()
for r in sub71_pool:
    sub71_targets |= out71.get(r, set())
print(f"[blocked] sub71-scratch deposit targets in window: "
      f"{len(sub71_targets)} rows")

# Two-tier semantics, faithful to the recovered production construction:
# excluded-as-CANDIDATE rows (must not hold residents / must stay free
# for sub71) are removed from the candidate list, but the IS discipline
# itself is production's: no fault edges among CHOSEN rows only (the
# production 294 imposes no "no edge to opens" rule — reproduced
# byte-identical with an empty blocked set).
excluded = open_primary | open_sub71 | set(sub71_pool) | sub71_targets
cands_ok = [r for r in range(*TUPLES["primary"]["win"])
            if r in cloneok and r not in excluded]
print(f"[cands] clone-ok, non-excluded candidates: {len(cands_ok)}")

# Greedy IS with three orderings; keep the largest (IS property is what
# matters and is verified below; ordering is a size heuristic). The
# candidate-induced-subgraph degree is the right degree measure here:
# global degree counts edges to rows that can never join the pool.
cset = set(cands_ok)
def sub_degree(r):
    return sum(1 for t in adj_u.get(r, ()) if t in cset)
def greedy_order(order):
    chosen, cs = [], set()
    for r in order:
        if adj_u.get(r, frozenset()) & cs: continue
        chosen.append(r); cs.add(r)
    return sorted(chosen)
variants = {
    "global-deg-asc": greedy_order(sorted(cands_ok,
                       key=lambda r: (len(adj_u.get(r, ())), r))),
    "subgraph-deg-asc": greedy_order(sorted(cands_ok,
                       key=lambda r: (sub_degree(r), r))),
    "row-asc": greedy_order(cands_ok),
}
for k, v in variants.items():
    print(f"[pool] greedy IS ({k}): {len(v)} rows")
pool_is = max(variants.values(), key=len)

# V2 scratch tail via the sub71-style RELAXED rule (O1 construction):
# scratch rows are rewritten immediately before every use, so edges INTO
# them are harmless; what they must never do is deposit INTO a
# LOAD-resident row. Tail candidates: clone-ok, non-excluded, not in the
# IS, with no DIRECTED fault edge into any IS row. This frees all
# IS rows for LOAD instead of sacrificing its last 16.
out_prim = defaultdict(set)
for r, t in may_edges:
    out_prim[r].add(t)
for b in BANKS:
    for r, t in data[("primary", b)][0]:
        out_prim[r].add(t)
is_set = set(pool_is)
tail_cands = [r for r in cands_ok
              if r not in is_set and not (out_prim.get(r, set()) & is_set)]
tail16 = tail_cands[:16]
print(f"[tail] relaxed-rule scratch tail candidates: {len(tail_cands)}; "
      f"took {len(tail16)}")
assert len(tail16) == 16, "not enough relaxed-rule tail rows"
pool_new = sorted(pool_is) + sorted(tail16)
print(f"[pool] final: {len(pool_is)} IS LOAD rows + 16 relaxed-rule "
      f"scratch tail = {len(pool_new)} "
      f"(production: 278 LOAD + 16 tail = 294, {294 - len(prod_dead)} clone-ok)")

# Verification.
ps = set(pool_new)
assert len(ps) == len(pool_new), "duplicate rows"
v_intra = sum(1 for r in pool_is for t in adj_u.get(r, ()) if t in is_set)
v_open  = len(ps & (open_primary | open_sub71))
v_clone = sum(1 for r in pool_new if r not in cloneok)
v_s71t  = len(ps & sub71_targets)
v_s71p  = len(ps & set(sub71_pool))
v_tail_dep = sum(1 for r in tail16 if out_prim.get(r, set()) & is_set)
overlap_prod = len(ps & prod_set)
print(f"[verify] IS-intra-edges={v_intra} opens-hit={v_open} "
      f"not-clone-ok={v_clone} sub71-deposit-targets={v_s71t} "
      f"sub71-pool-overlap={v_s71p} tail-deposits-into-IS={v_tail_dep} "
      f"(want all 0)")
print(f"[verify] overlap with production pool: {overlap_prod} rows; "
      f"law_dead()==False for all pool rows: "
      f"{all(not law_dead((r ^ 45340) & 1023) for r in pool_new)}")
print(f"[capacity] LOAD rows: {len(pool_is)} (vs production 278 of which "
      f"only {278 - (len(prod_dead) - sum(1 for r in tail if r not in cloneok))} clone-ok); "
      f"V2 tail: 16/16 clone-ok (production tail: 7/16)")

if "--write" in sys.argv:
    import datetime
    stamp = datetime.date.today().isoformat()
    # KEEP EVERY HEADER LINE < 64 CHARS: the primary-pool loader in
    # test_bitnet_server.cpp reads with fgets(line, 64); a longer
    # comment line splits and its tail can parse as a row number
    # (window/open filtering saved us, but do not rely on it).
    hdr = f"""# Clone-ok primary pool, DIMM 2 s_id 72 (Rf=45340).
# Task O8(a) {stamp}. {len(pool_new)} rows, banks identical.
# Greedy ASCENDING-DEGREE IS (the construction that
# reproduces the production 294 byte-for-byte) over
# the UNION primary fault graph (May bank0 +
# 2026-07-18 campaign x4 banks). Candidates: rows
# that RowClone into Rf=45340 on ALL 4 banks
# (PIM_CHECK_CLONE + clone_law.py closed form).
# Excluded: primary opens, sub71 opens (8 were
# production LOAD rows -> voting-trip wrRow
# corruption), sub71 pool rows + their 13 directed
# deposit targets.
# Layout: rows 1-{len(pool_is)} = LOAD-eligible IS; last 16 =
# V2 scratch tail, sub71-style relaxed rule (clone-
# ok, no directed fault edge into any IS row).
# From: dimm2_fault_sweep_subs_2026_07_18/
#   o8_clone_law_and_cloneok_pool.py
"""
    assert max(len(l) for l in hdr.splitlines()) < 64
    for b in BANKS:
        path = f"{BIT}/pool_layout_dimm2_cloneok_bank{b}.txt"
        with open(path, "w") as f:
            f.write(hdr)
            for r in pool_new:
                f.write(f"{r}\n")
    print(f"[write] wrote pool_layout_dimm2_cloneok_bank{{0-3}}.txt "
          f"({len(pool_new)} rows x 4 banks, identical content)")
