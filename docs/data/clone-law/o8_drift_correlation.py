#!/usr/bin/env python3
"""O8(c) host-side: correlate O1's [mm3d-verify] drift prints with pool
geometry — which rows does each handle own, which handles are dense vs
sparse, does the dense cohort align with (i) clone-dead round-0 rows,
(ii) sub71-open overlap rows, (iii) LOAD-order position, (iv) when the
first corrupt verify happened (drift vs load-time)."""
import re
from collections import defaultdict

D   = "/home/deni/Claude/dimm2_fault_sweep_subs_2026_07_18"
BIT = ("/home/deni/Claude/SiMRA-DRAM-main/DRAM-Bender/sources/apps/"
       "DSN_AE_APPS/BitNet")
LOG = f"{D}/fullmodel_o1_fused_server.log"

prod = [int(l) for l in open(f"{BIT}/pool_layout_dimm2_bank0.txt")
        if l.strip() and not l.startswith("#")]
pools_extra = {}
for sub in (84, 71, 77, 76):
    pools_extra[sub] = [int(l) for l in
                        open(f"{BIT}/pool_layout_dimm2_sub{sub}_bank0.txt")
                        if l.strip() and not l.startswith("#")]

# primary clone status
clone_ok_prim = None
for b in range(4):
    s = set()
    for line in open(f"{D}/primary_bank{b}.log"):
        m = re.match(r"^CLONE R=(\d+) match=\d+ ok=(\d)", line)
        if m and m.group(2) == "1":
            s.add(int(m.group(1)))
    clone_ok_prim = s if clone_ok_prim is None else (clone_ok_prim & s)

SUB71_OPEN = {45440,45441,45464,45465,45472,45473,45496,45497,
              45824,45825,45848,45849,45856,45857,45880,45881}

# Parse LOAD lines to reconstruct handle -> rows.
# "[server] LOAD_WEIGHTS handle=%u n_chunks=%u rounds=%zu pool_cursor[0]=%zu"
# pool_cursor printed AFTER allocation; primary rows = prod[cur-rounds:cur]
# ... except overflow handles: cursor only advances by the primary part.
load_info = {}
order = []
prev_cursor = 0
overflow_note = {}
for line in open(LOG):
    m = re.search(r"LOAD_WEIGHTS handle=(\d+) n_chunks=(\d+) rounds=(\d+) "
                  r"pool_cursor\[0\]=(\d+)", line)
    if m:
        h, nc, rd, cur = map(int, m.groups())
        prim_taken = cur - prev_cursor
        load_info[h] = dict(rounds=rd, cur0=prev_cursor, prim=prim_taken,
                            over=rd - prim_taken)
        order.append(h)
        prev_cursor = cur
print(f"handles: {len(load_info)}; total primary rounds "
      f"{sum(v['prim'] for v in load_info.values())}, overflow rounds "
      f"{sum(v['over'] for v in load_info.values())}")

# round-0 row per handle (what mm3d-verify reads): primary if prim>0.
for h, v in load_info.items():
    v["rows_prim"] = prod[v["cur0"]:v["cur0"] + v["prim"]]
    v["r0"] = v["rows_prim"][0] if v["rows_prim"] else None  # None=overflow

# Parse verify outcomes in order.
verifies = defaultdict(list)   # h -> [mismatch_segs or 0]
n_verify_lines = 0
for line in open(LOG):
    m = re.search(r"\[mm3d-verify\] handle=(\d+) DECAY/CORRUPTION: "
                  r"(\d+)/(\d+) segs", line)
    if m:
        verifies[int(m.group(1))].append(int(m.group(2)))
        n_verify_lines += 1
        continue
    m = re.search(r"\[mm3d-verify\] handle=(\d+) round-0 popcounts OK", line)
    if m:
        verifies[int(m.group(1))].append(0)
        n_verify_lines += 1
print(f"verify observations: {n_verify_lines} on {len(verifies)} handles "
      f"(OK prints are subsampled 1/50 after the first 3)")

print(f"{'h':>3} {'r0 row':>7} {'cl-ok':>5} {'s71op':>5} {'#prim':>5} "
      f"{'#over':>5} {'first-mm':>8} {'last-mm':>8} {'n':>3}  class")
n_dense = n_sparse = n_clean = 0
dense_r0, sparse_r0, clean_r0 = [], [], []
for h in order:
    v = load_info[h]
    obs = verifies.get(h, [])
    if not obs: continue
    first, last = obs[0], obs[-1]
    frac = last / 8192.0
    cls = ("DENSE" if frac > 0.8 else
           "sparse" if frac > 0.005 else "clean")
    if cls == "DENSE": n_dense += 1; dense_r0.append(v["r0"])
    elif cls == "sparse": n_sparse += 1; sparse_r0.append(v["r0"])
    else: n_clean += 1; clean_r0.append(v["r0"])
    r0 = v["r0"]
    print(f"{h:>3} {str(r0):>7} "
          f"{'-' if r0 is None else ('y' if r0 in clone_ok_prim else 'DEAD'):>5} "
          f"{'-' if r0 is None else ('Y' if r0 in SUB71_OPEN else '.'):>5} "
          f"{v['prim']:>5} {v['over']:>5} {first:>8} {last:>8} "
          f"{len(obs):>3}  {cls}")
print(f"\nsummary: DENSE={n_dense} sparse={n_sparse} clean={n_clean}")

# Dense-cohort r0 clone status
def st(rows):
    rows = [r for r in rows if r is not None]
    dead = sum(1 for r in rows if r not in clone_ok_prim)
    s71 = sum(1 for r in rows if r in SUB71_OPEN)
    return f"n={len(rows)} r0-clone-dead={dead} r0-in-sub71open={s71}"
print(f"DENSE  cohort: {st(dense_r0)}")
print(f"sparse cohort: {st(sparse_r0)}")
print(f"clean  cohort: {st(clean_r0)}")

# All rows of dense handles: how many are clone-dead / sub71-open?
for name, cohort in (("DENSE", [h for h in order
                                if verifies.get(h) and
                                verifies[h][-1] / 8192.0 > 0.8]),):
    rows = []
    for h in cohort:
        rows += load_info[h]["rows_prim"]
    dead = sum(1 for r in rows if r not in clone_ok_prim)
    s71 = sum(1 for r in rows if r in SUB71_OPEN)
    print(f"{name} handles' ALL primary rows: {len(rows)}; clone-dead "
          f"{dead}; sub71-open {s71}")
