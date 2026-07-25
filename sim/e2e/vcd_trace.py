#!/usr/bin/env python3
"""Where does the first flush come from?

Walks the VCD in time order and prints, for the first N events:
  - every softmc_fin rising edge   (a program finished)
  - every flush rising edge        (fin + 32, the framing event)
  - every trailer beat transferred (the record delimiter on the wire)
with an ACCURATE c2h byte offset: transfers are counted once per clock
cycle (Verilator dumps twice per cycle: t*10 and t*10+5).
"""
import sys

VCD = sys.argv[1] if len(sys.argv) > 1 else "e2e.vcd"
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 40

want_exact = {"flush", "tlast", "c2h_tvalid_0", "c2h_tready_0", "rbf_empty",
              "proc_flush_r", "flush_pend_r", "softmc_fin", "flush_is_maint",
              "exec_bank_r", "loaded_r", "swap_pending_r", "fetch_hold_r"}
ids = {}
scope = []
with open(VCD) as f:
    for line in f:
        line = line.strip()
        if line.startswith("$scope"):
            scope.append(line.split()[2])
        elif line.startswith("$upscope"):
            if scope: scope.pop()
        elif line.startswith("$var"):
            p = line.split()
            ident, nm = p[3], p[4]
            full = ".".join(scope + [nm])
            if nm in want_exact:
                # prefer the rbe / frontend copies
                ids.setdefault(nm, []).append((ident, full))
        elif line.startswith("$enddefinitions"):
            break

    # pick one identifier per signal, preferring rbe then frontend then top
    pick = {}
    for nm, lst in ids.items():
        best = None
        for ident, full in lst:
            score = (3 if ".rbe." in full else 2 if ".frontend." in full else 1)
            if best is None or score > best[0]:
                best = (score, ident, full)
        pick[nm] = best[1]
    rev = {v: k for k, v in pick.items()}
    print("[vcd] signals:")
    for nm in sorted(pick):
        print(f"   {nm:18s} id={pick[nm]}")

    cur = {k: "0" for k in pick}
    t = 0
    cyc = -1
    bytes_out = 0
    prev = dict(cur)
    shown = 0
    for line in f:
        line = line.strip()
        if not line:
            continue
        if line[0] == "#":
            newt = int(line[1:])
            newcyc = newt // 10
            # one transfer per CLOCK CYCLE, not per dump
            if newcyc != cyc:
                if cur.get("c2h_tvalid_0") == "1" and cur.get("c2h_tready_0") == "1":
                    bytes_out += 32
                cyc = newcyc
            t = newt
            continue
        if line[0] in "01xz":
            val, ident = line[0], line[1:]
        elif line[0] in "bB":
            parts = line.split()
            val, ident = parts[0][1:], parts[1]
        else:
            continue
        if ident not in rev:
            continue
        nm = rev[ident]
        old = cur.get(nm, "0")
        cur[nm] = val
        if shown >= LIMIT:
            continue
        ev = None
        if nm == "softmc_fin" and old == "0" and val == "1":
            ev = "FIN   (program finished)"
        elif nm == "flush" and old == "0" and val == "1":
            ev = "FLUSH (fin+32 -> framing)"
        elif nm == "tlast" and val == "1":
            ev = "TLAST (record closed)"
        if ev:
            shown += 1
            print(f"cyc={t//10:<9} bytes={bytes_out:<7} {ev:26s} "
                  f"rbf_empty={cur.get('rbf_empty')} pf_r={cur.get('proc_flush_r')} "
                  f"pend={cur.get('flush_pend_r')} maint={cur.get('flush_is_maint')} "
                  f"exec_bank={cur.get('exec_bank_r')} loaded={cur.get('loaded_r')} "
                  f"swap={cur.get('swap_pending_r')} hold={cur.get('fetch_hold_r')}")
