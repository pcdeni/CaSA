#!/usr/bin/env python3
"""O8(c): resident-row drift isolation — one-variable arms.

Protocol per session (fresh server each time, PRODUCTION pool, the
environment where O1 observed the drift):
  1. LOAD n_loads 1-round handles (d_in=64 -> n_chunks=2 -> 4 units ->
     1 round; round-0 verify covers every row the handle owns).
  2. Verify pass 1 (MM3D each handle once) — baseline right after LOAD.
  3. Treatment (one variable):
       idle      : sleep --idle-s seconds
       mm3d      : hammer MM3D x N on handle 0 (body traffic in the
                   primary subarray; PIM_FUSED_COSET decides body type)
       v2prim    : hammer V2 x N at calib_idx=0 (primary-tuple bodies,
                   scratch = production V2 tail, same subarray)
       v2sub84   : hammer V2 x N at calib_idx=1 (sub84 tuple, bodies in
                   the s86 subarray — different subarray, same banks)
       v2sub71   : hammer V2 x N at calib_idx=2 (sub71 tuple, bodies in
                   the SAME subarray; its opens include 8 production
                   pool rows — positive control for open-overlap writes)
       none      : no treatment (back-to-back verify passes)
  4. Verify pass 2 (MM3D each handle once).
  5. Summary: per-handle mismatch pass1 vs pass2 parsed from the server
     log (a handle with no DECAY line in a pass is clean — DECAY lines
     always print; OK lines are subsampled).

Env knobs passed through: PIM_FUSED_COSET, PIM_REFRESH.
"""
import os, random, re, struct, subprocess, sys, time

MAGIC_V2, MAGIC_LOAD, MAGIC_MM3D = 0xB17EF002, 0xB17EF003, 0xB17EF004
D_OUT = 2048
BITNET = ("/home/deni/Claude/SiMRA-DRAM-main/DRAM-Bender/sources/apps/"
          "DSN_AE_APPS/BitNet")

def popcount(x): return bin(x).count("1")

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    choices=["idle", "mm3d", "v2prim", "v2sub84",
                             "v2sub71", "none"])
    ap.add_argument("--n-loads", type=int, default=24)
    ap.add_argument("--n-loads-b", type=int, default=0,
                    help="second LOAD batch AFTER verify pass 1 (drives "
                         "the pool into overflow so the multi-window "
                         "refresh — and its label collisions — only "
                         "exists for pass 2)")
    ap.add_argument("--hammer-n", type=int, default=200)
    ap.add_argument("--idle-s", type=int, default=300)
    ap.add_argument("--bitplanes", type=int, default=4)
    ap.add_argument("--d-in", type=int, default=64)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--log", required=True)
    ap.add_argument("--sim", action="store_true")
    a = ap.parse_args()
    n_chunks = a.d_in // 32
    rng = random.Random(a.seed)

    env = dict(os.environ)
    if a.sim:
        env["PIM_BACKEND"] = "sim"
    env.setdefault("PIM_SUB_START", "45312")
    env.setdefault("PIM_SUB_END", "45952")
    env.setdefault("PIM_POOL_LIST_FILE",
                   f"{BITNET}/pool_layout_dimm2_bank{{bank}}.txt")
    env.setdefault("PIM_POOL_LIST_FILE_SUB",
                   f"{BITNET}/pool_layout_dimm2_sub{{sub}}_bank{{bank}}.txt")
    env.setdefault("PIM_RECV_TIMEOUT_MS", "15000")
    env.setdefault("BITSTREAM_IMEM", "8192")
    env.setdefault("PIM_VERIFY_LOAD", "1")

    log_f = open(a.log, "wb")
    srv = subprocess.Popen([f"{BITNET}/bitnet-proj-server", "2",
                            f"{BITNET}/calib_dimm2.txt", "0,1,2,3"],
                           stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                           stderr=log_f, env=env, cwd=BITNET)

    def send(body):
        srv.stdin.write(struct.pack("<I", len(body)))
        srv.stdin.write(body)
        srv.stdin.flush()

    def read_exact(n):
        buf = b""
        while len(buf) < n:
            c = srv.stdout.read(n - len(buf))
            if not c:
                raise RuntimeError(f"server closed stdout (see {a.log})")
            buf += c
        return buf

    def gen_weights():
        pos = [[0] * D_OUT for _ in range(n_chunks)]
        neg = [[0] * D_OUT for _ in range(n_chunks)]
        for c in range(n_chunks):
            for s in range(D_OUT):
                p = rng.getrandbits(32)
                n = rng.getrandbits(32) & ~p
                thin = rng.getrandbits(32)
                pos[c][s] = p & thin
                neg[c][s] = n & (thin ^ 0xFFFFFFFF) & rng.getrandbits(32)
        return pos, neg

    factors = [1 << b for b in range(a.bitplanes)]

    def mm3d(hid, pos, neg, check=True):
        xbp = [[rng.getrandbits(32) for _ in range(a.bitplanes)]
               for _ in range(n_chunks)]
        body = struct.pack("<5I", MAGIC_MM3D, hid, D_OUT, n_chunks,
                           a.bitplanes)
        for c in range(n_chunks):
            body += struct.pack(f"<{a.bitplanes}I", *xbp[c])
        body += struct.pack(f"<{a.bitplanes}i", *factors)
        send(body)
        y = struct.unpack(f"<{D_OUT}i", read_exact(D_OUT * 4))
        if not check:
            return -1
        bad = 0
        for s in range(D_OUT):
            ref = 0
            for c in range(n_chunks):
                pc, nc, xc = pos[c][s], neg[c][s], xbp[c]
                for b in range(a.bitplanes):
                    ref += factors[b] * (popcount(pc & xc[b])
                                         - popcount(nc & xc[b]))
            if ref != y[s]:
                bad += 1
        return bad

    def v2(calib_idx):
        pos, neg = gen_weights()
        xbp = [[rng.getrandbits(32) for _ in range(a.bitplanes)]
               for _ in range(n_chunks)]
        body = struct.pack("<5I", MAGIC_V2, a.d_in, D_OUT, n_chunks,
                           a.bitplanes)
        for c in range(n_chunks):
            body += struct.pack(f"<{D_OUT}I", *pos[c])
        for c in range(n_chunks):
            body += struct.pack(f"<{D_OUT}I", *neg[c])
        for c in range(n_chunks):
            body += struct.pack(f"<{a.bitplanes}I", *xbp[c])
        body += struct.pack(f"<{a.bitplanes}i", *factors)
        body += struct.pack("<I", calib_idx)
        send(body)
        read_exact(D_OUT * 4)

    # Server stderr is unbuffered (glibc stderr stays unbuffered when
    # redirected); every verify fprintf lands in the log BEFORE the
    # request's response bytes reach us. Byte offsets sampled after each
    # phase are therefore exact phase boundaries.
    def log_size():
        return os.fstat(log_f.fileno()).st_size

    handles = {}

    def load_batch(hid_from, n, tag):
        t0 = time.time()
        n_ok = 0
        for hid in range(hid_from, hid_from + n):
            pos, neg = gen_weights()
            body = struct.pack("<5I", MAGIC_LOAD, hid, a.d_in, D_OUT,
                               n_chunks)
            for c in range(n_chunks):
                body += struct.pack(f"<{D_OUT}I", *pos[c])
            for c in range(n_chunks):
                body += struct.pack(f"<{D_OUT}I", *neg[c])
            send(body)
            ack = struct.unpack("<I", read_exact(4))[0]
            if ack != 0:
                print(f"[arms] ENOSPC at handle {hid} ({tag})")
                break
            handles[hid] = (pos, neg)
            n_ok += 1
        print(f"[arms] batch {tag}: loaded {n_ok} 1-round handles in "
              f"{time.time()-t0:.1f}s")
        return n_ok

    load_batch(1, a.n_loads, "A")

    def verify_pass(tag):
        t = time.time()
        bads = {}
        for hid, (pos, neg) in handles.items():
            bads[hid] = mm3d(hid, pos, neg, check=True)
        n_bad = sum(1 for v in bads.values() if v)
        print(f"[arms] verify pass '{tag}': {len(bads)} handles, "
              f"{n_bad} with y-mismatch ({time.time()-t:.1f}s) "
              f"worst={max(bads.values())}")
        return bads

    off_loads = log_size()
    n_batch_a = len(handles)
    p1 = verify_pass("post-load")
    off_p1 = log_size()

    if a.n_loads_b > 0:
        load_batch(a.n_loads + 1, a.n_loads_b, "B")

    t = time.time()
    if a.arm == "idle":
        print(f"[arms] idling {a.idle_s}s...")
        time.sleep(a.idle_s)
    elif a.arm == "mm3d":
        hid = min(handles)
        pos, neg = handles[hid]
        for i in range(a.hammer_n):
            mm3d(hid, pos, neg, check=False)
        print(f"[arms] hammered MM3D x{a.hammer_n} on handle {hid} "
              f"({time.time()-t:.1f}s)")
    elif a.arm in ("v2prim", "v2sub84", "v2sub71"):
        ci = {"v2prim": 0, "v2sub84": 1, "v2sub71": 2}[a.arm]
        for i in range(a.hammer_n):
            v2(ci)
        print(f"[arms] hammered V2 x{a.hammer_n} at calib_idx={ci} "
              f"({time.time()-t:.1f}s)")
    elif a.arm == "none":
        pass

    off_treat = log_size()
    p2 = verify_pass("post-treatment")
    off_p2 = log_size()

    srv.stdin.write(struct.pack("<I", 0))
    srv.stdin.flush()
    srv.stdin.close()
    srv.wait(timeout=60)
    log_f.close()

    # Parse verify events per byte-offset region. DECAY lines always
    # print; a handle verified inside a region with no DECAY line there
    # is clean (OK prints are subsampled but DECAY is not).
    def parse_region(lo, hi):
        out = {}
        with open(a.log, "rb") as f:
            f.seek(lo)
            blob = f.read(hi - lo).decode(errors="replace")
        for line in blob.splitlines():
            m = re.search(r"\[mm3d-verify\] handle=(\d+) DECAY/CORRUPTION: "
                          r"(\d+)/(\d+) segs", line)
            if m:
                out[int(m.group(1))] = int(m.group(2))
                continue
            m = re.search(r"\[mm3d-verify\] handle=(\d+) round-0 "
                          r"popcounts OK", line)
            if m:
                out.setdefault(int(m.group(1)), 0)
        return out
    r1 = parse_region(off_loads, off_p1)
    r2 = parse_region(off_treat, off_p2)
    hs = sorted(handles)
    print(f"[arms] {'h':>4} {'pass1':>6} {'pass2':>6}  (segs/8192 differ; "
          f"0 = clean; batch-B handles have no pass1)")
    n_drift = 0
    lvl2 = []
    for h in hs:
        v1 = r1.get(h, 0)
        v2_ = r2.get(h, 0)
        is_b = h > n_batch_a
        if not is_b and v2_ > v1 + 50:
            n_drift += 1
        if v2_ > 50:
            lvl2.append(v2_)
        if v1 or v2_ or h <= 4 or is_b and v2_:
            print(f"[arms] {h:>4} {('B' if is_b else v1):>6} {v2_:>6}")
    import statistics as st
    print(f"[arms] RESULT arm={a.arm} fused={env.get('PIM_FUSED_COSET','?')}"
          f" refresh={env.get('PIM_REFRESH','1(dflt)')} "
          f"loadsA={n_batch_a} loadsB={len(hs)-n_batch_a}: "
          f"{n_drift}/{n_batch_a} batch-A handles transitioned "
          f"(>50 segs growth); dirty-pass2 handles={len(lvl2)}"
          + (f" median-level={st.median(lvl2):.0f}/8192" if lvl2 else "")
          + f"; y-mismatch p1={sum(1 for v in p1.values() if v)}"
            f" p2={sum(1 for v in p2.values() if v)}")

if __name__ == "__main__":
    main()
