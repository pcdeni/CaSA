#!/usr/bin/env python3
"""replay_ab.py — seconds-scale A/B on CAPTURED production requests.

Replays a PIM_REQ_CAPTURE file (u32 body_len, u32 resp_len, body)*
against fresh bitnet-proj-server instances under different env arms and
compares responses request-by-request. Fixes mixed_probe.py's flaw
(synthetic bodies failed the legacy control): these are the real
client's bytes, LOAD uploads included, in production order.

Usage:
  replay_ab.py <capture> <arm1> <arm2> [--limit N] [--tag T]
where an arm is  name:ENV=V,ENV=V,...   e.g.
  legacy:PIM_STREAM=0    stream:PIM_STREAM=1
  wcol:PIM_STREAM=1,PIM_STREAM_SCOPE=wcol
Base env (dimm2 trio + production flags) is applied to every arm.
"""
import os, struct, subprocess, sys
from collections import Counter

BN = "/home/deni/Claude/SiMRA-DRAM-main/DRAM-Bender/sources/apps/DSN_AE_APPS/BitNet"
SERVER = f"{BN}/bitnet-proj-server"
BASE_ENV = {
    "BITSTREAM_IMEM": "8192", "PIM_FUSED_COSET": "1",
    "PIM_PARALLEL_BANKS": "0", "PIM_SEGPOP": "1",
    "PIM_SUB_START": "45312", "PIM_SUB_END": "45952",
    "PIM_POOL_LIST_FILE": f"{BN}/pool_layout_dimm2_bank{{bank}}.txt",
}
BENDER, CALIB, BANKS = "2", f"{BN}/calib_dimm2.txt", "0,1,2,3"

def load_capture(path, limit):
    recs = []
    with open(path, "rb") as f:
        while True:
            h = f.read(8)
            if len(h) < 8: break
            blen, rlen = struct.unpack("<II", h)
            body = f.read(blen)
            if len(body) < blen: break
            recs.append((body, rlen))
            if limit and len(recs) >= limit: break
    return recs

def run_arm(name, envkv, recs, tag):
    env = dict(os.environ); env.update(BASE_ENV); env.update(envkv)
    log = open(f"/home/deni/Claude/roadb_build9_2026_07_22/replay_{tag}_{name}.log", "w")
    p = subprocess.Popen([SERVER, BENDER, CALIB, BANKS],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=log, bufsize=0, env=env)
    out = []
    try:
        for i, (body, rlen) in enumerate(recs):
            p.stdin.write(struct.pack("<I", len(body))); p.stdin.write(body)
            p.stdin.flush()
            got = b""
            while len(got) < rlen:
                ch = p.stdout.read(rlen - len(got))
                if not ch:
                    raise RuntimeError(f"[{name}] server closed at req {i}")
                got += ch
            out.append(got)
    finally:
        try:
            p.stdin.write(struct.pack("<I", 0)); p.stdin.flush()
            p.wait(timeout=30)
        except Exception:
            p.kill()
        log.close()
    return out

V2_MAGICS = {0xB17EF002, 0xB17EF005, 0xB17EF006, 0xB17EF007}
MM3D_MAGIC = 0xB17EF004

def run_dup(recs, tag, envkv=None):
    """SAME-PROCESS A/B: one server with PIM_STREAM=1 and
    PIM_STREAM_ALTERNATE=1; every V2-family request is sent TWICE so
    the twins hit (legacy, streamed) back-to-back inside one process.
    This comparison sits below the cross-process odd-segment floor.
    Override env via an arm spec (e.g. control:PIM_STREAM=0 makes BOTH
    twins legacy — the same-process rerun-determinism control)."""
    env = dict(os.environ); env.update(BASE_ENV)
    env.update({"PIM_STREAM": "1", "PIM_STREAM_ALTERNATE": "1"})
    if envkv: env.update(envkv)
    log = open(f"/home/deni/Claude/roadb_build9_2026_07_22/replay_{tag}_dup.log", "w")
    p = subprocess.Popen([SERVER, BENDER, CALIB, BANKS],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=log, bufsize=0, env=env)
    def one(body, rlen):
        p.stdin.write(struct.pack("<I", len(body))); p.stdin.write(body)
        p.stdin.flush()
        got = b""
        while len(got) < rlen:
            ch = p.stdout.read(rlen - len(got))
            if not ch: raise RuntimeError("server closed")
            got += ch
        return got
    dup_mm = os.environ.get("REPLAY_DUP_MM3D", "0") == "1"
    ndiff = 0; nv2 = 0; first = None; par_all = Counter(); nel = 0
    by_magic = Counter(); by_magic_ok = Counter()
    try:
        for i, (body, rlen) in enumerate(recs):
            magic = struct.unpack("<I", body[:4])[0]
            twin = magic in V2_MAGICS or (dup_mm and magic == MM3D_MAGIC)
            if not twin:
                one(body, rlen); continue
            ra = one(body, rlen)   # legacy (even counter)
            rb = one(body, rlen)   # streamed (odd counter; MM3D: legacy)
            nv2 += 1
            if ra != rb:
                ndiff += 1
                by_magic[magic] += 1
                ya = struct.unpack(f"<{len(ra)//4}i", ra)
                yb = struct.unpack(f"<{len(rb)//4}i", rb)
                d = [j for j in range(len(ya)) if ya[j] != yb[j]]
                for j in d: par_all[j % 2] += 1
                nel += len(d)
                if first is None:
                    first = i
                    print(f"[dup] FIRST TWIN DIFF req #{i} magic {magic:08x}: "
                          f"{len(d)}/{len(ya)} el parity="
                          f"{dict(Counter(j % 2 for j in d))}")
                    print(f"      idx {d[:10]}")
                    print(f"      vals {[(ya[j], yb[j]) for j in d[:6]]}")
            else:
                by_magic_ok[magic] += 1
    finally:
        try:
            p.stdin.write(struct.pack("<I", 0)); p.stdin.flush()
            p.wait(timeout=30)
        except Exception:
            p.kill()
        log.close()
    print(f"[dup] SAME-PROCESS twins: {ndiff}/{nv2} differ, first={first}, "
          f"elements={nel} parity={dict(par_all)} "
          f"{'** EXACT **' if ndiff == 0 else ''}")
    for m in sorted(set(by_magic) | set(by_magic_ok)):
        print(f"      magic {m:08x}: {by_magic[m]} differ / "
              f"{by_magic[m] + by_magic_ok[m]} twins")

def main():
    cap = sys.argv[1]
    arms = []
    limit = 0; tag = "r"; dup = False
    i = 2
    while i < len(sys.argv):
        a = sys.argv[i]
        if a == "--limit": limit = int(sys.argv[i+1]); i += 2; continue
        if a == "--tag": tag = sys.argv[i+1]; i += 2; continue
        if a == "--dup": dup = True; i += 1; continue
        name, _, kvs = a.partition(":")
        envkv = dict(kv.split("=", 1) for kv in kvs.split(",") if kv)
        arms.append((name, envkv)); i += 1
    recs = load_capture(cap, limit)
    magics = Counter(struct.unpack("<I", b[:4])[0] for b, _ in recs)
    print(f"[replay] {len(recs)} requests  magics=" +
          " ".join(f"{m:08x}:{c}" for m, c in sorted(magics.items())))
    if dup:
        run_dup(recs, tag, arms[0][1] if arms else None)
        return
    results = {}
    for name, envkv in arms:
        print(f"[replay] arm {name} ({envkv}) ...", flush=True)
        results[name] = run_arm(name, envkv, recs, tag)
    a, b = arms[0][0], arms[1][0]
    A, B = results[a], results[b]
    ndiff = 0; first = None
    for i in range(len(recs)):
        if A[i] != B[i]:
            ndiff += 1
            if first is None:
                first = i
                ya = struct.unpack(f"<{len(A[i])//4}i", A[i])
                yb = struct.unpack(f"<{len(B[i])//4}i", B[i])
                d = [j for j in range(len(ya)) if ya[j] != yb[j]]
                par = Counter(j % 2 for j in d)
                print(f"[replay] FIRST DIFF req #{i} "
                      f"(magic {struct.unpack('<I', recs[i][0][:4])[0]:08x}): "
                      f"{len(d)}/{len(ya)} el  parity={dict(par)}")
                print(f"         idx {d[:10]}")
                print(f"         vals {[(ya[j], yb[j]) for j in d[:6]]}")
    # per-parity totals across ALL diffs
    par_all = Counter(); nel = 0
    for i in range(len(recs)):
        if A[i] != B[i] and len(A[i]) == len(B[i]) and len(A[i]) % 4 == 0:
            ya = struct.unpack(f"<{len(A[i])//4}i", A[i])
            yb = struct.unpack(f"<{len(B[i])//4}i", B[i])
            for j in range(len(ya)):
                if ya[j] != yb[j]: par_all[j % 2] += 1; nel += 1
    print(f"[replay] {a} vs {b}: {ndiff}/{len(recs)} requests differ, "
          f"first={first}, elements={nel} parity={dict(par_all)}")

main()
