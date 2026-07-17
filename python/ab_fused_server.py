#!/usr/bin/env python3
"""A/B driver for bitnet-proj-server (LOAD_WEIGHTS + MATMUL_HANDLE).

Generates synthetic ternary weights (d_out=2048 x d_in as disjoint pos/neg
masks) and activation bitplanes, drives the server over its binary
stdin/stdout protocol, and checks every returned y against the exact
host-side reference:

    y[s] = sum_c sum_b factor[b] * ( popcount(pos[c][s] & xbp[c][b])
                                   - popcount(neg[c][s] & xbp[c][b]) )

Same --seed => byte-identical requests, so a baseline run and a --fused run
(PIM_FUSED_COSET=1) are directly comparable. Server stderr goes to
--log (default ab_server_<mode>.log).

Usage (from the BitNet dir; server env is inherited — set PIM_SUB_START/
PIM_SUB_END/PIM_POOL_LIST_FILE/PIM_RECV_TIMEOUT_MS outside):
  python3 ab_fused_server.py --bender 2 --calib calib_dimm2.txt --banks 0 \
      --d-in 256 --bitplanes 4 --matmuls 8 --seed 1 [--fused]
"""
import argparse, os, random, struct, subprocess, sys, time

MAGIC_LOAD = 0xB17EF003
MAGIC_MM3D = 0xB17EF004
D_OUT = 2048


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bender", type=int, default=2)
    ap.add_argument("--calib", default="calib_dimm2.txt")
    ap.add_argument("--banks", default="0")
    ap.add_argument("--d-in", type=int, default=256, help="multiple of 32")
    ap.add_argument("--bitplanes", type=int, default=4)
    ap.add_argument("--matmuls", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--fused", action="store_true",
                    help="set PIM_FUSED_COSET=1 in the server env")
    ap.add_argument("--server", default="./bitnet-proj-server")
    ap.add_argument("--log", default=None)
    a = ap.parse_args()
    assert a.d_in % 32 == 0
    n_chunks = a.d_in // 32
    mode = "fused" if a.fused else "baseline"
    log_path = a.log or f"ab_server_{mode}.log"
    rng = random.Random(a.seed)

    # Ternary weights as disjoint pos/neg 32-bit masks per (chunk, segment).
    pos = [[0] * D_OUT for _ in range(n_chunks)]
    neg = [[0] * D_OUT for _ in range(n_chunks)]
    for c in range(n_chunks):
        for s in range(D_OUT):
            p = rng.getrandbits(32)
            n = rng.getrandbits(32) & ~p
            thin = rng.getrandbits(32)
            pos[c][s] = p & thin
            neg[c][s] = n & (thin ^ 0xFFFFFFFF) & rng.getrandbits(32)

    env = dict(os.environ)
    if a.fused:
        env["PIM_FUSED_COSET"] = "1"
    log_f = open(log_path, "wb")
    srv = subprocess.Popen([a.server, str(a.bender), a.calib, a.banks],
                           stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                           stderr=log_f, env=env)

    def send(body):
        srv.stdin.write(struct.pack("<I", len(body)))
        srv.stdin.write(body)
        srv.stdin.flush()

    def read_exact(n):
        buf = b""
        while len(buf) < n:
            chunk = srv.stdout.read(n - len(buf))
            if not chunk:
                raise RuntimeError("server closed stdout (see %s)" % log_path)
            buf += chunk
        return buf

    body = struct.pack("<5I", MAGIC_LOAD, 1, a.d_in, D_OUT, n_chunks)
    for c in range(n_chunks):
        body += struct.pack("<%dI" % D_OUT, *pos[c])
    for c in range(n_chunks):
        body += struct.pack("<%dI" % D_OUT, *neg[c])
    t0 = time.monotonic()
    send(body)
    ack = struct.unpack("<I", read_exact(4))[0]
    t_load = (time.monotonic() - t0) * 1e3
    if ack != 0:
        print("[ab] LOAD ack=%d (pool exhausted?)" % ack)
        sys.exit(2)
    print("[ab] LOAD ok: d_in=%d n_chunks=%d  %.0f ms" %
          (a.d_in, n_chunks, t_load))

    factors = [1 << b for b in range(a.bitplanes)]
    times, all_ok = [], True
    for req in range(a.matmuls):
        xbp = [[rng.getrandbits(32) for _ in range(a.bitplanes)]
               for _ in range(n_chunks)]
        body = struct.pack("<5I", MAGIC_MM3D, 1, D_OUT, n_chunks, a.bitplanes)
        for c in range(n_chunks):
            body += struct.pack("<%dI" % a.bitplanes, *xbp[c])
        body += struct.pack("<%di" % a.bitplanes, *factors)
        t0 = time.monotonic()
        send(body)
        y = struct.unpack("<%di" % D_OUT, read_exact(D_OUT * 4))
        dt = (time.monotonic() - t0) * 1e3
        times.append(dt)
        bad, first = 0, None
        for s in range(D_OUT):
            ref = 0
            for c in range(n_chunks):
                pc, nc, xc = pos[c][s], neg[c][s], xbp[c]
                for b in range(a.bitplanes):
                    ref += factors[b] * (bin(pc & xc[b]).count("1")
                                         - bin(nc & xc[b]).count("1"))
            if ref != y[s]:
                bad += 1
                if first is None:
                    first = (s, ref, y[s])
        ok = bad == 0
        all_ok &= ok
        nz = sum(1 for v in y if v != 0)
        line = ("[ab] mm3d %d: %7.1f ms   exact=%d/%d   y_nonzero=%d max|y|=%d"
                % (req, dt, D_OUT - bad, D_OUT, nz, max(abs(v) for v in y)))
        if not ok:
            line += "   first_mismatch s=%d ref=%d got=%d" % first
        print(line, flush=True)

    srv.stdin.write(struct.pack("<I", 0))   # quit sentinel
    srv.stdin.flush()
    srv.wait(timeout=30)
    log_f.close()
    n = len(times)
    mean = sum(times) / n
    med = sorted(times)[n // 2]
    print("[ab] %s: %d matmuls  mean %.1f ms  median %.1f ms  all_exact=%s"
          % (mode.upper(), n, mean, med, all_ok))
    sys.exit(0 if all_ok else 1)


main()
