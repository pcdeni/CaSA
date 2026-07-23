#!/usr/bin/env python3
"""Producer-loop divergence isolator (2026-07-23): LOAD -> MM3D(exact)
-> V2 streamed sessions (exact-checked) -> MM3D again (exact?).

Answers, per PIM_STREAM arm:
  (a) are V2 responses themselves exact when the request runs in a
      stream session with LOAD residents present?
  (b) do V2 sessions damage SUBSEQUENT MM3D (resident-row) results?

Same seed => byte-identical requests across arms. Reference math and
mask generation copied from ab_fused_server.py (the exact-reference
driver)."""
import os, random, struct, subprocess, sys, time

MAGIC_V2   = 0xB17EF002
MAGIC_LOAD = 0xB17EF003
MAGIC_MM3D = 0xB17EF004
D_OUT = 2048

def main():
    bender, calib, banks = "2", sys.argv[1], sys.argv[2]
    d_in, bitplanes, seed = 256, 4, 1
    n_v2 = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    server = sys.argv[4]
    log_path = sys.argv[5]
    n_chunks = d_in // 32
    rng = random.Random(seed)

    pos = [[0] * D_OUT for _ in range(n_chunks)]
    neg = [[0] * D_OUT for _ in range(n_chunks)]
    for c in range(n_chunks):
        for s in range(D_OUT):
            p = rng.getrandbits(32)
            n = rng.getrandbits(32) & ~p
            thin = rng.getrandbits(32)
            pos[c][s] = p & thin
            neg[c][s] = n & (thin ^ 0xFFFFFFFF) & rng.getrandbits(32)

    log_f = open(log_path, "wb")
    srv = subprocess.Popen([server, bender, calib, banks],
                           stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                           stderr=log_f, env=dict(os.environ))

    def send(body):
        srv.stdin.write(struct.pack("<I", len(body)))
        srv.stdin.write(body); srv.stdin.flush()

    def read_exact(n):
        buf = b""
        while len(buf) < n:
            ch = srv.stdout.read(n - len(buf))
            if not ch: raise RuntimeError("server closed stdout")
            buf += ch
        return buf

    def ref_y(xbp, factors):
        out = []
        for s in range(D_OUT):
            r = 0
            for c in range(n_chunks):
                pc, nc, xc = pos[c][s], neg[c][s], xbp[c]
                for b in range(len(factors)):
                    r += factors[b] * (bin(pc & xc[b]).count("1")
                                       - bin(nc & xc[b]).count("1"))
            out.append(r)
        return out

    def mm3d_round(tag, count):
        ok_all = True
        for req in range(count):
            xbp = [[rng.getrandbits(32) for _ in range(bitplanes)]
                   for _ in range(n_chunks)]
            factors = [1 << b for b in range(bitplanes)]
            body = struct.pack("<5I", MAGIC_MM3D, 1, D_OUT, n_chunks, bitplanes)
            for c in range(n_chunks):
                body += struct.pack("<%dI" % bitplanes, *xbp[c])
            body += struct.pack("<%di" % bitplanes, *factors)
            send(body)
            y = struct.unpack("<%di" % D_OUT, read_exact(D_OUT * 4))
            ref = ref_y(xbp, factors)
            bad = sum(1 for s in range(D_OUT) if ref[s] != y[s])
            print("[probe] %s mm3d %d: exact=%d/%d" % (tag, req, D_OUT - bad, D_OUT),
                  flush=True)
            ok_all &= (bad == 0)
        return ok_all

    def v2_round(tag, count):
        ok_all = True
        for req in range(count):
            xbp = [[rng.getrandbits(32) for _ in range(bitplanes)]
                   for _ in range(n_chunks)]
            factors = [1 << b for b in range(bitplanes)]
            body = struct.pack("<5I", MAGIC_V2, d_in, D_OUT, n_chunks, bitplanes)
            for c in range(n_chunks):
                body += struct.pack("<%dI" % D_OUT, *pos[c])
            for c in range(n_chunks):
                body += struct.pack("<%dI" % D_OUT, *neg[c])
            for c in range(n_chunks):
                body += struct.pack("<%dI" % bitplanes, *xbp[c])
            body += struct.pack("<%di" % bitplanes, *factors)
            send(body)
            y = struct.unpack("<%di" % D_OUT, read_exact(D_OUT * 4))
            ref = ref_y(xbp, factors)
            bad = sum(1 for s in range(D_OUT) if ref[s] != y[s])
            print("[probe] %s v2 %d: exact=%d/%d" % (tag, req, D_OUT - bad, D_OUT),
                  flush=True)
            ok_all &= (bad == 0)
        return ok_all

    # LOAD residents
    body = struct.pack("<5I", MAGIC_LOAD, 1, d_in, D_OUT, n_chunks)
    for c in range(n_chunks):
        body += struct.pack("<%dI" % D_OUT, *pos[c])
    for c in range(n_chunks):
        body += struct.pack("<%dI" % D_OUT, *neg[c])
    send(body)
    ack = struct.unpack("<I", read_exact(4))[0]
    print("[probe] LOAD ack=%d" % ack, flush=True)
    if ack != 0: sys.exit(2)

    ok1 = mm3d_round("pre ", 4)
    okv = v2_round("mid ", n_v2)
    ok2 = mm3d_round("post", 4)
    print("[probe] VERDICT: mm3d_pre=%s v2=%s mm3d_post=%s"
          % (ok1, okv, ok2), flush=True)

    srv.stdin.write(struct.pack("<I", 0)); srv.stdin.flush()
    srv.wait(timeout=30)
    log_f.close()
    sys.exit(0 if (ok1 and okv and ok2) else 1)

main()
