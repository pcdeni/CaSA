#!/usr/bin/env python3
"""Lane-2 GeMV server smoke/verification client (phase 1).

Generates a random q-bit signed matrix W [M,K] and an r-bit activation
vector x [K] (r=1: binary {0,1} — the phase-1 verified mode), computes the
integer reference y = W @ x, drives the server over the length-prefixed
stdin/stdout protocol (LOAD_MATRIX 0x4D563001, GEMV 0x4D563002), and prints
per-output exact/near-exact stats + timing.

Usage:
  python3 lane2_client_smoke.py --K 4096 --M 4096 [--qbits 4] [--rbits 1]
      [--bender 2] [--bank 0] [--sid 86] [--seed 7] [--gemvs 1] [--vote3 0]

The server is launched as a subprocess with the mandatory silicon env
(BITSTREAM_IMEM=8192, PIM_RECV_TIMEOUT_MS=15000). Shutdown is always
graceful (quit sentinel + wait) — never SIGKILL anything holding XDMA.
"""
import argparse
import os
import struct
import subprocess
import sys
import time

import numpy as np

MAGIC_LOAD = 0x4D563001
MAGIC_GEMV = 0x4D563002
MAGIC_PART = 0x4D563003
MAGIC_LOAD_ACK = 0x4D5630F1
MAGIC_GEMV_ACK = 0x4D5630F2
MAGIC_PART_ACK = 0x4D5630F3

HERE = os.path.dirname(os.path.abspath(__file__))


def pack_bitplanes_matrix(W, qbits):
    """W int [M,K] -> qbits planes, each M rows x ceil(K/8) bytes, LSB-first.
    Bit i of two's-complement W[m][n]."""
    M, K = W.shape
    U = (W.astype(np.int64) & ((1 << qbits) - 1)).astype(np.uint8)  # two's complement q-bit
    out = bytearray()
    for i in range(qbits):
        bits = ((U >> i) & 1).astype(np.uint8)          # [M,K]
        out += np.packbits(bits, axis=1, bitorder="little").tobytes()
    return bytes(out)


def pack_bitplanes_vector(x, rbits):
    K = x.shape[0]
    U = (x.astype(np.int64) & ((1 << rbits) - 1)).astype(np.uint8)
    out = bytearray()
    for c in range(rbits):
        bits = ((U >> c) & 1).astype(np.uint8)
        out += np.packbits(bits[None, :], axis=1, bitorder="little").tobytes()
    return bytes(out)


def send_req(proc, body):
    proc.stdin.write(struct.pack("<I", len(body)))
    proc.stdin.write(body)
    proc.stdin.flush()


def read_exact(proc, n):
    buf = b""
    while len(buf) < n:
        chunk = proc.stdout.read(n - len(buf))
        if not chunk:
            raise RuntimeError("server closed stdout (crashed?) — check stderr above")
        buf += chunk
    return buf


def read_resp(proc):
    (ln,) = struct.unpack("<I", read_exact(proc, 4))
    return read_exact(proc, ln)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=4096)
    ap.add_argument("--M", type=int, default=4096)
    ap.add_argument("--qbits", type=int, default=4)
    ap.add_argument("--rbits", type=int, default=1)
    ap.add_argument("--bender", type=int, default=2)
    ap.add_argument("--bank", type=int, default=0)
    ap.add_argument("--sid", type=int, default=86)
    ap.add_argument("--calib", default=os.path.join(HERE, "calib_maj5_dimm2.txt"))
    ap.add_argument("--colmask", default=os.path.join(HERE, "colmask_dimm2_s86_robust.txt"))
    ap.add_argument("--server", default=os.path.join(HERE, "lane2-gemv-server"))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--gemvs", type=int, default=1, help="GEMV calls on the same handle")
    ap.add_argument("--vote3", type=int, default=0)
    ap.add_argument("--pack", type=int, default=1)
    ap.add_argument("--backend", default="", help="'sim' for SimDramModel dry-run")
    ap.add_argument("--dualtrack", type=int, default=0,
                    help="LANE2_DUALTRACK=1: Fig-15 dual-track (in-DRAM complements)")
    ap.add_argument("--encode", default="host", choices=["host", "clone"],
                    help="LANE2_ENCODE: clone = RowClone-encoded products (silicon-only)")
    ap.add_argument("--partials", type=int, default=0,
                    help="use GEMV_PARTIALS and verify per-32-block integer partials")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    if args.qbits == 1:
        # server convention (matches the paper's 1-bit operands): qb=1 is
        # UNSIGNED binary {0,1} — FAC(0,1)=+1, no sign plane.
        lo, hi = 0, 1
    else:
        lo, hi = -(1 << (args.qbits - 1)), (1 << (args.qbits - 1)) - 1
    W = rng.integers(lo, hi + 1, size=(args.M, args.K), dtype=np.int64)

    env = dict(os.environ)
    env["BITSTREAM_IMEM"] = "8192"
    env["PIM_RECV_TIMEOUT_MS"] = "15000"
    env["PIM_VOTE3"] = str(args.vote3)
    env["LANE2_PACK"] = str(args.pack)
    env["LANE2_DUALTRACK"] = str(args.dualtrack)
    env["LANE2_ENCODE"] = args.encode
    if args.backend:
        env["LANE2_BACKEND"] = args.backend

    cmd = [args.server, str(args.bender), args.calib, str(args.bank),
           str(args.sid), args.colmask]
    print(f"[client] starting: {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            env=env)  # stderr passes through to our stderr
    ok_overall = False
    try:
        # ---- LOAD_MATRIX ----
        t0 = time.time()
        payload = pack_bitplanes_matrix(W, args.qbits)
        body = struct.pack("<5I", MAGIC_LOAD, 1, args.qbits, args.K, args.M) + payload
        t_pack = time.time() - t0
        t0 = time.time()
        send_req(proc, body)
        resp = read_resp(proc)
        magic, handle, status = struct.unpack("<3I", resp[:12])
        t_load = time.time() - t0
        if magic != MAGIC_LOAD_ACK or status != 0:
            raise RuntimeError(f"LOAD failed: magic={magic:#x} status={status}")
        print(f"[client] LOAD ok: q={args.qbits} K={args.K} M={args.M} "
              f"({len(payload)} B payload; pack {t_pack:.2f}s, load {t_load:.2f}s)",
              flush=True)

        # ---- GEMV(s) ----
        all_exact = True
        for g in range(args.gemvs):
            if args.rbits == 1:
                x = rng.integers(0, 2, size=args.K, dtype=np.int64)
            else:
                rlo, rhi = -(1 << (args.rbits - 1)), (1 << (args.rbits - 1)) - 1
                x = rng.integers(rlo, rhi + 1, size=args.K, dtype=np.int64)
            y_ref = W @ x
            if args.partials:
                nblk = (args.K + 31) // 32
                body = struct.pack("<3I", MAGIC_PART, 1, args.rbits) + \
                    pack_bitplanes_vector(x, args.rbits)
                t0 = time.time()
                send_req(proc, body)
                resp = read_resp(proc)
                t_gemv = time.time() - t0
                magic, handle, status, M, NBLK = struct.unpack("<5I", resp[:20])
                if magic != MAGIC_PART_ACK or status != 0:
                    raise RuntimeError(f"PARTIALS failed: magic={magic:#x} status={status}")
                assert M == args.M and NBLK == nblk
                P = np.frombuffer(resp[20:], dtype="<i4").astype(np.int64)
                P = P.reshape(args.M, nblk)
                Kpad = nblk * 32
                Wp = np.zeros((args.M, Kpad), dtype=np.int64); Wp[:, :args.K] = W
                xp = np.zeros(Kpad, dtype=np.int64); xp[:args.K] = x
                P_ref = (Wp.reshape(args.M, nblk, 32) *
                         xp.reshape(nblk, 32)).sum(axis=2)
                n_bad = int((P != P_ref).sum())
                total = args.M * nblk
                sum_ok = np.array_equal(P.sum(axis=1), y_ref)
                print(f"[client] PARTIALS#{g}: {t_gemv:.1f}s wall  "
                      f"exact {total-n_bad}/{total} block-partials "
                      f"({100.0*(total-n_bad)/total:.4f}%)  "
                      f"sum-vs-GEMV-ref {'OK' if sum_ok else 'MISMATCH'}", flush=True)
                if n_bad:
                    all_exact = False
                    bm, bb = np.nonzero(P != P_ref)
                    show = list(zip(bm[:8], bb[:8]))
                    print("[client]   bad partials (first 8): "
                          + ", ".join(f"(m={m},b={b}) ref={int(P_ref[m,b])} got={int(P[m,b])}"
                                      for m, b in show), flush=True)
                continue
            body = struct.pack("<3I", MAGIC_GEMV, 1, args.rbits) + \
                pack_bitplanes_vector(x, args.rbits)
            t0 = time.time()
            send_req(proc, body)
            resp = read_resp(proc)
            t_gemv = time.time() - t0
            magic, handle, status, M = struct.unpack("<4I", resp[:16])
            if magic != MAGIC_GEMV_ACK or status != 0:
                raise RuntimeError(f"GEMV failed: magic={magic:#x} status={status}")
            y = np.frombuffer(resp[16:], dtype="<i4").astype(np.int64)
            assert M == args.M and y.shape[0] == args.M

            err = y - y_ref
            n_exact = int((err == 0).sum())
            aerr = np.abs(err)
            print(f"[client] GEMV#{g}: {t_gemv:.1f}s wall  "
                  f"exact {n_exact}/{args.M} ({100.0*n_exact/args.M:.4f}%)  "
                  f"|err|: max={int(aerr.max())} mean={aerr.mean():.4f}", flush=True)
            if n_exact != args.M:
                all_exact = False
                bad = np.nonzero(err)[0]
                show = bad[:12]
                print(f"[client]   mismatch outputs (first {len(show)} of {len(bad)}): "
                      + ", ".join(f"m={int(m)} ref={int(y_ref[m])} got={int(y[m])}"
                                  for m in show), flush=True)
                # err magnitude histogram (near-exact = cell-noise envelope doc)
                vals, cnts = np.unique(aerr[bad], return_counts=True)
                print("[client]   |err| histogram: "
                      + ", ".join(f"{int(v)}x{int(c)}" for v, c in zip(vals, cnts)),
                      flush=True)
        print(f"[client] RESULT: {'PASS — bit-exact' if all_exact else 'RESIDUAL — see histogram (cell-noise envelope?)'}",
              flush=True)
        ok_overall = all_exact
    finally:
        # graceful shutdown: quit sentinel, then wait. NEVER SIGKILL (XDMA).
        try:
            proc.stdin.write(struct.pack("<I", 0))
            proc.stdin.flush()
            proc.stdin.close()
        except Exception:
            pass
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            print("[client] server slow to exit; SIGTERM and wait (no SIGKILL)",
                  flush=True)
            proc.terminate()
            proc.wait()
    sys.exit(0 if ok_overall else 1)


if __name__ == "__main__":
    main()
