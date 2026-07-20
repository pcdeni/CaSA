#!/usr/bin/env python3
"""Task O7(c): first EXACT fp32 from the MVDRAM reproduction.

Drives lane2-gemv-server's GEMV_PARTIALS (0x4D563003) on a REAL Llama-2-7B
q4_0 tensor (the bit-exact-validated de-repack dump of blk.0.attn_q.weight:
descaled ints q-8 in [-8,7] + raw fp16 per-32-block scales), with a
q8_0-quantized activation vector (ggml quantize_row_q8_0 convention:
per-32-block d = amax/127 stored as fp16, ints = roundf(x/d)).

The server returns exact integer per-block partial sums P[m,b] =
sum_{n in block b} W_int[m,n] * x_int[n], computed by the in-DRAM CSA tree
per block (the paper's per-subarray partial sums, SII-C2/SVII, retrieved and
aggregated by the processor). The host then reconstructs

    y_fp32[m] = sum_b fp32(d4[m,b]) * fp32(d8[b]) * P[m,b]

accumulated sequentially in float32, block-ascending — the SAME formula,
order and association as ggml's scalar ggml_vec_dot_q4_0_q8_0 (sumf +=
(GGML_FP16_TO_FP32(d4)*GGML_FP16_TO_FP32(d8))*sumi). The CPU reference
computes identical integer partials + identical fp32 reconstruction on the
host; the verdict is a BYTE-level comparison of the two fp32 vectors. A
float64 dequantized oracle is also reported as a sanity check (near, not
bit-exact — different association).

Run with PIM_VOTE3=1 (default here): the documented recovery of the unvoted
transient cell-noise envelope; the bit-exact claim needs every partial exact.

Usage:
  python3 lane2_partials_fp32.py [--M 4096] [--vote3 1] [--seed 11]
      [--dump /home/deni/mvdram_bench/smoke_2026_07_18/dump_first_tensor.bin]
"""
import argparse
import os
import struct
import subprocess
import sys
import time

import numpy as np

from lane2_client_smoke import (MAGIC_LOAD, MAGIC_LOAD_ACK, MAGIC_PART,
                                MAGIC_PART_ACK, pack_bitplanes_matrix,
                                pack_bitplanes_vector, read_resp, send_req)

HERE = os.path.dirname(os.path.abspath(__file__))


def load_dump(path):
    with open(path, "rb") as f:
        magic, K, M = struct.unpack("<3I", f.read(12))
        assert magic == 0x4450564D, hex(magic)
        name = f.read(128).split(b"\0")[0].decode()
        W = np.frombuffer(f.read(M * K), dtype=np.int8).reshape(M, K)
        nblk = K // 32
        d4 = np.frombuffer(f.read(M * nblk * 2), dtype="<u2").reshape(M, nblk)
    return name, W, d4.view(np.float16)


def quantize_q8_0(a):
    """ggml quantize_row_q8_0_ref, vectorized: per-32 block, d=amax/127 (fp16
    stored), q=roundf(x/d) (round half away from zero)."""
    K = a.shape[0]
    assert K % 32 == 0
    blocks = a.reshape(-1, 32).astype(np.float32)
    amax = np.abs(blocks).max(axis=1)
    d = (amax / np.float32(127.0)).astype(np.float32)
    idv = np.where(d > 0, np.float32(1.0) / d, np.float32(0.0)).astype(np.float32)
    v = blocks * idv[:, None]
    q = (np.sign(v) * np.floor(np.abs(v) + np.float32(0.5))).astype(np.int64)
    d16 = d.astype(np.float16)  # stored scale (ggml keeps fp16)
    return q.reshape(K), d16


def fp32_reconstruct(P, d4f16, d8f16):
    """Sequential per-block fp32 accumulation, ggml scalar vec_dot order."""
    M, nblk = P.shape
    d4 = d4f16.astype(np.float32)
    d8 = d8f16.astype(np.float32)
    acc = np.zeros(M, dtype=np.float32)
    for b in range(nblk):
        term = (d4[:, b] * d8[b]) * P[:, b].astype(np.float32)
        acc = acc + term.astype(np.float32)
    return acc.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="/home/deni/mvdram_bench/smoke_2026_07_18/dump_first_tensor.bin")
    ap.add_argument("--M", type=int, default=4096, help="row slice of the tensor")
    ap.add_argument("--bender", type=int, default=2)
    ap.add_argument("--bank", type=int, default=0)
    ap.add_argument("--sid", type=int, default=86)
    ap.add_argument("--calib", default=os.path.join(HERE, "calib_maj5_dimm2.txt"))
    ap.add_argument("--colmask", default=os.path.join(HERE, "colmask_dimm2_s86_robust.txt"))
    ap.add_argument("--server", default=os.path.join(HERE, "lane2-gemv-server"))
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--vote3", type=int, default=1)
    ap.add_argument("--backend", default="", help="'sim' for bring-up")
    args = ap.parse_args()

    name, Wfull, d4full = load_dump(args.dump)
    Mfull, K = Wfull.shape
    M = min(args.M, Mfull)
    W = Wfull[:M].astype(np.int64)
    d4 = d4full[:M]
    nblk = K // 32
    print(f"[fp32] tensor {name}: using M={M} of {Mfull}, K={K}, {nblk} blocks/row", flush=True)

    rng = np.random.default_rng(args.seed)
    a_f32 = rng.normal(0.0, 1.0, size=K).astype(np.float32)
    x, d8 = quantize_q8_0(a_f32)
    dens = [float(((x >> c) & 1).mean()) for c in range(8)]
    print(f"[fp32] activations: q8_0 ints, |x|max={int(np.abs(x).max())}, "
          f"plane densities {['%.3f' % v for v in dens]}", flush=True)

    env = dict(os.environ)
    env["BITSTREAM_IMEM"] = "8192"
    env["PIM_RECV_TIMEOUT_MS"] = "15000"
    env["PIM_VOTE3"] = str(args.vote3)
    if args.backend:
        env["LANE2_BACKEND"] = args.backend

    cmd = [args.server, str(args.bender), args.calib, str(args.bank),
           str(args.sid), args.colmask]
    print(f"[fp32] starting: {' '.join(cmd)} (vote3={args.vote3})", flush=True)
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env)
    ok = False
    try:
        t0 = time.time()
        payload = pack_bitplanes_matrix(W, 4)
        send_req(proc, struct.pack("<5I", MAGIC_LOAD, 1, 4, K, M) + payload)
        resp = read_resp(proc)
        magic, handle, status = struct.unpack("<3I", resp[:12])
        assert magic == MAGIC_LOAD_ACK and status == 0, (hex(magic), status)
        print(f"[fp32] LOAD ok ({time.time()-t0:.2f}s)", flush=True)

        t0 = time.time()
        send_req(proc, struct.pack("<3I", MAGIC_PART, 1, 8) + pack_bitplanes_vector(x, 8))
        resp = read_resp(proc)
        wall = time.time() - t0
        magic, handle, status, Mr, NBLK = struct.unpack("<5I", resp[:20])
        assert magic == MAGIC_PART_ACK and status == 0, (hex(magic), status)
        assert Mr == M and NBLK == nblk
        P = np.frombuffer(resp[20:], dtype="<i4").astype(np.int64).reshape(M, nblk)

        # CPU integer reference partials (exact)
        P_ref = (W.reshape(M, nblk, 32) * x.reshape(nblk, 32)).sum(axis=2)
        n_bad = int((P != P_ref).sum())
        total = M * nblk
        print(f"[fp32] PARTIALS: {wall:.1f}s wall  int-exact {total-n_bad}/{total} "
              f"({100.0*(total-n_bad)/total:.5f}%)", flush=True)
        if n_bad:
            bm, bb = np.nonzero(P != P_ref)
            print("[fp32]   bad partials (first 8): "
                  + ", ".join(f"(m={int(m)},b={int(b)}) ref={int(P_ref[m, b])} got={int(P[m, b])}"
                              for m, b in zip(bm[:8], bb[:8])), flush=True)

        # fp32 reconstruction, identical formula/order both sides
        y_pim = fp32_reconstruct(P, d4, d8)
        y_ref = fp32_reconstruct(P_ref, d4, d8)
        bitexact = y_pim.tobytes() == y_ref.tobytes()
        n_valexact = int((y_pim.view(np.uint32) == y_ref.view(np.uint32)).sum())
        print(f"[fp32] fp32 reconstruction: {'BIT-EXACT' if bitexact else 'NOT bit-exact'} "
              f"vs CPU fp32 reference ({n_valexact}/{M} outputs bit-identical)", flush=True)

        # independent float64 dequantized oracle (sanity; association differs)
        y64 = (W.astype(np.float64) * d4.astype(np.float64).repeat(32, axis=1)) @ \
              (x.astype(np.float64) * d8.astype(np.float64).repeat(32))
        denom = np.maximum(np.abs(y64), 1e-6)
        relmax = float(np.max(np.abs(y_ref.astype(np.float64) - y64) / denom))
        print(f"[fp32] fp64 dequant oracle: max rel diff vs reference {relmax:.3e} "
              f"(fp32-accumulation rounding; sanity only)", flush=True)
        ok = bitexact
        print(f"[fp32] RESULT: {'PASS — first exact fp32 from the reproduction' if ok else 'RESIDUAL'}",
              flush=True)
    finally:
        try:
            proc.stdin.write(struct.pack("<I", 0))
            proc.stdin.flush()
            proc.stdin.close()
        except Exception:
            pass
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            print("[fp32] server slow to exit; SIGTERM and wait (no SIGKILL)", flush=True)
            proc.terminate()
            proc.wait()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
