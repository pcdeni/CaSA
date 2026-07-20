#!/usr/bin/env python3
"""B2 — MVDRAM-paper GeMV table on the Lane-2 in-DRAM GeMV server.

Drives lane2-gemv-server DIRECTLY (no llama.cpp) over the length-prefixed
stdin/stdout protocol (LANE2_GEMV_SERVER.md; client pattern =
lane2_client_smoke.py) and produces the paper's SVIII-A GeMV benchmark
table at their dims and conventions, with documented deviations.

Paper conventions reproduced (arXiv:2503.23817 SVIII-A, PAPER_CONTRAST.md):
  - dims "used in modern LLMs": 4096x4096, 4096x11008, 11008x4096,
    32000x4096 (in our terms K = reduction dim, M = outputs; 32000x4096 =
    K 4096 -> M 32000, the Llama-2-7B head).
  - weight precisions: qb in {1,2,3,4} (their 2-8 bit sweep starts at 2;
    the server caps qb<=4; qb 1 and 3 included because the server supports
    them natively).
  - inputs at 50% bit sparsity ("typical LLM distribution"): activation
    bit-planes i.i.d. Bernoulli(0.5). For rb=1 that is a random binary
    vector; for rb>1 the planes decode to uniform signed two's-complement
    ints — exactly the 50% convention.
  - PLUS an honest second arm at the MEASURED real-activation plane
    densities from the B1 silicon/dry runs (Llama-2-7B, r=8):
      4096x4096   0.276  (attn q/v, silicon 2026-07-18)
      4096x11008  0.251  (ffn_gate/up blk.31, dry 2026-07-19)
      11008x4096  0.049  (ffn_down blk.31, silicon 2026-07-19)
      4096x32000  0.433  (q6_K head, silicon 2026-07-19)
    Both arms appear in the table — the 50% column is their convention,
    the measured column is what real Llama-2-7B activations do on this
    kernel (zero-skip makes wall proportional to set bits).

Documented deviations (the honesty convention, LANE2_GEMV_SERVER.md):
  - their 1000-iteration average is infeasible at our per-op walls
    (host-round-trip rig, no SV-E command streaming — the structural gap,
    PAPER_CONTRAST.md S4.2). Default --iters 5 with variance reporting
    (mean/std/min/max); --full sets 1000 and would run multi-day.
  - single module / single screened MAJ5 tuple, time-multiplexed
    (their N<=128-per-subarray partitioning across subarrays x 4 modules).
  - per-op verification against a host int64 reference over the same ints
    every iteration (int-exact %% reported; unvoted = the documented
    cell-noise envelope; --vote3 1 for bit-exact at 3x wall).
  - CPU baseline column = same-host numpy int64 GeMV over IDENTICAL ints
    (their Table II CPU baseline is ggml on an i7-9700K with the same DDR4
    modules; our same-host llama-bench analogs are recorded in
    REPRODUCTION.md: 7B Q4_0 pp8 55.7 t/s, tg4 14.2-14.9 t/s, 6 threads).

Estimated walls (calibrated model: GEMV wall ~= 4.4 ms x qb x total set
activation bits; validated across the 07-18/19 silicon table):
  - defaults (--iters 5, rb=1, both arms, qb 1..4, 4 shapes):
    ~59 min GEMV + ~2 min init/LOADs  ->  ~1-1.5 h total.
  - --full (1000 iters): ~8 days — schedule deliberately or trim
    shapes/qbits/arms with the filter flags.
  - rb=8 multiplies GEMV walls ~8x (plane-pairs scale with qb x rb).

Gating (same as run_b1_silicon_smoke.sh; the script is unattended-safe):
  (1) card enumerates: lspci -nn -d 10ee: non-empty (NEVER by BDF),
  (2) no process holds /dev/xdma0_* (FPGA-free check),
  (3) no live lane2/BitNet PIM server process,
  (4) B2_CONFIRM=1 in the environment (FPGA ownership is a human call).
  --backend sim (SimDramModel, in-process, no silicon) skips all gates —
  host validation of the driver/protocol/table plumbing only.

Usage:
  B2_CONFIRM=1 python3 b2_gemv_table.py                    # full default table
  python3 b2_gemv_table.py --backend sim --shapes 64x32 --iters 2   # host check
  B2_CONFIRM=1 nohup python3 b2_gemv_table.py > b2.log 2>&1 &       # unattended

Output: b2_results/<stamp>/b2_gemv_table.{md,csv} (+ partial rows on abort).
NEVER SIGKILL the server (XDMA); shutdown is quit-sentinel + wait, SIGTERM
only as a last resort. No arbitrary timeouts on FPGA calls (blocking reads;
the server itself carries PIM_RECV_TIMEOUT_MS=15000 for silicon hangs).
"""
import argparse
import csv
import datetime
import os
import struct
import subprocess
import sys
import time

import numpy as np

MAGIC_LOAD = 0x4D563001
MAGIC_GEMV = 0x4D563002
MAGIC_LOAD_ACK = 0x4D5630F1
MAGIC_GEMV_ACK = 0x4D5630F2

HERE = os.path.dirname(os.path.abspath(__file__))

# Paper dims: (label, K, M). K = reduction (ne00), M = outputs (ne01).
PAPER_SHAPES = [
    ("4096x4096",  4096,  4096),
    ("4096x11008", 4096,  11008),
    ("11008x4096", 11008, 4096),
    ("32000x4096", 4096,  32000),   # llama2-7b head: K=4096 -> M=32000
]

# Measured real-activation plane densities (Llama-2-7B, r=8; provenance in
# the module docstring). Keyed by shape label.
MEASURED_DENSITY = {
    "4096x4096":  0.276,
    "4096x11008": 0.251,
    "11008x4096": 0.049,
    "32000x4096": 0.433,
}

SEC_PER_FA = 4.4e-3   # calibrated 07-18/19: wall ~= 4.4 ms x qb x set bits


def log(msg):
    stamp = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[b2 {stamp}] {msg}", flush=True)


# ---------------------------------------------------------------- gating ----

def run_gates():
    """Silicon-run gates, mirroring run_b1_silicon_smoke.sh. Abort hard."""
    def fail(msg):
        print(f"[b2] ABORT: {msg}", file=sys.stderr)
        sys.exit(1)

    out = subprocess.run(["lspci", "-nn", "-d", "10ee:"],
                         capture_output=True, text=True).stdout.strip()
    if not out:
        fail("no Xilinx device in lspci -d 10ee: (card missing/wedged — "
             "see RUNBOOK_TOWER.md; a wedge can need a full cold power cycle)")

    import glob
    for dev in (glob.glob("/dev/xdma0_h2c_*") + glob.glob("/dev/xdma0_c2h_*")
                + glob.glob("/dev/xdma0_user")):
        r = subprocess.run(["fuser", "-s", dev], capture_output=True)
        if r.returncode == 0:
            who = subprocess.run(["fuser", "-v", dev], capture_output=True,
                                 text=True)
            fail(f"{dev} is in use (FPGA not free): "
                 f"{(who.stderr or who.stdout).strip().splitlines()[-1:]}")

    r = subprocess.run(["pgrep", "-af",
                        "lane2-gemv-server|pim_server|bitnet.*server"],
                       capture_output=True, text=True)
    if r.returncode == 0 and r.stdout.strip():
        fail("a PIM/bender server process is running: "
             + r.stdout.strip().splitlines()[0])

    if os.environ.get("B2_CONFIRM", "0") != "1":
        fail("set B2_CONFIRM=1 to confirm the FPGA is yours to use "
             "(coordination with other agents is a human decision)")


# ------------------------------------------------------------- bit packing --

def pack_bitplanes_matrix(W, qbits):
    """W int [M,K] -> qbits planes, each M x ceil(K/8) bytes, LSB-first
    (bit i of two's-complement W[m][n]) — the server LOAD wire format."""
    U = (W.astype(np.int64) & ((1 << qbits) - 1)).astype(np.uint8)
    out = bytearray()
    for i in range(qbits):
        bits = ((U >> i) & 1).astype(np.uint8)
        out += np.packbits(bits, axis=1, bitorder="little").tobytes()
    return bytes(out)


def pack_bitplanes_vector(x, rbits):
    U = (x.astype(np.int64) & ((1 << rbits) - 1)).astype(np.uint8)
    out = bytearray()
    for c in range(rbits):
        bits = ((U >> c) & 1).astype(np.uint8)
        out += np.packbits(bits[None, :], axis=1, bitorder="little").tobytes()
    return bytes(out)


def gen_weights(rng, qbits, M, K):
    """qb>1: uniform signed two's-complement; qb==1: binary {0,1} (the
    server's FAC(0,1)=+1 unsigned convention — same as rb==1)."""
    if qbits == 1:
        return rng.integers(0, 2, size=(M, K), dtype=np.int64)
    lo, hi = -(1 << (qbits - 1)), (1 << (qbits - 1)) - 1
    return rng.integers(lo, hi + 1, size=(M, K), dtype=np.int64)


def gen_activations(rng, rbits, K, density):
    """Activation bit-planes i.i.d. Bernoulli(density), decoded to the
    server's int convention (rb==1: binary {0,1}; rb>1: two's-complement
    signed). density=0.5 == the paper's 50%% bit-sparsity convention ==
    uniform ints for rb>1. Returns (x ints, actual plane density)."""
    bits = (rng.random((rbits, K)) < density).astype(np.int64)
    if rbits == 1:
        x = bits[0]
    else:
        x = np.zeros(K, dtype=np.int64)
        for c in range(rbits - 1):
            x += bits[c] << c
        x -= bits[rbits - 1] << (rbits - 1)   # top bit negative (FAC)
    return x, float(bits.mean())


# ------------------------------------------------------------ server I/O ----

class Lane2Server:
    def __init__(self, args):
        env = dict(os.environ)
        env["BITSTREAM_IMEM"] = "8192"          # mandatory silicon env
        env["PIM_RECV_TIMEOUT_MS"] = "15000"    # mandatory silicon env
        env["PIM_VOTE3"] = str(args.vote3)
        env["LANE2_PACK"] = str(args.pack)
        if args.backend:
            env["LANE2_BACKEND"] = args.backend
        cmd = [args.server, str(args.bender), args.calib, str(args.bank),
               str(args.sid), args.colmask]
        log("starting server: " + " ".join(cmd)
            + (f"  (LANE2_BACKEND={args.backend})" if args.backend else ""))
        # stderr passes through (screen/plane logs land in our log file)
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, env=env)

    def _read_exact(self, n):
        buf = b""
        while len(buf) < n:
            chunk = self.proc.stdout.read(n - len(buf))
            if not chunk:
                raise RuntimeError("server closed stdout — check stderr above")
            buf += chunk
        return buf

    def request(self, body):
        self.proc.stdin.write(struct.pack("<I", len(body)))
        self.proc.stdin.write(body)
        self.proc.stdin.flush()
        (ln,) = struct.unpack("<I", self._read_exact(4))
        return self._read_exact(ln)

    def load(self, handle, qbits, K, M, payload):
        body = struct.pack("<5I", MAGIC_LOAD, handle, qbits, K, M) + payload
        t0 = time.time()
        resp = self.request(body)
        wall = time.time() - t0
        magic, h, status = struct.unpack("<3I", resp[:12])
        if magic != MAGIC_LOAD_ACK or status != 0:
            raise RuntimeError(f"LOAD failed: magic={magic:#x} status={status}")
        return wall

    def gemv(self, handle, rbits, payload, M):
        body = struct.pack("<3I", MAGIC_GEMV, handle, rbits) + payload
        t0 = time.time()
        resp = self.request(body)
        wall = time.time() - t0
        magic, h, status, Mr = struct.unpack("<4I", resp[:16])
        if magic != MAGIC_GEMV_ACK or status != 0 or Mr != M:
            raise RuntimeError(f"GEMV failed: magic={magic:#x} status={status} M={Mr}")
        y = np.frombuffer(resp[16:], dtype="<i4").astype(np.int64)
        return y, wall

    def shutdown(self):
        """Graceful: len-0 quit sentinel + wait. NEVER SIGKILL (XDMA)."""
        if self.proc.poll() is not None:
            return
        try:
            self.proc.stdin.write(struct.pack("<I", 0))
            self.proc.stdin.flush()
            self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            log("server slow to exit; SIGTERM and wait (no SIGKILL)")
            self.proc.terminate()
            self.proc.wait()
        log(f"server exited (status {self.proc.returncode})")


# ------------------------------------------------------------------ main ----

def parse_shapes(spec):
    """--shapes '4096x4096,64x32' -> [(label,K,M)] (KxM order = paper label
    for the known dims; for ad-hoc dims the label is 'KxM' literally)."""
    known = {lb: (lb, k, m) for lb, k, m in PAPER_SHAPES}
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok in known:
            out.append(known[tok])
        else:
            a, b = tok.lower().split("x")
            out.append((tok, int(a), int(b)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--iters", type=int, default=5,
                    help="iterations per cell (paper: 1000 — see --full; "
                         "default 5, documented deviation)")
    ap.add_argument("--full", action="store_true",
                    help="paper-faithful 1000 iterations (multi-day!)")
    ap.add_argument("--rbits", type=int, default=1,
                    help="activation bits (default 1 = their headline "
                         "1-bit-vector convention; 8 = the B1 e2e setting)")
    ap.add_argument("--qbits", default="1,2,3,4",
                    help="comma list of weight precisions (server cap 4)")
    ap.add_argument("--shapes", default=",".join(lb for lb, _, _ in PAPER_SHAPES),
                    help="comma list: paper labels and/or ad-hoc KxM dims")
    ap.add_argument("--arms", default="paper50,measured",
                    help="input arms: paper50 (50%% bit sparsity), measured "
                         "(real-activation densities), or e.g. 'd0.3'")
    ap.add_argument("--seed", type=int, default=20260719)
    ap.add_argument("--vote3", type=int, default=0)
    ap.add_argument("--pack", type=int, default=1)
    ap.add_argument("--backend", default="",
                    help="'sim' = SimDramModel host validation (skips gates)")
    ap.add_argument("--bender", type=int, default=2)
    ap.add_argument("--bank", type=int, default=0)
    ap.add_argument("--sid", type=int, default=86)
    ap.add_argument("--calib", default=os.path.join(HERE, "calib_maj5_dimm2.txt"))
    ap.add_argument("--colmask", default=os.path.join(HERE, "colmask_dimm2_s86_robust.txt"))
    ap.add_argument("--server", default=os.path.join(HERE, "lane2-gemv-server"))
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--estimate-only", action="store_true",
                    help="print the projected wall table and exit (no server)")
    args = ap.parse_args()
    if args.full:
        args.iters = 1000

    shapes = parse_shapes(args.shapes)
    qbits_list = [int(q) for q in args.qbits.split(",") if q.strip()]
    arms = []
    for a in args.arms.split(","):
        a = a.strip()
        if a == "paper50":
            arms.append(("paper50", None))       # density 0.5
        elif a == "measured":
            arms.append(("measured", None))      # per-shape lookup
        elif a.startswith("d"):
            arms.append((a, float(a[1:])))
        elif a:
            raise SystemExit(f"unknown arm {a!r}")

    def arm_density(arm, label):
        name, fixed = arm
        if name == "paper50":
            return 0.5
        if name == "measured":
            d = MEASURED_DENSITY.get(label)
            if d is None:
                log(f"NOTE: no measured density for {label}; using 0.5")
                return 0.5
            return d
        return fixed

    # ---- projected walls (calibrated FA model) ----
    est_rows, total_est = [], 0.0
    for label, K, M in shapes:
        for qb in qbits_list:
            for arm in arms:
                d = arm_density(arm, label)
                per_iter = SEC_PER_FA * qb * d * args.rbits * K
                cell = per_iter * args.iters
                total_est += cell
                est_rows.append((label, qb, arm[0], d, per_iter, cell))
    log(f"projected GEMV wall: {total_est/60.0:.1f} min for "
        f"{len(est_rows)} cells x {args.iters} iters (rb={args.rbits}; "
        f"model {SEC_PER_FA*1e3:.1f} ms/FA x qb x set bits; + ~2 min "
        f"init/LOADs; vote3 triples GEMV walls)")
    if args.estimate_only:
        for label, qb, arm, d, per_iter, cell in est_rows:
            log(f"  {label:11s} qb={qb} {arm:9s} d={d:.3f}: "
                f"{per_iter:7.1f} s/GeMV, {cell/60.0:6.1f} min/cell")
        return

    if args.backend != "sim":
        run_gates()

    outdir = args.outdir or os.path.join(
        HERE, "b2_results", datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        + ("_sim" if args.backend == "sim" else ""))
    os.makedirs(outdir, exist_ok=True)
    csv_path = os.path.join(outdir, "b2_gemv_table.csv")
    md_path = os.path.join(outdir, "b2_gemv_table.md")
    log(f"output -> {outdir}")

    rng = np.random.default_rng(args.seed)
    srv = Lane2Server(args)
    rows = []
    t_run0 = time.time()
    try:
        for label, K, M in shapes:
            for qb in qbits_list:
                W = gen_weights(rng, qb, M, K)
                payload = pack_bitplanes_matrix(W, qb)
                load_s = srv.load(1, qb, K, M, payload)   # handle 1 reused:
                # the server replaces MATS[1] so its memory stays ~1 matrix
                log(f"LOAD {label} qb={qb}: {len(payload)/1e6:.1f} MB, "
                    f"{load_s:.2f} s")
                for arm in arms:
                    d = arm_density(arm, label)
                    walls, cpu_ms, exacts, dens = [], [], [], []
                    for it in range(args.iters):
                        x, d_act = gen_activations(rng, args.rbits, K, d)
                        t0 = time.time()
                        y_ref = W @ x
                        cpu_ms.append((time.time() - t0) * 1e3)
                        xp = pack_bitplanes_vector(x, args.rbits)
                        y, wall = srv.gemv(1, args.rbits, xp, M)
                        n_exact = int((y == y_ref).sum())
                        walls.append(wall)
                        exacts.append(100.0 * n_exact / M)
                        dens.append(d_act)
                        log(f"  {label} qb={qb} {arm[0]} it{it}: "
                            f"{wall:.1f} s, exact {n_exact}/{M} "
                            f"({exacts[-1]:.3f}%), d_act={d_act:.3f}")
                    w = np.array(walls)
                    c = np.array(cpu_ms)
                    e = np.array(exacts)
                    rows.append(dict(
                        shape=label, K=K, M=M, qb=qb, rb=args.rbits,
                        arm=arm[0], d_target=round(d, 3),
                        d_actual=round(float(np.mean(dens)), 4),
                        iters=args.iters, load_s=round(load_s, 2),
                        gemv_s_mean=round(float(w.mean()), 2),
                        gemv_s_std=round(float(w.std(ddof=1)) if len(w) > 1 else 0.0, 2),
                        gemv_s_min=round(float(w.min()), 2),
                        gemv_s_max=round(float(w.max()), 2),
                        cpu_ms_mean=round(float(c.mean()), 2),
                        cpu_ms_std=round(float(c.std(ddof=1)) if len(c) > 1 else 0.0, 2),
                        exact_pct_mean=round(float(e.mean()), 4),
                        exact_pct_min=round(float(e.min()), 4),
                        vote3=args.vote3,
                    ))
                    write_outputs(rows, csv_path, md_path, args, t_run0,
                                  partial=True)
    except (RuntimeError, OSError) as ex:
        log(f"ABORTING on error: {ex} (partial table kept: {csv_path})")
        raise
    finally:
        srv.shutdown()

    write_outputs(rows, csv_path, md_path, args, t_run0, partial=False)
    log(f"DONE: {len(rows)} cells in {(time.time()-t_run0)/60.0:.1f} min -> "
        f"{md_path}")


def write_outputs(rows, csv_path, md_path, args, t_run0, partial):
    if not rows:
        return
    cols = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=cols)
        wtr.writeheader()
        wtr.writerows(rows)

    lines = []
    lines.append("# B2 — MVDRAM-paper GeMV table (Lane-2 in-DRAM server)\n")
    lines.append(f"Generated {datetime.datetime.now():%Y-%m-%d %H:%M} on "
                 f"{'SimDramModel (HOST VALIDATION, not silicon)' if args.backend == 'sim' else f'bender {args.bender} bank {args.bank} s{args.sid}'}"
                 f"; rb={args.rbits}, iters={args.iters}"
                 f"{' (PARTIAL — run aborted/in progress)' if partial else ''}.\n")
    lines.append("Conventions: paper dims (SVIII-A); arm `paper50` = their "
                 "50% bit-sparsity inputs; arm `measured` = real Llama-2-7B "
                 "activation plane densities from the B1 runs. Deviations "
                 f"(documented): {args.iters} iters vs their 1000; single "
                 "module / single screened tuple vs their 4-module N<=128 "
                 "partitioning; host-round-trip execution vs their SV-E "
                 "streaming (PAPER_CONTRAST.md S4). CPU column = same-host "
                 "numpy int64 GeMV over identical ints (single-thread); "
                 "same-host llama.cpp CPU baselines: 7B Q4_0 pp8 55.7 t/s, "
                 "tg4 14.2-14.9 t/s (llama-bench, 6 threads, "
                 "REPRODUCTION.md 2026-07-18). Their headline for "
                 "32000x4096 @ 2-bit weights, 1-bit vector: 0.19 ms "
                 "(0.14 in-DRAM + 0.05 aggregation).\n")
    hdr = ("| shape (KxM) | qb | rb | arm | density | GeMV s (mean+/-std) "
           "[min,max] | CPU ms | int-exact % (mean/min) | iters |")
    sep = "|---|---|---|---|---|---|---|---|---|"
    lines.append(hdr)
    lines.append(sep)
    for r in rows:
        lines.append(
            f"| {r['shape']} | {r['qb']} | {r['rb']} | {r['arm']} "
            f"| {r['d_actual']:.3f} "
            f"| {r['gemv_s_mean']:.2f}+/-{r['gemv_s_std']:.2f} "
            f"[{r['gemv_s_min']:.2f},{r['gemv_s_max']:.2f}] "
            f"| {r['cpu_ms_mean']:.2f} "
            f"| {r['exact_pct_mean']:.3f}/{r['exact_pct_min']:.3f} "
            f"| {r['iters']} |")
    lines.append("")
    if args.vote3:
        lines.append("PIM_VOTE3=1 (3x vote, bit-exact recovery mode).\n")
    with open(md_path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
