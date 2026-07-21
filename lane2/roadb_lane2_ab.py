#!/usr/bin/env python3
"""Road-B lane2 A/B — accum-readout GeMV vs read-arm baseline (2026-07-21).

Task #10 deliverable: the LANE2_ACCUM product-row dataflow measured both
ways on the SAME program stream:
  arm R (LANE2_ACCUM=1): READ_MODE readout, host popcount per product row.
  arm A (LANE2_ACCUM=2): build-6 DIFF-accum readout, one 32-bit total per
                          product program (96 B vs 8 KB on the wire).
Identical program bytes in both arms; y agreement isolates the readout
mechanism (plus any run-to-run in-DRAM compute variance, quantified by the
startup diagnostics and reported per element). numpy int64 over the same
ints is the fidelity reference (junk columns show here, in BOTH arms).

Per ADR-005 this is the FPGA-accelerated NON-reproduction arm — the number
feeds the B2 table as a distinctly-labelled row, never blended with Road A.

Usage (silicon; gates + B2_CONFIRM=1 required, same as b2_gemv_table.py):
  B2_CONFIRM=1 python3 roadb_lane2_ab.py                 # bring-up + 4096x4096
  B2_CONFIRM=1 python3 roadb_lane2_ab.py --shapes 128x32 --qbits 2  # bring-up only
Arms run serially (one server owns the FPGA at a time): all shapes on the
read arm, restart, all shapes on the accum arm. NEVER SIGKILL the server.
"""
import argparse
import datetime
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from b2_gemv_table import (Lane2Server, gen_activations, gen_weights, log,
                           pack_bitplanes_matrix, pack_bitplanes_vector,
                           parse_shapes, run_gates)

HERE = os.path.dirname(os.path.abspath(__file__))


def run_arm(args, accum, jobs):
    """Spawn one server in the given LANE2_ACCUM mode and run every job.
    jobs: list of dicts {label,K,M,qb,rb,W,x} — results are written back in
    as y_<arm> / wall_<arm> / load_<arm>."""
    arm = "A" if accum == 2 else "R"
    os.environ["LANE2_ACCUM"] = str(accum)
    os.environ["LANE2_XREFRESH"] = str(args.xrefresh)
    if accum == 2:
        # batched-stream receiver knobs: empty windows cost ~one tick, not
        # 500 ms (platform floors the post-DIFF transition drain at 500 ms
        # regardless — see consumeDataAccum).
        os.environ["PIM_ACCUM_QUIET_MS"] = str(args.quiet_ms)
        os.environ["PIM_ACCUM_TICK_MS"] = str(args.tick_ms)
    else:
        os.environ.pop("PIM_ACCUM_QUIET_MS", None)
        os.environ.pop("PIM_ACCUM_TICK_MS", None)
    srv = Lane2Server(args)
    try:
        for jb in jobs:
            handle = jb["handle"]
            wl = srv.load(handle, jb["qb"], jb["K"], jb["M"],
                          pack_bitplanes_matrix(jb["W"], jb["qb"]))
            xpl = pack_bitplanes_vector(jb["x"], jb["rb"])
            walls = []
            y = None
            for it in range(args.iters):
                y, w = srv.gemv(handle, jb["rb"], xpl, jb["M"])
                walls.append(w)
            jb[f"y_{arm}"] = y
            jb[f"wall_{arm}"] = walls
            jb[f"load_{arm}"] = wl
            log(f"arm {arm} {jb['label']} qb{jb['qb']} rb{jb['rb']}: "
                f"load {wl:.2f} s, gemv "
                + "/".join(f"{w:.2f}" for w in walls) + " s")
    finally:
        srv.shutdown()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--shapes", default="128x32,4096x4096",
                    help="comma list of KxM dims; first is the bring-up shape")
    ap.add_argument("--qbits", default="2,4",
                    help="comma list of weight precisions per shape sweep")
    ap.add_argument("--rbits", type=int, default=1)
    ap.add_argument("--iters", type=int, default=1,
                    help="GEMV repeats per cell (variance)")
    ap.add_argument("--density", type=float, default=0.5,
                    help="activation bit density (paper convention 0.5)")
    ap.add_argument("--seed", type=int, default=20260721)
    ap.add_argument("--xrefresh", type=int, default=64,
                    help="LANE2_XREFRESH sentinel/refresh cadence")
    ap.add_argument("--quiet-ms", type=int, default=5,
                    help="accum arm PIM_ACCUM_QUIET_MS")
    ap.add_argument("--tick-ms", type=int, default=5,
                    help="accum arm PIM_ACCUM_TICK_MS")
    ap.add_argument("--arms", default="R,A",
                    help="which arms to run (R=read, A=accum), in order")
    ap.add_argument("--vote3", type=int, default=0)   # Lane2Server expects it
    ap.add_argument("--pack", type=int, default=1)
    ap.add_argument("--backend", default="")          # ACCUM is silicon-only
    ap.add_argument("--bender", type=int, default=2)
    ap.add_argument("--bank", type=int, default=0)
    ap.add_argument("--sid", type=int, default=86)
    ap.add_argument("--calib", default=os.path.join(HERE, "calib_maj5_dimm2.txt"))
    ap.add_argument("--colmask", default=os.path.join(HERE, "colmask_dimm2_s86_robust.txt"))
    ap.add_argument("--server", default=os.path.join(HERE, "lane2-gemv-server"))
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    if args.backend == "sim":
        sys.exit("LANE2_ACCUM is silicon-only; there is no sim arm")
    run_gates()

    shapes = parse_shapes(args.shapes)
    qbits_list = [int(q) for q in args.qbits.split(",") if q.strip()]
    rng = np.random.default_rng(args.seed)
    jobs = []
    handle = 700
    for label, K, M in shapes:
        for qb in qbits_list:
            W = gen_weights(rng, qb, M, K)
            x, dens = gen_activations(rng, args.rbits, K, args.density)
            jobs.append(dict(label=label, K=K, M=M, qb=qb, rb=args.rbits,
                             W=W, x=x, dens=dens, handle=handle))
            handle += 1

    outdir = args.outdir or os.path.join(
        HERE, "b2_results",
        datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + "_roadb_ab")
    os.makedirs(outdir, exist_ok=True)
    log(f"outdir {outdir}; {len(jobs)} cells, arms {args.arms}")

    for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
        run_arm(args, 1 if arm == "R" else 2, jobs)

    # ---- report ----
    lines = []
    lines.append("| shape | qb | rb | y_R==y_A | max|Δ| | vs numpy R (max/nz) | "
                 "vs numpy A | wall R (s) | wall A (s) | R/A |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    all_exact = True
    for jb in jobs:
        y_ref = (jb["W"].astype(np.int64) @ jb["x"].astype(np.int64))
        row = [jb["label"], str(jb["qb"]), str(jb["rb"])]
        yR = jb.get("y_R"); yA = jb.get("y_A")
        if yR is not None and yA is not None:
            d = np.abs(yR - yA)
            exact = int(d.max()) == 0
            all_exact &= exact
            row += ["EXACT" if exact else f"{(d > 0).sum()}/{len(d)} differ",
                    str(int(d.max()))]
        else:
            row += ["-", "-"]
        for y in (yR, yA):
            if y is None:
                row.append("-")
            else:
                dv = np.abs(y - y_ref)
                row.append(f"{int(dv.max())}/{int((dv > 0).sum())}nz")
        for arm in ("R", "A"):
            w = jb.get(f"wall_{arm}")
            row.append("-" if not w else f"{min(w):.2f}")
        wR, wA = jb.get("wall_R"), jb.get("wall_A")
        row.append(f"{min(wR)/min(wA):.2f}x" if wR and wA else "-")
        lines.append("| " + " | ".join(row) + " |")
    report = "\n".join(lines)
    print("\n" + report + "\n")
    if jobs and jobs[0].get("y_R") is not None and jobs[0].get("y_A") is not None:
        print(f"[ab] arm agreement over all cells: "
              f"{'EXACT' if all_exact else 'DIFFERS (see table)'}")
    with open(os.path.join(outdir, "roadb_ab.md"), "w") as f:
        f.write("# Road-B lane2 A/B — " + str(datetime.datetime.now()) + "\n\n"
                + "args: " + " ".join(sys.argv[1:]) + "\n\n" + report + "\n")
    for jb in jobs:
        for arm in ("R", "A"):
            y = jb.get(f"y_{arm}")
            if y is not None:
                np.save(os.path.join(
                    outdir, f"y_{arm}_{jb['label']}_qb{jb['qb']}.npy"), y)
    log(f"report -> {outdir}/roadb_ab.md")


if __name__ == "__main__":
    main()
