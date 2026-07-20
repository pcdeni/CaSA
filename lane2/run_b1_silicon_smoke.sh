#!/usr/bin/env bash
# B1 silicon smoke — Q2_K + 11008-dim + q3_K ops through the Lane-2 GeMV
# server. Written 2026-07-19 (host-side B1 extension session); Phase D
# (q3_K) added later the same day. PHASES A/B/C EXECUTED on silicon
# 2026-07-19 (shape-coverage table COMPLETE — REPRODUCTION.md); Phase D is
# the remaining pending item, staged NOT RUN (FPGA owned by another agent).
# Default is therefore B1_SMOKE_PHASES=D; set B1_SMOKE_PHASES=ABCD to
# re-run everything. All phases dry-run-validated (extraction bit-exact vs
# gguf; marshaling + host int reference exercised; see smoke_2026_07_19/
# and REPRODUCTION.md "2026-07-19" sections).
#
# What it runs (route c2: PIM = verified side-channel, dst stays CPU):
#   Phase A: Llama-2-7B Q2_K — 2 sampled q2_K ops (blk.0 + blk.1 attn_q,
#            K=M=4096, qb=2). Expected wall ~80 s/op unvoted at r=8
#            (16 plane-pairs vs 32 for q4) + ~1.5 min server init.
#            [RAN 2026-07-19: 83.2 s / 43.5 s, 99.90/99.95% int-exact]
#   Phase B: Llama-2-7B Q4_0 — ONE 11008-dim op: ffn_down (K=11008,
#            M=4096, qb=4). K=11008 <= server cap 16384, single LOAD/GEMV
#            (no host K-split). Expected wall ~2.7x the 160 s 4096-op
#            (~7 min) at r=8 — taps scale with K.
#            [RAN 2026-07-19: 75.5 s (zero-skip, density 0.049), 99.88%]
#   Phase C (optional, B1_SMOKE_EXTRA=1 — kept for reruns):
#            - ffn_gate (K=4096, M=11008): M>4096 single-pass (344
#              screened segments; server had 1631 on 07-18), ~160 s.
#            - q6_K head (K=4096, M=32000): 3-handle plane split
#              (qb 4+1+2), ~4.7 min GEMV + 3 LOADs, 1000 segments.
#            [head RAN 2026-07-19: 449.1 s, 99.63%; gate not run]
#   Phase D: Llama-2-7B Q2_K — ONE q3_K op: blk.0.attn_v (K=M=4096,
#            qb=3, native single handle, w = q-4 in {-4..3}). Expected
#            wall ~= 3/2 x the Phase-A q2 op at the same density: measured
#            dry density 0.290 -> ~125 s unvoted at r=8 (N_FA ~= qb x
#            set-activation-bits x 4.4 ms/FA). The other q3_K census
#            shapes (ffn_gate 4096->11008 d=0.291, ffn_down 11008->4096
#            d=0.048) are dry-validated and reachable via
#            MVDRAM_PIM_ONLY=ffn_gate / ffn_down on the Q2_K model.
#
# Verification: every op is compared on the fly against the host int64
# reference over the same ints (shim mv_verify_op); expected unvoted
# int-exact is the documented cell-noise envelope (07-18: 99.88-99.90%
# of outputs at r=8); MVDRAM_PIM_VOTE3=1 recovers bit-exact at 3x wall.
#
# Gating: refuses to run unless
#   (1) the card enumerates (lspci -d 10ee: non-empty — NEVER locate by BDF),
#   (2) no process holds /dev/xdma0_* (the FPGA-free check),
#   (3) no lane2-gemv-server / BitNet PIM server process is alive,
#   (4) B1_SMOKE_CONFIRM=1 is set (coordination with the other agent is a
#       human decision, not something fuser can prove).
set -u

BENCH=/home/deni/mvdram_bench
REPRO=/home/deni/Claude/mvdram-repro
CLI=$BENCH/llama.cpp/build/bin/llama-cli
OUT=$BENCH/smoke_2026_07_19/silicon
STAMP=$(date +%Y%m%d_%H%M%S)

fail() { echo "[b1-smoke] ABORT: $*" >&2; exit 1; }

# ---- gate 1: card enumerated (vendor probe, never BDF) ----
lspci -nn -d 10ee: | grep -q . || \
    fail "no Xilinx device in lspci -d 10ee: (card missing/wedged — see RUNBOOK_TOWER.md; a wedge can need a full cold power cycle)"

# ---- gate 2: XDMA nodes free ----
for dev in /dev/xdma0_h2c_* /dev/xdma0_c2h_* /dev/xdma0_user; do
    [ -e "$dev" ] || continue
    if fuser -s "$dev" 2>/dev/null; then
        fail "$dev is in use (FPGA not free): $(fuser -v "$dev" 2>&1 | tail -1)"
    fi
done

# ---- gate 3: no PIM server processes ----
if pgrep -af 'lane2-gemv-server|pim_server|bitnet.*server' >/dev/null 2>&1; then
    fail "a PIM/bender server process is running: $(pgrep -af 'lane2-gemv-server|pim_server|bitnet.*server' | head -2)"
fi

# ---- gate 4: explicit operator confirmation ----
[ "${B1_SMOKE_CONFIRM:-0}" = "1" ] || \
    fail "set B1_SMOKE_CONFIRM=1 to confirm the FPGA is yours to use (another agent owned it when this script was staged)"

[ -x "$CLI" ] || fail "$CLI missing"
[ -x "$REPRO/lane2-gemv-server" ] || fail "lane2-gemv-server binary missing"
mkdir -p "$OUT"
cd "$OUT" || fail "cannot cd $OUT"

# ---- standing env (silicon-mandatory trio is ALSO set by the shim for the
#      server child; exported here for belt and braces) ----
export BITSTREAM_IMEM=8192
export PIM_RECV_TIMEOUT_MS=15000
export MVDRAM_PIM=1
export MVDRAM_PIM_SAMPLE=1
export MVDRAM_PIM_RBITS=8
export MVDRAM_PIM_VOTE3=${MVDRAM_PIM_VOTE3:-0}   # 1 = bit-exact at 3x wall
# server/calib/colmask/bender/bank/sid: shim defaults = verified
# bender2/bank0/s86 + calib_maj5_dimm2.txt + colmask_dimm2_s86_robust.txt.
# NO timeout(1) wrappers anywhere — FPGA runs get no arbitrary timeouts.

run_cli() { # run_cli <tag> <model> <max_ops> <only> <n_tokens>
    local tag=$1 model=$2 max_ops=$3 only=$4 ntok=$5
    echo "[b1-smoke] phase $tag: model=$(basename "$model") only=$only max_ops=$max_ops"
    MVDRAM_PIM_MAX_OPS=$max_ops MVDRAM_PIM_ONLY=$only \
    MVDRAM_PIM_OPLOG=${tag}_${STAMP}_ops.log MVDRAM_PIM_LOG=${tag}_${STAMP}_census.tsv \
    "$CLI" -m "$model" -p "The capital of France is" -n "$ntok" \
        -st -no-cnv --temp 0 -t 4 </dev/null \
        >"${tag}_${STAMP}.out" 2>"${tag}_${STAMP}.err"
    local rc=$?
    echo "[b1-smoke] phase $tag exit=$rc; verdicts:"
    grep -E "OP#" "${tag}_${STAMP}_ops.log" 2>/dev/null || echo "  (no ops logged!)"
    [ $rc -eq 0 ] || fail "phase $tag llama-cli exited $rc (do NOT SIGKILL anything; check ${tag}_${STAMP}.err and card state)"
    # card still there?
    lspci -nn -d 10ee: | grep -q . || fail "card disappeared after phase $tag (wedge — RUNBOOK recovery)"
}

# Phase selection: A/B/C completed on silicon 2026-07-19; D (q3_K) pending.
PHASES=${B1_SMOKE_PHASES:-D}

case "$PHASES" in *A*)
    # Phase A: 2 q2_K ops (dry-verified trigger: blk.0.attn_q, blk.1.attn_q)
    run_cli q2k "$BENCH/models/llama-2-7b.Q2_K.gguf" 2 attn_q 2
;; esac

case "$PHASES" in *B*)
    # Phase B: one 11008-contraction op (dry-verified trigger: blk.31.ffn_down)
    run_cli ffndown "$BENCH/models/llama-2-7b.Q4_0.gguf" 1 ffn_down 1
;; esac

case "$PHASES" in *C*)
    # Phase C (optional): M=11008 single-pass + the q6_K 3-handle head
    if [ "${B1_SMOKE_EXTRA:-0}" = "1" ]; then
        run_cli ffngate "$BENCH/models/llama-2-7b.Q4_0.gguf" 1 ffn_gate 1
        run_cli q6khead "$BENCH/models/llama-2-7b.Q2_K.gguf" 1 output.weight 1
    fi
;; esac

case "$PHASES" in *D*)
    # Phase D: one q3_K op (dry-verified trigger: blk.0.attn_v on the Q2_K
    # model — attn_v/attn_output + all ffn tensors are q3_K there). Native
    # qb=3 handle; extraction bit-exact vs gguf (check_q3k.py, 2026-07-19).
    # NB: 2 tokens, not 1 — attn tensors only present a batch=1 GeMV from
    # the SECOND generated token (token 1's logits come out of the prefill
    # batch; verified in the 07-19 censuses — ffn/head fire with -n 1,
    # attn_* do not).
    run_cli q3k "$BENCH/models/llama-2-7b.Q2_K.gguf" 1 attn_v 2
;; esac

echo "[b1-smoke] DONE (phases: $PHASES). Logs in $OUT (stamp $STAMP)."
echo "[b1-smoke] Append the OP# verdict lines to REPRODUCTION.md '2026-07-19' section (shape-coverage table)."
