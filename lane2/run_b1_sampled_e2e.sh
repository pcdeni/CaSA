#!/usr/bin/env bash
# B1 sampled end-to-end — 256 generated tokens through llama.cpp with the
# MVDRAM shim sampling eligible GeMVs to the Lane-2 in-DRAM server
# (2026-07-21; the honest substitute for the paper's §VIII-A e2e, per the
# LANE2_GEMV_SERVER.md scope decision: every routed op is verified against
# the host int reference on the fly; wall-clock e2e is NOT claimed).
#
# Default: Llama-2-7B Q4_0, 256 tokens, MVDRAM_PIM_SAMPLE=2048 over ~224
# eligible ops/token => ~28 PIM-routed ops spread across the whole run
# (every projection class gets sampled repeatedly), MAX_OPS=32 cap.
# Expected wall: ~1-1.5 h (PIM ops ~80-450 s each by shape) + CPU
# generation. Model/tokens/sample overridable via env.
set -u
BENCH=/home/deni/mvdram_bench
CLI=$BENCH/llama.cpp/build/bin/llama-cli
OUT=$BENCH/e2e_2026_07_21
mkdir -p "$OUT"; cd "$OUT"
STAMP=$(date +%Y%m%d_%H%M%S)
MODEL=${E2E_MODEL:-$BENCH/models/llama-2-7b.Q4_0.gguf}
NTOK=${E2E_TOKENS:-256}

fail() { echo "[b1-e2e] ABORT: $*" >&2; exit 1; }
lspci -nn -d 10ee: | grep -q . || fail "no Xilinx device (never locate by BDF)"
for dev in /dev/xdma0_h2c_* /dev/xdma0_c2h_* /dev/xdma0_user; do
    fuser -s "$dev" 2>/dev/null && fail "$dev busy (FPGA not free)"
done
pgrep -f "lane2-gemv-server|bitnet-proj-server" | grep -v $$ >/dev/null && \
    fail "a PIM server is alive"
[ "${B1_SMOKE_CONFIRM:-0}" = "1" ] || fail "set B1_SMOKE_CONFIRM=1 (FPGA ownership is a human/operator call)"

export BITSTREAM_IMEM=8192
export PIM_RECV_TIMEOUT_MS=15000
export MVDRAM_PIM=1
export MVDRAM_PIM_RBITS=8
export MVDRAM_PIM_VOTE3=${MVDRAM_PIM_VOTE3:-0}
export MVDRAM_PIM_SAMPLE=${E2E_SAMPLE:-2048}
export MVDRAM_PIM_MAX_OPS=${E2E_MAX_OPS:-32}
export MVDRAM_PIM_OPLOG=e2e_${STAMP}_ops.log
export MVDRAM_PIM_LOG=e2e_${STAMP}_census.tsv

echo "[b1-e2e] model=$(basename "$MODEL") n=$NTOK sample=$MVDRAM_PIM_SAMPLE max_ops=$MVDRAM_PIM_MAX_OPS -> $OUT (stamp $STAMP)"
"$CLI" -m "$MODEL" \
    -p "Write a short story about a lighthouse keeper who discovers something unusual." \
    -n "$NTOK" -st -no-cnv --temp 0 -t 4 </dev/null \
    > e2e_${STAMP}.out 2> e2e_${STAMP}.err
rc=$?
echo "[b1-e2e] llama-cli exit=$rc; op verdicts:"
grep -E "OP#" e2e_${STAMP}_ops.log 2>/dev/null | tail -40 || echo "  (no ops logged!)"
lspci -nn -d 10ee: | grep -q . || fail "card disappeared (wedge — RUNBOOK recovery)"
n_ok=$(grep -cE "OP#.*OK" e2e_${STAMP}_ops.log 2>/dev/null || true)
n_all=$(grep -cE "OP#" e2e_${STAMP}_ops.log 2>/dev/null || true)
echo "[b1-e2e] DONE rc=$rc — $n_ok/$n_all routed ops verified; logs: $OUT/e2e_${STAMP}*"
exit $rc
