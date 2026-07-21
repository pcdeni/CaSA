#!/usr/bin/env bash
# validate_build5.sh — proves the build5 FWFT-trailer fix (RESULT.md 20e).
# Expected: 0-latency -> b4 PASS, b5 PASS ; fill-latency -> b4 FAIL, b5 PASS.
set -uo pipefail
cd "$(dirname "$0")"
W=(-Wno-fatal -Wno-WIDTH -Wno-CASEINCOMPLETE -Wno-UNOPTFLAT -Wno-UNUSED -Wno-PINMISSING -Wno-IMPLICIT)
run() { # $1=engine $2=cflags $3=fifo $4=label
  rm -rf obj_v
  verilator --cc --exe --Mdir obj_v -I. "${W[@]}" +define+POPCOUNT_ACCUM_MODE=1 \
    --top-module readback_engine -CFLAGS "$2" \
    "$1" popcount_accum.v pop_count4.v "$3" tb_readback.cpp >/dev/null 2>&1
  OPT_FAST=-O2 make -j"$(nproc)" -C obj_v -f Vreadback_engine.mk -s >/dev/null 2>&1
  printf '%-26s ' "$4"; ./obj_v/Vreadback_engine x 2>&1 | tail -1
}
echo "== 0-latency FIFO =="
run readback_engine.v        "-DTB_BUILD4"             rdback_fifo_sim.v             "build4 / 0-latency:"
run readback_engine_build5.v "-DTB_BUILD4 -DTB_BUILD5" rdback_fifo_sim.v             "build5 / 0-latency:"
echo "== FWFT fill-latency FIFO =="
run readback_engine.v        "-DTB_BUILD4"             rdback_fifo_sim_filllatency.v "build4 / fill-latency:"
run readback_engine_build5.v "-DTB_BUILD4 -DTB_BUILD5" rdback_fifo_sim_filllatency.v "build5 / fill-latency:"
