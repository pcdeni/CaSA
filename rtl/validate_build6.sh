#!/usr/bin/env bash
# validate_build6.sh — proves the build6 buffer_space-conservation fix on
# top of build5 (FWFT trailer) and documents the leak on <=build5.
# Expected matrix:
#   0-latency FIFO    : b4 PASS, b5 PASS, b6 PASS
#   fill-latency FIFO : b4 FAIL (the FWFT bug — proves the stub still
#                       discriminates), b5 PASS, b6 PASS
# plus: scenario (j) inside the TB asserts bs==8-after-8-programs on
# b4/b5 (the silicon starvation state) and bs==1024 conservation on b6,
# and READ_MODE data beats must be bit-identical between b5 and b6.
set -uo pipefail
cd "$(dirname "$0")"
W=(-Wno-fatal -Wno-WIDTH -Wno-CASEINCOMPLETE -Wno-UNOPTFLAT -Wno-UNUSED -Wno-PINMISSING -Wno-IMPLICIT)
run() { # $1=engine $2=cflags $3=fifo $4=label $5=variant-arg
  rm -rf obj_v
  verilator --cc --exe --Mdir obj_v -I. "${W[@]}" +define+POPCOUNT_ACCUM_MODE=1 \
    --top-module readback_engine -CFLAGS "$2" \
    "$1" popcount_accum.v pop_count4.v "$3" tb_readback.cpp >/dev/null 2>&1
  OPT_FAST=-O2 make -j"$(nproc)" -C obj_v -f Vreadback_engine.mk -s >/dev/null 2>&1
  printf '%-26s ' "$4"
  ./obj_v/Vreadback_engine "$5" 2>&1 | tail -1
  return "${PIPESTATUS[0]}"
}
declare -A RC
echo "== 0-latency FIFO =="
run readback_engine.v        "-DTB_BUILD4"             rdback_fifo_sim.v "build4 / 0-latency:" b4_0; RC[b4_0]=$?
run readback_engine_build5.v "-DTB_BUILD4 -DTB_BUILD5" rdback_fifo_sim.v "build5 / 0-latency:" b5_0; RC[b5_0]=$?
run readback_engine_build6.v "-DTB_BUILD4 -DTB_BUILD6" rdback_fifo_sim.v "build6 / 0-latency:" b6_0; RC[b6_0]=$?
cp -f readmode_beats_b5_0.txt beats_b5.txt 2>/dev/null || true
cp -f readmode_beats_b6_0.txt beats_b6.txt 2>/dev/null || true
echo "== FWFT fill-latency FIFO =="
run readback_engine.v        "-DTB_BUILD4"             rdback_fifo_sim_filllatency.v "build4 / fill-latency:" b4_f; RC[b4_f]=$?
run readback_engine_build5.v "-DTB_BUILD4 -DTB_BUILD5" rdback_fifo_sim_filllatency.v "build5 / fill-latency:" b5_f; RC[b5_f]=$?
run readback_engine_build6.v "-DTB_BUILD4 -DTB_BUILD6" rdback_fifo_sim_filllatency.v "build6 / fill-latency:" b6_f; RC[b6_f]=$?
echo
BEATS=1
if cmp -s beats_b5.txt beats_b6.txt; then
  echo "READ_MODE data beats b5 vs b6: BIT-IDENTICAL  PASS"; BEATS=0
else
  echo "READ_MODE data beats b5 vs b6: DIFFER  FAIL"
fi
OK=1
[[ ${RC[b4_0]} -eq 0 && ${RC[b5_0]} -eq 0 && ${RC[b6_0]} -eq 0 \
   && ${RC[b4_f]} -ne 0 && ${RC[b5_f]} -eq 0 && ${RC[b6_f]} -eq 0 \
   && $BEATS -eq 0 ]] && OK=0
if [[ $OK -eq 0 ]]; then echo "OVERALL: BUILD6 VALIDATED (matrix as expected)"; else
  echo "OVERALL: FAIL (b4_0=${RC[b4_0]} b5_0=${RC[b5_0]} b6_0=${RC[b6_0]} b4_f=${RC[b4_f]} b5_f=${RC[b5_f]} b6_f=${RC[b6_f]} beats=$BEATS)"; fi
exit $OK
