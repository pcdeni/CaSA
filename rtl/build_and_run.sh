#!/usr/bin/env bash
# build_and_run.sh — readback_engine build4 drain-capture-timing validation.
#
# Builds TWO Verilator models of the full engine in POPCOUNT_ACCUM_MODE:
#   obj_b4 : readback_engine.v         (build4 fix)        [-DTB_BUILD4]
#   obj_b3 : readback_engine_build3.v  (flashed build3, repro baseline)
# plus box-exact popcount_accum.v / pop_count4.v and the FWFT behavioral
# rdback_fifo (rdback_fifo_sim.v) matching the QUAD IP config.
#
# Scenario groups (tb_readback.cpp):
#   (a)-(e)  build3 suite, fixed expectations — BOTH variants must pass
#            (b4 must be behavior-identical when quiet at the flush edge)
#   (f1/f2/g) drain-capture-timing repro — b3 demonstrates the silicon
#            corruption (short/zero/lagged totals), b4 must be exact
#   (h)(i)   b4-only: SET-word idempotence + maintenance-in-deferral
# then diffs the READ_MODE data-beat dumps (b3 vs b4 bit-identical).
#
# History: the build3-era version of this script compared
# readback_engine_orig.v (build2) against build3; that pair's verdict is
# recorded in RESULT.md addendum 20b (31/31).
set -euo pipefail
cd "$(dirname "$0")"

# NOTE: box verilator is 4.028 — no -Wno-LATCH there (flag added later).
WARNS=(-Wno-fatal -Wno-WIDTH -Wno-CASEINCOMPLETE
       -Wno-UNOPTFLAT -Wno-UNUSED -Wno-PINMISSING -Wno-IMPLICIT)

build_one() {
    local name="$1" engine="$2"; shift 2
    rm -rf "obj_$name"
    verilator --cc --exe --Mdir "obj_$name" \
        -I. \
        "${WARNS[@]}" \
        +define+POPCOUNT_ACCUM_MODE=1 \
        --top-module readback_engine \
        "$@" \
        "$engine" popcount_accum.v pop_count4.v rdback_fifo_sim.v \
        tb_readback.cpp
    OPT_FAST="-O2" make -j"$(nproc)" -C "obj_$name" -f Vreadback_engine.mk -s
}

echo "== building BUILD4 =="
build_one b4 readback_engine.v -CFLAGS -DTB_BUILD4
echo "== building BUILD3 (repro baseline) =="
build_one b3 readback_engine_build3.v

echo
B4_RC=0; ./obj_b4/Vreadback_engine b4 || B4_RC=$?
echo
B3_RC=0; ./obj_b3/Vreadback_engine b3 || B3_RC=$?

echo
if cmp -s readmode_beats_b4.txt readmode_beats_b3.txt; then
    echo "READ_MODE data beats: BIT-IDENTICAL between b3 and b4  PASS"
    DIFF_RC=0
else
    echo "READ_MODE data beats: DIFFER  FAIL"
    diff readmode_beats_b4.txt readmode_beats_b3.txt || true
    DIFF_RC=1
fi

if [[ $B4_RC -eq 0 && $B3_RC -eq 0 && $DIFF_RC -eq 0 ]]; then
    echo "OVERALL: ALL SCENARIOS PASS"
    exit 0
else
    echo "OVERALL: FAIL (b4=$B4_RC b3=$B3_RC readmode_diff=$DIFF_RC)"
    exit 1
fi
