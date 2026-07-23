#!/bin/bash
set -uo pipefail
cd ~/Claude/bcu1525/readback_race_sim
WARNS=(-Wno-fatal -Wno-WIDTH -Wno-CASEINCOMPLETE -Wno-UNOPTFLAT -Wno-UNUSED -Wno-PINMISSING -Wno-IMPLICIT)
build_one() {
  local name="$1" engine="$2"; shift 2
  rm -rf "obj_$name"
  verilator --cc --exe --Mdir "obj_$name" -I. "${WARNS[@]}" \
    +define+POPCOUNT_ACCUM_MODE=1 --top-module readback_engine \
    "$@" "$engine" popcount_accum.v pop_count4.v rdback_fifo_sim.v tb_readback.cpp 2> "verr_$name.txt"
  OPT_FAST="-O2" make -j"$(nproc)" -C "obj_$name" -f Vreadback_engine.mk -s 2>> "verr_$name.txt"
}
build_one b9r readback_engine_build9.v -CFLAGS -DTB_BUILD9  || { echo "B9 COMPILE FAIL"; exit 3; }
build_one b10 readback_engine_build10.v -CFLAGS -DTB_BUILD10 || { echo "B10 COMPILE FAIL"; exit 3; }
./obj_b9r/Vreadback_engine b9r > run_b9r.txt 2>&1
./obj_b10/Vreadback_engine b10 > run_b10.txt 2>&1
echo "== stale-wdata verdicts =="
grep "stale-wdata segpop byte" run_b9r.txt
grep "stale-wdata segpop byte" run_b10.txt
echo "== failure-set diff (excluding the stale-wdata scenario) =="
grep -iE 'FAIL' run_b9r.txt | grep -v "stale-wdata" | sed 's/[0-9]\+/N/g' | sort > f9r.txt
grep -iE 'FAIL' run_b10.txt | grep -v "stale-wdata" | sed 's/[0-9]\+/N/g' | sort > f10.txt
if diff -q f9r.txt f10.txt >/dev/null; then
  echo "NON-REGRESSION OK: b10 shares b9's known failure set exactly ($(wc -l < f9r.txt) known-bug checks)"
else
  echo "REGRESSION on shared scenarios:"; diff f9r.txt f10.txt
fi
echo "== gate =="
REPRO=$(grep -c "stale-wdata segpop byte.*FAIL" run_b9r.txt || true)
FIXED=$(grep -c "stale-wdata segpop byte.*PASS" run_b10.txt || true)
if [ "$REPRO" -ge 1 ] && [ "$FIXED" -ge 1 ] && diff -q f9r.txt f10.txt >/dev/null; then
  echo "GATE OK: repro on b9 + fix on b10 + failure-set identical elsewhere"
else
  echo "GATE NOT MET"
fi
