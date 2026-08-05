#!/usr/bin/env bash
# Verilator gate for inter_bender_link (#76). Local, no card, no box.
set -e
cd "$(dirname "$0")"
rm -rf obj_dir
verilator --cc --exe --build -j 0 \
  -Wall -Wno-fatal -Wno-DECLFILENAME -Wno-UNUSEDSIGNAL -Wno-UNUSEDPARAM \
  -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC -Wno-PINCONNECTEMPTY -Wno-TIMESCALEMOD -Wno-GENUNNAMED \
  -GDW=32 -GKW=4 -GFIFO_DEPTH=8 \
  --top-module inter_bender_link \
  -Irtl -Isim \
  rtl/inter_bender_link.v rtl/link_async_fifo.v sim/xpm_fifo_async.v \
  tb/tb_link.cpp \
  -o Vlink
echo "---- RUN ----"
./obj_dir/Vlink
