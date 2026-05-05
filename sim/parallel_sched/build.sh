#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

DRB=/home/deni/Claude/SiMRA-DRAM-main/DRAM-Bender/sources

verilator --cc -Wno-fatal -Wno-WIDTH -Wno-CASEINCOMPLETE -Wno-LATCH \
  -Wno-UNOPTFLAT -Wno-UNUSED -Wno-PINMISSING -Wno-IMPLICIT \
  -DXILINX_SIMULATOR \
  -I${DRB}/hdl/header_verilog \
  -I${DRB}/hdl/verilog \
  -I/home/deni/Claude/dram_bender_sim/phy_util \
  -I${DRB}/api \
  -I${DRB}/apps/DSN_AE_APPS \
  --top-module softmc_pipeline_top \
  --exe sim_compare.cpp parallel_emit.cpp \
        ${DRB}/api/instruction.cpp \
        ${DRB}/api/prog.cpp \
        ${DRB}/apps/DSN_AE_APPS/util.cpp \
  --build \
  --CFLAGS "-I${DRB}/api -I${DRB}/apps/DSN_AE_APPS -std=c++17" \
  ${DRB}/hdl/verilog/pre_decode.v \
  ${DRB}/hdl/verilog/fetch_stage.v \
  ${DRB}/hdl/verilog/decode_stage.v \
  ${DRB}/hdl/verilog/exe_pipeline.v \
  ${DRB}/hdl/verilog/ddr_pipeline.v \
  ${DRB}/hdl/verilog/execute_stage.v \
  ${DRB}/hdl/verilog/register_file.v \
  ${DRB}/hdl/verilog/reg_mem.v \
  ${DRB}/hdl/verilog/softmc_pipeline.v \
  /home/deni/Claude/dram_bender_sim/phy_util/softmc_pipeline_top.v

echo
echo "=== Running ==="
./obj_dir/Vsoftmc_pipeline_top
