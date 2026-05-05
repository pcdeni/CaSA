#!/usr/bin/env bash
# Build the SoftMC pipeline + behavioral IMEM under Verilator and run the
# PHY-utilization sim. Re-running this script after editing any source under
# DRAM-Bender/sources/hdl/verilog/ picks up the changes (Verilator caches in
# obj_dir/).

set -euo pipefail
cd "$(dirname "$0")"

DRB=/home/deni/Claude/SiMRA-DRAM-main/DRAM-Bender/sources

# +XILINX_SIMULATOR makes execute_stage instantiate the behavioral reg_mem
# (1024×32) instead of the Vivado-IP scratchpad, which has no Verilator-
# visible body.
verilator --cc -Wno-fatal -Wno-WIDTH -Wno-CASEINCOMPLETE -Wno-LATCH \
  -Wno-UNOPTFLAT -Wno-UNUSED -Wno-PINMISSING -Wno-IMPLICIT \
  -DXILINX_SIMULATOR \
  -I${DRB}/hdl/header_verilog \
  -I${DRB}/hdl/verilog \
  --top-module softmc_pipeline_top \
  --exe sim_main.cpp \
  --build \
  ${DRB}/hdl/verilog/pre_decode.v \
  ${DRB}/hdl/verilog/fetch_stage.v \
  ${DRB}/hdl/verilog/decode_stage.v \
  ${DRB}/hdl/verilog/exe_pipeline.v \
  ${DRB}/hdl/verilog/ddr_pipeline.v \
  ${DRB}/hdl/verilog/execute_stage.v \
  ${DRB}/hdl/verilog/register_file.v \
  ${DRB}/hdl/verilog/reg_mem.v \
  ${DRB}/hdl/verilog/softmc_pipeline.v \
  softmc_pipeline_top.v

echo
echo "=== Running ==="
./obj_dir/Vsoftmc_pipeline_top
