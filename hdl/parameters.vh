// Fetch
`define INSTR_WIDTH 64
// Bumped 11 → 13 to fit a full BitNet-PIM outer loop in one execute.
// REQUIRES: Vivado-side regen of instr_blk_mem(.xci) IP with ADDR=13/depth=8192.
// Without that regen, the upper PC bits silently truncate.
`define IMEM_ADDR_WIDTH 13

// Decode - Execute
`define DDR_UOP_WIDTH 27
`define EXE_UOP_WIDTH 62
`define BG_WIDTH       2
`define BANK_WIDTH     2
`define COL_WIDTH     10
`define ROW_WIDTH     17

//Frontend
`define XDMA_AXI_DATA_WIDTH 256
`define IMEM_RD_LATENCY 1
//`define IMEM_SR // please only define when IMEM_RD_LATENCY > 1

//Common
`define HIGH 1'b1
`define LOW  1'b0

