`timescale 1ns/1ps
`include "parameters.vh"

// e2e_sim phase-2: behavioral DRAM behind the ddr_* command interface.
// The entire model (open-row tracking per {bg,bank}, sparse content
// store with deterministic seeding, fixed-latency in-order read-return
// queue) lives in ONE DPI-C call per cycle (dram_dpi.cpp); this shim
// just passes the packed slot buses and registers the return beat.
// Widths (parameters.vh): BG 2, BANK 2, COL 10, ROW 17; 4 cmd slots.
import "DPI-C" function void dram_tick(
  input  int              act_m,
  input  int              rd_m,
  input  int              wr_m,
  input  int              pre_m,
  input  bit [4*`BG_WIDTH-1:0]   bgs,
  input  bit [4*`BANK_WIDTH-1:0] banks,
  input  bit [4*`COL_WIDTH-1:0]  cols,
  input  bit [4*`ROW_WIDTH-1:0]  rows,
  input  bit [511:0]             wdata,
  output bit [511:0]             rdata,
  output int                     rvalid
);

module dram_model(
  input clk,
  input rst,
  input [3:0]                  ddr_act,
  input [3:0]                  ddr_read,
  input [3:0]                  ddr_write,
  input [3:0]                  ddr_pre,
  input [4*`BG_WIDTH-1:0]      ddr_bg,
  input [4*`BANK_WIDTH-1:0]    ddr_bank,
  input [4*`COL_WIDTH-1:0]     ddr_col,
  input [4*`ROW_WIDTH-1:0]     ddr_row,
  input [511:0]                ddr_wdata,
  output reg [511:0]           rd_data,
  output reg                   rd_valid
);

  bit [511:0] rdata_c;
  int         rvalid_c;

  always @(posedge clk) begin
    if (rst) begin
      rd_valid <= 1'b0;
      rd_data  <= 512'd0;
    end
    else begin
      dram_tick({28'd0, ddr_act}, {28'd0, ddr_read}, {28'd0, ddr_write},
                {28'd0, ddr_pre}, ddr_bg, ddr_bank, ddr_col, ddr_row,
                ddr_wdata, rdata_c, rvalid_c);
      rd_data  <= rdata_c;
      rd_valid <= rvalid_c[0];
    end
  end

endmodule
