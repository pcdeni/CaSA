// -----------------------------------------------------------------------------
// Behavioral stand-in for Xilinx `xpm_cdc_array_single` — GATE ONLY.
// Synthesis resolves the real XPM; Verilator compiles this.
// Multi-bit single-register-per-bit synchronizer. Contract (Xilinx): the SOURCE
// bus must be quasi-static / gray-coded when coherence across bits matters. The
// link uses it only on a FROZEN (stable) counter snapshot (CONTRACT.md §6), so a
// straight 2..N-flop sync per bit is coherent by construction.
// -----------------------------------------------------------------------------
`timescale 1ns/1ps
module xpm_cdc_array_single #(
  parameter integer DEST_SYNC_FF   = 4,
  parameter integer WIDTH          = 32,
  parameter integer SIM_ASSERT_CHK = 0,
  parameter integer SRC_INPUT_REG  = 1
)(
  input  wire              src_clk,
  input  wire [WIDTH-1:0]  src_in,
  input  wire              dest_clk,
  output wire [WIDTH-1:0]  dest_out
);
  reg [WIDTH-1:0] src_ff = 0;
  reg [WIDTH-1:0] sync [0:DEST_SYNC_FF-1];
  integer i;

  always @(posedge src_clk) src_ff <= (SRC_INPUT_REG != 0) ? src_in : src_ff;
  wire [WIDTH-1:0] src_sel = (SRC_INPUT_REG != 0) ? src_ff : src_in;

  always @(posedge dest_clk) begin
    sync[0] <= src_sel;
    for (i = 1; i < DEST_SYNC_FF; i = i + 1) sync[i] <= sync[i-1];
  end
  assign dest_out = sync[DEST_SYNC_FF-1];

  initial begin
    src_ff = 0;
    for (i = 0; i < DEST_SYNC_FF; i = i + 1) sync[i] = 0;
  end
endmodule
