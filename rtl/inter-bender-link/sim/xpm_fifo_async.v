// -----------------------------------------------------------------------------
// Behavioral stand-in for Xilinx `xpm_fifo_async` — GATE ONLY.
//
// Same-name substitution pattern as verilator_local/rdback_fifo_sim.v standing
// in for the `rdback_fifo` IP: synthesis resolves `xpm_fifo_async` to the real
// encrypted Xilinx XPM (via the xpm library); Verilator compiles THIS file.
//
// Models the subset of the XPM contract the link uses, in READ_MODE="fwft":
//   * write side:  wr_clk, wr_en, din, full, wr_rst_busy
//   * read  side:  rd_clk, rd_en, dout, empty, data_valid, rd_rst_busy
//   * common reset: rst (async assert, synchronous-ish release per domain)
// FWFT show-ahead: dout presents the head whenever data_valid=1; rd_en pops it.
//
// Faithfulness boundary (documented in CONTRACT.md §6): this is a 2-state
// FUNCTIONAL model. Cross-domain pointers are carried through 2-flop
// synchronizers to reproduce CDC LATENCY (so the FIFO genuinely fills under a
// stalled reader and the drop path is exercised), but METASTABILITY is not
// modelled — that is the real XPM's job, asserted binary by timing closure
// (L12). CDC_SYNC_STAGES / gray-coding are abstracted to a 2-flop binary sync;
// data integrity and full/empty/overflow behaviour match the real part.
// -----------------------------------------------------------------------------
`timescale 1ns/1ps
module xpm_fifo_async #(
  parameter integer WRITE_DATA_WIDTH = 289,
  parameter integer READ_DATA_WIDTH  = 289,
  parameter integer FIFO_WRITE_DEPTH = 512,
  parameter         READ_MODE        = "fwft",
  parameter integer CDC_SYNC_STAGES  = 4,
  parameter integer FIFO_READ_LATENCY= 0,
  parameter         FIFO_MEMORY_TYPE = "block",
  parameter integer WR_DATA_COUNT_WIDTH = 10,
  parameter integer RD_DATA_COUNT_WIDTH = 10,
  parameter integer PROG_FULL_THRESH = 0,
  parameter         USE_ADV_FEATURES = "0707"
)(
  input  wire                         rst,

  input  wire                         wr_clk,
  input  wire                         wr_en,
  input  wire [WRITE_DATA_WIDTH-1:0]  din,
  output wire                         full,
  output wire [WR_DATA_COUNT_WIDTH-1:0] wr_data_count,
  output wire                         wr_rst_busy,
  output wire                         prog_full,
  output wire                         overflow,

  input  wire                         rd_clk,
  input  wire                         rd_en,
  output wire [READ_DATA_WIDTH-1:0]   dout,
  output wire                         empty,
  output wire                         data_valid,
  output wire                         rd_rst_busy,
  output wire                         underflow,

  // tie-off ports present on the real primitive but unused here
  input  wire                         sleep,
  input  wire                         injectsbiterr,
  input  wire                         injectdbiterr,
  output wire                         sbiterr,
  output wire                         dbiterr
);
  // ceil(log2(DEPTH)) + 1 guard bit for wrap distinction
  function integer clog2; input integer v; integer i; begin
    clog2 = 0; for (i = v-1; i > 0; i = i >> 1) clog2 = clog2 + 1; end
  endfunction
  localparam integer AW = clog2(FIFO_WRITE_DEPTH);

  reg [WRITE_DATA_WIDTH-1:0] mem [0:FIFO_WRITE_DEPTH-1];
  reg [AW:0] wp = 0;   // write pointer (binary, +1 guard bit)
  reg [AW:0] rp = 0;   // read  pointer

  // 2-flop synchronizers carrying the opposite pointer across the domain
  reg [AW:0] rp_wsync0 = 0, rp_wsync1 = 0;  // rp seen in wr_clk
  reg [AW:0] wp_rsync0 = 0, wp_rsync1 = 0;  // wp seen in rd_clk

  // reset handshake (approximate: a couple of cycles of busy after rst release)
  reg [2:0] wr_rst_sr = 3'b111, rd_rst_sr = 3'b111;

  wire [AW:0] used_w = wp - rp_wsync1;          // occupancy seen by writer (conservative)
  assign full        = (used_w >= FIFO_WRITE_DEPTH[AW:0]);
  assign wr_data_count = used_w;   // LHS wider -> zero-extended; narrower -> truncated
  assign prog_full   = (PROG_FULL_THRESH != 0) && (used_w >= PROG_FULL_THRESH[AW:0]);
  assign wr_rst_busy = wr_rst_sr[0];

  wire [AW:0] used_r = wp_rsync1 - rp;          // occupancy seen by reader
  assign empty       = (used_r == 0);
  assign data_valid  = ~empty & ~rd_rst_busy;   // fwft show-ahead
  assign dout        = mem[rp[AW-1:0]];
  assign rd_rst_busy = rd_rst_sr[0];

  assign overflow    = wr_en & full & ~wr_rst_busy;
  assign underflow   = rd_en & empty & ~rd_rst_busy;
  assign sbiterr = 1'b0; assign dbiterr = 1'b0;

  // write domain
  always @(posedge wr_clk or posedge rst) begin
    if (rst) begin
      wp <= 0; rp_wsync0 <= 0; rp_wsync1 <= 0; wr_rst_sr <= 3'b111;
    end else begin
      wr_rst_sr <= {1'b0, wr_rst_sr[2:1]};
      rp_wsync0 <= rp;  rp_wsync1 <= rp_wsync0;   // sync rd-ptr in
      if (wr_en && !full && !wr_rst_busy) begin
        mem[wp[AW-1:0]] <= din;
        wp <= wp + 1'b1;
      end
    end
  end

  // read domain
  always @(posedge rd_clk or posedge rst) begin
    if (rst) begin
      rp <= 0; wp_rsync0 <= 0; wp_rsync1 <= 0; rd_rst_sr <= 3'b111;
    end else begin
      rd_rst_sr <= {1'b0, rd_rst_sr[2:1]};
      wp_rsync0 <= wp;  wp_rsync1 <= wp_rsync0;   // sync wr-ptr in
      if (rd_en && !empty && !rd_rst_busy)
        rp <= rp + 1'b1;
    end
  end
endmodule
