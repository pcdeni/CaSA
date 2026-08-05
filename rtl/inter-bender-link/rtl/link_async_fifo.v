// -----------------------------------------------------------------------------
// link_async_fifo — thin wrapper making the xpm_fifo_async contract EXPLICIT in
// ONE place (L12b: vendor primitive, contract stated in source, never inferred).
// Presents a clean FWFT interface to the router.
//
//   Contract (asserted here, not left to the tool's mood):
//     * READ_MODE = "fwft"      -> `valid` shows the head; `rd_en` pops it.
//     * CDC_SYNC_STAGES = 4     -> 4-stage pointer sync (metastability margin).
//     * WIDTH carries {tlast, tkeep[31:0], tdata[255:0]} verbatim (289 bits).
//     * `full` is the WRITE-side backpressure the router uses to DECIDE A DROP;
//       the router never asserts wr_en when full (tap never stalls c2h).
//
// Synthesis binds the real Xilinx xpm_fifo_async; Verilator binds
// sim/xpm_fifo_async.v (same name). No `ifdef in the router — file selection
// alone swaps the model, exactly as rdback_fifo / rdback_fifo_sim.
// -----------------------------------------------------------------------------
module link_async_fifo #(
  parameter integer WIDTH = 289,
  parameter integer DEPTH = 512
)(
  input  wire              rst,
  // write side (source core ui_clk)
  input  wire              wr_clk,
  input  wire              wr_en,
  input  wire [WIDTH-1:0]  din,
  output wire              full,
  output wire [9:0]        wr_count,   // write-side occupancy (link status / L25 obs)
  // read side (dest core ui_clk) — FWFT
  input  wire              rd_clk,
  input  wire              rd_en,
  output wire [WIDTH-1:0]  dout,
  output wire              valid
);
  wire empty, dvalid;
  assign valid = dvalid;

  xpm_fifo_async #(
    .FIFO_MEMORY_TYPE   ("block"),
    .FIFO_WRITE_DEPTH   (DEPTH),
    .WRITE_DATA_WIDTH   (WIDTH),
    .READ_DATA_WIDTH    (WIDTH),
    .READ_MODE          ("fwft"),
    .FIFO_READ_LATENCY  (0),
    .CDC_SYNC_STAGES    (4),
    .WR_DATA_COUNT_WIDTH(10),
    .RD_DATA_COUNT_WIDTH(10),
    .PROG_FULL_THRESH   (0),
    .USE_ADV_FEATURES   ("0707")
  ) u_fifo (
    .rst          (rst),
    .wr_clk       (wr_clk),
    .wr_en        (wr_en),
    .din          (din),
    .full         (full),
    .wr_data_count(wr_count),
    .wr_rst_busy  (),
    .prog_full    (),
    .overflow     (),
    .rd_clk       (rd_clk),
    .rd_en        (rd_en),
    .dout         (dout),
    .empty        (empty),
    .data_valid   (dvalid),
    .rd_rst_busy  (),
    .underflow    (),
    .sleep        (1'b0),
    .injectsbiterr(1'b0),
    .injectdbiterr(1'b0),
    .sbiterr      (),
    .dbiterr      ()
  );
endmodule
