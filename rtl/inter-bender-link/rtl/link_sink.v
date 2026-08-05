// -----------------------------------------------------------------------------
// link_sink — the L17 consumer for a core's link_rx port until the fabric
// attention pipeline (SCOPE §3) exists. Always-ready drain (never backpressures,
// so a source stall converts to counted tap-drops, never a wedge) + a frame
// counter for host observability. On silicon this module is the capability port
// the attention block later replaces; no link_rx stream is ever left unconsumed.
// -----------------------------------------------------------------------------
module link_sink #(
  parameter integer DW = 256,
  parameter integer KW = 32,
  parameter integer CTRW = 32
)(
  input  wire            clk,
  input  wire            rst,
  input  wire [DW-1:0]   rx_tdata,
  input  wire [KW-1:0]   rx_tkeep,
  input  wire            rx_tvalid,
  output wire            rx_tready,
  input  wire            rx_tlast,
  input  wire [1:0]      rx_tsrc,
  output reg  [CTRW-1:0] sink_beats,
  output reg  [CTRW-1:0] sink_frames,
  output reg  [DW-1:0]   sink_last_word   // last received data (cheap scope tap)
);
  assign rx_tready = 1'b1;  // always drain
  always @(posedge clk) begin
    if (rst) begin
      sink_beats <= {CTRW{1'b0}}; sink_frames <= {CTRW{1'b0}}; sink_last_word <= {DW{1'b0}};
    end else if (rx_tvalid) begin
      sink_beats     <= sink_beats + 1'b1;
      sink_last_word <= rx_tdata;
      if (rx_tlast) sink_frames <= sink_frames + 1'b1;
    end
  end
endmodule
