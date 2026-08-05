// =============================================================================
// popcount_bist — one-register self-test for the pop_count4 tree (L25, #77).
// A dropin for the readback-engine author. Turns the recurring "is the 0xe
// undercount fixed in THIS image?" question into: host writes a test pattern,
// reads back the popcount the HARDWARE computed. If the flashed image carries
// the buggy pop_count4 (6490c9b7, missing 4'b1110), a 0xe-dense pattern returns
// a value LOWER than the true popcount -> the BIST fails on the exact defect.
//
// It runs the pattern through the SAME pop_count4 module the production datapath
// instantiates (readback_engine.v:81), so it tests the actually-synthesized tree.
// Return path = reuse the temp-read control-word -> c2h mux (feed bist_word into
// rdback_din when bist_rd is high), NOT new c2h logic.
//
// STRONGER FORM (specified in MANIFEST.md §4, preferred if cheap on the box):
//   mux the host pattern into the EXISTING production popcount input instead of
//   this parallel copy -> tests the exact production adder wiring, not a twin.
//   This module is the fallback when that mux is not readily reachable.
//
// WIDTH default 512 = the readback datapath beat. The sum of a random 512-bit
// vector's popcount fits in 10 bits.
// =============================================================================
module popcount_bist #(
  parameter integer WIDTH = 512
)(
  input  wire              clk,
  input  wire              rst,
  input  wire              patt_wr,        // host writes the test pattern (control word / DLOAD)
  input  wire [WIDTH-1:0]  patt_wdata,
  output reg  [15:0]       bist_popcount   // pack into rdback_din on the bist read control word
);
  // tree of pop_count4 over nibbles — identical primitive to production
  localparam integer NN = WIDTH/4;
  reg  [WIDTH-1:0] patt_r;
  wire [2:0] nib_cnt [0:NN-1];
  genvar i;
  generate for (i = 0; i < NN; i = i + 1) begin : PC
    pop_count4 u_pc (.in(patt_r[i*4 +: 4]), .out(nib_cnt[i]));
  end endgenerate

  integer k;
  reg [15:0] sum;
  always @* begin
    sum = 16'd0;
    for (k = 0; k < NN; k = k + 1) sum = sum + {13'd0, nib_cnt[k]};
  end

  always @(posedge clk) begin
    if (rst) begin patt_r <= {WIDTH{1'b0}}; bist_popcount <= 16'd0; end
    else begin
      if (patt_wr) patt_r <= patt_wdata;
      bist_popcount <= sum;   // registered result, ready to return next control word
    end
  end
endmodule
