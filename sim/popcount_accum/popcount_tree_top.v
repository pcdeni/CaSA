// End-to-end popcount path used by readback_engine with POPCOUNT_ACCUM_MODE
// enabled: 128 instances of pop_count4 + 6-level tree adder + accumulator.
// The tree topology is copied verbatim from readback_engine.v (lines around
// 60-105) so any HDL bug in the tree is caught here too.
//
// Simulated under Verilator without any Xilinx IP dependency.

module popcount_tree_top(
  input         clk,
  input         rst,
  input [511:0] in_data,
  input         in_valid,
  input         flush,           // pulse to drain accumulator
  output [31:0] accum_out,
  output        accum_valid,
  // Also expose the per-cycle 16-b popcount + valid for visibility.
  output reg [15:0] popcount_value,
  output reg        popcount_valid
);

  wire [2:0] pc_out [127:0];
  reg  [3:0] pc_l2 [63:0];
  reg  [4:0] pc_l3 [31:0];
  reg  [5:0] pc_l4 [15:0];
  reg  [6:0] pc_l5 [7:0];
  reg  [7:0] pc_l6 [3:0];
  reg  [8:0] pc_l7 [1:0];

  genvar pcs;
  generate
    for (pcs = 0; pcs < 128; pcs = pcs + 1) begin: gen_pcs
      pop_count4 pci(.in(in_data[pcs*4 +: 4]), .out(pc_out[pcs]));
    end
  endgenerate

  integer l1, l2, l3, l4, l5, l6;
  always @* begin
    for (l1 = 0; l1 < 64; l1 = l1 + 1)
      pc_l2[l1] = pc_out[2*l1] + pc_out[2*l1+1];
    for (l2 = 0; l2 < 32; l2 = l2 + 1)
      pc_l3[l2] = pc_l2[2*l2] + pc_l2[2*l2+1];
    for (l3 = 0; l3 < 16; l3 = l3 + 1)
      pc_l4[l3] = pc_l3[2*l3] + pc_l3[2*l3+1];
    for (l4 = 0; l4 < 8;  l4 = l4 + 1)
      pc_l5[l4] = pc_l4[2*l4] + pc_l4[2*l4+1];
    for (l5 = 0; l5 < 4;  l5 = l5 + 1)
      pc_l6[l5] = pc_l5[2*l5] + pc_l5[2*l5+1];
    for (l6 = 0; l6 < 2;  l6 = l6 + 1)
      pc_l7[l6] = pc_l6[2*l6] + pc_l6[2*l6+1];
  end

  always @(posedge clk) begin
    if (rst) begin
      popcount_value <= 16'b0;
      popcount_valid <= 1'b0;
    end else begin
      popcount_value <= pc_l7[0] + pc_l7[1];
      popcount_valid <= in_valid;     // 1-cycle latency (tree is combinational, valid is synced)
    end
  end

  popcount_accum #(.ACCUM_WIDTH(32)) pca(
    .clk(clk),
    .rst(rst),
    .in_value(popcount_value),
    .in_valid(popcount_valid),
    .drain(flush),
    .accum_out(accum_out),
    .accum_valid(accum_valid)
  );
endmodule
