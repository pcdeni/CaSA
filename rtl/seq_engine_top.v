// Verilator wrapper for seq_engine.v — exposes the opcode interface and the
// DDR command bus so the C++ harness can drive bursts and capture each
// cycle's command output for comparison vs the baseline softmc_pipeline.

`include "encoding.vh"

module seq_engine_top(
  input         clk,
  input         rst,

  // opcode
  input         start,
  input  [2:0]  op_code,
  input  [3:0]  bank_mask,
  input  [1:0]  bank0_id,
  input  [9:0]  base_col,
  input  [16:0] base_row,
  input  [9:0]  count,
  input  [3:0]  stride,
  input         bl4,
  input         ap,

  output        busy,
  output        done,

  output [3:0]      ddr_act,
  output [3:0]      ddr_read,
  output [3:0]      ddr_write,
  output [3:0]      ddr_pre,
  output [3:0]      ddr_nop,
  output [3:0]      ddr_ap,
  output [3:0]      ddr_pall,
  output [3:0]      ddr_half_bl,
  output [4*2-1:0]  ddr_bg,
  output [4*2-1:0]  ddr_bank,
  output [4*10-1:0] ddr_col,
  output [4*17-1:0] ddr_row
);
  seq_engine se(
    .clk(clk),
    .rst(rst),
    .start(start),
    .op_code(op_code),
    .bank_mask(bank_mask),
    .bank0_id(bank0_id),
    .base_col(base_col),
    .base_row(base_row),
    .count(count),
    .stride(stride),
    .bl4(bl4),
    .ap(ap),
    .busy(busy),
    .done(done),
    .ddr_act(ddr_act),
    .ddr_read(ddr_read),
    .ddr_write(ddr_write),
    .ddr_pre(ddr_pre),
    .ddr_nop(ddr_nop),
    .ddr_ap(ddr_ap),
    .ddr_pall(ddr_pall),
    .ddr_half_bl(ddr_half_bl),
    .ddr_bg(ddr_bg),
    .ddr_bank(ddr_bank),
    .ddr_col(ddr_col),
    .ddr_row(ddr_row)
  );
endmodule
