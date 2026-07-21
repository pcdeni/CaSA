// Sim-only dual-top: instantiates BOTH the baseline softmc_pipeline (with
// behavioral IMEM) and the new seq_engine in parallel. The C++ harness
// drives whichever one it's testing and observes its DDR command bus
// independently. A single Verilator binary owns both DUTs so we don't need
// to build / link two separate libraries.

`include "parameters.vh"
`include "encoding.vh"

module dual_top(
  input  clk,
  input  rst,

  // ----- baseline softmc_pipeline driver -----
  input                          base_run,
  input                          base_imem_we,
  input  [`IMEM_ADDR_WIDTH-1:0]  base_imem_addr,
  input  [`INSTR_WIDTH-1:0]      base_imem_data,
  output                         base_softmc_end,
  output [3:0]                   base_ddr_act,
  output [3:0]                   base_ddr_read,
  output [3:0]                   base_ddr_write,
  output [3:0]                   base_ddr_pre,
  output [3:0]                   base_ddr_ref,
  output [3:0]                   base_ddr_zq,
  output [4*2-1:0]               base_ddr_bank,
  output [4*10-1:0]              base_ddr_col,

  // ----- sequence engine driver -----
  input         seq_start,
  input  [1:0]  seq_op_mode,        // 0 = PLAIN, 1 = DOUBLE_ACT
  input  [2:0]  seq_op_code,
  input  [3:0]  seq_bank_mask,
  input  [1:0]  seq_bank0_id,
  input  [9:0]  seq_base_col,
  input  [16:0] seq_base_row,
  input  [9:0]  seq_count,
  input  [3:0]  seq_stride,
  input         seq_bl4,
  input         seq_ap,
  input  [16:0] seq_da_r_first,
  input  [16:0] seq_da_r_second,
  input  [5:0]  seq_da_t12,
  input  [5:0]  seq_da_t23,
  output        seq_busy,
  output        seq_done,
  output [3:0]  seq_ddr_act,
  output [3:0]  seq_ddr_read,
  output [3:0]  seq_ddr_write,
  output [3:0]  seq_ddr_pre,
  output [4*2-1:0]  seq_ddr_bank,
  output [4*10-1:0] seq_ddr_col,
  output [4*17-1:0] seq_ddr_row,

  // baseline ddr_row also exposed (for doubleACT comparison).
  output [4*17-1:0] base_ddr_row
);

  // ===== baseline =====
  softmc_pipeline_top base(
    .clk(clk),
    .rst(rst),
    .imem_init_we(base_imem_we),
    .imem_init_addr(base_imem_addr),
    .imem_init_data(base_imem_data),
    .run(base_run),
    .ddr_write(base_ddr_write),
    .ddr_read(base_ddr_read),
    .ddr_pre(base_ddr_pre),
    .ddr_act(base_ddr_act),
    .ddr_ref(base_ddr_ref),
    .ddr_zq(base_ddr_zq),
    .ddr_nop(),
    .ddr_ap(),
    .ddr_pall(),
    .ddr_half_bl(),
    .ddr_bg(),
    .ddr_bank(base_ddr_bank),
    .ddr_col(base_ddr_col),
    .ddr_row(base_ddr_row),
    .ddr_wdata(),
    .softmc_end(base_softmc_end),
    .read_size(),
    .read_seq_incoming()
  );

  // ===== sequence engine =====
  seq_engine se(
    .clk(clk),
    .rst(rst),
    .start(seq_start),
    .op_mode(seq_op_mode),
    .op_code(seq_op_code),
    .bank_mask(seq_bank_mask),
    .bank0_id(seq_bank0_id),
    .base_col(seq_base_col),
    .base_row(seq_base_row),
    .count(seq_count),
    .stride(seq_stride),
    .bl4(seq_bl4),
    .ap(seq_ap),
    .da_r_first(seq_da_r_first),
    .da_r_second(seq_da_r_second),
    .da_t12(seq_da_t12),
    .da_t23(seq_da_t23),
    .busy(seq_busy),
    .done(seq_done),
    .ddr_act(seq_ddr_act),
    .ddr_read(seq_ddr_read),
    .ddr_write(seq_ddr_write),
    .ddr_pre(seq_ddr_pre),
    .ddr_nop(),
    .ddr_ap(),
    .ddr_pall(),
    .ddr_half_bl(),
    .ddr_bg(),
    .ddr_bank(seq_ddr_bank),
    .ddr_col(seq_ddr_col),
    .ddr_row(seq_ddr_row)
  );
endmodule
