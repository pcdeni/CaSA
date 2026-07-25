`timescale 1ns/1ps
`include "parameters.vh"

// GOLDEN REFERENCE top (2026-07-25): UNMODIFIED upstream DRAM Bender
// frontend + softmc_pipeline + readback_engine, in the SAME harness our
// e2e_sim uses (same behavioural DRAM, same IMEM model, same FIFO model,
// same TB, same programs).
//
// Purpose: record what the PHY sees and what the host receives for a
// given program, from a design nobody has modified — so any difference
// our engine shows is OURS, named and justified or fixed. Upstream has
// no streaming and no SEG_POP, so the comparison runs in LEGACY cadence
// and READ mode, which is exactly the regime both designs must agree on.
module e2e_ref_top(
  input                               clk,
  input                               rst,
  input                               init_calib_complete,

  input  [`XDMA_AXI_DATA_WIDTH-1:0]   h2c_tdata,
  input                               h2c_tlast,
  input                               h2c_tvalid,
  output                              h2c_tready,
  input  [`XDMA_AXI_DATA_WIDTH/8-1:0] h2c_tkeep,

  output [`XDMA_AXI_DATA_WIDTH-1:0]   c2h_tdata,
  output                              c2h_tlast,
  output                              c2h_tvalid,
  input                               c2h_tready,
  output [`XDMA_AXI_DATA_WIDTH/8-1:0] c2h_tkeep,

  output                              softmc_fin,
  output [3:0]                        ddr_read,
  output [3:0]                        ddr_write,
  output [3:0]                        ddr_act,
  output [3:0]                        ddr_pre,
  output [11:0]                       read_size,
  output                              read_seq_incoming,
  output                              frontend_ready_obs,
  output                              user_rst_obs,
  output                              per_rd_init_obs,
  output                              per_zq_init_obs,
  output                              per_ref_init_obs
);

  wire [`IMEM_ADDR_WIDTH-1:0] fr_addr_in, fr_addr_out;
  wire                        fr_valid_in, fr_valid_out, fr_ready_out;
  wire [`INSTR_WIDTH-1:0]     fr_data_out;
  wire user_rst, frontend_ready;

  wire [3:0] ddr_ref, ddr_sre, ddr_srx, ddr_zq, ddr_nop, ddr_ap,
             ddr_pall, ddr_half_bl, ddr_rank;
  wire [4*`HBM_CH_WIDTH-1:0] hbm_ch;
  wire [4*`BG_WIDTH-1:0]     ddr_bg;
  wire [4*`BANK_WIDTH-1:0]   ddr_bank;
  wire [4*`COL_WIDTH-1:0]    ddr_col;
  wire [4*`ROW_WIDTH-1:0]    ddr_row;
  wire [511:0]               ddr_wdata;

  wire [11:0]  buffer_space;
  wire         rbe_switch_mode;
  wire [511:0] dram_rd_data;
  wire         dram_rd_valid;

  assign frontend_ready_obs = frontend_ready;
  assign user_rst_obs       = user_rst;

  softmc_pipeline pipeline(
    .clk(clk),
    .rst(rst || user_rst),

    .softmc_end(softmc_fin),
    .read_size(read_size),
    .read_seq_incoming(read_seq_incoming),
    .buffer_space(buffer_space),

    .addr_out(fr_addr_in),
    .valid_out(fr_valid_in),
    .data_in(fr_data_out),
    .valid_in(fr_valid_out),
    .addr_in(fr_addr_out),
    .ready_out(fr_ready_out),

    .ddr_write(ddr_write),
    .ddr_read(ddr_read),
    .ddr_pre(ddr_pre),
    .ddr_act(ddr_act),
    .ddr_ref(ddr_ref),
    .ddr_sre(ddr_sre),
    .ddr_srx(ddr_srx),
    .ddr_zq(ddr_zq),
    .ddr_nop(ddr_nop),
    .ddr_ap(ddr_ap),
    .ddr_pall(ddr_pall),
    .ddr_half_bl(ddr_half_bl),
    .ddr_rank(ddr_rank),
    .hbm_ch(hbm_ch),
    .ddr_bg(ddr_bg),
    .ddr_bank(ddr_bank),
    .ddr_col(ddr_col),
    .ddr_row(ddr_row),
    .ddr_wdata(ddr_wdata)
  );

  frontend #(.SIM_MEM("true")) frontend(
    .clk(clk),
    .rst(rst),

    .init_calib_complete(init_calib_complete),
    .softmc_fin(softmc_fin),
    .user_rst(user_rst),

    .dllt_begin(),
    .frontend_ready(frontend_ready),

    .addr_in(fr_addr_in),
    .valid_in(fr_valid_in),
    .data_out(fr_data_out),
    .valid_out(fr_valid_out),
    .addr_out(fr_addr_out),
    .ready_in(fr_ready_out),

    .h2c_tdata_0(h2c_tdata),
    .h2c_tlast_0(h2c_tlast),
    .h2c_tvalid_0(h2c_tvalid),
    .h2c_tready_0(h2c_tready),
    .h2c_tkeep_0(h2c_tkeep),

    .per_rd_init(per_rd_init_obs),
    .per_zq_init(per_zq_init_obs),
    .per_ref_init(per_ref_init_obs),
    .rbe_switch_mode(rbe_switch_mode)
  );

  dram_model dram(
    .clk(clk),
    .rst(rst),
    .ddr_act(ddr_act),
    .ddr_read(ddr_read),
    .ddr_write(ddr_write),
    .ddr_pre(ddr_pre),
    .ddr_bg(ddr_bg),
    .ddr_bank(ddr_bank),
    .ddr_col(ddr_col),
    .ddr_row(ddr_row),
    .ddr_wdata(ddr_wdata),
    .rd_data(dram_rd_data),
    .rd_valid(dram_rd_valid)
  );

  readback_engine rbe(
    .clk(clk),
    .rst(rst || user_rst),

    .flush(frontend_ready),
    .switch_mode(rbe_switch_mode),

    .read_seq_incoming(read_seq_incoming),
    .incoming_reads(read_size),
    .buffer_space(buffer_space),

    .rd_data(dram_rd_data),
    .rd_valid(dram_rd_valid),

    .ddr_wdata(ddr_wdata),

    .per_rd_init(per_rd_init_obs),
    .per_zq_init(per_zq_init_obs),
    .per_ref_init(per_ref_init_obs),

    .hbm_temp_rd(1'b0),
    .hbm0_temp(7'd0),
    .hbm1_temp(7'd0),

    .c2h_tdata_0(c2h_tdata),
    .c2h_tlast_0(c2h_tlast),
    .c2h_tvalid_0(c2h_tvalid),
    .c2h_tready_0(c2h_tready),
    .c2h_tkeep_0(c2h_tkeep)
  );

endmodule
