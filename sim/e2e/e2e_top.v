`timescale 1ns/1ps
`include "parameters.vh"

// e2e_sim phase-1 top (2026-07-24): the REAL frontend + REAL
// softmc_pipeline (fetch/decode/execute/regfile), wired VERBATIM from
// softmc_top.v lines 470-560. The TB drives the h2c AXIS exactly where
// the silicon's post-CDC boundary sits. buffer_space is tied never-full
// (no readback engine in phase 1); completion oracle = softmc_fin.
// Phase 2 adds a behavioral DRAM array + readback engine behind ddr_*.
module e2e_top(
  input                               clk,
  input                               rst,
  input                               init_calib_complete,

  input  [`XDMA_AXI_DATA_WIDTH-1:0]   h2c_tdata,
  input                               h2c_tlast,
  input                               h2c_tvalid,
  output                              h2c_tready,
  input  [`XDMA_AXI_DATA_WIDTH/8-1:0] h2c_tkeep,

  output                              softmc_fin,
  output [3:0]                        ddr_read,
  output [3:0]                        ddr_write,
  output [3:0]                        ddr_act,
  output [3:0]                        ddr_pre,
  output [11:0]                       read_size,
  output                              read_seq_incoming,
  output                              fetch_restart_obs,
  output                              user_rst_obs,
  output                              frontend_ready_obs,
  output                              per_rd_init_obs,
  output                              per_zq_init_obs,
  output                              per_ref_init_obs,
  output [1:0]                        dbg_state,
  output                              dbg_maint_req,
  output                              dbg_maint_process,
  output                              dbg_fetch_hold
);

  wire [`IMEM_ADDR_WIDTH-1:0] fr_addr_in, fr_addr_out;
  wire                        fr_valid_in, fr_valid_out, fr_ready_out;
  wire [`INSTR_WIDTH-1:0]     fr_data_out;
  wire user_rst, fetch_restart, frontend_ready;

  wire [3:0] ddr_ref, ddr_sre, ddr_srx, ddr_zq, ddr_nop, ddr_ap,
             ddr_pall, ddr_half_bl, ddr_rank;
  wire [4*`HBM_CH_WIDTH-1:0] hbm_ch;
  wire [4*`BG_WIDTH-1:0]     ddr_bg;
  wire [4*`BANK_WIDTH-1:0]   ddr_bank;
  wire [4*`COL_WIDTH-1:0]    ddr_col;
  wire [4*`ROW_WIDTH-1:0]    ddr_row;
  wire [511:0]               ddr_wdata;

  assign fetch_restart_obs  = fetch_restart;
  assign user_rst_obs       = user_rst;
  assign frontend_ready_obs = frontend_ready;

  softmc_pipeline pipeline(
    .clk(clk),
    .rst(rst || user_rst),

    .softmc_end(softmc_fin),
    .read_size(read_size),
    .read_seq_incoming(read_seq_incoming),
    .buffer_space(12'hFFF),

    .addr_out(fr_addr_in),
    .valid_out(fr_valid_in),
    .data_in(fr_data_out),
    .valid_in(fr_valid_out),
    .addr_in(fr_addr_out),
    .restart(fetch_restart),
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
    .fetch_restart(fetch_restart),

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
    // unconnected in softmc_top too -> Vivado ties 0; sim matches
    .program_pending(1'b0),
    .dbg_state(dbg_state),
    .dbg_maint_req(dbg_maint_req),
    .dbg_maint_process(dbg_maint_process),
    .dbg_fetch_hold(dbg_fetch_hold),
    .rbe_switch_mode(),
    .rbe_set_read(),
    .rbe_set_diff(),
    .rbe_set_segpop(),
    .rbe_set_accxbp(),
    .rbe_set_accw(),
    .rbe_accw_pl(),
    .rbe_flush_acc()
  );

endmodule
