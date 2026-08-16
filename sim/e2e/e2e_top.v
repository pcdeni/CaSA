`timescale 1ns/1ps
`include "parameters.vh"

// e2e_sim phase-2 top (2026-07-24): the REAL frontend + softmc_pipeline
// (fetch/decode/execute/regfile) + the REAL readback_engine (build-13,
// magic 0C) + a behavioral DRAM (dram_model + dram_dpi.cpp) behind the
// ddr_* command interface. Wiring copied VERBATIM from softmc_top.v and
// softmc_core.v. The TB drives h2c and receives c2h — the loop is
// host-bytes-in -> host-bytes-out. buffer_space is REAL (engine-driven).
module e2e_top(
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
  output                              fetch_restart_obs,
  output                              user_rst_obs,
  output                              frontend_ready_obs,
  // Origin of the flush that frontend_ready announces: 1 = maintenance,
  // 0 = user program. The frontend has driven it since build 16; the
  // scenario suite counts maintenance vs user ready pulses with it.
  output                              frontend_ready_maint_obs,
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
  wire user_rst, fetch_restart, frontend_ready, frontend_ready_maint;

  wire [3:0] ddr_ref, ddr_sre, ddr_srx, ddr_zq, ddr_nop, ddr_ap,
             ddr_pall, ddr_half_bl, ddr_rank;
  wire [4*`HBM_CH_WIDTH-1:0] hbm_ch;
  wire [4*`BG_WIDTH-1:0]     ddr_bg;
  wire [4*`BANK_WIDTH-1:0]   ddr_bank;
  wire [4*`COL_WIDTH-1:0]    ddr_col;
  wire [4*`ROW_WIDTH-1:0]    ddr_row;
  wire [511:0]               ddr_wdata;

  wire [11:0] buffer_space;                   // REAL: driven by the rbe
  wire rbe_switch_mode, rbe_set_read, rbe_set_diff, rbe_set_segpop,
       rbe_set_accxbp, rbe_set_accw, rbe_flush_acc;
  wire [3:0] rbe_accw_pl;
  wire [511:0] dram_rd_data;
  wire         dram_rd_valid;

  assign fetch_restart_obs  = fetch_restart;
  assign user_rst_obs       = user_rst;
  assign frontend_ready_obs = frontend_ready;
  assign frontend_ready_maint_obs = frontend_ready_maint;

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
    // 2026-07-24 incident: the QUAD wrapper (softmc_core.v) left BOTH
    // restart ports unconnected since build-11 (Synth 8-7071 warnings,
    // unread) — silicon ran with restart==0 while this sim, wired from
    // softmc_top.v, had it connected. The define reproduces silicon-as-
    // was; the default is the build-14 (fixed softmc_core) wiring.
`ifdef SILICON_ASIS_UNWIRED_RESTART
    .restart(1'b0),
`else
    .restart(fetch_restart),
`endif
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
    .frontend_ready_maint(frontend_ready_maint),

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
    .rbe_switch_mode(rbe_switch_mode),
    .rbe_set_read(rbe_set_read),
    .rbe_set_diff(rbe_set_diff),
    .rbe_set_segpop(rbe_set_segpop),
    .rbe_set_accxbp(rbe_set_accxbp),
    .rbe_set_accw(rbe_set_accw),
    .rbe_accw_pl(rbe_accw_pl),
    .rbe_flush_acc(rbe_flush_acc)
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

  // wiring copied from softmc_core.v (rbe instantiation)
  readback_engine rbe(
    .clk(clk),
    .rst(rst || user_rst),

    .flush(frontend_ready),
    .switch_mode(rbe_switch_mode),
    .set_mode_read(rbe_set_read),
    .set_mode_diff(rbe_set_diff),
    .set_mode_segpop(rbe_set_segpop),
    .set_mode_accxbp(rbe_set_accxbp),
    .set_acc_weight(rbe_set_accw),
    .acc_weight_pl(rbe_accw_pl),
    .flush_acc(rbe_flush_acc),

    .read_seq_incoming(read_seq_incoming),
    .incoming_reads(read_size),
    .buffer_space(buffer_space),

    .rd_data(dram_rd_data),
    .rd_valid(dram_rd_valid),

    .ddr_wdata(ddr_wdata),

    .per_rd_init(per_rd_init_obs),
    .per_zq_init(per_zq_init_obs),
    .per_ref_init(per_ref_init_obs),

    .c2h_tdata_0(c2h_tdata),
    .c2h_tlast_0(c2h_tlast),
    .c2h_tvalid_0(c2h_tvalid),
    .c2h_tready_0(c2h_tready),
    .c2h_tkeep_0(c2h_tkeep)
  );

endmodule
