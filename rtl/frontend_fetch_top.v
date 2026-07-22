`include "parameters.vh"
// build9 streaming TB top: wires the real frontend (DUT) to the real
// fetch_stage + pre_decode, a behavioral IMEM (tb_stubs.v) and a stub
// maintenance controller. Mirrors softmc_core's frontend<->fetch
// connection. The TB drives h2c (producer) + softmc_fin (execute done)
// and observes the dispatched instruction stream (instr/instr_valid) and
// softmc_end (END fetched → time the fin pulse).
module frontend_fetch_top(
  input                          clk,
  input                          rst,
  input                          init_calib_complete,
  // XDMA producer side
  input  [`XDMA_AXI_DATA_WIDTH-1:0] h2c_tdata_0,
  input                          h2c_tlast_0,
  input                          h2c_tvalid_0,
  output                         h2c_tready_0,
  // execute-done pulse (TB-modelled)
  input                          softmc_fin,
  // observation
  output [`INSTR_WIDTH-1:0]      obs_instr,
  output                         obs_instr_valid,
  output [`IMEM_ADDR_WIDTH-1:0]  obs_instr_pc,
  output                         obs_softmc_end,
  output [1:0]                   obs_state,
  output                         obs_exec_bank,
  output [1:0]                   obs_loaded,
  output                         obs_swap_pending,
  output                         obs_fetch_hold,
  output                         obs_tready
);
  // frontend <-> fetch wires
  wire [`IMEM_ADDR_WIDTH-1:0] fe_addr_in;   // fetch -> frontend (req addr)
  wire                        fe_valid_in;
  wire [`INSTR_WIDTH-1:0]     fe_data_out;  // frontend -> fetch
  wire                        fe_valid_out;
  wire [`IMEM_ADDR_WIDTH-1:0] fe_addr_out;
  wire                        fe_ready_in;  // frontend ready for a req

  wire                        fs_softmc_end;
  wire [11:0]                 fs_read_size;
  wire                        fs_read_seq_incoming;

  // large buffer_space so need_flush never fires in the TB
  wire [11:0] buffer_space = 12'hFFF;

  // frontend outputs we don't check (mode SETs etc.) — leave open
  wire rbe_switch_mode, rbe_set_read, rbe_set_diff, rbe_set_segpop;
  wire rbe_set_accxbp, rbe_set_accw, rbe_flush_acc, dllt_begin;
  wire [3:0] rbe_accw_pl;
  wire user_rst, frontend_ready;
  wire per_rd_init, per_zq_init, per_ref_init;
  wire dbg_maint_req, dbg_maint_process;

  frontend #(.SIM_MEM("false")) dut(
    .clk(clk), .rst(rst),
    .softmc_fin(softmc_fin),
    .user_rst(user_rst),
    .init_calib_complete(init_calib_complete),
    .rbe_switch_mode(rbe_switch_mode),
    .rbe_set_read(rbe_set_read), .rbe_set_diff(rbe_set_diff),
    .rbe_set_segpop(rbe_set_segpop), .rbe_set_accxbp(rbe_set_accxbp),
    .rbe_set_accw(rbe_set_accw), .rbe_accw_pl(rbe_accw_pl),
    .rbe_flush_acc(rbe_flush_acc), .dllt_begin(dllt_begin),
    .frontend_ready(frontend_ready),
    .addr_in(fe_addr_in), .valid_in(fe_valid_in),
    .data_out(fe_data_out), .valid_out(fe_valid_out),
    .addr_out(fe_addr_out), .ready_in(fe_ready_in),
    .h2c_tdata_0(h2c_tdata_0), .h2c_tlast_0(h2c_tlast_0),
    .h2c_tvalid_0(h2c_tvalid_0), .h2c_tready_0(h2c_tready_0),
    .h2c_tkeep_0({(`XDMA_AXI_DATA_WIDTH/8){1'b1}}),
    .per_rd_init(per_rd_init), .per_zq_init(per_zq_init),
    .per_ref_init(per_ref_init),
    .program_pending(1'b0),
    .dbg_state(obs_state), .dbg_maint_req(dbg_maint_req),
    .dbg_maint_process(dbg_maint_process),
    .dbg_exec_bank(obs_exec_bank), .dbg_loaded(obs_loaded),
    .dbg_swap_pending(obs_swap_pending), .dbg_fetch_hold(obs_fetch_hold)
  );
  assign obs_tready = h2c_tready_0;

  fetch_stage fs(
    .clk(clk), .rst(rst),
    .softmc_end(fs_softmc_end),
    .read_size(fs_read_size),
    .read_seq_incoming(fs_read_seq_incoming),
    .buffer_space(buffer_space),
    .br_resolve(1'b0),
    .br_target({`IMEM_ADDR_WIDTH{1'b0}}),
    // fetch <-> frontend
    .addr_out(fe_addr_in),   // fetch requests addr -> frontend.addr_in
    .valid_out(fe_valid_in),
    .data_in(fe_data_out),   // frontend.data_out -> fetch.data_in
    .valid_in(fe_valid_out),
    .addr_in(fe_addr_out),
    .ready_out(fe_ready_in),
    // fetch -> decode (observation)
    .instr(obs_instr),
    .instr_pc(obs_instr_pc),
    .instr_valid(obs_instr_valid)
  );

  assign obs_softmc_end = fs_softmc_end;
endmodule
