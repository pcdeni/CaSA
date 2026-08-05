`include "parameters.vh"
// =============================================================================
// frontend_linked.v — COPY of sources/hdl/verilog/frontend.v with ONE surgical
// addition: a link-route-config control word, following the EXACT convention of
// the existing host-set control words (bits [`INSTR_WIDTH+1..+4]).
//
//   NEW opcode: h2c_tdata_0[`INSTR_WIDTH+5]  (bit 69) = LINK-CFG
//     decoded in INIT_MEM_S like autoref/temp: single-beat, returns to IDLE,
//     does NOT touch IMEM. Exports a 1-cycle strobe + the 16-bit cfg word
//     {magic[10:3],dst[2:1],en[0]} to inter_bender_link (which latches ROUTE[s]
//     in this same core_ui_clk domain -> no config CDC).
//
// Property (a) preservation: with no bit[69] word ever sent, this module is
// cycle-identical to frontend.v (the new branch is mutually exclusive with every
// existing decode and with the IMEM-write fall-through). Diff vs frontend.v:
//   + 2 output ports (link_cfg_stb, link_cfg_data)
//   + link_cfg_ns/_r, link_cfg_data_ns/_r regs (mirrors hbm_temp_rd_ns/_r)
//   + one `else if` decode branch
//   + reset + register of the two new regs
// The base frontend is UNCHANGED in the CaSA tree (this is a local copy).
// =============================================================================
module frontend_linked#(parameter SIM_MEM = "false")(
  input                         clk,
  input                         rst,
  input                         softmc_fin,
  output                        user_rst,
  input                         init_calib_complete,
  output reg                    rbe_switch_mode,
  output reg                    dllt_begin,
  output                        frontend_ready,
  input  [`IMEM_ADDR_WIDTH-1:0] addr_in,
  input                         valid_in,
  output [`INSTR_WIDTH-1:0]     data_out,
  output                        valid_out,
  output [`IMEM_ADDR_WIDTH-1:0] addr_out,
  output                        ready_in,
  `ifdef HBM_BENDER
  output                              hbm_temp_rd,
  `endif
  input  [`XDMA_AXI_DATA_WIDTH-1:0]   h2c_tdata_0,
  input                               h2c_tlast_0,
  input                               h2c_tvalid_0,
  output                              h2c_tready_0,
  input  [`XDMA_AXI_DATA_WIDTH/8-1:0] h2c_tkeep_0,
  output                              per_rd_init,
  output                              per_zq_init,
  output                              per_ref_init,
  input                               program_pending,
  output [1:0]                        dbg_state,
  output                              dbg_maint_req,
  output                              dbg_maint_process,
  // ---- NEW: inter-bender link route-config export (L25/#76) ----
  output                              link_cfg_stb,   // 1-cycle pulse
  output [15:0]                       link_cfg_data   // {magic[10:3],dst[2:1],en[0]}
  );

  reg[31:0] delay_fin;
  always @(posedge clk) begin
    if(rst || user_rst)
      delay_fin <= 32'b0;
    else
      delay_fin[1+:31] <= delay_fin[0+:31];
      delay_fin[0]    <= softmc_fin;
  end
  assign frontend_ready = delay_fin[31];

  wire                        imem_wr_en, imem_rd_en;
  wire [`IMEM_ADDR_WIDTH-1:0] imem_addr;
  wire [`INSTR_WIDTH-1:0]     imem_wr_data, imem_rd_data;

  generate
    if(SIM_MEM == "true") begin
      instr_blk_mem_sim imem(.addra(imem_addr), .clka(clk), .dina(imem_wr_data),
        .douta(imem_rd_data), .ena(imem_rd_en || imem_wr_en), .wea(imem_wr_en));
    end else begin
      instr_blk_mem imem(.addra(imem_addr), .clka(clk), .dina(imem_wr_data),
        .douta(imem_rd_data), .ena(imem_rd_en || imem_wr_en), .wea(imem_wr_en));
    end
  endgenerate

  wire [`INSTR_WIDTH-1:0]     maint_inst;
  wire                        maint_valid;
  wire [`IMEM_ADDR_WIDTH-1:0] maint_addr;
  wire                        maint_req;
  reg                         maint_ack;
  wire                        maint_process;
  wire                        program_process;
  reg                         aref_en;
  reg                         aref_en_valid;

  maintenance_controller maint_ctrl (
    .clk(clk), .rst(rst | user_rst),
    .init_calib_complete(init_calib_complete), .softmc_fin(softmc_fin),
    .aref_en(aref_en), .aref_en_valid(aref_en_valid),
    .maint_req(maint_req), .maint_ack(maint_ack),
    .per_rd_init(per_rd_init), .per_zq_init(per_zq_init), .per_ref_init(per_ref_init),
    .maint_process(maint_process), .program_process(program_process),
    .in_addr(addr_in), .in_valid(valid_in),
    .out_data(maint_inst), .out_valid(maint_valid), .out_addr(maint_addr)
  );

  localparam IDLE_S = 2'd0, INIT_MEM_S = 2'd1, EXECUTE_S = 2'd2;
  reg [1:0] state_r, state_ns;
  reg [4:0]                  rst_ctr_ns, rst_ctr_r;
  reg [`IMEM_ADDR_WIDTH-1:0] xfer_ctr_r, xfer_ctr_ns;
  reg [`IMEM_RD_LATENCY-1:0] valid_out_sr;
  reg [(`IMEM_RD_LATENCY * `IMEM_ADDR_WIDTH)-1:0] addr_out_sr;
  reg hbm_temp_rd_ns, hbm_temp_rd_r;

  // NEW link-cfg regs (mirror the hbm_temp_rd _ns/_r pattern)
  reg        link_cfg_ns, link_cfg_r;
  reg [15:0] link_cfg_data_ns, link_cfg_data_r;
  assign link_cfg_stb  = link_cfg_r;
  assign link_cfg_data = link_cfg_data_r;

  assign hbm_temp_rd  = hbm_temp_rd_r;
  assign user_rst     = (|rst_ctr_r);
  assign h2c_tready_0 = state_r == INIT_MEM_S;
  assign imem_wr_en   = h2c_tvalid_0 && (state_r == INIT_MEM_S);
  assign imem_wr_data = h2c_tdata_0[`INSTR_WIDTH-1:0];
  assign imem_addr    = state_r == INIT_MEM_S ? xfer_ctr_r : addr_in;
  assign imem_rd_en   = valid_in && (program_process);
  assign data_out     = program_process ? imem_rd_data : maint_inst;
  assign valid_out    = program_process ? valid_out_sr[0] : maint_valid;
  assign addr_out     = program_process ? addr_out_sr[`IMEM_ADDR_WIDTH-1:0] : maint_addr;
  generate
  if(SIM_MEM=="false") assign ready_in = state_r == EXECUTE_S;
  else                 assign ready_in = state_r == EXECUTE_S && ~rst;
  endgenerate
  assign program_process = (state_r == EXECUTE_S) && ~maint_process;

  always @* begin
    aref_en_valid   = `LOW;
    aref_en         = `LOW;
    `ifdef HBM_BENDER
    hbm_temp_rd_ns  = `LOW;
    `endif
    link_cfg_ns      = `LOW;             // NEW: pulse defaults low
    link_cfg_data_ns = link_cfg_data_r;  // NEW: hold last written cfg word
    state_ns        = state_r;
    xfer_ctr_ns     = xfer_ctr_r;
    rst_ctr_ns      = {5{`LOW}};
    maint_ack       = `LOW;
    rbe_switch_mode = `LOW;
    dllt_begin      = `LOW;
    case (state_r)
      IDLE_S: begin
        if(~((|delay_fin) || softmc_fin)) begin
            if(h2c_tvalid_0)
              state_ns = INIT_MEM_S;
            else begin
              if(maint_req && !program_pending) begin
                maint_ack = `HIGH;
                state_ns = EXECUTE_S;
              end
            end
        end
      end
      INIT_MEM_S: begin
        if(h2c_tvalid_0) begin
          if(h2c_tdata_0[`INSTR_WIDTH]) //indicates a reset
            rst_ctr_ns = {5{1'b1}};
          else if(h2c_tdata_0[`INSTR_WIDTH+1]) // switch between readback modes
            rbe_switch_mode = `HIGH;
          else if(h2c_tdata_0[`INSTR_WIDTH+2]) // dll toggle off WIP
            dllt_begin = `HIGH;
          else if(h2c_tdata_0[`INSTR_WIDTH+3]) begin // enable-disable autoref
            aref_en_valid = `HIGH;
            aref_en       = h2c_tdata_0[0];
            state_ns      = IDLE_S;
          end
          `ifdef HBM_BENDER
          else if(h2c_tdata_0[`INSTR_WIDTH+4]) begin // read HBM temp
            hbm_temp_rd_ns  = `HIGH;
            state_ns        = IDLE_S;
          end
          `endif
          else if(h2c_tdata_0[`INSTR_WIDTH+5]) begin // NEW: inter-bender link route config
            link_cfg_ns      = `HIGH;
            link_cfg_data_ns = h2c_tdata_0[15:0];
            state_ns         = IDLE_S;
          end
          else begin
            xfer_ctr_ns = xfer_ctr_r + 1;
            if(h2c_tlast_0) begin
              state_ns    = EXECUTE_S;
              xfer_ctr_ns = {`IMEM_ADDR_WIDTH{`LOW}};
            end
          end
        end
      end
      EXECUTE_S: begin
        if(h2c_tvalid_0) begin
          if(h2c_tdata_0[`INSTR_WIDTH]) rst_ctr_ns = {5{1'b1}};
        end
        if(softmc_fin) state_ns = IDLE_S;
      end
    endcase
  end

  always @(posedge clk) begin
    if(rst || (|rst_ctr_r)) begin
      if(SIM_MEM == "false") state_r <= IDLE_S;
      else                   state_r <= EXECUTE_S;
      xfer_ctr_r   <= {`IMEM_ADDR_WIDTH{`LOW}};
      valid_out_sr <= {`IMEM_RD_LATENCY{`LOW}};
      addr_out_sr  <= 0;
      if(rst_ctr_r > 0) rst_ctr_r <= rst_ctr_r - 1;
      else              rst_ctr_r <= 0;
      `ifdef HBM_BENDER
      hbm_temp_rd_r <= 0;
      `endif
      link_cfg_r      <= 1'b0;         // NEW
      link_cfg_data_r <= 16'b0;        // NEW (default = all routes disabled)
    end
    else begin
      state_r      <= state_ns;
      xfer_ctr_r   <= xfer_ctr_ns;
      rst_ctr_r    <= rst_ctr_ns;
      valid_out_sr[`IMEM_RD_LATENCY-1] <= valid_in && (state_r == EXECUTE_S);
      addr_out_sr[`IMEM_RD_LATENCY*`IMEM_ADDR_WIDTH-1 :
          (`IMEM_RD_LATENCY-1)*`IMEM_ADDR_WIDTH] <= addr_in;
      `ifdef IMEM_SR
        valid_out_sr[`IMEM_RD_LATENCY-1:0] <= valid_out_sr >> 1;
        addr_out_sr[(`IMEM_RD_LATENCY-1)*`IMEM_ADDR_WIDTH-1:0]
                         <= addr_out_sr >> `IMEM_ADDR_WIDTH;
      `endif
      `ifdef HBM_BENDER
      hbm_temp_rd_r <= hbm_temp_rd_ns;
      `endif
      link_cfg_r      <= link_cfg_ns;        // NEW
      link_cfg_data_r <= link_cfg_data_ns;   // NEW
    end
  end

  assign dbg_state         = state_r;
  assign dbg_maint_req     = maint_req;
  assign dbg_maint_process = maint_process;
endmodule
