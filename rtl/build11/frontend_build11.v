`include "parameters.vh"
/*
 * This module is responsible for the interface
 * between XDMA IP and the fetch stage.
 * This module encapsulates a X KiB BRAM which
 * is used as an instruction memory.
 *
 * build9 (2026-07-22): STREAMING PROGRAM FETCH (Rung 1,
 * docs/STREAMING_FETCH_DESIGN.md). The serialization point of the whole
 * host-orchestrated system is the single line
 *     h2c_tready_0 = (state == INIT_MEM_S)
 * — during EXECUTE_S the host cannot stream the next program, so every
 * program pays a full host round-trip (~150-200 us) of fetch-idle.
 * build9 adds a PING-PONG IMEM pair: while fetch executes program N from
 * the active bank, the h2c loader fills the idle bank with N+1; on
 * program end the banks swap and execution continues without leaving
 * EXECUTE_S. Gated by a new idempotent control word STREAM_EN
 * (bit INSTR_WIDTH+11, payload tdata[0] = on/off): with streaming OFF
 * (reset default) the FSM is behavior-identical to build8 — the second
 * bank simply alternates as the load target, invisible to software.
 *
 * Streaming rules (v1 scope, enforced by the HOST contract):
 *  - Only READ / SEG_POP readback modes may stream (per-program trailers
 *    are the in-order result delimiters). Accum-family modes keep the
 *    execute->receive cadence (their capture logic assumes read-quiet
 *    windows that overlapped programs would destroy).
 *  - Mid-EXECUTE control words: only reset (INSTR_WIDTH), the idempotent
 *    SETs (+5..+10) and STREAM_EN (+11) are legal while streaming;
 *    +1..+4 (legacy toggle, dllt, aref, hbm-temp) remain
 *    INIT_MEM/IDLE-only.
 *  - The bank swap waits for frontend_ready (softmc_fin delayed 32) so
 *    the readback trailer of program N is enqueued before N+1's first
 *    fetch can issue — and fetch is held (ready_in low) across the
 *    fin->swap window so the old bank's word 0 cannot re-dispatch (the
 *    legacy design relied on leaving EXECUTE_S for that).
 *  - If the next bank is not loaded when a program ends, degrade to the
 *    legacy IDLE_S path: a slow producer gets exactly build8 behavior,
 *    never worse. A partially-streamed load survives the seam (the
 *    write pointer and bank assignment are state-independent).
 */
module frontend#(parameter SIM_MEM = "false")(
  // common signals
  input                         clk,
  input                         rst,

  // other control signals
  input                         softmc_fin,
  output                        user_rst,
  // build11: pulse to fetch_stage at every program start (streamed
  // swap, legacy INIT_MEM handoff, maintenance entry) — forces pc=0.
  output reg                    fetch_restart,
  input                         init_calib_complete,
  output reg                    rbe_switch_mode,
  // build4 (2026-07-20): idempotent readback-mode SET words. The legacy
  // INSTR_WIDTH+1 toggle is decode-state dependent on silicon (lost or
  // double-applied per boot phase — RESULT.md addendum 20c); a lost or
  // duplicated TOGGLE flips parity, while a lost SET is repaired by
  // simply sending it again. Control words: bit INSTR_WIDTH+5 = SET
  // READ_MODE, bit INSTR_WIDTH+6 = SET DIFF_MODE (+4 is taken by the
  // HBM_BENDER temp-read word). Projects that do not connect these
  // outputs are unaffected.
  output reg                    rbe_set_read,
  output reg                    rbe_set_diff,
  // build7: SET SEG_POP readout mode (bit INSTR_WIDTH+7). Per-32b-segment
  // popcount readback; idempotent SET, same pattern as +5/+6.
  output reg                    rbe_set_segpop,
  // build8: ACCUM_XBP control words. +8 = SET mode, +9 = SET weight
  // (payload {neg,shift[2:0]} in tdata[3:0]), +10 = FLUSH accumulator.
  output reg                    rbe_set_accxbp,
  output reg                    rbe_set_accw,
  output [3:0]                  rbe_accw_pl,
  output reg                    rbe_flush_acc,
  output reg                    dllt_begin,
  output                        frontend_ready,

  // frontend <-> fetch stage interface
  input  [`IMEM_ADDR_WIDTH-1:0] addr_in,
  input                         valid_in,
  output [`INSTR_WIDTH-1:0]     data_out,
  output                        valid_out,
  output [`IMEM_ADDR_WIDTH-1:0] addr_out,
  output                        ready_in,

  `ifdef HBM_BENDER
  output                              hbm_temp_rd,
  `endif

  // frontend <-> xdma interface
  input  [`XDMA_AXI_DATA_WIDTH-1:0]   h2c_tdata_0,
  input                               h2c_tlast_0,
  input                               h2c_tvalid_0,
  output                              h2c_tready_0,
  input  [`XDMA_AXI_DATA_WIDTH/8-1:0] h2c_tkeep_0,

  // maintenance signals
  output                              per_rd_init,
  output                              per_zq_init,
  output                              per_ref_init,

  // program_pending: when asserted, blocks maint_req from preempting IDLE_S.
  // Use this when an on-chip instruction source (e.g., u_rowclone_app) is about
  // to drive H2C, to prevent maintenance from racing into EXECUTE_S first.
  input                               program_pending,

  // Debug outputs (unconnected-safe; optional taps for uram_capture)
  output [1:0]                        dbg_state,
  output                              dbg_maint_req,
  output                              dbg_maint_process,
  // build9 streaming debug taps (unconnected-safe)
  output                              dbg_exec_bank,
  output [1:0]                        dbg_loaded,
  output                              dbg_swap_pending,
  output                              dbg_fetch_hold
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

  // ------------------------------------------------------------------
  // build9: ping-pong IMEM pair. exec_bank_r = the bank fetch reads;
  // the OTHER bank (load_bank) is the only write target. loaded_r[b]
  // marks bank b holding a complete, not-yet-consumed program.
  // ------------------------------------------------------------------
  wire                        imem_rd_en;
  wire [`INSTR_WIDTH-1:0]     imem_wr_data;
  wire [`INSTR_WIDTH-1:0]     imem0_rd_data, imem1_rd_data;
  wire                        imem0_wr_en, imem1_wr_en;
  wire [`IMEM_ADDR_WIDTH-1:0] imem0_addr, imem1_addr;

  reg        exec_bank_r,    exec_bank_ns;
  reg [1:0]  loaded_r,       loaded_ns;
  reg        stream_en_r,    stream_en_ns;    // +11 STREAM_EN payload bit0
  reg        fetch_hold_r,   fetch_hold_ns;   // fin->swap window guard
  reg        swap_pending_r, swap_pending_ns;
  reg [2:0]  swap_settle_r,  swap_settle_ns;  // post-swap read-pipe settle
  wire       load_bank = ~exec_bank_r;

  generate
    if(SIM_MEM == "true") begin
      instr_blk_mem_sim imem0(
      .addra(imem0_addr), .clka(clk), .dina(imem_wr_data),
      .douta(imem0_rd_data),
      .ena((imem_rd_en && (exec_bank_r == 1'b0)) || imem0_wr_en),
      .wea(imem0_wr_en));
      instr_blk_mem_sim imem1(
      .addra(imem1_addr), .clka(clk), .dina(imem_wr_data),
      .douta(imem1_rd_data),
      .ena((imem_rd_en && (exec_bank_r == 1'b1)) || imem1_wr_en),
      .wea(imem1_wr_en));
    end
    else begin
      instr_blk_mem imem0(
      .addra(imem0_addr), .clka(clk), .dina(imem_wr_data),
      .douta(imem0_rd_data),
      .ena((imem_rd_en && (exec_bank_r == 1'b0)) || imem0_wr_en),
      .wea(imem0_wr_en));
      instr_blk_mem imem1(
      .addra(imem1_addr), .clka(clk), .dina(imem_wr_data),
      .douta(imem1_rd_data),
      .ena((imem_rd_en && (exec_bank_r == 1'b1)) || imem1_wr_en),
      .wea(imem1_wr_en));
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

  maintenance_controller maint_ctrl
  (
    .clk(clk),
    .rst(rst | user_rst),

    .init_calib_complete(init_calib_complete),
    .softmc_fin(softmc_fin),

    .aref_en(aref_en),
    .aref_en_valid(aref_en_valid),
    .maint_req(maint_req),
    .maint_ack(maint_ack),
    .per_rd_init(per_rd_init),
    .per_zq_init(per_zq_init),
    .per_ref_init(per_ref_init),
    .maint_process(maint_process),
    .program_process(program_process),

    .in_addr(addr_in),
    .in_valid(valid_in),

    .out_data(maint_inst),
    .out_valid(maint_valid),
    .out_addr(maint_addr)
  );

  localparam IDLE_S     = 2'd0;
  localparam INIT_MEM_S = 2'd1;
  localparam EXECUTE_S  = 2'd2;

  reg [1:0] state_r, state_ns;

  reg [4:0]                  rst_ctr_ns, rst_ctr_r;
  reg [`IMEM_ADDR_WIDTH-1:0] xfer_ctr_r, xfer_ctr_ns;
  reg [`IMEM_RD_LATENCY-1:0] valid_out_sr;
  reg [(`IMEM_RD_LATENCY * `IMEM_ADDR_WIDTH)-1:0] addr_out_sr;

  reg hbm_temp_rd_ns, hbm_temp_rd_r;

  assign hbm_temp_rd = hbm_temp_rd_r;

  assign user_rst     = (|rst_ctr_r);

  // Any control bit set => the word is NOT an instruction. Covers
  // INSTR_WIDTH .. INSTR_WIDTH+11 (build9's STREAM_EN included).
  wire h2c_is_ctrl_any = |h2c_tdata_0[`INSTR_WIDTH +: 12];

  // imem <-> xdma interface
  // build9: the host may also stream while a program executes, into the
  // idle bank, once STREAM_EN is set — but only while that bank is not
  // already holding a complete pending program (XDMA back-pressures the
  // producer via tready, exactly as it back-pressured INIT_MEM loads).
  assign h2c_tready_0 = (state_r == INIT_MEM_S)
      || (stream_en_r && (state_r == EXECUTE_S) && ~loaded_r[load_bank]);
  wire   h2c_fire     = h2c_tvalid_0 && h2c_tready_0;
  // Writes always target the load bank; like build8, a control word may
  // transiently write its slot, and the real instruction overwrites it
  // (xfer_ctr only advances on instruction words).
  wire   imem_wr_fire = h2c_fire &&
      ((state_r == INIT_MEM_S) || (stream_en_r && (state_r == EXECUTE_S)));
  assign imem_wr_data = h2c_tdata_0[`INSTR_WIDTH-1:0];
  assign imem0_wr_en  = imem_wr_fire && (load_bank == 1'b0);
  assign imem1_wr_en  = imem_wr_fire && (load_bank == 1'b1);
  // Per-bank address: the exec bank serves fetch, the load bank takes
  // the write pointer — both in the same cycle, which is the point.
  assign imem0_addr   = (exec_bank_r == 1'b0) ? addr_in : xfer_ctr_r;
  assign imem1_addr   = (exec_bank_r == 1'b1) ? addr_in : xfer_ctr_r;
  // imem <-> pipeline interface
  assign imem_rd_en   = valid_in && (program_process);
  assign data_out     = program_process
      ? (exec_bank_r ? imem1_rd_data : imem0_rd_data) : maint_inst;
  assign valid_out    = program_process ? valid_out_sr[0] : maint_valid;
  assign addr_out     = program_process ? addr_out_sr[`IMEM_ADDR_WIDTH-1:0] : maint_addr;

  // build9: hold fetch across the fin->swap window (legacy relied on
  // leaving EXECUTE_S to stop the post-END pc=0 refetch).
  generate
  if(SIM_MEM=="false")
    assign ready_in     = (state_r == EXECUTE_S) && ~fetch_hold_r;
  else
    assign ready_in     = (state_r == EXECUTE_S) && ~fetch_hold_r && ~rst;
  endgenerate
  assign program_process = (state_r == EXECUTE_S) && ~maint_process;

  always @* begin
    aref_en_valid   = `LOW;
    aref_en         = `LOW;
    `ifdef HBM_BENDER
    hbm_temp_rd_ns  = `LOW;
    `endif
    state_ns        = state_r;
    xfer_ctr_ns     = xfer_ctr_r;
    rst_ctr_ns      = {5{`LOW}};
    maint_ack       = `LOW;
    rbe_switch_mode = `LOW;
    rbe_set_read    = `LOW;
    rbe_set_diff    = `LOW;
    rbe_set_segpop  = `LOW;
    rbe_set_accxbp  = `LOW;
    rbe_set_accw    = `LOW;
    rbe_flush_acc   = `LOW;
    dllt_begin      = `LOW;
    exec_bank_ns    = exec_bank_r;
    loaded_ns       = loaded_r;
    stream_en_ns    = stream_en_r;
    fetch_hold_ns   = fetch_hold_r;
    swap_pending_ns = swap_pending_r;
    swap_settle_ns  = swap_settle_r;
    fetch_restart   = 1'b0;
    case (state_r)
      IDLE_S: begin
        if(~((|delay_fin) || softmc_fin)) begin
            if(h2c_tvalid_0)
              state_ns = INIT_MEM_S;
            else begin
              // program_pending blocks maint_req to prevent a race where
              // maintenance enters EXECUTE_S before H2C instructions arrive
              // through the CDC FIFO or an on-chip test controller.
              if(maint_req && !program_pending) begin
                maint_ack = `HIGH;
                state_ns = EXECUTE_S;
                fetch_restart = 1'b1;   // build11: pc=0 at maint start
              end
            end
        end
      end
      INIT_MEM_S: begin
        if(h2c_tvalid_0) begin
          if(h2c_tdata_0[`INSTR_WIDTH]) //indicates a reset
            rst_ctr_ns = {5{1'b1}};
          else if(h2c_tdata_0[`INSTR_WIDTH+1]) // indicate switch between readback modes
            rbe_switch_mode = `HIGH;
          else if(h2c_tdata_0[`INSTR_WIDTH+2]) // indicate dll toggle off WIP
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
          else if(h2c_tdata_0[`INSTR_WIDTH+5]) begin // build4: SET READ_MODE (idempotent)
            rbe_set_read = `HIGH;
            state_ns     = IDLE_S; // return to IDLE (aref-word pattern):
                                   // no INIT_MEM_S camping, maintenance
                                   // stays schedulable after a SET word
          end
          else if(h2c_tdata_0[`INSTR_WIDTH+6]) begin // build4: SET DIFF_MODE (idempotent)
            rbe_set_diff = `HIGH;
            state_ns     = IDLE_S;
          end
          else if(h2c_tdata_0[`INSTR_WIDTH+7]) begin // build7: SET SEG_POP (idempotent)
            rbe_set_segpop = `HIGH;
            state_ns       = IDLE_S;
          end
          else if(h2c_tdata_0[`INSTR_WIDTH+8]) begin // build8: SET ACCUM_XBP
            rbe_set_accxbp = `HIGH;
            state_ns       = IDLE_S;
          end
          else if(h2c_tdata_0[`INSTR_WIDTH+9]) begin // build8: SET_ACC_WEIGHT
            rbe_set_accw   = `HIGH;
            state_ns       = IDLE_S;
          end
          else if(h2c_tdata_0[`INSTR_WIDTH+10]) begin // build8: FLUSH_ACC
            rbe_flush_acc  = `HIGH;
            state_ns       = IDLE_S;
          end
          else if(h2c_tdata_0[`INSTR_WIDTH+11]) begin // build9: STREAM_EN (idempotent)
            stream_en_ns   = h2c_tdata_0[0];
            state_ns       = IDLE_S;
          end
          else begin
            xfer_ctr_ns = xfer_ctr_r + 1;
            if(h2c_tlast_0) begin
              // build9: the program just landed in load_bank — make it
              // the exec bank and run it (legacy flow, bank-alternating).
              exec_bank_ns  = load_bank;
              fetch_hold_ns = `LOW;
              fetch_restart = 1'b1;   // build11: pc=0 at legacy start
              state_ns      = EXECUTE_S;
              xfer_ctr_ns   = {`IMEM_ADDR_WIDTH{`LOW}};
            end
          end
        end
      end
      EXECUTE_S: begin
        // Legacy peek-reset: acts even while tready is low (the word is
        // then consumed and re-decoded in INIT_MEM later — idempotent).
        if(h2c_tvalid_0) begin
          if(h2c_tdata_0[`INSTR_WIDTH]) //indicates a reset
            rst_ctr_ns = {5{1'b1}};
        end
        // build9: words that actually TRANSFER during execution
        // (stream_en_r && load bank free). Only the idempotent SETs,
        // STREAM_EN and instruction words are legal mid-stream (host
        // contract); +1..+4 stay INIT_MEM/IDLE-only.
        if(h2c_fire) begin
          if(h2c_tdata_0[`INSTR_WIDTH])
            ; // reset — already handled by the peek above
          else if(h2c_tdata_0[`INSTR_WIDTH+5])  rbe_set_read   = `HIGH;
          else if(h2c_tdata_0[`INSTR_WIDTH+6])  rbe_set_diff   = `HIGH;
          else if(h2c_tdata_0[`INSTR_WIDTH+7])  rbe_set_segpop = `HIGH;
          else if(h2c_tdata_0[`INSTR_WIDTH+8])  rbe_set_accxbp = `HIGH;
          else if(h2c_tdata_0[`INSTR_WIDTH+9])  rbe_set_accw   = `HIGH;
          else if(h2c_tdata_0[`INSTR_WIDTH+10]) rbe_flush_acc  = `HIGH;
          else if(h2c_tdata_0[`INSTR_WIDTH+11]) stream_en_ns   = h2c_tdata_0[0];
          else if(~h2c_is_ctrl_any) begin       // instruction word
            xfer_ctr_ns = xfer_ctr_r + 1;
            if(h2c_tlast_0) begin
              loaded_ns[load_bank] = `HIGH;     // program N+1 fully staged
              xfer_ctr_ns = {`IMEM_ADDR_WIDTH{`LOW}};
            end
          end
        end
        // Program end: swap if the next program is staged, else degrade
        // to the legacy IDLE path (slow producer == build8 behavior).
        if(softmc_fin) begin
          // build11: count a program whose tlast lands THIS cycle —
          // fin coincident with staging otherwise drops to IDLE with a
          // staged bank + a stranded stream (TB-caught race).
          if(stream_en_r && (loaded_r[load_bank] ||
                             (imem_wr_fire && h2c_tlast_0))) begin
            swap_pending_ns = `HIGH;   // swap on frontend_ready (fin+32):
            fetch_hold_ns   = `HIGH;   // trailer beat precedes next fetch
          end
          else
            state_ns = IDLE_S;
        end
        if(swap_pending_r && frontend_ready) begin
          exec_bank_ns          = load_bank;  // consume the staged bank
          loaded_ns[load_bank]  = `LOW;
          swap_pending_ns       = `LOW;
          // DON'T release fetch yet: exec_bank_r flips next cycle, and the
          // per-bank addr mux + 1-cycle IMEM read latency need to settle
          // on the new bank before fetch re-issues pc=0 — otherwise the
          // new program's word 0 is skipped (the swap-seam hazard the sim
          // TB caught: word1 dispatched first). Hold through a short
          // settle window; pc stays 0 (fetch stalled) meanwhile.
          swap_settle_ns        = 3'd3;
          fetch_restart         = 1'b1;   // build11: pc=0 at swap
        end
        else if(swap_settle_r != 3'd0) begin
          swap_settle_ns = swap_settle_r - 3'd1;
          if(swap_settle_r == 3'd1) fetch_hold_ns = `LOW;  // release, settled
        end
      end
    endcase

  end

  always @(posedge clk) begin
    if(rst || (|rst_ctr_r)) begin
      if(SIM_MEM == "false")
        state_r      <= IDLE_S;
      else
        state_r      <= EXECUTE_S;
      xfer_ctr_r   <= {`IMEM_ADDR_WIDTH{`LOW}};
      valid_out_sr <= {`IMEM_RD_LATENCY{`LOW}};
      addr_out_sr  <= 0;
      if(rst_ctr_r > 0)
        rst_ctr_r <= rst_ctr_r - 1;
      else
        rst_ctr_r <= 0;
      exec_bank_r    <= 1'b0;
      loaded_r       <= 2'b00;
      stream_en_r    <= `LOW;
      fetch_hold_r   <= `LOW;
      swap_pending_r <= `LOW;
      swap_settle_r  <= 3'd0;
      `ifdef HBM_BENDER
      hbm_temp_rd_r <= 0;
      `endif
    end
    else begin
      state_r      <= state_ns;
      xfer_ctr_r   <= xfer_ctr_ns;
      rst_ctr_r    <= rst_ctr_ns;
      exec_bank_r    <= exec_bank_ns;
      loaded_r       <= loaded_ns;
      stream_en_r    <= stream_en_ns;
      fetch_hold_r   <= fetch_hold_ns;
      swap_pending_r <= swap_pending_ns;
      swap_settle_r  <= swap_settle_ns;
      // compute when we should assert valid data to
      // fetch stage.
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
    end
  end

  // Debug taps (placed after all declarations to avoid forward-reference errors)
  assign dbg_state         = state_r;
  assign dbg_maint_req     = maint_req;
  assign dbg_maint_process = maint_process;
  assign dbg_exec_bank     = exec_bank_r;
  assign dbg_loaded        = loaded_r;
  assign dbg_swap_pending  = swap_pending_r;
  assign dbg_fetch_hold    = fetch_hold_r;

  assign rbe_accw_pl = h2c_tdata_0[3:0];
endmodule
