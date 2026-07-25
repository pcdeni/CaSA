`include "parameters.vh"

module fetch_stage(
  // common signals
  input                          clk,
  input                          rst,

  // other control signals
  output                         softmc_end,
  output  [11:0]                 read_size,
  output reg                     read_seq_incoming,
  input  [11:0]                  buffer_space,

  // branch unit <-> fetch stage interface
  // build15-dbg: additive observation-only outputs for the URAM ILA.
  output [2:0]                   dbg_fetch_state,
  output [`IMEM_ADDR_WIDTH-1:0]  dbg_fetch_pc,
  output                         dbg_br_outstanding,
  // build17: origin in, origin out -- registered with the instruction
  input                          data_in_maint,
  output                         instr_maint,
  input                          br_resolve,
  input   [`IMEM_ADDR_WIDTH-1:0] br_target,

  // build11: program-start strobe — forces pc=0 regardless of the
  // re-loop freeze phase (the stale-pc loss window: fetch re-loops
  // pc=0..END between END and fin; a hold/state-exit parks pc at an
  // arbitrary phase and the next program resumed from it, losing its
  // first word(s) — latent in LEGACY too, masked by fixed-fin TBs).
  input                          restart,

  // fetch stage <-> frontend interface
  output  [`IMEM_ADDR_WIDTH-1:0] addr_out,
  output                         valid_out,
  input   [`INSTR_WIDTH-1:0]     data_in,
  input                          valid_in,
  input   [`IMEM_ADDR_WIDTH-1:0] addr_in,
  input                          ready_out, // frontend is ready for a valid request

  // fetch stage <-> decode stage interface
  output  [`INSTR_WIDTH-1:0]     instr,
  output  [`IMEM_ADDR_WIDTH-1:0] instr_pc,
  output                         instr_valid
  );

  wire inst_is_br, is_end, is_ddr_start, need_flush, is_sleep;

  reg [31:0] sleep_ctr_r, sleep_ctr_ns;

  localparam WAIT_RESOLVE_S      = 0;
  localparam FETCH_NEXT_LINE_S   = 1;
  localparam WAIT_BUFFER_SPACE_S = 2;
  localparam WAIT_SLEEP_S        = 3;
  // build12: post-END park — fetch stays here until the frontend's
  // restart strobe starts the next program. Kills the re-loop and with
  // it the whole stale-state hazard class (stale pc, stale branches,
  // post-END junk dispatches).
  localparam WAIT_RESTART_S      = 4;

  reg [2:0] state_r, state_ns;
  // build12: a br_resolve is only valid while a branch WE dispatched is
  // outstanding. restart clears it — a stale resolve from the previous
  // program's parked post-END re-loop must not pair with the next
  // program's first branch (build-11 E14 silicon regression).
  reg br_outstanding_r, br_outstanding_ns;

  // kind of confusing but these PCs map to
  // an instruction instead of to a byte.
  // i.e. each PC addresses an instruction.
  reg [`IMEM_ADDR_WIDTH-1:0] pc_r, pc_ns;

  // register outputs, decode will receive
  // stuff we've received with one cycle latency
  // i.e. marks the end of fetch_stage cycle
  reg [`IMEM_ADDR_WIDTH-1:0] instr_pc_r, instr_pc_ns;
  reg [`INSTR_WIDTH-1:0]     instr_r, instr_ns;
  reg                        instr_valid_r, instr_valid_ns;
  
  pre_decode pdec(
    .buffer_space(buffer_space),
    .read_size(read_size),
    .instruction(state_r == WAIT_BUFFER_SPACE_S ? instr_r : data_in),
    .is_branch(inst_is_br),
    .is_end(is_end),
    .is_ddr_start(is_ddr_start),
    .need_flush(need_flush),
    .is_sleep(is_sleep)
  );

  assign instr       = instr_r;
  assign instr_valid = instr_valid_r;
  assign instr_pc    = instr_pc_r;

  // request instr @ pc from frontend
  assign valid_out   = ready_out && (state_r == FETCH_NEXT_LINE_S) && ~need_flush && ~(valid_in && is_sleep);
  assign addr_out    = pc_r;

  assign softmc_end  = is_end && valid_in && (state_r == FETCH_NEXT_LINE_S);

  // build17: one register, in lockstep with instr_r
  reg instr_maint_r;
  always @(posedge clk) begin
    if(rst)          instr_maint_r <= 1'b0;
    else if(valid_in) instr_maint_r <= data_in_maint;
  end
  assign instr_maint = instr_maint_r;

  // build15-dbg: observation only
  assign dbg_fetch_state     = state_r;
  assign dbg_fetch_pc        = pc_r;
  assign dbg_br_outstanding  = br_outstanding_r;

  always @* begin
    sleep_ctr_ns   = sleep_ctr_r;
    state_ns       = state_r;
    br_outstanding_ns = br_outstanding_r;
    pc_ns          = pc_r;
    instr_ns       = instr_r;
    instr_pc_ns    = instr_pc_r;
    instr_valid_ns = ~is_end && valid_in && (state_r == FETCH_NEXT_LINE_S) 
        && ~is_ddr_start;
    read_seq_incoming = `LOW;
    case(state_r)
      WAIT_RESOLVE_S: begin
        if(br_resolve && br_outstanding_r) begin
          state_ns = FETCH_NEXT_LINE_S;
          pc_ns    = br_target;
          br_outstanding_ns = 1'b0;
        end
        // build12: a resolve with no outstanding branch is stale — drop.
      end
      FETCH_NEXT_LINE_S: begin
        if(ready_out && valid_out)
          pc_ns = pc_r + 1;
        if(valid_in) begin
          instr_ns = data_in;
          instr_pc_ns = addr_in;
          if(is_sleep) begin
            sleep_ctr_ns = data_in[31:0];
            state_ns = WAIT_SLEEP_S;
          end
          if(is_ddr_start && ~need_flush && (|data_in[9:0]))
            read_seq_incoming = `HIGH;
          if(inst_is_br && ~is_sleep) begin // we don't have the ability to perform well
            state_ns = WAIT_RESOLVE_S;
            br_outstanding_ns = 1'b1;   // build12: our branch is in flight
          end
          else if(is_end) begin
            pc_ns    = {`IMEM_ADDR_WIDTH{`LOW}};
            state_ns = WAIT_RESTART_S;   // build12: park, no re-loop
          end
          else if(need_flush) begin
            state_ns = WAIT_BUFFER_SPACE_S;
            pc_ns    = addr_in;     // register the info packet
                                    // as the next instruction to fetch
          end
        end  
      end
      WAIT_BUFFER_SPACE_S: begin
        if(~need_flush) begin
          state_ns = FETCH_NEXT_LINE_S;
        end
      end
      WAIT_SLEEP_S: begin
        sleep_ctr_ns = sleep_ctr_r - 1;
        if(sleep_ctr_r == 32'b1)
          state_ns = FETCH_NEXT_LINE_S;
      end
      WAIT_RESTART_S: begin
        ; // build12: only the restart override below exits this state
      end
    endcase
    // build11: program-start override — unconditional, wins over every
    // state. pc=0, clean fetch state, no stale dispatch.
    if(restart) begin
      pc_ns          = {`IMEM_ADDR_WIDTH{1'b0}};
      state_ns       = FETCH_NEXT_LINE_S;
      instr_valid_ns = 1'b0;
      br_outstanding_ns = 1'b0;   // build12: fence stale resolves
    end
  end

  always @(posedge clk) begin
    if (rst) begin
      pc_r          <= {`IMEM_ADDR_WIDTH{`LOW}};
      state_r       <= FETCH_NEXT_LINE_S;
      instr_valid_r <= `LOW;
      instr_r       <= {`INSTR_WIDTH{`LOW}};
      instr_pc_r    <= {`IMEM_ADDR_WIDTH{`LOW}};
      sleep_ctr_r   <= {32{`LOW}};
      br_outstanding_r <= `LOW;
    end
    else begin
      state_r       <= state_ns;
      pc_r          <= pc_ns;
      instr_r       <= instr_ns;
      instr_pc_r    <= instr_pc_ns;
      instr_valid_r <= instr_valid_ns;
      sleep_ctr_r   <= sleep_ctr_ns;
      br_outstanding_r <= br_outstanding_ns;
    end
  end

endmodule
