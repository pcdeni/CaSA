// Sim-only top wrapping softmc_pipeline + a behavioral instruction
// memory. The real frontend stitches this together with a Vivado-generated
// instr_blk_mem BRAM IP and an XDMA AXI-Stream front door; for measuring
// FPGA-side DDR command-bus utilization, none of that matters — we just
// need the pipeline to fetch from a memory we preload.
//
// IMEM read latency is fixed at 1 (matches `IMEM_RD_LATENCY = 1` in
// parameters.vh) so frontend's behaviour is mirrored exactly.

`include "parameters.vh"

module softmc_pipeline_top(
  input  clk,
  input  rst,

  // Synchronous IMEM init port. Drive these before raising `run`.
  input                          imem_init_we,
  input  [`IMEM_ADDR_WIDTH-1:0]  imem_init_addr,
  input  [`INSTR_WIDTH-1:0]      imem_init_data,

  // Free-running gate. Hold low during IMEM init, raise to start fetch.
  input                          run,

  // Observability — DDR command bus, 4 slots per fabric cycle (DDR4 PHY
  // clock is 4× the fabric clock, hence 4 commands per cycle).
  output [3:0]                ddr_write,
  output [3:0]                ddr_read,
  output [3:0]                ddr_pre,
  output [3:0]                ddr_act,
  output [3:0]                ddr_ref,
  output [3:0]                ddr_zq,
  output [3:0]                ddr_nop,
  output [3:0]                ddr_ap,
  output [3:0]                ddr_pall,
  output [3:0]                ddr_half_bl,
  output [4*`BG_WIDTH-1:0]    ddr_bg,
  output [4*`BANK_WIDTH-1:0]  ddr_bank,
  output [4*`COL_WIDTH-1:0]   ddr_col,
  output [4*`ROW_WIDTH-1:0]   ddr_row,
  output [511:0]              ddr_wdata,

  // Pipeline back-pressure (mostly informational for the harness).
  output                      softmc_end,
  output [11:0]               read_size,
  output                      read_seq_incoming
);

  // ------ Behavioral IMEM ----------------------------------------------------
  // A flat reg array; capacity = 1 << IMEM_ADDR_WIDTH (8192 entries after
  // the 11→13 bump). `addr_out` is the fetch's request address; `data_in`/
  // `addr_in`/`valid_in` are the IMEM's response, registered one cycle later.

  wire [`IMEM_ADDR_WIDTH-1:0] addr_out;
  wire                        valid_out;

  reg [`INSTR_WIDTH-1:0]      imem_storage [0:(1<<`IMEM_ADDR_WIDTH)-1];

  reg [`INSTR_WIDTH-1:0]      data_pipe;
  reg [`IMEM_ADDR_WIDTH-1:0]  addr_pipe;
  reg                         valid_pipe;

  always @(posedge clk) begin
    if (imem_init_we)
      imem_storage[imem_init_addr] <= imem_init_data;
  end

  always @(posedge clk) begin
    if (rst) begin
      valid_pipe <= 1'b0;
      data_pipe  <= {`INSTR_WIDTH{1'b0}};
      addr_pipe  <= {`IMEM_ADDR_WIDTH{1'b0}};
    end else begin
      valid_pipe <= valid_out & run;
      data_pipe  <= imem_storage[addr_out];
      addr_pipe  <= addr_out;
    end
  end

  // ------ Pipeline ----------------------------------------------------------
  // buffer_space stays large so the readback path never throttles fetch.

  softmc_pipeline pipeline (
    .clk(clk),
    .rst(rst),
    .softmc_end(softmc_end),
    .read_size(read_size),
    .read_seq_incoming(read_seq_incoming),
    .buffer_space(12'd2048),
    .addr_out(addr_out),
    .valid_out(valid_out),
    .data_in(data_pipe),
    .valid_in(valid_pipe),
    .addr_in(addr_pipe),
    .ready_out(run),
    .ddr_write(ddr_write),
    .ddr_read(ddr_read),
    .ddr_pre(ddr_pre),
    .ddr_act(ddr_act),
    .ddr_ref(ddr_ref),
    .ddr_sre(),
    .ddr_srx(),
    .ddr_zq(ddr_zq),
    .ddr_nop(ddr_nop),
    .ddr_ap(ddr_ap),
    .ddr_pall(ddr_pall),
    .ddr_half_bl(ddr_half_bl),
    .ddr_bg(ddr_bg),
    .ddr_bank(ddr_bank),
    .ddr_col(ddr_col),
    .ddr_row(ddr_row),
    .ddr_wdata(ddr_wdata)
  );

endmodule
