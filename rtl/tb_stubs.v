`include "parameters.vh"
// Behavioral stubs for the build9 frontend streaming TB. The real
// instr_blk_mem is a Xilinx BRAM IP; here a simple synchronous-read RAM
// with the same port names (addra/clka/dina/douta/ena/wea). Depth =
// 2^IMEM_ADDR_WIDTH. Two instances (imem0/imem1) are created by the
// frontend under test; this one behavioral model serves both.
module instr_blk_mem(
  input                          clka,
  input                          ena,
  input                          wea,
  input  [`IMEM_ADDR_WIDTH-1:0]  addra,
  input  [`INSTR_WIDTH-1:0]      dina,
  output reg [`INSTR_WIDTH-1:0]  douta
);
  reg [`INSTR_WIDTH-1:0] mem [0:(1<<`IMEM_ADDR_WIDTH)-1];
  always @(posedge clka) begin
    if (ena) begin
      if (wea) mem[addra] <= dina;
      douta <= mem[addra];   // read-first is fine: fetch never reads a slot the same cycle it writes
    end
  end
endmodule

module instr_blk_mem_sim(
  input                          clka,
  input                          ena,
  input                          wea,
  input  [`IMEM_ADDR_WIDTH-1:0]  addra,
  input  [`INSTR_WIDTH-1:0]      dina,
  output reg [`INSTR_WIDTH-1:0]  douta
);
  reg [`INSTR_WIDTH-1:0] mem [0:(1<<`IMEM_ADDR_WIDTH)-1];
  always @(posedge clka) begin
    if (ena) begin
      if (wea) mem[addra] <= dina;
      douta <= mem[addra];
    end
  end
endmodule

// Maintenance controller: never requests (the TB drives programs only,
// no aref/zq/ref maintenance interleaving — that path is unchanged from
// build8 and out of scope for the streaming seam).
module maintenance_controller(
  input                          clk,
  input                          rst,
  input                          init_calib_complete,
  input                          softmc_fin,
  input                          aref_en,
  input                          aref_en_valid,
  output                         maint_req,
  input                          maint_ack,
  output                         per_rd_init,
  output                         per_zq_init,
  output                         per_ref_init,
  output                         maint_process,
  input                          program_process,
  input  [`IMEM_ADDR_WIDTH-1:0]  in_addr,
  input                          in_valid,
  output [`INSTR_WIDTH-1:0]      out_data,
  output                         out_valid,
  output [`IMEM_ADDR_WIDTH-1:0]  out_addr
);
  assign maint_req     = 1'b0;
  assign per_rd_init   = 1'b0;
  assign per_zq_init   = 1'b0;
  assign per_ref_init  = 1'b0;
  assign maint_process = 1'b0;
  assign out_data      = {`INSTR_WIDTH{1'b0}};
  assign out_valid     = 1'b0;
  assign out_addr      = {`IMEM_ADDR_WIDTH{1'b0}};
endmodule
