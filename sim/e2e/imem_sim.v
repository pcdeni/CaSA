`timescale 1ns/1ps
`include "parameters.vh"

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
