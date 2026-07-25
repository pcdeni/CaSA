`timescale 1ns/1ps
// e2e_sim behavioral maintenance microcode ROMs, loaded with the REAL
// QUAD project microcode (converted from projects/BCU1525_QUAD/coe/ —
// pr_read 13, pr_zq 40, pr_ref 49 words). Ports match the Xilinx IP.
module pr_read_mem(
  input clka, input ena, input wea,
  input [3:0] addra, input [63:0] dina,
  output reg [63:0] douta
);
  reg [63:0] mem [0:15];
  integer i; initial begin
    for(i=0;i<16;i=i+1) mem[i]=64'd0;
    $readmemh("pr_read_rom.hex", mem);
  end
  always @(posedge clka) if (ena) douta <= mem[addra];
endmodule

module zq_calib_mem(
  input clka, input ena, input wea,
  input [5:0] addra, input [63:0] dina,
  output reg [63:0] douta
);
  reg [63:0] mem [0:63];
  integer i; initial begin
    for(i=0;i<64;i=i+1) mem[i]=64'd0;
    $readmemh("pr_zq_rom.hex", mem);
  end
  always @(posedge clka) if (ena) douta <= mem[addra];
endmodule

module pr_ref_mem(
  input clka, input ena, input wea,
  input [6:0] addra, input [63:0] dina,
  output reg [63:0] douta
);
  reg [63:0] mem [0:127];
  integer i; initial begin
    for(i=0;i<128;i=i+1) mem[i]=64'd0;
    $readmemh("pr_ref_rom.hex", mem);
  end
  always @(posedge clka) if (ena) douta <= mem[addra];
endmodule
