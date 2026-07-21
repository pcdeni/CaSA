// Behavioral stand-in for the BCU1525_QUAD `rdback_fifo` Xilinx
// fifo_generator IP (create_project.tcl: First_Word_Fall_Through,
// asymmetric 512b write / 256b read, Valid_Flag=true, Output_Depth 2048).
//
// FWFT show-ahead semantics: `dout` presents the head entry whenever the
// FIFO is non-empty and `valid` is high in exactly those cycles (this is
// what lets readback_engine drive c2h_tvalid_0 straight from `valid` and
// c2h_tdata_0 straight from `dout`). `rd_en` pops the presented entry.
// Each 512b write becomes two 256b read entries, low half first
// (din[255:0] then din[511:256]) — matching the IP behaviour the
// engine's half-swap on `din` was written against.
module rdback_fifo (
    input  wire         clk,
    input  wire         srst,
    input  wire [511:0] din,
    input  wire         wr_en,
    input  wire         rd_en,
    output wire [255:0] dout,
    output wire         full,
    output wire         empty,
    output wire         valid,
    output wire         prog_full
);
    localparam DEPTH = 4096; // 256b entries (= 2048 x 512b writes)
    reg [255:0] mem [0:DEPTH-1];
    reg [12:0]  wp;
    reg [12:0]  rp;

    wire [12:0] count = wp - rp;

    assign empty     = (count == 13'd0);
    assign full      = (count > DEPTH - 2); // cannot take another 512b write
    assign prog_full = (count > DEPTH - 16);
    // FWFT show-ahead
    assign valid = !empty;
    assign dout  = mem[rp[11:0]];

    always @(posedge clk) begin
        if (srst) begin
            wp <= 13'd0;
            rp <= 13'd0;
        end
        else begin
            if (wr_en && !full) begin
                mem[wp[11:0]]                 <= din[255:0];
                mem[(wp[11:0] + 12'd1) & 12'hfff] <= din[511:256];
                wp <= wp + 13'd2;
            end
            if (rd_en && !empty)
                rp <= rp + 13'd1;
        end
    end
endmodule
