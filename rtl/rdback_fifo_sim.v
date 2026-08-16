// Behavioral stand-in for the BCU1525_QUAD `rdback_fifo` Xilinx
// fifo_generator IP (create_project.tcl: First_Word_Fall_Through,
// asymmetric 512b write / 256b read, Valid_Flag=true, Output_Depth 2048).
//
// FWFT show-ahead semantics: `dout` presents the head entry whenever the
// FIFO is non-empty and `valid` is high in exactly those cycles (this is
// what lets readback_engine drive c2h_tvalid_0 straight from `valid` and
// c2h_tdata_0 straight from `dout`). `rd_en` pops the presented entry.
// Each 512b write becomes two 256b read entries, MOST-SIGNIFICANT half
// first (din[511:256] then din[255:0]). That is the asymmetric-aspect-ratio
// behaviour of the IP, and it is what readback_engine's half-swap on `din`
// ("shuffle data because fifo outputs them on wrong order") is written
// against: the engine pre-swaps the two halves, so the pair leaves the FIFO
// in natural low-half-first order and the receiver reads it straight, with
// no de-swap. A receiver that de-swaps is compensating for a model, not for
// the hardware — and the compensation sits upstream of the point where the
// engine taps rd_data for the SEG_POP popcounts, so it cancels for READ and
// does not cancel for SEG_POP.
//
// `+define+SIM_RBF_LSB_FIRST` restores the legacy low-half-first model for
// byte-exact replay of older runs; the receiver model must be built with the
// matching `-CFLAGS -DSIM_RBF_LSB_FIRST` (see tb_readback.cpp).
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
    // Storage ceiling (keeps the 12-bit mem indexing in range) vs the ACTIVE
    // depth. The IP's Output_Depth is 2048 on the 256-bit read side = 65,536 B,
    // so that is the value the flags are derived from; a 4096-entry model is
    // twice the hardware and puts the real fill boundary out of reach of every
    // scenario. `+define+SIM_RBF_DEPTH4096` restores the legacy value.
    localparam DEPTH = 4096;
`ifdef SIM_RBF_DEPTH4096
    localparam ADEPTH = 4096;
`else
    localparam ADEPTH = 2048;
`endif
    reg [255:0] mem [0:DEPTH-1];
    reg [12:0]  wp;
    reg [12:0]  rp;

    wire [12:0] count = wp - rp;

    assign empty     = (count == 13'd0);
    assign full      = (count > ADEPTH - 2); // cannot take another 512b write
    // NOTE: the IP's Full_Threshold_Assert_Value is 895 of 1024 write-side
    // entries = 1790 of 2048 read entries, not ADEPTH-16. The mismatch is
    // inert: prog_full drives readback_engine's `fifo_almost_full`, which is
    // declared, connected — and never read, in this copy and in the
    // synthesized one alike.
    assign prog_full = (count > ADEPTH - 16);
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
`ifdef SIM_RBF_LSB_FIRST
                mem[wp[11:0]]                 <= din[255:0];
                mem[(wp[11:0] + 12'd1) & 12'hfff] <= din[511:256];
`else
                mem[wp[11:0]]                 <= din[511:256];
                mem[(wp[11:0] + 12'd1) & 12'hfff] <= din[255:0];
`endif
                wp <= wp + 13'd2;
            end
            if (rd_en && !empty)
                rp <= rp + 13'd1;
        end
    end
endmodule
