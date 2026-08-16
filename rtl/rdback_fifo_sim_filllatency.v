// FWFT rdback_fifo with a realistic Xilinx fifo_generator FALL-THROUGH /
// FILL latency (First_Word_Fall_Through). The original behavioral stub made
// `empty`/`valid`/`dout` update the SAME cycle as the write — zero latency —
// which hid the readback_engine's trailer race: on silicon the real IP needs
// a few cycles after wr_en before `empty` deasserts and the word falls
// through to `dout`. Modeling that latency reproduces the trailer-before-
// chunk delivery offset seen on the tower (RESULT.md addendum 20e).
//
// Read order and depth follow `rdback_fifo_sim.v`: the IP presents the
// MOST-SIGNIFICANT half of each 512b write first, and the active depth is the
// IP's Output_Depth of 2048 read-side entries. `+define+SIM_RBF_LSB_FIRST` and
// `+define+SIM_RBF_DEPTH4096` restore the legacy model in both files together.
module rdback_fifo #(parameter FILL_LAT = 3) (
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
    localparam DEPTH = 4096;   // storage ceiling (12-bit mem indexing)
`ifdef SIM_RBF_DEPTH4096
    localparam ADEPTH = 4096;
`else
    localparam ADEPTH = 2048;  // IP Output_Depth: 2048 x 32 B = 65,536 B
`endif
    reg [255:0] mem [0:DEPTH-1];
    reg [12:0]  wp;
    reg [12:0]  rp;
    // The write pointer becomes VISIBLE to empty/valid/dout only FILL_LAT
    // cycles after the write (fall-through latency). Reads are immediate.
    reg [12:0]  wp_vis [0:FILL_LAT-1];
    integer k;

    wire [12:0] count_true = wp - rp;             // for full/prog_full (space)
    wire [12:0] count_vis  = wp_vis[FILL_LAT-1] - rp; // for empty/valid (data)

    assign empty     = (count_vis == 13'd0);
    assign full      = (count_true > ADEPTH - 2);
    assign prog_full = (count_true > ADEPTH - 16);
    assign valid     = !empty;
    assign dout      = mem[rp[11:0]];

    always @(posedge clk) begin
        if (srst) begin
            wp <= 13'd0;
            rp <= 13'd0;
            for (k = 0; k < FILL_LAT; k = k + 1) wp_vis[k] <= 13'd0;
        end
        else begin
            if (wr_en && !full) begin
`ifdef SIM_RBF_LSB_FIRST
                mem[wp[11:0]]                    <= din[255:0];
                mem[(wp[11:0] + 12'd1) & 12'hfff] <= din[511:256];
`else
                mem[wp[11:0]]                    <= din[511:256];
                mem[(wp[11:0] + 12'd1) & 12'hfff] <= din[255:0];
`endif
                wp <= wp + 13'd2;
            end
            // shift the visibility pipeline: the current wp is seen by the
            // empty/valid logic FILL_LAT cycles later.
            wp_vis[0] <= wp;
            for (k = 1; k < FILL_LAT; k = k + 1) wp_vis[k] <= wp_vis[k-1];
            if (rd_en && !empty)
                rp <= rp + 13'd1;
        end
    end
endmodule
