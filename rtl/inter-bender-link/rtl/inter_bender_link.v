// =============================================================================
// inter_bender_link — any-bender-to-any-bender AXI-Stream routing stage (#76)
//
// Snoops each core's outbound c2h (read-only) and injects the copy into any
// peer's tagged link_rx ingress, one xpm_fifo_async per ordered (src,dst) pair.
// Default-inert: all routes disabled at power-up => the QUAD is the current star
// and BOTH host datapaths (h2c instructions, c2h readback/ACK) are bit-identical
// because this module NEVER drives a host-path signal. See CONTRACT.md.
//
// Ports are FLATTENED vectors (Verilog-2001 portable). Per-core UI clocks come
// in as a packed bus and are bit-selected as clock nets (Vivado/Verilator OK).
//
// L25 observability: per-route beats/frames/drops counters + a link status word
// per dest (fill / stall-cause), CDC stalls being the likely failure mode.
// =============================================================================
module inter_bender_link #(
  parameter integer N_CORES    = 4,
  parameter integer DW         = 256,               // XDMA_AXI_DATA_WIDTH
  parameter integer KW         = 32,                // DW/8
  parameter integer FIFO_DEPTH = 512,
  parameter integer CTRW       = 32,                // counter width
  parameter [7:0]   LINK_MAGIC = 8'h76,             // cfg-word guard (self-id, L17)
  parameter integer IDW        = 2                  // clog2(N_CORES) for N=4
)(
  // ---- per-core UI clocks / resets (active-high rst) ----
  input  wire [N_CORES-1:0]        core_ui_clk,
  input  wire [N_CORES-1:0]        core_ui_rst,

  // ---- SOURCE snoop taps: read-only observation of each core's c2h ----
  input  wire [N_CORES*DW-1:0]     c2h_tdata,
  input  wire [N_CORES*KW-1:0]     c2h_tkeep,
  input  wire [N_CORES-1:0]        c2h_tvalid,
  input  wire [N_CORES-1:0]        c2h_tready,       // driven by the existing cc; we only READ it
  input  wire [N_CORES-1:0]        c2h_tlast,

  // ---- route config write from each core's frontend_linked export ----
  input  wire [N_CORES-1:0]        cfg_stb,          // 1-cycle pulse in core_ui_clk[s]
  input  wire [N_CORES*16-1:0]     cfg_data,         // {..,magic[10:3],dst[2:1],en[0]}

  // ---- DEST inject ports: tagged link ingress, one per dest core ----
  output wire [N_CORES*DW-1:0]     linkrx_tdata,
  output wire [N_CORES*KW-1:0]     linkrx_tkeep,
  output wire [N_CORES-1:0]        linkrx_tvalid,
  input  wire [N_CORES-1:0]        linkrx_tready,
  output wire [N_CORES-1:0]        linkrx_tlast,
  output wire [N_CORES*IDW-1:0]    linkrx_tsrc,      // origin core id

  // ---- host-readable observability (natural domain; snapshotted for c2h) ----
  output wire [N_CORES*N_CORES*CTRW-1:0] stat_beats,   // [s*N+d]
  output wire [N_CORES*N_CORES*CTRW-1:0] stat_frames,  // [s*N+d] (tlast count)
  output wire [N_CORES*N_CORES*CTRW-1:0] stat_drops,   // [s*N+d] (tap FIFO-full)
  output wire [N_CORES*CTRW-1:0]         stat_injframes, // [d]
  output wire [N_CORES*N_CORES*10-1:0]   stat_fill,    // [s*N+d] write-side occupancy
  output wire [N_CORES*8-1:0]            stat_status   // [d] {cause[1:0],sel[IDW],dvalid,tready,locked}
);
  localparam integer WORDW = 1 + KW + DW;  // {tlast, tkeep, tdata} = 289 for defaults

  // ---- per-pair fabric wires (flat, index p = s*N + d) ----
  wire [N_CORES*N_CORES-1:0]        fifo_full;
  wire [N_CORES*N_CORES-1:0]        fifo_wren;
  wire [N_CORES*N_CORES-1:0]        fifo_rden;
  wire [N_CORES*N_CORES-1:0]        fifo_dvalid;
  wire [N_CORES*N_CORES*WORDW-1:0]  fifo_dout;
  wire [N_CORES*N_CORES*WORDW-1:0]  fifo_din;
  wire [N_CORES*N_CORES*10-1:0]     fifo_wrcnt;

  // ---- per-source combinational routing decision (driven by SRC blocks) ----
  wire [N_CORES-1:0]                s_committed;
  wire [N_CORES-1:0]                s_cur_en;
  wire [N_CORES*2-1:0]              s_cur_dst;

  genvar gs, gd;

  // =========================================================================
  // SOURCE side: route regs, frame-atomic dst latch, snoop push, counters.
  // Everything here is in core_ui_clk[gs].
  // =========================================================================
  generate
  for (gs = 0; gs < N_CORES; gs = gs + 1) begin : SRC
    wire clk = core_ui_clk[gs];
    wire rstb = core_ui_rst[gs];

    reg        r_en;                // ROUTE[gs].en
    reg [1:0]  r_dst;               // ROUTE[gs].dst
    reg        f_active;            // inside a frame currently being captured
    reg        r_en_lat;            // en held for the current frame
    reg [1:0]  r_dst_lat;           // dst held for the current frame

    wire committed = c2h_tvalid[gs] & c2h_tready[gs];
    wire beat_last = c2h_tlast[gs];
    // frame-atomic: outside a frame use the live ROUTE, inside use the latched copy
    wire       cur_en  = f_active ? r_en_lat  : r_en;
    wire [1:0] cur_dst = f_active ? r_dst_lat : r_dst;

    assign s_committed[gs]        = committed;
    assign s_cur_en[gs]           = cur_en;
    assign s_cur_dst[gs*2 +: 2]   = cur_dst;

    // cfg word fields (core gs writes ROUTE[gs])
    wire       cfg_en   = cfg_data[gs*16 + 0];
    wire [1:0] cfg_dst  = cfg_data[gs*16 + 1 +: 2];
    wire [7:0] cfg_mag  = cfg_data[gs*16 + 3 +: 8];

    // per-dest counters (domain gs)
    reg [CTRW-1:0] c_beats  [0:N_CORES-1];
    reg [CTRW-1:0] c_frames [0:N_CORES-1];
    reg [CTRW-1:0] c_drops  [0:N_CORES-1];
    integer i;

    // full flag for the current target dst
    wire cur_full = fifo_full[gs*N_CORES + cur_dst];

    always @(posedge clk) begin
      if (rstb) begin
        r_en <= 1'b0; r_dst <= 2'd0; f_active <= 1'b0;
        r_en_lat <= 1'b0; r_dst_lat <= 2'd0;
        for (i = 0; i < N_CORES; i = i + 1) begin
          c_beats[i] <= {CTRW{1'b0}}; c_frames[i] <= {CTRW{1'b0}}; c_drops[i] <= {CTRW{1'b0}};
        end
      end else begin
        // ---- host config write (guarded by magic; no-hardcode: dst is data) ----
        if (cfg_stb[gs] && (cfg_mag == LINK_MAGIC)) begin
          r_en  <= cfg_en & (cfg_dst != gs); // reject self-route by construction
          r_dst <= cfg_dst;
        end
        // ---- frame tracking (sample dst at frame start, hold to tlast) ----
        if (committed) begin
          if (!f_active) begin
            r_en_lat  <= r_en;
            r_dst_lat <= r_dst;
            f_active  <= ~beat_last;      // single-beat frame closes immediately
          end else if (beat_last) begin
            f_active  <= 1'b0;
          end
          // ---- snoop push / drop, counted ----
          if (cur_en && (cur_dst != gs)) begin
            if (!cur_full) begin
              c_beats[cur_dst]  <= c_beats[cur_dst] + 1'b1;
              if (beat_last) c_frames[cur_dst] <= c_frames[cur_dst] + 1'b1;
            end else begin
              c_drops[cur_dst]  <= c_drops[cur_dst] + 1'b1;   // NEVER stall c2h — drop+count
            end
          end
        end
      end
    end

    // per-pair write connections + counter export
    for (gd = 0; gd < N_CORES; gd = gd + 1) begin : WR
      if (gs != gd) begin : LIVE
        wire push_sel = committed & cur_en & (cur_dst == gd[1:0]);
        assign fifo_wren[gs*N_CORES+gd] = push_sel & ~fifo_full[gs*N_CORES+gd];
        assign fifo_din[(gs*N_CORES+gd)*WORDW +: WORDW] =
                 { c2h_tlast[gs], c2h_tkeep[gs*KW +: KW], c2h_tdata[gs*DW +: DW] };
      end else begin : SELF
        assign fifo_wren[gs*N_CORES+gd] = 1'b0;
        assign fifo_din[(gs*N_CORES+gd)*WORDW +: WORDW] = {WORDW{1'b0}};
      end
      assign stat_beats [(gs*N_CORES+gd)*CTRW +: CTRW] = c_beats [gd];
      assign stat_frames[(gs*N_CORES+gd)*CTRW +: CTRW] = c_frames[gd];
      assign stat_drops [(gs*N_CORES+gd)*CTRW +: CTRW] = c_drops [gd];
      assign stat_fill  [(gs*N_CORES+gd)*10   +: 10]   = fifo_wrcnt[(gs*N_CORES+gd)*10 +: 10];
    end
  end
  endgenerate

  // =========================================================================
  // The async FIFOs — one per ordered (src,dst) pair. Self pairs tied off.
  // =========================================================================
  generate
  for (gs = 0; gs < N_CORES; gs = gs + 1) begin : FIFO_S
    for (gd = 0; gd < N_CORES; gd = gd + 1) begin : FIFO_D
      if (gs != gd) begin : PAIR
        link_async_fifo #(.WIDTH(WORDW), .DEPTH(FIFO_DEPTH)) u (
          .rst     (core_ui_rst[gs] | core_ui_rst[gd]),
          .wr_clk  (core_ui_clk[gs]),
          .wr_en   (fifo_wren[gs*N_CORES+gd]),
          .din     (fifo_din[(gs*N_CORES+gd)*WORDW +: WORDW]),
          .full    (fifo_full[gs*N_CORES+gd]),
          .wr_count(fifo_wrcnt[(gs*N_CORES+gd)*10 +: 10]),
          .rd_clk  (core_ui_clk[gd]),
          .rd_en   (fifo_rden[gs*N_CORES+gd]),
          .dout    (fifo_dout[(gs*N_CORES+gd)*WORDW +: WORDW]),
          .valid   (fifo_dvalid[gs*N_CORES+gd])
        );
      end else begin : NOPAIR
        assign fifo_full[gs*N_CORES+gd]   = 1'b1;   // never a push target
        assign fifo_dvalid[gs*N_CORES+gd] = 1'b0;
        assign fifo_dout[(gs*N_CORES+gd)*WORDW +: WORDW] = {WORDW{1'b0}};
        assign fifo_wrcnt[(gs*N_CORES+gd)*10 +: 10]      = 10'd0;
      end
    end
  end
  endgenerate

  // =========================================================================
  // DEST side: per-dest round-robin arbiter, frame-atomic egress lock.
  // Everything here is in core_ui_clk[gd].
  // =========================================================================
  generate
  for (gd = 0; gd < N_CORES; gd = gd + 1) begin : DST
    wire clk = core_ui_clk[gd];
    wire rstb = core_ui_rst[gd];

    reg              locked;
    reg  [IDW-1:0]   sel;
    reg  [CTRW-1:0]  c_inj;

    // round-robin candidate scan starting just after `sel`
    reg  [IDW-1:0]   cand;
    reg              cand_v;
    integer k; integer st;
    always @* begin
      cand = sel; cand_v = 1'b0;
      for (k = 1; k <= N_CORES; k = k + 1) begin
        st = (sel + k) % N_CORES;
        if (!cand_v && (st != gd) && fifo_dvalid[st*N_CORES + gd]) begin
          cand = st[IDW-1:0]; cand_v = 1'b1;
        end
      end
    end

    wire         head_valid = fifo_dvalid[sel*N_CORES + gd];
    wire [WORDW-1:0] head    = fifo_dout[(sel*N_CORES+gd)*WORDW +: WORDW];
    wire         head_last   = head[WORDW-1];
    wire [KW-1:0] head_keep  = head[DW +: KW];
    wire [DW-1:0] head_data  = head[0 +: DW];

    wire tvalid = locked & head_valid;
    wire fire   = tvalid & linkrx_tready[gd];

    assign linkrx_tvalid[gd]              = tvalid;
    assign linkrx_tdata[gd*DW +: DW]      = head_data;
    assign linkrx_tkeep[gd*KW +: KW]      = head_keep;
    assign linkrx_tlast[gd]               = tvalid & head_last;
    assign linkrx_tsrc[gd*IDW +: IDW]     = sel;

    // rd_en only to the selected pair, only when consumer takes it
    for (gs = 0; gs < N_CORES; gs = gs + 1) begin : RD
      if (gs != gd) begin : L
        assign fifo_rden[gs*N_CORES+gd] = locked & (sel == gs) & fire;
      end else begin : S
        assign fifo_rden[gs*N_CORES+gd] = 1'b0;
      end
    end

    always @(posedge clk) begin
      if (rstb) begin
        locked <= 1'b0; sel <= {IDW{1'b0}}; c_inj <= {CTRW{1'b0}};
      end else begin
        if (!locked) begin
          if (cand_v) begin sel <= cand; locked <= 1'b1; end
        end else begin
          if (fire && head_last) begin
            locked <= 1'b0;                       // frame-atomic: unlock at tlast
            c_inj  <= c_inj + 1'b1;
          end
        end
      end
    end

    assign stat_injframes[gd*CTRW +: CTRW] = c_inj;

    // ---- link status (L25): stall cause is the headline (CDC stalls likely) ----
    // cause: 0 idle, 1 streaming, 2 consumer-stall (locked,dvalid,~tready),
    //        3 starved (locked, source FIFO empty mid-frame = CDC underrun)
    wire [1:0] cause = (!locked)                       ? 2'd0 :
                       (head_valid & ~linkrx_tready[gd])? 2'd2 :
                       (~head_valid)                    ? 2'd3 : 2'd1;
    assign stat_status[gd*8 +: 8] = { 1'b0,               // [7] reserved
                                      cause,               // [6:5]
                                      sel,                 // [4:3]
                                      head_valid,          // [2]
                                      linkrx_tready[gd],   // [1]
                                      locked };            // [0]
  end
  endgenerate

endmodule
