`include "parameters.vh"
`include "project.vh"

// BCU1525 Phase-2 top: 1× PCIe endpoint driving 4 independent DRAM-Bender
// cores, one per DIMM socket. Each core owns its own DDR4 MIG PHY and runs
// in its own UI-clock domain; a per-channel AXI-stream clock converter pair
// bridges it back to the single XDMA axi_clk.
//
// bcu1525_quad_top_linked = COPY of bcu1525_quad_top.v with the #76 inter-bender
// link added (L24/L25). Delta vs the star top:
//   + softmc_core -> softmc_core_linked (exports link route-config)
//   + inter_bender_link instance: SNOOPS each core_c2h_* (read-only) and INJECTS
//     to a per-core link_rx port; ONE xpm_fifo_async per (src,dst).
//   + per-core link_sink (L17 consumer until the fabric-attention block exists).
// The router drives NO host-path signal, so with all routes disabled (power-up
// default) this is bit-identical to the star. See CONTRACT.md.
module bcu1525_quad_top_linked #(parameter tCK = 1500, SIM = "false",
                                 parameter N_CORES = 4) (
  // ---- DIMM0..3 reference clocks (300 MHz differential)
  input  c0_sys_clk_p, input c0_sys_clk_n,
  input  c1_sys_clk_p, input c1_sys_clk_n,
  input  c2_sys_clk_p, input c2_sys_clk_n,
  input  c3_sys_clk_p, input c3_sys_clk_n,

  // ---- DIMM0 DDR4 pins
  output c0_ddr4_act_n, output [`ROW_ADDR_WIDTH-1:0] c0_ddr4_adr,
  output [1:0] c0_ddr4_ba,  output [1:0] c0_ddr4_bg,
  output [0:0] c0_ddr4_cke, output [0:0] c0_ddr4_odt, output [0:0] c0_ddr4_cs_n,
  output [0:0] c0_ddr4_ck_t, output [0:0] c0_ddr4_ck_c, output c0_ddr4_reset_n,
  inout  [7:0] c0_ddr4_dqs_c, inout [7:0] c0_ddr4_dqs_t,
  inout  [63:0] c0_ddr4_dq,   inout [7:0] c0_ddr4_dm_dbi_n,

  // ---- DIMM1
  output c1_ddr4_act_n, output [`ROW_ADDR_WIDTH-1:0] c1_ddr4_adr,
  output [1:0] c1_ddr4_ba,  output [1:0] c1_ddr4_bg,
  output [0:0] c1_ddr4_cke, output [0:0] c1_ddr4_odt, output [0:0] c1_ddr4_cs_n,
  output [0:0] c1_ddr4_ck_t, output [0:0] c1_ddr4_ck_c, output c1_ddr4_reset_n,
  inout  [7:0] c1_ddr4_dqs_c, inout [7:0] c1_ddr4_dqs_t,
  inout  [63:0] c1_ddr4_dq,   inout [7:0] c1_ddr4_dm_dbi_n,

  // ---- DIMM2
  output c2_ddr4_act_n, output [`ROW_ADDR_WIDTH-1:0] c2_ddr4_adr,
  output [1:0] c2_ddr4_ba,  output [1:0] c2_ddr4_bg,
  output [0:0] c2_ddr4_cke, output [0:0] c2_ddr4_odt, output [0:0] c2_ddr4_cs_n,
  output [0:0] c2_ddr4_ck_t, output [0:0] c2_ddr4_ck_c, output c2_ddr4_reset_n,
  inout  [7:0] c2_ddr4_dqs_c, inout [7:0] c2_ddr4_dqs_t,
  inout  [63:0] c2_ddr4_dq,   inout [7:0] c2_ddr4_dm_dbi_n,

  // ---- DIMM3
  output c3_ddr4_act_n, output [`ROW_ADDR_WIDTH-1:0] c3_ddr4_adr,
  output [1:0] c3_ddr4_ba,  output [1:0] c3_ddr4_bg,
  output [0:0] c3_ddr4_cke, output [0:0] c3_ddr4_odt, output [0:0] c3_ddr4_cs_n,
  output [0:0] c3_ddr4_ck_t, output [0:0] c3_ddr4_ck_c, output c3_ddr4_reset_n,
  inout  [7:0] c3_ddr4_dqs_c, inout [7:0] c3_ddr4_dqs_t,
  inout  [63:0] c3_ddr4_dq,   inout [7:0] c3_ddr4_dm_dbi_n,

  // ---- PCIe Gen3 x8 endpoint
  input        clk_ref_p, input clk_ref_n, input pcie_rst,
  output [7:0] pci_exp_txp, output [7:0] pci_exp_txn,
  input  [7:0] pci_exp_rxp, input  [7:0] pci_exp_rxn
);

  // -------------------------------------------------------------------------
  // PCIe refclk buffer
  // -------------------------------------------------------------------------
  wire sys_clk, sys_clk_gt;
  IBUFDS_GTE4 refclk_ibuf (
    .O   (sys_clk_gt),
    .ODIV2(sys_clk),
    .I   (clk_ref_p),
    .CEB (1'b0),
    .IB  (clk_ref_n)
  );

  // -------------------------------------------------------------------------
  // XDMA (4 H2C + 4 C2H AXI-stream channels) — axi_clk domain
  // -------------------------------------------------------------------------
  wire axi_clk, axi_rst_n;

  // Per-channel h2c/c2h (XDMA side, axi_clk domain)
  wire [`XDMA_AXI_DATA_WIDTH-1:0]   h2c_tdata   [3:0];
  wire                              h2c_tlast   [3:0];
  wire                              h2c_tvalid  [3:0];
  wire                              h2c_tready  [3:0];
  wire [`XDMA_AXI_DATA_WIDTH/8-1:0] h2c_tkeep   [3:0];

  wire [`XDMA_AXI_DATA_WIDTH-1:0]   c2h_tdata   [3:0];
  wire                              c2h_tlast   [3:0];
  wire                              c2h_tvalid  [3:0];
  wire                              c2h_tready  [3:0];
  wire [`XDMA_AXI_DATA_WIDTH/8-1:0] c2h_tkeep   [3:0];

  xdma xdma_i (
    .sys_rst_n          (pcie_rst),
    .sys_clk            (sys_clk),
    .sys_clk_gt         (sys_clk_gt),

    .pci_exp_txn        (pci_exp_txn),
    .pci_exp_txp        (pci_exp_txp),
    .pci_exp_rxn        (pci_exp_rxn),
    .pci_exp_rxp        (pci_exp_rxp),

    .m_axis_h2c_tdata_0 (h2c_tdata[0]),  .m_axis_h2c_tlast_0 (h2c_tlast[0]),
    .m_axis_h2c_tvalid_0(h2c_tvalid[0]), .m_axis_h2c_tready_0(h2c_tready[0]),
    .m_axis_h2c_tkeep_0 (h2c_tkeep[0]),
    .s_axis_c2h_tdata_0 (c2h_tdata[0]),  .s_axis_c2h_tlast_0 (c2h_tlast[0]),
    .s_axis_c2h_tvalid_0(c2h_tvalid[0]), .s_axis_c2h_tready_0(c2h_tready[0]),
    .s_axis_c2h_tkeep_0 (c2h_tkeep[0]),

    .m_axis_h2c_tdata_1 (h2c_tdata[1]),  .m_axis_h2c_tlast_1 (h2c_tlast[1]),
    .m_axis_h2c_tvalid_1(h2c_tvalid[1]), .m_axis_h2c_tready_1(h2c_tready[1]),
    .m_axis_h2c_tkeep_1 (h2c_tkeep[1]),
    .s_axis_c2h_tdata_1 (c2h_tdata[1]),  .s_axis_c2h_tlast_1 (c2h_tlast[1]),
    .s_axis_c2h_tvalid_1(c2h_tvalid[1]), .s_axis_c2h_tready_1(c2h_tready[1]),
    .s_axis_c2h_tkeep_1 (c2h_tkeep[1]),

    .m_axis_h2c_tdata_2 (h2c_tdata[2]),  .m_axis_h2c_tlast_2 (h2c_tlast[2]),
    .m_axis_h2c_tvalid_2(h2c_tvalid[2]), .m_axis_h2c_tready_2(h2c_tready[2]),
    .m_axis_h2c_tkeep_2 (h2c_tkeep[2]),
    .s_axis_c2h_tdata_2 (c2h_tdata[2]),  .s_axis_c2h_tlast_2 (c2h_tlast[2]),
    .s_axis_c2h_tvalid_2(c2h_tvalid[2]), .s_axis_c2h_tready_2(c2h_tready[2]),
    .s_axis_c2h_tkeep_2 (c2h_tkeep[2]),

    .m_axis_h2c_tdata_3 (h2c_tdata[3]),  .m_axis_h2c_tlast_3 (h2c_tlast[3]),
    .m_axis_h2c_tvalid_3(h2c_tvalid[3]), .m_axis_h2c_tready_3(h2c_tready[3]),
    .m_axis_h2c_tkeep_3 (h2c_tkeep[3]),
    .s_axis_c2h_tdata_3 (c2h_tdata[3]),  .s_axis_c2h_tlast_3 (c2h_tlast[3]),
    .s_axis_c2h_tvalid_3(c2h_tvalid[3]), .s_axis_c2h_tready_3(c2h_tready[3]),
    .s_axis_c2h_tkeep_3 (c2h_tkeep[3]),

    .usr_irq_req       (1'b0),
    .usr_irq_ack       (),
    .msi_enable        (),
    .msi_vector_width  (),

    .cfg_mgmt_addr            (19'b0),
    .cfg_mgmt_write           (1'b0),
    .cfg_mgmt_write_data      (32'b0),
    .cfg_mgmt_byte_enable     (4'b0),
    .cfg_mgmt_read            (1'b0),
    .cfg_mgmt_read_data       (),
    .cfg_mgmt_read_write_done (),

    .axi_aclk          (axi_clk),
    .axi_aresetn       (axi_rst_n),
    .user_lnk_up       ()
  );

  // -------------------------------------------------------------------------
  // Per-DIMM: softmc_core + 2× axis_clock_converter
  // -------------------------------------------------------------------------
  // Post-converter signals (DDR4 UI clock domain) — one set per DIMM
  wire [`XDMA_AXI_DATA_WIDTH-1:0]   core_h2c_tdata   [3:0];
  wire                              core_h2c_tlast   [3:0];
  wire                              core_h2c_tvalid  [3:0];
  wire                              core_h2c_tready  [3:0];
  wire [`XDMA_AXI_DATA_WIDTH/8-1:0] core_h2c_tkeep   [3:0];
  wire [`XDMA_AXI_DATA_WIDTH-1:0]   core_c2h_tdata   [3:0];
  wire                              core_c2h_tlast   [3:0];
  wire                              core_c2h_tvalid  [3:0];
  wire                              core_c2h_tready  [3:0];
  wire [`XDMA_AXI_DATA_WIDTH/8-1:0] core_c2h_tkeep   [3:0];
  wire                              core_ui_clk      [3:0];
  wire                              core_ui_rst      [3:0];
  wire                              core_calib_done  [3:0];

  // ---- #76 link: per-core route-config export + inject (link_rx) ports ----
  wire                              link_cfg_stb     [3:0];
  wire [15:0]                       link_cfg_data    [3:0];
  wire [`XDMA_AXI_DATA_WIDTH-1:0]   linkrx_tdata     [3:0];
  wire [`XDMA_AXI_DATA_WIDTH/8-1:0] linkrx_tkeep     [3:0];
  wire                              linkrx_tvalid    [3:0];
  wire                              linkrx_tready    [3:0];
  wire                              linkrx_tlast     [3:0];
  wire [1:0]                        linkrx_tsrc      [3:0];

`define DIMM_INST(IDX, SYSCLK_P, SYSCLK_N, DDR4_PFX) \
  axis_clock_converter h2c_cc_``IDX`` ( \
    .s_axis_tvalid (h2c_tvalid[IDX]),  .s_axis_tlast (h2c_tlast[IDX]), \
    .s_axis_tdata  (h2c_tdata[IDX]),   .s_axis_tkeep (h2c_tkeep[IDX]), \
    .s_axis_tready (h2c_tready[IDX]), \
    .m_axis_tvalid (core_h2c_tvalid[IDX]), .m_axis_tlast (core_h2c_tlast[IDX]), \
    .m_axis_tdata  (core_h2c_tdata[IDX]),  .m_axis_tkeep (core_h2c_tkeep[IDX]), \
    .m_axis_tready (core_h2c_tready[IDX]), \
    .s_axis_aresetn(axi_rst_n),  .s_axis_aclk(axi_clk), \
    .m_axis_aresetn(~core_ui_rst[IDX]), .m_axis_aclk(core_ui_clk[IDX]) \
  ); \
  axis_clock_converter c2h_cc_``IDX`` ( \
    .s_axis_tvalid (core_c2h_tvalid[IDX]), .s_axis_tlast (core_c2h_tlast[IDX]), \
    .s_axis_tdata  (core_c2h_tdata[IDX]),  .s_axis_tkeep (core_c2h_tkeep[IDX]), \
    .s_axis_tready (core_c2h_tready[IDX]), \
    .m_axis_tvalid (c2h_tvalid[IDX]), .m_axis_tlast (c2h_tlast[IDX]), \
    .m_axis_tdata  (c2h_tdata[IDX]),  .m_axis_tkeep (c2h_tkeep[IDX]), \
    .m_axis_tready (c2h_tready[IDX]), \
    .s_axis_aresetn(~core_ui_rst[IDX]), .s_axis_aclk(core_ui_clk[IDX]), \
    .m_axis_aresetn(axi_rst_n), .m_axis_aclk(axi_clk) \
  ); \
  wire [`CKE_WIDTH-1:0] cke_int_``IDX``; \
  wire [`ODT_WIDTH-1:0] odt_int_``IDX``; \
  wire [`CS_WIDTH-1:0]  cs_n_int_``IDX``; \
  wire                  parity_int_``IDX``; \
  assign DDR4_PFX``_cke[0]  = cke_int_``IDX``[0]; \
  assign DDR4_PFX``_odt[0]  = odt_int_``IDX``[0]; \
  assign DDR4_PFX``_cs_n[0] = cs_n_int_``IDX``[0]; \
  softmc_core_linked #(.tCK(tCK), .SIM(SIM)) softmc_``IDX`` ( \
    .link_cfg_stb  (link_cfg_stb[IDX]),  .link_cfg_data (link_cfg_data[IDX]), \
    .c0_sys_clk_p (SYSCLK_P), .c0_sys_clk_n (SYSCLK_N), \
    .sys_rst_l    (1'b1), \
    .c0_ddr4_act_n   (DDR4_PFX``_act_n), .c0_ddr4_adr (DDR4_PFX``_adr), \
    .c0_ddr4_ba      (DDR4_PFX``_ba),    .c0_ddr4_bg  (DDR4_PFX``_bg), \
    .c0_ddr4_cke     (cke_int_``IDX``),  .c0_ddr4_odt (odt_int_``IDX``), \
    .c0_ddr4_cs_n    (cs_n_int_``IDX``), \
    .c0_ddr4_ck_t    (DDR4_PFX``_ck_t),  .c0_ddr4_ck_c(DDR4_PFX``_ck_c), \
    .c0_ddr4_reset_n (DDR4_PFX``_reset_n), \
    .c0_ddr4_dqs_c   (DDR4_PFX``_dqs_c), .c0_ddr4_dqs_t(DDR4_PFX``_dqs_t), \
    .c0_ddr4_dq      (DDR4_PFX``_dq),    .c0_ddr4_dm_dbi_n(DDR4_PFX``_dm_dbi_n), \
    .c0_ddr4_parity  (parity_int_``IDX``), \
    .core_ui_clk               (core_ui_clk[IDX]), \
    .core_ui_rst               (core_ui_rst[IDX]), \
    .core_init_calib_complete  (core_calib_done[IDX]), \
    .m_axis_h2c_tdata_0     (core_h2c_tdata[IDX]),  .m_axis_h2c_tlast_0 (core_h2c_tlast[IDX]), \
    .m_axis_h2c_tvalid_0    (core_h2c_tvalid[IDX]), .m_axis_h2c_tready_0(core_h2c_tready[IDX]), \
    .m_axis_h2c_tkeep_0     (core_h2c_tkeep[IDX]), \
    .s_axis_c2h_tdata_0     (core_c2h_tdata[IDX]),  .s_axis_c2h_tlast_0 (core_c2h_tlast[IDX]), \
    .s_axis_c2h_tvalid_0    (core_c2h_tvalid[IDX]), .s_axis_c2h_tready_0(core_c2h_tready[IDX]), \
    .s_axis_c2h_tkeep_0     (core_c2h_tkeep[IDX]) \
  )

  `DIMM_INST(0, c0_sys_clk_p, c0_sys_clk_n, c0_ddr4);
  `DIMM_INST(1, c1_sys_clk_p, c1_sys_clk_n, c1_ddr4);
  `DIMM_INST(2, c2_sys_clk_p, c2_sys_clk_n, c2_ddr4);
  `DIMM_INST(3, c3_sys_clk_p, c3_sys_clk_n, c3_ddr4);

  // =========================================================================
  // #76 INTER-BENDER LINK  (default-inert; router drives no host-path signal)
  // Tap  = core_c2h_* (source ui_clk, snoop only). Inject = link_rx (dest ui_clk).
  // =========================================================================
  // pack the per-core unpacked wires into the router's flattened buses
  wire [N_CORES-1:0]                   l_ui_clk  = {core_ui_clk[3],  core_ui_clk[2],  core_ui_clk[1],  core_ui_clk[0]};
  wire [N_CORES-1:0]                   l_ui_rst  = {core_ui_rst[3],  core_ui_rst[2],  core_ui_rst[1],  core_ui_rst[0]};
  wire [N_CORES*`XDMA_AXI_DATA_WIDTH-1:0]   l_c2h_tdata =
        {core_c2h_tdata[3], core_c2h_tdata[2], core_c2h_tdata[1], core_c2h_tdata[0]};
  wire [N_CORES*(`XDMA_AXI_DATA_WIDTH/8)-1:0] l_c2h_tkeep =
        {core_c2h_tkeep[3], core_c2h_tkeep[2], core_c2h_tkeep[1], core_c2h_tkeep[0]};
  wire [N_CORES-1:0] l_c2h_tvalid = {core_c2h_tvalid[3], core_c2h_tvalid[2], core_c2h_tvalid[1], core_c2h_tvalid[0]};
  wire [N_CORES-1:0] l_c2h_tready = {core_c2h_tready[3], core_c2h_tready[2], core_c2h_tready[1], core_c2h_tready[0]};
  wire [N_CORES-1:0] l_c2h_tlast  = {core_c2h_tlast[3],  core_c2h_tlast[2],  core_c2h_tlast[1],  core_c2h_tlast[0]};
  wire [N_CORES-1:0] l_cfg_stb    = {link_cfg_stb[3],    link_cfg_stb[2],    link_cfg_stb[1],    link_cfg_stb[0]};
  wire [N_CORES*16-1:0] l_cfg_data = {link_cfg_data[3],  link_cfg_data[2],   link_cfg_data[1],   link_cfg_data[0]};

  wire [N_CORES*`XDMA_AXI_DATA_WIDTH-1:0]     l_rx_tdata;
  wire [N_CORES*(`XDMA_AXI_DATA_WIDTH/8)-1:0] l_rx_tkeep;
  wire [N_CORES-1:0]                          l_rx_tvalid;
  wire [N_CORES-1:0]                          l_rx_tready = {linkrx_tready[3], linkrx_tready[2], linkrx_tready[1], linkrx_tready[0]};
  wire [N_CORES-1:0]                          l_rx_tlast;
  wire [N_CORES*2-1:0]                        l_rx_tsrc;

  // observability buses (feed the stat-return path specified in MANIFEST.md §3)
  wire [N_CORES*N_CORES*32-1:0] l_stat_beats, l_stat_frames, l_stat_drops;
  wire [N_CORES*32-1:0]         l_stat_injframes;
  wire [N_CORES*N_CORES*10-1:0] l_stat_fill;
  wire [N_CORES*8-1:0]          l_stat_status;

  inter_bender_link #(.N_CORES(N_CORES),
                      .DW(`XDMA_AXI_DATA_WIDTH), .KW(`XDMA_AXI_DATA_WIDTH/8),
                      .FIFO_DEPTH(512), .CTRW(32), .LINK_MAGIC(8'h76)) u_link (
    .core_ui_clk (l_ui_clk),   .core_ui_rst (l_ui_rst),
    .c2h_tdata   (l_c2h_tdata), .c2h_tkeep (l_c2h_tkeep),
    .c2h_tvalid  (l_c2h_tvalid),.c2h_tready(l_c2h_tready), .c2h_tlast(l_c2h_tlast),
    .cfg_stb     (l_cfg_stb),   .cfg_data  (l_cfg_data),
    .linkrx_tdata(l_rx_tdata),  .linkrx_tkeep(l_rx_tkeep),
    .linkrx_tvalid(l_rx_tvalid),.linkrx_tready(l_rx_tready), .linkrx_tlast(l_rx_tlast),
    .linkrx_tsrc (l_rx_tsrc),
    .stat_beats  (l_stat_beats),.stat_frames(l_stat_frames), .stat_drops(l_stat_drops),
    .stat_injframes(l_stat_injframes), .stat_fill(l_stat_fill), .stat_status(l_stat_status)
  );

  // unpack inject bus + per-core sink (L17 consumer until fabric attention lands)
  genvar gi;
  generate for (gi = 0; gi < N_CORES; gi = gi + 1) begin : LINK_RX
    assign linkrx_tdata[gi]  = l_rx_tdata [gi*`XDMA_AXI_DATA_WIDTH +: `XDMA_AXI_DATA_WIDTH];
    assign linkrx_tkeep[gi]  = l_rx_tkeep [gi*(`XDMA_AXI_DATA_WIDTH/8) +: (`XDMA_AXI_DATA_WIDTH/8)];
    assign linkrx_tvalid[gi] = l_rx_tvalid[gi];
    assign linkrx_tlast[gi]  = l_rx_tlast [gi];
    assign linkrx_tsrc[gi]   = l_rx_tsrc  [gi*2 +: 2];
    wire [31:0] sink_beats, sink_frames; wire [`XDMA_AXI_DATA_WIDTH-1:0] sink_last;
    link_sink #(.DW(`XDMA_AXI_DATA_WIDTH), .KW(`XDMA_AXI_DATA_WIDTH/8)) u_sink (
      .clk(core_ui_clk[gi]), .rst(core_ui_rst[gi]),
      .rx_tdata(linkrx_tdata[gi]), .rx_tkeep(linkrx_tkeep[gi]),
      .rx_tvalid(linkrx_tvalid[gi]), .rx_tready(linkrx_tready[gi]),
      .rx_tlast(linkrx_tlast[gi]), .rx_tsrc(linkrx_tsrc[gi]),
      .sink_beats(sink_beats), .sink_frames(sink_frames), .sink_last_word(sink_last)
    );
  end endgenerate

endmodule
