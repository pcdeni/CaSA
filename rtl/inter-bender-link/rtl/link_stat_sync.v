// -----------------------------------------------------------------------------
// link_stat_sync — CDC-clean host readback of the link counters (CONTRACT §9a).
// L12b-explicit multi-bit synchroniser into the readback engine's ui_clk.
//
// CONTRACT (honest scope): valid under the QUIESCENT-READ discipline — the host
// reads link stats BETWEEN operations (the same diagnostic-read discipline the
// temp-read path uses), so the counter buses are STATIC at read time and a
// DEST_SYNC_FF chain is a correct quasi-static CDC of a stable value (exactly
// what xpm_cdc_array_single is contracted for). Source-domain identity is
// immaterial for a static bus (the integration XDC applies set_false_path on
// these registers).
//
// LIVE-READ (counters moving) is NOT covered here: it needs an upstream `freeze`
// in inter_bender_link that gates every counter (a per-domain freeze handshake).
// That is flagged as integration work (MANIFEST §7 item 2); `freeze` is carried
// as an advisory port now, not yet load-bearing. Do NOT rely on this for a
// live-traffic snapshot until the router `freeze` lands.
//
// Synthesis binds the real xpm_cdc_array_single; Verilator binds the sim model.
// -----------------------------------------------------------------------------
module link_stat_sync #(
  parameter integer N_CORES = 4,
  parameter integer CTRW    = 32
)(
  input  wire                              stat_clk,     // readback engine ui_clk
  input  wire                              freeze,       // hold high across the read
  // raw counters (each in its own write domain; stable while frozen)
  input  wire [N_CORES*N_CORES*CTRW-1:0]   in_beats,
  input  wire [N_CORES*N_CORES*CTRW-1:0]   in_frames,
  input  wire [N_CORES*N_CORES*CTRW-1:0]   in_drops,
  input  wire [N_CORES*CTRW-1:0]           in_injframes,
  input  wire [N_CORES*8-1:0]              in_status,
  // synchronized snapshot in stat_clk (valid a few cycles after freeze)
  output wire [N_CORES*N_CORES*CTRW-1:0]   sy_beats,
  output wire [N_CORES*N_CORES*CTRW-1:0]   sy_frames,
  output wire [N_CORES*N_CORES*CTRW-1:0]   sy_drops,
  output wire [N_CORES*CTRW-1:0]           sy_injframes,
  output wire [N_CORES*8-1:0]              sy_status
);
  // Per-bus-width instances keep the XPM contract explicit (one primitive per bus).
  localparam integer WB = N_CORES*N_CORES*CTRW;
  localparam integer WI = N_CORES*CTRW;
  localparam integer WS = N_CORES*8;

  // src_clk: for a frozen bus, any clock works for the input register; use
  // stat_clk-synchronous capture of the frozen value (SRC_INPUT_REG=1).
  xpm_cdc_array_single #(.DEST_SYNC_FF(4), .WIDTH(WB), .SRC_INPUT_REG(1))
    s_b (.src_clk(stat_clk), .src_in(in_beats),     .dest_clk(stat_clk), .dest_out(sy_beats));
  xpm_cdc_array_single #(.DEST_SYNC_FF(4), .WIDTH(WB), .SRC_INPUT_REG(1))
    s_f (.src_clk(stat_clk), .src_in(in_frames),    .dest_clk(stat_clk), .dest_out(sy_frames));
  xpm_cdc_array_single #(.DEST_SYNC_FF(4), .WIDTH(WB), .SRC_INPUT_REG(1))
    s_d (.src_clk(stat_clk), .src_in(in_drops),     .dest_clk(stat_clk), .dest_out(sy_drops));
  xpm_cdc_array_single #(.DEST_SYNC_FF(4), .WIDTH(WI), .SRC_INPUT_REG(1))
    s_i (.src_clk(stat_clk), .src_in(in_injframes), .dest_clk(stat_clk), .dest_out(sy_injframes));
  xpm_cdc_array_single #(.DEST_SYNC_FF(4), .WIDTH(WS), .SRC_INPUT_REG(1))
    s_s (.src_clk(stat_clk), .src_in(in_status),    .dest_clk(stat_clk), .dest_out(sy_status));

  // `freeze` is documented as gating the source counters upstream; unused here
  // beyond intent-marking (kept for the port contract).
  wire _unused_freeze = freeze;
endmodule
