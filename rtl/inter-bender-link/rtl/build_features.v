// =============================================================================
// build_features — bundle provenance + per-feature runtime enables (L25).
// A dropin for the readback-engine author. Returns a self-identifying register
// via the SAME control-word -> c2h path the readback engine already uses for
// "read HBM temp" (readback_engine.v:148-158): a host control word raises a read
// strobe, the engine muxes `feat_word` into rdback_din and returns it on c2h.
//
//   FEATURES / MANIFEST_HASH / BUILD_MAGIC : BAKED at synthesis (provenance —
//     "is the fix in THIS image?" becomes a register read). Read-only.
//   ENABLE : host-set, default-inert (all NEW datapaths OFF -> proven fallback),
//     so a silicon A/B isolates a misbehaving ingredient WITHOUT reflashing.
//
// FEATURES / ENABLE bit map (shared with MANIFEST.md §1-§2, no-hardcode):
//   bit0 POPCOUNT4_FIX (0abccc0e)   bit1 MAJ5_DATAPATH
//   bit2 PACK4_SEQGEN               bit3 INTER_BENDER_LINK
// FEATURES bitN = ingredient present; ENABLE bitN = ingredient active
// (popcount4-fix has no runtime toggle -> ENABLE bit0 reserved=1).
// =============================================================================
module build_features #(
  // Defaults = the 4-ingredient bundle (MANIFEST.md §1); overridden per build at
  // the top-level instantiation, recomputed at bundle freeze.
  parameter [31:0] FEATURES      = 32'h0000_000F,   // bit0..3: popcnt-fix/MAJ5/pack4/link present
  parameter [31:0] MANIFEST_HASH = 32'hC176_89CF,   // CRC32 over bundled sources' sorted md5 list (excl. this file)
  parameter [31:0] BUILD_MAGIC   = 32'hDBC0DE76      // link/obs self-id (per-image, L17)
)(
  input  wire        clk,
  input  wire        rst,
  // host-set ENABLE register write (frontend control-word export; no config CDC:
  // same ui_clk as the readback engine that reads it)
  input  wire        en_wr,
  input  wire [31:0] en_wdata,
  // baked/host-readable snapshot to pack into the readback datapath (512b word)
  output wire [511:0] feat_word,
  // per-feature runtime enables consumed by the respective datapaths
  output wire        en_maj5,
  output wire        en_pack4
);
  reg [31:0] enable_r;
  always @(posedge clk)
    if (rst)        enable_r <= 32'h0000_0001; // bit0=1 (popcount fix always on), rest 0 = inert
    else if (en_wr) enable_r <= en_wdata | 32'h0000_0001;

  assign en_maj5  = enable_r[1];
  assign en_pack4 = enable_r[2];

  // self-identifying snapshot: [magic | features | hash | enables] in the low 128b
  assign feat_word = { {384{1'b0}}, enable_r, MANIFEST_HASH, FEATURES, BUILD_MAGIC };
endmodule
