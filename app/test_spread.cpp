// Characterize the SiMRA doubleACT row-spread side-effect that caused
// the LOAD_WEIGHTS bug.  Bug summary: doubleACT(t_12=30, t_23=1, src,
// dst) deposits src's data into row src+8 within src's mat group as a
// charge-sharing artifact.
//
// This test answers, deterministically, on real silicon:
//   1. Is the spread only at +8 — or also at -8? +16? +24?
//   2. Is it a one-shot at +8, or a chain (+8 → +16 → +24)?
//   3. Does it happen for every doubleACT timing, or only the long-hold
//      RowClone (t_12=30)?  What about MAJ3 (0, 0) and Broadcast (10, 2)?
//   4. Does a plain single ACT-PRE (no second ACT) cause spread?
//
// Method: write distinct fingerprint patterns to a constellation of rows
// {R-32, R-24, R-16, R-8, R, R+8, R+16, R+24, R+32}.  Read all back to
// confirm baseline cleanliness.  Then fire ONE experimental program
// (single doubleACT or single ACT-PRE) referencing only R as source and
// a far-away DST row.  Read all constellation rows back and classify
// each as one of: matches-R (= got R's data), matches-original (=
// untouched), or other (= flake / partial corruption).
//
// Build: make spread-test-exe && ./spread-test-exe 0 calib_dimm2.txt 0
// (bender_id, calib_path, bank_id)

#include "platform.h"
#include "instruction.h"
#include "prog.h"
#include "../util.h"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <string>

// Center row R — overridable via PIM_TEST_R env var.
// BANK_ID overridable via PIM_BANK (default 0).
// DST_FAR overridable via PIM_DST_FAR (default 38000 for DIMM 0).
static int            BANK_ID = 0;
static uint32_t       R       = 38640;
static uint32_t       DST_FAR = 38000;

// Constellation of probe offsets relative to R. Cover every single-bit
// XOR distance up to bit 9 (±1, ±2, ±4, ±8, ±16, ±32, ±64, ±128, ±256,
// ±512) plus a few multi-bit combos. The wider net catches all 10
// possible single-bit decoder-coupling distances observed across DIMMs.
static const int OFFSETS[] = {
    -512, -256, -128, -64, -48, -32, -24, -16, -8, -4, -2, -1, 0,
    +1, +2, +4, +8, +16, +24, +32, +48, +64, +128, +256, +512};
static const int N_OFF     = sizeof(OFFSETS) / sizeof(OFFSETS[0]);

// Build a unique 2048-segment u32 fingerprint mask for an offset value.
// Fold `offset` into the LFSR seed so each row's content is distinct
// and order-independent.
static void make_mask(int offset, uint32_t* mask_out) {
  uint32_t s = (uint32_t)(0xC0FFEE00u ^ (uint32_t)(offset * 0x9E3779B9u));
  for (int i = 0; i < 2048; i++) {
    s = s * 1664525u + 1013904223u;
    mask_out[i] = s;
  }
}

// Per-col write a row in 3 chunks (mirrors test_bitnet_server.cpp's
// per_column_write_row {43,43,42} layout, fits the 2048-IMEM bitstream).
static void per_col_write(SoftMCPlatform& platform, uint32_t row,
                           const uint32_t* mask) {
  static const int N_COLS = 128;
  static const int CHUNKS[3] = {43, 43, 42};
  int col_start = 0;
  for (int chunk = 0; chunk < 3; chunk++) {
    Program p;
    p.add_inst(SMC_LI(8, CASR));
    p.add_inst(SMC_LI(BANK_ID, BAR));
    p.add_inst(SMC_LI(row, RAR));
    p.add_inst(SMC_LI(col_start * 8, CAR));
    p.add_below(PRE(BAR, 0, 0));
    p.add_below(ACT(BAR, 0, RAR, 0));
    int n_cols = CHUNKS[chunk];
    for (int k = 0; k < n_cols; k++) {
      const uint32_t* slots = mask + (col_start + k) * 16;
      for (int slot = 0; slot < 16; slot++) {
        p.add_inst(SMC_LI(slots[slot], PATTERN_REG));
        p.add_inst(SMC_LDWD(PATTERN_REG, slot));
      }
      p.add_below(WRITE(BAR, CAR, 1));
      p.add_inst(SMC_SLEEP(8));
    }
    p.add_inst(SMC_SLEEP(8));
    p.add_below(PRE(BAR, 0, 0));
    p.add_inst(SMC_SLEEP(4));
    p.add_inst(SMC_END());
    platform.execute(p);
    col_start += n_cols;
  }
}

// Read a row's 8192 bytes back into rb.
static void read_row(SoftMCPlatform& platform, uint32_t row,
                     uint8_t* rb_out, int label) {
  Program p;
  p.add_inst(SMC_LI(8, CASR));
  p.add_inst(SMC_LI(BANK_ID, BAR));
  p.add_below(PRE(BAR, 0, 0));
  p.add_below(rdRow_immediate_label(BAR, row, label));
  p.add_inst(all_nops());
  p.add_inst(all_nops());
  p.add_below(PRE(BAR, 0, 0));
  p.add_inst(all_nops());
  p.add_inst(SMC_END());
  platform.execute(p);
  platform.receiveData(rb_out, 8192);
}

// Calibrated open_rows + Rfirst/Rsecond. Default = DIMM 0 / s_id 77.
// Overridable via PIM_CALIB_LINE = "Rfirst Rsecond r0 r1 ... r15"
// (same format as fault-sweep / production calib file lines after the
// leading "s_id bank" fields are stripped).
static uint32_t OPEN_ROWS[16] = {
    38408, 38412, 38424, 38428, 38472, 38476, 38488, 38492,
    38920, 38924, 38936, 38940, 38984, 38988, 39000, 39004};
static uint32_t RFIRST  = 38424;
static uint32_t RSECOND = 38988;

static void apply_calib_env_overrides() {
  if (const char* bv = getenv("PIM_BANK")) BANK_ID = atoi(bv);
  if (const char* dv = getenv("PIM_DST_FAR")) DST_FAR = (uint32_t)atoi(dv);
  const char* line = getenv("PIM_CALIB_LINE");
  if (line && *line) {
    uint32_t parsed[18];
    int n = 0;
    while (*line && n < 18) {
      while (*line == ' ' || *line == '\t') line++;
      if (!*line) break;
      parsed[n++] = (uint32_t)atoi(line);
      while (*line && *line != ' ' && *line != '\t') line++;
    }
    if (n == 18) {
      RFIRST  = parsed[0];
      RSECOND = parsed[1];
      for (int i = 0; i < 16; i++) OPEN_ROWS[i] = parsed[2 + i];
      fprintf(stderr, "[calib] PIM_CALIB_LINE applied: Rfirst=%u Rsecond=%u\n",
              RFIRST, RSECOND);
    } else {
      fprintf(stderr, "[calib] PIM_CALIB_LINE has %d ints, expected 18 — ignoring\n", n);
    }
  }
  fprintf(stderr, "[calib] BANK_ID=%d DST_FAR=%u R=%u\n", BANK_ID, DST_FAR, R);
}

// Mirror of `emit_bank_combined_body` from test_bitnet_server.cpp.  ONE
// MM3D round body — the actual sequence that production runs and that
// mysteriously corrupts R+8.
static void emit_full_mm3d(Program& p, uint32_t backup_row,
                            uint32_t x_pattern, int label_base) {
  p.add_inst(SMC_LI(BANK_ID, BAR));
  p.add_inst(SMC_LI(128, NUM_COLS_REG));
  // 1. RowClone backup_row → Rfirst
  p.add_below(PRE(BAR, 0, 0));
  p.add_inst(SMC_SLEEP(6));
  p.add_below(doubleACT(30, 1, backup_row, RFIRST));
  p.add_inst(SMC_SLEEP(6));
  p.add_below(PRE(BAR, 0, 0));
  p.add_inst(SMC_SLEEP(6));
  // 2. Broadcast
  p.add_below(doubleACT(10, 2, RFIRST, RSECOND));
  p.add_inst(SMC_SLEEP(6));
  p.add_below(PRE(BAR, 0, 0));
  p.add_inst(SMC_SLEEP(6));
  // 3. Overwrite 11 non-weight slots
  static const int act_pos[5]  = {1, 4, 7, 10, 13};
  static const int zero_pos[5] = {2, 5, 8, 11, 14};
  p.add_below(wrRow_immediate_label(BAR, OPEN_ROWS[0], 0xFFFFFFFFu, label_base + 0));
  for (int i = 0; i < 5; i++)
    p.add_below(wrRow_immediate_label(BAR, OPEN_ROWS[act_pos[i]],  x_pattern, label_base + 1 + i));
  for (int i = 0; i < 5; i++)
    p.add_below(wrRow_immediate_label(BAR, OPEN_ROWS[zero_pos[i]], 0u,        label_base + 100 + i));
  p.add_inst(SMC_SLEEP(6));
  p.add_below(PRE(BAR, 0, 0));
  p.add_inst(SMC_SLEEP(6));
  // 4. Frac × 3 on OPEN_ROWS[0] (using bare ACT-PRE, no helper).
  for (int j = 0; j < 3; j++) {
    p.add_inst(SMC_SLEEP(6));
    p.add_inst(SMC_LI(OPEN_ROWS[0], RAR));
    p.add_below(ACT(BAR, 0, RAR, 0));
    p.add_below(PRE(BAR, 0, 0));
    p.add_inst(SMC_SLEEP(6));
  }
  p.add_inst(SMC_SLEEP(6));
  p.add_below(PRE(BAR, 0, 0));
  p.add_inst(SMC_SLEEP(6));
  // 5. MAJ3
  p.add_below(doubleACT(0, 0, RFIRST, RSECOND));
  p.add_inst(SMC_SLEEP(6));
  p.add_below(PRE(BAR, 0, 0));
  p.add_inst(SMC_SLEEP(6));
  // 6. Read result
  p.add_below(rdRow_immediate_label(BAR, OPEN_ROWS[0], label_base + 999));
  p.add_inst(all_nops());
  p.add_inst(all_nops());
  p.add_below(PRE(BAR, 0, 0));
  p.add_inst(all_nops());
  p.add_inst(all_nops());
}

// Run a full MM3D body (RowClone+Broadcast+wrRows+frac+MAJ3+rdRow)
// using R as backup_row and Rfirst/Rsecond/open_rows from calibration.
// Drains the rdRow data so receiveData buffer doesn't overflow on next
// program.
static void fire_full_mm3d(SoftMCPlatform& platform, uint32_t backup_row) {
  Program p;
  p.add_inst(SMC_LI(8, CASR));
  emit_full_mm3d(p, backup_row, 0xCAFEBABEu, /*label_base=*/3000);
  p.add_inst(SMC_END());
  platform.execute(p);
  std::vector<uint8_t> drain(8192);
  platform.receiveData(drain.data(), 8192);
}

// Build and execute a one-shot doubleACT(t_12, t_23, R, DST_FAR).
static void fire_doubleACT(SoftMCPlatform& platform, int t_12, int t_23,
                            uint32_t src, uint32_t dst) {
  Program p;
  p.add_inst(SMC_LI(8, CASR));
  p.add_inst(SMC_LI(BANK_ID, BAR));
  p.add_inst(SMC_LI(128, NUM_COLS_REG));
  p.add_below(PRE(BAR, 0, 0));
  p.add_inst(SMC_SLEEP(6));
  p.add_below(doubleACT(t_12, t_23, src, dst));
  p.add_inst(SMC_SLEEP(6));
  p.add_below(PRE(BAR, 0, 0));
  p.add_inst(SMC_SLEEP(6));
  p.add_inst(SMC_END());
  platform.execute(p);
}

// Build and execute a single ACT-PRE on `row` with `t_open` cycles open.
static void fire_single_act(SoftMCPlatform& platform, uint32_t row, int t_open) {
  Program p;
  p.add_inst(SMC_LI(8, CASR));
  p.add_inst(SMC_LI(BANK_ID, BAR));
  p.add_inst(SMC_LI(row, RAR));
  p.add_below(PRE(BAR, 0, 0));
  p.add_inst(SMC_SLEEP(6));
  p.add_below(ACT(BAR, 0, RAR, 0));
  if (t_open > 0) p.add_inst(SMC_SLEEP(t_open));
  p.add_below(PRE(BAR, 0, 0));
  p.add_inst(SMC_SLEEP(6));
  p.add_inst(SMC_END());
  platform.execute(p);
}

// Compare buf to mask (both 8192 bytes / 2048 u32). Returns segs that
// match exactly.
static int n_match(const uint8_t* buf, const uint32_t* mask) {
  const uint32_t* bu32 = (const uint32_t*)buf;
  int n = 0;
  for (int i = 0; i < 2048; i++) if (bu32[i] == mask[i]) n++;
  return n;
}

int main(int argc, char** argv) {
  int bender_id = (argc > 1) ? atoi(argv[1]) : 0;
  if (const char* rv = getenv("PIM_TEST_R")) R = (uint32_t)atoi(rv);
  apply_calib_env_overrides();
  fprintf(stderr, "Using R = %u  bender_id = %d\n", R, bender_id);
  SoftMCPlatform platform(bender_id);
  if (platform.init() != SOFTMC_SUCCESS) {
    fprintf(stderr, "platform init failed\n");
    return 1;
  }
  fprintf(stderr, "platform init OK; resetting FPGA\n");
  platform.reset_fpga();

  // Pre-compute fingerprint masks for every offset.
  std::vector<std::vector<uint32_t>> masks(N_OFF, std::vector<uint32_t>(2048));
  for (int i = 0; i < N_OFF; i++) make_mask(OFFSETS[i], masks[i].data());

  // Index of the offset == 0 entry (= R itself).
  int R_idx = -1;
  for (int i = 0; i < N_OFF; i++) if (OFFSETS[i] == 0) { R_idx = i; break; }

  // 'is_double' field reused as a 3-way mode selector via -1 = full MM3D.
  // Cleaner: introduce a separate enum, but TimingCase is private here
  // and "-1 = full" is fine for a one-off characterization tool.
  enum { MODE_FULL_MM3D = -1, MODE_SINGLE_ACT = 0, MODE_DOUBLE_FAR = 1,
         MODE_DOUBLE_RFIRST = 2, MODE_DOUBLE_RFIRST_SRC = 3,
         MODE_DOUBLE_BOTH_TUPLE = 4 };  // src=Rfirst, dst=Rsecond — MAJ3 case
  struct ExtCase { const char* name; int mode; int t_12; int t_23; };
  ExtCase cases[] = {
      {"single-ACT (t_open=6)",                 MODE_SINGLE_ACT,    6, 0},
      {"single-ACT (t_open=30)",                MODE_SINGLE_ACT,   30, 0},
      {"doubleACT(0, 0)  → DST_FAR",            MODE_DOUBLE_FAR,    0, 0},
      {"doubleACT(10, 2) → DST_FAR",            MODE_DOUBLE_FAR,   10, 2},
      {"doubleACT(30, 1) → DST_FAR",            MODE_DOUBLE_FAR,   30, 1},
      {"doubleACT(30, 1) → Rfirst (calib tup)", MODE_DOUBLE_RFIRST,30, 1},
      {"doubleACT(10, 2) → Rfirst (calib tup)", MODE_DOUBLE_RFIRST,10, 2},
      {"doubleACT(0, 0)  → Rfirst (calib tup)", MODE_DOUBLE_RFIRST, 0, 0},
      // Reverse direction: SRC = Rfirst (calib), DST = R.  Probes whether
      // Rfirst+8 gets Rfirst's data (i.e. spread off the SOURCE side when
      // source is in a tuple).
      {"doubleACT(30, 1) Rfirst → R",            MODE_DOUBLE_RFIRST_SRC,30, 1},
      {"doubleACT(10, 2) Rfirst → R",            MODE_DOUBLE_RFIRST_SRC,10, 2},
      // BOTH src and dst are tuple members — exactly the MAJ3 case. The
      // probe constellation is AROUND R (set R = Rfirst via PIM_TEST_R
      // before running this case for the spread off Rfirst to be visible).
      {"doubleACT(30, 1) Rfirst→Rsecond [MAJ3-like]", MODE_DOUBLE_BOTH_TUPLE, 30, 1},
      {"doubleACT(10, 2) Rfirst→Rsecond [MAJ3-like]", MODE_DOUBLE_BOTH_TUPLE, 10, 2},
      {"doubleACT(0, 0)  Rfirst→Rsecond [MAJ3-like]", MODE_DOUBLE_BOTH_TUPLE,  0, 0},
      {"FULL MM3D BODY (production sequence)",  MODE_FULL_MM3D,     0, 0},
  };

  for (size_t c = 0; c < sizeof(cases)/sizeof(cases[0]); c++) {
    const auto& tc = cases[c];
    printf("\n=========================================\n");
    printf("CASE: %s\n", tc.name);
    printf("=========================================\n");

    // Reset FPGA between cases so prior-case state doesn't leak.
    platform.reset_fpga();

    // Re-write all constellation rows with their unique fingerprints.
    fprintf(stderr, "[%s] writing constellation rows\n", tc.name);
    for (int i = 0; i < N_OFF; i++) {
      uint32_t row = (uint32_t)((int)R + OFFSETS[i]);
      per_col_write(platform, row, masks[i].data());
    }

    // Baseline read-back: confirm each row contains its own mask (no
    // residual spread from the writes themselves).
    int baseline_clean = 0;
    for (int i = 0; i < N_OFF; i++) {
      uint32_t row = (uint32_t)((int)R + OFFSETS[i]);
      std::vector<uint8_t> rb(8192);
      read_row(platform, row, rb.data(), 100000 + i);
      int match = n_match(rb.data(), masks[i].data());
      if (match >= 2046) baseline_clean++;
      else printf("  baseline FAIL: offset=%+d row=%u match=%d/2048 (writes themselves spread!)\n",
                  OFFSETS[i], row, match);
    }
    printf("baseline: %d/%d rows match own fingerprint\n", baseline_clean, N_OFF);

    // FIRE the experimental program.
    switch (tc.mode) {
      case MODE_SINGLE_ACT:
        fire_single_act(platform, R, tc.t_12);
        break;
      case MODE_DOUBLE_FAR:
        fire_doubleACT(platform, tc.t_12, tc.t_23, R, DST_FAR);
        break;
      case MODE_DOUBLE_RFIRST:
        fire_doubleACT(platform, tc.t_12, tc.t_23, R, RFIRST);
        break;
      case MODE_DOUBLE_RFIRST_SRC:
        fire_doubleACT(platform, tc.t_12, tc.t_23, RFIRST, R);
        break;
      case MODE_FULL_MM3D:
        fire_full_mm3d(platform, R);
        break;
      case MODE_DOUBLE_BOTH_TUPLE:
        // src = Rfirst, dst = Rsecond — both are tuple members. This is
        // exactly the doubleACT pattern the MAJ3 step in MajOps fires.
        fire_doubleACT(platform, tc.t_12, tc.t_23, RFIRST, RSECOND);
        break;
    }

    // For reverse-direction (SRC=Rfirst) cases, probe rows around
    // Rfirst (38424) too — that's where any source-side spread would
    // land. Write known fingerprints to Rfirst±{1,8,16}, fire, read.
    if (tc.mode == MODE_DOUBLE_RFIRST_SRC) {
      uint32_t Rf_neighbors[] = {RFIRST-16, RFIRST-8, RFIRST-1, RFIRST,
                                  RFIRST+1, RFIRST+8, RFIRST+16};
      uint32_t rf_mask[2048];
      // Build a UNIQUE Rfirst-data mask so we can tell when it spreads.
      uint32_t s = 0xDEADBEEFu;
      for (int i = 0; i < 2048; i++) { s = s * 1664525u + 1013904223u; rf_mask[i] = s; }
      // Re-write Rfirst with this distinctive mask AND its neighbors.
      // We use a different LFSR seed per neighbor to keep them distinct.
      for (size_t k = 0; k < sizeof(Rf_neighbors)/sizeof(*Rf_neighbors); k++) {
        uint32_t row = Rf_neighbors[k];
        uint32_t neighbor_mask[2048];
        uint32_t s2 = 0xBEEF0000u ^ (uint32_t)(row * 0x9E3779B9u);
        for (int i = 0; i < 2048; i++) { s2 = s2 * 1664525u + 1013904223u; neighbor_mask[i] = s2; }
        if (row == RFIRST) memcpy(neighbor_mask, rf_mask, sizeof(rf_mask));
        per_col_write(platform, row, neighbor_mask);
      }
      // Re-fire the experiment (since the neighbor writes happened
      // AFTER the original fire).
      fire_doubleACT(platform, tc.t_12, tc.t_23, RFIRST, R);
      // Read back Rfirst neighbors and report if they got Rfirst's data.
      printf("Rfirst-neighbors after fire (Rfirst data spread test):\n");
      for (size_t k = 0; k < sizeof(Rf_neighbors)/sizeof(*Rf_neighbors); k++) {
        uint32_t row = Rf_neighbors[k];
        std::vector<uint8_t> rb(8192);
        read_row(platform, row, rb.data(), 300000 + (int)k);
        int rf_m = n_match(rb.data(), rf_mask);
        printf("  Rfirst%+5d  row=%5u   Rfirst-match=%4d/2048\n",
               (int)row - (int)RFIRST, row, rf_m);
      }
    }

    // Read every constellation row back and classify.
    printf("after fire:\n");
    printf("  offset  row     own-match  R-match  classification\n");
    for (int i = 0; i < N_OFF; i++) {
      uint32_t row = (uint32_t)((int)R + OFFSETS[i]);
      std::vector<uint8_t> rb(8192);
      read_row(platform, row, rb.data(), 200000 + (int)c * 1000 + i);
      int own_m = n_match(rb.data(), masks[i].data());
      int  R_m  = n_match(rb.data(), masks[R_idx].data());
      const char* tag;
      if (i == R_idx)                           tag = "(source row)";
      else if (R_m >= 2046)                     tag = "MATCHES R   ← spread reached here";
      else if (own_m >= 2046)                   tag = "untouched";
      else if (R_m  > own_m && R_m > 1024)      tag = "partial-R   ← spread, partial";
      else                                       tag = "garbage / flake";
      printf("  %+5d  %5u   %4d/2048    %4d/2048   %s\n",
             OFFSETS[i], row, own_m, R_m, tag);
    }
  }

  return 0;
}
