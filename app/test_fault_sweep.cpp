// Full per-row fault characterization sweep for the SiMRA pool zone on
// DIMM 0 / s_id 77 calibration.
//
// For each candidate source row R in the safe zone, fire a SiMRA MM3D
// body that uses R as backup_row, then read every other row in a wide
// neighborhood and record which ones got R's data.
//
// Critical: tests run in MULTI-ROUND CONTEXT — N "warmup" MM3D bodies
// fire on neighboring rows BEFORE the test row, because the bit-5 /
// XOR-32 fault is state-dependent (doesn't manifest in single-MM3D
// isolation but appears after multi-round cumulative subarray bias).
//
// Output (stdout, machine-parseable):
//   FAULT R=<src> -> T=<target> (offset=<delta>, xor=0x<diff>)
// One line per fault. Plus summary stats at the end.
//
// Build: make fault-sweep-exe

#include "platform.h"
#include "instruction.h"
#include "prog.h"
#include "../util.h"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <set>

static int            BANK_ID = 0;  // overridable via argv[4]

// Calibrated open_rows + Rfirst + Rsecond.  Default: the live trio's bank-0
// tuple (calibration/calib_dimm2.txt, s_id 72, Rfirst=45340).  Override at
// runtime via the PIM_CALIB_LINE env (one tuple-line in calib file format)
// or the individual PIM_RFIRST / PIM_RSECOND / PIM_OPEN_ROWS env vars.
static uint32_t OPEN_ROWS[16] = {
    45340, 45341, 45342, 45343, 45436, 45437, 45438, 45439,
    45724, 45725, 45726, 45727, 45820, 45821, 45822, 45823};
static uint32_t RFIRST  = 45340;
static uint32_t RSECOND = 45823;
// The 640-row window containing the default tuple. Recomputed from the calib
// line's open rows when PIM_CALIB_LINE is given, and overridable outright
// with PIM_SUB_START / PIM_SUB_END.
static uint32_t SUB_START = 45312;
static uint32_t SUB_END   = 45952;

// Apply env override: PIM_CALIB_LINE="<Rfirst> <Rsecond> <r0> <r1> ... <r15>"
// (16 open_rows after Rfirst/Rsecond).  Resets SUB_START / SUB_END to the
// 640-row sub-array containing the first open_row.
static void apply_calib_env() {
  const char* line = getenv("PIM_CALIB_LINE");
  if (!line || !*line) return;
  uint32_t parsed[18];
  int n = 0;
  while (*line && n < 18) {
    while (*line == ' ' || *line == '\t') line++;
    if (!*line) break;
    parsed[n++] = (uint32_t)atoi(line);
    while (*line && *line != ' ' && *line != '\t') line++;
  }
  if (n < 18) {
    fprintf(stderr, "[calib] PIM_CALIB_LINE has %d ints, expected 18 "
            "(Rfirst Rsecond + 16 open_rows). Ignoring.\n", n);
    return;
  }
  RFIRST  = parsed[0];
  RSECOND = parsed[1];
  for (int i = 0; i < 16; i++) OPEN_ROWS[i] = parsed[2 + i];
  // Default subarray chunk: 640 rows starting at floor(lowest_open/640)*640.
  // Derived from the calib line, so it follows whichever die is being swept.
  // For DIMMs whose subarrays don't align to 640-row boundaries, override with
  // PIM_SUB_START / PIM_SUB_END to point at the actual s_id range from
  // FindOpenRows/dimm<N>/selected_subarrays.txt.
  uint32_t any_open = OPEN_ROWS[0];
  SUB_START = (any_open / 640) * 640;
  SUB_END   = SUB_START + 640;
  const char* ss = getenv("PIM_SUB_START");
  const char* se = getenv("PIM_SUB_END");
  if (ss && *ss && se && *se) {
    SUB_START = (uint32_t)atoi(ss);
    SUB_END   = (uint32_t)atoi(se);
  }
  fprintf(stderr, "[calib] applied: Rfirst=%u Rsecond=%u sub=[%u, %u)%s\n",
          RFIRST, RSECOND, SUB_START, SUB_END,
          (ss && *ss && se && *se) ? " (PIM_SUB_*)" : "");
}

static bool is_open_row(uint32_t r) {
  for (auto x : OPEN_ROWS) if (x == r) return true;
  return false;
}

// ----- helpers (mirror of test_spread.cpp's primitives) ---------------

static void make_mask(uint32_t seed, uint32_t* mask_out) {
  uint32_t s = seed;
  for (int i = 0; i < 2048; i++) {
    s = s * 1664525u + 1013904223u;
    mask_out[i] = s;
  }
}

static void per_col_write(SoftMCPlatform& platform, uint32_t row,
                           const uint32_t* mask) {
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

// Mirror of emit_bank_combined_body in test_bitnet_server.cpp — simplified
// inline, drains the rdRow data so receive buffer doesn't clog.
static void fire_full_mm3d(SoftMCPlatform& platform, uint32_t backup_row,
                            int label_base) {
  Program p;
  p.add_inst(SMC_LI(8, CASR));
  p.add_inst(SMC_LI(BANK_ID, BAR));
  p.add_inst(SMC_LI(128, NUM_COLS_REG));
  // RowClone
  p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_SLEEP(6));
  p.add_below(doubleACT(30, 1, backup_row, RFIRST));
  p.add_inst(SMC_SLEEP(6));
  p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_SLEEP(6));
  // Broadcast
  p.add_below(doubleACT(10, 2, RFIRST, RSECOND));
  p.add_inst(SMC_SLEEP(6));
  p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_SLEEP(6));
  // wrRow setup (5 act_pos + 5 zero_pos + ONE)
  static const int act_pos[5]  = {1, 4, 7, 10, 13};
  static const int zero_pos[5] = {2, 5, 8, 11, 14};
  p.add_below(wrRow_immediate_label(BAR, OPEN_ROWS[0], 0xFFFFFFFFu, label_base + 0));
  for (int i = 0; i < 5; i++)
    p.add_below(wrRow_immediate_label(BAR, OPEN_ROWS[act_pos[i]], 0xCAFEBABEu, label_base + 1 + i));
  for (int i = 0; i < 5; i++)
    p.add_below(wrRow_immediate_label(BAR, OPEN_ROWS[zero_pos[i]], 0u, label_base + 100 + i));
  p.add_inst(SMC_SLEEP(6));
  p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_SLEEP(6));
  // Frac × 3 on OPEN_ROWS[0]
  for (int j = 0; j < 3; j++) {
    p.add_inst(SMC_SLEEP(6));
    p.add_inst(SMC_LI(OPEN_ROWS[0], RAR));
    p.add_below(ACT(BAR, 0, RAR, 0));
    p.add_below(PRE(BAR, 0, 0));
    p.add_inst(SMC_SLEEP(6));
  }
  p.add_inst(SMC_SLEEP(6));
  p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_SLEEP(6));
  // MAJ3
  p.add_below(doubleACT(0, 0, RFIRST, RSECOND));
  p.add_inst(SMC_SLEEP(6));
  p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_SLEEP(6));
  // rdRow drain
  p.add_below(rdRow_immediate_label(BAR, OPEN_ROWS[0], label_base + 999));
  p.add_inst(all_nops()); p.add_inst(all_nops());
  p.add_below(PRE(BAR, 0, 0));
  p.add_inst(all_nops()); p.add_inst(all_nops());
  p.add_inst(SMC_END());
  platform.execute(p);
  std::vector<uint8_t> drain(8192);
  platform.receiveData(drain.data(), 8192);
}

static int n_match(const uint8_t* buf, const uint32_t* mask) {
  const uint32_t* bu32 = (const uint32_t*)buf;
  int n = 0;
  for (int i = 0; i < 2048; i++) if (bu32[i] == mask[i]) n++;
  return n;
}

// ----- Main sweep ----------------------------------------------------

int main(int argc, char** argv) {
  int bender_id = (argc > 1) ? atoi(argv[1]) : 0;
  int n_warmup  = (argc > 2) ? atoi(argv[2]) : 4;
  int src_step  = (argc > 3) ? atoi(argv[3]) : 1;
  if (argc > 4) BANK_ID = atoi(argv[4]);
  apply_calib_env();
  fprintf(stderr, "Bank: %d  Warmup: %d  Step: %d  Rfirst: %u\n",
          BANK_ID, n_warmup, src_step, RFIRST);

  SoftMCPlatform platform(bender_id);
  if (platform.init() != SOFTMC_SUCCESS) { fprintf(stderr, "init failed\n"); return 1; }
  platform.reset_fpga();

  // Build candidate source row list.
  std::vector<uint32_t> sources;
  for (uint32_t r = SUB_START; r < SUB_END; r += (uint32_t)src_step) {
    if (is_open_row(r)) continue;
    sources.push_back(r);
  }
  fprintf(stderr, "Sources: %zu rows  (warmup=%d)\n", sources.size(), n_warmup);
  fprintf(stderr, "Probe set per source: ALL safe-zone rows (~640)\n");

  // Pre-compute fingerprint masks per row in the safe zone (one mask per
  // row, derived deterministically from the row index).
  std::vector<std::vector<uint32_t>> row_mask;
  std::vector<uint32_t>              row_addr;
  row_mask.reserve(SUB_END - SUB_START);
  for (uint32_t r = SUB_START; r < SUB_END; r++) {
    if (is_open_row(r)) continue;
    row_addr.push_back(r);
    row_mask.emplace_back(2048);
    make_mask((uint32_t)(0xC0FFEE00u ^ (uint32_t)r * 0x9E3779B9u),
              row_mask.back().data());
  }
  fprintf(stderr, "Probe rows: %zu\n", row_addr.size());

  // ONE-TIME setup: write every probe row's fingerprint. This takes the
  // longest (~640 × ~3 ms = ~2 s).
  //
  // PIM_SCREEN_RW=1 (2026-07-18, per-subarray pool screening): also do a
  // pattern + antipattern write/read screen per row — write mask, read,
  // verify; write ~mask, read, verify; re-write mask (so the sweep's
  // fingerprint state is exactly as without the screen). Machine-parseable
  // RWBAD lines flag rows with any mismatching word. Default off.
  bool screen_rw = getenv("PIM_SCREEN_RW") && atoi(getenv("PIM_SCREEN_RW")) > 0;
  int rw_bad_rows = 0;
  fprintf(stderr, "[setup] pre-writing %zu rows%s... ", row_addr.size(),
          screen_rw ? " (+RW screen)" : "");
  fflush(stderr);
  for (size_t i = 0; i < row_addr.size(); i++) {
    per_col_write(platform, row_addr[i], row_mask[i].data());
    if (screen_rw) {
      std::vector<uint8_t> rb(8192);
      std::vector<uint32_t> anti(2048);
      for (int k = 0; k < 2048; k++) anti[k] = ~row_mask[i][k];
      read_row(platform, row_addr[i], rb.data(), 700000 + (int)i);
      int pat_m = n_match(rb.data(), row_mask[i].data());
      per_col_write(platform, row_addr[i], anti.data());
      read_row(platform, row_addr[i], rb.data(), 750000 + (int)i);
      int anti_m = n_match(rb.data(), anti.data());
      per_col_write(platform, row_addr[i], row_mask[i].data());
      if (pat_m < 2048 || anti_m < 2048) {
        printf("RWBAD R=%u pat_match=%d anti_match=%d\n",
               row_addr[i], pat_m, anti_m);
        rw_bad_rows++;
      }
    }
  }
  fprintf(stderr, "done\n");
  if (screen_rw)
    printf("# rw_screen: %d/%zu rows with any pattern/antipattern miss\n",
           rw_bad_rows, row_addr.size());

  // Iterate test sources.
  int total_faults = 0;
  for (size_t s_i = 0; s_i < sources.size(); s_i++) {
    uint32_t R = sources[s_i];

    // Look up R's mask index in row_addr (must exist — it's in safe zone).
    int R_idx = -1;
    for (size_t i = 0; i < row_addr.size(); i++)
      if (row_addr[i] == R) { R_idx = (int)i; break; }
    if (R_idx < 0) continue;

    // Re-fingerprint the entire single-bit XOR neighborhood + R + the
    // warmup rows BEFORE the test fires.  Without this, prior test
    // sources' XOR-spread artifacts pollute the neighbors and the
    // sweep reports false positives.
    {
      // Probe targets at all single-bit distances.
      for (int bit = 0; bit < 10; bit++) {
        uint32_t T = R ^ (1u << bit);
        if (T < SUB_START || T >= SUB_END) continue;
        if (is_open_row(T)) continue;
        int Ti = -1;
        for (size_t i = 0; i < row_addr.size(); i++)
          if (row_addr[i] == T) { Ti = (int)i; break; }
        if (Ti >= 0) per_col_write(platform, T, row_mask[Ti].data());
      }
      // Warmup rows.
      for (int k = 1; k <= n_warmup; k++) {
        uint32_t W = R - (uint32_t)(40 * k);
        if (W < SUB_START + 16 || is_open_row(W)) continue;
        int Wi = -1;
        for (size_t i = 0; i < row_addr.size(); i++)
          if (row_addr[i] == W) { Wi = (int)i; break; }
        if (Wi >= 0) per_col_write(platform, W, row_mask[Wi].data());
      }
      // R itself.
      per_col_write(platform, R, row_mask[R_idx].data());
    }

    // Warmup: run n_warmup MM3D bodies on rows physically near R.  Try
    // BELOW first (R-40k); if that lands outside the safe zone (because
    // R is near the bottom), fall back to ABOVE (R+40k).  Without this
    // fallback, sources near the bottom of the safe zone got zero
    // warmups and we missed state-dependent fault edges.
    int warmups_run = 0;
    for (int sign = -1; sign <= 1 && warmups_run < n_warmup; sign += 2) {
      for (int k = 1; k <= n_warmup && warmups_run < n_warmup; k++) {
        int32_t off = sign * 40 * k;
        if ((int)R + off < (int)(SUB_START + 16)) continue;
        if ((uint32_t)((int)R + off) >= SUB_END) continue;
        uint32_t warm_src = (uint32_t)((int)R + off);
        if (is_open_row(warm_src)) continue;
        int wi = -1;
        for (size_t i = 0; i < row_addr.size(); i++)
          if (row_addr[i] == warm_src) { wi = (int)i; break; }
        if (wi < 0) continue;
        per_col_write(platform, warm_src, row_mask[wi].data());
        fire_full_mm3d(platform, warm_src,
                        /*label_base=*/4000 + 1000 * k + (int)s_i * 7
                                     + (sign > 0 ? 500000 : 0));
        warmups_run++;
      }
    }

    // Re-write R AGAIN in case warmups disturbed it.
    per_col_write(platform, R, row_mask[R_idx].data());

    // Fire ONE MM3D body using R as backup_row.
    fire_full_mm3d(platform, R, /*label_base=*/8000 + (int)s_i * 17);

    // PAIR-WISE PROBE: read EVERY row in the safe zone back, check
    // against R's mask. Captures multi-bit XOR fault paths the
    // single-bit probe misses. Cost ~640 reads × ~1 ms per source =
    // <1 s/source × 624 sources = ~10 min for the full sweep.
    int faults_for_this_R = 0;
    std::vector<uint8_t> rb(8192);
    for (size_t i = 0; i < row_addr.size(); i++) {
      uint32_t T = row_addr[i];
      if (T == R) continue;  // source itself — skip
      read_row(platform, T, rb.data(),
               /*label=*/200000 + (int)s_i * 700 + (int)i);
      int R_m = n_match(rb.data(), row_mask[R_idx].data());
      if (R_m >= 2046) {
        // Decompose XOR into bit positions for downstream analysis.
        uint32_t xor_d = R ^ T;
        int popcnt = __builtin_popcount(xor_d);
        printf("FAULT R=%u -> T=%u  offset=%+d  xor=0x%05x  popcnt=%d\n",
               R, T, (int)T - (int)R, xor_d, popcnt);
        faults_for_this_R++;
        total_faults++;
      }
    }
    if (s_i % 16 == 0) {
      fprintf(stderr, "  R=%u  (%zu/%zu)  faults=%d  cum=%d\n",
              R, s_i + 1, sources.size(), faults_for_this_R, total_faults);
    }
  }

  printf("\n# total_faults=%d  sources_tested=%zu  warmup=%d\n",
         total_faults, sources.size(), n_warmup);

  // PIM_CHECK_CLONE=1 (2026-07-18): separate post-sweep pass — for each
  // source R, re-write R's fingerprint, fire ONLY the RowClone doubleACT
  // (t12=30, t23=1, R -> RFIRST) and read RFIRST back. Reports whether R
  // can act as a scratch/backup row for this calib at all. Needed for
  // windows that straddle a 1024-row predecoder block boundary (s86 on
  // DIMM 2 splits at 54272): cross-block APA pairs deposit nothing (see
  // sublattice_broadcast_2026_07_17/RESULT.md addendum 11), so rows on
  // the far side of the boundary from RFIRST cannot clone-load. Kept out
  // of the main sweep loop so the measured fault graph is untouched.
  if (getenv("PIM_CHECK_CLONE") && atoi(getenv("PIM_CHECK_CLONE")) > 0) {
    fprintf(stderr, "[clone-check] probing %zu sources...\n", sources.size());
    int n_ok = 0, n_fail = 0;
    std::vector<uint8_t> rb(8192);
    for (size_t s_i = 0; s_i < sources.size(); s_i++) {
      uint32_t R = sources[s_i];
      int R_idx = -1;
      for (size_t i = 0; i < row_addr.size(); i++)
        if (row_addr[i] == R) { R_idx = (int)i; break; }
      if (R_idx < 0) continue;
      per_col_write(platform, R, row_mask[R_idx].data());
      Program p;
      p.add_inst(SMC_LI(8, CASR));
      p.add_inst(SMC_LI(BANK_ID, BAR));
      p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_SLEEP(6));
      p.add_below(doubleACT(30, 1, R, RFIRST));
      p.add_inst(SMC_SLEEP(6));
      p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_SLEEP(6));
      p.add_inst(SMC_END());
      platform.execute(p);
      read_row(platform, RFIRST, rb.data(), 800000 + (int)s_i);
      int m = n_match(rb.data(), row_mask[R_idx].data());
      int ok = (m >= 2046) ? 1 : 0;
      printf("CLONE R=%u match=%d ok=%d\n", R, m, ok);
      if (ok) n_ok++; else n_fail++;
    }
    printf("# clone_check: ok=%d fail=%d\n", n_ok, n_fail);
  }
  return 0;
}
