// Persistent-weights prereq: verify SiMRA RowClone (the 2-row
// charge-sharing copy at t_12=30, t_23=1) works between an
// "out-of-set" backup row and the Rfirst of one of our calibrated
// MAJ3-perfect tuples in the SAME subarray. If yes, we can:
//   (1) per-column write the weight ONCE to backup_row per matmul
//   (2) per MAJ3, do a fast doubleACT to refresh Rfirst from backup
// This eliminates the per-MAJ3 per-column write that today dominates
// per-MAJ3 cost (~700 µs).
//
// Test:
//   - per-column write a deterministic non-uniform 8192-byte buffer W
//     to backup_row (outside the calibrated open_rows[16])
//   - doubleACT(t_12=30, t_23=1, backup_row, target_row)  ← RowClone
//   - read target_row, compare byte-exact to W
//   - report PASS/FAIL + how many bytes match
//
// We sweep t_23 ∈ {1,2,3,4} per the SiMRA RowClone test, since the
// best t_23 for an arbitrary subarray isn't a priori known.
//
// Argv:
//   ./rowclone-smoke <bender_id> <bank_id> <backup_row> <target_row> [seed]
#include "instruction.h"
#include "prog.h"
#include "platform.h"
#include "../util.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <unistd.h>
#include <vector>

using namespace std;

static const int CHUNK_COLS[3] = {43, 43, 42};

static Program build_chunk_program(int bank_id, uint32_t row_addr,
                                    const uint32_t* col_data,
                                    int col_start, int n_cols) {
  Program p;
  p.add_inst(SMC_LI(8, CASR));
  p.add_inst(SMC_LI(bank_id, BAR));
  p.add_inst(SMC_LI(row_addr, RAR));
  p.add_inst(SMC_LI(col_start * 8, CAR));
  p.add_below(PRE(BAR, 0, 0));
  p.add_below(ACT(BAR, 0, RAR, 0));
  for (int k = 0; k < n_cols; k++) {
    const uint32_t* slots = col_data + k * 16;
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
  return p;
}

static void per_column_write_row(SoftMCPlatform& platform, int bank_id,
                                  uint32_t row, const uint32_t* data_2048) {
  int col_start = 0;
  for (int chunk = 0; chunk < 3; chunk++) {
    int n_cols = CHUNK_COLS[chunk];
    Program p = build_chunk_program(bank_id, row,
                                     data_2048 + col_start * 16,
                                     col_start, n_cols);
    platform.execute(p);
    col_start += n_cols;
  }
}

// RowClone (charge-sharing 2-row copy): doubleACT(30, t_23, src, dst).
// Reads dst back. Returns # bytes matching expected.
static int try_rowclone(SoftMCPlatform& platform, int bank_id,
                        uint32_t src_row, uint32_t dst_row,
                        int t_23, const uint8_t* expected_8192,
                        int label_seed) {
  Program p;
  p.add_inst(SMC_LI(8, CASR));
  p.add_inst(SMC_LI(bank_id, BAR));
  p.add_inst(SMC_LI(128, NUM_COLS_REG));
  p.add_below(PRE(BAR, 0, 0));
  p.add_inst(SMC_SLEEP(6));
  // RowClone: high t_12 lets SAs lock to src content, then short t_23
  // before the second ACT lets dst row latch the SA's content.
  p.add_below(doubleACT(/*t_12=*/30, /*t_23=*/t_23, src_row, dst_row));
  p.add_inst(SMC_SLEEP(6));
  p.add_below(PRE(BAR, 0, 0));
  p.add_inst(SMC_SLEEP(6));
  p.add_below(rdRow_immediate_label(BAR, dst_row, label_seed));
  p.add_inst(SMC_END());
  platform.execute(p);

  uint8_t row[8192];
  int rc = platform.receiveData(row, 8192);
  if (rc != 8192) return -1;

  int matches = 0;
  for (int i = 0; i < 8192; i++)
    if (row[i] == expected_8192[i]) matches++;
  return matches;
}

int main(int argc, char** argv) {
  if (argc < 5 || argc > 6) {
    cerr << "Usage: " << argv[0]
         << " <bender_id> <bank_id> <backup_row> <target_row> [seed]" << endl;
    return 1;
  }
  int bender_id = atoi(argv[1]);
  int bank_id   = atoi(argv[2]);
  uint32_t bk   = (uint32_t)atoi(argv[3]);
  uint32_t tg   = (uint32_t)atoi(argv[4]);
  unsigned seed = (argc == 6) ? (unsigned)atoi(argv[5]) : 0xC0FFEE;

  std::mt19937 rng(seed);
  vector<uint32_t> data(2048);
  for (auto& v : data) v = rng();

  // Build expected 8192-byte buffer.
  uint8_t exp[8192];
  for (int i = 0; i < 2048; i++) {
    exp[i*4 + 0] = (uint8_t)(data[i] & 0xFFu);
    exp[i*4 + 1] = (uint8_t)((data[i] >>  8) & 0xFFu);
    exp[i*4 + 2] = (uint8_t)((data[i] >> 16) & 0xFFu);
    exp[i*4 + 3] = (uint8_t)((data[i] >> 24) & 0xFFu);
  }

  cerr << "[rcsmoke] bender=" << bender_id << " bank=" << bank_id
       << " backup=" << bk << " target=" << tg << " seed=" << seed << endl;

  SoftMCPlatform platform(bender_id);
  if (platform.init() != SOFTMC_SUCCESS) { cerr << "[rcsmoke] init failed\n"; return 2; }
  platform.reset_fpga();

  // Pre-fill target row with all 0s so any non-zero readback came from clone.
  vector<uint32_t> zeros(2048, 0);
  per_column_write_row(platform, bank_id, tg, zeros.data());
  // Per-column write W to backup row.
  per_column_write_row(platform, bank_id, bk, data.data());
  cerr << "[rcsmoke] wrote zeros to target (" << tg << "), W to backup (" << bk
       << ")" << endl;

  // Sweep t_23 ∈ {1, 2, 3, 4} as in SiMRA RowClone.
  for (int t_23 = 1; t_23 <= 4; t_23++) {
    // Re-init target to zeros each iteration (since previous attempt may
    // have partially clobbered it).
    per_column_write_row(platform, bank_id, tg, zeros.data());
    int matches = try_rowclone(platform, bank_id, bk, tg, t_23, exp, t_23);
    cerr << "  t_23=" << t_23 << ": match=" << matches << "/8192"
         << " (" << (100.0 * matches / 8192) << "%)";
    if (matches == 8192) cerr << "  PERFECT_CLONE";
    cerr << endl;
  }

  cerr << "[rcsmoke] DONE" << endl;
  _exit(0);
}
