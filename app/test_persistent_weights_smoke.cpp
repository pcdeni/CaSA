// Persistent-weights end-to-end smoke. Verifies the FULL flow:
//   1. per-column write W to a backup row (OUTSIDE the calibrated open_rows)
//   2. RowClone(backup → Rfirst)        ← cheap doubleACT, the win
//   3. Broadcast doubleACT(10, 2)       ← spreads W to all 16 open rows
//   4. Overwrite 11 non-weight slots (uniform x/0/ONE)
//   5. frac × 3
//   6. MAJ3 doubleACT(0, 0)
//   7. read result row
//   8. compare to direct-per-col baseline (same W, same x, today's path)
//
// If results match, persistent weights (Tier B #1) is unblocked: per
// matmul we per-col-write each weight ONCE to its backup row, then use
// fast RowClones to refresh Rfirst before each MAJ3. ~3-5× speedup.
//
// Argv:
//   ./persistent-smoke <bender_id> <calib_file> <bank_id> <s_id>
//                      <backup_row> [seed]
#include "instruction.h"
#include "prog.h"
#include "platform.h"
#include "../util.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
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

// Same broadcast-MAJ3 program structure as test_bitnet_proj.cpp, but
// the caller has ALREADY put W into Rfirst (either via per-col write
// directly OR via RowClone from a backup row). The choice of how Rfirst
// got loaded is OPAQUE to this program.
static Program build_bcast_maj3_program(int bank_id,
                                         uint32_t Rfirst, uint32_t Rsecond,
                                         const uint32_t* open_rows,
                                         uint32_t x_pattern,
                                         int label_seed) {
  Program p;
  p.add_inst(SMC_LI(8, CASR));
  p.add_inst(SMC_LI(bank_id, BAR));
  p.add_inst(SMC_LI(128, NUM_COLS_REG));
  p.add_below(PRE(BAR, 0, 0));
  p.add_inst(SMC_SLEEP(6));
  p.add_below(doubleACT(10, 2, Rfirst, Rsecond));
  p.add_inst(SMC_SLEEP(6));
  p.add_below(PRE(BAR, 0, 0));
  p.add_inst(SMC_SLEEP(6));
  static const int act_pos[5]  = {1, 4, 7, 10, 13};
  static const int zero_pos[5] = {2, 5, 8, 11, 14};
  p.add_below(wrRow_immediate_label(BAR, open_rows[0], ONE, label_seed));
  for (int i = 0; i < 5; i++)
    p.add_below(wrRow_immediate_label(BAR, open_rows[act_pos[i]],
                                       x_pattern, label_seed + 1 + i));
  for (int i = 0; i < 5; i++)
    p.add_below(wrRow_immediate_label(BAR, open_rows[zero_pos[i]], 0u,
                                       label_seed + 100 + i));
  p.add_inst(SMC_SLEEP(6));
  p.add_below(PRE(BAR, 0, 0));
  p.add_inst(SMC_SLEEP(6));
  for (int j = 0; j < 3; j++) {
    p.add_inst(SMC_SLEEP(6));
    p.add_inst(SMC_LI(open_rows[0], RF_REG));
    p.add_inst(SMC_ACT(BAR, 0, RF_REG, 0),
               SMC_PRE(BAR, 0, 0),
               SMC_NOP(),
               SMC_NOP());
    p.add_inst(SMC_SLEEP(6));
  }
  p.add_inst(SMC_SLEEP(6));
  p.add_below(PRE(BAR, 0, 0));
  p.add_inst(SMC_SLEEP(6));
  p.add_below(doubleACT(0, 0, Rfirst, Rsecond));
  p.add_inst(SMC_SLEEP(6));
  p.add_below(PRE(BAR, 0, 0));
  p.add_inst(SMC_SLEEP(6));
  p.add_below(rdRow_immediate_label(BAR, open_rows[0], label_seed + 999));
  p.add_inst(SMC_END());
  return p;
}

// RowClone in one program: doubleACT(30, 1, src, dst). Bumps dst's
// content to a copy of src's. dst MUST be precharged before the clone
// (the program does PRE first).
static Program build_rowclone_program(int bank_id,
                                       uint32_t src_row, uint32_t dst_row,
                                       int label_seed) {
  Program p;
  p.add_inst(SMC_LI(8, CASR));
  p.add_inst(SMC_LI(bank_id, BAR));
  p.add_inst(SMC_LI(128, NUM_COLS_REG));
  p.add_below(PRE(BAR, 0, 0));
  p.add_inst(SMC_SLEEP(6));
  p.add_below(doubleACT(/*t_12=*/30, /*t_23=*/1, src_row, dst_row));
  p.add_inst(SMC_SLEEP(6));
  p.add_below(PRE(BAR, 0, 0));
  p.add_inst(SMC_SLEEP(6));
  p.add_inst(SMC_END());
  return p;
}

struct Calib {
  int s_id, bank;
  uint32_t Rfirst, Rsecond;
  vector<uint32_t> open_rows;
};

static vector<Calib> read_calib(const string& path, int wanted_bank,
                                 int wanted_sid) {
  vector<Calib> out;
  ifstream f(path);
  string line;
  while (getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    istringstream iss(line);
    Calib c;
    if (!(iss >> c.s_id >> c.bank >> c.Rfirst >> c.Rsecond)) continue;
    uint32_t v;
    while (iss >> v) c.open_rows.push_back(v);
    if (c.open_rows.size() == 16 && c.bank == wanted_bank
        && c.s_id == wanted_sid) out.push_back(c);
  }
  return out;
}

static int build_16row_pos(int /*pos*/, vector<uint32_t>& patterns,
                            uint32_t W, uint32_t x) {
  // 5 W at positions {3,6,9,12,15}, 5 x at {1,4,7,10,13}, 5 zero at
  // {2,5,8,11,14}, ONE at 0. Same layout used in dense-matvec-bcast.
  patterns.assign(16, 0);
  patterns[0] = ONE;
  static const int wp[5]  = {3, 6, 9, 12, 15};
  static const int xp[5]  = {1, 4, 7, 10, 13};
  static const int zp[5]  = {2, 5, 8, 11, 14};
  for (int i = 0; i < 5; i++) {
    patterns[wp[i]] = W;
    patterns[xp[i]] = x;
    patterns[zp[i]] = 0;
  }
  return 0;
}

int main(int argc, char** argv) {
  if (argc < 6 || argc > 7) {
    cerr << "Usage: " << argv[0]
         << " <bender_id> <calib_file> <bank_id> <s_id> <backup_row> [seed]"
         << endl;
    return 1;
  }
  int bender_id = atoi(argv[1]);
  string calib_p = argv[2];
  int bank_id   = atoi(argv[3]);
  int sid       = atoi(argv[4]);
  uint32_t backup = (uint32_t)atoi(argv[5]);
  unsigned seed = (argc == 7) ? (unsigned)atoi(argv[6]) : 0xC0FFEE;

  vector<Calib> calib = read_calib(calib_p, bank_id, sid);
  if (calib.empty()) {
    cerr << "[psw] no calib for bank=" << bank_id << " s_id=" << sid << endl;
    return 2;
  }
  Calib c = calib[0];
  cerr << "[psw] bank=" << bank_id << " s_id=" << sid
       << " Rfirst=" << c.Rfirst << " Rsecond=" << c.Rsecond
       << " backup=" << backup << " seed=" << seed << endl;

  // Build a non-uniform 8192-byte W and a uniform 32-bit activation x.
  std::mt19937 rng(seed);
  vector<uint32_t> W(2048);
  for (auto& v : W) v = rng();
  uint32_t x = rng();
  cerr << "[psw] x=0x" << hex << x << dec << endl;

  SoftMCPlatform platform(bender_id);
  if (platform.init() != SOFTMC_SUCCESS) { cerr << "[psw] init failed\n"; return 3; }
  platform.reset_fpga();

  // ---------- BASELINE: today's per-MAJ3 per-column-write path ----------
  // 1. per-col write W directly to Rfirst.
  per_column_write_row(platform, bank_id, c.Rfirst, W.data());
  // 2. broadcast + uniform writes + frac + MAJ3 + read.
  Program bp = build_bcast_maj3_program(bank_id, c.Rfirst, c.Rsecond,
                                         c.open_rows.data(), x, 1000);
  platform.execute(bp);
  uint8_t base_row[8192];
  if (platform.receiveData(base_row, 8192) != 8192) return 4;
  cerr << "[psw] baseline (per-col→Rfirst) result done" << endl;

  // ---------- NEW: persistent-weights via backup + RowClone ----------
  // 1. per-col write W to backup row (this is ONE-TIME per matmul).
  per_column_write_row(platform, bank_id, backup, W.data());
  cerr << "[psw] wrote W to backup row " << backup << endl;
  // 2. RowClone backup → Rfirst (the cheap step that replaces per-col write).
  Program rcp = build_rowclone_program(bank_id, backup, c.Rfirst, 2000);
  platform.execute(rcp);
  // 3. Broadcast + uniform writes + frac + MAJ3 + read.
  Program bp2 = build_bcast_maj3_program(bank_id, c.Rfirst, c.Rsecond,
                                          c.open_rows.data(), x, 3000);
  platform.execute(bp2);
  uint8_t pwt_row[8192];
  if (platform.receiveData(pwt_row, 8192) != 8192) return 5;
  cerr << "[psw] persistent-weights (backup→clone→Rfirst) result done" << endl;

  // ---------- compare ----------
  int byte_match = 0;
  for (int i = 0; i < 8192; i++) if (base_row[i] == pwt_row[i]) byte_match++;
  cerr << "[psw] byte match (baseline vs persistent): " << byte_match
       << "/8192" << endl;

  // Also segment-level popcount comparison.
  long base_pop = 0, pwt_pop = 0;
  for (int i = 0; i < 8192; i += 8) {
    uint64_t a, b;
    memcpy(&a, base_row + i, 8); memcpy(&b, pwt_row + i, 8);
    base_pop += __builtin_popcountll(a);
    pwt_pop  += __builtin_popcountll(b);
  }
  cerr << "[psw] popcount: baseline=" << base_pop
       << "  persistent=" << pwt_pop << endl;

  // Run the persistent path AGAIN to confirm backup is preserved (subsequent
  // RowClones still produce the same result).
  // After first MAJ3, Rfirst is destroyed (it's in the open set).
  Program rcp2 = build_rowclone_program(bank_id, backup, c.Rfirst, 4000);
  platform.execute(rcp2);
  Program bp3 = build_bcast_maj3_program(bank_id, c.Rfirst, c.Rsecond,
                                          c.open_rows.data(), x, 5000);
  platform.execute(bp3);
  uint8_t pwt2_row[8192];
  if (platform.receiveData(pwt2_row, 8192) != 8192) return 6;
  int second_match = 0;
  for (int i = 0; i < 8192; i++) if (base_row[i] == pwt2_row[i]) second_match++;
  cerr << "[psw] 2nd persistent-weights run match: " << second_match
       << "/8192 (verifies backup survives MAJ3 destruction of Rfirst)" << endl;

  bool pass = (byte_match == 8192 && second_match == 8192);
  cerr << "[psw] DONE — " << (pass ? "PASS — persistent weights work end-to-end"
                                     : "FAIL") << endl;
  _exit(pass ? 0 : 6);
}
