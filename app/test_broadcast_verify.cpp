// BitNet PIM SW demo — Phase 3 prereq: verify NON-UNIFORM broadcast.
//
// MultiRowInit characterized broadcast for UNIFORM source patterns (all 0,
// all 1, or one repeated 32-bit). Phase 3 needs broadcast to preserve a
// NON-UNIFORM 8192-byte source (a real weight row) cell-by-cell across
// all 16 open rows. Physics says yes (per-bitline SAs), but verify.
//
// Test:
//   1. Per-column write a deterministic non-uniform 8192-byte W to Rfirst.
//   2. Uniform-write 0 to the other 15 open rows.
//   3. doubleACT(t_12=10, t_23=2, Rfirst, Rsecond) — broadcast.
//   4. Read back each of the 16 open rows.
//   5. Verify each row's 8192 bytes match W exactly.
//
// Pass = all 16×8192 = 131072 bytes match. Fail or partial = some bits
// average / lose / leak — broadcast doesn't preserve non-uniformity, and
// Phase 3 architecture must change.
//
// Argv:
//   ./broadcast-verify <bender_id> <calib_file> <bank_id> <s_id> [seed]
#include "instruction.h"
#include "prog.h"
#include "platform.h"
#include "../util.h"

#include <chrono>
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
    if (c.open_rows.size() != 16) continue;
    if (c.bank == wanted_bank && c.s_id == wanted_sid) out.push_back(c);
  }
  return out;
}

int main(int argc, char** argv) {
  if (argc < 5 || argc > 6) {
    cerr << "Usage: " << argv[0]
         << " <bender_id> <calib_file> <bank_id> <s_id> [seed]" << endl;
    return 1;
  }
  int bender_id   = atoi(argv[1]);
  string calib_p  = argv[2];
  int bank_id     = atoi(argv[3]);
  int target_sid  = atoi(argv[4]);
  unsigned seed   = (argc == 6) ? (unsigned)atoi(argv[5]) : 0xC0FFEE;

  vector<Calib> all_calibs = read_calib(calib_p, bank_id, target_sid);
  if (all_calibs.empty()) {
    cerr << "[bcast] no s_id=" << target_sid << " bank=" << bank_id
         << " in calib file" << endl;
    return 2;
  }
  Calib c = all_calibs[0];
  cerr << "[bcast] using s_id=" << c.s_id << " bank=" << c.bank
       << " Rfirst=" << c.Rfirst << " Rsecond=" << c.Rsecond << endl;

  // Generate non-uniform weight (2048 distinct uint32s = 8192 bytes).
  std::mt19937 rng(seed);
  vector<uint32_t> W(2048);
  for (auto& v : W) v = rng();

  SoftMCPlatform platform(bender_id);
  if (platform.init() != SOFTMC_SUCCESS) {
    cerr << "[bcast] platform init failed" << endl;
    return 3;
  }
  platform.reset_fpga();

  // Step 1: per-column write W to Rfirst.
  int col_start = 0;
  for (int chunk = 0; chunk < 3; chunk++) {
    int n_cols = CHUNK_COLS[chunk];
    Program p = build_chunk_program(bank_id, c.Rfirst,
                                     W.data() + col_start * 16,
                                     col_start, n_cols);
    platform.execute(p);
    col_start += n_cols;
  }
  cerr << "[bcast] step 1: per-column wrote W to Rfirst=" << c.Rfirst << endl;

  // Step 2: uniform 0 to the other 15 open rows.
  // Use one execute holding 15 wrRow_immediate_label calls (~30 inst each =
  // 450 inst, well within 2048).
  {
    Program p;
    p.add_inst(SMC_LI(8, CASR));
    p.add_inst(SMC_LI(bank_id, BAR));
    p.add_inst(SMC_LI(128, NUM_COLS_REG));
    p.add_below(PRE(BAR, 0, 0));
    int label_seed = 0;
    for (uint32_t r : c.open_rows) {
      if (r == c.Rfirst) continue;  // Don't clobber the source!
      p.add_below(wrRow_immediate_label(BAR, r, 0u, label_seed++));
    }
    p.add_inst(SMC_END());
    platform.execute(p);
    cerr << "[bcast] step 2: uniform-wrote 0 to other 15 open rows ("
         << label_seed << " writes, prog size " << (p.size()/8) << " insts)" << endl;
  }

  // Step 3: broadcast + read back all 16 rows in one program.
  // Program: PRE → doubleACT(broadcast) → PRE → 16× rdRow_immediate_label
  {
    Program p;
    p.add_inst(SMC_LI(8, CASR));
    p.add_inst(SMC_LI(bank_id, BAR));
    p.add_inst(SMC_LI(128, NUM_COLS_REG));
    p.add_below(PRE(BAR, 0, 0));
    p.add_inst(SMC_SLEEP(6));
    // Broadcast doubleACT — t_12=10, t_23=2 per MultiRowInit calibration
    // (sweet spot: any t_12 ≥ 10, t_23 ∈ {0..3} gives all_1=1.0 at fanout 16).
    p.add_below(doubleACT(10, 2, c.Rfirst, c.Rsecond));
    p.add_inst(SMC_SLEEP(6));
    p.add_below(PRE(BAR, 0, 0));
    p.add_inst(SMC_SLEEP(6));
    for (int i = 0; i < 16; i++) {
      // Each rdRow gets a unique label to avoid collision (see
      // softmc_label_collision memory).
      p.add_below(rdRow_immediate_label(BAR, c.open_rows[i], i));
    }
    p.add_inst(SMC_END());
    cerr << "[bcast] step 3: broadcast + 16× read prog size: "
         << (p.size()/8) << " insts" << endl;
    platform.execute(p);
  }

  // Receive 16 rows.
  vector<vector<uint8_t>> rows_back(16, vector<uint8_t>(8192));
  for (int i = 0; i < 16; i++) {
    int rc = platform.receiveData(rows_back[i].data(), 8192);
    if (rc != 8192) {
      cerr << "[bcast] receiveData row " << i << " rc=" << rc << endl;
      return 4;
    }
  }

  // Build expected non-uniform buffer (little-endian).
  uint8_t exp[8192];
  for (int s = 0; s < 2048; s++) {
    exp[s*4 + 0] = (uint8_t)((W[s] >>  0) & 0xFFu);
    exp[s*4 + 1] = (uint8_t)((W[s] >>  8) & 0xFFu);
    exp[s*4 + 2] = (uint8_t)((W[s] >> 16) & 0xFFu);
    exp[s*4 + 3] = (uint8_t)((W[s] >> 24) & 0xFFu);
  }

  // Score each of the 16 rows against W.
  cerr << "\n[bcast] === results ===" << endl;
  int total_pass = 0;
  for (int i = 0; i < 16; i++) {
    int byte_match = 0;
    long bit_match = 0;
    int byte_zero  = 0;  // bytes where read = 0 (i.e., kept the pre-broadcast 0)
    for (int b = 0; b < 8192; b++) {
      uint8_t read = rows_back[i][b];
      if (read == exp[b]) byte_match++;
      bit_match += 8 - __builtin_popcount((unsigned)(read ^ exp[b]));
      if (read == 0) byte_zero++;
    }
    bool is_rfirst = (c.open_rows[i] == c.Rfirst);
    cerr << "  pos=" << i
         << " row=" << c.open_rows[i]
         << (is_rfirst ? " (=Rfirst)" : "")
         << "  byte_match=" << byte_match << "/8192"
         << "  bit_match=" << bit_match << "/65536"
         << "  zero_bytes=" << byte_zero << "/8192"
         << endl;
    if (byte_match == 8192) total_pass++;
  }

  bool all_ok = (total_pass == 16);
  cerr << "\n[bcast] DONE — " << total_pass << "/16 rows hold W exactly. "
       << (all_ok ? "BROADCAST_OK" : "BROADCAST_PARTIAL_OR_FAIL") << endl;
  _exit(all_ok ? 0 : 6);
}
