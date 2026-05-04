// BitNet PIM SW demo — Phase 3: dense MatVec with broadcast weight refresh.
//
// Improvement over Phase 2b: instead of per-column writing the weight to
// 5 weight-slot rows (3.5 ms), we per-column write ONCE to Rfirst (one
// of the 16 open rows) then use `doubleACT(t_12=10, t_23=2, Rfirst,
// Rsecond)` to BROADCAST that weight to all 16 open rows in a single
// SoftMC operation (~6 insts). After broadcast, we overwrite 11 of the
// rows with uniform x/0/ONE patterns; the 5 weight slots retain W.
//
// Confirmed by `test_broadcast_verify` that non-uniform 8192-byte content
// broadcasts cell-by-cell at fanout 16 (16/16 rows byte-exact).
//
// Per-MAJ3 cost: 1 per-column write (~0.7 ms) + 1 small program with
// broadcast+uniform_writes+frac+MAJ3+read (~0.5 ms) = ~1.2 ms total,
// down from Phase 2b's ~3.7 ms. ~3× speedup.
//
// Note: r_first holds MAJ_RESULT (= W & x at each bit) AFTER the MAJ3,
// not W. So back-to-back MAJ3s sharing the same W still need to re-load
// W via per-column write each time — broadcast doesn't help across MAJ3
// trials without a "deep backup" outside the open set (Tier 2 #5 in the
// optimization log).
//
// Argv:
//   ./dense-matvec-bcast <bender_id> <calib_file> <bank_id> <K> [seed]
//
// Same K-trial structure as Phase 2b; each trial uses the next calibrated
// tuple, fresh random W, shared x.
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

// Per-column non-uniform write of one row, split across 3 chunks of
// ~43 columns each. See test_columnwise_smoke for the SMC_SLEEP(8)
// after every WRITE rationale (tWR + WDATA-modify race).
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

// One MAJ3 trial program — broadcast + uniform overwrite + frac + MAJ3.
// Assumes r_first already holds W (per-column-loaded by caller).
//
// Layout reminder:
//   open_rows[0]                 ← ONE (frac, discharged before MAJ3)
//   open_rows[1, 4, 7, 10, 13]   ← x (uniform 32-bit)
//   open_rows[2, 5, 8, 11, 14]   ← 0 (uniform)
//   open_rows[3, 6, 9, 12, 15]   ← W (preserved from broadcast)
//
// We must NOT overwrite the weight slots after broadcast. We overwrite
// the 5 act + 5 zero + 1 frac slots = 11 rows.
static Program build_broadcast_maj3_program(int bank_id,
                                             uint32_t Rfirst,
                                             uint32_t Rsecond,
                                             const uint32_t* open_rows,
                                             uint32_t x_pattern,
                                             int label_seed) {
  Program p;
  p.add_inst(SMC_LI(8, CASR));
  p.add_inst(SMC_LI(bank_id, BAR));
  p.add_inst(SMC_LI(128, NUM_COLS_REG));
  p.add_below(PRE(BAR, 0, 0));
  p.add_inst(SMC_SLEEP(6));

  // Step 1: broadcast doubleACT — W lands in all 16 open rows.
  p.add_below(doubleACT(/*t_12=*/10, /*t_23=*/2, Rfirst, Rsecond));
  p.add_inst(SMC_SLEEP(6));
  p.add_below(PRE(BAR, 0, 0));
  p.add_inst(SMC_SLEEP(6));

  // Step 2: overwrite the 11 non-weight slots with their target patterns.
  // Preserve weight in positions {3, 6, 9, 12, 15}.
  static const int act_pos[5]  = {1, 4, 7, 10, 13};
  static const int zero_pos[5] = {2, 5, 8, 11, 14};
  // Frac slot (position 0) gets ONE.
  p.add_below(wrRow_immediate_label(BAR, open_rows[0], ONE,
                                     label_seed + 0));
  for (int i = 0; i < 5; i++)
    p.add_below(wrRow_immediate_label(BAR, open_rows[act_pos[i]],
                                       x_pattern,
                                       label_seed + 1 + i));
  for (int i = 0; i < 5; i++)
    p.add_below(wrRow_immediate_label(BAR, open_rows[zero_pos[i]], 0u,
                                       label_seed + 100 + i));

  p.add_inst(SMC_SLEEP(6));
  p.add_below(PRE(BAR, 0, 0));
  p.add_inst(SMC_SLEEP(6));

  // Step 3: frac discharge × 3 on open_rows[0] (the ONE / frac slot).
  for (int j = 0; j < 3; j++) {
    p.add_inst(SMC_SLEEP(6));
    p.add_inst(SMC_LI(open_rows[0], RF_REG));
    Mininst quad[4] = {
      SMC_ACT(BAR, 0, RF_REG, 0),
      SMC_PRE(BAR, 0, 0),
      SMC_NOP(),
      SMC_NOP(),
    };
    p.add_inst(quad[0], quad[1], quad[2], quad[3]);
    p.add_inst(SMC_SLEEP(6));
  }
  p.add_inst(SMC_SLEEP(6));
  p.add_below(PRE(BAR, 0, 0));
  p.add_inst(SMC_SLEEP(6));

  // Step 4: MAJ3 doubleACT.
  p.add_below(doubleACT(/*t_12=*/0, /*t_23=*/0, Rfirst, Rsecond));
  p.add_inst(SMC_SLEEP(6));
  p.add_below(PRE(BAR, 0, 0));
  p.add_inst(SMC_SLEEP(6));

  // Step 5: read back open_rows[0] (all 16 hold MAJ_RESULT, just pick one).
  p.add_below(rdRow_immediate_label(BAR, open_rows[0], label_seed + 999));
  p.add_inst(SMC_END());
  return p;
}

struct Calib {
  int s_id, bank;
  uint32_t Rfirst, Rsecond;
  vector<uint32_t> open_rows;
};

static vector<Calib> read_calib(const string& path, int wanted_bank) {
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
    if (c.bank == wanted_bank) out.push_back(c);
  }
  return out;
}

int main(int argc, char** argv) {
  if (argc < 5 || argc > 6) {
    cerr << "Usage: " << argv[0]
         << " <bender_id> <calib_file> <bank_id> <K> [seed]" << endl;
    return 1;
  }
  int bender_id   = atoi(argv[1]);
  string calib_p  = argv[2];
  int bank_id     = atoi(argv[3]);
  int K           = atoi(argv[4]);
  unsigned seed   = (argc == 6) ? (unsigned)atoi(argv[5]) : 0xC0FFEE;

  vector<Calib> calib = read_calib(calib_p, bank_id);
  if ((int)calib.size() < K) {
    cerr << "[bcastv] calib file has only " << calib.size()
         << " tuples for bank " << bank_id << ", need K=" << K << endl;
    return 2;
  }
  cerr << "[bcastv] bank=" << bank_id << " K=" << K << endl;

  std::mt19937 rng(seed);
  uint32_t x = rng();
  vector<vector<uint32_t>> W_rows(K, vector<uint32_t>(2048));
  for (int k = 0; k < K; k++)
    for (auto& v : W_rows[k]) v = rng();
  cerr << "[bcastv] x=0x" << hex << x << dec << endl;

  SoftMCPlatform platform(bender_id);
  if (platform.init() != SOFTMC_SUCCESS) {
    cerr << "[bcastv] platform init failed" << endl;
    return 3;
  }
  platform.reset_fpga();

  using clk = std::chrono::steady_clock;
  using ns  = std::chrono::nanoseconds;
  long long t_w_ns = 0, t_m_ns = 0;
  int total_byte_pass = 0, total_pop_pass = 0;

  for (int k = 0; k < K; k++) {
    Calib& c = calib[k];

    // 1. Per-column write W to Rfirst (3 executes, ~0.7 ms).
    auto t0 = clk::now();
    per_column_write_row(platform, bank_id, c.Rfirst, W_rows[k].data());
    auto t1 = clk::now();

    // 2. Single program: broadcast → uniform writes → frac → MAJ3 → read.
    Program p = build_broadcast_maj3_program(bank_id, c.Rfirst, c.Rsecond,
                                              c.open_rows.data(), x,
                                              k * 1000);
    platform.execute(p);
    uint8_t row_buf[8192];
    int rc = platform.receiveData(row_buf, 8192);
    auto t2 = clk::now();
    if (rc != 8192) {
      cerr << "[bcastv] k=" << k << " receiveData rc=" << rc << endl;
      return 4;
    }

    // Verify each segment k = (W[k] & x).
    int byte_mismatches = 0;
    int seg_mismatches  = 0;
    int first_bad_seg = -1;
    long total_pop = 0;
    long expected_pop = 0;
    for (int s = 0; s < 2048; s++) {
      uint32_t actual = (uint32_t)row_buf[s*4]
                      | ((uint32_t)row_buf[s*4+1] << 8)
                      | ((uint32_t)row_buf[s*4+2] << 16)
                      | ((uint32_t)row_buf[s*4+3] << 24);
      uint32_t expected = W_rows[k][s] & x;
      if (actual != expected) {
        seg_mismatches++;
        if (first_bad_seg < 0) first_bad_seg = s;
      }
      total_pop += __builtin_popcount(actual);
      expected_pop += __builtin_popcount(expected);
    }
    for (int i = 0; i < 8192; i += 4) {
      uint32_t expected = W_rows[k][i/4] & x;
      uint8_t e[4] = {
        (uint8_t)(expected & 0xFF),
        (uint8_t)((expected >> 8) & 0xFF),
        (uint8_t)((expected >> 16) & 0xFF),
        (uint8_t)((expected >> 24) & 0xFF),
      };
      for (int j = 0; j < 4; j++)
        if (row_buf[i+j] != e[j]) byte_mismatches++;
    }

    bool byte_pass = (byte_mismatches == 0);
    bool pop_pass  = (total_pop == expected_pop);
    if (byte_pass) total_byte_pass++;
    if (pop_pass)  total_pop_pass++;

    t_w_ns += std::chrono::duration_cast<ns>(t1 - t0).count();
    t_m_ns += std::chrono::duration_cast<ns>(t2 - t1).count();

    cerr << "[bcastv] k=" << k
         << " s_id=" << c.s_id
         << " " << (byte_pass ? "PASS" : "FAIL")
         << " bytes=" << byte_mismatches << "/8192"
         << " segs=" << seg_mismatches << "/2048"
         << " host_pop=" << expected_pop << " fpga_pop=" << total_pop;
    if (!byte_pass)
      cerr << " first_bad_seg=" << first_bad_seg
           << " W=0x" << hex << W_rows[k][first_bad_seg]
           << " expected=0x" << (W_rows[k][first_bad_seg] & x) << dec;
    cerr << endl;
  }

  double avg_w_ms = t_w_ns / 1e6 / K;
  double avg_m_ms = t_m_ns / 1e6 / K;
  cerr << "[bcastv] timing per MAJ3: weight_load (1× per-col)=" << avg_w_ms
       << " ms  bcast+MAJ3+read=" << avg_m_ms << " ms  "
       << "total=" << (avg_w_ms + avg_m_ms) << " ms" << endl;

  bool all_ok = (total_byte_pass == K && total_pop_pass == K);
  cerr << "[bcastv] DONE — byte_exact " << total_byte_pass << "/" << K
       << " popcount " << total_pop_pass << "/" << K
       << " " << (all_ok ? "ALL_PASS" : "FAIL") << endl;
  _exit(all_ok ? 0 : 6);
}
