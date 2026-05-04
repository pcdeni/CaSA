// BitNet PIM SW demo — Phase 2b: dense MatVec (2048 outputs per MAJ3).
//
// Combines Phase 1 (MAJ3 with calibrated 16-open-row tuple) and Phase 2a
// (per-column non-uniform writes) to produce 2048 distinct
// (W[k] AND x) results from a single doubleACT.
//
// Layout per calibrated tuple (s_id, Rfirst, Rsecond, open_rows[16]):
//   open_rows[0]                 ← ONE (frac slot, discharged before MAJ3)
//   open_rows[1, 4, 7, 10, 13]   ← x (uniform 32-bit, 5 copies)
//   open_rows[2, 5, 8, 11, 14]   ← 0 (uniform, 5 copies)
//   open_rows[3, 6, 9, 12, 15]   ← W (NON-UNIFORM 8192-byte weight row,
//                                    2048 distinct 32-bit segments,
//                                    5 copies each per-column-written)
//
// After doubleACT(t_12=0, t_23=0, Rfirst, Rsecond): result row's
// segment k = MAJ3(W[k], x_uniform, 0) = AND(W[k], x). Reading back
// gives 2048 outputs of a 32-input × 2048-output binary MatVec in ONE
// MAJ3 trial.
//
// Per MAJ3 cost dominated by 5× per-column weight loads (~0.7 ms each =
// 3.5 ms) + 11× uniform writes (~0.5 ms) + frac/MAJ/read (~0.3 ms) ≈ 4.5 ms.
// For BitNet b1.58-2B-4T this projects to ~3 min / token (ballpark);
// plenty of room to optimize via Multi-RowCopy for weight refresh.
//
// Argv:
//   ./dense-matvec-smoke <bender_id> <calib_file> <bank_id> <K> [seed]
//
// K = number of MAJ3 trials (each trial uses next calibrated tuple). For
// each, generates fresh random W (2048 segments), shared x; verifies all
// 2048 segments against host MatVec. Reports per-trial PASS/FAIL plus
// timing breakdown.
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

// Same chunk split as Phase 2a (43 + 43 + 42 cols).
static const int CHUNK_COLS[3] = {43, 43, 42};

// Per-column write of arbitrary 8192-byte content into one row.
// data points to 2048 uint32 (16 per column × 128 columns).
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
    // tWR — see softmc_tWR_after_write memory.
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

// Uniform-pattern wrRow program for a single row + activation/zero
// fill. Reuses the existing wrRow_immediate_label primitive (per-column
// hw loop, ~30 inst). Used for the 11 uniform-pattern slot writes.
static Program build_uniform_writes_program(int bank_id,
                                             uint32_t one_row,
                                             const vector<uint32_t>& act_rows,  // 5
                                             uint32_t x_pattern,
                                             const vector<uint32_t>& zero_rows, // 5
                                             int label_seed) {
  Program p;
  p.add_inst(SMC_LI(8, CASR));
  p.add_inst(SMC_LI(bank_id, BAR));
  p.add_inst(SMC_LI(128, NUM_COLS_REG));
  p.add_below(PRE(BAR, 0, 0));
  // Frac (ONE) slot
  p.add_below(wrRow_immediate_label(BAR, one_row, ONE, label_seed + 0));
  // Activation slots
  for (size_t i = 0; i < act_rows.size(); i++)
    p.add_below(wrRow_immediate_label(BAR, act_rows[i], x_pattern,
                                       label_seed + 1 + (int)i));
  // Zero slots
  for (size_t i = 0; i < zero_rows.size(); i++)
    p.add_below(wrRow_immediate_label(BAR, zero_rows[i], 0,
                                       label_seed + 100 + (int)i));
  p.add_inst(SMC_END());
  return p;
}

// Frac discharge + doubleACT + result read.
static Program build_maj3_finish_program(int bank_id,
                                          uint32_t r_first,
                                          uint32_t r_second,
                                          uint32_t open_row_0,
                                          int n_frac_times,
                                          int label_seed) {
  Program p;
  p.add_inst(SMC_LI(8, CASR));
  p.add_inst(SMC_LI(bank_id, BAR));
  p.add_inst(SMC_LI(128, NUM_COLS_REG));
  p.add_below(PRE(BAR, 0, 0));
  p.add_inst(SMC_SLEEP(6));
  // Frac: ACT(open_row_0) + PRE — discharges that row's cells.
  for (int j = 0; j < n_frac_times; j++) {
    p.add_inst(SMC_SLEEP(6));
    p.add_inst(SMC_LI(open_row_0, RF_REG));
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
  p.add_below(doubleACT(0, 0, r_first, r_second));
  p.add_inst(SMC_SLEEP(6));
  p.add_below(PRE(BAR, 0, 0));
  p.add_inst(SMC_SLEEP(6));
  // rdRow_immediate_label — single use per program → label safe.
  p.add_below(rdRow_immediate_label(BAR, open_row_0, label_seed));
  p.add_inst(SMC_END());
  return p;
}

struct Calib {
  int s_id, bank;
  uint32_t Rfirst, Rsecond;
  vector<uint32_t> open_rows;  // 16
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
    cerr << "[densev] calib file has only " << calib.size()
         << " tuples for bank " << bank_id << ", need K=" << K << endl;
    return 2;
  }
  cerr << "[densev] bank=" << bank_id << " K=" << K
       << " (out of " << calib.size() << ")" << endl;

  std::mt19937 rng(seed);
  uint32_t x = rng();   // shared activation across all K MAJ3s
  vector<vector<uint32_t>> W_rows(K, vector<uint32_t>(2048));
  for (int k = 0; k < K; k++)
    for (auto& v : W_rows[k]) v = rng();

  cerr << "[densev] x=0x" << hex << x << dec << endl;

  SoftMCPlatform platform(bender_id);
  if (platform.init() != SOFTMC_SUCCESS) {
    cerr << "[densev] platform init failed" << endl;
    return 3;
  }
  platform.reset_fpga();

  using clk = std::chrono::steady_clock;
  using ns  = std::chrono::nanoseconds;

  // Slot positions in open_rows[]:
  //   frac (ONE):   [0]
  //   activation:   [1, 4, 7, 10, 13]
  //   zero:         [2, 5, 8, 11, 14]
  //   weight:       [3, 6, 9, 12, 15]
  static const int act_pos[5]    = {1, 4, 7, 10, 13};
  static const int zero_pos[5]   = {2, 5, 8, 11, 14};
  static const int weight_pos[5] = {3, 6, 9, 12, 15};

  int total_byte_pass = 0, total_pop_pass = 0;
  long long t_w_ns = 0, t_u_ns = 0, t_m_ns = 0;

  for (int k = 0; k < K; k++) {
    Calib& c = calib[k];
    vector<uint32_t> act_rows, zero_rows, weight_rows;
    for (int i = 0; i < 5; i++) act_rows.push_back(c.open_rows[act_pos[i]]);
    for (int i = 0; i < 5; i++) zero_rows.push_back(c.open_rows[zero_pos[i]]);
    for (int i = 0; i < 5; i++) weight_rows.push_back(c.open_rows[weight_pos[i]]);
    uint32_t one_row = c.open_rows[0];

    // 1. Per-column write the SAME weight pattern W_rows[k] to all 5 weight slots.
    auto t0 = clk::now();
    for (uint32_t wr : weight_rows)
      per_column_write_row(platform, bank_id, wr, W_rows[k].data());
    auto t1 = clk::now();

    // 2. Uniform writes for ONE / activation / zero — one big program.
    Program up = build_uniform_writes_program(bank_id, one_row,
                                               act_rows, x, zero_rows,
                                               /*label_seed=*/k * 1000);
    platform.execute(up);
    auto t2 = clk::now();

    // 3. Frac + doubleACT + read.
    Program mp = build_maj3_finish_program(bank_id, c.Rfirst, c.Rsecond,
                                            one_row, /*n_frac_times=*/3,
                                            /*label_seed=*/k * 1000 + 999);
    platform.execute(mp);
    uint8_t row_buf[8192];
    int rc = platform.receiveData(row_buf, 8192);
    auto t3 = clk::now();
    if (rc != 8192) {
      cerr << "[densev] k=" << k << " receiveData rc=" << rc << endl;
      return 4;
    }

    // Verify byte-exact: each segment k = AND(W_segment_k, x).
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
    t_u_ns += std::chrono::duration_cast<ns>(t2 - t1).count();
    t_m_ns += std::chrono::duration_cast<ns>(t3 - t2).count();

    cerr << "[densev] k=" << k
         << " s_id=" << c.s_id
         << " Rfirst=" << c.Rfirst << " Rsecond=" << c.Rsecond
         << " " << (byte_pass ? "PASS" : "FAIL")
         << " bytes=" << byte_mismatches << "/8192"
         << " segs=" << seg_mismatches << "/2048"
         << " host_pop=" << expected_pop << " fpga_pop=" << total_pop;
    if (!byte_pass)
      cerr << " first_bad_seg=" << first_bad_seg
           << " W[seg]=0x" << hex << W_rows[k][first_bad_seg]
           << " expected=0x" << (W_rows[k][first_bad_seg] & x)
           << dec;
    cerr << endl;
  }

  double avg_w_ms = t_w_ns / 1e6 / K;
  double avg_u_ms = t_u_ns / 1e6 / K;
  double avg_m_ms = t_m_ns / 1e6 / K;
  cerr << "[densev] timing per MAJ3: weight_load=" << avg_w_ms << " ms  "
       << "uniform_writes=" << avg_u_ms << " ms  "
       << "MAJ3+read=" << avg_m_ms << " ms  "
       << "total=" << (avg_w_ms + avg_u_ms + avg_m_ms) << " ms" << endl;

  bool all_ok = (total_byte_pass == K && total_pop_pass == K);
  cerr << "[densev] DONE — byte_exact " << total_byte_pass << "/" << K
       << ", popcount " << total_pop_pass << "/" << K
       << " " << (all_ok ? "ALL_PASS" : "FAIL") << endl;
  _exit(all_ok ? 0 : 6);
}
