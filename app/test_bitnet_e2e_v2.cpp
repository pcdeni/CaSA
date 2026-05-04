// BitNet PIM SW demo — Phase 4b: ternary W × int8 x via bit-decomposed
// activations. Builds on P4a's ternary-times-binary primitive.
//
// Per int8 activation x[j] = sum_b x_bit[j, b] * 2^b, where x_bit is the
// b-th bit of x[j] (b ∈ 0..7 for unsigned 8-bit, or sign-magnitude /
// two's-complement decomposition for signed). For DEMO clarity, use
// UNSIGNED int8 x ∈ [0, 255] so y = sum_j W[i,j] * x[j] = sum_b 2^b *
// (sum_j W[i,j] * x_bit[j, b]).
//
// Per bitplane: 2 MAJ3 (pos & neg). 16 MAJ3 total for the whole matmul.
// Each MAJ3 reuses the SAME pos_mask or neg_mask but a DIFFERENT
// activation pattern (one bit of x per dimension).
//
// Optimization: pos_mask and neg_mask are loaded ONCE each into Rfirst,
// then reused across the 8 bitplane MAJ3s for that sign — but Phase 3
// broadcast destroys r_first after each MAJ3, so we DO have to re-load
// the mask per MAJ3 (16 per-column writes total). Future phase: keep a
// deep backup (Tier 2 #5) to amortize.
//
// Argv:
//   ./bitnet-e2e-v2 <bender_id> <calib_file> <bank_id> [seed]
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
  p.add_below(doubleACT(/*t_12=*/10, /*t_23=*/2, Rfirst, Rsecond));
  p.add_inst(SMC_SLEEP(6));
  p.add_below(PRE(BAR, 0, 0));
  p.add_inst(SMC_SLEEP(6));
  static const int act_pos[5]  = {1, 4, 7, 10, 13};
  static const int zero_pos[5] = {2, 5, 8, 11, 14};
  p.add_below(wrRow_immediate_label(BAR, open_rows[0], ONE,
                                     label_seed + 0));
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
  p.add_below(doubleACT(/*t_12=*/0, /*t_23=*/0, Rfirst, Rsecond));
  p.add_inst(SMC_SLEEP(6));
  p.add_below(PRE(BAR, 0, 0));
  p.add_inst(SMC_SLEEP(6));
  p.add_below(rdRow_immediate_label(BAR, open_rows[0], label_seed + 999));
  p.add_inst(SMC_END());
  return p;
}

static void segment_popcount(const uint8_t* row_buf, int* out2048) {
  for (int s = 0; s < 2048; s++) {
    uint32_t actual = (uint32_t)row_buf[s*4]
                    | ((uint32_t)row_buf[s*4+1] << 8)
                    | ((uint32_t)row_buf[s*4+2] << 16)
                    | ((uint32_t)row_buf[s*4+3] << 24);
    out2048[s] = __builtin_popcount(actual);
  }
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
    if (c.open_rows.size() == 16 && c.bank == wanted_bank) out.push_back(c);
  }
  return out;
}

int main(int argc, char** argv) {
  if (argc < 4 || argc > 5) {
    cerr << "Usage: " << argv[0]
         << " <bender_id> <calib_file> <bank_id> [seed]" << endl;
    return 1;
  }
  int bender_id   = atoi(argv[1]);
  string calib_p  = argv[2];
  int bank_id     = atoi(argv[3]);
  unsigned seed   = (argc == 5) ? (unsigned)atoi(argv[4]) : 0xC0FFEE;

  vector<Calib> calib = read_calib(calib_p, bank_id);
  if (calib.empty()) { cerr << "[bnv2] no calib for bank " << bank_id << endl; return 2; }

  const int D_OUT = 2048;
  const int D_IN  = 32;
  const int N_BITPLANES = 8;

  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> trit(-1, 1);
  std::uniform_int_distribution<int> u8(0, 255);

  // Random ternary W (2048 × 32) and uint8 x (32 values).
  vector<vector<int>> W(D_OUT, vector<int>(D_IN));
  for (int i = 0; i < D_OUT; i++)
    for (int j = 0; j < D_IN; j++)
      W[i][j] = trit(rng);
  vector<int> x_int8(D_IN);
  for (int j = 0; j < D_IN; j++) x_int8[j] = u8(rng);

  // Pack W into pos_mask, neg_mask (one 32-bit segment per output).
  vector<uint32_t> pos_mask(D_OUT), neg_mask(D_OUT);
  for (int i = 0; i < D_OUT; i++) {
    uint32_t pm = 0, nm = 0;
    for (int j = 0; j < D_IN; j++) {
      if (W[i][j] == +1) pm |= (1u << j);
      if (W[i][j] == -1) nm |= (1u << j);
    }
    pos_mask[i] = pm;
    neg_mask[i] = nm;
  }
  // Bit-decompose x: x_bitplane[b] is a uint32 with j-th bit = bit b of x[j].
  vector<uint32_t> x_bitplane(N_BITPLANES, 0);
  for (int j = 0; j < D_IN; j++)
    for (int b = 0; b < N_BITPLANES; b++)
      if ((x_int8[j] >> b) & 1) x_bitplane[b] |= (1u << j);

  // Reference y on host.
  vector<int> y_ref(D_OUT);
  for (int i = 0; i < D_OUT; i++) {
    int s = 0;
    for (int j = 0; j < D_IN; j++) s += W[i][j] * x_int8[j];
    y_ref[i] = s;
  }
  cerr << "[bnv2] W: 2048×32 ternary, x: 32×uint8" << endl;
  cerr << "[bnv2] x_bitplanes[0..7]:";
  for (int b = 0; b < N_BITPLANES; b++)
    cerr << " 0x" << hex << x_bitplane[b];
  cerr << dec << endl;

  SoftMCPlatform platform(bender_id);
  if (platform.init() != SOFTMC_SUCCESS) { cerr << "[bnv2] init failed\n"; return 3; }
  platform.reset_fpga();

  using clk = std::chrono::steady_clock;
  using ns  = std::chrono::nanoseconds;

  const Calib& c = calib[0];
  cerr << "[bnv2] using s_id=" << c.s_id
       << " Rfirst=" << c.Rfirst << " Rsecond=" << c.Rsecond << endl;

  vector<int> y(D_OUT, 0);
  long t_total_ns = 0;

  // For each bitplane: load pos_mask, run MAJ3 for that bitplane's x,
  // popcount → pos_count[2048]. Then load neg_mask, MAJ3, popcount →
  // neg_count[2048]. y[i] += 2^b * (pos_count[i] - neg_count[i]).
  for (int b = 0; b < N_BITPLANES; b++) {
    uint32_t xb = x_bitplane[b];
    uint8_t row[8192];
    vector<int> pc(D_OUT), nc(D_OUT);
    auto t0 = clk::now();

    // pos
    per_column_write_row(platform, bank_id, c.Rfirst, pos_mask.data());
    {
      Program p = build_bcast_maj3_program(bank_id, c.Rfirst, c.Rsecond,
                                            c.open_rows.data(), xb,
                                            /*label_seed=*/b * 10000 + 100);
      platform.execute(p);
      int rc = platform.receiveData(row, 8192);
      if (rc != 8192) { cerr << "[bnv2] pos b=" << b << " rc=" << rc << endl; return 4; }
      segment_popcount(row, pc.data());
    }
    // neg
    per_column_write_row(platform, bank_id, c.Rfirst, neg_mask.data());
    {
      Program p = build_bcast_maj3_program(bank_id, c.Rfirst, c.Rsecond,
                                            c.open_rows.data(), xb,
                                            /*label_seed=*/b * 10000 + 200);
      platform.execute(p);
      int rc = platform.receiveData(row, 8192);
      if (rc != 8192) { cerr << "[bnv2] neg b=" << b << " rc=" << rc << endl; return 4; }
      segment_popcount(row, nc.data());
    }
    auto t1 = clk::now();
    t_total_ns += std::chrono::duration_cast<ns>(t1 - t0).count();

    int weight_2b = 1 << b;
    for (int i = 0; i < D_OUT; i++) y[i] += weight_2b * (pc[i] - nc[i]);
  }

  // Verify
  int n_match = 0;
  int max_err = 0;
  long sum_err = 0;
  int first_bad = -1;
  for (int i = 0; i < D_OUT; i++) {
    int err = std::abs(y[i] - y_ref[i]);
    if (err == 0) n_match++;
    if (err > max_err) max_err = err;
    sum_err += err;
    if (err != 0 && first_bad < 0) first_bad = i;
  }

  double ms = t_total_ns / 1e6;
  cerr << "[bnv2] pim_y[0..7]:  ";
  for (int i = 0; i < 8; i++) cerr << y[i] << " ";
  cerr << endl;
  cerr << "[bnv2] host_y[0..7]: ";
  for (int i = 0; i < 8; i++) cerr << y_ref[i] << " ";
  cerr << endl;
  cerr << "[bnv2] matches=" << n_match << "/" << D_OUT
       << " max_err=" << max_err
       << " mean_err=" << (double)sum_err / D_OUT
       << " first_bad=" << first_bad << endl;
  if (first_bad >= 0)
    cerr << "[bnv2] first_bad y=" << y[first_bad] << " y_ref=" << y_ref[first_bad] << endl;
  cerr << "[bnv2] total time: " << ms << " ms (16 MAJ3 = 8 bitplane × 2 sign)"
       << "  per_MAJ3=" << (ms / 16.0) << " ms" << endl;

  bool all_ok = (n_match == D_OUT);
  cerr << "[bnv2] DONE — " << (all_ok ? "ALL_PASS" : "FAIL") << endl;
  _exit(all_ok ? 0 : 6);
}
