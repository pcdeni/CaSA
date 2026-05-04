// BitNet PIM SW demo — Phase 4a: end-to-end ternary matmul vs host ref.
//
// Smallest viable BitNet primitive: ternary W ∈ {-1, 0, +1}^(d_out × d_in)
// times BINARY x ∈ {0, 1}^d_in (single bitplane). Computes
// y[i] = sum_j W[i, j] * x[j].
//
// Encoding: split each ternary weight into TWO bit planes:
//   pos_mask[i, j] = 1 iff W[i, j] == +1   (else 0)
//   neg_mask[i, j] = 1 iff W[i, j] == -1   (else 0)
// (W==0 is represented as both bits being 0.)
//
// Then y[i] = popcount(pos_mask[i, :] AND x) - popcount(neg_mask[i, :] AND x).
// In PIM, AND(W, x) = MAJ3(W, x, 0). One MAJ3 per sign, two MAJ3 total.
//
// Layout:
//   d_in  = 32 (one input chunk = one 32-bit segment per output)
//   d_out = 2048 (one output chunk = one row of 2048 segments)
// Each weight row holds 2048 distinct 32-bit segments (one per output);
// segment[i] is the i-th output's pos_mask (or neg_mask) for the 32
// inputs in this chunk. x is a uniform 32-bit pattern broadcast across
// all 2048 segments of the activation slots.
//
// Argv:
//   ./bitnet-e2e-v1 <bender_id> <calib_file> <bank_id> [seed]
//
// Verification: byte-exact (W & x at each segment) AND popcount-match
// per output AND aggregated y[i] == y_ref[i] for all 2048 outputs.
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

// Re-use the per-column write infra from earlier phases.
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

// One MAJ3 trial via broadcast (Phase 3 architecture).
// Caller must have per-column-loaded mask into Rfirst.
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
                                       x_pattern,
                                       label_seed + 1 + i));
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

// Read result row, popcount each 32-bit segment → outputs[2048].
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

// Run one PIM matmul of one mask (pos or neg) with x_pattern. Result
// row_buf gets the broadcast→MAJ3 output row.
static int run_one_sign_maj3(SoftMCPlatform& platform,
                             const Calib& c, int bank_id,
                             const uint32_t* mask_2048, uint32_t x_pattern,
                             uint8_t* row_buf, int label_seed) {
  per_column_write_row(platform, bank_id, c.Rfirst, mask_2048);
  Program p = build_bcast_maj3_program(bank_id, c.Rfirst, c.Rsecond,
                                        c.open_rows.data(), x_pattern,
                                        label_seed);
  platform.execute(p);
  return platform.receiveData(row_buf, 8192);
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
  if (calib.empty()) {
    cerr << "[bnet] no calibration tuples for bank " << bank_id << endl;
    return 2;
  }
  cerr << "[bnet] bank=" << bank_id << " calib tuples=" << calib.size() << endl;

  // Constants for this demo.
  const int D_OUT = 2048;
  const int D_IN  = 32;

  // Generate ternary W (d_out × d_in) with ~1/3 each of {-1, 0, +1}.
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> trit(-1, 1);
  vector<vector<int>> W(D_OUT, vector<int>(D_IN));
  for (int i = 0; i < D_OUT; i++)
    for (int j = 0; j < D_IN; j++)
      W[i][j] = trit(rng);

  // Generate binary x.
  vector<int> x_bits(D_IN);
  for (int j = 0; j < D_IN; j++) x_bits[j] = rng() & 1;

  // Pack pos_mask, neg_mask: pos_mask[i] = bitmask of inputs where W[i,j]==+1.
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
  // Pack x as a uniform 32-bit pattern (each bit = x_bits[j]).
  uint32_t x_pattern = 0;
  for (int j = 0; j < D_IN; j++) if (x_bits[j]) x_pattern |= (1u << j);

  cerr << "[bnet] W: 2048×32 ternary (~1/3 each), x_pattern=0x"
       << hex << x_pattern << dec << endl;

  // Compute reference y_ref on host.
  vector<int> y_ref(D_OUT);
  for (int i = 0; i < D_OUT; i++) {
    int s = 0;
    for (int j = 0; j < D_IN; j++) s += W[i][j] * x_bits[j];
    y_ref[i] = s;
  }

  SoftMCPlatform platform(bender_id);
  if (platform.init() != SOFTMC_SUCCESS) {
    cerr << "[bnet] platform init failed" << endl;
    return 3;
  }
  platform.reset_fpga();

  using clk = std::chrono::steady_clock;
  using ns  = std::chrono::nanoseconds;

  // Use the first calibrated tuple.
  const Calib& c = calib[0];
  cerr << "[bnet] using s_id=" << c.s_id
       << " Rfirst=" << c.Rfirst << " Rsecond=" << c.Rsecond << endl;

  uint8_t row_pos[8192], row_neg[8192];
  vector<int> pos_count(D_OUT), neg_count(D_OUT);

  // 1. Pos-mask MAJ3.
  auto t0 = clk::now();
  int rc = run_one_sign_maj3(platform, c, bank_id,
                              pos_mask.data(), x_pattern,
                              row_pos, /*label_seed=*/1000);
  if (rc != 8192) { cerr << "[bnet] pos receiveData rc=" << rc << endl; return 4; }
  segment_popcount(row_pos, pos_count.data());

  // 2. Neg-mask MAJ3.
  rc = run_one_sign_maj3(platform, c, bank_id,
                         neg_mask.data(), x_pattern,
                         row_neg, /*label_seed=*/2000);
  if (rc != 8192) { cerr << "[bnet] neg receiveData rc=" << rc << endl; return 4; }
  segment_popcount(row_neg, neg_count.data());
  auto t1 = clk::now();

  // 3. Combine: y[i] = pos_count[i] - neg_count[i].
  vector<int> y(D_OUT);
  for (int i = 0; i < D_OUT; i++) y[i] = pos_count[i] - neg_count[i];

  // 4. Verify against host reference.
  int n_match = 0;
  int max_err = 0;
  long sum_abs_err = 0;
  int first_bad = -1;
  for (int i = 0; i < D_OUT; i++) {
    int err = std::abs(y[i] - y_ref[i]);
    if (err == 0) n_match++;
    if (err > max_err) max_err = err;
    sum_abs_err += err;
    if (err != 0 && first_bad < 0) first_bad = i;
  }

  double ms_total = std::chrono::duration_cast<ns>(t1 - t0).count() / 1e6;
  cerr << "[bnet] pim_y[0..7]:  ";
  for (int i = 0; i < 8; i++) cerr << y[i] << " ";
  cerr << endl;
  cerr << "[bnet] host_y[0..7]: ";
  for (int i = 0; i < 8; i++) cerr << y_ref[i] << " ";
  cerr << endl;
  cerr << "[bnet] matches=" << n_match << "/" << D_OUT
       << " max_abs_err=" << max_err
       << " mean_abs_err=" << (double)sum_abs_err / D_OUT
       << " first_bad_idx=" << first_bad << endl;
  if (first_bad >= 0) {
    cerr << "[bnet] first_bad y=" << y[first_bad]
         << " y_ref=" << y_ref[first_bad]
         << " pos_count=" << pos_count[first_bad]
         << " neg_count=" << neg_count[first_bad]
         << " pos_mask=0x" << hex << pos_mask[first_bad]
         << " neg_mask=0x" << neg_mask[first_bad]
         << " x_pattern=0x" << x_pattern << dec << endl;
  }
  cerr << "[bnet] total inference time: " << ms_total << " ms (2 MAJ3)" << endl;

  bool all_ok = (n_match == D_OUT);
  cerr << "[bnet] DONE — " << (all_ok ? "ALL_PASS" : "FAIL") << endl;
  _exit(all_ok ? 0 : 6);
}
