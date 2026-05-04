// BitNet PIM SW demo — Phase 4c: multi-input-chunk ternary matmul.
//
// d_in arbitrary (default 256). d_out = 2048. Splits d_in into chunks
// of 32 (one MAJ3 per chunk per sign per bitplane). Each chunk needs
// its own pre-loaded weight row (separate pos_mask_chunk_c and
// neg_mask_chunk_c).
//
// Per (sign, bitplane, chunk): one MAJ3 trial. Accumulate per output:
//   y[i] = sum_chunk sum_b 2^b * (pos_count[i, c, b] - neg_count[i, c, b])
//
// This is the SAME pattern real BitNet uses — just bigger d_in (typically
// 2560 = 80 chunks). Per-column-write cost dominates; deep-backup
// optimization (Tier 2 #5) would amortize across the 8 bitplanes.
//
// Argv:
//   ./bitnet-e2e-v3 <bender_id> <calib_file> <bank_id> [d_in=256] [seed]
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
  if (argc < 4 || argc > 6) {
    cerr << "Usage: " << argv[0]
         << " <bender_id> <calib_file> <bank_id> [d_in=256] [seed]" << endl;
    return 1;
  }
  int bender_id   = atoi(argv[1]);
  string calib_p  = argv[2];
  int bank_id     = atoi(argv[3]);
  int D_IN        = (argc >= 5) ? atoi(argv[4]) : 256;
  unsigned seed   = (argc == 6) ? (unsigned)atoi(argv[5]) : 0xC0FFEE;
  if (D_IN % 32 != 0) {
    cerr << "[bnv3] d_in must be multiple of 32" << endl;
    return 2;
  }
  const int N_CHUNKS = D_IN / 32;

  vector<Calib> calib = read_calib(calib_p, bank_id);
  if (calib.empty()) { cerr << "[bnv3] no calib bank " << bank_id << endl; return 3; }

  const int D_OUT = 2048;
  const int N_BITPLANES = 8;

  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> trit(-1, 1);
  std::uniform_int_distribution<int> u8(0, 255);

  // Random ternary W and uint8 x.
  vector<vector<int>> W(D_OUT, vector<int>(D_IN));
  for (int i = 0; i < D_OUT; i++)
    for (int j = 0; j < D_IN; j++) W[i][j] = trit(rng);
  vector<int> x_int8(D_IN);
  for (int j = 0; j < D_IN; j++) x_int8[j] = u8(rng);

  // pos/neg masks per (chunk, output) — 32 inputs per chunk.
  // pos_mask[c][i] is the 32-bit pos-mask for output i, input chunk c.
  vector<vector<uint32_t>> pos_mask(N_CHUNKS, vector<uint32_t>(D_OUT, 0));
  vector<vector<uint32_t>> neg_mask(N_CHUNKS, vector<uint32_t>(D_OUT, 0));
  for (int i = 0; i < D_OUT; i++)
    for (int j = 0; j < D_IN; j++) {
      int c = j / 32, b = j % 32;
      if (W[i][j] == +1) pos_mask[c][i] |= (1u << b);
      if (W[i][j] == -1) neg_mask[c][i] |= (1u << b);
    }
  // x_bitplane[c][b] = uint32 with j-th bit = bit b of x_int8[c*32 + j].
  vector<vector<uint32_t>> x_bitplane(N_CHUNKS,
                                       vector<uint32_t>(N_BITPLANES, 0));
  for (int c = 0; c < N_CHUNKS; c++)
    for (int j = 0; j < 32; j++)
      for (int b = 0; b < N_BITPLANES; b++)
        if ((x_int8[c*32 + j] >> b) & 1) x_bitplane[c][b] |= (1u << j);

  // Reference y on host.
  vector<int> y_ref(D_OUT);
  for (int i = 0; i < D_OUT; i++) {
    int s = 0;
    for (int j = 0; j < D_IN; j++) s += W[i][j] * x_int8[j];
    y_ref[i] = s;
  }
  cerr << "[bnv3] W: " << D_OUT << "x" << D_IN << " ternary, x: "
       << D_IN << "x uint8, " << N_CHUNKS << " chunks × 8 bitplanes × 2 signs = "
       << (N_CHUNKS * N_BITPLANES * 2) << " MAJ3 total" << endl;

  SoftMCPlatform platform(bender_id);
  if (platform.init() != SOFTMC_SUCCESS) { cerr << "[bnv3] init failed\n"; return 4; }
  platform.reset_fpga();

  using clk = std::chrono::steady_clock;
  using ns  = std::chrono::nanoseconds;

  const Calib& c = calib[0];
  cerr << "[bnv3] using s_id=" << c.s_id
       << " Rfirst=" << c.Rfirst << " Rsecond=" << c.Rsecond << endl;

  vector<int> y(D_OUT, 0);
  long t_total_ns = 0;
  int n_maj3 = 0;
  int label_base = 0;

  // For each (chunk, sign): per-column write the mask, then run 8 bitplane MAJ3s.
  // (Mask is destroyed by each MAJ3, so we re-load between bitplanes.)
  // Future opt: deep backup + RowCopy refresh would let us load once per (chunk, sign).
  for (int chunk = 0; chunk < N_CHUNKS; chunk++) {
    for (int sign = 0; sign < 2; sign++) {
      const uint32_t* mask = (sign == 0)
        ? pos_mask[chunk].data()
        : neg_mask[chunk].data();
      int sign_factor = (sign == 0) ? +1 : -1;
      for (int b = 0; b < N_BITPLANES; b++) {
        uint32_t xb = x_bitplane[chunk][b];
        auto t0 = clk::now();
        per_column_write_row(platform, bank_id, c.Rfirst, mask);
        Program p = build_bcast_maj3_program(bank_id, c.Rfirst, c.Rsecond,
                                              c.open_rows.data(), xb,
                                              label_base);
        label_base += 1000;
        platform.execute(p);
        uint8_t row[8192];
        int rc = platform.receiveData(row, 8192);
        if (rc != 8192) { cerr << "[bnv3] rc=" << rc << endl; return 5; }
        vector<int> pc(D_OUT);
        segment_popcount(row, pc.data());
        auto t1 = clk::now();
        t_total_ns += std::chrono::duration_cast<ns>(t1 - t0).count();

        int weight = sign_factor * (1 << b);
        for (int i = 0; i < D_OUT; i++) y[i] += weight * pc[i];
        n_maj3++;
      }
    }
    cerr << "  chunk " << (chunk+1) << "/" << N_CHUNKS
         << " done (" << n_maj3 << " MAJ3 so far, "
         << (t_total_ns/1e6) << " ms)" << endl;
  }

  // Verify
  int n_match = 0, max_err = 0; long sum_err = 0; int first_bad = -1;
  for (int i = 0; i < D_OUT; i++) {
    int err = std::abs(y[i] - y_ref[i]);
    if (err == 0) n_match++;
    if (err > max_err) max_err = err;
    sum_err += err;
    if (err != 0 && first_bad < 0) first_bad = i;
  }
  double ms = t_total_ns / 1e6;
  cerr << "[bnv3] pim_y[0..7]:  ";
  for (int i = 0; i < 8; i++) cerr << y[i] << " ";
  cerr << endl;
  cerr << "[bnv3] host_y[0..7]: ";
  for (int i = 0; i < 8; i++) cerr << y_ref[i] << " ";
  cerr << endl;
  cerr << "[bnv3] matches=" << n_match << "/" << D_OUT
       << "  match_pct=" << (100.0 * n_match / D_OUT) << "%"
       << "  max_err=" << max_err
       << "  mean_err=" << (double)sum_err / D_OUT
       << "  first_bad=" << first_bad << endl;
  if (first_bad >= 0)
    cerr << "[bnv3] first_bad y=" << y[first_bad]
         << " y_ref=" << y_ref[first_bad] << endl;
  cerr << "[bnv3] " << n_maj3 << " MAJ3, total " << ms << " ms,"
       << " per_MAJ3=" << (ms / n_maj3) << " ms" << endl;

  cerr << "[bnv3] DONE" << endl;
  // Don't fail on small-percentage errors — see bitnet_phase4_validated
  // memory; a few cells exceed our 1000-pattern calibration's stability test.
  bool clean_pass = (n_match == D_OUT);
  bool acceptable = ((double)n_match / D_OUT >= 0.995);
  cerr << "[bnv3] " << (clean_pass ? "ALL_PASS" :
                       (acceptable ? "ACCEPTABLE (>=99.5%)" : "FAIL")) << endl;
  _exit(clean_pass ? 0 : (acceptable ? 0 : 6));
}
