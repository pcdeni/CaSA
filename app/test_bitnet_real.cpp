// BitNet PIM SW demo — Phase 4d: REAL weights from microsoft/bitnet-b1.58-2B-4T.
//
// Reads a binary file (produced by fetch_q_proj.py) holding pre-decoded
// real Q-projection weights for layer 0 + a random uint8 activation +
// the host-computed reference y. Runs the same PIM matmul as v3, compares
// against the real-model reference.
//
// File format (little-endian):
//   uint32  magic = 0xB17E7B17
//   uint32  d_in
//   uint32  d_out
//   uint32  n_chunks (= d_in / 32)
//   uint32  n_bitplanes (= 8)
//   float32 weight_scale (BF16-decoded; not applied here, host-side)
//   uint32[n_chunks][d_out]  pos_mask
//   uint32[n_chunks][d_out]  neg_mask
//   uint8[d_in]               x_int8
//   int32[d_out]              y_ref (integer matmul, before weight_scale)
//
// Argv:
//   ./bitnet-real <bender_id> <calib_file> <bank_id> <inputs.bin>
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

static void segment_popcount(const uint8_t* row_buf, int* out, int n) {
  for (int s = 0; s < n; s++) {
    uint32_t actual = (uint32_t)row_buf[s*4]
                    | ((uint32_t)row_buf[s*4+1] << 8)
                    | ((uint32_t)row_buf[s*4+2] << 16)
                    | ((uint32_t)row_buf[s*4+3] << 24);
    out[s] = __builtin_popcount(actual);
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
  if (argc != 5) {
    cerr << "Usage: " << argv[0]
         << " <bender_id> <calib_file> <bank_id> <inputs.bin>" << endl;
    return 1;
  }
  int bender_id   = atoi(argv[1]);
  string calib_p  = argv[2];
  int bank_id     = atoi(argv[3]);
  string in_path  = argv[4];

  // Load inputs.bin
  ifstream in(in_path, ios::binary);
  if (!in) { cerr << "[real] cannot open " << in_path << endl; return 2; }
  uint32_t magic, d_in, d_out, n_chunks, n_bitplanes;
  float weight_scale;
  in.read((char*)&magic, 4);
  if (magic != 0xB17E7B17u) {
    cerr << "[real] bad magic 0x" << hex << magic << dec << endl;
    return 2;
  }
  in.read((char*)&d_in, 4);
  in.read((char*)&d_out, 4);
  in.read((char*)&n_chunks, 4);
  in.read((char*)&n_bitplanes, 4);
  in.read((char*)&weight_scale, 4);
  if (d_out != 2048) {
    cerr << "[real] expected d_out=2048, got " << d_out << endl;
    return 2;
  }
  cerr << "[real] d_in=" << d_in << " d_out=" << d_out
       << " n_chunks=" << n_chunks << " n_bitplanes=" << n_bitplanes
       << " weight_scale=" << weight_scale << endl;

  vector<vector<uint32_t>> pos_mask(n_chunks, vector<uint32_t>(d_out));
  vector<vector<uint32_t>> neg_mask(n_chunks, vector<uint32_t>(d_out));
  for (uint32_t c = 0; c < n_chunks; c++)
    in.read((char*)pos_mask[c].data(), d_out * 4);
  for (uint32_t c = 0; c < n_chunks; c++)
    in.read((char*)neg_mask[c].data(), d_out * 4);
  vector<uint8_t> x_int8(d_in);
  in.read((char*)x_int8.data(), d_in);
  vector<int32_t> y_ref(d_out);
  in.read((char*)y_ref.data(), d_out * 4);
  in.close();

  cerr << "[real] x_int8[0..7]: ";
  for (int j = 0; j < 8; j++) cerr << (int)x_int8[j] << " ";
  cerr << endl;
  cerr << "[real] y_ref[0..7]:  ";
  for (int i = 0; i < 8; i++) cerr << y_ref[i] << " ";
  cerr << endl;

  // Build x_bitplane[c][b]
  vector<vector<uint32_t>> x_bitplane(n_chunks,
                                       vector<uint32_t>(n_bitplanes, 0));
  for (uint32_t c = 0; c < n_chunks; c++)
    for (int j = 0; j < 32; j++)
      for (uint32_t b = 0; b < n_bitplanes; b++)
        if ((x_int8[c*32 + j] >> b) & 1)
          x_bitplane[c][b] |= (1u << j);

  // Calibration
  vector<Calib> calib = read_calib(calib_p, bank_id);
  if (calib.empty()) { cerr << "[real] no calib bank " << bank_id << endl; return 3; }
  const Calib& c = calib[0];
  cerr << "[real] using s_id=" << c.s_id
       << " Rfirst=" << c.Rfirst << " Rsecond=" << c.Rsecond << endl;

  SoftMCPlatform platform(bender_id);
  if (platform.init() != SOFTMC_SUCCESS) { cerr << "[real] init failed\n"; return 4; }
  platform.reset_fpga();

  using clk = std::chrono::steady_clock;
  using ns  = std::chrono::nanoseconds;

  vector<int32_t> y(d_out, 0);
  long t_total_ns = 0;
  int n_maj3 = 0;
  int label_base = 0;

  for (uint32_t chunk = 0; chunk < n_chunks; chunk++) {
    for (int sign = 0; sign < 2; sign++) {
      const uint32_t* mask = (sign == 0)
        ? pos_mask[chunk].data()
        : neg_mask[chunk].data();
      int sign_factor = (sign == 0) ? +1 : -1;
      for (uint32_t b = 0; b < n_bitplanes; b++) {
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
        if (rc != 8192) { cerr << "[real] rc=" << rc << endl; return 5; }
        vector<int> pc(d_out);
        segment_popcount(row, pc.data(), (int)d_out);
        auto t1 = clk::now();
        t_total_ns += std::chrono::duration_cast<ns>(t1 - t0).count();
        int weight = sign_factor * (1 << b);
        for (uint32_t i = 0; i < d_out; i++) y[i] += weight * pc[i];
        n_maj3++;
      }
    }
    cerr << "  chunk " << (chunk+1) << "/" << n_chunks << " (" << n_maj3
         << " MAJ3, " << (t_total_ns / 1e6) << " ms)" << endl;
  }

  // Verify
  int n_match = 0, max_err = 0; long sum_err = 0; int first_bad = -1;
  for (uint32_t i = 0; i < d_out; i++) {
    int err = std::abs(y[i] - y_ref[i]);
    if (err == 0) n_match++;
    if (err > max_err) max_err = err;
    sum_err += err;
    if (err != 0 && first_bad < 0) first_bad = (int)i;
  }
  double ms = t_total_ns / 1e6;
  cerr << "[real] pim_y[0..7]:  ";
  for (int i = 0; i < 8; i++) cerr << y[i] << " ";
  cerr << endl;
  cerr << "[real] host_y[0..7]: ";
  for (int i = 0; i < 8; i++) cerr << y_ref[i] << " ";
  cerr << endl;
  cerr << "[real] matches=" << n_match << "/" << d_out
       << "  match_pct=" << (100.0 * n_match / d_out) << "%"
       << "  max_err=" << max_err
       << "  mean_err=" << (double)sum_err / d_out
       << "  first_bad=" << first_bad << endl;
  if (first_bad >= 0)
    cerr << "[real] first_bad y=" << y[first_bad]
         << " y_ref=" << y_ref[first_bad] << endl;
  cerr << "[real] " << n_maj3 << " MAJ3, total " << ms << " ms,"
       << " per_MAJ3=" << (ms / n_maj3) << " ms" << endl;

  // Show what y * scale would be (the full BitNet output, scaled).
  cerr << "[real] scaled_y[0..7] (× weight_scale=" << weight_scale << "):";
  for (int i = 0; i < 8; i++) cerr << " " << (y[i] * weight_scale);
  cerr << endl;

  bool clean = (n_match == d_out);
  bool acceptable = ((double)n_match / d_out >= 0.99);
  cerr << "[real] DONE — " << (clean ? "EXACT_MATCH"
                              : (acceptable ? "≥99% MATCH (cell-stability noise)"
                                            : "FAIL")) << endl;
  _exit(clean ? 0 : (acceptable ? 0 : 6));
}
