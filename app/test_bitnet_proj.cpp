// BitNet PIM SW demo — Phase 4e runner: arbitrary projection with
// signed-int8 activation support (bit 7 contributes -128, not +128).
//
// Differences vs bitnet-real-exe:
//   - Input file v2 format: includes bitplane_factor[n_bitplanes] (int32)
//     so the host can specify per-bit signed weights — pass {1,2,4,8,16,
//     32,64,-128} for true two's-complement int8.
//   - Reads x_bitplane[n_chunks][n_bitplanes] directly (precomputed by
//     Python orchestrator) rather than x_int8.
//   - Doesn't ship a y_ref; just outputs y[d_out] to a file. The Python
//     orchestrator does the comparison.
//
// Argv:
//   ./bitnet-proj-exe <bender_id> <calib_file> <bank_id>
//                     <inputs.bin> <output.bin>
//
// Input file v2 format (little-endian):
//   uint32  magic = 0xB17EF002
//   uint32  d_in
//   uint32  d_out (must be 2048)
//   uint32  n_chunks (= d_in / 32)
//   uint32  n_bitplanes
//   uint32[n_chunks][d_out]    pos_mask
//   uint32[n_chunks][d_out]    neg_mask
//   uint32[n_chunks][n_bitplanes]  x_bitplane (precomputed)
//   int32[n_bitplanes]         bitplane_factor (e.g. {1,2,4,...,-128})
//
// Output file format:
//   int32[d_out]               y
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
  if (argc != 6) {
    cerr << "Usage: " << argv[0]
         << " <bender_id> <calib_file> <bank_id> <inputs.bin> <output.bin>"
         << endl;
    return 1;
  }
  int bender_id   = atoi(argv[1]);
  string calib_p  = argv[2];
  int bank_id     = atoi(argv[3]);
  string in_path  = argv[4];
  string out_path = argv[5];

  ifstream in(in_path, ios::binary);
  if (!in) { cerr << "[proj] cannot open " << in_path << endl; return 2; }
  uint32_t magic, d_in, d_out, n_chunks, n_bitplanes;
  in.read((char*)&magic, 4);
  if (magic != 0xB17EF002u) {
    cerr << "[proj] bad magic 0x" << hex << magic << dec << endl;
    return 2;
  }
  in.read((char*)&d_in, 4);
  in.read((char*)&d_out, 4);
  in.read((char*)&n_chunks, 4);
  in.read((char*)&n_bitplanes, 4);
  if (d_out != 2048) {
    cerr << "[proj] expected d_out=2048, got " << d_out << endl;
    return 2;
  }

  vector<vector<uint32_t>> pos_mask(n_chunks, vector<uint32_t>(d_out));
  vector<vector<uint32_t>> neg_mask(n_chunks, vector<uint32_t>(d_out));
  for (uint32_t c = 0; c < n_chunks; c++)
    in.read((char*)pos_mask[c].data(), d_out * 4);
  for (uint32_t c = 0; c < n_chunks; c++)
    in.read((char*)neg_mask[c].data(), d_out * 4);
  vector<vector<uint32_t>> x_bitplane(n_chunks,
                                       vector<uint32_t>(n_bitplanes));
  for (uint32_t c = 0; c < n_chunks; c++)
    in.read((char*)x_bitplane[c].data(), n_bitplanes * 4);
  vector<int32_t> bitplane_factor(n_bitplanes);
  in.read((char*)bitplane_factor.data(), n_bitplanes * 4);
  in.close();

  cerr << "[proj] d_in=" << d_in << " d_out=" << d_out
       << " n_chunks=" << n_chunks << " n_bitplanes=" << n_bitplanes
       << " factors=[";
  for (uint32_t b = 0; b < n_bitplanes; b++)
    cerr << bitplane_factor[b] << (b+1<n_bitplanes?", ":"");
  cerr << "]" << endl;

  vector<Calib> calib = read_calib(calib_p, bank_id);
  if (calib.empty()) { cerr << "[proj] no calib bank " << bank_id << endl; return 3; }
  const Calib& c = calib[0];

  SoftMCPlatform platform(bender_id);
  if (platform.init() != SOFTMC_SUCCESS) { cerr << "[proj] init failed\n"; return 4; }
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
        if (rc != 8192) { cerr << "[proj] rc=" << rc << endl; return 5; }
        vector<int> pc(d_out);
        segment_popcount(row, pc.data(), (int)d_out);
        auto t1 = clk::now();
        t_total_ns += std::chrono::duration_cast<ns>(t1 - t0).count();
        int weight = sign_factor * bitplane_factor[b];
        for (uint32_t i = 0; i < d_out; i++) y[i] += weight * pc[i];
        n_maj3++;
      }
    }
  }

  // Write y to output file.
  ofstream out(out_path, ios::binary);
  out.write((char*)y.data(), d_out * 4);
  out.close();

  double ms = t_total_ns / 1e6;
  cerr << "[proj] " << n_maj3 << " MAJ3, " << ms << " ms ("
       << (ms / n_maj3) << " ms/MAJ3)" << endl;
  cerr << "[proj] y[0..7]: ";
  for (int i = 0; i < 8; i++) cerr << y[i] << " ";
  cerr << endl;
  cerr << "[proj] DONE — wrote " << out_path << endl;
  _exit(0);
}
