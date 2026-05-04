// BitNet PIM SW demo — Path C: multi-bank dense MatVec.
//
// Same broadcast architecture as Phase 3, but packs 4 MAJ3 trials (one
// per DDR4 bank, each with its own calibrated tuple and weight) into a
// SINGLE platform.execute(). Amortizes per-execute PCIe overhead across
// 4 trials. The 4 banks' MAJ3 sequences run sequentially WITHIN the
// SoftMC program (one SoftMC core, one instruction queue) — DDR4 bank
// parallelism only helps where banks can overlap (e.g. one bank's tRAS
// hides while another bank issues commands). Realistic speedup: ~1.3×
// per-MAJ3 over Phase 3.
//
// Per-bank weight pre-load is still 3 executes per row (per-column
// writes) — could be interleaved across banks in one execute for
// further savings, but marginal given the write-data path is shared.
//
// Argv:
//   ./multibank-bcast <bender_id> <calib_file> <n_batches> [seed]
//
// Each batch runs 4 MAJ3s (one per bank, picking the next calibrated
// tuple for each bank). Total MAJ3s = 4 * n_batches. Each result row's
// 2048 segments are verified byte-exact AND popcount-exact.
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

// One bank's MAJ3 trial body (broadcast → uniform writes → frac → MAJ3
// → read). Caller switches BAR before this. Mostly identical to Phase 3,
// but takes label_base for unique label naming across banks within one
// program.
static void emit_bank_maj3_body(Program& program,
                                 int bank_id,
                                 uint32_t Rfirst, uint32_t Rsecond,
                                 const uint32_t* open_rows,
                                 uint32_t x_pattern,
                                 int label_base) {
  // Switch BAR for this bank, set per-bank registers.
  program.add_inst(SMC_LI(bank_id, BAR));
  program.add_inst(SMC_LI(128, NUM_COLS_REG));
  program.add_below(PRE(BAR, 0, 0));
  program.add_inst(SMC_SLEEP(6));

  // Step 1: broadcast — W lands in all 16 open rows of this bank.
  program.add_below(doubleACT(/*t_12=*/10, /*t_23=*/2, Rfirst, Rsecond));
  program.add_inst(SMC_SLEEP(6));
  program.add_below(PRE(BAR, 0, 0));
  program.add_inst(SMC_SLEEP(6));

  // Step 2: overwrite 11 non-weight slots (preserve positions 3,6,9,12,15).
  static const int act_pos[5]  = {1, 4, 7, 10, 13};
  static const int zero_pos[5] = {2, 5, 8, 11, 14};
  program.add_below(wrRow_immediate_label(BAR, open_rows[0], ONE,
                                           label_base + 0));
  for (int i = 0; i < 5; i++)
    program.add_below(wrRow_immediate_label(BAR, open_rows[act_pos[i]],
                                             x_pattern,
                                             label_base + 1 + i));
  for (int i = 0; i < 5; i++)
    program.add_below(wrRow_immediate_label(BAR, open_rows[zero_pos[i]], 0u,
                                             label_base + 100 + i));

  program.add_inst(SMC_SLEEP(6));
  program.add_below(PRE(BAR, 0, 0));
  program.add_inst(SMC_SLEEP(6));

  // Step 3: frac discharge × 3.
  for (int j = 0; j < 3; j++) {
    program.add_inst(SMC_SLEEP(6));
    program.add_inst(SMC_LI(open_rows[0], RF_REG));
    program.add_inst(SMC_ACT(BAR, 0, RF_REG, 0),
                     SMC_PRE(BAR, 0, 0),
                     SMC_NOP(),
                     SMC_NOP());
    program.add_inst(SMC_SLEEP(6));
  }
  program.add_inst(SMC_SLEEP(6));
  program.add_below(PRE(BAR, 0, 0));
  program.add_inst(SMC_SLEEP(6));

  // Step 4: MAJ3.
  program.add_below(doubleACT(/*t_12=*/0, /*t_23=*/0, Rfirst, Rsecond));
  program.add_inst(SMC_SLEEP(6));
  program.add_below(PRE(BAR, 0, 0));
  program.add_inst(SMC_SLEEP(6));

  // Step 5: read result.
  program.add_below(rdRow_immediate_label(BAR, open_rows[0],
                                           label_base + 999));
}

struct Calib {
  int s_id, bank;
  uint32_t Rfirst, Rsecond;
  vector<uint32_t> open_rows;
};

static vector<Calib> read_calib_all(const string& path) {
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
    if (c.open_rows.size() == 16) out.push_back(c);
  }
  return out;
}

int main(int argc, char** argv) {
  if (argc < 4 || argc > 5) {
    cerr << "Usage: " << argv[0]
         << " <bender_id> <calib_file> <n_batches> [seed]" << endl;
    return 1;
  }
  int bender_id   = atoi(argv[1]);
  string calib_p  = argv[2];
  int n_batches   = atoi(argv[3]);
  unsigned seed   = (argc == 5) ? (unsigned)atoi(argv[4]) : 0xC0FFEE;

  vector<Calib> all = read_calib_all(calib_p);
  // Group by bank for easy per-bank indexing.
  vector<vector<Calib>> by_bank(4);
  for (auto& c : all) if (c.bank >= 0 && c.bank < 4) by_bank[c.bank].push_back(c);
  for (int b = 0; b < 4; b++) {
    if ((int)by_bank[b].size() < n_batches) {
      cerr << "[mb] bank " << b << " has only " << by_bank[b].size()
           << " calibrated tuples (need " << n_batches << ")" << endl;
      return 2;
    }
  }
  cerr << "[mb] per-bank calibrated tuples: "
       << by_bank[0].size() << " "
       << by_bank[1].size() << " "
       << by_bank[2].size() << " "
       << by_bank[3].size()
       << "  n_batches=" << n_batches << endl;

  std::mt19937 rng(seed);
  uint32_t x = rng();   // shared across all MAJ3s
  // 4 weights per batch, n_batches batches → 4*n_batches total weights.
  vector<vector<uint32_t>> W_all(4 * n_batches, vector<uint32_t>(2048));
  for (auto& W : W_all)
    for (auto& v : W) v = rng();
  cerr << "[mb] x=0x" << hex << x << dec << endl;

  SoftMCPlatform platform(bender_id);
  if (platform.init() != SOFTMC_SUCCESS) {
    cerr << "[mb] platform init failed" << endl;
    return 3;
  }
  platform.reset_fpga();

  using clk = std::chrono::steady_clock;
  using ns  = std::chrono::nanoseconds;
  long long t_w_ns = 0, t_m_ns = 0;
  int total_pass = 0;
  int total_outputs = 4 * n_batches;

  for (int batch = 0; batch < n_batches; batch++) {
    // 1. Preload 4 weights to 4 banks' Rfirst rows (sequential per bank).
    auto t0 = clk::now();
    for (int b = 0; b < 4; b++) {
      Calib& c = by_bank[b][batch];
      per_column_write_row(platform, b, c.Rfirst,
                           W_all[batch * 4 + b].data());
    }
    auto t1 = clk::now();

    // 2. ONE program with 4 MAJ3 bodies (one per bank, sequential).
    Program p;
    p.add_inst(SMC_LI(8, CASR));
    for (int b = 0; b < 4; b++) {
      Calib& c = by_bank[b][batch];
      emit_bank_maj3_body(p, b, c.Rfirst, c.Rsecond,
                          c.open_rows.data(), x,
                          /*label_base=*/batch * 10000 + b * 1000);
    }
    p.add_inst(SMC_END());
    if (batch == 0)
      cerr << "[mb] 4-bank program size: " << p.size()
           << " bytes (" << (p.size() / 8) << " insts)" << endl;
    platform.execute(p);

    // 3. Receive 4 rows in order (bank 0, 1, 2, 3).
    int batch_pass = 0;
    for (int b = 0; b < 4; b++) {
      Calib& c = by_bank[b][batch];
      uint8_t row_buf[8192];
      int rc = platform.receiveData(row_buf, 8192);
      if (rc != 8192) {
        cerr << "[mb] batch=" << batch << " b=" << b
             << " receiveData rc=" << rc << endl;
        return 4;
      }
      // Verify
      const auto& W = W_all[batch * 4 + b];
      int byte_mm = 0, seg_mm = 0;
      long fpga_pop = 0, host_pop = 0;
      for (int s = 0; s < 2048; s++) {
        uint32_t actual = (uint32_t)row_buf[s*4]
                        | ((uint32_t)row_buf[s*4+1] << 8)
                        | ((uint32_t)row_buf[s*4+2] << 16)
                        | ((uint32_t)row_buf[s*4+3] << 24);
        uint32_t expected = W[s] & x;
        if (actual != expected) seg_mm++;
        fpga_pop += __builtin_popcount(actual);
        host_pop += __builtin_popcount(expected);
      }
      for (int i = 0; i < 8192; i += 4) {
        uint32_t exp32 = W[i/4] & x;
        if (row_buf[i+0] != (uint8_t)(exp32      & 0xFF)) byte_mm++;
        if (row_buf[i+1] != (uint8_t)((exp32>>8) & 0xFF)) byte_mm++;
        if (row_buf[i+2] != (uint8_t)((exp32>>16)& 0xFF)) byte_mm++;
        if (row_buf[i+3] != (uint8_t)((exp32>>24)& 0xFF)) byte_mm++;
      }
      bool pass = (byte_mm == 0 && fpga_pop == host_pop);
      if (pass) { batch_pass++; total_pass++; }
      else {
        cerr << "[mb] FAIL batch=" << batch << " bank=" << b
             << " s_id=" << c.s_id
             << " byte_mm=" << byte_mm << " seg_mm=" << seg_mm
             << " host_pop=" << host_pop << " fpga_pop=" << fpga_pop << endl;
      }
    }
    auto t2 = clk::now();
    t_w_ns += std::chrono::duration_cast<ns>(t1 - t0).count();
    t_m_ns += std::chrono::duration_cast<ns>(t2 - t1).count();
    cerr << "[mb] batch=" << batch << " banks_passed=" << batch_pass
         << "/4" << endl;
  }

  double avg_w_ms = t_w_ns / 1e6 / n_batches;
  double avg_m_ms = t_m_ns / 1e6 / n_batches;
  double per_maj3_ms = (avg_w_ms + avg_m_ms) / 4.0;
  cerr << "[mb] timing per BATCH (4 MAJ3): preload=" << avg_w_ms
       << " ms  exec+recv=" << avg_m_ms << " ms  "
       << "total=" << (avg_w_ms + avg_m_ms) << " ms" << endl;
  cerr << "[mb] per-MAJ3 amortized: " << per_maj3_ms << " ms" << endl;

  bool all_ok = (total_pass == total_outputs);
  cerr << "[mb] DONE — " << total_pass << "/" << total_outputs
       << " " << (all_ok ? "ALL_PASS" : "FAIL") << endl;
  _exit(all_ok ? 0 : 6);
}
