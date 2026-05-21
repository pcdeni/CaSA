// BitNet PIM SW demo — Phase 1a: K MAJ3 ops batched into one
// platform.execute(), proving (a) per-program PCIe overhead amortizes
// over K trials and (b) we can read back K result rows in order.
//
// Each MAJ3 trial uses a different calibrated (s_id, Rfirst, Rsecond,
// open_rows[16]) tuple from calib_dimm0.txt. All K ops compute
// AND(W[k], x) = MAJ3(W[k], x, 0) for one shared activation x and K
// distinct weight patterns W[0..K-1] — so the K results form a
// "binary MatVec" y[k] = popcount(W[k] AND x).
//
// Limits of this stage (deferred to later phases):
//   - Uniform 32-bit pattern per row (wrRow_immediate). Real BitNet
//     wants 8192 arbitrary bits per row → Phase 1.5: per-column WDATA
//     reload via SMC_LDWD between SMC_WRITEs.
//   - Re-writes the activation pattern x into 5 of the 16 open rows
//     each MAJ3 (since doubleACT is destructive). Phase 2 will keep
//     a backup x row outside the open set and Multi-RowCopy it in.
//   - Single bank, sequential MAJ3s. Phase 3 will spread across banks
//     for tFAW/tRRD-bounded parallelism.
//
// Argv:
//   ./matvec-smoke-exe <bender_id> <calib_file> <bank_id> <K> [seed]
//
// calib_file format (one tuple per line, # comments ignored):
//   s_id bank Rfirst Rsecond open0 open1 ... open15
//
// Verification: each result row must be byte-exact (W[k] & x) repeated
// 2048×, AND popcount(row)/2048 must equal host popcount(W[k] & x).
// Exit 0 iff all K trials pass both checks.
#include "instruction.h"
#include "prog.h"
#include "platform.h"
#include "../util.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

using namespace std;

#define NUM_BANKS_HERE 1
#define NUM_ROWS_DEF 2048
#define LOOP_ITER 15
#define ITER_REG 9

// ----- SoftMC builders mirrored from MajOperations test.cpp / BitNet
// test_maj3_smoke.cpp (calibration regime fidelity) -----

static Program frac_builder(int t_frac, int r_frac_addr) {
  Program p;
  p.add_inst(all_nops());
  int R_FRAC_REG = RF_REG;
  int bank_reg = BAR;
  p.add_inst(SMC_LI(r_frac_addr, R_FRAC_REG));
  int num_cmd = 2 + t_frac;
  num_cmd += 4 - (num_cmd % 4);
  Mininst q_inst[num_cmd];
  for (int i = 0; i < num_cmd; i++) q_inst[i] = SMC_NOP();
  q_inst[0]          = SMC_ACT(bank_reg, 0, R_FRAC_REG, 0);
  q_inst[t_frac + 1] = SMC_PRE(bank_reg, 0, 0);
  for (int i = 0; i < num_cmd; i += 4)
    p.add_inst(q_inst[i], q_inst[i + 1], q_inst[i + 2], q_inst[i + 3]);
  return p;
}

static Program _init(uint32_t bank_id, uint32_t num_iter) {
  Program p;
  p.add_inst(SMC_LI(NUM_ROWS_DEF, NUM_ROWS_REG));
  p.add_inst(SMC_LI(NUM_BANKS_HERE, NUM_BANKS_REG));
  p.add_inst(SMC_LI(num_iter, ITER_REG));
  p.add_inst(SMC_LI(8, CASR));
  p.add_inst(SMC_LI(1, BASR));
  p.add_inst(SMC_LI(1, RASR));
  p.add_inst(SMC_LI(128, NUM_COLS_REG));
  p.add_inst(SMC_LI(bank_id, BAR));
  return p;
}

// Pattern-shared row write. Loads PATTERN_REG once, writes N rows of
// the same pattern. Saves the per-row 17-instruction PATTERN_REG +
// LDWD setup that wrRow_immediate_label re-emits each call.
//
// NUM_COLS_REG must already be set to 128 (handled in _init).
// Per-row cost: 1+1+1+2+2+(2+1)+1+1 = 12 insts (vs 30 for a fresh
// wrRow_immediate_label call).
static void emit_pattern_setup(Program& program, uint32_t pattern) {
  program.add_inst(SMC_LI(pattern, PATTERN_REG));
  for (int i = 0; i < 16; i++)
    program.add_inst(SMC_LDWD(PATTERN_REG, i));
}

static void emit_write_row_with_loaded_pattern(Program& program,
                                                uint32_t row_immd,
                                                int label) {
  program.add_inst(SMC_LI(row_immd, RAR));
  program.add_inst(SMC_LI(0, CAR));
  program.add_inst(SMC_LI(0, LOOP_COLS));
  program.add_below(PRE(BAR, 0, 0));
  program.add_below(ACT(BAR, 0, RAR, 0));
  string lbl = "WR_FAST_" + std::to_string(label);
  program.add_label(lbl);
    program.add_below(WRITE(BAR, CAR, 1));
    program.add_inst(SMC_ADDI(LOOP_COLS, 1, LOOP_COLS));
  program.add_branch(Program::BR_TYPE::BL, LOOP_COLS, NUM_COLS_REG, lbl);
  program.add_inst(all_nops());
}

// One MAJ3 trial body — wrap in its own ITER_LOOP (num_iter=1) so the
// emitted instruction sequence matches the calibration regime. Each
// trial's loop label is suffixed with the trial index k for uniqueness.
//
// Why the iter-loop wrapper: the calibrated MAJ3 regime + Phase 0 smoke
// test both ran with this LOOP_ITER + SMC_ADDI + branch wrapper around
// the body. Removing it caused hangs on K≥2 (likely a SoftMC FPGA
// state-machine quirk; not investigated further since the wrapper is
// only ~3 inst).
//
// Why pattern-grouped wrRows: out of 16 open rows, only 4 unique
// patterns (ONE for the frac slot + W + x + 0 for the MAJ3 inputs).
// Loading PATTERN_REG + 16 LDWDs once per pattern instead of 16 times
// saves ~250 insts/trial, lifting the in-buffer K limit from 3 to ~6.
static void emit_maj3_trial(Program& program,
                            int trial_index_k,
                            const vector<uint32_t>& patterns,      // 16
                            const vector<uint32_t>& open_row_idx,  // 16
                            uint32_t Rfirst, uint32_t Rsecond,
                            uint32_t t_12, uint32_t t_23,
                            uint32_t n_frac_times, uint32_t t_frac) {
  string lbl = "ITER_LOOP_" + std::to_string(trial_index_k);
  program.add_inst(SMC_LI(0, LOOP_ITER));
  program.add_label(lbl);
    program.add_below(PRE(BAR, 0, 0));
    // Group rows by pattern so we load PATTERN_REG / LDWD-16 once per
    // unique pattern instead of once per row.
    std::map<uint32_t, vector<uint32_t>> by_pattern;
    for (size_t i = 0; i < open_row_idx.size(); i++)
      by_pattern[patterns[i]].push_back(open_row_idx[i]);
    int row_label = trial_index_k * 64;  // unique label seed per trial
    for (auto& kv : by_pattern) {
      emit_pattern_setup(program, kv.first);
      for (uint32_t r : kv.second)
        emit_write_row_with_loaded_pattern(program, r, row_label++);
    }
    program.add_inst(SMC_SLEEP(6));
    program.add_below(PRE(BAR, 0, 0));
    program.add_inst(SMC_SLEEP(6));
    for (uint32_t j = 0; j < n_frac_times; j++) {
      program.add_inst(SMC_SLEEP(6));
      program.add_below(frac_builder(t_frac, open_row_idx[0]));
      program.add_inst(SMC_SLEEP(6));
    }
    program.add_inst(SMC_SLEEP(6));
    program.add_below(PRE(BAR, 0, 0));
    program.add_inst(SMC_SLEEP(6));
    program.add_below(doubleACT(t_12, t_23, Rfirst, Rsecond));
    program.add_inst(SMC_SLEEP(6));
    program.add_below(PRE(BAR, 0, 0));
    program.add_inst(SMC_SLEEP(6));
    // rdRow_immediate uses a HARDCODED label "READ_ROW_IMMD" — using
    // _label variant for multi-trial programs (see softmc_label_collision
    // memory).
    program.add_below(rdRow_immediate_label(BAR, open_row_idx[0],
                                            trial_index_k));
    program.add_inst(all_nops());
    program.add_inst(all_nops());
    program.add_below(PRE(BAR, 0, 0));
    program.add_inst(all_nops());
    program.add_inst(all_nops());
    program.add_inst(SMC_ADDI(LOOP_ITER, 1, LOOP_ITER));
  program.add_branch(Program::BR_TYPE::BL, LOOP_ITER, ITER_REG, lbl);
}

static vector<uint32_t> build_16row_pattern(uint32_t A, uint32_t B, uint32_t C) {
  vector<uint32_t> lst(16, 0);
  uint32_t base[3] = {A, B, C};
  for (int r = 0; r < 5; r++)
    for (int i = 0; i < 3; i++)
      lst[r * 3 + i] = base[i];
  lst[15] = lst[0];
  lst[0]  = ONE;
  return lst;
}

static uint32_t maj3(uint32_t a, uint32_t b, uint32_t c) {
  return (a & b) | (a & c) | (b & c);
}

// AVX2-accelerated total bit-popcount over 8192 bytes.
static long popcount_row(const uint8_t* row) {
  long n = 0;
  for (int i = 0; i < 8192; i += 8) {
    uint64_t w;
    memcpy(&w, row + i, 8);
    n += __builtin_popcountll(w);
  }
  return n;
}

// ----- calibration parser -----

struct Calib {
  int s_id, bank;
  uint32_t Rfirst, Rsecond;
  vector<uint32_t> open_rows;  // size 16
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

// ----- main -----

int main(int argc, char** argv) {
  if (argc < 5 || argc > 7) {
    cerr << "Usage: " << argv[0]
         << " <bender_id> <calib_file> <bank_id> <K> [repeat=1] [seed]" << endl;
    return 1;
  }
  int bender_id  = atoi(argv[1]);
  string calib_p = argv[2];
  int bank_id    = atoi(argv[3]);
  int K          = atoi(argv[4]);
  int repeat     = (argc >= 6) ? atoi(argv[5]) : 1;
  unsigned seed  = (argc >= 7) ? (unsigned)atoi(argv[6]) : 0xC0FFEE;
  if (repeat < 1) repeat = 1;

  vector<Calib> calib = read_calib(calib_p, bank_id);
  int total_outputs = K * repeat;
  if ((int)calib.size() < total_outputs) {
    cerr << "[matvec] calib file has only " << calib.size()
         << " tuples for bank " << bank_id
         << " — requested K*repeat=" << total_outputs << endl;
    return 2;
  }
  cerr << "[matvec] bank=" << bank_id << " K=" << K
       << " repeat=" << repeat << " (total_outputs=" << total_outputs
       << " of " << calib.size() << " calibrated)" << endl;

  // Generate inputs: one shared activation x, total_outputs distinct weight rows.
  srand(seed);
  uint32_t x = (uint32_t)rand() ^ ((uint32_t)rand() << 16);
  vector<uint32_t> W(total_outputs);
  vector<uint32_t> expected_and(total_outputs);
  vector<int> host_popcount(total_outputs);
  for (int k = 0; k < total_outputs; k++) {
    W[k] = (uint32_t)rand() ^ ((uint32_t)rand() << 16);
    expected_and[k] = W[k] & x;
    host_popcount[k] = __builtin_popcount(expected_and[k]);
  }
  cerr << "[matvec] x=0x" << hex << x << dec << endl;

  SoftMCPlatform platform(bender_id);
  if (platform.init() != SOFTMC_SUCCESS) {
    cerr << "[matvec] platform init failed" << endl;
    return 3;
  }
  platform.reset_fpga();

  vector<int> fpga_y(total_outputs, -1);
  int total_byte_pass = 0, total_pop_pass = 0;

  // Per-execute timing accumulator.
  using clk = std::chrono::steady_clock;
  using ns  = std::chrono::nanoseconds;
  long long t_build_ns = 0, t_exec_ns = 0, t_recv_ns = 0;

  for (int rep = 0; rep < repeat; rep++) {
    int base_k = rep * K;
    auto t0 = clk::now();
    srand(0xDEADBEEF + rep);
    Program prog;
    prog.add_below(_init(bank_id, /*num_iter=*/1));
    prog.add_inst(all_nops());
    prog.add_inst(all_nops());
    prog.add_below(PRE(BAR, 0, 0));
    for (int k = 0; k < K; k++) {
      int gk = base_k + k;
      vector<uint32_t> patterns = build_16row_pattern(W[gk], x, /*C=*/0u);
      // Env-overridable MAJ3 timing (default 0,0 = DIMM 0 production).
      static int mv_t_12 = []{ const char* v = getenv("PIM_T12"); return v?atoi(v):0; }();
      static int mv_t_23 = []{ const char* v = getenv("PIM_T23"); return v?atoi(v):0; }();
      emit_maj3_trial(prog, /*trial_index_k=*/k,
                      patterns, calib[gk].open_rows,
                      calib[gk].Rfirst, calib[gk].Rsecond,
                      /*t_12=*/mv_t_12, /*t_23=*/mv_t_23,
                      /*n_frac_times=*/3, /*t_frac=*/0);
    }
    prog.add_inst(SMC_END());
    if (rep == 0) {
      cerr << "[matvec] program size per execute: " << prog.size()
           << " bytes (" << (prog.size() / 8) << " insts)" << endl;
    }
    auto t1 = clk::now();
    platform.execute(prog);
    auto t2 = clk::now();

    for (int k = 0; k < K; k++) {
      int gk = base_k + k;
      uint8_t row[8192];
      int rc = platform.receiveData(row, 8192);
      if (rc != 8192) {
        cerr << "[matvec] rep=" << rep << " k=" << k
             << " receiveData rc=" << rc << endl;
        return 4;
      }
      uint8_t exp[8192];
      for (int i = 0; i < 8192; i += 4) {
        exp[i + 0] = (uint8_t)((expected_and[gk] >>  0) & 0xFFu);
        exp[i + 1] = (uint8_t)((expected_and[gk] >>  8) & 0xFFu);
        exp[i + 2] = (uint8_t)((expected_and[gk] >> 16) & 0xFFu);
        exp[i + 3] = (uint8_t)((expected_and[gk] >> 24) & 0xFFu);
      }
      int byte_mismatches = 0;
      for (int i = 0; i < 8192; i++)
        if (row[i] != exp[i]) byte_mismatches++;
      long row_pop = popcount_row(row);
      int derived_y = (int)(row_pop / 2048);
      fpga_y[gk] = derived_y;
      bool byte_pass = (byte_mismatches == 0);
      bool pop_pass  = (derived_y == host_popcount[gk]
                        && row_pop == 2048L * host_popcount[gk]);
      if (byte_pass) total_byte_pass++;
      if (pop_pass)  total_pop_pass++;
      if (!byte_pass || !pop_pass) {
        cerr << "[matvec] FAIL rep=" << rep << " k=" << k << " gk=" << gk
             << " W=0x" << hex << W[gk] << " x=0x" << x
             << " expected_and=0x" << expected_and[gk] << dec
             << " | byte_mismatches=" << byte_mismatches << "/8192"
             << " | row_pop=" << row_pop
             << " (y_fpga=" << derived_y << " y_host=" << host_popcount[gk] << ")"
             << endl;
      }
    }
    auto t3 = clk::now();
    t_build_ns += std::chrono::duration_cast<ns>(t1 - t0).count();
    t_exec_ns  += std::chrono::duration_cast<ns>(t2 - t1).count();
    t_recv_ns  += std::chrono::duration_cast<ns>(t3 - t2).count();
  }

  cerr << "[matvec] bank=" << bank_id
       << " K=" << K << " repeat=" << repeat
       << " total_outputs=" << total_outputs
       << "  byte_exact " << total_byte_pass << "/" << total_outputs
       << "  popcount_exact " << total_pop_pass << "/" << total_outputs
       << endl;
  if (total_outputs <= 32) {
    cerr << "[matvec] FPGA y:";
    for (int gk = 0; gk < total_outputs; gk++) cerr << " " << fpga_y[gk];
    cerr << endl;
    cerr << "[matvec] HOST y:";
    for (int gk = 0; gk < total_outputs; gk++) cerr << " " << host_popcount[gk];
    cerr << endl;
  }
  // Timing: aggregate vs per-execute vs per-MAJ3
  double ms_per_exec = (double)t_exec_ns / 1e6 / repeat;
  double ms_per_recv = (double)t_recv_ns / 1e6 / repeat;
  double ms_per_build = (double)t_build_ns / 1e6 / repeat;
  double us_per_maj3  = (double)(t_exec_ns + t_recv_ns) / 1e3 / total_outputs;
  cerr << "[matvec] timing: build=" << ms_per_build << " ms/exec  "
       << "execute=" << ms_per_exec << " ms/exec  "
       << "recv=" << ms_per_recv << " ms/exec  "
       << "→ " << us_per_maj3 << " us/MAJ3" << endl;

  bool all_ok = (total_byte_pass == total_outputs && total_pop_pass == total_outputs);
  cerr << "[matvec] DONE — " << (all_ok ? "ALL_PASS" : "FAIL") << endl;
  _exit(all_ok ? 0 : 6);
}
