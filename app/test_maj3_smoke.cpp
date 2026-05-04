// BitNet PIM SW demo — Phase 0: prove software-controlled MAJ3 on DIMM 0.
//
// Mirrors MajOperations test.cpp's program structure (PRE, wrRow×N, PRE,
// frac×n, PRE, doubleACT, PRE, rdRow) so we run the SAME instruction sequence
// the calibration sweep did. The calibration produced full_stable_cells == 100
// at (t_12=0, t_23=0) for several (s_id, bank, Rfirst, Rsecond) tuples — we
// re-use those tuples here and inject our own 32-bit input patterns instead
// of the maj_all_1_0 enumeration patterns.
//
// What this proves:
//   1. We can software-control a MAJ3 op end-to-end via SoftMCPlatform.
//   2. The 16-open-row layout (5 copies each of A/B/C + 1 fractional row)
//      gives bit-exact MAJ3(A, B, C) at every bit position of the result.
//   3. AND(W, x) = MAJ3(W, x, 0) and OR(W, x) = MAJ3(W, x, 1) work on real
//      silicon, which is the BitNet-relevant primitive.
//
// What it does NOT cover (Phase 1+):
//   - Multiple MAJ3 ops in one program (PCIe overhead amortization)
//   - Per-column non-uniform row contents (real BitNet weights need this)
//   - Activation broadcast via Multi-RowCopy
//   - Host-side popcount accumulation
//
// Argv:
//   ./maj3-smoke-exe <bender_id> <samples_file> <bank_id> <s_id>
//
// Output: stdout/stderr only (no CSV). PASS/FAIL per test case + summary.
// Exit code: 0 = all pass, non-zero = at least one fail.
#include "instruction.h"
#include "prog.h"
#include "platform.h"
#include "../util.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

using namespace std;

#define NUM_BANKS_HERE 1
#define NUM_ROWS_DEF 2048

#define LOOP_ITER 15
#define ITER_REG 9

// ----- SoftMC builders kept identical to MajOperations test.cpp so we
// run the exact instruction sequence the calibration was performed under.

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

static Program test_prog(uint32_t bank_id,
                         const vector<uint32_t>& patterns,        // 16 entries
                         const vector<uint32_t>& open_row_idx,    // 16 entries
                         uint32_t r_frac_addr,
                         uint32_t Rfirst, uint32_t Rsecond,
                         uint32_t t_12, uint32_t t_23,
                         uint32_t n_frac_times, uint32_t t_frac,
                         uint32_t num_iter) {
  Program program;
  program.add_below(_init(bank_id, num_iter));
  program.add_inst(all_nops());
  program.add_inst(all_nops());
  program.add_below(PRE(BAR, 0, 0));
  program.add_inst(SMC_LI(0, LOOP_ITER));
  program.add_label("ITER_LOOP");
    program.add_below(PRE(BAR, 0, 0));
    for (size_t i = 0; i < open_row_idx.size(); i++) {
      int label = rand();
      program.add_below(wrRow_immediate_label(BAR, open_row_idx[i],
                                              patterns[i], label));
    }
    program.add_inst(SMC_SLEEP(6));
    program.add_below(PRE(BAR, 0, 0));
    program.add_inst(SMC_SLEEP(6));
    for (uint32_t j = 0; j < n_frac_times; j++) {
      program.add_inst(SMC_SLEEP(6));
      program.add_below(frac_builder(t_frac, r_frac_addr));
      program.add_inst(SMC_SLEEP(6));
    }
    program.add_inst(SMC_SLEEP(6));
    program.add_below(PRE(BAR, 0, 0));
    program.add_inst(SMC_SLEEP(6));
    program.add_below(doubleACT(t_12, t_23, Rfirst, Rsecond));
    program.add_inst(SMC_SLEEP(6));
    program.add_below(PRE(BAR, 0, 0));
    program.add_inst(SMC_SLEEP(6));
    program.add_below(rdRow_immediate(BAR, open_row_idx[0]));
    program.add_inst(all_nops());
    program.add_inst(all_nops());
    program.add_below(PRE(BAR, 0, 0));
    program.add_inst(all_nops());
    program.add_inst(all_nops());
    program.add_inst(SMC_ADDI(LOOP_ITER, 1, LOOP_ITER));
  program.add_branch(program.BR_TYPE::BL, LOOP_ITER, ITER_REG, "ITER_LOOP");
  program.add_inst(SMC_END());
  return program;
}

// Build the 16-row pattern array for one (A, B, C) MAJ3 trial. Layout
// follows MajOperations' maj_all_1_0_pattern_creator(n=16, input_size=3,
// r_frac_idx=[0]):
//   - 5 copies of A at positions {3, 6, 9, 12, 15}
//   - 5 copies of B at positions {1, 4, 7, 10, 13}
//   - 5 copies of C at positions {2, 5, 8, 11, 14}
//   - 1 ONE-pattern at position 0 (the fractional row, discharged by
//     n_frac_times ACT-PRE pulses before doubleACT)
// After discharge, the SA majority over the 15 contributing rows is
// MAJ3(A_bit, B_bit, C_bit) at every bit position (5×weight per input).
static vector<uint32_t> build_16row_pattern(uint32_t A, uint32_t B, uint32_t C) {
  vector<uint32_t> lst(16, 0);
  uint32_t base[3] = {A, B, C};  // base[0]=A → goes to slot 15 after swap
  for (int r = 0; r < 5; r++)
    for (int i = 0; i < 3; i++)
      lst[r * 3 + i] = base[i];
  lst[15] = lst[0];   // = A
  lst[0]  = ONE;      // fractional row
  return lst;
}

struct Sample {
  uint32_t Rfirst, Rsecond, n_open_rows, min_row, max_row;
  int s_id;
  vector<uint32_t> open_rows;
};

static vector<Sample> read_samples(const string& path) {
  vector<Sample> out;
  ifstream f(path);
  string line;
  while (getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    istringstream iss(line);
    Sample s;
    if (!(iss >> s.Rfirst >> s.Rsecond >> s.n_open_rows
              >> s.min_row >> s.max_row >> s.s_id)) continue;
    uint32_t v;
    while (iss >> v) s.open_rows.push_back(v);
    if (s.open_rows.size() == s.n_open_rows) out.push_back(s);
  }
  return out;
}

struct TestCase {
  uint32_t A, B, C, expected;
  const char* name;
};

static uint32_t maj3(uint32_t a, uint32_t b, uint32_t c) {
  return (a & b) | (a & c) | (b & c);
}

int main(int argc, char** argv) {
  if (argc != 5) {
    cerr << "Usage: " << argv[0]
         << " <bender_id> <samples_file> <bank_id> <s_id>" << endl;
    return 1;
  }
  int bender_id   = atoi(argv[1]);
  string samples_path = argv[2];
  int bank_id     = atoi(argv[3]);
  int target_sid  = atoi(argv[4]);

  vector<Sample> samples = read_samples(samples_path);
  Sample target;
  bool found = false;
  for (auto& s : samples) {
    if (s.s_id == target_sid && s.n_open_rows == 16) {
      target = s; found = true; break;
    }
  }
  if (!found) {
    cerr << "[smoke] no s_id=" << target_sid
         << " (n_open=16) in " << samples_path << endl;
    return 2;
  }
  cerr << "[smoke] using s_id=" << target_sid
       << " bank=" << bank_id
       << " Rfirst=" << target.Rfirst
       << " Rsecond=" << target.Rsecond
       << " open_rows=[" << target.open_rows[0] << "..."
       << target.open_rows.back() << "]" << endl;

  SoftMCPlatform platform(bender_id);
  if (platform.init() != SOFTMC_SUCCESS) {
    cerr << "[smoke] platform init failed (bender " << bender_id << ")" << endl;
    return 3;
  }
  platform.reset_fpga();

  srand(12345);

  TestCase cases[] = {
    // truth-table exhaustive: lower 8 bits of each 32-bit word cover all
    // 8 (a, b, c) combinations. Result MAJ3 = 0xE8 in lower byte = 0xE8E8E8E8.
    {0xF0F0F0F0u, 0xCCCCCCCCu, 0xAAAAAAAAu, 0xE8E8E8E8u, "truth_table"},
    // AND(W, x) = MAJ3(W, x, 0) — the BitNet-relevant case.
    {0x12345678u, 0x9ABCDEF0u, 0x00000000u, 0x12345670u, "AND(W,x)"},
    // OR(W, x) = MAJ3(W, x, 1).
    {0x12345678u, 0x9ABCDEF0u, 0xFFFFFFFFu, 0x9ABCDEF8u, "OR(W,x)"},
    // MAJ3(x, x, y) = x — idempotency check.
    {0xCAFEF00Du, 0xCAFEF00Du, 0x12345678u, 0xCAFEF00Du, "MAJ3(x,x,y)=x"},
    // MAJ3(x, ~x, y) = y — three-input vote, x cancels.
    {0xAAAAAAAAu, 0x55555555u, 0xDEADBEEFu, 0xDEADBEEFu, "MAJ3(x,~x,y)=y"},
    // Single 1 vs two 0s — minority loses.
    {0x00000000u, 0x00000000u, 0xFFFFFFFFu, 0x00000000u, "minority_loses"},
    // Single 0 vs two 1s — majority wins.
    {0xFFFFFFFFu, 0xFFFFFFFFu, 0x00000000u, 0xFFFFFFFFu, "majority_wins"},
  };
  int n_cases = sizeof(cases) / sizeof(cases[0]);

  // Sanity-check the test data against the host MAJ3 reference.
  for (int t = 0; t < n_cases; t++) {
    uint32_t e = maj3(cases[t].A, cases[t].B, cases[t].C);
    if (e != cases[t].expected) {
      cerr << "[smoke] BUG: test case '" << cases[t].name
           << "' expected mismatch — host_maj3=0x" << hex << e
           << " stored=0x" << cases[t].expected << dec << endl;
      return 4;
    }
  }

  int n_pass = 0;
  for (int t = 0; t < n_cases; t++) {
    TestCase& tc = cases[t];

    vector<uint32_t> patterns = build_16row_pattern(tc.A, tc.B, tc.C);
    Program p = test_prog(bank_id, patterns, target.open_rows,
                          /*r_frac_addr=*/target.open_rows[0],
                          target.Rfirst, target.Rsecond,
                          /*t_12=*/0, /*t_23=*/0,
                          /*n_frac_times=*/3, /*t_frac=*/0,
                          /*num_iter=*/1);
    platform.execute(p);

    uint8_t row[8192];
    int rc = platform.receiveData(row, 8192);
    if (rc != 8192) {
      cerr << "[smoke] case '" << tc.name << "' receiveData rc=" << rc
           << endl;
      return 5;
    }

    // Build expected 8192-byte buffer (4-byte pattern repeated 2048×).
    uint8_t exp[8192];
    for (int i = 0; i < 8192; i += 4) {
      exp[i + 0] = (uint8_t)((tc.expected >>  0) & 0xFFu);
      exp[i + 1] = (uint8_t)((tc.expected >>  8) & 0xFFu);
      exp[i + 2] = (uint8_t)((tc.expected >> 16) & 0xFFu);
      exp[i + 3] = (uint8_t)((tc.expected >> 24) & 0xFFu);
    }

    int byte_mismatches = 0;
    long bit_mismatches = 0;
    for (int i = 0; i < 8192; i++) {
      if (row[i] != exp[i]) {
        byte_mismatches++;
        bit_mismatches += __builtin_popcount((unsigned)(row[i] ^ exp[i]));
      }
    }

    bool pass = (byte_mismatches == 0);
    cerr << "[smoke] " << (pass ? "PASS" : "FAIL")
         << " " << tc.name
         << ": A=0x" << hex << tc.A << " B=0x" << tc.B
         << " C=0x" << tc.C << " expected=0x" << tc.expected
         << dec << "  byte_mismatches=" << byte_mismatches << "/8192"
         << " bit_mismatches=" << bit_mismatches << "/65536" << endl;
    if (!pass) {
      cerr << "        actual[0..15]   ={";
      for (int i = 0; i < 16; i++) {
        char buf[8]; snprintf(buf, sizeof(buf), " 0x%02X", row[i]);
        cerr << buf;
      }
      cerr << "}" << endl;
      cerr << "        expected[0..15] ={";
      for (int i = 0; i < 16; i++) {
        char buf[8]; snprintf(buf, sizeof(buf), " 0x%02X", exp[i]);
        cerr << buf;
      }
      cerr << "}" << endl;
    }
    if (pass) n_pass++;
  }

  cerr << "[smoke] DONE — passed " << n_pass << "/" << n_cases
       << " (bender=" << bender_id << ", bank=" << bank_id
       << ", s_id=" << target_sid << ")" << endl;
  // Match other DSN_AE_APPS apps' destructor-bypass pattern.
  _exit(n_pass == n_cases ? 0 : 6);
}
