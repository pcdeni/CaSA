// Sub-lattice broadcast experiment (2026-07, tower).
//
// HYPOTHESIS (from the SiMRA exchange, memory lattice_as_primitive_implications):
//   A doubleACT between TWO MEMBERS of a calibrated open-rows tuple, at
//   k-generator distance, deposits the sensed-first row's data into exactly
//   the 2^k sub-coset {X (+) S : S subseteq bits(X^Y)} — a TARGETED broadcast,
//   not the indiscriminate all-16 broadcast. If true, staging x into one tuple
//   row + one intra-lattice doubleACT replaces the 5 activation wrRows that
//   dominate per-MAJ cost.
//
// This tool, for a calibrated tuple (Rfirst = base X, 16 open rows):
//   For each other open row as PARTNER Y:
//     re-init (W to Rfirst, 0 to other 15) -> doubleACT(t12,t23, Rfirst, Y)
//     -> read all 16 -> classify each open row as IN-coset (predicted to hold
//     W) or OUT-coset (predicted to stay 0), using LOCAL coords (row-SUB_START):
//        member M in-coset(Y)  iff  ((localM ^ localX) & ~(localY ^ localX)) == 0
//     Report deposit rate on IN rows and leak rate on OUT rows.
//   Runs REPEATS per config (determinism) and a TIMING sweep on the money pair.
//
// Selectivity is the load-bearing metric: a config is USEFUL only if IN rows
// hold W AND OUT rows stay 0. All-16 leak = useless; clean partition = the win.
//
// Argv:  ./sublattice-bcast <bender> <calib_file> <bank> <s_id> <sub_start> [seed]
#include "instruction.h"
#include "prog.h"
#include "platform.h"
#include "../util.h"

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

struct Calib {
  int s_id, bank;
  uint32_t Rfirst, Rsecond;
  vector<uint32_t> open_rows;
};

static vector<Calib> read_calib(const string& path, int wanted_bank,
                                int wanted_sid) {
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
    if (c.bank == wanted_bank && c.s_id == wanted_sid) out.push_back(c);
  }
  return out;
}

static int BANK = 0;
static uint32_t SUB_START = 0;

// Write W to Rfirst, uniform 0 to the other 15 open rows.
static void reinit(SoftMCPlatform& pf, const Calib& c, const vector<uint32_t>& W) {
  int col_start = 0;
  for (int chunk = 0; chunk < 3; chunk++) {
    int n_cols = CHUNK_COLS[chunk];
    Program p = build_chunk_program(BANK, c.Rfirst, W.data() + col_start * 16,
                                    col_start, n_cols);
    pf.execute(p);
    col_start += n_cols;
  }
  Program p;
  p.add_inst(SMC_LI(8, CASR));
  p.add_inst(SMC_LI(BANK, BAR));
  p.add_inst(SMC_LI(128, NUM_COLS_REG));
  p.add_below(PRE(BAR, 0, 0));
  int lbl = 0;
  for (uint32_t r : c.open_rows) {
    if (r == c.Rfirst) continue;
    p.add_below(wrRow_immediate_label(BAR, r, 0u, lbl++));
  }
  p.add_inst(SMC_END());
  pf.execute(p);
}

// doubleACT(t12,t23, Rfirst, partner) then read back all 16 open rows.
static void fire_and_read(SoftMCPlatform& pf, const Calib& c, uint32_t partner,
                          int t12, int t23, vector<vector<uint8_t>>& rows_back) {
  Program p;
  p.add_inst(SMC_LI(8, CASR));
  p.add_inst(SMC_LI(BANK, BAR));
  p.add_inst(SMC_LI(128, NUM_COLS_REG));
  p.add_below(PRE(BAR, 0, 0));
  p.add_inst(SMC_SLEEP(6));
  p.add_below(doubleACT(t12, t23, c.Rfirst, partner));
  p.add_inst(SMC_SLEEP(6));
  p.add_below(PRE(BAR, 0, 0));
  p.add_inst(SMC_SLEEP(6));
  for (int i = 0; i < 16; i++)
    p.add_below(rdRow_immediate_label(BAR, c.open_rows[i], i));
  p.add_inst(SMC_END());
  pf.execute(p);
  for (int i = 0; i < 16; i++)
    pf.receiveData(rows_back[i].data(), 8192);
}

int main(int argc, char** argv) {
  if (argc < 6) {
    cerr << "Usage: " << argv[0]
         << " <bender> <calib_file> <bank> <s_id> <sub_start> [seed]" << endl;
    return 1;
  }
  int bender    = atoi(argv[1]);
  string calibp = argv[2];
  BANK          = atoi(argv[3]);
  int sid       = atoi(argv[4]);
  SUB_START     = (uint32_t)strtoul(argv[5], nullptr, 10);
  unsigned seed = (argc > 6) ? (unsigned)atoi(argv[6]) : 0xC0FFEE;

  vector<Calib> cs = read_calib(calibp, BANK, sid);
  if (cs.empty()) { cerr << "[sub] no s_id=" << sid << " bank=" << BANK << endl; return 2; }
  Calib c = cs[0];
  uint32_t localX = c.Rfirst - SUB_START;
  printf("# tuple s_id=%d bank=%d Rfirst=%u (localX=%u) sub_start=%u\n",
         c.s_id, BANK, c.Rfirst, localX, SUB_START);
  printf("# open_rows(local):");
  for (uint32_t r : c.open_rows) printf(" %u", r - SUB_START);
  printf("\n");

  std::mt19937 rng(seed);
  vector<uint32_t> W(2048);
  for (auto& v : W) v = rng();
  uint8_t exp[8192];
  for (int s = 0; s < 2048; s++)
    for (int b = 0; b < 4; b++) exp[s*4+b] = (uint8_t)((W[s] >> (8*b)) & 0xFF);

  SoftMCPlatform pf(bender);
  if (pf.init() != SOFTMC_SUCCESS) { cerr << "[sub] init failed\n"; return 3; }
  pf.reset_fpga();
  vector<vector<uint8_t>> rb(16, vector<uint8_t>(8192));

  // Classify: does open row i match W ("W"), stay 0 ("0"), or mixed ("~")?
  auto classify = [&](int i) -> char {
    int mW = 0, mZ = 0;
    for (int b = 0; b < 8192; b++) {
      if (rb[i][b] == exp[b]) mW++;
      if (rb[i][b] == 0)      mZ++;
    }
    if (mW == 8192) return 'W';
    if (mZ == 8192) return '0';
    if (mW > 8192 - 64) return 'w';   // near-W (a few flaky cells)
    if (mZ > 8192 - 64) return '.';   // near-0
    return '~';                        // genuinely mixed
  };

  auto run_cfg = [&](uint32_t partner, int t12, int t23, int rep) {
    reinit(pf, c, W);
    fire_and_read(pf, c, partner, t12, t23, rb);
    uint32_t g = (partner - SUB_START) ^ localX;
    // predicted membership per open row
    string got, pred;
    int in_hold = 0, in_tot = 0, out_leak = 0, out_tot = 0;
    for (int i = 0; i < 16; i++) {
      uint32_t d = (c.open_rows[i] - SUB_START) ^ localX;
      bool in_coset = ((d & ~g) == 0);
      char cl = classify(i);
      got += cl;
      pred += in_coset ? 'W' : '0';
      if (in_coset) { in_tot++; if (cl=='W'||cl=='w') in_hold++; }
      else          { out_tot++; if (cl!='0'&&cl!='.') out_leak++; }
    }
    int k = __builtin_popcount(g & 0b11) + ((g & 96)?1:0) + ((g & 384)?1:0);
    printf("g=%-4u k=%d partner=%u t=(%d,%d) rep=%d  IN_hold=%d/%d OUT_leak=%d/%d  pred=%s got=%s\n",
           g, k, partner, t12, t23, rep, in_hold, in_tot, out_leak, out_tot,
           pred.c_str(), got.c_str());
    return (in_hold==in_tot && out_leak==0);
  };

  // Phase A: every partner at broadcast timing (10,2), 3 repeats.
  printf("\n## Phase A — all partners, t=(10,2), 3 repeats\n");
  for (uint32_t Y : c.open_rows) {
    if (Y == c.Rfirst) continue;
    for (int r = 0; r < 3; r++) run_cfg(Y, 10, 2, r);
  }

  // Phase B: money pair (g=3, the 4-row mat-group) across timings.
  printf("\n## Phase B — money pair g=3 timing sweep\n");
  uint32_t money = SUB_START + (localX ^ 3u);
  bool is_member = false;
  for (uint32_t r : c.open_rows) if (r == money) is_member = true;
  if (is_member)
    for (int t12 : {0, 10, 30})
      for (int t23 : {0, 1, 2})
        for (int r = 0; r < 3; r++) run_cfg(money, t12, t23, r);
  else
    printf("# g=3 partner %u not a tuple member; skipping\n", money);

  // Phase C: external-leak probe. After the money-pair doubleACT (g=3, opens
  // only the {1,2} sub-coset), read rows OUTSIDE the tuple — the partial-bit
  // neighbors of the 96={32,64} and 384={128,256} groups, plus a few far rows.
  // None should hold W: proves (a) no pollution beyond the tuple, (b) group
  // generators stay atomic under a sub-pair (no partial-bit opening).
  printf("\n## Phase C — external-leak probe after money-pair (g=3) doubleACT t=(10,2)\n");
  if (is_member) {
    // external candidate rows (global) via local offsets that are NOT tuple members
    uint32_t offs[] = {32, 64, 128, 256, 96+256, 8, 512, 620};
    // write W to Rfirst, 0 to the 15 tuple rows, AND 0 to the external probes
    reinit(pf, c, W);
    {
      Program p;
      p.add_inst(SMC_LI(8, CASR));
      p.add_inst(SMC_LI(BANK, BAR));
      p.add_inst(SMC_LI(128, NUM_COLS_REG));
      p.add_below(PRE(BAR, 0, 0));
      int lbl = 100;
      for (uint32_t o : offs)
        p.add_below(wrRow_immediate_label(BAR, SUB_START + (localX ^ o), 0u, lbl++));
      p.add_inst(SMC_END());
      pf.execute(p);
    }
    // fire money-pair doubleACT, then read the external rows
    {
      Program p;
      p.add_inst(SMC_LI(8, CASR));
      p.add_inst(SMC_LI(BANK, BAR));
      p.add_inst(SMC_LI(128, NUM_COLS_REG));
      p.add_below(PRE(BAR, 0, 0));
      p.add_inst(SMC_SLEEP(6));
      p.add_below(doubleACT(10, 2, c.Rfirst, money));
      p.add_inst(SMC_SLEEP(6));
      p.add_below(PRE(BAR, 0, 0));
      p.add_inst(SMC_SLEEP(6));
      for (size_t j = 0; j < sizeof(offs)/sizeof(offs[0]); j++)
        p.add_below(rdRow_immediate_label(BAR, SUB_START + (localX ^ offs[j]), (int)j));
      p.add_inst(SMC_END());
      pf.execute(p);
      for (size_t j = 0; j < sizeof(offs)/sizeof(offs[0]); j++)
        pf.receiveData(rb[j].data(), 8192);
    }
    int any_leak = 0;
    for (size_t j = 0; j < sizeof(offs)/sizeof(offs[0]); j++) {
      char cl = classify((int)j);
      bool leak = (cl != '0' && cl != '.');
      if (leak) any_leak++;
      printf("  ext local_off=%-4u row=%u  class=%c %s\n",
             offs[j], SUB_START + (localX ^ offs[j]), cl, leak ? "<-- LEAK" : "clean");
    }
    printf("  external leak: %d/%zu rows\n", any_leak, sizeof(offs)/sizeof(offs[0]));
  }

  printf("\n[sub] DONE\n");
  return 0;
}
