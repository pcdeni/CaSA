// Safe-source fast loading experiment (2026-07-17).
//
// June's MVDRAM post-mortem: RowClone INTO a tuple row lands, but disturbs
// coupled neighbor open rows; "non-shadow source selection necessary but NOT
// sufficient"; verdict "blocked deeper than source-row choice". That work used
// the SOURCE-specific vulnerability model. The validated pair-lattice model
// says the deposit set of doubleACT(src,dst) is the coset
//     {src (+) S : S subseteq bits(local(src)^local(dst))}
// so tuple corruption is a function of the PAIR OFFSET d = local(src)^local(dst):
// pick d with no tuple-generator-sum as a subset -> coset ∩ tuple = {dst} only
// -> the load is tuple-clean BY CONSTRUCTION. s72 generators {1,2,96,384}:
// safe d must avoid bit0, bit1, {bit5&bit6} together, {bit7&bit8} together.
//
// Phases:
//  1  safe external RowClone loads: targets across the tuple x safe offsets;
//     predict ONLY the target row receives the payload.
//  2  UNSAFE-offset controls: predict EXACTLY which extra tuple rows corrupt.
//  3  off-anchor intra-tuple mat-group broadcast (g=3 anchored at each of the
//     4 mat-groups, not just Rfirst) — position independence.
//  4  the full fast loader: 4 distinct patterns -> 4 external staging rows ->
//     4 safe RowClones into group anchors -> 4 intra-group broadcasts ->
//     all 16 rows hold their group's pattern. 8 doubleACTs replace 16 loads.
//
// Argv: ./safe-load <bender> <calib_file> <bank> <s_id> <sub_start> [seed]
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
static SoftMCPlatform* PF = nullptr;
static Calib C;

static void write_pattern(uint32_t row, const vector<uint32_t>& P) {
  int col_start = 0;
  for (int chunk = 0; chunk < 3; chunk++) {
    Program p = build_chunk_program(BANK, row, P.data() + col_start * 16,
                                    col_start, CHUNK_COLS[chunk]);
    PF->execute(p);
    col_start += CHUNK_COLS[chunk];
  }
}

static void zero_rows(const vector<uint32_t>& rows) {
  Program p;
  p.add_inst(SMC_LI(8, CASR));
  p.add_inst(SMC_LI(BANK, BAR));
  p.add_inst(SMC_LI(128, NUM_COLS_REG));
  p.add_below(PRE(BAR, 0, 0));
  int lbl = 0;
  for (uint32_t r : rows)
    p.add_below(wrRow_immediate_label(BAR, r, 0u, lbl++));
  p.add_inst(SMC_END());
  PF->execute(p);
}

static void fire(uint32_t rf, uint32_t rs, int t12, int t23) {
  Program p;
  p.add_inst(SMC_LI(8, CASR));
  p.add_inst(SMC_LI(BANK, BAR));
  p.add_below(PRE(BAR, 0, 0));
  p.add_inst(SMC_SLEEP(6));
  p.add_below(doubleACT(t12, t23, rf, rs));
  p.add_inst(SMC_SLEEP(6));
  p.add_below(PRE(BAR, 0, 0));
  p.add_inst(SMC_SLEEP(6));
  p.add_inst(SMC_END());
  PF->execute(p);
}

static void read_tuple(vector<vector<uint8_t>>& rb) {
  Program p;
  p.add_inst(SMC_LI(8, CASR));
  p.add_inst(SMC_LI(BANK, BAR));
  p.add_inst(SMC_LI(128, NUM_COLS_REG));
  p.add_below(PRE(BAR, 0, 0));
  for (int i = 0; i < 16; i++)
    p.add_below(rdRow_immediate_label(BAR, C.open_rows[i], i));
  p.add_inst(SMC_END());
  PF->execute(p);
  for (int i = 0; i < 16; i++)
    PF->receiveData(rb[i].data(), 8192);
}

// expand a 2048-word pattern to its 8192-byte little-endian image
static void expand(const vector<uint32_t>& P, uint8_t* out) {
  for (int s = 0; s < 2048; s++)
    for (int b = 0; b < 4; b++) out[s*4+b] = (uint8_t)((P[s] >> (8*b)) & 0xFF);
}

// classify one read row against a set of named expected images
static char classify(const uint8_t* got, const vector<pair<char,const uint8_t*>>& refs) {
  int z = 0;
  for (int b = 0; b < 8192; b++) if (got[b] == 0) z++;
  if (z == 8192) return '0';
  for (auto& r : refs) {
    int m = 0;
    for (int b = 0; b < 8192; b++) if (got[b] == r.second[b]) m++;
    if (m == 8192) return r.first;
    if (m > 8192 - 64) return (char)tolower(r.first);
  }
  if (z > 8192 - 64) return '.';
  return '~';
}

int main(int argc, char** argv) {
  if (argc < 6) {
    cerr << "Usage: " << argv[0]
         << " <bender> <calib_file> <bank> <s_id> <sub_start> [seed]" << endl;
    return 1;
  }
  int bender    = atoi(argv[1]);
  BANK          = atoi(argv[3]);
  int sid       = atoi(argv[4]);
  SUB_START     = (uint32_t)strtoul(argv[5], nullptr, 10);
  unsigned seed = (argc > 6) ? (unsigned)atoi(argv[6]) : 0x5AFE;

  vector<Calib> cs = read_calib(argv[2], BANK, sid);
  if (cs.empty()) { cerr << "[safe] no calib line\n"; return 2; }
  C = cs[0];
  uint32_t base  = C.open_rows[0];
  uint32_t lbase = base - SUB_START;
  printf("# s_id=%d bank=%d base=%u local=%u\n", C.s_id, BANK, base, lbase);

  std::mt19937 rng(seed);
  vector<vector<uint32_t>> P(5, vector<uint32_t>(2048));
  for (auto& v : P) for (auto& w : v) w = rng();
  vector<vector<uint8_t>> Pi(5, vector<uint8_t>(8192));
  for (int i = 0; i < 5; i++) expand(P[i], Pi[i].data());

  SoftMCPlatform pf(bender);
  if (pf.init() != SOFTMC_SUCCESS) { cerr << "[safe] init failed\n"; return 3; }
  pf.reset_fpga();
  PF = &pf;
  vector<vector<uint8_t>> rb(16, vector<uint8_t>(8192));

  auto coset_hits = [&](uint32_t lsrc, uint32_t d) {
    // predicted tuple hits of the (src,dst) coset
    string pred(16, '0');
    for (int i = 0; i < 16; i++) {
      uint32_t dl = (C.open_rows[i] - SUB_START) ^ lsrc;
      if ((dl & ~d) == 0) pred[i] = 'S';
    }
    return pred;
  };

  // ---- Phase 1: safe external loads ----
  printf("\n## Phase 1 — safe external RowClone loads (t=30,1)\n");
  int tidx[5] = {0, 1, 5, 9, 15};
  uint32_t safe_d[4] = {4, 8, 16, 32};
  for (int ti : tidx) {
    uint32_t T  = C.open_rows[ti];
    uint32_t lT = T - SUB_START;
    for (uint32_t d : safe_d) {
      uint32_t src = SUB_START + (lT ^ d);
      write_pattern(src, P[4]);
      zero_rows(C.open_rows);
      fire(src, T, 30, 1);
      read_tuple(rb);
      string pred = coset_hits(lT ^ d, d), got;
      for (int i = 0; i < 16; i++)
        got += classify(rb[i].data(), {{'S', Pi[4].data()}});
      bool ok = true;
      for (int i = 0; i < 16; i++) {
        char e = pred[i], g = got[i];
        if (e=='S' ? (g!='S'&&g!='s') : (g!='0'&&g!='.')) ok = false;
      }
      printf("T=%u d=%-3u src=%u  pred=%s got=%s  %s\n",
             T, d, src, pred.c_str(), got.c_str(), ok ? "CLEAN" : "MISMATCH");
    }
  }

  // ---- Phase 2: unsafe-offset precision controls ----
  printf("\n## Phase 2 — UNSAFE offsets: predict exactly which rows corrupt\n");
  uint32_t unsafe_d[4] = {5, 6, 100, 389};
  for (uint32_t d : unsafe_d) {
    uint32_t T = base, lT = lbase;
    uint32_t src = SUB_START + (lT ^ d);
    bool src_in_tuple = false;
    for (uint32_t r : C.open_rows) if (r == src) src_in_tuple = true;
    if (src_in_tuple) { printf("d=%-3u src=%u IS a tuple member — skipped\n", d, src); continue; }
    write_pattern(src, P[4]);
    zero_rows(C.open_rows);
    fire(src, T, 30, 1);
    read_tuple(rb);
    string pred = coset_hits(lT ^ d, d), got;
    for (int i = 0; i < 16; i++)
      got += classify(rb[i].data(), {{'S', Pi[4].data()}});
    printf("T=%u d=%-3u src=%u  pred=%s got=%s\n",
           T, d, src, pred.c_str(), got.c_str());
  }

  // ---- Phase 3: off-anchor mat-group broadcasts ----
  printf("\n## Phase 3 — g=3 broadcast anchored at each mat-group (t=10,2)\n");
  for (int gidx = 0; gidx < 4; gidx++) {
    uint32_t A  = C.open_rows[gidx * 4];
    uint32_t lA = A - SUB_START;
    write_pattern(A, P[gidx]);
    { vector<uint32_t> others; for (uint32_t r : C.open_rows) if (r != A) others.push_back(r);
      zero_rows(others); }
    fire(A, SUB_START + (lA ^ 3u), 10, 2);
    read_tuple(rb);
    string got;
    for (int i = 0; i < 16; i++)
      got += classify(rb[i].data(), {{(char)('A'+gidx), Pi[gidx].data()}});
    printf("anchor=%u group=%d  got=%s\n", A, gidx, got.c_str());
  }

  // ---- Phase 4: the full fast loader ----
  printf("\n## Phase 4 — full fast tuple load: 4 stage-writes + 8 doubleACTs\n");
  uint32_t stage_off = 8;  // safe: bit3 is no generator on s72
  zero_rows(C.open_rows);
  vector<uint32_t> stages;
  for (int g = 0; g < 4; g++) {
    uint32_t A = C.open_rows[g * 4];
    uint32_t src = SUB_START + ((A - SUB_START) ^ stage_off);
    stages.push_back(src);
    write_pattern(src, P[g]);            // in a real flow this row already holds data
  }
  for (int g = 0; g < 4; g++) {          // 4 safe RowClones into anchors
    uint32_t A = C.open_rows[g * 4];
    fire(stages[g], A, 30, 1);
  }
  for (int g = 0; g < 4; g++) {          // 4 intra-group broadcasts
    uint32_t A = C.open_rows[g * 4];
    fire(A, SUB_START + ((A - SUB_START) ^ 3u), 10, 2);
  }
  read_tuple(rb);
  string got;
  int good = 0;
  for (int i = 0; i < 16; i++) {
    char cl = classify(rb[i].data(), {{'A', Pi[0].data()}, {'B', Pi[1].data()},
                                      {'C', Pi[2].data()}, {'D', Pi[3].data()}});
    got += cl;
    if (cl == 'A' + (i / 4) || cl == 'a' + (i / 4)) good++;
  }
  printf("expected=AAAABBBBCCCCDDDD got=%s  -> %d/16 rows correct  %s\n",
         got.c_str(), good, good == 16 ? "FAST_LOAD_OK" : "FAST_LOAD_FAIL");

  printf("\n[safe] DONE\n");
  return 0;
}
