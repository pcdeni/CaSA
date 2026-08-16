// Fused-LAYOUT column screen (2026-07-20, Task O10) — dimm0's fused residual.
//
// Addendum 24 diagnosed bender 0's fused-arm residual (~52 y-segments,
// deterministic core + flicker margin, popcount-HIGH bias) as the fused
// OPERAND LAYOUT mis-computing on this die's weaker margins: the May
// calibration screened columns for the BASELINE interleaved layout
// (x@{1,4,7,10,13} 0@{2,5,8,11,14} W@{3,6,9,12,15}) and was layout-blind;
// PIM_FUSED_COSET=2 (fused positions via explicit wrRows, no cosets)
// reproduces the failure, exonerating the coset broadcasts.
//
// SHAPE FIDELITY (the load-bearing detail): the 2026-07-17 fused per-MAJ
// A/B (test_fused_maj, ONE body per program, single bank) was EXACT on
// bender 0 — the residual only exists in the server MATVEC shape. This
// tool therefore replicates the server's per-(round, bitplane) emission
// EXACTLY: one program = the 4 banks' combined bodies back-to-back
// (build_multibank_combined_program, K=1, serial builder, no consts),
// one receiveData(4 x 8192) per program, per-bank calibs (banks 0/2/3
// share the s61 tuple, bank 1 has its own), backup rows drawn from the
// per-bank cloneok pools (rotating like LOAD rounds), and the MM3D-entry
// subarray refresh loop every 16 programs (= one gate matmul's worth).
// Only the LAYOUT STEP is parameterized.
//
// W draws include gate-shaped sparse ternary masks (~25% and ~6% ones)
// next to dense randoms and structured patterns; x draws are uniform
// 32-bit (bank-pair split like the gate's chunk pairing).
//
// Layouts:
//   base     wrRow  x@{1,4,7,10,13}  0@{2,5,8,11,14}   W@{3,6,9,12,15}
//   fusedwr  wrRow  x@{1,5,9,13,4}   0@{2,6,10,14,8}   W@{3,7,11,12,15}
//   fused    coset  x=wr@1+dACT(1,13)+wr@4  0=wr@2+dACT(2,14)+wr@8
//   mirror   coset  roles x<->0 swapped on the same cosets
//   altfused coset  x=wr@1+dACT(1,11)+wr@2  0=wr@4+dACT(4,14)+wr@8
//                   (x@{1,3,9,11}+2, 0@{4,6,12,14}+8, W@{5,7,10,13,15})
//   hybA/B/C wrRow  base + ONE role-pair swap each: A=swap(5,10)
//                   B=swap(7,9) C=swap(6,11)
//
// Output: <out_dir>/fused_screen.csv (layout,bank,col,fails,runs,hi,lo)
// and <out_dir>/fused_colmask_b<bender>_bank<B>.txt per bank (columns exact
// in EVERY 'fused' run — the server-side colmask). The file names the
// channel it was measured on: a colmask belongs to one die.
//
// Argv: <bender> <calib> <banks_csv> <pool_pattern({bank})> <layouts_csv|all>
//       [n_wrand=8] [n_xrand=8] [trials=2] [out_dir=.]
// 'fused' runs trials*3 (flicker-margin depth).
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
#include <set>
#include <sstream>
#include <string>
#include <vector>
using namespace std;

static const int CHUNK_COLS[3] = {43, 43, 42};

// ---- per-column write of a full row (server per_column_write_row shape) --
static Program chunkp(int bank, uint32_t row, const uint32_t* cd, int cs, int n) {
  Program p; p.add_inst(SMC_LI(8, CASR)); p.add_inst(SMC_LI(bank, BAR));
  p.add_inst(SMC_LI(row, RAR)); p.add_inst(SMC_LI(cs * 8, CAR));
  p.add_below(PRE(BAR, 0, 0)); p.add_below(ACT(BAR, 0, RAR, 0));
  for (int k = 0; k < n; k++) { const uint32_t* sl = cd + k * 16;
    for (int s = 0; s < 16; s++) { p.add_inst(SMC_LI(sl[s], PATTERN_REG)); p.add_inst(SMC_LDWD(PATTERN_REG, s)); }
    p.add_below(WRITE(BAR, CAR, 1)); p.add_inst(SMC_SLEEP(8)); }
  p.add_inst(SMC_SLEEP(8)); p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_SLEEP(4)); p.add_inst(SMC_END());
  return p;
}
static void pcw(SoftMCPlatform& pf, int bank, uint32_t row, const uint32_t* d) {
  int cs = 0; for (int ch = 0; ch < 3; ch++) { Program p = chunkp(bank, row, d + cs * 16, cs, CHUNK_COLS[ch]); pf.execute(p); cs += CHUNK_COLS[ch]; }
}

// ---- server's frac template (test_bitnet_server.cpp, verbatim) ----------
static Program frac_template(int t_frac, uint32_t r_frac_addr) {
  Program p;
  int R_FRAC_REG = RF_REG;
  int bank_reg = BAR;
  p.add_inst(all_nops());
  p.add_inst(SMC_LI(r_frac_addr, R_FRAC_REG));
  int num_cmd = 2 + t_frac;
  num_cmd += (num_cmd % 4) ? (4 - (num_cmd % 4)) : 0;
  Mininst* q_inst = new Mininst[num_cmd];
  for (int i = 0; i < num_cmd; i++) q_inst[i] = SMC_NOP();
  q_inst[0]          = SMC_ACT(bank_reg, 0, R_FRAC_REG, 0);
  q_inst[t_frac + 1] = SMC_PRE(bank_reg, 0, 0);
  for (int i = 0; i < num_cmd; i += 4)
    p.add_inst(q_inst[i], q_inst[i + 1], q_inst[i + 2], q_inst[i + 3]);
  delete[] q_inst;
  return p;
}

struct Calib { int s_id, bank; uint32_t Rf, Rs; vector<uint32_t> r; };
// Server semantics: cs[0] = FIRST calib line for the bank.
static Calib readc_first(const string& path, int wb) {
  ifstream f(path); string line;
  while (getline(f, line)) { if (line.empty() || line[0] == '#') continue; istringstream is(line);
    Calib c; if (!(is >> c.s_id >> c.bank >> c.Rf >> c.Rs)) continue; uint32_t v; while (is >> v) c.r.push_back(v);
    if (c.r.size() == 16 && c.bank == wb) return c; }
  return Calib{-1, 0, 0, 0, {}};
}

// ---- layouts -------------------------------------------------------------
struct Lay {
  const char* name;
  bool coset;
  int xs[5], zs[5];
  int xa, xp, xe, za, zp, ze;
  int depth_mult;
};
static const Lay LAYS[] = {
  {"base",    false, {1,4,7,10,13}, {2,5,8,11,14}, 0,0,0,0,0,0, 1},
  {"fusedwr", false, {1,5,9,13,4},  {2,6,10,14,8}, 0,0,0,0,0,0, 1},
  {"fused",   true,  {0},{0},           1,13,4, 2,14,8,          3},
  {"mirror",  true,  {0},{0},           2,14,8, 1,13,4,          1},
  {"altfused",true,  {0},{0},           1,11,2, 4,14,8,          1},
  {"hybA",    false, {1,4,5,7,13},  {2,8,10,11,14},0,0,0,0,0,0, 1},
  {"hybB",    false, {1,4,9,10,13}, {2,5,8,11,14}, 0,0,0,0,0,0, 1},
  {"hybC",    false, {1,4,7,10,13}, {2,5,6,8,14},  0,0,0,0,0,0, 1},
};
static const int NLAYS = sizeof(LAYS) / sizeof(LAYS[0]);

// One bank's body — emit_bank_combined_body (no consts), layout step
// parameterized. Caller emits SMC_LI(8, CASR) once and SMC_END() last.
static void emit_body(Program& p, int bank, const Calib& c, uint32_t backup,
                      const Lay& L, uint32_t x, int lb) {
  p.add_inst(SMC_LI(bank, BAR));
  p.add_inst(SMC_LI(128, NUM_COLS_REG));
  // 1. RowClone backup -> Rfirst.
  p.add_below(PRE(BAR, 0, 0));
  p.add_inst(SMC_SLEEP(6));
  p.add_below(doubleACT(30, 1, backup, c.Rf));
  p.add_inst(SMC_SLEEP(6));
  p.add_below(PRE(BAR, 0, 0));
  p.add_inst(SMC_SLEEP(6));
  // 2. Broadcast Rfirst -> all 16.
  p.add_below(doubleACT(10, 2, c.Rf, c.Rs));
  p.add_inst(SMC_SLEEP(6));
  p.add_below(PRE(BAR, 0, 0));
  p.add_inst(SMC_SLEEP(6));
  // 3. Layout writes (+ cosets).
  if (L.coset) {
    p.add_below(wrRow_immediate_label(BAR, c.r[0],    ONE, lb + 0));
    p.add_below(wrRow_immediate_label(BAR, c.r[L.xa], x,   lb + 1));
    p.add_below(wrRow_immediate_label(BAR, c.r[L.za], 0u,  lb + 2));
    p.add_below(wrRow_immediate_label(BAR, c.r[L.xe], x,   lb + 3));
    p.add_below(wrRow_immediate_label(BAR, c.r[L.ze], 0u,  lb + 4));
    p.add_inst(SMC_SLEEP(6));
    p.add_below(PRE(BAR, 0, 0));
    p.add_inst(SMC_SLEEP(6));
    p.add_below(doubleACT(10, 2, c.r[L.xa], c.r[L.xp]));
    p.add_inst(SMC_SLEEP(6));
    p.add_below(PRE(BAR, 0, 0));
    p.add_inst(SMC_SLEEP(6));
    p.add_below(doubleACT(10, 2, c.r[L.za], c.r[L.zp]));
  } else {
    p.add_below(wrRow_immediate_label(BAR, c.r[0], ONE, lb + 0));
    for (int i = 0; i < 5; i++)
      p.add_below(wrRow_immediate_label(BAR, c.r[L.xs[i]], x,  lb + 1 + i));
    for (int i = 0; i < 5; i++)
      p.add_below(wrRow_immediate_label(BAR, c.r[L.zs[i]], 0u, lb + 100 + i));
  }
  p.add_inst(SMC_SLEEP(6));
  p.add_below(PRE(BAR, 0, 0));
  p.add_inst(SMC_SLEEP(6));
  // 4. Frac x3 on r0.
  for (int j = 0; j < 3; j++) {
    p.add_inst(SMC_SLEEP(6));
    p.add_below(frac_template(0, c.r[0]));
    p.add_inst(SMC_SLEEP(6));
  }
  p.add_inst(SMC_SLEEP(6));
  p.add_below(PRE(BAR, 0, 0));
  p.add_inst(SMC_SLEEP(6));
  // 5. MAJ3.
  p.add_below(doubleACT(0, 0, c.Rf, c.Rs));
  p.add_inst(SMC_SLEEP(6));
  p.add_below(PRE(BAR, 0, 0));
  p.add_inst(SMC_SLEEP(6));
  // 6. Read + trailer.
  p.add_below(rdRow_immediate_label(BAR, c.r[0], lb + 999));
  p.add_inst(all_nops());
  p.add_inst(all_nops());
  p.add_below(PRE(BAR, 0, 0));
  p.add_inst(all_nops());
  p.add_inst(all_nops());
}

// MM3D-entry subarray refresh (server build_refresh_subarray_loop_program).
static Program build_refresh(const vector<int>& banks, uint32_t r0, uint32_t r1) {
  Program p;
  p.add_inst(SMC_LI(8, CASR));
  for (size_t bi = 0; bi < banks.size(); bi++) {
    p.add_inst(SMC_LI(banks[bi], BAR));
    p.add_inst(SMC_LI(r0, LOOP_ROWS));
    p.add_inst(SMC_LI(r1, NUM_ROWS_REG));
    string lab = "REFRESH_SUBARR_B" + to_string(banks[bi]) + "_" + to_string(bi);
    p.add_label(lab);
      p.add_inst(SMC_ADDI(LOOP_ROWS, 0, RAR));
      p.add_below(PRE(BAR, 0, 0));
      p.add_below(ACT(BAR, 0, RAR, 0));
      p.add_inst(SMC_SLEEP(4));
      p.add_below(PRE(BAR, 0, 0));
      p.add_inst(SMC_SLEEP(4));
      p.add_inst(SMC_ADDI(LOOP_ROWS, 1, LOOP_ROWS));
    p.add_branch(p.BR_TYPE::BL, LOOP_ROWS, NUM_ROWS_REG, lab);
    p.add_inst(all_nops());
  }
  p.add_inst(SMC_END());
  return p;
}

static vector<uint32_t> pool_rows(const string& pattern, int bank,
                                  const Calib& c, size_t n_want) {
  string path = pattern;
  size_t pos;
  while ((pos = path.find("{bank}")) != string::npos)
    path.replace(pos, 6, to_string(bank));
  ifstream f(path);
  if (!f) { fprintf(stderr, "[fcs] cannot read pool file %s\n", path.c_str()); _exit(4); }
  set<uint32_t> avoid(c.r.begin(), c.r.end());
  avoid.insert(c.Rf); avoid.insert(c.Rs);
  vector<uint32_t> out; string line;
  while (getline(f, line) && out.size() < n_want) {
    size_t i = line.find_first_not_of(" \t");
    if (i == string::npos || line[i] == '#') continue;
    uint32_t r = (uint32_t)atoi(line.c_str() + i);
    if (r > 0 && !avoid.count(r)) out.push_back(r);
  }
  if (out.size() < n_want) {
    fprintf(stderr, "[fcs] pool %s too small (%zu < %zu)\n", path.c_str(),
            out.size(), n_want);
    _exit(4);
  }
  return out;
}

static uint32_t rnd32() {
  return ((uint32_t)rand() << 17) ^ ((uint32_t)rand() << 3) ^ (uint32_t)rand();
}

int main(int argc, char** argv) {
  if (argc < 6) {
    fprintf(stderr, "Usage: %s <bender> <calib> <banks_csv> "
            "<pool_pattern({bank})> <layouts_csv|all> [n_wrand=8] "
            "[n_xrand=8] [trials=2] [out_dir=.]\n", argv[0]);
    return 1;
  }
  int bender = atoi(argv[1]);
  string calib_path = argv[2];
  vector<int> banks;
  { stringstream ss(argv[3]); string t;
    while (getline(ss, t, ',')) if (!t.empty()) banks.push_back(atoi(t.c_str())); }
  string pool_pat = argv[4];
  string lays_csv = argv[5];
  int n_wrand = (argc > 6) ? atoi(argv[6]) : 8;
  int n_xrand = (argc > 7) ? atoi(argv[7]) : 8;
  int trials  = (argc > 8) ? atoi(argv[8]) : 2;
  string out_dir = (argc > 9) ? argv[9] : ".";
  const int NB = (int)banks.size();
  const int N_ROT = 4;   // backup rows rotated across W draws, LOAD-style

  vector<Calib> cs;
  vector<vector<uint32_t>> pools;
  for (int b : banks) {
    Calib c = readc_first(calib_path, b);
    if (c.s_id < 0) { fprintf(stderr, "[fcs] no calib bank=%d\n", b); return 2; }
    pools.push_back(pool_rows(pool_pat, b, c, N_ROT));
    cs.push_back(c);
  }

  vector<int> use;
  if (lays_csv == "all") { for (int i = 0; i < NLAYS; i++) use.push_back(i); }
  else {
    stringstream ss(lays_csv); string t;
    while (getline(ss, t, ',')) {
      bool hit = false;
      for (int i = 0; i < NLAYS; i++)
        if (t == LAYS[i].name) { use.push_back(i); hit = true; }
      if (!hit) { fprintf(stderr, "[fcs] unknown layout '%s'\n", t.c_str()); return 1; }
    }
  }

  // W draws (per bank per draw — banks get independent masks like the
  // gate's per-unit masks): 6 structured classes + n_wrand randoms.
  // Classes: dense-1s, 0xAA, 0x55, col-ramp, sparse ~25% (pos-mask-like),
  // sparse ~6% (neg-mask-like).
  srand(20260720);
  int n_W = 6 + n_wrand;
  // Ws[wi][bk][col]
  vector<vector<vector<uint32_t>>> Ws(n_W,
      vector<vector<uint32_t>>(NB, vector<uint32_t>(2048)));
  for (int bk = 0; bk < NB; bk++) {
    for (int s = 0; s < 2048; s++) {
      Ws[0][bk][s] = 0xFFFFFFFFu;
      Ws[1][bk][s] = 0xAAAAAAAAu;
      Ws[2][bk][s] = 0x55555555u;
      Ws[3][bk][s] = 0x01010101u * (uint32_t)((s + bk) & 0xFF);
      Ws[4][bk][s] = rnd32() & rnd32();                       // ~25%
      Ws[5][bk][s] = rnd32() & rnd32() & rnd32() & rnd32();   // ~6%
    }
  }
  for (int wi = 6; wi < n_W; wi++)
    for (int bk = 0; bk < NB; bk++) {
      int cls = wi % 3;  // rotate dense / 25% / 6% random densities
      for (int s = 0; s < 2048; s++) {
        uint32_t v = rnd32();
        if (cls == 1) v &= rnd32();
        if (cls == 2) v &= rnd32() & rnd32() & rnd32();
        Ws[wi][bk][s] = v;
      }
    }
  // x draws: 4 structured + n_xrand randoms. Per program, bank pair 0/1
  // gets Xs[xi], pair 2/3 gets Xs[(xi + 7) % n_X] (gate chunk pairing).
  vector<uint32_t> Xs = {0x00000000u, 0xFFFFFFFFu, 0xAAAAAAAAu, 0x55555555u};
  for (int k = 0; k < n_xrand; k++) Xs.push_back(rnd32());
  const int n_X = (int)Xs.size();

  SoftMCPlatform pf(bender);
  if (pf.init() != SOFTMC_SUCCESS) { fprintf(stderr, "[fcs] init fail\n"); return 3; }
  pf.reset_fpga();

  // Refresh window: env PIM_SUB_START/END or 640-aligned default.
  uint32_t sub_s = (cs[0].r[0] / 640) * 640, sub_e = sub_s + 640;
  if (const char* ss = getenv("PIM_SUB_START")) if (*ss) sub_s = (uint32_t)atoi(ss);
  if (const char* se = getenv("PIM_SUB_END"))   if (*se) sub_e = (uint32_t)atoi(se);
  Program refresh_prog = build_refresh(banks, sub_s, sub_e);

  fprintf(stderr, "[fcs] bender=%d banks=%d layouts=%zu W=%d x=%d trials=%d "
          "refresh=[%u,%u)\n", bender, NB, use.size(), n_W, n_X, trials,
          sub_s, sub_e);
  for (int bk = 0; bk < NB; bk++)
    fprintf(stderr, "[fcs]   bank %d: s_id=%d Rf=%u Rs=%u backups=%u,%u,%u,%u\n",
            banks[bk], cs[bk].s_id, cs[bk].Rf, cs[bk].Rs,
            pools[bk][0], pools[bk][1], pools[bk][2], pools[bk][3]);

  char csvp[512];
  snprintf(csvp, sizeof csvp, "%s/fused_screen.csv", out_dir.c_str());
  FILE* csv = fopen(csvp, "w");
  if (!csv) { fprintf(stderr, "[fcs] cannot write %s\n", csvp); return 4; }
  fprintf(csv, "layout,bank,col,fails,runs,hi,lo\n");

  vector<uint8_t> buf((size_t)NB * 8192);
  // fused per-bank fail counts for the colmask.
  vector<vector<int>> fused_fails(NB, vector<int>(2048, 0));
  long fused_runs = 0;
  long progs_since_refresh = 0;

  for (int li : use) {
    const Lay& L = LAYS[li];
    int T = trials * L.depth_mult;
    vector<vector<int>> fails(NB, vector<int>(2048, 0)),
                        hi(NB, vector<int>(2048, 0)),
                        lo(NB, vector<int>(2048, 0));
    long runs = 0, bad_total = 0;
    long n_inst = 0;
    auto t0 = chrono::steady_clock::now();
    for (int wi = 0; wi < n_W; wi++) {
      // Rotate backup rows LOAD-style; write each bank's W mask.
      int rot = wi % N_ROT;
      for (int bk = 0; bk < NB; bk++)
        pcw(pf, banks[bk], pools[bk][rot], Ws[wi][bk].data());
      for (int xi = 0; xi < n_X; xi++) {
        uint32_t x01 = Xs[xi], x23 = Xs[(xi + 7) % n_X];
        // One program = all banks' bodies (gate M=4 emission).
        Program p;
        p.add_inst(SMC_LI(8, CASR));
        for (int bk = 0; bk < NB; bk++)
          emit_body(p, banks[bk], cs[bk], pools[bk][rot], L,
                    (bk < 2) ? x01 : x23, bk * 2000);
        p.add_inst(SMC_END());
        n_inst = p.size() / 8;
        for (int t = 0; t < T; t++) {
          // MM3D-entry refresh cadence: every 16 programs = 1 gate matmul.
          if (progs_since_refresh % 16 == 0) pf.execute(refresh_prog);
          progs_since_refresh++;
          pf.execute(p);
          int rc = pf.receiveData(buf.data(), NB * 8192);
          if (rc != NB * 8192) {
            fprintf(stderr, "[fcs] recv rc=%d layout=%s wi=%d xi=%d t=%d "
                    "— ABORT\n", rc, L.name, wi, xi, t);
            fclose(csv); _exit(5);
          }
          runs++;
          for (int bk = 0; bk < NB; bk++) {
            const uint32_t x = (bk < 2) ? x01 : x23;
            const uint8_t* rb = buf.data() + (size_t)bk * 8192;
            for (int s = 0; s < 2048; s++) {
              uint32_t got; memcpy(&got, rb + s * 4, 4);
              uint32_t exp = Ws[wi][bk][s] & x;
              if (got != exp) {
                fails[bk][s]++; bad_total++;
                if (got & ~exp) hi[bk][s]++;
                if (exp & ~got) lo[bk][s]++;
              }
            }
          }
        }
      }
    }
    auto t1 = chrono::steady_clock::now();
    double secs = chrono::duration<double>(t1 - t0).count();
    int cols_bad = 0;
    for (int bk = 0; bk < NB; bk++)
      for (int s = 0; s < 2048; s++) if (fails[bk][s]) cols_bad++;
    for (int bk = 0; bk < NB; bk++)
      for (int s = 0; s < 2048; s++)
        if (fails[bk][s])
          fprintf(csv, "%s,%d,%d,%d,%ld,%d,%d\n", L.name, banks[bk], s,
                  fails[bk][s], runs, hi[bk][s], lo[bk][s]);
    fprintf(csv, "%s,-1,-1,%ld,%ld,%d,%ld\n", L.name, bad_total, runs,
            cols_bad, n_inst);
    fflush(csv);
    fprintf(stderr, "[fcs] %-8s: %4d bad (bank,col)s, %6ld bad cells / %ld "
            "runs x %d banks (%ld inst/prog, %.1fs)\n",
            L.name, cols_bad, bad_total, runs, NB, n_inst, secs);
    if (!strcmp(L.name, "fused")) {
      fused_fails = fails;
      fused_runs = runs;
    }
  }
  fclose(csv);

  if (fused_runs > 0) {
    for (int bk = 0; bk < NB; bk++) {
      char cmp[512];
      snprintf(cmp, sizeof cmp, "%s/fused_colmask_b%d_bank%d.txt",
               out_dir.c_str(), bender, banks[bk]);
      FILE* f = fopen(cmp, "w");
      if (!f) { fprintf(stderr, "[fcs] cannot write %s\n", cmp); _exit(4); }
      int good = 0;
      for (int s = 0; s < 2048; s++) if (fused_fails[bk][s] == 0) good++;
      // Headers < 64 chars each (O8 fgets hazard convention).
      fprintf(f, "# fused-layout colmask bender=%d bank=%d\n", bender, banks[bk]);
      fprintf(f, "# good %d/2048 runs=%ld tool=fused-colmask-exe\n",
              good, fused_runs);
      fprintf(f, "# Rf=%u Rs=%u s_id=%d\n", cs[bk].Rf, cs[bk].Rs, cs[bk].s_id);
      for (int s = 0; s < 2048; s++)
        if (fused_fails[bk][s] == 0) fprintf(f, "%d\n", s);
      fclose(f);
      fprintf(stderr, "[fcs] colmask bank %d: %d/2048 fused-good -> %s\n",
              banks[bk], good, cmp);
    }
  }
  fprintf(stderr, "[fcs] DONE\n");
  fflush(stderr);
  _exit(0);
}
