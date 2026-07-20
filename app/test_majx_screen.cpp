// MAJX × reference-policy screen v3 (2026-07-17, Task S2) — generalizes the
// frac-maj5 sweep to MAJ3/5/7/9/11/15 on calibrated 16-row tuples AND
// law-constructed 32-row cosets, testing the vote-algebra model of the free
// rows:
//   C copies × X inputs = C·X data votes on N co-activated rows; the
//   F = N−C·X free rows carry a total charge of r votes. Correct MAJX on
//   BOTH input sides requires
//     N/2 − C·ceil(X/2)  <  r  <  N/2 − C·floor(X/2)   (open, width C)
//   N=16: MAJ7 (C=2) only r=1; MAJ9/15 (C=1) have NO valid binary r —
//         a fractional row would be structurally required.
//   N=32: intervals reopen — MAJ7 C=4 r=2 (true balanced), MAJ9 C=3
//         r∈{2,3}, MAJ11 C=2 r=5, MAJ15 C=2 r=1. This mirrors SiMRA's
//         Observation 6 (input replication raises success) and their Fig-7
//         32-row protocol (MAJ5/7/9 = 79.64/33.87/5.91% avg; footnote 11:
//         MAJ9+ <1% on Mfr. M, MAJ11+ <1% on Mfr. H).
//   Their timing optimum t1=1.5ns/t2=3ns = doubleACT(1,2) is included as
//   _t12 variants next to our calibrated (0,0).
// Diagnostics per config: jmaj (every bit at minimal majority, exp=1) and
// jmin (maximal minority, exp=0) read the two margin sides directly.
// 32-row cosets are built BY FORMULA from the selection law: units
// {1,2,8,32,128} (one per predecoder group + bit0), d=171, coset
// {A ⊕ S : S ⊆ units} — 32 rows, span 171, inside one subarray.
// Modes / argv:
//   sweep:   <bender> <calib> <bank> <s_id> sweep [n_rand=12] [trials=6]
//   sweep32: <bender> <SUB> <bank> <A_local> sweep32 [n_rand=12] [trials=6]
//   colmask: <bender> <calib> <bank> <s_id> colmask [n_rand=18] [trials=25] <out>
//   select:  <bender> <calib[,c2...]> <bank> -1 select [n_rand=18] [trials=25] <out_prefix>
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
#include <set>
#include <sstream>
#include <string>
#include <vector>
using namespace std;

#define NUM_BANKS_HERE 1
#define NUM_ROWS_DEF 2048
#define LOOP_ITER 15
#define ITER_REG 9
#define ZERO 0x00000000u
#define ONE  0xFFFFFFFFu

static Program frac_builder(int t_frac, int r_frac_addr) {
  Program p;
  p.add_inst(all_nops());
  int R_FRAC_REG = RF_REG, bank_reg = BAR;
  p.add_inst(SMC_LI(r_frac_addr, R_FRAC_REG));
  int num_cmd = 2 + t_frac; num_cmd += 4 - (num_cmd % 4);
  Mininst q[num_cmd];
  for (int i = 0; i < num_cmd; i++) q[i] = SMC_NOP();
  q[0] = SMC_ACT(bank_reg, 0, R_FRAC_REG, 0);
  q[t_frac + 1] = SMC_PRE(bank_reg, 0, 0);
  for (int i = 0; i < num_cmd; i += 4) p.add_inst(q[i], q[i+1], q[i+2], q[i+3]);
  return p;
}
static Program _init(uint32_t bank_id, uint32_t num_iter) {
  Program p;
  p.add_inst(SMC_LI(NUM_ROWS_DEF, NUM_ROWS_REG));
  p.add_inst(SMC_LI(NUM_BANKS_HERE, NUM_BANKS_REG));
  p.add_inst(SMC_LI(num_iter, ITER_REG));
  p.add_inst(SMC_LI(8, CASR)); p.add_inst(SMC_LI(1, BASR));
  p.add_inst(SMC_LI(1, RASR)); p.add_inst(SMC_LI(128, NUM_COLS_REG));
  p.add_inst(SMC_LI(bank_id, BAR));
  return p;
}
static Program test_prog(uint32_t bank_id, const vector<uint32_t>& patterns,
                         const vector<uint32_t>& open_row_idx,
                         const vector<uint32_t>& frac_rows,
                         uint32_t Rf, uint32_t Rs,
                         uint32_t n_frac, uint32_t t_frac,
                         int t12, int t23, uint32_t num_iter) {
  Program p;
  p.add_below(_init(bank_id, num_iter));
  p.add_inst(all_nops()); p.add_inst(all_nops());
  p.add_below(PRE(BAR, 0, 0));
  p.add_inst(SMC_LI(0, LOOP_ITER));
  p.add_label("ITER_LOOP");
    p.add_below(PRE(BAR, 0, 0));
    for (size_t i = 0; i < open_row_idx.size(); i++)
      p.add_below(wrRow_immediate_label(BAR, open_row_idx[i], patterns[i], rand()));
    p.add_inst(SMC_SLEEP(6)); p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_SLEEP(6));
    for (uint32_t j = 0; j < n_frac; j++)
      for (uint32_t fr : frac_rows) {
        p.add_inst(SMC_SLEEP(6));
        p.add_below(frac_builder(t_frac, fr));
        p.add_inst(SMC_SLEEP(6));
      }
    p.add_inst(SMC_SLEEP(6)); p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_SLEEP(6));
    p.add_below(doubleACT(t12, t23, Rf, Rs));
    p.add_inst(SMC_SLEEP(6)); p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_SLEEP(6));
    p.add_below(rdRow_immediate(BAR, open_row_idx[0]));
    p.add_inst(all_nops()); p.add_inst(all_nops());
    p.add_below(PRE(BAR, 0, 0));
    p.add_inst(all_nops()); p.add_inst(all_nops());
    p.add_inst(SMC_ADDI(LOOP_ITER, 1, LOOP_ITER));
  p.add_branch(p.BR_TYPE::BL, LOOP_ITER, ITER_REG, "ITER_LOOP");
  p.add_inst(SMC_END());
  return p;
}

struct Cfg {
  const char* name; int X; int ns;   // ns = co-activated rows (16 or 32)
  uint32_t one_mask;   // bit s → free slot s init ONE (else ZERO)
  uint32_t frac_mask;  // bit s → slot s receives the frac pulses
  int nf, tf;
  int t12, t23;        // MAJ doubleACT timing
};
static int copies_for(int X, int ns){
  if (ns == 16) return (X==3)?5:(X==5)?3:(X==7)?2:1;
  return (X==3)?10:(X==5)?6:(X==7)?4:(X==9)?3:2;   // ns=32: X=11,15 → 2
}
static vector<uint32_t> buildN(const Cfg& cf, const uint32_t* I) {
  int nd = copies_for(cf.X, cf.ns) * cf.X;
  vector<uint32_t> lst(cf.ns, 0);
  for (int j = 0; j < nd; j++) lst[1 + j] = I[j % cf.X];
  lst[0] = (cf.one_mask & 1u) ? ONE : ZERO;
  for (int s = nd + 1; s < cf.ns; s++) lst[s] = ((cf.one_mask >> s) & 1u) ? ONE : ZERO;
  return lst;
}
static uint32_t majx(const uint32_t* I, int X) {
  uint32_t out = 0;
  for (int b = 0; b < 32; b++) {
    int c = 0; for (int i = 0; i < X; i++) c += (I[i] >> b) & 1;
    if (c * 2 > X) out |= 1u << b;
  }
  return out;
}
static const Cfg CFGS[] = {
  {"m3_Z0",    3,16, 0x0000, 0x0000, 0,0, 0,0},
  {"m3_O0",    3,16, 0x0001, 0x0000, 0,0, 0,0},
  {"m3_simra", 3,16, 0x0001, 0x0001, 3,0, 0,0},
  {"m5_Z0",    5,16, 0x0000, 0x0000, 0,0, 0,0},
  {"m5_Z2",    5,16, 0x0000, 0x0001, 2,0, 0,0},
  {"m5_simra", 5,16, 0x0001, 0x0001, 3,0, 0,0},
  {"m7_ZZ",    7,16, 0x0000, 0x0000, 0,0, 0,0},
  {"m7_OO",    7,16, 0x8001, 0x0000, 0,0, 0,0},
  {"m7_BAL",   7,16, 0x0001, 0x0000, 0,0, 0,0},
  {"m7_LAB",   7,16, 0x8000, 0x0000, 0,0, 0,0},
  {"m7_FF2",   7,16, 0x8001, 0x8001, 2,0, 0,0},
  {"m7_FF3",   7,16, 0x8001, 0x8001, 3,0, 0,0},
  {"m9_4O",    9,16, 0x3C00, 0x0000, 0,0, 0,0},
  {"m9_3O",    9,16, 0x1C00, 0x0000, 0,0, 0,0},
  {"m9_3O_OF1",9,16, 0x1C01, 0x0001, 1,0, 0,0},
  {"m9_3O_OF2",9,16, 0x1C01, 0x0001, 2,0, 0,0},
  {"m9_3O_OF3",9,16, 0x1C01, 0x0001, 3,0, 0,0},
  {"m9_3O_OF6",9,16, 0x1C01, 0x0001, 6,0, 0,0},
  {"m9_3O_ZF2",9,16, 0x1C00, 0x0001, 2,0, 0,0},
  {"m9_3O_ZF4",9,16, 0x1C00, 0x0001, 4,0, 0,0},
  {"m9_3O_ZF8",9,16, 0x1C00, 0x0001, 8,0, 0,0},
  {"m15_Z0",  15,16, 0x0000, 0x0000, 0,0, 0,0},
  {"m15_O0",  15,16, 0x0001, 0x0000, 0,0, 0,0},
  {"m15_OF1", 15,16, 0x0001, 0x0001, 1,0, 0,0},
  {"m15_OF2", 15,16, 0x0001, 0x0001, 2,0, 0,0},
  {"m15_OF3", 15,16, 0x0001, 0x0001, 3,0, 0,0},
  {"m15_OF4", 15,16, 0x0001, 0x0001, 4,0, 0,0},
  {"m15_OF6", 15,16, 0x0001, 0x0001, 6,0, 0,0},
  {"m15_OF8", 15,16, 0x0001, 0x0001, 8,0, 0,0},
  {"m15_OF12",15,16, 0x0001, 0x0001,12,0, 0,0},
  {"m15_ZF2", 15,16, 0x0000, 0x0001, 2,0, 0,0},
  {"m15_ZF4", 15,16, 0x0000, 0x0001, 4,0, 0,0},
  {"m15_ZF6", 15,16, 0x0000, 0x0001, 6,0, 0,0},
  {"m15_ZF8", 15,16, 0x0000, 0x0001, 8,0, 0,0},
  {"m15_ZF12",15,16, 0x0000, 0x0001,12,0, 0,0},
  {"m15_ZF16",15,16, 0x0000, 0x0001,16,0, 0,0},
};
static const int NCFG = sizeof(CFGS)/sizeof(CFGS[0]);
// 32-row configs. Free slots = {0} ∪ {nd+1..31}. Structural intervals:
//   X=3 C=10: r∈(1,16);  X=5 C=6: r∈(−2,4);  X=7 C=4: r∈(0,4) → r=2 mid;
//   X=9 C=3: r∈(1,4) → r∈{2,3};  X=11 C=2: r∈(4,6) → r=5;
//   X=15 C=2: r∈(0,2) → r=1.
static const Cfg CFGS32[] = {
  {"m3_32_ZZ",     3,32, 0x00000000, 0, 0,0, 0,0},
  {"m3_32_BAL",    3,32, 0x00000001, 0, 0,0, 0,0},   // r=1
  {"m3_32_t12",    3,32, 0x00000000, 0, 0,0, 1,2},
  {"m5_32_ZZ",     5,32, 0x00000000, 0, 0,0, 0,0},   // r=0 (valid)
  {"m5_32_BAL",    5,32, 0x00000001, 0, 0,0, 0,0},   // r=1
  {"m5_32_ZZ_t12", 5,32, 0x00000000, 0, 0,0, 1,2},
  {"m7_32_2O2Z",   7,32, 0x20000001, 0, 0,0, 0,0},   // r=2, midpoint
  {"m7_32_2O2Z_t12",7,32,0x20000001, 0, 0,0, 1,2},
  {"m7_32_ZZZZ",   7,32, 0x00000000, 0, 0,0, 0,0},   // r=0 boundary
  {"m7_32_1O3Z",   7,32, 0x00000001, 0, 0,0, 0,0},   // r=1
  {"m7_32_3O1Z",   7,32, 0x60000001, 0, 0,0, 0,0},   // r=3
  {"m9_32_2O3Z",   9,32, 0x10000001, 0, 0,0, 0,0},   // r=2 (valid)
  {"m9_32_3O2Z",   9,32, 0x30000001, 0, 0,0, 0,0},   // r=3 (valid)
  {"m9_32_2O3Z_t12",9,32,0x10000001, 0, 0,0, 1,2},
  {"m9_32_Z5",     9,32, 0x00000000, 0, 0,0, 0,0},   // r=0 (invalid ctrl)
  {"m11_32_5O5Z", 11,32, 0x07800001, 0, 0,0, 0,0},   // r=5 (the only valid)
  {"m11_32_4O6Z", 11,32, 0x03800001, 0, 0,0, 0,0},   // r=4 boundary
  {"m11_32_6O4Z", 11,32, 0x0F800001, 0, 0,0, 0,0},   // r=6 boundary
  {"m15_32_1O1Z", 15,32, 0x00000001, 0, 0,0, 0,0},   // r=1 (the only valid)
  {"m15_32_ZZ",   15,32, 0x00000000, 0, 0,0, 0,0},   // r=0 boundary
  {"m15_32_OO",   15,32, 0x80000001, 0, 0,0, 0,0},   // r=2 boundary
};
static const int NCFG32 = sizeof(CFGS32)/sizeof(CFGS32[0]);

struct Pat { uint32_t I[15]; uint32_t exp; };
static const uint32_t BASEW[15] = {
  0xAAAAAAAAu,0xCCCCCCCCu,0xF0F0F0F0u,0xFF00FF00u,0xFFFF0000u,
  0x55555555u,0x33333333u,0x0F0F0F0Fu,0x00FF00FFu,0x0000FFFFu,
  0xA5A5A5A5u,0xC3C3C3C3u,0x96969696u,0x69696969u,0x5A5A5A5Au,
};
// pats[0..1] structured, pats[2]=jmaj (exp=ONE), pats[3]=jmin (exp=ZERO),
// pats[4..] random
static vector<Pat> make_pats(int X, int n_rand) {
  vector<Pat> pats; Pat pa, pb, pj, pm;
  for (int k = 0; k < X; k++) { pa.I[k] = BASEW[k]; pb.I[k] = ~BASEW[k]; }
  pa.exp = majx(pa.I, X); pb.exp = majx(pb.I, X);
  pats.push_back(pa); pats.push_back(pb);
  int hi = (X + 1) / 2;
  for (int k = 0; k < X; k++) { pj.I[k] = (k < hi) ? ONE : ZERO; pm.I[k] = (k < hi - 1) ? ONE : ZERO; }
  pj.exp = ONE; pm.exp = ZERO;
  pats.push_back(pj); pats.push_back(pm);
  srand(98765 + X);
  for (int r = 0; r < n_rand; r++) { Pat pr;
    for (int k = 0; k < X; k++)
      pr.I[k] = ((uint32_t)rand() << 17) ^ ((uint32_t)rand() << 3) ^ (uint32_t)rand();
    pr.exp = majx(pr.I, X); pats.push_back(pr); }
  return pats;
}

struct Calib { int s_id, bank; uint32_t Rf, Rs; vector<uint32_t> open; string src; };
static vector<Calib> readc_all(const string& paths_csv, int wb, int ws /* -1 = any */) {
  vector<Calib> out; set<uint64_t> seen;
  stringstream ps(paths_csv); string path;
  while (getline(ps, path, ',')) {
    ifstream f(path); string line;
    while (getline(f, line)) { if (line.empty()||line[0]=='#') continue; istringstream is(line);
      Calib c; if(!(is>>c.s_id>>c.bank>>c.Rf>>c.Rs)) continue; uint32_t v; while(is>>v) c.open.push_back(v);
      if (c.open.size()!=16 || c.bank!=wb) continue;
      if (ws >= 0 && c.s_id != ws) continue;
      uint64_t key = ((uint64_t)c.Rf<<32)|c.Rs;
      if (seen.count(key)) continue; seen.insert(key);
      size_t sl = path.find_last_of('/');
      c.src = (sl==string::npos)?path:path.substr(sl+1);
      out.push_back(c);
    }
  }
  return out;
}

static void run_cfg(SoftMCPlatform& pf, int bank, const Calib& c, const Cfg& cf,
                    const vector<Pat>& pats, int trials,
                    vector<int>& ok, int& runs, vector<double>& pat_pct) {
  ok.assign(2048, 0); runs = 0; pat_pct.assign(pats.size(), 0.0);
  vector<uint32_t> frows;
  for (int s = 0; s < cf.ns; s++) if ((cf.frac_mask >> s) & 1u) frows.push_back(c.open[s]);
  for (size_t pi = 0; pi < pats.size(); pi++) {
    const Pat& pt = pats[pi];
    vector<uint32_t> patterns = buildN(cf, pt.I);
    long pmatch = 0;
    for (int tr = 0; tr < trials; tr++) {
      Program p = test_prog(bank, patterns, c.open, frows, c.Rf, c.Rs,
                            cf.nf, cf.tf, cf.t12, cf.t23, 1);
      pf.execute(p);
      uint8_t row[8192];
      if (pf.receiveData(row, 8192) != 8192) { fprintf(stderr,"recv fail (%s)\n", cf.name); _exit(5); }
      runs++;
      for (int s = 0; s < 2048; s++) { uint32_t g; memcpy(&g, &row[s*4], 4);
        if (g == pt.exp) { ok[s]++; pmatch++; } }
    }
    pat_pct[pi] = 100.0 * pmatch / ((double)trials * 2048);
  }
}
static void tally(const vector<int>& ok, int runs, int& strict, int& soft, double& mr) {
  strict = 0; soft = 0; mr = 0;
  for (int s = 0; s < 2048; s++) {
    if (ok[s] == runs) strict++;
    if (ok[s] >= (int)(0.95 * runs)) soft++;
    mr += 100.0 * ok[s] / runs;
  }
  mr /= 2048;
}
static void write_colmask(const char* out, int bender, int bank, const Calib& c,
                          const char* polname, int q0, int q2, int npat, int trials,
                          int runs, int strict, int soft, double mr, const vector<int>& ok) {
  FILE* f = fopen(out, "w");
  if (!f) { fprintf(stderr,"cannot write %s\n", out); _exit(4); }
  time_t now = time(0); char ts[64]; strftime(ts,sizeof ts,"%Y-%m-%d %H:%M",localtime(&now));
  fprintf(f,"# reliable MAJ5 columns  bender=%d bank=%d s_id=%d  %d/2048\n",bender,bank,c.s_id,strict);
  fprintf(f,"# tuple Rf=%u Rs=%u  calib=%s\n",c.Rf,c.Rs,c.src.c_str());
  fprintf(f,"# open_rows:"); for (uint32_t r : c.open) fprintf(f," %u",r); fprintf(f,"\n");
  fprintf(f,"# policy=%s (quick: Z0=%d Z2=%d)  layout=ADR-001 3x5+1 interleaved\n",polname,q0,q2);
  fprintf(f,"# depth: %d patterns x %d trials = %d runs, strict criterion; soft95=%d mean=%.2f%%\n",npat,trials,runs,soft,mr);
  fprintf(f,"# tool=majx-screen-exe v3 (Task S2) %s\n",ts);
  for (int s = 0; s < 2048; s++) if (ok[s] == runs) fprintf(f,"%d\n",s);
  fclose(f);
}
static void sweep_table(SoftMCPlatform& pf, int bank, const Calib& c,
                        const Cfg* tbl, int n, int n_rand, int trials) {
  printf("name,X,ns,nf,tf,t12,t23,strict,soft95,meanrate_pct,jmaj_pct,jmin_pct,runs\n");
  int curX = -1; vector<Pat> pats;
  vector<int> ok; int runs; vector<double> pp;
  for (int ci = 0; ci < n; ci++) {
    const Cfg& cf = tbl[ci];
    if (cf.X != curX) { curX = cf.X; pats = make_pats(curX, n_rand); }
    int strict, soft; double mr;
    run_cfg(pf, bank, c, cf, pats, trials, ok, runs, pp);
    tally(ok, runs, strict, soft, mr);
    printf("%s,%d,%d,%d,%d,%d,%d,%d,%d,%.3f,%.2f,%.2f,%d\n",
           cf.name,cf.X,cf.ns,cf.nf,cf.tf,cf.t12,cf.t23,strict,soft,mr,pp[2],pp[3],runs);
    fprintf(stderr,"[mxs] %-14s X=%2d ns=%d nf=%2d t=%d,%d  strict=%4d/2048 soft95=%4d mean=%6.2f%%  jmaj=%6.2f%% jmin=%6.2f%% (%d runs)\n",
            cf.name,cf.X,cf.ns,cf.nf,cf.t12,cf.t23,strict,soft,mr,pp[2],pp[3],runs);
    fflush(stdout);
  }
  fprintf(stderr,"[mxs] DONE\n");
}

int main(int argc, char** argv) {
  if (argc < 5) {
    fprintf(stderr,"Usage: %s <bender> <calib|SUB> <bank> <s_id|A_local|-1> [sweep|sweep32|colmask|select] [n_rand] [trials] [out]\n",argv[0]);
    return 1; }
  int bender=atoi(argv[1]), bank=atoi(argv[3]), sid=atoi(argv[4]);
  const char* mode = (argc>5)?argv[5]:"sweep";
  bool colmask = !strcmp(mode,"colmask"), select = !strcmp(mode,"select");
  bool sweep32 = !strcmp(mode,"sweep32");
  int n_rand=(argc>6)?atoi(argv[6]):((colmask||select)?18:12);
  int trials=(argc>7)?atoi(argv[7]):((colmask||select)?25:6);
  const char* out=(argc>8)?argv[8]:nullptr;
  if ((colmask||select) && !out) { fprintf(stderr,"%s mode needs <out>\n",mode); return 1; }
  const Cfg CZ0 = {"m5_Z0", 5,16, 0x0000, 0x0000, 0,0, 0,0};
  const Cfg CZ2 = {"m5_Z2", 5,16, 0x0000, 0x0001, 2,0, 0,0};

  if (sweep32) {
    uint32_t SUB = (uint32_t)strtoul(argv[2],0,10);
    uint32_t A_local = (uint32_t)sid;
    static const uint32_t UNITS[5] = {1,2,8,32,128};   // one per group + bit0
    uint32_t d = 0; for (uint32_t u : UNITS) d |= u;   // 171
    Calib c; c.s_id = -32; c.bank = bank; c.src = "law-coset";
    for (int i = 0; i < 32; i++) {
      uint32_t S = 0;
      for (int j = 0; j < 5; j++) if ((i >> j) & 1) S |= UNITS[j];
      c.open.push_back(SUB + (A_local ^ S));
    }
    c.Rf = SUB + A_local; c.Rs = SUB + (A_local ^ d);
    fprintf(stderr,"[mxs] SWEEP32 bender=%d bank=%d SUB=%u A_local=%u d=%u Rf=%u Rs=%u rows[%u..%u]\n",
            bender,bank,SUB,A_local,d,c.Rf,c.Rs,c.open[0],c.open[31]);
    SoftMCPlatform pf(bender);
    if (pf.init()!=SOFTMC_SUCCESS){ fprintf(stderr,"init fail\n"); return 3; }
    pf.reset_fpga();
    srand(12345);
    sweep_table(pf, bank, c, CFGS32, NCFG32, n_rand, trials);
    _exit(0);
  }

  vector<Calib> cands = readc_all(argv[2], bank, select ? -1 : sid);
  if (cands.empty()){ fprintf(stderr,"no calib bank=%d s_id=%d in %s\n",bank,sid,argv[2]); return 2; }

  SoftMCPlatform pf(bender);
  if (pf.init()!=SOFTMC_SUCCESS){ fprintf(stderr,"init fail\n"); return 3; }
  pf.reset_fpga();
  srand(12345);
  vector<int> ok; int runs; vector<double> pp;

  if (select) {
    fprintf(stderr,"[mxs] SELECT bender=%d bank=%d: %zu candidate tuples from %s\n",
            bender,bank,cands.size(),argv[2]);
    vector<Pat> pq = make_pats(5, 8);
    string rank_path = string(out) + "_ranking.csv";
    FILE* rk = fopen(rank_path.c_str(), "w");
    if (!rk) { fprintf(stderr,"cannot write %s\n", rank_path.c_str()); return 4; }
    fprintf(rk,"calib,s_id,Rf,Rs,quick_Z0,quick_Z2,best\n");
    int best_i = -1, best_q = -1, best_q0 = 0, best_q2 = 0;
    for (size_t i = 0; i < cands.size(); i++) {
      int s0,s2,soft; double mr;
      run_cfg(pf, bank, cands[i], CZ0, pq, 2, ok, runs, pp); tally(ok, runs, s0, soft, mr);
      run_cfg(pf, bank, cands[i], CZ2, pq, 2, ok, runs, pp); tally(ok, runs, s2, soft, mr);
      int q = (s0 > s2) ? s0 : s2;
      fprintf(rk,"%s,%d,%u,%u,%d,%d,%s\n",cands[i].src.c_str(),cands[i].s_id,
              cands[i].Rf,cands[i].Rs,s0,s2,(s0>=s2)?"Z0":"Z2");
      fprintf(stderr,"[mxs] cand %2zu/%zu s%d Rf=%u Rs=%u  Z0=%4d Z2=%4d\n",
              i+1,cands.size(),cands[i].s_id,cands[i].Rf,cands[i].Rs,s0,s2);
      if (q > best_q) { best_q = q; best_i = (int)i; best_q0 = s0; best_q2 = s2; }
    }
    fclose(rk);
    const Calib& w = cands[best_i];
    const Cfg& wp = (best_q0 >= best_q2) ? CZ0 : CZ2;
    fprintf(stderr,"[mxs] WINNER s%d Rf=%u Rs=%u (%s, quick %d) — deep screen…\n",
            w.s_id,w.Rf,w.Rs,wp.name,best_q);
    vector<Pat> pd = make_pats(5, n_rand);
    int strict, soft; double mr;
    run_cfg(pf, bank, w, wp, pd, trials, ok, runs, pp); tally(ok, runs, strict, soft, mr);
    string cm_path = string(out) + ".txt";
    write_colmask(cm_path.c_str(), bender, bank, w, wp.name, best_q0, best_q2,
                  (int)pd.size(), trials, runs, strict, soft, mr, ok);
    printf("select,%d,%d,%d,%u,%u,%s,%d,%d,%.3f\n",bender,bank,w.s_id,w.Rf,w.Rs,wp.name,strict,soft,mr);
    fprintf(stderr,"[mxs] colmask %s: %d/2048 strict (soft95=%d mean=%.2f%%) -> %s\n",
            wp.name,strict,soft,mr,cm_path.c_str());
    _exit(0);
  }

  const Calib& c = cands[0];
  fprintf(stderr,"[mxs] bender=%d bank=%d s_id=%d Rf=%u Rs=%u calib=%s mode=%s n_rand=%d trials=%d\n",
          bender,bank,c.s_id,c.Rf,c.Rs,c.src.c_str(),mode,n_rand,trials);

  if (colmask) {
    vector<Pat> pq = make_pats(5, 8);
    int s0,s2,sf,soft; double mr;
    run_cfg(pf, bank, c, CZ0, pq, 2, ok, runs, pp); tally(ok, runs, s0, soft, mr);
    run_cfg(pf, bank, c, CZ2, pq, 2, ok, runs, pp); tally(ok, runs, s2, soft, mr);
    const Cfg& win = (s0 >= s2) ? CZ0 : CZ2;
    fprintf(stderr,"[mxs] policy pick: Z0=%d Z2=%d -> %s\n", s0, s2, win.name);
    vector<Pat> pd = make_pats(5, n_rand);
    run_cfg(pf, bank, c, win, pd, trials, ok, runs, pp); tally(ok, runs, sf, soft, mr);
    write_colmask(out, bender, bank, c, win.name, s0, s2, (int)pd.size(), trials, runs, sf, soft, mr, ok);
    printf("colmask,%d,%d,%d,%u,%u,%s,%d,%d,%.3f\n",bender,bank,c.s_id,c.Rf,c.Rs,win.name,sf,soft,mr);
    fprintf(stderr,"[mxs] colmask %s: %d/2048 strict (soft95=%d mean=%.2f%%) -> %s\n",win.name,sf,soft,mr,out);
    _exit(0);
  }

  sweep_table(pf, bank, c, CFGS, NCFG, n_rand, trials);
  _exit(0);
}
