// Multi-bit dual-track ripple adder with REAL MAJ5 + ZERO policy
// (2026-07-17, Task S4 — the June chain-collapse retest, on the S2 per-bank
// tuples/policies). Scientific question: does a faithful dual-track ripple
// adder built from real (noisy) MAJ5 sum + MAJ3 carry ACCUMULATE error
// across ripple stages, or does each bit compute independently at the
// single-op reliability?  June saw a chain collapse under the frac'd-ONE
// recipe; this reruns it under the addendum-8 ZERO reference policy.
//
// Per bit i (MVDRAM §II-C dual track):
//   cbar_{i+1} = MAJ3(¬a_i, ¬b_i, cbar_i)            [carry track, De Morgan]
//   s_i        = MAJ5(a_i, b_i, c_i, cbar_{i+1}, cbar_{i+1})     [sum output]
//   c_{i+1}    = MAJ3(a_i, b_i, c_i)                 [carry track]
// Each MAJ result is harvested from the tuple's read row O(0) (where the
// APA majority reliably lands). Primary inputs a,b,¬a,¬b are host-written
// (they ARE host-known — same as MVDRAM loading matrix/vector). Sum bits
// are OUTPUTS → read to host. The carry pair (c_i, cbar_i) are the in-DRAM
// intermediates: each is READ BACK from DRAM (O(0)) and fed into the next
// stage, so a DRAM-corrupted carry propagates forward faithfully.
// FAITHFULNESS CAVEAT (stated in the writeup): carries round-trip through
// the host register file between stages; a fully in-DRAM chain keeps them
// resident and fans them into the 5 operand slots by coset broadcast — that
// fan-out is Task M3, not measured here. What IS measured here: the real
// MAJ5/MAJ3 error at each ripple position and whether it compounds.
//
// Tuple + policy + reliable columns come from an S2 colmask file
// (colmasks_maj5_zero_2026_07_17/maj5_best_*.txt), full provenance in-band.
// Argv: <bender> <colmask_path> <bank> [bits=4] [trials=50] [seed=7]
#include "instruction.h"
#include "prog.h"
#include "platform.h"
#include "../util.h"
#include "lattice_alloc.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
using namespace std;

static const int CHUNK_COLS[3] = {43, 43, 42};
static SoftMCPlatform* PF = nullptr;
static int BANK = 0;

// write a full 2048-word row (per-column payload) into `row`
static void pcwrite(uint32_t row, const uint32_t* seg) {
  int cs = 0;
  for (int ch = 0; ch < 3; ch++) {
    int n = CHUNK_COLS[ch]; const uint32_t* cd = seg + cs * 16;
    Program p;
    p.add_inst(SMC_LI(8, CASR)); p.add_inst(SMC_LI(BANK, BAR));
    p.add_inst(SMC_LI(row, RAR)); p.add_inst(SMC_LI(cs * 8, CAR));
    p.add_below(PRE(BAR, 0, 0)); p.add_below(ACT(BAR, 0, RAR, 0));
    for (int k = 0; k < n; k++) {
      const uint32_t* sl = cd + k * 16;
      for (int s = 0; s < 16; s++) { p.add_inst(SMC_LI(sl[s], PATTERN_REG)); p.add_inst(SMC_LDWD(PATTERN_REG, s)); }
      p.add_below(WRITE(BAR, CAR, 1)); p.add_inst(SMC_SLEEP(8));
    }
    p.add_inst(SMC_SLEEP(8)); p.add_below(PRE(BAR, 0, 0));
    p.add_inst(SMC_SLEEP(4)); p.add_inst(SMC_END());
    PF->execute(p); cs += n;
  }
}
static void wrconst(uint32_t row, uint32_t val) {
  Program p;
  p.add_inst(SMC_LI(8, CASR)); p.add_inst(SMC_LI(BANK, BAR));
  p.add_inst(SMC_LI(128, NUM_COLS_REG));
  p.add_below(PRE(BAR, 0, 0));
  p.add_below(wrRow_immediate_label(BAR, row, val, 1));
  p.add_inst(SMC_SLEEP(6)); p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_END());
  PF->execute(p);
}
static Program frac_builder(int t_frac, int r) {
  Program p; p.add_inst(all_nops());
  p.add_inst(SMC_LI(r, RF_REG));
  int nc = 2 + t_frac; nc += 4 - (nc % 4);
  Mininst q[nc]; for (int i = 0; i < nc; i++) q[i] = SMC_NOP();
  q[0] = SMC_ACT(BAR, 0, RF_REG, 0); q[t_frac + 1] = SMC_PRE(BAR, 0, 0);
  for (int i = 0; i < nc; i += 4) p.add_inst(q[i], q[i+1], q[i+2], q[i+3]);
  return p;
}
// one MAJ: reference into slot0, frac nf times, doubleACT, harvest O(0)
static void fire_maj(uint32_t Rf, uint32_t Rs, uint32_t ref_row, uint32_t ref_val,
                     int nf, uint8_t* out) {
  wrconst(ref_row, ref_val);
  Program p;
  p.add_inst(SMC_LI(8, CASR)); p.add_inst(SMC_LI(BANK, BAR));
  p.add_inst(SMC_LI(128, NUM_COLS_REG));
  p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_SLEEP(6));
  for (int j = 0; j < nf; j++) {
    p.add_inst(SMC_SLEEP(6)); p.add_below(frac_builder(0, ref_row)); p.add_inst(SMC_SLEEP(6));
  }
  p.add_inst(SMC_SLEEP(6)); p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_SLEEP(6));
  p.add_below(doubleACT(0, 0, Rf, Rs));
  p.add_inst(SMC_SLEEP(6)); p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_SLEEP(6));
  p.add_below(rdRow_immediate(BAR, ref_row));
  p.add_inst(all_nops()); p.add_inst(all_nops());
  p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_END());
  PF->execute(p);
  if (PF->receiveData(out, 8192) != 8192) { fprintf(stderr, "recv fail\n"); _exit(5); }
}

struct Cm { vector<uint32_t> open; uint32_t Rf=0, Rs=0; int nf=0; int sid=0; vector<int> cols; string pol; };
static bool read_colmask(const char* path, Cm& c) {
  ifstream f(path); if (!f) return false;
  string line;
  while (getline(f, line)) {
    if (line.empty()) continue;
    if (line[0] == '#') {
      if (line.find("open_rows:") != string::npos) {
        istringstream is(line.substr(line.find(':') + 1));
        uint32_t v; while (is >> v) c.open.push_back(v);
      } else if (line.find("tuple Rf=") != string::npos) {
        sscanf(line.c_str(), "# tuple Rf=%u Rs=%u", &c.Rf, &c.Rs);
      } else if (line.find("s_id=") != string::npos && !c.sid) {
        size_t p = line.find("s_id="); sscanf(line.c_str()+p, "s_id=%d", &c.sid);
      } else if (line.find("policy=m5_Z2") != string::npos) { c.nf = 2; c.pol="Z2"; }
      else if (line.find("policy=m5_Z0") != string::npos) { c.nf = 0; c.pol="Z0"; }
      continue;
    }
    c.cols.push_back(atoi(line.c_str()));
  }
  return c.open.size() == 16 && c.Rf && !c.cols.empty();
}
static uint32_t maj3w(uint32_t a, uint32_t b, uint32_t c) { return (a&b)|(a&c)|(b&c); }
static uint32_t maj5w(uint32_t a, uint32_t b, uint32_t c, uint32_t d, uint32_t e) {
  uint32_t out = 0;
  for (int t = 0; t < 32; t++) {
    int n = ((a>>t)&1)+((b>>t)&1)+((c>>t)&1)+((d>>t)&1)+((e>>t)&1);
    if (n >= 3) out |= 1u << t;
  }
  return out;
}

int main(int argc, char** argv) {
  if (argc < 4) { fprintf(stderr,"Usage: %s <bender> <colmask_path> <bank> [bits=4] [trials=50] [seed=7]\n",argv[0]); return 1; }
  int bender = atoi(argv[1]); BANK = atoi(argv[3]);
  int BITS = (argc>4)?atoi(argv[4]):4;
  int TRIALS = (argc>5)?atoi(argv[5]):50;
  unsigned seed = (argc>6)?(unsigned)atoi(argv[6]):7;
  if (BITS < 1 || BITS > 16) { fprintf(stderr,"bits out of range\n"); return 1; }

  Cm cm;
  if (!read_colmask(argv[2], cm)) { fprintf(stderr,"bad colmask file %s\n", argv[2]); return 2; }
  // policy override: PIM_CHAIN_POLICY=Z0|Z2|ONE3 forces the MAJ5 reference
  // (ONE3 = SiMRA frac'd-ONE recipe, the June-collapse config) for A/B.
  uint32_t ref_val = 0x00000000u;   // ZERO reference default
  const char* pov = getenv("PIM_CHAIN_POLICY");
  if (pov) {
    if (!strcmp(pov,"Z0")) { cm.nf=0; cm.pol="Z0"; ref_val=0; }
    else if (!strcmp(pov,"Z2")) { cm.nf=2; cm.pol="Z2"; ref_val=0; }
    else if (!strcmp(pov,"ONE3")) { cm.nf=3; cm.pol="ONE3"; ref_val=0xFFFFFFFFu; }
  }
  TupleAlloc TA; TA.open = cm.open; TA.Rf = cm.Rf; TA.Rs = cm.Rs;
  bool fit = TA.fit();
  fprintf(stderr,"[chain] tuple s%d Rf=%u Rs=%u policy=%s(nf=%d) fit=%s block=%u cols=%zu\n",
          cm.sid, cm.Rf, cm.Rs, cm.pol.c_str(), cm.nf, fit?"yes":"NO", TA.block, cm.cols.size());

  // slot maps (list positions 1..15; slot 0 = reference/read row)
  const int a3[5]={1,4,7,10,13}, b3[5]={2,5,8,11,14}, c3[5]={3,6,9,12,15};
  const int a5[3]={1,6,11}, b5[3]={2,7,12}, cc5[3]={3,8,13}, d5[3]={4,9,14}, e5[3]={5,10,15};
  auto O = [&](int j){ return cm.open[j]; };

  SoftMCPlatform pf(bender);
  if (pf.init()!=SOFTMC_SUCCESS){ fprintf(stderr,"init fail\n"); return 3; }
  pf.reset_fpga(); PF = &pf;
  srand(seed);

  vector<vector<uint32_t>> A(BITS, vector<uint32_t>(2048)), B(BITS, vector<uint32_t>(2048));
  vector<vector<uint32_t>> Sexp(BITS, vector<uint32_t>(2048));
  vector<uint32_t> Cexp(2048);
  uint8_t buf[8192];
  vector<uint32_t> na(2048), nb(2048), cprev(2048), cbar(2048);
  vector<long> s_err(BITS, 0), s_dram_vs_host_maj(BITS,0);
  vector<long> cbar_err(BITS,0), carry_err(BITS,0);
  long cout_err = 0, all_ok = 0, ncol = 0;

  for (int tr = 0; tr < TRIALS; tr++) {
    for (int i = 0; i < BITS; i++)
      for (int s = 0; s < 2048; s++) {
        A[i][s] = ((uint32_t)rand()<<17)^((uint32_t)rand()<<3)^(uint32_t)rand();
        B[i][s] = ((uint32_t)rand()<<17)^((uint32_t)rand()<<3)^(uint32_t)rand();
      }
    // host reference
    for (int s = 0; s < 2048; s++) {
      uint32_t c = 0;
      for (int i = 0; i < BITS; i++) {
        uint32_t cb1 = maj3w(~A[i][s], ~B[i][s], ~c);
        Sexp[i][s] = maj5w(A[i][s], B[i][s], c, cb1, cb1);
        c = maj3w(A[i][s], B[i][s], c);
      }
      Cexp[s] = c;
    }
    // DRAM chain: c_0 = 0, cbar_0 = 1
    for (int s = 0; s < 2048; s++) { cprev[s] = 0u; cbar[s] = 0xFFFFFFFFu; }

    for (int i = 0; i < BITS; i++) {
      for (int s = 0; s < 2048; s++) { na[s] = ~A[i][s]; nb[s] = ~B[i][s]; }
      // stage 1: cbar_{i+1} = MAJ3(¬a, ¬b, cbar_i)
      for (int k=0;k<5;k++){ pcwrite(O(a3[k]), na.data()); pcwrite(O(b3[k]), nb.data()); pcwrite(O(c3[k]), cbar.data()); }
      fire_maj(cm.Rf, cm.Rs, O(0), 0x00000000u, 0, buf);
      vector<uint32_t> cbar_next(2048);
      for (int s=0;s<2048;s++) memcpy(&cbar_next[s], &buf[s*4], 4);
      // stage 2: s_i = MAJ5(a, b, c_i, cbar_{i+1}, cbar_{i+1})
      for (int k=0;k<3;k++){ pcwrite(O(a5[k]),A[i].data()); pcwrite(O(b5[k]),B[i].data()); pcwrite(O(cc5[k]),cprev.data());
                             pcwrite(O(d5[k]),cbar_next.data()); pcwrite(O(e5[k]),cbar_next.data()); }
      fire_maj(cm.Rf, cm.Rs, O(0), ref_val, cm.nf, buf);
      // count s_i error vs host reference (final adder output)
      for (int ci : cm.cols) { uint32_t g; memcpy(&g,&buf[ci*4],4); if (g != Sexp[i][ci]) s_err[i]++; }
      // stage 3: c_{i+1} = MAJ3(a, b, c_i)
      for (int k=0;k<5;k++){ pcwrite(O(a3[k]),A[i].data()); pcwrite(O(b3[k]),B[i].data()); pcwrite(O(c3[k]),cprev.data()); }
      fire_maj(cm.Rf, cm.Rs, O(0), 0x00000000u, 0, buf);
      vector<uint32_t> cnext(2048);
      for (int s=0;s<2048;s++) memcpy(&cnext[s], &buf[s*4], 4);
      // per-stage carry-track diagnostics vs the TRUE host carry at this bit
      for (int ci : cm.cols) {
        uint32_t hc=0; for(int ii=0;ii<i;ii++) hc=maj3w(A[ii][ci],B[ii][ci],hc);
        uint32_t hcb=maj3w(~A[i][ci],~B[i][ci],~hc);
        uint32_t hc1=maj3w(A[i][ci],B[i][ci],hc);
        if (cbar_next[ci]!=hcb) cbar_err[i]++;
        if (cnext[ci]!=hc1) carry_err[i]++;
      }
      cbar.swap(cbar_next); cprev.swap(cnext);
    }
    // final carry
    for (int ci : cm.cols) { if (cprev[ci]!=Cexp[ci]) cout_err++; }
    // whole-word all-correct
    for (int ci : cm.cols) {
      bool ok = (cprev[ci]==Cexp[ci]);
      // (s_i already counted; recompute all-ok needs per-bit — approximate via error flags below)
      (void)ok;
    }
    ncol += cm.cols.size();
    if ((tr+1)%10==0) fprintf(stderr,"[chain] trial %d/%d\n", tr+1, TRIALS);
  }

  printf("tuple_s,bits,trials,cols,policy,cout_err_pct\n");
  printf("%d,%d,%d,%zu,%s,%.4f\n", cm.sid, BITS, TRIALS, cm.cols.size(), cm.pol.c_str(), 100.0*cout_err/ncol);
  printf("bit,s_err_pct,cbar_err_pct,carry_err_pct\n");
  for (int i=0;i<BITS;i++)
    printf("%d,%.4f,%.4f,%.4f\n", i, 100.0*s_err[i]/ncol, 100.0*cbar_err[i]/ncol, 100.0*carry_err[i]/ncol);
  fprintf(stderr,"[chain] === s%d %d-bit, %s policy, %ld (col,trial) samples ===\n", cm.sid, BITS, cm.pol.c_str(), ncol);
  fprintf(stderr,"[chain] final carry-out error: %.4f%%\n", 100.0*cout_err/ncol);
  for (int i=0;i<BITS;i++)
    fprintf(stderr,"[chain]  bit%-2d  sum %.4f%%   cbar %.4f%%   carry %.4f%%\n",
            i, 100.0*s_err[i]/ncol, 100.0*cbar_err[i]/ncol, 100.0*carry_err[i]/ncol);
  _exit(0);
}
