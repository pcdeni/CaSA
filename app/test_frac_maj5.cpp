// Frac × MAJ5 reliable-column sweep (2026-07-17) — THE MVDRAM frac question.
// MVDRAM §VII: "Frac operations [34] and calibration techniques [48] to
// increase the number of reliable columns, which achieves error-free
// computation." Our MAJ5 single-op reliable-column count is strongly
// subarray-dependent (s72: 28/2048; s86: ~1784/2048). This tool reruns the
// exact mvdram-maj5 screen (ADR-001 3+3+3+3+3+1 layout, identical
// instruction sequence) while sweeping the frac conditioning of the
// reference row:
//   n_frac ∈ {0,1,2,3,4,6,8,12,16} × t_frac ∈ {0,2} × init ∈ {ONE, ZERO}
// Metrics per config:
//   strict   — segments correct in ALL patterns×trials (the colmask criterion)
//   soft95   — segments correct in ≥95% of runs (calibratable-with-voting)
//   meanrate — mean per-segment correctness rate
// Argv: <bender> <calib_file> <bank> <s_id> [n_pat_rand=14] [n_trials=8]
#include "instruction.h"
#include "prog.h"
#include "platform.h"
#include "../util.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
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
                         uint32_t r_frac_addr, uint32_t Rf, uint32_t Rs,
                         uint32_t n_frac, uint32_t t_frac, uint32_t num_iter) {
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
    for (uint32_t j = 0; j < n_frac; j++) {
      p.add_inst(SMC_SLEEP(6));
      p.add_below(frac_builder(t_frac, r_frac_addr));
      p.add_inst(SMC_SLEEP(6));
    }
    p.add_inst(SMC_SLEEP(6)); p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_SLEEP(6));
    p.add_below(doubleACT(0, 0, Rf, Rs));
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
static vector<uint32_t> build_16row_maj5(const uint32_t I[5], uint32_t frac_init) {
  vector<uint32_t> lst(16, 0);
  for (int r = 0; r < 3; r++)
    for (int i = 0; i < 5; i++)
      lst[1 + r * 5 + i] = I[i];
  lst[0] = frac_init;
  return lst;
}
static uint32_t maj5(uint32_t a, uint32_t b, uint32_t c, uint32_t d, uint32_t e) {
  return (a&b&c)|(a&b&d)|(a&b&e)|(a&c&d)|(a&c&e)|(a&d&e)
       | (b&c&d)|(b&c&e)|(b&d&e)|(c&d&e);
}
struct Calib { int s_id, bank; uint32_t Rf, Rs; vector<uint32_t> open; };
static Calib readc(const string& path, int wb, int ws) {
  ifstream f(path); string line;
  while (getline(f, line)) { if (line.empty()||line[0]=='#') continue; istringstream is(line);
    Calib c; if(!(is>>c.s_id>>c.bank>>c.Rf>>c.Rs)) continue; uint32_t v; while(is>>v) c.open.push_back(v);
    if (c.open.size()==16 && c.bank==wb && c.s_id==ws) return c; }
  return Calib{-1,0,0,0,{}};
}

int main(int argc, char** argv) {
  if (argc < 5) { fprintf(stderr,"Usage: %s <bender> <calib> <bank> <s_id> [n_rand=14] [trials=8]\n",argv[0]); return 1; }
  int bender=atoi(argv[1]), bank=atoi(argv[3]), sid=atoi(argv[4]);
  int n_rand=(argc>5)?atoi(argv[5]):14, trials=(argc>6)?atoi(argv[6]):8;
  Calib c = readc(argv[2], bank, sid);
  if (c.s_id<0){ fprintf(stderr,"no calib bank=%d s_id=%d in %s\n",bank,sid,argv[2]); return 2; }
  fprintf(stderr,"[fm5] bender=%d bank=%d s_id=%d Rf=%u Rs=%u  n_rand=%d trials=%d\n",
          bender,bank,sid,c.Rf,c.Rs,n_rand,trials);

  // pattern set: 2 structured + n_rand random (same generator as mvdram-maj5)
  struct Pat { uint32_t I[5]; uint32_t exp; };
  vector<Pat> pats;
  { uint32_t A[5] = {0xAAAAAAAAu,0xCCCCCCCCu,0xF0F0F0F0u,0xFF00FF00u,0xFFFF0000u};
    Pat pa; memcpy(pa.I,A,sizeof A); pa.exp=maj5(A[0],A[1],A[2],A[3],A[4]); pats.push_back(pa);
    Pat pb; for(int k=0;k<5;k++) pb.I[k]=~A[k]; pb.exp=maj5(pb.I[0],pb.I[1],pb.I[2],pb.I[3],pb.I[4]); pats.push_back(pb); }
  srand(98765);
  for (int r = 0; r < n_rand; r++) { Pat pr;
    for (int k=0;k<5;k++) pr.I[k]=((uint32_t)rand()<<17)^((uint32_t)rand()<<3)^(uint32_t)rand();
    pr.exp=maj5(pr.I[0],pr.I[1],pr.I[2],pr.I[3],pr.I[4]); pats.push_back(pr); }

  SoftMCPlatform pf(bender);
  if (pf.init()!=SOFTMC_SUCCESS){ fprintf(stderr,"init fail\n"); return 3; }
  pf.reset_fpga();
  srand(12345);

  const int NSEG = 2048;
  printf("n_frac,t_frac,frac_init,strict,soft95,meanrate_pct\n");
  int NF[] = {0,1,2,3,4,6,8,12,16};
  for (uint32_t fi : {ONE, ZERO}) {
    for (int tf : {0,2}) {
      for (int nf : NF) {
        vector<int> ok(NSEG,0); int runs=0;
        for (auto& pt : pats) {
          vector<uint32_t> patterns = build_16row_maj5(pt.I, fi);
          for (int tr=0; tr<trials; tr++) {
            Program p = test_prog(bank, patterns, c.open, c.open[0],
                                  c.Rf, c.Rs, nf, tf, 1);
            pf.execute(p);
            uint8_t row[8192];
            if (pf.receiveData(row,8192)!=8192){ fprintf(stderr,"recv fail\n"); return 5; }
            runs++;
            for (int s=0;s<NSEG;s++){ uint32_t g; memcpy(&g,&row[s*4],4); ok[s]+= (g==pt.exp); }
          }
        }
        int strict=0, soft=0; double mr=0;
        for (int s=0;s<NSEG;s++){
          if (ok[s]==runs) strict++;
          if (ok[s]>=(int)(0.95*runs)) soft++;
          mr += 100.0*ok[s]/runs;
        }
        printf("%d,%d,%s,%d,%d,%.3f\n", nf, tf, fi==ONE?"ONE":"ZERO", strict, soft, mr/NSEG);
        fprintf(stderr,"[fm5] n_frac=%2d t_frac=%d init=%s  strict=%4d/2048 soft95=%4d  meanrate=%.2f%%\n",
                nf, tf, fi==ONE?"ONE ":"ZERO", strict, soft, mr/NSEG);
        fflush(stdout);
      }
    }
  }
  fprintf(stderr,"[fm5] DONE\n");
  _exit(0);
}
