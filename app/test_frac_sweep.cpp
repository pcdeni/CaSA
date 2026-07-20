// Frac-conditioning sensitivity sweep (2026-07-17). The SiMRA MajOperations
// sweeps hardcode n_frac_times=3, t_frac=0 and characterize everything AROUND
// that fixed recipe — they never test frac itself. This tool replays the
// EXACT calibration MAJ3 sequence (test_maj3_smoke / MajOperations test.cpp,
// verbatim builders) on a calibrated 16-row tuple and sweeps the frac knobs:
//
//   n_frac_times ∈ {0,1,2,3,4,6,8,12,16}   (# ACT-PRE conditioning pulses)
//   t_frac       ∈ {0,1,2,3}               (ACT→PRE gap: how far activation
//                                            proceeds before the interrupt)
//   frac init    ∈ {ONE, ZERO}             (FracDRAM Fig 7: conditioning from
//                                            all-1s vs all-0s lands on the two
//                                            sides of the Vdd/2 fractional curve)
//   retention    : optional inserted delay between frac and the MAJ doubleACT
//
// Observable per config: MAJ3 correctness (fraction of 2048 segments bit-exact
// vs host maj3(A,B,C)) averaged over random (A,B,C) trials × iterations. This
// is FracDRAM's own MAJ3-bias verification (their §V) turned into a per-module
// reliability-vs-conditioning surface — the prerequisite for the MVDRAM MAJ5
// "error-free via Frac+calibration" idea.
//
// Argv: <bender> <calib_file> <bank> <s_id> [trials=16] [num_iter=4]
#include "instruction.h"
#include "prog.h"
#include "platform.h"
#include "../util.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
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
  Mininst q_inst[num_cmd];
  for (int i = 0; i < num_cmd; i++) q_inst[i] = SMC_NOP();
  q_inst[0] = SMC_ACT(bank_reg, 0, R_FRAC_REG, 0);
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
  p.add_inst(SMC_LI(8, CASR)); p.add_inst(SMC_LI(1, BASR));
  p.add_inst(SMC_LI(1, RASR)); p.add_inst(SMC_LI(128, NUM_COLS_REG));
  p.add_inst(SMC_LI(bank_id, BAR));
  return p;
}
// Exact MajOperations sequence + an optional retention delay (extra SLEEPs)
// between the last frac and the MAJ doubleACT.
static Program test_prog(uint32_t bank_id, const vector<uint32_t>& patterns,
                         const vector<uint32_t>& open_row_idx,
                         uint32_t r_frac_addr, uint32_t Rfirst, uint32_t Rsecond,
                         uint32_t t_12, uint32_t t_23,
                         uint32_t n_frac_times, uint32_t t_frac,
                         uint32_t num_iter, int retention_sleeps) {
  Program program;
  program.add_below(_init(bank_id, num_iter));
  program.add_inst(all_nops()); program.add_inst(all_nops());
  program.add_below(PRE(BAR, 0, 0));
  program.add_inst(SMC_LI(0, LOOP_ITER));
  program.add_label("ITER_LOOP");
    program.add_below(PRE(BAR, 0, 0));
    for (size_t i = 0; i < open_row_idx.size(); i++)
      program.add_below(wrRow_immediate_label(BAR, open_row_idx[i], patterns[i], rand()));
    program.add_inst(SMC_SLEEP(6)); program.add_below(PRE(BAR, 0, 0)); program.add_inst(SMC_SLEEP(6));
    for (uint32_t j = 0; j < n_frac_times; j++) {
      program.add_inst(SMC_SLEEP(6));
      program.add_below(frac_builder(t_frac, r_frac_addr));
      program.add_inst(SMC_SLEEP(6));
    }
    for (int j = 0; j < retention_sleeps; j++) program.add_inst(SMC_SLEEP(15));
    program.add_inst(SMC_SLEEP(6)); program.add_below(PRE(BAR, 0, 0)); program.add_inst(SMC_SLEEP(6));
    program.add_below(doubleACT(t_12, t_23, Rfirst, Rsecond));
    program.add_inst(SMC_SLEEP(6)); program.add_below(PRE(BAR, 0, 0)); program.add_inst(SMC_SLEEP(6));
    program.add_below(rdRow_immediate(BAR, open_row_idx[0]));
    program.add_inst(all_nops()); program.add_inst(all_nops());
    program.add_below(PRE(BAR, 0, 0));
    program.add_inst(all_nops()); program.add_inst(all_nops());
    program.add_inst(SMC_ADDI(LOOP_ITER, 1, LOOP_ITER));
  program.add_branch(program.BR_TYPE::BL, LOOP_ITER, ITER_REG, "ITER_LOOP");
  program.add_inst(SMC_END());
  return program;
}
// 5A+5B+5C over positions 1..15, frac row at position 0 (init = frac_init).
static vector<uint32_t> build_16row(uint32_t A, uint32_t B, uint32_t C, uint32_t frac_init) {
  vector<uint32_t> lst(16, 0);
  uint32_t base[3] = {A, B, C};
  for (int r = 0; r < 5; r++) for (int i = 0; i < 3; i++) lst[r * 3 + i] = base[i];
  lst[15] = lst[0]; lst[0] = frac_init;
  return lst;
}
static uint32_t maj3(uint32_t a, uint32_t b, uint32_t c){ return (a&b)|(a&c)|(b&c); }

struct Calib { int s_id, bank; uint32_t Rf, Rs; vector<uint32_t> r; };
static Calib readc(const string& path, int wb, int ws) {
  ifstream f(path); string line;
  while (getline(f, line)) { if (line.empty() || line[0]=='#') continue; istringstream is(line);
    Calib c; if(!(is>>c.s_id>>c.bank>>c.Rf>>c.Rs)) continue; uint32_t v; while(is>>v) c.r.push_back(v);
    if (c.r.size()==16 && c.bank==wb && c.s_id==ws) return c; }
  return Calib{-1,0,0,0,{}};
}

int main(int argc, char** argv) {
  if (argc < 5) { fprintf(stderr,"Usage: %s <bender> <calib> <bank> <s_id> [trials=16] [num_iter=4]\n",argv[0]); return 1; }
  int bender=atoi(argv[1]), bank=atoi(argv[3]), sid=atoi(argv[4]);
  int trials=(argc>5)?atoi(argv[5]):16, num_iter=(argc>6)?atoi(argv[6]):4;
  Calib c = readc(argv[2], bank, sid);
  if (c.s_id<0){ fprintf(stderr,"no calib bank=%d s_id=%d\n",bank,sid); return 2; }
  fprintf(stderr,"[frac] tuple s_id=%d bank=%d Rf=%u Rs=%u frac_row=%u trials=%d num_iter=%d\n",
          sid, bank, c.Rf, c.Rs, c.r[0], trials, num_iter);

  SoftMCPlatform pf(bender);
  if (pf.init()!=SOFTMC_SUCCESS){ fprintf(stderr,"init fail\n"); return 3; }
  pf.reset_fpga();
  std::mt19937 rng(0xC0FFEE);  // deterministic

  auto run_config = [&](int n_frac, int t_frac, uint32_t frac_init, int ret)->double{
    long ok=0, tot=0;
    for (int tr=0; tr<trials; tr++) {
      uint32_t A=rng(), B=rng(), C=rng();
      uint32_t exp = maj3(A,B,C);
      vector<uint32_t> pat = build_16row(A,B,C,frac_init);
      Program p = test_prog(bank, pat, c.r, c.r[0], c.Rf, c.Rs,
                            /*t_12=*/0, /*t_23=*/0, n_frac, t_frac, num_iter, ret);
      pf.execute(p);
      for (int it=0; it<num_iter; it++) {
        uint8_t row[8192]; if (pf.receiveData(row,8192)!=8192) return -1.0;
        for (int s=0;s<2048;s++){ uint32_t g; memcpy(&g,&row[s*4],4); ok+=(g==exp); tot++; }
      }
    }
    return 100.0*ok/tot;
  };

  // Phase 1: n_frac × t_frac × init, no retention delay.
  printf("phase,n_frac,t_frac,frac_init,retention,coverage_pct\n");
  int NF[]={0,1,2,3,4,6,8,12,16};
  for (uint32_t fi : {ONE, ZERO}) {
    for (int tf : {0,1,2,3}) {
      for (int nf : NF) {
        double cov = run_config(nf, tf, fi, 0);
        printf("A,%d,%d,%s,0,%.4f\n", nf, tf, fi==ONE?"ONE":"ZERO", cov);
        fprintf(stderr,"[frac] n_frac=%2d t_frac=%d init=%s  cov=%.3f%%\n", nf, tf, fi==ONE?"ONE ":"ZERO", cov);
        fflush(stdout); fflush(stderr);
      }
    }
  }
  // Phase 2: retention — fixed best-guess recipe (n=3,t=0,init=ONE), grow delay.
  for (int ret : {0,1,2,4,8,16}) {
    double cov = run_config(3, 0, ONE, ret);
    printf("B,3,0,ONE,%d,%.4f\n", ret, cov);
    fprintf(stderr,"[frac-ret] retention_sleeps=%2d  cov=%.3f%%\n", ret, cov);
    fflush(stdout);
  }
  fprintf(stderr,"[frac] DONE\n");
  _exit(0);
}
