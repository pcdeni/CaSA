// Item 3 — per-MAJ activation-update cost A/B (2026-07-17).
// The BitNet dense-MatVec per-MAJ cost is dominated by the ACTIVATION UPDATE:
// writing the input x into the tuple's activation slots. Today that is 5 full
// per-column wrRow programs (act_pos = open_rows[1,4,7,10,13]). The validated
// sub-lattice broadcast replaces them with: 1 per-column write of x into one
// anchor row + 1 intra-coset doubleACT. This measures BOTH the SoftMC program
// instruction count (the on-DRAM tCK proxy) AND host wall-clock, A vs B.
// Argv: <bender> <calib_file> <bank> <s_id> <sub_start> [iters]
#include "instruction.h"
#include "prog.h"
#include "platform.h"
#include "../util.h"
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
using namespace std;
static const int CHUNK_COLS[3]={43,43,42};
static int BANK=0;

static Program chunk_prog(uint32_t row,const uint32_t* cd,int cs,int n){
  Program p; p.add_inst(SMC_LI(8,CASR)); p.add_inst(SMC_LI(BANK,BAR)); p.add_inst(SMC_LI(row,RAR)); p.add_inst(SMC_LI(cs*8,CAR));
  p.add_below(PRE(BAR,0,0)); p.add_below(ACT(BAR,0,RAR,0));
  for(int k=0;k<n;k++){ const uint32_t* sl=cd+k*16; for(int s=0;s<16;s++){ p.add_inst(SMC_LI(sl[s],PATTERN_REG)); p.add_inst(SMC_LDWD(PATTERN_REG,s)); }
    p.add_below(WRITE(BAR,CAR,1)); p.add_inst(SMC_SLEEP(8)); }
  p.add_inst(SMC_SLEEP(8)); p.add_below(PRE(BAR,0,0)); p.add_inst(SMC_SLEEP(4)); p.add_inst(SMC_END()); return p; }

struct Calib{int s_id,bank;uint32_t Rf,Rs;vector<uint32_t> rows;};
static Calib read_calib(const string& path,int wb,int ws){
  ifstream f(path); string line;
  while(getline(f,line)){ if(line.empty()||line[0]=='#')continue; istringstream is(line);
    Calib c; if(!(is>>c.s_id>>c.bank>>c.Rf>>c.Rs))continue; uint32_t v; while(is>>v)c.rows.push_back(v);
    if(c.rows.size()==16&&c.bank==wb&&c.s_id==ws)return c; }
  return Calib{-1,0,0,0,{}}; }

int main(int argc,char**argv){
  if(argc<6){cerr<<"Usage: "<<argv[0]<<" <bender> <calib> <bank> <s_id> <sub_start> [iters]\n";return 1;}
  int bender=atoi(argv[1]); BANK=atoi(argv[3]); int sid=atoi(argv[4]);
  uint32_t SUB=strtoul(argv[5],0,10); int iters=(argc>6)?atoi(argv[6]):200;
  Calib c=read_calib(argv[2],BANK,sid); if(c.s_id<0){cerr<<"no calib\n";return 2;}
  uint32_t anchor=c.Rf, lanchor=anchor-SUB;

  vector<uint32_t> X(2048); for(int s=0;s<2048;s++)X[s]=0x13570000u^(s*2246822519u);
  SoftMCPlatform pf(bender); if(pf.init()!=SOFTMC_SUCCESS){cerr<<"init\n";return 3;} pf.reset_fpga();

  // ---- Build program A: old activation update = 5 per-column wrRow to act_pos ----
  int act_pos[5]={1,4,7,10,13};
  auto build_A=[&](){
    vector<Program> ps;
    for(int a=0;a<5;a++){ uint32_t row=c.rows[act_pos[a]];
      int cs=0; for(int ch=0;ch<3;ch++){ ps.push_back(chunk_prog(row,X.data()+cs*16,cs,CHUNK_COLS[ch])); cs+=CHUNK_COLS[ch]; } }
    return ps; };
  // ---- Build program B: 1 per-column write to anchor + 1 coset doubleACT ----
  auto build_B=[&](){
    vector<Program> ps;
    int cs=0; for(int ch=0;ch<3;ch++){ ps.push_back(chunk_prog(anchor,X.data()+cs*16,cs,CHUNK_COLS[ch])); cs+=CHUNK_COLS[ch]; }
    // coset partner that opens exactly the k=3 sub-coset covering the 5 act slots'
    // mat-groups; for a demo we use full-tuple broadcast partner Rs (covers all).
    Program d; d.add_inst(SMC_LI(8,CASR)); d.add_inst(SMC_LI(BANK,BAR));
    d.add_below(PRE(BAR,0,0)); d.add_inst(SMC_SLEEP(6));
    d.add_below(doubleACT(10,2,anchor,c.Rs)); d.add_inst(SMC_SLEEP(6));
    d.add_below(PRE(BAR,0,0)); d.add_inst(SMC_SLEEP(6)); d.add_inst(SMC_END());
    ps.push_back(d); return ps; };

  auto A=build_A(); auto B=build_B();
  long instA=0,instB=0; for(auto&p:A)instA+=p.size()/8; for(auto&p:B)instB+=p.size()/8;

  auto run=[&](vector<Program>& ps){ for(auto&p:ps) pf.execute(p); };
  // warm
  run(A); run(B);
  auto t0=chrono::steady_clock::now();
  for(int i=0;i<iters;i++) run(A);
  auto t1=chrono::steady_clock::now();
  for(int i=0;i<iters;i++) run(B);
  auto t2=chrono::steady_clock::now();
  double msA=chrono::duration<double,milli>(t1-t0).count()/iters;
  double msB=chrono::duration<double,milli>(t2-t1).count()/iters;

  fprintf(stderr,"[ab] A (old 5x wrRow): %ld SoftMC insts, %d programs, %.3f ms/update\n",instA,(int)A.size(),msA);
  fprintf(stderr,"[ab] B (1 write + 1 doubleACT): %ld SoftMC insts, %d programs, %.3f ms/update\n",instB,(int)B.size(),msB);
  fprintf(stderr,"[ab] instruction reduction: %.2fx   wall-clock speedup: %.2fx\n",
          (double)instA/instB,msA/msB);
  fprintf(stderr,"[ab] activation update is ~80%% of per-MAJ tCK; net per-MAJ ~%.2fx if it dominates\n",
          1.0/(0.2+0.8*instB/instA));
  _exit(0);
}
