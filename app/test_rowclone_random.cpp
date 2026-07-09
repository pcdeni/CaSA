// Random-pair RowClone scan. Tests whether ANY two rows anywhere in the module
// interact via charge-share RowClone — robust to scrambled row-address->physical
// mapping (a fixed-stride pair like 40000->40016 may straddle subarrays and never
// clone even if PUD works; random pairs occasionally land in the same physical
// subarray and would light up). One platform init, many random pairs, fast.
//   write rA=0xA5.. , rB=0x5A.. ; RowClone(rA->rB) ; read rB ; count bytes==0xA5.
//   8192 = real clone (interacting pair found). ~41 = noise floor (no interaction).
// Argv: ./mvdram-rcrand-exe <bender> <bank> <n_pairs> [seed]
// Env:  PIM_T12 (default 30), PIM_ROWMAX (default 65536)
#include "instruction.h"
#include "prog.h"
#include "platform.h"
#include "../util.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
using namespace std;

static int LBL = 1;
static void write_uniform(SoftMCPlatform& pf, int bank, uint32_t row, uint32_t pat){
  Program p; p.add_inst(SMC_LI(8,CASR)); p.add_inst(SMC_LI(bank,BAR)); p.add_inst(SMC_LI(128,NUM_COLS_REG));
  p.add_below(PRE(BAR,0,0)); p.add_inst(SMC_SLEEP(6));
  p.add_below(wrRow_immediate_label(BAR,row,pat,LBL++)); p.add_inst(SMC_SLEEP(6));
  p.add_below(PRE(BAR,0,0)); p.add_inst(SMC_SLEEP(4)); p.add_inst(SMC_END()); pf.execute(p);
}
static int clone_read(SoftMCPlatform& pf, int bank, uint32_t src, uint32_t dst,
                      int t12, int t23, uint8_t expect){
  Program p; p.add_inst(SMC_LI(8,CASR)); p.add_inst(SMC_LI(bank,BAR)); p.add_inst(SMC_LI(128,NUM_COLS_REG));
  p.add_below(PRE(BAR,0,0)); p.add_inst(SMC_SLEEP(6));
  p.add_below(doubleACT(t12,t23,src,dst)); p.add_inst(SMC_SLEEP(6));
  p.add_below(PRE(BAR,0,0)); p.add_inst(SMC_SLEEP(6));
  p.add_below(rdRow_immediate_label(BAR,dst,LBL++)); p.add_inst(SMC_END());
  pf.execute(p); uint8_t buf[8192]; if(pf.receiveData(buf,8192)!=8192) return -1;
  int m=0; for(int i=0;i<8192;i++) if(buf[i]==expect) m++; return m;
}

int main(int argc,char**argv){
  if(argc<4){cerr<<"Usage: "<<argv[0]<<" <bender> <bank> <n_pairs> [seed]\n";return 1;}
  int bender=atoi(argv[1]),bank=atoi(argv[2]),N=atoi(argv[3]); unsigned seed=(argc>4)?atoi(argv[4]):12345;
  int t12=getenv("PIM_T12")?atoi(getenv("PIM_T12")):30;
  uint32_t ROWMAX=getenv("PIM_ROWMAX")?strtoul(getenv("PIM_ROWMAX"),0,10):65536;
  srand(seed);
  SoftMCPlatform pf(bender); if(pf.init()!=SOFTMC_SUCCESS){cerr<<"init fail\n";return 3;}
  pf.reset_fpga();
  cerr<<"[rcrand] bender "<<bender<<" bank "<<bank<<" : "<<N<<" random pairs, rows [0,"<<ROWMAX
      <<"), t_12="<<t12<<", t_23 in {1,2}\n";
  int best=0; uint32_t bestA=0,bestB=0; int bestT=0; long clones=0, partial=0;
  for(int i=0;i<N;i++){
    uint32_t rA=((uint32_t)rand()*1u)%ROWMAX, rB=((uint32_t)rand()*1u)%ROWMAX;
    while(rB==rA) rB=((uint32_t)rand())%ROWMAX;
    write_uniform(pf,bank,rA,0xA5A5A5A5u);
    write_uniform(pf,bank,rB,0x5A5A5A5Au);
    int bm=0;
    for(int t23=1;t23<=2;t23++){ int m=clone_read(pf,bank,rA,rB,t12,t23,0xA5); if(m>bm)bm=m; if(m>=8000)break; }
    if(bm>best){ best=bm; bestA=rA; bestB=rB; }
    if(bm>=8000){ clones++; cerr<<"[rcrand] *** CLONE: rA="<<rA<<" rB="<<rB<<" match="<<bm<<"/8192 ***\n"; }
    else if(bm>1000) partial++;
    if((i+1)%200==0) cerr<<"[rcrand] "<<(i+1)<<"/"<<N<<" pairs; clones="<<clones
        <<" partial(>1000)="<<partial<<" best="<<best<<"/8192 (rA="<<bestA<<",rB="<<bestB<<")\n";
  }
  cerr<<"[rcrand] DONE "<<N<<" pairs: full-clones="<<clones<<" partial(>1000)="<<partial
      <<" BEST="<<best<<"/8192 at rA="<<bestA<<" rB="<<bestB<<"\n";
  cerr<<"[rcrand] "<<(clones>0?"INTERACTION FOUND — some rows DO RowClone (addressing may be scrambled)":
      best>1000?"only partial — marginal/none":"NO INTERACTION — no two random rows charge-share (no PUD)")<<"\n";
  _exit(0);
}
