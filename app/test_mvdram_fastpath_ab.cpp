// MVDRAM performance-shape A/B/C (2026-07-17): the popcount-4 carry-save DAG
// (18 MAJ3 gates, identical to test_mvdram_compute_rows_safe.cpp) run three
// ways on the SAME safe row placement, measuring accuracy + ms/gate:
//
//   C  (June correctness shape): operands per-column-WRITTEN into the tuple
//      inputs each gate, Tfr per-column-written, separate executes.
//   A  (July safe fast load, unfused): operands RowCloned from value rows
//      (safe antichain placement, i1->i2->i0 order), Tfr via in-program
//      wrRow(ONE), but each primitive its own execute (compute-rows cadence).
//   B  (fused): the ENTIRE gate in ONE program — clone i1, clone i2,
//      clone i0, wrRow ONE->Tfr, frac x3, MAJ3, rdRow(Ti0), clone
//      Ti0->result — one execute + one receive per gate. Programs are
//      pre-built once per gate and re-executed across iterations (safe
//      since the Program finalization idempotency fix).
//
// End-to-end check per mode: the 3 output bitplanes vs the ideal 4-bit
// popcount of the a,b,c,d literals, plus per-op MAJ accuracy.
// Argv: <bender> <bank> [iters=5] [seed=7]
#include "instruction.h"
#include "prog.h"
#include "platform.h"
#include "../util.h"
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
using namespace std;
static const int CHUNK_COLS[3]={43,43,42};
static int BANKG=0;
static void pcwrite(SoftMCPlatform& pf,int bank,uint32_t row,const uint32_t* seg){
  int cs=0; for(int ch=0;ch<3;ch++){ int n=CHUNK_COLS[ch]; const uint32_t* cd=seg+cs*16;
    Program p; p.add_inst(SMC_LI(8,CASR)); p.add_inst(SMC_LI(bank,BAR)); p.add_inst(SMC_LI(row,RAR)); p.add_inst(SMC_LI(cs*8,CAR));
    p.add_below(PRE(BAR,0,0)); p.add_below(ACT(BAR,0,RAR,0));
    for(int k=0;k<n;k++){ const uint32_t* sl=cd+k*16; for(int s=0;s<16;s++){ p.add_inst(SMC_LI(sl[s],PATTERN_REG)); p.add_inst(SMC_LDWD(PATTERN_REG,s)); }
      p.add_below(WRITE(BAR,CAR,1)); p.add_inst(SMC_SLEEP(8)); }
    p.add_inst(SMC_SLEEP(8)); p.add_below(PRE(BAR,0,0)); p.add_inst(SMC_SLEEP(4)); p.add_inst(SMC_END()); pf.execute(p); cs+=n; }
}
static Program frac_b(int tf,int r){ Program p; p.add_inst(all_nops()); p.add_inst(SMC_LI(r,RF_REG));
  int nc=2+tf; nc+=4-(nc%4); Mininst q[nc]; for(int i=0;i<nc;i++)q[i]=SMC_NOP();
  q[0]=SMC_ACT(BAR,0,RF_REG,0); q[tf+1]=SMC_PRE(BAR,0,0); for(int i=0;i<nc;i+=4)p.add_inst(q[i],q[i+1],q[i+2],q[i+3]); return p; }
static void rowclone(SoftMCPlatform& pf,int bank,uint32_t src,uint32_t dst){
  Program p; p.add_inst(SMC_LI(8,CASR)); p.add_inst(SMC_LI(bank,BAR)); p.add_inst(SMC_LI(128,NUM_COLS_REG));
  p.add_below(PRE(BAR,0,0)); p.add_inst(SMC_SLEEP(6));
  p.add_below(doubleACT(30,1,src,dst)); p.add_inst(SMC_SLEEP(6)); p.add_below(PRE(BAR,0,0)); p.add_inst(SMC_SLEEP(6));
  p.add_inst(SMC_END()); pf.execute(p); }
static void wr_one(SoftMCPlatform& pf,int bank,uint32_t row,int lbl){
  Program p; p.add_inst(SMC_LI(8,CASR)); p.add_inst(SMC_LI(bank,BAR)); p.add_inst(SMC_LI(128,NUM_COLS_REG));
  p.add_below(PRE(BAR,0,0)); p.add_inst(SMC_SLEEP(6));
  p.add_below(wrRow_immediate_label(BAR,row,ONE,lbl)); p.add_inst(SMC_SLEEP(6)); p.add_below(PRE(BAR,0,0)); p.add_inst(SMC_SLEEP(6));
  p.add_inst(SMC_END()); pf.execute(p); }
static void maj_op(SoftMCPlatform& pf,int bank,uint32_t rf,uint32_t rs,uint32_t fr,uint32_t rd,uint8_t* out){
  Program p; p.add_inst(SMC_LI(8,CASR)); p.add_inst(SMC_LI(bank,BAR)); p.add_inst(SMC_LI(128,NUM_COLS_REG));
  p.add_below(PRE(BAR,0,0)); p.add_inst(SMC_SLEEP(6));
  for(int j=0;j<3;j++){ p.add_inst(SMC_SLEEP(6)); p.add_below(frac_b(0,fr)); p.add_inst(SMC_SLEEP(6)); }
  p.add_inst(SMC_SLEEP(6)); p.add_below(PRE(BAR,0,0)); p.add_inst(SMC_SLEEP(6));
  p.add_below(doubleACT(0,0,rf,rs)); p.add_inst(SMC_SLEEP(6)); p.add_below(PRE(BAR,0,0)); p.add_inst(SMC_SLEEP(6));
  p.add_below(rdRow_immediate(BAR,rd)); p.add_inst(all_nops()); p.add_inst(all_nops()); p.add_below(PRE(BAR,0,0)); p.add_inst(SMC_END());
  pf.execute(p); pf.receiveData(out,8192); }
static void read_row(SoftMCPlatform& pf,int bank,uint32_t row,uint8_t* out){
  Program p; p.add_inst(SMC_LI(8,CASR)); p.add_inst(SMC_LI(bank,BAR)); p.add_inst(SMC_LI(128,NUM_COLS_REG));
  p.add_below(PRE(BAR,0,0)); p.add_inst(SMC_SLEEP(6));
  p.add_below(rdRow_immediate(BAR,row)); p.add_inst(all_nops()); p.add_inst(all_nops()); p.add_below(PRE(BAR,0,0)); p.add_inst(SMC_END());
  pf.execute(p); pf.receiveData(out,8192); }
static uint32_t maj3f(uint32_t a,uint32_t b,uint32_t c){ return (a&b)|(a&c)|(b&c); }

// Fused per-gate program (mode B): everything in one execute.
static Program build_fused_gate(int bank, uint32_t src1, uint32_t src2,
                                 uint32_t src0, uint32_t Ti0, uint32_t Ti1,
                                 uint32_t Ti2, uint32_t Tfr, uint32_t Trf,
                                 uint32_t Trs, uint32_t dst, int lbl){
  Program p;
  p.add_inst(SMC_LI(8,CASR)); p.add_inst(SMC_LI(bank,BAR)); p.add_inst(SMC_LI(128,NUM_COLS_REG));
  p.add_below(PRE(BAR,0,0)); p.add_inst(SMC_SLEEP(6));
  // safe load order i1 -> i2 -> i0 (Ti0 enveloped by the first two)
  p.add_below(doubleACT(30,1,src1,Ti1)); p.add_inst(SMC_SLEEP(6)); p.add_below(PRE(BAR,0,0)); p.add_inst(SMC_SLEEP(6));
  p.add_below(doubleACT(30,1,src2,Ti2)); p.add_inst(SMC_SLEEP(6)); p.add_below(PRE(BAR,0,0)); p.add_inst(SMC_SLEEP(6));
  p.add_below(doubleACT(30,1,src0,Ti0)); p.add_inst(SMC_SLEEP(6)); p.add_below(PRE(BAR,0,0)); p.add_inst(SMC_SLEEP(6));
  // frac-row refresh: Tfr holds a uniform ONE — a wrRow, not a per-col write
  p.add_below(wrRow_immediate_label(BAR,Tfr,ONE,lbl)); p.add_inst(SMC_SLEEP(6)); p.add_below(PRE(BAR,0,0)); p.add_inst(SMC_SLEEP(6));
  for(int j=0;j<3;j++){ p.add_inst(SMC_SLEEP(6)); p.add_below(frac_b(0,Tfr)); p.add_inst(SMC_SLEEP(6)); }
  p.add_inst(SMC_SLEEP(6)); p.add_below(PRE(BAR,0,0)); p.add_inst(SMC_SLEEP(6));
  p.add_below(doubleACT(0,0,Trf,Trs)); p.add_inst(SMC_SLEEP(6)); p.add_below(PRE(BAR,0,0)); p.add_inst(SMC_SLEEP(6));
  p.add_below(rdRow_immediate_label(BAR,Ti0,lbl+500)); p.add_inst(all_nops()); p.add_inst(all_nops()); p.add_below(PRE(BAR,0,0)); p.add_inst(all_nops()); p.add_inst(all_nops());
  // result copy-out, tuple-clean by construction
  p.add_below(doubleACT(30,1,Ti0,dst)); p.add_inst(SMC_SLEEP(6)); p.add_below(PRE(BAR,0,0)); p.add_inst(SMC_SLEEP(6));
  p.add_inst(SMC_END());
  return p;
}

struct Vec{uint32_t v[2048];};
int main(int argc,char**argv){
  if(argc<3){cerr<<"Usage: "<<argv[0]<<" <bender> <bank> [iters=5] [seed=7]\n";return 1;}
  int bender=atoi(argv[1]),bank=atoi(argv[2]); int iters=(argc>3)?atoi(argv[3]):5;
  int seed=(argc>4)?atoi(argv[4]):7; BANKG=bank;
  const uint32_t Trf=54340,Trs=54725,Ti0=54340,Ti1=54341,Ti2=54724,Tfr=54725;
  const uint32_t SUB=54272;
  auto lo=[&](uint32_t r){ return r-SUB; };
  struct G{const char*n; const char*i0,*i1,*i2;};
  vector<G> gates={
   {"m1","a","b","nc"},{"m2","a","nb","c"},{"m3","na","b","c"},{"carry1","a","b","c"},
   {"sum1","m1","m2","m3"},{"m4","na","b","nc"},{"m5","na","nb","c"},{"nsum1","m4","m5","carry1"},
   {"carry2","sum1","d","Z"},{"ncarry1","na","nb","nc"},{"ncarry2","nsum1","nd","O"},
   {"p1","sum1","nd","Z"},{"p2","nsum1","d","Z"},{"bit0","p1","p2","O"},
   {"q1","carry1","ncarry2","Z"},{"q2","ncarry1","carry2","Z"},{"bit1","q1","q2","O"},{"bit2","carry1","carry2","Z"}};
  vector<pair<string,int>> outs={{"bit0",0},{"bit1",1},{"bit2",2}};

  vector<uint32_t> masks;
  { int sb[9]={1,2,3,4,5,6,7,8,9};
    for(int i=0;i<9;i++)for(int j=i+1;j<9;j++)for(int k=j+1;k<9;k++){
      uint32_t m=(1u<<sb[i])|(1u<<sb[j])|(1u<<sb[k]);
      if((m&384)==384) continue;
      masks.push_back(m); } }

  SoftMCPlatform pf(bender); if(pf.init()!=SOFTMC_SUCCESS){cerr<<"init fail\n";return 3;}
  pf.reset_fpga(); srand(seed);

  // screening (identical to compute_rows_safe)
  vector<uint32_t> usable;
  {
    Vec marker; for(int s=0;s<2048;s++) marker.v[s]=0xA5A50000u^(s*2654435761u);
    uint8_t mexp[8192]; for(int s=0;s<2048;s++) for(int b=0;b<4;b++) mexp[s*4+b]=(uint8_t)((marker.v[s]>>(8*b))&0xFF);
    auto zero_tuple=[&](){
      Program p; p.add_inst(SMC_LI(8,CASR)); p.add_inst(SMC_LI(bank,BAR)); p.add_inst(SMC_LI(128,NUM_COLS_REG));
      p.add_below(PRE(BAR,0,0)); int lbl=0;
      for(uint32_t r : {Ti0,Ti1,Ti2,Tfr}) p.add_below(wrRow_immediate_label(BAR,r,0u,lbl++));
      p.add_inst(SMC_END()); pf.execute(p); };
    for(uint32_t m : masks){
      if((int)usable.size()>=28) break;
      uint32_t src=SUB+(lo(Ti0)^m);
      pcwrite(pf,bank,src,marker.v);
      bool ok=true;
      for(uint32_t T : {Ti0,Ti1,Ti2}){
        zero_tuple();
        rowclone(pf,bank,src,T);
        uint8_t buf[8192]; read_row(pf,bank,T,buf);
        int match=0; for(int b=0;b<8192;b++) if(buf[b]==mexp[b]) match++;
        if(match<8192-64){ ok=false; break; }
      }
      if(ok) usable.push_back(m);
    }
    cerr<<"[fp-ab] screening: "<<usable.size()<<"/"<<masks.size()<<" masks usable\n";
    if(usable.size()<28){cerr<<"[fp-ab] not enough usable masks\n";_exit(9);}
  }
  size_t mi=0;
  auto alloc=[&](){ return SUB+(lo(Ti0)^usable[mi++]); };
  unordered_map<string,uint32_t> vr;
  const char* lits[10]={"a","b","c","d","na","nb","nc","nd","Z","O"};
  for(auto l:lits) vr[l]=alloc();
  for(auto&g:gates) vr[g.n]=alloc();

  // ideal values
  srand(seed);
  unordered_map<string,Vec> id;
  auto mk=[&](Vec&d){for(int s=0;s<2048;s++)d.v[s]=((uint32_t)rand()<<17)^((uint32_t)rand()<<3)^rand();};
  for(int i=0;i<4;i++){Vec t;mk(t);id[lits[i]]=t;}
  for(int i=0;i<4;i++){Vec t;for(int s=0;s<2048;s++)t.v[s]=~id[lits[i]].v[s];id[lits[i+4]]=t;}
  {Vec z;for(int s=0;s<2048;s++)z.v[s]=0;id["Z"]=z;Vec o;for(int s=0;s<2048;s++)o.v[s]=0xFFFFFFFFu;id["O"]=o;}
  for(auto&g:gates){Vec t;for(int s=0;s<2048;s++)t.v[s]=maj3f(id[g.i0].v[s],id[g.i1].v[s],id[g.i2].v[s]);id[g.n]=t;}

  // mode-B fused programs, pre-built ONCE (idempotent finalization)
  vector<Program> fused;
  int lbl=1000;
  for(auto&g:gates){
    fused.push_back(build_fused_gate(bank,vr[g.i1],vr[g.i2],vr[g.i0],
                    Ti0,Ti1,Ti2,Tfr,Trf,Trs,vr[g.n],lbl));
    lbl+=1000;
  }
  { long mx=0; for(auto&p:fused) if(p.size()/8>mx) mx=p.size()/8;
    cerr<<"[fp-ab] fused gate program: max "<<mx<<" insts (IMEM 2048)\n";
    if(mx>2040){ cerr<<"[fp-ab] ABORT oversize\n"; _exit(5); } }

  auto run_mode=[&](char mode)->pair<double,pair<double,double>>{
    // reload literal value rows fresh for every mode
    for(auto l:lits) pcwrite(pf,bank,vr[l],id[l].v);
    long opsok=0,opstot=0, exact=0; double ms_gate=0; int gate_execs=0;
    for(int it=0; it<iters; it++){
      auto t0=chrono::steady_clock::now();
      for(size_t gi=0; gi<gates.size(); gi++){
        auto&g=gates[gi];
        uint8_t buf[8192];
        if(mode=='C'){
          pcwrite(pf,bank,Ti1,id[g.i1].v);
          pcwrite(pf,bank,Ti2,id[g.i2].v);
          pcwrite(pf,bank,Ti0,id[g.i0].v);
          pcwrite(pf,bank,Tfr,id["O"].v);
          maj_op(pf,bank,Trf,Trs,Tfr,Ti0,buf);
          rowclone(pf,bank,Ti0,vr[g.n]);
        } else if(mode=='A'){
          rowclone(pf,bank,vr[g.i1],Ti1);
          rowclone(pf,bank,vr[g.i2],Ti2);
          rowclone(pf,bank,vr[g.i0],Ti0);
          wr_one(pf,bank,Tfr,(int)(9000+gi));
          maj_op(pf,bank,Trf,Trs,Tfr,Ti0,buf);
          rowclone(pf,bank,Ti0,vr[g.n]);
        } else { // 'B' fused
          pf.execute(fused[gi]);
          pf.receiveData(buf,8192);
        }
        gate_execs++;
        long ok=0; for(int s=0;s<2048;s++){uint32_t hv;memcpy(&hv,&buf[s*4],4);
          for(int b=0;b<32;b++) if(((hv>>b)&1)==((id[g.n].v[s]>>b)&1))ok++;}
        opsok+=ok; opstot+=65536;
      }
      ms_gate+=chrono::duration<double,milli>(chrono::steady_clock::now()-t0).count();
      // end-to-end popcount check for this iteration
      unordered_map<string,Vec> hw;
      for(auto&o:outs){uint8_t buf[8192];read_row(pf,bank,vr[o.first],buf);Vec t;for(int s=0;s<2048;s++)memcpy(&t.v[s],&buf[s*4],4);hw[o.first]=t;}
      for(int s=0;s<2048;s++)for(int b=0;b<32;b++){
        int ic=0;for(int i=0;i<4;i++)ic+=(id[lits[i]].v[s]>>b)&1;
        int hc=0;for(auto&o:outs)hc+=(int)((hw[o.first].v[s]>>b)&1)<<o.second;
        if(ic==hc)exact++;}
    }
    double per_op=100.0*opsok/opstot;
    double e2e=100.0*exact/(65536.0*iters);
    return {ms_gate/gate_execs,{per_op,e2e}};
  };

  for(char m : {'C','A','B'}){
    auto r=run_mode(m);
    const char* nm = (m=='C')?"C write-load (June shape)   "
                   : (m=='A')?"A clone-load, unfused       "
                   :          "B clone-load, FUSED 1-exec  ";
    fprintf(stderr,"[fp-ab] %s: %7.3f ms/gate   per-op %.3f%%   e2e popcount %.3f%%\n",
            nm, r.first, r.second.first, r.second.second);
  }
  _exit(0);
}
