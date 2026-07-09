// MVDRAM repro — FAITHFUL computation-rows dataflow (paper Fig 2 / Ambit model).
// Each MAJ gate: RowCopy its 3 input values from their value-rows into a dedicated
// COMPUTATION TUPLE (originals preserved), MAJ there, RowCopy the result out to the
// gate's value-row. No in-place chaining, no interlock — operands are MOVED by
// RowCopy, exactly as the paper does. This is the implementation that a module with
// "strict RowCopy" (no XOR-spread) enables. On our commodity DIMMs the RowCopy-into-
// compute-tuple is XOR-spread-corrupted (expected to fail); ready for the clean part.
//   PIM_RESTORE=1 : insert an ACT-PRE (full cell restore) after each operand RowCopy,
//                   before the MAJ — tests whether restoring charge recovers it.
// Runs the N=4 popcount DAG; validates count vs ideal. Argv: <bender> <bank> [seed]
#include "instruction.h"
#include "prog.h"
#include "platform.h"
#include "../util.h"
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
static void restore_row(SoftMCPlatform& pf,int bank,uint32_t row){ // full ACT-PRE restore
  Program p; p.add_inst(SMC_LI(8,CASR)); p.add_inst(SMC_LI(bank,BAR)); p.add_inst(SMC_LI(row,RAR));
  p.add_below(PRE(BAR,0,0)); p.add_inst(SMC_SLEEP(6)); p.add_below(ACT(BAR,0,RAR,0)); p.add_inst(SMC_SLEEP(20));
  p.add_below(PRE(BAR,0,0)); p.add_inst(SMC_SLEEP(6)); p.add_inst(SMC_END()); pf.execute(p); }
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
static uint32_t maj3(uint32_t a,uint32_t b,uint32_t c){ return (a&b)|(a&c)|(b&c); }

struct Vec{uint32_t v[2048];};
int main(int argc,char**argv){
  if(argc<3){cerr<<"Usage: "<<argv[0]<<" <bender> <bank> [seed]\n";return 1;}
  int bender=atoi(argv[1]),bank=atoi(argv[2]); int seed=(argc>3)?atoi(argv[3]):7; BANKG=bank;
  int RESTORE=(getenv("PIM_RESTORE")&&atoi(getenv("PIM_RESTORE")))?1:0;
  // compute tuple (a known-good MAJ3 4-row tuple in s86): inputs 54340/54341/54724, frac 54725
  const uint32_t Trf=54340,Trs=54725,Ti0=54340,Ti1=54341,Ti2=54724,Tfr=54725;
  // popcount-4 DAG
  struct G{const char*n; const char*i0,*i1,*i2;};
  vector<G> gates={
   {"m1","a","b","nc"},{"m2","a","nb","c"},{"m3","na","b","c"},{"carry1","a","b","c"},
   {"sum1","m1","m2","m3"},{"m4","na","b","nc"},{"m5","na","nb","c"},{"nsum1","m4","m5","carry1"},
   {"carry2","sum1","d","Z"},{"ncarry1","na","nb","nc"},{"ncarry2","nsum1","nd","O"},
   {"p1","sum1","nd","Z"},{"p2","nsum1","d","Z"},{"bit0","p1","p2","O"},
   {"q1","carry1","ncarry2","Z"},{"q2","ncarry1","carry2","Z"},{"bit1","q1","q2","O"},{"bit2","carry1","carry2","Z"}};
  vector<pair<string,int>> outs={{"bit0",0},{"bit1",1},{"bit2",2}};
  // value rows (free s86 rows, avoid the compute tuple)
  unordered_map<string,uint32_t> vr; uint32_t nx=54144;
  auto isT=[&](uint32_t r){return r==Ti0||r==Ti1||r==Ti2||r==Tfr;};
  auto alloc=[&](){ while(isT(nx))nx++; return nx++; };
  const char* lits[10]={"a","b","c","d","na","nb","nc","nd","Z","O"};
  for(auto l:lits) vr[l]=alloc();
  for(auto&g:gates) vr[g.n]=alloc();

  SoftMCPlatform pf(bender); if(pf.init()!=SOFTMC_SUCCESS){cerr<<"init fail\n";return 3;}
  pf.reset_fpga(); srand(seed);
  // ideal values
  unordered_map<string,Vec> id;
  auto mk=[&](Vec&d){for(int s=0;s<2048;s++)d.v[s]=((uint32_t)rand()<<17)^((uint32_t)rand()<<3)^rand();};
  for(int i=0;i<4;i++){Vec t;mk(t);id[lits[i]]=t;}
  for(int i=0;i<4;i++){Vec t;for(int s=0;s<2048;s++)t.v[s]=~id[lits[i]].v[s];id[lits[i+4]]=t;}
  {Vec z;for(int s=0;s<2048;s++)z.v[s]=0;id["Z"]=z;Vec o;for(int s=0;s<2048;s++)o.v[s]=0xFFFFFFFFu;id["O"]=o;}
  for(auto&g:gates){Vec t;for(int s=0;s<2048;s++)t.v[s]=maj3(id[g.i0].v[s],id[g.i1].v[s],id[g.i2].v[s]);id[g.n]=t;}
  // preload literal value rows (written once)
  for(auto l:lits) pcwrite(pf,bank,vr[l],id[l].v);

  cerr<<"[cr] computation-rows dataflow: "<<gates.size()<<" gates, RowCopy in->MAJ->RowCopy out"
      <<(RESTORE?"  [+restore-after-clone]":"")<<"\n";
  long opsok=0,opstot=0;
  for(auto&g:gates){
    rowclone(pf,bank,vr[g.i0],Ti0); rowclone(pf,bank,vr[g.i1],Ti1); rowclone(pf,bank,vr[g.i2],Ti2);
    pcwrite(pf,bank,Tfr,id["O"].v);                       // frac = ones, discharged in maj_op
    if(RESTORE){ restore_row(pf,bank,Ti0); restore_row(pf,bank,Ti1); restore_row(pf,bank,Ti2); }
    uint8_t buf[8192]; maj_op(pf,bank,Trf,Trs,Tfr,Ti0,buf);
    rowclone(pf,bank,Ti0,vr[g.n]);                        // copy result out to value row
    // per-op vs ideal
    long ok=0; for(int s=0;s<2048;s++){uint32_t hv;memcpy(&hv,&buf[s*4],4);
      for(int b=0;b<32;b++) if(((hv>>b)&1)==((id[g.n].v[s]>>b)&1))ok++;}
    opsok+=ok; opstot+=65536;
  }
  // read outputs, validate popcount
  unordered_map<string,Vec> hw;
  for(auto&o:outs){uint8_t buf[8192];read_row(pf,bank,vr[o.first],buf);Vec t;for(int s=0;s<2048;s++)memcpy(&t.v[s],&buf[s*4],4);hw[o.first]=t;}
  long exact=0; for(int s=0;s<2048;s++)for(int b=0;b<32;b++){
    int ic=0;for(int i=0;i<4;i++)ic+=(id[lits[i]].v[s]>>b)&1;
    int hc=0;for(auto&o:outs)hc+=(int)((hw[o.first].v[s]>>b)&1)<<o.second;
    if(ic==hc)exact++;}
  cerr<<"[cr] per-op MAJ vs ideal: "<<100.0*opsok/opstot<<"%\n";
  cerr<<"[cr] END-TO-END popcount bit-exact: "<<100.0*exact/65536<<"%\n";
  cerr<<"[cr] "<<(100.0*exact/65536>99?"FAITHFUL DATAFLOW WORKS (strict RowCopy present)":
      "BROKEN on this silicon (RowCopy-into-compute-tuple XOR-spread corrupted)")<<"\n";
  _exit(0);
}
