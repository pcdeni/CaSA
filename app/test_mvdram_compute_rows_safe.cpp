// MVDRAM faithful computation-rows dataflow — SAFE-SOURCE variant (2026-07-17).
// Identical DAG / MAJ / stats to test_mvdram_compute_rows.cpp; ONLY the row
// placement and load order change, per the validated pair-lattice model:
//
//  1. Value rows live INSIDE the compute tuple's subarray [54272, 54912)
//     (the original allocated from 54144 — sources in the NEIGHBORING
//     subarray, i.e. cross-subarray clones).
//  2. Every value row r is chosen so bits(local(r) ^ local(Ti0)) contains
//     neither bit0 nor {bit7&bit8} together -> the load into Ti0 and the
//     result copy-out of Ti0 are tuple-clean BY CONSTRUCTION (tuple pair
//     bits = {0,7,8}; generators {1, 384}).
//  3. Deposit envelopes of every planned clone are BLACKLISTED so no splash
//     row is ever another value row.
//  4. Load order i1 -> i2 -> i0: the envelopes of the i1/i2 loads may cover
//     Ti0 (unavoidable: d1=d0^1, d2=d0^384 supersets of d0), so Ti0 loads
//     LAST; the i1/i2 loads cannot cover each other or Tfr (their deltas
//     need bit0 / group-384, excluded by construction).
//
// A/B against the original on the same bender/bank quantifies exactly how
// much of June's 6.1% end-to-end was addressing, not physics.
// Argv: <bender> <bank> [seed]   (same as original; PIM_RESTORE ignored)
#include "instruction.h"
#include "prog.h"
#include "platform.h"
#include "../util.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <set>
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
  // compute tuple (same as original): s86 4-row group, pair bits {0,7,8}
  const uint32_t Trf=54340,Trs=54725,Ti0=54340,Ti1=54341,Ti2=54724,Tfr=54725;
  const uint32_t SUB=54272, SUBEND=54912;
  auto lo=[&](uint32_t r){ return r-SUB; };
  // popcount-4 DAG (identical to original)
  struct G{const char*n; const char*i0,*i1,*i2;};
  vector<G> gates={
   {"m1","a","b","nc"},{"m2","a","nb","c"},{"m3","na","b","c"},{"carry1","a","b","c"},
   {"sum1","m1","m2","m3"},{"m4","na","b","nc"},{"m5","na","nb","c"},{"nsum1","m4","m5","carry1"},
   {"carry2","sum1","d","Z"},{"ncarry1","na","nb","nc"},{"ncarry2","nsum1","nd","O"},
   {"p1","sum1","nd","Z"},{"p2","nsum1","d","Z"},{"bit0","p1","p2","O"},
   {"q1","carry1","ncarry2","Z"},{"q2","ncarry1","carry2","Z"},{"bit1","q1","q2","O"},{"bit2","carry1","carry2","Z"}};
  vector<pair<string,int>> outs={{"bit0",0},{"bit1",1},{"bit2",2}};

  // SAFE allocator — deterministic antichain scheme.
  // All value rows sit at r = SUB + (local(Ti0) ^ m) where m is a weight-3
  // mask over the safe bits {1,2,3,4,5,6,9} (m & 0x181 == 0 keeps bit0 and
  // the 384-group out of every pair offset -> loads into Ti0 and copy-outs
  // are tuple-clean; bit0/384-splashes land on rows that can never be value
  // rows). Uniform weight makes the mask set an ANTICHAIN under bit-subset,
  // so no load/copy-out envelope point can coincide with another value row.
  // C(7,3)=35 masks >= 28 rows needed.
  // Uniform-weight-3 masks (antichain -> no clone envelope hits another value
  // row) over bits {1..9} except bit0 (generator 1, splashes Ti1). Masks with
  // BOTH bit7&bit8 excluded (that pair = generator 384). Individual bit7/bit8
  // don't fire alone (group-exclusivity), so are safe fillers. C≈77 candidates.
  vector<uint32_t> masks;
  { int sb[9]={1,2,3,4,5,6,7,8,9};
    for(int i=0;i<9;i++)for(int j=i+1;j<9;j++)for(int k=j+1;k<9;k++){
      uint32_t m=(1u<<sb[i])|(1u<<sb[j])|(1u<<sb[k]);
      if((m&384)==384) continue;                          // not both bit7 & bit8
      masks.push_back(m); } }

  SoftMCPlatform pf(bender); if(pf.init()!=SOFTMC_SUCCESS){cerr<<"init fail\n";return 3;}
  pf.reset_fpga(); srand(seed);

  // ---- SCREENING PASS (enumerate-then-verify): a mask is usable only if a
  // clone from its row LANDS on all three tuple inputs (full-distance lattice
  // points are selection-dependent per pair, but deterministic -> screen once).
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
    cerr<<"[cr-safe] screening: "<<usable.size()<<"/"<<masks.size()<<" masks land on all 3 inputs\n";
    if(usable.size()<28){cerr<<"[cr-safe] not enough usable masks\n";_exit(9);}
  }
  size_t mi=0;
  auto alloc=[&](){ return SUB+(lo(Ti0)^usable[mi++]); };

  unordered_map<string,uint32_t> vr;
  const char* lits[10]={"a","b","c","d","na","nb","nc","nd","Z","O"};
  for(auto l:lits) vr[l]=alloc();
  for(auto&g:gates) vr[g.n]=alloc();
  cerr<<"[cr-safe] value rows (local offsets to Ti0):";
  for(auto l:lits) cerr<<" "<<(lo(vr[l])^lo(Ti0));
  cerr<<" | gates:"; for(auto&g:gates) cerr<<" "<<(lo(vr[g.n])^lo(Ti0));
  cerr<<"\n";

  srand(seed);
  unordered_map<string,Vec> id;
  auto mk=[&](Vec&d){for(int s=0;s<2048;s++)d.v[s]=((uint32_t)rand()<<17)^((uint32_t)rand()<<3)^rand();};
  for(int i=0;i<4;i++){Vec t;mk(t);id[lits[i]]=t;}
  for(int i=0;i<4;i++){Vec t;for(int s=0;s<2048;s++)t.v[s]=~id[lits[i]].v[s];id[lits[i+4]]=t;}
  {Vec z;for(int s=0;s<2048;s++)z.v[s]=0;id["Z"]=z;Vec o;for(int s=0;s<2048;s++)o.v[s]=0xFFFFFFFFu;id["O"]=o;}
  for(auto&g:gates){Vec t;for(int s=0;s<2048;s++)t.v[s]=maj3(id[g.i0].v[s],id[g.i1].v[s],id[g.i2].v[s]);id[g.n]=t;}
  for(auto l:lits) pcwrite(pf,bank,vr[l],id[l].v);

  cerr<<"[cr-safe] computation-rows dataflow (SAFE placement + i1,i2,i0 order): "
      <<gates.size()<<" gates\n";
  long opsok=0,opstot=0; int loads_ok=0,loads_tot=0; long majok_clean=0,majtot_clean=0;
  for(auto&g:gates){
    // SAFE ORDER: i1 -> i2 -> i0 (Ti0 is enveloped by the first two loads, so it loads last)
    rowclone(pf,bank,vr[g.i1],Ti1);
    rowclone(pf,bank,vr[g.i2],Ti2);
    rowclone(pf,bank,vr[g.i0],Ti0);
    // diagnostic: did all three operands actually land?
    bool landed=true;
    { const char* ins[3]={g.i0,g.i1,g.i2}; uint32_t Ts[3]={Ti0,Ti1,Ti2};
      for(int j=0;j<3;j++){ uint8_t buf[8192]; read_row(pf,bank,Ts[j],buf);
        long ok=0; for(int s=0;s<2048;s++){uint32_t hv;memcpy(&hv,&buf[s*4],4);
          for(int b=0;b<32;b++) if(((hv>>b)&1)==((id[ins[j]].v[s]>>b)&1))ok++;}
        loads_tot++; if(ok>65536-512) loads_ok++; else landed=false; } }
    pcwrite(pf,bank,Tfr,id["O"].v);
    uint8_t buf[8192]; maj_op(pf,bank,Trf,Trs,Tfr,Ti0,buf);
    rowclone(pf,bank,Ti0,vr[g.n]);
    long ok=0; for(int s=0;s<2048;s++){uint32_t hv;memcpy(&hv,&buf[s*4],4);
      for(int b=0;b<32;b++) if(((hv>>b)&1)==((id[g.n].v[s]>>b)&1))ok++;}
    opsok+=ok; opstot+=65536;
    if(landed){ majok_clean+=ok; majtot_clean+=65536; }
  }
  cerr<<"[cr-safe] operand loads landed: "<<loads_ok<<"/"<<loads_tot<<"\n";
  if(majtot_clean) cerr<<"[cr-safe] MAJ accuracy on fully-landed gates: "
                       <<100.0*majok_clean/majtot_clean<<"% ("<<majtot_clean/65536<<" gates)\n";
  unordered_map<string,Vec> hw;
  for(auto&o:outs){uint8_t buf[8192];read_row(pf,bank,vr[o.first],buf);Vec t;for(int s=0;s<2048;s++)memcpy(&t.v[s],&buf[s*4],4);hw[o.first]=t;}
  long exact=0; for(int s=0;s<2048;s++)for(int b=0;b<32;b++){
    int ic=0;for(int i=0;i<4;i++)ic+=(id[lits[i]].v[s]>>b)&1;
    int hc=0;for(auto&o:outs)hc+=(int)((hw[o.first].v[s]>>b)&1)<<o.second;
    if(ic==hc)exact++;}
  cerr<<"[cr-safe] per-op MAJ vs ideal: "<<100.0*opsok/opstot<<"%\n";
  cerr<<"[cr-safe] END-TO-END popcount bit-exact: "<<100.0*exact/65536<<"%\n";
  _exit(0);
}
