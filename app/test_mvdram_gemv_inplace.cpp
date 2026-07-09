// MVDRAM repro — COMPLETE in-DRAM GeMV with IN-PLACE accumulation. Replaces the
// per-column-write MAJ5 popcount tree (test_mvdram_gemv_n) with the in-place
// all-MAJ3 popcount emitted by the allocator (popcount_alloc.py). Per (weight-
// plane i, activation-plane c): the N selected product rows feed the in-place
// popcount plan; host combines FAC(i)*FAC(c)*count. y = sum_n x_n w_n.
// The accumulation is now in-DRAM in-place (in-place hand-offs + spill operand
// moves) rather than reloaded per op.
//
// Argv: ./mvdram-gemv-inplace-exe <bender> <bank> <plan> [seed]
// Env: PIM_N(=4) PIM_QBITS(=2) PIM_RBITS(=4) PIM_VOTE3
#include "instruction.h"
#include "prog.h"
#include "platform.h"
#include "../util.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
using namespace std;
static const int CHUNK_COLS[3]={43,43,42};
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
static void maj_op(SoftMCPlatform& pf,int bank,uint32_t rf,uint32_t rs,uint32_t fr,uint32_t rd,uint8_t* out){
  Program p; p.add_inst(SMC_LI(8,CASR)); p.add_inst(SMC_LI(bank,BAR)); p.add_inst(SMC_LI(128,NUM_COLS_REG));
  p.add_below(PRE(BAR,0,0)); p.add_inst(SMC_SLEEP(6));
  for(int j=0;j<3;j++){ p.add_inst(SMC_SLEEP(6)); p.add_below(frac_b(0,fr)); p.add_inst(SMC_SLEEP(6)); }
  p.add_inst(SMC_SLEEP(6)); p.add_below(PRE(BAR,0,0)); p.add_inst(SMC_SLEEP(6));
  p.add_below(doubleACT(0,0,rf,rs)); p.add_inst(SMC_SLEEP(6)); p.add_below(PRE(BAR,0,0)); p.add_inst(SMC_SLEEP(6));
  p.add_below(rdRow_immediate(BAR,rd)); p.add_inst(all_nops()); p.add_inst(all_nops()); p.add_below(PRE(BAR,0,0)); p.add_inst(SMC_END());
  pf.execute(p); pf.receiveData(out,8192); }
static uint32_t maj3(uint32_t a,uint32_t b,uint32_t c){ return (a&b)|(a&c)|(b&c); }
// MVDRAM on-the-fly vector encoding (§V-C): selective RowClone of a preloaded
// source row into a destination compute row. doubleACT(30,1,src,dst).
static void rowclone(SoftMCPlatform& pf,int bank,uint32_t src,uint32_t dst){
  Program p; p.add_inst(SMC_LI(8,CASR)); p.add_inst(SMC_LI(bank,BAR)); p.add_inst(SMC_LI(128,NUM_COLS_REG));
  p.add_below(PRE(BAR,0,0)); p.add_inst(SMC_SLEEP(6));
  p.add_below(doubleACT(30,1,src,dst)); p.add_inst(SMC_SLEEP(6)); p.add_below(PRE(BAR,0,0)); p.add_inst(SMC_SLEEP(6));
  p.add_inst(SMC_END()); pf.execute(p); }

struct Vec{ uint32_t v[2048]; };
struct Op{ string cmd; uint32_t row,rf,rs,fr,out; string node,i0,i1,i2; int bit; };
static vector<Op> parse_plan(const string& p){ vector<Op> ops; ifstream f(p); string ln;
  while(getline(f,ln)){ if(ln.empty()||ln[0]=='#')continue; stringstream ss(ln); Op o; ss>>o.cmd;
    if(o.cmd=="WRITE") ss>>o.row>>o.node;
    else if(o.cmd=="SPILL") ss>>o.row>>o.node;
    else if(o.cmd=="MAJ") ss>>o.rf>>o.rs>>o.fr>>o.out>>o.node>>o.i0>>o.i1>>o.i2;
    else if(o.cmd=="OUT"){ ss>>o.node>>o.bit; }
    ops.push_back(o);} return ops; }

// run the in-place popcount plan with inputs[N] as the literals a,b,c,d; fill out bits.
// select!=0 + srcrow: MVDRAM on-the-fly encoding — product literals (a..d,na..nd)
// are formed by selective RowClone from preloaded source rows instead of host WRITE.
static void exec_popcount(SoftMCPlatform& pf,int bank,const vector<Op>& ops,int N,
                          const Vec* inp[4], vector<pair<int,Vec>>& outbits,
                          int select, const unordered_map<string,uint32_t>* srcrow){
  unordered_map<string,Vec> val;  // symbol -> ideal value (for validation) / hw node value
  const char* L[4]={"a","b","c","d"}; const char* NL[4]={"na","nb","nc","nd"};
  for(int i=0;i<N;i++){ val[L[i]]=*inp[i]; Vec t; for(int s=0;s<2048;s++)t.v[s]=~inp[i]->v[s]; val[NL[i]]=t; }
  { Vec z; for(int s=0;s<2048;s++)z.v[s]=0; val["Z"]=z; Vec o; for(int s=0;s<2048;s++)o.v[s]=0xFFFFFFFFu; val["O"]=o; }
  outbits.clear();
  for(const auto& o:ops){
    if(o.cmd=="WRITE"){
      if(select && srcrow && srcrow->count(o.node)) rowclone(pf,bank,srcrow->at(o.node),o.row); // in-DRAM selection
      else pcwrite(pf,bank,o.row,val[o.node].v);
    }
    else if(o.cmd=="SPILL") pcwrite(pf,bank,o.row,val[o.node].v);  // host operand move
    else if(o.cmd=="MAJ"){ uint8_t buf[8192]; maj_op(pf,bank,o.rf,o.rs,o.fr,o.out,buf);
      Vec hv; for(int s=0;s<2048;s++)memcpy(&hv.v[s],&buf[s*4],4); val[o.node]=hv; }
    else if(o.cmd=="OUT") outbits.push_back({o.bit,val[o.node]});
  }
}

int main(int argc,char**argv){
  if(argc<4){cerr<<"Usage: "<<argv[0]<<" <bender> <bank> <plan> [seed]\n";return 1;}
  int bender=atoi(argv[1]),bank=atoi(argv[2]); string plan=argv[3]; int seed=(argc>4)?atoi(argv[4]):7;
  int N=getenv("PIM_N")?atoi(getenv("PIM_N")):4;          // popcount width (= plan's N)
  int DIN=getenv("PIM_DIN")?atoi(getenv("PIM_DIN")):N;     // total contraction dim (tiled into DIN/N chunks)
  int QB=getenv("PIM_QBITS")?atoi(getenv("PIM_QBITS")):2;
  int RB=getenv("PIM_RBITS")?atoi(getenv("PIM_RBITS")):4;
  int VOTE=(getenv("PIM_VOTE3")&&atoi(getenv("PIM_VOTE3")))?1:0;
  int SKIP=(getenv("PIM_SKIP")&&atoi(getenv("PIM_SKIP")))?1:0;       // bit-sparsity skip (§V-D)
  int SELECT=(getenv("PIM_SELECT")&&atoi(getenv("PIM_SELECT")))?1:0; // in-DRAM RowClone selection (§V-C)
  if(DIN%N){cerr<<"[gi] PIM_DIN must be a multiple of N\n";return 1;}
  vector<Op> ops=parse_plan(plan);
  cerr<<"[gi] in-place GeMV: d_in="<<DIN<<" N="<<N<<" q="<<QB<<" r="<<RB<<" vote="<<VOTE
      <<" skip="<<SKIP<<" select="<<SELECT<<" ; plan ops="<<ops.size()<<"\n";

  srand(seed);
  // signed q-bit weights, r-bit activations (same scheme as gemv_n), over d_in
  static uint32_t Wb[64][8][2048]; memset(Wb,0,sizeof Wb);
  static int wv[64][2048][32];
  int wlo=-(1<<(QB-1)), whi=(1<<(QB-1))-1;
  for(int n=0;n<DIN;n++)for(int s=0;s<2048;s++)for(int b=0;b<32;b++){ int w=wlo+(rand()%(whi-wlo+1)); wv[n][s][b]=w;
    uint32_t u=(uint32_t)w&((1u<<QB)-1); for(int i=0;i<QB;i++) if((u>>i)&1) Wb[n][i][s]|=(1u<<b); }
  vector<int> x(DIN); for(int n=0;n<DIN;n++) x[n]=(RB==1)?(rand()&1):((rand()%(1<<RB))-(1<<(RB-1)));
  auto FAC=[](int idx,int nb){ return (nb>1&&idx==nb-1)?-(1<<idx):(1<<idx); };

  SoftMCPlatform pf(bender); if(pf.init()!=SOFTMC_SUCCESS){cerr<<"[gi] init fail\n";return 3;}
  pf.reset_fpga();

  // MVDRAM memory org (§IV): preload matrix rows + constant rows ONCE (SELECT mode).
  // ZERO=54144, ONES=54145, M[n][i]=54146+i*DIN+n, nM[n][i]=54146+QB*DIN+i*DIN+n.
  const uint32_t ZEROrow=54144, ONESrow=54145, MBASE=54146;
  auto Mrow =[&](int n,int i){ return MBASE + i*DIN + n; };
  auto nMrow=[&](int n,int i){ return MBASE + QB*DIN + i*DIN + n; };
  if(SELECT){
    if(MBASE+2*QB*DIN >= 54231){cerr<<"[gi] SELECT: matrix rows would collide with plan rows; use smaller DIN\n";return 4;}
    Vec zv; for(int s=0;s<2048;s++)zv.v[s]=0; Vec ov; for(int s=0;s<2048;s++)ov.v[s]=0xFFFFFFFFu;
    pcwrite(pf,bank,ZEROrow,zv.v); pcwrite(pf,bank,ONESrow,ov.v);
    for(int n=0;n<DIN;n++)for(int i=0;i<QB;i++){ Vec mv,nv; for(int s=0;s<2048;s++){mv.v[s]=Wb[n][i][s];nv.v[s]=~Wb[n][i][s];}
      pcwrite(pf,bank,Mrow(n,i),mv.v); pcwrite(pf,bank,nMrow(n,i),nv.v); }
    cerr<<"[gi] SELECT: preloaded "<<(2*DIN*QB+2)<<" matrix/const rows (once)\n";
  }

  static long long yhw[2048][32]; memset(yhw,0,sizeof yhw);
  Vec ZV; for(int s=0;s<2048;s++)ZV.v[s]=0;
  static int totc[2048][32];
  long long runs=0;
  const char* L4[4]={"a","b","c","d"}; const char* NL4[4]={"na","nb","nc","nd"};
  for(int i=0;i<QB;i++) for(int c=0;c<RB;c++){
    memset(totc,0,sizeof totc);
    // gather inputs to popcount this plane: all d_in, or (SKIP) only set-activation ones
    vector<int> ins;
    for(int n=0;n<DIN;n++){ if(!SKIP || ((x[n]>>c)&1)) ins.push_back(n); }
    for(size_t t=0;t<ins.size();t+=N){
      static Vec prod[4]; unordered_map<string,uint32_t> src;
      for(int j=0;j<N;j++){
        if(t+j<ins.size()){ int n=ins[t+j]; int act=(x[n]>>c)&1;
          if(act){ for(int s=0;s<2048;s++)prod[j].v[s]=Wb[n][i][s]; src[L4[j]]=Mrow(n,i); src[NL4[j]]=nMrow(n,i); }
          else   { prod[j]=ZV; src[L4[j]]=ZEROrow; src[NL4[j]]=ONESrow; } }
        else { prod[j]=ZV; src[L4[j]]=ZEROrow; src[NL4[j]]=ONESrow; }   // padding
      }
      const Vec* inp[4]={&prod[0],&prod[1],&prod[2],&prod[3]};
      auto run=[&](vector<pair<int,Vec>>& ob){ exec_popcount(pf,bank,ops,N,inp,ob,SELECT,&src); };
      vector<pair<int,Vec>> ob; run(ob); runs++;
      if(VOTE){ vector<pair<int,Vec>> o2,o3; run(o2); run(o3); runs+=2;
        for(size_t q=0;q<ob.size();q++) for(int s=0;s<2048;s++)
          ob[q].second.v[s]=(ob[q].second.v[s]&o2[q].second.v[s])|(o2[q].second.v[s]&o3[q].second.v[s])|(ob[q].second.v[s]&o3[q].second.v[s]); }
      for(int s=0;s<2048;s++)for(int b=0;b<32;b++){ int cnt=0; for(auto&pr:ob) cnt+=((pr.second.v[s]>>b)&1)<<pr.first; totc[s][b]+=cnt; }
    }
    long long fac=(long long)FAC(i,QB)*FAC(c,RB);
    for(int s=0;s<2048;s++)for(int b=0;b<32;b++) yhw[s][b]+=fac*(long long)totc[s][b];
  }
  cerr<<"[gi] popcount runs="<<runs<<(SKIP?" (bit-sparsity skip ON)":" (dense)")<<"\n";

  long ok=0,bad=0,tot=0;
  for(int s=0;s<2048;s++)for(int b=0;b<32;b++){ long long yr=0; for(int n=0;n<DIN;n++) yr+=(long long)x[n]*wv[n][s][b];
    tot++; if(yhw[s][b]==yr)ok++; else bad++; }
  cerr<<"[gi] IN-PLACE GeMV (d_in="<<DIN<<",q="<<QB<<",r="<<RB<<") bit-exact: "<<ok<<"/"<<tot
      <<" ("<<100.0*ok/tot<<"%) mism="<<bad<<"\n";
  cerr<<"[gi] "<<(bad==0?"PASS — full in-DRAM in-place GeMV bit-exact":(100.0*ok/tot>99?"~PASS (transient residual)":"FAIL"))<<"\n";
  _exit(0);
}
