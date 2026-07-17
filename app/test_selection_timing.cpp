// Selection-vs-timing characterization (2026-07-17) — the per-module timing
// study the SAFARI reply (SiMRA-DRAM#1, §3.2 advice) recommends, aimed at
// the selection question our thread raised: for an APA pair (R_F, R_S) at
// k differing subarray-local bits, WHICH of the 2^k predicted rows fire,
// and how does the (t1, t2) timing shift that on OUR module?
//
// Method (mirrors the sublattice/broadcast detection): per trial —
//   1. per-column write a marker pattern to anchor A (= R_F),
//   2. wrRow zeros to every OTHER predicted-set member {A ^ S : S ⊆ bits(d), S != 0},
//   3. doubleACT(t1, t2, A, A^d),
//   4. read every member; fired(S) := member content == marker (Multi-RowCopy
//      deposit of R_F's data), zero = not fired, else mixed.
// CSV per member row to stdout: k,d,A_local,t12,t23,S,fired(0/1/2=mixed)
// Summary per (k, t12, t23) to stderr: full-set rate, mean fired fraction.
//
// Argv: <bender> <bank> [pairs_per_k=4] [seed=11]
// Subarray: [45312, 45952) (bender 2/0 characterized region). Avoids the
// production tuple rows [45340..45343, 45436..45439, 45724..45727,
// 45820..45823] as anchors/members so calibration rows stay untouched.
#include "instruction.h"
#include "prog.h"
#include "platform.h"
#include "../util.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
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
static void read_row(SoftMCPlatform& pf,int bank,uint32_t row,uint8_t* out){
  Program p; p.add_inst(SMC_LI(8,CASR)); p.add_inst(SMC_LI(bank,BAR)); p.add_inst(SMC_LI(128,NUM_COLS_REG));
  p.add_below(PRE(BAR,0,0)); p.add_inst(SMC_SLEEP(6));
  p.add_below(rdRow_immediate(BAR,row)); p.add_inst(all_nops()); p.add_inst(all_nops()); p.add_below(PRE(BAR,0,0)); p.add_inst(SMC_END());
  pf.execute(p); pf.receiveData(out,8192); }

int main(int argc,char**argv){
  if(argc<3){fprintf(stderr,"Usage: %s <bender> <bank> [pairs_per_k=4] [seed=11]\n",argv[0]);return 1;}
  int bender=atoi(argv[1]),bank=atoi(argv[2]);
  int PPK=(argc>3)?atoi(argv[3]):4; int seed=(argc>4)?atoi(argv[4]):11;
  const uint32_t SUB=45312, SLEN=640;
  // production tuple rows to avoid
  set<uint32_t> avoid;
  for(uint32_t r:{45340u,45436u,45724u,45820u}) for(uint32_t i=0;i<4;i++) avoid.insert(r+i);

  const int T12S[4]={5,10,20,30};
  const int T23S[2]={1,2};
  srand(seed);

  SoftMCPlatform pf(bender); if(pf.init()!=SOFTMC_SUCCESS){fprintf(stderr,"init fail\n");return 3;}
  pf.reset_fpga();

  uint32_t marker[2048]; for(int s=0;s<2048;s++) marker[s]=0x5EEC0000u^(s*2654435761u);
  uint8_t mexp[8192]; for(int s=0;s<2048;s++) for(int b=0;b<4;b++) mexp[s*4+b]=(uint8_t)((marker[s]>>(8*b))&0xFF);

  printf("k,d,A_local,t12,t23,S,fired\n");
  // stats[k][t12i][t23i] = {trials, full_set_hits, fired_members, member_slots}
  long st[6][4][2][4]; memset(st,0,sizeof st);

  for(int k=1;k<=5;k++){
    // sample PPK distinct (d, A) pairs for this k
    int made=0, guard=0;
    while(made<PPK && guard++<4000){
      // random d with popcount k over bits 0..9
      uint32_t d=0; while(__builtin_popcount(d)!=k) d=((uint32_t)rand())&0x3FF;
      uint32_t A_local=((uint32_t)rand())%SLEN;
      // whole predicted set must stay in-subarray and avoid production rows
      bool ok=true; vector<uint32_t> members;
      // enumerate S subsets of bits(d)
      vector<int> bits; for(int b=0;b<10;b++) if(d>>b&1) bits.push_back(b);
      for(uint32_t m=0;m<(1u<<k);m++){
        uint32_t S=0; for(int j=0;j<k;j++) if(m>>j&1) S|=1u<<bits[j];
        uint32_t loc=A_local^S;
        if(loc>=SLEN){ok=false;break;}
        uint32_t row=SUB+loc;
        if(avoid.count(row)){ok=false;break;}
        members.push_back(S);
      }
      if(!ok) continue;
      made++;
      uint32_t A=SUB+A_local;
      for(int ti=0;ti<4;ti++) for(int tj=0;tj<2;tj++){
        int t12=T12S[ti], t23=T23S[tj];
        // 1. marker to A (rewritten every trial — the APA may disturb it)
        pcwrite(pf,bank,A,marker);
        // 2. zero all other members in one program
        { Program p; p.add_inst(SMC_LI(8,CASR)); p.add_inst(SMC_LI(bank,BAR)); p.add_inst(SMC_LI(128,NUM_COLS_REG));
          p.add_below(PRE(BAR,0,0)); int lbl=0;
          for(uint32_t S:members){ if(S==0) continue;
            p.add_below(wrRow_immediate_label(BAR,SUB+(A_local^S),0u,lbl++)); }
          p.add_inst(SMC_SLEEP(6)); p.add_below(PRE(BAR,0,0)); p.add_inst(SMC_END());
          pf.execute(p); }
        // 3. the APA
        { Program p; p.add_inst(SMC_LI(8,CASR)); p.add_inst(SMC_LI(bank,BAR)); p.add_inst(SMC_LI(128,NUM_COLS_REG));
          p.add_below(PRE(BAR,0,0)); p.add_inst(SMC_SLEEP(6));
          p.add_below(doubleACT(t12,t23,A,SUB+(A_local^d)));
          p.add_inst(SMC_SLEEP(6)); p.add_below(PRE(BAR,0,0)); p.add_inst(SMC_SLEEP(6));
          p.add_inst(SMC_END()); pf.execute(p); }
        // 4. read + classify each non-anchor member
        int fired_cnt=0, member_cnt=0; bool full=true;
        for(uint32_t S:members){ if(S==0) continue;
          uint8_t buf[8192]; read_row(pf,bank,SUB+(A_local^S),buf);
          int match=0, zero=0;
          for(int b=0;b<8192;b++){ if(buf[b]==mexp[b]) match++; if(buf[b]==0) zero++; }
          int fired = (match>8192-256)?1:((zero>8192-256)?0:2);
          printf("%d,%u,%u,%d,%d,%u,%d\n",k,d,A_local,t12,t23,S,fired);
          member_cnt++; if(fired==1) fired_cnt++; else full=false;
        }
        st[k][ti][tj][0]++; if(full) st[k][ti][tj][1]++;
        st[k][ti][tj][2]+=fired_cnt; st[k][ti][tj][3]+=member_cnt;
      }
    }
    if(made<PPK) fprintf(stderr,"[sel-t] k=%d: only %d/%d pairs sampled\n",k,made,PPK);
  }
  fprintf(stderr,"[sel-t] k t12 t23  full-set%%  fired-members%%  (trials)\n");
  for(int k=1;k<=5;k++) for(int ti=0;ti<4;ti++) for(int tj=0;tj<2;tj++){
    if(!st[k][ti][tj][0]) continue;
    fprintf(stderr,"[sel-t] %d  %2d  %d    %5.1f      %5.1f      (%ld)\n",
      k,T12S[ti],T23S[tj],
      100.0*st[k][ti][tj][1]/st[k][ti][tj][0],
      st[k][ti][tj][3]?100.0*st[k][ti][tj][2]/st[k][ti][tj][3]:0.0,
      st[k][ti][tj][0]);
  }
  _exit(0);
}
