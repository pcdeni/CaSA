// MAJ3-on-an-explicit-tuple test (2026-07-17). Items 1 & 4:
//  - Item 4 (DIMM 1/3 rescue): run the canonical MAJ3 on an algebraically
//    ENUMERATED 4-row tuple (from enumerate_rescue.py) on benders 1/3, whose
//    historical "MAJ unreliability" was tuple-geometry ties, not silicon.
//  - Item 1 (MAJ on fast-loaded operands): operands are written then MAJ3'd
//    with the same load->vote path the loader feeds.
// For each of 7 truth-table cases: write A,B,C to rows[0..2], frac=ones to
// rows[3], discharge x3, doubleACT(0,0,Rf,Rs), read rows[0], compare to MAJ3.
// Argv: <bender> <bank> <Rf> <Rs> <r0> <r1> <r2> <r3>
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
static const int CHUNK_COLS[3]={43,43,42};
static int BANK=0;
static void pcwrite(SoftMCPlatform& pf,uint32_t row,uint32_t word){
  int cs=0; for(int ch=0;ch<3;ch++){ int n=CHUNK_COLS[ch];
    Program p; p.add_inst(SMC_LI(8,CASR)); p.add_inst(SMC_LI(BANK,BAR)); p.add_inst(SMC_LI(row,RAR)); p.add_inst(SMC_LI(cs*8,CAR));
    p.add_below(PRE(BAR,0,0)); p.add_below(ACT(BAR,0,RAR,0));
    for(int k=0;k<n;k++){ for(int s=0;s<16;s++){ p.add_inst(SMC_LI(word,PATTERN_REG)); p.add_inst(SMC_LDWD(PATTERN_REG,s)); }
      p.add_below(WRITE(BAR,CAR,1)); p.add_inst(SMC_SLEEP(8)); }
    p.add_inst(SMC_SLEEP(8)); p.add_below(PRE(BAR,0,0)); p.add_inst(SMC_SLEEP(4)); p.add_inst(SMC_END()); pf.execute(p); cs+=n; }
}
static Program frac_b(int tf,int r){ Program p; p.add_inst(all_nops()); p.add_inst(SMC_LI(r,RF_REG));
  int nc=2+tf; nc+=4-(nc%4); Mininst q[nc]; for(int i=0;i<nc;i++)q[i]=SMC_NOP();
  q[0]=SMC_ACT(BAR,0,RF_REG,0); q[tf+1]=SMC_PRE(BAR,0,0); for(int i=0;i<nc;i+=4)p.add_inst(q[i],q[i+1],q[i+2],q[i+3]); return p; }
static uint32_t maj3(uint32_t a,uint32_t b,uint32_t c){ return (a&b)|(a&c)|(b&c); }

int main(int argc,char**argv){
  if(argc<9){cerr<<"Usage: "<<argv[0]<<" <bender> <bank> <Rf> <Rs> <r0> <r1> <r2> <r3>\n";return 1;}
  int bender=atoi(argv[1]); BANK=atoi(argv[2]);
  uint32_t Rf=strtoul(argv[3],0,10), Rs=strtoul(argv[4],0,10);
  uint32_t r[4]={(uint32_t)strtoul(argv[5],0,10),(uint32_t)strtoul(argv[6],0,10),
                 (uint32_t)strtoul(argv[7],0,10),(uint32_t)strtoul(argv[8],0,10)};
  SoftMCPlatform pf(bender); if(pf.init()!=SOFTMC_SUCCESS){cerr<<"init fail\n";return 3;}
  pf.reset_fpga();
  struct TC{uint32_t a,b,c; const char*n;};
  TC cases[7]={{0xF0F0F0F0,0xCCCCCCCC,0xAAAAAAAA,"truth"},{0x12345678,0x9ABCDEF0,0,"AND"},
    {0x12345678,0x9ABCDEF0,0xFFFFFFFF,"OR"},{0xCAFEF00D,0xCAFEF00D,0x12345678,"x,x,y=x"},
    {0xAAAAAAAA,0x55555555,0xDEADBEEF,"x,~x,y=y"},{0,0,0xFFFFFFFF,"minority"},{0xFFFFFFFF,0xFFFFFFFF,0,"majority"}};
  int pass=0; long bit_err_tot=0;
  for(auto&t:cases){
    pcwrite(pf,r[0],t.a); pcwrite(pf,r[1],t.b); pcwrite(pf,r[2],t.c); pcwrite(pf,r[3],0xFFFFFFFF);
    Program p; p.add_inst(SMC_LI(8,CASR)); p.add_inst(SMC_LI(BANK,BAR)); p.add_inst(SMC_LI(128,NUM_COLS_REG));
    p.add_below(PRE(BAR,0,0)); p.add_inst(SMC_SLEEP(6));
    for(int j=0;j<3;j++){ p.add_inst(SMC_SLEEP(6)); p.add_below(frac_b(0,r[3])); p.add_inst(SMC_SLEEP(6)); }
    p.add_inst(SMC_SLEEP(6)); p.add_below(PRE(BAR,0,0)); p.add_inst(SMC_SLEEP(6));
    p.add_below(doubleACT(0,0,Rf,Rs)); p.add_inst(SMC_SLEEP(6)); p.add_below(PRE(BAR,0,0)); p.add_inst(SMC_SLEEP(6));
    p.add_below(rdRow_immediate(BAR,r[0])); p.add_inst(all_nops()); p.add_inst(all_nops()); p.add_below(PRE(BAR,0,0)); p.add_inst(SMC_END());
    pf.execute(p); uint8_t buf[8192]; pf.receiveData(buf,8192);
    uint32_t want=maj3(t.a,t.b,t.c); long be=0;
    for(int s=0;s<2048;s++){uint32_t hv;memcpy(&hv,&buf[s*4],4); be+=__builtin_popcount(hv^want);}
    bit_err_tot+=be; bool ok=(be==0); pass+=ok;
    printf("  %-9s bit_err=%ld/65536 %s\n",t.n,be,ok?"PASS":"");
  }
  printf("[maj3-tuple] bender=%d tuple={%u,%u,%u,%u} Rf=%u Rs=%u -> %d/7 exact, mean bit_err=%.1f/65536\n",
         bender,r[0],r[1],r[2],r[3],Rf,Rs,pass,bit_err_tot/7.0);
  fflush(stdout);
  _exit(0);
}
