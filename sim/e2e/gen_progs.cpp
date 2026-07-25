// e2e_sim stimulus generator (2026-07-24): emits the EXACT instruction
// words the silicon probes used, via the real Program encoder — the sim
// is fed by the same machine code as the hardware, by construction.
//   s1_read.hex   = probe P1: branch-free 46-inst unrolled 8-beat read
//   s2_brloop.hex = probe P2: 6-inst BL loop, no DDR ops (the wedge)
#include "instruction.h"
#include "prog.h"
#include "platform.h"
#include "../util.h"
#include <cstdio>
#include <cstdint>

static void emit(Program& p, const char* path){
  uint64_t* w = (uint64_t*) p.get_inst_array();
  int n = p.size()/8;
  FILE* f = fopen(path,"w");
  for(int i=0;i<n;i++) fprintf(f,"%016lx\n",(unsigned long)w[i]);
  fclose(f);
  printf("%s: %d insts\n",path,n);
}

int main(int argc, char** argv){
  const char* dir = argc>1 ? argv[1] : ".";
  char path[512];
  int bank=1; uint32_t row=60000;

  { Program p;
    p.add_inst(SMC_LI(bank,BAR));
    p.add_inst(SMC_LI(8,CASR));
    p.add_inst(SMC_LI(row,RAR));
    p.add_inst(SMC_LI(0,CAR));
    p.add_below(PRE(BAR,0,0));
    p.add_below(ACT(BAR,0,RAR,0));
    p.add_inst(SMC_SLEEP(4));
    for(int c=0;c<8;c++){ p.add_below(READ(BAR,CAR,1)); p.add_inst(SMC_SLEEP(4)); }
    p.add_inst(SMC_SLEEP(8));
    p.add_below(PRE(BAR,0,0));
    p.add_inst(SMC_SLEEP(4));
    p.add_inst(SMC_END());
    snprintf(path,sizeof path,"%s/s1_read.hex",dir); emit(p,path); }

  { Program q;
    q.add_inst(SMC_LI(0, LOOP_ROWS));
    q.add_inst(SMC_LI(2, NUM_ROWS_REG));
    q.add_label("PROBE_BR");
    q.add_inst(SMC_ADDI(LOOP_ROWS, 1, LOOP_ROWS));
    q.add_branch(q.BR_TYPE::BL, LOOP_ROWS, NUM_ROWS_REG, "PROBE_BR");
    q.add_inst(SMC_SLEEP(4));
    q.add_inst(SMC_END());
    snprintf(path,sizeof path,"%s/s2_brloop.hex",dir); emit(q,path); }

  // phase-2: the REAL production shapes (branch loops via util.cpp).
  auto full_read=[&](uint32_t r, int label)->Program{
    Program p;
    p.add_inst(SMC_LI(bank,BAR)); p.add_inst(SMC_LI(8,CASR));
    p.add_inst(SMC_LI(128,NUM_COLS_REG));
    p.add_below(rdRow_immediate_label(BAR,r,label)); p.add_inst(SMC_END());
    return p; };
  { Program p=full_read(row,7001);
    snprintf(path,sizeof path,"%s/s4_read128.hex",dir); emit(p,path); }

  // E14 loop_prog shape verbatim (test_stream_hw.cpp): LDWD prologue +
  // 128-iteration WRITE loop, slot-0 = column index each iteration.
  { uint32_t wrow=60016;
    uint32_t prol[16];
    for(int q2=0;q2<16;q2++) prol[q2]=0xE1400000u + 0x01010101u*(uint32_t)q2 + 0xB1u;
    Program p;
    p.add_inst(SMC_LI(8,CASR)); p.add_inst(SMC_LI(bank,BAR));
    p.add_inst(SMC_LI(wrow,RAR)); p.add_inst(SMC_LI(0,CAR));
    p.add_below(PRE(BAR,0,0)); p.add_below(ACT(BAR,0,RAR,0));
    for(int q2=0;q2<16;q2++){ p.add_inst(SMC_LI(prol[q2],PATTERN_REG));
      p.add_inst(SMC_LDWD(PATTERN_REG,q2)); }
    p.add_inst(SMC_LI(0,LOOP_ROWS)); p.add_inst(SMC_LI(128,NUM_ROWS_REG));
    p.add_label("E2E_WLOOP");
      p.add_inst(SMC_ADDI(LOOP_ROWS,0,PATTERN_REG));
      p.add_inst(SMC_LDWD(PATTERN_REG,0));
      p.add_below(WRITE(BAR,CAR,1));
      p.add_inst(SMC_SLEEP(8));
      p.add_inst(SMC_ADDI(LOOP_ROWS,1,LOOP_ROWS));
    p.add_branch(p.BR_TYPE::BL, LOOP_ROWS, NUM_ROWS_REG, "E2E_WLOOP");
    p.add_inst(SMC_SLEEP(8)); p.add_below(PRE(BAR,0,0));
    p.add_inst(SMC_SLEEP(4)); p.add_inst(SMC_END());
    snprintf(path,sizeof path,"%s/s5_wrloop.hex",dir); emit(p,path);
    Program r=full_read(wrow,7002);
    snprintf(path,sizeof path,"%s/s6_readw.hex",dir); emit(r,path); }
  return 0;
}
