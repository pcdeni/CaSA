// build9 streaming-fetch silicon validation + wall A/B. The build9 image
// (trailer 0xDBC0DE08) adds a ping-pong IMEM: with STREAM_EN set, the
// host loads program N+1 into the idle bank while N executes, so the DDR
// command bus stops waiting on per-program host round-trips.
//
// Test: write N distinct known rows. Then read all N in READ mode two
// ways and check the returned 8192-B rows match, IN ORDER, both ways:
//   (A) LEGACY: execute(read_prog) + receiveData(8192) per row (the
//       per-program-round-trip cadence).
//   (B) STREAM: set_stream_en(true); stream_start(); stream_send() all N
//       read programs back-to-back; stream_recv(8192) x N; stream_stop().
// Report wall for each — B should be materially faster (fetch-idle hidden
// under execution) while returning byte-identical in-order results.
// Argv: ./stream-hw-exe <bender> <bank> <base_row> [N=64]
#include "instruction.h"
#include "prog.h"
#include "platform.h"
#include "../util.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
using namespace std;
using clk = std::chrono::steady_clock;
static double ms_since(clk::time_point t){ return std::chrono::duration_cast<
  std::chrono::microseconds>(clk::now()-t).count()/1000.0; }

// per-column write a full 2048-u32 row (the pcwrite idiom, 3 chunks).
static void write_row(SoftMCPlatform& pf, int bank, uint32_t row,
                      const uint32_t* seg) {
  static const int CH[3] = {43,43,42};
  int cs=0;
  for(int c=0;c<3;c++){
    int n=CH[c]; const uint32_t* cd=seg+cs*16;
    Program p;
    p.add_inst(SMC_LI(8,CASR)); p.add_inst(SMC_LI(bank,BAR));
    p.add_inst(SMC_LI(row,RAR)); p.add_inst(SMC_LI(cs*8,CAR));
    p.add_below(PRE(BAR,0,0)); p.add_below(ACT(BAR,0,RAR,0));
    for(int k=0;k<n;k++){ const uint32_t* sl=cd+k*16;
      for(int s=0;s<16;s++){ p.add_inst(SMC_LI(sl[s],PATTERN_REG)); p.add_inst(SMC_LDWD(PATTERN_REG,s)); }
      p.add_below(WRITE(BAR,CAR,1)); p.add_inst(SMC_SLEEP(8)); }
    p.add_inst(SMC_SLEEP(8)); p.add_below(PRE(BAR,0,0)); p.add_inst(SMC_SLEEP(4)); p.add_inst(SMC_END());
    pf.execute(p); cs+=n;
  }
}
// a single READ program of `row` (READ mode → 8192 B back).
static Program read_prog(int bank, uint32_t row, int label){
  Program p;
  p.add_inst(SMC_LI(bank,BAR)); p.add_inst(SMC_LI(8,CASR)); p.add_inst(SMC_LI(128,NUM_COLS_REG));
  p.add_below(rdRow_immediate_label(BAR,row,label)); p.add_inst(SMC_END());
  return p;
}
static void mkpat(uint32_t row, uint32_t* seg){
  uint32_t s = 0xA5A50000u ^ (row*2654435761u);
  for(int i=0;i<2048;i++){ s=s*1664525u+1013904223u; seg[i]=s; }
}

int main(int argc,char** argv){
  if(argc<4){ fprintf(stderr,"Usage: %s <bender> <bank> <base_row> [N=64]\n",argv[0]); return 1; }
  setvbuf(stdout,NULL,_IONBF,0);
  int bender=atoi(argv[1]), bank=atoi(argv[2]); uint32_t base=(uint32_t)atoi(argv[3]);
  int N=(argc>4)?atoi(argv[4]):64;
  SoftMCPlatform pf(bender);
  if(pf.init()!=SOFTMC_SUCCESS){ fprintf(stderr,"[stream] init failed\n"); return 1; }
  pf.reset_fpga(); pf.set_aref(false);
  pf.set_readback_mode(false); pf.set_readback_mode(false);   // READ mode

  // N distinct rows, known patterns.
  vector<vector<uint32_t>> pat(N, vector<uint32_t>(2048));
  for(int i=0;i<N;i++){ mkpat(base+i, pat[i].data()); write_row(pf,bank,base+i,pat[i].data()); }

  auto check_row=[&](const uint8_t* buf,int i)->bool{
    return memcmp(buf, pat[i].data(), 8192)==0; };

  // ---- A: legacy execute+receive per program ----
  int badA=0; auto tA=clk::now();
  { vector<uint8_t> b(8192);
    for(int i=0;i<N;i++){ Program p=read_prog(bank,base+i,3000+i);
      pf.execute(p); int rc=pf.receiveData(b.data(),8192);
      if(rc!=8192 || !check_row(b.data(),i)) badA++; } }
  double wA=ms_since(tA);
  printf("[stream] A legacy   : %d rows, %d bad, %.1f ms (%.3f ms/row)\n",
         N,badA,wA,wA/N);

  // ---- B: streaming ----
  int badB=0;
  pf.set_stream_en(true);
  pf.stream_start();
  auto tB=clk::now();
  for(int i=0;i<N;i++){ Program p=read_prog(bank,base+i,4000+i); pf.stream_send(p); }
  { vector<uint8_t> b(8192);
    for(int i=0;i<N;i++){ int rc=pf.stream_recv(b.data(),8192);
      if(rc!=8192 || !check_row(b.data(),i)) badB++; } }
  double wB=ms_since(tB);
  pf.stream_stop();
  pf.set_stream_en(false);
  printf("[stream] B streaming: %d rows, %d bad, %.1f ms (%.3f ms/row)  speedup %.2fx\n",
         N,badB,wB,wB/N, wA>0? wA/wB : 0.0);

  int fails = (badA?1:0)+(badB?1:0);
  printf("[stream] %s (A_bad=%d B_bad=%d)\n", fails?"FAIL":"ALL_PASS", badA, badB);
  return fails?1:0;
}
