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

  // ---- C/D: SEG_POP framing (2048 B/row, one popcount byte per 32-bit
  // segment) — the production readback mode; NEVER validated under
  // streaming until 2026-07-22 (the full-model gate divergence). ----
  // Oracle = stream-vs-legacy BYTE EQUALITY (the hardware's segment
  // packing is its own layout; uniform-pattern suites can't see order,
  // so a host-side popcount model is not a valid reference here).
  pf.set_readback_mode_segpop(); pf.set_readback_mode_segpop();
  vector<vector<uint8_t>> cref(N, vector<uint8_t>(2048));
  int badC=0;
  for(int i=0;i<N;i++){ Program p=read_prog(bank,base+i,5000+i);
    pf.execute(p); int rc=pf.receiveData(cref[i].data(),2048);
    if(rc!=2048) badC++; }
  printf("[stream] C legacy segpop : %d rows, %d short-reads (reference set)\n",N,badC);

  int badD=0; long badD_bytes=0;
  pf.set_stream_en(true);
  pf.stream_start(2048);
  for(int i=0;i<N;i++){ Program p=read_prog(bank,base+i,6000+i); pf.stream_send(p); }
  { vector<uint8_t> b(2048);
    for(int i=0;i<N;i++){ int rc=pf.stream_recv(b.data(),2048);
      int m=2048;
      if(rc==2048){ m=0; for(int s=0;s<2048;s++) if(b[s]!=cref[i][s]) m++; }
      if(m){ badD++; badD_bytes+=m; } } }
  pf.stream_stop();
  printf("[stream] D stream segpop : %d rows, %d differ from legacy (%ld bytes)\n",N,badD,badD_bytes);

  // ---- E: MIXED sized-mode stream — the production V2 round shape:
  // per row, 3 write-chunk programs (payload 0) then one segpop read
  // (payload 2048), recv per round like the phase-1 server. Fresh
  // patterns so E cannot pass on stale content. ----
  int badE=0; long badE_bytes=0;
  { vector<vector<uint32_t>> pat2(N, vector<uint32_t>(2048));
    for(int i=0;i<N;i++) mkpat(base+i+7777u, pat2[i].data());
    vector<vector<uint8_t>> ebuf(N, vector<uint8_t>(2048));
    pf.stream_start(SoftMCPlatform::STREAM_SIZED);
    vector<uint8_t> b(2048);
    for(int i=0;i<N;i++){
      int cs=0;
      for(int c=0;c<3;c++){
        int n=(c<2)?43:42;
        Program p;
        p.add_inst(SMC_LI(8,CASR)); p.add_inst(SMC_LI(bank,BAR));
        p.add_inst(SMC_LI(base+i,RAR)); p.add_inst(SMC_LI(cs*8,CAR));
        p.add_below(PRE(BAR,0,0)); p.add_below(ACT(BAR,0,RAR,0));
        for(int k=0;k<n;k++){ const uint32_t* sl=pat2[i].data()+ (cs+k)*16;
          for(int s=0;s<16;s++){ p.add_inst(SMC_LI(sl[s],PATTERN_REG)); p.add_inst(SMC_LDWD(PATTERN_REG,s)); }
          p.add_below(WRITE(BAR,CAR,1)); p.add_inst(SMC_SLEEP(8)); }
        p.add_inst(SMC_SLEEP(8)); p.add_below(PRE(BAR,0,0)); p.add_inst(SMC_SLEEP(4)); p.add_inst(SMC_END());
        pf.stream_send(p, 0);
        cs+=n;
      }
      Program r=read_prog(bank,base+i,7000+i);
      pf.stream_send(r, 2048);
      int rc=pf.stream_recv(ebuf[i].data(),2048);
      if(rc!=2048) badE++;
    }
    pf.stream_stop();
    pf.set_stream_en(false);
    // Oracle: legacy segpop re-read of the SAME rows (content = pat2 as
    // the mixed stream left it) must match the streamed bytes. Keep all
    // references to build a shift map when rows mismatch.
    vector<vector<uint8_t>> lref2(N, vector<uint8_t>(2048));
    for(int i=0;i<N;i++){ Program p=read_prog(bank,base+i,8000+i);
      pf.execute(p); int rc=pf.receiveData(lref2[i].data(),2048);
      int m=2048;
      if(rc==2048){ m=0; for(int s=0;s<2048;s++) if(lref2[i][s]!=ebuf[i][s]) m++; }
      if(m){ badE++; badE_bytes+=m; } }
    if(badE){
      int clean_i=-1;
      for(int i=0;i<N;i++){ int m=0;
        for(int s=0;s<2048;s++) if(ebuf[i][s]!=lref2[i][s]) m++;
        if(m==0){ clean_i=i; break; } }
      printf("[stream] E clean row index: %d\n", clean_i);
      printf("[stream] E shift map (streamed row i best-matches legacy row j):\n  ");
      for(int i=0;i<N && i<12;i++){
        int bestj=-1, bestm=1<<30;
        for(int j=0;j<N;j++){ int m=0;
          for(int s=0;s<2048;s++) if(ebuf[i][s]!=lref2[j][s]) m++;
          if(m<bestm){ bestm=m; bestj=j; } }
        printf("%d->%d(%d) ", i, bestj, bestm);
      }
      printf("\n");
    }

    // E2: SAME mixed content, but send-all-then-recv-all (no interleave).
    { int bad2=0;
      pf.set_stream_en(true); pf.stream_start(SoftMCPlatform::STREAM_SIZED);
      for(int i=0;i<N;i++){
        int cs=0;
        for(int c=0;c<3;c++){
          int n=(c<2)?43:42;
          Program p;
          p.add_inst(SMC_LI(8,CASR)); p.add_inst(SMC_LI(bank,BAR));
          p.add_inst(SMC_LI(base+i,RAR)); p.add_inst(SMC_LI(cs*8,CAR));
          p.add_below(PRE(BAR,0,0)); p.add_below(ACT(BAR,0,RAR,0));
          for(int k=0;k<n;k++){ const uint32_t* sl=pat2[i].data()+(cs+k)*16;
            for(int s=0;s<16;s++){ p.add_inst(SMC_LI(sl[s],PATTERN_REG)); p.add_inst(SMC_LDWD(PATTERN_REG,s)); }
            p.add_below(WRITE(BAR,CAR,1)); p.add_inst(SMC_SLEEP(8)); }
          p.add_inst(SMC_SLEEP(8)); p.add_below(PRE(BAR,0,0)); p.add_inst(SMC_SLEEP(4)); p.add_inst(SMC_END());
          pf.stream_send(p,0); cs+=n;
        }
        Program r=read_prog(bank,base+i,7100+i); pf.stream_send(r,2048);
      }
      vector<uint8_t> rb(2048);
      for(int i=0;i<N;i++){ int rc=pf.stream_recv(rb.data(),2048);
        int m=2048;
        if(rc==2048){ m=0; for(int s=0;s<2048;s++) if(rb[s]!=lref2[i][s]) m++; }
        if(m) bad2++; }
      pf.stream_stop(); pf.set_stream_en(false);
      printf("[stream] E2 mixed no-interleave: %d/%d rows differ\n",bad2,N);
    }
    // E4: E's exact mixed+interleaved cadence, but rows spaced by 16
    // (the production pool discipline). If clean, E/E2's failures were
    // WRITE DISTURB from this test's consecutive-row layout — after a
    // row's write+read, writing its neighbors degraded it before the
    // late oracle re-read — and streaming is exonerated at the
    // primitive level.
    { int bad4=0;
      pf.set_stream_en(true); pf.stream_start(SoftMCPlatform::STREAM_SIZED);
      vector<uint8_t> rb(2048);
      vector<vector<uint8_t>> e4buf(N, vector<uint8_t>(2048));
      for(int i=0;i<N;i++){
        uint32_t row4 = base + 64u + 16u*(uint32_t)i;
        int cs=0;
        for(int c=0;c<3;c++){
          int n=(c<2)?43:42;
          Program p;
          p.add_inst(SMC_LI(8,CASR)); p.add_inst(SMC_LI(bank,BAR));
          p.add_inst(SMC_LI(row4,RAR)); p.add_inst(SMC_LI(cs*8,CAR));
          p.add_below(PRE(BAR,0,0)); p.add_below(ACT(BAR,0,RAR,0));
          for(int k=0;k<n;k++){ const uint32_t* sl=pat2[i].data()+(cs+k)*16;
            for(int s=0;s<16;s++){ p.add_inst(SMC_LI(sl[s],PATTERN_REG)); p.add_inst(SMC_LDWD(PATTERN_REG,s)); }
            p.add_below(WRITE(BAR,CAR,1)); p.add_inst(SMC_SLEEP(8)); }
          p.add_inst(SMC_SLEEP(8)); p.add_below(PRE(BAR,0,0)); p.add_inst(SMC_SLEEP(4)); p.add_inst(SMC_END());
          pf.stream_send(p,0); cs+=n;
        }
        Program r=read_prog(bank,row4,7300+i); pf.stream_send(r,2048);
        int rc=pf.stream_recv(e4buf[i].data(),2048);
        if(rc!=2048) bad4++;
      }
      pf.stream_stop(); pf.set_stream_en(false);
      for(int i=0;i<N;i++){
        uint32_t row4 = base + 64u + 16u*(uint32_t)i;
        Program p=read_prog(bank,row4,8300+i);
        pf.execute(p); int rc=pf.receiveData(rb.data(),2048);
        int m=2048;
        if(rc==2048){ m=0; for(int s=0;s<2048;s++) if(rb[s]!=e4buf[i][s]) m++; }
        if(m) bad4++;
      }
      printf("[stream] E4 mixed interleaved, stride-16 rows: %d/%d rows differ\n",bad4,N);
    }
    // E5: E4's shape but READ MODE (8192 B raw rows). Clean here + dirty
    // E4 => the corruption is SEG_POP-engine-specific under streamed
    // h2c backpressure (the accxbp read_seq_incoming class). Dirty here
    // too => mode-independent readback interference.
    { int bad5=0;
      pf.set_readback_mode(false); pf.set_readback_mode(false);
      pf.set_stream_en(true); pf.stream_start(SoftMCPlatform::STREAM_SIZED);
      vector<uint8_t> rb8(8192);
      vector<vector<uint8_t>> e5buf(N, vector<uint8_t>(8192));
      for(int i=0;i<N;i++){
        uint32_t row5 = base + 64u + 16u*(uint32_t)i;   // same rows as E4 (hold pat2)
        int cs=0;
        for(int c=0;c<3;c++){
          int n=(c<2)?43:42;
          Program p;
          p.add_inst(SMC_LI(8,CASR)); p.add_inst(SMC_LI(bank,BAR));
          p.add_inst(SMC_LI(row5,RAR)); p.add_inst(SMC_LI(cs*8,CAR));
          p.add_below(PRE(BAR,0,0)); p.add_below(ACT(BAR,0,RAR,0));
          for(int k=0;k<n;k++){ const uint32_t* sl=pat2[i].data()+(cs+k)*16;
            for(int s=0;s<16;s++){ p.add_inst(SMC_LI(sl[s],PATTERN_REG)); p.add_inst(SMC_LDWD(PATTERN_REG,s)); }
            p.add_below(WRITE(BAR,CAR,1)); p.add_inst(SMC_SLEEP(8)); }
          p.add_inst(SMC_SLEEP(8)); p.add_below(PRE(BAR,0,0)); p.add_inst(SMC_SLEEP(4)); p.add_inst(SMC_END());
          pf.stream_send(p,0); cs+=n;
        }
        Program r=read_prog(bank,row5,7400+i); pf.stream_send(r,8192);
        int rc=pf.stream_recv(e5buf[i].data(),8192);
        if(rc!=8192) bad5++;
      }
      pf.stream_stop(); pf.set_stream_en(false);
      for(int i=0;i<N;i++){
        if(memcmp(e5buf[i].data(), pat2[i].data(), 8192)!=0) bad5++;
      }
      printf("[stream] E5 mixed READ-mode, stride-16: %d/%d rows differ (raw-row oracle)\n",bad5,N);
      pf.set_readback_mode_segpop(); pf.set_readback_mode_segpop();
    }
    // E6: E4's segpop shape but SMALL writes (1 chunk, ~43 cols) — does
    // the damage scale with h2c load size?
    { int bad6=0;
      pf.set_stream_en(true); pf.stream_start(SoftMCPlatform::STREAM_SIZED);
      vector<uint8_t> rb(2048);
      vector<vector<uint8_t>> e6buf(N, vector<uint8_t>(2048));
      for(int i=0;i<N;i++){
        uint32_t row6 = base + 64u + 16u*(uint32_t)i;
        Program p;
        p.add_inst(SMC_LI(8,CASR)); p.add_inst(SMC_LI(bank,BAR));
        p.add_inst(SMC_LI(row6,RAR)); p.add_inst(SMC_LI(0,CAR));
        p.add_below(PRE(BAR,0,0)); p.add_below(ACT(BAR,0,RAR,0));
        for(int k=0;k<43;k++){ const uint32_t* sl=pat2[i].data()+k*16;
          for(int s=0;s<16;s++){ p.add_inst(SMC_LI(sl[s],PATTERN_REG)); p.add_inst(SMC_LDWD(PATTERN_REG,s)); }
          p.add_below(WRITE(BAR,CAR,1)); p.add_inst(SMC_SLEEP(8)); }
        p.add_inst(SMC_SLEEP(8)); p.add_below(PRE(BAR,0,0)); p.add_inst(SMC_SLEEP(4)); p.add_inst(SMC_END());
        pf.stream_send(p,0);
        Program r=read_prog(bank,row6,7500+i); pf.stream_send(r,2048);
        int rc=pf.stream_recv(e6buf[i].data(),2048);
        if(rc!=2048) bad6++;
      }
      pf.stream_stop(); pf.set_stream_en(false);
      vector<uint8_t> lb(2048);
      for(int i=0;i<N;i++){
        uint32_t row6 = base + 64u + 16u*(uint32_t)i;
        Program p=read_prog(bank,row6,8500+i);
        pf.execute(p); int rc=pf.receiveData(lb.data(),2048);
        int m=2048;
        if(rc==2048){ m=0; for(int s=0;s<2048;s++) if(lb[s]!=e6buf[i][s]) m++; }
        if(m) bad6++;
      }
      printf("[stream] E6 mixed segpop, SMALL writes: %d/%d rows differ\n",bad6,N);
    }

    // E7: E4's shape with REPLICATED masks (512-segment period x4 — the
    // client's partial-slice replication; the first-diverging-op class).
    { int bad7=0;
      vector<vector<uint32_t>> pat7(N, vector<uint32_t>(2048));
      for(int i=0;i<N;i++){ mkpat(base+i+31000u, pat7[i].data());
        for(int s=512;s<2048;s++) pat7[i][s]=pat7[i][s%512]; }
      vector<vector<uint8_t>> e7buf(N, vector<uint8_t>(2048));
      pf.stream_start(SoftMCPlatform::STREAM_SIZED);
      for(int i=0;i<N;i++){
        uint32_t row7 = base + 64u + 16u*(uint32_t)i;
        int cs=0;
        for(int c=0;c<3;c++){
          int n=(c<2)?43:42;
          Program p;
          p.add_inst(SMC_LI(8,CASR)); p.add_inst(SMC_LI(bank,BAR));
          p.add_inst(SMC_LI(row7,RAR)); p.add_inst(SMC_LI(cs*8,CAR));
          p.add_below(PRE(BAR,0,0)); p.add_below(ACT(BAR,0,RAR,0));
          for(int k=0;k<n;k++){ const uint32_t* sl=pat7[i].data()+(cs+k)*16;
            for(int q=0;q<16;q++){ p.add_inst(SMC_LI(sl[q],PATTERN_REG)); p.add_inst(SMC_LDWD(PATTERN_REG,q)); }
            p.add_below(WRITE(BAR,CAR,1)); p.add_inst(SMC_SLEEP(8)); }
          p.add_inst(SMC_SLEEP(8)); p.add_below(PRE(BAR,0,0)); p.add_inst(SMC_SLEEP(4)); p.add_inst(SMC_END());
          pf.stream_send(p,0); cs+=n;
        }
        Program r=read_prog(bank,row7,7700+i); pf.stream_send(r,2048);
        int rc=pf.stream_recv(e7buf[i].data(),2048);
        if(rc!=2048) bad7++;
      }
      pf.stream_stop();
      vector<uint8_t> lb(2048);
      long bad7b=0;
      for(int i=0;i<N;i++){
        uint32_t row7 = base + 64u + 16u*(uint32_t)i;
        Program p=read_prog(bank,row7,8700+i);
        pf.execute(p); int rc=pf.receiveData(lb.data(),2048);
        int m=2048;
        if(rc==2048){ m=0; for(int q=0;q<2048;q++) if(lb[q]!=e7buf[i][q]) m++; }
        if(m){ bad7++; bad7b+=m; }
      }
      printf("[stream] E7 REPLICATED masks: %d/%d rows differ (%ld bytes)\n",N>32?32:N,N,bad7b);
      printf("[stream] E7 verdict: %s\n", bad7? "REPRODUCED" : "clean");
    }

    // E8: the ALTERNATION primitive (2026-07-23) — mirror the LOAD-mix
    // server shape: legacy exec/recv "MM3D-analog" requests interleaved
    // with per-request streamed sessions ("V2-analog"), including the
    // stream_start/stop churn. Checks BOTH sides stay exact.
    { int badL=0, badS=0;
      vector<uint32_t> patL(2048), patS(2048);
      vector<uint8_t> refL(2048), gotL(2048), gotS(2048), lb2(2048);
      for(int it=0; it<16; it++){
        uint32_t rowL = base + 64u + 16u*(uint32_t)(it*2);
        uint32_t rowS = base + 64u + 16u*(uint32_t)(it*2+1);
        mkpat(rowL^0xAAAA5555u^it, patL.data());
        mkpat(rowS^0x5555AAAAu^it, patS.data());
        // legacy "MM3D-analog": write + segpop read (execute/receiveData)
        write_row(pf,bank,rowL,patL.data());
        { Program p=read_prog(bank,rowL,10000+it); pf.execute(p);
          if(pf.receiveData(refL.data(),2048)!=2048) badL++; }
        // streamed "V2-analog" session: write + read, then close
        pf.stream_start(SoftMCPlatform::STREAM_SIZED);
        { int cs=0;
          for(int c=0;c<3;c++){ int n=(c<2)?43:42;
            Program p;
            p.add_inst(SMC_LI(8,CASR)); p.add_inst(SMC_LI(bank,BAR));
            p.add_inst(SMC_LI(rowS,RAR)); p.add_inst(SMC_LI(cs*8,CAR));
            p.add_below(PRE(BAR,0,0)); p.add_below(ACT(BAR,0,RAR,0));
            for(int k=0;k<n;k++){ const uint32_t* sl=patS.data()+(cs+k)*16;
              for(int q=0;q<16;q++){ p.add_inst(SMC_LI(sl[q],PATTERN_REG)); p.add_inst(SMC_LDWD(PATTERN_REG,q)); }
              p.add_below(WRITE(BAR,CAR,1)); p.add_inst(SMC_SLEEP(8)); }
            p.add_inst(SMC_SLEEP(8)); p.add_below(PRE(BAR,0,0)); p.add_inst(SMC_SLEEP(4)); p.add_inst(SMC_END());
            pf.stream_send(p,0); cs+=n; }
          Program r=read_prog(bank,rowS,10100+it); pf.stream_send(r,2048);
          if(pf.stream_recv(gotS.data(),2048)!=2048) badS++;
        }
        pf.stream_stop();
        // legacy re-reads AFTER the session: both rows must hold
        { Program p=read_prog(bank,rowL,10200+it); pf.execute(p);
          if(pf.receiveData(gotL.data(),2048)!=2048) badL++;
          int m=0; for(int q=0;q<2048;q++) if(gotL[q]!=refL[q]) m++;
          if(m) badL++; }
        { Program p=read_prog(bank,rowS,10300+it); pf.execute(p);
          if(pf.receiveData(lb2.data(),2048)!=2048) badS++;
          int m=0; for(int q=0;q<2048;q++) if(lb2[q]!=gotS[q]) m++;
          if(m) badS++; }
      }
      printf("[stream] E8 ALTERNATION x16: legacy-side bad=%d streamed-side bad=%d -> %s\n",
             badL, badS, (badL||badS)? "REPRODUCED" : "clean");
    }

    // E9 (2026-07-23): write -> PRE -> segpop-read WITHIN ONE PROGRAM —
    // the production V2 exec program's internal shape, the last
    // untested primitive. Both arms start from IDENTICAL DRAM state;
    // oracle = streamed-vs-legacy byte equality; per-program verdicts
    // (program 0 = IDLE-path load, 1.. = swap-path) + parity histogram
    // of differing byte indices (the odd-index signature decode).
    { const int NE=16;
      vector<vector<uint32_t>> pw(NE, vector<uint32_t>(2048));
      vector<vector<uint32_t>> prr(NE, vector<uint32_t>(2048));
      for(int i=0;i<NE;i++){ mkpat(0xE9000000u+i, pw[i].data());
                             mkpat(0xE9100000u+i, prr[i].data()); }
      auto prep=[&](){
        for(int i=0;i<NE;i++){
          uint32_t rowW=base+64u+16u*(uint32_t)i, rowR=rowW+8u;
          // zero rowW, write known pattern to rowR (legacy, outside test)
          Program z; z.add_inst(SMC_LI(8,CASR)); z.add_inst(SMC_LI(bank,BAR));
          z.add_inst(SMC_LI(128,NUM_COLS_REG)); z.add_below(PRE(BAR,0,0));
          z.add_below(wrRow_immediate_label(BAR,rowW,0u,11000+i));
          z.add_inst(SMC_END()); pf.execute(z);
          write_row(pf,bank,rowR,prr[i].data());
        } };
      auto mkprog=[&](int i,int label)->Program{
        uint32_t rowW=base+64u+16u*(uint32_t)i, rowR=rowW+8u;
        Program p;
        p.add_inst(SMC_LI(8,CASR)); p.add_inst(SMC_LI(bank,BAR));
        p.add_inst(SMC_LI(rowW,RAR)); p.add_inst(SMC_LI(0,CAR));
        p.add_below(PRE(BAR,0,0)); p.add_below(ACT(BAR,0,RAR,0));
        for(int k=0;k<43;k++){ const uint32_t* sl=pw[i].data()+k*16;
          for(int q=0;q<16;q++){ p.add_inst(SMC_LI(sl[q],PATTERN_REG)); p.add_inst(SMC_LDWD(PATTERN_REG,q)); }
          p.add_below(WRITE(BAR,CAR,1)); p.add_inst(SMC_SLEEP(8)); }
        p.add_inst(SMC_SLEEP(8)); p.add_below(PRE(BAR,0,0)); p.add_inst(SMC_SLEEP(6));
        p.add_inst(SMC_LI(128,NUM_COLS_REG));
        p.add_below(rdRow_immediate_label(BAR,rowR,label));
        p.add_inst(SMC_END());
        return p; };
      vector<vector<uint8_t>> l9(NE, vector<uint8_t>(2048)), s9(NE, vector<uint8_t>(2048));
      prep();
      for(int i=0;i<NE;i++){ Program p=mkprog(i,11100+i);
        pf.execute(p); pf.receiveData(l9[i].data(),2048); }
      prep();   // identical state again
      pf.stream_start(SoftMCPlatform::STREAM_SIZED);
      for(int i=0;i<NE;i++){ Program p=mkprog(i,11200+i); pf.stream_send(p,2048); }
      for(int i=0;i<NE;i++) pf.stream_recv(s9[i].data(),2048);
      pf.stream_stop();
      int bad9=0; long oddd=0, evend=0;
      printf("[stream] E9 per-program diffs:");
      for(int i=0;i<NE;i++){ int m=0;
        for(int q=0;q<2048;q++) if(l9[i][q]!=s9[i][q]){ m++; if(q&1) oddd++; else evend++; }
        printf(" %d", m); if(m) bad9++; }
      printf("\n[stream] E9 within-program write->read: %d/%d programs differ (odd-idx %ld, even-idx %ld) -> %s\n",
             bad9, NE, oddd, evend, bad9? "REPRODUCED" : "clean");
    }
    // E3: interleaved recv, PURE reads (rows already hold pat2).
    { int bad3=0;
      pf.set_stream_en(true); pf.stream_start(SoftMCPlatform::STREAM_SIZED);
      vector<uint8_t> rb(2048);
      for(int i=0;i<N;i++){
        Program r=read_prog(bank,base+i,7200+i); pf.stream_send(r,2048);
        int rc=pf.stream_recv(rb.data(),2048);
        int m=2048;
        if(rc==2048){ m=0; for(int s=0;s<2048;s++) if(rb[s]!=lref2[i][s]) m++; }
        if(m) bad3++;
      }
      pf.stream_stop(); pf.set_stream_en(false);
      printf("[stream] E3 pure-read interleaved: %d/%d rows differ\n",bad3,N);
    }
  }
  printf("[stream] E mixed stream  : %d rows, %d differ from legacy re-read (%ld bytes)\n",N,badE,badE_bytes);

  // ---- F: discriminate write-corruption vs read-after-write artifact.
  // F1: STREAMED writes only (pat3), then LEGACY read      -> f1
  // F2: LEGACY   writes       (pat3), then LEGACY read      -> f2
  //     f1 != f2  => streamed WRITES corrupt row content.
  // F3: streamed writes + TWO back-to-back streamed reads (r1, r2),
  //     then legacy read (r3): r1!=r3 but r2==r3 => transient
  //     first-read-after-write read-path artifact.
  { vector<uint32_t> pat3(2048); mkpat(base+31337u, pat3.data());
    uint32_t row = base;   // reuse row 0
    auto wchunks=[&](bool stream){
      int cs=0;
      for(int c=0;c<3;c++){
        int n=(c<2)?43:42;
        Program p;
        p.add_inst(SMC_LI(8,CASR)); p.add_inst(SMC_LI(bank,BAR));
        p.add_inst(SMC_LI(row,RAR)); p.add_inst(SMC_LI(cs*8,CAR));
        p.add_below(PRE(BAR,0,0)); p.add_below(ACT(BAR,0,RAR,0));
        for(int k=0;k<n;k++){ const uint32_t* sl=pat3.data()+(cs+k)*16;
          for(int s=0;s<16;s++){ p.add_inst(SMC_LI(sl[s],PATTERN_REG)); p.add_inst(SMC_LDWD(PATTERN_REG,s)); }
          p.add_below(WRITE(BAR,CAR,1)); p.add_inst(SMC_SLEEP(8)); }
        p.add_inst(SMC_SLEEP(8)); p.add_below(PRE(BAR,0,0)); p.add_inst(SMC_SLEEP(4)); p.add_inst(SMC_END());
        if(stream) pf.stream_send(p,0); else pf.execute(p);
        cs+=n;
      } };
    auto lread=[&](uint8_t* dst){
      Program p=read_prog(bank,row,9000+(rand()&255));
      pf.execute(p); return pf.receiveData(dst,2048); };
    vector<uint8_t> f1(2048), f2(2048), r1(2048), r2(2048), r3(2048);
    // F1
    pf.set_stream_en(true); pf.stream_start(SoftMCPlatform::STREAM_SIZED);
    wchunks(true); pf.stream_stop(); pf.set_stream_en(false);
    lread(f1.data());
    // F2
    wchunks(false); lread(f2.data());
    int f12=0; for(int s=0;s<2048;s++) if(f1[s]!=f2[s]) f12++;
    // F3
    pf.set_stream_en(true); pf.stream_start(SoftMCPlatform::STREAM_SIZED);
    wchunks(true);
    { Program p=read_prog(bank,row,9500); pf.stream_send(p,2048); }
    { Program p=read_prog(bank,row,9501); pf.stream_send(p,2048); }
    pf.stream_recv(r1.data(),2048); pf.stream_recv(r2.data(),2048);
    pf.stream_stop(); pf.set_stream_en(false);
    lread(r3.data());
    int r13=0, r23=0, r12=0;
    for(int s=0;s<2048;s++){ if(r1[s]!=r3[s]) r13++; if(r2[s]!=r3[s]) r23++; if(r1[s]!=r2[s]) r12++; }
    printf("[stream] F verdicts: f1_vs_f2=%d (streamed-writes integrity), "
           "r1_vs_r3=%d r2_vs_r3=%d r1_vs_r2=%d (read-after-write)\n",
           f12, r13, r23, r12);
  }
  pf.set_readback_mode(false); pf.set_readback_mode(false);

  // ---- G: RETIRED 2026-07-22 night (STREAM_HW_G=1 to re-enable).
  // Its claim (streamed segpop read == post-session legacy read after a
  // write) is arm E's exact comparison, which is CLEAN 32/32 on
  // build-10. G's own triangle (pre-session oracle vs r-reads vs
  // post-read) is self-inconsistent in a fix- and row-invariant way —
  // a probe-design artifact (served its purpose pre-fix by proving
  // stickiness), not a silicon claim. Kept for archaeology.
  if(getenv("STREAM_HW_G") && atoi(getenv("STREAM_HW_G"))!=0){
  // ---- G: transient vs sticky segpop misalignment after a zero-beat
  // (write-only) streamed program. [write X][read X][read X][read X] in
  // one sized session; if only r1 is dirty -> per-read transient (host
  // workaround = one discarded read after write batches); r1..r3 all
  // dirty -> sticky until mode reset (RTL fix required).
  { uint32_t rowX = 45950u;   // SCREENED pool row (the base+600 pick was unscreened and decays between reads — G v1 artifact)
    vector<uint32_t> patX(2048); mkpat(rowX^0xBEEF, patX.data());
    // baseline content + legacy reference
    { int cs=0; for(int c=0;c<3;c++){ int n=(c<2)?43:42;
        Program p;
        p.add_inst(SMC_LI(8,CASR)); p.add_inst(SMC_LI(bank,BAR));
        p.add_inst(SMC_LI(rowX,RAR)); p.add_inst(SMC_LI(cs*8,CAR));
        p.add_below(PRE(BAR,0,0)); p.add_below(ACT(BAR,0,RAR,0));
        for(int k=0;k<n;k++){ const uint32_t* sl=patX.data()+(cs+k)*16;
          for(int s=0;s<16;s++){ p.add_inst(SMC_LI(sl[s],PATTERN_REG)); p.add_inst(SMC_LDWD(PATTERN_REG,s)); }
          p.add_below(WRITE(BAR,CAR,1)); p.add_inst(SMC_SLEEP(8)); }
        p.add_inst(SMC_SLEEP(8)); p.add_below(PRE(BAR,0,0)); p.add_inst(SMC_SLEEP(4)); p.add_inst(SMC_END());
        pf.execute(p); cs+=n; } }
    vector<uint8_t> L(2048);
    { Program p=read_prog(bank,rowX,9700); pf.execute(p); pf.receiveData(L.data(),2048); }
    // streamed: small write (same content, 1 chunk) then 3 reads
    vector<uint8_t> r1(2048),r2(2048),r3(2048);
    pf.set_stream_en(true); pf.stream_start(SoftMCPlatform::STREAM_SIZED);
    { Program p;
      p.add_inst(SMC_LI(8,CASR)); p.add_inst(SMC_LI(bank,BAR));
      p.add_inst(SMC_LI(rowX,RAR)); p.add_inst(SMC_LI(0,CAR));
      p.add_below(PRE(BAR,0,0)); p.add_below(ACT(BAR,0,RAR,0));
      for(int k=0;k<43;k++){ const uint32_t* sl=patX.data()+k*16;
        for(int s=0;s<16;s++){ p.add_inst(SMC_LI(sl[s],PATTERN_REG)); p.add_inst(SMC_LDWD(PATTERN_REG,s)); }
        p.add_below(WRITE(BAR,CAR,1)); p.add_inst(SMC_SLEEP(8)); }
      p.add_inst(SMC_SLEEP(8)); p.add_below(PRE(BAR,0,0)); p.add_inst(SMC_SLEEP(4)); p.add_inst(SMC_END());
      pf.stream_send(p,0); }
    { Program p=read_prog(bank,rowX,9701); pf.stream_send(p,2048); }
    { Program p=read_prog(bank,rowX,9702); pf.stream_send(p,2048); }
    { Program p=read_prog(bank,rowX,9703); pf.stream_send(p,2048); }
    pf.stream_recv(r1.data(),2048); pf.stream_recv(r2.data(),2048); pf.stream_recv(r3.data(),2048);
    pf.stream_stop(); pf.set_stream_en(false);
    int m1=0,m2=0,m3=0;
    for(int s=0;s<2048;s++){ if(r1[s]!=L[s])m1++; if(r2[s]!=L[s])m2++; if(r3[s]!=L[s])m3++; }
    printf("[stream] G write+3reads: r1=%d r2=%d r3=%d bytes differ vs legacy\n",m1,m2,m3);
    // Discriminator: POST-session legacy re-read L2. If r-reads match L2
    // (and L2 differs from the stale pre-session L), the streamed reads
    // were FAITHFUL and G's partial 1-chunk write genuinely changed the
    // row — a test-design artifact, not a silicon bug.
    { vector<uint8_t> L2(2048);
      Program p=read_prog(bank,rowX,9704);
      pf.execute(p); pf.receiveData(L2.data(),2048);
      int r1L2=0, LL2=0;
      for(int s=0;s<2048;s++){ if(r1[s]!=L2[s])r1L2++; if(L[s]!=L2[s])LL2++; }
      printf("[stream] G discriminator: r1_vs_L2=%d  L_vs_L2=%d\n", r1L2, LL2);
    }
    printf("[stream] G sample L : %02x %02x %02x %02x %02x %02x %02x %02x\n",
           L[0],L[1],L[2],L[3],L[4],L[5],L[6],L[7]);
    printf("[stream] G sample r1: %02x %02x %02x %02x %02x %02x %02x %02x\n",
           r1[0],r1[1],r1[2],r1[3],r1[4],r1[5],r1[6],r1[7]);
  }

  }
  int fails = (badA?1:0)+(badB?1:0)+(badC?1:0)+(badD?1:0)+(badE?1:0);
  printf("[stream] %s (A=%d B=%d C=%d D=%d E=%d)\n",
         fails?"FAIL":"ALL_PASS", badA,badB,badC,badD,badE);
  return fails?1:0;
}
