// Extended ACCUM_XBP isolation: recover the EFFECTIVE per-plane weight the
// fabric applies, with a PLAIN rdRow read, across ALL 8 production planes
// (neg,shift) = (0,0)..(0,6),(1,7) — the factor set [1,2,4,8,16,32,64,-128].
// Method: write a row whose every 32-bit segment has popcount == 1 (one bit
// set). For each plane p: set_acc_weight(neg_p,shift_p); accum_read; flush;
// acc[s] should == intended_weight_p (since pc==1). Prints effective vs
// intended per plane. This tells whether the doubling seen in production is a
// SHIFT/PLANE bug (would show here, plain read) or a MAJ3-BODY interaction
// (would NOT show here). Argv: ./accxbp-hw2-exe <bender> <bank> <row> [double|actpad]
#include "instruction.h"
#include "prog.h"
#include "platform.h"
#include "../util.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
using namespace std;

static void write_row(SoftMCPlatform& pf, int bank, uint32_t row, const uint32_t* seg) {
  static const int CHUNK_COLS[3] = {43, 43, 42};
  int cs = 0;
  for (int ch = 0; ch < 3; ch++) {
    int n = CHUNK_COLS[ch]; const uint32_t* cd = seg + cs * 16;
    Program p;
    p.add_inst(SMC_LI(8, CASR)); p.add_inst(SMC_LI(bank, BAR));
    p.add_inst(SMC_LI(row, RAR)); p.add_inst(SMC_LI(cs * 8, CAR));
    p.add_below(PRE(BAR, 0, 0)); p.add_below(ACT(BAR, 0, RAR, 0));
    for (int k = 0; k < n; k++) {
      const uint32_t* sl = cd + k * 16;
      for (int s = 0; s < 16; s++) { p.add_inst(SMC_LI(sl[s], PATTERN_REG)); p.add_inst(SMC_LDWD(PATTERN_REG, s)); }
      p.add_below(WRITE(BAR, CAR, 1)); p.add_inst(SMC_SLEEP(8));
    }
    p.add_inst(SMC_SLEEP(8)); p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_SLEEP(4)); p.add_inst(SMC_END());
    pf.execute(p); cs += n;
  }
}

// one accumulate read; mode: 0 = plain rdRow, 1 = read row TWICE in one program,
// 2 = extra ACT/PRE before the read (doubleACT-ish padding).
static void accum_read(SoftMCPlatform& pf, int bank, uint32_t row, int label, int mode) {
  Program p;
  p.add_inst(SMC_LI(bank, BAR)); p.add_inst(SMC_LI(8, CASR)); p.add_inst(SMC_LI(128, NUM_COLS_REG));
  p.add_inst(SMC_LI(0, PATTERN_REG));
  for (int i = 0; i < 16; i++) p.add_inst(SMC_LDWD(PATTERN_REG, i));
  if (mode == 2) { p.add_below(ACT(BAR, 0, RAR, 0)); p.add_inst(SMC_SLEEP(4)); p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_SLEEP(4)); }
  p.add_below(rdRow_immediate_label(BAR, row, label));
  if (mode == 1) { p.add_inst(SMC_SLEEP(4)); p.add_below(PRE(BAR,0,0)); p.add_inst(SMC_SLEEP(4)); p.add_below(rdRow_immediate_label(BAR, row, label+50000)); }
  p.add_inst(SMC_END());
  pf.execute(p);
}

int main(int argc, char** argv) {
  if (argc < 4) { fprintf(stderr, "Usage: %s <bender> <bank> <row> [double|actpad]\n", argv[0]); return 1; }
  setvbuf(stdout, NULL, _IONBF, 0);
  int bender = atoi(argv[1]), bank = atoi(argv[2]); uint32_t row = (uint32_t)atoi(argv[3]);
  int rmode = 0; if (argc > 4 && !strcmp(argv[4], "double")) rmode = 1; else if (argc > 4 && !strcmp(argv[4], "actpad")) rmode = 2;
  bool batch = (argc > 4 && !strcmp(argv[4], "batch"));
  bool bodybatch = (argc > 4 && !strcmp(argv[4], "bodybatch"));
  SoftMCPlatform pf(bender);
  if (pf.init() != SOFTMC_SUCCESS) { fprintf(stderr, "init failed\n"); return 1; }
  pf.reset_fpga(); pf.set_aref(false);

  // Production plane weights (neg,shift): (0,0)..(0,6),(1,7).
  const int NB = 8; int NEG[NB]={0,0,0,0,0,0,0,1}, SH[NB]={0,1,2,3,4,5,6,7};
  auto intended = [&](int p)->long { return (NEG[p]?-1L:1L) * (1L<<SH[p]); };
  // row: every 32-bit segment popcount == 1 (bit 0 set).
  vector<uint32_t> onerow(2048, 1u);

  if (bodybatch) {
    // Batch cadence + a LONG body preamble before each plane's rdRow, mimicking
    // the production MAJ3 body length (doubleACTs on OTHER rows + NOPs, so the
    // read row's pc==1-per-segment content is preserved). Tests whether the
    // long-body read latency lets set_acc_weight(b+1) race plane b's beats.
    vector<uint32_t> onerow(2048, 1u), zerorow(2048, 0u);
    for (int p = 0; p < NB; p++) write_row(pf, bank, row + p, onerow.data());
    write_row(pf, bank, row + 20, zerorow.data());   // scratch pair = zero so
    write_row(pf, bank, row + 21, zerorow.data());   // any stray accum adds 0
    pf.set_readback_mode_accxbp(); pf.set_readback_mode_accxbp();
    int blabel = 9500;
    for (int p = 0; p < NB; p++) {
      pf.set_acc_weight(NEG[p], SH[p]);
      // long body: several doubleACTs on a scratch pair far away + rdRow(row+p)
      Program pr;
      pr.add_inst(SMC_LI(bank, BAR)); pr.add_inst(SMC_LI(8, CASR)); pr.add_inst(SMC_LI(128, NUM_COLS_REG));
      pr.add_inst(SMC_LI(0, PATTERN_REG));
      for (int i=0;i<16;i++) pr.add_inst(SMC_LDWD(PATTERN_REG,i));
      for (int j=0;j<4;j++){ pr.add_below(doubleACT(30,1, row+20, row+21)); pr.add_inst(SMC_SLEEP(6));
                             pr.add_below(PRE(BAR,0,0)); pr.add_inst(SMC_SLEEP(6)); }
      for (int j=0;j<20;j++) pr.add_inst(all_nops());
      pr.add_below(rdRow_immediate_label(BAR, row + p, blabel++));
      pr.add_inst(all_nops()); pr.add_inst(all_nops());
      pf.execute(pr);
    }
    pf.flush_acc();
    vector<uint8_t> buf(8192,0); int got=pf.receiveData(buf.data(),8192);
    pf.set_readback_mode(false); pf.set_readback_mode(false); pf.drain_stray(1500,8);
    // pc==1 per segment => acc[s] == sum of effective per-plane weights; but with
    // distinct-per-plane needed for regression we used onerow (same pc). So just
    // report the summed acc vs sum(true) and sum(lag) to detect the shift.
    long accmid=0; { int32_t v; memcpy(&v,buf.data()+512*4,4); accmid=v; }
    long sum_true=1+2+4+8+16+32+64-128;                 // = -1
    long sum_lag =2+4+8+16+32+64-128-128;               // = -130
    long sum_2x  =2+4+8+16+32+64+128-256;               // = -2
    printf("[accxbp2] BODYBATCH recv=%d  acc[mid]=%ld  (Sum true=%ld, Sum lag[factors b+1]=%ld, Sum 2x=%ld)\n",
           got, accmid, sum_true, sum_lag, sum_2x);
    return 0;
  }
  if (batch) {
    // Mimic production cadence: accumulate ALL 8 planes back-to-back with NO
    // per-plane flush/drain, then ONE flush. Distinct ramp rows per plane so
    // the effective weights are recoverable by regressing acc[s] on pc_p[s].
    // Dumps acc + the 8 rows' per-segment popcounts for offline regression.
    vector<vector<uint32_t>> rows(NB, vector<uint32_t>(2048));
    for (int p = 0; p < NB; p++)
      for (int s = 0; s < 2048; s++) { int nb=(s*7 + p*13)%33; rows[p][s]=(nb>=32)?0xFFFFFFFFu:((1u<<nb)-1u); }
    // Pre-write 8 DISTINCT rows (row+p) FIRST — production writes masks before
    // the plane loop, then fires set_acc_weight->execute back-to-back with NO
    // write in between (the tight cadence that lets set_acc_weight(b+1) race
    // plane b's still-draining read).
    for (int p = 0; p < NB; p++) write_row(pf, bank, row + p, rows[p].data());
    pf.set_readback_mode_accxbp(); pf.set_readback_mode_accxbp();   // enter+clear ONCE
    int blabel = 9000;
    for (int p = 0; p < NB; p++) {
      pf.set_acc_weight(NEG[p], SH[p]);
      accum_read(pf, bank, row + p, blabel++, 0);   // tight: no write, no flush between
    }
    pf.flush_acc();
    vector<uint8_t> buf(8192,0); int got=pf.receiveData(buf.data(),8192);
    pf.set_readback_mode(false); pf.set_readback_mode(false); pf.drain_stray(1500,8);
    FILE* fa=fopen("/home/deni/Claude/accxbp_root_2026_08_02/hw2_batch_acc.txt","w");
    fprintf(fa,"# s acc pc0 pc1 pc2 pc3 pc4 pc5 pc6 pc7  (intended w=[1,2,4,8,16,32,64,-128])\n");
    for (int s=0;s<2048;s++){ int32_t v; memcpy(&v,buf.data()+s*4,4);
      fprintf(fa,"%d %d",s,v);
      for(int p=0;p<NB;p++) fprintf(fa," %d",__builtin_popcount(rows[p][s]));
      fprintf(fa,"\n"); }
    fclose(fa);
    printf("[accxbp2] BATCH (8 planes back-to-back, one flush) recv=%d -> dumped hw2_batch_acc.txt\n",got);
    return 0;
  }
  printf("[accxbp2] read-mode=%s  per-plane effective weight (pc==1 so acc==weight):\n",
         rmode==1?"double-read":rmode==2?"act-pad":"plain");
  write_row(pf, bank, row, onerow.data());
  int label = 7000, allbad = 0;
  for (int p = 0; p < NB; p++) {
    pf.set_readback_mode_accxbp();          // enter + clear
    pf.set_readback_mode_accxbp();
    pf.set_acc_weight(NEG[p], SH[p]);
    accum_read(pf, bank, row, label++, rmode);
    pf.flush_acc();
    vector<uint8_t> buf(8192, 0);
    int got = pf.receiveData(buf.data(), 8192);
    // effective weight = median over segments of acc (pc==1 => acc==weight)
    vector<long> vals; int nz=0;
    for (int s = 0; s < 2048; s++) { int32_t v; memcpy(&v, buf.data()+s*4, 4); if(s<2048) vals.push_back(v); if(v)nz++; }
    // mode(most common) value
    long eff = vals[0]; { // pick the value at a mid segment as representative
      long c0=vals[16], c1=vals[64], c2=vals[512]; eff = (c0==c1||c0==c2)?c0:(c1==c2?c1:c0); }
    long want = intended(p);
    bool ok = (eff == want);
    if(!ok) allbad++;
    printf("  plane %d (neg=%d,shift=%d) intended=%+5ld  effective=%+6ld  nz=%d/2048 recv=%d %s\n",
           p, NEG[p], SH[p], want, eff, nz, got, ok?"OK":"<< MISMATCH");
    pf.set_readback_mode(false); pf.set_readback_mode(false); pf.drain_stray(1500, 8);
  }
  printf("[accxbp2] %s (%d plane-weight mismatches, read-mode=%s)\n",
         allbad?"DOUBLING/MISMATCH PRESENT":"ALL PLANES CORRECT", allbad, rmode==1?"double-read":rmode==2?"act-pad":"plain");
  return allbad?1:0;
}
