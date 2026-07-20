// FCDRAM NOT detection (2026-07-17, Task S3) — replays FCDRAM's (papers/
// FCDRAM.pdf, Yüksel et al.) neighboring-subarray detection protocol §4/§5
// on our modules:
//   1) init BOTH subarrays (background), 2) doubleACT(t12,t23,src,dst) =
//   their ACT R_F → PRE → ACT R_L with reduced timings, 3) WR a marker
//   while rows are open — R_L-subarray activated rows latch the marker;
//   R_F-subarray coupled rows receive the INVERSE through the shared
//   sense amplifiers on HALF the bitlines (open-bitline architecture),
//   4) scan both subarrays and classify rows by one-bit fraction.
// Two complementary marker rounds (background 0 + marker FF; background FF
// + marker 00) so a genuine NOT bitline mask inverts consistently in both.
// Paper prediction (§4.3): works on SK Hynix (our DIMM 1/3 = Mfr. H
// class), "neither simultaneous nor sequential" activation on Micron (our
// DIMM 0/2 = Mfr. M class) — bender 0 is the predicted-negative control.
// Row grid: close/middle/far from the shared boundary in each subarray,
// both directions (src below / src above), timing grid over (t12, t23)
// with t23 = their "reduced tRP" knob.
// Argv: <bender> <bank> <subA_base> <subB_base> [sublen=640] [ncols=128]
// Output: per (pair, timing, round): rows with full-marker latch (act'd
// set) and half-inverted rows (NOT candidates) + their stable bit masks.
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

static SoftMCPlatform* PF = nullptr;
static int BANK = 0;

static void init_rows(uint32_t base, uint32_t n, uint32_t val) {
  // 16 rows/program: wrRow_immediate expands to ~80 instructions; larger
  // batches silently overflow the 2048-instruction IMEM (rows past the
  // truncation point never get written — the v1 run's artifact).
  uint32_t done = 0;
  while (done < n) {
    uint32_t chunk = (n - done > 16) ? 16 : (n - done);
    Program p;
    p.add_inst(SMC_LI(8, CASR));
    p.add_inst(SMC_LI(BANK, BAR));
    p.add_inst(SMC_LI(128, NUM_COLS_REG));
    p.add_below(PRE(BAR, 0, 0));
    int lbl = 0;
    for (uint32_t r = 0; r < chunk; r++)
      p.add_below(wrRow_immediate_label(BAR, base + done + r, val, lbl++));
    p.add_inst(SMC_SLEEP(6));
    p.add_below(PRE(BAR, 0, 0));
    p.add_inst(SMC_END());
    PF->execute(p);
    done += chunk;
  }
}
// doubleACT(t12,t23,src,dst) then column-writes of `marker` into the open
// row buffer (no re-ACT): all simultaneously-activated rows latch it;
// coupled neighbor-subarray rows receive the inverse on shared bitlines.
static void op_and_write(uint32_t src, uint32_t dst, int t12, int t23,
                         uint32_t marker, int ncols) {
  Program p;
  p.add_inst(SMC_LI(8, CASR));
  p.add_inst(SMC_LI(BANK, BAR));
  p.add_inst(SMC_LI(128, NUM_COLS_REG));
  p.add_below(PRE(BAR, 0, 0));
  p.add_inst(SMC_SLEEP(6));
  p.add_below(doubleACT(t12, t23, src, dst));
  p.add_inst(SMC_SLEEP(4));
  p.add_inst(SMC_LI(0, CAR));
  p.add_inst(SMC_LI(marker, PATTERN_REG));
  for (int s = 0; s < 16; s++) p.add_inst(SMC_LDWD(PATTERN_REG, s));
  for (int k = 0; k < ncols; k++) {
    p.add_below(WRITE(BAR, CAR, 1));
    p.add_inst(SMC_SLEEP(8));
  }
  p.add_inst(SMC_SLEEP(8));
  p.add_below(PRE(BAR, 0, 0));
  p.add_inst(SMC_SLEEP(6));
  p.add_inst(SMC_END());
  PF->execute(p);
}
static void read_rows(uint32_t base, uint32_t n, vector<vector<uint8_t>>& out) {
  out.assign(n, vector<uint8_t>(8192));
  uint32_t done = 0;
  while (done < n) {
    uint32_t chunk = (n - done > 16) ? 16 : (n - done);
    Program p;
    p.add_inst(SMC_LI(8, CASR));
    p.add_inst(SMC_LI(BANK, BAR));
    p.add_inst(SMC_LI(128, NUM_COLS_REG));
    p.add_below(PRE(BAR, 0, 0));
    for (uint32_t r = 0; r < chunk; r++)
      p.add_below(rdRow_immediate_label(BAR, base + done + r, r));
    p.add_inst(SMC_END());
    PF->execute(p);
    for (uint32_t r = 0; r < chunk; r++)
      if (PF->receiveData(out[done + r].data(), 8192) != 8192) {
        fprintf(stderr, "recv fail row %u\n", base + done + r); _exit(5);
      }
    done += chunk;
  }
}
// one-bit fraction over the first ncols*4 words (written region)
static double one_frac(const vector<uint8_t>& row, int ncols) {
  long ones = 0, bits = 0;
  int nb = ncols * 16 * 4;   // ncols bursts × 16 words × 4 bytes... capped
  if (nb > 8192) nb = 8192;
  for (int b = 0; b < nb; b++) { ones += __builtin_popcount(row[b]); bits += 8; }
  return (double)ones / bits;
}

int main(int argc, char** argv) {
  if (argc < 5) {
    fprintf(stderr,"Usage: %s <bender> <bank> <subA_base> <subB_base> [sublen=640] [ncols=128]\n",argv[0]);
    return 1; }
  int bender = atoi(argv[1]); BANK = atoi(argv[2]);
  uint32_t A = (uint32_t)strtoul(argv[3],0,10), B = (uint32_t)strtoul(argv[4],0,10);
  uint32_t SL = (argc>5)?(uint32_t)strtoul(argv[5],0,10):640u;
  int NC = (argc>6)?atoi(argv[6]):128;

  SoftMCPlatform pf(bender);
  if (pf.init()!=SOFTMC_SUCCESS){ fprintf(stderr,"init fail\n"); return 3; }
  pf.reset_fpga();
  PF = &pf;
  fprintf(stderr,"[fcd] bender=%d bank=%d subA=[%u,%u) subB=[%u,%u) ncols=%d\n",
          bender,BANK,A,A+SL,B,B+SL,NC);

  // row positions: far/mid/close relative to the A|B boundary (end of A,
  // start of B). If A > B the "boundary" is start of A / end of B.
  bool AbelowB = A < B;
  uint32_t a_close = AbelowB ? A + SL - 4 : A + 4;
  uint32_t a_mid   = A + SL/2;
  uint32_t a_far   = AbelowB ? A + 16 : A + SL - 16;
  uint32_t b_close = AbelowB ? B + 4 : B + SL - 4;
  uint32_t b_mid   = B + SL/2;
  uint32_t b_far   = AbelowB ? B + SL - 16 : B + 16;
  struct PairT { uint32_t src, dst; const char* name; };
  vector<PairT> pairs = {
    {a_close, b_close, "Ac->Bc"}, {a_mid, b_mid, "Am->Bm"},
    {a_far,  b_far,  "Af->Bf"},  {a_close, b_mid, "Ac->Bm"},
    {a_mid,  b_close,"Am->Bc"},  {b_close, a_close, "Bc->Ac"},
    {b_mid,  a_mid,  "Bm->Am"},  {b_close, a_mid, "Bc->Am"},
  };
  struct TT { int t12, t23; };
  vector<TT> times = {{0,0},{0,1},{0,2},{1,1},{2,1},{4,2},{8,2},{30,1}};

  printf("pair,t12,t23,round,n_full,n_half,half_rows,full_rows\n");
  // baseline: init + scan with NO op — rows that fail init (flaky cells,
  // esp. on DIMM 1) get reported under pair=BASE and masked in analysis
  for (int round = 0; round < 2; round++) {
    uint32_t bg = round ? 0xFFFFFFFFu : 0x00000000u;
    init_rows(A, SL, bg); init_rows(B, SL, bg);
    vector<vector<uint8_t>> ra, rb;
    read_rows(A, SL, ra); read_rows(B, SL, rb);
    int n_bad = 0; string desc;
    auto scan0 = [&](vector<vector<uint8_t>>& rr, uint32_t base) {
      for (uint32_t r = 0; r < SL; r++) {
        double f = one_frac(rr[r], NC);
        double dev = round ? 1.0 - f : f;    // deviation from background
        if (dev > 0.05) { n_bad++;
          char buf[64]; snprintf(buf,sizeof buf," %u:%.2f",base+r,dev);
          if (desc.size() < 3000) desc += buf; }
      }
    };
    scan0(ra, A); scan0(rb, B);
    printf("BASE,-,-,%d,0,%d,%s\n", round, n_bad, desc.c_str());
    fprintf(stderr,"[fcd] BASELINE r%d: %d rows deviate from background\n", round, n_bad);
  }
  for (auto& pr : pairs) {
    for (auto& tt : times) {
      // two complementary rounds; a true NOT bitline mask inverts in both
      for (int round = 0; round < 2; round++) {
        uint32_t bg = round ? 0xFFFFFFFFu : 0x00000000u;
        uint32_t mk = round ? 0x00000000u : 0xFFFFFFFFu;
        init_rows(A, SL, bg);
        init_rows(B, SL, bg);
        op_and_write(pr.src, pr.dst, tt.t12, tt.t23, mk, NC);
        vector<vector<uint8_t>> ra, rb;
        read_rows(A, SL, ra); read_rows(B, SL, rb);
        int n_full = 0, n_half = 0;
        string halfdesc, fulldesc;
        auto scan = [&](vector<vector<uint8_t>>& rr, uint32_t base) {
          for (uint32_t r = 0; r < SL; r++) {
            double f = one_frac(rr[r], NC);
            double fm = round ? 1.0 - f : f;    // fraction toward marker
            if (fm > 0.9) { n_full++;
              if (n_full <= 40) { char buf[32]; snprintf(buf,sizeof buf," %u",base+r); fulldesc += buf; }
            } else if (fm > 0.2 && fm < 0.8) {
              n_half++;
              char buf[64]; snprintf(buf,sizeof buf," %u:%.2f",base+r,fm);
              halfdesc += buf;
            }
          }
        };
        scan(ra, A); scan(rb, B);
        printf("%s,%d,%d,%d,%d,%d,%s,%s\n", pr.name, tt.t12, tt.t23, round,
               n_full, n_half, halfdesc.c_str(), fulldesc.c_str());
        if (n_half || n_full > 2)
          fprintf(stderr,"[fcd] %s t=%d,%d r%d: full=%d half=%d%s\n",
                  pr.name, tt.t12, tt.t23, round, n_full, n_half, halfdesc.c_str());
        fflush(stdout);
      }
    }
  }
  fprintf(stderr,"[fcd] DONE\n");
  _exit(0);
}
