// xe_oracle 2026-08-04 (task #77): 0xe-dense ACCUM_XBP oracle.
//
// THIN VARIANT of test_accxbp_hw.cpp — identical accumulator machinery; only
// the planes/weights and reporting differ. The stock accxbp-hw-exe uses
// (1<<nb)-1 ramp rows which NEVER produce a 0xe nibble (0xe-free), so it
// cannot see the pop_count4 4'b1110 bug. This variant uses two UNIFORM planes,
// both weight +1:
//   plane A = 0xEEEEEEEE  (0xe-dense: 8 x 0xe/seg, pop 24)
//   plane B = 0x77777777  (0xe-FREE control, pop 24)
// -> acc[seg] = 24 + 24 = 48 on a FIXED image. Buggy(RTL 4'b1110 missing ->
//    0xe->0) = 0 + 24 = 24. task-premise(0xe->2) = 16 + 24 = 40. Plane B is
//    the control: it proves the accumulator + weight path add exactly 24, so
//    any deficit vs 48 is attributable to plane A's 0xe nibbles alone.
//
// Argv: ./accxbp-xe-exe <bender> <bank> <row>
// Exit 0 iff both passes byte-exact (acc==48 every seg) => FIXED image.
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

static int popcount32(uint32_t v) { return __builtin_popcount(v); }

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
    pf.execute(p);
    cs += n;
  }
}

static void accum_read(SoftMCPlatform& pf, int bank, uint32_t row, int label) {
  Program p;
  p.add_inst(SMC_LI(bank, BAR)); p.add_inst(SMC_LI(8, CASR)); p.add_inst(SMC_LI(128, NUM_COLS_REG));
  p.add_inst(SMC_LI(0, PATTERN_REG));
  for (int i = 0; i < 16; i++) p.add_inst(SMC_LDWD(PATTERN_REG, i));
  p.add_below(rdRow_immediate_label(BAR, row, label));
  p.add_inst(SMC_END());
  pf.execute(p);
}

int main(int argc, char** argv) {
  if (argc < 4) { fprintf(stderr, "Usage: %s <bender> <bank> <row>\n", argv[0]); return 1; }
  setvbuf(stdout, NULL, _IONBF, 0);
  int bender = atoi(argv[1]), bank = atoi(argv[2]);
  uint32_t row = (uint32_t)atoi(argv[3]);
  SoftMCPlatform pf(bender);
  if (pf.init() != SOFTMC_SUCCESS) { fprintf(stderr, "[accxbp-xe] init failed\n"); return 1; }
  pf.reset_fpga();
  pf.set_aref(false);

  const uint32_t PLANE[2] = {0xEEEEEEEEu, 0x77777777u};  // A=0xe-dense, B=0xe-free control
  const int      WSH[2]   = {0, 0};                       // both weight +1 (neg=0,shift=0)
  int expect = 0;
  for (int p = 0; p < 2; p++) expect += popcount32(PLANE[p]);   // 24 + 24 = 48 fixed

  int fails = 0, label = 2000;
  for (int pass = 0; pass < 2; pass++) {
    pf.set_readback_mode_accxbp();
    pf.set_readback_mode_accxbp();     // idempotent; runs the 128-cyc clear
    for (int p = 0; p < 2; p++) {
      vector<uint32_t> r(2048, PLANE[p]);
      write_row(pf, bank, row, r.data());
      pf.set_acc_weight(0, WSH[p]);    // +1
      accum_read(pf, bank, row, label++);
    }
    pf.flush_acc();
    vector<uint8_t> buf(8192, 0);
    int got = pf.receiveData(buf.data(), 8192);
    int bad = 0, first = -1, minv = 1<<30, maxv = -(1<<30);
    for (int s = 0; s < 2048; s++) {
      int32_t v; memcpy(&v, buf.data() + s * 4, 4);
      if (v < minv) minv = v; if (v > maxv) maxv = v;
      if ((int)v != expect) { bad++; if (first < 0) first = s; }
    }
    printf("[accxbp-xe] pass %d: recv=%d  fixed_expect=%d  measured[min=%d max=%d]  %d/2048 wrong%s\n",
           pass, got, expect, minv, maxv, bad, bad ? "" : "  (EXACT->FIXED)");
    if (bad) {
      fails++;
      int32_t v; memcpy(&v, buf.data() + first * 4, 4);
      int deficit = expect - v;
      bool uniform = (minv == maxv);
      printf("      DEFICIT: measured=%d expect=%d deficit=%d ; plane A has 8 x 0xe ; %s\n",
             v, expect, deficit, uniform ? "UNIFORM (pattern-proportional -> pop_count4)"
                                         : "NON-uniform (check defect-B)");
      if (uniform && deficit % 8 == 0)
        printf("      -> undercount per 0xe nibble = %d (RTL-missing-case=3, task-premise=1)\n", deficit/8);
    }
    pf.set_readback_mode(false); pf.set_readback_mode(false);
    pf.drain_stray(1500, 8);
  }
  printf("[accxbp-xe] %s (%d fails)\n", fails ? "FAIL" : "ALL_PASS", fails);
  return fails ? 1 : 0;
}
