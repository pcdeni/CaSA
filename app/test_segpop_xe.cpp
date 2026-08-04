// xe_oracle 2026-08-04 (task #77 silicon half): 0xe-dense SEG_POP oracle.
//
// THIN VARIANT of test_segpop_hw.cpp — identical write/read/drain machinery;
// only the pattern set and the reporting differ. The stock segpop-hw-exe uses
// all-ones/ramp/random patterns (no KNOWN-magnitude 0xe case); this variant
// installs three UNIFORM patterns so every one of the 2048 segments is a
// controlled 0xe test with a stated deficit magnitude:
//   (a) 0xEEEEEEEE  0xe-dense : 8 x 0xe/seg. pop=24 fixed. buggy(RTL 4'b1110
//       missing -> 0xe->0) = 0 ; task-premise(0xe->2) = 16.
//   (b) 0x77777777  0xe-FREE control, equal total pop=24 (8 x 0x7). = 24 on
//       fixed AND buggy -> isolates the 0xe effect from any global miscount.
//   (c) 0xE7E7E7E7  MIXED : 4 x 0xe + 4 x 0x7. pop=24 fixed. buggy(RTL)=12 ;
//       premise=20. Deficit tracks the 0xe count -> pattern-proportional.
//
// Verdict per pattern: byte[g] must == popcount(seg g) on a FIXED image. A
// pop_count4 bug shows a deficit PROPORTIONAL to the 0xe-nibble count (same
// on every segment, since patterns are uniform). Defect-B (the byte-lane
// latch) would instead be POSITION-periodic (~1/8 of bytes, lanes[16k+4]==
// [16k+5]) and pattern-INDEPENDENT -> the 0xe-free control (b) would ALSO be
// hit. So (b) EXACT + (a)/(c) deficit == 0xe-count => pop_count4 bug, not B.
//
// Argv: ./segpop-xe-exe <bender> <bank> <row>   (default row scratch, >prod)
// Exit 0 iff all three patterns byte-exact (FIXED image) and READ toggle-back.
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
static int xe_nibbles(uint32_t v) { int c=0; for (int i=0;i<8;i++) if (((v>>(4*i))&0xF)==0xE) c++; return c; }

// write a full 2048-u32 row by per-column write (the pcwrite idiom) — verbatim
// from test_segpop_hw.cpp.
static void write_row_pattern(SoftMCPlatform& pf, int bank, uint32_t row,
                              const uint32_t* seg, int label) {
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

// one SEG_POP read program — verbatim from test_segpop_hw.cpp.
static void segpop_read_program(SoftMCPlatform& pf, int bank, uint32_t row, int label) {
  Program p;
  p.add_inst(SMC_LI(bank, BAR)); p.add_inst(SMC_LI(8, CASR)); p.add_inst(SMC_LI(128, NUM_COLS_REG));
  p.add_inst(SMC_LI(0, PATTERN_REG));
  for (int i = 0; i < 16; i++) p.add_inst(SMC_LDWD(PATTERN_REG, i));  // ddr_wdata := 0
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
  if (pf.init() != SOFTMC_SUCCESS) { fprintf(stderr, "[xe] init failed\n"); return 1; }
  pf.reset_fpga();
  pf.set_aref(false);

  struct Pat { uint32_t word; const char* name; };
  Pat pats[3] = {
    {0xEEEEEEEEu, "0xe-dense(0xEEEEEEEE)"},
    {0x77777777u, "0xe-free-ctrl(0x77777777)"},
    {0xE7E7E7E7u, "mixed(0xE7E7E7E7)"},
  };

  int fails = 0, label = 1000;
  for (int c = 0; c < 3; c++) {
    uint32_t w = pats[c].word;
    int expect = popcount32(w);       // FIXED per-segment popcount
    int xe = xe_nibbles(w);           // 0xe nibbles per segment
    vector<uint32_t> row_words(2048, w);
    write_row_pattern(pf, bank, row, row_words.data(), label); label += 200;
    pf.set_readback_mode_segpop();
    pf.set_readback_mode_segpop();
    segpop_read_program(pf, bank, row, label); label += 200;
    vector<uint8_t> buf(2048, 0xFF);
    int got = pf.receiveData(buf.data(), 2048);
    // classify: uniform pattern => every segment should read `expect` on FIXED.
    int bad = 0, first_bad = -1;
    // histogram of measured byte values + parity of position for the odd ones
    int minv = 255, maxv = 0;
    long sum = 0;
    for (int s = 0; s < 2048; s++) {
      int v = buf[s]; sum += v;
      if (v < minv) minv = v; if (v > maxv) maxv = v;
      if (v != expect) { bad++; if (first_bad < 0) first_bad = s; }
    }
    double mean = (double)sum / 2048.0;
    printf("[xe] pat %-28s recv=%d  0xe/seg=%d  fixed_expect=%d  measured[min=%d max=%d mean=%.3f]  %d/2048 wrong%s\n",
           pats[c].name, got, xe, expect, minv, maxv, mean, bad, bad ? "" : "  (EXACT->FIXED)");
    if (bad) {
      fails++;
      int meas = buf[first_bad];
      int deficit = expect - meas;
      // is the deficit uniform (pop_count4 signature) or position-periodic (defect B)?
      bool uniform = (minv == maxv);
      printf("     DEFICIT: measured=%d expect=%d deficit=%d ; 0xe/seg=%d ; %s\n",
             meas, expect, deficit, xe,
             uniform ? "UNIFORM across all segments (pattern-proportional)" :
                       "NON-uniform (check for position-periodic defect-B ÷8 signature)");
      if (uniform && xe > 0) {
        int per_nib = (deficit % xe == 0) ? deficit / xe : -1;
        printf("     -> pop_count4 undercount per 0xe nibble = %d  (RTL-missing-case=3, task-premise=1)\n", per_nib);
      }
    }
    pf.set_readback_mode(false);
    pf.set_readback_mode(false);
    pf.drain_stray(1500, 8);
  }

  // READ-mode toggle-back sanity (bit-identical READ path) — verbatim.
  vector<uint32_t> marker(2048, 0x3C3C3C3Cu);
  write_row_pattern(pf, bank, row, marker.data(), label); label += 200;
  {
    Program p;
    p.add_inst(SMC_LI(bank, BAR)); p.add_inst(SMC_LI(8, CASR)); p.add_inst(SMC_LI(128, NUM_COLS_REG));
    p.add_inst(SMC_LI(0, PATTERN_REG));
    p.add_below(rdRow_immediate_label(BAR, row, label)); p.add_inst(SMC_END());
    pf.execute(p);
  }
  vector<uint8_t> rb(8192, 0);
  pf.receiveData(rb.data(), 8192);
  int rbad = 0;
  for (int i = 0; i < 8192; i += 4) { uint32_t v; memcpy(&v, rb.data()+i, 4); if (v != 0x3C3C3C3Cu) rbad++; }
  printf("[xe] READ_MODE toggle-back: %d/2048 words wrong (expect 0)\n", rbad);
  if (rbad) fails++;

  printf("[xe] %s (%d fails)\n", fails ? "FAIL" : "ALL_PASS", fails);
  return fails ? 1 : 0;
}
