// build8 ACCUM_XBP silicon validation: in-fabric cross-bit-plane
// accumulator. The build8 image (trailer 0xDBC0DE06) adds a 4th readback
// mode: reads are consumed into a 128×(16 int32-lane) accumulator as
// acc[seg] += weight · popcount(row segment), with the weight
// (sign + power-of-two shift) latched out-of-band per read program via
// set_acc_weight(); flush_acc() drains the 2048 int32s (8192 B) and zeroes.
//
// Test: pick P "planes", each a (weight, known 2048-u32 row). For each
// plane, write its row, set_acc_weight(neg,shift), read the full row
// (ddr_wdata=0 so the segment popcounts feed the accumulator). After all
// planes, flush_acc + receiveData(8192); assert acc[s] == Σ_p w_p·pc(row_p[s])
// for all 2048 segments. Then a 2nd accumulate+flush to confirm the
// accumulator zeroed. Argv: ./accxbp-hw-exe <bender> <bank> <row>
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

static void write_row(SoftMCPlatform& pf, int bank, uint32_t row,
                      const uint32_t* seg) {
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

// one accumulate read: ddr_wdata := 0 (the compare ref), then rdRow.
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
  if (pf.init() != SOFTMC_SUCCESS) { fprintf(stderr, "[accxbp] init failed\n"); return 1; }
  pf.reset_fpga();
  pf.set_aref(false);

  // 4 planes: weights +1, +2, +4, -8 (shifts 0/1/2/3, neg on last), each a
  // distinct known row (per-segment popcount ramp offset per plane).
  const int W[4][2] = {{0,0},{0,1},{0,2},{1,3}};
  vector<vector<uint32_t>> rows(4, vector<uint32_t>(2048));
  for (int p = 0; p < 4; p++)
    for (int s = 0; s < 2048; s++) {
      int nb = (s + p * 7) % 33;
      rows[p][s] = (nb >= 32) ? 0xFFFFFFFFu : ((1u << nb) - 1u);
    }
  auto expected = [&](int s) -> long {
    long e = 0;
    for (int p = 0; p < 4; p++) {
      long w = (W[p][0] ? -1L : 1L) * (1L << W[p][1]);
      e += w * popcount32(rows[p][s]);
    }
    return e;
  };

  int fails = 0, label = 2000;
  for (int pass = 0; pass < 2; pass++) {
    pf.set_readback_mode_accxbp();
    pf.set_readback_mode_accxbp();     // idempotent; also runs the 128-cyc clear
    for (int p = 0; p < 4; p++) {
      // NB: do NOT re-assert accxbp mode here — set_mode_accxbp re-triggers
      // the 128-cycle accumulator clear, which would wipe prior planes.
      // Mode is set once per pass; writes are execs (no c2h, no accum).
      write_row(pf, bank, row, rows[p].data());
      pf.set_acc_weight(W[p][0], W[p][1]);
      accum_read(pf, bank, row, label++);
    }
    pf.flush_acc();
    vector<uint8_t> buf(8192, 0);
    int got = pf.receiveData(buf.data(), 8192);
    int bad = 0, first = -1;
    for (int s = 0; s < 2048; s++) {
      int32_t v; memcpy(&v, buf.data() + s * 4, 4);
      if ((long)v != expected(s)) { bad++; if (first < 0) first = s; }
    }
    printf("[accxbp] pass %d: recv=%d  %d/2048 segments wrong%s\n",
           pass, got, bad, bad ? "" : "  (EXACT)");
    if (bad) { fails++;
      int32_t v; memcpy(&v, buf.data() + first * 4, 4);
      printf("         first bad seg %d: got %d expect %ld\n", first, v, expected(first)); }
    pf.set_readback_mode(false); pf.set_readback_mode(false);
    pf.drain_stray(1500, 8);
  }
  printf("[accxbp] %s (%d fails)\n", fails ? "FAIL" : "ALL_PASS", fails);
  return fails ? 1 : 0;
}
