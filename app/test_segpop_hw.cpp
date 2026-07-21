// build7 SEG_POP silicon validation: per-32b-segment popcount readout.
//
// The build7 image (trailer magic 0xDBC0DE05) adds a third readback mode,
// SET via platform.set_readback_mode_segpop() (control byte[8]=0x80). In
// SEG_POP the readback engine popcounts each 32-bit segment of the read
// row and drains 2048 B/row (2048 x 6-bit popcounts, byte-packed) instead
// of the 8 KB raw row — 4x c2h collapse, keeping the vertical layout, and
// it eliminates the host-side segment_popcount step. ddr_wdata must be 0
// (the compare ref) so read_diff == rd_data and byte[g] == popcount(row
// segment g).
//
// Per case: write a known 2048-u32 pattern into <row>; SET SEG_POP; read
// the row; receiveData(2048); assert byte[g] == popcount(pattern[g]) for
// all 2048 segments. Then a READ-mode toggle-back sanity (8192 B exact) to
// confirm READ is bit-identical to build6.
//
// Argv: ./segpop-hw-exe <bender_id> <bank_id> <row>
// Exit 0 iff all cases byte-exact and READ toggle-back works.
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

// write a full 2048-u32 row by per-column write (the pcwrite idiom).
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

// one SEG_POP read program: LDWD 0 as the compare ref, rdRow the row.
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
  if (pf.init() != SOFTMC_SUCCESS) { fprintf(stderr, "[segpop] init failed\n"); return 1; }
  pf.reset_fpga();
  pf.set_aref(false);

  int fails = 0;
  // three patterns: all-ones, an incrementing per-segment popcount ramp,
  // and a pseudo-random fill.
  vector<vector<uint32_t>> pats(3, vector<uint32_t>(2048));
  for (int s = 0; s < 2048; s++) {
    pats[0][s] = 0xFFFFFFFFu;                              // popcount 32
    int nb = s % 33; pats[1][s] = (nb >= 32) ? 0xFFFFFFFFu : ((1u << nb) - 1u); // ramp 0..32
    uint32_t x = (uint32_t)(s * 2654435761u); pats[2][s] = x ^ (x >> 15);      // random-ish
  }
  const char* names[3] = {"all-ones(pc=32)", "popcount-ramp", "random"};

  int label = 1000;
  for (int c = 0; c < 3; c++) {
    write_row_pattern(pf, bank, row, pats[c].data(), label); label += 200;
    pf.set_readback_mode_segpop();
    pf.set_readback_mode_segpop();   // idempotent, repairs a lost word
    segpop_read_program(pf, bank, row, label); label += 200;
    vector<uint8_t> buf(2048, 0xFF);
    int got = pf.receiveData(buf.data(), 2048);
    int bad = 0, maxseg = -1;
    for (int s = 0; s < 2048; s++) {
      int expect = popcount32(pats[c][s]);
      if ((int)buf[s] != expect) { bad++; if (maxseg < 0) maxseg = s; }
    }
    printf("[segpop] case %-16s recv=%d  %d/2048 segment-bytes wrong%s\n",
           names[c], got, bad, bad ? "" : "  (EXACT)");
    if (bad) { fails++;
      printf("         first bad segment %d: got %u expect %d\n",
             maxseg, buf[maxseg], popcount32(pats[c][maxseg])); }
    // back to READ for the next write (READ-mode writes are unaffected).
    pf.set_readback_mode(false);
    pf.set_readback_mode(false);
    pf.drain_stray(1500, 8);
  }

  // READ-mode toggle-back sanity: write a marker, read the full 8192 B row.
  write_row_pattern(pf, bank, row, vector<uint32_t>(2048, 0x3C3C3C3Cu).data(), label); label += 200;
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
  printf("[segpop] READ_MODE toggle-back: %d/2048 words wrong (expect 0)\n", rbad);
  if (rbad) fails++;

  printf("[segpop] %s (%d fails)\n", fails ? "FAIL" : "ALL_PASS", fails);
  return fails ? 1 : 0;
}
