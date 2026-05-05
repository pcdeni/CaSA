// End-to-end test for the readback_engine popcount path:
// 512-bit input → 128 × pop_count4 → tree adder → 16-bit popcount →
// popcount_accum → 32-bit sum-on-drain.
//
// Drives N synthetic 512-bit "diff" patterns and verifies the accumulator
// returns the software-computed total bit count. This is the exact path
// used by readback_engine when POPCOUNT_ACCUM_MODE is defined.

#include "Vpopcount_tree_top.h"
#include "verilated.h"
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>

static int popcount512(const uint8_t* data) {
  int c = 0;
  for (int i = 0; i < 64; i++) c += __builtin_popcount(data[i]);
  return c;
}

struct Tick {
  Vpopcount_tree_top* dut;
  vluint64_t t = 0;
  void cycle() {
    dut->clk = 0; dut->eval(); t++;
    dut->clk = 1; dut->eval(); t++;
  }
  void reset() {
    dut->rst = 1;
    dut->in_valid = 0;
    dut->flush    = 0;
    for (int i = 0; i < 16; i++) dut->in_data[i] = 0;
    cycle(); cycle();
    dut->rst = 0;
  }
};

// Verilator packs a 512-bit input into in_data[0..15] (16 × 32-bit words).
static void load_in(Vpopcount_tree_top* dut, const uint8_t* bytes64) {
  for (int w = 0; w < 16; w++) {
    uint32_t x = 0;
    for (int b = 0; b < 4; b++) x |= ((uint32_t)bytes64[w * 4 + b]) << (b * 8);
    dut->in_data[w] = x;
  }
}

static int run_n_reads(Vpopcount_tree_top* dut, int n_reads, unsigned seed) {
  Tick T{dut};
  T.reset();
  srand(seed);
  uint64_t sw_sum = 0;
  for (int r = 0; r < n_reads; r++) {
    uint8_t buf[64];
    for (int b = 0; b < 64; b++) buf[b] = (uint8_t)(rand() & 0xFF);
    sw_sum += popcount512(buf);
    load_in(dut, buf);
    dut->in_valid = 1;
    dut->flush    = 0;
    T.cycle();
  }
  // Tree has 1-cycle latency, accum has 1-cycle latency — let it drain.
  dut->in_valid = 0;
  T.cycle();
  T.cycle();
  // Pulse drain.
  dut->flush = 1;
  T.cycle();
  dut->flush = 0;
  T.cycle();

  uint32_t got = (uint32_t)dut->accum_out;
  bool ok = (got == sw_sum);
  printf("  N=%5d reads, seed=%u: sw_sum=%lu got=%u  %s\n",
         n_reads, seed, (unsigned long)sw_sum, got, ok ? "OK" : "MISMATCH");
  return ok ? 0 : 1;
}

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  auto* dut = new Vpopcount_tree_top;
  int errors = 0;

  printf("=== popcount tree+accum end-to-end ===\n");
  errors += run_n_reads(dut, 1,    1);
  errors += run_n_reads(dut, 10,   2);
  errors += run_n_reads(dut, 100,  3);
  errors += run_n_reads(dut, 1024, 4);
  errors += run_n_reads(dut, 4096, 5);   // BitNet typical matmul shape

  printf("\n=== %d errors total ===\n", errors);
  delete dut;
  return errors;
}
