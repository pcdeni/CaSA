// Verilator C++ harness for pop_count4.v
//
// Walks all 16 input patterns and compares the DUT's output against the
// reference popcount. Prints OK/MISMATCH per input. Exits non-zero on
// any mismatch.

#include "Vpop_count4.h"
#include "verilated.h"
#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  auto* dut = new Vpop_count4;

  int errors = 0;
  for (int i = 0; i < 16; i++) {
    dut->in = (uint8_t)i;
    dut->eval();
    int got = dut->out;
    int expected = __builtin_popcount(i);
    const char* tag = (got == expected) ? "OK     " : "MISMATCH";
    printf("%s in=%2d (0b%c%c%c%c)  exp=%d  got=%d\n",
           tag, i,
           (i & 8) ? '1' : '0',
           (i & 4) ? '1' : '0',
           (i & 2) ? '1' : '0',
           (i & 1) ? '1' : '0',
           expected, got);
    if (got != expected) errors++;
  }

  printf("\n=== %d/%d inputs mismatched ===\n", errors, 16);
  delete dut;
  return errors ? 1 : 0;
}
