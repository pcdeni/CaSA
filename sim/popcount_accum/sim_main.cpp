// Verilator unit test for popcount_accum.v.
//
// Drives synthetic streams through the DUT, pulses drain at known points,
// and checks the output equals the software-computed sum. Exits non-zero
// on any mismatch.

#include "Vpopcount_accum.h"
#include "verilated.h"
#include <cstdio>
#include <cstdlib>
#include <vector>

struct Tick {
  Vpopcount_accum* dut;
  vluint64_t       t = 0;
  void cycle() {
    dut->clk = 0; dut->eval(); t++;
    dut->clk = 1; dut->eval(); t++;
  }
  void reset() {
    dut->rst = 1; dut->in_value = 0; dut->in_valid = 0; dut->drain = 0;
    cycle(); cycle();
    dut->rst = 0;
  }
};

// Drive `values.size()` cycles, asserting in_valid each time. After the
// final value, pulse drain on the next cycle and read accum_out the cycle
// after that.
static int test_drain_after(Vpopcount_accum* dut, const std::vector<uint16_t>& values) {
  Tick T{dut};
  T.reset();
  uint64_t sw_sum = 0;
  for (auto v : values) {
    dut->in_value = v;
    dut->in_valid = 1;
    dut->drain    = 0;
    T.cycle();
    sw_sum += v;
  }
  // Pulse drain (with no in_valid this cycle).
  dut->in_value = 0;
  dut->in_valid = 0;
  dut->drain    = 1;
  T.cycle();
  // Read output one cycle later.
  dut->drain    = 0;
  T.cycle();
  uint32_t got = (uint32_t)dut->accum_out;
  bool ok = (got == sw_sum);
  printf("  drain_after %3zu vals: sum=%u got=%u valid=%d  %s\n",
         values.size(), (unsigned)sw_sum, got, dut->accum_valid,
         ok ? "OK" : "MISMATCH");
  return ok ? 0 : 1;
}

// Drive values, pulse drain on the SAME cycle as the last value (boundary
// condition — the final value should be included in the drained sum).
static int test_drain_coincident(Vpopcount_accum* dut, const std::vector<uint16_t>& values) {
  Tick T{dut};
  T.reset();
  uint64_t sw_sum = 0;
  for (size_t i = 0; i < values.size(); i++) {
    dut->in_value = values[i];
    dut->in_valid = 1;
    dut->drain    = (i + 1 == values.size()) ? 1 : 0;
    T.cycle();
    sw_sum += values[i];
  }
  dut->drain = 0; dut->in_valid = 0;
  T.cycle();
  uint32_t got = (uint32_t)dut->accum_out;
  bool ok = (got == sw_sum);
  printf("  drain_coincident %3zu vals: sum=%u got=%u  %s\n",
         values.size(), (unsigned)sw_sum, got, ok ? "OK" : "MISMATCH");
  return ok ? 0 : 1;
}

// Multiple back-to-back drains — verify each drain returns its own sum and
// the internal counter resets between them.
static int test_repeated_drains(Vpopcount_accum* dut) {
  Tick T{dut};
  T.reset();
  int errors = 0;
  for (int round = 0; round < 4; round++) {
    uint64_t sw_sum = 0;
    int N = 50 + round * 25;
    for (int i = 0; i < N; i++) {
      dut->in_value = (uint16_t)((round * 17 + i * 3) & 0xFFF);
      dut->in_valid = 1;
      dut->drain    = 0;
      T.cycle();
      sw_sum += dut->in_value;
    }
    dut->in_valid = 0; dut->drain = 1; T.cycle();
    dut->drain = 0;                    T.cycle();
    uint32_t got = (uint32_t)dut->accum_out;
    bool ok = (got == sw_sum);
    printf("  round %d (N=%d): sum=%u got=%u  %s\n",
           round, N, (unsigned)sw_sum, got, ok ? "OK" : "MISMATCH");
    if (!ok) errors++;
  }
  return errors;
}

// Stream with random valid drops — values only count when in_valid=1.
static int test_with_invalid_gaps(Vpopcount_accum* dut) {
  Tick T{dut};
  T.reset();
  uint64_t sw_sum = 0;
  srand(42);
  for (int i = 0; i < 200; i++) {
    bool valid = (rand() & 3) != 0;       // ~75% valid
    uint16_t v = (uint16_t)((i * 13) & 0x7F);
    dut->in_value = v;
    dut->in_valid = valid;
    dut->drain    = 0;
    T.cycle();
    if (valid) sw_sum += v;
  }
  dut->in_valid = 0; dut->drain = 1; T.cycle();
  dut->drain = 0;                    T.cycle();
  uint32_t got = (uint32_t)dut->accum_out;
  bool ok = (got == sw_sum);
  printf("  with_invalid_gaps: sum=%u got=%u  %s\n",
         (unsigned)sw_sum, got, ok ? "OK" : "MISMATCH");
  return ok ? 0 : 1;
}

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  auto* dut = new Vpopcount_accum;
  int errors = 0;

  printf("=== popcount_accum unit tests ===\n");

  errors += test_drain_after(dut, {1, 2, 3, 4, 5, 6, 7, 8});
  errors += test_drain_after(dut, {0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF});  // saturation check
  errors += test_drain_coincident(dut, {10, 20, 30});
  errors += test_drain_coincident(dut, {1, 2, 3, 4, 5});
  errors += test_repeated_drains(dut);
  errors += test_with_invalid_gaps(dut);

  // BitNet representative load: 4096 reads × ~64 popcount per read avg.
  std::vector<uint16_t> bnet(4096);
  for (size_t i = 0; i < bnet.size(); i++) bnet[i] = (uint16_t)(64 + (i % 33));
  errors += test_drain_after(dut, bnet);

  printf("\n=== %d errors total ===\n", errors);
  delete dut;
  return errors;
}
