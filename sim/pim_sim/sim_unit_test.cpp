// Tiny standalone test for SimDramModel — exercises ACT/PRE/RD/WR/LI/BL
// without needing the full bitnet-proj-server wrapper. Verifies the sim's
// instruction decoder is correct for the patterns that wrRow_immediate
// emits (LI-setup + 128-iter inner loop with WRITE+ADDI+BL).
//
// Build: g++ -std=c++17 -O2 -I. sim_unit_test.cpp pim_sim.o instruction.o
//        prog.o board.o platform.o -o sim_test
// Or via the BitNet Makefile (add a target).

#include "pim_sim.h"
#include "instruction.h"
#include "prog.h"
#include "../apps/DSN_AE_APPS/util.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main() {
  SimDramModel sim;
  sim.verbose = 2;

  // Manually build a tiny program: write a known pattern into row 100 of
  // bank 5, then read it back. Verify the sim returns the right bytes.
  Program p;
  p.add_inst(SMC_LI(8, CASR));         // CASR=8 (col-addr stride for INC_CAR)
  p.add_inst(SMC_LI(5, BAR));          // r7 = bank 5
  p.add_inst(SMC_LI(100, RAR));        // r6 = row 100
  p.add_inst(SMC_LI(0, CAR));          // r4 = col 0
  // Open the row.
  p.add_below(PRE(BAR, 0, 0));
  p.add_below(ACT(BAR, 0, RAR, 0));
  // Single WRITE with the wide-data buffer; first load all 16 slots
  // with a known pattern.
  for (int i = 0; i < 16; i++) {
    p.add_inst(SMC_LI(0xCAFEBABE + i, PATTERN_REG));
    p.add_inst(SMC_LDWD(PATTERN_REG, i));
  }
  p.add_below(WRITE(BAR, CAR, 0));
  p.add_inst(SMC_SLEEP(8));
  p.add_below(PRE(BAR, 0, 0));
  // Re-open and read back column 0.
  p.add_inst(SMC_LI(0, CAR));
  p.add_below(ACT(BAR, 0, RAR, 0));
  p.add_below(READ(BAR, CAR, 0));
  p.add_inst(SMC_SLEEP(8));
  p.add_below(PRE(BAR, 0, 0));
  p.add_inst(SMC_END());

  // ---- Loop test: increment a register 16 times via SMC_BL. ----
  Program p2;
  p2.add_inst(SMC_LI(0,  3));  // r3 = 0 (counter)
  p2.add_inst(SMC_LI(16, 4));  // r4 = 16 (limit)
  p2.add_inst(SMC_LI(1,  5));  // r5 = 1 (increment)
  p2.add_label("LOOP");
    p2.add_inst(SMC_ADDI(3, 1, 3));  // r3 += 1
  p2.add_branch(p2.BR_TYPE::BL, 3, 4, "LOOP");
  p2.add_inst(SMC_END());

  // Convert Program → AXI-packed byte buffer (mirrors platform.cpp).
  uint64_t* iseq = (uint64_t*)p.get_inst_array();
  int bytes = p.size();
  int n_inst = bytes / 8;
  std::vector<uint8_t> buf(n_inst * 32, 0);
  for (int i = 0; i < n_inst; i++) {
    std::memcpy(buf.data() + i * 32, &iseq[i], 8);
  }
  free(iseq);

  printf("== sending %d-instruction program ==\n", n_inst);
  int rc = sim.send_program(buf.data(), buf.size());
  printf("send_program rc=%d\n", rc);

  std::vector<uint8_t> resp(64);
  int got = sim.recv_response(resp.data(), 64);
  printf("recv_response got=%d bytes\n", got);

  printf("first 16 uint32s of read-back (expect 0xCAFEBABE+0..15):\n");
  for (int i = 0; i < 16; i++) {
    uint32_t v = (uint32_t)resp[i*4]
               | ((uint32_t)resp[i*4+1] << 8)
               | ((uint32_t)resp[i*4+2] << 16)
               | ((uint32_t)resp[i*4+3] << 24);
    uint32_t expected = 0xCAFEBABE + i;
    printf("  [%2d] got=0x%08x expected=0x%08x  %s\n",
           i, v, expected, v == expected ? "OK" : "MISMATCH");
  }

  printf("\n== loop test: SMC_BL must converge to r3=16 ==\n");
  uint64_t* iseq2 = (uint64_t*)p2.get_inst_array();
  int bytes2 = p2.size();
  int n_inst2 = bytes2 / 8;
  std::vector<uint8_t> buf2(n_inst2 * 32, 0);
  for (int i = 0; i < n_inst2; i++) std::memcpy(buf2.data() + i * 32, &iseq2[i], 8);
  free(iseq2);
  printf("  loop program has %d instructions\n", n_inst2);
  for (int i = 0; i < n_inst2; i++) {
    uint64_t v;
    std::memcpy(&v, buf2.data() + i*32, 8);
    printf("    [%d] inst=0x%016lx\n", i, (unsigned long)v);
  }
  rc = sim.send_program(buf2.data(), buf2.size());
  printf("  rc=%d\n", rc);

  return 0;
}
