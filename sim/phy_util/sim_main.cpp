// Verilator harness measuring FPGA-side DDR command-bus utilization for the
// SoftMC pipeline. The pipeline is the real RTL (sources/hdl/verilog/) wired
// up by softmc_pipeline_top.v with a behavioral IMEM. We count, per fabric
// cycle, how many of the four DDR command slots issue a real DRAM command
// (ACT|RD|WR|PRE|REF|ZQ — NOP and idle don't count).
//
// Two scenarios are exercised back-to-back:
//   A: per-MAJ3 cadence — a small program is run, ends, host gap is modelled
//      by N idle cycles, then the next program is loaded and run. Repeated.
//   B: outer-loop cadence — the same body is wrapped in an SMC_BL loop and
//      the entire batch runs as ONE program (no host gap between bodies).
//
// Scenario B is what the IMEM_ADDR_WIDTH 11→13 bump enables — the looped
// program would not fit in 2048 instructions for typical BitNet matmul sizes.
//
// Utilization is reported separately for "active phase" (cycles a program is
// running) and "wall" (all cycles including the modelled host gap), so we
// can see both the per-program cadence and the across-program effect.

#include "Vsoftmc_pipeline_top.h"
#include "verilated.h"
#include <cstdio>
#include <cstdint>
#include <cassert>
#include <vector>
#include <string>

// ---------- SoftMC instruction encoding (mirrors api/instruction.h) -----------
// Bits / function codes copied verbatim so this harness stays standalone.

namespace smc {

// Inst (64-bit) layout
constexpr int IS_BR    = 62;
constexpr int IS_DDR   = 63;
constexpr int FU_CODE  = 48;
constexpr int OP_CODE  = 59;
// EXE field positions
constexpr int RS1      = 0;
constexpr int RS2      = 4;
constexpr int RT       = 20;
constexpr int IMD1     = 4;
constexpr int BR_TGT   = 8;
constexpr int J_TGT    = 0;
// EXE function codes
constexpr int FN_ADDI  = 1;
constexpr int FN_LI    = 6;
constexpr int FN_BL    = 0;
constexpr int FN_JUMP  = 2;
// DDR field positions
constexpr int DDR_CMD  = 12;
constexpr int DDR_CAR  = 4;
constexpr int DDR_BAR  = 0;
constexpr int DDR_RAR  = 4;
constexpr int DDR_IBAR = 10;
constexpr int DDR_PALL = 11;
// DDR function codes (the encoding has bit 3 set so the 4th slot's bit 15
// becomes the IS_DDR flag in the packed Inst)
constexpr int FN_WRITE = 8;
constexpr int FN_READ  = 9;
constexpr int FN_PRE   = 10;
constexpr int FN_ACT   = 11;
constexpr int FN_REF   = 13;
constexpr int FN_NOP   = 15;
// MISC opcodes (BR_TYPE)
constexpr int OP_INFO  = 0;

using Inst    = uint64_t;
using Mininst = uint16_t;

inline Mininst NOP() {
  return (Mininst)(FN_NOP << DDR_CMD);
}
inline Mininst ACT(int bar, int rar) {
  return (Mininst)((FN_ACT << DDR_CMD) | (rar << DDR_RAR) | (bar & 0xF));
}
inline Mininst PRE(int bar, int pall) {
  return (Mininst)((FN_PRE << DDR_CMD) | (pall ? (1 << DDR_PALL) : 0) | (bar & 0xF));
}
inline Mininst RD(int bar, int car) {
  return (Mininst)((FN_READ << DDR_CMD) | (car << DDR_CAR) | (bar & 0xF));
}
inline Mininst WR(int bar, int car) {
  return (Mininst)((FN_WRITE << DDR_CMD) | (car << DDR_CAR) | (bar & 0xF));
}

inline Inst pack4(Mininst a, Mininst b, Mininst c, Mininst d) {
  return ((Inst)d << 48) | ((Inst)c << 32) | ((Inst)b << 16) | (Inst)a;
}

inline Inst LI(uint32_t imd, int rt) {
  return ((Inst)FN_LI << FU_CODE) | ((Inst)(imd & 0xFFFFFFu) << IMD1) | ((Inst)rt << RT);
}
inline Inst ADDI(int rs1, uint32_t imd, int rt) {
  return ((Inst)FN_ADDI << FU_CODE) | ((Inst)rs1 << RS1)
       | ((Inst)(imd & 0xFFFF) << IMD1) | ((Inst)rt << RT);
}
inline Inst BL(int rs1, int rs2, int tgt) {
  // BL is an EXE-class branch: bit 62 set, fu_code = FN_BL, branch target at BR_TGT.
  return ((Inst)1 << IS_BR) | ((Inst)FN_BL << FU_CODE)
       | ((Inst)rs1 << RS1) | ((Inst)rs2 << RS2)
       | ((Inst)(tgt & 0x7FFFF) << BR_TGT);
}
// Marks end of program. Encoding: misc (bit 61) + INFO with read-count = 0.
// Pre-decode treats `is_end` as misc + zero rdcnt. To stay simple we use the
// spec form from prog.cpp: SMC_END() is implemented as INFO with rdcnt=0.
inline Inst END() {
  // misc/INFO instruction with rdcnt=0; pre_decode flags this as END.
  // (Bit 61 = IS_MISC; FU_CODE = INFO=0; rdcnt operand at low bits = 0.)
  return ((Inst)1 << 61);
}
// SLEEP — for "host gap" emulation during a single program run.
inline Inst SLEEP(uint32_t cycles) {
  // EXE-class, fu_code = FN_SLEEP=3, amount in low bits.
  return ((Inst)1 << 61) | ((Inst)3 << FU_CODE) | (Inst)cycles;
}

}  // namespace smc

// ---------- Program builder ----------------------------------------------------

struct Program {
  std::vector<smc::Inst> insts;
  void emit(smc::Inst i) { insts.push_back(i); }
};

// A representative MAJ3 body: ACT row R, READ col 0, READ col 1, PRE bank.
// Padded with NOP packs to satisfy minimum tRCD/tRP-style spacing. Numbers
// are illustrative — the point is to demonstrate the measurement, not to
// model exact DDR4 timings.
static void emit_maj3_body(Program& p, int bar, int rar) {
  using namespace smc;
  p.emit(pack4(ACT(bar, rar), NOP(), NOP(), NOP()));
  p.emit(pack4(NOP(), NOP(), NOP(), NOP()));      // tRCD pad
  p.emit(pack4(RD(bar, 0), NOP(), NOP(), NOP()));
  p.emit(pack4(RD(bar, 1), NOP(), NOP(), NOP()));
  p.emit(pack4(NOP(), NOP(), NOP(), NOP()));
  p.emit(pack4(PRE(bar, 0), NOP(), NOP(), NOP()));
  p.emit(pack4(NOP(), NOP(), NOP(), NOP()));      // tRP pad
}

// Multi-bank pipelined body: same ACT-RD-RD-PRE pattern but on four banks
// in parallel — each fabric cycle issues four DDR commands (one per bank,
// one per slot). This is the actual "bus-bound" target shape: every slot
// busy on every command-cycle. Padding rows still respect tRCD / tRP.
static void emit_maj3_body_multibank(Program& p, int rar) {
  using namespace smc;
  p.emit(pack4(ACT(0, rar), ACT(1, rar), ACT(2, rar), ACT(3, rar)));
  p.emit(pack4(NOP(), NOP(), NOP(), NOP()));         // tRCD
  p.emit(pack4(RD(0, 0), RD(1, 0), RD(2, 0), RD(3, 0)));
  p.emit(pack4(RD(0, 1), RD(1, 1), RD(2, 1), RD(3, 1)));
  p.emit(pack4(NOP(), NOP(), NOP(), NOP()));         // tCL pad before PRE
  p.emit(pack4(PRE(0, 0), PRE(1, 0), PRE(2, 0), PRE(3, 0)));
  p.emit(pack4(NOP(), NOP(), NOP(), NOP()));         // tRP
}

// Scenario A: tiny program holding ONE body. Caller invokes this many times
// with a host-gap idle between runs.
static Program build_single_body(int bar, int rar) {
  Program p;
  emit_maj3_body(p, bar, rar);
  p.emit(smc::END());
  return p;
}

// Scenario B: ONE big program where the body runs `count` times via SMC_BL.
// Requires IMEM > 2048 entries when count ⨯ body_len gets large.
static Program build_outerloop(int bar, int rar, int count, bool multibank=false) {
  using namespace smc;
  Program p;
  // r1 = 1 (increment), r2 = count, r3 = 0 (loop ctr)
  p.emit(LI(1, 1));
  p.emit(LI((uint32_t)count, 2));
  p.emit(LI(0, 3));
  int loop_top = (int)p.insts.size();    // PC of body start
  if (multibank) emit_maj3_body_multibank(p, rar);
  else           emit_maj3_body(p, bar, rar);
  p.emit(ADDI(3, 1, 3));                 // r3 += 1
  p.emit(BL(3, 2, loop_top));            // if r3 < r2 goto loop_top
  p.emit(END());
  return p;
}

// Scenario D: bodies inlined with NO loop. Forces large IMEM. Models the
// per-column non-uniform writes the BitNet weight loader emits today
// (each column has different data, so a loop wouldn't help even if we had
// one). With 7 instructions per body × N bodies, N=400 already needs >2048
// (the pre-bump IMEM cap), so this scenario only runs after the 11→13 bump.
static Program build_inline_bodies(int bar, int n_bodies) {
  using namespace smc;
  Program p;
  for (int i = 0; i < n_bodies; i++) {
    emit_maj3_body(p, bar, /*rar=*/i & 0xFFFF);
  }
  p.emit(END());
  return p;
}

// ---------- Verilator harness --------------------------------------------------

struct Stats {
  uint64_t active_slots = 0;
  uint64_t total_slots  = 0;
  uint64_t cycles       = 0;
  void tally(uint8_t act, uint8_t rd, uint8_t wr, uint8_t pre, uint8_t ref, uint8_t zq) {
    uint8_t any = act | rd | wr | pre | ref | zq;
    // popcount of the OR — count slots with at least one command type
    for (int s = 0; s < 4; s++) active_slots += (any >> s) & 1u;
    total_slots += 4;
    cycles      += 1;
  }
  double util() const { return total_slots ? (double)active_slots / total_slots : 0.0; }
};

static void load_imem(Vsoftmc_pipeline_top* dut, vluint64_t& tick,
                      const Program& p) {
  // Hold rst high for a few cycles, drive imem_init_we per word.
  dut->rst = 1;
  dut->run = 0;
  dut->imem_init_we = 0;
  for (int i = 0; i < 4; i++) {
    dut->clk = 0; dut->eval(); tick++;
    dut->clk = 1; dut->eval(); tick++;
  }
  dut->rst = 0;
  for (size_t i = 0; i < p.insts.size(); i++) {
    dut->imem_init_we   = 1;
    dut->imem_init_addr = (uint32_t)i;
    dut->imem_init_data = (uint64_t)p.insts[i];
    dut->clk = 0; dut->eval(); tick++;
    dut->clk = 1; dut->eval(); tick++;
  }
  dut->imem_init_we = 0;
}

// Run the loaded IMEM until softmc_end. Tally utilization in `s`.
// Returns the number of cycles consumed.
static uint64_t run_until_end(Vsoftmc_pipeline_top* dut, vluint64_t& tick,
                              Stats& s, uint64_t cycle_cap = 200000) {
  dut->run = 1;
  uint64_t c = 0;
  for (; c < cycle_cap; c++) {
    dut->clk = 0; dut->eval(); tick++;
    dut->clk = 1; dut->eval(); tick++;
    s.tally(dut->ddr_act, dut->ddr_read, dut->ddr_write,
            dut->ddr_pre, dut->ddr_ref, dut->ddr_zq);
    if (dut->softmc_end) break;
  }
  dut->run = 0;
  return c + 1;
}

// Idle the pipeline for `cycles` cycles (no IMEM activity, run gated low).
// All gap cycles count toward the "wall" stats but not the active stats.
static void idle(Vsoftmc_pipeline_top* dut, vluint64_t& tick,
                 Stats& wall, uint64_t cycles) {
  dut->run = 0;
  for (uint64_t i = 0; i < cycles; i++) {
    dut->clk = 0; dut->eval(); tick++;
    dut->clk = 1; dut->eval(); tick++;
    wall.tally(0, 0, 0, 0, 0, 0);  // pure idle
  }
}

static void scenario_per_maj3(Vsoftmc_pipeline_top* dut, vluint64_t& tick,
                              int n_bodies, uint64_t host_gap_cycles) {
  Stats active, wall;
  uint64_t total_cycles_run = 0;
  for (int i = 0; i < n_bodies; i++) {
    Program p = build_single_body(/*bar=*/1, /*rar=*/i & 0xFFFF);
    load_imem(dut, tick, p);
    Stats s;
    uint64_t c = run_until_end(dut, tick, s);
    total_cycles_run += c;
    active.active_slots += s.active_slots;
    active.total_slots  += s.total_slots;
    active.cycles       += s.cycles;
    wall.active_slots   += s.active_slots;
    wall.total_slots    += s.total_slots;
    wall.cycles         += s.cycles;
    // Model host gap (PCIe round-trip + program build/send) by going idle.
    idle(dut, tick, wall, host_gap_cycles);
  }
  printf("[scenario A: %d × per-MAJ3, host_gap=%lu cyc/program]\n",
         n_bodies, (unsigned long)host_gap_cycles);
  printf("  IMEM size used / program: %zu insts\n", build_single_body(0,0).insts.size());
  printf("  Program-active utilization : %.2f%% (%lu / %lu slots over %lu cyc)\n",
         100.0 * active.util(),
         (unsigned long)active.active_slots, (unsigned long)active.total_slots,
         (unsigned long)active.cycles);
  printf("  Wall utilization (incl gap): %.2f%% (%lu / %lu slots over %lu cyc)\n",
         100.0 * wall.util(),
         (unsigned long)wall.active_slots, (unsigned long)wall.total_slots,
         (unsigned long)wall.cycles);
}

static void scenario_outer_loop(Vsoftmc_pipeline_top* dut, vluint64_t& tick,
                                int n_bodies) {
  Program p = build_outerloop(/*bar=*/1, /*rar=*/0, n_bodies);
  printf("[scenario B: %d-body outer loop in 1 program (IMEM size = %zu insts)]\n",
         n_bodies, p.insts.size());
  load_imem(dut, tick, p);
  Stats s;
  uint64_t c = run_until_end(dut, tick, s);
  printf("  Program-active utilization : %.2f%% (%lu / %lu slots over %lu cyc)\n",
         100.0 * s.util(),
         (unsigned long)s.active_slots, (unsigned long)s.total_slots,
         (unsigned long)s.cycles);
  printf("  Wall utilization (no gap)  : %.2f%% (same — single program)\n",
         100.0 * s.util());
  (void)c;
}

static void scenario_inline(Vsoftmc_pipeline_top* dut, vluint64_t& tick,
                            int n_bodies) {
  Program p = build_inline_bodies(/*bar=*/1, n_bodies);
  printf("[scenario D: %d-body INLINE program (no loop, IMEM = %zu insts)]\n",
         n_bodies, p.insts.size());
  if (p.insts.size() > 2048) {
    printf("  >>> NOTE: %zu insts exceeds the pre-bump IMEM cap (2048).\n"
           "      This program only loads after IMEM_ADDR_WIDTH 11 → 13. <<<\n",
           p.insts.size());
  }
  load_imem(dut, tick, p);
  Stats s;
  uint64_t c = run_until_end(dut, tick, s, /*cycle_cap=*/1000000);
  printf("  Program-active utilization : %.2f%% (%lu / %lu slots over %lu cyc)\n",
         100.0 * s.util(),
         (unsigned long)s.active_slots, (unsigned long)s.total_slots,
         (unsigned long)s.cycles);
  (void)c;
}

static void scenario_outer_loop_multibank(Vsoftmc_pipeline_top* dut,
                                          vluint64_t& tick, int n_bodies) {
  Program p = build_outerloop(/*bar=*/0, /*rar=*/0, n_bodies, /*multibank=*/true);
  printf("[scenario E: %d-body MULTIBANK outer loop (IMEM = %zu insts)]\n",
         n_bodies, p.insts.size());
  load_imem(dut, tick, p);
  Stats s;
  uint64_t c = run_until_end(dut, tick, s);
  printf("  Program-active utilization : %.2f%% (%lu / %lu slots over %lu cyc)\n",
         100.0 * s.util(),
         (unsigned long)s.active_slots, (unsigned long)s.total_slots,
         (unsigned long)s.cycles);
  (void)c;
}

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  auto* dut = new Vsoftmc_pipeline_top;
  vluint64_t tick = 0;

  // Scenario A: 32 single-body programs, each with a 5000-cycle host gap
  // (≈ 30 µs at 167 MHz fabric clock — conservative PCIe round-trip cost
  // including program build, sendData, drain spawn, consumeData wait).
  scenario_per_maj3(dut, tick, /*n_bodies=*/32, /*host_gap_cycles=*/5000);

  printf("\n");

  // Scenario B: same 32 bodies, but as one outer-loop program. No host gap.
  scenario_outer_loop(dut, tick, /*n_bodies=*/32);

  printf("\n");

  // Scenario C: 512-body outer loop. Stress the loop, but the program text
  // itself stays small (it's a loop body). IMEM bump not strictly required.
  scenario_outer_loop(dut, tick, /*n_bodies=*/512);

  printf("\n");

  // Scenario D: 400-body INLINE program — actually requires the IMEM bump
  // (400 × 7 instr ≈ 2800 > 2048). Tests whether the bump took effect.
  scenario_inline(dut, tick, /*n_bodies=*/400);

  printf("\n");

  // Scenario E: 32-body multibank outer loop. Same loop count as B, but
  // each body issues 4 commands per slot — the actual "bus-bound" target.
  scenario_outer_loop_multibank(dut, tick, /*n_bodies=*/32);

  delete dut;
  return 0;
}
