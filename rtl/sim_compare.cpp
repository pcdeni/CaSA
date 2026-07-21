// A/B comparison harness: same READ workload run two ways.
//   BASELINE: pristine DRAM-Bender softmc_pipeline executing a SoftMC outer
//             loop with one pack4(RD bk0..bk3) per body.
//   SEQ:      new seq_engine.v executing one start-pulse with op=READ,
//             count=N, bank_mask=4'b1111.
//
// Both DUTs live in dual_top.v so a single Verilator binary owns them.
// For each implementation the harness records every fabric cycle's command
// output (per-slot ddr_read / ddr_bank / ddr_col), then:
//   (1) compares the multiset of (slot, bank, col) tuples for correctness
//       (baseline command stream MUST be a subset/superset match — the
//       seq engine emits the same logical commands);
//   (2) reports cycles, active slots, and PHY utilization; and
//   (3) reports the speedup ratio.

#include "verilated.h"
#include "Vdual_top.h"
#include <cstdio>
#include <cstdint>
#include <vector>
#include <set>
#include <algorithm>

// ----- minimal SoftMC instruction builder (mirrors api/instruction.h) -----
namespace smc {
constexpr int IS_BR    = 62;
constexpr int FU_CODE  = 48;
constexpr int RS1      = 0;
constexpr int RS2      = 4;
constexpr int RT       = 20;
constexpr int IMD1     = 4;
constexpr int BR_TGT   = 8;
constexpr int FN_ADDI  = 1;
constexpr int FN_LI    = 6;
constexpr int FN_BL    = 0;
constexpr int DDR_CMD  = 12;
constexpr int DDR_CAR  = 4;
constexpr int DDR_BAR  = 0;
constexpr int DDR_RAR  = 4;
constexpr int FN_READ  = 9;
constexpr int FN_NOP   = 15;
constexpr int FN_PRE   = 10;
constexpr int FN_ACT   = 11;
using Inst    = uint64_t;
using Mininst = uint16_t;
inline Mininst NOP() { return (Mininst)(FN_NOP << DDR_CMD); }
inline Mininst RD(int bar, int car) {
  return (Mininst)((FN_READ << DDR_CMD) | (car << DDR_CAR) | (bar & 0xF));
}
inline Mininst PRE(int bar, int pall) {
  return (Mininst)((FN_PRE << DDR_CMD) | (pall ? (1<<11) : 0) | (bar & 0xF));
}
inline Mininst ACT(int bar, int rar) {
  return (Mininst)((FN_ACT << DDR_CMD) | (rar << DDR_RAR) | (bar & 0xF));
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
  return ((Inst)1 << IS_BR) | ((Inst)FN_BL << FU_CODE)
       | ((Inst)rs1 << RS1) | ((Inst)rs2 << RS2)
       | ((Inst)(tgt & 0x7FFFF) << BR_TGT);
}
inline Inst END() { return 0; }   // pre_decode flags is_end on all-zero word
}  // namespace smc

struct Cmd {
  enum Kind { K_RD=0, K_ACT=1, K_PRE=2 };
  int  slot = 0;
  int  bank = 0;
  Kind kind = K_RD;
  int  col  = 0;     // valid only for K_RD
  int  row  = 0;     // valid only for K_ACT
  bool operator<(const Cmd& o) const {
    if (bank != o.bank) return bank < o.bank;
    if (kind != o.kind) return kind < o.kind;
    int my_v  = (kind == K_ACT) ? row : (kind == K_RD ? col : 0);
    int oth_v = (kind == K_ACT) ? o.row : (kind == K_RD ? o.col : 0);
    if (my_v != oth_v) return my_v < oth_v;
    return slot < o.slot;
  }
  bool operator==(const Cmd& o) const {
    if (bank != o.bank || slot != o.slot || kind != o.kind) return false;
    if (kind == K_ACT) return row == o.row;
    if (kind == K_RD)  return col == o.col;
    return true;   // PRE has no payload
  }
};

// Extract a 17-bit slice from a VlWide<N> bus (DDR row field, 4*17=68 bits).
// Verilator packs >64-bit signals into uint32_t arrays; this helper handles
// the cross-word case for the slot*17 offset.
template <typename W>
static uint32_t slice17(const W& w, int slot) {
  int bit = slot * 17;
  int word_idx = bit / 32;
  int bit_off  = bit % 32;
  uint64_t lo = (uint64_t)w.at(word_idx);
  uint64_t hi = (bit_off + 17 > 32) ? (uint64_t)w.at(word_idx + 1) : 0;
  uint64_t val = (lo >> bit_off) | (hi << (32 - bit_off));
  return (uint32_t)(val & 0x1FFFF);
}

struct Stats {
  uint64_t cycles = 0, active_slots = 0, total_slots = 0;
  std::vector<Cmd> cmds;
  void tally(uint8_t any_op_mask) {
    for (int s = 0; s < 4; s++) active_slots += (any_op_mask >> s) & 1u;
    total_slots += 4;
    cycles += 1;
  }
  double util() const { return total_slots ? (double)active_slots / total_slots : 0.0; }
};

static void cycle(Vdual_top* dut, vluint64_t& tick) {
  dut->clk = 0; dut->eval(); tick++;
  dut->clk = 1; dut->eval(); tick++;
}

static std::vector<smc::Inst> build_baseline_program(int n_iter) {
  using namespace smc;
  // The BAR/CAR fields in a DDR mininst are 4-bit register-file indexes,
  // not literal addresses — the actual bank/col is read from the register
  // file at execute. To get a fair baseline that emits 4 reads per cycle
  // on 4 different banks, pre-load reg0..reg3 with bank IDs 0..3 and
  // reg4 with the column value, then reference those reg IDs from each
  // slot's mininst.
  std::vector<Inst> p;
  p.push_back(LI(0, 0));   // r0 = bank 0
  p.push_back(LI(1, 1));   // r1 = bank 1
  p.push_back(LI(2, 2));   // r2 = bank 2
  p.push_back(LI(3, 3));   // r3 = bank 3
  p.push_back(LI(0, 4));   // r4 = column 0
  // Loop counter / limit / increment (use higher reg ids to avoid collisions).
  p.push_back(LI(1,        8));    // r8 = increment
  p.push_back(LI((uint32_t)n_iter, 9)); // r9 = trip count
  p.push_back(LI(0,       10));    // r10 = loop counter
  int loop_top = (int)p.size();
  // pack4: slot i = RD with BAR-reg i (bank i), CAR-reg 4 (col 0)
  p.push_back(pack4(RD(0,4), RD(1,4), RD(2,4), RD(3,4)));
  p.push_back(ADDI(10, 1, 10));      // r10 += 1
  p.push_back(BL(10, 9, loop_top));  // if r10 < r9 goto loop_top
  p.push_back(END());
  return p;
}

// Capture per-cycle commands from the BASELINE side of dual_top.
static Stats run_baseline(Vdual_top* dut, vluint64_t& tick, int n_iter) {
  Stats s;
  // Reset both DUTs and load baseline IMEM.
  dut->rst = 1;
  dut->base_run = 0; dut->base_imem_we = 0;
  dut->seq_start = 0;
  for (int i = 0; i < 4; i++) cycle(dut, tick);
  dut->rst = 0;
  auto p = build_baseline_program(n_iter);
  for (size_t i = 0; i < p.size(); i++) {
    dut->base_imem_we   = 1;
    dut->base_imem_addr = (uint32_t)i;
    dut->base_imem_data = (uint64_t)p[i];
    cycle(dut, tick);
  }
  dut->base_imem_we = 0;
  dut->base_run = 1;
  for (uint64_t c = 0; c < 200000; c++) {
    cycle(dut, tick);
    uint8_t any = dut->base_ddr_act | dut->base_ddr_read | dut->base_ddr_write
                | dut->base_ddr_pre | dut->base_ddr_ref | dut->base_ddr_zq;
    s.tally(any);
    for (int slot = 0; slot < 4; slot++) {
      if ((dut->base_ddr_read >> slot) & 1u) {
        Cmd cmd;
        cmd.slot = slot;
        cmd.bank = (dut->base_ddr_bank >> (slot * 2)) & 0x3;
        cmd.col  = (dut->base_ddr_col  >> (slot * 10)) & 0x3FF;
        s.cmds.push_back(cmd);
      }
    }
    if (dut->base_softmc_end) break;
  }
  dut->base_run = 0;
  return s;
}

static Stats run_seq(Vdual_top* dut, vluint64_t& tick, int n_cycles) {
  Stats s;
  dut->rst = 1;
  dut->base_run = 0;
  dut->seq_start    = 0;
  dut->seq_op_code  = 1;       // READ
  dut->seq_bank_mask = 0xF;
  dut->seq_bank0_id  = 0;
  dut->seq_base_col  = 0;
  dut->seq_base_row  = 0;
  dut->seq_count     = (uint16_t)n_cycles;
  dut->seq_stride    = 0;
  dut->seq_bl4 = 0; dut->seq_ap = 0;
  for (int i = 0; i < 4; i++) cycle(dut, tick);
  dut->rst = 0;
  // Pulse start. After the cycle the engine has latched (busy=1, ctr=0);
  // the combinational outputs now reflect cycle 0 of the burst, so the
  // first tally must happen BEFORE any further clock advance.
  dut->seq_start = 1; cycle(dut, tick); dut->seq_start = 0;
  for (uint64_t c = 0; c < 200000; c++) {
    if (dut->seq_busy) {
      uint8_t any = dut->seq_ddr_act | dut->seq_ddr_read
                  | dut->seq_ddr_write | dut->seq_ddr_pre;
      s.tally(any);
      for (int slot = 0; slot < 4; slot++) {
        if ((dut->seq_ddr_read >> slot) & 1u) {
          Cmd cmd;
          cmd.slot = slot;
          cmd.bank = (dut->seq_ddr_bank >> (slot * 2)) & 0x3;
          cmd.col  = (dut->seq_ddr_col  >> (slot * 10)) & 0x3FF;
          s.cmds.push_back(cmd);
        }
      }
    }
    if (dut->seq_done) break;
    cycle(dut, tick);
  }
  return s;
}

static int compare_cmd_streams(const Stats& a, const Stats& b) {
  std::vector<Cmd> A = a.cmds, B = b.cmds;
  std::sort(A.begin(), A.end());
  std::sort(B.begin(), B.end());
  if (A.size() != B.size()) {
    printf("  size mismatch: baseline=%zu seq=%zu\n", a.cmds.size(), b.cmds.size());
    return 1;
  }
  int diffs = 0;
  for (size_t i = 0; i < A.size(); i++) {
    if (!(A[i] == B[i])) {
      if (diffs < 5)
        printf("  diff @ %zu: A(slot=%d bank=%d col=%d) vs B(slot=%d bank=%d col=%d)\n",
               i, A[i].slot, A[i].bank, A[i].col, B[i].slot, B[i].bank, B[i].col);
      diffs++;
    }
  }
  printf("  %d diffs across %zu cmds\n", diffs, A.size());
  return diffs;
}

// ----- doubleACT BASELINE: SoftMC running the SiMRA doubleACT(t12,t23) -----
// Mirrors api/util.cpp `doubleACT()` exactly: q_inst[0]=ACT R_first,
// q_inst[t12+1]=PRE, q_inst[t12+t23+2]=ACT R_second; pad to multiple of 4
// with NOPs; pack4-emit. Wrapped in LI(BAR=bank), LI(RF_REG=R_first),
// LI(LOOP_COLS=R_second) so the per-bank RF/COLS registers are loaded.
//
// Register IDs from sources/apps/DSN_AE_APPS/util.h:
//   BAR = 7, RF_REG = 10, LOOP_COLS = 13
static std::vector<smc::Inst> build_baseline_doubleACT_program(
    int t12, int t23, int bank, int r_first, int r_second) {
  using namespace smc;
  constexpr int BAR_REG_ID       = 7;
  constexpr int RF_REG_ID        = 10;
  constexpr int LOOP_COLS_REG_ID = 13;
  std::vector<Inst> p;
  // Set up registers (BAR=bank, RF_REG=R_first, LOOP_COLS=R_second).
  p.push_back(LI((uint32_t)bank,     BAR_REG_ID));
  p.push_back(LI((uint32_t)r_first,  RF_REG_ID));
  p.push_back(LI((uint32_t)r_second, LOOP_COLS_REG_ID));
  // Build the q_inst[] pattern.
  int num_cmd = 3 + t12 + t23;
  num_cmd += 4 - (num_cmd % 4);
  std::vector<Mininst> q(num_cmd, NOP());
  q[0]            = ACT(BAR_REG_ID, RF_REG_ID);
  q[t12 + 1]      = PRE(BAR_REG_ID, 0);
  q[t12 + t23 + 2] = ACT(BAR_REG_ID, LOOP_COLS_REG_ID);
  for (int i = 0; i < num_cmd; i += 4)
    p.push_back(pack4(q[i], q[i+1], q[i+2], q[i+3]));
  p.push_back(END());
  return p;
}

// Capture all interesting baseline commands (ACT, PRE, RD) per cycle.
static Stats run_baseline_doubleACT(Vdual_top* dut, vluint64_t& tick,
                                     int t12, int t23, int bank,
                                     int r_first, int r_second) {
  Stats s;
  dut->rst = 1; dut->base_run = 0; dut->base_imem_we = 0;
  dut->seq_start = 0;
  for (int i = 0; i < 4; i++) cycle(dut, tick);
  dut->rst = 0;
  auto p = build_baseline_doubleACT_program(t12, t23, bank, r_first, r_second);
  for (size_t i = 0; i < p.size(); i++) {
    dut->base_imem_we   = 1;
    dut->base_imem_addr = (uint32_t)i;
    dut->base_imem_data = (uint64_t)p[i];
    cycle(dut, tick);
  }
  dut->base_imem_we = 0;
  dut->base_run = 1;
  // Count from run=1 onwards — this is the REAL per-primitive cost
  // including LI-register-load overhead and pipeline fill, the cycles a
  // host program actually pays per `platform.execute(doubleACT(...))`.
  bool end_fired = false;
  uint64_t trailing_idle = 0;
  uint64_t post_end_cycles = 0;
  for (uint64_t c = 0; c < 200000; c++) {
    cycle(dut, tick);
    uint8_t any_useful = dut->base_ddr_act | dut->base_ddr_read
                       | dut->base_ddr_write | dut->base_ddr_pre;
    {
      uint8_t any = any_useful | dut->base_ddr_ref | dut->base_ddr_zq;
      s.tally(any);
      if (any_useful) trailing_idle = 0; else trailing_idle++;
      for (int slot = 0; slot < 4; slot++) {
        if ((dut->base_ddr_act >> slot) & 1u) {
          Cmd cmd; cmd.slot = slot; cmd.kind = Cmd::K_ACT;
          cmd.bank = (dut->base_ddr_bank >> (slot * 2)) & 0x3;
          cmd.row  = slice17(dut->base_ddr_row, slot);
          s.cmds.push_back(cmd);
        }
        if ((dut->base_ddr_pre >> slot) & 1u) {
          Cmd cmd; cmd.slot = slot; cmd.kind = Cmd::K_PRE;
          cmd.bank = (dut->base_ddr_bank >> (slot * 2)) & 0x3;
          s.cmds.push_back(cmd);
        }
      }
    }
    if (dut->base_softmc_end) {
      end_fired = true;
      // Drop ready_out so fetch stops re-fetching from PC=0 (which
      // softmc_end resets it to). Without this the program loops forever.
      dut->base_run = 0;
    }
    // Drain pipeline for 32 more cycles so any in-flight pack4 from the
    // last fetch can emit on ddr_*.
    if (end_fired) {
      post_end_cycles++;
      if (post_end_cycles > 32) break;
    }
  }
  // Strip trailing idle cycles so utilization reflects the burst.
  if (trailing_idle > 0 && s.cycles >= trailing_idle) {
    s.cycles -= trailing_idle;
    s.total_slots -= trailing_idle * 4;
  }
  dut->base_run = 0;
  return s;
}

static Stats run_seq_doubleACT(Vdual_top* dut, vluint64_t& tick,
                                int t12, int t23, int bank0_id,
                                uint8_t bank_mask,
                                int r_first, int r_second) {
  Stats s;
  dut->rst = 1;
  dut->base_run = 0;
  dut->seq_start    = 0;
  dut->seq_op_mode  = 1;       // DOUBLE_ACT
  dut->seq_op_code  = 0;
  dut->seq_bank_mask = bank_mask;
  dut->seq_bank0_id  = (uint8_t)bank0_id;
  dut->seq_base_col  = 0;
  dut->seq_base_row  = 0;
  dut->seq_count     = 0;
  dut->seq_stride    = 0;
  dut->seq_bl4 = 0; dut->seq_ap = 0;
  dut->seq_da_r_first  = (uint32_t)r_first;
  dut->seq_da_r_second = (uint32_t)r_second;
  dut->seq_da_t12      = (uint8_t)t12;
  dut->seq_da_t23      = (uint8_t)t23;
  for (int i = 0; i < 4; i++) cycle(dut, tick);
  dut->rst = 0;
  dut->seq_start = 1; cycle(dut, tick); dut->seq_start = 0;
  for (uint64_t c = 0; c < 200000; c++) {
    if (dut->seq_busy) {
      uint8_t any = dut->seq_ddr_act | dut->seq_ddr_read
                  | dut->seq_ddr_write | dut->seq_ddr_pre;
      s.tally(any);
      for (int slot = 0; slot < 4; slot++) {
        if ((dut->seq_ddr_act >> slot) & 1u) {
          Cmd cmd; cmd.slot = slot; cmd.kind = Cmd::K_ACT;
          cmd.bank = (dut->seq_ddr_bank >> (slot * 2)) & 0x3;
          cmd.row  = slice17(dut->seq_ddr_row, slot);
          s.cmds.push_back(cmd);
        }
        if ((dut->seq_ddr_pre >> slot) & 1u) {
          Cmd cmd; cmd.slot = slot; cmd.kind = Cmd::K_PRE;
          cmd.bank = (dut->seq_ddr_bank >> (slot * 2)) & 0x3;
          s.cmds.push_back(cmd);
        }
      }
    }
    if (dut->seq_done) break;
    cycle(dut, tick);
  }
  return s;
}

// Filter a multiset to a specific bank — used so we can compare a 1-bank
// baseline doubleACT against ONE bank-slice of the engine's 4-bank parallel
// emission.
static Stats filter_bank(const Stats& s, int bank) {
  Stats out;
  for (auto& c : s.cmds) if (c.bank == bank) out.cmds.push_back(c);
  return out;
}

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  auto* dut = new Vdual_top;
  vluint64_t tick = 0;

  // ============================================================
  // PART 1 — plain READ burst (existing test, kept as a sanity check).
  // ============================================================
  printf("############################################################\n");
  printf("PART 1: plain READ burst (PHY ceiling for standard DDR4 ops)\n");
  printf("############################################################\n");
  for (int N_CYCLES : {16, 64, 256}) {
    int N_CMDS = N_CYCLES * 4;
    printf("\n--- READ workload: %d cmds (ideal %d cycles) ---\n", N_CMDS, N_CYCLES);
    Stats base = run_baseline(dut, tick, N_CYCLES);
    // For PLAIN seq, op_mode = 0:
    dut->seq_op_mode = 0;
    Stats seq = run_seq(dut, tick, N_CYCLES);
    printf("  baseline: cyc=%lu util=%.2f%%   seq: cyc=%lu util=%.2f%%   speedup=%.1fx\n",
           (unsigned long)base.cycles, 100.0 * base.util(),
           (unsigned long)seq.cycles,  100.0 * seq.util(),
           (double)base.cycles / std::max((double)seq.cycles, 1.0));
    int diffs = compare_cmd_streams(base, seq);
    if (diffs) { printf(">>> CORRECTNESS FAIL <<<\n"); delete dut; return 1; }
  }

  // ============================================================
  // PART 2 — doubleACT primitives: BitNet PIM's actual hot path.
  // Three SiMRA timings used by BitNet on DIMM 0:
  //   (t_12=30, t_23=1) RowClone
  //   (t_12=10, t_23=2) broadcast
  //   (t_12=0,  t_23=0) MAJ3
  // For each timing:
  //   • baseline = pristine SoftMC running the api/util.cpp doubleACT() output
  //   • engine   = DOUBLE_ACT mode, 4 banks parallel
  // We also compare the engine's bank-0 slice against the 1-bank baseline
  // command stream to confirm bit-exact emission.
  // ============================================================
  printf("\n############################################################\n");
  printf("PART 2: SiMRA doubleACT — BitNet PIM hot path\n");
  printf("############################################################\n");
  struct Tcase { const char* name; int t12, t23; };
  Tcase tcs[] = {
    {"MAJ3      (t12=0,  t23=0)", 0,  0},
    {"broadcast (t12=10, t23=2)", 10, 2},
    {"RowClone  (t12=30, t23=1)", 30, 1},
  };
  const int R_FIRST = 100, R_SECOND = 200;
  int total_fail = 0;
  for (auto& tc : tcs) {
    printf("\n--- %s ---\n", tc.name);
    // Baseline: 1-bank (per current SoftMC pattern)
    Stats base = run_baseline_doubleACT(dut, tick, tc.t12, tc.t23,
                                        /*bank=*/0, R_FIRST, R_SECOND);
    // Engine: 4-bank PARALLEL (banks 0..3)
    Stats seq4 = run_seq_doubleACT(dut, tick, tc.t12, tc.t23,
                                    /*bank0=*/0, /*mask=*/0xF,
                                    R_FIRST, R_SECOND);
    // Engine: 1-bank for bit-exact correctness compare
    Stats seq1 = run_seq_doubleACT(dut, tick, tc.t12, tc.t23,
                                    /*bank0=*/0, /*mask=*/0x1,
                                    R_FIRST, R_SECOND);
    printf("  baseline (1bk): cyc=%lu cmds=%zu util=%.2f%%\n",
           (unsigned long)base.cycles, base.cmds.size(), 100.0 * base.util());
    printf("  seq      (1bk): cyc=%lu cmds=%zu util=%.2f%%\n",
           (unsigned long)seq1.cycles, seq1.cmds.size(), 100.0 * seq1.util());
    printf("  seq      (4bk): cyc=%lu cmds=%zu util=%.2f%%\n",
           (unsigned long)seq4.cycles, seq4.cmds.size(), 100.0 * seq4.util());
    // Correctness: engine 1-bk slice MUST equal baseline (modulo slot column,
    // which the baseline encodes as 0 for ACT/PRE — same in engine).
    int diffs = compare_cmd_streams(base, seq1);
    if (diffs) total_fail++;
    printf("  speedup vs baseline: cyc=%.1fx (1bk)  cyc=%.1fx (4bk parallel, 4× work)\n",
           (double)base.cycles / std::max((double)seq1.cycles, 1.0),
           (double)(4 * base.cycles) / std::max((double)seq4.cycles, 1.0));
    printf("  util ratio (4bk):    %.1fx  (baseline=%.2f%% seq4=%.2f%%)\n",
           seq4.util() / std::max(base.util(), 1e-9),
           100.0 * base.util(), 100.0 * seq4.util());
  }

  // ============================================================
  // PART 3 — BitNet PIM body chain: realistic hot path.
  // The SiMRA `emit_bank_combined_body` chains
  //   RowClone (30,1)  →  broadcast (10,2)  →  MAJ3 (0,0)
  // (plus 11 wrRow_immediate + 3 frac discharge that we skip here for the
  // doubleACT-only A/B; the wrRows are essentially WRITE primitives that
  // would slot into the seq_engine PLAIN-WRITE mode in a real integration).
  //
  // Baseline: chain of three SoftMC programs (= 3 platform.execute calls
  //           today; each carries the doubleACT pattern + LI overhead).
  // Engine:   3 back-to-back start-pulses on seq_engine in DOUBLE_ACT
  //           mode, no setup overhead between.
  // For each, sum cycles and active slots to get end-to-end util.
  // ============================================================
  printf("\n############################################################\n");
  printf("PART 3: BitNet PIM body-chain (RowClone → broadcast → MAJ3)\n");
  printf("############################################################\n");
  Tcase chain[] = {
    {"RowClone  (30,1)", 30, 1},
    {"broadcast (10,2)", 10, 2},
    {"MAJ3      (0, 0)", 0,  0},
  };
  Stats base_total, seq_total;
  for (auto& tc : chain) {
    Stats b = run_baseline_doubleACT(dut, tick, tc.t12, tc.t23, 0, 100, 200);
    Stats s = run_seq_doubleACT(dut, tick, tc.t12, tc.t23, 0, 0xF, 100, 200);
    base_total.cycles       += b.cycles;
    base_total.active_slots += b.active_slots;
    base_total.total_slots  += b.total_slots;
    seq_total.cycles        += s.cycles;
    seq_total.active_slots  += s.active_slots;
    seq_total.total_slots   += s.total_slots;
    printf("  %s :  base 1bk cyc=%lu (util=%.2f%%)   seq 4bk cyc=%lu (util=%.2f%%)\n",
           tc.name, (unsigned long)b.cycles, 100.0 * b.util(),
                    (unsigned long)s.cycles, 100.0 * s.util());
  }
  printf("\n  body chain total:\n");
  printf("    baseline (1 bank):   cyc=%lu  active=%lu  util=%.2f%%\n",
         (unsigned long)base_total.cycles,
         (unsigned long)base_total.active_slots, 100.0 * base_total.util());
  printf("    seq engine (4 bk):   cyc=%lu  active=%lu  util=%.2f%%\n",
         (unsigned long)seq_total.cycles,
         (unsigned long)seq_total.active_slots, 100.0 * seq_total.util());
  printf("    speedup (4× work):   %.1fx cycles  (4 banks done in seq's time)\n",
         (double)(4 * base_total.cycles) / std::max((double)seq_total.cycles, 1.0));
  printf("    util ratio:          %.1fx  (engine util / baseline util)\n",
         seq_total.util() / std::max(base_total.util(), 1e-9));

  delete dut;
  if (total_fail) { printf("\n>>> %d doubleACT correctness fail(s) <<<\n", total_fail); return 1; }
  printf("\n=== ALL CORRECT ===\n");
  return 0;
}
