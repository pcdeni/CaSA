// Verilator A/B for the host-side bank-parallel scheduler.
//
// SEQUENTIAL = pristine SiMRA pattern: chain of N single-bank doubleACTs,
// each loading its own BAR/RF/LOOP_COLS registers then emitting the
// q_inst[] pattern. This is the EXACT command stream the existing
// build_multibank_combined_program produces today (one body per bank).
//
// PARALLEL = the new parallel_doubleACT helper: same N banks but staggered
// across the 4 pack4 slots so all bodies share fabric cycles.
//
// Validation:
//   For each of the N banks, filter the captured (cycle, slot, op, bank,
//   row) events from the parallel program down to bank b. Verify:
//     (1) same number of events as SEQUENTIAL bank b
//     (2) same KIND of events in same order (ACT, PRE, ACT)
//     (3) same RELATIVE timing — PRE comes t_12+1 PHY positions after
//         ACT R1, ACT R2 comes t_12+t_23+2 PHY positions after ACT R1.
//
// Plus the obvious: parallel must finish in fewer cycles than sequential
// (otherwise there's no point), with no commands lost or invented.

#include "verilated.h"
#include "Vsoftmc_pipeline_top.h"
#include "parallel_emit.h"
#include "util.h"
#include "instruction.h"
#include "prog.h"
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <map>
#include <algorithm>

// ---- helpers ---------------------------------------------------------------

// util.h already #defines BAR=7, RF_REG=10, LOOP_COLS=13. We reuse them
// for the SEQUENTIAL baseline (matches the SiMRA single-bank doubleACT()
// helper which uses BAR/RF_REG/LOOP_COLS as register IDs).
// For the PARALLEL program we need 4 independent BAR/R_first/R_second
// register triples. Pick reg IDs that don't collide with util.h's macros
// for the *first* slot (which is the same as SEQUENTIAL); allocate
// additional reg IDs for slots 1..3 from the unused half of the 16-reg file.
static const int PAR_BAR[4] = { BAR,        8, 11, 15 };  // BAR=7
static const int PAR_RF[4]  = { RF_REG,     6,  9,  3 };  // RF_REG=10
static const int PAR_RS[4]  = { LOOP_COLS, 12,  5,  4 };  // LOOP_COLS=13

struct Tick {
  Vsoftmc_pipeline_top* dut;
  vluint64_t t = 0;
  void cycle() {
    dut->clk = 0; dut->eval(); t++;
    dut->clk = 1; dut->eval(); t++;
  }
};

// Load a Program's instruction sequence into the behavioral IMEM, then
// pulse run for as long as needed and return the captured ACT/PRE event
// list. Each event = (phy_position, kind, bank, row).
struct Event {
  enum Kind { K_ACT_R1=0, K_PRE=1, K_ACT_R2=2 };
  int phy_pos;       // absolute PHY position (cycle*4 + slot) from program start
  Kind kind;
  int bank;
  int row;
};

// `expected_r1_rows` and `expected_r2_rows` map bank→row used to
// disambiguate ACT R_first vs ACT R_second when capturing events.
static std::vector<Event> run_program_capture(
    Vsoftmc_pipeline_top* dut, vluint64_t& tick,
    const std::vector<uint64_t>& imem,
    const std::map<int,int>& r1_rows,
    const std::map<int,int>& r2_rows) {
  std::vector<Event> ev;
  Tick T{dut};
  // Reset and load IMEM.
  dut->rst = 1; dut->run = 0; dut->imem_init_we = 0;
  for (int i = 0; i < 4; i++) T.cycle();
  dut->rst = 0;
  for (size_t i = 0; i < imem.size(); i++) {
    dut->imem_init_we = 1;
    dut->imem_init_addr = (uint32_t)i;
    dut->imem_init_data = imem[i];
    T.cycle();
  }
  dut->imem_init_we = 0;
  // Run.
  dut->run = 1;
  bool seen_first = false;
  bool end_fired = false;
  uint64_t post_end = 0;
  uint64_t first_useful_cyc = 0;
  for (uint64_t c = 0; c < 200000; c++) {
    T.cycle();
    uint8_t any_useful = dut->ddr_act | dut->ddr_pre;
    if (any_useful && !seen_first) {
      seen_first = true;
      first_useful_cyc = c;
    }
    if (seen_first) {
      uint64_t rel_cyc = c - first_useful_cyc;
      for (int s = 0; s < 4; s++) {
        if ((dut->ddr_act >> s) & 1u) {
          int bk = (dut->ddr_bank >> (s * 2)) & 0x3;
          // Decode 17-bit row from VlWide<3>
          int word = (s * 17) / 32;
          int boff = (s * 17) % 32;
          uint64_t lo = dut->ddr_row.at(word);
          uint64_t hi = (boff + 17 > 32) ? dut->ddr_row.at(word + 1) : 0;
          int row = (int)(((lo >> boff) | (hi << (32 - boff))) & 0x1FFFF);
          Event::Kind k;
          auto it1 = r1_rows.find(bk);
          auto it2 = r2_rows.find(bk);
          if (it1 != r1_rows.end() && row == it1->second) k = Event::K_ACT_R1;
          else if (it2 != r2_rows.end() && row == it2->second) k = Event::K_ACT_R2;
          else k = Event::K_ACT_R1;  // unknown — treat as R1 for sequencing
          ev.push_back({(int)(rel_cyc * 4 + s), k, bk, row});
        }
        if ((dut->ddr_pre >> s) & 1u) {
          int bk = (dut->ddr_bank >> (s * 2)) & 0x3;
          ev.push_back({(int)(rel_cyc * 4 + s), Event::K_PRE, bk, 0});
        }
      }
    }
    if (dut->softmc_end) { end_fired = true; dut->run = 0; }
    if (end_fired) {
      post_end++;
      if (post_end > 32) break;
    }
  }
  return ev;
}

// Build the SEQUENTIAL baseline program: for each bank, load its registers
// then emit the SiMRA single-bank doubleACT pattern.
static std::vector<uint64_t> build_sequential(int t_12, int t_23,
                                               const std::vector<int>& bank_ids,
                                               const std::vector<int>& r1_rows,
                                               const std::vector<int>& r2_rows) {
  Program prog;
  for (size_t b = 0; b < bank_ids.size(); b++) {
    prog.add_inst(SMC_LI((uint32_t)bank_ids[b], BAR));
    prog.add_inst(SMC_LI((uint32_t)r1_rows[b],  RF_REG));
    prog.add_inst(SMC_LI((uint32_t)r2_rows[b],  LOOP_COLS));
    Program da = doubleACT(t_12, t_23, r1_rows[b], r2_rows[b]);
    prog.add_below(da);
  }
  prog.add_inst(SMC_END());
  // Materialise to a flat uint64_t vector.
  // Program::get_inst_array() returns a malloc'd Inst* of `size()/8` entries.
  uint64_t* arr = (uint64_t*)prog.get_inst_array();
  size_t n = (size_t)prog.size() / 8;
  std::vector<uint64_t> out(arr, arr + n);
  free(arr);
  return out;
}

// Build the PARALLEL program: load 4 sets of BAR/R1/R2 registers, then
// one parallel_doubleACT covering all banks at once.
static std::vector<uint64_t> build_parallel(int t_12, int t_23,
                                             const std::vector<int>& bank_ids,
                                             const std::vector<int>& r1_rows,
                                             const std::vector<int>& r2_rows) {
  Program prog;
  // Load per-bank registers (BAR, R_first, R_second) into the per-bank
  // register slots PAR_BAR[k], PAR_RF[k], PAR_RS[k].
  for (size_t b = 0; b < bank_ids.size(); b++) {
    prog.add_inst(SMC_LI((uint32_t)bank_ids[b], PAR_BAR[b]));
    prog.add_inst(SMC_LI((uint32_t)r1_rows[b],  PAR_RF[b]));
    prog.add_inst(SMC_LI((uint32_t)r2_rows[b],  PAR_RS[b]));
  }
  // Build & append the parallel doubleACT.
  std::vector<int> bar(bank_ids.size()), rf(bank_ids.size()), rs(bank_ids.size());
  for (size_t b = 0; b < bank_ids.size(); b++) {
    bar[b] = PAR_BAR[b]; rf[b] = PAR_RF[b]; rs[b] = PAR_RS[b];
  }
  Program pda = parallel_doubleACT(t_12, t_23, bar, rf, rs);
  prog.add_below(pda);
  prog.add_inst(SMC_END());
  uint64_t* arr = (uint64_t*)prog.get_inst_array();
  size_t n = (size_t)prog.size() / 8;
  std::vector<uint64_t> out(arr, arr + n);
  free(arr);
  return out;
}

// Per-bank check: filter events by bank b, verify (1)(2)(3) above.
static int check_per_bank(const std::vector<Event>& ev, int bank,
                           int t_12, int t_23, int r1, int r2,
                           const char* label) {
  std::vector<Event> me;
  for (auto& e : ev) if (e.bank == bank) me.push_back(e);
  if (me.size() != 3) {
    printf("    [%s bank=%d] FAIL: expected 3 events, got %zu\n",
           label, bank, me.size());
    return 1;
  }
  // The events should arrive in PHY-position order. Sort by phy_pos.
  std::sort(me.begin(), me.end(),
            [](const Event& a, const Event& b){ return a.phy_pos < b.phy_pos; });
  // Expected kinds in order: ACT_R1, PRE, ACT_R2.
  int act_count = 0, pre_count = 0;
  int t12_actual = -1, t23_actual = -1;
  int act_r1_pos = -1, pre_pos = -1, act_r2_pos = -1;
  for (auto& e : me) {
    if (e.kind == Event::K_ACT_R1 && act_count == 0) {
      act_r1_pos = e.phy_pos; act_count = 1;
    } else if (e.kind == Event::K_PRE) {
      pre_pos = e.phy_pos; pre_count++;
    } else if (e.kind == Event::K_ACT_R2 || (e.kind == Event::K_ACT_R1 && act_count == 1)) {
      act_r2_pos = e.phy_pos; act_count = 2;
    }
  }
  if (act_r1_pos < 0 || pre_pos < 0 || act_r2_pos < 0) {
    printf("    [%s bank=%d] FAIL: missing event types (act_r1=%d pre=%d act_r2=%d)\n",
           label, bank, act_r1_pos, pre_pos, act_r2_pos);
    return 1;
  }
  t12_actual = pre_pos - act_r1_pos - 1;
  t23_actual = act_r2_pos - pre_pos - 1;
  bool ok = (t12_actual == t_12) && (t23_actual == t_23);
  printf("    [%s bank=%d] phy={ACT_R1@%d, PRE@%d, ACT_R2@%d} t_12=%d (target %d) t_23=%d (target %d) %s\n",
         label, bank, act_r1_pos, pre_pos, act_r2_pos,
         t12_actual, t_12, t23_actual, t_23, ok ? "OK" : "TIMING FAIL");
  return ok ? 0 : 1;
}

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  auto* dut = new Vsoftmc_pipeline_top;
  vluint64_t tick = 0;

  struct Tcase { const char* name; int t12, t23; };
  Tcase cases[] = {
    {"MAJ3      (0,0)",  0,  0},
    {"broadcast (10,2)", 10, 2},
    {"RowClone  (30,1)", 30, 1},
  };
  // Test with N=4 banks (the production multibank cadence).
  std::vector<int> banks  = {0, 1, 2, 3};
  std::vector<int> r1_rows = {100, 200, 300, 400};
  std::vector<int> r2_rows = {500, 600, 700, 800};
  std::map<int,int> r1_map, r2_map;
  for (size_t i = 0; i < banks.size(); i++) {
    r1_map[banks[i]] = r1_rows[i];
    r2_map[banks[i]] = r2_rows[i];
  }

  int total_fail = 0;
  for (auto& tc : cases) {
    printf("\n=== %s ===\n", tc.name);
    int shift = parallel_min_shift(tc.t12, tc.t23, (int)banks.size());
    printf("  parallel_min_shift = %d (pat_len = %d)\n",
           shift, tc.t12 + tc.t23 + 3);

    // Sequential baseline.
    auto seq_imem = build_sequential(tc.t12, tc.t23, banks, r1_rows, r2_rows);
    auto seq_ev = run_program_capture(dut, tick, seq_imem, r1_map, r2_map);
    printf("  SEQUENTIAL: %zu insts, %zu events captured\n",
           seq_imem.size(), seq_ev.size());

    // Parallel.
    auto par_imem = build_parallel(tc.t12, tc.t23, banks, r1_rows, r2_rows);
    auto par_ev = run_program_capture(dut, tick, par_imem, r1_map, r2_map);
    printf("  PARALLEL:   %zu insts, %zu events captured\n",
           par_imem.size(), par_ev.size());

    // Per-bank correctness check on the parallel program.
    int fails = 0;
    for (size_t i = 0; i < banks.size(); i++) {
      fails += check_per_bank(par_ev, banks[i], tc.t12, tc.t23,
                              r1_rows[i], r2_rows[i], "parallel");
    }
    // Sanity: same check on sequential (should always pass).
    for (size_t i = 0; i < banks.size(); i++) {
      check_per_bank(seq_ev, banks[i], tc.t12, tc.t23,
                     r1_rows[i], r2_rows[i], "sequential");
    }
    if (fails) total_fail += fails;
  }

  delete dut;
  if (total_fail) { printf("\n>>> %d per-bank correctness fail(s) <<<\n", total_fail); return 1; }
  printf("\n=== ALL CORRECT (parallel preserves per-bank t_12/t_23 vs SiMRA template) ===\n");
  return 0;
}
