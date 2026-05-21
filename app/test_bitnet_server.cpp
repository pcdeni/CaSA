// BitNet PIM server — long-running daemon that opens the FPGA platform
// once and processes matmul requests over stdin/stdout. Eliminates the
// per-call subprocess startup + platform.init() + reset_fpga() overhead
// that dominates today's measured throughput.
//
// Protocol (binary, all little-endian):
//   - server reads:  u32 req_len   (0 = quit sentinel)
//                    req_len bytes (identical to bitnet-proj-exe's
//                                   v2 inputs.bin body — magic +
//                                   header + masks + x_bitplane +
//                                   bitplane_factor)
//   - server writes: 8192 bytes of int32[2048] = the y output array
//                    (same as bitnet-proj-exe's output.bin body)
//
// Status / errors written to stderr only (so stdout stays a clean
// binary stream).
//
// Multi-bank parallelism (Path C): bank_arg can be a single integer
// ("1") for single-bank operation, or comma-separated ("0,1,2,3") to
// distribute (chunk, sign) work units round-robin across N banks. Each
// bank gets one calibrated tuple in its own subarray, its own backup
// pool of persistent-weight rows, and its own MAJ3 body inside the
// combined program. N banks per execute amortizes per-execute PCIe
// overhead and gives ~1.3-1.5× per-MAJ3 speedup (Path C smoke test).
//
// Argv:
//   ./bitnet-proj-server <bender_id> <calib_file> <bank_arg>
//     bank_arg: "1" (single) or "0,1,2,3" (multi-bank, up to 4 banks)
#include "instruction.h"
#include "prog.h"
#include "platform.h"
#include "sim_platform.h"
#include "../util.h"
#include "parallel_emit.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

using namespace std;

static const int CHUNK_COLS[3] = {43, 43, 42};
static constexpr uint32_t MAGIC_V2 = 0xB17EF002u;
static constexpr uint32_t MAGIC_LOAD = 0xB17EF003u;   // LOAD_WEIGHTS
static constexpr uint32_t MAGIC_MM3D = 0xB17EF004u;   // MATMUL with handle

static Program build_chunk_program(int bank_id, uint32_t row_addr,
                                    const uint32_t* col_data,
                                    int col_start, int n_cols) {
  Program p;
  p.add_inst(SMC_LI(8, CASR));
  p.add_inst(SMC_LI(bank_id, BAR));
  p.add_inst(SMC_LI(row_addr, RAR));
  p.add_inst(SMC_LI(col_start * 8, CAR));
  p.add_below(PRE(BAR, 0, 0));
  p.add_below(ACT(BAR, 0, RAR, 0));
  for (int k = 0; k < n_cols; k++) {
    const uint32_t* slots = col_data + k * 16;
    for (int slot = 0; slot < 16; slot++) {
      p.add_inst(SMC_LI(slots[slot], PATTERN_REG));
      p.add_inst(SMC_LDWD(PATTERN_REG, slot));
    }
    p.add_below(WRITE(BAR, CAR, 1));
    p.add_inst(SMC_SLEEP(8));
  }
  p.add_inst(SMC_SLEEP(8));
  p.add_below(PRE(BAR, 0, 0));
  p.add_inst(SMC_SLEEP(4));
  p.add_inst(SMC_END());
  return p;
}

static void per_column_write_row(SoftMCPlatform& platform, int bank_id,
                                  uint32_t row, const uint32_t* data_2048) {
  int col_start = 0;
  for (int chunk = 0; chunk < 3; chunk++) {
    int n_cols = CHUNK_COLS[chunk];
    Program p = build_chunk_program(bank_id, row,
                                     data_2048 + col_start * 16,
                                     col_start, n_cols);
    platform.execute(p);
    col_start += n_cols;
  }
}

// SiMRA-style frac discharge — single ACT-PRE pair on a row, with proper
// NOP padding around it. Lifted directly from MajOperations/test.cpp so
// we use the same template the calibration tests use; that prevents
// hand-inlined ACT+PRE patterns from drifting in subtle ways.
static Program frac_template(int t_frac, uint32_t r_frac_addr) {
  Program p;
  int R_FRAC_REG = RF_REG;
  int bank_reg = BAR;
  p.add_inst(all_nops());
  p.add_inst(SMC_LI(r_frac_addr, R_FRAC_REG));
  int num_cmd = 2 + t_frac;
  num_cmd += (num_cmd % 4) ? (4 - (num_cmd % 4)) : 0;
  Mininst* q_inst = new Mininst[num_cmd];
  for (int i = 0; i < num_cmd; i++) q_inst[i] = SMC_NOP();
  q_inst[0]            = SMC_ACT(bank_reg, 0, R_FRAC_REG, 0);
  q_inst[t_frac + 1]   = SMC_PRE(bank_reg, 0, 0);
  for (int i = 0; i < num_cmd; i += 4)
    p.add_inst(q_inst[i], q_inst[i + 1], q_inst[i + 2], q_inst[i + 3]);
  delete[] q_inst;
  return p;
}

// Build a "refresh specific rows" program: ACT each row + PRE, in a
// specified bank. ACT pulls each row's content into the sense amps,
// which then drive the cells back to full charge — that's what refresh
// physically does. We call this at the start of each MM3D request to
// keep all loaded backup rows alive across long idle gaps.
//
// Why not SMC_REF? Standard DDR4 REF refreshes only ONE row per bank
// per command (internal counter cycles through 8K rows over 64 ms).
// One REF per program = useless for our needs. Explicit ACT+PRE on
// each loaded row IS the refresh.
static Program build_refresh_rows_program(
    int bank_id, const std::vector<uint32_t>& rows) {
  Program p;
  p.add_inst(SMC_LI(8, CASR));
  p.add_inst(SMC_LI(bank_id, BAR));
  for (uint32_t row : rows) {
    p.add_inst(SMC_LI(row, RAR));
    p.add_below(PRE(BAR, 0, 0));   // ensure bank idle
    p.add_below(ACT(BAR, 0, RAR, 0));  // open row → sense amps refresh cells
    p.add_below(PRE(BAR, 0, 0));   // close row, write-back cells
  }
  p.add_inst(SMC_END());
  return p;
}

// Looped refresh of a contiguous row range across one or more banks.
// Uses SoftMC's branch instruction so program size is O(1) regardless
// of row count — fits in the 2048-instruction buffer for any range.
//
// Per bank: SMC_LI(start, LOOP_ROWS), SMC_LI(end, NUM_ROWS_REG), then
//   loop: copy LOOP_ROWS→RAR, PRE+ACT+PRE, increment, branch-if-less.
// ~10 instructions outside the loop per bank; loop body is ~9 inst.
//
// Each bank's range may differ (bank-specific subarrays), so the
// caller passes parallel vectors row_starts[i] and row_ends[i] for
// bank_ids[i]. Total wall-clock: ~640 rows × ~25 cycles @ 250 MHz
// ≈ 64 µs per bank.
static Program build_refresh_subarray_loop_program(
    const std::vector<int>& bank_ids,
    const std::vector<uint32_t>& row_starts,
    const std::vector<uint32_t>& row_ends /* exclusive */) {
  Program p;
  p.add_inst(SMC_LI(8, CASR));
  for (size_t bi = 0; bi < bank_ids.size(); bi++) {
    int bank = bank_ids[bi];
    uint32_t r0 = row_starts[bi];
    uint32_t r1 = row_ends[bi];
    p.add_inst(SMC_LI(bank, BAR));
    p.add_inst(SMC_LI(r0, LOOP_ROWS));
    p.add_inst(SMC_LI(r1, NUM_ROWS_REG));
    std::string lab = "REFRESH_SUBARR_B" + std::to_string(bank);
    p.add_label(lab);
      p.add_inst(SMC_ADDI(LOOP_ROWS, 0, RAR));      // RAR = LOOP_ROWS
      p.add_below(PRE(BAR, 0, 0));                   // ensure idle
      p.add_below(ACT(BAR, 0, RAR, 0));              // open → refresh
      p.add_inst(SMC_SLEEP(4));                      // tRAS guard
      p.add_below(PRE(BAR, 0, 0));                   // close & writeback
      p.add_inst(SMC_SLEEP(4));                      // tRP guard
      p.add_inst(SMC_ADDI(LOOP_ROWS, 1, LOOP_ROWS));
    p.add_branch(p.BR_TYPE::BL, LOOP_ROWS, NUM_ROWS_REG, lab);
    p.add_inst(all_nops());
  }
  p.add_inst(SMC_END());
  return p;
}

// Read-back a single row's contents into row_buf (must be ≥ 8192 bytes).
// Used by LOAD-time write verification and MM3D-start decay detection.
static int read_row_to_buffer(SoftMCPlatform& platform, int bank_id,
                              uint32_t row, uint8_t* row_buf,
                              int label_seed) {
  Program p;
  p.add_inst(SMC_LI(8, CASR));
  p.add_inst(SMC_LI(bank_id, BAR));
  p.add_below(PRE(BAR, 0, 0));
  p.add_below(rdRow_immediate_label(BAR, row, label_seed));
  p.add_inst(all_nops());
  p.add_inst(all_nops());
  p.add_below(PRE(BAR, 0, 0));
  p.add_inst(all_nops());
  p.add_inst(all_nops());
  p.add_inst(SMC_END());
  platform.execute(p);
  return platform.receiveData(row_buf, 8192);
}

// RowClone: charge-sharing 2-row copy via doubleACT(t_12=30, t_23=1).
// Used for "persistent weights" mode: backup row → Rfirst before each
// MAJ3. Cheap (single doubleACT) compared to per-column write of W.
static Program build_rowclone_program(int bank_id,
                                       uint32_t src_row, uint32_t dst_row) {
  Program p;
  p.add_inst(SMC_LI(8, CASR));
  p.add_inst(SMC_LI(bank_id, BAR));
  p.add_inst(SMC_LI(128, NUM_COLS_REG));
  p.add_below(PRE(BAR, 0, 0));
  p.add_inst(SMC_SLEEP(6));
  p.add_below(doubleACT(/*t_12=*/30, /*t_23=*/1, src_row, dst_row));
  p.add_inst(SMC_SLEEP(6));
  p.add_below(PRE(BAR, 0, 0));
  p.add_inst(SMC_SLEEP(6));
  p.add_inst(SMC_END());
  return p;
}

// One bank's combined RowClone → broadcast → uniform writes → frac →
// MAJ3 → read body. Caller must have set CASR (SMC_LI(8, CASR)) once
// before the first body and emit SMC_END() after the last body. Bank
// switching done at start of body via SMC_LI(bank_id, BAR).
static void emit_bank_combined_body(Program& p,
                                     int bank_id,
                                     uint32_t backup_row,
                                     uint32_t Rfirst, uint32_t Rsecond,
                                     const uint32_t* open_rows,
                                     uint32_t x_pattern,
                                     int label_base) {
  // Per-bank register setup: BAR + NUM_COLS_REG must be (re-)set when
  // this bank's body starts so prior bank's state doesn't leak in.
  p.add_inst(SMC_LI(bank_id, BAR));
  p.add_inst(SMC_LI(128, NUM_COLS_REG));

  // 1. RowClone scratch_row → Rfirst. Per-column writes happened on
  // scratch_row (in the safe zone, far from open_rows[]) to avoid
  // disturbing the calibrated open_rows during the 128 ACT/WRITE cycles.
  // RowClone is a single doubleACT — minimal disturb — copies scratch
  // to Rfirst so the broadcast step has fresh weight data.
  p.add_below(PRE(BAR, 0, 0));
  p.add_inst(SMC_SLEEP(6));
  p.add_below(doubleACT(/*t_12=*/30, /*t_23=*/1, backup_row, Rfirst));
  p.add_inst(SMC_SLEEP(6));
  p.add_below(PRE(BAR, 0, 0));
  p.add_inst(SMC_SLEEP(6));

  // 2. Broadcast Rfirst → all 16 open rows.
  p.add_below(doubleACT(10, 2, Rfirst, Rsecond));
  p.add_inst(SMC_SLEEP(6));
  p.add_below(PRE(BAR, 0, 0));
  p.add_inst(SMC_SLEEP(6));

  // 3. Overwrite 11 non-weight slots with x / 0 / ONE.
  static const int act_pos[5]  = {1, 4, 7, 10, 13};
  static const int zero_pos[5] = {2, 5, 8, 11, 14};
  p.add_below(wrRow_immediate_label(BAR, open_rows[0], ONE, label_base + 0));
  for (int i = 0; i < 5; i++)
    p.add_below(wrRow_immediate_label(BAR, open_rows[act_pos[i]],
                                       x_pattern, label_base + 1 + i));
  for (int i = 0; i < 5; i++)
    p.add_below(wrRow_immediate_label(BAR, open_rows[zero_pos[i]], 0u,
                                       label_base + 100 + i));
  p.add_inst(SMC_SLEEP(6));
  p.add_below(PRE(BAR, 0, 0));
  p.add_inst(SMC_SLEEP(6));

  // 4. Frac discharge × 3 on open_rows[0] — using SiMRA's frac()
  // template (matches MajOperations/test.cpp exactly) instead of an
  // inline ACT-PRE pair, so we don't accidentally drop the PRE between
  // ACTs the way a hand-inlined sequence might.
  for (int j = 0; j < 3; j++) {
    p.add_inst(SMC_SLEEP(6));
    p.add_below(frac_template(/*t_frac=*/0, open_rows[0]));
    p.add_inst(SMC_SLEEP(6));
  }
  p.add_inst(SMC_SLEEP(6));
  p.add_below(PRE(BAR, 0, 0));
  p.add_inst(SMC_SLEEP(6));

  // 5. MAJ3 doubleACT(0, 0).
  p.add_below(doubleACT(0, 0, Rfirst, Rsecond));
  p.add_inst(SMC_SLEEP(6));
  p.add_below(PRE(BAR, 0, 0));
  p.add_inst(SMC_SLEEP(6));

  // 6. Read result for this bank — followed by SiMRA's standard
  // post-rdRow trailing pattern (8 NOPs + PRE + 8 NOPs) so the data
  // burst fully drains and the bank is precharged before whatever
  // comes next (SMC_END or another bank's body in multi-bank mode).
  p.add_below(rdRow_immediate_label(BAR, open_rows[0], label_base + 999));
  p.add_inst(all_nops());
  p.add_inst(all_nops());
  p.add_below(PRE(BAR, 0, 0));
  p.add_inst(all_nops());
  p.add_inst(all_nops());
}

// Single-bank combined program. Wrapper around emit_bank_combined_body.
// Kept for direct/test paths; the server uses build_multibank_*.
static Program build_combined_clone_bcast_maj3_program(
    int bank_id,
    uint32_t backup_row,
    uint32_t Rfirst, uint32_t Rsecond,
    const uint32_t* open_rows,
    uint32_t x_pattern,
    int label_seed) {
  Program p;
  p.add_inst(SMC_LI(8, CASR));
  emit_bank_combined_body(p, bank_id, backup_row, Rfirst, Rsecond,
                          open_rows, x_pattern, label_seed);
  p.add_inst(SMC_END());
  return p;
}

// Multi-bank combined program — up to 4 banks' MAJ3 bodies emitted
// back-to-back inside one Program. Each entry in `bank_ids/backup_rows
// /Rfirsts/Rseconds/open_rows_list/x_patterns` is one bank's work unit
// for this execute. After execute, caller must call receiveData(8192)
// once per emitted bank in the same order.
//
// Program size: ~280 instructions per bank × N banks. With N=4 → ~1120
// instructions, well under the 2048-instruction buffer.
static Program build_multibank_combined_program(
    const std::vector<int>& bank_ids,
    const std::vector<uint32_t>& backup_rows,
    const std::vector<uint32_t>& Rfirsts,
    const std::vector<uint32_t>& Rseconds,
    const std::vector<const uint32_t*>& open_rows_list,
    const std::vector<uint32_t>& x_patterns,
    int label_seed) {
  Program p;
  p.add_inst(SMC_LI(8, CASR));
  for (size_t i = 0; i < bank_ids.size(); i++) {
    emit_bank_combined_body(p, bank_ids[i], backup_rows[i],
                            Rfirsts[i], Rseconds[i],
                            open_rows_list[i], x_patterns[i],
                            label_seed + (int)i * 2000);
    // PIM_REFRESH_BETWEEN: optional periodic SMC_REF between bank bodies.
    // SMC_REF advances DDR4's internal refresh counter (one row per bank
    // per command, cycles through 8K rows over 64 ms). Manual placement
    // here — at safe inter-body boundaries, not inside doubleACT — sidesteps
    // the auto-refresh issue (auto fires mid-doubleACT and disrupts
    // charge-sharing). Tests whether ACT-disturb on open_rows can be
    // mitigated this way.
    static int s_refresh_between = -1;
    if (s_refresh_between < 0) {
      const char* v = getenv("PIM_REFRESH_BETWEEN");
      s_refresh_between = (v && *v) ? atoi(v) : 0;
    }
    if (s_refresh_between > 0 &&
        ((int)(i + 1) % s_refresh_between == 0)) {
      p.add_inst(SMC_REF(), SMC_NOP(), SMC_NOP(), SMC_NOP());
    }
  }
  p.add_inst(SMC_END());
  return p;
}

// Bank-parallel variant: emit the SiMRA doubleACTs (RowClone t_12=30/t_23=1,
// Broadcast 10/2, MAJ3 0/0) as 4-bank parallel pack4 sequences via
// `parallel_doubleACT`, while keeping the wrRow/frac/rdRow serial bodies
// per bank (those use shared regs / wdata that aren't easily 4-way
// parallelized today). Net win: ~3-4× compression on the SiMRA stages
// (RowClone goes 4×34 PHY → 37 PHY, Broadcast 4×14 → 21, MAJ3 stays).
//
// Register layout (live across the parallel sections):
//   slot 0: CASR   = 8 (kept stable; column-stride for ICAR)
//   slots 1, 2, 3, 9: bar0..bar3 (LI'd once at body top, never overwritten)
//   slots 11, 4, 5, 8: src_reg[0..3] — re-LI'd per parallel phase
//   slots 13, 6, 14, 15: dst_reg[0..3] — re-LI'd per parallel phase
// Other reg uses (PATTERN_REG=12, BAR=7, etc.) are confined to the
// per-bank serial sections and re-LI their inputs themselves; we
// re-establish CASR=8 and the canonical BAR=bank_id at the top of each
// serial section.
static Program build_multibank_parallel_program(
    const std::vector<int>& bank_ids,
    const std::vector<uint32_t>& backup_rows,
    const std::vector<uint32_t>& Rfirsts,
    const std::vector<uint32_t>& Rseconds,
    const std::vector<const uint32_t*>& open_rows_list,
    const std::vector<uint32_t>& x_patterns,
    int label_seed)
{
  const int N = (int)bank_ids.size();
  // Today the parallel scheduler is wired for N=4. Smaller N falls back
  // to the serial multibank emit (parallel-of-1 is identical to serial).
  if (N != 4) {
    return build_multibank_combined_program(
        bank_ids, backup_rows, Rfirsts, Rseconds,
        open_rows_list, x_patterns, label_seed);
  }
  Program p;
  p.add_inst(SMC_LI(8, CASR));
  // Pre-LI bar regs (stable across the whole body).
  const std::vector<int> bar_reg = {1, 2, 3, 9};
  for (int b = 0; b < 4; b++)
    p.add_inst(SMC_LI((uint32_t)bank_ids[b], bar_reg[b]));
  // src/dst row reg slots used by parallel_doubleACT.
  const std::vector<int> src_reg = {11, 4, 5, 8};
  const std::vector<int> dst_reg = {13, 6, 14, 15};

  auto li_rows = [&](const std::vector<int>& slots,
                     const std::vector<uint32_t>& vals) {
    for (int b = 0; b < 4; b++)
      p.add_inst(SMC_LI(vals[b], slots[b]));
  };
  auto pre_all = [&]() {
    p.add_inst(SMC_PRE(bar_reg[0], 0, 0),
               SMC_PRE(bar_reg[1], 0, 0),
               SMC_PRE(bar_reg[2], 0, 0),
               SMC_PRE(bar_reg[3], 0, 0));
  };

  // Phase 1: parallel RowClone scratch[k] → Rfirst[k] (t_12=30, t_23=1).
  pre_all();
  p.add_inst(SMC_SLEEP(6));
  li_rows(src_reg, backup_rows);
  li_rows(dst_reg, Rfirsts);
  p.add_below(parallel_doubleACT(30, 1, bar_reg, src_reg, dst_reg));
  p.add_inst(SMC_SLEEP(6));
  pre_all();
  p.add_inst(SMC_SLEEP(6));

  // Phase 2: parallel Broadcast Rfirst[k] → all 16 open_rows[k] (t_12=10, t_23=2).
  li_rows(src_reg, Rfirsts);
  li_rows(dst_reg, Rseconds);
  p.add_below(parallel_doubleACT(10, 2, bar_reg, src_reg, dst_reg));
  p.add_inst(SMC_SLEEP(6));
  pre_all();
  p.add_inst(SMC_SLEEP(6));

  // Phase 3 + 4: per-bank serial wrRow setup × 11 + frac × 3.
  // BAR / PATTERN_REG / LOOP_COLS / NUM_COLS_REG / RAR / CAR are
  // re-LI'd by the wrRow_immediate_label / frac_template helpers.
  // We re-establish BAR=bank_id, NUM_COLS_REG=128 at the top of each
  // bank's serial section.
  static const int act_pos[5]  = {1, 4, 7, 10, 13};
  static const int zero_pos[5] = {2, 5, 8, 11, 14};
  for (int b = 0; b < 4; b++) {
    p.add_inst(SMC_LI((uint32_t)bank_ids[b], BAR));
    p.add_inst(SMC_LI(128, NUM_COLS_REG));
    p.add_below(wrRow_immediate_label(BAR, open_rows_list[b][0],
                                       ONE, label_seed + b*2000 + 0));
    for (int i = 0; i < 5; i++)
      p.add_below(wrRow_immediate_label(BAR, open_rows_list[b][act_pos[i]],
                                         x_patterns[b],
                                         label_seed + b*2000 + 1 + i));
    for (int i = 0; i < 5; i++)
      p.add_below(wrRow_immediate_label(BAR, open_rows_list[b][zero_pos[i]],
                                         0u,
                                         label_seed + b*2000 + 100 + i));
    p.add_inst(SMC_SLEEP(6));
    p.add_below(PRE(BAR, 0, 0));
    p.add_inst(SMC_SLEEP(6));
    for (int j = 0; j < 3; j++) {
      p.add_inst(SMC_SLEEP(6));
      p.add_below(frac_template(/*t_frac=*/0, open_rows_list[b][0]));
      p.add_inst(SMC_SLEEP(6));
    }
    p.add_inst(SMC_SLEEP(6));
    p.add_below(PRE(BAR, 0, 0));
    p.add_inst(SMC_SLEEP(6));
  }

  // Phase 5: parallel MAJ3 (t_12=0, t_23=0). Re-LI src=Rfirst, dst=Rsecond
  // since the serial wrRow/frac sections clobbered the row regs.
  li_rows(src_reg, Rfirsts);
  li_rows(dst_reg, Rseconds);
  p.add_inst(SMC_SLEEP(6));   // RAW hazard barrier: let LI writes propagate
  p.add_below(parallel_doubleACT(0, 0, bar_reg, src_reg, dst_reg));
  p.add_inst(SMC_SLEEP(6));
  pre_all();
  p.add_inst(SMC_SLEEP(6));

  // Phase 6: per-bank serial rdRow (data bus serializes anyway).
  for (int b = 0; b < 4; b++) {
    p.add_inst(SMC_LI((uint32_t)bank_ids[b], BAR));
    p.add_inst(SMC_LI(128, NUM_COLS_REG));
    p.add_below(rdRow_immediate_label(BAR, open_rows_list[b][0],
                                        label_seed + b*2000 + 999));
    p.add_inst(all_nops());
    p.add_inst(all_nops());
    p.add_below(PRE(BAR, 0, 0));
    p.add_inst(all_nops());
    p.add_inst(all_nops());
  }

  p.add_inst(SMC_END());
  return p;
}

static Program build_bcast_maj3_program(int bank_id,
                                         uint32_t Rfirst, uint32_t Rsecond,
                                         const uint32_t* open_rows,
                                         uint32_t x_pattern,
                                         int label_seed) {
  Program p;
  p.add_inst(SMC_LI(8, CASR));
  p.add_inst(SMC_LI(bank_id, BAR));
  p.add_inst(SMC_LI(128, NUM_COLS_REG));
  p.add_below(PRE(BAR, 0, 0));
  p.add_inst(SMC_SLEEP(6));
  p.add_below(doubleACT(10, 2, Rfirst, Rsecond));
  p.add_inst(SMC_SLEEP(6));
  p.add_below(PRE(BAR, 0, 0));
  p.add_inst(SMC_SLEEP(6));
  static const int act_pos[5]  = {1, 4, 7, 10, 13};
  static const int zero_pos[5] = {2, 5, 8, 11, 14};
  p.add_below(wrRow_immediate_label(BAR, open_rows[0], ONE, label_seed));
  for (int i = 0; i < 5; i++)
    p.add_below(wrRow_immediate_label(BAR, open_rows[act_pos[i]],
                                       x_pattern, label_seed + 1 + i));
  for (int i = 0; i < 5; i++)
    p.add_below(wrRow_immediate_label(BAR, open_rows[zero_pos[i]], 0u,
                                       label_seed + 100 + i));
  p.add_inst(SMC_SLEEP(6));
  p.add_below(PRE(BAR, 0, 0));
  p.add_inst(SMC_SLEEP(6));
  for (int j = 0; j < 3; j++) {
    p.add_inst(SMC_SLEEP(6));
    p.add_below(frac_template(/*t_frac=*/0, open_rows[0]));
    p.add_inst(SMC_SLEEP(6));
  }
  p.add_inst(SMC_SLEEP(6));
  p.add_below(PRE(BAR, 0, 0));
  p.add_inst(SMC_SLEEP(6));
  p.add_below(doubleACT(0, 0, Rfirst, Rsecond));
  p.add_inst(SMC_SLEEP(6));
  p.add_below(PRE(BAR, 0, 0));
  p.add_inst(SMC_SLEEP(6));
  p.add_below(rdRow_immediate_label(BAR, open_rows[0], label_seed + 999));
  p.add_inst(SMC_END());
  return p;
}

static void segment_popcount(const uint8_t* row_buf, int* out, int n) {
  for (int s = 0; s < n; s++) {
    uint32_t actual = (uint32_t)row_buf[s*4]
                    | ((uint32_t)row_buf[s*4+1] << 8)
                    | ((uint32_t)row_buf[s*4+2] << 16)
                    | ((uint32_t)row_buf[s*4+3] << 24);
    out[s] = __builtin_popcount(actual);
  }
}

struct Calib {
  int s_id, bank;
  uint32_t Rfirst, Rsecond;
  vector<uint32_t> open_rows;
};

static vector<Calib> read_calib(const string& path, int wanted_bank) {
  vector<Calib> out;
  ifstream f(path);
  string line;
  while (getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    istringstream iss(line);
    Calib c;
    if (!(iss >> c.s_id >> c.bank >> c.Rfirst >> c.Rsecond)) continue;
    uint32_t v;
    while (iss >> v) c.open_rows.push_back(v);
    if (c.open_rows.size() == 16 && c.bank == wanted_bank) out.push_back(c);
  }
  return out;
}

// Read exactly `n` bytes from stdin or return 0 (EOF).
static bool read_exact(void* buf, size_t n) {
  size_t got = 0;
  char* p = (char*)buf;
  while (got < n) {
    ssize_t r = read(0, p + got, n - got);
    if (r <= 0) return false;
    got += (size_t)r;
  }
  return true;
}

// Per-bank server state: one calibrated MAJ3 tuple + persistent-weight
// backup pool, sized for whatever fraction of (chunk, sign) work units
// this bank will own across the largest expected matmul.
//
// Dual-subarray pool (Optimization B, 2026-05-05): each bank optionally
// holds TWO calibrated tuples from physically distinct subarrays. Per
// round R, the body uses subarray (R & 1) — its (Rfirst, Rsecond,
// open_rows) for the compute, and its own backup_pool for the
// per-column-write scratchpad. This doubles the effective pool size
// for projections like down_proj where n_rounds (108 with N=4 banks)
// exceeds a single-subarray pool (~78 rows). Each subarray's pool is
// still intra-subarray (RowClone is a charge-sharing operation, only
// works within ONE subarray's bitline pairs).
struct BankConfig {
  int bank_id;
  Calib calib;                        // primary (calib_idx=0)
  std::vector<uint32_t> backup_pool;  // primary's safe-zone pool
  size_t pool_cursor = 0;  // legacy LOAD_WEIGHTS reservation; 0 in v2-only mode
  bool dual = false;       // legacy dual-subarray (unused; default false)
  Calib calib_b;
  std::vector<uint32_t> backup_pool_b;
  // For 3-vote majority-correction (D): additional calibs from
  // physically-distinct sub-clusters of the same bank, each with their
  // own backup_pool. Indexed by the per-request `calib_idx` field. Index
  // 0 = the primary `calib` above; indices 1, 2, … pull from this list.
  std::vector<Calib>                cs_extra;
  std::vector<std::vector<uint32_t>> pool_extra;
};

// Round → (subarray_idx, in-pool index). In dual mode, even rounds use
// subarray 0 and odd rounds use subarray 1; each gets ceil(R/2) rows.
// In single mode (dual=false), always returns subarray 0.
static inline int round_to_subarray(const BankConfig& bc, size_t round) {
  return (bc.dual ? (int)(round & 1) : 0);
}
static inline size_t round_to_pool_idx(const BankConfig& bc, size_t round) {
  // Modular wrap when n_rounds exceeds the per-subarray pool size. Each
  // round per-column-rewrites its backup row before use, so cycling
  // through the same pool is safe within a single matmul request — adjacent
  // ACTs occur on the SAME round's row, not on temporally-distant pool
  // entries, so the empirical stride-8 disturb-free property holds.
  size_t base = bc.dual ? (round / 2) : round;
  const auto& pool = (bc.dual && (round & 1)) ? bc.backup_pool_b : bc.backup_pool;
  if (pool.empty()) return 0;  // shouldn't happen — caller validates
  return base % pool.size();
}
static inline const Calib& bc_calib(const BankConfig& bc, size_t round) {
  return (bc.dual && (round & 1)) ? bc.calib_b : bc.calib;
}
static inline const std::vector<uint32_t>& bc_pool(const BankConfig& bc, size_t round) {
  return (bc.dual && (round & 1)) ? bc.backup_pool_b : bc.backup_pool;
}
// D: select calib + pool by calib_idx (0 = primary; 1, 2, ... = cs_extra).
static inline const Calib& bc_calib_idx(const BankConfig& bc, uint32_t idx) {
  if (idx == 0) return bc.calib;
  size_t i = (idx - 1) % bc.cs_extra.size();
  return bc.cs_extra[i];
}
static inline const std::vector<uint32_t>& bc_pool_idx(const BankConfig& bc, uint32_t idx) {
  if (idx == 0) return bc.backup_pool;
  size_t i = (idx - 1) % bc.pool_extra.size();
  return bc.pool_extra[i];
}
// Maximum pool occupancy this set of rounds will require, per subarray.
// Used to validate sizing before doing any work.
static inline size_t bc_max_pool_idx_for(const BankConfig& bc, size_t n_rounds, int sub) {
  // With modular pool indexing (round_to_pool_idx wraps mod pool size),
  // we need at least ONE pool entry — enough to cycle through. The hard
  // requirement was the legacy non-modular behaviour where each round
  // needed a distinct pool slot.
  (void)n_rounds; (void)sub;
  return 1;
}

// One LOAD_WEIGHTS-issued handle: which backup-pool indices were taken
// per bank for this matmul-slice's (chunk, sign) work units. Indexed by
// `round` (= work-unit index / N_banks).
struct LoadedHandle {
  uint32_t handle_id;
  uint32_t n_chunks;
  size_t n_units;        // n_chunks * 2 (signs)
  size_t n_rounds;       // ceil(n_units / N_banks)
  // Per round, per active-bank, the backup row index in that bank's pool.
  // Same shape as the round/bk indexing used in process_matmul.
  // [round][bank_idx] -> backup_pool absolute row index
  std::vector<std::vector<uint32_t>> per_round_backup_rows;
  // Expected popcount per d_out segment for each backup row, captured
  // at LOAD time (before any decay). Indexed [round][bk][seg], 32 bits
  // per segment so popcount ∈ [0, 32]. Used by MM3D start to detect
  // data corruption / decay vs the LOADed contents.
  std::vector<std::vector<std::vector<uint8_t>>> expected_popcounts;
  // Full first row's input mask saved for exact-bit comparison at MM3D
  // verify time. Only round-0 / bank-0..N kept (one row per bank, 8K
  // bytes each) — enough to identify systematic bit corruption.
  std::vector<std::vector<uint32_t>> expected_first_row_mask;  // [bk][seg]
  // Full mask for ALL rounds, saved when PIM_LOAD_REWRITE_ON_MM3D=1.
  // Used to re-write each backup_row at MM3D dispatch to refresh cells
  // to "freshly-written" voltage state (the SiMRA RowClone primitive
  // depends on this; DDR-spec retention is not enough). [round][bk][seg].
  std::vector<std::vector<std::vector<uint32_t>>> all_round_masks;
  // Subarray range to refresh on every MM3D entry. Captured at LOAD
  // time from each bank's calib so we don't have to recompute it.
  std::vector<uint32_t> refresh_row_start;  // per bank
  std::vector<uint32_t> refresh_row_end;    // per bank
};

// Verbosity for data-integrity instrumentation. Set via env
// PIM_VERIFY_LOAD=1 to enable LOAD-time read-back; PIM_VERIFY_MM3D=1
// to enable MM3D-start decay check; PIM_REFRESH=1 to inject the
// looped subarray refresh at MM3D start.
static int g_verify_load = -1;
static int g_verify_mm3d = -1;
static int g_refresh    = -1;
// PIM_INLINE_BITPLANES = K means dispatch K bitplanes per platform.execute,
// chaining K × active_in_round bank bodies in one program. K=1 is the
// historical per-bitplane cadence. Higher K amortises host-FPGA round-trip
// overhead at the cost of more c2h drain per execute (K × N × 8 KB) and
// more instructions per program (~416 / body for the multibank-combined
// MAJ3 body, so K=4 N=4 ≈ 6656 insts — fits the 8192 IMEM after the
// IMEM_ADDR_WIDTH 11→13 bitstream rebuild but blows today's 2048 cap).
//
// Pair with BITSTREAM_IMEM=8192 env at runtime when running on the
// rebuilt bitstream; default 2048 keeps current production behavior.
// g_bitstream_imem mirrors the platform's runtime IMEM ceiling so K-cap
// decisions can be made server-side without touching api/.
static int g_inline_bp  = -1;
static int g_bitstream_imem = -1;
// PIM_PARALLEL_BANKS = 1 swaps build_multibank_combined_program for
// build_multibank_parallel_program: SiMRA doubleACTs (RowClone /
// Broadcast / MAJ3) run as 4-bank parallel pack4 sequences, while
// wrRow / frac / rdRow stay per-bank serial. Default OFF for back-compat.
static int g_parallel_banks = -1;
// PIM_REFRESH_BETWEEN = N: insert SMC_REF in the multibank-combined
// program after every N bank-bodies. Default 0 = no in-program refresh
// (matches today's behaviour: auto-refresh is OFF, only intermittent
// refresh when the host gap allows). Tests if manual refresh between
// bodies (vs the auto-refresh that interleaves into doubleACT and
// disrupts charge-sharing) helps with cumulative ACT-disturb on
// open_rows over many MAJ3 invocations.
static int g_refresh_between = -1;
static int env_flag(const char* name, int dflt) {
  const char* v = getenv(name);
  if (!v || !*v) return dflt;
  return atoi(v);
}
static void init_debug_flags() {
  if (g_verify_load < 0) g_verify_load = env_flag("PIM_VERIFY_LOAD", 0);
  if (g_verify_mm3d < 0) g_verify_mm3d = env_flag("PIM_VERIFY_MM3D", 1);
  if (g_refresh    < 0) g_refresh    = env_flag("PIM_REFRESH",    1);
  if (g_inline_bp  < 0) g_inline_bp  = env_flag("PIM_INLINE_BITPLANES", 1);
  if (g_inline_bp < 1) g_inline_bp = 1;
  if (g_parallel_banks < 0) g_parallel_banks = env_flag("PIM_PARALLEL_BANKS", 0);
  if (g_refresh_between < 0) g_refresh_between = env_flag("PIM_REFRESH_BETWEEN", 0);
  if (g_bitstream_imem < 0) {
    g_bitstream_imem = env_flag("BITSTREAM_IMEM", 2048);
    if (g_bitstream_imem <= 0) g_bitstream_imem = 2048;
    fprintf(stderr, "[server] BITSTREAM_IMEM=%d (set BITSTREAM_IMEM=8192"
                    " on the rebuilt bitstream)\n", g_bitstream_imem);
    // Heuristic K-cap warning. Body sizes (multibank-combined MAJ3
    // body): ~416 inst/bank serial, ~104 inst/bank with PARALLEL_BANKS.
    // Cap so we warn early, before platform.cpp's gate kills the
    // program at execute time.
    int per_body = g_parallel_banks ? 104 : 416;
    int n_banks  = 4;  // production default
    int max_K_fit = g_bitstream_imem / (per_body * n_banks);
    if (max_K_fit < 1) max_K_fit = 1;
    if (g_inline_bp > max_K_fit) {
      fprintf(stderr, "[server] WARN PIM_INLINE_BITPLANES=%d likely won't fit"
                      " IMEM=%d (est. body=%d × banks=%d × K=%d = %d > %d);"
                      " auto-cap is NOT applied — platform gate will skip"
                      " the program. Either lower K or raise BITSTREAM_IMEM.\n",
              g_inline_bp, g_bitstream_imem, per_body, n_banks, g_inline_bp,
              per_body * n_banks * g_inline_bp, g_bitstream_imem);
    }
  }
}

// Allocate per-bank backup rows for a new handle and per-col write the
// supplied weight masks into them. Returns 0 on success, -1 on error
// (e.g., backup pool exhausted).
static int process_load_weights(SoftMCPlatform& platform,
                                 std::vector<BankConfig>& banks,
                                 std::map<uint32_t, LoadedHandle>& handles,
                                 const uint8_t* req, size_t req_len,
                                 int response_fd) {
  if (req_len < 6 * 4) {
    fprintf(stderr, "[server] LOAD_WEIGHTS too small (%zu B)\n", req_len);
    return -1;
  }
  size_t off = 0;
  auto rd_u32 = [&](uint32_t& v) { memcpy(&v, req + off, 4); off += 4; };
  uint32_t magic, handle_id, d_in, d_out, n_chunks;
  rd_u32(magic); rd_u32(handle_id); rd_u32(d_in); rd_u32(d_out); rd_u32(n_chunks);
  if (d_out != 2048) {
    fprintf(stderr, "[server] LOAD_WEIGHTS expects d_out=2048, got %u\n", d_out);
    return -1;
  }
  size_t need = 5 * 4 + (size_t)n_chunks * d_out * 4 * 2;  // pos + neg
  if (req_len < need) {
    fprintf(stderr, "[server] LOAD_WEIGHTS short: need %zu got %zu\n",
            need, req_len);
    return -1;
  }
  const uint32_t* pos_mask_all = (const uint32_t*)(req + off);
  off += (size_t)n_chunks * d_out * 4;
  const uint32_t* neg_mask_all = (const uint32_t*)(req + off);

  const int N = (int)banks.size();
  size_t n_units = (size_t)n_chunks * 2;
  size_t n_rounds = (n_units + N - 1) / N;

  // Check pool space on every bank. We reserve V2_SCRATCH rows at the END
  // of each pool for v2-fallback (per-request) requests so they never
  // collide with handle-allocated rows. Send non-zero ack on exhausted so
  // the client falls back to v2 for this slice.
  // Reduced from 110 → 0: with the new safe-zone backup pool (offset
  // 240, stride 8), pool size is ~50 rows max per bank, so we can't
  // afford to reserve 110 for v2 fallback. v2 path will need to
  // allocate from elsewhere if mixed in. For pure-LOAD usage (which
  // is what the corruption fix is for), no scratch is needed.
  static constexpr size_t V2_SCRATCH = 0;
  for (int bk = 0; bk < N; bk++) {
    size_t needed = banks[bk].pool_cursor + n_rounds + V2_SCRATCH;
    if (needed > banks[bk].backup_pool.size()) {
      fprintf(stderr, "[server] LOAD_WEIGHTS handle=%u: bank %d pool would "
              "overflow (cursor=%zu + n_rounds=%zu + v2_scratch=%zu > %zu) "
              "— sending ENOSPC ack\n",
              handle_id, banks[bk].bank_id,
              banks[bk].pool_cursor, n_rounds, V2_SCRATCH,
              banks[bk].backup_pool.size());
      uint32_t ack = 1;  // non-zero = pool exhausted
      ssize_t w = write(response_fd, &ack, 4);
      (void)w;
      return 0;
    }
  }

  init_debug_flags();

  LoadedHandle h;
  h.handle_id = handle_id;
  h.n_chunks = n_chunks;
  h.n_units = n_units;
  h.n_rounds = n_rounds;
  h.per_round_backup_rows.assign(n_rounds, std::vector<uint32_t>(N, 0));
  h.expected_popcounts.assign(
      n_rounds, std::vector<std::vector<uint8_t>>(N));
  h.expected_first_row_mask.assign(N, std::vector<uint32_t>());
  // Subarray bounds for this handle's banks (used by MM3D refresh).
  h.refresh_row_start.assign(N, 0);
  h.refresh_row_end.assign(N, 0);
  for (int bk = 0; bk < N; bk++) {
    uint32_t any_open = banks[bk].calib.open_rows[0];
    h.refresh_row_start[bk] = (any_open / 640) * 640;
    h.refresh_row_end[bk]   = h.refresh_row_start[bk] + 640;
    // PIM_SUB_START/END override for non-640-aligned subarrays (DIMM 2).
    if (const char* ss = getenv("PIM_SUB_START")) if (*ss) h.refresh_row_start[bk] = (uint32_t)atoi(ss);
    if (const char* se = getenv("PIM_SUB_END"))   if (*se) h.refresh_row_end[bk]   = (uint32_t)atoi(se);
  }

  // Allocate + per-col write each (chunk, sign) unit's mask. Optionally
  // verify by reading the row back and comparing per-segment popcounts
  // to the input mask's popcount.
  std::vector<uint8_t> rb(8192);
  long total_segs = 0, mismatch_segs = 0;
  for (size_t u = 0; u < n_units; u++) {
    uint32_t chunk = (uint32_t)(u / 2);
    int sign = (int)(u % 2);
    const uint32_t* mask = (sign == 0)
        ? pos_mask_all + (size_t)chunk * d_out
        : neg_mask_all + (size_t)chunk * d_out;
    int bk = (int)(u % (size_t)N);
    size_t round = u / (size_t)N;
    uint32_t backup_row = banks[bk].backup_pool[banks[bk].pool_cursor + round];
    per_column_write_row(platform, banks[bk].bank_id, backup_row, mask);
    h.per_round_backup_rows[round][bk] = backup_row;

    // Capture expected popcount per segment from the input mask.
    std::vector<uint8_t> exp_pc(d_out);
    for (uint32_t s = 0; s < d_out; s++) {
      exp_pc[s] = (uint8_t)__builtin_popcount(mask[s]);
    }
    h.expected_popcounts[round][bk] = std::move(exp_pc);
    // Snapshot the full mask for round-0 of each bank — used for
    // exact-bit XOR comparison at MM3D verify time.
    if (round == 0) {
      h.expected_first_row_mask[bk].assign(mask, mask + d_out);
    }
    // PIM_LOAD_REWRITE_ON_MM3D=1 (or PIM_VERIFY_AT_MM3D=1): also keep
    // the FULL mask per (round, bk) so MM3D can re-drive the backup_row
    // before each use, OR so the verify-at-mm3d probe can compare. Costs
    // n_rounds × N × 8KB memory per handle.
    bool keep_masks = false;
    if (getenv("PIM_LOAD_REWRITE_ON_MM3D") &&
        atoi(getenv("PIM_LOAD_REWRITE_ON_MM3D")) > 0) keep_masks = true;
    if (getenv("PIM_VERIFY_AT_MM3D") &&
        atoi(getenv("PIM_VERIFY_AT_MM3D")) > 0) keep_masks = true;
    if (keep_masks) {
      if (h.all_round_masks.empty())
        h.all_round_masks.assign(n_rounds, std::vector<std::vector<uint32_t>>(N));
      h.all_round_masks[round][bk].assign(mask, mask + d_out);
    }

    if (g_verify_load) {
      int rc = read_row_to_buffer(platform, banks[bk].bank_id, backup_row,
                                   rb.data(), 1000000 + (int)u);
      if (rc != 8192) {
        fprintf(stderr, "[load-verify] handle=%u u=%zu rdRow rc=%d\n",
                handle_id, u, rc);
      } else {
        // BYTE-LEVEL verify, not just popcount. Two different 32-bit
        // values can have the same popcount; popcount-verify would
        // miss a write-pattern bug.
        const uint32_t* rbu32 = (const uint32_t*)rb.data();
        int byte_mm = 0, byte_first = -1;
        uint32_t byte_first_exp = 0, byte_first_got = 0;
        for (uint32_t s = 0; s < d_out; s++) {
          if (rbu32[s] != mask[s]) {
            if (byte_first < 0) {
              byte_first = (int)s;
              byte_first_exp = mask[s];
              byte_first_got = rbu32[s];
            }
            byte_mm++;
          }
        }
        if (byte_mm > 0) {
          fprintf(stderr,
              "[load-verify-bytes] handle=%u u=%zu bk=%d row=%u: %d/%u segs "
              "differ at BIT level (first @s=%d exp=0x%08x got=0x%08x xor=0x%08x)\n",
              handle_id, u, banks[bk].bank_id, backup_row,
              byte_mm, d_out, byte_first, byte_first_exp, byte_first_got,
              byte_first_exp ^ byte_first_got);
        }
        std::vector<int> got_pc(d_out);
        segment_popcount(rb.data(), got_pc.data(), (int)d_out);
        int mm = 0, mm_first = -1;
        for (uint32_t s = 0; s < d_out; s++) {
          if ((uint8_t)got_pc[s] != h.expected_popcounts[round][bk][s]) {
            if (mm_first < 0) mm_first = (int)s;
            mm++;
          }
        }
        total_segs += d_out;
        mismatch_segs += mm;
        if (mm > 0 && (u < 4 || u % 32 == 0)) {
          fprintf(stderr,
              "[load-verify] handle=%u u=%zu bk=%d row=%u: %d/%u segs "
              "differ (first @s=%d exp=%u got=%d)\n",
              handle_id, u, banks[bk].bank_id, backup_row,
              mm, d_out, mm_first,
              h.expected_popcounts[round][bk][mm_first],
              got_pc[mm_first]);
        }
      }
    }
  }
  // Commit the cursors.
  for (int bk = 0; bk < N; bk++) banks[bk].pool_cursor += n_rounds;

  handles[handle_id] = std::move(h);

  if (g_verify_load) {
    fprintf(stderr,
        "[load-verify] handle=%u write-readback summary: "
        "%ld/%ld segs mismatched (%.4f%%)\n",
        handle_id, mismatch_segs, total_segs,
        total_segs ? 100.0 * mismatch_segs / total_segs : 0.0);
  }

  // Acknowledge with a 4-byte status (0 = OK).
  uint32_t ack = 0;
  if (write(response_fd, &ack, 4) != 4) {
    fprintf(stderr, "[server] LOAD_WEIGHTS ack write failed\n");
    return -1;
  }
  fprintf(stderr, "[server] LOAD_WEIGHTS handle=%u n_chunks=%u rounds=%zu "
          "pool_cursor[0]=%zu (verify_load=%d)\n",
          handle_id, n_chunks, n_rounds, banks[0].pool_cursor, g_verify_load);
  return 0;
}

// Run a matmul using a previously-loaded handle's backup rows; identical to
// process_request's inner loop but skips the per-col writes.
static int process_matmul_handle(SoftMCPlatform& platform,
                                  std::vector<BankConfig>& banks,
                                  const std::map<uint32_t, LoadedHandle>& handles,
                                  const uint8_t* req, size_t req_len,
                                  int& label_base, int response_fd);

// Process one request body. Distributes (chunk, sign) work units
// round-robin across `banks` (1..N). For N>1, each platform.execute()
// runs N banks' MAJ3 bodies in one program; receiveData() is called
// N times after each execute, in the same order banks were emitted.
//
// Persistent weights: each work unit's mask is per-col written to its
// owning bank's backup row ONCE (outer setup loop), then re-used via
// fast RowClone in every bitplane's combined program (inner loop).
static int process_request(SoftMCPlatform& platform,
                            std::vector<BankConfig>& banks,
                            const uint8_t* req, size_t req_len,
                            int& label_base, int response_fd) {
  init_debug_flags();
  if (banks.empty()) {
    fprintf(stderr, "[server] no banks configured\n");
    return -1;
  }
  if (req_len < 5 * 4) {
    fprintf(stderr, "[server] request too small (%zu B)\n", req_len);
    return -1;
  }
  size_t off = 0;
  auto rd_u32 = [&](uint32_t& v) {
    memcpy(&v, req + off, 4); off += 4;
  };
  uint32_t magic, d_in, d_out, n_chunks, n_bitplanes;
  rd_u32(magic); rd_u32(d_in); rd_u32(d_out);
  rd_u32(n_chunks); rd_u32(n_bitplanes);
  if (magic != MAGIC_V2) {
    fprintf(stderr, "[server] bad magic 0x%x\n", magic);
    return -1;
  }
  if (d_out != 2048) {
    fprintf(stderr, "[server] expected d_out=2048, got %u\n", d_out);
    return -1;
  }
  // D: optional calib_idx. The legacy V2 body had 5 header fields (20
  // bytes); clients that want cross-calib voting append a 6th u32 at
  // the END of the body (after pos_mask + neg_mask + x_bitplane + bp_factor).
  // We detect by total size and read it from the tail without advancing
  // the in-band parse offset.
  size_t need_no_idx = (size_t)5*4 + (size_t)n_chunks*d_out*4*2
                     + (size_t)n_chunks*n_bitplanes*4 + (size_t)n_bitplanes*4;
  size_t need_with_idx = need_no_idx + 4;
  uint32_t calib_idx = 0;
  if (req_len >= need_with_idx) {
    memcpy(&calib_idx, req + need_no_idx, 4);
  } else if (req_len < need_no_idx) {
    fprintf(stderr, "[server] short request: need %zu got %zu\n",
            need_no_idx, req_len);
    return -1;
  }

  // Slice into views.
  const uint32_t* pos_mask_all = (const uint32_t*)(req + off);
  off += (size_t)n_chunks * d_out * 4;
  const uint32_t* neg_mask_all = (const uint32_t*)(req + off);
  off += (size_t)n_chunks * d_out * 4;
  const uint32_t* x_bitplane_all = (const uint32_t*)(req + off);
  off += (size_t)n_chunks * n_bitplanes * 4;
  const int32_t*  bitplane_factor = (const int32_t*)(req + off);

  const int N = (int)banks.size();
  const size_t n_units = (size_t)n_chunks * 2;        // (chunk, sign) pairs
  const size_t n_rounds = (n_units + N - 1) / N;       // # of N-bank executes per bitplane

  // Per-request silicon-side timing (server-internal profile).
  using clk = std::chrono::steady_clock;
  using ns_t = std::chrono::nanoseconds;
  auto t_req_start = clk::now();
  long long t_wcol_ns = 0, t_exec_ns = 0, t_recv_ns = 0, t_pop_ns = 0;
  int n_wcol_execs = 0, n_maj3_execs = 0;

  // BACKUP POOL REMOVED. Weights are written directly to each bank's
  // calib.Rfirst (no separate backup row, no RowClone). Matches the
  // historical subprocess `bitnet-proj-exe` scheme.

  // Iteration order chosen to PRESERVE write-then-use locality (every
  // backup row is read by RowClone within ~milliseconds of its per-col
  // write, same as the pre-multibank single-bank server). Without this
  // locality the model produces nonsense — see the back-compat test
  // failure 2026-05-04 where batching ALL writes upfront broke q_proj.
  //
  //   for round = 0..n_rounds:
  //     write up to N backups (one per bank, this round's work units)
  //     for bitplane = 0..n_bitplanes:
  //       multibank execute with all N banks' MAJ3 bodies
  //       receive + popcount + accumulate per bank
  vector<int32_t> y(d_out, 0);
  for (size_t round = 0; round < n_rounds; round++) {
    // 1. Per-col write each active bank's backup row for this round.
    int active_in_round = 0;
    for (int bk = 0; bk < N; bk++) {
      size_t u = round * (size_t)N + (size_t)bk;
      if (u >= n_units) break;
      uint32_t chunk = (uint32_t)(u / 2);
      int sign = (int)(u % 2);
      const uint32_t* mask = (sign == 0)
          ? pos_mask_all + (size_t)chunk * d_out
          : neg_mask_all + (size_t)chunk * d_out;
      // PROVEN multi-bank scheme: per-round pool slot, spread across the
      // safe-zone backup pool so each backup row is touched only once per
      // request. Modular wrap absorbs n_rounds > pool_size. MM3D body
      // then RowClones scratch→Rfirst. With D-mode (calib_idx > 0) we
      // pull from the cs_extra/pool_extra entries — different open_rows
      // → different cells → different cell-flake noise sample for voting.
      const std::vector<uint32_t>& pool_for_round =
          bc_pool_idx(banks[bk], calib_idx);
      size_t pool_idx = round_to_pool_idx(banks[bk], round);
      if (pool_idx >= pool_for_round.size())
        pool_idx %= pool_for_round.size();
      uint32_t scratch_row = pool_for_round[pool_idx];
      auto t0 = clk::now();
      per_column_write_row(platform, banks[bk].bank_id, scratch_row, mask);
      t_wcol_ns += std::chrono::duration_cast<ns_t>(clk::now() - t0).count();
      n_wcol_execs += 3;  // per_column_write_row issues 3 platform.execute calls
      active_in_round++;
    }
    if (active_in_round == 0) break;

    // 2. Bitplane dispatch — chunked by g_inline_bp (= PIM_INLINE_BITPLANES).
    // K=1 reproduces the historical per-bitplane cadence; K>1 chains
    // K × active_in_round bank bodies into one program, doing a single
    // platform.execute + single receiveData per chunk. Each extra bitplane
    // amortises one host-FPGA round-trip (~30 µs) at the cost of K× more
    // c2h drain per execute (K × N × 8 KB).
    for (uint32_t bp_start = 0; bp_start < n_bitplanes;
         bp_start += (uint32_t)g_inline_bp) {
      uint32_t K = std::min((uint32_t)g_inline_bp, n_bitplanes - bp_start);
      size_t   M = (size_t)K * (size_t)active_in_round;
      std::vector<int>             ex_bank_ids;
      std::vector<uint32_t>        ex_backup_rows;
      std::vector<uint32_t>        ex_Rfirsts;
      std::vector<uint32_t>        ex_Rseconds;
      std::vector<const uint32_t*> ex_open_rows;
      std::vector<uint32_t>        ex_x_patterns;
      std::vector<int>             ex_signs;
      ex_bank_ids.reserve(M);
      ex_backup_rows.reserve(M);
      ex_Rfirsts.reserve(M);
      ex_Rseconds.reserve(M);
      ex_open_rows.reserve(M);
      ex_x_patterns.reserve(M);
      ex_signs.reserve(M);

      // Order in the program (and thus in the c2h drain) is bp-major:
      // bp_start/bk0, bp_start/bk1, ..., bp_start+K-1/bk{N-1}.
      // Each bank picks the round-appropriate calib + pool slot via the
      // dual-subarray helpers (single-subarray mode = subarray 0 always).
      for (uint32_t kp = 0; kp < K; kp++) {
        uint32_t b = bp_start + kp;
        for (int bk = 0; bk < active_in_round; bk++) {
          size_t u = round * (size_t)N + (size_t)bk;
          uint32_t chunk = (uint32_t)(u / 2);
          int sign = (int)(u % 2);
          uint32_t xb = x_bitplane_all[(size_t)chunk * n_bitplanes + b];
          // D: choose calib + pool by request's calib_idx.
          const Calib& c = bc_calib_idx(banks[bk], calib_idx);
          const std::vector<uint32_t>& pool = bc_pool_idx(banks[bk], calib_idx);
          size_t pool_idx_mm3d = round_to_pool_idx(banks[bk], round);
          if (pool_idx_mm3d >= pool.size()) pool_idx_mm3d %= pool.size();
          uint32_t scratch_row = pool[pool_idx_mm3d];
          ex_bank_ids.push_back(banks[bk].bank_id);
          ex_backup_rows.push_back(scratch_row);
          ex_Rfirsts.push_back(c.Rfirst);
          ex_Rseconds.push_back(c.Rsecond);
          ex_open_rows.push_back(c.open_rows.data());
          ex_x_patterns.push_back(xb);
          ex_signs.push_back(sign);
        }
      }

      Program p = (g_parallel_banks
          ? build_multibank_parallel_program
          : build_multibank_combined_program)(
              ex_bank_ids, ex_backup_rows, ex_Rfirsts, ex_Rseconds,
              ex_open_rows, ex_x_patterns, label_base);
      label_base += 2000 * (int)M + 1000;
      // PIM_DUMP_PROGRAM=1: one-shot dump of the first MM3D u64 inst stream
      // to /tmp/dump_program_par<P>_n<M>.txt for diff between single-bank
      // serial (P=0, M=1) and multi-bank parallel (P=1, M=4).
      static bool s_dumped = false;
      if (!s_dumped && getenv("PIM_DUMP_PROGRAM")) {
        s_dumped = true;
        char path[256];
        snprintf(path, sizeof(path), "/tmp/dump_program_par%d_n%zu.txt",
                 g_parallel_banks ? 1 : 0, M);
        FILE* fp = fopen(path, "w");
        if (fp) {
          uint64_t* iseq = (uint64_t*)p.get_inst_array();
          int n_inst = p.size() / 8;
          fprintf(fp, "# program u64 inst stream: parallel=%d M=%zu n_inst=%d\n",
                  g_parallel_banks ? 1 : 0, M, n_inst);
          fprintf(fp, "# bank_ids:");
          for (size_t b = 0; b < M; b++) fprintf(fp, " %d", ex_bank_ids[b]);
          fprintf(fp, "\n");
          for (int i = 0; i < n_inst; i++)
            fprintf(fp, "[%4d]  %016lx\n", i, (unsigned long)iseq[i]);
          fclose(fp);
          fprintf(stderr, "[dump] wrote %d insts to %s\n", n_inst, path);
          free(iseq);
        }
      }
      auto t_exec0 = clk::now();
      platform.execute(p);
      t_exec_ns += std::chrono::duration_cast<ns_t>(clk::now() - t_exec0).count();
      n_maj3_execs++;

      static thread_local std::vector<uint8_t> rows_buf;
      size_t total_bytes = M * 8192u;
      if (rows_buf.size() < total_bytes) rows_buf.resize(total_bytes);
      auto t_recv0 = clk::now();
      int rc = platform.receiveData(rows_buf.data(), (int)total_bytes);
      t_recv_ns += std::chrono::duration_cast<ns_t>(clk::now() - t_recv0).count();
      if (rc != (int)total_bytes) {
        fprintf(stderr, "[server] receiveData rc=%d expected=%zu "
                "(round=%zu bp=[%u..%u))\n", rc, total_bytes,
                round, bp_start, bp_start + K);
        return -1;
      }
      auto t_pop0 = clk::now();
      for (uint32_t kp = 0; kp < K; kp++) {
        uint32_t b = bp_start + kp;
        for (int bk = 0; bk < active_in_round; bk++) {
          size_t idx = (size_t)kp * (size_t)active_in_round + (size_t)bk;
          const uint8_t* row = rows_buf.data() + idx * 8192u;
          vector<int> pc(d_out);
          segment_popcount(row, pc.data(), (int)d_out);
          int sign_factor = (ex_signs[idx] == 0) ? +1 : -1;
          int weight = sign_factor * bitplane_factor[b];
          for (uint32_t j = 0; j < d_out; j++) y[j] += weight * pc[j];
          if (getenv("PIM_DEBUG_RX")) {
            fprintf(stderr,
                "[srv-rx] round=%zu bp=%u bk=%d sign=%d weight=%d "
                "row[0..15]=%02x%02x%02x%02x %02x%02x%02x%02x "
                "%02x%02x%02x%02x %02x%02x%02x%02x  "
                "pc[0..7]=%d %d %d %d %d %d %d %d  "
                "y[0..7]=%d %d %d %d %d %d %d %d\n",
                round, b, banks[bk].bank_id, sign_factor, weight,
                row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7],
                row[8],row[9],row[10],row[11],row[12],row[13],row[14],row[15],
                pc[0],pc[1],pc[2],pc[3],pc[4],pc[5],pc[6],pc[7],
                y[0],y[1],y[2],y[3],y[4],y[5],y[6],y[7]);
          }
        }
      }
      t_pop_ns += std::chrono::duration_cast<ns_t>(clk::now() - t_pop0).count();
    }
  }

  long long t_total_ns = std::chrono::duration_cast<ns_t>(
      clk::now() - t_req_start).count();
  // Print server-side per-request timing on stderr (every 50th request to
  // avoid noise; first few always).
  static int s_req_n = 0;
  s_req_n++;
  if (s_req_n <= 5 || s_req_n % 50 == 0) {
    long long unaccounted = t_total_ns - t_wcol_ns - t_exec_ns - t_recv_ns - t_pop_ns;
    fprintf(stderr,
        "[srv-prof #%d] total=%.1fms wcol=%.1fms (%dx) exec=%.1fms (%dx) "
        "recv=%.1fms pop=%.1fms other=%.1fms\n",
        s_req_n, t_total_ns/1e6, t_wcol_ns/1e6, n_wcol_execs,
        t_exec_ns/1e6, n_maj3_execs,
        t_recv_ns/1e6, t_pop_ns/1e6, unaccounted/1e6);
  }

  // Write 8192 byte (= int32 × 2048) response on the saved response_fd
  // (NOT FD 1, which we permanently redirected to stderr).
  ssize_t total = (ssize_t)d_out * 4;
  ssize_t written = 0;
  while (written < total) {
    ssize_t w = write(response_fd, ((char*)y.data()) + written, total - written);
    if (w <= 0) {
      fprintf(stderr, "[server] write failed: %s\n", strerror(errno));
      return -1;
    }
    written += w;
  }
  return 0;
}

// Run a matmul using a previously-loaded handle's backup rows. Skips the
// per-col writes since the weights are already in DRAM.
static int process_matmul_handle(SoftMCPlatform& platform,
                                  std::vector<BankConfig>& banks,
                                  const std::map<uint32_t, LoadedHandle>& handles,
                                  const uint8_t* req, size_t req_len,
                                  int& label_base, int response_fd) {
  init_debug_flags();
  if (req_len < 5 * 4) {
    fprintf(stderr, "[server] MATMUL_HANDLE too small (%zu B)\n", req_len);
    return -1;
  }
  size_t off = 0;
  auto rd_u32 = [&](uint32_t& v) { memcpy(&v, req + off, 4); off += 4; };
  uint32_t magic, handle_id, d_out, n_chunks, n_bitplanes;
  rd_u32(magic); rd_u32(handle_id); rd_u32(d_out);
  rd_u32(n_chunks); rd_u32(n_bitplanes);
  if (d_out != 2048) {
    fprintf(stderr, "[server] MATMUL_HANDLE expects d_out=2048, got %u\n", d_out);
    return -1;
  }
  auto it = handles.find(handle_id);
  if (it == handles.end()) {
    fprintf(stderr, "[server] MATMUL_HANDLE unknown handle %u\n", handle_id);
    return -1;
  }
  const LoadedHandle& h = it->second;
  if (h.n_chunks != n_chunks) {
    fprintf(stderr, "[server] MATMUL_HANDLE n_chunks mismatch: handle has %u, request has %u\n",
            h.n_chunks, n_chunks);
    return -1;
  }
  size_t need = 5 * 4 + (size_t)n_chunks * n_bitplanes * 4
              + (size_t)n_bitplanes * 4;
  if (req_len < need) {
    fprintf(stderr, "[server] MATMUL_HANDLE short: need %zu got %zu\n",
            need, req_len);
    return -1;
  }
  const uint32_t* x_bitplane_all = (const uint32_t*)(req + off);
  off += (size_t)n_chunks * n_bitplanes * 4;
  const int32_t*  bitplane_factor = (const int32_t*)(req + off);

  const int N = (int)banks.size();
  const size_t n_units = h.n_units;
  const size_t n_rounds = h.n_rounds;

  using clk = std::chrono::steady_clock;
  using ns_t = std::chrono::nanoseconds;
  auto t_req_start = clk::now();
  long long t_exec_ns = 0, t_recv_ns = 0, t_pop_ns = 0;
  long long t_refresh_ns = 0, t_verify_ns = 0;
  int n_maj3_execs = 0;

  init_debug_flags();

  // Refresh ALL handles' subarrays before doing any MM3D work. With
  // auto-refresh disabled, weights loaded by an earlier LOAD_WEIGHTS
  // sit cold in DRAM during subsequent matmuls — refresh resets the
  // 64 ms retention clock for all of them. Use the per-bank subarray
  // ranges captured at LOAD time; these cover every backup row used.
  if (g_refresh && !handles.empty()) {
    auto t0 = clk::now();
    // Union of all handles' subarray ranges (per bank). For each bank,
    // take the min(start) and max(end) across all loaded handles.
    std::vector<int>      ref_bank_ids;
    std::vector<uint32_t> ref_starts;
    std::vector<uint32_t> ref_ends;
    for (int bk = 0; bk < N; bk++) {
      uint32_t mn = 0xFFFFFFFFu, mx = 0;
      for (const auto& kv : handles) {
        const LoadedHandle& lh = kv.second;
        if (bk < (int)lh.refresh_row_start.size()) {
          if (lh.refresh_row_start[bk] < mn) mn = lh.refresh_row_start[bk];
          if (lh.refresh_row_end[bk]   > mx) mx = lh.refresh_row_end[bk];
        }
      }
      if (mn < mx) {
        ref_bank_ids.push_back(banks[bk].bank_id);
        ref_starts.push_back(mn);
        ref_ends.push_back(mx);
      }
    }
    if (!ref_bank_ids.empty()) {
      Program rp = build_refresh_subarray_loop_program(
          ref_bank_ids, ref_starts, ref_ends);
      platform.execute(rp);
    }
    t_refresh_ns = std::chrono::duration_cast<ns_t>(clk::now() - t0).count();
  }

  // Decay/corruption check: read back the FIRST round's first row of
  // each bank for THIS handle and compare popcounts to LOAD-time
  // expected. If the popcount diverges, the data has been corrupted
  // or has decayed between LOAD and MM3D — pinpoints whether the bug
  // is on the storage side or the compute side.
  if (g_verify_mm3d && h.n_rounds > 0) {
    auto t0 = clk::now();
    std::vector<uint8_t> rb(8192);
    long total_segs = 0, mismatch_segs = 0;
    int first_mm_bk = -1, first_mm_seg = -1;
    uint32_t first_exp_word = 0, first_got_word = 0;
    int first_exp_pc = -1, first_got_pc = -1;
    uint32_t first_or = 0, first_xor = 0;
    std::vector<uint8_t> rb2(8192);
    for (int bk = 0; bk < N; bk++) {
      if ((size_t)bk >= h.per_round_backup_rows[0].size()) break;
      uint32_t row = h.per_round_backup_rows[0][bk];
      int rc = read_row_to_buffer(platform, banks[bk].bank_id, row,
                                   rb.data(), 2000000 + (int)handle_id * 100 + bk);
      if (rc != 8192) {
        fprintf(stderr, "[mm3d-verify] handle=%u bk=%d rdRow rc=%d\n",
                handle_id, banks[bk].bank_id, rc);
        continue;
      }
      // Read the SAME row again to test read stability — if rb != rb2,
      // the read itself is unstable / cells are flaky.
      int rc2 = read_row_to_buffer(platform, banks[bk].bank_id, row,
                                    rb2.data(),
                                    3000000 + (int)handle_id * 100 + bk);
      if (rc2 == 8192) {
        long diff = 0;
        for (int i = 0; i < 8192; i++)
          if (rb[i] != rb2[i]) diff++;
        if (diff > 0) {
          fprintf(stderr,
              "[mm3d-verify-stab] handle=%u bk=%d row=%u read1≠read2: "
              "%ld bytes differ (read instability!)\n",
              handle_id, banks[bk].bank_id, row, diff);
        }
      }
      std::vector<int> got_pc(d_out);
      segment_popcount(rb.data(), got_pc.data(), (int)d_out);
      const auto& exp_pc = h.expected_popcounts[0][bk];
      const auto& exp_mask = h.expected_first_row_mask[bk];
      // Per-bit OR/XOR accumulator over ALL mismatched segments —
      // shows which bit positions are systematically being flipped.
      uint32_t bit_or_acc = 0, bit_xor_acc = 0;
      for (uint32_t s = 0; s < d_out; s++) {
        if ((uint8_t)got_pc[s] != exp_pc[s]) {
          uint32_t got_w = (uint32_t)rb[s*4]
                         | ((uint32_t)rb[s*4+1] << 8)
                         | ((uint32_t)rb[s*4+2] << 16)
                         | ((uint32_t)rb[s*4+3] << 24);
          uint32_t exp_w = (s < exp_mask.size()) ? exp_mask[s] : 0u;
          uint32_t flipped = got_w ^ exp_w;
          uint32_t set_bits = got_w & ~exp_w;  // bits 0→1
          bit_xor_acc |= flipped;
          bit_or_acc  |= set_bits;
          if (first_mm_bk < 0) {
            first_mm_bk = bk;
            first_mm_seg = (int)s;
            first_exp_pc = exp_pc[s]; first_got_pc = got_pc[s];
            first_got_word = got_w;
            first_or = exp_w;        // re-use field for exp_word
            first_xor = flipped;
          }
          mismatch_segs++;
        }
      }
      total_segs += d_out;
      if (bit_xor_acc) {
        fprintf(stderr,
            "[mm3d-verify-bits] handle=%u bk=%d "
            "OR_set_bits_in_mismatch=0x%08x XOR_flipped=0x%08x\n",
            handle_id, banks[bk].bank_id, bit_or_acc, bit_xor_acc);
      }
    }
    t_verify_ns = std::chrono::duration_cast<ns_t>(clk::now() - t0).count();
    if (mismatch_segs > 0) {
      fprintf(stderr,
          "[mm3d-verify] handle=%u DECAY/CORRUPTION: %ld/%ld segs "
          "differ in round-0 (%.4f%%); first @bk=%d s=%d "
          "exp_pc=%d got_pc=%d exp_word=0x%08x got_word=0x%08x "
          "xor=0x%08x (refresh=%d)\n",
          handle_id, mismatch_segs, total_segs,
          100.0 * mismatch_segs / total_segs,
          first_mm_bk, first_mm_seg,
          first_exp_pc, first_got_pc, first_or, first_got_word,
          first_xor, g_refresh);
    } else {
      static int s_clean = 0;
      s_clean++;
      if (s_clean <= 3 || s_clean % 50 == 0) {
        fprintf(stderr,
            "[mm3d-verify] handle=%u round-0 popcounts OK "
            "(%ld segs, refresh=%d)\n",
            handle_id, total_segs, g_refresh);
      }
    }
  }

  // PIM_LOAD_REWRITE_ON_MM3D=1: confirming fix for the LOAD-vs-V2
  // corruption — re-write each backup_row IMMEDIATELY before its MM3D
  // round dispatch (inside the round loop below) to restore
  // "freshly-written" cell voltage that SiMRA's RowClone needs.
  // (Up-front re-write for all rounds wouldn't help: pool[0] becomes
  // stale by the time round 15's MM3D fires.)
  bool g_load_rewrite = (getenv("PIM_LOAD_REWRITE_ON_MM3D") &&
                          atoi(getenv("PIM_LOAD_REWRITE_ON_MM3D")) > 0 &&
                          !h.all_round_masks.empty());

  vector<int32_t> y(d_out, 0);
  for (size_t round = 0; round < n_rounds; round++) {
    int active_in_round = 0;
    for (int bk = 0; bk < N; bk++) {
      size_t u = round * (size_t)N + (size_t)bk;
      if (u >= n_units) break;
      active_in_round++;
    }
    if (active_in_round == 0) break;

    // PIM_VERIFY_AT_MM3D=1: before the rewrite-or-no-rewrite branch,
    // read EVERY pool entry (all rounds × all banks) back and compare
    // against its stored mask. This tells us whether the bug is
    //  - corruption that happens at round 0's MM3D (then pool[≥1] reads
    //    clean at round 0, dirty at round 1+)
    //  - corruption that already happened before any MM3D
    //    (then pool[≥1] reads dirty even at round 0).
    if (getenv("PIM_VERIFY_AT_MM3D") &&
        atoi(getenv("PIM_VERIFY_AT_MM3D")) > 0) {
      if (!h.all_round_masks.empty()) {
        std::vector<uint8_t> rb(8192);
        for (size_t r2 = 0; r2 < h.all_round_masks.size(); r2++) {
          for (int bk = 0; bk < active_in_round; bk++) {
            if ((size_t)bk >= h.all_round_masks[r2].size()) continue;
            const auto& m = h.all_round_masks[r2][bk];
            if (m.empty()) continue;
            if ((size_t)bk >= h.per_round_backup_rows[r2].size()) continue;
            uint32_t br = h.per_round_backup_rows[r2][bk];
            int rc = read_row_to_buffer(platform, banks[bk].bank_id, br,
                                         rb.data(),
                                         5000000 + (int)round * 1000
                                                  + (int)r2 * 10 + bk);
            if (rc == 8192) {
              const uint32_t* rbu32 = (const uint32_t*)rb.data();
              int byte_mm = 0; int first = -1;
              uint32_t exp = 0, got = 0;
              for (size_t s = 0; s < m.size(); s++) {
                if (rbu32[s] != m[s]) {
                  if (first < 0) { first = (int)s; exp = m[s]; got = rbu32[s]; }
                  byte_mm++;
                }
              }
              fprintf(stderr,
                  "[verify-at-mm3d] handle=%u at-round=%zu probed-pool=%zu bk=%d row=%u: "
                  "%d/%zu segs differ "
                  "(first @s=%d exp=0x%08x got=0x%08x xor=0x%08x)\n",
                  handle_id, round, r2, banks[bk].bank_id, br,
                  byte_mm, m.size(), first, exp, got, exp ^ got);
            } else {
              fprintf(stderr, "[verify-at-mm3d] rdRow rc=%d\n", rc);
            }
          }
        }
      } else {
        fprintf(stderr, "[verify-at-mm3d] all_round_masks empty — "
                "set PIM_VERIFY_AT_MM3D=1 at LOAD time to populate.\n");
      }
    }

    // Per-round re-write of this round's backup_rows IMMEDIATELY before
    // its MM3D dispatches. This is the confirming fix for SiMRA RowClone
    // freshness dependency.
    if (g_load_rewrite) {
      for (int bk = 0; bk < active_in_round; bk++) {
        if ((size_t)bk >= h.all_round_masks[round].size()) continue;
        const auto& m = h.all_round_masks[round][bk];
        if (m.empty()) continue;
        uint32_t br = h.per_round_backup_rows[round][bk];
        // PIM_REWRITE_DUMMY_ROW=1: write to an UNRELATED row (not the
        // backup_row that MM3D will RowClone from) with dummy data.
        // If this still fixes the bug, the rewrite's side-effect (PHY
        // / controller / register state) is the cure — NOT the
        // freshness of pool[X]'s cells.
        const char* dum = getenv("PIM_REWRITE_DUMMY_ROW");
        const char* act_only = getenv("PIM_REWRITE_ACT_ONLY");
        if (dum && atoi(dum) > 0) {
          uint32_t sub_start = (banks[bk].calib.open_rows[0] / 640) * 640;
          uint32_t dummy_row = sub_start + 100;
          static std::vector<uint32_t> dummy_mask(2048, 0xDEADBEEFu);
          per_column_write_row(platform, banks[bk].bank_id, dummy_row,
                                dummy_mask.data());
        } else if (act_only && atoi(act_only) > 0) {
          // Just ACT(pool[X])-PRE — no WRITEs. If THIS fixes the bug,
          // the cure is just "recent ACT", not the full WRITE drive.
          // If it DOESN'T, WRITE-class drive is what's needed.
          Program p_act;
          p_act.add_inst(SMC_LI(banks[bk].bank_id, BAR));
          p_act.add_inst(SMC_LI(br, RAR));
          p_act.add_below(ACT(BAR, 0, RAR, 0));
          p_act.add_inst(SMC_SLEEP(8));
          p_act.add_below(PRE(BAR, 0, 0));
          p_act.add_inst(SMC_END());
          platform.execute(p_act);
        } else {
          per_column_write_row(platform, banks[bk].bank_id, br, m.data());
        }
      }
      // PIM_REWRITE_DELAY_US: artificial delay between re-write and MM3D
      // dispatch — used to characterise the cell-voltage decay window.
      const char* dly = getenv("PIM_REWRITE_DELAY_US");
      if (dly) {
        int us = atoi(dly);
        if (us > 0) usleep(us);
      }
      // PIM_REWRITE_DISTURB_ACTS: between re-write of backup_row[round]
      // and its MM3D dispatch, fire K extra ACT-PRE pulses on rows
      // ADJACENT to backup_row in the same subarray. This probes whether
      // adjacent-row activations DISTURB the just-re-written backup_row,
      // which is the disturbance theory's prediction for why production
      // LOAD's many-adjacent-pool-ACTs pattern fails the bug.
      const char* dist = getenv("PIM_REWRITE_DISTURB_ACTS");
      if (dist) {
        int n_acts = atoi(dist);
        if (n_acts > 0) {
          for (int bk = 0; bk < active_in_round; bk++) {
            uint32_t br = h.per_round_backup_rows[round][bk];
            Program disturb;
            disturb.add_inst(SMC_LI(banks[bk].bank_id, BAR));
            for (int k = 0; k < n_acts; k++) {
              uint32_t aggressor = br + (k % 2 == 0 ? (uint32_t)(k+1) : (uint32_t)-(int32_t)(k+1));
              disturb.add_inst(SMC_LI(aggressor, RAR));
              disturb.add_below(ACT(BAR, 0, RAR, 0));
              disturb.add_below(PRE(BAR, 0, 0));
            }
            disturb.add_inst(SMC_END());
            platform.execute(disturb);
          }
        }
      }
      // PIM_REWRITE_DISTURB_WRITES=K: after the rewrite, do K full
      // per_column_write_rows on OTHER pool entries — to inject
      // WRITE-class disturbance (the qualitatively-stronger class that
      // ACT-PREs alone don't reproduce). This mimics what the
      // production LOAD path does between LOAD-time-write and use:
      // 15+ per_column_writes to neighboring pool entries, each = 128
      // WRITEs. If this BREAKS the rewrite fix, WRITE-class disturbance
      // is confirmed as the bug's mechanism.
      const char* dist_w = getenv("PIM_REWRITE_DISTURB_WRITES");
      if (dist_w) {
        int n_writes = atoi(dist_w);
        if (n_writes > 0) {
          // Use a dummy mask (zeros are fine for disturbance — what
          // matters is that the rows get hammered with WRITEs).
          static std::vector<uint32_t> dummy_mask(2048, 0xCAFEBABEu);
          for (int bk = 0; bk < active_in_round; bk++) {
            uint32_t br = h.per_round_backup_rows[round][bk];
            for (int k = 0; k < n_writes; k++) {
              // Pick a neighbor row — back up_row ± (k+1). Stays in same
              // subarray for k small enough.
              uint32_t target = br + (k % 2 == 0 ? (uint32_t)(k+1) : (uint32_t)-(int32_t)(k+1));
              per_column_write_row(platform, banks[bk].bank_id, target,
                                    dummy_mask.data());
            }
          }
        }
      }
    }

    // Bitplane dispatch — chunked by g_inline_bp; see process_request for
    // the matching v2-path comment.
    for (uint32_t bp_start = 0; bp_start < n_bitplanes;
         bp_start += (uint32_t)g_inline_bp) {
      uint32_t K = std::min((uint32_t)g_inline_bp, n_bitplanes - bp_start);
      size_t   M = (size_t)K * (size_t)active_in_round;
      std::vector<int>             ex_bank_ids;
      std::vector<uint32_t>        ex_backup_rows;
      std::vector<uint32_t>        ex_Rfirsts;
      std::vector<uint32_t>        ex_Rseconds;
      std::vector<const uint32_t*> ex_open_rows;
      std::vector<uint32_t>        ex_x_patterns;
      std::vector<int>             ex_signs;
      ex_bank_ids.reserve(M);
      ex_backup_rows.reserve(M);
      ex_Rfirsts.reserve(M);
      ex_Rseconds.reserve(M);
      ex_open_rows.reserve(M);
      ex_x_patterns.reserve(M);
      ex_signs.reserve(M);
      for (uint32_t kp = 0; kp < K; kp++) {
        uint32_t b = bp_start + kp;
        for (int bk = 0; bk < active_in_round; bk++) {
          size_t u = round * (size_t)N + (size_t)bk;
          uint32_t chunk = (uint32_t)(u / 2);
          int sign = (int)(u % 2);
          uint32_t xb = x_bitplane_all[(size_t)chunk * n_bitplanes + b];
          ex_bank_ids.push_back(banks[bk].bank_id);
          ex_backup_rows.push_back(h.per_round_backup_rows[round][bk]);
          ex_Rfirsts.push_back(banks[bk].calib.Rfirst);
          ex_Rseconds.push_back(banks[bk].calib.Rsecond);
          ex_open_rows.push_back(banks[bk].calib.open_rows.data());
          ex_x_patterns.push_back(xb);
          ex_signs.push_back(sign);
        }
      }
      if (getenv("PIM_DEBUG_RX")) {
        for (size_t i = 0; i < ex_backup_rows.size(); i++) {
          fprintf(stderr, "[mm3d-build] round=%zu bp=%u idx=%zu bk=%d backup_row=%u sign=%d\n",
                  round, bp_start, i, ex_bank_ids[i], ex_backup_rows[i], ex_signs[i]);
        }
      }
      Program p = (g_parallel_banks
          ? build_multibank_parallel_program
          : build_multibank_combined_program)(
              ex_bank_ids, ex_backup_rows, ex_Rfirsts, ex_Rseconds,
              ex_open_rows, ex_x_patterns, label_base);
      label_base += 2000 * (int)M + 1000;
      // PIM_DUMP_MM3D_PROGRAMS=N: dump the first N MM3D-handle programs
      // u64 inst stream for diffing across rounds.
      static int s_mm3d_dumped = 0;
      const char* mm3d_dump_n = getenv("PIM_DUMP_MM3D_PROGRAMS");
      int max_dumps = mm3d_dump_n ? atoi(mm3d_dump_n) : 0;
      if (s_mm3d_dumped < max_dumps) {
        s_mm3d_dumped++;
        char path[256];
        snprintf(path, sizeof(path), "/tmp/mm3d_program_dump_%d.txt", s_mm3d_dumped);
        FILE* fp = fopen(path, "w");
        if (fp) {
          uint64_t* iseq = (uint64_t*)p.get_inst_array();
          int n_inst = p.size() / 8;
          fprintf(fp, "# MM3D handle program: round=%zu bp=%u backup_row=%u\n",
                  round, bp_start, ex_backup_rows[0]);
          for (int i = 0; i < n_inst; i++)
            fprintf(fp, "[%4d]  %016lx\n", i, (unsigned long)iseq[i]);
          fclose(fp);
          free(iseq);
          fprintf(stderr, "[mm3d-dump #%d] round=%zu wrote %d insts to %s\n",
                  s_mm3d_dumped, round, n_inst, path);
        }
      }
      auto t_exec0 = clk::now();
      platform.execute(p);
      t_exec_ns += std::chrono::duration_cast<ns_t>(clk::now() - t_exec0).count();
      n_maj3_execs++;

      static thread_local std::vector<uint8_t> rows_buf;
      size_t total_bytes = M * 8192u;
      if (rows_buf.size() < total_bytes) rows_buf.resize(total_bytes);
      auto t_recv0 = clk::now();
      int rc = platform.receiveData(rows_buf.data(), (int)total_bytes);
      t_recv_ns += std::chrono::duration_cast<ns_t>(clk::now() - t_recv0).count();
      if (rc != (int)total_bytes) {
        fprintf(stderr, "[server] MM3D receiveData rc=%d expected=%zu "
                "(round=%zu bp=[%u..%u))\n", rc, total_bytes,
                round, bp_start, bp_start + K);
        return -1;
      }
      auto t_pop0 = clk::now();
      for (uint32_t kp = 0; kp < K; kp++) {
        uint32_t b = bp_start + kp;
        for (int bk = 0; bk < active_in_round; bk++) {
          size_t idx = (size_t)kp * (size_t)active_in_round + (size_t)bk;
          const uint8_t* row = rows_buf.data() + idx * 8192u;
          vector<int> pc(d_out);
          segment_popcount(row, pc.data(), (int)d_out);
          int sign_factor = (ex_signs[idx] == 0) ? +1 : -1;
          int weight = sign_factor * bitplane_factor[b];
          for (uint32_t j = 0; j < d_out; j++) y[j] += weight * pc[j];
          if (getenv("PIM_DEBUG_RX")) {
            fprintf(stderr,
                "[srv-rx-mm3d] round=%zu bp=%u bk=%d sign=%d weight=%d "
                "row[0..15]=%02x%02x%02x%02x %02x%02x%02x%02x "
                "%02x%02x%02x%02x %02x%02x%02x%02x  "
                "pc[0..7]=%d %d %d %d %d %d %d %d  "
                "y[0..7]=%d %d %d %d %d %d %d %d\n",
                round, b, banks[bk].bank_id, sign_factor, weight,
                row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7],
                row[8],row[9],row[10],row[11],row[12],row[13],row[14],row[15],
                pc[0],pc[1],pc[2],pc[3],pc[4],pc[5],pc[6],pc[7],
                y[0],y[1],y[2],y[3],y[4],y[5],y[6],y[7]);
          }
        }
      }
      t_pop_ns += std::chrono::duration_cast<ns_t>(clk::now() - t_pop0).count();

      // PIM_VERIFY_AT_MM3D=1 also probes ALL pool entries AFTER each
      // bitplane execution, to catch the exact moment pool[1] gets
      // corrupted within round 0's MM3D body.
      if (getenv("PIM_VERIFY_AT_MM3D") &&
          atoi(getenv("PIM_VERIFY_AT_MM3D")) > 0 &&
          !h.all_round_masks.empty()) {
        std::vector<uint8_t> rb_post(8192);
        for (size_t r2 = 0; r2 < h.all_round_masks.size(); r2++) {
          for (int bk = 0; bk < active_in_round; bk++) {
            if ((size_t)bk >= h.all_round_masks[r2].size()) continue;
            const auto& m = h.all_round_masks[r2][bk];
            if (m.empty()) continue;
            uint32_t br = h.per_round_backup_rows[r2][bk];
            int rc = read_row_to_buffer(platform, banks[bk].bank_id, br,
                                         rb_post.data(),
                                         6000000 + (int)round * 1000
                                                  + (int)r2 * 10 + bk);
            if (rc == 8192) {
              const uint32_t* rbu32 = (const uint32_t*)rb_post.data();
              int byte_mm = 0;
              for (size_t s = 0; s < m.size(); s++) {
                if (rbu32[s] != m[s]) byte_mm++;
              }
              uint32_t got0 = (m.empty() ? 0 : rbu32[0]);
              uint32_t got1 = (m.size() < 2 ? 0 : rbu32[1]);
              uint32_t got2 = (m.size() < 3 ? 0 : rbu32[2]);
              uint32_t got3 = (m.size() < 4 ? 0 : rbu32[3]);
              fprintf(stderr,
                  "[verify-post-mm3d] handle=%u after-round=%zu bp=%u "
                  "probed-pool=%zu bk=%d row=%u: %d/%zu segs differ "
                  "got[0..3]=0x%08x 0x%08x 0x%08x 0x%08x\n",
                  handle_id, round, bp_start, r2,
                  banks[bk].bank_id, br, byte_mm, m.size(),
                  got0, got1, got2, got3);
            }
          }
        }
      }
    }
  }

  // POST-MM3D verify: re-read first round's first row of each bank
  // immediately after the MM3D work. If this shows corruption that
  // wasn't visible at the START verify, the MM3D itself corrupted the
  // data. Compare to expected_first_row_mask for exact bit pattern.
  if (g_verify_mm3d && h.n_rounds > 0) {
    std::vector<uint8_t> rb(8192);
    long total_segs = 0, mismatch_segs = 0;
    int first_bk = -1, first_seg = -1;
    uint32_t first_exp_w = 0, first_got_w = 0;
    for (int bk = 0; bk < N; bk++) {
      if ((size_t)bk >= h.per_round_backup_rows[0].size()) break;
      uint32_t row = h.per_round_backup_rows[0][bk];
      int rc = read_row_to_buffer(platform, banks[bk].bank_id, row,
                                   rb.data(),
                                   4000000 + (int)handle_id * 100 + bk);
      if (rc != 8192) continue;
      std::vector<int> got_pc(d_out);
      segment_popcount(rb.data(), got_pc.data(), (int)d_out);
      const auto& exp_pc  = h.expected_popcounts[0][bk];
      const auto& exp_msk = h.expected_first_row_mask[bk];
      for (uint32_t s = 0; s < d_out; s++) {
        if ((uint8_t)got_pc[s] != exp_pc[s]) {
          if (first_bk < 0) {
            first_bk = bk; first_seg = (int)s;
            first_got_w = (uint32_t)rb[s*4]
                        | ((uint32_t)rb[s*4+1] << 8)
                        | ((uint32_t)rb[s*4+2] << 16)
                        | ((uint32_t)rb[s*4+3] << 24);
            first_exp_w = (s < exp_msk.size()) ? exp_msk[s] : 0u;
          }
          mismatch_segs++;
        }
      }
      total_segs += d_out;
    }
    if (mismatch_segs > 0) {
      static int s_post_n = 0;
      s_post_n++;
      if (s_post_n <= 5 || s_post_n % 20 == 0) {
        fprintf(stderr,
            "[mm3d-verify-post] handle=%u POST-MM3D CORRUPTION: %ld/%ld "
            "segs (%.2f%%); @bk=%d s=%d exp=0x%08x got=0x%08x "
            "xor=0x%08x\n",
            handle_id, mismatch_segs, total_segs,
            100.0 * mismatch_segs / total_segs,
            first_bk, first_seg, first_exp_w, first_got_w,
            first_exp_w ^ first_got_w);
      }
    } else {
      static int s_post_clean = 0;
      s_post_clean++;
      if (s_post_clean <= 5 || s_post_clean % 50 == 0) {
        fprintf(stderr,
            "[mm3d-verify-post] handle=%u POST-MM3D clean (%ld segs)\n",
            handle_id, total_segs);
      }
    }
  }

  long long t_total_ns = std::chrono::duration_cast<ns_t>(
      clk::now() - t_req_start).count();
  static int s_mh_n = 0;
  s_mh_n++;
  if (s_mh_n <= 5 || s_mh_n % 50 == 0) {
    long long unaccounted = t_total_ns - t_exec_ns - t_recv_ns - t_pop_ns
                          - t_refresh_ns - t_verify_ns;
    fprintf(stderr,
        "[mm3d-prof #%d handle=%u] total=%.1fms refresh=%.1fms verify=%.1fms "
        "exec=%.1fms (%dx) recv=%.1fms pop=%.1fms other=%.1fms\n",
        s_mh_n, handle_id, t_total_ns/1e6,
        t_refresh_ns/1e6, t_verify_ns/1e6,
        t_exec_ns/1e6, n_maj3_execs, t_recv_ns/1e6, t_pop_ns/1e6,
        unaccounted/1e6);
  }

  ssize_t total = (ssize_t)d_out * 4;
  ssize_t written = 0;
  while (written < total) {
    ssize_t w = write(response_fd, ((char*)y.data()) + written, total - written);
    if (w <= 0) {
      fprintf(stderr, "[server] MM3D write failed: %s\n", strerror(errno));
      return -1;
    }
    written += w;
  }
  return 0;
}

// Parse "0,1,2,3" or "1" → vector<int>{0,1,2,3} / {1}. Returns empty on
// parse error. Caps at 8 banks (more would overflow program buffer).
static std::vector<int> parse_bank_arg(const std::string& s) {
  std::vector<int> out;
  std::string cur;
  for (size_t i = 0; i <= s.size(); i++) {
    char ch = (i < s.size()) ? s[i] : ',';
    if (ch == ',') {
      if (cur.empty()) { out.clear(); return out; }
      try {
        int v = std::stoi(cur);
        if (v < 0 || v > 15) { out.clear(); return out; }
        out.push_back(v);
      } catch (...) { out.clear(); return out; }
      cur.clear();
    } else {
      cur += ch;
    }
  }
  if (out.size() > 8) out.clear();
  // Reject duplicate banks.
  std::set<int> seen(out.begin(), out.end());
  if (seen.size() != out.size()) out.clear();
  return out;
}

// Build the persistent-weight backup pool for one bank's calibrated
// tuple. Pool = rows in the same 640-aligned subarray as the tuple's
// open_rows[0], excluding any row in the 16-row open set.
//
// CRITICAL: Empirical adjacent-row-disturb finding (test_multibank_disturb
// 2026-05-05): backup rows that are too close to the calibrated open_rows
// or to other backup rows get bit-corrupted by the MM3D body's RowClone
// + broadcast operations on the open_rows. With OFFSET=240 and STRIDE=8
// (= start 240 rows past the open_rows cluster, take every 8th non-open
// row), 40 backup rows survive 40 rounds × 8 bitplanes per MM3D
// undisturbed on all 4 banks of DIMM 0. Below this spacing, pool[0]
// gets bit-flipped after just 1 round of operations on pool[1+].
//
// Env knobs let us tune without recompiling: PIM_POOL_OFFSET, PIM_POOL_STRIDE.
// If PIM_POOL_LIST_FILE is set, read pool rows from that file (one row per
// line, # = comment) — bypasses stride-based selection. Used for the
// fault-aware layout produced by the per-row fault sweep.
static std::vector<uint32_t> build_backup_pool(const Calib& c) {
  if (const char* path_pat = getenv("PIM_POOL_LIST_FILE")) {
    // {bank} token substitution → per-bank layouts. e.g.
    // /path/pool_layout_bank{bank}.txt resolves to ..._bank0.txt for
    // bank 0, ..._bank1.txt for bank 1, etc.  If the substituted file
    // doesn't exist, fall back to the un-substituted path (single shared
    // layout across banks — current behavior).
    std::string path_str = path_pat;
    std::string pat = "{bank}";
    size_t pos = path_str.find(pat);
    if (pos != std::string::npos) {
      char buf[16];
      snprintf(buf, sizeof(buf), "%d", c.bank);
      path_str.replace(pos, pat.size(), buf);
    }
    const char* path = path_str.c_str();
    std::vector<uint32_t> pool;
    FILE* fp = fopen(path, "r");
    if (!fp) {
      fprintf(stderr, "[backup_pool] PIM_POOL_LIST_FILE='%s' unreadable, "
              "falling through to stride-based\n", path);
    } else {
      char line[64];
      while (fgets(line, sizeof(line), fp)) {
        char* s = line;
        while (*s == ' ' || *s == '\t') s++;
        if (*s == '#' || *s == '\n' || *s == 0) continue;
        uint32_t r = (uint32_t)atoi(s);
        if (r > 0) pool.push_back(r);
      }
      fclose(fp);
      // Sub-array sanity: filter to this calib's safe zone. The
      // (any_open/640)*640 default only aligns when the subarray starts
      // on a 640-row boundary (true for DIMM 0 s_id 77, NOT for DIMM 2
      // s_id 72 which starts at 45312). Override with PIM_SUB_START /
      // PIM_SUB_END to point at the real subarray range from
      // FindOpenRows/dimm<N>/selected_subarrays.txt.
      uint32_t any_open = c.open_rows[0];
      uint32_t sub_start = (any_open / 640) * 640;
      uint32_t sub_end   = sub_start + 640;
      if (const char* ss = getenv("PIM_SUB_START")) if (*ss) sub_start = (uint32_t)atoi(ss);
      if (const char* se = getenv("PIM_SUB_END"))   if (*se) sub_end   = (uint32_t)atoi(se);
      std::set<uint32_t> open_set(c.open_rows.begin(), c.open_rows.end());
      std::vector<uint32_t> filtered;
      for (uint32_t r : pool) {
        if (r < sub_start || r >= sub_end) continue;
        if (open_set.count(r)) continue;
        filtered.push_back(r);
      }
      fprintf(stderr, "[backup_pool] loaded %zu rows from '%s' "
              "(filtered to %zu in safe zone [%u, %u))\n",
              pool.size(), path, filtered.size(), sub_start, sub_end);
      if (!filtered.empty()) return filtered;
    }
  }
  std::set<uint32_t> open_set(c.open_rows.begin(), c.open_rows.end());
  uint32_t any_open = c.open_rows[0];
  uint32_t subarray_start = (any_open / 640) * 640;
  int offset = atoi(getenv("PIM_POOL_OFFSET")
                    ? getenv("PIM_POOL_OFFSET") : "240");
  // Default stride=16 (NOT 8). With stride=8, pool[k+1] - pool[k] = 8,
  // which lands consecutive pool entries inside the same DDR4 sub-array
  // mat group. SiMRA's RowClone (doubleACT t_12=30, t_23=1) of any pool
  // entry physically replicates that row's bits into the row at +8 in
  // the same mat group — destroying pool[k+1]. Empirical sweep on this
  // calibration: stride=8 → ~33-45% match (catastrophic); stride=14, 16
  // → 100%. See bitnet_load_weights_corruption.md.
  int stride = atoi(getenv("PIM_POOL_STRIDE")
                    ? getenv("PIM_POOL_STRIDE") : "16");
  // PIM_POOL_RANGE_BLOCKS = N → scan N consecutive 640-row blocks starting
  // at this subarray. Default 1 = legacy single-block behaviour. N>1 is
  // needed for single-bank (N_banks=1) runs where n_rounds = n_units = 160+
  // and one block's ~50-row stride-8 pool isn't enough. The s_id=77 calib
  // tuple's open_rows already cross the 640 boundary, so a physical
  // subarray is at least 1280 rows on this DIMM.
  int range_blocks = atoi(getenv("PIM_POOL_RANGE_BLOCKS")
                          ? getenv("PIM_POOL_RANGE_BLOCKS") : "1");
  if (range_blocks < 1) range_blocks = 1;
  if (stride < 1) stride = 1;
  uint32_t start = subarray_start + (uint32_t)offset;
  if (start >= subarray_start + 640) start = subarray_start;
  uint32_t end = subarray_start + (uint32_t)(640 * range_blocks);
  std::vector<uint32_t> pool;
  int idx = 0;
  for (uint32_t r = start; r < end && pool.size() < 500; r++) {
    if (open_set.find(r) != open_set.end()) continue;
    if ((idx++) % stride != 0) continue;
    pool.push_back(r);
  }
  return pool;
}

int main(int argc, char** argv) {
  if (argc != 4) {
    fprintf(stderr,
        "Usage: %s <bender_id> <calib_file> <bank_arg>\n"
        "  bank_arg: \"1\" (single bank) or \"0,1,2,3\" (multi-bank Path C)\n",
        argv[0]);
    return 1;
  }
  int bender_id = atoi(argv[1]);
  string calib_p = argv[2];
  std::string bank_arg = argv[3];

  std::vector<int> wanted_banks = parse_bank_arg(bank_arg);
  if (wanted_banks.empty()) {
    fprintf(stderr, "[server] could not parse bank_arg '%s' "
            "(want \"1\" or e.g. \"0,1,2,3\", banks 0..15 unique)\n",
            bank_arg.c_str());
    return 2;
  }

  // Load calibrated tuples per requested bank. PIM_DUAL_SUBARRAY=1
  // would let each bank use TWO calibrated subarrays alternating per
  // round, doubling the effective backup pool to fit larger projections
  // (e.g. down_proj's n_rounds=108 with N=4 banks).
  // **CURRENTLY DEFAULT OFF** because alternating compute subarrays per
  // round empirically produces wrong PIM output (verified 2026-05-05:
  // layer-0-only test gives correct ',' from CPU vs wrong 'L' under
  // dual mode, despite the per-round dispatch logic being self-consistent
  // by code review). Likely a physical interaction (row-decoder timing,
  // bank-state leak, cross-subarray disturb) that needs more investigation.
  // Until debugged, ship single-subarray + skip down_proj for correct
  // output. See task #75.
  bool dual_mode = atoi(getenv("PIM_DUAL_SUBARRAY")
                       ? getenv("PIM_DUAL_SUBARRAY") : "0") != 0;
  std::vector<BankConfig> banks;
  for (int bk : wanted_banks) {
    vector<Calib> cs = read_calib(calib_p, bk);
    if (cs.empty()) {
      fprintf(stderr, "[server] no calib for bank %d in %s\n",
              bk, calib_p.c_str());
      return 2;
    }
    BankConfig bc;
    bc.bank_id = bk;
    bc.calib = cs[0];
    bc.backup_pool = build_backup_pool(bc.calib);
    if (bc.backup_pool.empty()) {
      fprintf(stderr, "[server] empty backup pool for bank %d\n", bk);
      return 2;
    }
    // D-mode extras: pick distinct sub-clusters, ranked by population
    // (denser cluster = more tuples passed calibration screening = lower
    // expected per-cell flake noise). Skip sparse clusters (<10 tuples)
    // entirely — including them in the 3-vote median actually hurts
    // accuracy (median of {strong, weak, weak} biases toward weak).
    {
      // Count tuples per sub-cluster.
      std::map<uint32_t, size_t> cluster_count;
      for (const auto& c : cs) {
        uint32_t sub = (c.open_rows[0] / 640) * 640;
        cluster_count[sub]++;
      }
      uint32_t sub0 = (cs[0].open_rows[0] / 640) * 640;
      // Build list of (count, sub_start) for non-primary clusters with
      // ≥10 tuples; sort descending by count.
      std::vector<std::pair<size_t,uint32_t>> ranked;
      for (auto& kv : cluster_count) {
        if (kv.first == sub0) continue;
        if (kv.second < 10) continue;
        ranked.emplace_back(kv.second, kv.first);
      }
      std::sort(ranked.begin(), ranked.end(),
                [](const auto& a, const auto& b){ return a.first > b.first; });
      for (auto& [cnt, sub] : ranked) {
        if (bc.cs_extra.size() >= 4) break;
        // Pick the first calib in cs whose open_rows[0] falls in this sub.
        for (const auto& c : cs) {
          uint32_t s = (c.open_rows[0] / 640) * 640;
          if (s != sub) continue;
          std::vector<uint32_t> pool_i = build_backup_pool(c);
          if (pool_i.empty()) break;
          bc.cs_extra.push_back(c);
          bc.pool_extra.push_back(std::move(pool_i));
          break;
        }
      }
      fprintf(stderr, "[server] bank %d: %zu extra calibs (dense clusters)",
              bk, bc.cs_extra.size());
      for (const auto& c : bc.cs_extra)
        fprintf(stderr, " sub=%u", (c.open_rows[0] / 640));
      fprintf(stderr, "\n");
    }
    // Legacy dual-subarray (kept default OFF; same code path as D's first
    // extra calib but with the "alternate per round" semantics, never
    // validated to give correct output).
    if (dual_mode && cs.size() >= 2) {
      uint32_t sub0_start = (cs[0].open_rows[0] / 640) * 640;
      for (size_t i = 1; i < cs.size(); i++) {
        uint32_t subi_start = (cs[i].open_rows[0] / 640) * 640;
        if (subi_start != sub0_start) {
          bc.calib_b = cs[i];
          bc.backup_pool_b = build_backup_pool(bc.calib_b);
          if (!bc.backup_pool_b.empty()) {
            bc.dual = true;
          }
          break;
        }
      }
      if (!bc.dual) {
        fprintf(stderr, "[server] bank %d: no second-subarray calib found "
                "(have %zu calibs, all in same subarray as cs[0])\n",
                bk, cs.size());
      }
    }
    banks.push_back(std::move(bc));
  }

  // Critical: SoftMCPlatform::init(), reset_fpga(), and any library code
  // can write to STDOUT via std::cout. Stdout is our binary response
  // channel — any text written there corrupts the response.
  // Robust fix: SAVE original stdout to a high fd (response_fd), then
  // PERMANENTLY redirect FD 1 to stderr. All library prints go to
  // stderr; binary responses go to response_fd.
  fflush(stdout);
  int response_fd = dup(STDOUT_FILENO);
  if (response_fd < 0) {
    fprintf(stderr, "[server] dup(stdout) failed\n"); return 3;
  }
  if (dup2(STDERR_FILENO, STDOUT_FILENO) < 0) {
    fprintf(stderr, "[server] dup2(stderr→stdout) failed\n"); return 3;
  }

  // PIM_BACKEND=sim: run programs against the in-process behavioral
  // SimDramModel (deterministic, no cell flake) instead of the FPGA.
  // Useful for fast-iteration LLM coherence checking without burning
  // FPGA wall time. Default = real FPGA.
  bool sim_backend = (getenv("PIM_BACKEND") &&
                      std::string(getenv("PIM_BACKEND")) == "sim");
  std::unique_ptr<SoftMCPlatform> platform_owner;
  if (sim_backend) {
    auto sp = std::make_unique<SimPlatform>();
    sp->load_calib(calib_p);
    fprintf(stderr, "[server] PIM_BACKEND=sim — using in-process SimDramModel\n");
    platform_owner = std::move(sp);
  } else {
    platform_owner = std::make_unique<SoftMCPlatform>(bender_id);
    if (platform_owner->init() != SOFTMC_SUCCESS) {
      fprintf(stderr, "[server] platform init failed\n"); return 3;
    }
    platform_owner->reset_fpga();
  }
  SoftMCPlatform& platform = *platform_owner;
  // NOTE: platform.set_aref(true) breaks MAJ3 reliability — auto-refresh
  // commands interleave with the precision-timed doubleACT operations and
  // disrupt the charge-sharing physics. Keep it OFF. Manual REF placement
  // (SMC_REF in our programs at safe points) is the only correct path.
  std::cout.flush();
  fflush(stdout);

  fprintf(stderr, "[server] ready: bender=%d N_banks=%zu "
          "(response_fd=%d)\n",
          bender_id, banks.size(), response_fd);
  for (size_t i = 0; i < banks.size(); i++) {
    fprintf(stderr, "[server]   bank %d: s_id=%d Rfirst=%u Rsecond=%u "
            "backup_pool=%zu rows starting at %u\n",
            banks[i].bank_id, banks[i].calib.s_id,
            banks[i].calib.Rfirst, banks[i].calib.Rsecond,
            banks[i].backup_pool.size(), banks[i].backup_pool[0]);
    if (banks[i].dual) {
      fprintf(stderr, "[server]   bank %d (dual): s_id=%d Rfirst=%u "
              "Rsecond=%u backup_pool_b=%zu rows starting at %u "
              "(combined effective pool=%zu)\n",
              banks[i].bank_id, banks[i].calib_b.s_id,
              banks[i].calib_b.Rfirst, banks[i].calib_b.Rsecond,
              banks[i].backup_pool_b.size(),
              banks[i].backup_pool_b[0],
              banks[i].backup_pool.size() + banks[i].backup_pool_b.size());
    }
  }

  vector<uint8_t> req_buf;
  int label_base = 0;
  std::map<uint32_t, LoadedHandle> handles;
  while (true) {
    uint32_t req_len = 0;
    if (!read_exact(&req_len, 4)) {
      fprintf(stderr, "[server] EOF on stdin, exiting\n");
      break;
    }
    if (req_len == 0) {
      fprintf(stderr, "[server] quit sentinel received\n");
      break;
    }
    if (req_len > 64u * 1024u * 1024u) {  // 64 MB sanity cap
      fprintf(stderr, "[server] request too large: %u\n", req_len);
      return 4;
    }
    req_buf.resize(req_len);
    if (!read_exact(req_buf.data(), req_len)) {
      fprintf(stderr, "[server] short read of request body\n");
      return 5;
    }
    // Dispatch by magic in the first 4 bytes.
    if (req_len < 4) {
      fprintf(stderr, "[server] runt request: %u bytes\n", req_len);
      return 6;
    }
    uint32_t magic;
    memcpy(&magic, req_buf.data(), 4);
    int rc;
    if (magic == MAGIC_V2) {
      rc = process_request(platform, banks,
                           req_buf.data(), req_len, label_base, response_fd);
    } else if (magic == MAGIC_LOAD) {
      rc = process_load_weights(platform, banks, handles,
                                req_buf.data(), req_len, response_fd);
    } else if (magic == MAGIC_MM3D) {
      rc = process_matmul_handle(platform, banks, handles,
                                  req_buf.data(), req_len, label_base,
                                  response_fd);
    } else {
      fprintf(stderr, "[server] unknown magic 0x%x\n", magic);
      return 6;
    }
    if (rc != 0) return 6;
  }
  _exit(0);
}
