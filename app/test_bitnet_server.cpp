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
#include "../util.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
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

  // 1. RowClone backup → Rfirst (refresh weight from backup).
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
  // (Per-row handle refresh is done OUT-OF-BAND in the server's
  // process_matmul_handle, not inside this program — refresh requires
  // ACT+PRE on each backup row, which is too many instructions to fit
  // in this combined program's buffer.)
  // emit_bank_combined_body now ends with SiMRA's standard post-rdRow
  // pattern (NOPs + PRE + NOPs), so consecutive bank bodies are safe
  // to chain back-to-back without inter-bank padding.
  for (size_t i = 0; i < bank_ids.size(); i++) {
    emit_bank_combined_body(p, bank_ids[i], backup_rows[i],
                            Rfirsts[i], Rseconds[i],
                            open_rows_list[i], x_patterns[i],
                            label_seed + (int)i * 2000);
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
  Calib calib;
  std::vector<uint32_t> backup_pool;  // backup rows in this bank's subarray
  size_t pool_cursor = 0;  // next free index in backup_pool for handle allocation
  // Optional second subarray (only populated when dual mode is enabled).
  bool dual = false;
  Calib calib_b;
  std::vector<uint32_t> backup_pool_b;
};

// Round → (subarray_idx, in-pool index). In dual mode, even rounds use
// subarray 0 and odd rounds use subarray 1; each gets ceil(R/2) rows.
// In single mode (dual=false), always returns subarray 0.
static inline int round_to_subarray(const BankConfig& bc, size_t round) {
  return (bc.dual ? (int)(round & 1) : 0);
}
static inline size_t round_to_pool_idx(const BankConfig& bc, size_t round) {
  return (bc.dual ? (round / 2) : round);
}
static inline const Calib& bc_calib(const BankConfig& bc, size_t round) {
  return (bc.dual && (round & 1)) ? bc.calib_b : bc.calib;
}
static inline const std::vector<uint32_t>& bc_pool(const BankConfig& bc, size_t round) {
  return (bc.dual && (round & 1)) ? bc.backup_pool_b : bc.backup_pool;
}
// Maximum pool occupancy this set of rounds will require, per subarray.
// Used to validate sizing before doing any work.
static inline size_t bc_max_pool_idx_for(const BankConfig& bc, size_t n_rounds, int sub) {
  if (!bc.dual) return n_rounds;
  // Subarray 0 hosts rounds 0,2,4,...; subarray 1 hosts 1,3,5,...
  // Need at least ceil((n_rounds - sub) / 2) entries in subarray `sub`.
  if (n_rounds <= (size_t)sub) return 0;
  return (n_rounds - sub + 1) / 2;
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
// more instructions per program (~120 / body, so K=2 N=4 ≈ 960 insts —
// still fits today's 2048 IMEM; K=8 N=4 ≈ 3840 insts — needs the
// IMEM_ADDR_WIDTH 11→13 bitstream rebuild).
static int g_inline_bp  = -1;
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

    if (g_verify_load) {
      int rc = read_row_to_buffer(platform, banks[bk].bank_id, backup_row,
                                   rb.data(), 1000000 + (int)u);
      if (rc != 8192) {
        fprintf(stderr, "[load-verify] handle=%u u=%zu rdRow rc=%d\n",
                handle_id, u, rc);
      } else {
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
  size_t need = (size_t)5*4 + (size_t)n_chunks*d_out*4*2
              + (size_t)n_chunks*n_bitplanes*4 + (size_t)n_bitplanes*4;
  if (req_len < need) {
    fprintf(stderr, "[server] short request: need %zu got %zu\n", need, req_len);
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

  // Each bank needs enough backup rows in each subarray it uses.
  // In single-subarray mode: subarray 0 holds all n_rounds rows.
  // In dual-subarray mode: subarray 0 holds even rounds, subarray 1 holds odd rounds.
  for (int bk = 0; bk < N; bk++) {
    size_t need_a = bc_max_pool_idx_for(banks[bk], n_rounds, /*sub=*/0);
    size_t need_b = bc_max_pool_idx_for(banks[bk], n_rounds, /*sub=*/1);
    if (banks[bk].backup_pool.size() < need_a) {
      fprintf(stderr, "[server] bank %d subarray-0 backup pool too small: have %zu, "
              "need %zu (n_units=%zu, n_rounds=%zu, N=%d, dual=%d)\n",
              banks[bk].bank_id, banks[bk].backup_pool.size(),
              need_a, n_units, n_rounds, N, banks[bk].dual);
      return -1;
    }
    if (banks[bk].dual && banks[bk].backup_pool_b.size() < need_b) {
      fprintf(stderr, "[server] bank %d subarray-1 backup pool too small: have %zu, "
              "need %zu (n_units=%zu, n_rounds=%zu, N=%d)\n",
              banks[bk].bank_id, banks[bk].backup_pool_b.size(),
              need_b, n_units, n_rounds, N);
      return -1;
    }
  }

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
      // v2 must NOT collide with rows already taken by LOAD_WEIGHTS.
      // Use slots starting at pool_cursor (= first free slot above any
      // loaded handle data).
      // Dual-subarray pool: round R picks subarray (R&1); within that
      // subarray, the (R/2)-th pool slot. Single-subarray mode uses
      // subarray 0 + pool slot R. pool_cursor reserves the first
      // `pool_cursor` slots of subarray 0 for any LOAD_WEIGHTS handles.
      const std::vector<uint32_t>& pool_for_round = bc_pool(banks[bk], round);
      size_t pool_idx = round_to_pool_idx(banks[bk], round);
      // pool_cursor only applies to subarray 0 (where handles allocate from).
      if (round_to_subarray(banks[bk], round) == 0)
        pool_idx += banks[bk].pool_cursor;
      uint32_t backup_row = pool_for_round[pool_idx];
      auto t0 = clk::now();
      per_column_write_row(platform, banks[bk].bank_id, backup_row, mask);
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
          const Calib& c = bc_calib(banks[bk], round);
          const std::vector<uint32_t>& pool_for_round = bc_pool(banks[bk], round);
          size_t pool_idx = round_to_pool_idx(banks[bk], round);
          if (round_to_subarray(banks[bk], round) == 0)
            pool_idx += banks[bk].pool_cursor;
          ex_bank_ids.push_back(banks[bk].bank_id);
          ex_backup_rows.push_back(pool_for_round[pool_idx]);
          ex_Rfirsts.push_back(c.Rfirst);
          ex_Rseconds.push_back(c.Rsecond);
          ex_open_rows.push_back(c.open_rows.data());
          ex_x_patterns.push_back(xb);
          ex_signs.push_back(sign);
        }
      }

      Program p = build_multibank_combined_program(
          ex_bank_ids, ex_backup_rows, ex_Rfirsts, ex_Rseconds,
          ex_open_rows, ex_x_patterns, label_base);
      label_base += 2000 * (int)M + 1000;
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

  vector<int32_t> y(d_out, 0);
  for (size_t round = 0; round < n_rounds; round++) {
    int active_in_round = 0;
    for (int bk = 0; bk < N; bk++) {
      size_t u = round * (size_t)N + (size_t)bk;
      if (u >= n_units) break;
      active_in_round++;
    }
    if (active_in_round == 0) break;

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
      Program p = build_multibank_combined_program(
          ex_bank_ids, ex_backup_rows, ex_Rfirsts, ex_Rseconds,
          ex_open_rows, ex_x_patterns, label_base);
      label_base += 2000 * (int)M + 1000;
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
        }
      }
      t_pop_ns += std::chrono::duration_cast<ns_t>(clk::now() - t_pop0).count();
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
static std::vector<uint32_t> build_backup_pool(const Calib& c) {
  std::set<uint32_t> open_set(c.open_rows.begin(), c.open_rows.end());
  uint32_t any_open = c.open_rows[0];
  uint32_t subarray_start = (any_open / 640) * 640;
  int offset = atoi(getenv("PIM_POOL_OFFSET")
                    ? getenv("PIM_POOL_OFFSET") : "240");
  int stride = atoi(getenv("PIM_POOL_STRIDE")
                    ? getenv("PIM_POOL_STRIDE") : "8");
  if (stride < 1) stride = 1;
  uint32_t start = subarray_start + (uint32_t)offset;
  if (start >= subarray_start + 640) start = subarray_start;
  std::vector<uint32_t> pool;
  int idx = 0;
  for (uint32_t r = start;
       r < subarray_start + 640 && pool.size() < 500; r++) {
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
    // Pick a second calib whose subarray_start differs from cs[0]'s.
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

  SoftMCPlatform platform(bender_id);
  if (platform.init() != SOFTMC_SUCCESS) {
    fprintf(stderr, "[server] platform init failed\n"); return 3;
  }
  platform.reset_fpga();
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
