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
static constexpr uint32_t MAGIC_V2G = 0xB17EF005u;    // V2 with per-group partial response
static constexpr uint32_t MAGIC_V2S = 0xB17EF006u;    // V2 single-track (2026-07-21):
// payload carries the POS masks only; every unit is sign 0 (n_units ==
// n_chunks). For 1-bit models (neg == ~pos) the client reconstructs
// y = 2*y_pos - sum_b fac_b*popcount(x_b) host-side — halves the per-
// request DRAM work (scratch writes + MAJ3 bodies + drains).
static constexpr uint32_t MAGIC_V2GS = 0xB17EF007u;   // V2 grouped + single
// (2026-07-21 night): the request-batching lever. V2G (per-group partial
// response, kills the one-request-per-group amplification) AND V2S
// (single-track, pos masks only) at once, so a group-scaled 1-bit model
// (Bonsai g128) collapses its n_groups per-slice round-trips into ONE
// request/server while keeping the half-DRAM single-track compute. Body =
// V2G body with pos masks only; response = n_groups × d_out pos-track
// partials; client reconstructs y_g = 2*y_pos_g - Σ_{c∈g} fac·pc(x_c).

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

// PIM_V2_PACK (2026-07-21 late): append one chunk's write sequence to an
// existing program — byte-identical instruction stream to
// build_chunk_program (fully unrolled, no labels/branches), minus END.
static void emit_chunk_body(Program& p, int bank_id, uint32_t row_addr,
                            const uint32_t* col_data,
                            int col_start, int n_cols) {
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
}

// Write several banks' scratch rows in as few programs as the IMEM
// allows (greedy fill to ~7600 insts: 5 chunks/program → a 4-bank round
// = 3 programs instead of 12). Write-only programs carry no c2h, so
// unlike K-batched execs this packing has no recv-growth downside —
// it purely removes h2c round-trips.
struct ScratchWrite { int bank_id; uint32_t row; const uint32_t* data; };
static void per_column_write_rows_packed(SoftMCPlatform& platform,
                                         const std::vector<ScratchWrite>& ws,
                                         int* n_execs_out) {
  Program p;
  bool empty = true;
  for (const auto& w : ws) {
    int col_start = 0;
    for (int chunk = 0; chunk < 3; chunk++) {
      int n_cols = CHUNK_COLS[chunk];
      // a chunk body is ~1.5K insts; flush before it would overflow.
      if (!empty && p.size() / 8 + 1600 > 7600) {
        p.add_inst(SMC_END());
        platform.execute(p);
        (*n_execs_out)++;
        p = Program();
        empty = true;
      }
      emit_chunk_body(p, w.bank_id, w.row,
                      w.data + col_start * 16, col_start, n_cols);
      empty = false;
      col_start += n_cols;
    }
  }
  if (!empty) {
    p.add_inst(SMC_END());
    platform.execute(p);
    (*n_execs_out)++;
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
    // Label must be unique per ENTRY, not per bank: with LOAD-overflow
    // extra pools the same bank appears once per subarray window, and a
    // bank-only label makes every branch resolve to the last window's
    // loop (earlier windows' setup is skipped entirely).
    std::string lab = "REFRESH_SUBARR_B" + std::to_string(bank) +
                      "_" + std::to_string(bi);
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

// PIM_SEGPOP=1 (build7 image ONLY, trailer magic 0xDBC0DE05): run matvec
// row reads in the readback engine's SEG_POP mode — 2048 B/row of
// per-32b-segment popcount BYTES instead of the 8 KB raw row (4x recv
// collapse) with host segment_popcount eliminated. Raw-byte paths
// (LOAD verify, decay detection) transparently switch back to READ via
// ensure_readback(). Default 0 = build-6 behavior, byte-identical.
// HAZARD: on a pre-build7 image the SET word falls through the frontend
// decode into instruction-load — never set PIM_SEGPOP=1 on older images.
static int g_segpop = -1;               // env PIM_SEGPOP (init_debug_flags)
static bool g_mode_segpop_now = false;  // host view of the engine mode
static void ensure_readback(SoftMCPlatform& platform, bool segpop) {
  if (g_segpop <= 0) return;            // feature off/unparsed: never switch
  if (segpop == g_mode_segpop_now) return;
  if (segpop) { platform.set_readback_mode_segpop(); platform.set_readback_mode_segpop(); }
  else        { platform.set_readback_mode(false);   platform.set_readback_mode(false); }
  g_mode_segpop_now = segpop;
}
static inline size_t row_read_bytes() { return g_segpop > 0 ? 2048u : 8192u; }
// mode-aware per-segment popcounts of one received row image (stride
// row_read_bytes()): SEG_POP bytes ARE the counts; READ rows go through
// segment_popcount (defined below).
static void segment_popcount(const uint8_t* row_buf, int* out, int n);
static inline void row_pc(const uint8_t* row, int* out, int n) {
  if (g_segpop > 0) { for (int s = 0; s < n; s++) out[s] = row[s]; }
  else segment_popcount(row, out, n);
}

// PIM_ACCUM_XBP=1 (build8b image ONLY, trailer magic 0xDBC0DE07): on
// eligible single-track matvec requests, fold the per-bitplane
// place-value sum into the readback engine's ACCUM_XBP accumulator —
// per plane the host latches ±2^shift out-of-band and executes (the
// reads emit NO c2h); ONE flush_acc + 8192-B int32 drain per round
// replaces the per-program drains (recv wakes ÷ n_bitplanes).
// Request-level eligibility (else byte-identical fallback): single-track
// (all sign +), K==1 (one plane per program), every bitplane_factor
// = ±2^k (k ≤ 7), d_out ≤ 2048, no fused per-row repair possible,
// every round's units in ONE output-group slot, PIM_DEBUG_RX unset.
// HAZARD: on a pre-build8 image the SET word falls through the frontend
// decode into instruction-load — never set PIM_ACCUM_XBP=1 on older
// images (same class as the PIM_SEGPOP hazard above).
static int g_accxbp = -1;               // env PIM_ACCUM_XBP (init_debug_flags)
static bool g_mode_accxbp_now = false;  // host view: engine in ACCUM_XBP
static void ensure_accxbp(SoftMCPlatform& platform, bool on) {
  if (g_accxbp <= 0) return;
  if (on == g_mode_accxbp_now) return;
  if (on) {
    // idempotent SET ×2 (lost-word repair). Each send re-runs the
    // 128-cycle accumulator clear — safe HERE only because nothing has
    // accumulated yet; NEVER re-send between planes (it wipes the sum;
    // the accxbp-hw tool learned this on silicon).
    platform.set_readback_mode_accxbp();
    platform.set_readback_mode_accxbp();
    g_mode_segpop_now = false;          // engine left SEG_POP if it was there
  } else {
    platform.set_readback_mode(false);
    platform.set_readback_mode(false);
    platform.drain_stray(1500, 8);      // absorb mode-exit stragglers
    g_mode_segpop_now = false;          // next ensure_readback re-arms SEG_POP
  }
  g_mode_accxbp_now = on;
}
// ±2^k (k ≤ 7) → build8 weight encode {neg, shift}; false = not encodable.
static inline bool accxbp_encode(int32_t f, int* neg, int* shift) {
  if (f == 0) return false;
  uint32_t a = (uint32_t)(f < 0 ? -(int64_t)f : (int64_t)f);
  if (a & (a - 1)) return false;        // not a power of two
  int s = __builtin_ctz(a);
  if (s > 7) return false;
  *neg = (f < 0) ? 1 : 0; *shift = s;
  return true;
}

// Read-back a single row's contents into row_buf (must be ≥ 8192 bytes).
// Used by LOAD-time write verification and MM3D-start decay detection.
// Always a RAW read: drops to READ mode first when SEG_POP is active.
static int read_row_to_buffer(SoftMCPlatform& platform, int bank_id,
                              uint32_t row, uint8_t* row_buf,
                              int label_seed) {
  ensure_readback(platform, false);
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
// Fused-coset eligibility of the calib the CURRENT program is being built
// for. The fused body requires a separated-generator rank-4 tuple —
// validated ONLY for the primary calibs (s72 class on DIMM 2, s61 class
// on DIMM 0). The cs_extra voting calibs live in other subarrays and are
// NOT validated: fused on them corrupts their trips, and the client's
// median-of-3 then votes garbage in (the 2026-07-18 full-model
// '<|end|><|assistant|' failure). Set explicitly before EVERY builder
// call; single-threaded request loop makes a file-scope flag safe.
static bool g_fused_calib_ok = true;

// PIM_FUSED_COSET mode, read once (shared by the serial body and the
// bank-parallel builder so both emit the same step-3 variant):
//   0 = off (11 uniform wrRows — historic production body)
//   1 = coset mode (5 wrRows + 2 in-tuple coset doubleACTs)
//   2 = DIAGNOSTIC: fused position layout via 10 explicit wrRows, no cosets
//   3 = DIAGNOSTIC: explicit wrRows AND the coset doubleACTs (redundant)
static int fused_coset_mode() {
  static const int m = []{
    const char* v = getenv("PIM_FUSED_COSET");
    int mode = (v && *v) ? atoi(v) : 0;
    if (mode)
      fprintf(stderr, "[server] PIM_FUSED_COSET=%d: coset activation update "
                      "in the MAJ3 body\n", mode);
    return mode;
  }();
  return m;
}

// ---------------------------------------------------------------------------
// O4 (a): Fig-15-style RESIDENT CONSTANT ROWS (PIM_RESIDENT_CONSTS,
// default off — emission byte-identical when unset).
//
// MVDRAM's Fig-15 convention stores W AND ¬W resident (doubled matrix
// storage) so complement operands never need per-op writes. Audit of OUR
// dataflow (2026-07-19): no body ever consumes a complemented weight
// operand. The ternary split stores pos_mask and neg_mask as two
// INDEPENDENT resident rows (process_load_weights writes one per
// (chunk, sign) unit) and applies the sign ARITHMETICALLY at accumulate
// time (sign_factor * bitplane_factor). neg_mask != ~pos_mask (ternary:
// ~pos would cover {-1, 0}), so Fig-15's complement-doubling DOES NOT
// APPLY here. The applicable transfer is the underlying principle: any
// row whose content never changes per-op should be RESIDENT and cloned,
// not rewritten.
//
// In the fused body (PIM_FUSED_COSET=1) the 5 remaining wrRows are
// ONE@op[0], x@op[1], 0@op[2], x@op[4], 0@op[8]. The x rows carry the
// per-op activation bitplane — they must stay wrRows. The ONE row and
// the two 0 rows are CONSTANT: with a resident all-ones row and a
// resident all-zeros row per bank (selected from the screened primary
// pool at startup, per-column-written once, ACT-refreshed with the pool
// by the MM3D-entry refresh loop — "refreshed like weights"), they
// become RowClones. A wrRow runs a 128-iteration WRITE loop (~2k PHY
// slots; the 5 wrRows are the dominant share of fused-body PHY time); a
// framed clone is one doubleACT(30,1) (~120 PHY slots incl. PRE/SLEEP
// framing) and also skips the LDWD pattern-load prologue. IMEM cost
// drops too (~32 -> ~17 words per replaced row), which composes with
// PIM_PACK_ROUNDS.
//
// Deposit safety (pair-lattice law, test_safe_load.cpp): an external
// RowClone src->dst deposits src's pattern at {src ^ S : S subset of
// bits(src ^ dst)}; the tuple-intersecting deposits are open_rows[
// dst_idx ^ j] for every generator-sum e_j that is a bit-subset of
// (src ^ dst). setup_resident_consts() picks const rows such that
//   ONE  src -> op[0]:      NO tuple deposit besides the target;
//   ZERO src -> op[2],op[8]: deposits confined to {op[0]} + zero rows
//     ({2,6,10,14} coset + {8}) — zeros onto zero rows are no-ops, and a
//     zero deposit on op[0] is erased because the ONE fill is emitted
//     AFTER the zero clones (ordering is load-bearing).
// The const pair is also checked for mutual deposits. Src-side deposits
// into the wider pool have the same geometry as the production
// backup->Rfirst clones (screened pool rows, same subarray) — the
// envelope production already runs in.
//
// NOT modeled in sim: a clone leaves charge-shared (not write-driven)
// voltage; the frac x3 discharge on a CLONED ONE anchor is the most
// plausible silicon divergence. PIM_RESIDENT_CONSTS=2 keeps ONE as a
// wrRow (zeros-only) as the decomposition arm for the silicon A/B.
//   0/unset = off, 1 = clone ONE + zeros, 2 = clone zeros only.
static const uint32_t RES_ROW_NONE = 0xFFFFFFFFu;
static int resident_consts_mode() {
  static const int m = []{
    const char* v = getenv("PIM_RESIDENT_CONSTS");
    int mode = (v && *v) ? atoi(v) : 0;
    if (mode)
      fprintf(stderr, "[server] PIM_RESIDENT_CONSTS=%d: fused-body constant "
                      "rows cloned from resident pool rows (%s)\n",
              mode, mode == 2 ? "zeros only, ONE stays wrRow" : "ONE + zeros");
    return mode;
  }();
  return m;
}
// O4 (a) DRIFT FIX (2026-07-19, RESULT.md addendum 23): cadence for
// re-writing the resident const rows at MM3D-request granularity.
// Addendum 22's drift physics: same-subarray tuple traffic transitions
// resident rows' CONTENT one-shot (coupling-class-ordered saturation,
// ~65% of pool rows within ~80-160 bodies), and the MM3D-entry
// ACT-refresh PRESERVES the drifted content — it restores charge, not
// data. Weight rows are re-written per LOAD slice, but the const rows
// sat resident for the whole run, so every fused body cloned from
// progressively degraded ONE/ZERO sources — invisible to short
// protocol runs (fresh consts), fatal at full-model traffic scale
// (consts-only full model 2026-07-19: '1. I am a helpful AI assistant'
// vs the features-off control's Paris on the same binary).
// PIM_CONSTS_REWRITE_EVERY=N: rewrite every Nth MM3D request.
// Default 1 = every request (cost: one program with 2 uniform wrRows
// per const-bank vs 3 clones saved x hundreds of bodies/request).
// 0 = never = the pre-fix behavior, kept as the drift-probe arm.
static int consts_rewrite_every() {
  static const int v = []{
    const char* e = getenv("PIM_CONSTS_REWRITE_EVERY");
    long n = (e && *e) ? atol(e) : 1;
    if (n < 0) n = 1;
    if (e && *e)
      fprintf(stderr, "[server] PIM_CONSTS_REWRITE_EVERY=%ld%s\n", n,
              n == 0 ? " (NEVER re-write — pre-drift-fix diagnostic arm)"
                     : "");
    return (int)n;
  }();
  return v;
}

// Framed RowClone src->dst — identical dwell + framing to the step-1
// backup->Rfirst clone (PRE / SLEEP / doubleACT(30,1) / SLEEP / PRE /
// SLEEP). Caller must have BAR set for the target bank.
static void emit_const_clone(Program& p, uint32_t src_row, uint32_t dst_row) {
  p.add_below(PRE(BAR, 0, 0));
  p.add_inst(SMC_SLEEP(6));
  p.add_below(doubleACT(/*t_12=*/30, /*t_23=*/1, src_row, dst_row));
  p.add_inst(SMC_SLEEP(6));
  p.add_below(PRE(BAR, 0, 0));
  p.add_inst(SMC_SLEEP(6));
}

static void emit_bank_combined_body(Program& p,
                                     int bank_id,
                                     uint32_t backup_row,
                                     uint32_t Rfirst, uint32_t Rsecond,
                                     const uint32_t* open_rows,
                                     uint32_t x_pattern,
                                     int label_base,
                                     uint32_t res_one = RES_ROW_NONE,
                                     uint32_t res_zero = RES_ROW_NONE) {
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

  // 3. Overwrite the 11 non-weight slots with x / 0 / ONE.
  //    Default: 11 uniform wrRows (ONE + 5×x + 5×zero) — the wrRows are
  //    ~80% of the per-MAJ program cost.
  //    PIM_FUSED_COSET=1: lattice-addressing update — 5 wrRows + 2 in-tuple
  //    coset broadcasts produce the same 5/5/5 vote balance with x on
  //    positions {1,5,9,13}+{4} and zeros on {2,6,10,14}+{8}; W stays on
  //    {3,7,11,12,15}. Requires (a) a separated-generator rank-4 tuple so
  //    the sorted position index equals the generator bitmask (true for the
  //    calib_dimm2 s72 and calib_dimm0 s61 classes), and (b) t_12 ≥ 10 on
  //    the coset doubleACTs. Bit-exact on silicon for the s72 calib on
  //    benders 0 & 2 (fused-maj A/B campaign, 0/114688 bad segments,
  //    2026-07-17); validate before enabling on other calibs. The
  //    unbalanced 7W/4x/4z variant is KNOWN WRONG (79-90% bad) — do not
  //    drop the two extra anchor writes.
  // 1 = coset mode (5 wrRows + 2 coset doubleACTs).
  // 2 = DIAGNOSTIC: fused position layout via 10 explicit wrRows, no cosets.
  // 3 = DIAGNOSTIC: explicit wrRows AND the coset doubleACTs (redundant).
  // (Mode accessor hoisted to fused_coset_mode() so the bank-parallel
  // builder emits the identical step-3 variant.)
  const int s_fused_coset = fused_coset_mode();
  // O4 (a): resident-const clones are wired for the FUSED mode-1 body only
  // (the production winner whose wrRows they attack); modes 0/2/3 keep
  // their historic emission byte-for-byte. Per-body res rows come from the
  // caller (primary-calib bodies only); RES_ROW_NONE falls back to wrRow.
  const int rc_mode = resident_consts_mode();
  const bool use_consts = rc_mode > 0 && s_fused_coset == 1 &&
                          g_fused_calib_ok && res_zero != RES_ROW_NONE;
  if (s_fused_coset && g_fused_calib_ok) {
    static const int f_act[5]  = {1, 5, 9, 13, 4};
    static const int f_zero[5] = {2, 6, 10, 14, 8};
    if (use_consts) {
      // Order is load-bearing (deposit safety, see the block comment at
      // resident_consts_mode): zero clones first — their only permitted
      // off-target deposits are zeros onto zero rows or onto op[0], and
      // the ONE fill emitted after them erases the op[0] case. The x
      // wrRows carry per-op data and stay writes.
      emit_const_clone(p, res_zero, open_rows[2]);
      emit_const_clone(p, res_zero, open_rows[8]);
      if (rc_mode == 2 || res_one == RES_ROW_NONE)
        p.add_below(wrRow_immediate_label(BAR, open_rows[0], ONE,
                                           label_base + 0));
      else
        emit_const_clone(p, res_one, open_rows[0]);
      p.add_below(wrRow_immediate_label(BAR, open_rows[1], x_pattern,
                                         label_base + 1));
      p.add_below(wrRow_immediate_label(BAR, open_rows[4], x_pattern,
                                         label_base + 3));
    } else {
    p.add_below(wrRow_immediate_label(BAR, open_rows[0], ONE, label_base + 0));
    if (s_fused_coset == 1) {
      p.add_below(wrRow_immediate_label(BAR, open_rows[1], x_pattern,
                                         label_base + 1));
      p.add_below(wrRow_immediate_label(BAR, open_rows[2], 0u, label_base + 2));
      p.add_below(wrRow_immediate_label(BAR, open_rows[4], x_pattern,
                                         label_base + 3));
      p.add_below(wrRow_immediate_label(BAR, open_rows[8], 0u, label_base + 4));
    } else {
      for (int i = 0; i < 5; i++)
        p.add_below(wrRow_immediate_label(BAR, open_rows[f_act[i]],
                                           x_pattern, label_base + 1 + i));
      for (int i = 0; i < 5; i++)
        p.add_below(wrRow_immediate_label(BAR, open_rows[f_zero[i]], 0u,
                                           label_base + 100 + i));
    }
    }
    if (s_fused_coset != 2) {
      p.add_inst(SMC_SLEEP(6));
      p.add_below(PRE(BAR, 0, 0));
      p.add_inst(SMC_SLEEP(6));
      // x → coset {1,5,9,13}: partner at generator distance g2^g3.
      p.add_below(doubleACT(10, 2, open_rows[1], open_rows[13]));
      p.add_inst(SMC_SLEEP(6));
      p.add_below(PRE(BAR, 0, 0));
      p.add_inst(SMC_SLEEP(6));
      // 0 → coset {2,6,10,14}.
      p.add_below(doubleACT(10, 2, open_rows[2], open_rows[14]));
    }
  } else {
    static const int act_pos[5]  = {1, 4, 7, 10, 13};
    static const int zero_pos[5] = {2, 5, 8, 11, 14};
    p.add_below(wrRow_immediate_label(BAR, open_rows[0], ONE, label_base + 0));
    for (int i = 0; i < 5; i++)
      p.add_below(wrRow_immediate_label(BAR, open_rows[act_pos[i]],
                                         x_pattern, label_base + 1 + i));
    for (int i = 0; i < 5; i++)
      p.add_below(wrRow_immediate_label(BAR, open_rows[zero_pos[i]], 0u,
                                         label_base + 100 + i));
  }
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
    int label_seed,
    // O4 (a): optional per-body (ONE row, ZERO row) resident-const pairs;
    // nullptr / short vector / RES_ROW_NONE entries = wrRow fallback.
    const std::vector<std::pair<uint32_t,uint32_t>>* res_consts = nullptr) {
  Program p;
  p.add_inst(SMC_LI(8, CASR));
  for (size_t i = 0; i < bank_ids.size(); i++) {
    uint32_t rc1 = RES_ROW_NONE, rc0 = RES_ROW_NONE;
    if (res_consts && i < res_consts->size()) {
      rc1 = (*res_consts)[i].first;
      rc0 = (*res_consts)[i].second;
    }
    emit_bank_combined_body(p, bank_ids[i], backup_rows[i],
                            Rfirsts[i], Rseconds[i],
                            open_rows_list[i], x_patterns[i],
                            label_seed + (int)i * 2000, rc1, rc0);
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
// PIM_FUSED_COSET composition (task O3): when the fused coset body is
// enabled AND the calib is fused-eligible (g_fused_calib_ok — same gate
// as the serial body), step 3 emits 5 wrRows per bank serially and then
// the TWO in-tuple coset doubleACTs (t_12=10 / t_23=2, exactly the
// serial dwells) as pack4-parallel sequences across the 4 banks —
// followed by the per-bank frac sections, preserving the serial fused
// body's per-bank order: wrRows → coset dACTs → frac ×3 → MAJ3 → rdRow.
//
// Register layout (live across the parallel sections):
//   slot 0: CASR   = 8 (kept stable; column-stride for ICAR)
//   slots 1, 2, 3, 9: bar0..bar3 (LI'd once at body top, never overwritten)
//   slots 11, 4, 5, 8: src_reg[0..3] — re-LI'd per parallel phase
//   slots 13, 6, 14, 15: dst_reg[0..3] — re-LI'd per parallel phase
// Other reg uses (PATTERN_REG=12, BAR=7, etc.) are confined to the
// per-bank serial sections and re-LI their inputs themselves; we
// re-establish CASR=8 and the canonical BAR=bank_id at the top of each
// serial section. The serial wrRow helper clobbers slots 4/6/13/14
// (CAR/RAR/LOOP_COLS/NUM_COLS_REG overlap src/dst slots), so EVERY
// parallel phase that follows a serial section re-LIs src/dst — the
// coset phases do this exactly like Phase 5 (MAJ3) always has.
static Program build_multibank_parallel_program(
    const std::vector<int>& bank_ids,
    const std::vector<uint32_t>& backup_rows,
    const std::vector<uint32_t>& Rfirsts,
    const std::vector<uint32_t>& Rseconds,
    const std::vector<const uint32_t*>& open_rows_list,
    const std::vector<uint32_t>& x_patterns,
    int label_seed,
    // O4 (a): optional per-body resident-const pairs — see the serial
    // builder. Forwarded through the serial fallbacks; honored in the
    // per-bank serial wrRow section of the fused (mode 1) path.
    const std::vector<std::pair<uint32_t,uint32_t>>* res_consts = nullptr)
{
  const int N = (int)bank_ids.size();
  // Today the parallel scheduler is wired for N=4. Smaller N falls back
  // to the serial multibank emit (parallel-of-1 is identical to serial).
  if (N != 4) {
    return build_multibank_combined_program(
        bank_ids, backup_rows, Rfirsts, Rseconds,
        open_rows_list, x_patterns, label_seed, res_consts);
  }
  // The pack4 scheme assumes one slot per DISTINCT bank: 4 work units on
  // a repeated bank (e.g. single-bank + PIM_INLINE_BITPLANES=4) would
  // interleave two doubleACT patterns on the same bank — uncalibrated
  // timing. Fall back to the serial emitter for those.
  for (int a = 0; a < 4; a++)
    for (int b2 = a + 1; b2 < 4; b2++)
      if (bank_ids[a] == bank_ids[b2]) {
        static bool warned = false;
        if (!warned) {
          warned = true;
          fprintf(stderr, "[server] PIM_PARALLEL_BANKS: duplicate bank %d "
                  "in 4-unit batch — using serial multibank emit\n",
                  bank_ids[a]);
        }
        return build_multibank_combined_program(
            bank_ids, backup_rows, Rfirsts, Rseconds,
            open_rows_list, x_patterns, label_seed, res_consts);
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

  // Phase 3 + 4: per-bank serial wrRow setup + frac × 3, honoring the
  // same PIM_FUSED_COSET step-3 variants as emit_bank_combined_body
  // (identical rows, patterns, dwells, and per-bank order; only the
  // scheduling of the two coset doubleACTs differs — they run as
  // pack4-parallel sequences in Phase 3b below).
  //   mode 0: 11 uniform wrRows/bank, frac inside this loop (emission
  //           byte-identical to the pre-fused parallel builder).
  //   mode 1: 5 wrRows/bank (ONE@0, x@1, 0@2, x@4, 0@8); cosets in 3b;
  //           frac moves to Phase 4 (after the cosets, as in serial).
  //   mode 2: DIAGNOSTIC — fused layout via 10 explicit wrRows, no
  //           cosets; frac stays inside this loop.
  //   mode 3: DIAGNOSTIC — explicit wrRows AND the coset doubleACTs.
  // The fused 5/5/5 vote balance (x on {1,5,9,13}+{4}, zeros on
  // {2,6,10,14}+{8}, W on {3,7,11,12,15}) is a hard rule — the
  // unbalanced variant is KNOWN WRONG on silicon.
  const int fused_mode = g_fused_calib_ok ? fused_coset_mode() : 0;
  const bool fused_cosets = (fused_mode == 1 || fused_mode == 3);
  static const int act_pos[5]  = {1, 4, 7, 10, 13};
  static const int zero_pos[5] = {2, 5, 8, 11, 14};
  static const int f_act[5]  = {1, 5, 9, 13, 4};
  static const int f_zero[5] = {2, 6, 10, 14, 8};
  for (int b = 0; b < 4; b++) {
    p.add_inst(SMC_LI((uint32_t)bank_ids[b], BAR));
    p.add_inst(SMC_LI(128, NUM_COLS_REG));
    // O4 (a): fused mode-1 bodies may clone their constant rows from
    // resident pool rows (same substitution + ordering as the serial
    // body). The clone's doubleACT LIs RF_REG(10)/LOOP_COLS(13) — both
    // already in the serial-section clobber set (every parallel phase
    // re-LIs src/dst after serial sections), so no new register hazard.
    uint32_t rc1 = RES_ROW_NONE, rc0 = RES_ROW_NONE;
    if (res_consts && (size_t)b < res_consts->size()) {
      rc1 = (*res_consts)[b].first;
      rc0 = (*res_consts)[b].second;
    }
    const int rc_mode = resident_consts_mode();
    const bool use_consts_b = rc_mode > 0 && fused_mode == 1 &&
                              rc0 != RES_ROW_NONE;
    if (use_consts_b) {
      emit_const_clone(p, rc0, open_rows_list[b][2]);
      emit_const_clone(p, rc0, open_rows_list[b][8]);
      if (rc_mode == 2 || rc1 == RES_ROW_NONE)
        p.add_below(wrRow_immediate_label(BAR, open_rows_list[b][0],
                                           ONE, label_seed + b*2000 + 0));
      else
        emit_const_clone(p, rc1, open_rows_list[b][0]);
      p.add_below(wrRow_immediate_label(BAR, open_rows_list[b][1],
                                         x_patterns[b],
                                         label_seed + b*2000 + 1));
      p.add_below(wrRow_immediate_label(BAR, open_rows_list[b][4],
                                         x_patterns[b],
                                         label_seed + b*2000 + 3));
    } else {
    p.add_below(wrRow_immediate_label(BAR, open_rows_list[b][0],
                                       ONE, label_seed + b*2000 + 0));
    if (fused_mode == 1) {
      p.add_below(wrRow_immediate_label(BAR, open_rows_list[b][1],
                                         x_patterns[b],
                                         label_seed + b*2000 + 1));
      p.add_below(wrRow_immediate_label(BAR, open_rows_list[b][2], 0u,
                                         label_seed + b*2000 + 2));
      p.add_below(wrRow_immediate_label(BAR, open_rows_list[b][4],
                                         x_patterns[b],
                                         label_seed + b*2000 + 3));
      p.add_below(wrRow_immediate_label(BAR, open_rows_list[b][8], 0u,
                                         label_seed + b*2000 + 4));
    } else {
      const int* ap = (fused_mode >= 2) ? f_act  : act_pos;
      const int* zp = (fused_mode >= 2) ? f_zero : zero_pos;
      for (int i = 0; i < 5; i++)
        p.add_below(wrRow_immediate_label(BAR, open_rows_list[b][ap[i]],
                                           x_patterns[b],
                                           label_seed + b*2000 + 1 + i));
      for (int i = 0; i < 5; i++)
        p.add_below(wrRow_immediate_label(BAR, open_rows_list[b][zp[i]],
                                           0u,
                                           label_seed + b*2000 + 100 + i));
    }
    }
    p.add_inst(SMC_SLEEP(6));
    p.add_below(PRE(BAR, 0, 0));
    p.add_inst(SMC_SLEEP(6));
    if (!fused_cosets) {
      for (int j = 0; j < 3; j++) {
        p.add_inst(SMC_SLEEP(6));
        p.add_below(frac_template(/*t_frac=*/0, open_rows_list[b][0]));
        p.add_inst(SMC_SLEEP(6));
      }
      p.add_inst(SMC_SLEEP(6));
      p.add_below(PRE(BAR, 0, 0));
      p.add_inst(SMC_SLEEP(6));
    }
  }

  // Phase 3b (fused modes 1 & 3): the two in-tuple coset doubleACTs as
  // pack4-parallel sequences — the 4 banks' (src → dst) pairs of ONE
  // coset in one packed sequence, per-bank dwell exactly t_12=10 /
  // t_23=2 (same as the serial fused body; parallel_doubleACT preserves
  // per-bank dwells and spaces cross-bank ACTs to honor tRRD_S / tFAW
  // via schedule_bank_starts). src/dst regs are re-LI'd here because the
  // serial wrRow sections clobber slots 4/6/13/14; the SLEEP(6) after
  // the LIs is the same RAW-hazard barrier Phase 5 uses after serial
  // sections.
  if (fused_cosets) {
    std::vector<uint32_t> coset_src(4), coset_dst(4);
    // x → coset {1,5,9,13}: partner at generator distance g2^g3.
    for (int b = 0; b < 4; b++) {
      coset_src[b] = open_rows_list[b][1];
      coset_dst[b] = open_rows_list[b][13];
    }
    li_rows(src_reg, coset_src);
    li_rows(dst_reg, coset_dst);
    p.add_inst(SMC_SLEEP(6));   // RAW hazard barrier: let LI writes propagate
    p.add_below(parallel_doubleACT(10, 2, bar_reg, src_reg, dst_reg));
    p.add_inst(SMC_SLEEP(6));
    pre_all();
    p.add_inst(SMC_SLEEP(6));
    // 0 → coset {2,6,10,14}.
    for (int b = 0; b < 4; b++) {
      coset_src[b] = open_rows_list[b][2];
      coset_dst[b] = open_rows_list[b][14];
    }
    li_rows(src_reg, coset_src);
    li_rows(dst_reg, coset_dst);
    p.add_inst(SMC_SLEEP(6));   // RAW hazard barrier
    p.add_below(parallel_doubleACT(10, 2, bar_reg, src_reg, dst_reg));
    p.add_inst(SMC_SLEEP(6));
    pre_all();
    p.add_inst(SMC_SLEEP(6));

    // Phase 4 (fused): per-bank serial frac × 3 AFTER the coset
    // broadcasts — mirrors the serial fused body's per-bank order.
    for (int b = 0; b < 4; b++) {
      p.add_inst(SMC_LI((uint32_t)bank_ids[b], BAR));
      p.add_inst(SMC_LI(128, NUM_COLS_REG));
      for (int j = 0; j < 3; j++) {
        p.add_inst(SMC_SLEEP(6));
        p.add_below(frac_template(/*t_frac=*/0, open_rows_list[b][0]));
        p.add_inst(SMC_SLEEP(6));
      }
      p.add_inst(SMC_SLEEP(6));
      p.add_below(PRE(BAR, 0, 0));
      p.add_inst(SMC_SLEEP(6));
    }
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
  // 2026-07-18 per-subarray screened pools: the REAL subarray window
  // [start, end) each extra's pool rows live in (parsed from the
  // "# window" comment of its PIM_POOL_LIST_FILE_SUB file). Used by the
  // MM3D refresh when LOAD-overflow places handle rows there.
  std::vector<std::pair<uint32_t,uint32_t>> pool_extra_win;
  // LOAD-overflow allocation cursors into pool_extra[i] (only advanced
  // when PIM_LOAD_OVERFLOW_SUBS=1).
  std::vector<size_t> pool_extra_cursor;
  // O4 (a): resident constant rows (PIM_RESIDENT_CONSTS). Selected from
  // the screened primary pool at startup and REMOVED from backup_pool
  // (so neither LOAD_WEIGHTS handles nor the V2 scratch tail can touch
  // them), per-column-written once (all-ones / all-zeros), ACT-refreshed
  // with the subarray by the MM3D-entry refresh loop. RES_ROW_NONE =
  // unavailable (feature off or selection failed) -> wrRow fallback.
  uint32_t res_one_row  = RES_ROW_NONE;
  uint32_t res_zero_row = RES_ROW_NONE;
  // O10 (2026-07-20): per-bank FUSED-layout column mask. dimm0's fused
  // residual (addendum 24) is the fused OPERAND LAYOUT mis-computing on
  // marginal columns of this die (content-conditional per column; the
  // May calibration screened the BASELINE layout only). Columns flagged
  // here get their popcount HOST-REPAIRED from the unit's weight mask
  // whenever the executed body used the fused layout — substitution is
  // exact by definition (MAJ3(W,x,0) = W & x), so repairing a correct
  // column is a no-op; the mask size is the honest "not computed
  // in-DRAM" fraction. Empty = feature off for this bank.
  std::vector<uint8_t> fused_col_bad;   // [2048], 1 = host-repair
  int fused_bad_count = 0;
};

// O10: any bank has a fused colmask -> LOAD keeps full masks per round
// (repair needs them at MM3D time). Set once at startup.
static bool g_fused_colmask_any = false;

// PIM_FUSED_COLMASK_FILE: path pattern with {bank} token; the file lists
// the fused-GOOD columns (one integer per line, '#' comments — the
// fused-colmask-exe output). Complement = repair set. Missing file =
// colmask disabled for that bank (loud stderr). Present but empty of
// integers = FATAL (explicit config that can't be honored).
static void load_fused_colmask(BankConfig& b) {
  const char* pat = getenv("PIM_FUSED_COLMASK_FILE");
  if (!pat || !*pat) return;
  std::string path = pat;
  size_t pos;
  while ((pos = path.find("{bank}")) != std::string::npos)
    path.replace(pos, 6, std::to_string(b.bank_id));
  FILE* fp = fopen(path.c_str(), "r");
  if (!fp) {
    fprintf(stderr, "[fused-colmask] bank %d: no file '%s' — colmask "
            "DISABLED for this bank (fused stays unrepaired)\n",
            b.bank_id, path.c_str());
    return;
  }
  std::vector<uint8_t> bad(2048, 1);
  long n_good = 0;
  char line[128];
  while (fgets(line, sizeof(line), fp)) {
    char* s = line;
    while (*s == ' ' || *s == '\t') s++;
    if (*s == '#' || *s == '\n' || *s == 0) continue;
    long c = atol(s);
    if (c >= 0 && c < 2048 && bad[c]) { bad[c] = 0; n_good++; }
  }
  fclose(fp);
  if (n_good == 0) {
    fprintf(stderr, "[fused-colmask] FATAL: '%s' lists no columns\n",
            path.c_str());
    exit(1);
  }
  b.fused_bad_count = 2048 - (int)n_good;
  if (b.fused_bad_count == 0) {
    fprintf(stderr, "[fused-colmask] bank %d: all 2048 columns fused-good "
            "('%s') — no repairs\n", b.bank_id, path.c_str());
    return;  // fused_col_bad stays empty -> zero-cost path
  }
  b.fused_col_bad = std::move(bad);
  g_fused_colmask_any = true;
  fprintf(stderr, "[fused-colmask] bank %d: %d/2048 columns host-repaired "
          "on fused bodies ('%s')\n", b.bank_id, b.fused_bad_count,
          path.c_str());
}

// Substitute exact popcount(mask & x) for the repair columns. Call ONLY
// when the body that produced `pc` ran the FUSED layout (mode 1/3 AND
// all-primary program). `mask` = the unit's weight row (2048 x u32).
static inline void fused_repair_pc(const BankConfig& b, int* pc,
                                   const uint32_t* mask, uint32_t xb) {
  if (b.fused_col_bad.empty() || !mask) return;
  const uint8_t* fb = b.fused_col_bad.data();
  for (int j = 0; j < 2048; j++)
    if (fb[j]) pc[j] = __builtin_popcount(mask[j] & xb);
}

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
  if (idx == 0 || bc.cs_extra.empty()) return bc.calib;
  size_t i = (idx - 1) % bc.cs_extra.size();
  return bc.cs_extra[i];
}
static inline const std::vector<uint32_t>& bc_pool_idx(const BankConfig& bc, uint32_t idx) {
  if (idx == 0 || bc.pool_extra.empty()) return bc.backup_pool;
  size_t i = (idx - 1) % bc.pool_extra.size();
  return bc.pool_extra[i];
}
// V2-fallback scratch reserve. The tail rows of each PRIMARY pool are
// never handed to LOAD_WEIGHTS handles and are the only rows V2-mode
// requests may scratch on. Without this, mixed LOAD+V2 traffic (any
// full-model run: LOAD fills the pool, later slices fall back to V2)
// had V2's round-cycled scratch writes land on handle-RESIDENT weight
// rows — per_column_write_row destroyed the loaded weights and every
// subsequent LOAD-mode matmul computed garbage (2026-07-18 full-model
// '<|end|>' regression; V2_SCRATCH had been reduced 110 → 0 in the
// ~50-row DIMM-0-pool era under a "pure-LOAD usage" assumption).
// Extra-calib pools (calib_idx > 0) hold no handles → no reserve needed.
static size_t v2_scratch_reserve() {
  static const size_t v = []{
    const char* e = getenv("PIM_V2_SCRATCH");
    long n = (e && *e) ? atol(e) : 16;
    return (size_t)(n < 0 ? 0 : n);
  }();
  return v;
}
// LOAD-overflow into extra subarrays (2026-07-18, default ON;
// PIM_LOAD_OVERFLOW_SUBS=0 opts out): LOAD_WEIGHTS may overflow handle
// rows into the per-subarray screened extra pools once the primary pool
// is full, lifting the 294-row LOAD ceiling. Each overflowed
// (round, bank) unit records which calib owns its subarray (the
// RowClone scratch→Rfirst and the MAJ3 tuple must be same-subarray),
// and MM3D dispatches that unit on the owning calib's tuple (plain
// body — fused stays primary-only). Extras whose window overlaps the
// PRIMARY subarray (sub 71 on DIMM 2: scratch-only pools, relaxed
// screening) are never overflow targets.
static bool load_overflow_enabled() {
  static const bool v = []{
    const char* e = getenv("PIM_LOAD_OVERFLOW_SUBS");
    return !(e && *e && atoi(e) == 0);
  }();
  return v;
}
static inline size_t v2_pool_idx(const BankConfig& bc, uint32_t calib_idx,
                                 size_t round, size_t pool_size) {
  const size_t rsv = v2_scratch_reserve();
  // Primary pools always protect LOAD-resident rows behind the tail
  // reserve. Extra pools hold residents only under LOAD-overflow — then
  // their V2/voting scratch draws must use the tail reserve too.
  if ((calib_idx == 0 || load_overflow_enabled()) && rsv > 0 && rsv < pool_size)
    return pool_size - rsv + (round % rsv);
  size_t i = round_to_pool_idx(bc, round);
  return (i >= pool_size) ? i % pool_size : i;
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

// O4 (a) DRIFT FIX: one program that re-writes every bank's resident
// const rows with their nominal content (ZERO = 0x00000000 fill,
// ONE = 0xFFFFFFFF fill). Uses the SAME uniform-fill wrRow primitive
// the non-consts fused body uses for these constants
// (wrRow_immediate_label: write-driven levels, 128-col WRITE loop) —
// NOT per_column_write_row, whose 3-programs-per-row shape would cost
// ~24 execute round-trips per request; this is ONE execute (~2 wrRows
// x N banks ~= a few hundred IMEM words, no receiveData).
// Returns false if no bank has resident consts (nothing executed).
static bool rewrite_resident_const_rows(SoftMCPlatform& platform,
                                        std::vector<BankConfig>& banks) {
  Program p;
  p.add_inst(SMC_LI(8, CASR));
  int lbl = 0;
  for (auto& b : banks) {
    if (b.res_zero_row == RES_ROW_NONE && b.res_one_row == RES_ROW_NONE)
      continue;
    p.add_inst(SMC_LI(b.bank_id, BAR));
    if (b.res_zero_row != RES_ROW_NONE)
      p.add_below(wrRow_immediate_label(BAR, b.res_zero_row, 0u,
                                        900000 + lbl++));
    if (b.res_one_row != RES_ROW_NONE)
      p.add_below(wrRow_immediate_label(BAR, b.res_one_row, ONE,
                                        900000 + lbl++));
    // tWR guard + close the last written row before the next bank /
    // program end (mirrors the chunk-program trailer).
    p.add_inst(SMC_SLEEP(8));
    p.add_below(PRE(BAR, 0, 0));
    p.add_inst(SMC_SLEEP(6));
  }
  if (lbl == 0) return false;
  p.add_inst(SMC_END());
  platform.execute(p);
  return true;
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
  // LOAD-overflow (PIM_LOAD_OVERFLOW_SUBS=1): which calib owns each
  // (round, bank) unit. 0 = primary, i+1 = cs_extra[i]. Empty vector =
  // all-primary (legacy handles). MM3D emits each unit's body on its
  // owning calib's tuple.
  std::vector<std::vector<uint8_t>> per_round_calib_sel;  // [round][bk]
  // Extra refresh windows this handle's overflow rows live in, per bank
  // (deduped at MM3D dispatch across handles).
  std::vector<std::vector<std::pair<uint32_t,uint32_t>>> extra_refresh_wins;  // [bk]
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
// PIM_V2_PACK = 1: pack each V2 round's per-bank scratch writes into as
// few programs as IMEM allows (write-only programs, no c2h — pure
// round-trip removal). Default OFF for back-compat.
static int g_v2_pack = -1;
// PIM_REFRESH_BETWEEN = N: insert SMC_REF in the multibank-combined
// program after every N bank-bodies. Default 0 = no in-program refresh
// (matches today's behaviour: auto-refresh is OFF, only intermittent
// refresh when the host gap allows). Tests if manual refresh between
// bodies (vs the auto-refresh that interleaves into doubleACT and
// disrupts charge-sharing) helps with cumulative ACT-disturb on
// open_rows over many MAJ3 invocations.
static int g_refresh_between = -1;
// O4 (b): PIM_PACK_ROUNDS = N packs MM3D bodies from up to N consecutive
// rounds (x the PIM_INLINE_BITPLANES chunking) into ONE program + ONE
// receiveData, up to the BITSTREAM_IMEM envelope. Default 1 = the historic
// per-(round x bp-chunk) cadence, code path untouched. MM3D-handle path
// only: the V2 path's per-round scratch writes need write-then-use
// locality (2026-05-04: batching writes upfront broke q_proj), so V2
// never packs across rounds.
static int g_pack_rounds = -1;
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
  if (g_segpop < 0) {
    g_segpop = env_flag("PIM_SEGPOP", 0);
    if (g_segpop)
      fprintf(stderr, "[server] PIM_SEGPOP=1: build7 SEG_POP readback "
              "(2048 B/row, host segment_popcount eliminated)\n");
  }
  if (g_accxbp < 0) {
    g_accxbp = env_flag("PIM_ACCUM_XBP", 0);
    if (g_accxbp) {
      fprintf(stderr, "[server] PIM_ACCUM_XBP=1: build8b in-fabric "
              "cross-bit-plane accumulator on eligible single-track "
              "matvecs (one 8 KB drain per round)\n");
      // Accumulate/write programs emit NO c2h in this mode: each
      // execute's bounded accum receiver would otherwise idle the full
      // PIM_ACCUM_QUIET_MS (default 500 ms). Shrink both knobs unless
      // the user pinned them explicitly.
      setenv("PIM_ACCUM_QUIET_MS", "8", 0);
      setenv("PIM_ACCUM_TICK_MS",  "4", 0);
    }
  }
  if (g_v2_pack < 0) {
    // Default ON since 2026-07-21 (full-model token-identical, wcol
    // 10.8->6.4 ms/request, 2-tok wall 147.6->132.8 s). =0 restores the
    // 12-programs-per-round legacy cadence byte-for-byte.
    g_v2_pack = env_flag("PIM_V2_PACK", 1);
    if (g_v2_pack)
      fprintf(stderr, "[server] PIM_V2_PACK=1: per-round scratch writes "
              "packed (12 -> ~3 programs/round)\n");
  }
  if (g_refresh_between < 0) g_refresh_between = env_flag("PIM_REFRESH_BETWEEN", 0);
  if (g_pack_rounds < 0) {
    g_pack_rounds = env_flag("PIM_PACK_ROUNDS", 1);
    if (g_pack_rounds < 1) g_pack_rounds = 1;
    if (g_pack_rounds > 1)
      fprintf(stderr, "[server] PIM_PACK_ROUNDS=%d: MM3D bodies packed "
              "across rounds up to the IMEM envelope\n", g_pack_rounds);
  }
  if (g_bitstream_imem < 0) {
    g_bitstream_imem = env_flag("BITSTREAM_IMEM", 2048);
    if (g_bitstream_imem <= 0) g_bitstream_imem = 2048;
    fprintf(stderr, "[server] BITSTREAM_IMEM=%d (set BITSTREAM_IMEM=8192"
                    " on the rebuilt bitstream)\n", g_bitstream_imem);
    // Heuristic K-cap warning. Body sizes (measured 2026-07-18 via
    // PIM_DUMP_MM3D_PROGRAMS): ~416 inst/bank serial (~258 fused),
    // ~407 inst/bank parallel (~244 parallel-fused). K>1 batches have
    // M != 4 work units, so build_multibank_parallel_program falls back
    // to the SERIAL emitter regardless of PIM_PARALLEL_BANKS — use the
    // serial (largest) figure for the fit estimate so we warn early,
    // before platform.cpp's gate kills the program at execute time.
    int per_body = 416;
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

  // Check pool space on every bank. We reserve v2_scratch_reserve() rows
  // at the END of each pool for v2-fallback (per-request) requests so
  // they never collide with handle-allocated rows. Send non-zero ack on
  // exhausted so the client falls back to v2 for this slice.
  // (History: this reserve was 110, then 0 in the ~50-row DIMM-0-pool
  // era assuming pure-LOAD usage — which made every mixed LOAD+V2 run
  // corrupt resident weights. See v2_scratch_reserve().)
  const size_t V2_SCRATCH = v2_scratch_reserve();
  const bool overflow_subs = load_overflow_enabled();
  // Per-bank allocation plan: for each round, (backup_row, calib_sel).
  // calib_sel 0 = primary pool/calib; i+1 = cs_extra[i]'s pool/calib.
  // Primary fills first (unchanged legacy layout), then extras in cs_extra
  // order. Every pool keeps its V2_SCRATCH tail free for voting/V2 scratch.
  std::vector<std::vector<std::pair<uint32_t,uint8_t>>> plan(N);
  std::vector<size_t> primary_take(N, 0);
  std::vector<std::vector<size_t>> extra_take(N);
  for (int bk = 0; bk < N; bk++) {
    BankConfig& b = banks[bk];
    extra_take[bk].assign(b.pool_extra.size(), 0);
    size_t prim_avail = 0;
    if (b.backup_pool.size() > b.pool_cursor + V2_SCRATCH)
      prim_avail = b.backup_pool.size() - b.pool_cursor - V2_SCRATCH;
    size_t remaining = n_rounds;
    size_t take = std::min(remaining, prim_avail);
    for (size_t k = 0; k < take; k++)
      plan[bk].emplace_back(
          b.backup_pool[b.pool_cursor + primary_take[bk]++], (uint8_t)0);
    remaining -= take;
    if (remaining > 0 && overflow_subs) {
      // Primary subarray window for this bank (same math as the MM3D
      // refresh): extras overlapping it are scratch-only (their pools
      // share the primary window's row budget and use the relaxed
      // directed-edge screen) — never place resident data there.
      uint32_t prim_ws = (b.calib.open_rows[0] / 640) * 640;
      uint32_t prim_we = prim_ws + 640;
      if (const char* ss = getenv("PIM_SUB_START")) if (*ss) prim_ws = (uint32_t)atoi(ss);
      if (const char* se = getenv("PIM_SUB_END"))   if (*se) prim_we = (uint32_t)atoi(se);
      for (size_t ei = 0; ei < b.pool_extra.size() && remaining > 0; ei++) {
        // Extras without a real window (legacy shared-primary pools) are
        // NOT overflow targets — their rows are the primary's rows.
        if (!(b.pool_extra_win[ei].first < b.pool_extra_win[ei].second))
          continue;
        // Skip primary-window-overlapping extras (scratch-only).
        if (b.pool_extra_win[ei].first < prim_we &&
            b.pool_extra_win[ei].second > prim_ws)
          continue;
        size_t avail = 0;
        if (b.pool_extra[ei].size() > b.pool_extra_cursor[ei] + V2_SCRATCH)
          avail = b.pool_extra[ei].size() - b.pool_extra_cursor[ei] - V2_SCRATCH;
        size_t t = std::min(remaining, avail);
        for (size_t k = 0; k < t; k++)
          plan[bk].emplace_back(
              b.pool_extra[ei][b.pool_extra_cursor[ei] + extra_take[bk][ei]++],
              (uint8_t)(ei + 1));
        remaining -= t;
      }
    }
    if (remaining > 0) {
      fprintf(stderr, "[server] LOAD_WEIGHTS handle=%u: bank %d pool would "
              "overflow (cursor=%zu + n_rounds=%zu + v2_scratch=%zu > %zu%s) "
              "— sending ENOSPC ack\n",
              handle_id, b.bank_id,
              b.pool_cursor, n_rounds, V2_SCRATCH,
              b.backup_pool.size(),
              overflow_subs ? " incl. extra pools" : "");
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
  h.per_round_calib_sel.assign(n_rounds, std::vector<uint8_t>(N, 0));
  h.extra_refresh_wins.assign(N, {});
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
    // Overflow rows live in extra subarrays — remember those windows so
    // the MM3D refresh covers them as well.
    for (size_t ei = 0; ei < banks[bk].pool_extra.size(); ei++) {
      if (extra_take[bk].size() > ei && extra_take[bk][ei] > 0)
        h.extra_refresh_wins[bk].push_back(banks[bk].pool_extra_win[ei]);
    }
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
    uint32_t backup_row = plan[bk][round].first;
    uint8_t  calib_sel  = plan[bk][round].second;
    per_column_write_row(platform, banks[bk].bank_id, backup_row, mask);
    h.per_round_backup_rows[round][bk] = backup_row;
    h.per_round_calib_sel[round][bk]  = calib_sel;

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
    // O10: fused-colmask repair needs the unit masks at MM3D time.
    if (g_fused_colmask_any) keep_masks = true;
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
  // Commit the cursors (primary + any overflowed extras).
  for (int bk = 0; bk < N; bk++) {
    banks[bk].pool_cursor += primary_take[bk];
    for (size_t ei = 0; ei < extra_take[bk].size(); ei++)
      banks[bk].pool_extra_cursor[ei] += extra_take[bk][ei];
    if (overflow_subs && primary_take[bk] < n_rounds) {
      fprintf(stderr, "[server] LOAD_WEIGHTS handle=%u bank %d: overflowed "
              "%zu/%zu rounds into extra-subarray pools\n",
              handle_id, banks[bk].bank_id, n_rounds - primary_take[bk],
              n_rounds);
    }
  }

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
  // V2G (group-partial response, 2026-07-21): identical body and DRAM
  // work to V2, but the header carries a 6th field `group_chunks` and
  // the response is per-group int32 partial vectors
  // ([n_chunks/group_chunks][d_out]) instead of one summed [d_out].
  // Lets a group-scaled client (Bonsai g128) fetch a whole slice in ONE
  // round-trip and rescale host-side — removes the one-request-per-group
  // amplification. group_chunks == n_chunks reproduces V2 exactly.
  bool grouped = (magic == MAGIC_V2G || magic == MAGIC_V2GS);
  bool single = (magic == MAGIC_V2S || magic == MAGIC_V2GS);
  uint32_t group_chunks = n_chunks;
  if (grouped) {
    if (req_len < 6 * 4) {
      fprintf(stderr, "[server] V2G request too small (%zu B)\n", req_len);
      return -1;
    }
    rd_u32(group_chunks);
    if (group_chunks == 0 || n_chunks % group_chunks != 0) {
      fprintf(stderr, "[server] V2G bad group_chunks=%u (n_chunks=%u)\n",
              group_chunks, n_chunks);
      return -1;
    }
  } else if (magic != MAGIC_V2 && magic != MAGIC_V2S) {
    fprintf(stderr, "[server] bad magic 0x%x\n", magic);
    return -1;
  }
  if (group_chunks == 0) group_chunks = 1;
  const uint32_t n_groups = n_chunks / group_chunks;
  if (d_out != 2048) {
    fprintf(stderr, "[server] expected d_out=2048, got %u\n", d_out);
    return -1;
  }
  // D: optional calib_idx. The legacy V2 body had 5 header fields (20
  // bytes; V2G: 6 fields, 24 bytes); clients that want cross-calib voting
  // append a trailing u32 at the END of the body (after pos_mask +
  // neg_mask + x_bitplane + bp_factor). We detect by total size and read
  // it from the tail without advancing the in-band parse offset.
  size_t header_bytes = grouped ? (size_t)6*4 : (size_t)5*4;
  size_t mask_blocks = single ? 1 : 2;   // V2S: pos only
  size_t need_no_idx = header_bytes + (size_t)n_chunks*d_out*4*mask_blocks
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
  // No usable extras → every voting trip runs on the primary calib
  // (3 temporal noise samples still make a valid median). Also guards
  // the (idx-1) % cs_extra.size() UB when extras are empty.
  if (calib_idx > 0) {
    bool have_extras = true;
    for (const auto& b : banks) if (b.cs_extra.empty()) { have_extras = false; break; }
    if (!have_extras) calib_idx = 0;
  }

  // Slice into views.
  const uint32_t* pos_mask_all = (const uint32_t*)(req + off);
  off += (size_t)n_chunks * d_out * 4;
  const uint32_t* neg_mask_all = pos_mask_all;   // V2S: no neg block (unused)
  if (!single) {
    neg_mask_all = (const uint32_t*)(req + off);
    off += (size_t)n_chunks * d_out * 4;
  }
  const uint32_t* x_bitplane_all = (const uint32_t*)(req + off);
  off += (size_t)n_chunks * n_bitplanes * 4;
  const int32_t*  bitplane_factor = (const int32_t*)(req + off);

  const int N = (int)banks.size();
  // (chunk, sign) pairs; V2S enumerates chunks only (all sign 0).
  const size_t n_units = (size_t)n_chunks * (single ? 1 : 2);
  const size_t n_rounds = (n_units + N - 1) / N;       // # of N-bank executes per bitplane

  // PIM_ACCUM_XBP request-level eligibility (see the g_accxbp block).
  bool req_accxbp = g_accxbp > 0 && single && g_inline_bp == 1
                    && d_out <= 2048 && n_bitplanes <= 32
                    && !getenv("PIM_DEBUG_RX");
  int axb_neg[32] = {0}, axb_shift[32] = {0};
  if (req_accxbp) {
    for (uint32_t b = 0; req_accxbp && b < n_bitplanes; b++)
      if (!accxbp_encode(bitplane_factor[b], &axb_neg[b], &axb_shift[b]))
        req_accxbp = false;
    // fused per-row pc repair cannot ride the in-fabric sum
    if (req_accxbp && calib_idx == 0) {
      int fm = fused_coset_mode();
      if (fm == 1 || fm == 3)
        for (auto& bkk : banks)
          if (!bkk.fused_col_bad.empty()) { req_accxbp = false; break; }
    }
    // every round's units must land in ONE output-group slot
    for (size_t r = 0; req_accxbp && r < n_rounds; r++) {
      size_t u_lo = r * (size_t)N;
      size_t u_hi = std::min(n_units, u_lo + (size_t)N) - 1;
      if (u_lo / group_chunks != u_hi / group_chunks) req_accxbp = false;
    }
    if (!req_accxbp)
      fprintf(stderr, "[server] PIM_ACCUM_XBP: request ineligible — "
              "per-program readout fallback\n");
  }
  const long axb_skips0 = platform.oversize_skips();

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
  // V2G: y holds n_groups slots of d_out each; a unit's contribution
  // lands in its chunk's group. n_groups == 1 (plain V2) is the
  // historical single-accumulator behavior, bit for bit.
  vector<int32_t> y((size_t)n_groups * d_out, 0);
  for (size_t round = 0; round < n_rounds; round++) {
    // 1. Per-col write each active bank's backup row for this round.
    //    PIM_V2_PACK=1 packs the round's writes into ~3 IMEM-bounded
    //    programs (write-only, no c2h) instead of 12; instruction
    //    stream per chunk is byte-identical either way.
    int active_in_round = 0;
    std::vector<ScratchWrite> round_writes;
    for (int bk = 0; bk < N; bk++) {
      size_t u = round * (size_t)N + (size_t)bk;
      if (u >= n_units) break;
      uint32_t chunk = single ? (uint32_t)u : (uint32_t)(u / 2);
      int sign = single ? 0 : (int)(u % 2);
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
      // Draw from the V2 scratch reserve (pool tail) — never from rows
      // that LOAD_WEIGHTS handles hold resident weights on.
      size_t pool_idx = v2_pool_idx(banks[bk], calib_idx, round,
                                    pool_for_round.size());
      uint32_t scratch_row = pool_for_round[pool_idx];
      if (g_v2_pack > 0) {
        round_writes.push_back({banks[bk].bank_id, scratch_row, mask});
      } else {
        auto t0 = clk::now();
        per_column_write_row(platform, banks[bk].bank_id, scratch_row, mask);
        t_wcol_ns += std::chrono::duration_cast<ns_t>(clk::now() - t0).count();
        n_wcol_execs += 3;  // per_column_write_row issues 3 platform.execute calls
      }
      active_in_round++;
    }
    if (g_v2_pack > 0 && !round_writes.empty()) {
      auto t0 = clk::now();
      per_column_write_rows_packed(platform, round_writes, &n_wcol_execs);
      t_wcol_ns += std::chrono::duration_cast<ns_t>(clk::now() - t0).count();
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
          uint32_t chunk = single ? (uint32_t)u : (uint32_t)(u / 2);
          int sign = single ? 0 : (int)(u % 2);
          uint32_t xb = x_bitplane_all[(size_t)chunk * n_bitplanes + b];
          // D: choose calib + pool by request's calib_idx.
          const Calib& c = bc_calib_idx(banks[bk], calib_idx);
          const std::vector<uint32_t>& pool = bc_pool_idx(banks[bk], calib_idx);
          // Must match the scratch-write draw above: V2 reserve tail only.
          size_t pool_idx_mm3d = v2_pool_idx(banks[bk], calib_idx, round,
                                             pool.size());
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

      // V2 path selects calibs by calib_idx: fused only on the validated
      // primary (idx 0); cs_extra trips get the plain 11-wrRow body.
      // O4 (a): resident consts are NOT wired into the V2 path (scratch-
      // weight fallback traffic; its bodies keep wrRows — nullptr).
      g_fused_calib_ok = (calib_idx == 0);
      Program p = (g_parallel_banks
          ? build_multibank_parallel_program
          : build_multibank_combined_program)(
              ex_bank_ids, ex_backup_rows, ex_Rfirsts, ex_Rseconds,
              ex_open_rows, ex_x_patterns, label_base, nullptr);
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
      if (req_accxbp) {
        // In-fabric accumulation: enter the mode lazily (idempotent; the
        // entry clear runs before anything has accumulated), latch this
        // plane's ±2^shift, execute — and DON'T receive: the drain
        // happens once per round, after the plane loop.
        ensure_accxbp(platform, true);
        platform.set_acc_weight(axb_neg[bp_start], axb_shift[bp_start]);
        auto t_exec0 = clk::now();
        platform.execute(p);
        t_exec_ns += std::chrono::duration_cast<ns_t>(clk::now() - t_exec0).count();
        n_maj3_execs++;
        continue;
      }
      ensure_readback(platform, true);   // PIM_SEGPOP: matvec reads in SEG_POP
      auto t_exec0 = clk::now();
      platform.execute(p);
      t_exec_ns += std::chrono::duration_cast<ns_t>(clk::now() - t_exec0).count();
      n_maj3_execs++;

      static thread_local std::vector<uint8_t> rows_buf;
      size_t total_bytes = M * row_read_bytes();
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
          const uint8_t* row = rows_buf.data() + idx * row_read_bytes();
          vector<int> pc(d_out);
          row_pc(row, pc.data(), (int)d_out);
          // O10: host-repair fused-marginal columns. This V2 program ran
          // the fused layout iff calib_idx==0 (g_fused_calib_ok above)
          // and the coset mode is 1/3; the unit's mask is in-request.
          if (calib_idx == 0 && !banks[bk].fused_col_bad.empty()) {
            int fm = fused_coset_mode();
            if (fm == 1 || fm == 3) {
              size_t u2 = round * (size_t)N + (size_t)bk;
              // single-track: chunk = u2, always the pos mask (V2GS/V2S);
              // dual-track: chunk = u2/2, pos on even units / neg on odd.
              uint32_t ch2 = single ? (uint32_t)u2 : (uint32_t)(u2 / 2);
              const uint32_t* m2 = (single || (u2 % 2) == 0)
                  ? pos_mask_all + (size_t)ch2 * d_out
                  : neg_mask_all + (size_t)ch2 * d_out;
              fused_repair_pc(banks[bk], pc.data(), m2,
                              x_bitplane_all[(size_t)ch2 * n_bitplanes + b]);
            }
          }
          int sign_factor = (ex_signs[idx] == 0) ? +1 : -1;
          int weight = sign_factor * bitplane_factor[b];
          // V2G: route this unit's contribution into its chunk's group
          // slot (g == 0 always for plain V2, where group_chunks==n_chunks).
          // single-track: chunk = u_acc; dual-track: chunk = u_acc/2.
          size_t u_acc = round * (size_t)N + (size_t)bk;
          uint32_t chunk_acc = single ? (uint32_t)u_acc : (uint32_t)(u_acc / 2);
          size_t g_acc = (size_t)(chunk_acc / group_chunks);
          int32_t* y_g = y.data() + g_acc * d_out;
          for (uint32_t j = 0; j < d_out; j++) y_g[j] += weight * pc[j];
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

    // PIM_ACCUM_XBP: this round's planes accumulated in-fabric — ONE
    // drain returns the finished per-segment place-value sums.
    if (req_accxbp) {
      auto t_recv0 = clk::now();
      platform.flush_acc();
      static thread_local std::vector<int32_t> axb_acc(2048);
      int rc = platform.receiveData(axb_acc.data(), 8192);
      t_recv_ns += std::chrono::duration_cast<ns_t>(clk::now() - t_recv0).count();
      if (rc != 8192) {
        fprintf(stderr, "[server] ACCUM_XBP drain rc=%d expected=8192 "
                "(round=%zu)\n", rc, round);
        return -1;
      }
      auto t_pop0 = clk::now();
      // single-track: chunk == unit, and the whole round is one group
      // slot (request-gated) — route by the round's first unit.
      size_t g_acc = (round * (size_t)N) / group_chunks;
      int32_t* y_g = y.data() + g_acc * d_out;
      for (uint32_t j = 0; j < d_out; j++) y_g[j] += axb_acc[j];
      t_pop_ns += std::chrono::duration_cast<ns_t>(clk::now() - t_pop0).count();
    }
  }
  // PIM_ACCUM_XBP epilogue: an oversize-skipped program would have
  // silently dropped a whole plane from the in-fabric sums (the
  // accum-stream hazard class — see oversize_skips()) — fail loudly.
  // Then restore READ mode (SEG_POP re-arms on the next ensure_readback).
  if (req_accxbp) {
    if (platform.oversize_skips() != axb_skips0) {
      fprintf(stderr, "[server] ACCUM_XBP: oversize skips advanced "
              "mid-request — results incomplete, aborting\n");
      ensure_accxbp(platform, false);
      return -1;
    }
    ensure_accxbp(platform, false);
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

  // Write the response on the saved response_fd (NOT FD 1, which we
  // permanently redirected to stderr). Plain V2: 8192 B (int32 × 2048).
  // V2G: n_groups × 8192 B of per-group partials, group-major.
  ssize_t total = (ssize_t)n_groups * d_out * 4;
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
      std::set<std::pair<uint32_t,uint32_t>> extra_wins;
      for (const auto& kv : handles) {
        const LoadedHandle& lh = kv.second;
        if (bk < (int)lh.refresh_row_start.size()) {
          if (lh.refresh_row_start[bk] < mn) mn = lh.refresh_row_start[bk];
          if (lh.refresh_row_end[bk]   > mx) mx = lh.refresh_row_end[bk];
        }
        // LOAD-overflow: refresh the extra subarray windows too. Kept as
        // separate deduped entries — a min/max union across subarrays
        // 9k rows apart would refresh thousands of unrelated rows.
        if (bk < (int)lh.extra_refresh_wins.size())
          for (const auto& w : lh.extra_refresh_wins[bk])
            if (w.first < w.second) extra_wins.insert(w);
      }
      if (mn < mx) {
        ref_bank_ids.push_back(banks[bk].bank_id);
        ref_starts.push_back(mn);
        ref_ends.push_back(mx);
      }
      for (const auto& w : extra_wins) {
        if (mn < mx && w.first >= mn && w.second <= mx) continue;  // covered
        ref_bank_ids.push_back(banks[bk].bank_id);
        ref_starts.push_back(w.first);
        ref_ends.push_back(w.second);
      }
    }
    if (!ref_bank_ids.empty()) {
      Program rp = build_refresh_subarray_loop_program(
          ref_bank_ids, ref_starts, ref_ends);
      platform.execute(rp);
    }
    t_refresh_ns = std::chrono::duration_cast<ns_t>(clk::now() - t0).count();
  }

  // O4 (a) DRIFT FIX (addendum 23): re-write the resident const rows at
  // the start of the MM3D request so bodies clone from write-driven,
  // undrifted ONE/ZERO. Deliberately AFTER the refresh loop — refresh
  // restores charge of whatever content the rows currently hold; the
  // rewrite then restores the CONTENT. Shared by the serial and the
  // pack/parallel builders (both consume banks[].res_*_row below), so
  // this is the single rewrite site. Cadence: consts_rewrite_every().
  if (resident_consts_mode() > 0) {
    static long s_consts_seq = 0;
    const int rw_every = consts_rewrite_every();
    if (rw_every > 0 && (s_consts_seq % rw_every) == 0) {
      auto t0 = clk::now();
      bool did = rewrite_resident_const_rows(platform, banks);
      long long dt = std::chrono::duration_cast<ns_t>(clk::now() - t0).count();
      t_refresh_ns += dt;
      if (did) {
        static long s_rw_n = 0;
        s_rw_n++;
        if (s_rw_n <= 3 || s_rw_n % 1000 == 0)
          fprintf(stderr, "[res-consts-rw #%ld] req_seq=%ld every=%d "
                  "%.2f ms\n", s_rw_n, s_consts_seq, rw_every, dt / 1e6);
      }
    }
    s_consts_seq++;
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

  // ---------------------------------------------------------------------
  // O4 (b): T2-style round packing (PIM_PACK_ROUNDS > 1, MM3D only).
  // Bodies from up to N consecutive rounds (x the PIM_INLINE_BITPLANES
  // chunking) are queued and flushed as ONE program + ONE
  // receiveData(M x 8192 B) in emission order — fewer h2c round-trips
  // and fewer per-program overheads (where the O3-measured per-program
  // cost amortizes). Rounds are independent by construction: each body
  // RowClones its OWN resident pool row and runs on the same tuple —
  // exactly the shape PIM_INLINE_BITPLANES already chains K of inside
  // one round (silicon-validated to K=20 on the 8K IMEM, 2026-07-17).
  //
  // IMEM envelope: bodies are estimated conservatively per mode (O3
  // dump-measured serial figures + margin; a packed M != 4 program uses
  // the serial emitter regardless of PIM_PARALLEL_BANKS) and a flush is
  // forced BEFORE a queued batch would overflow BITSTREAM_IMEM.
  // Diagnostics that depend on per-round dispatch adjacency
  // (PIM_LOAD_REWRITE_ON_MM3D, PIM_VERIFY_AT_MM3D) force pack=1.
  int pack_rounds = g_pack_rounds;
  if (pack_rounds > 1 &&
      (g_load_rewrite ||
       (getenv("PIM_VERIFY_AT_MM3D") &&
        atoi(getenv("PIM_VERIFY_AT_MM3D")) > 0))) {
    static bool s_pack_warned = false;
    if (!s_pack_warned) {
      s_pack_warned = true;
      fprintf(stderr, "[server] PIM_PACK_ROUNDS=%d ignored: per-round "
              "rewrite/verify diagnostics need per-round dispatch\n",
              g_pack_rounds);
    }
    pack_rounds = 1;
  }
  struct PendBody {
    int bank_id;
    uint32_t backup_row, Rf, Rs;
    const uint32_t* orows;
    uint32_t xpat;
    int sign;
    uint32_t bp;
    uint32_t res_one, res_zero;
    // O10 fused-colmask repair: the unit's LOAD-time mask (nullptr when
    // unavailable) + owning bank config.
    const uint32_t* rep_mask = nullptr;
    const BankConfig* bcfg = nullptr;
  };
  std::vector<PendBody> pend;
  bool   pend_all_primary    = true;
  size_t pend_round_lo       = 0;
  size_t pend_rounds_spanned = 0;
  // Per-body IMEM estimates: O3 dump-measured serial figures (base 416,
  // fused 258 inst/body) + margin; EST_BODY_BASE also covers the mode-3
  // diagnostic (wrRows AND cosets). Resident-const bodies are SMALLER
  // (~213), so the plain-fused estimate only ever under-packs — safe.
  const int EST_BODY_FUSED = 270, EST_BODY_BASE = 460, EST_FIXED = 64;
  auto flush_pend = [&]() -> int {
    if (pend.empty()) return 0;
    const size_t M = pend.size();
    std::vector<int>             fx_bank_ids(M);
    std::vector<uint32_t>        fx_backup_rows(M);
    std::vector<uint32_t>        fx_Rfirsts(M);
    std::vector<uint32_t>        fx_Rseconds(M);
    std::vector<const uint32_t*> fx_open_rows(M);
    std::vector<uint32_t>        fx_x_patterns(M);
    std::vector<std::pair<uint32_t,uint32_t>> fx_consts(M);
    for (size_t i = 0; i < M; i++) {
      fx_bank_ids[i]    = pend[i].bank_id;
      fx_backup_rows[i] = pend[i].backup_row;
      fx_Rfirsts[i]     = pend[i].Rf;
      fx_Rseconds[i]    = pend[i].Rs;
      fx_open_rows[i]   = pend[i].orows;
      fx_x_patterns[i]  = pend[i].xpat;
      fx_consts[i]      = {pend[i].res_one, pend[i].res_zero};
    }
    g_fused_calib_ok = pend_all_primary;
    Program p = (g_parallel_banks
        ? build_multibank_parallel_program
        : build_multibank_combined_program)(
            fx_bank_ids, fx_backup_rows, fx_Rfirsts, fx_Rseconds,
            fx_open_rows, fx_x_patterns, label_base, &fx_consts);
    label_base += 2000 * (int)M + 1000;
    // Same dump env + file naming as the unpacked path (only one of the
    // two paths runs in a given process — env is fixed at startup).
    static int s_pack_dumped = 0;
    const char* pk_dump_n = getenv("PIM_DUMP_MM3D_PROGRAMS");
    int pk_max_dumps = pk_dump_n ? atoi(pk_dump_n) : 0;
    if (s_pack_dumped < pk_max_dumps) {
      s_pack_dumped++;
      char path[256];
      snprintf(path, sizeof(path), "/tmp/mm3d_program_dump_%d.txt",
               s_pack_dumped);
      FILE* fp = fopen(path, "w");
      if (fp) {
        uint64_t* iseq = (uint64_t*)p.get_inst_array();
        int n_inst = p.size() / 8;
        fprintf(fp, "# MM3D PACKED program: bodies=%zu rounds=[%zu..+%zu) "
                "backup_row0=%u\n", M, pend_round_lo, pend_rounds_spanned + 1,
                fx_backup_rows[0]);
        for (int i = 0; i < n_inst; i++)
          fprintf(fp, "[%4d]  %016lx\n", i, (unsigned long)iseq[i]);
        fclose(fp);
        free(iseq);
        fprintf(stderr, "[mm3d-dump #%d packed] bodies=%zu wrote %d insts "
                "to %s\n", s_pack_dumped, M, n_inst, path);
      }
    }
    ensure_readback(platform, true);   // PIM_SEGPOP: matvec reads in SEG_POP
    auto t_exec0 = clk::now();
    platform.execute(p);
    t_exec_ns += std::chrono::duration_cast<ns_t>(clk::now() - t_exec0).count();
    n_maj3_execs++;
    static thread_local std::vector<uint8_t> pk_rows_buf;
    size_t total_bytes = M * row_read_bytes();
    if (pk_rows_buf.size() < total_bytes) pk_rows_buf.resize(total_bytes);
    auto t_recv0 = clk::now();
    int rc = platform.receiveData(pk_rows_buf.data(), (int)total_bytes);
    t_recv_ns += std::chrono::duration_cast<ns_t>(clk::now() - t_recv0).count();
    if (rc != (int)total_bytes) {
      fprintf(stderr, "[server] MM3D packed receiveData rc=%d expected=%zu "
              "(bodies=%zu rounds=[%zu..+%zu))\n", rc, total_bytes, M,
              pend_round_lo, pend_rounds_spanned + 1);
      return -1;
    }
    auto t_pop0 = clk::now();
    // O10: this packed program ran the fused layout iff it was
    // all-primary (pend_all_primary -> g_fused_calib_ok at build) and
    // the coset mode is 1/3. Capture BEFORE pend state is reset below.
    const int fm_pk = fused_coset_mode();
    const bool fused_ran_pk = pend_all_primary && (fm_pk == 1 || fm_pk == 3);
    for (size_t i = 0; i < M; i++) {
      const uint8_t* row = pk_rows_buf.data() + i * row_read_bytes();
      vector<int> pc(d_out);
      row_pc(row, pc.data(), (int)d_out);
      if (fused_ran_pk && pend[i].bcfg && pend[i].rep_mask)
        fused_repair_pc(*pend[i].bcfg, pc.data(), pend[i].rep_mask,
                        pend[i].xpat);
      int sign_factor = (pend[i].sign == 0) ? +1 : -1;
      int weight = sign_factor * bitplane_factor[pend[i].bp];
      for (uint32_t j = 0; j < d_out; j++) y[j] += weight * pc[j];
    }
    t_pop_ns += std::chrono::duration_cast<ns_t>(clk::now() - t_pop0).count();
    pend.clear();
    pend_all_primary = true;
    pend_rounds_spanned = 0;
    return 0;
  };

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

    // O4 (b): packed path — enqueue this round's bodies; flush on
    // IMEM-fit or when the queued span reaches PIM_PACK_ROUNDS. The
    // enqueue math mirrors the unpacked path below body-for-body.
    if (pack_rounds > 1) {
      bool round_all_primary = true;
      if (!h.per_round_calib_sel.empty())
        for (int bk = 0; bk < active_in_round; bk++)
          if (h.per_round_calib_sel[round][bk] != 0)
            round_all_primary = false;
      for (uint32_t bp_start = 0; bp_start < n_bitplanes;
           bp_start += (uint32_t)g_inline_bp) {
        uint32_t K = std::min((uint32_t)g_inline_bp, n_bitplanes - bp_start);
        size_t M_next = (size_t)K * (size_t)active_in_round;
        bool new_all_primary = pend_all_primary && round_all_primary;
        int est = (new_all_primary && fused_coset_mode() == 1)
                  ? EST_BODY_FUSED : EST_BODY_BASE;
        if (!pend.empty() &&
            (pend.size() + M_next) * (size_t)est + (size_t)EST_FIXED >
                (size_t)g_bitstream_imem) {
          if (flush_pend() != 0) return -1;
        }
        if (pend.empty()) pend_round_lo = round;
        for (uint32_t kp = 0; kp < K; kp++) {
          uint32_t b = bp_start + kp;
          for (int bk = 0; bk < active_in_round; bk++) {
            size_t u = round * (size_t)N + (size_t)bk;
            uint32_t chunk = (uint32_t)(u / 2);
            int sign = (int)(u % 2);
            uint32_t xb = x_bitplane_all[(size_t)chunk * n_bitplanes + b];
            uint8_t sel = h.per_round_calib_sel.empty()
                ? 0 : h.per_round_calib_sel[round][bk];
            const Calib& uc = (sel == 0 ||
                               (size_t)(sel - 1) >= banks[bk].cs_extra.size())
                ? banks[bk].calib : banks[bk].cs_extra[sel - 1];
            PendBody pb;
            pb.bank_id    = banks[bk].bank_id;
            pb.backup_row = h.per_round_backup_rows[round][bk];
            pb.Rf         = uc.Rfirst;
            pb.Rs         = uc.Rsecond;
            pb.orows      = uc.open_rows.data();
            pb.xpat       = xb;
            pb.sign       = sign;
            pb.bp         = b;
            pb.res_one    = (sel == 0) ? banks[bk].res_one_row
                                       : RES_ROW_NONE;
            pb.res_zero   = (sel == 0) ? banks[bk].res_zero_row
                                       : RES_ROW_NONE;
            // O10 fused-colmask repair inputs (masks kept at LOAD when
            // any bank has a colmask; guard anyway).
            pb.bcfg = &banks[bk];
            pb.rep_mask =
                (!banks[bk].fused_col_bad.empty() &&
                 !h.all_round_masks.empty() &&
                 (size_t)bk < h.all_round_masks[round].size() &&
                 !h.all_round_masks[round][bk].empty())
                ? h.all_round_masks[round][bk].data() : nullptr;
            pend.push_back(pb);
          }
        }
        pend_all_primary = pend_all_primary && round_all_primary;
      }
      pend_rounds_spanned++;
      if (pend_rounds_spanned >= (size_t)pack_rounds) {
        if (flush_pend() != 0) return -1;
      }
      continue;  // next round — the unpacked path below is not taken
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
      std::vector<std::pair<uint32_t,uint32_t>> ex_consts;  // O4 (a)
      ex_bank_ids.reserve(M);
      ex_backup_rows.reserve(M);
      ex_Rfirsts.reserve(M);
      ex_Rseconds.reserve(M);
      ex_open_rows.reserve(M);
      ex_x_patterns.reserve(M);
      ex_signs.reserve(M);
      ex_consts.reserve(M);
      // LOAD-overflow: a unit whose row was allocated from an extra
      // subarray's pool MUST run on that subarray's calib — RowClone
      // scratch→Rfirst and the MAJ3 tuple are same-subarray operations.
      bool all_primary = true;
      for (uint32_t kp = 0; kp < K; kp++) {
        uint32_t b = bp_start + kp;
        for (int bk = 0; bk < active_in_round; bk++) {
          size_t u = round * (size_t)N + (size_t)bk;
          uint32_t chunk = (uint32_t)(u / 2);
          int sign = (int)(u % 2);
          uint32_t xb = x_bitplane_all[(size_t)chunk * n_bitplanes + b];
          uint8_t sel = h.per_round_calib_sel.empty()
              ? 0 : h.per_round_calib_sel[round][bk];
          const Calib& uc = (sel == 0 ||
                             (size_t)(sel - 1) >= banks[bk].cs_extra.size())
              ? banks[bk].calib : banks[bk].cs_extra[sel - 1];
          if (sel != 0) all_primary = false;
          ex_bank_ids.push_back(banks[bk].bank_id);
          ex_backup_rows.push_back(h.per_round_backup_rows[round][bk]);
          ex_Rfirsts.push_back(uc.Rfirst);
          ex_Rseconds.push_back(uc.Rsecond);
          ex_open_rows.push_back(uc.open_rows.data());
          ex_x_patterns.push_back(xb);
          ex_signs.push_back(sign);
          // O4 (a): resident consts are primary-calib only.
          ex_consts.emplace_back(
              sel == 0 ? banks[bk].res_one_row  : RES_ROW_NONE,
              sel == 0 ? banks[bk].res_zero_row : RES_ROW_NONE);
        }
      }
      if (getenv("PIM_DEBUG_RX")) {
        for (size_t i = 0; i < ex_backup_rows.size(); i++) {
          fprintf(stderr, "[mm3d-build] round=%zu bp=%u idx=%zu bk=%d backup_row=%u sign=%d Rf=%u\n",
                  round, bp_start, i, ex_bank_ids[i], ex_backup_rows[i], ex_signs[i], ex_Rfirsts[i]);
        }
      }
      // LOAD-mode builds on the primary calib unless this round holds
      // overflowed units — fused is validated only for the primary
      // (separated-generator) tuples, so any non-primary body in the
      // program forces the plain 11-wrRow variant for the whole program.
      // Set explicitly: a prior V2 request with calib_idx>0 must not
      // leave the flag sticky-false.
      g_fused_calib_ok = all_primary;
      Program p = (g_parallel_banks
          ? build_multibank_parallel_program
          : build_multibank_combined_program)(
              ex_bank_ids, ex_backup_rows, ex_Rfirsts, ex_Rseconds,
              ex_open_rows, ex_x_patterns, label_base, &ex_consts);
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
      ensure_readback(platform, true);   // PIM_SEGPOP: matvec reads in SEG_POP
      auto t_exec0 = clk::now();
      platform.execute(p);
      t_exec_ns += std::chrono::duration_cast<ns_t>(clk::now() - t_exec0).count();
      n_maj3_execs++;

      static thread_local std::vector<uint8_t> rows_buf;
      size_t total_bytes = M * row_read_bytes();
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
          const uint8_t* row = rows_buf.data() + idx * row_read_bytes();
          vector<int> pc(d_out);
          row_pc(row, pc.data(), (int)d_out);
          // O10: this program ran the fused layout iff all_primary
          // (g_fused_calib_ok at build) and the coset mode is 1/3.
          if (all_primary && !banks[bk].fused_col_bad.empty()) {
            int fm = fused_coset_mode();
            if ((fm == 1 || fm == 3) && !h.all_round_masks.empty() &&
                (size_t)bk < h.all_round_masks[round].size() &&
                !h.all_round_masks[round][bk].empty())
              fused_repair_pc(banks[bk], pc.data(),
                              h.all_round_masks[round][bk].data(),
                              ex_x_patterns[idx]);
          }
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

  // O4 (b): drain any bodies still queued when the round loop ends (the
  // request's tail rounds may not fill a whole PIM_PACK_ROUNDS span).
  if (pack_rounds > 1 && flush_pend() != 0) return -1;

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
//
// 2026-07-18 per-subarray screened pools: for NON-primary calibs
// (is_primary=false: the cs_extra voting calibs, legacy dual), if
// PIM_POOL_LIST_FILE_SUB is set (path pattern with {sub} and {bank}
// tokens, sub = open_rows[0]/640), the extra's pool comes from ITS OWN
// screened per-subarray file. The file must carry a "# window <start>
// <end>" comment naming the REAL subarray range (FindOpenRows windows
// are not 640-aligned on DIMM 2); rows are filtered against it and the
// calib's open set. A missing file means "this sub-cluster has no
// screened pool yet" → return empty, caller skips the extra (the
// 2026-07-18 scoping behavior). A present-but-malformed file is FATAL —
// explicit config that can't be honored must never silently degrade.
// When PIM_POOL_LIST_FILE_SUB is unset, extras fall through to the
// legacy PIM_POOL_LIST_FILE logic below (NOTE its known wart: an extra
// whose tuple sits INSIDE the env window — sub 71 on DIMM 2 — then
// shares the PRIMARY pool's rows, and its voting scratch draws cycle
// over LOAD-resident rows; set PIM_POOL_LIST_FILE_SUB in any
// LOAD-mode + voting production run to retire that hazard).
static std::vector<uint32_t> build_backup_pool(
    const Calib& c, bool is_primary,
    std::pair<uint32_t,uint32_t>* win_out = nullptr) {
  if (!is_primary) {
    if (const char* sub_pat = getenv("PIM_POOL_LIST_FILE_SUB")) {
      std::string path_str = sub_pat;
      char buf[16];
      size_t pos;
      snprintf(buf, sizeof(buf), "%u", c.open_rows[0] / 640);
      while ((pos = path_str.find("{sub}")) != std::string::npos)
        path_str.replace(pos, 5, buf);
      snprintf(buf, sizeof(buf), "%d", c.bank);
      while ((pos = path_str.find("{bank}")) != std::string::npos)
        path_str.replace(pos, 6, buf);
      FILE* fp = fopen(path_str.c_str(), "r");
      if (!fp) {
        fprintf(stderr, "[backup_pool] no per-sub pool file '%s' for calib "
                "(row %u, sub %u) — extra skipped\n",
                path_str.c_str(), c.open_rows[0], c.open_rows[0] / 640);
        return {};
      }
      std::vector<uint32_t> pool;
      uint32_t ws = 0, we = 0;
      char line[128];
      while (fgets(line, sizeof(line), fp)) {
        char* s = line;
        while (*s == ' ' || *s == '\t') s++;
        if (*s == '#') {
          unsigned a, b;
          if (sscanf(s, "# window %u %u", &a, &b) == 2) { ws = a; we = b; }
          continue;
        }
        if (*s == '\n' || *s == 0) continue;
        uint32_t r = (uint32_t)atoi(s);
        if (r > 0) pool.push_back(r);
      }
      fclose(fp);
      if (ws >= we) {
        fprintf(stderr, "[backup_pool] FATAL: per-sub pool file '%s' has no "
                "'# window <start> <end>' comment\n", path_str.c_str());
        exit(1);
      }
      if (!(c.open_rows[0] >= ws && c.open_rows[0] < we)) {
        fprintf(stderr, "[backup_pool] FATAL: per-sub pool file '%s' window "
                "[%u, %u) does not contain calib row %u — wrong file for "
                "this calib\n", path_str.c_str(), ws, we, c.open_rows[0]);
        exit(1);
      }
      std::set<uint32_t> open_set(c.open_rows.begin(), c.open_rows.end());
      std::vector<uint32_t> filtered;
      size_t out_of_win = 0;
      for (uint32_t r : pool) {
        if (r < ws || r >= we) { out_of_win++; continue; }
        if (open_set.count(r)) continue;
        filtered.push_back(r);
      }
      if (out_of_win > 0) {
        fprintf(stderr, "[backup_pool] FATAL: per-sub pool file '%s' has %zu "
                "rows outside its own window [%u, %u)\n",
                path_str.c_str(), out_of_win, ws, we);
        exit(1);
      }
      fprintf(stderr, "[backup_pool] per-sub pool '%s': %zu rows in window "
              "[%u, %u) for calib row %u\n", path_str.c_str(),
              filtered.size(), ws, we, c.open_rows[0]);
      if (win_out) *win_out = {ws, we};
      return filtered;  // may be empty → caller skips the extra
    }
  }
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
      // An explicitly configured layout that can't be honored is fatal.
      // The silent stride-based fallback here put production weights into
      // 25 unscreened rows outside the calibrated window (2026-07-17
      // regression: relative path + client cwd → ' the the the the').
      fprintf(stderr, "[backup_pool] FATAL: PIM_POOL_LIST_FILE='%s' "
              "unreadable. Relative paths resolve against the server's "
              "cwd — pass an absolute path.\n", path);
      exit(1);
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
      // The PIM_SUB_START/END window describes ONE subarray (the primary,
      // non-640-aligned s72 case). Apply it only to calibs whose rows lie
      // inside it. A calib OUTSIDE the window (a cs_extra candidate in
      // another subarray) has no rows in the explicit pool file at all —
      // filtering the file's rows into the WRONG subarray handed extras a
      // pool of s72 rows: cross-subarray RowClones (garbage votes) plus
      // scratch writes onto LOAD-resident weight rows, bypassing the V2
      // scratch reserve (2026-07-18 full-model regression, residual part).
      // For such calibs return empty: no screened pool exists, and the
      // stride fallback would invent unscreened rows — the caller skips
      // the extra instead.
      {
        uint32_t es = 0, ee = 0; bool env_win = false;
        if (const char* ss = getenv("PIM_SUB_START")) if (*ss) { es = (uint32_t)atoi(ss); env_win = true; }
        if (const char* se = getenv("PIM_SUB_END"))   if (*se) { ee = (uint32_t)atoi(se); }
        if (env_win) {
          if (any_open >= es && any_open < ee) {
            sub_start = es; sub_end = ee;
          } else {
            fprintf(stderr, "[backup_pool] calib (row %u) outside env window "
                    "[%u, %u) — no screened pool for its subarray, returning "
                    "empty\n", any_open, es, ee);
            return {};
          }
        }
      }
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

// ---------------------------------------------------------------------------
// O4 (a): resident-const selection (see resident_consts_mode's block
// comment for the model). Generator extraction + deposit math are exact
// per-calib — no hardcoded s72 bit rules.

// Derive the 4 tuple generators from open_rows and verify the tuple is
// separated-generator (sorted position index == generator bitmask:
// open_rows[i] == open_rows[0] ^ XOR of g[k] over set bits k of i).
// The fused body is gated to exactly these calibs, so consts inherit the
// same eligibility.
static bool tuple_generators(const Calib& c, uint32_t g[4]) {
  if (c.open_rows.size() != 16) return false;
  for (int k = 0; k < 4; k++) g[k] = c.open_rows[1u << k] ^ c.open_rows[0];
  for (int i = 0; i < 16; i++) {
    uint32_t want = c.open_rows[0];
    for (int k = 0; k < 4; k++)
      if (i & (1 << k)) want ^= g[k];
    if (c.open_rows[i] != want) return false;
  }
  return true;
}

// Deposit set of an external RowClone src -> open_rows[dst_idx]
// (pair-lattice law, test_safe_load.cpp): rows {src ^ S : S subset of
// bits(src ^ dst)}. A tuple row open_rows[dst_idx ^ j] is hit iff the
// span element e_j (XOR of generators in combo j) is a bit-subset of
// (src ^ dst) — then S = (src ^ dst) ^ e_j deposits src's pattern there.
// Returns the bitmask of tuple indices hit BEYOND the target itself.
static uint16_t clone_tuple_deposits(uint32_t src, const Calib& c,
                                     const uint32_t g[4], int dst_idx) {
  uint32_t d = src ^ c.open_rows[dst_idx];
  uint16_t hit = 0;
  for (int j = 1; j < 16; j++) {
    uint32_t e = 0;
    for (int k = 0; k < 4; k++)
      if (j & (1 << k)) e ^= g[k];
    if (e != 0 && (e & d) == e)
      hit |= (uint16_t)(1u << (dst_idx ^ j));
  }
  return hit;
}

// Select + claim + write the per-bank resident constant rows. Runs once
// at startup, AFTER pools are built and the platform is up, BEFORE any
// request (so the claim precedes every LOAD allocation). No-op unless
// PIM_RESIDENT_CONSTS is set. Selection failure on a bank leaves that
// bank on wrRows (RES_ROW_NONE) — never fatal.
static void setup_resident_consts(SoftMCPlatform& platform,
                                  std::vector<BankConfig>& banks) {
  const int rc_mode = resident_consts_mode();
  if (rc_mode <= 0) return;
  const size_t rsv = v2_scratch_reserve();
  static std::vector<uint32_t> ones_mask(2048, 0xFFFFFFFFu);
  static std::vector<uint32_t> zeros_mask(2048, 0x00000000u);
  for (auto& b : banks) {
    uint32_t g[4];
    if (!tuple_generators(b.calib, g)) {
      fprintf(stderr, "[res-consts] bank %d: primary tuple is not "
              "separated-generator — consts DISABLED for this bank\n",
              b.bank_id);
      continue;
    }
    // Scan only the pool front — the tail rsv rows stay V2 scratch.
    size_t scan_end = b.backup_pool.size() > rsv + 2
                      ? b.backup_pool.size() - rsv : 0;
    if (scan_end < 3) {
      fprintf(stderr, "[res-consts] bank %d: pool too small (%zu rows, "
              "v2_scratch=%zu) — consts DISABLED for this bank\n",
              b.bank_id, b.backup_pool.size(), rsv);
      continue;
    }
    auto usable = [&](uint32_t r) {
      return r != b.calib.Rfirst && r != b.calib.Rsecond;
    };
    // Allowed off-target deposit indices for the ZERO clones: op[0]
    // (erased by the ONE fill emitted after them) + the zero rows of the
    // fused layout ({2,6,10,14} coset + {8}). Zeros onto zero rows are
    // no-ops; anything else (W rows {3,7,11,12,15}, x rows {1,5,9,13,4})
    // disqualifies the candidate.
    const uint16_t zero_ok = (1u << 0) | (1u << 2) | (1u << 6) |
                             (1u << 8) | (1u << 10) | (1u << 14);
    size_t zi = (size_t)-1;
    for (size_t i = 0; i < scan_end; i++) {
      uint32_t r = b.backup_pool[i];
      if (!usable(r)) continue;
      if ((clone_tuple_deposits(r, b.calib, g, 2) & ~zero_ok) == 0 &&
          (clone_tuple_deposits(r, b.calib, g, 8) & ~zero_ok) == 0) {
        zi = i;
        break;
      }
    }
    if (zi == (size_t)-1) {
      fprintf(stderr, "[res-consts] bank %d: no deposit-safe ZERO source in "
              "%zu scanned pool rows — consts DISABLED for this bank\n",
              b.bank_id, scan_end);
      continue;
    }
    const uint32_t zrow = b.backup_pool[zi];
    // ONE source (mode 1 only): strictly deposit-free into op[0] — it is
    // the LAST const fill, nothing may repair its off-target deposits.
    // Also reject pairs whose clones could deposit on EACH OTHER:
    // "other = src ^ S" requires (src ^ other) subset of bits(src ^ dst).
    size_t oi = (size_t)-1;
    if (rc_mode == 1) {
      const uint32_t d_z2 = zrow ^ b.calib.open_rows[2];
      const uint32_t d_z8 = zrow ^ b.calib.open_rows[8];
      for (size_t i = 0; i < scan_end; i++) {
        if (i == zi) continue;
        uint32_t r = b.backup_pool[i];
        if (!usable(r)) continue;
        if (clone_tuple_deposits(r, b.calib, g, 0) != 0) continue;
        const uint32_t pd = r ^ zrow;
        const uint32_t d_one = r ^ b.calib.open_rows[0];
        if ((pd & d_one) == pd) continue;  // ONE clone could hit zero row
        if ((pd & d_z2) == pd || (pd & d_z8) == pd) continue;  // and back
        oi = i;
        break;
      }
      if (oi == (size_t)-1)
        fprintf(stderr, "[res-consts] bank %d: no deposit-free ONE source — "
                "zeros-only for this bank (ONE stays a wrRow)\n", b.bank_id);
    }
    // Claim: remove the const rows from the pool so LOAD handles and the
    // V2 scratch tail can never land on them. Erase higher index first.
    b.res_zero_row = zrow;
    if (oi != (size_t)-1) b.res_one_row = b.backup_pool[oi];
    size_t hi = (oi == (size_t)-1) ? zi : std::max(zi, oi);
    size_t lo = (oi == (size_t)-1) ? zi : std::min(zi, oi);
    b.backup_pool.erase(b.backup_pool.begin() + hi);
    if (hi != lo) b.backup_pool.erase(b.backup_pool.begin() + lo);
    // Write the constants once. They are ordinary pool-subarray rows, so
    // the MM3D-entry refresh loop ACT-refreshes them with everything else.
    per_column_write_row(platform, b.bank_id, b.res_zero_row,
                         zeros_mask.data());
    if (b.res_one_row != RES_ROW_NONE)
      per_column_write_row(platform, b.bank_id, b.res_one_row,
                           ones_mask.data());
    fprintf(stderr, "[res-consts] bank %d: ZERO=%u ONE=%s "
            "(pool now %zu rows; d(z,op2)=0x%x d(z,op8)=0x%x)\n",
            b.bank_id, b.res_zero_row,
            b.res_one_row == RES_ROW_NONE
                ? "wrRow" : std::to_string(b.res_one_row).c_str(),
            b.backup_pool.size(),
            zrow ^ b.calib.open_rows[2], zrow ^ b.calib.open_rows[8]);
  }
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
    load_fused_colmask(bc);   // O10: PIM_FUSED_COLMASK_FILE, {bank} token
    bc.backup_pool = build_backup_pool(bc.calib, /*is_primary=*/true);
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
      // O9 (2026-07-20): extras cap env-gated. Default 4 = historic
      // behavior byte-for-byte; the new-subarray residency pilot raises
      // it so freshly-calibrated clusters join LOAD-overflow.
      static int s_max_extras = -1;
      if (s_max_extras < 0) {
        const char* v = getenv("PIM_MAX_EXTRAS");
        s_max_extras = (v && *v) ? atoi(v) : 4;
        if (s_max_extras != 4)
          fprintf(stderr, "[server] PIM_MAX_EXTRAS=%d\n", s_max_extras);
      }
      for (auto& [cnt, sub] : ranked) {
        if ((int)bc.cs_extra.size() >= s_max_extras) break;
        // Pick the first calib in cs whose open_rows[0] falls in this sub.
        for (const auto& c : cs) {
          uint32_t s = (c.open_rows[0] / 640) * 640;
          if (s != sub) continue;
          std::pair<uint32_t,uint32_t> win{0, 0};
          std::vector<uint32_t> pool_i =
              build_backup_pool(c, /*is_primary=*/false, &win);
          if (pool_i.empty()) break;
          bc.cs_extra.push_back(c);
          bc.pool_extra.push_back(std::move(pool_i));
          bc.pool_extra_win.push_back(win);
          bc.pool_extra_cursor.push_back(0);
          break;
        }
      }
      fprintf(stderr, "[server] bank %d: %zu extra calibs (dense clusters)",
              bk, bc.cs_extra.size());
      for (size_t ei = 0; ei < bc.cs_extra.size(); ei++)
        fprintf(stderr, " sub=%u(pool=%zu@[%u,%u))",
                (bc.cs_extra[ei].open_rows[0] / 640),
                bc.pool_extra[ei].size(),
                bc.pool_extra_win[ei].first, bc.pool_extra_win[ei].second);
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
          bc.backup_pool_b = build_backup_pool(bc.calib_b, /*is_primary=*/false);
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

  // O4 (a): claim + write the resident constant rows (no-op unless
  // PIM_RESIDENT_CONSTS is set). Must precede the first LOAD_WEIGHTS so
  // the pool claim happens before any handle allocation.
  setup_resident_consts(platform, banks);

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
  // Request-path profiling (2026-07-21): the SEG_POP full-model A/B
  // showed the fused-V2S wall is only ~half program round-trips.
  // Decompose the server's view of the remainder per request:
  //   gap     = blocking on the next request HEADER (client think/build
  //             time between requests — body-concat, python, etc.)
  //   body    = streaming the request payload through the pipe
  //   handler = process_* wall (includes programs; subtract the
  //             srv-prof exec/recv/wcol/pop terms for pure overhead)
  using rclk = std::chrono::steady_clock;
  long long rq_n = 0; double rq_gap_s = 0, rq_body_s = 0, rq_h_s = 0;
  while (true) {
    uint32_t req_len = 0;
    auto t_g0 = rclk::now();
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
    auto t_b0 = rclk::now();
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
    auto t_h0 = rclk::now();
    int rc;
    if (magic == MAGIC_V2 || magic == MAGIC_V2G || magic == MAGIC_V2S
        || magic == MAGIC_V2GS) {
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
    auto t_h1 = rclk::now();
    rq_n++;
    rq_gap_s  += std::chrono::duration<double>(t_b0 - t_g0).count();
    rq_body_s += std::chrono::duration<double>(t_h0 - t_b0).count();
    rq_h_s    += std::chrono::duration<double>(t_h1 - t_h0).count();
    if (rq_n % 2000 == 0) {
      fprintf(stderr, "[req-prof #%lld] avg/req last 2000: gap=%.2fms "
              "body=%.2fms handler=%.2fms (gap=client-side think/build)\n",
              rq_n, rq_gap_s * 1e3 / 2000, rq_body_s * 1e3 / 2000,
              rq_h_s * 1e3 / 2000);
      rq_gap_s = rq_body_s = rq_h_s = 0;
    }
  }
  _exit(0);
}
