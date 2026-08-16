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
#include "desc_walk_body.h"   // Task #50 M3: the walk-compatible MAJ3 body emitter

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <set>
#include <map>
#include <tuple>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>
#include <deque>

using namespace std;

// build43 (2026-08-01): the ONE expected trailer/recorder build-tag for the
// build-43 image (mig_reinit_ctrl recovery-LOCKOUT fix for defect-B Mismatch A;
// readback_engine magic 0xDBC0DE27 -> 0xDBC0DE28). Defined once and used
// everywhere a build-tag equality is checked (the recorder-dump decoder and the
// swap-storm mode) so a magic bump is a one-line change, never a scatter of
// hardcoded literals. On a pre-flash tower still running the 0x27 image this
// binary only WARNS "image mismatch" (the compute path is tag-independent).
static const uint32_t PIM_EXPECTED_BUILD_TAG = 0xDBC0DE28u;

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
// [#65 2026-08-04] Runtime CONFIG-UPDATE request (host reconfigures the bank
// set / per-bank state / windows WITHOUT restarting the server — DESIGN LAW:
// "everything can be configured by the host on power up, and can be changed
// later"). Wire format + opcodes at handle_config_request().
static constexpr uint32_t MAGIC_CONFIG = 0xB17EF00Au;
static constexpr uint32_t CFG_QUERY     = 1u;  // read the per-bank config table
static constexpr uint32_t CFG_RECONFIG  = 2u;  // replace the bank set (full)
static constexpr uint32_t CFG_SET_STATE = 3u;  // transition named banks' state
static constexpr uint32_t MAGIC_FUSE = 0xB17EF008u;   // P1 RUNG-2 request fusion
// (2026-08-02): a pure DELIVERY-FRAMING wrapper. Body = [MAGIC_FUSE][u32 K]
// then K× ([u32 sublen][subbody]); each subbody is a COMPLETE, unmodified
// request of any other magic (V2/V2G/V2S/V2GS/LOAD/MM3D). The server loops the
// UNCHANGED per-request dispatch over the K sub-bodies in order; each sub's
// response is written to response_fd back-to-back, so the client reads the K
// sub-responses concatenated (each its own natural length). The per-bank DDR
// command stream is byte-IDENTICAL to sending the K requests separately
// (trace-equiv L3 by construction) — the ONLY thing removed is (K-1) pipe
// read_exact round-trips + the server's per-request header-block wait (the
// per-request-fixed `gap`). Client-gated by PIM_REQ_FUSE=N; default off / K=1
// path is never emitted, so the server is byte-identical when unused.

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

// fwd decl (defined with the Rung-1 stream-session block below): stream
// inside an open session, legacy execute otherwise.
static inline void pexec(SoftMCPlatform& platform, Program& p,
                         int payload_bytes);


// M3 PIM_BCAST_LOAD (coset-deposit / write-elision lever) REMOVED 2026-07-28 --
// resolved NEGATIVE: MAJ3 body never reads the shadow, 99.4% of deposits were
// d=0 no-ops, elision served 87% stale masks. See memory m3-coset-deposit-negative
// + e2e_sim_2026_07_24/M3_UNDERSTANDING_2026_07_26.md.

static void per_column_write_row(SoftMCPlatform& platform, int bank_id,
                                  uint32_t row, const uint32_t* data_2048) {
  int col_start = 0;
  for (int chunk = 0; chunk < 3; chunk++) {
    int n_cols = CHUNK_COLS[chunk];
    Program p = build_chunk_program(bank_id, row,
                                     data_2048 + col_start * 16,
                                     col_start, n_cols);
    pexec(platform, p, 0);   // write-only: bare-trailer record in a stream
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
struct ScratchWrite { int bank_id; uint32_t row; const uint32_t* data;
                      // LEVERS #28 X-MASTER: uniform single-wrRow fill (the
                      // master row holds one activation bitplane xb, not a
                      // 2048-col mask; .data is ignored when uniform).
                      bool uniform = false; uint32_t uniform_val = 0;
                      int uniform_label = 0; };
// LEVERS #28: uniform whole-row fill (one value across all 2048 cols) via the
// same write-driven wrRow primitive rewrite_resident_const_rows uses. Self-
// contained (sets CASR + BAR) so it packs alongside chunk bodies. One wrRow
// == one master write (the §0 accounting unit).
static void emit_uniform_row(Program& p, int bank_id, uint32_t row,
                             uint32_t val, int label) {
  p.add_inst(SMC_LI(8, CASR));
  p.add_inst(SMC_LI(bank_id, BAR));
  p.add_below(wrRow_immediate_label(BAR, row, val, label));
  p.add_inst(SMC_SLEEP(8));           // tWR guard before PRE / next body
  p.add_below(PRE(BAR, 0, 0));
  p.add_inst(SMC_SLEEP(6));
}
static void per_column_write_rows_packed(SoftMCPlatform& platform,
                                         const std::vector<ScratchWrite>& ws,
                                         int* n_execs_out) {
  Program p;
  bool empty = true;
  for (const auto& w : ws) {
    if (w.uniform) {
      // LEVERS #28: a uniform master fill is ~40 insts; flush if it would
      // overflow the IMEM envelope, then emit it as one write-driven wrRow.
      if (!empty && p.size() / 8 + 200 > 7600) {
        p.add_inst(SMC_END());
        pexec(platform, p, 0);
        (*n_execs_out)++;
        p = Program();
        empty = true;
      }
      emit_uniform_row(p, w.bank_id, w.row, w.uniform_val, w.uniform_label);
      empty = false;
      continue;
    }
    int col_start = 0;
    for (int chunk = 0; chunk < 3; chunk++) {
      int n_cols = CHUNK_COLS[chunk];
      // a chunk body is ~1.5K insts; flush before it would overflow.
      if (!empty && p.size() / 8 + 1600 > 7600) {
        p.add_inst(SMC_END());
        pexec(platform, p, 0);
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
    pexec(platform, p, 0);
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
static bool g_mode_accxbp_now = false;  // host view: engine in ACCUM_XBP
static void ensure_readback(SoftMCPlatform& platform, bool segpop) {
  // Leaving ACCUM_XBP is LAZY: the engine stays in accxbp across
  // eligible requests (a per-request exit costs ~1.5 s of drain_stray
  // plus the next execute's 500 ms transition-drain floor — measured as
  // the Arm-B ~10x tax, 2026-07-22). Only a consumer that actually
  // needs READ/SEG_POP framing pays the one-time transition here.
  if (g_mode_accxbp_now) {
    platform.set_readback_mode(false);
    platform.set_readback_mode(false);
    platform.drain_stray(1500, 8);
    g_mode_accxbp_now = false;
    g_mode_segpop_now = false;
  }
  if (g_segpop <= 0) return;            // feature off/unparsed: never switch
  if (segpop == g_mode_segpop_now) return;
  if (segpop) { platform.set_readback_mode_segpop(); platform.set_readback_mode_segpop(); }
  else        { platform.set_readback_mode(false);   platform.set_readback_mode(false); }
  g_mode_segpop_now = segpop;
}
static inline size_t row_read_bytes() { return g_segpop > 0 ? 2048u : 8192u; }

// ---- Rung-1 producer loop (PIM_STREAM=1; build-9+ image ONLY, trailer
// magic 0xDBC0DE08 — the STREAM_EN word corrupts IMEM on older images).
// One stream session per eligible request: write programs stream with
// payload 0 (bare trailers), exec programs with their exact receive
// size; receiveData pops from the same queue the persistent drain
// fills, so recv sites are unchanged. Mode SETs happen BEFORE the
// session opens (a mid-pipeline mode flip would change in-flight
// programs' framing). accxbp requests keep the legacy cadence.
static int  g_stream = -1;              // env PIM_STREAM, resolved once
static bool g_stream_session = false;   // a session is open NOW
// PIM_STREAM_ALTERNATE=1 (2026-07-23 DIAGNOSTIC): the SAME-PROCESS
// per-request A/B. With PIM_STREAM=1, V2-family requests alternate
// legacy/streamed by arrival index. Replaying every captured request
// TWICE makes consecutive twins hit the two paths inside ONE process —
// the only comparison that sits below the cross-process odd-segment
// floor (two separate processes differ on ~all odd elements even
// legacy-vs-legacy; see BUILD10_VERIFICATION 07-23).
static int  g_stream_alternate = -1;
static long g_v2_req_counter = 0;
static bool stream_alternate() {
  if (g_stream_alternate < 0) {
    const char* v = getenv("PIM_STREAM_ALTERNATE");
    g_stream_alternate = (v && *v) ? atoi(v) : 0;
    if (g_stream_alternate > 0)
      fprintf(stderr, "[server] PIM_STREAM_ALTERNATE=1: V2 requests "
              "alternate legacy/streamed (same-process A/B)\n");
  }
  return g_stream_alternate > 0;
}
static bool stream_on() {
  if (g_stream < 0) {
    const char* v = getenv("PIM_STREAM");
    g_stream = (v && *v) ? atoi(v) : 0;
    // [seamfix 2026-08-04] THE desc-serve<->V2 SEAM CRASH FIX. Rung-1 streaming
    // turns on the FPGA STREAM_EN egress (set_stream_en, "Set STREAM_EN: on"),
    // which frames ALL readback. That egress is INCOMPATIBLE with the desc-serve
    // <-> V2 mixed regime: once a V2 fallback slice runs after a desc-serve frame,
    // its readback strands past the seam (silicon: the streaming drain waits on a
    // record tail that never surfaces, rec_off=32128/PAY=32768; the execute drain
    // blocks in the kernel c2h read) and the stuck drain POISONS the platform --
    // SILENTLY via stream_stop()'s join-guard if V2 streams (platform.cpp:579, the
    // one poison site with no log line), LOUDLY via execute()'s join-guard if it
    // does not -> the deterministic full-token crash ("PIM server closed stdout").
    // Rung-1 streaming is WALL-NEUTRAL for the desc-serve engine (LEVERS C01 /
    // memory rung1_streaming_validated; the NS battery and deep-2560 oracle are
    // measured with it OFF and are unchanged by this gate), so force it OFF
    // whenever the desc-serve engine is active. PIM_STREAM=0-proven bit-exact and
    // crash-free over 2000+ mixed requests in the same full-model run. Pure-V2 (no
    // desc-serve) is unaffected. Override for A/B with PIM_STREAM_FORCE=1.
    const bool desc_serve_on = []{ const char* e = getenv("PIM_DESC_SERVE");
                                   return e && *e && atoi(e) > 0; }();
    const bool stream_force  = []{ const char* e = getenv("PIM_STREAM_FORCE");
                                   return e && *e && atoi(e) > 0; }();
    if (g_stream > 0 && desc_serve_on && !stream_force) {
      fprintf(stderr, "[server] PIM_STREAM forced OFF under PIM_DESC_SERVE "
              "(Rung-1 STREAM_EN egress strands V2 reads across the desc-serve"
              " seam; wall-neutral for the engine). Set PIM_STREAM_FORCE=1 to"
              " override.\n");
      g_stream = 0;
    }
    if (g_stream > 0 && getenv("PIM_BACKEND") &&
        std::string(getenv("PIM_BACKEND")) == "sim") {
      fprintf(stderr, "[server] PIM_STREAM forced OFF under "
              "PIM_BACKEND=sim (the sim cannot stream)\n");
      g_stream = 0;
    }
    if (g_stream > 0)
      fprintf(stderr, "[server] PIM_STREAM=1: per-request stream "
              "sessions (Rung-1 producer loop)\n");
  }
  return g_stream > 0;
}
// PIM_STREAM_SCOPE (2026-07-23 DIAGNOSTIC): stream only ONE program
// class to isolate which class's streaming triggers the odd-byte
// corruption. "wcol" = only the payload-0 scratch writes stream (exec/
// readout legacy); "exec" = only the exec/readout programs stream
// (writes legacy). Sessions become per-burst (churn is E8-clean); a
// legacy execute NEVER runs inside an open session (the platform guard
// stays honest). Default/unset = "all" (the production request-scoped
// session, unchanged).
static int  g_stream_scope = -1;        // 0 all, 1 wcol, 2 exec
static bool g_stream_requested = false; // this request is stream-eligible
static int stream_scope() {
  if (g_stream_scope < 0) {
    const char* v = getenv("PIM_STREAM_SCOPE");
    g_stream_scope = 0;
    if (v && strcmp(v, "wcol") == 0) g_stream_scope = 1;
    else if (v && strcmp(v, "exec") == 0) g_stream_scope = 2;
    if (g_stream_scope)
      fprintf(stderr, "[server] PIM_STREAM_SCOPE=%s: DIAGNOSTIC per-burst "
              "sessions — only that class streams\n", v);
  }
  return g_stream_scope;
}
static void scoped_stream_transition(SoftMCPlatform& pf, bool want_open) {
  if (want_open == g_stream_session) return;
  if (want_open) {
    ensure_readback(pf, true);          // pipeline empty between bursts
    pf.stream_start(SoftMCPlatform::STREAM_SIZED);
    g_stream_session = true;
  } else {
    pf.stream_stop();
    g_stream_session = false;
  }
}
// Send-or-execute: inside an open session, stream with the program's
// exact expected payload; otherwise the pristine legacy execute.
// Under a scoped diagnostic, payload==0 identifies the wcol class.
static inline void pexec(SoftMCPlatform& platform, Program& p,
                         int payload_bytes) {
  if (g_stream_requested && stream_scope()) {
    bool is_wcol = (payload_bytes == 0);
    bool stream_this = (stream_scope() == 1) ? is_wcol : !is_wcol;
    scoped_stream_transition(platform, stream_this);
    if (stream_this) platform.stream_send(p, payload_bytes);
    else             platform.execute(p);
    return;
  }
  // 2026-07-26 REFRESH-STARVATION FIX. Maintenance flows THROUGH fetch and
  // tPRDI (~1 us) brackets every HOST ROUND-TRIP. Streaming exists precisely to
  // remove round-trips -- so an unbroken stream removes the maintenance
  // brackets, resident weight rows go unrefreshed and DECAY. Observed on
  // silicon (PIM_STREAM=1): mm3d-verify DECAY/CORRUPTION growing with handle
  // index, 21.8% -> 89.9% of segs; invisible in Verilator, which has no
  // retention model.
  // Fix: yield to maintenance on a bounded cadence -- every Nth program takes
  // the legacy execute() path, costing one round-trip but guaranteeing a
  // maintenance window. N-1 of every N programs still stream.
  if (g_stream_session) {
    static const long s_yield_every = []{
      const char* v = getenv("PIM_STREAM_MAINT_EVERY");
      long n = (v && *v) ? atol(v) : 32;
      return n > 0 ? n : 0;          // 0 disables the yield (old behaviour)
    }();
    static long s_streamed = 0;
    if (s_yield_every && ++s_streamed >= s_yield_every) {
      s_streamed = 0;
      // Mixing execute() into an OPEN stream session is illegal (the platform
      // is in stream mode; doing so closes the server). Properly close the
      // session, take the legacy round-trip so maintenance gets its window,
      // then reopen -- leaving g_stream_session exactly as we found it.
      platform.stream_stop();
      platform.execute(p);
      ensure_readback(platform, true);      // pipeline empty before reopening
      platform.stream_start(SoftMCPlatform::STREAM_SIZED);
      return;
    }
    platform.stream_send(p, payload_bytes);
  } else {
    platform.execute(p);
  }
}
struct StreamSession {
  SoftMCPlatform& pf;
  bool open = false;
  bool scoped = false;
  StreamSession(SoftMCPlatform& p, bool want, bool segpop_mode) : pf(p) {
    if (!want) return;
    if (stream_scope()) {               // diagnostic: no request session;
      g_stream_requested = true;        // pexec brackets per-burst sessions
      scoped = true;
      return;
    }
    ensure_readback(pf, segpop_mode);   // set mode while pipeline empty
    pf.stream_start(SoftMCPlatform::STREAM_SIZED);
    g_stream_session = true;
    open = true;
  }
  ~StreamSession() {
    if (scoped) {
      scoped_stream_transition(pf, false);  // close any dangling burst
      g_stream_requested = false;
      return;
    }
    if (!open) return;
    pf.stream_stop();
    g_stream_session = false;
  }
};
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

// PIM_SKIP_ZERO_PLANES (default OFF; A/B-gated, default flipped only after
// parity gates pass): a MAJ3 body whose activation word x==0 contributes
// popcount(W AND 0)==0 to the accumulation, so eliding its emission (body
// wrRows + exec + readback) is exactness-preserving by identity — a skipped
// body contributes the same zero it would have added. Read once. Callers
// additionally require fused_coset_mode()==0: the fused body variant has a
// hard 5/5/5 shape rule that forbids per-program body-count changes, so the
// skip applies ONLY on the plain (non-fused) body builder.
static bool skip_zero_planes_on() {
  static const bool on = []{
    const char* v = getenv("PIM_SKIP_ZERO_PLANES");
    bool e = (v && atoi(v) > 0);
    if (e)
      fprintf(stderr, "[server] PIM_SKIP_ZERO_PLANES=1: eliding zero-x MAJ3 "
                      "bodies (exactness-preserving; plain body builder only)\n");
    return e;
  }();
  return on;
}

// Cumulative zero-plane elision census (PIM_SKIP_ZERO_PLANES). The throttled
// [zskip] line lets a gate confirm the elision path is live and nonzero;
// single-threaded request loop makes the file-scope counters safe.
static long g_zskip_skipped = 0;
static long g_zskip_total   = 0;
static void zskip_report() {
  static long s_reports = 0;
  s_reports++;
  if (s_reports <= 3 || s_reports % 50 == 0)
    fprintf(stderr, "[zskip] cumulative skipped=%ld/%ld units (%.1f%%)\n",
            g_zskip_skipped, g_zskip_total,
            g_zskip_total ? 100.0 * g_zskip_skipped / g_zskip_total : 0.0);
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

// ---------------------------------------------------------------------------
// LEVERS #28 X-MASTER-CLONE (PIM_XMASTER, default OFF — emission
// byte-identical when unset). The fused body re-establishes the activation
// x-plane every execute with two 128-column WRITE loops (open_rows[1] and
// open_rows[4]); those are the x-side twin of the O4(a) resident consts.
// When enabled, per (round, bank, plane) MASTER rows hold the activation
// bitplane xb — filled once per round riding the packed scratch write — and
// the two per-body x wrRows become framed doubleACT(30,1) clones
// master -> open_rows[{1,4}]. Requires PIM_FUSED_COSET=1 and a fused-
// eligible primary calib (g_fused_calib_ok, calib_idx==0), same gate as the
// resident consts. Masters live+die within one round (~ms) so they need no
// refresh enrollment. See xmaster_clone_plan_2026_07_27.md.
//   0/unset = off; 1 = clone both x seeds (op[1] and op[4]); 2 = DIAGNOSTIC:
//   clone op[4] only, op[1] stays a wrRow (charge-state isolation arm, §e.4).
static int  g_xmaster        = -1;   // PIM_XMASTER (feature mode)
static bool g_req_xmaster    = false; // per-request arm (alternate twin)
static int  g_xmaster_alt    = -1;   // PIM_XMASTER_ALTERNATE (same-proc twin)
static long g_xm_req_counter = 0;
static void resolve_xmaster_flags() {
  if (g_xmaster < 0) {
    const char* v = getenv("PIM_XMASTER");
    g_xmaster = (v && *v) ? atoi(v) : 0;
    if (g_xmaster < 0) g_xmaster = 0;
    if (g_xmaster)
      fprintf(stderr, "[server] PIM_XMASTER=%d: fused-body x seeds cloned "
              "from per-round master rows (%s)\n", g_xmaster,
              g_xmaster == 2 ? "op[4] only, op[1] stays wrRow" : "op[1]+op[4]");
  }
  if (g_xmaster_alt < 0) {
    const char* v = getenv("PIM_XMASTER_ALTERNATE");
    g_xmaster_alt = (v && *v) ? atoi(v) : 0;
    if (g_xmaster_alt < 0) g_xmaster_alt = 0;
    if (g_xmaster_alt)
      fprintf(stderr, "[server] PIM_XMASTER_ALTERNATE=1: per-request x-master "
              "off/on twins (same-process gate)\n");
  }
}
// True when master rows should be ALLOCATED at startup — either the feature
// or its twin gate is armed (the twin's off-arm still needs the rows claimed
// so both arms share pool geometry).
static bool xmaster_armed() {
  resolve_xmaster_flags();
  return g_xmaster > 0 || g_xmaster_alt > 0;
}
// Effective per-body clone mode when x-master is active: 1 = clone both x
// seeds, 2 = clone op[4] only (op[1] stays a wrRow). Pure-alternate (feature
// mode 0, twin on) defaults to mode 1.
static int xmaster_clone_mode() {
  resolve_xmaster_flags();
  return (g_xmaster == 2) ? 2 : 1;
}
// Per-request arm (call once per request, next to set_req_rc): under
// PIM_XMASTER_ALTERNATE odd requests run the clone arm, even the wrRow arm —
// a same-process A/B below the cross-process MAJ3 floor.
static void set_req_xmaster() {
  resolve_xmaster_flags();
  if (g_xmaster_alt > 0) g_req_xmaster = ((g_xm_req_counter++ & 1) == 1);
  else                   g_req_xmaster = (g_xmaster > 0);
}
// Master rows to claim per bank (one per activation bitplane). BitNet uses
// n_bitplanes=8; planes beyond the claimed count fall back to wrRow per
// plane. PIM_XMASTER_ROWS overrides (default 8, clamped [1,32]).
static uint32_t xmaster_row_count() {
  static const uint32_t n = []{
    const char* v = getenv("PIM_XMASTER_ROWS");
    long r = (v && *v) ? atol(v) : 8;
    if (r < 1) r = 1;
    if (r > 32) r = 32;
    return (uint32_t)r;
  }();
  return n;
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
                                     uint32_t res_zero = RES_ROW_NONE,
                                     // LEVERS #28: x-master row for this body
                                     // (RES_ROW_NONE -> keep the x wrRows).
                                     uint32_t x_master = RES_ROW_NONE) {
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
  // LEVERS #28: replace the two x wrRows (op[1], op[4]) with clones from a
  // per-round master row. Fused mode-1 only (op[1]/[4] fan to {5,9,13} via
  // the coset doubleACT which is unchanged). xm_mode 2 keeps op[1] a wrRow.
  const bool use_xmaster = (x_master != RES_ROW_NONE) && s_fused_coset == 1 &&
                           g_fused_calib_ok;
  const int  xm_mode = use_xmaster ? xmaster_clone_mode() : 1;
  auto emit_x_seed1 = [&]() {
    if (use_xmaster && xm_mode != 2)
      emit_const_clone(p, x_master, open_rows[1]);
    else
      p.add_below(wrRow_immediate_label(BAR, open_rows[1], x_pattern,
                                         label_base + 1));
  };
  auto emit_x_seed4 = [&]() {
    if (use_xmaster)
      emit_const_clone(p, x_master, open_rows[4]);
    else
      p.add_below(wrRow_immediate_label(BAR, open_rows[4], x_pattern,
                                         label_base + 3));
  };
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
      emit_x_seed1();   // LEVERS #28: clone x-master -> op[1] (or wrRow)
      emit_x_seed4();   // LEVERS #28: clone x-master -> op[4] (or wrRow)
    } else {
    p.add_below(wrRow_immediate_label(BAR, open_rows[0], ONE, label_base + 0));
    if (s_fused_coset == 1) {
      emit_x_seed1();   // LEVERS #28: clone x-master -> op[1] (or wrRow)
      p.add_below(wrRow_immediate_label(BAR, open_rows[2], 0u, label_base + 2));
      emit_x_seed4();   // LEVERS #28: clone x-master -> op[4] (or wrRow)
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
    const std::vector<std::pair<uint32_t,uint32_t>>* res_consts = nullptr,
    // LEVERS #28: optional per-body x-master rows; nullptr / short vector /
    // RES_ROW_NONE entries = keep the x wrRows.
    const std::vector<uint32_t>* x_masters = nullptr) {
  Program p;
  p.add_inst(SMC_LI(8, CASR));
  for (size_t i = 0; i < bank_ids.size(); i++) {
    uint32_t rc1 = RES_ROW_NONE, rc0 = RES_ROW_NONE;
    if (res_consts && i < res_consts->size()) {
      rc1 = (*res_consts)[i].first;
      rc0 = (*res_consts)[i].second;
    }
    uint32_t xm = (x_masters && i < x_masters->size()) ? (*x_masters)[i]
                                                       : RES_ROW_NONE;
    emit_bank_combined_body(p, bank_ids[i], backup_rows[i],
                            Rfirsts[i], Rseconds[i],
                            open_rows_list[i], x_patterns[i],
                            label_seed + (int)i * 2000, rc1, rc0, xm);
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
    const std::vector<std::pair<uint32_t,uint32_t>>* res_consts = nullptr,
    // LEVERS #28: optional per-body x-master rows — see the serial builder.
    // Forwarded through the serial fallbacks; honored in the per-bank serial
    // wrRow section of the fused (mode 1) path.
    const std::vector<uint32_t>* x_masters = nullptr)
{
  const int N = (int)bank_ids.size();
  // Today the parallel scheduler is wired for N=4. Smaller N falls back
  // to the serial multibank emit (parallel-of-1 is identical to serial).
  if (N != 4) {
    return build_multibank_combined_program(
        bank_ids, backup_rows, Rfirsts, Rseconds,
        open_rows_list, x_patterns, label_seed, res_consts, x_masters);
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
            open_rows_list, x_patterns, label_seed, res_consts, x_masters);
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
    // LEVERS #28: x-master clone substitution for this bank's x seeds — same
    // fused-mode-1 gate as the serial body. The clone's doubleACT re-LIs
    // RF_REG(10)/LOOP_COLS(13), already in the serial-section clobber set.
    uint32_t xm_b = (x_masters && (size_t)b < x_masters->size())
                    ? (*x_masters)[b] : RES_ROW_NONE;
    const bool use_xmaster_b = (xm_b != RES_ROW_NONE) && fused_mode == 1;
    const int  xm_mode_b = use_xmaster_b ? xmaster_clone_mode() : 1;
    auto emit_x1_b = [&]() {
      if (use_xmaster_b && xm_mode_b != 2)
        emit_const_clone(p, xm_b, open_rows_list[b][1]);
      else
        p.add_below(wrRow_immediate_label(BAR, open_rows_list[b][1],
                                           x_patterns[b],
                                           label_seed + b*2000 + 1));
    };
    auto emit_x4_b = [&]() {
      if (use_xmaster_b)
        emit_const_clone(p, xm_b, open_rows_list[b][4]);
      else
        p.add_below(wrRow_immediate_label(BAR, open_rows_list[b][4],
                                           x_patterns[b],
                                           label_seed + b*2000 + 3));
    };
    if (use_consts_b) {
      emit_const_clone(p, rc0, open_rows_list[b][2]);
      emit_const_clone(p, rc0, open_rows_list[b][8]);
      if (rc_mode == 2 || rc1 == RES_ROW_NONE)
        p.add_below(wrRow_immediate_label(BAR, open_rows_list[b][0],
                                           ONE, label_seed + b*2000 + 0));
      else
        emit_const_clone(p, rc1, open_rows_list[b][0]);
      emit_x1_b();   // LEVERS #28: clone x-master -> op[1] (or wrRow)
      emit_x4_b();   // LEVERS #28: clone x-master -> op[4] (or wrRow)
    } else {
    p.add_below(wrRow_immediate_label(BAR, open_rows_list[b][0],
                                       ONE, label_seed + b*2000 + 0));
    if (fused_mode == 1) {
      emit_x1_b();   // LEVERS #28: clone x-master -> op[1] (or wrRow)
      p.add_below(wrRow_immediate_label(BAR, open_rows_list[b][2], 0u,
                                         label_seed + b*2000 + 2));
      emit_x4_b();   // LEVERS #28: clone x-master -> op[4] (or wrRow)
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
// [#65 2026-08-04] Per-bank residency STATE — the conveyor/working-set model
// (user directive: "if it doesn't fit, move the data before it is needed while
// the other bank/DIMM is utilized"). Host-supplied and runtime-mutable; the
// PREFETCH SCHEDULER that drives transitions is task #67 (built on top of this
// plumbing), NOT this pass. States are representable now so #67 has a home:
//   ACTIVE  — in the compute working set, serves matmul requests now.
//   STAGING — being preloaded with the next stage's weights while idle.
//   STORAGE — holds data (a parked slice) but is not in the active set.
//   FREE    — no data; available for allocation.
// Invalidation rule (see the MAGIC_CONFIG handler): transitions LEAVING ACTIVE
// invalidate that bank's resident scratch; transitions among the idle states
// (FREE/STAGING/STORAGE) are pure metadata and never stop the world — so a
// membership change touching only idle banks is free, as the directive requires.
enum class BankState { ACTIVE, STAGING, STORAGE, FREE };
static const char* bank_state_name(BankState s) {
  switch (s) { case BankState::ACTIVE:  return "ACTIVE";
               case BankState::STAGING: return "STAGING";
               case BankState::STORAGE: return "STORAGE";
               case BankState::FREE:    return "FREE"; }
  return "?";
}

struct BankConfig {
  int bank_id;
  // [#65 2026-08-04] Owning DIMM (bender id) and role — CONFIG, not code. One
  // server process drives ONE DIMM today, so dimm_id == the process's bender;
  // carrying it per bank makes the 4-DIMM×16-bank grid expressible without a
  // code change when a future multi-DIMM server or the #67 conveyor needs it.
  int dimm_id = -1;
  // Per-bank residency state (default ACTIVE = today's behaviour).
  BankState state = BankState::ACTIVE;
  // Per-bank primary subarray window [win_start, win_end). 0/0 => fall back to
  // the global PIM_SUB_START/PIM_SUB_END env (today's single-window behaviour).
  // The boundary atlas (bank_audit) proved the long-range coupling is subarray-
  // position-dependent, so a segment move (conveyor) needs a per-bank window —
  // hence this is per-bank, not one global constant.
  uint32_t win_start = 0, win_end = 0;
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
  // LEVERS #28 X-MASTER: per-activation-bitplane master rows, claimed out of
  // the primary pool at startup (setup_x_masters) with the safe-load pair-
  // offset discipline and REMOVED from backup_pool. Empty = feature off /
  // selection failed -> per-plane wrRow fallback. Content is per-round (the
  // activation bitplane xb), so nothing is pre-written here.
  std::vector<uint32_t> x_master_rows;
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
// ===========================================================================
// P1 RUNG-1 — SINGLE-TRACK RESIDENT WEIGHTS (2026-08-02, STOCKTAKE P1 / C12).
//
// The V2S single-track handler re-uploads each request's weight mask to a
// round-indexed scratch row (per_column_write_row / _packed) BEFORE the body
// RowClones scratch->Rfirst. That per-request wcol re-upload is ~pure delivery
// overhead (81.9 s / 22.8% of the whole-run wall, §1.8 S7) — the SAME weights
// are written every request for a given projection slice.
//
// PIM_W_RESIDENT=1 elides the scratch write whenever the target scratch row
// ALREADY holds the identical weight mask, written recently enough that
// retention has not decayed it. The MM3D body's RowClone(scratch->Rfirst) is
// the refresh: each body ACTs the scratch row (restores charge; content set
// write-driven once), and residents survive as clone sources bit-perfect
// (persistent_weight_throughput: 0/204800 mismatch, 20k back-to-back deposits;
// drift_screen: 0/65536 passive flips). Retention onset on this refresh-less
// platform is ~30 s (b2); the age guard forces a rewrite well under that.
//
// CORRECT-BY-CONSTRUCTION: a write is elided ONLY when the exact 8192 mask
// bytes are provably already resident (memcmp) AND fresh (age < floor). Nothing
// but per_column_write* writes these reserve-tail scratch rows (the body's
// RowClone READS them; activations/MAJ3 land on the calib open_rows; the
// clone-ok pool layout keeps XOR-spread off other used rows), so the row still
// holds what we last wrote it. Output is therefore BIT-IDENTICAL to a fresh
// write. Default OFF: the registry is never consulted and every write fires as
// today — byte-identical current behavior.
//
// The elision fires only when consecutive requests to the SAME round-indexed
// scratch row carry identical content — i.e. same projection slice, no
// intervening different-slice write, no intra-request wrap (n_rounds <=
// v2_scratch_reserve()). Bump PIM_V2_SCRATCH so n_rounds <= reserve to give
// every round a distinct stable row (d_in=2048 -> 16 rounds fits the default 16;
// d_in>2048 wraps at the default and self-corrects to a write).
static int       g_w_resident = -1;             // env PIM_W_RESIDENT (0=off)
static long long g_w_resident_max_age_ns = -1;  // env PIM_W_RESIDENT_MAX_AGE_MS
struct ResidentScratch {
  std::vector<uint32_t> mask;                    // last write-driven content (d_out u32)
  std::chrono::steady_clock::time_point ts;      // wall-time of that write
};
static std::map<std::pair<int,uint32_t>, ResidentScratch> g_resident_rows;
static long g_w_resident_skips_total  = 0;       // whole-run elided scratch writes
static long g_w_resident_writes_total = 0;       // whole-run actual scratch writes (residency on)
static void init_w_resident() {
  if (g_w_resident >= 0) return;
  const char* v = getenv("PIM_W_RESIDENT");
  g_w_resident = (v && *v) ? atoi(v) : 0;
  const char* a = getenv("PIM_W_RESIDENT_MAX_AGE_MS");
  long long ms = (a && *a) ? atoll(a) : 15000;   // < ~30 s b2 retention onset
  g_w_resident_max_age_ns = ms * 1000000LL;
  if (g_w_resident > 0)
    fprintf(stderr, "[server] PIM_W_RESIDENT=%d: single-track resident-weight "
            "scratch elision ON (max_age=%lldms). Default-off byte-identical.\n",
            g_w_resident, ms);
}
// True iff the scratch row already holds `mask` (d_out u32) fresh enough to skip
// its re-write. Never updates the registry (read-only probe).
static bool w_resident_can_skip(int bank_id, uint32_t row,
                                const uint32_t* mask, uint32_t d_out) {
  if (g_w_resident <= 0) return false;
  auto it = g_resident_rows.find({bank_id, row});
  if (it == g_resident_rows.end()) return false;
  const ResidentScratch& e = it->second;
  if (e.mask.size() != (size_t)d_out) return false;
  long long age = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now() - e.ts).count();
  if (age > g_w_resident_max_age_ns) return false;                   // retention guard
  if (memcmp(e.mask.data(), mask, (size_t)d_out * 4) != 0) return false; // content guard
  return true;
}
// Record that `mask` is now resident at (bank_id,row) — call exactly when the
// write is (about to be) issued.
static void w_resident_record(int bank_id, uint32_t row,
                              const uint32_t* mask, uint32_t d_out) {
  if (g_w_resident <= 0) return;
  ResidentScratch& e = g_resident_rows[{bank_id, row}];
  e.mask.assign(mask, mask + d_out);
  e.ts = std::chrono::steady_clock::now();
}

// P1 RUNG-1 DISCRIMINATOR (2026-08-02, RESULT.md §5/§6): classify WHY the
// resident scratch source fails on reuse. Both hooks fire at request START
// only (pipeline empty, BEFORE the stream session opens), both default-off,
// both additive:
//   PIM_WRES_VERIFY=1      : rdRow (raw READ path) every registry row and
//     byte-compare vs the registry's stored write-driven mask.
//     [A] all rows CLEAN  => charge-margin decay (content survives the sense
//         amp; the clone-source quality is what degraded)
//     [B] rows DIRTY      => content pollution (flip geometry vs the body's
//         doubleACT offsets names the polluting op).
//     PIM_WRES_DUMP=<dir> additionally dumps got/exp row images per row.
//   PIM_WRES_ACT_REFRESH=1 : ONE program that ACT+PRE-touches every registry
//     row (~8 insts/row, no c2h) — the candidate cheap [A] fix (charge
//     restore without the 2560-column write). Same template as
//     build_refresh_subarray_loop_program's loop body.
static void wres_verify_registry(SoftMCPlatform& platform, int req_n) {
  static int s_lbl = 0;
  const char* dump_dir = getenv("PIM_WRES_DUMP");
  std::vector<uint8_t> rb(8192);
  long total_rows = 0, dirty_rows = 0; long long total_flip_bits = 0;
  for (const auto& kv : g_resident_rows) {
    int bank = kv.first.first; uint32_t row = kv.first.second;
    const ResidentScratch& e = kv.second;
    if (e.mask.size() != 2048) continue;
    int rc = read_row_to_buffer(platform, bank, row, rb.data(),
                                5000000 + (s_lbl++) * 16);
    if (rc != 8192) {
      fprintf(stderr, "[wres-verify req#%d] bank=%d row=%u rdRow rc=%d\n",
              req_n, bank, row, rc);
      continue;
    }
    total_rows++;
    long mm_bytes = 0; int mm_segs = 0;
    uint32_t xor_acc = 0, or_set = 0;
    int first_seg = -1; uint32_t first_got = 0, first_exp = 0;
    for (int s = 0; s < 2048; s++) {
      uint32_t got_w; memcpy(&got_w, rb.data() + (size_t)s * 4, 4);
      uint32_t exp_w = e.mask[s];
      if (got_w != exp_w) {
        mm_segs++;
        uint32_t fl = got_w ^ exp_w;
        xor_acc |= fl; or_set |= (got_w & ~exp_w);
        total_flip_bits += __builtin_popcount(fl);
        for (int b = 0; b < 4; b++)
          if (((got_w >> (8 * b)) & 0xffu) != ((exp_w >> (8 * b)) & 0xffu))
            mm_bytes++;
        if (first_seg < 0) { first_seg = s; first_got = got_w; first_exp = exp_w; }
      }
    }
    if (mm_segs > 0) {
      dirty_rows++;
      fprintf(stderr,
          "[wres-verify req#%d] bank=%d row=%u DIRTY: segs=%d/2048 bytes=%ld "
          "first(seg=%d exp=0x%08x got=0x%08x) acc_xor=0x%08x acc_set0to1=0x%08x\n",
          req_n, bank, row, mm_segs, mm_bytes, first_seg, first_exp, first_got,
          xor_acc, or_set);
    } else {
      fprintf(stderr, "[wres-verify req#%d] bank=%d row=%u CLEAN 8192/8192\n",
              req_n, bank, row);
    }
    if (dump_dir) {
      char pth[512];
      snprintf(pth, sizeof pth, "%s/wres_req%d_b%d_row%u.bin",
               dump_dir, req_n, bank, row);
      if (FILE* f = fopen(pth, "wb")) { fwrite(rb.data(), 1, 8192, f); fclose(f); }
      snprintf(pth, sizeof pth, "%s/wres_req%d_b%d_row%u.exp.bin",
               dump_dir, req_n, bank, row);
      if (FILE* f = fopen(pth, "wb")) {
        fwrite(e.mask.data(), 1, 8192, f); fclose(f);
      }
    }
  }
  fprintf(stderr, "[wres-verify req#%d] SUMMARY rows=%ld dirty=%ld "
          "flip_bits=%lld => %s\n",
          req_n, total_rows, dirty_rows, total_flip_bits,
          total_rows == 0 ? "(registry empty)"
                          : (dirty_rows == 0 ? "[A]-class (content clean)"
                                             : "[B]-class (content polluted)"));
}
static void wres_act_refresh(SoftMCPlatform& platform) {
  if (g_resident_rows.empty()) return;
  Program p;
  p.add_inst(SMC_LI(8, CASR));
  int n = 0;
  for (const auto& kv : g_resident_rows) {
    p.add_inst(SMC_LI(kv.first.first, BAR));
    p.add_inst(SMC_LI(kv.first.second, RAR));
    p.add_below(PRE(BAR, 0, 0));                 // ensure idle
    p.add_below(ACT(BAR, 0, RAR, 0));            // open → sense-amp restore
    p.add_inst(SMC_SLEEP(4));                    // tRAS guard
    p.add_below(PRE(BAR, 0, 0));                 // close & writeback
    p.add_inst(SMC_SLEEP(4));                    // tRP guard
    n++;
  }
  p.add_inst(SMC_END());
  platform.execute(p);                           // legacy cadence, no c2h payload
  static bool s_once = false;
  if (!s_once) {
    s_once = true;
    fprintf(stderr, "[wres-act-refresh] ACT+PRE touch of %d resident rows per "
            "request (one program, no readback)\n", n);
  }
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
// PIM_VERIFY_ROUNDS: how many rounds mm3d-verify samples, strided across the
// whole round range (so late-round corruption — e.g. the M3 elision — is
// covered, not just round 0). -2 = uninit; default 1 = round-0 only (cheap,
// production); -1 or >= n_rounds = ALL rounds (deep gate; more DRAM readbacks).
static int g_verify_rounds = -2;
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
// PIM_K_ALTERNATE=1 (2026-07-23 DIAGNOSTIC): per-request effective K
// alternates 1 / PIM_INLINE_BITPLANES so replay --dup twins become
// same-process (K=1, K=hi) pairs — the valid exactness gate for the
// K-batching lever (V2 twins judged gate≈control against the rerun
// distribution; MM3D twins must stay bit-exact). Off: g_req_K is
// simply g_inline_bp for every request.
static int  g_req_K        = 1;
static int  g_k_alternate  = -1;
static long g_k_req_counter = 0;
static void set_req_K() {
  if (g_k_alternate < 0) {
    const char* v = getenv("PIM_K_ALTERNATE");
    g_k_alternate = (v && *v) ? atoi(v) : 0;
    if (g_k_alternate > 0)
      fprintf(stderr, "[server] PIM_K_ALTERNATE=1: per-request K "
              "alternates 1/%d (same-process twin gate)\n", g_inline_bp);
  }
  g_req_K = (g_k_alternate > 0 && ((g_k_req_counter++ & 1) == 0))
            ? 1 : g_inline_bp;
}
// PIM_STREAM_PIPE=1 (phase-2, 2026-07-24): within a stream session,
// exec SENDS run ahead of receives (receives defer to a FIFO drained
// opportunistically / at request end) so program build+send for later
// rounds hides under earlier rounds' silicon time. PIM_PIPE_ALTERNATE=1
// = the same-process twin gate (per-request pipe alternates off/on).
static int  g_stream_pipe      = -1;
static bool g_req_pipe         = false;
static int  g_pipe_alternate   = -1;
static long g_pipe_req_counter = 0;
static void set_req_pipe() {
  if (g_stream_pipe < 0) {
    const char* v = getenv("PIM_STREAM_PIPE");
    g_stream_pipe = (v && *v) ? atoi(v) : 0;
    if (g_stream_pipe > 0)
      fprintf(stderr, "[server] PIM_STREAM_PIPE=1: phase-2 send-ahead "
              "pipeline (stream sessions only)\n");
  }
  if (g_pipe_alternate < 0) {
    const char* v = getenv("PIM_PIPE_ALTERNATE");
    g_pipe_alternate = (v && *v) ? atoi(v) : 0;
    if (g_pipe_alternate > 0)
      fprintf(stderr, "[server] PIM_PIPE_ALTERNATE=1: per-request pipe "
              "off/on twins (same-process gate)\n");
  }
  if (g_pipe_alternate > 0)
    g_req_pipe = ((g_pipe_req_counter++ & 1) == 1);
  else
    g_req_pipe = (g_stream_pipe > 0);
}
// PIM_PIPE_DEPTH (default 1): max exec payloads outstanding before the
// next consume. The build-9 ping-pong was only ever validated with the
// host ≤1 program ahead (phase-1 recv pacing); UNBOUNDED deferral hangs
// the engine (2026-07-24: drain starved in xdma_engine_read_cyclic,
// main spinning in receiveData — programs likely clobbered in the idle
// IMEM bank when h2c outruns execution). Depth 1 already hides the
// host build+send under silicon time — the phase-2 win as designed.
// PIM_RC_V2=1: wire the O4(a) resident-const clones into the V2 path
// (historically nullptr — "fallback traffic" rationale obsolete now that
// V2 IS the wall and streaming makes recv = silicon-production time, so
// body cuts translate ~1:1). PIM_RC_ALTERNATE=1 = same-process twin gate
// (consts-off / consts-on per request). Needs PIM_RESIDENT_CONSTS=1 for
// the body-level machinery (validated + drift-hardened 07-19).
static int  g_rc_v2        = -1;
static bool g_req_rc       = false;
static int  g_rc_alternate = -1;
static long g_rc_req_counter = 0;
static void set_req_rc() {
  if (g_rc_v2 < 0) {
    const char* v = getenv("PIM_RC_V2");
    g_rc_v2 = (v && *v) ? atoi(v) : 0;
    if (g_rc_v2 > 0)
      fprintf(stderr, "[server] PIM_RC_V2=1: resident consts wired into "
              "the V2 path\n");
  }
  if (g_rc_alternate < 0) {
    const char* v = getenv("PIM_RC_ALTERNATE");
    g_rc_alternate = (v && *v) ? atoi(v) : 0;
    if (g_rc_alternate > 0)
      fprintf(stderr, "[server] PIM_RC_ALTERNATE=1: per-request V2 consts "
              "off/on twins (same-process gate)\n");
  }
  if (g_rc_alternate > 0)
    g_req_rc = ((g_rc_req_counter++ & 1) == 1);
  else
    g_req_rc = (g_rc_v2 > 0);
}
static int g_pipe_depth = -1;
static int pipe_depth() {
  if (g_pipe_depth < 0) {
    const char* v = getenv("PIM_PIPE_DEPTH");
    g_pipe_depth = (v && *v) ? atoi(v) : 1;
    if (g_pipe_depth < 0) g_pipe_depth = 0;
  }
  return g_pipe_depth;
}
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
// [#65 2026-08-04] Configured bank count — set to banks.size() once the bank
// set is built (main + MAGIC_CONFIG reconfigure). Replaces the hardcoded
// n_banks=4 in the IMEM K-fit heuristic so the estimate tracks the ACTUAL
// bank set (16 banks pack a bigger combined program, so K>1 clamps sooner).
// Default 4 only until the first build_banks() call.
static int g_cfg_n_banks = 4;
static void init_debug_flags() {
  if (g_verify_load < 0) g_verify_load = env_flag("PIM_VERIFY_LOAD", 0);
  if (g_verify_mm3d < 0) g_verify_mm3d = env_flag("PIM_VERIFY_MM3D", 1);
  if (g_verify_rounds == -2) { const char* v = getenv("PIM_VERIFY_ROUNDS"); g_verify_rounds = (v && *v) ? atoi(v) : 1; }
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
    int n_banks  = g_cfg_n_banks;  // [#65] configured count (was hardcoded 4)
    if (n_banks < 1) n_banks = 1;
    int max_K_fit = g_bitstream_imem / (per_body * n_banks);
    if (max_K_fit < 1) max_K_fit = 1;
    if (g_inline_bp > max_K_fit) {
      // 2026-07-24: CLAMP (was warn-only). The warn's own prophecy came
      // true: K=8 built an 8258-inst MM3D program (> IMEM 8192), the
      // platform gate fired, and the truncated program spun execute()'s
      // completion poll forever (11 h at 100% CPU in the K-gate replay).
      // per_body=416 is the serial-emitter worst case, so this clamp is
      // safe by construction; actual fused bodies (~258) would allow
      // K≈7 — a measured-size gate can lift this later if K>4 matters.
      fprintf(stderr, "[server] PIM_INLINE_BITPLANES=%d exceeds IMEM fit "
                      "(est. body=%d × banks=%d × K = %d > %d) — CLAMPED "
                      "to K=%d.\n",
              g_inline_bp, per_body, n_banks,
              per_body * n_banks * g_inline_bp, g_bitstream_imem, max_K_fit);
      g_inline_bp = max_K_fit;
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
      // [#65] Per-bank subarray window wins over the global env. b.win_* is
      // seeded FROM the global PIM_SUB_START/END at build time, so the default
      // (all banks one window) is byte-identical; distinct per-bank windows
      // become possible once the conveyor (#67) moves a bank's segment.
      if (b.win_start || b.win_end) { prim_ws = b.win_start; prim_we = b.win_end; }
      else {
        if (const char* ss = getenv("PIM_SUB_START")) if (*ss) prim_ws = (uint32_t)atoi(ss);
        if (const char* se = getenv("PIM_SUB_END"))   if (*se) prim_we = (uint32_t)atoi(se);
      }
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
    // [#65] Per-bank window preferred (seeded from the global env => identical
    // default); lets the conveyor give banks in different segments distinct
    // refresh ranges without a global env flip.
    if (banks[bk].win_start || banks[bk].win_end) {
      h.refresh_row_start[bk] = banks[bk].win_start;
      h.refresh_row_end[bk]   = banks[bk].win_end;
    } else {
      if (const char* ss = getenv("PIM_SUB_START")) if (*ss) h.refresh_row_start[bk] = (uint32_t)atoi(ss);
      if (const char* se = getenv("PIM_SUB_END"))   if (*se) h.refresh_row_end[bk]   = (uint32_t)atoi(se);
    }
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
    // Task #50 item 2: drift-aware desc-serve re-writes each walk chunk's backup
    // rows from these masks (PIM_DESC_REWRITE, default ON when PIM_DESC_SERVE is
    // set) — keep the full per-(round,bank) masks at LOAD so the serve path has
    // the weight data to refresh with. Same n_rounds*N*8KB cost as the handle
    // path's PIM_LOAD_REWRITE_ON_MM3D.
    if (getenv("PIM_DESC_SERVE") && atoi(getenv("PIM_DESC_SERVE")) > 0 &&
        (!getenv("PIM_DESC_REWRITE") || atoi(getenv("PIM_DESC_REWRITE")) != 0))
      keep_masks = true;
    // Task #59 DEFECT B guard: the beat-periodic word-4/5 exact-repair recomputes
    // the affected columns from these masks (no re-read — the armed fabric read
    // path is corrupted). Keep masks whenever the guard is active on a desc-serve
    // handle, even at PIM_DESC_REWRITE=0 (else the repair bails to the also-armed
    // handle path). Default ON; PIM_DESC_B_GUARD=0 disables.
    if (getenv("PIM_DESC_SERVE") && atoi(getenv("PIM_DESC_SERVE")) > 0 &&
        !(getenv("PIM_DESC_B_GUARD") && atoi(getenv("PIM_DESC_B_GUARD")) == 0))
      keep_masks = true;
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
  init_w_resident();    // P1 rung-1 single-track resident-weight scratch elision
  set_req_K();          // per-request effective K (PIM_K_ALTERNATE twin gate)
  set_req_pipe();       // per-request phase-2 pipe (PIM_PIPE_ALTERNATE gate)
  set_req_rc();         // per-request V2 resident-consts (PIM_RC_ALTERNATE)
  set_req_xmaster();    // per-request x-master arm (PIM_XMASTER_ALTERNATE)
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
  bool req_accxbp = g_accxbp > 0 && single && g_req_K == 1
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

  // Rung-1 producer loop: one stream session for this request (V2-family
  // only in phase 1; accxbp keeps legacy cadence — its capture needs
  // read-quiet windows). Mode is set inside the ctor BEFORE the pipeline
  // opens; the in-loop ensure_readback calls become host-side no-ops.
  // O4(a) drift rewrite, V2 side (07-24): const rows decay under the
  // same-subarray tuple traffic regardless of which handler drives it;
  // the historical rewrite fired only per MM3D request. Runs BEFORE the
  // session opens (legacy writes, pipeline empty). Shares the env
  // cadence; separate sequence counter (V2 request stream).
  if (resident_consts_mode() > 0 && g_req_rc) {
    static long s_v2_consts_seq = 0;
    const int rw_every = consts_rewrite_every();
    if (rw_every > 0 && (s_v2_consts_seq % rw_every) == 0)
      rewrite_resident_const_rows(platform, banks);
    s_v2_consts_seq++;
  }
  // P1 rung-1 discriminator hooks (request START, pipeline empty, BEFORE the
  // stream session opens; both default-off — see wres_verify_registry).
  {
    static int s_disc_req = 0;
    s_disc_req++;
    static const int s_wv = []{ const char* v = getenv("PIM_WRES_VERIFY");
                                return (v && *v) ? atoi(v) : 0; }();
    static const int s_wr = []{ const char* v = getenv("PIM_WRES_ACT_REFRESH");
                                return (v && *v) ? atoi(v) : 0; }();
    if (s_wv > 0 && !g_resident_rows.empty())
      wres_verify_registry(platform, s_disc_req);
    if (s_wr > 0) wres_act_refresh(platform);
  }
  const bool alt_stream_this =
      !stream_alternate() || ((g_v2_req_counter++ & 1) == 1);
  StreamSession stream_sess(platform,
                            stream_on() && !req_accxbp && alt_stream_this,
                            /*segpop_mode=*/true);

  // Per-request silicon-side timing (server-internal profile).
  using clk = std::chrono::steady_clock;
  using ns_t = std::chrono::nanoseconds;
  auto t_req_start = clk::now();
  long long t_wcol_ns = 0, t_exec_ns = 0, t_recv_ns = 0, t_pop_ns = 0;
  int n_wcol_execs = 0, n_maj3_execs = 0;
  int n_wres_skips = 0;   // P1 rung-1: per-request elided scratch writes (residency)
  // PIM_SKIP_ZERO_PLANES: elide zero-x MAJ3 bodies. Plain body builder only
  // (fused_coset_mode()!=0 keeps the 5/5/5 shape); the in-fabric ACCUM_XBP
  // accumulator sums bodies under one per-plane weight, so leave its body
  // set intact and let it drain per-round as usual.
  const bool do_skip_zero = skip_zero_planes_on()
                            && fused_coset_mode() == 0 && !req_accxbp;
  int    n_zskip_units = 0;
  size_t n_zunits_seen = 0;

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
  // Phase-2 producer pipeline (PIM_STREAM_PIPE, 2026-07-24): EVERY exec
  // flows through pend_recv + consume_one. Pipe OFF: each push is
  // consumed immediately — the platform call sequence is byte-identical
  // to the historical send->recv->pop cadence. Pipe ON (stream session
  // only): receives defer, so later rounds' program build+send hides
  // under earlier rounds' silicon time. The 2026-05-04 write-then-use
  // locality constraint is untouched: the SEND order (= the DDR-side
  // timeline) is identical; only host-side receive timing moves.
  struct PendingRecv {
    size_t   total_bytes;
    uint32_t bp_start, K;
    int      active;
    size_t   round;
    std::vector<int>      signs;
    // Emitted-order maps (parallel to signs): bank index + bitplane per body
    // actually queued. With PIM_SKIP_ZERO_PLANES the queued set is a
    // compacted subsequence of (kp,bk), so consume_one can no longer
    // reconstruct bk/bitplane from loop indices — carry them here.
    std::vector<int>      bks;
    std::vector<uint32_t> bps;
  };
  std::deque<PendingRecv> pend_recv;
  auto consume_one = [&]() -> int {
    PendingRecv pr = std::move(pend_recv.front());
    pend_recv.pop_front();
    static thread_local std::vector<uint8_t> rows_buf;
    if (rows_buf.size() < pr.total_bytes) rows_buf.resize(pr.total_bytes);
    auto t_recv0 = clk::now();
    int rc = platform.receiveData(rows_buf.data(), (int)pr.total_bytes);
    t_recv_ns += std::chrono::duration_cast<ns_t>(clk::now() - t_recv0).count();
    if (rc != (int)pr.total_bytes) {
      fprintf(stderr, "[server] receiveData rc=%d expected=%zu "
              "(round=%zu bp=[%u..%u))\n", rc, pr.total_bytes,
              pr.round, pr.bp_start, pr.bp_start + pr.K);
      return -1;
    }
    // PIM_DUMP_ROWS=<dir> forensics (moved with the recv, 07-24).
    if (const char* dd = getenv("PIM_DUMP_ROWS")) {
      static int dump_n = 0;
      static int dump_max = getenv("PIM_DUMP_ROWS_MAX")
                            ? atoi(getenv("PIM_DUMP_ROWS_MAX")) : 64;
      if (dump_n < dump_max) {
        char path[512];
        snprintf(path, sizeof path, "%s/rows_%05d_r%zu_bp%u_M%zu.bin",
                 dd, dump_n++, pr.round, pr.bp_start,
                 pr.total_bytes / row_read_bytes());
        if (FILE* fp = fopen(path, "wb")) {
          fwrite(rows_buf.data(), 1, pr.total_bytes, fp);
          fclose(fp);
        }
      }
    }
    auto t_pop0 = clk::now();
    // Emitted-order: idx runs over the bodies actually queued (== rows_buf
    // order == ex_* order). bk/bitplane come from pr.bks/pr.bps so a filtered
    // (zero-x-elided) ordering stays self-consistent.
    for (size_t idx = 0; idx < pr.signs.size(); idx++) {
      int      bk = pr.bks[idx];
      uint32_t b  = pr.bps[idx];
      const uint8_t* row = rows_buf.data() + idx * row_read_bytes();
      vector<int> pc(d_out);
      row_pc(row, pc.data(), (int)d_out);
      // O10: host-repair fused-marginal columns (fused layout iff
      // calib_idx==0 and coset mode 1/3; the unit's mask is in-request).
      if (calib_idx == 0 && !banks[bk].fused_col_bad.empty()) {
        int fm = fused_coset_mode();
        if (fm == 1 || fm == 3) {
          size_t u2 = pr.round * (size_t)N + (size_t)bk;
          uint32_t ch2 = single ? (uint32_t)u2 : (uint32_t)(u2 / 2);
          const uint32_t* m2 = (single || (u2 % 2) == 0)
              ? pos_mask_all + (size_t)ch2 * d_out
              : neg_mask_all + (size_t)ch2 * d_out;
          fused_repair_pc(banks[bk], pc.data(), m2,
                          x_bitplane_all[(size_t)ch2 * n_bitplanes + b]);
        }
      }
      int sign_factor = (pr.signs[idx] == 0) ? +1 : -1;
      int weight = sign_factor * bitplane_factor[b];
      size_t u_acc = pr.round * (size_t)N + (size_t)bk;
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
            pr.round, b, banks[bk].bank_id, sign_factor, weight,
            row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7],
            row[8],row[9],row[10],row[11],row[12],row[13],row[14],row[15],
            pc[0],pc[1],pc[2],pc[3],pc[4],pc[5],pc[6],pc[7],
            y[0],y[1],y[2],y[3],y[4],y[5],y[6],y[7]);
      }
    }
    t_pop_ns += std::chrono::duration_cast<ns_t>(clk::now() - t_pop0).count();
    return 0;
  };
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
      // P1 rung-1: elide the re-upload when this row already holds `mask`
      // (fresh). Body RowClones the resident content — bit-identical to a
      // fresh write. Off (g_w_resident<=0) => can_skip is always false =>
      // byte-identical current behavior.
      const bool w_skip =
          w_resident_can_skip(banks[bk].bank_id, scratch_row, mask, d_out);
      if (g_v2_pack > 0) {
        if (w_skip) {
          n_wres_skips++;
        } else {
          ScratchWrite sw{banks[bk].bank_id, scratch_row, mask};
          round_writes.push_back(sw);
          w_resident_record(banks[bk].bank_id, scratch_row, mask, d_out);
        }
      } else {
        if (w_skip) {
          n_wres_skips++;
        } else {
          auto t0 = clk::now();
          per_column_write_row(platform, banks[bk].bank_id, scratch_row, mask);
          t_wcol_ns += std::chrono::duration_cast<ns_t>(clk::now() - t0).count();
          n_wcol_execs += 3;  // per_column_write_row issues 3 platform.execute calls
          w_resident_record(banks[bk].bank_id, scratch_row, mask, d_out);
        }
      }
      active_in_round++;
    }
    // LEVERS #28 X-MASTER: fill this round's per-(bank,plane) master rows with
    // the activation bitplane xb, riding the same packed write program (one
    // uniform wrRow per master). Gated to the fused primary path with packing
    // on; the bitplane dispatch below clones ONLY the masters filled here.
    // Send order writes-before-bodies, so each master is written once and read
    // once within this round (no refresh enrollment needed).
    // X-master is only sound on the DUAL track: the V2S host reconstruction
    // y = 2*y_pos - Σx pairs an in-DRAM pc_pos against an EXACT host Σx, so
    // the clone's charge-shared seed deviation doubles and never cancels
    // (dual measures both tracks in-DRAM against the same seed → coherent).
    // Proven 2026-07-28: V2S oracle deep shape XM=0 bit-exact 2048/2048 vs
    // XM=1 corr 0.951, 4/2048. Gate !single here AND at xm_this below.
    const bool xm_round = g_req_xmaster && !single && g_v2_pack > 0 &&
                          fused_coset_mode() == 1 && calib_idx == 0;
    if (xm_round) {
      for (int bk = 0; bk < active_in_round; bk++) {
        const std::vector<uint32_t>& xmr = banks[bk].x_master_rows;
        if (xmr.empty()) continue;
        size_t u = round * (size_t)N + (size_t)bk;
        uint32_t chunk = single ? (uint32_t)u : (uint32_t)(u / 2);
        uint32_t nfill = (uint32_t)std::min((size_t)n_bitplanes, xmr.size());
        for (uint32_t b = 0; b < nfill; b++) {
          uint32_t xb = x_bitplane_all[(size_t)chunk * n_bitplanes + b];
          ScratchWrite mw{banks[bk].bank_id, xmr[b], nullptr};
          mw.uniform = true;
          mw.uniform_val = xb;
          mw.uniform_label = 800000 + bk * 64 + (int)b;   // unique per round pgm
          round_writes.push_back(mw);
        }
      }
    }
    if (g_v2_pack > 0 && !round_writes.empty()) {
      auto t0 = clk::now();
      per_column_write_rows_packed(platform, round_writes, &n_wcol_execs);
      t_wcol_ns += std::chrono::duration_cast<ns_t>(clk::now() - t0).count();
    }
    if (active_in_round == 0) break;

    // 2. Bitplane dispatch — chunked by g_req_K (= PIM_INLINE_BITPLANES,
    //    or the per-request alternated K under PIM_K_ALTERNATE).
    // K=1 reproduces the historical per-bitplane cadence; K>1 chains
    // K × active_in_round bank bodies into one program, doing a single
    // platform.execute + single receiveData per chunk. Each extra bitplane
    // amortises one host-FPGA round-trip (~30 µs) at the cost of K× more
    // c2h drain per execute (K × N × 8 KB).
    for (uint32_t bp_start = 0; bp_start < n_bitplanes;
         bp_start += (uint32_t)g_req_K) {
      uint32_t K = std::min((uint32_t)g_req_K, n_bitplanes - bp_start);
      size_t   M = (size_t)K * (size_t)active_in_round;   // upper bound
      std::vector<int>             ex_bank_ids;
      std::vector<uint32_t>        ex_backup_rows;
      std::vector<uint32_t>        ex_Rfirsts;
      std::vector<uint32_t>        ex_Rseconds;
      std::vector<const uint32_t*> ex_open_rows;
      std::vector<uint32_t>        ex_x_patterns;
      std::vector<int>             ex_signs;
      // Emitted-order maps carried into PendingRecv (see consume_one).
      std::vector<uint32_t>        ex_bp;
      std::vector<int>             ex_bk;
      // V2 path selects calibs by calib_idx: fused only on the validated
      // primary (idx 0); cs_extra trips get the plain 11-wrRow body.
      // PIM_RC_V2 resident-const pairs and LEVERS #28 x-master rows are now
      // built INLINE with the ex_* bodies (hoisted out of their old
      // post-loops) so a PIM_SKIP_ZERO_PLANES elision filters every parallel
      // array identically — a separate post-loop would drift out of order.
      // Both are primary-calib only (the fused-eligible class); RES_ROW_NONE
      // entries fall back to wrRow inside the body. xm_this implies fused
      // mode 1, where do_skip_zero is off, so no elision to stay aligned with.
      const bool rc_this = g_req_rc && resident_consts_mode() > 0 &&
                           calib_idx == 0;
      const bool xm_this = g_req_xmaster && !single && g_v2_pack > 0 &&
                           calib_idx == 0;
      std::vector<std::pair<uint32_t,uint32_t>> ex_res;
      std::vector<uint32_t> ex_x_masters;
      ex_bank_ids.reserve(M);
      ex_backup_rows.reserve(M);
      ex_Rfirsts.reserve(M);
      ex_Rseconds.reserve(M);
      ex_open_rows.reserve(M);
      ex_x_patterns.reserve(M);
      ex_signs.reserve(M);
      ex_bp.reserve(M);
      ex_bk.reserve(M);
      if (rc_this) ex_res.reserve(M);
      if (xm_this) ex_x_masters.reserve(M);

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
          n_zunits_seen++;
          // PIM_SKIP_ZERO_PLANES: zero-x body contributes nothing — elide.
          if (do_skip_zero && xb == 0) { n_zskip_units++; continue; }
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
          ex_bp.push_back(b);
          ex_bk.push_back(bk);
          if (rc_this)
            ex_res.emplace_back(banks[bk].res_one_row, banks[bk].res_zero_row);
          if (xm_this) {
            const std::vector<uint32_t>& xmr = banks[bk].x_master_rows;
            ex_x_masters.push_back(b < xmr.size() ? xmr[b] : RES_ROW_NONE);
          }
        }
      }
      // Emitted body count after any zero-x elision — payload, readback and
      // the PendingRecv indexing all key off this (not K*active).
      M = ex_bank_ids.size();
      if (M == 0) continue;   // whole (bp_start) group elided — no exec/recv

      g_fused_calib_ok = (calib_idx == 0);
      Program p = (g_parallel_banks
          ? build_multibank_parallel_program
          : build_multibank_combined_program)(
              ex_bank_ids, ex_backup_rows, ex_Rfirsts, ex_Rseconds,
              ex_open_rows, ex_x_patterns, label_base,
              rc_this ? &ex_res : nullptr,
              xm_this ? &ex_x_masters : nullptr);
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
        // ROOT-CAUSE PROBE (2026-08-02, env-gated default-off = byte-identical):
        // PIM_ACCXBP_PLANE_BARRIER=N inserts an N-NOP barrier program AFTER this
        // plane's execute and BEFORE the next plane's set_acc_weight, so this
        // plane's still-in-flight read beats finish RMW under THIS plane's weight
        // (tests the weight-vs-drain race hypothesis).
        { static int axb_bar = -1;
          if (axb_bar < 0) { const char* v = getenv("PIM_ACCXBP_PLANE_BARRIER"); axb_bar = v ? atoi(v) : 0; }
          if (axb_bar > 0) { Program bp2; for (int q = 0; q < axb_bar; q++) bp2.add_inst(all_nops()); bp2.add_inst(SMC_END()); platform.execute(bp2); } }
        // PIM_ACCXBP_FLUSH_PER_PLANE=1: drain THIS plane's accumulator now
        // (guaranteed no cross-plane weight race), accumulate its partial into y,
        // and suppress the round-end flush (see the req_accxbp round-end block).
        { static int axb_fpp = -1;
          if (axb_fpp < 0) { const char* v = getenv("PIM_ACCXBP_FLUSH_PER_PLANE"); axb_fpp = v ? atoi(v) : 0; }
          if (axb_fpp > 0) {
            platform.flush_acc();
            static thread_local std::vector<int32_t> axb_pp(2048);
            int rcpp = platform.receiveData(axb_pp.data(), 8192);
            if (rcpp != 8192) { fprintf(stderr, "[server] ACCXBP per-plane drain rc=%d\n", rcpp); return -1; }
            size_t g_pp = (round * (size_t)N) / group_chunks;
            int32_t* y_pp = y.data() + g_pp * d_out;
            for (uint32_t j = 0; j < d_out; j++) y_pp[j] += axb_pp[j];
          } }
        continue;
      }
      ensure_readback(platform, true);   // PIM_SEGPOP: matvec reads in SEG_POP
      auto t_exec0 = clk::now();
      pexec(platform, p, (int)(M * row_read_bytes()));
      t_exec_ns += std::chrono::duration_cast<ns_t>(clk::now() - t_exec0).count();
      n_maj3_execs++;
      pend_recv.push_back(PendingRecv{M * row_read_bytes(), bp_start, K,
                                      active_in_round, round,
                                      std::move(ex_signs),
                                      std::move(ex_bk), std::move(ex_bp)});
      // Pipe OFF (or no session): consume immediately — byte-identical
      // platform call sequence to the historical cadence. Pipe ON:
      // consume down to PIM_PIPE_DEPTH outstanding (bounded send-ahead;
      // depth 1 = the validated ping-pong envelope).
      if (!(g_req_pipe && g_stream_session)) {
        if (consume_one() != 0) return -1;
      } else {
        while (pend_recv.size() > (size_t)pipe_depth())
          if (consume_one() != 0) return -1;
      }
    }

    // PIM_ACCUM_XBP: this round's planes accumulated in-fabric — ONE
    // drain returns the finished per-segment place-value sums.
    // (Skipped when PIM_ACCXBP_FLUSH_PER_PLANE drained each plane above.)
    static int axb_fpp_r = -1;
    if (axb_fpp_r < 0) { const char* v = getenv("PIM_ACCXBP_FLUSH_PER_PLANE"); axb_fpp_r = v ? atoi(v) : 0; }
    if (req_accxbp && axb_fpp_r <= 0) {
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
  // Phase-2 drain: anything still in flight lands here, in order.
  while (!pend_recv.empty())
    if (consume_one() != 0) return -1;

  if (do_skip_zero) {
    g_zskip_total   += (long)n_zunits_seen;
    g_zskip_skipped += (long)n_zskip_units;
    zskip_report();
  }

  // PIM_ACCUM_XBP epilogue: an oversize-skipped program would have
  // silently dropped a whole plane from the in-fabric sums (the
  // accum-stream hazard class — see oversize_skips()) — fail loudly.
  // Then restore READ mode (SEG_POP re-arms on the next ensure_readback).
  if (req_accxbp) {
    if (platform.oversize_skips() != axb_skips0) {
      fprintf(stderr, "[server] ACCUM_XBP: oversize skips advanced "
              "mid-request — results incomplete, aborting\n");
      // Accumulator may hold a partial sum: force the next eligible
      // request's mode entry to re-send the SET (128-cycle clear).
      g_mode_accxbp_now = false;
      return -1;
    }
    // STAY in ACCUM_XBP across requests — the flush already zeroed the
    // accumulator, and per-request exit/re-entry costs ~2 s (the Arm-B
    // tax). Ineligible consumers leave the mode lazily via
    // ensure_readback().
  }

  long long t_total_ns = std::chrono::duration_cast<ns_t>(
      clk::now() - t_req_start).count();
  // Print server-side per-request timing on stderr (every 50th request to
  // avoid noise; first few always).
  static int s_req_n = 0;
  s_req_n++;
  g_w_resident_skips_total  += n_wres_skips;
  g_w_resident_writes_total += (n_wcol_execs / 3);  // approx; packed path folds
  if (s_req_n <= 5 || s_req_n % 50 == 0) {
    long long unaccounted = t_total_ns - t_wcol_ns - t_exec_ns - t_recv_ns - t_pop_ns;
    fprintf(stderr,
        "[srv-prof #%d] total=%.1fms wcol=%.1fms (%dx) exec=%.1fms (%dx) "
        "recv=%.1fms pop=%.1fms other=%.1fms wres_skip=%dx\n",
        s_req_n, t_total_ns/1e6, t_wcol_ns/1e6, n_wcol_execs,
        t_exec_ns/1e6, n_maj3_execs,
        t_recv_ns/1e6, t_pop_ns/1e6, unaccounted/1e6, n_wres_skips);
  }
  // TELEMETRY LIVENESS ASSERT (2026-07-29): mirror of the MM3D assert — a
  // V2 request with work MUST have executed programs. See the mm3d-prof
  // site for the rationale (the brace-bug lesson). Same opt-out.
  if (n_maj3_execs == 0) {
    static const bool s_allow_v2 = []{
      const char* v = getenv("PIM_ALLOW_ZERO_EXEC");
      return v && atoi(v) > 0;
    }();
    // PIM_SKIP_ZERO_PLANES: a request whose every body was a zero-x plane
    // legitimately runs zero execs (all contributions 0). Allowed iff the
    // elision accounts for it; the genuine zero-exec-with-no-skip case
    // (the brace-bug signature) still hard-fails.
    const bool zskip_covers = (n_zskip_units > 0);
    fprintf(stderr, "[LIVENESS-ASSERT] V2 request #%d ran ZERO MAJ3 execs "
            "(zskip=%d) — %s%s\n", s_req_n, n_zskip_units,
            zskip_covers ? "all bodies were zero-x planes (skip-covered)"
                         : "response would be fabricated",
            (zskip_covers || s_allow_v2)
                ? (zskip_covers ? " continuing (skip-covered)."
                                : " PIM_ALLOW_ZERO_EXEC=1: continuing.")
                : " ABORTING.");
    if (!zskip_covers && !s_allow_v2) return -1;
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
  // PIM_SKIP_ZERO_PLANES: elide zero-x MAJ3 bodies. Only on the plain
  // (non-fused) body builder — fused_coset_mode()!=0 keeps the 5/5/5 shape.
  const bool do_skip_zero = skip_zero_planes_on() && fused_coset_mode() == 0;
  size_t n_zskip_units = 0, n_zunits_seen = 0;

  init_debug_flags();
  set_req_K();          // per-request effective K (PIM_K_ALTERNATE twin gate)

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
    int vr_nr = (int)h.n_rounds, vr = g_verify_rounds;
    if (vr < 0 || vr > vr_nr) vr = vr_nr;   // -1 / oversize => sample ALL rounds
    if (vr < 1) vr = 1;                     // ROUNDS=0 => round-0 (to disable, use PIM_VERIFY_MM3D=0)
    for (int vr_i = 0; vr_i < vr; vr_i++) {
     int round = (vr >= vr_nr) ? vr_i : (int)((long)vr_i * vr_nr / vr);  // strided across 0..n_rounds-1
     if ((size_t)round >= h.per_round_backup_rows.size()) break;
     for (int bk = 0; bk < N; bk++) {
      if ((size_t)bk >= h.per_round_backup_rows[round].size()) break;
      uint32_t row = h.per_round_backup_rows[round][bk];
      if (h.expected_popcounts[round][bk].empty()) continue;  // Bug1 fix 2026-07-27: partial final round — bank has no unit; skip (empty exp_pc would OOB below)
      int rc = read_row_to_buffer(platform, banks[bk].bank_id, row,
                                   rb.data(), 2000000 + round*4000000 + (int)handle_id * 100 + bk);
      if (rc != 8192) {
        fprintf(stderr, "[mm3d-verify] handle=%u bk=%d rdRow rc=%d\n",
                handle_id, banks[bk].bank_id, rc);
        continue;
      }
      // Read the SAME row again to test read stability — if rb != rb2,
      // the read itself is unstable / cells are flaky.
      int rc2 = read_row_to_buffer(platform, banks[bk].bank_id, row,
                                    rb2.data(),
                                    3000000 + round*4000000 + (int)handle_id * 100 + bk);
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
      const auto& exp_pc = h.expected_popcounts[round][bk];
      const auto& exp_mask = h.expected_first_row_mask[bk];  // round-0's mask (byte-detail only)
      // Per-bit OR/XOR accumulator over ALL mismatched segments —
      // shows which bit positions are systematically being flipped.
      uint32_t bit_or_acc = 0, bit_xor_acc = 0;
      for (uint32_t s = 0; s < d_out; s++) {
        if ((uint8_t)got_pc[s] != exp_pc[s]) {
          uint32_t got_w = (uint32_t)rb[s*4]
                         | ((uint32_t)rb[s*4+1] << 8)
                         | ((uint32_t)rb[s*4+2] << 16)
                         | ((uint32_t)rb[s*4+3] << 24);
          uint32_t exp_w = (round == 0 && s < exp_mask.size()) ? exp_mask[s] : 0u;
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
    }
    t_verify_ns = std::chrono::duration_cast<ns_t>(clk::now() - t0).count();
    if (mismatch_segs > 0) {
      fprintf(stderr,
          "[mm3d-verify] handle=%u DECAY/CORRUPTION: %ld/%ld segs "
          "differ (sampled rounds) (%.4f%%); first @bk=%d s=%d "
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
            "[mm3d-verify] handle=%u popcounts OK "
            "(%ld segs sampled, refresh=%d)\n",
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
            fx_open_rows, fx_x_patterns, label_base, &fx_consts,
            /*x_masters=*/nullptr);   // LEVERS #28: LOAD-handle path stays on
                                      // wrRow (parity deferred, plan §Sequence)
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
    // This lambda is invoked from INSIDE the MM3D StreamSession scope (both
    // call sites are in the packed bitplane loop below), so a raw
    // platform.execute() here hits the platform's session guard — "execute()
    // called with a STREAM SESSION ACTIVE ... poisoning" — after which the
    // next receiveData returns 0 and the server exits. That is the failure
    // mode of every PIM_PACK_ROUNDS>1 + PIM_STREAM_MM3D=1 run. Dispatch
    // through the streaming-aware wrapper instead, matching the unpacked
    // lane. With no session open pexec() IS platform.execute(), so the
    // shipped default (PIM_PACK_ROUNDS=1, where this lambda is unreachable)
    // is byte-identical.
    pexec(platform, p, (int)(M * row_read_bytes()));
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
      // 2026-07-26 STREAM MM3D (LEVERS lever #3 phase-2, previously OPEN).
      // MM3D never opened a stream session, so PIM_STREAM had NO EFFECT on the
      // dominant path — only the V2 handler streamed. The main dispatch below
      // already goes through pexec(), so it streams as soon as a session is
      // open. Scope is EXACTLY this bitplane loop: every dispatch reachable
      // from it goes through pexec() — including the packed path inside the
      // flush_pend lambda above, which is called from here even though it is
      // written outside the loop body (the auxiliary refresh/verify/disturb
      // executes live outside the scope entirely). Mixing a direct
      // platform.execute() into an open session is illegal — it closes the
      // server — so any new dispatch added here must use pexec() too.
      // Opt-in while it earns trust: PIM_STREAM_MM3D=1 (task #37 recorded a
      // streamed-MM3D wedge on build-14; that wedge is fixed but this stays
      // A/B-able).
      {
      StreamSession mm3d_stream(platform,
                                stream_on() && env_flag("PIM_STREAM_MM3D", 0),
                                /*segpop_mode=*/true);
      for (uint32_t bp_start = 0; bp_start < n_bitplanes;
           bp_start += (uint32_t)g_req_K) {
        uint32_t K = std::min((uint32_t)g_req_K, n_bitplanes - bp_start);
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
    }   // FIX 2026-07-29: close if(pack_rounds>1) HERE. The unpacked
        // bitplane dispatch below is the pack_rounds<=1 FALL-THROUGH path.
        // It was erroneously nested INSIDE this if (the packed StreamSession
        // scope's `}` closed the scope but not the if), making it dead code:
        // skipped entirely when pack_rounds==1 (the default), and jumped by
        // the `continue` above when pack_rounds>1. Net effect: the LOAD/MM3D
        // handle path executed ZERO MAJ3 programs (mm3d-prof exec=0x) and
        // returned an all-zero y for every request. The matching redundant
        // `}` that used to close this if far below (after the unpacked
        // pexec) is removed.

    // Bitplane dispatch — chunked by g_req_K; see process_request for
    // the matching v2-path comment.
    for (uint32_t bp_start = 0; bp_start < n_bitplanes;
         bp_start += (uint32_t)g_req_K) {
      uint32_t K = std::min((uint32_t)g_req_K, n_bitplanes - bp_start);
      size_t   M = (size_t)K * (size_t)active_in_round;
      std::vector<int>             ex_bank_ids;
      std::vector<uint32_t>        ex_backup_rows;
      std::vector<uint32_t>        ex_Rfirsts;
      std::vector<uint32_t>        ex_Rseconds;
      std::vector<const uint32_t*> ex_open_rows;
      std::vector<uint32_t>        ex_x_patterns;
      std::vector<int>             ex_signs;
      std::vector<std::pair<uint32_t,uint32_t>> ex_consts;  // O4 (a)
      // Emitted-order maps: with PIM_SKIP_ZERO_PLANES the ex_* vectors are a
      // compacted subsequence of (kp,bk), so the post-recv loop can no longer
      // reconstruct bk/bitplane from the loop indices — carry them per body.
      std::vector<uint32_t>        ex_bp;
      std::vector<int>             ex_bk;
      ex_bank_ids.reserve(M);
      ex_backup_rows.reserve(M);
      ex_Rfirsts.reserve(M);
      ex_Rseconds.reserve(M);
      ex_open_rows.reserve(M);
      ex_x_patterns.reserve(M);
      ex_signs.reserve(M);
      ex_consts.reserve(M);
      ex_bp.reserve(M);
      ex_bk.reserve(M);
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
          n_zunits_seen++;
          // PIM_SKIP_ZERO_PLANES: a zero-x body's popcount(W AND 0)==0
          // contributes nothing; elide it (no push -> not emitted).
          if (do_skip_zero && xb == 0) { n_zskip_units++; continue; }
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
          ex_bp.push_back(b);
          ex_bk.push_back(bk);
          // O4 (a): resident consts are primary-calib only.
          ex_consts.emplace_back(
              sel == 0 ? banks[bk].res_one_row  : RES_ROW_NONE,
              sel == 0 ? banks[bk].res_zero_row : RES_ROW_NONE);
        }
      }
      // Emitted body count after any zero-x elision — payload, readback and
      // the post-recv indexing below all key off this (not K*active).
      M = ex_bank_ids.size();
      if (M == 0) continue;   // whole (bp_start) group elided — no exec/recv
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
              ex_open_rows, ex_x_patterns, label_base, &ex_consts,
              /*x_masters=*/nullptr);   // LEVERS #28: LOAD-handle path stays on
                                        // wrRow (parity deferred, plan §Sequence)
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
      pexec(platform, p, (int)(M * row_read_bytes()));
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
      // Emitted-order: idx runs over the bodies actually queued (== rows_buf
      // order == ex_* order). bk/bitplane come from ex_bk/ex_bp so a filtered
      // (zero-x-elided) ordering stays self-consistent.
      for (size_t idx = 0; idx < M; idx++) {
        int      bk = ex_bk[idx];
        uint32_t b  = ex_bp[idx];
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
      }   // end MM3D StreamSession scope (bitplane loop)
    // FIX 2026-07-29: the `}` that used to be here closed if(pack_rounds>1)
    // AFTER the unpacked dispatch — that mis-nesting was the zero-exec bug.
    // The if now closes right after the packed block above; this scope's
    // remaining brace belongs to the round loop.
  }

  // O4 (b): drain any bodies still queued when the round loop ends (the
  // request's tail rounds may not fill a whole PIM_PACK_ROUNDS span).
  if (pack_rounds > 1 && flush_pend() != 0) return -1;

  if (do_skip_zero) {
    g_zskip_total   += (long)n_zunits_seen;
    g_zskip_skipped += (long)n_zskip_units;
    zskip_report();
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
  // TELEMETRY LIVENESS ASSERT (2026-07-29, the brace-bug lesson): a metric
  // nobody asserts on is decoration. This exact profiler printed exec=(0x)
  // for three days while the handle path returned all-zero y and every
  // token-level gate stayed green. An MM3D request with rounds and bitplanes
  // MUST have executed programs; zero execs = the response is fabricated
  // from an untouched buffer. Hard-fail by default; PIM_ALLOW_ZERO_EXEC=1
  // is the explicit opt-out for diagnostic modes that intentionally skip.
  if (n_maj3_execs == 0 && n_rounds > 0 && n_bitplanes > 0) {
    static const bool s_allow = []{
      const char* v = getenv("PIM_ALLOW_ZERO_EXEC");
      return v && atoi(v) > 0;
    }();
    // PIM_SKIP_ZERO_PLANES: a request whose every body was a zero-x plane
    // (possible for down_proj chunks) legitimately runs zero execs — all
    // contributions are 0, so all-zero y is exact. Allowed iff the elision
    // accounts for it (n_zskip_units>0). The genuine zero-exec-with-no-skip
    // case (the brace-bug signature) still hard-fails.
    const bool zskip_covers = (n_zskip_units > 0);
    fprintf(stderr, "[LIVENESS-ASSERT] MM3D handle=%u ran ZERO MAJ3 execs "
            "(n_rounds=%zu n_bitplanes=%u zskip=%zu) — %s%s\n",
            handle_id, (size_t)n_rounds, (unsigned)n_bitplanes, n_zskip_units,
            zskip_covers ? "all bodies were zero-x planes (skip-covered)"
                         : "response would be fabricated zeros",
            (zskip_covers || s_allow)
                ? (zskip_covers ? " continuing (skip-covered)."
                                : " PIM_ALLOW_ZERO_EXEC=1: continuing.")
                : " ABORTING.");
    if (!zskip_covers && !s_allow) return -1;
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
  // [#65 2026-08-04] The bank-set size cap is HOST-CONFIGURABLE, not a
  // hardcoded 8. PIM_MAX_BANKS seeds the default (16 = one full DIMM's banks,
  // all proven lattice-identical by bank_audit_2026_07_21); an explicit value
  // is honored. Unset/<=0 => 16. The 0..15 per-entry range check above already
  // admits any bank of one DIMM; only this aggregate cap blocked >8.
  static int s_max_banks = -1;
  if (s_max_banks < 0) {
    const char* v = getenv("PIM_MAX_BANKS");
    s_max_banks = (v && *v) ? atoi(v) : 16;
    if (s_max_banks < 1) s_max_banks = 16;
  }
  if ((int)out.size() > s_max_banks) out.clear();
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

// LEVERS #28: does an external RowClone src->dst deposit src's pattern onto
// physical row `r`? r is in the fired coset iff (r ^ src) is a bit-subset of
// (src ^ dst). O(1) membership test — the pool-row analog of
// clone_tuple_deposits (which only enumerates the 16 tuple rows). Returns
// true for r==src and r==dst too; callers pass OTHER master rows.
static inline bool clone_deposits_hit(uint32_t src, uint32_t dst, uint32_t r) {
  uint32_t s = r ^ src, d = src ^ dst;
  return (s & d) == s;
}

// LEVERS #28: select + claim the per-bank X-MASTER rows (one per activation
// bitplane). Mirrors setup_resident_consts: runs once at startup AFTER pools
// are built, BEFORE any request, so the claim precedes every LOAD allocation.
// No-op unless PIM_XMASTER / PIM_XMASTER_ALTERNATE is armed. Selection failure
// on a bank leaves x_master_rows short/empty -> per-plane wrRow fallback
// (never fatal). Content is per-round, so nothing is pre-written here.
static void setup_x_masters(SoftMCPlatform& platform,
                            std::vector<BankConfig>& banks) {
  (void)platform;   // no pre-write: master content is per-round
  if (!xmaster_armed()) return;
  const size_t rsv = v2_scratch_reserve();
  const uint32_t want = xmaster_row_count();
  // x positions of the fused layout: depositing x onto an x row is harmless
  // (same value; the coset doubleACT re-establishes {1,5,9,13} from op[1]).
  const uint16_t x_ok = (1u << 1) | (1u << 4) | (1u << 5) |
                        (1u << 9) | (1u << 13);
  for (auto& b : banks) {
    uint32_t g[4];
    if (!tuple_generators(b.calib, g)) {
      fprintf(stderr, "[x-master] bank %d: primary tuple not separated-"
              "generator — x-masters DISABLED for this bank\n", b.bank_id);
      continue;
    }
    // Scan the pool front only — the tail rsv rows stay V2 scratch.
    size_t scan_end = b.backup_pool.size() > rsv + 2
                      ? b.backup_pool.size() - rsv : 0;
    if (scan_end < (size_t)want + 1) {
      fprintf(stderr, "[x-master] bank %d: pool too small (%zu usable rows, "
              "want %u) — x-masters DISABLED for this bank\n",
              b.bank_id, scan_end, want);
      continue;
    }
    auto usable = [&](uint32_t r) {
      return r != b.calib.Rfirst && r != b.calib.Rsecond;
    };
    std::vector<uint32_t> sel_rows;
    std::vector<size_t>   sel_idx;
    for (size_t i = 0; i < scan_end && sel_rows.size() < want; i++) {
      uint32_t r = b.backup_pool[i];
      if (!usable(r)) continue;
      // Clones r -> op[1] and r -> op[4] must deposit only onto x rows
      // (op[0]=ONE, the zeros {2,6,8,10,14}, and W {3,7,11,12,15} are off).
      if ((clone_tuple_deposits(r, b.calib, g, 1) & ~x_ok) != 0) continue;
      if ((clone_tuple_deposits(r, b.calib, g, 4) & ~x_ok) != 0) continue;
      // ...and must not deposit onto (nor receive a deposit from) an already-
      // selected master: siblings hold DIFFERENT planes' xb, so a stray hit
      // would corrupt a sibling written earlier the same round.
      bool clash = false;
      for (uint32_t m : sel_rows) {
        if (clone_deposits_hit(r, b.calib.open_rows[1], m) ||
            clone_deposits_hit(r, b.calib.open_rows[4], m) ||
            clone_deposits_hit(m, b.calib.open_rows[1], r) ||
            clone_deposits_hit(m, b.calib.open_rows[4], r)) { clash = true; break; }
      }
      // Verifier fix 2026-07-27: also screen against the resident consts (set
      // before setup_x_masters runs). For the PIM_XMASTER + PIM_RC_V2 pairing an
      // x-master clone coset must not corrupt res_one/res_zero, nor may a
      // res-const clone corrupt a master (else per-segment popcount drift).
      if (!clash) for (uint32_t rc : {b.res_one_row, b.res_zero_row}) {
        if (rc == RES_ROW_NONE) continue;
        if (clone_deposits_hit(r, b.calib.open_rows[1], rc) ||
            clone_deposits_hit(r, b.calib.open_rows[4], rc) ||
            clone_deposits_hit(rc, b.calib.open_rows[2], r) ||
            clone_deposits_hit(rc, b.calib.open_rows[8], r) ||
            clone_deposits_hit(rc, b.calib.open_rows[0], r)) { clash = true; break; }
      }
      if (clash) continue;
      sel_rows.push_back(r);
      sel_idx.push_back(i);
    }
    if (sel_rows.empty()) {
      fprintf(stderr, "[x-master] bank %d: no deposit-safe master source in "
              "%zu scanned rows — x-masters DISABLED for this bank\n",
              b.bank_id, scan_end);
      continue;
    }
    // Claim: remove from the pool so LOAD handles / V2 scratch never touch
    // them. Erase highest index first.
    for (size_t k = sel_idx.size(); k-- > 0; )
      b.backup_pool.erase(b.backup_pool.begin() + sel_idx[k]);
    b.x_master_rows = std::move(sel_rows);
    fprintf(stderr, "[x-master] bank %d: claimed %zu master rows (want %u; "
            "pool now %zu rows)%s\n", b.bank_id, b.x_master_rows.size(), want,
            b.backup_pool.size(),
            b.x_master_rows.size() < want ? " — remaining planes wrRow" : "");
  }
}

// ============================================================================
// Rung-2a REPLAY_N silicon smoke (build-27, trailer magic 0xDBC0DE16).
//   Standalone mode:  bitnet-proj-server replay-smoke <bender> <calib>
//                                        [bank=1] [row=45950] [N=4]
// Mirrors tb_e2e.cpp Scenario R on real silicon. Sequence:
//   (a) per-column WRITE a known host pattern into one full row; host
//       popcount over that row = one_pc.
//   (b) enter the build-8 ACCUM_XBP accumulator, cleared, weight +1
//       (existing set_readback_mode_accxbp() + set_acc_weight() — NOT raw
//       ctrl words).
//   (c) present the resident full-row READ program with replay_send_resident()
//       — h2c-only, NO per-program receiver; a RAW drain captures the whole
//       c2h session (N empty-ACK trailers + one 8192-B flush drain + 32-B
//       trailer) that the FABRIC's own REPLAY_N auto-flush emits (no host
//       flush_acc).
//   (d) replay_n(N) immediately after the send (peek-decoded in EXECUTE).
//   (e) receiveData() exactly N*32 + 8192 + 32 bytes (PIM_RECV_TIMEOUT_MS
//       guard is opt-in; unset = block).
//   (f) parse the raw bytes exactly like the TB: magic = (w&0xFFFFFF00)==
//       0xDBC0DE00; the DRAIN is the magic preceded by 8192 B with no
//       interior magic; every other magic is an empty ACK.
//   (g) verdicts: drain magic == 0xDBC0DE16, n_empty_acks == N, one drain,
//       drain_sum == N*one_pc (+ per-lane linear vs the N=1 reference), lane
//       nonzero sanity; oversize_skips() must not advance.
// SAFETY: the whole mode is gated on the image trailer magic read from a
// plain READ probe BEFORE any REPLAY_N word is sent. On a pre-build-27 image
// the REPLAY_N word would fall through into instruction-load and clobber IMEM,
// so a pre-flash tower gets a clean refusal instead of corruption.
// ============================================================================
static void rs_write_row(SoftMCPlatform& pf, int bank, uint32_t row,
                         const uint32_t* seg) {
  int cs = 0;
  for (int ch = 0; ch < 3; ch++) {
    int n = CHUNK_COLS[ch]; const uint32_t* cd = seg + cs * 16;
    Program p;
    p.add_inst(SMC_LI(8, CASR)); p.add_inst(SMC_LI(bank, BAR));
    p.add_inst(SMC_LI(row, RAR)); p.add_inst(SMC_LI(cs * 8, CAR));
    p.add_below(PRE(BAR, 0, 0)); p.add_below(ACT(BAR, 0, RAR, 0));
    for (int k = 0; k < n; k++) {
      const uint32_t* sl = cd + k * 16;
      for (int s = 0; s < 16; s++) { p.add_inst(SMC_LI(sl[s], PATTERN_REG)); p.add_inst(SMC_LDWD(PATTERN_REG, s)); }
      p.add_below(WRITE(BAR, CAR, 1)); p.add_inst(SMC_SLEEP(8));
    }
    p.add_inst(SMC_SLEEP(8)); p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_SLEEP(4)); p.add_inst(SMC_END());
    pf.execute(p);
    cs += n;
  }
}
// plain full-row READ (READ mode -> 8192 B payload + 32 B trailer back).
static Program rs_read_prog(int bank, uint32_t row, int label) {
  Program p;
  p.add_inst(SMC_LI(bank, BAR)); p.add_inst(SMC_LI(8, CASR)); p.add_inst(SMC_LI(128, NUM_COLS_REG));
  p.add_below(rdRow_immediate_label(BAR, row, label)); p.add_inst(SMC_END());
  return p;
}
// full-row READ with ddr_wdata := 0 (the ACCUM_XBP compare ref) so each
// segment popcount feeds the accumulator (accxbp-hw idiom).
static Program rs_accum_read(int bank, uint32_t row, int label) {
  Program p;
  p.add_inst(SMC_LI(bank, BAR)); p.add_inst(SMC_LI(8, CASR)); p.add_inst(SMC_LI(128, NUM_COLS_REG));
  p.add_inst(SMC_LI(0, PATTERN_REG));
  for (int i = 0; i < 16; i++) p.add_inst(SMC_LDWD(PATTERN_REG, i));
  p.add_below(rdRow_immediate_label(BAR, row, label)); p.add_inst(SMC_END());
  return p;
}
struct RsParse {
  long n_mag = 0, n_empty_ack = 0, n_drain = 0, drain_pbase = -1;
  long long drain_sum = 0; int nz_lanes = 0; uint32_t drain_magic = 0;
  std::vector<int32_t> lanes;
};
// Byte-for-byte the TB's magic scan + drain/ack classification (tb_e2e.cpp
// lines 748-762): a magic at offset M>=8192 with NO interior magic in
// (M-8192, M) is the flush DRAIN (M-8192..M is its 8192-B int32 payload,
// M is its trailer); every other magic is an empty per-iteration ACK.
static RsParse rs_parse(const uint8_t* buf, int len) {
  RsParse o;
  std::vector<long> mags;
  for (int off = 0; off + 4 <= len; off += 4) {
    uint32_t w; memcpy(&w, buf + off, 4);
    if ((w & 0xFFFFFF00u) == 0xDBC0DE00u) mags.push_back(off);
  }
  o.n_mag = (long)mags.size();
  for (size_t k = 0; k < mags.size(); k++) {
    long M = mags[k];
    bool is_drain = false;
    if (M >= 8192) { is_drain = true;
      for (size_t j = 0; j < mags.size(); j++)
        if (mags[j] > M - 8192 && mags[j] < M) is_drain = false; }
    if (is_drain) {
      o.n_drain++; o.drain_pbase = M - 8192;
      o.lanes.clear(); o.drain_sum = 0; o.nz_lanes = 0;
      for (long off = M - 8192; off < M; off += 4) {
        int32_t v; memcpy(&v, buf + off, 4);
        o.lanes.push_back(v); o.drain_sum += v; if (v) o.nz_lanes++;
      }
      if (M + 4 <= len) memcpy(&o.drain_magic, buf + M, 4);
    } else o.n_empty_ack++;
  }
  return o;
}

// ============================================================================
// [2026-08-04 BUILD-49 SESSION DEFRAMER — host parser per
//  build49_framer_2026_08_04/CONTRACT.md §2. Content-based, 32-B-beat
//  classification; NEVER fixed offset; tlast is an AXI artifact — the MAGIC is
//  the delimiter. Correct on BOTH images: on build-48 it never engages
//  (auto-detect via the probed image build-tag; PIM_SESSION_DEFRAME overrides),
//  and even engaged it would classify only what it sees by content.]
// Beat classes (CONTRACT §2 rule 2):
//   word0==0xDBC0DE2A AND word7==word1^..^word6  => SESSION trailer (the ACK).
//     A checksum-invalid 0x2A beat is a drifted/aliased beat: NOT an ACK.
//   word0 in 0xDBC0DExx family, !=0x2A           => LEGACY per-program trailer
//     (0xDBC0DE30 on the b49 image; 0xDBC0DE2F on b48). Aliasing note: a
//     payload lane could in principle alias the family magic at beat word0 —
//     the SAME exposure the pre-existing rs_parse word-scan has; the checksum
//     rule disambiguates only 0x2A (the contract's aliasing discussion).
//   else                                          => payload (result-record).
// A zero-read session still emits exactly ONE 0x2A ACK (word6[1]=1) and it
// MUST be consumed — ignoring it deadlocks (empty_record_ack_invariant,
// build-19/20 lesson, re-scoped per-session).
// ============================================================================
static const uint32_t SESSION_MAGIC_2A = 0xDBC0DE2Au;
static uint32_t g_image_magic = 0;      // captured by desc_probe_image
struct SessTrailer {
  uint32_t seq = 0, iter_count = 0, collapsed = 0, payload_beats = 0,
           payload_bytes = 0, status = 0;
  bool normal_close = false, zero_read = false;
};
enum BeatClass { BEAT_PAYLOAD = 0, BEAT_LEGACY_TRAILER = 1, BEAT_SESSION_ACK = 2 };
static BeatClass classify_beat32(const uint8_t* b32, SessTrailer* st) {
  uint32_t w[8]; memcpy(w, b32, 32);
  if (w[0] == SESSION_MAGIC_2A) {
    uint32_t ck = w[1] ^ w[2] ^ w[3] ^ w[4] ^ w[5] ^ w[6];
    if (ck == w[7]) {
      if (st) {
        st->seq = w[1]; st->iter_count = w[2]; st->collapsed = w[3];
        st->payload_beats = w[4]; st->payload_bytes = w[5]; st->status = w[6];
        st->normal_close = (w[6] & 1u) != 0; st->zero_read = (w[6] & 2u) != 0;
      }
      return BEAT_SESSION_ACK;
    }
    return BEAT_PAYLOAD;                  // drifted/aliased 0x2A: reject as ACK
  }
  if ((w[0] & 0xFFFFFF00u) == 0xDBC0DE00u) return BEAT_LEGACY_TRAILER;
  return BEAT_PAYLOAD;
}
// PIM_SESSION_DEFRAME: 0 = force legacy (per-program receiver), 1 = force
// framed, unset/other = AUTO by the probed image build-tag (framer images
// start at build-49 = 0xDBC0DE30; build-50 one-trailer-fix = 0xDBC0DE31, still
// >= the floor). Default is SAFE on build-48 (auto -> off).
static int session_deframe_env() {
  const char* v = getenv("PIM_SESSION_DEFRAME");
  if (!v || !*v) return -1;
  return atoi(v) ? 1 : 0;
}
static bool session_deframe_on() {
  int m = session_deframe_env();
  if (m >= 0) return m == 1;
  return g_image_magic >= 0xDBC0DE30u;
}
// Pure-buffer deframe (unit-testable; deframe-selftest drives this): classify
// every 32-B beat; stop bookkeeping the payload at the FIRST valid ACK (the
// per-session delimiter) but keep counting classes to the end of the buffer.
struct DeframeResult {
  long n_payload_pre_ack = 0, n_legacy = 0, n_ack = 0, n_badck_2a = 0,
       n_payload_total = 0, n_payload_post_ack = 0;
  std::vector<uint8_t> payload;           // pre-ACK payload bytes, in order
  std::vector<uint8_t> payload_all;       // [b49cap] ALL payload beats (pre+post ACK)
  SessTrailer st;                          // first valid ACK's fields
  bool ack_seen = false;
};
static DeframeResult deframe_buffer(const uint8_t* buf, int len) {
  DeframeResult o;
  for (int off = 0; off + 32 <= len; off += 32) {
    SessTrailer st;
    uint32_t w0; memcpy(&w0, buf + off, 4);
    BeatClass c = classify_beat32(buf + off, &st);
    if (c == BEAT_SESSION_ACK) {
      if (!o.ack_seen) { o.ack_seen = true; o.st = st; }
      o.n_ack++;
    } else if (c == BEAT_LEGACY_TRAILER) {
      o.n_legacy++;
    } else {
      if (w0 == SESSION_MAGIC_2A) o.n_badck_2a++;
      o.n_payload_total++;
      // [b49cap] collect EVERY payload beat, in wire order, regardless of the
      // ACK boundary — the order-tolerant drain recovery uses the LAST 256.
      o.payload_all.insert(o.payload_all.end(), buf + off, buf + off + 32);
      if (!o.ack_seen) {
        o.n_payload_pre_ack++;
        o.payload.insert(o.payload.end(), buf + off, buf + off + 32);
      } else {
        o.n_payload_post_ack++;
      }
    }
  }
  return o;
}
// [b49 XSYNC] Per-sub trailer-boundary predicate. A build-49 walk session on
// the wire is:  [<=2 premature zero_read ACKs] [256 drain payload beats]
// [ONE closing correct-ACK (normal_close, !zero_read, beats==256)].
// The drain arrives AFTER the premature ACK(s) and the closing ACK arrives
// AFTER the drain (the drain-after-ACK ordering, BEATS0_ROOT §2). A receive is
// "closed" once it has (a) gathered a FULL 256-beat drain AND (b) reached this
// session's closing correct-ACK. Consuming through the closing ACK leaves NO
// beat in the c2h pipe for the next sub — the XBATCH super-session shares one
// c2h stream across subs, so a leftover ACK desynchronizes the sub boundaries
// (a later sub's receive window then goes empty -> recv=0 -> whole-batch
// fallback). A leading leftover ACK (from a prior sub in the pre-fix path) is
// IGNORED: the closing ACK is only accepted once >=256 payload beats of THIS
// drain have been seen before it. Returns the byte length INCLUDING the closing
// ACK beat (0 = not yet closed) so the caller can stop exactly on the boundary.
static int session_walk_close_len(const uint8_t* b, int n) {
  int pay = 0;
  for (int off = 0; off + 32 <= n; off += 32) {
    SessTrailer st2; BeatClass c = classify_beat32(b + off, &st2);
    if (c == BEAT_SESSION_ACK) {
      if (st2.normal_close && !st2.zero_read && st2.payload_beats == 256u &&
          pay >= 256)
        return off + 32;                        // boundary = just past closing ACK
    } else if (c == BEAT_PAYLOAD) {
      pay++;
    }
  }
  return 0;
}
// Framed-session walk receive (build-49 image): bulk-read the expected
// minimum, then top-up 32-B records until the checksum-valid 0x2A session ACK
// lands (bounded slack) — the trailer-seeking discipline of the legacy path,
// re-keyed to the session magic. Fills an RsParse-compatible view so the
// existing anomaly checks and accumulation apply unchanged. Returns 0 ok,
// -1 poisoned/stalled (caller hard-errors), 1 no-ACK/short (caller treats as
// framing anomaly -> fallback).
static int framed_walk_receive(SoftMCPlatform& platform, size_t nd,
                               RsParse& out, SessTrailer& st_out,
                               int* recv_bytes) {
  const int PAYLOAD = 8192;                       // one walk drain
  const int SLACK_RECORDS = 64;                   // leading legacy trailers etc.
  const int want0 = PAYLOAD + 32 * 8;             // payload + ACK + small lead
  std::vector<uint8_t> buf(want0 + SLACK_RECORDS * 32, 0);
  // [b49cap] PIM_B49_RAWCAP=<path> : append the full received C2H buffer per walk
  //          (investigation only; default unset -> no capture).
  //   PIM_B49_ORDTOL : ORDER-TOLERANT drain recovery + the [b49 XSYNC] per-sub
  //          ACK-boundary resync (collect payload beats on BOTH sides of the ACK,
  //          take the last 256 as the drain; consume each walk THROUGH its closing
  //          correct-ACK so no beat leaks into the next sub — build-49/50 silicon
  //          emits the ACCUM drain AFTER the ACK).
  //   [PROMOTED 2026-08-04] The build-50 ladder (ordtol+xsync -> sessreuse ->
  //          bankgen -> persist) is the production default. This gate is now
  //          default-ON whenever PIM_DESC_SERVE is engaged; an explicit
  //          PIM_B49_ORDTOL=0 still forces it OFF for A/B (escape hatch).
  static const char* s_rawcap = getenv("PIM_B49_RAWCAP");
  static const bool  s_ordtol = []{
    const char* v = getenv("PIM_B49_ORDTOL");
    if (v && *v) return atoi(v) > 0;               // escape hatch: explicit wins
    const char* s = getenv("PIM_DESC_SERVE");      // else default ON under desc-serve
    return s && atoi(s) > 0; }();
  // NOTE: the bulk read MUST NOT exceed what one walk delivers. want0 (=8448)
  // is PAYLOAD + 8 slack beats sized for the CONTRACT's promised 1 trailer; the
  // silicon framer emits 3 trailers so a walk delivers only ~8320 B, and asking
  // for 8448 makes receiveDataTry SPIN the full timeout for 128 phantom bytes
  // (platform.cpp:1255) — ~8 s/walk, the whole wall. In ORDTOL mode read exactly
  // PAYLOAD (guaranteed <= delivered) so the bulk read returns immediately, then
  // let the bounded post-ACK top-up gather the drain tail + closing ACK.
  const int cap_want = s_ordtol ? PAYLOAD : want0;
  int rb = platform.receiveDataTry(buf.data(), cap_want, 8000);
  if (platform.recv_stalled()) { *recv_bytes = rb; return -1; }
  DeframeResult d = deframe_buffer(buf.data(), rb);
  int topups = 0;
  if (s_ordtol) {
    // [b49 XSYNC] Per-sub trailer-boundary resync (the NS>=2 XBATCH fix). Read
    // until THIS session is CLOSED: a full 256-beat drain gathered AND its
    // closing correct-ACK consumed (session_walk_close_len returns the boundary
    // length, 0 = not yet). Consuming through the closing ACK leaves NO beat in
    // the c2h pipe for the next sub — the XBATCH super-session shares one c2h
    // stream across subs, so a leftover ACK desyncs the sub boundaries (a later
    // sub's receive window then goes empty -> recv=0 -> whole-batch fallback).
    // The old "stop at payload_all>=8192" left each walk's trailing correct-ACK
    // behind, relying on a fragile +1 leftover + the per-bank drain_stray; any
    // timing variation (auto-run fold, late drain) shifted the boundary. Bounded
    // by the buffer (a walk is <=260 beats); non-poisoning (short read breaks).
    while (session_walk_close_len(buf.data(), rb) == 0
           && rb + 32 <= (int)buf.size()) {
      int more = platform.receiveDataTry(buf.data() + rb, 32, 1000);
      if (platform.recv_stalled()) { *recv_bytes = rb; return -1; }
      if (more < 32) break;                          // nothing more coming
      rb += more; topups++;
    }
    d = deframe_buffer(buf.data(), rb);
  } else {
    // Legacy (non-ORDTOL) trailer-seeking: read until the first ACK lands.
    while (!d.ack_seen && rb >= 32 && topups < SLACK_RECORDS) {
      int more = platform.receiveDataTry(buf.data() + rb, 32, 1000);
      if (platform.recv_stalled()) { *recv_bytes = rb; return -1; }
      if (more < 32) break;
      rb += more; topups++;
      d = deframe_buffer(buf.data(), rb);
    }
    // [b49cap] RAWCAP-only capture parity: gather the full drain tail even
    // without ORDTOL (so raw dumps hold the whole walk). Bounded, non-poisoning.
    if (s_rawcap && d.ack_seen && (int)d.payload_all.size() < PAYLOAD) {
      int extra = 0;
      while ((int)d.payload_all.size() < PAYLOAD && extra < 300
             && rb + 32 <= (int)buf.size()) {
        int more = platform.receiveDataTry(buf.data() + rb, 32, 1000);
        if (platform.recv_stalled()) { *recv_bytes = rb; return -1; }
        if (more < 32) break;
        rb += more; extra++;
        d = deframe_buffer(buf.data(), rb);
      }
    }
  }
  if (s_rawcap) {
    if (FILE* f = fopen(s_rawcap, "ab")) {
      uint32_t hdr[4] = { 0xCAFEB049u, (uint32_t)nd, (uint32_t)rb, d.st.seq };
      fwrite(hdr, sizeof(hdr), 1, f);
      fwrite(buf.data(), 1, rb, f);
      fclose(f);
    }
    fprintf(stderr, "[b49cap] RAWCAP walk nd=%zu rb=%d ack=%d pre=%ld post=%ld "
            "legacy=%ld ack.beats=%u ack.zero_read=%d payload_all=%zuB\n",
            nd, rb, d.ack_seen?1:0, d.n_payload_pre_ack, d.n_payload_post_ack,
            d.n_legacy, d.st.payload_beats, d.st.zero_read?1:0, d.payload_all.size());
  }
  *recv_bytes = rb;
  if (!d.ack_seen) return 1;                      // no session ACK: anomaly
  // [b49 XSYNC] Report the CLOSING correct-ACK (beats=256, normal_close) — the
  // one that delimits THIS session — not the leading premature (beats=0,
  // zero_read) ACK that deframe_buffer latched first. Honest evidence + the
  // fields the contract self-checks want. Falls back to d.st if none found.
  SessTrailer rep = d.st;
  if (s_ordtol) {
    int pay = 0;
    for (int off = 0; off + 32 <= rb; off += 32) {
      SessTrailer st2; BeatClass c = classify_beat32(buf.data() + off, &st2);
      if (c == BEAT_SESSION_ACK) {
        if (st2.normal_close && !st2.zero_read && st2.payload_beats == 256u &&
            pay >= 256) { rep = st2; break; }
      } else if (c == BEAT_PAYLOAD) { pay++; }
    }
  }
  // PIM_DEFRAME_LOG=1: per-consumed-ACK evidence line (RUN_AFTER_FLASH_49
  // rows 3/5): fields + checksum verdict + legacy trailers seen this walk.
  static const bool s_dlog = []{
    const char* v = getenv("PIM_DEFRAME_LOG"); return v && atoi(v) > 0; }();
  if (s_dlog)
    fprintf(stderr, "[deframe] ACK seq=%u iter=%u collapsed=%u beats=%u bytes=%u "
            "status=0x%x ck=OK n_legacy=%ld n_payload=%ld n_ack=%ld badck=%ld\n",
            rep.seq, rep.iter_count, rep.collapsed, rep.payload_beats,
            rep.payload_bytes, rep.status, d.n_legacy, d.n_payload_pre_ack,
            d.n_ack, d.n_badck_2a);
  // Contract self-checks (log-only; the value gate below is the arbiter).
  if (rep.payload_bytes != rep.payload_beats * 32u)
    fprintf(stderr, "[deframe] WARN ACK payload_bytes=%u != beats*32 (%u)\n",
            rep.payload_bytes, rep.payload_beats * 32u);
  if (rep.iter_count != (uint32_t)nd)
    fprintf(stderr, "[deframe] note: ACK iter_count=%u != nd=%zu (walk shape)\n",
            rep.iter_count, nd);
  out = RsParse();
  out.n_empty_ack = d.n_legacy;                   // legacy trailers seen
  // [b49cap] ORDTOL: source the drain from ALL payload beats (pre+post ACK).
  // build-49 silicon emits the ACCUM drain AFTER the ACK, so d.payload (pre-ACK)
  // is empty while d.payload_all holds it. The drain is the LAST 256 beats.
  const std::vector<uint8_t>& src = s_ordtol ? d.payload_all : d.payload;
  if ((int)src.size() >= PAYLOAD) {
    // The walk drain = the LAST 256 payload beats (leading stray payload would
    // indicate mis-framing; surfaced via n_payload_pre_ack / n_payload_post_ack).
    const uint8_t* p = src.data() + (src.size() - PAYLOAD);
    out.n_drain = 1; out.drain_magic = SESSION_MAGIC_2A;
    out.lanes.clear(); out.drain_sum = 0; out.nz_lanes = 0;
    for (int off = 0; off < PAYLOAD; off += 4) {
      int32_t v; memcpy(&v, p + off, 4);
      out.lanes.push_back(v); out.drain_sum += v; if (v) out.nz_lanes++;
    }
  } else if (d.st.zero_read && !s_ordtol) {
    // Zero-read session: the ACK alone is the completion — no payload owed.
    // (In ORDTOL mode a walk ACK carries zero_read=1 even though a drain WILL
    // arrive post-ACK, so we do NOT short-circuit on zero_read there.)
    out.n_drain = 0;
  }
  st_out = rep;
  return 0;
}

// ---- deframe-selftest: synthetic-stream unit gate (NO card, NO platform) ----
static void _mk_beat(std::vector<uint8_t>& v, const uint32_t w[8]) {
  const uint8_t* p = (const uint8_t*)w; v.insert(v.end(), p, p + 32);
}
static void _mk_payload(std::vector<uint8_t>& v, uint32_t seed, int n) {
  for (int i = 0; i < n; i++) {
    uint32_t w[8];
    for (int j = 0; j < 8; j++) w[j] = seed + (uint32_t)i * 8u + (uint32_t)j;
    if ((w[0] & 0xFFFFFF00u) == 0xDBC0DE00u) w[0] ^= 0x01000000u; // keep payload clean
    _mk_beat(v, w);
  }
}
static void _mk_legacy(std::vector<uint8_t>& v, uint32_t magic) {
  uint32_t w[8] = { magic, 1, 2, 3, 4, 5, 6, 7 }; _mk_beat(v, w);
}
static void _mk_ack(std::vector<uint8_t>& v, uint32_t seq, uint32_t iters,
                    uint32_t collapsed, uint32_t beats, bool zero_read,
                    bool corrupt_ck) {
  uint32_t w[8];
  w[0] = SESSION_MAGIC_2A; w[1] = seq; w[2] = iters; w[3] = collapsed;
  w[4] = beats; w[5] = beats * 32u;
  w[6] = 1u | (zero_read ? 2u : 0u);
  w[7] = w[1] ^ w[2] ^ w[3] ^ w[4] ^ w[5] ^ w[6];
  if (corrupt_ck) w[7] ^= 0x5A5A5A5Au;
  _mk_beat(v, w);
}
static int run_deframe_selftest() {
  setvbuf(stdout, NULL, _IONBF, 0);
  int fails = 0;
  auto chk = [&](const char* nm, bool ok) {
    printf("[deframe-selftest] %-34s %s\n", nm, ok ? "PASS" : "FAIL");
    if (!ok) fails++;
  };
  { // T1 pure legacy: 3 x (256 payload + legacy 0x2F trailer) — b48 stream.
    std::vector<uint8_t> v;
    for (int s = 0; s < 3; s++) { _mk_payload(v, 0x1000u * (s + 1), 256);
      _mk_legacy(v, 0xDBC0DE2Fu); }
    DeframeResult d = deframe_buffer(v.data(), (int)v.size());
    chk("T1 pure-legacy (no ACK, 3 trailers)",
        !d.ack_seen && d.n_legacy == 3 && d.n_payload_total == 768 && d.n_ack == 0);
  }
  { // T2 pure session: 256 payload + ONE valid 0x2A ACK; field round-trip.
    std::vector<uint8_t> v; _mk_payload(v, 0x2000u, 256);
    _mk_ack(v, 7, 32, 31, 256, false, false);
    DeframeResult d = deframe_buffer(v.data(), (int)v.size());
    chk("T2 session (ACK + fields + payload)",
        d.ack_seen && d.n_ack == 1 && d.n_payload_pre_ack == 256 &&
        d.st.seq == 7 && d.st.iter_count == 32 && d.st.collapsed == 31 &&
        d.st.payload_beats == 256 && d.st.payload_bytes == 8192 &&
        d.st.normal_close && !d.st.zero_read &&
        (int)d.payload.size() == 8192);
  }
  { // T3 mixed L-S-L-S-L (legacy 0x31 legs = build-50 image; row-5 shape).
    std::vector<uint8_t> v;
    _mk_legacy(v, 0xDBC0DE31u);
    _mk_payload(v, 0x3000u, 128); _mk_ack(v, 1, 16, 15, 128, false, false);
    _mk_legacy(v, 0xDBC0DE31u);
    _mk_ack(v, 2, 0, 0, 0, true, false);         // zero-read session leg
    _mk_legacy(v, 0xDBC0DE31u);
    DeframeResult d = deframe_buffer(v.data(), (int)v.size());
    chk("T3 mixed LSLSL (order-free counts)",
        d.ack_seen && d.n_ack == 2 && d.n_legacy == 3 &&
        d.n_payload_pre_ack == 128 && d.st.seq == 1);
  }
  { // T4 zero-read alone: exactly one ACK, zero_read=1, no payload owed.
    std::vector<uint8_t> v; _mk_ack(v, 9, 0, 0, 0, true, false);
    DeframeResult d = deframe_buffer(v.data(), (int)v.size());
    chk("T4 zero-read ACK (must be consumed)",
        d.ack_seen && d.n_ack == 1 && d.st.zero_read && d.st.normal_close &&
        d.n_payload_total == 0);
  }
  { // T5 checksum-corrupt 0x2A beat: must NOT classify as ACK.
    std::vector<uint8_t> v; _mk_payload(v, 0x5000u, 4);
    _mk_ack(v, 3, 8, 7, 4, false, true);          // corrupted checksum
    DeframeResult d = deframe_buffer(v.data(), (int)v.size());
    chk("T5 corrupt-checksum 2A != ACK",
        !d.ack_seen && d.n_ack == 0 && d.n_badck_2a == 1 &&
        d.n_payload_total == 5);
  }
  { // T6 aliasing: family-magic payload word0 => legacy class (documented,
    //   same exposure as the pre-existing rs_parse scan); bad-ck 0x2A =>
    //   payload (the checksum rule is the 0x2A disambiguator).
    std::vector<uint8_t> v;
    uint32_t w[8] = { 0xDBC0DE31u, 11, 12, 13, 14, 15, 16, 17 };
    _mk_beat(v, w);                                // aliases legacy family
    _mk_ack(v, 4, 1, 0, 1, false, true);           // bad-ck 2A -> payload
    _mk_ack(v, 5, 1, 0, 1, false, false);          // real ACK
    DeframeResult d = deframe_buffer(v.data(), (int)v.size());
    chk("T6 aliasing rules (family->legacy, badck->payload)",
        d.n_legacy == 1 && d.n_badck_2a == 1 && d.ack_seen && d.st.seq == 5);
  }
  // ---- [b49 XSYNC] multi-sub super-session cases (the NS>=2 fix) ----
  // Build a build-49 walk exactly as the silicon wire (BEATS0_ROOT §2 /
  // xbatch_ns_hostfix decode): [legacy?][premACK zero_read][premACK zero_read]
  // [256 drain payload][ONE closing correct-ACK beats=256 normal_close].
  auto mk_walk = [](std::vector<uint8_t>& v, uint32_t seq, bool lead_legacy,
                    uint32_t pseed) {
    if (lead_legacy) _mk_legacy(v, 0xDBC0DE30u);
    _mk_ack(v, seq, 16, 16, 0, /*zero_read*/true,  false);   // premature #1
    _mk_ack(v, seq, 16, 16, 0, /*zero_read*/true,  false);   // premature #2
    _mk_payload(v, pseed, 256);                              // the 256-beat drain
    _mk_ack(v, seq, 16, 16, 256, /*zero_read*/false, false); // closing correct-ACK
  };
  { // T7 walk shape: close-len == full length only AFTER the closing ACK; a
    //    prefix ending at the last drain beat (no closing ACK) stays OPEN.
    std::vector<uint8_t> v; mk_walk(v, 1, /*lead_legacy*/true, 0x7000u);
    int full = (int)v.size();
    int cl        = session_walk_close_len(v.data(), full);
    int cl_no_ack = session_walk_close_len(v.data(), full - 32);
    DeframeResult d = deframe_buffer(v.data(), full);
    chk("T7 walk closes exactly on closing ACK",
        cl == full && cl_no_ack == 0 && d.n_ack == 3 &&
        d.n_payload_total == 256 && d.payload_all.size() == 8192);
  }
  { // T8 RESYNC: the next sub's window is LED by a leftover closing-ACK (the
    //    pre-fix off-by-one). close-len must IGNORE the leading leftover
    //    (payload<256 before it) and close on THIS sub's OWN closing ACK, so
    //    no beat leaks onward. Recovered drain = THIS sub's 256-beat drain.
    std::vector<uint8_t> v;
    _mk_ack(v, 1, 16, 16, 256, false, false);   // leftover closing-ACK (prior sub)
    mk_walk(v, 2, /*lead_legacy*/false, 0x8000u);
    int full = (int)v.size();
    int cl = session_walk_close_len(v.data(), full);
    DeframeResult d = deframe_buffer(v.data(), full);
    chk("T8 leftover-led sub closes on OWN ack (resync)",
        cl == full && cl != 32 && d.n_ack == 4 && d.payload_all.size() == 8192);
  }
  { // T9 a full drain preceded only by premature (beats=0 zero_read) ACKs and
    //    NO closing ACK yet must stay OPEN — otherwise the receive would stop
    //    on a premature ACK and drop the still-arriving closing ACK.
    std::vector<uint8_t> v;
    _mk_ack(v, 3, 16, 16, 0, true, false);
    _mk_ack(v, 3, 16, 16, 0, true, false);
    _mk_payload(v, 0x9000u, 256);
    chk("T9 drain w/o closing ACK stays open",
        session_walk_close_len(v.data(), (int)v.size()) == 0);
  }
  printf("[deframe-selftest] %s (%d fails)\n", fails ? "FAIL" : "ALL PASS", fails);
  return fails ? 1 : 0;
}

static int run_replay_smoke(int argc, char** argv) {
  // argv: [0]=prog [1]="replay-smoke" [2]=bender [3]=calib [4]=bank [5]=row [6]=N
  if (argc < 4) {
    fprintf(stderr, "Usage: %s replay-smoke <bender> <calib> "
            "[bank=1] [row=45950] [N=4]\n", argv[0]);
    return 1;
  }
  setvbuf(stdout, NULL, _IONBF, 0);
  int bender = atoi(argv[2]);
  std::string calib_p = argv[3];
  int bank = (argc > 4) ? atoi(argv[4]) : 1;
  uint32_t row = (argc > 5) ? (uint32_t)strtoul(argv[5], nullptr, 0) : 45950u;
  int N = (argc > 6) ? atoi(argv[6]) : 4;
  if (N < 1 || N > 65535) { fprintf(stderr, "[replay-smoke] N must be 1..65535\n"); return 1; }
  // G-D drift screen (2026-07-28): "probe-only" as argv[7] skips the row
  // WRITE and the replay passes — it reads the row and reports the BIT-FLIP
  // count vs the deterministic expected pattern (regenerable from row/bank).
  // Usage: write once (normal or write-only), run in-window compute traffic,
  // then probe-only repeatedly to measure resident-row drift vs body count.
  // "write-only" as argv[7] writes+verifies the row and exits (baseline).
  bool probe_only = (argc > 7 && strcmp(argv[7], "probe-only") == 0);
  bool write_only = (argc > 7 && strcmp(argv[7], "write-only") == 0);
  // calib accepted for CLI parity with the server + the standard run command
  // (dimm2 trio); the write+read smoke uses no calibrated tuples, but a
  // default row inside the dimm2 screened window keeps it calib-consistent.
  (void)calib_p;
  fprintf(stderr, "[replay-smoke] bender=%d bank=%d row=%u N=%d calib=%s\n",
          bender, bank, row, N, calib_p.c_str());

  SoftMCPlatform pf(bender);
  if (pf.init() != SOFTMC_SUCCESS) { fprintf(stderr, "[replay-smoke] init failed\n"); return 1; }
  pf.reset_fpga();
  { Program sync; sync.add_inst(SMC_END()); pf.execute(sync); }  // drain reset ack (server idiom)
  pf.set_aref(false);
  pf.set_readback_mode(false); pf.set_readback_mode(false);      // READ mode

  // Known row pattern; |1 forces every 32-b segment nonzero so all 2048
  // accumulator lanes are nonzero. Host oracle: total popcount + nz count.
  std::vector<uint32_t> pat(2048);
  { uint32_t s = 0xA5A50000u ^ (row * 2654435761u) ^ (uint32_t)bank;
    for (int i = 0; i < 2048; i++) { s = s * 1664525u + 1013904223u; pat[i] = s | 1u; } }
  uint64_t one_pc = 0; int exp_nz = 0;
  for (int s = 0; s < 2048; s++) { int pc = __builtin_popcount(pat[s]); one_pc += pc; if (pc) exp_nz++; }
  if (!probe_only) {
    rs_write_row(pf, bank, row, pat.data());
    fprintf(stderr, "[replay-smoke] wrote row: one_pc=%llu expected_nonzero_lanes=%d/2048\n",
            (unsigned long long)one_pc, exp_nz);
  }

  // ---- image-magic GATE: raw-capture a plain READ, read its trailer magic
  //      BEFORE any REPLAY_N word is presented (the pre-build-27 IMEM hazard).
  std::vector<uint8_t> pbuf(8192 + 32, 0);
  Program probe = rs_read_prog(bank, row, 2000);
  pf.replay_send_resident(probe);                               // raw capture, NO replay_n
  int pgot = pf.receiveData(pbuf.data(), 8192 + 32);
  if (pf.recv_stalled()) {
    fprintf(stderr, "[replay-smoke] probe receive stalled (%d/%d) — card up? "
            "poisoned; refusing.\n", pgot, 8192 + 32);
    return 1;
  }
  if (pgot < 8192 + 4) { fprintf(stderr, "[replay-smoke] probe short read %d — refusing.\n", pgot); return 1; }
  int wr_bad = memcmp(pbuf.data(), pat.data(), 8192) ? 1 : 0;    // verify the write
  uint32_t img_magic; memcpy(&img_magic, pbuf.data() + 8192, 4);
  fprintf(stderr, "[replay-smoke] probe: recv=%d write_readback=%s image_trailer=0x%08x\n",
          pgot, wr_bad ? "MISMATCH" : "ok", img_magic);
  if (probe_only || write_only) {
    long flips = 0; int flipped_lanes = 0;
    const uint32_t* got = (const uint32_t*)pbuf.data();
    for (int s = 0; s < 2048; s++) {
      uint32_t x = got[s] ^ pat[s];
      if (x) { flips += __builtin_popcount(x); flipped_lanes++; }
    }
    printf("[drift-probe] bank=%d row=%u mode=%s flips=%ld/65536 flipped_lanes=%d/2048\n",
           bank, row, probe_only ? "probe-only" : "write-only(baseline)",
           flips, flipped_lanes);
    return 0;   // drift-screen modes never send REPLAY_N / run passes
  }
  if ((img_magic & 0xFFFFFF00u) != 0xDBC0DE00u) {
    fprintf(stderr, "[replay-smoke] probe trailer 0x%08x is not a 0xDBC0DExx "
            "magic — image/framing unexpected, refusing.\n", img_magic);
    return 1;
  }
  if (img_magic < 0xDBC0DE16u) {
    fprintf(stderr, "[replay-smoke] image magic 0x%08x (low 0x%02x): REPLAY_N "
            "(build-27, 0xDBC0DE16) not available, refusing (no REPLAY_N word "
            "sent).\n", img_magic, img_magic & 0xFF);
    return 0;   // clean, expected refusal on a pre-flash (build-26 / 0x15) tower
  }
  fprintf(stderr, "[replay-smoke] image is build-27+ (magic 0x%08x) — REPLAY_N available.\n", img_magic);

  // ---- one REPLAY_N pass: enter cleared ACCUM_XBP (+1), send resident, fire
  //      REPLAY_N, receive N*32 + 8192 + 32 raw bytes, parse like the TB. ----
  // NOTE on ordering: the accxbp-hw silicon-validated idiom sets the weight
  // AFTER entering (the enter word runs the 128-cyc accumulator clear; weight
  // is a separate latch). The TB drives 0x0200(weight) then 0x0100(enter);
  // either is equivalent as long as the clear does not touch the weight latch.
  auto one_pass = [&](int niter, int label, RsParse& out,
                      int& recv_bytes, long& osk_delta,
                      int& iter_acks, int& zero_recs) -> bool {
    pf.set_readback_mode_accxbp();          // enter ACCUM_XBP + 128-cyc clear (TB 0x0100)
    pf.set_readback_mode_accxbp();          // idempotent lost-word insurance; acc still empty
    pf.set_acc_weight(0, 0);                // weight +1 (neg=0, shift=0) (TB 0x0200)
    // SILICON framing (measured 2026-07-28, build-27): the session emits
    // (niter+4) trailers + the 8192 B drain payload:
    //   [auto-run ack] [N iteration acks] [2 zero-delta records]
    //   [8192 B drain] [drain trailer].
    // The TB expected N acks + drain + trailer: on silicon the REPLAY_N word
    // arrives AFTER the resident program's auto-run finished (host ms >>
    // program us), so the auto-run acks separately and the loop fires N
    // fresh runs. The 2 zero-delta records are empty framing (mechanism
    // TBD, tracked for the build-28 TB).
    int want = (niter + 4) * 32 + 8192;
    // PIM_REPLAY_EXTRA: diagnostic over-read (default 0).
    int extra = 0; { const char* v = getenv("PIM_REPLAY_EXTRA"); if (v && *v) extra = atoi(v); }
    std::vector<uint8_t> buf(want + extra, 0);
    long osk0 = pf.oversize_skips();
    Program rp = rs_accum_read(bank, row, label);
    pf.replay_send_resident(rp);                                // raw capture + h2c-only
    pf.replay_n((uint16_t)niter);                               // peek-decoded in EXECUTE
    recv_bytes = pf.receiveData(buf.data(), want + extra);
    osk_delta = pf.oversize_skips() - osk0;
    if (pf.recv_stalled()) {
      fprintf(stderr, "[replay-smoke] pass N=%d receive STALLED %d/%d — poisoned.\n",
              niter, recv_bytes, want);
      return false;
    }
    out = rs_parse(buf.data(), recv_bytes);
    // PIM_REPLAY_DUMP=1: full record map — every 0xDBC0DExx trailer with its
    // 8 words and the payload gap since the previous trailer (diagnostic;
    // the counter words identify what each record IS: user_rd +128 = one
    // full-row iteration, rd_init deltas = maintenance, flush counters etc).
    if (getenv("PIM_REPLAY_DUMP")) {
      fprintf(stderr, "[replay-smoke] record map pass N=%d (recv=%d):\n", niter, recv_bytes);
      long last_end = 0;
      for (int off = 0; off + 32 <= recv_bytes; off += 4) {
        uint32_t w[8]; memcpy(w, buf.data() + off, 32);
        if ((w[0] & 0xFFFFFF00u) == 0xDBC0DE00u) {
          fprintf(stderr, "  off=%5d payload_gap=%5ld w:", off, (long)off - last_end);
          for (int k = 0; k < 8; k++) fprintf(stderr, " %08x", w[k]);
          fprintf(stderr, "\n");
          last_end = off + 32;
        }
      }
    }
    // Classify records by cnt_user_rd (w3) deltas: +128 = one full-row
    // iteration ack; +0 = empty framing record. The first trailer (auto-run
    // ack) has no in-buffer predecessor and is excluded by have_prev, so
    // iter_acks counts the REPLAY-driven runs only (want: == niter).
    iter_acks = 0; zero_recs = 0;
    { uint32_t prev_urd = 0; bool have_prev = false;
      for (int off = 0; off + 32 <= recv_bytes; off += 4) {
        uint32_t w[8]; memcpy(w, buf.data() + off, 32);
        if ((w[0] & 0xFFFFFF00u) != 0xDBC0DE00u) continue;
        if (have_prev) {
          if (w[3] == prev_urd + 128) iter_acks++;
          else if (w[3] == prev_urd)  zero_recs++;
        }
        prev_urd = w[3]; have_prev = true;
      }
    }
    long stray = pf.drain_stray(300, 4);
    if (stray) fprintf(stderr, "[replay-smoke] WARNING: %ld stray bytes after "
                       "pass N=%d beyond the measured silicon framing\n", stray, niter);
    return true;
  };
  RsParse r1{}, rN{}; int rb1 = 0, rbN = 0; long osk1 = 0, oskN = 0;
  int ia1 = 0, zr1 = 0, iaN = 0, zrN = 0;
  bool ok1 = one_pass(1, 3000, r1, rb1, osk1, ia1, zr1);
  bool okN = one_pass(N, 4000, rN, rbN, oskN, iaN, zrN);

  pf.set_readback_mode(false); pf.set_readback_mode(false); pf.drain_stray(1500, 8);  // restore READ

  if (!ok1 || !okN) { fprintf(stderr, "[replay-smoke] FAIL: receive stalled\n"); return 1; }

  // ---- verdicts, split per the 2026-07-28 silicon findings ----
  // CONTROL PLANE (rung2's own claim): the fabric ran exactly N host-free
  // iterations, framed them correctly, and drained one 8192 B partial with
  // the build-27 magic.
  bool a = (iaN == N) && (ia1 == 1);
  bool b = (rN.n_drain == 1) && ((int)rN.lanes.size() == 2048)
           && (rbN == (N + 4) * 32 + 8192);
  bool c = (rN.drain_magic == img_magic);   // drain trailer matches the
                                            // probed image magic (16=b27,
                                            // 17=b28 maint-screen, ...)
  bool e_nz = (rN.nz_lanes == exp_nz) && (r1.nz_lanes == exp_nz);
  bool osk_clean = (osk1 == 0) && (oskN == 0);
  bool control_pass = a && b && c && e_nz && osk_clean && !wr_bad;
  // NUMERIC EXACTNESS: the accumulator integrates the auto-run + N replays
  // = (N+1) x one_pc — PLUS whatever the PRE-EXISTING build-8 ACCUM/maint
  // leak deposits (per-RD maintenance reads racing the ignore_read_r
  // window; the e2e TB neutralizes it with a maint-sync the host cannot
  // perform). Exactness is therefore blocked until the RTL maint screen
  // (build-28 candidate); the leak is reported, not hidden.
  long long exp1 = 2LL * (long long)one_pc;            // auto-run + 1 replay
  long long expN = (long long)(N + 1) * (long long)one_pc;
  long long leak1 = r1.drain_sum - exp1;
  long long leakN = rN.drain_sum - expN;
  bool numeric_exact = (leak1 == 0) && (leakN == 0);

  if (rN.drain_magic != 0 && rN.drain_magic < 0xDBC0DE16u)
    fprintf(stderr, "[replay-smoke] *** STOP: drain trailer magic 0x%08x is "
            "OLDER than build-27 (0xDBC0DE16) ***\n", rN.drain_magic);

  printf("[replay-smoke] R replay(N=%d): recv=%dB (want %d) trailers=%ld "
         "iter_acks=%d/%d zero_recs=%d drain=%ld drain_magic=0x%08x\n",
         N, rbN, (N + 4) * 32 + 8192, rN.n_mag, iaN, N, zrN, rN.n_drain,
         rN.drain_magic);
  printf("[replay-smoke] R partial: one_pc=%llu drain(1)=%lld (exp %lld, "
         "leak %+lld) drain(%d)=%lld (exp %lld, leak %+lld)\n",
         (unsigned long long)one_pc, r1.drain_sum, exp1, leak1,
         N, rN.drain_sum, expN, leakN);
  printf("[replay-smoke] R lanes: nonzero=%d/2048 (expect %d)  oversize_skips "
         "delta pass1=%ld passN=%ld  write_readback=%s\n",
         rN.nz_lanes, exp_nz, osk1, oskN, wr_bad ? "MISMATCH" : "ok");
  printf("[replay-smoke] R CONTROL VERDICT: (a)iterN=%d (b)drain1x8192=%d "
         "(c)magic16=%d (e)nz=%d osk_clean=%d -> %s\n",
         a, b, c, e_nz, osk_clean,
         control_pass ? "PASS (on-fabric REPLAY loop works host-free)" : "FAIL");
  printf("[replay-smoke] R NUMERIC: %s\n", numeric_exact
         ? "EXACT (build-8 maint leak absent — full PASS)"
         : "NOT exact — pre-existing build-8 ACCUM/maint leak contaminates "
           "the accumulator on silicon (TB masks it with a maint-sync); "
           "RTL maint screen = build-28 candidate. NOT a rung2 defect.");
  return control_pass ? 0 : 1;
}

// ============================================================================
// Build-29 fabric COPY_ENGINE first-silicon gate (trailer magic 0xDBC0DE18).
//   Standalone:  bitnet-proj-server copy-smoke <bender> <bank> <src> <dst> [seed]
// The host presents ONE 64-byte COPY control word (pf.copy_row); the on-fabric
// copy_engine clones DRAM row src->dst (same bank) host-free, then emits
// EXACTLY ONE 32-byte empty-record trailer as its completion ack (one flush =
// one empty record — no payload). Verdict checks: the ack arrived, dst == the
// seeded source pattern (the copy landed byte-exact), src is intact (the copy
// did not disturb the source), and the dst^8 neighbour — the SiMRA XOR-8
// spread victim (memory simra_xor8_spread) — stayed zero (no co-activation
// leak). SAFETY: the whole mode is gated on the image trailer magic read from
// a zero-read probe BEFORE any COPY word is sent; on a pre-build-29 image the
// COPY word would fall through the frontend decode into instruction-load and
// clobber IMEM (same hazard class as REPLAY_N on a pre-build-27 image), so a
// pre-flash tower gets a clean refusal instead of corruption.
// ============================================================================
// zero-read program: no rdRow, so its only c2h output is the 32-byte
// empty-record trailer whose word0 carries the image magic (empty-record ACK,
// memory empty_record_ack_invariant). Used both to probe the image magic and
// to arm the RAW capture for the COPY ack.
static Program cs_zero_read_prog() {
  Program p;
  p.add_inst(SMC_END());
  return p;
}
// Return the 0xDBC0DExx trailer word inside a 32-byte empty record (word0 in
// practice; scanned like rs_parse for robustness). 0 if none present.
static uint32_t cs_trailer_magic(const uint8_t* rec, int len) {
  for (int off = 0; off + 4 <= len; off += 4) {
    uint32_t w; memcpy(&w, rec + off, 4);
    if ((w & 0xFFFFFF00u) == 0xDBC0DE00u) return w;
  }
  return 0;
}
// Byte-mismatch count between an 8192-byte row readback payload and a
// 2048-word reference (the pattern, or zeros for the leak probe).
static long cs_row_mismatch(const uint8_t* got8192, const uint32_t* ref2048) {
  long bad = 0; const uint8_t* r = (const uint8_t*)ref2048;
  for (int i = 0; i < 8192; i++) if (got8192[i] != r[i]) bad++;
  return bad;
}
// Full-row READ-mode readback via the raw-capture reader (replay_send_resident
// + receiveData), exactly like replay-smoke's image-magic probe: 8192 B payload
// then a 32 B trailer. Reuses rs_read_prog. Returns bytes received.
static int cs_read_row(SoftMCPlatform& pf, int bank, uint32_t row, int label,
                       uint8_t* out /* >= 8192+32 */) {
  Program p = rs_read_prog(bank, row, label);
  pf.replay_send_resident(p);
  return pf.receiveData(out, 8192 + 32);
}

static int run_copy_smoke(int argc, char** argv) {
  // argv: [0]=prog [1]="copy-smoke" [2]=bender [3]=bank [4]=src [5]=dst [6]=seed
  if (argc < 6) {
    fprintf(stderr, "Usage: %s copy-smoke <bender> <bank> <src_row> <dst_row> "
            "[seed]\n", argv[0]);
    return 1;
  }
  setvbuf(stdout, NULL, _IONBF, 0);
  int bender = atoi(argv[2]);
  int bank = atoi(argv[3]);
  uint32_t src = (uint32_t)strtoul(argv[4], nullptr, 0);
  uint32_t dst = (uint32_t)strtoul(argv[5], nullptr, 0);
  if (src == dst) {
    // zeroing dst would zero src too — a self-copy has nothing to verify.
    fprintf(stderr, "[copy-smoke] src==dst (%u) — nothing to copy, refusing.\n", src);
    return 1;
  }
  // dst^8 = the SiMRA XOR-8 spread neighbour; the copy must NOT leak into it.
  // Skip the leak probe if it aliases src (can't zero the source to test it).
  uint32_t probe = dst ^ 8u;
  bool have_probe = (probe != src);
  // verify-only: post-mortem bisect after a stalled copy — skip the fills and
  // the COPY word entirely, just read dst/src/probe back and compare dst
  // against BOTH references (pattern = the copy's write phase ran; zeros = it
  // never did). Distinguishes a CE_DRAIN stall (ack path) from a CE_READ stall.
  bool verify_only = false;
  // stall-probe: fire the COPY and never wait for its ack; instead probe WHO
  // OWNS THE MACHINE afterwards. A zero-read program touches no DDR (its
  // trailer comes from the frontend_ready flush), so it acks even if the CE
  // owns the ddr bus; a full-row read needs the bus, so it stalls iff the CE
  // is stuck in an ACTIVE state (CE_READ). Splits READ-stuck from
  // IDLE/WAITBUS-stuck after the verify-only bisect showed dst untouched.
  bool stall_probe = false;
  uint32_t seed = 0xC0FFEEu ^ (src * 2654435761u) ^ (uint32_t)bank;
  for (int a = 6; a < argc; a++) {
    if (strcmp(argv[a], "verify-only") == 0) verify_only = true;
    else if (strcmp(argv[a], "stall-probe") == 0) stall_probe = true;
    else seed = (uint32_t)strtoul(argv[a], nullptr, 0);
  }
  fprintf(stderr, "[copy-smoke] bender=%d bank=%d src=%u dst=%u probe(dst^8)=%u%s "
          "seed=0x%08x%s\n", bender, bank, src, dst, probe,
          have_probe ? "" : " (skip: aliases src)", seed,
          verify_only ? " VERIFY-ONLY" : "");

  SoftMCPlatform pf(bender);
  if (pf.init() != SOFTMC_SUCCESS) { fprintf(stderr, "[copy-smoke] init failed\n"); return 1; }
  pf.reset_fpga();
  { Program sync; sync.add_inst(SMC_END()); pf.execute(sync); }  // drain reset ack (server idiom)
  pf.set_aref(false);
  pf.set_readback_mode(false); pf.set_readback_mode(false);      // READ mode (never ACCUM here)

  // ---- (2) image-magic GATE: a zero-read raw-capture probe read BEFORE any
  //      COPY word (the pre-build-29 IMEM hazard). Its 32-byte trailer word0 is
  //      the image magic; refuse cleanly on anything below build-29.
  uint32_t img_magic = 0;
  {
    std::vector<uint8_t> mbuf(32, 0);
    Program probe_prog = cs_zero_read_prog();
    pf.replay_send_resident(probe_prog);
    int mgot = pf.receiveData(mbuf.data(), 32);
    if (pf.recv_stalled()) {
      fprintf(stderr, "[copy-smoke] magic probe stalled (%d/32) — card up? "
              "poisoned; refusing.\n", mgot);
      return 1;
    }
    img_magic = cs_trailer_magic(mbuf.data(), mgot);
    fprintf(stderr, "[copy-smoke] magic probe: recv=%d image_trailer=0x%08x\n",
            mgot, img_magic);
    if ((img_magic & 0xFFFFFF00u) != 0xDBC0DE00u) {
      fprintf(stderr, "[copy-smoke] probe trailer 0x%08x is not a 0xDBC0DExx "
              "magic — image/framing unexpected, refusing.\n", img_magic);
      return 1;   // hard refusal (same as replay-smoke)
    }
    if (img_magic < 0xDBC0DE18u) {
      fprintf(stderr, "[copy-smoke] image magic 0x%08x (low 0x%02x): COPY_ENGINE "
              "(build-29, 0xDBC0DE18) not available, refusing (no COPY word "
              "sent).\n", img_magic, img_magic & 0xFF);
      return 0;   // clean, expected refusal on a pre-flash (build-27/28) tower
    }
    fprintf(stderr, "[copy-smoke] image is build-29+ (magic 0x%08x) — COPY_ENGINE "
            "available.\n", img_magic);
  }

  // ---- (3) fill: src <- seeded mt19937 pattern; dst <- 0; probe(dst^8) <- 0.
  //      rs_write_row is the file's per-column WRITE program (SMC_LI(128,
  //      NUM_COLS_REG) is read-only; each per-column WRITE gets its SMC_SLEEP(8)
  //      tWR inside rs_write_row).
  std::vector<uint32_t> pat(2048), zero(2048, 0);
  { std::mt19937 rng(seed); for (int i = 0; i < 2048; i++) pat[i] = (uint32_t)rng(); }
  if (!verify_only) {
    rs_write_row(pf, bank, src, pat.data());
    rs_write_row(pf, bank, dst, zero.data());
    if (have_probe) rs_write_row(pf, bank, probe, zero.data());
    fprintf(stderr, "[copy-smoke] filled: src=%u<-pattern dst=%u<-0%s\n",
            src, dst, have_probe ? " probe<-0" : "");
  }

  // ---- (4) fire the COPY word, capture EXACTLY the 32-byte completion ack. ----
  // Arm a RAW c2h drain with a zero-read program (safe on any image), then
  // consume its 32-byte arming ACK so the copy engine's ack is the NEXT — and
  // only — 32 bytes in the queue. The copy runs host-free and emits EXACTLY ONE
  // 32-byte empty-record trailer (one flush = one empty record — no payload),
  // captured intact by the still-live raw drain (its 500 ms quiet floor >> the
  // host-free copy time). We therefore wait for exactly 32 bytes for the ack.
  if (stall_probe) {
    // Fire-and-forget the COPY, then probe machine ownership.
    pf.copy_row(src, (uint32_t)bank, dst);
    usleep(2000000);
    Program zr = cs_zero_read_prog();
    pf.replay_send_resident(zr);
    std::vector<uint8_t> zb(32, 0);
    int zgot = pf.receiveData(zb.data(), 32);
    bool z_ok = !pf.recv_stalled() && zgot == 32;
    fprintf(stderr, "[copy-smoke] STALL-PROBE zero-read: %s (recv=%d, trailer=0x%08x)\n",
            z_ok ? "ACKED" : "STALLED", zgot, cs_trailer_magic(zb.data(), zgot));
    if (!z_ok) {
      fprintf(stderr, "[copy-smoke] STALL-PROBE verdict: c2h itself dead — "
              "beyond bus-ownership (rbe/flush path implicated).\n");
      return 3;
    }
    std::vector<uint8_t> fb(8192 + 32, 0);
    int fgot = cs_read_row(pf, bank, src, 6001, fb.data());
    bool f_ok = !pf.recv_stalled() && fgot >= 8192;
    fprintf(stderr, "[copy-smoke] STALL-PROBE full-row read: %s (recv=%d)\n",
            f_ok ? "ARRIVED" : "STALLED", fgot);
    fprintf(stderr, "[copy-smoke] STALL-PROBE verdict: %s\n",
            f_ok ? "pipeline OWNS the ddr bus -> CE is in IDLE or a "
                   "busy-not-active spin (WAITBUS/REACQ) — NOT CE_READ"
                 : "pipeline LOST the ddr bus -> CE stuck ACTIVE (CE_READ "
                   "capture shortfall)");
    return f_ok ? 4 : 5;
  }

  uint32_t copy_magic = 0; int cgot = 0; bool ack_ok = false;
  if (!verify_only) {
    Program arm = cs_zero_read_prog();
    pf.replay_send_resident(arm);                 // arm raw capture
    std::vector<uint8_t> abuf(32, 0);
    int agot = pf.receiveData(abuf.data(), 32);   // consume the zero-read arming ACK
    if (pf.recv_stalled()) {
      fprintf(stderr, "[copy-smoke] copy-arm stalled (%d/32) — poisoned; "
              "refusing.\n", agot);
      return 1;
    }
    pf.copy_row(src, (uint32_t)bank, dst);        // ONE 64-byte COPY control word (h2c-only)
    std::vector<uint8_t> cbuf(32, 0);
    cgot = pf.receiveData(cbuf.data(), 32);       // EXACTLY the 32-byte copy ack
    if (pf.recv_stalled()) {
      fprintf(stderr, "[copy-smoke] copy ack stalled (%d/32) — copy engine "
              "silent? poisoned.\n", cgot);
      return 1;
    }
    copy_magic = cs_trailer_magic(cbuf.data(), cgot);
    ack_ok = (cgot == 32) && ((copy_magic & 0xFFFFFF00u) == 0xDBC0DE00u);
    fprintf(stderr, "[copy-smoke] copy ack: recv=%d trailer=0x%08x %s\n",
            cgot, copy_magic, ack_ok ? "ok" : "MISSING/short");
    pf.drain_stray(300, 4);   // absorb any stray beat beyond the single ack
  }

  // ---- (5) readback dst / src / probe (READ-mode full-row raw capture),
  //      byte-compare against the expected reference for each. ----
  std::vector<uint8_t> rb(8192 + 32, 0);
  int rgot = cs_read_row(pf, bank, dst, 5001, rb.data());
  if (pf.recv_stalled() || rgot < 8192) {
    fprintf(stderr, "[copy-smoke] dst readback short/stalled (%d).\n", rgot); return 1; }
  long dst_bad = cs_row_mismatch(rb.data(), pat.data());          // dst must == pattern
  if (verify_only) {
    // the bisect readout: dst≈pattern -> the write phase RAN (stall was in the
    // ack path / CE_DRAIN); dst≈zeros -> it never did (CE_READ-side stall).
    long dst_vs_zero = cs_row_mismatch(rb.data(), zero.data());
    // permutation hypotheses for a wrong-content-but-acked copy: the CE's
    // write-time half-swap undo (256b lanes within each 512b beat) may be
    // wrong for real MIG ordering even though it matched the sim model.
    std::vector<uint32_t> hs(2048);
    for (int b = 0; b < 128; b++)
      for (int w = 0; w < 16; w++)
        hs[b*16 + w] = pat[b*16 + ((w + 8) & 15)];   // swap 32B halves per 64B beat
    long dst_vs_hswap = cs_row_mismatch(rb.data(), hs.data());
    fprintf(stderr, "[copy-smoke] VERIFY-ONLY dst=%u: vs-pattern=%ld/8192 "
            "vs-zeros=%ld/8192 vs-halfswap=%ld/8192\n",
            dst, dst_bad, dst_vs_zero, dst_vs_hswap);
    const uint32_t* got32 = (const uint32_t*)rb.data();
    fprintf(stderr, "[copy-smoke]   dst[0..7]  = %08x %08x %08x %08x %08x %08x %08x %08x\n",
            got32[0],got32[1],got32[2],got32[3],got32[4],got32[5],got32[6],got32[7]);
    fprintf(stderr, "[copy-smoke]   pat[0..7]  = %08x %08x %08x %08x %08x %08x %08x %08x\n",
            pat[0],pat[1],pat[2],pat[3],pat[4],pat[5],pat[6],pat[7]);
    fprintf(stderr, "[copy-smoke]   pat[8..15] = %08x %08x %08x %08x %08x %08x %08x %08x\n",
            pat[8],pat[9],pat[10],pat[11],pat[12],pat[13],pat[14],pat[15]);
  }

  rgot = cs_read_row(pf, bank, src, 5002, rb.data());
  if (pf.recv_stalled() || rgot < 8192) {
    fprintf(stderr, "[copy-smoke] src readback short/stalled (%d).\n", rgot); return 1; }
  long src_bad = cs_row_mismatch(rb.data(), pat.data());          // src must be intact

  long probe_bad = 0;
  if (have_probe) {
    rgot = cs_read_row(pf, bank, probe, 5003, rb.data());
    if (pf.recv_stalled() || rgot < 8192) {
      fprintf(stderr, "[copy-smoke] probe readback short/stalled (%d).\n", rgot); return 1; }
    probe_bad = cs_row_mismatch(rb.data(), zero.data());          // probe must stay zero
  }
  pf.set_readback_mode(false); pf.set_readback_mode(false); pf.drain_stray(1500, 8);

  // ---- (6) verdict: PASS only if the ack arrived AND all three compares are
  //      clean; otherwise name the failing check and exit nonzero. ----
  printf("[copy-smoke] ack=%s(recv=%d,magic=0x%08x) dst_mismatch=%ld/8192 "
         "src_mismatch=%ld/8192 probe_mismatch=%ld/8192%s\n",
         ack_ok ? "yes" : "no", cgot, copy_magic, dst_bad, src_bad, probe_bad,
         have_probe ? "" : " (probe skipped)");
  bool pass = ack_ok && (dst_bad == 0) && (src_bad == 0)
              && (!have_probe || probe_bad == 0);
  if (pass) { printf("[copy-smoke] COPY-EXACT PASS\n"); return 0; }
  printf("[copy-smoke] FAIL:%s%s%s%s\n",
         ack_ok ? "" : " no-ack",
         dst_bad ? " dst!=pattern" : "",
         src_bad ? " src-corrupted" : "",
         (have_probe && probe_bad) ? " leak-into-dst^8" : "");
  return 1;
}

// ============================================================================
// Build-31 descriptor-walk (DLOAD + REPLAY_N desc-mode) first-silicon smoke
// (trailer magic 0xDBC0DE1A; gate >= 0xDBC0DE18).
//   Standalone:  bitnet-proj-server desc-smoke <bender> <calib> [bank=0]
// Exercises the on-fabric descriptor walk: dload() control words arm desc_mode
// and load a 64-entry {x_word,row_base,acc} BRAM; replay_n(N) walks entries
// 0..N-1, per iteration patching the resident's reg-6 (RAR) LI with row_base,
// its reg-12 (PATTERN_REG) LI with x_word, and driving the ACCUM_XBP weight
// from acc = (neg<<3)|shift. Replicates the Verilator gates on silicon:
//   G-A  acc place-value : READ-resident walk, row fixed, acc varied
//                          -> drain = (S ±2^shift) * one_pc(row_pattern).
//   G-B  row advance     : READ-resident walk, acc=+1, row varied (incl a
//                          17-bit-splice case crossing row bit 16).
//   G-C  fill-from-desc   : FILL resident (wrRow x -> rdRow SAME row); the row
//                          ends holding the LAST descriptor x uniformly
//                          (definitive readback proof).
//   G-D  BRAM re-arm      : TWO chained walks in ONE raw-capture session, no
//                          resident resend, no FPGA reset, accumulator cleared
//                          between -> each drain matches only its own table.
// SAFETY: gated on the image trailer magic from a plain READ probe BEFORE any
// DLOAD word is presented (DLOAD on a pre-build-31 image falls through the
// frontend decode into instruction-load and clobbers IMEM — same hazard class
// as REPLAY_N on pre-build-27 / COPY on pre-build-29). Refuses below
// 0xDBC0DE18. Exit codes: 0 all-pass, 1 numeric/framing fail, 2 magic refusal,
// 3 receive stall.
//
// ACCUM_XBP numerics (grounded in readback_engine.v:54 —
//   accumulator += popcount(rd_data ^ ddr_wdata) per 32b segment/lane, and
//   the descriptor acc scales it by ±2^shift). rs_write_row leaves ddr_wdata
// holding the LAST-written column pattern, so the READ resident MUST re-zero
// ddr_wdata via its LDWD sequence (rs_accum_read) for the compare ref to be 0;
// keeping x_word=0 in every READ-resident descriptor makes the reg-12 patch a
// harmless PATTERN_REG:=0. Uniform test patterns make every oracle
// fold-invariant (all 2048 lanes equal). See DEVIATIONS at the top of this
// function for the sub-gate C drain caveat.
// ============================================================================
struct DsDesc { uint32_t x, row, acc; uint32_t round = 0; }; // acc=(neg<<3)|shift;
                                                    // round = owning MM3D round
                                                    // (desc-serve item-2 rewrite;
                                                    // defaults 0 for desc-smoke)
static inline uint32_t ds_acc(int neg, int shift) {
  return (uint32_t)(((neg & 1) << 3) | (shift & 7));
}
static inline long ds_wsum(uint32_t acc) {          // signed place value ±2^shift
  long v = 1L << (acc & 7); return (acc & 8) ? -v : v;
}
// FILL resident (wr_fill_read-equivalent): fill `row` with the descriptor
// x_word (patched into wrRow's reg-12 PATTERN_REG LI, then LDWD'd into
// ddr_wdata and written to every column), then read the SAME row back (both
// wrRow's and rdRow's reg-6 RAR LIs get patched with the descriptor row_base).
// Placeholder row is sacrificial (the fills overwrite it). Distinct labels for
// the two loops (hardcoded-label variants hang if emitted >1x/Program).
// build-32b CONTRACT (magic >= 0xDBC0DE1C): X_PATCH_REG moved 12 -> 5. The
// smoke's x conduits go through reg 5 now; reg-12 LIs are plain LIs.
static const int DS_XREG = 5;
static Program ds_fill_read_prog(int bank, uint32_t row, int l1, int l2,
                                 uint32_t placeholder = 0u) {
  Program p;
  p.add_inst(SMC_LI(bank, BAR));
  p.add_inst(SMC_LI(8, CASR));
  p.add_below(wrRow_immediate_label_regs(BAR, row, placeholder /*-> patched by x_word*/,
                                         l1, RAR, DS_XREG));
  p.add_below(rdRow_immediate_label(BAR, row, l2));
  p.add_inst(SMC_END());
  return p;
}
// Write a full row with a uniform 32-bit pattern (rs_write_row), then READ-mode
// raw-capture readback and verify byte-exact. Returns true iff the row is legal
// on this DIMM (readback == pattern). Non-poisoning (receiveDataTry).
// Padded READ resident: rs_accum_read + explicit row close + ~2000 idle cycles
// before END. Probes whether the B-drift is inter-iteration spacing at walk
// cadence (S_KICK comes SETTLE_CYCLES=48 after fin; padding stretches the gap
// host-side, no RTL change).
static Program ds_accum_read_padded(int bank, uint32_t row, int label) {
  Program p;
  p.add_inst(SMC_LI(bank, BAR)); p.add_inst(SMC_LI(8, CASR)); p.add_inst(SMC_LI(128, NUM_COLS_REG));
  p.add_inst(SMC_LI(0, PATTERN_REG));
  for (int i = 0; i < 16; i++) p.add_inst(SMC_LDWD(PATTERN_REG, i));
  p.add_below(rdRow_immediate_label(BAR, row, label));
  p.add_inst(SMC_SLEEP(8));
  p.add_inst(SMC_PRE(BAR, 0, 0), SMC_NOP(), SMC_NOP(), SMC_NOP());   // close the row ourselves
  for (int i = 0; i < 20; i++) p.add_inst(SMC_SLEEP(100));           // ~2000 idle cycles
  p.add_inst(SMC_END());
  return p;
}

// Head-padded FILL resident: ~24 idle instructions BEFORE the (patched)
// pattern LI. Probes the early-fetch hypothesis: the fill resident's pattern
// LI is instruction #3 — fetched within cycles of the walk kick — while the
// row LI (~#20, always patches right) is fetched later. If padding fixes the
// WDATA-ZERO, patch-target LIs must simply sit past the first ~N instructions
// (program-layer placement rule, no RTL needed).
static Program ds_fill_read_prog_padded(int bank, uint32_t row, int l1, int l2) {
  Program p;
  p.add_inst(SMC_LI(bank, BAR)); p.add_inst(SMC_LI(8, CASR));
  for (int i = 0; i < 24; i++) p.add_inst(SMC_SLEEP(4));
  p.add_below(wrRow_immediate_label_regs(BAR, row, 0, l1, RAR, DS_XREG));
  p.add_below(rdRow_immediate_label(BAR, row, l2));
  p.add_inst(SMC_END());
  return p;
}

// C4 discriminator: fill whose PATTERN goes through reg 15 — a register the
// walk patch NEVER touches (patch matches RT==12 only) — with a literal
// immediate. Row also literal. If the row receives the literal in a walk,
// the regfile->LDWD->wdata->DDR-write chain is healthy in walk context and
// the fault is isolated to the reg-12 patch value (pf_val/rep_x_word).
static Program ds_fill_read_r15(int bank, uint32_t row, uint32_t pat, int l1, int l2) {
  const int ALT_PAT_REG = 15;
  Program p;
  p.add_inst(SMC_LI(bank, BAR)); p.add_inst(SMC_LI(8, CASR));
  p.add_inst(SMC_LI(pat, ALT_PAT_REG));
  for (int i = 0; i < 16; i++) p.add_inst(SMC_LDWD(ALT_PAT_REG, i));
  p.add_inst(SMC_LI(row, RAR));
  p.add_inst(SMC_LI(0, CAR));
  p.add_inst(SMC_LI(0, LOOP_COLS));
  p.add_inst(SMC_LI(128, NUM_COLS_REG));
  p.add_below(PRE(BAR, 0, 0));
  p.add_below(ACT(BAR, 0, RAR, 0));
  p.add_label("DS_R15_WR_" + std::to_string(l1));
  p.add_below(WRITE(BAR, CAR, 1));
  p.add_inst(SMC_ADDI(LOOP_COLS, 1, LOOP_COLS));
  p.add_branch(p.BR_TYPE::BL, LOOP_COLS, NUM_COLS_REG, "DS_R15_WR_" + std::to_string(l1));
  p.add_inst(all_nops());
  p.add_below(rdRow_immediate_label(BAR, row, l2));
  p.add_inst(SMC_END());
  return p;
}

// C5 discriminator: READ resident whose ROW ADDRESS comes from reg 12 — the
// x patch target. The descriptor's x_word becomes the row number. Row-
// addressing observability is proven (sub-gate B), so the drain tells us
// whether x was delivered: reads row <x> (its known popcount) vs row 0.
static Program ds_read_row_via_r12(int bank, int label) {
  Program p;
  p.add_inst(SMC_LI(bank, BAR)); p.add_inst(SMC_LI(8, CASR));
  p.add_inst(SMC_LI(0, 5));                      // placeholder — PATCHED with x (the row); b32b X reg
  p.add_inst(SMC_LI(128, NUM_COLS_REG));
  p.add_inst(SMC_LI(0, CAR));
  p.add_inst(SMC_LI(0, LOOP_COLS));
  p.add_below(PRE(BAR, 0, 0));
  p.add_below(ACT(BAR, 0, 5, 0));                // row = reg 5 (the patched x)
  p.add_inst(SMC_SLEEP(4));
  p.add_label("DS_R12ROW_" + std::to_string(label));
  p.add_below(READ(BAR, CAR, 1));
  p.add_inst(SMC_SLEEP(4));
  p.add_inst(SMC_ADDI(LOOP_COLS, 1, LOOP_COLS));
  p.add_branch(p.BR_TYPE::BL, LOOP_COLS, NUM_COLS_REG, "DS_R12ROW_" + std::to_string(label));
  p.add_inst(all_nops());
  p.add_inst(SMC_END());
  return p;
}

static bool ds_write_verify(SoftMCPlatform& pf, int bank, uint32_t row,
                            uint32_t p, int label, long recv_ms) {
  std::vector<uint32_t> seg(2048, p);
  rs_write_row(pf, bank, row, seg.data());
  std::vector<uint8_t> rb(8192 + 32, 0);
  Program rp = rs_read_prog(bank, row, label);
  pf.replay_send_resident(rp);
  int got = pf.receiveDataTry(rb.data(), 8192 + 32, recv_ms);
  if (got < 8192) return false;
  const uint32_t* g = (const uint32_t*)rb.data();
  for (int i = 0; i < 2048; i++) if (g[i] != p) return false;
  return true;
}

static int run_desc_smoke(int argc, char** argv) {
  // argv: [0]=prog [1]="desc-smoke" [2]=bender [3]=calib [4]=bank
  // DEVIATIONS from the literal task spec (all forced by silicon mechanism):
  //  * READ resident = rs_accum_read (LDWD-zeroed ddr_wdata) NOT rs_read_prog:
  //    rs_write_row leaves ddr_wdata holding the last-written pattern, so a
  //    resident without an LDWD-zero would compare rd_data against garbage.
  //    x_word=0 in every READ-resident descriptor keeps the reg-12 patch a
  //    harmless PATTERN_REG:=0.
  //  * Sub-gate C drain: the FILL resident LDWDs ddr_wdata:=x then reads the
  //    just-filled row (=x), so popcount(x^x)=0 per lane -> the accumulator
  //    drain is ~0 (fill+read cancel), NOT the task's Σ2048*popcount(x). The
  //    DEFINITIVE fill-from-descriptor proof is the row readback (== last x);
  //    the drain is reported as a diagnostic and does not gate.
  //  * Receives use receiveDataTry (bounded, NON-poisoning) not receiveData:
  //    one sub-gate's framing surprise must not poison the rest of a
  //    first-silicon smoke. Walk buffers are oversized and records classified
  //    by magic scan (no fixed offsets).
  //  * Sub-gate D captures BOTH walks in ONE raw-capture session (single
  //    replay_send_resident) so walk2 truly re-arms WITHOUT resending the
  //    resident (the host raw-capture drain is armed only by
  //    replay_send_resident; a second send would defeat the "no resend" claim).
  if (argc < 4) {
    fprintf(stderr, "Usage: %s desc-smoke <bender> <calib> [bank=0]\n", argv[0]);
    return 1;
  }
  setvbuf(stdout, NULL, _IONBF, 0);
  int bender = atoi(argv[2]);
  std::string calib_p = argv[3];
  int bank = (argc > 4) ? atoi(argv[4]) : 0;
  (void)calib_p;   // CLI parity (dimm2 trio); the raw-row smoke uses no tuples
  const long recv_ms = 4000;      // per-receive bounded window (walks)
  const long read_ms = 5000;      // per plain-read window (returns at 8224 B)
  fprintf(stderr, "[desc-smoke] bender=%d bank=%d calib=%s\n",
          bender, bank, calib_p.c_str());

  SoftMCPlatform pf(bender);
  if (pf.init() != SOFTMC_SUCCESS) { fprintf(stderr, "[desc-smoke] init failed\n"); return 1; }
  pf.reset_fpga();
  { Program sync; sync.add_inst(SMC_END()); pf.execute(sync); }   // drain reset ack
  pf.set_aref(false);
  pf.set_readback_mode(false); pf.set_readback_mode(false);        // READ mode

  // ---- test rows (far from production pools 45312-45952) ----
  const uint32_t ZEROROW = 60008;   // READ-resident placeholder (auto-run reads 0)
  const uint32_t FILLROW = 60016;   // FILL-resident scratch (fills overwrite it)
  struct Row { uint32_t row, pat; bool legal; };
  Row R60000{60000, 0xF0F0F0F0u, false};
  Row R60001{60001, 0xAAAAAAAAu, false};
  Row R60002{60002, 0x0000FFFFu, false};
  Row R60003{60003, 0x12345678u, false};
  Row R70000{70000, 0x000000FFu, false};       // 17-bit: row bit16 set
  Row R131070{131070, 0x80000001u, false};      // 17-bit: bit16 set + high nibble
  int lab = 7000;
  auto wv = [&](Row& r) {
    r.legal = ds_write_verify(pf, bank, r.row, r.pat, lab++, read_ms);
    fprintf(stderr, "[desc-smoke] write-verify row=%u pat=0x%08x -> %s\n",
            r.row, r.pat, r.legal ? "legal" : "FAILED (skip its sub-cases)");
  };
  wv(R60000); wv(R60001); wv(R60002); wv(R60003); wv(R70000); wv(R131070);
  bool zero_ok = ds_write_verify(pf, bank, ZEROROW, 0u, lab++, read_ms);
  bool fill_ok = ds_write_verify(pf, bank, FILLROW, 0u, lab++, read_ms);
  fprintf(stderr, "[desc-smoke] ZEROROW=%u->%s FILLROW=%u->%s\n",
          ZEROROW, zero_ok ? "ok" : "FAILED", FILLROW, fill_ok ? "ok" : "FAILED");
  if (!zero_ok)
    fprintf(stderr, "[desc-smoke] WARNING: ZEROROW illegal — READ-resident "
            "auto-run may deposit nonzero (cleared post-send anyway).\n");

  // ---- image-magic GATE: plain READ raw-capture probe BEFORE any DLOAD ----
  uint32_t img_magic = 0;
  {
    std::vector<uint8_t> pbuf(8192 + 32, 0);
    Program probe = rs_read_prog(bank, ZEROROW, 2000);
    pf.replay_send_resident(probe);
    int pgot = pf.receiveDataTry(pbuf.data(), 8192 + 32, read_ms + 2000);
    if (pgot < 8192 + 4) {
      fprintf(stderr, "[desc-smoke] image probe short/stalled (%d/%d) — card up? "
              "refusing.\n", pgot, 8192 + 32);
      return 3;
    }
    memcpy(&img_magic, pbuf.data() + 8192, 4);
    fprintf(stderr, "[desc-smoke] image probe: recv=%d image_trailer=0x%08x\n", pgot, img_magic);
    if ((img_magic & 0xFFFFFF00u) != 0xDBC0DE00u) {
      fprintf(stderr, "[desc-smoke] probe trailer 0x%08x is not a 0xDBC0DExx magic "
              "— image/framing unexpected, refusing.\n", img_magic);
      return 2;
    }
    if (img_magic < 0xDBC0DE18u) {
      fprintf(stderr, "[desc-smoke] image magic 0x%08x (low 0x%02x): descriptor walk "
              "(build-31, 0xDBC0DE1A / gate >=0xDBC0DE18) not available, refusing "
              "(no DLOAD word sent).\n", img_magic, img_magic & 0xFF);
      return 2;
    }
    fprintf(stderr, "[desc-smoke] image is build-31+ (magic 0x%08x) — DLOAD/REPLAY_N "
            "descriptor walk available.\n", img_magic);
  }

  int g_label = 3000;
  bool any_stall = false, any_fail = false;
  auto note_stall = [&](const char* where, int recv) {
    fprintf(stderr, "[desc-smoke] STALL in %s (recv=%d) — capturing what arrived; "
            "marking FAIL.\n", where, recv);
    any_stall = true;
  };
  auto lane_stats = [](const RsParse& r, long long& mn, long long& mx, bool& eq) {
    mn = 0; mx = 0; eq = false;
    if (r.lanes.empty()) return;
    mn = mx = r.lanes[0];
    for (int32_t v : r.lanes) { if (v < mn) mn = v; if (v > mx) mx = v; }
    eq = (mn == mx);
  };

  // One descriptor walk, raw-captured, single drain parsed. fill=false -> READ
  // resident (rs_accum_read, placeholder ZEROROW, ddr_wdata:=0); fill=true ->
  // FILL resident (ds_fill_read_prog, placeholder FILLROW). Non-poisoning.
  auto do_walk = [&](bool fill, const std::vector<DsDesc>& ds,
                     RsParse& out, int& recv_bytes, long& osk_delta,
                     bool pad = false) -> bool {
    int N = (int)ds.size();
    pf.set_readback_mode_accxbp();
    pf.set_readback_mode_accxbp();            // enter ACCUM_XBP + clear accumulator
    long osk0 = pf.oversize_skips();
    // pre-compute distinct labels (avoid unsequenced g_label++ in one call)
    int lread = g_label++, lwr = g_label++, lrd = g_label++;
    Program res = fill ? (pad ? ds_fill_read_prog_padded(bank, FILLROW, lwr, lrd)
                              : ds_fill_read_prog(bank, FILLROW, lwr, lrd))
                       : (pad ? ds_accum_read_padded(bank, ZEROROW, lread)
                              : rs_accum_read(bank, ZEROROW, lread));
    // ORDER (b32b finding): DLOAD while the frontend is IDLE — BEFORE the
    // resident send — exactly like the sim TB (send_desc then send_program).
    // DLOADs issued after the send (EXECUTE-adjacent state) correlate with
    // dead walks (C7: walk-2's idle-time dloads alive, walk-1's post-send
    // dloads dead). PIM_DESC_AFTER=1 restores the old order for A/B.
    bool dload_after = getenv("PIM_DESC_AFTER") != nullptr;
    auto send_descs = [&]() {
      for (int k = 0; k < N; k++) {
        pf.dload((uint32_t)k, ds[k].x, ds[k].row, ds[k].acc);
        if (getenv("PIM_DESC_DOUBLE")) pf.dload((uint32_t)k, ds[k].x, ds[k].row, ds[k].acc);
        if (getenv("PIM_DESC_SPACE")) usleep(2000);
      }
    };
    if (!dload_after) send_descs();
    pf.replay_send_resident(res);             // auto-runs once on placeholder; raw capture armed
    usleep(5000);                             // let the auto-run ack land (<< 500 ms quiet window)
    pf.set_readback_mode_accxbp();            // clear the auto-run deposit
    if (dload_after) send_descs();
    pf.replay_n((uint16_t)N);
    int want = 8192 + (N + 24) * 32;          // generous upper bound; classify by magic
    std::vector<uint8_t> buf(want, 0);
    recv_bytes = pf.receiveDataTry(buf.data(), want, recv_ms);
    osk_delta = pf.oversize_skips() - osk0;
    out = rs_parse(buf.data(), recv_bytes);
    if (getenv("PIM_DESC_DUMP")) {
      fprintf(stderr, "[desc-smoke] record map (fill=%d N=%d recv=%d):\n", (int)fill, N, recv_bytes);
      long last_end = 0;
      for (int off = 0; off + 32 <= recv_bytes; off += 4) {
        uint32_t w[8]; memcpy(w, buf.data() + off, 32);
        if ((w[0] & 0xFFFFFF00u) == 0xDBC0DE00u) {
          fprintf(stderr, "  off=%6d gap=%6ld w:", off, (long)off - last_end);
          for (int k = 0; k < 8; k++) fprintf(stderr, " %08x", w[k]);
          fprintf(stderr, "\n");
          last_end = off + 32;
        }
      }
    }
    pf.drain_stray(300, 4);
    return recv_bytes >= 8192 && out.n_drain >= 1;
  };

  // Unified READ-resident case (covers G-A same-row-varied-acc AND G-B
  // varied-row): items = list of (row, acc); x_word forced to 0 so ddr_wdata=0
  // and each lane accumulates ds_wsum(acc)*popcount(row_pattern).
  auto read_walk = [&](const char* nm, const std::vector<std::pair<Row*,uint32_t>>& items,
                       bool pad = false) {
    for (auto& it : items)
      if (!it.first->legal) {
        printf("[desc-smoke] %s: SKIP (row %u illegal)\n", nm, it.first->row);
        any_fail = true; return;
      }
    std::vector<DsDesc> ds; long long exp = 0, exp_lane = 0;
    for (auto& it : items) {
      long w = ds_wsum(it.second);
      ds.push_back({0u, it.first->row, it.second});
      exp      += (long long)w * (2048LL * __builtin_popcount(it.first->pat));
      exp_lane += (long long)w * __builtin_popcount(it.first->pat);
    }
    // FRESHNESS (retention finding, runs 1-5): these rows are outside the
    // maintained subarrays and aref is off — weak cells decay in seconds.
    // Rewrite every distinct row immediately before the walk so the
    // write-to-walk window is ~1 s and the walk reads fresh cells.
    {
      pf.set_readback_mode(false); pf.set_readback_mode(false); pf.drain_stray(800, 6);
      std::vector<uint32_t> done;
      for (auto& it : items) {
        if (std::find(done.begin(), done.end(), it.first->row) != done.end()) continue;
        done.push_back(it.first->row);
        if (!ds_write_verify(pf, bank, it.first->row, it.first->pat, g_label++, read_ms)) {
          printf("[desc-smoke] %s: refresh-rewrite of row %u FAILED\n", nm, it.first->row);
          any_fail = true; return;
        }
      }
    }
    RsParse r; int rb = 0; long osk = 0;
    bool ok = do_walk(false, ds, r, rb, osk, pad);
    if (!ok) { note_stall(nm, rb); printf("[desc-smoke] %s: STALL/no-drain -> FAIL\n", nm);
               any_fail = true; return; }
    long long mn, mx; bool eq; lane_stats(r, mn, mx, eq);
    bool num = (r.drain_sum == exp) && eq && (mn == exp_lane);
    bool ctl = (r.n_drain == 1) && ((int)r.lanes.size() == 2048) && (osk == 0)
               && (r.drain_magic == img_magic);
    printf("[desc-smoke] %s: N=%zu drain=%lld exp=%lld leak=%+lld "
           "lane[min=%lld max=%lld eq=%d exp=%lld] drain_magic=0x%08x n_drain=%ld "
           "osk=%ld recv=%d -> %s\n",
           nm, items.size(), r.drain_sum, exp, r.drain_sum - exp, mn, mx, (int)eq,
           exp_lane, r.drain_magic, r.n_drain, osk, rb, (num && ctl) ? "PASS" : "FAIL");
    if (!num && (int)r.lanes.size() == 2048) {
      int shown = 0, nbad = 0;
      for (int i = 0; i < 2048; i++) if ((long long)r.lanes[i] != exp_lane) {
        nbad++;
        if (shown < 32) { fprintf(stderr, "[desc-smoke]   %s lane[%d]=%d (d=%+lld)\n",
                                  nm, i, r.lanes[i], (long long)r.lanes[i] - exp_lane); shown++; }
      }
      fprintf(stderr, "[desc-smoke]   %s bad_lanes=%d/2048\n", nm, nbad);
    }
    if (!(num && ctl)) any_fail = true;
  };

  // ---- Sub-gate A: acc place-value (READ resident, row 60000, x=0) ----
  printf("[desc-smoke] ==== Sub-gate A: acc place-value (READ resident, row %u) ====\n", R60000.row);
  read_walk("A[+1,+2]",       {{&R60000, ds_acc(0,0)}, {&R60000, ds_acc(0,1)}});
  read_walk("A[+1,+2,+4,+8]", {{&R60000, ds_acc(0,0)}, {&R60000, ds_acc(0,1)},
                               {&R60000, ds_acc(0,2)}, {&R60000, ds_acc(0,3)}});
  read_walk("A[+1,-1]",       {{&R60000, ds_acc(0,0)}, {&R60000, ds_acc(1,0)}});
  read_walk("A[+2,-8]",       {{&R60000, ds_acc(0,1)}, {&R60000, ds_acc(1,3)}});

  // ---- Sub-gate B: row advance (READ resident, acc=+1 unless noted, x=0) ----
  printf("[desc-smoke] ==== Sub-gate B: row advance (READ resident) ====\n");
  read_walk("B[60000,60001,60002,60003]",
            {{&R60000, ds_acc(0,0)}, {&R60001, ds_acc(0,0)},
             {&R60002, ds_acc(0,0)}, {&R60003, ds_acc(0,0)}});
  read_walk("B[splice 60000,70000,131070]",
            {{&R60000, ds_acc(0,0)}, {&R70000, ds_acc(0,0)}, {&R131070, ds_acc(0,0)}});
  read_walk("B[mixed 60000@+1,60001@+2]",
            {{&R60000, ds_acc(0,0)}, {&R60001, ds_acc(0,1)}});
  // ---- B-probes: transition mechanism ----
  printf("[desc-smoke] ==== B-probes: transition spacing / duplication ====\n");
  // dup: same rows, each visited twice consecutively — halves the number of
  // row CHANGES; if errors track transitions, error pattern shrinks/moves.
  read_walk("Bdup[60000,60000,60001,60001]",
            {{&R60000, ds_acc(0,0)}, {&R60000, ds_acc(0,0)},
             {&R60001, ds_acc(0,0)}, {&R60001, ds_acc(0,0)}});
  // padded resident: ~2000 extra idle cycles + explicit PRE per iteration.
  read_walk("Bpad[60000,60001,60002,60003]",
            {{&R60000, ds_acc(0,0)}, {&R60001, ds_acc(0,0)},
             {&R60002, ds_acc(0,0)}, {&R60003, ds_acc(0,0)}}, true);
  read_walk("Bpad[mixed 60000@+1,60001@+2]",
            {{&R60000, ds_acc(0,0)}, {&R60001, ds_acc(0,1)}}, true);
  read_walk("Bpad[splice 60000,70000,131070]",
            {{&R60000, ds_acc(0,0)}, {&R70000, ds_acc(0,0)}, {&R131070, ds_acc(0,0)}}, true);

  // ---- Sub-gate C2 (diagnostic discriminator, FIRST fill walk of the run):
  // prior row content 0xDEADBEEF (pc=24), ONE iteration x=0xFFFFFFFF (+1).
  //   drain=65536 & row==FFFFFFFF -> patched fill WORKS
  //   drain=49152 & row==DEADBEEF -> WRITEs never issued
  //   drain=0     & row==0        -> WRITEs issued with wdata=0 (patch->LDWD broken)
  printf("[desc-smoke] ==== Sub-gate C2: fill discriminator (prior=DEADBEEF, x=FFFFFFFF, N=1) ====\n");
  if (fill_ok) {
    pf.set_readback_mode(false); pf.set_readback_mode(false); pf.drain_stray(800, 6);
    bool pw = ds_write_verify(pf, bank, FILLROW, 0xDEADBEEFu, g_label++, read_ms);
    std::vector<DsDesc> ds1 = {{0xFFFFFFFFu, FILLROW, ds_acc(0,0)}};
    RsParse r; int rb = 0; long osk = 0;
    bool ok = do_walk(true, ds1, r, rb, osk);
    pf.set_readback_mode(false); pf.set_readback_mode(false); pf.drain_stray(1500, 8);
    std::vector<uint8_t> rbk(8192 + 32, 0);
    Program rp2 = rs_read_prog(bank, FILLROW, g_label++);
    pf.replay_send_resident(rp2);
    int got = pf.receiveDataTry(rbk.data(), 8192 + 32, read_ms);
    long n_ff = 0, n_db = 0, n_z = 0, n_o = 0;
    if (got >= 8192) { const uint32_t* g = (const uint32_t*)rbk.data();
      for (int i = 0; i < 2048; i++) {
        if (g[i] == 0xFFFFFFFFu) n_ff++;
        else if (g[i] == 0xDEADBEEFu) n_db++;
        else if (g[i] == 0u) n_z++; else n_o++;
      } }
    const char* verdict = (n_ff == 2048) ? "FILL-WORKS" :
                          (n_db == 2048) ? "WRITES-NOT-ISSUED" :
                          (n_z == 2048) ? "WDATA-ZERO" : "MIXED/OTHER";
    printf("[desc-smoke] C2: prewrite=%d walk_ok=%d drain=%lld (works:65536 nowrite:49152 "
           "wzero:0) row census FF=%ld DB=%ld Z=%ld other=%ld -> %s\n",
           (int)pw, (int)ok, r.drain_sum, n_ff, n_db, n_z, n_o, verdict);
  }

  // ---- Sub-gate C3: same discriminator, HEAD-PADDED fill resident (pattern
  // LI pushed past the first ~26 instructions — early-fetch hypothesis) ----
  printf("[desc-smoke] ==== Sub-gate C3: fill discriminator, head-padded resident ====\n");
  if (fill_ok) {
    pf.set_readback_mode(false); pf.set_readback_mode(false); pf.drain_stray(800, 6);
    bool pw = ds_write_verify(pf, bank, FILLROW, 0xDEADBEEFu, g_label++, read_ms);
    std::vector<DsDesc> ds1 = {{0xFFFFFFFFu, FILLROW, ds_acc(0,0)}};
    RsParse r; int rb = 0; long osk = 0;
    bool ok = do_walk(true, ds1, r, rb, osk, /*pad=*/true);
    pf.set_readback_mode(false); pf.set_readback_mode(false); pf.drain_stray(1500, 8);
    std::vector<uint8_t> rbk(8192 + 32, 0);
    Program rp3 = rs_read_prog(bank, FILLROW, g_label++);
    pf.replay_send_resident(rp3);
    int got = pf.receiveDataTry(rbk.data(), 8192 + 32, read_ms);
    long n_ff = 0, n_db = 0, n_z = 0, n_o = 0;
    if (got >= 8192) { const uint32_t* g = (const uint32_t*)rbk.data();
      for (int i = 0; i < 2048; i++) {
        if (g[i] == 0xFFFFFFFFu) n_ff++;
        else if (g[i] == 0xDEADBEEFu) n_db++;
        else if (g[i] == 0u) n_z++; else n_o++;
      } }
    const char* verdict = (n_ff == 2048) ? "FILL-WORKS" :
                          (n_db == 2048) ? "WRITES-NOT-ISSUED" :
                          (n_z == 2048) ? "WDATA-ZERO" : "MIXED/OTHER";
    printf("[desc-smoke] C3(padded): prewrite=%d walk_ok=%d drain=%lld row census "
           "FF=%ld DB=%ld Z=%ld other=%ld -> %s\n",
           (int)pw, (int)ok, r.drain_sum, n_ff, n_db, n_z, n_o, verdict);
  }

  // ---- Sub-gate C4: reg-15 literal fill in walk context (patch untouched) ----
  printf("[desc-smoke] ==== Sub-gate C4: reg-15 literal fill (0xCAFEBABE) in walk context ====\n");
  if (fill_ok) {
    pf.set_readback_mode(false); pf.set_readback_mode(false); pf.drain_stray(800, 6);
    bool pw = ds_write_verify(pf, bank, FILLROW, 0xDEADBEEFu, g_label++, read_ms);
    pf.set_readback_mode_accxbp(); pf.set_readback_mode_accxbp();
    int l1 = g_label++, l2 = g_label++;
    Program r15p = ds_fill_read_r15(bank, FILLROW, 0xCAFEBABEu, l1, l2);
    pf.replay_send_resident(r15p);           // auto-run fills CAFEBABE (normal ctx)
    usleep(5000);
    pf.set_readback_mode_accxbp();           // clear auto-run deposit
    pf.dload(0, 0x11111111u, FILLROW, ds_acc(0,0));
    pf.replay_n(1);
    int want = 8192 + 16 * 32;
    std::vector<uint8_t> wbuf(want, 0);
    int rb = pf.receiveDataTry(wbuf.data(), want, recv_ms);
    RsParse r = rs_parse(wbuf.data(), rb);
    pf.drain_stray(300, 4);
    pf.set_readback_mode(false); pf.set_readback_mode(false); pf.drain_stray(1500, 8);
    std::vector<uint8_t> rbk(8192 + 32, 0);
    Program rp4 = rs_read_prog(bank, FILLROW, g_label++);
    pf.replay_send_resident(rp4);
    int got = pf.receiveDataTry(rbk.data(), 8192 + 32, read_ms);
    long n_cb = 0, n_z = 0, n_x1 = 0, n_o = 0;
    if (got >= 8192) { const uint32_t* g = (const uint32_t*)rbk.data();
      for (int i = 0; i < 2048; i++) {
        if (g[i] == 0xCAFEBABEu) n_cb++;
        else if (g[i] == 0u) n_z++;
        else if (g[i] == 0x11111111u) n_x1++; else n_o++;
      } }
    const char* verdict = (n_cb == 2048) ? "WALK-WDATA-HEALTHY (fault = patch value)" :
                          (n_z == 2048) ? "WALK-WRITE-PATH-BROKEN (any reg)" :
                          (n_x1 == 2048) ? "R15-HIJACKED-BY-PATCH?!" : "MIXED/OTHER";
    printf("[desc-smoke] C4: prewrite=%d drain=%lld (healthy-raw:45056 broken:0) "
           "row census CAFE=%ld Z=%ld x11=%ld other=%ld -> %s\n",
           (int)pw, r.drain_sum, n_cb, n_z, n_x1, n_o, verdict);
  }

  // ---- Sub-gate C5: x observed via row-addressing (reg-12 as row register) ----
  printf("[desc-smoke] ==== Sub-gate C5: x delivered? (row address via reg 12) ====\n");
  if (R60003.legal) {
    pf.set_readback_mode(false); pf.set_readback_mode(false); pf.drain_stray(800, 6);
    bool p0 = ds_write_verify(pf, bank, 0u, 0xFFFFFFFFu, g_label++, read_ms);       // row 0 sentinel
    bool p3 = ds_write_verify(pf, bank, R60003.row, R60003.pat, g_label++, read_ms); // fresh 0x12345678
    pf.set_readback_mode_accxbp(); pf.set_readback_mode_accxbp();
    Program r12row = ds_read_row_via_r12(bank, g_label++);
    pf.replay_send_resident(r12row);      // auto-run reads row 0 (placeholder) — cleared next
    usleep(5000);
    pf.set_readback_mode_accxbp();
    pf.dload(0, /*x = the ROW*/ R60003.row, /*row_base unused*/ 0u, ds_acc(0,0));
    pf.replay_n(1);
    int want = 8192 + 16 * 32;
    std::vector<uint8_t> wb(want, 0);
    int rb = pf.receiveDataTry(wb.data(), want, recv_ms);
    RsParse r = rs_parse(wb.data(), rb);
    pf.drain_stray(300, 4);
    long long exp_x  = 2048LL * __builtin_popcount(R60003.pat);   // 26624: x delivered
    long long exp_z  = 2048LL * 32;                               // 65536: r12=0 -> read row 0
    const char* verdict = (r.drain_sum == exp_x) ? "X-DELIVERED (fault downstream of patch)" :
                          (r.drain_sum == exp_z) ? "X-IS-ZERO at patch (desc x-slice dead)" :
                          "OTHER (see drain)";
    printf("[desc-smoke] C5: prewrites=%d/%d drain=%lld (x-delivered:%lld zero:%lld) -> %s\n",
           (int)p0, (int)p3, r.drain_sum, exp_x, exp_z, verdict);
  }

  // ---- Sub-gate C7: corruption topology (2 entries + retry, r12-row resident) ----
  // Walk1 iter k reads row <entry k's x>. Sum decodes which entries delivered:
  //   both alive: 2048*(16+13)=59392 | only e0 dead: 2048*(32+13)=92160
  //   only e1 dead: 2048*(16+32)=98304 | both dead: 2048*64=131072
  // Walk2 = re-dload + re-walk in the SAME session: recovers? (glitch-kill vs
  // persistent path fault.)
  printf("[desc-smoke] ==== Sub-gate C7: x-corruption topology (entries + retry) ====\n");
  if (R60000.legal && R60003.legal) {
    pf.set_readback_mode(false); pf.set_readback_mode(false); pf.drain_stray(800, 6);
    ds_write_verify(pf, bank, 0u, 0xFFFFFFFFu, g_label++, read_ms);
    ds_write_verify(pf, bank, R60000.row, R60000.pat, g_label++, read_ms);
    ds_write_verify(pf, bank, R60003.row, R60003.pat, g_label++, read_ms);
    pf.set_readback_mode_accxbp(); pf.set_readback_mode_accxbp();
    Program r12row = ds_read_row_via_r12(bank, g_label++);
    pf.replay_send_resident(r12row);
    usleep(5000);
    pf.set_readback_mode_accxbp();
    pf.dload(0, R60000.row, 0u, ds_acc(0,0));
    pf.dload(1, R60003.row, 0u, ds_acc(0,0));
    pf.replay_n(2);
    usleep(20000);
    pf.set_readback_mode_accxbp();                 // clear between walks
    pf.dload(0, R60000.row, 0u, ds_acc(0,0));      // re-arm identically
    pf.dload(1, R60003.row, 0u, ds_acc(0,0));
    pf.replay_n(2);
    int want = 2 * 8192 + 40 * 32;
    std::vector<uint8_t> buf7(want, 0);
    int rb = pf.receiveDataTry(buf7.data(), want, recv_ms + 2000);
    pf.drain_stray(300, 4);
    std::vector<long> mags;
    for (int off = 0; off + 4 <= rb; off += 4) {
      uint32_t w; memcpy(&w, buf7.data() + off, 4);
      if ((w & 0xFFFFFF00u) == 0xDBC0DE00u) mags.push_back(off);
    }
    std::vector<long long> drains;
    for (size_t k = 0; k < mags.size(); k++) {
      long M = mags[k]; bool is_drain = (M >= 8192);
      if (is_drain) for (size_t j = 0; j < mags.size(); j++)
                      if (mags[j] > M - 8192 && mags[j] < M) is_drain = false;
      if (!is_drain) continue;
      long long s = 0;
      for (long off = M - 8192; off < M; off += 4) { int32_t v; memcpy(&v, buf7.data() + off, 4); s += v; }
      drains.push_back(s);
    }
    auto dec = [](long long s) -> const char* {
      if (s == 59392)  return "both-alive";
      if (s == 92160)  return "e0-dead";
      if (s == 98304)  return "e1-dead";
      if (s == 131072) return "both-dead";
      return "other";
    };
    printf("[desc-smoke] C7: n_drains=%zu walk1=%lld (%s) walk2=%lld (%s) recv=%d\n",
           drains.size(), drains.size() > 0 ? drains[0] : -1,
           drains.size() > 0 ? dec(drains[0]) : "none",
           drains.size() > 1 ? drains[1] : -1,
           drains.size() > 1 ? dec(drains[1]) : "none", rb);
  }

  // ---- Sub-gate C8: is the x-kill scoped to the first DLOAD or the first WALK
  // after a resident send? [send][clear][sacrificial dload(63)][real dloads]
  // [walk]: alive -> one sacrificial DLOAD suffices; dead -> dummy WALK needed.
  printf("[desc-smoke] ==== Sub-gate C8: sacrificial-dload workaround probe ====\n");
  if (R60000.legal && R60003.legal) {
    pf.set_readback_mode(false); pf.set_readback_mode(false); pf.drain_stray(800, 6);
    ds_write_verify(pf, bank, 0u, 0xFFFFFFFFu, g_label++, read_ms);
    ds_write_verify(pf, bank, R60000.row, R60000.pat, g_label++, read_ms);
    ds_write_verify(pf, bank, R60003.row, R60003.pat, g_label++, read_ms);
    pf.set_readback_mode_accxbp(); pf.set_readback_mode_accxbp();
    Program r12row = ds_read_row_via_r12(bank, g_label++);
    pf.replay_send_resident(r12row);
    usleep(5000);
    pf.set_readback_mode_accxbp();
    pf.dload(63, R60000.row, 0u, ds_acc(0,0));     // sacrificial (never walked)
    pf.dload(0, R60000.row, 0u, ds_acc(0,0));
    pf.dload(1, R60003.row, 0u, ds_acc(0,0));
    pf.replay_n(2);
    int want = 8192 + 24 * 32;
    std::vector<uint8_t> wb(want, 0);
    int rb = pf.receiveDataTry(wb.data(), want, recv_ms);
    RsParse r = rs_parse(wb.data(), rb);
    pf.drain_stray(300, 4);
    const char* verdict = (r.drain_sum == 59392)  ? "ALIVE — sacrificial DLOAD suffices" :
                          (r.drain_sum == 131072) ? "DEAD — dummy WALK needed" :
                          (r.drain_sum == 92160)  ? "e0-dead (sacrifice absorbed only itself?)" :
                          "OTHER";
    printf("[desc-smoke] C8: drain=%lld (alive:59392 dead:131072) -> %s\n",
           r.drain_sum, verdict);
  }

  // ---- Sub-gate C11: placeholder test — did the patch FIRE at all? Bake a
  // NONZERO placeholder (0xBBBBBBBB) into the fill resident's x LI. Dead walk:
  //   row == 0xBBBBBBBB -> pf_x never fired (patch-match failure; "0" was
  //                        always just the old placeholder)
  //   row == 0          -> patch fired and delivered a genuine zero x
  //   row == x          -> alive walk (phase landed right)
  printf("[desc-smoke] ==== Sub-gate C11: placeholder test (pf_x fired?) ====\n");
  if (fill_ok) {
    pf.set_readback_mode(false); pf.set_readback_mode(false); pf.drain_stray(800, 6);
    bool pw = ds_write_verify(pf, bank, FILLROW, 0xDEADBEEFu, g_label++, read_ms);
    pf.set_readback_mode_accxbp(); pf.set_readback_mode_accxbp();
    int l1 = g_label++, l2 = g_label++;
    Program fp = ds_fill_read_prog(bank, FILLROW, l1, l2, 0xBBBBBBBBu);
    pf.replay_send_resident(fp);           // auto-run fills 0xBBBBBBBB (normal ctx)
    usleep(5000);
    pf.set_readback_mode_accxbp();
    pf.dload(0, 0xFFFFFFFFu, FILLROW, ds_acc(0,0));
    pf.replay_n(1);
    int want = 8192 + 16 * 32;
    std::vector<uint8_t> wb(want, 0);
    int rb = pf.receiveDataTry(wb.data(), want, recv_ms);
    RsParse r = rs_parse(wb.data(), rb);
    pf.drain_stray(300, 4);
    pf.set_readback_mode(false); pf.set_readback_mode(false); pf.drain_stray(1500, 8);
    std::vector<uint8_t> rbk(8192 + 32, 0);
    Program rp11 = rs_read_prog(bank, FILLROW, g_label++);
    pf.replay_send_resident(rp11);
    int got = pf.receiveDataTry(rbk.data(), 8192 + 32, read_ms);
    long n_bb = 0, n_ff = 0, n_z = 0, n_o = 0;
    if (got >= 8192) { const uint32_t* g = (const uint32_t*)rbk.data();
      for (int i = 0; i < 2048; i++) {
        if (g[i] == 0xBBBBBBBBu) n_bb++;
        else if (g[i] == 0xFFFFFFFFu) n_ff++;
        else if (g[i] == 0u) n_z++; else n_o++;
      } }
    const char* verdict = (n_bb == 2048) ? "PATCH-DID-NOT-FIRE (placeholder wrote through)" :
                          (n_z == 2048) ? "PATCH-FIRED-WITH-ZERO (genuine zero x)" :
                          (n_ff == 2048) ? "ALIVE (x delivered)" : "MIXED/OTHER";
    printf("[desc-smoke] C11: prewrite=%d drain=%lld row census BB=%ld FF=%ld Z=%ld "
           "other=%ld -> %s\n", (int)pw, r.drain_sum, n_bb, n_ff, n_z, n_o, verdict);
  }

  // ---- Sub-gate C12: THREE-WAY hypothesis split, per walk (5 identical
  // groups in one session). Fill resident baked: row=FILLROW, pattern=0xBB..
  // Descriptor: {x=FFFFFFFF, row=OTHERROW, acc=+2}. Per group, readback BOTH
  // rows in READ mode and decode:
  //   OTHERROW==FFFFFFFF                       -> ALIVE (row+x patched)
  //   OTHERROW==BBBBBBBB, FILLROW untouched    -> ROW fired, X did not (RT-selective)
  //   FILLROW==BBBBBBBB, OTHERROW untouched    -> PATCH ENTIRELY OFF
  // drain scale (vs popcount of what was read): x2 -> desc-acc fired (desc_mode
  // armed); x1 -> desc-acc off too (DLOADs never decoded).
  printf("[desc-smoke] ==== Sub-gate C12: row-vs-x-vs-acc split per dead walk ====\n");
  {
    const uint32_t OTHERROW = 60032;
    bool oo = ds_write_verify(pf, bank, OTHERROW, 0u, g_label++, read_ms);
    if (oo && fill_ok) {
      pf.set_readback_mode(false); pf.set_readback_mode(false); pf.drain_stray(800, 6);
      ds_write_verify(pf, bank, FILLROW, 0xDEADBEEFu, g_label++, read_ms);
      ds_write_verify(pf, bank, OTHERROW, 0x0F0F0F0Fu, g_label++, read_ms);
      pf.set_readback_mode_accxbp(); pf.set_readback_mode_accxbp();
      int l1 = g_label++, l2 = g_label++;
      Program fp = ds_fill_read_prog(bank, FILLROW, l1, l2, 0xBBBBBBBBu);
      pf.replay_send_resident(fp);
      usleep(5000);
      pf.set_readback_mode_accxbp();
      for (int grp = 0; grp < 5; grp++) {
        // re-prime both rows in READ mode, then walk once
        pf.set_readback_mode(false); pf.set_readback_mode(false); pf.drain_stray(800, 6);
        bool pr1 = ds_write_verify(pf, bank, FILLROW, 0xDEADBEEFu, g_label++, read_ms);
        bool pr2 = ds_write_verify(pf, bank, OTHERROW, 0x0F0F0F0Fu, g_label++, read_ms);
        pf.set_readback_mode_accxbp(); pf.set_readback_mode_accxbp();
        pf.dload(0, 0xFFFFFFFFu, OTHERROW, ds_acc(0,1));    // x=FF.., row=OTHER, acc=+2
        pf.replay_n(1);
        int want = 8192 + 16 * 32;
        std::vector<uint8_t> wb(want, 0);
        int rb = pf.receiveDataTry(wb.data(), want, recv_ms);
        RsParse r = rs_parse(wb.data(), rb);
        pf.drain_stray(300, 4);
        pf.set_readback_mode(false); pf.set_readback_mode(false); pf.drain_stray(1200, 8);
        // census v2: frame the probe payload by the trailing magic (the 8192 B
        // preceding the trailer), never raw offset 0 (a leading stray record
        // shifts the payload and fakes "mixed").
        auto census = [&](uint32_t row, uint32_t cand1, uint32_t cand2, uint32_t cand3,
                          long& n1, long& n2, long& n3, long& no, uint32_t& w0) -> bool {
          std::vector<uint8_t> rbk(8192 + 8 * 32, 0);
          Program rp = rs_read_prog(bank, row, g_label++);
          pf.replay_send_resident(rp);
          int got = pf.receiveDataTry(rbk.data(), (int)rbk.size(), read_ms);
          long pay = -1;
          for (int off = 8192; off + 4 <= got; off += 4) {
            uint32_t w; memcpy(&w, rbk.data() + off, 4);
            if ((w & 0xFFFFFF00u) == 0xDBC0DE00u) {
              bool interior = false;
              for (int o2 = off - 8192 + 4; o2 < off; o2 += 4) {
                uint32_t v; memcpy(&v, rbk.data() + o2, 4);
                if ((v & 0xFFFFFF00u) == 0xDBC0DE00u) { interior = true; break; }
              }
              if (!interior) { pay = off - 8192; break; }
            }
          }
          if (pay < 0) return false;
          const uint32_t* g = (const uint32_t*)(rbk.data() + pay);
          n1 = n2 = n3 = no = 0; w0 = g[0];
          for (int i = 0; i < 2048; i++) {
            if (g[i] == cand1) n1++;
            else if (g[i] == cand2) n2++;
            else if (g[i] == cand3) n3++; else no++;
          }
          return true;
        };
        long f1, f2, f3, fo, o1, o2, o3, oo2; uint32_t fw0 = 0, ow0 = 0;
        bool fOk = census(FILLROW, 0xDEADBEEFu, 0xBBBBBBBBu, 0xFFFFFFFFu, f1, f2, f3, fo, fw0);
        bool oOk = census(OTHERROW, 0x0F0F0F0Fu, 0xBBBBBBBBu, 0xFFFFFFFFu, o1, o2, o3, oo2, ow0);
        printf("[desc-smoke] C12 grp%d: prime=%d/%d FILL[db=%ld bb=%ld ff=%ld o=%ld w0=%08x ok=%d] "
               "OTHER[0f=%ld bb=%ld ff=%ld o=%ld w0=%08x ok=%d] drain=%lld\n",
               grp, (int)pr1, (int)pr2, f1, f2, f3, fo, fw0, (int)fOk,
               o1, o2, o3, oo2, ow0, (int)oOk, r.drain_sum);
        pf.set_readback_mode_accxbp(); pf.set_readback_mode_accxbp();
      }
      pf.set_readback_mode(false); pf.set_readback_mode(false); pf.drain_stray(1200, 8);
    } else printf("[desc-smoke] C12: SKIP (rows unavailable)\n");
  }

  // ---- Sub-gates C9/C10: x-death pattern across consecutive identical walks.
  // C9: send once, then 5x [dload(0,60000); dload(1,60003); walk N=2; clear].
  // C10: same but 2 dummy dloads right after the send (cumulative-threshold probe).
  auto run_seq_probe = [&](const char* nm, int n_dummy) {
    pf.set_readback_mode(false); pf.set_readback_mode(false); pf.drain_stray(800, 6);
    ds_write_verify(pf, bank, 0u, 0xFFFFFFFFu, g_label++, read_ms);
    ds_write_verify(pf, bank, R60000.row, R60000.pat, g_label++, read_ms);
    ds_write_verify(pf, bank, R60003.row, R60003.pat, g_label++, read_ms);
    pf.set_readback_mode_accxbp(); pf.set_readback_mode_accxbp();
    Program r12row = ds_read_row_via_r12(bank, g_label++);
    pf.replay_send_resident(r12row);
    usleep(5000);
    pf.set_readback_mode_accxbp();
    for (int d = 0; d < n_dummy; d++) pf.dload((uint32_t)(60 + d), R60000.row, 0u, ds_acc(0,0));
    const int NW = 5;
    for (int wk = 0; wk < NW; wk++) {
      pf.dload(0, R60000.row, 0u, ds_acc(0,0));
      pf.dload(1, R60003.row, 0u, ds_acc(0,0));
      pf.replay_n(2);
      usleep(20000);
      if (wk < NW - 1) pf.set_readback_mode_accxbp();   // clear between walks
    }
    int want = NW * 8192 + 80 * 32;
    std::vector<uint8_t> buf9(want, 0);
    int rb = pf.receiveDataTry(buf9.data(), want, recv_ms + 4000);
    pf.drain_stray(300, 4);
    std::vector<long> mags;
    for (int off = 0; off + 4 <= rb; off += 4) {
      uint32_t w; memcpy(&w, buf9.data() + off, 4);
      if ((w & 0xFFFFFF00u) == 0xDBC0DE00u) mags.push_back(off);
    }
    std::string pat;
    int nd = 0;
    for (size_t k = 0; k < mags.size(); k++) {
      long M = mags[k]; bool is_drain = (M >= 8192);
      if (is_drain) for (size_t j = 0; j < mags.size(); j++)
                      if (mags[j] > M - 8192 && mags[j] < M) is_drain = false;
      if (!is_drain) continue;
      long long s = 0;
      for (long off = M - 8192; off < M; off += 4) { int32_t v; memcpy(&v, buf9.data() + off, 4); s += v; }
      nd++;
      pat += (s == 59392) ? 'A' : (s == 131072) ? 'D' : (s == 92160) ? '0' : (s == 98304) ? '1' : '?';
    }
    printf("[desc-smoke] %s: dummies=%d walks=%d drains=%d pattern=%s "
           "(A=alive D=dead 0=e0dead 1=e1dead) recv=%d\n", nm, n_dummy, NW, nd, pat.c_str(), rb);
  };
  printf("[desc-smoke] ==== Sub-gates C9/C10: x-death sequence probes ====\n");
  if (R60000.legal && R60003.legal) { run_seq_probe("C9", 0); run_seq_probe("C10", 2); }

  // ---- Sub-gate C: fill-from-descriptor (FILL resident, all desc row=FILLROW) ----
  printf("[desc-smoke] ==== Sub-gate C: fill-from-descriptor (FILL resident, row %u) ====\n", FILLROW);
  if (!fill_ok) { printf("[desc-smoke] C: SKIP (FILLROW illegal)\n"); any_fail = true; }
  else {
    std::vector<uint32_t> xs = {0xFFFFFFFFu, 0x0000FFFFu, 0xA5A5A5A5u, 0x00000001u};
    std::vector<DsDesc> ds;
    for (uint32_t x : xs) ds.push_back({x, FILLROW, ds_acc(0,0)});
    RsParse r; int rb = 0; long osk = 0;
    bool ok = do_walk(true, ds, r, rb, osk);
    if (!ok) note_stall("C(walk)", rb);
    // Drain is ~0 (fill+read cancel: ddr_wdata=x, rd_data=x). Definitive proof
    // is the readback below.
    pf.set_readback_mode(false); pf.set_readback_mode(false); pf.drain_stray(1500, 8);
    std::vector<uint8_t> rbk(8192 + 32, 0);
    Program rp = rs_read_prog(bank, FILLROW, 5555);
    pf.replay_send_resident(rp);
    int got = pf.receiveDataTry(rbk.data(), 8192 + 32, read_ms);
    uint32_t last_x = xs.back();
    bool uniform = (got >= 8192);
    if (got >= 8192) { const uint32_t* g = (const uint32_t*)rbk.data();
      long n_zero = 0, n_last = 0, n_other = 0; uint32_t first_other = 0; int fo_at = -1;
      for (int i = 0; i < 2048; i++) {
        if (g[i] == last_x) n_last++;
        else { uniform = false;
               if (g[i] == 0) n_zero++;
               else { if (!n_other) { first_other = g[i]; fo_at = i; } n_other++; } }
      }
      fprintf(stderr, "[desc-smoke]   C readback census: ==last_x %ld, ==0 %ld, other %ld"
              " (first other w[%d]=0x%08x); w[0..3]= %08x %08x %08x %08x\n",
              n_last, n_zero, n_other, fo_at, first_other, g[0], g[1], g[2], g[3]);
    }
    if (got < 8192) any_stall = true;
    bool ctl = ok && (r.n_drain >= 1) && (osk == 0);
    bool pass = uniform && ctl;
    printf("[desc-smoke] C: walk drain=%lld (expect ~0: ddr_wdata=x cancels on read) "
           "n_drain=%ld osk=%ld recv=%d ; readback row %u uniform==0x%08x? %s -> %s\n",
           r.drain_sum, r.n_drain, osk, rb, FILLROW, last_x, uniform ? "YES" : "NO",
           pass ? "PASS" : "FAIL");
    if (!pass) any_fail = true;
  }

  // ---- Sub-gate D: BRAM re-arm isolation (READ resident, ONE capture, TWO walks) ----
  printf("[desc-smoke] ==== Sub-gate D: BRAM re-arm isolation (READ resident) ====\n");
  if (!(R60000.legal && R60001.legal && R60002.legal && R60003.legal)) {
    printf("[desc-smoke] D: SKIP (a row illegal)\n"); any_fail = true;
  } else {
    // freshness rewrite (see read_walk comment)
    pf.set_readback_mode(false); pf.set_readback_mode(false); pf.drain_stray(800, 6);
    bool fr = ds_write_verify(pf, bank, R60000.row, R60000.pat, g_label++, read_ms)
           && ds_write_verify(pf, bank, R60001.row, R60001.pat, g_label++, read_ms)
           && ds_write_verify(pf, bank, R60002.row, R60002.pat, g_label++, read_ms)
           && ds_write_verify(pf, bank, R60003.row, R60003.pat, g_label++, read_ms);
    if (!fr) { printf("[desc-smoke] D: freshness rewrite FAILED\n"); any_fail = true; }
    pf.set_readback_mode_accxbp();
    pf.set_readback_mode_accxbp();                 // enter + clear
    long osk0 = pf.oversize_skips();
    Program res = rs_accum_read(bank, ZEROROW, g_label++);
    pf.replay_send_resident(res);                  // ONE resident send; walk2 does NOT resend
    usleep(5000);
    pf.set_readback_mode_accxbp();                 // clear the auto-run deposit
    // walk1 table {60000,60001} @ +1
    pf.dload(0, 0u, R60000.row, ds_acc(0,0));
    pf.dload(1, 0u, R60001.row, ds_acc(0,0));
    pf.replay_n(2);
    // walk1 needs tens of us on fabric (2 full-row reads + settle + flush);
    // the clear below must not land mid-walk — give it real margin.
    usleep(20000);
    // clear accumulator between walks (isolation), overwrite the desc BRAM,
    // fire again WITHOUT resending the resident (still in IMEM).
    pf.set_readback_mode_accxbp();                 // clear walk1 accumulator
    pf.dload(0, 0u, R60002.row, ds_acc(0,0));
    pf.dload(1, 0u, R60003.row, ds_acc(0,0));
    pf.replay_n(2);
    // One raw-capture session holds BOTH walks: the drain thread stays alive
    // (gaps between drains are h2c control words << the 500 ms quiet window).
    int want = 2 * 8192 + 40 * 32;
    std::vector<uint8_t> buf(want, 0);
    int rb = pf.receiveDataTry(buf.data(), want, recv_ms + 2000);
    long osk = pf.oversize_skips() - osk0;
    pf.drain_stray(300, 4);
    if (rb < 2 * 8192) any_stall = true;
    // Multi-drain scan (rs_parse keeps only the last drain; we need both). A
    // magic at M>=8192 with NO interior magic in (M-8192,M) is a flush drain.
    std::vector<long> mags;
    for (int off = 0; off + 4 <= rb; off += 4) {
      uint32_t w; memcpy(&w, buf.data() + off, 4);
      if ((w & 0xFFFFFF00u) == 0xDBC0DE00u) mags.push_back(off);
    }
    struct Dr { long long sum, lmn, lmx; uint32_t magic; };
    std::vector<Dr> drs;
    for (size_t k = 0; k < mags.size(); k++) {
      long M = mags[k]; bool is_drain = (M >= 8192);
      if (is_drain) for (size_t j = 0; j < mags.size(); j++)
                      if (mags[j] > M - 8192 && mags[j] < M) is_drain = false;
      if (!is_drain) continue;
      Dr d; d.sum = 0; d.lmn = 0; d.lmx = 0; d.magic = 0; bool first = true;
      for (long off = M - 8192; off < M; off += 4) {
        int32_t v; memcpy(&v, buf.data() + off, 4);
        d.sum += v;
        if (first) { d.lmn = d.lmx = v; first = false; }
        else { if (v < d.lmn) d.lmn = v; if (v > d.lmx) d.lmx = v; }
      }
      if (M + 4 <= rb) memcpy(&d.magic, buf.data() + M, 4);
      drs.push_back(d);
    }
    long long exp1 = 2048LL * __builtin_popcount(R60000.pat)
                   + 2048LL * __builtin_popcount(R60001.pat);
    long long exp2 = 2048LL * __builtin_popcount(R60002.pat)
                   + 2048LL * __builtin_popcount(R60003.pat);
    long long lane1 = (long long)__builtin_popcount(R60000.pat) + __builtin_popcount(R60001.pat);
    long long lane2 = (long long)__builtin_popcount(R60002.pat) + __builtin_popcount(R60003.pat);
    bool got2 = (drs.size() == 2);
    bool w1ok = got2 && drs[0].sum == exp1 && drs[0].lmn == drs[0].lmx && drs[0].lmn == lane1
                && drs[0].magic == img_magic;
    bool w2ok = got2 && drs[1].sum == exp2 && drs[1].lmn == drs[1].lmx && drs[1].lmn == lane2
                && drs[1].magic == img_magic;
    bool differ = got2 && (drs[0].sum != drs[1].sum);
    bool pass = got2 && (osk == 0) && w1ok && w2ok && differ;
    printf("[desc-smoke] D: n_drains=%zu osk=%ld recv=%d ; walk1 drain=%lld exp=%lld "
           "(%s) ; walk2 drain=%lld exp=%lld (%s) ; differ=%d -> %s\n",
           drs.size(), osk, rb, got2 ? drs[0].sum : 0, exp1, w1ok ? "ok" : "BAD",
           got2 ? drs[1].sum : 0, exp2, w2ok ? "ok" : "BAD", (int)differ,
           pass ? "PASS" : "FAIL");
    if (!pass) any_fail = true;
  }

  // ---- Sub-gate E (diagnostic): post-walk byte audit of the test rows ----
  // H-retention: aref is OFF and rows 60000+ are outside the maintained
  // production subarrays. If the walk "errors" are really cell decay between
  // write-verify and walk, the rows themselves now differ from their patterns
  // in READ mode too — the walk datapath would be exonerated.
  printf("[desc-smoke] ==== E (diag): post-walk byte audit (READ mode) ====\n");
  pf.set_readback_mode(false); pf.set_readback_mode(false); pf.drain_stray(1500, 8);
  auto audit_row = [&](const Row& r) {
    if (!r.legal) return;
    std::vector<uint8_t> rbuf(8192 + 32, 0);
    Program rp = rs_read_prog(bank, r.row, g_label++);
    pf.replay_send_resident(rp);
    int got = pf.receiveDataTry(rbuf.data(), 8192 + 32, read_ms);
    if (got < 8192) { printf("[desc-smoke] E row %u: short read %d\n", r.row, got); return; }
    const uint32_t* g = (const uint32_t*)rbuf.data();
    int nbad = 0, shown = 0;
    for (int i = 0; i < 2048; i++) if (g[i] != r.pat) {
      nbad++;
      if (shown < 6) { printf("[desc-smoke] E row %u w[%d]=0x%08x want 0x%08x xor=0x%08x\n",
                              r.row, i, g[i], r.pat, g[i] ^ r.pat); shown++; }
    }
    printf("[desc-smoke] E row %u: bad_words=%d/2048\n", r.row, nbad);
  };
  audit_row(R60000); audit_row(R60001); audit_row(R60002); audit_row(R60003);
  audit_row(R70000); audit_row(R131070);

  // ---- Sub-gate F (diagnostic): pure-time retention control, no walk ----
  // rewrite 60001, verify (t=0), idle 30 s with ZERO card activity, re-read.
  // Decay with no walk in the loop = retention, full stop.
  printf("[desc-smoke] ==== F (diag): 30 s retention control on row %u ====\n", R60001.row);
  if (R60001.legal) {
    bool w0 = ds_write_verify(pf, bank, R60001.row, R60001.pat, g_label++, read_ms);
    printf("[desc-smoke] F rewrite+verify t=0: %s\n", w0 ? "exact" : "BAD");
    sleep(30);
    audit_row(R60001);
  }

  // ---- restore READ mode + final verdict ----
  pf.set_readback_mode(false); pf.set_readback_mode(false); pf.drain_stray(1500, 8);
  if (any_stall) { printf("[desc-smoke] VERDICT: FAIL (receive stall)\n"); return 3; }
  if (any_fail)  { printf("[desc-smoke] VERDICT: FAIL (numeric/framing)\n"); return 1; }
  printf("[desc-smoke] VERDICT: ALL PASS\n");
  return 0;
}

// ============================================================================
// Task #50 M3 — server-side descriptor serving (PIM_DESC_SERVE=1).
//
// Replaces the handle path's per-(round x bp-chunk) execute/receiveData
// round-trips with, per bank: ONE walk-compatible resident + a chunked
// descriptor walk driven by DLOAD/REPLAY_N on the build-32 fabric. The
// readback engine's ACCUM_XBP accumulator folds sum_units sum_planes
// (+/-2^shift)*popcount(W & x) in hardware; the flush drain int32[2048] IS
// this bank's y contribution. Design of record:
// task50_desc_serve_design_2026_07_29.md (Serving shape M3 + v1 restrictions);
// silicon facts: desc_smoke_2026_07_29/RESULT.md; build contract:
// roadb_build32_2026_07_29/NOTES.md (trailer magic 0xDBC0DE1B).
//
// v1 restrictions (assert-and-fallback — NEVER a wrong answer): all rounds
// primary calib (open_rows baked into the resident); K=n_bitplanes <= 8 and
// every |bitplane_factor| a power of two <= 128 (the acc nibble is +/-2^shift,
// shift 0..7); PIM_FUSED_COSET in {1,3} (the walk body is intrinsically
// fused-5/5/5-shaped — see emit_desc_walk_body; a colmask bank additionally
// needs its per-round masks kept at LOAD so the marginal columns can be
// host-repaired exactly). Anything else -> the untouched process_matmul_handle.
// ============================================================================

static bool desc_serve_enabled() {
  static const int v = []{ const char* e = getenv("PIM_DESC_SERVE"); return (e && *e) ? atoi(e) : 0; }();
  return v > 0;
}
static bool desc_verbose() {
  static const int v = []{ const char* e = getenv("PIM_DESC_VERBOSE"); return (e && *e) ? atoi(e) : 0; }();
  return v > 0;
}
// Task #50 item 2 — drift-aware per-walk backup-row rewrite. DEFAULT ON; set
// PIM_DESC_REWRITE=0 for the A/B that leaves resident rows to drift.
static bool desc_rewrite_on() {
  static const int v = []{ const char* e = getenv("PIM_DESC_REWRITE"); return (e && *e) ? atoi(e) : 1; }();
  return v > 0;
}

// [sessreuse 2026-08-04] SESSION-REUSE rung (build50_flashday/SESSION_REUSE.md).
// Built ON TOP of the xsync twin (session_walk_close_len ACK-boundary
// discipline). When ON, process_matmul_desc_batch drops the three DEAD
// full-timeout drain_stray waits (entry 1500ms, per-bank tail 300ms x4, exit
// 1500ms) that DESC_PROF localized as ~82% of the per-frame constant C. Safe to
// drop under the build-50 one-ACK-per-session wire contract + the xsync per-sub
// ACK-boundary discipline: each framed walk self-delimits on its own closing
// correct-ACK (256 drain beats + ACK all consumed), so nothing crosses a
// walk/sub/bank boundary and there is no stray for a timeout-drain to find
// (DESC_PROF proved ZERO discarded bytes in every drain). The
// set_readback_mode(false) READ re-establish + per-bank rewrite mode-cycle are
// KEPT (retention correctness; the REWRITE=0 A/B goes non-exact) and cost
// ~0.4 ms of mode-SET words each. Default OFF -> byte-identical to the xsync
// twin so an A/B in one binary is a pure drain toggle.
static bool sessreuse_on() {
  // [PROMOTED 2026-08-04] Default ON under desc-serve (the build-50 ladder is the
  // production default); explicit PIM_DESC_SESSREUSE=0 still forces OFF for A/B.
  // Inert unless the XBATCH batch frame path (process_matmul_desc_batch) is entered.
  static const int v = []{ const char* e = getenv("PIM_DESC_SESSREUSE");
                           if (e && *e) return atoi(e);
                           return desc_serve_enabled() ? 1 : 0; }();
  return v > 0;
}

// [bankgen 2026-08-04] BANK-GENERIC RESIDENT rung (build50_flashday/BANKGEN.md).
// Built ON TOP of the sessreuse twin. When ON, process_matmul_desc_batch sends
// the resident walk-body ONCE (bank-generic) for the WHOLE frame instead of once
// per bank (x4), then walks every bank's descriptors out of that single resident
// with the build-32 per-descriptor bank patch armed (dload(..., bank, /*patch*/
// true) => the walker splices bank[1:0] into the resident's LI(bank, BAR) at
// fetch — platform.dload contract, re-validated by the b48 M2 bank-projection
// fix). This eliminates 3 of the 4 replay_send_resident calls and their ~590 ms
// raw-capture receiver quiet-drain floors (platform.cpp:1137-1139), the residual
// C ≈ 2.41 s the corrected SESSION_REUSE model localized as 98% the four floors.
//
// SOUNDNESS: a single resident is correct for all 4 banks ONLY if the resident
// is bank-INVARIANT apart from the bank register. The resident body embeds
// {bank_id -> BAR, Rfirst, Rsecond, open_rows[16], res_one/res_zero, use_consts,
// label_base}. bank_id is exactly what the bank patch overrides; the rest must
// be identical across banks. On the dimm2 trio they are (each bank's cs[0] is
// the SAME calibrated tuple 45340/45823/open_rows[..]; the cloneok pool is
// byte-identical per bank; PIM_RESIDENT_CONSTS off => use_consts=false, no res
// rows). We do NOT assume this: build_bankgen_invariant() rebuilds each bank's
// resident with bank[0]'s geometry vs its own and byte-compares — if ANY bank's
// geometry diverges, bankgen is DISABLED for the frame and it falls back to the
// per-bank send path (no fudge). Because per_column_write_row (the retention
// rewrite) executes programs that CLOBBER the resident IMEM, the single send
// must follow ALL banks' rewrites: the frame becomes two passes — pass 1 rewrite
// every bank (READ mode), then ONE enter+send, then pass 2 walk every bank
// (ACCXBP, no IMEM-clobbering op between the send and the walks). Default OFF =>
// byte-identical to the sessreuse twin (so an A/B in one binary isolates the
// single-send delta). Requires PIM_DESC_SESSREUSE=1 for the number to mean the
// corrected model's 0.402 -> ~0.09 s/sub move (drains already dropped).
static bool bankgen_on() {
  // [PROMOTED 2026-08-04] Default ON under desc-serve; explicit PIM_DESC_BANKGEN=0
  // forces OFF for A/B. Invariance-guarded (build_bankgen_invariant falls back to
  // the per-bank send path on any geometry divergence), inert unless XBATCH.
  static const int v = []{ const char* e = getenv("PIM_DESC_BANKGEN");
                           if (e && *e) return atoi(e);
                           return desc_serve_enabled() ? 1 : 0; }();
  return v > 0;
}
// [persist 2026-08-04] PERSISTENT-RECEIVER rung (build50_flashday/PERSIST.md).
// Built ON TOP of the bankgen twin. bankgen collapsed the 4 per-bank resident
// sends to 1, leaving ONE raw-capture receiver 500 ms quiet-drain floor per
// frame (platform.cpp consumeDataAccum QUIET_MS; the residual C ≈ 0.61 s, of
// which ~0.50 s is that single floor). That floor exists to absorb a
// stray/lagged beat after the send — but on the b50 image the walker's ORDTOL
// framed receive consumes each walk THROUGH its closing correct-ACK
// (session_walk_close_len: "Consuming through the closing ACK leaves NO beat in
// the c2h pipe"), and the b50 framer emits exactly ONE such ACK per session.
// So after the frame's LAST walk closes, the c2h pipe is provably empty and the
// quiet-drain is dead wait. When PIM_DESC_PERSIST=1 (bankgen already ON), the
// twin calls platform.receiver_boundary_stop() right after PASS 2 — the
// receiver exits on its next tick instead of the 500 ms floor, and the next
// frame's first rewrite execute() joins an already-finished receiver (0 ms
// instead of the floor). Boundary-terminated capture (task APPROACH 1), the
// first provably-safe option. Default OFF => byte-identical to bankgen (A/B in
// one binary). Requires PIM_DESC_BANKGEN=1 (and PIM_DESC_SESSREUSE=1) for the
// number to mean "bankgen minus the last floor".
static bool persist_on() {
  // [PROMOTED 2026-08-04] Default ON under desc-serve; explicit PIM_DESC_PERSIST=0
  // forces OFF for A/B (isolates the boundary-termination delta). Only takes
  // effect when bankgen is active (and requires XBATCH); boundary-terminated
  // capture is provably safe under the b50 one-ACK-per-session wire contract.
  static const int v = []{ const char* e = getenv("PIM_DESC_PERSIST");
                           if (e && *e) return atoi(e);
                           return desc_serve_enabled() ? 1 : 0; }();
  return v > 0;
}
// build_bankgen_invariant() (the geometry guard) is defined below,
// after build_desc_walk_resident / desc_use_consts which it calls.

// Task #50 per-session COUNTER GATE (the desc-serve ship blocker). Reads the
// recorder's fetch-corruption skip counters (malformed_skips/parity_skips) ONCE
// at server session teardown and fails loudly + nonzero-exits if either is
// nonzero (genuine IMEM storage/read corruption slipped behind the seam this
// session => every response already returned is SUSPECT). DEFAULT ON when
// PIM_DESC_SERVE=1; explicit PIM_COUNTER_GATE overrides either way. The read is
// a single out-of-band card round-trip at teardown ONLY — never in the timed
// request path.
static bool counter_gate_on() {
  static const int v = []{
    const char* e = getenv("PIM_COUNTER_GATE");
    if (e && *e) return atoi(e);
    return desc_serve_enabled() ? 1 : 0;   // default ON under desc-serve
  }();
  return v > 0;
}

// LI / ACT instruction decoders — byte-for-byte the gen_body_walk.cpp static
// check (frontend.v LI detection). Used ONLY by the desc-plan static
// discipline check (never on a Program that will be sent — get_inst_array()
// is non-idempotent, so each Program is decoded at most once).
static inline bool dsv_is_li(uint64_t w) {
  return ((w >> 48) & 0xFFu) == 6u && ((w >> 56) & 0xFFu) == 0u;
}
static inline int  dsv_li_rt(uint64_t w)  { return (int)((w >> 20) & 0xFu); }
static inline bool dsv_is_ddr(uint64_t w) { return (w >> 63) & 1u; }

// THE single resident-build code path — used at serve time AND by the
// desc-plan static check. Mirrors build_combined_clone_bcast_maj3_program's
// wrapper (CASR=8 stride, body, END); emit_desc_walk_body sets BAR/NUM_COLS
// itself and keeps the reg-6 (row) / reg-5 (x) patch surfaces free.
static Program build_desc_walk_resident(int bank_id, uint32_t Rfirst, uint32_t Rsecond,
                                        const uint32_t* open_rows, int label_base,
                                        uint32_t res_one, uint32_t res_zero, bool use_consts) {
  Program p;
  p.add_inst(SMC_LI(8, CASR));
  emit_desc_walk_body(p, bank_id, Rfirst, Rsecond, open_rows, label_base,
                      res_one, res_zero, use_consts);
  p.add_inst(SMC_END());
  return p;
}

// Static register-discipline check on a resident (desc-plan only). Returns
// true iff exactly one reg-6 LI (row patch site), two reg-5 LIs (x patch
// sites), exactly one ACT reading reg 6, and instr count <= 8192.
static bool desc_check_discipline(Program& p, int& n_inst, int& li_r6, int& li_r5, int& act_r6) {
  uint64_t* w = (uint64_t*)p.get_inst_array();
  n_inst = p.size() / 8;
  li_r6 = li_r5 = act_r6 = 0;
  for (int i = 0; i < n_inst; i++) {
    uint64_t inst = w[i];
    if (dsv_is_li(inst)) { int rt = dsv_li_rt(inst); if (rt == 6) li_r6++; if (rt == 5) li_r5++; }
    else if (dsv_is_ddr(inst)) {
      for (int s = 0; s < 4; s++) {
        uint16_t slot = (uint16_t)(inst >> (s * 16));
        int cmd = (slot >> 12) & 0xF;          // __DDR_CMD
        if (cmd == 11 /*__ACT*/) { int rar = (slot >> 4) & 0xF; if (rar == 6) act_r6++; }
      }
    }
  }
  free(w);
  return (li_r6 == 1) && (li_r5 == 2) && (act_r6 == 1) && (n_inst <= 8192);
}

// use_consts decision for a bank — mirrors emit_bank_combined_body's
// use_consts (resident-const clones instead of x/0 wrRows). The walk body
// only supports the fm==1 res-clone shape (clones BOTH res_one and res_zero),
// so require both rows present; if either is missing, fall back to the
// all-wrRow deposits (identical result, just more instructions). Gated on
// fm==1 to match the handle path exactly; fm==3 keeps use_consts=false.
static bool desc_use_consts(const BankConfig& b) {
  return resident_consts_mode() > 0 && fused_coset_mode() == 1 &&
         b.res_one_row != RES_ROW_NONE && b.res_zero_row != RES_ROW_NONE;
}

// [bankgen] Bank-invariance guard: is bank[bk]'s resident geometry byte-identical
// to bank[0]'s (holding bank_id EQUAL, so ONLY the geometry — Rfirst/Rsecond/
// open_rows/res rows/use_consts — is compared, never the bank register)? If every
// bank passes, one bank[0]-geometry resident + the build-32 per-descriptor bank
// patch is correct-by-construction for all banks. Returns false => a bank diverges
// => caller must NOT collapse the sends (falls back to the per-bank send path).
// Pure host-side (no card). Called once per frame.
static bool build_bankgen_invariant(std::vector<BankConfig>& banks, int label_base) {
  const int N = (int)banks.size();
  if (N <= 1) return true;
  const BankConfig& b0 = banks[0];
  const bool uc0 = desc_use_consts(b0);
  for (int bk = 1; bk < N; bk++) {
    const BankConfig& b = banks[bk];
    // (a) bank[0] geometry carrying bk's bank_id; (b) bk's own geometry + bk's
    // bank_id. Identical bank_id => any byte difference is pure geometry.
    Program a = build_desc_walk_resident(
        b.bank_id, b0.calib.Rfirst, b0.calib.Rsecond, b0.calib.open_rows.data(),
        label_base, b0.res_one_row, b0.res_zero_row, uc0);
    Program bb = build_desc_walk_resident(
        b.bank_id, b.calib.Rfirst, b.calib.Rsecond, b.calib.open_rows.data(),
        label_base, b.res_one_row, b.res_zero_row, desc_use_consts(b));
    bool same = (a.size() == bb.size());
    if (same) {
      uint64_t* aw = (uint64_t*)a.get_inst_array();
      uint64_t* bw = (uint64_t*)bb.get_inst_array();
      for (int i = 0; i < (int)(a.size() / 8); i++) if (aw[i] != bw[i]) { same = false; break; }
      free(aw); free(bw);
    }
    if (!same) {
      fprintf(stderr, "[bankgen] bank %d resident geometry DIVERGES from bank %d "
              "(Rf=%u/%u Rs=%u/%u) — NOT bank-invariant; per-bank send fallback\n",
              b.bank_id, b0.bank_id, b.calib.Rfirst, b0.calib.Rfirst,
              b.calib.Rsecond, b0.calib.Rsecond);
      return false;
    }
  }
  return true;
}

// acc nibble = (weight_neg << 3) | shift for a descriptor whose unit has the
// given sign (0 = +, 1 = -) and plane the given (encodable) factor. The
// weight's sign is (unit sign) XOR (factor sign) — exactly the handle path's
// weight = ((sign==0)?+1:-1) * bitplane_factor[b] fold at
// process_matmul_handle ~:3695-3697.
static inline uint32_t desc_acc_for(int sign, int32_t factor) {
  int fneg = 0, shift = 0;
  accxbp_encode(factor, &fneg, &shift);          // caller guarantees encodable
  return ds_acc((sign & 1) ^ fneg, shift);
}

// Lazy, once-per-process image-magic probe (replay-smoke idiom: raw-capture a
// plain READ, read its trailer magic). Returns true iff the flashed image is
// build-32+ (magic >= 0xDBC0DE1B). Leaves the engine in READ mode.
static bool desc_probe_image(SoftMCPlatform& pf, std::vector<BankConfig>& banks) {
  if (banks.empty()) { fprintf(stderr, "[desc-serve] probe: no banks — disabled\n"); return false; }
  int bank = banks[0].bank_id;
  uint32_t row = banks[0].calib.open_rows.empty() ? 0u : banks[0].calib.open_rows[0];
  pf.set_readback_mode(false); pf.set_readback_mode(false);
  pf.drain_stray(1500, 8);
  std::vector<uint8_t> pbuf(8192 + 32, 0);
  Program probe = rs_read_prog(bank, row, 990001);
  pf.replay_send_resident(probe);
  int pgot = pf.receiveDataTry(pbuf.data(), 8192 + 32, 6000);
  pf.drain_stray(800, 6);
  if (pf.recv_stalled() || pgot < 8192 + 4) {
    fprintf(stderr, "[desc-serve] image probe short/stalled (%d/%d) — desc-serve "
            "DISABLED this process (permanent fallback)\n", pgot, 8192 + 32);
    return false;
  }
  uint32_t img = 0; memcpy(&img, pbuf.data() + 8192, 4);
  if ((img & 0xFFFFFF00u) != 0xDBC0DE00u) {
    fprintf(stderr, "[desc-serve] image trailer 0x%08x is not a 0xDBC0DExx magic — "
            "desc-serve DISABLED this process\n", img);
    return false;
  }
  // [b49 deframer] capture the image build-tag for session_deframe_on()'s
  // auto-detect (framer images start at 0xDBC0DE30). The probe program runs
  // with the walker IDLE, so its trailer is pass-through on BOTH images.
  g_image_magic = img;
  fprintf(stderr, "[deframe] image magic 0x%08x -> session deframer %s (env=%d)\n",
          img, session_deframe_on() ? "ON (framed sessions)" : "off (per-program)",
          session_deframe_env());
  if (img < 0xDBC0DE1Bu) {
    fprintf(stderr, "[desc-serve] image magic 0x%08x < 0xDBC0DE1B (build-32): descriptor "
            "serving (8-bit idx + retimed x-patch) unavailable — permanent fallback to the "
            "handle path this process\n", img);
    return false;
  }
  fprintf(stderr, "[desc-serve] image magic 0x%08x (>= 0xDBC0DE1B) — descriptor serving ENABLED\n", img);
  return true;
}

// The serving path. Returns 0 (handled + response written), 2 (ineligible or
// probe failed — the caller MUST fall back to process_matmul_handle, which
// re-runs the whole request from the intact req_buf), or -1 (hard error, e.g.
// a poisoned platform the handle path could not recover from either).
static int process_matmul_desc(SoftMCPlatform& platform,
                               std::vector<BankConfig>& banks,
                               const std::map<uint32_t, LoadedHandle>& handles,
                               const uint8_t* req, size_t req_len,
                               int& label_base, int response_fd) {
  init_debug_flags();
  using clk = std::chrono::steady_clock;
  using ns_t = std::chrono::nanoseconds;
  auto t_req_start = clk::now();

  // Fast permanent-fallback once the probe has failed this process.
  static int s_desc_probe = -1;   // -1 unprobed, 0 fail (perm fallback), 1 ok
  if (s_desc_probe == 0) return 2;

  // ---- parse header (identical to process_matmul_handle) ----
  if (req_len < 5 * 4) return 2;
  size_t off = 0;
  auto rd_u32 = [&](uint32_t& v){ memcpy(&v, req + off, 4); off += 4; };
  uint32_t magic, handle_id, d_out, n_chunks, n_bitplanes;
  rd_u32(magic); rd_u32(handle_id); rd_u32(d_out);
  rd_u32(n_chunks); rd_u32(n_bitplanes);
  if (d_out != 2048) return 2;
  auto it = handles.find(handle_id);
  if (it == handles.end()) return 2;             // handle path emits the canonical error
  const LoadedHandle& h = it->second;
  if (h.n_chunks != n_chunks) return 2;
  size_t need = 5 * 4 + (size_t)n_chunks * n_bitplanes * 4 + (size_t)n_bitplanes * 4;
  if (req_len < need) return 2;
  const uint32_t* x_bitplane_all = (const uint32_t*)(req + off);
  off += (size_t)n_chunks * n_bitplanes * 4;
  const int32_t*  bitplane_factor = (const int32_t*)(req + off);

  const int N = (int)banks.size();
  const size_t n_units  = h.n_units;
  const size_t n_rounds = h.n_rounds;
  const uint32_t K = n_bitplanes;

  auto inel = [&](const char* why) -> int {
    static std::set<std::string> seen;
    if (seen.insert(std::string(why)).second)
      fprintf(stderr, "[desc-serve] handle=%u INELIGIBLE (%s) — using the handle path\n",
              handle_id, why);
    return 2;
  };

  // ---- eligibility (assert-and-fallback) ----
  if (K == 0 || K > 8) return inel("n_bitplanes(K) not in [1,8]");
  for (uint32_t b = 0; b < n_bitplanes; b++) {
    int neg, shift;
    if (!accxbp_encode(bitplane_factor[b], &neg, &shift))
      return inel("a bitplane_factor is not +/- a power of two <= 128");
  }
  const int fm = fused_coset_mode();
  if (!(fm == 1 || fm == 3))
    return inel("PIM_FUSED_COSET not in {1,3} (the walk body is fused-shaped)");
  if (!h.per_round_calib_sel.empty()) {
    for (size_t r = 0; r < n_rounds && r < h.per_round_calib_sel.size(); r++)
      for (int bk = 0; bk < N && bk < (int)h.per_round_calib_sel[r].size(); bk++)
        if (h.per_round_calib_sel[r][bk] != 0)
          return inel("a round is on a non-primary calib (per_round_calib_sel!=0)");
  }
  for (int bk = 0; bk < N; bk++)
    if (banks[bk].calib.open_rows.size() < 16)
      return inel("a bank calib has < 16 open_rows");
  // Colmask banks: masks must cover every unit they own (for exact repair).
  bool any_colmask = false;
  for (int bk = 0; bk < N; bk++) if (!banks[bk].fused_col_bad.empty()) any_colmask = true;
  if (any_colmask) {
    if (h.all_round_masks.size() < n_rounds)
      return inel("colmask bank but per-round masks not kept at LOAD");
    for (size_t r = 0; r < n_rounds; r++)
      for (int bk = 0; bk < N; bk++) {
        size_t u = r * (size_t)N + (size_t)bk;
        if (u >= n_units || banks[bk].fused_col_bad.empty()) continue;
        if (r >= h.all_round_masks.size() || bk >= (int)h.all_round_masks[r].size() ||
            h.all_round_masks[r][bk].empty())
          return inel("colmask bank missing a unit's mask");
      }
  }

  // ---- POOL-WRAP eligibility (Task #50 item 3) ----
  // The desc walk (and item-2's per-walk rewrite) assume each descriptor's
  // backup row holds exactly that (round,bank) unit's weights. If the per-bank
  // backup pool WRAPPED, one physical row is aliased across two units — a walk
  // that rewrites it for one round clobbers the other round's weights. v1 stays
  // conservative: require distinct rows per bank, else fall back to the handle
  // path (which re-writes every round's pool entry itself, wrap-safe). (With
  // item-2's rewrite this aliasing is arguably servable per-walk; deferred.)
  for (int bk = 0; bk < N; bk++) {
    std::map<uint32_t, size_t> row_first_round;      // backup row -> first owning round
    for (size_t r = 0; r < n_rounds; r++) {
      size_t u = r * (size_t)N + (size_t)bk;
      if (u >= n_units) continue;                    // unit not owned by this bank
      if (r >= h.per_round_backup_rows.size() ||
          bk >= (int)h.per_round_backup_rows[r].size()) continue;
      uint32_t row = h.per_round_backup_rows[r][bk];
      auto ins = row_first_round.emplace(row, r);
      if (!ins.second) {
        char reason[192];
        snprintf(reason, sizeof(reason),
                 "backup-row aliasing: bank %d row %u used by rounds %zu and %zu (pool wrapped)",
                 banks[bk].bank_id, row, ins.first->second, r);
        return inel(reason);
      }
    }
  }

  // ---- lazy image-magic probe (once/process) ----
  if (s_desc_probe < 0) s_desc_probe = desc_probe_image(platform, banks) ? 1 : 0;
  if (s_desc_probe == 0) return 2;

  // ---- ensure NO stream session, land a clean READ baseline ----
  if (g_stream_session) {
    platform.stream_stop();
    g_stream_session = false;
    platform.drain_stray(800, 6);
  }
  platform.set_readback_mode(false); platform.set_readback_mode(false);
  platform.drain_stray(1500, 8);
  g_mode_accxbp_now = false; g_mode_segpop_now = false;

  // Restore-and-fallback helper: leave the engine in READ, reset the host mode
  // view, and hand the WHOLE request to process_matmul_handle (correct answer;
  // wasted desc work only on the rare framing surprise). No response written.
  auto restore_and_fallback = [&]() -> int {
    platform.set_readback_mode(false); platform.set_readback_mode(false);
    platform.drain_stray(1500, 8);
    g_mode_accxbp_now = false; g_mode_segpop_now = false;
    return 2;
  };

  // ---- serving state ----
  vector<int32_t> y(d_out, 0);
  const bool do_skip = skip_zero_planes_on();   // design doc: zero-x descriptors
                                                // contribute 0; skip trivially (fused too)
  // [desc-dbg] PIM_DESC_WALK_MAX: cap the walk chunk (1..256) so each drain
  // isolates fewer descriptors (1 = per-descriptor contributions). DEFAULT is
  // 256 since 2026-08-04 (the hardware dload idx bound): gated on clean b48
  // silicon — C55 oracle desc-serve guard-off d2560/n512 = 512/512 bit-exact
  // at 256 with 2.95x wall (29.07 -> 9.84 s; descent_2026_08_04 batch_B).
  // Was 32 (the desc-dbg isolation default). Override via env for A/B;
  // desc-serve only — the handle path never reads this.
  static const int s_walk_max = []{
    const char* v = getenv("PIM_DESC_WALK_MAX");
    long n = (v && *v) ? atol(v) : 256;
    if (n < 1) n = 1; if (n > 256) n = 256;
    return (int)n;
  }();
  const int  WALK_MAX = s_walk_max;             // walk chunk cap (dload idx < 256)
  long n_desc_emitted = 0, n_zunits_seen = 0, n_zskip_units = 0;
  long n_walks = 0, n_recv_wakes = 0;
  long n_maj3_execs = 0;                         // = walked descriptors (liveness)
  long n_rewrites = 0;                           // item 2: per-walk backup-row rewrites
  const bool verbose = desc_verbose();
  const bool rewrite_on = desc_rewrite_on();     // item 2: drift-aware per-walk rewrite

  // ---- DEFECT B interim guard (task #59, 2026-08-01) ----------------------
  // The gross "s=4/5 twin" is a beat-periodic word-pinning defect in the shared
  // ACCUM_XBP drain datapath: lanes[16k+4]==lanes[16k+5] (nonzero) on EVERY
  // 512-bit beat of the raw drain (bytes 16-23 of each beat), armed mid-run and
  // persistent for the rest of the process. It is a FABRIC defect (the raw
  // rs_parse drain bytes are already track-signed-pinned, upstream of any host
  // math), so a per-walk redo cannot clear it and a handle-path fallback would
  // re-hit the same armed ACCUM datapath. The only host-side guaranteed-correct
  // recovery is EXACT re-computation of the affected columns from the stored
  // LOAD masks (no re-read needed) — the same substitution the fused_col_bad
  // path already performs. Default ON (no-op on clean runs: the signature is
  // absent, so behavior is bit-identical); PIM_DESC_B_GUARD=0 disables for A/B.
  static const bool s_bguard = []{
    const char* v = getenv("PIM_DESC_B_GUARD"); return !(v && atoi(v) == 0); }();
  // #beats (of 128) that must show the equal-nonzero word-4/5 pair to declare
  // a walk armed. Gross mode pins ALL 128; benign runs show ~0 (chance of a
  // spurious equal-nonzero pair at a fixed stride-16 offset is negligible).
  static const int B_GUARD_BEATS = []{
    const char* v = getenv("PIM_DESC_B_GUARD_BEATS"); return (v && *v) ? atoi(v) : 16; }();
  long n_bguard_repairs = 0;                      // #banks exact-repaired for defect B

  // [desc-prof 2026-08-01 task#61] Additive, env-gated (PIM_DESC_PROF=1) per-phase
  // session-machinery timers. clk::now() is ~20ns so always-on accumulation costs
  // ~a few us/request; the report line is printed only when s_prof. These name the
  // ~725 ms/walk session overhead into its parts (build/enter/resend/usleep/rewrite/
  // dload/replay/recv/drain) for the session-per-walk vs multiwalk A/B.
  static const bool s_prof = []{ const char* v = getenv("PIM_DESC_PROF"); return v && atoi(v) > 0; }();
  long long tp_build=0, tp_enter=0, tp_resend=0, tp_usleep=0,
            tp_rewrite=0, tp_dload=0, tp_replay=0, tp_recv=0, tp_drain=0;
  long tp_enter_calls=0, tp_drain_calls=0;

  for (int bk = 0; bk < N; bk++) {
    const BankConfig& b = banks[bk];

    // Enumerate this bank's descriptors: rounds it owns x planes.
    std::vector<DsDesc> descs;
    for (size_t r = 0; r < n_rounds; r++) {
      size_t u = r * (size_t)N + (size_t)bk;
      if (u >= n_units) continue;
      uint32_t chunk = (uint32_t)(u / 2);
      int sign = (int)(u % 2);
      uint32_t row = h.per_round_backup_rows[r][bk];
      for (uint32_t bp = 0; bp < K; bp++) {
        uint32_t xb = x_bitplane_all[(size_t)chunk * K + bp];
        n_zunits_seen++;
        if (do_skip && xb == 0) { n_zskip_units++; continue; }
        uint32_t acc = desc_acc_for(sign, bitplane_factor[bp]);
        if (verbose)
          fprintf(stderr, "[desc-dbg] emit k=%zu bank=%d r=%zu u=%zu chunk=%u sign=%d "
                  "bp=%u factor=%d x=%08x row=%u acc=0x%x (neg=%d shift=%d)\n",
                  descs.size(), b.bank_id, r, u, chunk, sign, bp,
                  (int)bitplane_factor[bp], xb, row, acc,
                  (int)((acc >> 3) & 1), (int)(acc & 7));
        descs.push_back({ xb, row, acc, (uint32_t)r });   // carry owning round (item 2)
      }
    }
    n_desc_emitted += (long)descs.size();
    if (descs.empty()) {
      if (verbose) fprintf(stderr, "[desc-serve]   bank %d: 0 descriptors (all zero-x) — skipped\n", b.bank_id);
      continue;   // zero contribution; do not disturb the engine
    }

    // Build the resident ONCE for this bank's primary calib. The SAME Program
    // object is re-sent for EVERY walk in session-per-walk mode; get_inst_array()
    // finalizes exactly once (prog.cpp `finalized` flag), so repeated
    // replay_send_resident(resident) is idempotent — this is exactly what the
    // desc-smoke rs_read_prog probes do across a session.
    const bool uc = desc_use_consts(b);
    auto _tbd = clk::now();
    Program resident = build_desc_walk_resident(
        b.bank_id, b.calib.Rfirst, b.calib.Rsecond, b.calib.open_rows.data(),
        label_base + bk * 1000, b.res_one_row, b.res_zero_row, uc);
    tp_build += std::chrono::duration_cast<ns_t>(clk::now() - _tbd).count();
    if (resident.size() / 8 > g_bitstream_imem) {
      fprintf(stderr, "[desc-serve] handle=%u bank %d resident %d insts > IMEM %d — fallback\n",
              handle_id, b.bank_id, (int)(resident.size() / 8), g_bitstream_imem);
      return restore_and_fallback();
    }

    // Session shape. DEFAULT = SESSION-PER-WALK: every walk chunk gets its own
    // raw-capture session (enter+clear accxbp x2 -> send resident (auto-runs
    // once) -> usleep -> clear auto-run deposit -> dloads -> replay_n ->
    // trailer-seeking receive -> drain_stray). On silicon, MULTI-walk sessions
    // wedge the fabric nondeterministically (walk 2/7/8/16 across runs, recv=0,
    // durable rig corruption); single-walk-per-session is empirically safe at
    // scale (desc-smoke: ~20 sessions/process, 1-2 walks each, dozens of runs,
    // zero wedges). PIM_DESC_MULTIWALK=1 restores the build-32 shape (send
    // once/bank, many walks share one session) for A/B.
    static const bool s_multiwalk = []{
      const char* v = getenv("PIM_DESC_MULTIWALK"); return v && atoi(v) > 0; }();

    // Enter ACCUM_XBP (enter + 128-cyc clear), send resident (auto-runs once,
    // raw-capture armed), let the auto-run ack land, clear its garbage deposit
    // — the silicon-proven do_walk sequence (desc_smoke RESULT.md finding #3).
    // Returns 0 on success, 2 to request the handle-path fallback.
    auto enter_and_send = [&]() -> int {
      auto _ten = clk::now();
      platform.set_readback_mode_accxbp();
      platform.set_readback_mode_accxbp();
      g_mode_accxbp_now = true; g_mode_segpop_now = false;
      long osk0 = platform.oversize_skips();
      auto _trs = clk::now();
      platform.replay_send_resident(resident);
      tp_resend += std::chrono::duration_cast<ns_t>(clk::now() - _trs).count();
      if (platform.oversize_skips() != osk0) {
        fprintf(stderr, "[desc-serve] handle=%u bank %d resident oversize-skipped — fallback\n",
                handle_id, b.bank_id);
        return 2;
      }
      auto _tus = clk::now();
      usleep(5000);
      tp_usleep += std::chrono::duration_cast<ns_t>(clk::now() - _tus).count();
      platform.set_readback_mode_accxbp();        // clear the auto-run deposit
      tp_enter += std::chrono::duration_cast<ns_t>(clk::now() - _ten).count();
      tp_enter_calls++;
      return 0;
    };

    // [2026-08-04 MULTIWALK STRICT GATE] REWRITE-IN-MULTIWALK: multiwalk
    // previously skipped the drift-aware rewrite entirely (rewrites need READ
    // mode; the shared session holds ACCUM_XBP), which left its 3.91x A/B
    // provisional (511/512, rewrites=0). Do the whole bank's distinct-round
    // rewrites HERE, before the session enter — the same mechanism as the
    // session-per-walk block, hoisted to the bank head. Exposure bound = all
    // of this bank's walks in one session (<= descs.size() bodies); the
    // WALK_MAX=256 session-per-walk gate already passed 512/512 at the same
    // exposure class. PIM_DESC_REWRITE=0 still disables for A/B.
    if (s_multiwalk && rewrite_on && !h.all_round_masks.empty()) {
      auto _trw = clk::now();
      platform.set_readback_mode(false); platform.set_readback_mode(false);
      g_mode_accxbp_now = false; g_mode_segpop_now = false;
      std::set<uint32_t> done_rounds;
      for (size_t k = 0; k < descs.size(); k++) {
        uint32_t r = descs[k].round;
        if (!done_rounds.insert(r).second) continue;    // each round once/bank
        if (r >= h.all_round_masks.size() || bk >= (int)h.all_round_masks[r].size() ||
            h.all_round_masks[r][bk].empty()) continue;  // no mask kept -> skip
        per_column_write_row(platform, b.bank_id, descs[k].row,
                             h.all_round_masks[r][bk].data());
        n_rewrites++;
      }
      tp_rewrite += std::chrono::duration_cast<ns_t>(clk::now() - _trw).count();
    }
    // Multiwalk (A/B): one enter+send per bank, then all walks share the
    // session. Session-per-walk defers enter+send into each walk iteration.
    if (s_multiwalk && enter_and_send() == 2) return restore_and_fallback();

    // Chunked descriptor walk. One receive per walk (the walk's own drain +
    // trailer is the sync); rs_parse gives that walk's single 8192-B drain.
    std::vector<int64_t> bank_acc(d_out, 0);
    bool b_armed = false; long b_armed_beats = 0;   // defect B guard, per bank
    const size_t n_bank_walks = (descs.size() + (size_t)WALK_MAX - 1) / (size_t)WALK_MAX;
    bool bank_ok = true;
    // [desc-dbg] PIM_DESC_CLEAR_BEFORE=1: A/B the accumulator-clear position
    // (before the DLOAD burst = the original order; default = after, right
    // before REPLAY_N, which shields the DLOAD-window maintenance deposits).
    static const bool s_clear_before = []{
      const char* v = getenv("PIM_DESC_CLEAR_BEFORE"); return v && atoi(v) > 0; }();
    for (size_t w = 0; w < n_bank_walks && bank_ok; w++) {
      const size_t base = w * (size_t)WALK_MAX;
      const size_t nd   = std::min((size_t)WALK_MAX, descs.size() - base);
      // ---- Task #50 item 2 — DRIFT-AWARE REWRITE (session-per-walk only) ----
      // BEFORE this chunk's walk, per-column re-write the backup rows its
      // descriptors reference, from h.all_round_masks — EXACTLY the refresh the
      // handle path does per round (g_load_rewrite, ~:3355). The desc walk reads
      // resident backup rows raw; O4(a) drift saturates a same-subarray backup
      // row within ~80-160 MAJ3 bodies, and only the handle path's per-round
      // rewrite kept it correct. A walk chunk is ~WALK_MAX (32) bodies, so a
      // per-walk refresh bounds each row's drift exposure to ONE walk — well
      // under the saturation onset. WRITES require READ mode, so this MUST
      // precede enter_and_send()'s ACCUM_XBP. Distinct rounds only (a round's K
      // bitplanes share one backup row; enumeration is round-major so a chunk
      // sees each round's descriptors contiguously). PIM_DESC_REWRITE=0 for A/B.
      if (rewrite_on && !s_multiwalk && !h.all_round_masks.empty()) {
        auto _trw = clk::now();
        platform.set_readback_mode(false); platform.set_readback_mode(false);
        g_mode_accxbp_now = false; g_mode_segpop_now = false;
        std::set<uint32_t> done_rounds;
        for (size_t k = 0; k < nd; k++) {
          uint32_t r = descs[base + k].round;
          if (!done_rounds.insert(r).second) continue;    // rewrite each round once/walk
          if (r >= h.all_round_masks.size() || bk >= (int)h.all_round_masks[r].size() ||
              h.all_round_masks[r][bk].empty()) continue;  // no mask kept -> skip (best-effort)
          per_column_write_row(platform, b.bank_id, descs[base + k].row,
                               h.all_round_masks[r][bk].data());
          n_rewrites++;
        }
        tp_rewrite += std::chrono::duration_cast<ns_t>(clk::now() - _trw).count();
      }
      // Session-per-walk: fresh enter+send+auto-run-clear session for THIS chunk.
      if (!s_multiwalk && enter_and_send() == 2) return restore_and_fallback();
      if (s_clear_before) platform.set_readback_mode_accxbp();
      auto _tdl = clk::now();
      for (size_t k = 0; k < nd; k++) {
        const DsDesc& d = descs[base + k];
        platform.dload((uint32_t)k, d.x, d.row, d.acc);   // idx<256, bank_patch_en=false (v1)
      }
      tp_dload += std::chrono::duration_cast<ns_t>(clk::now() - _tdl).count();
      // [desc-fix 2026-07-29] Clear the accumulator AFTER the DLOAD burst,
      // immediately before REPLAY_N — not before the DLOADs. The DLOAD
      // window is multi-ms of frontend IDLE_S where maintenance passes run;
      // reads landing in the accumulator during that window polluted the
      // walk drains (deep d2560: nz lanes 2048/2048, periodic-pattern
      // deposits, run-to-run nondeterministic). The clear (128-cyc sweep,
      // <1 us) completes long before the REPLAY_N word's separate h2c beat
      // arrives, and it touches neither desc_mem nor desc_mode. Walk 0's
      // auto-run garbage deposit is erased by this same clear.
      if (!s_clear_before) platform.set_readback_mode_accxbp();
      auto _trp = clk::now();
      platform.replay_n((uint16_t)nd);
      tp_replay += std::chrono::duration_cast<ns_t>(clk::now() - _trp).count();
      // Silicon framing (replay-smoke proven): walk 0 also carries the send's
      // auto-run ack (+4 records = auto-run + 2 zero-delta + drain trailer);
      // later walks +3. `want` normally lands EXACTLY on the drain trailer so
      // receiveDataTry returns promptly; a framing surprise never poisons
      // (non-poisoning try) — it is caught by the n_drain check below.
      //
      // [desc-fix 2026-07-29] The pre-drain record count is NOT a constant:
      // the "~2 zero-delta records" of the build-27 framing note is measured
      // variability (deep d2560 walk 2 carried THREE — census n_mag=67 for
      // nd=64 — pushing the drain trailer past the fixed window and tripping
      // a spurious FRAMING ANOMALY -> handle-path fallback). The receive is
      // therefore TRAILER-SEEKING: bulk-read `want`; if the window filled but
      // the drain trailer is not inside it, keep pulling 32-B records
      // (bounded slack) until the trailer lands. A SHORT bulk read (session
      // starved/wedged) skips the top-up and falls to the anomaly branch.
      // Session-per-walk: EVERY walk has its own send, so every walk carries the
      // auto-run ack (+4). Multiwalk: only walk 0 carries the single send's
      // auto-run ack (+4); later walks +3.
      const int extra = (!s_multiwalk || w == 0) ? 4 : 3;
      const int want  = 8192 + (int)((nd + (size_t)extra) * 32);
      const int SLACK_RECORDS = 16;                 // top-up budget (32 B each)
      std::vector<uint8_t> buf(want + SLACK_RECORDS * 32, 0);
      auto _trc = clk::now();
      RsParse r; int rb = 0; int topups = 0;
      if (session_deframe_on()) {
        // [b49] framed session: N per-program trailers are COLLAPSED into one
        // checksum-valid 0x2A ACK per walk session (CONTRACT §1/§2); receive
        // by content until the ACK, never by the legacy per-program count.
        SessTrailer st;
        int rc_f = framed_walk_receive(platform, nd, r, st, &rb);
        n_recv_wakes++;
        if (rc_f == -1) {
          fprintf(stderr, "[desc-serve] handle=%u bank %d walk %zu FRAMED receive "
                  "STALLED (%d) — platform poisoned, hard error\n",
                  handle_id, b.bank_id, w, rb);
          return -1;
        }
        // rc_f==1 (no ACK) leaves r.n_drain==0 -> the anomaly branch below.
      } else {
        rb = platform.receiveDataTry(buf.data(), want, 8000);
        n_recv_wakes++;
        if (platform.recv_stalled()) {
          fprintf(stderr, "[desc-serve] handle=%u bank %d walk %zu receive STALLED (%d/%d) "
                  "— platform poisoned, hard error\n", handle_id, b.bank_id, w, rb, want);
          return -1;
        }
        r = rs_parse(buf.data(), rb);
        while (r.n_drain == 0 && rb >= want && topups < SLACK_RECORDS) {
          int more = platform.receiveDataTry(buf.data() + rb, 32, 1000);
          if (platform.recv_stalled()) {
            fprintf(stderr, "[desc-serve] handle=%u bank %d walk %zu top-up STALLED "
                    "— platform poisoned, hard error\n", handle_id, b.bank_id, w);
            return -1;
          }
          if (more < 32) break;                     // nothing more coming
          rb += more; topups++;
          r = rs_parse(buf.data(), rb);
        }
      }
      tp_recv += std::chrono::duration_cast<ns_t>(clk::now() - _trc).count();
      if (verbose && topups)
        fprintf(stderr, "[desc-dbg]   walk %zu framing top-up: +%d records "
                "(variable pre-drain count), drain %s\n",
                w, topups, r.n_drain ? "FOUND" : "still missing");
      if (r.n_drain != 1 || (int)r.lanes.size() != (int)d_out ||
          (r.drain_magic != 0 && r.drain_magic < 0xDBC0DE1Bu)) {
        fprintf(stderr, "[desc-serve] handle=%u bank %d walk %zu FRAMING ANOMALY "
                "(n_drain=%ld lanes=%zu magic=0x%08x recv=%d want=%d) — falling back\n",
                handle_id, b.bank_id, w, r.n_drain, r.lanes.size(), r.drain_magic, rb, want);
        if (verbose && rb > 0) {
          // [desc-dbg] anomaly census: what DID arrive? magic count, ack
          // count, zero/nonzero word census, head+tail hexdump.
          long nz_words = 0, z_words = 0;
          for (int off = 0; off + 4 <= rb; off += 4) {
            uint32_t wv; memcpy(&wv, buf.data() + off, 4);
            if (wv) nz_words++; else z_words++;
          }
          fprintf(stderr, "[desc-dbg] anomaly census: n_mag=%ld n_empty_ack=%ld "
                  "nz_words=%ld zero_words=%ld\n", r.n_mag, r.n_empty_ack, nz_words, z_words);
          for (int base_off = 0; base_off < rb && base_off < 128; base_off += 32) {
            uint32_t wv[8] = {0};
            memcpy(wv, buf.data() + base_off, std::min(32, rb - base_off));
            fprintf(stderr, "[desc-dbg]   head+%04x: %08x %08x %08x %08x %08x %08x %08x %08x\n",
                    base_off, wv[0], wv[1], wv[2], wv[3], wv[4], wv[5], wv[6], wv[7]);
          }
          int tail0 = (rb > 128) ? (rb - 128) & ~31 : 0;
          for (int base_off = tail0; base_off < rb && base_off > 128 - 32; base_off += 32) {
            uint32_t wv[8] = {0};
            memcpy(wv, buf.data() + base_off, std::min(32, rb - base_off));
            fprintf(stderr, "[desc-dbg]   tail+%04x: %08x %08x %08x %08x %08x %08x %08x %08x\n",
                    base_off, wv[0], wv[1], wv[2], wv[3], wv[4], wv[5], wv[6], wv[7]);
          }
        }
        bank_ok = false;
        break;
      }
      // ---- DEFECT B detection: beat-periodic word-4/5 equal-pair signature ----
      // Scan this walk's RAW drain (rs_parse, pre-accumulation). Count 512-bit
      // beats where lane 4 == lane 5 (nonzero). Gross mode pins all 128.
      if (s_bguard) {
        long eqp = 0;
        for (int k = 0; k + 5 < (int)r.lanes.size(); k += 16)
          if (r.lanes[k + 4] == r.lanes[k + 5] && r.lanes[k + 4] != 0) eqp++;
        if (eqp >= B_GUARD_BEATS) {
          b_armed = true; b_armed_beats += eqp;
          if (verbose)
            fprintf(stderr, "[desc-Bguard] bank %d walk %zu: word-4/5 equal-pair on %ld/%d "
                    "beats — DEFECT B armed (will exact-repair cols%%16in{4,5})\n",
                    b.bank_id, w, eqp, (int)(r.lanes.size() / 16));
        }
      }
      for (int j = 0; j < (int)d_out; j++) bank_acc[j] += r.lanes[j];
      n_walks++;
      n_maj3_execs += (long)nd;
      if (verbose) {
        fprintf(stderr, "[desc-serve]   bank %d walk %zu: nd=%zu recv=%d drain_sum=%lld magic=0x%08x\n",
                b.bank_id, w, nd, rb, (long long)r.drain_sum, r.drain_magic);
        fprintf(stderr, "[desc-dbg]   walk %zu (desc %zu..%zu) lanes[0..7]= %d %d %d %d %d %d %d %d  "
                "acks=%ld nz=%d\n", w, base, base + nd - 1,
                r.lanes[0], r.lanes[1], r.lanes[2], r.lanes[3],
                r.lanes[4], r.lanes[5], r.lanes[6], r.lanes[7],
                r.n_empty_ack, r.nz_lanes);
        // [desc-dbg] PIM_DESC_DUMP_LANES=1: full nonzero-lane dump per walk
        // (idx:val), for host-side transform identification (rotation /
        // word-collapse / pollution) against the Python oracle.
        static const bool s_dump_lanes = []{
          const char* v = getenv("PIM_DESC_DUMP_LANES"); return v && atoi(v) > 0; }();
        if (s_dump_lanes) {
          fprintf(stderr, "[desc-lanes] walk %zu:", w);
          for (int j = 0; j < (int)r.lanes.size(); j++)
            if (r.lanes[j]) fprintf(stderr, " %d:%d", j, r.lanes[j]);
          fprintf(stderr, "\n");
        }
      }
      // Session-per-walk: close THIS walk's raw-capture session cleanly before
      // the next walk's enter+send. (Multiwalk drains once after the loop.)
      if (!s_multiwalk) { auto _tdr = clk::now(); platform.drain_stray(300, 4);
        tp_drain += std::chrono::duration_cast<ns_t>(clk::now() - _tdr).count(); tp_drain_calls++; }
      // [drift-ladder rung 3 2026-08-01] PIM_DESC_WALK_SLEEP_MS: env-gated idle
      // stretch INSIDE the session-boundary window (between walks, not after a
      // bank's last). Fixed session count, stretched un-recharged exposure —
      // the exposure-TIME vs boundary-EVENT-COUNT discriminator. Default off.
      static const int s_walk_sleep_ms = []{
        const char* v = getenv("PIM_DESC_WALK_SLEEP_MS");
        return (v && *v) ? atoi(v) : 0; }();
      if (s_walk_sleep_ms > 0 && w + 1 < n_bank_walks)
        usleep((useconds_t)s_walk_sleep_ms * 1000);
    }
    if (s_multiwalk) { auto _tdr = clk::now(); platform.drain_stray(300, 4);   // clean the raw-capture tail
      tp_drain += std::chrono::duration_cast<ns_t>(clk::now() - _tdr).count(); tp_drain_calls++; }
    if (!bank_ok) return restore_and_fallback();

    // ---- fused marginal-column repair (exact host substitution) ----
    // The walk body is fused-shaped, so bank_acc[j] at this bank's
    // fused_col_bad columns is the (possibly-wrong) fused popcount. Overwrite
    // with the exact per-bank sum — identical to what the handle path's
    // per-unit fused_repair_pc sums to (design doc §v1 restrictions):
    //   bank_acc[j] = sum_units sum_planes weight * popcount(mask[round][bk][j] & x_bp).
    // The repair set = the static fused_col_bad columns UNION, when defect B
    // armed this bank, every 512-bit beat's word-4/5 columns (j%16 in {4,5}).
    std::vector<int> bad_cols;
    {
      std::set<int> seen;
      if (!b.fused_col_bad.empty()) {
        const uint8_t* fbad = b.fused_col_bad.data();
        for (int j = 0; j < (int)d_out; j++)
          if (fbad[j] && seen.insert(j).second) bad_cols.push_back(j);
      }
      if (b_armed) {
        for (int j = 0; j < (int)d_out; j++)
          if ((j % 16 == 4 || j % 16 == 5) && seen.insert(j).second) bad_cols.push_back(j);
        n_bguard_repairs++;
        fprintf(stderr, "[desc-Bguard] bank %d: DEFECT B armed (%ld armed beats) — exact-"
                "repairing %zu cols (fused_col_bad + word-4/5)\n",
                b.bank_id, b_armed_beats, bad_cols.size());
      }
    }
    if (!bad_cols.empty()) {
      std::vector<int64_t> rep(d_out, 0);
      for (size_t r = 0; r < n_rounds; r++) {
        size_t u = r * (size_t)N + (size_t)bk;
        if (u >= n_units) continue;
        // Eligibility already required every colmask-bank unit to have a mask;
        // a missing one here would mean an unrepairable per-column overwrite
        // (we cannot reconstruct that unit's drain contribution) — bail to the
        // handle path rather than emit a wrong answer.
        if (r >= h.all_round_masks.size() || bk >= (int)h.all_round_masks[r].size() ||
            h.all_round_masks[r][bk].empty()) {
          fprintf(stderr, "[desc-serve] handle=%u bank %d round %zu mask vanished mid-repair "
                  "— falling back\n", handle_id, b.bank_id, r);
          return restore_and_fallback();
        }
        const uint32_t* mask = h.all_round_masks[r][bk].data();
        uint32_t chunk = (uint32_t)(u / 2);
        int sign = (int)(u % 2);
        for (uint32_t bp = 0; bp < K; bp++) {
          uint32_t xb = x_bitplane_all[(size_t)chunk * K + bp];
          if (do_skip && xb == 0) continue;   // popcount(mask & 0) == 0 either way
          int64_t weight = (int64_t)((sign == 0) ? 1 : -1) * (int64_t)bitplane_factor[bp];
          for (int j : bad_cols)
            rep[j] += weight * (int64_t)__builtin_popcount(mask[j] & xb);
        }
      }
      for (int j : bad_cols) bank_acc[j] = rep[j];
    }

    if (verbose)
      fprintf(stderr, "[desc-dbg] bank %d done: bank_acc[0..7]= %lld %lld %lld %lld %lld %lld %lld %lld  repaired=%s\n",
              b.bank_id,
              (long long)bank_acc[0], (long long)bank_acc[1], (long long)bank_acc[2],
              (long long)bank_acc[3], (long long)bank_acc[4], (long long)bank_acc[5],
              (long long)bank_acc[6], (long long)bank_acc[7],
              b_armed ? "yes(defectB word4/5)" : (b.fused_col_bad.empty() ? "no(colmask empty)" : "yes"));
    for (int j = 0; j < (int)d_out; j++) y[j] += (int32_t)bank_acc[j];
  }

  // ---- restore READ mode; leave the engine clean for the next request ----
  platform.set_readback_mode(false); platform.set_readback_mode(false);
  platform.drain_stray(1500, 8);
  g_mode_accxbp_now = false; g_mode_segpop_now = false;
  label_base += N * 1000 + 1000;

  // ---- zskip accounting + liveness assert (mirror the handle path) ----
  if (do_skip) {
    g_zskip_total   += n_zunits_seen;
    g_zskip_skipped += n_zskip_units;
    zskip_report();
  }
  if (n_maj3_execs == 0 && n_rounds > 0 && n_bitplanes > 0) {
    static const bool s_allow = []{ const char* v = getenv("PIM_ALLOW_ZERO_EXEC"); return v && atoi(v) > 0; }();
    const bool zskip_covers = (n_zskip_units > 0);
    fprintf(stderr, "[LIVENESS-ASSERT] desc-serve handle=%u ran ZERO walked descriptors "
            "(n_rounds=%zu n_bitplanes=%u zskip=%ld) — %s%s\n",
            handle_id, (size_t)n_rounds, (unsigned)n_bitplanes, n_zskip_units,
            zskip_covers ? "all bodies were zero-x planes (skip-covered)"
                         : "response would be fabricated zeros",
            (zskip_covers || s_allow) ? " continuing." : " ABORTING.");
    if (!zskip_covers && !s_allow) return -1;
  }

  // ---- respond exactly like process_matmul_handle ----
  if (verbose)
    fprintf(stderr, "[desc-dbg] final y[0..7]= %d %d %d %d %d %d %d %d\n",
            y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7]);
  ssize_t total = (ssize_t)d_out * 4;
  ssize_t written = 0;
  while (written < total) {
    ssize_t wq = write(response_fd, ((char*)y.data()) + written, total - written);
    if (wq <= 0) { fprintf(stderr, "[desc-serve] response write failed: %s\n", strerror(errno)); return -1; }
    written += wq;
  }

  // ---- instrumentation ----
  long long t_total_ns = std::chrono::duration_cast<ns_t>(clk::now() - t_req_start).count();
  static long s_ds_n = 0, s_ds_recv = 0, s_ds_walks = 0, s_ds_desc = 0;
  s_ds_n++; s_ds_recv += n_recv_wakes; s_ds_walks += n_walks; s_ds_desc += n_desc_emitted;
  fprintf(stderr, "[desc-serve #%ld handle=%u] banks=%d units=%zu planes=%u desc=%ld walks=%ld "
          "recv_wakes=%ld zskip=%ld rewrites=%ld(%s) bguard_repairs=%ld total=%.1fms\n",
          s_ds_n, handle_id, N, n_units, (unsigned)K, n_desc_emitted, n_walks,
          n_recv_wakes, n_zskip_units, n_rewrites, rewrite_on ? "on" : "off",
          n_bguard_repairs, t_total_ns / 1e6);
  if (s_prof) {
    const double M = 1e6;                       // ns -> ms
    const double w = (n_walks > 0) ? (double)n_walks : 1.0;
    const double modeset = (tp_enter - tp_resend - tp_usleep) / M;  // enter minus its named subparts
    const double sess = (tp_build + tp_enter + tp_rewrite + tp_drain) / M;  // per-walk MACHINERY
    const double work = (tp_dload + tp_replay + tp_recv) / M;               // irreducible walk work
    fprintf(stderr,
      "[desc-prof #%ld handle=%u] walks=%ld enter_calls=%ld drain_calls=%ld | TOTALS(ms) "
      "build=%.1f enter=%.1f[resend=%.1f usleep=%.1f modeset=%.1f] rewrite=%.1f "
      "dload=%.1f replay=%.1f recv=%.1f drain=%.1f | machinery=%.1f work=%.1f | "
      "PER-WALK(ms) enter=%.2f recv=%.2f replay=%.2f dload=%.2f drain=%.2f rewrite=%.2f build/walk=%.2f\n",
      s_ds_n, handle_id, n_walks, tp_enter_calls, tp_drain_calls,
      tp_build/M, tp_enter/M, tp_resend/M, tp_usleep/M, modeset, tp_rewrite/M,
      tp_dload/M, tp_replay/M, tp_recv/M, tp_drain/M, sess, work,
      tp_enter/M/w, tp_recv/M/w, tp_replay/M/w, tp_dload/M/w, tp_drain/M/w,
      tp_rewrite/M/w, tp_build/M/w);
  }
  if (n_bguard_repairs > 0)
    fprintf(stderr, "[desc-serve #%ld handle=%u] *** DEFECT B (task #59) detected + exact-repaired "
            "on %ld bank(s); output is correct. Fabric ACCUM_XBP word-4/5 pin armed this process — "
            "mig-reinit 2 to clear the fabric latch before the next request. ***\n",
            s_ds_n, handle_id, n_bguard_repairs);
  if (s_ds_n <= 5 || s_ds_n % 50 == 0)
    fprintf(stderr, "[desc-serve-prof] cumulative: reqs=%ld desc=%ld walks=%ld recv_wakes=%ld\n",
            s_ds_n, s_ds_desc, s_ds_walks, s_ds_recv);
  return 0;
}

// ----------------------------------------------------------------------------
// [2026-08-04 CROSS-REQUEST BATCHING — host-only, NO RTL; descent_2026_08_04]
// process_matmul_desc_batch: serve N MAGIC_MM3D sub-requests (a MAGIC_FUSE
// frame of MM3D bodies) through ONE desc-serve super-session per bank:
//   per bank: [rewrite ALL subs' rounds] -> [enter+send resident ONCE] ->
//             per sub: dload+replay+recv walks (WALK_MAX chunks, slots reused
//             per walk so the 256-idx store never binds across subs) ->
//             [one tail drain]  — amortizing the measured session machinery
// (entry/exit drains ~2.6 s/req + per-bank enter + rewrite mode round-trips)
// that is 99.7% of the naive 5.111 s/request model-shape wall.
// ELIGIBILITY = every sub passes the SAME checks as process_matmul_desc; any
// failure returns 2 and the caller falls back to the per-sub dispatch loop
// (desc-serve-then-handle per sub — identical answers). Additive + env-gated:
// PIM_DESC_XBATCH=1 required; default OFF = this function is never called.
// Responses are written per sub in frame order (client reads N x 8192 B).
// ----------------------------------------------------------------------------
static int process_matmul_desc_batch(SoftMCPlatform& platform,
                                     std::vector<BankConfig>& banks,
                                     const std::map<uint32_t, LoadedHandle>& handles,
                                     const std::vector<std::pair<const uint8_t*, size_t>>& subs,
                                     int& label_base, int response_fd) {
  init_debug_flags();
  using clk = std::chrono::steady_clock;
  using ns_t = std::chrono::nanoseconds;
  auto t_req_start = clk::now();
  const int N = (int)banks.size();
  const size_t NS = subs.size();
  if (NS == 0 || NS > 16) return 2;              // sane batch bound (16 model reqs)

  auto inel = [&](const char* why) -> int {
    static std::set<std::string> seen;
    if (seen.insert(std::string(why)).second)
      fprintf(stderr, "[desc-xbatch] INELIGIBLE (%s) — per-sub fallback\n", why);
    return 2;
  };

  // ---- parse + eligibility, every sub (same checks as process_matmul_desc) ----
  struct SubReq {
    uint32_t handle_id = 0, d_out = 0, n_chunks = 0, K = 0;
    const LoadedHandle* h = nullptr;
    const uint32_t* xbp = nullptr;
    const int32_t* factors = nullptr;
  };
  std::vector<SubReq> S(NS);
  const int fm = fused_coset_mode();
  if (!(fm == 1 || fm == 3))
    return inel("PIM_FUSED_COSET not in {1,3}");
  for (int bk = 0; bk < N; bk++)
    if (banks[bk].calib.open_rows.size() < 16)
      return inel("a bank calib has < 16 open_rows");
  for (size_t s = 0; s < NS; s++) {
    const uint8_t* req = subs[s].first; size_t req_len = subs[s].second;
    if (req_len < 5 * 4) return inel("runt sub body");
    uint32_t magic; memcpy(&magic, req, 4);
    if (magic != MAGIC_MM3D) return inel("non-MM3D sub in frame");
    SubReq& q = S[s];
    memcpy(&q.handle_id, req + 4, 4);  memcpy(&q.d_out, req + 8, 4);
    memcpy(&q.n_chunks, req + 12, 4);  memcpy(&q.K, req + 16, 4);
    if (q.d_out != 2048) return inel("d_out != 2048");
    auto it = handles.find(q.handle_id);
    if (it == handles.end()) return inel("unknown handle in batch");
    q.h = &it->second;
    if (q.h->n_chunks != q.n_chunks) return inel("n_chunks mismatch");
    size_t need = 5 * 4 + (size_t)q.n_chunks * q.K * 4 + (size_t)q.K * 4;
    if (req_len < need) return inel("short sub body");
    q.xbp = (const uint32_t*)(req + 20);
    q.factors = (const int32_t*)(req + 20 + (size_t)q.n_chunks * q.K * 4);
    if (q.K == 0 || q.K > 8) return inel("K not in [1,8]");
    for (uint32_t b = 0; b < q.K; b++) {
      int neg, shift;
      if (!accxbp_encode(q.factors[b], &neg, &shift))
        return inel("factor not +/- power of two <= 128");
    }
    if (!q.h->per_round_calib_sel.empty()) {
      for (size_t r = 0; r < q.h->n_rounds && r < q.h->per_round_calib_sel.size(); r++)
        for (int bk = 0; bk < N && bk < (int)q.h->per_round_calib_sel[r].size(); bk++)
          if (q.h->per_round_calib_sel[r][bk] != 0)
            return inel("non-primary calib round");
    }
    // Colmask coverage (exact-repair needs every owned unit's mask).
    bool any_colmask = false;
    for (int bk = 0; bk < N; bk++) if (!banks[bk].fused_col_bad.empty()) any_colmask = true;
    if (any_colmask) {
      if (q.h->all_round_masks.size() < q.h->n_rounds)
        return inel("colmask bank but masks not kept");
      for (size_t r = 0; r < q.h->n_rounds; r++)
        for (int bk = 0; bk < N; bk++) {
          size_t u = r * (size_t)N + (size_t)bk;
          if (u >= q.h->n_units || banks[bk].fused_col_bad.empty()) continue;
          if (r >= q.h->all_round_masks.size() || bk >= (int)q.h->all_round_masks[r].size() ||
              q.h->all_round_masks[r][bk].empty())
            return inel("colmask bank missing a unit's mask");
        }
    }
  }
  // Pool-wrap within each sub + backup-row aliasing ACROSS subs (a shared row
  // would make one sub's rewrite clobber another's weights mid-session).
  for (int bk = 0; bk < N; bk++) {
    std::map<uint32_t, std::pair<size_t,size_t>> row_owner;   // row -> (sub, round)
    for (size_t s = 0; s < NS; s++) {
      const LoadedHandle& h = *S[s].h;
      for (size_t r = 0; r < h.n_rounds; r++) {
        size_t u = r * (size_t)N + (size_t)bk;
        if (u >= h.n_units) continue;
        if (r >= h.per_round_backup_rows.size() ||
            bk >= (int)h.per_round_backup_rows[r].size()) continue;
        uint32_t row = h.per_round_backup_rows[r][bk];
        auto ins = row_owner.emplace(row, std::make_pair(s, r));
        if (!ins.second) return inel("backup-row aliasing (within or across subs)");
      }
    }
  }
  // Image probe (shares the desc-probe class; own lazy static).
  static int s_probe = -1;
  if (s_probe < 0) s_probe = desc_probe_image(platform, banks) ? 1 : 0;
  if (s_probe == 0) return 2;

  // ---- session baseline (identical to process_matmul_desc entry) ----
  const bool s_sessreuse = sessreuse_on();
  if (s_sessreuse) {
    static bool once = false;
    if (!once) { once = true;
      fprintf(stderr, "[sessreuse] ON: dropping entry/exit/per-bank tail "
              "drain_stray (b50 one-ACK + xsync boundary contract); READ "
              "re-establish + per-bank rewrite mode-cycle KEPT (retention).\n"); }
  }
  if (g_stream_session) { platform.stream_stop(); g_stream_session = false;
    platform.drain_stray(800, 6); }
  platform.set_readback_mode(false); platform.set_readback_mode(false);
  if (!s_sessreuse) platform.drain_stray(1500, 8);   // entry_drain (dead: DESC_PROF #2)
  g_mode_accxbp_now = false; g_mode_segpop_now = false;
  auto restore_and_fallback = [&]() -> int {
    platform.set_readback_mode(false); platform.set_readback_mode(false);
    platform.drain_stray(1500, 8);
    g_mode_accxbp_now = false; g_mode_segpop_now = false;
    return 2;
  };

  const bool do_skip = skip_zero_planes_on();
  const bool rewrite_on = desc_rewrite_on();
  static const bool s_bguard = []{
    const char* v = getenv("PIM_DESC_B_GUARD"); return !(v && atoi(v) == 0); }();
  static const int B_GUARD_BEATS = []{
    const char* v = getenv("PIM_DESC_B_GUARD_BEATS"); return (v && *v) ? atoi(v) : 16; }();
  static const int WALK_MAX = []{
    const char* v = getenv("PIM_DESC_WALK_MAX");
    long n = (v && *v) ? atol(v) : 256;
    if (n < 1) n = 1; if (n > 256) n = 256;
    return (int)n;
  }();
  long n_desc_emitted = 0, n_zskip_units = 0, n_zunits_seen = 0;
  long n_walks = 0, n_rewrites = 0, n_bguard_repairs = 0;

  std::vector<std::vector<int32_t>> y(NS, std::vector<int32_t>(2048, 0));

  // ===================== BANK-GENERIC single-send path (bankgen) =====================
  // Collapse the 4 per-bank replay_send_resident calls (=> 4 raw-capture receiver
  // 500 ms quiet-drain floors, ~98% of the corrected model's residual C) to ONE
  // bank-generic send + per-descriptor bank patch. Two passes because the retention
  // rewrite (per_column_write_row -> pexec) CLOBBERS the resident IMEM, so the single
  // send must follow every bank's rewrite: pass 1 rewrites all banks (READ mode),
  // then ONE enter+send (ACCXBP), then pass 2 walks all banks (no IMEM-clobbering op
  // between the send and the walks). Gated (PIM_DESC_BANKGEN) + invariance-guarded:
  // any geometry divergence or IMEM-oversize leaves bankgen_active=false and falls
  // through to the per-bank loop below (identical answers, no fudge).
  const bool s_bankgen = bankgen_on();
  bool bankgen_active = false;
  if (s_bankgen && build_bankgen_invariant(banks, label_base)) {
    // Pass 0: enumerate every bank's descriptors (counters folded once here).
    std::vector<std::vector<std::vector<DsDesc>>> BSD(
        N, std::vector<std::vector<DsDesc>>(NS));
    std::vector<size_t> bank_total(N, 0);
    for (int bk = 0; bk < N; bk++) {
      for (size_t s = 0; s < NS; s++) {
        const LoadedHandle& h = *S[s].h;
        for (size_t r = 0; r < h.n_rounds; r++) {
          size_t u = r * (size_t)N + (size_t)bk;
          if (u >= h.n_units) continue;
          uint32_t chunk = (uint32_t)(u / 2);
          int sign = (int)(u % 2);
          uint32_t row = h.per_round_backup_rows[r][bk];
          for (uint32_t bp = 0; bp < S[s].K; bp++) {
            uint32_t xb = S[s].xbp[(size_t)chunk * S[s].K + bp];
            n_zunits_seen++;
            if (do_skip && xb == 0) { n_zskip_units++; continue; }
            uint32_t acc = desc_acc_for(sign, S[s].factors[bp]);
            BSD[bk][s].push_back({ xb, row, acc, (uint32_t)r });
          }
        }
        n_desc_emitted += (long)BSD[bk][s].size();
      }
      for (size_t s = 0; s < NS; s++) bank_total[bk] += BSD[bk][s].size();
    }

    // THE single bank-generic resident: bank[0] geometry (proven invariant above).
    // The walker overrides its LI(bank_id, BAR) per descriptor via the bank patch;
    // every other operand is bank-invariant. One label_base (internal jump targets).
    const BankConfig& b0 = banks[0];
    Program resident = build_desc_walk_resident(
        b0.bank_id, b0.calib.Rfirst, b0.calib.Rsecond, b0.calib.open_rows.data(),
        label_base, b0.res_one_row, b0.res_zero_row, desc_use_consts(b0));
    if (resident.size() / 8 > g_bitstream_imem) {
      fprintf(stderr, "[bankgen] canonical resident %d insts > IMEM %d — fallback\n",
              (int)(resident.size() / 8), g_bitstream_imem);
      return restore_and_fallback();
    }

    // ---- PASS 1: rewrite ALL banks' rounds (READ mode) BEFORE the single send.
    //      Clobbers IMEM freely (no resident live yet). Retention: the first-
    //      written bank is read ~one frame (~0.5 s) later in pass 2 — verified by
    //      the 512/512 battery. ----
    if (rewrite_on) {
      platform.set_readback_mode(false); platform.set_readback_mode(false);
      g_mode_accxbp_now = false; g_mode_segpop_now = false;
      for (int bk = 0; bk < N; bk++) {
        if (bank_total[bk] == 0) continue;
        const BankConfig& b = banks[bk];
        for (size_t s = 0; s < NS; s++) {
          const LoadedHandle& h = *S[s].h;
          if (h.all_round_masks.empty()) continue;
          std::set<uint32_t> done_rounds;
          for (size_t k = 0; k < BSD[bk][s].size(); k++) {
            uint32_t r = BSD[bk][s][k].round;
            if (!done_rounds.insert(r).second) continue;
            if (r >= h.all_round_masks.size() || bk >= (int)h.all_round_masks[r].size() ||
                h.all_round_masks[r][bk].empty()) continue;
            per_column_write_row(platform, b.bank_id, BSD[bk][s][k].row,
                                 h.all_round_masks[r][bk].data());
            n_rewrites++;
          }
        }
      }
    }

    // ---- ONE session: enter ACCXBP, send the bank-generic resident ONCE (the
    //      single receiver floor), clear the one auto-run deposit. ----
    platform.set_readback_mode_accxbp();
    platform.set_readback_mode_accxbp();
    g_mode_accxbp_now = true; g_mode_segpop_now = false;
    long osk0 = platform.oversize_skips();
    platform.replay_send_resident(resident);
    if (platform.oversize_skips() != osk0) {
      fprintf(stderr, "[bankgen] canonical resident oversize-skipped — fallback\n");
      return restore_and_fallback();
    }
    usleep(5000);
    platform.set_readback_mode_accxbp();          // clear the auto-run deposit

    // ---- PASS 2: walk every bank out of the single resident, bank patched per
    //      descriptor. session_walk is FRAME-GLOBAL (one auto-run for the whole
    //      frame => only the very first walk folds the +4 ack; the rest +3). ----
    long session_walk = 0;
    for (int bk = 0; bk < N; bk++) {
      if (bank_total[bk] == 0) continue;
      const BankConfig& b = banks[bk];
      bool b_armed = false; long b_armed_beats = 0;
      std::vector<std::vector<int64_t>> bank_acc(NS, std::vector<int64_t>(2048, 0));
      bool bank_ok = true;
      for (size_t s = 0; s < NS && bank_ok; s++) {
        const std::vector<DsDesc>& descs = BSD[bk][s];
        const size_t n_sub_walks = (descs.size() + (size_t)WALK_MAX - 1) / (size_t)WALK_MAX;
        for (size_t w = 0; w < n_sub_walks && bank_ok; w++) {
          const size_t base = w * (size_t)WALK_MAX;
          const size_t nd   = std::min((size_t)WALK_MAX, descs.size() - base);
          for (size_t k = 0; k < nd; k++) {
            const DsDesc& d = descs[base + k];
            // build-32 per-descriptor bank patch: bank=b.bank_id, patch_en=true
            // => walker splices bank[1:0] into the resident's LI(bank, BAR).
            platform.dload((uint32_t)k, d.x, d.row, d.acc, (uint32_t)b.bank_id, true);
          }
          platform.set_readback_mode_accxbp();      // clear right before REPLAY_N
          platform.replay_n((uint16_t)nd);
          const int extra = (session_walk == 0) ? 4 : 3;
          const int want  = 8192 + (int)((nd + (size_t)extra) * 32);
          const int SLACK_RECORDS = 16;
          std::vector<uint8_t> buf(want + SLACK_RECORDS * 32, 0);
          RsParse r; int rb = 0;
          if (session_deframe_on()) {
            SessTrailer st;
            int rc_f = framed_walk_receive(platform, nd, r, st, &rb);
            if (rc_f == -1) {
              fprintf(stderr, "[bankgen] bank %d sub %zu walk %zu FRAMED receive "
                      "STALLED (%d)\n", b.bank_id, s, w, rb);
              return -1;
            }
          } else {
            rb = platform.receiveDataTry(buf.data(), want, 8000);
            if (platform.recv_stalled()) {
              fprintf(stderr, "[bankgen] bank %d sub %zu walk %zu receive STALLED (%d/%d)\n",
                      b.bank_id, s, w, rb, want);
              return -1;
            }
            r = rs_parse(buf.data(), rb);
            int topups = 0;
            while (r.n_drain == 0 && rb >= want && topups < SLACK_RECORDS) {
              int more = platform.receiveDataTry(buf.data() + rb, 32, 1000);
              if (platform.recv_stalled()) {
                fprintf(stderr, "[bankgen] bank %d sub %zu walk %zu top-up STALLED\n",
                        b.bank_id, s, w);
                return -1;
              }
              if (more < 32) break;
              rb += more; topups++;
              r = rs_parse(buf.data(), rb);
            }
          }
          if (r.n_drain != 1 || (int)r.lanes.size() != 2048 ||
              (r.drain_magic != 0 && r.drain_magic < 0xDBC0DE1Bu)) {
            fprintf(stderr, "[bankgen] bank %d sub %zu walk %zu FRAMING ANOMALY "
                    "(n_drain=%ld lanes=%zu magic=0x%08x recv=%d want=%d) — falling back\n",
                    b.bank_id, s, w, r.n_drain, r.lanes.size(), r.drain_magic, rb, want);
            bank_ok = false; break;
          }
          if (s_bguard) {
            long eqp = 0;
            for (int k = 0; k + 5 < (int)r.lanes.size(); k += 16)
              if (r.lanes[k + 4] == r.lanes[k + 5] && r.lanes[k + 4] != 0) eqp++;
            if (eqp >= B_GUARD_BEATS) { b_armed = true; b_armed_beats += eqp; }
          }
          for (int j = 0; j < 2048; j++) bank_acc[s][j] += r.lanes[j];
          n_walks++; session_walk++;
        }
      }
      if (!bank_ok) return restore_and_fallback();

      // ---- per-sub repair (fused_col_bad U defect-B cols), then fold into y ----
      for (size_t s = 0; s < NS; s++) {
        const LoadedHandle& h = *S[s].h;
        std::vector<int> bad_cols;
        {
          std::set<int> seen2;
          if (!b.fused_col_bad.empty()) {
            const uint8_t* fbad = b.fused_col_bad.data();
            for (int j = 0; j < 2048; j++)
              if (fbad[j] && seen2.insert(j).second) bad_cols.push_back(j);
          }
          if (b_armed)
            for (int j = 0; j < 2048; j++)
              if ((j % 16 == 4 || j % 16 == 5) && seen2.insert(j).second) bad_cols.push_back(j);
        }
        if (b_armed && s == 0) {
          n_bguard_repairs++;
          fprintf(stderr, "[bankgen] bank %d: DEFECT B armed (%ld beats) — exact-repairing "
                  "%zu cols for ALL %zu subs\n", b.bank_id, b_armed_beats, bad_cols.size(), NS);
        }
        if (!bad_cols.empty()) {
          std::vector<int64_t> rep(2048, 0);
          for (size_t r = 0; r < h.n_rounds; r++) {
            size_t u = r * (size_t)N + (size_t)bk;
            if (u >= h.n_units) continue;
            if (r >= h.all_round_masks.size() || bk >= (int)h.all_round_masks[r].size() ||
                h.all_round_masks[r][bk].empty()) {
              fprintf(stderr, "[bankgen] sub %zu bank %d round %zu mask vanished mid-repair "
                      "— falling back\n", s, b.bank_id, r);
              return restore_and_fallback();
            }
            const uint32_t* mask = h.all_round_masks[r][bk].data();
            uint32_t chunk = (uint32_t)(u / 2);
            int sign = (int)(u % 2);
            for (uint32_t bp = 0; bp < S[s].K; bp++) {
              uint32_t xb = S[s].xbp[(size_t)chunk * S[s].K + bp];
              if (do_skip && xb == 0) continue;
              int64_t weight = (int64_t)((sign == 0) ? 1 : -1) * (int64_t)S[s].factors[bp];
              for (int j : bad_cols)
                rep[j] += weight * (int64_t)__builtin_popcount(mask[j] & xb);
            }
          }
          for (int j : bad_cols) bank_acc[s][j] = rep[j];
        }
        for (int j = 0; j < 2048; j++) y[s][j] += (int32_t)bank_acc[s][j];
      }
    }
    // [persist] BOUNDARY-TERMINATED CAPTURE. Every walk above was consumed by
    // framed_walk_receive's ORDTOL loop THROUGH its closing correct-ACK
    // (session_walk_close_len(...) != 0), and the b50 framer emits exactly ONE
    // such ACK per session — so the shared c2h pipe is now provably EMPTY (no
    // stray beat for the raw-capture receiver's 500 ms quiet-drain to absorb).
    // Signal the resident receiver to exit on its next tick and bounded-join it
    // now, instead of letting the next frame's first rewrite execute() pay the
    // full quiet-drain floor when it joins this receiver. Gated: default OFF =>
    // never called => byte-identical to bankgen.
    if (persist_on()) {
      platform.receiver_boundary_stop();
      static bool once_p = false;
      if (!once_p) { once_p = true;
        fprintf(stderr, "[persist] ON: boundary-terminated the resident "
                "receiver after PASS 2 (b50 one-ACK contract => c2h empty); "
                "500 ms quiet-drain floor bypassed for the frame.\n"); }
    }
    fprintf(stderr, "[bankgen] ON: 1 bank-generic resident served %d banks "
            "(was %d per-bank sends); invariance PASS, bank_patch armed, "
            "session_walk=%ld\n", N, N, session_walk);
    bankgen_active = true;
  } else if (s_bankgen) {
    fprintf(stderr, "[bankgen] invariance guard FAILED — per-bank send fallback "
            "(sessreuse behaviour, still correct)\n");
  }

  if (!bankgen_active)
  for (int bk = 0; bk < N; bk++) {
    const BankConfig& b = banks[bk];
    // Enumerate every sub's descriptors for THIS bank (identical enumeration).
    std::vector<std::vector<DsDesc>> sub_descs(NS);
    for (size_t s = 0; s < NS; s++) {
      const LoadedHandle& h = *S[s].h;
      for (size_t r = 0; r < h.n_rounds; r++) {
        size_t u = r * (size_t)N + (size_t)bk;
        if (u >= h.n_units) continue;
        uint32_t chunk = (uint32_t)(u / 2);
        int sign = (int)(u % 2);
        uint32_t row = h.per_round_backup_rows[r][bk];
        for (uint32_t bp = 0; bp < S[s].K; bp++) {
          uint32_t xb = S[s].xbp[(size_t)chunk * S[s].K + bp];
          n_zunits_seen++;
          if (do_skip && xb == 0) { n_zskip_units++; continue; }
          uint32_t acc = desc_acc_for(sign, S[s].factors[bp]);
          sub_descs[s].push_back({ xb, row, acc, (uint32_t)r });
        }
      }
      n_desc_emitted += (long)sub_descs[s].size();
    }
    size_t bank_total = 0;
    for (size_t s = 0; s < NS; s++) bank_total += sub_descs[s].size();
    if (bank_total == 0) continue;               // all zero-x on this bank

    const bool uc = desc_use_consts(b);
    Program resident = build_desc_walk_resident(
        b.bank_id, b.calib.Rfirst, b.calib.Rsecond, b.calib.open_rows.data(),
        label_base + bk * 1000, b.res_one_row, b.res_zero_row, uc);
    if (resident.size() / 8 > g_bitstream_imem) {
      fprintf(stderr, "[desc-xbatch] bank %d resident %d insts > IMEM %d — fallback\n",
              b.bank_id, (int)(resident.size() / 8), g_bitstream_imem);
      return restore_and_fallback();
    }

    // ---- bank-head rewrite: ALL subs' distinct rounds (READ mode) ----
    if (rewrite_on) {
      platform.set_readback_mode(false); platform.set_readback_mode(false);
      g_mode_accxbp_now = false; g_mode_segpop_now = false;
      for (size_t s = 0; s < NS; s++) {
        const LoadedHandle& h = *S[s].h;
        if (h.all_round_masks.empty()) continue;
        std::set<uint32_t> done_rounds;
        for (size_t k = 0; k < sub_descs[s].size(); k++) {
          uint32_t r = sub_descs[s][k].round;
          if (!done_rounds.insert(r).second) continue;
          if (r >= h.all_round_masks.size() || bk >= (int)h.all_round_masks[r].size() ||
              h.all_round_masks[r][bk].empty()) continue;
          per_column_write_row(platform, b.bank_id, sub_descs[s][k].row,
                               h.all_round_masks[r][bk].data());
          n_rewrites++;
        }
      }
    }

    // ---- ONE session for the whole bank (enter + resident + auto-run clear) ----
    platform.set_readback_mode_accxbp();
    platform.set_readback_mode_accxbp();
    g_mode_accxbp_now = true; g_mode_segpop_now = false;
    long osk0 = platform.oversize_skips();
    platform.replay_send_resident(resident);
    if (platform.oversize_skips() != osk0) {
      fprintf(stderr, "[desc-xbatch] bank %d resident oversize-skipped — fallback\n", b.bank_id);
      return restore_and_fallback();
    }
    usleep(5000);
    platform.set_readback_mode_accxbp();          // clear the auto-run deposit

    long session_walk = 0;
    bool b_armed = false; long b_armed_beats = 0;
    std::vector<std::vector<int64_t>> bank_acc(NS, std::vector<int64_t>(2048, 0));
    bool bank_ok = true;
    for (size_t s = 0; s < NS && bank_ok; s++) {
      const std::vector<DsDesc>& descs = sub_descs[s];
      const size_t n_sub_walks = (descs.size() + (size_t)WALK_MAX - 1) / (size_t)WALK_MAX;
      for (size_t w = 0; w < n_sub_walks && bank_ok; w++) {
        const size_t base = w * (size_t)WALK_MAX;
        const size_t nd   = std::min((size_t)WALK_MAX, descs.size() - base);
        for (size_t k = 0; k < nd; k++) {
          const DsDesc& d = descs[base + k];
          platform.dload((uint32_t)k, d.x, d.row, d.acc);   // slots reused per walk
        }
        platform.set_readback_mode_accxbp();      // clear right before REPLAY_N
        platform.replay_n((uint16_t)nd);
        // Framing: the SESSION's first walk carries the send's auto-run ack
        // (+4 records); every later walk in the shared session carries +3
        // (identical to the multiwalk framing law).
        const int extra = (session_walk == 0) ? 4 : 3;
        const int want  = 8192 + (int)((nd + (size_t)extra) * 32);
        const int SLACK_RECORDS = 16;
        std::vector<uint8_t> buf(want + SLACK_RECORDS * 32, 0);
        RsParse r; int rb = 0;
        if (session_deframe_on()) {
          // [b49] framed session receive (content-based, per CONTRACT §2);
          // same integration as process_matmul_desc.
          SessTrailer st;
          int rc_f = framed_walk_receive(platform, nd, r, st, &rb);
          if (rc_f == -1) {
            fprintf(stderr, "[desc-xbatch] bank %d sub %zu walk %zu FRAMED receive "
                    "STALLED (%d)\n", b.bank_id, s, w, rb);
            return -1;
          }
        } else {
          rb = platform.receiveDataTry(buf.data(), want, 8000);
          if (platform.recv_stalled()) {
            fprintf(stderr, "[desc-xbatch] bank %d sub %zu walk %zu receive STALLED (%d/%d)\n",
                    b.bank_id, s, w, rb, want);
            return -1;
          }
          r = rs_parse(buf.data(), rb);
          int topups = 0;
          while (r.n_drain == 0 && rb >= want && topups < SLACK_RECORDS) {
            int more = platform.receiveDataTry(buf.data() + rb, 32, 1000);
            if (platform.recv_stalled()) {
              fprintf(stderr, "[desc-xbatch] bank %d sub %zu walk %zu top-up STALLED\n",
                      b.bank_id, s, w);
              return -1;
            }
            if (more < 32) break;
            rb += more; topups++;
            r = rs_parse(buf.data(), rb);
          }
        }
        if (r.n_drain != 1 || (int)r.lanes.size() != 2048 ||
            (r.drain_magic != 0 && r.drain_magic < 0xDBC0DE1Bu)) {
          fprintf(stderr, "[desc-xbatch] bank %d sub %zu walk %zu FRAMING ANOMALY "
                  "(n_drain=%ld lanes=%zu magic=0x%08x recv=%d want=%d) — falling back\n",
                  b.bank_id, s, w, r.n_drain, r.lanes.size(), r.drain_magic, rb, want);
          bank_ok = false; break;
        }
        if (s_bguard) {
          long eqp = 0;
          for (int k = 0; k + 5 < (int)r.lanes.size(); k += 16)
            if (r.lanes[k + 4] == r.lanes[k + 5] && r.lanes[k + 4] != 0) eqp++;
          if (eqp >= B_GUARD_BEATS) { b_armed = true; b_armed_beats += eqp; }
        }
        for (int j = 0; j < 2048; j++) bank_acc[s][j] += r.lanes[j];
        n_walks++; session_walk++;
      }
    }
    if (!s_sessreuse)
      platform.drain_stray(300, 4);               // per-bank tail drain (dead: DESC_PROF #3)
    if (!bank_ok) return restore_and_fallback();

    // ---- per-sub repair (fused_col_bad U defect-B cols), then fold into y ----
    for (size_t s = 0; s < NS; s++) {
      const LoadedHandle& h = *S[s].h;
      std::vector<int> bad_cols;
      {
        std::set<int> seen2;
        if (!b.fused_col_bad.empty()) {
          const uint8_t* fbad = b.fused_col_bad.data();
          for (int j = 0; j < 2048; j++)
            if (fbad[j] && seen2.insert(j).second) bad_cols.push_back(j);
        }
        if (b_armed)
          for (int j = 0; j < 2048; j++)
            if ((j % 16 == 4 || j % 16 == 5) && seen2.insert(j).second) bad_cols.push_back(j);
      }
      if (b_armed && s == 0) {
        n_bguard_repairs++;
        fprintf(stderr, "[desc-xbatch] bank %d: DEFECT B armed (%ld beats) — exact-repairing "
                "%zu cols for ALL %zu subs\n", b.bank_id, b_armed_beats, bad_cols.size(), NS);
      }
      if (!bad_cols.empty()) {
        std::vector<int64_t> rep(2048, 0);
        for (size_t r = 0; r < h.n_rounds; r++) {
          size_t u = r * (size_t)N + (size_t)bk;
          if (u >= h.n_units) continue;
          if (r >= h.all_round_masks.size() || bk >= (int)h.all_round_masks[r].size() ||
              h.all_round_masks[r][bk].empty()) {
            fprintf(stderr, "[desc-xbatch] sub %zu bank %d round %zu mask vanished mid-repair "
                    "— falling back\n", s, b.bank_id, r);
            return restore_and_fallback();
          }
          const uint32_t* mask = h.all_round_masks[r][bk].data();
          uint32_t chunk = (uint32_t)(u / 2);
          int sign = (int)(u % 2);
          for (uint32_t bp = 0; bp < S[s].K; bp++) {
            uint32_t xb = S[s].xbp[(size_t)chunk * S[s].K + bp];
            if (do_skip && xb == 0) continue;
            int64_t weight = (int64_t)((sign == 0) ? 1 : -1) * (int64_t)S[s].factors[bp];
            for (int j : bad_cols)
              rep[j] += weight * (int64_t)__builtin_popcount(mask[j] & xb);
          }
        }
        for (int j : bad_cols) bank_acc[s][j] = rep[j];
      }
      for (int j = 0; j < 2048; j++) y[s][j] += (int32_t)bank_acc[s][j];
    }
  }

  // ---- restore READ; clean exit (once per BATCH — the amortized term) ----
  platform.set_readback_mode(false); platform.set_readback_mode(false);
  if (!s_sessreuse) platform.drain_stray(1500, 8);   // exit_drain (dead: DESC_PROF #1)
  g_mode_accxbp_now = false; g_mode_segpop_now = false;
  label_base += N * 1000 + 1000;

  if (do_skip) { g_zskip_total += n_zunits_seen; g_zskip_skipped += n_zskip_units;
    zskip_report(); }
  if (n_walks == 0 && n_zskip_units == 0) {
    fprintf(stderr, "[LIVENESS-ASSERT] desc-xbatch ran ZERO walks with zero zskip "
            "— response would be fabricated zeros. ABORTING.\n");
    return -1;
  }

  // ---- respond: N x 8192 B, frame order ----
  for (size_t s = 0; s < NS; s++) {
    ssize_t total = 2048 * 4, written = 0;
    while (written < total) {
      ssize_t wq = write(response_fd, ((char*)y[s].data()) + written, total - written);
      if (wq <= 0) { fprintf(stderr, "[desc-xbatch] response write failed: %s\n",
                             strerror(errno)); return -1; }
      written += wq;
    }
  }

  long long t_total_ns = std::chrono::duration_cast<ns_t>(clk::now() - t_req_start).count();
  static long s_xb_n = 0;
  s_xb_n++;
  fprintf(stderr, "[desc-xbatch #%ld] subs=%zu banks=%d desc=%ld walks=%ld rewrites=%ld(%s) "
          "zskip=%ld bguard_repairs=%ld total=%.1fms (%.1fms/sub)\n",
          s_xb_n, NS, N, n_desc_emitted, n_walks, n_rewrites, rewrite_on ? "on" : "off",
          n_zskip_units, n_bguard_repairs, t_total_ns / 1e6, t_total_ns / 1e6 / (double)NS);
  return 0;
}

// ----------------------------------------------------------------------------
// desc-plan: standalone dry-run of the scheduling math + static discipline
// check. NO platform init, NO card. Usage:
//   bitnet-proj-server desc-plan <d_in> <d_out> <n_chunks> <K> [N_banks=4]
// ----------------------------------------------------------------------------
static int run_desc_plan(int argc, char** argv) {
  if (argc < 6) {
    fprintf(stderr, "Usage: %s desc-plan <d_in> <d_out> <n_chunks> <K> [N_banks=4]\n"
                    "  Dry-run of the descriptor schedule + static register-discipline\n"
                    "  check. No platform init (no card).\n", argv[0]);
    return 1;
  }
  setvbuf(stdout, NULL, _IONBF, 0);
  long d_in     = atol(argv[2]);
  long d_out    = atol(argv[3]);
  long n_chunks = atol(argv[4]);
  long K        = atol(argv[5]);
  int  N        = (argc > 6) ? atoi(argv[6]) : 4;
  if (N < 1) N = 1;
  const long WALK_MAX = 256;
  const long n_units  = n_chunks * 2;
  const long n_rounds = (n_units + N - 1) / N;

  printf("[desc-plan] shape: d_in=%ld d_out=%ld n_chunks=%ld K(=n_bitplanes)=%ld N_banks=%d\n",
         d_in, d_out, n_chunks, K, N);
  printf("[desc-plan] n_units=%ld n_rounds=%ld   (u=round*N+bk, chunk=u/2, sign=u%%2)\n",
         n_units, n_rounds);
  printf("[desc-plan] eligibility: K<=8 %s ; d_out==2048 %s\n",
         (K >= 1 && K <= 8) ? "OK" : "FAIL", (d_out == 2048) ? "OK" : "FAIL");

  long tot_desc = 0, tot_walks = 0, tot_recv = 0;
  printf("[desc-plan] per-bank schedule (dense, no zero-x skip):\n");
  printf("            bank | units | descriptors(units*K) | walks(ceil/256) | recv_wakes\n");
  for (int bk = 0; bk < N; bk++) {
    long units = 0;
    for (long r = 0; r < n_rounds; r++) if (r * N + bk < n_units) units++;
    long desc  = units * K;
    long walks = (desc == 0) ? 0 : (desc + WALK_MAX - 1) / WALK_MAX;
    long recv  = walks;   // one receive per walk (walk 0 folds the auto-run ack)
    tot_desc += desc; tot_walks += walks; tot_recv += recv;
    printf("            %4d | %5ld | %20ld | %15ld | %10ld\n", bk, units, desc, walks, recv);
  }
  long handle_recv_k1 = n_rounds * K;   // handle path, g_req_K=1: one exec/plane/round
  printf("[desc-plan] TOTAL desc=%ld walks=%ld  desc-serve recv_wakes=%ld\n",
         tot_desc, tot_walks, tot_recv);
  printf("[desc-plan] handle-path recv_wakes = n_rounds*K = %ld*%ld = %ld  (g_req_K=1, per-plane)\n",
         n_rounds, K, handle_recv_k1);
  printf("[desc-plan] recv-wake reduction: %ld -> %ld  (%.1fx fewer host recv wakes)\n",
         handle_recv_k1, tot_recv, tot_recv ? (double)handle_recv_k1 / (double)tot_recv : 0.0);

  printf("[desc-plan] acc nibble = (weight_neg<<3)|shift for factors {1,2,..,128, -1,..,-128}:\n");
  const int32_t facs[] = {1,2,4,8,16,32,64,128, -1,-2,-4,-8,-16,-32,-64,-128};
  for (int32_t f : facs) {
    int neg, shift;
    if (!accxbp_encode(f, &neg, &shift)) { printf("            factor %+5d : NOT ENCODABLE\n", f); continue; }
    uint32_t acc_pos = ds_acc(0 ^ neg, shift);   // positive unit (sign=0)
    uint32_t acc_neg = ds_acc(1 ^ neg, shift);   // negative unit (sign=1)
    printf("            factor %+5d : |f|=2^%d fneg=%d  sign+ -> acc=0x%X (wsum %+ld)   "
           "sign- -> acc=0x%X (wsum %+ld)\n",
           f, shift, neg, acc_pos, ds_wsum(acc_pos), acc_neg, ds_wsum(acc_neg));
  }

  printf("[desc-plan] static register-discipline check (serve-time resident, dummy rows):\n");
  uint32_t open_rows[16];
  for (int i = 0; i < 16; i++) open_rows[i] = 60040u + (uint32_t)i;
  bool all_ok = true;
  for (int uc = 0; uc <= 1; uc++) {
    Program p = build_desc_walk_resident(/*bank_id=*/1, /*Rfirst=*/60020u, /*Rsecond=*/60021u,
                                         open_rows, /*label_base=*/5000,
                                         /*res_one=*/60000u, /*res_zero=*/60001u, uc != 0);
    int n_inst = 0, li6 = 0, li5 = 0, act6 = 0;
    bool ok = desc_check_discipline(p, n_inst, li6, li5, act6);
    all_ok = all_ok && ok;
    printf("            use_consts=%d : insts=%d (<=8192? %s)  LI RT==6:%d(exp1) RT==5:%d(exp2) "
           "ACT-reads-reg6:%d(exp1) -> %s\n",
           uc, n_inst, n_inst <= 8192 ? "yes" : "NO", li6, li5, act6, ok ? "PASS" : "FAIL");
  }
  printf("[desc-plan] VERDICT: %s\n", all_ok ? "DISCIPLINE PASS" : "DISCIPLINE FAIL");
  return all_ok ? 0 : 1;
}

// ============================================================================
// recorder_read_counters — shared, out-of-band 64-byte black-box read.
// Sends a zero-payload resident read + RECORDER_DUMP (h2c-only), then does a
// bounded, NON-poisoning receive scanning for the build-35 v2 record magic
// (0xDBC0DEC1) and pulls the corruption-skip counters out of word2 plus the
// watchdog byte (word3[7:0]) and the build_tag echo (word15). This is the same
// read the swap-storm sampler and run_recorder_dump do, extracted so the
// per-session PIM_COUNTER_GATE (below, in main) can reuse it at teardown ONLY
// (never in the timed request path). Returns false if the 64-byte record never
// arrived within the budget (c2h wedged).
// ============================================================================
static bool recorder_read_counters(SoftMCPlatform& pf, uint32_t& malf,
                                    uint32_t& par, uint32_t& wdog,
                                    uint32_t& btag, long budget_ms = 6000) {
  Program tr = cs_zero_read_prog();
  pf.replay_send_resident(tr);
  pf.recorder_dump();
  const int CAP = 8192 + 64 * 32;
  std::vector<uint8_t> buf(CAP, 0);
  int rb = 0; long found = -1;
  auto s0 = std::chrono::steady_clock::now();
  while (rb + 64 <= CAP) {
    for (int off = 0; off + 4 <= rb; off += 4) {
      uint32_t wv; memcpy(&wv, buf.data() + off, 4);
      if (wv == 0xDBC0DEC1u) { found = off; break; }
    }
    if (found >= 0 && found + 64 <= rb) break;
    long el = std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::steady_clock::now() - s0).count();
    if (el >= budget_ms) break;
    int more = pf.receiveDataTry(buf.data() + rb, std::min(4096, CAP - rb), 1000);
    if (pf.recv_stalled()) break;
    if (more == 0) { if (found >= 0 && found + 64 <= rb) break; else continue; }
    rb += more;
  }
  if (found < 0 || found + 64 > rb) return false;
  uint32_t w[16]; memcpy(w, buf.data() + found, 64);
  malf = w[2] >> 16; par = w[2] & 0xFFFF; wdog = w[3] & 0xFF; btag = w[15];
  return true;
}

// ============================================================================
// recorder-dump: standalone BLACK-BOX post-mortem readout (build-34).
//   Usage:  bitnet-proj-server recorder-dump <bender>
// The post-mortem tool for a MULTI-WALK wedge. Minimal + robust by design:
//  * init the platform (tolerate a corrupted card — the whole point is to read
//    state out AFTER a wedge; recorder_dump() is h2c-only and does NOT check
//    the poison flag);
//  * DO NOT reset_fpga / full_reset / toggle modes — a reset WIPES the black
//    box (RUN_AFTER_FLASH.md step 1);
//  * arm a raw-capture c2h reader over a trivial read (its payload is
//    irrelevant — we scan for the magic), send RECORDER_DUMP, then receive the
//    the 64-byte 0xDBC0DEC1+0xDBC0DEC2 v2 dump with the bounded, NON-poisoning receiveDataTry;
//  * decode + print every field per build-34 NOTES.md / RUN_AFTER_FLASH.md; exit 0.
// No calib, no production rows, no modes beyond arming the reader.
// ============================================================================
static int run_recorder_dump(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "Usage: %s recorder-dump <bender>\n"
                    "  build-35 black-box post-mortem v2 (records 0xDBC0DEC1 + 0xDBC0DEC2, 64 B).\n"
                    "  Does NOT reset the card — a reset destroys the recorder state.\n",
            argv[0]);
    return 1;
  }
  setvbuf(stdout, NULL, _IONBF, 0);
  int bender = atoi(argv[2]);
  fprintf(stderr, "[recorder-dump] bender=%d — arming c2h reader + dumping black box "
          "(NO reset)\n", bender);

  SoftMCPlatform pf(bender);
  if (pf.init() != SOFTMC_SUCCESS) {
    fprintf(stderr, "[recorder-dump] platform init failed (card not enumerated?) — "
            "cannot read the recorder.\n");
    return 1;
  }
  // Deliberately NO reset_fpga / set_aref / set_readback_mode: reading the box
  // must not disturb the state it captured, and a reset would clear it.

  // Arm a raw-capture receiver so the 32-B dump beat is caught. A trivial read
  // is the vehicle (RUN_AFTER_FLASH.md step 1); fresh process => no prior
  // receiver to join, so this cannot block on a stuck drain thread.
  Program trivial_read = rs_read_prog(/*bank=*/0, /*row=*/0, /*label=*/424242);
  pf.replay_send_resident(trivial_read);
  pf.recorder_dump();                              // h2c-only +15 word, byte[9]=0x80

  // Bounded, non-poisoning receive: pop c2h bytes into a growing buffer and scan
  // for 0xDBC0DEC1 (build35 recorder v2 record-1 magic; the build34 v1 magic was
  // 0xDBC0DEC0). The dump is now 64 BYTES = record 1 (0xDBC0DEC1) immediately
  // followed by record 2 (0xDBC0DEC2, MIG/DDR domain). The trivial read's payload
  // + its own trailer may precede it; keep pulling until the full 64 B are found
  // or the budget expires (a full XDMA wedge yields nothing -> RUN_AFTER_FLASH.md
  // step 4).
  const int CAP = 8192 + 64 * 32;                  // read payload + record slack
  std::vector<uint8_t> buf(CAP, 0);
  int rb = 0;
  long found = -1;
  const long budget_ms = 8000;
  auto t0 = std::chrono::steady_clock::now();
  while (rb + 64 <= CAP) {
    for (int off = 0; off + 4 <= rb; off += 4) {
      uint32_t wv; memcpy(&wv, buf.data() + off, 4);
      if (wv == 0xDBC0DEC1u) { found = off; break; }
    }
    if (found >= 0 && found + 64 <= rb) break;      // need BOTH records (64 B)
    long elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                       std::chrono::steady_clock::now() - t0).count();
    if (elapsed >= budget_ms) break;
    int more = pf.receiveDataTry(buf.data() + rb, std::min(4096, CAP - rb), 1000);
    if (pf.recv_stalled()) break;                  // Try never poisons; defensive
    if (more == 0) { if (found >= 0 && found + 64 <= rb) break; else continue; }
    rb += more;
  }

  if (found < 0 || found + 64 > rb) {
    fprintf(stderr, "[recorder-dump] NO 0xDBC0DEC1 (build35 v2) 64-byte record after %ld ms "
            "(received %d B). The c2h path is likely FULLY wedged (XDMA wedge, not just a "
            "walk wedge) — escalate: reset_fpga -> DDR full_reset -> cold power-cycle "
            "(RUN_AFTER_FLASH.md step 4).\n", budget_ms, rb);
    return 4;
  }

  // ---- decode the 16 little-endian uint32 words (build-35 recorder v2) ----
  // words 0..7 = record 1 (fabric/rung2 + session); words 8..15 = record 2 (MIG).
  uint32_t w[16];
  memcpy(w, buf.data() + found, 64);
  static const char* RS_NAMES[10] = {
    "IDLE","KICK","RUN","SETTLE","FLUSH","DRAIN","DONE","WAIT_START","MAINT_YIELD","QUIESCE" };
  static const char* FE_NAMES[3] = { "IDLE","INIT_MEM","EXECUTE" };
  auto rs_name = [&](uint32_t s){ return s < 10 ? RS_NAMES[s] : "?"; };
  auto fe_name = [&](uint32_t s){ return s < 3  ? FE_NAMES[s] : "?"; };

  uint32_t last_n         = w[1] >> 16,       iters_finned = w[1] & 0xFFFF;
  // build42: word2 REPURPOSED (readback_engine rec_word2_skips) -- was
  // {user_fins, maint_fins}, now the two fetch-corruption skip tripwires.
  uint32_t malformed_skips = w[2] >> 16,      parity_skips = w[2] & 0xFFFF;
  uint32_t st             = w[3];
  uint32_t wdog_fires     = st & 0xFF;
  uint32_t abort_state    = (st >> 8)  & 0xF;
  uint32_t rep_state_now  = (st >> 12) & 0xF;
  uint32_t fe_state       = (st >> 16) & 0x3;
  uint32_t exec_bank      = (st >> 18) & 0x1;
  uint32_t loaded         = (st >> 19) & 0x3;
  uint32_t desc_mode      = (st >> 21) & 0x1;
  uint32_t rep_busy       = (st >> 22) & 0x1;
  uint32_t abort_seen     = (st >> 23) & 0x1;
  uint32_t maint_req      = w[4], maint_gnt = w[5];
  uint32_t records_emitted = w[6] >> 16,      flush_edges = w[6] & 0xFFFF;
  uint32_t c2h_idle_hw    = w[7];
  // ---- record 2 (build35 recorder v2, MIG/DDR domain) ----
  uint32_t rec2_magic     = w[8];
  uint32_t mig_cmd_idle_hw= w[9];    // DDR command-bus silent high-water (MIG/DDR wedge)
  uint32_t mig_rd_idle_hw = w[10];   // read-data silent high-water (MC servicing)
  uint32_t mig_cmd_cnt    = w[11];   // saturating DDR command activity count
  uint32_t migst          = w[12];
  uint32_t mig_calib_now  = migst & 0x1;
  uint32_t mig_calib_lost = (migst >> 1) & 0x1;
  uint32_t mig_mc_rst_now = (migst >> 2) & 0x1;
  uint32_t mig_mc_rst_seen= (migst >> 3) & 0x1;
  uint32_t mig_mc_rst_cnt = migst >> 16;
  uint32_t mig_cmd_idle_now = w[13]; // live DDR command idle counter at dump time
  uint32_t mig_rd_idle_now  = w[14]; // live read-data idle counter at dump time
  uint32_t mig_build_tag  = w[15];   // trailer-magic echo (confirms the flashed image)

  printf("[recorder-dump] ==== build-35 black-box record v2 (0xDBC0DEC1 + 0xDBC0DEC2) ====\n");
  printf("  raw:  %08x %08x %08x %08x %08x %08x %08x %08x\n",
         w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]);
  printf("  raw2: %08x %08x %08x %08x %08x %08x %08x %08x\n",
         w[8], w[9], w[10], w[11], w[12], w[13], w[14], w[15]);
  printf("  w0 magic             = 0x%08X %s\n", w[0],
         w[0] == 0xDBC0DEC1u ? "(OK v2)" : "(UNEXPECTED)");
  printf("  w1 last_n            = %u   (last walk's requested iter/desc count)\n", last_n);
  printf("     iters_finned      = %u   (iterations that FINNED before the abort)\n", iters_finned);
  printf("  w2 malformed_skips   = %u   (build41 BRANCH&INFO-shape fetch skips, cumulative from rst)\n", malformed_skips);
  printf("     parity_skips      = %u   (build42 per-byte IMEM-parity fetch skips, cumulative from rst)\n", parity_skips);
  printf("  w3 status            = 0x%08X\n", st);
  printf("     wdog_fires        = %u\n", wdog_fires);
  printf("     abort_state       = %u (%s)   (replay FSM state captured AT the abort)\n",
         abort_state, rs_name(abort_state));
  printf("     rep_state_now     = %u (%s)   (current replay FSM state)\n",
         rep_state_now, rs_name(rep_state_now));
  printf("     fe_state          = %u (%s)   (frontend FSM state)\n", fe_state, fe_name(fe_state));
  printf("     exec_bank         = %u\n", exec_bank);
  printf("     loaded[1:0]       = %u\n", loaded);
  printf("     desc_mode         = %u\n", desc_mode);
  printf("     rep_busy          = %u\n", rep_busy);
  printf("     abort_seen        = %u (sticky)\n", abort_seen);
  // build37: frontend fingerprint latched AT the watchdog abort (word3[31:24]).
  printf("     FE-fingerprint@abort: fetch_hold=%u swap_pend=%u await_flush=%u "
         "stream_en=%u maint_proc=%u maint_req=%u prog_proc=%u swap_settle=%u\n",
         (st>>31)&1, (st>>30)&1, (st>>29)&1, (st>>28)&1,
         (st>>27)&1, (st>>26)&1, (st>>25)&1, (st>>24)&1);
  if (((st>>31)&1) && ((st>>30)&1) && ((st>>29)&1) && ((st>>28)&1) && !((st>>25)&1))
    printf("     ^ MATCHES the build-37 streaming-swap fetch_hold deadlock signature\n");
  printf("  w4 maint_req         = %u   (maintenance-request rising edges)\n", maint_req);
  printf("  w5 maint_gnt         = %u   (maintenance-grant pulses)\n", maint_gnt);
  printf("  w6 records_emitted   = %u   (per-program trailers this session)\n", records_emitted);
  printf("     flush_edges       = %u\n", flush_edges);
  printf("  w7 c2h_idle_hw       = %u cycles   (longest c2h-silent run — the wedge signature)\n",
         c2h_idle_hw);
  // ---- record 2: MIG/DDR domain (build35 recorder v2) ----
  printf("  --- record 2 (MIG/DDR domain) ---\n");
  printf("  w8  rec2_magic       = 0x%08X %s\n", rec2_magic,
         rec2_magic == 0xDBC0DEC2u ? "(OK)" : "(UNEXPECTED)");
  printf("  w9  mig_cmd_idle_hw  = %u cycles   (longest DDR command-bus SILENT run — THE MIG/DDR wedge signature)\n",
         mig_cmd_idle_hw);
  printf("  w10 mig_rd_idle_hw   = %u cycles   (longest read-data SILENT run — MC not servicing)\n",
         mig_rd_idle_hw);
  printf("  w11 mig_cmd_count    = %u          (DDR commands issued this session, saturating)\n", mig_cmd_cnt);
  printf("  w12 mig_status       = 0x%08X\n", migst);
  printf("      calib_complete   = %u   calib_LOST(sticky) = %u\n", mig_calib_now, mig_calib_lost);
  printf("      mc_reset_now     = %u   mc_reset_seen      = %u   mc_reset_count = %u\n",
         mig_mc_rst_now, mig_mc_rst_seen, mig_mc_rst_cnt);
  printf("  w13 mig_cmd_idle_now = %u cycles   (live at dump time)\n", mig_cmd_idle_now);
  printf("  w14 mig_rd_idle_now  = %u cycles   (live at dump time)\n", mig_rd_idle_now);
  printf("  w15 build_tag        = 0x%08X %s\n", mig_build_tag,
         mig_build_tag == PIM_EXPECTED_BUILD_TAG ? "(build-43 image OK)" : "(image mismatch!)");

  // VERDICT: distinguish a fabric-FSM wedge (record 1) from a MIG/DDR-domain wedge
  // (record 2). The build34 finding was that the durable corruption "lives BELOW
  // the FSM layer the recorder sees" -- record 2 is exactly that layer.
  if (wdog_fires > 0 || abort_seen)
    printf("[recorder-dump] VERDICT(fabric): watchdog FIRED (abort_state=%s, %u/%u iters finned) — "
           "walk aborted cleanly; the sequencer self-recovered.\n",
           rs_name(abort_state), iters_finned, last_n);
  else
    printf("[recorder-dump] VERDICT(fabric): watchdog did NOT fire — c2h-idle high-water=%u cyc.\n",
           c2h_idle_hw);
  // build42: fetch-corruption skip verdict (Fix 1's live readout). Each skip =
  // one glitched fetch turned into a 1-instruction NOP instead of a 2M-cycle
  // wedge. parity_skips discriminates a bit-corrupt IMEM READ (odd-weight per
  // byte) that b41's narrow BRANCH&INFO detector cannot see.
  if (parity_skips > 0 || malformed_skips > 0)
    printf("[recorder-dump] VERDICT(fetch): corruption SURVIVED as skips — "
           "malformed_skips=%u parity_skips=%u. A nonzero parity_skips with the "
           "stored IMEM intact on re-fetch = a READ-TRANSIENT corruption CONFIRMED "
           "(Fix 1 territory); a zero parity_skips but wrong IMEM = write mislocation.\n",
           malformed_skips, parity_skips);
  else
    printf("[recorder-dump] VERDICT(fetch): no fetch-corruption skips this session "
           "(malformed_skips=0 parity_skips=0).\n");
  if (mig_calib_lost || mig_mc_rst_seen)
    printf("[recorder-dump] VERDICT(MIG): **calib_lost=%u mc_reset_seen=%u** — the MIG/DDR "
           "controller was RESET or LOST CALIBRATION during the session. This is the "
           "durable, below-FSM corruption: a cold power-cycle + reconfigure is REQUIRED "
           "(reset_fpga / full_reset will NOT recover it).\n", mig_calib_lost, mig_mc_rst_seen);
  else if (mig_cmd_idle_hw > 1000000u)
    printf("[recorder-dump] VERDICT(MIG): DDR command bus went SILENT for %u cyc while calib "
           "held — the wedge is in the MC/PHY servicing path (backpressure/QoS), not a calib "
           "loss. Dump again after a few ms and diff mig_cmd_count to see if the bus is dead.\n",
           mig_cmd_idle_hw);
  else
    printf("[recorder-dump] VERDICT(MIG): MIG healthy (calib held, cmd bus active, cmd_count=%u). "
           "The wedge, if any, is above the MIG layer (fabric/XDMA).\n", mig_cmd_cnt);
  return 0;
}

// ---------------------------------------------------------------------------
// build43 Fix B helper: read recorder record 2 (MIG/DDR domain) WITHOUT a
// reset. Fills w_out[0..15] and returns true iff the 64-B v2 record was found.
// ============================================================================
// build44 -- profiler capture helpers (RUN_AFTER_FLASH recipe, additive).
// cap128(): arm a trivial read + RECORDER_DUMP and capture the FULL 128-byte
// build-44 dump (records 0xDBC0DEC1..C4). Same non-poisoning idiom as
// run_recorder_dump, sizes 64->128. Returns false if the record never arrives.
// ============================================================================
static bool cap128(SoftMCPlatform& pf, uint32_t w_out[32], long budget_ms = 8000) {
  Program trivial_read = rs_read_prog(/*bank=*/0, /*row=*/0, /*label=*/424244);
  pf.replay_send_resident(trivial_read);
  pf.recorder_dump();
  const int CAP = 8192 + 128 * 32;
  std::vector<uint8_t> buf(CAP, 0);
  int rb = 0; long found = -1;
  auto t0 = std::chrono::steady_clock::now();
  while (rb + 128 <= CAP) {
    for (int off = 0; off + 4 <= rb; off += 4) {
      uint32_t wv; memcpy(&wv, buf.data() + off, 4);
      if (wv == 0xDBC0DEC1u) { found = off; break; }
    }
    if (found >= 0 && found + 128 <= rb) break;
    long el = std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::steady_clock::now() - t0).count();
    if (el >= budget_ms) break;
    int more = pf.receiveDataTry(buf.data() + rb, std::min(4096, CAP - rb), 1000);
    if (pf.recv_stalled()) break;
    if (more == 0) { if (found >= 0 && found + 128 <= rb) break; else continue; }
    rb += more;
  }
  if (found < 0 || found + 128 > rb) return false;
  memcpy(w_out, buf.data() + found, 128);
  return true;
}

// Hardened build-44 capture: a bare 0xDBC0DEC1 word can occur in stale row
// payload (observed on silicon flash-day: first-match anchoring framed
// garbage). Anchor on the VALIDATED C1+C2 pair (C2 magic at +32 B), then take
// C3/C4 at +64/+96 if present, else scan forward for a validated C3+C4 pair
// (non-contiguous emission tolerated). Reports layout via *gap_out (bytes
// between C2 end and C3 start; 0 = contiguous; -1 = C3/C4 not found).
static bool cap128v2(SoftMCPlatform& pf, uint32_t w_out[32], long* gap_out,
                     long budget_ms = 8000) {
  Program trivial_read = rs_read_prog(/*bank=*/0, /*row=*/0, /*label=*/424245);
  pf.replay_send_resident(trivial_read);
  pf.recorder_dump();
  const int CAP = 8192 + 128 * 64;
  std::vector<uint8_t> buf(CAP, 0);
  int rb = 0;
  auto rd32 = [&](long off) { uint32_t v; memcpy(&v, buf.data() + off, 4); return v; };
  long c1 = -1, c3 = -1;
  auto t0 = std::chrono::steady_clock::now();
  while (rb + 64 <= CAP) {
    c1 = -1; c3 = -1;
    for (long off = 0; off + 4 <= rb; off += 4) {
      uint32_t wv = rd32(off);
      if (c1 < 0 && wv == 0xDBC0DEC1u && off + 36 <= rb && rd32(off + 32) == 0xDBC0DEC2u)
        c1 = off;
      if (c1 >= 0 && c3 < 0 && off > c1 && wv == 0xDBC0DEC3u && off + 36 <= rb &&
          rd32(off + 32) == 0xDBC0DEC4u)
        c3 = off;
    }
    if (c1 >= 0 && c3 >= 0 && c3 + 64 <= rb) break;
    long el = std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::steady_clock::now() - t0).count();
    if (el >= budget_ms) break;
    int more = pf.receiveDataTry(buf.data() + rb, std::min(4096, CAP - rb), 1000);
    if (pf.recv_stalled()) break;
    if (more == 0) continue;
    rb += more;
  }
  if (c1 < 0) return false;
  memcpy(w_out, buf.data() + c1, 64);
  if (c3 >= 0 && c3 + 64 <= rb) {
    memcpy(((uint8_t*)w_out) + 64, buf.data() + c3, 64);
    if (gap_out) *gap_out = c3 - (c1 + 64);
  } else {
    memset(((uint8_t*)w_out) + 64, 0, 64);
    if (gap_out) *gap_out = -1;
  }
  return true;
}

// `recorder-dump-prof <bender>` : capture the 128-B dump, print the 32 words as
// hex for profile_parse. `prof-snap <bender>` : magic-gate (>=0xDBC0DE29 via a
// cap128 build_tag read), then send PROF_SNAP (+18). Bracketing recipe:
//   prof-snap 2  ->  <workload>  ->  prof-snap 2  ->  recorder-dump-prof 2 | profile_parse -
static int run_recorder_dump_prof(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "Usage: %s recorder-dump-prof <bender>\n", argv[0]); return 1; }
  setvbuf(stdout, NULL, _IONBF, 0);
  // stdout carries ONLY the 32 hex words (profile_parse tokenizes stdin —
  // platform chatter like "Sent RECORDER_DUMP ... 0xDBC0DEC1" corrupts the
  // parse; observed flash-day). Route std::cout to stderr for the whole mode.
  std::streambuf* cout_old = std::cout.rdbuf(std::cerr.rdbuf());
  int bender = atoi(argv[2]);
  SoftMCPlatform pf(bender);
  if (pf.init() != SOFTMC_SUCCESS) { fprintf(stderr, "[dump-prof] platform init failed.\n"); return 1; }
  // NO reset: reading must not disturb the recorder/profiler shadow state.
  uint32_t w[32]; long gap = -2;
  if (!cap128v2(pf, w, &gap)) {
    fprintf(stderr, "[dump-prof] NO validated C1+C2 dump (c2h wedged or pre-build-44 image).\n");
    std::cout.rdbuf(cout_old);
    return 4;
  }
  fprintf(stderr, "[dump-prof] C3/C4 gap after C2: %ld bytes (%s)\n", gap,
          gap == 0 ? "contiguous" : (gap < 0 ? "C3/C4 NOT FOUND" : "non-contiguous"));
  std::cout.rdbuf(cout_old);   // hex words below are the ONLY stdout
  if (w[16] != 0xDBC0DEC3u || w[24] != 0xDBC0DEC4u)
    fprintf(stderr, "[dump-prof] WARN: C3/C4 magics = 0x%08X/0x%08X (expected DBC0DEC3/C4) — "
            "pre-build-44 image or framing slip; hex below is raw.\n", w[16], w[24]);
  for (int i = 0; i < 32; i++) printf("%08X ", w[i]);
  printf("\n");
  return 0;
}

// ============================================================================
// build-46 DIAG (ADDITIVE, observation-only): capture the full 192-byte SIX-record
// recorder dump (C1..C6) for the descriptor->register PATCH SEAM. Anchors on the
// VALIDATED C1(+0)+C2(+32) pair AND requires C5(+128)+C6(+160) framed contiguous
// (tlast on beat 6), so a stale bare-0xDBC0DEC1 word cannot mis-anchor (same
// hardening as cap128v2; flash-day false-anchor trap). Non-poisoning read idiom
// (no reset). seam_parse expects C5 at w[32] and C6 at w[40] => contiguous 192 B
// from the C1 anchor. Fills w_out[0..47]; false if the 192-B frame never arrives.
static bool cap192(SoftMCPlatform& pf, uint32_t w_out[48], long budget_ms = 8000) {
  Program trivial_read = rs_read_prog(/*bank=*/0, /*row=*/0, /*label=*/424246);
  pf.replay_send_resident(trivial_read);
  pf.recorder_dump();                              // existing +15 word triggers the 6-beat dump
  const int CAP = 8192 + 192 * 64;
  std::vector<uint8_t> buf(CAP, 0);
  int rb = 0;
  auto rd32 = [&](long off) { uint32_t v; memcpy(&v, buf.data() + off, 4); return v; };
  long c1 = -1;
  auto t0 = std::chrono::steady_clock::now();
  while (rb + 192 <= CAP) {
    c1 = -1;
    for (long off = 0; off + 192 <= rb; off += 4) {
      if (rd32(off) == 0xDBC0DEC1u && rd32(off + 32) == 0xDBC0DEC2u &&
          rd32(off + 128) == 0xDBC0DEC5u && rd32(off + 160) == 0xDBC0DEC6u) {
        c1 = off; break;
      }
    }
    if (c1 >= 0) break;
    long el = std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::steady_clock::now() - t0).count();
    if (el >= budget_ms) break;
    int more = pf.receiveDataTry(buf.data() + rb, std::min(4096, CAP - rb), 1000);
    if (pf.recv_stalled()) break;
    if (more == 0) continue;
    rb += more;
  }
  if (c1 < 0) return false;
  memcpy(w_out, buf.data() + c1, 192);             // C1..C6 = 48 words
  return true;
}

// build-47: capture the 256-B EIGHT-record dump (C1..C8; tlast on beat 8 = C8).
// Same non-poisoning read idiom + robust anchor as cap192, extended to the two
// DIAG3 records: C7 (0xDBC0DEC7 tag-bump census @ +192) and C8 (0xDBC0DEC8
// M2-ADDR first-WR @ +224). Anchors on the VALIDATED C1(+0)/C2(+32)/C5(+128)/
// C6(+160) frame AND additionally requires C8(+224)=0xDBC0DEC8 so a truncated
// 6-record (pre-build-47) dump cannot mis-anchor. Fills w_out[0..63]; false if
// the full 256-B frame never arrives. seam_parse expects 64 words.
static bool cap256(SoftMCPlatform& pf, uint32_t w_out[64], long budget_ms = 8000) {
  Program trivial_read = rs_read_prog(/*bank=*/0, /*row=*/0, /*label=*/424247);
  pf.replay_send_resident(trivial_read);
  pf.recorder_dump();                              // existing +15 word triggers the 8-beat dump
  const int CAP = 8192 + 256 * 64;
  std::vector<uint8_t> buf(CAP, 0);
  int rb = 0;
  auto rd32 = [&](long off) { uint32_t v; memcpy(&v, buf.data() + off, 4); return v; };
  long c1 = -1;
  auto t0 = std::chrono::steady_clock::now();
  while (rb + 256 <= CAP) {
    c1 = -1;
    for (long off = 0; off + 256 <= rb; off += 4) {
      if (rd32(off) == 0xDBC0DEC1u && rd32(off + 32) == 0xDBC0DEC2u &&
          rd32(off + 128) == 0xDBC0DEC5u && rd32(off + 160) == 0xDBC0DEC6u &&
          rd32(off + 224) == 0xDBC0DEC8u) {
        c1 = off; break;
      }
    }
    if (c1 >= 0) break;
    long el = std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::steady_clock::now() - t0).count();
    if (el >= budget_ms) break;
    int more = pf.receiveDataTry(buf.data() + rb, std::min(4096, CAP - rb), 1000);
    if (pf.recv_stalled()) break;
    if (more == 0) continue;
    rb += more;
  }
  if (c1 < 0) return false;
  memcpy(w_out, buf.data() + c1, 256);             // C1..C8 = 64 words
  return true;
}

// ===========================================================================
// `seam-sweep <bender> <calib> <bank> <Kmax> <variant> [reps=1]`  (build-46 DIAG
// PER-INDEX MISS-MAP campaign, 2026-08-02) — ADDITIVE, observation-only.
//
// THE METHOD (latch-last-event trick): the C5/C6 seam shadows hold the LAST
// patch event per path + cumulative counters. This runs a CONTROLLED single
// descriptor walk of length k for k=1..Kmax, each with FRESH counters
// (reset_fpga per k), then reads the seam (cap192). Deliverables per k:
//   * cnt_x_li/cnt_pf_x  -> x-path fire delta at index (k-1)   (reg-5/act_bram)
//   * cnt_row_li/cnt_pf_row -> row-path fire delta at index (k-1) (reg-6/desc_bram)
//   * rep_x_word / act_rd_data = index (k-1)'s DELIVERED x VALUE (cap192's own
//     trivial read has no reg-5 LI, so the C5 x-shadow is clean)
//   * rep_row_base = cur_desc_r[59:43] after the k-walk = index (k-1)'s row VALUE
//     (cap192's trivial read is not a walk, so cur_desc_r is unchanged)
// Per the coordinator's FWFT/read-latency directive, we verify the VALUE (both
// paths) against the expected per-index descriptor content, not just fire counts.
//
// variants (one card session each): base | content | addr | read | bank | pad
//   base    x=const, row=const  (identical descriptors: any miss is INDEX-driven)
//   content x=0xC0DE0000+j, row=const   (distinct x -> INDEX-vs-CONTENT; FWFT: is
//           rep_x_word == this-index x, or the PREVIOUS index's? = stale-by-one)
//   addr    x=const, row=60100+j        (distinct row -> is rep_row_base correct
//           per index or stale? row VALUE was NEVER verified in the §CAPTURE run)
//   read    READ resident, row=60100+j  (row-only path, NO reg-5: tests whether an
//           x-fetch adjacency changes row's fate = the act_bram-vs-desc_bram split)
//   bank    x=0xB0000000+j, bank=j&3, bpen  (per-bank vs global)
//   pad     head-padded FILL resident (reg-5 LI fetched ~24 instr later: if the
//           x-miss is act_bram read latency, more slack before the reg-5 fetch cures it)
//
// The walk does not need valid DRAM data — only the reg-5/reg-6 LI FETCH events
// drive the seam shadows — so rows are NOT pre-written (drain is discarded).
// Determinism: pass reps>1 to capture each k reps times (fine-grain determinism
// check for the CDC/metastability class). NOT in BINARIES => production untouched.
static int run_seam_sweep(int argc, char** argv) {
  if (argc < 7) {
    fprintf(stderr, "Usage: %s seam-sweep <bender> <calib> <bank> <Kmax> "
            "<base|content|addr|read|bank|pad> [reps=1]\n", argv[0]);
    return 1;
  }
  setvbuf(stdout, NULL, _IONBF, 0);
  // stdout carries ONLY the CSV; route platform std::cout chatter to stderr.
  std::streambuf* cout_old = std::cout.rdbuf(std::cerr.rdbuf());
  int bender = atoi(argv[2]);
  int bank = atoi(argv[4]);
  int Kmax = atoi(argv[5]);
  std::string var = argv[6];
  int reps = (argc > 7) ? atoi(argv[7]) : 1;
  if (Kmax < 1 || Kmax > 60) { fprintf(stderr, "[seam-sweep] Kmax 1..60\n"); std::cout.rdbuf(cout_old); return 1; }
  if (reps < 1 || reps > 16) reps = 1;
  const uint32_t FILLROW = 60016;
  const long recv_ms = 4000;
  bool is_read = (var == "read");
  bool is_pad  = (var == "pad");
  bool is_bank = (var == "bank");
  fprintf(stderr, "[seam-sweep] bender=%d bank=%d Kmax=%d variant=%s reps=%d\n",
          bender, bank, Kmax, var.c_str(), reps);

  // per-index descriptor generator (returns expected x/row for verification)
  auto gen = [&](int j, uint32_t& x, uint32_t& row, uint32_t& acc,
                 uint32_t& bnk, bool& bpen) {
    acc = ds_acc(0, 0); bnk = (uint32_t)bank; bpen = false;
    if (var == "content")      { x = 0xC0DE0000u + (uint32_t)j; row = FILLROW; }
    else if (var == "addr")    { x = 0xA5A5A5A5u; row = 60100u + (uint32_t)j; }
    else if (var == "read")    { x = 0u;          row = 60100u + (uint32_t)j; }
    else if (is_bank)          { x = 0xB0000000u + (uint32_t)j; row = FILLROW;
                                 bnk = (uint32_t)(j & 3); bpen = true; }
    else /* base | pad */      { x = 0xA5A5A5A5u; row = FILLROW; }
  };

  SoftMCPlatform pf(bender);
  if (pf.init() != SOFTMC_SUCCESS) { fprintf(stderr, "[seam-sweep] init failed\n"); std::cout.rdbuf(cout_old); return 1; }

  // CSV header (stdout): analysis-ready
  printf("# seam-sweep variant=%s bank=%d Kmax=%d reps=%d\n", var.c_str(), bank, Kmax, reps);
  printf("var,k,rep,cnt_x_li,cnt_pf_x,cnt_row_li,cnt_pf_row,cnt_settle,"
         "rep_x_word,act_rd_data,exp_x_km1,rep_state,pf_x,patch_en,act_rd_addr,desc_rd_addr,"
         "fetch_pc,fetch_hi,rep_row_base,exp_row_km1,pf_row,r_patch_en,r_fetch_pc,"
         "ss_act_rd_data,ss_act_rd_addr,desc_lo,desc_hi,build_tag,"
         "c1_last_n,c1_iters_done,c1_wdog_fires,c1_abort_state,c1_abort_seen,c1_abort_fp,"
         "recv_bytes,n_mag,n_ack,n_drain,c1_records,c1_flushes,c2_cmd_idle_hw,c2_cmd_act,"
         "c1_malformed_skips,c1_parity_skips,row_census\n");

  for (int k = 1; k <= Kmax; k++) {
    for (int rp = 0; rp < reps; rp++) {
      // ---- fresh session for clean per-k counters ----
      pf.reset_fpga();
      { Program s; s.add_inst(SMC_END()); pf.execute(s); }
      pf.set_aref(false);
      pf.set_readback_mode(false); pf.set_readback_mode(false);
      pf.drain_stray(400, 4);
      pf.set_readback_mode_accxbp(); pf.set_readback_mode_accxbp();  // ACCUM_XBP + clear
      int l1 = 9000 + k * 4 + rp, l2 = 9500 + k * 4 + rp, lr = 8500 + k * 4 + rp;
      // PIM_SWEEP_PLACEHOLDER (sweep3b): bake a nonzero placeholder (C11 idiom)
      // into the fill resident so the census three-way discriminates:
      // row==placeholder -> iter body never executed; row==0 -> executed with
      // wdata=0; row==x -> patched fill landed.
      uint32_t s_ph = 0;
      if (const char* e = getenv("PIM_SWEEP_PLACEHOLDER")) s_ph = (uint32_t)strtoul(e, nullptr, 0);
      Program res = is_read ? rs_accum_read(bank, 60100u, lr)
                  : is_pad  ? ds_fill_read_prog_padded(bank, FILLROW, l1, l2)
                            : ds_fill_read_prog(bank, FILLROW, l1, l2, s_ph);
      // ORDER (b32b / C7 finding, matches do_walk default): DLOAD while the
      // frontend is IDLE, BEFORE the resident send. Post-send DLOADs correlate
      // with dead/stalled walks. So: dload 0..k-1 -> send resident (auto-run)
      // -> clear -> replay_n(k).
      // PIM_SWEEP_LOADHOLD (sweep3c): stage the resident WITHOUT the auto-run
      // (build-35 LOAD_HOLD +16; image 0x2C >= gate 0x20). With a pre-written
      // sentinel in FILLROW this makes the post-walk census 3-way DECISIVE:
      // sentinel -> the walk never wrote; placeholder -> walk wrote RAW;
      // x(k-1) -> patched writes landed (no auto-run confound).
      bool lhold = getenv("PIM_SWEEP_LOADHOLD") != nullptr;
      if (lhold && !is_read) {
        pf.set_readback_mode(false); pf.set_readback_mode(false); pf.drain_stray(600, 5);
        if (!ds_write_verify(pf, bank, FILLROW, 0x5EED5EEDu, 6000 + k * 16 + rp, recv_ms))
          fprintf(stderr, "[seam-sweep] k=%d rep=%d sentinel prewrite FAILED\n", k, rp);
        pf.set_readback_mode_accxbp(); pf.set_readback_mode_accxbp();
      }
      uint32_t xx=0,rr=0,aa=0,bb=0; bool bp=false;
      for (int j = 0; j < k; j++) { gen(j, xx, rr, aa, bb, bp); pf.dload((uint32_t)j, xx, rr, aa, bb, bp); }
      if (lhold) pf.set_load_hold();       // next load stages WITHOUT auto-run
      pf.replay_send_resident(res);        // auto-run unless load-held
      usleep(5000);
      pf.set_readback_mode_accxbp();       // clear (auto-run deposit, or no-op)
      pf.replay_n((uint16_t)k);
      int want = 8192 + (k + 32) * 64;
      std::vector<uint8_t> buf(want, 0);
      int recv_bytes = pf.receiveDataTry(buf.data(), want, recv_ms);
      // ACK-EVIDENCE (hang-vs-finloss discriminator): classify the walk's c2h
      // stream. Each COMPLETED iteration leaves a record (per-program ack /
      // trailer); a stalled-with-ack walk = fin EMITTED but not consumed
      // (tag-mismatch class); stalled-without-ack = program never ENDed (hang).
      RsParse wr = rs_parse(buf.data(), recv_bytes > 0 ? recv_bytes : 0);
      pf.drain_stray(300, 4);
      // WRITES-LANDED census (fill variants): read FILLROW back in READ mode.
      // Stalled walk + census==last-x  -> writes landed (END/fin side lost);
      // stalled walk + census==auto-placeholder(0) -> execute never ran (C11 face).
      char census[24] = "na";
      if (!is_read && getenv("PIM_SWEEP_CENSUS")) {
        // census v2 framing (battery C12 lesson): locate the TRAILING magic at
        // offset >= 8192 with no interior magic; the 8192 B before it are the
        // row payload. Never trust offset 0 (a stray record fakes the census).
        pf.set_readback_mode(false); pf.set_readback_mode(false); pf.drain_stray(1500, 8);
        std::vector<uint8_t> rbk(2 * 8192 + 16 * 32, 0);
        Program rpc = rs_read_prog(bank, FILLROW, 7000 + k * 16 + rp);
        pf.replay_send_resident(rpc);
        int got = pf.receiveDataTry(rbk.data(), (int)rbk.size(), recv_ms);
        long pay = -1;
        if (got >= 8192 + 4) {
          std::vector<long> mags;
          for (long off = 0; off + 4 <= got; off += 4) {
            uint32_t wv; memcpy(&wv, rbk.data() + off, 4);
            if ((wv & 0xFFFFFF00u) == 0xDBC0DE00u) mags.push_back(off);
          }
          for (long M : mags) {
            if (M < 8192) continue;
            bool clean = true;
            for (long j : mags) if (j > M - 8192 && j < M) { clean = false; break; }
            if (clean) { pay = M - 8192; break; }
          }
        }
        if (pay >= 0) {
          const uint8_t* base_p = rbk.data() + pay;
          uint32_t x0; memcpy(&x0, base_p, 4); bool uni = true;
          for (int i = 1; i < 2048; i++) { uint32_t gv; memcpy(&gv, base_p + 4L*i, 4);
            if (gv != x0) { uni = false; break; } }
          if (uni) snprintf(census, sizeof census, "U:%08X", x0);
          else     snprintf(census, sizeof census, "MIXED");
        } else snprintf(census, sizeof census, "NOFRAME");
        pf.drain_stray(300, 4);
      }
      uint32_t w[48];
      if (!cap192(pf, w)) { fprintf(stderr, "[seam-sweep] k=%d rep=%d cap192 FAILED\n", k, rp);
                            printf("%s,%d,%d,CAP_FAIL\n", var.c_str(), k, rp); continue; }
      // decode C5/C6
      uint32_t cnt_x_li=w[39]&0xFFFF, cnt_pf_x=(w[39]>>16)&0xFFFF;
      uint32_t cnt_row_li=w[44]&0xFFFF, cnt_pf_row=(w[44]>>16)&0xFFFF;
      uint32_t cnt_settle=(w[47]>>16)&0xFFFF;
      uint32_t w37=w[37];
      uint32_t act_rd_addr=w37&0x1FFF, desc_rd_addr=(w37>>14)&0xFF;
      uint32_t patch_en=(w37>>22)&1, pf_xb=(w37>>23)&1, rep_state=(w37>>25)&0xF;
      uint32_t w38=w[38]; uint32_t fetch_pc=w38&0x1FFF, fetch_hi=(w38>>24)&0xFF;
      uint32_t rep_row_base=w[41]&0x1FFFF;
      uint32_t w43=w[43]; uint32_t r_fetch_pc=w43&0x1FFF, r_patch_en=(w43>>21)&1, pf_rowb=(w43>>22)&1;
      uint32_t ss_act_rd_data=w[45]; uint32_t ss_act_rd_addr=w[47]&0x1FFF;
      // expected value for the LAST walked index (k-1)
      uint32_t ex_x=0,ex_row=0,ea=0,eb=0; bool ep=false; gen(k-1, ex_x, ex_row, ea, eb, ep);
      // C1 walker post-mortem: w1={last_n[31:16],iters_done[15:0]}; w3 status
      // ([7:0]=wdog_fires [11:8]=abort_state [23]=abort_seen [31:24]=abort fingerprint)
      uint32_t c1w1=w[1], c1w3=w[3];
      printf("%s,%d,%d,%u,%u,%u,%u,%u,0x%08X,0x%08X,0x%08X,%u,%u,%u,%u,%u,%u,0x%02X,"
             "%u,%u,%u,%u,%u,0x%08X,%u,0x%08X,0x%08X,0x%08X,"
             "%u,%u,%u,%u,%u,0x%02X,"
             "%d,%ld,%ld,%ld,%u,%u,%u,%u,%u,%u,%s\n",
             var.c_str(), k, rp, cnt_x_li, cnt_pf_x, cnt_row_li, cnt_pf_row, cnt_settle,
             w[33], w[34], ex_x, rep_state, pf_xb, patch_en, act_rd_addr, desc_rd_addr,
             fetch_pc, fetch_hi, rep_row_base, ex_row, pf_rowb, r_patch_en, r_fetch_pc,
             ss_act_rd_data, ss_act_rd_addr, w[35], w[36], w[15],
             (c1w1>>16)&0xFFFF, c1w1&0xFFFF, c1w3&0xFF, (c1w3>>8)&0xF,
             (c1w3>>23)&1, (c1w3>>24)&0xFF,
             recv_bytes, wr.n_mag, wr.n_empty_ack, wr.n_drain,
             (w[6]>>16)&0xFFFF, w[6]&0xFFFF, w[9], w[11],
             (w[2]>>16)&0xFFFF, w[2]&0xFFFF, census);
    }
  }
  fprintf(stderr, "[seam-sweep] DONE variant=%s\n", var.c_str());
  std::cout.rdbuf(cout_old);
  return 0;
}

// `recorder-dump-seam <bender>` : capture the 192-B six-record dump, print the 48
// words as little-endian hex for seam_parse. The seam shadows are always-on (no
// arm) — they latch the LAST patch event of whatever walk just ran, so the recipe
// is: run the failing desc-smoke walk, then this, then seam_parse -.
static int run_recorder_dump_seam(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "Usage: %s recorder-dump-seam <bender>\n", argv[0]); return 1; }
  setvbuf(stdout, NULL, _IONBF, 0);
  // stdout carries ONLY the 48 hex words (seam_parse tokenizes stdin); route
  // std::cout chatter to stderr for the whole mode, same as recorder-dump-prof.
  std::streambuf* cout_old = std::cout.rdbuf(std::cerr.rdbuf());
  int bender = atoi(argv[2]);
  SoftMCPlatform pf(bender);
  if (pf.init() != SOFTMC_SUCCESS) { fprintf(stderr, "[dump-seam] platform init failed.\n"); std::cout.rdbuf(cout_old); return 1; }
  // NO reset: reading must not disturb the seam shadow state.
  // build-47: read the full 256-B EIGHT-record dump (C1..C8) so seam_parse can
  // decode the DIAG3 words (C7 tag-bump census, C8 M2-ADDR). cap192 (C1..C6) is
  // still used by seam-sweep; recorder-dump-seam needs the two DIAG3 records.
  uint32_t w[64];
  if (!cap256(pf, w)) {
    fprintf(stderr, "[dump-seam] NO validated 256-B C1..C8 dump (c2h wedged or pre-build-47 image "
                    "without C7/C8). Confirm build_tag=0xDBC0DE2E first.\n");
    std::cout.rdbuf(cout_old);
    return 4;
  }
  std::cout.rdbuf(cout_old);   // hex words below are the ONLY stdout
  if (w[15] != 0xDBC0DE2Eu)
    fprintf(stderr, "[dump-seam] WARN: build_tag=0x%08X (expected 0xDBC0DE2E build-47).\n", w[15]);
  if (w[32] != 0xDBC0DEC5u || w[40] != 0xDBC0DEC6u ||
      w[48] != 0xDBC0DEC7u || w[56] != 0xDBC0DEC8u)
    fprintf(stderr, "[dump-seam] WARN: C5/C6/C7/C8 magics = 0x%08X/0x%08X/0x%08X/0x%08X "
                    "(expected DBC0DEC5/C6/C7/C8).\n", w[32], w[40], w[48], w[56]);
  for (int i = 0; i < 64; i++) printf("%08X ", w[i]);
  printf("\n");
  return 0;
}

static int run_prof_snap(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "Usage: %s prof-snap <bender>\n", argv[0]); return 1; }
  setvbuf(stdout, NULL, _IONBF, 0);
  int bender = atoi(argv[2]);
  SoftMCPlatform pf(bender);
  if (pf.init() != SOFTMC_SUCCESS) { fprintf(stderr, "[prof-snap] platform init failed.\n"); return 1; }
  uint32_t w[32];
  if (!cap128v2(pf, w, nullptr)) { fprintf(stderr, "[prof-snap] no dump — refusing to send +18 blind.\n"); return 4; }
  uint32_t btag = w[15];
  if (btag < 0xDBC0DE29u) {
    fprintf(stderr, "[prof-snap] build_tag 0x%08X < 0xDBC0DE29: PROF_SNAP (+18) NOT available "
            "on this image — refusing (would clobber IMEM).\n", btag);
    return 1;
  }
  pf.prof_snap();
  fprintf(stderr, "[prof-snap] window latched + counters cleared (build_tag 0x%08X).\n", btag);
  return 0;
}

// Same non-poisoning read idiom as run_recorder_dump; a short per-call budget so
// it can be polled. Record 2 (w[8..15]) carries the MIG status word w[12]:
//   bit0 = calib_now, bit1 = calib_lost(sticky), bit2 = mc_rst_now,
//   bit3 = mc_rst_seen(sticky), [31:16] = mc_rst_count(sat16).
// ---------------------------------------------------------------------------
static bool mig_read_recorder(SoftMCPlatform& pf, uint32_t w_out[16], long budget_ms) {
  Program trivial_read = rs_read_prog(/*bank=*/0, /*row=*/0, /*label=*/424242);
  pf.replay_send_resident(trivial_read);
  pf.recorder_dump();                              // h2c-only +15 word, byte[9]=0x80
  const int CAP = 8192 + 64 * 32;
  std::vector<uint8_t> buf(CAP, 0);
  int rb = 0; long found = -1;
  auto t0 = std::chrono::steady_clock::now();
  while (rb + 64 <= CAP) {
    for (int off = 0; off + 4 <= rb; off += 4) {
      uint32_t wv; memcpy(&wv, buf.data() + off, 4);
      if (wv == 0xDBC0DEC1u) { found = off; break; }
    }
    if (found >= 0 && found + 64 <= rb) break;
    long el = std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::steady_clock::now() - t0).count();
    if (el >= budget_ms) break;
    int more = pf.receiveDataTry(buf.data() + rb, std::min(4096, CAP - rb), 1000);
    if (pf.recv_stalled()) break;
    if (more == 0) { if (found >= 0 && found + 64 <= rb) break; else continue; }
    rb += more;
  }
  if (found < 0 || found + 64 > rb) return false;
  memcpy(w_out, buf.data() + found, 64);
  return true;
}

// ============================================================================
// build36 -- `mig-reinit <bender> [bank] [row]` : host-reachable per-DIMM MIG
// re-initialisation + recovery probe. THE FIX for the "unrecoverable corruption"
// class: sys_rst_l was tied 1'b1, so MIG calibration never re-ran on a soft
// recovery. This mode fires the MIG_REINIT (+17) word, waits for the MIG to
// re-init + recalibrate, re-brings-up the card, and PROVES recovery with a
// simple WRITE/READ ROW PROBE.
//
// Why a write/read probe and NOT the recorder: the recorder's sticky bits
// (calib_lost / mc_reset_seen) RESET WITH THE CORE, so after a successful
// re-init the recorder is FRESH and cannot show "a loss then a recovery". A
// direct write-then-read of a DRAM row is the honest functional witness.
//
// It runs a BASELINE probe (before re-init) and a RECOVERY probe (after), so on
// a corrupted rig you see baseline-FAIL -> recovery-PASS (the corruption was
// cleared), and on a healthy card you see PASS -> PASS (re-init is
// non-destructive). Recovery-PASS is the verdict; recovery-FAIL => the fault is
// below even a MIG re-init: escalate to a cold power-cycle + JTAG reconfigure.
// ============================================================================
static int run_mig_reinit(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "Usage: %s mig-reinit <bender> [bank] [row]\n"
                    "  build-36 per-DIMM MIG re-init (+17) + write/read recovery probe.\n"
                    "  Requires a build-36 image (trailer magic >= 0xDBC0DE21).\n",
            argv[0]);
    return 1;
  }
  setvbuf(stdout, NULL, _IONBF, 0);
  int bender = atoi(argv[2]);
  int bank   = (argc > 3) ? atoi(argv[3]) : 0;
  uint32_t row = (argc > 4) ? (uint32_t)strtoul(argv[4], nullptr, 0) : 0x400u;
  fprintf(stderr, "[mig-reinit] bender=%d bank=%d row=%u\n", bender, bank, row);

  SoftMCPlatform pf(bender);
  if (pf.init() != SOFTMC_SUCCESS) {
    fprintf(stderr, "[mig-reinit] platform init failed (card not enumerated?).\n");
    return 1;
  }
  // bring-up idiom (mirror copy-smoke): reset, drain the reset ack, aref off,
  // READ mode (never ACCUM here).
  auto bringup = [&](){
    pf.reset_fpga();
    { Program sync; sync.add_inst(SMC_END()); pf.execute(sync); }   // drain reset ack
    pf.set_aref(false);
    pf.set_readback_mode(false); pf.set_readback_mode(false);
  };
  bringup();

  // ---- image-magic GATE: the +17 word clobbers IMEM on any pre-build-36 image
  //      (its ctrl-detect mask does not cover +17). Refuse below 0xDBC0DE21.
  {
    std::vector<uint8_t> mbuf(32, 0);
    Program probe = cs_zero_read_prog();
    pf.replay_send_resident(probe);
    int mgot = pf.receiveData(mbuf.data(), 32);
    if (pf.recv_stalled()) {
      fprintf(stderr, "[mig-reinit] magic probe stalled (%d/32) — card poisoned; refusing.\n", mgot);
      return 1;
    }
    uint32_t img = cs_trailer_magic(mbuf.data(), mgot);
    fprintf(stderr, "[mig-reinit] image trailer magic = 0x%08x\n", img);
    if ((img & 0xFFFFFF00u) != 0xDBC0DE00u) {
      fprintf(stderr, "[mig-reinit] not a 0xDBC0DExx magic — image/framing unexpected, refusing.\n");
      return 1;
    }
    if (img < 0xDBC0DE21u) {
      fprintf(stderr, "[mig-reinit] image magic 0x%08x < 0xDBC0DE21: MIG_REINIT (+17) NOT available "
              "on this image — refusing (no +17 word sent, would clobber IMEM).\n", img);
      return 0;   // clean, expected refusal on a pre-build-36 tower
    }
  }

  // ---- BASELINE probe (before re-init): write pat0, read back, count mismatch.
  std::vector<uint32_t> pat0(2048), pat1(2048);
  { std::mt19937 r(0xB36A5E ^ (row * 2654435761u) ^ (uint32_t)bank);
    for (int i = 0; i < 2048; i++) pat0[i] = (uint32_t)r(); }
  { std::mt19937 r(0x5EC0DE ^ (row * 40503u) ^ (uint32_t)(bank + 7));
    for (int i = 0; i < 2048; i++) pat1[i] = (uint32_t)r(); }
  std::vector<uint8_t> rb(8192 + 32, 0);
  rs_write_row(pf, bank, row, pat0.data());
  int bgot = cs_read_row(pf, bank, row, 36001, rb.data());
  long base_bad = (bgot >= 8192 && !pf.recv_stalled())
                    ? cs_row_mismatch(rb.data(), pat0.data()) : -1;
  bool base_ok = (base_bad == 0);
  fprintf(stderr, "[mig-reinit] BASELINE write/read: recv=%d mismatch=%ld -> %s\n",
          bgot, base_bad, base_ok ? "ok" : (base_bad < 0 ? "STALLED" : "MISMATCH"));

  // ---- fire MIG_REINIT and WAIT for re-init + calibration ----
  fprintf(stderr, "[mig-reinit] sending MIG_REINIT (+17); the MIG will re-init + recalibrate...\n");
  pf.mig_reinit();
  usleep(5000000);   // ~5 s: DDR4 init + MRS + read/write calibration (generous)

  // ---- build43 Fix B (Mismatch B): CALIB HANDSHAKE, not a blind sleep alone.
  //      Read recorder record 2 PRE-bringup (the sticky mc_rst_seen/count + the
  //      re-asserted calib_now still hold here; bringup()'s reset_fpga clears
  //      them). Confirm (1) the MIG reset was SEEN and (2) calib RE-ASSERTED,
  //      with a bounded retry + LOUD report so a reset that was never seen, or a
  //      cal that never completed, is surfaced instead of silently assumed.
  bool hs_reset_seen = false, hs_calib_ok = false; uint32_t hs_mc_cnt = 0;
  {
    uint32_t w[16];
    for (int p = 0; p < 8; p++) {           // ~2 s of bounded retry after the settle
      if (mig_read_recorder(pf, w, /*budget_ms=*/2000)) {
        uint32_t migst = w[12];
        uint32_t calib_now = migst & 0x1u;
        uint32_t mc_seen   = (migst >> 3) & 0x1u;
        uint32_t mc_cnt    = migst >> 16;
        if (mc_seen || mc_cnt > 0) hs_reset_seen = true;
        hs_mc_cnt = mc_cnt;
        if (calib_now) { hs_calib_ok = true; break; }
      }
      usleep(250000);
    }
    if (!hs_reset_seen)
      fprintf(stderr, "[mig-reinit] WARN: recorder shows NO mc_reset_seen/count — the MIG reset "
                      "may not have been seen (CDC/stretcher?). Treat recovery as SUSPECT.\n");
    if (!hs_calib_ok)
      fprintf(stderr, "[mig-reinit] FAIL: init_calib_complete did NOT re-assert (calib handshake "
                      "timed out) — calibration did not complete.\n");
    fprintf(stderr, "[mig-reinit] calib handshake: reset_seen=%d calib_reasserted=%d mc_rst_cnt=%u\n",
            (int)hs_reset_seen, (int)hs_calib_ok, hs_mc_cnt);
  }

  // ---- re-bring-up (the core reset during re-init cleared IMEM/state) ----
  bringup();

  // ---- RECOVERY probe (after re-init): write pat1, read back, count mismatch.
  std::fill(rb.begin(), rb.end(), 0);
  rs_write_row(pf, bank, row, pat1.data());
  int rgot = cs_read_row(pf, bank, row, 36002, rb.data());
  long rec_bad = (rgot >= 8192 && !pf.recv_stalled())
                   ? cs_row_mismatch(rb.data(), pat1.data()) : -1;
  bool rec_ok = (rec_bad == 0);
  fprintf(stderr, "[mig-reinit] RECOVERY write/read: recv=%d mismatch=%ld -> %s\n",
          rgot, rec_bad, rec_ok ? "ok" : (rec_bad < 0 ? "STALLED" : "MISMATCH"));

  // ---- build43 Fix B (Mismatch B): READ-GATE B_GUARD SELF-CHECK on the recovery
  //      read. The defect-B latch pins bytes 16-23 (lanes {4,5} mod 16) of every
  //      512-bit beat, so it corrupts READ-mode reads too (RESULT.md §3). Apply
  //      the canonical B_GUARD rule to the just-read row: per 128 beats flag
  //      lanes[16k+4]==lanes[16k+5] nonzero; >=16/128 => ARMED. This catches the
  //      LATCHED (gross) armed state a plain mismatch count may attribute to
  //      noise, and prints ARMED/CLEAN so a reinit-on-detection policy is
  //      self-verifying. NOTE: a not-yet-tipped MARGINAL gate needs the full
  //      ACCUM walk-boundary cadence (the campaign's bguard probe-verify step);
  //      this single READ catches the persistent latch that mig-reinit clears.
  int bg_twins = 0;
  if (rgot >= 8192 && !pf.recv_stalled()) {
    const uint32_t* L = reinterpret_cast<const uint32_t*>(rb.data());
    for (int b = 0; b < 128; b++) {
      uint32_t l4 = L[b * 16 + 4], l5 = L[b * 16 + 5];
      if (l4 == l5 && l4 != 0) bg_twins++;
    }
  }
  bool bg_armed = (bg_twins >= 16);
  fprintf(stderr, "[mig-reinit] READ-GATE B_GUARD self-check: twin_beats=%d/128 -> %s\n",
          bg_twins, bg_armed ? "ARMED (defect B latched)" : "CLEAN");

  // ---- verdict ----
  printf("[mig-reinit] ==== build-36 MIG re-init verdict (bender=%d bank=%d row=%u) ====\n",
         bender, bank, row);
  printf("  baseline write/read : %s (mismatch=%ld)\n", base_ok ? "PASS" : "FAIL", base_bad);
  printf("  recovery write/read : %s (mismatch=%ld)\n", rec_ok  ? "PASS" : "FAIL", rec_bad);
  printf("  calib handshake     : reset_seen=%d calib_reasserted=%d (mc_rst_cnt=%u)\n",
         (int)hs_reset_seen, (int)hs_calib_ok, hs_mc_cnt);
  printf("  read-gate B_GUARD   : %s (twin_beats=%d/128)\n",
         bg_armed ? "ARMED" : "CLEAN", bg_twins);
  if (bg_armed) {
    printf("[mig-reinit] READ-GATE STRESS: defect-B latch present post-reinit — calib=%d but the "
           "read path is MARGINAL/latched. Re-issuing mig-reinit is the recovery.\n", (int)hs_calib_ok);
    rec_ok = false;   // a marginal/latched read gate is NOT a clean recovery
  }
  if (rec_ok && !base_ok)
    printf("[mig-reinit] VERDICT: RECOVERED — the DRAM row was NOT functional before the MIG "
           "re-init and IS after it. The MIG re-init cleared the below-FSM corruption.\n");
  else if (rec_ok && base_ok)
    printf("[mig-reinit] VERDICT: PASS (non-destructive) — the card was already healthy; the MIG "
           "re-init + recalibration completed and the row still reads back exactly.\n");
  else
    printf("[mig-reinit] VERDICT: FAIL — the row does NOT read back after the MIG re-init. The "
           "fault is below even a MIG re-init: escalate to a cold power-cycle + JTAG reconfigure.\n");
  return rec_ok ? 0 : 4;
}

// ============================================================================
// 2026-08-01 -- `mig-reinit-trace <bender> [duration_s=8]` : from-t0 calib
// timeline through ONE mig-reinit. Discriminator for the build-43 null A/B
// (lockout in image, arming rate unchanged): does a SECOND reset event still
// fire after the first recalibration completes (lockout window mis-sized), or
// is there exactly one reset event (double-reset mechanism refuted)?
//
// Method: fire MIG_REINIT with NO settle sleep, then poll mig_read_recorder()
// (bounded 250 ms budget per sample) continuously, timestamping each sample's
// start/end and decoding w12 (calib_now / calib_lost / mc_rst_now / seen /
// count). The fabric is DEAD during sys_rst + (possibly) calibration, so a
// sample in that window either blocks in the h2c send or misses its record —
// both mark a DEAD WINDOW whose edges time the reset events:
//   ONE reset  : one dead window, then ok samples with calib=1 forever.
//   TWO resets : ok/calib=1 appears, then a SECOND dead window (or calib drop)
//                ~at the moment the spurious recovery toggle fires, then ok.
// The summary counts dead-window onsets after the first recovery.
// Patient by design: no process timeout may kill this mid-h2c (XDMA wedge).
// ============================================================================
static int run_mig_reinit_trace(int argc, char** argv) {
  if (argc < 3) {
    fprintf(stderr, "Usage: %s mig-reinit-trace <bender> [duration_s=8]\n", argv[0]);
    return 1;
  }
  setvbuf(stdout, NULL, _IONBF, 0);
  int bender = atoi(argv[2]);
  long dur_s = (argc > 3) ? atol(argv[3]) : 8;
  fprintf(stderr, "[reinit-trace] bender=%d duration=%lds\n", bender, dur_s);

  SoftMCPlatform pf(bender);
  if (pf.init() != SOFTMC_SUCCESS) {
    fprintf(stderr, "[reinit-trace] platform init failed.\n");
    return 1;
  }
  pf.reset_fpga();
  { Program sync; sync.add_inst(SMC_END()); pf.execute(sync); }
  pf.set_aref(false);
  pf.set_readback_mode(false); pf.set_readback_mode(false);

  // image-magic gate (same contract as mig-reinit: +17 needs >= build-36)
  {
    std::vector<uint8_t> mbuf(32, 0);
    Program probe = cs_zero_read_prog();
    pf.replay_send_resident(probe);
    int mgot = pf.receiveData(mbuf.data(), 32);
    if (pf.recv_stalled()) {
      fprintf(stderr, "[reinit-trace] magic probe stalled (%d/32) — refusing.\n", mgot);
      return 1;
    }
    uint32_t img = cs_trailer_magic(mbuf.data(), mgot);
    fprintf(stderr, "[reinit-trace] image trailer magic = 0x%08x\n", img);
    if ((img & 0xFFFFFF00u) != 0xDBC0DE00u || img < 0xDBC0DE21u) {
      fprintf(stderr, "[reinit-trace] image lacks MIG_REINIT (+17) — refusing.\n");
      return 1;
    }
  }

  auto ms_since = [](std::chrono::steady_clock::time_point a,
                     std::chrono::steady_clock::time_point b) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count();
  };

  // 3 pre-reinit baseline samples (expect ok, calib=1)
  for (int i = 0; i < 3; i++) {
    uint32_t w[16];
    auto a = std::chrono::steady_clock::now();
    bool ok = mig_read_recorder(pf, w, /*budget_ms=*/250);
    auto b = std::chrono::steady_clock::now();
    printf("[reinit-trace] PRE  %d: dt=%ldms ok=%d calib=%d\n",
           i, ms_since(a, b), (int)ok, ok ? (int)(w[12] & 1u) : -1);
  }

  // fire + poll from t0
  auto t0 = std::chrono::steady_clock::now();
  pf.mig_reinit();
  printf("[reinit-trace] MIG_REINIT sent at t0. Polling...\n");
  bool ever_ok_calib = false;
  long first_ok_ms = -1;
  int dead_after_recovery = 0;   // dead-window onsets AFTER the first recovery
  int calib_drops = 0;           // ok samples with calib=0 after a calib=1 sample
  bool prev_ok = true;           // treat pre-reinit as responsive
  while (true) {
    auto a = std::chrono::steady_clock::now();
    if (ms_since(t0, a) > dur_s * 1000) break;
    uint32_t w[16] = {0};
    bool ok = mig_read_recorder(pf, w, /*budget_ms=*/250);
    auto b = std::chrono::steady_clock::now();
    int calib = ok ? (int)(w[12] & 1u) : -1;
    int lost  = ok ? (int)((w[12] >> 1) & 1u) : -1;
    int rnow  = ok ? (int)((w[12] >> 2) & 1u) : -1;
    int rseen = ok ? (int)((w[12] >> 3) & 1u) : -1;
    unsigned rcnt = ok ? (w[12] >> 16) : 0;
    printf("[reinit-trace] T +%6ld..+%6ld ms  ok=%d calib=%d lost=%d rst_now=%d rst_seen=%d rst_cnt=%u\n",
           ms_since(t0, a), ms_since(t0, b), (int)ok, calib, lost, rnow, rseen, rcnt);
    if (ok && calib == 1) {
      if (first_ok_ms < 0) first_ok_ms = ms_since(t0, b);
      ever_ok_calib = true;
    }
    if (ever_ok_calib) {
      if (prev_ok && !ok) dead_after_recovery++;      // second dead window onset
      if (ok && calib == 0) calib_drops++;            // calib visibly dropped
    }
    prev_ok = ok;
    usleep(25000);
  }
  printf("[reinit-trace] ==== SUMMARY ====\n");
  printf("  first ok+calib sample : %+ld ms after MIG_REINIT\n", first_ok_ms);
  printf("  dead windows AFTER first recovery : %d\n", dead_after_recovery);
  printf("  calib=0 samples AFTER first calib=1 : %d\n", calib_drops);
  printf("  VERDICT: %s\n",
         (first_ok_ms < 0) ? "NO RECOVERY SEEN (extend duration / escalate)" :
         (dead_after_recovery == 0 && calib_drops == 0)
           ? "SINGLE reset event — no post-recovery dead window (lockout covers the toggle, or no toggle)"
           : "SECOND event AFTER recovery — spurious reset still firing (lockout mis-sized)");

  // leave the card clean
  pf.reset_fpga();
  { Program sync; sync.add_inst(SMC_END()); pf.execute(sync); }
  pf.set_aref(false);
  pf.set_readback_mode(false); pf.set_readback_mode(false);
  return 0;
}

// ============================================================================
// build42 -- `swap-storm <bender> <calib> [bank=0] [minutes=1]` : IMEM
// stream/swap/fetch STRESS reproducer for the physical ping-pong glitch.
//
// Maximizes the IMEM stream/swap/fetch event rate by streaming a small valid
// READ program back-to-back under a PIM_STREAM producer-style session (the
// exact build-9 ping-pong path the silicon glitch lives in -- each stream_send
// stages the next program into the idle bank and swaps on program end),
// executing and receiving each, for [minutes]. Host-prints progress every
// SAMPLE_EVERY programs; at the end reads the black-box recorder ONCE and
// reports programs_run, swaps, malformed_skips, parity_skips, wdog_fires,
// wedges.
//
// PURPOSE (post-flash): quantify the physical IMEM-corruption RATE in minutes
// and PROVE Fix 1 catches it -- parity_skips must increment with ZERO wedges (a
// bit-corrupt fetch becomes a 1-instruction NOP skip, not a 2M-cycle stall). A
// nonzero parity_skips with a clean re-fetch of the same IMEM = a READ-TRANSIENT
// corruption CONFIRMED (the durable-vs-transient discriminator).
//
// PURE additional mode: it touches no shared state and changes no other mode's
// behaviour. The card is required (it drives real streaming); no card => refuse.
static int run_swap_storm(int argc, char** argv) {
  if (argc < 4) {
    fprintf(stderr, "Usage: %s swap-storm <bender> <calib> [bank=0] [minutes=1]\n"
            "  IMEM stream/swap/fetch stress: streams small READ programs back-to-back\n"
            "  (PIM_STREAM ping-pong swaps) for [minutes], then reads the recorder and\n"
            "  reports parity_skips / malformed_skips / wdog_fires / wedges.\n"
            "  Needs a build-9+ streaming image; build-42 (magic 0x%08X) for parity_skips.\n",
            argv[0], PIM_EXPECTED_BUILD_TAG);
    return 1;
  }
  setvbuf(stdout, NULL, _IONBF, 0);
  int bender = atoi(argv[2]);
  std::string calib_p = argv[3]; (void)calib_p;   // CLI parity (dimm2 trio); storm uses a raw read
  int bank = (argc > 4) ? atoi(argv[4]) : 0;
  double minutes = (argc > 5) ? atof(argv[5]) : 1.0;
  if (minutes <= 0) minutes = 1.0;
  const long recv_ms = 4000;
  const uint32_t STORM_ROW = 60000;               // far from production pools 45312-45952
  fprintf(stderr, "[swap-storm] bender=%d bank=%d minutes=%.2f row=%u\n",
          bender, bank, minutes, STORM_ROW);

  SoftMCPlatform pf(bender);
  if (pf.init() != SOFTMC_SUCCESS) { fprintf(stderr, "[swap-storm] init failed (card up?)\n"); return 1; }
  pf.reset_fpga();
  { Program sync; sync.add_inst(SMC_END()); pf.execute(sync); }    // drain reset ack + zero the recorder
  pf.set_aref(false);
  pf.set_readback_mode(false); pf.set_readback_mode(false);        // READ mode (streaming needs READ/SEG_POP)

  // ---- recorder sampler (out-of-band; call only with c2h quiet / stream closed) ----
  // Now delegates to the shared recorder_read_counters() helper (same read the
  // per-session PIM_COUNTER_GATE uses at teardown).
  auto sample_recorder = [&](uint32_t& malf, uint32_t& par, uint32_t& wdog, uint32_t& btag)->bool {
    return recorder_read_counters(pf, malf, par, wdog, btag);
  };

  // ---- image-magic gate (uses PIM_EXPECTED_BUILD_TAG) ----
  {
    std::vector<uint8_t> mbuf(32, 0);
    Program probe = cs_zero_read_prog();
    pf.replay_send_resident(probe);
    int mgot = pf.receiveDataTry(mbuf.data(), 32, recv_ms);
    uint32_t img = cs_trailer_magic(mbuf.data(), mgot);
    fprintf(stderr, "[swap-storm] image trailer magic = 0x%08x\n", img);
    if ((img & 0xFFFFFF00u) != 0xDBC0DE00u) {
      fprintf(stderr, "[swap-storm] not a 0xDBC0DExx magic — image/framing unexpected, refusing.\n");
      return 2;
    }
    if (img < 0xDBC0DE09u) {   // streaming ping-pong = build-9+
      fprintf(stderr, "[swap-storm] image magic 0x%08x < build-9: no streaming ping-pong to storm, refusing.\n", img);
      return 2;
    }
    if (img != PIM_EXPECTED_BUILD_TAG)
      fprintf(stderr, "[swap-storm] WARNING: image magic 0x%08x != build-42 0x%08x — parity_skips "
              "will read 0 (Fix 1 absent); the storm still measures swaps/wedges/wdog.\n",
              img, PIM_EXPECTED_BUILD_TAG);
  }

  // ---- THE STORM ----
  Program prog = rs_read_prog(bank, STORM_ROW, /*label=*/990001);  // small READ = real fetch+swap
  const int payload = (int)row_read_bytes();
  const long SAMPLE_EVERY = 2000;
  long programs_run = 0, swaps = 0, wedges = 0;
  std::vector<uint8_t> rb(payload + 32, 0);
  auto t0 = std::chrono::steady_clock::now();
  auto deadline = t0 + std::chrono::milliseconds((long)(minutes * 60000.0));
  ensure_readback(pf, false);                     // pipeline empty before opening the session
  pf.stream_start(SoftMCPlatform::STREAM_SIZED);
  bool broke = false;
  while (std::chrono::steady_clock::now() < deadline) {
    pf.stream_send(prog, payload);                // ping-pong swap into the idle bank
    int got = pf.receiveDataTry(rb.data(), payload + 32, recv_ms);
    if (pf.recv_stalled() || got < payload) {
      wedges++;
      fprintf(stderr, "[swap-storm] ** WEDGE: recv stalled after %ld programs (got=%d/%d) — a "
              "corrupted fetch was NOT skipped (or a non-parity wedge). Stopping; recorder-dump follows.\n",
              programs_run, got, payload + 32);
      broke = true;
      break;
    }
    programs_run++; swaps++;                       // each streamed program = one ping-pong swap
    if (programs_run % SAMPLE_EVERY == 0) {
      double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      fprintf(stderr, "[swap-storm] progress: programs=%ld swaps=%ld elapsed=%.1fs rate=%.0f/s\n",
              programs_run, swaps, el, programs_run / (el > 0 ? el : 1.0));
    }
  }
  pf.stream_stop();

  // ---- final recorder read (cumulative since the start-of-run reset_fpga) ----
  uint32_t malf = 0, par = 0, wdog = 0, btag = 0;
  bool ok = sample_recorder(malf, par, wdog, btag);
  double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  printf("\n[swap-storm] ==== SUMMARY (bank=%d, %.2f min) ====\n", bank, minutes);
  printf("  programs_run   = %ld\n", programs_run);
  printf("  swaps          = %ld   (ping-pong IMEM swaps; rate %.0f/s)\n", swaps,
         programs_run / (el > 0 ? el : 1.0));
  printf("  wedges         = %ld   (recv stalls = a fetch NOT skipped)\n", wedges);
  if (ok) {
    printf("  malformed_skips= %u   (recorder w2 hi; build41 BRANCH&INFO fetch skips)\n", malf);
    printf("  parity_skips   = %u   (recorder w2 lo; build42 per-byte IMEM-parity fetch skips)\n", par);
    printf("  wdog_fires     = %u   (recorder w3[7:0])\n", wdog);
    printf("  build_tag      = 0x%08X %s\n", btag,
           btag == PIM_EXPECTED_BUILD_TAG ? "(build-43 OK)" : "(unexpected — parity may be absent)");
  } else {
    printf("  [recorder read FAILED — c2h may be fully wedged; escalate per RUN_AFTER_FLASH]\n");
  }
  if (wedges == 0 && (par > 0 || malf > 0))
    printf("[swap-storm] VERDICT: %u+%u corrupted fetches SURVIVED as skips over %ld swaps with "
           "ZERO wedges — Fix 1 CONFIRMED (physical glitch ~%.2g/swap; parity_skips>0 + clean "
           "re-fetch = read-transient).\n", par, malf, swaps, (double)(par + malf) / (swaps > 0 ? swaps : 1));
  else if (wedges == 0 && par == 0 && malf == 0)
    printf("[swap-storm] VERDICT: ZERO wedges, ZERO skips over %ld swaps — no IMEM corruption "
           "surfaced this run (the glitch is rare; run longer / more banks).\n", swaps);
  else
    printf("[swap-storm] VERDICT: %ld WEDGE(s) — a fetch was NOT skipped. If an even-weight per-byte "
           "corruption slipped the check, recorder-dump instr_hi + widen the detector. Escalate.\n", wedges);
  return (broke || wedges > 0) ? 1 : 0;
}

// ===================================================================
// [#65 2026-08-04] HOST-DRIVEN CONFIG SCHEMA + build path (bank-set /
// per-bank window / per-bank state generalization). NO HARDCODED VALUES:
// the bank set, windows, states, DIMM/role are all host-supplied, seeded
// from argv+env at power-up AND mutable at runtime via MAGIC_CONFIG.
// ===================================================================
struct BankSpec {
  int dimm_id = -1;                    // owning DIMM (bender); config, not code
  int bank_id = 0;
  BankState state = BankState::ACTIVE;
  uint32_t win_start = 0, win_end = 0; // 0/0 => inherit the global env window
};
struct ServerConfig {
  int dimm_id = 0;                     // this process's DIMM (== bender)
  std::string role = "compute";        // "compute" (0/2) | "storage" (1/3)
  std::string calib_path;              // per-bank tuples read from here
  bool dual_mode = false;              // PIM_DUAL_SUBARRAY (legacy)
  std::vector<BankSpec> banks;         // the working set for this process
};
// Startup config remembered so a runtime CFG_RECONFIG can reuse the calib
// file / dual flag while only the bank membership + states change.
static ServerConfig g_server_cfg;

// Scoped PIM_SUB_START/END override so build_backup_pool (which reads the
// env) honors a per-bank window during pool build, then restores it — keeps
// the change bank-local; the default (win 0/0) leaves the env untouched.
struct EnvWindowGuard {
  bool active = false; std::string oss, ose; bool hss = false, hse = false;
  EnvWindowGuard(uint32_t ws, uint32_t we) {
    if (!(ws || we)) return; active = true;
    if (const char* p = getenv("PIM_SUB_START")) { oss = p; hss = true; }
    if (const char* p = getenv("PIM_SUB_END"))   { ose = p; hse = true; }
    char b1[16], b2[16];
    snprintf(b1, sizeof b1, "%u", ws); snprintf(b2, sizeof b2, "%u", we);
    setenv("PIM_SUB_START", b1, 1); setenv("PIM_SUB_END", b2, 1);
  }
  ~EnvWindowGuard() {
    if (!active) return;
    if (hss) setenv("PIM_SUB_START", oss.c_str(), 1); else unsetenv("PIM_SUB_START");
    if (hse) setenv("PIM_SUB_END",   ose.c_str(), 1); else unsetenv("PIM_SUB_END");
  }
};

// Build ONE bank's config (calib + screened pools + extras + dual). This is
// the SINGLE per-bank build path — main() and CFG_RECONFIG both go through it,
// so startup and runtime reconfigure can never drift. Extracted verbatim from
// the old inline main() loop; the only additions are dimm/state/window.
static bool build_one_bank(const BankSpec& spec, const std::string& calib_p,
                           bool dual_mode, BankConfig& bc) {
  vector<Calib> cs = read_calib(calib_p, spec.bank_id);
  if (cs.empty()) {
    fprintf(stderr, "[server] no calib for bank %d in %s\n",
            spec.bank_id, calib_p.c_str());
    return false;
  }
  bc.bank_id = spec.bank_id;
  bc.dimm_id = spec.dimm_id;
  bc.state   = spec.state;
  // Resolve the effective window: explicit per-bank spec wins, else the global
  // env (seeded into bc.win_* so request-time refresh uses the same window).
  uint32_t ws = spec.win_start, we = spec.win_end;
  if (!(ws || we)) {
    if (const char* ss = getenv("PIM_SUB_START")) if (*ss) ws = (uint32_t)atoi(ss);
    if (const char* se = getenv("PIM_SUB_END"))   if (*se) we = (uint32_t)atoi(se);
  }
  bc.win_start = ws; bc.win_end = we;
  bc.calib = cs[0];
  load_fused_colmask(bc);                 // O10: PIM_FUSED_COLMASK_FILE, {bank}
  EnvWindowGuard _wg(spec.win_start, spec.win_end);  // only when spec is explicit
  bc.backup_pool = build_backup_pool(bc.calib, /*is_primary=*/true);
  if (bc.backup_pool.empty()) {
    fprintf(stderr, "[server] empty backup pool for bank %d\n", spec.bank_id);
    return false;
  }
  // D-mode extras (dense sub-clusters, ranked by population; skip <10).
  {
    std::map<uint32_t, size_t> cluster_count;
    for (const auto& c : cs) cluster_count[(c.open_rows[0] / 640) * 640]++;
    uint32_t sub0 = (cs[0].open_rows[0] / 640) * 640;
    std::vector<std::pair<size_t,uint32_t>> ranked;
    for (auto& kv : cluster_count) {
      if (kv.first == sub0) continue;
      if (kv.second < 10) continue;
      ranked.emplace_back(kv.second, kv.first);
    }
    std::sort(ranked.begin(), ranked.end(),
              [](const auto& a, const auto& b){ return a.first > b.first; });
    static int s_max_extras = -1;
    if (s_max_extras < 0) {
      const char* v = getenv("PIM_MAX_EXTRAS");
      s_max_extras = (v && *v) ? atoi(v) : 4;
      if (s_max_extras != 4) fprintf(stderr, "[server] PIM_MAX_EXTRAS=%d\n", s_max_extras);
    }
    for (auto& [cnt, sub] : ranked) {
      if ((int)bc.cs_extra.size() >= s_max_extras) break;
      for (const auto& c : cs) {
        if ((c.open_rows[0] / 640) * 640 != sub) continue;
        std::pair<uint32_t,uint32_t> win{0, 0};
        std::vector<uint32_t> pool_i = build_backup_pool(c, /*is_primary=*/false, &win);
        if (pool_i.empty()) break;
        bc.cs_extra.push_back(c);
        bc.pool_extra.push_back(std::move(pool_i));
        bc.pool_extra_win.push_back(win);
        bc.pool_extra_cursor.push_back(0);
        break;
      }
    }
    fprintf(stderr, "[server] bank %d (dimm %d, %s): %zu extra calibs",
            spec.bank_id, spec.dimm_id, bank_state_name(spec.state),
            bc.cs_extra.size());
    for (size_t ei = 0; ei < bc.cs_extra.size(); ei++)
      fprintf(stderr, " sub=%u(pool=%zu@[%u,%u))",
              (bc.cs_extra[ei].open_rows[0] / 640), bc.pool_extra[ei].size(),
              bc.pool_extra_win[ei].first, bc.pool_extra_win[ei].second);
    fprintf(stderr, "\n");
  }
  if (dual_mode && cs.size() >= 2) {
    uint32_t sub0_start = (cs[0].open_rows[0] / 640) * 640;
    for (size_t i = 1; i < cs.size(); i++) {
      if ((cs[i].open_rows[0] / 640) * 640 != sub0_start) {
        bc.calib_b = cs[i];
        bc.backup_pool_b = build_backup_pool(bc.calib_b, /*is_primary=*/false);
        if (!bc.backup_pool_b.empty()) bc.dual = true;
        break;
      }
    }
    if (!bc.dual)
      fprintf(stderr, "[server] bank %d: no second-subarray calib found\n", spec.bank_id);
  }
  return true;
}

// Build the whole bank set from a ServerConfig. Returns false (and leaves
// `banks` empty) if any bank fails — the caller decides fatal vs. reject.
static bool build_banks(const ServerConfig& cfg, std::vector<BankConfig>& banks) {
  banks.clear();
  for (const auto& spec : cfg.banks) {
    BankConfig bc;
    if (!build_one_bank(spec, cfg.calib_path, cfg.dual_mode, bc)) { banks.clear(); return false; }
    banks.push_back(std::move(bc));
  }
  g_cfg_n_banks = (int)banks.size();   // keep the IMEM K-fit heuristic honest
  return true;
}

// Invalidate ALL resident state (handles + the resident-scratch registry).
// Used on a full CFG_RECONFIG (bank set changed => every handle's per-bank
// row reservations are stale).
static void invalidate_resident_all(std::map<uint32_t, LoadedHandle>& handles) {
  size_t nh = handles.size();
  handles.clear();
  g_resident_rows.clear();
  fprintf(stderr, "[cfg] invalidated ALL resident state (%zu handles, "
          "resident-scratch registry cleared)\n", nh);
}
// Invalidate only one bank's resident-scratch entries. NOTE (honest scope):
// LOAD handles today reserve rows on EVERY active bank (cross-bank residency),
// so a bank leaving ACTIVE conservatively also drops the handle map — decoupling
// handle residency to a per-bank subset is task #67. Idle-only transitions
// (FREE/STAGING/STORAGE among themselves) call NEITHER path => no stop-the-world.
static size_t invalidate_resident_bank(int bank_id,
                                       std::map<uint32_t, LoadedHandle>& handles) {
  size_t erased = 0;
  for (auto it = g_resident_rows.begin(); it != g_resident_rows.end();) {
    if (it->first.first == bank_id) { it = g_resident_rows.erase(it); erased++; }
    else ++it;
  }
  return erased;
}

static bool cfg_write(int fd, const void* buf, size_t n) {
  const char* p = (const char*)buf; size_t off = 0;
  while (off < n) { ssize_t w = write(fd, p + off, n - off);
                    if (w <= 0) return false; off += (size_t)w; }
  return true;
}

// Handle a MAGIC_CONFIG request. Runs BETWEEN complete requests (the request
// loop is single-threaded), so a config change never races an in-flight matmul
// — that is the "when it may apply" contract. Writes its own response to
// response_fd. Returns 0 on success (response written), -1 on hard error.
static int handle_config_request(SoftMCPlatform* platform,
                                 std::vector<BankConfig>& banks,
                                 std::map<uint32_t, LoadedHandle>& handles,
                                 const uint8_t* body, size_t blen, int response_fd) {
  if (blen < 8) { fprintf(stderr, "[cfg] runt config request %zu B\n", blen); return -1; }
  uint32_t op = 0; memcpy(&op, body + 4, 4);
  auto emit_table = [&](uint32_t status) -> int {
    // Response: [u32 status][u32 n] then n×[i32 dimm][i32 bank][u32 state]
    //           [u32 pool_size][u32 win_start][u32 win_end]
    std::vector<uint8_t> out;
    auto pu32 = [&](uint32_t v){ uint8_t* q=(uint8_t*)&v; out.insert(out.end(),q,q+4); };
    pu32(status); pu32((uint32_t)banks.size());
    for (const auto& b : banks) {
      pu32((uint32_t)b.dimm_id); pu32((uint32_t)b.bank_id);
      pu32((uint32_t)b.state);   pu32((uint32_t)b.backup_pool.size());
      pu32(b.win_start);         pu32(b.win_end);
    }
    return cfg_write(response_fd, out.data(), out.size()) ? 0 : -1;
  };
  if (op == CFG_QUERY) return emit_table(0);

  if (op == CFG_RECONFIG) {
    // Payload: [u32 n] then n×[i32 dimm][i32 bank][u32 state][u32 ws][u32 we]
    // = 20 bytes/spec (pool_size is OUTPUT-only, present only in the response).
    if (blen < 12) { fprintf(stderr, "[cfg] runt RECONFIG\n"); return emit_table(1); }
    uint32_t n = 0; memcpy(&n, body + 8, 4);
    size_t off = 12, need = (size_t)n * 20;
    if (n == 0 || off + need != blen) { fprintf(stderr, "[cfg] RECONFIG size mismatch n=%u\n", n); return emit_table(1); }
    ServerConfig ncfg = g_server_cfg;   // reuse calib_path/dual/role/dimm
    ncfg.banks.clear();
    for (uint32_t i = 0; i < n; i++) {
      BankSpec s; int32_t d, bk; uint32_t st, ws, we;
      memcpy(&d, body+off, 4); memcpy(&bk, body+off+4, 4); memcpy(&st, body+off+8, 4);
      memcpy(&ws, body+off+12, 4); memcpy(&we, body+off+16, 4); off += 20;
      s.dimm_id = (d >= 0) ? d : g_server_cfg.dimm_id;
      s.bank_id = bk; s.state = (BankState)(st <= 3 ? st : 0);
      s.win_start = ws; s.win_end = we;
      if (bk < 0 || bk > 15) { fprintf(stderr, "[cfg] RECONFIG bad bank %d\n", bk); return emit_table(1); }
      ncfg.banks.push_back(s);
    }
    std::vector<BankConfig> nb;
    if (!build_banks(ncfg, nb)) { fprintf(stderr, "[cfg] RECONFIG build failed — keeping old set\n"); return emit_table(2); }
    invalidate_resident_all(handles);          // bank set changed => all stale
    banks.swap(nb);
    g_server_cfg = ncfg;
    if (platform) {                            // null only in the card-free selftest
      setup_resident_consts(*platform, banks); // re-claim consts on the new set
      setup_x_masters(*platform, banks);
    }
    fprintf(stderr, "[cfg] RECONFIG OK: N_banks=%zu\n", banks.size());
    return emit_table(0);
  }

  if (op == CFG_SET_STATE) {
    // Payload: [u32 n] then n×[i32 bank][u32 new_state]. Only banks already in
    // the set are affected. Idle<->idle transitions never invalidate.
    if (blen < 12) return emit_table(1);
    uint32_t n = 0; memcpy(&n, body + 8, 4);
    size_t off = 12, need = (size_t)n * 8;
    if (off + need != blen) return emit_table(1);
    size_t n_changed = 0, n_inval = 0;
    for (uint32_t i = 0; i < n; i++) {
      int32_t bk; uint32_t st; memcpy(&bk, body+off, 4); memcpy(&st, body+off+4, 4); off += 8;
      BankState ns = (BankState)(st <= 3 ? st : 0);
      for (auto& b : banks) {
        if (b.bank_id != bk) continue;
        if (b.state == ns) break;
        bool leaving_active = (b.state == BankState::ACTIVE && ns != BankState::ACTIVE);
        b.state = ns; n_changed++;
        if (leaving_active) { n_inval += invalidate_resident_bank(b.bank_id, handles); }
        break;
      }
    }
    // If ANY bank left ACTIVE we also drop the cross-bank handle map (handles
    // span all active banks today — see invalidate_resident_bank note).
    if (n_inval > 0) { size_t nh = handles.size(); handles.clear();
      fprintf(stderr, "[cfg] SET_STATE dropped %zu cross-bank handles (a bank left ACTIVE)\n", nh); }
    fprintf(stderr, "[cfg] SET_STATE: %zu banks changed, %zu resident rows invalidated\n",
            n_changed, n_inval);
    return emit_table(0);
  }
  fprintf(stderr, "[cfg] unknown config op %u\n", op);
  return emit_table(1);
}

// ------------------------------------------------------------------
// [#65] config-selftest: CARD-FREE unit gate (like deframe-selftest). Builds a
// 4-bank and a 16-bank config and asserts the 16-bank pool-slot total is ~4×
// the 4-bank total; then exercises a runtime CFG_SET_STATE transition + the
// invalidation model on in-memory structures. NEVER opens the card.
//   usage: <exe> config-selftest <calib_file> <pool_pattern> [sub_start sub_end]
// ------------------------------------------------------------------
static int run_config_selftest(int argc, char** argv) {
  if (argc < 4) {
    fprintf(stderr, "usage: %s config-selftest <calib_file> <pool_pattern{bank}> [sub_start sub_end]\n", argv[0]);
    return 2;
  }
  std::string calib = argv[2];
  setenv("PIM_POOL_LIST_FILE", argv[3], 1);
  if (argc >= 6) { setenv("PIM_SUB_START", argv[4], 1); setenv("PIM_SUB_END", argv[5], 1); }
  // Seed the startup config so a wire CFG_RECONFIG can reuse the calib path.
  g_server_cfg = ServerConfig{};
  g_server_cfg.calib_path = calib; g_server_cfg.dimm_id = 2; g_server_cfg.role = "compute";
  auto mk = [&](std::vector<int> bset, ServerConfig& cfg) {
    cfg.calib_path = calib; cfg.dimm_id = 2; cfg.role = "compute";
    for (int b : bset) { BankSpec s; s.dimm_id = 2; s.bank_id = b; s.state = BankState::ACTIVE; cfg.banks.push_back(s); }
  };
  auto total_slots = [&](const std::vector<BankConfig>& banks) {
    size_t t = 0; for (const auto& b : banks) t += b.backup_pool.size(); return t;
  };
  int fails = 0;
  // --- Gate A: 4-bank vs 16-bank pool-slot enumeration ---
  ServerConfig c4, c16; mk({0,1,2,3}, c4); mk({0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}, c16);
  std::vector<BankConfig> b4, b16;
  bool ok4 = build_banks(c4, b4), ok16 = build_banks(c16, b16);
  if (!ok4 || b4.size() != 4)   { fprintf(stderr, "[selftest] FAIL: 4-bank build (ok=%d n=%zu)\n", ok4, b4.size()); fails++; }
  if (!ok16 || b16.size() != 16){ fprintf(stderr, "[selftest] FAIL: 16-bank build (ok=%d n=%zu) — the OLD 8-cap would fail here\n", ok16, b16.size()); fails++; }
  size_t s4 = ok4 ? total_slots(b4) : 0, s16 = ok16 ? total_slots(b16) : 0;
  double ratio = (s4 > 0) ? (double)s16 / (double)s4 : 0.0;
  fprintf(stderr, "[selftest] Gate A: 4-bank pool slots=%zu, 16-bank=%zu, ratio=%.3f (expect ~4.0)\n", s4, s16, ratio);
  if (!(ratio > 3.9 && ratio < 4.1)) { fprintf(stderr, "[selftest] FAIL: 16-bank did NOT enumerate ~4x the pool slots\n"); fails++; }
  else fprintf(stderr, "[selftest] PASS: 16-bank config enumerates 4x pool capacity, no new sweeps\n");
  // capacity math echo
  if (b16.size() == 16) {
    size_t per_bank = s16 / 16;
    fprintf(stderr, "[selftest] capacity: ~%zu pool rows/bank => 4-bank=%zu, 16-bank=%zu rows resident-capable (x4)\n",
            per_bank, s4, s16);
  }
  // --- Gate B: runtime state transitions + invalidation model ---
  std::map<uint32_t, LoadedHandle> fake_handles;
  g_resident_rows.clear();
  // Seed a resident-scratch entry on bank 3 and a fake cross-bank handle.
  g_resident_rows[{3, 45400}] = ResidentScratch{};
  LoadedHandle fh; fh.handle_id = 1; fake_handles[1] = fh;
  // B1: idle->idle transition on bank 10 (FREE->STAGING) must NOT invalidate.
  size_t inv_before = g_resident_rows.size(), h_before = fake_handles.size();
  for (auto& b : b16) if (b.bank_id == 10) b.state = BankState::STAGING;  // no leave-ACTIVE
  size_t inv_idle = invalidate_resident_bank(10, fake_handles);  // bank 10 has no rows
  if (inv_idle != 0 || g_resident_rows.size() != inv_before || fake_handles.size() != h_before)
    { fprintf(stderr, "[selftest] FAIL: idle-only transition disturbed state\n"); fails++; }
  else fprintf(stderr, "[selftest] PASS: idle-bank transition is metadata-only (no stop-the-world)\n");
  // B2: bank 3 leaves ACTIVE -> its resident row is invalidated.
  size_t inv3 = invalidate_resident_bank(3, fake_handles);
  if (inv3 != 1 || g_resident_rows.count({3,45400})) { fprintf(stderr, "[selftest] FAIL: leaving-ACTIVE did not invalidate bank 3 (%zu)\n", inv3); fails++; }
  else fprintf(stderr, "[selftest] PASS: leaving-ACTIVE invalidates that bank's resident scratch\n");
  // B3: full reconfigure clears everything.
  invalidate_resident_all(fake_handles);
  if (!fake_handles.empty() || !g_resident_rows.empty()) { fprintf(stderr, "[selftest] FAIL: full invalidate left residue\n"); fails++; }
  else fprintf(stderr, "[selftest] PASS: full reconfigure invalidates all resident state\n");

  // --- Gate C: MAGIC_CONFIG WIRE handler round-trip (card-free, platform=null) ---
  // Drives the exact request-loop handler over a pipe with hand-built frames.
  {
    auto pu = [](std::vector<uint8_t>& v, uint32_t x){ uint8_t* q=(uint8_t*)&x; v.insert(v.end(),q,q+4); };
    std::map<uint32_t, LoadedHandle> h2;
    // CFG_QUERY on the 4-bank set.
    int pfd[2];
    if (pipe(pfd) != 0) { fprintf(stderr, "[selftest] FAIL: pipe()\n"); fails++; }
    else {
      std::vector<uint8_t> q; pu(q, MAGIC_CONFIG); pu(q, CFG_QUERY);
      int cr = handle_config_request(nullptr, b4, h2, q.data(), q.size(), pfd[1]);
      uint32_t hdr[2] = {9,9}; ssize_t rn = read(pfd[0], hdr, 8);
      if (cr != 0 || rn != 8 || hdr[0] != 0 || hdr[1] != 4) {
        fprintf(stderr, "[selftest] FAIL: CFG_QUERY wire (cr=%d rn=%zd status=%u n=%u)\n", cr, rn, hdr[0], hdr[1]); fails++;
      } else {
        // drain the 4x24 table body
        std::vector<uint8_t> tbl(4*24); ssize_t tb = read(pfd[0], tbl.data(), tbl.size());
        (void)tb;
        fprintf(stderr, "[selftest] PASS: CFG_QUERY wire returns %u-bank table\n", hdr[1]);
      }
      close(pfd[0]); close(pfd[1]);
    }
    // CFG_SET_STATE: move bank 1 -> STORAGE(2); verify the returned table shows it.
    if (pipe(pfd) != 0) { fprintf(stderr, "[selftest] FAIL: pipe()\n"); fails++; }
    else {
      std::vector<uint8_t> s; pu(s, MAGIC_CONFIG); pu(s, CFG_SET_STATE);
      pu(s, 1); pu(s, 1); pu(s, (uint32_t)BankState::STORAGE);
      int cr = handle_config_request(nullptr, b4, h2, s.data(), s.size(), pfd[1]);
      uint32_t hdr[2] = {9,9}; ssize_t rn = read(pfd[0], hdr, 8);
      bool ok = (cr == 0 && rn == 8 && hdr[0] == 0 && hdr[1] == 4);
      // read table, find bank 1's state field
      uint32_t bank1_state = 99;
      if (ok) {
        std::vector<uint8_t> tbl(4*24); ssize_t tb = read(pfd[0], tbl.data(), tbl.size());
        if (tb == (ssize_t)tbl.size()) {
          for (int i = 0; i < 4; i++) {
            int32_t bk; uint32_t st;
            memcpy(&bk, tbl.data()+i*24+4, 4); memcpy(&st, tbl.data()+i*24+8, 4);
            if (bk == 1) bank1_state = st;
          }
        }
      }
      if (!ok || bank1_state != (uint32_t)BankState::STORAGE) {
        fprintf(stderr, "[selftest] FAIL: CFG_SET_STATE wire (ok=%d bank1_state=%u)\n", ok, bank1_state); fails++;
      } else {
        fprintf(stderr, "[selftest] PASS: CFG_SET_STATE wire transitions bank 1 -> STORAGE at runtime\n");
      }
      close(pfd[0]); close(pfd[1]);
    }
    // CFG_RECONFIG: replace the set with 2 banks {0,2}; verify re-derive + ack.
    if (pipe(pfd) != 0) { fprintf(stderr, "[selftest] FAIL: pipe()\n"); fails++; }
    else {
      std::vector<uint8_t> r; pu(r, MAGIC_CONFIG); pu(r, CFG_RECONFIG); pu(r, 2);
      // spec = [i32 dimm=-1][i32 bank][u32 state=ACTIVE][u32 ws=0][u32 we=0] (20 B)
      for (int bk : {0, 2}) { int32_t d=-1; pu(r,(uint32_t)d); pu(r,(uint32_t)bk); pu(r,0u); pu(r,0u); pu(r,0u); }
      std::vector<BankConfig> bx = b4;   // handler swaps into this
      int cr = handle_config_request(nullptr, bx, h2, r.data(), r.size(), pfd[1]);
      uint32_t hdr[2] = {9,9}; ssize_t rn = read(pfd[0], hdr, 8);
      if (cr != 0 || rn != 8 || hdr[0] != 0 || hdr[1] != 2 || bx.size() != 2) {
        fprintf(stderr, "[selftest] FAIL: CFG_RECONFIG wire (cr=%d status=%u n=%u banks=%zu)\n", cr, hdr[0], hdr[1], bx.size()); fails++;
      } else {
        std::vector<uint8_t> tbl(2*24); ssize_t tb = read(pfd[0], tbl.data(), tbl.size()); (void)tb;
        fprintf(stderr, "[selftest] PASS: CFG_RECONFIG wire re-derives a 2-bank set at runtime\n");
      }
      close(pfd[0]); close(pfd[1]);
    }
  }

  fprintf(stderr, "[selftest] %s (%d failures)\n", fails ? "OVERALL FAIL" : "OVERALL PASS", fails);
  return fails ? 1 : 0;
}

int main(int argc, char** argv) {
  if (argc >= 2 && strcmp(argv[1], "config-selftest") == 0)
    return run_config_selftest(argc, argv);   // #65 host-only config gate; NO card
  if (argc >= 2 && strcmp(argv[1], "swap-storm") == 0)
    return run_swap_storm(argc, argv);           // build-42 IMEM stream/swap/fetch stress reproducer
  if (argc >= 2 && strcmp(argv[1], "replay-smoke") == 0)
    return run_replay_smoke(argc, argv);
  if (argc >= 2 && strcmp(argv[1], "copy-smoke") == 0)
    return run_copy_smoke(argc, argv);
  if (argc >= 2 && strcmp(argv[1], "deframe-selftest") == 0)
    return run_deframe_selftest();       // b49 host-deframer unit gate; NO card
  if (argc >= 2 && strcmp(argv[1], "desc-smoke") == 0)
    return run_desc_smoke(argc, argv);
  if (argc >= 2 && strcmp(argv[1], "desc-plan") == 0)
    return run_desc_plan(argc, argv);            // Task #50 M3 dry-run (no card)
  if (argc >= 2 && strcmp(argv[1], "recorder-dump") == 0)
    return run_recorder_dump(argc, argv);        // build-34 black-box post-mortem
  if (argc >= 2 && strcmp(argv[1], "mig-reinit") == 0)
    return run_mig_reinit(argc, argv);           // build-36 per-DIMM MIG re-init + recovery probe
  if (argc >= 2 && strcmp(argv[1], "mig-reinit-trace") == 0)
    return run_mig_reinit_trace(argc, argv);     // 2026-08-01 from-t0 calib timeline (defect-B discriminator)
  if (argc >= 2 && strcmp(argv[1], "recorder-dump-prof") == 0)
    return run_recorder_dump_prof(argc, argv);   // build-44 128-B C1..C4 capture -> hex for profile_parse
  if (argc >= 2 && strcmp(argv[1], "recorder-dump-seam") == 0)
    return run_recorder_dump_seam(argc, argv);   // build-46 DIAG 192-B C1..C6 seam capture -> hex for seam_parse
  if (argc >= 2 && strcmp(argv[1], "seam-sweep") == 0)
    return run_seam_sweep(argc, argv);           // build-46 DIAG per-index miss-map sweep (controlled walk, length k)
  if (argc >= 2 && strcmp(argv[1], "prof-snap") == 0)
    return run_prof_snap(argc, argv);            // build-44 PROF_SNAP (+18), magic-gated
  if (argc != 4) {
    fprintf(stderr,
        "Usage: %s <bender_id> <calib_file> <bank_arg>\n"
        "  bank_arg: \"1\" (single bank) or \"0,1,2,3\" (multi-bank Path C)\n"
        "  (or:  %s replay-smoke <bender> <calib> [bank] [row] [N])\n"
        "  (or:  %s copy-smoke <bender> <bank> <src_row> <dst_row> [seed])\n"
        "  (or:  %s desc-smoke <bender> <calib> [bank])\n"
        "  (or:  %s desc-plan <d_in> <d_out> <n_chunks> <K> [N_banks=4])\n"
        "  (or:  %s recorder-dump <bender>)\n"
        "  (or:  %s swap-storm <bender> <calib> [bank] [minutes])\n",
        argv[0], argv[0], argv[0], argv[0], argv[0], argv[0], argv[0]);
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
  // [#65 2026-08-04] Seed the ServerConfig from argv+env (behaviour-identical
  // to the old inline loop): the bank set is host-supplied (argv[3]); the DIMM
  // (== bender) and role are config; per-bank window/state default to the
  // global env / ACTIVE. Both startup and the runtime MAGIC_CONFIG path go
  // through the SAME build_banks(), so they can never diverge. PIM_DIMM_ROLE
  // seeds the role (compute/storage) — carried as config, not code.
  g_server_cfg = ServerConfig{};
  g_server_cfg.dimm_id = bender_id;
  g_server_cfg.calib_path = calib_p;
  g_server_cfg.dual_mode = dual_mode;
  if (const char* r = getenv("PIM_DIMM_ROLE")) if (*r) g_server_cfg.role = r;
  for (int bk : wanted_banks) {
    BankSpec s; s.dimm_id = bender_id; s.bank_id = bk;
    s.state = BankState::ACTIVE; s.win_start = 0; s.win_end = 0;  // 0 => global env
    g_server_cfg.banks.push_back(s);
  }
  std::vector<BankConfig> banks;
  if (!build_banks(g_server_cfg, banks)) {
    fprintf(stderr, "[server] bank set build failed for '%s'\n", bank_arg.c_str());
    return 2;
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
    // reset_fpga() sends a 32 B control word with NO receiver thread
    // (platform.cpp:158), so the device's reply for it is never consumed and
    // EVERY later program reads its PREDECESSOR's reply. Invisible while all
    // replies are identical 32 B acks; fatal on the first program returning
    // real data -- the rdRow then gets a 32 B ack (got=32 useful=0) with its
    // 8192 B row orphaned behind it. Measured: 482 consumers vs 483 device
    // records, off by exactly this one.
    // drain_stray() does NOT work here (it never reaches rtl_recv_record), but
    // execute() DOES spawn a receiver and consumeData loops until the device
    // is empty -- so one throwaway program re-aligns the stream.
    {
      Program sync;
      sync.add_inst(SMC_END());
      platform_owner->execute(sync);
      fprintf(stderr, "[startup] sync program executed "
                      "(drains the unconsumed reset ack)\n");
    }   if (stream_on()) {
      // Rung-1: arm the frontend once (idempotent SET; reset above
      // cleared it). Build-9+ image REQUIRED — trailer magic 0xDBC0DE08;
      // on older images this word falls through into instruction-load.
      platform_owner->set_stream_en(true);
    }
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
  // LEVERS #28: claim the X-MASTER rows (no-op unless PIM_XMASTER armed).
  // AFTER the consts claim so both precede the first LOAD allocation.
  setup_x_masters(platform, banks);

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
    // Dispatch ONE complete request body by its magic. Identical logic for a
    // standalone request and for each sub-body of a MAGIC_FUSE frame — the
    // per-request DRAM work + response write are byte-identical either way.
    static const bool g_disp_hash = (getenv("PIM_DISPATCH_HASH") != nullptr);
    auto dispatch_one = [&](const uint8_t* body, size_t blen) -> int {
      if (blen < 4) { fprintf(stderr, "[server] runt sub-request: %zu B\n", blen); return -1; }
      uint32_t m; memcpy(&m, body, 4);
      if (g_disp_hash) {
        // Wire-hash gate (P1 RUNG-2): FNV-1a of the exact dispatched body.
        // Logged for BOTH fused sub-bodies and standalone requests, so the
        // (magic,len,hash) sequence must be identical fusion-on vs fusion-off.
        uint64_t h = 1469598103934665603ull;
        for (size_t k = 0; k < blen; k++) { h ^= body[k]; h *= 1099511628211ull; }
        fprintf(stderr, "[disp] magic=0x%08x len=%zu fnv=0x%016llx\n",
                m, blen, (unsigned long long)h);
      }
      if (m == MAGIC_V2 || m == MAGIC_V2G || m == MAGIC_V2S || m == MAGIC_V2GS) {
        return process_request(platform, banks, body, blen, label_base, response_fd);
      } else if (m == MAGIC_LOAD) {
        return process_load_weights(platform, banks, handles, body, blen, response_fd);
      } else if (m == MAGIC_MM3D) {
        int r = 2;
        if (desc_serve_enabled())
          r = process_matmul_desc(platform, banks, handles, body, blen, label_base, response_fd);
        if (r == 2)
          r = process_matmul_handle(platform, banks, handles, body, blen, label_base, response_fd);
        return r;
      }
      fprintf(stderr, "[server] unknown magic 0x%x\n", m);
      return -2;
    };
    if (magic == MAGIC_FUSE) {
      // P1 RUNG-2 request fusion: [MAGIC_FUSE][u32 K] then K×([u32 sublen][subbody]).
      // Loop the UNCHANGED per-request dispatch; each sub writes its own response
      // to response_fd in order, so the client reads K concatenated sub-responses.
      rc = 0;
      if (req_len < 8) { fprintf(stderr, "[server] runt FUSE frame: %u B\n", req_len); return 6; }
      uint32_t K = 0; memcpy(&K, req_buf.data() + 4, 4);
      // [2026-08-04 CROSS-REQUEST BATCHING] PIM_DESC_XBATCH=1: a FUSE frame
      // whose sub-bodies are ALL MAGIC_MM3D is served by ONE desc-serve
      // super-session per bank (process_matmul_desc_batch) — the session
      // machinery (entry/exit drains, per-bank enter, rewrite mode switches)
      // is paid once per FRAME instead of once per request. rc==2 (any sub
      // ineligible / probe fail) falls through to the unchanged per-sub loop.
      // Default OFF: behavior is byte-identical unless the env is set.
      static const bool s_xbatch = []{
        const char* v = getenv("PIM_DESC_XBATCH"); return v && atoi(v) > 0; }();
      bool xbatch_served = false;
      if (s_xbatch && desc_serve_enabled() && K > 0) {
        std::vector<std::pair<const uint8_t*, size_t>> xsubs;
        bool all_mm3d = true;
        size_t soff = 8;
        for (uint32_t i = 0; i < K && all_mm3d; i++) {
          if (soff + 4 > req_len) { all_mm3d = false; break; }
          uint32_t sl = 0; memcpy(&sl, req_buf.data() + soff, 4); soff += 4;
          if (soff + sl > req_len || sl < 4) { all_mm3d = false; break; }
          uint32_t sm = 0; memcpy(&sm, req_buf.data() + soff, 4);
          if (sm != MAGIC_MM3D) all_mm3d = false;
          else xsubs.emplace_back(req_buf.data() + soff, (size_t)sl);
          soff += sl;
        }
        if (all_mm3d && xsubs.size() == K) {
          int rx = process_matmul_desc_batch(platform, banks, handles, xsubs,
                                             label_base, response_fd);
          if (rx == 0) xbatch_served = true;       // frame fully served + responded
          else if (rx < 0) { fprintf(stderr, "[server] desc-xbatch hard error\n"); return 7; }
          /* rx == 2: fall through to the unchanged per-sub loop */
        }
      }
      size_t foff = 8;
      for (uint32_t i = 0; !xbatch_served && i < K; i++) {
        if (foff + 4 > req_len) { fprintf(stderr, "[server] FUSE sublen overrun i=%u\n", i); return 6; }
        uint32_t sublen = 0; memcpy(&sublen, req_buf.data() + foff, 4); foff += 4;
        if (foff + sublen > req_len) { fprintf(stderr, "[server] FUSE subbody overrun i=%u sublen=%u\n", i, sublen); return 6; }
        int r = dispatch_one(req_buf.data() + foff, sublen);
        foff += sublen;
        if (r != 0) { rc = 6; break; }
      }
    } else if (magic == MAGIC_CONFIG) {
      // [#65] Runtime config-update. Dispatched BETWEEN complete requests (the
      // loop is single-threaded), so it never races an in-flight matmul. The
      // handler writes its own response to response_fd and may mutate `banks` /
      // `handles`. A hard error (bad frame) tears the session down; a soft
      // reject is signalled inside the response status word.
      int cr = handle_config_request(&platform, banks, handles,
                                     req_buf.data(), req_len, response_fd);
      rc = (cr < 0) ? 6 : 0;
    } else if (magic == MAGIC_V2 || magic == MAGIC_V2G || magic == MAGIC_V2S
        || magic == MAGIC_V2GS || magic == MAGIC_LOAD || magic == MAGIC_MM3D) {
      rc = dispatch_one(req_buf.data(), req_len);
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

  // ---- Task #50 PER-SESSION COUNTER GATE (desc-serve ship blocker) ----
  // Session teardown ONLY (the request loop has drained; no request is in
  // flight; nothing here touches the timed request path). Read the black-box
  // recorder ONCE and fail the session if the fetch-corruption skip counters
  // (word2 = {malformed_skips[31:16], parity_skips[15:0]}) are nonzero. A
  // nonzero count = genuine IMEM storage/read corruption survived as a skip
  // behind the seam this session -> the responses already returned are SUSPECT
  // (there is no per-response status field to retroactively flag, so the loud
  // log + nonzero process exit ARE the client-visible signal). Default ON when
  // PIM_DESC_SERVE=1; PIM_COUNTER_GATE overrides. Skipped on the sim backend
  // (no recorder) and when the feature is off.
  int gate_exit = 0;
  if (counter_gate_on() && !sim_backend) {
    // Close any lingering stream session so the out-of-band read is clean
    // (each request already closes its own session; this is belt-and-braces).
    if (g_stream_session) { platform.stream_stop(); g_stream_session = false; }
    uint32_t g_malf = 0, g_par = 0, g_wdog = 0, g_btag = 0;
    // Bound the read to 1500 ms so the server always exits inside the client's
    // 2 s proc.wait() window (pim_linear _cleanup) — no SIGKILL-mid-read race.
    bool ok = recorder_read_counters(platform, g_malf, g_par, g_wdog, g_btag, 1500);
    if (!ok) {
      fprintf(stderr, "[PIM_COUNTER_GATE] INCONCLUSIVE: recorder read FAILED at "
              "teardown (c2h may be wedged) — cannot verify skip counters. NOT "
              "failing the session on a read miss (a wedge is its own signal).\n");
    } else if (g_par > 0 || g_malf > 0) {
      fprintf(stderr,
              "\n**** PIM_COUNTER_GATE FAIL **** malformed_skips=%u parity_skips=%u "
              "(build_tag=0x%08X wdog_fires=%u) — genuine IMEM fetch-corruption "
              "skips occurred THIS SESSION; every response returned is SUSPECT. "
              "Re-run; do NOT ship these tokens.\n\n", g_malf, g_par, g_btag, g_wdog);
      gate_exit = 42;
    } else {
      fprintf(stderr, "[PIM_COUNTER_GATE] PASS: malformed_skips=0 parity_skips=0 "
              "(build_tag=0x%08X wdog_fires=%u) — no fetch-corruption skips this "
              "session; responses clean by the seam discriminator.\n", g_btag, g_wdog);
    }
  }
  _exit(gate_exit);
}
