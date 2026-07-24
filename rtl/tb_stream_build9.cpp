// build9 streaming-fetch TB — real frontend + fetch_stage + pre_decode +
// behavioral IMEM. Drives the XDMA producer (h2c) and models execute
// completion (softmc_fin fired a fixed delay after fetch_stage reports
// softmc_end), and checks the dispatched instruction stream equals the
// programs streamed, in order, with no corruption at the ping-pong swap.
//
//   A. STREAM OFF, one program        — legacy sanity
//   B. STREAM OFF, three programs      — legacy back-to-back baseline
//   C. STREAM ON, eager producer       — the ping-pong (N+1 loads while N
//                                        runs; swap on fin)
//   D. STREAM ON, slow producer         — next bank not ready at END →
//                                        degrade to legacy, still correct
//   E. STREAM ON + a SET word between programs — must not corrupt stream
// Pass = every dispatched stream equals the streamed programs, in order.
#include <verilated.h>
#include "Vfrontend_fetch_top.h"
#include <cstdio>
#include <cstdint>
#include <vector>
#include <string>
using namespace std;

static Vfrontend_fetch_top* top;
static int fails = 0;
static vector<uint32_t> dispatched;
static int fin_delay_ctr = -1;
static int fin_delay_next = 40;   // F overrides per program
#include <deque>
static std::deque<int> fin_q;      // per-program fin delays (send order)
static bool fin_armed = false;     // one fin per executed program

static const int      INSTR_WIDTH = 64;
static const uint64_t END_WORD    = 0ull;    // is_end = &~instr
static uint64_t branch_word(uint32_t tag) {  // bit62 = BRANCH_OFFSET
  return (((uint64_t)(tag & 0x00FFFFFF)) | (1ull << 40)) | (1ull << 62);
}
static uint64_t instr_word(uint32_t tag) {   // nonzero, no INFO/BRANCH/ctrl bits
  return ((uint64_t)(tag & 0x00FFFFFF)) | (1ull << 40);
}
static uint32_t tag_of(uint64_t w) { return (uint32_t)(w & 0x00FFFFFF); }

// tdata is 256b across 8 uint32 words. bits 0..63 = dword0/1 (the
// instruction / payload); control flags live at bit INSTR_WIDTH+N, i.e.
// dword2 bit N (N=0..11).
static void set_tdata(uint64_t lo64, uint32_t ctrl_dword2) {
  top->h2c_tdata_0[0] = (uint32_t)lo64;
  top->h2c_tdata_0[1] = (uint32_t)(lo64 >> 32);
  top->h2c_tdata_0[2] = ctrl_dword2;
  for (int i = 3; i < 8; i++) top->h2c_tdata_0[i] = 0;
}

// prog_done: set the moment a program's END is dispatched; while set, the
// fetch stage re-loops pc=0..end (legacy behavior — the execute pipeline
// ignores post-END re-fetches) so we must NOT count those. Cleared when
// softmc_fin advances execution to the next program. This makes the TB
// observe exactly ONE clean pass per executed program — the sequence the
// DDR/execute side actually consumes.
static bool prog_done = false;
static int  prev_exec_bank = -1;
// branch execute-model: each dispatched BR resolves after BR_DELAY;
// taken (target=g_br_target) g_br_takes times per program pass-window,
// then not-taken (target = br_pc+1). Post-END sightings resolve TOO —
// deliberately, to model the silicon stale-resolve hazard (E14).
static int  g_br_delay   = 12;
static int  g_br_target  = 1;     // loop start address
static int  g_br_takes   = 2;     // taken twice then fall through
static int  g_br_ctr     = -1;    // countdown to pulse
static int  g_br_taken_n = 0;
static int  g_br_fallpc  = 4;     // BR at addr 3 -> fallthrough 4
static bool g_br_stale = false;
static bool g_trace = false;
static int  g_cyc = 0;
static uint32_t g_last_sig = 0xFFFFFFFFu;
static void tick_obs() {
  top->clk = 0; top->eval();
  top->clk = 1; top->eval();
  // A new PROGRAM begins when the exec bank flips (build9 alternates the
  // bank on every program start — both the INIT_MEM path and the swap).
  // Clear prog_done there so the next program's words are recorded. In
  // eager streaming the swap lands well within the 40-cycle fin delay,
  // so keying the boundary on fin (as a first cut did) drops the whole
  // next program — the bank flip is the correct boundary.
  int eb = top->obs_exec_bank;
  if (prev_exec_bank >= 0 && eb != prev_exec_bank) {
    prog_done = false;
    fin_armed = false;   // new program began — its fin may arm
  }
  prev_exec_bank = eb;
  if (g_trace) {
    g_cyc++;
    uint32_t sig = (top->obs_state << 24) | (eb << 20) |
                   (top->obs_loaded << 16) | (top->obs_swap_pending << 12) |
                   (top->obs_fetch_hold << 8) | (top->obs_tready << 4) |
                   (top->obs_instr_valid);
    if (sig != g_last_sig) {
      fprintf(stderr, "T%5d st=%d eb=%d ld=%d sp=%d fh=%d trdy=%d iv=%d instr=%x\n",
              g_cyc, top->obs_state, eb, top->obs_loaded,
              top->obs_swap_pending, top->obs_fetch_hold, top->obs_tready,
              top->obs_instr_valid, (unsigned)top->obs_instr);
      g_last_sig = sig;
    }
  }
  // fetch_stage DEASSERTS instr_valid for the END word, so END arrives
  // only as obs_softmc_end. Record real instr dispatches (one clean pass
  // per program), stop at END.
  if (top->obs_instr_valid && !prog_done) {
    uint64_t w = (uint64_t)top->obs_instr;
    if (w != END_WORD) dispatched.push_back(tag_of(w));
  }
  // branch model: sight EVERY dispatched branch (even post-END ones —
  // prog_done only gates counting) and schedule a resolve.
  if (top->obs_instr_valid) {
    uint64_t w = (uint64_t)top->obs_instr;
    if (w & (1ull << 62)) {
      if (g_br_ctr < 0) { g_br_ctr = g_br_delay; g_br_stale = prog_done; }
    }
  }
  top->tb_br_resolve = 0;
  if (g_br_ctr > 0) g_br_ctr--;
  else if (g_br_ctr == 0) {
    top->tb_br_resolve = 1;
    if (g_br_stale) {
      top->tb_br_target = g_br_target;   // poison: legit counter untouched
    } else if (g_br_taken_n < g_br_takes) {
      top->tb_br_target = g_br_target; g_br_taken_n++;
    } else {
      top->tb_br_target = g_br_fallpc; g_br_taken_n = 0;
    }
    g_br_ctr = -1;
  }
  if (top->obs_softmc_end) prog_done = true;
  top->softmc_fin = 0;
  if (top->obs_softmc_end && fin_delay_ctr < 0 && !fin_armed) {
    if (!fin_q.empty()) { fin_delay_ctr = fin_q.front(); fin_q.pop_front(); }
    else fin_delay_ctr = fin_delay_next;
    fin_armed = true;   // re-loop ENDs must not pop the next program's fin
  }
  if (fin_delay_ctr > 0) fin_delay_ctr--;
  else if (fin_delay_ctr == 0) { top->softmc_fin = 1; fin_delay_ctr = -1; }
}

// drive one beat until it fires (tready), keeping the execute model live.
static bool beat(uint64_t lo64, uint32_t ctrl_dword2, bool last, int maxc = 6000) {
  for (int c = 0; c < maxc; c++) {
    set_tdata(lo64, ctrl_dword2);
    top->h2c_tlast_0 = last ? 1 : 0;
    top->h2c_tvalid_0 = 1;
    top->eval();
    bool ready = top->h2c_tready_0;
    tick_obs();
    if (ready) {
      if (g_trace)
        fprintf(stderr, "T%5d H2C fire lo=%llx last=%d\n", g_cyc,
                (unsigned long long)lo64, last ? 1 : 0);
      top->h2c_tvalid_0 = 0; top->h2c_tlast_0 = 0; top->eval(); return true;
    }
  }
  top->h2c_tvalid_0 = 0; top->h2c_tlast_0 = 0;
  return false;
}
static void idle(int n) {
  top->h2c_tvalid_0 = 0; top->h2c_tlast_0 = 0;
  for (int i = 0; i < n; i++) tick_obs();
}

// control word: flag at dword2 bit `plusbit`, payload in lo64. Not tlast.
static bool send_ctrl(int plusbit, uint64_t payload_lo) {
  return beat(payload_lo, (uint32_t)(1u << plusbit), false);
}
static bool stream_en(bool on) { return send_ctrl(11, on ? 1 : 0); }   // +11 STREAM_EN

static bool stream_program(vector<uint32_t>& expect, uint32_t base, int n) {
  for (int i = 0; i < n; i++) {
    uint64_t w = instr_word(base + i);
    expect.push_back(tag_of(w));
    if (!beat(w, 0, false)) return false;   // instruction word (ctrl=0)
  }
  return beat(END_WORD, 0, true);           // END + tlast
}

static void check(const char* name, bool ok) {
  printf("  [%s] %s\n", ok ? "PASS" : "FAIL", name);
  if (!ok) fails++;
}
static bool seq_ok(const vector<uint32_t>& exp) {
  bool ok = dispatched.size() == exp.size();
  if (ok) for (size_t i = 0; i < exp.size(); i++) if (dispatched[i] != exp[i]) { ok = false; break; }
  if (!ok) {
    printf("    exp(%zu):", exp.size());
    for (auto v : exp) printf(" %x", v);
    printf("\n    got(%zu):", dispatched.size());
    for (auto v : dispatched) printf(" %x", v);
    printf("\n");
  }
  return ok;
}
static void hard_reset() {
  dispatched.clear(); fin_delay_ctr = -1; prog_done = false; prev_exec_bank = -1;
  fin_q.clear(); fin_armed = false;
  g_br_ctr = -1; g_br_taken_n = 0; g_br_stale = false;
  top->tb_br_resolve = 0; top->tb_br_target = 0;
  top->rst = 1; top->softmc_fin = 0; top->h2c_tvalid_0 = 0;
  top->h2c_tlast_0 = 0; top->init_calib_complete = 1;
  for (int i = 0; i < 8; i++) tick_obs();
  top->rst = 0;
  for (int i = 0; i < 6; i++) tick_obs();
}
static void drain_until(size_t target, int budget = 40000) {
  for (int c = 0; c < budget && dispatched.size() < target; c++) idle(1);
}


// ---- traced C: dump the swap sequence to stderr ----
static void run_C_traced() {
  hard_reset();
  bool ok = stream_en(true); idle(4);
  vector<uint32_t> exp;
  fprintf(stderr,"cyc st eb ld sp fh trdy send  instr iv send_end\n");
  for (int p = 0; p < 3 && ok; p++) {
    for (int i=0;i<5;i++){ uint64_t w=instr_word((0x300+p*0x10)+i); exp.push_back(tag_of(w)); ok=beat(w,0,false);}
    ok = ok && beat(END_WORD,0,true);
  }
  for (int c=0;c<160;c++){
    idle(1);
    fprintf(stderr,"%3d  %d  %d  %d  %d  %d   %d    -   %6x %d\n",
      c, top->obs_state, top->obs_exec_bank, top->obs_loaded,
      top->obs_swap_pending, top->obs_fetch_hold, top->obs_tready,
      (unsigned)top->obs_instr, top->obs_instr_valid);
  }
  fprintf(stderr,"C got:"); for(auto v:dispatched) fprintf(stderr," %x",v); fprintf(stderr,"\n");
}

int main(int argc, char** argv) {
  Verilated::commandArgs(argc, argv);
  top = new Vfrontend_fetch_top;
  printf("build9 streaming-fetch TB (real frontend+fetch)\n");
  if (argc>1 && std::string(argv[1])=="trace") { run_C_traced(); return 0; }

  // A: stream off, single program
  { hard_reset();
    vector<uint32_t> exp; bool ok = stream_program(exp, 0x100, 6);
    drain_until(exp.size());
    check("A stream-off single program in order", ok && seq_ok(exp)); }

  // B: stream off, three programs (legacy: fin between each)
  { hard_reset();
    vector<uint32_t> exp; bool ok = true;
    for (int p = 0; p < 3 && ok; p++) {
      ok = stream_program(exp, 0x200 + p*0x10, 5);
      drain_until(exp.size()); idle(60);
    }
    check("B stream-off three programs in order", ok && seq_ok(exp)); }

  // C: stream on, eager producer (ping-pong)
  { hard_reset();
    bool ok = stream_en(true); idle(4);
    vector<uint32_t> exp;
    for (int p = 0; p < 3 && ok; p++) ok = stream_program(exp, 0x300 + p*0x10, 5);
    drain_until(exp.size());
    check("C stream-on eager three programs (ping-pong) in order", ok && seq_ok(exp)); }

  // D: stream on, slow producer (degrade to legacy)
  { hard_reset();
    bool ok = stream_en(true); idle(4);
    vector<uint32_t> exp;
    for (int p = 0; p < 3 && ok; p++) {
      ok = stream_program(exp, 0x400 + p*0x10, 5);
      idle(300);   // next bank not ready at END → legacy path
    }
    drain_until(exp.size());
    check("D stream-on slow producer degrades, still correct", ok && seq_ok(exp)); }

  // E: stream on, a SET word interleaved between programs
  { hard_reset();
    bool ok = stream_en(true); idle(4);
    vector<uint32_t> exp;
    ok = ok && stream_program(exp, 0x500, 5);
    ok = ok && send_ctrl(7, 0);   // +7 SET SEG_POP mid-stream, no payload
    ok = ok && stream_program(exp, 0x510, 5);
    drain_until(exp.size());
    check("E stream-on SET word between programs, stream intact", ok && seq_ok(exp)); }

  // helper for F/G: replay an explicit timing script; returns pass/fail
  // and (on fail) prints the script for a hand-authored regression.
  struct FProg { int len; int fin; int pause; int setw; };
  auto run_script = [&](const std::vector<FProg>& sc, uint32_t base,
                        bool verbose) -> bool {
    hard_reset();
    bool ok = stream_en(true); idle(4);
    std::vector<uint32_t> exp;
    for (size_t q = 0; q < sc.size() && ok; q++) {
      fin_q.push_back(sc[q].fin);
      if (sc[q].setw) ok = ok && send_ctrl(7, 0);
      ok = ok && stream_program(exp, base + (uint32_t)q * 0x40, sc[q].len);
      if (sc[q].pause) idle(sc[q].pause);
    }
    drain_until(exp.size(), 4000000);
    bool pass = ok && dispatched.size() == exp.size();
    if (pass) for (size_t i = 0; i < exp.size(); i++)
      if (dispatched[i] != exp[i]) { pass = false; break; }
    if (!pass && verbose) {
      seq_ok(exp);   // dumps exp vs got
      printf("    SHRUNK SCRIPT (%zu progs): {len,fin,pause,set}\n", sc.size());
      for (size_t q = 0; q < sc.size(); q++)
        printf("      {%d,%d,%d,%d},\n", sc[q].len, sc[q].fin,
               sc[q].pause, sc[q].setw);
      printf("    sent=%zu got=%zu\n", exp.size(), dispatched.size());
    }
    return pass;
  };

  // F: randomized-timing stress — the production lost-record hunt
  // (2026-07-24: depth-1 exec-overlap stalled the full model ~1/4k
  // requests; a lost program = silently missing record). Production
  // regime = program N+1 FULLY loaded long before N's fin (fin delays
  // ~78k cycles on silicon); C's fixed 40-cycle fin barely enters it.
  // Random program lengths, random fin delays sweeping the whole
  // execution window incl. the END/fin/swap seam, occasional SET words
  // and producer pauses. Deterministic LCG; on FAIL prints seed +
  // position for shrinking.
  { int F_PROGS = 4000;
    if (const char* e = getenv("F_PROGS")) F_PROGS = atoi(e);
    int seed0 = 1, seeds = 3;
    if (const char* e = getenv("F_SEED"))  seed0 = atoi(e);
    if (const char* e = getenv("F_SEEDS")) seeds = atoi(e);
    for (int sd = seed0; sd < seed0 + seeds; sd++) {
      hard_reset();
      uint64_t rng = (uint64_t)sd * 2654435761u + 12345u;
      auto rnd = [&](int lo, int hi){ rng = rng*6364136223846793005ull+1442695040888963407ull;
                                      return lo + (int)((rng>>33) % (uint32_t)(hi-lo+1)); };
      bool ok = stream_en(true); idle(4);
      vector<uint32_t> exp;
      std::vector<FProg> hist;
      int fail_at = -1;
      for (int p = 0; p < F_PROGS && ok; p++) {
        FProg e;
        e.fin   = rnd(10, 1200);
        e.len   = rnd(3, 40);
        if (rnd(0,6) == 0) e.len = rnd(1,2);
        e.setw  = (rnd(0,9) == 0) ? 1 : 0;
        e.pause = (rnd(0,4) == 0) ? rnd(1, 900) : 0;
        fin_q.push_back(e.fin);
        if (e.setw) ok = ok && send_ctrl(7, 0);
        ok = ok && stream_program(exp, 0xF0000 + p*0x40, e.len);
        if (e.pause) idle(e.pause);
        hist.push_back(e);
        if ((p & 63) == 63) {
          drain_until(exp.size(), 4000000);
          if (dispatched.size() != exp.size()) { fail_at = p; break; }
        }
      }
      if (fail_at >= 0) {
        for (int w = 4; w <= 64 && w <= (int)hist.size(); w *= 2) {
          std::vector<FProg> sc(hist.end() - w, hist.end());
          if (!run_script(sc, 0xA0000, w <= 16)) {
            printf("    seed=%d: WINDOW OF %d REPRODUCES\n", sd, w);
            break;
          }
          if (w == 64) printf("    seed=%d: no trailing window <=64 "
                              "reproduces (history-dependent)\n", sd);
        }
        hard_reset();
      }
      drain_until(exp.size(), 4000000);
      bool pass = ok && dispatched.size() == exp.size();
      if (pass) for (size_t i = 0; i < exp.size(); i++)
        if (dispatched[i] != exp[i]) { pass = false; fail_at = (int)(i/20); break; }
      char nm[128];
      snprintf(nm, sizeof nm, "F randomized stress seed=%d progs=%d%s%s",
               sd, F_PROGS,
               pass ? "" : " LOST/CORRUPT",
               fail_at >= 0 ? " (see fail_at)" : "");
      if (!pass)
        printf("    seed=%d fail_at~prog %d: sent=%zu got=%zu\n",
               sd, fail_at, exp.size(), dispatched.size());
      check(nm, pass);
    }
  }

  // H: LEGACY (stream OFF) with LONG fin — does the phase bug predate
  // build-9? The original TB used fin=40 with 5-word programs; if the
  // re-loop period divides 40, legacy scenarios froze at pc~0 by
  // COINCIDENCE. Long fin sweeps the freeze phase.
  { hard_reset();
    vector<uint32_t> exp; bool ok = true;
    for (int p = 0; p < 3 && ok; p++) {
      fin_q.push_back(890 + p * 121);   // vary phase
      ok = stream_program(exp, 0x800 + p*0x10, 5);
      drain_until(exp.size(), 4000000); idle(60);
    }
    check("H stream-OFF legacy with long/varied fin", ok && seq_ok(exp)); }

  // G: traced replay of the shrunk 4-program repro (env G_TRACE=1)
  if (getenv("G_TRACE")) {
    std::vector<FProg> sc = { {23,1002,0,0}, {8,331,288,0},
                              {2,277,0,0}, {16,11,0,1} };
    g_trace = true; g_cyc = 0; g_last_sig = 0xFFFFFFFFu;
    bool pass = run_script(sc, 0xA0000, true);
    g_trace = false;
    check("G traced 4-program repro", pass);
  }

  // I: STREAMED BRANCH-LOOP programs — the E14 silicon shape the TB
  // never covered (branches were tied off). Program: [A][B][C][BR->B]
  // [D][END], BR taken twice (3 body passes). Streamed back-to-back
  // with long fins; the post-END re-loop re-dispatches the BR and the
  // execute model resolves it — the stale-resolve hazard. build-11
  // fetch pairs it with the NEXT program's first branch (corruption);
  // build-12 (br_outstanding fence) must drop it.
  { hard_reset();
    bool ok = stream_en(true); idle(4);
    vector<uint32_t> exp;
    for (int p = 0; p < 4 && ok; p++) {
      uint32_t base = 0xB0000 + p * 0x100;
      fin_q.push_back(700 + 137 * p);
      // send: A(base+0) B(+1) C(+2) BR D(+4) END
      ok = ok && beat(instr_word(base + 0), 0, false);
      ok = ok && beat(instr_word(base + 1), 0, false);
      ok = ok && beat(instr_word(base + 2), 0, false);
      ok = ok && beat(branch_word(base + 3), 0, false);
      ok = ok && beat(instr_word(base + 4), 0, false);
      ok = ok && beat(END_WORD, 0, true);
      // expected dispatch: A B C BR | B C BR | B C BR | D
      exp.push_back(base + 0);
      for (int pass = 0; pass < 3; pass++) {
        if (pass) { exp.push_back(base + 1); }
        else      { exp.push_back(base + 1); }
        exp.push_back(base + 2);
        exp.push_back(base + 3);
      }
      exp.push_back(base + 4);
    }
    drain_until(exp.size(), 6000000);
    check("I streamed branch-loops (E14 shape)", ok && seq_ok(exp));
  }

  printf("build9 stream TB: %s (%d fails)\n", fails ? "FAIL" : "ALL_PASS", fails);
  delete top;
  return fails ? 1 : 0;
}
