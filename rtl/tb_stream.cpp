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

static const int      INSTR_WIDTH = 64;
static const uint64_t END_WORD    = 0ull;    // is_end = &~instr
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
  if (prev_exec_bank >= 0 && eb != prev_exec_bank) prog_done = false;
  prev_exec_bank = eb;
  // fetch_stage DEASSERTS instr_valid for the END word, so END arrives
  // only as obs_softmc_end. Record real instr dispatches (one clean pass
  // per program), stop at END.
  if (top->obs_instr_valid && !prog_done) {
    uint64_t w = (uint64_t)top->obs_instr;
    if (w != END_WORD) dispatched.push_back(tag_of(w));
  }
  if (top->obs_softmc_end) prog_done = true;
  top->softmc_fin = 0;
  if (top->obs_softmc_end && fin_delay_ctr < 0) fin_delay_ctr = 40;
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
    if (ready) { top->h2c_tvalid_0 = 0; top->h2c_tlast_0 = 0; top->eval(); return true; }
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

  printf("build9 stream TB: %s (%d fails)\n", fails ? "FAIL" : "ALL_PASS", fails);
  delete top;
  return fails ? 1 : 0;
}
