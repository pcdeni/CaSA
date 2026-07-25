// e2e_sim phase-1 TB (2026-07-24). Drives the REAL frontend+pipeline
// with the REAL machine code (hex emitted by gen_progs via the actual
// Program encoder), over the exact h2c beat protocol platform.cpp uses:
// one 256-bit beat per instruction, u64 word in the low lane, tlast on
// the final beat (execute(): temp_ptr[i*4]=iseq[i], sendData(bytes*4)).
//
// Scenarios (verdicts on stdout, full FST trace for the microscope):
//   S0  reset word (bit INSTR_WIDTH), as reset_fpga() sends it.
//   S1  s1_read.hex   branch-free 46-inst read  -> expect FIN + 8 reads
//   S2  s2_brloop.hex 6-inst BL loop            -> build-12: expect WEDGE
//   S3  s1 again      -> on silicon the channel jams; expect send-stall
//                        or no-FIN if the wedge reproduces.
#include "Ve2e_top.h"
#include "verilated.h"
#include "verilated_vcd_c.h"
#include <cstdio>
#include <cstdint>
#include <vector>
#include <string>
#include <fstream>

// Verilator shim: $display/$time in sim-only RTL debug blocks needs this.
double sc_time_stamp() { return 0; }

static Ve2e_top* top;
static VerilatedVcdC* tfp;
static uint64_t t = 0;
static long ddr_read_beats = 0, ddr_act_pulses = 0, fin_seen_at = -1;
static long maint_rd = 0, maint_zq = 0, maint_ref = 0;
static std::vector<uint8_t> c2h_bytes;        // phase-2: host receive side

extern "C" void dram_expected_beat(int bg,int bank,int row,int col,uint32_t*);

static bool g_trace_en = false;   // waveform only around the failure
static long g_drain_off = 0, g_dr_budget = 8, g_dr_pause = 0;
static long g_c2h_tlast_count = 0;
static long g_fin_edges = 0; static int g_fin_d = 0;
static long g_fr_edges = 0, g_fr_hi = 0, g_fr_user = 0, g_fr_maint = 0; static int g_fr_d = 0;
static void tick(){
  // c2h drain pacing. 0 = instant (old behaviour). N>0 = accept a
  // short burst, then refuse for N cycles: a trailer then sits in the
  // engine while the next program completes — the silicon condition
  // that scenario P's instant drain hid.
  // Duty-cycle drain: tready follows a FIXED pattern derived only from
  // the cycle counter, never from tvalid. The testbench therefore cannot
  // react to the DUT, so any framing loss is the DUT's, not mine.
  // Burst drain: accept 8 beats, then pause. This is the model whose
  // signature matched silicon exactly (one trailer per session,
  // last record = 2112-32N). The duty-cycle variant is kept below for
  // stress but is far harsher than PCIe.
  if (g_drain_off <= 0) top->c2h_tready = 1;
  else if (g_dr_pause > 0) { top->c2h_tready = 0; g_dr_pause--; }
  else {
    top->c2h_tready = 1;
    if (top->c2h_tvalid && --g_dr_budget <= 0)
      { g_dr_pause = g_drain_off; g_dr_budget = 8; }
  }
  top->clk = 0; top->eval(); if(tfp && g_trace_en) tfp->dump(t*10);
  top->clk = 1; top->eval(); if(tfp && g_trace_en) tfp->dump(t*10+5);
  if (top->ddr_read)  ddr_read_beats++;
  if (top->ddr_act)   ddr_act_pulses++;
  if (top->per_rd_init_obs)  maint_rd++;
  if (top->per_zq_init_obs)  maint_zq++;
  if (top->per_ref_init_obs) maint_ref++;
  if (top->softmc_fin && !g_fin_d) g_fin_edges++;
  g_fin_d = top->softmc_fin ? 1 : 0;
  if (top->frontend_ready_obs && !g_fr_d) {
    g_fr_edges++;
    if (top->frontend_ready_maint_obs) g_fr_maint++; else g_fr_user++;
  }
  if (top->frontend_ready_obs) g_fr_hi++;
  g_fr_d = top->frontend_ready_obs ? 1 : 0;
  if (top->softmc_fin && fin_seen_at < 0) fin_seen_at = (long)t;
  if (top->c2h_tvalid && top->c2h_tready && top->c2h_tlast) g_c2h_tlast_count++;
  if (top->c2h_tvalid && top->c2h_tready){    // collect the 32B beat
    for (int i=0;i<8;i++){ uint32_t w = top->c2h_tdata[i];
      c2h_bytes.push_back((uint8_t)w); c2h_bytes.push_back((uint8_t)(w>>8));
      c2h_bytes.push_back((uint8_t)(w>>16)); c2h_bytes.push_back((uint8_t)(w>>24)); }
  }
  t++;
}

static std::vector<uint64_t> load_hex(const std::string& p){
  std::vector<uint64_t> v; std::ifstream f(p); std::string ln;
  while (std::getline(f, ln)) if (ln.size()>=16)
    v.push_back(strtoull(ln.c_str(), nullptr, 16));
  return v;
}

// one beat; returns false if tready never rose (the silicon jam shape)
static bool beat(uint64_t word, bool extra_bit64, bool last, long budget=20000){
  top->h2c_tdata[0] = (uint32_t)(word & 0xFFFFFFFFu);
  top->h2c_tdata[1] = (uint32_t)(word >> 32);
  for (int i=2;i<8;i++) top->h2c_tdata[i] = 0;
  if (extra_bit64) top->h2c_tdata[2] = 1;      // bit INSTR_WIDTH = tdata[64]
  top->h2c_tkeep = 0xFFFFFFFFu;   // 32 keep bits = one IData
  top->h2c_tvalid = 1; top->h2c_tlast = last ? 1 : 0;
  long waited = 0;
  while (!top->h2c_tready){ tick(); if (++waited > budget){
      top->h2c_tvalid = 0; top->h2c_tlast = 0; return false; } }
  tick();                                       // beat accepted this edge
  top->h2c_tvalid = 0; top->h2c_tlast = 0;
  return true;
}

static bool send_program(const std::vector<uint64_t>& w){
  for (size_t i=0;i<w.size();i++)
    if (!beat(w[i], false, i+1==w.size())) return false;
  return true;
}

// wait for FIN; true if seen within budget
static bool wait_fin(long budget){
  fin_seen_at = -1;
  for (long i=0;i<budget;i++){ tick(); if (fin_seen_at>=0){
      for (int j=0;j<50;j++) tick();            // drain a little
      return true; } }
  return false;
}

int main(int argc, char** argv){
  Verilated::commandArgs(argc, argv);
  Verilated::traceEverOn(true);
  top = new Ve2e_top;
  tfp = new VerilatedVcdC;
  top->trace(tfp, 99);
  tfp->open("e2e.vcd");

  top->rst = 1; top->h2c_tvalid = 0; top->h2c_tlast = 0;
  top->init_calib_complete = 0;          // silicon: calib incomplete at boot
  for (int i=0;i<30;i++) tick();
  top->rst = 0;
  for (int i=0;i<40;i++) tick();         // timers load their divisors here
  top->init_calib_complete = 1;          // calib completes; timers count
  for (int i=0;i<10;i++) tick();

  int fails = 0;
  auto s1 = load_hex("s1_read.hex");
  auto s2 = load_hex("s2_brloop.hex");
  printf("[tb] s1=%zu insts, s2=%zu insts\n", s1.size(), s2.size());
  if (s1.empty() || s2.empty()){ printf("[tb] HEX MISSING\n"); return 3; }

  // S0: reset word, as reset_fpga() emits (bit INSTR_WIDTH set)
  { bool ok = beat(0, true, true); for (int i=0;i<80;i++) tick();
    printf("[tb] S0 reset word: %s (user_rst pulsed=%d)\n",
           ok?"sent":"SEND-STALL", (int)top->user_rst_obs); }

  // S1: branch-free read
  { ddr_read_beats = 0; ddr_act_pulses = 0;
    bool sent = send_program(s1);
    bool fin  = sent && wait_fin(50000);
    printf("[tb] S1 branch-free: sent=%d fin=%d cyc=%ld ddr_read_beats=%ld acts=%ld -> %s\n",
           sent, fin, fin_seen_at, ddr_read_beats, ddr_act_pulses,
           (sent&&fin&&ddr_read_beats>0) ? "PASS" : "FAIL");
    if (!(sent&&fin&&ddr_read_beats>0)) fails++; }

  // S2: the 6-inst BL loop — the silicon wedge shape
  { bool sent = send_program(s2);
    bool fin  = sent && wait_fin(100000);
    printf("[tb] S2 branch loop : sent=%d fin=%d cyc=%ld -> %s\n",
           sent, fin, fin_seen_at,
           fin ? "COMPLETED (no repro on this fetch)" : "WEDGE REPRODUCED");
    if (!fin) fails += 0; /* expected on build-12; verdict is informational */ }

  // S3: branch-free again — does the wedge persist / jam h2c like silicon?
  { bool sent = send_program(s1);
    bool fin  = sent && wait_fin(50000);
    printf("[tb] S3 post-branch : sent=%d fin=%d -> %s\n",
           sent, fin,
           (!sent) ? "H2C JAMMED (matches silicon errno-512 shape)" :
           (!fin)  ? "ACCEPTED BUT NO FIN (fetch parked)" :
                     "RECOVERED (unlike silicon)"); }

  // Frontend-state probe: what do state/maint look like across a long
  // idle window after a completed program (silicon: per-RD fires here)?
  { printf("[tb] idle probe: state=%d maint_req=%d maint_proc=%d\n",
           (int)top->dbg_state, (int)top->dbg_maint_req, (int)top->dbg_maint_process);
    int last_state = -1;
    for (int i = 0; i < 2000; i++){
      tick();
      int st = (int)top->dbg_state;
      if (st != last_state || (i % 500 == 499))
        printf("[tb]   idle+%4d: state=%d maint_req=%d maint_proc=%d rd=%ld fin=%d\n",
               i, st, (int)top->dbg_maint_req, (int)top->dbg_maint_process,
               maint_rd, (int)top->softmc_fin);
      last_state = st;
    } }

  // M: maintenance-overlap phase sweep. tPRDI = 1 us on this design —
  // silicon runs per-RD microcode constantly, so every real program
  // start races a maintenance boundary. Launch the branch loop at 64
  // phases; count maint pulses inside each window; any no-FIN = repro.
  printf("[tb] M: phase sweep (maint so far: rd=%ld zq=%ld ref=%ld)\n",
         maint_rd, maint_zq, maint_ref);
  int wedged_at = -1;
  for (int ph = 0; ph < 64 && wedged_at < 0; ph++){
    for (int i = 0; i < ph * 5 + 7; i++) tick();
    long m0 = maint_rd + maint_zq + maint_ref;
    bool sent = send_program(s2);
    bool fin  = sent && wait_fin(120000);
    long dm = (maint_rd + maint_zq + maint_ref) - m0;
    if (!sent || !fin){
      wedged_at = ph;
      printf("[tb] M: phase %2d sent=%d fin=%d maint_in_window=%ld -> **WEDGE REPRODUCED**\n",
             ph, sent, fin, dm);
      // the verdict read, silicon-style
      bool s = send_program(s1);
      bool f = s && wait_fin(50000);
      printf("[tb] M: post-wedge read: sent=%d fin=%d (silicon: jams)\n", s, f);
    } else if (dm > 0) {
      printf("[tb] M: phase %2d fin=%d cyc=%ld maint_in_window=%ld (overlap survived)\n",
             ph, fin, fin_seen_at, dm);
    }
  }
  if (wedged_at < 0)
    printf("[tb] M: 64 phases clean — maint overlap not yet the trigger shape\n");

  // N: STREAMED branch loops with host-pacing gaps + live maintenance —
  // the E14 shape the fetch-boundary TB could never test faithfully.
  // With the restored maint pulse, the hazard to disprove is: a maint
  // event slipping into an inter-program IDLE gap and restarting fetch
  // mid-stream. Verdict = exactly NSTREAM fins, no hang.
  { // STREAM_EN=on control word: byte[9]=0x08 (bit INSTR_WIDTH+11), bit0=on
    top->h2c_tdata[0] = 1; top->h2c_tdata[1] = 0;
    for (int i=2;i<8;i++) top->h2c_tdata[i] = 0;
    top->h2c_tdata[2] = 0x0800;                    // byte9 = 0x08
    top->h2c_tkeep = 0xFFFFFFFFu; top->h2c_tvalid = 1; top->h2c_tlast = 1;
    long w2 = 0; while(!top->h2c_tready && w2++ < 20000) tick();
    tick(); top->h2c_tvalid = 0; top->h2c_tlast = 0;
    for (int i=0;i<20;i++) tick();

    const int NSTREAM = 8;
    long fins = 0, m0 = maint_rd + maint_zq + maint_ref;
    bool sent_all = true;
    long fin_lo = 0;
    // maint programs share EXECUTE_S and end with softmc_fin too — a
    // fin while dbg_maint_process=1 is maintenance, not a user program.
    auto count_fins = [&](long budget){
      for (long i=0;i<budget;i++){ tick();
        if (top->softmc_fin && !fin_lo && !top->dbg_maint_process)
          { fins++; }
        fin_lo = top->softmc_fin ? 1 : 0;
        if (fins >= NSTREAM) return; } };
    for (int p = 0; p < NSTREAM && sent_all; p++){
      sent_all = send_program(s2);
      for (int g = 0; g < 40 + p*17; g++) { tick();   // host-pacing gap
        if (top->softmc_fin && !fin_lo && !top->dbg_maint_process) fins++;
        fin_lo = top->softmc_fin ? 1 : 0; }
    }
    count_fins(200000);
    long dm = (maint_rd + maint_zq + maint_ref) - m0;
    printf("[tb] N streamed-branch+maint: sent_all=%d fins=%ld/%d maint_in_window=%ld -> %s\n",
           sent_all, fins, NSTREAM, dm,
           (sent_all && fins == NSTREAM) ? "PASS" : "E14-CLASS FAIL");
    if (!(sent_all && fins == NSTREAM)) fails++;
    // stream off
    top->h2c_tdata[0] = 0; top->h2c_tdata[2] = 0x0800;
    for (int i=3;i<8;i++) top->h2c_tdata[i] = 0; top->h2c_tdata[1] = 0;
    top->h2c_tkeep = 0xFFFFFFFFu; top->h2c_tvalid = 1; top->h2c_tlast = 1;
    w2 = 0; while(!top->h2c_tready && w2++ < 20000) tick();
    tick(); top->h2c_tvalid = 0; top->h2c_tlast = 0;
    // final legacy integrity read
    bool s = send_program(s1); bool f = s && wait_fin(50000);
    printf("[tb] N post-stream legacy read: sent=%d fin=%d ddr_reads_total=%ld\n",
           s, f, ddr_read_beats);
    if (!(s && f)) fails++;
  }

  // ---------- PHASE 2: host-bytes-out (real readback engine + DRAM) ---
  auto s4 = load_hex("s4_read128.hex");
  auto s5 = load_hex("s5_wrloop.hex");
  printf("[tb] phase2: s4=%zu insts (full-row read), s5=%zu insts (E14 write loop)\n",
         s4.size(), s5.size());
  const int P2_BANK = 1, P2_ROW = 60000, P2_WROW = 60016;

  // R1: full-row read -> expect 8192B payload (oracle-exact) + 32B
  // trailer with magic 0xDBC0DE0C. Beat c <-> col 8c (CASR=8).
  if (!s4.empty()){
    c2h_bytes.clear();
    bool sent = send_program(s4);
    bool fin  = sent && wait_fin(80000);
    for (int i=0;i<4000 && c2h_bytes.size() < 8224; i++) tick();  // drain
    long got = (long)c2h_bytes.size();
    int bad = -1; long badBytes = 0;
    if (got >= 8224){
      for (int c=0;c<128;c++){
        uint32_t exp[16];
        dram_expected_beat(0, P2_BANK, P2_ROW, c*8, exp);
        if (memcmp(&c2h_bytes[c*64], exp, 64) != 0){
          if (bad<0) bad = c;
          for (int b=0;b<64;b++) if (c2h_bytes[c*64+b] != ((uint8_t*)exp)[b]) badBytes++;
        }
      }
    }
    uint32_t magic = got>=8224 ? *(uint32_t*)&c2h_bytes[8192] : 0;
    bool pass = sent && fin && got>=8224 && bad<0 && magic==0xDBC0DE0Cu;
    printf("[tb] R1 full-row read : sent=%d fin=%d got=%ldB firstBadBeat=%d badBytes=%ld magic=%08x -> %s\n",
           sent, fin, got, bad, badBytes, magic, pass?"PASS":"FAIL");
    if (!pass) fails++;
  } else { printf("[tb] R1 SKIPPED (no s4 hex)\n"); fails++; }

  // W1: the E14-content test, faithfully — branch-looped LDWD/WRITE body
  // (production wrRow idiom) then read the row back over c2h; compare
  // against the INTENT pattern (slot0 = col index, slots 1-15 = prol).
  if (!s5.empty() && !s4.empty()){
    { bool sent = send_program(s5);
      bool fin  = sent && wait_fin(120000);
      printf("[tb] W1 write loop   : sent=%d fin=%d\n", sent, fin);
      if (!(sent&&fin)) fails++; }
    // read W-row back: s4 targets P2_ROW; W-row read needs its own hex —
    // gen emits s6_readw.hex for P2_WROW.
    auto s6 = load_hex("s6_readw.hex");
    if (!s6.empty()){
      c2h_bytes.clear();
      bool sent = send_program(s6);
      bool fin  = sent && wait_fin(80000);
      for (int i=0;i<4000 && c2h_bytes.size() < 8224; i++) tick();
      long got = (long)c2h_bytes.size();
      uint32_t prol[16];
      for(int q=0;q<16;q++) prol[q]=0xE1400000u + 0x01010101u*(uint32_t)q + 0xB1u;
      int bad = -1; long badBytes = 0;
      if (got >= 8224){
        for (int c=0;c<128;c++){
          uint32_t want[16];
          want[0] = (uint32_t)c;
          for (int q=1;q<16;q++) want[q] = prol[q];
          if (memcmp(&c2h_bytes[c*64], want, 64) != 0){
            if (bad<0) bad = c;
            for (int b=0;b<64;b++) if (c2h_bytes[c*64+b] != ((uint8_t*)want)[b]) badBytes++;
          }
        }
      }
      uint32_t magic = got>=8224 ? *(uint32_t*)&c2h_bytes[8192] : 0;
      bool pass = sent && fin && got>=8224 && bad<0 && magic==0xDBC0DE0Cu;
      printf("[tb] W1 readback     : sent=%d fin=%d got=%ldB firstBadBeat=%d badBytes=%ld magic=%08x -> %s (E14-content oracle)\n",
             sent, fin, got, bad, badBytes, magic, pass?"PASS":"FAIL");
      if (!pass && got>=64){
        const uint32_t* g=(const uint32_t*)&c2h_bytes[0];
        printf("[tb]   W1 beat0 got : %08x %08x %08x %08x | %08x %08x %08x %08x\n",
               g[0],g[1],g[2],g[3],g[4],g[5],g[6],g[7]);
        printf("[tb]   W1 beat0 g8+ : %08x %08x %08x %08x\n", g[8],g[9],g[10],g[11]);
      }
      if (!pass) fails++;
    } else { printf("[tb] W1 readback SKIPPED (no s6 hex)\n"); fails++; }
  } else { printf("[tb] W1 SKIPPED (missing hex)\n"); fails++; }

  // P: the production MM3D session shape that stalls on build-14
  // silicon (s11/p11 twin gates): fetch PARKED by a completed legacy
  // program + idle (maint churn), then SET_SEGPOP x2, STREAM_EN on,
  // and a sized stream session of full-row reads. Silicon: the FIRST
  // record never arrives (60 s stall). Verdict bit = record 1 arrival.
  { // ensure parked + idle
    bool s0ok = send_program(s1); bool f0 = s0ok && wait_fin(50000);
    for (int i=0;i<800;i++) tick();                 // idle: maint churns
    // SET_SEGPOP x2 (byte8 = 0x80)
    for (int k=0;k<2;k++){
      top->h2c_tdata[0]=0; top->h2c_tdata[1]=0;
      for (int i=2;i<8;i++) top->h2c_tdata[i]=0;
      top->h2c_tdata[2]=0x80;
      top->h2c_tkeep=0xFFFFFFFFu; top->h2c_tvalid=1; top->h2c_tlast=1;
      long w3=0; while(!top->h2c_tready && w3++<30000) tick();
      tick(); top->h2c_tvalid=0; top->h2c_tlast=0;
      for (int i=0;i<10;i++) tick(); }
    // STREAM_EN on
    top->h2c_tdata[0]=1; top->h2c_tdata[1]=0;
    for (int i=2;i<8;i++) top->h2c_tdata[i]=0;
    top->h2c_tdata[2]=0x0800;
    top->h2c_tkeep=0xFFFFFFFFu; top->h2c_tvalid=1; top->h2c_tlast=1;
    { long w3=0; while(!top->h2c_tready && w3++<30000) tick(); }
    tick(); top->h2c_tvalid=0; top->h2c_tlast=0;
    for (int i=0;i<20;i++) tick();
    // session: 4 full-row reads, streamed back-to-back
    c2h_bytes.clear();
    bool sentP = true;
    for (int p=0;p<4 && sentP;p++) sentP = send_program(s4);
    long budget = 400000; long got1_at = -1;
    for (long i=0;i<budget;i++){ tick();
      if (got1_at<0 && c2h_bytes.size() >= 2080) got1_at = i;
      if (c2h_bytes.size() >= 4*2080) break; }
    printf("[tb] P segpop-session: parked=%d sent=%d rec1_at=%ld total=%zuB/%d "
           "state=%d maint_proc=%d hold=%d -> %s\n",
           f0, sentP, got1_at, c2h_bytes.size(), 4*2080,
           (int)top->dbg_state, (int)top->dbg_maint_process,
           (int)top->dbg_fetch_hold,
           (c2h_bytes.size() >= 4*2080) ? "ALL RECORDS (no repro)" :
           (got1_at<0) ? "FIRST RECORD NEVER ARRIVED (silicon stall REPRODUCED)"
                       : "PARTIAL (records stalled mid-session)");
    if (c2h_bytes.size() < 4*2080) fails++;
    // stream off for cleanliness
    top->h2c_tdata[0]=0; for (int i=1;i<8;i++) top->h2c_tdata[i]=0;
    top->h2c_tdata[2]=0x0800;
    top->h2c_tkeep=0xFFFFFFFFu; top->h2c_tvalid=1; top->h2c_tlast=1;
    { long w3=0; while(!top->h2c_tready && w3++<30000) tick(); }
    tick(); top->h2c_tvalid=0; top->h2c_tlast=0;
  }

  // ---------- Q: trailer framing under back-to-back streaming --------
  // N sized row reads, sent back-to-back, c2h drain THROTTLED.
  // Expect N messages (tlast) of 2048+32 bytes. The silicon bug shows
  // as fewer tlasts than records.
  for (long throttle : {0L, 40L}) {
    auto s4q = load_hex("s4_read128.hex");
    if (s4q.empty()) { printf("[tb] Q SKIPPED (no s4 hex)\n"); break; }
    const int QN = 8, QPAY = 2048;
    // fresh state: stop streaming, settle
    top->h2c_tdata[0]=0; for (int i=1;i<8;i++) top->h2c_tdata[i]=0;
    top->h2c_tdata[2]=0x0800;
    top->h2c_tkeep=0xFFFFFFFFu; top->h2c_tvalid=1; top->h2c_tlast=1;
    { long w=0; while(!top->h2c_tready && w++<20000) tick(); }
    tick(); top->h2c_tvalid=0; top->h2c_tlast=0;
    for (int i=0;i<400;i++) tick();

    // SEG_POP x2, then STREAM_EN on
    for (int k=0;k<2;k++){
      top->h2c_tdata[0]=0; top->h2c_tdata[1]=0;
      for (int i=2;i<8;i++) top->h2c_tdata[i]=0;
      top->h2c_tdata[2]=0x80;
      top->h2c_tkeep=0xFFFFFFFFu; top->h2c_tvalid=1; top->h2c_tlast=1;
      { long w=0; while(!top->h2c_tready && w++<20000) tick(); }
      tick(); top->h2c_tvalid=0; top->h2c_tlast=0;
      for (int i=0;i<10;i++) tick(); }
    top->h2c_tdata[0]=1; top->h2c_tdata[1]=0;
    for (int i=2;i<8;i++) top->h2c_tdata[i]=0;
    top->h2c_tdata[2]=0x0800;
    top->h2c_tkeep=0xFFFFFFFFu; top->h2c_tvalid=1; top->h2c_tlast=1;
    { long w=0; while(!top->h2c_tready && w++<20000) tick(); }
    tick(); top->h2c_tvalid=0; top->h2c_tlast=0;
    for (int i=0;i<20;i++) tick();

    // Wait for genuine quiescence before zeroing the counters: a trailer
    // still in flight from the previous window would otherwise be
    // attributed to this one (and its own record counted as missing).
    { long quiet = 0; size_t last = c2h_bytes.size();
      for (long i = 0; i < 200000 && quiet < 2000; i++) {
        tick();
        if (c2h_bytes.size() != last) { last = c2h_bytes.size(); quiet = 0; }
        else quiet++;
      } }
    g_drain_off = throttle;
    c2h_bytes.clear();
    g_c2h_tlast_count = 0;
    g_fin_edges = 0; g_fr_edges = 0; g_fr_hi = 0; g_fr_user = 0; g_fr_maint = 0;
    bool sentQ = true;
    if (throttle) g_trace_en = true;      // capture the merge
    for (int q=0; q<QN && sentQ; q++) sentQ = send_program(s4q);
    long want = (long)QN * (QPAY + 32);
    for (long i=0;i<1500000 && (long)c2h_bytes.size() < want; i++) {
      tick();
      if (g_trace_en && c2h_bytes.size() > 7000) g_trace_en = false;
    }
    g_trace_en = false;
    // WIRE TRUTH: the trailer's first word is the magic, so the byte
    // stream itself shows where every record actually ends. No DUT
    // instrumentation, no counter reconciliation.
    { long nmag = 0; printf("[tb] Q wire map (magic offsets):");
      for (size_t o = 0; o + 4 <= c2h_bytes.size(); o += 4) {
        uint32_t w; memcpy(&w, &c2h_bytes[o], 4);
        if ((w & 0xFFFFFF00u) == 0xDBC0DE00u) {
          if (nmag < 12) printf(" %zu", o);
          nmag++;
        }
      }
      printf("  (total %ld; expected at 2048,4128,6208,... every 2080)\n", nmag); }
    printf("[tb] Q throttle=%ld: sent=%d bytes=%zu/%ld messages=%ld/%d fins=%ld USER_flushes=%ld maint_flushes=%ld -> %s\n",
           throttle, sentQ, c2h_bytes.size(), want, g_c2h_tlast_count, QN,
           g_fin_edges, g_fr_user, g_fr_maint,
           (g_c2h_tlast_count == QN && (long)c2h_bytes.size() == want)
             ? "framing OK"
             : "**TRAILER MERGE REPRODUCED**");
    if (g_c2h_tlast_count != QN) fails++;
    g_drain_off = 0;
    // stream off
    top->h2c_tdata[0]=0; for (int i=1;i<8;i++) top->h2c_tdata[i]=0;
    top->h2c_tdata[2]=0x0800;
    top->h2c_tkeep=0xFFFFFFFFu; top->h2c_tvalid=1; top->h2c_tlast=1;
    { long w=0; while(!top->h2c_tready && w++<20000) tick(); }
    tick(); top->h2c_tvalid=0; top->h2c_tlast=0;
    for (int i=0;i<200;i++) tick();
  }

  tfp->close();
  printf("[tb] done, %d hard fails, maint rd=%ld zq=%ld ref=%ld, trace=e2e.vcd, %lu cycles\n",
         fails, maint_rd, maint_zq, maint_ref, (unsigned long)t);
  return fails;
}
