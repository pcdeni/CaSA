// =============================================================================
// tb_link.cpp — Verilator gate for inter_bender_link (Task #76).
//
// Pattern from verilator_local/tb_readback.cpp: behavioral sim models stand in
// for the encrypted XPM IP (sim/xpm_fifo_async.v <-> real xpm_fifo_async).
//
// FOUR async UI clocks derived from one integer time base at PAIRWISE-COPRIME
// half-periods {5,7,9,11} (periods 10/14/18/22) => genuinely asynchronous, no
// `--timing needed. Verilator triggers each posedge exactly once per rising
// transition regardless of how many evals we run while the clock is held high.
//
// Faithfulness boundary (CONTRACT.md §6): 2-state FUNCTIONAL sim. It proves data
// integrity / loss-only-on-overflow / counter correctness / frame-atomicity
// across clock RATIOS. Metastability is the real XPM's job (L12, timing-binary).
//
// Gate params: DW=32,KW=4,FIFO_DEPTH=8 (small, to force overflow in scenario c).
// The router carries {tlast,tkeep,tdata} VERBATIM, so width is a parameter with
// no logical role; production instance is DW=256,KW=32,FIFO_DEPTH=512.
//
// Scenarios: (a) idle inertness  (b) core0->core2 across async clks
//            (c) backpressure/no-wedge/drops-counted  (d) frame-atomic reroute
// =============================================================================
#include <verilated.h>
#include "Vinter_bender_link.h"
#include <cstdio>
#include <cstdint>
#include <deque>
#include <vector>
#include <functional>

static int g_fail = 0;
#define CHECK(cond, ...) do{ if(!(cond)){ printf("  [FAIL] "); printf(__VA_ARGS__); printf("\n"); g_fail++; } }while(0)
#define NOTE(...)        do{ printf("  [note] "); printf(__VA_ARGS__); printf("\n"); }while(0)

struct Beat { uint32_t token; uint8_t keep; bool last; uint8_t src; };

struct Harness {
  Vinter_bender_link* d;
  long t = 0;
  int  hp[4] = {5,7,9,11};
  int  prev[4] = {0,0,0,0};

  std::deque<Beat> sq[4];         // per-source send queues
  long src_commits[4] = {0,0,0,0};
  bool host_ready[4] = {1,1,1,1}; // c2h_tready modelled by TB (host converter)
  bool src_never_stalled[4] = {1,1,1,1};
  int  throttle[4] = {1,1,1,1};   // present a beat every Nth source cycle (models readback gaps)
  long src_tick[4] = {0,0,0,0};
  bool presented[4] = {0,0,0,0};

  std::function<bool(int)> dest_ready = [](int){ return true; }; // linkrx_tready pattern
  std::vector<Beat> rx[4];        // per-dest scoreboard

  struct Cfg { bool v=false; uint16_t word=0; } pend_cfg[4];
  std::function<void(int,long)> on_commit = [](int,long){};

  Harness(){ d = new Vinter_bender_link; }
  ~Harness(){ d->final(); delete d; }

  // ---- flattened-port accessors (DW=32 => one word per core) ----
  void set_clk(){ uint8_t v=0; for(int c=0;c<4;c++) v |= (prev[c]&1)<<c; d->core_ui_clk=v; }
  void set_bit(uint8_t& reg,int i,bool b){ reg = (reg & ~(1<<i)) | ((b?1:0)<<i); }

  void reset(){
    d->core_ui_rst = 0xF; d->core_ui_clk=0;
    d->c2h_tvalid=0; d->c2h_tready=0; d->c2h_tlast=0; d->cfg_stb=0; d->cfg_data=0;
    d->linkrx_tready=0;
    for(int i=0;i<4;i++){ d->c2h_tdata[i]=0; } d->c2h_tkeep=0;
    t=0; for(int c=0;c<4;c++) prev[c]=0;
    for(int k=0;k<80;k++){ t++; for(int c=0;c<4;c++) prev[c]=(t/hp[c])&1; set_clk(); d->eval(); }
    d->core_ui_rst = 0x0;
    for(int k=0;k<40;k++){ t++; for(int c=0;c<4;c++) prev[c]=(t/hp[c])&1; set_clk(); d->eval(); }
  }

  void drive_source(int s){
    uint8_t tv=d->c2h_tvalid, tl=d->c2h_tlast, tr=d->c2h_tready;
    bool present = !sq[s].empty() && ((src_tick[s] % throttle[s])==0);
    presented[s]=present;
    if(present){
      Beat& b = sq[s].front();
      set_bit(tv,s,true); set_bit(tl,s,b.last);
      d->c2h_tdata[s]=b.token;
      d->c2h_tkeep = (d->c2h_tkeep & ~(0xF<<(s*4))) | ((b.keep&0xF)<<(s*4));
    } else { set_bit(tv,s,false); set_bit(tl,s,false); }
    set_bit(tr,s,host_ready[s]);
    d->c2h_tvalid=tv; d->c2h_tlast=tl; d->c2h_tready=tr;
    src_tick[s]++;
  }

  Beat cap[4]; bool cap_fire[4]={0,0,0,0};
  void drive_dest(int dd){
    uint8_t rr=d->linkrx_tready; bool rdy=dest_ready(dd); set_bit(rr,dd,rdy); d->linkrx_tready=rr;
    bool tv = (d->linkrx_tvalid>>dd)&1;
    if(tv && rdy){
      cap[dd].token = d->linkrx_tdata[dd];
      cap[dd].keep  = (d->linkrx_tkeep>>(dd*4))&0xF;
      cap[dd].last  = (d->linkrx_tlast>>dd)&1;
      cap[dd].src   = (d->linkrx_tsrc>>(dd*2))&3;
      cap_fire[dd]=true;
    } else cap_fire[dd]=false;
  }

  // one integer-time step: pre-edge drive on rising domains, eval, post-edge commit
  void step(){
    t++;
    int nl[4]; for(int c=0;c<4;c++) nl[c]=(t/hp[c])&1;
    bool rising[4];
    for(int c=0;c<4;c++){ rising[c] = (nl[c]==1 && prev[c]==0); }
    uint8_t stb=d->cfg_stb;
    for(int c=0;c<4;c++) if(rising[c]){
      // check c2h not stalled: if we have data and host_ready but couldn't send last time
      drive_source(c);
      drive_dest(c);
      if(pend_cfg[c].v){ set_bit(stb,c,true);
        d->cfg_data = (d->cfg_data & ~((uint64_t)0xFFFF<<(c*16))) | ((uint64_t)pend_cfg[c].word<<(c*16)); }
    }
    d->cfg_stb=stb;
    for(int c=0;c<4;c++) prev[c]=nl[c];
    set_clk(); d->eval();
    // post-edge
    uint8_t stb2=d->cfg_stb;
    for(int c=0;c<4;c++) if(rising[c]){
      if(pend_cfg[c].v){ set_bit(stb2,c,false); pend_cfg[c].v=false; }
      // source commit
      bool committed = ((d->c2h_tvalid>>c)&1) && ((d->c2h_tready>>c)&1);
      if(committed){ sq[c].pop_front(); src_commits[c]++; on_commit(c, src_commits[c]); }
      else if(presented[c] && host_ready[c]) { src_never_stalled[c]=false; } // presented+ready but not taken => c2h stall
      // dest capture
      if(cap_fire[c]) rx[c].push_back(cap[c]);
    }
    d->cfg_stb=stb2;
  }
  void run(int nsteps){ for(int i=0;i<nsteps;i++) step(); }

  // helper: enqueue F frames of B beats on source s, tokens from base
  void enqueue(int s, int frames, int beats, uint32_t base){
    for(int f=0; f<frames; f++)
      for(int b=0;b<beats;b++){
        Beat bt; bt.token = base + f*100 + b; bt.keep = (bt.token & 0xF); bt.last=(b==beats-1); bt.src=s;
        sq[s].push_back(bt);
      }
  }
  void cfg(int s, bool en, int dst){ // schedule a route write on core s
    uint16_t w = (en?1:0) | ((dst&3)<<1) | (0x76<<3);
    pend_cfg[s] = { true, w };
  }
  uint32_t beats_ctr(int s,int dd){ return d->stat_beats[s*4+dd]; }
  uint32_t frames_ctr(int s,int dd){ return d->stat_frames[s*4+dd]; }
  uint32_t drops_ctr(int s,int dd){ return d->stat_drops[s*4+dd]; }
  uint32_t inj_ctr(int dd){ return d->stat_injframes[dd]; }
  uint8_t  status(int dd){ return (d->stat_status>>(dd*8))&0xFF; }
};

// -------------------------------------------------------------------- (a)
static void scen_idle(){
  printf("[a] idle-route inertness (host path bit-identical by construction)\n");
  Harness h; h.reset();
  h.enqueue(0, /*frames*/3, /*beats*/4, 0x1000); // host traffic on core0, NO route enabled
  h.run(4000);
  CHECK(h.src_commits[0]==12, "core0 should commit all 12 host beats, got %ld", h.src_commits[0]);
  CHECK(h.src_never_stalled[0], "host c2h path must never stall");
  int tv_seen=0; // (already ran) — assert no link activity at all
  for(int s=0;s<4;s++) for(int dd=0;dd<4;dd++){
    CHECK(h.beats_ctr(s,dd)==0,  "beats[%d][%d]=%u (must be 0 when inert)",s,dd,h.beats_ctr(s,dd));
    CHECK(h.drops_ctr(s,dd)==0,  "drops[%d][%d]=%u",s,dd,h.drops_ctr(s,dd));
  }
  for(int dd=0;dd<4;dd++){ CHECK(h.rx[dd].empty(),"dest %d received %zu beats while inert",dd,h.rx[dd].size());
                           CHECK(h.inj_ctr(dd)==0,"injframes[%d]=%u",dd,h.inj_ctr(dd)); }
  (void)tv_seen;
  NOTE("router has NO host-path drivers (c2h_tready is an input); inertness => star topology intact");
}

// -------------------------------------------------------------------- (b)
static void scen_route(){
  printf("[b] routed core0->core2 intact across asynchronous clocks (T=10 vs 18)\n");
  Harness h; h.reset();
  h.throttle[0]=3;                   // source injects with gaps (readback-engine-like) < sink rate
  h.cfg(0, true, 2);
  h.run(200);                        // let the cfg latch
  const int F=5, B=6;
  h.enqueue(0, F, B, 0x5000);
  h.run(8000);
  CHECK(h.src_commits[0]==F*B, "core0 committed %ld/%d", h.src_commits[0], F*B);
  CHECK(h.rx[2].size()==(size_t)(F*B), "dest2 got %zu beats, expected %d", h.rx[2].size(), F*B);
  // integrity + order + tsrc + tkeep witness + tlast structure
  bool ok=true; int frames_seen=0;
  for(int i=0;i<(int)h.rx[2].size();i++){
    int f=i/B, b=i%B; uint32_t exp=0x5000 + f*100 + b;
    Beat& r=h.rx[2][i];
    if(r.token!=exp) { ok=false; }
    if(r.keep!=(exp&0xF)) ok=false;
    if(r.src!=0) ok=false;
    bool wantlast=(b==B-1);
    if(r.last!=wantlast) ok=false;
    if(r.last) frames_seen++;
  }
  CHECK(ok, "dest2 stream integrity (token/keep/tsrc/tlast) mismatch");
  CHECK(frames_seen==F, "dest2 saw %d tlast frames, expected %d", frames_seen, F);
  CHECK(h.beats_ctr(0,2)==(uint32_t)(F*B), "beats[0][2]=%u", h.beats_ctr(0,2));
  CHECK(h.frames_ctr(0,2)==(uint32_t)F,    "frames[0][2]=%u", h.frames_ctr(0,2));
  CHECK(h.drops_ctr(0,2)==0,               "drops[0][2]=%u (expected 0)", h.drops_ctr(0,2));
  CHECK(h.inj_ctr(2)==(uint32_t)F,         "injframes[2]=%u", h.inj_ctr(2));
  for(int dd=0;dd<4;dd++) if(dd!=2) CHECK(h.rx[dd].empty(), "dest %d should be idle", dd);
}

// -------------------------------------------------------------------- (c)
static void scen_backpressure(){
  printf("[c] backpressure: stalled peer never wedges c2h; drops counted; drain intact\n");
  Harness h; h.reset();
  h.cfg(0, true, 2);
  h.run(200);
  // dest2 STALLS for the whole send window, then releases
  static bool release=false; release=false;
  h.dest_ready = [](int dd){ return dd==2 ? release : true; };
  const int TOT=40; // >> FIFO_DEPTH(8)
  h.enqueue(0, 1, TOT, 0x7000);
  // run while stalled: source must keep committing (c2h never blocked)
  h.run(4000);
  CHECK(h.src_commits[0]==TOT, "core0 committed %ld/%d WHILE peer stalled (no wedge)", h.src_commits[0], TOT);
  CHECK(h.src_never_stalled[0], "c2h path stalled by a slow peer => WEDGE (forbidden)");
  CHECK(h.drops_ctr(0,2)>0, "expected drops once FIFO(8) filled, got %u", h.drops_ctr(0,2));
  // conservation: every routed committed beat either enqueued(beats) or dropped
  uint32_t bc=h.beats_ctr(0,2), dc=h.drops_ctr(0,2);
  CHECK(bc+dc==(uint32_t)TOT, "beats(%u)+drops(%u) != committed(%d)", bc, dc, TOT);
  uint8_t st=h.status(2);
  NOTE("dest2 status byte while stalled = 0x%02X (cause bits[6:5]; 2=consumer-stall)", st);
  CHECK(((st>>5)&3)==2 || ((st>>5)&3)==3, "status cause should flag stall/starve, got %d",(st>>5)&3);
  // now release and drain
  release=true;
  h.run(4000);
  CHECK(h.rx[2].size()==bc, "drained %zu, expected the %u enqueued (rest were dropped)", h.rx[2].size(), bc);
  // received tokens must be a contiguous, in-order prefix of the sent stream (no corruption)
  bool ok=true; for(int i=0;i<(int)h.rx[2].size();i++){ if(h.rx[2][i].token != (uint32_t)(0x7000+i)) ok=false; }
  CHECK(ok, "drained data must be the in-order prefix that fit the FIFO");
  CHECK(h.drops_ctr(0,2)==dc, "no new drops after release, drops went %u->%u", dc, h.drops_ctr(0,2));
}

// -------------------------------------------------------------------- (d)
static void scen_reroute(){
  printf("[d] runtime reroute applies at a FRAME BOUNDARY (never splits a frame)\n");
  Harness h; h.reset();
  h.throttle[0]=3;                 // gaps so neither dest FIFO overflows during the test
  h.cfg(0, true, 1);               // frames initially -> dest1
  h.run(200);
  const int B=6;
  // frame1 (->dest1), frame2 (->dest1, reconfig fires MID-frame2), frame3 (->dest2)
  h.enqueue(0, 1, B, 0x100);       // frame1 tokens 0x100..0x105
  h.enqueue(0, 1, B, 0x200);       // frame2 tokens 0x200..0x205
  h.enqueue(0, 1, B, 0x300);       // frame3 tokens 0x300..0x305
  // reconfigure to dest2 after 2 beats of frame2 have committed (commit #8)
  h.on_commit = [&h](int s, long n){ if(s==0 && n==8){ h.cfg(0,true,2); } };
  h.run(9000);
  CHECK(h.src_commits[0]==3*B, "core0 committed %ld/%d", h.src_commits[0], 3*B);
  // dest1 must have EXACTLY frame1+frame2 (12 beats), dest2 EXACTLY frame3 (6 beats)
  CHECK(h.rx[1].size()==(size_t)(2*B), "dest1 got %zu, expected %d (frame1+frame2 whole)", h.rx[1].size(), 2*B);
  CHECK(h.rx[2].size()==(size_t)(B),   "dest2 got %zu, expected %d (frame3 whole)", h.rx[2].size(), B);
  bool d1ok=true;
  for(int b=0;b<B;b++){ if(h.rx[1][b].token   != (uint32_t)(0x100+b)) d1ok=false; }   // frame1
  for(int b=0;b<B;b++){ if(h.rx[1][B+b].token != (uint32_t)(0x200+b)) d1ok=false; }   // frame2 (NOT split away)
  CHECK(d1ok, "dest1 must hold all of frame1 THEN all of frame2 (frame2 not split to dest2)");
  bool d2ok=true; for(int b=0;b<B;b++){ if(h.rx[2][b].token != (uint32_t)(0x300+b)) d2ok=false; }
  CHECK(d2ok, "dest2 must hold exactly frame3");
  CHECK(h.frames_ctr(0,1)==2, "frames[0][1]=%u (expected 2)", h.frames_ctr(0,1));
  CHECK(h.frames_ctr(0,2)==1, "frames[0][2]=%u (expected 1)", h.frames_ctr(0,2));
  CHECK(h.drops_ctr(0,1)==0 && h.drops_ctr(0,2)==0, "no drops expected in reroute");
}

int main(int argc, char** argv){
  Verilated::commandArgs(argc, argv);
  printf("==== inter_bender_link Verilator gate (N_CORES=4, async UI clocks) ====\n");
  scen_idle();
  scen_route();
  scen_backpressure();
  scen_reroute();
  printf("==== %s (%d failures) ====\n", g_fail? "GATE FAIL":"GATE PASS", g_fail);
  return g_fail ? 1 : 0;
}
