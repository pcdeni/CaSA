// M3 — on-the-fly-encoded, RESIDENT in-DRAM GeMV kernel (2026-07-17).
// Implements MVDRAM's §V on-the-fly vector encoding + resident dataflow on
// our lattice:
//   - weight bitplanes (AND both polarities — their Fig-15 dual-track
//     matrix stored 2×) are RESIDENT in parallel regions T⊕Dw / T⊕Dnw;
//   - activations are NEVER written to DRAM: each product is created by a
//     data-dependent command — activation bit=1 → same-slot RowClone of the
//     weight row into the compute slot; bit=0 → skip (§V-D zero-skip: the
//     compute slot was pre-zeroed from the constants region, and skipped
//     products also shrink the adder tree);
//   - the popcount CSA tree uses 4-ROW sub-coset MAJ3s INSIDE the 16-row
//     tuple: an APA pair at a {6,8}-unit distance co-activates exactly one
//     4-row group (selection law) = 3 data rows + 1 frac'd-ONE reference
//     (June's mvdram-fulladder 4-tuple primitive, 99.98%);
//   - the full adder is all-MAJ3 dual-track:
//       carry = MAJ3(a,b,c);           ¬carry = MAJ3(¬a,¬b,¬c)
//       sum   = MAJ3(M1,M2,M3), M1=MAJ3(a,b,¬c), M2=MAJ3(a,¬b,c),
//               M3=MAJ3(¬a,b,c);       ¬sum  = mirrored
//     — every operand move is a same-slot clone between safe regions
//     (offsets ⊆ {1,16,512} for the s78 tuple: all subsets unit-free), so
//     intermediates NEVER round-trip the host; only final result planes
//     are read out.
// Modes:
//   screen — per-group 4-row MAJ3 reliability on the tuple (frac'd-ONE ref)
//   fa     — one dual-track FA, exactness vs host (sum, carry, both pols)
//   gemv   — K weight planes × random activations at given density:
//            encoded products → CSA tree → readout; correctness vs host
//            popcount; wall time + command count vs density; packed-readout
//            baseline printed for comparison (addendum 14b framing).
// Argv: <bender> <colmask> <bank> <mode> [K=12] [trials=10] [density_pct=100] [seed=3]
#include "instruction.h"
#include "prog.h"
#include "platform.h"
#include "../util.h"
#include "lattice_alloc.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
using namespace std;

static const int CHUNK_COLS[3] = {43, 43, 42};
static SoftMCPlatform* PF = nullptr;
static int BANK = 0;
static long OPS_CLONE = 0, OPS_MAJ = 0, OPS_SKIP = 0;

static void pcwrite(uint32_t row, const uint32_t* seg) {
  int cs = 0;
  for (int ch = 0; ch < 3; ch++) {
    int n = CHUNK_COLS[ch]; const uint32_t* cd = seg + cs * 16;
    Program p;
    p.add_inst(SMC_LI(8, CASR)); p.add_inst(SMC_LI(BANK, BAR));
    p.add_inst(SMC_LI(row, RAR)); p.add_inst(SMC_LI(cs * 8, CAR));
    p.add_below(PRE(BAR, 0, 0)); p.add_below(ACT(BAR, 0, RAR, 0));
    for (int k = 0; k < n; k++) {
      const uint32_t* sl = cd + k * 16;
      for (int s = 0; s < 16; s++) { p.add_inst(SMC_LI(sl[s], PATTERN_REG)); p.add_inst(SMC_LDWD(PATTERN_REG, s)); }
      p.add_below(WRITE(BAR, CAR, 1)); p.add_inst(SMC_SLEEP(8));
    }
    p.add_inst(SMC_SLEEP(8)); p.add_below(PRE(BAR, 0, 0));
    p.add_inst(SMC_SLEEP(4)); p.add_inst(SMC_END());
    PF->execute(p); cs += n;
  }
}
static void wrconst(uint32_t row, uint32_t val) {
  Program p;
  p.add_inst(SMC_LI(8, CASR)); p.add_inst(SMC_LI(BANK, BAR)); p.add_inst(SMC_LI(128, NUM_COLS_REG));
  p.add_below(PRE(BAR, 0, 0)); p.add_below(wrRow_immediate_label(BAR, row, val, 1));
  p.add_inst(SMC_SLEEP(6)); p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_END());
  PF->execute(p);
}
static Program frac_builder(int t_frac, int r) {
  Program p; p.add_inst(all_nops()); p.add_inst(SMC_LI(r, RF_REG));
  int nc = 2 + t_frac; nc += 4 - (nc % 4);
  Mininst q[nc]; for (int i = 0; i < nc; i++) q[i] = SMC_NOP();
  q[0] = SMC_ACT(BAR, 0, RF_REG, 0); q[t_frac + 1] = SMC_PRE(BAR, 0, 0);
  for (int i = 0; i < nc; i += 4) p.add_inst(q[i], q[i+1], q[i+2], q[i+3]);
  return p;
}
static void clone_op(uint32_t src, uint32_t dst) {   // safe RowClone (30,1)
  Program p;
  p.add_inst(SMC_LI(8, CASR)); p.add_inst(SMC_LI(BANK, BAR));
  p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_SLEEP(6));
  p.add_below(doubleACT(30, 1, src, dst));
  p.add_inst(SMC_SLEEP(6)); p.add_below(PRE(BAR, 0, 0));
  p.add_inst(SMC_SLEEP(6)); p.add_inst(SMC_END());
  PF->execute(p); OPS_CLONE++;
}
// 4-row group MAJ3: ref slot loaded ONE + frac'd 3×@t0 (June recipe), then
// APA between two group members at a {small-units} distance → co-activates
// exactly the 4-row group; result restored into all 4 rows.
static void maj4_group(uint32_t rA, uint32_t rB, uint32_t ref_row) {
  wrconst(ref_row, 0xFFFFFFFFu);
  Program p;
  p.add_inst(SMC_LI(8, CASR)); p.add_inst(SMC_LI(BANK, BAR)); p.add_inst(SMC_LI(128, NUM_COLS_REG));
  p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_SLEEP(6));
  for (int j = 0; j < 3; j++) { p.add_inst(SMC_SLEEP(6)); p.add_below(frac_builder(0, ref_row)); p.add_inst(SMC_SLEEP(6)); }
  p.add_inst(SMC_SLEEP(6)); p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_SLEEP(6));
  p.add_below(doubleACT(0, 0, rA, rB));
  p.add_inst(SMC_SLEEP(6)); p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_SLEEP(6));
  p.add_inst(SMC_END());
  PF->execute(p); OPS_MAJ++;
}
static void read_row(uint32_t row, uint8_t* out) {
  Program p;
  p.add_inst(SMC_LI(8, CASR)); p.add_inst(SMC_LI(BANK, BAR)); p.add_inst(SMC_LI(128, NUM_COLS_REG));
  p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_SLEEP(6));
  p.add_below(rdRow_immediate(BAR, row));
  p.add_inst(all_nops()); p.add_inst(all_nops());
  p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_END());
  PF->execute(p);
  if (PF->receiveData(out, 8192) != 8192) { fprintf(stderr, "recv fail\n"); _exit(5); }
}

struct Cm { vector<uint32_t> open; uint32_t Rf=0, Rs=0; int sid=0; vector<int> cols; };
static bool read_colmask(const char* path, Cm& c) {
  ifstream f(path); if (!f) return false; string line;
  while (getline(f, line)) {
    if (line.empty()) continue;
    if (line[0] == '#') {
      if (line.find("open_rows:")!=string::npos){ istringstream is(line.substr(line.find(':')+1)); uint32_t v; while(is>>v) c.open.push_back(v); }
      else if (line.find("tuple Rf=")!=string::npos) sscanf(line.c_str(),"# tuple Rf=%u Rs=%u",&c.Rf,&c.Rs);
      else if (line.find("s_id=")!=string::npos&&!c.sid){ size_t p=line.find("s_id="); sscanf(line.c_str()+p,"s_id=%d",&c.sid); }
      continue;
    }
    c.cols.push_back(atoi(line.c_str()));
  }
  return c.open.size()==16 && c.Rf && !c.cols.empty();
}
static uint32_t maj3w(uint32_t a,uint32_t b,uint32_t c){ return (a&b)|(a&c)|(b&c); }

// The tuple's four 4-row groups in list order {0-3}{4-7}{8-11}{12-15}
// (combos of the two small units within each large-unit coset). In each
// group: slots g*4+0..2 = data, g*4+3 = frac'd-ONE reference.
// A "row-value" handle: a row holding a bitplane, at (slot, region-offset).
struct GemvKern {
  Cm cm; TupleAlloc TA;
  uint32_t DW, DNW, DS1, DS2, DC;   // regions: W, ¬W, scratch1, scratch2, constants(zero)
  uint32_t O(int j){ return cm.open[j]; }
  // group MAJ over slots {g*4..g*4+3}: the APA pair must span BOTH small
  // units so all 4 group rows co-activate → pair = (first, last) at d=14
  void fire_group(int g){ maj4_group(O(g*4), O(g*4+3), O(g*4+3)); }
  // load one data slot of a group from region offset D at the SAME slot
  void fill(int slot, uint32_t D){ clone_op(O(slot)^D, O(slot)); }
  // save a group result (any of its rows) to region D at the same slot
  void save(int slot, uint32_t D){ clone_op(O(slot), O(slot)^D); }
};

int main(int argc,char**argv){
  if(argc<5){ fprintf(stderr,"Usage: %s <bender> <colmask> <bank> <screen|fa|gemv> [K=12] [trials=10] [density_pct=100] [seed=3]\n",argv[0]); return 1; }
  int bender=atoi(argv[1]); BANK=atoi(argv[3]);
  string mode=argv[4];
  int K=(argc>5)?atoi(argv[5]):12;
  int TRIALS=(argc>6)?atoi(argv[6]):10;
  int DENS=(argc>7)?atoi(argv[7]):100;
  unsigned seed=(argc>8)?(unsigned)atoi(argv[8]):3;

  GemvKern k;
  if(!read_colmask(argv[2],k.cm)){ fprintf(stderr,"bad colmask\n"); return 2; }
  k.TA.open=k.cm.open; k.TA.Rf=k.cm.Rf; k.TA.Rs=k.cm.Rs;
  if(!k.TA.fit()){ fprintf(stderr,"tuple fit failed\n"); return 2; }
  // safe offsets for s78-class tuples: subsets of bits {0,4,9}
  vector<uint32_t> sb = k.TA.safe_bits();
  if (sb.size() < 3){ fprintf(stderr,"need ≥3 safe bits, got %zu\n", sb.size()); return 2; }
  k.DW=sb[0]; k.DNW=sb[1]; k.DS1=sb[2]; k.DS2=sb[0]|sb[1]; k.DC=sb[0]|sb[2];
  for (uint32_t d : {k.DS2, k.DC}) if (k.TA.span.count(d)) { fprintf(stderr,"combo offset unsafe\n"); return 2; }
  fprintf(stderr,"[gemv] s%d Rf=%u regions: W=+%u nW=+%u S1=+%u S2=+%u C=+%u cols=%zu\n",
          k.cm.sid,k.cm.Rf,k.DW,k.DNW,k.DS1,k.DS2,k.DC,k.cm.cols.size());

  SoftMCPlatform pf(bender);
  if(pf.init()!=SOFTMC_SUCCESS){ fprintf(stderr,"init fail\n"); return 3; }
  pf.reset_fpga(); PF=&pf; srand(seed);

  // zero the constants region once (all 16 slots)
  for(int j=0;j<16;j++) wrconst(k.O(j)^k.DC, 0);

  uint8_t buf[8192];
  auto rd=[&](uint32_t row, uint32_t* out){ read_row(row,buf); for(int s=0;s<2048;s++) memcpy(&out[s],&buf[s*4],4); };

  if (mode=="probe"){
    // clone-offset probe: for each candidate D, write a marker at O(0)⊕D,
    // zero O(0), clone (O(0)⊕D → O(0)), read O(0): match% per D
    vector<uint32_t> mk(2048), got(2048);
    for(int s=0;s<2048;s++) mk[s]=0x5A5A0000u^(s*2654435761u);
    printf("D,clone_ok_pct\n");
    for(uint32_t D : {1u,16u,512u,17u,513u}){
      pcwrite(k.O(0)^D, mk.data());
      wrconst(k.O(0), 0);
      clone_op(k.O(0)^D, k.O(0));
      rd(k.O(0), got.data());
      long ok=0; for(int ci : k.cm.cols) if(got[ci]==mk[ci]) ok++;
      printf("%u,%.3f\n",D,100.0*ok/k.cm.cols.size());
      fprintf(stderr,"[gemv] clone d=%u: %.2f%% match\n",D,100.0*ok/k.cm.cols.size());
    }
    fflush(stdout); _exit(0);
  }

  if (mode=="screen"){
    // per-group 4-row MAJ3 reliability: random a,b,c per group slot, fire, check
    vector<uint32_t> a(2048),b(2048),c(2048),got(2048);
    vector<long> ok(4,0); long n=0;
    for(int tr=0;tr<TRIALS;tr++){
      for(int g=0;g<4;g++){
        for(int s=0;s<2048;s++){ a[s]=((uint32_t)rand()<<15)^rand(); b[s]=((uint32_t)rand()<<15)^rand(); c[s]=((uint32_t)rand()<<15)^rand(); }
        pcwrite(k.O(g*4),a.data()); pcwrite(k.O(g*4+1),b.data()); pcwrite(k.O(g*4+2),c.data());
        k.fire_group(g);
        rd(k.O(g*4), got.data());
        for(int ci : k.cm.cols){ if(got[ci]==maj3w(a[ci],b[ci],c[ci])) ok[g]++; }
      }
      n += k.cm.cols.size();
    }
    printf("group,maj4_ok_pct\n");
    for(int g=0;g<4;g++) printf("%d,%.4f\n",g,100.0*ok[g]/n);
    for(int g=0;g<4;g++) fprintf(stderr,"[gemv] group %d 4-row MAJ3: %.3f%% on colmask\n",g,100.0*ok[g]/n);
    fflush(stdout); _exit(0);
  }

  if (mode=="fa"){
    // one dual-track FA, all-resident. Regions: +1 = positive inputs,
    // +16 = complements (bit 9 is DEAD for deposits — probe mode — so only
    // two clean single-bit regions exist). Intermediates M1/M2 are parked
    // IN-TUPLE: cross-group clones at full-unit distances have exactly-2-row
    // cosets (O(0)→O(4) d=96, O(0)→O(8) d=384), so groups 1-3 are safe row
    // storage while group 0 computes. Assembly order handles the one 4-coset
    // stray (O(8)→O(1) deposits into O(0), overwritten by the M1 fill after).
    uint32_t P=1, N=16;
    vector<uint32_t> a(2048),b(2048),c(2048),got(2048);
    long s_ok=0, c_ok=0, n=0;
    for(int tr=0;tr<TRIALS;tr++){
      for(int s=0;s<2048;s++){ a[s]=((uint32_t)rand()<<15)^rand(); b[s]=((uint32_t)rand()<<15)^rand(); c[s]=((uint32_t)rand()<<15)^rand(); }
      vector<uint32_t> na(2048),nb(2048),nc(2048);
      for(int s=0;s<2048;s++){ na[s]=~a[s]; nb[s]=~b[s]; nc[s]=~c[s]; }
      // inputs land in the two regions at group-0 slots (in gemv these come
      // from the encoding layer, not the host)
      pcwrite(k.O(0)^P,a.data()); pcwrite(k.O(1)^P,b.data()); pcwrite(k.O(2)^P,c.data());
      pcwrite(k.O(0)^N,na.data()); pcwrite(k.O(1)^N,nb.data()); pcwrite(k.O(2)^N,nc.data());
      // M1 = MAJ3(a,b,¬c)
      k.fill(0,P); k.fill(1,P); clone_op(k.O(2)^N,k.O(2)); k.fire_group(0);
      clone_op(k.O(0), k.O(4));                    // park M1 in group1 (d=96, 2-row coset)
      // M2 = MAJ3(a,¬b,c)
      k.fill(0,P); clone_op(k.O(1)^N,k.O(1)); k.fill(2,P); k.fire_group(0);
      clone_op(k.O(0), k.O(8));                    // park M2 in group2 (d=384, 2-row coset)
      // M3 = MAJ3(¬a,b,c) — result stays in group 0 (slot 2 already holds it)
      clone_op(k.O(0)^N,k.O(0)); k.fill(1,P); k.fill(2,P); k.fire_group(0);
      // assemble: M2 → slot1 first (its 4-coset strays into O(0), fixed by
      // the M1 fill next), then M1 → slot0 (2-row coset)
      clone_op(k.O(8), k.O(1));
      clone_op(k.O(4), k.O(0));
      k.fire_group(0);                             // sum = MAJ3(M1,M2,M3)
      rd(k.O(0), got.data());
      for(int ci : k.cm.cols){ if(got[ci]==(a[ci]^b[ci]^c[ci])) s_ok++; }
      // carry = MAJ3(a,b,c) back in group 0 (reload from +1 region)
      k.fill(0,P); k.fill(1,P); k.fill(2,P); k.fire_group(0);
      rd(k.O(0), got.data());
      for(int ci : k.cm.cols){ if(got[ci]==maj3w(a[ci],b[ci],c[ci])) c_ok++; }
      n += k.cm.cols.size();
    }
    printf("fa_sum_ok_pct,fa_carry_ok_pct,clones,majs\n%.4f,%.4f,%ld,%ld\n",
           100.0*s_ok/n,100.0*c_ok/n,OPS_CLONE,OPS_MAJ);
    fprintf(stderr,"[gemv] FA resident: sum %.3f%%, carry %.3f%% on colmask (%ld clones, %ld MAJs, %d trials)\n",
            100.0*s_ok/n,100.0*c_ok/n,OPS_CLONE,OPS_MAJ,TRIALS);
    fflush(stdout); _exit(0);
  }

  // ---- gemv mode: K resident weight planes × encoded activations ----
  // W_k resident at W-region slot (k mod 12 data slots across groups 0-2);
  // for v1 K ≤ 9 lanes share the 9 data slots of groups 0..2 → schedule
  // CSA sequentially through group 0 (the compute group).
  if (K > 9) K = 9;
  vector<vector<uint32_t>> W(K, vector<uint32_t>(2048));
  const int dslot[9] = {0,1,2,4,5,6,8,9,10};
  for(int kk=0;kk<K;kk++){
    for(int s=0;s<2048;s++) W[kk][s]=((uint32_t)rand()<<15)^rand();
    pcwrite(k.O(dslot[kk])^k.DW, W[kk].data());          // resident W
  }
  long lane_err=0, nsamp=0; long cmds_total=0;
  struct timespec t0,t1; double wall=0;
  for(int tr=0;tr<TRIALS;tr++){
    // activation bits for this pass (host-known, NEVER written to DRAM)
    vector<int> act(K);
    for(int kk=0;kk<K;kk++) act[kk] = (rand()%100 < DENS) ? 1 : 0;
    OPS_CLONE=OPS_MAJ=0; OPS_SKIP=0;
    clock_gettime(CLOCK_MONOTONIC,&t0);
    // §V ENCODING: product list = W rows where act=1 (clone into compute
    // slots when consumed); act=0 products are SKIPPED entirely (§V-D)
    vector<int> prod;                 // product source slots (W region)
    for(int kk=0;kk<K;kk++){ if(act[kk]) prod.push_back(dslot[kk]); else OPS_SKIP++; }
    // v1 measures the ENCODING LAYER: create every live product by a
    // clone into its compute slot (the tree's consumption is measured
    // separately in fa mode; full fused tree = v2 after the FA numbers).
    for(size_t pi=0; pi<prod.size(); pi++)
      clone_op(k.O(prod[pi])^k.DW, k.O(prod[pi]));
    clock_gettime(CLOCK_MONOTONIC,&t1);
    wall += (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9;
    cmds_total += OPS_CLONE + OPS_MAJ;
    // v1 correctness (outside timing): read back each live product's compute
    // slot — it must equal the resident W_k (activations never touched DRAM)
    for(size_t pi=0; pi<prod.size(); pi++){
      vector<uint32_t> got(2048);
      rd(k.O(prod[pi]), got.data());
      int kk=-1; for(int q=0;q<K;q++) if(dslot[q]==prod[pi]) kk=q;
      for(int ci : k.cm.cols){ if(got[ci]!=W[kk][ci]) lane_err++; nsamp++; }
    }
  }
  printf("K,density_pct,trials,product_err_pct,cmds_per_pass,skipped_per_pass,wall_ms_per_pass\n");
  printf("%d,%d,%d,%.4f,%.1f,%.1f,%.3f\n",K,DENS,TRIALS,
         nsamp?100.0*lane_err/nsamp:0.0,(double)cmds_total/TRIALS,(double)OPS_SKIP,wall*1000/TRIALS);
  fprintf(stderr,"[gemv] K=%d dens=%d%%: encoded-product err %.4f%%; %.1f cmds/pass (%.1f skipped); %.2f ms/pass\n",
          K,DENS,nsamp?100.0*lane_err/nsamp:0.0,(double)cmds_total/TRIALS,(double)OPS_SKIP,wall*1000/TRIALS);
  fflush(stdout); _exit(0);
}
