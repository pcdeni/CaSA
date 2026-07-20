// In-DRAM popcount accumulation — the MERGE kernel (2026-07-17, Task M2 /
// CLICK 1). The ternary-LLM matvec's readout wall is: for each output lane
// it reads K product rows (weight AND activation) to the host and popcounts
// them there — casa_readout = n_weights × 16/8 bytes (exp0_readout_floor).
// MVDRAM eliminates that by summing IN DRAM. This kernel does the K-input
// popcount with a carry-save-adder tree of dual-track full-adders (the S4
// primitive: sum = MAJ5(a,b,c,¬carry,¬carry) = XOR3; carry = MAJ3(a,b,c) =
// the 3:2 compressor), reading out only the ceil(log2(K+1)) result
// bitplanes instead of all K product planes.
//
// Design: K product rows p_0..p_{K-1} are given as host bitplanes (in a
// real matmul these are AND(w,x) computed in DRAM; here host-provided so
// the accumulator is measured in isolation, exactness vs host popcount).
// Reduction = iterated CSA: maintain a list of "weight buckets"; each bucket
// b holds rows of value 2^b. Repeatedly take any 3 rows in a bucket, run one
// dual-track FA → 1 sum row (same bucket) + 1 carry row (bucket b+1), until
// every bucket has ≤2 rows; then a ripple carry-propagate adds the final two
// numbers. Each FA is the S4 adder's inner op, harvested from tuple row O(0),
// placed by lattice_alloc safe offsets. Result: b_out = ceil(log2(K+1))
// bitplanes read to host and compared to host popcount(sum of p_k) per lane.
//
// Metrics: exactness (per-lane popcount == host) on screened columns;
// readout rows (b_out) vs host-path (K); MAJ-op count; wall time.
// Argv: <bender> <colmask> <bank> [K=8] [trials=20] [seed=5]
#include "instruction.h"
#include "prog.h"
#include "platform.h"
#include "../util.h"
#include "lattice_alloc.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <ctime>
using namespace std;

static const int CHUNK_COLS[3] = {43, 43, 42};
static SoftMCPlatform* PF = nullptr;
static int BANK = 0;
static long MAJ_OPS = 0;

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
static void fire_maj(uint32_t Rf, uint32_t Rs, uint32_t ref_row, uint32_t ref_val, int nf, uint8_t* out) {
  wrconst(ref_row, ref_val);
  Program p;
  p.add_inst(SMC_LI(8, CASR)); p.add_inst(SMC_LI(BANK, BAR)); p.add_inst(SMC_LI(128, NUM_COLS_REG));
  p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_SLEEP(6));
  for (int j = 0; j < nf; j++) { p.add_inst(SMC_SLEEP(6)); p.add_below(frac_builder(0, ref_row)); p.add_inst(SMC_SLEEP(6)); }
  p.add_inst(SMC_SLEEP(6)); p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_SLEEP(6));
  p.add_below(doubleACT(0, 0, Rf, Rs));
  p.add_inst(SMC_SLEEP(6)); p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_SLEEP(6));
  p.add_below(rdRow_immediate(BAR, ref_row));
  p.add_inst(all_nops()); p.add_inst(all_nops()); p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_END());
  PF->execute(p);
  if (PF->receiveData(out, 8192) != 8192) { fprintf(stderr, "recv fail\n"); _exit(5); }
  MAJ_OPS++;
}

struct Cm { vector<uint32_t> open; uint32_t Rf=0, Rs=0; int nf=0; int sid=0; vector<int> cols; string pol; };
static bool read_colmask(const char* path, Cm& c) {
  ifstream f(path); if (!f) return false; string line;
  while (getline(f, line)) {
    if (line.empty()) continue;
    if (line[0] == '#') {
      if (line.find("open_rows:")!=string::npos){ istringstream is(line.substr(line.find(':')+1)); uint32_t v; while(is>>v) c.open.push_back(v); }
      else if (line.find("tuple Rf=")!=string::npos) sscanf(line.c_str(),"# tuple Rf=%u Rs=%u",&c.Rf,&c.Rs);
      else if (line.find("s_id=")!=string::npos&&!c.sid){ size_t p=line.find("s_id="); sscanf(line.c_str()+p,"s_id=%d",&c.sid); }
      else if (line.find("policy=m5_Z2")!=string::npos){ c.nf=2; c.pol="Z2"; }
      else if (line.find("policy=m5_Z0")!=string::npos){ c.nf=0; c.pol="Z0"; }
      continue;
    }
    c.cols.push_back(atoi(line.c_str()));
  }
  return c.open.size()==16 && c.Rf && !c.cols.empty();
}
static uint32_t maj3w(uint32_t a,uint32_t b,uint32_t c){ return (a&b)|(a&c)|(b&c); }

// slot maps
static const int A3[5]={1,4,7,10,13}, B3[5]={2,5,8,11,14}, C3[5]={3,6,9,12,15};
static const int A5[3]={1,6,11}, B5[3]={2,7,12}, CC5[3]={3,8,13}, D5[3]={4,9,14}, E5[3]={5,10,15};

struct Kern {
  Cm* cm; TupleAlloc* TA; uint32_t Dw;   // Dw = safe scratch region offset
  // full adder: inputs a,b,cin (host bitplanes) -> (sum,carry) host bitplanes.
  // computes cbar=MAJ3(¬a,¬b,¬cin) implicitly via De Morgan; sum=MAJ5(a,b,cin,cbar,cbar); carry=MAJ3(a,b,cin)
  void fa(const uint32_t* a, const uint32_t* b, const uint32_t* c, uint32_t* sum, uint32_t* carry) {
    auto O=[&](int j){ return cm->open[j]; };
    uint8_t buf[8192];
    static vector<uint32_t> na(2048), nb(2048), nc(2048), cbar(2048);
    for (int s=0;s<2048;s++){ na[s]=~a[s]; nb[s]=~b[s]; nc[s]=~c[s]; }
    // cbar = MAJ3(¬a,¬b,¬c)
    for(int k=0;k<5;k++){ pcwrite(O(A3[k]),na.data()); pcwrite(O(B3[k]),nb.data()); pcwrite(O(C3[k]),nc.data()); }
    fire_maj(cm->Rf,cm->Rs,O(0),0,0,buf);
    for(int s=0;s<2048;s++) memcpy(&cbar[s],&buf[s*4],4);
    // sum = MAJ5(a,b,c,cbar,cbar)
    for(int k=0;k<3;k++){ pcwrite(O(A5[k]),a); pcwrite(O(B5[k]),b); pcwrite(O(CC5[k]),c); pcwrite(O(D5[k]),cbar.data()); pcwrite(O(E5[k]),cbar.data()); }
    fire_maj(cm->Rf,cm->Rs,O(0),0,cm->nf,buf);
    for(int s=0;s<2048;s++) memcpy(&sum[s],&buf[s*4],4);
    // carry = MAJ3(a,b,c)
    for(int k=0;k<5;k++){ pcwrite(O(A3[k]),a); pcwrite(O(B3[k]),b); pcwrite(O(C3[k]),c); }
    fire_maj(cm->Rf,cm->Rs,O(0),0,0,buf);
    for(int s=0;s<2048;s++) memcpy(&carry[s],&buf[s*4],4);
  }
  // half adder (for the final ripple): sum=XOR, carry=AND -> as MAJ with a
  // zero third input: carry=MAJ3(a,b,0)=AND; sum=MAJ5(a,b,0,¬carry,¬carry)=XOR
  void ha(const uint32_t* a, const uint32_t* b, uint32_t* sum, uint32_t* carry) {
    static vector<uint32_t> z(2048, 0);
    fa(a, b, z.data(), sum, carry);
  }
};

int main(int argc,char**argv){
  if(argc<4){ fprintf(stderr,"Usage: %s <bender> <colmask> <bank> [K=8] [trials=20] [seed=5]\n",argv[0]); return 1; }
  int bender=atoi(argv[1]); BANK=atoi(argv[3]);
  int K=(argc>4)?atoi(argv[4]):8;
  int TRIALS=(argc>5)?atoi(argv[5]):20;
  unsigned seed=(argc>6)?(unsigned)atoi(argv[6]):5;
  if(K<2||K>64){ fprintf(stderr,"K out of range\n"); return 1; }

  Cm cm;
  if(!read_colmask(argv[2],cm)){ fprintf(stderr,"bad colmask %s\n",argv[2]); return 2; }
  TupleAlloc TA; TA.open=cm.open; TA.Rf=cm.Rf; TA.Rs=cm.Rs;
  if(!TA.fit()){ fprintf(stderr,"tuple fit failed\n"); return 2; }
  int BOUT=0; { int t=K; while(t){ BOUT++; t>>=1; } }  // ceil(log2(K+1)) bits
  fprintf(stderr,"[pc] s%d Rf=%u Rs=%u pol=%s K=%d -> %d result bits; cols=%zu; host-path reads %d rows/tile\n",
          cm.sid,cm.Rf,cm.Rs,cm.pol.c_str(),K,BOUT,cm.cols.size(),K);

  SoftMCPlatform pf(bender);
  if(pf.init()!=SOFTMC_SUCCESS){ fprintf(stderr,"init fail\n"); return 3; }
  pf.reset_fpga(); PF=&pf; srand(seed);
  Kern kern; kern.cm=&cm; kern.TA=&TA; kern.Dw=1;

  vector<vector<uint32_t>> P(K, vector<uint32_t>(2048));
  long lane_err=0, bit_err=0, nsamp=0;
  struct timespec t0,t1; clock_gettime(CLOCK_MONOTONIC,&t0); MAJ_OPS=0;

  for(int tr=0; tr<TRIALS; tr++){
    for(int k=0;k<K;k++) for(int s=0;s<2048;s++)
      P[k][s]=((uint32_t)rand()<<17)^((uint32_t)rand()<<3)^(uint32_t)rand();
    // host reference popcount per bit-lane per column
    // result[b][s] = bit b of sum_k bit(P[k][s])
    vector<vector<uint32_t>> Rexp(BOUT, vector<uint32_t>(2048,0));
    for(int s=0;s<2048;s++) for(int t=0;t<32;t++){
      int cnt=0; for(int k=0;k<K;k++) cnt += (P[k][s]>>t)&1;
      for(int b=0;b<BOUT;b++) if((cnt>>b)&1) Rexp[b][s] |= 1u<<t;
    }
    // CSA reduction: buckets[b] = list of rows (host bitplanes) of weight 2^b
    vector<vector<vector<uint32_t>>> buckets(BOUT+1);
    for(int k=0;k<K;k++) buckets[0].push_back(P[k]);
    for(int b=0;b<BOUT;b++){
      // compress bucket b with FAs while ≥3 rows
      while(buckets[b].size()>=3){
        vector<uint32_t> a=buckets[b].back(); buckets[b].pop_back();
        vector<uint32_t> bb=buckets[b].back(); buckets[b].pop_back();
        vector<uint32_t> c=buckets[b].back(); buckets[b].pop_back();
        vector<uint32_t> sum(2048), carry(2048);
        kern.fa(a.data(),bb.data(),c.data(),sum.data(),carry.data());
        buckets[b].push_back(sum);
        if(b+1<=BOUT) buckets[b+1].push_back(carry);
      }
    }
    // now each bucket has ≤2 rows; ripple-add the two "numbers"
    // number X = bit b is buckets[b][0], number Y = bit b is buckets[b][1] (if present)
    static const vector<uint32_t> ZROW(2048, 0);
    vector<vector<uint32_t>> Rgot(BOUT, vector<uint32_t>(2048,0));
    vector<uint32_t> carry(2048,0); bool have_carry=false;
    for(int b=0;b<BOUT;b++){
      const vector<uint32_t>& x = buckets[b].size()>=1 ? buckets[b][0] : ZROW;
      if(buckets[b].size()>=2){
        vector<uint32_t> s1(2048), c1(2048);
        kern.fa(x.data(), buckets[b][1].data(), (have_carry?carry:ZROW).data(), s1.data(), c1.data());
        Rgot[b]=s1; carry=c1; have_carry=true;
      } else if(have_carry){
        vector<uint32_t> s1(2048), c1(2048);
        kern.ha(x.data(), carry.data(), s1.data(), c1.data());
        Rgot[b]=s1; carry=c1; have_carry=true;
      } else {
        Rgot[b]=x; have_carry=false;
      }
    }
    // compare on screened columns
    for(int ci : cm.cols){
      bool lane_ok=true;
      for(int b=0;b<BOUT;b++){ if(Rgot[b][ci]!=Rexp[b][ci]){ bit_err++; lane_ok=false; } }
      if(!lane_ok) lane_err++;
      nsamp++;
    }
    if((tr+1)%5==0) fprintf(stderr,"[pc] trial %d/%d (maj_ops=%ld)\n",tr+1,TRIALS,MAJ_OPS);
  }
  clock_gettime(CLOCK_MONOTONIC,&t1);
  double wall=(t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9;
  double maj_per_tile = (double)MAJ_OPS/TRIALS;
  printf("K,result_bits,cols,trials,lane_err_pct,bit_err_pct,maj_ops_per_tile,readout_rows_indram,readout_rows_host,readout_reduction\n");
  printf("%d,%d,%zu,%d,%.4f,%.4f,%.1f,%d,%d,%.2fx\n",
         K,BOUT,cm.cols.size(),TRIALS,100.0*lane_err/nsamp,100.0*bit_err/(nsamp*BOUT),
         maj_per_tile,BOUT,K,(double)K/BOUT);
  fprintf(stderr,"[pc] === K=%d popcount: lane-exact %.4f%% wrong, bit-err %.4f%%; %.1f MAJ/tile; readout %d rows vs host %d (%.2fx); %.2f s ===\n",
          K,100.0*lane_err/nsamp,100.0*bit_err/(nsamp*BOUT),maj_per_tile,BOUT,K,(double)K/BOUT,wall);
  fflush(stdout);
  _exit(0);
}
