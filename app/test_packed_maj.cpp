// Multi-body MAJ packing — throughput unlock (2026-07-17, Task T2). M2's
// in-DRAM popcount works but is round-trip-bound: each MAJ is its own
// Program (one h2c dispatch) + one receiveData (one c2h). The 8K-IMEM
// bitstream (T1) lets us pack many independent MAJ bodies into ONE Program
// with ONE multi-row readback, amortizing the per-program overhead. This
// tool measures the amortization directly: M independent MAJ3 ops on M
// row-disjoint tuples, run (a) unpacked = M Programs + M receives, (b)
// packed = 1 Program with M (doubleACT + labelled rdRow) bodies + M
// receives from the single execute. Both verified bit-exact vs host MAJ3.
// The packed µs/MAJ is the number that decides M2's host-popcount crossover.
// Argv: <bender> <calib> <bank> <s_id_csv> [reps=20]
// (s_id_csv = comma list; the first row-disjoint tuple per s_id is used)
#include "instruction.h"
#include "prog.h"
#include "platform.h"
#include "../util.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>
using namespace std;

static const int CHUNK_COLS[3] = {43, 43, 42};
static SoftMCPlatform* PF = nullptr;
static int BANK = 0;

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

struct Calib { int s_id, bank; uint32_t Rf, Rs; vector<uint32_t> open; };
static vector<Calib> read_all(const string& path, int wb) {
  vector<Calib> out; ifstream f(path); string line;
  while (getline(f, line)) { if (line.empty()||line[0]=='#') continue; istringstream is(line);
    Calib c; if(!(is>>c.s_id>>c.bank>>c.Rf>>c.Rs)) continue; uint32_t v; while(is>>v) c.open.push_back(v);
    if (c.open.size()==16 && c.bank==wb) out.push_back(c); }
  return out;
}
static uint32_t maj3w(uint32_t a,uint32_t b,uint32_t c){ return (a&b)|(a&c)|(b&c); }

// emit one body (doubleACT and/or labelled read) into an existing Program
static void emit_maj_body(Program& p, const Calib& c, int label,
                          bool do_maj = true, bool do_read = true) {
  p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_SLEEP(6));
  if (do_maj) {
    p.add_below(doubleACT(0, 0, c.Rf, c.Rs));
    p.add_inst(SMC_SLEEP(6)); p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_SLEEP(6));
  }
  if (do_read) {
    p.add_below(rdRow_immediate_label(BAR, c.open[0], label));
    p.add_inst(all_nops()); p.add_inst(all_nops());
    p.add_below(PRE(BAR, 0, 0));
  }
  p.add_inst(SMC_SLEEP(4));
}

int main(int argc,char**argv){
  if(argc<5){ fprintf(stderr,"Usage: %s <bender> <calib> <bank> <s_id_csv|*> [reps=20] [maxM=64]\n",argv[0]); return 1; }
  int bender=atoi(argv[1]); BANK=atoi(argv[3]);
  int reps=(argc>5)?atoi(argv[5]):20;
  vector<Calib> all = read_all(argv[2], BANK);

  // s_id_csv: comma list of allowed s_ids, or "*" for all. Greedily select
  // as many mutually ROW-DISJOINT tuples as possible (maximizes packing M).
  set<int> want; bool anysid=false;
  { stringstream ss(argv[4]); string t; while(getline(ss,t,',')) if(!t.empty()){ if(t=="*") anysid=true; else want.insert(atoi(t.c_str())); } }
  int MAXM = (argc>6)?atoi(argv[6]):64;
  vector<Calib> T; set<uint32_t> used_rows;
  for (auto& c : all) {
    if (!anysid && !want.count(c.s_id)) continue;
    bool clash=false; for (uint32_t r : c.open) if (used_rows.count(r)) { clash=true; break; }
    if (clash) continue;
    for (uint32_t r : c.open) used_rows.insert(r);
    T.push_back(c);
    if ((int)T.size() >= MAXM) break;
  }
  int M = T.size();
  if (M < 1) { fprintf(stderr,"no tuples found\n"); return 2; }
  fprintf(stderr,"[pack] bender=%d bank=%d packing M=%d MAJ3 bodies:", bender, BANK, M);
  for (auto& c : T) fprintf(stderr," s%d@%u", c.s_id, c.Rf);
  fprintf(stderr,"\n");

  SoftMCPlatform pf(bender);
  if(pf.init()!=SOFTMC_SUCCESS){ fprintf(stderr,"init fail\n"); return 3; }
  pf.reset_fpga(); PF=&pf; srand(31);

  // per-tuple operands: a=0xAAA.., b=0xCCC.., c=random; expected maj3
  const int A3[5]={1,4,7,10,13}, B3[5]={2,5,8,11,14}, C3[5]={3,6,9,12,15};
  vector<vector<uint32_t>> av(M,vector<uint32_t>(2048)), bv(M,vector<uint32_t>(2048)), cv(M,vector<uint32_t>(2048)), ev(M,vector<uint32_t>(2048));
  for(int m=0;m<M;m++) for(int s=0;s<2048;s++){
    av[m][s]=0xAAAAAAAAu^(s*2654435761u); bv[m][s]=0xCCCCCCCCu^(s*40503u); cv[m][s]=((uint32_t)rand()<<15)^rand();
    ev[m][s]=maj3w(av[m][s],bv[m][s],cv[m][s]);
  }
  // load all tuples' operands (setup, not timed)
  auto load_all=[&](){
    for(int m=0;m<M;m++){ auto&c=T[m];
      for(int k=0;k<5;k++){ pcwrite(c.open[A3[k]],av[m].data()); pcwrite(c.open[B3[k]],bv[m].data()); pcwrite(c.open[C3[k]],cv[m].data()); }
      wrconst(c.open[0],0x00000000u);
    }
  };

  auto verify=[&](vector<vector<uint8_t>>& rb)->long{
    long bad=0;
    for(int m=0;m<M;m++){ for(int s=0;s<2048;s++){ uint32_t g; memcpy(&g,&rb[m][s*4],4); if(g!=ev[m][s]) bad++; } }
    return bad;
  };

  struct timespec t0,t1;
  vector<vector<uint8_t>> rb(M, vector<uint8_t>(8192));

  // ---- unpacked: M programs + M receives ----
  double up_s=0; long up_bad=0;
  for(int r=0;r<reps;r++){
    load_all();
    clock_gettime(CLOCK_MONOTONIC,&t0);
    for(int m=0;m<M;m++){
      Program p; p.add_inst(SMC_LI(8,CASR)); p.add_inst(SMC_LI(BANK,BAR)); p.add_inst(SMC_LI(128,NUM_COLS_REG));
      emit_maj_body(p, T[m], 0); p.add_inst(SMC_END());
      PF->execute(p); PF->receiveData(rb[m].data(),8192);
    }
    clock_gettime(CLOCK_MONOTONIC,&t1);
    up_s += (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9;
    if(r==0) up_bad=verify(rb);
  }

  // ---- packed: 1 program (M bodies) + M receives from the one execute ----
  double pk_s=0; long pk_bad=0; int pk_insts=0;
  for(int r=0;r<reps;r++){
    load_all();
    clock_gettime(CLOCK_MONOTONIC,&t0);
    Program p; p.add_inst(SMC_LI(8,CASR)); p.add_inst(SMC_LI(BANK,BAR)); p.add_inst(SMC_LI(128,NUM_COLS_REG));
    for(int m=0;m<M;m++) emit_maj_body(p, T[m], m);
    p.add_inst(SMC_END());
    PF->execute(p);
    for(int m=0;m<M;m++) PF->receiveData(rb[m].data(),8192);
    clock_gettime(CLOCK_MONOTONIC,&t1);
    pk_s += (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9;
    if(r==0){ pk_bad=verify(rb); pk_insts=p.size(); }
  }

  // ---- primitive rates (packed): read-only bodies; maj-only bodies + 1 read fence ----
  double ro_s=0;
  for(int r=0;r<reps;r++){
    clock_gettime(CLOCK_MONOTONIC,&t0);
    Program p; p.add_inst(SMC_LI(8,CASR)); p.add_inst(SMC_LI(BANK,BAR)); p.add_inst(SMC_LI(128,NUM_COLS_REG));
    for(int m=0;m<M;m++) emit_maj_body(p, T[m], m, false, true);
    p.add_inst(SMC_END());
    PF->execute(p);
    for(int m=0;m<M;m++) PF->receiveData(rb[m].data(),8192);
    clock_gettime(CLOCK_MONOTONIC,&t1);
    ro_s += (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9;
  }
  double mo_s=0;
  for(int r=0;r<reps;r++){
    clock_gettime(CLOCK_MONOTONIC,&t0);
    Program p; p.add_inst(SMC_LI(8,CASR)); p.add_inst(SMC_LI(BANK,BAR)); p.add_inst(SMC_LI(128,NUM_COLS_REG));
    for(int m=0;m<M;m++) emit_maj_body(p, T[m], m, true, false);
    emit_maj_body(p, T[0], M, false, true);   // single read as completion fence
    p.add_inst(SMC_END());
    PF->execute(p);
    PF->receiveData(rb[0].data(),8192);
    clock_gettime(CLOCK_MONOTONIC,&t1);
    mo_s += (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9;
  }

  double up_us = up_s/reps/M*1e6, pk_us = pk_s/reps/M*1e6;
  double ro_us = ro_s/reps/M*1e6, mo_us = mo_s/reps/M*1e6;
  printf("M,reps,unpacked_us_per_maj,packed_us_per_maj,speedup,packed_insts,unpacked_bad,packed_bad,readonly_us,majonly_us\n");
  printf("%d,%d,%.2f,%.2f,%.2fx,%d,%ld,%ld,%.2f,%.2f\n",M,reps,up_us,pk_us,up_us/pk_us,pk_insts,up_bad,pk_bad,ro_us,mo_us);
  fprintf(stderr,"[pack] M=%d: unpacked %.1f us/MAJ, packed %.1f -> %.2fx; PRIMITIVES packed: read-only %.1f us, maj-only %.1f us (fence-amortized); bad up=%ld pk=%ld\n",
          M,up_us,pk_us,up_us/pk_us,ro_us,mo_us,up_bad,pk_bad);
  fflush(stdout);
  _exit(0);
}
