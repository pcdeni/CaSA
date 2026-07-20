// M3 part 2 — the fused RESIDENT in-DRAM popcount tile (2026-07-17 late).
// Implements mvdram-repro/M3_TILE_DESIGN.md exactly:
//   - zero-op encoding: resident W/¬W cells ARE the tree leaves; activation
//     bits only select which leaves exist (§V + §V-D);
//   - dual-track all-MAJ3 CSA tree in tuple group 0; values live in cells
//     (slot, region ∈ {+1,+16}); staging via position-preserving hops;
//     in-tuple parking; free-cell allocator with leaf retirement;
//   - the op list is HOST-SIMULATED first with coset-faithful semantics
//     (clones deposit into their whole coset; bit-9 offsets deposit
//     nothing) and must reproduce host popcount BEFORE silicon runs it;
//   - execution packs bodies into ~6800-inst Programs (T2 packing).
// Measures: correctness on the colmask; wall time vs the packed host-
// readout baseline of the live product rows (the 14b crossover), at the
// given activation density.
// Argv: <bender> <colmask> <bank> [density_pct=100] [trials=5] [seed=11] [simonly=0]
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
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>
using namespace std;

// ---------------- op list ----------------
struct Op { int t; uint32_t a, b; };   // t: 0=CLONE(src,dst) 1=MAJ4(rA,rB[ref=rB]) 2=WRC(row,val) 3=READ(row,id)
static vector<Op> OPS;
static void opClone(uint32_t s, uint32_t d){ OPS.push_back({0,s,d}); }
static void opMaj4(uint32_t rA, uint32_t rB){ OPS.push_back({1,rA,rB}); }
static void opWrc(uint32_t r, uint32_t v){ OPS.push_back({2,r,v}); }
static void opRead(uint32_t r, uint32_t id){ OPS.push_back({3,r,id}); }

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

// ---------------- scheduler ----------------
// cell = slot s (0..15, position s%4 != 3) + region D ∈ {1,16}
struct Cell { int s; uint32_t D; };
struct Val { Cell pos, neg; };          // dual-track value
static Cm CM; static vector<uint32_t> OPEN;
static uint32_t O(int j){ return OPEN[j]; }
static set<uint32_t> LIVE;              // live tuple rows (parked values in-tuple)
static vector<Cell> FREE_CELLS;
static set<int> RETIRED;                // retired leaf slots

// position-aware allocation: a Val's pos and neg cells always share one
// within-group position, so any mixed-polarity staging triple of three
// distinct-position Vals stages onto distinct operand slots.
static Cell cell_alloc(int wantpos, uint32_t wantD){
  for(size_t i=0;i<FREE_CELLS.size();i++)
    if(FREE_CELLS[i].s%4==wantpos && FREE_CELLS[i].D==wantD){
      Cell c=FREE_CELLS[i]; FREE_CELLS.erase(FREE_CELLS.begin()+i); return c; }
  fprintf(stderr,"[tile] no free cell at pos %d region +%u\n",wantpos,wantD); _exit(9);
}
static void cell_free(Cell c){ FREE_CELLS.push_back(c); }
// choose a position (prefer want) where BOTH regions have a free cell —
// keeps the invariant that a Val's two tracks share one position
static int choose_pos(int want){
  for(int k=0;k<3;k++){ int p=(want+k)%3; bool h1=false,h16=false;
    for(auto& c : FREE_CELLS){ if(c.s%4==p){ if(c.D==1u)h1=true; else if(c.D==16u)h16=true; } }
    if(h1&&h16) return p; }
  fprintf(stderr,"[tile] no position with a free pair\n"); _exit(9);
}

// stage value from cell (s,D) into group-0 operand slot p = s%4 (∈0..2).
// route: cell→O(s), then hops to O(p) avoiding LIVE transit rows.
static void stage(Cell c){
  int s=c.s, p=s%4, g=s/4;
  opClone(O(s)^c.D, O(s));
  if (g==0) return;
  if (g==1 || g==2){ opClone(O(s), O(p)); return; }
  // g==3: two hops, via g1 (slot 4+p) or g2 (slot 8+p), avoid live rows
  int via = LIVE.count(O(4+p)) ? 8+p : 4+p;
  if (LIVE.count(O(via))){ fprintf(stderr,"[tile] router stuck slot %d\n", s); _exit(9); }
  opClone(O(s), O(via));
  opClone(O(via), O(p));
}
// one MAJ3 over staged slots 0,1,2 of group 0 (ref = slot 3, done in-body)
static void fire0(){ opMaj4(O(0), O(3)); }
// park current group-0 result into a fresh cell at the REQUESTED position
static Cell park(int wantpos, uint32_t wantD){
  Cell c = cell_alloc(wantpos, wantD);
  int p=c.s%4, g=c.s/4;
  // route group0 result (any row O(0..2)) to O(c.s), then out to region
  if (g==0){ opClone(O(p), O(p)^c.D); }              // direct (result is in all 4 rows)
  else if (g==1||g==2){ opClone(O(p), O(c.s)); opClone(O(c.s), O(c.s)^c.D); }
  else { int via = LIVE.count(O(4+p)) ? 8+p : 4+p;
         opClone(O(p), O(via)); opClone(O(via), O(c.s)); opClone(O(c.s), O(c.s)^c.D); }
  return c;
}
// dual-track FA: (a,b,cin) -> (sum @pSum, carry @pCarry), all cells resident
static Val fa(Val A, Val B, Val C, int pSum, int pCarry){
  // ---- positive sum: M1=MAJ3(a,b,¬c) M2=MAJ3(a,¬b,c) M3=MAJ3(¬a,b,c); sum=MAJ3(M1,M2,M3)
  auto inner=[&](Cell x, Cell y, Cell z, int parkslot){
    // stage x→slot(x.s%4) — but operands must land on DISTINCT slots 0,1,2.
    // scheduler guarantee: A,B,C cells sit at distinct positions 0,1,2
    stage(x); stage(y); stage(z); fire0();
    // parks live in the position-3 lane (slots 7/11): those rows are never
    // cells and never staging transits, so leaf stagings can't clobber them
    if (parkslot==7){ opClone(O(3), O(7)); LIVE.insert(O(7)); }
    else if (parkslot==11){ opClone(O(3), O(11)); LIVE.insert(O(11)); }
  };
  // parks use distinct positions (O(1)→O(5) d=96, O(2)→O(10) d=384), so at
  // most one transit via per position is blocked — the router always routes.
  // Assembly clones are pure 2-row cosets: no strays at all.
  inner(A.pos, B.pos, C.neg, 7);         // M1 → O(7) (from O(3), d=96)
  inner(A.pos, B.neg, C.pos, 11);        // M2 → O(11) (from O(3), d=384)
  inner(A.neg, B.pos, C.pos, -1);        // M3 stays in group 0 (slot 0 holds it)
  // assembly strays hit only O(3) (rewritten as ref) and O(5)/O(10) residue
  opClone(O(7), O(1)); LIVE.erase(O(7));   // M1 → slot 1
  opClone(O(11), O(2)); LIVE.erase(O(11)); // M2 → slot 2
  fire0();
  int psum = choose_pos(pSum);
  Val R; R.pos = park(psum, 1u);
  // ---- negative sum (mirror): M1'=(¬a,¬b,c) M2'=(¬a,b,¬c) M3'=(a,¬b,¬c)
  inner(A.neg, B.neg, C.pos, 7);
  inner(A.neg, B.pos, C.neg, 11);
  inner(A.pos, B.neg, C.neg, -1);
  opClone(O(7), O(1)); LIVE.erase(O(7));
  opClone(O(11), O(2)); LIVE.erase(O(11));
  fire0();
  R.neg = park(psum, 16u);
  // ---- carry both tracks; inputs retire at their LAST staging use so the
  // cell pool self-sustains (parks would otherwise exhaust it)
  stage(A.pos); stage(B.pos); stage(C.pos); fire0();
  cell_free(A.pos); cell_free(B.pos); cell_free(C.pos);
  int pcar = choose_pos(pCarry);
  Val Cy; Cy.pos = park(pcar, 1u);
  stage(A.neg); stage(B.neg); stage(C.neg); fire0();
  cell_free(A.neg); cell_free(B.neg); cell_free(C.neg);
  Cy.neg = park(pcar, 16u);
  extern Val CARRY_OUT; CARRY_OUT = Cy;
  return R;
}
Val CARRY_OUT;

// ---------------- simulator (coset-faithful) ----------------
struct Sim {
  map<uint32_t, vector<uint32_t>> mem;
  TupleAlloc* TA;
  vector<uint32_t>& row(uint32_t r){ auto it=mem.find(r); if(it==mem.end()) it=mem.emplace(r, vector<uint32_t>(2048,0xDEADBEEFu)).first; return it->second; }
  vector<uint32_t> deposit_set(uint32_t src, uint32_t dst){
    uint32_t d = src^dst;
    if (d & 0x200u) return {};                       // bit 9: dead generator
    static const uint32_t G[4][2] = {{2,4},{8,16},{32,64},{128,256}};
    vector<uint32_t> units;
    if (d & 1u) units.push_back(1u);
    for (auto& g : G){ uint32_t m=(g[0]|g[1])&d; if(m) units.push_back(m); }
    vector<uint32_t> out;
    for (uint32_t m=0; m < (1u<<units.size()); m++){
      uint32_t S=0; for (size_t j=0;j<units.size();j++) if((m>>j)&1) S^=units[j];
      out.push_back(src^S);
    }
    return out;
  }
  void run(const vector<Op>& ops, map<uint32_t,vector<uint32_t>>& reads){
    for (auto& op : ops){
      if (op.t==0){ auto& s=row(op.a); for (uint32_t r : deposit_set(op.a,op.b)) if(r!=op.a) row(r)=s; }
      else if (op.t==1){ uint32_t base=op.a; // group of 4: {base, base^6, base^8, base^14} sorted = slots
        // find group index: base == O(g*4)
        int g=-1; for(int q=0;q<4;q++) if(O(q*4)==op.a) g=q;
        auto a=row(O(g*4)), b=row(O(g*4+1)), c=row(O(g*4+2));
        vector<uint32_t> res(2048);
        for(int s2=0;s2<2048;s2++) res[s2]=maj3w(a[s2],b[s2],c[s2]);
        for(int q=0;q<4;q++) row(O(g*4+q))=res;
        (void)base;
      }
      else if (op.t==2){ auto& r=row(op.a); for(int s2=0;s2<2048;s2++) r[s2]=op.b; }
      else { reads[op.b] = row(op.a); }
    }
  }
};

// ---------------- silicon executor (packed) ----------------
static SoftMCPlatform* PF=nullptr; static int BANK=0;
static int emit_body(Program& p, const Op& op, int& lbl){
  if (op.t==0){
    p.add_below(PRE(BAR,0,0)); p.add_inst(SMC_SLEEP(6));
    p.add_below(doubleACT(30,1,op.a,op.b));
    p.add_inst(SMC_SLEEP(6)); p.add_below(PRE(BAR,0,0)); p.add_inst(SMC_SLEEP(4));
    return 14;
  }
  if (op.t==1){
    p.add_below(PRE(BAR,0,0));
    p.add_below(wrRow_immediate_label(BAR,op.b,0xFFFFFFFFu,lbl++));  // ref=ONE
    p.add_inst(SMC_SLEEP(6)); p.add_below(PRE(BAR,0,0)); p.add_inst(SMC_SLEEP(6));
    for(int j=0;j<3;j++){ // 3 frac pulses t0 on ref
      p.add_inst(all_nops()); p.add_inst(SMC_LI(op.b,RF_REG));
      Mininst q[4]={SMC_ACT(BAR,0,RF_REG,0),SMC_PRE(BAR,0,0),SMC_NOP(),SMC_NOP()};
      p.add_inst(q[0],q[1],q[2],q[3]); p.add_inst(SMC_SLEEP(6));
    }
    p.add_below(PRE(BAR,0,0)); p.add_inst(SMC_SLEEP(6));
    p.add_below(doubleACT(0,0,op.a,op.b));
    p.add_inst(SMC_SLEEP(6)); p.add_below(PRE(BAR,0,0)); p.add_inst(SMC_SLEEP(4));
    return 130;
  }
  if (op.t==2){
    p.add_below(PRE(BAR,0,0));
    p.add_below(wrRow_immediate_label(BAR,op.a,op.b,lbl++));
    p.add_inst(SMC_SLEEP(6)); p.add_below(PRE(BAR,0,0));
    return 95;
  }
  p.add_below(PRE(BAR,0,0)); p.add_inst(SMC_SLEEP(6));
  p.add_below(rdRow_immediate_label(BAR,op.a,lbl++));
  p.add_inst(all_nops()); p.add_inst(all_nops());
  p.add_below(PRE(BAR,0,0));
  return 100;
}
static double exec_packed(const vector<Op>& ops, map<uint32_t,vector<uint32_t>>& reads){
  struct timespec t0,t1; clock_gettime(CLOCK_MONOTONIC,&t0);
  size_t i=0;
  vector<uint32_t> pending;   // read ids in flight for current program
  while (i<ops.size()){
    Program p; p.add_inst(SMC_LI(8,CASR)); p.add_inst(SMC_LI(BANK,BAR)); p.add_inst(SMC_LI(128,NUM_COLS_REG));
    int est=10, lbl=0; pending.clear();
    while (i<ops.size() && est<6800){
      if (ops[i].t==3) pending.push_back(ops[i].b);
      est += emit_body(p, ops[i], lbl);
      i++;
    }
    p.add_inst(SMC_END());
    PF->execute(p);
    for (uint32_t id : pending){
      vector<uint32_t> v(2048); uint8_t buf[8192];
      if (PF->receiveData(buf,8192)!=8192){ fprintf(stderr,"recv fail\n"); _exit(5); }
      for(int s=0;s<2048;s++) memcpy(&v[s],&buf[s*4],4);
      reads[id]=v;
    }
  }
  clock_gettime(CLOCK_MONOTONIC,&t1);
  return (t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)/1e9;
}
static void pcwrite(uint32_t row, const uint32_t* seg){
  static const int CH[3]={43,43,42}; int cs=0;
  for (int ch=0; ch<3; ch++){
    int n=CH[ch]; const uint32_t* cd=seg+cs*16;
    Program p; p.add_inst(SMC_LI(8,CASR)); p.add_inst(SMC_LI(BANK,BAR));
    p.add_inst(SMC_LI(row,RAR)); p.add_inst(SMC_LI(cs*8,CAR));
    p.add_below(PRE(BAR,0,0)); p.add_below(ACT(BAR,0,RAR,0));
    for(int k2=0;k2<n;k2++){ const uint32_t* sl=cd+k2*16;
      for(int s=0;s<16;s++){ p.add_inst(SMC_LI(sl[s],PATTERN_REG)); p.add_inst(SMC_LDWD(PATTERN_REG,s)); }
      p.add_below(WRITE(BAR,CAR,1)); p.add_inst(SMC_SLEEP(8)); }
    p.add_inst(SMC_SLEEP(8)); p.add_below(PRE(BAR,0,0)); p.add_inst(SMC_SLEEP(4)); p.add_inst(SMC_END());
    PF->execute(p); cs+=n;
  }
}

int main(int argc,char**argv){
  if(argc<4){ fprintf(stderr,"Usage: %s <bender> <colmask> <bank> [density=100] [trials=5] [seed=11] [simonly=0]\n",argv[0]); return 1; }
  int bender=atoi(argv[1]); BANK=atoi(argv[3]);
  int DENS=(argc>4)?atoi(argv[4]):100;
  int TRIALS=(argc>5)?atoi(argv[5]):5;
  unsigned seed=(argc>6)?(unsigned)atoi(argv[6]):11;
  int SIMONLY=(argc>7)?atoi(argv[7]):0;
  if(!read_colmask(argv[2],CM)){ fprintf(stderr,"bad colmask\n"); return 2; }
  OPEN=CM.open;
  TupleAlloc TA; TA.open=CM.open; TA.Rf=CM.Rf; TA.Rs=CM.Rs;
  if(!TA.fit()){ fprintf(stderr,"fit failed\n"); return 2; }
  const int K=9; const int dslot[9]={0,1,2,4,5,6,8,9,10};
  srand(seed);

  SoftMCPlatform* pf=nullptr;
  if(!SIMONLY){
    pf=new SoftMCPlatform(bender);
    if(pf->init()!=SOFTMC_SUCCESS){ fprintf(stderr,"init fail\n"); return 3; }
    pf->reset_fpga(); PF=pf;
  }

  long lane_err=0, nsamp=0; double tile_s=0, base_s=0; long ops_n=0; int live_n=0;
  for(int tr=0;tr<TRIALS;tr++){
    // weights + activations
    vector<vector<uint32_t>> W(K, vector<uint32_t>(2048));
    for(int k=0;k<K;k++) for(int s=0;s<2048;s++) W[k][s]=((uint32_t)rand()<<15)^rand();
    vector<int> act(K); for(int k=0;k<K;k++) act[k]=(rand()%100<DENS)?1:0;
    vector<int> leaves; for(int k=0;k<K;k++) if(act[k]) leaves.push_back(k);
    if(leaves.size()<2){ continue; }   // degenerate pass
    live_n=(int)leaves.size();
    // host expectation
    int BOUT=0; { int t=(int)leaves.size(); do{BOUT++; t>>=1;}while(t); }
    vector<vector<uint32_t>> Rexp(BOUT, vector<uint32_t>(2048,0));
    for(int s=0;s<2048;s++) for(int t=0;t<32;t++){
      int cnt=0; for(int k : leaves) cnt += (W[k][s]>>t)&1;
      for(int b=0;b<BOUT;b++) if((cnt>>b)&1) Rexp[b][s]|=1u<<t;
    }
    // ---- build the op list ----
    OPS.clear(); LIVE.clear(); FREE_CELLS.clear(); RETIRED.clear();
    for(int g=3;g>=3;g--) for(int p=0;p<3;p++) for(uint32_t D : {1u,16u}) FREE_CELLS.push_back({12+p,D});
    // leaves as dual-track cells (W at +1, ¬W at +16); retirement frees them
    struct Node { Val v; int bucket; };
    vector<vector<Val>> buckets(BOUT+2);
    for(int k : leaves){ Val v; v.pos={dslot[k],1u}; v.neg={dslot[k],16u}; buckets[0].push_back(v); }
    // CSA: positions of any FA's three inputs must be distinct 0,1,2 —
    // scheduler picks triples with distinct positions; leaves at dslot cover
    // positions 0,1,2 evenly; parked cells at position of their slot.
    auto pos_of=[&](const Val& v){ return v.pos.s%4; };
    // move a value to a fresh cell pair at a new position (through group 0)
    auto move_val=[&](Val v, int newpos)->Val{
      // no MAJ involved: stage the track into group 0, cross-position clone
      // inside the group (small-unit cosets stay within the group), park.
      int np = choose_pos(newpos);
      int p = v.pos.s%4;
      Val r;
      stage(v.pos); if(p!=np) opClone(O(p), O(np));
      cell_free(v.pos); r.pos = park(np, 1u);
      stage(v.neg); if(p!=np) opClone(O(p), O(np));
      cell_free(v.neg); r.neg = park(np, 16u);
      return r;
    };
    for(int b=0;b<=BOUT;b++){
      while(buckets[b].size()>=3){
        // pick any 3 with pairwise-distinct positions; reposition if stuck
        int idx[3]={-1,-1,-1}; int got=0; bool used[64]={false}; bool posused[3]={false,false,false};
        for(size_t i2=0;i2<buckets[b].size() && got<3;i2++){
          int p=pos_of(buckets[b][i2]);
          if(!posused[p]){ posused[p]=true; used[i2]=true; idx[got++]=(int)i2; }
        }
        if(got<3){
          // reposition the first unused value to a missing position
          int missing=0; while(posused[missing]) missing++;
          for(size_t i2=0;i2<buckets[b].size();i2++) if(!used[i2]){
            buckets[b][i2]=move_val(buckets[b][i2], missing); break; }
          continue;
        }
        Val A=buckets[b][idx[0]], Bv=buckets[b][idx[1]], Cv=buckets[b][idx[2]];
        vector<int> rem={idx[0],idx[1],idx[2]}; sort(rem.rbegin(),rem.rend());
        for(int r : rem) buckets[b].erase(buckets[b].begin()+r);
        int pS=(int)buckets[b].size()%3, pC=(int)buckets[b+1].size()%3;
        Val S=fa(A,Bv,Cv,pS,pC); Val Cy=CARRY_OUT;   // fa() retires inputs
        buckets[b].push_back(S); buckets[b+1].push_back(Cy);
      }
      if(buckets[b].size()==2){
        Val A=buckets[b][0], Bv=buckets[b][1];
        if(pos_of(A)==pos_of(Bv)) Bv=buckets[b][1]=move_val(Bv,(pos_of(A)+1)%3);
        int zp=3-pos_of(A)-pos_of(Bv);              // the missing position
        Cell z0=cell_alloc(zp,1u), z1=cell_alloc(zp,16u);
        opWrc(O(z0.s)^z0.D, 0x00000000u); opWrc(O(z1.s)^z1.D, 0xFFFFFFFFu);
        Val Z; Z.pos=z0; Z.neg=z1;
        buckets[b].clear();
        int pS=0, pC=(int)buckets[b+1].size()%3;
        Val S=fa(A,Bv,Z,pS,pC); Val Cy=CARRY_OUT;   // fa() retires inputs incl. Z
        buckets[b].push_back(S); buckets[b+1].push_back(Cy);
      }
    }
    // readout: bucket b's single value (positive track)
    for(int b=0;b<BOUT;b++){
      if(buckets[b].empty()){ opWrc(O(12)^1u, 0); opClone(O(12)^1u, O(12)); opRead(O(12), b); continue; }
      Cell c=buckets[b][0].pos;
      opClone(O(c.s)^c.D, O(c.s)); opRead(O(c.s), b);
    }
    ops_n=(long)OPS.size();
    // ---- simulate ----
    Sim sim; sim.TA=&TA;
    for(int k : leaves){ sim.row(O(dslot[k])^1u)=W[k]; auto nw=W[k]; for(auto& w:nw) w=~w; sim.row(O(dslot[k])^16u)=nw; }
    map<uint32_t,vector<uint32_t>> simreads;
    sim.run(OPS, simreads);
    long sim_bad=0;
    for(int b=0;b<BOUT;b++){ auto it=simreads.find(b);
      long bb=0;
      if(it==simreads.end()) bb=2048;
      else for(int s=0;s<2048;s++) if(it->second[s]!=Rexp[b][s]) bb++;
      if(bb) fprintf(stderr,"[tile]   sim bucket %d: %ld/2048 wrong; got[0]=%08x exp[0]=%08x got~exp? %s\n",
        b,bb,(it!=simreads.end())?it->second[0]:0,Rexp[b][0],
        (it!=simreads.end()&&it->second[0]==~Rexp[b][0])?"YES":"no");
      sim_bad+=bb;
    }
    if(sim_bad){ fprintf(stderr,"[tile] SIM MISMATCH %ld words (trial %d) — schedule bug, aborting silicon\n",sim_bad,tr); if(SIMONLY){continue;} _exit(7); }
    if(SIMONLY){ fprintf(stderr,"[tile] trial %d SIM OK (%zu ops, %d leaves)\n",tr,OPS.size(),live_n); continue; }
    // ---- silicon: preload W, execute packed, verify, time ----
    for(int k : leaves){ pcwrite(O(dslot[k])^1u, W[k].data());
      vector<uint32_t> nw=W[k]; for(auto&w:nw) w=~w; pcwrite(O(dslot[k])^16u, nw.data()); }
    map<uint32_t,vector<uint32_t>> got;
    tile_s += exec_packed(OPS, got);
    for(int ci : CM.cols){ bool ok=true;
      for(int b=0;b<BOUT;b++){ auto it=got.find(b); if(it==got.end()||it->second[ci]!=Rexp[b][ci]){ ok=false; break; } }
      if(!ok) lane_err++; nsamp++;
    }
    // ---- baseline: packed readout of the live product rows ----
    vector<Op> bl;
    for(size_t i2=0;i2<leaves.size();i2++){ bl.push_back({3, O(dslot[leaves[i2]])^1u, (uint32_t)(100+i2)}); }
    map<uint32_t,vector<uint32_t>> bgot;
    base_s += exec_packed(bl, bgot);
  }
  if(SIMONLY){ fprintf(stderr,"[tile] SIMONLY done\n"); return 0; }
  printf("density,trials,live_K,ops,lane_err_pct,tile_ms,baseline_readout_ms,ratio\n");
  printf("%d,%d,%d,%ld,%.4f,%.2f,%.2f,%.2f\n",DENS,TRIALS,live_n,ops_n,
         nsamp?100.0*lane_err/nsamp:0.0,tile_s*1000/TRIALS,base_s*1000/TRIALS,
         base_s>0?tile_s/base_s:0.0);
  fprintf(stderr,"[tile] dens=%d%%: lane-err %.3f%%; tile %.2f ms vs readout-baseline %.2f ms (ratio %.2fx); %ld ops\n",
          DENS,nsamp?100.0*lane_err/nsamp:0.0,tile_s*1000/TRIALS,base_s*1000/TRIALS,
          base_s>0?tile_s/base_s:0.0,ops_n);
  fflush(stdout); _exit(0);
}
