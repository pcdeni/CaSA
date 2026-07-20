// Persistent-weight throughput A/B (2026-07-17) — the real per-token cost.
// dense-matvec-bcast's own header names the wall: after each MAJ3, Rfirst holds
// the RESULT, so back-to-back MAJ3s re-load W via a ~0.7 ms per-column write.
// Safe-load makes the "deep backup outside the open set" (opt-log Tier 2 #5)
// actually work: keep W in a backup row at a SAFE pair-offset from Rfirst, and
// refresh Rfirst each iteration with one cheap doubleACT clone instead of a
// per-column write.
//
//   Mode A (today): per iter = per_column_write(W->Rfirst) + broadcast+MAJ3
//   Mode B (persistent): setup once = per_column_write(W->backup);
//                        per iter = rowclone(backup->Rfirst) + broadcast+MAJ3
//
// Both verify the result == (W & x) each iter. T iters model reuse of the same
// resident weight across tokens/columns. Reports per-iter time + speedup.
// Argv: <bender> <calib_file> <bank> <s_id> <sub_start> [iters]
#include "instruction.h"
#include "prog.h"
#include "platform.h"
#include "../util.h"
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>
using namespace std;
static const int CHUNK_COLS[3] = {43, 43, 42};
static int BANK = 0;

static Program build_chunk_program(uint32_t row_addr, const uint32_t* col_data, int col_start, int n_cols) {
  Program p;
  p.add_inst(SMC_LI(8, CASR)); p.add_inst(SMC_LI(BANK, BAR));
  p.add_inst(SMC_LI(row_addr, RAR)); p.add_inst(SMC_LI(col_start * 8, CAR));
  p.add_below(PRE(BAR, 0, 0)); p.add_below(ACT(BAR, 0, RAR, 0));
  for (int k = 0; k < n_cols; k++) {
    const uint32_t* slots = col_data + k * 16;
    for (int slot = 0; slot < 16; slot++) { p.add_inst(SMC_LI(slots[slot], PATTERN_REG)); p.add_inst(SMC_LDWD(PATTERN_REG, slot)); }
    p.add_below(WRITE(BAR, CAR, 1)); p.add_inst(SMC_SLEEP(8));
  }
  p.add_inst(SMC_SLEEP(8)); p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_SLEEP(4)); p.add_inst(SMC_END());
  return p;
}
static void per_column_write_row(SoftMCPlatform& pf, uint32_t row, const uint32_t* d) {
  int cs = 0; for (int ch = 0; ch < 3; ch++) { Program p = build_chunk_program(row, d + cs * 16, cs, CHUNK_COLS[ch]); pf.execute(p); cs += CHUNK_COLS[ch]; }
}
static void rowclone(SoftMCPlatform& pf, uint32_t src, uint32_t dst) {
  Program p; p.add_inst(SMC_LI(8, CASR)); p.add_inst(SMC_LI(BANK, BAR)); p.add_inst(SMC_LI(128, NUM_COLS_REG));
  p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_SLEEP(6));
  p.add_below(doubleACT(30, 1, src, dst)); p.add_inst(SMC_SLEEP(6)); p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_SLEEP(6));
  p.add_inst(SMC_END()); pf.execute(p);
}
// broadcast + uniform overwrite + frac + MAJ3 + read (verbatim structure from dense-matvec-bcast)
static Program build_bmaj(uint32_t Rf, uint32_t Rs, const uint32_t* orows, uint32_t x, int ls) {
  Program p; p.add_inst(SMC_LI(8, CASR)); p.add_inst(SMC_LI(BANK, BAR)); p.add_inst(SMC_LI(128, NUM_COLS_REG));
  p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_SLEEP(6));
  p.add_below(doubleACT(10, 2, Rf, Rs)); p.add_inst(SMC_SLEEP(6)); p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_SLEEP(6));
  static const int act_pos[5] = {1,4,7,10,13}, zero_pos[5] = {2,5,8,11,14};
  p.add_below(wrRow_immediate_label(BAR, orows[0], ONE, ls));
  for (int i = 0; i < 5; i++) p.add_below(wrRow_immediate_label(BAR, orows[act_pos[i]], x, ls+1+i));
  for (int i = 0; i < 5; i++) p.add_below(wrRow_immediate_label(BAR, orows[zero_pos[i]], 0u, ls+100+i));
  p.add_inst(SMC_SLEEP(6)); p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_SLEEP(6));
  for (int j = 0; j < 3; j++) { p.add_inst(SMC_SLEEP(6)); p.add_inst(SMC_LI(orows[0], RF_REG));
    Mininst q[4] = {SMC_ACT(BAR,0,RF_REG,0), SMC_PRE(BAR,0,0), SMC_NOP(), SMC_NOP()}; p.add_inst(q[0],q[1],q[2],q[3]); p.add_inst(SMC_SLEEP(6)); }
  p.add_inst(SMC_SLEEP(6)); p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_SLEEP(6));
  p.add_below(doubleACT(0, 0, Rf, Rs)); p.add_inst(SMC_SLEEP(6)); p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_SLEEP(6));
  p.add_below(rdRow_immediate_label(BAR, orows[0], ls+999)); p.add_inst(SMC_END());
  return p;
}
struct Calib { int s_id, bank; uint32_t Rf, Rs; vector<uint32_t> rows; };
static Calib read_calib(const string& path, int wb, int ws) {
  ifstream f(path); string line;
  while (getline(f, line)) { if (line.empty() || line[0]=='#') continue; istringstream is(line);
    Calib c; if (!(is>>c.s_id>>c.bank>>c.Rf>>c.Rs)) continue; uint32_t v; while (is>>v) c.rows.push_back(v);
    if (c.rows.size()==16 && c.bank==wb && c.s_id==ws) return c; }
  return Calib{-1,0,0,0,{}};
}

int main(int argc, char** argv) {
  if (argc < 6) { cerr << "Usage: " << argv[0] << " <bender> <calib> <bank> <s_id> <sub_start> [iters]\n"; return 1; }
  int bender = atoi(argv[1]); BANK = atoi(argv[3]); int sid = atoi(argv[4]);
  uint32_t SUB = strtoul(argv[5],0,10); int iters = (argc>6)?atoi(argv[6]):100;
  Calib c = read_calib(argv[2], BANK, sid); if (c.s_id<0) { cerr<<"no calib\n"; return 2; }
  uint32_t backup = SUB + ((c.Rf - SUB) ^ 8u);  // safe offset (bit3, no s72 generator)

  std::mt19937 rng(0xBEEF); uint32_t x = rng();
  vector<uint32_t> W(2048); for (auto& v : W) v = rng();

  SoftMCPlatform pf(bender); if (pf.init()!=SOFTMC_SUCCESS) { cerr<<"init\n"; return 3; } pf.reset_fpga();

  auto verify = [&](uint8_t* buf) { int bad=0; for (int s=0;s<2048;s++){ uint32_t a; memcpy(&a,&buf[s*4],4); if (a != (W[s]&x)) bad++; } return bad; };
  uint8_t buf[8192];

  // ---- Mode A: per-column write W -> Rfirst every iteration ----
  long badA=0; auto tA0=chrono::steady_clock::now();
  for (int i=0;i<iters;i++){
    per_column_write_row(pf, c.Rf, W.data());
    Program p = build_bmaj(c.Rf, c.Rs, c.rows.data(), x, i*1000%60000);
    pf.execute(p); pf.receiveData(buf, 8192); badA += verify(buf);
  }
  auto tA1=chrono::steady_clock::now();
  double msA = chrono::duration<double,milli>(tA1-tA0).count()/iters;

  // ---- Mode B: W persistent in backup; refresh Rfirst by clone each iter ----
  per_column_write_row(pf, backup, W.data());     // one-time (amortized setup)
  long badB=0; auto tB0=chrono::steady_clock::now();
  for (int i=0;i<iters;i++){
    rowclone(pf, backup, c.Rf);                   // cheap refresh
    Program p = build_bmaj(c.Rf, c.Rs, c.rows.data(), x, i*1000%60000);
    pf.execute(p); pf.receiveData(buf, 8192); badB += verify(buf);
  }
  auto tB1=chrono::steady_clock::now();
  double msB = chrono::duration<double,milli>(tB1-tB0).count()/iters;

  fprintf(stderr, "[persist] iters=%d  x=0x%08x  backup=%u (Rf=%u)\n", iters, x, backup, c.Rf);
  fprintf(stderr, "[persist] A (per-col write each MAJ):   %.3f ms/MAJ  mismatch_segs=%ld/%d\n", msA, badA, iters*2048);
  fprintf(stderr, "[persist] B (persistent + clone refresh): %.3f ms/MAJ  mismatch_segs=%ld/%d\n", msB, badB, iters*2048);
  fprintf(stderr, "[persist] speedup: %.2fx   correctness B: %.4f%%\n", msA/msB, 100.0*(1.0 - (double)badB/(iters*2048)));
  fflush(stderr); _exit(0);
}
