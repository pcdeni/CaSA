// Fused per-MAJ throughput A/B (2026-07-17): both validated levers in ONE
// per-MAJ path, verified every iteration (result == W & x).
//
//  A  (production shape): per-column write W -> Rfirst (3 execs) + program
//     [broadcast, 11 uniform wrRows (frac/5x/5zero), frac x3, MAJ3, read].
//  B1 (fused, coset-margins 7W/4x/4z+frac): clone backup->Rfirst + program
//     [broadcast, wr ONE->r0, wr x->r1 + doubleACT(r1,r13) coset {1,5,9,13},
//      wr 0->r2 + doubleACT(r2,r14) coset {2,6,10,14}, frac x3, MAJ3, read].
//  B2 (fused, production-margins 5W/5x/5z+frac): B1 + wr x->r4 + wr 0->r8.
//
// Coset partners use generators {96,384}: local(r1)^local(r13)=480, same for
// r2/r14 (validated selective deposit, t=10,2). W backup at safe offset ^8.
// Argv: <bender> <calib_file> <bank> <s_id> <sub_start> [iters]
#include "instruction.h"
#include "prog.h"
#include "platform.h"
#include "../util.h"
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>
using namespace std;
static const int CHUNK_COLS[3] = {43, 43, 42};
static int BANK = 0;

static Program chunkp(uint32_t row, const uint32_t* cd, int cs, int n) {
  Program p; p.add_inst(SMC_LI(8, CASR)); p.add_inst(SMC_LI(BANK, BAR));
  p.add_inst(SMC_LI(row, RAR)); p.add_inst(SMC_LI(cs * 8, CAR));
  p.add_below(PRE(BAR, 0, 0)); p.add_below(ACT(BAR, 0, RAR, 0));
  for (int k = 0; k < n; k++) { const uint32_t* sl = cd + k * 16;
    for (int s = 0; s < 16; s++) { p.add_inst(SMC_LI(sl[s], PATTERN_REG)); p.add_inst(SMC_LDWD(PATTERN_REG, s)); }
    p.add_below(WRITE(BAR, CAR, 1)); p.add_inst(SMC_SLEEP(8)); }
  p.add_inst(SMC_SLEEP(8)); p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_SLEEP(4)); p.add_inst(SMC_END());
  return p;
}
static void pcw(SoftMCPlatform& pf, uint32_t row, const uint32_t* d) {
  int cs = 0; for (int ch = 0; ch < 3; ch++) { Program p = chunkp(row, d + cs * 16, cs, CHUNK_COLS[ch]); pf.execute(p); cs += CHUNK_COLS[ch]; }
}
static void addclone(Program& p, uint32_t src, uint32_t dst) {
  p.add_inst(SMC_SLEEP(6)); p.add_below(doubleACT(30, 1, src, dst));
  p.add_inst(SMC_SLEEP(6)); p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_SLEEP(6));
}
static void addbcast(Program& p, uint32_t rf, uint32_t rs) {
  p.add_inst(SMC_SLEEP(6)); p.add_below(doubleACT(10, 2, rf, rs));
  p.add_inst(SMC_SLEEP(6)); p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_SLEEP(6));
}
static void addfrac3_maj_read(Program& p, uint32_t frac_row, uint32_t Rf, uint32_t Rs, uint32_t rd, int ls) {
  for (int j = 0; j < 3; j++) { p.add_inst(SMC_SLEEP(6)); p.add_inst(SMC_LI(frac_row, RF_REG));
    Mininst q[4] = {SMC_ACT(BAR, 0, RF_REG, 0), SMC_PRE(BAR, 0, 0), SMC_NOP(), SMC_NOP()};
    p.add_inst(q[0], q[1], q[2], q[3]); p.add_inst(SMC_SLEEP(6)); }
  p.add_inst(SMC_SLEEP(6)); p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_SLEEP(6));
  p.add_below(doubleACT(0, 0, Rf, Rs)); p.add_inst(SMC_SLEEP(6)); p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_SLEEP(6));
  p.add_below(rdRow_immediate_label(BAR, rd, ls)); p.add_inst(SMC_END());
}
struct Calib { int s_id, bank; uint32_t Rf, Rs; vector<uint32_t> r; };
static Calib readc(const string& path, int wb, int ws) {
  ifstream f(path); string line;
  while (getline(f, line)) { if (line.empty() || line[0] == '#') continue; istringstream is(line);
    Calib c; if (!(is >> c.s_id >> c.bank >> c.Rf >> c.Rs)) continue; uint32_t v; while (is >> v) c.r.push_back(v);
    if (c.r.size() == 16 && c.bank == wb && c.s_id == ws) return c; }
  return Calib{-1, 0, 0, 0, {}};
}

int main(int argc, char** argv) {
  if (argc < 6) { cerr << "Usage: <bender> <calib> <bank> <s_id> <sub_start> [iters]\n"; return 1; }
  int bender = atoi(argv[1]); BANK = atoi(argv[3]); int sid = atoi(argv[4]);
  uint32_t SUB = strtoul(argv[5], 0, 10); int iters = (argc > 6) ? atoi(argv[6]) : 100;
  Calib c = readc(argv[2], BANK, sid); if (c.s_id < 0) { cerr << "no calib\n"; return 2; }
  uint32_t backup = SUB + ((c.Rf - SUB) ^ 8u);

  std::mt19937 rng(0xF00D); uint32_t x = rng();
  vector<uint32_t> W(2048); for (auto& v : W) v = rng();
  SoftMCPlatform pf(bender); if (pf.init() != SOFTMC_SUCCESS) { cerr << "init\n"; return 3; }
  pf.reset_fpga();
  uint8_t buf[8192];
  auto verify = [&]() { int bad = 0; for (int s = 0; s < 2048; s++) { uint32_t a; memcpy(&a, &buf[s * 4], 4); if (a != (W[s] & x)) bad++; } return bad; };

  // ---------- Mode B builder (variant: extra=0 -> B1, extra=1 -> B2) ----------
  auto buildB = [&](int extra) {
    Program p;
    p.add_inst(SMC_LI(8, CASR)); p.add_inst(SMC_LI(BANK, BAR)); p.add_inst(SMC_LI(128, NUM_COLS_REG));
    p.add_below(PRE(BAR, 0, 0));
    addclone(p, backup, c.Rf);                       // W refresh (persistent lever)
    addbcast(p, c.Rf, c.Rs);                         // W -> all 16
    p.add_below(wrRow_immediate_label(BAR, c.r[0], ONE, 0));
    p.add_below(wrRow_immediate_label(BAR, c.r[1], x, 1));       // x anchor
    p.add_below(wrRow_immediate_label(BAR, c.r[2], 0u, 2));      // zero anchor
    if (extra) { p.add_below(wrRow_immediate_label(BAR, c.r[4], x, 3));
                 p.add_below(wrRow_immediate_label(BAR, c.r[8], 0u, 4)); }
    p.add_inst(SMC_SLEEP(6)); p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_SLEEP(6));
    addbcast(p, c.r[1], c.r[13]);                    // x coset {1,5,9,13}
    addbcast(p, c.r[2], c.r[14]);                    // 0 coset {2,6,10,14}
    addfrac3_maj_read(p, c.r[0], c.Rf, c.Rs, c.r[0], 950);
    return p; };

  // ---------- Mode A: production shape ----------
  Program progA;
  { Program& p = progA;
    p.add_inst(SMC_LI(8, CASR)); p.add_inst(SMC_LI(BANK, BAR)); p.add_inst(SMC_LI(128, NUM_COLS_REG));
    p.add_below(PRE(BAR, 0, 0));
    addbcast(p, c.Rf, c.Rs);
    static const int act[5] = {1, 4, 7, 10, 13}, zer[5] = {2, 5, 8, 11, 14};
    p.add_below(wrRow_immediate_label(BAR, c.r[0], ONE, 0));
    for (int i = 0; i < 5; i++) p.add_below(wrRow_immediate_label(BAR, c.r[act[i]], x, 1 + i));
    for (int i = 0; i < 5; i++) p.add_below(wrRow_immediate_label(BAR, c.r[zer[i]], 0u, 20 + i));
    p.add_inst(SMC_SLEEP(6)); p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_SLEEP(6));
    addfrac3_maj_read(p, c.r[0], c.Rf, c.Rs, c.r[0], 900); }
  // IMEM gate — the bitstream rejects >2048-inst programs and then receiveData
  // hangs forever (platform_receivedata_no_timeout). Refuse oversize up front.
  long szA = progA.size() / 8, szB1 = buildB(0).size() / 8, szB2 = buildB(1).size() / 8;
  fprintf(stderr, "[fused] program sizes (insts, IMEM=2048): A=%ld B1=%ld B2=%ld\n", szA, szB1, szB2);
  if (szA > 2040 || szB1 > 2040 || szB2 > 2040) {
    fprintf(stderr, "[fused] ABORT: a program exceeds IMEM — will not execute (would hang)\n");
    fflush(stderr); _exit(5);
  }

  fprintf(stderr, "[fused] starting Mode A...\n"); fflush(stderr);
  long badA = 0; auto tA0 = chrono::steady_clock::now();
  for (int i = 0; i < iters; i++) {
    pcw(pf, c.Rf, W.data());
    pf.execute(progA); pf.receiveData(buf, 8192); badA += verify();
    fprintf(stderr, "[fused]   A iter %d ok\n", i); fflush(stderr);
  }
  auto tA1 = chrono::steady_clock::now();
  double msA = chrono::duration<double, milli>(tA1 - tA0).count() / iters;
  fprintf(stderr, "[fused] Mode A done\n"); fflush(stderr);

  auto runB = [&](int extra, double& ms, long& bad) {
    fprintf(stderr, "[fused] starting Mode B extra=%d...\n", extra); fflush(stderr);
    Program p = buildB(extra);
    pcw(pf, backup, W.data());                       // one-time resident W
    bad = 0; auto t0 = chrono::steady_clock::now();
    for (int i = 0; i < iters; i++) { pf.execute(p); pf.receiveData(buf, 8192); bad += verify();
      fprintf(stderr, "[fused]   B%d iter %d ok\n", extra, i); fflush(stderr); }
    auto t1 = chrono::steady_clock::now();
    ms = chrono::duration<double, milli>(t1 - t0).count() / iters; };

  double msB1, msB2; long badB1, badB2;
  runB(0, msB1, badB1);
  runB(1, msB2, badB2);

  long iA = progA.size() / 8, iB1 = buildB(0).size() / 8, iB2 = buildB(1).size() / 8;
  fprintf(stderr, "[fused] iters=%d bender=%d  (A adds 3 per-col-write execs/iter outside its program)\n", iters, bender);
  fprintf(stderr, "[fused] A  prod 11-wrRow + W-rewrite : %7.3f ms/MAJ  bad=%ld/%d  prog=%ld insts\n", msA, badA, iters * 2048, iA);
  fprintf(stderr, "[fused] B1 fused 7W/4x/4z coset      : %7.3f ms/MAJ  bad=%ld/%d  prog=%ld insts\n", msB1, badB1, iters * 2048, iB1);
  fprintf(stderr, "[fused] B2 fused 5W/5x/5z coset+2wr  : %7.3f ms/MAJ  bad=%ld/%d  prog=%ld insts\n", msB2, badB2, iters * 2048, iB2);
  fprintf(stderr, "[fused] speedup A->B1: %.2fx   A->B2: %.2fx\n", msA / msB1, msA / msB2);
  fflush(stderr); _exit(0);
}
