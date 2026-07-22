// M3 first gate (2026-07-22, docs/M3_COSET_FANOUT_DESIGN.md): scratch-load
// A/B on PRODUCTION pool rows — one coset doubleACT deposit vs the 3-chunk
// per-column write it would replace in the V2 wcol path.
//
// Unlike test_sublattice_bcast (deposits between CALIBRATED tuple members),
// this fires law-built pairs among SCREENED BACKUP-POOL rows — the actual
// M3 shape: weight resident in one pool row (SRC), one doubleACT deposits
// it into the scratch target(s) DST = SRC (+) S for S in span(d), d = the
// chosen pair distance. The selection law (zero exceptions, both dies)
// predicts the fired set; this tool checks BYTE-exact deposit content,
// source retention, zero leak on law-adjacent + far probes, and the
// per-op instruction/wall cost of both arms.
//
//   ./m3-scratch-ab <bender> <calib_file> <bank> <s_id> <pool_file> [seed] [iters]
//
// Exit 0 iff every content check is clean (deposits byte-exact, no leak).
#include "instruction.h"
#include "prog.h"
#include "platform.h"
#include "../util.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

using namespace std;

static const int CHUNK_COLS[3] = {43, 43, 42};
static int BANK = 0;

// Proven per-column write builder (test_sublattice_bcast / production
// per_column_write_row shape): 16 LDWD slots per column, WRITE, then
// SMC_SLEEP(8) for tWR.
static Program build_chunk_program(int bank_id, uint32_t row_addr,
                                   const uint32_t* col_data,
                                   int col_start, int n_cols) {
  Program p;
  p.add_inst(SMC_LI(8, CASR));
  p.add_inst(SMC_LI(bank_id, BAR));
  p.add_inst(SMC_LI(row_addr, RAR));
  p.add_inst(SMC_LI(col_start * 8, CAR));
  p.add_below(PRE(BAR, 0, 0));
  p.add_below(ACT(BAR, 0, RAR, 0));
  for (int k = 0; k < n_cols; k++) {
    const uint32_t* slots = col_data + k * 16;
    for (int slot = 0; slot < 16; slot++) {
      p.add_inst(SMC_LI(slots[slot], PATTERN_REG));
      p.add_inst(SMC_LDWD(PATTERN_REG, slot));
    }
    p.add_below(WRITE(BAR, CAR, 1));
    p.add_inst(SMC_SLEEP(8));
  }
  p.add_inst(SMC_SLEEP(8));
  p.add_below(PRE(BAR, 0, 0));
  p.add_inst(SMC_SLEEP(4));
  p.add_inst(SMC_END());
  return p;
}

struct Calib {
  int s_id, bank;
  uint32_t Rfirst, Rsecond;
  vector<uint32_t> open_rows;
};

static vector<Calib> read_calib_all(const string& path, int wanted_bank) {
  vector<Calib> out;
  ifstream f(path);
  string line;
  while (getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    istringstream iss(line);
    Calib c;
    if (!(iss >> c.s_id >> c.bank >> c.Rfirst >> c.Rsecond)) continue;
    uint32_t v;
    while (iss >> v) c.open_rows.push_back(v);
    if (c.open_rows.size() != 16) continue;
    if (c.bank == wanted_bank) out.push_back(c);
  }
  return out;
}

static vector<uint32_t> read_pool(const string& path) {
  vector<uint32_t> out;
  ifstream f(path);
  string line;
  while (getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    out.push_back((uint32_t)strtoul(line.c_str(), nullptr, 10));
  }
  return out;
}

// The fired coset of doubleACT(A, B): {A ^ S : S subseteq units(d)},
// d = A^B, where units are the per-predecoder-group projections of d
// (groups over row-address bits: {1,2},{3,4},{5,6},{7,8}; bits 0 and 9
// are singletons). Complete selection law, 1691/1691 both dies.
static vector<uint32_t> coset_of_pair(uint32_t A, uint32_t B) {
  uint32_t d = A ^ B;
  static const uint32_t GROUPS[6] = {0x001, 0x006, 0x018, 0x060, 0x180, 0x200};
  vector<uint32_t> units;
  for (uint32_t g : GROUPS) if (d & g) units.push_back(d & g);
  vector<uint32_t> out;
  for (uint32_t m = 0; m < (1u << units.size()); m++) {
    uint32_t s = 0;
    for (size_t i = 0; i < units.size(); i++) if (m & (1u << i)) s ^= units[i];
    out.push_back(A ^ s);
  }
  sort(out.begin(), out.end());
  return out;
}

int main(int argc, char** argv) {
  if (argc < 6) {
    cerr << "Usage: " << argv[0]
         << " <bender> <calib_file> <bank> <s_id> <pool_file> [seed] [iters]\n";
    return 1;
  }
  int bender    = atoi(argv[1]);
  string calibp = argv[2];
  BANK          = atoi(argv[3]);
  int sid       = atoi(argv[4]);
  string poolp  = argv[5];
  unsigned seed = (argc > 6) ? (unsigned)atoi(argv[6]) : 0xC0FFEE;
  int iters     = (argc > 7) ? atoi(argv[7]) : 100;

  vector<Calib> all_cs = read_calib_all(calibp, BANK);
  if (all_cs.empty()) { cerr << "[m3] no calibs for bank " << BANK << "\n"; return 2; }
  // Exclusion set: every calibrated row of this bank (any s_id) — a fired
  // coset must never touch a tuple.
  set<uint32_t> excl;
  const Calib* prim = nullptr;
  for (const auto& c : all_cs) {
    for (uint32_t r : c.open_rows) excl.insert(r);
    if (!prim && c.s_id == sid) prim = &c;
  }
  if (!prim) { cerr << "[m3] no s_id=" << sid << " calib\n"; return 2; }

  vector<uint32_t> pool = read_pool(poolp);
  if (pool.size() < 8) { cerr << "[m3] pool too small\n"; return 2; }
  set<uint32_t> pset(pool.begin(), pool.end());
  printf("# bank=%d s_id=%d pool=%zu rows [%u..%u] excl(tuple rows)=%zu\n",
         BANK, sid, pool.size(), pool.front(), pool.back(), excl.size());

  // Pair selection: SRC is a screened pool row (the resident weight);
  // every OTHER fired-coset member is a scratch target in the pool's
  // coupled SHADOW — inside the safe window, outside every tuple, and NOT
  // itself a pool row. The legacy pool is an independent set over the
  // coupling graph (no two pool rows at spread offsets — that protected
  // per-column writes), so deposit targets are by construction the rows
  // that layout excluded; M3 turns the coupling into the load channel.
  const uint32_t win_lo = pool.front(), win_hi = pool.back() + 1;
  auto coset_ok = [&](uint32_t src, const vector<uint32_t>& cs) {
    if (!pset.count(src)) return false;
    for (uint32_t r : cs) {
      if (r == src) continue;
      if (r < win_lo || r >= win_hi) return false;
      if (excl.count(r) || pset.count(r)) return false;
    }
    return true;
  };
  // P1: minimal deposit, d = 1 (bit-0 singleton) -> coset {SRC, SRC^1}.
  uint32_t p1_src = 0, p1_dst = 0;
  for (uint32_t r : pool) {
    if (coset_ok(r, coset_of_pair(r, r ^ 1u))) { p1_src = r; p1_dst = r ^ 1u; break; }
  }
  // P2: fan-out k=2. Try d in preference order; coset = 4 rows, 3 targets.
  uint32_t p2_src = 0, p2_d = 0;
  for (uint32_t d : {97u, 385u, 480u, 25u, 7u}) {
    for (uint32_t r : pool) {
      if (coset_ok(r, coset_of_pair(r, r ^ d))) { p2_src = r; p2_d = d; break; }
    }
    if (p2_src) break;
  }
  if (!p1_src) { printf("[m3] NO k=1 (pool row, shadow row) pair in window\n"); return 2; }
  printf("# P1 (k=1): SRC=%u DST=%u (d=1)\n", p1_src, p1_dst);
  if (p2_src) {
    auto cs2 = coset_of_pair(p2_src, p2_src ^ p2_d);
    printf("# P2 (k=2): SRC=%u d=%u coset:", p2_src, p2_d);
    for (uint32_t r : cs2) printf(" %u", r);
    printf("\n");
  } else {
    printf("# P2: no all-screened k=2 coset found — fan-out arm skipped\n");
  }

  // Leak probes for P1: law-adjacent rows that must NOT fire (other units
  // relative to SRC) + far pool rows. All must exist in the window; probes
  // need not be pool rows (they are only read, plus zero-initialized).
  vector<uint32_t> probes;
  for (uint32_t off : {2u, 96u, 384u, 512u}) probes.push_back(p1_src ^ off);
  probes.push_back(pool[pool.size() / 2]);
  probes.push_back(pool.back());
  // Drop any probe that collides with the pair or tuples.
  probes.erase(remove_if(probes.begin(), probes.end(), [&](uint32_t r) {
    return r == p1_src || r == p1_dst || excl.count(r);
  }), probes.end());

  std::mt19937 rng(seed);
  SoftMCPlatform pf(bender);
  if (pf.init() != SOFTMC_SUCCESS) { cerr << "[m3] init failed\n"; return 3; }
  pf.reset_fpga();

  vector<uint8_t> rb(8192), exp(8192);
  auto mism = [&](const uint8_t* got, const uint8_t* want) {
    int n = 0;
    for (int i = 0; i < 8192; i++) if (got[i] != want[i]) n++;
    return n;
  };
  auto write_row = [&](uint32_t row, const vector<uint32_t>& W) {
    int col_start = 0;
    for (int chunk = 0; chunk < 3; chunk++) {
      Program p = build_chunk_program(BANK, row, W.data() + col_start * 16,
                                      col_start, CHUNK_COLS[chunk]);
      pf.execute(p);
      col_start += CHUNK_COLS[chunk];
    }
  };
  auto zero_rows = [&](const vector<uint32_t>& rows) {
    // <=8 wrRows per program (wrRow ~80 insts; stay far from the IMEM gate).
    size_t i = 0;
    int lbl = 0;
    while (i < rows.size()) {
      Program p;
      p.add_inst(SMC_LI(8, CASR));
      p.add_inst(SMC_LI(BANK, BAR));
      p.add_inst(SMC_LI(128, NUM_COLS_REG));
      p.add_below(PRE(BAR, 0, 0));
      for (size_t k = 0; k < 8 && i < rows.size(); k++, i++)
        p.add_below(wrRow_immediate_label(BAR, rows[i], 0u, lbl++));
      p.add_inst(SMC_END());
      pf.execute(p);
    }
  };
  auto make_deposit_prog = [&](uint32_t src, uint32_t dst, int t12, int t23) {
    Program p;
    p.add_inst(SMC_LI(8, CASR));
    p.add_inst(SMC_LI(BANK, BAR));
    p.add_below(PRE(BAR, 0, 0));
    p.add_inst(SMC_SLEEP(6));
    p.add_below(doubleACT(t12, t23, src, dst));
    p.add_inst(SMC_SLEEP(6));
    p.add_below(PRE(BAR, 0, 0));
    p.add_inst(SMC_SLEEP(4));
    p.add_inst(SMC_END());
    return p;
  };
  auto read_rows = [&](const vector<uint32_t>& rows,
                       vector<vector<uint8_t>>& out) {
    Program p;
    p.add_inst(SMC_LI(8, CASR));
    p.add_inst(SMC_LI(BANK, BAR));
    p.add_inst(SMC_LI(128, NUM_COLS_REG));
    p.add_below(PRE(BAR, 0, 0));
    for (size_t i = 0; i < rows.size(); i++)
      p.add_below(rdRow_immediate_label(BAR, rows[i], (int)i));
    p.add_inst(SMC_END());
    pf.execute(p);
    out.assign(rows.size(), vector<uint8_t>(8192));
    for (size_t i = 0; i < rows.size(); i++)
      pf.receiveData(out[i].data(), 8192);
  };

  // ---- Phase 1: P1 content trials --------------------------------------
  printf("\n## Phase 1 — k=1 deposit content, %d trials, t=(10,2)\n", 20);
  int bad_trials = 0;
  for (int t = 0; t < 20; t++) {
    vector<uint32_t> W(2048);
    for (auto& v : W) v = rng();
    for (int s = 0; s < 2048; s++)
      for (int b = 0; b < 4; b++) exp[s*4+b] = (uint8_t)((W[s] >> (8*b)) & 0xFF);
    vector<uint32_t> zrows = probes;
    zrows.push_back(p1_dst);
    zero_rows(zrows);
    write_row(p1_src, W);
    Program dep = make_deposit_prog(p1_src, p1_dst, 10, 2);
    pf.execute(dep);
    vector<uint32_t> rrows = {p1_dst, p1_src};
    rrows.insert(rrows.end(), probes.begin(), probes.end());
    vector<vector<uint8_t>> got;
    read_rows(rrows, got);
    int m_dst = mism(got[0].data(), exp.data());
    int m_src = mism(got[1].data(), exp.data());
    vector<uint8_t> zeros(8192, 0);
    int leaks = 0;
    for (size_t j = 2; j < rrows.size(); j++)
      if (mism(got[j].data(), zeros.data()) > 64) leaks++;
    bool ok = (m_dst == 0 && m_src == 0 && leaks == 0);
    if (!ok) bad_trials++;
    printf("t=%02d dst_mism=%d src_mism=%d probe_leaks=%d/%zu %s\n",
           t, m_dst, m_src, leaks, rrows.size() - 2, ok ? "OK" : "<-- BAD");
  }

  // ---- Phase 2: P2 fan-out trials --------------------------------------
  int bad_fan = 0;
  if (p2_src) {
    printf("\n## Phase 2 — k=2 fan-out (1 op -> 3 targets), %d trials, t=(10,2)\n", 20);
    auto cs2 = coset_of_pair(p2_src, p2_src ^ p2_d);
    vector<uint32_t> targets;
    for (uint32_t r : cs2) if (r != p2_src) targets.push_back(r);
    for (int t = 0; t < 20; t++) {
      vector<uint32_t> W(2048);
      for (auto& v : W) v = rng();
      for (int s = 0; s < 2048; s++)
        for (int b = 0; b < 4; b++) exp[s*4+b] = (uint8_t)((W[s] >> (8*b)) & 0xFF);
      zero_rows(targets);
      write_row(p2_src, W);
      Program dep = make_deposit_prog(p2_src, p2_src ^ p2_d, 10, 2);
      pf.execute(dep);
      vector<uint32_t> rrows = targets;
      rrows.push_back(p2_src);
      vector<vector<uint8_t>> got;
      read_rows(rrows, got);
      int worst = 0;
      for (size_t j = 0; j < targets.size(); j++)
        worst = max(worst, mism(got[j].data(), exp.data()));
      int m_src = mism(got[targets.size()].data(), exp.data());
      bool ok = (worst == 0 && m_src == 0);
      if (!ok) bad_fan++;
      printf("t=%02d worst_target_mism=%d src_mism=%d %s\n",
             t, worst, m_src, ok ? "OK" : "<-- BAD");
    }
  }

  // ---- Phase 3: timing confirm on P1 -----------------------------------
  printf("\n## Phase 3 — timing confirm (content-exact?)\n");
  for (int t12 : {10, 30}) for (int t23 : {1, 2}) {
    vector<uint32_t> W(2048);
    for (auto& v : W) v = rng();
    for (int s = 0; s < 2048; s++)
      for (int b = 0; b < 4; b++) exp[s*4+b] = (uint8_t)((W[s] >> (8*b)) & 0xFF);
    zero_rows({p1_dst});
    write_row(p1_src, W);
    Program dep = make_deposit_prog(p1_src, p1_dst, t12, t23);
    pf.execute(dep);
    vector<vector<uint8_t>> got;
    read_rows({p1_dst}, got);
    printf("t=(%d,%d) dst_mism=%d\n", t12, t23, mism(got[0].data(), exp.data()));
  }

  // ---- Phase 4: wall + instruction A/B ----------------------------------
  printf("\n## Phase 4 — cost A/B, iters=%d\n", iters);
  {
    vector<uint32_t> W(2048);
    for (auto& v : W) v = rng();
    // Arm A: the production wcol shape — 3 chunk programs, pre-built once
    // (finalize-once fix makes re-execution safe; fused-B2 iters pattern).
    Program a0 = build_chunk_program(BANK, p1_dst, W.data() + 0,   0, 43);
    Program a1 = build_chunk_program(BANK, p1_dst, W.data() + 43*16, 43, 43);
    Program a2 = build_chunk_program(BANK, p1_dst, W.data() + 86*16, 86, 42);
    long ia = (long)(a0.size() + a1.size() + a2.size()) / 8;
    auto ta0 = chrono::steady_clock::now();
    for (int i = 0; i < iters; i++) { pf.execute(a0); pf.execute(a1); pf.execute(a2); }
    auto ta1 = chrono::steady_clock::now();
    double wa = chrono::duration<double, milli>(ta1 - ta0).count() / iters;
    // Arm B: one deposit program (weight already resident in SRC).
    write_row(p1_src, W);
    Program dep = make_deposit_prog(p1_src, p1_dst, 10, 2);
    long ib = (long)dep.size() / 8;
    auto tb0 = chrono::steady_clock::now();
    for (int i = 0; i < iters; i++) pf.execute(dep);
    auto tb1 = chrono::steady_clock::now();
    double wb = chrono::duration<double, milli>(tb1 - tb0).count() / iters;
    // Content still exact after the hammering?
    vector<vector<uint8_t>> got;
    for (int s = 0; s < 2048; s++)
      for (int b = 0; b < 4; b++) exp[s*4+b] = (uint8_t)((W[s] >> (8*b)) & 0xFF);
    read_rows({p1_dst, p1_src}, got);
    printf("armA(wcol 3-chunk): %ld insts, 3 programs, %.3f ms/load\n", ia, wa);
    printf("armB(coset deposit): %ld insts, 1 program,  %.3f ms/load\n", ib, wb);
    printf("ratio: insts %.1fx, wall %.2fx  (post-hammer dst_mism=%d src_mism=%d)\n",
           (double)ia / ib, wa / wb,
           mism(got[0].data(), exp.data()), mism(got[1].data(), exp.data()));
  }

  printf("\n[m3] P1 bad=%d/20  P2 bad=%d/%s  -> %s\n",
         bad_trials, bad_fan, p2_src ? "20" : "skipped",
         (bad_trials == 0 && bad_fan == 0) ? "ALL_CLEAN" : "FAILURES");
  return (bad_trials == 0 && bad_fan == 0) ? 0 : 10;
}
