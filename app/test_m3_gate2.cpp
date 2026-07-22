// M3 gate 2 (2026-07-22): the four questions the server integration
// depends on, all rig-answerable without touching the server.
//
//   A. SHADOW SUPPLY CENSUS — for every screened pool row, count law-valid
//      single-unit shadow targets (in-window, non-pool, non-tuple). This
//      is the PIM_BCAST_LOAD allocator's input table. Also searches for
//      all-clear k=3 (1->7 rows) and k=4 (1->15 rows) fan-out cosets.
//   B. SOURCE RETENTION — deposits repeat across tokens with aref OFF;
//      write SRC once, deposit+verify at t = 0/10/30/60/120 s.
//   C. DEPOSIT CHAINING — production consumes scratch via a RowClone hop
//      (scratch -> Rfirst). Two-hop chain SRC -(10,2)-> DST -(30,1)->
//      PROBE; PROBE must hold W byte-exact.
//   D. DEPOSIT BURST — N scratch loads in ONE program (N deposits) vs N
//      per-column writes (3N programs): does M3 pay BEFORE streaming?
//
//   ./m3-gate2-exe <bender> <calib_file> <bank> <s_id> <pool_file> [seed]
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
    if (c.open_rows.size() == 16) { if (c.bank == wanted_bank) out.push_back(c); }
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

// Selection-law fired coset of pair (A, A^d): units = per-group
// projections of d; groups {bit0},{1,2},{3,4},{5,6},{7,8},{bit9}.
static vector<uint32_t> coset_of_d(uint32_t A, uint32_t d) {
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
         << " <bender> <calib_file> <bank> <s_id> <pool_file> [seed]\n";
    return 1;
  }
  int bender    = atoi(argv[1]);
  string calibp = argv[2];
  BANK          = atoi(argv[3]);
  int sid       = atoi(argv[4]);
  string poolp  = argv[5];
  unsigned seed = (argc > 6) ? (unsigned)atoi(argv[6]) : 0xC0FFEE;

  vector<Calib> all_cs = read_calib_all(calibp, BANK);
  if (all_cs.empty()) { cerr << "[g2] no calibs\n"; return 2; }
  set<uint32_t> excl;
  for (const auto& c : all_cs) for (uint32_t r : c.open_rows) excl.insert(r);
  vector<uint32_t> pool = read_pool(poolp);
  set<uint32_t> pset(pool.begin(), pool.end());
  const uint32_t win_lo = pool.front(), win_hi = pool.back() + 1;
  printf("# bank=%d s_id=%d pool=%zu window=[%u,%u) tuple_rows=%zu\n",
         BANK, sid, pool.size(), win_lo, win_hi, excl.size());

  auto shadow_ok = [&](uint32_t r) {
    return r >= win_lo && r < win_hi && !pset.count(r) && !excl.count(r);
  };
  auto coset_clear = [&](uint32_t src, uint32_t d) {
    for (uint32_t r : coset_of_d(src, d))
      if (r != src && !shadow_ok(r)) return false;
    return true;
  };

  // ---- Phase A: census ---------------------------------------------------
  // Single-unit offsets (k=1). Bit-9 (512) is excluded BY DEFAULT: its
  // latch behavior is subarray-position-dependent (rigchar 2026-07-22).
  // M3G2_D512=1 adds it — the per-position probe for THIS window (the
  // A2 deposit-verify then delivers the verdict).
  vector<uint32_t> UNIT_OFFS = {1, 2, 4, 6, 8, 16, 24, 32, 64, 96, 128, 256, 384};
  if (getenv("M3G2_D512") && atoi(getenv("M3G2_D512")) != 0)
    UNIT_OFFS.push_back(512);
  printf("\n## Phase A — shadow supply census (k=1 single-unit offsets)\n");
  long total_pairs = 0;
  set<uint32_t> covered_src;
  vector<pair<uint32_t,uint32_t>> k1_pairs;  // (src, dst) for phase D
  for (uint32_t d : UNIT_OFFS) {
    long n = 0;
    for (uint32_t r : pool) {
      if (!shadow_ok(r ^ d)) continue;
      n++;
      covered_src.insert(r);
      if (k1_pairs.size() < 4096) k1_pairs.emplace_back(r, r ^ d);
    }
    printf("  d=%-4u pairs=%ld\n", d, n);
    total_pairs += n;
  }
  printf("  TOTAL k=1 pairs=%ld  sources_covered=%zu/%zu (%.1f%%)\n",
         total_pairs, covered_src.size(), pool.size(),
         100.0 * covered_src.size() / pool.size());
  // k=3 / k=4 all-clear fan-out cosets.
  uint32_t k3_src = 0, k3_d = 0, k4_src = 0, k4_d = 0;
  long k3_n = 0, k4_n = 0;
  // TRUE 3-unit distances only (bit0 + two group projections, or three
  // group projections). 385 = 1+384 is TWO units and was wrongly in this
  // list in v1 — caught by m3_alloc_test; b0's first "k=3" run was k=2.
  for (uint32_t d : {99u, 481u, 27u, 103u, 35u, 121u}) {
    for (uint32_t r : pool) if (coset_clear(r, d)) { k3_n++; if (!k3_src) { k3_src = r; k3_d = d; } }
  }
  for (uint32_t d : {483u, 487u, 127u, 411u}) {          // 4 units each
    for (uint32_t r : pool) if (coset_clear(r, d)) { k4_n++; if (!k4_src) { k4_src = r; k4_d = d; } }
  }
  printf("  k=3 all-clear cosets=%ld%s  k=4 all-clear=%ld%s\n",
         k3_n, k3_src ? "" : " (none)", k4_n, k4_src ? "" : " (none)");

  if (k1_pairs.empty()) { printf("[g2] no k=1 pairs — abort\n"); return 2; }

  std::mt19937 rng(seed);
  SoftMCPlatform pf(bender);
  if (pf.init() != SOFTMC_SUCCESS) { cerr << "[g2] init failed\n"; return 3; }
  pf.reset_fpga();

  vector<uint8_t> exp(8192);
  auto set_exp = [&](const vector<uint32_t>& W) {
    for (int s = 0; s < 2048; s++)
      for (int b = 0; b < 4; b++) exp[s*4+b] = (uint8_t)((W[s] >> (8*b)) & 0xFF);
  };
  auto mism = [&](const uint8_t* got, const uint8_t* want) {
    int n = 0;
    for (int i = 0; i < 8192; i++) if (got[i] != want[i]) n++;
    return n;
  };
  auto write_row = [&](uint32_t row, const vector<uint32_t>& W) {
    int cs0 = 0;
    for (int c = 0; c < 3; c++) {
      Program p = build_chunk_program(BANK, row, W.data() + cs0 * 16, cs0, CHUNK_COLS[c]);
      pf.execute(p);
      cs0 += CHUNK_COLS[c];
    }
  };
  auto zero_rows = [&](const vector<uint32_t>& rows) {
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
  auto deposit_prog = [&](uint32_t src, uint32_t dst, int t12, int t23) {
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
  auto read_rows = [&](const vector<uint32_t>& rows, vector<vector<uint8_t>>& out) {
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
    for (size_t i = 0; i < rows.size(); i++) pf.receiveData(out[i].data(), 8192);
  };

  // ---- Phase A2: one deposit-verify per census offset --------------------
  // The census is geometric; the law carries deposit-firing for every unit
  // offset (1691/1691), but give each offset class one direct content
  // check on shadow rows so the allocator table rests on today's silicon.
  printf("\n## Phase A2 — per-offset deposit verify (1 pair each)\n");
  for (uint32_t d : UNIT_OFFS) {
    uint32_t src = 0;
    for (uint32_t r : pool) if (shadow_ok(r ^ d)) { src = r; break; }
    if (!src) { printf("  d=%-4u (no pair)\n", d); continue; }
    uint32_t dst = src ^ d;
    vector<uint32_t> W(2048);
    for (auto& v : W) v = rng();
    set_exp(W);
    zero_rows({dst});
    write_row(src, W);
    Program dep = deposit_prog(src, dst, 10, 2);
    pf.execute(dep);
    vector<vector<uint8_t>> got;
    read_rows({dst, src}, got);
    printf("  d=%-4u %u->%u dst_mism=%d src_mism=%d\n", d, src, dst,
           mism(got[0].data(), exp.data()), mism(got[1].data(), exp.data()));
  }

  const bool fast = getenv("M3G2_FAST") && atoi(getenv("M3G2_FAST")) != 0;

  // ---- Phase B: source retention (aref off, production condition) --------
  printf("\n## Phase B — source retention across deposit reuse (aref off)\n");
  {
    uint32_t src = k1_pairs[0].first, dst = k1_pairs[0].second;
    vector<uint32_t> W(2048);
    for (auto& v : W) v = rng();
    set_exp(W);
    write_row(src, W);
    auto t_start = chrono::steady_clock::now();
    vector<int> marks = fast ? vector<int>{0, 10} : vector<int>{0, 10, 30, 60, 120};
    for (int mark : marks) {
      for (;;) {
        double el = chrono::duration<double>(chrono::steady_clock::now() - t_start).count();
        if (el >= mark) break;
        usleep(200000);
      }
      zero_rows({dst});
      Program dep = deposit_prog(src, dst, 10, 2);
      pf.execute(dep);
      vector<vector<uint8_t>> got;
      read_rows({dst, src}, got);
      printf("  t=%3ds dst_mism=%d src_mism=%d\n", mark,
             mism(got[0].data(), exp.data()), mism(got[1].data(), exp.data()));
    }
  }

  // ---- Phase C: two-hop deposit chain ------------------------------------
  printf("\n## Phase C — deposit chain SRC -(10,2)-> DST -(30,1)-> PROBE\n");
  {
    // need (src, src^1, src^3): dst = src^1, probe = dst^2 = src^3.
    uint32_t src = 0;
    for (uint32_t r : pool)
      if (shadow_ok(r ^ 1u) && shadow_ok(r ^ 3u)) { src = r; break; }
    if (!src) printf("  no (src, src^1, src^3) triple — skipped\n");
    else {
      uint32_t dst = src ^ 1u, probe = src ^ 3u;
      printf("  chain: %u -> %u -> %u\n", src, dst, probe);
      int bad = 0;
      for (int t = 0; t < 10; t++) {
        vector<uint32_t> W(2048);
        for (auto& v : W) v = rng();
        set_exp(W);
        zero_rows({dst, probe});
        write_row(src, W);
        Program d1 = deposit_prog(src, dst, 10, 2);
        pf.execute(d1);
        Program d2 = deposit_prog(dst, probe, 30, 1);   // RowClone timing hop
        pf.execute(d2);
        vector<vector<uint8_t>> got;
        read_rows({probe, dst, src}, got);
        int mp = mism(got[0].data(), exp.data());
        int md = mism(got[1].data(), exp.data());
        int ms = mism(got[2].data(), exp.data());
        if (mp || md || ms) bad++;
        printf("  t=%02d probe_mism=%d dst_mism=%d src_mism=%d %s\n",
               t, mp, md, ms, (mp||md||ms) ? "<-- BAD" : "OK");
      }
      printf("  chain verdict: %s (%d/10 bad)\n", bad ? "FAIL" : "CLEAN", bad);
    }
  }

  // ---- Phase E: k=3 fan-out (1 op -> 7 targets) ---------------------------
  if (k3_src) {
    auto cs3 = coset_of_d(k3_src, k3_d);
    vector<uint32_t> targets;
    for (uint32_t r : cs3) if (r != k3_src) targets.push_back(r);
    printf("\n## Phase E — fan-out d=%u (%zu units, 1 op -> %zu targets), 10 trials\n",
           k3_d, coset_of_d(0, k3_d).size() == 8 ? (size_t)3 : (size_t)2, targets.size());
    int bad3 = 0;
    for (int t = 0; t < 10; t++) {
      vector<uint32_t> W(2048);
      for (auto& v : W) v = rng();
      set_exp(W);
      zero_rows(targets);
      write_row(k3_src, W);
      Program dep = deposit_prog(k3_src, k3_src ^ k3_d, 10, 2);
      pf.execute(dep);
      int worst = 0;
      for (size_t i = 0; i < targets.size(); i += 8) {
        vector<uint32_t> g(targets.begin() + i, targets.begin() + min(targets.size(), i + 8));
        vector<vector<uint8_t>> got;
        read_rows(g, got);
        for (auto& r : got) worst = max(worst, mism(r.data(), exp.data()));
      }
      vector<vector<uint8_t>> gs;
      read_rows({k3_src}, gs);
      int ms = mism(gs[0].data(), exp.data());
      if (worst || ms) bad3++;
      printf("  t=%02d worst_target_mism=%d src_mism=%d %s\n", t, worst, ms,
             (worst || ms) ? "<-- BAD" : "OK");
    }
    printf("  k=3 verdict: %s (%d/10 bad)\n", bad3 ? "FAIL" : "CLEAN", bad3);
  } else {
    printf("\n## Phase E — k=3: no all-clear coset in this window, skipped\n");
  }

  // ---- Phase D: deposit burst vs per-column writes ------------------------
  if (fast) { printf("\n[g2] DONE (fast mode: phase D skipped)\n"); return 0; }
  printf("\n## Phase D — N scratch loads: ONE burst program vs 3N pcwrite programs\n");
  {
    // Pick N=32 pairs with DISTINCT sources and distinct targets, none of
    // whose fired pair overlaps another pair's rows.
    vector<pair<uint32_t,uint32_t>> burst;
    set<uint32_t> used;
    for (auto& pr : k1_pairs) {
      if (burst.size() >= 32) break;
      if (used.count(pr.first) || used.count(pr.second)) continue;
      burst.push_back(pr);
      used.insert(pr.first);
      used.insert(pr.second);
    }
    printf("  burst size N=%zu (distinct-row pairs)\n", burst.size());
    vector<uint32_t> W(2048);
    for (auto& v : W) v = rng();
    set_exp(W);
    // All sources hold W (residency precondition, written once, untimed).
    for (auto& pr : burst) write_row(pr.first, W);
    // ONE program: PRE + N deposits + PRE.
    Program pb;
    pb.add_inst(SMC_LI(8, CASR));
    pb.add_inst(SMC_LI(BANK, BAR));
    pb.add_below(PRE(BAR, 0, 0));
    pb.add_inst(SMC_SLEEP(6));
    for (auto& pr : burst) {
      pb.add_below(doubleACT(10, 2, pr.first, pr.second));
      pb.add_inst(SMC_SLEEP(6));
      pb.add_below(PRE(BAR, 0, 0));
      pb.add_inst(SMC_SLEEP(4));
    }
    pb.add_inst(SMC_END());
    long insts_b = (long)pb.size() / 8;
    const int iters = 50;
    auto tb0 = chrono::steady_clock::now();
    for (int i = 0; i < iters; i++) pf.execute(pb);
    auto tb1 = chrono::steady_clock::now();
    double wb = chrono::duration<double, milli>(tb1 - tb0).count() / iters;
    // Verify all targets after the burst.
    int worst = 0;
    {
      vector<uint32_t> tg;
      for (auto& pr : burst) tg.push_back(pr.second);
      // read in groups of 8 to keep programs modest
      for (size_t i = 0; i < tg.size(); i += 8) {
        vector<uint32_t> g(tg.begin() + i, tg.begin() + min(tg.size(), i + 8));
        vector<vector<uint8_t>> got;
        read_rows(g, got);
        for (auto& r : got) worst = max(worst, mism(r.data(), exp.data()));
      }
    }
    // Arm A: 3 chunk programs per load, pre-built for one target, executed
    // N times (same shape/cost as N distinct targets).
    Program a0 = build_chunk_program(BANK, burst[0].second, W.data() + 0,     0, 43);
    Program a1 = build_chunk_program(BANK, burst[0].second, W.data() + 43*16, 43, 43);
    Program a2 = build_chunk_program(BANK, burst[0].second, W.data() + 86*16, 86, 42);
    long insts_a = (long)(a0.size() + a1.size() + a2.size()) / 8 * (long)burst.size();
    auto ta0 = chrono::steady_clock::now();
    for (size_t i = 0; i < burst.size(); i++) { pf.execute(a0); pf.execute(a1); pf.execute(a2); }
    auto ta1 = chrono::steady_clock::now();
    double wa = chrono::duration<double, milli>(ta1 - ta0).count();
    printf("  armA: %zu loads = %zu programs, %ld insts, %.2f ms total (%.3f ms/load)\n",
           burst.size(), burst.size() * 3, insts_a, wa, wa / burst.size());
    printf("  armB: %zu loads = 1 program (%ld insts), %.3f ms total (%.4f ms/load)\n",
           burst.size(), insts_b, wb, wb / burst.size());
    printf("  ratio: wall %.1fx, insts %.1fx  worst_target_mism=%d %s\n",
           wa / wb, (double)insts_a / insts_b, worst,
           worst == 0 ? "ALL_EXACT" : "<-- CONTENT BAD");
  }

  printf("\n[g2] DONE\n");
  return 0;
}
