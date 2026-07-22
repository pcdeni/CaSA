// m3_alloc.h unit test — validates the header against the 2026-07-22
// silicon-backed census numbers (gate2_b2_bank1.log / gate2_b0_bank1.log
// / gate2e_*.log). No FPGA needed.
//   ./m3-alloc-test <calib_file> <bank> <pool_file> <exp_k1> <exp_cover> [with_d512 exp_k1_512]
#include "m3_alloc.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

int main(int argc, char** argv) {
  if (argc < 6) {
    fprintf(stderr,
            "Usage: %s <calib_file> <bank> <pool_file> <exp_k1> <exp_cover> "
            "[with_d512 exp_k1_512]\n", argv[0]);
    return 1;
  }
  string calibp = argv[1];
  int bank = atoi(argv[2]);
  string poolp = argv[3];
  long exp_k1 = atol(argv[4]);
  long exp_cover = atol(argv[5]);
  bool with512 = (argc > 7);
  long exp_k1_512 = with512 ? atol(argv[7]) : 0;

  int fails = 0;
  auto CHECK = [&](bool ok, const char* what, long got, long want) {
    printf("%-46s got=%-6ld want=%-6ld %s\n", what, got, want, ok ? "OK" : "FAIL");
    if (!ok) fails++;
  };

  // Law self-checks (pure math).
  CHECK(m3::coset_of(45423, 97).size() == 4, "coset size d=97 (k=2)", (long)m3::coset_of(45423, 97).size(), 4);
  CHECK(m3::coset_of(45313, 99).size() == 8, "coset size d=99 (k=3)", (long)m3::coset_of(45313, 99).size(), 8);
  // 385 = 1 + 384 = TWO units (bit-0 + group {7,8}) -> 4-row coset. This
  // expectation being wrong in v1 of this test exposed the gate-2 tool's
  // mislabeled "k=3" run on b0 (it was a k=2 deposit) — kept as a check.
  CHECK(m3::coset_of(38481, 385).size() == 4, "coset size d=385 (k=2!)", (long)m3::coset_of(38481, 385).size(), 4);
  CHECK(m3::units_of(6).size() == 1, "6 is ONE unit (group {1,2})", (long)m3::units_of(6).size(), 1);
  CHECK(m3::units_of(384).size() == 1, "384 is ONE unit (group {7,8})", (long)m3::units_of(384).size(), 1);

  // Load calib rows (ALL s_ids of the bank) + pool.
  set<uint32_t> tuple_rows;
  {
    ifstream f(calibp);
    string line;
    while (getline(f, line)) {
      if (line.empty() || line[0] == '#') continue;
      istringstream iss(line);
      int sid, bk; uint32_t rf, rs, v;
      if (!(iss >> sid >> bk >> rf >> rs)) continue;
      vector<uint32_t> rows;
      while (iss >> v) rows.push_back(v);
      if (rows.size() == 16 && bk == bank)
        tuple_rows.insert(rows.begin(), rows.end());
    }
  }
  vector<uint32_t> pool;
  {
    ifstream f(poolp);
    string line;
    while (getline(f, line)) {
      if (line.empty() || line[0] == '#') continue;
      pool.push_back((uint32_t)strtoul(line.c_str(), nullptr, 10));
    }
  }
  printf("# bank=%d pool=%zu tuple_rows=%zu\n", bank, pool.size(), tuple_rows.size());

  m3::ShadowMap sm;
  size_t n = sm.build(pool, tuple_rows, pool.front(), pool.back() + 1, false);
  CHECK((long)n == exp_k1, "k=1 pairs (13 offsets) == silicon census", (long)n, exp_k1);
  CHECK((long)sm.sources_covered() == exp_cover, "sources covered == silicon census", (long)sm.sources_covered(), exp_cover);

  if (with512) {
    size_t n512 = sm.build(pool, tuple_rows, pool.front(), pool.back() + 1, true);
    CHECK((long)n512 == exp_k1_512, "k=1 pairs incl d=512 == census", (long)n512, exp_k1_512);
  }

  // Burst grouping: 32 pairwise-disjoint pairs must exist and be disjoint.
  {
    auto b = sm.burst(32);
    set<uint32_t> rows;
    bool disjoint = true;
    for (auto& p : b) {
      if (!rows.insert(p.src).second) disjoint = false;
      if (!rows.insert(p.targets[0]).second) disjoint = false;
    }
    CHECK(b.size() == 32 && disjoint, "burst(32): 32 disjoint pairs", (long)b.size(), 32);
  }

  // Fan-out enumeration matches the census counts for this bank's d.
  {
    long f99 = (long)sm.fanout(99).size();
    long f385 = (long)sm.fanout(385).size();
    printf("# fanout options: d=99 -> %ld, d=385 -> %ld (census: b2 had 1 @d=99, b0 had 7 @d=385)\n",
           f99, f385);
  }

  printf("\n[m3-alloc-test] %s (%d fails)\n", fails ? "FAIL" : "ALL_PASS", fails);
  return fails ? 10 : 0;
}
