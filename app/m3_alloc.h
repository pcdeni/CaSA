// m3_alloc.h — M3 shadow-pair enumeration (lever #6), the gate-2 census
// machinery as a reusable header for the future PIM_BCAST_LOAD server
// integration. Header-only, no server dependencies.
//
// Everything here is the validated selection law + the 2026-07-22 gate
// results (m3_scratch_ab_2026_07_22/RESULT.md):
//   - fired coset of doubleACT(A, A^d) = {A ^ S : S built from the
//     per-predecoder-group projections of d} (zero-exception law);
//   - deposit SOURCES are screened pool rows; deposit TARGETS live in the
//     pool's coupled SHADOW (in-window, non-pool, non-tuple) — the legacy
//     pool is an independent set over the coupling graph, so targets are
//     by construction the rows that layout excluded;
//   - every single-unit offset class deposits byte-exact on both dies;
//     bit-9 (512) is window-dependent (usable only where the per-window
//     probe passed — b2/45312 yes, gate2e_b2.log);
//   - fan-out validated through k=3 (one op -> 7 targets).
#pragma once
#include <algorithm>
#include <cstdint>
#include <set>
#include <vector>

namespace m3 {

// Predecoder pair-groups over row-address bits: {0} {1,2} {3,4} {5,6}
// {7,8} {9} (selection law, 1691/1691 both dies, block-scoped).
static const uint32_t GROUPS[6] = {0x001, 0x006, 0x018, 0x060, 0x180, 0x200};

// Units of a pair distance d: the nonzero per-group projections
// (all-or-nothing per group — S∩g ∈ {∅, d∩g}).
inline std::vector<uint32_t> units_of(uint32_t d) {
  std::vector<uint32_t> u;
  for (uint32_t g : GROUPS)
    if (d & g) u.push_back(d & g);
  return u;
}

// Fired coset of doubleACT(A, A^d), sorted. Size = 2^(#units).
inline std::vector<uint32_t> coset_of(uint32_t A, uint32_t d) {
  std::vector<uint32_t> u = units_of(d), out;
  for (uint32_t m = 0; m < (1u << u.size()); m++) {
    uint32_t s = 0;
    for (size_t i = 0; i < u.size(); i++)
      if (m & (1u << i)) s ^= u[i];
    out.push_back(A ^ s);
  }
  std::sort(out.begin(), out.end());
  return out;
}

// The k=1 offsets validated byte-exact on silicon (gate-2 phase A2,
// both dies). with_d512 adds bit-9 — ONLY set it for a window whose
// per-window probe passed (M3G2_D512 run of m3-gate2-exe).
inline const std::vector<uint32_t>& unit_offsets(bool with_d512) {
  static const std::vector<uint32_t> base =
      {1, 2, 4, 6, 8, 16, 24, 32, 64, 96, 128, 256, 384};
  static const std::vector<uint32_t> with512 = [] {
    auto v = base;
    v.push_back(512);
    return v;
  }();
  return with_d512 ? with512 : base;
}

struct ShadowPair {
  uint32_t src = 0;                // pool row holding the resident mask
  uint32_t d = 0;                  // pair distance (fired with src^d)
  std::vector<uint32_t> targets;   // fired coset minus src (1, 3 or 7 rows)
};

struct ShadowMap {
  uint32_t win_lo = 0, win_hi = 0;
  std::set<uint32_t> pool, excl;
  std::vector<ShadowPair> k1;      // all validated-offset k=1 options

  // A row usable as a deposit TARGET: inside the screened window,
  // not a pool row (pool rows may hold residents), not a tuple row.
  bool shadow_ok(uint32_t r) const {
    return r >= win_lo && r < win_hi && !pool.count(r) && !excl.count(r);
  }

  // Enumerate all k=1 (src, d) options. Returns the pair count.
  // tuple_rows must contain EVERY calibrated row of the bank (all s_ids)
  // — a fired coset must never touch a tuple.
  size_t build(const std::vector<uint32_t>& pool_rows,
               const std::set<uint32_t>& tuple_rows,
               uint32_t lo, uint32_t hi, bool with_d512) {
    win_lo = lo; win_hi = hi;
    pool.clear(); pool.insert(pool_rows.begin(), pool_rows.end());
    excl = tuple_rows;
    k1.clear();
    for (uint32_t d : unit_offsets(with_d512))
      for (uint32_t r : pool_rows)
        if (shadow_ok(r ^ d)) k1.push_back({r, d, {r ^ d}});
    return k1.size();
  }

  // Number of distinct pool rows with at least one shadow option.
  size_t sources_covered() const {
    std::set<uint32_t> s;
    for (const auto& p : k1) s.insert(p.src);
    return s.size();
  }

  // Fan-out options for a multi-unit d: sources whose ENTIRE fired coset
  // (minus src) is shadow-clear. Validated on silicon through k=3.
  std::vector<ShadowPair> fanout(uint32_t d) const {
    std::vector<ShadowPair> out;
    for (uint32_t r : pool) {
      ShadowPair p{r, d, {}};
      bool ok = true;
      for (uint32_t c : coset_of(r, d)) {
        if (c == r) continue;
        if (!shadow_ok(c)) { ok = false; break; }
        p.targets.push_back(c);
      }
      if (ok && !p.targets.empty()) out.push_back(std::move(p));
    }
    return out;
  }

  // Up to n pairwise-row-disjoint k=1 pairs (the burst grouping of
  // gate-2 phase D: all deposits packable into one program, no shared
  // rows between pairs).
  std::vector<ShadowPair> burst(size_t n) const {
    std::vector<ShadowPair> out;
    std::set<uint32_t> used;
    for (const auto& p : k1) {
      if (out.size() >= n) break;
      if (used.count(p.src) || used.count(p.targets[0])) continue;
      out.push_back(p);
      used.insert(p.src);
      used.insert(p.targets[0]);
    }
    return out;
  }
};

}  // namespace m3
