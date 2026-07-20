// lattice_alloc.h — spread-aware placement helpers (Task M1 core, 2026-07-17).
// Everything here is derived from the validated selection law:
//   - a calibrated 16-row tuple is the coset {Rf ⊕ S : S ∈ span(units)} of
//     the APA pair distance d = Rf⊕Rs partitioned by the predecoder groups;
//   - the deposit set of doubleACT(src,dst) is the coset of src⊕dst, so an
//     offset D is a SAFE clone offset iff no coset member other than the
//     intended dst lands on live rows — for single-bit D that reduces to
//     (1<<b) ∉ span(tuple units);
//   - scope is the 1024-row predecoder BLOCK (addendum 11): all placements
//     must stay within one block, and 640-row "subarray" boundaries do NOT
//     stop deposits.
// Units are recovered from the observed 16 rows by GF(2) Gaussian
// elimination on {row ⊕ anchor}; the fit must reproduce the tuple exactly
// (16/16) or the tuple is rejected — no screening pass, but no guessing
// either.
#pragma once
#include <cstdint>
#include <set>
#include <vector>

struct TupleAlloc {
  std::vector<uint32_t> open;   // the 16 calibrated rows (as listed)
  uint32_t Rf = 0, Rs = 0;
  std::vector<uint32_t> basis;  // GF(2) basis of {row ⊕ anchor}
  std::set<uint32_t> span;      // all 2^k combos of basis
  uint32_t span_bits = 0;       // OR of all span elements
  uint32_t block = 0;           // row >> 10, common to all rows
  bool ok = false;

  // fit basis from rows; anchor = Rf (must be a member)
  bool fit() {
    if (open.size() != 16) return false;
    bool rf_in = false;
    for (uint32_t r : open) if (r == Rf) rf_in = true;
    if (!rf_in) return false;
    block = open[0] >> 10;
    for (uint32_t r : open) if ((r >> 10) != block) return false;
    std::vector<uint32_t> diffs;
    for (uint32_t r : open) if (r != Rf) diffs.push_back(r ^ Rf);
    // GF(2) elimination
    basis.clear();
    for (uint32_t v : diffs) {
      uint32_t x = v;
      for (uint32_t b : basis) { uint32_t hb = 1u << (31 - __builtin_clz(b));
        if (x & hb) x ^= b; }
      if (x) basis.push_back(x);
    }
    if (basis.size() != 4) return false;
    span.clear(); span_bits = 0;
    for (int m = 0; m < 16; m++) {
      uint32_t s = 0;
      for (int j = 0; j < 4; j++) if ((m >> j) & 1) s ^= basis[j];
      span.insert(s); span_bits |= s;
    }
    // exact reproduction check
    std::set<uint32_t> pred, got(open.begin(), open.end());
    for (uint32_t s : span) pred.insert(Rf ^ s);
    ok = (pred == got);
    return ok;
  }
  // conservative safe single-bit clone offsets: bit not touched by any
  // generator AND (1<<b) not in span; stays in-block automatically (b<10)
  std::vector<uint32_t> safe_bits() const {
    std::vector<uint32_t> out;
    for (int b = 0; b < 10; b++) {
      uint32_t D = 1u << b;
      if (span_bits & D) continue;
      if (span.count(D)) continue;
      out.push_back(D);
    }
    return out;
  }
  // region T ⊕ D
  std::vector<uint32_t> shifted(uint32_t D) const {
    std::vector<uint32_t> out;
    for (uint32_t r : open) out.push_back(r ^ D);
    return out;
  }
  // no row of (T ⊕ Da) may collide with (T ⊕ Db) — true iff Da^Db ∉ span
  bool regions_disjoint(uint32_t Da, uint32_t Db) const {
    return !span.count(Da ^ Db);
  }
};
