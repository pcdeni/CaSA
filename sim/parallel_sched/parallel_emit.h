// Host-side bank-parallel SoftMC program emitters.
//
// Each helper takes 1..4 banks worth of parameters and emits a single
// program where the 4 pack4 slots are used for 4 different banks
// concurrently, instead of the SiMRA convention of slot-0-only-then-
// chain-banks. Per-bank command timing is preserved BIT-EXACT to the
// SiMRA single-bank template — verified in Verilator (see sim_compare.cpp
// in this directory) by filtering each bank's events out of the parallel
// program's command stream and comparing to the sequential template's
// per-bank stream.
//
// Why this matters: the SiMRA per-DIMM calibration measures specific
// `t_12` / `t_23` dwells (RowClone 30/1, broadcast 10/2, MAJ3 0/0 on
// DIMM 0). Lockstep schemes that round dwells up to fabric-cycle
// boundaries DO NOT preserve the calibrated dwell — even +1 PHY position
// of dwell on MAJ3 (target t_12=0) changes the in-DRAM compute outcome.
// The staggered scheme below offsets bank N's pattern start by N*shift
// PHY positions, where shift is chosen so no two banks' events collide
// AND each bank's per-event spacing is exactly t_12+1 / t_12+t_23+2
// PHY positions — same as SiMRA's q_inst[] template.

#pragma once
#include "instruction.h"
#include "prog.h"
#include <vector>

// Compute the smallest stagger S such that, with bank N's pattern at PHY
// positions {N*S, N*S+t_12+1, N*S+t_12+t_23+2}, no two banks' event sets
// overlap. Bounded brute-force search (S = 1..pat_len). Returns pat_len
// if no shorter S works (= identical to back-to-back chaining).
int parallel_min_shift(int t_12, int t_23, int n_banks);

// 4-bank parallel doubleACT — emits the same per-bank command pattern as
// SiMRA's `doubleACT()` helper, but for `n_banks` banks concurrently.
// Caller must have pre-loaded:
//   - bar_reg[k]      = register holding bank ID for slot k
//   - rfirst_reg[k]   = register holding R_first row for bank k (for ACT)
//   - rsecond_reg[k]  = register holding R_second row for bank k (for ACT)
// `n_banks` ∈ {1..4}; 1 → identical to SiMRA single-bank doubleACT
// emission for backward-compat checking.
Program parallel_doubleACT(int t_12, int t_23,
                           const std::vector<int>& bar_reg,
                           const std::vector<int>& rfirst_reg,
                           const std::vector<int>& rsecond_reg);

// 4-bank parallel PRE — packs 4 PRE commands across the 4 slots of one
// pack4. Each bank's PRE goes to its corresponding slot.
Program parallel_PRE(const std::vector<int>& bar_reg);

// 4-bank parallel ACT — analogous to PRE.
Program parallel_ACT(const std::vector<int>& bar_reg,
                     const std::vector<int>& rar_reg);
