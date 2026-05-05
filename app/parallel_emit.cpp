#include "parallel_emit.h"
#include <set>

int parallel_min_shift(int t_12, int t_23, int n_banks) {
  if (n_banks <= 1) return 0;
  // Brute-force search S = 1, 2, ..., t_12+t_23+3. The pattern length
  // ALWAYS works since back-to-back chaining is collision-free by
  // construction; we just want the smallest value.
  int pat_len = t_12 + t_23 + 3;
  for (int S = 1; S <= pat_len; S++) {
    std::set<int> used;
    bool ok = true;
    for (int b = 0; b < n_banks; b++) {
      int e0 = b * S;
      int e1 = b * S + t_12 + 1;
      int e2 = b * S + t_12 + t_23 + 2;
      if (used.count(e0) || used.count(e1) || used.count(e2)) { ok = false; break; }
      used.insert(e0); used.insert(e1); used.insert(e2);
    }
    if (ok) return S;
  }
  return pat_len;
}

Program parallel_doubleACT(int t_12, int t_23,
                           const std::vector<int>& bar_reg,
                           const std::vector<int>& rfirst_reg,
                           const std::vector<int>& rsecond_reg) {
  Program p;
  int n_banks = (int)bar_reg.size();

  // Compute the schedule: bank b's events at PHY positions
  // {b*S, b*S + t_12 + 1, b*S + t_12 + t_23 + 2}. This makes bank b's
  // measured t_12 = (PHY_pos_of_PRE - PHY_pos_of_ACT_first - 1) =
  // (b*S + t_12 + 1 - b*S - 1) = t_12 — bit-exact to SiMRA's template.
  int shift = parallel_min_shift(t_12, t_23, n_banks);

  // Total PHY positions emitted = max event position + 1.
  int max_pos = (n_banks - 1) * shift + t_12 + t_23 + 2;
  int num_cmd = max_pos + 1;
  // Pad to multiple of 4 for pack4 emit.
  if (num_cmd % 4) num_cmd += 4 - (num_cmd % 4);

  // Build the q_inst array — fill events first, NOPs everywhere else.
  std::vector<Mininst> q(num_cmd, SMC_NOP());
  for (int b = 0; b < n_banks; b++) {
    int e0 = b * shift;
    int e1 = b * shift + t_12 + 1;
    int e2 = b * shift + t_12 + t_23 + 2;
    // ACT R_first / PRE / ACT R_second on bank b's BAR register.
    q[e0] = SMC_ACT(bar_reg[b],   /*ibar=*/0, rfirst_reg[b],  /*irar=*/0);
    q[e1] = SMC_PRE(bar_reg[b],   /*ibar=*/0, /*pall=*/0);
    q[e2] = SMC_ACT(bar_reg[b],   /*ibar=*/0, rsecond_reg[b], /*irar=*/0);
  }

  // Pack into pack4 instructions. Each absolute position p maps to
  // slot (p % 4) of cycle (p / 4).
  for (int i = 0; i < num_cmd; i += 4)
    p.add_inst(q[i], q[i+1], q[i+2], q[i+3]);

  return p;
}

Program parallel_PRE(const std::vector<int>& bar_reg) {
  Program p;
  Mininst slots[4] = { SMC_NOP(), SMC_NOP(), SMC_NOP(), SMC_NOP() };
  for (size_t k = 0; k < bar_reg.size() && k < 4; k++)
    slots[k] = SMC_PRE(bar_reg[k], 0, 0);
  p.add_inst(slots[0], slots[1], slots[2], slots[3]);
  return p;
}

Program parallel_ACT(const std::vector<int>& bar_reg,
                     const std::vector<int>& rar_reg) {
  Program p;
  Mininst slots[4] = { SMC_NOP(), SMC_NOP(), SMC_NOP(), SMC_NOP() };
  for (size_t k = 0; k < bar_reg.size() && k < 4; k++)
    slots[k] = SMC_ACT(bar_reg[k], 0, rar_reg[k], 0);
  p.add_inst(slots[0], slots[1], slots[2], slots[3]);
  return p;
}
