#ifndef DESC_WALK_BODY_H
#define DESC_WALK_BODY_H

// Task #50 M2 -- the WALK-COMPATIBLE MAJ3 body (single source for the e2e sim
// gate and, later, the server). Mirrors emit_bank_combined_body's fused
// (fused_coset_mode()==1) shape (test_bitnet_server.cpp:835-993) but keeps the
// descriptor-walk patch surfaces free:
//
//   * The weight working row (per-round RowClone source) is the ONE reg-6 (RAR)
//     LI at the body top -- THE row patch site. The fabric splices the
//     descriptor's row_base[16:0] into every fetched LI with RT==6
//     (frontend.v:361-373), so the resident stays fixed while each iteration
//     clones a different weight row.
//   * The two x-seed wrRows carry their pattern through reg 5 (LOOP_ROWS) --
//     THE x patch sites (build32: was reg 12/PATTERN_REG; moved to a no-bit-23
//     RT so the frontend x-compare mirrors the never-flaky ROW=6 compare, per
//     desc_smoke RESULT.md). The fabric splices x_word[31:0] into every LI with
//     RT==5. Both sites get the same descriptor x_word (correct semantics).
//   * Every OTHER row goes through reg 8 (NUM_ROWS_REG, off surface) and every
//     constant pattern (ONE / 0) through reg 15 (off surface), so the walk
//     never corrupts the open_rows[] addresses or the ONE/0 constants.
//
// Register discipline (statically asserted by gen_body_walk --check): exactly
// one LI with RT==6, exactly two LIs with RT==5, reg 6 read only by the
// RowClone's first ACT, reg 8/15 re-LI'd before every use.
//
// Emission substitutions vs emit_bank_combined_body:
//   step 1 RowClone : doubleACT_regfirst(30,1, RAR, Rfirst)  (source from reg 6)
//   step 3 x seeds  : wrRow_immediate_label_regs(..., row_reg=8, pat_reg=5)
//   step 3 consts   : res-clone (use_consts) OR wrRow ... (row_reg=8, pat_reg=15)
//   step 6 rdRow    : rdRow_immediate_label_regs(..., row_reg=8)
//   + an LDWD-zero prologue (LI 0->reg15 + 16xLDWD(15,i)) before the rdRow so
//     ddr_wdata==0 when the result streams (ACCUM XOR-against-wdata insurance;
//     readback_engine.v:260 zeros diff_ref in ACCUM mode, so this is harmless
//     in sim but load-bearing on any DIFF-mode silicon path).

#include "../util.h"
#include "prog.h"
#include <string>

namespace desc_walk {

// SiMRA frac() template -- byte-for-byte the static frac_template in
// test_bitnet_server.cpp:224-240 (a static there; replicated here so the header
// is self-contained). Emits an ACT(r_frac_addr) / PRE pair with t_frac spacing.
static inline Program frac_template(int t_frac, uint32_t r_frac_addr) {
  Program p;
  int R_FRAC_REG = RF_REG;
  int bank_reg = BAR;
  p.add_inst(all_nops());
  p.add_inst(SMC_LI(r_frac_addr, R_FRAC_REG));
  int num_cmd = 2 + t_frac;
  num_cmd += (num_cmd % 4) ? (4 - (num_cmd % 4)) : 0;
  Mininst* q_inst = new Mininst[num_cmd];
  for (int i = 0; i < num_cmd; i++) q_inst[i] = SMC_NOP();
  q_inst[0]            = SMC_ACT(bank_reg, 0, R_FRAC_REG, 0);
  q_inst[t_frac + 1]   = SMC_PRE(bank_reg, 0, 0);
  for (int i = 0; i < num_cmd; i += 4)
    p.add_inst(q_inst[i], q_inst[i + 1], q_inst[i + 2], q_inst[i + 3]);
  delete[] q_inst;
  return p;
}

// Framed RowClone src->dst -- identical dwell + framing to emit_const_clone
// (test_bitnet_server.cpp:812-819) and the step-1 clone: PRE / SLEEP /
// doubleACT(30,1) / SLEEP / PRE / SLEEP. Off the patch surface entirely (both
// rows are immediates inside doubleACT's RF_REG / LOOP_COLS LIs).
static inline void emit_const_clone(Program& p, uint32_t src_row, uint32_t dst_row) {
  p.add_below(PRE(BAR, 0, 0));
  p.add_inst(SMC_SLEEP(6));
  p.add_below(doubleACT(/*t_12=*/30, /*t_23=*/1, src_row, dst_row));
  p.add_inst(SMC_SLEEP(6));
  p.add_below(PRE(BAR, 0, 0));
  p.add_inst(SMC_SLEEP(6));
}

// Off-surface registers for the walk body.
static const int WALK_ROW_REG = NUM_ROWS_REG;   // reg 8  : constant op rows
static const int WALK_CONST_PAT_REG = 15;       // reg 15 : ONE / 0 constants (never patched)
// build32 (task #50 / desc_smoke fix): the x-seed pattern register moved
// 12 (PATTERN_REG) -> 5 (LOOP_ROWS). The frontend x-patch surface is now
// RT==5 (frontend.v X_PATCH_REG = 4'd5 = 0101, no bit-23 term, mirroring
// ROW=6=0110) -- the silicon-marginal RT==12 compare (1100) is retired.
// reg 5 is unused everywhere else in the walk body, so ONLY these two
// x-seed data LIs are patched. util.cpp uses LOOP_COLS(13)/NUM_COLS_REG(14)
// for its column loops, never LOOP_ROWS, so the builders are unaffected.
static const int WALK_X_PAT_REG = LOOP_ROWS;    // reg 5  : x-seed pattern (THE x patch site; was PATTERN_REG=12)

} // namespace desc_walk

// The walk-compatible body. `open_rows` must supply at least indices
// {0,1,2,4,8,13,14}. res_one_row / res_zero_row are only consulted when
// use_consts is true.
static inline void emit_desc_walk_body(Program& p,
                                       int bank_id,
                                       uint32_t Rfirst, uint32_t Rsecond,
                                       const uint32_t open_rows[16],
                                       int label_base,
                                       uint32_t res_one_row,
                                       uint32_t res_zero_row,
                                       bool use_consts) {
  using namespace desc_walk;
  const int RR = WALK_ROW_REG;        // 8
  const int CP = WALK_CONST_PAT_REG;  // 15
  const int XP = WALK_X_PAT_REG;      // 5  (build32: x-seed patch site, was PATTERN_REG=12)

  // Per-bank register setup (mirror :835-836).
  p.add_inst(SMC_LI(bank_id, BAR));
  p.add_inst(SMC_LI(128, NUM_COLS_REG));

  // THE single reg-6 patch site: placeholder weight working row. The fabric
  // splices descriptor row_base here every iteration. Placeholder is harmless
  // (the RowClone that reads it does not write storage in the behavioral sim,
  // and on silicon the patch overwrites it before the ACT).
  p.add_inst(SMC_LI(open_rows[0] /*placeholder -> row_base*/, RAR));

  // 1. RowClone weight(reg6) -> Rfirst (mirror :843-848). Source from reg 6.
  p.add_below(PRE(BAR, 0, 0));
  p.add_inst(SMC_SLEEP(6));
  p.add_below(doubleACT_regfirst(/*t_12=*/30, /*t_23=*/1, /*r_first_reg=*/RAR, /*r_second=*/Rfirst));
  p.add_inst(SMC_SLEEP(6));
  p.add_below(PRE(BAR, 0, 0));
  p.add_inst(SMC_SLEEP(6));

  // 2. Broadcast Rfirst -> all 16 open rows (mirror :851-854). Off surface.
  p.add_below(doubleACT(10, 2, Rfirst, Rsecond));
  p.add_inst(SMC_SLEEP(6));
  p.add_below(PRE(BAR, 0, 0));
  p.add_inst(SMC_SLEEP(6));

  // 3. Deposits, fused (fused_coset_mode()==1) shape: ONE@op[0], x@op[1],
  //    0@op[2], x@op[4], 0@op[8]; then the two coset doubleACTs (mirror
  //    :903-948). x-seed patterns ride reg 12 (patched); constants ride
  //    reg 15 / res-clones (off surface); all rows ride reg 8.
  if (use_consts) {
    // Order load-bearing (deposit safety, :906-918): zero clones first, then
    // ONE, then the x seeds.
    emit_const_clone(p, res_zero_row, open_rows[2]);
    emit_const_clone(p, res_zero_row, open_rows[8]);
    emit_const_clone(p, res_one_row,  open_rows[0]);
    p.add_below(wrRow_immediate_label_regs(BAR, open_rows[1], 0u /*->x_word*/,
                                           label_base + 1, RR, XP));
    p.add_below(wrRow_immediate_label_regs(BAR, open_rows[4], 0u /*->x_word*/,
                                           label_base + 3, RR, XP));
  } else {
    // ONE, x, 0, x, 0 -- exactly the :922-927 order/labels, with the row and
    // const-pattern registers moved off the patch surface.
    p.add_below(wrRow_immediate_label_regs(BAR, open_rows[0], ONE,
                                           label_base + 0, RR, CP));
    p.add_below(wrRow_immediate_label_regs(BAR, open_rows[1], 0u /*->x_word*/,
                                           label_base + 1, RR, XP));
    p.add_below(wrRow_immediate_label_regs(BAR, open_rows[2], 0u,
                                           label_base + 2, RR, CP));
    p.add_below(wrRow_immediate_label_regs(BAR, open_rows[4], 0u /*->x_word*/,
                                           label_base + 3, RR, XP));
    p.add_below(wrRow_immediate_label_regs(BAR, open_rows[8], 0u,
                                           label_base + 4, RR, CP));
  }
  // coset doubleACTs (mirror :937-948) -- off surface (immediate rows).
  p.add_inst(SMC_SLEEP(6));
  p.add_below(PRE(BAR, 0, 0));
  p.add_inst(SMC_SLEEP(6));
  // x -> coset {1,5,9,13}
  p.add_below(doubleACT(10, 2, open_rows[1], open_rows[13]));
  p.add_inst(SMC_SLEEP(6));
  p.add_below(PRE(BAR, 0, 0));
  p.add_inst(SMC_SLEEP(6));
  // 0 -> coset {2,6,10,14}
  p.add_below(doubleACT(10, 2, open_rows[2], open_rows[14]));

  p.add_inst(SMC_SLEEP(6));
  p.add_below(PRE(BAR, 0, 0));
  p.add_inst(SMC_SLEEP(6));

  // 4. Frac discharge x3 on open_rows[0] (mirror :968-972). Uses RF_REG=10.
  for (int j = 0; j < 3; j++) {
    p.add_inst(SMC_SLEEP(6));
    p.add_below(desc_walk::frac_template(/*t_frac=*/0, open_rows[0]));
    p.add_inst(SMC_SLEEP(6));
  }
  p.add_inst(SMC_SLEEP(6));
  p.add_below(PRE(BAR, 0, 0));
  p.add_inst(SMC_SLEEP(6));

  // 5. MAJ3 doubleACT(0,0) (mirror :978-981). Off surface.
  p.add_below(doubleACT(0, 0, Rfirst, Rsecond));
  p.add_inst(SMC_SLEEP(6));
  p.add_below(PRE(BAR, 0, 0));
  p.add_inst(SMC_SLEEP(6));

  // LDWD-zero prologue (17 instr): guarantee ddr_wdata==0 when the result rdRow
  // streams. Reg 15 is never patched (RT==15). Cheap insurance for the ACCUM
  // XOR-against-wdata term (design note "ACCUM XOR-against-wdata trap").
  p.add_inst(SMC_LI(0, CP));
  for (int i = 0; i < 16; i++) p.add_inst(SMC_LDWD(CP, i));

  // 6. Read result open_rows[0] (mirror :987-992). Row via reg 8 (NOT reg 6,
  //    so the descriptor row_base never lands on the read address).
  p.add_below(rdRow_immediate_label_regs(BAR, open_rows[0], label_base + 999, RR));
  p.add_inst(all_nops());
  p.add_inst(all_nops());
  p.add_below(PRE(BAR, 0, 0));
  p.add_inst(all_nops());
  p.add_inst(all_nops());
}

#endif // DESC_WALK_BODY_H
