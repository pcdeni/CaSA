// Lane-2 MVDRAM GeMV server — PHASE 1 (LANE2_GEMV_SERVER.md order-of-work #1).
// 2026-07-18. MVDRAM-reproduction lane ONLY: their conventions, Road-A
// in-DRAM accumulation (ADR-005), strictly separate from the BitNet
// production server (separate magics, separate calib, separate rows).
//
// What it does
//   LOAD_MATRIX (0x4D563001): {handle,q_bits,K,M, bitplanes} -> ack.
//   GEMV        (0x4D563002): {handle,r_bits, activation bitplanes} -> y[M] i32.
//   y[m] = sum_n x[n] * W[m][n], W q-bit two's-complement signed, x r-bit
//   (r_bits==1: binary {0,1} — matches the existing gemv kernels' RB==1 math;
//   r_bits>1: two's-complement signed, FAC top bit negative), computed as
//   y = sum_i sum_c FAC(i,QB)*FAC(c,RB) * popcount_plane(W_i selected by x_c)
//   with the popcount done IN-DRAM by the validated dual-track MAJ carry-save
//   adder tree (carry = MAJ3(a,b,cin); sum = MAJ5(a,b,cin,~carry,~carry)) on
//   the MAJ5-screened 16-row tuple (calib_maj5_dimm2.txt s86, bender 2 bank 0).
//
// Faithfulness grade (phase 1 = the June "correctness-faithful" shape,
// FAITHFULNESS.md; deviations documented in REPRODUCTION.md):
//   - On-the-fly encoding resolved HOST-side BY DEFAULT: the activation bit
//     selects the weight row vs zero row as the adder-tree input
//     (test_mvdram_gemv.cpp convention); zero-activation taps are SKIPPED
//     entirely (SV-D zero-skip, test_mvdram_gemv_inplace PIM_SKIP convention).
//     2026-07-20: LANE2_ENCODE=clone switches to the paper's SV-C
//     RowClone-encoded products — the activation bit selects the RowCopy
//     SOURCE and the product is physically created by a clone into the
//     computation row (test_mvdram_fastpath_ab.cpp fused-gate machinery:
//     4-row MAJ3 tuple {54340,54341,54724,54725} at SUB 54272, screened
//     3-bit antichain value-row masks, load order i1->i2->i0, whole gate =
//     ONE program). Clone mode computes the FA with the all-MAJ3 dual-rail
//     DAG (test_mvdram_fulladder.cpp identity) because MAJ5 is not
//     clone-loadable on this tuple class — so clone mode is inherently
//     dual-track (complements formed in-DRAM at every level).
//   - Operands loaded by per-column WRITE in host mode (writes don't
//     XOR-spread); the ADD itself is in-DRAM. Intermediate ~carry formed on
//     host between the MAJ3 and MAJ5 (test_mvdram_adder.cpp convention)
//     BY DEFAULT. 2026-07-20: LANE2_DUALTRACK=1 implements the paper's
//     Fig-15/SVII dual-track instead: LOAD_MATRIX also prepares the
//     inverted matrix bitplanes (doubling matrix rows exactly as the paper
//     describes), and every FA computes carry AND ~carry (MAJ3 De Morgan),
//     sum AND ~sum (MAJ5 on complement rails) IN-DRAM — the host transports
//     rail data between tiled ops but never applies NOT anywhere in the
//     chain. Rail consistency (v XOR nv == all-ones) is checked host-side
//     per FA on screened lanes as a free integrity diagnostic.
//   - One screened MAJ5 tuple => the contraction is TILED through it (the
//     tree time-multiplexes the same 16 rows); outputs are bitline-parallel:
//     M outputs live on ceil(M/32) op-screened column segments.
//   - GEMV_PARTIALS (0x4D563003, 2026-07-20): same request body as GEMV but
//     returns per-32-weight-block partial sums (M x ceil(K/32) i32, m-major)
//     instead of the whole-K dot — the paper's SII-C2/SVII "partial sums per
//     subarray, aggregated by the processor" at q4_0/q8_0 block granularity.
//     This is what makes EXACT fp32 reconstruction possible host-side
//     (per-block scales applied to exact integer partials).
//
// Program shapes are copied from the validated tools; the only change is
// 8K-IMEM packing (LANE2_PACK=0 restores the per-chunk 2K-era shape):
//   - pcwrite: 3 chunk blocks in ONE program (~4.4K insts) instead of 3
//     programs — identical instruction sequence, fewer round trips.
//   - uniform rows: wrRow_immediate_label batched <=16 per program (the
//     proven idiom; unique label counter, never >1x same label per Program).
//   - MAJ finish (frac'd-ONE ref + doubleACT(0,0) + read) verbatim from
//     test_mvdram_gemv.cpp maj_finish.
//
// Protocol (stdin/stdout, binary little-endian, length-prefixed both ways):
//   request:  u32 len, then len bytes (len==0 => clean shutdown)
//   response: u32 len, then len bytes
//   LOAD_MATRIX req:  u32 magic=0x4D563001, u32 handle, u32 q_bits, u32 K,
//                     u32 M, then q_bits * M * ceil(K/8) bytes: bitplane-major
//                     (i=0..q_bits-1), row-major (m=0..M-1), each row
//                     ceil(K/8) bytes, bit n = (byte[n>>3]>>(n&7))&1 = bit i
//                     of two's-complement W[m][n]  (SVI horizontal layout,
//                     serialized row-major as the spec says).
//   LOAD ack resp:    u32 magic=0x4D5630F1, u32 handle, u32 status (0=ok).
//   GEMV req:         u32 magic=0x4D563002, u32 handle, u32 r_bits, then
//                     r_bits * ceil(K/8) bytes, plane-major; plane c bit n =
//                     bit c of x[n] (two's-complement if r_bits>1).
//   GEMV resp:        u32 magic=0x4D5630F2, u32 handle, u32 status, u32 M,
//                     then M * i32 y (status!=0 => M==0, no payload).
//   PARTIALS req:     u32 magic=0x4D563003, u32 handle, u32 r_bits, then the
//                     same activation planes as GEMV.
//   PARTIALS resp:    u32 magic=0x4D5630F3, u32 handle, u32 status, u32 M,
//                     u32 NBLK=ceil(K/32), then M*NBLK i32 partials, m-major
//                     (p[m*NBLK+b] = sum over taps n in [32b,32b+32) of
//                     x[n]*W[m][n], exact integers).
//   status: 0=ok 1=bad request 2=silicon error 3=unknown handle
//           4=not enough screened columns 5=shape unsupported
//
// Argv: ./lane2-gemv-server <bender> <calib> <bank> <s_id> <colmask>
//   (same argv conventions as mvdram-gemv-exe; calib = calib_maj5_dimm2.txt,
//    colmask = colmask_dimm2_s86_robust.txt)
// Env:  BITSTREAM_IMEM=8192 PIM_RECV_TIMEOUT_MS=15000  (mandatory on silicon)
//       PIM_VOTE3=1     3x majority vote per MAJ (recovers transient noise;
//                       host/dualtrack encode only — ignored in clone mode)
//       LANE2_PACK=0    disable 8K-IMEM packing (debug fallback)
//       LANE2_DUALTRACK=1  paper Fig-15 dual-track: ~W planes at LOAD, all
//                          complements formed in-DRAM (default 0 = host ~)
//       LANE2_ENCODE=clone RowClone-encoded products (default host); implies
//                          dual-rail; SILICON-ONLY (sim has no model for the
//                          fastpath tuple)
//       LANE2_SCREEN_TRIALS=N  op-screen repeats (default 3), both engines
//       LANE2_BACKEND=sim  use in-process SimDramModel (bring-up only; the
//                          sim's MAJ model is MAJ3-tuple-oriented — silicon
//                          is the deliverable)
//       LANE2_ACCUM=1|2    Road-B product-row dataflow (2026-07-21, build-6
//                          image): y[m] = Σ_i Σ_c FAC·popcount(W_i[m] AND x_c)
//                          with ONE fused program per product (clone-x +
//                          pcwrite-W + MAJ3 AND on the fastpath tuple + read
//                          of the product row) — no CSA tree, no per-FA row
//                          transport. 1 = READ_MODE readout + host popcount
//                          (the A/B baseline arm); 2 = DIFF-accum readout:
//                          each program's read drains as ONE 32-bit total
//                          (96 B vs 8 KB), batched-receive per the
//                          test_popcount_hw build-6 consumption pattern.
//                          Identical program bytes in both arms. Per ADR-005
//                          this is the FPGA-accelerated NON-reproduction arm
//                          (labelled distinctly; never blended with Road A).
//                          Silicon-only; requires LANE2_PACK=1 + the build-6
//                          bitstream (trailer magic 0xDBC0DE04) for
//                          unbounded DIFF sessions.
//       LANE2_XREFRESH=N   ACCUM modes: rewrite + re-verify the resident
//                          activation row every N products (default 64;
//                          doubles as an in-stream order/integrity sentinel
//                          in the accum arm — its total is known)
//       LANE2_PLANE_PACK=1 (accum arm 2 only, 2026-07-21) fold all qb
//                          plane-gates of one (m,c) product into ONE
//                          program with 2^i replicated reads per plane and
//                          a complemented top plane: the single accum
//                          total IS the plane-weighted partial (host adds
//                          −2^(qb-1)·pc(x_c)). Program count ÷qb; the
//                          multi-read accum regime's first exerciser.
//                          Wants LANE2_WRES=1 (spilled outputs fall back
//                          to per-plane pcwrite gates).
#include "instruction.h"
#include "prog.h"
#include "platform.h"
#include "sim_platform.h"
#include "util.h"

#include <algorithm>
#include <array>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <vector>

using namespace std;
typedef vector<uint32_t> Row;   // 2048 x u32 = one DRAM row (65536 bitlines)

static constexpr uint32_t MAGIC_LOAD = 0x4D563001u;  // "MV0\x01"
static constexpr uint32_t MAGIC_GEMV = 0x4D563002u;
static constexpr uint32_t MAGIC_PART = 0x4D563003u;
static constexpr uint32_t MAGIC_LOAD_ACK = 0x4D5630F1u;
static constexpr uint32_t MAGIC_GEMV_ACK = 0x4D5630F2u;
static constexpr uint32_t MAGIC_PART_ACK = 0x4D5630F3u;

static const int CHUNK_COLS[3] = {43, 43, 42};

static SoftMCPlatform* PF = nullptr;
static int BANK = 0;
static int PACK = 1;        // LANE2_PACK
static int MV = 0;          // PIM_VOTE3
static int DT = 0;          // LANE2_DUALTRACK
static int ENC_CLONE = 0;   // LANE2_ENCODE=clone
static int ACCUM = 0;       // LANE2_ACCUM: 0 off, 1 read arm, 2 accum arm
// Reference policy for the 16-row MAJ tuple (LANE2_REF_POLICY):
// default = SiMRA's frac'd-ONE (init ONE, 3 frac pulses, t_frac 0);
// "zero2" = the frac-maj5 sweep winner (init ZERO, 2 pulses, t0):
// 93.5% strict MAJ5 cols on s86 vs 89.1% legacy — the FracDRAM-style
// conditioning MVDRAM cites for error-free MAJX, applied to the chained
// adder for the first time here. MAJ3 is policy-insensitive (measured),
// so the fastpath/clone engines keep their own convention untouched.
static uint32_t REF_INIT = ONE;
static int REF_NFRAC = 3;
static int response_fd = 1; // dup'd stdout (binary channel)
static long N_EXEC = 0, N_MAJ = 0, N_FA = 0, N_PCW = 0;
static long N_RAIL_LANES = 0, N_RAIL_VIOL = 0;  // dual-rail consistency diag

static void die(const char* msg, int code) {
  fprintf(stderr, "[lane2] FATAL: %s\n", msg);
  _exit(code);
}

// ---------- silicon primitives (copied from test_mvdram_gemv.cpp) ----------

// per-column write of an arbitrary 2048-u32 row. PACK=1: the three 2K-era
// chunk programs concatenated into one (~4.4K insts, needs the 8K IMEM);
// PACK=0: verbatim three programs. Same instruction sequence either way.
static void pcwrite(uint32_t row, const uint32_t* seg) {
  N_PCW++;
  if (PACK) {
    Program p;
    p.add_inst(SMC_LI(8, CASR)); p.add_inst(SMC_LI(BANK, BAR));
    int cs = 0;
    for (int ch = 0; ch < 3; ch++) {
      int n = CHUNK_COLS[ch]; const uint32_t* cd = seg + cs * 16;
      p.add_inst(SMC_LI(row, RAR)); p.add_inst(SMC_LI(cs * 8, CAR));
      p.add_below(PRE(BAR, 0, 0)); p.add_below(ACT(BAR, 0, RAR, 0));
      for (int k = 0; k < n; k++) {
        const uint32_t* sl = cd + k * 16;
        for (int s = 0; s < 16; s++) { p.add_inst(SMC_LI(sl[s], PATTERN_REG)); p.add_inst(SMC_LDWD(PATTERN_REG, s)); }
        p.add_below(WRITE(BAR, CAR, 1)); p.add_inst(SMC_SLEEP(8));
      }
      p.add_inst(SMC_SLEEP(8)); p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_SLEEP(4));
      cs += n;
    }
    p.add_inst(SMC_END());
    PF->execute(p); N_EXEC++;
    return;
  }
  int cs = 0;
  for (int ch = 0; ch < 3; ch++) {
    int n = CHUNK_COLS[ch]; const uint32_t* cd = seg + cs * 16;
    Program p;
    p.add_inst(SMC_LI(8, CASR)); p.add_inst(SMC_LI(BANK, BAR));
    p.add_inst(SMC_LI(row, RAR)); p.add_inst(SMC_LI(cs * 8, CAR));
    p.add_below(PRE(BAR, 0, 0)); p.add_below(ACT(BAR, 0, RAR, 0));
    for (int k = 0; k < n; k++) {
      const uint32_t* sl = cd + k * 16;
      for (int s = 0; s < 16; s++) { p.add_inst(SMC_LI(sl[s], PATTERN_REG)); p.add_inst(SMC_LDWD(PATTERN_REG, s)); }
      p.add_below(WRITE(BAR, CAR, 1)); p.add_inst(SMC_SLEEP(8));
    }
    p.add_inst(SMC_SLEEP(8)); p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_SLEEP(4)); p.add_inst(SMC_END());
    PF->execute(p); N_EXEC++;
    cs += n;
  }
}

// batched uniform-row writes: <=16 wrRow_immediate_label per Program (the
// proven idiom), monotonic labels (hardcoded-label macros hang if a label is
// emitted >1x per Program — softmc_label_collision).
static void uwrite_batch(const vector<pair<uint32_t, uint32_t>>& rows) {
  static int lbl = 1000;
  size_t i = 0;
  while (i < rows.size()) {
    size_t n = rows.size() - i; if (n > (size_t)(PACK ? 16 : 1)) n = PACK ? 16 : 1;
    Program p;
    p.add_inst(SMC_LI(8, CASR)); p.add_inst(SMC_LI(BANK, BAR));
    p.add_inst(SMC_LI(128, NUM_COLS_REG));
    p.add_below(PRE(BAR, 0, 0));
    for (size_t k = 0; k < n; k++)
      p.add_below(wrRow_immediate_label(BAR, rows[i + k].first, rows[i + k].second, lbl++));
    p.add_inst(SMC_SLEEP(6)); p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_END());
    PF->execute(p); N_EXEC++;
    i += n;
  }
}

// frac discharge builder — verbatim test_mvdram_gemv.cpp frac_b.
static Program frac_b(int t_frac, int r) {
  Program p; p.add_inst(all_nops());
  p.add_inst(SMC_LI(r, RF_REG)); int nc = 2 + t_frac; nc += 4 - (nc % 4); Mininst q[nc];
  for (int i = 0; i < nc; i++) q[i] = SMC_NOP();
  q[0] = SMC_ACT(BAR, 0, RF_REG, 0); q[t_frac + 1] = SMC_PRE(BAR, 0, 0);
  for (int i = 0; i < nc; i += 4) p.add_inst(q[i], q[i + 1], q[i + 2], q[i + 3]);
  return p;
}

// frac(open[0]) x3 + doubleACT(0,0,Rf,Rs) + read open[0] — verbatim
// test_mvdram_gemv.cpp maj_finish.
static void maj_finish(uint32_t Rf, uint32_t Rs, uint32_t open0, uint8_t out[8192]) {
  Program p; p.add_inst(SMC_LI(8, CASR)); p.add_inst(SMC_LI(BANK, BAR)); p.add_inst(SMC_LI(128, NUM_COLS_REG));
  p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_SLEEP(6));
  for (int j = 0; j < REF_NFRAC; j++) { p.add_inst(SMC_SLEEP(6)); p.add_below(frac_b(0, open0)); p.add_inst(SMC_SLEEP(6)); }
  p.add_inst(SMC_SLEEP(6)); p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_SLEEP(6));
  p.add_below(doubleACT(0, 0, Rf, Rs)); p.add_inst(SMC_SLEEP(6)); p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_SLEEP(6));
  p.add_below(rdRow_immediate(BAR, open0)); p.add_inst(all_nops()); p.add_inst(all_nops());
  p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_END());
  PF->execute(p); N_EXEC++;
  if (PF->receiveData(out, 8192) != 8192)
    die("receiveData failed/short — silicon wedge? (recover per RUNBOOK: fpga-helper unload/load)", 6);
}

// one MAJ operand: per-column data or a uniform pattern
struct Opnd { const uint32_t* data; bool uniform; uint32_t pat; };
static Opnd OD(const Row& r) { return {r.data(), false, 0}; }
static Opnd OU(uint32_t pat) { return {nullptr, true, pat}; }

struct Calib { int s_id, bank; uint32_t Rf, Rs; vector<uint32_t> open; };
static Calib CAL;

// MAJ3 over the tuple: op0 -> open[1..5], op1 -> open[6..10], op2 -> open[11..15]
// (test_mvdram_gemv.cpp hw_maj3 placement); frac = open[0] = ONE.
static void hw_maj3(const Opnd& a, const Opnd& b, const Opnd& c, uint32_t res[2048]) {
  const Opnd* ops[3] = {&a, &b, &c};
  vector<pair<uint32_t, uint32_t>> uni;
  for (int g = 0; g < 3; g++)
    for (int i = 1 + g * 5; i <= 5 + g * 5; i++) {
      if (ops[g]->uniform) uni.push_back({CAL.open[i], ops[g]->pat});
      else pcwrite(CAL.open[i], ops[g]->data);
    }
  uni.push_back({CAL.open[0], REF_INIT});
  uwrite_batch(uni);
  uint8_t rowb[8192]; maj_finish(CAL.Rf, CAL.Rs, CAL.open[0], rowb);
  for (int s = 0; s < 2048; s++) memcpy(&res[s], &rowb[s * 4], 4);
  N_MAJ++;
}

// MAJ5: operand g -> open[1+g*3 .. 3+g*3], g=0..4 (test_mvdram_gemv.cpp
// hw_maj5 placement); frac = open[0] = ONE.
static void hw_maj5(const Opnd I[5], uint32_t res[2048]) {
  vector<pair<uint32_t, uint32_t>> uni;
  for (int g = 0; g < 5; g++)
    for (int r = 0; r < 3; r++) {
      uint32_t row = CAL.open[1 + g * 3 + r];
      if (I[g].uniform) uni.push_back({row, I[g].pat});
      else pcwrite(row, I[g].data);
    }
  uni.push_back({CAL.open[0], REF_INIT});
  uwrite_batch(uni);
  uint8_t rowb[8192]; maj_finish(CAL.Rf, CAL.Rs, CAL.open[0], rowb);
  for (int s = 0; s < 2048; s++) memcpy(&res[s], &rowb[s * 4], 4);
  N_MAJ++;
}

static void vote3(const uint32_t* r0, const uint32_t* r1, const uint32_t* r2, uint32_t* o) {
  for (int s = 0; s < 2048; s++) o[s] = (r0[s] & r1[s]) | (r1[s] & r2[s]) | (r0[s] & r2[s]);
}
static void maj3_v(const Opnd& a, const Opnd& b, const Opnd& c, uint32_t res[2048]) {
  if (!MV) { hw_maj3(a, b, c, res); return; }
  static uint32_t r0[2048], r1[2048], r2[2048];
  hw_maj3(a, b, c, r0); hw_maj3(a, b, c, r1); hw_maj3(a, b, c, r2);
  vote3(r0, r1, r2, res);
}
static void maj5_v(const Opnd I[5], uint32_t res[2048]) {
  if (!MV) { hw_maj5(I, res); return; }
  static uint32_t r0[2048], r1[2048], r2[2048];
  hw_maj5(I, r0); hw_maj5(I, r1); hw_maj5(I, r2);
  vote3(r0, r1, r2, res);
}

// ---------- dual-track full adder ----------
// Default (phase-1) shape: carry = MAJ3(a,b,cin); ~carry on HOST; sum =
// MAJ5(a,b,cin,~carry,~carry). (test_mvdram_adder.cpp / test_mvdram_gemv.cpp
// shape — the intermediate complement is the host-glued part, documented.)
// LANE2_DUALTRACK=1 (full_add_dual): the paper's Fig-15/SVII dual-track —
// both rails resident; ~carry = MAJ3(~a,~b,~cin) IN-DRAM (De Morgan), sum =
// MAJ5(a,b,cin,~carry,~carry), ~sum = MAJ5(~a,~b,~cin,carry,carry) IN-DRAM.
// The host never applies NOT in the chain: leaf complements come from the ~W
// bitplanes prepared at LOAD (the paper's inverted matrix rows), intermediate
// complements from silicon; the host only transports rail data (the
// documented tiled-transport deviation, unchanged).
struct Node { Row v; Row nv; bool zero; };  // zero => uniform-0 (complement ONE)
static Opnd node_op(const Node& n) { return n.zero ? OU(0u) : OD(n.v); }
static Opnd node_nop(const Node& n) { return n.zero ? OU(ONE) : OD(n.nv); }

static void full_add(const Node& a, const Node& b, const Node& c, Node& sum, Node& carry) {
  N_FA++;
  carry.zero = false; carry.v.assign(2048, 0);
  maj3_v(node_op(a), node_op(b), node_op(c), carry.v.data());
  static uint32_t nc[2048];
  for (int s = 0; s < 2048; s++) nc[s] = ~carry.v[s];
  Opnd ncop = {nc, false, 0};
  Opnd I[5] = {node_op(a), node_op(b), node_op(c), ncop, ncop};
  sum.zero = false; sum.v.assign(2048, 0);
  maj5_v(I, sum.v.data());
}

// dual-rail consistency diagnostic (host-side observation only)
static void rail_check(const Row& v, const Row& nv);

static void full_add_dual(const Node& a, const Node& b, const Node& c, Node& sum, Node& carry) {
  N_FA++;
  carry.zero = false; carry.v.assign(2048, 0); carry.nv.assign(2048, 0);
  maj3_v(node_op(a), node_op(b), node_op(c), carry.v.data());
  maj3_v(node_nop(a), node_nop(b), node_nop(c), carry.nv.data());  // ~carry IN-DRAM
  Opnd ncop = OD(carry.nv);
  Opnd I[5] = {node_op(a), node_op(b), node_op(c), ncop, ncop};
  sum.zero = false; sum.v.assign(2048, 0); sum.nv.assign(2048, 0);
  maj5_v(I, sum.v.data());
  Opnd cop = OD(carry.v);
  Opnd J[5] = {node_nop(a), node_nop(b), node_nop(c), cop, cop};
  maj5_v(J, sum.nv.data());                                        // ~sum IN-DRAM
  rail_check(carry.v, carry.nv); rail_check(sum.v, sum.nv);
}

// clone-encode FA (defined below with the clone engine)
static void cl_full_add(const Node& a, const Node& b, const Node& c, Node& sum, Node& carry);

// in-DRAM carry-save popcount tree over the selected leaves (bit-packed rows).
// Returns count-bit rows by weight (test_mvdram_gemv_n.cpp popcount_tree
// bucket walk). The FA is dispatched by mode: default = host-glued ~carry
// (full_add, byte-identical to phase 1); LANE2_DUALTRACK => full_add_dual;
// LANE2_ENCODE=clone => cl_full_add (clone-encoded, dual-rail all-MAJ3).
static void fa_dispatch(const Node& a, const Node& b, const Node& c, Node& s, Node& cy) {
  if (ENC_CLONE) cl_full_add(a, b, c, s, cy);
  else if (DT) full_add_dual(a, b, c, s, cy);
  else full_add(a, b, c, s, cy);
}
static vector<Row> popcount_tree(vector<Node> inp) {
  const int WMAX = 20;
  vector<vector<Node>> bk(WMAX + 1);
  for (auto& d : inp) bk[0].push_back(std::move(d));
  for (int pass = 0; pass < 128; pass++) {
    bool ch = false;
    for (int w = 0; w < WMAX; w++) {
      while (bk[w].size() >= 3) {
        Node a = std::move(bk[w].back()); bk[w].pop_back();
        Node b = std::move(bk[w].back()); bk[w].pop_back();
        Node c = std::move(bk[w].back()); bk[w].pop_back();
        Node s, cy; fa_dispatch(a, b, c, s, cy);
        bk[w].push_back(std::move(s)); bk[w + 1].push_back(std::move(cy)); ch = true;
      }
      if (bk[w].size() == 2) {
        Node a = std::move(bk[w].back()); bk[w].pop_back();
        Node b = std::move(bk[w].back()); bk[w].pop_back();
        Node z; z.zero = true; z.v.assign(2048, 0);
        Node s, cy; fa_dispatch(a, b, z, s, cy);
        bk[w].push_back(std::move(s)); bk[w + 1].push_back(std::move(cy)); ch = true;
      }
    }
    if (!ch) break;
  }
  vector<Row> cnt;
  for (int w = 0; w <= WMAX; w++) {
    if (!bk[w].empty()) cnt.push_back(bk[w][0].zero ? Row(2048, 0) : bk[w][0].v);
    else cnt.push_back(Row(2048, 0));
  }
  return cnt;
}

// ---------- calib / colmask ----------
static vector<Calib> read_calib(const string& p) {
  vector<Calib> o; ifstream f(p); string ln;
  while (getline(f, ln)) {
    if (ln.empty() || ln[0] == '#') continue;
    istringstream s(ln); Calib c;
    if (!(s >> c.s_id >> c.bank >> c.Rf >> c.Rs)) continue;
    uint32_t v; while (s >> v) c.open.push_back(v);
    if (c.open.size() == 16) o.push_back(c);
  }
  return o;
}
static vector<uint8_t> read_mask(const string& p) {
  vector<uint8_t> m(2048, 0); ifstream f(p); string ln;
  while (getline(f, ln)) { if (ln.empty() || ln[0] == '#') continue; int c = atoi(ln.c_str()); if (c >= 0 && c < 2048) m[c] = 1; }
  return m;
}

// ---------- op-matched column screen (server startup) ----------
// A column is GeMV-reliable iff it computes carry=MAJ3 and sum=MAJ5(...,~c,~c)
// correctly for the half-add pairs (test_mvdram_gemv.cpp screen, incl. its
// exact pattern list + 16 srand(13579) randoms) AND for full-adder triples
// (test_mvdram_adder.cpp abc + randoms). Intersected with the input colmask.
static uint32_t maj3w(uint32_t a, uint32_t b, uint32_t c) { return (a & b) | (a & c) | (b & c); }
// LANE2_SCREEN_TRIALS (default 3): the whole pattern set is repeated and the
// masks AND-ed — marginal columns that pass a single point-in-time sample but
// flake later (the 07-18 full-shape residual: repeat-offender lanes) get
// excluded up front. mvdram_adder's 30-trial screen is the precedent.
static vector<uint8_t> op_screen(const vector<uint8_t>& mask, int trials) {
  vector<uint8_t> gm = mask;
  uint32_t cr[2048], sm[2048];
  for (int tr = 0; tr < trials; tr++) {
    // half-add pairs (verbatim list)
    vector<pair<uint32_t, uint32_t>> ab = {{0, 0}, {0xFFFFFFFFu, 0xFFFFFFFFu},
      {0xAAAAAAAAu, 0xCCCCCCCCu}, {0xFFFFFFFFu, 0}, {0, 0xFFFFFFFFu}, {0xAAAAAAAAu, 0x55555555u}};
    srand(13579 + tr);
    for (int r = 0; r < 16; r++) ab.push_back({(uint32_t)(rand() << 16 ^ rand()), (uint32_t)(rand() << 16 ^ rand())});
    for (auto& pr : ab) {
      uint32_t a = pr.first, b = pr.second, ec = a & b, es = a ^ b, ncu = ~ec;
      hw_maj3(OU(a), OU(b), OU(0u), cr);
      Opnd I[5] = {OU(a), OU(b), OU(0u), OU(ncu), OU(ncu)};
      hw_maj5(I, sm);
      for (int s = 0; s < 2048; s++) if (cr[s] != ec || sm[s] != es) gm[s] = 0;
    }
    // full-adder triples (mvdram_adder abc + randoms)
    vector<array<uint32_t, 3>> abc = {{0xAAAAAAAAu, 0xCCCCCCCCu, 0xF0F0F0F0u},
      {0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu}, {0, 0, 0xFFFFFFFFu}};
    for (int r = 0; r < 8; r++) abc.push_back({(uint32_t)(rand() << 16 ^ rand()),
      (uint32_t)(rand() << 16 ^ rand()), (uint32_t)(rand() << 16 ^ rand())});
    for (auto& t : abc) {
      uint32_t a = t[0], b = t[1], c = t[2];
      uint32_t ec = maj3w(a, b, c), es = a ^ b ^ c, ncu = ~ec;
      hw_maj3(OU(a), OU(b), OU(c), cr);
      Opnd I[5] = {OU(a), OU(b), OU(c), OU(ncu), OU(ncu)};
      hw_maj5(I, sm);
      for (int s = 0; s < 2048; s++) if (cr[s] != ec || sm[s] != es) gm[s] = 0;
    }
  }
  return gm;
}

// ---------- matrix store + protocol ----------
struct Matrix {
  uint32_t qb = 0, K = 0, M = 0;
  vector<vector<Row>> wrow;   // [n][i] -> 2048-u32 row (bit-packed over lanes)
  vector<vector<Row>> nwrow;  // inverted matrix bitplanes (Fig 15) — only
                              // filled when LANE2_DUALTRACK / clone encode
  vector<uint8_t> plane;      // ACCUM modes: raw wire bitplanes (qb*M*bpr);
                              // horizontal W rows are built per product
  vector<uint32_t> wres_row;  // LANE2_WRES: (i*M+m) -> resident value row
                              // holding that W bitplane row (0 = spill,
                              // pcwrite path). Residency owned by the last
                              // LOADed handle.
};
static unordered_map<uint32_t, Matrix> MATS;
static vector<int> LANES;   // screened segment ids, ascending

static inline int FAC(int idx, int nb) { return (nb > 1 && idx == nb - 1) ? -(1 << idx) : (1 << idx); }

// dual-rail consistency: on screened lanes, v XOR nv must be all-ones. A
// violation means one rail took a transient hit — counted, not corrected
// (observation only; no behavior change).
static void rail_check(const Row& v, const Row& nv) {
  for (int seg : LANES) {
    uint32_t x = v[seg] ^ nv[seg];
    N_RAIL_LANES += 32;
    if (x != 0xFFFFFFFFu) N_RAIL_VIOL += __builtin_popcount(~x);
  }
}

// ---------- clone-encode engine (LANE2_ENCODE=clone) ----------
// test_mvdram_fastpath_ab.cpp machinery verbatim: 4-row MAJ3 tuple
// {Ti0,Ti1,Ti2,Tfr} = local {68,69,452,453} of SUB 54272 (the same physical
// subarray as the s86 MAJ5 tuple), on-silicon-screened 3-bit value-row masks
// over bits 1..9 (never bits 7+8 together — the tuple generators are
// {1,384}), safe load order i1->i2->i0 (Ti0 enveloped), whole gate fused in
// ONE program: clone x->Ti1, clone y->Ti2, clone z->Ti0, wrRow(ONE)->Tfr,
// frac x3, doubleACT(0,0) MAJ3, rdRow(Ti0), clone result -> dst value row.
// The FA is the all-MAJ3 dual-rail DAG (test_mvdram_fulladder.cpp identity):
//   carry = M(a,b,c)             ~carry = M(~a,~b,~c)
//   sum   = M(M(a,b,~c), M(a,~b,c), M(~a,b,c))
//   ~sum  = M(M(~a,b,~c), M(~a,~b,c), carry)
// = 9 fused gates per FA. MAJ5 is not clone-loadable on this tuple class, so
// the paper's MAJ5 sum is replaced by the MAJ3 identity (the documented
// commodity-silicon workaround, PAPER_CONTRAST S3). Leaf products and their
// complements are cloned from rows holding the W / ~W bitplane data — the
// paper's SV-C source selection made physical; intermediates' rails are
// silicon-formed, host-transported between gates (tiled shape).
static constexpr uint32_t CL_SUB = 54272;
static constexpr uint32_t CL_Ti0 = 54340, CL_Ti1 = 54341, CL_Ti2 = 54724, CL_Tfr = 54725;
static constexpr uint32_t CL_Trf = CL_Ti0, CL_Trs = CL_Tfr;
static int LBL = 20000;  // global label counter (unique per emission — the
                         // hardcoded-label macros hang if reused in a Program)
enum { RA, RNA, RB, RNB, RC, RNC, RCAR, RNCAR, RM1, RM2, RM3, RSUM, RM4, RM5, RNSUM, NROLE };
static uint32_t CROW[NROLE];  // screened value rows by role

struct COp { bool uniform; uint32_t row; uint32_t pat; };

static void cl_read_row(uint32_t row, uint8_t out[8192]) {
  Program p;
  p.add_inst(SMC_LI(8, CASR)); p.add_inst(SMC_LI(BANK, BAR)); p.add_inst(SMC_LI(128, NUM_COLS_REG));
  p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_SLEEP(6));
  p.add_below(rdRow_immediate_label(BAR, row, LBL++)); p.add_inst(all_nops()); p.add_inst(all_nops());
  p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_END());
  PF->execute(p); N_EXEC++;
  if (PF->receiveData(out, 8192) != 8192) die("receiveData failed in cl_read_row", 6);
}
static void cl_rowclone(uint32_t src, uint32_t dst) {
  Program p;
  p.add_inst(SMC_LI(8, CASR)); p.add_inst(SMC_LI(BANK, BAR)); p.add_inst(SMC_LI(128, NUM_COLS_REG));
  p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_SLEEP(6));
  p.add_below(doubleACT(30, 1, src, dst)); p.add_inst(SMC_SLEEP(6)); p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_SLEEP(6));
  p.add_inst(SMC_END());
  PF->execute(p); N_EXEC++;
}
static void cl_zero_tuple() {
  vector<pair<uint32_t, uint32_t>> z = {{CL_Ti0, 0}, {CL_Ti1, 0}, {CL_Ti2, 0}, {CL_Tfr, 0}};
  uwrite_batch(z);
}

// fused gate: MAJ3 of three operands (row-clone or uniform-write sources),
// result read back AND cloned out to dst. z loads LAST into Ti0 (safe order).
static void cl_gate(const COp& x, const COp& y, const COp& z, uint32_t dst, uint8_t out[8192]) {
  Program p;
  p.add_inst(SMC_LI(8, CASR)); p.add_inst(SMC_LI(BANK, BAR)); p.add_inst(SMC_LI(128, NUM_COLS_REG));
  p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_SLEEP(6));
  auto put = [&](const COp& o, uint32_t T) {
    if (o.uniform) p.add_below(wrRow_immediate_label(BAR, T, o.pat, LBL++));
    else p.add_below(doubleACT(30, 1, o.row, T));
    p.add_inst(SMC_SLEEP(6)); p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_SLEEP(6));
  };
  put(x, CL_Ti1); put(y, CL_Ti2); put(z, CL_Ti0);
  p.add_below(wrRow_immediate_label(BAR, CL_Tfr, ONE, LBL++));
  p.add_inst(SMC_SLEEP(6)); p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_SLEEP(6));
  for (int j = 0; j < 3; j++) { p.add_inst(SMC_SLEEP(6)); p.add_below(frac_b(0, CL_Tfr)); p.add_inst(SMC_SLEEP(6)); }
  p.add_inst(SMC_SLEEP(6)); p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_SLEEP(6));
  p.add_below(doubleACT(0, 0, CL_Trf, CL_Trs)); p.add_inst(SMC_SLEEP(6)); p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_SLEEP(6));
  p.add_below(rdRow_immediate_label(BAR, CL_Ti0, LBL++)); p.add_inst(all_nops()); p.add_inst(all_nops());
  p.add_below(PRE(BAR, 0, 0)); p.add_inst(all_nops()); p.add_inst(all_nops());
  p.add_below(doubleACT(30, 1, CL_Ti0, dst)); p.add_inst(SMC_SLEEP(6)); p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_SLEEP(6));
  p.add_inst(SMC_END());
  PF->execute(p); N_EXEC++; N_MAJ++;
  if (PF->receiveData(out, 8192) != 8192)
    die("receiveData failed in clone gate — silicon wedge? (RUNBOOK: fpga-helper)", 6);
}

// materialize a node's rails into role rows; return operand descriptors
static void cl_mat(const Node& n, int rv, int rn, COp& ov, COp& on) {
  if (n.zero) { ov = {true, 0, 0u}; on = {true, 0, ONE}; return; }
  pcwrite(CROW[rv], n.v.data());
  pcwrite(CROW[rn], n.nv.data());
  ov = {false, CROW[rv], 0}; on = {false, CROW[rn], 0};
}

static void cl_full_add(const Node& a, const Node& b, const Node& c, Node& sum, Node& carry) {
  N_FA++;
  COp av, an, bv, bn, cv, cn;
  cl_mat(a, RA, RNA, av, an); cl_mat(b, RB, RNB, bv, bn); cl_mat(c, RC, RNC, cv, cn);
  static uint8_t bcar[8192], bncar[8192], bm1[8192], bm2[8192], bm3[8192],
                 bsum[8192], bm4[8192], bm5[8192], bnsum[8192];
  cl_gate(av, bv, cv, CROW[RCAR], bcar);                    // carry
  cl_gate(an, bn, cn, CROW[RNCAR], bncar);                  // ~carry (De Morgan)
  cl_gate(av, bv, cn, CROW[RM1], bm1);
  cl_gate(av, bn, cv, CROW[RM2], bm2);
  cl_gate(an, bv, cv, CROW[RM3], bm3);
  COp m1{false, CROW[RM1], 0}, m2{false, CROW[RM2], 0}, m3{false, CROW[RM3], 0};
  cl_gate(m1, m2, m3, CROW[RSUM], bsum);                    // sum
  cl_gate(an, bv, cn, CROW[RM4], bm4);
  cl_gate(an, bn, cv, CROW[RM5], bm5);
  COp m4{false, CROW[RM4], 0}, m5{false, CROW[RM5], 0}, car{false, CROW[RCAR], 0};
  cl_gate(m4, m5, car, CROW[RNSUM], bnsum);                 // ~sum
  sum.zero = carry.zero = false;
  sum.v.assign(2048, 0); sum.nv.assign(2048, 0);
  carry.v.assign(2048, 0); carry.nv.assign(2048, 0);
  for (int s = 0; s < 2048; s++) {
    memcpy(&sum.v[s], &bsum[s * 4], 4);   memcpy(&sum.nv[s], &bnsum[s * 4], 4);
    memcpy(&carry.v[s], &bcar[s * 4], 4); memcpy(&carry.nv[s], &bncar[s * 4], 4);
  }
  rail_check(carry.v, carry.nv); rail_check(sum.v, sum.nv);
}

// startup screen 1: value-row masks (fastpath marker-containment screen)
static vector<uint32_t> cl_screen_masks(const vector<uint32_t>& maj5_open) {
  vector<uint32_t> cand;
  int sb[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  for (int i = 0; i < 9; i++)
    for (int j = i + 1; j < 9; j++)
      for (int k = j + 1; k < 9; k++) {
        uint32_t m = (1u << sb[i]) | (1u << sb[j]) | (1u << sb[k]);
        if ((m & 384) == 384) continue;
        uint32_t row = CL_SUB + ((CL_Ti0 - CL_SUB) ^ m);
        bool clash = false;  // never place a value row on a lattice-tuple row
        for (uint32_t r : maj5_open) if (row == r) clash = true;
        if (!clash) cand.push_back(m);
      }
  Row marker(2048);
  for (int s = 0; s < 2048; s++) marker[s] = 0xA5A50000u ^ (s * 2654435761u);
  uint8_t mexp[8192];
  for (int s = 0; s < 2048; s++)
    for (int b = 0; b < 4; b++) mexp[s * 4 + b] = (uint8_t)((marker[s] >> (8 * b)) & 0xFF);
  vector<uint32_t> usable;
  for (uint32_t m : cand) {
    uint32_t src = CL_SUB + ((CL_Ti0 - CL_SUB) ^ m);
    pcwrite(src, marker.data());
    bool ok = true;
    for (uint32_t T : {CL_Ti0, CL_Ti1, CL_Ti2}) {
      cl_zero_tuple();
      cl_rowclone(src, T);
      uint8_t buf[8192]; cl_read_row(T, buf);
      int match = 0; for (int b = 0; b < 8192; b++) if (buf[b] == mexp[b]) match++;
      if (match < 8192 - 64) { ok = false; break; }
    }
    if (ok) usable.push_back(m);
  }
  fprintf(stderr, "[lane2] clone mask screen: %zu/%zu usable\n", usable.size(), cand.size());
  return usable;
}

// startup screen 2: op-matched column screen through the COMPLETE 9-gate FA
// (exercises every role row + both rails); AND-ed over patterns and trials.
// A chained depth-2 phase (FA feeding FA, hardware rails carried forward)
// catches lanes that pass single ops but flake under tree depth — the R3
// failure mode (deep count-bit errors on repeat-offender segments).
static vector<uint8_t> cl_op_screen(const vector<uint8_t>& mask, int trials) {
  vector<uint8_t> gm = mask;
  auto mk = [](Node& n, uint32_t pat) {
    n.zero = false; n.v.assign(2048, pat); n.nv.assign(2048, ~pat);
  };
  for (int tr = 0; tr < trials; tr++) {
    vector<array<uint32_t, 3>> abc = {{0xAAAAAAAAu, 0xCCCCCCCCu, 0xF0F0F0F0u},
      {0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu}, {0, 0, 0xFFFFFFFFu},
      {0xFFFFFFFFu, 0, 0}, {0x55555555u, 0xAAAAAAAAu, 0xFFFFFFFFu}};
    srand(24680 + tr);
    for (int r = 0; r < 8; r++) abc.push_back({(uint32_t)(rand() << 16 ^ rand()),
      (uint32_t)(rand() << 16 ^ rand()), (uint32_t)(rand() << 16 ^ rand())});
    for (auto& t : abc) {
      Node a, b, c;
      mk(a, t[0]); mk(b, t[1]); mk(c, t[2]);
      Node s, cy; cl_full_add(a, b, c, s, cy);
      uint32_t ec = maj3w(t[0], t[1], t[2]), es = t[0] ^ t[1] ^ t[2];
      for (int g = 0; g < 2048; g++)
        if (cy.v[g] != ec || cy.nv[g] != ~ec || s.v[g] != es || s.nv[g] != ~es) gm[g] = 0;
    }
    // depth-2 chains: FA(FA(a,b,c).sum, d, e) with the HW rails fed forward
    for (int r = 0; r < 6; r++) {
      uint32_t t0 = (uint32_t)(rand() << 16 ^ rand()), t1 = (uint32_t)(rand() << 16 ^ rand()),
               t2 = (uint32_t)(rand() << 16 ^ rand()), t3 = (uint32_t)(rand() << 16 ^ rand()),
               t4 = (uint32_t)(rand() << 16 ^ rand());
      Node a, b, c, d, e;
      mk(a, t0); mk(b, t1); mk(c, t2); mk(d, t3); mk(e, t4);
      Node s1, c1; cl_full_add(a, b, c, s1, c1);
      Node s2, c2; cl_full_add(s1, d, e, s2, c2);
      uint32_t es1 = t0 ^ t1 ^ t2;
      uint32_t es2 = es1 ^ t3 ^ t4, ec2 = maj3w(es1, t3, t4);
      for (int g = 0; g < 2048; g++) {
        if (!gm[g]) continue;  // judge the chain only on already-good lanes
        if (s2.v[g] != es2 || s2.nv[g] != ~es2 || c2.v[g] != ec2 || c2.nv[g] != ~ec2) gm[g] = 0;
      }
    }
  }
  return gm;
}

// ---------- protocol plumbing (used by both the tree and product paths) ----
static bool read_exact(void* buf, size_t n) {
  size_t got = 0; char* p = (char*)buf;
  while (got < n) { ssize_t r = read(0, p + got, n - got); if (r <= 0) return false; got += (size_t)r; }
  return true;
}
static void write_resp(const void* body, uint32_t len) {
  uint32_t l = len;
  if (write(response_fd, &l, 4) != 4) die("response write failed", 7);
  size_t got = 0; const char* p = (const char*)body;
  while (got < len) { ssize_t r = write(response_fd, p + got, len - got); if (r <= 0) die("response write failed", 7); got += (size_t)r; }
}
static void ack_load(uint32_t handle, uint32_t status) {
  uint32_t b[3] = {MAGIC_LOAD_ACK, handle, status};
  write_resp(b, sizeof b);
}
static void ack_gemv_err(uint32_t handle, uint32_t status) {
  uint32_t b[4] = {MAGIC_GEMV_ACK, handle, status, 0};
  write_resp(b, sizeof b);
}
static double now_s() { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec / 1e9; }

// ---------- Road-B product engine (LANE2_ACCUM, 2026-07-21) ----------
// The build-6 accumulator collapses a row read to one 32-bit popcount total,
// which INVERTS the optimal dataflow: instead of the CSA tree (whose per-FA
// readbacks carry values the host must transport — totals can't replace
// them), the per-output product-row shape becomes optimal:
//   y[m] = Σ_i Σ_c FAC(i,qb)·FAC(c,rb) · popcount( W_i[m] AND x_c )
// — one MAJ3 AND per (m,i,c), product bits placed on screened segments,
// zeros elsewhere, whole-row popcount consumed as the accum total. This is
// MVDRAM's own §V per-output product dataflow (processor aggregates
// popcounts), which the exp0 readout wall had forced the June design away
// from; the FPGA does the "processor" popcount at line rate. Junk model:
// off-segment columns compute MAJ3(0,0,0) — their deviation is measured at
// startup (baseline probes) and shows identically in both arms.
// Order accounting (accum arm): EVERY executed program contains >=1 read,
// so every flush is accum-armed and drains exactly one chunk — the delivered
// stream maps 1:1, in order, onto the executed programs (test_popcount_hw
// build-4 delivery model). x-loads and kickers read the x row, whose
// popcount is known => in-stream integrity sentinels.
static uint32_t XROW = 0;      // resident activation value row (sentinel row)
static vector<int> PSEG;       // AND-screened segments for product bits
static long N_DRAIN_TRY = 0, N_DRAIN_HIT = 0;
// LANE2_WRES=1: W-residency clone loads. LOAD pcwrites as many W bitplane
// rows as fit into screened antichain value rows; GEMV then computes those
// products with a clone-gate (~700 instrs: clone W->Ti1, clone x->Ti2 —
// the fastpath-validated put order) instead of the ~4.4K-instr pcwrite
// gate. Spill (m,i) pairs keep the pcwrite path. This is MVDRAM's own
// resident-weights convention (their per-op numbers assume the matrix
// lives in DRAM); it also measures the per-product floor with operand
// streaming removed — the enabler for plane-packed multi-read totals.
static int WRES = 0;
static vector<uint32_t> WPOOL;           // usable value rows for W residency
static long N_GATE_RES = 0, N_GATE_PCW = 0;
static double T_GATE_RES = 0, T_GATE_PCW = 0;
// LANE2_PLANE_PACK=1 (accum arm only): fold ALL qb plane-gates of one (m,c)
// product into ONE program, reading plane i's product row 2^i times — the
// program's single accum total is then already the plane-weighted partial:
//   T = Σ_{i<qb-1} 2^i·pc(W_i ∧ x)  +  2^(qb-1)·pc(~W_top ∧ x)
// and by the in-extent identity pc(~W∧x) = pc(x) − pc(W∧x) the host applies
//   y[m] += FAC(c,rb) · (T − 2^(qb-1)·pc(x_c))       [pc(x_c) = the xload
// sentinel value]. Program count ÷qb; validates the MULTI-READ accum regime
// (every prior accum program held exactly one read). The negative top plane
// is handled by storing/writing it COMPLEMENTED (resident rows at LOAD,
// spill pcwrites in the packed builder). Outputs whose packed program would
// blow the IMEM budget (any spilled pcwrite plane) fall back to per-plane
// pcwrite gates with plain W and plain FAC math (residency skipped there —
// the resident top row holds ~W and must not enter the unpacked math).
static int PPACK = 0;
static long N_GATE_PACKED = 0;
static double T_GATE_PACKED = 0;

// pcwrite body emitted into an existing program (the PACK 3-chunk shape,
// verbatim instruction sequence — same registers, same timing).
static void pr_emit_pcwrite(Program& p, uint32_t row, const uint32_t* seg) {
  N_PCW++;
  int cs = 0;
  for (int ch = 0; ch < 3; ch++) {
    int n = CHUNK_COLS[ch]; const uint32_t* cd = seg + cs * 16;
    p.add_inst(SMC_LI(row, RAR)); p.add_inst(SMC_LI(cs * 8, CAR));
    p.add_below(PRE(BAR, 0, 0)); p.add_below(ACT(BAR, 0, RAR, 0));
    for (int k = 0; k < n; k++) {
      const uint32_t* sl = cd + k * 16;
      for (int s = 0; s < 16; s++) { p.add_inst(SMC_LI(sl[s], PATTERN_REG)); p.add_inst(SMC_LDWD(PATTERN_REG, s)); }
      p.add_below(WRITE(BAR, CAR, 1)); p.add_inst(SMC_SLEEP(8));
    }
    p.add_inst(SMC_SLEEP(8)); p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_SLEEP(4));
    cs += n;
  }
}

// compare-reference for the accum popcount: ddr_wdata := 0 (must be the LAST
// LDWD load before the read — pcwrite/wrRow stomp the slots).
static void pr_emit_zero_ref(Program& p) {
  p.add_inst(SMC_LI(0, PATTERN_REG));
  for (int i = 0; i < 16; i++) p.add_inst(SMC_LDWD(PATTERN_REG, i));
}

static void pr_exec_checked(Program& p) {
  // p.size() is BYTES at 8 B per u64 instruction (the MM3D dump convention
  // n_inst = size/8). The old `> 8000*16` guard therefore allowed up to
  // 16000 insts against the 8192 IMEM — a latent unit bug nothing hit
  // until PLANE_PACK's spilled builds (2026-07-21: 2x pcwrite sections
  // ~9K insts slipped through and executed SILENTLY TRUNCATED on the
  // FPGA — the deterministic qb2 tail-output corruption). Gate in insts.
  if (p.size() / 8 > 8000) die("product program exceeds the 8K IMEM budget", 5);
  PF->execute(p); N_EXEC++;
}

// One fused product program: AND(W, x) on the fastpath tuple.
// Order: clone XROW->Ti2 FIRST (the only deposit-capable op — everything
// written after it is immune to its deposits), then W->Ti1 (pcwrite, or
// uniform wrRow when wseg==null), z=0 ->Ti0, ONE->Tfr, frac x3,
// doubleACT(0,0) MAJ3, zero compare-ref, rdRow(Ti0).
// out!=null (read arm / screens): receive the full 8 KB row.
// out==null (accum arm): execute only; the caller drains the total.
static void pr_gate(const uint32_t* wseg, uint32_t wpat, uint8_t* out) {
  Program p;
  p.add_inst(SMC_LI(8, CASR)); p.add_inst(SMC_LI(BANK, BAR)); p.add_inst(SMC_LI(128, NUM_COLS_REG));
  p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_SLEEP(6));
  p.add_below(doubleACT(30, 1, XROW, CL_Ti2));            // x -> Ti2 (clone)
  p.add_inst(SMC_SLEEP(6)); p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_SLEEP(6));
  if (wseg) pr_emit_pcwrite(p, CL_Ti1, wseg);             // W -> Ti1
  else {
    p.add_below(wrRow_immediate_label(BAR, CL_Ti1, wpat, LBL++));
    p.add_inst(SMC_SLEEP(6)); p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_SLEEP(6));
  }
  p.add_below(wrRow_immediate_label(BAR, CL_Ti0, 0u, LBL++));   // z = 0
  p.add_inst(SMC_SLEEP(6)); p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_SLEEP(6));
  p.add_below(wrRow_immediate_label(BAR, CL_Tfr, ONE, LBL++));  // reference
  p.add_inst(SMC_SLEEP(6)); p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_SLEEP(6));
  for (int j = 0; j < 3; j++) { p.add_inst(SMC_SLEEP(6)); p.add_below(frac_b(0, CL_Tfr)); p.add_inst(SMC_SLEEP(6)); }
  p.add_inst(SMC_SLEEP(6)); p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_SLEEP(6));
  p.add_below(doubleACT(0, 0, CL_Trf, CL_Trs));           // MAJ3 = AND (z=0)
  p.add_inst(SMC_SLEEP(6)); p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_SLEEP(6));
  pr_emit_zero_ref(p);
  p.add_below(rdRow_immediate_label(BAR, CL_Ti0, LBL++));
  p.add_inst(all_nops()); p.add_inst(all_nops());
  p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_END());
  pr_exec_checked(p); N_MAJ++;
  if (out && PF->receiveData(out, 8192) != 8192)
    die("receiveData failed in product gate — silicon wedge? (RUNBOOK: fpga-helper)", 6);
}

// Resident-W product gate: both operands arrive by RowClone (W from its
// resident value row, x from XROW), in the fastpath-validated put order
// (Ti1 first, Ti2 second, uniform z last — writes after the deposit-ops
// are immune). ~700 instrs vs pr_gate's ~4.7K. Same MAJ3-AND + zero-ref
// + rdRow(Ti0) tail, so totals/rows are directly comparable arm-to-arm.
static void pr_gate_res(uint32_t wrow, uint8_t* out) {
  Program p;
  p.add_inst(SMC_LI(8, CASR)); p.add_inst(SMC_LI(BANK, BAR)); p.add_inst(SMC_LI(128, NUM_COLS_REG));
  p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_SLEEP(6));
  p.add_below(doubleACT(30, 1, wrow, CL_Ti1));            // W -> Ti1 (clone)
  p.add_inst(SMC_SLEEP(6)); p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_SLEEP(6));
  p.add_below(doubleACT(30, 1, XROW, CL_Ti2));            // x -> Ti2 (clone)
  p.add_inst(SMC_SLEEP(6)); p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_SLEEP(6));
  p.add_below(wrRow_immediate_label(BAR, CL_Ti0, 0u, LBL++));   // z = 0
  p.add_inst(SMC_SLEEP(6)); p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_SLEEP(6));
  p.add_below(wrRow_immediate_label(BAR, CL_Tfr, ONE, LBL++));  // reference
  p.add_inst(SMC_SLEEP(6)); p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_SLEEP(6));
  for (int j = 0; j < 3; j++) { p.add_inst(SMC_SLEEP(6)); p.add_below(frac_b(0, CL_Tfr)); p.add_inst(SMC_SLEEP(6)); }
  p.add_inst(SMC_SLEEP(6)); p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_SLEEP(6));
  p.add_below(doubleACT(0, 0, CL_Trf, CL_Trs));           // MAJ3 = AND (z=0)
  p.add_inst(SMC_SLEEP(6)); p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_SLEEP(6));
  pr_emit_zero_ref(p);
  p.add_below(rdRow_immediate_label(BAR, CL_Ti0, LBL++));
  p.add_inst(all_nops()); p.add_inst(all_nops());
  p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_END());
  pr_exec_checked(p); N_MAJ++;
  if (out && PF->receiveData(out, 8192) != 8192)
    die("receiveData failed in resident product gate", 6);
}

// x-load: pcwrite the activation row AND read it back in the SAME program
// (a no-read program's DIFF flush is nondeterministically eaten by the
// maintenance ignore_flush race — a trailing read makes the flush armed and
// the chunk deterministic; its total == popcount(x) => sentinel).
static void pr_xload(const uint32_t* xseg, uint8_t* out) {
  Program p;
  p.add_inst(SMC_LI(8, CASR)); p.add_inst(SMC_LI(BANK, BAR)); p.add_inst(SMC_LI(128, NUM_COLS_REG));
  pr_emit_pcwrite(p, XROW, xseg);
  pr_emit_zero_ref(p);
  p.add_below(rdRow_immediate_label(BAR, XROW, LBL++));
  p.add_inst(all_nops()); p.add_inst(all_nops());
  p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_END());
  pr_exec_checked(p);
  if (out && PF->receiveData(out, 8192) != 8192)
    die("receiveData failed in x-load", 6);
}

// read-only kicker (rdRow of the x row): flushes lagged totals out of the
// c2h path at end-of-stream; its own total is the known x popcount.
static void pr_kick() {
  Program p;
  p.add_inst(SMC_LI(8, CASR)); p.add_inst(SMC_LI(BANK, BAR)); p.add_inst(SMC_LI(128, NUM_COLS_REG));
  p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_SLEEP(6));
  pr_emit_zero_ref(p);
  p.add_below(rdRow_immediate_label(BAR, XROW, LBL++));
  p.add_inst(all_nops()); p.add_inst(all_nops());
  p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_END());
  pr_exec_checked(p);
}

// pull every currently-available 64 B chunk into `stream` (values in
// delivery order; -2 = malformed multi-nonzero chunk). first_tmo bounds the
// first try only; continuation tries are near-nonblocking polls.
static int pr_drain_avail(vector<long>& stream, long first_tmo) {
  int got = 0; bool first = true;
  for (int guard = 0; guard < 4096; guard++) {
    uint8_t a[64]; memset(a, 0xFF, sizeof a);
    N_DRAIN_TRY++;
    int rx = PF->receiveDataTry(a, 64, first ? first_tmo : 0);
    first = false;
    if (rx != 64 || PF->recv_stalled()) break;
    N_DRAIN_HIT++;
    uint32_t v = 0; int nz = 0;
    for (int j = 0; j < 64; j += 4) { uint32_t w; memcpy(&w, a + j, 4);
      if (w && w != 0xFFFFFFFFu) { v = w; nz++; } }
    stream.push_back(nz <= 1 ? (long)v : -2);
    got++;
  }
  return got;
}

static long pr_popcount_row(const uint8_t* rowb) {
  long pc = 0;
  for (int s = 0; s < 2048; s++) { uint32_t w; memcpy(&w, rowb + s * 4, 4); pc += __builtin_popcount(w); }
  return pc;
}

// place a wire bitplane row's K bits onto the PSEG segments (zeros elsewhere)
static void pr_place_bits(const uint8_t* wr, uint32_t K, Row& out) {
  fill(out.begin(), out.end(), 0u);
  for (uint32_t n = 0; n < K; n++)
    if ((wr[n >> 3] >> (n & 7)) & 1) out[PSEG[n >> 5]] |= 1u << (n & 31);
}

// as pr_place_bits but with each of the K in-extent bits FLIPPED (~W; zeros
// stay everywhere else) — the negative-top-plane pattern for PLANE_PACK.
static void pr_place_bits_c(const uint8_t* wr, uint32_t K, Row& out) {
  fill(out.begin(), out.end(), 0u);
  for (uint32_t n = 0; n < K; n++)
    if (!((wr[n >> 3] >> (n & 7)) & 1)) out[PSEG[n >> 5]] |= 1u << (n & 31);
}

// PLANE_PACK gate: all qb plane sections of one (m,c) product in ONE
// program. Section i is the pr_gate_res / pr_gate body VERBATIM (same put
// order, sleeps, PREs): resident planes clone W from their value row
// (Ti1 first, x second); spilled planes clone x first then pcwrite W (the
// pr_gate order — writes after the deposit-capable clone are immune). Each
// section ends with its own zero compare-ref (the section's wrRows stomp
// the LDWD slots) and 2^i back-to-back rdRows of the product row; the
// accum engine sums every read in the program into ONE total. The top
// plane (qb>1) uses ~W: resident top rows are LOADed complemented,
// spilled top planes are placed complemented here.
// Build-only until the size check: returns false (silicon untouched) if
// the packed program exceeds the IMEM budget — caller falls back.
static bool pr_gate_packed(Matrix& mat, uint32_t m, Row& wseg) {
  size_t bpr = ((size_t)mat.K + 7) / 8;
  Program p;
  p.add_inst(SMC_LI(8, CASR)); p.add_inst(SMC_LI(BANK, BAR)); p.add_inst(SMC_LI(128, NUM_COLS_REG));
  for (uint32_t i = 0; i < mat.qb; i++) {
    bool top = (mat.qb > 1 && i == mat.qb - 1);
    uint32_t wrow_res = mat.wres_row.empty()
        ? 0u : mat.wres_row[(size_t)i * mat.M + m];
    p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_SLEEP(6));
    if (wrow_res) {                                        // resident plane
      p.add_below(doubleACT(30, 1, wrow_res, CL_Ti1));     // W -> Ti1 (clone)
      p.add_inst(SMC_SLEEP(6)); p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_SLEEP(6));
      p.add_below(doubleACT(30, 1, XROW, CL_Ti2));         // x -> Ti2 (clone)
      p.add_inst(SMC_SLEEP(6)); p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_SLEEP(6));
    } else {                                               // spilled plane
      p.add_below(doubleACT(30, 1, XROW, CL_Ti2));         // x -> Ti2 (clone)
      p.add_inst(SMC_SLEEP(6)); p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_SLEEP(6));
      const uint8_t* wr = mat.plane.data() + ((size_t)i * mat.M + m) * bpr;
      if (top) pr_place_bits_c(wr, mat.K, wseg);
      else     pr_place_bits(wr, mat.K, wseg);
      pr_emit_pcwrite(p, CL_Ti1, wseg.data());             // W -> Ti1
    }
    p.add_below(wrRow_immediate_label(BAR, CL_Ti0, 0u, LBL++));   // z = 0
    p.add_inst(SMC_SLEEP(6)); p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_SLEEP(6));
    p.add_below(wrRow_immediate_label(BAR, CL_Tfr, ONE, LBL++));  // reference
    p.add_inst(SMC_SLEEP(6)); p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_SLEEP(6));
    for (int j = 0; j < 3; j++) { p.add_inst(SMC_SLEEP(6)); p.add_below(frac_b(0, CL_Tfr)); p.add_inst(SMC_SLEEP(6)); }
    p.add_inst(SMC_SLEEP(6)); p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_SLEEP(6));
    p.add_below(doubleACT(0, 0, CL_Trf, CL_Trs));          // MAJ3 = AND (z=0)
    p.add_inst(SMC_SLEEP(6)); p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_SLEEP(6));
    pr_emit_zero_ref(p);
    for (uint32_t rr = 0; rr < (1u << i); rr++) {          // 2^i replicated reads
      p.add_below(rdRow_immediate_label(BAR, CL_Ti0, LBL++));
      p.add_inst(all_nops()); p.add_inst(all_nops());
    }
  }
  p.add_below(PRE(BAR, 0, 0)); p.add_inst(SMC_END());
  if (p.size() / 8 > 7800) return false;   // insts; margin under pr_exec_checked
  pr_exec_checked(p); N_MAJ += mat.qb;
  return true;
}

// startup: pick XROW, AND-screen segments through the real product gate,
// junk baseline + read-stability diagnostics. READ mode throughout.
static void pr_startup(const vector<uint8_t>& mask, int trials) {
  double ts = now_s();
  // 1. XROW: an antichain-mask value row (clone-mode rule: 3 bits of 1..9,
  //    never bits 7+8 together, no tuple clash) that (a) reads back a
  //    pcwritten marker EXACTLY (sentinel requirement) and (b) clones into
  //    Ti2 intact (<=64 flaked bytes, the fastpath screen tolerance).
  Row marker(2048);
  for (int s = 0; s < 2048; s++) marker[s] = 0xA5A50000u ^ (s * 2654435761u);
  uint8_t mexp[8192];
  for (int s = 0; s < 2048; s++)
    for (int b = 0; b < 4; b++) mexp[s * 4 + b] = (uint8_t)((marker[s] >> (8 * b)) & 0xFF);
  int sb[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  bool found = false;
  for (int i = 0; i < 9 && !found; i++)
    for (int j = i + 1; j < 9 && !found; j++)
      for (int k = j + 1; k < 9 && !found; k++) {
        uint32_t m = (1u << sb[i]) | (1u << sb[j]) | (1u << sb[k]);
        if ((m & 384) == 384) continue;
        uint32_t row = CL_SUB + ((CL_Ti0 - CL_SUB) ^ m);
        bool clash = false;
        for (uint32_t r : CAL.open) if (row == r) clash = true;
        for (uint32_t r : {CL_Ti0, CL_Ti1, CL_Ti2, CL_Tfr}) if (row == r) clash = true;
        if (clash) continue;
        pcwrite(row, marker.data());
        uint8_t buf[8192]; cl_read_row(row, buf);
        if (memcmp(buf, mexp, 8192) != 0) continue;      // sentinel needs exact
        cl_zero_tuple();
        cl_rowclone(row, CL_Ti2);
        cl_read_row(CL_Ti2, buf);
        int match = 0; for (int b = 0; b < 8192; b++) if (buf[b] == mexp[b]) match++;
        if (match < 8192 - 64) continue;
        XROW = row; found = true;
        fprintf(stderr, "[lane2] ACCUM x-row: local offset %u (mask 0x%03x), "
                "direct r/w exact, clone->Ti2 %d/8192\n", row - CL_SUB, m, match);
      }
  if (!found) die("ACCUM: no usable x value row (sentinel-grade)", 4);

  // 2. AND-screen: uniform W/x pairs through the REAL product path (x is
  //    cloned from XROW; W uniform-written). Expected AND = P & Q on every
  //    segment; AND-ed over patterns and trials.
  vector<uint8_t> gm = mask;
  uint8_t rowb[8192];
  for (int tr = 0; tr < trials; tr++) {
    vector<pair<uint32_t, uint32_t>> pq = {{0, 0}, {ONE, ONE}, {ONE, 0}, {0, ONE},
      {0xAAAAAAAAu, 0xCCCCCCCCu}, {0xA5A5A5A5u, 0x5A5A5A5Au}, {0xEEEEEEEEu, 0xFFFF0000u}};
    srand(31415 + tr);
    for (int r = 0; r < 12; r++) pq.push_back({(uint32_t)(rand() << 16 ^ rand()), (uint32_t)(rand() << 16 ^ rand())});
    for (auto& t : pq) {
      uwrite_batch({{XROW, t.second}});
      pr_gate(nullptr, t.first, rowb);
      uint32_t expect = t.first & t.second;
      for (int s = 0; s < 2048; s++) {
        uint32_t w; memcpy(&w, rowb + s * 4, 4);
        if (w != expect) gm[s] = 0;
      }
    }
  }
  PSEG.clear();
  int n_in = 0; for (auto v : mask) n_in += v;
  for (int s = 0; s < 2048; s++) if (gm[s]) PSEG.push_back(s);
  fprintf(stderr, "[lane2] ACCUM AND-screen (%d trials): %zu/%d segments -> "
          "K capacity %zu product bits\n", trials, PSEG.size(), n_in, PSEG.size() * 32);
  if (PSEG.empty()) die("ACCUM: no reliable AND segments", 4);

  // 3. Junk baseline: all-zero products (W=0, x=0) — off-segment MAJ3(0,0,0)
  //    deviation IS the accum-total junk floor. Report bits + stability.
  uwrite_batch({{XROW, 0u}});
  long jmin = LONG_MAX, jmax = 0, jsum = 0; int jn = 6;
  for (int r = 0; r < jn; r++) {
    pr_gate(nullptr, 0u, rowb);
    long pc = pr_popcount_row(rowb);
    jsum += pc; if (pc < jmin) jmin = pc; if (pc > jmax) jmax = pc;
  }
  fprintf(stderr, "[lane2] ACCUM zero-product junk baseline: min/mean/max = "
          "%ld/%.1f/%ld set bits per row (of 65536; 0 = clean)\n",
          jmin, (double)jsum / jn, jmax);

  // 4. End-to-end data-path probe: marker W on PSEG via pcwrite, x = all-ones
  //    on PSEG via pcwrite — product must equal W on PSEG, 0 elsewhere; plus
  //    a same-row repeat read for read-stability.
  Row wseg(2048, 0), xseg(2048, 0);
  int nprobe = (int)PSEG.size(); if (nprobe > 128) nprobe = 128;
  for (int t = 0; t < nprobe; t++) { wseg[PSEG[t]] = marker[t]; xseg[PSEG[t]] = ONE; }
  pr_xload(xseg.data(), rowb);
  int xbad = 0;
  for (int s = 0; s < 2048; s++) { uint32_t w; memcpy(&w, rowb + s * 4, 4); if (w != xseg[s]) xbad++; }
  pr_gate(wseg.data(), 0u, rowb);
  long on_bad = 0, off_bits = 0;
  for (int s = 0; s < 2048; s++) {
    uint32_t w; memcpy(&w, rowb + s * 4, 4);
    bool on = false; for (int t = 0; t < nprobe; t++) if (PSEG[t] == s) { on = true; if (w != wseg[s]) on_bad++; break; }
    if (!on) off_bits += __builtin_popcount(w);
  }
  long pc1 = pr_popcount_row(rowb);
  uint8_t rowb2[8192]; cl_read_row(CL_Ti0, rowb2);
  long pc2 = pr_popcount_row(rowb2);
  fprintf(stderr, "[lane2] ACCUM data-path probe: xload %d/2048 words off; "
          "product on-segment %ld words wrong, off-segment %ld junk bits; "
          "repeat-read popcount %ld -> %ld (delta %ld)\n",
          xbad, on_bad, off_bits, pc1, pc2, pc2 - pc1);
  // 5. LANE2_WRES: collect the remaining usable value rows as the W
  //    residency pool — same antichain-mask rule and marker screen as the
  //    XROW pick (direct r/w exact + clone-to-Ti1 intact, since W clones
  //    into Ti1).
  if (WRES) {
    for (int i = 0; i < 9; i++)
      for (int j = i + 1; j < 9; j++)
        for (int k = j + 1; k < 9; k++) {
          uint32_t m = (1u << sb[i]) | (1u << sb[j]) | (1u << sb[k]);
          if ((m & 384) == 384) continue;
          uint32_t row = CL_SUB + ((CL_Ti0 - CL_SUB) ^ m);
          if (row == XROW) continue;
          bool clash = false;
          for (uint32_t r : CAL.open) if (row == r) clash = true;
          for (uint32_t r : {CL_Ti0, CL_Ti1, CL_Ti2, CL_Tfr}) if (row == r) clash = true;
          if (clash) continue;
          pcwrite(row, marker.data());
          uint8_t buf[8192]; cl_read_row(row, buf);
          if (memcmp(buf, mexp, 8192) != 0) continue;
          cl_zero_tuple();
          cl_rowclone(row, CL_Ti1);
          cl_read_row(CL_Ti1, buf);
          int match = 0; for (int b = 0; b < 8192; b++) if (buf[b] == mexp[b]) match++;
          if (match < 8192 - 64) continue;
          WPOOL.push_back(row);
        }
    fprintf(stderr, "[lane2] WRES pool: %zu resident W rows (marker + clone-to-Ti1 screened)\n",
            WPOOL.size());
  }
  fprintf(stderr, "[lane2] ACCUM startup done in %.1f s (%ld execs)\n",
          now_s() - ts, N_EXEC);
}

// the ACCUM-mode GEMV: identical program stream in both arms; readout via
// full rows + host popcount (ACCUM=1) or batched accum totals (ACCUM=2).
struct PrRec { uint8_t kind; uint32_t m, i, c; long expect; };  // kind: 0=product 1=sentinel

static void handle_gemv_accum(uint32_t handle, Matrix& mat, uint32_t rb, const uint8_t* xp) {
  const bool RD = (ACCUM == 1);
  size_t bpr = (mat.K + 7) / 8;
  long xrefresh = 64;
  if (const char* e = getenv("LANE2_XREFRESH")) xrefresh = atol(e);
  if (xrefresh < 8) xrefresh = 8;

  double t0 = now_s();
  long exec0 = N_EXEC, pcw0 = N_PCW, try0 = N_DRAIN_TRY, hit0 = N_DRAIN_HIT;
  long skips0 = PF->oversize_skips();
  vector<long long> y64(mat.M, 0);
  vector<PrRec> recs; recs.reserve((size_t)rb * mat.qb * mat.M + 64);
  vector<long> stream;
  vector<long> rd_pc; if (RD) rd_pc.reserve((size_t)rb * mat.qb * mat.M + 64);
  vector<uint8_t> rowbuf(8192);
  Row xrow(2048), wseg(2048);
  bool fail = false;
  long planes_skipped = 0, xload_badw = 0;

  if (!RD) { PF->set_readback_mode(true); PF->set_readback_mode(true); }

  long since_x = 0;
  for (uint32_t c = 0; c < rb && !fail; c++) {
    const uint8_t* plane = xp + (size_t)c * bpr;
    fill(xrow.begin(), xrow.end(), 0u);
    long xpc = 0;
    for (uint32_t n = 0; n < mat.K; n++)
      if ((plane[n >> 3] >> (n & 7)) & 1) { xrow[PSEG[n >> 5]] |= 1u << (n & 31); xpc++; }
    if (xpc == 0) {  // whole plane zero => every product is zero: skip
      planes_skipped++;
      continue;
    }
    auto do_xload = [&]() {
      pr_xload(xrow.data(), RD ? rowbuf.data() : nullptr);
      recs.push_back({1, 0, 0, c, xpc});
      if (RD) {
        rd_pc.push_back(pr_popcount_row(rowbuf.data()));
        for (int s = 0; s < 2048; s++) { uint32_t w; memcpy(&w, rowbuf.data() + s * 4, 4); if (w != xrow[s]) xload_badw++; }
      } else pr_drain_avail(stream, 2);
      since_x = 0;
    };
    do_xload();
    // free-poll + windowed backpressure after every executed accum program
    // (identical logic for packed and per-plane programs: 1 total each).
    auto after_accum_prog = [&]() {
      pr_drain_avail(stream, 0);   // free poll: pop whatever surfaced
      if ((long)recs.size() - (long)stream.size() > 96) {
        // stream fell behind the window: block until it catches up
        int spins = 0;
        while ((long)stream.size() < (long)recs.size() - 8 &&
               !PF->recv_stalled() && spins < 40)
          if (!pr_drain_avail(stream, 300)) spins++;
        if (PF->recv_stalled() || (long)recs.size() - (long)stream.size() > 96) {
          fprintf(stderr, "[lane2] ACCUM: totals stream fell behind (exec %zu got %zu)%s\n",
                  recs.size(), stream.size(), PF->recv_stalled() ? " POISONED" : "");
          fail = true;
        }
      }
    };
    if (PPACK && !RD) {
      // plane-packed: one program per output m covers all qb planes.
      for (uint32_t m = 0; m < mat.M && !fail; m++) {
        if (since_x >= xrefresh) do_xload();
        double tg0 = now_s();
        if (pr_gate_packed(mat, m, wseg)) {
          N_GATE_PACKED++; T_GATE_PACKED += now_s() - tg0;
          recs.push_back({2, m, 0, c, xpc});   // expect carries pc(x_c)
          since_x += mat.qb;
          after_accum_prog();
        } else {
          // packed program over IMEM budget (spilled pcwrite planes):
          // per-plane pcwrite gates, plain W, plain FAC math. Residency is
          // deliberately skipped here — the resident top row holds ~W.
          for (uint32_t i = 0; i < mat.qb && !fail; i++) {
            if (since_x >= xrefresh) do_xload();
            const uint8_t* wr = mat.plane.data() + ((size_t)i * mat.M + m) * bpr;
            pr_place_bits(wr, mat.K, wseg);
            double tp0 = now_s();
            pr_gate(wseg.data(), 0u, nullptr);
            N_GATE_PCW++; T_GATE_PCW += now_s() - tp0;
            recs.push_back({0, m, i, c, -1});
            since_x++;
            after_accum_prog();
          }
        }
      }
    } else {
    for (uint32_t i = 0; i < mat.qb && !fail; i++) {
      long long fac = (long long)FAC((int)i, (int)mat.qb) * FAC((int)c, (int)rb);
      for (uint32_t m = 0; m < mat.M && !fail; m++) {
        if (since_x >= xrefresh) do_xload();
        uint32_t wrow_res = mat.wres_row.empty()
            ? 0u : mat.wres_row[(size_t)i * mat.M + m];
        double tg0 = now_s();
        if (wrow_res) {
          pr_gate_res(wrow_res, RD ? rowbuf.data() : nullptr);
          N_GATE_RES++; T_GATE_RES += now_s() - tg0;
        } else {
          const uint8_t* wr = mat.plane.data() + ((size_t)i * mat.M + m) * bpr;
          pr_place_bits(wr, mat.K, wseg);
          pr_gate(wseg.data(), 0u, RD ? rowbuf.data() : nullptr);
          N_GATE_PCW++; T_GATE_PCW += now_s() - tg0;
        }
        recs.push_back({0, m, i, c, -1});
        since_x++;
        if (RD) {
          long pc = pr_popcount_row(rowbuf.data());
          rd_pc.push_back(pc);
          y64[m] += fac * pc;
        } else {
          after_accum_prog();
        }
      }
    }
    }
  }

  long sent_mis = 0, malformed = 0;
  if (!RD) {
    if (!fail && !recs.empty()) {
      // end-of-stream: messages only surface into api_recv_buf through an
      // execute's receive window, so fire kickers (read-only programs, NOT
      // part of the parity set) until the REAL programs' totals all arrive.
      // Kicker chunks are surplus behind the real prefix (FIFO order) and
      // are cleared by the exit drain — the suite's over-provisioning shape.
      int kicks = 0;
      pr_drain_avail(stream, 0);
      while (stream.size() < recs.size() && kicks < 12 && !PF->recv_stalled()) {
        pr_kick(); kicks++;
        pr_drain_avail(stream, 300);
      }
      if (stream.size() < recs.size()) {
        fprintf(stderr, "[lane2] ACCUM: stream short after %d kickers: %zu/%zu\n",
                kicks, stream.size(), recs.size());
        fail = true;
      }
    }
    // exit hygiene (suite-proven shape): clear surfaced strays, SET-READ x2,
    // one no-read write to run the platform's transition drain, drain again.
    // The pre-switch pass is harvest-only (100 ms): anything still stranded
    // is exactly what the transition drain exists to absorb; a straggler
    // that somehow leaked past both would surface as a position-0 sentinel
    // mismatch in the NEXT accum GEMV (fail-safe, not silent).
    PF->drain_stray(100, 4);
    PF->set_readback_mode(false); PF->set_readback_mode(false);
    uwrite_batch({{CL_Tfr, ONE}});
    PF->drain_stray(1500, 8);
    if (!fail) {
      for (size_t k = 0; k < recs.size(); k++) {
        long v = stream[k];
        if (v < 0) { malformed++; fail = true; continue; }
        if (recs[k].kind == 0)
          y64[recs[k].m] += (long long)FAC((int)recs[k].i, (int)mat.qb) * FAC((int)recs[k].c, (int)rb) * v;
        else if (recs[k].kind == 2) {
          // packed product: total is the plane-weighted partial with the
          // top plane complemented; expect carries pc(x_c).
          long long topc = mat.qb > 1 ? (1LL << (mat.qb - 1)) : 0LL;
          y64[recs[k].m] += (long long)FAC((int)recs[k].c, (int)rb)
                            * ((long long)v - topc * (long long)recs[k].expect);
        }
        else if (v != recs[k].expect) {
          sent_mis++;
          fprintf(stderr, "[lane2] ACCUM sentinel MISMATCH at prog %zu: total %ld expect %ld\n",
                  k, v, recs[k].expect);
        }
      }
      if (sent_mis) fail = true;   // order/integrity broken: refuse the result
    }
    // any platform IMEM-gate skip means a program never ran and the kicker
    // backfill silently stood in for its total (the 2026-07-21 PLANE_PACK
    // spill incident: every skipped output got y = -pc(x), deterministic
    // and sentinel-clean). Refuse the result outright.
    if (PF->oversize_skips() != skips0) {
      fprintf(stderr, "[lane2] ACCUM: %ld program(s) refused by the platform "
              "IMEM gate during this GEMV — totals stream is backfilled, "
              "refusing result\n", PF->oversize_skips() - skips0);
      fail = true;
    }
  }

  double wall = now_s() - t0;
  long nprod = 0, nsent = 0;
  for (auto& r : recs) (r.kind == 1 ? nsent : nprod)++;
  fprintf(stderr, "[lane2] GEMV[%s-arm] handle=%u K=%u M=%u q=%u r=%u: %.2f s wall, "
          "%ld products + %ld sentinels (%ld execs, %ld pcwrites), %ld zero planes skipped",
          RD ? "read" : "accum", handle, mat.K, mat.M, mat.qb, rb, wall,
          nprod, nsent, N_EXEC - exec0, N_PCW - pcw0, planes_skipped);
  if (RD) fprintf(stderr, ", xload %ld words off\n", xload_badw);
  else fprintf(stderr, ", drains %ld/%ld hit, sentinel mis %ld, malformed %ld\n",
               N_DRAIN_HIT - hit0, N_DRAIN_TRY - try0, sent_mis, malformed);
  if (N_GATE_RES + N_GATE_PCW > 0)
    fprintf(stderr, "[lane2]   gates: %ld resident (%.0f us/gate) + %ld pcwrite "
            "(%.0f us/gate)\n",
            N_GATE_RES, N_GATE_RES ? 1e6 * T_GATE_RES / N_GATE_RES : 0.0,
            N_GATE_PCW, N_GATE_PCW ? 1e6 * T_GATE_PCW / N_GATE_PCW : 0.0);
  if (N_GATE_PACKED > 0)
    fprintf(stderr, "[lane2]   packed: %ld programs (%.0f us/program = %.0f us "
            "per plane-gate at qb=%u)\n",
            N_GATE_PACKED, 1e6 * T_GATE_PACKED / N_GATE_PACKED,
            1e6 * T_GATE_PACKED / N_GATE_PACKED / mat.qb, mat.qb);

  if (fail) { ack_gemv_err(handle, 2); return; }
  vector<uint8_t> resp(16 + (size_t)mat.M * 4);
  uint32_t hdr[4] = {MAGIC_GEMV_ACK, handle, 0, mat.M};
  memcpy(resp.data(), hdr, 16);
  for (uint32_t m = 0; m < mat.M; m++) {
    long long v = y64[m];
    if (v > 0x7FFFFFFFLL || v < -0x80000000LL) { fprintf(stderr, "[lane2] y overflow m=%u\n", m); ack_gemv_err(handle, 2); return; }
    int32_t v32 = (int32_t)v;
    memcpy(resp.data() + 16 + (size_t)m * 4, &v32, 4);
  }
  write_resp(resp.data(), (uint32_t)resp.size());
}

static void handle_load(const uint8_t* req, size_t len) {
  if (len < 20) { ack_load(0, 1); return; }
  uint32_t handle, qb, K, M;
  memcpy(&handle, req + 4, 4); memcpy(&qb, req + 8, 4);
  memcpy(&K, req + 12, 4); memcpy(&M, req + 16, 4);
  if (qb < 1 || qb > 4 || K < 1 || K > 16384 || M < 1 || M > 65536) { ack_load(handle, 5); return; }
  size_t bpr = (K + 7) / 8;
  if (len != 20 + (size_t)qb * M * bpr) {
    fprintf(stderr, "[lane2] LOAD len mismatch: got %zu want %zu\n", len, 20 + (size_t)qb * M * bpr);
    ack_load(handle, 1); return;
  }
  if (ACCUM) {
    // product mode: outputs are sequential in time, not parallel in columns
    // — capacity is K product bits on the AND-screened segments.
    size_t need_segs = ((size_t)K + 31) / 32;
    if (need_segs > PSEG.size()) {
      fprintf(stderr, "[lane2] LOAD: K=%u needs %zu AND-screened segments, have %zu\n",
              K, need_segs, PSEG.size());
      ack_load(handle, 4); return;
    }
    Matrix mat; mat.qb = qb; mat.K = K; mat.M = M;
    mat.plane.assign(req + 20, req + 20 + (size_t)qb * M * bpr);
    if (WRES && !WPOOL.empty()) {
      // Fill the residency pool: complete qb-plane sets for the first
      // M_res outputs (a partial set would split one output across paths
      // for no reason). One pcwrite per resident row, ONCE per LOAD.
      double t0 = now_s();
      mat.wres_row.assign((size_t)qb * M, 0);
      size_t M_res = WPOOL.size() / qb; if (M_res > M) M_res = M;
      Row wseg(2048);
      size_t r = 0;
      for (size_t m = 0; m < M_res; m++)
        for (uint32_t i = 0; i < qb; i++) {
          const uint8_t* wr = mat.plane.data() + ((size_t)i * M + m) * bpr;
          // PLANE_PACK: the negative top plane lives resident as ~W (the
          // packed total consumes it via pc(~W∧x) = pc(x) − pc(W∧x)).
          if (PPACK && qb > 1 && i == qb - 1) pr_place_bits_c(wr, K, wseg);
          else                                pr_place_bits(wr, K, wseg);
          pcwrite(WPOOL[r], wseg.data());
          mat.wres_row[(size_t)i * M + m] = WPOOL[r];
          r++;
        }
      fprintf(stderr, "[lane2] WRES: %zu W rows resident (outputs 0..%zu x qb=%u%s) "
              "in %.2f s; %zu pool rows unused\n",
              r, M_res ? M_res - 1 : 0, qb,
              (PPACK && qb > 1) ? ", top plane stored ~W for PLANE_PACK" : "",
              now_s() - t0, WPOOL.size() - r);
    }
    MATS[handle] = std::move(mat);
    fprintf(stderr, "[lane2] LOAD handle=%u q=%u K=%u M=%u (ACCUM product mode: "
            "raw planes host-resident, W rows built per product)\n", handle, qb, K, M);
    ack_load(handle, 0); return;
  }
  size_t need_segs = (M + 31) / 32;
  if (need_segs > LANES.size()) {
    fprintf(stderr, "[lane2] LOAD: M=%u needs %zu screened segments, have %zu\n", M, need_segs, LANES.size());
    ack_load(handle, 4); return;
  }
  double t0 = now_s();
  Matrix mat; mat.qb = qb; mat.K = K; mat.M = M;
  mat.wrow.assign(K, vector<Row>(qb, Row(2048, 0)));
  const uint8_t* pl = req + 20;
  for (uint32_t i = 0; i < qb; i++)
    for (uint32_t m = 0; m < M; m++) {
      const uint8_t* rowb = pl + ((size_t)i * M + m) * bpr;
      int seg = LANES[m >> 5]; uint32_t bit = 1u << (m & 31);
      for (uint32_t n = 0; n < K; n++)
        if ((rowb[n >> 3] >> (n & 7)) & 1) mat.wrow[n][i][seg] |= bit;
    }
  if (DT || ENC_CLONE) {
    // Fig-15 inverted matrix rows: complements prepared at LOAD time (as the
    // paper's host writes inverted rows at matrix load), doubling the matrix
    // bitplane storage. No NOT ever happens at GeMV time.
    mat.nwrow.assign(K, vector<Row>(qb, Row(2048, 0)));
    for (uint32_t n = 0; n < K; n++)
      for (uint32_t i = 0; i < qb; i++)
        for (int s = 0; s < 2048; s++) mat.nwrow[n][i][s] = ~mat.wrow[n][i][s];
  }
  MATS[handle] = std::move(mat);
  fprintf(stderr, "[lane2] LOAD handle=%u q=%u K=%u M=%u -> %u x %u bitplane rows%s "
          "(host-resident; streamed into the tuple per-op — see header) in %.2f s\n",
          handle, qb, K, M, K, qb,
          (DT || ENC_CLONE) ? " (+inverted planes, Fig-15 dual-track)" : "",
          now_s() - t0);
  ack_load(handle, 0);
}

static void handle_gemv(const uint8_t* req, size_t len) {
  if (len < 12) { ack_gemv_err(0, 1); return; }
  uint32_t handle, rb;
  memcpy(&handle, req + 4, 4); memcpy(&rb, req + 8, 4);
  auto it = MATS.find(handle);
  if (it == MATS.end()) { ack_gemv_err(handle, 3); return; }
  Matrix& mat = it->second;
  if (rb < 1 || rb > 8) { ack_gemv_err(handle, 5); return; }
  size_t bpr = (mat.K + 7) / 8;
  if (len != 12 + (size_t)rb * bpr) { ack_gemv_err(handle, 1); return; }
  const uint8_t* xp = req + 12;
  if (ACCUM) { handle_gemv_accum(handle, mat, rb, xp); return; }

  double t0 = now_s();
  long exec0 = N_EXEC, maj0 = N_MAJ, fa0 = N_FA, pcw0 = N_PCW;
  vector<long long> y64(mat.M, 0);
  long skipped = 0;
  for (uint32_t i = 0; i < mat.qb; i++)
    for (uint32_t c = 0; c < rb; c++) {
      // SV-D zero-skip: only taps with activation bit c set enter the tree
      vector<Node> inp;
      for (uint32_t n = 0; n < mat.K; n++) {
        const uint8_t* plane = xp + (size_t)c * bpr;
        if ((plane[n >> 3] >> (n & 7)) & 1) {
          Node d; d.zero = false; d.v = mat.wrow[n][i];
          if (DT || ENC_CLONE) d.nv = mat.nwrow[n][i];
          inp.push_back(std::move(d));
        } else if (i == 0) skipped++;
      }
      long long fac = (long long)FAC((int)i, (int)mat.qb) * FAC((int)c, (int)rb);
      if (inp.empty()) continue;
      double tp0 = now_s();
      vector<Row> cnt;
      if (inp.size() == 1) {
        // single selected tap: count == the (host-known) leaf itself; no adds
        // to run in-DRAM. (Degenerate case of the host-resolved encoding.)
        cnt.push_back(inp[0].v);
      } else {
        cnt = popcount_tree(std::move(inp));
      }
      for (uint32_t m = 0; m < mat.M; m++) {
        int seg = LANES[m >> 5]; int b = m & 31;
        long long ps = 0;
        for (size_t w = 0; w < cnt.size(); w++) ps += (long long)((cnt[w][seg] >> b) & 1) << w;
        y64[m] += fac * ps;
      }
      fprintf(stderr, "[lane2]   plane i=%u c=%u done (%.1f s, FA=%ld MAJ=%ld)\n",
              i, c, now_s() - tp0, N_FA - fa0, N_MAJ - maj0);
    }
  double wall = now_s() - t0;
  fprintf(stderr, "[lane2] GEMV handle=%u K=%u M=%u q=%u r=%u: %.2f s wall, "
          "%ld FA, %ld MAJ, %ld execs, %ld pcwrites, %ld taps zero-skipped\n",
          handle, mat.K, mat.M, mat.qb, rb, wall,
          N_FA - fa0, N_MAJ - maj0, N_EXEC - exec0, N_PCW - pcw0, skipped);
  if (DT || ENC_CLONE)
    fprintf(stderr, "[lane2]   dual-rail consistency: %ld violations / %ld lane-checks\n",
            N_RAIL_VIOL, N_RAIL_LANES);

  vector<uint8_t> resp(16 + (size_t)mat.M * 4);
  uint32_t hdr[4] = {MAGIC_GEMV_ACK, handle, 0, mat.M};
  memcpy(resp.data(), hdr, 16);
  for (uint32_t m = 0; m < mat.M; m++) {
    long long v = y64[m];
    if (v > 0x7FFFFFFFLL || v < -0x80000000LL) { fprintf(stderr, "[lane2] y overflow m=%u\n", m); ack_gemv_err(handle, 2); return; }
    int32_t v32 = (int32_t)v;
    memcpy(resp.data() + 16 + (size_t)m * 4, &v32, 4);
  }
  write_resp(resp.data(), (uint32_t)resp.size());
}

// GEMV_PARTIALS: identical contraction, but the CSA tree runs PER 32-WEIGHT
// BLOCK and the exact integer per-block partial sums are returned instead of
// the whole-K dot (the paper's per-subarray partial sums, SII-C2/SVII, at
// q4_0/q8_0 block granularity — processor does the weighted aggregation).
static void ack_part_err(uint32_t handle, uint32_t status) {
  uint32_t b[5] = {MAGIC_PART_ACK, handle, status, 0, 0};
  write_resp(b, sizeof b);
}
static void handle_partials(const uint8_t* req, size_t len) {
  if (len < 12) { ack_part_err(0, 1); return; }
  uint32_t handle, rb;
  memcpy(&handle, req + 4, 4); memcpy(&rb, req + 8, 4);
  auto it = MATS.find(handle);
  if (it == MATS.end()) { ack_part_err(handle, 3); return; }
  Matrix& mat = it->second;
  if (rb < 1 || rb > 8) { ack_part_err(handle, 5); return; }
  size_t bpr = (mat.K + 7) / 8;
  if (len != 12 + (size_t)rb * bpr) { ack_part_err(handle, 1); return; }
  const uint8_t* xp = req + 12;
  uint32_t NBLK = (mat.K + 31) / 32;

  double t0 = now_s();
  long exec0 = N_EXEC, maj0 = N_MAJ, fa0 = N_FA, pcw0 = N_PCW;
  vector<long long> p64((size_t)mat.M * NBLK, 0);
  long skipped = 0;
  for (uint32_t i = 0; i < mat.qb; i++)
    for (uint32_t c = 0; c < rb; c++) {
      const uint8_t* plane = xp + (size_t)c * bpr;
      long long fac = (long long)FAC((int)i, (int)mat.qb) * FAC((int)c, (int)rb);
      double tp0 = now_s();
      long fap0 = N_FA;
      for (uint32_t b = 0; b < NBLK; b++) {
        uint32_t n0 = b * 32, n1 = (n0 + 32 < mat.K) ? n0 + 32 : mat.K;
        vector<Node> inp;
        for (uint32_t n = n0; n < n1; n++) {
          if ((plane[n >> 3] >> (n & 7)) & 1) {
            Node d; d.zero = false; d.v = mat.wrow[n][i];
            if (DT || ENC_CLONE) d.nv = mat.nwrow[n][i];
            inp.push_back(std::move(d));
          } else if (i == 0 && c == 0) skipped++;
        }
        if (inp.empty()) continue;
        vector<Row> cnt;
        if (inp.size() == 1) cnt.push_back(inp[0].v);  // degenerate: count == leaf
        else cnt = popcount_tree(std::move(inp));
        for (uint32_t m = 0; m < mat.M; m++) {
          int seg = LANES[m >> 5]; int bt = m & 31;
          long long ps = 0;
          for (size_t w = 0; w < cnt.size(); w++) ps += (long long)((cnt[w][seg] >> bt) & 1) << w;
          p64[(size_t)m * NBLK + b] += fac * ps;
        }
      }
      fprintf(stderr, "[lane2]   partials plane i=%u c=%u done (%.1f s, FA=%ld)\n",
              i, c, now_s() - tp0, N_FA - fap0);
    }
  double wall = now_s() - t0;
  fprintf(stderr, "[lane2] PARTIALS handle=%u K=%u M=%u q=%u r=%u NBLK=%u: %.2f s wall, "
          "%ld FA, %ld MAJ, %ld execs, %ld pcwrites, %ld taps zero-skipped\n",
          handle, mat.K, mat.M, mat.qb, rb, NBLK, wall,
          N_FA - fa0, N_MAJ - maj0, N_EXEC - exec0, N_PCW - pcw0, skipped);
  if (DT || ENC_CLONE)
    fprintf(stderr, "[lane2]   dual-rail consistency: %ld violations / %ld lane-checks\n",
            N_RAIL_VIOL, N_RAIL_LANES);

  vector<uint8_t> resp(20 + (size_t)mat.M * NBLK * 4);
  uint32_t hdr[5] = {MAGIC_PART_ACK, handle, 0, mat.M, NBLK};
  memcpy(resp.data(), hdr, 20);
  for (size_t j = 0; j < (size_t)mat.M * NBLK; j++) {
    long long v = p64[j];
    if (v > 0x7FFFFFFFLL || v < -0x80000000LL) { fprintf(stderr, "[lane2] partial overflow j=%zu\n", j); ack_part_err(handle, 2); return; }
    int32_t v32 = (int32_t)v;
    memcpy(resp.data() + 20 + j * 4, &v32, 4);
  }
  write_resp(resp.data(), (uint32_t)resp.size());
}

int main(int argc, char** argv) {
  if (argc < 6) {
    fprintf(stderr, "Usage: %s <bender> <calib> <bank> <s_id> <colmask>\n", argv[0]);
    return 1;
  }
  int bender = atoi(argv[1]); string calib_p = argv[2];
  BANK = atoi(argv[3]); int sid = atoi(argv[4]);
  vector<uint8_t> mask = read_mask(argv[5]);
  MV = (getenv("PIM_VOTE3") && atoi(getenv("PIM_VOTE3"))) ? 1 : 0;
  PACK = (getenv("LANE2_PACK") && !atoi(getenv("LANE2_PACK"))) ? 0 : 1;
  DT = (getenv("LANE2_DUALTRACK") && atoi(getenv("LANE2_DUALTRACK"))) ? 1 : 0;
  ENC_CLONE = (getenv("LANE2_ENCODE") && string(getenv("LANE2_ENCODE")) == "clone") ? 1 : 0;
  if (ENC_CLONE && MV) {
    fprintf(stderr, "[lane2] PIM_VOTE3 is not supported in clone mode — ignored\n");
    MV = 0;
  }
  if (const char* e = getenv("LANE2_ACCUM")) ACCUM = atoi(e);
  if (ACCUM < 0 || ACCUM > 2) die("LANE2_ACCUM must be 0, 1 (read arm) or 2 (accum arm)", 1);
  if (ACCUM) WRES = (getenv("LANE2_WRES") && atoi(getenv("LANE2_WRES"))) ? 1 : 0;
  if (ACCUM) PPACK = (getenv("LANE2_PLANE_PACK") && atoi(getenv("LANE2_PLANE_PACK"))) ? 1 : 0;
  if (PPACK && ACCUM != 2) die("LANE2_PLANE_PACK requires LANE2_ACCUM=2 (totals arm)", 1);
  if (PPACK && !WRES)
    fprintf(stderr, "[lane2] WARN: LANE2_PLANE_PACK without LANE2_WRES — every "
            "output over-budgets the packed program and falls back per-plane\n");
  if (const char* e = getenv("LANE2_REF_POLICY")) {
    if (string(e) == "zero2") { REF_INIT = 0u; REF_NFRAC = 2; }
    else if (string(e) != "" && string(e) != "legacy")
      die("LANE2_REF_POLICY must be 'legacy' (ONE+3) or 'zero2' (ZERO+2)", 1);
  }
  if (ACCUM) {
    if (!PACK) die("LANE2_ACCUM requires the 8K-IMEM packed shape (LANE2_PACK=1)", 1);
    if (MV || DT || ENC_CLONE) {
      fprintf(stderr, "[lane2] ACCUM mode: PIM_VOTE3/LANE2_DUALTRACK/LANE2_ENCODE ignored\n");
      MV = DT = 0; ENC_CLONE = 0;
    }
  }

  vector<Calib> cal = read_calib(calib_p); bool found = false;
  for (auto& c : cal) if (c.s_id == sid && c.bank == BANK) { CAL = c; found = true; break; }
  if (!found) { fprintf(stderr, "[lane2] tuple s_id=%d bank=%d not in %s\n", sid, BANK, calib_p.c_str()); return 2; }
  int n_mask = 0; for (auto v : mask) n_mask += v;

  // stdout is the binary response channel; all library prints must go to
  // stderr (BitNet-server dup trick).
  fflush(stdout);
  response_fd = dup(STDOUT_FILENO);
  if (response_fd < 0) { fprintf(stderr, "[lane2] dup(stdout) failed\n"); return 3; }
  if (dup2(STDERR_FILENO, STDOUT_FILENO) < 0) { fprintf(stderr, "[lane2] dup2 failed\n"); return 3; }

  bool sim_backend = (getenv("LANE2_BACKEND") && string(getenv("LANE2_BACKEND")) == "sim");
  if (ENC_CLONE && sim_backend)
    die("LANE2_ENCODE=clone is silicon-only (the sim has no model for the fastpath tuple)", 2);
  if (ACCUM && sim_backend)
    die("LANE2_ACCUM is silicon-only (fastpath tuple + build-6 accum HDL)", 2);
  unique_ptr<SoftMCPlatform> owner;
  if (sim_backend) {
    auto sp = make_unique<SimPlatform>();
    sp->load_calib(calib_p);
    fprintf(stderr, "[lane2] LANE2_BACKEND=sim — SimDramModel (bring-up only)\n");
    owner = std::move(sp);
  } else {
    owner = make_unique<SoftMCPlatform>(bender);
    if (owner->init() != SOFTMC_SUCCESS) { fprintf(stderr, "[lane2] platform init failed\n"); return 3; }
    owner->reset_fpga();
  }
  PF = owner.get();

  fprintf(stderr, "[lane2] server: bender=%d s_id=%d bank=%d pack=%d vote3=%d "
          "dualtrack=%d encode=%s accum=%d input colmask=%d/2048\n", bender, sid, BANK, PACK, MV,
          DT, ENC_CLONE ? "clone" : "host", ACCUM, n_mask);

  int strials = getenv("LANE2_SCREEN_TRIALS") ? atoi(getenv("LANE2_SCREEN_TRIALS")) : 3;
  if (strials < 1) strials = 1;
  double ts = now_s();
  if (ACCUM) {
    // Road-B product mode: aref off (the suite precaution — maintenance
    // pulses race DIFF flush slots); x-row integrity is refresh-by-rewrite
    // (LANE2_XREFRESH) + the in-stream sentinel totals.
    PF->set_aref(false);
    pr_startup(mask, strials);
    LANES = PSEG;   // satisfies the shared capacity/report plumbing
  } else if (ENC_CLONE) {
    // clone engine bring-up: value-row mask screen, role assignment, then an
    // op-matched column screen through the complete 9-gate dual-rail FA.
    vector<uint32_t> usable = cl_screen_masks(CAL.open);
    if ((int)usable.size() < NROLE)
      die("clone mask screen: not enough usable value rows", 4);
    for (int r = 0; r < NROLE; r++) CROW[r] = CL_SUB + ((CL_Ti0 - CL_SUB) ^ usable[r]);
    fprintf(stderr, "[lane2] clone roles (local offsets):");
    for (int r = 0; r < NROLE; r++) fprintf(stderr, " %u", CROW[r] - CL_SUB);
    fprintf(stderr, "\n");
    vector<uint8_t> gm = cl_op_screen(mask, strials);
    int n_gm = 0; for (auto v : gm) n_gm += v;
    LANES.clear();
    for (int s = 0; s < 2048; s++) if (gm[s]) LANES.push_back(s);
    fprintf(stderr, "[lane2] clone FA op-screen (%d trials): %d/%d reliable segments "
            "(%.1f s total init, %ld MAJs) -> capacity M<=%zu outputs/pass\n",
            strials, n_gm, n_mask, now_s() - ts, N_MAJ, LANES.size() * 32);
    N_RAIL_LANES = N_RAIL_VIOL = 0;  // screen-phase rail stats don't count
  } else {
    // op-matched screen at startup (test_mvdram_gemv.cpp convention)
    vector<uint8_t> gm = op_screen(mask, strials);
    int n_gm = 0; for (auto v : gm) n_gm += v;
    LANES.clear();
    for (int s = 0; s < 2048; s++) if (gm[s]) LANES.push_back(s);
    fprintf(stderr, "[lane2] op-matched screen (%d trials): %d/%d reliable segments "
            "(%.1f s, %ld MAJs) -> capacity M<=%zu outputs/pass\n",
            strials, n_gm, n_mask, now_s() - ts, N_MAJ, LANES.size() * 32);
  }
  if (LANES.empty()) die("no reliable columns after screen", 4);

  fprintf(stderr, "[lane2] ready\n");
  while (true) {
    uint32_t req_len;
    if (!read_exact(&req_len, 4)) { fprintf(stderr, "[lane2] EOF, exiting\n"); break; }
    if (req_len == 0) { fprintf(stderr, "[lane2] quit sentinel\n"); break; }
    if (req_len > (256u << 20)) die("oversize request", 1);
    vector<uint8_t> req(req_len);
    if (!read_exact(req.data(), req_len)) { fprintf(stderr, "[lane2] short request, exiting\n"); break; }
    uint32_t magic = 0; if (req_len >= 4) memcpy(&magic, req.data(), 4);
    if (magic == MAGIC_LOAD) handle_load(req.data(), req_len);
    else if (magic == MAGIC_GEMV) handle_gemv(req.data(), req_len);
    else if (magic == MAGIC_PART) handle_partials(req.data(), req_len);
    else { fprintf(stderr, "[lane2] bad magic %08x\n", magic); uint32_t b[3] = {0, 0, 1}; write_resp(b, sizeof b); }
  }
  return 0;
}
