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
#include "instruction.h"
#include "prog.h"
#include "platform.h"
#include "sim_platform.h"
#include "util.h"

#include <array>
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
  for (int j = 0; j < 3; j++) { p.add_inst(SMC_SLEEP(6)); p.add_below(frac_b(0, open0)); p.add_inst(SMC_SLEEP(6)); }
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
  uni.push_back({CAL.open[0], ONE});
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
  uni.push_back({CAL.open[0], ONE});
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
          "dualtrack=%d encode=%s input colmask=%d/2048\n", bender, sid, BANK, PACK, MV,
          DT, ENC_CLONE ? "clone" : "host", n_mask);

  int strials = getenv("LANE2_SCREEN_TRIALS") ? atoi(getenv("LANE2_SCREEN_TRIALS")) : 3;
  if (strials < 1) strials = 1;
  double ts = now_s();
  if (ENC_CLONE) {
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
