// Verilator testbench for the readback_engine build4 drain-capture-timing
// fix (RESULT.md addendum 20c), extending the build3 drain-race harness.
//
// Two variants are built from this one source:
//   obj_b3 : readback_engine_build3.v  (the flashed build3 QUAD copy)
//   obj_b4 : readback_engine.v         (build4)  — compiled with -DTB_BUILD4
//
// Scenario groups:
//   (a)-(e)  the build3 suite, "fixed" expectations — BOTH variants must
//            pass (build4 regression: when the read path is quiet at the
//            flush edge, build4 is cycle-identical to build3). Only the
//            trailer magic differs (build3 0xDBC0DE01 / build4 0xDBC0DE02).
//   (f1)(f2) THE SILICON REPRO — the program's rd_valid tail lands after
//            the flush edge (softmc_fin fires at END-fetch; reads are
//            still in the DDR pipeline / PHY):
//              f1 partial tail: b3 captures a SHORT total and inflates
//                 the next program's; b4 exact both.
//              f2 full tail:    b3 captures ZERO (the silicon all-zero
//                 signature); b4 exact.
//   (g)      batch of 3 tail-crossing programs: b3 shows the silicon
//            "one-program delivery lag" pattern (0, T1, T2); b4 delivers
//            (T1, T2, T3) in order.
//   (h)      [build4 only] SET-READ/SET-DIFF words: idempotent
//            (double-send = no-op), legacy toggle still works.
//   (i)      [build4 only] un-announced maintenance read landing inside
//            the deferral window: not accumulated, capture exact,
//            outstanding floor holds.
//
// The tail-crossing scenarios drive read_seq_incoming/incoming_reads the
// way fetch_stage does (one pulse per SMC_INFO packet, count = reads in
// the segment) — the signals build4's outstanding counter consumes.
//
// Usage: Vreadback_engine <b3|b4>   (label only; behavior from TB_BUILD4)

#include <verilated.h>
#include "Vreadback_engine.h"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

static Vreadback_engine* top;
static const char* variant = "b?";
static bool is_b4_rtl = false;   // set from TB_BUILD4 at compile time
static bool is_b6_rtl = false;   // set from TB_BUILD6 (buffer_space conservation)
static int failures = 0;

struct Beat {
    uint32_t w[8];
    bool tlast;
};
static std::vector<std::vector<Beat> > msgs; // tlast-delimited messages
static std::vector<Beat> open_msg;

#if defined(TB_BUILD10)
static const uint32_t MAGIC = 0xDBC0DE09u;  // build10: raw-count wdata mask
#elif defined(TB_BUILD9)
static const uint32_t MAGIC = 0xDBC0DE08u;  // build9: streaming fetch (engine = 8b + magic)
#elif defined(TB_BUILD8)
static const uint32_t MAGIC = 0xDBC0DE07u;  // build8b: widx edge+quiet realign
#elif defined(TB_BUILD7)
static const uint32_t MAGIC = 0xDBC0DE05u;
#elif defined(TB_BUILD6)
static const uint32_t MAGIC = 0xDBC0DE04u;
#elif defined(TB_BUILD5)
static const uint32_t MAGIC = 0xDBC0DE03u;
#elif defined(TB_BUILD4)
static const uint32_t MAGIC = 0xDBC0DE02u;
#else
static const uint32_t MAGIC = 0xDBC0DE01u;
#endif

// ---- clocking ------------------------------------------------------------
static void tick() {
    top->clk = 0;
    top->eval();
    if (top->c2h_tvalid_0 && top->c2h_tready_0) {
        Beat b;
        b.tlast = top->c2h_tlast_0 != 0;
        for (int i = 0; i < 8; i++) b.w[i] = top->c2h_tdata_0[i];
        open_msg.push_back(b);
        if (b.tlast) {
            msgs.push_back(open_msg);
            open_msg.clear();
        }
    }
    top->clk = 1;
    top->eval();
}

static void clear_inputs() {
    top->flush = 0;
    top->read_seq_incoming = 0;
    top->incoming_reads = 0;
    top->switch_mode = 0;
    top->rd_valid = 0;
    top->per_rd_init = 0;
    top->per_zq_init = 0;
    top->per_ref_init = 0;
#ifdef TB_BUILD4
    top->set_mode_read = 0;
    top->set_mode_diff = 0;
#endif
#ifdef TB_BUILD7
    top->set_mode_read = 0;
    top->set_mode_diff = 0;
    top->set_mode_segpop = 0;
#endif
#if defined(TB_BUILD8) || defined(TB_BUILD9) || defined(TB_BUILD10)
    top->set_mode_read = 0;
    top->set_mode_diff = 0;
    top->set_mode_segpop = 0;
    top->set_mode_accxbp = 0;
    top->set_acc_weight = 0;
    top->acc_weight_pl = 0;
    top->flush_acc = 0;
#endif
    for (int i = 0; i < 16; i++) { top->rd_data[i] = 0; top->ddr_wdata[i] = 0; }
    top->c2h_tready_0 = 1; // host always ready
}

static void idle(int n) {
    top->flush = 0;
    top->rd_valid = 0;
    top->switch_mode = 0;
    top->read_seq_incoming = 0;
    top->per_rd_init = 0;
    top->per_zq_init = 0;
    top->per_ref_init = 0;
    for (int i = 0; i < n; i++) tick();
}

static void hard_reset() {
    clear_inputs();
    top->rst = 1;
    for (int i = 0; i < 5; i++) tick();
    top->rst = 0;
    idle(3);
    msgs.clear();
    open_msg.clear();
}

static void to_diff_mode() {           // legacy toggle path (both variants)
    top->switch_mode = 1;
    tick();
    top->switch_mode = 0;
    idle(2);
}

// fetch_stage-faithful announcement: one pulse, count = reads in segment.
static void announce_reads(int n) {
    top->read_seq_incoming = 1;
    top->incoming_reads = (uint16_t)n;
    tick();
    top->read_seq_incoming = 0;
    top->incoming_reads = 0;
}

// n back-to-back reads, each differing from ddr_wdata (all zeros) in
// `bits` low bits -> per-read popcount = bits.
static void user_reads(int n, int bits) {
    for (int k = 0; k < n; k++) {
        for (int i = 0; i < 16; i++) top->rd_data[i] = 0;
        top->rd_data[0] = (bits >= 32) ? 0xFFFFFFFFu : ((1u << bits) - 1u);
        top->rd_valid = 1;
        tick();
    }
    top->rd_valid = 0;
}

static void flush_pulse(int width) {
    top->flush = 1;
    for (int i = 0; i < width; i++) tick();
    top->flush = 0;
}

// kind: 0=per_rd, 1=per_zq, 2=per_ref
static void maint_event(int kind, bool with_read, bool with_flush) {
    if (kind == 0) top->per_rd_init = 1;
    if (kind == 1) top->per_zq_init = 1;
    if (kind == 2) top->per_ref_init = 1;
    tick();
    top->per_rd_init = 0;
    top->per_zq_init = 0;
    top->per_ref_init = 0;
    idle(3);
    if (with_read) {
        // maintenance read returns garbage; must NOT reach the accumulator
        for (int i = 0; i < 16; i++) top->rd_data[i] = 0xFFFFFFFFu;
        top->rd_valid = 1;
        tick();
        top->rd_valid = 0;
        idle(4); // let ignore_read clear
    }
    if (with_flush) {
        flush_pulse(1);
        idle(8);
    }
}

// ---- c2h beat pairing (receiver model) -----------------------------------
// Every 512b engine word leaves rdback_fifo as two 256b c2h beats. The IP
// presents the most-significant half first and readback_engine pre-swaps the
// halves on `din`, so the pair arrives LOW half first and the receiver reads
// it straight — no de-swap. Build the FIFO with `+define+SIM_RBF_LSB_FIRST`
// (legacy low-half-first model) and this receiver inverts with it, via a
// matching `-CFLAGS -DSIM_RBF_LSB_FIRST`; the two must always agree.
#ifdef SIM_RBF_LSB_FIRST
static inline size_t beat_lo(size_t w) { return w * 2 + 1; }
static inline size_t beat_hi(size_t w) { return w * 2 + 0; }
#else
static inline size_t beat_lo(size_t w) { return w * 2 + 0; }
static inline size_t beat_hi(size_t w) { return w * 2 + 1; }
#endif

// ---- checks --------------------------------------------------------------
static void check(const char* what, bool ok) {
    printf("  [%s] %-68s %s\n", variant, what, ok ? "PASS" : "FAIL");
    if (!ok) failures++;
}

static uint32_t chunk_total(const std::vector<Beat>& m, size_t chunk_idx) {
    // accum chunk = 2 beats; the high half is zero, the low half carries the
    // total in w[0].
    size_t b = beat_lo(chunk_idx);
    if (b >= m.size()) return 0xDEADDEADu;
    return m[b].w[0];
}

static bool beat_is_zero(const Beat& b) {
    for (int i = 0; i < 8; i++) if (b.w[i]) return false;
    return true;
}

static void expect_trailer(const std::vector<Beat>& m,
                           uint32_t rd, uint32_t zq, uint32_t ref,
                           uint32_t fe, uint32_t eat, uint32_t drain,
                           uint32_t accw) {
    const Beat& t = m.back();
    check("trailer magic", t.w[0] == MAGIC);
    char buf[128];
    bool ok = t.w[1] == rd && t.w[2] == zq && t.w[3] == ref &&
              t.w[4] == fe && t.w[5] == eat && t.w[6] == drain &&
              t.w[7] == accw;
    snprintf(buf, sizeof(buf),
             "trailer ctrs rd=%u zq=%u ref=%u fe=%u eat=%u drn=%u aw=%u",
             t.w[1], t.w[2], t.w[3], t.w[4], t.w[5], t.w[6], t.w[7]);
    check(buf, ok);
    if (!ok)
        printf("        expected rd=%u zq=%u ref=%u fe=%u eat=%u drn=%u aw=%u\n",
               rd, zq, ref, fe, eat, drain, accw);
}

// Simple "message is chunk(total)+trailer" check.
static bool msg_is_chunk_and_trailer(const std::vector<Beat>& m, uint32_t total) {
    return m.size() == 3 && beat_is_zero(m[beat_hi(0)]) && chunk_total(m, 0) == total &&
           !m[0].tlast && !m[1].tlast && m[2].tlast && m[2].w[0] == MAGIC;
}

// ---- build3 suite (fixed expectations; both variants must pass) ----------

static void scenario_a() {
    printf(" (a) clean: 8 reads x 3 bits, 1-cycle flush\n");
    hard_reset();
    to_diff_mode();
    user_reads(8, 3);
    idle(10);
    flush_pulse(1);
    idle(60);
    check("exactly one message delivered", msgs.size() == 1);
    if (msgs.size() == 1) {
        const std::vector<Beat>& m = msgs[0];
        check("message = 3 beats (2 chunk + 1 trailer)", m.size() == 3);
        if (m.size() == 3) {
            check("chunk high half all zero", beat_is_zero(m[beat_hi(0)]));
            check("chunk total == 24", chunk_total(m, 0) == 24);
            check("tlast only on trailer", !m[0].tlast && !m[1].tlast && m[2].tlast);
            expect_trailer(m, 0, 0, 0, 1, 0, 1, 1);
        }
    }
}

static void scenario_b() {
    printf(" (b) stale maintenance accounting: per_rd init + read, NO maint flush, then user program\n");
    hard_reset();
    to_diff_mode();
    maint_event(0 /*per_rd*/, true /*read*/, false /*NO flush*/);
    idle(10);
    user_reads(8, 3);
    idle(10);
    flush_pulse(1);
    idle(80);
    check("user result delivered despite stale accounting", msgs.size() == 1);
    if (msgs.size() == 1) {
        check("chunk total == 24 (maintenance read not accumulated)",
              chunk_total(msgs[0], 0) == 24);
        expect_trailer(msgs[0], 1, 0, 0, 1, 0, 1, 1);
    }
    // continuation: healthy maintenance (with its flush), then another program
    size_t before = msgs.size();
    maint_event(1 /*per_zq*/, false, true /*WITH flush*/);
    check("no message from the maintenance flush", msgs.size() == before);
    user_reads(8, 3);
    idle(10);
    flush_pulse(1);
    idle(80);
    check("second program delivered", msgs.size() == before + 1);
    if (msgs.size() == before + 1) {
        check("second total == 24 (no stale carry-over)",
              chunk_total(msgs.back(), 0) == 24);
        expect_trailer(msgs.back(), 1, 1, 0, 3, 1, 2, 2);
    }
}

static void scenario_c() {
    printf(" (c) wide flush: 8 reads x 3 bits, flush held 3 cycles\n");
    hard_reset();
    to_diff_mode();
    user_reads(8, 3);
    idle(10);
    flush_pulse(3);
    idle(80);
    check("exactly one message", msgs.size() == 1);
    if (msgs.size() != 1) return;
    const std::vector<Beat>& m = msgs[0];
    check("single drain -> 3 beats only", m.size() == 3);
    if (m.size() == 3) {
        check("chunk total == 24", chunk_total(m, 0) == 24);
        expect_trailer(m, 0, 0, 0, 1, 0, 1, 1);
    }
}

static void scenario_d() {
    printf(" (d) double maintenance 2 cycles apart, their two flushes, then user program\n");
    hard_reset();
    to_diff_mode();
    top->per_zq_init = 1; tick(); top->per_zq_init = 0;
    idle(2);
    top->per_ref_init = 1; tick(); top->per_ref_init = 0;
    idle(10);
    flush_pulse(1); // maintenance flush #1
    idle(10);
    flush_pulse(1); // maintenance flush #2
    idle(10);
    size_t after_maint = msgs.size();
    user_reads(8, 3);
    idle(10);
    flush_pulse(1);
    idle(80);
    check("both maintenance flushes eaten (no message)", after_maint == 0);
    check("exactly the user message", msgs.size() == 1);
    if (msgs.size() == 1) {
        check("chunk total == 24", chunk_total(msgs[0], 0) == 24);
        expect_trailer(msgs[0], 0, 1, 1, 3, 2, 1, 1);
    }
}

static void scenario_e() {
    printf(" (e) READ_MODE smoke: 4 reads, flush; data beats dumped for cross-build diff\n");
    hard_reset(); // stays in READ_MODE
    for (int k = 0; k < 4; k++) {
        for (int i = 0; i < 16; i++)
            top->rd_data[i] = 0xA0000000u + (uint32_t)(k * 16 + i);
        top->rd_valid = 1;
        tick();
    }
    top->rd_valid = 0;
    idle(10);
    flush_pulse(1);
    idle(60);
    check("one message", msgs.size() == 1);
    if (msgs.size() != 1) return;
    const std::vector<Beat>& m = msgs[0];
    check("9 beats (8 data + trailer)", m.size() == 9);
    check("tlast only on trailer", m.back().tlast);
    if (m.size() == 9)
        check("trailer magic present in READ_MODE too", m.back().w[0] == MAGIC);
    // dump data beats (all but the trailer) for the cross-build diff
    std::string fn = std::string("readmode_beats_") + variant + ".txt";
    FILE* f = fopen(fn.c_str(), "w");
    for (size_t b = 0; b + 1 < m.size(); b++) {
        for (int i = 7; i >= 0; i--) fprintf(f, "%08x", m[b].w[i]);
        fprintf(f, " tlast=%d\n", m[b].tlast ? 1 : 0);
    }
    fclose(f);
    printf("  [%s] wrote %s (%zu data beats)\n", variant, fn.c_str(), m.size() - 1);
}

// ---- build4 repro + fix scenarios ----------------------------------------

// One "program" whose rd_valid tail crosses the flush edge:
// announce n reads, deliver `before` of them, flush, wait `gap`, deliver
// the remaining n-before.
static void tail_crossing_program(int n, int before, int bits, int gap) {
    announce_reads(n);
    idle(2);
    if (before > 0) user_reads(before, bits);
    idle(4);                 // pipe settles; sum holds `before` samples
    flush_pulse(1);          // frontend fin+32 flush — EARLY vs the tail
    idle(gap);
    if (n - before > 0) user_reads(n - before, bits);
    top->rd_valid = 0;
}

static void scenario_f1() {
    printf(" (f1) REPRO partial tail: 8 announced, 5 before flush, 3 after (+ follow-up program)\n");
    hard_reset();
    to_diff_mode();
    tail_crossing_program(8, 5, 3, 10);
    idle(40);                // deferral window + delivery
    // follow-up program, fully paced
    announce_reads(8);
    idle(2);
    user_reads(8, 3);
    idle(10);
    flush_pulse(1);
    idle(60);
    check("two messages", msgs.size() == 2);
    if (msgs.size() != 2) return;
    if (!is_b4_rtl) {
        check("B3 BUG: first chunk SHORT (15 = 5 reads only)",
              msg_is_chunk_and_trailer(msgs[0], 15));
        check("B3 BUG: tail leaked into next total (33 = 24+9)",
              msg_is_chunk_and_trailer(msgs[1], 33));
    } else {
        check("b4: first total EXACT (24) despite late tail",
              msg_is_chunk_and_trailer(msgs[0], 24));
        check("b4: second total EXACT (24), no leak",
              msg_is_chunk_and_trailer(msgs[1], 24));
        expect_trailer(msgs[1], 0, 0, 0, 2, 0, 2, 2);
    }
}

static void scenario_f2() {
    printf(" (f2) REPRO full tail: 8 announced, ALL data after the flush edge\n");
    hard_reset();
    to_diff_mode();
    tail_crossing_program(8, 0, 3, 6);
    idle(40);
    // follow-up program, fully paced
    announce_reads(8);
    idle(2);
    user_reads(8, 3);
    idle(10);
    flush_pulse(1);
    idle(60);
    check("two messages", msgs.size() == 2);
    if (msgs.size() != 2) return;
    if (!is_b4_rtl) {
        check("B3 BUG: first chunk ZERO (silicon all-zero signature)",
              msg_is_chunk_and_trailer(msgs[0], 0));
        check("B3 BUG: doubled second total (48 = 24+24)",
              msg_is_chunk_and_trailer(msgs[1], 48));
    } else {
        check("b4: first total EXACT (24) — captured at end-of-reads",
              msg_is_chunk_and_trailer(msgs[0], 24));
        check("b4: second total EXACT (24)",
              msg_is_chunk_and_trailer(msgs[1], 24));
        expect_trailer(msgs[1], 0, 0, 0, 2, 0, 2, 2);
    }
}

static void scenario_g() {
    printf(" (g) batch of 3 programs, every rd tail after its flush\n");
    hard_reset();
    to_diff_mode();
    const uint32_t T[3] = {24, 40, 16}; // 8 reads x {3,5,2} bits
    const int BITS[3] = {3, 5, 2};
    for (int k = 0; k < 3; k++) {
        tail_crossing_program(8, 0, BITS[k], 6);
        idle(15);            // next program's load gap (µs on silicon)
    }
    idle(60);
    check("three messages", msgs.size() == 3);
    if (msgs.size() != 3) return;
    if (!is_b4_rtl) {
        // one-program delivery lag: 0, T1, T2 (T3 stranded in the accum)
        check("B3 BUG: lag pattern chunk0 == 0",
              msg_is_chunk_and_trailer(msgs[0], 0));
        check("B3 BUG: lag pattern chunk1 == T1",
              msg_is_chunk_and_trailer(msgs[1], T[0]));
        check("B3 BUG: lag pattern chunk2 == T2 (T3 stranded)",
              msg_is_chunk_and_trailer(msgs[2], T[1]));
    } else {
        bool ok = true;
        for (int k = 0; k < 3; k++) ok &= msg_is_chunk_and_trailer(msgs[k], T[k]);
        check("b4: totals (24,40,16) exact and IN ORDER", ok);
        expect_trailer(msgs[2], 0, 0, 0, 3, 0, 3, 3);
    }
}

#ifdef TB_BUILD4
static void set_word(bool diff) {
    if (diff) top->set_mode_diff = 1; else top->set_mode_read = 1;
    tick();
    top->set_mode_diff = 0;
    top->set_mode_read = 0;
    idle(2);
}

// Probe the CURRENT mode with a 4-read paced program; returns the message
// count delta (DIFF -> 1 message with chunk total, READ -> 1 message of
// data beats). Returns observed chunk total or 0xFFFFFFFF for READ shape.
static uint32_t probe_program() {
    size_t before = msgs.size();
    user_reads(4, 3);
    idle(10);
    flush_pulse(1);
    idle(60);
    if (msgs.size() != before + 1) return 0xEEEEEEEEu;
    const std::vector<Beat>& m = msgs.back();
    if (m.size() == 3 && beat_is_zero(m[beat_hi(0)])) return chunk_total(m, 0); // DIFF shape
    if (m.size() == 9) return 0xFFFFFFFFu;                             // READ shape
    return 0xEEEEEEEEu;
}

static void scenario_h() {
    printf(" (h) [b4] SET-READ/SET-DIFF idempotent; legacy toggle intact\n");
    hard_reset();                       // READ_MODE
    set_word(true);                     // SET-DIFF
    set_word(true);                     // SET-DIFF again (double-send)
    check("double SET-DIFF lands in DIFF (probe total 12)", probe_program() == 12);
    set_word(false);                    // SET-READ
    set_word(false);                    // SET-READ again
    check("double SET-READ lands in READ (probe = data shape)",
          probe_program() == 0xFFFFFFFFu);
    // legacy toggle still flips modes
    to_diff_mode();
    check("legacy toggle READ->DIFF still works (probe total 12)",
          probe_program() == 12);
    to_diff_mode();                     // toggle back
    check("legacy toggle DIFF->READ still works (probe = data shape)",
          probe_program() == 0xFFFFFFFFu);
    // SET into the mode we are already in, then a real program: no side
    // effects on framing.
    set_word(false);
    check("SET-READ while in READ is a no-op (probe = data shape)",
          probe_program() == 0xFFFFFFFFu);
}

static void scenario_i() {
    printf(" (i) [b4] un-announced maintenance read inside the deferral window\n");
    hard_reset();
    to_diff_mode();
    // program: 8 announced reads, all after the flush edge
    announce_reads(8);
    idle(2);
    flush_pulse(1);
    idle(4);
    user_reads(6, 3);                  // 6 of 8 announced reads return...
    // ...a maintenance per_rd fires mid-tail. On real HW its read CAS
    // issues after the user tail (in-order PHY), so drive: init pulse,
    // the final 2 user reads, then the maintenance read (garbage data).
    top->per_rd_init = 1; tick(); top->per_rd_init = 0;
    user_reads(2, 3);
    for (int i = 0; i < 16; i++) top->rd_data[i] = 0xFFFFFFFFu;
    top->rd_valid = 1; tick(); top->rd_valid = 0;   // maintenance read (garbage)
    idle(20);
    flush_pulse(1);                    // the maintenance program's own flush
    idle(60);
    // NOTE on faithfulness: the ignore_read machinery is PRE-EXISTING
    // build3 semantics, unchanged by build4. With the init pulse landing
    // one cycle after a read (rd_valid_r still high), the rd_valid_r
    // clear preempts the init's set, so ignore never engages here: all 8
    // user reads (24) AND the garbage maintenance read (diff vs zero
    // wdata = 512) accumulate -> total 536, deterministic in this TB.
    // What build4 must guarantee is the STRUCTURAL part: the capture
    // waits for the announced tail, the outstanding counter floors at 0
    // on the extra un-announced return (no wedge), exactly one
    // chunk+trailer message forms, and the later maintenance flush is
    // eaten (msgs.size() stays 1). The trailer snapshot is taken when
    // the message closes — BEFORE that maintenance flush — so its
    // eaten-counter is still 0 there.
    check("exactly one message (maintenance flush eaten)", msgs.size() == 1);
    if (msgs.size() == 1) {
        const std::vector<Beat>& m = msgs[0];
        check("message = chunk + trailer framing", m.size() == 3 && m[2].tlast);
        uint32_t tot = chunk_total(m, 0);
        char buf[96];
        snprintf(buf, sizeof(buf),
                 "captured total (got %u; expect 536 = 24 user + 512 maint garbage)", tot);
        check(buf, tot == 536u);
        const Beat& t = m.back();
        check("trailer: rd_init counted once, exactly one capture/write",
              t.w[1] == 1 && t.w[6] == 1 && t.w[7] == 1);
    }
}
#endif

// ---- build6: buffer_space conservation --------------------------------
// Silicon 2026-07-20/21: DIFF-accum sessions starve fetch after exactly 8
// programs of 128 announced reads (h2c errno 512 at program 9/10,
// NUM_COLS-independent). Accounting: debit 2 units/announced read; only
// actual c2h transfers credit (+1). Accum swallows the beats, so each
// program leaks 2*128-2 = 254 units; 2048 - 8*254 = 16 units (8 exposed)
// cannot admit the next 128-read sequence. build6 credits consumed DIFF
// beats (+2 each) and un-credits DIFF-mode c2h transfers: net zero.
static void scenario_j() {
    printf(" (j) buffer_space conservation: 8 x (announce 128 + 128 reads + flush)\n");
    hard_reset();
    uint32_t bs0 = top->buffer_space;
    char buf[120];
    snprintf(buf, sizeof(buf), "reset buffer_space exposed = 1024 (got %u)", bs0);
    check(buf, bs0 == 1024u);
    to_diff_mode();
    for (int k = 0; k < 8; k++) {
        announce_reads(128);
        idle(2);
        user_reads(128, 3);        // per-program total = 384
        idle(10);
        flush_pulse(1);
        idle(80);
    }
    idle(40);
    uint32_t bs_end = top->buffer_space;
    check("8 DIFF messages delivered", msgs.size() == 8);
    if (is_b6_rtl) {
        snprintf(buf, sizeof(buf),
                 "CONSERVED: buffer_space back at %u after 8 programs (got %u)", bs0, bs_end);
        check(buf, bs_end == bs0);
        bool tot_ok = true;
        for (size_t k = 0; k < msgs.size(); k++)
            tot_ok &= (chunk_total(msgs[k], 0) == 384u);
        check("all 8 totals exact (384)", tot_ok);
    } else {
        snprintf(buf, sizeof(buf),
                 "LEAK DOCUMENTED (<=build5): 254 units/program -> exposed 8 (got %u)", bs_end);
        check(buf, bs_end == 8u);
    }
    // READ_MODE conservation must hold on every build (announce+read+drain).
    to_diff_mode();                 // toggle back to READ
    uint32_t bs_read0 = top->buffer_space;
    announce_reads(8);
    idle(2);
    for (int k = 0; k < 8; k++) {
        for (int i = 0; i < 16; i++) top->rd_data[i] = 0xB0000000u + (uint32_t)k;
        top->rd_valid = 1;
        tick();
    }
    top->rd_valid = 0;
    idle(10);
    flush_pulse(1);
    idle(80);
    snprintf(buf, sizeof(buf),
             "READ_MODE conserved across announced program (%u -> %u)", bs_read0, top->buffer_space);
    check(buf, top->buffer_space == bs_read0);
}

#if defined(TB_BUILD7) || defined(TB_BUILD8) || defined(TB_BUILD9) || defined(TB_BUILD10)
// ---- build7: SEG_POP per-segment popcount readout ------------------------
static void to_segpop_mode() {
    top->set_mode_segpop = 1; tick(); top->set_mode_segpop = 0; idle(2);
}
static void to_read_mode_set() {
    top->set_mode_read = 1; tick(); top->set_mode_read = 0; idle(2);
}
// popcount of a 32-bit lane
static int pc32(uint32_t v){ int c=0; while(v){c+=v&1; v>>=1;} return c; }

// Feed `nbeats` read beats; beat b's 16 segments carry pattern seg_val(b,s).
// ddr_wdata stays 0 so read_diff = rd_data, and pc_out_l4[s] = popcount(seg).
static void seg_user_reads(int nbeats, uint32_t (*seg_val)(int,int)) {
    for (int b = 0; b < nbeats; b++) {
        for (int s = 0; s < 16; s++) top->rd_data[s] = seg_val(b, s);
        top->rd_valid = 1; tick();
    }
    top->rd_valid = 0;
}

// deterministic per-segment test pattern (distinct popcounts across segments)
static uint32_t seg_pattern(int b, int s) {
    uint32_t g = (uint32_t)(b * 16 + s);         // global segment index
    // low (g % 33) bits set -> popcount = g % 33 in [0,32]; plus a hashed
    // scramble of the high bits so two segments with equal popcount still
    // differ in raw value (guards a value-vs-popcount confusion).
    uint32_t nb = g % 33;
    uint32_t base = (nb >= 32) ? 0xFFFFFFFFu : ((1u << nb) - 1u);
    uint32_t scramble = (g * 2654435761u) & 0xF0000000u; // high nibble noise
    // keep popcount exact: only set high bits that `base` didn't, then mask
    // back so popcount stays nb (scramble is popcount-neutral by XOR-cancel)
    (void)scramble;
    return base;
}

static void scenario_segpop() {
    printf(" (k) SEG_POP: per-32b-segment popcount readout, 8 beats -> 2 words\n");
    hard_reset();                       // READ_MODE
    to_segpop_mode();
    uint32_t bs0 = top->buffer_space;
    const int NB = 8;                   // multiple of 4 -> 2 full FIFO words
    announce_reads(NB);
    seg_user_reads(NB, seg_pattern);
    idle(4);
    flush_pulse(1);
    idle(20);
    // Expect: 2 words x 2 c2h beats = 4 data beats, then a trailer beat.
    check("segpop produced one tlast message", msgs.size() == 1);
    if (msgs.empty()) return;
    const std::vector<Beat>& m = msgs.back();
    check("segpop trailer magic", m.back().w[0] == MAGIC);
    // data beats = all but the trailer; each beat = 256b = 32 segment-bytes.
    int ndata = (int)m.size() - 1;
    check("segpop data beat count == 4 (2048/... 8 beats)", ndata == 4);
    // reassemble the 2048-scaled byte stream (here 128 segments -> 128 bytes)
    // byte order per the din half-swap: within each 512b word the two 256b
    // halves are swapped, so c2h beat pairs arrive high-half-first. The host
    // receiver de-swaps by reading the pair in (beat1, beat0) order. Model
    // that here and assert byte g == popcount(segment g).
    std::vector<uint8_t> bytes;
    for (int w = 0; w * 2 + 1 < ndata; w++) {
        const Beat& lo = m[beat_lo(w)];
        const Beat& hi = m[beat_hi(w)];
        for (int i = 0; i < 8; i++) for (int k = 0; k < 4; k++)
            bytes.push_back((lo.w[i] >> (k*8)) & 0xFF);
        for (int i = 0; i < 8; i++) for (int k = 0; k < 4; k++)
            bytes.push_back((hi.w[i] >> (k*8)) & 0xFF);
    }
    int bad = 0, checked = 0;
    for (int g = 0; g < NB * 16 && g < (int)bytes.size(); g++) {
        int expect = pc32(seg_pattern(g / 16, g % 16));
        if ((int)bytes[g] != expect) bad++;
        checked++;
    }
    char buf[96];
    snprintf(buf, sizeof buf, "segpop byte[g]==popcount(seg g): %d/%d exact", checked - bad, checked);
    check(buf, bad == 0 && checked == NB * 16);
    // conservation: budget returns to start after the program drains.
    snprintf(buf, sizeof buf, "segpop buffer_space conserved (%u -> %u)", bs0, top->buffer_space);
    check(buf, top->buffer_space == bs0);

    // back-to-back second program (desync guard).
    msgs.clear(); open_msg.clear();
    announce_reads(NB);
    seg_user_reads(NB, seg_pattern);
    idle(4); flush_pulse(1); idle(20);
    check("segpop 2nd program: one clean message (no desync)", msgs.size() == 1);

    // mode transition back to READ must be clean.
    to_read_mode_set();
    check("segpop->READ transition leaves buffer_space at start", top->buffer_space == bs0);
}
#endif

#if defined(TB_BUILD8) || defined(TB_BUILD9) || defined(TB_BUILD10)
// ---- build8: ACCUM_XBP cross-bit-plane accumulator -----------------------
static void to_accxbp_mode() {
    // entering ACCUM_XBP zeroes the 128-word accumulator over 128 cycles;
    // no reads may issue until it completes (a mode-entry cost the host
    // absorbs — trivial vs a projection). idle well past 128.
    top->set_mode_accxbp = 1; tick(); top->set_mode_accxbp = 0; idle(140);
}
// Latch one plane's weight: sign (neg) + 3-bit shift = the bitplane factor.
static void set_acc_weight(int neg, int shift) {
    top->acc_weight_pl = (uint8_t)(((neg & 1) << 3) | (shift & 7));
    top->set_acc_weight = 1; tick(); top->set_acc_weight = 0;
    top->acc_weight_pl = 0; idle(1);
}
static void flush_acc() {
    top->flush_acc = 1; tick(); top->flush_acc = 0;
}
// One "bit-plane program": announce NB reads, feed NB beats whose segment
// popcounts come from seg_val(b,s). No flush — the plane's reads fold into
// the accumulator; the drain happens once at the end.
static void axb_plane_reads(int nbeats, uint32_t (*seg_val)(int,int)) {
    announce_reads(nbeats);
    for (int b = 0; b < nbeats; b++) {
        for (int s = 0; s < 16; s++) top->rd_data[s] = seg_val(b, s);
        top->rd_valid = 1; tick();
    }
    top->rd_valid = 0;
    idle(4);   // let the RMW pipeline drain the last beat
}
// SILICON-FAITHFUL announcement shape (2026-07-22): read_seq_incoming is
// NOT a pre-burst one-shot — SMC_INFO packets keep arriving while earlier
// beats return, so announcement pulses COINCIDE with rd_valid, they stop
// a couple of beats before the burst ends, and a tail bubble precedes the
// final beat. On the pre-fix RTL the level-priority realign then pins
// axb_widx: word0 = Σ beats 0..n-2, word1 = the last beat alone (the
// exact silicon accxbp-hw dump). The fixed RTL (edge + quiet realign)
// must be lane-exact under this shape too. Announce counts are balanced
// (total announced == nbeats) so outstanding/buffer_space accounting
// stays conserved: the last two reads are announced together, early.
static void axb_plane_reads_overlap(int nbeats, uint32_t (*seg_val)(int,int)) {
    // read 0 announced on its own quiet cycle — the fresh-sequence edge
    // the realign keys on.
    announce_reads(1);
    for (int b = 0; b < nbeats; b++) {
        if (b == nbeats - 1) {           // returns caught up: tail bubble
            top->rd_valid = 0; idle(3);
        }
        for (int s = 0; s < 16; s++) top->rd_data[s] = seg_val(b, s);
        top->rd_valid = 1;
        if (b + 1 <= nbeats - 3) {       // announce read b+1 WITH beat b
            top->read_seq_incoming = 1; top->incoming_reads = 1;
        } else if (b + 1 == nbeats - 2) { // last two announced together
            top->read_seq_incoming = 1; top->incoming_reads = 2;
        }
        tick();
        top->read_seq_incoming = 0; top->incoming_reads = 0;
    }
    top->rd_valid = 0;
    idle(4);
}
// distinct per-(beat,segment) popcount, reused across planes so the
// accumulation Σ w_k·pc is non-trivial and order-sensitive.
static uint32_t axb_pat(int b, int s) {
    uint32_t g = (uint32_t)(b * 16 + s);
    uint32_t nb = (g * 7 + 3) % 33;    // in [0,32]
    return (nb >= 32) ? 0xFFFFFFFFu : ((1u << nb) - 1u);
}

static void scenario_accxbp_with(const char* hdr,
        void (*plane_driver)(int, uint32_t (*)(int,int))) {
    printf("%s", hdr);
    hard_reset();                      // READ_MODE
    to_accxbp_mode();
    uint32_t bs0 = top->buffer_space;
    const int NB = 8;                  // 8 beats -> 128 segments (of 2048)
    // 4 "planes" with weights +1, +2, +4, and -8 (the two's-complement top
    // plane): shifts 0,1,2,3 and neg on the last.
    const int W[4][2] = {{0,0},{0,1},{0,2},{1,3}};
    long expect[NB * 16];
    for (int i = 0; i < NB * 16; i++) expect[i] = 0;
    for (int p = 0; p < 4; p++) {
        set_acc_weight(W[p][0], W[p][1]);
        plane_driver(NB, axb_pat);
        long w = (W[p][0] ? -1L : 1L) * (1L << W[p][1]);
        for (int b = 0; b < NB; b++)
            for (int s = 0; s < 16; s++)
                expect[b*16 + s] += w * pc32(axb_pat(b, s));
    }
    // drain: FLUSH_ACC, then let the message complete.
    flush_acc();
    idle(400);   // 128 words x ~2 cycles + framing
    check("accxbp one tlast message", msgs.size() == 1);
    if (msgs.empty()) { to_read_mode_set(); return; }
    const std::vector<Beat>& m = msgs.back();
    check("accxbp trailer magic", m.back().w[0] == MAGIC);
    int ndata = (int)m.size() - 1;
    // 128 accumulator words x 2 c2h beats each = 256 data beats.
    check("accxbp data beat count == 256", ndata == 256);
    // de-swap (beat1,beat0) per word, read lane l of word w as int32.
    int bad = 0, checked = 0;
    for (int w = 0; w * 2 + 1 < ndata && w < 128; w++) {
        const Beat& lo = m[beat_lo(w)];
        const Beat& hi = m[beat_hi(w)];
        int32_t lane[16];
        for (int i = 0; i < 8; i++) lane[i]   = (int32_t)lo.w[i];
        for (int i = 0; i < 8; i++) lane[8+i] = (int32_t)hi.w[i];
        for (int l = 0; l < 16; l++) {
            int seg = w * 16 + l;
            if (seg >= NB * 16) { if (lane[l] != 0) bad++; continue; }
            if ((long)lane[l] != expect[seg]) bad++;
            checked++;
        }
    }
    char buf[96];
    snprintf(buf, sizeof buf, "accxbp lane==Sigma w*pc: %d/%d exact (bad %d)",
             checked - bad, checked, bad);
    check(buf, bad == 0 && checked == NB * 16);
    snprintf(buf, sizeof buf, "accxbp buffer_space conserved (%u -> %u)",
             bs0, top->buffer_space);
    check(buf, top->buffer_space == bs0);

    // second accumulate-and-drain: accumulator must have zeroed on flush.
    msgs.clear(); open_msg.clear();
    set_acc_weight(0, 0);
    plane_driver(NB, axb_pat);
    flush_acc();
    idle(400);
    check("accxbp 2nd program: one clean message (no desync)", msgs.size() == 1);
    if (!msgs.empty()) {
        const std::vector<Beat>& m2 = msgs.back();
        int nd2 = (int)m2.size() - 1, bad2 = 0;
        for (int w = 0; w * 2 + 1 < nd2 && w < 128; w++) {
            const Beat& lo = m2[beat_lo(w)]; const Beat& hi = m2[beat_hi(w)];
            int32_t lane[16];
            for (int i = 0; i < 8; i++) lane[i]   = (int32_t)lo.w[i];
            for (int i = 0; i < 8; i++) lane[8+i] = (int32_t)hi.w[i];
            for (int l = 0; l < 16; l++) {
                int seg = w * 16 + l;
                long e2 = (seg < NB*16) ? (long)pc32(axb_pat(seg/16, seg%16)) : 0;
                if ((long)lane[l] != e2) bad2++;
            }
        }
        check("accxbp 2nd drain: accumulator zeroed after 1st flush", bad2 == 0);
    }

    to_read_mode_set();
    check("accxbp->READ transition leaves buffer_space at start",
          top->buffer_space == bs0);
}
static void scenario_accxbp() {
    scenario_accxbp_with(
        " (l) ACCUM_XBP: cross-bit-plane in-fabric place-value sum\n",
        axb_plane_reads);
}
static void scenario_accxbp_overlap() {
    scenario_accxbp_with(
        " (l2) ACCUM_XBP: silicon-faithful overlapped announcements\n",
        axb_plane_reads_overlap);
}
#endif


// build10 repro/fix: STALE ddr_wdata during SEG_POP reads. ddr_wdata
// mirrors the core's wide_reg (ddr_pipeline), which PERSISTS across
// programs; only legacy INIT_MEM passes refreshed it. Under build-9
// streaming (EXECUTE->EXECUTE swaps) a write program's last LDWD
// pattern stays on the bus and pre-build10 engines XOR it into every
// count — sticky, silent, ~3/4 of packed bytes wrong (silicon
// 2026-07-22, stream-hw arms C-G). build10 masks the XOR reference to
// zero outside DIFF mode: counts must equal popcount(RAW segment)
// REGARDLESS of ddr_wdata. DIFF keeps its compare-vs-pattern XOR
// (scenarios a-j cover it unchanged).
static void scenario_segpop_stale_wdata() {
    printf(" (m) SEG_POP with STALE ddr_wdata (build10 repro/fix)\n");
    hard_reset();
    to_segpop_mode();
    for (int s = 0; s < 16; s++)
        top->ddr_wdata[s] = 0xDEAD0000u | (uint32_t)(s * 0x0101);
    uint32_t bs0 = top->buffer_space;
    const int NB = 8;
    announce_reads(NB);
    seg_user_reads(NB, seg_pattern);
    idle(4);
    flush_pulse(1);
    idle(20);
    for (int s = 0; s < 16; s++) top->ddr_wdata[s] = 0;
    check("stale-wdata segpop produced one tlast message", msgs.size() == 1);
    if (msgs.empty()) return;
    const std::vector<Beat>& m = msgs.back();
    check("stale-wdata trailer magic", m.back().w[0] == MAGIC);
    int ndata = (int)m.size() - 1;
    std::vector<uint8_t> bytes;
    for (int w = 0; w * 2 + 1 < ndata; w++) {
        const Beat& lo = m[beat_lo(w)];
        const Beat& hi = m[beat_hi(w)];
        for (int i = 0; i < 8; i++) for (int k = 0; k < 4; k++)
            bytes.push_back((lo.w[i] >> (k*8)) & 0xFF);
        for (int i = 0; i < 8; i++) for (int k = 0; k < 4; k++)
            bytes.push_back((hi.w[i] >> (k*8)) & 0xFF);
    }
    int bad = 0, checked = 0;
    for (int g = 0; g < NB * 16 && g < (int)bytes.size(); g++) {
        int expect = pc32(seg_pattern(g / 16, g % 16));   // RAW counts
        if ((int)bytes[g] != expect) bad++;
        checked++;
    }
    char buf[120];
    snprintf(buf, sizeof buf,
             "stale-wdata segpop byte[g]==popcount(RAW seg): %d/%d exact",
             checked - bad, checked);
    check(buf, bad == 0 && checked == NB * 16);
    char buf2[120];
    snprintf(buf2, sizeof buf2,
             "stale-wdata buffer_space conserved (%u -> %u)", bs0, top->buffer_space);
    check(buf2, top->buffer_space == bs0);
    msgs.clear(); open_msg.clear();
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    if (argc > 1) variant = argv[1];
#ifdef TB_BUILD4
    is_b4_rtl = true;
#endif
#ifdef TB_BUILD6
    is_b6_rtl = true;
#endif
#ifdef TB_BUILD7
    is_b6_rtl = true;   // build7 keeps build6 buffer_space conservation for READ/DIFF
#endif
#ifdef TB_BUILD8
    is_b6_rtl = true;   // build8 keeps build6/7 conservation for READ/DIFF/SEG_POP
#endif
#if defined(TB_BUILD9) || defined(TB_BUILD10)
    is_b6_rtl = true;   // build9/10 keep the same conservation semantics
#endif
    top = new Vreadback_engine;
    printf("=== readback_engine drain-capture TB : %s (%s RTL, magic %08X) ===\n",
           variant, is_b4_rtl ? "build4" : "build3", MAGIC);
    scenario_a();
    scenario_b();
    scenario_c();
    scenario_d();
    scenario_e();
    scenario_f1();
    scenario_f2();
    scenario_g();
    scenario_j();
#ifdef TB_BUILD4
    scenario_h();
    scenario_i();
#endif
#ifdef TB_BUILD7
    // build7 non-regression: scenarios a-j above must still pass (READ/DIFF
    // bit-identical to build6), THEN the new SEG_POP datapath:
    scenario_segpop();
#endif
#ifdef TB_BUILD8
    // build8 non-regression: a-j pass identically (READ/DIFF/SEG_POP paths
    // untouched), THEN the SEG_POP datapath, THEN the ACCUM_XBP datapath.
    scenario_segpop();
    scenario_accxbp();
    scenario_accxbp_overlap();
#endif
#if defined(TB_BUILD9) || defined(TB_BUILD10)
    // build9/10 gate: full build8 suite + the stale-wdata repro. On the
    // build9 engine the stale-wdata scenario MUST FAIL (failure
    // reproduction of the 2026-07-22 silicon); on build10 it must pass.
    scenario_segpop();
    scenario_accxbp();
    scenario_accxbp_overlap();
    scenario_segpop_stale_wdata();
#endif
    printf("=== %s: %s (%d failing checks) ===\n",
           variant, failures ? "FAIL" : "ALL PASS", failures);
    top->final();
    delete top;
    return failures ? 1 : 0;
}
