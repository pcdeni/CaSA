// V2 Road-B silicon validation: pop_count4 0xe fix + popcount_accum drain.
//
// The flashed 2026-07-18 bitstream compiles readback_engine.v with
// POPCOUNT_ACCUM_MODE: in DIFF_MODE the popcount channel no longer streams
// per-read 16b popcounts — it accumulates popcount(rd_data ^ ddr_wdata)
// across the whole program and drains ONE 32-bit total (in a single 512b
// chunk, 64 B c2h) at program flush. READ_MODE (full row data) unchanged.
//
// Per case: wrRow(P) into <row>; LI(Q, PATTERN_REG); rdRow(row); execute
// in DIFF_MODE; receiveData(64); total must equal
//   2048 * popcount32(P ^ Q)      (8 KB row = 128 beats x 512 b; each
//                                  beat = 16 replicas of the 32b diff)
// Case list hits every diff nibble 0x0..0xf — including 0xe, the
// pop_count4 missing-case bug (undercount on the OLD image) — plus
// random patterns.
//
// Argv: ./popcount-hw-exe <bender_id> <bank_id> <row> [row2=row+1]
// Exit 0 iff all cases exact and normal readback works after toggling back.
#include "instruction.h"
#include "prog.h"
#include "platform.h"
#include "../util.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

using namespace std;

static int popcount32(uint32_t v) { return __builtin_popcount(v); }

// One write program: pattern P into row (uniform 32b immediate).
static void write_row(SoftMCPlatform& platform, int bank, uint32_t row,
                      uint32_t P, int label) {
  Program p;
  p.add_inst(SMC_LI(bank, BAR));
  p.add_inst(SMC_LI(8, CASR));
  p.add_inst(SMC_LI(128, NUM_COLS_REG));
  p.add_below(wrRow_immediate_label(BAR, row, P, label));
  p.add_inst(SMC_SLEEP(8));
  p.add_inst(SMC_END());
  platform.execute(p);
}

// One read program: LI the compare pattern Q, then read the row.
static void read_row_program(SoftMCPlatform& platform, int bank, uint32_t row,
                             uint32_t Q, int label) {
  Program p;
  p.add_inst(SMC_LI(bank, BAR));
  p.add_inst(SMC_LI(8, CASR));
  p.add_inst(SMC_LI(128, NUM_COLS_REG));
  p.add_inst(SMC_LI((int)Q, PATTERN_REG));   // ddr_wdata = replicated Q
  p.add_below(rdRow_immediate_label(BAR, row, label));
  p.add_inst(SMC_END());
  platform.execute(p);
}

// One COMBINED diff-case program: write P, load Q into the wdata buffer
// (ddr_wdata — the readback compare reference comes from the 16 SMC_LDWD
// slots, NOT from PATTERN_REG directly; first tool version missed this
// and diffed P^P=0), read the row, END. Exactly ONE program flush →
// exactly ONE accum drain chunk per case — a separate write program's
// empty flush would emit its own 64 B chunk in DIFF_MODE and desync the
// c2h stream (which is what hung version 1's final read).
static void diff_case_program(SoftMCPlatform& platform, int bank,
                              uint32_t row, uint32_t P, uint32_t Q,
                              int label) {
  Program p;
  p.add_inst(SMC_LI(bank, BAR));
  p.add_inst(SMC_LI(8, CASR));
  p.add_inst(SMC_LI(128, NUM_COLS_REG));
  p.add_below(wrRow_immediate_label(BAR, row, P, label));
  p.add_inst(SMC_SLEEP(8));
  p.add_inst(SMC_LI((int)Q, PATTERN_REG));
  for (int i = 0; i < 16; i++)
    p.add_inst(SMC_LDWD(PATTERN_REG, i));    // ddr_wdata := replicated Q
  p.add_below(rdRow_immediate_label(BAR, row, label + 100));
  p.add_inst(SMC_END());
  platform.execute(p);
}

// build4 mode switch: idempotent SET-READ/SET-DIFF control words (byte8
// 0x20/0x40 via platform.set_readback_mode). These are decode-state safe
// (a lost word is repaired by resending; a duplicate is a no-op), unlike
// the legacy toggle whose parity flips on any lost/doubled word.
//
// NOTE (2026-07-20): the OLD verified-toggle probe here used a NO-READ
// armed WRITE program and asserted "DIFF emits a 64 B chunk". On silicon
// that probe is unreliable: constant per_rd_init maintenance (~1 M/s)
// keeps ignore_flush_ctr>=1, so a no-read DIFF program's flush is EATEN
// (accum_armed_r==0) and emits NOTHING — indistinguishable from READ. We
// send the SET word idempotently twice and trust it (build4).
static bool set_diff_mode(SoftMCPlatform& platform, int bank, uint32_t row,
                          bool want_diff, int* label) {
  (void)bank; (void)row; (void)label;
  platform.set_readback_mode(want_diff);
  platform.set_readback_mode(want_diff);   // idempotent; repairs a lost word
  return true;
}

int main(int argc, char** argv) {
  if (argc < 4) {
    cerr << "Usage: " << argv[0] << " <bender_id> <bank_id> <row>" << endl;
    return 1;
  }
  setvbuf(stdout, NULL, _IONBF, 0);   // progress visible even if a later receive hangs
  int bender_id = atoi(argv[1]);
  int bank      = atoi(argv[2]);
  uint32_t row  = (uint32_t)atoi(argv[3]);
  bool diag     = (argc >= 5 && strcmp(argv[4], "diag") == 0);
  bool diag2    = (argc >= 5 && strcmp(argv[4], "diag2") == 0);
  bool root     = (argc >= 5 && strcmp(argv[4], "root") == 0);
  bool lag      = (argc >= 5 && strcmp(argv[4], "lag") == 0);

  SoftMCPlatform platform(bender_id);
  if (platform.init() != SOFTMC_SUCCESS) {
    cerr << "[pchw] platform init failed" << endl;
    return 1;
  }
  platform.reset_fpga();
  // Periodic maintenance inits (per_rd/zq/ref) race user flushes for the
  // ignore_flush slot — an eaten flush loses the drain chunk AND the
  // trailer (boot-phase-dependent: worked/alternated/failed across three
  // flashes of the same .bit). Disable the source explicitly.
  platform.set_aref(false);

  // Recovery mode: force READ mode (SET word) + bounded-drain any stale
  // c2h beats left by an unclean DIFF run, WITHOUT the 8 KB read that would
  // hang on a desynced stream. Use after a batch-all-execute desync.
  if (argc >= 5 && strcmp(argv[4], "recover") == 0) {
    platform.set_readback_mode(false);
    platform.set_readback_mode(false);
    for (int s = 0; s < 12; s++) {
      vector<uint8_t> stray(4096);
      int rx = platform.receiveDataTry(stray.data(), 4096, 1500);
      printf("[recover] drain %d: %d bytes%s\n", s, rx, platform.recv_stalled() ? " (poisoned)" : "");
      if (platform.recv_stalled()) break;
    }
    printf("[recover] done (mode=READ, stream drained)\n");
    return 0;
  }

  int label = 1000;
  vector<uint8_t> buf(8192);

  // 0. READ_MODE sanity: write a marker, read the full row back.
  write_row(platform, bank, row, 0xA5A5A5A5u, label); label += 200;
  read_row_program(platform, bank, row, 0, label); label += 200;
  platform.receiveData(buf.data(), 8192);
  int bad = 0;
  for (int i = 0; i < 8192; i += 4) {
    uint32_t v; memcpy(&v, buf.data() + i, 4);
    if (v != 0xA5A5A5A5u) bad++;
  }
  printf("[pchw] READ_MODE sanity: %d/2048 words wrong (expect 0)\n", bad);
  if (bad) { fprintf(stderr, "[pchw] row unusable, pick another\n"); return 1; }

  // Diff cases: {P, Q}. P^Q covers every nibble incl. 0xe (the fix).
  struct Case { uint32_t P, Q; const char* note; };
  vector<Case> cases = {
    {0x00000000u, 0x00000000u, "zero diff"},
    {0xEEEEEEEEu, 0x00000000u, "nibble 0xE — the pop_count4 fixed case"},
    {0x00000000u, 0xEEEEEEEEu, "0xE via Q side"},
    {0xFFFFFFFFu, 0x00000000u, "all ones"},
    {0x01234567u, 0x00000000u, "nibbles 0-7"},
    {0x89ABCDEFu, 0x00000000u, "nibbles 8-F"},
    {0xA5A5A5A5u, 0x5A5A5A5Au, "alternating -> 0xFF diff"},
    {0xDEADBEEFu, 0x12345678u, "random-ish 1"},
    {0xC0FFEE00u, 0x00EEFF0Cu, "random-ish 2"},
  };

  if (!set_diff_mode(platform, bank, row, true, &label)) return 1;

  if (argc >= 5 && strcmp(argv[4], "count") == 0) {
    // Wedge-driver probe: run many identical DIFF read programs and count
    // how many chunks deliver before the c2h wedges. If the buffer_space
    // accum-accounting bug is the cause, chunks-before-wedge scales
    // INVERSELY with the per-program read count (NUM_COLS): ~2048 /
    // (2*reads - 2). Env PCHW_COLS sets NUM_COLS (default 128).
    int cols = 128; if (const char* e = getenv("PCHW_COLS")) cols = atoi(e);
    printf("[count] NUM_COLS=%d  (predict ~%d chunks before wedge if buffer_space bug)\n",
           cols, 2048 / (2*cols - 2 > 0 ? 2*cols - 2 : 1));
    int delivered = 0;
    for (int prog = 0; prog < 60; prog++) {
      Program p;
      p.add_inst(SMC_LI(bank, BAR)); p.add_inst(SMC_LI(8, CASR));
      p.add_inst(SMC_LI(cols, NUM_COLS_REG));
      p.add_below(wrRow_immediate_label(BAR, row, 0xEEEEEEEEu, 6000 + prog));
      p.add_inst(SMC_SLEEP(8));
      p.add_inst(SMC_LI(0, PATTERN_REG));
      for (int i = 0; i < 16; i++) p.add_inst(SMC_LDWD(PATTERN_REG, i));
      p.add_below(rdRow_immediate_label(BAR, row, 6600 + prog));
      p.add_inst(SMC_END());
      platform.execute(p);
      if (platform.recv_stalled()) { printf("[count] platform poisoned at prog %d\n", prog); break; }
      // drain whatever is available
      for (int g = 0; g < 4; g++) {
        vector<uint8_t> a(64, 0xFF);
        int rx = platform.receiveDataTry(a.data(), 64, 300);
        if (rx != 64) break;
        delivered++;
      }
    }
    printf("[count] NUM_COLS=%d -> delivered %d chunks before stream dried up\n", cols, delivered);
    platform.set_readback_mode(false);
    return 0;
  }

  if (lag) {
    // Interleaved execute+receive to localize WHERE a read program's total
    // drains: at its OWN flush (in-place) or the NEXT program's (lag).
    // PCHW_SLEEP>0 drains the read tail before END.
    uint32_t sc = 0; if (const char* e = getenv("PCHW_SLEEP")) sc = (uint32_t)atol(e);
    auto compute_prog = [&](uint32_t P, int lab) {
      Program p;
      p.add_inst(SMC_LI(bank, BAR)); p.add_inst(SMC_LI(8, CASR)); p.add_inst(SMC_LI(128, NUM_COLS_REG));
      p.add_below(wrRow_immediate_label(BAR, row, P, lab));
      p.add_inst(SMC_SLEEP(8));
      p.add_inst(SMC_LI(0, PATTERN_REG));
      for (int i = 0; i < 16; i++) p.add_inst(SMC_LDWD(PATTERN_REG, i));
      p.add_below(rdRow_immediate_label(BAR, row, lab + 100));
      if (sc) p.add_inst(SMC_SLEEP(sc));
      p.add_inst(SMC_END());
      platform.execute(p);
    };
    auto recv1 = [&](const char* when) {
      vector<uint8_t> a(64, 0xFF); int rx = platform.receiveDataTry(a.data(), 64, 4000);
      if (platform.recv_stalled()) { printf("[lag] %-22s POISONED\n", when); return; }
      if (rx != 64) { printf("[lag] %-22s no-chunk (short=%d)\n", when, rx); return; }
      uint32_t v = 0; for (int i=0;i<64;i+=4){uint32_t w;memcpy(&w,a.data()+i,4);if(w&&w!=0xFFFFFFFFu)v=w;}
      printf("[lag] %-22s chunk=%u %s\n", when, v,
             v==49152u?"(A)":v==16384u?"(B)":v==0?"(zero)":"(?)");
    };
    printf("[lag] PCHW_SLEEP=%u\n", sc);
    compute_prog(0xEEEEEEEEu, 3000);              // A -> 49152
    recv1("after A");                             // in-place? -> 49152 ; lag? -> no-chunk
    compute_prog(0x000000FFu, 3400);              // B -> 16384
    recv1("after B");                             // lag? -> 49152 (A drained at B's flush)
    write_row(platform, bank, row, 0u, 3800);     // kicker (no reads)
    recv1("after kicker1");
    write_row(platform, bank, row, 0u, 3850);
    recv1("after kicker2");
    platform.set_readback_mode(false);
    return 0;
  }

  if (root) {
    // ROOT-CAUSE PROBE (2026-07-20). Hypothesis: on silicon a read
    // program's DDR data returns AFTER the fin+32 flush edge (addendum
    // 20d late tail). At that edge accum_armed_r==0, and the ctr==0
    // fallback that makes flush_proc fire in the maintenance-free sim
    // does NOT hold on silicon (per_rd_init ~1M/s keeps ignore_flush_ctr
    // >=1) — so the read program's flush is EATEN, capture_pending is
    // never set, and build4's deferred capture never arms. A big
    // SMC_SLEEP BEFORE the END makes the reads finish (and accumulate,
    // arming accum_armed_r) BEFORE the flush edge, so flush_proc fires
    // via the armed term and the capture drains the correct total in the
    // program's OWN chunk.
    //
    // Decisive: NOSLEEP should yield 0 / one-program lag; SLEEP should
    // yield the exact total in-place. Two distinguishable payloads per
    // pass so we can see WHERE each total lands:
    //   A: P=0xEEEEEEEE^Q=0 -> pc32=24 -> total 49152
    //   B: P=0x000000FF^Q=0 -> pc32=8  -> total 16384
    uint32_t sleep_cyc = 0;
    if (const char* e = getenv("PCHW_SLEEP")) sleep_cyc = (uint32_t)atol(e);
    struct RC { uint32_t P, Q, exp; const char* tag; };
    RC seq[2] = {
      {0xEEEEEEEEu, 0u, 49152u, "A(0xE->49152)"},
      {0x000000FFu, 0u, 16384u, "B(0xFF->16384)"},
    };
    auto compute_prog = [&](uint32_t P, uint32_t Q, uint32_t sc, int lab) {
      Program p;
      p.add_inst(SMC_LI(bank, BAR));
      p.add_inst(SMC_LI(8, CASR));
      p.add_inst(SMC_LI(128, NUM_COLS_REG));
      p.add_below(wrRow_immediate_label(BAR, row, P, lab));
      p.add_inst(SMC_SLEEP(8));                    // tWR after write
      p.add_inst(SMC_LI((int)Q, PATTERN_REG));
      for (int i = 0; i < 16; i++) p.add_inst(SMC_LDWD(PATTERN_REG, i));
      p.add_below(rdRow_immediate_label(BAR, row, lab + 100));
      if (sc) p.add_inst(SMC_SLEEP(sc));           // <-- drain the read tail
                                                   //     BEFORE END/flush
      p.add_inst(SMC_END());
      platform.execute(p);
    };
    printf("[root] PCHW_SLEEP=%u  (0 = no tail-drain sleep; big = drain reads before flush)\n",
           sleep_cyc);
    // two compute programs back-to-back, then 3 no-op EOP kickers to push
    // any lagged/stranded chunk out of the FIFO.
    compute_prog(seq[0].P, seq[0].Q, sleep_cyc, 2000);
    compute_prog(seq[1].P, seq[1].Q, sleep_cyc, 2400);
    for (int k = 0; k < 3; k++) { write_row(platform, bank, row, 0u, 2800 + k*50); }
    // Drain every chunk that shows up; print each nonzero value + position.
    printf("[root] receiving chunks (value at each position):\n");
    int pos = 0, seen_nonzero = 0;
    for (int h = 0; h < 8; h++) {
      vector<uint8_t> acc(64, 0xFF);
      int rx = platform.receiveDataTry(acc.data(), 64, 6000);
      if (platform.recv_stalled()) { printf("[root]   pos %d: platform poisoned\n", pos); break; }
      if (rx != 64) { printf("[root]   pos %d: (no chunk / short=%d)\n", pos, rx); pos++; continue; }
      uint32_t v = 0; int nz = 0;
      for (int i = 0; i < 64; i += 4) { uint32_t w; memcpy(&w, acc.data()+i, 4); if (w && w != 0xFFFFFFFFu) { v = w; nz++; } }
      const char* label_s = (v == 49152u) ? " <= A total" : (v == 16384u) ? " <= B total" : "";
      printf("[root]   pos %d: chunk value = %u (nz words=%d)%s\n", pos, v, nz, label_s);
      if (v) seen_nonzero++;
      pos++;
    }
    printf("[root] VERDICT: sleep=%u  nonzero_chunks=%u  (expect A=49152 and B=16384 to appear)\n",
           sleep_cyc, seen_nonzero);
    platform.set_readback_mode(false);
    return 0;
  }

  if (diag2) {
    // Image-identity fingerprint. OLD (pre-accum) bitstream DIFF mode
    // STREAMS per-read 16b popcounts: a 128-read program yields 4×512b
    // chunks = 256 useful bytes. NEW accum image yields 64. Sequence:
    // read-only program → recv(64) → recv(192).
    //   both succeed          → OLD image flashed (streaming path).
    //   64 ok, 192 times out  → NEW image, accumulator WORKS read-only.
    //   64 times out          → NEW image, drain emits nothing (HDL bug).
    uint32_t P = 0xEEEEEEEEu, Q = 0;
    write_row(platform, bank, row, P, 7000);
    // deliberately do NOT receive the write program's (possible) drain —
    // recv(64) below tolerates either a leading zero-chunk or the read
    // chunk; we only fingerprint SIZES here.
    {
      Program p;
      p.add_inst(SMC_LI(bank, BAR));
      p.add_inst(SMC_LI(8, CASR));
      p.add_inst(SMC_LI(128, NUM_COLS_REG));
      p.add_inst(SMC_LI((int)Q, PATTERN_REG));
      for (int i = 0; i < 16; i++) p.add_inst(SMC_LDWD(PATTERN_REG, i));
      p.add_below(rdRow_immediate_label(BAR, row, 7600));
      p.add_inst(SMC_END());
      platform.execute(p);
    }
    vector<uint8_t> b1(64, 0xFF);
    platform.receiveData(b1.data(), 64);
    bool got64 = false; for (int i = 0; i < 64; i += 4) { uint32_t v; memcpy(&v, b1.data()+i, 4); if (v != 0xFFFFFFFFu) { got64 = true; break; } }
    printf("[diag2] recv(64): %s\n", got64 ? "DATA" : "TIMEOUT/untouched");
    if (got64 && !platform.recv_stalled()) {
      uint32_t w0; memcpy(&w0, b1.data(), 4);
      vector<uint8_t> b2(192, 0xFF);
      platform.receiveData(b2.data(), 192);
      bool got192 = false; for (int i = 0; i < 192; i += 4) { uint32_t v; memcpy(&v, b2.data()+i, 4); if (v != 0xFFFFFFFFu) { got192 = true; break; } }
      printf("[diag2] first word=%u; recv(192): %s → %s\n", w0,
             got192 ? "DATA" : "TIMEOUT",
             got192 ? "OLD streaming image is flashed"
                    : "NEW accum image, read-only drain WORKS");
    } else {
      printf("[diag2] verdict: NEW image, drain emits nothing even read-only (HDL bug)\n");
    }
    platform.toggle_readback_mode();
    return 0;
  }

  if (diag) {
    // Single-case discriminator: SEPARATE write and read programs (v1
    // shape) with the LDWD reference fix. Two drains → two 64 B receives:
    //   1) the write program's empty drain (expect a zero chunk — v1
    //      showed with-no-reads drains DO emit),
    //   2) the read-only program's drain (expect 49152 for 0xE vs 0).
    // If (2) arrives → v3's combined write+read program shape is the
    // problem, HDL fine. If (2) times out → drain-after-reads never
    // emits on silicon (sim-vs-silicon HDL bug).
    uint32_t P = 0xEEEEEEEEu, Q = 0;
    write_row(platform, bank, row, P, 5000);
    vector<uint8_t> a1(64, 0xFF);
    platform.receiveData(a1.data(), 64);
    int nz1 = 0; for (int i = 0; i < 64; i += 4) { uint32_t v; memcpy(&v, a1.data()+i, 4); if (v) nz1++; }
    printf("[diag] write-program drain: nz=%d (16=timeout/untouched, 0=zero chunk arrived)\n", nz1);
    {
      Program p;
      p.add_inst(SMC_LI(bank, BAR));
      p.add_inst(SMC_LI(8, CASR));
      p.add_inst(SMC_LI(128, NUM_COLS_REG));
      p.add_inst(SMC_LI((int)Q, PATTERN_REG));
      for (int i = 0; i < 16; i++) p.add_inst(SMC_LDWD(PATTERN_REG, i));
      p.add_below(rdRow_immediate_label(BAR, row, 5600));
      p.add_inst(SMC_END());
      platform.execute(p);
    }
    vector<uint8_t> a2(64, 0xFF);
    platform.receiveData(a2.data(), 64);
    uint32_t got = 0; int nz2 = 0;
    for (int i = 0; i < 64; i += 4) { uint32_t v; memcpy(&v, a2.data()+i, 4); if (v && v != 0xFFFFFFFFu) { got = v; } if (v) nz2++; }
    printf("[diag] read-program drain: nz=%d got=%u expect=49152  %s\n", nz2, got,
           (nz2 == 1 && got == 49152) ? "OK — HDL fine, tool shape was the bug"
                                      : (nz2 == 16 ? "TIMEOUT — drain-after-reads never emits (HDL bug)"
                                                   : "UNEXPECTED — see bytes"));
    platform.toggle_readback_mode();
    return 0;
  }

  // ------------------------------------------------------------------
  // build4 accum delivery model (MEASURED on silicon 2026-07-20, tower
  // bender 2; trailer cnt_drain/cnt_accum_write correlated to program
  // boundaries — RESULT.md addendum 20e):
  //   * Every DIFF program that CONTAINS READS drains its popcount total
  //     at its OWN flush. cnt_drain increments 1:1 with read programs;
  //     the accum_armed override (build3) makes a read program's flush
  //     immune to the maintenance ignore_flush_ctr race. Totals are EXACT
  //     (0xE -> 49152 etc. verified in-place).
  //   * The 96 B (64 B chunk + 32 B trailer) message is NOT pushed over
  //     c2h until the NEXT program's EOP — a one-program XDMA small-
  //     transfer delivery lag. (The 8 KB READ_MODE payload is large
  //     enough to push immediately, which is why READ never shows it.)
  //     So the chunk for program[i] arrives while program[i+1] runs.
  //   * BARE no-read programs (plain writes) are subject to the ctr race
  //     (per_rd_init ~1 M/s keeps ignore_flush_ctr>=1) and are usually
  //     EATEN -> no chunk. The OLD tool assumed writes/kickers emit a
  //     zero chunk (3 chunks/case) — that assumption is what failed, not
  //     the datapath.
  //
  // Correct consumption: make EVERY case a self-contained READ program
  // (write P + LDWD Q + rdRow), execute all N back-to-back, append TWO
  // trailing READ kickers to push case[N-1] (and N-2) out of the c2h
  // pipe, then receive N chunks IN ORDER — chunk[i] carries case[i]'s
  // total. This is exactly how a batched server integration should drain
  // (N matmul programs + a flush kicker -> N totals in order).
  int fails = 0;
  const int N = (int)cases.size();
  // INTERLEAVED execute+receive (NOT batch-all-then-receive, which desyncs
  // the c2h stream and zeroes everything — measured). Because of the
  // one-program delivery lag, exactly ONE receiveDataTry after program[k]
  // yields program[k-1]'s chunk. We execute cases[0..N-1] then TWO trailing
  // read-kickers (total 0), doing one receive after each execute, so the
  // delivered sequence is:
  //   out[0] = lag-fill (case_0 still held) -> no chunk
  //   out[1..N] = case_0 .. case_{N-1} totals, IN ORDER
  //   out[N+1] = kicker0 total (0)
  // PCHW_SLEEP>0 pads each read program with an SMC_SLEEP before END so the
  // read tail finishes (and accum_armed_r asserts) BEFORE the fin+32 flush
  // edge — the diagnostic that isolates the "late-tail flush eaten under
  // constant maintenance" RTL deficiency (RESULT.md addendum 20e).
  uint32_t suite_sleep = 0;
  if (const char* e = getenv("PCHW_SLEEP")) suite_sleep = (uint32_t)atol(e);
  auto exec_recv = [&](uint32_t P, uint32_t Q, int lab, long tmo) -> long {
    if (suite_sleep) {
      Program p;
      p.add_inst(SMC_LI(bank, BAR)); p.add_inst(SMC_LI(8, CASR)); p.add_inst(SMC_LI(128, NUM_COLS_REG));
      p.add_below(wrRow_immediate_label(BAR, row, P, lab));
      p.add_inst(SMC_SLEEP(8));
      p.add_inst(SMC_LI((int)Q, PATTERN_REG));
      for (int i = 0; i < 16; i++) p.add_inst(SMC_LDWD(PATTERN_REG, i));
      p.add_below(rdRow_immediate_label(BAR, row, lab + 100));
      p.add_inst(SMC_SLEEP(suite_sleep));
      p.add_inst(SMC_END());
      platform.execute(p);
    } else {
      diff_case_program(platform, bank, row, P, Q, lab);
    }
    vector<uint8_t> acc(64, 0xFF);
    int rx = platform.receiveDataTry(acc.data(), 64, tmo);
    if (rx != 64) return -1;
    uint32_t v = 0; int nz = 0;
    for (int j = 0; j < 64; j += 4) { uint32_t w; memcpy(&w, acc.data()+j, 4);
      if (w && w != 0xFFFFFFFFu) { v = w; nz++; } }
    return (nz <= 1) ? (long)v : -2;   // -2 = malformed (multi-nonzero)
  };
  (void)exec_recv;   // superseded by drain-all consumption below
  // The c2h small-transfer delivery is LUMPY and lags several programs: a
  // program's 96 B message is not pushed to the host until later programs'
  // c2h traffic flushes the XDMA path. A single receive per program
  // strands the tail. Robust consumption: execute a program, then DRAIN
  // ALL currently-available chunks (short-timeout loop), collecting every
  // delivered value IN ORDER; over-provision trailing read-kickers so the
  // last real cases get flushed out. Then match the 9 totals as the
  // ordered delivered sequence (FIFO order == execution order).
  vector<long> stream;
  auto build_diff = [&](uint32_t P, uint32_t Q, int lab) {
    if (suite_sleep) {
      Program p;
      p.add_inst(SMC_LI(bank, BAR)); p.add_inst(SMC_LI(8, CASR)); p.add_inst(SMC_LI(128, NUM_COLS_REG));
      p.add_below(wrRow_immediate_label(BAR, row, P, lab));
      p.add_inst(SMC_SLEEP(8));
      p.add_inst(SMC_LI((int)Q, PATTERN_REG));
      for (int i = 0; i < 16; i++) p.add_inst(SMC_LDWD(PATTERN_REG, i));
      p.add_below(rdRow_immediate_label(BAR, row, lab + 100));
      p.add_inst(SMC_SLEEP(suite_sleep));
      p.add_inst(SMC_END());
      platform.execute(p);
    } else diff_case_program(platform, bank, row, P, Q, lab);
  };
  auto drain_all = [&](long first_tmo) {
    bool first = true;
    for (int guard = 0; guard < 16; guard++) {
      vector<uint8_t> a(64, 0xFF);
      int rx = platform.receiveDataTry(a.data(), 64, first ? first_tmo : 250);
      first = false;
      if (rx != 64 || platform.recv_stalled()) break;
      uint32_t v = 0; int nz = 0;
      for (int j = 0; j < 64; j += 4) { uint32_t w; memcpy(&w, a.data()+j, 4);
        if (w && w != 0xFFFFFFFFu) { v = w; nz++; } }
      stream.push_back(nz <= 1 ? (long)v : -2);
    }
  };
  for (int i = 0; i < N; i++) { build_diff(cases[i].P, cases[i].Q, label); label += 200; drain_all(50); }
  // Trailing read-kickers (total 0) to flush the tail out of the c2h path.
  for (int k = 0; k < 8; k++) { build_diff(0u, 0u, label); label += 200; drain_all(1500); }
  // Match: the 9 case totals are the first N values in `stream` (delivery
  // order == execution order; leading lag-fill produced no chunk so is
  // absent). Report and also dump the raw stream for diagnosis.
  printf("[pchw] delivered stream (%zu chunks):", stream.size());
  for (long v : stream) printf(" %ld", v);
  printf("\n");
  for (int i = 0; i < N; i++) {
    uint32_t expect = 2048u * (uint32_t)popcount32(cases[i].P ^ cases[i].Q);
    long got = (i < (int)stream.size()) ? stream[i] : -1;
    bool ok = (got == (long)expect);
    printf("[pchw] P=%08X Q=%08X diff-pc32=%2d expect=%6u got=%6ld  %s  (%s)\n",
           cases[i].P, cases[i].Q, popcount32(cases[i].P ^ cases[i].Q), expect, got,
           ok ? "OK" : "MISMATCH", cases[i].note);
    if (!ok) fails++;
  }
  // Drain any straggler chunks so the toggle-back READ sanity starts clean.
  for (int s = 0; s < 6; s++) {
    vector<uint8_t> stray(64);
    (void)platform.receiveDataTry(stray.data(), 64, 2000);
  }

  if (!set_diff_mode(platform, bank, row, false, &label)) return 1;  // -> READ_MODE, verified

  // Final READ_MODE sanity after the round trip.
  write_row(platform, bank, row, 0x3C3C3C3Cu, label); label += 200;
  read_row_program(platform, bank, row, 0, label); label += 200;
  platform.receiveData(buf.data(), 8192);
  bad = 0;
  for (int i = 0; i < 8192; i += 4) {
    uint32_t v; memcpy(&v, buf.data() + i, 4);
    if (v != 0x3C3C3C3Cu) bad++;
  }
  printf("[pchw] READ_MODE after toggle-back: %d/2048 words wrong (expect 0)\n", bad);
  if (bad) fails++;

  printf("[pchw] %s (%d fails)\n", fails ? "FAIL" : "ALL_PASS", fails);
  return fails ? 1 : 0;
}
