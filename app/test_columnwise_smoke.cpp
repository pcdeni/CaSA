// BitNet PIM SW demo — Phase 2a: prove per-column non-uniform writes.
//
// Goal: write an arbitrary 8192-byte buffer to a single DRAM row, then
// read it back byte-exact. Foundation for Phase 2b's dense MatVec where
// each row holds 2048 distinct 32-bit weight segments (one per output
// neuron) instead of one uniform repeating pattern.
//
// Per-column instruction count (one column = 64 bytes = 16×32-bit slots):
//     16× (SMC_LI value → PATTERN_REG, SMC_LDWD PATTERN_REG → slot)
//     1×  WRITE (2 insts: packed mininst quad + all_nops)
//   = 34 insts per column.
//
// One row has 128 columns → 128×34 + setup ≈ 4380 insts, far over the
// 2048-inst SoftMC instruction buffer. So we split the row into three
// "chunks" of ~43 columns each (~1500 insts per execute, fits) and emit
// a separate `platform.execute()` per chunk. Each chunk re-precharges
// and re-activates the row, sets CAR to the chunk's start column, and
// runs its WRITE-loop.
//
// Reference row picked inside the calibrated subarray window (from
// all_subarrays.csv) at row 38500 — well outside the calibrated open-row
// window (38786..39028) so it doesn't conflict with Phase 1 demo rows.
//
// Argv:
//   ./columnwise-smoke <bender_id> <bank_id> <row_addr> [seed]
#include "instruction.h"
#include "prog.h"
#include "platform.h"
#include "../util.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>
#include <random>
#include <string>
#include <unistd.h>
#include <vector>

using namespace std;

// Three chunks of columns: 43 + 43 + 42 = 128. Each chunk fits in the
// 2048-inst SoftMC buffer (43*34 + ~10 setup = 1472 insts < 2048).
static const int CHUNK_COLS[3] = {43, 43, 42};
static const int N_COLS_TOTAL  = 128;

// Build one chunk-write Program. Each chunk re-precharges and re-activates
// the row, then unrolls n_cols column writes starting at col_start (in
// physical column units; CAR is set to col_start * CASR=8).
static Program build_chunk_program(int bank_id, uint32_t row_addr,
                                    const uint32_t* col_data,
                                    int col_start, int n_cols) {
  Program p;
  // Minimal init — only registers we touch: CASR (column stride),
  // BAR, RAR, CAR. PATTERN_REG/wide-data slots get loaded per column.
  p.add_inst(SMC_LI(8, CASR));
  p.add_inst(SMC_LI(bank_id, BAR));
  p.add_inst(SMC_LI(row_addr, RAR));
  p.add_inst(SMC_LI(col_start * 8, CAR));
  p.add_below(PRE(BAR, 0, 0));
  p.add_below(ACT(BAR, 0, RAR, 0));
  for (int k = 0; k < n_cols; k++) {
    const uint32_t* slots = col_data + k * 16;
    for (int slot = 0; slot < 16; slot++) {
      p.add_inst(SMC_LI(slots[slot], PATTERN_REG));
      p.add_inst(SMC_LDWD(PATTERN_REG, slot));
    }
    p.add_below(WRITE(BAR, CAR, 1));     // CAR += CASR=8
    // tWR + BL/2 burst time: WRITE macro emits 7 NOPs after the command,
    // but the next column's LDWDs would otherwise modify WDATA mid-burst.
    // Sleep gives the BL=8 burst (~4 cycles) + tWR (~10 cycles) headroom
    // before we touch WDATA again. Without this, the LAST column of each
    // chunk corrupts (because the closing PRE fires before tWR elapses).
    p.add_inst(SMC_SLEEP(8));
  }
  // Extra recovery before the closing PRE for the last column.
  p.add_inst(SMC_SLEEP(8));
  p.add_below(PRE(BAR, 0, 0));
  p.add_inst(SMC_SLEEP(4));
  p.add_inst(SMC_END());
  return p;
}

// Read back one row (single execute, no label collision risk).
static Program build_read_program(int bank_id, uint32_t row_addr) {
  Program p;
  p.add_inst(SMC_LI(8, CASR));
  p.add_inst(SMC_LI(bank_id, BAR));
  p.add_inst(SMC_LI(128, NUM_COLS_REG));
  p.add_below(rdRow_immediate(BAR, row_addr));   // single use → no collision
  p.add_inst(SMC_END());
  return p;
}

int main(int argc, char** argv) {
  if (argc < 4 || argc > 5) {
    cerr << "Usage: " << argv[0]
         << " <bender_id> <bank_id> <row_addr> [seed]" << endl;
    return 1;
  }
  int bender_id   = atoi(argv[1]);
  int bank_id     = atoi(argv[2]);
  uint32_t row    = (uint32_t)atoi(argv[3]);
  unsigned seed   = (argc == 5) ? (unsigned)atoi(argv[4]) : 0xC0FFEE;

  // Generate deterministic 8192-byte buffer = 2048 32-bit values.
  // Each 32-bit value is unique (no compressible structure).
  std::mt19937 rng(seed);
  vector<uint32_t> data(2048);
  for (auto& v : data) v = rng();

  cerr << "[colsmoke] bender=" << bender_id << " bank=" << bank_id
       << " row=" << row << " seed=" << seed << endl;
  cerr << "[colsmoke] data[0..3]: 0x" << hex
       << data[0] << " 0x" << data[1] << " 0x" << data[2] << " 0x" << data[3]
       << dec << endl;

  SoftMCPlatform platform(bender_id);
  if (platform.init() != SOFTMC_SUCCESS) {
    cerr << "[colsmoke] platform init failed" << endl;
    return 2;
  }
  platform.reset_fpga();

  using clk = std::chrono::steady_clock;
  using ns  = std::chrono::nanoseconds;

  // Write 3 chunks back-to-back.
  auto t0 = clk::now();
  int col_start = 0;
  for (int chunk = 0; chunk < 3; chunk++) {
    int n_cols = CHUNK_COLS[chunk];
    Program p = build_chunk_program(bank_id, row,
                                     data.data() + col_start * 16,
                                     col_start, n_cols);
    if (chunk == 0) {
      cerr << "[colsmoke] chunk 0 program size: " << p.size() << " bytes ("
           << (p.size() / 8) << " insts) for " << n_cols << " columns" << endl;
    }
    platform.execute(p);
    col_start += n_cols;
  }
  auto t1 = clk::now();

  // Read back the whole row.
  Program rprog = build_read_program(bank_id, row);
  cerr << "[colsmoke] read program size: " << rprog.size() << " bytes ("
       << (rprog.size() / 8) << " insts)" << endl;
  platform.execute(rprog);

  uint8_t row_buf[8192];
  int rc = platform.receiveData(row_buf, 8192);
  auto t2 = clk::now();
  if (rc != 8192) {
    cerr << "[colsmoke] receiveData rc=" << rc << endl;
    return 3;
  }

  // Build expected buffer (little-endian 4-byte chunks of data[]).
  uint8_t exp[8192];
  for (int i = 0; i < 2048; i++) {
    uint32_t v = data[i];
    exp[i*4 + 0] = (uint8_t)((v >>  0) & 0xFFu);
    exp[i*4 + 1] = (uint8_t)((v >>  8) & 0xFFu);
    exp[i*4 + 2] = (uint8_t)((v >> 16) & 0xFFu);
    exp[i*4 + 3] = (uint8_t)((v >> 24) & 0xFFu);
  }

  int byte_mismatches = 0;
  long bit_mismatches = 0;
  for (int i = 0; i < 8192; i++) {
    if (row_buf[i] != exp[i]) {
      byte_mismatches++;
      bit_mismatches += __builtin_popcount((unsigned)(row_buf[i] ^ exp[i]));
    }
  }

  // Per-segment diagnostic: which 32-bit segments differ?
  int segment_mismatches = 0;
  int first_bad_seg = -1;
  for (int s = 0; s < 2048; s++) {
    uint32_t actual = (uint32_t)row_buf[s*4]
                    | ((uint32_t)row_buf[s*4+1] << 8)
                    | ((uint32_t)row_buf[s*4+2] << 16)
                    | ((uint32_t)row_buf[s*4+3] << 24);
    if (actual != data[s]) {
      segment_mismatches++;
      if (first_bad_seg < 0) first_bad_seg = s;
    }
  }

  bool pass = (byte_mismatches == 0);
  cerr << "[colsmoke] " << (pass ? "PASS" : "FAIL")
       << "  byte_mismatches=" << byte_mismatches << "/8192"
       << "  bit_mismatches=" << bit_mismatches << "/65536"
       << "  segment_mismatches=" << segment_mismatches << "/2048"
       << endl;

  if (!pass) {
    cerr << "[colsmoke] first bad segment: " << first_bad_seg
         << "  expected=0x" << hex << data[first_bad_seg]
         << "  got=0x" << ((uint32_t)row_buf[first_bad_seg*4]
                           | ((uint32_t)row_buf[first_bad_seg*4+1] << 8)
                           | ((uint32_t)row_buf[first_bad_seg*4+2] << 16)
                           | ((uint32_t)row_buf[first_bad_seg*4+3] << 24))
         << dec << endl;
    cerr << "[colsmoke] first 16 bytes  actual: ";
    for (int i = 0; i < 16; i++) {
      char buf[8]; snprintf(buf, sizeof(buf), "%02X ", row_buf[i]);
      cerr << buf;
    }
    cerr << endl;
    cerr << "[colsmoke] first 16 bytes expected: ";
    for (int i = 0; i < 16; i++) {
      char buf[8]; snprintf(buf, sizeof(buf), "%02X ", exp[i]);
      cerr << buf;
    }
    cerr << endl;
  }

  double ms_write = std::chrono::duration_cast<ns>(t1 - t0).count() / 1e6;
  double ms_read  = std::chrono::duration_cast<ns>(t2 - t1).count() / 1e6;
  cerr << "[colsmoke] timing: write=" << ms_write << " ms (3 chunks), "
       << "read=" << ms_read << " ms (1 execute)" << endl;

  cerr << "[colsmoke] DONE — " << (pass ? "PASS" : "FAIL") << endl;
  _exit(pass ? 0 : 6);
}
