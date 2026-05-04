// BitNet PIM server — long-running daemon that opens the FPGA platform
// once and processes matmul requests over stdin/stdout. Eliminates the
// per-call subprocess startup + platform.init() + reset_fpga() overhead
// that dominates today's measured throughput.
//
// Protocol (binary, all little-endian):
//   - server reads:  u32 req_len   (0 = quit sentinel)
//                    req_len bytes (identical to bitnet-proj-exe's
//                                   v2 inputs.bin body — magic +
//                                   header + masks + x_bitplane +
//                                   bitplane_factor)
//   - server writes: 8192 bytes of int32[2048] = the y output array
//                    (same as bitnet-proj-exe's output.bin body)
//
// Status / errors written to stderr only (so stdout stays a clean
// binary stream).
//
// Multi-bank parallelism (Path C): bank_arg can be a single integer
// ("1") for single-bank operation, or comma-separated ("0,1,2,3") to
// distribute (chunk, sign) work units round-robin across N banks. Each
// bank gets one calibrated tuple in its own subarray, its own backup
// pool of persistent-weight rows, and its own MAJ3 body inside the
// combined program. N banks per execute amortizes per-execute PCIe
// overhead and gives ~1.3-1.5× per-MAJ3 speedup (Path C smoke test).
//
// Argv:
//   ./bitnet-proj-server <bender_id> <calib_file> <bank_arg>
//     bank_arg: "1" (single) or "0,1,2,3" (multi-bank, up to 4 banks)
#include "instruction.h"
#include "prog.h"
#include "platform.h"
#include "../util.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

using namespace std;

static const int CHUNK_COLS[3] = {43, 43, 42};
static constexpr uint32_t MAGIC_V2 = 0xB17EF002u;

static Program build_chunk_program(int bank_id, uint32_t row_addr,
                                    const uint32_t* col_data,
                                    int col_start, int n_cols) {
  Program p;
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
    p.add_below(WRITE(BAR, CAR, 1));
    p.add_inst(SMC_SLEEP(8));
  }
  p.add_inst(SMC_SLEEP(8));
  p.add_below(PRE(BAR, 0, 0));
  p.add_inst(SMC_SLEEP(4));
  p.add_inst(SMC_END());
  return p;
}

static void per_column_write_row(SoftMCPlatform& platform, int bank_id,
                                  uint32_t row, const uint32_t* data_2048) {
  int col_start = 0;
  for (int chunk = 0; chunk < 3; chunk++) {
    int n_cols = CHUNK_COLS[chunk];
    Program p = build_chunk_program(bank_id, row,
                                     data_2048 + col_start * 16,
                                     col_start, n_cols);
    platform.execute(p);
    col_start += n_cols;
  }
}

// RowClone: charge-sharing 2-row copy via doubleACT(t_12=30, t_23=1).
// Used for "persistent weights" mode: backup row → Rfirst before each
// MAJ3. Cheap (single doubleACT) compared to per-column write of W.
static Program build_rowclone_program(int bank_id,
                                       uint32_t src_row, uint32_t dst_row) {
  Program p;
  p.add_inst(SMC_LI(8, CASR));
  p.add_inst(SMC_LI(bank_id, BAR));
  p.add_inst(SMC_LI(128, NUM_COLS_REG));
  p.add_below(PRE(BAR, 0, 0));
  p.add_inst(SMC_SLEEP(6));
  p.add_below(doubleACT(/*t_12=*/30, /*t_23=*/1, src_row, dst_row));
  p.add_inst(SMC_SLEEP(6));
  p.add_below(PRE(BAR, 0, 0));
  p.add_inst(SMC_SLEEP(6));
  p.add_inst(SMC_END());
  return p;
}

// One bank's combined RowClone → broadcast → uniform writes → frac →
// MAJ3 → read body. Caller must have set CASR (SMC_LI(8, CASR)) once
// before the first body and emit SMC_END() after the last body. Bank
// switching done at start of body via SMC_LI(bank_id, BAR).
static void emit_bank_combined_body(Program& p,
                                     int bank_id,
                                     uint32_t backup_row,
                                     uint32_t Rfirst, uint32_t Rsecond,
                                     const uint32_t* open_rows,
                                     uint32_t x_pattern,
                                     int label_base) {
  // Per-bank register setup: BAR + NUM_COLS_REG must be (re-)set when
  // this bank's body starts so prior bank's state doesn't leak in.
  p.add_inst(SMC_LI(bank_id, BAR));
  p.add_inst(SMC_LI(128, NUM_COLS_REG));

  // 1. RowClone backup → Rfirst (refresh weight from backup).
  p.add_below(PRE(BAR, 0, 0));
  p.add_inst(SMC_SLEEP(6));
  p.add_below(doubleACT(/*t_12=*/30, /*t_23=*/1, backup_row, Rfirst));
  p.add_inst(SMC_SLEEP(6));
  p.add_below(PRE(BAR, 0, 0));
  p.add_inst(SMC_SLEEP(6));

  // 2. Broadcast Rfirst → all 16 open rows.
  p.add_below(doubleACT(10, 2, Rfirst, Rsecond));
  p.add_inst(SMC_SLEEP(6));
  p.add_below(PRE(BAR, 0, 0));
  p.add_inst(SMC_SLEEP(6));

  // 3. Overwrite 11 non-weight slots with x / 0 / ONE.
  static const int act_pos[5]  = {1, 4, 7, 10, 13};
  static const int zero_pos[5] = {2, 5, 8, 11, 14};
  p.add_below(wrRow_immediate_label(BAR, open_rows[0], ONE, label_base + 0));
  for (int i = 0; i < 5; i++)
    p.add_below(wrRow_immediate_label(BAR, open_rows[act_pos[i]],
                                       x_pattern, label_base + 1 + i));
  for (int i = 0; i < 5; i++)
    p.add_below(wrRow_immediate_label(BAR, open_rows[zero_pos[i]], 0u,
                                       label_base + 100 + i));
  p.add_inst(SMC_SLEEP(6));
  p.add_below(PRE(BAR, 0, 0));
  p.add_inst(SMC_SLEEP(6));

  // 4. Frac discharge × 3 on open_rows[0].
  for (int j = 0; j < 3; j++) {
    p.add_inst(SMC_SLEEP(6));
    p.add_inst(SMC_LI(open_rows[0], RF_REG));
    p.add_inst(SMC_ACT(BAR, 0, RF_REG, 0),
               SMC_PRE(BAR, 0, 0),
               SMC_NOP(),
               SMC_NOP());
    p.add_inst(SMC_SLEEP(6));
  }
  p.add_inst(SMC_SLEEP(6));
  p.add_below(PRE(BAR, 0, 0));
  p.add_inst(SMC_SLEEP(6));

  // 5. MAJ3 doubleACT(0, 0).
  p.add_below(doubleACT(0, 0, Rfirst, Rsecond));
  p.add_inst(SMC_SLEEP(6));
  p.add_below(PRE(BAR, 0, 0));
  p.add_inst(SMC_SLEEP(6));

  // 6. Read result for this bank.
  p.add_below(rdRow_immediate_label(BAR, open_rows[0], label_base + 999));
}

// Single-bank combined program. Wrapper around emit_bank_combined_body.
// Kept for direct/test paths; the server uses build_multibank_*.
static Program build_combined_clone_bcast_maj3_program(
    int bank_id,
    uint32_t backup_row,
    uint32_t Rfirst, uint32_t Rsecond,
    const uint32_t* open_rows,
    uint32_t x_pattern,
    int label_seed) {
  Program p;
  p.add_inst(SMC_LI(8, CASR));
  emit_bank_combined_body(p, bank_id, backup_row, Rfirst, Rsecond,
                          open_rows, x_pattern, label_seed);
  p.add_inst(SMC_END());
  return p;
}

// Multi-bank combined program — up to 4 banks' MAJ3 bodies emitted
// back-to-back inside one Program. Each entry in `bank_ids/backup_rows
// /Rfirsts/Rseconds/open_rows_list/x_patterns` is one bank's work unit
// for this execute. After execute, caller must call receiveData(8192)
// once per emitted bank in the same order.
//
// Program size: ~280 instructions per bank × N banks. With N=4 → ~1120
// instructions, well under the 2048-instruction buffer.
static Program build_multibank_combined_program(
    const std::vector<int>& bank_ids,
    const std::vector<uint32_t>& backup_rows,
    const std::vector<uint32_t>& Rfirsts,
    const std::vector<uint32_t>& Rseconds,
    const std::vector<const uint32_t*>& open_rows_list,
    const std::vector<uint32_t>& x_patterns,
    int label_seed) {
  Program p;
  p.add_inst(SMC_LI(8, CASR));
  // 2000-label spacing per bank gives plenty of room for all
  // wrRow/rdRow labels each body emits (~250 labels max per bank).
  for (size_t i = 0; i < bank_ids.size(); i++) {
    emit_bank_combined_body(p, bank_ids[i], backup_rows[i],
                            Rfirsts[i], Rseconds[i],
                            open_rows_list[i], x_patterns[i],
                            label_seed + (int)i * 2000);
  }
  p.add_inst(SMC_END());
  return p;
}

static Program build_bcast_maj3_program(int bank_id,
                                         uint32_t Rfirst, uint32_t Rsecond,
                                         const uint32_t* open_rows,
                                         uint32_t x_pattern,
                                         int label_seed) {
  Program p;
  p.add_inst(SMC_LI(8, CASR));
  p.add_inst(SMC_LI(bank_id, BAR));
  p.add_inst(SMC_LI(128, NUM_COLS_REG));
  p.add_below(PRE(BAR, 0, 0));
  p.add_inst(SMC_SLEEP(6));
  p.add_below(doubleACT(10, 2, Rfirst, Rsecond));
  p.add_inst(SMC_SLEEP(6));
  p.add_below(PRE(BAR, 0, 0));
  p.add_inst(SMC_SLEEP(6));
  static const int act_pos[5]  = {1, 4, 7, 10, 13};
  static const int zero_pos[5] = {2, 5, 8, 11, 14};
  p.add_below(wrRow_immediate_label(BAR, open_rows[0], ONE, label_seed));
  for (int i = 0; i < 5; i++)
    p.add_below(wrRow_immediate_label(BAR, open_rows[act_pos[i]],
                                       x_pattern, label_seed + 1 + i));
  for (int i = 0; i < 5; i++)
    p.add_below(wrRow_immediate_label(BAR, open_rows[zero_pos[i]], 0u,
                                       label_seed + 100 + i));
  p.add_inst(SMC_SLEEP(6));
  p.add_below(PRE(BAR, 0, 0));
  p.add_inst(SMC_SLEEP(6));
  for (int j = 0; j < 3; j++) {
    p.add_inst(SMC_SLEEP(6));
    p.add_inst(SMC_LI(open_rows[0], RF_REG));
    p.add_inst(SMC_ACT(BAR, 0, RF_REG, 0),
               SMC_PRE(BAR, 0, 0),
               SMC_NOP(),
               SMC_NOP());
    p.add_inst(SMC_SLEEP(6));
  }
  p.add_inst(SMC_SLEEP(6));
  p.add_below(PRE(BAR, 0, 0));
  p.add_inst(SMC_SLEEP(6));
  p.add_below(doubleACT(0, 0, Rfirst, Rsecond));
  p.add_inst(SMC_SLEEP(6));
  p.add_below(PRE(BAR, 0, 0));
  p.add_inst(SMC_SLEEP(6));
  p.add_below(rdRow_immediate_label(BAR, open_rows[0], label_seed + 999));
  p.add_inst(SMC_END());
  return p;
}

static void segment_popcount(const uint8_t* row_buf, int* out, int n) {
  for (int s = 0; s < n; s++) {
    uint32_t actual = (uint32_t)row_buf[s*4]
                    | ((uint32_t)row_buf[s*4+1] << 8)
                    | ((uint32_t)row_buf[s*4+2] << 16)
                    | ((uint32_t)row_buf[s*4+3] << 24);
    out[s] = __builtin_popcount(actual);
  }
}

struct Calib {
  int s_id, bank;
  uint32_t Rfirst, Rsecond;
  vector<uint32_t> open_rows;
};

static vector<Calib> read_calib(const string& path, int wanted_bank) {
  vector<Calib> out;
  ifstream f(path);
  string line;
  while (getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    istringstream iss(line);
    Calib c;
    if (!(iss >> c.s_id >> c.bank >> c.Rfirst >> c.Rsecond)) continue;
    uint32_t v;
    while (iss >> v) c.open_rows.push_back(v);
    if (c.open_rows.size() == 16 && c.bank == wanted_bank) out.push_back(c);
  }
  return out;
}

// Read exactly `n` bytes from stdin or return 0 (EOF).
static bool read_exact(void* buf, size_t n) {
  size_t got = 0;
  char* p = (char*)buf;
  while (got < n) {
    ssize_t r = read(0, p + got, n - got);
    if (r <= 0) return false;
    got += (size_t)r;
  }
  return true;
}

// Per-bank server state: one calibrated MAJ3 tuple + persistent-weight
// backup pool, sized for whatever fraction of (chunk, sign) work units
// this bank will own across the largest expected matmul.
struct BankConfig {
  int bank_id;
  Calib calib;
  std::vector<uint32_t> backup_pool;  // backup rows in this bank's subarray
};

// Process one request body. Distributes (chunk, sign) work units
// round-robin across `banks` (1..N). For N>1, each platform.execute()
// runs N banks' MAJ3 bodies in one program; receiveData() is called
// N times after each execute, in the same order banks were emitted.
//
// Persistent weights: each work unit's mask is per-col written to its
// owning bank's backup row ONCE (outer setup loop), then re-used via
// fast RowClone in every bitplane's combined program (inner loop).
static int process_request(SoftMCPlatform& platform,
                            std::vector<BankConfig>& banks,
                            const uint8_t* req, size_t req_len,
                            int& label_base, int response_fd) {
  if (banks.empty()) {
    fprintf(stderr, "[server] no banks configured\n");
    return -1;
  }
  if (req_len < 5 * 4) {
    fprintf(stderr, "[server] request too small (%zu B)\n", req_len);
    return -1;
  }
  size_t off = 0;
  auto rd_u32 = [&](uint32_t& v) {
    memcpy(&v, req + off, 4); off += 4;
  };
  uint32_t magic, d_in, d_out, n_chunks, n_bitplanes;
  rd_u32(magic); rd_u32(d_in); rd_u32(d_out);
  rd_u32(n_chunks); rd_u32(n_bitplanes);
  if (magic != MAGIC_V2) {
    fprintf(stderr, "[server] bad magic 0x%x\n", magic);
    return -1;
  }
  if (d_out != 2048) {
    fprintf(stderr, "[server] expected d_out=2048, got %u\n", d_out);
    return -1;
  }
  size_t need = (size_t)5*4 + (size_t)n_chunks*d_out*4*2
              + (size_t)n_chunks*n_bitplanes*4 + (size_t)n_bitplanes*4;
  if (req_len < need) {
    fprintf(stderr, "[server] short request: need %zu got %zu\n", need, req_len);
    return -1;
  }

  // Slice into views.
  const uint32_t* pos_mask_all = (const uint32_t*)(req + off);
  off += (size_t)n_chunks * d_out * 4;
  const uint32_t* neg_mask_all = (const uint32_t*)(req + off);
  off += (size_t)n_chunks * d_out * 4;
  const uint32_t* x_bitplane_all = (const uint32_t*)(req + off);
  off += (size_t)n_chunks * n_bitplanes * 4;
  const int32_t*  bitplane_factor = (const int32_t*)(req + off);

  const int N = (int)banks.size();
  const size_t n_units = (size_t)n_chunks * 2;        // (chunk, sign) pairs
  const size_t n_rounds = (n_units + N - 1) / N;       // # of N-bank executes per bitplane

  // Each bank needs `n_rounds` backup rows (one per round it participates in).
  for (int bk = 0; bk < N; bk++) {
    if (banks[bk].backup_pool.size() < n_rounds) {
      fprintf(stderr, "[server] bank %d backup pool too small: have %zu, "
              "need %zu (n_units=%zu, N=%d)\n",
              banks[bk].bank_id, banks[bk].backup_pool.size(),
              n_rounds, n_units, N);
      return -1;
    }
  }

  // Iteration order chosen to PRESERVE write-then-use locality (every
  // backup row is read by RowClone within ~milliseconds of its per-col
  // write, same as the pre-multibank single-bank server). Without this
  // locality the model produces nonsense — see the back-compat test
  // failure 2026-05-04 where batching ALL writes upfront broke q_proj.
  //
  //   for round = 0..n_rounds:
  //     write up to N backups (one per bank, this round's work units)
  //     for bitplane = 0..n_bitplanes:
  //       multibank execute with all N banks' MAJ3 bodies
  //       receive + popcount + accumulate per bank
  vector<int32_t> y(d_out, 0);
  for (size_t round = 0; round < n_rounds; round++) {
    // 1. Per-col write each active bank's backup row for this round.
    int active_in_round = 0;
    for (int bk = 0; bk < N; bk++) {
      size_t u = round * (size_t)N + (size_t)bk;
      if (u >= n_units) break;
      uint32_t chunk = (uint32_t)(u / 2);
      int sign = (int)(u % 2);
      const uint32_t* mask = (sign == 0)
          ? pos_mask_all + (size_t)chunk * d_out
          : neg_mask_all + (size_t)chunk * d_out;
      uint32_t backup_row = banks[bk].backup_pool[round];
      per_column_write_row(platform, banks[bk].bank_id, backup_row, mask);
      active_in_round++;
    }
    if (active_in_round == 0) break;

    // 2. For each bitplane, dispatch one multibank execute over the
    //    same active banks (their backup rows still hold the masks
    //    we just wrote).
    for (uint32_t b = 0; b < n_bitplanes; b++) {
      std::vector<int>             ex_bank_ids;
      std::vector<uint32_t>        ex_backup_rows;
      std::vector<uint32_t>        ex_Rfirsts;
      std::vector<uint32_t>        ex_Rseconds;
      std::vector<const uint32_t*> ex_open_rows;
      std::vector<uint32_t>        ex_x_patterns;
      std::vector<int>             ex_signs;
      ex_bank_ids.reserve(active_in_round);
      ex_backup_rows.reserve(active_in_round);
      ex_Rfirsts.reserve(active_in_round);
      ex_Rseconds.reserve(active_in_round);
      ex_open_rows.reserve(active_in_round);
      ex_x_patterns.reserve(active_in_round);
      ex_signs.reserve(active_in_round);

      for (int bk = 0; bk < active_in_round; bk++) {
        size_t u = round * (size_t)N + (size_t)bk;
        uint32_t chunk = (uint32_t)(u / 2);
        int sign = (int)(u % 2);
        uint32_t xb = x_bitplane_all[(size_t)chunk * n_bitplanes + b];
        ex_bank_ids.push_back(banks[bk].bank_id);
        ex_backup_rows.push_back(banks[bk].backup_pool[round]);
        ex_Rfirsts.push_back(banks[bk].calib.Rfirst);
        ex_Rseconds.push_back(banks[bk].calib.Rsecond);
        ex_open_rows.push_back(banks[bk].calib.open_rows.data());
        ex_x_patterns.push_back(xb);
        ex_signs.push_back(sign);
      }

      Program p = build_multibank_combined_program(
          ex_bank_ids, ex_backup_rows, ex_Rfirsts, ex_Rseconds,
          ex_open_rows, ex_x_patterns, label_base);
      label_base += 2000 * N + 1000;
      platform.execute(p);

      for (int i = 0; i < active_in_round; i++) {
        uint8_t row[8192];
        int rc = platform.receiveData(row, 8192);
        if (rc != 8192) {
          fprintf(stderr, "[server] receiveData rc=%d (round=%zu bp=%u i=%d)\n",
                  rc, round, b, i);
          return -1;
        }
        vector<int> pc(d_out);
        segment_popcount(row, pc.data(), (int)d_out);
        int sign_factor = (ex_signs[i] == 0) ? +1 : -1;
        int weight = sign_factor * bitplane_factor[b];
        for (uint32_t j = 0; j < d_out; j++) y[j] += weight * pc[j];
      }
    }
  }

  // Write 8192 byte (= int32 × 2048) response on the saved response_fd
  // (NOT FD 1, which we permanently redirected to stderr).
  ssize_t total = (ssize_t)d_out * 4;
  ssize_t written = 0;
  while (written < total) {
    ssize_t w = write(response_fd, ((char*)y.data()) + written, total - written);
    if (w <= 0) {
      fprintf(stderr, "[server] write failed: %s\n", strerror(errno));
      return -1;
    }
    written += w;
  }
  return 0;
}

// Parse "0,1,2,3" or "1" → vector<int>{0,1,2,3} / {1}. Returns empty on
// parse error. Caps at 8 banks (more would overflow program buffer).
static std::vector<int> parse_bank_arg(const std::string& s) {
  std::vector<int> out;
  std::string cur;
  for (size_t i = 0; i <= s.size(); i++) {
    char ch = (i < s.size()) ? s[i] : ',';
    if (ch == ',') {
      if (cur.empty()) { out.clear(); return out; }
      try {
        int v = std::stoi(cur);
        if (v < 0 || v > 15) { out.clear(); return out; }
        out.push_back(v);
      } catch (...) { out.clear(); return out; }
      cur.clear();
    } else {
      cur += ch;
    }
  }
  if (out.size() > 8) out.clear();
  // Reject duplicate banks.
  std::set<int> seen(out.begin(), out.end());
  if (seen.size() != out.size()) out.clear();
  return out;
}

// Build the persistent-weight backup pool for one bank's calibrated
// tuple. Pool = rows in the same 640-aligned subarray as the tuple's
// open_rows[0], excluding any row in the 16-row open set.
static std::vector<uint32_t> build_backup_pool(const Calib& c) {
  std::set<uint32_t> open_set(c.open_rows.begin(), c.open_rows.end());
  uint32_t any_open = c.open_rows[0];
  uint32_t subarray_start = (any_open / 640) * 640;
  std::vector<uint32_t> pool;
  for (uint32_t r = subarray_start;
       r < subarray_start + 640 && pool.size() < 500; r++) {
    if (open_set.find(r) == open_set.end()) pool.push_back(r);
  }
  return pool;
}

int main(int argc, char** argv) {
  if (argc != 4) {
    fprintf(stderr,
        "Usage: %s <bender_id> <calib_file> <bank_arg>\n"
        "  bank_arg: \"1\" (single bank) or \"0,1,2,3\" (multi-bank Path C)\n",
        argv[0]);
    return 1;
  }
  int bender_id = atoi(argv[1]);
  string calib_p = argv[2];
  std::string bank_arg = argv[3];

  std::vector<int> wanted_banks = parse_bank_arg(bank_arg);
  if (wanted_banks.empty()) {
    fprintf(stderr, "[server] could not parse bank_arg '%s' "
            "(want \"1\" or e.g. \"0,1,2,3\", banks 0..15 unique)\n",
            bank_arg.c_str());
    return 2;
  }

  // Load one calibrated tuple per requested bank.
  std::vector<BankConfig> banks;
  for (int bk : wanted_banks) {
    vector<Calib> cs = read_calib(calib_p, bk);
    if (cs.empty()) {
      fprintf(stderr, "[server] no calib for bank %d in %s\n",
              bk, calib_p.c_str());
      return 2;
    }
    BankConfig bc;
    bc.bank_id = bk;
    bc.calib = cs[0];
    bc.backup_pool = build_backup_pool(bc.calib);
    if (bc.backup_pool.empty()) {
      fprintf(stderr, "[server] empty backup pool for bank %d\n", bk);
      return 2;
    }
    banks.push_back(std::move(bc));
  }

  // Critical: SoftMCPlatform::init(), reset_fpga(), and any library code
  // can write to STDOUT via std::cout. Stdout is our binary response
  // channel — any text written there corrupts the response.
  // Robust fix: SAVE original stdout to a high fd (response_fd), then
  // PERMANENTLY redirect FD 1 to stderr. All library prints go to
  // stderr; binary responses go to response_fd.
  fflush(stdout);
  int response_fd = dup(STDOUT_FILENO);
  if (response_fd < 0) {
    fprintf(stderr, "[server] dup(stdout) failed\n"); return 3;
  }
  if (dup2(STDERR_FILENO, STDOUT_FILENO) < 0) {
    fprintf(stderr, "[server] dup2(stderr→stdout) failed\n"); return 3;
  }

  SoftMCPlatform platform(bender_id);
  if (platform.init() != SOFTMC_SUCCESS) {
    fprintf(stderr, "[server] platform init failed\n"); return 3;
  }
  platform.reset_fpga();
  std::cout.flush();
  fflush(stdout);

  fprintf(stderr, "[server] ready: bender=%d N_banks=%zu "
          "(response_fd=%d)\n",
          bender_id, banks.size(), response_fd);
  for (size_t i = 0; i < banks.size(); i++) {
    fprintf(stderr, "[server]   bank %d: s_id=%d Rfirst=%u Rsecond=%u "
            "backup_pool=%zu rows starting at %u\n",
            banks[i].bank_id, banks[i].calib.s_id,
            banks[i].calib.Rfirst, banks[i].calib.Rsecond,
            banks[i].backup_pool.size(), banks[i].backup_pool[0]);
  }

  vector<uint8_t> req_buf;
  int label_base = 0;
  while (true) {
    uint32_t req_len = 0;
    if (!read_exact(&req_len, 4)) {
      fprintf(stderr, "[server] EOF on stdin, exiting\n");
      break;
    }
    if (req_len == 0) {
      fprintf(stderr, "[server] quit sentinel received\n");
      break;
    }
    if (req_len > 64u * 1024u * 1024u) {  // 64 MB sanity cap
      fprintf(stderr, "[server] request too large: %u\n", req_len);
      return 4;
    }
    req_buf.resize(req_len);
    if (!read_exact(req_buf.data(), req_len)) {
      fprintf(stderr, "[server] short read of request body\n");
      return 5;
    }
    if (process_request(platform, banks,
                        req_buf.data(), req_len, label_base,
                        response_fd) != 0) {
      return 6;
    }
  }
  _exit(0);
}
