// e2e_sim phase-2: the behavioral DRAM (DPI-C side).
// - open-row per {bg,bank} latched on ACT (slot order 0..3)
// - sparse content store keyed {bg,bank,row,col}; untouched beats seed
//   deterministically via splitmix64 so reads verify without writes
// - reads return IN ORDER after RL cycles, serialized one beat/cycle
// Widths fixed to parameters.vh: BG 2, BANK 2, COL 10, ROW 17.
#include "svdpi.h"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <unordered_map>
#include <array>
#include <deque>

namespace {
constexpr int BGW = 2, BAW = 2, CW = 10, RW = 17;
constexpr uint64_t RL = 24;          // read latency, fabric cycles
uint64_t g_cycle = 0;
uint64_t g_next_ret = 0;             // serialize returns
uint32_t g_open_row[1 << (BGW + BAW)];
std::unordered_map<uint64_t, std::array<uint32_t, 16>> g_mem;
struct Ret { uint64_t at; std::array<uint32_t, 16> d; };
std::deque<Ret> g_pipe;

inline uint64_t sm64(uint64_t x) {
  x += 0x9e3779b97f4a7c15ull;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  return x ^ (x >> 31);
}
inline uint64_t keyof(int bg, int bank, int row, int col) {
  return ((uint64_t)bg << 40) | ((uint64_t)bank << 36) |
         ((uint64_t)row << 12) | (uint64_t)col;
}
inline std::array<uint32_t, 16> seed_beat(uint64_t key) {
  std::array<uint32_t, 16> a;
  for (int i = 0; i < 16; i += 2) {
    uint64_t v = sm64(key * 16 + i);
    a[i] = (uint32_t)v; a[i + 1] = (uint32_t)(v >> 32);
  }
  return a;
}
inline uint32_t fld(const svBitVecVal* v, int slot, int w) {
  int lo = slot * w;
  uint64_t lo64 = ((uint64_t)v[lo / 32]) | ((uint64_t)v[lo / 32 + 1] << 32);
  return (uint32_t)((lo64 >> (lo % 32)) & ((1u << w) - 1u));
}
} // namespace

// exposed to the TB for oracle computation
extern "C" void dram_expected_beat(int bg, int bank, int row, int col,
                                   uint32_t* out16) {
  uint64_t key = keyof(bg, bank, row, col);
  auto it = g_mem.find(key);
  std::array<uint32_t, 16> a = (it != g_mem.end()) ? it->second
                                                   : seed_beat(key);
  memcpy(out16, a.data(), 64);
}

extern "C" void dram_tick(int act_m, int rd_m, int wr_m, int pre_m,
                          const svBitVecVal* bgs, const svBitVecVal* banks,
                          const svBitVecVal* cols, const svBitVecVal* rows,
                          const svBitVecVal* wdata,
                          svBitVecVal* rdata, int* rvalid) {
  g_cycle++;
  (void)pre_m;
  for (int s = 0; s < 4; s++) {
    int bg = fld(bgs, s, BGW), ba = fld(banks, s, BAW);
    if (act_m & (1 << s))
      g_open_row[(bg << BAW) | ba] = fld(rows, s, RW);
    if (wr_m & (1 << s)) {
      uint64_t key = keyof(bg, ba, g_open_row[(bg << BAW) | ba],
                           fld(cols, s, CW));
      std::array<uint32_t, 16> a;
      for (int i = 0; i < 16; i++) a[i] = wdata[i];
      g_mem[key] = a;
      static int wlog = 0;
      if (wlog < 6) { wlog++;
        fprintf(stderr, "[dpi] WR bg%d ba%d row%d col%d w0=%08x w1=%08x w15=%08x\n",
                bg, ba, g_open_row[(bg << BAW) | ba], fld(cols, s, CW),
                a[0], a[1], a[15]); }
    }
    if (rd_m & (1 << s)) {
      static int rlog = 0;
      if (rlog < 6) { rlog++;
        fprintf(stderr, "[dpi] RD bg%d ba%d row%d col%d\n",
                bg, ba, g_open_row[(bg << BAW) | ba], fld(cols, s, CW)); }
      uint64_t key = keyof(bg, ba, g_open_row[(bg << BAW) | ba],
                           fld(cols, s, CW));
      auto it = g_mem.find(key);
      std::array<uint32_t, 16> a =
          (it != g_mem.end()) ? it->second : seed_beat(key);
      // The beat is returned on rd_data in natural word order. There is
      // exactly ONE half-swap on the readback path and it lives in the
      // engine ("shuffle data because fifo outputs them on wrong order"),
      // which cancels the rdback_fifo's most-significant-half-first read
      // order. A second swap here would cancel for a host READ round trip
      // and NOT cancel for SEG_POP, because the engine taps rd_data for the
      // per-segment popcounts upstream of its own swap.
      // SIM_RBF_LSB_FIRST selects the legacy low-half-first FIFO model, and
      // that model needs this compensating swap back; the two are one knob.
      Ret r;
#ifdef SIM_RBF_LSB_FIRST
      for (int i = 0; i < 8; i++) { r.d[i] = a[i + 8]; r.d[i + 8] = a[i]; }
#else
      for (int i = 0; i < 16; i++) r.d[i] = a[i];
#endif
      uint64_t at = g_cycle + RL;
      if (at <= g_next_ret) at = g_next_ret + 1;
      r.at = at; g_next_ret = at;
      g_pipe.push_back(r);
    }
  }
  *rvalid = 0;
  memset(rdata, 0, 64);
  if (!g_pipe.empty() && g_pipe.front().at <= g_cycle) {
    memcpy(rdata, g_pipe.front().d.data(), 64);
    *rvalid = 1;
    g_pipe.pop_front();
  }
}
