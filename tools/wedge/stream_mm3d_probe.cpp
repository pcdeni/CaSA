// Standalone streamed-MM3D bisection (2026-07-24 night).
//
// Replays the EXACT instruction words of a captured production MM3D
// program (from PIM_H2C_CAPTURE) on silicon, with no server, no request
// history and no mode churn. Splits the failure space in one run:
//
//   mode      : SEG_POP vs READ
//   dispatch  : legacy execute+receiveData vs streamed session
//   size      : the 1483-inst fused MM3D body vs a 28-inst row read
//
// Legacy arm is the reference; the streamed arm is compared byte-wise
// against it, so a stall AND a corruption both show up, in-process
// (no cross-process floor to confuse the verdict).
//
// argv: ./stream-mm3d-probe <bender> <hexfile> <payload_bytes> <N> <mode:segpop|read> <arm:leg|str|both>
#include "instruction.h"
#include "prog.h"
#include "platform.h"
#include "../util.h"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <chrono>
#include <unistd.h>

using clk = std::chrono::steady_clock;
static double ms_since(clk::time_point t0){
  return std::chrono::duration_cast<std::chrono::microseconds>(
             clk::now() - t0).count() / 1000.0;
}

static std::vector<uint64_t> load_hex(const std::string& p){
  std::vector<uint64_t> v; std::ifstream f(p); std::string ln;
  while (std::getline(f, ln))
    if (ln.size() >= 16) v.push_back(strtoull(ln.c_str(), nullptr, 16));
  return v;
}

// Rebuild a fresh Program from the captured words on every use: the
// words are already finalized (absolute branch targets), and a fresh
// object sidesteps any re-execution bookkeeping.
static Program mk(const std::vector<uint64_t>& w){
  Program p;
  for (uint64_t x : w) p.add_inst((Inst)x);
  return p;
}

int main(int argc, char** argv){
  if (argc < 7){
    fprintf(stderr, "usage: %s <bender> <hex> <payload> <N> <segpop|read> <leg|str|both>\n", argv[0]);
    return 1;
  }
  setvbuf(stdout, NULL, _IONBF, 0);
  int bender = atoi(argv[1]);
  std::string hexf = argv[2];
  int payload = atoi(argv[3]);
  int N = atoi(argv[4]);
  bool segpop = (std::string(argv[5]) == "segpop");
  std::string arm = argv[6];

  auto words = load_hex(hexf);
  if (words.empty()){ fprintf(stderr, "[probe] empty hex\n"); return 1; }
  printf("[probe] bender=%d prog=%s insts=%zu payload=%d N=%d mode=%s arm=%s\n",
         bender, hexf.c_str(), words.size(), payload, N,
         segpop ? "SEG_POP" : "READ", arm.c_str());

  SoftMCPlatform pf(bender);
  if (pf.init() != SOFTMC_SUCCESS){ fprintf(stderr, "[probe] init failed\n"); return 1; }
  pf.reset_fpga();
  pf.set_aref(false);
  if (segpop){ pf.set_readback_mode_segpop(); pf.set_readback_mode_segpop(); }
  else       { pf.set_readback_mode(false);   pf.set_readback_mode(false);   }

  std::vector<std::vector<uint8_t>> ref(N), got(N);
  for (auto& b : ref) b.assign(payload, 0);
  for (auto& b : got) b.assign(payload, 0);

  int leg_bad = 0, str_bad = 0;
  double leg_ms = 0, str_ms = 0;

  // ---- legacy reference ------------------------------------------------
  if (arm == "leg" || arm == "both"){
    auto t0 = clk::now();
    for (int i = 0; i < N; i++){
      Program p = mk(words);
      pf.execute(p);
      int rc = pf.receiveData(ref[i].data(), payload);
      if (rc != payload){
        printf("[probe] LEGACY iter %d rc=%d (expected %d) -> STALL/SHORT\n",
               i, rc, payload);
        leg_bad++;
        break;
      }
    }
    leg_ms = ms_since(t0);
    printf("[probe] legacy: %d iters, %d bad, %.1f ms (%.2f ms/prog)\n",
           N, leg_bad, leg_ms, leg_ms / N);
  }

  // ---- streamed --------------------------------------------------------
  if (arm == "str" || arm == "both"){
    auto t0 = clk::now();
    pf.set_stream_en(true);
    // PROBE_UNIFORM=1: uniform-payload session (the ladder's shape)
    // instead of STREAM_SIZED (the server's shape). Same programs, same
    // payload -> isolates the session MODE from everything else.
    bool uniform = getenv("PROBE_UNIFORM") && atoi(getenv("PROBE_UNIFORM"));
    pf.stream_start(uniform ? payload : SoftMCPlatform::STREAM_SIZED);
    // PROBE_GAP_US: idle between sends. Maintenance fires on an idle
    // timer, so widening the gap raises the number of maintenance
    // events inside the session — the discriminator for the
    // "maintenance eats user trailers" mechanism.
    long gap_us = 0;
    if (const char* g = getenv("PROBE_GAP_US")) gap_us = atol(g);
    for (int i = 0; i < N; i++){
      Program p = mk(words);
      if (uniform) pf.stream_send(p);
      else         pf.stream_send(p, payload);
      if (gap_us) usleep(gap_us);
    }
    for (int i = 0; i < N; i++){
      int rc = pf.stream_recv(got[i].data(), payload);
      if (rc != payload){
        printf("[probe] STREAM iter %d rc=%d (expected %d) -> STALL/SHORT\n",
               i, rc, payload);
        str_bad++;
        break;
      }
    }
    pf.stream_stop();
    pf.set_stream_en(false);
    str_ms = ms_since(t0);
    printf("[probe] stream: %d iters, %d bad, %.1f ms (%.2f ms/prog)%s\n",
           N, str_bad, str_ms, str_ms / N,
           (leg_ms > 0 && str_ms > 0) ? "" : "");
    if (leg_ms > 0 && str_ms > 0)
      printf("[probe] speedup %.2fx\n", leg_ms / str_ms);
  }

  // ---- byte comparison -------------------------------------------------
  if (arm == "both" && !leg_bad && !str_bad){
    long diff_bytes = 0; int diff_iters = 0, first_bad = -1;
    for (int i = 0; i < N; i++){
      long d = 0;
      for (int b = 0; b < payload; b++) if (ref[i][b] != got[i][b]) d++;
      if (d){ diff_iters++; diff_bytes += d; if (first_bad < 0) first_bad = i; }
    }
    printf("[probe] COMPARE: %d/%d iters differ, %ld bytes total, first=%d -> %s\n",
           diff_iters, N, diff_bytes, first_bad,
           diff_iters ? "STREAM CORRUPTION REPRODUCED" : "clean");
    if (first_bad >= 0){
      const uint32_t* a = (const uint32_t*)ref[first_bad].data();
      const uint32_t* b = (const uint32_t*)got[first_bad].data();
      int shown = 0;
      printf("[probe]   first diffs (leg/str): ");
      for (int w = 0; w < payload/4 && shown < 6; w++)
        if (a[w] != b[w]){ printf("[w%d]=%08x/%08x ", w, a[w], b[w]); shown++; }
      printf("\n");
    }
  }
  printf("[probe] VERDICT: legacy_bad=%d stream_bad=%d\n", leg_bad, str_bad);
  return (leg_bad || str_bad) ? 2 : 0;
}
