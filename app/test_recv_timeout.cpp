// Deterministic regression test for the PIM_RECV_TIMEOUT_MS stall guard
// (api/platform.{h,cpp}). Opens the platform, sends NO program, then calls
// receiveData: with the env set it must return 0 bytes after ~T ms instead
// of spinning forever, flag recv_stalled(), and exit cleanly through the
// destructor (poisoned path must not hang on join).
//
// Run: PIM_RECV_TIMEOUT_MS=1000 ./recv-timeout-test-exe [bender]
// Without the env the test refuses to run (it would block forever by design).
#include "platform.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv) {
  const char* env = getenv("PIM_RECV_TIMEOUT_MS");
  if (!env || atol(env) <= 0) {
    fprintf(stderr, "set PIM_RECV_TIMEOUT_MS>0 (unset = pristine block-forever semantics)\n");
    return 2;
  }
  long want = atol(env);
  int bender = (argc > 1) ? atoi(argv[1]) : 2;
  SoftMCPlatform pf(bender);
  if (pf.init() != SOFTMC_SUCCESS) { fprintf(stderr, "init failed\n"); return 3; }
  uint8_t buf[64];
  auto t0 = std::chrono::steady_clock::now();
  int got = pf.receiveData(buf, 64);   // nothing was executed: must time out
  double ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - t0).count();
  bool pass = (got == 0) && pf.recv_stalled() && ms >= want * 0.9 && ms < want * 3;
  printf("[recv-timeout] got=%d bytes, stalled=%d, waited=%.0f ms (limit %ld)\n",
         got, (int)pf.recv_stalled(), ms, want);
  printf(pass ? "TIMEOUT-TEST PASS\n" : "TIMEOUT-TEST FAIL\n");
  return pass ? 0 : 1;
}
