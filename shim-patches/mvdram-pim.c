// MVDRAM reproduction (Lane 2) — mulmat census / interception shim.
// See mvdram-pim.h for modes, env knobs, and the mode-1 threading contract.
// Census is thread-safe: only thread 0 of a compute op records (all threads
// enter mul_mat with the same dst).
//
// Mode 1 (MVDRAM_PIM=1), route c2 (2026-07-18): sampled eligible GeMVs run
// their INTEGER bitplane contraction on the Lane-2 in-DRAM GeMV server
// (lane2-gemv-server, protocol per LANE2_GEMV_SERVER.md) and are verified
// against a host integer reference over the same quantized ints; dst always
// gets the stock CPU fp32 result. Rationale: q4_0 applies an fp16 scale per
// 32-weight block (q2_K: per-16-weight sub-block 4-bit scales/mins under
// fp16 d/dmin; q6_K: per-16-weight int8 scales under fp16 d), so the exact
// fp32 op is NOT expressible as one whole-K integer GeMV; the MVDRAM
// paper's §VI-B math is the pure integer bit-decomposition and the paper
// keeps floating-point on the processor (§V-A) without ever mentioning
// scale handling — so the in-DRAM part we reproduce and verify is exactly
// the integer bitplane GeMV.
//
// Integer mappings (2026-07-19, B1 host-side extension):
//   q4_0: q(0..15) -> q-8 in {-8..7}, 4-bit two's-complement (validated
//         bit-exact vs gguf 07-18). One qb=4 handle.
//   q2_K: q(0..3) -> q-2 in {-2..1}, 2-bit two's-complement (the top-bit
//         flip q^2, the exact structural analog of q4_0's q^8). One qb=2
//         handle — the server's FAC(1,2)=-2 signed convention matches
//         natively; RAW unsigned {0..3} through qb=2 would be mis-signed
//         (P0-2P1), and is host-recoverable anyway: sum q*x = sum (q-2)*x
//         + 2*sum x (whole-K identity), so verifying the signed contraction
//         verifies the raw one. Per-sub-block affine (d*sc, dmin*m, +2
//         de-bias) is host work, keeping the DRAM part scale-free.
//   q3_K: q(0..7) -> q-4 in {-4..3}, 3-bit two's-complement (top-bit flip
//         q^4 — same structural analog; verified: q^4 for q=0..7 is exactly
//         the 3-bit two's-complement pattern of q-4). The biased q is
//         ASSEMBLED from the split storage: q = low2 + 4*hmask_bit, because
//         dequantize_row_q3_K computes low2 - (hmask_bit ? 0 : 4) = q - 4.
//         One native qb=3 handle (server FAC(2,3) = -4 matches). Raw-q
//         identity: sum q*x = sum (q-4)*x + 4*sum x. Per-16-weight 6-bit
//         scale (d*(sc-32), no min term) host-applied. NOTE 2026-07-19:
//         q3_K has NO repack variant in this checkout on ANY arch
//         (ggml_repack_get_optimal_repack_type has no Q3_K case) — always
//         canonical at the generic hook.
//   q6_K: q(0..63) -> q-32 in {-32..31}, 6-bit two's-complement. The server
//         caps q_bits<=4, so ONE tensor becomes THREE server handles (an
//         exact plane split, no server change):
//           y = y[qb4: bits0..3] + 16*y[qb1: bit3] + 16*y[qb2: bits4..5]
//             = (P0+2P1+4P2-8P3) + 16*P3 + 16*(P4-2P5)
//             = P0+2P1+4P2+8P3+16P4-32P5   (the 6-bit signed contraction)
//         Cost: 7 plane-pair units vs 6 for a hypothetical native qb=6
//         (+17%); per-16-weight int8 scale (d*sc) host-applied, no min term.
#include "mvdram-pim.h"
#include "ggml-impl.h"
#include "ggml-cpu-impl.h"

#include <errno.h>
#include <math.h>
#include <poll.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
#include <pthread.h>

#define MVDRAM_MAX_SHAPES 256

typedef struct {
    enum ggml_type t0;      // weight (src0) type
    int64_t k;              // ne00: reduction dim
    int64_t m;              // ne01: output rows
    int64_t batch;          // ne11: 1 = GeMV (token gen), >1 = prefill GeMM
    uint64_t count;         // times seen
} mvdram_shape;

static mvdram_shape g_shapes[MVDRAM_MAX_SHAPES];
static int g_n_shapes = 0;
static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
static int g_mode = -1;     // -1 unresolved, 0 off, 1 census, 2 intercept

static void mvdram_dump(void) {
    const char * path = getenv("MVDRAM_PIM_LOG");
    if (!path || !*path) path = "mvdram_census.tsv";
    FILE * f = fopen(path, "w");
    if (!f) return;
    fprintf(f, "type\tK\tM\tbatch\tcount\n");
    for (int i = 0; i < g_n_shapes; i++) {
        fprintf(f, "%s\t%lld\t%lld\t%lld\t%llu\n",
                ggml_type_name(g_shapes[i].t0),
                (long long) g_shapes[i].k, (long long) g_shapes[i].m,
                (long long) g_shapes[i].batch,
                (unsigned long long) g_shapes[i].count);
    }
    fclose(f);
    fprintf(stderr, "[mvdram-pim] census: %d shapes -> %s\n", g_n_shapes, path);
}

static int mvdram_mode(void) {
    if (g_mode < 0) {
        pthread_mutex_lock(&g_mu);
        if (g_mode < 0) {
            const char * v = getenv("MVDRAM_PIM");
            int m = 0;
            if (v && *v) {
                if (strcmp(v, "census") == 0) m = 1;
                else if (strcmp(v, "0") != 0) m = 2;
            }
            if (m) atexit(mvdram_dump);
            g_mode = m;
        }
        pthread_mutex_unlock(&g_mu);
    }
    return g_mode;
}

// ===========================================================================
// Mode 1 (route c2) — Lane-2 server binding. Everything below runs ONLY on
// thread 0 of an intercept-mode process; state is not shared across threads.
// ===========================================================================

// Lane-2 protocol (LANE2_GEMV_SERVER.md / lane2_gemv_server.cpp)
#define MV_MAGIC_LOAD     0x4D563001u
#define MV_MAGIC_GEMV     0x4D563002u
#define MV_MAGIC_LOAD_ACK 0x4D5630F1u
#define MV_MAGIC_GEMV_ACK 0x4D5630F2u

// Server caps, read from lane2_gemv_server.cpp handle_load/handle_gemv
// (qb 1..4, K<=16384, M<=65536, rb 1..8, request<=256MB). Segment capacity
// is a runtime property of the startup screen (07-18: 1631 segs -> M<=52192
// per pass); the server enforces it at LOAD with status 4. Consequence:
// K=11008 fits in ONE LOAD/GEMV (no host K-split), M=11008 needs 344
// segments and M=32000 needs 1000 — both single-pass.
#define MV_SRV_KMAX  16384
#define MV_SRV_MMAX  65536
#define MV_SRV_QBMAX 4

// Eligible shapes = the paper's Llama-2-7B census dims (census_llama2_7b_q4
// + the 07-19 Q2_K census). MVDRAM_PIM_ANY_SHAPE=1 relaxes to server-cap
// bounds (for the 13B/8B/phi-4 dims later).
typedef struct { enum ggml_type t; int64_t k, m; } mv_shape_t;
static const mv_shape_t MV_SHAPES[] = {
    { GGML_TYPE_Q4_0, 4096,  4096  },   // attn q/k/v/o projections
    { GGML_TYPE_Q4_0, 4096,  11008 },   // ffn gate/up
    { GGML_TYPE_Q4_0, 11008, 4096  },   // ffn down
    { GGML_TYPE_Q2_K, 4096,  4096  },
    { GGML_TYPE_Q2_K, 4096,  11008 },
    { GGML_TYPE_Q2_K, 11008, 4096  },
    // q3_K = the Q2_K model's v/o + ALL ffn tensors (2026-07-19 census of
    // llama-2-7b.Q2_K.gguf: q3_K 4096x4096 attn_v/attn_output, 4096->11008
    // gate/up, 11008->4096 down; only attn_q/attn_k are q2_K there).
    { GGML_TYPE_Q3_K, 4096,  4096  },
    { GGML_TYPE_Q3_K, 4096,  11008 },
    { GGML_TYPE_Q3_K, 11008, 4096  },
    { GGML_TYPE_Q6_K, 4096,  32000 },   // output head (3-handle qb split)
};

// per-tensor handle plan (see integer-mapping block at the top of the file)
typedef struct { int qbits, shift; int32_t mult; } mv_hspec_t;
static int mv_handle_plan(enum ggml_type t, mv_hspec_t hs[3]) {
    switch (t) {
        case GGML_TYPE_Q4_0: hs[0] = (mv_hspec_t){ 4, 0, 1 }; return 1;
        case GGML_TYPE_Q2_K: hs[0] = (mv_hspec_t){ 2, 0, 1 }; return 1;
        case GGML_TYPE_Q3_K: hs[0] = (mv_hspec_t){ 3, 0, 1 }; return 1;
        case GGML_TYPE_Q6_K:
            hs[0] = (mv_hspec_t){ 4, 0, 1  };
            hs[1] = (mv_hspec_t){ 1, 3, 16 };
            hs[2] = (mv_hspec_t){ 2, 4, 16 };
            return 3;
        default: return 0;
    }
}

typedef struct {
    // env-derived config (read once)
    long sample;            // MVDRAM_PIM_SAMPLE
    long max_ops;           // MVDRAM_PIM_MAX_OPS
    int  rbits;             // MVDRAM_PIM_RBITS
    int  dry;               // MVDRAM_PIM_DRY
    int  any_shape;         // MVDRAM_PIM_ANY_SHAPE
    long resp_timeout_ms;   // MVDRAM_PIM_RESP_TIMEOUT_S * 1000
    const char * server, * calib, * colmask, * bender, * bank, * sid;
    const char * vote3, * oplog, * dump, * dump_tensor, * only;
} mv_cfg_t;

static mv_cfg_t g_cfg;
static int      g_cfg_ok = 0;

static uint64_t g_seen_eligible = 0;   // eligible GeMVs seen (thread 0 only)
static long     g_ops_done      = 0;   // PIM ops executed
static int      g_pim_dead      = 0;   // fatal marshaling/server error: stop
static int      g_dumped        = 0;

// server process
static pid_t g_srv_pid = -1;
static int   g_srv_in  = -1;   // write end -> server stdin
static int   g_srv_out = -1;   // read end  <- server stdout

// per-tensor cache: LOADed handle set + full ints for the host reference
#define MV_MAX_TENSORS 24
typedef struct {
    const void * data;      // src0->data (weights are static)
    enum ggml_type type;
    int64_t k, m;
    int repack;             // 0 canonical, 8 = q4_0x8
    int n_h;                // server handles (1; q6_K: 3)
    uint32_t handle[3];
    mv_hspec_t hs[3];
    int8_t * w;             // [m*k] full ints (q4:{-8..7} q2:{-2..1} q3:{-4..3} q6:{-32..31})
    double load_s;          // total LOAD wall (0 if dry)
} mv_tensor_t;
static mv_tensor_t g_tensors[MV_MAX_TENSORS];
static int g_n_tensors = 0;
static uint32_t g_next_handle = 1;

static double mv_now(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec + 1e-9 * ts.tv_nsec;
}

static const char * mv_env(const char * name, const char * dflt) {
    const char * v = getenv(name);
    return (v && *v) ? v : dflt;
}

static void mv_load_cfg(void) {
    if (g_cfg_ok) return;
    g_cfg.sample   = atol(mv_env("MVDRAM_PIM_SAMPLE", "9999"));
    if (g_cfg.sample < 1) g_cfg.sample = 1;
    g_cfg.max_ops  = atol(mv_env("MVDRAM_PIM_MAX_OPS", "3"));
    g_cfg.rbits    = atoi(mv_env("MVDRAM_PIM_RBITS", "8"));
    if (g_cfg.rbits < 1) g_cfg.rbits = 1;
    if (g_cfg.rbits > 8) g_cfg.rbits = 8;
    g_cfg.dry      = atoi(mv_env("MVDRAM_PIM_DRY", "0"));
    g_cfg.any_shape = atoi(mv_env("MVDRAM_PIM_ANY_SHAPE", "0"));
    g_cfg.resp_timeout_ms = atol(mv_env("MVDRAM_PIM_RESP_TIMEOUT_S", "1800")) * 1000L;
    g_cfg.server   = mv_env("MVDRAM_PIM_SERVER",
                            "/home/deni/Claude/mvdram-repro/lane2-gemv-server");
    g_cfg.calib    = mv_env("MVDRAM_PIM_CALIB",
                            "/home/deni/Claude/mvdram-repro/calib_maj5_dimm2.txt");
    g_cfg.colmask  = mv_env("MVDRAM_PIM_COLMASK",
                            "/home/deni/Claude/mvdram-repro/colmask_dimm2_s86_robust.txt");
    g_cfg.bender   = mv_env("MVDRAM_PIM_BENDER", "2");
    g_cfg.bank     = mv_env("MVDRAM_PIM_BANK", "0");
    g_cfg.sid      = mv_env("MVDRAM_PIM_SID", "86");
    g_cfg.vote3    = mv_env("MVDRAM_PIM_VOTE3", "0");
    g_cfg.oplog    = mv_env("MVDRAM_PIM_OPLOG", "mvdram_pim_ops.log");
    g_cfg.dump     = getenv("MVDRAM_PIM_DUMP");
    g_cfg.dump_tensor = getenv("MVDRAM_PIM_DUMP_TENSOR"); // substring filter
    g_cfg.only     = getenv("MVDRAM_PIM_ONLY");           // substring filter
    g_cfg_ok = 1;
    fprintf(stderr, "[mvdram-pim] mode 1 (route c2): sample=%ld max_ops=%ld r=%d "
            "dry=%d any_shape=%d only=%s server=%s bender=%s bank=%s sid=%s vote3=%s\n",
            g_cfg.sample, g_cfg.max_ops, g_cfg.rbits, g_cfg.dry, g_cfg.any_shape,
            g_cfg.only ? g_cfg.only : "-",
            g_cfg.server, g_cfg.bender, g_cfg.bank, g_cfg.sid, g_cfg.vote3);
}

// ---- robust pipe I/O -------------------------------------------------------

static int mv_write_full(int fd, const void * buf, size_t n) {
    const char * p = (const char *) buf;
    while (n > 0) {
        ssize_t r = write(fd, p, n);
        if (r < 0) { if (errno == EINTR) continue; return -1; }
        p += r; n -= (size_t) r;
    }
    return 0;
}

static int mv_read_full_timeout(int fd, void * buf, size_t n, long timeout_ms) {
    char * p = (char *) buf;
    double deadline = mv_now() + timeout_ms / 1000.0;
    while (n > 0) {
        long left_ms = (long) ((deadline - mv_now()) * 1000.0);
        if (left_ms <= 0) { errno = ETIMEDOUT; return -1; }
        struct pollfd pfd = { .fd = fd, .events = POLLIN };
        int pr = poll(&pfd, 1, (int) (left_ms > 1000 ? 1000 : left_ms));
        if (pr < 0) { if (errno == EINTR) continue; return -1; }
        if (pr == 0) continue;
        ssize_t r = read(fd, p, n);
        if (r < 0) { if (errno == EINTR) continue; return -1; }
        if (r == 0) { errno = EPIPE; return -1; } // server closed
        p += r; n -= (size_t) r;
    }
    return 0;
}

// ---- server lifecycle ------------------------------------------------------

static void mv_shutdown_server(void) {
    if (g_srv_pid <= 0) return;
    // graceful: len-0 quit sentinel, close stdin, wait. NEVER SIGKILL — the
    // server holds XDMA; kill -9 mid-transfer wedges the DMA engine.
    uint32_t zero = 0;
    if (g_srv_in >= 0) {
        (void) mv_write_full(g_srv_in, &zero, 4);
        close(g_srv_in); g_srv_in = -1;
    }
    int st = 0, done = 0;
    for (int i = 0; i < 1200; i++) { // up to 120 s
        pid_t r = waitpid(g_srv_pid, &st, WNOHANG);
        if (r == g_srv_pid || (r < 0 && errno == ECHILD)) { done = 1; break; }
        struct timespec ts = { 0, 100 * 1000 * 1000 };
        nanosleep(&ts, NULL);
    }
    if (!done) {
        fprintf(stderr, "[mvdram-pim] server slow to exit; SIGTERM and wait (no SIGKILL)\n");
        kill(g_srv_pid, SIGTERM);
        waitpid(g_srv_pid, &st, 0);
    }
    fprintf(stderr, "[mvdram-pim] lane2 server pid %d exited (status %d)\n",
            (int) g_srv_pid, st);
    if (g_srv_out >= 0) { close(g_srv_out); g_srv_out = -1; }
    g_srv_pid = -1;
}

static int mv_ensure_server(void) {
    if (g_srv_pid > 0) return 0;
    int inp[2], outp[2];
    if (pipe(inp) != 0) return -1;
    if (pipe(outp) != 0) { close(inp[0]); close(inp[1]); return -1; }
    signal(SIGPIPE, SIG_IGN);   // a dying server must not SIGPIPE-kill llama
    fprintf(stderr, "[mvdram-pim] spawning lane2 server: %s %s %s %s %s %s "
            "(BITSTREAM_IMEM=8192 PIM_RECV_TIMEOUT_MS=15000 PIM_VOTE3=%s)\n",
            g_cfg.server, g_cfg.bender, g_cfg.calib, g_cfg.bank, g_cfg.sid,
            g_cfg.colmask, g_cfg.vote3);
    pid_t pid = fork();
    if (pid < 0) { close(inp[0]); close(inp[1]); close(outp[0]); close(outp[1]); return -1; }
    if (pid == 0) {
        // child: stdin <- inp, stdout -> outp, stderr inherited
        dup2(inp[0], 0); dup2(outp[1], 1);
        close(inp[0]); close(inp[1]); close(outp[0]); close(outp[1]);
        setenv("BITSTREAM_IMEM", "8192", 1);          // mandatory silicon env
        setenv("PIM_RECV_TIMEOUT_MS", "15000", 1);    // mandatory silicon env
        setenv("PIM_VOTE3", g_cfg.vote3, 1);
        // LANE2_PACK / LANE2_BACKEND inherit from the parent environment.
        execl(g_cfg.server, g_cfg.server, g_cfg.bender, g_cfg.calib,
              g_cfg.bank, g_cfg.sid, g_cfg.colmask, (char *) NULL);
        fprintf(stderr, "[mvdram-pim] exec %s failed: %s\n", g_cfg.server, strerror(errno));
        _exit(127);
    }
    close(inp[0]); close(outp[1]);
    g_srv_pid = pid; g_srv_in = inp[1]; g_srv_out = outp[0];
    atexit(mv_shutdown_server);
    return 0;
}

static int mv_send_req(const void * body, uint32_t len) {
    uint32_t l = len;
    if (mv_write_full(g_srv_in, &l, 4) != 0) return -1;
    return mv_write_full(g_srv_in, body, len);
}

// reads a length-prefixed response; caller frees *body
static int mv_read_resp(uint8_t ** body, uint32_t * len) {
    uint32_t l = 0;
    if (mv_read_full_timeout(g_srv_out, &l, 4, g_cfg.resp_timeout_ms) != 0) return -1;
    uint8_t * b = (uint8_t *) malloc(l ? l : 1);
    if (!b) return -1;
    if (l && mv_read_full_timeout(g_srv_out, b, l, g_cfg.resp_timeout_ms) != 0) {
        free(b); return -1;
    }
    *body = b; *len = l;
    return 0;
}

// ---- q4_0 extraction (canonical and q4_0x8-repacked) -----------------------
// Canonical block_q4_0 (18 B): fp16 d, then qs[16]; element j (j<16) = low
// nibble of qs[j], element j+16 = high nibble; stored nibble is the biased
// value q (0..15) meaning q-8, so two's-complement 4-bit = q ^ 8.
// Repacked block_q4_0x8 (144 B, repack.cpp make_block_q4_0x8 inter=8): 8
// consecutive rows interleaved; d[8] one per row; qs = 16 chunks of 8 bytes,
// chunk i <- row i%8, canonical qs bytes [(i/8)*8 .. +8), XOR 0x88 — i.e.
// the STORED nibbles are already two's-complement.

typedef struct { uint16_t d[8]; uint8_t qs[128]; } mv_block_q4_0x8;

// w_out[m*k+n] = descaled int {-8..7}; d_out (optional) = raw fp16 per block
static int mv_extract_q4_0(const struct ggml_tensor * t, int repack,
                           int8_t * w_out, uint16_t * d_out) {
    const int64_t k = t->ne[0], m = t->ne[1];
    const int64_t nblk = k / 32;
    if (repack == 0) {
        for (int64_t r = 0; r < m; r++) {
            const uint8_t * row = (const uint8_t *) t->data + r * t->nb[1];
            for (int64_t x = 0; x < nblk; x++) {
                const uint8_t * blk = row + x * 18;
                if (d_out) memcpy(&d_out[r * nblk + x], blk, 2);
                const uint8_t * qs = blk + 2;
                int8_t * wr = w_out + r * k + x * 32;
                for (int j = 0; j < 16; j++) {
                    uint8_t lo = (uint8_t) ((qs[j] & 0x0F) ^ 0x8);
                    uint8_t hi = (uint8_t) (((qs[j] >> 4) & 0x0F) ^ 0x8);
                    wr[j]      = (int8_t) (lo >= 8 ? (int) lo - 16 : (int) lo);
                    wr[j + 16] = (int8_t) (hi >= 8 ? (int) hi - 16 : (int) hi);
                }
            }
        }
        return 0;
    }
    if (repack == 8) {
        if (m % 8 != 0 || k % 32 != 0) return -1;
        const mv_block_q4_0x8 * base = (const mv_block_q4_0x8 *) t->data;
        for (int64_t r = 0; r < m; r++) {
            const int64_t g = r / 8, ri = r % 8;
            for (int64_t x = 0; x < nblk; x++) {
                const mv_block_q4_0x8 * blk = &base[g * nblk + x];
                if (d_out) d_out[r * nblk + x] = blk->d[ri];
                int8_t * wr = w_out + r * k + x * 32;
                for (int j = 0; j < 16; j++) {
                    uint8_t b = blk->qs[((j >> 3) * 8 + ri) * 8 + (j & 7)];
                    uint8_t lo = (uint8_t) (b & 0x0F);        // already two's-c
                    uint8_t hi = (uint8_t) ((b >> 4) & 0x0F);
                    wr[j]      = (int8_t) (lo >= 8 ? (int) lo - 16 : (int) lo);
                    wr[j + 16] = (int8_t) (hi >= 8 ? (int) hi - 16 : (int) hi);
                }
            }
        }
        return 0;
    }
    return -1;
}

// ---- q2_K extraction (canonical only on this tower) ------------------------
// block_q2_K (84 B / 256 weights): scales[16] (low nibble = scale, high =
// min, one byte per 16-weight sub-block), qs[64] (2-bit quants), fp16 d,
// fp16 dmin (union dm). Weight t (0..255) of a superblock lives in byte
// qs[(t/128)*32 + t%32] at bit offset 2*((t%128)/32) — transcribed from
// dequantize_row_q2_K (ggml-quants.c). Sub-block index = t/16.
// Integer mapping: w = q - 2 in {-2..1} (2-bit two's-complement, see header).
// sc_out (optional): per (row, superblock) 20-byte record
//   [ scales[16] raw | fp16 d | fp16 dmin ]  for gguf ground-truth checks.
static int mv_extract_q2_K(const struct ggml_tensor * t, int8_t * w_out,
                           uint8_t * sc_out) {
    const int64_t k = t->ne[0], m = t->ne[1];
    if (k % 256 != 0) return -1;
    const int64_t nblk = k / 256;
    for (int64_t r = 0; r < m; r++) {
        const uint8_t * row = (const uint8_t *) t->data + r * t->nb[1];
        for (int64_t x = 0; x < nblk; x++) {
            const uint8_t * blk = row + x * 84;      // sizeof(block_q2_K)
            const uint8_t * qs  = blk + 16;
            if (sc_out) {
                uint8_t * o = sc_out + (size_t) (r * nblk + x) * 20;
                memcpy(o, blk, 16);          // scales[16]
                memcpy(o + 16, blk + 80, 4); // d, dmin (fp16 each)
            }
            int8_t * wr = w_out + r * k + x * 256;
            for (int tp = 0; tp < 256; tp++) {
                int by = (tp / 128) * 32 + (tp % 32);
                int sh = 2 * ((tp % 128) / 32);
                wr[tp] = (int8_t) (((qs[by] >> sh) & 3) - 2);
            }
        }
    }
    return 0;
}

// ---- q3_K extraction (canonical; NO repack variant exists for q3_K) --------
// block_q3_K (110 B / 256 weights): hmask[32] (high bit, INVERTED sense),
// qs[64] (low 2 bits), scales[12] (16 6-bit scales packed), fp16 d.
// Bit gather per dequantize_row_q3_K (ggml-quants.c):
//   low2      = (qs[(t/128)*32 + t%32] >> (2*((t%128)/32))) & 3
//               (identical addressing to q2_K's qs)
//   hmask bit = (hmask[t%32] >> (t/32)) & 1
//   value     = low2 - (hmask_bit ? 0 : 4)  ==  (low2 + 4*hmask_bit) - 4
// Integer mapping: w = q - 4 in {-4..3} with q = low2 | (hmask_bit << 2)
// (3-bit two's-complement = q ^ 4; see the mapping block at the top).
// Per-16-weight 6-bit scale used as d*(sc-32), NO min term — host-side.
// sc_out (optional): per (row, superblock) 14-byte record
//   [ scales[12] raw | fp16 d ]  for gguf ground-truth checks.
static int mv_extract_q3_K(const struct ggml_tensor * t, int8_t * w_out,
                           uint8_t * sc_out) {
    const int64_t k = t->ne[0], m = t->ne[1];
    if (k % 256 != 0) return -1;
    const int64_t nblk = k / 256;
    for (int64_t r = 0; r < m; r++) {
        const uint8_t * row = (const uint8_t *) t->data + r * t->nb[1];
        for (int64_t x = 0; x < nblk; x++) {
            const uint8_t * blk = row + x * 110;     // sizeof(block_q3_K)
            const uint8_t * hm  = blk;               // hmask[32]
            const uint8_t * qs  = blk + 32;          // qs[64]
            if (sc_out) {
                uint8_t * o = sc_out + (size_t) (r * nblk + x) * 14;
                memcpy(o, blk + 96, 12);      // scales[12] raw
                memcpy(o + 12, blk + 108, 2); // d (fp16)
            }
            int8_t * wr = w_out + r * k + x * 256;
            for (int tp = 0; tp < 256; tp++) {
                int by = (tp / 128) * 32 + (tp % 32);
                int sh = 2 * ((tp % 128) / 32);
                int lo2 = (qs[by] >> sh) & 3;
                int hb  = (hm[tp % 32] >> (tp / 32)) & 1;
                wr[tp] = (int8_t) ((lo2 | (hb << 2)) - 4);
            }
        }
    }
    return 0;
}

// ---- q6_K extraction (canonical; never repacked on x86) --------------------
// block_q6_K (210 B / 256 weights): ql[128] (low 4 bits), qh[64] (high 2
// bits), scales[16] (int8 per 16-weight sub-block), fp16 d. Bit gather per
// dequantize_row_q6_K: half h=t/128, p=t%128, quarter=p/32, l=p%32:
//   q0 = (ql[h*64+l]    & 0xF) | ((qh[h*32+l]>>0 & 3) << 4)
//   q1 = (ql[h*64+32+l] & 0xF) | ((qh[h*32+l]>>2 & 3) << 4)
//   q2 = (ql[h*64+l]    >> 4)  | ((qh[h*32+l]>>4 & 3) << 4)
//   q3 = (ql[h*64+32+l] >> 4)  | ((qh[h*32+l]>>6 & 3) << 4)
// Integer mapping: w = q - 32 in {-32..31} (6-bit two's-complement).
// sc_out (optional): per (row, superblock) 18-byte record
//   [ scales[16] raw int8 | fp16 d ].
static int mv_extract_q6_K(const struct ggml_tensor * t, int8_t * w_out,
                           uint8_t * sc_out) {
    const int64_t k = t->ne[0], m = t->ne[1];
    if (k % 256 != 0) return -1;
    const int64_t nblk = k / 256;
    for (int64_t r = 0; r < m; r++) {
        const uint8_t * row = (const uint8_t *) t->data + r * t->nb[1];
        for (int64_t x = 0; x < nblk; x++) {
            const uint8_t * blk = row + x * 210;     // sizeof(block_q6_K)
            const uint8_t * ql  = blk;
            const uint8_t * qh  = blk + 128;
            if (sc_out) {
                uint8_t * o = sc_out + (size_t) (r * nblk + x) * 18;
                memcpy(o, blk + 192, 16);     // scales[16] (int8)
                memcpy(o + 16, blk + 208, 2); // d (fp16)
            }
            int8_t * wr = w_out + r * k + x * 256;
            for (int tp = 0; tp < 256; tp++) {
                int h = tp / 128, p = tp % 128, qtr = p / 32, l = p % 32;
                const uint8_t * qlh = ql + h * 64;
                const uint8_t * qhh = qh + h * 32;
                int q;
                switch (qtr) {
                    case 0:  q = (qlh[l]      & 0xF) | (((qhh[l] >> 0) & 3) << 4); break;
                    case 1:  q = (qlh[l + 32] & 0xF) | (((qhh[l] >> 2) & 3) << 4); break;
                    case 2:  q = (qlh[l]      >> 4)  | (((qhh[l] >> 4) & 3) << 4); break;
                    default: q = (qlh[l + 32] >> 4)  | (((qhh[l] >> 6) & 3) << 4); break;
                }
                wr[tp] = (int8_t) (q - 32);
            }
        }
    }
    return 0;
}

// pack bit (p+shift) of vals[rows*k] (two's-complement masked) into
// plane-major bitplanes: plane p row r byte n>>3, LSB-first — the server's
// wire format. shift>0 selects higher source bits (q6_K handle split).
static void mv_pack_planes(const int8_t * vals, int64_t rows, int64_t k,
                           int nbits, int shift,
                           uint8_t * out /* nbits*rows*bpr, zeroed */) {
    const size_t bpr = (size_t) ((k + 7) / 8);
    for (int p = 0; p < nbits; p++) {
        for (int64_t r = 0; r < rows; r++) {
            uint8_t * dst = out + ((size_t) p * rows + r) * bpr;
            const int8_t * src = vals + r * k;
            for (int64_t n = 0; n < k; n++) {
                if ((((uint32_t) (uint8_t) src[n]) >> (p + shift)) & 1)
                    dst[n >> 3] |= (uint8_t) (1u << (n & 7));
            }
        }
    }
}

static void mv_oplog(const char * line) {
    fprintf(stderr, "%s", line);
    FILE * f = fopen(g_cfg.oplog, "a");
    if (f) { fputs(line, f); fclose(f); }
}

// ---- ground-truth dump (per type; magics distinguish the layouts) ----------
// q4_0: 0x4450564D  hdr {magic,K,M} name[128] w[M*K] d16[M*(K/32)]  (07-18
//        format, check_derepack.py-compatible).
// q2_K: 0x4432564D  hdr name w[M*K] then per (row,blk) 20-byte records
//        [scales[16] | d | dmin]           -> check_q2k.py
// q3_K: 0x4433564D  hdr name w[M*K] then per (row,blk) 14-byte records
//        [scales[12] raw | d]              -> check_q3k.py
// q6_K: 0x4436564D  hdr name w[M*K] then per (row,blk) 18-byte records
//        [scales[16] int8 | d]             -> check_q6k.py
static void mv_dump_tensor(const struct ggml_tensor * src0, int repack,
                           const int8_t * w, const uint8_t * scpay,
                           size_t scpay_len, uint32_t magic) {
    FILE * f = fopen(g_cfg.dump, "wb");
    if (!f) return;
    const int64_t k = src0->ne[0], m = src0->ne[1];
    uint32_t hdr[3] = { magic, (uint32_t) k, (uint32_t) m };
    char name[128]; memset(name, 0, sizeof(name));
    snprintf(name, sizeof(name), "%s", src0->name);
    fwrite(hdr, 4, 3, f); fwrite(name, 1, 128, f);
    fwrite(w, 1, (size_t) m * k, f);
    if (scpay && scpay_len) fwrite(scpay, 1, scpay_len, f);
    fclose(f);
    fprintf(stderr, "[mvdram-pim] dumped %s type=%s (repack=%d) -> %s\n",
            src0->name, ggml_type_name(src0->type), repack, g_cfg.dump);
    g_dumped = 1;
}

// ---- the sampled op (thread 0 only) ---------------------------------------

static void mv_verify_op(const struct ggml_tensor * dst, int repack) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
    const enum ggml_type t0 = src0->type;
    const int64_t k = src0->ne[0], m = src0->ne[1];
    const size_t bpr = (size_t) ((k + 7) / 8);
    char line[640];

    // 1) tensor cache
    mv_tensor_t * ten = NULL;
    for (int i = 0; i < g_n_tensors; i++) {
        if (g_tensors[i].data == src0->data && g_tensors[i].k == k &&
            g_tensors[i].m == m && g_tensors[i].repack == repack &&
            g_tensors[i].type == t0) {
            ten = &g_tensors[i]; break;
        }
    }
    if (!ten) {
        if (g_n_tensors >= MV_MAX_TENSORS) {
            static int warned = 0;
            if (!warned) {
                warned = 1;
                fprintf(stderr, "[mvdram-pim] tensor cache full (%d) — further "
                        "NEW tensors stay CPU-only (cached ones keep working)\n",
                        MV_MAX_TENSORS);
            }
            return; // non-fatal: sampling continues on already-cached tensors
        }
        mv_hspec_t hs[3] = { { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 } };
        const int n_h = mv_handle_plan(t0, hs);
        if (n_h == 0) { g_pim_dead = 1; return; }

        int8_t * w = (int8_t *) malloc((size_t) m * k);
        if (!w) { g_pim_dead = 1; return; }
        const int want_dump = g_cfg.dump && !g_dumped &&
            (!g_cfg.dump_tensor || strstr(src0->name, g_cfg.dump_tensor));
        uint8_t * scpay = NULL; size_t scpay_len = 0; uint32_t dmagic = 0;

        int erc = -1;
        if (t0 == GGML_TYPE_Q4_0) {
            uint16_t * d16 = NULL;
            if (want_dump) {
                scpay_len = (size_t) m * (k / 32) * 2; dmagic = 0x4450564Du;
                d16 = (uint16_t *) malloc(scpay_len);
                scpay = (uint8_t *) d16;
            }
            erc = mv_extract_q4_0(src0, repack, w, d16);
        } else if (t0 == GGML_TYPE_Q2_K) {
            if (want_dump) {
                scpay_len = (size_t) m * (k / 256) * 20; dmagic = 0x4432564Du;
                scpay = (uint8_t *) malloc(scpay_len);
            }
            erc = mv_extract_q2_K(src0, w, scpay);
        } else if (t0 == GGML_TYPE_Q3_K) {
            if (want_dump) {
                scpay_len = (size_t) m * (k / 256) * 14; dmagic = 0x4433564Du;
                scpay = (uint8_t *) malloc(scpay_len);
            }
            erc = mv_extract_q3_K(src0, w, scpay);
        } else if (t0 == GGML_TYPE_Q6_K) {
            if (want_dump) {
                scpay_len = (size_t) m * (k / 256) * 18; dmagic = 0x4436564Du;
                scpay = (uint8_t *) malloc(scpay_len);
            }
            erc = mv_extract_q6_K(src0, w, scpay);
        }
        if (erc != 0) {
            fprintf(stderr, "[mvdram-pim] extract failed (type=%s repack=%d)\n",
                    ggml_type_name(t0), repack);
            free(w); free(scpay); g_pim_dead = 1; return;
        }
        if (want_dump) mv_dump_tensor(src0, repack, w, scpay, scpay_len, dmagic);
        free(scpay);

        // LOAD each handle: frames are ALWAYS marshaled (dry = full marshal,
        // just never sent).
        double load_s = 0.0;
        uint32_t handles[3] = { 0, 0, 0 };
        for (int j = 0; j < n_h; j++) {
            handles[j] = g_next_handle++;
            const size_t payload = (size_t) hs[j].qbits * m * bpr;
            uint8_t * frame = (uint8_t *) calloc(1, 20 + payload);
            if (!frame) { free(w); g_pim_dead = 1; return; }
            uint32_t h[5] = { MV_MAGIC_LOAD, handles[j], (uint32_t) hs[j].qbits,
                              (uint32_t) k, (uint32_t) m };
            memcpy(frame, h, 20);
            mv_pack_planes(w, m, k, hs[j].qbits, hs[j].shift, frame + 20);
            if (!g_cfg.dry) {
                if (mv_ensure_server() != 0) { free(frame); free(w); g_pim_dead = 1; return; }
                double t0s = mv_now();
                int rc = mv_send_req(frame, (uint32_t) (20 + payload));
                free(frame);
                uint8_t * resp = NULL; uint32_t rlen = 0;
                if (rc != 0 || mv_read_resp(&resp, &rlen) != 0 || rlen < 12) {
                    fprintf(stderr, "[mvdram-pim] LOAD failed (%s)\n", strerror(errno));
                    free(resp); free(w); g_pim_dead = 1; return;
                }
                uint32_t am, ah, ast;
                memcpy(&am, resp, 4); memcpy(&ah, resp + 4, 4); memcpy(&ast, resp + 8, 4);
                free(resp);
                (void) ah;
                if (am != MV_MAGIC_LOAD_ACK || ast != 0) {
                    fprintf(stderr, "[mvdram-pim] LOAD ack bad: magic=%#x status=%u\n", am, ast);
                    free(w); g_pim_dead = 1; return;
                }
                load_s += mv_now() - t0s;
            } else {
                free(frame);
            }
        }
        g_tensors[g_n_tensors] = (mv_tensor_t) { src0->data, t0, k, m, repack,
                                                 n_h, { handles[0], handles[1], handles[2] },
                                                 { hs[0], hs[1], hs[2] }, w, load_s };
        ten = &g_tensors[g_n_tensors++];
        fprintf(stderr, "[mvdram-pim] LOAD handles=%d (first=%u) tensor=%s type=%s "
                "K=%lld M=%lld repack=%d load=%.2fs%s\n", n_h, handles[0],
                src0->name, ggml_type_name(t0),
                (long long) k, (long long) m, repack, load_s,
                g_cfg.dry ? " (dry: marshaled, not sent)" : "");
    }

    // 2) quantize activations fp32 -> signed rbits two's-complement
    const int rb = g_cfg.rbits;
    const float * x = (const float *) src1->data;
    float amax = 0.0f;
    for (int64_t n = 0; n < k; n++) { float a = fabsf(x[n]); if (a > amax) amax = a; }
    const int qmax = (rb > 1) ? ((1 << (rb - 1)) - 1) : 1;
    const float scale = (amax > 0.0f) ? amax / (float) qmax : 1.0f;
    int8_t * xq = (int8_t *) malloc((size_t) k);
    if (!xq) { g_pim_dead = 1; return; }
    long setbits = 0;
    for (int64_t n = 0; n < k; n++) {
        long v = lrintf(x[n] / scale);
        if (rb > 1) {
            if (v > qmax) v = qmax;
            if (v < -qmax - 1) v = -qmax - 1;
        } else {
            v = (v > 0) ? 1 : 0; // rb==1: server convention is binary {0,1}
        }
        xq[n] = (int8_t) v;
        uint32_t u = ((uint32_t) (uint8_t) xq[n]) & ((1u << rb) - 1u);
        setbits += __builtin_popcount(u);
    }

    // 3) host integer reference over the SAME ints (full-width per type;
    //    for q6_K this is what the 3-handle sum must reproduce)
    int64_t * yref = (int64_t *) malloc((size_t) m * 8);
    if (!yref) { free(xq); g_pim_dead = 1; return; }
    for (int64_t r = 0; r < m; r++) {
        const int8_t * wr = ten->w + r * k;
        int64_t acc = 0;
        for (int64_t n = 0; n < k; n++) acc += (int64_t) wr[n] * xq[n];
        yref[r] = acc;
    }

    // 4) GEMV frame(s): activation planes marshaled once, shared per handle
    const size_t xpl = (size_t) rb * bpr;
    uint8_t * xplanes = (uint8_t *) calloc(1, xpl);
    if (!xplanes) { free(xq); free(yref); g_pim_dead = 1; return; }
    mv_pack_planes(xq, 1, k, rb, 0, xplanes);

    if (g_cfg.dry) {
        int64_t ymax = 0, ynz = 0; double ysum = 0.0;
        for (int64_t r = 0; r < m; r++) {
            int64_t a = yref[r] < 0 ? -yref[r] : yref[r];
            if (a > ymax) ymax = a;
            if (yref[r] != 0) ynz++;
            ysum += (double) a;
        }
        g_ops_done++;
        snprintf(line, sizeof(line),
                 "[mvdram-pim] OP#%ld DRY tensor=%s type=%s K=%lld M=%lld repack=%d "
                 "handles=%d r=%d amax=%.4g plane_density=%.3f "
                 "href: nz=%lld/%lld max|y|=%lld mean|y|=%.1f "
                 "(extraction+full marshal, no silicon)\n",
                 g_ops_done, src0->name, ggml_type_name(t0),
                 (long long) k, (long long) m, repack, ten->n_h, rb, amax,
                 (double) setbits / ((double) rb * k),
                 (long long) ynz, (long long) m, (long long) ymax,
                 ysum / (double) m);
        mv_oplog(line);
        free(xplanes); free(xq); free(yref);
        return;
    }

    // 4b) on silicon: one GEMV per handle, host-summed with the plan's
    //     multipliers (q4_0/q2_K: single handle, mult 1)
    int64_t * ypim = (int64_t *) calloc((size_t) m, 8);
    if (!ypim) { free(xplanes); free(xq); free(yref); g_pim_dead = 1; return; }
    double gemv_s = 0.0;
    for (int j = 0; j < ten->n_h; j++) {
        uint8_t * frame = (uint8_t *) malloc(12 + xpl);
        if (!frame) { free(ypim); free(xplanes); free(xq); free(yref); g_pim_dead = 1; return; }
        uint32_t h[3] = { MV_MAGIC_GEMV, ten->handle[j], (uint32_t) rb };
        memcpy(frame, h, 12);
        memcpy(frame + 12, xplanes, xpl);
        double t0s = mv_now();
        int rc = mv_send_req(frame, (uint32_t) (12 + xpl));
        free(frame);
        uint8_t * resp = NULL; uint32_t rlen = 0;
        if (rc != 0 || mv_read_resp(&resp, &rlen) != 0 || rlen < 16) {
            fprintf(stderr, "[mvdram-pim] GEMV failed (%s)\n", strerror(errno));
            free(resp); free(ypim); free(xplanes); free(xq); free(yref);
            g_pim_dead = 1; return;
        }
        gemv_s += mv_now() - t0s;
        uint32_t gm, gh, gst, gM;
        memcpy(&gm, resp, 4); memcpy(&gh, resp + 4, 4);
        memcpy(&gst, resp + 8, 4); memcpy(&gM, resp + 12, 4);
        (void) gh;
        if (gm != MV_MAGIC_GEMV_ACK || gst != 0 || gM != (uint32_t) m ||
            rlen < 16 + (size_t) m * 4) {
            fprintf(stderr, "[mvdram-pim] GEMV ack bad: magic=%#x status=%u M=%u len=%u\n",
                    gm, gst, gM, rlen);
            free(resp); free(ypim); free(xplanes); free(xq); free(yref);
            g_pim_dead = 1; return;
        }
        const int32_t mult = ten->hs[j].mult;
        for (int64_t r = 0; r < m; r++) {
            int32_t y; memcpy(&y, resp + 16 + (size_t) r * 4, 4);
            ypim[r] += (int64_t) mult * y;
        }
        free(resp);
    }

    // 5) verdict
    int64_t n_exact = 0, max_err = 0; double sum_err = 0.0;
    for (int64_t r = 0; r < m; r++) {
        int64_t e = ypim[r] - yref[r];
        if (e == 0) n_exact++;
        int64_t a = e < 0 ? -e : e;
        if (a > max_err) max_err = a;
        sum_err += (double) a;
    }
    free(ypim); free(xplanes); free(xq); free(yref);
    g_ops_done++;
    snprintf(line, sizeof(line),
             "[mvdram-pim] OP#%ld tensor=%s type=%s K=%lld M=%lld repack=%d r=%d "
             "handles=%d load=%.2fs gemv=%.1fs int-exact=%lld/%lld (%.4f%%) "
             "max|err|=%lld mean|err|=%.4f plane_density=%.3f route=c2(dst=CPU)\n",
             g_ops_done, src0->name, ggml_type_name(t0),
             (long long) k, (long long) m, repack, rb,
             ten->n_h, ten->load_s, gemv_s,
             (long long) n_exact, (long long) m, 100.0 * (double) n_exact / (double) m,
             (long long) max_err, sum_err / (double) m,
             (double) setbits / ((double) rb * k));
    mv_oplog(line);
    ten->load_s = 0.0; // report LOAD once; later hits are cache hits
}

// ---- entry points ----------------------------------------------------------

bool mvdram_pim_maybe_intercept_ext(const struct ggml_compute_params * params,
                                    struct ggml_tensor * dst,
                                    int inter_size, int nb_cols) {
    const int mode = mvdram_mode();
    if (mode == 0) {
        return false;
    }
    // Threads ith!=0 never do work and never claim the op (mode-1 route c2:
    // the op is ALWAYS computed by the stock CPU path on all threads).
    if (params->ith != 0) {
        return false;
    }
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
    const enum ggml_type t0 = src0->type;
    const int64_t k = src0->ne[0], m = src0->ne[1], batch = src1->ne[1];

    // census (both modes)
    pthread_mutex_lock(&g_mu);
    int i = 0;
    for (; i < g_n_shapes; i++) {
        if (g_shapes[i].t0 == t0 && g_shapes[i].k == k &&
            g_shapes[i].m == m && g_shapes[i].batch == batch) {
            g_shapes[i].count++;
            break;
        }
    }
    if (i == g_n_shapes && g_n_shapes < MVDRAM_MAX_SHAPES) {
        g_shapes[g_n_shapes++] = (mvdram_shape) { t0, k, m, batch, 1 };
    }
    pthread_mutex_unlock(&g_mu);
    if (mode == 1) {
        return false; // census only
    }

    // ---- mode 1 (route c2) ----
    if (g_pim_dead) return false;
    mv_load_cfg();

    // eligibility: GeMV (batch 1), f32 activations, type+shape in the census
    // table (or within server caps under MVDRAM_PIM_ANY_SHAPE=1)
    if (t0 != GGML_TYPE_Q4_0 && t0 != GGML_TYPE_Q2_K && t0 != GGML_TYPE_Q3_K &&
        t0 != GGML_TYPE_Q6_K)
        return false;
    if (batch != 1 || src1->ne[2] != 1 || src1->ne[3] != 1 ||
        src0->ne[2] != 1 || src0->ne[3] != 1) return false;
    if (src1->type != GGML_TYPE_F32 || src1->nb[0] != sizeof(float)) return false;
    int shape_ok = 0;
    if (g_cfg.any_shape) {
        const int64_t blk = (t0 == GGML_TYPE_Q4_0) ? 32 : 256;
        shape_ok = (k % blk == 0 && k <= MV_SRV_KMAX && m <= MV_SRV_MMAX);
    } else {
        for (size_t s = 0; s < sizeof(MV_SHAPES) / sizeof(MV_SHAPES[0]); s++) {
            if (MV_SHAPES[s].t == t0 && MV_SHAPES[s].k == k && MV_SHAPES[s].m == m) {
                shape_ok = 1; break;
            }
        }
    }
    if (!shape_ok) return false;
    int repack;
    if (inter_size == 0 && nb_cols == 0) repack = 0;      // canonical layout
    else if (t0 == GGML_TYPE_Q4_0 && inter_size == 8 && nb_cols == 8) repack = 8;
    else {
        // e.g. block_q2_Kx8 (needs AVX512 — not this tower) or a future
        // variant: censused only until a de-interleave is implemented.
        static int warned = 0;
        if (!warned) {
            warned = 1;
            fprintf(stderr, "[mvdram-pim] repack %dx%d not supported for "
                    "interception (type=%s tensor=%s) — censused only\n",
                    inter_size, nb_cols, ggml_type_name(t0), src0->name);
        }
        return false;
    }
    if (g_cfg.only && !strstr(src0->name, g_cfg.only)) return false;
    {
        static int announced_types[8];
        static int n_announced = 0;
        int seen = 0;
        for (int a = 0; a < n_announced; a++) if (announced_types[a] == (int) t0) seen = 1;
        if (!seen && n_announced < 8) {
            announced_types[n_announced++] = (int) t0;
            fprintf(stderr, "[mvdram-pim] eligible path: src0->type=%s tensor=%s "
                    "K=%lld M=%lld layout=%s (inter=%d nb_cols=%d)\n",
                    ggml_type_name(t0), src0->name, (long long) k, (long long) m,
                    repack ? "q4_0x8 (repacked)" : "canonical",
                    inter_size, nb_cols);
        }
    }

    // sampling + cap (thread 0 only touches these)
    if (g_ops_done >= g_cfg.max_ops) return false;
    g_seen_eligible++;
    if ((g_seen_eligible - 1) % (uint64_t) g_cfg.sample != 0) return false;

    mv_verify_op(dst, repack);
    return false; // route c2: dst is ALWAYS computed by the stock CPU path
}

bool mvdram_pim_maybe_intercept(const struct ggml_compute_params * params,
                                struct ggml_tensor * dst) {
    return mvdram_pim_maybe_intercept_ext(params, dst, 0, 0);
}
