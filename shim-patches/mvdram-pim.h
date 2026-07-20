// MVDRAM reproduction (Lane 2): mulmat_op interception point, per the
// paper's §VIII-A protocol ("we replace mulmat_op operations with our
// MVDRAM implementation"). Additive file — upstream llama.cpp untouched
// except the two-line call in ggml_compute_forward_mul_mat and the
// repack-path call in repack.cpp's compute_forward.
//
// Modes (env MVDRAM_PIM):
//   unset      — dead: single cached getenv, zero interception.
//   "census"   — record every mulmat shape (type, K=ne00, M=ne01, batch),
//                fall through to the stock CPU path. Shape table written
//                at exit to MVDRAM_PIM_LOG (default mvdram_census.tsv in
//                cwd). Feeds the per-op benchmark list (B2) and the
//                GeMV-eligibility audit.
//   "1"        — sampled PIM execution of eligible GeMVs on the Lane-2
//                in-DRAM server (route c2, 2026-07-18): the sampled op's
//                INTEGER bitplane GeMV (the part MVDRAM's §VI-B math puts
//                in DRAM) runs on silicon and is verified against a host
//                integer reference over the same quantized ints and
//                activations; dst always gets the stock CPU fp32 result
//                (per-block fp16/int scales make the exact fp32 op
//                inexpressible as ONE whole-K integer GeMV, and the paper
//                is silent on scale handling — fp stays on the processor
//                per their §V-A). Per-op verdicts logged to
//                MVDRAM_PIM_OPLOG (default mvdram_pim_ops.log).
//
// Eligible types/shapes (2026-07-19, B1): q4_0 (canonical + q4_0x8),
// q2_K (canonical; repack is AVX512-only, never taken on this tower),
// q3_K (canonical; NO repack variant exists for q3_K on any arch — the
// Q2_K models' v/o/ffn tensors), q6_K head (canonical; 3-handle plane
// split under the server's qb<=4 cap: y = y[qb4 b0..3] + 16*y[qb1 b3] +
// 16*y[qb2 b4..5]); shapes = Llama-2-7B census dims {4096x4096,
// 4096->11008, 11008->4096, 32000x4096(q6_K)} for BOTH quant families.
// Integer mappings: q4_0 q-8, q2_K q-2, q3_K q-4, q6_K q-32 (two's-
// complement; scales stay host-side — see mvdram-pim.c header). K=11008
// fits the server's K<=16384 in one LOAD/GEMV (no host K-split); M up to
// 52K fits one pass.
//
// Mode-1 env knobs (defaults in parentheses):
//   MVDRAM_PIM_SAMPLE (9999)  intercept every Nth eligible GeMV
//   MVDRAM_PIM_MAX_OPS (3)    hard cap of PIM ops per process
//   MVDRAM_PIM_RBITS (8)      activation quantization bits (1..8, signed
//                             two's-complement for >1)
//   MVDRAM_PIM_SERVER/CALIB/COLMASK/BENDER/BANK/SID  server binary+argv
//                             (defaults = the verified bender2/bank0/s86)
//   MVDRAM_PIM_VOTE3 (0)      3x vote on the server (3x wall)
//   MVDRAM_PIM_RESP_TIMEOUT_S (1800)  poll timeout per server response
//   MVDRAM_PIM_DRY (0)        extract + FULL marshal (LOAD/GEMV frames
//                             built) but nothing sent — no silicon; for
//                             extraction/marshaling validation
//   MVDRAM_PIM_DUMP (unset)   dump one eligible tensor's full ints + raw
//                             scale payload to this path (ground-truth
//                             check against the gguf; format per type —
//                             see mv_dump_tensor in mvdram-pim.c)
//   MVDRAM_PIM_DUMP_TENSOR (unset)  substring filter choosing WHICH
//                             eligible tensor gets dumped (default: first)
//   MVDRAM_PIM_ONLY (unset)   substring filter on tensor name for
//                             eligibility (e.g. "ffn_down" to target the
//                             11008-contraction ops)
//   MVDRAM_PIM_ANY_SHAPE (0)  1 = accept any dims within server caps
//                             (K<=16384, M<=65536) instead of the census
//                             shape table
//
// Threading contract (mode 1): every thread returns false, so ggml runs
// the stock CPU op on all threads exactly as if the shim were absent;
// thread 0 additionally performs the PIM call + verification BEFORE
// returning false. No dst writes, no extra barriers — threads ith!=0 that
// enter the op's internal ggml_barrier simply wait until thread 0 arrives
// (barriers count all nth threads). This is the safe degenerate case of
// the LANE2_GEMV_SERVER.md thread contract.
#pragma once
#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_compute_params;

// Returns true if the op was fully handled on PIM (caller must return);
// false = caller continues with the stock CPU path. (Route c2 always
// returns false: PIM runs as a verified side-channel.)
bool mvdram_pim_maybe_intercept(const struct ggml_compute_params * params,
                                struct ggml_tensor * dst);

// Repack-path variant: inter_size/nb_cols are the repack template
// constants (e.g. 8,8 for block_q4_0x8 on AVX2) so the shim can
// de-interleave src0->data; (0,0) = canonical layout.
bool mvdram_pim_maybe_intercept_ext(const struct ggml_compute_params * params,
                                    struct ggml_tensor * dst,
                                    int inter_size, int nb_cols);

#ifdef __cplusplus
}
#endif
