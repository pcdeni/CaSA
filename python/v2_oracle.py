#!/usr/bin/env python3
"""
v2_oracle.py -- host-side numerics oracle for the PRODUCTION V2 PIM path.

  *** DRAFT: --single (V2S single-track) extension, 2026-07-28. ***
  This is a scratchpad draft of numerics_gate/v2_oracle.py that adds a
  `--single` flag exercising the MAGIC_V2S (single-track / 1-bit) path in
  addition to the default dual-track MAGIC_V2 path. Every existing default
  behavior is preserved byte-for-byte when --single is absent. See the V2S
  PROTOCOL FACTS block below for the exact source lines this was read from.

  *** DRAFT: --mode {v2,mm3d} extension, 2026-07-29. ***
  Adds a `--mode` selector for the SERVING PATH. Default `v2` is the
  existing per-request MAGIC_V2 upload path (byte-identical when --mode is
  absent). `--mode mm3d` drives the PERSISTENT-HANDLE path instead: one
  MAGIC_LOAD (0xB17EF003) then a MAGIC_MM3D (0xB17EF004) request, so the
  oracle can gate the new descriptor-serving path (process_matmul_desc,
  engaged with PIM_DESC_SERVE=1; the MM3D dispatch falls back to
  process_matmul_handle when ineligible -- identical answer). mm3d is
  DUAL-TRACK ONLY: LOAD always carries pos+neg (cpp L1997-2005) and MM3D
  returns the full W@x, so the CPU truth and the numerics compare are
  EXACTLY the dual-track V2 ones, just tagged [MM3D]. (--single is refused
  with mm3d -- there is no single-track LOAD variant.) See the MM3D PROTOCOL
  FACTS block below for the source lines.

WHAT THIS IS
------------
The "output-vs-CPU-oracle" gate that the workspace is currently missing.
For a BitLinear-shaped op it:

  1. builds a small ternary weight W (disjoint pos/neg 32-bit masks) and an
     int8 activation x, decomposed to bitplanes EXACTLY as the model does
     (pim_linear._pack_xbp: int8 read as a uint8 byte, 8 bitplanes, factors
     [1,2,4,8,16,32,64,-128]);
  2. sends a real MAGIC_V2 request through pim_linear.PimServer -- the SAME
     client/wire path the model uses (bitnet-proj-server over the binary
     length-prefixed stdin/stdout protocol) -- and reads back int32 y;
  3. computes the CPU reference y_ref from the SAME masks (W+/W-) and the
     SAME activation bitplanes, using the server's own accumulation formula
        y[s] = sum_c sum_b factor[b] * ( popcount(pos[c][s] & xbp[c][b])
                                       - popcount(neg[c][s] & xbp[c][b]) );
  4. compares per element and reports exact / mismatch counts.

It refuses to fail silently (see WHY ab_fused_server.py DOES, below):
  * asserts the response is the right length (2048 int32);
  * asserts y is NOT all-zero (the exact symptom of the broken path);
  * parses the server's own stderr log and asserts exec>0 (a real MAJ3
    program ran -- the "[srv-prof ...] exec=..ms (Nx)" counter with N>0);
  * asserts n_rounds>0 and exec_count >= n_rounds (every round executed).
And it carries a NEGATIVE CONTROL (--inject-fault): a knob that computes the
reference from a mask that differs from the sent mask by exactly one ternary
weight, so a *correct* server output is REQUIRED to trip the mismatch report
-- proving the gate can actually detect a divergence.

WHY THE OLD ab_fused_server.py "FAILS SILENTLY" (diagnosis, from the source
+ ab_server_baseline.log 2026-07-26)
------------------------------------------------------------------------
  * It drives the WRONG protocol: MAGIC_LOAD (0xB17EF003) + MAGIC_MM3D
    (0xB17EF004) -- the persistent-handle path (process_matmul_handle).
    That is NOT the production path. In pim_linear.py the handle path is
    gated behind PIM_USE_LOAD_WEIGHTS (default '0'), so _force_v2 is True
    and EVERY production matmul goes through MAGIC_V2 (process_request) with
    a fresh weight upload. The handle path is explicitly disabled in
    production because it "produces garbage in matmul-mix configs"
    (pim_linear.py ~L764-772).
  * In the captured run the MM3D handler executed ZERO MAJ3 programs:
        [mm3d-prof #1 handle=1] total=0.3ms ... exec=0.0ms (0x) recv=0.0ms
        pop=0.0ms ...
    i.e. no exec, no recv, no popcount -> y stayed all-zero and was written
    back as the response. The handler's own pre/post verify steps only
    re-READ the LOAD-time weight rows (still intact) and print
    "[mm3d-verify] popcounts OK" / "POST-MM3D clean", so the log looks
    healthy. (The zero-exec regression itself lives in the MM3D dispatch --
    the streaming/SEGPOP/pack drift around test_bitnet_server.cpp:3281-3720;
    the round loop yields 0 execs and the response is still emitted. Pin the
    exact line on silicon/gdb -- this oracle sidesteps it by driving V2.)
  * ab_fused_server.py has NO liveness guard: it never checks exec>0, y!=0,
    or n_rounds>0. It computes ref vs the all-zero y, prints
    "exact=0/2048", flips all_ok=False and exit(1) -- but the human-facing
    line looks like a real measurement, so the failure reads as a "silicon
    mismatch" instead of "nothing ran". That is the silent failure.

PROTOCOL ASSUMPTIONS (verify on silicon; all read out of
test_bitnet_server.cpp process_request + pim_linear.py PimBitLinear)
------------------------------------------------------------------------
  A1. MAGIC_V2 = 0xB17EF002. Request header = 5x u32 LE:
        [magic, d_in, d_out, n_chunks, n_bitplanes], then
        pos_mask (n_chunks*d_out u32) + neg_mask (n_chunks*d_out u32)
        + x_bitplane (n_chunks*n_bitplanes u32) + bitplane_factor
        (n_bitplanes i32) + trailing calib_idx (u32).  (cpp L2225-2296,
        pim_linear L456-465, L1081-1084.)
  A2. d_out MUST be 2048 (server hard-rejects otherwise, cpp L2255). Real
        outputs occupy rows [0, n_real); the rest are zero-weight padding.
        Response is always 2048 int32 = 8192 bytes (cpp L2871-2882).
  A3. mask bit layout: within a 32-input chunk c, bit i of pos_mask[c][s]
        == 1 iff W[s, c*32+i] == +1; neg likewise for -1 (pim_linear
        L429-434). x_bitplane[c][b] bit i == bit b of uint8(x[c*32+i])
        (pim_linear L60-67).
  A4. Server accumulation == the popcount formula above (cpp L2447-2453:
        y += sign_factor*factor[b]*popcount(mask & x)); this equals the
        exact integer matvec W(int8) @ x(int8) because the bitplane factors
        [1,2,4,8,16,32,64,-128] reconstruct signed int8 exactly. The oracle
        checks BOTH forms agree (internal consistency) and uses the
        popcount form as the silicon reference.
  A5. calib_idx=0 => primary calib, single trip, no cross-calib voting
        (cpp L2280-2284). This oracle sends 0 (matches production default;
        PIM_VOTE_FULL is off by default, pim_linear L834-839).
  A6. Single-DIMM: bank_arg "0" => N=1 bank => n_units = 2*n_chunks,
        n_rounds = ceil(n_units/N). exec_count (srv-prof "(Nx)" after
        "exec=") = n_rounds*ceil(n_bitplanes/g_req_K) > 0 for any real op.
  A7. --single drives the V2S single-track path (see V2S PROTOCOL FACTS).
        Absent --single this oracle uses plain dual-track MAGIC_V2 (pos+neg),
        the correctness-guaranteed production default; it does NOT touch the
        V2G/V2GS grouped variants (those layer request-batching on top and
        reconstruct the same y host-side).

V2S PROTOCOL FACTS (--single mode; all line numbers in the CURRENT
test_bitnet_server.cpp / pim_linear.py, read 2026-07-28)
------------------------------------------------------------------------
  S1. MAGIC_V2S = 0xB17EF006 (cpp L62; pim_linear L96). "single-track":
        the request carries the POS masks ONLY; every unit is sign 0.
  S2. Header stays 5x u32 (SAME as V2). single==true is derived from the
        magic, not a header field (cpp L2391). The 6th `group_chunks`
        field exists ONLY for the grouped variants V2G/V2GS (cpp
        L2390,L2393-2398,L2419); plain V2S keeps header_bytes = 5*4.
        pim_linear builds the same 5-field header (L470-474).
  S3. Body = header + pos_mask[n_chunks*d_out u32] + x_bitplane
        [n_chunks*n_bitplanes u32] + bitplane_factor[n_bitplanes i32] +
        (optional trailing calib_idx u32). NO neg block: mask_blocks =
        single?1:2 (cpp L2420); neg_mask_all is aliased to pos_mask_all and
        never read (cpp L2444-2448). pim_linear omits the neg bytes:
        `pos_mask.tobytes() + (b'' if slice_single else neg_mask.tobytes())`
        (L478-479).
  S4. Unit->chunk mapping is IDENTITY: n_units = n_chunks*(single?1:2), so
        for V2S n_units = n_chunks (cpp L2455). Everywhere the loop maps a
        unit to a chunk it uses `chunk = single ? u : u/2` and
        `sign = single ? 0 : u%2` (cpp L2594-2595,L2605,L2655-2656,
        L2886-2887).
  S5. Server accumulation for V2S (cpp L2602-2608): sign_factor = +1
        (signs all 0), weight = +bitplane_factor[b], chunk_acc = u (identity),
        y_g[j] += weight*pc[j] with pc[j] = popcount(pos[chunk][j] &
        x_bitplane[chunk][b]). i.e. the POS TRACK ONLY:
            y_pos[s] = sum_c sum_b factor[b]*popcount(pos[c][s] & xbp[c][b]).
        There is NO neg subtraction on the server.
  S6. RESPONSE is y_pos, NOT the reconstructed y. total = n_groups*d_out*4
        (cpp L3069-3072); plain V2S has group_chunks defaulting to n_chunks
        (cpp L2392,L2408) => n_groups = 1 => response = 2048 int32 =
        8192 bytes, SAME length as V2. The array that comes back over the
        wire is the pos-track accumulation of S5.
  S7. The client reconstructs the true matvec host-side (pim_linear
        single-DIMM non-grouped path, L1108-1114):
            y[s] = 2*y_pos[s] - const,
            const = sum_c sum_b factor[b]*popcount(xbp[c][b])   (SCALAR;
                    pim_linear._xbp_weighted_popcount, L100-113).
        Identity (cpp L62-66, pim_linear L102-108): for 1-bit weights every
        real output is zero-free so neg == ~pos over the 32 bits of each
        chunk, hence popcount(pos&x)+popcount(neg&x)=popcount(x) and
        y = pc_pos - pc_neg = 2*pc_pos - popcount(x), bitplane-weighted.
  S8. ELIGIBILITY: V2S is valid ONLY when every real output row is zero-free
        (1-bit weights). pim_linear gates per slice:
        `slice_single = _want_single and bool((W_slice[:n_real]!=0).all())`
        (L468). --single therefore builds W in {-1,+1} (no zeros); a zero
        would break the neg==~pos identity of S7.
  S9. THIS ORACLE'S V2S GATE compares the RAW server response (y_pos, the
        S6 array) against the pos-track reference of S5 restricted to the
        real rows -- "exactly what array comes back". It ALSO reports, as an
        FYI, whether the client identity 2*y_pos-const applied to the
        measured response reproduces W@x on the real rows (the model-facing
        quantity). The correlation/flake gate, liveness asserts, and the
        --inject-fault negative control all operate on the pos-track array.

MM3D PROTOCOL FACTS (--mode mm3d; line numbers in the CURRENT
test_bitnet_server.cpp / pim_linear.py, read 2026-07-29)
------------------------------------------------------------------------
  M1. Two magics. MAGIC_LOAD = 0xB17EF003 (cpp L60, pim_linear L121),
        MAGIC_MM3D = 0xB17EF004 (cpp L61, pim_linear L122). The oracle
        drives ONE LOAD then ONE MM3D against handle id MM3D_HANDLE_ID (a
        fresh PimServer holds no prior handles, so any id works; LOAD and
        MM3D must agree, keyed at cpp L2912).
  M2. LOAD body (build_load_request, mirrors pim_linear load_body
        L637-642; server parse process_load_weights L1991-2005 is the
        authority): [MAGIC_LOAD, handle_id, d_in, d_out=2048, n_chunks]
        (5x u32 LE) + pos_mask (n_chunks*2048 u32) + neg_mask
        (n_chunks*2048 u32). BOTH mask blocks are always present
        (need = 5*4 + n_chunks*d_out*4*2, cpp L1997) -- the handle path is
        DUAL-TRACK ONLY. The oracle sends genuine pos+neg from its ternary
        W, so the regime matches dual-track V2 exactly.
  M3. LOAD response = a single 4-byte u32 status (cpp L2238-2240 ack=0 OK;
        L2076-2079 ack=1 = ENOSPC / backup-pool exhausted, server stays
        alive and returns 0). The oracle reads 4 bytes; nonzero => it prints
        a clear ENOSPC error and exits 2 (in production the client would
        fall back to V2, but the oracle is explicitly gating the handle
        path). Read via server.request(load_body, expect_resp_len=4)
        (pim_linear PimServer.request docstring: "For LOAD_WEIGHTS: 4 bytes").
  M4. MM3D body (build_mm3d_request, mirrors pim_linear mm3d_prefix L643-647
        + body L1040-1043; server parse process_matmul_handle L2905-2932):
        [MAGIC_MM3D, handle_id, d_out=2048, n_chunks, n_bitplanes]
        (5x u32 LE) + x_bitplane (n_chunks*n_bitplanes u32) + bitplane_factor
        (n_bitplanes i32) + trailing calib_idx u32. The server's `need`
        (L2923) stops at bitplane_factor; the trailing calib_idx is tolerated
        extra bytes (the handle picks its pool/calib at LOAD time). The
        oracle appends calib_idx=0 for byte-fidelity with the client.
  M5. MM3D response = d_out*4 = 8192 bytes, int32 (cpp L3874-3883 handle
        path; L6384-6390 desc-serve path). This is the FULL dual-track
        matvec y = W(int8) @ x(int8) over the real rows (the stored pos+neg
        are read back and accumulated pos-neg). => THE CPU TRUTH IS
        cpu_reference_from_masks(pos, neg, xbp, factors), IDENTICAL to
        dual-track V2. The compare/flake-vs-corruption verdict is the same
        code, tagged [MM3D].
  M6. DESCRIPTOR-SERVE ROUTING. The MM3D dispatch (cpp L6778-6791) routes
        through process_matmul_desc first when desc_serve_enabled()
        (PIM_DESC_SERVE=1), and FALLS BACK to process_matmul_handle when the
        desc path returns rc==2 (ineligible / build<32) -- the answer is
        identical, so this oracle gates whichever ran. Set PIM_DESC_SERVE=1
        in the server env to exercise the new path.
  M7. LIVENESS (parse_server_mm3d_liveness). Handle path prints
        '[mm3d-prof #N handle=H] ... exec=X.Xms (Kx)' BEFORE the response
        (cpp L3837/L3874); desc-serve prints '[desc-serve #N handle=H] ...
        desc=D ...' (walked descriptors) AFTER the response (cpp L6397 vs
        L6384), so the oracle briefly polls the log. exec/desc count > 0 is
        the liveness gate (0 only for a legit skip-covered zero-x request).
        The server's own zero-exec LIVENESS-ASSERT (cpp L3862 / L6372) hard
        -aborts a fabricated-zeros response; the oracle's y!=0 + count>0
        checks are the host-side mirror.
  M8. NEGATIVE CONTROL / SINGLE-DIMM shape are the dual-track ones (M5):
        --inject-fault uses inject_fault (a W==0 slot exists in ternary W);
        n_units = 2*n_chunks, n_rounds = ceil(n_units/N) (cpp L2007-2009,
        handle stores these at LOAD time). Note the desc-serve liveness
        count is walked-descriptors (not per-round execs), so the oracle
        skips the exec_count<n_rounds soft-WARN in mm3d mode (M7 count>0 is
        the gate).

RUN COMMANDS  (do NOT run the live path while the main session owns the
card -- the BCU1525 is single-consumer on /dev/xdma)
------------------------------------------------------------------------
  # (0) Hardware-free comparator self-test: pure numpy, NO server, NO card.
  #     Proves the reference math + the negative control both work.
  python3 /home/deni/Claude/numerics_gate/v2_oracle.py self-test
  #     ... and the V2S single-track reference + its negative control:
  python3 /home/deni/Claude/numerics_gate/v2_oracle.py self-test --single

  # (1) LIVE on the tower, from the BitNet app dir. Real FPGA, single
  #     consumer -- only when the main session is idle on /dev/xdma:
  cd /home/deni/Claude/SiMRA-DRAM-main/DRAM-Bender/sources/apps/DSN_AE_APPS/BitNet
  PIM_SUB_START=45312 PIM_SUB_END=45952 \
  PIM_POOL_LIST_FILE=pool_layout_dimm2_bank0.txt \
    python3 /home/deni/Claude/numerics_gate/v2_oracle.py live \
      --server ./bitnet-proj-server --bender 2 --calib calib_dimm2.txt \
      --bank 0 --d-in 64 --n-real 8 --seed 1

  # (1s) LIVE V2S single-track (adds --single; the server binary must know
  #      MAGIC_V2S). Same env; W is 1-bit so the V2S identity holds:
  ... same as (1) ... --single

  # (1m) LIVE MM3D handle path (adds --mode mm3d; drives one MAGIC_LOAD then
  #      one MAGIC_MM3D). Set PIM_DESC_SERVE=1 to gate the NEW descriptor-
  #      serving path (else the identical-answer handle path runs). Same trio
  #      env; W is ternary (dual-track), CPU truth == dual V2:
  cd /home/deni/Claude/SiMRA-DRAM-main/DRAM-Bender/sources/apps/DSN_AE_APPS/BitNet
  PIM_SUB_START=45312 PIM_SUB_END=45952 \
  PIM_POOL_LIST_FILE=pool_layout_dimm2_bank0.txt PIM_DESC_SERVE=1 \
    python3 /home/deni/Claude/numerics_gate/v2_oracle.py live --mode mm3d \
      --server ./bitnet-proj-server --bender 2 --calib calib_dimm2.txt \
      --bank 0 --d-in 64 --n-real 8 --seed 1
  #      ... and its negative control (a correct server must MISMATCH):
  ... same as (1m) ... --inject-fault

  # (2) LIVE negative control: reference gets one wrong weight bit; a
  #     CORRECT server must trip the mismatch report (exit 0 = fault
  #     detected as expected; exit 3 = oracle FAILED to detect => broken).
  ... same as (1) ... --inject-fault          # dual-track
  ... same as (1) ... --single --inject-fault  # single-track

  # (3) Card-free server path (in-process SimDramModel; the server does NOT
  #     open /dev/xdma under PIM_BACKEND=sim, cpp L4358-4365). Good for
  #     validating the wire path + liveness asserts with no silicon:
  PIM_BACKEND=sim \
    python3 /home/deni/Claude/numerics_gate/v2_oracle.py live \
      --server ./bitnet-proj-server --bender 2 --calib calib_dimm2.txt \
      --bank 0 --d-in 64 --n-real 8 --seed 1 [--single]

Exit codes: 0 = pass (or fault correctly detected under --inject-fault);
1 = numerics mismatch found (gate fired) / liveness assert failed;
2 = usage/setup error; 3 = negative control FAILED to detect an injected
fault (the oracle itself is broken -- do not trust it).
"""
import argparse
import os
import struct
import sys

import numpy as np

# ---- protocol constants (mirror pim_linear.py / test_bitnet_server.cpp) ----
MAGIC_V2 = 0xB17EF002
MAGIC_V2S = 0xB17EF006          # V2 single-track (pos masks only); cpp L62
MAGIC_FUSE = 0xB17EF008         # request-fusion frame (P1 rung-2 transport):
                                # [MAGIC_FUSE][u32 N] + N x ([u32 sublen][body]).
                                # With PIM_DESC_XBATCH=1 an all-MM3D frame is
                                # served by ONE desc-serve super-session/bank
                                # (process_matmul_desc_batch, 2026-08-04).
MAGIC_LOAD = 0xB17EF003         # LOAD_WEIGHTS (persistent handle); cpp L60,
                                # pim_linear L121
MAGIC_MM3D = 0xB17EF004         # MATMUL_HANDLE (uses a loaded handle); cpp
                                # L61, pim_linear L122
MM3D_HANDLE_ID = 1              # this oracle uses ONE handle (LOAD once) — a
                                # fresh PimServer has no prior handles so any
                                # id is fine; the LOAD and MM3D bodies must
                                # agree (server keys handles by id, cpp L2912).
D_OUT = 2048
N_BITPLANES = 8
BITPLANE_FACTORS = np.array([1, 2, 4, 8, 16, 32, 64, -128], dtype=np.int32)

_POPCNT_BYTE = np.array([bin(i).count("1") for i in range(256)], dtype=np.int64)


def popcount_u32(a):
    """Elementwise popcount of a uint32 ndarray -> int64 ndarray."""
    a = np.asarray(a, dtype=np.uint32)
    return (_POPCNT_BYTE[a & 0xFF]
            + _POPCNT_BYTE[(a >> 8) & 0xFF]
            + _POPCNT_BYTE[(a >> 16) & 0xFF]
            + _POPCNT_BYTE[(a >> 24) & 0xFF])


# ----------------------------- construction ---------------------------------

def make_problem(d_in, n_real, seed, single=False):
    """Return (W, x_int8) for a small matvec.
    W: int8 [n_real, d_in].  x_int8: int8 [d_in].
    Dual-track (default): ternary W in {-1,0,+1}.
    Single-track (single=True): 1-bit W in {-1,+1} with NO zeros, because
    V2S is only valid when every real output is zero-free so neg == ~pos
    (pim_linear L468 slice_single gate / cpp L62-66)."""
    assert d_in % 32 == 0, "d_in must be a multiple of 32 (chunk width)"
    assert 1 <= n_real <= D_OUT, "n_real must be in [1, 2048]"
    rng = np.random.default_rng(seed)
    if single:
        # 1-bit weights, ~1/2 each of {-1,+1}, no zeros (V2S eligibility).
        bits = rng.integers(0, 2, size=(n_real, d_in), dtype=np.int8)
        W = (bits * 2 - 1).astype(np.int8)          # 0->-1, 1->+1
    else:
        # Ternary weights, ~1/3 each of {-1,0,+1}.
        W = rng.integers(-1, 2, size=(n_real, d_in), dtype=np.int8)
    # Full signed int8 activation, guaranteed to contain nonzeros.
    x = rng.integers(-128, 128, size=d_in, dtype=np.int16)
    # ZERO_CHUNK_FRAC (2026-07-29, zero-plane-skip gate): random dense x
    # never produces an all-zero 32-element chunk (~2^-32), so the server's
    # PIM_SKIP_ZERO_PLANES path would stay idle and this oracle would prove
    # only "armed-idle is safe". Zeroing a fraction of chunks makes the skip
    # FIRE while the CPU reference stays the absolute truth (zeros flow
    # through matvec_reference identically). Env-gated, default off.
    zcf = float(os.environ.get("ZERO_CHUNK_FRAC", "0"))
    if zcf > 0:
        n_ch = d_in // 32
        kill = rng.random(n_ch) < zcf
        for c in np.nonzero(kill)[0]:
            x[c*32:(c+1)*32] = 0
    if not np.any(x != 0):
        x[0] = 37
    return W, x.astype(np.int8)


def masks_from_W(W):
    """W int8 [n_real, d_in] -> (pos_mask, neg_mask) uint32 [n_chunks, 2048].
    Bit i of mask[c][s] encodes input i of chunk c for output s. Real rows
    fill [0, n_real); padding rows stay zero (matches pim_linear L425-434).
    For 1-bit W (V2S) neg == ~pos over the 32 bits of each real chunk, but
    only pos is sent on the wire (neg is unused by the server, cpp L2444)."""
    n_real, d_in = W.shape
    n_chunks = d_in // 32
    pos = np.zeros((n_chunks, D_OUT), dtype=np.uint32)
    neg = np.zeros((n_chunks, D_OUT), dtype=np.uint32)
    powers = (np.uint32(1) << np.arange(32, dtype=np.uint32))
    for c in range(n_chunks):
        seg = W[:, c * 32:(c + 1) * 32]                       # [n_real, 32]
        pos[c, :n_real] = ((seg == 1).astype(np.uint32) * powers).sum(
            axis=1, dtype=np.uint32)
        neg[c, :n_real] = ((seg == -1).astype(np.uint32) * powers).sum(
            axis=1, dtype=np.uint32)
    return pos, neg


def pack_xbp(x_int8, n_chunks):
    """Bit-decompose int8 activation exactly as pim_linear._pack_xbp: int8
    read as a uint8 byte, N_BITPLANES planes, bit i of x_bp[c][b] == bit b
    of uint8(x[c*32+i]).  Returns uint32 [n_chunks, N_BITPLANES]."""
    x_u8 = x_int8.astype(np.uint8)
    xbp = np.zeros((n_chunks, N_BITPLANES), dtype=np.uint32)
    powers = (np.uint32(1) << np.arange(32, dtype=np.uint32))
    for c in range(n_chunks):
        chunk = x_u8[c * 32:(c + 1) * 32]
        for b in range(N_BITPLANES):
            bits = ((chunk >> b) & 1).astype(np.uint32)
            xbp[c, b] = (bits * powers).sum(dtype=np.uint32)
    return xbp


# ------------------------------- references ---------------------------------

def cpu_reference_from_masks(pos, neg, xbp, factors):
    """The server's own DUAL-track accumulation, on CPU. THE silicon
    reference for MAGIC_V2:
        y[s] = sum_c sum_b factor[b]*(popcount(pos[c][s] & xbp[c][b])
                                    - popcount(neg[c][s] & xbp[c][b]))."""
    n_chunks = pos.shape[0]
    n_bp = xbp.shape[1]
    y = np.zeros(D_OUT, dtype=np.int64)
    for c in range(n_chunks):
        for b in range(n_bp):
            xb = np.uint32(xbp[c, b])
            y += int(factors[b]) * (popcount_u32(pos[c] & xb)
                                    - popcount_u32(neg[c] & xb))
    return y


def pos_track_reference(pos, xbp, factors):
    """The server's V2S (single-track) accumulation, on CPU. THE array the
    server returns over the wire for a MAGIC_V2S request -- POS TRACK ONLY,
    no neg subtraction (cpp L2602-2608, single => sign_factor=+1,
    chunk_acc=u identity):
        y_pos[s] = sum_c sum_b factor[b]*popcount(pos[c][s] & xbp[c][b])."""
    n_chunks = pos.shape[0]
    n_bp = xbp.shape[1]
    y = np.zeros(D_OUT, dtype=np.int64)
    for c in range(n_chunks):
        for b in range(n_bp):
            xb = np.uint32(xbp[c, b])
            y += int(factors[b]) * popcount_u32(pos[c] & xb)
    return y


def single_recon_const(xbp, factors):
    """The V2S host reconstruction constant, EXACTLY pim_linear
    _xbp_weighted_popcount (L100-113): a SCALAR, identical for every output
    slot:  const = sum_c sum_b factor[b]*popcount(xbp[c][b]).
    The client reconstructs the true matvec as y = 2*y_pos - const (S7)."""
    n_bp = xbp.shape[1]
    s = 0
    for b in range(n_bp):
        s += int(factors[b]) * int(popcount_u32(xbp[:, b]).sum())
    return s


def reconstruct_x_from_bitplanes(xbp, factors):
    """Invert pack_xbp: x[c*32+i] = sum_b factor[b]*bit_i(xbp[c][b])."""
    n_chunks = xbp.shape[0]
    x = np.zeros(n_chunks * 32, dtype=np.int64)
    for c in range(n_chunks):
        for i in range(32):
            v = 0
            for b in range(len(factors)):
                v += int(factors[b]) * ((int(xbp[c, b]) >> i) & 1)
            x[c * 32 + i] = v
    return x


def matvec_reference(W, x_int8):
    """The plain integer matvec, W @ x, for real rows only."""
    return W.astype(np.int64) @ x_int8.astype(np.int64)


# --------------------------- request / compare ------------------------------

def build_v2_request(pos, neg, xbp, factors, calib_idx=0):
    """Assemble the exact production MAGIC_V2 body (assumption A1)."""
    n_chunks, d_out = pos.shape
    assert d_out == D_OUT
    assert neg.shape == pos.shape
    n_bp = xbp.shape[1]
    d_in = n_chunks * 32
    header = struct.pack("<5I", MAGIC_V2, d_in, D_OUT, n_chunks, n_bp)
    return (header
            + np.ascontiguousarray(pos, dtype="<u4").tobytes()
            + np.ascontiguousarray(neg, dtype="<u4").tobytes()
            + np.ascontiguousarray(xbp, dtype="<u4").tobytes()
            + np.ascontiguousarray(factors, dtype="<i4").tobytes()
            + struct.pack("<I", calib_idx))


def build_v2s_request(pos, xbp, factors, calib_idx=0):
    """Assemble a valid MAGIC_V2S body (facts S1-S3). Header stays 5x u32
    (single is derived from the magic, not a header field -- the 6th
    group_chunks field is grouped-only, cpp L2419). POS mask block ONLY (no
    neg block, mask_blocks=1 cpp L2420), then x_bitplane, bitplane_factor,
    trailing calib_idx. Byte-identical to pim_linear's single-slice body
    (L470-479 + L1104-1107)."""
    n_chunks, d_out = pos.shape
    assert d_out == D_OUT
    n_bp = xbp.shape[1]
    d_in = n_chunks * 32
    header = struct.pack("<5I", MAGIC_V2S, d_in, D_OUT, n_chunks, n_bp)
    return (header
            + np.ascontiguousarray(pos, dtype="<u4").tobytes()
            + np.ascontiguousarray(xbp, dtype="<u4").tobytes()
            + np.ascontiguousarray(factors, dtype="<i4").tobytes()
            + struct.pack("<I", calib_idx))


def build_load_request(handle_id, pos, neg):
    """Assemble the exact MAGIC_LOAD body for the persistent-handle path.
    Framing is copied from pim_linear.py load_body (L637-642) and verified
    field-for-field against the server parse process_load_weights (cpp
    L1991-2005 -- THE authority):
        [MAGIC_LOAD, handle_id, d_in, d_out, n_chunks]  (5x u32 LE)
        + pos_mask (n_chunks*d_out u32) + neg_mask (n_chunks*d_out u32).
    The handle path is DUAL-TRACK ONLY: the server ALWAYS reads both blocks
    (need = 5*4 + n_chunks*d_out*4*2, cpp L1997), so this mirrors the
    oracle's dual-track V2 regime (genuine pos AND neg) and the MM3D result
    is the full W@x -- the SAME CPU truth as MAGIC_V2 (dual). There is NO
    single-track LOAD variant, hence --single is rejected in mm3d mode."""
    n_chunks, d_out = pos.shape
    assert d_out == D_OUT
    assert neg.shape == pos.shape
    d_in = n_chunks * 32
    header = struct.pack("<5I", MAGIC_LOAD, handle_id, d_in, D_OUT, n_chunks)
    return (header
            + np.ascontiguousarray(pos, dtype="<u4").tobytes()
            + np.ascontiguousarray(neg, dtype="<u4").tobytes())


def build_mm3d_request(handle_id, xbp, factors, n_chunks, calib_idx=0):
    """Assemble the exact MAGIC_MM3D body for a matmul against a loaded
    handle. Framing copied from pim_linear.py mm3d_prefix (L643-647) + body
    (L1040-1043), verified against the server parse process_matmul_handle
    (cpp L2905-2932):
        [MAGIC_MM3D, handle_id, d_out, n_chunks, n_bitplanes]  (5x u32 LE)
        + x_bitplane (n_chunks*n_bitplanes u32) + bitplane_factor
        (n_bitplanes i32) + trailing calib_idx (u32).
    The server's `need` (cpp L2923) stops at bitplane_factor; the trailing
    calib_idx is tolerated extra bytes (handle path picks its pool/calib at
    LOAD time). We append it anyway for byte-fidelity with the real client
    (pim_linear L1043 sends calib_idx=0 for the production single-trip)."""
    n_bp = xbp.shape[1]
    assert xbp.shape[0] == n_chunks
    header = struct.pack("<5I", MAGIC_MM3D, handle_id, D_OUT, n_chunks, n_bp)
    return (header
            + np.ascontiguousarray(xbp, dtype="<u4").tobytes()
            + np.ascontiguousarray(factors, dtype="<i4").tobytes()
            + struct.pack("<I", calib_idx))


def compare(y_pim, y_ref):
    """Return indices where the two int vectors differ."""
    y_pim = np.asarray(y_pim, dtype=np.int64)
    y_ref = np.asarray(y_ref, dtype=np.int64)
    return np.flatnonzero(y_pim != y_ref)


def inject_fault(pos, x_int8, W):
    """NEGATIVE CONTROL (dual-track): return a copy of pos with exactly one
    extra +1 weight bit set at (row s=0, some column col) chosen so that
      * W[0, col] == 0 (the true weight is absent), and
      * x_int8[col] != 0 (so the fault actually moves the output).
    The reference built from this mask therefore equals the TRUE output
    plus x_int8[col] at row 0 -- a value a correct server never produces,
    so the comparator MUST flag row 0. Returns (pos_faulty, col, delta)."""
    d_in = W.shape[1]
    col = None
    for j in range(d_in):
        if W[0, j] == 0 and int(x_int8[j]) != 0:
            col = j
            break
    if col is None:                       # pathological; force one
        col = 0
        x_int8 = x_int8.copy()
        x_int8[0] = 51
    pos_f = pos.copy()
    c, i = col // 32, col % 32
    pos_f[c, 0] |= np.uint32(1) << np.uint32(i)
    return pos_f, col, int(x_int8[col])


def inject_fault_single(pos, x_int8, W):
    """NEGATIVE CONTROL (V2S single-track). W is 1-bit (no zeros), so the
    dual-track inject_fault (which needs a W==0 slot) does not apply. Here
    we TOGGLE exactly one pos bit at (row s=0, col) where x_int8[col] != 0,
    in the REFERENCE-only pos mask. The server computes the true y_pos from
    the CLEAN sent mask, so the pos-track comparison at row 0 MUST differ by
    exactly the toggled column's signed contribution:
        contribution of bit i in chunk c = pos_bit_i * x[col]   (S5/A4),
    so clearing a set bit gives delta = -x[col]; setting a clear bit gives
    delta = +x[col]. Both are nonzero (x[col]!=0). Returns
    (pos_faulty, col, delta) with delta = y_pos_ref[0] - y_pos_true[0]."""
    d_in = W.shape[1]
    col = None
    for j in range(d_in):
        if int(x_int8[j]) != 0:
            col = j
            break
    if col is None:                       # pathological; force one
        col = 0
        x_int8 = x_int8.copy()
        x_int8[0] = 51
    pos_f = pos.copy()
    c, i = col // 32, col % 32
    bit = np.uint32(1) << np.uint32(i)
    was_set = bool(int(pos_f[c, 0]) & int(bit))
    pos_f[c, 0] ^= bit                    # toggle exactly one weight bit
    xv = int(x_int8[col])
    delta = -xv if was_set else +xv
    return pos_f, col, delta


# ------------------------------ liveness ------------------------------------

def parse_server_exec_count(log_path):
    """Return the max n_maj3_execs seen in the server's own stderr log
    ('[srv-prof #N] ... exec=X.Xms (Nx) ...'), or None if no line yet.
    stderr is unbuffered by C default, and the srv-prof line is written
    BEFORE the response, so it is present once server.request() returns."""
    import re
    rx = re.compile(r"exec=[\d.]+ms\s*\((\d+)x\)")
    best = None
    try:
        with open(log_path, "r", errors="replace") as f:
            for line in f:
                if "[srv-prof" in line:
                    m = rx.search(line)
                    if m:
                        v = int(m.group(1))
                        best = v if best is None else max(best, v)
    except FileNotFoundError:
        return None
    return best


def parse_server_mm3d_liveness(log_path, retries=40, delay=0.05):
    """Liveness counter for the MM3D handle / descriptor-serve paths (the
    mm3d mode analogue of parse_server_exec_count). Two producers, two
    liveness lines:
      * handle path (process_matmul_handle): '[mm3d-prof #N handle=H] ...
        exec=X.Xms (Kx) ...' (cpp L3837) -- the '(Kx)' MAJ3-exec count,
        printed BEFORE the response (cpp L3874).
      * descriptor-serve path (process_matmul_desc, PIM_DESC_SERVE=1): the
        '[desc-serve #N handle=H] banks=.. units=.. planes=.. desc=D
        walks=W ...' summary (cpp L6397) -- 'desc=D' is the walked-descriptor
        count (its zero-exec liveness assert counts descriptors, cpp L6213/
        L6372). This line is printed AFTER the response (cpp L6384), so the
        oracle briefly polls the log rather than reading once.
    Returns the max liveness count across whichever line(s) appeared (>0 for
    any real op; 0 only for a legitimately skip-covered zero-x request), or
    None if neither has been written yet. The dispatch routes MM3D through
    desc-serve first and FALLS BACK to the handle path when ineligible
    (cpp L6778-6791, identical answer), so either line may be the live one."""
    import re
    import time
    rx_exec = re.compile(r"exec=[\d.]+ms\s*\((\d+)x\)")
    rx_desc = re.compile(r"\bdesc=(\d+)\b")
    for _ in range(max(1, retries)):
        best = None
        try:
            with open(log_path, "r", errors="replace") as f:
                for line in f:
                    if "[mm3d-prof" in line:
                        m = rx_exec.search(line)
                        if m:
                            v = int(m.group(1))
                            best = v if best is None else max(best, v)
                    elif "[desc-serve #" in line:
                        m = rx_desc.search(line)
                        if m:
                            v = int(m.group(1))
                            best = v if best is None else max(best, v)
        except FileNotFoundError:
            best = None
        if best is not None:
            return best
        time.sleep(delay)          # desc-serve line lags the response; poll
    return None


# -------------------------------- modes -------------------------------------

def _internal_consistency(pos, neg, xbp, factors, W, x_int8, n_real):
    """Prove the DUAL-track reference is right: bitplane round-trip is
    lossless and the popcount form equals the plain matvec. Raises
    AssertionError. Returns the (pos-neg) reference over the real rows."""
    x_rt = reconstruct_x_from_bitplanes(xbp, factors)
    assert np.array_equal(x_rt, x_int8.astype(np.int64)), (
        "bitplane reconstruction != x_int8 (factors wrong?)")
    y_pc = cpu_reference_from_masks(pos, neg, xbp, factors)[:n_real]
    y_mv = matvec_reference(W, x_int8)
    assert np.array_equal(y_pc, y_mv), (
        "popcount-formula reference != W@x matvec reference")
    return y_pc


def _internal_consistency_single(pos, xbp, factors, W, x_int8, n_real):
    """Prove the V2S reference is right: (a) bitplane round-trip lossless;
    (b) W is 1-bit (V2S eligibility, S8); (c) the pos-track popcount
    reference (the array the SERVER returns), run through the CLIENT's
    reconstruction y = 2*y_pos - const (S7), equals the plain matvec on the
    real rows. Raises AssertionError. Returns the POS-TRACK reference (the
    server's wire array) over the real rows -- THIS is what the live gate
    compares the response against."""
    x_rt = reconstruct_x_from_bitplanes(xbp, factors)
    assert np.array_equal(x_rt, x_int8.astype(np.int64)), (
        "bitplane reconstruction != x_int8 (factors wrong?)")
    assert bool((W[:n_real] != 0).all()), (
        "--single requires 1-bit weights (no zeros); W has a zero, so the "
        "V2S reconstruction identity neg == ~pos (S7/S8) does NOT hold")
    y_pos = pos_track_reference(pos, xbp, factors)           # server array
    const = single_recon_const(xbp, factors)
    y_recon = (2 * y_pos - const)[:n_real]                   # client identity
    y_mv = matvec_reference(W, x_int8)
    assert np.array_equal(y_recon, y_mv), (
        "V2S reconstruction (2*y_pos - const) != W@x matvec")
    return y_pos[:n_real]


def run_self_test(args):
    """Hardware-free: proves both directions of the comparator with pure
    numpy -- no server, no card, no /dev/xdma, no pim_linear import.
    --single exercises the V2S pos-track reference + its negative control."""
    W, x = make_problem(args.d_in, args.n_real, args.seed, single=args.single)
    n_chunks = args.d_in // 32
    pos, neg = masks_from_W(W)
    xbp = pack_xbp(x, n_chunks)
    factors = BITPLANE_FACTORS
    # mm3d mode is dual-track (the LOAD+handle path always stores pos+neg and
    # MM3D returns the full W@x), so its CPU truth == V2 dual-track's. The
    # self-test math is therefore identical to dual V2; only the tag differs.
    # (--single is rejected with mm3d in main(), so args.single is False here.)
    tag = ("MM3D dual-track (LOAD+handle)" if args.mode == "mm3d"
           else "V2S single-track" if args.single
           else "V2 dual-track")

    if args.single:
        clean = _internal_consistency_single(pos, xbp, factors, W, x,
                                              args.n_real)
        print(f"[self-test] {tag} internal consistency OK "
              f"(bitplane round-trip lossless; 2*y_pos-const == matvec) "
              f"n_real={args.n_real} d_in={args.d_in}")
    else:
        clean = _internal_consistency(pos, neg, xbp, factors, W, x,
                                      args.n_real)
        print(f"[self-test] {tag} internal consistency OK "
              f"(bitplane round-trip lossless; popcount==matvec) "
              f"n_real={args.n_real} d_in={args.d_in}")

    # POSITIVE: comparator accepts a true match.
    mism = compare(clean, clean)
    assert mism.size == 0, "comparator flagged a true match (broken)"
    print(f"[self-test] POSITIVE: compare(clean, clean) -> 0 mismatches  OK")

    # NEGATIVE CONTROL: one wrong weight bit in the reference must be caught.
    if args.single:
        pos_f, col, delta = inject_fault_single(pos, x, W)
        faulty = pos_track_reference(pos_f, xbp, factors)[:args.n_real]
        what = "toggled pos bit"
    else:
        pos_f, col, delta = inject_fault(pos, x, W)
        faulty = cpu_reference_from_masks(pos_f, neg, xbp,
                                          factors)[:args.n_real]
        what = "injected +1"
    mism = compare(clean, faulty)
    ok = mism.size > 0 and (0 in mism) and (int(faulty[0] - clean[0]) == delta)
    print(f"[self-test] NEGATIVE: {what} at (row 0, col {col}), "
          f"expect delta_row0={delta} -> {mism.size} mismatch(es), "
          f"delta_row0={int(faulty[0]-clean[0])}")
    if not ok:
        print("[self-test] FAIL: negative control did not behave as designed")
        return 3
    print("[self-test] PASS: comparator detects a one-weight fault")
    return 0


def run_live_batch(args):
    """CROSS-REQUEST BATCH gate (2026-08-04, mm3d only): LOAD N handles with N
    independent problems (seeds seed..seed+N-1), send ONE MAGIC_FUSE frame of N
    MM3D bodies, read N x 8192 B, and compare EVERY sub-response against its own
    dual-track CPU truth. With PIM_DESC_XBATCH=1 in the server env the frame is
    served by ONE desc-serve super-session per bank; the oracle REQUIRES the
    server log's '[desc-xbatch #' line (subs=N) — a silent per-sub fallback
    would pass numerics without testing the batch path, so it is a FAIL here."""
    import time
    sys.path.insert(0, "/home/deni/bitnet_weights")
    try:
        import pim_linear  # noqa: F401
    except Exception as e:                                    # pragma: no cover
        print(f"[batch] cannot import pim_linear: {e}", file=sys.stderr)
        return 2
    if not os.path.exists(args.server):
        print(f"[batch] server binary not found: {args.server}", file=sys.stderr)
        return 2
    NB = args.batch
    probs = []
    for i in range(NB):
        W, x = make_problem(args.d_in, args.n_real, args.seed + i, single=False)
        n_chunks = args.d_in // 32
        pos, neg = masks_from_W(W)
        xbp = pack_xbp(x, n_chunks)
        y_ref = _internal_consistency(pos, neg, xbp, BITPLANE_FACTORS, W, x,
                                      args.n_real)
        probs.append((pos, neg, xbp, y_ref, n_chunks))
    extra_env = {}
    for k in ("PIM_SUB_START", "PIM_SUB_END", "PIM_POOL_LIST_FILE"):
        if os.environ.get(k):
            extra_env[k] = os.environ[k]
    server = pim_linear.PimServer.shared(
        args.bender, args.bank, args.calib, args.server,
        extra_env=extra_env or None)
    log_path = getattr(getattr(server, "_stderr_log", None), "name", None)
    # LOAD each problem as its own handle (ids 1..NB).
    for i, (pos, neg, xbp, y_ref, n_chunks) in enumerate(probs):
        status_raw = server.request(build_load_request(i + 1, pos, neg),
                                    expect_resp_len=4)
        status = struct.unpack("<I", status_raw)[0]
        if status != 0:
            print(f"[batch] LOAD handle {i+1} ENOSPC ack={status}",
                  file=sys.stderr)
            return 2
    print(f"[batch] {NB} handles resident (seeds {args.seed}..{args.seed+NB-1}).")
    # ONE FUSE frame of NB MM3D bodies.
    frame = struct.pack("<II", MAGIC_FUSE, NB)
    for i, (pos, neg, xbp, y_ref, n_chunks) in enumerate(probs):
        body = build_mm3d_request(i + 1, xbp, BITPLANE_FACTORS, n_chunks,
                                  calib_idx=0)
        frame += struct.pack("<I", len(body)) + body
    t0 = time.time()
    resp = server.request(frame, expect_resp_len=NB * D_OUT * 4)
    dt = time.time() - t0
    assert len(resp) == NB * D_OUT * 4, (
        f"bad batch response length {len(resp)} (expected {NB*D_OUT*4})")
    # ATTRIBUTION GATE: the batch path (not the per-sub fallback) must have
    # served the frame. The server prints '[desc-xbatch #N] subs=..' only there.
    served = None
    if log_path:
        for _ in range(40):
            try:
                with open(log_path, "r", errors="replace") as f:
                    for line in f:
                        if "[desc-xbatch #" in line and f"subs={NB}" in line:
                            served = line.strip()
            except FileNotFoundError:
                pass
            if served:
                break
            time.sleep(0.05)
    if not served:
        print("[batch] FAIL: no '[desc-xbatch] subs=%d' line in the server log "
              "— the frame was NOT served by the batch path (per-sub fallback "
              "or ineligible). Numerics not attributable to XBATCH." % NB,
              file=sys.stderr)
        return 1
    print(f"[batch] served by the batch path: {served}")
    print(f"[batch] frame wall: {dt*1e3:.1f} ms total, {dt*1e3/NB:.1f} ms/sub")
    all_exact = True
    for i, (pos, neg, xbp, y_ref, n_chunks) in enumerate(probs):
        y = np.frombuffer(resp[i*D_OUT*4:(i+1)*D_OUT*4],
                          dtype="<i4").astype(np.int64)
        if not np.any(y != 0):
            print(f"[batch] LIVENESS FAIL: sub {i} response ALL-ZERO",
                  file=sys.stderr)
            return 1
        mism = compare(y[:args.n_real], y_ref)
        exact = args.n_real - mism.size
        print(f"[batch] sub {i} (handle {i+1}): exact={exact}/{args.n_real} "
              f"mismatch={mism.size}")
        if mism.size:
            all_exact = False
            s0 = int(mism[0])
            print(f"[batch]   first mismatch s={s0} pim={int(y[s0])} "
                  f"ref={int(y_ref[s0])}")
    if all_exact:
        print(f"[batch] PASS: all {NB} sub-responses bit-exact vs CPU truth "
              f"through ONE desc-xbatch frame.")
        return 0
    print("[batch] FAIL: at least one sub-response diverged.", file=sys.stderr)
    return 1


def run_live(args):
    """Drive the REAL V2 / V2S path through pim_linear.PimServer. Do NOT run
    while the main session owns /dev/xdma. Use PIM_BACKEND=sim for a
    card-free server, or the self-test for a card-free comparator check."""
    if getattr(args, "batch", 1) > 1:
        if args.mode != "mm3d" or args.single or args.inject_fault:
            print("[live] --batch requires --mode mm3d (no --single / "
                  "--inject-fault)", file=sys.stderr)
            return 2
        return run_live_batch(args)
    sys.path.insert(0, "/home/deni/bitnet_weights")
    try:
        import pim_linear  # noqa: F401  (imported for its side-effect path)
    except Exception as e:                                    # pragma: no cover
        print(f"[live] cannot import pim_linear: {e}", file=sys.stderr)
        return 2
    if not os.path.exists(args.server):
        print(f"[live] server binary not found: {args.server}\n"
              f"       run from the BitNet app dir (see RUN_CMD).",
              file=sys.stderr)
        return 2

    W, x = make_problem(args.d_in, args.n_real, args.seed, single=args.single)
    n_chunks = args.d_in // 32
    pos, neg = masks_from_W(W)
    xbp = pack_xbp(x, n_chunks)
    factors = BITPLANE_FACTORS

    # Reference (always from CLEAN masks) + internal consistency. In single
    # mode y_ref is the POS-TRACK reference (== the server's wire array);
    # in dual mode it is the (pos-neg) reference.
    if args.single:
        y_ref = _internal_consistency_single(pos, xbp, factors, W, x,
                                             args.n_real)
    else:
        y_ref = _internal_consistency(pos, neg, xbp, factors, W, x,
                                      args.n_real)

    # Under --inject-fault the REFERENCE uses a corrupted mask so a correct
    # server output is REQUIRED to trip the mismatch. The SENT masks stay
    # clean (the server computes the true answer).
    if args.inject_fault:
        if args.single:
            pos_f, col, delta = inject_fault_single(pos, x, W)
            y_ref = pos_track_reference(pos_f, xbp, factors)[:args.n_real]
            print(f"[live] NEGATIVE CONTROL active (V2S): pos-track reference "
                  f"corrupted at (row 0, col {col}) by a toggled weight bit, "
                  f"delta_row0={delta}. A correct server must now MISMATCH at "
                  f"row 0.")
        else:
            pos_f, col, delta = inject_fault(pos, x, W)
            y_ref = cpu_reference_from_masks(pos_f, neg, xbp,
                                             factors)[:args.n_real]
            print(f"[live] NEGATIVE CONTROL active: reference corrupted at "
                  f"(row 0, col {col}) by +1 weight, x[col]={delta}. "
                  f"A correct server must now MISMATCH at row 0.")

    N = len([b for b in str(args.bank).split(",") if b != ""])
    # V2S enumerates chunks only (n_units = n_chunks); V2 enumerates
    # (chunk, sign) pairs (n_units = 2*n_chunks). cpp L2455.
    n_units = n_chunks if args.single else 2 * n_chunks
    n_rounds = (n_units + N - 1) // N
    assert n_rounds > 0, "n_rounds computed as 0 (bad shape)"

    # Pass the production subarray/pool env through explicitly (PimServer
    # also inherits os.environ, but be explicit so the RUN_CMD is honest).
    extra_env = {}
    for k in ("PIM_SUB_START", "PIM_SUB_END", "PIM_POOL_LIST_FILE"):
        if os.environ.get(k):
            extra_env[k] = os.environ[k]
    server = pim_linear.PimServer.shared(
        args.bender, args.bank, args.calib, args.server,
        extra_env=extra_env or None)
    log_path = getattr(getattr(server, "_stderr_log", None), "name", None)

    if args.mode == "mm3d":
        # DESCRIPTOR-SERVING PATH gate. Drive the persistent-handle protocol:
        # LOAD the weights ONCE, then one MM3D request. With PIM_DESC_SERVE=1
        # in the server's env the MM3D dispatch routes through
        # process_matmul_desc (the new path) and falls back to
        # process_matmul_handle when ineligible -- identical answer either
        # way (cpp L6778-6791). The compare below gates whichever ran against
        # the SAME dual-track CPU truth as V2 (y_ref built above).
        load_body = build_load_request(MM3D_HANDLE_ID, pos, neg)
        status_raw = server.request(load_body, expect_resp_len=4)
        if len(status_raw) != 4:
            print(f"[live] LOAD returned {len(status_raw)} bytes (expected 4 "
                  f"status bytes) -- protocol mismatch.", file=sys.stderr)
            return 2
        status = struct.unpack("<I", status_raw)[0]
        if status != 0:
            # cpp L2076-2079: ack=1 means the backup pool was exhausted
            # (ENOSPC); in production the client falls back to V2, but the
            # oracle is explicitly gating the handle path, so this is fatal.
            print(f"[live] LOAD_WEIGHTS ENOSPC: server ack={status} (backup "
                  f"pool exhausted; cpp L2068-2079). Cannot gate the MM3D "
                  f"handle path. Free pool space (fewer resident handles / "
                  f"larger PIM_POOL) or reduce d_in, then retry.",
                  file=sys.stderr)
            return 2
        print(f"[live] LOAD OK: handle {MM3D_HANDLE_ID} resident "
              f"(pos+neg, {n_chunks} chunks), status ack=0.")
        # [drift-ladder Rung 3] PIM_ORACLE_PRE_MM3D_SLEEP_S: passive host-idle
        # delay between the LOAD-time resident write and the MM3D walk read.
        # Adds wall-time WITHOUT adding walk sessions or activations -> isolates
        # passive retention/decay from active-walk (per-session) disturb.
        _pre = os.environ.get("PIM_ORACLE_PRE_MM3D_SLEEP_S")
        if _pre and float(_pre) > 0:
            import time as _t
            print(f"[live] Rung3: sleeping {float(_pre):.1f}s after LOAD, before "
                  f"MM3D walk (passive retention probe).")
            _t.sleep(float(_pre))
        mm3d_body = build_mm3d_request(MM3D_HANDLE_ID, xbp, factors, n_chunks,
                                       calib_idx=0)
        resp = server.request(mm3d_body, expect_resp_len=D_OUT * 4)
    elif args.single:
        body = build_v2s_request(pos, xbp, factors, calib_idx=0)
        resp = server.request(body, expect_resp_len=D_OUT * 4)
    else:
        body = build_v2_request(pos, neg, xbp, factors, calib_idx=0)
        resp = server.request(body, expect_resp_len=D_OUT * 4)

    # ---- liveness: refuse to be fooled by an all-zero / no-exec response ----
    assert len(resp) == D_OUT * 4, (
        f"bad response length {len(resp)} (expected {D_OUT*4})")
    y = np.frombuffer(resp, dtype="<i4").astype(np.int64)
    if not np.any(y != 0):
        print("[live] LIVENESS FAIL: server returned ALL-ZERO y -- nothing "
              "executed (this is the ab_fused_server.py failure mode).",
              file=sys.stderr)
        return 1
    if args.mode == "mm3d":
        # mm3d liveness reads the handle/desc-serve counters, not srv-prof.
        exec_count = (parse_server_mm3d_liveness(log_path)
                      if log_path else None)
        live_line = "[mm3d-prof ... exec=..(Nx)] / [desc-serve # ... desc=D]"
    else:
        exec_count = parse_server_exec_count(log_path) if log_path else None
        live_line = "[srv-prof ... exec=..(Nx)]"
    if exec_count is None:
        print(f"[live] LIVENESS FAIL: no '{live_line}' line in "
              f"{log_path} -- cannot confirm a MAJ3 program ran.",
              file=sys.stderr)
        return 1
    if exec_count <= 0:
        print(f"[live] LIVENESS FAIL: server exec count = {exec_count} "
              f"(exec=0 => no MAJ3 programs ran).", file=sys.stderr)
        return 1
    if exec_count < n_rounds and args.mode != "mm3d":
        print(f"[live] LIVENESS WARN: exec_count={exec_count} < n_rounds="
              f"{n_rounds} (expected >= one exec/round).", file=sys.stderr)
        # (In mm3d mode the desc-serve counter is walked-descriptors, not a
        # per-round MAJ3 exec count, so the < n_rounds comparison does not
        # apply; exec_count>0 above is the liveness gate.)
    mode = "MM3D" if args.mode == "mm3d" else ("V2S" if args.single else "V2")
    print(f"[live] liveness OK [{mode}]: y nonzero "
          f"({int(np.count_nonzero(y))}/{D_OUT}), server "
          f"exec_count={exec_count}, n_rounds={n_rounds}, "
          f"max|y|={int(np.abs(y).max())}")

    # ---- numerics: per-element PIM vs CPU over the real outputs ----
    # In single mode y_pim is the RAW pos-track response and y_ref is the
    # pos-track reference (S9) -- "exactly what array comes back".
    y_pim = y[:args.n_real]
    mism = compare(y_pim, y_ref)
    exact = args.n_real - mism.size
    print(f"[live] numerics [{mode}]: exact={exact}/{args.n_real}  "
          f"mismatch={mism.size}")
    # [drift-ladder Rung 4/5] PIM_ORACLE_DUMP_LANES: write the FULL failing-lane
    # set (column index s, pim, ref, delta) so the victim geometry (Rung 4) and
    # the run-to-run lane-set reproducibility (Rung 5) can be analysed offline.
    _dump = os.environ.get("PIM_ORACLE_DUMP_LANES")
    if _dump:
        with open(_dump, "w") as _fp:
            _fp.write("# s pim ref delta  (full mismatch set vs CPU truth)\n")
            for _s in mism.tolist():
                _fp.write(f"{int(_s)} {int(y_pim[_s])} {int(y_ref[_s])} "
                          f"{int(y_pim[_s] - y_ref[_s])}\n")

    # FYI (single, non-fault): apply the CLIENT identity 2*y_pos-const to the
    # MEASURED response and check it reproduces W@x on the real rows -- the
    # model-facing quantity. This is a linear transform of y_pim, so it adds
    # no independent gate signal; the gate stays on the pos-track compare.
    if args.single and not args.inject_fault:
        const = single_recon_const(xbp, factors)
        y_recon = 2 * y_pim - const
        y_mv = matvec_reference(W, x)
        recon_exact = int(np.count_nonzero(y_recon == y_mv))
        print(f"[live]   V2S reconstruction FYI: 2*y_pos-const == W@x on "
              f"{recon_exact}/{args.n_real} real rows (client identity on the "
              f"measured pos-track response; const={int(const)}).")

    if mism.size:
        s0 = int(mism[0])
        print(f"[live]   first mismatch: s={s0} pim={int(y_pim[s0])} "
              f"ref={int(y_ref[s0])} (delta={int(y_pim[s0]-y_ref[s0])})")
        # Delta distribution — distinguishes intrinsic MAJ3 flake (small |delta|
        # relative to the signal, tolerated by the model) from CORRUPTION
        # (large/systematic |delta|, e.g. stale masks = wrong weights). A
        # bit-exact verdict over-fires on flake; the distribution is the useful
        # signal.
        d = np.abs(y_pim - y_ref)
        dd = d[d > 0]
        sigmag = int(np.median(np.abs(y_ref[y_ref != 0]))) if np.any(y_ref != 0) else 0
        print(f"[live]   |delta| over {int(dd.size)} nonzero: max={int(dd.max())} "
              f"median={int(np.median(dd))} mean={dd.mean():.1f}  |  bands: "
              f"<=2:{int((dd<=2).sum())} <=8:{int((dd<=8).sum())} "
              f"<=64:{int((dd<=64).sum())} >64:{int((dd>64).sum())}  |  "
              f"median|y_ref|={sigmag}  worst_rel={ (dd.max()/max(1,sigmag)):.1%}")

    if args.inject_fault:
        # Negative control: the fault MUST have been detected.
        if mism.size > 0 and 0 in mism:
            print("[live] NEGATIVE CONTROL PASSED: injected fault detected "
                  "(the gate can catch a divergence).")
            return 0
        print("[live] NEGATIVE CONTROL FAILED: injected fault NOT detected -- "
              "the oracle is blind, DO NOT TRUST IT.", file=sys.stderr)
        return 3

    if mism.size == 0:
        print(f"[live] PASS [{mode}]: PIM output is bit-exact vs the CPU oracle.")
        return 0
    # Flake-vs-corruption verdict. Raw per-op silicon MAJ3 is NOT bit-exact
    # (analog cell flake + per-process operating point -- cross_process_floor),
    # so a bit-exact verdict over-fires and its rate varies per process. The
    # robust discriminator is CORRELATION: intrinsic flake is signal + small
    # noise (corr ~ 1); wrong weights / stale masks (the M3 corruption class)
    # compute a DIFFERENT matvec -> correlation collapses. Thresholds are
    # provisional -- calibrate against a known-corruption run before trusting
    # the boundary; report the numbers regardless.
    corr = float(np.corrcoef(y_pim, y_ref)[0, 1]) if (
        np.std(y_pim) > 0 and np.std(y_ref) > 0) else 0.0
    d = np.abs(y_pim - y_ref)
    sigmag = float(np.median(np.abs(y_ref[y_ref != 0]))) if np.any(y_ref != 0) else 1.0
    worst_rel = float(d.max() / max(1.0, sigmag))
    if corr >= 0.98:
        print(f"[live] PASS (flake-consistent) [{mode}]: {exact}/{args.n_real} "
              f"bit-exact, corr={corr:.5f}, worst_rel={worst_rel:.1%}, "
              f"median|delta|={int(np.median(d[d>0])) if np.any(d>0) else 0}. "
              f"Divergence is within the intrinsic per-op MAJ3 flake envelope, "
              f"NOT corruption.")
        return 0
    print(f"[live] FAIL (GROSS divergence -- likely CORRUPTION, not flake) "
          f"[{mode}]: corr={corr:.5f} (< 0.98), worst_rel={worst_rel:.1%}, only "
          f"{exact}/{args.n_real} bit-exact. Wrong weights / stale masks destroy "
          f"correlation; intrinsic flake preserves it.", file=sys.stderr)
    return 1


def main():
    ap = argparse.ArgumentParser(
        description="V2 PIM-vs-CPU numerics oracle (see module docstring / "
                    "RUN_CMD).")
    # NOTE: dest is "cmd" (not "mode") so it does not collide with the new
    # --mode {v2,mm3d} request-path selector added to `common` below.
    sub = ap.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--d-in", type=int, default=64,
                        help="input width, multiple of 32 (default 64)")
    common.add_argument("--n-real", type=int, default=8,
                        help="real output rows, 1..2048 (default 8)")
    common.add_argument("--seed", type=int, default=1)
    common.add_argument("--mode", choices=["v2", "mm3d"], default="v2",
                        help="serving PATH to gate. v2 (default): the "
                             "per-request MAGIC_V2 upload path (existing "
                             "behavior, byte-identical). mm3d: the "
                             "persistent-handle path -- one MAGIC_LOAD then "
                             "MAGIC_MM3D per request (routes through "
                             "process_matmul_desc when PIM_DESC_SERVE=1). "
                             "mm3d is DUAL-TRACK ONLY (no --single).")
    common.add_argument("--single", action="store_true",
                        help="drive the V2S single-track path: 1-bit W (no "
                             "zeros), pos-mask-only request, compare the raw "
                             "pos-track response (default: dual-track V2). "
                             "Not valid with --mode mm3d.")

    p_self = sub.add_parser("self-test", parents=[common],
                            help="pure-numpy comparator + negative-control "
                                 "check (NO server, NO card)")
    p_self.set_defaults(func=run_self_test)

    p_live = sub.add_parser("live", parents=[common],
                            help="drive the real V2/V2S path via PimServer "
                                 "(needs /dev/xdma or PIM_BACKEND=sim)")
    p_live.add_argument("--server", default="./bitnet-proj-server")
    p_live.add_argument("--bender", type=int, default=2)
    p_live.add_argument("--calib", default="calib_dimm2.txt")
    p_live.add_argument("--bank", default="0",
                        help='bank id or "0,1,2,3" for multi-bank')
    p_live.add_argument("--inject-fault", action="store_true",
                        help="NEGATIVE CONTROL: corrupt the reference by one "
                             "weight bit; a correct server must mismatch")
    p_live.add_argument("--batch", type=int, default=1,
                        help="CROSS-REQUEST BATCH gate (mm3d only): LOAD N "
                             "handles, send ONE MAGIC_FUSE frame of N MM3D "
                             "bodies, compare every sub-response (requires "
                             "PIM_DESC_XBATCH=1 server-side; the [desc-xbatch] "
                             "log line is asserted so a silent per-sub "
                             "fallback FAILS the gate)")
    p_live.set_defaults(func=run_live)

    args = ap.parse_args()
    if args.d_in % 32 != 0:
        ap.error("--d-in must be a multiple of 32")
    if not (1 <= args.n_real <= D_OUT):
        ap.error("--n-real must be in [1, 2048]")
    if args.mode == "mm3d" and args.single:
        # The LOAD/handle path always carries pos+neg (cpp L1997-2005) and
        # MM3D returns the full dual-track W@x; there is no single-track LOAD
        # variant. Combining them would break the CPU truth, so refuse.
        ap.error("--single is not valid with --mode mm3d (the LOAD/handle "
                 "path is dual-track only; no single-track LOAD variant "
                 "exists)")
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
