"""PIM-backed BitLinear: drop-in for transformers.AutoBitLinear that runs
the integer matmul on real DRAM-Bender silicon. Two backends:

(a) subprocess-per-call via `bitnet-proj-exe` (legacy; ~150 ms overhead/call).
(b) **long-running server via `bitnet-proj-server`** — opens FPGA once,
    streams requests over stdin/stdout. ~50 ms saved per call. Default.

Use PimServer.shared(bender, bank, ...) to get/create one server per
(bender, bank) pair; PimBitLinear instances share it.

Workflow per forward(input):
  1. Apply BitNet's symmetric int8 activation_quant to `input`.
  2. Pre-decompose to per-chunk x_bitplane (uint32, 8 bitplanes).
  3. For each output slice (2048 outputs each, last padded with zero
     weights): build a v2 request (pos_mask + neg_mask + x_bitplane +
     bitplane_factor), send to server, receive int32 y.
  4. Concatenate slices; rescale by weight_scale / input_scale.

2026-07-20 extension (default-off): PimBitLinear(weight_spec=...) accepts
external int8 codes with OPTIONAL per-(row, 128-input-group) scales and
sparse exact residuals (PrismML Bonsai g128 format). Group mode dispatches
per-group partial requests (vote-then-scale) — see __init__. With
weight_spec=None everything is the legacy BitNet path, byte-identical.
"""
import atexit, os, struct, subprocess, sys, tempfile, threading
import numpy as np
import torch
import torch.nn as nn

D_OUT_SLICE = 2048
MAX_CHUNKS_PER_SUB = int(os.environ.get('PIM_MAX_CHUNKS_PER_SUB', '16'))

# x_bp pack cache.  q, k, v projections share the same input x within one
# token; o uses different x but same shape.  Caching x_bp by (id(x_int8),
# d_in) lets us pack once per unique x and reuse across all linears that
# get called with that tensor (typically 3-4× per layer in BitNet).
# Per-token win is ~1 ms × #linears-sharing × #layers.  Bounded LRU so
# we don't leak memory across tokens.  Disabled with PIM_XBP_CACHE=0.
_XBP_CACHE = {}
_XBP_CACHE_ORDER = []
_XBP_CACHE_MAX = 8

def _pack_xbp(x_int8, n_chunks):
    cache_on = os.environ.get('PIM_XBP_CACHE', '1') == '1'
    if cache_on:
        # Key by CONTENT, not id().  The cache does not retain x_int8, so
        # a freed array's address gets recycled by the allocator while its
        # id() lives on as a key — in batched prefill the per-position x
        # arrays ping-pong through the same addresses and later positions
        # hit earlier positions' stale bitplanes (2026-07-17: deterministic
        # ' The capital of of capital...' derail at the first token that
        # needs real attention retrieval; PIM_INT_DIFF showed q/k/v/o
        # decorrelated from position 0 while down/gate tails were normal).
        # tobytes() is a ~2.5 KB copy, ~µs vs the ~1 ms pack it saves;
        # dict equality on bytes keys makes hits exact, not just probable.
        key = (x_int8.tobytes(), n_chunks)
        cached = _XBP_CACHE.get(key)
        if cached is not None:
            return cached
    x_u8 = x_int8.astype(np.uint8)
    x_bitplane = np.zeros((n_chunks, N_BITPLANES), dtype=np.uint32)
    powers = (np.uint32(1) << np.arange(32, dtype=np.uint32))
    for c in range(n_chunks):
        chunk = x_u8[c*32:(c+1)*32]
        for b in range(N_BITPLANES):
            bits = ((chunk >> b) & 1).astype(np.uint32)
            x_bitplane[c, b] = (bits * powers).sum(dtype=np.uint32)
    out = (x_bitplane, x_bitplane.tobytes())
    if cache_on:
        _XBP_CACHE[key] = out
        _XBP_CACHE_ORDER.append(key)
        while len(_XBP_CACHE_ORDER) > _XBP_CACHE_MAX:
            old = _XBP_CACHE_ORDER.pop(0)
            _XBP_CACHE.pop(old, None)
    return out
N_BITPLANES = 8
BITPLANE_FACTORS = np.array([1, 2, 4, 8, 16, 32, 64, -128], dtype=np.int32)
MAGIC_V2 = 0xB17EF002
MAGIC_LOAD = 0xB17EF003   # LOAD_WEIGHTS (one-time per linear's slice)
MAGIC_MM3D = 0xB17EF004   # MATMUL_HANDLE (subsequent calls)
MAGIC_V2G = 0xB17EF005    # V2 with per-group partial response (PIM_GROUP_RESP=1)
MAGIC_V2S = 0xB17EF006    # V2 single-track (PIM_1BIT_SINGLE=1): pos masks only


def _xbp_weighted_popcount(x_bp_sub):
    """sum_b fac_b * popcount(x bitplane b) over a chunk range — the
    single-track reconstruction constant. For 1-bit weights (neg == ~pos
    on every real/replicated output slot) the dual-track dot decomposes as
    y = pc_pos - pc_neg = 2*pc_pos - popcount(x), bitplane-weighted, so
    the server only computes the pos track (half the DRAM work) and the
    host applies y = 2*y_resp - this. Slots whose masks are all-zero
    padding get -this instead of 0 — every consumer trims to n_real (or
    n_copies*n_real) before use, so the garbage never escapes."""
    s = 0
    for b in range(N_BITPLANES):
        col = np.ascontiguousarray(x_bp_sub[:, b]).view(np.uint8)
        s += int(BITPLANE_FACTORS[b]) * int(np.unpackbits(col).sum())
    return s
_NEXT_HANDLE_ID = [0]     # global handle-id counter
_V2_FALLBACK_COUNT = [0]  # slices that fell back LOAD→V2 (diagnostics)


def _bf16_to_f32_scalar(t):
    return float(t.float().item())


# --------------------- long-running server backend ---------------------

class PimServer:
    """One long-running `bitnet-proj-server` subprocess per (bender, bank).
    Opens the FPGA platform once at startup, then accepts requests over
    stdin/stdout (binary length-prefixed protocol)."""

    _shared = {}    # (bender, bank, calib_file, server_path) -> PimServer
    _lock = threading.Lock()

    def __init__(self, bender_id, bank_id, calib_file, server_path, extra_env=None):
        self.bender_id = bender_id
        self.bank_id = bank_id
        # bank_id can be an int (single bank) or a string like "0,2,3" for
        # multi-bank.  Multi-bank skips bank 1 by default on DIMM 0 because
        # bank 1's calibration uses a different open_row tuple than banks
        # 0/2/3, and mixing calibs in one matmul currently breaks
        # persistent-weight LOAD path (see simra_xor8_spread.md).
        bank_str = str(bank_id)
        # Sanitize for filename (include bender so multi-DIMM logs don't collide)
        bank_safe = bank_str.replace(',', '_')
        self._stderr_log = open(f"/tmp/pim_server_b{bender_id}_{bank_safe}.log", "w")
        # Per-subprocess env: each DIMM's server may need its own
        # PIM_SUB_START / PIM_SUB_END (the subarray range differs per DIMM —
        # DIMM 0's s_id 77 is 640-aligned, DIMM 2's s_id 72 is not).
        proc_env = dict(os.environ)
        if extra_env:
            proc_env.update({k: str(v) for k, v in extra_env.items()})
        self.proc = subprocess.Popen(
            [server_path, str(bender_id), calib_file, bank_str],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self._stderr_log,
            bufsize=0,
            env=proc_env,
        )
        self.lock = threading.Lock()
        self._n_calls = 0
        self._t_write_ms = 0.0
        self._t_read_ms = 0.0
        self._n_bytes_sent = 0
        atexit.register(self._cleanup)

    @staticmethod
    def _write_all(f, data):
        """Loop until all bytes are written. With bufsize=0, the underlying
        os.write may return fewer bytes than requested for big writes to
        pipes — Python's RawIO doesn't loop for us."""
        view = memoryview(data)
        total = len(view)
        sent = 0
        while sent < total:
            n = f.write(view[sent:])
            if n is None:        # would-block on non-blocking fd
                raise RuntimeError("PIM server pipe would-block")
            if n == 0:
                raise RuntimeError("PIM server pipe wrote 0 bytes")
            sent += n

    def request(self, body_bytes, expect_resp_len=8192):
        """Send one request, return `expect_resp_len` bytes of response.
        For matmul (v2 / MATMUL_HANDLE): 8192 bytes (default).
        For LOAD_WEIGHTS: 4 bytes (status code)."""
        import time as _time
        with self.lock:
            try:
                # u32 length prefix + body, both with looped writes.
                _ts = _time.perf_counter()
                self._write_all(self.proc.stdin, struct.pack('<I', len(body_bytes)))
                self._write_all(self.proc.stdin, body_bytes)
                self.proc.stdin.flush()
                self._t_write_ms += (_time.perf_counter() - _ts) * 1000
                # Read exactly `expect_resp_len` bytes back.
                _ts = _time.perf_counter()
                got = b""
                while len(got) < expect_resp_len:
                    chunk = self.proc.stdout.read(expect_resp_len - len(got))
                    if not chunk:
                        raise RuntimeError("PIM server closed stdout")
                    got += chunk
                self._t_read_ms += (_time.perf_counter() - _ts) * 1000
                self._n_calls += 1
                self._n_bytes_sent += len(body_bytes) + 4
                return got
            except (BrokenPipeError, OSError) as e:
                raise RuntimeError(f"PIM server died: {e}") from None

    def _cleanup(self):
        try:
            # 0-length sentinel = quit.
            self.proc.stdin.write(struct.pack('<I', 0))
            self.proc.stdin.flush()
            self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=2)
        except Exception:
            self.proc.kill()

    @classmethod
    def shared(cls, bender_id, bank_id, calib_file, server_path, extra_env=None):
        # extra_env is part of the identity: two servers on the same bender
        # with different sub-ranges would be a config error, but keying on
        # the frozenset makes accidental collisions impossible.
        env_key = tuple(sorted(extra_env.items())) if extra_env else ()
        key = (bender_id, bank_id, calib_file, server_path, env_key)
        with cls._lock:
            if key not in cls._shared:
                cls._shared[key] = cls(bender_id, bank_id, calib_file,
                                       server_path, extra_env=extra_env)
            return cls._shared[key]


class PimBitLinear(nn.Module):
    """Wraps an AutoBitLinear; runs matmul on PIM. Default backend is the
    long-running server (PimServer); pass `use_server=False` for the
    legacy subprocess-per-call path."""

    def __init__(self, base, *, bender_id, calib_file, bank_id, runner_path,
                 server_path=None, use_server=True, verbose=False,
                 dimm_configs=None, weight_spec=None):
        super().__init__()
        # Keep the original module's tensors (registered as attributes)
        self.base = base
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.bender_id = bender_id
        self.calib_file = calib_file
        self.bank_id = bank_id
        self.runner_path = runner_path
        self.use_server = use_server
        self.server_path = server_path
        self.verbose = verbose
        # Multi-DIMM: dimm_configs is a list of dicts, one per DIMM:
        #   {bender, calib, bank, pool_layout (opt), sub_start (opt), sub_end (opt)}
        # When >1 entry, d_in sub-handles are round-robin'd across the DIMMs'
        # servers and run concurrently (each computes a partial sum; host
        # adds). Single-DIMM (default) builds a 1-element list from the
        # legacy (bender_id, calib_file, bank_id) args.
        if dimm_configs is None:
            dimm_configs = [{
                'bender': bender_id, 'calib': calib_file, 'bank': bank_id,
                'pool_layout': os.environ.get('PIM_POOL_LIST_FILE'),
                'sub_start': os.environ.get('PIM_SUB_START'),
                'sub_end': os.environ.get('PIM_SUB_END'),
            }]
        self._dimm_configs = dimm_configs
        self._servers = []
        if use_server:
            if not server_path:
                raise ValueError("server_path required when use_server=True")
            for cfg in dimm_configs:
                extra_env = {}
                if cfg.get('pool_layout'): extra_env['PIM_POOL_LIST_FILE'] = cfg['pool_layout']
                if cfg.get('sub_start') is not None: extra_env['PIM_SUB_START'] = cfg['sub_start']
                if cfg.get('sub_end') is not None: extra_env['PIM_SUB_END'] = cfg['sub_end']
                # O10 (2026-07-20): per-DIMM fused-layout colmask (host
                # repair of fused-marginal columns — dimm0 only today).
                if cfg.get('fused_colmask'):
                    extra_env['PIM_FUSED_COLMASK_FILE'] = cfg['fused_colmask']
                self._servers.append(PimServer.shared(
                    cfg['bender'], cfg['bank'], cfg['calib'],
                    server_path, extra_env=extra_env or None))
            self._server = self._servers[0]   # back-compat alias
        else:
            self._server = None

        # ------- optional external weight spec (Bonsai, 2026-07-20) -------
        # weight_spec=None (default) → legacy BitNet path, byte-identical.
        # weight_spec dict keys:
        #   codes         int8 [out, in] in {-1,0,+1}      (required)
        #   group_scales  f32 [out, in//group_size] or None — per-(row,
        #                 128-input-group) scales (PrismML Bonsai g128).
        #                 None → scalar path with 'weight_scale'.
        #   group_size    int, default 128 (must be a multiple of 32)
        #   residual_idx  int64 [n, 2] (row, col) — sparse exact deltas
        #   residual_val  f32 [n]        w_true − s·code (~1e-5 of weights)
        #   weight_scale  float scalar (only used when group_scales=None)
        # Group mode changes the math to
        #   y[r] = ( Σ_g s[r,g] · popcount-partial_g[r] + Σ resid Δ·x[col] )
        #          / x_scale
        # with vote-then-scale ordering: calib/copy votes happen on the
        # int32 per-group partials BEFORE s[r,g] is applied.
        self._group_scales = None
        self._group_size = 0
        self._group_chunks = 0
        self._resid_rows = None
        self._resid_cols = None
        self._resid_vals = None
        # Residual-correction telemetry (host-side add is ~1e-5-magnitude;
        # measured so the run report can quantify its contribution).
        self._resid_terms_applied = 0
        self._resid_abs_sum = 0.0
        self._resid_abs_max = 0.0
        if weight_spec is None:
            # Pre-extract weight as int8 ternary [out, in].
            w = base.weight  # bf16 [out, in] in {-1, 0, +1}
            if w.dtype == torch.uint8:
                # Older fallback: uint8 {0, 1, 255} → int8.
                w_int = w.view(torch.int8).to(torch.int8)
            else:
                w_int = w.to(torch.int8)
            self._w_int = w_int.detach().cpu().numpy()    # [out, in]
            self._weight_scale = _bf16_to_f32_scalar(base.weight_scale)
        else:
            codes = np.ascontiguousarray(
                np.asarray(weight_spec['codes'], dtype=np.int8))
            if codes.shape != (self.out_features, self.in_features):
                raise ValueError(
                    f"weight_spec codes shape {codes.shape} != "
                    f"({self.out_features}, {self.in_features})")
            self._w_int = codes
            gs = weight_spec.get('group_scales')
            if gs is not None:
                if not use_server:
                    raise ValueError("group_scales requires the server "
                                     "backend (per-group partial dispatch)")
                self._group_size = int(weight_spec.get('group_size', 128))
                if self._group_size % 32 != 0:
                    raise ValueError(f"group_size {self._group_size} must be "
                                     f"a multiple of 32 (chunk width)")
                self._group_chunks = self._group_size // 32
                if self.in_features % self._group_size != 0:
                    raise ValueError(
                        f"d_in {self.in_features} not divisible by "
                        f"group_size {self._group_size}")
                gs = np.ascontiguousarray(np.asarray(gs, dtype=np.float32))
                want = (self.out_features,
                        self.in_features // self._group_size)
                if gs.shape != want:
                    raise ValueError(f"group_scales shape {gs.shape} != {want}")
                self._group_scales = gs
            ridx = weight_spec.get('residual_idx')
            rval = weight_spec.get('residual_val')
            if ridx is not None and rval is not None and len(rval) > 0:
                ridx = np.asarray(ridx, dtype=np.int64)
                self._resid_rows = np.ascontiguousarray(ridx[:, 0])
                self._resid_cols = np.ascontiguousarray(ridx[:, 1])
                self._resid_vals = np.asarray(rval, dtype=np.float32)
            self._weight_scale = float(weight_spec.get('weight_scale', 1.0))
        self._n_calls = 0
        # Profiling counters
        self._t_total_ms = 0.0
        self._t_quant_ms = 0.0
        self._t_matmul_ms = 0.0
        self._t_xbp_ms = 0.0
        self._t_body_build_ms = 0.0
        self._t_request_ms = 0.0
        self._t_parse_ms = 0.0
        self._n_token_matmuls = 0
        self._n_inner_requests = 0
        self._sub_rr = 0   # round-robin cursor for assigning subs to DIMM servers

        # Pre-pack per-slice (pos_mask, neg_mask, request_prefix) for every
        # output slice this linear's forward will dispatch. Weights are
        # immutable across forward calls so this is one-time work.
        self._slices = []
        d_in = self.in_features
        d_out = self.out_features
        n_chunks = d_in // 32
        n_slices = (d_out + D_OUT_SLICE - 1) // D_OUT_SLICE
        powers = (np.uint32(1) << np.arange(32, dtype=np.uint32))[None, :]
        bp_factor_bytes = BITPLANE_FACTORS.tobytes()
        # Single-track (1-bit) mode: server computes only the pos track and
        # the host reconstructs via the complement identity. Requires a
        # server that knows MAGIC_V2S, so server-mode only; incompatible
        # with V2G (grouped responses stay dual).
        _want_single = (os.environ.get('PIM_1BIT_SINGLE', '0') == '1'
                        and use_server
                        and os.environ.get('PIM_GROUP_RESP') != '1')
        for s in range(n_slices):
            a = s * D_OUT_SLICE
            b = min(a + D_OUT_SLICE, d_out)
            n_real = b - a
            W_slice = np.zeros((D_OUT_SLICE, d_in), dtype=np.int8)
            W_slice[:n_real, :] = self._w_int[a:b, :]
            pos_mask = np.zeros((n_chunks, D_OUT_SLICE), dtype=np.uint32)
            neg_mask = np.zeros((n_chunks, D_OUT_SLICE), dtype=np.uint32)
            for c in range(n_chunks):
                seg = W_slice[:, c*32:(c+1)*32]
                pos_mask[c, :n_real] = ((seg[:n_real] == 1).astype(np.uint32)
                                          * powers).sum(axis=1, dtype=np.uint32)
                neg_mask[c, :n_real] = ((seg[:n_real] == -1).astype(np.uint32)
                                          * powers).sum(axis=1, dtype=np.uint32)
            # Free row-replication for partial slices: when n_real < D_OUT_SLICE,
            # the row has unused output slots. Replicate the weight masks into
            # those slots so each output gets ≥2 (preferably 3) physically
            # distinct cell-positions to compute its popcount in. Host majority-
            # votes on the K copies after receive — eliminates the per-output
            # cell-flip outliers that propagate as residual through the model.
            #
            # The only thing that matters for correctness is that the SAME
            # weight pattern lands at copy-offsets 1*n_real, 2*n_real, … —
            # the SiMRA primitive itself is unchanged, so each MAJ3 produces
            # bit-for-bit identical answers across copies modulo intrinsic
            # cell-flip noise (~0.02-0.05% per cell).
            n_copies = max(1, D_OUT_SLICE // n_real) if n_real > 0 else 1
            for k in range(1, n_copies):
                start = k * n_real
                pos_mask[:, start:start+n_real] = pos_mask[:, :n_real]
                neg_mask[:, start:start+n_real] = neg_mask[:, :n_real]
            # Single-track eligibility is per slice: every real output must
            # be zero-free (1-bit weights) so neg == ~pos on real bits.
            slice_single = _want_single and bool((W_slice[:n_real] != 0).all())
            v2_magic = MAGIC_V2S if slice_single else MAGIC_V2
            header = (struct.pack('<I', v2_magic)
                      + struct.pack('<I', d_in)
                      + struct.pack('<I', D_OUT_SLICE)
                      + struct.pack('<I', n_chunks)
                      + struct.pack('<I', N_BITPLANES))
            # Three body templates:
            # 1. v2 fallback (full d_in mask each call): kept as the
            #    correctness-guaranteed path. One request per d_out slice.
            static_prefix_v2 = header + pos_mask.tobytes() + (
                b'' if slice_single else neg_mask.tobytes())

            # 1b. Multi-DIMM V2 split (2026-07-20 balance fix): per-server
            #     chunk-range V2 bodies. The full-body V2 fallback used to
            #     go to self._server = servers[0] unconditionally, so once
            #     the LOAD pools filled (ENOSPC after ~18-22 sub-handles
            #     per DIMM — the full model needs 2940), ~98% of traffic
            #     landed on DIMM 0 and the other DIMM starved (o5 run:
            #     15510 calls/23.5GB on D0 vs 628 calls/51MB on D2).
            #     Split by d_in instead: server i computes chunks
            #     [c_a, c_b) as a partial sum (y = sum_c x[c]*W[:,c] —
            #     same math as the LOAD-mode sub split), host adds.
            #     Built only when >1 server; single-DIMM keeps the
            #     byte-identical legacy full-body path.
            v2_parts = None
            _n_srv_cfg = len(self._servers) if use_server else 1
            if self._group_scales is not None and os.environ.get('PIM_GROUP_RESP') == '1':
                # V2G (2026-07-21): ONE grouped request per server per
                # slice. The server (MAGIC_V2G) returns per-group int32
                # partials [n_groups_part, 2048] instead of one summed
                # vector, so the whole slice needs n_srv round-trips
                # instead of n_groups — same bytes, same DRAM work, same
                # vote-then-scale math. Groups split CONTIGUOUSLY across
                # servers (equal group counts → byte-balanced).
                v2_parts = []
                gc_ = self._group_chunks
                n_groups_ = n_chunks // gc_
                _ns = max(1, _n_srv_cfg)
                per_g = (n_groups_ + _ns - 1) // _ns
                for si in range(_ns):
                    g_a = si * per_g
                    g_b = min(g_a + per_g, n_groups_)
                    if g_a >= g_b:
                        continue
                    c_a = g_a * gc_
                    c_b = g_b * gc_
                    part_header = (struct.pack('<I', MAGIC_V2G)
                                   + struct.pack('<I', (c_b - c_a) * 32)
                                   + struct.pack('<I', D_OUT_SLICE)
                                   + struct.pack('<I', c_b - c_a)
                                   + struct.pack('<I', N_BITPLANES)
                                   + struct.pack('<I', gc_))
                    v2_parts.append({
                        'server_idx': si,
                        'c_a': c_a, 'c_b': c_b,
                        'g_a': g_a, 'g_b': g_b,
                        'prefix': (part_header
                                   + pos_mask[c_a:c_b].tobytes()
                                   + neg_mask[c_a:c_b].tobytes()),
                    })
            elif self._group_scales is not None:
                # Bonsai group mode: per-GROUP V2 part bodies (each part =
                # one 128-input scale window = group_chunks chunks). The
                # single full-body V2 request cannot carry per-group scales
                # so group mode NEVER uses it; parts are round-robin'd
                # across servers (all → server 0 when single-DIMM).
                v2_parts = []
                gc_ = self._group_chunks
                for g in range(n_chunks // gc_):
                    c_a = g * gc_
                    c_b = c_a + gc_
                    part_header = (struct.pack('<I', v2_magic)
                                   + struct.pack('<I', (c_b - c_a) * 32)
                                   + struct.pack('<I', D_OUT_SLICE)
                                   + struct.pack('<I', c_b - c_a)
                                   + struct.pack('<I', N_BITPLANES))
                    v2_parts.append({
                        'server_idx': g % max(1, _n_srv_cfg),
                        'c_a': c_a, 'c_b': c_b, 'group': g,
                        'single': slice_single,
                        'prefix': (part_header
                                   + pos_mask[c_a:c_b].tobytes()
                                   + (b'' if slice_single
                                      else neg_mask[c_a:c_b].tobytes())),
                    })
            elif _n_srv_cfg > 1:
                v2_parts = []
                per = (n_chunks + _n_srv_cfg - 1) // _n_srv_cfg
                for si in range(_n_srv_cfg):
                    c_a = si * per
                    c_b = min(c_a + per, n_chunks)
                    if c_a >= c_b:
                        continue
                    part_header = (struct.pack('<I', v2_magic)
                                   + struct.pack('<I', (c_b - c_a) * 32)
                                   + struct.pack('<I', D_OUT_SLICE)
                                   + struct.pack('<I', c_b - c_a)
                                   + struct.pack('<I', N_BITPLANES))
                    v2_parts.append({
                        'server_idx': si, 'c_a': c_a, 'c_b': c_b,
                        'single': slice_single,
                        'prefix': (part_header
                                   + pos_mask[c_a:c_b].tobytes()
                                   + (b'' if slice_single
                                      else neg_mask[c_a:c_b].tobytes())),
                    })

            # 2. Per-sub-d_in LOAD_WEIGHTS bodies (each sub ≤ MAX_CHUNKS_PER_SUB
            #    chunks → fits inside the validated single-bank persistent
            #    ceiling). For d_in ≤ 512 this collapses to a single sub.
            #    For d_in > 512 (e.g. BitNet's 2048 / 5376) this splits into
            #    ⌈n_chunks / MAX_CHUNKS_PER_SUB⌉ sub-handles, each with its
            #    own pool reservation on the server.
            load_subs = []
            sub_size = max(1, MAX_CHUNKS_PER_SUB)
            if self._group_scales is not None:
                # Group mode: a sub-handle must not straddle a scale-group
                # boundary (its MM3D partial gets exactly ONE s[r,g]).
                # Clamp to the group width, then round down to a divisor of
                # it. Default group 128 → 4 chunks: PIM_MAX_CHUNKS_PER_SUB
                # =16 self-aligns to 4 without needing the env override.
                sub_size = min(sub_size, self._group_chunks)
                while self._group_chunks % sub_size != 0:
                    sub_size -= 1
            n_subs = (n_chunks + sub_size - 1) // sub_size
            for sub_i in range(n_subs):
                c_a = sub_i * sub_size
                c_b = min(c_a + sub_size, n_chunks)
                n_chunks_sub = c_b - c_a
                d_in_sub = n_chunks_sub * 32
                pos_sub = pos_mask[c_a:c_b].copy()
                neg_sub = neg_mask[c_a:c_b].copy()
                handle_id = _NEXT_HANDLE_ID[0]; _NEXT_HANDLE_ID[0] += 1
                load_body = (struct.pack('<I', MAGIC_LOAD)
                             + struct.pack('<I', handle_id)
                             + struct.pack('<I', d_in_sub)
                             + struct.pack('<I', D_OUT_SLICE)
                             + struct.pack('<I', n_chunks_sub)
                             + pos_sub.tobytes() + neg_sub.tobytes())
                mm3d_prefix = (struct.pack('<I', MAGIC_MM3D)
                               + struct.pack('<I', handle_id)
                               + struct.pack('<I', D_OUT_SLICE)
                               + struct.pack('<I', n_chunks_sub)
                               + struct.pack('<I', N_BITPLANES))
                # Round-robin this sub-handle across the DIMM servers so the
                # MM3D partial-sum work is balanced. Single-DIMM → always 0.
                n_srv = max(1, len(self._servers))
                srv_idx = self._sub_rr % n_srv
                self._sub_rr += 1
                sub_entry = {
                    'handle_id': handle_id,
                    'c_a': c_a, 'c_b': c_b,
                    'n_chunks_sub': n_chunks_sub,
                    'load_body': load_body,
                    'mm3d_prefix': mm3d_prefix,
                    'loaded': False,
                    'server_idx': srv_idx,
                }
                if self._group_scales is not None:
                    g = c_a // self._group_chunks
                    assert (c_b - 1) // self._group_chunks == g, \
                        "load sub straddles a scale-group boundary"
                    sub_entry['group'] = g
                load_subs.append(sub_entry)
            slice_entry = {
                'a': a, 'b': b, 'n_real': n_real,
                'n_copies': n_copies,  # 1 = no replication; ≥2 = vote across copies
                'single': slice_single,   # V2S single-track (1-bit) slice
                'static_prefix': static_prefix_v2,
                'v2_parts': v2_parts,   # None single-DIMM; per-server split multi
                'bp_factor_bytes': bp_factor_bytes,
                'load_subs': load_subs,
                # Stashed for int-level diff log (PIM_INT_DIFF=1).
                'W_slice_int': W_slice[:n_real, :].astype(np.int32),
            }
            if self._group_scales is not None:
                # f32 [n_real, n_groups] — REAL output rows only. Votes run
                # on int partials first, so copy rows never need scales.
                slice_entry['group_scales'] = np.ascontiguousarray(
                    self._group_scales[a:b, :])
            self._slices.append(slice_entry)

    @torch.no_grad()
    def forward(self, x):
        """x: bf16 [..., in_features]. Returns bf16 [..., out_features]."""
        import time as _time
        _t_total_start = _time.perf_counter()
        # Apply BitNet's per-token symmetric activation quant.
        in_dtype = x.dtype
        x_f32 = x.float()
        Qn, Qp = -128, 127
        abs_max = x_f32.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5)
        x_scale = Qp / abs_max
        x_q = (x_f32 * x_scale).round().clamp(Qn, Qp).to(torch.int32)
        # x_q: int32 [..., in], x_scale: f32 [..., 1]

        # Process each token independently (PIM matmul = one row).
        orig_shape = x_q.shape   # [..., in]
        flat = x_q.reshape(-1, self.in_features).cpu().numpy().astype(np.int32)
        flat_scale = x_scale.reshape(-1, 1).cpu().numpy().astype(np.float32)
        n_tokens = flat.shape[0]
        _t_quant = _time.perf_counter() - _t_total_start

        y_out_f32 = np.zeros((n_tokens, self.out_features), dtype=np.float32)
        _t_matmul_total = 0.0
        for t in range(n_tokens):
            x_int8_t = flat[t].astype(np.int8)   # int8 [in_features]
            _ts = _time.perf_counter()
            y_int = self._pim_matmul_one_token(x_int8_t)
            _t_matmul_total += _time.perf_counter() - _ts
            if self._group_scales is not None:
                # Group mode: s[r,g] (and the sparse residual correction)
                # were already applied inside the matmul → only the
                # activation rescale remains.
                y_f32 = y_int / flat_scale[t, 0]
            else:
                y_f32 = (y_int.astype(np.float32) * self._weight_scale) / flat_scale[t, 0]
            y_out_f32[t] = y_f32

        out_shape = list(orig_shape[:-1]) + [self.out_features]
        y_out = torch.from_numpy(y_out_f32).reshape(out_shape).to(in_dtype)

        if self.base.bias is not None:
            y_out = y_out + self.base.bias

        # Per-call diff vs PyTorch reference. Tells us, layer-by-layer,
        # how much PIM diverges from the canonical bf16 forward. Gated
        # behind PIM_DIFF_LOG=1 (the comparison adds a CPU forward pass
        # so it isn't free — but it's cheap relative to the PIM time).
        if os.environ.get("PIM_DIFF_LOG"):
            try:
                y_ref = self.base.forward(x).to(in_dtype)
                y_pim = y_out
                # Compare in float32 to avoid bf16 quantisation surprises.
                a = y_ref.float()
                b = y_pim.float()
                diff = (a - b).abs()
                # Per-output match rate at small tolerance — exact int
                # match isn't realistic (rescale + bf16 cast both lose
                # bits), so use rtol=1e-3 + atol=0.5 (1/2 LSB of int).
                tol = 0.5 + 1e-3 * a.abs()
                n_match = int((diff <= tol).sum())
                n_total = int(a.numel())
                max_err = float(diff.max())
                mean_err = float(diff.mean())
                tag = getattr(self, "_diff_tag", "?")
                print(f"   [pim-diff] {tag} call#{self._n_calls} "
                      f"match={n_match}/{n_total} ({100*n_match/n_total:.3f}%) "
                      f"max_err={max_err:.3f} mean_err={mean_err:.4f}",
                      flush=True)
            except Exception as e:
                print(f"   [pim-diff] {getattr(self,'_diff_tag','?')} compare failed: {e}",
                      flush=True)

        self._n_calls += 1
        _t_total = _time.perf_counter() - _t_total_start
        # Sum into PER-INSTANCE counters; the run script can read them at end.
        self._t_total_ms += _t_total * 1000
        self._t_quant_ms += _t_quant * 1000
        self._t_matmul_ms += _t_matmul_total * 1000
        self._n_token_matmuls += n_tokens
        if self.verbose and self._n_calls % 20 == 0:
            print(f"   [pim] forward call #{self._n_calls} ({n_tokens} tokens)", flush=True)
        return y_out

    def _pim_matmul_one_token(self, x_int8):
        """Run integer ternary @ int8 matmul for one token via PIM.
        x_int8: int8 [in_features]. Returns int32 [out_features] in the
        legacy scalar path; float32 [out_features] in group mode (per-group
        scales + sparse residuals already applied, x_scale NOT yet)."""
        import time as _time
        d_in = self.in_features
        d_out = self.out_features
        assert d_in % 32 == 0
        n_chunks = d_in // 32

        # Bit-decompose x_int8 (treated as uint8 byte pattern).  Cached
        # by id+shape so q/k/v calls within the same token reuse the work.
        _ts = _time.perf_counter()
        x_bitplane, x_bp_bytes = _pack_xbp(x_int8, n_chunks)
        self._t_xbp_ms += (_time.perf_counter() - _ts) * 1000

        y = np.zeros(d_out, dtype=np.int32)
        # Group mode accumulates in f32 (scales applied per slice); the
        # legacy scalar path keeps the int32 buffer + late scalar rescale.
        y_f32 = (np.zeros(d_out, dtype=np.float32)
                 if self._group_scales is not None else None)
        for s, slc in enumerate(self._slices):
            a, b, n_real = slc['a'], slc['b'], slc['n_real']
            if self._server is not None:
                # Pre-loaded weights (LOAD_WEIGHTS + MATMUL_HANDLE protocol)
                # is gated behind PIM_USE_LOAD_WEIGHTS=1 because we observed
                # incorrect outputs in some matmul-mix configurations
                # (q-only + gate-only each work; q+gate together produces
                # garbage). Likely related to backup-row data retention
                # between handle-load and handle-use, but not yet root-caused.
                # Default OFF: use v2 protocol (fresh weight upload each call,
                # known correct).
                _use_loadweights = os.environ.get('PIM_USE_LOAD_WEIGHTS', '0') == '1'
                _force_v2 = not _use_loadweights or os.environ.get('PIM_FORCE_V2', '0') == '1'

                # Try to pre-load every sub-handle on first call.  Any failure
                # disables LOAD-mode for the slice (V2 fallback for the whole
                # slice — keeps correctness simple).
                if not _force_v2 and not slc.get('loaded_state_done', False):
                    all_ok = True
                    for sub in slc['load_subs']:
                        if sub['loaded']: continue
                        # Each sub loads its weights into ITS assigned DIMM's
                        # server pool (multi-DIMM); single-DIMM → server 0.
                        srv = self._servers[sub['server_idx']]
                        # ENOSPC latch: once a server refused a LOAD for
                        # pool exhaustion (ack byte 1), its cursor never
                        # retreats — skip the round-trip for all later
                        # slices (the o5 run wasted ~1250 probe calls).
                        if getattr(srv, 'load_enospc', False):
                            all_ok = False
                            break
                        ack = srv.request(sub['load_body'], expect_resp_len=4)
                        if len(ack) != 4 or ack[0] != 0:
                            all_ok = False
                            if len(ack) == 4 and ack[0] == 1:
                                if not getattr(srv, 'load_enospc', False):
                                    srv.load_enospc = True
                                    print(f"[pim] server bender="
                                          f"{srv.bender_id}: LOAD pool "
                                          f"exhausted (ENOSPC) — later "
                                          f"slices use V2"
                                          + (" (multi-DIMM split)"
                                             if len(self._servers) > 1
                                             else ""),
                                          flush=True)
                            break
                        sub['loaded'] = True
                    slc['loaded_state_done'] = True
                    slc['load_mode_ok']      = all_ok
                    if not all_ok:
                        _V2_FALLBACK_COUNT[0] += 1
                        n_fb = _V2_FALLBACK_COUNT[0]
                        if n_fb == 1 or n_fb % 100 == 0:
                            print(f"[pim] LOAD→V2 fallback slices so far: "
                                  f"{n_fb}", flush=True)

                use_load_mode = (not _force_v2 and slc.get('load_mode_ok', False))
                self._t_body_build_ms += (_time.perf_counter() - _ts) * 1000
                # D: 3-way cross-calib voting on full-row slices.
                # n_copies > 1 already covers partial-row in-row voting → 1 trip.
                # For n_copies == 1, do 3 trips with calib_idx 0/1/2 and median.
                # PIM_FORCE_CALIB=N overrides everything — always uses calib N
                # for a single trip. Useful for A/B'ing different sub-clusters.
                slice_n_copies = slc.get('n_copies', 1)
                _force_calib = os.environ.get('PIM_FORCE_CALIB')
                if _force_calib is not None:
                    d_full_vote = False
                else:
                    d_full_vote = (slice_n_copies == 1
                                   and os.environ.get('PIM_VOTE_FULL', '1') == '1')
                # ---------- Bonsai group mode (per-group scales) ----------
                # Vote-then-scale: fetch int32 per-group partials (LOAD →
                # one MM3D per sub-handle routed into its group's slot;
                # V2 → one request per group part), vote on the INT
                # partials (calib trips and/or in-row copies), then apply
                # s[r,g] and reduce over groups in f32. Skips the entire
                # legacy y_slice flow below.
                if self._group_scales is not None:
                    n_groups = n_chunks // self._group_chunks
                    items = (slc['load_subs'] if use_load_mode
                             else slc['v2_parts'])
                    bp_factor_bytes_g = slc['bp_factor_bytes']
                    n_srv = len(self._servers)

                    def _one_calib_groups(cal_idx):
                        partials = [np.zeros((n_groups, D_OUT_SLICE),
                                             dtype=np.int32)
                                    for _ in range(n_srv)]
                        errs = [None] * n_srv

                        def _run_srv(si):
                            try:
                                for it in items:
                                    if it['server_idx'] != si:
                                        continue
                                    x_bp_sub = x_bitplane[it['c_a']:it['c_b']]
                                    prefix = (it['mm3d_prefix']
                                              if use_load_mode
                                              else it['prefix'])
                                    body = (prefix
                                            + x_bp_sub.tobytes()
                                            + bp_factor_bytes_g
                                            + struct.pack('<I', cal_idx))
                                    if 'g_a' in it:
                                        # V2G grouped part: per-group
                                        # partial vectors in one response.
                                        ng_part = it['g_b'] - it['g_a']
                                        resp = self._servers[si].request(
                                            body,
                                            expect_resp_len=ng_part
                                            * D_OUT_SLICE * 4)
                                        partials[si][it['g_a']:it['g_b']] += (
                                            np.frombuffer(resp, dtype=np.int32)
                                            .reshape(ng_part, D_OUT_SLICE))
                                    else:
                                        resp = self._servers[si].request(body)
                                        y_p = np.frombuffer(
                                            resp, dtype=np.int32,
                                            count=D_OUT_SLICE)
                                        if it.get('single'):
                                            # V2S: y = 2*y_pos - Σx (weighted)
                                            y_p = 2 * y_p - np.int32(
                                                _xbp_weighted_popcount(x_bp_sub))
                                        partials[si][it['group']] += y_p
                            except Exception as e:
                                errs[si] = e

                        _serial = (n_srv == 1 or
                                   os.environ.get('PIM_MULTIDIMM_SERIAL') == '1')
                        if _serial:
                            for si in range(n_srv):
                                _run_srv(si)
                        else:
                            ths = [threading.Thread(target=_run_srv,
                                                    args=(si,))
                                   for si in range(n_srv)]
                            for th in ths: th.start()
                            for th in ths: th.join()
                        for e in errs:
                            if e is not None:
                                raise e
                        self._n_inner_requests += len(items)
                        G_acc = partials[0]
                        for p in partials[1:]:
                            G_acc = G_acc + p
                        return G_acc

                    _ts = _time.perf_counter()
                    if d_full_vote:
                        g_trips = [_one_calib_groups(c) for c in (0, 1, 2)]
                        G = np.median(np.stack(g_trips, axis=0),
                                      axis=0).astype(np.int32)
                    else:
                        cidx = (int(_force_calib)
                                if _force_calib is not None else 0)
                        G = _one_calib_groups(cidx)
                    self._t_request_ms += (_time.perf_counter() - _ts) * 1000
                    n_copies_g = slc.get('n_copies', 1)
                    if n_copies_g > 1:
                        copies = G[:, :n_copies_g * n_real].reshape(
                            n_groups, n_copies_g, n_real)
                        if n_copies_g >= 3:
                            Gv = np.median(copies, axis=1).astype(np.int32)
                        else:
                            Gv = ((copies[:, 0].astype(np.int64)
                                   + copies[:, 1].astype(np.int64)) // 2
                                  ).astype(np.int32)
                    else:
                        Gv = G[:, :n_real]
                    # Scale AFTER voting; per-row dot over groups in f32.
                    gs_slice = slc['group_scales']   # f32 [n_real, n_groups]
                    y_f32[a:b] = (gs_slice * Gv.T.astype(np.float32)
                                  ).sum(axis=1, dtype=np.float32)
                    if os.environ.get("PIM_INT_DIFF"):
                        W_slice_int = slc.get('W_slice_int')
                        if W_slice_int is not None:
                            y_pim_int = Gv.astype(np.int64).sum(axis=0)
                            y_ref_int = (W_slice_int.astype(np.int64)
                                         @ x_int8.astype(np.int64))
                            diff = np.abs(y_pim_int - y_ref_int)
                            n_exact = int((diff == 0).sum())
                            tag = getattr(self, "_diff_tag", "?")
                            print(f"   [pim-int-diff] {tag} slice{s} "
                                  f"call#{self._n_calls} groups-sum "
                                  f"exact={n_exact}/{n_real} "
                                  f"({100*n_exact/max(1,n_real):.4f}%) "
                                  f"max_err={int(diff.max())} "
                                  f"mean_err={float(diff.mean()):.4f}",
                                  flush=True)
                    continue
                # ---------- end group mode ----------

                # Helper: produce one slice's y at a given calib index. In
                # LOAD-mode this dispatches one MM3D request per d_in
                # sub-handle and sums the partial outputs (math: y = sum_c
                # x[c]·W[:,c] = sum_sub sum_{c in sub} x[c]·W[:,c] = sum_sub
                # y_sub).  In V2-mode it's one request with the full d_in
                # body.
                def _one_calib(cal_idx):
                    bp_factor_bytes = slc['bp_factor_bytes']
                    if use_load_mode:
                        n_srv = len(self._servers)
                        if n_srv == 1:
                            y_acc = np.zeros(D_OUT_SLICE, dtype=np.int32)
                            for sub in slc['load_subs']:
                                x_bp_sub = x_bitplane[sub['c_a']:sub['c_b']]
                                body = (sub['mm3d_prefix']
                                        + x_bp_sub.tobytes()
                                        + bp_factor_bytes
                                        + struct.pack('<I', cal_idx))
                                resp = self._servers[0].request(body)
                                y_acc += np.frombuffer(resp, dtype=np.int32,
                                                        count=D_OUT_SLICE)
                                self._n_inner_requests += 1
                            return y_acc
                        # Multi-DIMM: one thread per server fires that server's
                        # sub-handles concurrently (independent subprocesses /
                        # XDMA channels), each accumulating a partial sum; host
                        # adds. The partial sums are valid because
                        # y = Σ_c x[c]·W[:,c] = Σ_subs (Σ_{c∈sub} x[c]·W[:,c]).
                        partials = [np.zeros(D_OUT_SLICE, dtype=np.int32)
                                    for _ in range(n_srv)]
                        # PIM_MULTIDIMM_SERIAL=1: run the per-server work
                        # sequentially (no concurrency). Correctness-only mode
                        # — validates the d_in split + sum without exercising
                        # concurrent c2h readback across benders (which can
                        # wedge a channel — see investigation).
                        _serial = os.environ.get('PIM_MULTIDIMM_SERIAL') == '1'
                        def _run_srv(si):
                            for sub in slc['load_subs']:
                                if sub['server_idx'] != si:
                                    continue
                                x_bp_sub = x_bitplane[sub['c_a']:sub['c_b']]
                                body = (sub['mm3d_prefix']
                                        + x_bp_sub.tobytes()
                                        + bp_factor_bytes
                                        + struct.pack('<I', cal_idx))
                                resp = self._servers[si].request(body)
                                partials[si] += np.frombuffer(
                                    resp, dtype=np.int32, count=D_OUT_SLICE)
                        if _serial:
                            for si in range(n_srv): _run_srv(si)
                        else:
                            ths = [threading.Thread(target=_run_srv, args=(si,))
                                   for si in range(n_srv)]
                            for th in ths: th.start()
                            for th in ths: th.join()
                        self._n_inner_requests += len(slc['load_subs'])
                        y_acc = np.zeros(D_OUT_SLICE, dtype=np.int32)
                        for p in partials: y_acc += p
                        return y_acc
                    else:
                        parts = slc.get('v2_parts')
                        if parts and os.environ.get('PIM_V2_SPLIT', '1') != '1':
                            parts = None   # kill-switch: legacy servers[0] route
                        if parts:
                            # 2026-07-20 BALANCE FIX: multi-DIMM V2. Each
                            # server computes its d_in chunk range as a
                            # partial sum, concurrently; host adds. The
                            # old path sent the full body to servers[0]
                            # only, starving every other DIMM once the
                            # LOAD pools were exhausted.
                            partials = [None] * len(parts)
                            errs = [None] * len(parts)
                            def _run_part(pi):
                                try:
                                    part = parts[pi]
                                    x_bp_part = x_bitplane[part['c_a']:part['c_b']]
                                    body = (part['prefix']
                                            + x_bp_part.tobytes()
                                            + bp_factor_bytes
                                            + struct.pack('<I', cal_idx))
                                    resp = self._servers[part['server_idx']].request(body)
                                    y_p = np.frombuffer(
                                        resp, dtype=np.int32,
                                        count=D_OUT_SLICE)
                                    if part.get('single'):
                                        y_p = 2 * y_p - np.int32(
                                            _xbp_weighted_popcount(x_bp_part))
                                    partials[pi] = y_p
                                except Exception as e:
                                    errs[pi] = e
                            _serial = os.environ.get('PIM_MULTIDIMM_SERIAL') == '1'
                            if _serial:
                                for pi in range(len(parts)): _run_part(pi)
                            else:
                                ths = [threading.Thread(target=_run_part,
                                                        args=(pi,))
                                       for pi in range(len(parts))]
                                for th in ths: th.start()
                                for th in ths: th.join()
                            for e in errs:
                                if e is not None:
                                    raise e
                            self._n_inner_requests += len(parts)
                            y_acc = np.zeros(D_OUT_SLICE, dtype=np.int32)
                            for p in partials: y_acc += p
                            return y_acc
                        body = (slc['static_prefix']
                                + x_bp_bytes
                                + bp_factor_bytes
                                + struct.pack('<I', cal_idx))
                        resp = self._server.request(body)
                        self._n_inner_requests += 1
                        y_full = np.frombuffer(resp, dtype=np.int32,
                                               count=D_OUT_SLICE).copy()
                        if slc.get('single'):
                            y_full = 2 * y_full - np.int32(
                                _xbp_weighted_popcount(x_bitplane))
                        return y_full

                _ts = _time.perf_counter()
                if d_full_vote:
                    y_trips = [_one_calib(c) for c in (0, 1, 2)]
                    y_slice = np.median(np.stack(y_trips, axis=0),
                                          axis=0).astype(np.int32)
                else:
                    cidx = int(_force_calib) if _force_calib is not None else 0
                    y_slice = _one_calib(cidx)
                self._t_request_ms += (_time.perf_counter() - _ts) * 1000
                _ts = _time.perf_counter()
                self._t_parse_ms += (_time.perf_counter() - _ts) * 1000
            else:
                # Legacy subprocess-per-call backend.  Always uses the V2
                # full-d_in body — no persistent handles in this backend.
                body = (slc['static_prefix']
                        + x_bp_bytes
                        + slc['bp_factor_bytes']
                        + struct.pack('<I', 0))
                with tempfile.NamedTemporaryFile(prefix='pim_in_', suffix='.bin',
                                                  delete=False) as f:
                    in_path = f.name; f.write(body)
                out_path = in_path.replace('pim_in_', 'pim_out_')
                rc = subprocess.call([self.runner_path, str(self.bender_id),
                                      self.calib_file, str(self.bank_id),
                                      in_path, out_path],
                                     stderr=subprocess.STDOUT,
                                     stdout=subprocess.DEVNULL)
                if rc != 0:
                    os.unlink(in_path); raise RuntimeError(f"runner rc={rc}")
                with open(out_path, 'rb') as f:
                    y_slice = np.frombuffer(f.read(), dtype=np.int32,
                                             count=D_OUT_SLICE).copy()
                os.unlink(in_path); os.unlink(out_path)
            n_copies = slc.get('n_copies', 1)
            if n_copies > 1:
                # Reshape into [n_copies, n_real]; vote.
                # 3+ copies → element-wise median (handles ties cleanly).
                # 2 copies → take element-wise mean (no clean majority,
                # but mean of 2 close-to-correct values is still a
                # better estimate than picking either copy).
                copies = y_slice[:n_copies * n_real].reshape(n_copies, n_real)
                if n_copies >= 3:
                    voted = np.median(copies, axis=0).astype(np.int32)
                else:
                    voted = ((copies[0].astype(np.int64)
                              + copies[1].astype(np.int64)) // 2
                            ).astype(np.int32)
                y[a:b] = voted
            else:
                y[a:b] = y_slice[:n_real]

            # PIM-level int diff: compare PIM's int32 output (after vote)
            # to the int reference y_int_ref = W_slice @ x_int8. This is
            # the *purely PIM* error metric — independent of bf16
            # accumulation order, rescale precision, etc. Phase 4d showed
            # this should be 0 errors in isolation; cumulative ACT-disturb
            # may introduce some.
            if os.environ.get("PIM_INT_DIFF"):
                W_slice_int = slc.get('W_slice_int')
                if W_slice_int is not None:
                    n_real_s = slc['n_real']
                    y_pim_int = y[a:b].astype(np.int64)
                    y_ref_int = (W_slice_int.astype(np.int64)
                                 @ x_int8.astype(np.int64))
                    diff = np.abs(y_pim_int - y_ref_int)
                    n_exact = int((diff == 0).sum())
                    n_total = n_real_s
                    max_e = int(diff.max())
                    mean_e = float(diff.mean())
                    tag = getattr(self, "_diff_tag", "?")
                    print(f"   [pim-int-diff] {tag} slice{s} call#{self._n_calls} "
                          f"exact={n_exact}/{n_total} ({100*n_exact/max(1,n_total):.4f}%) "
                          f"max_err={max_e} mean_err={mean_e:.4f} "
                          f"y_pim[0..5]={y_pim_int[:5].tolist()} "
                          f"y_ref[0..5]={y_ref_int[:5].tolist()}", flush=True)
        if self._group_scales is not None:
            # Sparse residual correction: y[r] += Δ·x_int8[col] for the
            # ~1e-5 of weights that sit a few fp16 ulps off their group
            # level. Host-side, f32, index order = npz order (np.add.at is
            # unbuffered → deterministic). Telemetry accumulated so runs
            # can report the measured contribution.
            if self._resid_rows is not None:
                contrib = (self._resid_vals
                           * x_int8[self._resid_cols].astype(np.float32))
                np.add.at(y_f32, self._resid_rows, contrib)
                self._resid_terms_applied += int(contrib.size)
                _abs = np.abs(contrib)
                self._resid_abs_sum += float(_abs.sum())
                _m = float(_abs.max()) if contrib.size else 0.0
                if _m > self._resid_abs_max:
                    self._resid_abs_max = _m
            return y_f32
        return y


def pim_substitute(model, layer_indices, projections, *,
                   bender_id, calib_file, bank_id, runner_path,
                   server_path=None, use_server=True, verbose=False,
                   dimm_configs=None, weight_spec_fn=None):
    """In-place: replace selected projections in selected layers with
    PIM-backed versions.

    layer_indices: iterable of layer indices to substitute.
    projections: list of attribute paths within the layer, e.g.
       ["self_attn.q_proj", "self_attn.k_proj", "mlp.gate_proj", ...]
    use_server: if True (default), use the long-running PIM server
       (1 subprocess shared across all PimBitLinear instances).
    weight_spec_fn: optional callable (layer_idx, proj_path) → weight_spec
       dict (see PimBitLinear.__init__) for models whose codes/scales come
       from external files (PrismML Bonsai). None (default) = legacy
       BitNet extraction from the module's own weight/weight_scale.
    """
    n_replaced = 0
    for li in layer_indices:
        layer = model.model.layers[li]
        for proj_path in projections:
            parts = proj_path.split('.')
            parent = layer
            for p in parts[:-1]:
                parent = getattr(parent, p)
            base = getattr(parent, parts[-1])
            spec = weight_spec_fn(li, proj_path) if weight_spec_fn else None
            wrapped = PimBitLinear(
                base, bender_id=bender_id, calib_file=calib_file,
                bank_id=bank_id, runner_path=runner_path,
                server_path=server_path, use_server=use_server,
                verbose=verbose, dimm_configs=dimm_configs,
                weight_spec=spec)
            wrapped._diff_tag = f"L{li:02d}.{parts[-1]}"
            setattr(parent, parts[-1], wrapped)
            n_replaced += 1
            if verbose:
                print(f"[pim] substituted layer {li}.{proj_path} "
                      f"({base.in_features} → {base.out_features})", flush=True)
    backend = "server" if use_server else "subprocess-per-call"
    print(f"[pim] {n_replaced} projections now run on PIM ({backend})",
          flush=True)


def print_pim_timing_summary(model):
    """Walk the model and print accumulated timing counters for every
    PimBitLinear. Use after generation to see where Python time went."""
    print("\n=== PIM forward-path timing summary ===")
    print(f"{'layer':>5} {'projection':<22} {'forward':>9} {'quant':>8} "
          f"{'matmul':>9} {'xbp':>7} {'body':>7} {'request':>9} {'parse':>7} "
          f"{'#calls':>7} {'#tok':>5} {'#req':>5}")
    total_total = 0.0; total_request = 0.0; total_xbp = 0.0; total_body = 0.0
    for li, layer in enumerate(model.model.layers):
        for proj_name in ['self_attn.q_proj', 'self_attn.k_proj',
                           'self_attn.v_proj', 'self_attn.o_proj',
                           'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj']:
            parts = proj_name.split('.')
            obj = layer
            for p in parts:
                obj = getattr(obj, p, None)
                if obj is None:
                    break
            if obj is None or not isinstance(obj, PimBitLinear):
                continue
            print(f"{li:>5} {proj_name:<22} "
                  f"{obj._t_total_ms:>8.1f} {obj._t_quant_ms:>7.1f} "
                  f"{obj._t_matmul_ms:>8.1f} {obj._t_xbp_ms:>6.1f} "
                  f"{obj._t_body_build_ms:>6.1f} {obj._t_request_ms:>8.1f} "
                  f"{obj._t_parse_ms:>6.1f} "
                  f"{obj._n_calls:>7} {obj._n_token_matmuls:>5} "
                  f"{obj._n_inner_requests:>5}")
            total_total += obj._t_total_ms
            total_request += obj._t_request_ms
            total_xbp += obj._t_xbp_ms
            total_body += obj._t_body_build_ms
    print(f"-- totals: forward={total_total:.0f}ms  "
          f"server-request={total_request:.0f}ms  "
          f"x_bitplane-build={total_xbp:.0f}ms  "
          f"body-concat={total_body:.0f}ms")
    # Bonsai group mode only: sparse-residual telemetry (silent for BitNet).
    _rt = 0; _rs = 0.0; _rm = 0.0
    for layer in model.model.layers:
        for m in layer.modules():
            if isinstance(m, PimBitLinear) and m._resid_terms_applied:
                _rt += m._resid_terms_applied
                _rs += m._resid_abs_sum
                _rm = max(_rm, m._resid_abs_max)
    if _rt:
        print(f"-- residual correction: {_rt} terms applied, "
              f"sum|Δy_int|={_rs:.3f}, max single |Δy_int|={_rm:.5f} "
              f"(pre-x_scale units)")
    # Per-PimServer pipe timing (one server shared across PimBitLinears).
    print()
    for key, srv in PimServer._shared.items():
        print(f"PimServer{key}:  pipe-write={srv._t_write_ms:.0f}ms  "
              f"pipe-read={srv._t_read_ms:.0f}ms  "
              f"calls={srv._n_calls}  "
              f"bytes-sent={srv._n_bytes_sent/1e6:.1f}MB  "
              f"server-time-implied={(total_request - srv._t_write_ms - srv._t_read_ms):.0f}ms")
