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
  4. Concatenate slices, rescale by 1/(input_scale * weight_scale).
"""
import atexit, os, struct, subprocess, sys, tempfile, threading
import numpy as np
import torch
import torch.nn as nn

D_OUT_SLICE = 2048
N_BITPLANES = 8
BITPLANE_FACTORS = np.array([1, 2, 4, 8, 16, 32, 64, -128], dtype=np.int32)
MAGIC_V2 = 0xB17EF002


def _bf16_to_f32_scalar(t):
    return float(t.float().item())


# --------------------- long-running server backend ---------------------

class PimServer:
    """One long-running `bitnet-proj-server` subprocess per (bender, bank).
    Opens the FPGA platform once at startup, then accepts requests over
    stdin/stdout (binary length-prefixed protocol)."""

    _shared = {}    # (bender, bank, calib_file, server_path) -> PimServer
    _lock = threading.Lock()

    def __init__(self, bender_id, bank_id, calib_file, server_path):
        self.bender_id = bender_id
        self.bank_id = bank_id
        self.proc = subprocess.Popen(
            [server_path, str(bender_id), calib_file, str(bank_id)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,    # silence per-token noise
            bufsize=0,
        )
        self.lock = threading.Lock()
        self._n_calls = 0
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

    def request(self, body_bytes):
        """Send one request, return 8192 bytes of int32 result."""
        with self.lock:
            try:
                # u32 length prefix + body, both with looped writes.
                self._write_all(self.proc.stdin, struct.pack('<I', len(body_bytes)))
                self._write_all(self.proc.stdin, body_bytes)
                self.proc.stdin.flush()
                # Read exactly 8192 bytes back (= int32 × 2048).
                got = b""
                while len(got) < 8192:
                    chunk = self.proc.stdout.read(8192 - len(got))
                    if not chunk:
                        raise RuntimeError("PIM server closed stdout")
                    got += chunk
                self._n_calls += 1
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
    def shared(cls, bender_id, bank_id, calib_file, server_path):
        key = (bender_id, bank_id, calib_file, server_path)
        with cls._lock:
            if key not in cls._shared:
                cls._shared[key] = cls(bender_id, bank_id, calib_file, server_path)
            return cls._shared[key]


class PimBitLinear(nn.Module):
    """Wraps an AutoBitLinear; runs matmul on PIM. Default backend is the
    long-running server (PimServer); pass `use_server=False` for the
    legacy subprocess-per-call path."""

    def __init__(self, base, *, bender_id, calib_file, bank_id, runner_path,
                 server_path=None, use_server=True, verbose=False):
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
        if use_server:
            if not server_path:
                raise ValueError("server_path required when use_server=True")
            self._server = PimServer.shared(bender_id, bank_id,
                                             calib_file, server_path)
        else:
            self._server = None

        # Pre-extract weight as int8 ternary [out, in].
        w = base.weight  # bf16 [out, in] in {-1, 0, +1}
        if w.dtype == torch.uint8:
            # Older fallback: uint8 {0, 1, 255} → int8.
            w_int = w.view(torch.int8).to(torch.int8)
        else:
            w_int = w.to(torch.int8)
        self._w_int = w_int.detach().cpu().numpy()    # [out, in]
        self._weight_scale = _bf16_to_f32_scalar(base.weight_scale)
        self._n_calls = 0

    @torch.no_grad()
    def forward(self, x):
        """x: bf16 [..., in_features]. Returns bf16 [..., out_features]."""
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

        y_out_f32 = np.zeros((n_tokens, self.out_features), dtype=np.float32)
        for t in range(n_tokens):
            x_int8_t = flat[t].astype(np.int8)   # int8 [in_features]
            y_int = self._pim_matmul_one_token(x_int8_t)
            y_f32 = y_int.astype(np.float32) / (flat_scale[t, 0] * self._weight_scale)
            y_out_f32[t] = y_f32

        out_shape = list(orig_shape[:-1]) + [self.out_features]
        y_out = torch.from_numpy(y_out_f32).reshape(out_shape).to(in_dtype)

        if self.base.bias is not None:
            y_out = y_out + self.base.bias

        self._n_calls += 1
        if self.verbose and self._n_calls % 20 == 0:
            print(f"   [pim] forward call #{self._n_calls} ({n_tokens} tokens)", flush=True)
        return y_out

    def _pim_matmul_one_token(self, x_int8):
        """Run integer ternary @ int8 matmul for one token via PIM.
        x_int8: int8 [in_features]. Returns int32 [out_features]."""
        d_in = self.in_features
        d_out = self.out_features
        assert d_in % 32 == 0
        n_chunks = d_in // 32

        # Bit-decompose x_int8 (treated as uint8 byte pattern).
        x_u8 = x_int8.astype(np.uint8)
        x_bitplane = np.zeros((n_chunks, N_BITPLANES), dtype=np.uint32)
        for c in range(n_chunks):
            chunk = x_u8[c*32:(c+1)*32]
            for b in range(N_BITPLANES):
                bits = ((chunk >> b) & 1).astype(np.uint32)
                x_bitplane[c, b] = (bits *
                    (np.uint32(1) << np.arange(32, dtype=np.uint32))).sum(dtype=np.uint32)

        n_slices = (d_out + D_OUT_SLICE - 1) // D_OUT_SLICE
        y = np.zeros(d_out, dtype=np.int32)
        for s in range(n_slices):
            a = s * D_OUT_SLICE
            b = min(a + D_OUT_SLICE, d_out)
            n_real = b - a

            # Build pos/neg masks for this slice (zero-pad outputs).
            # Free row-replication: when n_real < D_OUT_SLICE, the row has
            # unused output slots. Replicate the weight masks into them so
            # each output gets ≥2 (preferably 3) physically distinct cell
            # positions to compute its popcount in. Host majority-votes
            # after receive — eliminates the per-output cell-flip outliers
            # that propagate as residual through the model. Costs zero
            # extra MAJ3 ops because the row was already going to be
            # written; we just fill the spare bytes with copies.
            W_slice = np.zeros((D_OUT_SLICE, d_in), dtype=np.int8)
            W_slice[:n_real, :] = self._w_int[a:b, :]
            pos_mask = np.zeros((n_chunks, D_OUT_SLICE), dtype=np.uint32)
            neg_mask = np.zeros((n_chunks, D_OUT_SLICE), dtype=np.uint32)
            powers = (np.uint32(1) << np.arange(32, dtype=np.uint32))[None, :]
            for c in range(n_chunks):
                seg = W_slice[:n_real, c*32:(c+1)*32]
                pos_mask[c, :n_real] = ((seg == 1).astype(np.uint32) * powers).sum(axis=1, dtype=np.uint32)
                neg_mask[c, :n_real] = ((seg == -1).astype(np.uint32) * powers).sum(axis=1, dtype=np.uint32)
            n_copies = max(1, D_OUT_SLICE // n_real) if n_real > 0 else 1
            for k in range(1, n_copies):
                start = k * n_real
                pos_mask[:, start:start + n_real] = pos_mask[:, :n_real]
                neg_mask[:, start:start + n_real] = neg_mask[:, :n_real]

            # Build the V2 body. Header is 5 × uint32; the optional
            # trailing calib_idx is appended per request below.
            body_no_idx = (struct.pack('<I', MAGIC_V2)
                           + struct.pack('<I', d_in)
                           + struct.pack('<I', D_OUT_SLICE)
                           + struct.pack('<I', n_chunks)
                           + struct.pack('<I', N_BITPLANES)
                           + pos_mask.tobytes()
                           + neg_mask.tobytes()
                           + x_bitplane.tobytes()
                           + BITPLANE_FACTORS.tobytes())
            # 3-vote cross-calib correction on FULL-row slices (n_copies==1).
            # Partial-row slices (n_copies>=2) are corrected in-row by C and
            # send 1 trip. Set PIM_VOTE_FULL=0 to disable full-row voting.
            d_full_vote = (n_copies == 1
                           and os.environ.get('PIM_VOTE_FULL', '1') == '1')
            if self._server is not None:
                # Long-running server backend.
                if d_full_vote:
                    y_trips = []
                    for cal_idx in (0, 1, 2):
                        body = body_no_idx + struct.pack('<I', cal_idx)
                        resp = self._server.request(body)
                        y_trips.append(np.frombuffer(resp, dtype=np.int32,
                                                       count=D_OUT_SLICE).copy())
                    y_slice = np.median(np.stack(y_trips, axis=0),
                                          axis=0).astype(np.int32)
                else:
                    body = body_no_idx + struct.pack('<I', 0)
                    resp = self._server.request(body)
                    y_slice = np.frombuffer(resp, dtype=np.int32,
                                              count=D_OUT_SLICE).copy()
            else:
                # Legacy subprocess-per-call backend (no D voting in this
                # path — only the long-running server supports calib_idx).
                body = body_no_idx + struct.pack('<I', 0)
                with tempfile.NamedTemporaryFile(prefix='pim_in_', suffix='.bin',
                                                  delete=False) as f:
                    in_path = f.name; f.write(body)
                out_path = in_path.replace('pim_in_', 'pim_out_')
                rc = subprocess.call([self.runner_path, str(self.bender_id),
                                      self.calib_file, str(self.bank_id),
                                      in_path, out_path],
                                     stderr=subprocess.DEVNULL,
                                     stdout=subprocess.DEVNULL)
                if rc != 0:
                    os.unlink(in_path); raise RuntimeError(f"runner rc={rc}")
                with open(out_path, 'rb') as f:
                    y_slice = np.frombuffer(f.read(), dtype=np.int32,
                                             count=D_OUT_SLICE).copy()
                os.unlink(in_path); os.unlink(out_path)
            # Vote across replicated copies (free if n_copies > 1).
            if n_copies > 1:
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
        return y


def pim_substitute(model, layer_indices, projections, *,
                   bender_id, calib_file, bank_id, runner_path,
                   server_path=None, use_server=True, verbose=False):
    """In-place: replace selected projections in selected layers with
    PIM-backed versions.

    layer_indices: iterable of layer indices to substitute.
    projections: list of attribute paths within the layer, e.g.
       ["self_attn.q_proj", "self_attn.k_proj", "mlp.gate_proj", ...]
    use_server: if True (default), use the long-running PIM server
       (1 subprocess shared across all PimBitLinear instances).
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
            wrapped = PimBitLinear(
                base, bender_id=bender_id, calib_file=calib_file,
                bank_id=bank_id, runner_path=runner_path,
                server_path=server_path, use_server=use_server,
                verbose=verbose)
            setattr(parent, parts[-1], wrapped)
            n_replaced += 1
            if verbose:
                print(f"[pim] substituted layer {li}.{proj_path} "
                      f"({base.in_features} → {base.out_features})", flush=True)
    backend = "server" if use_server else "subprocess-per-call"
    print(f"[pim] {n_replaced} projections now run on PIM ({backend})",
          flush=True)
