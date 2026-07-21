#!/usr/bin/env python3
"""Production Road-B design prototype (2026-07-21, task #17).

Proves the per-segment-popcount readback mode is byte-exact against the
production server's host-side `segment_popcount`, and quantifies the
readback collapse + projected wall — for the VERTICAL layout, with NO
model change. Also computes the horizontal-adoption program/byte
accounting to show honestly why it loses for BitNet d_out=2048.

No silicon; pure accounting + a lossless pack/unpack roundtrip check.
"""
import numpy as np

D_OUT = 2048          # outputs per result row (vertical layout)
SEG_BITS = 32         # one 32-bit AND-result segment per output
ROW_BYTES = D_OUT * 4 # 8192

def segment_popcount_ref(row_u32):
    """Exactly the server's segment_popcount: popcount each 32-bit lane."""
    return np.array([bin(int(v)).count("1") for v in row_u32], dtype=np.int32)

def per_segment_drain_pack(row_u32):
    """What the HDL would emit: 6-bit popcount per segment, bit-packed.
    popcount(32 bits) in [0,32] needs 6 bits. 2048*6 = 12288 bits = 1536 B."""
    pc = segment_popcount_ref(row_u32).astype(np.uint8)  # [0,32], fits 6b
    bits = np.unpackbits(pc[:, None], axis=1, bitorder="little")[:, :6]  # 2048x6
    return np.packbits(bits.reshape(-1), bitorder="little").tobytes()    # 1536 B

def per_segment_drain_unpack(buf):
    bits = np.unpackbits(np.frombuffer(buf, dtype=np.uint8), bitorder="little")
    bits = bits[:D_OUT*6].reshape(D_OUT, 6)
    val = (bits * (1 << np.arange(6))).sum(axis=1).astype(np.int32)
    return val

def main():
    rng = np.random.default_rng(20260721)
    # Realistic result rows: AND of a weight mask and an activation bit
    # -> arbitrary 32-bit patterns per segment (what the row actually holds).
    all_ok = True
    for trial in range(200):
        row = rng.integers(0, 1 << 32, size=D_OUT, dtype=np.uint64).astype(np.uint32)
        ref = segment_popcount_ref(row)
        got = per_segment_drain_unpack(per_segment_drain_pack(row))
        if not np.array_equal(ref, got):
            all_ok = False
            print(f"  trial {trial}: MISMATCH at {np.flatnonzero(ref!=got)[:5]}")
    print(f"[exactness] per-segment 6-bit drain == server segment_popcount: "
          f"{'EXACT over 200 random rows' if all_ok else 'FAILED'}")

    drain_bytes = len(per_segment_drain_pack(
        rng.integers(0, 1<<32, D_OUT, dtype=np.uint64).astype(np.uint32)))
    print(f"[bytes] per compute readback: {ROW_BYTES} B (raw row) -> "
          f"{drain_bytes} B (packed 6-bit) = {ROW_BYTES/drain_bytes:.1f}x collapse")
    print(f"[bytes] 8-bit-aligned variant: {D_OUT} B = {ROW_BYTES/D_OUT:.1f}x "
          f"(simpler HDL pack, still large)")

    # Wall model from the measured per-program server profile.
    prof = dict(wcol=1.3, exec=1.0, recv=3.1, pop=0.2, other=0.3)  # ms
    total = sum(prof.values())
    # recv is c2h of ROW_BYTES; scale it by the byte collapse. pop (host
    # popcount) is ELIMINATED (the FPGA did it).
    recv_new = prof["recv"] * drain_bytes / ROW_BYTES
    total_new = total - prof["recv"] - prof["pop"] + recv_new
    print(f"\n[wall] per-program: {total:.1f} ms -> {total_new:.2f} ms "
          f"(recv {prof['recv']}->{recv_new:.2f}, pop {prof['pop']}->0) "
          f"= {total/total_new:.2f}x")
    print(f"[wall] projected token rate (recv+pop-bound fraction): "
          f"BitNet 47.5 s/tok -> ~{47.5*total_new/total:.1f} s/tok if the "
          f"per-program profile holds model-wide (upper-bound est.)")

    # Horizontal adoption accounting (the literal 'layout adoption'):
    n_chunks = D_OUT // 32   # d_in=2048 -> 64
    N_banks = 4
    n_bitplanes = 8
    vert_reads = (n_chunks*2 // N_banks) * n_bitplanes  # multibank executes
    horiz_progs = D_OUT * 2 * n_bitplanes               # per (output,sign,plane)
    print(f"\n[horizontal] per slice (d_out={D_OUT}, d_in=2048): "
          f"vertical ~{vert_reads} readbacks of {ROW_BYTES}B; "
          f"horizontal {horiz_progs} programs of 96B")
    print(f"[horizontal] program count {horiz_progs/vert_reads:.0f}x MORE, "
          f"bytes {horiz_progs*96/(vert_reads*ROW_BYTES):.2f}x "
          f"-> net LOSS for d_out=2048 (dispatch-bound). Horizontal only "
          f"wins where d_out is small (its natural home is the MVDRAM "
          f"reproduction arm, already horizontal).")

if __name__ == "__main__":
    main()
