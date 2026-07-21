# Road B in the production server: the layout question, answered

> **STATUS 2026-07-21 (built + silicon-validated)**: the recommended
> SEG_POP mode below is now REAL — build 7 (`rtl/readback_engine_build7.v`,
> trailer `0xDBC0DE05`), Verilator-proven and validated on the BCU1525
> (3 pattern cases 2048/2048 segment-bytes exact; READ/DIFF bit-identical
> to build 6). One deliberate divergence from the sketch: the built
> engine packs each 6-bit popcount into a **byte** (2048 B/row, 4×)
> rather than 96-bit-packed beats (1536 B, 5.3×) — byte alignment makes
> the host unpack a plain array read, worth the extra 512 B. Production
> integration: `app/test_bitnet_server.cpp` `PIM_SEGPOP=1`. Results:
> `docs/ROADB_2026_07.md`.

**Goal**: collapse the readback term that dominates the production
BitNet/Bonsai per-token wall, the way Road B already does for the
MVDRAM-reproduction (lane2) server. The obvious route is to make the
production model adopt the horizontal (row-per-output) layout lane2
uses. This document works that through — and lands on a **better answer
that keeps the vertical layout unchanged**.

Measured starting point (per-program server profile, 47.5 s/tok BitNet):

    total 5.9 ms = wcol 1.3 + exec 1.0 + recv 3.1 + pop 0.2 + other 0.3

`recv` (the 8 KB c2h of the result row) is the largest term; `pop` is
the host-side per-segment popcount that turns that row into 2048 output
partials.

## Why the two layouts differ for readback

- **Vertical (production BitNet/Bonsai)**: one program computes a
  (weight-chunk, sign, activation-plane) unit for **all 2048 outputs at
  once** — each output owns a 32-bit segment of the result row, and the
  host popcounts each segment (`segment_popcount`). Few programs,
  massively parallel — but the outputs live *across* the row's bitlines,
  so a whole-row popcount (what Road B's accumulator does) sums *across
  outputs* and is meaningless. Road B as-built cannot help here.
- **Horizontal (lane2/MVDRAM)**: one program computes **one output's**
  full-K AND-popcount — the whole-row total *is* that output's partial,
  so Road B collapses it 8 KB → 96 B. But one output per program.

## The horizontal-adoption accounting (why it loses for BitNet)

For a d_out=2048, d_in=2048 slice (`rtl/seg_pop_prototype.py`):

| | vertical (today) | horizontal adoption |
|---|---|---|
| programs / slice | ~256 readbacks | **32,768** (out × sign × plane) |
| bytes / readback | 8192 | 96 |
| total readback bytes | ~2.0 MB | ~3.0 MB |
| program count | 1× | **128× more** |

Horizontal adoption **increases** both program count (128×,
dispatch-bound → the binding cost) and total bytes (1.5×) for BitNet's
wide projections. It is the right layout only where d_out is small —
which is exactly the MVDRAM reproduction arm, already horizontal. **Do
not adopt it for the production vertical server.**

## The better answer: a per-segment-popcount readback mode

The `pop` step already reduces each 8 KB row to 2048 six-bit popcounts
(`segment_popcount`: `out[s] = popcount(row_segment_s)`, each ∈ [0,32]).
Move that reduction **into the readback datapath** — a third readback
mode alongside READ and DIFF-accum — draining the 2048 packed 6-bit
counts (1536 B) instead of the raw 8 KB row. This:

- **keeps the vertical layout and the whole production server unchanged**
  — no model-layout adoption, no program-count change;
- is **byte-exact** — it is literally the operation the host does now
  (prototype: identical over 200 random rows);
- collapses the compute readback **8192 → 1536 B (5.3×)** and
  **eliminates the host `pop` term** (the FPGA did it);
- projects per-program **5.9 → 3.18 ms (1.85×)** → **~25.6 s/tok**
  BitNet upper-bound if the profile holds model-wide (and stacks with
  the existing dual-DIMM/fused/single-track ladder).

The periodic MM3D byte-verify (`mm3d-verify`, which must see raw bytes
because two 32-bit values can share a popcount) stays in full READ mode
— exactly the lane2 "keep essential reads, collapse the rest" split.

### HDL sketch (extends `rtl/readback_engine_build6.v`)

Mode select gains a third state SEG_POP (control byte, like DIFF's
0x40). In SEG_POP the engine does **not** accumulate across beats: each
512-bit beat carries 16 output segments; popcount each 32-bit lane
(`popcount32 = Σ 8× pop_count4`, the 0xE-fixed nibble popcounter is
already in the datapath), pack the sixteen 6-bit results into 96 bits,
and stream 96 bits/beat → 128 beats = 1536 B/program. No `buffer_space`
subtlety like DIFF (this drains a fixed 1536 B per read, credited
normally). READ and DIFF-accum modes bit-identical to build-6.

Server change: `receiveData(1536)` + unpack (replaces `segment_popcount`)
on the compute path, gated behind an env flag (default off → build-6
behavior). Validation: Verilator (per-segment drain == software
`segment_popcount` on captured rows), then a full-model A/B (byte-
identical text, lower wall) — the same gate every production change
gets.

## Recommendation

1. Build **SEG_POP** into the next readback-engine bitstream (it composes
   with build-6; the DIFF-accum path is untouched, so the lane2 Road-B
   arm keeps working). This is the production readback collapse, no
   layout adoption.
2. Horizontal layout stays the MVDRAM-reproduction arm's shape only.
3. `seq_engine` (command issue) sequences after this, once `recv` is no
   longer the top term (`rtl/SEQ_ENGINE.md`).

Prototype + accounting: `rtl/seg_pop_prototype.py`
(exactness proof, byte/wall/program tables).
