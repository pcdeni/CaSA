# Build-8 design: the cross-bit-plane accumulator (ACCUM_XBP)

The step-5 partition move from `METHOD_MVDRAM_LENS.md`: the host still
performs the place-value sum y_g[s] += weight_b · pc_b[s] across
bit-plane programs — which forces one c2h drain (and one ~190 µs
interrupt wake; poll_mode measured broken on this driver build) per
bit-plane program. Fold the sum into the readback engine and a group's
8 bit-plane drains become **one**.

## What the engine adds (on top of build-7)

- **Accumulator**: 2048 × signed 32-bit BRAM (one slot per output
  segment). On each accumulated read beat, `acc[s] += w · pc_seg[s]`
  for the beat's 16 segments — `pc_seg` is the same `pc_out_l4` tap
  SEG_POP already uses; the multiplier is a shift (weights are ±2^i,
  encode w as sign + 3-bit shift).
- **Weight register**: set OUT-OF-BAND via a new SET word
  (`SET_ACCUM_WEIGHT`, control byte + an immediate in the word's free
  payload bytes — the frontend control-word format has them). Same
  idempotent decode pattern as the build-4/7 SETs. The host sets it
  once per bit-plane program, between programs: h2c-only, no wake.
- **Flush**: a `FLUSH_ACCUM` SET word drains the 2048 int32s
  (8 KB, READ-style framing + trailer) and zeroes the accumulator.
- **Mode**: a fourth readback state ACCUM_XBP alongside
  READ/DIFF/SEG_POP; reads in this mode are consumed into the
  accumulator (buffer_space credited per consumed beat, the build-6/7
  conservation pattern — nothing new to invent).

## Host flow (V2GS single-track, per group)

    for bp in 0..n_bitplanes-1:
        SET_ACCUM_WEIGHT(bitplane_factor[bp])      # h2c word, no wake
        execute(4-bank bodies for this plane)      # reads accumulate
    FLUSH_ACCUM; receiveData(8192)                 # ONE drain per group

## Integration boundary (the honest constraint)

The accumulator applies ONE latched weight to every read in a program's
window. That fits the production layout only when every read a program
issues shares that weight — which holds in exactly one case, and it is
the case that matters most:

- **Single-track (Bonsai V2S), K=1 — CLEAN FIT.** One program per
  bitplane; its M bank-rows are M different input *chunks* at that
  bitplane, all sign +, so they accumulate at the plane's single weight
  and — because different chunks contribute additively to the same
  output — the per-segment accumulator sum is exactly right. This is the
  current production config, so ACCUM_XBP lands where it is needed.
- **Dual-track (BitNet) — NOT a drop-in.** A program's bank-rows mix pos
  and neg units (different sign → different weight). ACCUM_XBP would
  need the pos and neg units split into separate programs (each with its
  own sign in the weight), or a per-read weight sequence (more RTL).
- **K>1 — excluded.** K>1 packs multiple bitplanes into one program;
  their weights differ, so ACCUM_XBP requires K=1. (K=1 is already the
  production cadence; K>1 was measured a net loss anyway — `ROADB` §7.)

So the production integration is: `PIM_ACCUM_XBP=1` on the single-track
K=1 path, set the weight per bitplane program, drain once per group. The
server change is in the same three readout sites SEG_POP touched, gated
so dual-track/K>1 fall back to SEG_POP. Write it AFTER silicon validates
the mode (`accxbp-hw-exe`) — not before.

Round-trips per group: 8 execs + **1 drain** (today: 8 execs + 8
drains). recv wakes ÷8; bytes 8 KB vs 8 × 2048 B (2×). Single-track
first: sign is uniformly +1, so the weight table is just the 8
bitplane factors. Dual-track needs per-unit sign — deliverable 2
(either a signed weight per SET, which the format already carries, or
the host splits pos/neg groups).

## Why this beats the alternatives considered

- *In-band weights* (snooping LDWD or a new SoftMC instruction) would
  let multiple planes share one program, but ddr_wdata must stay 0 as
  the popcount reference, and a new fabric instruction touches the
  decode pipeline — high risk for one fewer exec.
- *Per-plane accumulator banks* (8 × 2048 × 32b, select in-band)
  drains 8 vectors again — no wake saving.
- *poll_mode* at the driver: measured EIO on this build (see
  `ROADMAP.md`); the fabric-side cut does not depend on driver
  behavior.

## Budgets and gates

- BRAM: 2048 × 32b = 8 KB — negligible on the VU9P.
- Timing: one shift-add per segment per beat, 16 lanes — same
  pipeline depth class as the popcount tree feeding it.
- Verification: the build-7 discipline verbatim — extend the Verilator
  TB (accumulate-vs-software-model over random weight sequences ×
  random rows, buffer_space conservation, flush framing, mode
  transitions), require an identical failure-set diff vs build-7,
  then the silicon tool, then `PIM_ACCUM_XBP=1` in the server with
  layer-0-exact → full-model token-identity.
- Trailer magic increments (0xDBC0DE06); the SET word takes the next
  free frontend control bit.

## Where it lands

Handler today ≈ wcol 6.4 + exec 9.3 + recv 24.5 ms per 16-group
request (post-V2_PACK). This lever attacks the recv term's *wake
count* directly: 128 → 16 wakes/request. If wakes dominate recv as
measured, recv → ~6–8 ms ⇒ handler ~43 → ~25 ms ⇒ roughly **1.6–1.7×
wall** — and it composes with, rather than competes against, the
Rung-1 streaming fetch (`CONTROLLER_NATIVE.md`), which amortizes the
*exec* round-trips the same way.
