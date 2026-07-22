# M3: coset-broadcast operand fan-out — the wcol killer

Roadmap #6. After V2_PACK, the production per-request handler is
~6.4 ms wcol + ~9.3 ms exec + ~24.7 ms recv. The recv term is attacked by
SEG_POP (bytes) and ACCUM_XBP (wakes). `wcol` — host-marshalled
per-column weight writes into scratch rows — is the second term, and it
is exactly what the co-activation lattice was built to eliminate. This is
our analog of MVDRAM's horizontal layout: use the substrate's own cheap
quasi-horizontal movement (the lattice) instead of moving operands
through the host.

## The mechanism we already proved

`LATTICE_ADDRESSING_2026_07.md`: a `doubleACT` between two members of a
calibrated tuple at k-generator distance deposits the sensed row's data
into **exactly** the 2^k sub-coset — a targeted in-DRAM broadcast, zero
external leak, t₁₂ ≥ 10. And RowClone-into-tuple is corruption-free by
construction when the (src,dst) offset contains no tuple generator
(safe-load, 20/20 clean). Measured: **1 coset doubleACT = 4.98× fewer
SoftMC instructions, 2.81× wall** vs per-column broadcast; persistent-W +
1 clone/token = **2.6× per-MAJ, bit-exact**.

## The wcol problem in the V2 path

Each V2 round writes 4 banks' scratch rows by `per_column_write_row` —
3 chunk-programs of ~1.5K instructions each (43/43/42 columns × 16 LDWD +
WRITE). That is the entire wcol term. The weight for a given (output,
plane) is the SAME 2048-bit mask written fresh every round it recurs.

## Design: broadcast-load the scratch operand

Replace the per-column write of a scratch row with a **lattice deposit**:
1. Hold the weight mask in a *resident* backup row at a safe offset (the
   persistent-W pattern — already in the server for LOAD-mode handles).
2. At GeMV time, one `doubleACT(t₁₂≥10)` between the backup row and the
   tuple deposits it into the target sub-coset — the scratch row the MAJ3
   body then consumes — in ~tens of instructions instead of ~1.5K.
3. Pool-layout co-design: the scratch row and the backup row must sit at
   a safe offset (no tuple generator in the (src,dst) XOR) so the deposit
   is corruption-free by construction, AND the deposit's 2^k sub-coset
   must not overlap other live operands. This is the new constraint the
   allocator must satisfy — the natural extension of the existing
   cloneok pool screen.

## Why it also unlocks V2-path packing

`PIM_PACK_ROUNDS` is MM3D-only today because the V2 scratch path needs
write-then-use *locality* (batching writes rounds ahead broke q_proj,
2026-05-04). Broadcast loading changes the locality story: the deposit is
a single cheap op immediately before use, so a broadcast-loaded V2 path
can pack rounds the way MM3D does — folding the wcol reduction and the
round-packing (LEVERS 3d) into one.

## Relationship to the residency finding (2026-07-22)

Backup-resident weights are the LOAD-mode path, which spills for the full
model (only 3 screened subarrays × ≤16 banks of capacity — the residency
ceiling). Broadcast-load does NOT need the weight to be tuple-resident,
only backup-row-resident at a safe offset, which is cheaper on pool
budget than a full LOAD handle. So it partially sidesteps the residency
ceiling: a weight can live in one backup row and be broadcast into the
compute tuple per use, rather than occupying a calibrated tuple slot.

## Gate

Software-only (no bitstream): a `PIM_BCAST_LOAD` path in the V2 emitter
that replaces `per_column_write_row` with a resident-backup + coset
`doubleACT`. Validate: bit-exact scratch content vs per-column write on
the screened tuple (the `test_sublattice_bcast` machinery), then
layer-0-exact, then full-model token-identity, then the wcol/handler
delta. Design-next; the pool-layout allocator change is the real work.

## First gate: PASSED (2026-07-22, both dies, zero exceptions)

Tool `app/test_m3_scratch_ab.cpp` (`m3-scratch-ab-exe`); raw logs
`docs/data/m3/`. Run on production geometry (bender 2 / bank 1 / s72
pool, cross-checked bender 0 / bank 1 / s77 pool), build-8b image
(the primitive is bitstream-independent), `BITSTREAM_IMEM=8192`.

| phase | b2 | b0 |
|---|---|---|
| k=1 deposit, 20 trials, t=(10,2) | 20/20 byte-exact, src intact, 0 leak | 20/20, 0 leak |
| k=2 fan-out (1 op → 3 targets), 20 trials | 20/20 all targets exact | 20/20 |
| timing (10,1)(10,2)(30,1)(30,2) | all exact | all exact |
| wcol 3-chunk (4519 insts, 3 programs) | 0.176 ms/load | 0.259 ms/load |
| coset deposit (17 insts, 1 program) | 0.044 ms/load | 0.066 ms/load |
| ratio | **265.8× insts, 4.02× wall** | **265.8× insts, 3.94× wall** |

This is the first deposit validation entirely **outside calibrated
tuples** — source and targets are production pool / shadow rows, the
fired set predicted by the selection law held in every trial, and the
content survived 100 back-to-back re-executions.

**Design finding — the legacy pool is M3-hostile by construction.** The
production pool is an *independent set over the coupling conflict
graph*: it contains no two rows at spread offsets (that is what made
independent per-column writes safe), i.e. it deliberately excludes
exactly the rows M3 deposits into. The M3 shape is therefore SRC = a
screened pool row (resident weight), DST(s) = rows in the pool's
*coupled shadow* — in-window, non-pool, non-tuple rows at law-unit
offsets. The coupling the legacy screen avoided is the load channel.
The allocator pairs each resident row with its shadow coset; a one-time
byte-verify of candidate shadow rows is the margin screen
(`CALIBRATION_TRANSFER.md` pattern). Empirically the shadow rows used
here were byte-perfect across all trials on both dies.

The wall ratio ≈ the program-count ratio (3→1): per-execute round-trips
bind, as everywhere else (`METHOD_MVDRAM_LENS.md`). Under Rung-1
streaming the 265.8× instruction cut is what survives — wcol becomes
DDR-bus time. Next gate: server `PIM_BCAST_LOAD` in the V2 emitter +
the shadow-pair allocator, sequenced after the Rung-1 producer loop
lands (one invasive server change at a time).

## Gate 2 (2026-07-22, same day): the four server-integration questions

Tool `app/test_m3_gate2.cpp`; raw logs `docs/data/m3/gate2_*`.

1. **Shadow supply census** — 801 (b2/bank1/s72) / 694 (b0/bank1/s77)
   law-valid k=1 pairs across 13 single-unit offsets; 91.5% / 73.6% of
   pool sources have ≥1 shadow. **Every geometrically-available offset
   class deposits byte-exact on both dies** (b0 including d=256; bit-9
   excluded pending per-position characterization). k=3 all-clear
   cosets exist (1 b2 / 7 b0); k=4 none in these windows.
2. **Source retention (aref off)** — first flip ~30 s (b2) / ~120 s
   (b0); `dst_mism == src_mism` at every mark, i.e. the deposit adds
   ZERO error of its own. Consequence: enroll deposit sources in the
   existing MM3D-entry ACT-refresh windows (sub-second cadence in
   production) — no new mechanism.
3. **Deposit chaining** — SRC –(10,2)→ DST –(30,1)→ PROBE clean 10/10
   on both dies: deposit targets are valid RowClone sources, validating
   the scratch→Rfirst consumption hop at the row level.
4. **Deposit burst — M3 pays BEFORE streaming**: 32 scratch loads in
   ONE program (358 insts) vs 96 pcwrite programs (144,608 insts) =
   **114.4× (b2) / 105.3× (b0) wall, all targets byte-exact**
   (0.0026 ms/load). The wcol term collapses today; open question for
   the server gate is deposit-ahead-of-use freshness across a request's
   MAJ3 activity (the 2026-05-04 rounds-ahead lesson) — layer-0 /
   full-model gates answer it.

Strategic note: with round-trip-bound now triple-confirmed (SEG_POP /
ACCUM_XBP / M3 gate 1), Rung-1 streaming inverts the regime — bus time
(= instructions) becomes the wall, making the 265.8×/403.9× instruction
cuts the top post-streaming lever; the two compose.
