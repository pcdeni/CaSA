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
