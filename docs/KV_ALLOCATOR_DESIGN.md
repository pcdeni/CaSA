# KV PAGE ALLOCATOR — design spec (Task #70 step 1, 2026-08-04)

**Architecture is settled, not a menu** (USER RULING 2026-08-04, verbatim): *"KV
cache STORED in DRAM COMPUTED IN RTL."* Layers (i) KV-as-storage + (ii) fabric
attention ARE the design; (iii) KV-compute-in-PuD stays CLOSED-INFEASIBLE
(attention is not ternary; softmax has no MAJ/popcount mapping —
`bus_bound_ladder` §4a, LEVERS #70). This doc specifies **layer (i)**: the
subarray-aware KV page allocator that parks KV in card DRAM and manages
copy/fork/defrag with RowClone. Layer (ii) is #72 (a fabric-attention build);
this doc states only what the allocator must *guarantee* to that consumer (§5).

Scope of this pass: **design + a staged silicon A/B** (`KV_ALLOCATOR_AB_PLAN.md`),
no silicon dependency for the design itself. Every quantitative claim below
carries an evidence tag: `MEASURED` (silicon/host) · `SIM` (casa_sched, a
ceiling) · `ARITHMETIC` (derived) · `RECORDED-IDEA` (design, untested) ·
`UNKNOWN` (flagged §7).

---

## 0. The four hardware quanta everything hangs on (MEASURED)

| quantum | size | what it bounds | source |
|---|---:|---|---|
| **row** | 8192 B (8 KiB) | the atomic read/write/RowClone unit; 65536 bits | tower_bringup logs (`bit_match 65536/65536`); MENTAL_MODEL §1.8 |
| **sense-amp SEGMENT (= subarray)** | **640 rows** = 5.0 MiB | **RowClone / doubleACT locality** — a copy's src+dst MUST share this | MM §2.3 ("640 rows = sense-amp SEGMENT, RowClone/tuple locality"); LEVERS #65b |
| **predecoder BLOCK** | **1024 rows** = 8.0 MiB | **co-activation scope** — MAJ/broadcast spread reaches within this | MM §2.3 ("1024 rows = predecoder block, co-activation scope") |
| **bank / rank / DIMM** | 65536 rows/bank = 0.5 GiB; 16 banks = **8 GiB/rank** = 1,048,576 rows | capacity ceiling; ≈102 segments/bank, ≈1638 segments/rank | capacity.py (8 GiB compute-addressable/rank); DDR4 8Gb x8 geometry |

**The one law that shapes the allocator:** RowClone is **intra-subarray only**
(both ACTs open rows sharing one sense-amp segment — MM §2.1/§2.3). So a KV copy
that must be **free** (zero bus, in-DRAM) requires src and dst **inside the same
640-row segment**. Cross-segment / cross-bank / cross-DIMM copies fall back to
**read-out + write-in over the bus** at **≈8.5 GB/s per storage channel**
(MEASURED, casa_sched effective-BW, `bus_bound_ladder` §a′) — still zero PCIe,
still cheap, but not free and not RowClone.

---

## 1. Page geometry (design question 1)

### 1.1 A KV *stream* is the natural streaming object
Fabric attention (#72) for layer ℓ needs, per KV-head-group, the run
`K[ℓ, 0:pos, :]` and `V[ℓ, 0:pos, :]` delivered **in position order**. So the
storage object that reads at line rate is a **(layer, K|V) stream**: a run of
rows where *ascending row address == ascending token position* for one fixed
(layer, K-or-V, head-group). One row holds:

| model | n_kv·head_dim·2B per token | tokens per 8 KiB row | rows per token per (layer,K|V) |
|---|---:|---:|---:|
| Llama2-7B (MHA 32/32) | 32·128·2 = **8192 B** | **1.00** | 1 |
| Llama2-13B (MHA 40/40) | 40·128·2 = **10240 B** | 0.80 | 1.25 |
| Llama3-8B (GQA 32/8) | 8·128·2 = **2048 B** | **4.00** | 0.25 |
| Phi-4 (GQA 40/10) | 10·128·2 = **2560 B** | 3.20 | 0.3125 |

(dims from the project's own `mainstream_faithful_2026_08_02/capacity.py`,
MEASURED against the gguf metadata; fp16 shown.) Note Llama2-7B lands **exactly
one token per row** — the KV layout is naturally row-quantized.

### 1.2 A KV *page* = a segment-aligned sub-run of one stream
Define the allocation unit:

> **A KV page = P contiguous rows of ONE (layer, K|V) stream, where P divides
> 640 and a page NEVER crosses a 640-row segment boundary.**

Default **P = 128 rows = 1.0 MiB** (5 pages / segment) — tunable, carried as a
config constant (`PIM_KV_PAGE_ROWS`), never hardcoded (design law, LEVERS #65b /
L24). Alternatives 64 (10/seg) or the whole 640 (1 page = 1 segment, coarsest)
are legal. The **segment is the copy-on-write quantum**; the page is the finer
allocation grain inside it.

Why divide 640 and not 1024: the copyable domain is the **sense-amp segment
(640)**, not the predecoder block (1024). Aligning to 640 guarantees every page
is RowClone-copyable as a unit; aligning to 1024 would let a page straddle two
subarrays and break the free-copy property. (The 1024 block only matters when
PuD compute shares the die — §2.)

### 1.3 The allocator invariant (derived from RowClone-intra-subarray)

> **INVARIANT KV-1 (co-residency for free fork):** any two pages that may
> participate in a **copy-on-fork / prefix-share materialization** must be
> allocated in the **same 640-row segment**. The allocator reserves a page's
> copy-on-write destination slot *within its own segment*; a fork that must
> land outside the segment is a bus read+write, not a RowClone.

> **INVARIANT KV-2 (page never straddles a segment):** `page_base mod 640 +
> P ≤ 640`. Pages tile segments; a stream longer than one segment is a *list*
> of segment-aligned pages, and the segment boundary is where "free copy" ends.

Consequence for prefix-share (the G4 headline lever): the shared prompt prefix
occupies **whole segments** that both sequences reference **by page-table
pointer — ZERO copy** (no RowClone needed at all for fully-shared segments).
Only the **one boundary segment** straddling the fork point needs a private copy
for the diverging sequence, and that copy is **intra-segment RowClone = free,
≥99.98% reliable** (SiMRA Multi-RowCopy, LEVERS #70; RowClone (30,1) PERFECT on
both hynix dies and t23=1..3 on Micron — MM §2.1, dimm_population). This is the
whole "copy is free" claim, made precise: *segment-granular sharing is free by
reference; the single fork-boundary segment is free by RowClone; only a
cross-segment relocation costs bus bytes.*

---

## 2. Placement law — do storage pages need spread-safe placement? (design question 2)

**Precise answer: it depends on whether any PuD (doubleACT) op ever touches the
page's predecoder block. Plain parked bytes need nothing; the RowClone *copy
operation itself* is a doubleACT and carries collateral.** Three regimes,
mapped to L24 roles:

| regime | who / where | PuD compute in the block? | RowClone issued? | placement constraint |
|---|---|---|---|---|
| **R1 pure park** | D1/D3 (Micron, MAJ3-dead) or any bank in `STORAGE` state, KV read-only after write | none | none (write-once, read by bus) | **NONE.** Any row is safe. Pack densely. Retention (§3) is the only concern. |
| **R2 storage + fork** | same banks, but KV forks/defrags via RowClone | none | yes | **safe-load offset only.** The copy's src/dst pair must be chosen so the co-activation coset of `local(src)⊕local(dst)` contains **no live KV page** (safe-load, MM §2.4, n=20/20 clean). |
| **R3 shared with compute** | KV parked in spare capacity of a **compute** DIMM (D0/D2 hynix) while that die runs MAJ/broadcast | yes, in the block | maybe | **block-disjoint.** KV pages must occupy **predecoder blocks (1024-row) disjoint** from every active compute tuple, because a concurrent MAJ/broadcast deposits into its coset family (WRITE-direction spread, MM §2.3/§2.7). Simplest: give KV its own banks. |

**Why even a "storage" RowClone has collateral (the subtle part):** a RowClone
is a `doubleACT(30,1)`; t23=1 sits **inside** the ≤3-NOP co-activation window
(MEASURED boundary C64, `kubo_xorspread_repro_2026_08_03`: the src-into-src⊕8
deposit is PRESENT at ≤3 NOP slots, ABSENT at ≥4). So on a **hynix** die the
copy also writes the source pattern into the coset members of
`local(src)⊕local(dst)` (the "free XOR-8 clone", `simra_xor8_spread`). The
**source row survives** (it is the copy source); the **collateral is
unintended writes to coset rows** — those must be dead/reserved. That is exactly
what safe-load placement guarantees (offsets with no generator-sum subset are
tuple-clean by construction, MM §2.4).

- On **Micron D1/D3**: the selection law does NOT hold and only **k=1 deposits
  are clean** (MM §2.8) — a single-generator RowClone works, its **one**
  `⊕generator` collateral partner must land on a dead row; multi-bit offsets are
  dirty and must not be used for KV forks. The clean-offset menu for Micron is
  **not yet enumerated** (hynix's is) — UNKNOWN §7, screen in the A/B.

**Bottom line:** the **primary KV home is R1/R2 on D1/D3** (storage role, L24),
where static parking needs **zero** placement care and forks need only a
safe-load offset. The spread/self-pollution machinery that governs compute
tuples does **not** burden static KV; it re-enters *only* through the RowClone
copy, and only as a src/dst offset choice — never as a per-cell margin screen
(no MAJ, no compute calibration — LEVERS #70).

---

## 3. Retention — sizing the KV refresh on a refresh-less platform (design question 3)

**The platform has NO auto-refresh** (MM §2.5: the refresh API has zero callers;
`ddr_ref` never fires; the tPRDI conveyor issues bank0/row0 reads only). A KV row
gets charge topped up **only** when it is ACT'd. Measured retention (hynix, drift
ladder seed 502, `drift_ladder_2026_08_01`, MEASURED):

- Age-CDF of mismatched lanes/512: 0.1 s→0, **29 s→34**, 101 s→345–468, 211 s→502.
- First-flip onset ≈ **30 s (bender 2) / 120 s (bender 0)** (M3-era aref-off, MEASURED).
- Victim set DETERMINISTIC (fixed weakest cells, Jaccard 0.91–0.99); damage tracks
  **un-refreshed elapsed time, not activity**.
- **"ACT = refresh restores CHARGE, not CONTENT"** — an ACT on an already-drifted
  cell re-latches the *drifted* value; only a **WRITE before the flip** preserves
  data. `PIM_DESC_REWRITE=1` → 512/512 at every dose, **~0.2% of wall** (MM §2.5).

### 3.1 The key structural observation (ARITHMETIC, to be confirmed in the A/B)
**Active generation *can be* self-refreshing for charge — but only once the wall
is fast enough.** Every decode step, attention reads the *entire* KV cache (all
past positions), so every KV row is ACT'd **once per token**; the inter-read gap
for a given row = **one full decode step**. This tops up charge **iff that gap <
onset (~30 s)**. HONEST CAVEAT: at *today's* slow per-token times this does NOT
hold — the mainstream streamed-weight forward is 400–900 s for a prefill+token
(`flagships_complete`), far longer than the 30 s onset, so per-token reads would
NOT beat the flip today; the self-refresh becomes free only as the wall descends
toward the projected sub-second floors. And because ACT restores charge not
content (MM §2.5), it is a *pre-flip* top-up only — prevents onset, never corrects
an already-flipped cell. So the regime split is **rate-dependent**:

- **Actively-attended KV, once per-token gap < onset** (the fast-floor regime):
  refreshed by the attention reads themselves; **no extra refresh transport
  needed**. (Confirm on the real per-row read gap — A/B arm 2c.)
- **Actively-attended KV at today's slow wall** (per-token gap > onset): NOT self-
  safe yet → needs the same explicit scrub as idle KV until the wall descends.
- **Idle / parked KV** (a shared prompt-prefix cached between requests; a paused
  session; a beam that stops being read): NOT read every token → **must be scrubbed
  explicitly** before onset, at every wall speed.

### 3.2 Which refresh for storage-only rows
Two candidates:

1. **Preventive charge top-up (RowClone-to-self / periodic ACT), period < onset.**
   Zero bus (in-DRAM). Restores charge **around whatever content is currently
   latched** → correct **only if run before any cell flips**. A single missed
   deadline loses data (no ECC on KV bytes). Cheap but fragile.
2. **Corrective read-rewrite scrub (read the row, write it back full-swing),
   period < onset.** Costs one bus read + one write. Because the read happens
   *before* the flip, it returns the correct value and the write re-establishes
   full charge margin — content-preserving **and** margin-restoring. Robust.

**Recommendation: read-rewrite scrub for idle KV, at period ≤ ½·onset (≤ ~15 s on
b2), from the page's own current (pre-flip) content.** Preventive ACT-only is the
fallback where bus budget is tight, but only inside a proven-safe period. (The
`REWRITE=1` result is the anchor that a pre-flip rewrite fully restores — MM §2.5.)

### 3.3 Refresh transport budget (ARITHMETIC)
Only *idle* KV needs scrubbing; size the worst case = the whole resident KV
scrubbed every 15 s.

| model, ctx (fp16) | resident KV | read-rewrite bytes / 15 s | fraction of one 8.5 GB/s storage channel |
|---|---:|---:|---:|
| Llama3-8B 4k | 0.50 GiB | 2 × 0.50 GiB / 15 s = 71 MB/s | **0.8 %** |
| Llama2-7B 8k | 4.0 GiB | 2 × 4.0 GiB / 15 s = 573 MB/s | **6.7 %** |
| Llama2-13B 8k | 6.25 GiB | 2 × 6.25 GiB / 15 s = 895 MB/s | **10.5 %** |

If done as **RowClone-to-self (preventive)** instead, it is **zero bus** (internal
ACT bandwidth only) — 6.25 GiB / 15 s of intra-subarray ACTs is trivially within a
channel's ACT budget. Either way the retention tax is small (consistent with the
~0.2%-of-wall `REWRITE=1` anchor). **Retention is a solved-shape problem for KV, not
a blocker** — but the scrub period must be set from a **D1/D3 (Micron) retention
bracket**, which is UNKNOWN (§7) — the drift ladder is hynix.

---

## 4. Capacity sizing table (design question 4)

**KV bytes per token** = `2 (K+V) · n_layer · n_kv_head · head_dim · dtype_bytes`
(the exact formula in `capacity.py`). Dims MEASURED from gguf metadata:

| model | L | n_kv | head_dim | **fp16 KV/token** | int8 KV/token |
|---|---:|---:|---:|---:|---:|
| Llama2-7B (MHA) | 32 | 32 | 128 | **512.0 KiB** | 256 KiB |
| Llama2-13B (MHA) | 40 | 40 | 128 | **800.0 KiB** | 400 KiB |
| Llama3-8B (GQA) | 32 | 8 | 128 | **128.0 KiB** | 64 KiB |
| Phi-4 (GQA) | 40 | 10 | 128 | **200.0 KiB** | 100 KiB |

**KV footprint × context (fp16), and where it fits** (1 bank = 0.5 GiB = 65,536
rows; 1 storage DIMM/rank = 8 GiB = 1,048,576 rows; D1+D3 = 16 GiB):

| model | 2k | 4k | 8k | 4k in rows / banks | fits… |
|---|---:|---:|---:|---:|---|
| **Llama3-8B** | 0.25 GiB | **0.50** | 1.0 GiB | 65,536 rows = **1.0 bank** | **one bank** (≤4k); ≤2 banks at 8k |
| **Phi-4** | 0.39 GiB | **0.78** | 1.56 GiB | 102,400 rows = **1.56 banks** | ~2 banks (4k); ≤4 banks (8k) — **≪ one DIMM** |
| **Llama2-7B** | 1.0 GiB | **2.0** | 4.0 GiB | 262,144 rows = **4.0 banks** | **one DIMM** (4k = ¼ rank, 8k = ½ rank) |
| **Llama2-13B** | 1.56 GiB | **3.12** | 6.25 GiB | 409,600 rows = **6.25 banks** | **one DIMM** (8k = 6.25/8 GiB, tight +22% headroom; prefer D1+D3 split or int8) |

Cross-check against the project's own `capacity.py` / flagship placement table:
matches exactly (8B KV 0.50, Phi-4 0.78, 13B 3.12 at 4k). Its verdict stands:
weights on D2 rank-0; **13B's 4k KV is the term that overruns a single 8 GiB
compute rank → KV → storage DIMM (D1/D3)** — which is precisely the KV-in-DRAM
storage tier this allocator manages.

**Readings:**
- **One bank** holds a full-context small-KV model: Llama3-8B ≤4k, Phi-4 ≤2k.
- **One storage DIMM (8 GiB)** holds **every** flagship at ≤8k (13B 8k is the only
  tight case at 6.25/8 GiB). int8 KV halves all of it → all four fit one DIMM with
  wide margin at 8k+.
- **D1+D3 (16 GiB)** hold all four at all listed contexts **with weight-parking
  room to spare** — the conveyor working-set home (#67).
- **Assumption (UNKNOWN §7):** D1/D3 taken as 8 GiB single-rank each. If either is
  dual-rank (like D2's 16 GiB), double its column. Design is capacity-agnostic;
  only the headroom numbers move.

---

## 5. The interface — mapping onto the #65 config schema + what #72 needs (design question 5)

### 5.1 KV banks are already representable in the #65 grid schema
`bank16_config_2026_08_04` (#65, DONE host-only-gated) makes bank set, per-bank
window, per-bank **state**, and DIMM **role** host-config, runtime-mutable via
`MAGIC_CONFIG` (no restart). KV maps straight onto it:

- **State** = `BankState.STORAGE` (enum already defined; `pim_grid_config.py`).
  A KV bank is one held in STORAGE. Allocating/reclaiming a KV bank =
  `CFG_SET_STATE` → **metadata-only, no stop-the-world** for idle↔idle
  transitions (MEASURED gate B, `bank16` RESULT) — exactly the conveyor contract.
- **Role** = `DimmRole.STORAGE` for D1/D3 (Micron); `default_bitnet(storage_dimms
  =(1,3))` already seeds them. KV's natural home is these two DIMMs.
- **Per-bank window** = `BankSpec.win_start/win_end` (already per-bank). A KV bank's
  window is its KV region; distinct from a compute bank's screened pool window. For
  R1/R2 pure-storage KV the window may be the **whole bank** (no screened-pool
  restriction — no MAJ), a key simplification the schema already permits.
- **Transport** = the existing `MAGIC_CONFIG` wire (`CFG_QUERY/RECONFIG/SET_STATE`,
  `PimServer.set_bank_state/reconfigure/query_config`). No new server compute path:
  KV needs only **RowClone + read + write** addressed by absolute row, which the
  server already has (LOAD/RowClone primitives).

### 5.2 What the allocator adds (host-side, additive — specified, NOT built here)
A new host module `pim_kv_alloc.py`, sibling to `pim_grid_config.py`, pure-data +
testable card-free:

- **KV page table**: `(seq_id, layer, kv∈{K,V}, pos_block) → (dimm, bank, segment,
  page_base_row, P)`. Reference-counted per page (prefix sharing = multiple
  sequences pointing at one physical page; refcount 0 → free).
- **Segment free-list per (dimm, bank)** with the CoW-twin reservation (INVARIANT
  KV-1): allocating a page reserves its in-segment fork destination.
- **Fork op**: `fork(seq, at_pos)` → shared full pages get a pointer bump (refcount++,
  zero copy); the one boundary segment is RowClone-copied intra-segment (safe-load
  offset from the offset menu; §2 R2).
- **Defrag op**: coalesce free pages by intra-segment RowClone (free) where possible;
  cross-segment relocation only when a segment can't be compacted internally (bus
  read+write, 8.5 GB/s).
- **Scrub scheduler** (§3): per-page last-touch timestamp; scrub any idle page whose
  age nears onset. Composes with #67's conveyor (same "move before needed" cadence,
  #64 paced-copy is the mover).

This is **#67-adjacent host orchestration** — one `PimServer` drives one DIMM
today, so cross-DIMM KV (weights on D2, KV on D1/D3) is python-orchestrator work
across processes, exactly as `bank16` RESULT scopes for #67.

### 5.3 What layer (ii) fabric attention (#72) needs the layout to GUARANTEE
The allocator must promise the fabric consumer (the RTL that does Q·Kᵀ / softmax·V
at line rate, readback-engine precedent — SEG_POP/DIFF stream at line rate,
MEASURED, `roadb_build7`/`lane2`):

1. **Streaming order** — within a (layer, K|V, head-group) stream, **ascending
   physical row address == ascending token position**. The fabric walks a row range
   and gets positions in order; **no gather, no host indirection**.
2. **Segment alignment** — every page is 640-aligned and never straddles a segment
   (INVARIANT KV-2), so a page read is a **contiguous line-rate burst** with no
   mid-burst discontinuity. A stream longer than a segment is a descriptor *list* of
   segment-aligned runs (the descriptor-walker precedent, build-45/#67 handles lists).
3. **Base + stride per (layer, K|V)** — a stable `(base_row, P, n_pages)` tuple so a
   fabric descriptor can walk the whole stream without a per-page host round trip.
4. **Channel co-residency of K and V for a layer** — K[ℓ] and V[ℓ] on the **same
   storage channel** so the attention pipeline reads both without a cross-DIMM hop
   (each DIMM = its own XDMA channel + MIG, no die-to-die link — MM §1.0).
5. **No PuD in a KV block during a read** — R3 disallowed while attention streams
   (or KV kept on D1/D3 where PuD never runs) so a concurrent deposit can't perturb
   the stream mid-read.

**LAYOUT LAW (one line):** *a (layer, K|V) stream = a position-ordered,
segment-aligned list of same-channel pages; K and V of a layer co-channel;
descriptor-walkable base+stride.* Everything #72 does rides this.

---

## 6. Summary — the allocator in five invariants

- **KV-1** free fork ⇒ a page and its copy-on-write twin **co-reside in one 640-row
  segment** (RowClone is intra-subarray).
- **KV-2** a page is **segment-aligned and never straddles** a 640 boundary
  (`P | 640`, default P=128 rows = 1 MiB, config not hardcode).
- **KV-3 (placement)** pure park needs **no** placement; a RowClone fork needs a
  **safe-load offset** (coset clean); KV sharing a compute die needs **1024-block
  disjointness** from active tuples. R1/R2 on D1/D3 is the primary home.
- **KV-4 (retention)** active KV self-refreshes via attention reads (ACT/token <
  onset); **idle** KV is scrubbed by read-rewrite at period ≤ ½·onset (≤~15 s);
  budget ≤ ~10% of one 8.5 GB/s channel worst case, ~0.2% typical.
- **KV-5 (streaming)** a (layer,K|V) stream is a position-ordered, segment-aligned,
  same-channel, descriptor-walkable page list — the contract #72 consumes.

Capacity verdict: **every flagship's KV fits one storage DIMM at ≤8k** (13B 8k the
only tight case); **small-KV models (Llama3-8B, Phi-4) fit one to a few banks**;
**D1+D3 (16 GiB) is ample** for all four plus weight-parking. This removes host RAM
from the KV **capacity** wall (LEVERS #70) — it does **not** move the native-lane
token floor (a workload enabler, not a throughput lever; MM/LEVERS #70).

---

## 7. Honest unknowns (flagged, not hand-waved)

1. **D1/D3 module capacity** — assumed 8 GiB single-rank each; not measured in the
   workspace (no inventory list; dimm_population says "ask the user"). If dual-rank,
   double that DIMM's storage. Affects headroom only.
2. **RowClone source-survival** — prefix-share REQUIRES the shared source page to
   stay byte-exact after a fork clone. The XOR-spread model implies src survives (it
   is the copy source; collateral is the *other* coset rows), but the tower
   PERFECT_CLONE logs verify **dst**, not src survival. → explicit src-readback in
   the A/B.
3. **Micron (D1/D3) clean-offset menu for forks** — hynix's safe-load offsets are
   enumerated (MM §2.4); Micron's are not (selection law doesn't hold; only k=1
   clean, MM §2.8). A small offset screen on D1/D3 is needed before trusting forks
   there. → A/B screens it.
4. **Micron retention onset** — the ~30 s/120 s onsets are hynix (drift ladder). The
   scrub period must be set from a D1/D3 retention bracket. → A/B measures it.
5. **Self-refresh-by-read** — the "attention reads keep charge topped" claim needs
   confirming on the real per-row read gap (ACT restores charge only if it beats the
   flip). → A/B read-cadence arm.
6. **Fabric line-rate attention over KV** — asserted from the readback-engine /
   descriptor-walker precedent (SEG_POP/DIFF at line rate, MEASURED); the attention
   descriptor walker itself is **unbuilt** (#72). This doc guarantees the layout;
   #72 must build the consumer.
7. **RowClone-to-self semantics** — whether a degenerate self-clone (vs a genuine
   read-rewrite) restores content is unverified; recommendation defaults to
   read-rewrite for correctness. → A/B compares both.
