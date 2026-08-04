# Campaign record — July 2026 (sub-lattice broadcast → merge → reproduction)

> **Repo note.** This is the verbatim working record (`RESULT.md`) of the
> 2026-07-17 → 07-20 silicon campaign, preserved here as the primary
> citation target for the [explainer decks](explainer/) and their claim
> ledgers. It is a lab log — dense, dated, addendum-structured (27 addenda),
> written for the authors. The decks and
> [`MVDRAM_REPRODUCTION.md`](MVDRAM_REPRODUCTION.md) are the reader-facing
> synthesis; this file is the ground truth they point at, copied unedited.

---

# Sub-lattice broadcast — silicon confirmation (2026-07-17, tower)

**Claim tested** (from the SiMRA exchange, memory `lattice_as_primitive_implications` #1):
a `doubleACT` between **two members** of a calibrated open-rows tuple, at
k-generator distance, deposits the sensed-first row's data into **exactly** the
2^k sub-coset `{X ⊕ S : S ⊆ bits(X⊕Y)}` (subarray-local coords) — a *targeted*
broadcast, versus the indiscriminate all-16 broadcast used today.

**Verdict: CONFIRMED, bit-exact, on both full-PUD dies.** The address algebra
predicts silicon behaviour with zero error.

## Setup
- Tower `radarskinpc` (BCU1525, PCIe 01:00.0, Gen3×8, kernel 6.11).
- Production tuple, DIMM-2-class die, s_id 72, bank 0: Rfirst=45340,
  16 open rows, subarray base 45312. Generators (local) **{1, 2, 96, 384}**
  where 96={bit5,6}, 384={bit7,8} — a clean 2⁴ lattice.
- Tool: `test_sublattice_bcast.cpp` (this dir). Per config: write random W to
  Rfirst, 0 to the other 15 open rows, `doubleACT(t12,t23, Rfirst, partner)`,
  read back all 16, classify each row W / 0 / mixed; predict IN-coset vs
  OUT-coset from the algebra; compare.
- Run on **bender 2** and **bender 0**.

## Phase A — every partner, t=(10,2), 3 repeats
All 15 non-self partners (g = 1,2,3,96,97,98,99,384,385,386,387,480,481,482,483):
**`pred == got` for every partner × every repeat × both benders.** IN-coset rows
hold W; OUT-coset rows stay 0. Example — the "money pair" g=3 (partner 45343):
deposits into exactly `{45340,45341,45342,45343}` (one 4-row mat-group) with a
single doubleACT: `pred=WWWW000000000000 got=WWWW000000000000`.
Determinism: 3/3 repeats identical. Cross-die: bender 0 ≡ bender 2.

## Phase B — timing sweep on the money pair (g=3)
| t₁₂ | result |
|---|---|
| 0  | **nothing deposited** (got=0000…) — tRP not violated, predecoder latches clear |
| 10 | perfect `WWWW…` at t₂₃ ∈ {0,1,2} |
| 30 | perfect `WWWW…` at t₂₃ ∈ {0,1,2} |

**Design rule: t₁₂ ≥ 10 required; t₂₃ free.** The production broadcast timing
(10,2) already satisfies it.

## Phase C — external-leak probe (my addition)
After the money-pair doubleACT, read 8 rows OUTSIDE the tuple — the partial-bit
neighbors of the 96 and 384 groups (local offsets 32,64,128,256,352) and far
rows (8,512,620). **0/8 leak on both benders.** Two things proven:
1. No leak beyond the tuple — the deposit is contained in the sub-coset,
   safe for a persistent-weight pool.
2. Group generators stay **atomic** under a sub-pair: offsets 32/64/128/256
   (the individual bits inside the 96/384 groups) never open — consistent with
   the group-exclusivity finding (`xor_spread_provenance` gap #1).

---

# Part 2 — Safe-source fast loading (`test_safe_load.cpp`, same day)

**Claim tested:** tuple corruption during RowClone loading is a function of the
PAIR offset `d = local(src)⊕local(dst)`; choosing d with no generator-sum subset
(s72: avoid bit0, bit1, bits5+6 together, bits7+8 together) makes the load
tuple-clean **by construction** — reversing June's MVDRAM verdict ("blocked
deeper than source-row choice", 5/16 rows surviving a sequential tuple load).

**Verdict: CONFIRMED on both benders, bit-identical.**

- **Phase 1 — safe loads: 20/20 CLEAN.** 5 tuple targets × 4 safe offsets
  (d ∈ {4,8,16,32}): payload lands on the target row only, all 15 other tuple
  rows untouched, every time.
- **Phase 2 — unsafe controls:** d=5 → corrupts exactly {T, T⊕1} as predicted;
  d=100 → exactly {T, T⊕96}; d=389 → exactly {T, T⊕1, T⊕384, T⊕385}; d=6 →
  bit-1 point didn't fire (selection inside the envelope, known ~50% group
  behaviour). **Containment never violated** — corruption is predictable and,
  with safe offsets, avoidable without relying on selection.
- **Phase 3 — off-anchor group broadcasts:** g=3 sub-coset broadcast works
  anchored at ALL four mat-groups (not just Rfirst) — position-independent.
- **Phase 4 — the fast loader:** 4 distinct patterns → 4 external staging rows
  → 4 safe RowClones into group anchors → 4 intra-group broadcasts =
  **16/16 rows correct** (`AAAABBBBCCCCDDDD`). 8 doubleACTs replace 16 row
  loads. June's equivalent on this die class: 5/16.

**Why June failed:** its "non-shadow source selection" used the source-specific
vulnerability model (fingerprints measured against ONE fixed dst). The real
knob is the pair offset per (src, target). Consistent retro-diction: June saw
DIMM-0 milder (12/16 vs 5/16) — the s61-region generators are {4,16,64,512}
(small src offsets safe), s72-region generators {1,2,96,384} (small offsets
deadly).

---

# Part 3 — MVDRAM faithful dataflow reproduced (`test_mvdram_compute_rows_safe.cpp`)

**The headline.** June's mvdram-repro concluded the MVDRAM Fig-2 faithful
computation-rows dataflow was *"physically impossible on coupled-tuple
silicon"* — 6.1% end-to-end. With safe-source placement (same MAJ, same DAG,
only addressing changed) it runs at **99.98% end-to-end / 99.99% per-op,
54/54 operand loads landed, on benders 2 AND 0.**

| | per-op MAJ | end-to-end |
|---|---|---|
| June architecture (reproduced, b2) | 50.3% | 6.14% |
| Safe placement (b2) | 99.99% | 99.98% |
| Safe placement (b0) | 99.99% | 99.96% |

Fixes: value rows in-subarray (June clced from the neighboring subarray),
pair offset free of generators {1,384} with antichain masks, load order
i1→i2→i0, and on-silicon mask screening (only 28/77 masks manifest onto all
3 inputs — selection is real). **Answers "does this help reproduce MVDRAM":
yes — the paper's performance shape on ordinary commodity DDR4, no golden
"strict RowCopy" module needed.** Does not rehabilitate the HMA851 claim
(those had zero PUD); it's a stronger result.

# Part 4 — DIMM 1/3 MAJ3 rescue (`enumerate_rescue.py`, `test_maj3_tuple.cpp`)

**Partial — with the mechanism.** Enumeration of the April sweeps (old D1 =
bender 1, old D3 = bender 3) found clean pair-lattices with Rf-in-set ONLY at
size 4, all generators {1,2}. Harness validated: known-good s86 tuple = 1.9
bit-err/65536 (99.997%).
- Rescue tuples: **3/7 exact, ~12.5% mean bit-err** — beat the historical
  best (old D1 s24 = 0/7, 30.7%; old D3 s51 = 1/7).
- Fail the mixed votes (truth/AND/OR: 22–37%); pass only obvious majorities.
- **Reason:** {1,2} = 4 adjacent rows → no **separated** operand geometry.
  The working s86 tuple's {1,384} keeps one operand 384 rows away.
  DIMM 1/3 lack a separated generator, not clean co-activation.
- Next: enumerate {small,large}-generator tuples; if none, swap from inventory.

# Part 5 — Per-MAJ activation-update A/B (`test_actupdate_ab.cpp`)

Old = 5 per-column wrRow to act_pos; new = 1 anchor write + 1 coset doubleACT.
Bender 2, s72, 200 iters:
- A: 22,595 SoftMC insts, 0.982 ms/update
- B: 4,536 SoftMC insts, 0.349 ms/update
- **4.98× instruction reduction, 2.81× wall-clock, ~2.8× net per-MAJ** (if the
  update dominates). Conservative on-silicon confirmation of the casa_sched
  I-series activation-elimination lever.

# Part 6 — Persistent-weight throughput A/B (`test_matvec_persist_ab.cpp`)

**The per-token win for the ternary LLM.** dense-matvec-bcast re-loads W every
MAJ (0.7 ms per-column write) because Rfirst holds the result afterward. With
W resident in a backup row at a SAFE offset (Rf^8) and refreshed by one
doubleACT clone, on benders 2 & 0, 100 iters:

| mode | ms/MAJ | correct |
|---|---|---|
| A — per-column write W each MAJ | 0.487 / 0.508 | 100% |
| B — persistent + clone refresh | 0.191 / 0.193 | **100% bit-exact (0/204800)** |

**2.6× per-MAJ, bit-exact, both dies.** Composes with Part 5's activation-coset
lever (different line of the per-MAJ program) → combined ~4–5× target. Both
software-only, no bitstream change. This is the mechanism behind a persistent
weight POOL: load K weights once, refresh each active tuple by clone per token.

# Part 7 — DIMM 1/3 are MAJ3-limited (definitive)

Refined enumeration (rank by operand separation = largest generator): on the
partial dies, EVERY co-activation pair spanning ≥8 rows is a non-decomposable
(dirty) lattice — **zero** clean lattices with a generator ≥8. Their only clean
co-activation is the adjacent {1,2} scale, whose un-separated operands the
mixed (non-majority) MAJ patterns cannot resolve (Part 4). Conclusion: DIMM 1/3's "partial" status is a decoder-structure
property — they lack the separated-generator geometry MAJ3 needs — not fixable
by tuple selection. Engineering call: use for storage/non-MAJ roles, or swap
from inventory. (Contrast: full-PUD dies 0/2 decompose 704/704 wide tuples.)

## What this unlocks / what it does NOT yet prove
- **Unlocks the mechanism** for implication #1: a chosen 2^k sub-coset can be
  written with 1 wrRow (stage x into Rfirst) + 1 doubleACT, replacing k full-row
  writes — deterministically and selectively, on stock DDR4, no die change.
- **Not yet an end-to-end speedup.** To replace the 5 activation wrRows in the
  real per-MAJ program, the 5 activation slots must themselves form a sub-coset
  — i.e. geometry-aware role assignment (implication #2). That's the next step:
  re-derive the tuple layout so act-positions are a coset, then A/B the per-MAJ
  cycle count vs the current 5-wrRow path. Readout wall & orchestration overhead
  untouched (as flagged in the memo).
- SAFARI design rule cross-checked: their k-decay (99/95/83/63/32/0% for k=1..6)
  says keep constructed cosets at k≤3–4; here k≤4 within a screened tuple was
  100% — consistent, since these are already-calibrated members.

---

# Addendum (same day, tower-local) — fused per-MAJ STABLE measurement

Post-wedge-recovery continuation (cold power cycle, boot 11:30; bring-up +
RowClone smoke 100%). Goal: replace the interrupted iters=50 run with a
statistically solid fused-maj number. Method: batch-of-2-iteration processes
(the tool's empirical safe envelope, below), TERM-only guards, driver reload
+ full_reset + smoke after any stall. Logs: `fused_largeN_b2.log`,
`fused_largeN_b0.log` (this dir).

| bender | batches | A ms/MAJ | B2 ms/MAJ | A→B2 | B2 correctness |
|---|---|---|---|---|---|
| 2 | 20 (40 iters/mode) | 1.216 ± 0.353 | 0.142 ± 0.017 | **8.48× ± 1.93** (pooled 8.59×) | **0/81,920** |
| 0 | 8 (16 iters/mode)  | 1.179 ± 0.226 | 0.169 ± 0.034 | **7.15× ± 1.73** (pooled 6.97×) | **0/32,768** |

- **B2 (production-margins fused) is bit-exact at scale on both dies** —
  0/114,688 bad segments total. The provisional "~9.1×" iters=2 sample
  resolves to 8.5× (b2) / 7.2× (b0) with error bars.
- **B1 (7W/4x/4z) failure replicated at scale: 90.2% / 78.7% bad.** The
  5/5/5 vote balance is a hard design rule, not small-sample noise.
- Caveats: 2-iter process means include per-process cold-start; fixed-seed
  workload (mt19937 0xF00D) — the claim is timing/correctness stability
  across 28 independent processes, not workload diversity.

## Tool stall envelope (why batches of 2)
- iters=2: **29/29 processes clean** (incl. yesterday's runs).
- iters≥3: **2/2 stalled**, at different positions (iters=8 → A iter 3;
  iters=3 → B1 iter 1). Kernel signature both times:
  `xdma: 0-C2H2-ST stopped half-way, 21/256` (resp. `24/256`) — c2h
  starvation mid-row, probabilistic, not a deterministic iteration count.
- TERM-guarded cleanup after a stall was survivable both times (link stayed
  Gen3×8, card enumerated) — unlike the morning wedge; mandatory follow-up
  anyway: `fpga-helper unload` → `load` → `full_reset` ×4 → rowclone smoke
  (validated 2×). Root fix: receiveData timeout in the api (pending task).

---

# Addendum 2 (same day, evening) — receiveData timeout + PIM_FUSED_COSET in production

## receiveData stall guard (api/platform.{h,cpp}) — LANDED
Opt-in via `PIM_RECV_TIMEOUT_MS=<ms>`: receiveData returns short once no NEW
c2h data arrives for that long (progress resets the clock) and poisons the
platform; poisoned state refuses further execute() (whose receiver.join on
the kernel-blocked drain thread was the second, unguarded hang point),
fast-fails further receives, and the destructor detaches instead of joining.
Unset env = pristine block-forever behavior, bit-identical. Regression test
`recv-timeout-test-exe` (deterministic); live-fire on a real iters=3 stall:
the process self-terminated in 8 s wall, rc=0, no external kill, no wedge.
Long unattended runs are now kill-free — set the env on everything.

## PIM_FUSED_COSET — the coset activation update in bitnet-proj-server
`emit_bank_combined_body` step 3 now honors `PIM_FUSED_COSET`: 1 = 5 wrRows
+ 2 in-tuple coset doubleACTs (x→{1,5,9,13}+{4}, 0→{2,6,10,14}+{8}, W on
{3,7,11,12,15}); 2/3 = diagnostic variants. Validated with the new protocol
driver `ab_fused_server.py` (synthetic ternary masks vs exact host
reference; same seed ⇒ byte-identical requests across arms):

| config (d_in=256, bp=4) | baseline | fused | speedup | exactness |
|---|---|---|---|---|
| bank 0 | 9.7–9.8 ms/matmul | 6.4–6.9 | **1.45–1.53×** | both bit-exact |
| banks 0,1,2,3 | 9.5–9.7 | 6.1 | **1.56×** | both bit-exact |

- One transient non-exact run per arm in ~35 requests, neither reproducible
  on rerun — the known intrinsic flake background, shared with production.
- 4-bank exactness ⇒ ALL FOUR DIMM-2 bank tuples have the
  separated-generator geometry the position algebra needs.
- 8.5× (tool) vs 1.5× (server): production MM3D already banked the
  persistent-weight clone; the coset lever is the remaining per-body term,
  and fixed request costs bound the per-request gain.
- The May server state-leak did NOT reproduce (36+ consecutive exact
  requests) — likely cured by the 2026-05-05 fixes.
- **PITFALL (cost an hour):** running the dimm2 server without
  `PIM_SUB_START=45312 PIM_SUB_END=45952 PIM_POOL_LIST_FILE=
  pool_layout_dimm2_bank{bank}.txt` silently builds a synthetic pool from a
  mis-aligned subarray base (s72 is not 640-aligned) → clone sources land
  outside the subarray → plausible-looking garbage y. Always set the trio.
- Next for per-token numbers: `pim_linear.py` / `run_bitnet_pim.py` /
  `bitnet_chat.py` + `~/bitnet_weights` are NOT on the tower (full-tree
  search) — pull from the Windows snapshot before real-model A/B.

---

# Addendum 3 (same day, night) — stall ROOT CAUSE found and FIXED

User called the "probabilistic stall" suspicious ("on the laptop, effect
had a cause") — correctly. Diagnosis chain:
1. `PIM_RECV_DEBUG=1` (new, in consumeData) captured a stall live: healthy
   execs = 3×(got=32) write-program trailers + got=8224 (row+trailer); the
   stalled exec delivered **ZERO bytes — the FPGA never started the read
   burst**. Not a host-side trailer mis-strip.
2. Discriminator: the server (fresh Program per execute) had run thousands
   of clean execs; the 07-17 tools pre-build one Program and re-execute
   it. Our own logs showed the re-executed program GROWING (A: 392→395→398
   insts).
3. Root cause — `api/prog.cpp Program::get_inst_array()` finalizes the
   program on EVERY call: `linear_analysis()` records SMC_INFO read-burst
   packets in `warnings` (never cleared), `insert_generated()` re-inserts
   all of them each call (growth), and `preprocess_branches()` does
   `br |= target` — OR-ing a SHIFTED target over the previously baked bits.
   Once an insertion lands before a label, branch destinations become
   garbage → rdRow never streams → the 0-byte c2h stall. A stale SMC_INFO
   beat count likewise explains the earlier mid-row "stopped half-way,
   21/256".
4. Why "2 safe / ≥3 fatal": for these programs the first re-execution's
   insertion lands after all labels (OR idempotent → benign — which is why
   the batch-of-2 campaign stayed bit-exact); accumulation eventually
   shifts labels/segments → corruption. Deterministic cause, per-program
   luck — and the laptop era never stalled because those tools didn't
   re-execute pre-built Programs.

**Fix:** `finalized` flag in Program — finalize exactly once, clear
`warnings`; later get_inst_array() calls only re-serialize. Offline
idempotency test: byte-identical arrays across calls (PASS).

**Validation ladder (previously 2/2 stalls at iters≥3):** iters=8, 20, 50
(yesterday's wedge-maker config), 100 — ALL clean, zero timeouts, both
benders. **The "iters≤2 per process" envelope from Addendum 1 is
OBSOLETE.**

## Definitive steady-state numbers (iters=100, one process — supersede the
batch-of-2 campaign, whose means carried per-process cold-start in BOTH arms)

| bender | A (11-wrRow + W-rewrite) | B2 (fused) | A→B2 | B2 correctness |
|---|---|---|---|---|
| 2 | 0.357 ms/MAJ | **0.089 ms/MAJ** | **3.99×** | **0/204,800** |
| 0 | 0.556 ms/MAJ | **0.085 ms/MAJ** | **6.53×** | **0/204,800** |

B1 (unbalanced margins) stays catastrophically wrong at N=100 (99.7% /
97.9% bad) — the 5/5/5 rule holds. Server re-validated on the rebuilt
objects: exact (its fresh-Program-per-execute pattern was never affected;
the PIM_FUSED_COSET production numbers stand).

**Upstream note:** `DRAM-Bender-master` carries the IDENTICAL latent bug
(same get_inst_array, same `br |=`). Any DRAM-Bender user who executes a
pre-built Program more than once gets silent program corruption —
outreach-worthy (user-gated).

---

# Addendum 4 (same day) — MVDRAM performance-shape kernel ASSEMBLED

`mvdram-fastpath-ab-exe` (new): the popcount-4 carry-save DAG (18 MAJ3
gates, same safe placement as Part 3) run three ways — C = June write-load
shape, A = safe clone-load unfused, B = clone-load with the WHOLE gate
fused into one program (3 operand clones + wrRow(ONE)→Tfr + frac×3 + MAJ3
+ rdRow + result clone-out; 154 insts; pre-built once per gate and
re-executed, which the idempotency fix makes legal).

| | C (June) | A | B (fused) | B speedup |
|---|---|---|---|---|
| bender 2 | 1.585 ms/gate, 99.990% e2e | 0.885, 99.865% | **0.683, 99.808%** | **2.32×** |
| bender 0 | 1.574 ms/gate, 99.974% e2e | 0.898, 99.839% | **0.705, 99.814%** | **2.23×** |

Fast loading costs ~0.17% e2e; fusing costs nothing (A≈B accuracy).
Cross-die reproducible. Supersedes REPRODUCTION.md's "performance-faithful:
silicon-blocked" verdict (both repro docs updated with banners). Logs:
`fastpath_ab_b2.log`, `fastpath_ab_b0.log`.

---

# Addendum 5 (same day) — the COMPLETE selection law (timing-invariant, cross-die)

Following the SAFARI reply's per-module-characterization advice
(SiMRA-DRAM#1, their §3.2 pointer), `selection-timing-exe` (new) measured
WHICH predicted-set members fire for arbitrary APA pairs: k=1..5 differing
subarray-local bits × 4 anchors × t12 ∈ {5,10,20,30} × t23 ∈ {1,2}, s72
subarray, 1,691 member-observations per die (CSVs + logs in this dir).

**Result — a zero-exception law (1691/1691 on bender 2 AND bender 0):**
subarray-local row bits partition into predecoder groups **{1,2}, {3,4},
{5,6}, {7,8}** (bits 0 and 9 singleton), and member A⊕S activates
**iff for every group g: S∩g ∈ {∅, d∩g}** — each group's latch holds d's
whole within-group pattern; only entire latched group-units are selectable.

- **Timing-invariant**: identical firing sets at all 8 (t12,t23) combos —
  selection is decode structure, not analog. Timing cannot buy higher-k
  addressing on this module (contrast their §3.2 success-rate advice,
  which concerns MAJX analog margins, not selection).
- **Cross-die byte-identical** (1,693-line CSVs differ only in the
  /dev/xdma banner lines) — the law is model-level digital structure,
  consistent with [[cross_die_determinism]].
- **The k-decay is combinatorial here, not stochastic**: firing count =
  2^(#group-units in d); the full 2^k set fires iff d contains no complete
  pair-group. (Their module's 99/95/83/63/32 curve may be the same law
  under their group structure + a genuine stochastic term.)
- Retro-explains: the calibrated 16-row tuple = exactly the law's firing
  set for its pair (units {0},{1},{5,6},{7,8} → 2^4); the safe-load d=6
  "bit-1 point didn't fire, ~50% group behaviour" note (d∩{1,2}={1,2} →
  S={1} structurally silent — not probabilistic); Phase C group-atomicity.
- **Engineering**: mask screening (e.g., the fastpath's 28/77 pass) can
  become a computed predicate; lattice-addressing design rules are now
  closed-form for this module class.
- Thread-worthy (user-gated): this partially answers our own Question 1
  on our silicon and gives the per-module "fingerprint" procedure.

---

# Addendum 6 (same day) — REAL-MODEL per-token A/B: fused coset = 1.6×

The python client turned out to live in the public repo (python/
pim_linear.py + run_bitnet_pim.py — current versions); model re-downloaded
from HF (1,178,623,988 B, manifest-exact). torch 2.7.1 + transformers
4.52.0 (2.13 breaks the bitnet integration import). Config: layer 0 × all
7 projections, bender 2, banks 0,1,2,3, calib_dimm2 + per-bank pools +
SUB_START/END trio, PIM_NO_CHAT_TEMPLATE=1, 8 tokens, greedy.

| arm | 8-token wall | PIM server-request | response |
|---|---|---|---|
| baseline | 117.2 s | 112.5 s | " Answer: Paris Paris\n Paris is Paris" |
| **PIM_FUSED_COSET=1** | **71.8 s** | **70.2 s** | " Answer: Paris Paris Paris Paris Paris The" |

**1.63× wall / 1.60× PIM time on the real model** — matching the
ab_fused_server.py prediction (1.45–1.6×). Both arms answer "Paris"; the
textual divergence is the known per-bank-flake sampling effect (layer-0
substitution + minimal prompt is the stress config). Logs:
`token_ab_baseline.log`, `token_ab_fused.log`.

---

# Addendum 7 (2026-07-17, idle-FPGA session while Vivado built off-box)

Two questions answered on otherwise-idle silicon + the newly-local 37 GB
April archive. Tool: `frac-sweep-exe` (new) replays the EXACT SiMRA
MajOperations MAJ3 sequence and sweeps the frac knobs the sweeps hardcode
(n_frac ∈ {0..16}, t_frac ∈ {0..3}, reference-row init ONE/ZERO =
FracDRAM's two curve sides, + a retention delay).

## Finding A — frac conditioning does NOT help our MAJ3 (two dies)
| tuple | die | coverage across ALL 78 configs |
|---|---|---|
| s72 (production) | bender 2 | 99.986 – 100.000% (flat) |
| s77 | bender 0 | 99.82 – 100.00% (flat) |
On a well-calibrated 5+5+5 MAJ3 vote the single reference row can't tip a
decisive majority, so **n_frac (0→16), t_frac, and init are all irrelevant**
— the SiMRA hardcoded n_frac=3/t_frac=0 was never a fragile choice, and more
conditioning buys nothing. Frac can't improve an already-good tuple.

## Finding B — frac does NOT rescue the MAJ3-limited die (DIMM 1)
DIMM 1 (Mfr-H, s24) sits at ~30% coverage (matches its known
MAJ3-limited status). Sweeping frac: **corr(n_frac, coverage) = −0.44** —
more conditioning slightly HURTS (33.4% at n_frac=0 → ~30% by n_frac=8);
init ONE vs ZERO ≈ equal (30.4 vs 29.9%); full range 28.4–33.4%, σ=0.75.
**Decisive: frac cannot fix DIMM 1/3.** Physically expected — their limit
is decoder GEOMETRY (adjacent {1,2} generator → no separated operand set),
and frac conditions reference-row
CHARGE, not which rows co-activate. Confirms the structural verdict on
silicon and kills the "maybe frac rescues D1/D3" hope cheaply.

**Consequence for MVDRAM:** frac's only remaining possible value is MAJ5
reliable-column count on the GOOD dies (its actual role in the paper) — NOT
MAJ3, NOT geometry-limited dies. The frac experiment is now narrowed to
exactly that one target; two dead ends removed before spending the bigger
MAJ5 experiment.

---

# Addendum 8 — MAJ5 IS frac-sensitive, and the reference-row POLICY is the lever

`frac-maj5-exe` (new): the exact mvdram-maj5 per-column screen (ADR-001
3+3+3+3+3+1 layout, identical instruction sequence) sweeping n_frac ∈
{0..16} × t_frac ∈ {0,2} × reference-row init ∈ {ONE, ZERO}. Metrics:
strict (correct in ALL runs — the colmask criterion), soft95 (≥95% of
runs), mean rate. 128 runs/config for the sweeps; winners confirmed at
600 runs/config.

## s72, bender 0 (the "MAJ5 silicon-limited" subarray; July colmask 28/2048)
| reference policy | strict /2048 | mean rate |
|---|---|---|
| ONE, no frac | **0** | 4.0% |
| ONE + 3 fracs t0 (SiMRA standard) | 157 | 72.6% |
| ONE + 8 fracs t0 (plateau) | 206 | 75.5% |
| ONE + any fracs t2 | 0 | ~25% |
| **ZERO, no frac** | **963 (47%)** | 65.9% |
| ZERO + fracs t0 (pulled back to ~½) | 224→166 | ~75% |
| ZERO + fracs t2 | ~940 | ~78% |

## s86, bender 2 (the good subarray; standard recipe ≈ 87–93%)
| reference policy | strict (600-run confirm) | soft95 |
|---|---|---|
| ONE + 3 fracs t0 (standard) | 1824 (89.1%) | 1997 |
| **ZERO + 2 fracs t0** | **1915 (93.5%)** | **2010 (98.1%)** |
| ZERO, no frac | 1052 | 1163 |

## Findings
1. **MAJ5 is strongly conditioning-sensitive** (unlike MAJ3, addendum 7):
   the 15+ref vote's margins are 1-row thin, so the reference row's charge
   matters enormously.
2. **The dominant knob is the reference INIT, not the pulse count.** On the
   marginal subarray, a full-ZERO reference (least biasing) beats the
   SiMRA frac'd-ONE recipe **963 vs ~200 strict columns (≈5×; 34× vs the
   n=3 July-criterion baseline of 28)**. On the good subarray the optimum
   is ZERO + 1–2 fracs (+~91 strict columns over standard, 98.1% soft95).
3. **t_frac spacing is critical for ONE-init** (t2 interrupts too late →
   too much restore → dead), mostly irrelevant for ZERO-init.
4. This **qualitatively validates MVDRAM §VII** (conditioning raises
   reliable columns) while adding the law the paper doesn't state — and it
   **reopens faithful MAJ5**: s86-class subarray + ZERO+2 reference +
   column screening ≈ their Table-I range (82.9–95.1%). Their EXACT
   dual-track adder (carry=MAJ3, sum=MAJ5) is now plausibly runnable on
   our commodity silicon.

**Next:** rerun mvdram-adder-exe / the dual-track chain with the ZERO+2
reference policy on s86 screened columns — if chain accuracy holds, the
all-MAJ3 substitute becomes optional and the reproduction is
mechanism-exact. Logs: frac_maj5_b0_s72.*, frac_maj5_b2_s86.*,
frac_maj5_b2_s86_confirm.*.

## Addendum 8b — the EXACT dual-track adder under the new policy + the layout question

`mvdram-adder2-exe` (new): the dual-track full adder (carry=MAJ3(a,b,c),
sum=MAJ5(a,b,c,¬carry,¬carry)) with the reference policy parameterized and
a self-screen at the same policy (MAJ5 strict, 50 runs), then 20 random
(a,b,c) trials. Also tests the deposit-immune "blocked" layout (each
input's copies grouped so positions {0,1,2} share one value → any first-row
deposit into those positions is a no-op; reference at position 15).

### s86 / bender 2
| policy, layout | screen | adder strict on screen | sum err (screened) |
|---|---|---|---|
| ONE+3@t0, std (SiMRA baseline) | 1952 | 1873 (95.95%) | 0.343% |
| **ZERO+2@t0, std** | **1991** | **1942 (97.54%)** = 94.8% of subarray | **0.224%** |
| ZERO+2@t0, blk | 1951 | 1876 (96.16%) | 0.387% |

### s72 / bender 0 (marginal)
| policy, layout | screen | adder strict on screen | sum err (screened) |
|---|---|---|---|
| ZERO+0, std | 982 | **938 (95.5%)** | 0.392% |
| ZERO+0, blk | 504 | 319 (63.3%) | 7.9% |

**Findings:**
1. **The exact MVDRAM adder (MAJ5 sum, dual-track) now runs strictly
   error-free across 20 random trials on 1942/2048 columns (94.8%) of
   s86** with the ZERO+2 reference — up from 1873 under the SiMRA recipe;
   sum error rate −35%. Carry (MAJ3) is 0.000% everywhere. Even the
   "MAJ5-dead" s72 now runs it on 938 columns. The all-MAJ3 substitute is
   now optional; the reproduction can be mechanism-exact.
2. **Layout question answered (user asked: is the row choice
   spread-aware?):** the tuple IS the co-activation lattice by
   construction, and loading is wrRow (no spread). The role assignment's
   deposit exposure (reference at Rf could inject into positions 1,2) was
   tested via the immune blocked layout — and **the standard
   interleaved layout wins decisively on both subarrays** (blk collapses
   the marginal subarray 982→504 screen, 96→63% adder). The first-row deposit is
   second-order for MAJ5; spatial interleaving of input copies across the
   tuple's extent is what matters. SiMRA's layout is already near-right;
   the missed lever was the reference POLICY, not the geometry.
3. GATE "PARTIAL" = strict-100%-of-screened not reached (residual ~2.5% of
   screened columns flake within 20 trials); the paper's "error-free"
   regime needs margin-aware screening (screen harder than use) and/or
   voting — same practice their §VII implies.

---

# Addendum 9 (Task S1) — selection law on DIMM 1/3: NOT cleanly lawful

`selection-timing-exe` (now with parameterized subarray base) run directly
on-silicon on **bender 1 (DIMM 1 / Mfr-H, subarray 19840)** and **bender 3
(DIMM 3, subarray 34560)** — the clean deposit observable (zero members,
APA, read each), superseding the confounded April-archive fit (addendum 7C).

| die | k=1 deposits | best group-partition fit @ t₁₂10/t₂₃1 | timing |
|---|---|---|---|
| DIMM 0/2 (M-die, full-PUD) | 100% | **{1,2}{3,4}{5,6}{7,8} = 100%** (1691/1691) | invariant |
| DIMM 1 (Mfr-H) | 100% | shifted {0,1}{2,3}{4,5}{6,7} = **61%** (best) | t₂₃=2 kills some at t₁₂≥20 |
| DIMM 3 | 100% | no partition > **42%** | same weak t-dependence |

**Finding:** single-bit (k=1) deposits are reliable and timing-invariant on
ALL dies, but **multi-bit (k≥2) deposit selection follows a clean
group-partition law ONLY on the full-PUD M-die.** DIMM 1/3 deposit
selection is substantially reduced (k=2 full-set 33% on D1 vs decisive on
M-die), does not fit any adjacent-pair partition cleanly (≤61%), and is
weakly timing-DEPENDENT (t₂₃=2 suppresses deposits) — unlike the M-die's
timing-invariant law. This is a THIRD independent corroboration (after
yield maps and MAJ3 limits) that DIMM 1/3 are structurally different, not
merely "less reliable."

**Consequence for CLICK 5 (DIMM 1/3 coset addressing):** weakened — their
multi-bit coset deposits aren't lawful/reliable enough to exploit for
accelerated IO. DIMM 1/3 are storage-only dies; no lawful coset-IO
acceleration. (k=1 clone IO still works.) The selection law is a
FULL-PUD-die property. Data: seltiming_b{1,3}.{csv,log}.

## Finding C — archive corroborates DIMM 1's distinct decoder structure
Fitting the selection-law group partition to the April FindOpenRows archive
(`experimental_data/simra_full_fo_simd_v2_2026_04_30`, `law_fit_archive.py`):
DIMM 0 fits {1,2}{3,4}{5,6}{7,8} best; **DIMM 1 fits a DIFFERENT, bit-shifted
partition** ({0,1}{2,3}…) — independent corroboration of July's "DIMM 1 extra
⊕2" anomaly from a completely separate dataset.
**Caveat (stated honestly):** the April sweeps ran at t₁₂=0 (co-activation
OPEN detection), whereas the selection LAW governs the DEPOSIT at t₁₂≥10, so
the fit percentages (D0 91%, D2 58%) are NOT a clean test of the deposit law
— only the qualitative per-die structural DIFFERENCE is load-bearing. A clean
test = re-run `selection-timing-exe` directly on bender 1 (idle-silicon
follow-up).

---

# Addendum 10 (Task S2) — MAJX × reference policy at 16 AND 32 rows; per-bank MAJ5 colmasks; SiMRA contrast

`majx-screen-exe` (new, v3): generalizes the frac-maj5 screen to
MAJ3/5/7/9/11/15 with the free-row policy as the swept variable, on (a) the
calibrated 16-row tuples and (b) **32-row cosets constructed purely by
formula from the selection law** (units {1,2,8,32,128}, d=171, no
calibration pass). Adds two diagnostic patterns per X — jmaj (every bit at
the minimal-majority vote, exp=1) and jmin (maximal minority, exp=0) — that
read the two margin sides directly. Vote-algebra model under test: with C
copies × X inputs on N rows, the free rows' total charge r must lie
strictly inside (N/2 − C·⌈X/2⌉, N/2 − C·⌊X/2⌋), an open interval of width
C — so N=16 gives MAJ7 only r=1, and MAJ9/15 (C=1) NO valid binary policy
(a fractional row would be structurally required); N=32 reopens all
intervals (MAJ7 r=2, MAJ9 r∈{2,3}, MAJ11 r=5, MAJ15 r=1).

**Source verification (SiMRA paper, papers/SiMRA.pdf):** their MAJX runs
use **32-row activation**; Fig-7 protocol average success rates MAJ5/7/9 =
**79.64% / 33.87% / 5.91%**; footnote 11 omits "MAJX operations that have
<1% success rate at most (i.e., MAJ11+ for Mfr. H, and **MAJ9+ for
Mfr. M**)"; Obs. 6: input replication raises success (MAJ3 32-row vs 4-row
+30.81%); Obs. 7: their timing optimum t1=1.5 ns, t2=3 ns; Obs. 9: random
data lowers MAJX success (MAJ7 −32.56% vs fixed patterns). The earlier
stocktake line "their MAJ15 claim" was WRONG — their ceiling claim is MAJ9,
and MVDRAM's adder needs only MAJ3+MAJ5 (§II-C). Corrected in
STOCKTAKE_2026_07_17.md.

## A. 16-row tuples (sweep, 96 runs/config, benders 0+2)
b0 = s72 production tuple (marginal anchor — m5 rows reproduce addendum 8:
965→968 Z0, 158→184 simra); b2 = the MAJ5-selected s86 tuple
Rf=54423/Rs=54474 from mvdram-repro/calib_maj5_dimm2.txt (good anchor —
m5_Z2 1958 ≈ the 1915 confirm).

| config | b0/s72 strict (jmaj/jmin %) | b2/s86maj5 strict (jmaj/jmin %) |
|---|---|---|
| m5_Z0 | 968 (56/100) | 1111 (61/100) |
| **m5_Z2** | 250 (79/72) | **1958 (99.2/99.9)** |
| m5_simra | 184 (85/59) | 1909 (99.9/99.8) |
| m7_ZZ (r=0) | 693 (37/100) | **737 (43/100)** |
| m7_OO (r=2) | 0 (100/0) | 0 (100/0) |
| m7_BAL (r=1) | 1 (53/38) | 11 (57/31) |
| m7_FF2/FF3 (frac'd pair) | 0 | 0–7 |
| m9_* (9 policies incl. frac'd both inits, nf 1–8) | ≤5 | ≤5 |
| m15_* (15 policies incl. frac'd, nf 1–16) | ≤2 | ≤2 |

- The MAJ7 diagnostics land EXACTLY on the vote-algebra tie predictions
  (ZZ: jmin=100.00, jmaj=43 = the 8v8 tie side; OO: jmaj=100, jmin=0) —
  but tie resolution is **position-weighted, not symmetric**: the same
  nominal 8v8 tie resolves ~43% high when the ONEs are mid-slot data rows
  and ~100% high when the ONEs sit on slot 0 (=R_F) / slot 15. This
  directly evidences SiMRA's Obs-7 hypothesis (R_F contributes more
  charge) as a STATIC slot-weight effect present even at t12=t23=0, and it
  is why their "balanced" r=1 policy fails on 16-row tuples (1–11 strict).
- **Frac conditioning does NOT unlock C=1 layouts**: all m9/m15 frac'd
  variants are byte-identical to each other (~3.5% mean, jmaj≈48, jmin≈0)
  regardless of init and pulse count — the reference row's charge is
  irrelevant there; the failure is structural. MVDRAM §VII's "Frac [34] +
  calibration [48]" therefore cannot be about enabling X>5 — consistent
  with their adder needing only MAJ3+MAJ5, and with our addendum-8 reading
  (frac = MAJ5 column-yield knob).

## B. 32-row law-constructed cosets (bender 0, SUB=45312/A=64 and s77 SUB=48640/A=64)
| config | s72-eq strict | s77 strict | note |
|---|---|---|---|
| **m3_32_BAL** | **2048** | **2048** | perfect MAJ3 on a FORMULA-BUILT 32-row coset — calibration-free tuple construction from the selection law works on silicon |
| m3_32_ZZ | 2042 | 2048 | |
| m5_32_BAL (r=1) | 1028 | 1573 | 32-row MAJ5 < 16-row (1952-class): replication scaling does NOT extend 16→32 here; BAL beats ZZ at N=32 (policy law is N-dependent) |
| m5_32_ZZ | 1022 | 1023 | |
| m7_32 best (ZZZZ/1O3Z) | 547 | 813 | same ~35–40%-of-columns tie regime as N=16 — our MAJ7 ceiling ≈ their 33.87% average |
| m7_32_2O2Z (predicted mid) | 88 | 3 | slot-weighting defeats the naive balanced policy |
| m9/m11/m15 all | ≤22 (m9_32_Z5: 424 on s72-eq only, 8 on s77 — not portable) | ≤8 | MAJ9+ unusable — **reproduces SiMRA footnote 11 for Mfr. M** on our M-die |
| *_t12 (their t1=1.5/t2=3 ns) | 0 (6.25% mean, all-zero reads) | 0 | their timing optimum does NOT transfer — t12=1 kills co-activation on our module; timing laws are module-specific |

## C. Per-bank MAJ5 tuple selection + colmasks (the S2 deliverable)
`select` mode: quick-screens EVERY calib tuple line per bank (73–98
candidates from calib_dimm{0,2} + mvdram-repro/calib_maj5_dimm{0,2}) under
Z0 and Z2, ranks, deep-screens the winner (20 pats × 25 trials = 500 runs,
strict). June's tuple selection used the SiMRA frac'd-ONE recipe
implicitly; re-selection under the ZERO policy re-ranks tuples (e.g. b2
bank0's winner is an s78 tuple that beats the June-selected s86 one).

| bank | bender 0 winner (strict/2048) | bender 2 winner (strict/2048) |
|---|---|---|
| 0 | s77 48710/48892 — 1939 | s78 49371/49461 — 1954 |
| 1 | s77 48710/48892 — 1952 | s86 54427/54444 — 1919 |
| 2 | s93 58924/58979 — 1952 | s78 49371/49461 — 1935 |
| 3 | s61 38643/38751 — 1920 | s78 49371/49461 — 1936 |

**All 8 banks (both M-dies) now hold MAJ5 colmasks at 94–95% strict** —
the top of MVDRAM Table I's reliable-column range (82.9–95.1%) — with full
tuple provenance (Rf/Rs, 16 open rows, calib source, policy, depth) in
each file. The same s78 tuple wins on 3 of 4 bender-2 banks (bank-invariant
structure, consistent with the byte-identical fault sets). Files:
`BitNet/colmasks_maj5_zero_2026_07_17/maj5_best_b{0,2}_bank{0..3}{.txt,_ranking.csv}`.
SUPERSEDED: the first-pass per-(s_id,bank) colmask files from
run_majx_s2.log (19:12) — tuple-ambiguous headers (readc took the FIRST
calib line; the numbers are valid but the b2 files screened generic
MAJ3-grade tuples, not the MAJ5-selected ones) — deleted and replaced by
the v2/v3 select-mode files at 19:22.

## Consequences
1. The practical MAJX menu on this module: **MAJ3 perfect (16 or 32 rows,
   incl. formula-built cosets) + MAJ5 at 94–95% of columns (16-row, ZERO
   policy, all 8 banks) + marginal MAJ7 (~35–40% of columns,
   tie-exploitation, screenable) + nothing above** — exactly the substrate
   MVDRAM's dual-track adder needs (MAJ3 carry + MAJ5 sum), now
   quality-matched to their best Table-I modules on every bank.
2. Our MAJ9-dead result is not a deficiency vs the papers — it REPRODUCES
   SiMRA's own per-manufacturer finding for Mfr. M silicon.
3. SAFARI-thread material: the slot-weighted tie mechanism (m7 ZZ/OO
   diagnostic asymmetry) is direct evidence for their Obs-7 R_F-dominance
   hypothesis, measured statically; and the reference-policy law (ZERO vs
   BAL vs frac'd, N-dependent) is the knob their §3.2/§VII advice doesn't
   state. (Posting user-gated.)
4. M1 gets silicon proof: law-constructed 32-cosets co-activate and compute
   perfectly (MAJ3 2048/2048) — allocator placement by formula is real.
Logs: majx_b0_s72.{csv,log}, majx_b2_s86maj5.{csv,log},
majx32_b0_s72eq.{csv,log}, majx32_b0_s77.{csv,log}, run_majx_s2_v2.log,
run_majx_s2.log (v1, superseded).

---

# Addendum 11 (Task S3) — FCDRAM NOT: clean negative on all die classes; the selection law is 1024-BLOCK-scoped

`fcdram-not-exe` (new): faithful replay of FCDRAM's neighboring-subarray
detection protocol (papers/FCDRAM.pdf §4–§5): init both 640-row subarrays
to a background, doubleACT(t12,t23,src,dst) = their ACT R_F→PRE→ACT R_L,
then WR a marker THROUGH the open row buffer, scan both subarrays. A
genuine shared-sense-amp coupling shows the marker's INVERSE on half the
bitlines of the other subarray's rows; two complementary rounds (bg 0/mk FF
and bg FF/mk 00) make the signature unambiguous. Grid: close/middle/far
row positions × both directions × 8 timings (t23 = their "reduced tRP").
Paper predictions: NOT works on SK Hynix; Micron shows "neither
simultaneous nor sequential" neighboring-subarray activation (§4.3).

## Verdict: no NOT on any tested die
| bender (class) | subarray pair | inverse/NOT signatures | notes |
|---|---|---|---|
| 1 (Mfr-H class) | 19840/20480 | **none** (128 configs) | only marker-direction deposits into D1's messy co-activation twins (⊕32-class, ~half-written, timing-dependent 0.50→0.74) |
| 3 (Mfr-H class) | 34560/35200 | **none** | partial non-coset deposit sets — matches addendum 9's "D3 not lawful" |
| 0 (Mfr-M, control) | 48640/49280 | **none** | as FCDRAM's own Micron finding predicts |

Scope honestly stated: one subarray pair per die, bank 0 (their coverage:
4 pairs/bank, many banks, 10 SK Hynix modules). Our Mfr-H-class DIMMs do
not exhibit the SK Hynix cross-subarray coupling in this region — either a
different die generation/geometry, or 640-row segments are not
shared-sense-amp neighbors on these modules. Consequence for MVDRAM: none
— their dual-track adder gets ¬c from the duplicated matrix track, not
from an in-DRAM NOT; no reproduction gap opens.

## THE positive result: the co-activation lattice is 1024-BLOCK-scoped
On bender 0 (M-die), same-1024-block src/dst pairs ACROSS the 640-row
boundary deposit the marker into the EXACT full-address XOR coset:
| pair | src→dst | d | units | predicted | observed |
|---|---|---|---|---|---|
| Ac→Bc | 49276→49284 | 248 | {24,96,128} | 8 rows | **8 — membership exact** (4 rows in EACH 640-segment) |
| Bc→Ac | 49284→49276 | 248 | {24,96,128} | 8 | **8 — same set** |
| Ac→Bm | 49276→49600 | 444 | {4,24,32,384} | 16 | **16 — membership exact** (8 per segment, incl. src) |
All timings (t12 0–30 × t23 0–2), both rounds, both directions.
Cross-1024-block pairs deposit NO lattice coset (n_full 0–3, non-coset)
and even degrade dst's own write (dst at 0.63 fraction). Hand-verified
memberships: {49284⊕S : S⊆{24,96,128}} = {49156,49180,49252,49276,49284,
49308,49380,49404} ✓ and the 16-row {4,24,32,384}-coset of 49600 ✓.

**Consequence:** "subarray-local" in the selection law actually means
**low-10-bit-local within a 1024-row predecoder block**; 640-row units are
sense-amp segments only. (All prior tuples sat inside one block, so the
distinction never surfaced.) Allocator safety rule: a doubleACT whose coset
crosses a 640 boundary WILL deposit into the neighboring segment — plan
placement block-scoped. Production s72 layout verified safe (subarray
wholly inside its block). Larger domain too: cosets may legally span two
segments of one block.

## Tooling lesson (cost us run v1)
`wrRow_immediate` expands to ~80 instructions; batching 100/program
SILENTLY overflows the 2048-instruction IMEM — truncated-tail rows never
get written, no error is raised. The v1 "978 half rows" artifact was
exactly the un-initialized residue (config-independent, Jaccard 0.997).
Fixed to the proven 16 rows/program idiom + a no-op BASELINE scan per run
(baselines: 0 deviating rows on every die — even DIMM 1 stores perfectly).
Logs: fcdram_b{0,1,3}.{csv,log}, fcdram_b0_v2.{csv,log} (full-row lists).

---

# Addendum 12 (Task T1) — 8K-IMEM bitstream VALIDATED on silicon

Bitstream: one-variable rebuild (IMEM 2048→8192 instructions; POPCOUNT_
ACCUM_MODE still disabled), built on the vivado box (WNS +0.083 ns, 0
failing endpoints), md5 8760ee19e24657a5e44845db7f89c48c, JTAG-programmed
live 2026-07-17 evening.

## Post-reprogram bring-up lesson (now in RUNBOOK_TOWER.md)
Live JTAG reprogram leaves the kernel's endpoint STALE. The fix sequence:
PCI **remove + rescan** (fpga-helper pci-reset is WRONG here — FLR restores
the pre-reprogram config snapshot; it produced "Failed to detect XDMA
config BAR"). If the endpoint won't answer config reads after remove
(link retrains to 2.5 GT/s but rescan finds nothing), a **warm reboot**
re-enumerates from POST and the JTAG image SURVIVES (slot power stays on).
NEVER cold-cycle after JTAG programming — that erases the volatile image.
Confirmed post-reboot: single 64K BAR = the design's normal layout, driver
"config bar 0" probe clean, link Gen3 8.0 GT/s x8, 4+4 channels.

## Validation results (BITSTREAM_IMEM=8192)
| check | result |
|---|---|
| PHY init full_reset ch0–3 | OK ×4 |
| rowclone-smoke b2 | PERFECT_CLONE (100%) |
| matvec-smoke K=6 (old gate) | ALL_PASS, 225 µs/MAJ3 |
| **matvec-smoke K=12** | **ALL_PASS** — past the old 2048-instr ceiling |
| **matvec-smoke K=20** | **ALL_PASS** (7.3 ms/exec, scaling normally) |
| recv-timeout stall guard | TIMEOUT-TEST PASS (fires, poisons, clean exit) |
| fused-maj B2 | bit-exact 0/4096 @ 0.086 ms/MAJ (pre-flash 0.085–0.089) |
| MAJX anchors (b2 s86maj5 sweep) | m3 2048/2048; m5_Z2 1966 vs 1958; m5_simra 1915 vs 1909; m5_Z0 1098 vs 1111 — identical within strict-count noise |

**The 2048-instruction program ceiling is GONE** (validated ≥K=20; exact
new ceiling to be mapped when T2 packs programs). May's "8K failed in
short testing" is now definitively attributed to the May-7 hybrid
DRAIN_POPCOUNT tree, not the IMEM depth — the one-variable rebuild works.
Unblocks: T2 (K-bitplane inlining + multi-body packing), larger wrRow
batches (the ~24/program truncation limit from addendum 11 relaxes ~4×).
Env: set BITSTREAM_IMEM=8192 for platform + server from now on.
Log: majx_b2_s86maj5_post8k.{csv,log}.

---

# Addendum 13 (Task S4) — the dual-track ripple adder does NOT collapse; it is architecturally non-accumulating

`adder-chain-exe` (new; uses `lattice_alloc.h`, the M1 allocator core):
the faithful MVDRAM dual-track ripple adder run for real over many bit
positions on the S2 per-bank tuples + ZERO policy. Per bit i:
  cbar_{i+1} = MAJ3(¬a_i, ¬b_i, cbar_i)          (carry track, De Morgan)
  s_i        = MAJ5(a_i, b_i, c_i, cbar_{i+1}, cbar_{i+1})   (sum, output)
  c_{i+1}    = MAJ3(a_i, b_i, c_i)               (carry track)
Every MAJ result harvested from the tuple read-row O(0). Primary inputs
host-written (host-known); sum bits read to host (they ARE outputs); the
carry pair (c, cbar) are the in-DRAM intermediates, read back from DRAM
each stage and fed forward — a DRAM-corrupted carry propagates.
FAITHFULNESS CAVEAT (as-run): carries round-trip the host register file
between stages; a fully-resident chain fans them into the 5 operand slots
by coset broadcast = Task M3. What is measured: the real per-position
MAJ5/MAJ3 error and whether it compounds.

## Results (col,trial samples on the screened colmask)
| tuple / die | policy | bits | final c_out err | sum err range | carry-track err |
|---|---|---|---|---|---|
| s78 / bender 2 | Z2 (ZERO+2) | 8 | **0.0000%** | 0.026–0.102% | 0.0000% (all 8) |
| s77 / bender 0 | Z2 | 8 | **0.0000%** | 0.064–0.141% | ≤0.0064% |
| s78 / bender 2 | ONE3 (frac'd-ONE, June recipe) | 8 | 0.0000% | 0.147–0.275% | 0.0000% |
| **s78 / bender 2** | **Z2** | **16** | **0.0000%** | **0.009–0.068%** | **0.0000% (all 16)** |

## Findings
1. **No chain collapse at any depth tested (up to 16 bits), on both dies.**
   The sum error is FLAT across bit position (bit0 ≈ bit15, no upward
   trend) — it is just the per-bit MAJ5 single-op reliability, not a
   compounding error.
2. **Why it can't accumulate (the architectural point):** the only value
   that ripples is the CARRY, computed by MAJ3 — which on screened columns
   is essentially error-free (0.0000% at 14/16 positions, ≤0.006%
   elsewhere). Each SUM bit is a terminal output that never feeds forward,
   so its ~0.05% MAJ5 error stays local. A dual-track adder is therefore
   non-accumulating by construction: robust MAJ3 carry + non-fed-forward
   MAJ5 sum. This is the deeper reason MVDRAM's adder is viable on
   commodity silicon.
3. **ZERO policy roughly halves the residual sum error** vs the frac'd-ONE
   recipe on the same tuple (0.03–0.10% vs 0.15–0.28% at 8 bits) — the
   addendum-8 policy law carried through to the full adder.
4. **The June "chain collapse" is explained and cured:** it was the
   frac'd-ONE recipe on marginal/unscreened columns (where the single-op
   MAJ5 is already poor); screening + the ZERO policy removes it. Even the
   old recipe holds on screened columns because the carry track is MAJ3.
5. **M1 allocator core validated in use:** `lattice_alloc.h` fit the S2
   tuples to their 4-generator coset (16/16 exact), confirmed one-1024-block
   containment, and — where the earlier scratch-parking variant was tried —
   flagged the multi-bit-offset spill that the O(0)-harvest design avoids.

## Consequence
The MVDRAM dual-track adder is now reproduced **mechanism-exact AND
multi-bit**, error-free at the carry level and at the single-op MAJ5 level
for the sum, on both M-dies, at up to 16-bit depth. Combined with addendum
10 (all 8 banks at 94–95% MAJ5 columns) the reproduction's arithmetic core
is complete; the remaining faithfulness gap is purely the in-DRAM carry
fan-out (M3) and streaming shape (T2), not the arithmetic. Data:
chain_b2_s78_Z2.csv, chain_b0_s77_Z2.csv, chain_b2_s78_16bit.csv,
and the ONE3 A/B in the run logs.

---

# Addendum 14 (Task M2 / CLICK 1) — the MERGE: in-DRAM popcount accumulation kills the readout wall

`popcount-indram-exe` (new; `lattice_alloc.h` + the S4 dual-track full
adder): the ternary-LLM matvec's readout wall — reading K product rows
(weight AND activation) per output lane to the host for popcounting
(casa_readout = n_weights×16/8 B, exp0_readout_floor) — is eliminated by
summing IN DRAM. K product bitplanes are reduced by a carry-save-adder tree
of dual-track full adders (sum=MAJ5(a,b,c,¬carry,¬carry)=XOR3;
carry=MAJ3(a,b,c) = the 3:2 compressor) to ceil(log2(K+1)) result
bitplanes, read out instead of all K. This is CLICK 1 realized: the
reproduction machinery (S4 adder) applied to the ternary LLM's own
bottleneck.

## Results (bender 2, s78 tuple, ZERO+2 policy, screened columns)
| K | lanes exactly correct | result-bit err | readout in-DRAM/host | reduction | MAJ/tile | s/tile |
|---|---|---|---|---|---|---|
| 8 | 99.73% | 0.111% | 4 / 8 rows | 2.0× | 24 | 0.08 |
| 16 | 99.47% | 0.182% | 5 / 16 | 3.2× | 48 | 0.18 |
| 32 | 98.98% | 0.341% | 6 / 32 | 5.3× | 96 | 0.35 |
| 64 | 98.64% | 0.488% | 7 / 64 | 9.1× | 192 | 0.59 |

## Findings
1. **The merge works on silicon.** A pure-DRAM popcount over K AND-product
   rows, exact for 98.6–99.7% of output lanes, with readout reduced to
   ceil(log2(K+1)) rows. The reduction is K/ceil(log2(K+1)):
   **2.0× (K=8) → 9.1× (K=64) → 213× at BitNet's K=2560** (12 result bits
   vs 2560 product reads). That is the readout wall (exp0's 8× amplification
   that disqualified the CaSA-shape pipeline) removed at the architectural
   level, not shaved.
2. **Accuracy degrades sub-linearly with tree size** (0.27%→1.36% lane
   error, K=8→64): each full-adder sum contributes ~0.05% MAJ5 error and
   the tree has O(K) adders but O(log K) depth on any one lane. The carry
   track stays ~exact (MAJ3), so errors don't cascade — consistent with
   addendum 13's non-accumulation. For an EXACT accumulator: screen harder
   / vote the marginal columns, or use the FPGA popcount_accum HDL (the T3
   decision — this addendum is the in-DRAM arm's data).
3. **Cost:** 3·(K−1)+ripple ≈ 3K MAJ ops per tile of 2048×32 lanes; at
   0.086 ms/MAJ that is round-trip-bound today (per-program overhead ≫ the
   DRAM op) — the crossover vs host-popcount flips decisively once T2's
   multi-body packing amortizes the round-trips (the shape work the 8K IMEM
   now enables).
4. **Sign handling** (ternary −1) maps on top as a second popcount over the
   negative-weight plane and a subtract — a constant number of extra rows,
   not per-K — so the reduction ratio is preserved. Implemented in M2's
   next iteration inside the tile kernel.

## Consequence
CLICK 1 is demonstrated end-to-end at the kernel level: the two tracks are
merged — the MVDRAM adder machinery, validated as a faithful reproduction
(addenda 8/13), now accumulates the ternary LLM's own dot products in DRAM
and removes its defining bottleneck. Remaining for the full headline:
coset-broadcast the products so operand fan-out is also in-DRAM (M3),
multi-body-pack to amortize round-trips (T2), then the full-model per-token
number (T4). Tool: BitNet/popcount-indram-exe. Data:
popcount_indram.log, popcount_indram_scaling.csv.

---

# Addendum 15 (Task T2) — multi-body MAJ packing on the 8K IMEM: ~2.3× throughput

`packed-maj-exe` (new): M independent MAJ3 ops on M row-disjoint tuples run
(a) unpacked = M Programs + M receives vs (b) packed = ONE Program with M
(doubleACT + labelled rdRow) bodies + M receives from the single execute.
Enabled by the T1 8K-IMEM bitstream (the packed program exceeds the old
2048-instruction ceiling).

| M | unpacked µs/MAJ | packed µs/MAJ | speedup | packed prog insts |
|---|---|---|---|---|
| 3 | 46.5 | 36.8 | 1.26× | 824 |
| 8 | 61.4 | 23.8 | **2.58×** | 2144 (>old ceiling) |
| 16 | 57.1 | 27.6 | 2.07× | 4256 |
| 29 | 60.7 | 26.6 | 2.29× | 7688 (near 8K) |

- **~2.3× throughput** from amortizing per-Program dispatch (one h2c +
  one execute for M bodies). Bit-exact (≤6 wrong words / 2048·M = MAJ3
  noise). The M=29 program at 7688 insts is direct proof the 8K IMEM is
  doing real work — this exact packing FAILED on the 2048 bitstream.
- **Plateau at ~2.3× is the c2h readback** (M×8KB is unavoidable once
  dispatch is amortized) — which is precisely what M2's log(K)-row readout
  reduction removes. **T2 (pack ops) × M2 (fewer readout rows) compose**:
  packing cuts dispatch, M2 cuts data volume.
- **Caveat / next:** this packs the doubleACT+readback. M2's dominant cost
  is the operand LOADS (each FA host-marshals ~15 full-row pcwrites) — that
  is M3's target (keep intermediates in DRAM, fan out by coset broadcast),
  the larger lever for the full kernel. T2 confirmed and quantified; the
  full-kernel throughput headline (T4) needs T2 ∘ M3 ∘ M2 together.
Tool: BitNet/packed-maj-exe. The 8K IMEM (addendum 12) is load-bearing here.

---

# Addendum 16 (Task T3) — readout-killer decision: keep BOTH roads, publish the comparison

Full decision record: mvdram-repro/ADR-005-readout-killer.md. Summary:

| | Road A: in-DRAM adder (M2) | Road B: popcount_accum HDL |
|---|---|---|
| mechanism | CSA tree of dual-track adders in DRAM | FPGA-side popcount aggregator |
| readout reduction | 213× (K=2560) | 2048× (8 KiB→4 B) |
| exactness | 98.6–99.7% lane (MAJ5 error) | bit-EXACT (Verilator 5/5) |
| bitstream | current (none) | rebuild (+POPCOUNT_ACCUM_MODE + pop_count4 0xe fix) |
| what it demonstrates | faithful MVDRAM in-DRAM accumulation | rig-specific systems acceleration |

**Decision:** not either/or. Road A is the FAITHFUL reproduction path (a
real DRAM-PIM ASIC has no FPGA popcount logic) and carries the reproduction
numbers. Road B is the PRACTICAL exact accelerator for a deployed ternary
LLM on this BCU1525. Publish the comparison — "two roads to kill the readout
wall, one portable to real PIM, one exact rig-specific" — never blended into
one headline. Road B is UNBLOCKED (T1 proved the one-variable Vivado flow)
and is the recommended next additive bitstream (default-off = current path
bit-identical). T4 reports Road A now; Road B once its bitstream lands.

---

# Addendum 14b — CORRECTION to the M2 economics: wall-time is upside down in the current shape; measured crossover says the resident+packed shape flips it

Addendum 14 under-stated the wall-time economics (user question caught it).
Measured primitive rates (packed-maj-exe, M=29 packed bodies, bender 2):

| primitive (packed) | µs/op | what moves |
|---|---|---|
| read-only (rdRow body) | **34.2** | 8 KB c2h per row |
| MAJ-only (doubleACT body) | **2.6** | ~nothing (commands only) |
| MAJ+read body | 32.7 | both |

## The three regimes, honestly
1. **M2 kernel AS MEASURED TODAY: in-DRAM is much SLOWER than reading
   out.** Each full adder host-marshals ~45 row-writes (pcwrites) — the
   operand loading moves MORE bus data than the readout it saves. K=64
   tile: measured 0.706 s in-DRAM vs ≈2.2 ms for packed readout of 64
   product rows (~320× slower). Addendum 14 demonstrates EXACTNESS and
   readout-ROW reduction — not a wall-time win. (This is also why the May
   analysis remembered "in-memory ops take longer than reading out": true
   in every host-marshalled shape.)
2. **Resident-operand + packed shape (M3+T2): in-DRAM WINS, now proven by
   the primitive rates.** The crossover condition is 3·t_maj < t_read →
   7.8 µs < 34.2 µs ✓ (4.4× margin). Projected per-tile from measured
   rates: K=64: host 64×34.2 ≈ 2.19 ms vs in-DRAM 189×2.6 + 7×34.2 ≈
   0.73 ms (≈3.0×; ≈1.8× with a conservative +100% for in-DRAM operand
   moves). K=2560: host ≈ 87.6 ms vs ≈ 20–40 ms (2–4×), plus the h2c side
   savings from never writing activations (their §V encoding). Labelled
   PROJECTION from measured primitives; the built kernel (M3) must confirm.
3. **Streaming / real PIM**: the margin grows further (§V-E; command issue
   ≈ free, data transfer the only cost) — the portable claim.
ADR-005 unchanged: for pure rig speed Road B (HDL popcount, exact, 2048×)
still beats both — the in-DRAM path's value is faithfulness + portability.

**Consequence:** M3 is not just faithfulness — it is THE wall-time flipper
(resident operands kill the 45-pcwrite marshalling; their on-the-fly
encoding kills activation writes entirely). M2's status: exactness +
readout-reduction DONE; shape (wall-time win) moves into M3's scope.

---

# Addendum 17 (Task M3, part 1) — on-the-fly encoding + fully-resident FA + in-tuple sub-coset MAJ, all demonstrated; bit 9 is a DEAD deposit generator

`gemv-encoded-exe` (new; probe/screen/fa/gemv modes, lattice_alloc-placed).
Lane note: these primitives serve BOTH lanes (see mvdram_vs_bitnet_
separation memory) — §V encoding is Lane-2 faithfulness AND Lane-1's
activation-write eliminator.

## 1. In-tuple 4-row sub-coset MAJ3 (the CLICK-2 compute primitive)
An APA pair spanning both small units (d=14) co-activates exactly one 4-row
group of the s78 tuple (selection law); 3 data rows + frac'd-ONE ref
computes MAJ3 **in place, per group**: 99.968 / 99.987 / 99.994 / 99.981%
on the colmask (groups 0-3). First-try bug for the record: a pair at d=6
(one unit) co-activates only 2 rows — the pair distance chooses the
sub-coset, exactly as the law says.

## 2. Bit 9 is DEAD as a deposit generator (new structural fact)
Clone probe on s78 (write marker at O(0)⊕D, clone → O(0), read): d=1:
100.00%, d=16: 100.00%, d=17: 100.00%, **d=512: 0.00%, d=513: 0.00%**.
Bit 9 — the law-fit's "singleton 9" — does not latch/deposit AT ALL here.
This retro-explains the s72 production-coset puzzle (d had bit 9 set; the
⊕512 members never fired → 16-row tuple, not 32). Allocator consequence:
only bits {0,4} give clean single-bit scratch regions on this tuple class
(+1, +16); multi-bit offsets are legal only when their whole coset is
dead/intended (a +17-source clone deposits into +1 AND +16 siblings — the
first fa-mode failure). Law memory updated.

## 3. Fully-resident dual-track full adder (zero host round-trips for operands)
All-MAJ3 identity (sum = MAJ3(MAJ3(a,b,¬c), MAJ3(a,¬b,c), MAJ3(¬a,b,c)),
carry = MAJ3(a,b,c)) on group 0, with intermediates PARKED IN-TUPLE:
cross-group clones at full-unit distances are exactly-2-row cosets
(O(0)→O(4) d=96, O(0)→O(8) d=384) — groups 1-3 double as safe row storage
while group 0 computes; one known 4-coset stray handled by assembly order.
**sum 98.976%, carry 99.299%** on the colmask; 16 clones + 5 MAJs per FA;
error composes as ~0.998^5 (the 4-row primitive's per-op rate) —
no-surprise arithmetic, and screen-hard/vote applies as usual.

## 4. §V on-the-fly encoding + §V-D sparsity skip (Lane-2 faithfulness)
Weight bitplanes resident (+1 region; dual-track W̄ region reserved);
activations NEVER written to DRAM — each live product = one same-slot
clone W→compute-slot chosen by the host-known activation bit; zero bits
SKIPPED entirely: **product error 0.0000% at both densities; commands/pass
9.0 (100%) → 5.1 (50%), wall 0.36 → 0.27 ms** — command count ∝ density,
their §V-D scaling reproduced at the command level.

## Remaining for M3 part 2 (the fused tile)
Fuse 1-4 into the K-product resident CSA tile and measure vs the packed
host-readout baseline (the 14b crossover test: FA ≈ 21 ops ≈ 55 µs packed
×2 tracks vs 34.2 µs/avoided read → single-bank ≈ par-to-loss, bank-parallel
×4 and/or §V-D density are the winning margins — measure, don't assert).
Tool: BitNet/gemv-encoded-exe. Data in run logs this addendum.

---

# Addendum 17b (Task M3, part 2 — COMPLETE) — the fused resident tile, sim-verified and measured

`resident-tile-exe`: the full M3_TILE_DESIGN.md kernel — zero-op encoded
leaves (resident W/¬W cells ARE the tree inputs; activation bits only
select), dual-track all-MAJ3 CSA tree in group 0, position-aware cell
allocator with pair-atomic positions, in-tuple parking in the position-3
lane (slots 7/11 — never cells, never staging transits; the earlier
O(5)/O(10) parking was clobbered by leaf stagings, caught by the
simulator), coset-faithful HOST SIMULATION gating silicon (schedule must
reproduce host popcount exactly — it caught every scheduling bug: position
starvation, pool exhaustion, router deadlock, park clobber), then packed
execution (T2 style).

## Silicon (bender 2, s78, K=9, 5 trials each)
| density | live ops | lane err | tile ms | packed-readout baseline ms | ratio |
|---|---|---|---|---|---|
| 100% | 522 | 2.99% | 1.08 | 0.16 | 6.6× slower |
| 50% | 94 | 1.15% | 0.36 | 0.08 | 4.6× slower |

- Packed op rate ≈ 2.1–2.3 µs/op — T2's rate exactly; the composition holds.
- §V-D sparsity works structurally: ops 522→94 and error 3.0%→1.15% at 50%.
- Error ≈ chained 4-row-MAJ noise (≈0.998^depth) — screenable/votable.

## The honest M3 verdict (completes the 14b economics arc)
Measured ops/product ≈ 58 (dual-track). At 2.3 µs/op packed that is ~133
µs/product vs 21–34 µs/read: on THIS FPGA-round-trip rig the faithful
resident in-DRAM tree is ~4–6× slower than packed readout at ANY K,
single-bank; bank-parallel ×4 + sparsity bring it to ≈par. The demonstrated
value is MECHANISM: a complete, sim-verified, silicon-running faithful
§V+§VI resident dataflow (Lane 2), and the portability claim — on a
streaming PIM controller where an op costs ~100 ns and a readout costs an
8 KB transfer, 58 ops/product wins decisively. For practical rig speed,
ADR-005's Road B stands. Lane-1 production therefore keeps host/HDL
readout; Lane-2 reproduction reports THIS kernel. M3 COMPLETE.

# Addendum 18 (Task T4 unblock) — the "8K regression" was TWO stacked host-side config/client bugs; image + server + fused path fully exonerated

Late-evening production runs on the freshly flashed 8K image generated
babble (' the the the the'), initially read as a bitstream regression.
Single-variable replay ladder (logs: t4_*.log in the session scratchpad,
token_ab_fused_8k.log here):

- R1 exact morning-fused replay + BITSTREAM_IMEM=8192 on the 8K image →
  ' Answer: Paris\nParis is is Paris', 74.7 s/8 tok (morning 2048-image:
  71.8 s) — image timing-neutral and correct at K=1. Chat-template shape
  was the remaining delta.
- R2 (template, IMEM unset) → ' you you you the': IMEM innocent.
- Server stderr (separate file /tmp/pim_server_b2_*.log, overwritten per
  run — NOT in the client log) held the cause: **BUG 1 — relative
  PIM_POOL_LIST_FILE**. Client cwd is /home/deni/bitnet_weights, the
  pattern didn't resolve, and build_backup_pool SILENTLY fell back to a
  stride pool: 25 unscreened rows at 45040, OUTSIDE [45312,45952). All
  weights loaded into wrong rows. R3 (absolute path) → prompt-anchored
  text. Fixes: absolute paths everywhere (memories corrected); server now
  HARD-FAILS on an unreadable explicit pool file (test_bitnet_server.cpp).
- Residual: template runs derailed at token 4, byte-identical across runs
  (' The capital of of capital capital capital capital') — deterministic,
  so not cell flake. Host reference (no PIM): 'The capital of France is
  Paris.<|eot_id|>'. PIM_INT_DIFF localized it: q/k/v/o decorrelated from
  position 0 (max_err 4–12k) while down/gate tails showed normal flake
  (130–250) in the SAME requests → PIM computed with a DIFFERENT
  position's activations than the reference. **BUG 2 — _pack_xbp cached
  bitplanes keyed by id(x_int8) without retaining the array**: in batched
  prefill the allocator recycles freed per-position arrays' addresses, so
  later positions hit earlier positions' stale bitplanes. Manifestation
  depends on the prompt shape's allocation choreography — the 9-position
  morning shape dodged it, the 24-position template shape hit it, always
  at the same spot (deterministic allocator sequence). Generation stayed
  superficially coherent ("The capital of") until the first token needing
  real attention retrieval — broken q/k/v then loop. PIM_XBP_CACHE=0 →
  ' The capital of France is Paris..' EXACT. Fix: content-keyed cache
  (key = raw x bytes + n_chunks; µs vs the ~1 ms pack it saves). The
  public-repo copy of pim_linear.py carries the same bug — fix at Phase P.

Lessons now standing: verify generated TEXT, not just plumbing stats;
absolute paths in all recorded commands; explicit config that can't be
honored must be fatal, never a silent fallback; the server's stderr lives
in its own per-run-overwritten file; and byte-identical wrong output
across runs means state/logic, never silicon noise.
CONFIRMED: content-keyed cache ON, template 8-tok → ' The capital of
France is Paris.' exact, 147.1 s (cache win preserved: 600 ms xbp-build).
T4 unblocked.

# Addendum 19 (Task T4) — full-model regression: two MORE stacked bugs (V2/LOAD pool collision; mis-scoped voting extras); non-fused full model RESTORED to Paris on the 8K image

Full-model runs (all 30 layers) failed with coherent-degrading-to-control-
token output while every layer-0 shape passed. Two further host-side causes,
both found by code reading after silicon discriminators (V2-forced layer-0
int-diff: V2 machinery healthy; fused/non-fused both failing equally):

- **Bug 3 — V2 scratch vs LOAD-resident collision.** V2-mode requests draw
  scratch rows round-robin over the WHOLE primary pool while LOAD_WEIGHTS
  keeps weights resident in pool rows; per_column_write_row destroyed the
  resident weights. The V2_SCRATCH reserve had been deliberately set 110→0
  in the ~50-row DIMM-0-pool era ("pure-LOAD usage" comment). Full model:
  layer 0 fills all 294 rows (L0 LOAD 300 req; L1-29 V2 60 req in stats),
  V2 traffic then smashes L0's weights → '<|end|>' babble. FIX:
  v2_scratch_reserve() (PIM_V2_SCRATCH, default 16 tail rows) + v2_pool_idx
  at both V2 draw sites + LOAD-side reserve check (ENOSPC → client V2
  fallback). Verified: 474 clean ENOSPC acks, cursor stopped at 272.
- **Bug 4 — voting extras mis-scoped by the global sub window.** cs_extra
  calibs (subs 84/71/77/76) got pools via build_backup_pool, which applied
  PIM_SUB_START/END to EVERY calib → extras received pools of s72 rows
  from the file: cross-subarray RowClones made every extra voting trip
  garbage (median(good,garbage,garbage)=garbage on all full-row V2 slices,
  PIM_VOTE_FULL defaults ON), and extras' scratch writes cycled the full
  s72 pool — re-introducing bug 3 via calib_idx>0, bypassing the tail
  reserve. FIX: env window applies only to calibs inside it; outside-window
  calibs with an explicit pool file get EMPTY pools (no stride-fallback
  inventions) and are skipped at construction; calib_idx normalized to 0
  when extras are absent (voting = 3 temporal primary samples);
  bc_calib_idx/bc_pool_idx guard empty extras (modulo-by-zero UB).
- Progression of outputs as fixes landed (fused arm): '<|end|><|assistant|'
  → (bug-3 fix) " I'm not sure, but I" → (bug-4 fix, non-fused arm)
  **' \nAnswer: Paris\n.. Paris'** at 438 s/tok — May-class correctness
  (May: 632 s/tok on the 2048 image; same protocol, 8-tok minimal prompt).
- Layer-0-only runs were structurally immune to both bugs (slices fit the
  pool → no V2 traffic, no voting) — which is why every layer-0 validation
  passed all session while the full model failed. Full-model-scale test
  now added to the regression habit.
Fused full-model headline run in flight; addendum 19b will carry it.

# Addendum 19b (Task T4 COMPLETE) — the full-model headline on the fully-fixed stack

All four fixes in (addenda 18, 19), 8-tok minimal-prefill protocol,
bender 2, 4 banks, 8K image, BitNet b1.58-2B all 30 layers × 7 projections:

| config                    | s/tok | output (first tokens)            |
|---------------------------|-------|----------------------------------|
| May 2048-image, non-fused | 632   | ' Answer: Paris\nCopenhagen\nC'  |
| 8K v3 server, non-fused   | 438   | ' \nAnswer: Paris\n.. Paris'     |
| **8K v3 server, FUSED**   | **360.8** | **' \nAnswer: Paris\nIn the United'** |

**1.75× vs the May production baseline** (1.44× from the 8K-era stack,
1.21× more from fused). Fused's full-model gain is smaller than its
layer-0 1.63× because layers 1-29 run V2-mode, whose per-request in-band
weight writes (per_column_write_row) and 3× voting trips dominate and are
untouched by the fused MAJ3 body. Correctness = May-class (correct primary
answer + per-bank flake tail), now with cross-calib voting degraded to
3 temporal primary samples (extras pending per-subarray screened pools).

Open levers (not tonight): pack4 × fused composition (pack4 alone was
112 s/8tok layer-0 ≈ non-fused serial — parallel builder lacks the fused
branch); Fig-15-style dual-track resident weights; per-subarray screened
pools to restore true cross-calib voting; Road-B per-token number after
the popcount_accum bitstream flash (V2 task, user-gated).

# Addendum 20 (Task V2) — Road-B image validated for production; accum-drain race characterized, fix staged for build3

The popcount_accum bitstream (md5 70bfa525…) is PRODUCTION-VALIDATED: the
entire T4/T5/T6 full-model campaign ran on it (READ_MODE untouched by the
accum ifdef), matvec K=20 bit-exact across four boots, 8K IMEM confirmed.

DIFF/accum drain protocol, silicon-decoded (tool: test_popcount_hw.cpp +
platform.toggle_readback_mode(), new): mode toggle = control word bit
INSTR_WIDTH+1; compare reference = 16× SMC_LDWD (PATTERN_REG LI alone is
NOT the readback reference); drain chunk = one 64 B (32b total in a
half-swapped 512b word) + 32 B trailer per program flush.

CORRECT TOTALS OBSERVED — including 0xE → 49152 twice (the pop_count4
missing-case fix WORKING; the old tree undercounts 0xe) — but delivery is
BOOT-PHASE-DEPENDENT: across three flashes of the SAME .bit the drain
worked / alternated per program / never fired. Sole surviving hypothesis:
the single-flag ignore_flush accounting (set by per_rd/zq/ref_init
maintenance, cleared by the next flush) can eat USER flushes depending on
maintenance phase — losing chunk AND trailer (21 recv timeouts/run in the
worst boot). aref-off does not silence it (per_rd_init is autonomous).
Long (write+read combined) programs fail ~always — consistent with
collision probability ∝ program length. The platform join-guard
(receiver_done + timed join, this session) held: every failure poisoned
cleanly, no process deadlocks; wedge hygiene = driver reload after any
unbalanced DIFF run (kernel rings hold stale chunks), and h2c can wedge
(errno 512) after repeated poisoned exits → warm reboot.

BUILD3 STAGING (vivado box, next revision): (1) debug-visibility first —
event counters for {user flush, eaten flush, per_rd_init, drain pulses,
accum FIFO writes} readable like dbg_rd_ctr, so the race is MEASURED not
inferred; (2) candidate fix: separate maintenance-ignore accounting from
user flushes (tag or counter+source), or latch drain_pending until
serviced; (3) keep POPCOUNT_ACCUM_MODE default-on (READ_MODE proven
unaffected). Road-B per-token number waits for build3; Road-A production
numbers (T4 tables) are unaffected.

# Addendum 20b — build3 sim verdict CORRECTS 20's hypothesis
Verilator (full engine, box-exact modules, FWFT FIFO stub): maintenance
DOES flush (flush = softmc_fin +32 cyc; maint programs share the
pipeline). The live mechanism is the MULTI-CYCLE FLUSH: softmc_end is
combinational, flush can be >1 cycle wide, and the original logic
eats-then-processes the SAME flush (and re-drains the accumulator every
flush cycle → the spurious zero chunks). Reproduces every silicon
signature. Second-order bug: an eaten flush leaves the sum armed — the
NEXT delivered total is silently DOUBLED (treat old DIFF-mode data from
bad boots accordingly). Fix (31/31 sim checks, READ_MODE byte-identical):
flush_edge + saturating ignore counter + accum_armed override (armed
flush always delivered; maintenance reads never arm). Debug counters on
the trailer beat (magic 0xDBC0DE01). Build3 running on the box.

# Addendum 20c — build3 on silicon: datapath PROVEN, drain-capture timing needs build4

build3 flashed and identity-verified (trailer magic 0xDBC0DE01). What the
campaign established on silicon:
- Compute datapath CORRECT: totals with exact expected values (incl. the
  pop_count4-fixed 0xE → 49152) observed repeatedly; READ_MODE untouched.
- Flush accounting FIXED and measured live: ~280 per_rd_init maintenance
  events fire between/during programs; the counter+armed-override
  accounting eats exactly the maintenance flushes (trailer counters:
  280/280 eaten, user drains delivered).
- Mode-toggle control word is RACY (frontend decode-state dependent):
  lost 1-of-2 to 4-of-4 times per boot. Host mitigation implemented and
  working: verified toggle (probe with an LDWD-armed write program via
  the new non-poisoning receiveDataTry; retry until observed mode
  matches). Also explains v1's tail-hang and several "boot-dependent"
  mysteries.
- REMAINING (the one blocker): drain-capture timing — the drain fires via
  the frontend flush pipe, which can lag into the NEXT program; the chunk
  then samples a disturbed/reset accumulator (zeros). Only the
  no-following-program-for-seconds pattern captured true totals. Delivery
  itself is fine (batched receives: 18/18 chunks, in order).
BUILD4 HDL item (small, surgical): fire the accum drain on the program's
OWN last-read/end-of-reads edge (or latch accum_out at flush-EDGE time
into a holding register that the c2h chunk reads), decoupling capture
from flush-pipe latency. Optionally: make the mode switch an
acknowledged/level control instead of a racy toggle. Then the suite
(paired, lag-tolerant, verified-toggle — all in test_popcount_hw.cpp)
should pass outright, and the server integration follows.
Host-side hardening landed this campaign (keep): receiveDataTry,
verified-toggle pattern, send-failure poison (no more SIGABRT wedges),
join-guard, pre-drain hygiene, trailer decode under PIM_RECV_DEBUG.

# Addendum 21 (Task O1) — per-subarray screened pools: true cross-calib voting restored, LOAD ceiling lifted 278 -> 572 rounds/bank, and the clone-dead-row discovery

2026-07-18/19, bender 2, 8K image. Goal: give the extra voting calibs
(subs 84/71/77/76, banner "extra calibs (dense clusters)") real
fault-screened pools in THEIR OWN subarrays — the 07-18 scoping fix had
left them poolless (voting degenerated toward temporal-primary sampling,
and the one surviving inside-window extra, sub 71, was sharing the
PRIMARY pool's rows: its voting scratch cycled over LOAD-resident rows).
Also: lift the 294-row LOAD ceiling by letting LOAD_WEIGHTS overflow into
the new pools.

## Method (May 2026-05-21 pool method recovered, replicated, extended)

The production 294-row pool's exact derivation was recovered by
reproduction: greedy independent set by ascending degree (tie-break
ascending row) over the undirected FAULT-edge conflict graph of the
May fault sweep — reproduces pool_layout_dimm2_bank{0-3}.txt
byte-for-byte from dimm2_fault_sweep/bank0.log. New campaign
(dimm2_fault_sweep_subs_2026_07_18/, 20 sweeps = 5 tuples x 4 banks,
n_warmup=4 src_step=1, ~23 s each, READ_MODE only) sweeps each extra
tuple over its REAL FindOpenRows window, with two env-gated additions to
fault-sweep-exe (default-off, method otherwise byte-identical):

- PIM_SCREEN_RW=1 — pattern + antipattern write/read screen per row
  (3,120 row-screens: 0 bad rows anywhere).
- PIM_CHECK_CLONE=1 — post-sweep isolated RowClone probe per source row
  (doubleACT 30/1 R -> Rfirst, read Rfirst, >=2046/2048 words = ok).

Sweep results (fault edges, cross-bank union; graphs bank-invariant to
<=2-edge fringes, unioned in): primary 3969 (May regression: 3950/3953
edges common, structure identical; production 294 still a valid IS under
the union graph — 0 violations), sub71 3955, sub76 4109, sub77 4168,
sub84 4109.

## THE discovery: clone-dead rows (deterministic in d = R xor Rfirst)

~17% of every window's rows CANNOT RowClone into their tuple's Rfirst —
match distribution vs source fingerprint: min ~400, median ~1140/2048
words (partial clones, not hard zeros). Deterministic per XOR distance d
(zero rows in both ok- and fail-sets), structured by the selection-law
unit count of d: all 6-unit cosets fail, 5-unit split (2^6-row
co-activation dilutes the clone charge below threshold). Consequences
for the EXISTING production stack, measured:

- 108/294 production pool rows are clone-dead for the primary tuple
  (Rfirst=45340) — 99/278 in the LOAD region: ~36% of LOAD-resident
  rounds have been computing on partial clones all along.
- 9/16 of the V2 scratch reserve tail (pool idx 278-293) are clone-dead.
  Silicon demonstration (overflow_regress driver, V2 d_in=2560 request):
  calib_idx=0 trip -> 2048/2048 segments wrong, systematically LOW
  (partial popcounts); calib_idx=1 and 2 trips (new per-sub pools,
  all-clone-ok) -> EXACT. This retro-explains T5's "unvoted full model
  loses Paris" (trip 0 rides the dead tail on most rounds) and why
  voting has been LOAD-BEARING.
- Production pool files stay untouched per task rules; the top follow-up
  is a clone-ok scratch annex for the primary trip (same construction as
  the sub71 pool, data already on disk in the campaign logs).

## New pools (pool_layout_dimm2_sub{S}_bank{B}.txt, banks byte-identical)

| sub | real window (s_id) | tuple Rf/Rs | rows | role |
|-----|--------------------|-------------|------|------|
| 84  | [54144,54784) s86  | 54150/54620 | 192  | voting + LOAD-overflow |
| 71  | [45312,45952) s72  | 45464/45857 | 26   | voting scratch-ONLY |
| 77  | [49152,49792) s78  | 49291/49719 | 77   | voting + LOAD-overflow |
| 76  | [49152,49792) s78  | 49178/49622 | 73   | voting + LOAD-overflow |

All rows: clone-ok on all 4 banks + RW-screened + IS discipline. s78's
two pools are one union-graph IS split between the tuples (zero
cross-edges). sub71 is special: the production 294 is a MAXIMAL IS of
s72, so a strict-rule disjoint pool is structurally impossible (measured
0 rows); the 26-row pool instead satisfies the scratch-sufficient rule —
no DIRECTED fault edge into any production row under the sub71 graph
(intra-pool edges permitted: V2 scratch is rewritten immediately before
use each round) — and is excluded from LOAD-overflow by the server's
primary-window-overlap guard. Files carry "# window <start> <end>"
(FindOpenRows windows are not 640-aligned; the /640 sub label is only a
cluster name — s72 spans labels 70+71, s78 spans 76+77).

The s86 window straddles the 1024-row predecoder block boundary at 54272
(addendum 11's cross-block scope): the clone probe shows NO
block-boundary clone cliff — block-53 sources clone fine into the
block-52 Rfirst; the fail set is the same selection-law unit-count
structure as everywhere else. (Addendum 11's cross-block no-deposit rule
evidently does not govern the t12=30 clone dst path.)

## Server plumbing (test_bitnet_server.cpp; no client changes)

- PIM_POOL_LIST_FILE_SUB env: path pattern with {sub} ( = open_rows[0]/640)
  and {bank} tokens. Extras (is_primary=false in build_backup_pool) load
  their OWN screened file, filtered to the file's window + their open
  set. Missing file = extra skipped (the 07-18 empty-return preserved);
  malformed file = fatal. Unset = legacy behavior bit-for-bit.
- Banner: "sub=84(pool=192@[54144,54784)) ..." per extra.
- LOAD-overflow (default ON, PIM_LOAD_OVERFLOW_SUBS=0 opts out): when
  the primary pool is full, LOAD allocates (round, bank) units from the
  extras' pools (in cs_extra order), each pool keeping a PIM_V2_SCRATCH
  tail reserve; primary-window-overlapping extras are never targets.
  LoadedHandle records per_round_calib_sel; MM3D emits each unit's body
  on its OWNING calib's tuple (per-unit Rfirst/Rsecond/open_rows — the
  builders were already per-unit), forces the plain body for any program
  containing a non-primary unit (fused stays validated-primary-only),
  and refreshes the extra windows as separate deduped ranges.
- V2 voting scratch for extras draws from their pools' tail reserve
  (residency-safe under overflow).

## Validation ladder

- Sim (PIM_BACKEND=sim, deterministic): 14 LOADs (overflow engages at
  the predicted handle), MM3D bit-exact on pure-primary / mixed /
  full-overflow handles, MM3D vote trips exact, V2 trips calib 0/1/2
  exact. 0 bad segments total.
- Silicon small-scale: same driver — all LOADs ok, MM3D EXACT including
  the full-overflow handle (weights resident in s78/s86 computed through
  the sub tuples' bodies), V2 calib 1/2 EXACT. (V2 calib 0 = the
  pre-existing dead-tail result above.) ab_fused_server.py A/B with the
  new env: baseline 9.6 ms / fused 6.5 ms mean per matmul, all_exact
  both arms — fused gate unaffected.
- Full model (the standing rule): exact T4 headline protocol + the new
  env (PIM_POOL_LIST_FILE_SUB), all defaults live (PIM_USE_LOAD_WEIGHTS=1,
  PIM_VOTE_FULL=1, PIM_FUSED_COSET=1, overflow ON). Command + logs:
  dimm2_fault_sweep_subs_2026_07_18/fullmodel_o1_fused{,_server}.log.

| config                                   | s/tok | output (8 tok) |
|------------------------------------------|-------|----------------|
| May 2048-image, non-fused                | 632   | ' Answer: Paris\nCopenhagen\nC' |
| T4 8K v3, non-fused                      | 438   | ' \nAnswer: Paris\n.. Paris' |
| T4 8K v3, fused (baseline)               | 360.8 | ' \nAnswer: Paris\nIn the United' |
| THIS: fused + per-sub pools + overflow   | 364.3 | **' Answer: Paris. The capital of France'** |

  - **Wall: 364.3 s/tok = +1.0% vs baseline — parity within the
    documented noise band** (T5's refresh A/B measured 364.1 vs 360.8 on
    identical configs). Expected, not disappointing: the overflow moved
    ~294 rounds/bank from V2 to LOAD ≈ 5.6% of the model's ~5.2k rounds
    × (1 − 1/1.45) ≈ 1.7% predicted saving — below the noise floor.
    Full-model wall harvest needs residency for the remaining ~4.6k
    rounds (≈9 more screened subarrays, a separate campaign).
  - **Text: the cleanest full-model output recorded on this rig** — both
    baselines wander after "Paris"; this run stays on-topic and
    grammatical to the last token. Cross-calib voting engaged with REAL
    extras (trip 1 = sub84, trip 2 = sub71, own pools, own subarrays).
  - Overflow engaged in production: 71 LOAD handles = 34 primary + 37
    overflow (148 bank-commits into sub84/77/76 pools), cursors monotone
    (no double allocation), then 466 clean ENOSPC acks → V2 fallback for
    the rest. Ceiling 278 → 572 rounds/bank, live.
  - **New instrumented observation (pre-existing-suspected, now
    quantified): LOAD-resident rows drift from load-time content under
    production traffic.** 1470 [mm3d-verify] round-0 prints, 70/71
    handles flagged, TWO signatures: sparse single-bit-per-segment
    (~27-65% segs, stable) and dense (~85-99%, mask-shaped) — the dense
    cohort is almost entirely the EARLIEST primary-pool handles (24
    dense: 23 primary, 1 overflow); the overflow handles in the NEW
    screened pools show only the sparse signature. Per-handle mismatch
    is saturated-stable across the run (handle 0: 7583→7575/8192 over
    42 verifies), and the OUTPUT was the cleanest ever — the model +
    voting absorb it. The client has carried a "backup-row data
    retention between handle-load and handle-use" known-issue comment
    since May (pim_linear.py, the original reason PIM_USE_LOAD_WEIGHTS
    once defaulted 0); this is the first full-model-scale measurement
    of it. Root cause open (candidate: reverse deposit into the backup
    row during its own RowClone use; discriminator: PIM_VERIFY_AT_MM3D
    full-pool probes + use-count correlation).

## Open items (ranked)

1. Primary V2 scratch annex: trip-0 V2 scratch rides the positional
   reserve tail, 9/16 of whose rows are clone-dead (the silicon calib-0
   2048/2048-low result). Fix = a clone-ok scratch annex for the primary
   tuple (sub70-style file, same construction as sub71's; data already
   in the campaign logs) + a primary-annex draw in v2_pool_idx. Pool
   files stay untouched.
2. Root-cause the resident-row drift (above). Note it caps LOAD-mode
   fidelity independently of pool screening.
3. Wall harvest: screen more subarrays (calib_dimm2 has only 3; a
   FindOpenRows + calibration pass on new subarrays would be needed)
   to push residency toward the full ~5.2k rounds.
4. Public-repo copies of server/client are behind (xbp cache bug, no
   per-sub plumbing) — sync at Phase P per repo_vision (user-gated).

## Files

- Campaign + derivation: /home/deni/Claude/dimm2_fault_sweep_subs_2026_07_18/
  (20 sweep logs + campaign.log + derive_sub_pools.py + fullmodel logs).
- 16 new pool files in the BitNet dir; production pool files untouched.
- test_fault_sweep.cpp: PIM_SCREEN_RW / PIM_CHECK_CLONE (default-off).
- Production env addition (RUNBOOK trio -> quartet):
  PIM_POOL_LIST_FILE_SUB='/home/deni/Claude/SiMRA-DRAM-main/DRAM-Bender/sources/apps/DSN_AE_APPS/BitNet/pool_layout_dimm2_sub{sub}_bank{bank}.txt'

# Addendum 22 (Task O8, part 1) — MM3D-entry refresh silently SKIPPED the overflow subarray windows (duplicate SoftMC labels); FIXED + sim-validated 2026-07-19

Found while staging the O8 drift arms: every server log with the
per-sub overflow pools engaged is full of SoftMC assembler warnings
"Trying to add label REFRESH_SUBARR_B<bank> multiple times!" (13140 /
4960 / 11120 lines in fullmodel_o1_fused_server.log /
o8_fullmodel_cloneok_unvoted / _voted server logs).

Root cause: since the 2026-07-18 per-sub extra pools (addendum 21),
build_refresh_subarray_loop_program() receives the SAME bank once per
refresh window (primary + each extra sub window), but its loop label
was keyed on bank only. Program::add_label (api/prog.cpp) OVERWRITES
the label on a duplicate add, so every branch for that bank resolves
to the bank's LAST window loop. Emitted-program trace: the primary
window still refreshes fully (it loops through the last window's
identical loop body with its own bounds), then execution falls
THROUGH to SMC_END — the extra windows' SMC_LI setup never runs.
Net effect on silicon (auto-refresh off): overflow-resident weights
in the extra per-sub windows got NO MM3D-entry refresh at all —
retention clock reset only by incidental traffic. Latent retention
hazard, and a covariate for every pre-fix overflow run: any run with
batch-B/overflow handles executed with extra-window refresh silently
absent (the drift analysis must treat pre/post-fix runs separately).

Fix (test_bitnet_server.cpp, software-only, no bitstream change):
loop label is now unique per ENTRY, not per bank —
"REFRESH_SUBARR_B{bank}_{bi}" (bi = entry index). bitnet-proj-server
rebuilt 2026-07-19 14:44.

Sim validation (PIM_BACKEND=sim; o8_drift_arms.py --arm none
--n-loads 80 --n-loads-b 220 — batch B drives the pool into overflow
so only pass 2 builds the multi-window refresh):
- pre-fix: 1200 warnings (300 pass-2 MM3D x 4 banks = 1 duplicate
  window per bank in sim); pass-2 refresh 1.3-1.5 ms = pass-1 (extra
  windows skipped, no extra work despite twice the windows).
- post-fix: 0 warnings; pass-2 refresh 2.8-3.1 ms ~= 2x pass-1
  (pass-1 unchanged ~1.4 ms) — the extra windows now execute.
- 0/300 y-mismatch in both runs (sim has no decay model; the
  functional harm is invisible to sim correctness — warning count +
  the pass-2 refresh-time jump are the discriminators).
Logs: dimm2_fault_sweep_subs_2026_07_18/o8_labelfix_sim_{prefix,postfix}.log
(driver) + the matching *_server.log pair.

Rebuild gotcha (tower): the Makefile's `-lboost_filesystem` no longer
resolves here (boost usage is header-only spsc_queue) and a FAILED
link deletes the previous binary — build with
`make bitnet-proj-server BOOST=` (also noted in RUNBOOK_TOWER.md).

# Addendum 22 (Task O8, part 2) — clone-dead is a LAW (held-out 2496/2496); voting cost recovered (137 s/tok Paris unvoted); the "drift" isolated to traffic-induced, coupling-ordered saturation + a retention term; capacity verdict for full residency

2026-07-19, bender 2, 8K image. Four sub-tasks on the O1 clone-check
data and pools. All silicon this session ran on the pre-fix binary for
the two full-model runs (extra-window refresh absent, part 1 covariate)
and on the FIXED 14:44 binary for every drift arm.

## (b) The clone-dead law — closed-form, held-out-validated 2496/2496

Data: the five O1 sweeps x 4 banks (12,480 CLONE observations) + a
fresh held-out sweep (sub85 tuple, Rf=54412, same physical s86 window,
identical calib line on all 4 banks; run_sub85.sh, ~23 s/bank).

- Outcome is deterministic per (tuple, row) across banks: 3,106/3,120
  (14 bank-marginal rows), and FULLY deterministic per (tuple, d_low),
  d_low = (R xor Rfirst) & 1023.
- CLOSED FORM for even-Rfirst tuples (groups G1..G4 = {1,2},{3,4},
  {5,6},{7,8}; u = [bit0] + #touched groups): clone into Rfirst is
  DEAD iff u >= 5, OR u == 4 with bit0 AND bit9 set and touched
  groups exactly {G1,G2,G3} (all 27 such classes exist, all dead; the
  81 siblings with another group untouched are all ok). Accuracy
  99.980% on the 9,984 even-Rf observations — the only 2 misses are
  the bank-marginal rows; zero false-dead.
- u <= 3 is clone-OK with ZERO exceptions anywhere (6,128/6,128,
  including the odd-Rf tuple).
- TWO failure families, not one dilution ladder: u5 fails at median
  match 1200/2048; the u4 bit0+bit9+{G1,G2,G3} family fails HARDER
  (median 620/2048) despite the nominally smaller coset — bit9 + the
  G4-vs-rest distinction encode something qualitatively different
  (physical section?), not one more halving.
- sub77 — the only ODD-Rfirst tuple — deviates BOTH directions (20
  u4-dead + 33 u5-ok of its own; the closed form scores 97.9% on it):
  the law is Rf-parity-conditional. Per-tuple determinism is still
  perfect, so odd anchors just need their own screen.
- Cross-1024-block sources obey the same law (zero d classes with
  in/cross-block verdict differences among even-Rf tuples).
- HELD-OUT TEST: predictions for all 624 sub85 rows committed to
  o8_sub85_clone_predictions.txt BEFORE the sweep; silicon returned
  516 ok / 108 dead exactly as predicted — 2,496/2,496 row-verdicts
  correct across 4 banks, including 40/40 in d classes never observed
  in the training tuples (pure closed-form fallback).
- WHY the May pool was 37% clone-dead: clone-dead rows have median
  fault-sweep degree 0 (vs 11 for clone-ok) — the same dilution that
  kills their clone hides them from the fault sweep, so the
  degree-ascending greedy IS PREFERENTIALLY selected clone-dead rows.
  The May construction anti-selected clone health.
- Reusable predicate: BitNet/clone_law.py (regression 12,478/12,480 =
  all minus the 2 marginals). Allocators can skip clone screening for
  even-Rf tuples on this die class entirely.

## (a) Clone-ok primary pool + the T5 redo: voting cost RECOVERED

pool_layout_dimm2_cloneok_bank{0-3}.txt (banks identical; production
files untouched): reproduction first — the production 294 is the
greedy ASCENDING-DEGREE IS over the May bank0 graph (byte-identical
reproduction; row-ascending order gives 178 and is NOT the method).
New pool: 181 LOAD rows (same construction, union graph = May +
2026-07-18 campaign x4 banks, candidates restricted to clone-ok on
all 4 banks) + 16-row V2 scratch tail by the sub71-style relaxed rule
(clone-ok, no directed fault edge into any IS row). Also excluded:
sub71's opens (8 of them are production LOAD rows today — every sub71
voting trip wrRow-writes activation over them; retired here), sub71's
scratch pool + its 13 directed deposit targets. Verified: 0 IS
intra-edges, 0 opens, 0 sub71 hazards, 197/197 clone-ok, law_dead()
false for every row.

Capacity tradeoff (anticipated): 181 LOAD rows vs production's 278
nominal — but only 179 of production's 278 are clone-ok, so
clone-CLEAN capacity is at parity (181 vs 179) while dropping 99
partial-clone rows; V2 tail 16/16 clone-ok vs production's 7/16.
Residency with the O1 extras: 181 + 294 = 475 rounds/bank (vs 572
mixed-health).

Full-model A/B (exact O1 headline protocol + per-sub pools env, only
--pool-layout and PIM_VOTE_FULL varied; 8-tok minimal-prefill):

| config                       | s/tok | output (8 tok) |
|------------------------------|-------|----------------|
| T5 unvoted, production pool  | 134.8 | garbage (the T5 verdict) |
| O1 voted, production pool    | 364.3 | ' Answer: Paris. The capital of France' |
| O8 UNVOTED, clone-ok pool    | **137.1** | **' \nAnswer: Paris. \n\nQuestion:'** |
| O8 voted, clone-ok pool      | 372.5 | ' \nAnswer: Paris. \n\nQuestion:' (identical) |

**The 2.7x voting cost is recoverable: unvoted on the clone-ok pool
retains Paris-class, on-topic, grammatical output at T5's unvoted
wall.** T5's garbage is retro-attributed to computing on clone-dead
rows (9/16 of the V2 scratch tail + 99/278 LOAD rows), not to any
intrinsic need for 3-trip voting. The voted clone-ok arm produced
BYTE-IDENTICAL text to the unvoted arm — at 8-tok scale voting adds
nothing once the pool is clone-clean. Both runs: 59 resident handles
= 181 primary + 291 overflow rounds/bank (both pools filled), 469
ENOSPC -> V2 slices, 0 recv-timeouts, 0 read-instability events.
Voted costs +2.3% vs O1 (372.5 vs 364.3) from the 97-round residency
drop. One-sample caveat: a single prompt at 8 tokens; longer-form
quality economics of voting remain open.

## (c) Drift isolation — traffic-induced, coupling-class-ordered,
## saturating; plus an independent retention term; NOT refresh, NOT
## voting, NOT V2-specific, NOT fused-specific

Host-side correlation first (o8_drift_correlation.py over O1's 1,488
verify observations + LOAD-cursor reconstruction): the two O1
signatures separate PERFECTLY on the round-0 row's clone class —
DENSE (92-99%, saturated-stable) <-> clone-OK rows; partial <->
clone-DEAD rows; overflow handles (clone-ok extras pools, plain body)
uniform ~32%. 0 read-stability failures in 7,467 verifies. Handles
0/1/2 verified CLEAN at first MM3D -> not a LOAD-time effect. The
sub71-open overlap explains none of the round-0 signal (0/23 dense
r0 rows are sub71 opens).

Full-model observations (clone-ok pool, this session): every handle
verifies clean at first use, then takes a ONE-SHOT transition and
stays flat (0.180 -> 0.180 over 7 verifies / 1,000 s). Same physical
row 45313: O1-config jump 92.5%, O8-unvoted 18.0%, O8-voted 32.6% —
config-dependent amplitude, zero dense class on the clone-ok pool.
New-1 bits confined to 0xff000000 in 1,512/1,540 unvoted
[mm3d-verify-bits] observations — the transition is byte-lane (one
x8 chip) localized single-bit flips.

Silicon arms (o8_drift_arms.py + run_o8_arms.sh, FIXED binary,
production pool, 80-300 one-round handles so round-0 verify covers
every row, y-EXACTNESS computed per handle per pass):

| arm | treatment | transitioned | pass-2 level (median) |
|-----|-----------|--------------|------------------------|
| R1  | nothing (LOAD -> verify -> verify) | 52/80 | 6904/8192 |
| R2  | + overflow engaged (300 handles)   | 52/80 (+47 batch-B) | 6908 |
| R3  | = R2 with PIM_REFRESH=0            | 52/80 (identical)   | 6905 |
| R4  | idle 300 s between passes          | **72/80, y-bad 80/80** | 6852 |
| R5  | + 50 sub71-trip V2s                | 52/80 | 6886 |
| R6  | + 200 fused MM3Ds on one handle    | 52/80 | 6881 |
| R7  | + 200 calib-0 V2s                  | 53/80 | 6883 |

- The verify pass ITSELF is the treatment: ~80-160 same-subarray MM3D
  bodies transition ~65% of rows; every added traffic flavor (V2
  either calib, fused or plain MM3D, 200x hammering) changes NOTHING
  — the effect SATURATES. Late handles are already y-broken at their
  FIRST use (19/80 at pass 1: their rows transitioned under earlier
  handles' bodies before first use).
- Saturation level is ordered by the row's selection-law coupling
  class: u3 rows -> median 6913/8192 (~84%, the dense class); u5
  (clone-dead) rows -> 2917 (~36%). The SAME coupling that makes a
  row a good clone source (the (b) law) makes it a strong deposit
  receiver under tuple traffic: one mechanism, two faces.
- REFRESH IS EXONERATED twice: R2 vs R3 identical on the fixed
  binary, and the pre-fix full models (primary window refreshed,
  extras not) show the same transitions on primary rows.
- RETENTION is a real, separate term: R4's 300 s idle (auto-refresh
  off) pushes 72/80 transitioned and 80/80 y-mismatch — decay on top
  of the traffic effect.
- Compute impact: REAL at exact-check granularity (60/80 handles
  y-mismatch after saturation at d_in=64 scale) while the FULL MODEL
  absorbs it (Paris, twice today, unvoted included) — BitNet
  quantization + popcount margins tolerate the flip density at the
  levels the production shape reaches (18-33%). This is May's
  "backup-row data retention between handle-load and handle-use"
  known-issue, now characterized and law-structured.
- No host-side allocator dodge exists (the pathway is same-subarray
  physics). Existing mitigation when exactness matters:
  PIM_LOAD_REWRITE_ON_MM3D=1 (per-round row rewrite — surrenders most
  of LOAD-mode's advantage). Report as an HDL/physics item; the
  practical production stance stands (clone-ok pool + the model's own
  robustness; voting NOT required for it).

## (d) Full-residency capacity — honest negative with numbers

calib_dimm2's ~96 tuples/bank span only THREE physical subarrays:
s72 (clusters 70/71), s78 (76/77), s86 (84/85). The one unscreened
cluster (85, 10-11 tuples/bank) was swept this session (bank-
invariant, ok=516/fail=108 x4 — the held-out data for (b)).

- s86 joint re-split (union graph sub84+sub85, s78-style): 133-row
  IS total -> LOAD 101 vs sub84-alone 176 — NET NEGATIVE (-75), the
  union graph is too dense. Not adopted.
- Subordinate sub85 annex (sub84 files untouched): 9 rows < the
  16-row tail reserve -> +0 LOAD rounds. Files
  pool_layout_dimm2_sub85_bank{0-3}.txt emitted for the record;
  inert under the 4-extra cap (sub85 ranks 5th by density; raising
  the cap buys nothing). Server LEFT UNTOUCHED by O8 (the only
  change this session is part 1's label fix by the parallel session).
- O1's extra pools re-checked with production (exclusion-only greedy)
  semantics: sub84 already optimal (192); s78 union would give 164 vs
  150 (+14 rounds/bank = noise; files left alone).
- VERDICT: max clean residency reachable from calib_dimm2 today =
  **475 rounds/bank ~= 9% of the ~5.2k the model needs**. The ~4.6k
  gap is NOT closable with the turnkey method on existing calibs —
  it needs the new-subarray campaign (FindOpenRows + MAJ calibration
  on fresh subarrays). The (b) law removes the clone-screening cost
  from that campaign (compute the predicate; sweep 23 s/bank for the
  fault graph), and the maximal-residency full model = the (a) voted
  run (475 resident rounds/bank, 372.5 s/tok, Paris).

## Pitfall found in passing (fixed)

The PRIMARY-pool loader reads pool files with fgets(line, 64):
header lines >= 64 chars split, and a split tail starting with a
digit parses as a row. My first clone-ok header did exactly that
(two "2026..." year fragments -> row 2026) — benign here (window
filter dropped them; fgets emulation confirms the effective pool ==
the intended 197 rows), but the files were rewritten with <64-char
headers + a warning, and the emitter now asserts the bound. The
per-sub loader uses line[128] (O1's longer headers are safe).

## Files

- Campaign dir dimm2_fault_sweep_subs_2026_07_18/: o8_clone_law_and_
  cloneok_pool.py (+ o8_sub85_clone_predictions.txt), run_sub85.sh +
  sub85_bank{0-3}.{log,err}, o8_derive_sub85_pool.py,
  o8_drift_correlation.py, o8_drift_arms.py + run_o8_arms.sh +
  o8_arm_R{1-7}*.log, run_o8_fullmodel.sh +
  o8_fullmodel_cloneok_{unvoted,voted}{,_server}.log.
- BitNet dir: pool_layout_dimm2_cloneok_bank{0-3}.txt (NEW),
  pool_layout_dimm2_sub85_bank{0-3}.txt (NEW, inert), clone_law.py
  (NEW). Production pool files, O1 sub-files, calib files untouched.

# Addendum 23 (Task O4 follow-up) — resident-consts DRIFT regression: first production-scale manifestation of the addendum-22 physics; FIXED by per-request const rewrite; Paris restored byte-identical; compound headline at the 137 s/tok mark

2026-07-19 (evening), bender 2, 8K image. Follow-up to the O4 session
(o4_silicon_2026_07_19_1454/): PIM_RESIDENT_CONSTS=1 was
protocol-validated 1.38x bit-exact, but the consts-only FULL MODEL
produced '1. I am a helpful AI assistant' where the features-off
control on the same binary produced Paris.

## Root cause — addendum 22 (c)'s drift, landing on the const rows

The resident ONE/ZERO rows are screened-pool rows (ZERO=45344,
ONE=45376 on all four banks) — deposit-safe by the pair-lattice law,
i.e. strong-coupling rows, the same u<=3 class addendum 22 measured
as the strongest deposit receivers (saturating ~84%). They were
written ONCE at startup and then sat resident while every fused
body's tuple traffic ran in the same subarray. Addendum 22:
same-subarray traffic transitions resident rows' CONTENT one-shot,
and the MM3D-entry ACT-refresh PRESERVES the drifted content — it
restores charge, not data. Weight rows are re-written per LOAD
slice; the const rows never were. So every fused body cloned
op[0]/op[2]/op[8] (and, via the coset broadcast from op[2], the
whole {2,6,10,14} zero coset) from progressively degraded ONE/ZERO
sources. Invisible to short protocol runs (fresh consts); fatal at
full-model traffic scale.

## Fix (test_bitnet_server.cpp; feature stays env-gated default-off)

Re-write the const rows at the START of every MM3D request, AFTER
the entry refresh (refresh restores charge of whatever the rows
hold; the rewrite then restores the CONTENT):
- rewrite_resident_const_rows(): ONE program per request — per
  const-bank, uniform-fill wrRow(ZERO,0x00000000)+wrRow(ONE,
  0xFFFFFFFF) via wrRow_immediate_label (the same write-driven
  primitive the non-consts body uses for these constants), SLEEP(8)
  + PRE trailer per bank, no receiveData. ~0.52 ms measured. NOT
  per_column_write_row (3 executes/row = ~24 round-trips/request).
- Single rewrite site in process_matmul_handle — serial, packed, and
  pack4-parallel builders all consume banks[].res_*_row after it.
  V2 path stays consts-free as before.
- PIM_CONSTS_REWRITE_EVERY=N: default 1 = every request; 0 = never
  (the pre-fix behavior, kept as the drift-probe diagnostic arm).
- [res-consts-rw] stderr lines (throttled); time folded into the
  prof's refresh bucket.
Sim (PIM_BACKEND=sim): fix arm exact with rewrites; EVERY=0 exact,
zero rewrites; consts-off run byte-inert (no consts lines).

## Silicon — protocol A/B + drift-specific 300-matmul probe

ab_fused_server.py, banks 0,1,2,3, d_in=256, bp=4, seed=1
(o4_constsfix_silicon_2026_07_19/):

| arm                            | matmuls | exact   | ms/matmul mean |
|--------------------------------|---------|---------|----------------|
| fused reference                | 8       | 8/8     | 6.1            |
| fused+consts, fix on           | 8       | 8/8     | 4.8            |
| LONG fused only (control)      | 300     | 300/300 | 6.2            |
| LONG consts, EVERY=0 (pre-fix) | 300     | 298/300 | 4.5            |
| LONG consts, fix on            | 300     | 299/300 | 4.5            |

- The consts win survives the fix: 4.8 vs 6.1 = 1.27x (1.38x
  without the rewrite; the ~0.5 ms rewrite is the delta).
- Drift MANIFESTS at protocol scale and is consts-specific: the
  no-consts control is clean over 300 while pre-fix consts drops 2
  matmuls (#19 s=1961 +1; #49 s=537 +1) and the fix drops 1 (#115
  s=537 +1) — single-count single-segment errors, the marginal tail
  of what is catastrophic at full-model scale. Honest residual: the
  per-request rewrite narrows the exposure window from whole-run to
  within-request; a mid-request transition can still touch that
  request's tail bodies (~0.3% of matmuls at +-1 count here; n too
  small to resolve EVERY=0 vs fix beyond the ordering).

## The decisive arm — consts-only full model on the fixed binary: PARIS

Exact isolation-run shape (production pool, chat-template 24-tok
prefill, voted default, PIM_USE_LOAD_WEIGHTS=1 PIM_FUSED_COSET=1
PIM_RESIDENT_CONSTS=1, 8 tok):

| binary            | output                                  | wall    |
|-------------------|-----------------------------------------|---------|
| pre-fix consts    | '1. I am a helpful AI assistant'        | 6021.1s |
| features-off ctrl | 'The capital of France is Paris. Paris' | 6064.3s |
| FIXED consts      | 'The capital of France is Paris. Paris' | 6060.9s |

BYTE-IDENTICAL to the features-off control at the control's wall;
the only delta vs the failing run is the rewrite -> root cause
CONFIRMED. 33,368 requests, 0 recv-timeouts, rewrites firing every
MM3D request. (Run 1 of this arm was killed at ~97/101 min by the
session's background-task reaper — client SIGKILLed externally, rig
verified healthy afterward with a 4/4-exact smoke, rerun detached
via setsid; artifacts of the killed run kept as *_run1_KILLED*.)

## Compound headline — clone-ok pool + unvoted + fused + consts

O8's exact 137.1 shape (pool_layout_dimm2_cloneok_bank{bank}.txt,
PIM_VOTE_FULL=0, PIM_NO_CHAT_TEMPLATE=1, per-sub extras env, 8 tok)
+ PIM_RESIDENT_CONSTS=1:

  ' \nAnswer: Paris. \n\nQuestion:'  — byte-identical to the O8
  reference — 8 tok in 1102.0 s = 137.8 s/tok vs the 137.1 mark
  (+0.5% ~= parity), 469 ENOSPC->V2 slices exactly matching O8's
  allocation trajectory, 0 timeouts.

Reading: consts are now SAFE at production scale but net ~zero at
this shape — the request is recv-dominated (mm3d-prof: recv 23.2 of
34.6 ms; exec 6.6 ms is what consts attack; the 469 V2 fallback
slices don't use consts at all). The 1.27x protocol-scale exec win
becomes real end-to-end only after the c2h/recv wall falls (the
in-DRAM accumulation track) or residency rises. Keep
PIM_RESIDENT_CONSTS as a validated, drift-hardened lever for
exec-bound shapes rather than a headline mover at this one.

## Files

- test_bitnet_server.cpp: consts_rewrite_every(),
  rewrite_resident_const_rows(), rewrite site + [res-consts-rw]
  logging in process_matmul_handle. Rebuilt bitnet-proj-server
  (make BOOST= now default-empty).
- o4_constsfix_silicon_2026_07_19/: env.txt, ab_fused.{out,log},
  ab_consts_fix.{out,log}, ab_long300_{noconsts,norewrite,fix}
  .{out,log}, ab_health_postkill.log, fullmodel_constsfix.out +
  _server.log, fullmodel_constsfix_run1_KILLED{.out,_server.log},
  run_compound_headline.sh + fullmodel_compound_cloneok_unvoted_
  consts{.log,_server.log}, compound_wrapper.log.

# Addendum 24 (Task O5) — dimm0 (bender 0) pool hygiene: the clone law holds on the SECOND die (2494/2496, zero false-dead); cloneok pools kill the baseline gate's systematic error; the residual is TWO other mechanisms, cleanly separated; dimm2-standard all_exact NOT reached on D0 → dual-DIMM full model not unlocked

2026-07-20 early morning, bender 0, 8K image, channels 0/1/3 PHY
full_reset earlier that day, bender 2 untouched throughout. Handoff
state: ab_fused_server.py --bender 0 --banks 0,1,2,3 with the May
pools (pool_layout_dimm0_bank{0-3}.txt) gave baseline 1635/2048,
fused 1980/2048 (logs ab_server_{baseline,fused}.log, 04:47).

## (1) Clone-law cross-die validation — the O8 held-out protocol on D0

Predictions for every window row committed BEFORE silicon
(o5_dimm0_clone_predictions.txt): 108/624 dead per tuple, law says
17.3% — the same d-class census as dimm2. Sweep: fault-sweep-exe 0 4
1 <bank>, PIM_SCREEN_RW=1 PIM_CHECK_CLONE=1, window [38400,39040),
per-bank primary tuples (banks 0/2/3 share Rf=38424; bank 1 its own
Rf=38446 — both even, law in scope). 4 banks in 87 s, rw_screen
0/624 bad everywhere.

- **Law accuracy 2494/2496 (99.920%)** — banks 2/3 PERFECT 624/624;
  the only 2 misses are bank-marginal (38445 bank 0 match=2045,
  38405 bank 1), BOTH pred_ok_but_dead: zero false-dead again.
  Measured dead 109/109/108/108.
- Bank-invariance is even stronger than dimm2's: banks 0/2/3 (same
  tuple) fault graphs BYTE-IDENTICAL (diff=0 across 4109 edges);
  bank 1 (own tuple) has its own graph (3926 edges).
- **The May pools are the anti-selection corollary in its purest
  form**: they are the "fault-free" (degree-0) row sets, and
  clone-dead rows have degree 0 — so the pools swallowed EVERY dead
  row of the window: 108/276 = 39.1% dead (banks 0/2/3), 109/332 =
  32.8% (bank 1). Worse than dimm2's greedy-IS 37%.

## (2) pool_layout_dimm0_cloneok_bank{0-3}.txt (existing files untouched)

Filter = MEASURED per-bank silicon verdicts (law was the predictor;
the two marginals prove the measured filter is the right one): kept
168/168/168 (banks 0/2/3) and 223 (bank 1); subset of the May pools
=> degree-0/IS discipline preserved; headers <64 chars (the O8
fgets(line,64) hazard). The gate-critical delta: bank 1 round-2 row
38405 — MEASURED dead, law-marginal — is dropped; banks 0/2/3's
first-4 (gate) rows were already clone-ok on silicon.

## (3) Gate rerun + the one-level-deeper diagnosis

d_in=256 bp=4 matmuls=8 seed=1, banks 0,1,2,3, cloneok pools env
PIM_POOL_LIST_FILE (o5_gate_* / o5_diag_* in the campaign dir):

| arm                                   | mean exact/2048 | ms/matmul |
|---------------------------------------|-----------------|-----------|
| May pools, baseline (handoff)         | 1635            | 9.5       |
| May pools, fused (handoff)            | 1980            | 5.9       |
| cloneok, baseline                     | **2035.6**      | 9.5       |
| cloneok, fused                        | 1995.0          | 5.9 (1.61x)|
| cloneok, baseline + VERIFY_AT_MM3D    | 2047.9          | (probed)  |
| cloneok, fused    + VERIFY_AT_MM3D    | 1995.8          | (probed)  |
| cloneok, PIM_FUSED_COSET=2 (layout, no cosets) | 1983.3 | 9.7      |
| cloneok, baseline + LOAD_REWRITE_ON_MM3D | **2047.6**   | 13.5      |

Clone hygiene recovered the baseline's systematic 400-seg error
(1635 -> 2035.6). The residuals are TWO DISTINCT mechanisms:

- **Baseline residual (~13 segs) = intra-request deposit
  accumulation on resident weight rows** (addendum 22 (c) physics at
  16-execute scale). Evidence, two independent levers: (i) observer
  effect — with PIM_VERIFY_AT_MM3D=1 full-pool probes between
  executes, EVERY probe reads 0/2048 differ and y goes exact
  (2048/2048 x7, one -1 flake); (ii) PIM_LOAD_REWRITE_ON_MM3D=1
  reaches the same floor (2048 x6, plus 2 matmuls with 1-2
  transient +1 single-count flakes at non-repeating segs = the
  intrinsic per-column flake floor, cf. o4's 1-2/300 on dimm2).
  Unobserved signature: bank 1 round-0 row 38401 stable 9/8192 segs
  (0.11%), first s=258 exp_pc=5 got_pc=4 xor=0x200, verify-bits
  OR=0x66000000 — byte-3 lane again (dimm2's drift OR was
  0xff000000; same x8-chip-lane class). Under the May pools this
  same mechanism read 324/8192 (3.96%) — the dead round-2 row's
  clone was the big depositor.
- **Fused residual (~52 segs) is NOT weight corruption — it is the
  fused OPERAND LAYOUT mis-computing on this die.** Evidence: (i)
  weight rows pristine under every probe while y errors persist
  UNCHANGED (per-matmul exact counts near-identical probed vs
  unprobed: 1981/1974/2014/2011/2006/2030/1971/1979 vs
  1981/1975/2020/2010/1999/2028/1973/1974 — a deterministic core
  set + small flicker margin); (ii) PIM_FUSED_COSET=2 (fused
  position layout via explicit wrRows, NO coset doubleACTs)
  reproduces the same failure (same recurring first-mismatch segs
  s=0/4/10, same ref values, same POSITIVE bias got>ref) — the
  coset broadcast is EXONERATED; (iii) errors are
  segment-concentrated and popcount-HIGH: marginal columns bias
  toward 1 under the fused role assignment. Checked and rejected:
  tuple-lattice geometry as the discriminator (dimm2's generators
  {1,2,96,896} contain bit9 content just like dimm0's
  {4,16,64,3584}; bank 1's {1,2,8,64} fails identically). Framing:
  the 07-17 fused validation on bender 0 (fused_final_b0.log) was
  per-MAJ protocol scale on the 2048 image; the server matvec shape
  had never been exact on D0 (the May colmask/calibration screened
  columns FOR the baseline layout; D0's weaker margins — May yield
  map D0 < D2 — surface layout-conditionally at matvec shape).

**Verdict: the dimm2 standard (all_exact both arms, no crutches) is
NOT reached on bender 0.** Baseline is one documented mitigation
away from the flake floor (rewrite, +42% ms/matmul); fused needs a
D0-specific column screen / geometry-aware role assignment for the
fused layout — a calibration campaign, out of O5 scope, and
per-column screening cannot be conjured from the existing May data
(it was layout-blind). Not brute-forced further per task rule.

## (4) Dual-DIMM full model — NOT run (conditional unmet)

Step 4 was gated on the gate passing; it did not. Ready-state for a
later green-light: dimm0 spec = calib_dimm0.txt + cloneok pools +
PIM_SUB_START=38400 PIM_SUB_END=39040 (+ rewrite if exactness
matters); bender 2's trained state untouched today; May's 1.47x
split precedent in multidimm_split_and_concurrency_wedge. Note the
full model absorbs FAR worse than 99.4%/97.4%-exact matmuls
(addendum 22 ran Paris on 18-33% drifted rows), so a dimm0-assisted
run is plausibly Paris-stable — but that is the user's call with
this gate verdict on the table, not a silent substitution for
all_exact.

## Files

- Campaign dir /home/deni/Claude/dimm0_fault_sweep_2026_07_20/:
  o5_predict_and_pools.py (predict/--compare/--emit-pools),
  o5_dimm0_clone_predictions.txt (committed pre-silicon),
  run_dimm0_sweep.sh + dimm0_bank{0-3}.{log,err} + sweep_wrapper.log,
  o5_gate_cloneok_{baseline,fused}{.out,_server.log},
  o5_diag_{baseline,fused,mode2,rewrite}{.out,_server.log}.
- BitNet dir: pool_layout_dimm0_cloneok_bank{0-3}.txt (NEW). May
  pools, calib files, server binary untouched (no rebuilds).

# Addendum 25 (Task O5 dual-DIMM) — the load-balance defect: V2 fallback was hardwired to servers[0]; fixed with a d_in-split V2 (concurrent partial sums); 47.5 s/tok / 36.8 s per token-matmul = 1.91x, at 96% of the ideal-halving bound

2026-07-20 morning, benders 0+2, 8K image, same env as the o5 dual
run (PIM_FUSED_COSET=1 + PIM_RESIDENT_CONSTS=1 + PIM_VOTE_FULL=0 +
PIM_NO_CHAT_TEMPLATE=1, cloneok pools both dies, DIMM_SPECS sub
windows). Client-only fix in /home/deni/bitnet_weights/pim_linear.py;
server binary, pools, calib untouched.

## The defect (why D2 got 4% of the o5 traffic)

The o5 dual-DIMM 24-tok run measured 90.3 s/tok — WORSE than the
80.5 s/tok single-DIMM baseline — with D0 = 15,510 calls / 23.5 GB
vs D2 = 628 calls / 51 MB. Mechanism (per-server stderr forensics):

1. Pool capacity bounds LOAD residency at 18 handles on D0
   (166-row cloneok pool: cursor 144 + rounds 8 + v2_scratch 16 >
   166) and 22 on D2 (195 rows). The full model wants 2,940
   sub-handles; 38-40 fit. Every LOAD past the bound gets the
   ENOSPC ack — by design (addendum 21's ceiling).
2. Client: ANY failed sub-LOAD correctly falls the slice back to
   V2 — but the V2 dispatch sent the full-d_in body to
   `self._server` = `self._servers[0]` UNCONDITIONALLY (a
   single-DIMM-era line the multi-DIMM plumbing never touched).
   So ~98% of matmul traffic pinned to DIMM 0; DIMM 2 served only
   its ~22 resident handles' MM3Ds plus ~1,250 fruitless LOAD
   probes (302/172 ENOSPC acks logged). o5's per-projection #req
   = tokens x n_slices exactly = the V2 signature; 23.5 GB = full
   weight masks re-sent every call.
3. May's "1.47x split" validation was layer-0-scale: everything
   fit the pools, all-LOAD, genuinely balanced. The defect only
   manifests at full-model scale where ENOSPC dominates — and
   single-DIMM production (80.5) is ALSO ~98% V2 (o2: 27,778
   calls / 41.9 GB), so the V2 path is THE production path and
   balancing it is the whole game. Defective-dual == single + dual
   overhead: 69.9 vs 70.2 s per token-matmul, identical.

## The fix (client-side, minimal, single-DIMM path untouched)

- **d_in-split V2** (`v2_parts`, built at init only when >1
  server): each server gets a contiguous chunk range of every V2
  slice (2560 -> 1280+1280, 6912 -> 3456+3456) as its own smaller
  V2 request; host adds the partial sums — the same y = SUM_sub
  math as the validated LOAD-mode split, so single-slice linears
  (k/v/down) balance too. Concurrent per-server threads
  (PIM_MULTIDIMM_SERIAL=1 serializes; PIM_V2_SPLIT=0 kill-switch
  restores legacy routing). Single-DIMM: v2_parts stays None →
  byte-identical legacy path.
- **ENOSPC latch**: first ENOSPC ack (byte 1) from a server stops
  further LOAD probes to it (server cursor never retreats). o5
  wasted ~1,250 probe round-trips; the balanced run logged exactly
  1 ENOSPC per server. LOAD→V2 fallbacks now logged (474/480
  slices fell back; the 6 layer-0 attention slices stayed
  LOAD-resident).
- Offline fake-server harness (test_balance_fix.py): bit-exact vs
  W@x on all 6 cases (single-DIMM V2/LOAD, dual mixed-ENOSPC,
  dual pure-V2, replication+VOTE_FULL, 216-chunk split); latch and
  kill-switch behaviors verified before touching silicon.

## Layer-0 silicon A/B (--layers 0, 4 tok, PIM_DIFF_LOG=1)

| arm | wall (4 tok) | request | D0 calls/MB | D2 calls/MB |
|-----|-------------|---------|-------------|-------------|
| A: PIM_V2_SPLIT=0 (o5 routing) | 23.8 s | 20.48 s | 305 / 198.7 | 174 / 5.3 |
| B: split (fix) | 12.9 s (1.84x) | 11.80 s (1.74x) | 305 / 101.9 | 284 / 102.1 |

Arm A per-slice V2 walls (D0): gate/up 120.7/120.8 ms, down 324.5
ms — identical to the o5 full-model profile (faithful miniature).
Arm B: 61.7/61.9/163.4 ms = 1.96-1.99x per slice; bytes equal
across dies to 0.2%. LOAD-mode attention walls unchanged (q 896.8
-> 908.3 ms), as designed. Diff-match A vs B: q 99.61/99.61, k
100/100, v 100/100, o 100/100, gate 89.90/90.87, up 91.83/92.33,
down 98.20/98.95 % — B equal or slightly better everywhere (half
of every V2 slice now computes on D2, the cleaner die; the D0
V2/fused residual of addendum 24 §3 stays on D0's half). No
correctness regression.

## 24-token full model, balanced (o5fix_dualdimm_24tok_balanced.log)

- **24 tok in 1140.6 s = 47.5 s/tok** (o5 defective: 2167.0 s =
  90.3; single-DIMM o2 baseline: 80.5).
- Text BYTE-IDENTICAL to o5's: ' \nAnswer: Paris. \n\nQuestion:
  What is the capital of France? \nAnswer: Paris. \n\nQuestion:
  What' — Paris-stable.
- **Balance: D0 = 15,209 calls / 11,731.1 MB, D2 = 15,151 calls /
  11,732.1 MB** — bytes within 0.01%, calls within 0.4% (server
  prof-line counts match too). The starvation is gone.
- Steady-state per-layer request walls (L16, vs o5): q 3897 vs
  7484, k 1967 vs 3752, v 1976 vs 3775, o 3887 vs 7423, gate 7666
  vs 14840, up 7693 vs 14813, down 10230 vs 19876 ms — 1.91-1.94x
  everywhere. Request total 2144.7 -> 1116.2 s = 1.921x.

## Verdict vs the expectation and the bound

- **Per token-matmul (prefill-normalized, the honest comparison):
  single 70.2 s (3862.7/55), defective dual 69.9 (2167.0/31),
  balanced dual 36.8 (1140.6/31) = 1.91x vs single.** The plain
  s/tok ratio is 80.5/47.5 = 1.69x, understated because the
  24-tok run amortizes its 8-token prefill over fewer tokens than
  the 48-tok baseline (31/24 vs 55/48 matmuls/token).
- Ideal-halving bound on the o5 shape (request/2 + host-serial
  unchanged): 1093.7 s = 45.6 s/tok. Measured 1140.6 s = 47.5 →
  **96% of bound**. Residual: LOAD-mode layer-0 slices (walls
  unchanged), ~1-4% per-half fixed server cost, thread orchestration.
- May's 1.47x expectation: EXCEEDED (1.91x per-matmul; even the
  raw 1.69x s/tok beats it). The correct mental model was never
  "LOAD-split gives 1.47x" — it is "V2 is 98% of production
  traffic; split V2 and the whole request wall halves."

## Open items

- **mm3d-verify readback artifact (pre-existing, NOT the fix's)**:
  one deterministic handle per die (D0 h15 = v_proj sub, 1920/8192
  segs; D2 h3 = q sub, 7842/8192, exp 0x92082010 got 0x00000000)
  reports DECAY/CORRUPTION on every MM3D, byte-identical across
  runs and arms — yet v matches 100.000% and q 99.61% at the
  output, so the MAJ3 compute is fine and the verify reads the
  wrong row (addendum-22 label/window class, verify-path only).
  Diagnose the verify addressing when convenient; it is noise in
  every server log until then.
- Pool capacity (18+22 handles) is now the LOAD-residency limiter;
  addendum 21's per-subarray extra pools + addendum 22's capacity
  verdict are the lever if LOAD coverage should grow beyond
  layer-0 attention. The split-V2 result reduces the urgency: V2
  at 36.8 s/matmul is within 4% of the dual-die bound.
- 3-DIMM/4-DIMM: the split generalizes (contiguous n_srv ranges);
  D1/D3 are storage-role dies (addendum 9 / dimm13 verdict), so
  only if a third compute-grade die appears.
- server-time-implied in the client summary is garbage for >1
  server (subtracts per-server pipe times from the GLOBAL request
  total); per-server calls/bytes/pipe-read are the real balance
  metrics. Cosmetic, not fixed.

## Files

- /home/deni/bitnet_weights/pim_linear.py — v2_parts build +
  split dispatch + ENOSPC latch + fallback logging (multi-DIMM
  gated; single-DIMM byte-identical; PIM_V2_SPLIT=0 reverts).
- Scratchpad logs: o5fix_l0_armA_legacy.log, o5fix_l0_armB_split.log,
  o5fix_dualdimm_24tok_balanced.log, o5fix_balanced_server_b{0,2}.log
  (snapshots), test_balance_fix.py, addendum25_draft.md.
- o5 forensics source: o5_dualdimm_24tok.log (+ the since-truncated
  /tmp/pim_server_b{0,2}_0_1_2_3.log, quoted above).

# Addendum 20d (2026-07-20) — build4: drain-capture timing fixed at the RTL root; sim-verified, Vivado build launched

Continues the 20/20b/20c readback thread. Agent flow mirrored build3:
sim-first on the vivado box, then the box build.

## Root cause pinned in RTL (why 20c's captures went wrong)

flush = frontend_ready = softmc_fin delayed 32 cycles, and softmc_fin
fires when the END word is FETCHED (fetch_stage.v: softmc_end = is_end
&& valid_in) — while already-fetched READ commands are still queued in
decode/exe/ddr_pipeline and their DATA returns through the PHY later
still. The 32-cycle delay is a settle heuristic, not a bound: when the
fin->last-rd_valid gap exceeds it (deep command queue on long programs,
pacing shifts when batched), build3 drained a partial or EMPTY
accumulator at flush_edge and the tail reads leaked into the NEXT
program's total. This one mechanism reproduces ALL three silicon
signatures: all-zero captures (whole read tail after the flush), the
one-program delivery lag (each flush drains the PREVIOUS program's
late-arriving total), and fail-rate ∝ program length (addendum 20).
Note build3's popcount_accum already REGISTERS the total at drain — a
holding register latched at flush_EDGE (20c's simpler option) would
capture the same wrong instant, so build4 moves the capture itself.

## The build4 fix (projects/BCU1525_QUAD/verilog/readback_engine.v)

- rd_outstanding_r: announced-minus-returned read counter fed by
  read_seq_incoming/incoming_reads — the exact SMC_INFO-derived signals
  buffer_space accounting already consumes; no new frontend plumbing.
  Floor-at-0 absorbs un-announced maintenance per_rd returns (frontend
  serializes programs; PHY returns in CAS order).
- flush_proc (build3's armed/counter accounting, UNCHANGED) now sets
  capture_pending instead of draining; the drain fires on the first
  QUIET cycle (outstanding==0 and the rd_valid->diff_valid->
  pop_count_valid pipe empty). Quiet at the flush edge (paced case) =
  same-cycle fire, cycle-identical to build3.
- Trailer waits for the capture (chunk-then-trailer framing preserved);
  exactly one capture per processed flush; 4096-cycle pending-age valve
  force-fires to keep c2h framing if read returns are ever lost.
- Mode-switch hardening: frontend.v decodes two NEW idempotent control
  words — SET-READ = bit INSTR_WIDTH+5 (byte8=0x20), SET-DIFF =
  INSTR_WIDTH+6 (byte8=0x40); +4 avoided (HBM_BENDER temp-read). SET
  words return the FSM to IDLE_S (aref-word pattern — no INIT_MEM_S
  camping). Legacy +1 toggle kept. softmc_core.v wires the two pulses.
- Trailer magic 0xDBC0DE01 -> 0xDBC0DE02 (host identifies build4);
  counter layout unchanged.
- READ_MODE fully ifdef-isolated: pending can only set in DIFF mode.

## Verilator verdict (box readback_race_sim, harness extended)

b3 = flashed build3 RTL, b4 = build4. Suite: (a)-(e) build3 regression;
(f1) partial read tail after flush; (f2) full tail after flush; (g)
batch of 3 tail-crossing programs; (h) SET-word idempotence + legacy
toggle; (i) maintenance read inside the deferral window.

| scenario | b3 (flashed image) | b4 |
|---|---|---|
| (a)-(d) paced accounting suite | PASS | PASS, same counters |
| (e) READ_MODE beat stream      | dump | BIT-IDENTICAL to b3 |
| (f1) partial tail              | BUG shown: 15 then 33 (leak) | 24, 24 exact |
| (f2) full tail                 | BUG shown: 0 then 48 (zero+double) | 24, 24 exact |
| (g) batched x3                 | BUG shown: lag 0,T1,T2 | T1,T2,T3 in order |
| (h) SET words                  | n/a | idempotent; toggle intact |
| (i) maint-in-deferral          | n/a | capture exact, floor holds |

b4 53/53 checks, b3 40/40 (incl. the three bug demonstrations) — the
sim reproduces every silicon signature on b3 and clears all on b4.
One pre-existing (build1-era) exposure documented, NOT changed: the
ignore_read clear can be preempted when per_rd_init lands within a
cycle of a read return (scenario (i) accumulates the maintenance read
— same behavior on build3).

## Host side (tower, landed + rebuilt)

platform.{h,cpp}: set_readback_mode(bool diff) using the SET words,
documented build4+ ONLY (on older images the words fall through decode
into the instruction path). toggle_readback_mode() kept; the
verified-toggle probes in test_popcount_hw.cpp remain valid on build4
and converge in one attempt. Rebuilt clean: bitnet-proj-server,
popcount-hw-exe (api objs purged first; BOOST default-empty).

## Box build (launched, running at write time)

Files installed with .bak chain (readback_engine.v.build3.bak,
softmc_core.v.build3.bak, frontend.v.bak) + staged in
incoming_popcount_2026_07_17/ as *_build4.v. build3 bitstream
preserved: popcountC_backup_20260718/bcu1525_quad_top_imem_popcount_2
.bit (md5 in md5.txt). Launched build.sh -> nohup, log
popcountD_build4.log; past create_project with 0 errors, ip_synth
running, 34G free. Poll:
  ssh daniel@100.117.150.4 'bash /home/daniel/Claude/bcu1525/popcountB_poll.sh /home/daniel/Claude/bcu1525/popcountD_build4.log'
Output on success:
  .../projects/BCU1525_QUAD/BCU1525_QUAD.runs/impl_1/bcu1525_quad_top.bit
md5sum it after completion; on silicon build4 identifies by trailer
magic 0xDBC0DE02. Then the 20c suite (paired, lag-tolerant,
verified-toggle) should pass outright — and batched all-execute-then-
receive, 20c's killer pattern, is the decisive new arm.

# Addendum 20e (2026-07-20) — build4 on silicon: datapath PERFECT, but the c2h TRAILER framing is FWFT-latency-broken; root-caused, build5 launched

build4 flashed + host rebooted; image confirmed build4 (trailer magic
0xDBC0DE02, `popcount_accum`/`rd_outstanding`/`capture_pending` present).
READ path perfect (matvec-smoke K=6/K=20 ALL_PASS). The DIFF/accum tool
(`popcount-hw-exe`) returned ZERO totals. This addendum supersedes 20d's
premise: the drain-capture *timing* build4 rebuilt was NOT the silicon
bug — the drain fires correctly at every program's own flush; the fault is
one layer down, in how the trailer beat is sequenced onto c2h.

## Trailer counter layout (decoded from readback_engine.v, verified live)
c2h trailer beat = `{cnt_accum_write, cnt_drain, cnt_flush_eaten,
cnt_flush_edge, cnt_ref_init, cnt_zq_init, cnt_rd_init, 32'hMAGIC}` (MSB→LSB
in the Verilog concat). On the wire / under `PIM_RECV_DEBUG=1` (word order):
`MAGIC  rd_init  zq_init  ref_init  flush_edge  flush_eaten  drain
accum_write`. Meaning: rd/zq/ref_init = maintenance-init edges; flush_edge =
all frontend flush rising edges; flush_eaten = flushes consumed by the
maintenance ignore-counter; **drain = accumulator captures; accum_write =
chunks written to the readback FIFO.** All cumulative from reset.

## What the counters MEASURED (bender 2, this boot)
1. `per_rd_init` fires **~1 M/s** (idle maintenance): `rd_init` jumped
   1126→3,013,225 across a single 3 s host wait. `flush_edge − flush_eaten`
   stays tiny (≈1–5 total) — under this storm the ignore-counter is almost
   always ≥1, so a no-read (bare-write) DIFF program's flush is EATEN and
   emits nothing. (The old tool assumed writes emit a zero chunk → its
   "3 chunks/case, match-by-value" framing mis-read the stream → the ZERO
   totals. Not the datapath.)
2. **Every DIFF program that CONTAINS READS drains at its OWN flush**:
   `cnt_drain`/`cnt_accum_write` increment exactly 1:1 with read programs
   (probe `lag`/`count` modes). The `accum_armed` override makes read-program
   flushes immune to the maintenance ctr race.
3. **Totals are EXACT and in-place**: 0xE→49152, 0xFF→16384, all-ones→65536,
   … verified bit-exact. The pop_count4 0xE fix is working. **The compute
   datapath, accumulation, deferred capture, and drain are all correct.**

## Root cause (RTL) — the trailer is emitted BEFORE its own chunk
Per-program c2h delivery (`PIM_RECV_DEBUG`): prog 0 delivers a **32 B
trailer ALONE** (drain already fired, chunk withheld); progs 1..7 deliver
96 B = **the PREVIOUS program's chunk + this program's trailer**. So the
accum message is offset by one program: chunk of prog i is pushed out only
by prog i+1's c2h traffic. After ~7–8 programs the host drain thread
(`consumeData`, one per `execute()`, joined before the next send) blocks on
the withheld chunk >15 s and the c2h wedges (needs `fpga-helper pci-reset`
to clear). Independent of read count (COLS 128/64/32 → identical 7-chunk /
prog-9 wedge) and unchanged by an SMC_SLEEP-before-END (so NOT a late-read
tail, NOT buffer_space draining — both hypotheses tested and killed on
silicon).

The mechanism: `trailer_beat` (and the tlast that closes the message) gate
on **`rbf_empty`**. The accum drain writes ONE 512 b word (= 2×256 b c2h
beats) into `rdback_fifo`, the Xilinx **First-Word-Fall-Through** IP. FWFT
has multi-cycle fall-through latency: `empty` stays HIGH for several cycles
after `wr_en` before the word surfaces at `dout`. `trailer_beat` fires
inside that window — BEFORE the chunk is readable — so the trailer (with
tlast) goes out first and the chunk is stranded in the FIFO until the next
program clocks it out. READ_MODE never shows this because its 8 KB payload
keeps `rbf` non-empty across the whole message.

### The sim gap that hid it
`readback_race_sim` compiles the **real** engine but stubs the FIFO with a
behavioral `rdback_fifo_sim.v` that has **zero fall-through latency**
(`empty`/`valid`/`dout` update the same cycle as the write). With that stub
the trailer correctly waits and build4 passes 53/53. Adding a realistic
FWFT fill-latency to the stub (`rdback_fifo_sim_filllatency.v`, FILL_LAT=3)
**reproduces the bug: build4 drops to 13 FAILs** (message ≠ [chunk,chunk,
trailer]; totals not in their message; f1/f2/g delivery scenarios all
break). build3 fails identically — same trailer logic — so **rollback to
build3 would NOT help.**

## Fix — build5 (readback_engine.v; magic 0xDBC0DE02 → 0xDBC0DE03)
Minimal, accum-only, READ_MODE bit-identical. Gate the trailer on the
chunk's two c2h beats having actually LEFT the FIFO, not on `rbf_empty`:
- new `reg [1:0] chunk_beats_r`: set to 2 at the drain (`dsr_valid`),
  decremented per `fifo_valid && c2h_tready_0` (each chunk beat out);
- `trailer_beat` and the proc_flush tlast now also require
  `chunk_beats_r == 0`.
This is robust to any FWFT latency (counts real transfers). READ_MODE never
sets `chunk_beats_r` (no `dsr_valid`), so its framing is untouched.

### Sim validation (box `readback_race_sim/validate_build5.sh`, 2×2 matrix)
| FIFO model | build4 | build5 |
|---|---|---|
| 0-latency stub (original) | ALL PASS | ALL PASS |
| FWFT fill-latency (realistic) | **FAIL (13)** | **ALL PASS (0)** |
READ_MODE data-beat dumps `b4`≡`b5` bit-identical. The fill-latency stub is
now a permanent harness artifact (the sim modeled the FIFO too ideally —
lesson logged).

### Tool side (tower, landed in test_popcount_hw.cpp)
- switched to `platform.set_readback_mode(bool)` (idempotent build4 SET
  word) instead of the racy toggle+probe (the old probe used a no-read
  write, which under the maintenance storm emits nothing → could not detect
  DIFF);
- suite rewritten to interleaved execute + DRAIN-ALL consumption with
  trailing read-kickers, matching totals IN ORDER (no more 3-chunk write/
  read/kicker assumption). New diagnostic modes: `root` (sleep A/B), `lag`
  (localizes drain vs delivery), `count` (wedge-vs-readcount), `recover`
  (set READ + bounded drain).
- On build4 this now reads the CORRECT totals for the first 7 cases
  (49152/49152/65536/24576/40960/65536, 0xE included) then hits the
  one-program-lag delivery wedge — the tool is correct; build4's c2h
  framing is the ceiling. On **build5** (delivery fixed → each message
  self-contained, no lag, no wedge) the full 9-case suite + toggle-back
  sanity should pass outright; re-run after the flash to confirm ALL_PASS.

## build5 status + how the user identifies/flashes it
Launched on the vivado box exactly like build3/4 (build4 engine saved
`readback_engine.v.build4.bak`; build4 bitstream backed up
`popcountD_backup_20260720/bcu1525_quad_top_imem_popcount_3.bit`,
md5 8510b7479233f458b1bbb121789068d6; build5 staged
`incoming_popcount_2026_07_17/readback_engine_build5.v`). `nohup bash
build.sh`, log `popcountE_build5.log`; verified past create_project into
ip_synth with **0 errors**. Not waited on (~2 h). Poll:
`ssh daniel@100.117.150.4 'bash /home/daniel/Claude/bcu1525/popcountB_poll.sh
/home/daniel/Claude/bcu1525/popcountE_build5.log'` — success prints
`.../BCU1525_QUAD.runs/impl_1/bcu1525_quad_top.bit`; md5sum it. On silicon
**build5 identifies by trailer magic 0xDBC0DE03**. Flash + bring-up per
RUNBOOK "After a LIVE JTAG reprogram" (remove+rescan or warm reboot; NEVER
pci-reset/cold-cycle post-JTAG); set `BITSTREAM_IMEM=8192`. Then re-run
`popcount-hw-exe 2 0 60000` for the 9-case ALL_PASS.

## Meanwhile / rollback
Tower stays on **build4** — production is READ_MODE (Road A) which is
perfect and untouched by this bug; the accum/DIFF feature (Road B, in-DRAM
popcount) is what's blocked, and it has never been used in production. **Do
NOT roll back to build3** (identical trailer bug, and it lacks build4's SET
words + deferred capture). After any DIFF-mode experiment the c2h can wedge
— clear with `fpga-helper unload → pci-reset → load` + full_reset ×4 +
rowclone smoke (validated here; READ path confirmed clean afterward).

# Addendum 26 (Task O10) — dimm0 fused-layout column screen: the residual is CONTENT-conditional per column; colmask + host-repair lands 1995 -> 2047.5/2048 (layout mechanism repaired), but all_exact is blocked by an intrinsic flake floor and the mask does not generalize across weight content

2026-07-20 midday, bender 0 (+ one dual-DIMM full model on 0+2), 8K
image. Follow-up to addendum 24 (3): build the layout-aware column
screen the May calibration never was, derive a D0 fused colmask, wire
it into the server, re-run the O5 gate.

## The screen tool (fused-colmask-exe, test_fused_colmask.cpp)

SHAPE FIDELITY IS THE LOAD-BEARING DETAIL: a first single-body-per-
program version (the test_fused_maj shape that was EXACT on b0 on
07-17) shows ZERO bad columns — the residual does not exist outside
the server emission shape. The tool therefore replicates the gate's
per-(round, bitplane) emission exactly — one program = 4 banks'
combined bodies (serial multibank builder, K=1, no consts), one
receiveData(4x8192), per-bank calibs (banks 0/2/3 = Rf 38424 tuple,
bank 1 = Rf 38446), backup rows from the cloneok pools rotated
LOAD-style, MM3D-entry subarray refresh every 16 programs — with ONLY
the step-3 layout parameterized. W draws include gate-shaped sparse
ternary masks (~25% / ~6% ones) + dense randoms + structured; x
uniform per row with the gate's bank-pair split; expected = W[s] & x.

## Screen results (14 W x 12 x, trials 2, fused x3 depth; 4 banks)

| layout   | what it isolates             | bad cells/run | bad cols b0/b1/b2/b3 | bias |
|----------|------------------------------|---------------|----------------------|------|
| base     | production interleaved       | 19.3          | 72/199/80/103        | hi 2.6:1 |
| fusedwr  | fused positions, wrRows only | 8.1           | 65/94/93/100         | lo 6:1 |
| fused    | production fused (cosets)    | 7.6           | 67/85/57/90          | lo 4.7:1 |
| mirror   | fused cosets, x<->0 swapped  | 10.9          | 106/126/108/135      | hi 1.3:1 |
| altfused | cosets {1,3,9,11}/{4,6,12,14}| 6.6           | 49/15/49/88          | lo 3.4:1 |
| hybA     | base + x@{4,5} adjacent pair | **42.5**      | **289/14/383/333**   | hi 3.3:1 |
| hybB     | base + swap(7,9) (W@6,7 adj) | 6.5           | 74/7/53/57           | lo 4.4:1 |
| hybC     | base + swap(6,11) (z@5,6 adj)| 7.5           | 103/8/119/132        | lo 1.8:1 |

- NOTE the screen's base number is an upper bound mixing mechanisms
  (x=0 margin probes + per-W-draw backup hammering) — screen numbers
  compare layouts within the same harness, they are NOT gate residuals.
- **hybA is the positional attribution**: the single role-pair swap
  that gives x the physically-adjacent slot pair {4,5} — the pair the
  fused layout also has — is 6.5x worse than every other single swap,
  popcount-HIGH (hi 10970 / lo 3337), and tuple-class-dependent:
  catastrophic on the {4,16,64,3584}-generator tuple (banks 0/2/3),
  nearly absent on bank 1's {1,2,8,64} tuple (14 cols). The fused
  layout's milder residual carries the same x@{4,5} pair.
- altfused (x@{1,3,9,11}+2, 0@{4,6,12,14}+8, W@{5,7,10,13,15}) is the
  mildest coset-implementable variant measured (bank1 15 cols) — a
  candidate replacement layout, NOT productionized in O10.
- NO column is deterministic (0 cols fail >=50% of runs): the failure
  is CONTENT-conditional per column — a (column, W, x) property, not a
  fixed bad-column set. This shapes everything below.

## Colmask + host repair (server PIM_FUSED_COLMASK_FILE, {bank} token)

fused_colmask_dimm0_bank{0-3}.txt lists fused-GOOD columns; the
complement is HOST-REPAIRED at the three accumulate sites (V2 /
MM3D unpacked / MM3D packed): pc[j] := popcount(mask[j] & x) — exact
by definition (MAJ3(W,x,0) = W&x), applied ONLY to bodies that ran the
fused layout (mode 1/3, all-primary program). Repairing a correct
column is a no-op, so the mask size is the honest "not computed
in-DRAM" fraction, and union-growing the mask is safe. LOAD keeps
full masks when a colmask is active (repair needs them at MM3D).
Sim: dimm2 fused all_exact with and without a synthetic mask (LOAD
and V2 paths; v2_repair_sim_test.py). NOTE: the sim cannot model
dimm0 fused at all (0/2048 even colmask-less — sim fused supports
the dimm2 lattice only; silicon says 1995).

## Gate ladder (O5 protocol: d_in=256 bp=4 seed=1, banks 0-3, cloneok pools)

| arm                          | mean exact | min  | fully-exact matmuls | repair cols/bank |
|------------------------------|-----------|------|---------------------|-------------------|
| fused, no colmask (control)  | 1994.9    | 1970 | 0/8                 | 0 |
| v1 = screen-derived mask     | 2017.6    | 1995 | 0/8                 | 67/85/57/90 |
| v2 = v1 + gate-seed1 union   | 2047.1    | 2045 | 3/8                 | 168/188/154/187 |
| v3 = v2 + 14 more            | 2047.2/2047.8 | 2046/2047 | 4-6/8      | 181/202/168/199 |
| v3 + LOAD_REWRITE            | 2047.2/2047.5 | 2045/2046 | 4-6/8      | (same) |
| v3, HELD-OUT seed 2          | 2031.2    | 2023 | 0/8                 | (same) |

ms/matmul 6.1 -> 6.2 = repair cost at gate scale is noise.

## Verdict (the definitive form of addendum 24's)

1. **The layout mechanism is REPAIRED**: colmask v3 removes the
   deterministic content-conditional core (bad observations per
   8-matmul gate: 425 no-mask -> 243 screen-mask -> 2-6 at v3;
   byte-identical reproduction of survivors confirmed before each
   augmentation step).
2. **all_exact is NOT reached, and cannot be, by a static colmask**:
   (a) beneath the layout mechanism sits an intrinsic stochastic
   flake floor — ~0.5-0.9 obs/matmul, fresh effectively-random
   columns every identical rerun, +-1-count at various bitplane
   factors, PIM_LOAD_REWRITE_ON_MM3D-IMMUNE (so NOT the addendum-22
   deposit/drift mechanism) — the same class as the D0 baseline+
   rewrite floor of addendum 24 (2047.6, transient flakes) at a
   somewhat higher rate; (b) the content-conditional core does not
   generalize across weight content — held-out seed 2 excites 50
   distinct segments of which only 4 were previously seen (v3 still
   recovers it 1994.9-class -> 2031.2). A mask guaranteeing all_exact
   for arbitrary content would have to grow toward full host compute.
3. The dimm2 standard (all_exact both arms, no crutches) therefore
   remains D0-unreachable — now with the mechanism split measured:
   layout core (repairable, repaired) vs intrinsic flake floor
   (die-class property, not repairable by masking).

## Dual-DIMM full model with D0 colmask (the one production run)

Protocol = addendum-25 balanced run byte-for-byte (fused + consts +
unvoted + cloneok both dies, split-V2, NO per-sub env), one delta:
DIMM_SPECS[0] fused_colmask -> PIM_FUSED_COLMASK_FILE on the D0
server (181-202 cols/bank = 8.8-9.9% host-repaired on D0's fused
bodies; D2 untouched).

| run | wall (24 tok) | s/tok | text |
|-----|---------------|-------|------|
| addendum-25 reference | 1140.6 s | 47.5 | ' \nAnswer: Paris. ...' |
| THIS: + D0 colmask    | 1174.4 s | 48.9 | BYTE-IDENTICAL |

- **Headline: 48.9 vs 47.5 = +3.0% — does NOT improve below 47.5.**
  Decomposition: the repair adds a measured ~0.8 ms of pop time per
  V2 request on the D0 server (srv-prof pop 1.6-1.8 vs 0.9 ms) ≈
  +12 s ≈ +1.1%; the rest is the run-to-run band (the O1/T5 A/Bs
  put ±1-4% on identical configs). Expected per the task framing —
  exactness was never predicted to change speed here.
- Traffic trajectory IDENTICAL to the reference run: D0 15,209
  calls / 11,731.1 MB, D2 15,151 / 11,732.1 MB — the same counts to
  the call, ENOSPC latch 1/server. Output text byte-identical: the
  D0 half's ~2.5%-inexact fused matmuls were already absorbed by the
  model, and the repaired (≈99.9%-exact on gate content) matmuls
  produce the same 24 tokens. The colmask's value at this scale is
  CHARACTERIZED EXACTNESS (and the mechanism split above), not wall
  or text. D0 consts were already active in the 47.5 reference
  ([res-consts] on all 4 banks in its server log) — no extra lever
  there.

## Files

- Campaign dir /home/deni/Claude/dimm0_fused_screen_2026_07_20/:
  screen_full.log + fused_screen.csv (per-layout per-column),
  gate_* ladder outs + server logs + bad-seg dumps,
  sim_* validation logs, v2_repair_sim_test.py,
  o10_dualdimm_24tok_colmask.log + o10_dualdimm_server_b{0,2}.log,
  run_o10_dualdimm.sh, addendum26_draft.md.
- BitNet dir: test_fused_colmask.cpp + fused-colmask-exe (NEW),
  fused_colmask_dimm0_bank{0-3}.txt (NEW, v3),
  test_bitnet_server.cpp (fused_col_bad + load_fused_colmask +
  fused_repair_pc at 3 accumulate sites + keep_masks force; also the
  O9 PIM_MAX_EXTRAS cap — default 4 byte-identical), rebuilt
  bitnet-proj-server; ab_fused_server.py --dump-bad (NEW flag).
- Client: pim_linear.py extra_env passthrough 'fused_colmask';
  run_bitnet_pim.py DIMM_SPECS[0] fused_colmask entry, gated by
  PIM_D0_FUSED_COLMASK=0 for A/B.

# Addendum 27 (Task O9) — new-subarray residency PILOT: the characterization pipeline is banked at ~32 min/subarray with 90-97% tuple yield, the +2-subarray arm measures NET +0.7% (parity) at the 8-tok shape, and the fragmentation accounting caps full-residency's ceiling at ~1.4x today — RECOMMENDATION: defer the 25-subarray campaign until build4/accum lands or the protocol goes long-form

2026-07-20 afternoon, bender 2, 8K image. Reduced scope per O5
(split-V2 is within 4% of the dual-die bound, so residency's value is
killing V2 recv volume + consts share): a MEASURED-VALUE PILOT on TWO
fresh subarrays before any 25-subarray commitment.

## Pilot windows (simra_full_2026_05_05 all_subarrays.csv)

| group | window | server cluster | note |
|-------|--------|----------------|------|
| 75 | [47232,47872) | 73 | between proven s72 and s78 |
| 81 | [51072,51712) | 79 | between proven s78 and s86 |

Cluster = open_rows[0]//640 (min open). Tuples STRADDLE the 640 label
boundary exactly like production s72 (labels 70/71) — of 50k viable
s75 tuples, 0 sit wholly in one label. The sampler keeps the window's
dominant cluster (73/79; the other-label complements got no pool files
so they can never half-join as extras).

## Characterization chain — restricted to what the calib join consumes

- FindOpenRows FOR_TT_LIST="0:1" (NEW env, default = full 4x4 grid):
  2 x 640-row subarrays in 2 x 33 min, 408,960 (r_first, r_second)
  pair rows, 0 stalls. ~29k distinct 16-open rate-1.0 tuples per
  cluster (14.5k even-Rf) — tuple-RICH silicon.
- Sampler: 60/cluster (seeded; 30/27 even-Rf).
- MajOperations MAJ_X_LIST=3 MAJ_T12_LIST=0 MAJ_T23_LIST=0 (NEW envs,
  default = original grids), stability 1000: 120 samples x 4 banks in
  17 min (April cost for this: ~a day — the restriction is 1/64th).
- **MAJ3-perfect yield (May criteria: full_stable==100 AND
  full_coverage==100 at (0,0)): 54-58/60 per bank; weakest bank2-s75
  36/60** — vs the May campaign's ~32%. Fresh subarrays of this die
  are calibration-grade almost everywhere.
- Join: production convention verified (every production cluster's
  first line is the SAME tuple on all 4 banks) — common even-Rf
  anchors ordered first: s73 Rf/Rs 47234/47579 (16 candidates), s79
  51076/51245 (23).
- Fault sweeps (O1 protocol, anchors only, PIM_SCREEN_RW=1, NO clone
  probe — the addendum-22 LAW replaces it on even-Rf): 8 x 23 s.
  **Cross-subarray reproducibility: union graph = 3969 edges on BOTH
  fresh windows = exactly the primary s72 count; bank-invariant (one
  13-edge fringe on s79 bank2); rw_screen 0 bad; law_dead = 108/640 =
  the exact 17% census everywhere.** The die's fault-graph structure
  and the clone law are subarray-invariant.
- Pools (greedy IS asc-degree, law-dead excluded from candidates):
  188 rows/cluster, banks byte-identical
  (pool_layout_dimm2_sub{73,79}_bank{B}.txt, "# window" headers).
  +2 x 172 = +344 LOAD rounds/bank on top of the O8 475.

Total silicon for the pilot characterization: **~63 min ≈ 32
min/subarray all-in** (33 FindOpenRows + 8.5 MajOps + 1.5 sweep).

## Pilot A/B (compound-headline shape, single-DIMM b2, 8 tok, unvoted
## + fused + consts + cloneok + per-sub extras env)

control = production calib (4 extras: 84/71/77/76). pilot = --calib
calib_dimm2_o9.txt (+430 lines, production file untouched) +
PIM_MAX_EXTRAS=7. The new clusters RANK FIRST by density (54-58
tuples > sub84's 23) so their pools drain first in overflow; sub85
joins inertly (9 rows < 16 tail).

| arm | s/tok (8 tok) | extras | resident handles | text |
|-----|---------------|--------|------------------|------|
| addendum-23 reference | 137.8 | 4 | ~59 | Paris |
| control (today) | **136.8** (1094.5 s) | 4 | 60 | ' \nAnswer: Paris. \n\nQuestion:' |
| pilot (+2 subs)  | **137.8** (1102.4 s) | 7 | 104 | byte-identical |

**NET: +0.7% — parity.** The accounting, measured:

- The +44 handles land ENTIRELY in layer 0's mlp (the pool fills
  during the first forward): L0 request wall +7.6 s (up_proj +31%,
  down_proj +47% = the in-run LOAD writes), L1-L15 flat (+-1%).
- Request classes (server prof): V2 147/149 ms; resident MM3D 35 ms
  (big) / 14 ms (small); MM3D-entry refresh 1.1 -> 1.9 ms (the 2 new
  windows enter every resident request's refresh).
- **The fragmentation insight (why conversion is NOT 147->35):** a
  converted big slice splits into MULTIPLE d_in sub-handles, each its
  own MM3D request per matmul (down_proj L0: 30 -> 420 requests!).
  3 x 35 ms + client overhead ~= one 147 ms V2 request — per-slice
  the conversion is worth ~1.4x, exactly the addendum-21 LOAD/V2
  ratio, NOT the per-request 4x illusion. The recv wall (23.2 of 35
  ms) rides every fragment.
- New-cluster units run the PLAIN body (fused stays validated-primary
  -only) — fused validation on fresh subarrays would be its own
  campaign (or the O10 colmask road).

## Projection to the full 25-subarray campaign — honest numbers

- Capacity: 475 + 25 x ~172 ~= 4,775 rounds/bank ~= 92% of the ~5.2k
  demand (near-full residency).
- Characterization cost at the banked pipeline rate: 25 x ~32 min ~=
  **13.5 h of silicon** (+ a D0 campaign for dual-DIMM symmetry).
- Steady-state ceiling TODAY: ~1.4x on the request wall IF fully
  amortized -> 137 -> ~98 s/tok single-DIMM, 47.5 -> ~34 dual. BUT
  the measured pilot says the 8-tok protocol cannot amortize even 44
  handles' LOAD (+7.6 s in-run vs ~7 s steady-state savings = the
  +0.7% net). Break-even needs run lengths several x longer (48+ tok
  or persistent sessions where LOAD is startup).
- AFTER build4/in-DRAM accumulation (the readout-wall track): the
  recv term (23 of 35 ms resident, ~90 of 147 ms V2) collapses;
  resident fragments -> ~12 ms; the residency ratio becomes ~3-4x
  and the campaign flips to clearly-worth-it.
- Drift at scale: unmeasured risk — full residency exposes ALL
  weights to the addendum-22 traffic-transition physics for the
  whole run (today ~9% of rounds are exposed and absorbed).

## RECOMMENDATION: DON'T run the 25-subarray campaign now.

At the current 8-tok benchmark shape it buys ~nothing (measured +0.7%
at +2 subarrays; scaling the same accounting to 25 stays inside the
noise band while costing 13.5 h of rig time). DO bank what the pilot
validated — the 32 min/subarray pipeline, the yield map, the law's
subarray-invariance, PIM_MAX_EXTRAS, and the two new production-grade
pools (sub73/sub79 stay installed and inert-cheap) — and pull the
trigger when EITHER (a) build4/accum lands (residency ratio 3-4x) or
(b) the benchmark protocol moves to long-form sessions where LOAD is
a startup cost. Those are the conditions under which 25 subarrays
convert 13.5 h of characterization into a >=1.4x headline factor.

## Files

- Campaign dir /home/deni/Claude/o9_newsub_pilot_2026_07_20/:
  subarrays_o9.txt, run_o9_char_chain.sh + chain{,_wrapper}.log,
  findopenrows_o9.log + open_rows_o9.csv (409k rows),
  o9_sample.py + samples_16_o9.txt, run_o9_step2_maj.sh +
  majops_o9.log + maj_coverage_16_o9.csv, o9_join_calib.py +
  o9_calib_append.txt + calib_dimm2_o9.txt + o9_anchors.txt,
  run_o9_step3_faultsweep.sh + sub{73,79}_bank{0-3}.{log,err},
  o9_derive_pools.py + pool_layout_dimm2_sub{73,79}_bank{0-3}.txt,
  run_o9_pilot.sh + o9_pilot_{control,pilot}{,_server}.log,
  o9_analyze_pilot.py, addendum27_final.md.
- Tool changes (all default-off/back-compat): FindOpenRows
  FOR_TT_LIST; MajOperations MAJ_X_LIST/MAJ_T12_LIST/MAJ_T23_LIST/
  MAJ_BANK_LIST (Makefile note: link with `make BOOST=` — same
  boost_filesystem gotcha as the BitNet dir); server PIM_MAX_EXTRAS.
- BitNet dir: pool_layout_dimm2_sub{73,79}_bank{0-3}.txt installed
  (production files + calib untouched; the o9 calib lives in the
  campaign dir).
