# PIM Explainer — Claim Ledger

Every factual claim in `pim_explainer.html` must trace to one of:
- **[textbook]** — standard DRAM operation, no specific citation needed
- **[paper:NAME §SEC]** — sourced from a paper file in `/home/deni/Claude/papers/`
- **[code:PATH:L]** — sourced from project source
- **[example]** — chip-specific value shown only as a labelled example, NOT as the concept

Claims that don't fit those tiers are **REMOVED**.

Papers (see `/home/deni/Claude/paper_mechanism_notes.md` for extraction):
- **SiMRA** — Yüksel et al., arXiv:2405.06081, 2024 (many-row activation, MAJX, Multi-RowCopy)
- **FracDRAM** — Gao et al., MICRO 2022 (Frac, Half-m, F-MAJ)
- **FCDRAM** — Yüksel et al., arXiv:2402.18736, 2024 (NOT, NAND, NOR, many-input AND/OR)
- **POPCNT3** — Kubo et al., NeurIPS MLNCP 2024 (bulk bitwise accumulation via DRAM-internal POPCNT3)

---

## Scene 1 — DRAM hierarchy

| Claim | Source |
|---|---|
| 1 bit = 1 capacitor + 1 access transistor (1T1C) | [textbook] |
| ~thousands of cells share one wordline → "row" | [textbook] |
| Rows are organised into "subarrays" with local sense amps | [textbook] |
| Subarrays form a "bank"; only one row open per bank at a time | [textbook] |
| 16 banks per rank (4 groups × 4) is the DDR4 standard | [textbook] |
| **PuD operations are confined to one subarray** (sense amps are local) | SiMRA §3.2 (all APA examples within one subarray); FCDRAM §1 extends to *neighboring* subarrays via shared sense amps |

## Scene 2 — One cell: write, destructive read, write-back

All [textbook]: charge sharing onto bitline, sense amp latch, destructive read, write-back.

## Scene 3 — Standard ACT / READ / PRE on a row

[textbook]; ACT opens entire row, READ pulls columns from row buffer, PRE writes back + precharges.

## Scene 4 — The APA envelope (ACT — wait t₁ — PRE — wait t₂ — ACT)

| Claim | Source |
|---|---|
| The primitive is `ACT R_F — wait t₁ — PRE — wait t₂ — ACT R_S` with both R in same subarray | [paper:SiMRA §3.2] |
| Standard DDR4 timing `tRAS` (ACT→PRE) and `tRP` (PRE→ACT) are violated by choosing small t₁, t₂ | [paper:SiMRA §3.2]; [paper:FracDRAM §III-A] |
| The same envelope produces DIFFERENT operations at different (t₁, t₂) — copy, multi-row activate, MAJ | [paper:SiMRA §3.2-§3.4] |
| **EXAMPLE** values from one DDR4 chip family (Hynix): RowCopy best at (≈36 ns, ≈3 ns); MAJ3 best at (1.5 ns, 3 ns); many-row activation at (3, 3) | [paper:SiMRA Obs. 14 / Obs. 7 / Obs. 1] (labelled as example) |
| The project's `doubleACT(t_12, t_23, ...)` is exactly this envelope | [code:util.cpp:242-286] |

## Scene 5 — One envelope, three operations (RowClone / Multi-RowCopy / Multi-Row Activation)

| Claim | Source |
|---|---|
| When t₁ ≥ tRAS, the sense amp fully resolves R_F's value before PRE; short t₂ then opens R_S onto bitlines still driven by sense amp → R_S absorbs R_F's value. **RowCopy.** | [paper:SiMRA §3.4 lines 313–328] (citation: "we issue the second ACT command by greatly violating tRP. The second ACT command interrupts the PRE command. By doing so, it 1) prevents the bitline from being precharged to V_DD/2, 2) keeps R_F and the sense amplifier enabled, and 3) simultaneously activates many rows. Finally, the sense amplifier overwrites all activated rows with the source row's data.") |
| Multi-RowCopy: same as above, but the second ACT's address triggers many destination wordlines simultaneously (row decoder predecoders latch overlapping signals) | [paper:SiMRA §3.4 + §7.1] |
| When t₁ is short, the sense amp does NOT fully resolve, and the second ACT opens many rows whose cells share charge directly on the bitlines (no source row dominating). This is the basis for MAJ/AND/OR. | [paper:SiMRA §3.2] |
| The number of rows opened (K) depends on the address relationship between R_F and R_S and on the row decoder structure | [paper:SiMRA §7.1 — hierarchical decoder hypothesis, 5 predecoders → max K = 32] |
| K is CHIP-DEPENDENT. Observed: K ∈ {2, 4, 8, 16, 32} on Hynix and Micron DDR4. Samsung supports K = 1 only (no multi-row activation). | [paper:SiMRA Lim. 1 + Lim. 2] |

## Scene 6 — MAJ via K-row charge share, with replication for stability

| Claim | Source |
|---|---|
| With K simultaneously activated rows, K cells per bitline share charge in parallel; sense amp resolves bitline to V_DD if majority of K cells held 1, else 0 → MAJ-of-K | [paper:SiMRA §2.2 (MAJ3); §3.3 (generalises to MAJX)]; analytic relation V_BL ≈ n_ones·V_DD/K (negligible bitline capacitance limit) | [paper:FCDRAM §6.1 footnote 10] |
| To do MAJ-of-X for small X (e.g., X=3) with HIGH success on real chips, **replicate** each of the X inputs N/X times across N activated rows; the leftover N mod X rows are initialised to V_DD/2 ("neutral") | [paper:SiMRA §3.3 lines 294–302] |
| The neutral rows' V_DD/2 is achieved via the **Frac** primitive | [paper:FracDRAM §III-A] (a separate APA-style sequence `PRE ACT(R) PRE` whose final PRE interrupts before the sense amp resolves) |
| Replication helps because bitline perturbation magnitude scales with replica count; sense amp threshold scatter is fixed; bigger perturbation crosses the threshold reliably under process variation | [paper:SiMRA §7.2 + Fig. 15] (SPICE: 32-row vs 4-row → 159% bigger perturbation; 32-row success degrades only 0.01% under 0-40% process variation, vs 46.58% for 4-row) |
| **EXAMPLE empirical values** (one chip family): MAJ3 with 32-row activation reaches ~99% success rate vs ~68% for 4-row activation (a 30.81 pp improvement) | [paper:SiMRA Obs. 6, line 444] (labelled as example) |
| **Different chip families need different layouts.** Our project's server emits one such layout (5 replicas of each input + neutrals on top of 16-row activation) — that specific count is a calibration result for our specific chips, not the general principle. | [code:test_bitnet_server.cpp:258-280]; concept from [paper:SiMRA §3.3] |
| MAJ(a, b, 0) = (a AND b); MAJ(a, b, 1) = (a OR b). The "0 input" is implemented by initialising one input row to all-zeros, OR by using a Frac'd V_DD/2 cell which contributes no bias. | [textbook Boolean]; [paper:FracDRAM §VI-A] (F-MAJ pattern: one row Frac'd → contributes no perturbation → MAJ of the other rows) |

## Scene 7 — From bitline AND to one ternary × int8 multiply

| Claim | Source |
|---|---|
| Ternary weight w ∈ {−1, 0, +1} represented as TWO bits: pos_mask (1 iff w=+1), neg_mask (1 iff w=−1) | [code:pim_linear.py:248-254] |
| Int8 activation decomposed into 8 bitplanes; bitplane b contributes with weight 2^b | [code:pim_linear.py:49-55, 64-65] |
| BITPLANE_FACTORS = [1, 2, 4, 8, 16, 32, 64, **−128**] — MSB negative for two's complement sign | [code:pim_linear.py:65] |
| Per (output, sign, bitplane, chunk-of-32-inputs): one MAJ-based AND op produces popcount of (mask_bit AND act_bit) across 32 inputs | [code:pim_linear.py:231 (n_chunks = d_in // 32) + server's emit_bank_combined_body] |
| Per output: 2 (signs) × 8 (bitplanes) × n_chunks MAJ ops; host accumulates `Σ_b factor[b] · (pos_pop[b] − neg_pop[b])` | [code:pim_linear.py] |
| Then ×bf16 weight_scale × input_scale → bf16 result; returned to PyTorch | [code:pim_linear.py:212] |

## Scene 8 — Inference loop (one token through the model)

| Claim | Source |
|---|---|
| Tokenize → embed → 30 layers → unembed → softmax → sample → loop | [textbook transformer]; BitNet b1.58-2B-4T model card |
| Per layer: 4 attention linears (q/k/v/o) + 3 MLP linears (gate/up/down) — these 7 are ternary and offloaded to PIM | [code:pim_linear.py + model architecture] |
| Attention compute (Q·Kᵀ, softmax, ·V), RMSNorms, residuals stay on CPU | [code: PyTorch model forward] |
| MAJ count per projection of d_in × d_out: `2 × N_bp × (d_in/32) × d_out` | derived from [code:pim_linear.py] |
| **DROP any absolute s/tok number** — never measured cleanly | n/a (removed from explainer) |

## Scene 9 — Bottleneck analysis (NEW)

| Claim | Source |
|---|---|
| Per MAJ, the on-chip compute is one APA + the readout. The on-chip phase is fast (parallel across bitlines in one subarray). | [paper:SiMRA] |
| The READ command pulls the result row through the column decoder, I/O drivers, DRAM data bus, and the bus from the controller to the host. The host then computes the popcount. | [textbook DDR4 READ path]; [code:test_bitnet_server.cpp:295 rdRow + segment_popcount on host] |
| Per MAJ, the on-host popcount processes one row of bits (8 KiB on DDR4 = 65,536 bits per row). One row → one number. | [code:test_bitnet_server.cpp:541 segment_popcount] |
| The accumulation step is the cumulative bottleneck for LLM workloads — many MAJs feed the same accumulator, but each MAJ's full row crosses the bus before the host reduces it. | [paper:POPCNT3 §1 lines 71-76]: "Executing accumulation with many inputs necessitates a proportional number of logical operations" and the "off-chip data transfer the primary performance bottleneck" framing |
| Other serial overheads per MAJ in our pipeline (not bottlenecks but contributors): RowCopy (setup), Broadcast (replicate weights across replication rows), 11 per-row writes (set up replica patterns), 3 Frac (neutralise selected rows), then the MAJ APA, then the READ. | [code:test_bitnet_server.cpp:228-300 emit_bank_combined_body] |

## Scene 10 — AI-memory wishlist (NEW)

Each wishlist item maps to a specific bottleneck or limitation established earlier. No speculative items.

| Item | Bottleneck it addresses | Source |
|---|---|---|
| **1. In-DRAM POPCNT (and partial accumulation)** — return *one integer* per MAJ batch instead of one full row | Result transfer dominates as MAJ count grows (Scene 9) | [paper:POPCNT3 §3 — explicit demonstration of POPCNT3 inside COTS DDR4, with up to 348× throughput vs A100 GPU] |
| **2. First-class MAJ-of-K instruction in the DDR command set** — no APA timing-trick, no per-chip calibration | Today the operation depends on out-of-spec timings that vary across manufacturers (SiMRA Limitation 1: Samsung does not support multi-row activation at all) | [paper:SiMRA Limitations §8] |
| **3. Larger guaranteed K** for many-row activation — more parallelism per cycle | Today's K is bounded by 5 predecoders → 32 rows (Hynix/Micron), with the precise K varying per chip and not user-controllable | [paper:SiMRA §7.1, Lim. 2] |
| **4. Direct support for fractional-value cells** — eliminate the Frac sequence's vendor variability | FracDRAM observes Micron/Elpida/Nanya reject the Frac sequence | [paper:FracDRAM §V-A lines 348-354] |
| **5. Cross-subarray compute via shared sense amps** — bigger compute fabric than one subarray | Today's PuD ops are confined to one subarray (SiMRA) or neighbor pairs (FCDRAM); extending across the whole bank would scale parallelism | [paper:FCDRAM §1, §2.1]; SiMRA acknowledges this as a limit |
| **6. Smaller calibration burden** via deterministic per-chip behaviour | Today we sweep millions of `(R_F, R_S, open_rows[K])` combos per DIMM to find usable tuples (per our project's FindOpenRows / MajOperations sweep) | [paper:SiMRA Limitation: per-chip variation]; [code: our project's calibration sweep] |

---

## Scene 9 addendum — "July 2026: two levers landed" (steps 6–7, added 2026-07-17)

Steps 1–5 of Scene 9 are unchanged and now carry an explicit "as of
May 2026" dating (their claim rows above still apply). The two new steps:

| Claim | Source |
|---|---|
| Lever 1 (persistent weights): the per-MAJ weight reload — a per-column write of W taking 3 host round-trips — becomes one in-DRAM clone from a backup row at a safe pair offset | [code:app/test_fused_maj.cpp] (header: "A (production shape): per-column write W → Rfirst (3 execs)"; "B: clone backup→Rfirst"); safe offset rule from [code:app/test_safe_load.cpp] |
| Lever 2 (coset activation update): the 11 uniform wrRows (ONE + 5×x + 5×zero) become 5 wrRows + 2 in-tuple coset doubleACTs — i.e. the 10 activation-slot writes collapse to 4 wrRows + 2 doubleACTs, x on positions {1,5,9,13}+{4}, zeros on {2,6,10,14}+{8}, same 5/5/5 vote balance | [code:app/test_bitnet_server.cpp:258-296] (the `PIM_FUSED_COSET` block and its comment) |
| The 5/5/5 vote balance is a hard rule; the unbalanced 7W/4x/4z variant is WRONG (99.7%/97.9% bad segments per die in the tool A/B) | [code:app/test_fused_maj.cpp] (B1 variant, iters=100 campaign 2026-07-17); the server comment independently marks the variant KNOWN WRONG [code:app/test_bitnet_server.cpp:270-272] |
| Fused tool A/B, steady-state, iters=100, one process, verified every iteration: 0.357 → 0.089 ms/MAJ = 3.99× (die A = bender 2); 0.556 → 0.085 ms/MAJ = 6.53× (die B = bender 0); bit-exact 0/204,800 bad segments per die | [code:app/test_fused_maj.cpp] (checks `result == W & x` per iteration; run 2026-07-17) |
| Production server, `PIM_FUSED_COSET=1`: 9.7–9.8 → 6.4–6.9 ms/matmul (bank 0) and 9.5–9.7 → 6.1 ms/matmul (4-bank) = 1.45–1.6×, every returned y bit-exact vs the exact host-side reference, both arms | [code:app/test_bitnet_server.cpp]; [code:python/ab_fused_server.py] (same `--seed` → byte-identical requests; per-request exact reference check) |
| Real model: BitNet b1.58-2B, layer 0 × 7 projections in DRAM, 4-bank, 8 tokens — 117.2 → 71.8 s wall = 1.63× (PIM request time 112.5 → 70.2 s); both arms answer "Paris" | [code:python/run_bitnet_pim.py] + [code:app/test_bitnet_server.cpp] (A/B identical except the env flag; run 2026-07-17) |
| The read-side path (READ, DDR bus, XDMA, host popcount) is untouched by both levers; the May bottleneck analysis stands, with a larger share of remaining time on the bus | [code:app/test_bitnet_server.cpp] (the fused path edits only the write-side program body — steps 1–3 of the May phase list); [editorial consequence] |
