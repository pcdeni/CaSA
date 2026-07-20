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

## Scene 12 (was Scene 10) — AI-memory wishlist (renumbered 2026-07-20)

Each wishlist item maps to a specific bottleneck or limitation established earlier. No speculative items.

| Item | Bottleneck it addresses | Source |
|---|---|---|
| **1. In-DRAM POPCNT (and partial accumulation)** — return *one integer* per MAJ batch instead of one full row | Result transfer dominates as MAJ count grows (Scene 9) | [paper:POPCNT3 §3 — explicit demonstration of POPCNT3 inside COTS DDR4, with up to 348× throughput vs A100 GPU]; caption addendum 2026-07-20: "our own July 2026 merge (Scene 10)" — grounded in the Scene 10 rows below |
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

---

## Scene 10 — Killing the readout wall: in-DRAM accumulation (NEW 2026-07-20)

Workspace-data note: rows sourced to July-17/20 campaign files cite tower
workspace paths; staged for `docs/data/` per `repo_sync_plan_2026_07_20.md`.

| Claim | Source |
|---|---|
| One BitNet output lane at d_in=2560 needs K=2560 product rows popcounted; host path reads K full rows | derived from [code:python/pim_linear.py] (chunk math) + Scene 9 rows; exp0 readout amplification background [code: new/exp0_readout_floor.py] |
| Road A: CSA tree of dual-track full adders (carry=MAJ3, sum=MAJ5(a,b,c,¬c,¬c)) reduces K product rows to ceil(log2(K+1)) result rows | [code:app/test_popcount_indram.cpp]; adder validated in RESULT.md addenda 13–14 |
| Measured (die A, s78 tuple, ZERO+2 policy, screened cols): K=8/16/32/64 → 99.73/99.47/98.98/98.64% lanes exact; result-bit err 0.111/0.182/0.341/0.488%; readout 2.0×/3.2×/5.3×/9.1× | [data:sublattice_broadcast_2026_07_17/popcount_indram.log] (lane-exact-wrong 0.27/0.5288/1.0171/1.3613% — complements verified against the log 2026-07-20) |
| 213× at BitNet K=2560 = K/ceil(log2(K+1)) extrapolation (12 result bits vs 2560 reads) | arithmetic from the measured law; RESULT.md addendum 14 |
| Non-accumulating by construction: carry track ~exact MAJ3; sum bits terminal | RESULT.md addendum 13 (16-bit chain: carry 0.0000% all positions, sum flat 0.009–0.068%) + [data:sublattice_broadcast_2026_07_17/chain_b2_s78_16bit.csv] |
| Honest economics: host-marshalled shape ~320× SLOWER at K=64 (0.706 s vs ≈2.2 ms packed readout) | RESULT.md addendum 14b (measured correction) |
| Measured packed primitive rates: row read 34.2 µs, MAJ-only doubleACT 2.6 µs; crossover 3·t_MAJ < t_read holds 4.4× | [code:app/test_packed_maj.cpp] rates; RESULT.md addendum 14b |
| Fully-resident faithful tile: 58 ops/product measured; ~4–6× slower than packed readout single-bank on this rig; ≈par with bank-parallel ×4 + sparsity; value = mechanism + portability | [code:app/test_resident_tile.cpp]; RESULT.md addendum 17b (M3 verdict) |
| Streaming-PIM projection framing ("~100 ns/op, 58 ops beats an 8 KB transfer") | [editorial: projection], grounded in addendum 14b regime 3 + MVDRAM §V-E command-rate observation |
| Road B: popcount_accum HDL, 8 KiB → 4 B (2048×), bit-exact in Verilator 5/5 incl. the 4096-input BitNet shape; requires bitstream rebuild | [code:hdl popcount_accum.v staging]; memory record bitnet_bus_bound_hdl_staged; ADR-005 |
| pop_count4.v stock tree undercounts nibble 0xE; fixed co-resident | memory record dram_bender_pop_count4_bug; observed firing on silicon (RESULT.md addendum 20/20c: 0xE → 49152 exact totals) |
| Road-B silicon status: datapath proven (exact totals incl. 0xE fix); flush accounting fixed (build3); drain-capture timing → build4 (53/53 Verilator checks), Vivado build in flight; no Road-B per-token number quoted until it lands | RESULT.md addenda 20/20b/20c/20d |
| ADR-005 decision: keep both roads, Road A carries the reproduction claim, Road B the rig claim; publish the comparison, never blend | [docs: mvdram-repro/ADR-005-readout-killer.md] |
| Multi-body packing ~2.3× (M=8..29 bodies/Program, 7688 insts near the 8K ceiling, bit-exact); plateaus on c2h readback → composes with the roads | [code:app/test_packed_maj.cpp]; RESULT.md addendum 15 (M=3/8/16/29 → 1.26/2.58/2.07/2.29×) |

## Scene 11 — The production arc: 632 → 47.5 s/token (NEW 2026-07-20)

| Claim | Source |
|---|---|
| Arc: 632 (May baseline) → 360.8 (8K + fused) → 137.1 (clone-ok unvoted) → 80.5 (48-tok steady state) → 47.5 (dual-DIMM balanced) = 13.3× | RESULT.md addenda 19b/22/25 + memory record bitnet_optimization_state (O2 close carries the 80.5) |
| 632 → 438 (8K non-fused) → 360.8 (fused) = 1.75× vs May; correct output every config | RESULT.md addendum 19b table (T4 headline) |
| Bug 1: relative PIM_POOL_LIST_FILE → silent stride-pool fallback outside the calibrated window; fix = absolute paths + fatal-on-unreadable | RESULT.md addendum 18 |
| Bug 2: id()-keyed bitplane cache fed stale positions''' activations in batched prefill; deterministic, prompt-shape-dependent; fix = content-keyed cache; PIM_XBP_CACHE=0 discriminator → exact | RESULT.md addendum 18; memory record bitnet_xbp_cache_bug |
| Bug 3: V2 scratch drew over the whole pool and destroyed LOAD-resident weights (full-model scale only); fix = PIM_V2_SCRATCH tail reserve | RESULT.md addendum 19 |
| Bug 4: env subarray window mis-scoped the voting extras''' pools (wrong-subarray rows → every voting trip garbage); fix = per-calib window scoping | RESULT.md addendum 19 |
| Silicon clean throughout; standing rule: layer-0 validation cannot see pool-scale bugs — every production change gets one full-model run | RESULT.md addenda 18/19 (lessons) + memory record bitnet_pool_collision_and_extras |
| Voting economics: voted 360.8 vs unvoted 134.8 (garbage) on the May pool; unvoted 137.1 (Paris) vs voted 372.5 (byte-identical text) on the clone-ok pool | [data:dimm2_fault_sweep_subs_2026_07_18/o8_fullmodel_cloneok_{unvoted,voted}.log] (1096.7 s / 2979.9 s, 8 tok, same text — verified 2026-07-20); T5 record via memory bitnet_optimization_state |
| Clone-dead law + anti-selection (108/294 pool, 9/16 tail) as the voting explanation | xor_spread_explainer_ledger.md Scene 6 rows (primary) |
| 80.5 s/tok = 48-token marginal steady-state rate, text stable all 48 tokens; voting re-tested, no benefit; 8-tok vs 48-tok are different quantities, both reported | memory record bitnet_optimization_state (2026-07-20 O2 close; primary log scratchpad-resident — flagged in publish_ledger_2026_07_20.md) |
| Dual-DIMM defect: streamed (V2) fallback hardwired to servers[0]; D0 15,510 calls / 23.5 GB vs D2 628 / 51 MB; first attempt 90.3 s/tok (worse) | RESULT.md addendum 25 (forensics) |
| Fix: d_in-split V2 (contiguous chunk ranges per server, concurrent, host-summed partial sums; PIM_V2_SPLIT=0 reverts; ENOSPC latch); offline fake-server harness bit-exact on 6 cases before silicon | RESULT.md addendum 25; [code:python/pim_linear.py] (v2_parts) |
| Balanced result: 24 tok in 1140.6 s = 47.5 s/tok; D0 15,209 calls / 11,731.1 MB vs D2 15,151 / 11,732.1 MB (0.01% bytes); text byte-identical | RESULT.md addendum 25 |
| Honest normalization: per token-matmul 70.2 → 36.8 s = 1.91×, at 96% of the ideal-halving bound (1093.7 s); plain s/tok ratio 1.69× understated by prefill amortization | RESULT.md addendum 25 (verdict section) |
| Drift physics predicted the resident-consts regression; per-request rewrite fix; byte-identical-to-control Paris | xor_spread_explainer_ledger.md Scene 7 rows (primary); RESULT.md addendum 23 |
| "Byte-identical wrong output = logic, never noise" as standing practice | RESULT.md addendum 18 (lessons); memory record bitnet_xbp_cache_bug |
| Cleanest recorded outputs came from the fastest configs (O1 voted run = cleanest full-model text; balanced dual-DIMM byte-identical to single) | RESULT.md addendum 21 (validation ladder: "cleanest full-model output recorded on this rig") + addendum 25 |
| Remaining levers named: Road-B bitstream (recv volume), residency campaign (deferred with measured accounting), streaming shape | RESULT.md addenda 20d/27; STOCKTAKE phase notes |
