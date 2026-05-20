# PIM Explainer — Adversarial Review

Review of `/home/deni/Claude/pim_explainer.html` against:
- ledger: `/home/deni/Claude/pim_explainer_ledger.md`
- mechanism notes: `/home/deni/Claude/paper_mechanism_notes.md`
- papers in `/home/deni/Claude/papers/` (SiMRA, FracDRAM, FCDRAM, POPCNT3)
- project source: `test_bitnet_server.cpp`, `util.cpp`, `pim_linear.py`
- model config: `/home/deni/bitnet_weights/config.json`

Each finding is tagged BLOCKER / MAJOR / MINOR / OK.

---

## Scene 1 — DRAM hierarchy

- **OK [scene 1, 1T1C]:** "one CELL = 1 access transistor + 1 capacitor" — textbook DRAM. | source: any DRAM textbook. | fix: none.
- **OK [scene 1, row = wordline]:** "thousands of cells share one wordline → one ROW" — textbook. | source: SiMRA §2.1 lines 110-114. | fix: none.
- **MINOR [scene 1, subarray contains 512-1024 rows]:** The on-canvas claim and caption assert subarrays hold "rows + their local sense amps" but do not give a count. The SiMRA paper says "many DRAM rows (512-1024)" (line 52) which would be a useful concrete bound but is not used. | source: SiMRA §1 Q1. | fix: optional — add "subarrays hold 512-1024 rows per the SiMRA analysis."
- **OK [scene 1, PuD ops in one subarray]:** "all PuD operations (RowCopy, MAJ, ...) live inside one subarray". | source: SiMRA §3.2, FCDRAM extends to *neighboring* subarrays but the in-subarray confinement is the SiMRA baseline. | fix: none, but FCDRAM extension is glossed (see scene 10).
- **MINOR [scene 1, 64 ms refresh]:** "cap holds ~tens of fC; leaks; needs refresh (~64 ms)" — this is JEDEC DDR4 textbook (tREFW = 64ms at <85°C), but no citation is given. The papers do not state 64ms. | source: JEDEC JESD79-4 DDR4 spec. | fix: cite the standard or omit the number; the qualitative "needs periodic refresh" suffices for the audience.
- **MINOR [scene 1, "1-2 ranks per DIMM (DDR4 standard)"]:** DDR4 DIMMs commonly come in 1, 2, 4, or even 8 ranks (LRDIMM). The "1-2 ranks" is a simplification that's only accurate for typical unbuffered/UDIMM consumer parts. Presented as "standard" suggests JEDEC mandates this — JEDEC allows more. | source: JEDEC JESD79-4. | fix: change to "typically 1-4 ranks per DIMM" or drop "(DDR4 standard)".
- **OK [scene 1, 16 banks per rank]:** "16 banks per rank (4 groups × 4)" matches DDR4 (DDR3 had 8, DDR5 has 32). | source: textbook DDR4. | fix: none.
- **OK [scene 1, cross-bank-group tighter timing]:** "Cross-bank-group commands have tighter timing parameters than same-group commands" — JEDEC DDR4 has tCCD_S < tCCD_L, tRRD_S < tRRD_L. | source: textbook DDR4 spec. | fix: none.

## Scene 2 — One cell: write, destructive read, write-back

- **OK [scene 2, all steps]:** All five steps (idle, ACT, sense amp resolve, READ destructive, PRE writeback) are textbook DRAM operation. Visual matches the caption description. | source: textbook. | fix: none.

## Scene 3 — ACT / READ / PRE on a whole row

- **OK [scene 3, all steps]:** Textbook DDR4 operation. The tRAS and tRP labels are correct. | source: textbook + SiMRA §2.1. | fix: none.
- **MINOR [scene 3, step 3 caption final sentence]:** "SiMRA, FracDRAM, FCDRAM all bend this 3-phase cycle by violating tRAS or tRP — that's where in-DRAM compute comes from." This is broadly correct but FracDRAM's mechanism involves a PRE-ACT-PRE sequence (not just shortened tRAS/tRP within a normal ACT-PRE-ACT cycle); the paper interrupts the activation before the sense amp is enabled, which is qualitatively different from "violating tRAS". | source: paper_mechanism_notes.md §1.2; FracDRAM §III-A lines 167-195. | fix: change to "...by issuing commands outside the timing windows JEDEC permits."

## Scene 4 — The APA primitive

- **OK [scene 4, APA envelope definition]:** "ACT, then PRE, then ACT. Both ACTs target rows in the same subarray." | source: SiMRA §3.2 lines 234-235 (`/tmp/simra.txt:234`). | fix: none.
- **OK [scene 4, tRAS/tRP violation]:** "Standard DDR4 demands t₁ ≥ tRAS [t₂ ≥ tRP], but APA can shorten it." | source: SiMRA §3.2 line 182 ("APA with violated tRAS and tRP timing constraints"). | fix: none.
- **OK [scene 4, step 4 EXAMPLE marker]:** "EXAMPLE values from one chip family appear in later scenes — the specific numbers vary across DDR4 manufacturers." — properly labelled as EXAMPLE. | source: SiMRA Lim. 1. | fix: none.
- **MINOR [scene 4, step 4 outcome table includes "PRE ACT PRE" → Frac]:** The outcome table inside the APA scene lists `PRE ACT PRE back-to-back → Frac (cell ← V_DD/2)`. This is technically NOT an APA sequence — it's the FracDRAM PRE-ACT-PRE sequence, which has a different command order than APA's ACT-PRE-ACT. Showing it inside the APA primitive scene conflates two different command envelopes. | source: FracDRAM §III-A (line 181). | fix: explicitly note "FracDRAM's Frac uses a DIFFERENT envelope (PRE-ACT-PRE), shown here for completeness."
- **OK [scene 4, codeRef]:** project's `doubleACT(t_12, t_23, RF, RS)` correctly maps to SiMRA's `ACT R_F – t₁ – PRE – t₂ – ACT R_S`. The instruction offsets `q_inst[0]`, `q_inst[t_12+1]`, `q_inst[2+t_12+t_23]` match the actual code at `util.cpp:275-278`. | source: `util.cpp:275-278`. | fix: none.

## Scene 5 — RowCopy / Multi-RowCopy / Multi-Row Activation

- **OK [scene 5, RowCopy mechanism]:** "t₁ ≥ tRAS, t₂ ≪ tRP — sense amp fully resolves R_F, then dst opens onto driven bitlines" matches SiMRA §3.4. | source: SiMRA §3.4 lines 313-328. | fix: none.
- **BLOCKER [scene 5, step 3 caption: "Multi-RowCopy. Same timing as RowCopy."]:** The caption asserts Multi-RowCopy uses the SAME timing as RowCopy (long t₁, short t₂). This matches the *paper* (Multi-RowCopy uses t₁=36ns, t₂=3ns). HOWEVER, the project's own code at `test_bitnet_server.cpp:247-253` and the codeRef on this same scene explicitly shows "RowClone backup → Rfirst" with `t_12=long`, but "Multi-RowCopy Rfirst → K open rows" with `t_12=short` (actual values `30/1` vs `10/2`). The caption and the codeRef directly contradict each other. The project's "Broadcast" is implemented with short t_12=10, which is neither the paper's RowCopy (36) nor the paper's many-row-activation (3) — it's a project-empirical middle ground. | source: contradiction between scene 5 caption and `test_bitnet_server.cpp:247-253` (also `util.cpp:242-286`). | fix: either (a) update the caption to acknowledge the project uses short t_12 for Broadcast as a chip-specific empirical choice, or (b) note that the paper's Multi-RowCopy uses long t_12 but the project's empirical sweet spot for its specific chips is shorter; do not present these as "same timing" when the code says otherwise.
- **OK [scene 5, step 3 K values]:** "EXAMPLE: SiMRA reports K ∈ {2, 4, 8, 16, 32} on Hynix/Micron DDR4" matches paper Obs. 14 (1, 3, 7, 15, 31 destination rows + 1 source = K = 2, 4, 8, 16, 32). EXAMPLE properly labelled. | source: SiMRA Obs. 14 (`/tmp/simra.txt:562`). | fix: none.
- **OK [scene 5, Samsung supports no multi-row activation]:** | source: SiMRA Limitation 1 lines 890-908. | fix: none.
- **OK [scene 5, step 4-5 Multi-Row Activation mechanism]:** "Now BOTH t₁ and t₂ are short. The sense amp does not fully resolve any one row before the next ACT fires." matches SiMRA §3.3. | source: SiMRA §3.3 + Obs. 1 (best timing t₁=t₂=3ns). | fix: none.
- **OK [scene 5, codeRef row decoder hypothesis]:** "Max K = 2⁵ = 32 simultaneous wordlines" matches SiMRA §7.1 lines 727-740. | source: SiMRA §7.1 line 740 ("five predecoders, and thus, we can activate up to 2⁵ rows"). | fix: none.

## Scene 6 — MAJ via K-row charge share; replication for stability

- **OK [scene 6, step 0 K=3 mechanism]:** "K rows simultaneously activated → K cells per bitline share charge → sense amp resolves majority" matches SiMRA §2.2 lines 187-196. | source: SiMRA §2.2. | fix: none.
- **MINOR [scene 6, step 0 SVG: V_BL ≈ (n_ones/K)·V_DD]:** The on-canvas formula `V_BL ≈ (n_ones / K) · V_DD` is the FCDRAM §6.1 analytical formula (under the "negligible bitline capacitance" simplifying assumption, footnote 10). SiMRA does NOT derive this; SiMRA only states the bitline reflects the charge balance. Presenting the formula without flagging the simplifying assumption is a minor overreach — the actual bitline voltage depends on the ratio of bitline capacitance to cell capacitance. | source: FCDRAM §6.1 footnote 10 (paper_mechanism_notes.md §1.3 line 127). | fix: append "(in the limit of negligible bitline capacitance)" to the formula.
- **OK [scene 6, "Ambit-style MAJ3 is unreliable on COTS chips"]:** Step 0 caption: "on real COTS chips, K=3 alone often gives unreliable resolution under process variation" — supported by SiMRA §5 success-rate data. | source: SiMRA §5 (4-row MAJ3 ≈ 68% success). | fix: none.
- **OK [scene 6, step 2 EXAMPLE: "+159% perturbation (SPICE)"]:** Matches SiMRA §7.2 Fig. 15a — "159.05% higher bitline voltage perturbation". Properly EXAMPLE-labelled. | source: `/tmp/simra.txt:732` ("159.05% higher"). | fix: none.
- **OK [scene 6, step 2 EXAMPLE: "MAJ3 success ~99% vs ~68%"]:** 99% matches SiMRA Obs. 7 line 462 ("99.00% average success rate"). 68% is correctly back-derived from 99% - 30.81pp (Obs. 6). Properly EXAMPLE-labelled. | source: SiMRA Obs. 6/7. | fix: none.
- **OK [scene 6, step 3 Frac mechanism]:** "FracDRAM's Frac primitive sets cells to V_DD/2 by interrupting the activation before the sense amp resolves" matches FracDRAM §III-A. | source: FracDRAM §III-A lines 167-195. | fix: none.
- **OK [scene 6, step 4 MAJ → AND/OR]:** MAJ(a,b,0)=AND, MAJ(a,b,1)=OR — textbook Boolean. | source: textbook. | fix: none.
- **OK [scene 6, step 5 5/5/5/1 layout properly EXAMPLE-labelled]:** Caption: "Our calibration sweep landed on a 16-row pattern: 5 rows of weight bit, 5 of activation bit, 5 of zero, 1 Frac'd row at position 0. The 5/5/5/1 split is empirically tuned for our chips." The pattern matches `test_bitnet_server.cpp:258-280`: open_rows[0] = ONE then Frac'd; act_pos = {1,4,7,10,13}; zero_pos = {2,5,8,11,14}; rows 3,6,9,12,15 keep w from Broadcast. EXAMPLE label is prominent. | source: `test_bitnet_server.cpp:258-280`. | fix: none.
- **MINOR [scene 6, step 5 "1 Frac'd row at position 0"]:** The caption omits that open_rows[0] is FIRST written to ONE then Frac'd three times. The end state is the Frac-degraded value (closer to V_DD/2 regardless of starting value), so the caption captures the effective state, but a reader could misinterpret the layout as "1 row simply pre-set to V_DD/2 by Frac". | source: `test_bitnet_server.cpp:261,276-280`. | fix: optional — note "row 0 is initialised to ONE then Frac'd 3 times, ending near V_DD/2."

## Scene 7 — Ternary × int8 multiply

- **OK [scene 7, ternary as 2 bits]:** "Storing one ternary value needs 2 bits" — correct as a storage statement (information content is log2(3) ≈ 1.58 bits, but storage uses 2 bits). | source: `pim_linear.py:248-254`. | fix: optional — note that the "1.58" in BitNet b1.58 refers to information content, not storage cost.
- **OK [scene 7, pos_mask/neg_mask split]:** | source: `pim_linear.py:248-254`. | fix: none.
- **OK [scene 7, 8 bitplanes with BITPLANE_FACTORS = [1,2,4,8,16,32,64,-128]]:** Matches `pim_linear.py:65` exactly. The MSB = -128 (two's complement sign) is correct for int8. | source: `pim_linear.py:65`. | fix: none.
- **OK [scene 7, n_chunks = d_in // 32]:** | source: `pim_linear.py:231`. | fix: none.
- **OK [scene 7, 2 (signs) × 8 (bitplanes) × n_chunks MAJs per output]:** Correct derivation. | source: `pim_linear.py` request schema. | fix: none.
- **BLOCKER [scene 7, step 4 caption: "Multiply by bf16 weight scale × input scale → bf16"]:** The caption says we MULTIPLY by input_scale. The actual code at `pim_linear.py:352` DIVIDES by input_scale: `y_f32 = (y_int.astype(np.float32) * self._weight_scale) / flat_scale[t, 0]`. The on-canvas SVG label at scene 7 step 4 also says `y_bf16 = y_int32 × weight_scale × input_scale` — same error. This is a sign-of-operation error in a load-bearing math statement. The rationale: BitNet quantises x → x_int8 = round(x · input_scale), so to recover original-magnitude dot product we must DIVIDE the int matmul result by input_scale. The user's own memory note `bitnet_two_bugs_2026_05_05.md` even flags this exact direction error as a prior bug. | source: `pim_linear.py:352`; memory `bitnet_two_bugs_2026_05_05.md`. | fix: change both the SVG label and the caption to `y_bf16 = y_int32 × weight_scale / input_scale`.

## Scene 8 — Inference loop

- **OK [scene 8, model architecture]:** "Per layer: 4 attention linears (q/k/v/o) + 3 MLP linears (gate/up/down) — these 7 are ternary and offloaded to PIM" matches BitNet model. | source: BitNet b1.58-2B-4T `config.json`. | fix: none.
- **OK [scene 8, attention compute / RMSNorms / residuals on CPU]:** | source: project code. | fix: none.
- **OK [scene 8, MAJ count per projection formula]:** `2 × N_bp × (d_in/32) × d_out` is the correct count derived from the bitplane × sign × chunk grid. | source: `pim_linear.py`. | fix: none.
- **BLOCKER [scene 8, step 2 caption: "For BitNet b1.58 the projection sizes are d_in, d_out ∈ {2048, 5120, 5376, 6912}"]:** The actual BitNet b1.58-2B-4T `config.json` has:
    - `hidden_size: 2560` (not 2048)
    - `intermediate_size: 6912` ✓ (one of four numbers is right)
    - `num_attention_heads: 20`, `num_key_value_heads: 5` → q/o projections are 2560→2560 (not 2048→2048); k/v projections are 2560→640 (not 2560→5120 nor 5120→anything)
    
    The actual projection dimensions are **{640, 2560, 6912}**, NOT {2048, 5120, 5376, 6912}. Three of the four numbers are made up. The "2048" appears to be a confusion with `D_OUT_SLICE` from `pim_linear.py:24` (the host-side output-slice batching constant), not a model dimension. | source: `/home/deni/bitnet_weights/config.json`. | fix: replace with `d_in, d_out ∈ {640, 2560, 6912}` and add the per-projection breakdown (q/o: 2560→2560; k/v: 2560→640; gate/up: 2560→6912; down: 6912→2560).
- **OK [scene 8, 21 s/tok removed]:** No "s/tok" or "second/token" string appears in the HTML. | source: grep over `pim_explainer.html`. | fix: none.
- **OK [scene 8, "× N_layers" placeholder]:** The diagram uses "× N_layers" rather than a hardcoded 30, which is the right call (lets the reader plug in their own model size). | source: convention. | fix: none.

## Scene 9 — Bottleneck analysis (NEW)

- **OK [scene 9, 8-phase pipeline list]:** Phases 1-8 (RowCopy → Multi-RowCopy → per-row writes → Frac × K → MAJ APA → READ → XDMA → host POPCNT) match `test_bitnet_server.cpp:228-300` (`emit_bank_combined_body`). | source: `test_bitnet_server.cpp:228-300`. | fix: none.
- **OK [scene 9, 8 KiB per DDR4 row]:** Matches DDR4 spec at the rank level (8 chips × 1 KiB) and is verified by the code `receiveData(8192)` at `test_bitnet_server.cpp:202`. | source: textbook DDR4 + `test_bitnet_server.cpp:202`. | fix: none.
- **OK [scene 9, 65,536 bits per row]:** = 8 KiB × 8. Correct. | source: derived. | fix: none.
- **OK [scene 9, step 3 POPCNT3 paper accumulation framing quote]:** "Executing accumulation with many inputs necessitates a proportional number of logical operations, resulting in slow performance when implemented in PuD" — matches the POPCNT3 paper at `/tmp/popcnt3.txt:74-76`. | source: POPCNT3 §1. | fix: none.
- **MAJOR [scene 9, step 3 "measured throughput: up to 348× vs A100 GPU on bulk bitwise accumulation"]:** The POPCNT3 paper's 348× number is for ONE specific kernel size (7×65536 — the smallest); the same paper reports only **27×** speedup at the largest kernel (127×65536) and reports the spectrum as "27-348×" (`/tmp/popcnt3.txt:623`). Presenting 348× as THE headline result without the range qualifier overstates the case. | source: POPCNT3 §5.1, Table 2 (`/tmp/popcnt3.txt:538-547`). | fix: change to "up to 348× (at smallest kernel size 7×65536); 27× at the largest tested kernel".
- **MAJOR [scene 9, step 3 "sized similarly to our setup"]:** POPCNT3's setup is "Xilinx Alveo U200 + 4 SK Hynix DDR4 DIMMs, parallel across 64 banks of 4 modules = 256 banks". The project's setup (per the memory notes) currently runs on 1 operational DIMM with bank-parallel batching across at most 4 banks. The setups are not "similar" in scale — POPCNT3 has ~64× more parallel banks than the project. | source: POPCNT3 §A (`/tmp/popcnt3.txt:716-718`). | fix: change to "(reported on 4 DDR4 modules behind a Xilinx Alveo U200 FPGA — POPCNT3 parallelises across all 256 banks; our project uses 1 operational DIMM × 4 banks)".
- **MINOR [scene 9, step 3 "bandwidth drop by row-width / log₂(K)"]:** POPCNT3 returns ceil(log₂(K+1)) bits per column (the binary representation of the count of 1s in K bits per column). The "bandwidth drop" framing is approximately correct (you replace K MAJ-result rows with log₂(K+1) result rows in the row-accumulation pipeline), but the formula `row_width / log₂(K)` is not a direct quote from POPCNT3 — it's a back-of-envelope by the explainer author. The paper itself doesn't state bandwidth in this form. | source: POPCNT3 §3.2. | fix: optional — either cite the formula's source or change to "bandwidth drop roughly by `K / log₂(K+1)` per accumulation chain". The current wording is also unclear in which direction the bandwidth drops.
- **MINOR [scene 9, step 4 "the set-up phases ... we do this already"]:** "amortise via persistent weights and broadcast-once-per-many-MAJs (we do this already)" — the persistent-weights path is real per the memory notes (`bitnet_persistent_weights.md`), but is a project-specific status claim about an evolving codebase. Worth confirming this is true in the current production server config. | source: memory `bitnet_persistent_weights.md`, `bitnet_load_weights_corruption.md`. | fix: optional — link to the LOAD_WEIGHTS code path so the claim is verifiable.

## Scene 10 — AI-memory wishlist (NEW)

- **OK [scene 10, item 1 In-DRAM POPCNT]:** Maps to the bottleneck established in Scene 9 (DDR bus + host popcount per MAJ). POPCNT3 paper is real and cited correctly. | source: POPCNT3 §3 + Scene 9. | fix: same as scene 9 — the "348× vs A100" headline number should be paired with "27×" worst-case for honesty.
- **OK [scene 10, item 2 First-class MAJ-of-K instruction]:** Tied to SiMRA Lim. 1 (Samsung doesn't support multi-row activation). | source: SiMRA Lim. 1 lines 890-908. | fix: none.
- **OK [scene 10, item 3 Larger guaranteed K]:** Tied to SiMRA §7.1 predecoder hypothesis (max K = 2^5 = 32 today). | source: SiMRA §7.1, Lim. 2. | fix: none.
- **OK [scene 10, item 4 Native fractional-value cells]:** Tied to FracDRAM §V-A — Micron/Elpida/Nanya reject the Frac sequence. | source: FracDRAM §V-A (`/tmp/fracdram.txt:217-220` for K/L/M groups). | fix: none.
- **MAJOR [scene 10, item 5 Cross-subarray compute]:** The bottleneck framing is fair (today's PuD is confined to one subarray or neighbouring pair). HOWEVER, the wishlist claim "extending the compute fabric to the whole bank multiplies parallelism by the subarray count" is presented as a clear win, but the user's own memory note `in_memory_levers_beyond_casa_sched.md` flags "cross-subarray compute" explicitly as a "dead end not to re-derive". There may be a fundamental physical reason this is hard (sense-amp sharing only exists at neighbour boundaries; routing analog charge across non-adjacent subarrays requires architectural changes the paper authors view as non-trivial). The wishlist should acknowledge this rather than present it as obvious. | source: memory `in_memory_levers_beyond_casa_sched.md`; FCDRAM §2.1 (open-bitline architecture). | fix: rewrite as "Cross-subarray compute would require changes to the analog routing between subarrays — this is non-trivial in current architectures (subarrays are physically isolated except for the neighbour-pair sense-amp sharing FCDRAM exploits). A radical architecture change."
- **OK [scene 10, item 6 Deterministic per-chip behaviour]:** Tied to project's calibration sweep effort + SiMRA's per-chip variation discussion. | source: project's FindOpenRows + MajOperations sweep code; SiMRA throughout. | fix: none.

## Title bar / nav

- **OK [title "DRAM as the compute substrate"]:** Subtitle says "Mechanism grounded in SiMRA, FracDRAM, FCDRAM, POPCNT3 + our project's server code" — accurate citation list. | source: explainer header. | fix: none.

---

## Summary count by severity

- **BLOCKER: 3**
  - Scene 5 step 3 caption claims Multi-RowCopy uses "same timing as RowCopy", contradicting the project's own code which uses `t_12=long` for RowClone (30/1) and `t_12=short` for Broadcast (10/2).
  - Scene 7 step 4 caption and SVG label both say `y = y_int × weight_scale × input_scale`, but the code DIVIDES by input_scale (`y_f32 = y_int * weight_scale / input_scale`). Sign-of-operation error in a load-bearing math statement.
  - Scene 8 step 2 caption: BitNet b1.58-2B-4T projection dimensions listed as `{2048, 5120, 5376, 6912}` — three of four numbers are wrong. Actual values from `config.json`: hidden=2560, intermediate=6912, kv_heads=5 → projections are `{640, 2560, 6912}`.

- **MAJOR: 3**
  - Scene 9 step 3 + Scene 10 item 1: POPCNT3's "348× vs A100" is presented as the headline; same paper reports 27-348× range, with 348× only at smallest kernel. Cherry-picking the best case.
  - Scene 9 step 3: "sized similarly to our setup" overstates similarity — POPCNT3 uses 256 parallel banks (4 modules × 64 banks); project uses 1 DIMM × 4 banks. Roughly 64× scale difference.
  - Scene 10 item 5: "Cross-subarray compute" wishlist item is presented as a clean win, but the user's own memory flags this as a "dead end" / non-trivial; the architectural barriers (only neighbour-pair sense-amp sharing exists) aren't acknowledged.

- **MINOR: 8**
  - Scene 1: subarray row count (not given) — optional.
  - Scene 1: 64 ms refresh number is uncited (JEDEC; not in the cited papers).
  - Scene 1: "1-2 ranks per DIMM (DDR4 standard)" — DDR4 allows up to 4-8 ranks; presented as standard when it's just typical.
  - Scene 3 step 3: "violating tRAS or tRP" framing under-describes FracDRAM (which uses PRE-ACT-PRE, not just a shortened APA).
  - Scene 4 step 4: Frac sequence (PRE-ACT-PRE) is listed inside the APA outcome table even though its envelope (PRE-ACT-PRE) is distinct from APA (ACT-PRE-ACT).
  - Scene 6 step 0: V_BL formula `(n_ones/K)·V_DD` is the FCDRAM §6.1 simplified analytical form (under negligible-bitline-capacitance assumption); presented without that caveat.
  - Scene 6 step 5: Caption omits that open_rows[0] is first written to ONE before being Frac'd 3 times.
  - Scene 7 step 0: "Storing one ternary value needs 2 bits" is correct for storage; could note the "1.58" naming refers to information content.
  - Scene 9 step 3: "bandwidth drop by row-width / log₂(K)" is a back-of-envelope formula not cited from POPCNT3 paper directly.
  - Scene 9 step 4: "we do this already" for persistent weights is a project-status claim worth confirming.

- **OK: ~35** (the bulk of the explainer's claims trace cleanly to the cited sources).

### Notable structural observations (not severity-tagged)

- The EXAMPLE labelling discipline is mostly followed. Scene 4 step 4, Scene 5 step 3, Scene 6 steps 2/5 all carry visible EXAMPLE / cyan markings on chip-specific numbers. Good.
- The 21 s/tok number is confirmed removed.
- The codeRef pane is generally the most accurate part of the explainer — caption text occasionally overreaches relative to what the cited code actually does (most notably scene 5 step 3 and scene 7 step 4).
- The wishlist (scene 10) is mostly grounded in concrete bottlenecks the prior scenes establish — except for item 5 (cross-subarray) which is the only item the user's own notes flag as not pursued.
