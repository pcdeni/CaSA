# In-DRAM Computation: Mechanism Extraction from Three SAFARI/Related Papers

Source papers (local plain-text extracts):
- SiMRA — Yüksel et al., "Simultaneous Many-Row Activation in Off-the-Shelf DRAM Chips," arXiv:2405.06081, 2024. `/home/deni/Claude/papers/SiMRA.txt`
- FracDRAM — Gao, Tziantzioulis, Wentzlaff, "FracDRAM: Fractional Values in Off-the-Shelf DRAM," MICRO 2022. `/home/deni/Claude/papers/FracDRAM.txt`
- FCDRAM — Yüksel et al., "Functionally-Complete Boolean Logic in Real DRAM Chips," arXiv:2402.18736, 2024. `/home/deni/Claude/papers/FCDRAM.txt`

All citations below are to the plain-text dumps (file:line) and the corresponding section/figure in the published paper. Lines refer to the local `.txt` files.

---

## 1. The Mechanism — Bitline/Sense-Amp Level

### 1.1 SiMRA — Simultaneous Many-Row Activation, MAJX, Multi-RowCopy

**Command sequence (all three operations use the same envelope, "APA")**
- `ACT R_F  –t1–  PRE  –t2–  ACT R_S` (SiMRA §3.2, line 234; "APA" defined at line 235).
- Both `R_F` and `R_S` index the same subarray (line 235).
- Standard `tRAS` (≥ ~35 ns DDR4) is violated by `t1`; standard `tRP` (~13–15 ns DDR4) is violated by `t2`. Tested values are `t1, t2 ∈ {1.5, 3.0, 4.5, 6.0} ns` (SiMRA §4, Fig. 3 / lines 333–348).

**Per-operation timing settings the paper reports as "best"**
| Operation | `t1` (ACT→PRE) | `t2` (PRE→ACT) | Source |
|---|---|---|---|
| Simultaneous many-row activation (rewrite via WR) | 3 ns | 3 ns | line 353 |
| MAJ3 (best for 32-row activation) | 1.5 ns | 3 ns | line 461 |
| Multi-RowCopy | 36 ns (≈ `tRAS`) | 3 ns | line 506 |

**Why timing violation produces multi-row activation (the row-decoder hypothesis, §7.1)**

- SiMRA does NOT explain multi-row activation at the bitline/sense-amp level; it explains it at the *row-decoder* level. The relevant statement is (lines 657–665, §7.1): "Reducing the latency between PRE and the second ACT commands (i.e., tRP) allows the predecoders to latch the next RA without deasserting the RA targeted by the first ACT command. Hence, after the second ACT command, depending on the target addresses of APA sequence, multiple latches of each predecoder in LWLD can be set."
- Hierarchical row decoder = Global Wordline Decoder (GWLD) selects the subarray via the high-order 7 RA bits; Local Wordline Decoder (LWLD) decodes the low-order 9 RA bits in two stages: Stage 1 = 5-tier predecoder bank (A,B,C,D,E) that latches predecoded P signals; Stage 2 = combinational decoder that asserts the LWL when all P signals on its branch are asserted (SiMRA §7.1, Fig. 13 / lines 626–656).
- "To activate 2^N rows, N different predecoders have to latch two predecoded P signals." (line 727). To activate 32 rows simultaneously, the APA addresses must differ in bits such that *all five* predecoders latch two outputs (line 735).
- **What the paper does NOT explain at the bitline level**: SiMRA does not explain *how the WR command then writes a value into all simultaneously activated rows once they are open.* It only states that "this WR command causes the sense amplifiers to overdrive their bitlines and thus updates the values of the cells in all simultaneously activated DRAM rows" (lines 244–246), citing prior works. So the "rewrite by WR" of N rows at once is treated as observed behaviour, not derived.

**MAJ3 mechanism (bitline + sense amp), per §2.2 lines 187–196**

> "When three rows are concurrently activated, three cells connected to each bitline share charge simultaneously and contribute to the perturbation of the bitline. Upon sensing the perturbation of the three simultaneously activated rows, the sense amplifier amplifies the bitline voltage to V_DD or 0 V if at least two of the three DRAM cells are charged or discharged, respectively. As such, simultaneously activating three rows results in a Boolean majority-of-three operation (MAJ3)."

That is the *entire* bitline-level mechanism statement for MAJ3 — three cells charge-share onto one bitline, the sense amp resolves to V_DD/0 by the simple majority of cell charges. No explicit voltage threshold; the implicit threshold is the sense-amplifier's reference (V_DD/2 after precharge, see FracDRAM §II below).

**MAJ-of-K (K > 3) mechanism (SiMRA §5)**

- Procedure: replicate each of the K logical inputs `N/K` times across an N-row activation (so K logical inputs are spread over N physical rows; lines 294–298). Example: MAJ3 with N=32 means each input is replicated 10 times; the leftover `N mod K` rows are initialized to V_DD/2 using FracDRAM's `Frac` operation, called "neutral rows" (lines 299–302).
- The paper does *not* state a new bitline equation for K>3; the same charge-sharing argument as MAJ3 applies, with the sense-amp threshold relative to the bitline voltage after K-way charge-share. The bitline voltage perturbation magnitude is reported empirically and via SPICE (Fig. 15 / lines 681–705).
- "Mfr. M, Frac operation is not supported. However, we observe that the sense amplifiers of these modules are always biased to one or zero. Initializing the neutral rows with all zeros/ones enables a MAJX operation." (lines 311–312, footnote 5).

**Multi-RowCopy mechanism (SiMRA §3.4 lines 313–319)**

- Same APA envelope, *different* timing: `t1 = tRAS = 36 ns`, `t2 ≈ 3 ns`. This is the key difference from MAJX/many-row activation.
- "First, we issue ACT to the source row (i.e., R_F) to assert the wordline and to connect R_F to the bitlines. Second, we issue PRE after waiting for `tRAS`. This ensures the sense amplifier senses R_F correctly and drives bitlines to the source row's charge. Third, we issue the second ACT command by greatly violating tRP (i.e., ≤ 3 ns). The second ACT command interrupts the PRE command. By doing so, it 1) prevents the bitline from being precharged to V_DD/2, 2) keeps R_F and the sense amplifier enabled, and 3) simultaneously activates many rows. Finally, the sense amplifier overwrites all activated rows with the source row's data."
- So Multi-RowCopy = source row gets a full RAS-time sense-amp resolution to V_DD/0; PRE is short-circuited by the second ACT before bitlines drift; the second ACT opens N additional destination rows whose cells then accept the latched source value from the sense amp. The sense amp is the source of the broadcast.

### 1.2 FracDRAM — Fractional Values (Frac, Half-m)

**Frac operation — store V_DD/2 in an entire row (FracDRAM §III-A, Fig. 3 / lines 167–195)**

- Command sequence: `PRECHARGE  ACTIVATE(R_target)  PRECHARGE  [≥5 cycles idle]`, where the second PRE comes in the cycle *immediately after* the ACT — back-to-back, no idle cycles between (line 181: "two basic DRAM commands... need to be issued back-to-back without any extra cycles in between").
- 2.5 ns memory cycle (line 156, line 280–282). Total latency = 7 memory cycles (2 commands + 5 idle).
- Per-step bitline + cell state (lines 187–195):
  1. First PRE drives bitline to V_DD/2.
  2. ACT raises wordline → cell capacitor connects to bitline → charge sharing begins. Bitline + cell equilibrate at "slightly higher than V_DD/2" if the cell stored V_DD (because bitline capacitance ≫ cell capacitance, the equilibrium is closer to the initial bitline voltage). Line 159: "The equilibrium voltage is closer to the initial bit-line voltage because the bit-line capacitance is much larger than the cell's."
  3. Immediate next-cycle PRE **interrupts row activation before the sense amplifier is enabled**. This is the load-bearing point — PRE is started, which de-asserts the wordline and disconnects the cell from the bitline, leaving the cell capacitor holding the partially-shared voltage. The sense amp is never enabled, so the cell value is *not* amplified back to V_DD/0.
- Repeating Frac drives the cell voltage progressively toward V_DD/2: each iteration starts the bitline at V_DD/2 (from the leading PRE) and equilibrates the cell with it; if the cell is already ~V_DD/2, no change; otherwise the cell moves toward V_DD/2. The paper proves this empirically via retention-time profiles (§V-A, Fig. 6).
- **The mechanism produces a non-binary stored voltage because the sense-amp is never enabled to restore the cell to its rail; the cell capacitor simply holds whatever voltage charge-sharing produced.** This is the entire story.

**Half-m operation — store Half in selected bits within a row, using four-row activation (FracDRAM §III-B, Fig. 4 / lines 226–263)**

- Command sequence: `PRE  ACT(R1)  PRE  ACT(R2)  PRE` (line 238–239). The middle three commands are the four-row-activation sequence; the trailing PRE interrupts before the sense amp resolves.
- The trailing PRE prevents the sense amp from being enabled, so the bitline (and connected cells) holds whatever the four-row charge-shared value is. For 4 rows storing (1,1,0,0) per column the equilibrium ≈ V_DD/2; for (1,1,1,1) or (0,0,0,0) ≈ V_DD or 0 (lines 254–257).
- "Without the help from the sense amplifier, both logical one and zero are not fully recovered to V_DD and 0... thus we call them 'weak' ones and zeros." (lines 263–266). The Half cells, weak-1 cells, and weak-0 cells all coexist within the same row depending on which initial-value combo the four rows held in that column.

**F-MAJ — MAJ3 using four-row activation with one neutralized row (FracDRAM §VI-A, lines 451–479)**

- Open four rows with the standard four-row-activation `ACT(R1)-PRE-ACT(R2)`. One of the four cells (per bitline) holds a Frac value (≈V_DD/2). The other three cells hold the three MAJ3 inputs.
- "The row that holds fractional values, which is close to V_DD/2, will have the least influence on the bit-line voltage during the charge sharing, and thus the final result will depend on the majority voltage among the other three rows." (lines 442–445). The V_DD/2 cell contributes nothing to the perturbation away from V_DD/2; the other three cells act as MAJ3 inputs.

### 1.3 FCDRAM — NOT, NAND, NOR, AND, OR via Neighboring-Subarray Activation

**Command sequence** (FCDRAM §4.1, line 381): `ACT R_F  PRE  ACT R_L` with reduced `tRAS` and `tRP` (e.g., `tRP < 3 ns`; line 386). Crucially `R_F` and `R_L` are in **two neighboring subarrays** that share sense amplifiers via the open-bitline architecture.

**Open-bitline architecture, the structural prerequisite** (FCDRAM §1, lines 112–137, and §2.1, lines 207–216)

- "In the widely adopted open-bitline DRAM architecture, two relatively far apart DRAM cells (e.g., Cell A and Cell B in Fig. 1a) connect to two opposite terminals of a sense amplifier via access transistors."
- "Shortly after a sense amplifier is enabled to access a DRAM cell, the opposite terminals of the sense amplifier are fundamentally driven by inverted voltage levels (i.e., inverted logic values) due to how the sense amplifier operates."
- "Only enough sense amplifiers are fitted in a row to sense half of the cells. To sense the entire row of cells, each subarray has bitlines that connect to two rows of sense amplifiers, one above and one below the subarray, which causes neighboring subarrays to share half of the sense amplifiers." (lines 210–214). **This is the load-bearing fact for FCDRAM: half of every subarray's bitlines run to a sense-amp row that the neighbor subarray's other half of bitlines also connects to, on the *opposite* terminal of those sense amps.**

**NOT mechanism (FCDRAM §5.1, Fig. 6 / lines 481–545)**

- Steps:
  1. `ACT src` and wait `tRAS` — sense amp resolves; src bitline = src value, src bitline-bar = ¬src value (lines 535–539).
  2. `PRE` with `tRP < 3 ns` — does NOT fully de-assert wordlines or precharge.
  3. `ACT dst` (in the *neighbor* subarray) → opens dst → dst is now connected to the bitline-bar that's already driven to ¬src. Charge sharing drives dst toward ¬src. Sense amp is still enabled (because PRE was short), so it restores dst to ¬src cleanly.
  4. Wait `tRAS` then `PRE` to finalize.
- Net effect: src is unchanged, dst stores ¬src. Half the cells in any dst row (the half whose bitlines are shared with src's subarray through the neighboring sense amps) get NOT'd; the other half do not change (line 405: "half of the cells in the simultaneously activated row(s) in R_L's subarray stores the negated value of the data pattern sent with the WR command. The remaining half retain their initial values"). The paper does not address how to NOT the "other half"; it accepts the half-bitwidth as a property of open-bitline.

**Many-input AND/OR mechanism (FCDRAM §6.1, Figs. 13–14 / lines 695–805)**

- Same `ACT R_REF  PRE  ACT R_COM` envelope, both with reduced timings. Now N rows are simultaneously activated in each of the two neighboring subarrays — N in the "reference" subarray, N in the "compute" subarray. Operation type is selected by what gets stored in the reference subarray before the operation.
- **N-input AND.** Initialize N–1 rows in REF with V_DD ("1"), and 1 row in REF with V_DD/2 (via Frac). The mean voltage on the REF bitline after charge sharing is `V_REF = ((N–1)·V_DD + 0.5·V_DD)/N = (N – 0.5)·V_DD/N`. The COM bitline charge-shares the N input cells: if all are V_DD, V_COM = V_DD > V_REF → sense amp drives COM bitline high; if any are 0, V_COM ≤ (N–1)·V_DD/N < V_REF → sense amp drives COM bitline low. The sense amp is acting as a comparator between V_COM and V_REF, with V_REF positioned just below V_DD and above the highest "fail" value (lines 753–772).
- **N-input OR.** Initialize N–1 rows in REF with 0 (GND), 1 row with V_DD/2 → `V_REF = 0.5·V_DD/N`. Output = 1 if V_COM > V_REF (i.e., any input is 1); 0 otherwise. (lines 773–786).
- **NAND/NOR** come "for free" on the REF subarray's side: since the REF and COM bitlines are connected through the sense amp's two terminals (one terminal is the inverse of the other), the REF cells get ¬AND = NAND or ¬OR = NOR written back into them (lines 768–771).

**What the paper is silent on**: FCDRAM does not address how to *initialize* the N–1 V_DD rows and 1 V_DD/2 row in the REF subarray efficiently for a given N — it assumes the user can. The Frac for the 1 row is referenced to FracDRAM (footnote 11, line 743).

---

## 2. The Role of Replication (SiMRA's central claim)

**The empirical evidence** (SiMRA §5, Fig. 6, line 444): "MAJ3 with 32-row activation achieves 30.81% higher success rate than MAJ3 with 4-row activation." Replication factor of 10 (since 32/3 ≈ 10) lifts MAJ3 success rate from ~68% to ~99%.

Storing multiple copies of each MAJX operand also lifts the success rate of MAJ5/MAJ7/MAJ9 (Obs. 10, lines 446–449): random data, increase from 4-row to 32-row activation gives +56.27% for MAJ5, +35.15% for MAJ7, +13.11% for MAJ9.

**The paper's hypothesis and SPICE-backed mechanism** (SiMRA §7.2 lines 742–763)

> "We hypothesize that by storing multiple copies of MAJX input operands on all simultaneously activated rows (which we call input replication), the bitline voltage can be increased and perturbed towards a safer margin and, thus, potentially increase the success rate of MAJX operations."

The mechanism is *bitline perturbation magnitude*, not noise averaging:
- With single-row activation: one cell sharing charge with the bitline → small perturbation (~0.1 V order, but the precise value is process-variation dependent).
- With K-row activation: K cells share charge in parallel → bitline ends up further from V_DD/2 in proportion to the imbalance among the K cells. The sense amp has a noise margin around its threshold; a larger perturbation crosses that margin reliably even under process variation.

SPICE result (Fig. 15a / lines 705–710): "Performing MAJ3 with 32-row activation (i.e., ten copies for each input operand) has 159.05% higher bitline voltage perturbation than performing MAJ3 with 4-row activation on average."

SPICE result on process-variation tolerance (Fig. 15b / lines 717–720): "The success rate of MAJ3 with 4-row activation reduces by 46.58% when process variation increases from 0% to 40%. In contrast, the success rate of MAJ3 with 32-row activation reduces only by 0.01%."

The paper explicitly rejects "more cells = noise averages out" framing — the explanation is "bigger differential signal pushes past the sense-amp's process-variation-induced threshold scatter." That is *signal amplitude*, not averaging.

**Limit of the paper's explanation**: SiMRA does NOT explain *why* the bitline perturbation scales the way it does as a function of K and the input pattern. The SPICE simulation shows it does, but the paper does not derive a closed-form for `V_BL(K, n_ones)`. FCDRAM §6.1 has the closest analytical statement: V_BL after charge-sharing of N cells holding `n_ones` 1s and `(N – n_ones)` 0s equals `n_ones·V_DD/N` (line 736, assuming negligible bitline capacitance — a simplifying assumption the paper flags in footnote 10).

---

## 3. Chip-Specific vs General

### SiMRA

| Aspect | Chip-specific (per SiMRA) | General (per SiMRA) |
|---|---|---|
| Timing values (`t1`, `t2`) | Best timing = (1.5 ns, 3 ns) for MAJ3; (3, 3) for many-row activation; (36, 3) for Multi-RowCopy — observed across 120 DDR4 chips from 2 manufacturers (Mfr.H / SK Hynix and Mfr.M / Micron). Samsung (64 chips) does NOT support multi-row activation at all (Limitation 1, lines 890–908). | The *form* of the APA sequence (ACT-PRE-ACT with violated tRAS/tRP) is general. |
| Supported K (rows activated) | Tested chips support exactly {2, 4, 8, 16, 32} rows simultaneously, not arbitrary counts (Limitation 2, lines 909–922). The paper attributes this to 1.5-ns command resolution of DRAM-Bender. | The upper bound = 2^(number of predecoders) = 2^5 = 32 on the analyzed chip (line 740). General principle: row decoder predecoder count determines max K. |
| Success-rate margins (per-chip) | Quantitative success-rate numbers (e.g., MAJ5 32-row = 79.64%; MAJ7 = 33.87%; MAJ9 = 5.91%, line 103–105) are chip-/manufacturer-specific. | Qualitative ordering "more replication → higher success rate" is robust across all tested chips. |
| Sample of generality | Mfr.M does NOT support `Frac`; works around it by exploiting that "sense amplifiers are always biased to one or zero" so initializing neutral rows with all-0 or all-1 works (footnote 5, lines 311–312). |  |

### FracDRAM

| Aspect | Chip-specific | General |
|---|---|---|
| Whether Frac works | Groups A–I (SK Hynix + Samsung + TimeTec + Corsair, etc.) support Frac; groups J/K/L (Micron, Elpida, Nanya) do not (lines 348–354): "We speculate that those DRAM chips implement time checking circuits to prevent different DRAM commands being executed too close to each other." | The conceptual mechanism (interrupt ACT with PRE before sense amp is enabled, leaving cell at fractional voltage) is general to any DRAM that obeys the issued commands. |
| Three- vs four-row activation | Group B opens 3 rows (DDR3, SK Hynix 1333MHz); Groups C, D open 4 rows; some open 2^k for k ≥ 1 (Table I, lines 198–219). | F-MAJ pattern (use V_DD/2 cell in 1 of 4 rows to demote it to a "neutral" input) is general wherever 4-row-activation is achievable. |
| Memory cycle assumption | 2.5 ns (SoftMC fixed at 400 MHz, line 280) — *all* "memory cycle" counts in the paper are referenced to this. | Conceptual back-to-back command requirement (no idle cycles between ACT and PRE for Frac) generalizes; absolute numbers don't. |

### FCDRAM

| Aspect | Chip-specific | General |
|---|---|---|
| Whether bitwise ops work | SK Hynix: full set works. Samsung: only NOT works (only sequential 2-row activation in neighbors). Micron: nothing works (no neighbor-subarray activation). 256 chips characterized. (lines 320–323, lines 575–581) | Conceptual mechanism (neighbor-subarray simultaneous activation through shared sense amps) requires the chip to obey the APA sequence under violated timings. |
| N:N vs N:2N activation pattern | Some Hynix modules support both → max 48 rows activated; others only N:N → max 32 rows (Obs. 2 lines 415–417, lines 423–428). | Both pattern types are explained by the same hypothesized row decoder design. |
| Success rate scaling with N | 16-input NAND/NOR/AND/OR success rates ~95% in the tested SK Hynix population (line 33). Numbers shift significantly with chip density and die revision (Obs. 19, line 962): e.g., 2-input AND success rate −27.47% from 4Gb A-die to 4Gb M-die. | The qualitative ordering (success rate falls with N inputs, falls with imbalanced V_REF) is general. |
| Distance-to-sense-amp variation | Per-chip "design-induced variation" — average success rate of NOT varies 85% (Middle src + Far dst) vs 44% (Far src + Close dst) (lines 622–630). | Open-bitline architecture is general; quantitative magnitude is chip-specific. |

---

## 4. Vocabulary — Project Terms ↔ Paper Terms

| Project term (server code) | SiMRA term | FracDRAM term | FCDRAM term |
|---|---|---|---|
| `doubleACT(t_12, t_23, r_first, r_second)` | `ACT R_F –t1– PRE –t2– ACT R_S` ("APA" sequence, §3.2 line 234) — `t_12` = `t1` (ACT→PRE), `t_23` = `t2` (PRE→ACT), `r_first` = `R_F`, `r_second` = `R_S` | `ACTIVATE(R1)-PRECHARGE-ACTIVATE(R2)` (§II-D line 156); paper assumes both intervals are 0-idle | `ACT R_F  PRE  ACT R_L` (§4.1 line 381) |
| "RowClone" (intra-subarray 1→1 copy) | Implicit: "consecutive activation of two DRAM rows in the same subarray" (footnote 6, line 318) — same APA sequence with `t1 = tRAS = 36ns` and `t2 ≈ 3ns`. SiMRA cites RowClone [67] for the modified-DRAM origin (§2.2 lines 137–145). | "intra-subarray RowClone operation" (§II-D line 158) — same `ACTIVATE-PRE-ACTIVATE` mechanism, different timing | Cited but not extended — "RowClone... can be performed in COTS DRAM chips by enabling sequential activation of two DRAM rows in the same subarray" (§2.2 lines 294–296) |
| "Broadcast" (intra-subarray 1→N) | **Multi-RowCopy** — defined SiMRA §3.4 (lines 313–328). Same APA sequence as RowClone but the second ACT opens many rows because timing pattern matches a many-row predecoder configuration. | Not defined in FracDRAM. | Not defined in FCDRAM (FCDRAM's analog is the NOT operation, which writes ¬src into multiple dst rows in the neighboring subarray). |
| "MAJ3" | MAJ3 (§2.2 line 187). Defined for 4-row, 8-row, 16-row, 32-row activation with input replication (§3.3 line 276–298). | MAJ3 (§II-D line 162); F-MAJ for the four-row-activation variant (§VI-A line 451). | MAJ3 cited but not extended; FCDRAM works on 2-input AND/OR/NAND/NOR built on the same charge-share+comparator primitive. |
| "frac" | "Frac" — used to make "neutral rows" that don't contribute to the bitline perturbation (§3.3 line 302; footnote 4 line 322). Cites FracDRAM [129]. | **Frac** — defined operation, §III-A line 167. | "Frac" — used to set up V_DD/2 reference cell in AND/OR (§6.2 line 794; footnote 11 line 743). Cites FracDRAM [38]. |
| "open rows" (simultaneously activated rows) | "simultaneously activated rows" / "simultaneous many-row activation" / "N-row activation" (passim from §1 onwards). | "multiple-row-activation" (§II-D line 149); "three-row-activation", "four-row-activation". | "simultaneous multiple-row activation in neighboring subarrays" (§4 line 354); `N_RF : N_RL` activation pattern terminology (§4.2 line 416). |

Note on terminology drift: the SiMRA paper's "Multi-RowCopy" is a single-source-to-many-destinations operation *within one subarray*. The project's "Broadcast" maps to this exactly. FCDRAM's NOT-with-many-destination-rows (§5.3 Obs. 4 line 549–551) is a related but distinct operation across two subarrays — it writes ¬src into many dst rows in the *neighbor* subarray, with success rate degrading as the destination row count grows (98% at 1 dst → 8% at 32 dsts).

---

## 5. Index of Claims with Citations

Every claim above is sourced from this table. "Line" = line number in the local `.txt` extract; "Loc" = section/figure in the paper.

### SiMRA

| Claim | Loc | Line(s) |
|---|---|---|
| APA = `ACT R_F –t1– PRE –t2– ACT R_S`; both rows in same subarray | §3.2 | 234–235 |
| Best timing for many-row activation = (t1=3, t2=3) ns; 32-row activation success rate 99.85% | §4 Fig. 3 / Obs. 1 | 333–355 |
| Best timing for MAJ3 32-row activation = (t1=1.5, t2=3) ns; 99% success rate | §5 Fig. 6 / Obs. 7 | 461–464 |
| Best timing for Multi-RowCopy = (t1=36, t2=3) ns; 99.98–99.99% success rate | §6 Fig. 10 / Obs. 14 | 500–506 |
| MAJ3 mechanism: three cells charge-share onto bitline; sense amp resolves to majority | §2.2 | 187–196 |
| Replicating MAJX inputs across N rows (N/X copies each); leftover N mod X rows are Frac'd ("neutral") | §3.3 | 294–302 |
| MAJ3 32-row activation vs 4-row activation: +30.81% success rate | §5 Obs. 6 | 444–447 |
| MAJ5/7/9 32-row vs 4-row: +56.27% / +35.15% / +13.11% success rate (random data) | §5 Obs. 10 | 446–449 |
| SPICE: 32-row activation gives 159.05% more bitline perturbation than 4-row | §7.2 Fig. 15a | 707–710 |
| SPICE: 4-row MAJ3 success rate falls 46.58% from 0% to 40% process variation; 32-row falls only 0.01% | §7.2 Fig. 15b | 717–720 |
| Multi-RowCopy mechanism (4 steps: ACT src, wait tRAS, PRE, ACT dst with <3ns tRP) | §3.4 | 313–328 |
| Row decoder hypothesis: hierarchical (GWLD + LWLD with 5 predecoders); 2^5 = 32 max rows | §7.1 Fig. 13 | 626–740 |
| Manufacturer split: SK Hynix + Micron support multi-row; Samsung does not | Limitation 1 | 890–908 |
| Supported K ∈ {2, 4, 8, 16, 32}, controlled by row-address patterns; finer K control would need <1.5ns command resolution | Limitation 2 | 909–922 |
| For Mfr.M, Frac is not supported; sense amps are biased so all-0/all-1 neutral rows work | footnote 5 | 311–312 |

### FracDRAM

| Claim | Loc | Line(s) |
|---|---|---|
| Frac sequence: `PRE ACT(R_target) PRE` (back-to-back, no idle); 7 memory cycles | §III-A | 167–195 |
| Frac mechanism: second PRE interrupts row activation before sense amp is enabled; cell holds partially-shared voltage | §III-A | 184–195 |
| Repeated Frac drives cell toward V_DD/2 because bitline cap >> cell cap | §III-A | 159, 169–174 |
| Half-m sequence: `PRE ACT(R1) PRE ACT(R2) PRE` (four-row-activation with trailing PRE) | §III-B | 236–239 |
| Half-m mechanism: trailing PRE prevents sense-amp enable; cells hold the 4-row charge-shared value | §III-B | 241–266 |
| Without sense amp, "weak 1" and "weak 0" coexist with Half values | §III-B | 263–266 |
| F-MAJ: 4-row activation with 1 row pre-Frac'd to V_DD/2; that row contributes least; other 3 drive MAJ3 | §VI-A | 451–479 |
| Vendor support: groups A–I support Frac; J/K/L (Micron, Elpida, Nanya) do not | §V-A | 348–354 |
| Memory cycle = 2.5 ns (SoftMC 400 MHz fixed) | §IV-A | 280–283 |

### FCDRAM

| Claim | Loc | Line(s) |
|---|---|---|
| Command sequence: `ACT R_F PRE ACT R_L` with R_F, R_L in *neighboring* subarrays; reduced tRAS, tRP | §4.1 | 381–394 |
| Open-bitline architecture: neighboring subarrays share half the sense amps; the shared sense amp inverts between its two terminals | §2.1 | 207–216 |
| NOT mechanism: ACT src + tRAS, PRE <3ns, ACT dst → dst gets ¬src via the inverted bitline-bar; sense amp restores it | §5.1 Fig. 6 | 481–545 |
| NOT operates on half the dst row only (the half whose bitlines are shared with the src subarray) | footnote 6 | 519, 405 |
| N:N vs N:2N activation patterns in neighbor subarrays; max 32 rows (N:N only) or 48 rows (N:N + N:2N) | §4.3 Obs. 2 | 415–428 |
| AND mechanism: REF subarray V_REF = (N-0.5)V_DD/N; COM subarray V_COM = (n_ones)V_DD/N; sense amp compares | §6.1.2 Fig. 14 | 753–772 |
| OR mechanism: V_REF = 0.5 V_DD/N; same comparator logic | §6.1.2 | 773–786 |
| NAND/NOR appears "for free" on the REF side because sense amp inverts | §6.1.3 | 768–771 |
| Frac (cited from FracDRAM [38]) sets up the 1 V_DD/2 cell needed for V_REF | footnote 11 | 743 |
| Manufacturer split: SK Hynix full set works; Samsung only NOT (sequential 2-row in neighbors); Micron nothing | §7 Lim. 1 | 320–323, 575–581 |
| 16-input AND/NAND/OR/NOR avg success rates: 94.94% / 94.94% / 95.85% / 95.87% (SK Hynix population) | §6.3 Obs. 10 | 858–867 |
| NOT success rate vs # destination rows: 98.37% at 1 dst → 7.95% at 32 dsts | §5.3 Obs. 4 | 549–553 |
| Success rate varies strongly with src/dst distance to shared sense amp (85% Middle-Far → 44% Far-Close) | §5.3 Obs. 6 Fig. 9 | 621–635 |
| AND/OR success rate depends on number of logic-1s in the inputs (small voltage diff at extreme imbalance) | §6.3 Obs. 14 Fig. 16 | 857–889 |

---

## 6. Explicit Gaps (where the paper is silent or only empirical)

- **SiMRA**: Bitline-level mechanism for how a WR command rewrites N simultaneously activated rows is asserted via citations [86, 173], not derived. The voltage-perturbation formula as a function of K and the input pattern is shown empirically and via SPICE but not given as a closed-form equation.
- **SiMRA**: No mechanism is offered for why MAJ5/7/9 success rates degrade as K grows beyond 3 — only that they do. The paper attributes failure to sense-amp threshold scatter, but does not derive the noise margin as a function of K and replication factor.
- **FracDRAM**: The mechanism for *why* certain vendors (Micron, Elpida, Nanya) reject the Frac sequence is speculative ("time checking circuits", line 351) — no schematic evidence.
- **FracDRAM**: Half-m's 16% effective coverage (line 408) is reported empirically without bitline-level explanation of why most columns fail.
- **FCDRAM**: The "design-induced variation" cited for src/dst distance effects is referenced to prior work [106]; FCDRAM does not derive bitline-resistance / sense-amp-strength models.
- **FCDRAM**: How to NOT the *other half* of a row (whose bitlines are not shared with the neighbor sense amps) is not addressed.
- **All three papers**: Treat the row decoder hypothesis (5 predecoders, 2^5 = 32 max simultaneous rows) as the *single* explanation for "why 32?". This is plausible from observed address patterns but not validated against actual silicon (the vendors do not publish row-decoder schematics, as the SiMRA authors note at lines 627–632).
