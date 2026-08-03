# Mechanism explainer — claim-evidence ledger

**Artifact:** `docs/explainer/xor-spread.html` — *"One command pair, two physics"*
(the mechanism explainer; path pinned by the two external SAFARI issues).
**Rebuilt:** 2026-08-03, public-docs overhaul. This ledger is the publish gate:
every factual statement in the explainer maps to a paper, production code, a
measured log, or a claim register entry. Chip-specific numbers are labelled in
the explainer itself as measured examples on named silicon (tuple `s61`,
bender 0, SK hynix, bank 0), never as universal laws.

**Supersedes** the July deck's ledger `xor_spread_ledger.md` (kept as prior
provenance for the retired 11-scene deck). The rebuilt explainer drops the
scene structure and the two scene framings the overhaul corrects — the claim
that the spread ignores timing, and the claim that the vote unconditionally
overwrites its own operands. This ledger covers the rebuilt content only.

## Source tiers

- **[paper:SiMRA]** — Yüksel et al., *Simultaneous Many-Row Activation in
  Off-the-Shelf DRAM Chips* (SiMRA-DRAM); Multi-RowCopy / MAJX substrate.
- **[repro]** — `kubo_xorspread_repro_2026_08_03/` : `RESULT.md` +
  `kubo_xorspread_reproducer.md` (the row-by-row deliverable).
- **[log:NAME]** — a raw measurement log under `kubo_xorspread_repro_2026_08_03/`.
- **[claim:CNN]** — `MENTAL_MODEL.md` §R register entry (measured, dated).
- **[data:PATH]** — measurement file staged under `docs/data/`.
- **[code:PATH]** — reproducer / production source in this repository.
- **[issue:URL]** — linked SAFARI GitHub issue.
- **[mem:NAME]** — project memory note (cross-die / selection-law provenance).
- **[example]** — chip-specific value, labelled EXAMPLE in the explainer.
- **[editorial]** — framing, grounded in the adjacent cited rows.

---

## §1 — A row address is just bits (decoder anatomy)

| Claim | Source |
|---|---|
| A row address is decoded by a hierarchical decoder; low bits are split across predecoder units, each driving wordlines | [paper:SiMRA §7.1] (hierarchical row decoder / local predecoders) |
| Address relationships are named by the differing bit; the four generators are bit 2 = +4, bit 4 = +16, bit 6 = +64, bit 9 = +512 | [repro §2] (generators {4,16,64,512}); [example] — this tuple's offsets |

## §2 — The double activation (one command pair)

| Claim | Source |
|---|---|
| The primitive is one command pair `ACT … first hold … PRE … gap … ACT`; first hold = t12, gap = t23 | [repro §2] (`ACT(38424) → t12 → PRE → t23 → ACT(38988)`); [paper:SiMRA §3.4] |
| Shortening the timings co-activates rows (the second ACT opens while the first still drives the wires); this is SiMRA's Multi-RowCopy mechanism, not a defect | [paper:SiMRA §3.4, §7.1]; [issue:https://github.com/CMU-SAFARI/DRAM-Bender/issues/12] |

## §3 — The timing dial (VOTE / TIE / COPY) + the second knob

| Claim | Source |
|---|---|
| One command pair, two physics selected by timing: clean majority vote at the operating point; the same pair becomes a multi-row copy a few slots away | [claim:C65]; [repro §4 "How the deposit was placed"] |
| First-hold (t12) map on bender 0 / s61: clean vote at 0–1 slots, tie at 2, full copy at ≥3 (every non-source operand takes the source's value byte-exact) | [claim:C65]; [log:kubo_maj_probe_b0.log] (t12=0/1 → 15 vote clean; t12=2 → mixed/tie; t12≥3 → 15 polluted, byte-exact 2048/2048) |
| The vote/tie/copy boundary is measured on three dies (2× SK hynix + 1 Micron); the copy never fires at the clean vote point; where it begins is per-die | [claim:C67] (D2 SK hynix), [claim:C68] (D3 Micron), [claim:C65] (b0); [repro §4]; overall verdict [repro RESULT "OVERALL at-(0,0) VERDICT"] |
| Second knob — the PRE→2nd-ACT gap (t23): a clean 2-row copy needs ≥ 4 NOP slots = 6.0 ns (tCK = 1.5 ns); below that the copy is multi-row | [claim:C64]; [log:kubo_demo_b0_run1.log] (t23 ≤3 → 3/3 contaminated; t23 ≥4 → 3/3 clean; header "t23=4 = Kubo 6ns clean boundary") |
| First-hit-wins: the standard SiMRA calibration scans the gap upward from 1 and stops on first success (= 1 slot, multi-row territory), so a calibrated "RowClone" already broadcasts to the coset — not a reliability choice | [paper:SiMRA §3.2] (gap scan / calibration); [claim:C64]; [editorial: consequence of C64 + the calibration scan] |
| tCK = 1.5 ns; 4 slots = 6.0 ns | [claim:C64] ("≈6 ns @ tCK=1.5 ns"); [log:kubo_demo_b0_run1.log] header |

## §4 — The vote, derived (the single binary figure)

| Claim | Source |
|---|---|
| A = F0, B = CC, C = AA; per-bit majority MAJ(F0,CC,AA) = E8 | [repro §2] (worked constants; bit-aligned majority table); arithmetic majority |

## §5 — The coset + the selection law

| Claim | Source |
|---|---|
| The copy lands in the coset of the pair's generators: 16 rows = every subset-sum of {4,16,64,512} on base 38408 (source 38424, second 38988) | [repro §2] (the 16-row table); [log:kubo_maj_campaign_b0.log] (16-row list, idx 0–15) |
| The source's four single-generator neighbours are 38408 (bit 4), 38428 (bit 2), 38488 (bit 6), 38936 (bit 9) | [repro §4]; [log:kubo_maj_campaign_b0.log] (substituted idx {0,3,6,10} = these four rows) |
| Selection law: on SK hynix the subarray-local bits group as {1,2}{3,4}{5,6}{7,8} with bits 0, 9 singleton; a candidate A⊕S fires iff for every group g, S∩g ∈ {∅, d∩g} | [mem:selection_law]; [data:docs/data/selection-law/selection_timing_b0.csv, selection_timing_b2.csv] |
| The law accounts for 1691/1691 observed rows on each of two SK hynix dies, zero exceptions | [data:docs/data/selection-law/] (1691 member rows/die, zero exceptions); [mem:selection_law] |
| For s61 each generator sits alone in its group, so all 16 combinations fire | [repro RESULT "Geometry"] (coset == the 16 open rows); [mem:selection_law] (firing count = 2^(#units d touches)) |
| Digital selection (bank-invariant, byte-exact, deterministic across power cycles and rigs), analog firing (timing/charge gated) | [mem:selection_law] (timing-invariant selection, 8 timing combos); [claim:C65] (firing timing-gated); [mem:cross_die_determinism] |
| A Micron die shows the same physics with a different grouping | [claim:C68] (D3 Micron deposit real + timing-gated); [mem:selection_law] "DIMM 1/3 … no clean partition" |

## §6 — Good case / bad case row tables + k-sweep

| Claim | Source |
|---|---|
| Layout A(6 incl. both opened)/B(6)/C(4); worst tally 6-vs-10, honest majority wins by ≥4 | [repro §2]; [log:kubo_maj_campaign_b0.log] (grp column: A×6, B×6, C×4) |
| Good case: all 16 rows end at E8 (destructive vote writes result back into every operand) | [repro §3]; [log:kubo_maj_campaign_b0.log] arm baseline |
| Bad case: source F0 deposited into the four C rows (38408/38428/38488/38936), freeze-frame byte-exact, other 11 untouched; vote then returns F0 | [repro §4]; [log:kubo_maj_campaign_b0.log] (phase-1 idx{0,3,6,10}=R_first 2048/2048); [claim:C66] |
| k-sweep: k=0,1 → E8; k=2 → tie; k=3,4 → F0 — flip exactly at the arithmetic k=3 | [claim:C66]; [log:kubo_maj_ksweep_b0.log] (k=0,1 Pprep; k=2 tie 8–8; k=3,4 Psubst; P_prepared 0/2048 at k≥3) |
| Honesty note: raw votes are analog (sub-percent same-session flake); deposits are byte-exact (all 8192 bytes) | [repro §4 honesty note]; [repro RESULT "pim_numerics_gating"] |

## §7 — Hazard and asset — decided by placement, not timing

| Claim | Source |
|---|---|
| The same coupling is a free one-to-many copy; hazard vs asset is decided by placement, not timing | [repro §4 "The spread is a tool"]; [code:app/test_safe_load.cpp] |
| Neutralize by placement: offsets whose generators avoid the coset are corruption-free by construction (20/20 clean safe loads; unsafe offsets corrupt exactly the predicted rows) | [code:app/test_safe_load.cpp]; [doc:LATTICE_ADDRESSING_2026_07 §2]; prior ledger `xor_spread_ledger.md` Scene 11 |
| Exploit as broadcast: the fused-coset activation path is in production, measured 1.63×/token on the real model (both arms answer "Paris") | [code:python/run_bitnet_pim.py] + [code:app/test_bitnet_server.cpp] (A/B, env-gated coset path); prior ledger `xor_spread_ledger.md` Scene 11 (117.2 → 71.8 s = 1.63×); [mem:bitnet_fused_coset_production] |
| Broadcast weight loading validated + queued | [doc:LATTICE_ADDRESSING_2026_07 §1] (sub-lattice broadcast, bit-exact); [doc:ROADMAP.md] (queued lever) |
| Keeping the short "dirty" calibration timing on purpose (widening the gap suppresses the free copies) | [repro §4]; [claim:C64] |

## §8 — Provenance and credit

| Claim | Source |
|---|---|
| The multi-row co-activation is SiMRA's own Multi-RowCopy; credit is theirs | [issue:https://github.com/CMU-SAFARI/DRAM-Bender/issues/12] (SiMRA co-author confirmation); [repro RESULT "Provenance framing"] |
| Our contribution: the address algebra (coset of the pair's generators), the selection law, and the vote-over-substituted-operands consequence | [mem:xor_spread_provenance]; [mem:selection_law]; [claim:C66] |
| Candidate selection byte-identical across our two SK hynix modules (two different part numbers) | [mem:cross_die_determinism] (byte-identical fault set across two SKUs + a rig change) |
| External exchange: DRAM-Bender #12 and SiMRA-DRAM #1 | [issue:https://github.com/CMU-SAFARI/DRAM-Bender/issues/12]; [issue:https://github.com/CMU-SAFARI/SiMRA-DRAM/issues/1] |

---

## Scope discipline (the claims the explainer holds itself to)

- The deposit is **timing-gated on every die measured** and does not fire at
  the clean vote point; operand corruption is never stated as unconditional or
  as intrinsic to the vote itself ([claim:C65], [claim:C67], [claim:C68]).
- *Selection* (which rows are candidates) is timing-invariant; *firing*
  (whether a deposit happens) is timing-gated — the explainer states both
  halves and never collapses them into a single blanket claim
  ([mem:selection_law], [claim:C65]).
- The co-activation is credited to SiMRA's Multi-RowCopy; only the address
  algebra + its consequences are claimed as ours — never presented as an
  unattributed discovery ([issue:.../DRAM-Bender/issues/12], [mem:xor_spread_provenance]).
- The multi-timing instrument-composition methodology point is **not** in this
  artifact; it lives once in the peer doc (`docs/RELATED_SYSTEMS.md`), per the
  single-source rule.
