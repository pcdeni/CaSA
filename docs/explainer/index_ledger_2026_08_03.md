# System explainer — claim-evidence ledger

**Artifact:** `docs/explainer/index.html` — *"How an LLM runs inside a DRAM chip"*
(the layman system explainer; the GitHub-Pages default at `explainer/`).
**Rebuilt:** 2026-08-03, public-docs overhaul. This ledger is the publish gate:
every factual statement in the explainer maps to a paper, production code, a
measured log, or a claim-register entry. The one chip-specific worked example
(tuple `s61`, bank 0, an SK hynix module) is labelled in the page as real,
named silicon.

The explainer
is a nine-section plain-language walk; its central object is a six-step
instruction walk (setup → write → clone → vote → read → iterate). It re-explains
none of the mechanism owned by the mechanism explainer — the coset, the
selection law, the timing dial, and the single binary vote-derivation figure all
live in `xor-spread.html`, linked, never restated here.

## Source tiers

- **[paper:SiMRA]** — Yüksel et al., *Simultaneous Many-Row Activation in
  Off-the-Shelf DRAM Chips*; the double-activation / MAJ-of-K substrate.
- **[fundamentals]** — standard DRAM operation (ACT/READ/WRITE/PRE,
  destructive read + restore, refresh, the cell→row→subarray→bank hierarchy);
  see `docs/explainer/paper_mechanism_notes.md` and `docs/HARDWARE.md`.
- **[repro]** — `kubo_xorspread_repro_2026_08_03/` (`RESULT.md`,
  `kubo_xorspread_reproducer.md`): the s61 worked example, the A/B/C/E8
  constants, the real addresses.
- **[claim:CNN]** — `MENTAL_MODEL.md` §R register entry (measured, dated).
- **[code:PATH]** — production / orchestration source in this repository.
- **[doc:NAME]** — a repo doc that carries the cited number.
- **[mem:NAME]** — project memory note.
- **[editorial]** — framing, grounded in the adjacent cited rows.

---

## §01 — The promise

| Claim | Source |
|---|---|
| A published multi-billion-parameter LLM answers a question (e.g. "capital of France?" → "Paris") with its matrix-multiplies executed inside DDR4 charge | [code:python/run_bitnet_pim.py] (full-model run answers "Paris"); [mem:bitnet_phase4f_pim_in_loop] |
| The memory does the arithmetic in place; weights sit in the array as charge; nothing about the DDR4 is modified | [doc:STOCKTAKE_2026_07_17.md] PROGRAM OF RECORD; [doc:HARDWARE.md] (stock DDR4) |
| What is unusual is the command *timing* — a controlled rule-break | [paper:SiMRA §3.4] (sub-spec activation timing); [claim:C64], [claim:C65] |
| Same physics as RowHammer/disturbance, opposite purpose; authorized research on owned hardware | [editorial]; framing carried from README |

## §02 — The substrate: a cell, ACT/READ/PRE

| Claim | Source |
|---|---|
| A DRAM bit is a capacitor (charged = 1, empty = 0); ACT tips it onto a shared bitline; a sense amplifier snaps the faint voltage to a clean 0/1 | [fundamentals] |
| Reading is destructive, so every read is a read-and-restore; refresh tops charge back up and preserves charge, not content | [fundamentals] |
| Three commands do the work: ACT (open a row), READ/WRITE, PRE (precharge/close) | [fundamentals]; [paper:SiMRA §3] |

## §03 — The layout: cell → row → subarray → bank → DIMM

| Claim | Source |
|---|---|
| Cells share a wordline and open together as a row; rows group into subarrays (own sense amps), subarrays into banks, banks into the DIMM; ACT opens a whole row at once | [fundamentals]; [paper:SiMRA §7.1] |

## §04 — The trick: open several rows and let the charge vote

| Claim | Source |
|---|---|
| Break the settling pauses and a small set of rows opens onto the same bitlines at once; their charge mixes; the sense amp snaps to the value most contributing rows held — that majority is the computation | [paper:SiMRA §3.4] (Multi-Row Activation / MAJX); [claim:C65] |
| The vote/copy line: the same pair stops voting and starts copying past a boundary of 4 command slots (6.0 ns; tCK = 1.5 ns) — shown here only as a static motif; the dial itself is the mechanism explainer's | [claim:C64]; [claim:C65]; owned by `xor-spread.html` (single-source) |

## §05 — The instruction walk (centerpiece)

| Claim | Source |
|---|---|
| Real addresses: tuple s61, bank 0, an SK hynix module; the vote opens 38424 first (source); the command names 38424 and 38988 | [repro §2]; [doc:calibration/calib_dimm0.txt] |
| The eight shown rows hold three F0, two CC, three AA and vote to E8 (the true reproducer layout; these 8 are a sub-set of the 16-row group and vote to E8 on their own) | [repro §2, §3] (A=F0/B=CC/C=AA, MAJ=E8; 38408/38428/38488 = C, 38412/38472 = B, 38424/38920/38988 = A) |
| The step spine — host writes activations (written), weights arrive by copying (copied), the pair votes (voted, result written back to every open row), one row is read back (read), the group iterates | [repro §3]; [code:app/test_bitnet_server.cpp] (write / RowClone / MAJ / readback loop); [mem:bitnet_phase4f_pim_in_loop] |
| Which rows a copy fills, and the exact copy-vs-vote timing, are deferred to the mechanism explainer (not re-explained here) | single-source rule; owned by `xor-spread.html` |

## §06 — From a vote to a matrix multiply

| Claim | Source |
|---|---|
| The model's weights are ternary (−1/0/+1), so a dot product needs only ANDs and counting | [paper: BitNet b1.58]; [code:python/pim_linear.py] |
| Split ternary weights into a +1 mask and a −1 mask; slice activations into bit-planes; AND each mask with each bit-slice (a majority with one input pinned low is an AND); popcount; host combines with place-values and subtracts the −1 count | [code:python/pim_linear.py] (bit-plane packing, pos/neg masks, popcount accumulate); [mem:bitnet_two_bugs_2026_05_05] |
| A whole projection is many thousands of votes plus a count (order-of-magnitude, non-numeric) | [editorial: order of magnitude from K-tiling in code]; softened to avoid an uncited hard figure |

## §07 — The inference loop and the CPU/DRAM split

| Claim | Source |
|---|---|
| In DRAM: every layer's projection matrix-multiplies run as votes and counts. On CPU: softmax, norms, sampling, tokenizer/embedding — the same split every PUD system makes | `docs/RELATED_SYSTEMS.md` §4 (all 210 BitLinear in DRAM; non-BitLinear on CPU) |
| One token is hundreds of thousands of votes across all layers (order-of-magnitude, non-numeric) | [editorial: order of magnitude]; softened |

## §08 — The wall (intuition only)

| Claim | Source |
|---|---|
| A vote takes tens of ns, but a token takes minutes because each small program is a fixed-cost host↔DRAM round-trip over PCIe, and there are thousands per token — the wall is the number of trips, not the physics | [claim:C01] (wall = round-trip COUNT × fixed latency); `docs/RELATED_SYSTEMS.md` §5 owns the numbers |
| Counting the yes-votes inside DRAM shrinks the answer that crosses the link | [claim:C-merge / in-DRAM popcount]; [mem:merge_indram_accumulation]; numbers deferred to peer doc |

## §09 — What runs today

| Claim | Source |
|---|---|
| Seven flagship model families run correctly on unmodified DDR4: BitNet-2B + two Bonsai forms live/token-exact on silicon; Llama-2 (7B and 13B), Llama-3, Phi-4 validated numerics-exact via the sampled end-to-end protocol | [claim:C46] (native trio token-exact); [claim:C53]/[claim:C54] (four mainstream validated); `docs/RELATED_SYSTEMS.md` §4 carries the split |
| The measured ladder runs 632 → 45 s/token; the memory-interface-bound floor is the target, not a current result; this does not beat a GPU today | [doc:STOCKTAKE_2026_07_17.md] PROGRAM OF RECORD; [claim:C03] (45 s/tok); [claim:C05] (floor target); numbers owned by README/peer doc |

---

## Scope discipline (what B holds itself to)

- **Single-source rule.** B teaches the substrate (cell / ACT-READ-PRE /
  hierarchy), the instruction walk, ternary-matmul-as-row-ops at layman level,
  the inference loop, and the readout-wall *intuition*. It does **not** explain
  the coset, the selection law, the timing dial, or the binary vote-derivation
  figure — all owned by `xor-spread.html` and linked, never restated.
- **No deposit / hazard-asset content.** The walk shows only a clean RowClone
  (`copied`); the copy-timing deposit, and hazard-vs-asset-by-placement, are the
  mechanism explainer's, linked once. No real address in the walk carries a
  value that contradicts `kubo_xorspread_reproducer.md` §3.
- **Corrected framing throughout.** The deposit is never described as intrinsic
  to the vote or as landing at the clean operating point; the spread's selection
  is not described as ignoring timing; the co-activation is credited to SiMRA, never
  presented as an unattributed discovery. B in fact carries no
  operand-deposit mechanism claim at all — the clean walk shows only a RowClone.
- **No historiography.** Present-tense knowledge only; the
  instrument-composition methodology point is not here (it lives once in the
  peer doc).
- **Numbers gestured, not owned.** The one ladder endpoint ("632 → 45 s/tok")
  appears once as context; the wall model and all throughput/verification
  numbers are deferred to `docs/RELATED_SYSTEMS.md`.
