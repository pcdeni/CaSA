# Running an LLM inside a DRAM chip — software side

A real, published 2.4-billion-parameter language model answers a
question, and the matrix-multiplications that produce the answer
happen *inside the DRAM cells* of a regular DDR4 memory module — via
charge-sharing, on real silicon.

> **What this is (and isn't).** This is *processing-using-DRAM* (PUD):
> we compute with the same physical charge-sharing effects that, aimed
> the other way, underlie disturbance attacks like RowHammer. The
> primitives look identical to attack tooling because the physics is
> identical — the difference is purpose. Everything here is authorized
> research on our own hardware: driving deliberate charge-sharing to do
> ternary matrix-multiplies, and independently reproducing a *published*
> academic paper (arXiv:2503.23817) on parts we bought. No system is
> attacked; nothing bypasses anyone's security. If you scan this repo
> for exploit patterns you will find DRAM-disturbance code — that is the
> computer, not the weapon.

> ▶ **Interactive walkthrough of how it works:**
> **<https://pcdeni.github.io/CaSA/explainer/>** — 13 stepped scenes
> from cell to inference loop (now including the in-DRAM accumulation
> "merge" and the full 632→47.5 s/tok arc), every claim sourced to a
> paper or the code in this repo. Reviewed adversarially before
> publication; the claim-to-source ledger lives next to the page in
> [`docs/explainer/`](docs/explainer/).
>
> ▶ **Plain-language tour:** **<https://pcdeni.github.io/CaSA/explainer/system.html>**
> — "what we built and how it works" end to end for a reader new to the
> project: what charge-sharing physically does, how a ternary matmul
> becomes row operations, why refresh restores charge but not content,
> and why a bit-exact unit test can still let a full model drift.
>
> ▶ **Companion:** **<https://pcdeni.github.io/CaSA/explainer/xor-spread.html>**
> — the `doubleACT` row-spread we found during calibration: a bit-exact
> copy deposited into address-XOR sibling rows, the MAJ3 self-pollution
> it causes, and how we engineer around it (or exploit it). *Update, July
> 2026:* a SiMRA co-author confirmed the mechanism (the source-side view of
> Multi-RowCopy's co-activation lattice) and assessed our address-algebra
> as new — see [DRAM-Bender#12](https://github.com/CMU-SAFARI/DRAM-Bender/issues/12)
> and [our verification + open questions](https://github.com/CMU-SAFARI/SiMRA-DRAM/issues/1),
> which now also bound Multi-RowCopy's own reliability envelope.
>
> ▶ **[MVDRAM reproduction study — updated](docs/MVDRAM_REPRODUCTION.md)**
> — we bought two new units of the exact DRAM part the MVDRAM paper
> (arXiv:2503.23817) names and attempted a full reproduction. The part
> performs **no processing-using-DRAM at all** in our hands (0 charge-share
> copies in 60,000 random row pairs, while working perfectly as ordinary
> memory) — that finding stands. *Update, July 2026:* our second finding —
> "on modules where PUD does work, the paper's chained dataflow collapses
> to 6–11%" — is **reversed**: the collapse was an addressing artifact of
> our own row placement. With pair-offset-safe placement derived from the
> co-activation lattice, the paper's computation-rows dataflow runs at
> 99.98% end-to-end and a fused fast-path kernel beats our host-mediated
> variant by 2.2–2.3× (study §8). MVDRAM's method is achievable on
> commodity spread-afflicted silicon — but only with spread-aware
> addressing the paper does not discuss. Reproducer code and raw logs
> included.
>
> ▶ **Companion:** **<https://pcdeni.github.io/CaSA/explainer/mvdram.html>**
> — the reproduction study as an interactive 11-scene deck: what the paper
> claims (and what its baselines are), Result A, the June→July Result B
> reversal, the mechanism scoreboard, the four-model sampled end-to-end
> protocol (36 silicon-verified ops), the first bit-exact fp32 the
> reproduction produced, and what remains open — every scene tied to the
> paper (§/Fig/Table), a reproducer, a data file, or an issue (claim
> ledger in [`docs/explainer/`](docs/explainer/)).

This repository contains the software side of that demonstration:

- **`scheduler/`** — `casa_sched.c`, a discrete-event scheduler that
  models a DDR4 channel running charge-sharing PIM primitives, with
  bus contention tracked explicitly. Used to project per-token
  throughput under different optimization stacks.
- **`app/`** — C++ apps that issue charge-sharing primitives
  (RowClone, MAJ3 via `doubleACT`, multi-row broadcast) on real
  DRAM-Bender silicon and run the matmuls of Microsoft's
  BitNet b1.58-2B-4T using them. Drop these into a DRAM-Bender
  checkout and `make`.
- **`python/`** — orchestrator that patches Hugging Face's
  `transformers` library to route specific projection layers to the
  FPGA-side server while the rest of the model runs on the CPU.
  Supports BitNet's per-tensor-scaled ternary weights and, since July
  2026, group-scaled (g128) external weight specs — PrismML
  Bonsai-1.7B in both its 1-bit and ternary variants.
- **`lane2/`** — the MVDRAM-reproduction GeMV server and drivers (their
  conventions end to end: LOAD/GEMV/PARTIALS protocol, dual-track adder,
  clone-encoded products, per-block exact partials, the B2 per-op table
  harness, the sampled llama.cpp e2e runners, and the Road-B
  `LANE2_ACCUM` product-dataflow arms).
- **`rtl/`** — the DIFF-accum readback engine (Road B) with its
  Verilator harness and the three silicon-found fixes, including the
  buffer_space leak that also lives in stock DRAM-Bender streaming DIFF.
- **`api-patches/`** / **`shim-patches/`** — unified diffs for the
  SiMRA/DRAM-Bender API (finalize-once, recv-timeout, the accum
  receiver) and the llama.cpp mulmat interception shim.
- **`calibration/`** — calibrated MAJ3-perfect open-row tuples for one
  of our test DIMMs. Format documented; you produce your own for new
  DIMMs.
- **`docs/`** — hardware requirements, calibration protocol,
  scheduler-projection methodology, and the
  [interactive explainer](https://pcdeni.github.io/CaSA/explainer/)
  (source in [`docs/explainer/`](docs/explainer/)).

### How the pieces connect

One arc runs through everything here, and each stage feeds the next:

1. **Characterize the physics** — calibrated MAJ3/RowClone tuples, and
   the two laws the literature didn't have: the co-activation
   **selection law** (which rows receive a `doubleACT` deposit — the
   address algebra of Multi-RowCopy's lattice) and the **clone-dead
   law** (which rows can be RowClone-refreshed). Both cross-die
   deterministic. → [`docs/LATTICE_ADDRESSING_2026_07.md`](docs/LATTICE_ADDRESSING_2026_07.md),
   [xor-spread explainer](https://pcdeni.github.io/CaSA/explainer/xor-spread.html).
2. **Turn the laws into engineering** — spread-safe pool placement,
   coset-broadcast loading, the fused activation update, per-bank
   parallelism. This is what took BitNet 632 → 47.5 s/tok with
   correctness *improving*. → [`docs/CAMPAIGN_2026_07.md`](docs/CAMPAIGN_2026_07.md).
3. **Reproduce the peer system** — MVDRAM (no public sources) rebuilt
   from the paper on commodity parts: mechanism scoreboard, honest
   negatives, per-op B2 tables, sampled llama.cpp e2e, first exact fp32.
   The reproduction machinery lives in [`lane2/`](lane2/). →
   [`docs/MVDRAM_REPRODUCTION.md`](docs/MVDRAM_REPRODUCTION.md),
   [`docs/PAPER_CONTRAST.md`](docs/PAPER_CONTRAST.md).
4. **Kill the readout wall both ways** — Road A (in-DRAM carry-save
   adders, MVDRAM-faithful, 213× readout reduction) and Road B (the
   FPGA popcount accumulator in [`rtl/`](rtl/), 8 KB → 96 B per read),
   never blended in one headline. Road B's completion re-opened the
   per-output product dataflow — the reproduction's own §V shape —
   at ~6–12× per GeMV. → [`docs/ROADB_2026_07.md`](docs/ROADB_2026_07.md).
5. **Generalize across models** — the same silicon path runs BitNet
   (per-tensor ternary) and PrismML Bonsai-1.7B (g128 group scales,
   1-bit and ternary), token-exact, with each model family exposing new
   levers (group-response protocol, 1-bit single-track). →
   [`docs/BONSAI_2026_07.md`](docs/BONSAI_2026_07.md).

Everything is measured on real silicon, every claim carries its log or
data file, and negatives are published next to positives — that is the
repo's standing offer to anyone building in-memory LLM inference.
(And the question everyone asks — *could it train, too?* — has a
measured, honest answer: [`docs/TRAINING.md`](docs/TRAINING.md). The
architect's question — *how far from optimal, what binds, what should
the die change?* — has one too: [`docs/UTILIZATION.md`](docs/UTILIZATION.md).
And what's next, with statuses and evidence, lives in
[`docs/ROADMAP.md`](docs/ROADMAP.md).)

This builds directly on prior research from the
[CMU SAFARI group](https://safari.ethz.ch/) — RowClone, Ambit,
SiMRA-DRAM, Multi-Row-Init, LISA, pLUTo — and the open-source
[DRAM-Bender](https://github.com/CMU-SAFARI/DRAM-Bender) FPGA platform.
We don't re-host either; you clone them yourself and place the C++
apps from `app/` into the right path. See `app/README.md`.

## Headline result — a measured arc, driven by understanding

**What we are running today** is BitNet b1.58-2B-4T with **all 30
transformer layers' projection matrix-multiplies executing in DRAM**
(7 projections per layer — q/k/v/o/gate/up/down — ternary weights
resident in the DIMM, MAJ3-based multiply-accumulate), producing the
model's correct output ("What is the capital of France?" → "Paris").
Attention softmax, norms, and sampling run in PyTorch on the CPU, as
in every PUD system.

Over one intensive campaign (July 2026), the **measured** full-model
per-token time on real silicon came down **13.3×** — and every step
was a thing understood, not a knob tuned. Correctness *improved* as
speed rose. These are measured numbers on this hardware, not scheduler
projections:

| Per-token (measured) | What changed | Output |
|---|---|---|
| **632 s** | May baseline — single DIMM, per-MAJ weight rewrite | correct |
| **360.8 s** | 8K-IMEM bitstream + fused coset activation update; four host-side bugs found and fixed (a stale-activation cache, a silent pool-path fallback, a scratch/weight collision, mis-scoped voting) | correct |
| **137.1 s** | **clone-dead law** — ~1/3 of the "fault-free" pool rows cannot be RowClone-refreshed (a *systematic*, closed-form-predictable defect the May screen selected *for*); screening it out let voting be switched off entirely | correct |
| **80.5 s** | steady-state marginal rate (the 137 figure carried per-run load overhead), confirmed stable over 48 tokens | correct |
| **47.5 s** | dual-DIMM (2 dies, work split + host-summed) after fixing a fallback-routing bug that had starved the second die — **1.91× per token-matmul, 96 % of the ideal-halving bound** | correct |

The story in three sentences:

1. **The gap was never physics — it was understanding.** The four bugs,
   the clone-dead law, the drift characterization, the voting economics
   (it *looked* load-bearing; it was masking dead rows), and the
   dual-DIMM balance fix are all in the campaign record
   ([`docs/CAMPAIGN_2026_07.md`](docs/CAMPAIGN_2026_07.md), 27 addenda).

2. **The measured rate is now recv-volume-bound, not compute-bound.**
   Profiling the 47.5 s/tok run shows the DDR-to-host readback of result
   rows dominates each request. The first of the two levers targeting
   that is now **complete on silicon**: the on-FPGA popcount accumulator
   (Road B, [`rtl/`](rtl/) + [`docs/ROADB_2026_07.md`](docs/ROADB_2026_07.md))
   collapses each result-row read 8 KB → 96 B, survives 65,000-program
   sessions with zero stream-integrity faults, and — the surprise — it
   *re-opened MVDRAM's own per-output product dataflow*, which beats our
   carry-save tree ~6–12× per GeMV on the same silicon. Full weight
   residency (removing per-request streaming) is the remaining lever.

3. **The cycle-level scheduler `casa_sched.c`** still projects the
   *bus-bound* floor beneath all of this — what remains once orchestration
   is fully engineered out and only the DDR bus is left. It respects every
   standard DDR4 timing parameter and tracks bus/bank utilization
   explicitly; use it to see where the measured arc is heading and why the
   remaining levers (Road B, residency, streaming) matter.

Output is **bit-exact correct on most cells** (~99.9 % per projection);
the stray flips come from cells that pass the calibrated stability test
but flip on uncalibrated bit-combinations, and ternary models are robust
to this by construction (the full model answers correctly and stably).
See [`docs/CAMPAIGN_2026_07.md`](docs/CAMPAIGN_2026_07.md) and
`docs/METHODOLOGY.md`.

The point of the work is not to beat a GPU on speed. It is to
**demonstrate the mechanism** on real silicon, drive it down honestly
with reproducible measurements, and — in the companion study — to
**independently reproduce a published PUD paper** end to end. Two things
this campaign produced that the literature did not have: a complete
*selection law* for which rows co-activate under `doubleACT`, and the
*clone-dead law* above — both cross-validated across dies and subarrays.

### A second model family: PrismML Bonsai-1.7B (2026-07-20)

The pipeline is no longer single-model. **PrismML Bonsai-1.7B** — 1-bit
{−1,+1} and ternary quantizations of Qwen3-1.7B with **per-128-input
group scales** (the g128 format family every llama.cpp-style quant
uses) — runs end-to-end on the same silicon with the **server binary
unchanged**: all 196 projection matmuls in DRAM, and all first-run
outputs **token-exact against the fp32 CPU golden** (8/8 in each of
three runs; the 1-bit variant's raw-mode echo and chat-mode correct
answer are both faithfully reproduced — a *capability difference between
two quantizations of the same base model, directly visible on DRAM*).
The client gained a default-off group-scale weight path, regression-proven
byte-identical for BitNet with the feature off. The next-day measured
ladder on the same shape — single-DIMM 100 s/tok → dual-DIMM 51.2 →
+ fused coset 33.7 → + the 1-bit **single-track protocol** (the server
computes only the pos track; the complement is reconstructed
arithmetically) **18.7 s/tok — 5.36× stacked**, every configuration
golden-exact, and the 1-bit variant now *beating* ternary on the wall —
plus the sim-vs-silicon fused A/B story and honest caveats:
[`docs/BONSAI_2026_07.md`](docs/BONSAI_2026_07.md).

## Related work — where this sits

- **[MVDRAM](https://arxiv.org/abs/2503.23817)** (Kubo et al., 2025) is the
  closest peer: GeMV for low-bit LLMs in unmodified DDR4 on the same
  DRAM-Bender testbed family, weights resident in DRAM, **measured faster
  than a CPU** (2.18× end-to-end for 2-bit Llama2-13B). It optimizes
  throughput via selective-RowClone partial products and in-DRAM MAJ-adder
  accumulation; its error model is per-column screening on a module chosen
  from 16 candidates, and the paper reports no output-accuracy evaluation.
  CaSA occupies the complementary lane: a native-ternary production model
  end-to-end with **whole-row bit-exact verification** on unscreened
  silicon — a criterion under which logical MAJ5 (which MVDRAM's in-DRAM
  adders use) yields zero perfect configurations on our modules while
  16-row-replicated MAJ3 yields hundreds — plus the failure-mechanism
  characterization (XOR row-spread, MAJ self-pollution) that column-static
  profiling cannot see. Full mechanics comparison:
  [`docs/MVDRAM_COMPARISON.md`](docs/MVDRAM_COMPARISON.md).
- **[SiTe CiM](https://arxiv.org/abs/2408.13617)** (Thakuria et al., Purdue,
  2024) is the custom-silicon end of the same goal: signed-ternary
  compute-in-memory via modified bit cells (8T-SRAM / 3T-eDRAM / FEMFET,
  simulation only). Useful as the measure of what a redesigned cell buys;
  our angle is the opposite — exploit the cell exactly as manufactured.
- **[PARBOR](https://users.ece.cmu.edu/~omutlu/pub/parbor-efficient-system-level-test-for-DRAM-failures_dsn16.pdf)**
  (Khan, Lee, Mutlu, DSN 2016) characterizes chip-specific *bitline*-side
  neighbor coupling at JEDEC timing; our XOR-spread work is the sibling
  result on the *row-decoder* side under PuD timing, which JEDEC-timing
  tests cannot reach.
- **[hifidram-ocsa-spice](https://github.com/pcdeni/hifidram-ocsa-spice)**
  (this author) is the *sense-amplifier-level* companion to this repo: an
  LTSpice study, built on the topologies reverse-engineered in HiFi-DRAM
  (ISCA 2024), of why charge-sharing PUD survives on some DIMMs and not
  others. It shows charge sharing is identical across the classic and the
  offset-cancellation (OCSA) sense amps, but the OCSA's boosted reference
  slides the majority decision threshold off the tie point — breaking MAJ
  out of the box and recovering it only under a Frac/precharge calibration.
  A circuit-level account of the reliability variation we hit empirically,
  and a prediction that COTS charge-sharing degrades as vendors migrate to
  OCSA (2 of 3 majors already have).
- The enabling canon — RowClone, Ambit, SiMRA, FracDRAM, FCDRAM, POPCNT3,
  DRAM-Bender — is credited in [Acknowledgments](#acknowledgments) and
  cited throughout the [explainer](https://pcdeni.github.io/CaSA/explainer/).

## Quick start (assuming you already have DRAM-Bender silicon)

```bash
# 1. Clone DRAM-Bender (the FPGA controller + bitstream)
git clone https://github.com/CMU-SAFARI/DRAM-Bender
# … bring up the BCU1525 bitstream per DRAM-Bender's docs.

# 2. Drop our C++ apps into DRAM-Bender's apps tree and build.
cp app/*.cpp app/Makefile DRAM-Bender/sources/apps/DSN_AE_APPS/BitNet/
cp calibration/calib_dimm0.txt   DRAM-Bender/sources/apps/DSN_AE_APPS/BitNet/
cd DRAM-Bender/sources/apps/DSN_AE_APPS/BitNet
make

# 3. Calibrate a DIMM (only needed once per chip — see docs/CALIBRATION.md).
#    The shipped calib_dimm0.txt is for one of our test DIMMs;
#    your hardware may need its own characterization.

# 4. Run an end-to-end smoke test:
./bitnet-real-exe 0 calib_dimm0.txt 1
# Expected: bit-exact match on a small ternary x int8 matrix multiply.

# 5. Hook the long-running PIM server into BitNet inference.
#    These two paths point the Python orchestrator at the binaries
#    you just built. (The python script reads them via env or CLI.)
export PIM_RUNNER=$PWD/bitnet-proj-exe
export PIM_SERVER=$PWD/bitnet-proj-server
export BITNET_CACHE=~/bitnet_weights         # any HF cache dir
cd <repo>/python
pip install transformers==4.52 torch
python3 run_bitnet_pim.py \
    --max-tokens 8 --projs all --bank "0,1,2,3" \
    --prompt "What is the capital of Hungary? Answer in one sentence."
# Expected output ends with "Budapest" after ~4 minutes.
```

See `python/README.md` for argument details and how the
`pim_substitute` swap works internally.

## Hardware requirements (summary)

- A Xilinx Alveo U200 / BCU1525 (or compatible) FPGA card flashed
  with the DRAM-Bender bitstream.
- One or more DDR4 1333 MT/s DIMMs in the FPGA's DIMM slots
  (you'll need to characterize them).
- A host with PCIe-attached FPGA, the Xilinx XDMA driver loaded,
  and the SoftMC API available.

Full details in `docs/HARDWARE.md`.

## Honest caveats

- **Per-cell yield**: a small fraction of cells (we measured
  ~5/22 144 in one full layer = 0.02 %) flip on input bit-patterns
  the calibration didn't exhaust. BitNet is robust enough to absorb
  this, and most outputs land bit-exact. The per-bank yield is
  run-to-run nondeterministic — `docs/METHODOLOGY.md` discusses.
- **Multi-bank divergence**: parallelizing across multiple banks
  uses multiple calibrated tuples, each with its own flaky-cell
  pattern. Output stays sensible but is not bit-exact across runs.
  For deterministic demos, pin to one bank.
- **Multi-DIMM scaling** is integrated for 2 of 4 DIMMs
  (`run_bitnet_pim.py --dimms "0,2"` splits each projection's input
  dimension across two concurrently-driven DIMMs; measured 1.47× on a
  1-layer A/B, short of 2× due to uneven sub-handle balance). The other
  two DIMMs were characterized and **rejected on silicon grounds** —
  one is cell-uniformity-limited, one collapses at the MAJ tie boundary
  exactly as the self-pollution mechanism predicts; see the
  [XOR-spread explainer](https://pcdeni.github.io/CaSA/explainer/xor-spread.html).
  The 4-DIMM scheduler projections assume four DIMM-0-class chips.
- **The simulator was written before the silicon implementation**.
  Its hardcoded charge-sharing latencies were patched against
  measured DIMM 0 values; numbers shift by <2% because the
  projections are bus-bound, not MAJ3-bound. See
  `docs/METHODOLOGY.md`.

## Acknowledgments

- **CMU SAFARI Group** (Onur Mutlu et al.) — RowClone, Ambit,
  SiMRA-DRAM, Multi-Row-Init, LISA, pLUTo. Without their decade of
  characterization papers and open-source toolkits, none of this is
  possible on existing silicon.
- **Microsoft Research** — BitNet b1.58-2B-4T, an open-weight
  2.4-billion-parameter ternary language model.
- **Prism ML, Inc.** — Bonsai-1.7B, open-weight 1-bit and ternary
  group-scaled quantizations of Alibaba's Qwen3-1.7B (Apache-2.0); the
  second model family this pipeline runs (see
  [`docs/BONSAI_2026_07.md`](docs/BONSAI_2026_07.md)).
- **Hugging Face** — `transformers` library (we test against
  v4.52).
- The communities behind `Manim`, `Piper TTS`, `matplotlib`, and
  `ffmpeg` for the video-production tooling used in the
  presentation (sources for those live in the private prototype
  repository, not here).

## License

MIT — see [LICENSE](LICENSE). Upstream components remain under their
own licenses (DRAM-Bender, SiMRA-DRAM, BitNet, transformers, …); we
don't ship them.
