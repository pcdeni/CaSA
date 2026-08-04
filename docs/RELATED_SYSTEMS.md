# Related systems and methodology

The peer-facing companion to the two explainers. It places CaSA next to the
published systems that share its substrate, maps CaSA and its closest peer
(MVDRAM) onto each other mechanism by mechanism, states how every number in
the project is verified, and gives the model of the performance wall that the
explainers only gesture at.

Two conventions hold throughout, matching the rest of the public docs:

- **Every MVDRAM number is from the paper** ([arXiv:2503.23817](https://arxiv.org/abs/2503.23817),
  v2, 23 Sep 2025), cited by section / figure / table.
- **Every CaSA number is measured on our silicon** and cited to a log or data
  file in this repo, or is a **scheduler projection** with its assumptions
  stated. The two are never mixed in a single claim.

A term used below once and then reused: a **double activation** is the one
command pair this whole family of work is built on — activate a row, hold the
array open, force it closed, activate a second row
(`ACT … first hold … PRE … gap … ACT`). The mechanism and its two timing
regimes (a clean majority **vote** versus a multi-row **copy**) are the subject
of the [mechanism explainer](explainer/xor-spread.html); this document treats
them as known and links there rather than re-deriving them. Two more terms
recur below: a **calibrated tuple** is the small, fixed set of rows co-activated
for one vote, and **MAJ-N** is the majority over N such co-activated rows (MAJ3
over three, MAJ5 over five, and so on).

---

## 1. The neighborhood

CaSA computes low-bit LLM matrix-multiplies inside unmodified commodity DDR4 by
driving the row decoder out of spec. Several published systems share pieces of
that idea; naming where CaSA sits keeps its contribution honest.

**MVDRAM** — the closest published peer, treated in full in §2. Matrix-vector
multiply for low-bit LLM inference in **unmodified commodity DDR4**, on the
**same testbed family** (DRAM-Bender on a Xilinx FPGA), weights resident in the
array, measured faster than a CPU on the same modules. Anyone evaluating CaSA
should read it.

**The SAFARI primitive lineage.** CaSA is an application built on primitives
that prior work established and named; it does not claim them. In order of how
directly they feed the stack:

| Primitive | What it gives CaSA | Reference |
|---|---|---|
| **SiMRA** — Simultaneous Many-Row Activation, MAJX, **Multi-RowCopy** | the double-activation itself, the MAJ-of-K majority, and the multi-row copy that CaSA's coset deposit *is* a special case of | Yüksel et al., arXiv:2405.06081, 2024 |
| **FracDRAM** — fractional (V_DD/2) charge in off-the-shelf cells | the neutral-row / `Frac` conditioning used to widen MAJ margins | Gao, Tziantzioulis, Wentzlaff, MICRO 2022 |
| **FCDRAM** — functionally-complete Boolean logic in real DRAM | the NOT / functional-completeness context for in-array logic | Yüksel et al., arXiv:2402.18736, 2024 |
| **POPCNT3** — bulk bitwise accumulation in commercial DRAM | the in-DRAM popcount that motivates CaSA's readout-wall work (§5); by Kubo, the MVDRAM first author | Kubo et al., "Bulk Bitwise Accumulation in Commercial DRAM," MLNCP @ NeurIPS 2024 |
| **RowClone / Ambit** | the in-DRAM copy and bulk-bitwise ancestors of every PUD (processing-using-DRAM) system here | Seshadri et al., MICRO 2013 / MICRO 2017 |
| **DRAM-Bender** | the FPGA testbed both MVDRAM and CaSA run on | Olgun et al., 2023 |

**ParBoR** — a sibling result from the reliability-testing side rather than the
compute side. It observes that a large class of DRAM failures is
**data-dependent**: a cell fails only when specific patterns sit in
**physically neighboring** cells, and it stresses that DRAM vendors **scramble
the address space internally**, so cells adjacent in the physical array are not
adjacent in the system address space. CaSA's coset / selection-law result is
the row-decoder-side analog of the same underlying fact — that address
structure invisible at the system level governs which cells interact. ParBoR
couples through cell/bitline proximity; CaSA's coupling is through the
predecoder wiring. (Khan, Lee, Mutlu et al., "PARBOR," DSN 2016.)

---

## 2. MVDRAM ↔ CaSA — a mechanics comparison

MVDRAM and CaSA sit on the same substrate and prove different legs of the same
thesis: MVDRAM proves the **throughput** leg on screened silicon; CaSA proves
the **correctness and characterization** leg on unscreened silicon. They
disagree on almost no facts and overlap on almost no contributions. This
section is the map.

> A full hardware reproduction of MVDRAM — including two new units of the exact
> DRAM part the paper names (SK Hynix HMA851U6CJR6N-UHN0) — is documented
> separately in **[MVDRAM_REPRODUCTION.md](MVDRAM_REPRODUCTION.md)** (and the
> [reproduction study deck](explainer/mvdram.html)). Two results from it frame
> everything below: the **named part performs no PUD in our hands** (0 of 60,000
> random pairs on two units), and MVDRAM's **chained MAJ5 adder does not reach
> bit-exactness on any PUD-capable module we own**. The comparison here takes the
> paper's claims at face value; the reproduction study carries the silicon
> verdicts.

### 2.1 TL;DR

| | MVDRAM (their paper) | CaSA (this repo, measured) |
|---|---|---|
| Goal | throughput-competitive GeMV (general matrix-vector multiply) vs CPU/GPU | verifiable end-to-end correctness of production ternary LLMs + failure-mechanism characterization |
| Platform | DRAM-Bender, Alveo U200, 4× SK Hynix DDR4-2400 | DRAM-Bender, BCU1525 quad, DDR4 |
| Partial products | **selective RowClone** — the activation bit picks the copy source; no MAJ | activation broadcast + MAJ3-as-AND on calibrated tuples |
| Accumulation | in-DRAM MAJ full adders (carry = MAJ3, **sum = MAJ5**, dual-rail) | readback + host / FPGA popcount |
| Error handling | per-column screening + Frac margin; reliable columns only (83–95 %/module) | per-tuple calibration, 16-row replication, spread-aware pool layouts, mechanism analysis |
| Headline | up to 7.29× GeMV vs CPU; 2.18× end-to-end (2-bit Llama2-13B, measured) | seven flagship LLM families token-/numerics-exact on unmodified DDR4; correct answers out of DRAM |
| Accuracy evaluation | **none reported** (latency / energy only) | bit-exact per-projection verification vs PyTorch reference |
| Correctness criterion | screened **column subset** (part does MAJX up to MAJ15) | **whole-row bit-exact**, under which logical MAJ5 yields zero perfect configs on our modules while 16-row-replicated MAJ3 yields hundreds |

The two systems demand different things of the silicon, so neither hot path
transplants directly. That asymmetry — correctness criterion, not raw
multi-row-activation capability — is §2.6.

### 2.2 MVDRAM's mechanisms, precisely (from the paper)

**On-the-fly vector encoding (§V).** The activation vector is known to the
*host* when it issues DRAM commands, so MVDRAM never stores activations in DRAM
and never computes `w AND a` in DRAM. The partial product is encoded into the
**choice of RowClone source**: activation bit `a = 1` clones the matrix row into
the compute row (result = w); `a = 0` clones a constant-zero row (result = 0),
or skips the operation entirely (their sparsity path, §V-D, assumes 50 % input
bit-sparsity). Two consequences: matrix rows are only ever ACT sources, so a
destructive-read-with-restore rewrites them for free and weights are written
once and never rewritten; and the product step needs **zero** multi-row
activations — the fragile operations are pushed entirely into accumulation.

**Horizontal matrix layout (§VI).** Rather than storing operands vertically
(one column per output, wasting the ~65,536-column parallelism and forcing a
transpose), MVDRAM bit-decomposes the MAC by matrix bits so each partial sum is
an independent binary inner product. The matrix stays row-major: one row per
input index, and a single selective RowClone feeds `q·M` partial products at
once (`q` = weight bits, `M` = outputs). Outputs land in rows the host reads
row-wise and combines with shifts — no transpose. They cap N ≤ 128 per subarray
and shard across subarrays and four modules.

**In-DRAM MAJ accumulation, dual-rail (§II-C, §VII).** The reduction happens in
the subarray with bit-serial full adders built from majority: carry
`s1 = MAJ3(x0,x1,x2)`, sum `s0 = MAJ5(x0,x1,x2,¬s1,¬s1)`. Unmodified DRAM has
no native NOT, so every value is kept with its complement throughout (the matrix
is stored twice — original + inverted — which dominates their capacity
overhead). Readback per GeMV is then only the `q × r` output rows.

**Reliability (§VII, Table I).** Per-column profiling plus FracDRAM-style Frac
operations to widen margins; the GeMV uses only consecutive runs of `q`
**reliable columns** — 54,365–61,727 of 65,536 (82.9–95.1 %) across their four
modules. The part (HMA851U6CJR6N-UHN0) was selected by characterizing **16 SK
Hynix models** and picking the most cooperative — footnote 3: "the most reliable
one that supports both strict RowCopy and MAJX operations (up to MAJ15)."

**Measured results (Table II, Figs 12–17).** Baselines: Intel i7-9700K with the
*same* DDR4-2400 modules (77 GB/s), llama.cpp/ggml low-bit kernels; GPU = Jetson
Orin Nano (LPDDR5, an edge part). GeMV 32000×4096, 1-bit vector × 2-bit matrix:
CPU 1.44 ms, GPU 1.70 ms, MVDRAM 0.19 ms (0.14 in-DRAM + 0.05 aggregation) →
7.29× / 8.55×. End-to-end (llama.cpp with `mulmat` routed to DRAM, four models —
Llama2-7B, Llama2-13B, Llama3-8B, Phi-4 — 256 tokens × 10 runs): 2-bit
Llama2-13B 2.18× vs CPU, 4-bit 1.31×. Energy 30.5× (GeMV) / 3.04× (end-to-end
2-bit) — **MVDRAM's own power is CACTI-modeled, not wall-measured**; baselines
are measured (RAPL / tegrastats). The paper reports **no accuracy or perplexity
numbers** for the PUD-executed models; correctness rests on the reliable-column
screening.

### 2.3 The central object: mechanism-by-mechanism scoreboard

This is the anchor of the comparison — MVDRAM's each mechanism, our
reproduction of it, and the source. It is the one table to read if you read
only one thing here.

| Mechanism (paper element) | MVDRAM's number | CaSA reproduction | CaSA source |
|---|---|---|---|
| Reliable-column fraction | 83–95 % (Table I) | 87–88 % MAJ5-reliable (single-op) | mvdram-maj5-exe; MVDRAM_REPRODUCTION §4 |
| RowClone | "strict" on screened part | 8192/8192 deterministic on commodity parts | rowclone-smoke |
| Dual-track adder | error-free after calibration | 99.94 % (MAJ5 sum on screened columns); **99.98 % in the all-MAJ3 variant**; a server mode that builds the needed complements in-DRAM reaches 99.49–99.83 % integer-exact on a 4096×4096 GeMV, at ~2× the wall of forming those complements on the host | mvdram-adder-exe, mvdram-fulladder-exe, LANE2_DUALTRACK |
| On-the-fly encoding (§V-C) | RowClone source picked by activation bit | reproduced **physically**: products created by clones from resident weight / inverted-weight rows, **3.0× faster** than host write-loading on a 4096×4096 GeMV (12 s vs 38 s), 94.7 % integer-exact (single-pass, no replication vote) on a commodity DIMM | lane2 clone mode; test_mvdram_fastpath_ab.cpp |
| Bit-sparsity skip (§V-D) | skip op when a = 0 | host-side analog: **4.37× command-stream reduction** measured | MVDRAM_REPRODUCTION; casa_sched notes |
| Horizontal layout + row-wise aggregation (§VI) | bitline = output, row-major readout | reproduced | mvdram-gemv*-exe |
| Partial sums retrieved + host-aggregated (§II-C2, §VII) | N ≤ 128-per-subarray partials | exact per-32-block integer partials → host applies per-block scales → **first exact FP32** from the reproduction (bit-exact vs CPU fp32 on real Llama-2-7B blk.0.attn_q) | lane2 GEMV_PARTIALS; lane2_partials_fp32.py |
| Faithful computation-rows dataflow (their Fig 2) | their normal operation | **99.98 % end-to-end with spread-safe placement** | test_mvdram_compute_rows_safe.cpp |
| Fast in-DRAM operand movement (their Fig 3 profile) | assumed | 2.2–2.3× per gate over host write-loading; **3.0× per GeMV** in the fused-clone server mode | test_mvdram_fastpath_ab.cpp; lane2 clone mode |

**Our Table-I analog (reliable columns, per subarray).** Their Table I is
per-module; ours is per-**subarray**, MAJ5-op-matched, counted in 32-bit
segments (2048 per row = 65,536 bits); "robust" applies a stricter repeat
criterion:

| die | subarray | criterion | reliable segs | fraction |
|---|---|---|---|---|
| DIMM 0 | s61 | standard | 882/2048 | 43.1 % |
| DIMM 0 | s77 | robust | 1806/2048 | **88.2 %** |
| DIMM 2 | s72 | standard | 1188/2048 | 58.0 % |
| DIMM 2 | s72 | robust | 62/2048 | **3.0 %** |
| DIMM 2 | s86 | robust | 1784/2048 | **87.1 %** |

Source: mvdram-repro/colmask_*.txt (mvdram-maj5-exe). The best subarrays land
inside their Table-I range, and the same module spans 3 %→87 % across subarrays
under the robust criterion — **subarray selection is load-bearing on commodity
parts**, the intra-module analog of their 16-module screening. (The granularity
differs from their table; that is stated when either number is published.)

### 2.4 Where our silicon differs from theirs — real divergences, not gaps

- **The exact part number does no PUD in our hands** (2 new units, 0/60,000
  random pairs). Their footnote 3 screened 16 modules — consistent with severe
  inter-module variance within one part number.
- **Chained MAJ5** on our commodity modules: only ~1.37 % of columns survive the
  *chained* MAJ5 adder, even though *single-op* MAJ5 column reliability (87–88 %)
  matches their Table I. Their "error-free" rests on Frac + calibration, which we
  implemented and measured to its limit (§3). Our all-MAJ3 adder is the
  workaround that needs no MAJ5.
- **The coset / co-activation spread governs everything on our dies** (selection
  law; MVDRAM_REPRODUCTION addendum 5). The paper never mentions it — either
  their screened part is low-spread, or screening + Frac masked it. Details in
  the [mechanism explainer](explainer/xor-spread.html); its consequence for the
  error model is §2.5.

### 2.5 The error model — corrected

MVDRAM's error model is **column-static**: a column is reliable or it is not,
independent of operands. We measure a second, deterministic error channel that
this model cannot express, and it is worth stating precisely.

**The correct statement.** The double activation has two timing regimes (the
[timing dial](explainer/xor-spread.html)). At the majority-vote operating point
it resolves a **clean vote**; a few command slots more of first hold and the
same pair becomes a **multi-row copy** that deposits the first-activated row's
data into the coset of the pair's address difference. On every module we own —
two SK Hynix dies and one Micron die — **the clean vote at the operating point
does not deposit**; a well-timed vote is correct, and the deposit is not part of
it. Operand corruption enters only through **copy-timing operations** (the loads
and preparation that *use* the copy regime deliberately) or through **drifted
timing**. When a deposit lands on rows that are themselves an operation's
operands, a subsequent vote is taken over a *substituted* operand set and
returns a deterministically wrong result — and the wrong result is exactly the
majority of the rewritten operands, flipping at the arithmetic vote-count
boundary, not a glitch. So the error depends on **row geometry, operand
placement, and timing** — never on the column alone.

**Why it matters for an MVDRAM-class system.** A system doing millions of
RowClones with weight rows as sources (exactly MVDRAM's §V) on a module with a
coupling fingerprint like ours would deposit copies of matrix-row data into
sibling rows in a pattern **no per-column profile predicts**. Their chosen part
may be mild in this respect — unknown; ours measurably is not. The fix is
**spread-aware row placement** (independent-set / stride pool layouts in
`calibration/`), which composes with their layout and neutralizes the deposit by
placement rather than by timing. The same coupling, placed deliberately, is a
free 1-to-M broadcast — the asset side, treated in the
[mechanism explainer](explainer/xor-spread.html) and in the
[roadmap](ROADMAP.md).

*(Selection is digital, the firing is analog: which coset members are candidates
is byte-exact and deterministic across power cycles, banks, and rigs; whether a
deposit fires is set by timing. Both facts are measured — see the mechanism
explainer's selection-law and timing-dial scenes.)*

### 2.6 The real asymmetry — correctness requirement, not row count

Easy to misread, so precisely: CaSA co-activates **16 rows in every production
MAJ3** (and has characterized 4/8/16/32-row simultaneous activation) — more rows
at once than MVDRAM's MAJ5 ever opens. The difference is not how many rows the
silicon activates; it is what each system demands of the result.

- MVDRAM runs logical MAJ5 with **one copy per operand** and needs correctness
  only on the **screened column subset** (their part supports MAJX up to 15).
- CaSA requires **bit-exact results across the full row**. Under that criterion,
  logical MAJ3 with 16-row replication yields **hundreds** of whole-row-perfect
  tuples on both production modules (312 calibrated on DIMM 0,
  `calibration/calib_dimm0.txt`; 384 on DIMM 2, `calibration/calib_dimm2.txt`),
  while logical MAJ5 yields **zero** (0 whole-row-perfect tuples across 6,400
  timing/tuple configurations, `mvdram-repro` MAJ5-viability sweep) — its best
  configurations reach 99.98–99.99 % per-cell stability, almost but never
  exactly whole-row-perfect.

Two corollaries. First, MVDRAM's adder tree would plausibly run on our modules
**under MVDRAM's own error model** — a 99.98 %-stable MAJ5 is exactly the regime
per-column screening exists for; untested, but not excluded by our data. Second,
the converse stands: a bit-exact pipeline cannot use MAJ5 adders on this
silicon, so our reduction stays outside the array (host today, FPGA accumulator
next). The "chip lottery" is real on both sides — their screening (16 module
models → 1 winner) versus our characterization of whatever is in the slot — but
it is a lottery over *margins under your correctness criterion*, not over raw
multi-row-activation capability.

*(These MAJX ratings come from the standard multi-timing MajOps methodology;
§6 explains why a per-timing, spread-aware re-screen could move the
zero-perfect-MAJ5 verdict. It has not been re-run.)*

---

## 3. What each system should borrow from the other

**CaSA ← MVDRAM.**

1. **Selective-RowClone partial products** (their §V, adapted to ternary).
   BitNet's ternary weights are exactly their `q = 2` case: positive and negative
   masks are two matrix bit-planes, and `y = Σ_j x_j·pos_j − Σ_j x_j·neg_j`. Per
   activation bit-plane, RowClone weight-row `j` into the accumulation region iff
   the activation bit is 1, else skip. This removes the activation broadcast, the
   MAJ3-as-AND, the calibrated-tuple constraint, and the per-MAJ3 weight reload
   from the product step — and on our silicon RowClone is the one primitive that
   works everywhere. The reduction must then happen FPGA-side (their MAJ5 adder
   has no whole-row-perfect config on our modules, §2.6), a sibling of our
   FPGA-side popcount accumulator.
2. **Frac conditioning — measured to its limit.** MVDRAM uses Frac to *increase
   the reliable-column count*. We implemented it: the reference-policy sweep found
   the single-op optimum (ZERO-init + 2 fracs = 93.5 % vs 89.1 % strict MAJ5
   columns on s86), but the chained dual-track A/B then showed the gain **does not
   compose** (99.512 % int-exact under zero2 vs 99.902 % under the SiMRA frac'd-ONE
   convention at 4096², same-hour, per-policy screens). Conditioning is exhausted
   as a lever on this module class; the residual to their "error-free" stays
   attributed to module screening.
3. **Streaming command generation.** Their encoding-overlapped-with-execution
   model (§V-E: a single-threaded CPU generates commands faster than DDR4-2400's
   ~1.5 ns/command consumption) is the software twin of our sequence-engine HDL
   work; it confirms that program-per-execute, not DRAM physics, is the
   controller-path ceiling on this testbed family (§5).

**MVDRAM-class ← CaSA.**

1. **A deterministic error channel their methodology cannot see** — the
   copy-timing coset deposit (§2.5), plus **spread-aware placement** as the fix
   that composes with their layout.
2. **A verification methodology** — bit-exact per-projection comparison against a
   PyTorch reference, per-cell stability calibration, and end-to-end correctness
   on production models (§4); the evaluation dimension their paper leaves open.
3. **A ternary-native path for stricter correctness targets** — chips or
   requirements where only replicated MAJ3 survives a whole-row bit-exact
   criterion (§2.6).

A merged design — selective-RowClone products + spread-aware layout +
Frac-conditioned margins + FPGA-side accumulation + bit-exact verification — is
the obvious next system, and most of its parts already exist across the two
codebases.

---

## 4. Rigor: what is measured, what is projected, what we do not claim

**The rule.** Every number in the project is either **measured** (with a date,
a configuration, and a way to reproduce) or **scheduler-bounded** (with the
assumed configuration explicit). Numbers estimated without measurement have,
in this project, come out wrong by orders of magnitude — so a projection is
treated as a **ceiling of what the silicon can do, not a promise of what an
implementation will hit**. The projection instrument is
`scheduler/casa_sched.c`, a discrete-event simulator over standard JEDEC DDR4
timings, bank/bus contention, and the measured charge-sharing latencies;
patching the measured latencies in shifted its projections by < 2 %, because
they are bus-bound and the exact per-op cost barely moves the wall.

**What runs today, and the model split.** The production configuration runs all
of a model's BitLinear matrix-multiplies in DRAM — for BitNet, the seven
projections (Q, K, V, O, gate, up, down) of every layer, i.e. all 210 BitLinear
modules. What stays in PyTorch on the CPU is the irreducible non-BitLinear
remainder — attention softmax, layernorms, sampling, embeddings, the LM head —
the same CPU/DRAM split every processing-in-DRAM system makes. Seven flagship
LLM families are validated on unmodified DDR4, in two lanes with two verification
standards:

- **Native lane — token-exact, live on silicon:** BitNet-b1.58-2B, Bonsai-ternary,
  Bonsai-1-bit. These run token-for-token identical to their reference on the
  card (BONSAI_SILICON_2026_07_20; the BitNet full-model run answers "Paris").
- **Mainstream lane — numerics-exact via the sampled end-to-end protocol:**
  Llama2-7B, Llama2-13B, Llama3-8B, Phi-4. These are validated by exact per-block
  integer partials reconciled to an FP32 reference through the sampled-e2e
  coverage (q2_K / q3_K / q4_0 / q6_K quants, dims 4096–32000). Their exact
  wall-clock end-to-end through DRAM at
  MVDRAM's benchmark shape remains out of scope — it is the streaming-execution
  gap of §5, not a correctness gap.

**The verification discipline.** Correctness is checked at the projection level,
not the token level: each in-DRAM projection is compared **bit-for-bit** against
the PyTorch float reference. In one full BitNet layer's worth of PIM-side outputs
(22,144 values) we measure **22,139 bit-exact matches — 99.98 %**. The five
mismatches are marginal cells that pass the calibrated 1000-pattern stability
test but flip on specific bit-combinations the calibration did not exhaust. These
cells are **not a fixed set**: the same bank can produce slightly different output
token sequences across two runs of the same prompt, because marginal cells flip
differently run to run. This is honest nondeterminism, and it is survivable *for
this model class*: ternary networks are robust to small per-weight perturbations
by design — a floating-point model would not survive these flips, a ternary one
does. For deterministic demos, pin to a bank that passed a recent canary; for
throughput, accept slightly-different-but-sensible outputs across runs. (Per-op
raw MAJ3 is an analog ~78 %-yield process; correctness is asserted at the
**projection** level via replication and, where used, correlation gating — not
by asserting bit-exactness on a single raw MAJ.)

**What we explicitly do not claim.**

- **Not** faster than a GPU on the same model today. The ladder is 632 → 45
  s/token measured (STOCKTAKE PROGRAM OF RECORD); a GPU is far ahead at this
  writing. The DDR-PHY-bound floor (§5) is the target, not a current result.
- **Not** a custom DRAM chip. Everything runs on stock DDR4.
- **Not** training in DRAM. Training stays on GPUs.
- **Not** general-purpose compute-in-memory. It is a specific set of operations
  (bitwise AND + popcount) on a specific model class (low-bit / ternary LLMs).

The contribution is to demonstrate the **mechanism** on real silicon, verify it
to bit-exactness against a reference, and put scheduler-bounded numbers on where
it could go with specific memory-die changes (§5).

---

## 5. The wall model

This is the authoritative home for the performance-wall numbers the explainers
only gesture at. It is short on purpose.

**The utilization pyramid.** Measured on this rig, resource by resource:

| Level | Resource | Utilization today | Evidence |
|---|---|---|---|
| 0 | The op itself — one double-activation MAJ3 = a 65,536-bitline majority in tens of ns | ~at physics | charge-sharing dwell is real, calibrated per DIMM, not shortenable |
| 1 | DDR command bus during command issue | ~9.5 % slot util (stock interpreter lane) | stock fetch/decode/execute emits 1 op/cycle; ~9 of 10 bus slots idle (seq_engine reaches 100 % in Verilator, `rtl/SEQ_ENGINE.md`) |
| 2 | DDR bus across a whole program | **< 1 %** | of a measured 5.9 ms/program, DDR-active time is tens of µs; the rest is transfer + turnaround |
| 3 | Banks / DIMMs | 4 of 16 banks per die; 2 of 4 DIMMs | dual-DIMM slice-partition **1.79×** on the token wall (grouped byte-split 1.47×); the compute itself halves at ~1.95× (98 % of ideal), the gap being the round-trip wall |
| 4 | Subarrays / rows as compute elements | ~0.2 % of rows touched | one 16-row tuple + a few-hundred-row pool per bank, of ~10⁵ rows/bank |
| 5 | PCIe bandwidth | **~0.3 %** | ≈17 MB/s effective vs multi-GB/s XDMA |
| 6 | Host orchestrator | ~free | collapsing 81,689 → 10,541 requests/token bought ~1 % of wall |

Multiply the levels and the honest headline is that **the die is ~99.99 % idle**.
That is not an indictment — it is the room, and it is why the measured ladder
keeps producing integer-factor wins (632 → 45 s/tok, STOCKTAKE) without touching
the physics.

**What binds, in order.** Not the bus and not PCIe bandwidth — the **per-program
round-trip at the FPGA/host interface**. In the measured 5.9 ms/program
decomposition (`wcol 1.3 + exec 1.0 + recv 3.1 + pop 0.2 + other 0.3` ms) the
binding term is `recv 3.1 ms` — XDMA small-transfer serialization, the readback
drain, *not* PCIe bandwidth — followed by operand loading (`wcol`) and
fetch-limited command issue (`exec`). The walls, ranked:

1. **Readout** — the largest term. In-DRAM popcount / segment reduction (POPCNT3
   in spirit) shrinks the answer at the sense amps and kills it. This is the whole
   Road-A / Road-B / SEG_POP arc — an emulation of one missing die feature.
2. **Operand movement** (`wcol`) — killed by first-class in-die RowClone /
   broadcast; the coset-broadcast technique is the bootleg version.
3. **Command issue** (`exec`) — killed by streamed execution (seq_engine); it is
   sequenced *after* readout because it is the smaller term.
4. **Only then the DDR bus** — the `casa_sched` bus-bound floor, ≈0.02–0.04
   s/token on two compute channels (STOCKTAKE, a projection labelled as such).
   MVDRAM's §V-E is the existence proof of that regime: a host generating commands
   faster than DDR4's ~1.5 ns/command consumption, i.e. a saturated PHY. We are
   ~2 orders below it; the fetch-side path is the climb.

**The request-count law.** The binding term is a per-request round-trip
(host↔DRAM, a roughly fixed cost × request count), so **only cuts to request
*count* move the wall.** Byte-size cuts and compute-issue levers are wall-neutral
against it — measured, not assumed: a request-batching change (internally V2G)
collapsing 81,689 → 10,541 requests/token bought ~1 % of wall (a count cut helps
only when it also removes round-trips), and a bank-parallel compute-issue lever
(internally pack4), dump-verified to actually engage, moved the wall **3.3 % —
within run variance**, because command
issue is a minority of the round-trip against readout. A compute-issue lever
cannot move a readout-bound wall; sequence the readout first.

**What a memory die would need** (ranked by which measured wall it kills; every
one is periphery, not a new cell):

1. **A reduction unit on the row buffer** (per-segment or whole-row popcount at
   the sense amps) — kills readout, the largest term. Patents of this shape exist
   (Micron US10068652, Samsung US9836277).
2. **Multi-row activation as a documented primitive** with decode isolation and
   guaranteed margins — kills the correctness tax: the selection law,
   spread-safe placement, and the per-module lottery all exist only because we
   drive the row decoder out of spec.
3. **In-die row-copy / broadcast as first-class commands** (RowClone,
   JEDEC-blessed) — kills operand movement.
4. **Subarray-parallel issue** (SALP-class) — the die already has the spatial
   parallelism; the interface serializes it. MVDRAM's subarray×module partitioning
   and our dual-DIMM compute-halving (~1.95×, 98 % of ideal) both sample its edge.
5. **Compute-region refresh semantics** — refresh restores charge, not content;
   compute rows want maskable refresh or latched staging.
6. **A "PUD-capable" datasheet bin** — the exact named part doing zero PUD across
   60,000 pairs while working perfectly as memory is a lottery a vendor could end
   with one binning test.

**One line:** the primitive is near physics; the die is ~99.99 % idle; the walls
are, in order, readout → operand movement → command issue → and only then the
bus — and every one is a periphery fix, not a cell fix. (Characterization
transfers across banks and same-model dies, so scaling from 4 banks to 16 needs a
cheap margin re-screen, not re-derivation — bank-similarity audit in
BANK_AUDIT_2026_07.)

---

## 6. A note on measurement: multi-timing sweeps confound geometry and timing

One methodology point for anyone characterizing multi-row activation on this
kind of silicon, stated once because it changes how "MAJ reliability" numbers
should be read.

A double activation is **two physics selected by a timing** — a clean majority
vote at one operating point, a multi-row copy a few command slots later (the
boundary is 4 NOP slots, 6.0 ns at tCK = 1.5 ns). A sweep that bundles several
timings into **one program** therefore cannot attribute an observed effect to a
single timing. The specific trap: a deposit fires at a *dirty* swept timing and
rewrites some of the operation's operands; a later *clean* vote in the same
program then votes correctly over those already-rewritten operands and reads
back as if the clean vote itself had deposited. The wrong reading is a lawful
majority over a substituted operand set — it looks like an at-the-operating-point
failure, and it is not.

The consequence is concrete: a standard multi-timing "MAJ reliability" score is
**geometry × timing confounded**, not a measure of cell quality alone. Two
practices follow.

- **Score each timing in isolation** — one timing per program, with an operand
  readback between preparation and the vote, so a deposit is attributed to the
  timing that caused it rather than to the timing that read it back.
- **Read a "reliability" grid as geometry × timing**, not as a per-cell quality
  map: the same cell can score reliable or unreliable depending only on where the
  swept program left the coset.

A second, independent trap — dissected on our silicon — scores **tie polarity**
rather than vote reliability. Trials that embed filler rows at full strength
(e.g. all-ONE reference rows the recipe assumes are charge-weakened, an
assumption some silicon does not honor) manufacture exact per-bit ties. At an
exact tie the sense amplifiers resolve by a fixed, row-region-specific polarity
— not by operand content (swap-tested: flipping the first row's data does not
change the direction). A scorer comparing against an expected value then reads
that fixed polarity as instability. Measured on one of our Micron modules: one
tuple's stability score collapses to ~3% at the clean operating point while a
control tuple in a region of opposite polarity scores ~79% — same die, same
program, reproduced across all four banks; direct operand readback under the
identical program shape shows zero deposits. So a low fanout-4 "MAJ
reliability" score can be pure tie-polarity accounting.

Applied to the comparison in this document: the zero-perfect-MAJ5 verdict (§2.6)
and the MAJX ratings that feed it come from the standard bundled-timing,
full-strength-filler methodology, so they bound *that* methodology's result — a
per-timing, spread-aware, tie-aware re-screen could move them. The mechanism itself (the two regimes,
the coset, the selection law) is in the [mechanism explainer](explainer/xor-spread.html).
