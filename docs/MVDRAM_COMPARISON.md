# MVDRAM vs CaSA — a mechanics comparison

> **Update (July 2026):** we have since attempted a full hardware
> reproduction of MVDRAM, including on two new units of the exact DRAM part
> number the paper names (SK Hynix HMA851U6CJR6N-UHN0). **Negative result —
> the named part performs no PUD at all in our hands, and the paper's chained
> dataflow breaks on every PUD-capable module we own.** See
> **[MVDRAM_REPRODUCTION.md](MVDRAM_REPRODUCTION.md)** for methods, numbers,
> reproducer code, and raw logs. The comparison below predates that study and
> takes the paper's claims at face value.

[MVDRAM (arXiv:2503.23817)](https://arxiv.org/abs/2503.23817) — Kubo, Tokuda,
Nagatani, Usui (U Tokyo), Qu (Microsoft Research), Cao (Tsinghua AIR),
Takamaeda-Yamazaki (U Tokyo) — is the closest published peer to this project:
GeMV for low-bit LLM inference executed in **unmodified commodity DDR4**, on
the **same testbed family** (DRAM-Bender on a Xilinx Alveo U200), with weights
resident in DRAM. Anyone evaluating CaSA should read it; this note maps the
two systems onto each other precisely — where the mechanisms differ, why the
performance differs by orders of magnitude, what each system can and cannot
run on, and what each should borrow from the other.

Everything below about MVDRAM is from the paper (v2). Everything about CaSA
is measured on our silicon and documented in this repo.

## TL;DR

| | MVDRAM | CaSA (this repo) |
|---|---|---|
| Goal | throughput-competitive GeMV vs CPU/GPU | verifiable end-to-end correctness of a production ternary LLM + failure-mechanism characterization |
| Platform | DRAM-Bender, Alveo U200, 4× SK Hynix DDR4-2400 | DRAM-Bender, BCU1525 quad, DDR4 |
| Partial products | **selective RowClone** (activation bit picks the copy source) — no MAJ involved | activation broadcast + MAJ3-as-AND on calibrated tuples |
| Accumulation | in-DRAM MAJ-based full adders (carry = MAJ3, **sum = MAJ5**, dual-rail complements) | readback + host/FPGA popcount |
| Error handling | per-column profiling + Frac margin; use reliable columns only (83–94 %/module) | per-tuple calibration, 16-row replication, fault-aware pool layouts, mechanism analysis |
| Headline | up to 7.29× GeMV speedup vs CPU; 2.18× end-to-end (2-bit Llama2-13B, measured) | BitNet b1.58-2B-4T end-to-end on silicon, 99.98 % bit-exact outputs, correct answer |
| Accuracy evaluation | **none reported** (latency/energy only) | bit-exact per-projection verification vs PyTorch reference |
| Correctness criterion | screened **column subset** (83–94 % of columns; part does MAJX up to 15) | **whole-row bit-exact** — under which logical MAJ5 yields zero perfect configs on our modules (best 99.98–99.99 % stability) while 16-row-replicated MAJ3 yields hundreds |

The punchline of the comparison: the two systems demand different things
from the silicon, so neither hot path transplants directly. MVDRAM's MAJ5
adders have zero whole-row-perfect configurations on our modules — best
cases reach 99.98–99.99 % per-cell stability, which is fine for per-column
screening and fatal for bit-exactness — while **CaSA's correctness
discipline addresses exactly what MVDRAM's evaluation leaves open** (no
accuracy numbers, column-static error model). The two systems are
complementary evidence for the same thesis: commodity DRAM is secretly a
compute substrate.

## 1. MVDRAM's three mechanisms, precisely

### 1.1 On-the-fly vector encoding — multiplication without computing

The activation vector is known to the *host* when it issues DRAM commands. So
MVDRAM never stores activations in DRAM and never computes `w AND a` in DRAM.
Instead, the partial product is encoded into the **choice of RowCopy source**:

- activation bit `a = 1` → RowClone the **matrix row** into the compute row
  (result = w)
- activation bit `a = 0` → RowClone the **constant-zero row** into the compute
  row (result = 0) — or skip the operation entirely (their sparsity
  optimization; their evaluation assumes 50 % input bit-sparsity)

Two structural consequences:

1. **Weight rows are only ever ACT sources.** A normal activation is a
   destructive read *with restore* — the sense amp rewrites the source row.
   So matrix rows survive every use without any explicit restore step.
   Weights are written once before inference and never rewritten.
2. **The product step needs zero multi-row activations.** RowClone is the
   most robust PUD primitive (it works at much wider margins than MAJ);
   the fragile operations are pushed entirely into the accumulation stage.

Command-stream cost: they note DDR4-2400 consumes roughly one command per
1.5 ns and a single-threaded encoder generates commands faster than that, so
encoding overlaps execution and is fully masked.

### 1.2 Horizontal matrix layout — linearity instead of transposition

Conventional PUD stores operands *vertically* (all bits of one value on one
bitline), which assigns one column per output, wastes the ~65 536-column
parallelism, and forces a bit-transposition pass on the outputs.

MVDRAM instead bit-decomposes the MAC **by matrix bits**:
`o_m = Σ_i 2^i · o_{m,i}` with `o_{m,i} = Σ_j a_j · w^(i)_{m,j}`. Each
partial sum `o_{m,i}` is an independent binary inner product, so the matrix
can stay in row-major (horizontal) layout: one DRAM row per input index `j`,
holding bit `i` of `w_{m,j}` for all outputs `m` along the columns. One
selective RowClone then feeds `q·M` partial products simultaneously (`q` =
weight bits, `M` = outputs). Output bits land in `q × r` rows that the host
reads **row-wise** and combines with shifts — no transposition. They cap
`N ≤ 128` per subarray and shard the matrix across subarrays and across the
four modules.

### 1.3 In-DRAM accumulation — MAJ adders, dual-rail

The `Σ_j` reduction happens inside the subarray with bit-serial full adders
built from majority: carry `s1 = MAJ3(x0,x1,x2)`, sum
`s0 = MAJ5(x0,x1,x2,s̄1,s̄1)`. Unmodified DRAM has no native NOT, so they run
**dual-track**: every value is kept with its complement throughout (the
matrix is stored twice — original + inverted — which dominates their capacity
overhead). Readback per GeMV is then only the `q × r` output rows
(e.g. 0.05 ms aggregation for a 32000-output GeMV), not raw bitwise results.

### 1.4 Reliability strategy

Per-column profiling + FracDRAM-style Frac operations to widen margins; the
GeMV uses only consecutive runs of `q` **reliable columns**. On their four
modules: 54 365–61 727 reliable columns of 65 536 (83–94 %). The part number
(HMA851U6CJR6N-UHN0) was selected by characterizing **16 different SK Hynix
models** and picking the most cooperative one — it supports strict RowClone
and MAJX up to **MAJ15**. Their robustness numbers across temperature
(50→90 °C: −0.07 % reliable columns) and voltage (2.5→2.1 V: −0.41 %) are
cited from prior characterization work.

### 1.5 Measured results (their Table II / Figs 12–17 setup)

Baselines: Intel i7-9700K with the *same* DDR4-2400 modules (77 GB/s),
llama.cpp/ggml low-bit kernels; GPU = Jetson Orin Nano (LPDDR5, energy
normalized). GeMV 32000×4096 at 1-bit vector × 2-bit matrix: CPU 1.44 ms,
GPU 1.70 ms, MVDRAM 0.19 ms (0.14 in-DRAM + 0.05 aggregation) → 7.29×/8.55×.
End-to-end (llama.cpp with `mulmat` routed to DRAM, 256 tokens × 10 runs):
2-bit Llama2-13B 2.18× vs CPU, 4-bit 1.31×. Energy 30.5× (GeMV, vs CPU
RAPL) and 3.04× (end-to-end 2-bit); MVDRAM's own energy is **CACTI-modeled**,
not wall-measured. The paper reports **no accuracy or perplexity numbers**
for the PUD-executed models — correctness rests on the reliable-column
screening.

## 2. Why MVDRAM's throughput is orders of magnitude above CaSA's measured number

Our measured full-model number is ~30 s/token (README); MVDRAM generates
2-bit Llama2-13B tokens 2.18× *faster* than a desktop CPU running llama.cpp.
The gap — several orders of magnitude in per-token wall time — decomposes
into identifiable, mostly architectural causes, not physics:

| Gap source | MVDRAM | CaSA today | CaSA's counter (status) |
|---|---|---|---|
| Weight residency | RowClone sources, restored for free by every ACT | per-MAJ3 weight reload from backup pool (dominant cost) | adopt selective-RowClone products (§3.1) |
| Error tax | column screening + Frac margin; ~1 RowClone per partial product | replicated 16-row tuples — the MAJ3 itself is **one** simultaneous 16-row activation (no op multiplication); the tax is ~6 row-scale setup ops per MAJ3 (broadcast + 5 activation-slot writes + frac) plus a 16-row footprint per logical op | Frac experiment; tuple-width reduction (planned) |
| Parallelism per op | q·M across ~60 k reliable columns | 2 048-column output slice × 4 banks | layout work; multi-subarray |
| Controller path | streamed commands, encoding overlapped, near line-rate | program-per-execute, 2 048-instruction IMEM, c2h drain per body (~2 % bus util measured) | seq_engine.v reaches 100 % PHY in Verilator (awaits bitstream) |
| Accumulation | in-DRAM MAJ adders; read q×r rows only | drain raw rows, popcount on host | FPGA popcount accumulator HDL staged |
| Sparsity | skips a=0 ops (50 % assumed) | none | free once products are RowClone-selected |

Two caveats in their favor stay caveats in ours: their CPU baseline is a
desktop part running low-bit ggml kernels (memory-bound, with unpack
overhead), and the 50 % sparsity assumption halves their op count. Neither
changes the architectural picture.

## 3. What CaSA should adopt from MVDRAM

1. **Selective-RowClone partial products** (their §V, adapted to ternary).
   BitNet's ternary weights are exactly their `q = 2` case: pos/neg masks
   are two "matrix bit-planes," and `y = Σ_j x_j·pos_j − Σ_j x_j·neg_j`.
   Per activation bit-plane `c`, RowClone weight-row `j` into the
   accumulation region iff `x_c[j] = 1`, else skip. This removes the
   activation broadcast, the MAJ3-as-AND, the calibrated-tuple constraint,
   *and* the per-MAJ3 weight reload from the product step — on our silicon,
   RowClone is the one primitive that works everywhere (100 % across all
   banks at the (30,1) timing). The missing piece on our chips is the
   accumulation: their MAJ5 adder tree has no whole-row-perfect
   configuration on our modules (see §4), so under our bit-exactness
   requirement the reduction must happen FPGA-side — a per-column vector
   accumulator in the readback path (a sibling of our staged popcount
   accumulator HDL).
2. **Frac conditioning.** They use Frac operations to *increase the number
   of reliable columns*. If targeted Frac tuning raises per-cell MAJ3
   stability on our modules, the replication width can shrink — cutting the
   per-MAJ3 setup writes that dominate our bus-bound ceiling and freeing
   tuple capacity — and marginal chips (our DIMM 3 class) may become
   usable. If it lifts logical MAJ5 to whole-row-perfect on even a few
   tuples, MVDRAM-style in-DRAM adders open up *under our bit-exactness
   criterion* (§4).
3. **Streaming command generation.** Their encoding-overlapped-with-
   execution model is the software twin of our sequence-engine HDL work;
   it confirms that program-per-execute (not DRAM physics) is the
   controller-path ceiling on this testbed family.

## 4. The real asymmetry — correctness requirement, not row count

This is easy to misread, so precisely: CaSA co-activates **16 rows in every
production MAJ3** (and characterized 4/8/16/32-row simultaneous activation) —
more rows at once than MVDRAM's MAJ5 ever opens. The difference is not how
many rows the silicon activates; it is what each system demands from the
result:

- MVDRAM runs logical MAJ5 with **one copy per operand** and needs
  correctness only on the **screened column subset** (83–94 % of columns on
  their part, which supports MAJX up to 15).
- CaSA requires **bit-exact results across the full row**. Under that
  criterion, on both of our production modules (6 400 timing/tuple
  configurations per rows×majX point): logical MAJ3 with 16-row replication
  yields hundreds of perfect tuples (363 on one module, 505 on the other),
  while logical MAJ5 yields **zero** — its best configurations reach
  99.98–99.99 % per-cell stability, almost but never exactly
  whole-row-perfect.

Two corollaries. First, MVDRAM's adder tree would plausibly run on our
modules **under MVDRAM's own error model** — a 99.98 %-stable MAJ5 is
exactly the regime per-column screening exists for. Untested, but not
excluded by our data. Second, the converse stands: a bit-exact pipeline
cannot use MAJ5 adders on this silicon, so our reduction stays outside the
array (host today, FPGA accumulator next). The "chip lottery" framing
remains — their screening (16 module models → 1 winner) vs our
characterization of whatever is in the slot — but it is a lottery over
*margins under your correctness criterion*, not over raw multi-row-activation
capability.

(Honesty note: these MAJX ratings come from the standard MajOps
methodology, which our self-pollution finding shows is geometry-confounded —
see the [XOR-spread explainer](https://pcdeni.github.io/CaSA/explainer/xor-spread.html).
The zero-perfect-MAJ5 verdict could in principle improve with spread-aware
tuple selection; we have not re-screened.)

## 5. What CaSA offers MVDRAM-class systems

1. **A deterministic error mechanism their methodology cannot see.**
   MVDRAM's error model is column-static: a column is reliable or it is not,
   independent of operands. We measured two effects that violate that
   assumption on every module we tested:
   - the [doubleACT row-spread](https://pcdeni.github.io/CaSA/explainer/xor-spread.html):
     any PRE-violating double activation deposits a bit-exact copy of its
     source row into address-XOR sibling rows (chip-specific vulnerable-bit
     fingerprint, row-decoder coupling);
   - **MAJ self-pollution**: when a shadow address falls inside the
     operation's own open-row set, the operation corrupts its own operands
     mid-flight — error rates then depend on *row geometry and operand
     placement*, not on the column.
   A system doing millions of RowClones with weight rows as sources (exactly
   MVDRAM's §V) on a module with our coupling fingerprint would corrupt
   matrix rows in a pattern no per-column profile predicts. Their chosen
   Hynix part may be mild in this respect — unknown; ours measurably is not.
   Spread-aware row placement (our independent-set pool layouts in
   `calibration/`) is the fix, and it composes with their layout.
2. **A verification methodology.** Bit-exact per-projection comparison
   against a PyTorch reference, per-cell stability calibration, and an
   end-to-end correctness demonstration on a production ternary model —
   the evaluation dimension their paper leaves entirely open.
3. **A ternary-native path for stricter correctness targets** — chips (or
   requirements) where only replicated MAJ3 survives a whole-row bit-exact
   criterion, per §4.

## 6. Bottom line

MVDRAM proves the *throughput* leg of the thesis on screened silicon;
CaSA proves the *correctness and characterization* leg on unscreened
silicon. The systems disagree on almost no facts and overlap on almost no
contributions. A merged design — selective-RowClone products + spread-aware
layout + Frac-conditioned margins + FPGA-side accumulation + bit-exact
verification — is the obvious next system, and most of its parts already
exist across the two codebases.
