# MVDRAM reproduction study — negative result

**We attempted to reproduce MVDRAM ([arXiv:2503.23817](https://arxiv.org/abs/2503.23817),
"Enabling GeMV Execution in Unmodified DRAM for Low-Bit LLM Acceleration") on
six DDR4 modules, including two brand-new units of the exact DRAM part number
the paper names. We could not reproduce it on any of them.**

- **The paper's named part (SK Hynix HMA851U6CJR6N-UHN0, 2 new units): performs
  no processing-using-DRAM at all in our hands.** The modules function
  perfectly as memory, but zero charge-sharing row copies occurred in 60,000
  random row-pair attempts plus exhaustive timing sweeps. A part that performs
  no PUD operations cannot execute any part of MVDRAM's method.
- **Four commodity DDR4 modules that DO perform PUD: every MVDRAM mechanism
  reproduces in isolation, but the paper's chained dataflow collapses**
  (6–11% correct vs 99.9% for our host-mediated variant), due to the
  [XOR-spread](https://pcdeni.github.io/CaSA/explainer/xor-spread.html) — a
  deterministic row-decoder artifact we characterized, which the paper does
  not mention and which makes copies into a co-activated row set inherently
  destructive on these dies.

Everything below is measured. Reproducer code is in this repository; raw logs
in [`data/mvdram-repro/`](data/mvdram-repro/).

---

## 1. What MVDRAM claims

MVDRAM executes GeMV for low-bit LLMs inside unmodified DDR4 using two PUD
primitives — RowCopy (charge-sharing row-to-row copy via ACT–PRE–ACT with
violated timing) and MAJX (majority over X co-activated rows) — chained so
that operands move between rows *inside the DRAM*, avoiding the memory bus.
Its performance claims (up to 7.29× over CPU) rest entirely on that
chainability: matrix rows are preloaded once, then per-inference the only
in-DRAM traffic is RowCopy + MAJX sequences, with a small result readout.

The paper's hardware basis (its §VII, footnote 3): four SK Hynix DDR4-2400
modules, part **HMA851U6CJR6N-UHN0**, selected by characterizing 16 modules
as "the most reliable one that supports both strict RowCopy and MAJX
operations (up to MAJ15)", on DRAM-Bender / Xilinx Alveo U200.

## 2. Our setup

- Xilinx BCU1525 FPGA, four DIMM slots, **DRAM-Bender** (the same open-source
  memory-controller framework the paper uses) with a quad-controller
  bitstream; command sequences and timings issued by the same primitives the
  SAFARI artifacts use (`doubleACT(t_12, t_23, src, dst)`).
- Six modules tested: **2× new HMA851U6CJR6N-UHN0** (purchased new,
  June 2026) and **4 commodity DDR4 UDIMMs** on which PUD demonstrably works
  (the modules this repo's BitNet-in-DRAM results run on).
- Ambient temperature, DRAM-Bender default frequency. (A temperature-controlled
  retest is planned; see §6.)

## 3. Result A — the paper's named part performs no PUD

The two new HMA851U6CJR6N-UHN0 units **work correctly as memory**: per-column
writes and reads of fingerprint patterns across many rows and banks verify
100% (25/25 constellation rows in the spread-test baseline; full row
write/readback used throughout).

They perform **zero** charge-sharing operations:

| Experiment | New unit A | New unit B | Control (PUD-capable module) |
|---|---|---|---|
| RowClone, fixed pairs, t_12 swept 5→150 (14 values), t_23 1→4 | best 41/8192 bytes (noise floor) at every timing | same | 8192/8192 at t_12=30 |
| RowClone, **30,000 random pairs** per module, rows uniform in [0, 65536) | **0 clones**, 0 partial (>1000 B), best 45/8192 | **0 clones**, best 47/8192 | 6 full clones in **500** pairs (1.2%) |
| SiMRA characterization sweep (RowClone stage) | 9.3M attempts, max match **0**, zero co-activatable row groups (all "subarrays" are singletons) | same | 640-row subarray structure found within minutes |

The random-pair experiment matters: row addresses in DRAM are scrambled
relative to physical location, so a fixed-stride pair can straddle physical
subarrays. Random pairs across the whole address space land in the same
physical subarray at the ~1% rate the control module shows — on the new
units, **not one pair in 60,000 interacted**. This rules out address
scrambling, timing choice, row choice, and dead modules as explanations. The
charge-share effect that all of MVDRAM rests on is absent.

Reproducers: [`app/test_rowclone_random.cpp`](../app/test_rowclone_random.cpp)
(random-pair scanner), [`app/test_rowclone_smoke.cpp`](../app/test_rowclone_smoke.cpp)
(fixed-pair with `PIM_T12` timing override). Logs:
[`data/mvdram-repro/`](data/mvdram-repro/).

## 4. Result B — where PUD works, MVDRAM's dataflow breaks

On our four PUD-capable modules, every MVDRAM *mechanism* reproduces in
isolation, loading operands by host-mediated per-column WRITE:

| MVDRAM mechanism | Our result (best module) |
|---|---|
| RowClone between ordinary rows | 8192/8192, deterministic |
| MAJ3 / MAJ5 via multi-row activation | 99.99% / 99.0% per-op |
| Reliable-column screening for MAJ5 | 87–88% reliable columns (paper reports 83–95%) |
| Dual-track full adder (carry=MAJ3, sum=MAJ5) | 99.94% on screened columns |
| In-DRAM carry-save popcount tree | 99.97–99.98% |
| Complete signed q-bit × r-bit GeMV (2-bit & 4-bit) | 99.9%+ bit-exact vs integer reference |

But MVDRAM's *dataflow* — the thing its performance claims rest on — requires
RowCopy-ing operands **into the co-activated row set** (its Fig. 2: copy
inputs to computation rows → MAJ → copy result out; its §V "on-the-fly vector
encoding": product = selective RowCopy into the compute region). On every
PUD-capable module we own, that step is destructive:

| MVDRAM dataflow step (faithful implementation) | Result |
|---|---|
| Computation-rows dataflow (RowCopy in → MAJ → RowCopy out), full popcount | **6.1%** end-to-end; per-op MAJ 50.3% (coin flip) |
| On-the-fly vector encoding (products by selective RowCopy of preloaded matrix) | **11.3%** vs **99.9%** for the identical GeMV with host-WRITE products |
| RowClone-loading a calibrated 16-row MAJ tuple | 50.1% (operands destroyed) |
| Same, minimal 4-row tuple | 75% |
| Mitigations (full-restore after each copy; non-shadow source selection) | no improvement |

The mechanism is the **XOR-spread**, a finding of this project
([interactive explainer](https://pcdeni.github.io/CaSA/explainer/xor-spread.html),
reproducers [`app/test_spread.cpp`](../app/test_spread.cpp),
[`app/test_fault_sweep.cpp`](../app/test_fault_sweep.cpp)): a `doubleACT`
whose destination is a member of a co-activatable row group also deposits the
source row's content, bit-exactly, into rows at address `src XOR (1<<b)` for
a chip-specific set of bits `b`. It is deterministic, repeatable,
timing-independent (fires at RowClone, broadcast, and MAJ timings alike), and
**bank-invariant** — the fault map is byte-identical (matching MD5) across
all four banks of a module, which places its origin in the **row-decoder
structure**, not cell physics. It is present on all four of our PUD-capable
modules, with chip-specific vulnerable-bit fingerprints. The MAJ operation
itself pollutes its own operands the same way (on one module, a tuple rated
"100% reliable" by standard characterization has 2 of its 16 open rows
silently overwritten during every MAJ — the result survives only because a
14-vs-2 majority absorbs it).

The consequence is structural: on these dies, the rows that can be
co-activated for MAJ are exactly the rows that are decoder-coupled, and
copying data into a decoder-coupled group corrupts the group. **The basic
operations work; chaining them through DRAM to avoid the memory bus — the
entire point of MVDRAM — does not.** We developed a restricted
operands-in-place chaining that avoids reloads (intermediates flow through
shared co-activatable rows; 99.97% on a 4-input popcount — see repo), but the
general copy-based dataflow the paper assumes is not achievable, and operand
loading + spills still cross the bus.

We note the field's standard model of DRAM (cells + sense amplifiers) does
not predict this artifact; the row decoder is usually treated as transparent.
It isn't. Prior work we could find documents bitline-side neighborhood
effects at JEDEC timing (PARBOR, DSN'16) but nothing describing this
row-decoder-side, PUD-timing effect. Characterization methodologies that
model errors per-column (reliable-column screening, per-column conditioning)
cannot represent it, because it depends on which rows are co-activated, not
on the column.

## 5. Conclusion

**We were unable to reproduce MVDRAM on the DRAM part number its authors
name, nor on any other module available to us (0 of 6).** The named part, as
purchasable today, performs no processing-using-DRAM whatsoever on a
DRAM-Bender rig — it cannot execute the paper's primitives, let alone its
pipeline. Modules that *can* execute the primitives cannot chain them the way
the paper requires, for a measured, mechanistic reason (a real row-decoder
artifact) that the paper does not mention and that its screening procedure
would have silently selected around. A method whose viability depends on
undisclosed properties of individual screened modules is not, in any
practical sense, "GeMV execution in unmodified DRAM."

## 6. Falsifiability — what would change this conclusion

We state plainly what evidence would revise this negative result:

1. **Module provenance**: date codes / SPD dumps of the four specific modules
   the paper used, so batch-level differences become checkable.
2. **A runnable reproducer**: the paper released no code. Ours is public and
   runs on any DRAM-Bender setup; the random-pair scan takes minutes.
3. **A loaner module** that demonstrably performs strict RowCopy: we will run
   the full study on it and publish the result either way.
4. **Temperature**: our tests ran at ambient; PUD margins are
   temperature-dependent and SAFARI-line characterizations are typically run
   temperature-controlled. We will publish a heated (~55–70 °C) retest of the
   new units. For the conclusion to flip, heat would have to take these parts
   from *zero* charge-share to "strict RowCopy + reliable MAJ15".
5. For any PUD-capable module claimed to run the full dataflow: run our
   1-minute spread test (`app/test_spread.cpp`) and publish the fingerprint.
   If a die shows genuinely clean copies into co-activated groups, MVDRAM's
   dataflow becomes plausible on that die — we have not encountered one.

## 7. What does work: LLM inference in unmodified DRAM, with the artifact engineered around

This repository documents the system that works on commodity, spread-afflicted
silicon: Microsoft's BitNet b1.58-2B-4T with **all 30 transformer layers'
projection matmuls executed inside unmodified DDR4** (ternary weights resident
in DRAM, MAJ3-based multiply-accumulate, correct model output), including the
calibration methodology that makes it reliable *because* it models the
XOR-spread (independent-set row pools, per-bank fault maps) instead of
assuming clean silicon. See the [README](../README.md) and
[interactive explainer](https://pcdeni.github.io/CaSA/explainer/) for the
mechanism, measurements, and the bus-bound ceiling analysis.

*Correspondence with the MVDRAM authors is ongoing; we will update this study
with any evidence they provide (date codes, reproducers, or modules).*
