# MVDRAM paper (arXiv:2503.23817v2) — precise contrast with our reproduction

Re-read 2026-07-17 against the v2 PDF (23 Sep 2025). Every claim below cites
the paper (§/Fig/Table) or our reproducer/log. Feeds the planned MVDRAM
reproduction explainer + doc updates.

## 1. What they ran, exactly

- **System** (§VII, Fig 11): host PC + DRAM Bender [their ref 47] on a
  **Xilinx Alveo U200** + SK Hynix **DDR4-2400 HMA851U6CJR6N-UHN0** modules.
  Footnote 3: they characterized **16 SK Hynix modules** and identified this
  part as "the most reliable one that supports both strict RowCopy and MAJX
  operations (up to MAJ15)".
- **Reliable columns** (Table I): min 54,365–61,727 of 65,536 per module
  (82.9–95.1%). They compute only on **consecutive runs of q reliable
  columns** (§VII "Reliable MAJX") and use **Frac operations [34] and
  calibration techniques [48]** to raise the reliable count, "which achieves
  error-free computation".
- **Dual-track** (§VII): both polarity copies of everything (Fig 15 shows
  matrix rows + inverted matrix rows ≈ doubling matrix storage); full adder
  carry s1=MAJ3(x0,x1,x2), sum s0=MAJ5(x0,x1,x2,¬s1,¬s1) (§II-C1).
- **Partitioning** (§VII): **N ≤ 128 per subarray**, partitioned across
  subarrays and 4 modules.
- **Baselines** (Table II): CPU = Intel i7-9700K with the **same DDR4-2400
  modules** (77 GB/s); GPU = **NVIDIA Jetson Orin Nano** (LPDDR5, 68 GB/s;
  an edge part — not a datacenter GPU). GeMV kernels via ggml [49].
- **GeMV benchmark method** (§VIII-A): dims of modern LLMs, weight
  precision 2–8 bit, averages over **1,000 iterations**, inputs with **50%
  bit sparsity** ("typical LLM distribution", [45][46]).
- **End-to-end method** (§VIII-A): four models — **Llama2-7B, Llama2-13B,
  Llama3-8B, Phi-4** — on **llama.cpp**, "where we replace mulmat_op
  operations with our MVDRAM implementation"; 256 generated tokens,
  averaged over 10 runs. This is a REAL end-to-end run through DRAM, not a
  projection.
- **Headline numbers** (§VIII-B): 32000×4096 GeMV (llama2-7B dims), 1-bit
  vector × 2-bit matrix: CPU 1.44 ms, GPU 1.70 ms, MVDRAM **0.14 ms
  in-DRAM + 0.05 ms aggregation = 0.19 ms → 7.29×/8.55×**. Scaling to
  32768×32768 2-bit: 3.38×/3.74×. End-to-end (Fig 16): 2-bit Llama2-13B
  **2.18×** vs CPU (3.33× vs GPU); 4-bit **1.31×** vs CPU. Energy (Fig 14,
  17): 30.5× GeMV / 3.04× per-token vs CPU — **MVDRAM power is
  CACTI-estimated** [56], baselines measured (RAPL/tegrastats).
- **Environmental robustness** (§IX): cited from prior work [36], not
  measured by them: 50→90 °C costs only 0.07% of reliable columns;
  2.5→2.1 V ≤0.41%.

## 2. What we reproduced (and where it matches)

| Paper element | Their number | Ours | Source |
|---|---|---|---|
| Reliable-column fraction | 83–95% (Table I) | 87–88% MAJ5-reliable | mvdram-maj5-exe, REPRODUCTION §4 |
| RowCopy | "strict" on screened part | 8192/8192 deterministic (commodity) | rowclone-smoke |
| Dual-track adder | error-free after calibration | 99.94% (MAJ5 sum, screened cols); **99.98% all-MAJ3 variant**; **2026-07-20: complements formed IN-DRAM (Fig-15 ~W planes + De-Morgan rails) as a server mode — 99.49% unvoted / 99.83% vote3 at 4096², 2× wall vs host-formed** | mvdram-adder-exe, mvdram-fulladder-exe, lane2 LANE2_DUALTRACK |
| On-the-fly encoding (§V-C) | RowCopy source selected by activation bit | **reproduced PHYSICALLY 2026-07-20 (was: host-resolved): LANE2_ENCODE=clone — products created by clones from resident W/~W rows, 3.0× faster than the write-load arm at 4096² GeMV scale (12 s vs 38 s), 94.7% unvoted on commodity DIMM** | lane2 clone mode, test_mvdram_fastpath_ab.cpp |
| Bit-sparsity skip | §V-D (skip op when a=0) | host-side analog measured 4.37× command-stream reduction; in-server zero-skip in all modes | REPRODUCTION, casa_sched notes |
| Horizontal layout + row-wise aggregation | §VI | reproduced (bitline=output, row-major readout) | mvdram-gemv*-exe |
| Partial sums retrieved + aggregated by processor (§II-C2, §VII) | N≤128-per-subarray partials | **GEMV_PARTIALS 2026-07-20: exact per-32-block integer partials (M×K/32 i32) → host applies q4_0/q8_0 per-block scales → FIRST EXACT FP32 from the reproduction (bit-exact vs CPU fp32 reference on the real Llama-2-7B blk.0.attn_q tensor)** | lane2 GEMV_PARTIALS, lane2_partials_fp32.py |
| Faithful computation-rows dataflow (their Fig 2) | (their normal operation) | **99.98% e2e with safe placement** (June's 6.1% reversed) | test_mvdram_compute_rows_safe.cpp |
| Fast in-DRAM operand movement | (assumed by their Fig 3 profile) | 2.2–2.3×/gate over host-write shape; **3.0× per-GeMV in the fused clone server mode** | test_mvdram_fastpath_ab.cpp, lane2 clone mode |

## 2b. Our Table-I analog (reliable columns, per subarray)

Their Table I reports per-module min/max reliable columns of 65,536
(82.9–95.1%). Ours are per-SUBARRAY, MAJ5-op-matched, counted in 32-bit
segments (2048/row = 65,536 bits); "robust" = stricter repeat criterion:

| die | subarray | criterion | reliable segs | fraction |
|---|---|---|---|---|
| DIMM 0 | s61 | standard | 882/2048 | 43.1% |
| DIMM 0 | s77 | robust | 1806/2048 | **88.2%** |
| DIMM 2 | s72 | standard | 1188/2048 | 58.0% |
| DIMM 2 | s72 | robust | 62/2048 | **3.0%** |
| DIMM 2 | s86 | robust | 1784/2048 | **87.1%** |

Source: mvdram-repro/colmask_*.txt (mvdram-maj5-exe). Two observations:
best subarrays land inside their Table-I range; and the same module spans
3%→87% across subarrays under the robust criterion — **subarray selection
is load-bearing on commodity parts**, the intra-module analog of their
16-module screening. (Granularity differs from their table; state that
when publishing.)

## 3. Where our silicon differs from theirs (real divergences, not gaps)

- **The exact part number does no PUD in our hands** (2 new units, 0/60,000
  random pairs; Result A) — their footnote 3 says they *screened 16
  modules*; consistent with severe inter-module variance within one part
  number (date-code question still open with the authors).
- **MAJ5 on our M-die commodity modules**: only 1.37% of columns survive
  the *chained* MAJ5 adder (fused_maj_and_mvdram memory) even though
  single-op MAJ5 column reliability (87–88%) matches their Table I. Their
  "error-free" rests on Frac + calibration [48] **which we have not
  implemented** (see §4). Our all-MAJ3 adder is the workaround that needs
  no MAJ5 at all.
- **XOR-spread / co-activation lattice governs everything on our dies**
  (selection law, RESULT.md addendum 5); the paper never mentions it —
  either their screened part is low-spread, or screening + Frac masked it.

## 4. What we missed (actionable gaps on OUR side)

1. **Frac conditioning + calibration [their refs 34, 48]** — never
   implemented on our rig (our `frac_template` is a discharge primitive,
   not FracDRAM fractional-charge conditioning). This is their stated
   mechanism for reaching "error-free" MAJX; it might rescue MAJ5-chain
   columns on our silicon. Concrete next experiment.
2. **Streaming-scale execution.** Their 0.14 ms in-DRAM for a full
   32000×4096 GeMV implies on the order of 10⁵ DRAM commands issued as a
   continuous stream (their §V-E: a single-threaded CPU generates commands
   faster than DDR4-2400's ~1.5 ns/command processing rate — i.e. command
   generation overlaps execution). Our rig executes ~150-instruction
   programs with a host round-trip each (2048-inst IMEM bitstream). **This
   is the structural performance gap** — and it maps exactly onto our
   staged-but-unbuilt HDL: IMEM 2048→8192 + seq_engine (100% command-bus
   utilization in Verilator). Their paper is evidence that the streaming
   shape is what closes the last 2–3 orders of magnitude.
3. **Their models + llama.cpp integration.** We ran OUR model (BitNet
   b1.58-2B via transformers) — deliberately, but it means we never
   executed their exact benchmark (llama.cpp mulmat_op replacement,
   Llama2-7B/13B / Llama3-8B / Phi-4 at Q2/Q4, 256 tokens × 10 runs).
   A kernel-level equivalence argument covers the compute, but the
   explainer must present this as a *different end-to-end* than theirs.
4. **Benchmark conventions**: 50%-sparsity inputs, 1,000-iteration
   averaging, N≤128-per-subarray partitioning, per-module reliable-column
   tables (we have the colmask data; format it like their Table I).
5. **Their energy method is CACTI-estimated** for MVDRAM (baselines
   measured). Not a gap of ours — but the explainer should state it
   plainly when quoting 30.5×/3.04×.

## 5. Explainer skeleton (mvdram reproduction explainer, planned)

Scene plan (each claim → ledger entry):
1. What MVDRAM claims (Fig 3 profile; 7.29×/2.18× headline; their setup).
2. Their two techniques in pictures (on-the-fly encoding Fig 6; horizontal
   layout Fig 7–10).
3. What "reproduce" means here: paper-only, no source released; our rig
   (BCU1525 vs their U200; commodity DIMMs + their exact part bought new).
4. Result A — the named part does no PUD in our hands (0/60,000; their
   16-module screening footnote; date-code question).
5. Result B journey — June: dataflow collapses (6–11%) → the XOR-spread →
   July: safe placement reverses it (99.98%) → fused fast path 2.2–3×.
   (Honest arc: our own error corrected in public.)
6. The mechanism-by-mechanism scoreboard (§2 table above).
7. What we haven't done: Frac/calibration, their llama.cpp benchmark,
   streaming-scale execution (with the §V-E command-rate insight and our
   seq_engine path to it).
8. What our silicon adds beyond the paper: selection law, cross-die
   determinism, self-pollution, all-MAJ3 adder (MAJ5-free accumulation).
9. Verdict: achievable-with-spread-aware-addressing; part-number lottery
   is real; the readout/streaming shape is where performance lives.
