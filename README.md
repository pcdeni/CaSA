# CaSA: Charge-Sharing Architecture for Ternary LLM Inference on Commodity DRAM

CaSA is an inference architecture for ternary large language models that performs matrix-vector multiplication inside unmodified commodity DRAM using charge-sharing majority operations. The computation exploits the analog interaction between simultaneously activated rows in a DRAM subarray: when three rows are driven onto shared bitlines, the sense amplifiers resolve to a majority vote (MAJ3) across all columns in parallel. The architecture has been designed and simulated but has not yet been validated on physical hardware.

## Mechanism

### Charge-sharing AND via MAJ3

When three DRAM rows within the same subarray are activated simultaneously, charge sharing across the bitlines produces a per-column majority vote. If one of the three rows is pre-zeroed, the result reduces to a bitwise AND of the other two rows. For a ternary weight encoded as a single bit and a binary activation, this AND operation computes the product. The operation executes across all columns of the row (up to 65,536 bits on DDR4 with 8 KB rows) in a single activation cycle.

### SA-mediated RowCopy

The MAJ3 operation is destructive: all three source rows are overwritten with the result. To reuse weight data across multiple inference steps, a backup-and-restore protocol is required. CaSA uses sense-amplifier-mediated RowCopy, in which a source row is activated and latched by the sense amplifiers, and a target row is then activated while the sense amplifiers are still holding the source data. The sense amplifiers, operating as full-swing digital latches, drive the latched values into the target row. This produces a bit-exact copy with zero bit error rate, unlike unmediated RowCopy (which has a measured 16.3% BER on DDR4).

### Inference protocol

The activation-sacrificial protocol proceeds as follows for each weight row:

1. Back up the weight row to a reserved row via SA-mediated RowCopy.
2. Compute AND via MAJ3(weight, activation, zero_row).
3. Read the result through the memory bus for accumulation.
4. Restore the weight row from the backup via SA-mediated RowCopy.

### Popcount

After the AND result is read out, the number of set bits (popcount) must be computed to obtain the dot-product partial sum. In the baseline configuration, the full row is transferred over the memory bus for external popcount on the controller. An in-DRAM popcount register (approximately 2,000 gates per bank) can perform this reduction locally and return a single count value, eliminating the bus transfer.

## Key Results

All results are from cycle-accurate simulation calibrated against published DRAM characterization data. The target model is BitNet b1.58-2B-4T (2 billion parameters, ternary weights).

### DDR4 throughput (single DIMM, 8 GB)

| Configuration | Tokens/s | Notes |
|---|---:|---|
| 1 DIMM, no pipelining | 0.40 | Baseline on unmodified DDR4 |
| 1 DIMM, 4-bank pipelining | 3.38 | Firmware optimization only |
| 4 DIMMs, pipelined | 13.53 | Commodity hardware, no die changes |
| 4 DIMMs + in-DRAM popcount | 47.82 | Requires ~2,000 gates per bank |

### Cross-technology projections

| Technology | Tokens/s | Notes |
|---|---:|---|
| DDR5-4800, 4 DIMMs + popcount | 38.73 | Mandatory ODECC limits throughput |
| HBM2, 8-channel + popcount | 49.82 | Single stack |
| HBM2, 2-stack + popcount | 99.67 | Server configuration |

### Inference time breakdown (4-bank pipelined)

| Component | Share of inference time |
|---|---:|
| RowCopy (backup + restore) | ~0% (hidden by pipelining) |
| MAJ3 AND | ~0% (hidden by pipelining) |
| Bus read (result transfer) | ~98% |

### BER tolerance

At a bit error rate of 0.01% (far above the measured BER of charge-sharing operations on validated hardware), cosine similarity between the error-free and noisy hidden states remains above 0.999 for typical model dimensions. The margin between measured operational BER and the onset of measurable accuracy degradation is approximately 50,000x.

## Corrections (March 2026)

The original version of this document described a two-row activation protocol for AND, which is incorrect. AND requires a three-row majority (MAJ3) with one row pre-zeroed. The activation-sacrificial protocol and throughput estimates have been revised accordingly. SA-mediated RowCopy replaces the previously described asymmetric row survival mechanism for weight restoration. See [spice/CORRECTION_MAJ3.md](spice/CORRECTION_MAJ3.md) for details.

## Repository Structure

| Path | Contents |
|---|---|
| `CaSA_Technical_Report.md` | Full technical report with derivations, error analysis, and hardware validation plan |
| `simulators/` | Cycle-accurate throughput simulators (DDR4, DDR5, HBM2), BER accumulation simulator, fixed-point nonlinear validation, SiMRA data analysis |
| `figures/` | Publication figures and the generation script (`generate_figures.py`) |
| `kaggle/` | MNIST ternary inference demonstration (training and PIM-simulated inference) |
| `spice/` | SPICE circuit simulations (DRAM cell, sense amplifier, MAJ3 AND, popcount, RowCopy) using discrete and SKY130 models |

## Limitations

- No physical hardware validation. All results are from simulation.
- Prefill (prompt processing) is slow; the architecture is suited to autoregressive token generation, not long-context processing.
- Charge-sharing compatibility varies by manufacturer and die revision. Samsung DDR4 is incompatible. SK Hynix C-die (2018-2020) is the validated target.
- The architecture applies to inference only, not training.

## References

- Gao et al., "ComputeDRAM: In-Memory Compute Using Off-the-Shelf DRAMs," MICRO 2019.
- Olgun et al., "SiMRA: A Framework for Assessing the Reliability of In-DRAM Computation," DSN 2024.
- Ma et al., "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits," arXiv 2402.17764, 2024.
- Seshadri et al., "RowClone: Fast and Energy-Efficient In-DRAM Bulk Data Copy and Initialization," MICRO 2013.
- Samsung, "In-Memory Popcount Operation," US Patent (2014).
- Hajinazar et al., "SIMDRAM: A Framework for Bit-Serial SIMD Processing Using DRAM," ASPLOS 2021.

---

*This work was conducted by an independent researcher using AI-assisted analysis tools. All design decisions and claims were verified by the human author. All errors are the author's responsibility.*
