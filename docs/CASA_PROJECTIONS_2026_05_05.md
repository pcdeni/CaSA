# BitNet PIM throughput — measured & casa_sched-projected

Workload: BitNet b1.58-2B-4T full inference (30 layers × 7 BitLinears = 210 PIM projections per token), seq_len=5, FPGA-side popcount unless noted.

## Today (silicon)
| Setup | tok/s | Source |
|---|---|---|
| 1 DIMM, multi-bank "0,1,2,3", server backend | TBD | fresh run in flight |
| 1 DIMM, all-7 multi-bank | 0.034 | memory `bitnet_persistent_weights.md` (29.76 s/tok) |
| Single-bank baseline | 0.027 | memory (1.28× ratio) |

## Silicon ceiling (casa_sched, full 30-layer)
| Stack | tok/s | What's required |
|---|---|---|
| **1 DIMM, current arch** | **0.33** | Tier-A SW: persistent weights, batched matmul (orchestration only, no HDL) |
| **4 DIMMs, current arch** | **1.19** | Above + multi-DIMM calibration (DIMMs 1/2/3 sweep is the input) |
| 4 DIMMs + popcount accumulator | 1.28 | + HDL change (`popcount_accum.v` + `IMEM_ADDR_WIDTH 11→13`, both staged in tree) |
| 4 DIMMs + in-DRAM popcount | 1.41 | + DRAM-vendor circuit |
| 4 DIMMs + LISA | 1.44 | + DRAM-vendor cross-subarray path |
| **4 DIMMs + LISA + binary activations** | **11.49** | + model retrain to 1-bit acts |
| 1 DIMM + LISA + binary activations | 2.90 | (single-DIMM upper bound for context) |

## Honest gap analysis
- Today (~0.034) vs sim ceiling for 1 DIMM (0.33) = **~10× behind silicon**, attributed to host orchestration overhead. **Closable in software** (Tier-A: persistent weights).
- 1 DIMM ceiling (0.33) vs 4 DIMM ceiling (1.19) = **3.6× from multi-DIMM**, requires DIMM 1/2/3 calibration (sweep running).
- 4 DIMM ceiling (1.19) vs 4 DIMM + binary acts ceiling (11.49) = **9.7× from binary acts**, requires model retraining (out of scope for current demo).
- HDL changes (popcount accumulator + IMEM bump) buy ~7%. Not the bottleneck.

## Why it's bus-bound everywhere
Per casa_sched: bus utilization = 96-97% per DIMM in every configuration. The bus moves 1.6-1.9 GB per token (mostly write side: weight loads + activation bitplane writes). Cutting this = either persistent weights, in-DRAM popcount (eliminates result-row drain), or fewer bitplanes (binary acts).

## Why my Verilator HDL/scheduler work doesn't move the table much
- `seq_engine.v` — pushes PHY *command*-bus to 100% util on plain READ workloads. BitNet workload is bus-bound on *data volume*, not command emission rate. casa_sched-projected impact: ~7%, captured in the "popcount accumulator" row above.
- Parallel scheduler — improves command-emission density during MAJ3, which is 2.7% of total. Same: ~7% impact.
- Both are real, just not the bottleneck. They become important AFTER persistent weights + multi-DIMM are deployed.
