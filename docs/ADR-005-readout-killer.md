# ADR-005 — The readout-killer decision: in-DRAM adder vs popcount_accum HDL

Status: DECIDED 2026-07-17 (Task T3). Supersedes the open question in
STOCKTAKE_2026_07_17.md Phase T3.

## Context

The ternary-LLM matvec's defining bottleneck is the readout wall: each
output lane reads K weight-AND-activation product rows to the host to
popcount them (exp0_readout_floor.py: 8× amplification vs the raw stream,
which disqualified the CaSA-shape pipeline vs a CPU). Two independent roads
kill it. This ADR decides which we use, for which claim.

## The two roads (both now measured)

### Road A — in-DRAM adder (Task M2, `popcount-indram-exe`)
- Mechanism: carry-save tree of the validated dual-track full adder
  (sum=MAJ5(a,b,c,¬carry,¬carry)=XOR3, carry=MAJ3) reduces K product rows
  to ceil(log2(K+1)) result rows, read out instead of all K.
- Measured (silicon, addendum 14): 98.6–99.7% lane-exact at K=8..64;
  readout reduction K/ceil(log2 K) = **213× at BitNet K=2560**; ~3K MAJ
  ops/tile; runs on the CURRENT bitstream, no HDL change.
- Error: ~0.05%/adder MAJ5 sum error, sub-linear in tree size (carry track
  ~exact). Non-accumulating (addendum 13). EXACT needs screen-harder/vote.

### Road B — popcount_accum HDL (`hdl/verilog/popcount_accum.v`)
- Mechanism: FPGA-side popcount aggregator behind
  `+define+POPCOUNT_ACCUM_MODE` in readback_engine.v — replaces per-read
  streaming with one 32-bit total drained at flush.
- Measured (Verilator, bitnet_bus_bound_hdl_staged): bit-EXACT vs software
  popcount, 5/5 cases incl. the 4096-input BitNet shape; readout 8 KiB → 4 B
  per matmul (**2048×**); kills the readback-FIFO back-pressure that
  throttles fetch. Requires the pop_count4.v 0xe-case fix (co-resident).
- Cost: needs a Vivado bitstream rebuild (now a proven one-variable flow,
  Task T1/addendum 12). FPGA logic, NOT in-DRAM computation.

## Decision

**Keep both — they answer different questions, and we publish the
comparison.**

1. **The MVDRAM reproduction result uses Road A (in-DRAM adder).** MVDRAM's
   claim is in-DRAM accumulation; computing the sum in FPGA logic would not
   be a faithful reproduction — a real DRAM-PIM ASIC has no such logic. Road
   A is the mechanism-faithful path and is what the reproduction writeup
   reports (addenda 13–14). Its error is the honest cost of commodity-DRAM
   MAJ5 and is bounded/screenable.

2. **The practical "fastest ternary LLM on our rig" uses Road B (HDL).**
   For an actually-deployed exact ternary LLM on THIS BCU1525, Road B is
   exact, larger (2048× vs 213×), and spends zero DRAM MAJ ops — strictly
   better when an FPGA IS present. It is a systems-engineering win, not a
   DRAM-PIM claim.

3. **Publish the comparison as a contribution.** "Two roads to kill the
   readout wall — one uses DRAM itself (portable to a real PIM substrate,
   faithful to MVDRAM), one uses FPGA logic (exact, rig-specific)" is a
   clearer systems story than either alone, and it is honest about what
   each does and does not demonstrate.

## Consequences / actions

- Road A is DONE and is the reproduction's accumulation path.
- Road B is UNBLOCKED (T1 proved the Vivado one-variable flow). Recommended
  as the NEXT bitstream build after any higher-priority one: single
  variable = `+POPCOUNT_ACCUM_MODE` + the pop_count4.v 0xe fix (both staged,
  Verilator-validated). Default-off keeps the current path bit-identical, so
  it is a safe additive build.
- The T4 full-model headline should report BOTH: the faithful in-DRAM
  per-token number (Road A, current bitstream) and, once Road B's bitstream
  lands, the exact FPGA-accelerated per-token number — labelled distinctly.
- seq_engine (100% command-bus utilization in Verilator) is a THIRD,
  orthogonal build (fetch-side, not readout-side) and stays deferred until
  a throughput measurement shows fetch is the binding constraint after
  Road A/B.

## Faithfulness note (for the writeup)

The reproduction's numbers must come from Road A. Road B numbers are
labelled "FPGA-accelerated (rig-specific), not part of the DRAM-PIM
reproduction." Never blend them into a single headline.
