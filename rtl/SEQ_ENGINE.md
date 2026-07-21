# seq_engine — the command-issue accelerator (validated module, staged integration)

The other half of the FPGA performance story. Road B (`readback_engine`,
this dir) attacks the **readback** side of a PIM program; `seq_engine`
attacks the **command-issue** side — it emits DDR commands 4-per-fabric-
cycle straight from a state machine, bypassing fetch/decode/execute.

## Status: module DONE + Verilator-validated; pipeline integration STAGED

`seq_engine.v` (sha256 `a1b330e969b4e3b4…`, the exact source shipped
here) is complete (287 lines, two modes) and was proven bit-exact
against the stock DRAM-Bender pipeline in Verilator (`dual_top.v` +
`sim_compare.cpp`, a self-checking A/B in one binary). Re-running the A/B
needs Verilator + the DRB source tree (it lives on the build host, not
the silicon tower); the module here is byte-identical to the validated
one, so the numbers below stand for this source:

| workload | stock pipeline | seq_engine | speedup |
|---|---|---|---|
| 64 / 256 / 1024 back-to-back READs | 12 % PHY util | **100 %** | 8.0–8.6× |
| MAJ3 (doubleACT 0,0), 4-bank staggered | 9.4 % | **100 %** | 10.7× util |
| BitNet body chain (RowClone→broadcast→MAJ3) | 6.4 % | 17.3 % | 2.7×/bank |

Multiset (slot, bank, kind, row/col) compare vs the baseline slice: **0
diffs** at every size. Per-bank doubleACT dwell (`t_12`/`t_23`) is
bit-exact to the SiMRA helper and taken as runtime params, so any DIMM's
calibration works without a rebuild.

## Why it is STAGED, not flashed (the honest sequencing)

The measured per-program server profile at 47.5 s/tok is:

    total 6.0 ms = wcol 1.3 + exec 1.0 + recv 3.1 + pop 0.2 + other 0.3

`seq_engine` accelerates `exec` (and part of `wcol`) — the ~1–1.3 ms of
command issue. **`recv` (3.1 ms, the c2h readback) is the larger term,
and it is exactly what Road B collapses.** Integrating and flashing a
command-issue accelerator *before* the readback collapse lands in the
production server would be optimizing the second-largest term while the
largest is untouched — the "structural before stochastic" trap this
project is on record avoiding. Correct order:

1. **Road B into the production server** (readback collapse) — shifts the
   bottleneck off `recv`.
2. **seq_engine integration** — then `exec`/`wcol` become the top term
   and this module removes ~8× of it.

So the module ships now (it belongs in the go-to repo and its A/B is
done), and the bitstream integration waits for step 1 to make it the
binding constraint.

## The integration design (ready to build when step 1 lands)

Four changes, all small, all in `sources/hdl`; the risk is that the
`ddr_*` mux sits on the bus **every** command uses, so the Verilator
A/B must prove non-regression on normal instruction streams, not just
the seq path.

1. **Encoding** (`encoding.vh`): reserve one instruction class. The
   64-bit instruction already routes on the high bits
   (`DDR_OFFSET=63`, `BRANCH_OFFSET=62`, `INFO_OFFSET=61`,
   `MEM_OFFSET=60`, `BW_OFFSET=59`, `SR_OFFSET=56`). Use a currently
   unused combination (e.g. `SR_OFFSET=1` with a new `FU_CODE` value ≥ 2,
   or a dedicated `IS_SEQ` bit if the assembler is extended) to carry:
   `{op_mode, op_code, bank_mask, bank0_id, base_col|da_r_first,
   base_row|da_r_second, count, stride, da_t12, da_t23, bl4, ap}` —
   they fit in the 60 payload bits (da_t12/da_t23 are 6 bits each,
   rows 17, count/col 10, the rest ≤4).

2. **decode_stage.v**: recognize the IS_SEQ class and emit a one-cycle
   `seq_start` pulse plus the latched param bundle (a new output group
   parallel to `exe_uop`/`ddr_uop`). No change to any existing case.

3. **softmc_pipeline.v**: instantiate `seq_engine`, feed it
   `seq_start` + params from decode, and **mux the `ddr_*` bus**:
   `ddr_x = seq_busy ? seq_engine.ddr_x : execute_stage.ddr_x`. The
   widths already match by construction (seq_engine's port list mirrors
   execute_stage's).

4. **fetch_stage backpressure**: while `seq_busy`, stall fetch (hold
   `valid_out` low / don't advance) so the next instruction's commands
   cannot collide with the in-flight burst; and if the burst issues
   READs, add their count to the `read_size` / `read_seq_incoming`
   accounting the readback path gates on (the same `buffer_space`
   budget Road B fixed — a seq READ burst must debit it too).

Validation gate before any flash: extend `dual_top.v` to run a **mixed**
stream (normal DDR/exe instructions interleaved with IS_SEQ bursts)
through the *integrated* `softmc_pipeline` and assert the `ddr_*` trace
is bit-identical to the stock pipeline on the non-seq instructions and
matches the standalone seq_engine on the bursts. Only then is it a
safe addition to the (Road-B) bitstream.

## Files

- `seq_engine.v` — the module (both modes, DIMM-portable params).
- `seq_engine_top.v` / `dual_top.v` — Verilator wrappers (standalone and
  in-binary A/B against `softmc_pipeline_top`).
- `sim_compare.cpp` — the 3-part self-checking A/B (PLAIN bursts,
  doubleACT primitives, BitNet body chain).
- `build.sh` — Verilator build.
