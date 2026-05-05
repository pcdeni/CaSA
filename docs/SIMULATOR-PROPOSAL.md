# Observable PIM simulator — design proposal

**Goal:** debug PIM correctness bugs (like today's dual-subarray B failure)
without 3-minute hardware round-trips, with full visibility into per-cycle
DDR commands and DRAM state, with **deterministic** SiMRA semantics so
cell-flip noise doesn't mask logic bugs.

## Two layers of fidelity

### Layer 1: pure-C++ behavioral model (faster to build, less rigorous)

```
                      ┌──────────────────────────────┐
                      │ pim-server-sim               │
                      │ (drop-in for bitnet-proj-srv)│
        SoftMC binary │  ┌──────────────────────┐   │  popcount/result
       ────────────► │   │ SoftMC inst decoder  │   │ ────────────►
       (h2c stream)  │   └──────────┬───────────┘   │  (c2h stream)
                      │              │                │
                      │   ┌──────────▼─────────────┐ │
                      │   │ DDR4 + SiMRA model     │ │
                      │   │ - per-bank state       │ │
                      │   │ - 16-row open buffer   │ │
                      │   │ - doubleACT semantics: │ │
                      │   │   30/1 → RowClone      │ │
                      │   │   10/2 → broadcast 16  │ │
                      │   │   0/0  → MAJ3 vote     │ │
                      │   │ - feeds calib_dimm0.txt│ │
                      │   └────────────────────────┘ │
                      └──────────────────────────────┘
```

bitnet-proj-server unchanged; replace `BoardInterface` (which talks to
`/dev/xdma*`) with a `SimInterface` selected via env var
`PIM_BACKEND=sim:tcp:localhost:9000` or similar.

The C++ model implements:
- Per-bank state: `open_row` (or NONE), `row_buffer[8192]`
- `ACT(bank, row)`: `row_buffer = stored[bank][row]`; `open_row = row`
- `PRE(bank)`: `stored[bank][open_row] = row_buffer`; `open_row = NONE`
- `WRITE(bank, col, wdata)`: `row_buffer[col*64..(col+1)*64] = wdata`
- `READ(bank, col)`: stream `row_buffer[col*64..(col+1)*64]` to c2h
- **doubleACT pattern detector** (peeks at the SoftMC instruction stream
  for `ACT-...-PRE-...-ACT` with specific NOP spacings):
  - `t_12=30, t_23=1` → RowClone R1→R2 = `stored[bank][R2] = stored[bank][R1]`
  - `t_12=10, t_23=2` → for the calibrated subarray containing R1 and R2,
    look up its open_rows from calib_dimm0.txt; broadcast R1 → all 16
  - `t_12=0, t_23=0` → MAJ3 vote: bit-by-bit majority across the 16 open_rows
- Any unknown timing → fail with "uncalibrated doubleACT pattern at PC X"

**Cost:** ~1 day to build. Model is ~500 lines C++. Hooks into the
existing platform.cpp via a `--backend sim` flag.

### Layer 2: Verilator + behavioral DDR (rigorous, slower to build)

If we want to also catch **softmc_pipeline** bugs (not just DDR-side
bugs), wrap the existing `softmc_pipeline_top.v` (which I built in
`dram_bender_sim/phy_util/`) and route its `ddr_*` outputs to a
Verilog/SystemC behavioral DDR model:

```
  Host ──/dev/xdma stub──► Verilator instance
                                  │
                          softmc_pipeline_top.v (real RTL)
                                  │
                         ddr_act, ddr_read, ddr_write, ddr_pre,
                         ddr_bank, ddr_col, ddr_row, ddr_wdata
                                  │
                          ddr_simra_model.v (NEW behavioral model)
                                  │
                         rd_data ──────────────► back to readback_engine
```

This catches both pipeline-side and DDR-side bugs. ~2 days to build.

## What this would have caught today

For the dual-subarray B failure: I would have seen, deterministically:
- Round 0: `bank=0 ACT row=38641` (subarray-0 backup). Then RowClone
  to row=38424 (subarray-0 Rfirst). Then broadcast to subarray-0's
  16 open rows. Then MAJ3. Read result. **Verify** model's
  output bits match the expected MAJ3 of (loaded weights + activations).
- Round 1: `bank=0 ACT row=48001` (subarray-1 backup). Then RowClone
  to row=48532 (subarray-1 Rfirst). Then broadcast to subarray-1's
  16 open rows. Then MAJ3. **Compare** to expected.

If round 1's output differs from expected, the model exposes WHICH STEP
broke the data — RowClone? Broadcast? MAJ3? Result row? With real
silicon, I just see "wrong final token, somewhere in 30 layers".

## Recommendation

Build **Layer 1 first** (pure C++, ~1 day). It's enough to debug the
dual-subarray B issue and any future scheduler bugs. Layer 2 only if
we suspect a pipeline-internal bug (rare; pipeline is well-tested).

If you OK the design, I start Layer 1 right after the current 6/7
silicon run finishes — that gives us both (a) a working silicon demo
on 6/7 + (b) the simulator infrastructure to fix dual-subarray and land
A/C correctly. From then on, every PIM-correctness change has a
deterministic regression test.
