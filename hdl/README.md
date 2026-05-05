# HDL changes — staged for next bitstream rebuild

These files are the SiMRA-DRAM (DRAM-Bender) modifications made on
2026-05-05. They are **Verilator-validated** but **not yet synthesized**
into a Vivado bitstream — until they are, they have no effect on the
deployed FPGA.

## Files

### Bitstream-side (require Vivado regen)

| File | Change | Why |
|---|---|---|
| `parameters.vh`   | `IMEM_ADDR_WIDTH` 11 → 13 (8192 instr/program, was 2048) | Lets one execute carry many primitives — cuts host-FPGA round-trips. **Also** requires regen of `instr_blk_mem` BRAM IP at depth=8192. |
| `pop_count4.v`    | Added missing `4'b1110: out_r=3'd3` case | Silent under-count on nibble 0xe. Doesn't affect today's host-side popcount path, but load-bearing for `popcount_accum.v` below. |
| `popcount_accum.v` | NEW — accumulates 16b popcount stream into one 32b sum, drains on flush | Cuts result-row c2h traffic from 8 KB/matmul to 4 B/matmul. |
| `readback_engine.v`| Wired `popcount_accum` behind `\`ifdef POPCOUNT_ACCUM_MODE` | Default-off keeps current behaviour bit-identical. |
| `seq_engine.v`    | NEW — parameterised opcode emits 4-cmd-per-cycle DDR command bursts | Pushes PHY *command*-bus utilization to 100 % on plain READ; not in the BitNet hot path's critical chain (data-volume bound, not command-bus bound) — but kept for future use cases. |

### Host-side helpers (require recompile only)

| File | Change |
|---|---|
| `board.h`     | `SEND_BUF_SIZE = 32 × 8192` (was `× 2048`) — host AXI staging buffer matches the post-bump IMEM. |
| `platform.h`  | `INSTR_BUF_SIZE = 32 × 8192`; new `BITSTREAM_IMEM_INSTS = 2048` constant for the **deployed** bitstream's IMEM cap. |
| `platform.cpp`| Runtime check in `execute()`: refuses with a clear warning when a program exceeds `BITSTREAM_IMEM_INSTS`. Prevents the silent FPGA-side truncation → SoftMC deadlock that was happening before this check (e.g. when `PIM_INLINE_BITPLANES > 1` runs into the IMEM cap). |

## Bring-up checklist for the Vivado machine

1. `cp parameters.vh` into the DRAM-Bender source tree, regenerate
   `instr_blk_mem.xci` IP at depth=8192, ADDR=13.
2. Build the bitstream. Flash to BCU1525.
3. After flash, update `BITSTREAM_IMEM_INSTS` in `platform.h` to `8192`.
4. Optional: rebuild with `+define+POPCOUNT_ACCUM_MODE` to enable the
   accumulator path.

## Verilator validation

See `../sim/` for the harnesses that validated each module against the
unmodified DRAM-Bender baseline:

- `sim/pop_count4/`         — exhaustive 16-input correctness test for the bug fix
- `sim/popcount_accum/`     — unit test + end-to-end with the actual popcount tree
- `sim/phy_util/`           — PHY command-bus utilization measurement
- `sim/seq_engine/`         — A/B against `softmc_pipeline` baseline (bit-exact multiset compare)
- `sim/parallel_sched/`     — host-side bank-parallel scheduler bit-exact per-bank vs SiMRA template
