# `scheduler/` — discrete-event scheduler for charge-sharing DRAM PIM

`casa_sched.c` is a single-translation-unit C program that models a
DDR4 channel running charge-sharing PIM primitives, with bus
contention tracked explicitly. Given a configuration (number of
DIMMs, banks per DIMM, primitive set, model dimensions) it estimates
**tokens per second** and per-resource utilization for that
configuration.

## Build and run

```bash
cc -O2 -o casa_sched casa_sched.c
./casa_sched              # prints the projection ladder for several configs
```

No external dependencies. The output is the per-config table that
populates the throughput projections in `docs/METHODOLOGY.md`.

## What it models (and what it doesn't)

**Models:**
- DDR4 timing parameters (tRCD, tRP, tRAS, tRC, tRRD, tFAW, tCCD,
  tBurst, tWR, tREFI, tRFC) at standard JEDEC values.
- Per-bank busy windows, per-bus busy windows, tFAW windowing.
- Charge-sharing latencies for MAJ3 (`doubleACT(0,0)`) and RowClone
  (`doubleACT(10,2)`) calibrated against measured DIMM 0 values.
- Bank-group-parallel bus when enabled.
- BitNet b1.58-2B-4T architecture (30 layers, d_model=2560,
  d_ff=6912, 20 attention heads, 5 KV heads, 128K vocab) for the
  per-token throughput estimate.
- Optional configurations: in-DRAM popcount circuit, LISA
  cross-subarray bus, binary-activation models — each toggled by a
  flag at the top of `main()`.

**Does NOT model:**
- PCIe latency, kernel driver overhead, host-side Python overhead.
  The scheduler assumes the FPGA is fed instructions at zero cost.
  Real measured throughput will be lower than the bus-bound
  projection until those overheads are engineered out.
- Cell-stability noise (small, but non-zero — see
  `docs/METHODOLOGY.md`).
- Tail effects of refresh interference with PIM ops at the very
  end of a layer.

## Honest framing

The numbers from this scheduler are **realistic ceilings** for the
specific configurations they describe, given the standard DDR4
timing rules and our measured charge-sharing latencies. They are not
peak / marketing numbers; bus utilization is tracked explicitly and
reported alongside throughput so you can see when the wall is the
bus and not the compute.

When the scheduler says "75 % bus at 60 tok/s", that is the literal
claim: ~25 % bus headroom remains at that throughput.
