# Conveyor server twin (#67) — staged, review-gated

A **review-gated experimental variant** of the canonical BitNet server
(`app/test_bitnet_server.cpp`). It is carried as a **patch** rather than a full
source copy so the delta is reviewable and can never drift from the canonical
lineage. The design behind it is `docs/CONVEYOR_DESIGN.md` and
`docs/CONVEYOR_SERVER_CHANGE.md`.

## What the twin changes

Exactly one thing: it decouples a LOAD handle's residency from *"all active
banks"* to a per-bank **subset** (`LoadedHandle.bank_ids`). A `CFG_SET_STATE`
that demotes a bank then invalidates **only** the handles that lived on that bank
— instead of the coarse `handles.clear()` that wipes every handle. That surgical
invalidation is what makes the conveyor promote/demote ping-pong legal (promote a
freshly-staged bank group to ACTIVE, demote the finished group to STORAGE, in the
same request, without dropping the promoted group's just-loaded handles).

**Default-inert:** with `PIM_HANDLE_SUBSET` unset and no `MAGIC_CONFIG` traffic, a
handle spans the full active set and behaviour is byte-identical to the canonical
server. The subset is an optional trailing field on the LOAD request.

## Reconstructing and building the twin

The patch applies to the published canonical server with zero offset:

```sh
cd app
patch -o test_bitnet_server_conveyor.cpp \
      test_bitnet_server.cpp < experimental/conveyor/test_bitnet_server_conveyor.diff
# build under a distinct name (production bitnet-proj-server untouched):
g++ -g -std=c++17 -pthread -O3 -mavx2 -I../../api -I../../../boost-lib \
    -c test_bitnet_server_conveyor.cpp -o test_bitnet_server_conveyor.o
make bitnet-proj-server-conveyor   # add the analogous link rule, or link test_bitnet_server_conveyor.o + $(SHARED_OBJS)
```

The patch reproduces the reviewed twin byte-for-byte (the delta is ~60 lines of
new code plus comments; the `bank_ids` field, subset-aware LOAD/MATMUL/invalidate
paths, and the `PIM_HANDLE_SUBSET` gate).

## Gates

- **`test_pim_conveyor.py`** — card-free dry test of the host scheduler
  (`python/pim_conveyor.py`): 9/9 assertions (degenerate BitNet, valid 13B
  schedule, the three named residency properties, the bandwidth crossover, the
  `CFG_SET_STATE` wire round-trip). No card required.
- **`conveyor_gate.py`** — on-card gate (needs the FPGA + `python/v2_oracle.py`
  CPU oracle + `python/pim_linear.py`): ping-pongs two bank groups on one DIMM and
  asserts, per cycle, that a demote invalidates exactly one resident handle, the
  surviving group still serves byte-exact with no reload, a freshly promoted group
  serves byte-exact, and a demoted handle is ENOENT (not stale-served). Run the
  numerics gate first; same-process A/B only; no `timeout` wrappers.
