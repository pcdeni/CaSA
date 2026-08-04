# Task #67 — server-side change spec: per-bank/subset handle residency

**Status: design spec (DESIGNED).** Written against the current canonical BitNet
server (`app/test_bitnet_server.cpp`, the `bitnet-proj-server` lineage with the
#65 bank-config plumbing). It specifies the per-bank/subset handle-residency
change that makes the #67 conveyor ping-pong legal; the line anchors below index
that server. A staged twin of this change is carried as
`app/experimental/conveyor/` (see its README).

## The one change (the main #67 server change, named in bank16 RESULT §"What #67 NEEDS")

> Decouple a LOAD handle's residency from **"all active banks"** to a **per-bank
> SUBSET**, so a bank leaving ACTIVE invalidates **only the handles that actually
> lived on it** — not the whole handle map. This is what makes the conveyor
> ping-pong (promote a freshly-staged group → ACTIVE, demote the finished group →
> STORAGE) legal without wiping the promoted group's freshly-loaded handles.

## Why it is needed (the exact coupling, with anchors)

Today a handle spans **every** active bank by construction:

- `process_load_weights` (`test_bitnet_server_bank16.cpp:2234`):
  - `const int N = (int)banks.size();` (`:2261`)
  - `size_t n_rounds = (n_units + N - 1) / N;` (`:2263`) — work units round-robin
    over **all N** banks.
  - reservation loop `for (int bk = 0; bk < N; bk++)` (`:2281`) checks pool space
    on **every** bank and sends ENOSPC if **any** overflows (`:2329-2341`).
  - handle populate: `h.per_round_backup_rows.assign(n_rounds, vector<uint32_t>(N,0))`
    (`:2351`), filled `h.per_round_backup_rows[round][bk] = backup_row` (`:2399`).
- `LoadedHandle` (`:1957`) stores rows as `[round][bank_idx]` (`:1965`), indexed
  **positionally against the global `banks` vector**.
- Both `process_matmul_handle` overloads (V2/serial `:2537`, MM3D `:3236`) iterate
  `for (int bk = 0; bk < N; bk++)` over that positional array
  (`:2856/:2919/:2996/:3307/:3389/:3639/:3703`).
- On a bank leaving ACTIVE, `handle_config_request`'s `CFG_SET_STATE` path calls
  `invalidate_resident_bank` (`:9855`) and then, because handles are cross-bank,
  **`handles.clear()`** — dropping *all* handles (`:9861-9862`). The code comment
  there (`:9860`) and the `invalidate_resident_bank` note (`:9760-9764`) already
  name this as the #67 work.

So promoting a staged group to ACTIVE and demoting the old group in the same
`CFG_SET_STATE` wipes the promoted group's handles too → the conveyor cannot
advance a layer without re-LOADing. Decoupling fixes exactly this.

## The change, precisely

### 1. Data model — record the subset on the handle
`struct LoadedHandle` (`:1957`): add
```cpp
std::vector<int> bank_ids;   // ordered bank-ids this handle's columns map to
                             // (a SUBSET of the active set). Column j of
                             // per_round_backup_rows[round] belongs to bank_ids[j].
```
Today's implicit invariant `bank_ids == [banks[0].bank_id .. banks[N-1].bank_id]`
becomes explicit and per-handle.

### 2. LOAD — reserve/populate over a subset, not all N
`process_load_weights` (`:2234`):
- Determine the handle's target subset `S` (ordered bank-ids). **Default =** all
  active banks (`banks` in ACTIVE state) ⇒ byte-identical to today. **New:**
  accept an optional subset from the request (see §5) — the #67 conveyor pins a
  layer's slices to that layer's compute GROUP.
- `int M = S.size(); n_rounds = ceil(n_units / M);` (was `/N`, `:2263`).
- The reservation loop (`:2281`), the `assign` (`:2351`), the populate (`:2360-2399`),
  and the ENOSPC check (`:2329`) iterate **`S`**, not `0..N`. Resolve each
  `bank_id ∈ S` to its `BankConfig` via a small `bank_id → index` map over `banks`.
- Set `h.bank_ids = S`.

### 3. MATMUL — iterate the handle's own subset
Both `process_matmul_handle` overloads (`:2537`, `:3236`): replace the
`for (int bk = 0; bk < N; bk++)` bank loops that touch
`h.per_round_backup_rows[round][bk]` with iteration over `h.bank_ids` (column `j`
→ `BankConfig& = banks[idx_of(h.bank_ids[j])]`). `active_in_round` and the program
emission (one MAJ3 body per subset bank per round) key off `h.bank_ids.size()`,
not `banks.size()`. The verify/refresh paths (`:3374-3391`, `:3637-3703`) use the
same `h.bank_ids` mapping.

### 4. Invalidation — per-handle by membership, drop the coarse clear
`invalidate_resident_bank(int bank_id, handles)` (`:9765`): in addition to erasing
`g_resident_rows` entries for `bank_id` (already per-bank, `:9768-9771`, keep),
**erase from `handles` every handle with `bank_id ∈ h.bank_ids`** and return the
count. Then in `CFG_SET_STATE` (`:9839`): **delete the `handles.clear()` at
`:9861`** — the per-handle erasure in `invalidate_resident_bank` now does the
surgical drop. Idle↔idle transitions still call neither path (metadata-only, no
stop-the-world) — unchanged. `STAGING→ACTIVE` still isn't "leaving ACTIVE"
(`leaving_active`, `:9853`) → still never invalidates — the property the conveyor
relies on, now safe because the promoted group's handles were pinned to that group
and survive the old group's demotion.
`invalidate_resident_all` (`:9753`) + `CFG_RECONFIG` (`:9828`) stay as-is (a bank-
set change legitimately invalidates everything).

### 5. Wire / compatibility (default-inert, like #65)
- New **optional** LOAD request field: a trailing `[u32 n_subset][i32 bank_id…]`
  after the existing header (`process_load_weights` header parse `:2243-2259`).
  Absent ⇒ subset = all active banks ⇒ **behaviour-identical to production**.
- Host side: `pim_conveyor.py` already assigns each layer's slices to a compute
  group (`TokenStep.moves` carry `dst_bank`); the LOAD call passes that group's
  bank-ids as the subset. `pim_grid_config.py` carries the per-bank `state`/`dimm`
  so the orchestrator knows which banks form each group.
- Keep the change behind a flag (e.g. `PIM_HANDLE_SUBSET=1`) for the A/B, default
  OFF, exactly the #65 default-inert pattern (twin binary; prod `.o` untouched).

### 6. Prerequisite already-present plumbing (no new work)
- Per-bank `state` + `win_start/win_end` on `BankConfig` (`:1519-1533`) — done (#65).
- `build_banks` single re-derivation path (`:9739`) — done; staging a bank just
  means it exists in `banks` with `STAGING` state and gets LOADed via its subset.
- `g_resident_rows` keyed `{bank_id,row}` (`:1763`) — already per-bank; no change.

### 7. Companion follow-ups (inventoried, not blocking)
- **#65a** pack4 fixed reg-slots wired for N=4 (`:1136-1169` in production
  numbering) — safe serial fallback for other N; generalize for throughput later.
- **#65b** geometry consts still literal (subarray 640, predecoder block 1024,
  pool-scan cap) — make `PIM_SUBARRAY_ROWS`/`PIM_PREDECODER_BLOCK`/`PIM_POOL_MAX_ROWS`.
- **Calib-by-copy**: the bank16 regate found the twin needs a per-bank calib for
  every built bank (no calib-by-copy in `build_one_bank`→`read_calib`). Staging
  into bank *b* needs bank *b*'s calib present — carried by `GridConfig` calib
  paths; a startup calib-copy helper would remove the per-bank fixture requirement.

## Gate (when the tree is free)
- **Command rail:** a `CFG_SET_STATE` that demotes group A and promotes group B
  keeps B's handles (assert `handles` count after = B's handle count, not 0) and
  drops only A's (trace-equiv per-bank).
- **Value rail:** `v2_oracle` 512/512 @ d2560 on the promoted group *after* a
  demote, same process (proves the surviving handles still compute bit-exact) —
  the `bus_bound_ladder` §5 conveyor gate ("oracle after each staged move").
- **Default-inert:** `PIM_HANDLE_SUBSET=0` ⇒ NS6 6/6 512/512 @ 0.0182 s/sub +
  `PIM_COUNTER_GATE PASS`, byte-identical to the promoted prod.
- Numerics-gate FIRST; same-process A/B only; no `timeout` wrappers.
