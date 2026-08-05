# Task #67 — the conveyor / schedule-driven prefetch scheduler (DESIGN, 2026-08-04)

**What this is.** The decode token loop walks the transformer layers in a fixed,
known order, so the *future working set is deterministic*. This design turns that
determinism into a prefetch schedule: while layer `L` computes on its ACTIVE
compute banks, idle banks preload layer `L+d`'s weight slices. Built ON the #65
grid-config plumbing (`pim_grid_config.py`: `BankState`/`DimmRole`/`GridConfig` +
the `MAGIC_CONFIG` codec, LIVE in production as a default-inert superset since
`bank16_config_2026_08_04`). Prototype + dry tests: §5.

Scope: **design + a card-free host-side prototype** (no silicon dependency). Every
quantitative claim carries an evidence tag: `MEASURED` · `SIM` · `ARITHMETIC` ·
`RECORDED-IDEA`.

---

## 0. The cost model — RECORDED, user-ruled (verbatim), used as-is

STOCKTAKE PLACEMENT LAW v2 (user 2026-08-02) / LEVERS #67 / `bus_bound_ladder`
rung a′. Cost is measured on the **binding (compute-DIMM) DDR bus**:

| tier | move | binding-bus cost | when |
|---|---|---|---|
| **free** | **in-subarray RowClone** (src+dst share one 640-row segment) | **0** (in-DRAM, ≥99.98%, SiMRA Multi-RowCopy) | slice already co-resident in the destination segment |
| **1×** | **cross-DIMM prefetch from storage** (D1/D3 Micron) | **1× write** on the compute bus; the **read rides the otherwise-idle storage channel**, hidden by read-ahead | THE conveyor tier — parked copy lives on a storage DIMM |
| **2×** | **same-DIMM cross-bank** staging | **2× (read + write)** — the **WORST tier** | AVOIDED by construction |

The design law that follows: **keep every parked/next-stage copy on a storage
DIMM (D1/D3), never on another compute bank.** Then every conveyor move is the 1×
tier (or free), and the 2× tier never occurs. The prototype's planner *raises* if
it is ever asked to emit a 2× move (`ScheduleError`), so the law is enforced
mechanically, not by convention.

---

## 1. The state machine — per bank, over the LIVE #65 config states

Each bank carries a runtime-mutable `BankState` (already in production,
`test_bitnet_server_bank16.cpp:1510`):

```
        stage-in (cross-DIMM 1× write; read on idle storage ch)
   STORAGE ───────────────────────────────────────────────▶ STAGING
      ▲                                                         │
      │ demote (leaving-ACTIVE: invalidates THIS bank only)     │ promote
      │                                                         ▼   (STAGING→ACTIVE:
   ACTIVE ◀──────────────────────────────────────────────── ACTIVE   NOT leaving-active,
      (serves matmul for its layer)                                    never invalidates)
```

- **ACTIVE** — in the compute working set; serves this layer's matmul now.
- **STAGING** — idle; being preloaded with a future layer's weight slices.
- **STORAGE** — holds a parked slice (or is the reservoir a slice streams *from*
  on the storage DIMM); not in the active set.
- **FREE** — empty; available for allocation.

The conveyor is a **ping-pong across `depth+1` bank GROUPS**. Partition the
compute-bank pool into `G = depth+1` groups of `banks_per_layer` banks each.
Layer `L` runs on group `L mod G`; while it computes, the group that will host
layer `L+depth` (currently idle) is filled from storage.

---

## 2. The schedule — which banks flip state when

For a per-token walk of layers `0..N-1` with read-ahead depth `d` and groups
`G=d+1`, at the **entry boundary of layer L** (which is *between* server
requests — the only point the #65 rule lets config change):

1. **Promote** group `g = L mod G` from **STAGING → ACTIVE**. It was filled
   during layers `L-d .. L-1`. STAGING→ACTIVE is *not* "leaving ACTIVE" → **no
   invalidation** (this is why staged handles survive promotion — the crux the
   server change must preserve, §4).
2. **Demote** the group that just finished, `g' = (L-1) mod G`, from **ACTIVE →
   STORAGE**. Leaving ACTIVE invalidates *that group's* resident scratch (and,
   under the decoupled server change, *only* handles that lived on it).
3. **Stage** layer `L+d` into the now-idle group `(L+d) mod G` (which equals `g'`
   for `G=d+1`): issue **STORAGE/FREE → STAGING** (idle↔idle, **metadata-only,
   no stop-the-world**), then the cross-DIMM 1× moves that fill it.

So per layer boundary the emitted `MAGIC_CONFIG`/`CFG_SET_STATE` payload is a
small set: `{promote group g → ACTIVE, demote group g' → STORAGE, mark stage
target → STAGING}`. The prototype emits exactly these (§5); they round-trip
through the #65 `encode_set_state` codec (dry test 8).

**Steady state (decode):** weights do not change token-to-token. If the whole
model fits the resident compute capacity, the conveyor stages *once* (prefill /
first token) and then runs **degenerate** — zero per-token staging, pure static
#65 residency. The conveyor only moves data per-token when the *working set
exceeds resident capacity* and layers must be evicted and reloaded each token
(the streaming regime priced in §3).

---

## 3. Read-ahead depth — how far ahead to stage to hide each tier

A stage of layer `L`'s bytes over the mover hides under compute iff the
**storage-channel demand during a compute window ≤ what the channel can move**:

```
ratio(L) = storage_demand(L) / (BW_mover · n_channels · T_compute(L))
storage_demand(L) = weight_bytes(L)   [read-ahead]
                  + KV_read_bytes(L)   [fabric-attention stream, if co-channel]
```

`BW_mover = 8.5 GB/s` per storage channel (MEASURED, casa_sched effective-BW,
`bus_bound_ladder` §a′); D1+D3 = 2 channels = 17 GB/s.

**Depth rule.** For uniform layers the per-layer ratio equals the per-token
average, so:
- `ratio ≤ 1` ⇒ **bandwidth suffices**; read-ahead **depth 1** hides the
  per-layer burst (stage `L+1` during `L`). Deeper read-ahead only smooths
  bursty (non-uniform) layers — it adds **no aggregate bandwidth**.
- `ratio > 1` ⇒ **BANDWIDTH-BOUND**; read-ahead cannot help (staging further
  ahead creates no bandwidth). The fix is **more channels (D1+D3) or a
  weight-split** — never more depth. The prototype detects this and flags it
  rather than silently inflating depth.

**Reproduces the ladder crossover exactly** (prototype output, ARITHMETIC on
MEASURED BW + SIM per-layer compute):

| case | weight MB/layer | +KV MB/layer | T_compute/layer | 1-channel ratio | 2-channel | verdict |
|---|---:|---:|---:|---:|---:|---|
| Llama2-7B q4 | 101.2 | 0 | 16.9 ms | **0.70** | — | hides, depth 1 (ladder +37%) |
| Llama2-13B q4 | 158.6 | 0 | 21.2 ms | **0.88** | — | hides, depth 1 (ladder +11%) |
| **Llama2-13B q4 + 4k KV** | 158.6 | 83.9 | 21.2 ms | **1.35 → INFEASIBLE** | **0.67 → hides** | **needs D1+D3 (17 GB/s) or weight-split** |
| BitNet-2B (native) | 17.4 | 0 | fits resident | **depth 0** | — | degenerate (static #65) |

The 13B+4kKV row is the ladder's named crossover: one storage channel is
bandwidth-bound (1.35 > 1); the two storage DIMMs D1+D3 (17 GB/s) restore the
hide (0.67). This is a **hard dry-test assertion**, not a report line (test 6).

**Which tier each depth hides:**
- **free (RowClone):** zero cost — no read-ahead needed at all.
- **1× cross-DIMM (the conveyor tier):** the *write* side (1× on the compute bus)
  must fit the compute window; the *read* side rides the idle storage channel and
  is hidden by **depth-1** read-ahead as long as `ratio ≤ 1`. This is the whole
  point of parking on storage: the expensive half (read) is on a bus that is
  otherwise idle w.r.t. compute.
- **2× same-DIMM:** never planned (design law §0).

---

## 4. Invalidation choreography against the #65 rule

The #65 rule (production, `handle_config_request`): config changes apply
*between* complete requests; **leaving ACTIVE invalidates that bank's resident
scratch**; **idle↔idle transitions are metadata-only, no stop-the-world**;
`STAGING→ACTIVE` is not "leaving ACTIVE" so it **never invalidates**.

The conveyor's ping-pong is *legal by construction* under this rule **only once
the main #67 server change lands** (CONVEYOR_SERVER_CHANGE.md). The tension:

- The staging fills group `g'` (freshly loaded, holds `L+d`'s slices). At the
  boundary we **promote `g'`→ACTIVE** (safe: no invalidation) and **demote the
  old ACTIVE group**→STORAGE (invalidates the old group).
- **Today's blocker:** a LOAD handle reserves `n_rounds = ceil(n_units/N)` rows
  on **every** active bank (`:2263`), so a handle spans all ACTIVE banks. When
  the old group leaves ACTIVE, the server conservatively drops the **whole
  cross-bank handle map** (`invalidate_resident_bank` note + the SET_STATE
  handler's `handles.clear()` at `:9861`) — which would wipe the freshly-promoted
  group's handles too. The ping-pong is illegal until handle residency is
  decoupled to a **per-bank subset** (each handle records the banks it actually
  lives on; leaving-ACTIVE drops only handles that used that bank). **That
  decoupling is the main #67 server change.**

**Choreography the scheduler emits (safe under the decoupled model):**
1. Stage `L+d` into idle group via idle↔idle `STAGING` + cross-DIMM 1× moves —
   the old ACTIVE group is untouched, its handles live.
2. At the layer boundary: `promote(g'→ACTIVE)` first (no invalidation), then
   `demote(old→STORAGE)` (invalidates only the old group's handles — which
   belong to a layer we are done computing).
3. Byte-verify each staged slice (RowClone dst-readback / read-write-in) as the
   value-rail gate before the promoted group serves (per the `bus_bound_ladder`
   §5 conveyor gate: "oracle after each staged move").

The prototype's check (5) asserts no step ever sets a bank ACTIVE and STAGING in
the same transition, and counts the ACTIVE→STORAGE demotions (each of which, post
server-change, invalidates only its own bank).

---

## 5. The prototype (card-free) + dry-test result

- **`python/pim_conveyor.py`** (additive; sibling to `python/pim_grid_config.py`).
  Pure data + arithmetic, no torch/FPGA deps. Consumes the #65 `GridConfig`.
  Provides:
  - `ModelMap` / `LayerMap` / `Slice` — a model's layer/slice map, sized in whole
    8 KiB rows (`_proj_rows` from the flagship shapes).
  - `ConveyorScheduler` — `required_depth()` (the §3 rule) and `plan_token()`,
    emitting `TokenStep`s each with `active_banks`, `staging_banks`, the
    cost-tiered `moves`, and the `MAGIC_CONFIG` `set_state` payload.
  - `validate()` — the assertions below (raises `ScheduleError` on violation).
  - Model factories `bitnet_2b()`, `llama2_13b_q4()`, `llama2_13b_q4_kv4k()`,
    `llama2_7b_q4()` (dims = `flagship_pricing.py` SHAPES).
- **`app/experimental/conveyor/test_pim_conveyor.py`** — 9 dry tests (card-free).

**Dry-test verdict — 9/9 PASS (card-free):**

| # | assertion | result |
|---|---|---|
| 1 | BitNet-2B degenerate (fits resident → depth 0, 0 staged bytes) | PASS |
| 2 | 13B-q4 conveyor schedule valid — all three named properties present | PASS |
| 3 | (1) **no bank computes while staging-in** (ACTIVE ∩ STAGING = ∅ every layer) | PASS |
| 4 | (2) **every slice resident before its layer starts** (staged in an earlier layer; L0 warmed at prefill) | PASS |
| 5 | (3) **read-ahead depth respected** (1 ≤ lead ≤ depth) & **no 2× tier** | PASS |
| 6 | 13B+4kKV **crossover**: 1 channel INFEASIBLE (bandwidth-bound), 2 channels feasible | PASS |
| 7 | stage/compute ratios reproduce the ladder (7B 0.73, 13B 0.90) | PASS |
| 8 | emitted `CFG_SET_STATE` round-trips through the #65 wire codec | PASS |
| 9 | all transitions legal under the #65 invalidation rule | PASS |

The three properties the task names are hard assertions (3,4,5), and the
crossover the task names is a hard assertion (6). BitNet degenerates correctly
(1); 13B exercises the full streaming conveyor.

---

## 6. Scope honesty (self-audit)

- **The 7 tracked flagships at q4 do not *strictly* need the weight conveyor.**
  Their weights fit one 8 GiB compute rank (13B q4 = 6.34 GB, the largest;
  `capacity.py`), so the DEGENERATE placement — weights resident on D2, KV parked
  on storage D1/D3 — needs no per-token weight staging (same static-#65 answer as
  BitNet). The conveyor's streaming regime (modeled here for 13B) is the
  ALTERNATIVE placement (KV resident on the compute rank → weights stream) and the
  general mechanism. It becomes **strictly load-bearing** at fp16 weights,
  >13B models, or multi-model residency — none of which are in the current
  seven-flagship set at q4.
- **This is a HOST-orchestration + server-change design; it does not move the
  native token floor.** BitNet/Bonsai fit one DIMM (degenerate). The conveyor is
  the *mainstream-lane enabler* (LEVERS #67, e2e-audit: mainstream four are
  command-stream-bound and "need #67 conveyor + #70 KV to run at all").
- **What is unbuilt / gated:** the server-side per-bank-subset handle residency
  (CONVEYOR_SERVER_CHANGE.md; a staged twin is in `app/experimental/conveyor/`);
  the actual RowClone/read-write-in mover execution (composes with #64 paced-copy);
  cross-DIMM orchestration across `PimServer` processes (one server = one DIMM
  today). The prototype models the decoupled target and the schedule; it does not
  move silicon data.
