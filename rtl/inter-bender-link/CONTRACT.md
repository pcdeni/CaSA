# Inter-bender fabric link — CONTRACT (Task #76, 2026-08-04)

**Status:** RTL authored + Verilator-gated, LOCAL. No box write, no card, no
git. This is a **capability** build (LEVERS #76 ruling (3): "build it as
CAPABILITY per L24"), **default-inert** like every twin — with no route
enabled the QUAD is bit-for-bit the current star topology.

**One-line architecture.** A routing stage (`inter_bender_link`) that
**snoops** each core's outbound c2h stream (read-only) and **injects** the
copied stream into a **separate, explicitly-tagged** `link_rx` ingress port on
any peer core, crossing the two independent MIG UI clocks through **one
`xpm_fifo_async` per ordered (src,dst) pair**. The router **never sits in
either host datapath** — it is a pure observer of c2h and a pure producer on new
ports — so both host paths (h2c instructions, c2h readback/ACK) are bit-identical
whether the link is idle *or* active. That property is **structural, not
tested-into-existence**.

---

## 0. Where the link attaches (verified against the RTL)

Topology today (`bcu1525_quad_top.v`, verified): 1 PCIe endpoint → 1 XDMA (4×H2C
+ 4×C2H, `axi_clk`) → per-DIMM `axis_clock_converter` pair → `softmc_core[N]`
(each on its own MIG `core_ui_clk[N]`). Inside a core: `frontend` consumes
`m_axis_h2c_tdata_0` (instructions); `readback_engine` produces
`s_axis_c2h_tdata_0` (readback + the 32-B completion-ACK trailer). Zero wires
between cores (MM §1.0). `XDMA_AXI_DATA_WIDTH = 256` for **both** h2c and c2h
(`parameters.vh`) → a c2h beat maps 1:1 onto an h2c-shaped beat, no reframing.

The link attaches at the **core (UI-clock) side** of the existing clock
converters — the `core_c2h_*` / `core_h2c_*` wires in `bcu1525_quad_top.v` — so
the cross-bender path stays at DDR line rate and never bounces to `axi_clk`
(the whole prize per LEVERS #76 / SCOPE §3.3: a continuous KV/Q stream into
fabric attention in the *destination* core's UI domain).

```
        core s (ui_clk[s])                              core d (ui_clk[d])
  readback_engine ─ core_c2h[s] ─┬─────────────► c2h_cc[s] ─► XDMA (host)   [UNTOUCHED]
                                 │ snoop (read-only: tvalid&tready)
                                 ▼
                    ┌── xpm_fifo_async f[s][d] ──┐   wr=ui_clk[s]  rd=ui_clk[d]
                    │  word = {tlast,tkeep,tdata} │
                    └────────────┬───────────────┘
                                 ▼  (per-dest round-robin arbiter)
                        link_rx[d] ──► [tagged data consumer in core d]   [NEW port]

  XDMA (host) ─► h2c_cc[d] ─► core_h2c[d] ─► frontend[d] (instructions)   [UNTOUCHED]
```

---

## 1. TAP POINT — outbound c2h snoop (source side)

- **Signal:** `core_c2h_tvalid[s]`, `core_c2h_tready[s]`, `core_c2h_tdata[s]`,
  `core_c2h_tkeep[s]`, `core_c2h_tlast[s]` — the source core's readback output,
  **before** its c2h clock converter, in `core_ui_clk[s]`.
- **What is captured:** every *committed* beat, i.e. the cycles where
  `core_c2h_tvalid[s] & core_c2h_tready[s]` (the AXI-Stream transfer condition).
  This is exactly the set of beats the host would have received — **including the
  bare 32-B completion-ACK trailer** (`readback_engine.v`: the
  `proc_flush_r && rbf_empty && ~dsr_valid` forced beat, MM §1.2 /
  empty_record_ack_invariant). On the link those are ordinary `tlast`-delimited
  frames; the tagged consumer treats `tlast` as the frame delimiter.
- **The snoop is READ-ONLY on the handshake.** The router does **not** drive
  `core_c2h_tready[s]`, `_tvalid[s]`, `_tdata[s]`. `core_c2h_tready[s]` is driven
  solely by the existing c2h clock converter (→ XDMA). So the host c2h path is
  **bit-identical** to today, link idle or active. This is the ironclad answer to
  the never-wedge-xdma history (feedback_never_sigkill_xdma,
  platform_receivedata_no_timeout): the c2h path carries the "one program owes
  exactly one c2h message" ACK; a router that can never gate it can never make a
  program's ACK fail to return, so it can never wedge the server's `receiveData`.

**Mirror, not divert.** When a route is active the source c2h still flows to the
host unchanged; the peer receives a **copy**. Diverting (steering c2h away from
the host) was rejected: it would silence the program's host ACK and reintroduce
the exact `receiveData`-blocks-forever wedge class. Mirror keeps property (a)
(host bit-identity) true even for an *active* route, at the cost of the peer also
seeing ACK-trailer frames (benign — see §3).

## 2. INJECT POINT — tagged link_rx ingress (destination side)

- **Signal (NEW port on the core):** `link_rx_tvalid[d]`, `link_rx_tready[d]`,
  `link_rx_tdata[d]` (256), `link_rx_tkeep[d]` (32), `link_rx_tlast[d]`,
  `link_rx_tsrc[d]` (2-bit origin id) — in `core_ui_clk[d]`.
- **NOT the frontend h2c port.** The receiving frontend frames h2c as an
  *instruction/control* stream (verified `frontend.v` `INIT_MEM_S`: `h2c_tready =
  state==INIT_MEM_S`; low 64 bits = instruction; control words via bits
  [64..68]; `tlast` → `EXECUTE_S`). Readback data injected there would be decoded
  as garbage instructions into IMEM — a message with **no valid consumer**,
  violating L17. So the link delivers to a **dedicated, explicitly-tagged** port,
  not the instruction decoder.

## 3. FRAMING — the L17 decision: EXPLICITLY TAGGED (not "indistinguishable")

**Decision:** the injected stream is **explicitly tagged as a link stream on its
own ingress port**, *not* made indistinguishable from a host h2c write.

**Why this is the defensible choice against the platform contract:**
1. The core's h2c port is an **instruction decoder** with a hard semantic; the
   c2h payload is readback DATA, not an instruction program. "Indistinguishable
   from a host h2c write" would require the source to emit instruction-framed
   c2h, which the `readback_engine` never does. Forcing it there = L17 violation.
2. The completion-ACK accounting ("one program = one c2h", MM §1.2) lives on the
   instruction/readback path. A link stream masquerading as a program would
   perturb that count. A separate tagged port keeps the ACK invariant **exactly**.
3. It matches the real consumer (SCOPE §3.2): a **fabric-attention / KV data
   pipeline** in the destination UI domain — a data sink, not the decoder.

**Frame format on the link (payload-agnostic):** the router forwards
`{tlast, tkeep[31:0], tdata[255:0]}` **verbatim**; it never re-frames. Frames are
delimited by `tlast` (source c2h `tlast`). `link_rx_tsrc` carries the origin core
id so a fan-in consumer can demux. Program-injection-into-frontend (a source
sending an executable program to a peer) is a **future, explicitly-declared**
option (would need the source to emit an instruction-framed stream and a config
bit selecting frontend-inject); it is **out of scope and disabled** here.

**This build's consumer (L17 satisfied now, not deferred):** until the fabric
attention pipeline exists, each core's `link_rx[d]` is terminated by a
**`link_sink`** — `tready=1` (always drains, never backpressures), counting
beats/frames; on silicon it is the capability port the attention block will
later replace. No `link_rx` stream is ever left unconsumed.

## 4. BACKPRESSURE — asymmetric by design, never a wedge

Two datapath edges, two different rules, each justified:

- **Tap edge (c2h snoop) → DROP, never backpressure.** If the async FIFO
  `f[s][d]` is full, the snooped beat is **dropped** and `drops[s][d]`
  increments. The c2h path proceeds unconditionally (the router does not own its
  `tready`). Justification: back-pressuring c2h to wait for a slow peer would
  wedge the source readback engine → no host ACK → `receiveData` blocks forever.
  A drop is a **counted, host-visible error** (L14) the higher-level protocol
  detects and retries — never a wedge.
- **Inject edge (`link_rx`) → BACKPRESSURE, bounded, terminates at the FIFO.**
  The destination consumer asserts `link_rx_tready[d]`; when it stalls, the FIFO
  read side holds (no data lost) and the FIFO fills toward the write side. That
  backpressure propagates **only to the FIFO**, never to any XDMA channel —
  because the write side (tap) **drops** instead of stalling c2h. So a
  destination stall is absorbed by FIFO depth, then converts to counted drops at
  the tap: **bounded, observable, wedge-free by construction.**

Net: neither host datapath can ever be stalled by the link. The only failure
mode under sustained peer stall is **lossy** (counted drops), never **deadlock** —
the correct trade for a system whose one unforgivable sin is a wedged XDMA.

## 5. ROUTE TABLE — host-configurable, no hardwired pairs (the no-hardcode law)

- **Model:** one route register **per source** core, `ROUTE[s] = {en, dst[1:0]}`,
  in `core_ui_clk[s]`. Any `dst != s` is legal (self-route rejected); full
  any-to-any incl. fan-in (many sources → one dest, resolved by the dest
  arbiter). Power-up default = all-zero = **all routes disabled** (inert).
- **How it is written (established convention, verified):** the existing
  host-set registers in this RTL are **frontend control words** decoded in
  `INIT_MEM_S` (bits [64..68]: reset / readback-mode / dll / autoref / temp —
  single-beat, return to IDLE, don't touch IMEM). The link route register is
  plumbed **identically**: `frontend_linked.v` (a COPY of `frontend.v`) adds one
  new opcode bit **`tdata[69]` = LINK-CFG**; on that beat it exports
  `link_cfg_stb` + `link_cfg_data = {magic8, dst[1:0], en}` and returns to IDLE.
  The router latches `ROUTE[s]` from core s's export (same clock domain → **no
  config CDC**). With no `tdata[69]` word ever sent, `frontend_linked` is
  cycle-identical to `frontend` (the new branch is mutually exclusive with every
  existing one) — property (a) holds for the config path too.
- **Magic (self-identify, L17):** the LINK-CFG word carries an 8-bit guard
  `0x76` in `link_cfg_data[10:3]`; a mismatched guard is ignored (a stray beat
  can't arm a route). The link subsystem's stat-readback frame (below) leads with
  `0xDBC0DE76` so the host parser recognises a link-stats frame **per image,
  independent of the build-tag magic** (the bundling build owns the ladder tag —
  next is `0xDBC0DE30`).
- **Frame-atomic (runtime-changeable, cleanly — property d):** the *active* dst
  used for a frame is **sampled at frame start** (first snooped beat after a
  `tlast`/idle) from `ROUTE[s]` and **held to that frame's `tlast`**. A route
  change written mid-frame lands in `ROUTE[s]` immediately but takes effect only
  at the **next frame boundary** — so no frame is ever split across two dests,
  and no partial frame is delivered to the wrong consumer.

## 6. CDC — exactly one async crossing per route, all XPM (L12b)

- **Data:** one `xpm_fifo_async` per ordered (src,dst) pair (12 for N_CORES=4),
  `WRITE/READ_DATA_WIDTH=289` (`{tlast,tkeep[31:0],tdata[255:0]}`),
  `READ_MODE="fwft"`, `CDC_SYNC_STAGES=4`, `FIFO_WRITE_DEPTH=LINK_FIFO_DEPTH`
  (param, default 512). Write side = `core_ui_clk[src]`, read side =
  `core_ui_clk[dst]`. **This is the only async boundary.** Per-pair FIFOs (vs one
  FIFO per source) are deliberate: each FIFO has exactly one write clock and one
  read clock — a single-source FIFO whose head could target different dests would
  need multiple read clocks, which an async FIFO cannot have. Per-pair also
  removes cross-dest head-of-line blocking.
- **No config CDC:** the routing decision is made entirely on the **write
  (source) side** (which FIFO to push), so `ROUTE[s]` is consumed only in its own
  domain. The dest arbiter just drains whatever FIFOs targeting it are non-empty.
- **Counters for host readback:** each counter lives in its write domain
  (`beats/frames/drops[s][d]` in `ui_clk[s]`; `inj_frames[d]` in `ui_clk[d]`). A
  stat read **freezes** counting, then crosses the now-static values via
  `xpm_cdc_array_single` into the stat domain — a multi-bit CDC of a *stable*
  value, which is exactly what `xpm_cdc_array_single` is contracted for. All CDC
  is vendor XPM; **nothing hand-rolled** (L12b / feedback_xpm_over_handrolled).
- **Verilator note:** the gate substitutes behavioral models
  (`sim/xpm_fifo_async.v`, `sim/xpm_cdc_array_single.v`) for the encrypted XPM
  IP — the same pattern as `verilator_local/rdback_fifo_sim.v` standing in for
  `rdback_fifo`. The gate proves **functional** CDC correctness (data integrity,
  loss only on real overflow, drops counted) across different clock *ratios*;
  metastability is out of scope for a 2-state cycle sim and is the real XPM's
  contract, asserted binary by timing closure (L12). Source instantiates the
  **real** `xpm_fifo_async`/`xpm_cdc_array_single` (guarded by
  `` `ifdef LINK_SIM_MODELS ``).

## 7. COUNTERS (host-readable, established style, L14)

Per ordered pair (s,d): `beats[s][d]`, `frames[s][d]` (tlast count),
`drops[s][d]` (FIFO-full at tap). Per dest: `inj_frames[d]` (frames delivered to
`link_rx[d]`). Each is a saturating counter with a width param; exposed as router
outputs for the gate and, on silicon, snapshotted (freeze + `xpm_cdc_array_single`)
into a `0xDBC0DE76`-led stats frame returned on core-0 c2h via a small tagged mux
(the same "control word triggers a c2h return" pattern as read-temp). Assert, do
not merely print (L14): a nonzero `drops` on a route the host believed loss-free
is a loud host-side failure.

---

## 8. Gate map (see tb/ and logs/)

| property | how the contract makes it true | gate |
|---|---|---|
| (a) idle = bit-identical passthrough | router never drives host h2c/c2h signals; snoop read-only; `frontend_linked` new branch inert | `tb_passthrough` |
| (b) core0→core2 intact across async clks | verbatim `{tlast,tkeep,tdata}` through `xpm_fifo_async`, dest arbiter, different TB clock periods | `tb_route_cdc` |
| (c) backpressure/stall, no wedge, counters | tap drops on FIFO-full (counted); inject backpressure terminates at FIFO; c2h never stalled | `tb_backpressure` |
| (d) runtime route change clean at frame boundary | dst sampled at frame start, held to tlast; mid-frame reconfig applies next frame | `tb_reroute` |

---

## 9. REGISTER MAP (L25 observability — every register + its host-side reader)

All registers follow the **established host-register conventions**: writes ride
frontend control-words (single-beat, `INSTR_WIDTH+n` opcode, return to IDLE);
reads ride the readback-engine's `hbm_temp_rd`→c2h return pattern
(`readback_engine.v:148-158`) — a self-identifying frame (magic first word) the
host parses by content (L17). **No fixed offsets; no hardcoded routes** (values
are data). Each register names the tool/env that reads or writes it — a register
with no reader is a message with no consumer (L17), so none ship without one.

### 9a. Link registers (in this deliverable — `inter_bender_link.v`)

| register | dir | opcode / path | fields | default | CONSUMER (tool/env) |
|---|---|---|---|---|---|
| `ROUTE[s]` (route table) | host-set | frontend ctrl-word `INSTR_WIDTH+5`, guard magic `0x76` | `{dst[2:1], en[0]}` per source core | **0 = all routes disabled (star)** | **host `pim_link.py`** (new): `link_route(src,dst,en)`; env `PIM_LINK_ROUTES` |
| `stat_beats[s][d]` | host-read | stat frame (magic `0xDBC0DE76`) | 32b saturating | 0 | `pim_link.py --stats`; asserted by the server teardown (L14) |
| `stat_frames[s][d]` | host-read | stat frame | 32b (tlast count) | 0 | `pim_link.py --stats` |
| `stat_drops[s][d]` | host-read | stat frame | 32b (tap FIFO-full) | 0 | `pim_link.py --stats`; **nonzero on a loss-free route ⇒ loud fail** (L14) |
| `stat_injframes[d]` | host-read | stat frame | 32b | 0 | `pim_link.py --stats` |
| `stat_fill[s][d]` | host-read | stat frame | 10b FIFO occupancy | 0 | `pim_link.py --status` (CDC-stall triage) |
| `stat_status[d]` | host-read | stat frame | `{cause[6:5],sel[4:3],dvalid,tready,locked}` cause∈{idle,stream,consumer-stall,starved} | 0 | `pim_link.py --status` — **the CDC-stall failure-mode readout** |

Stat readback mechanics (silicon): a host "read link stats" control word raises a
**freeze**, the now-static counters cross via `xpm_cdc_array_single` (`link_stat_sync.v`)
into the readback engine's ui_clk, and are muxed into `rdback_din` as a
`0xDBC0DE76`-led c2h frame. The Verilator gate reads the raw counter outputs
directly (same values, pre-CDC).

### 9b. Bundle registers (specified for the other authors — see MANIFEST.md; reference RTL provided)

| register | dir | path | reader/writer | reference RTL |
|---|---|---|---|---|
| `FEATURES` bitmap | host-read, **baked** | feat read ctrl-word → c2h (magic-led) | **`pim_features.py`** / server startup provenance assert ("is the fix in this image?") | `build_features.v` |
| `MANIFEST_HASH` | host-read, baked | same frame | `pim_features.py` cross-checks against the shipped `MANIFEST.md` hash | `build_features.v` |
| `ENABLE[feature]` | host-set, **default-inert** | ctrl-word write | `pim_features.py --enable/--disable`; env `PIM_FEATURE_ENABLE` for silicon A/B | `build_features.v` (`en_maj5/en_pack4`) |
| `POPCOUNT_BIST` (pattern in / count out) | host wr + read | pattern DLOAD + bist read ctrl-word → c2h | **`pim_popcount_bist.py`** — one-register 0xe self-test on every image | `popcount_bist.v` |

**Consumer discipline (L17):** `pim_link.py`, `pim_features.py`,
`pim_popcount_bist.py` are the named readers; the server's startup/teardown
asserts `FEATURES` matches the flashed manifest and fails loudly on a
`drops>0`/BIST-mismatch — no register is write-only.
