# MANIFEST — bundled flash-worthy build + observability spec (Task #76 + L25)

**⚠ DO NOT TRANSFER. The Vivado box has a LIVE session — this file STAGES the
transfer and BINDS the bundle's authors; it does not perform it.** Box access is
READ-ONLY reference only; no writes, no card, no git (per the work order).

This manifest is, per the L25 ruling, the **observability spec for the whole
bundled build**: "we can add multiple things into a new build, but expect things
to go wrong — add observability." Every ingredient inherits the requirements in
§1–§4; the file list + md5 checklist for the flash is §5–§6.

**Bundle (LEVERS #76 ruling (3)):** the inter-bender link rides the next
flash-worthy build together with:
1. **pop_count4 fix** (`0abccc0e`, task #77) — MANDATORY (the buggy `6490c9b7`
   is in the box build tree; every prior Verilator gate ran the fixed one).
2. **MAJ5 datapath** (C51 — mainstream-only value).
3. **pack-4 seqgen** (16-bank interleave).
4. **inter-bender link** (this task).

Next magic-ladder build tag = **`0xDBC0DE30`** (current flashed image = build-48
`0xDBC0DE2F`). Link/observability self-id magic = **`0xDBC0DE76`** (per-image,
independent of the build tag — L17 self-identify).

---

## 1. FEATURES bitmap register (baked provenance — "is the fix in THIS image?")

Baked at synthesis into `build_features.v` (reference RTL provided). Host reads
it via the feat-read control-word → c2h frame. **Bit map (shared by FEATURES and
ENABLE, no-hardcode):**

| bit | ingredient | FEATURES=1 means | present this bundle? |
|---:|---|---|---|
| 0 | POPCOUNT4_FIX (`0abccc0e`) | fixed pop_count4 synthesized | **required** |
| 1 | MAJ5_DATAPATH | MAJ5 datapath present | yes |
| 2 | PACK4_SEQGEN | pack-4 seqgen present | yes |
| 3 | INTER_BENDER_LINK | link present | yes (this task) |
| 7:4 | reserved | 0 | — |

`FEATURES = 32'h0000_000F` for the full four-ingredient bundle.
**Source-manifest hash register** (`MANIFEST_HASH`, cheap): 32-bit CRC32 over the
sorted md5 list of the bundled sources (**excluding `build_features.v` itself**,
which carries the hash — a file cannot hash-include itself) = **`0xC17689CF`** for
the currently-staged link+obs+popcount subset (§5a). **Recompute at bundle
freeze** once §5b (MAJ5, pack-4) land, and bake the new value; `pim_features.py`
cross-checks the register against the shipped manifest — a mismatch means "the flashed sources are not the
ones in this manifest" (the build-9 stale-flash class, L17/L18).
**Reader:** `pim_features.py` + server-startup provenance assert.

## 2. Per-feature ENABLE registers (host-set, DEFAULT-INERT — A/B without reflash)

Every NEW datapath is individually disable-able at runtime so a silicon A/B
isolates the misbehaving ingredient without a re-flash (L25). **Default = the
proven fallback** (feature OFF), so a bad ingredient cannot corrupt the baseline
until explicitly enabled.

| feature | ENABLE mechanism | default | BINDS the author to |
|---|---|---|---|
| **link** | its **route table** `ROUTE[s]` (all-disabled = star) | inert (star) | *(done — this task)* |
| **MAJ5 datapath** | `en_maj5` (`build_features.v`) gates the MAJ5 path; OFF ⇒ existing MAJ3/reference datapath | **OFF** | the MAJ5 author must consume `en_maj5` and fall back to the proven path when 0; no MAJ5 state may perturb the baseline when disabled |
| **pack-4 seqgen** | `en_pack4` gates pack-4 command generation; OFF ⇒ serial/proven seqgen | **OFF** | the pack-4 author must consume `en_pack4` and produce byte-identical baseline behaviour when 0 (BANKGATE-style gate, C81 precedent) |
| **pop_count4** | no runtime toggle (a fix, not a mode); provenance via FEATURES bit0 + the §4 BIST | fix always active | — |

**Requirement on the MAJ5 / pack-4 authors (binding):** ship the enable as a
host control-word register in the established style (CONTRACT.md §9), default 0,
and a **permanent sim gate** proving `enable=0` is byte-identical to the current
build (reproduce-then-enable, both polarities — the C81/BANKGATE discipline).
**Reader/writer:** `pim_features.py --enable/--disable`; env `PIM_FEATURE_ENABLE`.

## 3. Link observability (built — `inter_bender_link.v`)

Per-route `beats / frames / drops` + per-dest `injframes`, plus a **link status
register** (`stat_fill` FIFO occupancy, `stat_status` stall-cause ∈
{idle, stream, consumer-stall, starved}) — because a CDC stall is the likely
failure mode of an async-FIFO link. All documented in CONTRACT.md §9a with the
`pim_link.py` reader. **`drops>0` on a route the host believed loss-free is a
loud teardown failure** (L14). Gate-proven (logs/gate_verdicts.log).

## 4. pop_count4 BIST (one-register self-test on every future image)

Turns the recurring "is the 0xe undercount fixed in THIS image?" (task #77) into
a register round-trip: host writes a test pattern, reads back the popcount the
**hardware** computed through the real `pop_count4` tree. A 0xe-dense pattern on
a buggy image returns a value below the true popcount → BIST fails on the exact
defect. **Reference RTL: `popcount_bist.v`.**

**Reuse the existing debug path (checked, per the ruling):** the readback engine
ALREADY returns a register via control-word → c2h (`readback_engine.v:148-158`,
the `hbm_temp_rd` path muxing a value into `rdback_din`). The BIST result returns
the **same way** — no new c2h logic. **Preferred (if cheap on the box):** mux the
host pattern into the **existing production popcount input** (readback_engine.v:81
`pop_count4 pci`) rather than the parallel copy, so the BIST tests the exact
synthesized production tree; `popcount_bist.v` is the fallback when that input
mux is not readily reachable. **Reader:** `pim_popcount_bist.py` (run on every
flashed image; a mismatch blocks the rung-c / SEG_POP / #68 popcount claims that
#77 gates).

---

## 5. FILE LIST + md5 (L18: md5 every edited RTL against its box destination)

### 5a. This task's staged sources (LOCAL — `link_rtl_2026_08_04/`)

| file | md5 | box destination (on transfer) | role |
|---|---|---|---|
| `rtl/inter_bender_link.v` | `cdb4a3b5698d759438307331f30dc292` | `projects/BCU1525_QUAD/verilog/` | router (synth) |
| `rtl/link_async_fifo.v` | `a0adbc893fb4b4aee841cc6db8d5d3f1` | `projects/BCU1525_QUAD/verilog/` | XPM async FIFO wrapper (real `xpm_fifo_async`) |
| `rtl/link_sink.v` | `39ba50d2325be5451f9e5a49a5cfd8aa` | `projects/BCU1525_QUAD/verilog/` | L17 link_rx consumer (interim) |
| `rtl/frontend_linked.v` | `b70a220f540b6cb75a254d30106aad1b` | **replaces** `sources/hdl/verilog/frontend.v` use in the QUAD | route-cfg control-word |
| `rtl/softmc_core_linked.v` | `bb48a9882108a25d60785dcb881f822f` | `projects/BCU1525_QUAD/verilog/` | core exporting route-cfg |
| `rtl/bcu1525_quad_top_linked.v` | `be2ca40e9d4e8baab55960f508d89871` | `projects/BCU1525_QUAD/verilog/` (top) | star + router (tap/inject/sink) |
| `rtl/build_features.v` | `52fdf24c230be63c7baa3208409cb6af` | `projects/BCU1525_QUAD/verilog/` | FEATURES/ENABLE/hash (§1–§2) |
| `rtl/popcount_bist.v` | `dc484bae11168a789619170cd22d1854` | `projects/BCU1525_QUAD/verilog/` | pop_count4 BIST (§4) |
| `rtl/link_stat_sync.v` | `02f200055710b6b8566fae462f07622a` | `projects/BCU1525_QUAD/verilog/` | CDC-clean stat readback (real `xpm_cdc_array_single`) |
| `sim/xpm_fifo_async.v` | `0687b747b3e4b5e292c9c55aa84e894e` | **GATE ONLY — never flashed** | Verilator model of the XPM |
| `sim/xpm_cdc_array_single.v` | `f7b7bea357783d6f70be3fbb6ff45b37` | **GATE ONLY — never flashed** | Verilator model of the XPM |
| `tb/tb_link.cpp` | `9bec0415de7112ff9bbe8f86b055ce2f` | **GATE ONLY** | the 4-property gate |
| `run_gate.sh` | `2d22402afff5a001de8ee3ad23f3207b` | **GATE ONLY** | build+run the gate |

Synthesis binds the **real** `xpm_fifo_async` / `xpm_cdc_array_single` (Xilinx
XPM library) — the `sim/` models are Verilator stand-ins ONLY and MUST NOT be
added to the Vivado source set (L12b: contract explicit, vendor primitive on the
board).

### 5b. Co-bundle sources owned by other tasks (RTL not authored here — this manifest BINDS them)

| ingredient | fixed/known md5 | box destination | owner / status |
|---|---|---|---|
| **pop_count4 fix** | `0abccc0e…` (= `verilator_local/pop_count4.v`) | `sources/hdl/verilog/pop_count4.v` (**overwrite the buggy `fb5ba88e` / box-tree `6490c9b7`**) | task #77 — file exists, staged copy + L18 md5 when box idle |
| **MAJ5 datapath** | *TBD (author to fill)* | engine / descriptor-gen | task (C51) — **must add `en_maj5` gate + sim gate per §2** |
| **pack-4 seqgen** | *TBD (author to fill)* (ref `seq_engine.v` `1acc88cd…`) | seqgen / engine | task — **must add `en_pack4` gate + sim gate per §2** |

## 6. PRE-FLASH CHECKLIST (L18 — do ALL before any flash; box currently LIVE, so defer to idle)

- [ ] md5 every edited RTL file against **both** trees (shared `sources/hdl` AND
      QUAD-local overrides); paste the box-side md5 next to §5.
- [ ] grep the new magic `DBC0DE30` (build tag) **and** `DBC0DE76` (link self-id)
      present in the synthesized sources, both trees.
- [ ] `check_fidelity` both directions (sim ↔ box) on every touched file.
- [ ] post-build grep `Synth 8-7071` on the QUAD ports — the QUAD wrapper strands
      fixes silently if a new port is unconnected (builds 11–13 precedent);
      verify `link_cfg_*`, `linkrx_*`, `stat_*` all connected.
- [ ] `all_latches==0`, `check_timing no_clock==0`, **WNS ≥ 0 with corners**
      (timing closure is binary — L12); the two ui_clk domains of every routed
      pair are a real async crossing, covered by the `xpm_fifo_async` CDC
      constraints — confirm the XPM's generated XDC is applied.
- [ ] FEATURES/ENABLE default-inert proven in sim for MAJ5 + pack-4 (§2), link
      inert-by-route proven (this gate).
- [ ] `.bit` backups of the current image before flashing.
- [ ] one invasive change-set understood per build (L16): this is FOUR
      ingredients — the ENABLE registers are precisely how we de-risk that (A/B
      each OFF→ON on silicon).

## 7. WHAT THE BUNDLE STILL NEEDS BEFORE VIVADO (honest gaps)

1. **XPM instantiation on the box**: `link_async_fifo` instantiates
   `xpm_fifo_async`; the QUAD `create_project.tcl` must pull the `xpm` library
   (add `xpm_fifo_async`/`xpm_cdc_array_single` to the compile order). The link
   uses **12** async FIFOs (N·(N−1), depth 512, 289-bit) ≈ **~60 BRAM36** (2.8%
   of 2160; SCOPE §5 shows 81% BRAM headroom) — confirm on the box.
2. **Stat-return integration** into the readback engine: the freeze +
   `xpm_cdc_array_single` snapshot → `rdback_din` mux on the stat control word
   (CONTRACT §9a). The router already exposes the raw counters; the c2h-return
   mux is the readback-engine author's ~10-line addition (temp-read pattern).
3. **FEATURES/BIST integration**: `build_features.feat_word` and
   `popcount_bist.bist_popcount` muxed into `rdback_din` on their control words
   (§1/§4) — same readback-engine pattern.
4. **frontend opcode room**: `frontend_linked` uses `INSTR_WIDTH+5`; the FEATURES
   read, ENABLE write, and BIST pattern/read need `INSTR_WIDTH+6..+9` — reserve
   them in the shared frontend and keep the QUAD's copy in sync (L18 both trees).
5. **MAJ5 / pack-4 RTL + their md5s (§5b) + enable gates (§2)** — owned by their
   tasks; this manifest is the binding checklist they inherit.
6. **Per-SLR floorplan** (SCOPE §5 UNKNOWN-1): the router's 12 FIFOs + the
   storage-core attention consumer should share an SLR to avoid an SLR crossing
   on a line-rate routed path — a box characterisation question.

**Gate status:** all four link properties PASS locally
(`logs/gate_verdicts.log`); the bundle is Vivado-blocked only on the box being
idle + items 1–5. **Nothing in this manifest has been transferred.**
