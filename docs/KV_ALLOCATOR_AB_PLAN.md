# KV ALLOCATOR — staged silicon A/B

**Pre-registered silicon A/B plan (staged).** The procedure below is the plan to
run on a free card; it is written against existing, already-built probes and is
the companion to `KV_ALLOCATOR_DESIGN.md`.

**Rules baked in** (memory): NUMERICS/BYTE-EXACT GATE FIRST, never token identity
(`pim_numerics_gating_correlation`, `l0_gate_insensitive`); **no `timeout`
wrappers**, SIGTERM-and-wait only (`feedback_no_timeouts_on_fpga`,
`feedback_never_sigkill_xdma`); same-process compares only; use **existing
tooling** (`feedback_use_existing_tooling`) — the probes below already exist and
are built. Env every step: `BITSTREAM_IMEM=8192`.

**Which channel is which** comes from `calibration/DIMM_POPULATION.conf`, never
from this document. Below, `$KV` is a channel the population file puts in the
`STORAGE` role — the KV home — and `$CTL` is a channel in the `COMPUTE` role,
used as the control. `python3 python/dimm_population.py --sh <n>` emits the
calibration, pool and window for a channel; the commands use those rather than
naming a fixture file.

Two of the legs below exist because the co-activation *selection law* is a
die-family property. Every module in the present population belongs to the
family whose law and safe-offset menu are enumerated, so those legs are only
load-bearing when the KV home is a module from another family. They are kept
because that is exactly the case the allocator must not silently get wrong.

**Probes used (all already built, Makefile `BINARIES`):**
`.../DSN_AE_APPS/BitNet/`
- `rowclone-preserve-exe <bender> <calib> <bank> <n_iter>` — writes a pattern to a
  non-open `src` row, RowClones `src→Rfirst`, and reports **per-iter**
  `iter,src_pc,src_match,dst_pc,dst_match` (source-survival AND copy-exactness) +
  a full-body step-by-step src-preservation trace. **This is the fork probe.**
- `spread-test-exe <bender>` (env `PIM_BANK`, `PIM_TEST_R`, `PIM_DST_FAR`,
  `PIM_CALIB_LINE`) — constellation probe over 25 XOR offsets × doubleACT modes
  `(30,1)/(10,2)/(0,0)`; reports which offset rows receive collateral deposits.
  **This is the safe-load / coset-placement probe.**
- `rowclone-smoke-exe <bender> <bank> <src_row> <tgt_row> [seed]` (env `PIM_T12`) —
  single clone, prints `PERFECT_CLONE` at 8192/8192. **Intra- vs cross-segment probe.**

---

## What the A/B will PROVE (map to the DESIGN unknowns)

| A/B | proves | DESIGN ref |
|---|---|---|
| **1a** fork | RowClone fork is **byte-exact** (dst 8192/8192 vs the written pattern = "vs recompute") AND the **shared source survives** (src_match 8192/8192 across iters) — the two facts prefix-share needs | KV-1; unknown #2 |
| **1b** locality | **intra-segment clone = PERFECT, cross-segment clone fails/partial** — the RowClone-intra-subarray law that forces INVARIANT KV-1/KV-2 | KV-1, KV-2 |
| **1c** placement | the fork's collateral coset lands **only on the predicted offset rows** — validates the safe-load offset menu, and screens the clean k=1 offset if the KV home is a foreign-family die | KV-3; unknown #3 |
| **2a** retention | the **KV home's idle-age drift onset** — the number the scrub period is set from | KV-4; unknown #4 |
| **2b** scrub | a **read-rewrite scrub at period < onset → 0 mismatch** ("the rewrite IS the refresh" for storage KV) | KV-4 |
| **2c** self-refresh | whether **attention-cadence reads alone** (ACT top-up) hold a page below onset | KV-4; unknown #5 |

**Out of scope (honest):** this A/B validates the *primitives* the allocator
stands on. It does NOT test the multi-page allocator at scale, cross-DIMM staging
(#67), or fabric attention (#72) — those need the allocator + a fabric build.

---

## STEP 0 — bring-up (RUNBOOK_TOWER, user at keyboard)
```
lspci -nn -d 10ee:                       # expect 10ee:9038 (locate by VENDOR, never BDF)
# card must be FREE (one consumer, xdma refcount 0). If wedged: full power cycle (user).
```

---

## A/B-1 — PREFIX-SHARE FORK by RowClone (KV pages)

### 1a — fork byte-exactness + source survival (the core prefix-share test)
Run the fork probe on the **KV home** and on a **compute-role control**. The
written pattern is the "recompute" reference; `dst_match=8192` == byte-exact
fork; `src_match=8192` across all iters == the shared prefix survives the fork.
```
cd "$BN"; export BITSTREAM_IMEM=8192
eval "$(python3 <repo>/python/dimm_population.py --sh $KV | sed 's/^/KV_/')"
eval "$(python3 <repo>/python/dimm_population.py --sh $CTL | sed 's/^/CTL_/')"

# KV home, bank 0, 50 fork iterations:
./rowclone-preserve-exe $KV  "$KV_TRIO_CALIB"  0 50  > $OUT/ab1a_kv.csv
# compute-role control:
./rowclone-preserve-exe $CTL "$CTL_TRIO_CALIB" 0 50  > $OUT/ab1a_ctl.csv
```
**PASS (pre-registered):** every iter `dst_match=8192` (fork byte-exact) AND
`src_match=8192` (prefix source intact) on both channels. Any `src_match<8192`
would mean RowClone consumes/perturbs the shared source → prefix-share by
reference is unsafe and forks must copy-and-keep-golden instead (records the
finding, does not kill the design). RowClone (30,1) is PERFECT on the installed
family, and t23=1..3 on the other family we measured → expect PASS. If the KV
home is a foreign-family module, this leg is what retires the fork half of its
unknown.

### 1b — intra-segment vs cross-segment (INVARIANT KV-1/KV-2 on silicon)
A KV page fork is free **only** inside a 640-row sense-amp segment. Clone within a
segment (expect PERFECT) and across the 640 boundary (expect fail/partial).
Anchor to the KV home's window `[W, W+640)` — `W = $KV_TRIO_SUB_START`.
```
# intra-segment: src and tgt both inside one 640-row segment (offset < 640):
PIM_T12=30 ./rowclone-smoke-exe $KV 0 $((W+16)) $((W+272)) 0xC0FFEE   # expect PERFECT_CLONE 8192/8192
# cross-segment: tgt in the NEXT segment (offset >= 640):
PIM_T12=30 ./rowclone-smoke-exe $KV 0 $((W+16)) $((W+656)) 0xC0FFEE   # expect NOT perfect (intra-subarray law)
```
**PASS:** intra-segment `PERFECT_CLONE`; cross-segment materially below 8192/8192.
This is the silicon proof that pages must be segment-aligned (KV-2) and a fork's
CoW twin must be co-segment (KV-1). ⚠ Check `W` and `W+640` against every other
window in use modulo 2^15 first — aliased windows pass this test independently
and then overwrite each other.

### 1c — safe-load coset placement (which rows the fork collaterally writes)
The fork RowClone `(30,1)` sits in the ≤3-NOP co-activation window, so on a die
that obeys the selection law it also deposits into the coset of
`local(src)⊕local(dst)`. Map the coset so the allocator reserves those rows dead
(safe-load) and never places a live KV page in them. On a foreign-family die,
screen which single-generator (k=1) offset is clean instead.
```
# selection-law die: reproduce the lawful coset (deposits ONLY on lawful members):
PIM_BANK=0 ./spread-test-exe $KV > $OUT/ab1c_kv_coset.txt
# foreign-family die: screen the clean k=1 fork offset (expect ONLY the
# XOR-generator partner dirty):
PIM_BANK=0 PIM_CALIB_LINE="$(grep -v '^#' "$KV_TRIO_CALIB" | head -1)" \
    ./spread-test-exe $KV > $OUT/ab1c_kv_k1.txt
```
**PASS:** a selection-law die reproduces the lawful coset (deposits on predicted
members only — MM §2.3/§2.4); a foreign-family die shows a **single clean k=1
offset** whose only collateral is the `⊕generator` partner (that partner becomes
a reserved dead slot in the KV page layout). Output feeds the allocator's
per-die safe-offset table.

---

## A/B-2 — RETENTION AGE BRACKET (idle KV drift + scrub)

No existing exe does *timed* brackets, so stage one thin driver
`kv_retention_probe.py` (built on the same write→sleep→readback→diff primitives;
~40 lines; NOT written this pass). Pre-registered arms and expectations below.
**All sleeps are pure host idle with ZERO intervening card commands** (per the
drift-ladder finding that passive idle damages MOST — MM §2.5).

### 2a — idle-age drift on the KV home (measures the onset)
```
# write a known pattern to a KV row, then read back at increasing idle ages:
python3 kv_retention_probe.py --bender $KV --calib "$KV_TRIO_CALIB" --bank 0 \
    --mode idle --ages 0.1,10,30,60,120,300 > $OUT/ab2a_kv_retention.csv
# the compute-role control as the cross-die reference:
python3 kv_retention_probe.py --bender $CTL --calib "$CTL_TRIO_CALIB" --bank 0 \
    --mode idle --ages 0.1,10,30,60,120,300 > $OUT/ab2a_ctl_retention.csv
```
**Pre-registered expectation:** mismatched-cells CDF rising with age; ~30–120 s
onset on the installed family per the drift ladder (MM §2.5). A KV home from
another family has an unknown onset and this arm sets it. The scrub period
(KV-4) = ½ · the minimum measured onset across every channel holding KV.

### 2b — scrubbed: read-rewrite at period < onset → 0 mismatch
```
python3 kv_retention_probe.py --bender $KV --calib "$KV_TRIO_CALIB" --bank 0 \
    --mode scrub --scrub-period 15 --total 300 > $OUT/ab2b_kv_scrub.csv
```
**PASS:** 0 mismatched cells at t=300 s (vs 2a's drift at the same age) — the
"rewrite IS the refresh" result carried to storage-only KV. If a 15 s period does
not hold 0, shorten to ½ the measured onset from 2a.

### 2c — self-refresh by read cadence (does attention-rate ACT suffice?)
```
# read (ACT) the page every 5 s (simulating token cadence), NO explicit rewrite:
python3 kv_retention_probe.py --bender $KV --calib "$KV_TRIO_CALIB" --bank 0 \
    --mode read-only --read-period 5 --total 300 > $OUT/ab2c_kv_readrefresh.csv
```
**Pre-registered expectation:** if reads before onset keep charge topped, 0
mismatch (→ active generation is self-refreshing, no extra transport for hot KV).
If mismatches appear, ACT-charge-top-up does NOT preserve content at this cadence
→ even active KV needs the 2b read-rewrite scrub. Either result is decisive and
sizes KV-4 (unknown #5).

---

## Pass/fail summary (pre-registered gates)

| gate | pass condition | consequence if fail |
|---|---|---|
| 1a | dst 8192/8192 AND src 8192/8192, both channels | src-perturbed ⇒ prefix-share needs golden-keep, not by-reference (record) |
| 1b | intra-seg PERFECT, cross-seg not | (won't fail — it's the known law; confirms it on the KV window) |
| 1c | lawful coset on a selection-law die; 1 clean k=1 offset on a foreign-family one | no clean offset ⇒ forks on that die fall back to bus read+write |
| 2a | drift CDF measured; onset recorded per KV channel | — (this arm produces the number) |
| 2b | 0 mismatch at 300 s with 15 s scrub | shorten period to ½·onset from 2a |
| 2c | 0 mismatch with 5 s read cadence | hot KV also needs explicit scrub (2b path) |

**Card-health close gate** (every session, MM mig-reinit policy): rowclone
8192/8192 + card `10ee:9038` healthy + xdma refcount 0 at exit. Agent never
flashes; any recal is user/on-detection only.
