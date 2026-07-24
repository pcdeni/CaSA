# e2e_sim — end-to-end pipeline simulation (phase 1) + build-12 root cause

Date: 2026-07-24 (evening). Status: **phase 1 OPERATIONAL; build-12 wedge
REPRODUCED, root-caused, FIXED (= build-13 RTL), regression ALL GREEN.**

User-mandated instrument (asked for repeatedly; built same-day after the
build-12 silicon incident): simulate the REAL memory-controller pipeline
end to end instead of keyhole TBs. This sim caught in ~2 hours what three
generations of fetch-boundary TBs (builds 9–12, scenarios A–I) could not.

## What it is

- `rtl/` — the REAL RTL, verbatim from the synthesis tree: frontend
  (build-12), fetch_stage (build-12), softmc_pipeline with the real
  decode_stage / execute_stage / exe_pipeline / register_file /
  ddr_pipeline / pre_decode / maintenance_controller. Only three
  behavioral stand-ins: IMEM BRAM, scratchpad (`XILINX_SIMULATOR` path,
  upstream's own), and the three maintenance ROMs — loaded with the
  REAL microcode converted from the QUAD project .coe files.
- `e2e_top.v` — wiring copied VERBATIM from softmc_top.v (the two
  instantiations + the fr_* interconnect). h2c AXIS driven by the TB at
  the exact post-CDC boundary. buffer_space tied never-full (phase 1
  has no readback engine).
- Stimulus = REAL machine code: `gen_progs.cpp` links the actual
  Program/instruction encoder (same objects as the silicon tools) and
  emits hex; the TB streams it with the exact platform.cpp beat protocol
  (one 256-bit beat per instruction, u64 low lane, tlast on last).
- Faithful boot: init_calib_complete 0→1 transition (the maintenance
  timers only LOAD while calib is low — tie it high and maintenance
  never fires; that was the first sim-vs-silicon gap found).
- Runs on the vivado box (`/home/daniel/Claude/bcu1525/e2e_sim/`,
  Verilator, --trace-fst). Tower copy = this dir (authoring master).

## The build-12 root cause (reproduced, then fixed)

**The design truth my build-12 model got backwards:** maintenance does
NOT bypass fetch. `frontend.v`:

```
assign program_process = (state_r == EXECUTE_S) && ~maint_process;
assign valid_out = program_process ? valid_out_sr[0] : maint_valid;
```

Maintenance microcode is MUXED ONTO FETCH'S IMEM-RETURN PATH — fetch is
the conveyor for maintenance. Build-12's park-at-END + my removal of the
maint-site restart pulse ("maintenance bypasses fetch" — wrong) means:

1. Any program's END parks fetch (WAIT_RESTART_S).
2. Host idles ≥ ~1 µs → periodic-read maintenance fires (tPRDI is 1 µs
   in this RTL — maintenance runs CONSTANTLY on silicon).
3. The microcode arrives at a PARKED fetch → dropped → maint never
   finishes → `maint_process` sticks high.
4. Next program: restart un-parks fetch, but `program_process` stays 0
   (maint stuck) → the mux never switches to the program → no dispatch,
   no response; INIT_MEM backpressures → h2c write errno-512.

Sim reproduction (build-12 RTL): first per-RD after S3 stuck
`maint_proc=1` forever; next send: tready never rose — the exact
silicon jam, cycle-visible in `e2e.fst`.

**Silicon reconciliation:** "branch programs wedge" was a red herring of
ORDER, not content. P1 passed because it was the first program after
reset_fpga with no idle gap (no maint yet; and at cold boot fetch has
never seen an END, so nothing is parked — also why every build boots).
P2 died because ms of host idle guaranteed a stuck maint first. The
6-inst "branch" repro on silicon was really "second program after an
idle gap."

**Fix (= build-13):** restore `fetch_restart = 1'b1` at the maint-ack
site with the corrected comment. Park semantics become the clean
protocol: fetch parks at END until the NEXT source, sources =
{legacy load, stream swap, maintenance entry}. Park+fence (the real
E14 fixes) stay.

**E14 history settled:** build-11's E14 was the re-loop's stale branch
resolve (park+fence already address it). The maint pulse was innocent;
removing it was fixing the wrong suspect.

## Regression (build-13 RTL, one run, `0 hard fails`)

- S1 branch-free 46-inst read: FIN + exactly 8 ddr_read beats, 1 ACT.
- S2 6-inst BL loop: completes.
- S3 post-branch read: completes.
- Boot ZQ: ~20k-cycle power-on ZQ calibration runs to completion
  (explains the long first-send window; by design).
- Idle probe: per-RD every ~166 cyc, each completes in ~15 cyc,
  park→maint-pulse→run→re-park rhythm visible.
- M: 64-phase sweep of the branch loop against live maintenance —
  all complete; overlap-in-window case survived.
- N: 8 STREAMED branch-loop programs with host-pacing gaps + live
  maintenance: 8/8 fins (maint fins correctly excluded — maintenance
  shares EXECUTE_S and also ends with softmc_fin), post-stream legacy
  read clean. This is the E14 shape no prior TB could express.

## Sim-vs-silicon gaps found on the way (each a fidelity lesson)

1. `init_calib_complete` history matters (timer-load gating).
2. `program_pending` is unconnected in softmc_top → tied 0 on silicon;
   sim must tie 0 too (faithful), not leave floating assumptions.
3. Upstream BLKANDNBLK in maintenance_controller (line 265) — blocking
   assign in reset branch; patched in sim copy only.
4. tPRDI=1 µs (ps-vs-ns unit soup upstream) — maintenance is ~10³×
   more frequent than JEDEC nominal; a load-bearing fact for every
   idle-gap race on this platform.

## Phase 2 (planned)

Behavioral DRAM array behind ddr_* (per-bank rows, ACT/PRE/READ/WRITE,
optionally the charge-sharing lattice ops) + the readback engine →
host-bytes-in → host-bytes-out, replayable against reqcap captures.

## Files

Tower (authoring): /home/deni/Claude/e2e_sim_2026_07_24/
  rtl/*.v, rtl/*.hex (microcode), tb_e2e.cpp, gen_progs.cpp (in BitNet
  dir, emits s1_read.hex/s2_brloop.hex here), this RESULT.
Box (runs): /home/daniel/Claude/bcu1525/e2e_sim/ (+ e2e.fst traces).
Build-13 frontend: rtl/frontend.v here = the fixed build-12 copy.

## Phase 2 — CLOSED LOOP, ALL GREEN (2026-07-24 night)

Host-bytes-in → host-bytes-out now runs end to end: the REAL readback
engine (build-13 copy, magic 0C) + a behavioral DRAM (dram_model.v shim,
dram_dpi.cpp: open-row tracking, sparse content store with deterministic
seeding, in-order RL=24 return queue) behind the ddr_* interface;
buffer_space is engine-driven (real backpressure). rdback_fifo_sim from
the old harness provides the 512→256 width conversion (low-half-first,
IP-faithful).

Verdicts (one run, 0 hard fails, all phase-1 scenarios still green):
- R1: full-row read (REAL rdRow_immediate_label, 21 insts, branch loop)
  → 8192 B payload BYTE-EXACT vs the DRAM oracle + 32 B trailer,
  magic dbc0de0c.
- W1: the production wrRow idiom (LDWD prologue + 128-iter branch-looped
  WRITE, slot-0 column rotation — E14's exact shape) → readback over c2h
  BYTE-EXACT vs the INTENT pattern. This is the E14 content test running
  faithfully in sim — the test that would have caught build-11's E14 and
  build-12's wedge pre-synthesis.

Fidelity findings on the way:
- The Vivado build defines POPCOUNT_ACCUM_MODE=1 at the fileset level
  (visible only in the generated runs/*.tcl) — sims must add it or the
  engine's outstanding-reads accounting compiles out.
- MIG lane semantics: a beat's first 32 bytes ride the UPPER 256 bits of
  rd_data; the engine's din half-swap undoes it (the sim DRAM must
  present MIG order, not storage order).
- LDWD lane map confirmed: slot q = wdata[32q +: 32] exactly; the DPI
  write log showed the intent pattern byte-exact at the DRAM boundary.

Next uses: replay captured production request streams (reqcap) through
the sim; extend the DRAM model with charge-sharing lattice ops for
in-sim PIM semantics; every future fetch/frontend/engine RTL change
gates on scenarios S/M/N/R1/W1 before synthesis.

## Build-13 silicon + THE REAL ROOT CAUSE: the wrapper never wired restart

Build-13 flashed (magic 0C confirmed): P1 passed, but armA (single
not-taken BL) wedged — same external pattern as build-12. New probe arm
Z (branch-free second program after an idle gap) also WEDGED →
**branches exonerated entirely**; the failing shape is "any program
after the first park." The maint-pulse fix greens the sim but not
silicon → a fidelity gap remained.

Found it in the wrapper: the QUAD builds **softmc_core.v** (per-channel
wrapper), NOT softmc_top.v — and softmc_core NEVER CONNECTED
`.fetch_restart` (frontend) nor `.restart` (pipeline). Vivado said so
all along: `Synth 8-7071 port unconnected` ×2, present in build-11/12/13
logs, unread (watchdogs grepped ^ERROR only). Silicon has run
restart==0 since build-11:
- build-11: harmless (fetch free-runs; the "loss-window fix" silently
  never reached silicon),
- build-12/13: park-at-END + restart that can never fire = fetch parks
  at the FIRST program END forever. No maintenance subtlety required.
The sim was wired from softmc_top.v (restart connected) — faithful to
the wrong wrapper; that was the sim/silicon divergence.

Confirmation both directions (SILICON_ASIS_UNWIRED_RESTART define in
e2e_top.v): restart tied 0 → wedge class reproduced (everything dies at
first park; sim boot-ZQ's own END parks fetch; silicon's P1 only ever
worked because reset_fpga resets the pipeline → un-parks → pre-build-11
free-run mechanics execute the first program). Restart wired → ALL
GREEN (the standing regression).

**Build-14** = build-13 + three lines in softmc_core.v (wire + two port
connections) + magic 0xDBC0DE0D. New checklist rule: after every build,
grep the log for `Synth 8-7071` on our module ports; and the sim's
wrapper-of-record for the QUAD is softmc_core.v.

Silicon-evidence notes kept honest: trailer counters at P1 showed
cnt_rd≈0..1 over ms-scale gaps — the sim's tPRDI=1 µs maintenance rate
does NOT match silicon's actual per-RD rate (units/params differ
somewhere); irrelevant to this root cause but flagged for the phase-3
fidelity pass. reset_fpga does NOT read an info packet (pure h2c word);
counters ride normal read trailers only.

## BUILD-14 ON SILICON — GREEN SWEEP (2026-07-24 late night)

Magic 0D confirmed. Probe taxonomy: P1 OK, **armZ OK** (branch-free
after gap — the 12/13 killer), armA/B/C all COMPLETED; trailer counters
show per-RD maintenance running continuously between programs
(cnt_rd 0x153→0x808 across one probe) with the conveyor surviving
every park. Ladder: **ALL_PASS incl. E14 clean**
(legacy_vs_model=0; streamed 0/8 rows differ) — the arm build-11
corrupted and 12/13 never reached. Legacy 32-row read 1.5 ms
(wedge era: 14-min hangs); stream 2.60× on raw reads.
The verified-on-silicon stack: loss-window restart (wired at last),
park+fence (E14 stale resolve), maint-conveyor pulse, wrapper wiring.
Next: twin gates (c11/p11), then the ordered walls.

## Build-14 twin gates (late night): legacy floor clean; STREAM+PIPE arm FAILED

- c11 control (PIM_STREAM=0): exactly the established floor (V2 345/345
  element-jitter, MM3D 100/720 straddle band, LOADs clean) — build-14
  legacy is production-safe and now carries the REAL loss-window fix.
- p11 (STREAM=1 + PIPE alternation): req #17 MM3D twin 2041/2048 el
  (large deltas), then handle-9 resident verify DECAY/CORRUPTION
  46.1% of segs (±1 popcount steps, refresh=1 flag), then a 60 s c2h
  stall. Production-shape streaming on build-14 diverges where the
  ladder's stream arms (incl. E13/E14) stay clean — the ladder does not
  represent wcol sessions/sized records/pipe deferral/refresh trains.
- **Wedges are now SELF-HEALING**: a fresh process's reset recovers a
  jammed channel completely (probe all-green on just-wedged bender 2,
  no reboot) — the wired restart changed recovery semantics. No more
  channel-burn budget.
- Isolation in flight: s11 = PIM_STREAM_ALTERNATE only (no pipe).

## Day-end state (2026-07-24 ~night's end)

- **Build-14 legacy: GO for production.** Ladder ALL_PASS (E14 clean),
  probe taxonomy green, twin control at the exact floor, loss-window
  fix genuinely on silicon for the first time, wedges self-heal via
  process restart. Strictly better than every prior build for the
  PIM_STREAM=0 production config (which is the current default).
- **Streamed production shapes: QUARANTINED on build-14.** s11
  (stream-only) and p11 (+pipe) both die within the first few MM3D
  requests: c2h stall at a round-0 receive (60 s, 0/8192) right after
  SEG_POP mode sets; p11 additionally showed a 46%-of-segments
  resident-verify divergence (±1-popcount shaped, refresh-flagged)
  before its stall. V2 requests stream fine. The ladder's stream arms
  (incl. E13/E14) do not provoke it.
- Sim scenario P (parked fetch → maint churn → SEG_POP ×2 → STREAM_EN
  → sized session of full-row reads) PASSES — the approximation lacks
  the killing ingredient. The server's real MM3D flow: ensure_readback
  (SEG_POP) → pexec of ONE big multibank fused program (branch loops,
  M×2048 segpop expected) per round, 64 execs/handle, mode churn
  READ↔SEG_POP between request types, wcol|exec session scoping.
- **Phase-3 (next): capture-replay through the sim** — feed the exact
  reqcap byte stream (the same records replay_ab.py sends) into the
  e2e TB and watch the first stalling record with full signal
  visibility. The sim speaks the full protocol now; this was always
  its designed use.
