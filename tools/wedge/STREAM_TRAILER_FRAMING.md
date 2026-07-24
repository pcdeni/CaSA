# Streamed-MM3D wedge on build-14 — investigation ledger

Started 2026-07-24 late night. Directive: methodical, rigorous,
structured; proper fix over patch. No deferrals.

## 0. Problem statement

On build-14 (magic 0D, the first build with the full fetch protocol —
park-at-END, branch fence, WIRED restart pulses, fin/tlast harden —
actually in fabric), PRODUCTION-SHAPE STREAMING fails while every
synthetic gate passes.

## 1. Fact base (all from 2026-07-24 runs; logs in scratchpad + repo)

F1. Ladder ALL_PASS incl. E14 clean (streamed branch-loop writes).
F2. Probe taxonomy green: armZ (branch-free after gap), armA/B/C.
F3. c11 legacy twin gate (1,100 records): exact established floor
    (V2 345/345 element-jitter, MM3D 100/720 straddle, LOAD clean).
F4. s11 (PIM_STREAM_ALTERNATE=1, no pipe): dies within the first ~4
    MM3D requests. Server: "Set readback mode: SEG_POP" ×2 →
    receiveData TIMEOUT 60 s, 0/8192, at "(round=0 bp=[0..1))".
    345 V2 requests before the MM3Ds streamed FINE.
F5. p11 (STREAM=1 + PIPE_ALTERNATE): first twin diff at req #17
    (MM3D): 2041/2048 elements, LARGE deltas (not straddle-shaped);
    later handle=9 resident verify "DECAY/CORRUPTION" 46.1% of segs,
    ±1-popcount-step shaped, refresh=1 flag; then the same 60 s stall.
F6. Same replay arms on build-10 (restart UNWIRED silicon): clean over
    the full 4,288-record capture + the −29.2% e2e wall run.
F7. e2e sim scenario P (approximation: parked fetch → maint churn →
    SEG_POP ×2 → STREAM_EN → sized session of 4 full-row reads):
    PASSES. The approximation lacks the killing ingredient.
F8. Sim scenario N (streamed branch loops + live maint, unsized,
    payload-0): PASSES.
F9. Silicon maintenance rate ≠ sim rate: trailer counters showed
    cnt_rd advancing ~0x150 per inter-program gap (probe), i.e. maint
    IS running frequently on silicon between host programs; but the
    absolute rate differs from the sim's tPRDI=1 µs derivation
    (fidelity note, not yet load-bearing).
F10. Wedges self-heal via process restart on build-14 (restart wired).

## 2. What changed build-10 → build-14 on the streamed path

Fetch: park-at-END; br_outstanding fence; restart input LIVE (was
tied 0). Frontend: restart pulses at legacy tlast, stream swap, maint
entry; fin/tlast same-cycle harden (build-11 code, in fabric for the
first time). RBE: magic only. On build-10 silicon none of the pulses
existed and fetch free-ran/re-looped after END.

## 3. Hypotheses (each with its discriminating experiment)

H1. **Maintenance read-beats pollute armed sized records.** Per-RD
    microcode issues DDR READS; its beats enter the RBE. If the RBE's
    ignore/announce accounting assumed maint never overlaps an armed
    window — a property the old free-run timing may have provided —
    then on build-14 (maint pulses between streamed programs) maint
    beats can land inside an armed MM3D window: count desync → drain
    waits forever (STALL, F4) and/or beats interleave into records
    (CORRUPTION, F5). Experiment: read the RBE announce/ignore logic
    COMPLETELY (design intent); then capture-replay — the FST shows
    beats vs window arming directly. Sim can also force-fire maint at
    chosen phases.
H2. **Big multibank programs vs streamed IMEM banking.** MM3D exec =
    ONE fused multibank program (branch loops, large). If the
    ping-pong banks halve effective IMEM and the host gate checks
    8192, a large program silently truncates only when streamed.
    Experiment: instruction counts of the real MM3D programs
    (PIM_DUMP_MM3D_PROGRAMS=1 exists server-side) vs bank capacity in
    frontend RTL. Cheap, do early. (Weakened by F6 — build-10 streamed
    the same programs — unless program sizes differ per arm.)
H3. **Mode churn / session lifecycle vs park-restart.** READ↔SEG_POP
    SET words between requests, sessions opening on a parked fetch,
    wcol|exec scope mixing. Experiment: the capture-replay contains
    the exact word order; sim shows the FSM/fetch state at each.
H4. **fin/tlast harden misfire on the streamed path.** The build-11
    harden (`loaded_r[load_bank] || (imem_wr_fire && h2c_tlast_0)`)
    is in fabric for the first time. Experiment: visible in the same
    replay FST at swap boundaries.

## 4. Method (the proper instrument)

Step 1: PIM_H2C_CAPTURE hook in platform.cpp sendData path — dump
  every h2c burst (len + bytes; each sendData call = one tlast
  boundary, matching XDMA descriptor semantics). Opt-in env; zero
  behavior change unset.
Step 2: Rerun the dying arm (s11) on silicon with capture armed →
  the exact byte stream up to and including the stalling MM3D.
Step 3: Sim TB replay mode: consume the capture, drive the identical
  beats/tlast into the frontend; c2h always-ready; full FST.
Step 4: Repro gate: sim must stall at the same program index. If it
  does NOT: enumerate remaining sim-vs-silicon deltas (maint rate/
  phase sweep first per F9), close them one at a time — each closure
  is a fidelity lesson recorded here.
Step 5: Mechanism read off the FST → fix designed against the RBE/
  frontend design intent → sim regression (replay + S/M/N/P/R1/W1 +
  as-is arm) → build-15 → silicon: probe taxonomy, ladder, twin gates
  (c/s/p), THEN the walls.

## 5. Log

- [ ] Step 0: RBE announce/ignore/maint design-intent read (H1 truth)
- [ ] Step 0b: MM3D program sizes vs IMEM banking (H2 kill/confirm)
- [ ] Step 1: capture hook
- [ ] Step 2: silicon capture of dying s11
- [ ] Step 3: sim replay harness
- [ ] Step 4: repro
- [ ] Step 5: mechanism → fix → build-15 → full regression

## 6. Findings so far (updates to §1/§3)

F11. Capture map (h2c_s11_dying.cap, 6,388 records, 189 MB): largest
     program 1,518 insts vs 8,192/bank ping-pong capacity (frontend
     instantiates TWO full imem's) → **H2 KILLED.**
F12. Murder scene (records 6363–6387): 20× 1,034-inst programs (the
     V2 tail), then SET_READ ×2, a 28-inst READ-mode program (the
     mm3d-verify rdRow), then SET_SEGPOP ×2 — capture ENDS there.
F13. Server-log order (full context): handle-2 verify shows 92.7%
     WHOLE-WORD GARBAGE (0x80852114 vs 0x0323208a — misaligned-stream
     shaped, NOT ±1 decay steps; F5's "±1" sample was handle-9/p11) →
     mode churn (SEGPOP/READ ×2 cycles) → 20 s TIMEOUT on an
     8192-byte READ-mode receive → poison. The "receive with nothing
     sent" was post-poison zombie flow (execute() SKIPPED) — resolved,
     not a separate mechanism.
F14. mm3d-verify is NOT new/config: 5,937 verify lines in the CLEAN
     build-10 streamed arm (pg5) and in build-14 legacy (c11). The
     churn itself is survivable on build-10 streaming and on build-14
     legacy. The kill needs {build-14 RTL} × {streamed segpop MM3D}.
REVISED primary hypothesis: the streamed SEG_POP exec session mangles
RBE record framing/accounting on build-14 (misaligned garbage on the
next reads, then starvation) — H1 (maint beats vs announce contract)
and H3 (mode churn vs session lifecycle) are the surviving candidate
mechanisms; H4 (fin/tlast harden) still open.

## 7. Replay-instrument fidelity lessons

R1. Inter-record gaps must allow maintenance (silicon: µs–ms; sim
    per-RD needs ~166 idle cycles). Run 1 used 50 → no maint at all.
R2. Boot: the sim's config-time boot-ZQ overlaps record 0 if replay
    starts at calib-raise; silicon processes start long after config.
    Fix: idle-until-quiet before record 0 (run 2 does this).
R3. FST from record 0 is dead weight — gate tracing to the tail
    (CLI arg; run 2 traces from record 6,300).

## 8. Replay triangulation results (runs 3/4) + instrument pivot

- Run 3 (tail-start rec 6000, instant drain): NO STALL through the
  murder scene.
- Run 4 (tail-start, drain throttled 128-beat bursts / 8k-cycle
  pauses): NO STALL. Drain-backpressure alone (at this profile, from
  tail-start state) is not the ingredient.
- Run 2 (full history) still running (~rec 2000/6388) — tests
  early-state dependence.
- DECISION (user-proposed instrument): pivot to the SILICON side —
  port the FFT project's DIY URAM ILA (hdl/debug/uram_capture.v:
  288-bit probe, 4096 deep, host-settable triggers, AXI-Lite readback,
  xpm_cdc dual-domain, v71 battle-tested) into the QUAD as
  build-15-dbg (magic 0E, functional RTL identical to build-14).
  Readback path (AXI-Lite user BAR) shares NOTHING with the c2h
  machinery under suspicion. Capture a dying s11 on real silicon →
  decode → diff against the sim FST at the same records = the
  divergence point, directly.
- Integration facts: XDMA AXI-Lite master BAR currently DISABLED (no
  /dev/xdma0_user, no m_axil in the top) → IP reconfig required
  (CONFIG.axilite_master_en/size/scale). Probe map ~164b fits one
  core. Trigger v1 = SET_READ/SET_SEGPOP acceptance pulses, deep
  pre-trigger.

## 9. BREAKTHROUGH — minimal reproducer, mechanism localized (07-24 late)

The heavyweight instruments were the wrong first move. A standalone
tool (`stream_mm3d_probe.cpp`, BitNet dir) that replays captured
instruction words with NO server reproduces the failure in ~10 seconds.

### The reproducer
28-instruction row read (a plain `rdRow`), SEG_POP mode, streamed
N times back-to-back in one session, then received:
  N=3 -> last record returns 2016/2048
  N=4 -> 1984/2048
  N=8 -> 1792/2048
Legacy dispatch of the identical program: 4/4 clean, 0.12 ms each.

### The arithmetic (this is the mechanism)
`PIM_RECV_DEBUG` shows the wire total for N=3 = **6176 bytes**.
  3 payloads (3x2048=6144) + **ONE** 32-byte trailer = 6176. Exact.
The host's drain parses **PAY+32 per record**. So it eats 32 bytes of
record 1's payload as record 0's trailer, drifts 32 B per record, and
starves the last record. Predicted last-record size under a
one-trailer-per-SESSION wire is `2112 - 32N`:
  N=3 -> 2016 (observed 2016), N=4 -> 1984 (observed 1984).
Both exact. N=8 predicts 1856, observed 1792 (64 further short) — a
second effect stacks at longer sessions.

### Factors ELIMINATED (each by direct experiment)
- Program content: a locally generated read (s4_read128) fails identically.
- Program size: the 1483-inst fused MM3D body is not needed; 21-28 insts do it.
- Session mode: `STREAM_SIZED` vs uniform `stream_start(payload)` — identical
  failure. So the HOST parser is not at fault; the wire is short.
- The server, request history, mode churn, V2 prefix: all absent here.
- MM3D specifically: it is any streamed SEG_POP read session.
- Maintenance-eats-trailers (my first hypothesis): REFUTED — more idle
  time makes it BETTER, not worse.

### Factor that DOES matter: session length (2D sweep)
  N=4:  gap 0 FAIL | gap 500us CLEAN | gap 3000us CLEAN
  N=8:  gap 0 FAIL | gap 500us FAIL  | gap 3000us FAIL
  N=16: FAIL at every gap
Spacing rescues short sessions only; beyond ~4 outstanding records
nothing helps. Trailer framing collapses as un-drained records
accumulate — the host sends all N before receiving any, so the RBE
carries N records' worth of announcements/credit.

### Why every synthetic gate missed it
The ladder's stream arms build each Program inside the send loop
(string labels, per-row construction), which spaces sends enough at
their N. Production streaming, and especially phase-2 send-ahead,
deliberately removes that spacing — which is exactly why the wedge
appeared when streaming went to production shape.

### Next (in order)
1. Reproduce in e2e_sim: stream 8 sized reads, COUNT c2h trailers
   (scenario P at N=4 passed and must be re-run at N=8 with trailer
   accounting). The RBE + frontend are already in the sim.
2. Read the trailer-framing path against this: `flush = frontend_ready`
   (softmc_fin delayed, multi-cycle), `flush_edge`, `proc_flush`,
   and the outstanding/credit accounting.
3. Fix in RTL so framing is per-program, not per-flush-edge; regress in
   sim; build-16.
4. Interim software mitigation exists (cap outstanding records at <=4
   with spacing) but it forfeits the streaming win, so it is a
   fallback, not the fix.
