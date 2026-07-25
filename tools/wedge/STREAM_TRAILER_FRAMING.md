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

## 10. Sim reproduction achieved; three hypotheses killed by measurement

**Reproduced in e2e_sim (scenario Q).** 8 streamed SEG_POP row reads,
back-to-back, with the c2h drain THROTTLED:
  throttle=0  (instant drain): 8/8 messages, 16640/16640 B  -> OK
  throttle=40 (paced drain)  : 7/8 messages, 16608 B        -> REPRO
Instant drain was exactly why scenario P passed earlier: a trailer that
leaves immediately always clears before the next program ends. Silicon
has XDMA/PCIe/FIFO latency; the sim needed pacing to be faithful.
**Fidelity lesson: an always-ready consumer hides every framing bug.**

**Instrumented census (SIM_TRAILER_DEBUG, 1M cycles):**
  flush high-cycles 6007 == rising edges 6007   -> no edge merging
  processed 102 == tlast 102                    -> 1 trailer per
                                                   processed flush
So the framing chain from `flush_proc` onward is exactly 1:1, and the
loss is upstream of it or downstream of `tlast`.

**Hypotheses tested and REJECTED (each with data, not argument):**
1. `proc_flush_r` single-bit merge — implemented a saturating
   pending-trailer counter. Trace shows every flush edge is serviced
   immediately (`pend=0` -> TLAST); nothing was ever queued. Result
   unchanged (16608).
2. Maintenance eats user flushes via the `ignore_flush_ctr` heuristic —
   replaced the heuristic with a per-flush ORIGIN bit carried from the
   frontend (`frontend_ready_maint`, sampled at fin, delayed with
   delay_fin). Maintenance flushes are correctly suppressed (no trailer
   flood from 9118 maint events). Result unchanged (16608).
3. Same-cycle race tagging a user fin as maintenance — sampled the
   origin one cycle before the fin. Result unchanged (16608).
The byte count is IDENTICAL across all three, which is itself evidence:
none of these paths is the mechanism.

**Live lead (next):** the RBE's internal `tlast` count (102) is what the
engine *asserted*; the TB counts wire transfers
(`c2h_tvalid && c2h_tready && c2h_tlast`) and saw one fewer in the Q
window. So a trailer beat is very likely ASSERTED but never TRANSFERRED
under backpressure — a tvalid/tready handshake problem at the trailer
beat itself, not a framing-decision problem. Next experiment: count
tlast-with-tvalid vs tlast-with-tvalid-and-tready in the TB, and if
they differ, read `trailer_beat`/`c2h_tvalid_0` around the loss in the
FST.

**Status of the three RTL changes:** all are defensible robustness
improvements (a counter instead of a flag; origin instead of a
heuristic; no same-cycle sampling race) and all are currently in the
SIM tree only. None is validated as the fix, so none goes into a
bitstream yet.

## 11. build16: per-record framing implemented (partial)

**Root cause, stated precisely.** The record delimiter was positioned by
FIFO idleness plus GLOBAL counters. Nothing in the engine could answer
"is THIS record finished". Consequences, all reproduced:
  - trailer late  : next record's payload arrives first -> delimiter at
                    4096 instead of 2048 (silicon's signature)
  - trailer early : read credit throttles the DDR reads, FIFO drains
                    mid-record -> delimiters every 1216 B
  - merged pairs  : a global rd_outstanding gate waits for the NEXT
                    record's reads too
Six patches to WHEN the flush is processed could not fix a defect in
WHERE the boundary is.

**Implemented (readback_engine.v, build16):**
1. Per-record accounting: `read_seq_incoming`/`incoming_reads` are
   accumulated between flushes; at a processed (user) flush the total is
   converted to c2h beats (READ: reads*2, SEG_POP: reads/2) and queued;
   the trailer is emitted when exactly that many payload beats have
   been transferred. Queue depth 8 keeps overlapped records delimited.
2. `.rd_en(c2h_tready_0 && ~trailer_beat)` — the payload FIFO was popped
   on EVERY ready cycle including the trailer's, which would discard a
   payload word. This coupling was the ONLY reason the trailer had to
   wait for an empty FIFO; removing it is what allows a boundary that
   does not depend on quiescence.
3. Maintenance attribution: `in_maint` excludes maintenance
   announcements, and maintenance flushes no longer close the user
   accumulator.
Plus (earlier, kept): flush origin bit, swap acknowledge, pending
counter — each defensible on its own, none sufficient.

**Validation state (scenario Q, 8 streamed SEG_POP row reads):**
  fast drain        : 8/8 messages, magics EXACTLY at 2048,4128,...,16608
                      -> per-record framing is correct, no regression
  burst backpressure: 6/8, first magic at 6176 = 3 payloads + 1 trailer
                      -> the first expectation covers THREE records,
                         i.e. two user flushes did not close the
                         accumulator. NOT FIXED.
  all other scenarios (S1, N, R1/W1 payloads byte-exact) still pass;
  R1/W1 "FAIL" is a stale hard-coded magic 0C check in the TB.

**Next (do not guess again):** the accumulator is not closing on every
user flush under backpressure. Instrument `rec_acc_r`, `recq_wr/rd` and
`flush_proc` in the WAVEFORM at the first three flushes of the throttled
window and read why. Do not add another speculative patch.

**Fidelity caveat worth resolving first:** the sim runs ~1127
maintenance programs per user program; silicon's trailer counters imply
~170. Same order, but 6x. Every interleaving conclusion drawn here
should be re-checked at silicon's real maintenance rate.

## 12. THE LEAK — a real defect found and fixed (waveform-derived)

Reading `rec_acc_r` / `rec_sent_r` in the waveform (as instructed, not
guessed) showed each record emitting **66 c2h beats where 64 were
announced** — exactly +2 beats = one 512-bit DDR read = 64 bytes of
foreign data per record, drifting the delimiter by a whole record after
a few programs.

Source: maintenance read suppression was ONE-SHOT.

    if(per_rd_init || per_zq_init || per_ref_init) ignore_read_ns = per_rd_init;
    if(rd_valid_r) ignore_read_ns = `LOW;      // cleared after ONE beat

Only the FIRST read of a maintenance program was discarded; every later
maintenance read LEAKED into the user's payload stream. Because the
readback path carries no per-instruction metadata, a foreign beat is
indistinguishable from payload.

FIX (build16): suppress for the DURATION of maintenance —
`ignore_read_ns = in_maint;` (plus the init pulses). Result: byte totals
became EXACT (16640/16640) under both drain models, where they were
16608 before. This defect is independent of the framing question and
would corrupt legacy reads too whenever maintenance issues >1 read.

## 13. Why the per-record counter cannot work as built (structural)

Attempted a quiescent resync of `rec_sent_r`; it made things worse
(R1 payload corrupted, Q stalled) and the reason is structural, not a
tuning error:

  **the expectation only becomes known at the FLUSH, which happens
  AFTER the record's payload has already been transferred.**

Counting "payload beats since the last trailer" against an expectation
that arrives at the end is inherently racy, and the residue observed
(`sent=1466` vs an expectation of 64) is that race, not an accident.

CORRECT DESIGN (for build16 proper): establish the record boundary at
PROGRAM START, not at its end. `fetch_restart` already pulses at every
program start (legacy load, stream swap, maintenance entry) and
`in_maint` distinguishes the origin. On program start: close the
previous record, zero the per-record counters. During the program:
accumulate announcements and returns. The record is complete when its
own announced reads have all returned and its payload has drained —
all indices per-record, none global.

STATE: leak fix = keep (independent win, byte-exact). Per-record
counter = revert to inert (harmless) until rebuilt around program-start
boundaries. Fast drain 8/8 exact; burst backpressure 6/8.

## 14. Answers to the three questions (evidence, not opinion)

**"Byte totals are exact — but are they correct?"  NO.**
16640 = 8*2048 payload + 64 (one leaked 512-bit read = 2 beats) + 192
(SIX trailers). A leak and two missing delimiters cancelled in the sum.
Counting bytes is not verification. A content check was added: the 8
records run the SAME program on the SAME row, so every payload must be
byte-identical to record 0.

**"Why 6/8 and not 8/8?"** Two records have no delimiter; the byte
total hid it because the leaked 64 B exactly offset the two missing
32 B trailers.

**NEW, and worse than the framing bug:** with the content check in
place, even the FAST-DRAIN case — where framing is now provably perfect
(magics at exactly 2048, 4128, ... 16608) — reports 7/7 records
differing from record 0. Identical programs, identical row,
deterministic DRAM model: the payloads must match and do not. So there
is payload corruption INDEPENDENT of delimiter placement. Framing was
never the whole story.

## 15. The origin-tag design (user's proposal) — right, partially built

Proposal: tag/gate the returning data by origin (user program vs
maintenance), so we know when one IMEM bank's program is done and the
swap is safe. This is correct and is the only structure that fixes the
class: the readback path carries no per-instruction metadata, and DDR
returns are in order, so the announcement stream can label them:

    announce (read_seq_incoming, incoming_reads, in_maint)
        -> push {count, origin}
    return  (rd_valid)
        -> decrement head; pop when it empties
    => origin of the beat returning NOW, and
       "this program's reads have all returned"

BUILT: the (count, origin) queue, payload FIFO gated on user beats.
NOT WORKING YET: completion still used a RUNNING counter
(`user_outstanding_r`) that spans programs, so the next program's
announcements block the current record. Anchoring it at program start
(fetch_restart) produced byte-identical output = the path as wired is
not causal; it needs to be built deliberately.

STOPPED iterating here. Nine RTL attempts in this session; the last
several produced byte-identical results, which means the edits are not
in the causal path and more patches are waste.

## 16. State kept

- Maintenance read leak fix (level-based suppression): KEEP. Independent
  defect, byte totals became exact, would corrupt legacy reads too.
- Content self-consistency check in scenario Q: KEEP. It is the only
  check that would have caught the payload corruption.
- Per-record counter + origin queue: present but inert/incorrect;
  documented above. Do not ship.
- Fast drain: framing perfect. Burst: 6/8. Payload: wrong in BOTH.

## 17. Payload corruption — localized, mechanism found, partially fixed

CORRECTION to §14: the payload is NOT generally corrupt. With correct
framing (fast drain) the records classify as:
    r0 = class 0, r1..r7 = class 1   (records 1-7 byte-IDENTICAL)
Only the FIRST record differs, and every differing byte lay in
bytes 0..63 — exactly ONE 64-byte SEG_POP group.

**Mechanism.** SEG_POP assembles four 16-byte beats into a 512-bit word
through a shift register:
    seg_sr <= {seg_beat_bytes, seg_sr[511:128]};
    if(seg_cnt == 3) begin seg_word <= ...; seg_cnt <= 0; end
`seg_sr` and `seg_cnt` are reset ONLY on hard reset — never on mode
entry. A session that ends mid-group leaves residue, and the next
session's first 64-byte group is a blend of both. The server churns
READ<->SEG_POP constantly, so on silicon this corrupts the first group
after every mode switch.

**Partial fix applied**: clear seg_cnt/seg_sr on `set_mode_segpop`.
Corruption window shrank 43 bytes -> 13 bytes and moved from byte 0 to
byte 32 (beats 2-3 of the group). Remaining residue is almost certainly
`seg_word` (the output register) and the `seg_beat_valid` pipeline
stage, which are likewise cleared only on hard reset. Finish by clearing
the whole SEG_POP datapath on mode entry.

**Records 1-7 stay byte-identical throughout** — the defect is confined
to the first record of a session, which is exactly why it survived every
steady-state test.

## 18. Session close — three defects, two fixed

1. Maintenance read leak (one-shot suppression) — FIXED, byte-exact.
2. SEG_POP group-assembler residue — mechanism found, PARTIALLY fixed
   (43 -> 13 bytes); finish by clearing the rest of the datapath.
3. Framing under backpressure (6/8 delimiters) — OPEN. Design settled:
   tag returns by origin (announcement queue, in-order returns) and
   anchor the record boundary at PROGRAM START, not at the flush.
   Do not add more patches to the existing global heuristics; build the
   per-record structure and delete the heuristics it replaces.

Method note for whoever continues: byte totals are not verification
(16640 was a leak plus two missing delimiters cancelling). The content
self-consistency check — identical programs must yield identical
payloads — is what found defects 1 and 2. Keep it in every gate.

## 19. SEG_POP clear COMPLETE; ILA parked; drift measured

SEG_POP datapath now clears fully on mode entry: seg_beat_valid, seg_sr,
seg_cnt, seg_word, seg_word_valid (seg_beat_bytes is combinational).
Window went 43 bytes -> 13 bytes; the residual 13 (bytes 32..44, i.e.
beat 2 of the first group) is therefore NOT assembler residue and needs
the golden reference to attribute.

ILA / AXI-Lite: PARKED by decision. Root blocker recorded for whoever
returns: the upstream generate.tcl calls reset_target+generate_target on
every IP, which regenerates XDMA from its .xci and reverts the
AXI-Lite-master customisation (Basic mode wins), removing the m_axil
ports every build. A no-regeneration build flow (build_noregen.tcl,
written) is the way in if it is ever wanted. It was adding unknowns
faster than it removed them.

## 20. How far our engine has drifted from upstream (measured)

    readback_engine.v : 271 -> 1017 lines  (904 changed)  3.8x
    frontend.v        : 261 ->  547 lines  (396 changed)  2.1x
    fetch_stage.v     : 147 ->  193 lines  ( 52 changed)  1.3x

Upstream DRAM Bender is a request/response machine with ONE implicit
precondition: a single program in flight, fully drained before the next.
Every mechanism we have been fighting is correct under that precondition
and only that one --
  * trailer emitted when the payload FIFO happens to be empty
  * maintenance read suppression armed for exactly one beat
  * SEG_POP group assembler never cleared between sessions
  * single-bit "trailer owed" flag
  * saturating maintenance-vs-user flush heuristic
None of these is a bug in the original design. All of them are bugs the
moment programs overlap, which is exactly what streaming introduced. The
precondition was never written down, so each graft (DIFF, SEG_POP,
ACCUM_XBP, buffer_space conservation, ping-pong IMEM) was locally
reasonable and cumulatively violated it.

## 21. GOLDEN REFERENCE (next instrument, user's proposal)

Run the SAME command sequence through UNMODIFIED DRAM Bender and record
(a) the DDR command stream at the PHY and (b) the c2h byte stream. Use
it as ground truth and comb our design against it.

Value: it separates "our modifications broke this" from "it was always
so", which is exactly the question every defect above raised and none of
today's experiments could answer.

Concrete plan:
1. pristine tree extracted: golden_ref_2026_07_25/pristine/
2. build e2e_ref: pristine frontend/fetch/pipeline/readback in the SAME
   e2e_top harness, same DRAM model, same programs (s1/s4 hex).
3. record per run: ddr_act/read/write/pre + bank/row/col trace, and the
   full c2h byte stream.
4. GATE (new, and we have never had it): in LEGACY cadence our engine
   must produce a BYTE-IDENTICAL c2h stream to pristine for the same
   program. Any delta is ours, named and justified or fixed.
5. Only then re-open streaming, where pristine has no reference and the
   precondition must be replaced by an explicit contract (origin-tagged
   returns + record boundary at program start).

## 22. GOLDEN REFERENCE BUILT — first diff, three findings

Pristine upstream frontend + pipeline + readback_engine in the SAME
harness (same DRAM model, IMEM, FIFO, TB, program). Legacy cadence,
READ mode, 4 x s1_read.hex (8 reads each).

    pristine : c2h = 2176 B  = 4 x (512 payload + 32 trailer)  EXACT
    ours     : c2h = 618816 B                                  284x

1. **Our user-program payload is byte-identical to pristine.** The
   streams agree up to byte 513 — i.e. record 0's entire 512-byte
   payload matches. Execution, addressing and read data are correct;
   the divergence is entirely in the readback framing/gating.

2. **Maintenance produces ZERO c2h bytes in pristine and ~154 KB per
   inter-program gap in ours.** Our trailers land at 512, 155232,
   309952, 464672: record 0 is correctly framed (512+32), then ~154 KB
   of maintenance-generated traffic precedes each subsequent record. A
   host expecting one record per program cannot survive that; it is the
   desync mechanism, measured.

3. **DDR command RATE is identical** (0.018 cmd/cycle both), so
   maintenance runs the same in both designs. Only the readback
   treatment of it differs. Our engine is not doing more work — it is
   failing to discard work that is not the user's.

### Why our maintenance suppression still leaks
`ignore_read_ns = in_maint` samples the origin AT RETURN TIME, but read
data returns ~24 cycles after issue: a maintenance program's last reads
come back AFTER maint_process has dropped, so they are scored as user
payload. The origin must travel WITH the read, not be sampled when it
lands — which is exactly the announcement-queue design (push
{count, origin} on read_seq_incoming, pop as beats return). That design
was right; sampling a level at return time was not.

### The gate we never had, now available
"In legacy cadence our engine must produce a c2h stream byte-identical
to pristine for the same program." Today it produces 284x the bytes.
That is now a measurable, closeable target rather than an opinion.

## 23. ORIGIN TRAVELS WITH THE READ — leak FIXED, verified against baseline

Provenance corrected first: the chain is vanilla DRAM-Bender (271 lines)
-> **SiMRA fork (244)** -> ours (1017). We forked from SiMRA, so SiMRA
is the true baseline. Both baselines emit the SAME 2176 B for the test,
so the reference is anchored either way.

Fix: label each read at its ANNOUNCEMENT (`read_seq_incoming` +
`incoming_reads` + `in_maint`) and consume the labels in order as beats
return. DDR returns are in order, so the label travels with the read
instead of being re-derived at return time (which is what leaked: reads
come back ~24 cycles after issue, so a maintenance program's last reads
landed after `maint_process` dropped and were scored as user payload).
Scope kept narrow: PAYLOAD gating only, framing untouched.

RESULT, same harness / same program / legacy cadence:

    baseline (SiMRA) : 2176 B, 72763 cycles
    ours BEFORE      : 618816 B (284x), 1660272 cycles
    ours AFTER       : 2176 B, 72762 cycles

Record-by-record against the baseline:

    record 0..3: payload MATCH (all four, byte-exact)
                 trailer differs -- ours carries magic 0xDBC0DE0E plus
                 debug counters where the baseline has zeros. That is an
                 INTENTIONAL feature of our engine, not a defect.

So in legacy cadence our engine is now byte-identical to the baseline on
every payload byte, with framing identical (4 x 544) and timing within
one cycle. The maintenance leak is closed, and closed with evidence
rather than assertion.
