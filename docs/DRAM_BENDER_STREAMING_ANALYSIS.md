# DRAM Bender, combed out end to end — and what streaming actually requires

Written 2026-07-25 after a day of patching taught the wrong lesson twice.
The point of this document is to stop guessing: to state what each stage
holds, what it *assumes*, and which assumptions streaming voids. Every
claim marked **[M]** is measured in `e2e_sim` against the SiMRA baseline;
unmarked claims are read from RTL and should be treated as review notes,
not facts, until measured.

---

## 1. The path a program takes

    host --h2c--> frontend --> IMEM --> fetch --> decode --> execute
              --> ddr_pipeline --> DDR PHY
                                      |
                            (25 cycles [M])
                                      v
    host <--c2h-- readback_engine <-- rd_data / rd_valid

Two things enter this path that the host never asked for:

* **maintenance microcode** (periodic read, ZQ, refresh) — injected by
  `maintenance_controller`, muxed onto the SAME fetch path
  (`program_process = EXECUTE_S && ~maint_process`), issuing real DDR
  commands and producing real read data.
* **the readback engine's own trailer**, appended per program.

## 2. The one precondition the whole design rests on

> **One program is in flight at a time, and its results are fully drained
> before the next program starts.**

It is never written down anywhere in the source. Upstream guarantees it
structurally: `h2c_tready = (state == INIT_MEM_S)`, so the host *cannot*
send a program while one is executing. Every program is therefore
separated by a full host round-trip (~150–200 µs), which is orders of
magnitude longer than any drain.

## 3. Stage by stage: state held, and what it assumes

| stage | state it holds | what it assumes |
|---|---|---|
| frontend FSM | `state_r`, maint arbitration | a program ends before the next begins; maintenance only enters from IDLE |
| IMEM | one buffer (upstream) | load and execute are mutually exclusive |
| fetch | `pc_r`, `state_r`, branch resolve | nothing else drives the instruction path while it runs |
| decode/execute | uop pipeline regs | in-order, one program's stream |
| readback: `ignore_read` | 1 bit, one-shot | a maintenance program issues **exactly one** read **[M: false — it issues one per invocation but the flag clears on the first *returning beat*, so any later read leaks]** |
| readback: `ignore_flush_ctr` | 4-bit saturating | maintenance and user flushes interleave in a fixed cadence |
| readback: `proc_flush` | 1 bit | at most one trailer is owed at any time |
| readback: trailer gate | `rbf_empty` | **an empty payload FIFO means the record is complete** |
| readback: `rd_outstanding` | one global counter | only one program's reads are ever outstanding |
| SEG_POP assembler | `seg_sr`, `seg_cnt` | a session ends on a group boundary; never cleared between sessions **[M: false — first 64-byte group of a new session was a blend]** |

Every one of these is *correct under the precondition*. Not one of them
is a bug in the original design.

## 4. What streaming voids — measured

Streaming (ping-pong IMEM, build-9) removed the precondition and replaced
it with nothing. Consequences, each measured against the baseline:

1. **Origin is unknowable downstream.** `maint_process` is a
   *frontend-stage* signal; reads issue several stages later and return
   25 cycles after that. **[M]** At the command bus with the frontend
   signal: `user=465, maint=0` — every maintenance read misattributed.
   With the origin carried through the pipeline (build17):
   `user=32, maint=392` — exact.
2. **Announcements are fetch-order, not return-order.** They are also
   made only by user programs **[M: `user=32, maint=0`]**. Adding them to
   maintenance microcode makes the stream complete **[M: `maint=433`]**
   but does not fix ordering, because fetch order and command-bus order
   diverge the moment maintenance interleaves — costing exactly 2 beats.
3. **"FIFO empty" is not a record boundary.** **[M]** Streamed trailer
   magics land at 640/1184/1728/2144 where legacy gives
   512/1056/1600/2144: the first trailer is 2 beats late, everything
   shifts 128 B, the last record is 128 B short. Total length is correct
   — a pure boundary shift, not corruption.
4. **Maintenance c2h leakage.** **[M]** Baseline emits *zero* c2h bytes
   from maintenance; ours emitted ~154 KB per inter-program gap before
   the origin fix, 0 after (2176 B total, byte-exact).

## 5. The minimal correct modification set for streaming

Not a patch list — the contract that replaces the precondition:

1. **Every read carries its origin from where it is known to where it is
   used.** Tag at the frontend mux, carry one bit per pipeline stage,
   join at the command bus. Issue order is return order, so the queue is
   correct by construction. *(build17, done, measured exact.)*
2. **A record's extent is defined by its own content, not by idleness.**
   The record is the set of user-labelled reads issued between one
   program start and the next; its trailer is due when exactly that many
   payload beats have been delivered. `rbf_empty` must not appear in the
   trailer condition. *(designed, not yet built — this is the remaining
   128 B.)*
3. **Per-record state must be per-record.** `rd_outstanding`,
   `proc_flush`, `ignore_read` are global and must be replaced by
   per-record quantities anchored at program start, not tuned.
4. **The bank swap must be acknowledged, not timed.** Release the next
   program when the previous record is closed on the wire, not 32 cycles
   after its fin.
5. **The stream should be self-describing.** The trailer already carries
   a magic; it should carry a record index so a lost delimiter is
   *detectable* by the host instead of silently shifting everything.
6. **Delete what the contract replaces.** `ignore_flush_ctr`, one-shot
   `ignore_read`, the single-bit `proc_flush` are heuristics for a world
   that no longer exists. Leaving them alongside the new structure means
   two sources of truth — which is what made six of today's patches
   produce byte-identical output.

## 6. Verdict on our accumulated changes

* **Keep — earns its place, baseline-verified:** trailer magic/counters
  (record identity), SEG_POP readback (4× bandwidth), DIFF/ACCUM
  accumulation, buffer-space conservation, the origin bit (build17), the
  SEG_POP assembler clear, level-based maintenance suppression.
* **Delete — heuristics guessing at knowable state:** `ignore_flush_ctr`,
  one-shot `ignore_read`, single-bit `proc_flush`, and my six symptom
  patches from this session.
* **Insufficient — shipped without its contract:** the ping-pong IMEM.
  It is the right idea and the measured win is real, but it was merged
  with no record identity, no per-program accounting, no swap
  acknowledge, and framing still derived from an idle FIFO.

## 7. How to work on this from here

The gate exists now and it is cheap: **in legacy cadence our c2h stream
must be byte-identical to the SiMRA baseline** (it is, as of build17),
and **in streaming cadence it must be byte-identical to our own legacy
output** (it is not — 128 B boundary shift). Anything that cannot be
stated as a delta against one of those two references is not yet
understood well enough to implement.
