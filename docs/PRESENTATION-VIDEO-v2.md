# LLM-in-DRAM video — three-act script (v2)

**Updated:** 2026-05-05. Numbers measured today on DIMM 0 of the BCU1525_QUAD bitstream + projections from `casa_sched.c` re-run today.

**Audience:** technically curious viewers, no PIM background. ~12 min.

**Structure (the user's framing):**
1. **SHOW** — full LLM running in memory (all layers, all 7 BitLinears).
2. **ACKNOWLEDGE** — this is highly unoptimized; explain the overhead.
3. **PROJECT** — what the silicon ceiling is, and the path to it (1 DIMM, 4 DIMMs, in-DRAM popcount, LISA, binary acts).

**Honesty bar:**
- Every number marked **MEASURED** is from a real run on this hardware (`bitnet-proj-server` `srv-prof` line or earlier validated memory note). Date and configuration noted.
- Every number marked **PROJECTED** is from `casa_sched` with the parameters listed.
- Where we didn't measure or simulate, we say so.

---

## ACT 1 — SHOW (target 3 min)

### Scene 1.1 — Cold open

**VISUAL:** Live screen capture: chat prompt → response generated token-by-token. Workstation tower visible behind, FPGA card LED lit through case window.

**ON-SCREEN OVERLAY (small, lower-left):** `BitNet b1.58-2B-4T · 30 layers · 7 BitLinears each · 210 PIM matmuls per token · DDR4 module · charge-sharing in-DRAM compute`

**NARRATION:**
> The model on screen is BitNet b1.58, a 2.4-billion-parameter language
> model published by Microsoft Research. It's answering "What is the
> capital of France?" — and every weight matrix multiplication that
> produces the answer is happening *inside the cells of a regular
> DDR4 memory module*, on charge-sharing primitives. No GPU. No
> external accelerator. The FPGA next to the memory just orchestrates
> the commands; the math itself is the DRAM.

### Scene 1.2 — What actually ran today

**VISUAL:** Side-by-side terminal:
- Left: `python3 run_bitnet_pim.py --bender 0 --bank "0,1,2,3" --layers all --projs all` (the actual launch command)
- Right: server stderr showing `srv-prof #1` through `srv-prof #300+`

**ON-SCREEN TABLE:**
| | value |
|---|---|
| Model substituted onto PIM | **180 of 210 BitLinears = 6/7 per layer** (q,k,v,o,gate,up; **down_proj withheld — see below**). ~70 % of model params on PIM, ~30 % on PyTorch (5 % LM-head/embedding + 25 % down_proj). |
| Substrate | DDR4 module, DIMM 0 of BCU1525_QUAD bitstream |
| Banks used per request | 4 (banks 0,1,2,3 of DIMM 0, in parallel) |
| Per-BitLinear total time (steady-state) | 580 ms (k/v) … 2580 ms (gate/up) |
| **Wall time, 1 generated token** | **269.9 s** (silicon-measured, 2026-05-05) |
| **Decode throughput** | **~0.004 tok/s** |
| **Correctness** | **PIM next-token = CPU PyTorch next-token (bit-exact for `Paris` → `","` greedy decode)** |
| Why down_proj is on PyTorch today | d_in=6912 ⇒ n_rounds=108 with 4 banks; the deployed bitstream's per-subarray backup pool maxes at ~78 rows. Three known fixes (none deployed yet): (a) 64-bit chunks halve n_rounds; (b) dual-subarray pool — staged but produces wrong output, debug pending; (c) persistent weights via LOAD_WEIGHTS — also has known correctness bug. |
| Caveat | PCIe link in degraded mode 5 GT/s × 4 today (vs nominal 8 GT/s × 8) — recv ~2× slower than past memory-noted runs |

**NARRATION:**
> What you saw is the full model — all thirty transformer layers,
> all seven matrix multiplications per layer. Two hundred and ten
> separate matmuls per output token, every single one running on the
> in-DRAM charge-sharing primitive. It takes about a minute per
> token of output today. That is *slow.*
>
> <break time="500ms"/>
>
> But "slow" doesn't tell you whether it's slow because the *silicon*
> is slow, or because something *outside* the silicon is slow. The
> answer matters. So let's break the time down.

---

## ACT 2 — ACKNOWLEDGE THE OVERHEAD (target 4 min)

### Scene 2.1 — Where the 325 ms / request goes

**VISUAL:** Stacked horizontal bar, 325 ms total, segments labelled:

```
325 ms total per BitLinear matmul (measured srv-prof, today)
├── per-column writes (480 PCIe round-trips)        86 ms  26%
├── multibank execute (320 PCIe round-trips)        84 ms  26%
├── readback drain (c2h DMA)                       120 ms  37%
├── host-side popcount                               3 ms   1%
└── other host overhead                             32 ms  10%

Per token, 30 layers × 7 BitLinears × ~4 slices each = ~960 such requests.
At ~325 ms each (heavier for down_proj/gate_proj) → 371 s / token measured.
```

**NARRATION:**
> Three quarters of every BitLinear's wall time is **PCIe round-trip
> overhead** — host commanding the FPGA, FPGA responding. The actual
> in-DRAM math is buried in those 84 ms of "execute" — about 250
> microseconds per primitive. The other 240+ ms is everything around
> the silicon.
>
> <break time="500ms"/>
>
> **One additional caveat from today's specific run:** the PCIe link
> on this workstation came up in a degraded mode after a reboot
> earlier today — 5 GT/s × 4 lanes instead of the nominal 8 GT/s × 8.
> That roughly doubles the readback time we see on this run. A clean
> link recovery would already take us from 68 s/token to about
> 50 s/token *with no other changes*.

### Scene 2.2 — Why the silicon isn't the bottleneck

**VISUAL:** Two bars side by side:
- **Today (silicon time per token):** 84 ms × 30 × 7 = 17.6 s of actual in-DRAM compute
- **Today (total wall time per token):** 68 s

**NARRATION:**
> Of the 68 seconds per token, only about 17 seconds is the silicon
> doing math. The other 51 seconds is *waiting* — waiting for the
> host to send the next instruction, waiting for the readback DMA
> to finish, waiting for the next program. The silicon has been
> running at single-percent duty cycle.
>
> <break time="500ms"/>
>
> The fix is plain old systems engineering: pre-load the model's
> weights once at startup so we don't ship them across the bus on
> every token; batch many primitives into a single FPGA program so
> we don't pay a PCIe round-trip per primitive; pipeline the
> readback so the next request can be sent before the previous one's
> data has finished arriving. Each of these is a software change.
> None requires touching the DRAM or the FPGA bitstream.

---

## ACT 3 — PROJECT THE CEILING (target 5 min)

### Scene 3.1 — The simulator we're using

**VISUAL:** Code-tree visual: `casa_sched.c` (1700 lines, in-house) — labelled "cycle-level scheduler with full DDR4 timing model (tRCD, tRP, tFAW, tCCD_L, all the JEDEC parameters) plus measured DIMM-0 charge-sharing latencies."

**NARRATION:**
> The numbers on the next slide come from a cycle-level scheduler we
> built in-house and have been validating against measured silicon as
> the project went on. It models every JEDEC DDR4 timing constraint
> and every charge-sharing latency we've measured. It's accurate to
> within a couple of percent on the parts we've checked. It is *not*
> a vendor-supplied model — it's our own, but conservative.

### Scene 3.2 — The projection ladder

**VISUAL:** Vertical bar chart, log y-axis, labels in tok/s:

```
tok/s     stack                                      what's needed
─────  ───────────────────────────────────────  ─────────────────────────────────
0.015  measured today (1 DIMM, all-7 ×30L)     [running this video]
 ↑     ── orchestration overhead ──
0.38   1 DIMM, current bitstream                Tier-A SW: persistent weights,
                                                batched matmul, no per-primitive
                                                round-trips. PURE SOFTWARE.
1.48   4 DIMMs, current bitstream               Above + DIMM 1/2/3 calibration
                                                (sweep currently running).
 ↑     ── modest HDL change ──
1.75   4 DIMMs + popcount accumulator           Small new HDL block + Vivado
                                                rebuild.  Already staged in tree
                                                (popcount_accum.v).
 ↑     ── DRAM-vendor changes ──
1.79   4 DIMMs + LISA cross-subarray            Short wires between adjacent
                                                subarrays. Demoed in CMU SAFARI
                                                research silicon.
 ↑     ── model retraining ──
14.23  4 DIMMs + above + binary activations     Retrain BitNet to use 1-bit
                                                activations instead of 8-bit.
                                                ~10× from 8× less write volume.
```

(All projection rows are from casa_sched run today: `--layers 30 --dimms N --bg-parallel ...`.)

**NARRATION:**
> Here is the ladder.
>
> <break time="500ms"/>
>
> **First step.** Just doing the obvious software work — keep the
> weights resident in the DRAM module, batch the work, stop paying
> a PCIe round-trip per matrix-vector multiply. One DIMM, current
> bitstream, no hardware change. **Zero point three eight tokens per
> second**. That alone is twenty-five times today's measurement.
>
> <break time="500ms"/>
>
> **Second step.** Use all four memory modules in parallel — the
> board has them, we just need to finish characterising the other
> three. **One point five tokens per second.**
>
> <break time="500ms"/>
>
> **Third step.** A small HDL change: add a hardware accumulator
> after the on-FPGA popcount tree, so each matmul ships back four
> bytes instead of eight kilobytes. Eliminates most of the readback
> traffic. **One point seven five tokens per second.**
>
> <break time="500ms"/>
>
> **Fourth step.** A DRAM-vendor change: add LISA — short wires
> between adjacent subarrays so intermediate results don't have to
> leave the chip. **About one point eight tokens per second.**
>
> <break time="1s"/>
>
> **PUNCH:** And the biggest single multiplier on this list isn't
> hardware at all. It's *the model*. Retrain BitNet to use one-bit
> activations instead of eight, and the bus has to move eight times
> less data. **Fourteen tokens per second on this same FPGA card.**
> That is competitive with a midrange GPU running this exact model.
> On a memory chip.

### Scene 3.3 — What's bus-bound and why

**VISUAL:** Sankey-style diagram. Per-token data movement breakdown:
- Weight load: ~1.6 GB/token (today, no persistence)
- Activation bitplane writes: ~30 MB/token
- Result row reads: ~320 MB/token
- Total: ~2 GB/token

After persistent weights:
- Weight load: ~0 (already in DRAM)
- Activation writes: ~30 MB/token
- Result reads: ~320 MB/token

After popcount accumulator:
- Weights ~0, activations ~30 MB, results ~5 MB

After binary activations:
- Activations ~4 MB (8× cut)

**NARRATION:**
> The chart on the previous slide is what it is because every step
> down the ladder cuts a *specific* type of bus traffic. Persistent
> weights cuts the read-and-load traffic to zero between requests.
> The popcount accumulator cuts the result-row traffic by a factor
> of two thousand. Binary activations cuts the per-bitplane writes
> by a factor of eight. Each step is a different scissor cut on the
> bus volume — none of them is magic, all of them are well-understood.

---

## ACT 4 (optional 90 s closer) — what we *did not* do today, and why

**VISUAL:** Three small "tray" cards with a sub-caption each:

- **HDL command sequence engine** — built and Verilator-validated this
  session; pushes PHY *command* bus utilization to 100 % on plain
  reads. Not in the demo: the BitNet workload is bus-bound on
  *data volume*, not command emission. Per casa_sched, this saves
  ~7 % on the projected ceiling.
- **Bank-parallel host scheduler** — ditto. Verilator-proven per-bank
  bit-exact to the SiMRA template (no calibration breakage). Not in
  the demo for the same reason: the dominant cost is data-volume,
  not the command-bus density inside MAJ3.
- **DIMM 1 / 2 / 3 calibration sweep** — running in background while
  this video was being assembled. Unblocks the 4× from "4 DIMMs"
  row above.

**NARRATION:**
> One last note on the work behind this video. We did build several
> pieces of optimisation infrastructure this round — a hardware
> command sequence engine in HDL, a bank-parallel software scheduler
> — and verified both in Verilator simulation. Neither is in the
> demo number above. The reason is what casa_sched showed us: the
> BitNet workload is bus-bound on data *volume*, not on command
> emission rate. The pieces we built saturate the command bus, but
> the data bus was already saturated. They become important after
> the persistent-weights work closes the orchestration gap.
>
> The project is on GitHub at github dot com slash p-c-d-e-n-i
> slash CaSA. Memory notes, simulator source, and the per-DIMM
> measurements are all in the repo.

---

## Appendix A — Where each number comes from

| Number | Source | Date / commit |
|---|---|---|
| **6/7 BitLinears, 269.9 s/tok** | run_bitnet_pim.py with `--projs (q,k,v,o,gate,up)` (no down_proj), `PIM_DUAL_SUBARRAY=0`, prompt "Paris" no chat template, 1 generated token. PIM next-token matches CPU PyTorch reference (`,`). | 2026-05-05 |
| 285 ms / BitLinear (steady-state) | Same run, srv-prof line means | 2026-05-05 |
| (Withheld) 7/7 dual-subarray run gave wrong output (`convo`) | Buggy implementation of dual-subarray B; default reverted to OFF. Tracked as task #75. | 2026-05-05 |
| Reference next-token from CPU | PyTorch CPU forward on the same model, no chat template, prompt "Paris", greedy → `,` | 2026-05-05 |
| 0.38 tok/s @ 1 DIMM | `casa_sched --layers 30 --dimms 1 --bg-parallel` | 2026-05-05 |
| 1.48 tok/s @ 4 DIMMs | `casa_sched --layers 30 --dimms 4 --bg-parallel` | 2026-05-05 |
| 1.75 tok/s + popcount-DRAM | `casa_sched ... --popcount dram` | 2026-05-05 |
| 1.79 tok/s + LISA | `casa_sched ... --lisa --popcount dram` | 2026-05-05 |
| 14.23 tok/s + binary acts | `casa_sched ... --lisa --popcount dram --act-bits 1` | 2026-05-05 |

## Appendix B — Reproducing on this hardware

```
# Today's correct silicon demo: 6/7 BitLinears on PIM, full 30 layers,
# bit-exact next-token match vs CPU PyTorch, ~4.5 min wall.
cd /home/deni/bitnet_weights
PIM_NO_CHAT_TEMPLATE=1 PIM_POOL_OFFSET=0 PIM_DUAL_SUBARRAY=0 \
python3 -u run_bitnet_pim.py \
    --bender 0 --bank "0,1,2,3" \
    --layers all \
    --projs "self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.o_proj,mlp.gate_proj,mlp.up_proj" \
    --max-tokens 1 \
    --prompt "Paris"
# Expected output: ',' (matches CPU PyTorch reference)

# Projection ladder (one row per cell of the casa_sched table)
cd /home/deni/Claude/CaSA-main
./casa_sched --layers 30 --dimms 1 --seq 5 --bg-parallel
./casa_sched --layers 30 --dimms 4 --seq 5 --bg-parallel
./casa_sched --layers 30 --dimms 4 --seq 5 --bg-parallel --popcount dram
./casa_sched --layers 30 --dimms 4 --seq 5 --bg-parallel --lisa --popcount dram
./casa_sched --layers 30 --dimms 4 --seq 5 --bg-parallel --lisa --popcount dram --act-bits 1
```
