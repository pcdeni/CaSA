# RUN AFTER FLASH — build-11 (magic 0xDBC0DE0A)

Fixes on board: fetch_restart (pc=0 at every program start — kills the
stale-pc loss window, latent in legacy too) + fin/tlast same-cycle
race. Verilator regression A–H + F(5×4000) ALL_PASS pre-synth.

## 0. Bring-up (RUNBOOK_TOWER.md rules)
- USER flashes via JTAG. Then: `remove+rescan` or warm reboot —
  NEVER pci-reset post-JTAG, NEVER cold-cycle (erases the image).
- Card check: `lspci -nn -d 10ee:` (never by BDF).

## 1. Identity + ladder (~3 min)
```
cd /home/deni/Claude/SiMRA-DRAM-main/DRAM-Bender/sources/apps/DSN_AE_APPS/BitNet
BITSTREAM_IMEM=8192 ./stream-hw-exe 2 1 45320 32
```
- Trailer magic must read **0xDBC0DE0A** (PIM_RECV_DEBUG=1 if needed).
- All arms A–F14 must pass (E3's 16/32 vs stale reference is the known
  benign arm-ordering artifact).

## 2. Twin gates (replay, ~15 min each, watchdogged)
```
cd /home/deni/Claude/roadb_build9_2026_07_22
# control (rerun floor):
REPLAY_DUP_MM3D=1 python3 -u replay_ab.py reqcap.b2 --dup c11:PIM_STREAM=0 --limit 1100 --tag c11
# pipe gate:
REPLAY_DUP_MM3D=1 python3 -u replay_ab.py reqcap.b2 --dup p11:PIM_STREAM=1,PIM_PIPE_ALTERNATE=1 --limit 1100 --tag p11
```
Criterion: gate ≈ control (V2 at floor; MM3D in the ~100-105/720
straddle band). Analog-swap levers (consts) are judged at output
level, not twin level.

## 3. Walls (production config, dimm2 trio + guard)
Order matters — each isolates one lever on build-11:
1. `PIM_STREAM=1` only → expect ≈1,848 s (build-10 parity check).
2. `PIM_STREAM=1 PIM_STREAM_PIPE=1` → expect ≈1,307 s (−29%), and —
   the actual gate — NO STALL across the full run + a repeat run.
3. `PIM_STREAM=1 PIM_RESIDENT_CONSTS=1 PIM_RC_V2=1` → expect ≈−31%
   vs (1), tokens sane ("Paris" class output).
4. All three together → the ≈2× request-wall number.
Every run under a stall watchdog (log-frozen 300 s + wchan capture).

## 4. On success
- Flip defaults: PIM_STREAM_PIPE=1 + consts trio in run_bitnet_pim
  setdefaults (PIM_STREAM already ON).
- LEVERS #3/#28 close-out with numbers; PR the arc; memory updates.
- Next arcs: pack4 re-price gate (cheapest), X-master-clone (the
  remaining 2 x-wrRows), then THE MERGE (in-DRAM accumulation).

## If the ladder or gates fail
Old bitstream is preserved on the box
(readback_engine.v.build10.bak_20260724 + frontend .build9.bak_20260724
set); reflash build-10 restores the known-good −4% streaming state.
