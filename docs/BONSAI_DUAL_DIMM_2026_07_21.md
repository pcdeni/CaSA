> **Repo note.** Verbatim working record from the rig workspace
> (`/home/deni/Claude/bonsai_dualdimm_2026_07_21/README.md`), preserved unedited as a citation
> target. Reader-facing synthesis: [`BONSAI_2026_07.md`](BONSAI_2026_07.md).

---

# Bonsai-1.7B dual-DIMM silicon run (2026-07-21)

Follow-up to `bonsai_silicon_2026_07_20/` (run 1 = the single-DIMM
baseline for this shape). One run: full-model **bonsai_ternary**, raw
prompt "What is the capital of France?", 8 new tokens, **`--dimms 0,2`**
(group parts and load-subs round-robin across benders 0+2), banks
0,1,2,3, vote OFF, LOAD→V2 streaming on both dies.

```bash
cd /home/deni/bitnet_weights
BITSTREAM_IMEM=8192 PIM_NO_CHAT_TEMPLATE=1 PIM_VOTE_FULL=0 PIM_RECV_TIMEOUT_MS=15000 \
python3 -u run_bitnet_pim.py --model bonsai_ternary --dimms 0,2 --bank 0,1,2,3 \
    --prompt "What is the capital of France?" --max-tokens 8
```

## Result (run1_ternary_dimms02.log)

| metric | dual-DIMM (this run) | single-DIMM (run 1, 07-20) | ratio |
|---|---|---|---|
| generated text | ` The capital of France is Paris. Paris` | identical | 8/8 same tokens (golden raw prefix) |
| 8-token generation | **409.9 s → 51.2 s/generated-token** | 799.7 s → 99.96 | **1.95×** (98% of ideal halving) |
| per token-position (14 pos) | 29.3 s | 57.1 s | 1.95× |
| server calls | 40,845 (b0) + 40,844 (b2) = 81,689 | 81,627 | split ~50.0/50.0 |
| bytes sent | 2,618.4 MB per server (byte-identical) | 5,291.4 MB one server | balanced by construction |
| pipe-read | ~397/398 s per server, CONCURRENT | 787.8 s serial | the wall parallelizes |
| server-time-implied | 4.5 s (b0) + 3.3 s (b2) | 3.1 s | compute still negligible |
| FPGA incidents | zero (recv guard armed, never fired) | zero | |

Group-mode round-robin balances perfectly (equal group counts per
server → byte-identical volumes) — better than BitNet's 1.91×
balance-fixed split, with no balancing work needed. Preflight:
`preflight_rowclone_b0.log` (bender 0 PERFECT_CLONE ×4; bender 2 smoke
ran same morning, see fused-A/B session).

Published (pending user merge): PR #2 `docs/BONSAI_2026_07.md` addendum.
