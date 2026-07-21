> **Repo note.** Verbatim working record from the rig workspace
> (`/home/deni/Claude/bonsai_fused_ab_2026_07_21/README.md`), preserved unedited as a citation
> target. Reader-facing synthesis: [`BONSAI_2026_07.md`](BONSAI_2026_07.md).

---

# Bonsai × PIM_FUSED_COSET A/B — sim vs silicon (2026-07-21)

Task: clear (or reject) the fused coset activation update for Bonsai
group mode, per client README §5.1 ("A/B it with PIM_INT_DIFF=1 before
trusting it in a Bonsai production run"). Model: bonsai_ternary (+1bit
in sim). Sim = `bonsai_client_2026_07_20/runtime/sim-server` wrapper +
`unit_bonsai_l0.py`; silicon = production `run_bitnet_pim.py` single
projection probes on bender 2, then a full-model clearing run.

## 1. SIM: fused does NOT compose (two distinct failure signatures)

- `sim_fused_{ternary,1bit}_load.log` — `PIM_FUSED_COSET=1` + LOAD:
  every projection with LOAD-resident 4-chunk slices goes grossly
  int-wrong (`groups-sum exact=0/2048`, max_err ~7–10k); slices served
  V2 (post-ENOSPC) are 100.0000% exact. (Trailing FileNotFoundError in
  the load logs = bare-filename outjson, my harness invocation error;
  all 7 projections completed and the per-projection verdicts are in
  the logs. Non-fused control = the 07-20 unit runs, bitwise exact.)
- `sim_fused_{ternary,1bit}_v2.log/.json` — fused + forced all-V2:
  q/o/gate/up/down **bitwise exact**, but **k/v fail** (identical
  deltas to the load run) — k/v are the n_copies=2 replicated slices.
- Scope note: BitNet fused=1 sim evidence (o3, 07-18) used n_chunks=8
  subs and passed — the sim break is specific to group-mode shapes
  (4-chunk single-group subs; replicated slices).

## 2. SILICON: fused composes EXACTLY (sim gaps, not real bugs)

5 arms, bender 2, layer-0 single projection, 1 token, `PIM_INT_DIFF=1`
(`sil_{q,k}_{ctrl,fused}.log`, `sil_k_fused_vote.log`):

| arm | config | groups-sum int exactness |
|---|---|---|
| q ctrl | vote off | 100.0000% (all lines) |
| q fused | `PIM_FUSED_COSET=1` — the fused×4-chunk-LOAD case | **100.0000%** |
| k ctrl | vote off, n_copies=2 replication | 100.0000% |
| k fused | fused × replication | **100.0000%** |
| k fused+vote | fused × replication × copy-vote | **100.0000%** |

**35/35 int-diff lines exact, zero exceptions.** The sim's fused-body
emulation simply does not model the coset deposit for 4-chunk MM3D
bodies / copy rows; real silicon does the physics. Consequence for
future sim work: fused+group-mode sim results are UNTRUSTWORTHY until
pim_sim learns those two shapes — silicon is the arbiter here.

## 3. Full-model clearing run (pool-collision lesson: layer-0 proves nothing at pool scale)

`sil_fullmodel_fused_dimms02.log` — bonsai_ternary, raw, 8 tokens,
`--dimms 0,2`, vote off, `PIM_FUSED_COSET=1` (banner verified in both
servers' stderr at /tmp/pim_server_b{0,2}_0_1_2_3.log):

**8 tokens in 269.2 s = 33.7 s/generated-token, output
` The capital of France is Paris. Paris` — the same 8/8 golden-exact
text.** vs the same-morning non-fused dual-DIMM run (409.9 s):
**1.52× from fused alone**; vs the 07-20 single-DIMM baseline (799.7 s):
**2.97× stacked** (dual-DIMM 1.95× × fused 1.52×). 19.2 s/token-position.
Zero FPGA incidents.

PREDICTION CORRECTED (recorded, per structural-before-stochastic): this
file's pre-run note expected fused to be ~throughput-neutral because
">99% of wall is pipe-read". Wrong — pipe-read time IS the server's
per-request service time, which is DRAM-program-execution-bound; fused
shortens every MAJ3 body (coset broadcast replaces the 5 activation
wrRows), and 81,689 requests × ~1.7 ms saved ≈ the 140.7 s delta.
Matches BitNet's measured 1.45–1.6×/matmul. Lesson: "pipe-read" in the
client profile is not network latency — it contains the DRAM work.

## Verdict

- Fused coset is **silicon-cleared for Bonsai at unit AND full-model
  scale**, and is now the production recommendation for Bonsai runs:
  `PIM_FUSED_COSET=1` — worth 1.52× per token today, stacking with the
  dual-DIMM split (33.7 s/tok combined, golden-exact).
- The failed 07-21 first attempt (`EXIT *: 127` arms) never touched the
  rig — shell env-prefix parsing error, rerun fixed.
- Bug filed conceptually against pim_sim fidelity (fused × 4-chunk MM3D,
  fused × copy rows) — a future sim work item, not a production blocker.
