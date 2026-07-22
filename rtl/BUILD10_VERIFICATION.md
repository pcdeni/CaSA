# build-10 — SEG_POP/ACCXBP raw-count wdata mask (magic 0xDBC0DE09)

One-line semantic fix for the bug the Rung-1 producer loop exposed
(full arc: `BUILD9_VERIFICATION.md` §producer-loop gate):

**Root cause (wiring-confirmed).** `ddr_wdata` mirrors the core's
write-data register file (`ddr_pipeline: ddr_wdata = ddr_data_r <=
wide_reg`), which persists across programs. Only the legacy INIT_MEM
pass ever refreshed it; build-9's streaming swap goes EXECUTE→EXECUTE,
so after any write program the last LDWD pattern stays on the bus and
every SEG_POP/ACCXBP count becomes `popcount(rd_data ^ stale)` —
sticky, silent, ~3/4 of packed bytes wrong (the 4-counts-per-word
packer), reproduced on silicon 2026-07-22 (stream-hw arms C–G).

**Fix.** `read_diff` XORs against `diff_ref = (mode == DIFF) ?
ddr_wdata : 0`. SEG_POP and ACCUM_XBP count RAW rows (their documented
semantic); DIFF keeps its compare-vs-pattern XOR. When `ddr_wdata` is
zero the mask is an identity — every legacy-validated path is
byte-identical by construction.

**Verilator gate (gate_b10.sh, box harness): MET.**
`scenario_segpop_stale_wdata` (nonzero ddr_wdata held through segpop
reads) FAILS on the build-9 engine — 0/128 raw counts exact — the
silicon failure reproduced in cycle world; PASSES on build-10
(128/128). Failure-set diff identical everywhere else (the 7 known
documentation checks). Build-10 Vivado: all timing constraints met,
the 3 known softmc_core.v:7 critical warnings, bit md5 c0bc6233…

**Post-flash sequence.** Ladder (magic 0xDBC0DE09) → stream-hw arms
A–G must ALL_PASS including the mixed shapes E/E2/E4/E6 → full-model
PIM_STREAM token-identity + wall A/B → production-stack A/B
(PIM_USE_LOAD_WEIGHTS=1).
