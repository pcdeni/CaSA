# Road-B lane2 A/B — 2026-07-21 12:37:25.229760

args: --shapes 32000x4096 --qbits 2 --iters 1 --quiet-ms 2 --tick-ms 2 --outdir /home/deni/Claude/roadb_lane2_2026_07_21/ab_head

| shape | qb | rb | y_R==y_A | max|Δ| | vs numpy R (max/nz) | vs numpy A | wall R (s) | wall A (s) | R/A |
|---|---|---|---|---|---|---|---|---|---|
| 32000x4096 | 2 | 1 | 20280/32000 differ | 8 | 8/22068nz | 8/21821nz | 11.85 | 12.07 | 0.98x |
