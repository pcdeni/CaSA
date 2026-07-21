# Road-B lane2 A/B — 2026-07-21 16:29:01.744398

args: --shapes 512x256,4096x4096 --qbits 2 --iters 2 --quiet-ms 2 --tick-ms 2 --outdir /home/deni/Claude/roadb_lane2_2026_07_21/ab_wres

| shape | qb | rb | y_R==y_A | max|Δ| | vs numpy R (max/nz) | vs numpy A | wall R (s) | wall A (s) | R/A |
|---|---|---|---|---|---|---|---|---|---|
| 512x256 | 2 | 1 | 1/256 differ | 2 | 2/4nz | 2/3nz | 0.09 | 1.77 | 0.05x |
| 4096x4096 | 2 | 1 | 330/4096 differ | 5 | 4/178nz | 4/198nz | 1.47 | 2.94 | 0.50x |
