# Road-B lane2 A/B — 2026-07-21 12:33:23.514397

args: --shapes 128x32,512x256 --qbits 2 --xrefresh 16 --iters 2 --outdir /home/deni/Claude/roadb_lane2_2026_07_21/ab_bringup2

| shape | qb | rb | y_R==y_A | max|Δ| | vs numpy R (max/nz) | vs numpy A | wall R (s) | wall A (s) | R/A |
|---|---|---|---|---|---|---|---|---|---|
| 128x32 | 2 | 1 | EXACT | 0 | 0/0nz | 0/0nz | 0.01 | 5.51 | 0.00x |
| 512x256 | 2 | 1 | EXACT | 0 | 0/0nz | 0/0nz | 0.10 | 5.58 | 0.02x |
