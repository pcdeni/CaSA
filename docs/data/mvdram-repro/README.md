Raw logs from the MVDRAM reproduction study (docs/MVDRAM_REPRODUCTION.md).
- rcrand_b0.log / rcrand_b3.log — random-pair RowClone scans on the two new
  HMA851U6CJR6N-UHN0 units (30,000 pairs each, rows uniform in [0,65536),
  t_12=30, t_23∈{1,2}): 0 clones, best 45/8192 and 47/8192 (noise floor).
  Tool: app/test_rowclone_random.cpp. Control on a PUD-capable module found
  6 full clones in 500 pairs with the same tool.
