reference=original
settings=['compute_both_64', 'fixed_remap_both_64', 'final_raw_both_64']
all_assert_close=True

The final correctness smoke compares every optimized setting against the
original path in the same 2-rank layer instance. Metrics below are the worst
value across ranks and optimized settings for each case.

| case | output | max_abs_error | mean_abs_error | relative_l2_error | assert_close |
|---|---|---:|---:|---:|---|
| tokens=320 balanced | deepep_ht_final_correctness_tokens320_20260621.json | 0.00195312 | 7.05142e-05 | 0.00375014 | True |
| tokens=448 balanced | deepep_ht_final_correctness_tokens448_20260621.json | 0.00195312 | 7.02225e-05 | 0.00375855 | True |
| rank_tokens=128/512 balanced | deepep_ht_final_correctness_rank128_512_20260621.json | 0.00195312 | 6.99849e-05 | 0.00375052 | True |
| rank_tokens=128/512 target_rank=0 | deepep_ht_final_correctness_rank128_512_target0_20260621.json | 0.00195312 | 9.02659e-05 | 0.00321721 | True |
| rank_tokens=128/512 target_rank=1 | deepep_ht_final_correctness_rank128_512_target1_20260621.json | 0.00195312 | 9.11606e-05 | 0.00321549 | True |

Alignment regression:

```text
python -m pytest -q tests/kernels/moe/test_moe_align_block_size.py
477 passed, 16 warnings
```
