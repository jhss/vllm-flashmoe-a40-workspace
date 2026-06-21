rows=48
settings=['block_both_64', 'block_fixed_both_64']
baseline=block_both_64
pair_key=['threshold', 'input_seed_group', 'cycle', 'tokens']
threshold_values=[0]
input_seed_group_values=[1007, 2007, 3007]
cycle_values=[1, 2, 3, 4]
tokens_values=[320, 448]
missing=[]

## Critical Path Absolute
| threshold | tokens | setting | median | IQR | min | max |
|---|---|---|---|---|---|---|
| 0 | 320 | block_both_64 | 1461.8 | 12.8 | 1444.0 | 2046.8 |
| 0 | 320 | block_fixed_both_64 | 1406.6 | 36.6 | 1378.6 | 1985.8 |
| 0 | 448 | block_both_64 | 1514.1 | 26.3 | 1489.9 | 1585.7 |
| 0 | 448 | block_fixed_both_64 | 1451.1 | 19.0 | 1425.8 | 1462.9 |

## Paired Delta
| threshold | tokens | setting | min recv median | critical recv median | median delta | delta/median baseline | median pair pct | IQR | min | max | wins |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 320 | block_fixed_both_64 | 640 | 640 | -57.9 | -3.96% | -3.98% | 35.2 | -624.6 | +529.8 | 10/12 |
| 0 | 448 | block_fixed_both_64 | 896 | 896 | -67.0 | -4.43% | -4.45% | 18.2 | -132.0 | -40.9 | 12/12 |

## Seed-Level Paired Median
| threshold | tokens | input_seed_group | setting | median delta | wins |
|---|---|---|---|---|---|
| 0 | 320 | 1007 | block_fixed_both_64 | -64.4 | 3/4 |
| 0 | 320 | 2007 | block_fixed_both_64 | -52.9 | 4/4 |
| 0 | 320 | 3007 | block_fixed_both_64 | -67.6 | 3/4 |
| 0 | 448 | 1007 | block_fixed_both_64 | -80.5 | 4/4 |
| 0 | 448 | 2007 | block_fixed_both_64 | -63.8 | 4/4 |
| 0 | 448 | 3007 | block_fixed_both_64 | -68.1 | 4/4 |

## Rank Activation
| threshold | tokens | setting | rank0 true/total | rank1 true/total | num_tokens r0/r1 | recv r0/r1 | critical r0/r1/tie |
|---|---|---|---|---|---|---|---|
| 0 | 320 | block_both_64 | 12/12 | 12/12 | 638/638 | 638/638 | 6/6/0 |
| 0 | 320 | block_fixed_both_64 | 12/12 | 12/12 | 640/640 | 640/640 | 7/5/0 |
| 0 | 448 | block_both_64 | 12/12 | 12/12 | 894/892 | 894/892 | 7/5/0 |
| 0 | 448 | block_fixed_both_64 | 12/12 | 12/12 | 896/896 | 896/896 | 5/7/0 |

## Route Stats
| threshold | tokens | valid pairs r0/r1 | invalid pairs r0/r1 |
|---|---|---|---|
| 0 | 320 | 2570/2550 | 2534/2547 |
| 0 | 448 | 3598/3570 | 3554/3566 |

## Positive Delta Outliers
| group | input_seed_group | cycle | setting | delta | base | target | recv r0/r1 | active r0/r1 |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| threshold=0 tokens=320 | 1007 | 4 | block_fixed_both_64 | +529.8 | 1456.0 | 1985.8 | 640/640 | True/True |
| threshold=0 tokens=320 | 3007 | 2 | block_fixed_both_64 | +34.3 | 1464.5 | 1498.9 | 640/640 | True/True |
