rows=48
settings=['block_fixed_raw_both_64', 'block_fixed_remap_both_64']
baseline=block_fixed_raw_both_64
pair_key=['threshold', 'input_seed_group', 'cycle', 'tokens']
threshold_values=[0]
input_seed_group_values=[1007, 2007, 3007]
cycle_values=[1, 2, 3, 4]
tokens_values=[320, 448]
missing=[]

## Critical Path Absolute
| threshold | tokens | setting | median | IQR | min | max |
|---|---|---|---|---|---|---|
| 0 | 320 | block_fixed_raw_both_64 | 1458.9 | 30.1 | 1431.2 | 1546.3 |
| 0 | 320 | block_fixed_remap_both_64 | 1535.1 | 84.4 | 1521.0 | 1974.9 |
| 0 | 448 | block_fixed_raw_both_64 | 1508.2 | 33.0 | 1499.1 | 1935.4 |
| 0 | 448 | block_fixed_remap_both_64 | 1583.6 | 50.8 | 1557.0 | 1904.8 |

## Paired Delta
| threshold | tokens | setting | min recv median | critical recv median | median delta | delta/median baseline | median pair pct | IQR | min | max | wins |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 320 | block_fixed_remap_both_64 | 640 | 640 | +88.3 | +6.05% | +6.12% | 82.5 | -11.4 | +516.6 | 1/12 |
| 0 | 448 | block_fixed_remap_both_64 | 896 | 896 | +65.7 | +4.36% | +4.38% | 60.1 | -366.6 | +405.2 | 1/12 |

## Seed-Level Paired Median
| threshold | tokens | input_seed_group | setting | median delta | wins |
|---|---|---|---|---|---|
| 0 | 320 | 1007 | block_fixed_remap_both_64 | +149.2 | 0/4 |
| 0 | 320 | 2007 | block_fixed_remap_both_64 | +70.1 | 1/4 |
| 0 | 320 | 3007 | block_fixed_remap_both_64 | +76.0 | 0/4 |
| 0 | 448 | 1007 | block_fixed_remap_both_64 | +133.3 | 0/4 |
| 0 | 448 | 2007 | block_fixed_remap_both_64 | +52.1 | 1/4 |
| 0 | 448 | 3007 | block_fixed_remap_both_64 | +51.5 | 0/4 |

## Rank Activation
| threshold | tokens | setting | rank0 true/total | rank1 true/total | num_tokens r0/r1 | recv r0/r1 | critical r0/r1/tie |
|---|---|---|---|---|---|---|---|
| 0 | 320 | block_fixed_raw_both_64 | 12/12 | 12/12 | 640/640 | 640/640 | 4/8/0 |
| 0 | 320 | block_fixed_remap_both_64 | 12/12 | 12/12 | 640/640 | 640/640 | 4/8/0 |
| 0 | 448 | block_fixed_raw_both_64 | 12/12 | 12/12 | 896/896 | 896/896 | 4/8/0 |
| 0 | 448 | block_fixed_remap_both_64 | 12/12 | 12/12 | 896/896 | 896/896 | 6/6/0 |

## Route Stats
| threshold | tokens | valid pairs r0/r1 | invalid pairs r0/r1 |
|---|---|---|---|
| 0 | 320 | n/a | n/a |
| 0 | 448 | n/a | n/a |

## Positive Delta Outliers
| group | input_seed_group | cycle | setting | delta | base | target | recv r0/r1 | active r0/r1 |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| threshold=0 tokens=320 | 1007 | 1 | block_fixed_remap_both_64 | +516.6 | 1458.3 | 1974.9 | 640/640 | True/True |
| threshold=0 tokens=320 | 1007 | 2 | block_fixed_remap_both_64 | +64.4 | 1467.3 | 1531.7 | 640/640 | True/True |
| threshold=0 tokens=320 | 1007 | 3 | block_fixed_remap_both_64 | +93.1 | 1442.3 | 1535.4 | 640/640 | True/True |
| threshold=0 tokens=320 | 1007 | 4 | block_fixed_remap_both_64 | +205.4 | 1431.2 | 1636.6 | 640/640 | True/True |
| threshold=0 tokens=320 | 2007 | 1 | block_fixed_remap_both_64 | +86.3 | 1439.3 | 1525.6 | 640/640 | True/True |
| threshold=0 tokens=320 | 2007 | 2 | block_fixed_remap_both_64 | +53.9 | 1469.2 | 1523.1 | 640/640 | True/True |
| threshold=0 tokens=320 | 2007 | 4 | block_fixed_remap_both_64 | +121.1 | 1479.2 | 1600.3 | 640/640 | True/True |
| threshold=0 tokens=320 | 3007 | 1 | block_fixed_remap_both_64 | +319.2 | 1439.1 | 1758.3 | 640/640 | True/True |
| threshold=0 tokens=320 | 3007 | 2 | block_fixed_remap_both_64 | +61.6 | 1459.4 | 1521.0 | 640/640 | True/True |
| threshold=0 tokens=320 | 3007 | 3 | block_fixed_remap_both_64 | +90.3 | 1447.7 | 1538.0 | 640/640 | True/True |
| threshold=0 tokens=320 | 3007 | 4 | block_fixed_remap_both_64 | +38.3 | 1482.9 | 1521.2 | 640/640 | True/True |
| threshold=0 tokens=448 | 1007 | 1 | block_fixed_remap_both_64 | +405.2 | 1499.6 | 1904.8 | 896/896 | True/True |
| threshold=0 tokens=448 | 1007 | 2 | block_fixed_remap_both_64 | +120.2 | 1529.2 | 1649.4 | 896/896 | True/True |
| threshold=0 tokens=448 | 1007 | 3 | block_fixed_remap_both_64 | +146.3 | 1511.2 | 1657.6 | 896/896 | True/True |
| threshold=0 tokens=448 | 1007 | 4 | block_fixed_remap_both_64 | +56.2 | 1504.2 | 1560.4 | 896/896 | True/True |
| threshold=0 tokens=448 | 2007 | 1 | block_fixed_remap_both_64 | +93.5 | 1506.5 | 1600.0 | 896/896 | True/True |
| threshold=0 tokens=448 | 2007 | 2 | block_fixed_remap_both_64 | +10.9 | 1582.0 | 1592.9 | 896/896 | True/True |
| threshold=0 tokens=448 | 2007 | 4 | block_fixed_remap_both_64 | +93.3 | 1499.1 | 1592.4 | 896/896 | True/True |
| threshold=0 tokens=448 | 3007 | 1 | block_fixed_remap_both_64 | +53.2 | 1508.8 | 1562.0 | 896/896 | True/True |
| threshold=0 tokens=448 | 3007 | 2 | block_fixed_remap_both_64 | +75.2 | 1499.5 | 1574.7 | 896/896 | True/True |
| threshold=0 tokens=448 | 3007 | 3 | block_fixed_remap_both_64 | +0.7 | 1556.3 | 1557.0 | 896/896 | True/True |
| threshold=0 tokens=448 | 3007 | 4 | block_fixed_remap_both_64 | +49.9 | 1507.6 | 1557.4 | 896/896 | True/True |

