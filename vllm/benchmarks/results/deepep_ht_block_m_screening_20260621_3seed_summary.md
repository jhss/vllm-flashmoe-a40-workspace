rows=54
settings=['block_default', 'block_both_32', 'block_both_64']
baseline=block_default
pair_key=['threshold', 'input_seed_group', 'cycle', 'tokens']
threshold_values=[0]
input_seed_group_values=[1007, 2007, 3007]
cycle_values=[1, 2, 3]
tokens_values=[320, 448]
missing=[]

## Critical Path Absolute
| threshold | tokens | setting | median | IQR | min | max |
|---|---|---|---|---|---|---|
| 0 | 320 | block_default | 1619.5 | 26.1 | 1595.7 | 2123.0 |
| 0 | 320 | block_both_32 | 1499.0 | 12.3 | 1487.9 | 1607.4 |
| 0 | 320 | block_both_64 | 1471.4 | 19.6 | 1461.9 | 1542.9 |
| 0 | 448 | block_default | 1660.0 | 20.5 | 1634.2 | 1691.7 |
| 0 | 448 | block_both_32 | 1554.4 | 14.1 | 1535.2 | 1571.9 |
| 0 | 448 | block_both_64 | 1510.9 | 10.8 | 1498.4 | 1528.9 |

## Paired Delta
| threshold | tokens | setting | min recv median | critical recv median | median delta | delta/median baseline | median pair pct | IQR | min | max | wins |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 320 | block_both_32 | 638 | 638 | -124.7 | -7.70% | -7.74% | 18.8 | -623.9 | -0.2 | 9/9 |
| 0 | 320 | block_both_64 | 638 | 638 | -145.7 | -8.99% | -8.91% | 38.2 | -657.0 | -64.7 | 9/9 |
| 0 | 448 | block_both_32 | 892 | 892 | -93.4 | -5.63% | -5.69% | 22.8 | -156.5 | -77.8 | 9/9 |
| 0 | 448 | block_both_64 | 892 | 892 | -146.1 | -8.80% | -8.80% | 32.4 | -185.9 | -105.4 | 9/9 |

## Seed-Level Paired Median
| threshold | tokens | input_seed_group | setting | median delta | wins |
|---|---|---|---|---|---|
| 0 | 320 | 1007 | block_both_32 | -131.7 | 3/3 |
| 0 | 320 | 1007 | block_both_64 | -172.1 | 3/3 |
| 0 | 320 | 2007 | block_both_32 | -117.2 | 3/3 |
| 0 | 320 | 2007 | block_both_64 | -133.9 | 3/3 |
| 0 | 320 | 3007 | block_both_32 | -136.0 | 3/3 |
| 0 | 320 | 3007 | block_both_64 | -145.7 | 3/3 |
| 0 | 448 | 1007 | block_both_32 | -92.3 | 3/3 |
| 0 | 448 | 1007 | block_both_64 | -134.1 | 3/3 |
| 0 | 448 | 2007 | block_both_32 | -93.4 | 3/3 |
| 0 | 448 | 2007 | block_both_64 | -130.5 | 3/3 |
| 0 | 448 | 3007 | block_both_32 | -132.0 | 3/3 |
| 0 | 448 | 3007 | block_both_64 | -158.1 | 3/3 |

## Rank Activation
| threshold | tokens | setting | rank0 true/total | rank1 true/total | num_tokens r0/r1 | recv r0/r1 | critical r0/r1/tie |
|---|---|---|---|---|---|---|---|
| 0 | 320 | block_default | 9/9 | 9/9 | 638/638 | 638/638 | 3/6/0 |
| 0 | 320 | block_both_32 | 9/9 | 9/9 | 638/638 | 638/638 | 3/6/0 |
| 0 | 320 | block_both_64 | 9/9 | 9/9 | 638/638 | 638/638 | 4/5/0 |
| 0 | 448 | block_default | 9/9 | 9/9 | 894/892 | 894/892 | 6/3/0 |
| 0 | 448 | block_both_32 | 9/9 | 9/9 | 894/892 | 894/892 | 4/5/0 |
| 0 | 448 | block_both_64 | 9/9 | 9/9 | 894/892 | 894/892 | 4/5/0 |

## Route Stats
| threshold | tokens | valid pairs r0/r1 | invalid pairs r0/r1 |
|---|---|---|---|
| 0 | 320 | 2570/2550 | 2534/2547 |
| 0 | 448 | 3598/3570 | 3554/3566 |

## Positive Delta Outliers
| group | input_seed_group | cycle | setting | delta | base | target | recv r0/r1 | active r0/r1 |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| none |  |  |  |  |  |  |  |  |

