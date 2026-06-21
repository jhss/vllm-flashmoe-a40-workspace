rows=96
settings=['block_default', 'block_w1_64', 'block_w2_64', 'block_both_64']
baseline=block_default
pair_key=['threshold', 'input_seed_group', 'cycle', 'tokens']
threshold_values=[0]
input_seed_group_values=[1007, 2007, 3007]
cycle_values=[1, 2, 3, 4]
tokens_values=[320, 448]
missing=[]

## Critical Path Absolute
| threshold | tokens | setting | median | IQR | min | max |
|---|---|---|---|---|---|---|
| 0 | 320 | block_default | 1629.9 | 47.7 | 1597.9 | 2111.0 |
| 0 | 320 | block_w1_64 | 1539.2 | 14.2 | 1522.7 | 1565.9 |
| 0 | 320 | block_w2_64 | 1593.9 | 38.5 | 1575.8 | 2257.6 |
| 0 | 320 | block_both_64 | 1457.0 | 30.5 | 1444.7 | 1584.5 |
| 0 | 448 | block_default | 1655.5 | 17.3 | 1639.7 | 1681.3 |
| 0 | 448 | block_w1_64 | 1586.4 | 14.2 | 1563.6 | 1658.5 |
| 0 | 448 | block_w2_64 | 1626.1 | 13.6 | 1599.3 | 2121.1 |
| 0 | 448 | block_both_64 | 1517.7 | 29.0 | 1493.4 | 1592.2 |

## Paired Delta
| threshold | tokens | setting | min recv median | critical recv median | median delta | delta/median baseline | median pair pct | IQR | min | max | wins |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 320 | block_w1_64 | 638 | 638 | -93.6 | -5.74% | -5.73% | 59.7 | -567.1 | -51.0 | 12/12 |
| 0 | 320 | block_w2_64 | 638 | 638 | -29.5 | -1.81% | -1.82% | 51.8 | -521.7 | +628.3 | 10/12 |
| 0 | 320 | block_both_64 | 638 | 638 | -175.0 | -10.73% | -10.77% | 46.0 | -617.4 | -98.7 | 12/12 |
| 0 | 448 | block_w1_64 | 892 | 892 | -69.5 | -4.20% | -4.20% | 18.5 | -88.0 | +4.8 | 11/12 |
| 0 | 448 | block_w2_64 | 892 | 892 | -26.3 | -1.59% | -1.59% | 22.6 | -82.0 | +455.3 | 11/12 |
| 0 | 448 | block_both_64 | 892 | 894 | -143.6 | -8.67% | -8.62% | 27.6 | -155.9 | -73.8 | 12/12 |

## Seed-Level Paired Median
| threshold | tokens | input_seed_group | setting | median delta | wins |
|---|---|---|---|---|---|
| 0 | 320 | 1007 | block_w1_64 | -80.8 | 4/4 |
| 0 | 320 | 1007 | block_w2_64 | -54.3 | 4/4 |
| 0 | 320 | 1007 | block_both_64 | -168.1 | 4/4 |
| 0 | 320 | 2007 | block_w1_64 | -94.0 | 4/4 |
| 0 | 320 | 2007 | block_w2_64 | -28.7 | 4/4 |
| 0 | 320 | 2007 | block_both_64 | -160.6 | 4/4 |
| 0 | 320 | 3007 | block_w1_64 | -114.3 | 4/4 |
| 0 | 320 | 3007 | block_w2_64 | +14.7 | 2/4 |
| 0 | 320 | 3007 | block_both_64 | -188.2 | 4/4 |
| 0 | 448 | 1007 | block_w1_64 | -55.6 | 3/4 |
| 0 | 448 | 1007 | block_w2_64 | -19.8 | 4/4 |
| 0 | 448 | 1007 | block_both_64 | -140.8 | 4/4 |
| 0 | 448 | 2007 | block_w1_64 | -79.5 | 4/4 |
| 0 | 448 | 2007 | block_w2_64 | -39.3 | 4/4 |
| 0 | 448 | 2007 | block_both_64 | -114.9 | 4/4 |
| 0 | 448 | 3007 | block_w1_64 | -64.3 | 4/4 |
| 0 | 448 | 3007 | block_w2_64 | -20.4 | 3/4 |
| 0 | 448 | 3007 | block_both_64 | -145.9 | 4/4 |

## Rank Activation
| threshold | tokens | setting | rank0 true/total | rank1 true/total | num_tokens r0/r1 | recv r0/r1 | critical r0/r1/tie |
|---|---|---|---|---|---|---|---|
| 0 | 320 | block_default | 12/12 | 12/12 | 638/638 | 638/638 | 7/5/0 |
| 0 | 320 | block_w1_64 | 12/12 | 12/12 | 638/638 | 638/638 | 4/8/0 |
| 0 | 320 | block_w2_64 | 12/12 | 12/12 | 638/638 | 638/638 | 4/8/0 |
| 0 | 320 | block_both_64 | 12/12 | 12/12 | 638/638 | 638/638 | 5/7/0 |
| 0 | 448 | block_default | 12/12 | 12/12 | 894/892 | 894/892 | 5/7/0 |
| 0 | 448 | block_w1_64 | 12/12 | 12/12 | 894/892 | 894/892 | 6/6/0 |
| 0 | 448 | block_w2_64 | 12/12 | 12/12 | 894/892 | 894/892 | 6/6/0 |
| 0 | 448 | block_both_64 | 12/12 | 12/12 | 894/892 | 894/892 | 8/4/0 |

## Route Stats
| threshold | tokens | valid pairs r0/r1 | invalid pairs r0/r1 |
|---|---|---|---|
| 0 | 320 | 2570/2550 | 2534/2547 |
| 0 | 448 | 3598/3570 | 3554/3566 |

## Positive Delta Outliers
| group | input_seed_group | cycle | setting | delta | base | target | recv r0/r1 | active r0/r1 |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| threshold=0 tokens=320 | 3007 | 3 | block_w2_64 | +44.1 | 1617.0 | 1661.0 | 638/638 | True/True |
| threshold=0 tokens=320 | 3007 | 4 | block_w2_64 | +628.3 | 1629.3 | 2257.6 | 638/638 | True/True |
| threshold=0 tokens=448 | 1007 | 2 | block_w1_64 | +4.8 | 1653.7 | 1658.5 | 896/890 | True/True |
| threshold=0 tokens=448 | 3007 | 1 | block_w2_64 | +455.3 | 1665.9 | 2121.1 | 892/894 | True/True |

