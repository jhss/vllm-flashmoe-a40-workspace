rows=54
settings=['original', 'filtering', 'final_both_64']
baseline=original
pair_key=['threshold', 'input_seed_group', 'cycle', 'tokens']
threshold_values=[0]
input_seed_group_values=[1007, 2007, 3007]
cycle_values=[1, 2, 3]
tokens_values=[320, 448]
missing=[]

## Critical Path Absolute
| threshold | tokens | setting | median | IQR | min | max |
|---|---|---|---|---|---|---|
| 0 | 320 | original | 1659.3 | 29.2 | 1648.1 | 1733.5 |
| 0 | 320 | filtering | 1610.1 | 13.9 | 1588.7 | 1648.6 |
| 0 | 320 | final_both_64 | 1455.0 | 17.4 | 1442.7 | 1527.6 |
| 0 | 448 | original | 1698.6 | 11.2 | 1679.5 | 1718.8 |
| 0 | 448 | filtering | 1641.9 | 17.2 | 1634.6 | 1750.7 |
| 0 | 448 | final_both_64 | 1516.6 | 14.9 | 1499.9 | 1538.8 |

## Paired Delta
| threshold | tokens | setting | min recv median | critical recv median | median delta | delta/median baseline | median pair pct | IQR | min | max | wins |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 320 | filtering | 638 | 638 | -43.6 | -2.63% | -2.65% | 18.0 | -144.8 | -9.5 | 9/9 |
| 0 | 320 | final_both_64 | 638 | 638 | -213.4 | -12.86% | -12.87% | 17.9 | -253.4 | -120.9 | 9/9 |
| 0 | 448 | filtering | 892 | 894 | -56.7 | -3.34% | -3.34% | 32.3 | -78.4 | +38.9 | 8/9 |
| 0 | 448 | final_both_64 | 892 | 892 | -178.6 | -10.51% | -10.52% | 30.0 | -219.0 | -156.2 | 9/9 |

## Seed-Level Paired Median
| threshold | tokens | input_seed_group | setting | median delta | wins |
|---|---|---|---|---|---|
| 0 | 320 | 1007 | filtering | -41.3 | 3/3 |
| 0 | 320 | 1007 | final_both_64 | -204.1 | 3/3 |
| 0 | 320 | 2007 | filtering | -53.7 | 3/3 |
| 0 | 320 | 2007 | final_both_64 | -226.8 | 3/3 |
| 0 | 320 | 3007 | filtering | -43.6 | 3/3 |
| 0 | 320 | 3007 | final_both_64 | -209.1 | 3/3 |
| 0 | 448 | 1007 | filtering | -27.6 | 2/3 |
| 0 | 448 | 1007 | final_both_64 | -176.2 | 3/3 |
| 0 | 448 | 2007 | filtering | -65.8 | 3/3 |
| 0 | 448 | 2007 | final_both_64 | -203.0 | 3/3 |
| 0 | 448 | 3007 | filtering | -56.7 | 3/3 |
| 0 | 448 | 3007 | final_both_64 | -178.6 | 3/3 |

## Rank Activation
| threshold | tokens | setting | rank0 true/total | rank1 true/total | num_tokens r0/r1 | recv r0/r1 | critical r0/r1/tie |
|---|---|---|---|---|---|---|---|
| 0 | 320 | original | 0/9 | 0/9 | 638/638 | 638/638 | 5/4/0 |
| 0 | 320 | filtering | 9/9 | 9/9 | 638/638 | 638/638 | 4/5/0 |
| 0 | 320 | final_both_64 | 9/9 | 9/9 | 638/638 | 638/638 | 6/3/0 |
| 0 | 448 | original | 0/9 | 0/9 | 894/892 | 894/892 | 7/2/0 |
| 0 | 448 | filtering | 9/9 | 9/9 | 894/892 | 894/892 | 4/5/0 |
| 0 | 448 | final_both_64 | 9/9 | 9/9 | 894/892 | 894/892 | 5/4/0 |

## Route Stats
| threshold | tokens | valid pairs r0/r1 | invalid pairs r0/r1 |
|---|---|---|---|
| 0 | 320 | 2570/2550 | 2534/2547 |
| 0 | 448 | 3598/3570 | 3554/3566 |

## Positive Delta Outliers
| group | input_seed_group | cycle | setting | delta | base | target | recv r0/r1 | active r0/r1 |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| threshold=0 tokens=448 | 1007 | 1 | filtering | +38.9 | 1711.8 | 1750.7 | 896/890 | True/True |

