rows=96
settings=['original', 'compute_both_64', 'fixed_remap_both_64', 'final_raw_both_64']
baseline=original
pair_key=['threshold', 'input_seed_group', 'cycle', 'tokens']
threshold_values=[0]
input_seed_group_values=[1007, 2007, 3007]
cycle_values=[1, 2, 3, 4]
tokens_values=[320, 448]
missing=[]

## Critical Path Absolute
| threshold | tokens | setting | median | IQR | min | max |
|---|---|---|---|---|---|---|
| 0 | 320 | original | 1728.0 | 86.5 | 1705.2 | 2290.0 |
| 0 | 320 | compute_both_64 | 1515.3 | 14.6 | 1507.0 | 2101.4 |
| 0 | 320 | fixed_remap_both_64 | 1545.0 | 81.4 | 1510.1 | 1671.6 |
| 0 | 320 | final_raw_both_64 | 1461.3 | 21.1 | 1440.0 | 1545.6 |
| 0 | 448 | original | 1748.7 | 15.7 | 1737.5 | 1902.8 |
| 0 | 448 | compute_both_64 | 1570.0 | 22.4 | 1556.4 | 1836.3 |
| 0 | 448 | fixed_remap_both_64 | 1596.3 | 64.0 | 1557.5 | 1700.7 |
| 0 | 448 | final_raw_both_64 | 1502.4 | 23.0 | 1478.7 | 1624.5 |

## Paired Delta
| threshold | tokens | setting | min recv median | critical recv median | median delta | delta/median baseline | median pair pct | IQR | min | max | wins |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 320 | compute_both_64 | 638 | 638 | -214.4 | -12.41% | -12.43% | 35.6 | -764.3 | +277.1 | 11/12 |
| 0 | 320 | fixed_remap_both_64 | 640 | 640 | -181.9 | -10.53% | -10.53% | 157.0 | -755.2 | -52.9 | 12/12 |
| 0 | 320 | final_raw_both_64 | 640 | 640 | -268.0 | -15.51% | -15.53% | 49.8 | -850.0 | -206.7 | 12/12 |
| 0 | 448 | compute_both_64 | 892 | 894 | -180.9 | -10.34% | -10.28% | 23.4 | -344.0 | +89.7 | 11/12 |
| 0 | 448 | fixed_remap_both_64 | 896 | 896 | -170.9 | -9.77% | -9.81% | 62.0 | -292.2 | -43.5 | 12/12 |
| 0 | 448 | final_raw_both_64 | 896 | 896 | -248.7 | -14.22% | -14.17% | 31.5 | -413.4 | -123.6 | 12/12 |

## Stepwise Delta
| threshold | tokens | step | source median | target median | median delta | delta/source median | median pair pct | wins |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 320 | Original -> Compute | 1728.0 | 1515.3 | -214.4 | -12.41% | -12.43% | 11/12 |
| 0 | 320 | Compute -> Fixed remap | 1515.3 | 1545.0 | +32.2 | +2.12% | +2.13% | 2/12 |
| 0 | 320 | Fixed remap -> Raw local | 1545.0 | 1461.3 | -95.4 | -6.17% | -6.19% | 11/12 |
| 0 | 320 | Original -> Final | 1728.0 | 1461.3 | -268.0 | -15.51% | -15.53% | 12/12 |
| 0 | 448 | Original -> Compute | 1748.7 | 1570.0 | -180.9 | -10.34% | -10.28% | 11/12 |
| 0 | 448 | Compute -> Fixed remap | 1570.0 | 1596.3 | +36.2 | +2.30% | +2.32% | 4/12 |
| 0 | 448 | Fixed remap -> Raw local | 1596.3 | 1502.4 | -103.2 | -6.47% | -6.47% | 10/12 |
| 0 | 448 | Original -> Final | 1748.7 | 1502.4 | -248.7 | -14.22% | -14.17% | 12/12 |

## Seed-Level Paired Median
| threshold | tokens | input_seed_group | setting | median delta | wins |
|---|---|---|---|---|---|
| 0 | 320 | 1007 | compute_both_64 | -206.0 | 4/4 |
| 0 | 320 | 1007 | fixed_remap_both_64 | -99.9 | 4/4 |
| 0 | 320 | 1007 | final_raw_both_64 | -250.0 | 4/4 |
| 0 | 320 | 2007 | compute_both_64 | -224.4 | 4/4 |
| 0 | 320 | 2007 | fixed_remap_both_64 | -181.9 | 4/4 |
| 0 | 320 | 2007 | final_raw_both_64 | -279.8 | 4/4 |
| 0 | 320 | 3007 | compute_both_64 | -210.7 | 3/4 |
| 0 | 320 | 3007 | fixed_remap_both_64 | -248.5 | 4/4 |
| 0 | 320 | 3007 | final_raw_both_64 | -318.4 | 4/4 |
| 0 | 448 | 1007 | compute_both_64 | -184.4 | 4/4 |
| 0 | 448 | 1007 | fixed_remap_both_64 | -172.6 | 4/4 |
| 0 | 448 | 1007 | final_raw_both_64 | -240.2 | 4/4 |
| 0 | 448 | 2007 | compute_both_64 | -170.0 | 4/4 |
| 0 | 448 | 2007 | fixed_remap_both_64 | -118.8 | 4/4 |
| 0 | 448 | 2007 | final_raw_both_64 | -248.7 | 4/4 |
| 0 | 448 | 3007 | compute_both_64 | -207.0 | 3/4 |
| 0 | 448 | 3007 | fixed_remap_both_64 | -190.4 | 4/4 |
| 0 | 448 | 3007 | final_raw_both_64 | -278.9 | 4/4 |

## Rank Activation
| threshold | tokens | setting | rank0 true/total | rank1 true/total | num_tokens r0/r1 | recv r0/r1 | critical r0/r1/tie |
|---|---|---|---|---|---|---|---|
| 0 | 320 | original | 0/12 | 0/12 | 638/638 | 638/638 | 4/8/0 |
| 0 | 320 | compute_both_64 | 12/12 | 12/12 | 638/638 | 638/638 | 5/7/0 |
| 0 | 320 | fixed_remap_both_64 | 12/12 | 12/12 | 640/640 | 640/640 | 5/7/0 |
| 0 | 320 | final_raw_both_64 | 12/12 | 12/12 | 640/640 | 640/640 | 8/4/0 |
| 0 | 448 | original | 0/12 | 0/12 | 894/892 | 894/892 | 6/6/0 |
| 0 | 448 | compute_both_64 | 12/12 | 12/12 | 894/892 | 894/892 | 5/7/0 |
| 0 | 448 | fixed_remap_both_64 | 12/12 | 12/12 | 896/896 | 896/896 | 9/3/0 |
| 0 | 448 | final_raw_both_64 | 12/12 | 12/12 | 896/896 | 896/896 | 6/6/0 |

## Route Stats
| threshold | tokens | valid pairs r0/r1 | invalid pairs r0/r1 |
|---|---|---|---|
| 0 | 320 | 2570/2550 | 2534/2547 |
| 0 | 448 | 3598/3570 | 3554/3566 |

## Positive Delta Outliers
| group | input_seed_group | cycle | setting | delta | base | target | recv r0/r1 | active r0/r1 |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| threshold=0 tokens=320 | 3007 | 1 | compute_both_64 | +277.1 | 1824.2 | 2101.4 | 638/638 | True/True |
| threshold=0 tokens=448 | 3007 | 1 | compute_both_64 | +89.7 | 1746.6 | 1836.3 | 892/894 | True/True |

