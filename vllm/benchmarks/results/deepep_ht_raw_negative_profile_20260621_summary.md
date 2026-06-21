rows=2
settings=['block_both_64', 'block_fixed_both_64']
baseline=block_both_64
pair_key=['threshold', 'input_seed_group', 'cycle', 'tokens']
threshold_values=[0]
input_seed_group_values=[1007]
cycle_values=[1]
tokens_values=[320]
missing=[]

## Critical Path Absolute
| threshold | tokens | setting | median | IQR | min | max |
|---|---|---|---|---|---|---|
| 0 | 320 | block_both_64 | 1609.2 | 0.0 | 1609.2 | 1609.2 |
| 0 | 320 | block_fixed_both_64 | 1446.5 | 0.0 | 1446.5 | 1446.5 |

## Paired Delta
| threshold | tokens | setting | min recv median | critical recv median | median delta | delta/median baseline | median pair pct | IQR | min | max | wins |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 320 | block_fixed_both_64 | 640 | 640 | -162.6 | -10.11% | -10.11% | 0.0 | -162.6 | -162.6 | 1/1 |

## Seed-Level Paired Median
| threshold | tokens | input_seed_group | setting | median delta | wins |
|---|---|---|---|---|---|
| 0 | 320 | 1007 | block_fixed_both_64 | -162.6 | 1/1 |

## Rank Activation
| threshold | tokens | setting | rank0 true/total | rank1 true/total | num_tokens r0/r1 | recv r0/r1 | critical r0/r1/tie |
|---|---|---|---|---|---|---|---|
| 0 | 320 | block_both_64 | 1/1 | 1/1 | 638/637 | 638/637 | 1/0/0 |
| 0 | 320 | block_fixed_both_64 | 1/1 | 1/1 | 640/640 | 640/640 | 1/0/0 |

## Route Stats
| threshold | tokens | valid pairs r0/r1 | invalid pairs r0/r1 |
|---|---|---|---|
| 0 | 320 | 2571/2549 | 2533/2547 |

## Positive Delta Outliers
| group | input_seed_group | cycle | setting | delta | base | target | recv r0/r1 | active r0/r1 |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| none |  |  |  |  |  |  |  |  |

