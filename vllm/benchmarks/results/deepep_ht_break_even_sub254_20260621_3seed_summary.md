rows=168
settings=['baseline', 'global_ignore']
baseline=baseline
pair_key=['threshold', 'input_seed_group', 'cycle', 'tokens']
threshold_values=[0]
input_seed_group_values=[1007, 2007, 3007]
cycle_values=[1, 2, 3, 4]
tokens_values=[8, 16, 32, 48, 64, 96, 128]
missing=[]

## Critical Path Absolute
| threshold | tokens | setting | median | IQR | min | max |
|---|---|---|---|---|---|---|
| 0 | 8 | baseline | 1302.6 | 152.7 | 1247.4 | 1476.8 |
| 0 | 8 | global_ignore | 1283.9 | 62.1 | 1247.7 | 1375.9 |
| 0 | 16 | baseline | 1340.8 | 44.2 | 1301.6 | 1638.8 |
| 0 | 16 | global_ignore | 1317.2 | 56.4 | 1271.1 | 1425.7 |
| 0 | 32 | baseline | 1396.9 | 60.0 | 1355.2 | 1539.2 |
| 0 | 32 | global_ignore | 1359.9 | 39.6 | 1334.3 | 1463.1 |
| 0 | 48 | baseline | 1411.9 | 88.9 | 1370.6 | 1536.9 |
| 0 | 48 | global_ignore | 1363.5 | 34.8 | 1331.0 | 1469.8 |
| 0 | 64 | baseline | 1437.7 | 21.8 | 1415.7 | 1537.6 |
| 0 | 64 | global_ignore | 1387.0 | 35.9 | 1362.4 | 1422.0 |
| 0 | 96 | baseline | 1422.1 | 30.1 | 1400.2 | 1539.4 |
| 0 | 96 | global_ignore | 1386.3 | 60.8 | 1370.4 | 1486.8 |
| 0 | 128 | baseline | 1463.4 | 57.6 | 1424.7 | 1585.2 |
| 0 | 128 | global_ignore | 1425.4 | 69.9 | 1390.0 | 1503.9 |

## Paired Delta
| threshold | tokens | setting | min recv median | critical recv median | median delta | delta/median baseline | median pair pct | IQR | min | max | wins |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 8 | global_ignore | 16 | 16 | -28.8 | -2.21% | -2.23% | 86.8 | -161.1 | +108.7 | 9/12 |
| 0 | 16 | global_ignore | 31 | 32 | -21.1 | -1.57% | -1.58% | 35.8 | -272.3 | +54.0 | 9/12 |
| 0 | 32 | global_ignore | 63 | 64 | -19.3 | -1.38% | -1.38% | 38.1 | -100.7 | +17.9 | 10/12 |
| 0 | 48 | global_ignore | 95 | 95 | -47.8 | -3.38% | -3.41% | 60.1 | -177.4 | +45.4 | 10/12 |
| 0 | 64 | global_ignore | 127 | 127 | -48.9 | -3.40% | -3.41% | 40.7 | -116.8 | -13.1 | 12/12 |
| 0 | 96 | global_ignore | 191 | 191 | -30.9 | -2.17% | -2.19% | 34.3 | -167.5 | +57.3 | 9/12 |
| 0 | 128 | global_ignore | 254 | 255 | -46.3 | -3.17% | -3.22% | 90.8 | -116.0 | +44.7 | 8/12 |

## Seed-Level Paired Median
| threshold | tokens | input_seed_group | setting | median delta | wins |
|---|---|---|---|---|---|
| 0 | 8 | 1007 | global_ignore | -77.3 | 3/4 |
| 0 | 8 | 2007 | global_ignore | -48.1 | 3/4 |
| 0 | 8 | 3007 | global_ignore | -5.7 | 3/4 |
| 0 | 16 | 1007 | global_ignore | +0.2 | 2/4 |
| 0 | 16 | 2007 | global_ignore | -29.0 | 4/4 |
| 0 | 16 | 3007 | global_ignore | -22.4 | 3/4 |
| 0 | 32 | 1007 | global_ignore | -20.7 | 3/4 |
| 0 | 32 | 2007 | global_ignore | -47.2 | 4/4 |
| 0 | 32 | 3007 | global_ignore | -13.7 | 3/4 |
| 0 | 48 | 1007 | global_ignore | -27.2 | 3/4 |
| 0 | 48 | 2007 | global_ignore | -81.0 | 3/4 |
| 0 | 48 | 3007 | global_ignore | -33.8 | 4/4 |
| 0 | 64 | 1007 | global_ignore | -34.8 | 4/4 |
| 0 | 64 | 2007 | global_ignore | -64.8 | 4/4 |
| 0 | 64 | 3007 | global_ignore | -67.3 | 4/4 |
| 0 | 96 | 1007 | global_ignore | -40.7 | 4/4 |
| 0 | 96 | 2007 | global_ignore | -6.9 | 2/4 |
| 0 | 96 | 3007 | global_ignore | -30.5 | 3/4 |
| 0 | 128 | 1007 | global_ignore | +12.4 | 1/4 |
| 0 | 128 | 2007 | global_ignore | -83.6 | 4/4 |
| 0 | 128 | 3007 | global_ignore | -39.7 | 3/4 |

## Rank Activation
| threshold | tokens | setting | rank0 true/total | rank1 true/total | num_tokens r0/r1 | recv r0/r1 | critical r0/r1/tie |
|---|---|---|---|---|---|---|---|
| 0 | 8 | baseline | 0/12 | 0/12 | 16/16 | 16/16 | 6/6/0 |
| 0 | 8 | global_ignore | 12/12 | 12/12 | 16/16 | 16/16 | 4/8/0 |
| 0 | 16 | baseline | 0/12 | 0/12 | 31/32 | 31/32 | 7/5/0 |
| 0 | 16 | global_ignore | 12/12 | 12/12 | 31/32 | 31/32 | 7/5/0 |
| 0 | 32 | baseline | 0/12 | 0/12 | 63/64 | 63/64 | 5/7/0 |
| 0 | 32 | global_ignore | 12/12 | 12/12 | 63/64 | 63/64 | 5/7/0 |
| 0 | 48 | baseline | 0/12 | 0/12 | 95/95 | 95/95 | 6/6/0 |
| 0 | 48 | global_ignore | 12/12 | 12/12 | 95/95 | 95/95 | 8/4/0 |
| 0 | 64 | baseline | 0/12 | 0/12 | 127/127 | 127/127 | 5/7/0 |
| 0 | 64 | global_ignore | 12/12 | 12/12 | 127/127 | 127/127 | 6/6/0 |
| 0 | 96 | baseline | 0/12 | 0/12 | 191/191 | 191/191 | 8/4/0 |
| 0 | 96 | global_ignore | 12/12 | 12/12 | 191/191 | 191/191 | 8/4/0 |
| 0 | 128 | baseline | 0/12 | 0/12 | 255/255 | 255/255 | 7/5/0 |
| 0 | 128 | global_ignore | 12/12 | 12/12 | 255/255 | 255/255 | 6/6/0 |

## Route Stats
| threshold | tokens | valid pairs r0/r1 | invalid pairs r0/r1 |
|---|---|---|---|
| 0 | 8 | 68/60 | 59/68 |
| 0 | 16 | 128/128 | 122/128 |
| 0 | 32 | 257/255 | 255/257 |
| 0 | 48 | 378/390 | 382/378 |
| 0 | 64 | 509/515 | 507/501 |
| 0 | 96 | 776/760 | 759/768 |
| 0 | 128 | 1035/1013 | 1003/1021 |

## Positive Delta Outliers
| group | input_seed_group | cycle | setting | delta | base | target | recv r0/r1 | active r0/r1 |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| threshold=0 tokens=8 | 1007 | 4 | global_ignore | +15.5 | 1301.7 | 1317.2 | 15/16 | True/True |
| threshold=0 tokens=8 | 2007 | 4 | global_ignore | +51.9 | 1247.4 | 1299.3 | 16/16 | True/True |
| threshold=0 tokens=8 | 3007 | 1 | global_ignore | +108.7 | 1255.3 | 1363.9 | 16/16 | True/True |
| threshold=0 tokens=16 | 1007 | 1 | global_ignore | +23.4 | 1338.0 | 1361.5 | 31/32 | True/True |
| threshold=0 tokens=16 | 1007 | 3 | global_ignore | +54.0 | 1371.8 | 1425.7 | 31/32 | True/True |
| threshold=0 tokens=16 | 3007 | 4 | global_ignore | +4.6 | 1304.8 | 1309.3 | 32/32 | True/True |
| threshold=0 tokens=32 | 1007 | 4 | global_ignore | +17.9 | 1417.1 | 1435.0 | 63/64 | True/True |
| threshold=0 tokens=32 | 3007 | 4 | global_ignore | +2.7 | 1356.6 | 1359.3 | 64/64 | True/True |
| threshold=0 tokens=48 | 1007 | 3 | global_ignore | +23.7 | 1446.1 | 1469.8 | 95/95 | True/True |
| threshold=0 tokens=48 | 2007 | 3 | global_ignore | +45.4 | 1370.6 | 1416.0 | 95/96 | True/True |
| threshold=0 tokens=96 | 2007 | 1 | global_ignore | +2.6 | 1446.9 | 1449.5 | 191/191 | True/True |
| threshold=0 tokens=96 | 2007 | 3 | global_ignore | +57.3 | 1400.2 | 1457.5 | 191/191 | True/True |
| threshold=0 tokens=96 | 3007 | 3 | global_ignore | +26.5 | 1408.0 | 1434.5 | 192/191 | True/True |
| threshold=0 tokens=128 | 1007 | 1 | global_ignore | +44.7 | 1459.2 | 1503.9 | 254/255 | True/True |
| threshold=0 tokens=128 | 1007 | 2 | global_ignore | +13.3 | 1424.7 | 1438.0 | 254/255 | True/True |
| threshold=0 tokens=128 | 1007 | 3 | global_ignore | +11.6 | 1467.9 | 1479.5 | 254/255 | True/True |
| threshold=0 tokens=128 | 3007 | 4 | global_ignore | +13.5 | 1442.2 | 1455.7 | 255/254 | True/True |

