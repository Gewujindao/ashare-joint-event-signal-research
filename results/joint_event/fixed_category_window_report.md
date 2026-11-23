# Fixed-category report: JOINT buy opportunities and paths after de-duplication

This report lays out the de-duplicated joint-event summary by fixed categories.

Sample: non-overlapping original-W events of the same stock. Highs and lows within the window are computed relative to the original entry reference price; two weeks is the upper limit for observation and holding, and holding to the end is not required.

## Observation horizon

Complete original W: at most 10 trading days and no more than 14 calendar days; the table below is split by historical period, and W is not restarted for delayed entry.

## A Buy opportunities: within the fixed original W

| Historical period | Complete/all (incomplete) | Ever above entry price, incl. entry day | Above entry price from next trading day | Ever up ≥3% | Ever up ≥5% | Ever up ≥10% |
|---|---:|---:|---:|---:|---:|---:|
| 2023-01-01—2023-12-31 | 313/329 (16) | 299/313 (95.53%) | 265/313 (84.66%) | 184/313 (58.79%) | 122/313 (38.98%) | 50/313 (15.97%) |
| 2024-01-01—2024-12-31 | 347/360 (13) | 339/347 (97.69%) | 296/347 (85.30%) | 238/347 (68.59%) | 178/347 (51.30%) | 90/347 (25.94%) |
| 2025-01-01—2025-12-31 | 345/346 (1) | 341/345 (98.84%) | 314/345 (91.01%) | 249/345 (72.17%) | 195/345 (56.52%) | 99/345 (28.70%) |
| 2026-01-01—2026-08-21 (partial period) | 193/207 (14) | 188/193 (97.41%) | 165/193 (85.49%) | 137/193 (70.98%) | 111/193 (57.51%) | 59/193 (30.57%) |
| 2024-08-01—2025-07-31 | 359/372 (13) | 351/359 (97.77%) | 329/359 (91.64%) | 262/359 (72.98%) | 197/359 (54.87%) | 114/359 (31.75%) |
| 2025-08-01—2026-07-31 | 319/320 (1) | 313/319 (98.12%) | 277/319 (86.83%) | 224/319 (70.22%) | 181/319 (56.74%) | 90/319 (28.21%) |
| 2024-08-01—2026-07-31 | 678/692 (14) | 664/678 (97.94%) | 606/678 (89.38%) | 486/678 (71.68%) | 378/678 (55.75%) | 204/678 (30.09%) |
| 2026-08-01—2026-08-21 (partial period) | 6/20 (14) | 6/6 (100.00%) | 6/6 (100.00%) | 6/6 (100.00%) | 5/6 (83.33%) | 4/6 (66.67%) |
| 2023-01-01—2026-08-21 (overall) | 1198/1242 (44) | 1167/1198 (97.41%) | 1040/1198 (86.81%) | 808/1198 (67.45%) | 606/1198 (50.58%) | 298/1198 (24.87%) |

The denominator is the complete windows in each row. Incomplete and unfilled samples are not counted in the denominator; lower bounds on hits for partial observations are stored separately in the original JSON. The windows overlap one another. The opportunity share is the share of events that reached the corresponding gain at some point within the window.

## B Raw paths for stop-loss learning

| Historical period | Max gain mean/median | Full-window max drawdown mean/median (raw path) | Strict pre-peak drawdown mean/median | Pre-peak determinable count | Median peak day |
|---|---:|---:|---:|---:|---:|
| 2023-01-01—2023-12-31 | 6.13%/3.90% | 4.87%/4.32% | 2.46%/1.89% | 225 | 4.00 |
| 2024-01-01—2024-12-31 | 7.74%/5.21% | 8.04%/6.34% | 3.19%/2.49% | 259 | 4.00 |
| 2025-01-01—2025-12-31 | 8.55%/6.30% | 5.00%/3.96% | 2.91%/2.36% | 276 | 5.00 |
| 2026-01-01—2026-08-21 | 8.33%/6.60% | 7.44%/5.96% | 3.12%/2.40% | 136 | 4.00 |
| 2024-08-01—2025-07-31 | 8.86%/5.98% | 5.64%/4.34% | 3.08%/2.46% | 288 | 5.00 |
| 2025-08-01—2026-07-31 | 8.19%/6.33% | 6.68%/5.39% | 3.16%/2.56% | 238 | 4.00 |
| 2024-08-01—2026-07-31 | 8.55%/6.17% | 6.13%/4.80% | 3.11%/2.50% | 526 | 5.00 |
| 2026-08-01—2026-08-21 | 10.11%/10.34% | 2.85%/2.33% | 1.82%/1.85% | 4 | 5.00 |
| 2023-01-01—2026-08-21 | 7.65%/5.07% | 6.24%/4.88% | 2.91%/2.31% | 896 | 4.00 |

Drawdowns are shown as positive numbers giving the decline relative to the entry reference price. The full-window max drawdown is computed on the raw path held to the end of the window. The pre-peak measure strictly excludes the peak day itself; no inference is made about the order of the high and low within the same day.

## C Actual holding and exit

Results of the old fixed exit model on the original event definition (not de-duplicated): the model was trained on 2021-09 to 2022-12, subsequently validated on 2023/2024, final status DROP_FIXED_Q_EXIT_STACK.

## Training and evaluation periods

The original JOINT buy model was frozen in 2021-06; the 2023—2026 results in this report are a subsequent evaluation of that fixed model on already-seen history. The learning/evaluation dates of the exit model are listed separately in C.

