# JOINT buy events: same-stock overlapping-window deduplication comparison

"Win rate" at the current buy-research stage is defined as the "selection opportunity win rate": the share of events that traded above, or well above, the buy reference price within the original two-week window. Among the 1198 complete windows after dedup, maximum gain >0: 1167/1198 (97.41%); >=3%: 808/1198 (67.45%); >=5%: 606/1198 (50.58%); >=10%: 298/1198 (24.87%). These measures include the high on the buy day. Excluding that day, with the published day2plus_high_ceiling, 1040/1198 (86.81%) still traded above the buy reference price afterwards. The 44 incomplete/unfilled events are not counted as failures or zero returns.

This report compares the opportunity distribution within the original two-week window after stock selection. The original daily selections and states are unchanged.

Conclusion: overlapping windows of the same stock clearly weight events repeatedly, yet after dedup the mean maximum gain over complete paths moves only from 7.77% to 7.65%, the median from 4.89% to 5.07%, and the share reaching 5% from 48.87% to 50.58%. The typical upside therefore does not disappear after this dedup; the original 3524 selections cannot be read as 3524 independent opportunities.

The correction is larger for recent periods: the mean maximum gain for the partial year 2026 goes 9.38% → 8.33%; for the last complete 12 months, 9.09% → 8.19%. In both, the median and the share reaching 5% do not fall, so the repeated weighting affects the mean and the centre of the distribution differently, and no single average summarises the whole change.

After dedup, the median first-peak day over complete paths is observed trading day 4; among the 896 events with a determinable strict pre-peak interval, the pre-peak minimum return has mean -2.91% and median -2.31%, and 142 first fell at least 5%.

Original 3524 events → 1242 retained, 2282 overlapping events dropped, 0 calendar-unresolved. Original stocks: 596; after retention: 596.

For each stock the event with the earliest signal is retained first; a later event is retained only if its scheduled entry day falls after the end of the original W of the most recently retained event. Dropped events do not extend the window; deduplication runs over the full path before slicing by year. Unfilled events and events whose window has not ended also take part in the date-based deduplication.

| Window | Complete/all: original→dedup | Mean max gain: original→dedup | Median max gain: original→dedup | Reaching 5%: original→dedup | Count reaching 5% after dedup |
|---|---:|---:|---:|---:|---:|
| FULL_FUTURE (2023-01-01—2026-08-21) | 3403/3524 → 1198/1242 | 7.77% → 7.65% | 4.89% → 5.07% | 48.87% → 50.58% | 606/1198 |
| YEAR_2023 (2023-01-01—2023-12-31) | 928/968 → 313/329 | 5.98% → 6.13% | 3.70% → 3.90% | 36.10% → 38.98% | 122/313 |
| YEAR_2024 (2024-01-01—2024-12-31) | 928/968 → 347/360 | 7.43% → 7.74% | 4.89% → 5.21% | 49.14% → 51.30% | 178/347 |
| YEAR_2025 (2025-01-01—2025-12-31) | 971/972 → 345/346 | 8.85% → 8.55% | 6.17% → 6.30% | 56.75% → 56.52% | 195/345 |
| YEAR_2026_PARTIAL (2026-01-01—2026-08-21) | 576/616 → 193/207 | 9.38% → 8.33% | 6.12% → 6.60% | 55.73% → 57.51% | 111/193 |
| RECENT12_OLDER (2024-08-01—2025-07-31) | 928/968 → 359/372 | 8.81% → 8.86% | 5.72% → 5.98% | 53.77% → 54.87% | 197/359 |
| RECENT12_LATEST (2025-08-01—2026-07-31) | 967/968 → 319/320 | 9.09% → 8.19% | 6.27% → 6.33% | 56.15% → 56.74% | 181/319 |
| RECENT24 (2024-08-01—2026-07-31) | 1895/1936 → 678/692 | 8.95% → 8.55% | 5.98% → 6.17% | 54.99% → 55.75% | 378/678 |
| AUG2026_PARTIAL (2026-08-01—2026-08-21) | 20/60 → 6/20 | 7.32% → 10.11% | 7.40% → 10.34% | 75.00% → 83.33% | 5/6 |

The denominator of the gains and shares above is complete paths; unresolved events are not counted as having no gain. The JSON also lists the 3%/5%/10% hit counts, the complete-path shares, and the confirmed-hit lower bound and coverage over all retained events.

| Window | Median first-peak day: original→dedup | Mean pre-peak minimum return: original→dedup | Median pre-peak minimum return: original→dedup | Dedup pre-peak drop ≥5%/determinable | Median full-window minimum return: original→dedup |
|---|---:|---:|---:|---:|---:|
| FULL_FUTURE | 5.00 → 4.00 | -2.82% → -2.91% | -2.23% → -2.31% | 142/896 | -4.72% → -4.88% |
| YEAR_2023 | 5.00 → 4.00 | -2.53% → -2.46% | -1.92% → -1.89% | 26/225 | -4.12% → -4.32% |
| YEAR_2024 | 4.00 → 4.00 | -3.06% → -3.19% | -2.49% → -2.49% | 47/259 | -6.09% → -6.34% |
| YEAR_2025 | 5.00 → 5.00 | -2.69% → -2.91% | -2.19% → -2.36% | 44/276 | -3.73% → -3.96% |
| YEAR_2026_PARTIAL | 5.00 → 4.00 | -3.15% → -3.12% | -2.39% → -2.40% | 25/136 | -6.07% → -5.96% |
| RECENT12_OLDER | 5.00 → 5.00 | -2.99% → -3.08% | -2.35% → -2.46% | 48/288 | -4.18% → -4.34% |
| RECENT12_LATEST | 5.00 → 4.00 | -2.95% → -3.16% | -2.35% → -2.56% | 45/238 | -4.97% → -5.39% |
| RECENT24 | 5.00 → 5.00 | -2.97% → -3.11% | -2.35% → -2.50% | 93/526 | -4.61% → -4.80% |
| AUG2026_PARTIAL | 5.00 → 5.00 | -2.31% → -1.82% | -2.16% → -1.85% | 0/4 | -2.57% → -2.33% |

"Pre-peak" strictly excludes the day on which the high occurs and assumes no intraday order of high and low; when the peak falls on the entry day there may be no determinable pre-peak interval. The first-peak day follows the original upstream day index; a delayed entry does not restart the original window.

## Measurement notes

- Deduplication only removes repeated counting from overlapping W windows of the same stock; same-day co-movement across different stocks remains, so the retained count is not a count of independent samples.
- The maximum gain is an ex post opportunity range; the high on the buy day may not be sellable under T+1. The day2plus and executable-open upper-bound statistics are kept in the JSON.
- The original 2023/2024 year-end label censoring is kept as is; 2026 data end on 2026-08-21, so August 2026 is incomplete.

Per-event retain/drop decisions and anchors: membership_ledger.jsonl; full metrics: dedup_comparison.json.
