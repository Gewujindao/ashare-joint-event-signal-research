# Label form: a once-trained LightGBM with three labels: return regression, cross-sectional rank regression, up/down binary classification (2022–2026 window)

## Conclusion and account results


**Comparison conclusion: the rank label passes; binary classification does not.** Against return MSE, rank L2 wins the within-year return/drawdown ratio in 4/5 periods and rank IC in 5/5 periods, with both winning in the same period in 4 periods; for binary classification the figures are 0/5, 1/5 and 0 periods. The five periods are **the four full years 2022–2025 + the partial period of 2026 up to 8-21**. The comparison uses rule B (enter in the top 10%, exit on dropping out of the top 50%); A's 1% probability/rank threshold is not used to decide the outcome.


For the B account with a full 500,000 CNY budget and 0.1% slippage per side, rank L2 ends at **1,235,132 CNY**, annualised **21.59%**, max drawdown **27.03%**; return MSE gives **745,869 CNY, 9.03%, 50.82%**. Across ten start dates rank gives 19.30%–23.49% annualised, better than return MSE's 7.90%–9.78%. Rank's last 12 months are **−10.42%** and the 2026 partial year **−11.54%**; the full-budget drawdown exceeds the historical risk reference of about 17%; with half the budget the historical drawdown falls to 14.02%, with 10.78% annualised, and the last 12 months lost 5.11%.


The rank label has a higher overall IC (0.1038 vs 0.0753); top-decile gross return per R5 label is **0.339% vs 0.347%** respectively. The account improvement shows up together as lower turnover, different holding paths and longer holding: rank averages 10.48 trading days, return MSE 6.85 days. This line of work is a cross-sectional learning experiment.


### Rule-B main account: same capital, same window, same costs


| Score | Final capital/CNY | Annualised | Max drawdown | Longest underwater/trading days | Last 12 months | Prior 12 months | Combined 24 months |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Return MSE | 745,869.44 | 9.03% | 50.82% | 385 | 5.53% | 94.63% | 105.38% |
| Rank L2 | 1,235,132.25 | 21.59% | 27.03% | 146 | -10.42% | 88.02% | 68.42% |
| Up/down classification | 228,130.94 | -15.60% | 56.71% | 1116 | -26.84% | 27.68% | -6.58% |
| Fixed Ridge | 1,222,192.12 | 21.31% | 38.13% | 319 | -3.04% | 125.24% | 118.40% |
| Fixed HGB | 691,854.50 | 7.27% | 46.78% | 386 | 13.35% | 74.68% | 97.99% |


Last 12 months = 2025-08-22 to 2026-08-21; prior 12 months = 2024-08-22 to 2025-08-21. Fixed Ridge and rank L2 have similar full-period annualised returns (21.31% and 21.59%); rank has a shallower drawdown and is worse over the last 12/24 months. The gross daily equal-weight benchmark has 0.36% annualised, 38.11% drawdown and 1.67% cumulative, as market context.


### Trade quality and costs (B main account)


| Score | Closed trades | Win rate | Mean net return per trade | Mean gain | Mean loss | Mean gain/loss | Gross profit/gross loss | Mean holding/days | Mean position | Explicit fees/CNY | Annualised two-sided turnover/× |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Return MSE | 1593 | 45.64% | 0.40% | 7.31% | -5.40% | 1.35 | 1.10 | 6.85 | 95.32% | 61,746.95 | 67.01 |
| Rank L2 | 1047 | 51.48% | 1.05% | 6.66% | -4.91% | 1.36 | 1.36 | 10.48 | 96.62% | 64,290.11 | 43.98 |
| Up/down classification | 2704 | 43.42% | -0.20% | 4.08% | -3.49% | 1.17 | 0.87 | 3.42 | 86.19% | 75,924.26 | 116.50 |
| Fixed Ridge | 1133 | 47.22% | 0.97% | 8.59% | -5.84% | 1.47 | 1.23 | 9.69 | 97.29% | 65,088.96 | 46.88 |
| Fixed HGB | 1462 | 45.28% | 0.35% | 7.42% | -5.50% | 1.35 | 1.08 | 7.46 | 95.15% | 53,168.87 | 61.53 |


Per-trade statistics include closed trades only; final capital also includes the market value of open positions; longest underwater is counted in trading days. Two-sided turnover is buy plus sell amount / mean NAV, divided by the number of sample years. Exit reasons in the rank main account: SIGNAL 62.0%, 20-day cap TIME 26.8%, STOP_LOSS 11.2%. Exit reasons follow the shared engine; ST, limit-down and the model leaving the holding zone are recorded together as SIGNAL.


### Year-by-year verdict (B main account)


| Year | MSE return | MSE drawdown | MSE ratio | MSE IC | Rank return | Rank drawdown | Rank ratio | Rank IC | Classification return | Classification drawdown | Classification ratio | Classification IC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2022 | -0.01% | 24.94% | -0.001 | 0.0703 | 16.49% | 16.52% | 0.998 | 0.1011 | -12.49% | 26.01% | -0.480 | 0.0529 |
| 2023 | -9.98% | 23.93% | -0.417 | 0.0704 | 8.75% | 10.59% | 0.826 | 0.0966 | -12.65% | 28.71% | -0.441 | 0.0620 |
| 2024 | 14.76% | 38.34% | 0.385 | 0.0929 | 73.51% | 24.08% | 3.053 | 0.1249 | -23.03% | 40.10% | -0.574 | 0.0677 |
| 2025 | 36.45% | 14.21% | 2.564 | 0.0795 | 27.05% | 9.23% | 2.931 | 0.1149 | 3.90% | 15.81% | 0.247 | 0.0835 |
| 2026 (to 8-21) | 5.83% | 24.80% | 0.235 | 0.0556 | -11.54% | 24.98% | -0.462 | 0.0674 | -25.36% | 31.91% | -0.795 | 0.0462 |


The ratio is the year's return / max drawdown within that year, with drawdown measured from the previous year-end NAV (initial capital in the first year); 2026 is a partial year and is not annualised. Rank loses to MSE on the ratio only in 2026; binary classification has a higher IC than MSE only in 2025.


### Six accounts and cost stress (rule B)


| Label | Capital/CNY | Budget fraction | Slippage per side | Final capital/CNY | Annualised | Drawdown | Last 12 months |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Return MSE | 500,000 | 1.0 | 0.10% | 745,869.44 | 9.03% | 50.82% | 5.53% |
| Return MSE | 500,000 | 1.0 | 0.20% | 569,033.94 | 2.83% | 49.91% | -2.32% |
| Return MSE | 500,000 | 0.5 | 0.10% | 622,685.81 | 4.86% | 26.07% | 3.52% |
| Return MSE | 500,000 | 0.5 | 0.20% | 531,672.50 | 1.34% | 29.76% | -0.13% |
| Return MSE | 20,000 | 1.0 | 0.10% | 20,777.29 | 0.83% | 66.44% | 46.20% |
| Return MSE | 20,000 | 1.0 | 0.20% | 19,491.88 | -0.55% | 64.20% | 38.05% |
| Rank L2 | 500,000 | 1.0 | 0.10% | 1,235,132.25 | 21.59% | 27.03% | -10.42% |
| Rank L2 | 500,000 | 1.0 | 0.20% | 962,662.12 | 15.21% | 29.19% | -14.94% |
| Rank L2 | 500,000 | 0.5 | 0.10% | 802,899.38 | 10.78% | 14.02% | -5.11% |
| Rank L2 | 500,000 | 0.5 | 0.20% | 718,823.69 | 8.16% | 15.36% | -7.39% |
| Rank L2 | 20,000 | 1.0 | 0.10% | 52,343.48 | 23.11% | 26.19% | 15.99% |
| Rank L2 | 20,000 | 1.0 | 0.20% | 52,345.77 | 23.11% | 24.65% | 10.05% |
| Up/down classification | 500,000 | 1.0 | 0.10% | 228,130.94 | -15.60% | 56.71% | -26.84% |
| Up/down classification | 500,000 | 1.0 | 0.20% | 128,989.00 | -25.38% | 75.12% | -34.24% |
| Up/down classification | 500,000 | 0.5 | 0.10% | 346,618.56 | -7.61% | 32.57% | -14.36% |
| Up/down classification | 500,000 | 0.5 | 0.20% | 265,553.44 | -12.78% | 47.92% | -18.70% |
| Up/down classification | 20,000 | 1.0 | 0.10% | 4,871.79 | -26.30% | 78.44% | -33.97% |
| Up/down classification | 20,000 | 1.0 | 0.20% | 2,750.93 | -34.87% | 87.53% | -40.90% |


The 20,000 CNY accounts are affected by board lots, the minimum commission and cash constraints, so their holding paths differ after costs are doubled: the two generated ledgers differ in buy/sell quantities or number of holdings on 32 days, the first being 2022-09-15.


### Ten adjacent start dates (B; 500,000 CNY full budget, 0.1% slippage)


| Score | Median annualised | Annualised min–max | Median drawdown | Max drawdown |
| --- | --- | --- | --- | --- |
| Return MSE | 8.88% | 7.90%–9.78% | 50.65% | 50.82% |
| Rank L2 | 20.72% | 19.30%–23.49% | 28.11% | 28.19% |
| Up/down classification | -15.49% | -15.80%–-15.14% | 56.66% | 56.93% |
| Fixed Ridge | 21.34% | 20.58%–22.48% | 38.11% | 38.15% |
| Fixed HGB | 8.71% | 7.27%–9.78% | 46.82% | 47.03% |


### Signal layer and comparison with main rule A


| Score | Top-decile gross label | Bottom-decile gross label | Spread | Rank IC | Return-unit R²oos | Share of days with >1% candidates |
| --- | --- | --- | --- | --- | --- | --- |
| Return MSE | 0.35% | -0.53% | 0.88% | 0.0753 | -0.0295 | 44.94% |
| Rank L2 | 0.34% | -0.59% | 0.93% | 0.1038 | Different units, not interpreted | 100.00% |
| Up/down classification | 0.22% | -0.25% | 0.47% | 0.0639 | Different units, not interpreted | 100.00% |
| Fixed Ridge | 0.34% | -0.57% | 0.91% | 0.0822 | 0.0039 | 94.18% |
| Fixed HGB | 0.29% | -0.50% | 0.80% | 0.0676 | -0.0412 | 46.73% |


The candidate shares above are computed over 1,117 matured-label days (1,123 account days). Under rule A, annualised/drawdown is 9.39%/20.96% for return MSE, 18.30%/36.84% for fixed Ridge and −7.01%/51.04% for fixed HGB. Return MSE has a lower return and a shallower drawdown. The A rows for rank/bin are kept in accounts.csv to illustrate the scale issue.


### Non-overlapping 12-month windows, annual fees and trading limits


| Rank B window (first and last actual trading day) | Full 12 months | Return | Window drawdown | Explicit fees/CNY | Window two-sided turnover/× |
| --- | --- | --- | --- | --- | --- |
| 2022-01-04—2023-01-03 | Yes | 15.80% | 16.52% | 9,281.93 | 48.24 |
| 2023-01-04—2024-01-03 | Yes | 10.18% | 10.59% | 6,876.22 | 29.78 |
| 2024-01-04—2025-01-03 | Yes | 66.07% | 24.08% | 15,225.38 | 54.01 |
| 2025-01-06—2025-12-31 | Yes | 31.79% | 9.23% | 16,712.21 | 36.79 |
| 2026-01-05—2026-08-21 | Partial period | -11.54% | 24.98% | 16,194.39 | 33.59 |


Windows are cut consecutively from 2022-01-04, every 12 calendar months; trading days roll forward over holidays / end at the previous trading day; the two most recent full 12-month periods are shown in the table above. The same windows for all 60 accounts are in the blocks12 field of accounts.csv.


| Rank B year | Return | Drawdown | Explicit fees/CNY | Two-sided turnover in the year/× |
| --- | --- | --- | --- | --- |
| 2022 | 16.49% | 16.52% | 9,281.93 | 48.25 |
| 2023 | 8.75% | 10.59% | 6,831.39 | 29.60 |
| 2024 | 73.51% | 24.08% | 14,966.00 | 53.32 |
| 2025 | 27.05% | 9.23% | 17,016.42 | 37.51 |
| 2026 | -11.54% | 24.98% | 16,194.39 | 33.59 |


Pending-sale "position × trading day" counts in the B main accounts: MSE 25, rank 10, classification 933; rejected buys 232, 159, 262; positions terminated by the end of the data series 0, 0, 2. In the classification account, long pending sales and valuation at the carried-forward terminal price affect its loss path. All three B main accounts end with 10 open positions.


### Coverage, checks and evidence


Development panel `mainboard_2014_2026` (2014-01-02 to 2026-08-21), 3,073 days × 3,404 stocks, feature set C (108 features). The account window is 1,123 days, 3,822,692 day/stock cells in total; the eligibility mask covers 2,632,854 cells; 3,259 stocks were eligible at some point in the window and 145 never were. Cells with a finite label: 2,609,421; eligible but missing a label: 23,433; matured-label days: 1,117, last day 2026-08-13; the last 6 days do not enter the signal layer. The three LightGBM models have no missing scores on eligible cells; the above is the coverage of this panel.


Checks: Ridge top decile 0.338% and IC 0.08225 agree with the fixed2021 Ridge results in `docs/model_comparison/ml_ensemble_fixed.md`; the final values and explicit fees of the 60 account ledgers match the summary one by one, and the results for all 100 start dates are complete.


- Output directory: `output/research_once/label_form_20261009/`.
- `signal_by_year.csv`, `daily_<score>.csv`, `signals_check.json`: year-by-year/daily signals and reconciliation with earlier results.
- `accounts.csv`: summary of the 60 accounts, including annual_detail, blocks12, the last 24 months and trading limits.
- `ledger_<rule>_<score>_<capital>_g<fraction>_slip<friction>.csv`: daily NAV, cash and trade records for all 60 accounts.
- `nav_primary.csv`, `trades_<rule>_<score>.csv`: 10 main-account equity curves and trade details; `nav_ew_benchmark.csv` is the gross equal-weight context.
- `yearly_ratio.csv`, `criterion.json`: year-by-year verdict; `start_shifts.csv`, `start_shift_summary.csv` are the 100 start-date results.
- `model_<label>.pkl`, `scores_<label>.npy`, `fit.json`, `train_log.txt`: outputs of `train`.


## Notes

This panel has already been used by several studies, so all results are **already-seen development results**; "pass" in this study refers to the label-form comparison under the criteria above. The cross-sectional model studies are a comparison line to existing work.

- **Cache and fixed models**: the three models, the three `[3073, 3404]` float32 score arrays, `fit.json` and the training log are complete; model metadata all show `trained_before=2022-01-01` and 600,000 rows. A single 2021 validation selected 200 trees for MSE, 270 for rank and 40 for binary classification, which were then used unchanged.
- **Label maturity**: the original R5 label uses at latest the open of signal day `t+6`; the filter is `t+6 < cutoff`. The last training signal day was 2020-12-23 in the validation stage and 2021-12-20 in final training; 2021 validation labels matured by 2021-12-31 at latest. The R5 label ignores sell delays caused by limit-down and inherits the label-availability condition "buyable on the next day and finite future label", so signal-layer statistics are conditional on label availability; in the accounts, the engine handles rejected buys and pending sales separately.
- **Method sources**: [Jiang–Kelly–Xiu published paper](https://economics.yale.edu/sites/default/files/2023-11/The%20Journal%20of%20Finance%20-%202023%20-%20JIANG%20-%20Re%25E2%2580%2590%20Imag%20in%20ing%20Price%20Trends_0.pdf), pp. 3205–3206: directional binary classification, a single training/validation on 1993–2000, testing on 2001–2019 without periodic retraining. The label form is borrowed from it here. [Du preprint](https://arxiv.org/html/2506.06356v1), Section III-B-3, mixes listwise ranking with MSE; the percentile L2 in this line is a self-defined pointwise approximation; the original paper's date split has overlap. [Gu–Kelly–Xiu original](https://dachxiu.chicagobooth.edu/download/ML.pdf) provides return prediction and the R² definition against a zero-return benchmark; [Novy-Marx–Velikov](https://www.nber.org/papers/w20721) supports the cost mechanism of a stricter buy threshold than hold threshold, and the 10%/50% band in this project follows that mechanism.
- **Fair verdict**: all five scores use the B rank band as the decision rule for label form; in A, comparing rank/bin scores with 0.01 is only an illustrative row, since a 1% probability and a 1% return have different units, and it is not part of the verdict. "rank/bin has no level information" means no magnitude level in units of return; binary-classification probabilities contain directional confidence information.
- **Costs**: slippage 0.1%/0.2% per side, commission of one basis point with a 5 CNY minimum, transfer fee 0.001% per side, seller stamp duty fixed at 0.05% (historical tax rates are not switched in). The fee field is explicit commission/taxes; slippage is included in the fill price.
- **Account outputs**: each account exports a daily ledger (NAV, cash, number of holdings, buy/sell amounts, fees, rejected-buy and pending-sale counts), plus annual fees/turnover, non-overlapping 12-month windows, the combined last 24 months, the mean gain/loss ratio and the number of open positions at the end. The t-values of signal IC are not corrected for serial correlation; the 0.272% after-fee column uses a fixed round-trip cost for large trades.
- **Data and valuation**: forward-adjusted price ratios serve as the proxy for holding value; positions terminated by the end of the data are valued at the engine's last price. The equal-weight comparison is reweighted daily by the previous day's mask and is gross of fees. Final capital includes the market value of open positions; no forced liquidation on the last day is assumed.

The account part reuses existing scores: 160 account simulations in total, with no refitting.

## Criteria

(2) Rank L2 (`lgb_rank`) and (3) up/down classification (`lgb_bin`) are each compared with (1) return MSE (`lgb_mse`). Years = 2022, 2023, 2024, 2025, 2026 (partial year to 8-21, counted as one year); the within-year ratio uses the main account under rank-band rule B (500,000 CNY holding 10 stocks, fully invested, 0.1% slippage, from 2022-01-04): each year's after-fee return ÷ max drawdown within that year (drawdown measured from the previous year-end NAV; for 2022 from initial capital); rank IC is the signal layer's mean daily Spearman rank IC for each year. **A label form "wins" if and only if, among the 5 years, in ≥ 3 years its within-year ratio is higher than (1), and, among the 5 years, in ≥ 3 years its rank IC is also higher than (1)**; ties do not count as higher. The number of years in which both hold in the same year is for information only. Differences between LightGBM return regression and fixed2021 Ridge/HGB under the main rule are reported only, not judged.

## Sources and mechanisms

1. **Directional binary classification (Jiang-Kelly-Xiu 2023, JF 78(6))**: US stocks 1993–2019; daily OHLC drawn as 5/20/60-day candlestick images; a CNN classifies "whether the next 5/20/60-day return is positive", **trained once on 1993–2000 and tested on 2001–2019 without retraining** (figures are second-hand: equal-weight decile long-short annualised Sharpe for the 5-day model about 7.2; restricted to the largest 500 stocks, "slightly above 1"). Mechanism: classification only learns "up or down" and is not pulled around by the magnitude of extreme returns, so it is naturally robust to heavy-tailed labels; the cost is losing magnitude and level information: the predicted probability carries no abstention meaning such as "expected return > 1%". (3) in this study moves this label form onto our 5-day trade label, with the model replaced by boosted trees (image CNNs are outside the scope of this line).
2. **Ranking objective (Du 2025, arXiv 2506.06356)**: all Shanghai/Shenzhen A-shares 2010–2024; a feed-forward network with a weighted listwise ranking loss and MSE (α=0.7), daily re-evaluation and multi-day holding; cost per trade 22.1 bp, annual turnover 2,100%, fees 30.5% of gross return (single-author preprint whose validation period overlaps the training period; used only as an order-of-magnitude reference). Mechanism: stock selection needs only the cross-sectional relative order; using rank as the target makes the model spend its capacity on "who ranks first" rather than on fitting absolute returns. (2) in this study implements the same idea as "L2 regression on the day's cross-sectional percentile rank" (the simplest pointwise ranking objective, without lambdarank's grouping hyperparameters, so that it differs from (1) and (3) only in the y vector).
3. **Return-regression baseline (Gu-Kelly-Xiu 2020, RFS)**: the training target is (Huber) return regression, R²oos = 1 − Σ(y−ŝ)²/Σy² (not demeaned); the best monthly US model, NN3, reaches only 0.40%. In Leippold-Wang-Zhou (2022, JFE), GBRT has the highest R²oos on Chinese monthly data at 2.71%, but almost all of it comes from 2018, and it is negative in the top-70% market-cap subsample, a reminder that this study should look year by year rather than only at the aggregate.
4. **Why the account comparison uses the rank-band rule**: A comparison of fixed Ridge with annually retrained Ridge on the same panel found no difference in ranking (top-decile gross return 0.338% vs 0.338%, rank IC 0.082 vs 0.073); the account gap comes from the **level** of the scores: out of 1,123 days, the annual model had 368 days on which no stock was predicted above 1%, so it opened no new positions. Rank and probability labels have no level, so the current "score > 1%" entry is an empty condition for them, and applying the main rule directly would mix "no abstention" into "label form". The Novy-Marx-Velikov (2016/2019) buy/hold band (enter the top 10%, sell only on dropping out of the top 50%) is a prior for fast signals, and the 10/50 band had already been run on fixed2021 Ridge (it did not pass the criteria of that test, but it is the same rule that all five scores can face), so this study uses it as the rule for a fair comparison.
5. **Study design**: a ranking objective and directional classification, as LightGBM with three labels (the raw 5-day return, the cross-sectional rank and an up/down binary label), comparing top-decile net return and turnover.

## Method (script `scripts/research/label_form.py`)

- `train`: reads the XC memmap, rawT5 and the mask; sampling days and rows are chosen as in `fixed()` (from the 60th trading day, one day every 5 days). On each sampling day, eligible stocks with finite labels are taken, features are converted to float32 with missing values filled with 0.5, and three y are generated at the same time: the return clipped at ±0.2, the day's pandas percentile rank, and whether it is > 0. Validation stage: each of the three labels is trained on 523,642 rows with 300 trees (num_leaves 31, learning_rate 0.05, min_child_samples 100, feature_fraction 0.8, bagging_fraction 0.8, seed 0); after the training matrix is freed, the 2021 validation matrix is built (237 days, ≥ 50 eligible stocks), predictions are made at 10…300 trees, the mean daily raw label of the top 10% is computed and the maximum is taken (ties go to fewer trees); after the validation matrix is freed, the final matrix is built (of 610,053 rows, the last 600,000 are taken), the three models are retrained with the selected tree counts and scored daily from 2022-01-04 (the classifier uses the positive-class probability). Outputs `scores_<label>.npy` ([days × stocks] float32, NaN outside the mask), `model_<label>.pkl`, `fit.json` (validation curves, tree counts), `train_log.txt`.
- `signals`: five scores (three LightGBM, fixed2021 Ridge, fixed2021 HGB), 2022-01-04 to 2026-08-21; eligible cells = mask with finite label and score; only days with ≥ 50 eligible stocks count (in practice only the last 6 days without labels are skipped). Annual summaries: R²oos (all cells in the year), mean daily top/bottom-decile label (k = max(1, int(0.1n))), spread, top decile net of a 0.272% round-trip cost, mean daily Spearman rank IC and its t-value, share of days with positive IC, share of days with "score > 1%" candidates and the mean daily candidate count, mean daily score. Outputs `signal_by_year.csv`, `daily_<score>.csv`, `signals_check.json` (reconciliation with the fixed2021 Ridge results).
- `accounts`: five scores × two rules (main rule A: enter on score > 1%, exit on ST, limit-down, missing score, ≤ 0 or dropping out of the day's top 30%; rank band B: enter in the day's top 10% by rank, exit on ST, limit-down, missing score or rank > 50%; both with an 8% stop-loss and at most 20 days) × 6 accounts + 10 start dates. The accounting function uses the same definitions as `ashare_daily_judgment.account` (entry signals passed in externally). Equal-weight benchmark: each day, the stocks in the previous day's mask, forward-adjusted close to close, equally weighted, with no fees. Year-by-year ratio: main account (500,000 CNY fully invested, 0.1% slippage, from 2022-01-04), each year's return ÷ within-year drawdown (measured from the previous year-end NAV). Outputs `accounts.csv`, `start_shifts.csv`, `start_shift_summary.csv`, `nav_primary.csv`, `nav_ew_benchmark.csv`, `trades_<rule>_<score>.csv`, `yearly_ratio.csv`, `criterion.json`.
