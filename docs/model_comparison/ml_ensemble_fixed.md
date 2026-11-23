# Fixed-model ensemble: daily z-score average of fixed2021 Ridge and fixed2021 HGB (2022-01-04 to 2026-08-21)

**This study takes a daily z-score average of two existing component scores**: the two component scores are existing, once-trained outputs (`ashare_cost_aware_exit_20261009/scores_fixed2021_*.npy`). Everything is already-seen data on the development panel `mainboard_2014_2026`.

## Conclusions

1. **Under the criteria, the ensemble does not pass.** The criteria require both conditions to hold: (a) in the main account (500,000 CNY holding 10 stocks, fully invested, 0.1% slippage, from 2022-01-04), annualised ÷ drawdown is higher than both components: **holds** (ensemble 0.579, Ridge 0.497, HGB −0.137; medians over 10 start dates 0.581 vs 0.489 vs −0.126, same direction); (b) in the five years 2022–2026, rank IC is higher than both components at the same time in at least 3 years: **does not hold, only in one year, 2023** (ensemble 0.0827 vs Ridge 0.0823, HGB 0.0671). In the other four years the ensemble's rank IC is below Ridge (2022 0.083 vs 0.092, 2025 0.099 vs 0.104, 2026 0.049 vs 0.053) or below HGB (2024 0.0839 vs 0.0843).
2. **At the signal level the ensemble does not improve ranking.** 2022–2026 combined: mean daily top-decile gross return, ensemble 0.333% vs Ridge 0.338% (HGB 0.295%); top-minus-bottom spread 0.914% vs 0.911%; rank IC 0.0820 vs 0.0823. The only improvement is in R²oos as defined by GKX: ensemble 0.437% vs Ridge 0.390% (close to GKX 2020's "ensemble 0.43% vs single model 0.40%"; GKX's ensemble also wins on portfolio Sharpe, while the ensemble in this study does not win on ranking metrics). R²oos is dominated by cells with large |y|, and the improvement in mean squared error does not bring an improvement in selection at the top.
3. **The +3.4 percentage points of annualised return at the account level (21.7% vs 18.3%) come from the two years 2024 and 2025, with no corresponding improvement at the signal level.** By year: ensemble 2022 +15.1%, 2023 +12.1%, 2024 +12.5%, 2025 +78.9%, 2026 −4.4%; Ridge +24.9%, +17.5%, +2.7%, +51.1%, −4.4%. The ensemble trails in 2022 and 2023 by 10 and 5 percentage points, is level in 2026, and its whole advantage is in 2024 (+9.8) and 2025 (+27.8). A post-hoc comparison at account depth (among each day's > 1% candidates, the 10 highest-scoring stocks, on the 912 days when both have candidates): the 5-day gross return of the ensemble's selected stocks is 0.067 percentage points **lower** than Ridge's (Newey-West t −0.78; in 2025, 0.28 percentage points lower, t −1.36; higher only in 2024 and 2026, by 0.15 and 0.28 percentage points, t 1.0). So the account advantage comes from the following two mechanisms rather than from entry selection: (i) **the ensemble has more no-candidate days** (192 days vs 65 days, mean daily candidates 87 vs 104): after the two z-scores are averaged the right tail thins, and after rescaling to Ridge's mean and standard deviation fewer stocks exceed 1%; this is another form of the "level / fewer trades" effect of fixed Ridge versus annual retraining (see Sources and mechanisms), not GKX's variance hedging; (ii) different exit paths (the rule of exiting when the score drops out of the top 30% depends on rank, and the ensemble's ranks change faster: mean holding 7.5 days vs 8.5 days, SIGNAL exits 77% vs 73%, turnover 263× vs 240×), which in 2025 pushed the mean net return per trade from +1.88% to +2.38% and the win rate from 56% to 60%, while in the same year the day-level selection labels were in fact worse. These two mechanisms are not in the criteria; the account advantage is concentrated in two years, the same pattern as the one described by Leippold (2022): "GBRT's advantage comes almost entirely from 2018".
4. **The fixed HGB model on its own is unusable under the current rule because of its score level, not its ranking.** HGB's mean daily score swings between −8.2% and +4.4% (cross-day standard deviation of the daily mean 1.19%, Ridge only 0.25%); at the 1% threshold, out of 1,123 days, 596 days have no candidates and 264 days have more than 500 candidates; the account has −7.0% annualised, 51% drawdown, 48% position share, 2.6 days mean holding, and 94% of sells are "score drops out of the top 30% or ≤ 0": its scores drift up and down as a whole every day, so a position is judged to be weakening after one day. Placing HGB on Ridge's mean and standard deviation (sensitivity row S1, ranking unchanged): 6.2% annualised, 56% drawdown, turnover 379×, far worse than Ridge; at account depth S1's selection is 0.20 percentage points below Ridge (t −1.2; in 2022, 0.71 percentage points lower, t −3.1). So HGB's ranking itself is also worse and less stable (mean daily rank correlation 0.58, top-10% set overlap only 32%), consistent with Leippold's "tree models are rejected more often in CSPA and are unstable". HGB is also worse than Ridge when both are retrained annually.
5. **Comparison**: main-board equal weight (gross, zero cost, previous-day mask) 2022–2026 annualised +0.15%, drawdown 38.1%, longest underwater 876 days (the NAV is flat on the start day; including the return of 2022-01-04 itself, +0.97%, gives +0.4%/38.1%, with identical drawdown and underwater days). Ridge and the ensemble are both far above equal weight; HGB raw scores are below equal weight.
6. **Method-level conclusions**: (a) a fixed-model ensemble that averages "linear + tree" z-scores 1:1 does not improve ranking on local daily data, and GKX's ensemble effect does not replicate here; (b) any rule thresholded at "prediction > 1%" is extremely sensitive to score level, and levels must be aligned before two models are compared (otherwise the comparison is of levels, not of models); (c) for a result that wins at the account level but not at the signal level, check the number of no-candidate days and the exit paths first, before discussing the model.

**Files**: script `scripts/research/ml_ensemble_fixed.py` (`signals`, `accounts`, `posthoc`); outputs in `output/research_once/ml_ensemble_fixed_20261009/`: `scores_ensemble_fixed2021.npy`, `scores_hgb_on_ridge_scale_fixed2021.npy`, `ensemble_construction_daily.csv`, `signal_by_year.csv`, `r2_by_year.csv`, `daily_signal.csv`, `agreement_by_year.csv`, `agreement_daily.csv`, `accounts.csv`, `start_shifts.csv`, `start_shift_summary.csv`, `nav_primary.csv`, `trades_primary_<set>.csv`, `benchmark.csv`, `criterion.json`, `posthoc_top10_by_year.csv`, `posthoc_top10_diff.csv`, `run_log_signals.txt`, `run_log_accounts.txt`, `run_log_posthoc.txt`, `signals_run_info.json`.

## Criteria

The ensemble passes only if both of the following hold: (a) in the main account with 500,000 CNY holding 10 stocks, g = 1.0, 0.1% slippage, from 2022-01-04, the ensemble's annualised ÷ drawdown is strictly higher than Ridge's and also strictly higher than HGB's (the main rule buys exactly the highest-scoring candidates, so this account's net annualised ÷ drawdown is the after-fee performance of the top candidates); (b) in the five years 2022, 2023, 2024, 2025, 2026, rank IC is strictly higher than both Ridge and HGB in at least 3 years. The median over 10 start dates and the other 5 accounts serve only as robustness checks, reported but not part of the verdict; S1 is not part of the verdict.

## Sources and mechanisms

- **Gu, Kelly, Xiu (2020, RFS), Section 4.4**: an equal-weight ensemble of 11 models has a monthly R²oos of 0.43%, above the best single model, NN3, at 0.40%; the equal-weight decile long-short Sharpe is 2.49, above any single model. The mechanism is that the prediction errors of different model families have a correlation below 1, so averaging reduces variance. This study uses the two closest families: regularised linear (Ridge) and gradient-boosted trees (HGB), with the same features, the same label and the same training rows.
- **Leippold, Wang, Zhou (2022, JFE)**: on the full Chinese monthly sample GBRT has the highest R²oos at 2.71%, but the authors point out that its advantage comes almost entirely from 2018; in the top-70% market-cap subsample GBRT −0.38% and RF −0.04% are negative, while regularised linear models and neural networks are positive; in conditional CSPA tests, tree models are rejected when conditioning on market variance (GBRT 4 times, NN4 2 times). So the marginal information of tree models may be both useful and unstable, and averaging with a stable linear model is a natural hedge: this is the hypothesis of this study.
- **Chen-Pelger-Zhu (2024, MS), Jiang-Kelly-Xiu (2023, JF)**: models trained once and then fixed lose only a little Sharpe over 19–25-year US test periods, which supports the "train once, no rolling" setting; both components in this study use this setting.
- **Fixed Ridge vs. annual retraining**: fixed Ridge and annual retraining show no gap in ranking; the account gap comes from score level (the annual model opens no positions on low-prediction days). So the ensemble can only change ranking, not level: the ensemble score takes Ridge's mean and standard deviation each day. When the cross-sectional statistics of the two scores were inspected before the run (sampled once every 50 days, without involving return outcomes), HGB's daily mean was found to swing widely (Ridge's daily mean between −0.3% and +0.6%, standard deviation about 0.4%; HGB's daily mean between −2.4% and +1.4%), which made "level alignment" necessary rather than merely cautious.
- **Cost literature**: the trading rules and costs are those of the shared engine (commission of one basis point with a 5 CNY minimum, stamp duty 0.05% on sells, transfer fee, slippage of 0.1% and 0.2% per side).

## Method

- **Data**: panel `mainboard_2014_2026` (3,073 days × 3,404 stocks); mask `ashare_ml_signals_v1_20261006/mainboard_2014_2026_mask.npy`; label `ashare_ml_signals_v2_20261007/mainboard_2014_2026_rawT5.npy`; component scores `ashare_cost_aware_exit_20261009/scores_fixed2021_C_RIDGE_HR5.npy` (Ridge α=1000, 610,053 training rows) and `scores_fixed2021_HGB_C_HR5.npy` (HGB: learning_rate 0.02, 600 trees, 31 leaves, min_samples_leaf 20, L2 10, 600,000 training rows; all hyperparameters were selected only once, in 2021, by the mean top-10% label). The finite cells of the two scores are identical (2,632,854 cells, all inside the mask, from 2022-01-04; built-in check in the script).
- **Ensemble**: on each day t, on the cells in the mask ∧ finite in both scores, z_R and z_H are cross-sectional z-scores (population standard deviation), z = (z_R + z_H)/2, ensemble = mean_R + std_R × z/std(z). Built-in check: on sampled days the ensemble's cross-sectional mean and standard deviation differ from Ridge's by < 1e-9. The mean cross-sectional standard deviation of z (before rescaling) is 0.903 (a direct consequence of the mean daily correlation of 0.58 between the two components). S1 = mean_R + std_R × z_H. Both are stored as float32 `.npy`.
- **Signal layer**: eligible cells = mask ∧ finite label ∧ finite score; R²oos aggregates all eligible cells by year, 1 − Σ(y−s)²/Σy², not demeaned; daily top/bottom-decile mean labels with k = max(1, int(0.1n)) and the spread; mean, standard deviation and t of the daily Spearman(s, y); days with fewer than 50 eligible cells are skipped (in practice only the last 6 days of each score, which have no matured labels, are skipped; for 2026 this means month 1 through 8-13). Candidate count = mask ∧ finite score ∧ score > 1% (no label required).
- **Agreement**: daily Spearman and Pearson correlation between Ridge and HGB (cells in the mask where both are finite); top-10% set overlap |top_R ∩ top_H|/k.
- **Accounts**: the main rule of `ashare_daily_judgment` (entry: score finite and > 1%, free slots filled from the highest score down, open ≤ previous close × 1.01; exit: `risk_events` ∨ `weak_signal(q=0.3)`, 8% stop-loss, 20 days; `universe=mask`), engine `ashare_engine.simulate` unchanged. Built-in check: the Ridge baseline gives the same final capital and number of trades through this script's `account()` and through `dj.account()` (1,088,004.1 CNY, 1,254 trades, matching, in `window2022_accounts.csv`, the fixed2021 q0.3 row). Grid: 500,000 CNY holding 10 with g=1.0/0.5, 20,000 CNY holding 2, slippage 0.1%/0.2%; 10 start dates (2022-01-04 to 01-17, 10 consecutive trading days, 500,000 CNY fully invested, 0.1% slippage).
- **Comparison**: main-board equal weight = the close-to-close return of day t, equally weighted over mask[t−1] (no same-day look-ahead; stocks that are suspended or lack a price are left out of that day's average), zero cost, NAV flat at 500,000 CNY on the start day.
- **Verdict**: `criterion.json`: (a) main-account annualised ÷ drawdown strictly higher than both components; (b) the number of years among the five in which rank IC is strictly higher than both components is ≥ 3. S1 is not included.

## Result tables

### Table 1 Signal layer (by year; top/bottom decile and all are mean daily labels, gross; candidates = mean daily number of stocks with score > 1%; no candidates = days in the year without candidates)

| Score | Year | Days | R²oos | Top decile | Bottom decile | Spread | Rank IC | IC t | Mean daily candidates | No-candidate days |
|---|---|---|---|---|---|---|---|---|---|---|
| Ridge | 2022 | 242 | 0.337% | 0.276% | −0.874% | 1.150% | 0.0917 | 19.2 | 102.6 | 15 |
| Ridge | 2023 | 242 | −0.313% | 0.106% | −0.632% | 0.738% | 0.0823 | 15.7 | 36.8 | 18 |
| Ridge | 2024 | 242 | 0.902% | 0.227% | −0.549% | 0.776% | 0.0686 | 8.9 | 180.0 | 23 |
| Ridge | 2025 | 243 | 0.713% | 0.949% | −0.292% | 1.241% | 0.1040 | 18.1 | 81.5 | 9 |
| Ridge | 2026 | 148 | −0.224% | −0.001% | −0.486% | 0.485% | 0.0532 | 6.2 | 137.2 | 0 |
| Ridge | 2022–2026 | 1117 | 0.390% | 0.338% | −0.573% | 0.911% | 0.0823 | 28.7 | 105.1 | 65 |
| HGB | 2022 | 242 | −3.797% | 0.043% | −0.732% | 0.775% | 0.0599 | 10.0 | 400.3 | 121 |
| HGB | 2023 | 242 | −3.864% | −0.006% | −0.504% | 0.498% | 0.0671 | 8.8 | 179.8 | 175 |
| HGB | 2024 | 242 | −2.870% | 0.508% | −0.669% | 1.178% | 0.0843 | 7.8 | 357.1 | 136 |
| HGB | 2025 | 243 | −4.464% | 0.843% | −0.141% | 0.984% | 0.0766 | 11.4 | 472.0 | 109 |
| HGB | 2026 | 148 | −6.173% | −0.053% | −0.444% | 0.391% | 0.0392 | 4.0 | 719.2 | 54 |
| HGB | 2022–2026 | 1117 | −4.116% | 0.295% | −0.502% | 0.797% | 0.0677 | 18.2 | 401.0 | 595 |
| Ensemble | 2022 | 242 | 0.286% | 0.200% | −0.839% | 1.039% | 0.0826 | 15.9 | 90.8 | 33 |
| Ensemble | 2023 | 242 | −0.330% | 0.037% | −0.605% | 0.642% | 0.0827 | 13.0 | 25.7 | 58 |
| Ensemble | 2024 | 242 | 1.126% | 0.392% | −0.660% | 1.052% | 0.0839 | 10.1 | 166.5 | 61 |
| Ensemble | 2025 | 243 | 0.720% | 0.924% | −0.283% | 1.207% | 0.0991 | 17.2 | 54.6 | 32 |
| Ensemble | 2026 | 148 | −0.216% | −0.033% | −0.484% | 0.451% | 0.0490 | 5.2 | 115.4 | 8 |
| Ensemble | 2022–2026 | 1117 | 0.437% | 0.333% | −0.582% | 0.914% | 0.0820 | 26.4 | 88.5 | 192 |
| S1 (HGB ranking, Ridge scale) | 2022 | 242 | 0.084% | 0.043% | −0.732% | 0.775% | 0.0599 | 10.0 | 90.4 | 38 |
| S1 | 2023 | 242 | −0.503% | −0.006% | −0.504% | 0.498% | 0.0671 | 8.8 | 23.6 | 77 |
| S1 | 2024 | 242 | 1.213% | 0.508% | −0.669% | 1.178% | 0.0843 | 7.8 | 164.9 | 68 |
| S1 | 2025 | 243 | 0.527% | 0.843% | −0.141% | 0.984% | 0.0766 | 11.4 | 47.5 | 43 |
| S1 | 2026 | 148 | −0.275% | −0.053% | −0.444% | 0.391% | 0.0392 | 4.0 | 115.1 | 15 |
| S1 | 2022–2026 | 1117 | 0.339% | 0.295% | −0.502% | 0.797% | 0.0677 | 18.2 | 86.0 | 241 |

The Ridge rows reproduce the fixed2021 Ridge numbers of the fixed-vs-annual-retraining comparison (built-in consistency check). HGB's R²oos is negative throughout because of level error: its annual mean score ranges from −0.43% to +0.23% and its cross-sectional standard deviation is 1.1%–1.6% (Ridge 0.44%–0.59%), while the cell standard deviation of the label is about 5%. S1's rank IC and deciles are the same as HGB's (ranking unchanged), and its R²oos turning positive shows that after level alignment HGB's mean squared error is comparable to Ridge's (0.339% vs 0.390%).

**Verdict (b)**: the only year in which the ensemble's rank IC is higher than both components is 2023 (0.0827 > 0.0823, 0.0671); 2022, 2025 and 2026 are below Ridge, 2024 is below HGB (0.0839 vs 0.0843). 1/5, not passed.

### Table 2 Agreement of the two components (by year)

| Year | Days | Spearman mean | Std | Min | Pearson mean | Top-10% overlap mean | Std | Min | Mean daily candidates Ridge | Mean daily candidates HGB | HGB no-candidate days | HGB days with candidates ≥ 500 | Ridge no-candidate days |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2022 | 242 | 0.591 | 0.143 | 0.025 | 0.649 | 0.340 | 0.112 | 0.017 | 102.6 | 400.3 | 121 | 58 | 15 |
| 2023 | 242 | 0.557 | 0.140 | 0.129 | 0.617 | 0.303 | 0.105 | 0.073 | 36.8 | 179.8 | 175 | 27 | 18 |
| 2024 | 242 | 0.563 | 0.201 | −0.320 | 0.625 | 0.311 | 0.131 | 0.000 | 180.0 | 357.1 | 136 | 52 | 23 |
| 2025 | 243 | 0.593 | 0.186 | −0.082 | 0.645 | 0.316 | 0.133 | 0.038 | 81.5 | 472.0 | 109 | 63 | 9 |
| 2026 | 154 | 0.587 | 0.153 | 0.066 | 0.646 | 0.333 | 0.151 | 0.069 | 146.5 | 723.7 | 55 | 64 | 0 |
| 2022–2026 | 1123 | 0.577 | 0.168 | −0.320 | 0.635 | 0.319 | 0.126 | 0.000 | 106.6 | 403.3 | 596 | 264 | 65 |

The daily rank correlation of the two models is 0.58 and their top-10% sets overlap by about one third: the error correlation is not high, so the precondition of the ensemble (disagreement) is met; the absence of improvement shows that HGB's part of the disagreement does not, on average, contain more information about the label (its IC is lower). The cross-day standard deviation of HGB's daily mean is 1.19% (Ridge 0.25%), with extremes −8.2% / +4.4% (`ensemble_construction_daily.csv`).

### Table 3 Accounts (current main rule; 2022-01-04 to 2026-08-21; fees in CNY; turnover = total buy and sell amount in the window ÷ mean NAV)

| Score | Capital/holdings | g | Slippage | Final | Annualised | Drawdown | Annualised÷drawdown | Longest underwater | Trades | Win rate | Mean per trade | Mean gain/loss | Gain/loss ratio | Mean holding | Position share | Fees | Turnover | Last 12 months | Prior 12 months |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Ridge (baseline) | 500,000/10 | 1.0 | 0.1% | 1,088,004 | 18.3% | 36.8% | 0.497 | 340 | 1,254 | 48.6% | +0.75% | +7.27% / −5.40% | 1.21 | 8.5 | 95.0% | 66,392 | 240 | +0.9% | +105.2% |
| Ridge | 500,000/10 | 1.0 | 0.2% | 850,615 | 12.2% | 39.2% | 0.311 | 399 | 1,257 | 47.7% | +0.57% | +7.19% / −5.45% | 1.14 | 8.5 | 94.9% | 58,132 | 242 | −3.7% | +96.2% |
| Ridge | 500,000/10 | 0.5 | 0.1% | 767,846 | 9.7% | 19.8% | 0.491 | 317 | 1,254 | 48.6% | +0.74% | +7.25% / −5.41% | 1.23 | 8.5 | 47.4% | 32,344 | 121 | +1.1% | +44.8% |
| Ridge | 500,000/10 | 0.5 | 0.2% | 680,573 | 6.9% | 21.2% | 0.325 | 377 | 1,257 | 47.6% | +0.56% | +7.19% / −5.46% | 1.17 | 8.5 | 47.3% | 31,146 | 122 | −1.1% | +41.6% |
| Ridge | 20,000/2 | 1.0 | 0.1% | 30,101 | 9.2% | 61.0% | 0.151 | 577 | 246 | 47.2% | +0.47% | +7.48% / −5.78% | 1.12 | 8.8 | 93.4% | 3,594 | 229 | +24.9% | +127.7% |
| Ridge | 20,000/2 | 1.0 | 0.2% | 23,232 | 3.3% | 61.6% | 0.054 | 951 | 246 | 46.3% | +0.29% | +7.39% / −5.84% | 1.03 | 8.8 | 92.9% | 3,427 | 229 | +21.0% | +112.6% |
| HGB (raw) | 500,000/10 | 1.0 | 0.1% | 357,189 | −7.0% | 51.0% | −0.137 | 1,122 | 2,120 | 43.6% | −0.09% | +4.08% / −3.32% | 0.91 | 2.6 | 48.2% | 62,581 | 417 | −18.0% | +30.7% |
| HGB | 500,000/10 | 1.0 | 0.2% | 240,183 | −14.7% | 58.3% | −0.252 | 1,122 | 2,116 | 40.8% | −0.30% | +4.13% / −3.36% | 0.81 | 2.6 | 47.9% | 54,883 | 412 | −24.7% | +18.8% |
| HGB | 500,000/10 | 0.5 | 0.1% | 437,806 | −2.8% | 28.2% | −0.100 | 969 | 2,114 | 43.1% | −0.11% | +4.12% / −3.31% | 0.93 | 2.6 | 23.6% | 44,838 | 204 | −8.9% | +14.6% |
| HGB | 500,000/10 | 0.5 | 0.2% | 355,458 | −7.1% | 34.1% | −0.209 | 1,122 | 2,111 | 40.4% | −0.31% | +4.18% / −3.34% | 0.83 | 2.6 | 23.5% | 42,297 | 202 | −13.1% | +8.9% |
| HGB | 20,000/2 | 1.0 | 0.1% | 8,755 | −16.4% | 56.9% | −0.287 | 1,122 | 396 | 40.4% | −0.30% | +4.53% / −3.58% | 0.79 | 2.7 | 44.4% | 5,224 | 363 | −29.5% | −0.9% |
| HGB | 20,000/2 | 1.0 | 0.2% | 6,283 | −22.1% | 69.5% | −0.319 | 1,122 | 387 | 38.5% | −0.52% | +4.33% / −3.55% | 0.69 | 2.7 | 43.4% | 4,918 | 356 | −33.2% | −9.0% |
| **Ensemble** | 500,000/10 | 1.0 | 0.1% | 1,240,636 | 21.7% | 37.5% | 0.579 | 230 | 1,361 | 48.6% | +0.83% | +7.15% / −5.16% | 1.26 | 7.5 | 90.7% | 73,247 | 263 | +9.0% | +128.5% |
| Ensemble | 500,000/10 | 1.0 | 0.2% | 918,967 | 14.1% | 38.7% | 0.364 | 257 | 1,365 | 47.2% | +0.60% | +7.11% / −5.21% | 1.17 | 7.5 | 90.6% | 61,825 | 263 | +1.6% | +116.5% |
| Ensemble | 500,000/10 | 0.5 | 0.1% | 814,767 | 11.1% | 20.1% | 0.553 | 224 | 1,361 | 48.4% | +0.80% | +7.14% / −5.15% | 1.28 | 7.5 | 45.1% | 34,904 | 131 | +4.7% | +53.1% |
| Ensemble | 500,000/10 | 0.5 | 0.2% | 704,702 | 7.7% | 20.9% | 0.368 | 231 | 1,367 | 46.9% | +0.58% | +7.10% / −5.18% | 1.19 | 7.5 | 45.0% | 33,296 | 132 | +1.9% | +48.8% |
| Ensemble | 20,000/2 | 1.0 | 0.1% | 52,439 | 23.2% | 42.2% | 0.549 | 372 | 277 | 46.6% | +0.98% | +7.78% / −4.94% | 1.36 | 7.7 | 91.2% | 4,527 | 257 | +71.5% | +95.6% |
| Ensemble | 20,000/2 | 1.0 | 0.2% | 27,832 | 7.4% | 50.9% | 0.145 | 883 | 278 | 44.2% | +0.52% | +7.51% / −5.04% | 1.13 | 7.7 | 89.9% | 3,932 | 255 | +60.0% | +67.5% |
| S1 (sensitivity) | 500,000/10 | 1.0 | 0.1% | 659,270 | 6.2% | 56.4% | 0.109 | 733 | 1,927 | 45.9% | +0.22% | +5.87% / −4.57% | 1.06 | 4.9 | 82.6% | 66,789 | 379 | +13.2% | +79.1% |
| S1 | 500,000/10 | 1.0 | 0.2% | 469,256 | −1.4% | 62.6% | −0.022 | 908 | 1,927 | 44.8% | +0.04% | +5.82% / −4.66% | 0.97 | 4.9 | 82.4% | 57,219 | 376 | +4.9% | +69.2% |
| S1 | 500,000/10 | 0.5 | 0.1% | 637,026 | 5.4% | 30.5% | 0.176 | 676 | 1,925 | 46.2% | +0.25% | +5.87% / −4.57% | 1.11 | 4.9 | 40.6% | 43,789 | 185 | +7.8% | +38.3% |
| S1 | 500,000/10 | 0.5 | 0.2% | 528,358 | 1.2% | 36.2% | 0.033 | 756 | 1,927 | 44.9% | +0.06% | +5.85% / −4.65% | 1.02 | 4.9 | 40.5% | 41,429 | 185 | +3.2% | +32.1% |
| S1 | 20,000/2 | 1.0 | 0.1% | 26,331 | 6.1% | 55.9% | 0.110 | 381 | 389 | 45.5% | +0.23% | +6.20% / −4.75% | 1.03 | 4.9 | 78.0% | 5,791 | 356 | +12.3% | +22.5% |
| S1 | 20,000/2 | 1.0 | 0.2% | 14,703 | −6.4% | 61.3% | −0.105 | 784 | 384 | 43.2% | −0.11% | +6.15% / −4.88% | 0.91 | 4.8 | 77.5% | 5,369 | 349 | −25.4% | +32.9% |
| Main-board equal weight (gross, previous-day mask) | 500,000 | — | 0 | 503,457 | +0.15% | 38.1% | 0.004 | 876 | — | — | — | — | — | — | — | 0 | — | −1.0% | +57.9% |

Exit reasons (main account): Ridge SIGNAL 73.0% / stop-loss 16.3% / 20 days 10.7%; HGB 94.3% / 5.7% / 0; ensemble 77.4% / 13.9% / 8.7%; S1 87.6% / 9.9% / 2.5%. Days with unfilled pending orders 11 / 9 / 27 / 40; rejected buy orders 137 / 254 / 162 / 252.

### Table 4 By year (main account, 500,000 CNY fully invested, 0.1% slippage)

| Score | 2022 | 2023 | 2024 | 2025 | 2026 (to 8-21) |
|---|---|---|---|---|---|
| Ridge | +24.9% | +17.5% | +2.7% | +51.1% | −4.4% |
| HGB | −30.3% | −14.6% | +33.3% | +12.8% | −20.2% |
| Ensemble | +15.1% | +12.1% | +12.5% | +78.9% | −4.4% |
| S1 | −20.5% | −15.6% | +36.0% | +45.6% | −0.7% |
| Main-board equal weight | −14.7% | −3.1% | +0.7% | +28.2% | −5.6% |

Year-end NAV ratio of ensemble ÷ Ridge: 2022 0.921, 2023 0.879, 2024 0.963, 2025 1.140, 2026-08 1.140. The drawdown troughs are all on 2024-02-07 (Ridge 36.8%, peak 2023-07-13; ensemble 37.5%, peak 2023-11-20; HGB and S1 fall all the way from the start, 51%/56%).

### Table 5 Start-date sensitivity (10 start dates, 500,000 CNY fully invested, 0.1% slippage)

| Score | Median annualised | Min / max | Median drawdown | Max | Median annualised÷drawdown | Median turnover | Median holding |
|---|---|---|---|---|---|---|---|
| Ridge | 18.0% | 17.3% / 18.3% | 36.8% | 36.9% | 0.489 | 239 | 8.5 |
| HGB | −6.3% | −7.0% / −5.0% | 50.0% | 51.0% | −0.126 | 416 | 2.6 |
| Ensemble | 21.8% | 21.1% / 23.0% | 37.5% | 37.5% | 0.581 | 262 | 7.5 |
| S1 | 6.9% | 6.2% / 7.6% | 55.6% | 56.6% | 0.124 | 376 | 4.9 |

**Verdict (a)**: ensemble 0.579 > Ridge 0.497 and > HGB −0.137, holds; start-date medians 0.581 > 0.489 > −0.126, same direction. **Overall verdict: (a) holds, (b) does not hold → not passed** (`criterion.json`).

## Notes

1. HGB's account rows mainly reflect the **level** of its scores (see conclusion 4); for the ranking comparison see the S1 rows and the IC in Table 1: ranking is also worse, but by a margin (IC 0.068 vs 0.082) much smaller than the gap in account performance.
2. The ensemble score takes Ridge's mean and standard deviation, but the shape of the distribution differs (the right tail thins after z-averaging), so there are 16% fewer candidates than for Ridge and 127 more no-candidate days. This comes from the design choice of "rescaling to equal standard deviation", which makes the ensemble's account result partly inherit the "fewer trades" effect of the score level.
3. In the agreement diagnostics, around 2024-04 there are days with negative daily rank correlation (minimum −0.32) and days with a top-10% overlap of 0; on those days HGB's predictions point in the opposite direction to Ridge's.

## Post-hoc analysis: selection at account depth

Purpose: to explain "the account wins, the signal does not". Each day, candidates = mask ∧ finite label ∧ score > 1%, selected = the min(10, number of candidates) highest-scoring candidates; the 5-day gross return of the selected stocks is examined (label rawT5); subtracting 0.272% (the fixed cost of one buy and one sell) gives the net column. Differences are computed on days when both scores have candidates, and additionally restricted to days when both select exactly 10 stocks (same depth). Newey-West lag 10.

| Score | Year | No-candidate days | Mean daily candidates | Mean daily selected | Top-10 gross return | Top-30 gross return | Top-10 after fees |
|---|---|---|---|---|---|---|---|
| Ridge | 2022 | 15 | 102.4 | 8.72 | 0.515% | 0.420% | 0.243% |
| Ridge | 2023 | 18 | 36.8 | 7.75 | 0.042% | −0.023% | −0.230% |
| Ridge | 2024 | 23 | 175.7 | 8.16 | 0.423% | 0.358% | 0.151% |
| Ridge | 2025 | 9 | 81.4 | 8.83 | 0.941% | 0.913% | 0.669% |
| Ridge | 2026 | 0 | 137.0 | 9.53 | −0.234% | −0.178% | −0.506% |
| Ridge | 2022–2026 | 65 | 104.1 | 8.53 | 0.384% | 0.338% | 0.112% |
| Ensemble | 2022 | 33 | 90.6 | 8.33 | 0.353% | 0.324% | 0.081% |
| Ensemble | 2023 | 58 | 25.7 | 7.03 | 0.145% | 0.021% | −0.127% |
| Ensemble | 2024 | 61 | 162.1 | 8.48 | 0.668% | 0.669% | 0.396% |
| Ensemble | 2025 | 32 | 54.6 | 7.85 | 0.821% | 0.915% | 0.549% |
| Ensemble | 2026 | 8 | 115.3 | 8.51 | 0.214% | 0.120% | −0.058% |
| Ensemble | 2022–2026 | 192 | 87.5 | 8.02 | 0.459% | 0.435% | 0.187% |
| S1 | 2022 | 38 | 90.1 | 8.49 | −0.145% | −0.093% | −0.417% |
| S1 | 2023 | 77 | 23.6 | 7.56 | −0.000% | 0.046% | −0.272% |
| S1 | 2024 | 68 | 160.2 | 8.43 | 0.800% | 0.705% | 0.528% |
| S1 | 2025 | 43 | 47.4 | 8.02 | 0.934% | 0.953% | 0.662% |
| S1 | 2026 | 15 | 115.0 | 8.68 | 0.040% | −0.017% | −0.232% |
| S1 | 2022–2026 | 241 | 84.9 | 8.22 | 0.344% | 0.342% | 0.072% |

| Pair | Year | Days with candidates for both | Top-10 difference (percentage points) | NW t | Days both select 10 | Same-depth difference | NW t |
|---|---|---|---|---|---|---|---|
| Ensemble − Ridge | 2022 | 208 | −0.274 | −1.83 | 149 | −0.197 | −1.34 |
| Ensemble − Ridge | 2023 | 179 | −0.065 | −0.43 | 95 | −0.276 | −1.20 |
| Ensemble − Ridge | 2024 | 178 | +0.150 | 1.01 | 134 | +0.107 | 0.63 |
| Ensemble − Ridge | 2025 | 207 | −0.283 | −1.36 | 134 | −0.184 | −1.13 |
| Ensemble − Ridge | 2026 | 140 | +0.281 | 1.00 | 107 | +0.259 | 0.84 |
| Ensemble − Ridge | 2022–2026 | 912 | −0.067 | −0.78 | 619 | −0.062 | −0.67 |
| S1 − Ridge | 2022 | 197 | −0.712 | −3.10 | 144 | −0.614 | −2.26 |
| S1 − Ridge | 2023 | 156 | −0.409 | −1.63 | 90 | −0.855 | −2.81 |
| S1 − Ridge | 2024 | 171 | +0.220 | 0.63 | 123 | +0.376 | 1.17 |
| S1 − Ridge | 2025 | 195 | −0.146 | −0.56 | 124 | −0.424 | −1.23 |
| S1 − Ridge | 2026 | 133 | +0.175 | 0.25 | 106 | +0.045 | 0.07 |
| S1 − Ridge | 2022–2026 | 852 | −0.201 | −1.22 | 587 | −0.284 | −1.52 |

Reading: the ensemble's unconditional top-10 gross return (0.459%) is higher than Ridge's (0.384%) because it produces no candidates on worse days (no-candidate days 192 vs 65): this is the "fewer trades" effect; on days when both have candidates, the ensemble's top 10 is 0.067 percentage points below Ridge (t −0.78), and in 2025, 0.28 percentage points below, the opposite direction to the account result for 2025 of +78.9% vs +51.1%. By the engine's trade records, in 2025 the ensemble's 263 trades had a mean net return of +2.38% and a win rate of 60.1%, and Ridge's 237 trades +1.88% and 56.1%; in 2025, in months 3, 5, 8 and 9, the ensemble earned 5.9, 5.2, 5.8 and 5.0 percentage points more respectively. In other words, the account advantage comes from the actual holding paths (exit timing driven by faster rank changes) and the days without new positions, not from selection quality at entry. These two mechanisms are not in the criteria, and the signal-level metrics show no corresponding improvement either.
