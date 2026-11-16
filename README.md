# A-share Joint-Event Buy Signal Research

## At a glance

- Short-horizon stock-selection research on China A-share main-board stocks, ending in a blind test: split the data first, design on the training segment, freeze the scheme, then evaluate the holdout once.
- Over the 551-day holdout (2024-07 to 2026-10) the frozen scheme returned 48.27% annualised against 21.11% for benchmark B, an excess of +27.17 points with t = 2.33.
- Verified by a separate evaluation step that confirmed identical decisions on truncated and full data before the single holdout run, with predictions written down in advance (section 6).

![Blind holdout: annualised return vs. benchmark B, final NAV vs. random selection, and excess return across perturbed reruns](figures/summary.png)

**中文简介**：本仓库是一个中国内地 A 股（沪深主板）短线选股研究的公开精简版。核心问题是：能否用动量、波动率和流动性特征，预测一只股票在买入后约两周内"先涨到 +5% 且始终没有跌到 -5%"这一联合事件，并据此排序选股。仓库包含特征代码、联合事件建模与去重叠分析脚本、几组模型比较实验，以及原样保留数字的结果报告；另含 2026 年 11 月的一次盲测（训练/选择/留出三段切分，方案冻结后留出段只评估一次，见第 6 节）。行情数据不随仓库分发，完整复现需要自行准备本地数据缓存。

---

## 1. Research question

1. Can a causal 12-feature description of each stock (momentum, volatility, liquidity) rank A-share main-board stocks by the probability of a **joint event** within a short holding window: the daily high reaches +5% of the entry open while the daily low never touches -5%?
2. After removing overlapping signal windows of the same stock, how much upside, how much drawdown, and how quickly does the typical selected event move?
3. Over a 15-trading-day horizon, how do the joint-event selections compare with an earnings-announcement-return (EAR) event rule on the same days?
4. Separately, on a cross-sectional 5-day return panel: do Ridge, gradient-boosted trees (HGB / LightGBM with three label forms), a GKX-style NN3 network, or LambdaMART rank stocks better?

## 2. Data sources (not distributed)

- Daily A-share bars (open/high/low/close, volume, amount, adjustment factors) and the CSI 300 index were collected from **Baostock** and **AkShare** (see `requirements.txt`).
- The EAR comparison additionally used public listings of periodic-report announcements.
- **No market data, caches, model weights, score arrays or intermediate outputs are included.** To obtain the data yourself:
  - `pip install baostock akshare`
  - Baostock: `bs.login()` then `bs.query_history_k_data_plus(code, "date,open,high,low,close,volume,amount,adjustflag", start_date, end_date, frequency="d", adjustflag="2")` for forward-adjusted bars, and the same call with `adjustflag="3"` for raw prices.
  - AkShare: e.g. `ak.stock_zh_a_hist(symbol, period="daily", adjust="qfq")` and `ak.stock_zh_index_daily(symbol="sh000300")`.
  - The scripts expect these bars to be pre-assembled into the project's local cache layout (`output/research_once/...`, `research_data/...`), which is **not** part of this repository.

## 3. Methods

### 3.1 Features (`src/ashare_strategy/features/math12.py`)
Twelve causal features, each using information up to the close of day *t*:
momentum (`log_return_5d`, `log_return_21d`, `log_return_5_to_63d`, `log_return_21_to_252d`, `maximum_return_21d`),
volatility (`return_volatility_20d`, `return_volatility_252d`, `market_residual_volatility_60d` vs. CSI 300, `intraday_range_mean_20d`),
liquidity (`log_amihud_20d`, `log_amount_median_20d`, `log_volume_median20_to_median120`).
Cross-sectionally they are winsorised at 1%/99% and rank-mapped to [-1, 1].

### 3.2 Joint-event label (`scripts/research/build_short_w_joint_inputs_20260907.py`, `joint_labels`)
For each stock/origin, starting from the scheduled entry day's adjusted open, over a window W of at most 10 trading days (and at most 14 calendar days) each day is assigned a cumulative state:
`0` neither, `1` high ≥ 1.05 × open (up only), `2` low ≤ 0.95 × open (down only), `3` both.
The "clean" joint event is ending the window in state `1`: +5% reached and -5% never touched. Same-bar up/down order is treated as unknown.

### 3.3 State-transition likelihood model (`scripts/research/short_w_joint_model_20260907.py`)
- `JointObjective` (110 parameters = 5 transitions × (10 day offsets + 12 feature coefficients)): from state 0 a multinomial logit over {0,1,2,3}; from states 1 and 2 a logistic hazard into state 3; state 3 is absorbing. Fitted by penalised weighted likelihood over observed daily paths.
- `predict` marginalises the path to give state probabilities at every day of W; `ReductionObjective` (13 parameters) is a plain logistic control on the final clean event.
- `select` takes the top 4 names per day.
- Time-based roles: the buy model was frozen on data up to 2021-06 (results/joint_event/fixed_category_window_report.md, "学习与后续评价角色"); 2023–2026 results are a descriptive follow-up on history already seen during development, not a held-out test.

### 3.4 Overlap removal (`scripts/research/build_short_w_joint_dedup_20260907.py`, `membership`)
Per stock, the earliest signal is kept; a later signal is kept only if its scheduled entry is after the end of the last kept event's window W. Dropped events do not extend windows; unfilled or unfinished events still take part in date de-duplication.

### 3.5 EAR comparison (`scripts/research/compare_short_joint_h15_ear_matched_opportunity_20260908.py`)
EAR is an **event rule, not a fitted model** (definition in `scripts/research/short_joint_h15_ear_finite_history_compiler_20260908.py`, `score_event` / `threshold_for` / `daily_buy_candidates`): the stock's compound return from the close two sessions before the first post-announcement reaction session to the close one session after it, minus the CSI 300 return over the same four closes. A stock qualifies if EAR > 0 and above the previous quarter's 80th percentile; at most 4 names per day by EAR.
The comparison pairs EAR selections with the joint-model top-4 (`JOINT_COMMON`) on the same signal days, each day weighted once, over H = 5/10/15 trading days, measuring post-entry maximum favourable excursion (MFE, upside) and maximum adverse excursion (MAE, drawdown).

### 3.6 Model comparisons (2026-10, cross-sectional panel)
All on a 2014-01-02 to 2026-08-21 main-board development panel with a 108-column feature set (feature set C, not the 12 features above), 5-day trade-return label, models trained once on labels matured before 2022-01-01 and evaluated 2022-01-04 to 2026-08-21:
- Ridge vs. HGB and their z-score ensemble: `scripts/research/ml_ensemble_fixed_20261009.py`
- LightGBM with return-MSE, cross-sectional-rank and up/down binary labels: `scripts/research/label_form_20261009.py`
- GKX NN3 (108→32→16→8→1) vs. Ridge: `scripts/research/ashare_fixed_nn3_20261009.py`
- LambdaMART (LightGBM `LGBMRanker`): `experiments/experiment_stockranker_lambdamart.py` (separate, short window; see below)

Shared engine/rule modules: `ashare_engine_20261006.py`, `ashare_daily_judgment_20261008.py`, `ashare_ml_signals_20261006.py`, `ashare_rules_lib_20261006.py`, `ashare_cost_aware_exit_20261009.py`.

## 4. Main results (numbers quoted verbatim from files in this repository)

### 4.1 Overlap removal — `results/joint_event/dedup_comparison.md`
- Text, paragraph "原始 3524 个事件 → 保留 1242": 3524 raw events → 1242 kept, 2282 overlapping events removed; 596 stocks before and after.
- First table, row `FULL_FUTURE (2023-01-01—2026-08-21)`: complete/all 3403/3524 → 1198/1242; mean max gain 7.77% → 7.65%; median max gain 4.89% → 5.07%; share reaching +5% 48.87% → 50.58% (606/1198).
- First paragraph: max gain > 0 in 1167/1198 (97.41%); ≥ 3% 808/1198 (67.45%); ≥ 10% 298/1198 (24.87%); excluding the entry day, 1040/1198 (86.81%) later traded above the entry reference.
- Second table, row `FULL_FUTURE`: median first-peak day 5.00 → 4.00; pre-peak low mean -2.82% → -2.91%, median -2.23% → -2.31%; 142/896 determinable events fell at least 5% before the peak; full-window low median -4.72% → -4.88%.
- `results/joint_event/fixed_category_window_report.md`, table B, row `2023-01-01—2026-08-21`: full-window max drawdown mean/median 6.24%/4.88%.

### 4.2 Joint model vs. EAR, H = 15 — `results/joint_event/ear_matched_opportunity_h15_report.md`
Table "同日、等 origin 权重的主要比较" (percentage points; Δ = left − right; MAE more negative = deeper drawdown):

| Role | View | Pair | H | N | Δday1 MFE | Δday2+ MFE | ΔMAE |
|---|---|---|---:|---:|---:|---:|---:|
| SEEN_DEV_2023 | ALL_SIGNALS | EAR_COMMON − JOINT_COMMON | 15 | 91/91/91 | 2.077 | 1.838 | -1.385 |
| TRAIN_COHORT_2021SEP_2022 | ALL_SIGNALS | EAR_COMMON − JOINT_COMMON | 15 | 139/139/139 | 1.998 | 1.737 | -1.107 |

On these 91 paired 2023 signal days, the EAR selections had larger upside and deeper drawdown; equivalently the joint-model selections had shallower drawdown and lower upside. The report's verdict is **TRADEOFF_UNRESOLVED / NO_VIABLE_CLAIM**. Table "固定季度及12/24个月比较", row `2023Q4 | EAR_COMMON − JOINT_COMMON`: N = 14, Δday2+ MFE -0.284, ΔMAE -4.159 (the sign pattern differs in that quarter).

### 4.3 Model comparisons
- **Ridge vs. HGB vs. ensemble** — `docs/model_comparison/ml_ensemble_fixed_20261009.md`, "结论" item 1: criterion (a) CAGR/MDD ensemble 0.579, Ridge 0.497, HGB −0.137 (holds); criterion (b) ensemble rank IC above both components in only one year (2023) → **not passed**. Table 1, rows `2022–2026`: rank IC Ridge 0.0823, HGB 0.0677, ensemble 0.0820.
- **LightGBM label forms** — `results/model_comparison/label_form_criterion.json`: `lgb_rank` ratio_wins 4, ic_wins 5, passes true; `lgb_bin` ratio_wins 0, ic_wins 1, passes false (each vs. the return-MSE label, 5 periods). `docs/model_comparison/label_form_20261009.md`, table "B 规则主账户": rank L2 annualised 21.59%, max drawdown 27.03%, last 12 months -10.42%; return MSE 9.03%, 50.82%; fixed Ridge 21.31%, 38.13%.
- **NN3 vs. Ridge** — `results/model_comparison/nn3_criterion.json`: `A_ratio_nn3` -0.16270761260970473 vs. `A_ratio_ridge` 0.4892558985303913, `A_pass` false; `B_ic_years_won` 0 of `B_years` 5, `B_pass` false; `PASS` false.
- **LambdaMART** — `results/model_comparison/lambdamart_20260601_20260703_summary.md`: train 2025-04-01 to 2026-05-29, test 2026-06-01 to 2026-07-03; daily RankIC mean 0.025755320262357773, ICIR 0.13651478683915802; top-2 portfolio total return 0.22923766059993977, max drawdown -0.08517886995895119.

## 5. Limitations (as stated in the result files)

- The 2023–2026 joint-event statistics are a descriptive follow-up on history already seen during development, not a new held-out test (`fixed_category_window_report.md`, "学习与后续评价角色"). The model comparisons are likewise development results on seen data (`ml_ensemble_fixed_20261009.md` 注意事项 1; `label_form_20261009.md`).
- Max gains are ex-post opportunity space; the entry-day high may not be sellable under T+1. No account backtest, no market benchmark matching, no new exit rule was evaluated in the de-duplicated comparison (`dedup_comparison.md`, "解释边界"; `fixed_category_window_report.md` sections C/D).
- De-duplication only removes same-stock overlapping windows; same-day co-movement across different stocks remains, so retained counts are not independent sample sizes (`dedup_comparison.md`).
- The 15-day joint vs. EAR comparison is unresolved: a trade-off, not a profitable strategy; 2023Q4 is weak with only 14 pairable origins; day2+ MFE is an upper bound, not a realisable sale price (`ear_matched_opportunity_h15_report.md`).
- NN3 failed both pre-registered criteria; the verdict is taken from `nn3_criterion.json`.
- The ensemble did not pass; its account-level advantage is not supported at signal level (`ml_ensemble_fixed_20261009.md`). The rank-label "pass" refers only to the registered label-form comparison and is not a validated trading candidate; recent 12 months were negative (`label_form_20261009.md`).
- LambdaMART has only a short test window (2026-06-01 to 2026-07-03).
- 2026 data end on 2026-08-21; August 2026 in particular is immature (`dedup_comparison.md`).

## 6. Blind holdout test (2026-11) — `blind_holdout_2026-11/`

A separate, later study on the main-board daily panel, run with a pre-registered protocol and a single holdout evaluation.

### 6.1 Procedure
- **Split first, by a separate step.** Before any design work, a data-split step cut the panel into training (2014-01-02 to 2022-12-30), selection (2023-01-03 to 2024-06-28) and holdout (2024-07-01 to 2026-10-09, 551 trading days), and handed the designer only a truncated copy ending 2024-06-28 (`PROTOCOL.txt` section 四; `MANIFEST.txt`; export and self-check code in `gatekeeper/`).
- **Design on the training segment only.** Derivations were written before measurements; measurements and model fitting used the training segment; the selection segment was used once to make the go/no-go decision (`design/`, stages 阶段一 to 丙二).
- **Freeze, then evaluate the holdout once.** The frozen scheme (version 2): a `HistGradientBoostingRegressor` on 30 daily price/volume features, fitted on 2015–2019 signal days; hold 50 stocks equally weighted, at most 5 swaps per day; plus two mechanical exclusions X1/X2 (`design/stageC1/丙一_冻结与预测_运行前写定.txt`; `design/stageC2/丙二补_第二版与口径_运行前写定.txt`). Predictions for the holdout were written down beforehand (`PREDICTIONS_holdout.txt`). The evaluation side checked prefix invariance (identical daily decisions up to 2024-06-28 on truncated and full data), then ran the holdout once with no reruns and no code changes (`evaluator/留出段评测报告.txt`, 第五部分).
- Account rules: one 500,000 CNY long-only account; decision at close *t*, execution at the open of *t*+1; board lots; commission, transfer fee and stamp duty at actual rates, 0.1% slippage per side. Benchmark B: equal-weight buy-and-hold of the frozen stock pool on the first holdout day (`PROTOCOL_B.txt` sections 一–二).

### 6.2 Holdout result — `evaluator/留出段评测报告.txt`, 第二部分 (conservative valuation, main cost setting)

Over the 551-day holdout the frozen scheme returned 48.27% annualised against 21.11% for benchmark B, an excess of +27.17 points with *t* = 2.33; it met four of the five pre-registered targets. All targets:

| Pre-registered target | Requirement | Holdout | Result |
|---|---|---|---|
| 1 Magnitude | annualised net return ≥ benchmark B + 5 pts | 48.27% vs. 21.11%, +27.17 pts | pass |
| 2 Significance | ordinary *t* of daily return difference ≥ 2 | *t* = 2.33 (551 days) | pass |
| 3 Drawdown | max drawdown ≤ benchmark B | 29.45% vs. 23.09% | not met |
| 4 Selection vs. random | final NAV > 97.5th pct of 200 random-selection runs | 2.3661 vs. 1.5785 | pass |
| 5 Stability | 20 perturbed runs: min > 0, sd ≤ mean/3 | min 25.08, mean 26.37, sd 1.03 | pass |

All six pre-registered predictions with numeric ranges fell inside their ranges (第一部分).

### 6.3 Sources of return — `design/丙三_微盘暴露与档内选股.txt`, 总览 items 2 and 4
- Measured against an equal-weight portfolio of the whole pool, the holdout gross excess was 10.93 bp/day. Holding the smallest, least-traded stocks explains **39%** of it when bucketing by turnover amount alone, and **67%–70%** when bucketing by turnover amount and market cap together. The remaining within-bucket selection is 3.2–6.7 bp/day (*t* 1.85–2.8).
- A size-only reference rule ("hold the 50 smallest by a market-cap proxy, at most 5 swaps per day, same exclusions"), run in the same engine with the same costs, is reported alongside: 55.30% annualised with a 20.76% max drawdown in the holdout. It was defined after the holdout had been seen.

### 6.4 Scope and forward tracking
- The holdout was unseen by the design side of this study; earlier research in this project had used part of that period. Sensitivity accounts, quarterly results and the full list of disclosures are in `evaluator/留出段评测报告.txt` (第三、第四部分).
- Forward tracking, segment 1 (2026-10-12 to 2026-11-09, 21 trading days): −0.42 points vs. benchmark B and +1.74 points vs. the size-only reference rule, both inside the pre-registered ranges (`forward/前瞻记录_第1段.txt`; `PREDICTIONS_forward.txt`).
- Market data are not distributed (source licensing), and neither are the truncated/full panels, model file, score caches or daily records. The scripts keep placeholders such as `<BLIND_ROOT>` / `<DATA_ROOT>` in place of the original paths, so this part cannot be rerun from the repository.

## 7. Repository layout and file provenance

```
src/ashare_strategy/features/math12.py         12-feature transform
scripts/research/
  build_short_w_joint_inputs_20260907.py         joint-event labels and inputs
  short_w_joint_model_20260907.py                110-parameter transition likelihood + 13-parameter control
  build_short_w_joint_recent_buy_w_20260907.py   event summariser used by the dedup step
  build_short_w_joint_dedup_20260907.py          -> results/joint_event/dedup_comparison.md
  compare_short_joint_h15_ear_matched_opportunity_20260908.py
                                                 -> results/joint_event/ear_matched_opportunity_h15_report.md
  short_joint_h15_ear_finite_history_compiler_20260908.py   EAR definition
  ml_ensemble_fixed_20261009.py                  -> docs/model_comparison/ml_ensemble_fixed_20261009.md
  label_form_20261009.py                         -> results/model_comparison/label_form_criterion.json, docs/model_comparison/label_form_20261009.md
  ashare_fixed_nn3_20261009.py                   -> results/model_comparison/nn3_criterion.json
  ashare_engine_20261006.py, ashare_daily_judgment_20261008.py, ashare_ml_signals_20261006.py,
  ashare_rules_lib_20261006.py, ashare_cost_aware_exit_20261009.py   shared engine / rules
experiments/experiment_stockranker_lambdamart.py -> results/model_comparison/lambdamart_20260601_20260703_summary.md
results/joint_event/fixed_category_window_report.md   re-layout of the dedup summary (dedup_comparison.json, not included)
blind_holdout_2026-11/                         blind holdout test (section 6)
  PROTOCOL.txt, PROTOCOL_B.txt, MANIFEST.txt     protocols and data description, written before design
  PREDICTIONS_*.txt                              frozen predictions (selection, holdout, forward)
  gatekeeper/                                    data split: truncated export and self-check
  design/                                        derivations, training/selection measurements, engine, frozen scheme, attribution (丙三, 丙四)
  prep/                                          data-extension scripts and arithmetic notes
  evaluator/                                     single holdout run: scripts, logs, 留出段评测报告.txt
  forward/                                       forward-tracking scripts and segment-1 record
```

The intermediate JSON/JSONL/NPZ artifacts that the reports mention (e.g. `dedup_comparison.json`, `membership_ledger.jsonl`, `summary.json`, `accounts.csv`) are not included.

### Not included (internal modules imported by the scripts)
These modules are large and pull in much of the original data pipeline, so they were left out; the affected scripts therefore do not run from this repository alone:
- `build_short_h3_first_exit_inputs_20260907`, `build_short_cmlf_selected_w_paths_20260907`, `build_short_w5_buy_daily_inputs_20260907`, `ashare_strategy.features.m12_dated_monthly_input_assembly` (imported by `build_short_w_joint_inputs_20260907.py`)
- `build_short_w_joint_selected_exit_cal_20260907.py` (loaded dynamically by `build_short_w_joint_recent_buy_w_20260907.py`)
- `short_joint_h15_new_cohort_opportunity_source_20260908`, `compare_short_joint_h15_manual_reversal_opportunity_20260908` (imported by the EAR comparison script)
- `short_joint_h15_earnings_event_parser_20260908` (imported by the EAR compiler)
- `config`, `filters` (imported by the LambdaMART experiment)
- The model-fitting/selection run scripts of the joint model and the scripts that built the 108-column feature panel.

## 8. Reproduction

Full reproduction requires the local market-data cache and intermediate outputs described above, which are not in this repository. With such a cache in the original layout (`output/research_once/...`, `research_data/...`) and the missing modules, the entry points are, for example:

```
python scripts/research/build_short_w_joint_dedup_20260907.py ...
python scripts/research/ml_ensemble_fixed_20261009.py signals
python scripts/research/label_form_20261009.py signals
python scripts/research/ashare_fixed_nn3_20261009.py signal
```

Dependencies: `requirements.txt`, `requirements-research.lock`, `requirements-neural-research.lock`, `pyproject.toml`.

## License

MIT, see `LICENSE`.
