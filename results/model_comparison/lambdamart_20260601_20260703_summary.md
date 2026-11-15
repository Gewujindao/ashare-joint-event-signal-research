# StockRanker LambdaMART Local Analogue 20260703

- Train: `2025-04-01` to `2026-05-29`.
- Test: `2026-06-01` to `2026-07-03`.
- Label: signal T -> buy T+1 open -> sell T+2 close, 20 daily bins.
- Rows: train `174989`, test `22146`.
- Daily RankIC mean: `0.025755320262357773`; ICIR: `0.13651478683915802`.
- Portfolio top-2: total return `0.22923766059993977`, max drawdown `-0.08517886995895119`, win rate `0.5555555555555556`.

## Latest Top
|   stock_code | stock_name   |   prediction |   close | entry_date   | exit_date   |
|-------------:|:-------------|-------------:|--------:|:-------------|:------------|
|       600667 | 太极实业         |     0.834206 |   29.96 | NaT          | NaT         |
|       603823 | 百合花          |     0.832949 |   59.88 | NaT          | NaT         |
|       000566 | 海南海药         |     0.736253 |    6.14 | NaT          | NaT         |
|       600641 | 先导基电         |     0.722249 |   47.51 | NaT          | NaT         |
|       600857 | 宁波中百         |     0.537638 |   21.3  | NaT          | NaT         |
|       600206 | 有研新材         |     0.53305  |   54.3  | NaT          | NaT         |
|       002635 | 安洁科技         |     0.450502 |   23.36 | NaT          | NaT         |
|       002167 | 东方锆业         |     0.37     |   25.04 | NaT          | NaT         |
|       002354 | 天娱数科         |     0.35141  |    9.37 | NaT          | NaT         |
|       002141 | 贤丰控股         |     0.346881 |    5.14 | NaT          | NaT         |

## Top Feature Importance
| feature                      |   importance |
|:-----------------------------|-------------:|
| turnover_median20            |          114 |
| realized_vol_20d             |          102 |
| natr14                       |           97 |
| ma20_extension               |           94 |
| intraday_amplitude           |           92 |
| industry_momentum_5d_median  |           91 |
| industry_above_ma20_ratio    |           85 |
| max_drawdown_20d             |           83 |
| return_3d                    |           82 |
| realized_vol_5d              |           80 |
| industry_amount_ratio_median |           71 |
| obv_change_10d               |           71 |
| adx14                        |           71 |
| momentum_20d                 |           69 |
| rsi6                         |           69 |
| momentum_5d                  |           66 |
| macd_hist                    |           64 |
| mfi14                        |           63 |
| cci20                        |           61 |
| return_10d                   |           56 |