from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRanker
from scipy.stats import spearmanr

import config
from filters import is_st_stock


FEATURES = (
    "return_1d",
    "return_3d",
    "return_10d",
    "momentum_5d",
    "momentum_20d",
    "ma20_extension",
    "ma5_ma20",
    "ma10_ma30",
    "rsi6",
    "rsi14",
    "rsi24",
    "macd",
    "macd_signal",
    "macd_hist",
    "kdj_k",
    "kdj_d",
    "williams_r14",
    "cci20",
    "boll_pctb",
    "natr14",
    "realized_vol_5d",
    "realized_vol_20d",
    "intraday_amplitude",
    "boll_bandwidth",
    "max_drawdown_20d",
    "amount_ratio",
    "turnover",
    "turnover_median20",
    "volume_ratio",
    "obv_change_10d",
    "ad_change_10d",
    "mfi14",
    "plus_di14",
    "minus_di14",
    "adx14",
    "ma_bullish",
    "above_ma20",
    "breakout_20d",
    "breakdown_20d",
    "market_momentum_5d_median",
    "market_natr14_median",
    "market_above_ma20_ratio",
    "industry_momentum_5d_median",
    "industry_amount_ratio_median",
    "industry_above_ma20_ratio",
)


@dataclass(frozen=True)
class CostModel:
    buy_cost: float = 0.0003
    sell_cost: float = 0.0013
    min_cost: float = 5.0


def panel_path(cache_date: str) -> Path:
    return (
        config.PROJECT_ROOT
        / "research_data"
        / cache_date
        / "factor_rank_panel_20250101_full_v2.pkl"
    )


def load_panel(cache_date: str) -> pd.DataFrame:
    path = panel_path(cache_date)
    if not path.exists():
        raise FileNotFoundError(path)
    panel = pd.read_pickle(path)
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    panel["stock_code"] = panel["stock_code"].astype(str).str.zfill(6)
    return panel.sort_values(["stock_code", "date"]).reset_index(drop=True)


def add_forward_open_close_label(panel: pd.DataFrame, *, sell_offset_days: int) -> pd.DataFrame:
    """Attach leakage-safe BigQuant-style next-open entry / later-close exit label.

    A signal on day T is assumed to be known after T close. The first executable
    buy is T+1 open. Because A shares are T+1, the default sell offset is T+2
    close, matching a one-sellable-day holding period.
    """
    out = panel.sort_values(["stock_code", "date"]).copy()
    grouped = out.groupby("stock_code", sort=False)
    out["entry_date"] = grouped["date"].shift(-1)
    out["exit_date"] = grouped["date"].shift(-sell_offset_days)
    out["entry_open"] = grouped["open"].shift(-1)
    out["exit_close"] = grouped["close"].shift(-sell_offset_days)
    out["ranker_forward_return"] = out["exit_close"] / out["entry_open"] - 1.0
    return out


def tradable_universe(rows: pd.DataFrame, *, min_price: float, max_price: float) -> pd.DataFrame:
    valid_name = ~rows["stock_name"].astype(str).map(is_st_stock)
    valid_price = rows["close"].astype(float).between(min_price, max_price)
    valid_liquidity = (rows["amount_ma20"].astype(float) >= 1.5e8) & (
        rows["turnover_median20"].astype(float) >= 1.5
    )
    valid_bar = rows.get("signal_bar_valid", pd.Series(True, index=rows.index)).fillna(True)
    return rows[valid_name & valid_price & valid_liquidity & valid_bar].copy()


def cross_sectional_mad_zscore(rows: pd.DataFrame, features: tuple[str, ...]) -> pd.DataFrame:
    out = rows.copy()
    for col in features:
        values = pd.to_numeric(out[col], errors="coerce").astype(float)
        median = values.groupby(out["date"]).transform("median")
        mad = (values - median).abs().groupby(out["date"]).transform("median")
        std = values.groupby(out["date"]).transform("std").fillna(0.0)
        scale = np.maximum(mad * 1.4826, std * 0.5)
        scale = np.maximum(scale, 1e-9)
        clipped = values.clip(lower=median - 5.0 * scale, upper=median + 5.0 * scale)
        z = (clipped - median) / scale
        out[col] = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def add_rank_labels(rows: pd.DataFrame, *, bins: int) -> pd.DataFrame:
    out = rows.copy()
    rank_pct = out.groupby("date")["ranker_forward_return"].rank(pct=True, method="first")
    label = np.floor(rank_pct * bins).astype("Int64") - 1
    out["ranker_label"] = label.clip(lower=0, upper=bins - 1).astype(int)
    return out


def prepare_dataset(
    panel: pd.DataFrame,
    *,
    start: str,
    end: str,
    sell_offset_days: int,
    label_bins: int,
    min_price: float,
    max_price: float,
    require_label: bool = True,
) -> pd.DataFrame:
    rows = add_forward_open_close_label(panel, sell_offset_days=sell_offset_days)
    rows = rows[(rows["date"] >= pd.Timestamp(start)) & (rows["date"] <= pd.Timestamp(end))].copy()
    rows = tradable_universe(rows, min_price=min_price, max_price=max_price)
    feature_cols = [col for col in FEATURES if col in rows.columns]
    required = [*feature_cols]
    if require_label:
        required = ["ranker_forward_return", "entry_open", "exit_close", *required]
    rows = rows.dropna(subset=required).copy()
    rows = cross_sectional_mad_zscore(rows, tuple(feature_cols))
    if require_label:
        rows = add_rank_labels(rows, bins=label_bins)
    rows.attrs["features"] = feature_cols
    return rows


def group_sizes(rows: pd.DataFrame) -> list[int]:
    return rows.groupby("date", sort=True).size().astype(int).tolist()


def daily_rank_ic(rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    for date, day in rows.groupby("date", sort=True):
        if len(day) < 20 or day["prediction"].nunique() < 2:
            continue
        ic = spearmanr(day["prediction"], day["ranker_forward_return"], nan_policy="omit").correlation
        records.append({"date": date.date().isoformat(), "rank_ic": float(ic)})
    return pd.DataFrame(records)


def fee(value: float, rate: float, min_cost: float) -> float:
    return max(value * rate, min_cost)


def backtest_top_rank(
    rows: pd.DataFrame,
    *,
    account: float,
    top_n: int,
    cost: CostModel,
) -> tuple[pd.DataFrame, dict]:
    cash = float(account)
    equity_curve = []
    trades = []
    next_signal_allowed = pd.Timestamp.min
    weights = np.array([1.0 / math.log(i + 2.0) for i in range(top_n)], dtype=float)
    weights = weights / weights.sum()

    for signal_date, day in rows.groupby("date", sort=True):
        signal_date = pd.Timestamp(signal_date)
        if signal_date < next_signal_allowed:
            continue
        picks = day.sort_values(["prediction", "amount"], ascending=[False, False]).head(top_n)
        if picks.empty:
            continue

        starting_cash = cash
        exit_date = pd.Timestamp(picks["exit_date"].max())
        day_pnl = 0.0
        used = 0.0
        for i, pick in enumerate(picks.to_dict("records")):
            entry_price = float(pick["entry_open"])
            exit_price = float(pick["exit_close"])
            budget = starting_cash * float(weights[i])
            shares = math.floor((budget / (entry_price * (1.0 + cost.buy_cost))) / 100.0) * 100
            if shares <= 0:
                continue
            buy_value = shares * entry_price
            buy_fee = fee(buy_value, cost.buy_cost, cost.min_cost)
            if used + buy_value + buy_fee > cash:
                continue
            sell_value = shares * exit_price
            sell_fee = fee(sell_value, cost.sell_cost, cost.min_cost)
            pnl = sell_value - sell_fee - buy_value - buy_fee
            day_pnl += pnl
            used += buy_value + buy_fee
            trades.append(
                {
                    "signal_date": signal_date.date().isoformat(),
                    "entry_date": pd.Timestamp(pick["entry_date"]).date().isoformat(),
                    "exit_date": pd.Timestamp(pick["exit_date"]).date().isoformat(),
                    "stock_code": str(pick["stock_code"]).zfill(6),
                    "stock_name": pick["stock_name"],
                    "prediction": float(pick["prediction"]),
                    "entry_open": entry_price,
                    "exit_close": exit_price,
                    "shares": shares,
                    "gross_return_pct": exit_price / entry_price - 1.0,
                    "net_pnl": pnl,
                    "net_return_on_budget_pct": pnl / budget if budget else np.nan,
                }
            )
        if used <= 0:
            continue
        cash += day_pnl
        next_signal_allowed = exit_date
        equity_curve.append(
            {
                "date": exit_date.date().isoformat(),
                "equity": cash,
                "return_pct": cash / account - 1.0,
            }
        )

    trades_df = pd.DataFrame(trades)
    curve = pd.DataFrame(equity_curve)
    if curve.empty:
        summary = {
            "trades": 0,
            "total_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "win_rate": None,
        }
        return trades_df, summary
    running_max = curve["equity"].cummax()
    drawdown = curve["equity"] / running_max - 1.0
    summary = {
        "trades": int(len(trades_df)),
        "round_trips": int(len(curve)),
        "ending_equity": float(curve["equity"].iloc[-1]),
        "total_return_pct": float(curve["equity"].iloc[-1] / account - 1.0),
        "max_drawdown_pct": float(drawdown.min()),
        "win_rate": float((trades_df["net_pnl"] > 0).mean()) if not trades_df.empty else None,
    }
    return trades_df, summary


def attach_latest_position_plan(
    rows: pd.DataFrame,
    *,
    account: float,
    top_n: int,
    cost: CostModel,
) -> pd.DataFrame:
    """Attach lot feasibility for the current top-N sleeve.

    The latest row is a signal-time rank, so close is only a planning proxy.
    Next-open gaps can still change the actual share count.
    """
    if rows.empty:
        return rows.copy()
    out = rows.copy().reset_index(drop=True)
    out["strategy_rank"] = np.arange(1, len(out) + 1)
    weights = np.array([1.0 / math.log(i + 2.0) for i in range(top_n)], dtype=float)
    weights = weights / weights.sum()
    budgets = []
    planned_shares = []
    values = []
    feasible = []
    for _, row in out.iterrows():
        rank = int(row["strategy_rank"])
        if rank > top_n:
            budgets.append(np.nan)
            planned_shares.append(0)
            values.append(0.0)
            feasible.append(False)
            continue
        budget = account * float(weights[rank - 1])
        close = float(row["close"])
        shares = math.floor((budget / (close * (1.0 + cost.buy_cost))) / 100.0) * 100
        budgets.append(float(budget))
        planned_shares.append(int(shares))
        values.append(float(shares * close))
        feasible.append(shares >= 100)
    out["strategy_top_n"] = int(top_n)
    out["target_budget"] = budgets
    out["planned_shares_at_close"] = planned_shares
    out["planned_position_value_at_close"] = values
    out["lot_feasible_at_close"] = feasible
    out["planning_price_note"] = "signal close proxy; next-open gap can change feasibility"
    return out


def run(args: argparse.Namespace) -> dict:
    panel = load_panel(args.cache_date)
    end = args.test_end or str(pd.to_datetime(panel["date"]).max().date())
    rows = prepare_dataset(
        panel,
        start=args.train_start,
        end=end,
        sell_offset_days=args.sell_offset_days,
        label_bins=args.label_bins,
        min_price=args.min_price,
        max_price=args.max_price,
    )
    features = tuple(rows.attrs["features"])
    train = rows[(rows["date"] >= pd.Timestamp(args.train_start)) & (rows["date"] <= pd.Timestamp(args.train_end))].copy()
    test = rows[(rows["date"] >= pd.Timestamp(args.test_start)) & (rows["date"] <= pd.Timestamp(end))].copy()
    if train.empty or test.empty:
        raise RuntimeError(f"empty train/test: train={len(train)} test={len(test)}")

    train = train.sort_values(["date", "stock_code"]).copy()
    test = test.sort_values(["date", "stock_code"]).copy()
    model = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        random_state=args.random_state,
        n_jobs=args.num_threads,
        verbose=-1,
    )
    model.fit(
        train[list(features)].astype(float),
        train["ranker_label"].astype(int),
        group=group_sizes(train),
    )
    test["prediction"] = model.predict(test[list(features)].astype(float))
    ic = daily_rank_ic(test)
    trades, portfolio = backtest_top_rank(
        test,
        account=args.account,
        top_n=args.top_n,
        cost=CostModel(),
    )

    latest_signal_date = pd.Timestamp(panel["date"].max()).normalize()
    latest = prepare_dataset(
        panel,
        start=str(latest_signal_date.date()),
        end=str(latest_signal_date.date()),
        sell_offset_days=args.sell_offset_days,
        label_bins=args.label_bins,
        min_price=args.min_price,
        max_price=args.max_price,
        require_label=False,
    )
    if not latest.empty:
        latest = latest.sort_values(["date", "stock_code"]).copy()
        latest["prediction"] = model.predict(latest[list(features)].astype(float))
        latest_top = latest.sort_values(["prediction", "amount"], ascending=[False, False]).head(args.latest_top)
        latest_top = attach_latest_position_plan(
            latest_top,
            account=args.account,
            top_n=args.top_n,
            cost=CostModel(),
        )
    else:
        latest_top = pd.DataFrame()

    out_dir = config.PROJECT_ROOT / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.output_tag}" if args.output_tag else ""
    stem = (
        f"stockranker_lambdamart_top{args.top_n}_{args.cache_date}_{args.test_start}_{end}{suffix}"
    ).replace("-", "")
    pred_path = out_dir / f"{stem}_test_predictions.csv"
    trades_path = out_dir / f"{stem}_trades.csv"
    latest_path = out_dir / f"{stem}_latest_top.csv"
    importance_path = out_dir / f"{stem}_feature_importance.csv"
    summary_path = out_dir / f"{stem}_summary.json"
    md_path = out_dir / f"{stem}_summary.md"

    test[
        [
            "date",
            "stock_code",
            "stock_name",
            "prediction",
            "ranker_forward_return",
            "ranker_label",
            "entry_date",
            "exit_date",
            "entry_open",
            "exit_close",
        ]
    ].to_csv(pred_path, index=False, encoding="utf-8-sig")
    trades.to_csv(trades_path, index=False, encoding="utf-8-sig")
    if not latest_top.empty:
        latest_top[
            [
                "date",
                "stock_code",
                "stock_name",
                "strategy_rank",
                "strategy_top_n",
                "prediction",
                "close",
                "target_budget",
                "planned_shares_at_close",
                "planned_position_value_at_close",
                "lot_feasible_at_close",
                "planning_price_note",
                "entry_date",
                "exit_date",
                "entry_open",
                "exit_close",
            ]
        ].to_csv(latest_path, index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame().to_csv(latest_path, index=False, encoding="utf-8-sig")

    importance = pd.DataFrame(
        {
            "feature": features,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importance.to_csv(importance_path, index=False, encoding="utf-8-sig")

    summary = {
        "cache_date": args.cache_date,
        "claim_scope": "research_only_bigquant_stockranker_local_analogue",
        "account": args.account,
        "train": [args.train_start, args.train_end],
        "test": [args.test_start, end],
        "label": {
            "entry": "signal_date+1 open",
            "exit": f"signal_date+{args.sell_offset_days} close",
            "bins": args.label_bins,
        },
        "tradable_filter": {
            "non_st": True,
            "price_range": [args.min_price, args.max_price],
            "min_amount_ma20": 150000000.0,
            "min_turnover_median20": 1.5,
            "signal_bar_valid": True,
            "lot_feasibility_note": (
                "latest_top includes planned_shares_at_close for this top_n sleeve; "
                "next-open gap can still change feasibility"
            ),
        },
        "rows": {
            "train": int(len(train)),
            "test": int(len(test)),
            "latest_prediction": int(len(latest_top)),
        },
        "rank_ic_mean": float(ic["rank_ic"].mean()) if not ic.empty else None,
        "rank_ic_ir": float(ic["rank_ic"].mean() / ic["rank_ic"].std(ddof=0)) if len(ic) > 1 and ic["rank_ic"].std(ddof=0) else None,
        "portfolio": portfolio,
        "cost_model": {
            "buy_cost": CostModel.buy_cost,
            "sell_cost": CostModel.sell_cost,
            "min_cost": CostModel.min_cost,
        },
        "top_features": importance.head(20).to_dict("records"),
        "files": {
            "predictions": str(pred_path),
            "trades": str(trades_path),
            "latest_top": str(latest_path),
            "feature_importance": str(importance_path),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    md_lines = [
        f"# StockRanker LambdaMART Local Analogue {args.cache_date}",
        "",
        f"- Train: `{args.train_start}` to `{args.train_end}`.",
        f"- Test: `{args.test_start}` to `{end}`.",
        f"- Label: signal T -> buy T+1 open -> sell T+{args.sell_offset_days} close, {args.label_bins} daily bins.",
        f"- Rows: train `{len(train)}`, test `{len(test)}`.",
        f"- Daily RankIC mean: `{summary['rank_ic_mean']}`; ICIR: `{summary['rank_ic_ir']}`.",
        f"- Portfolio top-{args.top_n}: total return `{portfolio.get('total_return_pct')}`, max drawdown `{portfolio.get('max_drawdown_pct')}`, win rate `{portfolio.get('win_rate')}`.",
        "",
        "## Latest Top",
    ]
    if latest_top.empty:
        md_lines.append("No latest prediction rows.")
    else:
        md_lines.append(
            latest_top[
                [
                    "stock_code",
                    "stock_name",
                    "strategy_rank",
                    "prediction",
                    "close",
                    "target_budget",
                    "planned_shares_at_close",
                    "lot_feasible_at_close",
                ]
            ].to_markdown(index=False)
        )
    md_lines.extend(["", "## Top Feature Importance", importance.head(20).to_markdown(index=False)])
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-date", required=True)
    parser.add_argument("--train-start", default="2025-04-01")
    parser.add_argument("--train-end", default="2026-05-29")
    parser.add_argument("--test-start", default="2026-06-01")
    parser.add_argument("--test-end", default=None)
    parser.add_argument("--sell-offset-days", type=int, default=2)
    parser.add_argument("--label-bins", type=int, default=20)
    parser.add_argument("--account", type=float, default=20000.0)
    parser.add_argument("--top-n", type=int, default=1)
    parser.add_argument("--latest-top", type=int, default=10)
    parser.add_argument("--min-price", type=float, default=3.0)
    parser.add_argument("--max-price", type=float, default=40.0)
    parser.add_argument("--output-tag", default="")
    parser.add_argument("--n-estimators", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--random-state", type=int, default=7)
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
