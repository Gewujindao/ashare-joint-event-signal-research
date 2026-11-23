"""Causal twelve-feature transform for the v2 research lane.

This module owns feature mathematics only.  It does not decide train/test
splits, fit a model, select securities, or manufacture transaction amount from
price and volume.  Every value dated ``t`` uses stock and CSI 300 observations
at or before the close of ``t``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


FEATURE_SCHEMA_VERSION = "math12_monthly_features_v1.1.0"
FEATURE_SET_ID = "math12_price_liquidity_rank_mainboard_nonst_v1"
EPSILON = 1e-12

RAW_FEATURE_COLUMNS = (
    "log_return_5d",
    "log_return_21d",
    "log_return_5_to_63d",
    "log_return_21_to_252d",
    "return_volatility_20d",
    "return_volatility_252d",
    "market_residual_volatility_60d",
    "maximum_return_21d",
    "intraday_range_mean_20d",
    "log_amihud_20d",
    "log_amount_median_20d",
    "log_volume_median20_to_median120",
)

WINSORIZED_FEATURE_COLUMNS = tuple(
    f"xs_winsor_{column}" for column in RAW_FEATURE_COLUMNS
)
RANKED_FEATURE_COLUMNS = tuple(f"rank_{column}" for column in RAW_FEATURE_COLUMNS)

FEATURE_FORMULAS = {
    "log_return_5d": "ln(C[t] / C[t-5])",
    "log_return_21d": "ln(C[t] / C[t-21])",
    "log_return_5_to_63d": "ln(C[t-5] / C[t-63])",
    "log_return_21_to_252d": "ln(C[t-21] / C[t-252])",
    "return_volatility_20d": "sample_sd(r[t-19:t]), r=C[t]/C[t-1]-1",
    "return_volatility_252d": "sample_sd(r[t-251:t]), r=C[t]/C[t-1]-1",
    "market_residual_volatility_60d": (
        "sqrt(OLS_SSE/58) from r_stock=a+b*r_CSI300+e over 60 paired days"
    ),
    "maximum_return_21d": "max(r[t-20:t])",
    "intraday_range_mean_20d": "mean((H-L)/C[t-1], 20 days)",
    "log_amihud_20d": "ln(mean(abs(r)/observed_amount, 20 days)+1e-12)",
    "log_amount_median_20d": "ln(median(observed_amount, 20 days)+1e-12)",
    "log_volume_median20_to_median120": (
        "ln(median(volume,20 days)/median(volume,120 days))"
    ),
}

_REQUIRED_STOCK_COLUMNS = {
    "date",
    "stock_code",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "is_st",
}
_REQUIRED_BENCHMARK_COLUMNS = {"date", "close"}


def math12_feature_identity() -> dict[str, Any]:
    """Return a JSON-serialisable feature contract."""
    payload: dict[str, Any] = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "feature_set_id": FEATURE_SET_ID,
        "signal_timing": "market_month_end_after_close",
        "feature_columns": list(RAW_FEATURE_COLUMNS),
        "winsorized_feature_columns": list(WINSORIZED_FEATURE_COLUMNS),
        "ranked_feature_columns": list(RANKED_FEATURE_COLUMNS),
        "formulas": dict(FEATURE_FORMULAS),
        "daily_return_definition": "simple_adjusted_close_to_close_return",
        "minimum_valid_return_history": 252,
        "cross_sectional_transform": {
            "universe": "cn_mainboard_nonst_pit_v1 at the signal date",
            "winsor_quantiles": [0.01, 0.99],
            "rank_method": "average",
            "rank_map": "2*(rank-1)/(N-1)-1",
            "single_observation_cross_section": "missing",
            "missing_policy": "preserve_missing_no_fill",
        },
        "amount_contract": (
            "provider-observed amount is mandatory; adjusted_price*volume synthesis is forbidden"
        ),
    }
    return payload


def _normalise_dates(values: pd.Series, *, name: str) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    if parsed.isna().any():
        bad = int(parsed.isna().sum())
        raise ValueError(f"{name} contains {bad} unparseable date value(s)")
    if parsed.dt.tz is not None:
        parsed = parsed.dt.tz_localize(None)
    return parsed.dt.normalize()


def _normalise_stock_rows(rows: pd.DataFrame, *, amount_is_observed: bool) -> pd.DataFrame:
    missing = sorted(_REQUIRED_STOCK_COLUMNS - set(rows.columns))
    if missing:
        raise KeyError(f"missing stock daily-bar columns: {missing}")
    if amount_is_observed is not True:
        raise ValueError(
            "amount provenance is not provider-observed; refusing to construct amount features"
        )

    if rows.empty:
        raise ValueError("stock daily rows are empty")
    out = rows.copy()
    out["date"] = _normalise_dates(out["date"], name="stock daily rows")
    codes = out["stock_code"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    if codes.eq("").any():
        raise ValueError("stock_code contains an empty value")
    out["stock_code"] = codes.str.zfill(6)
    if out.duplicated(["stock_code", "date"]).any():
        raise ValueError("duplicate stock_code/date rows are not allowed")

    for column in ("open", "high", "low", "close", "volume", "amount"):
        out[column] = pd.to_numeric(out[column], errors="coerce")

    observed_amount = out["amount"].where(np.isfinite(out["amount"]) & out["amount"].gt(0.0))
    without_amount = observed_amount.groupby(out["stock_code"], sort=False).count().eq(0)
    if without_amount.any():
        blocked = ", ".join(without_amount[without_amount].index.astype(str).tolist()[:5])
        raise ValueError(f"no positive provider-observed amount for stock(s): {blocked}")

    return out.sort_values(["stock_code", "date"]).reset_index(drop=True)


def _normalise_benchmark_rows(rows: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(_REQUIRED_BENCHMARK_COLUMNS - set(rows.columns))
    if missing:
        raise KeyError(f"missing CSI300 daily-bar columns: {missing}")
    out = rows.copy()
    out["date"] = _normalise_dates(out["date"], name="CSI300 daily rows")
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    if out.duplicated("date").any():
        raise ValueError("duplicate CSI300 dates are not allowed")
    return out.sort_values("date").reset_index(drop=True)


def _rolling_ols_residual_std(
    stock_return: pd.Series,
    market_return: pd.Series,
    *,
    window: int = 60,
) -> pd.Series:
    """Rolling OLS residual standard deviation with an intercept and n-2 DOF."""
    paired = stock_return.notna() & market_return.notna()
    y = stock_return.where(paired, 0.0).astype(float)
    x = market_return.where(paired, 0.0).astype(float)
    count = paired.astype(float).rolling(window, min_periods=window).sum()
    sum_x = x.rolling(window, min_periods=window).sum()
    sum_y = y.rolling(window, min_periods=window).sum()
    sum_xx = x.mul(x).rolling(window, min_periods=window).sum()
    sum_yy = y.mul(y).rolling(window, min_periods=window).sum()
    sum_xy = x.mul(y).rolling(window, min_periods=window).sum()

    sxx = sum_xx - sum_x.mul(sum_x).div(window)
    syy = sum_yy - sum_y.mul(sum_y).div(window)
    sxy = sum_xy - sum_x.mul(sum_y).div(window)
    sse = syy - sxy.mul(sxy).div(sxx.where(sxx.gt(EPSILON)))
    result = np.sqrt(sse.clip(lower=0.0).div(window - 2))
    return result.where(count.eq(window) & sxx.gt(EPSILON))


def compute_math12_daily_features(
    stock_rows: pd.DataFrame,
    benchmark_rows: pd.DataFrame,
    *,
    amount_is_observed: bool,
) -> pd.DataFrame:
    """Attach all twelve causal raw features to adjusted stock daily bars.

    ``amount_is_observed`` is deliberately mandatory.  Passing ``True`` is an
    explicit upstream provenance assertion; a missing or synthetic amount
    series must fail before this function is called.
    """
    stocks = _normalise_stock_rows(stock_rows, amount_is_observed=amount_is_observed)
    benchmark = _normalise_benchmark_rows(benchmark_rows)
    market_close = benchmark["close"].where(benchmark["close"].gt(0.0))
    benchmark["_market_return"] = market_close.pct_change(fill_method=None)
    market_return_by_date = benchmark.set_index("date")["_market_return"]

    parts: list[pd.DataFrame] = []
    for _, group in stocks.groupby("stock_code", sort=False):
        out = group.copy()
        close = out["close"].where(out["close"].gt(0.0))
        high = out["high"].where(out["high"].gt(0.0))
        low = out["low"].where(out["low"].gt(0.0))
        volume = out["volume"].where(out["volume"].gt(0.0))
        amount = out["amount"].where(out["amount"].gt(0.0))
        daily_return = close.pct_change(fill_method=None)
        market_return = out["date"].map(market_return_by_date).astype(float)

        out["log_return_5d"] = np.log(close.div(close.shift(5)))
        out["log_return_21d"] = np.log(close.div(close.shift(21)))
        out["log_return_5_to_63d"] = np.log(close.shift(5).div(close.shift(63)))
        out["log_return_21_to_252d"] = np.log(close.shift(21).div(close.shift(252)))
        out["return_volatility_20d"] = daily_return.rolling(20, min_periods=20).std(ddof=1)
        out["return_volatility_252d"] = daily_return.rolling(252, min_periods=252).std(ddof=1)
        out["market_residual_volatility_60d"] = _rolling_ols_residual_std(
            daily_return, market_return, window=60
        )
        out["maximum_return_21d"] = daily_return.rolling(21, min_periods=21).max()
        out["intraday_range_mean_20d"] = (
            high.sub(low).div(close.shift(1)).rolling(20, min_periods=20).mean()
        )
        out["log_amihud_20d"] = np.log(
            daily_return.abs().div(amount).rolling(20, min_periods=20).mean() + EPSILON
        )
        out["log_amount_median_20d"] = np.log(
            amount.rolling(20, min_periods=20).median() + EPSILON
        )
        median_volume_20 = volume.rolling(20, min_periods=20).median()
        median_volume_120 = volume.rolling(120, min_periods=120).median()
        out["log_volume_median20_to_median120"] = np.log(
            median_volume_20.div(median_volume_120)
        )
        out["valid_return_history"] = daily_return.notna().cumsum().astype("int32")
        out["has_252_valid_return_history"] = out["valid_return_history"].ge(252)
        parts.append(out)

    if not parts:
        return stocks.assign(
            **{column: pd.Series(dtype=float) for column in RAW_FEATURE_COLUMNS}
        )
    return pd.concat(parts, ignore_index=True, sort=False)


def add_monthly_cross_sectional_features(
    panel: pd.DataFrame,
    *,
    cross_section_member: pd.Series | None = None,
) -> pd.DataFrame:
    """Winsorise/rank within the frozen model universe in each signal month.

    Missing observations remain missing.  A one-name cross-section has no
    meaningful rank and therefore also remains missing.  Rows outside
    ``cross_section_member`` receive missing winsor/rank values, which prevents
    a later model layer from silently reintroducing them.
    """
    required = {"signal_date", *RAW_FEATURE_COLUMNS}
    missing = sorted(required - set(panel.columns))
    if missing:
        raise KeyError(f"missing monthly feature columns: {missing}")

    out = panel.copy()
    out["signal_date"] = _normalise_dates(out["signal_date"], name="monthly panel")
    if cross_section_member is None:
        member = pd.Series(True, index=out.index, dtype=bool)
    else:
        if not cross_section_member.index.equals(out.index):
            raise ValueError("cross_section_member index must match the panel index")
        member = cross_section_member.fillna(False).astype(bool)
    for raw_column, winsor_column, rank_column in zip(
        RAW_FEATURE_COLUMNS,
        WINSORIZED_FEATURE_COLUMNS,
        RANKED_FEATURE_COLUMNS,
        strict=True,
    ):
        values = pd.to_numeric(out[raw_column], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        in_universe_values = values.where(member)
        grouped = in_universe_values.groupby(out["signal_date"], sort=False)
        lower = grouped.transform(lambda item: item.quantile(0.01))
        upper = grouped.transform(lambda item: item.quantile(0.99))
        winsorized = in_universe_values.clip(lower=lower, upper=upper)
        out[winsor_column] = winsorized

        ranks = winsorized.groupby(out["signal_date"], sort=False).rank(method="average")
        counts = winsorized.notna().groupby(out["signal_date"], sort=False).transform("sum")
        mapped = 2.0 * ranks.sub(1.0).div(counts.sub(1.0)) - 1.0
        out[rank_column] = mapped.where(counts.gt(1))

    finite_raw = (
        out.loc[:, RAW_FEATURE_COLUMNS]
        .replace([np.inf, -np.inf], np.nan)
        .notna()
        .all(axis=1)
    )
    finite_ranks = (
        out.loc[:, RANKED_FEATURE_COLUMNS]
        .replace([np.inf, -np.inf], np.nan)
        .notna()
        .all(axis=1)
    )
    out["feature_eligible"] = (
        member
        & out["has_252_valid_return_history"].fillna(False)
        & finite_raw
        & finite_ranks
    )
    return out
