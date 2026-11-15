"""A-share daily short-term trading engine (plan: docs/research/ashare_quant_trading_plan_20261006.md).

Panel: wide float32 arrays [days x stocks] built once from per-stock CSVs and cached as .npz.
Trading rules shared by every strategy:
- signal at close t, execution at the raw open of t+1; buys blocked when the open is at the
  limit-up price, sells blocked at the limit-down price or when not trading (kept pending);
- T+1: an exit decided at close t can only execute from the next open, so a stock bought at
  the open of t can be sold at the open of t+1 at the earliest;
- costs: commission max(0.01% x amount, 5 CNY), stamp tax 0.05% on sells, transfer fee
  0.001% both sides, slippage 0.1% per side (applied to the price);
- position value follows the qfq (forward-adjusted) price ratio from the entry open, so cash
  dividends and bonus shares need no event ledger; lots and limit checks use raw prices.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / 'output/research_once/ashare_engine_cache'
COMMISSION, MIN_COMMISSION, STAMP, TRANSFER, SLIP = 0.0001, 5.0, 0.0005, 0.00001, 0.001
FIELDS_F = ['open_raw', 'high_raw', 'low_raw', 'close_raw', 'preclose_raw', 'open_qfq', 'high_qfq', 'low_qfq',
            'close_qfq', 'volume_shares', 'amount_cny', 'turnover_pct']
FIELDS_B = ['is_st', 'is_suspended']


def build_panel(csv_dirs, name, start='2014-01-01', end='2026-09-30'):
    """Read per-stock CSVs (one file per code) into wide arrays and cache them."""
    CACHE.mkdir(parents=True, exist_ok=True)
    out = CACHE / f'{name}.npz'
    if out.exists():
        return load_panel(name)
    files = {}
    for d in csv_dirs:  # first directory wins for a code listed twice
        for f in sorted(Path(d).glob('*.csv')):
            files.setdefault(f.stem, f)
    codes = sorted(files)
    dates = set()
    for c in codes:  # pass 1: date union only (keeps memory small)
        dd = pd.read_csv(files[c], usecols=['date'], dtype={'date': str}).date
        dates.update(dd[(dd >= start) & (dd <= end)])
    dates = sorted(dates)
    di = {d: i for i, d in enumerate(dates)}
    arrays = {k: np.full((len(dates), len(codes)), np.nan, np.float32) for k in FIELDS_F}
    tradable = np.zeros((len(dates), len(codes)), bool)
    st = np.zeros((len(dates), len(codes)), bool)
    for j, c in enumerate(codes):  # pass 2: one file at a time
        x = pd.read_csv(files[c], usecols=['date', 'trade_status'] + FIELDS_F + FIELDS_B, dtype={'date': str})
        x = x[(x.date >= start) & (x.date <= end)].set_index('date')
        if x.empty:
            continue
        rows = np.array([di[d] for d in x.index])
        for k in FIELDS_F:
            arrays[k][rows, j] = x[k].to_numpy(np.float32)
        ok = (x.trade_status.to_numpy() == 1) & (~x.is_suspended.astype(str).eq('True').to_numpy())
        tradable[rows, j] = ok
        st[rows, j] = x.is_st.astype(str).isin(['1', 'True']).to_numpy()
    np.savez_compressed(out, dates=np.array(dates), codes=np.array(codes), tradable=tradable, st=st, **arrays)
    return load_panel(name)


@dataclass
class Panel:
    dates: np.ndarray
    codes: np.ndarray
    a: dict
    tradable: np.ndarray
    st: np.ndarray
    limit: np.ndarray = field(default=None)


def limit_pct(codes, dates):
    """Daily price-limit fraction by board and date (ST handled separately)."""
    lim = np.full((len(dates), len(codes)), 0.10, np.float32)
    d = pd.to_datetime(pd.Series(dates))
    for j, c in enumerate(codes):
        if c.startswith('688'):
            lim[:, j] = 0.20
        elif c.startswith('300') or c.startswith('301'):
            lim[(d >= '2020-08-24').to_numpy(), j] = 0.20
        elif c[0] in '48':
            lim[:, j] = 0.30
    return lim


def load_panel(name):
    z = np.load(CACHE / f'{name}.npz', allow_pickle=False)
    a = {k: z[k] for k in FIELDS_F}
    p = Panel(z['dates'], z['codes'], a, z['tradable'], z['st'])
    lim = limit_pct(p.codes, p.dates)
    p.limit = np.where(p.st, 0.05, lim).astype(np.float32)
    return p


def fee_buy(amount):
    return max(COMMISSION * amount, MIN_COMMISSION) + TRANSFER * amount


def fee_sell(amount, stamp=STAMP):
    return max(COMMISSION * amount, MIN_COMMISSION) + TRANSFER * amount + stamp * amount


def stamp_rate(date, historical):
    # current rate throughout by default; historical: 0.1% on sells before 2023-08-28
    return 0.001 if historical and str(date) < '2023-08-28' else STAMP


@dataclass
class ExitRule:
    stop_loss: float | None = None       # e.g. 0.08: close below entry*(1-0.08)
    trailing: float | None = None        # e.g. 0.10: close below peak close*(1-0.10)
    take_profit: float | None = None     # e.g. 0.15: close above entry*(1+0.15)
    max_hold: int | None = None          # sessions held (counted from entry day) before forced exit
    signal_exit: np.ndarray | None = None  # [days x stocks] bool, exit when True at close


def _ok(*x):
    return all(np.isfinite(v) and v > 0 for v in x)


def simulate(p: Panel, entry_signal, score, exit_rule: ExitRule, capital, slots, start, end,
             universe=None, label='', entry_max_up=None, historical_stamp=False, data_end_value=None,
             decide_on_last=False, budget_fraction=1.0, swap_margin=None,
             buy_price_raw=None, buy_price_qfq=None, buy_budget_raw=None):
    """entry_signal/score: [days x stocks]; universe: [days x stocks] bool filter at signal close.

    Every price used to settle a fill (raw open, raw previous close, qfq open) must be finite and
    positive, otherwise the order stays pending (sells) or is rejected (buys). entry_max_up is
    measured against the execution day's raw previous close (the exchange reference price, which
    already reflects ex-rights adjustments), not the signal day's raw close.
    data_end_value: None keeps a position whose price series has ended (delisting, merger) at its
    last close; a fraction f closes it on the next session at f x that value (sensitivity check).
    info['ledger'] has one row per day (cash, positions, fills, fees, pending and rejected orders).
    decide_on_last: also make entry decisions at the last close (forward tracking); info['holdings']
    and info['next_buys'] then give the orders for the next open.
    budget_fraction g: each new buy gets at most g x NAV / slots and may only spend cash above (1 - g) x NAV
    (NAV = previous close); existing positions are never sold to restore the fraction. g = 1 is the default.
    swap_margin (added 2026-10-09, docs/research/ashare_cost_aware_exit_20261009.md): after the rule exits, a held
    stock is also marked for sale (reason SWAP) when the best eligible unheld entry candidate's score exceeds its own
    score by more than swap_margin; the freed slot is then filled by that candidate at the next open. None = off.
    Optional buy_price_raw/buy_price_qfq are paired [days, stocks] pre-friction buy references,
    indexed by execution day. Missing/nonpositive references reject that day's buy (DAY expiry).
    They affect buy price checks and buy accounting only; sells always use the real opening prices.
    buy_budget_raw, when supplied, is a [days, stocks] maximum pre-friction sizing reference known
    when the order is formed. Shares and fees are reserved at that cap; cheaper fills do not enlarge
    the order. An actual friction-inclusive buy price above its sizing cap by >1e-8 is rejected.
    This price cap does not itself freeze a multi-asset account's cash; callers must provide a
    causal budget reference (the motivating use is a flat, single-stock account).
    With all three omitted the original price, sizing, cost and numerical paths are unchanged.
    """
    t0 = time.monotonic()
    D, N = p.a['close_raw'].shape
    if (buy_price_raw is None) != (buy_price_qfq is None):
        raise ValueError('buy_price_raw and buy_price_qfq must be supplied together')
    checked_buy_arrays = []
    for name, value in (('buy_price_raw', buy_price_raw), ('buy_price_qfq', buy_price_qfq),
                        ('buy_budget_raw', buy_budget_raw)):
        if value is not None:
            value = np.asarray(value)
            if value.shape != (D, N) or not np.issubdtype(value.dtype, np.number):
                raise ValueError(f'{name} must be a numeric array with shape {(D, N)}')
        checked_buy_arrays.append(value)
    buy_price_raw, buy_price_qfq, buy_budget_raw = checked_buy_arrays
    o_raw, c_raw, pre_raw = p.a['open_raw'], p.a['close_raw'], p.a['preclose_raw']
    o_q, c_q = p.a['open_qfq'], p.a['close_qfq']
    fin = np.isfinite(c_q)
    last_idx = np.where(fin.any(axis=0), D - 1 - np.argmax(fin[::-1], axis=0), -1)
    days = np.where((p.dates >= start) & (p.dates <= end))[0]
    cash = float(capital)
    pos = {}  # j -> dict(shares, basis_q (open_qfq at entry), entry_value, entry_day, peak_q, entry_cost)
    pend_sell, pend_buy = set(), []
    trades, navs, ledger = [], [], []
    stuck = set()

    def close_trade(j, q, t, value, f, reason):
        trades.append(dict(code=str(p.codes[j]), entry_date=str(p.dates[q['entry_day']]), exit_date=str(p.dates[t]),
                           hold=int(t - q['entry_day']), entry_cost=q['entry_cost'], exit_net=value - f,
                           net_return=(value - f) / q['entry_cost'] - 1, reason=reason))

    for t in days:
        day = dict(date=str(p.dates[t]), buys=0, sells=0, buy_amount=0.0, sell_amount=0.0, fees=0.0,
                   pending_sells=0, rejected_buys=0)
        # ---- positions whose price series has ended
        for j in [j for j in pos if t > last_idx[j]]:
            stuck.add(j)
            if data_end_value is not None:
                q = pos.pop(j)
                pend_sell.discard(j)
                value = q['entry_value'] * q.get('last_q', q['basis_q']) / q['basis_q'] * data_end_value
                cash += value
                close_trade(j, q, t, value, 0.0, 'DATA_END')
        # ---- open: sells first
        for j in list(pend_sell):
            if j not in pos:
                pend_sell.discard(j); continue
            op = o_raw[t, j]
            if (not p.tradable[t, j] or not _ok(op, pre_raw[t, j], o_q[t, j])
                    or op <= pre_raw[t, j] * (1 - p.limit[t, j]) + 0.005):
                day['pending_sells'] += 1
                continue  # cannot sell today, keep pending (PENDING_UNEXECUTABLE)
            q = pos.pop(j)
            value = q['entry_value'] * o_q[t, j] / q['basis_q'] * (1 - SLIP)
            f = fee_sell(value, stamp_rate(p.dates[t], historical_stamp))
            cash += value - f
            pend_sell.discard(j)
            close_trade(j, q, t, value, f, q.get('reason', ''))
            day['sells'] += 1; day['sell_amount'] += value; day['fees'] += f
        # ---- open: buys
        nav_ref = navs[-1] if navs else float(capital)
        for j in pend_buy:
            if len(pos) >= slots or j in pos:
                continue
            op = o_raw[t, j] if buy_price_raw is None else buy_price_raw[t, j]
            buy_oq = o_q[t, j] if buy_price_qfq is None else buy_price_qfq[t, j]
            if (not p.tradable[t, j] or not _ok(op, pre_raw[t, j], buy_oq)
                    or op >= pre_raw[t, j] * (1 + p.limit[t, j]) - 0.005):
                day['rejected_buys'] += 1
                continue  # limit-up open, not trading or unusable prices: skip this entry
            if entry_max_up is not None and op > pre_raw[t, j] * (1 + entry_max_up):
                day['rejected_buys'] += 1
                continue  # limit order not filled
            px = op * (1 + SLIP)
            spendable = max(0.0, cash - (1 - budget_fraction) * nav_ref)
            budget = min(nav_ref * budget_fraction / slots, spendable)
            if buy_budget_raw is None:
                shares = np.floor(budget / px / 100) * 100
                while shares >= 100 and shares * px + fee_buy(shares * px) > spendable:
                    shares -= 100
            else:
                cap_ref = buy_budget_raw[t, j]
                if not _ok(cap_ref):
                    day['rejected_buys'] += 1
                    continue
                size_px = cap_ref * (1 + SLIP)
                if float(px) > float(size_px) + 1e-8:
                    day['rejected_buys'] += 1
                    continue
                shares = np.floor(budget / size_px / 100) * 100
                while shares >= 100 and shares * size_px + fee_buy(shares * size_px) > budget:
                    shares -= 100
            if shares < 100:
                day['rejected_buys'] += 1
                continue
            amount = shares * px
            f = fee_buy(amount)
            cash -= amount + f
            pos[j] = dict(shares=shares, basis_q=buy_oq * (1 + SLIP), entry_value=amount, entry_day=t,
                          peak_q=buy_oq, entry_cost=amount + f)
            day['buys'] += 1; day['buy_amount'] += amount; day['fees'] += f
        pend_buy = []
        # ---- close: mark to market
        mv = 0.0
        for j, q in pos.items():
            cq = c_q[t, j]
            if np.isfinite(cq):
                q['last_q'] = cq
                q['peak_q'] = max(q['peak_q'], cq)
            mv += q['entry_value'] * q.get('last_q', q['basis_q']) / q['basis_q']
        nav = cash + mv
        navs.append(nav)
        day.update(nav=nav, cash=cash, positions=len(pos))
        ledger.append(day)
        # ---- close: exit decisions
        for j, q in pos.items():
            if j in pend_sell:
                continue
            cq = q.get('last_q', q['basis_q'])
            held = t - q['entry_day'] + 1
            r = cq / q['basis_q'] - 1
            reason = None
            if exit_rule.stop_loss is not None and r <= -exit_rule.stop_loss:
                reason = 'STOP_LOSS'
            elif exit_rule.trailing is not None and cq <= q['peak_q'] * (1 - exit_rule.trailing):
                reason = 'TRAILING'
            elif exit_rule.take_profit is not None and r >= exit_rule.take_profit:
                reason = 'TAKE_PROFIT'
            elif exit_rule.signal_exit is not None and exit_rule.signal_exit[t, j]:
                reason = 'SIGNAL'
            elif exit_rule.max_hold is not None and held >= exit_rule.max_hold:
                reason = 'TIME'
            if reason:
                q['reason'] = reason
                pend_sell.add(j)
        # ---- close: cost-aware swaps (optional): replace a holding when a better candidate exists by a margin
        if swap_margin is not None and pos and (t < days[-1] or decide_on_last):
            ok = entry_signal[t] & p.tradable[t] & ~p.st[t]
            if universe is not None:
                ok &= universe[t]
            cand = [j for j in np.where(ok)[0] if j not in pos]
            if cand:
                order = sorted(cand, key=lambda j: -score[t, j])
                held = sorted([j for j in pos if j not in pend_sell], key=lambda j: score[t, j])
                k = 0
                for j in held:                           # weakest holding first, strongest candidate first
                    if k >= len(order):
                        break
                    if score[t, order[k]] - score[t, j] > swap_margin:
                        pos[j]['reason'] = 'SWAP'
                        pend_sell.add(j)
                        k += 1
                    else:
                        break
        # ---- close: entry decisions for the next open
        free = slots - len(pos) + len(pend_sell)
        if free > 0 and (t < days[-1] or decide_on_last):
            ok = entry_signal[t] & p.tradable[t] & ~p.st[t]
            if universe is not None:
                ok &= universe[t]
            idx = np.where(ok)[0]
            if len(idx):
                idx = idx[np.argsort(-score[t, idx])]
                pend_buy = [j for j in idx if j not in pos][:free]
    nav = pd.Series(navs, index=pd.to_datetime(p.dates[days]))
    tr = pd.DataFrame(trades)
    led = pd.DataFrame(ledger)
    # orders decided at the last close, to execute at the next open (forward tracking)
    last = days[-1]
    holdings = [dict(code=str(p.codes[j]), entry_date=str(p.dates[q['entry_day']]), shares=float(q['shares']),
                     value=float(q['entry_value'] * q.get('last_q', q['basis_q']) / q['basis_q']),
                     ret=float(q.get('last_q', q['basis_q']) / q['basis_q'] - 1), sell_next_open=j in pend_sell,
                     reason=q.get('reason', '') if j in pend_sell else '') for j, q in pos.items()]
    next_buys = [dict(code=str(p.codes[j]), score=float(score[last, j]), ref_close_raw=float(c_raw[last, j]))
                 for j in pend_buy]
    return nav, tr, dict(label=label, seconds=round(time.monotonic() - t0, 1), open_positions=len(pos),
                         holdings=holdings, next_buys=next_buys,
                         fees=float(led.fees.sum()), turnover=float((led.buy_amount + led.sell_amount).sum()),
                         pending_sell_days=int(led.pending_sells.sum()), rejected_buys=int(led.rejected_buys.sum()),
                         data_end_positions=len(stuck), ledger=led)


def stats(nav, tr, capital):
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    out = dict(final=float(nav.iloc[-1]), cagr=float((nav.iloc[-1] / capital) ** (1 / years) - 1),
               mdd=float((1 - nav / nav.cummax()).max()))
    if len(tr):
        w = tr.net_return > 0
        gross_win = (tr.exit_net - tr.entry_cost)[w].sum()
        gross_loss = -(tr.exit_net - tr.entry_cost)[~w].sum()
        out.update(trades=int(len(tr)), win_rate=float(w.mean()), avg_return=float(tr.net_return.mean()),
                   avg_win=float(tr.net_return[w].mean()) if w.any() else np.nan,
                   avg_loss=float(tr.net_return[~w].mean()) if (~w).any() else np.nan,
                   profit_factor=float(gross_win / gross_loss) if gross_loss > 0 else np.nan,
                   avg_hold=float(tr.hold.mean()), trades_per_year=float(len(tr) / years))
    yearly = nav.resample('YE').last()
    yearly = pd.concat([pd.Series([capital], index=[nav.index[0]]), yearly]).pct_change().dropna()
    out['yearly'] = {str(d.year): round(float(v), 4) for d, v in yearly.items()}
    return out
