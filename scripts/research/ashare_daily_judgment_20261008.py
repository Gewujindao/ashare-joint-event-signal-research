"""Daily re-judgment of every position (docs/research/ashare_daily_judgment_20261008.md).

  python ashare_daily_judgment_20261008.py dev       2018-01 .. 2026-08-21 on mainboard_2014_2026
  python ashare_daily_judgment_20261008.py sealed    2026-08-21 close .. 2026-09-30 on mainboard_sealed_20260930

Each close, a held stock is sold at the next tradable open if any holds: it is ST/*ST that day, it closed
limit-down, its model score fell out of the day's top q of eligible stocks or to <= 0 or is missing, the close
is 8% below the entry, or it has been held 20 sessions. Entries: score > 1%, best first, open <= ref x 1.01.
"""
import json
import sys

import numpy as np
import pandas as pd

import ashare_engine_20261006 as e

V1 = e.ROOT / 'output/research_once/ashare_ml_signals_v1_20261006'
V2 = e.ROOT / 'output/research_once/ashare_ml_signals_v2_20261007'
WORK = e.ROOT / 'output/research_once/paper_tracking_work/mainboard_sealed_20260930'
OUT = e.ROOT / 'output/research_once/ashare_daily_judgment_20261008'
MODELS = {'R5C': ('C_RIDGE_HR5', e.ExitRule(stop_loss=0.04, take_profit=0.06, max_hold=5)),
          'A10B': ('B_RIDGE_HA10', e.ExitRule(max_hold=10)),
          'A20B': ('B_RIDGE_HA20', e.ExitRule(max_hold=20)),
          'A20MLP': ('B_MLP_HA20', e.ExitRule(max_hold=20))}
QS = (0.10, 0.30)
ENTRY, STOP, CAP_DAYS = 0.01, 0.08, 20


def risk_events(p):
    cr, pr = p.a['close_raw'], p.a['preclose_raw']
    with np.errstate(invalid='ignore'):
        ld = np.isfinite(cr) & np.isfinite(pr) & (cr <= pr * (1 - p.limit) + 0.005)
    return p.st | ld


def weak_signal(s, mask, q):
    """True where the score is missing, <= 0, or outside the day's top q of the eligible stocks."""
    top = pd.DataFrame(np.where(mask, s, np.nan)).rank(axis=1, ascending=False, pct=True).to_numpy()
    with np.errstate(invalid='ignore'):
        return ~np.isfinite(s) | (s <= 0) | ~(top <= q)


def account(p, mask, s, rule, cap, k, start, end, g=1.0):
    sig = np.isfinite(s) & (np.nan_to_num(s, nan=-9) > ENTRY)
    nav, tr, info = e.simulate(p, sig, np.nan_to_num(s, nan=-9).astype(np.float32), rule, cap, k, start, end,
                               universe=mask, entry_max_up=0.01, budget_fraction=g)
    st = e.stats(nav, tr, cap)
    dd = 1 - nav / nav.cummax()
    under = (dd > 1e-12).astype(int)
    led = info['ledger']

    def period(a, b):
        x, before = nav[a:b], nav[nav.index < a]
        return float(x.iloc[-1] / before.iloc[-1] - 1) if len(x) and len(before) else np.nan

    reasons = tr.reason.value_counts(normalize=True).round(3).to_dict() if len(tr) else {}
    return dict(final=st['final'], ret=float(nav.iloc[-1] / cap - 1), cagr=st['cagr'], mdd=st['mdd'],
                ratio=st['cagr'] / st['mdd'] if st['mdd'] > 0 else np.nan,
                longest_underwater=int(under.groupby((under == 0).cumsum()).sum().max()),
                trades=st.get('trades', 0), win_rate=st.get('win_rate'), avg_trade=st.get('avg_return'),
                avg_win=st.get('avg_win'), avg_loss=st.get('avg_loss'), profit_factor=st.get('profit_factor'),
                avg_hold=st.get('avg_hold'), invested=float((1 - led.cash / led.nav).mean()), fees=info['fees'],
                turnover_x=info['turnover'] / float(nav.mean()), last12=period('2025-08-22', '2026-08-21'),
                prev12=period('2024-08-22', '2025-08-21'), reasons=json.dumps(reasons), yearly=json.dumps(st['yearly']))


def daily_rule(p, mask, s, q):
    return e.ExitRule(stop_loss=STOP, max_hold=CAP_DAYS, signal_exit=risk_events(p) | weak_signal(s, mask, q))


def dev():
    p = e.load_panel('mainboard_2014_2026')
    mask = np.load(V1 / 'mainboard_2014_2026_mask.npy')
    end = str(p.dates[-1])
    d0 = int(np.searchsorted(p.dates, '2018-01-02'))
    events = risk_events(p)
    rows, shifts = [], []
    for m, (tag, fixed) in MODELS.items():
        s = np.load(V2 / f'mainboard_2014_2026_scores_{tag}.npy')
        variants = [('fixed', None, fixed)] + [(f'daily_q{q}', q, daily_rule(p, mask, s, q)) for q in QS]
        for name, q, rule in variants:
            for cap, k, g in ((500000, 10, 1.0), (500000, 10, 0.5), (20000, 2, 1.0)):
                for slip in (0.001, 0.002):
                    e.SLIP = slip
                    r = account(p, mask, s, rule, cap, k, '2018-01-02', end, g)
                    e.SLIP = 0.001
                    rows.append(dict(model=m, variant=name, capital=cap, g=g, slip=slip, **r))
                    print(json.dumps({x: (round(v, 4) if isinstance(v, float) else v) for x, v in rows[-1].items()
                                      if x not in ('yearly',)}), flush=True)
            for sh in range(10):                       # start-date sensitivity, 500k full budget
                r = account(p, mask, s, rule, 500000, 10, str(p.dates[d0 + sh]), end)
                shifts.append(dict(model=m, variant=name, shift=sh, cagr=r['cagr'], mdd=r['mdd'], avg_hold=r['avg_hold'],
                                   turnover_x=r['turnover_x']))
            # random entries with the same risk exits, held for the model version's average holding period
            if q is not None:
                hold = max(1, int(round(rows[-6]['avg_hold'] or 1)))
                for seed in range(3):
                    rnd = np.random.default_rng(seed).random(s.shape).astype(np.float32) * 0.1 + 0.02
                    rnd[~np.isfinite(s)] = np.nan
                    rr = e.ExitRule(stop_loss=STOP, max_hold=hold, signal_exit=events)
                    r = account(p, mask, rnd, rr, 500000, 10, '2018-01-02', end)
                    rows.append(dict(model=m, variant=f'random_{name}_seed{seed}', capital=500000, g=1.0, slip=0.001, **r))
                    print(json.dumps({x: (round(v, 4) if isinstance(v, float) else v) for x, v in rows[-1].items()
                                      if x not in ('yearly',)}), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / 'dev_accounts.csv', index=False)
    sh = pd.DataFrame(shifts)
    sh.to_csv(OUT / 'dev_start_shifts.csv', index=False)
    summ = sh.groupby(['model', 'variant']).agg(cagr_med=('cagr', 'median'), cagr_min=('cagr', 'min'), cagr_max=('cagr', 'max'),
                                                 mdd_med=('mdd', 'median'), mdd_max=('mdd', 'max'), hold=('avg_hold', 'median'),
                                                 turnover=('turnover_x', 'median')).reset_index()
    summ['ratio_med'] = summ.cagr_med / summ.mdd_med
    summ.to_csv(OUT / 'dev_start_shift_summary.csv', index=False)
    pd.set_option('display.width', 220)
    print(summ.round(3).to_string(index=False))


def sealed():
    """Same rules on the opened sealed segment, from the 2026-08-21 close (scores from the frozen 2026 models)."""
    import pickle
    p = e.load_panel('mainboard_sealed_20260930')
    mask = np.load(WORK / 'mainboard_sealed_20260930_mask.npy')
    X = {fs: np.load(WORK / f'mainboard_sealed_20260930_X{fs}.npy', mmap_mode='r') for fs in ('B', 'C')}
    days = np.where(p.dates >= '2026-07-01')[0]               # enough history for the daily ranks
    rows = []
    for m, (tag, fixed) in MODELS.items():
        mod = pickle.load(open(V2 / f'mainboard_2014_2026_model_{tag}_2026.pkl', 'rb'))
        s = np.full(mask.shape, np.nan, np.float32)
        for t in days:
            ok = mask[t]
            xt = np.nan_to_num(X[mod['fset']][:, t, ok].T.astype(np.float32), nan=0.5)
            s[t, ok] = np.mean([mm.predict(xt) for mm in mod['models']], axis=0)
        for name, rule in [('fixed', fixed)] + [(f'daily_q{q}', daily_rule(p, mask, s, q)) for q in QS]:
            for cap, k, g in ((500000, 10, 1.0), (500000, 10, 0.5), (20000, 2, 1.0)):
                r = account(p, mask, s, rule, cap, k, '2026-08-21', '2026-09-30', g)
                rows.append(dict(model=m, variant=name, capital=cap, g=g, ret=r['ret'], trades=r['trades'], win_rate=r['win_rate'],
                                 avg_hold=r['avg_hold'], reasons=r['reasons']))
                print(json.dumps({x: (round(v, 4) if isinstance(v, float) else v) for x, v in rows[-1].items()}), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / 'sealed_accounts.csv', index=False)


if __name__ == '__main__':
    dev() if sys.argv[1] == 'dev' else sealed()
