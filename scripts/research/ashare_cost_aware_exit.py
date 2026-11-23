"""Cost-aware swap rule and drawdown-constrained budget fraction.

  python ashare_cost_aware_exit.py fixed [cut] train the R5C Ridge once (labels matured before <cut>-01-01, default 2018; alpha chosen once on <cut>-1)
  python ashare_cost_aware_exit.py dev_window [cut]  addendum: once-trained model vs yearly refit on <cut>-01 .. 2026-08 (default 2022)
  python ashare_cost_aware_exit.py dev        rules x accounts on mainboard_2014_2026, two score sets (yearly refit / fixed 2017), start shifts
  python ashare_cost_aware_exit.py sealed     same rules on the opened sealed segment (2026-08-21 close .. 09-30)
  python ashare_cost_aware_exit.py bootstrap  stationary block bootstrap of daily returns -> drawdown by budget fraction g

Rules (every close, next open execution, T+1): hard exits ST / limit-down / 8% stop / 20 sessions / score <= 0 or
missing; then either the current quantile rule (sell when the score leaves the day's top q) or the swap rule
(sell a holding when the best unheld candidate's score exceeds it by more than kappa, engine swap_margin).
No rolling selection: kappa = 0.3% (one round trip's cost) is fixed in advance for the
whole period; the other kappas are a sensitivity table only. The fixed model is trained once and never refitted.
"""
import json
import pickle
import sys
import time

import numpy as np
import pandas as pd

import ashare_daily_judgment as dj
import ashare_engine as e
import ashare_ml_signals as v1

V1 = e.ROOT / 'output/research_once/ashare_ml_signals_v1_20261006'
V2 = e.ROOT / 'output/research_once/ashare_ml_signals_v2_20261007'
WORK = e.ROOT / 'output/research_once/paper_tracking_work/mainboard_sealed_20260930'
OUT = e.ROOT / 'output/research_once/ashare_cost_aware_exit_20261009'
KAPPAS = [0.003, 0.0, 0.005, 0.01, 0.02]          # 0.003 is the main rule; the rest are sensitivity
QS = [0.1, 0.3, 0.5]
ENTRY, STOP, CAP_DAYS = 0.01, 0.08, 20
BOOKS = ((500000, 10, 1.0), (500000, 10, 0.5), (20000, 2, 1.0))


def rules(p, mask, s):
    events = dj.risk_events(p)
    with np.errstate(invalid='ignore'):
        hard = ~np.isfinite(s) | (s <= 0)
    out = {}
    for q in QS:
        out[f'q{q}'] = (e.ExitRule(stop_loss=STOP, max_hold=CAP_DAYS, signal_exit=events | dj.weak_signal(s, mask, q)), None)
    for k in KAPPAS:
        out[f'kappa{k}'] = (e.ExitRule(stop_loss=STOP, max_hold=CAP_DAYS, signal_exit=events | hard), k)
    return out


def account(p, mask, s, rule, swap, cap, k, start, end, g=1.0):
    sfill = np.nan_to_num(s, nan=-9).astype(np.float32)
    sig = np.isfinite(s) & (sfill > ENTRY)
    nav, tr, info = e.simulate(p, sig, sfill, rule, cap, k, start, end, universe=mask, entry_max_up=0.01,
                               budget_fraction=g, swap_margin=swap)
    st = e.stats(nav, tr, cap)
    dd = 1 - nav / nav.cummax()
    under = (dd > 1e-12).astype(int)
    led = info['ledger']

    def period(a, b):
        x, before = nav[a:b], nav[nav.index < a]
        return float(x.iloc[-1] / before.iloc[-1] - 1) if len(x) and len(before) else np.nan

    reasons = tr.reason.value_counts(normalize=True).round(3).to_dict() if len(tr) else {}
    row = dict(final=st['final'], ret=float(nav.iloc[-1] / cap - 1), cagr=st['cagr'], mdd=st['mdd'],
               ratio=st['cagr'] / st['mdd'] if st['mdd'] > 0 else np.nan,
               longest_underwater=int(under.groupby((under == 0).cumsum()).sum().max()),
               trades=st.get('trades', 0), win_rate=st.get('win_rate'), avg_trade=st.get('avg_return'),
               avg_win=st.get('avg_win'), avg_loss=st.get('avg_loss'), profit_factor=st.get('profit_factor'),
               avg_hold=st.get('avg_hold'), invested=float((1 - led.cash / led.nav).mean()), fees=info['fees'],
               turnover_x=info['turnover'] / float(nav.mean()), last12=period('2025-08-22', '2026-08-21'),
               prev12=period('2024-08-22', '2025-08-21'), reasons=json.dumps(reasons), yearly=json.dumps(st['yearly']))
    daily = pd.DataFrame(dict(nav=nav.to_numpy(), fees=led.fees.to_numpy(), turnover=(led.buy_amount + led.sell_amount).to_numpy()),
                         index=nav.index)
    return row, daily, tr


def fixed(cut=2018, fam='RIDGE'):
    """One model (feature set C, R5 trade label; family RIDGE / HGB / MLP) trained on labels matured before <cut>-01-01;
    its hyper-parameters chosen once on <cut>-1, then never refitted. MLP: three seeds averaged."""
    cut = int(cut)
    t0 = time.monotonic()
    panel = 'mainboard_2014_2026'
    X = np.load(V2 / f'{panel}_XC.npy', mmap_mode='r')
    y = np.load(V2 / f'{panel}_rawT5.npy')
    mask = np.load(V1 / f'{panel}_mask.npy')
    dates = e.load_panel(panel).dates
    D, N = y.shape
    H = 5
    yclip = np.clip(y, -0.2, 0.2)
    sampled = np.arange(60, D, 5)
    a_val, a = int(np.searchsorted(dates, f'{cut - 1}-01-01')), int(np.searchsorted(dates, f'{cut}-01-01'))
    Xtr, ytr = v1.rows_for(X, yclip, mask, sampled[sampled + 1 + H < a_val])
    best, best_score, cand_scores = None, -np.inf, []
    for i, c in enumerate(v1.candidates(fam)):
        m = v1.model(fam, c).fit(Xtr, ytr)
        tops = []
        for t in range(a_val, a - 1 - H):
            ok = mask[t] & np.isfinite(y[t])
            if ok.sum() < 50:
                continue
            pr = m.predict(np.nan_to_num(X[:, t, ok].T.astype(np.float32), nan=0.5))
            k = max(1, int(0.1 * ok.sum()))
            tops.append(float(np.mean(y[t, ok][np.argsort(-pr)[:k]])))
        s = float(np.mean(tops)) if tops else -np.inf
        cand_scores.append(s)
        if s > best_score:
            best, best_score = i, s
    del Xtr, ytr
    Xf, yf = v1.rows_for(X, yclip, mask, sampled[sampled + 1 + H < a])
    Xf, yf = Xf[-600_000:], yf[-600_000:]
    seeds = (0, 1, 2) if fam == 'MLP' else (0,)
    ms = [v1.model(fam, v1.candidates(fam)[best], sd).fit(Xf, yf) for sd in seeds]
    scores = np.full((D, N), np.nan, np.float32)
    for t in range(a, D):
        ok = mask[t]
        if ok.any():
            scores[t, ok] = np.mean([m.predict(np.nan_to_num(X[:, t, ok].T.astype(np.float32), nan=0.5)) for m in ms], axis=0)
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f'fixed{cut - 1}' + ('' if fam == 'RIDGE' else f'_{fam}')
    np.save(OUT / f'scores_{tag}_C_RIDGE_HR5.npy' if fam == 'RIDGE' else OUT / f'scores_{tag}_C_HR5.npy', scores)
    pickle.dump(dict(models=ms, candidate=v1.candidates(fam)[best], fset='C', label='R5', fam=fam, trained_before=f'{cut}-01-01',
                     train_rows=int(len(yf)), last_label_day=str(dates[sampled[sampled + 1 + H < a][-1]])),
                open(OUT / f'model_{tag}_C_HR5.pkl' if fam != 'RIDGE' else OUT / f'model_{tag}_C_RIDGE_HR5.pkl', 'wb'))
    json.dump(dict(fam=fam, chosen=best, candidate=v1.candidates(fam)[best], val_top10=cand_scores, train_rows=int(len(yf)), cut=cut),
              open(OUT / f'{tag}_fit.json', 'w'), indent=1)
    print(json.dumps(dict(fam=fam, chosen=best, candidate=v1.candidates(fam)[best], val_top10=[round(x, 5) for x in cand_scores],
                          rows=int(len(yf)), seconds=round(time.monotonic() - t0))))


def dev_window(cut='2022'):
    """Addendum (after seeing fixed2017): a model trained once on everything before <cut>, tested <cut>-01 .. 2026-08-21
    against the yearly-refit scores on the same window. Rules q0.3, kappa 0.3% and 1%; 500k full/half, 20k; 10 starts."""
    cut = int(cut)
    tag = f'fixed{cut - 1}'
    p = e.load_panel('mainboard_2014_2026')
    mask = np.load(V1 / 'mainboard_2014_2026_mask.npy')
    score_sets = {'yearly': np.load(V2 / 'mainboard_2014_2026_scores_C_RIDGE_HR5.npy'),
                  tag: np.load(OUT / f'scores_{tag}_C_RIDGE_HR5.npy')}
    for fam in ('HGB', 'MLP'):                      # once-trained complex models (addendum 2), if fitted
        fsc = OUT / f'scores_{tag}_{fam}_C_HR5.npy'
        if fsc.exists():
            score_sets[f'{tag}_{fam}'] = np.load(fsc)
    start, end = f'{cut}-01-04', str(p.dates[-1])
    d0 = int(np.searchsorted(p.dates, start))
    rows, shifts = [], []
    for sname, s in score_sets.items():
        rl = rules(p, mask, s)
        for name in ('q0.3', 'kappa0.003', 'kappa0.01'):
            rule, swap = rl[name]
            for cap, k, g in BOOKS:
                r, _, _ = account(p, mask, s, rule, swap, cap, k, start, end, g)
                rows.append(dict(scores=sname, rule=name, capital=cap, g=g, **r))
                print(json.dumps({x: (round(v, 4) if isinstance(v, float) else v) for x, v in rows[-1].items() if x != 'yearly'}), flush=True)
            for sh in range(10):
                r, _, _ = account(p, mask, s, rule, swap, 500000, 10, str(p.dates[d0 + sh]), end)
                shifts.append(dict(scores=sname, rule=name, shift=sh, cagr=r['cagr'], mdd=r['mdd']))
    pd.DataFrame(rows).to_csv(OUT / f'window{cut}_accounts.csv', index=False)
    sh = pd.DataFrame(shifts)
    summ = sh.groupby(['scores', 'rule']).agg(cagr_med=('cagr', 'median'), cagr_min=('cagr', 'min'), cagr_max=('cagr', 'max'),
                                              mdd_med=('mdd', 'median'), mdd_max=('mdd', 'max')).reset_index()
    summ['ratio_med'] = summ.cagr_med / summ.mdd_med
    summ.to_csv(OUT / f'window{cut}_start_shift_summary.csv', index=False)
    pd.set_option('display.width', 220)
    print(summ.round(3).to_string(index=False))


def dev():
    t0 = time.monotonic()
    p = e.load_panel('mainboard_2014_2026')
    mask = np.load(V1 / 'mainboard_2014_2026_mask.npy')
    score_sets = {'yearly': np.load(V2 / 'mainboard_2014_2026_scores_C_RIDGE_HR5.npy'),
                  'fixed2017': np.load(OUT / 'scores_fixed2017_C_RIDGE_HR5.npy')}
    end = str(p.dates[-1])
    d0 = int(np.searchsorted(p.dates, '2018-01-02'))
    OUT.mkdir(parents=True, exist_ok=True)
    rows, shifts, dailies = [], [], {}
    for sname, s in score_sets.items():
        for name, (rule, swap) in rules(p, mask, s).items():
            for cap, k, g in BOOKS:
                for slip in (0.001, 0.002):
                    e.SLIP = slip
                    r, daily, tr = account(p, mask, s, rule, swap, cap, k, '2018-01-02', end, g)
                    e.SLIP = 0.001
                    rows.append(dict(scores=sname, rule=name, capital=cap, g=g, slip=slip, **r))
                    print(json.dumps({x: (round(v, 4) if isinstance(v, float) else v) for x, v in rows[-1].items() if x != 'yearly'}), flush=True)
                    if cap == 500000 and g == 1.0 and slip == 0.001:
                        dailies[f'{sname}_{name}'] = daily
                        tr.to_csv(OUT / f'dev_trades_{sname}_{name}.csv', index=False)
            for sh in range(10):
                r, _, _ = account(p, mask, s, rule, swap, 500000, 10, str(p.dates[d0 + sh]), end)
                shifts.append(dict(scores=sname, rule=name, shift=sh, cagr=r['cagr'], mdd=r['mdd'], turnover_x=r['turnover_x'], fees=r['fees']))
            print(json.dumps(dict(scores=sname, rule=name, seconds=round(time.monotonic() - t0))), flush=True)
    pd.DataFrame(rows).to_csv(OUT / 'dev_accounts.csv', index=False)
    sh = pd.DataFrame(shifts)
    sh.to_csv(OUT / 'dev_start_shifts.csv', index=False)
    summ = sh.groupby(['scores', 'rule']).agg(cagr_med=('cagr', 'median'), cagr_min=('cagr', 'min'), cagr_max=('cagr', 'max'),
                                              mdd_med=('mdd', 'median'), mdd_max=('mdd', 'max'), turnover=('turnover_x', 'median'),
                                              fees=('fees', 'median')).reset_index()
    summ['ratio_med'] = summ.cagr_med / summ.mdd_med
    summ.to_csv(OUT / 'dev_start_shift_summary.csv', index=False)
    pd.DataFrame({k: v.nav for k, v in dailies.items()}).to_csv(OUT / 'dev_navs_500k_full.csv')
    pd.set_option('display.width', 220)
    print(summ.round(3).to_string(index=False))


def sealed():
    p = e.load_panel('mainboard_sealed_20260930')
    mask = np.load(WORK / 'mainboard_sealed_20260930_mask.npy')
    X = np.load(WORK / 'mainboard_sealed_20260930_XC.npy', mmap_mode='r')
    models = {'yearly': pickle.load(open(V2 / 'mainboard_2014_2026_model_C_RIDGE_HR5_2026.pkl', 'rb')),
              'fixed2017': pickle.load(open(OUT / 'model_fixed2017_C_RIDGE_HR5.pkl', 'rb'))}
    rows = []
    for sname, mod in models.items():
        s = np.full(mask.shape, np.nan, np.float32)
        for t in np.where(p.dates >= '2026-07-01')[0]:
            ok = mask[t]
            xt = np.nan_to_num(X[:, t, ok].T.astype(np.float32), nan=0.5)
            s[t, ok] = np.mean([mm.predict(xt) for mm in mod['models']], axis=0)
        for name, (rule, swap) in rules(p, mask, s).items():
            for cap, k, g in BOOKS:
                r, _, _ = account(p, mask, s, rule, swap, cap, k, '2026-08-21', '2026-09-30', g)
                rows.append(dict(scores=sname, rule=name, capital=cap, g=g, ret=r['ret'], trades=r['trades'], win_rate=r['win_rate'],
                                 avg_hold=r['avg_hold'], fees=r['fees'], reasons=r['reasons']))
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / 'sealed_accounts.csv', index=False)
    pd.set_option('display.width', 220)
    print(d.round(4).to_string(index=False))


def bootstrap(n_paths=2000, block=20, seed=0):
    """Stationary block bootstrap of the full-budget daily returns; account return at budget fraction g ~ g x r."""
    navs = pd.read_csv(OUT / 'dev_navs_500k_full.csv', index_col=0, parse_dates=True)
    series = {k: navs[k] for k in ('yearly_q0.3', 'fixed2017_q0.3', 'yearly_kappa0.003', 'fixed2017_kappa0.003') if k in navs}
    rng = np.random.default_rng(seed)
    rows = []
    for name, nav in series.items():
        r = nav.pct_change().dropna().to_numpy()
        n = len(r)
        paths = np.empty((n_paths, n), np.float64)
        for i in range(n_paths):                     # stationary bootstrap: geometric block lengths, mean `block`
            pos, out = 0, np.empty(n)
            while pos < n:
                start = rng.integers(n)
                length = min(n - pos, rng.geometric(1 / block))
                idx = (start + np.arange(length)) % n
                out[pos:pos + length] = r[idx]
                pos += length
            paths[i] = out
        yrs = n / 243
        for g in (0.3, 0.4, 0.5, 0.6, 0.7, 1.0):
            x = np.cumprod(1 + g * paths, axis=1)
            mdd = (1 - x / np.maximum.accumulate(x, axis=1)).max(axis=1)
            cagr = x[:, -1] ** (1 / yrs) - 1
            hist = np.cumprod(1 + g * r)
            rows.append(dict(series=name, g=g, p_mdd_gt_17=float((mdd > 0.17).mean()), p_mdd_gt_25=float((mdd > 0.25).mean()),
                             mdd_med=float(np.median(mdd)), mdd_p90=float(np.quantile(mdd, 0.9)),
                             cagr_med=float(np.median(cagr)), cagr_p10=float(np.quantile(cagr, 0.1)),
                             historical_mdd=float((1 - hist / np.maximum.accumulate(hist)).max()),
                             historical_cagr=float(hist[-1] ** (1 / yrs) - 1)))
    d = pd.DataFrame(rows)
    d.to_csv(OUT / 'bootstrap_budget_fraction.csv', index=False)
    pd.set_option('display.width', 220)
    print(d.round(3).to_string(index=False))


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'fixed':
        fixed(*sys.argv[2:4])
    elif cmd == 'dev_window':
        dev_window(*sys.argv[2:3])
    else:
        {'dev': dev, 'sealed': sealed, 'bootstrap': bootstrap}[cmd]()
