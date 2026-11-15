"""Label form for a once-trained LightGBM (docs/research/label_form_20261009.md, registration written before running).

  python label_form_20261009.py train     three models (clipped-return MSE / daily rank MSE / up-down binary) on the fixed()
                                          rows; tree count chosen once on 2021 (top-10% mean raw label); scores 2022-01-04 ..
  python label_form_20261009.py signals   yearly R2oos, deciles, rank IC, candidate counts for the five score sets
  python label_form_20261009.py accounts  main rule and rank-band rule on the standard grid; registered criterion

Nothing is chosen after seeing results: every setting is in the registration. Existing scripts and the engine are not
modified; this file only imports them.
"""
import json
import os
import pickle
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '4')  # allow the continuation runner to cap reporting at two threads
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import ashare_daily_judgment_20261008 as dj
import ashare_engine_20261006 as e

V1 = e.ROOT / 'output/research_once/ashare_ml_signals_v1_20261006'
V2 = e.ROOT / 'output/research_once/ashare_ml_signals_v2_20261007'
CAE = e.ROOT / 'output/research_once/ashare_cost_aware_exit_20261009'
OUT = e.ROOT / 'output/research_once/label_form_20261009'
PANEL = 'mainboard_2014_2026'
H, STEP, FIRST, MAX_ROWS = 5, 5, 60, 600_000
VAL_CUT, CUT = '2021-01-01', '2022-01-01'            # searchsorted -> 2021-01-04 / 2022-01-04 (same as fixed())
START, END = '2022-01-04', '2026-08-21'
MIN_N, DECILE = 50, 0.1
PARAMS = dict(num_leaves=31, learning_rate=0.05, min_child_samples=100, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, seed=0, num_threads=4, verbose=-1)
MAX_TREES = 300
CHECKPOINTS = list(range(10, MAX_TREES + 1, 10))
LABELS = {'lgb_mse': 'regression', 'lgb_rank': 'regression', 'lgb_bin': 'binary'}
CONTROLS = {'ridge_fixed2021': CAE / 'scores_fixed2021_C_RIDGE_HR5.npy', 'hgb_fixed2021': CAE / 'scores_fixed2021_HGB_C_HR5.npy'}
SETS = list(LABELS) + list(CONTROLS)
ENTRY, STOP, CAP_DAYS, Q_MAIN = dj.ENTRY, dj.STOP, dj.CAP_DAYS, 0.3
X_IN, Y_OUT = 0.10, 0.50
BOOKS = ((500000, 10, 1.0), (500000, 10, 0.5), (20000, 2, 1.0))
SLIPS = (0.001, 0.002)
N_SHIFTS = 10
PRIMARY = (500000, 1.0, 0.001)
YEARS = ('2022', '2023', '2024', '2025', '2026')
T5_CHECK = dict(top=0.00338, ic=0.082)               # fixed2021 Ridge 2022-2026 from t5_fixed_gap_20261009.md


def log(**kw):
    print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in kw.items()}, default=str), flush=True)


# ----------------------------------------------------------------------------------------------- training
def rows(X, y, mask, days, with_labels=True):
    """Rows of the sampled days (same selection as ashare_ml_signals_20261006.rows_for); three label vectors."""
    Xs, yc, yr, yb, raw, lens = [], [], [], [], [], []
    for t in days:
        ok = mask[t] & np.isfinite(y[t])
        n = int(ok.sum())
        if n == 0:
            continue
        Xs.append(np.nan_to_num(X[:, t, ok].T.astype(np.float32), nan=0.5))
        yt = y[t, ok].astype(np.float32)
        raw.append(yt)
        lens.append(n)
        if with_labels:
            yc.append(np.clip(yt, -0.2, 0.2))
            yr.append(pd.Series(yt).rank(pct=True, method='average').to_numpy(np.float32))
            yb.append((yt > 0).astype(np.float32))
    Xm = np.vstack(Xs)
    labels = dict(lgb_mse=np.concatenate(yc), lgb_rank=np.concatenate(yr), lgb_bin=np.concatenate(yb)) if with_labels else None
    return Xm, labels, np.concatenate(raw), np.array(lens)


def make_model(name, n_trees):
    import lightgbm as lgb
    if LABELS[name] == 'binary':
        return lgb.LGBMClassifier(objective='binary', n_estimators=n_trees, **PARAMS)
    return lgb.LGBMRegressor(objective='regression', n_estimators=n_trees, **PARAMS)


def predict(m, name, Xm, k=None):
    if LABELS[name] == 'binary':
        return m.predict_proba(Xm, num_iteration=k)[:, 1]
    return m.predict(Xm, num_iteration=k)


def top10_by_day(pr, raw, lens):
    """Mean over days of the mean raw label of the top-10% predictions (validation metric of fixed())."""
    out, pos = [], 0
    for n in lens:
        seg_p, seg_y = pr[pos:pos + n], raw[pos:pos + n]
        pos += n
        if n < MIN_N:
            continue
        k = max(1, int(DECILE * n))
        out.append(float(np.mean(seg_y[np.argsort(-seg_p)[:k]])))
    return float(np.mean(out)) if out else -np.inf


def train():
    t0 = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    X = np.load(V2 / f'{PANEL}_XC.npy', mmap_mode='r')
    y = np.load(V2 / f'{PANEL}_rawT5.npy')
    mask = np.load(V1 / f'{PANEL}_mask.npy')
    dates = e.load_panel(PANEL).dates
    D, N = y.shape
    sampled = np.arange(FIRST, D, STEP)
    a_val, a = int(np.searchsorted(dates, VAL_CUT)), int(np.searchsorted(dates, CUT))
    assert str(dates[a]) == START
    # ---- stage 1: tree count chosen once on 2021
    Xtr, ytr, _, _ = rows(X, y, mask, sampled[sampled + 1 + H < a_val])
    log(stage='val_rows', rows=int(len(Xtr)), last_day=str(dates[sampled[sampled + 1 + H < a_val][-1]]), seconds=round(time.monotonic() - t0))
    models_val = {}
    for name in LABELS:
        m = make_model(name, MAX_TREES).fit(Xtr, ytr[name])
        models_val[name] = m
        log(stage='val_fit', model=name, rows=int(len(Xtr)), seconds=round(time.monotonic() - t0))
    del Xtr, ytr
    val_days = [t for t in range(a_val, a - 1 - H) if (mask[t] & np.isfinite(y[t])).sum() >= MIN_N]
    Xv, _, raw_v, lens_v = rows(X, y, mask, val_days, with_labels=False)
    log(stage='val_matrix', days=len(val_days), rows=int(len(Xv)), seconds=round(time.monotonic() - t0))
    chosen, curves = {}, {}
    for name, m in models_val.items():
        curve = [top10_by_day(predict(m, name, Xv, k), raw_v, lens_v) for k in CHECKPOINTS]
        best = int(np.argmax(curve))                       # argmax returns the first maximum -> fewer trees on ties
        chosen[name], curves[name] = CHECKPOINTS[best], curve
        log(stage='val_curve', model=name, chosen_trees=chosen[name], best_val_top10=curve[best],
            curve=[round(c, 5) for c in curve], seconds=round(time.monotonic() - t0))
    del Xv, raw_v, lens_v, models_val
    # ---- stage 2: final fit on the last 600k rows before 2022, then score 2022-01-04 .. end
    fin_days = sampled[sampled + 1 + H < a]
    Xf, yf, _, _ = rows(X, y, mask, fin_days)
    Xf = Xf[-MAX_ROWS:]
    yf = {k: v[-MAX_ROWS:] for k, v in yf.items()}
    log(stage='final_rows', rows=int(len(Xf)), last_day=str(dates[fin_days[-1]]), pos_share=float(yf['lgb_bin'].mean()),
        seconds=round(time.monotonic() - t0))
    models = {}
    for name in LABELS:
        models[name] = make_model(name, chosen[name]).fit(Xf, yf[name])
        log(stage='final_fit', model=name, trees=chosen[name], seconds=round(time.monotonic() - t0))
    del Xf, yf
    scores = {name: np.full((D, N), np.nan, np.float32) for name in LABELS}
    for t in range(a, D):
        ok = mask[t]
        if not ok.any():
            continue
        xt = np.nan_to_num(X[:, t, ok].T.astype(np.float32), nan=0.5)
        for name, m in models.items():
            scores[name][t, ok] = predict(m, name, xt)
    for name in LABELS:
        np.save(OUT / f'scores_{name}.npy', scores[name])
        pickle.dump(dict(model=models[name], label=name, objective=LABELS[name], params=PARAMS, trees=chosen[name],
                         trained_before=CUT, train_rows=MAX_ROWS), open(OUT / f'model_{name}.pkl', 'wb'))
    json.dump(dict(params=PARAMS, checkpoints=CHECKPOINTS, chosen_trees=chosen, val_curves=curves, val_days=len(val_days),
                   final_rows=MAX_ROWS, seconds=round(time.monotonic() - t0)), open(OUT / 'fit.json', 'w'), indent=1)
    log(stage='done', chosen_trees=chosen, seconds=round(time.monotonic() - t0))


# ----------------------------------------------------------------------------------------------- signal metrics
def load_scores():
    out = {name: np.load(OUT / f'scores_{name}.npy') for name in LABELS}
    out.update({name: np.load(path) for name, path in CONTROLS.items()})
    return out


def signals():
    t0 = time.monotonic()
    y = np.load(V2 / f'{PANEL}_rawT5.npy')
    mask = np.load(V1 / f'{PANEL}_mask.npy')
    dates = e.load_panel(PANEL).dates
    years = pd.Series(dates).str[:4].to_numpy()
    days = np.where((dates >= START) & (dates <= END))[0]
    rows_out, daily_all = [], {}
    for sname, s in load_scores().items():
        elig = mask & np.isfinite(y) & np.isfinite(s)
        with np.errstate(invalid='ignore'):
            cand = mask & np.isfinite(s) & (np.nan_to_num(s, nan=-9) > ENTRY)
        daily = []
        for t in days:
            m = elig[t]
            n = int(m.sum())
            if n < MIN_N:
                continue
            yt, st = y[t][m].astype(np.float64), s[t][m].astype(np.float64)
            k = max(1, int(DECILE * n))
            order = np.argsort(-st, kind='stable')
            daily.append(dict(date=dates[t], year=years[t], n=n, top=float(yt[order[:k]].mean()), bottom=float(yt[order[-k:]].mean()),
                              all=float(yt.mean()), ic=float(spearmanr(st, yt).correlation), cand=int(cand[t].sum()),
                              sse=float(((yt - st) ** 2).sum()), ssy=float((yt ** 2).sum())))
        d = pd.DataFrame(daily)
        d.to_csv(OUT / f'daily_{sname}.csv', index=False)
        daily_all[sname] = d
        for label, g in [(yr, d[d.year == yr]) for yr in YEARS] + [('2022-2026', d)]:
            if not len(g):
                continue
            rows_out.append(dict(scores=sname, year=label, days=int(len(g)), n_mean=float(g.n.mean()),
                                 r2oos=float(1 - g.sse.sum() / g.ssy.sum()),
                                 top_mean=float(g.top.mean()), bottom_mean=float(g.bottom.mean()), spread=float((g.top - g.bottom).mean()),
                                 all_mean=float(g['all'].mean()), top_net=float(g.top.mean()) - 0.00272,
                                 rank_ic=float(g.ic.mean()), ic_t=float(g.ic.mean() / g.ic.std() * np.sqrt(len(g))),
                                 ic_pos_share=float((g.ic > 0).mean()), cand_days_frac=float((g.cand > 0).mean()),
                                 cand_mean=float(g.cand.mean()), score_mean=np.nan))
        log(scores=sname, days=int(len(d)), seconds=round(time.monotonic() - t0))
    res = pd.DataFrame(rows_out)
    # score level by year (mean score over mask cells with a finite score) for the description of the level
    for sname, s in load_scores().items():
        for yr in YEARS:
            dd = days[years[days] == yr]
            m = mask[dd] & np.isfinite(s[dd])
            res.loc[(res.scores == sname) & (res.year == yr), 'score_mean'] = float(s[dd][m].mean()) if m.any() else np.nan
    res.to_csv(OUT / 'signal_by_year.csv', index=False)
    chk = res[(res.scores == 'ridge_fixed2021') & (res.year == '2022-2026')].iloc[0]
    check = dict(ridge_top_2022_2026=float(chk.top_mean), ridge_ic_2022_2026=float(chk.rank_ic),
                 matches_t5=bool(abs(chk.top_mean - T5_CHECK['top']) < 5e-5 and abs(chk.rank_ic - T5_CHECK['ic']) < 5e-4))
    json.dump(check, open(OUT / 'signals_check.json', 'w'), indent=1)
    log(**check)
    pd.set_option('display.width', 250)
    print(res.drop(columns=['ic_pos_share']).round(5).to_string(index=False))


# ----------------------------------------------------------------------------------------------- accounts
def pct_rank(s, mask):
    return pd.DataFrame(np.where(mask & np.isfinite(s), s, np.nan)).rank(axis=1, ascending=False, pct=True).to_numpy()


def rules(p, mask, s, events):
    """A: current main rule (score > 1%, exit out of top 30%); B: pure rank band (enter top 10%, exit out of top 50%)."""
    rank = pct_rank(s, mask)
    fin = np.isfinite(s)
    with np.errstate(invalid='ignore'):
        sig_main = fin & (np.nan_to_num(s, nan=-9) > ENTRY)
        sig_band = fin & (rank <= X_IN)
        exit_band = events | ~fin | ~(rank <= Y_OUT)
    return {'main': (sig_main, e.ExitRule(stop_loss=STOP, max_hold=CAP_DAYS, signal_exit=events | dj.weak_signal(s, mask, Q_MAIN))),
            'band': (sig_band, e.ExitRule(stop_loss=STOP, max_hold=CAP_DAYS, signal_exit=exit_band))}


def account(p, mask, sfill, sig, rule, cap, k, start, end, g=1.0, artifact_key=None):
    """Same bookkeeping as ashare_daily_judgment_20261008.account, with the entry signal passed in (as t1_bands)."""
    nav, tr, info = e.simulate(p, sig, sfill, rule, cap, k, start, end, universe=mask, entry_max_up=0.01, budget_fraction=g)
    st = e.stats(nav, tr, cap)
    dd = 1 - nav / nav.cummax()
    under = (dd > 1e-12).astype(int)
    led = info['ledger']
    years = (nav.index[-1] - nav.index[0]).days / 365.25

    def period(a, b):
        x, before = nav[a:b], nav[nav.index < a]
        return float(x.iloc[-1] / before.iloc[-1] - 1) if len(x) and len(before) else np.nan

    # Reporting completion added on 2026-10-10. No signal, execution or criterion changes.
    daily = led.copy()
    daily.index = pd.to_datetime(daily.date)

    def window(a, b):
        x, before = nav[a:b], nav[nav.index < a]
        if not len(x):
            return None
        base = float(before.iloc[-1]) if len(before) else float(cap)
        path = np.r_[base, x.to_numpy()]
        sub = daily.loc[a:b]
        amount = float((sub.buy_amount + sub.sell_amount).sum())
        return dict(start=str(x.index[0].date()), end=str(x.index[-1].date()),
                    ret=float(x.iloc[-1] / base - 1), mdd=float((1 - path / np.maximum.accumulate(path)).max()),
                    fees=float(sub.fees.sum()), turnover_amount=amount, turnover_x=amount / float(x.mean()))

    annual = {yr: window(f'{yr}-01-01', f'{yr}-12-31') for yr in YEARS}
    blocks = []
    left = pd.Timestamp(start)
    while left <= nav.index[-1]:
        right = left + pd.DateOffset(years=1)
        part = window(str(left.date()), str((right - pd.Timedelta(days=1)).date()))
        part['complete_12m'] = bool(right <= nav.index[-1] + pd.Timedelta(days=1))
        blocks.append(part)
        left = right
    if artifact_key is not None:
        daily.to_csv(OUT / f'ledger_{artifact_key}.csv', index=False)

    reasons = tr.reason.value_counts(normalize=True).round(3).to_dict() if len(tr) else {}
    row = dict(final=st['final'], ret=float(nav.iloc[-1] / cap - 1), cagr=st['cagr'], mdd=st['mdd'],
               ratio=st['cagr'] / st['mdd'] if st['mdd'] > 0 else np.nan,
               longest_underwater=int(under.groupby((under == 0).cumsum()).sum().max()),
               trades=st.get('trades', 0), win_rate=st.get('win_rate'), avg_trade=st.get('avg_return'),
               avg_win=st.get('avg_win'), avg_loss=st.get('avg_loss'), profit_factor=st.get('profit_factor'),
               payoff_ratio=(st.get('avg_win', np.nan) / abs(st['avg_loss']) if st.get('avg_loss', np.nan) < 0 else np.nan),
               avg_hold=st.get('avg_hold'), invested=float((1 - led.cash / led.nav).mean()), fees=info['fees'],
               turnover_x=info['turnover'] / float(nav.mean()), turnover_per_year=info['turnover'] / float(nav.mean()) / years,
               last12=period('2025-08-22', '2026-08-21'), prev12=period('2024-08-22', '2025-08-21'),
               last24=period('2024-08-22', '2026-08-21'),
               pending_sell_days=info['pending_sell_days'], rejected_buys=info['rejected_buys'],
               open_positions=info['open_positions'], data_end_positions=info['data_end_positions'],
               annual_detail=json.dumps(annual), blocks12=json.dumps(blocks),
               reasons=json.dumps(reasons), yearly=json.dumps(st['yearly']))
    return row, nav, tr


def yearly_ratio(nav, cap):
    """Per calendar year: net return / within-year max drawdown (drawdown measured from the previous year-end NAV)."""
    out = {}
    for yr in YEARS:
        x = nav[nav.index.year == int(yr)]
        if not len(x):
            continue
        before = nav[nav.index < f'{yr}-01-01']
        prev = float(before.iloc[-1]) if len(before) else float(cap)
        path = np.concatenate([[prev], x.to_numpy()])
        mdd = float((1 - path / np.maximum.accumulate(path)).max())
        ret = float(x.iloc[-1] / prev - 1)
        out[yr] = dict(ret=ret, mdd=mdd, ratio=ret / mdd if mdd > 0 else np.nan)
    return out


def ew_benchmark(p, mask, start, end):
    """Main-board equal weight with the previous day's eligibility mask, close to close, no costs."""
    c = p.a['close_qfq']
    d0 = int(np.searchsorted(p.dates, start))
    days = np.where((p.dates >= start) & (p.dates <= end))[0]
    r = np.empty(len(days))
    for i, t in enumerate(days):
        ok = mask[t - 1] & np.isfinite(c[t]) & np.isfinite(c[t - 1]) & (c[t - 1] > 0)
        r[i] = float(np.mean(c[t, ok] / c[t - 1, ok] - 1)) if ok.any() else 0.0
    nav = pd.Series(np.cumprod(1 + r), index=pd.to_datetime(p.dates[days]))
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    return dict(cagr=float(nav.iloc[-1] ** (1 / yrs) - 1), mdd=float((1 - nav / nav.cummax()).max()),
                ret=float(nav.iloc[-1] - 1), yearly=json.dumps({str(k.year): round(float(v), 4) for k, v in
                                                                 pd.concat([pd.Series([1.0], index=[nav.index[0]]), nav.resample('YE').last()])
                                                                 .pct_change().dropna().items()})), nav


def accounts():
    t0 = time.monotonic()
    p = e.load_panel(PANEL)
    mask = np.load(V1 / f'{PANEL}_mask.npy')
    events = dj.risk_events(p)
    d0 = int(np.searchsorted(p.dates, START))
    assert str(p.dates[d0]) == START
    OUT.mkdir(parents=True, exist_ok=True)
    bm, bm_nav = ew_benchmark(p, mask, START, END)
    bm_nav.to_csv(OUT / 'nav_ew_benchmark.csv')
    log(benchmark='ew_prev_mask', **bm)
    rows_out, shifts, ratios, navs = [], [], [], {}
    for sname, s in load_scores().items():
        sfill = np.nan_to_num(s, nan=-9).astype(np.float32)
        for rname, (sig, rule) in rules(p, mask, s, events).items():
            for cap, k, g in BOOKS:
                for slip in SLIPS:
                    e.SLIP = slip
                    try:
                        key = f'{rname}_{sname}_{cap}_g{g}_slip{slip}'
                        r, nav, tr = account(p, mask, sfill, sig, rule, cap, k, START, END, g, artifact_key=key)
                    finally:
                        e.SLIP = 0.001
                    rows_out.append(dict(scores=sname, rule=rname, capital=cap, g=g, slip=slip, **r))
                    log(**{x: v for x, v in rows_out[-1].items() if x not in ('yearly', 'reasons', 'annual_detail', 'blocks12')})
                    if (cap, g, slip) == PRIMARY:
                        navs[f'{rname}_{sname}'] = nav
                        tr.to_csv(OUT / f'trades_{rname}_{sname}.csv', index=False)
                        for yr, v in yearly_ratio(nav, cap).items():
                            ratios.append(dict(scores=sname, rule=rname, year=yr, **v))
            for sh in range(N_SHIFTS):
                r, _, _ = account(p, mask, sfill, sig, rule, 500000, 10, str(p.dates[d0 + sh]), END, 1.0)
                shifts.append(dict(scores=sname, rule=rname, shift=sh, start=str(p.dates[d0 + sh]), cagr=r['cagr'], mdd=r['mdd'],
                                   ratio=r['ratio'], trades=r['trades'], avg_hold=r['avg_hold'], turnover_x=r['turnover_x'], fees=r['fees']))
            log(scores=sname, rule=rname, seconds=round(time.monotonic() - t0))
        del s, sfill
    acc = pd.DataFrame(rows_out)
    acc.to_csv(OUT / 'accounts.csv', index=False)
    sh = pd.DataFrame(shifts)
    sh.to_csv(OUT / 'start_shifts.csv', index=False)
    summ = sh.groupby(['scores', 'rule']).agg(cagr_med=('cagr', 'median'), cagr_min=('cagr', 'min'), cagr_max=('cagr', 'max'),
                                              mdd_med=('mdd', 'median'), mdd_max=('mdd', 'max'), turnover_med=('turnover_x', 'median'),
                                              hold_med=('avg_hold', 'median'), fees_med=('fees', 'median')).reset_index()
    summ['ratio_med'] = summ.cagr_med / summ.mdd_med
    summ.to_csv(OUT / 'start_shift_summary.csv', index=False)
    pd.DataFrame(navs).to_csv(OUT / 'nav_primary.csv')
    rat = pd.DataFrame(ratios)
    rat.to_csv(OUT / 'yearly_ratio.csv', index=False)
    # ---- registered criterion: band rule primary account yearly ratio and signal-level yearly rank IC vs lgb_mse
    sig = pd.read_csv(OUT / 'signal_by_year.csv', dtype={'year': str})
    crit = {}
    for name in ('lgb_rank', 'lgb_bin'):
        years_win_ratio, years_win_ic, both = [], [], []
        for yr in YEARS:
            r_lab = rat[(rat.scores == name) & (rat.rule == 'band') & (rat.year == yr)].ratio
            r_ref = rat[(rat.scores == 'lgb_mse') & (rat.rule == 'band') & (rat.year == yr)].ratio
            i_lab = sig[(sig.scores == name) & (sig.year == yr)].rank_ic
            i_ref = sig[(sig.scores == 'lgb_mse') & (sig.year == yr)].rank_ic
            if not (len(r_lab) and len(r_ref) and len(i_lab) and len(i_ref)):
                continue
            wr, wi = bool(r_lab.iloc[0] > r_ref.iloc[0]), bool(i_lab.iloc[0] > i_ref.iloc[0])
            years_win_ratio.append(wr); years_win_ic.append(wi); both.append(wr and wi)
        crit[name] = dict(years=len(both), ratio_wins=int(sum(years_win_ratio)), ic_wins=int(sum(years_win_ic)), both_wins=int(sum(both)),
                          passes=bool(sum(years_win_ratio) >= 3 and sum(years_win_ic) >= 3))
    json.dump(dict(criterion=crit, benchmark=bm, seconds=round(time.monotonic() - t0)), open(OUT / 'criterion.json', 'w'), indent=1)
    log(criterion=crit)
    pd.set_option('display.width', 250)
    show = ['scores', 'rule', 'capital', 'g', 'slip', 'cagr', 'mdd', 'ratio', 'longest_underwater', 'trades', 'win_rate', 'avg_trade',
            'profit_factor', 'avg_hold', 'invested', 'fees', 'turnover_x', 'last12', 'prev12']
    print(acc[show].round(4).to_string(index=False))
    print(summ.round(4).to_string(index=False))
    print(rat.round(4).to_string(index=False))


if __name__ == '__main__':
    {'train': train, 'signals': signals, 'accounts': accounts}[sys.argv[1]]()
