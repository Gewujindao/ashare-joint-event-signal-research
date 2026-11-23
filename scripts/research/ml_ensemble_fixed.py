"""Fixed-model ensemble: daily cross-sectional z-score average of the two once-trained score sets
(fixed2021 Ridge and fixed2021 HGB). Report docs/model_comparison/ml_ensemble_fixed.md.
No training, no rolling, no tuning.

  python ml_ensemble_fixed.py signals    build ensemble (+ S1) scores; signal metrics by year; agreement diagnostics
  python ml_ensemble_fixed.py accounts   current main rule on Ridge / HGB / ensemble / S1: standard grid, 10 start
                                                  shifts, equal-weight benchmark (previous-day mask); criterion
  python ml_ensemble_fixed.py posthoc    POST-HOC: top-10 depth diagnostic, not part of the criterion

Ensemble (fixed in advance): per day t, among mask & finite(both scores):
  z_R = (s_R - mean_R) / std_R, z_H likewise (population std); z = (z_R + z_H) / 2;
  ensemble = mean_R + std_R * z / std(z)   -> same daily cross-sectional mean and std as the Ridge; only the ranking changes,
  so the 1% entry threshold keeps the Ridge's meaning (T5: the fixed-vs-yearly account gap is a score LEVEL effect).
S1 (one sensitivity row, never used to decide): HGB on the Ridge's scale = mean_R + std_R * z_H.
Signal metrics (per set, per year, eligible = mask & finite(y) & finite(s), days with n < 50 skipped):
  R2oos = 1 - sum((y - s)^2) / sum(y^2) (Gu-Kelly-Xiu, no demeaning); top / bottom decile mean label (gross), spread;
  rank IC = daily Spearman(s, y) averaged. Agreement: daily Spearman(s_R, s_H), top-10% overlap |top_R & top_H| / k.
Accounts: ashare_daily_judgment main rule (entry score > 1% ranked by score, exits risk_events | weak_signal(q=0.3),
  stop 0.08, max_hold 20, open <= ref x 1.01), engine unchanged. Memory < 1 GB; OMP_NUM_THREADS=4.
"""
import json
import os
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '4')
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import ashare_daily_judgment as dj
import ashare_engine as e

V1 = e.ROOT / 'output/research_once/ashare_ml_signals_v1_20261006'
V2 = e.ROOT / 'output/research_once/ashare_ml_signals_v2_20261007'
CA = e.ROOT / 'output/research_once/ashare_cost_aware_exit_20261009'
OUT = e.ROOT / 'output/research_once/ml_ensemble_fixed_20261009'
PANEL = 'mainboard_2014_2026'
START, END = '2022-01-04', '2026-08-21'
MIN_N, DECILE = 50, 0.1
YEARS = ('2022', '2023', '2024', '2025', '2026')
BOOKS = ((500000, 10, 1.0), (500000, 10, 0.5), (20000, 2, 1.0))
SLIPS = (0.001, 0.002)
N_SHIFTS = 10
PRIMARY = (500000, 1.0, 0.001)
BENCH_CAPITAL = 500000
COMPONENTS = ('ridge', 'hgb')
SETS = {'ridge': CA / 'scores_fixed2021_C_RIDGE_HR5.npy',
        'hgb': CA / 'scores_fixed2021_HGB_C_HR5.npy',
        'ensemble': OUT / 'scores_ensemble_fixed2021.npy',
        'hgb_on_ridge_scale': OUT / 'scores_hgb_on_ridge_scale_fixed2021.npy'}   # S1 sensitivity, not used to decide


def now():
    return time.strftime('%Y-%m-%d %H:%M:%S')


def build_ensemble(R, H, mask):
    """Ensemble and S1 arrays (float32, NaN outside mask & finite(both)); per-day construction stats."""
    D, N = R.shape
    ens = np.full((D, N), np.nan, np.float32)
    s1 = np.full((D, N), np.nan, np.float32)
    stats = []
    both = np.isfinite(R) & np.isfinite(H) & mask
    for t in np.where(both.any(axis=1))[0]:
        ok = both[t]
        r, h = R[t, ok].astype(np.float64), H[t, ok].astype(np.float64)
        mu_r, sd_r, mu_h, sd_h = r.mean(), r.std(), h.mean(), h.std()
        if sd_r <= 0 or sd_h <= 0 or ok.sum() < 2:
            continue
        z_r, z_h = (r - mu_r) / sd_r, (h - mu_h) / sd_h
        z = 0.5 * (z_r + z_h)
        sd_z = z.std()
        if sd_z <= 0:
            continue
        ens[t, ok] = (mu_r + sd_r * z / sd_z).astype(np.float32)
        s1[t, ok] = (mu_r + sd_r * z_h).astype(np.float32)
        stats.append(dict(t=int(t), n=int(ok.sum()), mean_r=mu_r, std_r=sd_r, mean_h=mu_h, std_h=sd_h, std_z=sd_z))
    return ens, s1, pd.DataFrame(stats)


def day_metrics(y, s):
    n = len(y)
    k = max(1, int(DECILE * n))
    order = np.argsort(-s, kind='stable')
    top, bot = order[:k], order[-k:]
    return dict(n=n, k=k, top=float(y[top].mean()), bottom=float(y[bot].mean()), all=float(y.mean()),
                ic=float(spearmanr(s, y).correlation), mean_s=float(s.mean()), std_s=float(s.std()))


def signals():
    t0 = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    print(json.dumps(dict(stage='signals', started=now())), flush=True)
    dates = e.load_panel(PANEL).dates
    years = pd.Series(dates).str[:4].to_numpy()
    mask = np.load(V1 / f'{PANEL}_mask.npy')
    y = np.load(V2 / f'{PANEL}_rawT5.npy')
    R, H = np.load(SETS['ridge']), np.load(SETS['hgb'])
    finR, finH = np.isfinite(R), np.isfinite(H)
    checks = dict(finite_cells_ridge=int(finR.sum()), finite_cells_hgb=int(finH.sum()), finite_cells_identical=bool(np.array_equal(finR, finH)),
                  finite_outside_mask=int((finR & ~mask).sum() + (finH & ~mask).sum()),
                  first_day_ridge=str(dates[np.where(finR.any(axis=1))[0][0]]), first_day_hgb=str(dates[np.where(finH.any(axis=1))[0][0]]))
    ens, s1, cons = build_ensemble(R, H, mask)
    np.save(SETS['ensemble'], ens)
    np.save(SETS['hgb_on_ridge_scale'], s1)
    cons['date'] = dates[cons.t.to_numpy()]
    cons.to_csv(OUT / 'ensemble_construction_daily.csv', index=False)
    # built-in check: the ensemble's daily mean/std equal the Ridge's (by construction)
    chk_days = cons.t.to_numpy()[::97]
    diffs = [(abs(np.nanmean(ens[t]) - np.nanmean(R[t])), abs(np.nanstd(ens[t]) - np.nanstd(R[t]))) for t in chk_days]
    checks['ensemble_mean_std_match_ridge_maxabs'] = [float(max(d[0] for d in diffs)), float(max(d[1] for d in diffs))]
    checks['ensemble_days'] = int(len(cons))
    print(json.dumps(dict(checks=checks, seconds=round(time.monotonic() - t0))), flush=True)

    scores = {'ridge': R, 'hgb': H, 'ensemble': ens, 'hgb_on_ridge_scale': s1}
    days = np.where((dates >= START) & (dates <= END))[0]
    daily_rows, r2_rows, agree_rows = [], [], []
    for sname, s in scores.items():
        elig = mask & np.isfinite(y) & np.isfinite(s)
        cand = mask & np.isfinite(s) & (np.nan_to_num(s, nan=-9) > dj.ENTRY)
        for yr in YEARS:
            d = days[years[days] == yr]
            m = elig[d]
            yy, ss = y[d][m].astype(np.float64), s[d][m].astype(np.float64)
            r2_rows.append(dict(scores=sname, year=yr, cells=int(m.sum()), r2oos=float(1 - np.sum((yy - ss) ** 2) / np.sum(yy ** 2)),
                                mean_y=float(yy.mean()), mean_s=float(ss.mean()), std_y=float(yy.std()), std_s=float(ss.std())))
        d = days
        m = elig[d]
        yy, ss = y[d][m].astype(np.float64), s[d][m].astype(np.float64)
        r2_rows.append(dict(scores=sname, year='2022-2026', cells=int(m.sum()), r2oos=float(1 - np.sum((yy - ss) ** 2) / np.sum(yy ** 2)),
                            mean_y=float(yy.mean()), mean_s=float(ss.mean()), std_y=float(yy.std()), std_s=float(ss.std())))
        for t in days:
            m = elig[t]
            n = int(m.sum())
            row = dict(scores=sname, date=dates[t], year=years[t], n=n, n_candidates=int(cand[t].sum()), skipped=n < MIN_N)
            if n >= MIN_N:
                row.update(day_metrics(y[t][m].astype(np.float64), s[t][m].astype(np.float64)))
            daily_rows.append(row)
        print(json.dumps(dict(scores=sname, seconds=round(time.monotonic() - t0))), flush=True)
    # agreement between the two components (all finite cells in mask, label not needed)
    both = mask & finR & finH
    for t in days:
        ok = both[t]
        n = int(ok.sum())
        if n < MIN_N:
            continue
        r, h = R[t, ok].astype(np.float64), H[t, ok].astype(np.float64)
        k = max(1, int(DECILE * n))
        top_r, top_h = set(np.argsort(-r, kind='stable')[:k]), set(np.argsort(-h, kind='stable')[:k])
        agree_rows.append(dict(date=dates[t], year=years[t], n=n, k=k, spearman=float(spearmanr(r, h).correlation),
                               pearson=float(np.corrcoef(r, h)[0, 1]), top10_overlap=len(top_r & top_h) / k,
                               cand_ridge=int((r > dj.ENTRY).sum()), cand_hgb=int((h > dj.ENTRY).sum())))
    daily = pd.DataFrame(daily_rows)
    daily.to_csv(OUT / 'daily_signal.csv', index=False)
    pd.DataFrame(r2_rows).to_csv(OUT / 'r2_by_year.csv', index=False)
    agree = pd.DataFrame(agree_rows)
    agree.to_csv(OUT / 'agreement_daily.csv', index=False)

    def agg(d):
        d = d[~d.skipped]
        return dict(days=int(len(d)), n_mean=float(d.n.mean()), top_mean=float(d.top.mean()), bottom_mean=float(d.bottom.mean()),
                    spread=float((d.top - d.bottom).mean()), all_mean=float(d['all'].mean()), rank_ic=float(d.ic.mean()),
                    ic_std=float(d.ic.std()), ic_t=float(d.ic.mean() / d.ic.std() * np.sqrt(len(d))), ic_pos_share=float((d.ic > 0).mean()),
                    cand_mean=float(d.n_candidates.mean()), no_cand_days=int((d.n_candidates == 0).sum()),
                    cand_ge_100_days=int((d.n_candidates >= 100).sum()))

    r2 = pd.DataFrame(r2_rows).set_index(['scores', 'year'])
    rows = []
    for sname in scores:
        d = daily[daily.scores == sname]
        for label, part in [(yr, d[d.year == yr]) for yr in YEARS] + [('2022-2026', d)]:
            rows.append(dict(scores=sname, year=label, r2oos=float(r2.loc[(sname, label), 'r2oos']), mean_s=float(r2.loc[(sname, label), 'mean_s']), **agg(part)))
    by_year = pd.DataFrame(rows)
    by_year.to_csv(OUT / 'signal_by_year.csv', index=False)
    arows = []
    for label, part in [(yr, agree[agree.year == yr]) for yr in YEARS] + [('2022-2026', agree)]:
        arows.append(dict(year=label, days=int(len(part)), spearman_mean=float(part.spearman.mean()), spearman_std=float(part.spearman.std()),
                          spearman_min=float(part.spearman.min()), pearson_mean=float(part.pearson.mean()),
                          top10_overlap_mean=float(part.top10_overlap.mean()), top10_overlap_std=float(part.top10_overlap.std()),
                          top10_overlap_min=float(part.top10_overlap.min()), cand_ridge_mean=float(part.cand_ridge.mean()),
                          cand_hgb_mean=float(part.cand_hgb.mean()), hgb_no_cand_days=int((part.cand_hgb == 0).sum()),
                          hgb_cand_ge_500_days=int((part.cand_hgb >= 500).sum()), ridge_no_cand_days=int((part.cand_ridge == 0).sum())))
    pd.DataFrame(arows).to_csv(OUT / 'agreement_by_year.csv', index=False)
    json.dump(dict(checks=checks, finished=now(), seconds=round(time.monotonic() - t0)), open(OUT / 'signals_run_info.json', 'w'), indent=1)
    pd.set_option('display.width', 250)
    print(by_year.round(5).to_string(index=False))
    print(pd.DataFrame(arows).round(4).to_string(index=False))
    print(json.dumps(dict(stage='signals', finished=now(), seconds=round(time.monotonic() - t0))), flush=True)


def account(p, mask, s, rule, cap, k, start, end, g=1.0):
    """Same bookkeeping as ashare_daily_judgment.account, also returning the NAV and the trades."""
    sfill = np.nan_to_num(s, nan=-9).astype(np.float32)
    sig = np.isfinite(s) & (sfill > dj.ENTRY)
    nav, tr, info = e.simulate(p, sig, sfill, rule, cap, k, start, end, universe=mask, entry_max_up=0.01, budget_fraction=g)
    st = e.stats(nav, tr, cap)
    dd = 1 - nav / nav.cummax()
    under = (dd > 1e-12).astype(int)
    led = info['ledger']
    years = (nav.index[-1] - nav.index[0]).days / 365.25

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
               turnover_x=info['turnover'] / float(nav.mean()), turnover_per_year=info['turnover'] / float(nav.mean()) / years,
               last12=period('2025-08-22', '2026-08-21'), prev12=period('2024-08-22', '2025-08-21'),
               pending_sell_days=info['pending_sell_days'], rejected_buys=info['rejected_buys'],
               reasons=json.dumps(reasons), yearly=json.dumps(st['yearly']))
    return row, nav, tr


def benchmark(p, mask, start, end, capital):
    """Main-board equal weight, gross, zero cost: day-t close-to-close returns weighted by the previous day's mask
    (no same-day look-ahead); NAV flat at `capital` on the start day (the accounts hold no position at the start close)."""
    r1 = pd.DataFrame(p.a['close_qfq']).pct_change(fill_method=None).to_numpy()
    m_lag = np.zeros_like(mask)
    m_lag[1:] = mask[:-1]
    with np.errstate(invalid='ignore'):
        ew = np.nan_to_num(np.nanmean(np.where(m_lag, r1, np.nan), axis=1), nan=0.0)
    days = np.where((p.dates >= start) & (p.dates <= end))[0]
    growth = np.concatenate([[1.0], np.cumprod(1.0 + ew[days[1:]].astype(np.float64))])
    nav = pd.Series(growth * capital, index=pd.to_datetime(p.dates[days]))
    st = e.stats(nav, pd.DataFrame(), capital)
    dd = 1 - nav / nav.cummax()
    under = (dd > 1e-12).astype(int)

    def period(a, b):
        x, before = nav[a:b], nav[nav.index < a]
        return float(x.iloc[-1] / before.iloc[-1] - 1) if len(x) and len(before) else np.nan

    row = dict(line='MAINBOARD_EQUAL_WEIGHT_GROSS_PREVDAY_MASK', capital=capital, final=st['final'], cagr=st['cagr'], mdd=st['mdd'],
               ratio=st['cagr'] / st['mdd'] if st['mdd'] > 0 else np.nan, longest_underwater=int(under.groupby((under == 0).cumsum()).sum().max()),
               last12=period('2025-08-22', '2026-08-21'), prev12=period('2024-08-22', '2025-08-21'), yearly=json.dumps(st['yearly']))
    return row, nav


def accounts():
    t0 = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    print(json.dumps(dict(stage='accounts', started=now())), flush=True)
    p = e.load_panel(PANEL)
    mask = np.load(V1 / f'{PANEL}_mask.npy')
    events = dj.risk_events(p)
    d0 = int(np.searchsorted(p.dates, START))
    assert str(p.dates[d0]) == START
    rows, shifts, navs, checks = [], [], {}, {}
    for sname, path in SETS.items():
        s = np.load(path)
        rule = e.ExitRule(stop_loss=dj.STOP, max_hold=dj.CAP_DAYS, signal_exit=events | dj.weak_signal(s, mask, 0.3))
        if sname == 'ridge':   # copy check: this file's account() must reproduce dj.account() on the baseline
            r_here, _, _ = account(p, mask, s, rule, 500000, 10, START, END, 1.0)
            r_dj = dj.account(p, mask, s, rule, 500000, 10, START, END, 1.0)
            checks['ridge_baseline_matches_dj_account'] = bool(abs(r_here['final'] - r_dj['final']) < 1e-6 and r_here['trades'] == r_dj['trades'])
            checks['ridge_baseline_final'] = r_dj['final']
            print(json.dumps(checks), flush=True)
        for cap, k, g in BOOKS:
            for slip in SLIPS:
                e.SLIP = slip
                try:
                    r, nav, tr = account(p, mask, s, rule, cap, k, START, END, g)
                finally:
                    e.SLIP = 0.001
                rows.append(dict(scores=sname, capital=cap, g=g, slip=slip, start=START, end=END, **r))
                print(json.dumps({x: (round(v, 4) if isinstance(v, float) else v) for x, v in rows[-1].items() if x != 'yearly'}), flush=True)
                if (cap, g, slip) == PRIMARY:
                    navs[sname] = nav
                    tr.to_csv(OUT / f'trades_primary_{sname}.csv', index=False)
        for sh in range(N_SHIFTS):
            r, _, _ = account(p, mask, s, rule, 500000, 10, str(p.dates[d0 + sh]), END, 1.0)
            shifts.append(dict(scores=sname, shift=sh, start=str(p.dates[d0 + sh]), cagr=r['cagr'], mdd=r['mdd'], ratio=r['ratio'],
                               trades=r['trades'], avg_hold=r['avg_hold'], turnover_x=r['turnover_x'], fees=r['fees'], final=r['final']))
        print(json.dumps(dict(scores=sname, seconds=round(time.monotonic() - t0))), flush=True)
        del s, rule
    brow, bnav = benchmark(p, mask, START, END, BENCH_CAPITAL)
    navs['benchmark_ew_prevmask'] = bnav
    pd.DataFrame([brow]).to_csv(OUT / 'benchmark.csv', index=False)
    print(json.dumps({x: (round(v, 4) if isinstance(v, float) else v) for x, v in brow.items()}), flush=True)
    pd.DataFrame(navs).to_csv(OUT / 'nav_primary.csv')
    acc = pd.DataFrame(rows)
    acc.to_csv(OUT / 'accounts.csv', index=False)
    sh = pd.DataFrame(shifts)
    sh.to_csv(OUT / 'start_shifts.csv', index=False)
    med = sh.groupby('scores').agg(cagr_med=('cagr', 'median'), cagr_min=('cagr', 'min'), cagr_max=('cagr', 'max'), mdd_med=('mdd', 'median'),
                                   mdd_max=('mdd', 'max'), ratio_min=('ratio', 'min'), ratio_max=('ratio', 'max'),
                                   turnover_med=('turnover_x', 'median'), hold_med=('avg_hold', 'median'), fees_med=('fees', 'median'))
    med['ratio_med'] = med.cagr_med / med.mdd_med
    med.reset_index().to_csv(OUT / 'start_shift_summary.csv', index=False)
    # ---- criterion (read on the primary account and on the yearly rank IC)
    prim = acc[(acc.capital == PRIMARY[0]) & (acc.g == PRIMARY[1]) & (acc.slip == PRIMARY[2])].set_index('scores')
    crit = dict(a_ratio={k: float(prim.loc[k, 'ratio']) for k in prim.index},
                a_pass=bool(prim.loc['ensemble', 'ratio'] > prim.loc['ridge', 'ratio'] and prim.loc['ensemble', 'ratio'] > prim.loc['hgb', 'ratio']),
                a_shift_median_ratio={k: float(med.loc[k, 'ratio_med']) for k in med.index},
                a_shift_median_pass=bool(med.loc['ensemble', 'ratio_med'] > med.loc['ridge', 'ratio_med'] and med.loc['ensemble', 'ratio_med'] > med.loc['hgb', 'ratio_med']))
    f_sig = OUT / 'signal_by_year.csv'
    if f_sig.exists():
        by = pd.read_csv(f_sig, dtype={'year': str}).set_index(['scores', 'year'])
        wins = {yr: bool(by.loc[('ensemble', yr), 'rank_ic'] > by.loc[('ridge', yr), 'rank_ic'] and by.loc[('ensemble', yr), 'rank_ic'] > by.loc[('hgb', yr), 'rank_ic'])
                for yr in YEARS}
        crit.update(b_ic_wins_by_year=wins, b_years_won=int(sum(wins.values())), b_pass=bool(sum(wins.values()) >= 3),
                    b_rank_ic={k: {yr: float(by.loc[(k, yr), 'rank_ic']) for yr in YEARS} for k in ('ridge', 'hgb', 'ensemble', 'hgb_on_ridge_scale')})
        crit['pass'] = bool(crit['a_pass'] and crit['b_pass'])
    json.dump(dict(criterion=crit, checks=checks, finished=now(), seconds=round(time.monotonic() - t0)), open(OUT / 'criterion.json', 'w'), indent=1, default=float)
    pd.set_option('display.width', 250)
    show = ['scores', 'capital', 'g', 'slip', 'cagr', 'mdd', 'ratio', 'longest_underwater', 'trades', 'win_rate', 'avg_trade', 'profit_factor', 'avg_hold',
            'invested', 'fees', 'turnover_x', 'last12', 'prev12']
    print(acc[show].round(4).to_string(index=False))
    print(med.round(4).to_string())
    print(json.dumps(crit, indent=1, default=float))
    print(json.dumps(dict(stage='accounts', finished=now(), seconds=round(time.monotonic() - t0))), flush=True)


def nw_t(x, lag=10):
    """t statistic of the mean of a daily series with Newey-West (Bartlett) long-run variance (labels overlap <= 5 sessions)."""
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return np.nan
    d = x - x.mean()
    var = float(d @ d) / n
    for l in range(1, min(lag, n - 1) + 1):
        var += 2 * (1 - l / (lag + 1)) * float(d[:-l] @ d[l:]) / n
    return float(x.mean() / np.sqrt(var / n)) if var > 0 else np.nan


def posthoc():
    """POST-HOC (not part of the criterion). Signal quality at the account's
    depth: each day the candidates are mask & finite(s) & s > 1% (with a matured label); picks = the top min(10, n) by score;
    mean gross label of the picks, by year, for ridge / ensemble / S1; the daily ensemble-minus-ridge difference with a
    Newey-West(10) t on common candidate days, and conditional on both picking exactly 10 (same depth, as in T5)."""
    t0 = time.monotonic()
    dates = e.load_panel(PANEL).dates
    years = pd.Series(dates).str[:4].to_numpy()
    mask = np.load(V1 / f'{PANEL}_mask.npy')
    y = np.load(V2 / f'{PANEL}_rawT5.npy')
    days = np.where((dates >= START) & (dates <= END))[0]
    picks = {}
    for sname in ('ridge', 'ensemble', 'hgb_on_ridge_scale'):
        s = np.load(SETS[sname])
        rows = []
        for t in days:
            ok = mask[t] & np.isfinite(y[t]) & np.isfinite(s[t])
            if ok.sum() < MIN_N:
                continue
            cand = ok & (s[t] > dj.ENTRY)
            idx = np.where(cand)[0]
            if len(idx) == 0:
                rows.append(dict(date=dates[t], year=years[t], n_cand=0, n_pick=0, pick_mean=np.nan, top30_mean=np.nan))
                continue
            order = idx[np.argsort(-s[t, idx], kind='stable')]
            top10 = order[:10]
            rows.append(dict(date=dates[t], year=years[t], n_cand=int(len(idx)), n_pick=int(len(top10)),
                             pick_mean=float(y[t, top10].mean()), top30_mean=float(y[t, order[:30]].mean())))
        picks[sname] = pd.DataFrame(rows).set_index('date')
        del s
    out = []
    for sname, d in picks.items():
        for label, part in [(yr, d[d.year == yr]) for yr in YEARS] + [('2022-2026', d)]:
            has = part[part.n_cand > 0]
            out.append(dict(scores=sname, year=label, days=int(len(part)), no_cand_days=int((part.n_cand == 0).sum()),
                            cand_mean=float(part.n_cand.mean()), pick_n_mean=float(has.n_pick.mean()) if len(has) else np.nan,
                            top10_mean=float(has.pick_mean.mean()) if len(has) else np.nan,
                            top30_mean=float(has.top30_mean.mean()) if len(has) else np.nan,
                            top10_net=float(has.pick_mean.mean() - 0.00272) if len(has) else np.nan))
    res = pd.DataFrame(out)
    res.to_csv(OUT / 'posthoc_top10_by_year.csv', index=False)
    comp = []
    a = picks['ridge']
    for other in ('ensemble', 'hgb_on_ridge_scale'):
        b = picks[other]
        common = a.index.intersection(b.index)
        aa, bb = a.loc[common], b.loc[common]
        both = (aa.n_cand > 0) & (bb.n_cand > 0)
        full = (aa.n_pick == 10) & (bb.n_pick == 10)
        for label in list(YEARS) + ['2022-2026']:
            sel = (aa.year == label) if label != '2022-2026' else pd.Series(True, index=common)
            diff = (bb.pick_mean - aa.pick_mean)[sel & both]
            diff_full = (bb.pick_mean - aa.pick_mean)[sel & full]
            comp.append(dict(pair=f'{other}-ridge', year=label, common_cand_days=int((sel & both).sum()), diff_pp=float(diff.mean() * 100),
                             nw_t=nw_t(diff), both_full10_days=int((sel & full).sum()), diff_full10_pp=float(diff_full.mean() * 100),
                             nw_t_full10=nw_t(diff_full)))
    comp = pd.DataFrame(comp)
    comp.to_csv(OUT / 'posthoc_top10_diff.csv', index=False)
    pd.set_option('display.width', 250)
    print(res.round(5).to_string(index=False))
    print(comp.round(4).to_string(index=False))
    print(json.dumps(dict(stage='posthoc', finished=now(), seconds=round(time.monotonic() - t0))), flush=True)


if __name__ == '__main__':
    {'signals': signals, 'accounts': accounts, 'posthoc': posthoc}[sys.argv[1]]()
