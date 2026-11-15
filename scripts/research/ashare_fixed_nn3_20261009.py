"""Fixed GKX-NN3 fair contest, trained once (criteria and verdict: results/model_comparison/nn3_criterion.json).

  python ashare_fixed_nn3_20261009.py train     stage 1: epoch choice on 2021 (models fit on labels matured before 2021-01-01);
                                                 stage 2: final fit on the fixed() rows (labels matured before 2022-01-01, last 600k),
                                                 seeds 0/1/2 each trained for the epoch count chosen in stage 1
  python ashare_fixed_nn3_20261009.py score     daily scores 2022-01-04 .. 2026-08-21: main = stage-2 ensemble; S1 = stage-1 ensemble
                                                 (no refit, GKX style); S2 = mean of the NN3 and fixed2021 Ridge predictions (GKX ensemble)
  python ashare_fixed_nn3_20261009.py signal    R2oos, top decile (gross / net of 0.272%), spread, rank IC by year vs fixed2021 Ridge, HGB
  python ashare_fixed_nn3_20261009.py account   current main rule (entry > 1%, q = 0.3, stop 8%, 20 sessions) on every score set vs the
                                                 Ridge baseline, HGB and the equal-weight benchmark (previous-day mask); pass criterion

Everything is fixed before running (pre-registration in the report): architecture 108-32-16-8-1 with batch norm and ReLU,
Adam 1e-3, batch 2048, L1 1e-5 on Linear weights, MSE on the +-20% clipped R5 trade label, early stopping on the 2021
top-10% mean label with patience 3 (max 30 epochs), 3 seeds averaged. No rolling refit, no grid, nothing tuned after 2021.
Resources: 4 threads, feature file memmapped (float16), rows float32, peak well under 1 GB.
"""
import copy
import json
import os
import sys
import time

os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
import numpy as np
import pandas as pd

import ashare_engine_20261006 as e
import ashare_ml_signals_20261006 as v1

V1 = e.ROOT / 'output/research_once/ashare_ml_signals_v1_20261006'
V2 = e.ROOT / 'output/research_once/ashare_ml_signals_v2_20261007'
CA = e.ROOT / 'output/research_once/ashare_cost_aware_exit_20261009'
OUT = e.ROOT / 'output/research_once/ashare_fixed_nn3_20261009'
PANEL = 'mainboard_2014_2026'
H, CUT, MAX_ROWS = 5, 2022, 600_000
HIDDEN, LR, BATCH, L1, PATIENCE, MAX_EPOCHS, SEEDS = (32, 16, 8), 1e-3, 2048, 1e-5, 3, 30, (0, 1, 2)
THREADS = 4
START, END = '2022-01-04', '2026-08-21'
MIN_N, DECILE, ROUND_TRIP = 50, 0.1, 0.00272          # T5 conventions; 0.272% = engine buy 0.111% + sell 0.161% at slip 0.001
ENTRY, Q = 0.01, 0.3                                   # current main rule
BOOKS = ((500000, 10, 1.0), (500000, 10, 0.5), (20000, 2, 1.0))
SLIPS = (0.001, 0.002)
SHIFTS = 10
SETS = {'NN3': OUT / 'scores_fixed2021_NN3_C_HR5.npy',
        'RIDGE': CA / 'scores_fixed2021_C_RIDGE_HR5.npy',
        'HGB': CA / 'scores_fixed2021_HGB_C_HR5.npy',
        'NN3_s1_norefit': OUT / 'scores_fixed2021_NN3s1_C_HR5.npy',
        'NN3_s2_meanRidge': OUT / 'scores_fixed2021_NN3blend_C_HR5.npy'}
FULL_GRID = ('NN3', 'RIDGE')                           # standard grid; the others get 500k full budget + start shifts only


def now():
    return time.strftime('%Y-%m-%d %H:%M:%S')


def free_gb():
    import ctypes

    class MS(ctypes.Structure):
        _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong), ('ullTotalPhys', ctypes.c_ulonglong),
                    ('ullAvailPhys', ctypes.c_ulonglong), ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong),
                    ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong), ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]
    m = MS()
    m.dwLength = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPhys / 2 ** 30


def wait_for_memory(min_gb=2.0, max_wait=1800):
    t0 = time.monotonic()
    while free_gb() < min_gb:
        if time.monotonic() - t0 > max_wait:
            raise RuntimeError(f'free memory below {min_gb} GB for {max_wait} s')
        print(json.dumps(dict(waiting=True, free_gb=round(free_gb(), 2), at=now())), flush=True)
        time.sleep(120)
    print(json.dumps(dict(free_gb=round(free_gb(), 2), at=now())), flush=True)


def dates_only():
    return np.load(e.CACHE / f'{PANEL}.npz', allow_pickle=False)['dates']


def build_rows(X, yclip, mask, sampled, a):
    """fixed() rows: sampled days whose labels matured before day a, most recent MAX_ROWS; standardized in place (float32)."""
    days = sampled[sampled + 1 + H < a]
    Xr, yr = v1.rows_for(X, yclip, mask, days)
    n_all = len(yr)
    Xr, yr = Xr[-MAX_ROWS:], yr[-MAX_ROWS:]
    mu = Xr.mean(0).astype(np.float32)
    sd = Xr.std(0).astype(np.float32)
    sd = np.where(sd < 1e-6, 1.0, sd).astype(np.float32)
    Xr -= mu
    Xr /= sd
    return Xr, yr.astype(np.float32), mu, sd, dict(days=int(len(days)), rows_all=int(n_all), rows=int(len(yr)),
                                                    first_day=str(days[0]), last_day=str(days[-1]))


def build_val(X, y, mask, a_val, a, mu, sd):
    """2021 validation days as in fixed(): mask & finite label, >= 50 stocks; labels unclipped."""
    import torch
    val = []
    for t in range(a_val, a - 1 - H):
        ok = mask[t] & np.isfinite(y[t])
        if ok.sum() < MIN_N:
            continue
        xt = (np.nan_to_num(X[:, t, ok].T.astype(np.float32), nan=0.5) - mu) / sd
        val.append((torch.from_numpy(np.ascontiguousarray(xt)), y[t, ok].astype(np.float64)))
    return val


def make_model(f):
    import torch.nn as nn
    layers, d = [], f
    for h in HIDDEN:
        layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU()]
        d = h
    layers.append(nn.Linear(d, 1))
    return nn.Sequential(*layers)


def l1_norm(model):
    import torch.nn as nn
    return sum(m.weight.abs().sum() for m in model if isinstance(m, nn.Linear))


def val_top10(model, val):
    import torch
    model.eval()
    tops = []
    with torch.no_grad():
        for xt, yt in val:
            pr = model(xt).squeeze(-1).numpy()
            k = max(1, int(DECILE * len(yt)))
            tops.append(float(np.mean(yt[np.argsort(-pr)[:k]])))
    return float(np.mean(tops))


def fit(Xr, yr, seed, val=None, epochs=MAX_EPOCHS):
    """One seed. With val: early stopping on the 2021 top-10% mean label (patience PATIENCE), returns the best-epoch weights.
    Without val: exactly `epochs` epochs, no validation (stage 2)."""
    import torch
    torch.manual_seed(seed)
    model = make_model(Xr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    Xt, yt = torch.from_numpy(Xr), torch.from_numpy(yr)
    n = len(yt)
    g = torch.Generator().manual_seed(seed)
    best, best_ep, best_state, bad, hist = -np.inf, 0, None, 0, []
    for ep in range(1, epochs + 1):
        t0 = time.monotonic()
        model.train()
        perm = torch.randperm(n, generator=g)
        tl = 0.0
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            if len(idx) < 16:                      # a final sliver is skipped (batch norm needs more than one row)
                break
            xb, yb = Xt[idx], yt[idx]
            opt.zero_grad()
            loss = torch.mean((model(xb).squeeze(-1) - yb) ** 2) + L1 * l1_norm(model)
            loss.backward()
            opt.step()
            tl += float(loss) * len(idx)
        rec = dict(epoch=ep, train_loss=tl / n, seconds=round(time.monotonic() - t0, 1))
        if val is not None:
            v = val_top10(model, val)
            rec['val_top10'] = v
            if v > best:
                best, best_ep, bad, best_state = v, ep, 0, copy.deepcopy(model.state_dict())
            else:
                bad += 1
        hist.append(rec)
        print(json.dumps(dict(seed=seed, stage=1 if val is not None else 2, **{k: (round(x, 7) if isinstance(x, float) else x) for k, x in rec.items()})), flush=True)
        if val is not None and bad >= PATIENCE:
            break
    if val is not None:
        model.load_state_dict(best_state)
    else:
        best_ep = epochs
    return model, best_ep, hist


def train():
    import torch
    torch.set_num_threads(THREADS)
    OUT.mkdir(parents=True, exist_ok=True)
    wait_for_memory()
    t_start, t0 = now(), time.monotonic()
    X = np.load(V2 / f'{PANEL}_XC.npy', mmap_mode='r')
    y = np.load(V2 / f'{PANEL}_rawT5.npy')
    mask = np.load(V1 / f'{PANEL}_mask.npy')
    dates = dates_only()
    D = len(dates)
    yclip = np.clip(y, -0.2, 0.2)
    sampled = np.arange(60, D, H)
    a_val, a = int(np.searchsorted(dates, f'{CUT - 1}-01-01')), int(np.searchsorted(dates, f'{CUT}-01-01'))
    # ---- stage 1: fit on labels matured before 2021, early-stop on 2021 (epoch choice only)
    X1, y1, mu1, sd1, info1 = build_rows(X, yclip, mask, sampled, a_val)
    val = build_val(X, y, mask, a_val, a, mu1, sd1)
    print(json.dumps(dict(stage=1, **info1, val_days=len(val), val_rows=int(sum(len(v[1]) for v in val)), free_gb=round(free_gb(), 2))), flush=True)
    stage1 = {}
    for sd_ in SEEDS:
        m, ep, hist = fit(X1, y1, sd_, val)
        stage1[sd_] = dict(state=m.state_dict(), best_epoch=ep, best_val=max(h['val_top10'] for h in hist), hist=hist)
        print(json.dumps(dict(stage=1, seed=sd_, best_epoch=ep, best_val_top10=round(stage1[sd_]['best_val'], 6),
                              epochs_run=len(hist), seconds=round(time.monotonic() - t0))), flush=True)
    del X1, y1, val
    t_stage1 = time.monotonic() - t0
    # ---- stage 2: the fixed() rows (labels matured before 2022, last 600k), each seed for its stage-1 epoch count
    X2, y2, mu2, sd2, info2 = build_rows(X, yclip, mask, sampled, a)
    print(json.dumps(dict(stage=2, **info2, free_gb=round(free_gb(), 2))), flush=True)
    stage2 = {}
    for sd_ in SEEDS:
        m, ep, hist = fit(X2, y2, sd_, None, stage1[sd_]['best_epoch'])
        stage2[sd_] = dict(state=m.state_dict(), epochs=ep, hist=hist)
    del X2, y2
    torch.save(dict(stage1={s: dict(state=v['state'], best_epoch=v['best_epoch']) for s, v in stage1.items()}, mu1=mu1, sd1=sd1,
                    stage2={s: dict(state=v['state'], epochs=v['epochs']) for s, v in stage2.items()}, mu2=mu2, sd2=sd2,
                    hidden=HIDDEN, features=X.shape[0]), OUT / 'nn3_models.pt')
    meta = dict(started=t_start, finished=now(), seconds_total=round(time.monotonic() - t0), seconds_stage1=round(t_stage1),
                stage1=dict(info1, seeds={s: dict(best_epoch=v['best_epoch'], best_val_top10=v['best_val'], hist=v['hist']) for s, v in stage1.items()}),
                stage2=dict(info2, seeds={s: dict(epochs=v['epochs'], hist=v['hist']) for s, v in stage2.items()}),
                config=dict(hidden=HIDDEN, lr=LR, batch=BATCH, l1=L1, patience=PATIENCE, max_epochs=MAX_EPOCHS, seeds=SEEDS, threads=THREADS,
                            label='rawT5 clipped +-0.2', loss='MSE + L1', val='2021 top-10% mean label (unclipped)', cut=CUT, max_rows=MAX_ROWS),
                free_gb_end=round(free_gb(), 2))
    json.dump(meta, open(OUT / 'nn3_fit.json', 'w'), indent=1, default=str)
    print(json.dumps(dict(done='train', seconds=meta['seconds_total'], epochs={s: v['best_epoch'] for s, v in stage1.items()})), flush=True)


def load_models(ck, key):
    models = []
    for s in SEEDS:
        m = make_model(ck['features'])
        m.load_state_dict(ck[key][s]['state'])
        m.eval()
        models.append(m)
    return models


def score():
    import torch
    torch.set_num_threads(THREADS)
    t0 = time.monotonic()
    ck = torch.load(OUT / 'nn3_models.pt', weights_only=False)
    X = np.load(V2 / f'{PANEL}_XC.npy', mmap_mode='r')
    mask = np.load(V1 / f'{PANEL}_mask.npy')
    dates = dates_only()
    D, N = mask.shape
    a = int(np.searchsorted(dates, START))
    b = int(np.searchsorted(dates, END, side='right'))
    outs = {}
    for name, key, mu, sd in (('main', 'stage2', ck['mu2'], ck['sd2']), ('s1', 'stage1', ck['mu1'], ck['sd1'])):
        models = load_models(ck, key)
        scores = np.full((D, N), np.nan, np.float32)
        with torch.no_grad():
            for t in range(a, b):
                ok = mask[t]
                if not ok.any():
                    continue
                xt = torch.from_numpy(np.ascontiguousarray((np.nan_to_num(X[:, t, ok].T.astype(np.float32), nan=0.5) - mu) / sd))
                scores[t, ok] = np.mean([m(xt).squeeze(-1).numpy() for m in models], axis=0)
        outs[name] = scores
    np.save(SETS['NN3'], outs['main'])
    np.save(SETS['NN3_s1_norefit'], outs['s1'])
    ridge = np.load(SETS['RIDGE'])
    blend = np.where(np.isfinite(outs['main']) & np.isfinite(ridge), 0.5 * (outs['main'] + ridge), np.nan).astype(np.float32)
    np.save(SETS['NN3_s2_meanRidge'], blend)
    cells = int(np.isfinite(outs['main']).sum())
    print(json.dumps(dict(done='score', days=b - a, first=str(dates[a]), last=str(dates[b - 1]), cells=cells,
                          ridge_cells=int(np.isfinite(ridge[a:b]).sum()), seconds=round(time.monotonic() - t0), at=now())), flush=True)


def spearman(a, b):
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    ra -= ra.mean()
    rb -= rb.mean()
    d = np.sqrt((ra @ ra) * (rb @ rb))
    return float(ra @ rb / d) if d > 0 else np.nan


def signal():
    """Signal metrics by calendar year and for the window, T5 definitions, every score set vs the fixed2021 Ridge."""
    t0 = time.monotonic()
    dates = dates_only()
    years = pd.Series(dates).str[:4].to_numpy()
    mask = np.load(V1 / f'{PANEL}_mask.npy')
    y = np.load(V2 / f'{PANEL}_rawT5.npy')
    days = np.where((dates >= START) & (dates <= END))[0]
    agg_rows, daily = [], {}
    for name, path in SETS.items():
        if not path.exists():
            continue
        s = np.load(path)
        elig = mask & np.isfinite(y) & np.isfinite(s)
        rows = []
        for t in days:
            m = elig[t]
            n = int(m.sum())
            ms = mask[t] & np.isfinite(s[t])
            st_all = s[t][ms].astype(np.float64)
            r = dict(date=dates[t], year=years[t], n=n, mean_s=float(st_all.mean()) if len(st_all) else np.nan,
                     cand_share=float((st_all > ENTRY).mean()) if len(st_all) else np.nan, n_cand=int((st_all > ENTRY).sum()))
            if n < MIN_N:
                r.update(skipped=True)
                rows.append(r)
                continue
            yt, st = y[t][m].astype(np.float64), s[t][m].astype(np.float64)
            k = max(1, int(DECILE * n))
            order = np.argsort(-st, kind='stable')
            r.update(skipped=False, k=k, top=float(yt[order[:k]].mean()), bottom=float(yt[order[-k:]].mean()), all=float(yt.mean()),
                     ic=spearman(st, yt), sse=float(((yt - st) ** 2).sum()), ss=float((yt ** 2).sum()))
            rows.append(r)
        df = pd.DataFrame(rows)
        df.to_csv(OUT / f'signal_daily_{name}.csv', index=False)
        daily[name] = df
        for label, part in [(yr, df[df.year == yr]) for yr in sorted(df.year.unique())] + [('2022-2026', df)]:
            d = part[~part.skipped]
            yy = years[days]
            sel = days if label == '2022-2026' else days[yy == label]
            m = elig[sel]
            yv, sv = y[sel][m].astype(np.float64), s[sel][m].astype(np.float64)
            agg_rows.append(dict(scores=name, year=label, days=int(len(part)), skipped=int(part.skipped.sum()), cells=int(m.sum()),
                                 r2oos=float(1 - np.sum((yv - sv) ** 2) / np.sum(yv ** 2)), n_mean=float(d.n.mean()),
                                 top=float(d.top.mean()), top_net=float(d.top.mean() - ROUND_TRIP), bottom=float(d.bottom.mean()),
                                 spread=float((d.top - d.bottom).mean()), all_mean=float(d['all'].mean()), top_excess=float((d.top - d['all']).mean()),
                                 rank_ic=float(d.ic.mean()), ic_t=float(d.ic.mean() / d.ic.std() * np.sqrt(len(d))), ic_pos_share=float((d.ic > 0).mean()),
                                 top_pos_share=float((d.top > 0).mean()), mean_s=float(part.mean_s.mean()), cand_share=float(part.cand_share.mean()),
                                 no_cand_days=int((part.n_cand == 0).sum())))
        print(json.dumps(dict(scores=name, days=len(df), skipped=int(df.skipped.sum()), seconds=round(time.monotonic() - t0))), flush=True)
    agg = pd.DataFrame(agg_rows)
    agg.to_csv(OUT / 'signal_by_year.csv', index=False)
    # ---- pairwise vs the Ridge baseline, by year: gaps and win flags
    pair_rows = []
    base = agg[agg.scores == 'RIDGE'].set_index('year')
    for name in agg.scores.unique():
        if name == 'RIDGE':
            continue
        a_ = agg[agg.scores == name].set_index('year')
        for yr in a_.index:
            b_ = base.loc[yr]
            r = a_.loc[yr]
            pair_rows.append(dict(scores=name, year=yr, r2_gap=r.r2oos - b_.r2oos, top_net_gap=r.top_net - b_.top_net, spread_gap=r.spread - b_.spread,
                                  ic_gap=r.rank_ic - b_.rank_ic, win_r2=bool(r.r2oos > b_.r2oos), win_top_net=bool(r.top_net > b_.top_net),
                                  win_spread=bool(r.spread > b_.spread), win_ic=bool(r.rank_ic > b_.rank_ic)))
    pair = pd.DataFrame(pair_rows)
    pair.to_csv(OUT / 'signal_vs_ridge.csv', index=False)
    wins = pair[pair.year != '2022-2026'].groupby('scores')[['win_r2', 'win_top_net', 'win_spread', 'win_ic']].sum().reset_index()
    wins['years'] = int((pair.year != '2022-2026').sum() / max(1, pair.scores.nunique()))
    wins.to_csv(OUT / 'signal_year_wins_vs_ridge.csv', index=False)
    pd.set_option('display.width', 250)
    print('\n== signal_by_year ==')
    print(agg.round(5).to_string(index=False))
    print('\n== signal_vs_ridge ==')
    print(pair.round(5).to_string(index=False))
    print('\n== year wins vs ridge ==')
    print(wins.to_string(index=False))
    print(json.dumps(dict(done='signal', seconds=round(time.monotonic() - t0), at=now())))


def ew_benchmark(p, mask, dates):
    """Equal-weight mainboard, weights = previous-day eligible mask (the same-day mask looks ahead), close-to-close qfq, gross."""
    c = p.a['close_qfq']
    d0 = int(np.searchsorted(dates, START))
    d1 = int(np.searchsorted(dates, END, side='right')) - 1
    rets = []
    for t in range(d0 + 1, d1 + 1):
        with np.errstate(invalid='ignore', divide='ignore'):
            r = c[t] / c[t - 1] - 1
        ok = mask[t - 1] & np.isfinite(r)
        rets.append(float(r[ok].mean()) if ok.any() else 0.0)
    nav = pd.Series(np.concatenate([[1.0], np.cumprod(1 + np.array(rets))]), index=pd.to_datetime(dates[d0:d1 + 1]))
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    dd = 1 - nav / nav.cummax()
    under = (dd > 1e-12).astype(int)
    yearly = nav.resample('YE').last()
    yearly = pd.concat([pd.Series([1.0], index=[nav.index[0]]), yearly]).pct_change().dropna()
    return dict(cagr=float(nav.iloc[-1] ** (1 / years) - 1), mdd=float(dd.max()), ret=float(nav.iloc[-1] - 1),
                longest_underwater=int(under.groupby((under == 0).cumsum()).sum().max()),
                last12=float(nav['2025-08-22':'2026-08-21'].iloc[-1] / nav[nav.index < '2025-08-22'].iloc[-1] - 1),
                prev12=float(nav['2024-08-22':'2025-08-21'].iloc[-1] / nav[nav.index < '2024-08-22'].iloc[-1] - 1),
                yearly=json.dumps({str(d.year): round(float(v), 4) for d, v in yearly.items()})), nav


def account():
    import ashare_cost_aware_exit_20261009 as ca
    import ashare_daily_judgment_20261008 as dj
    t0 = time.monotonic()
    p = e.load_panel(PANEL)
    mask = np.load(V1 / f'{PANEL}_mask.npy')
    dates = p.dates
    d0 = int(np.searchsorted(dates, START))
    rows, shifts, navs = [], [], {}
    for name, path in SETS.items():
        if not path.exists():
            continue
        s = np.load(path)
        rule = dj.daily_rule(p, mask, s, Q)
        full = name in FULL_GRID
        for cap, k, g in (BOOKS if full else BOOKS[:1]):
            for slip in (SLIPS if full else SLIPS[:1]):
                e.SLIP = slip
                r, daily, tr = ca.account(p, mask, s, rule, None, cap, k, START, END, g)
                e.SLIP = 0.001
                rows.append(dict(scores=name, capital=cap, g=g, slip=slip, **r))
                print(json.dumps({x: (round(v, 4) if isinstance(v, float) else v) for x, v in rows[-1].items() if x != 'yearly'}), flush=True)
                if cap == 500000 and g == 1.0 and slip == 0.001:
                    navs[name] = daily.nav
                    tr.to_csv(OUT / f'trades_{name}_500k_full.csv', index=False)
        for sh in range(SHIFTS):
            r, _, _ = ca.account(p, mask, s, rule, None, 500000, 10, str(dates[d0 + sh]), END)
            shifts.append(dict(scores=name, shift=sh, start=str(dates[d0 + sh]), cagr=r['cagr'], mdd=r['mdd'], ratio=r['ratio'],
                               trades=r['trades'], turnover_x=r['turnover_x'], fees=r['fees']))
        print(json.dumps(dict(scores=name, seconds=round(time.monotonic() - t0))), flush=True)
    ew, ew_nav = ew_benchmark(p, mask, dates)
    rows.append(dict(scores='EW_prevmask', capital=np.nan, g=np.nan, slip=0.0, **ew))
    navs['EW_prevmask'] = ew_nav
    acc = pd.DataFrame(rows)
    acc.to_csv(OUT / 'accounts.csv', index=False)
    sh = pd.DataFrame(shifts)
    sh.to_csv(OUT / 'start_shifts.csv', index=False)
    summ = sh.groupby('scores').agg(cagr_med=('cagr', 'median'), cagr_min=('cagr', 'min'), cagr_max=('cagr', 'max'),
                                    mdd_med=('mdd', 'median'), mdd_max=('mdd', 'max'), ratio_med_of_ratios=('ratio', 'median'),
                                    turnover=('turnover_x', 'median'), fees=('fees', 'median')).reset_index()
    summ['ratio_med'] = summ.cagr_med / summ.mdd_med
    summ.to_csv(OUT / 'start_shift_summary.csv', index=False)
    pd.DataFrame(navs).to_csv(OUT / 'navs_500k_full.csv')
    # ---- pre-registered pass criterion: (A) shift-median CAGR/MDD of NN3 > RIDGE; (B) rank IC NN3 > RIDGE in >= 3 of 5 years
    crit = dict(evaluated_at=now())
    sm = summ.set_index('scores')
    if 'NN3' in sm.index and 'RIDGE' in sm.index:
        crit.update(A_ratio_nn3=float(sm.loc['NN3', 'ratio_med']), A_ratio_ridge=float(sm.loc['RIDGE', 'ratio_med']),
                    A_pass=bool(sm.loc['NN3', 'ratio_med'] > sm.loc['RIDGE', 'ratio_med']))
    wins_file = OUT / 'signal_year_wins_vs_ridge.csv'
    if wins_file.exists():
        w = pd.read_csv(wins_file).set_index('scores')
        if 'NN3' in w.index:
            crit.update(B_ic_years_won=int(w.loc['NN3', 'win_ic']), B_years=int(w.loc['NN3', 'years']), B_pass=bool(w.loc['NN3', 'win_ic'] >= 3),
                        info_top_net_years_won=int(w.loc['NN3', 'win_top_net']), info_r2_years_won=int(w.loc['NN3', 'win_r2']))
    crit['PASS'] = bool(crit.get('A_pass', False) and crit.get('B_pass', False))
    json.dump(crit, open(OUT / 'criterion.json', 'w'), indent=1)
    pd.set_option('display.width', 250)
    print('\n== accounts ==')
    print(acc.drop(columns=['yearly', 'reasons']).round(4).to_string(index=False))
    print('\n== start shift summary ==')
    print(summ.round(4).to_string(index=False))
    print('\n== criterion ==')
    print(json.dumps(crit, indent=1))
    print(json.dumps(dict(done='account', seconds=round(time.monotonic() - t0), at=now())))


if __name__ == '__main__':
    {'train': train, 'score': score, 'signal': signal, 'account': account}[sys.argv[1]]()
