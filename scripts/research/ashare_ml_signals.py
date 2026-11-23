"""A-share multi-signal model.

  python ashare_ml_signals.py features <panel>        build ranked features + labels on disk
  python ashare_ml_signals.py fit <panel> RIDGE|HGB|MLP <H>  yearly walk-forward fit, daily scores
  python ashare_ml_signals.py trade <panel> <H>            engine backtest of the scores

Features are daily cross-sectional percentile ranks among eligible stocks (market features raw).
Label: open(t+1) -> open(t+6) return minus the same-day eligible mean. Training uses every 5th
day; each year from 2018 is refit on labels that ended before 1 January; hyperparameters are
chosen on the last pre-year year by the mean label of the top 10% predictions. All families see
the same rows (capped at the most recent 600k) and the same features.
"""
import json
import os
import sys
import time
import warnings

os.environ.setdefault('OMP_NUM_THREADS', '4')
import numpy as np
import pandas as pd

import ashare_engine as e
import ashare_rules_lib as L

warnings.filterwarnings('ignore')
OUT = e.ROOT / 'output/research_once/ashare_ml_signals_v1_20261006'
HS = (5, 10, 20)
MAX_ROWS = 300_000
YEARS = range(2018, 2027)


def features(panel):
    p = e.load_panel(panel)
    c, v = p.a['close_qfq'], p.a['volume_shares']
    o, hi, lo = p.a['open_qfq'], p.a['high_qfq'], p.a['low_qfq']
    cr, pr, orw = p.a['close_raw'], p.a['preclose_raw'], p.a['open_raw']
    D, N = c.shape
    uni = L.base_universe(p) & p.tradable & ~p.st
    nxt = np.zeros((D, N), bool)
    nxt[:-1] = p.tradable[1:] & np.isfinite(orw[1:]) & (orw[1:] < pr[1:] * (1 + p.limit[1:]) - 0.005)
    labels = {}
    for H in HS:
        fwd = np.full((D, N), np.nan, np.float32)
        fwd[:D - 1 - H] = p.a['open_qfq'][1 + H:] / p.a['open_qfq'][1:D - H] - 1
        lab_ok = uni & nxt & np.isfinite(fwd)
        with np.errstate(invalid='ignore'):
            lab = (fwd - np.nanmean(np.where(lab_ok, fwd, np.nan), axis=1, keepdims=True)).astype(np.float32)
        lab[~lab_ok] = np.nan
        labels[H] = lab
    with np.errstate(invalid='ignore', divide='ignore'):
        lr = np.log(c / np.roll(c, 1, axis=0)); lr[0] = np.nan
        lu = (cr >= pr * (1 + p.limit) - 0.005).astype(np.float32)
        ld = (cr <= pr * (1 - p.limit) + 0.005).astype(np.float32)
    vma20 = L.rolling_mean_prev(v, 20)

    def ret(k):
        x = c / np.roll(c, k, axis=0) - 1; x[:k] = np.nan; return x

    def range_pos(k):
        mx = pd.DataFrame(c).rolling(k, min_periods=k).max().to_numpy()
        mn = pd.DataFrame(c).rolling(k, min_periods=k).min().to_numpy()
        return (c - mn) / (mx - mn)

    def vol(k):
        return pd.DataFrame(lr).rolling(k, min_periods=k).std().to_numpy()

    makers = {**{f'ret{k}': (lambda k=k: ret(k)) for k in (1, 3, 5, 10, 20, 60)},
              **{f'dist_ma{k}': (lambda k=k: c / L.rolling_mean(c, k) - 1) for k in (5, 10, 20, 60)},
              **{f'range_pos{k}': (lambda k=k: range_pos(k)) for k in (20, 60)},
              **{f'vol{k}': (lambda k=k: vol(k)) for k in (5, 20, 60)},
              'vol_ratio': lambda: vol(5) / vol(20),
              'vr': lambda: v / vma20,
              'v5_v20': lambda: L.rolling_mean(v, 5) / vma20,
              'turnover': lambda: p.a['turnover_pct'],
              'turnover20': lambda: L.rolling_mean(p.a['turnover_pct'], 20),
              'log_amount': lambda: np.log(np.maximum(p.a['amount_cny'], 1)),
              'log_price': lambda: np.log(cr),
              'lu20': lambda: pd.DataFrame(lu).rolling(20, min_periods=1).sum().to_numpy(),
              'ld20': lambda: pd.DataFrame(ld).rolling(20, min_periods=1).sum().to_numpy(),
              'lu_today': lambda: lu, 'ld_today': lambda: ld,
              'body': lambda: c / o - 1,
              'amplitude': lambda: (hi - lo) / np.roll(c, 1, axis=0),
              'upper_shadow': lambda: (hi - np.maximum(o, c)) / c,
              'lower_shadow': lambda: (np.minimum(o, c) - lo) / c,
              'gap': lambda: orw / pr - 1}
    names = list(makers)
    F = len(names) + 3
    OUT.mkdir(parents=True, exist_ok=True)
    mm = np.lib.format.open_memmap(OUT / f'{panel}_X.npy', mode='w+', dtype=np.float16, shape=(F, D, N))
    for i, k in enumerate(names):  # one feature in memory at a time
        with np.errstate(invalid='ignore', divide='ignore'):
            x = np.where(uni, makers[k](), np.nan)
        mm[i] = pd.DataFrame(x).rank(axis=1, pct=True).to_numpy(np.float16)
        del x
    r5 = np.where(uni, np.roll(c, 0, axis=0) / np.roll(c, 5, axis=0) - 1, np.nan)
    r20 = np.where(uni, c / np.roll(c, 20, axis=0) - 1, np.nan)
    upfrac = np.where(uni, (c > np.roll(c, 1, axis=0)).astype(np.float32), np.nan)
    for i, x in enumerate((r5, r20, upfrac)):
        m = np.nanmean(x, axis=1, keepdims=True)
        mm[len(names) + i] = np.broadcast_to(m, (D, N)).astype(np.float16)
    mm.flush()
    for H, lab in labels.items():
        np.save(OUT / f'{panel}_y{H}.npy', lab)
    np.save(OUT / f'{panel}_mask.npy', uni)
    json.dump(dict(features=names + ['mkt_r5', 'mkt_r20', 'mkt_upfrac'], horizons=list(HS)), open(OUT / f'{panel}_meta.json', 'w'))
    print(json.dumps(dict(features=F, days=D, stocks=N, label_rows={H: int(np.isfinite(l).sum()) for H, l in labels.items()})))


def candidates(fam):
    if fam == 'RIDGE':
        return [dict(alpha=a) for a in (1, 10, 100, 1e3, 1e4, 1e5)]
    if fam == 'HGB':
        return [dict(learning_rate=.05, max_iter=100, max_leaf_nodes=15, min_samples_leaf=100, l2_regularization=1),
                dict(learning_rate=.05, max_iter=300, max_leaf_nodes=31, min_samples_leaf=100, l2_regularization=1),
                dict(learning_rate=.02, max_iter=600, max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=10),
                dict(learning_rate=.1, max_iter=300, max_leaf_nodes=63, min_samples_leaf=20, l2_regularization=0),
                dict(learning_rate=.05, max_iter=600, max_leaf_nodes=127, min_samples_leaf=20, l2_regularization=1),
                dict(learning_rate=.02, max_iter=300, max_leaf_nodes=15, min_samples_leaf=500, l2_regularization=100)]
    return [dict(hidden=(32,), alpha=1e-3, lr=1e-3), dict(hidden=(64, 32), alpha=1e-3, lr=1e-3),
            dict(hidden=(128, 64), alpha=1e-5, lr=1e-3), dict(hidden=(128, 64), alpha=1e-1, lr=3e-4),
            dict(hidden=(256, 128, 64), alpha=1e-3, lr=3e-4), dict(hidden=(256, 128, 64), alpha=1e-1, lr=1e-3)]


def model(fam, c, seed=0):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.neural_network import MLPRegressor
    if fam == 'RIDGE':
        return Ridge(alpha=c['alpha'])
    if fam == 'HGB':
        return HistGradientBoostingRegressor(early_stopping=False, random_state=seed, **c)
    return MLPRegressor(hidden_layer_sizes=c['hidden'], alpha=c['alpha'], learning_rate_init=c['lr'], max_iter=15,
                        batch_size=1024, early_stopping=False, tol=0, n_iter_no_change=10 ** 9, random_state=seed)


def rows_for(X, y, mask, day_idx):
    Xs, ys = [], []
    for t in day_idx:
        ok = mask[t] & np.isfinite(y[t])
        if ok.any():
            Xs.append(np.nan_to_num(X[:, t, ok].T.astype(np.float32), nan=0.5))
            ys.append(y[t, ok])
    if not Xs:
        return np.zeros((0, X.shape[0]), np.float32), np.zeros(0, np.float32)
    return np.vstack(Xs), np.concatenate(ys)


def fit(panel, fam, H):
    t0 = time.monotonic()
    X = np.load(OUT / f'{panel}_X.npy', mmap_mode='r')
    y = np.load(OUT / f'{panel}_y{H}.npy')
    mask = np.load(OUT / f'{panel}_mask.npy')
    dates = e.load_panel(panel).dates
    D, N = y.shape
    yclip = np.clip(y, -0.2, 0.2)
    sampled = np.arange(60, D, H)
    scores = np.full((D, N), np.nan, np.float32)
    log = []
    seeds = (0, 1, 2) if fam == 'MLP' else (0,)

    def first_idx(year):
        return int(np.searchsorted(dates, f'{year}-01-01'))

    for year in YEARS:
        a, a_prev, b = first_idx(year), first_idx(year - 1), first_idx(year + 1)
        if a >= D:
            break
        tr_inner = sampled[sampled + 1 + H < a_prev][-MAX_ROWS // 2000:]
        Xtr, ytr = rows_for(X, yclip, mask, tr_inner)
        Xtr, ytr = Xtr[-MAX_ROWS:], ytr[-MAX_ROWS:]
        val_days = np.arange(a_prev, a - 1 - H)
        best, best_score, cand_scores = None, -np.inf, []
        for i, c in enumerate(candidates(fam)):
            m = model(fam, c).fit(Xtr, ytr)
            tops = []
            for t in val_days:
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
        tr_full = sampled[sampled + 1 + H < a]
        Xf, yf = rows_for(X, yclip, mask, tr_full)
        Xf, yf = Xf[-MAX_ROWS:], yf[-MAX_ROWS:]
        ms = [model(fam, candidates(fam)[best], s).fit(Xf, yf) for s in seeds]
        for t in range(a, min(b, D)):
            ok = mask[t]
            if ok.any():
                xt = np.nan_to_num(X[:, t, ok].T.astype(np.float32), nan=0.5)
                scores[t, ok] = np.mean([m.predict(xt) for m in ms], axis=0)
        log.append(dict(year=year, chosen=best, val_top10=cand_scores, train_rows=int(len(yf))))
        print(json.dumps(dict(fam=fam, H=H, year=year, chosen=best, best_val_top10=round(best_score, 5), rows=len(yf),
                              seconds=round(time.monotonic() - t0))), flush=True)
    np.save(OUT / f'{panel}_scores_{fam}_H{H}.npy', scores)
    json.dump(log, open(OUT / f'{panel}_fit_{fam}_H{H}.json', 'w'), indent=1)


def trade(panel, H):
    p = e.load_panel(panel)
    y = np.load(OUT / f'{panel}_y{H}.npy')
    mask = np.load(OUT / f'{panel}_mask.npy')
    rows = []
    start, end = '2018-01-02', str(p.dates[-1])
    for fam in ('RIDGE', 'HGB', 'MLP'):
        f = OUT / f'{panel}_scores_{fam}_H{H}.npy'
        if not f.exists():
            continue
        s = np.load(f)
        # signal quality: daily mean label of the top 10% by score
        tops, bots = [], []
        for t in np.where((p.dates >= start))[0]:
            ok = mask[t] & np.isfinite(y[t]) & np.isfinite(s[t])
            if ok.sum() < 50:
                continue
            order = np.argsort(-s[t, ok]); k = max(1, int(0.1 * ok.sum()))
            tops.append(np.mean(y[t, ok][order[:k]])); bots.append(np.mean(y[t, ok][order[-k:]]))
        sig = np.isfinite(s)
        for cap, slots in ((500000, 10), (20000, 2)):
            nav, tr, info = e.simulate(p, sig, np.nan_to_num(s, nan=-9).astype(np.float32), e.ExitRule(max_hold=H), cap, slots,
                                       start, end, universe=mask)
            st = e.stats(nav, tr, cap)
            rows.append(dict(model=fam, H=H, capital=cap, top10_excess=float(np.mean(tops)), bottom10_excess=float(np.mean(bots)),
                             **{k: v for k, v in st.items() if k != 'yearly'}, yearly=json.dumps(st['yearly'])))
    res = pd.DataFrame(rows)
    res.to_csv(OUT / f'{panel}_trade_H{H}.csv', index=False)
    pd.set_option('display.width', 250)
    print(res.drop(columns='yearly').round(4).to_string(index=False))
    for _, r in res.iterrows():
        print(r.model, r.capital, r.yearly)


if __name__ == '__main__':
    cmd, panel = sys.argv[1], sys.argv[2]
    if cmd == 'features':
        features(panel)
    elif cmd == 'fit':
        fit(panel, sys.argv[3], int(sys.argv[4]))
    else:
        trade(panel, int(sys.argv[3]))
