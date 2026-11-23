# feature_rank_ic.py — reads: tradable, close_qfq, open_qfq, turnover_pct, high_raw, low_raw (training rows).
# Eight generic features known at the close of t: ret1, ret5, ret20, abnormal turnover, 20-day vol, 20-day max
# daily return, distance to 60-day high (close based), amplitude (high/low - 1).
# Targets: open(t+1)->open(t+1+L), L = 1, 5, 20.
# (a) cross-sectional Spearman rank IC per day -> mean and t over days;
# (b) pooled Pearson IC on total return (feature winsorised 1/99%, target clipped 0.5/99.5%), date-clustered t.
# Both by ALL, three sub-periods and ex-2015.
from common import *

start('q5_ic')
d = load('tradable', 'close_qfq', 'open_qfq', 'turnover_pct', 'high_raw', 'low_raw')
tr, yrs = d['tradable'], d['years']
T, N = tr.shape
seas = seasoned(tr, ~np.isnan(d['close_qfq']))
Cff = ffill(np.where(tr, d['close_qfq'], np.nan).astype(np.float32))
OQ = np.where(tr, d['open_qfq'], np.nan).astype(np.float32)
feats = {}
r1 = np.full((T, N), np.nan, np.float32)
r1[1:] = Cff[1:] / Cff[:-1] - 1
feats['ret1'] = r1
for k in (5, 20):
    x = np.full((T, N), np.nan, np.float32)
    x[k:] = Cff[k:] / Cff[:-k] - 1
    feats['ret%d' % k] = x
feats['abn_turn'] = abn_turnover(d['turnover_pct'], tr)
del d['turnover_pct']
mx = np.full((T, N), -np.inf, np.float32)
s1 = np.zeros((T, N), np.float32)
s2 = np.zeros((T, N), np.float32)
for j in range(20):
    sh = np.full((T, N), np.nan, np.float32)
    sh[j:] = r1[:T - j]
    mx = np.fmax(mx, sh)
    s1 += sh
    s2 += sh * sh
feats['vol20'] = np.sqrt(np.maximum(s2 / 20 - (s1 / 20) ** 2, 0))
mx[~np.isfinite(s1)] = np.nan
feats['max20'] = mx
del s1, s2, sh
hi = Cff.copy()
for j in range(1, 60):
    hi[j:] = np.fmax(hi[j:], Cff[:T - j])
feats['dist_hi60'] = Cff / hi - 1
del hi
feats['amplitude'] = np.where(tr, d['high_raw'] / d['low_raw'] - 1, np.nan).astype(np.float32)
del d['high_raw'], d['low_raw'], d['close_qfq'], d['open_qfq']
targs = {}
for L in (1, 5, 20):
    F = np.full((T, N), np.nan, np.float32)
    F[:T - 1 - L] = OQ[1 + L:] / OQ[1:T - L] - 1
    targs[L] = F
per = periods(yrs)


def rk(v):
    r = np.empty(v.size)
    r[np.argsort(v)] = np.arange(v.size)
    return r


names = list(feats)
ic = {(f, L): np.full(T, np.nan) for f in names for L in targs}
for t in range(T):
    st = seas[t]
    if st.sum() < 50:
        continue
    for L, F in targs.items():
        y = F[t]
        vy = st & np.isfinite(y)
        if vy.sum() < 50:
            continue
        for f in names:
            x = feats[f][t]
            v = vy & np.isfinite(x)
            if v.sum() < 50:
                continue
            ic[(f, L)][t] = np.corrcoef(rk(x[v]), rk(y[v]))[0, 1]
P('TABLE A  cross-sectional rank IC: feature | L | for ALL,P1,P2,P3,EX2015: mean IC (t over days; overlapping for L>1)')
for f in names:
    for L in targs:
        s = ic[(f, L)]
        row = []
        for pn, pm in per.items():
            v = s[pm & np.isfinite(s)]
            row.append('%.4f (%.1f)' % (v.mean(), v.mean() / v.std(ddof=1) * sqrt(v.size)))
        P('%s | %d | %s' % (f, L, '  '.join(row)))
P('\nTABLE B  pooled IC on total return: feature | L | for ALL,P1,P2,P3,EX2015: IC (date-clustered t)')
for f in names:
    x = feats[f]
    for L, F in targs.items():
        M = seas & np.isfinite(x) & np.isfinite(F)
        xv = x[M].astype(np.float64)
        yv = F[M].astype(np.float64)
        tt = np.nonzero(M)[0]
        lo, hi_ = np.percentile(xv, [1, 99])
        xv = np.clip(xv, lo, hi_)
        lo, hi_ = np.percentile(yv, [0.5, 99.5])
        yv = np.clip(yv, lo, hi_)
        row = []
        for pn, pm in per.items():
            sel = pm[tt]
            a = xv[sel]
            b = yv[sel]
            a = (a - a.mean()) / a.std()
            b = (b - b.mean()) / b.std()
            z = a * b
            icv = z.mean()
            S = np.bincount(tt[sel], weights=z - icv, minlength=T)
            row.append('%.4f (%.1f)' % (icv, icv / (sqrt((S ** 2).sum()) / z.size)))
        P('%s | %d | %s' % (f, L, '  '.join(row)))
        del M, xv, yv, tt
finish()
