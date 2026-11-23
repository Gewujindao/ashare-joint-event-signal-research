# market_predictability.py — reads: tradable, st, close_qfq, open_qfq, close_raw, preclose_raw, amount_cny (training rows).
# Market-level predictability: equal-weight forward market return (universe mean of open(t+1)->open(t+1+L),
# L = 1, 5, 20) against market states known at the close of t: past 5/20/60-day market return, breadth
# (share of stocks above their 20-day mean close), share of stocks closing limit-down, abnormal total amount
# (today / mean of previous 60 days). Pearson and Spearman by period; per-year sign for L = 5.
from common import *

start('q6_market')
d = load('tradable', 'st', 'close_qfq', 'open_qfq', 'close_raw', 'preclose_raw', 'amount_cny')
tr, st, yrs = d['tradable'], d['st'], d['years']
T, N = tr.shape
seas = seasoned(tr, ~np.isnan(d['close_qfq']))
Cff = ffill(np.where(tr, d['close_qfq'], np.nan).astype(np.float32))
OQ = np.where(tr, d['open_qfq'], np.nan).astype(np.float32)
rcc = np.full((T, N), np.nan, np.float32)
rcc[1:] = Cff[1:] / Cff[:-1] - 1
U = seas & np.isfinite(rcc)
cnt = np.maximum(U.sum(1), 1)
m = np.where(U, rcc, 0).sum(1, dtype=np.float64) / cnt
fm = {}
for L in (1, 5, 20):
    F = np.full((T, N), np.nan, np.float32)
    F[:T - 1 - L] = OQ[1 + L:] / OQ[1:T - L] - 1
    F[~seas] = np.nan
    with np.errstate(all='ignore'):
        fm[L] = np.nanmean(F, axis=1)
    del F
lm = np.concatenate([[0.0], np.cumsum(np.log1p(m))])   # lm[t+1] = sum log up to t
S = {}
for k in (5, 20, 60):
    x = np.full(T, np.nan)
    x[k:] = lm[k + 1:] - lm[1:T - k + 1]
    S['mret%d' % k] = x
c0 = np.nan_to_num(Cff, nan=0.0).astype(np.float64)
v0 = np.isfinite(Cff).astype(np.float64)
cs = np.vstack([np.zeros((1, N)), np.cumsum(c0, 0)])
vs = np.vstack([np.zeros((1, N)), np.cumsum(v0, 0)])
ma = np.full((T, N), np.nan)
ma[19:] = (cs[20:] - cs[:-20]) / 20
ok20 = np.zeros((T, N), bool)
ok20[19:] = (vs[20:] - vs[:-20]) == 20
del cs, vs, c0, v0
B = seas & ok20
S['breadth20'] = (B & (Cff > ma)).sum(1) / np.maximum(B.sum(1), 1)
S['breadth20'][:25] = np.nan
del ma
lim = np.where(st, 0.05, 0.10).astype(np.float32)
LD = seas & (d['close_raw'] <= d['preclose_raw'] * (1 - lim) + 0.005)
S['ld_share'] = LD.sum(1) / np.maximum(seas.sum(1), 1)
amt = np.nansum(np.where(tr, d['amount_cny'], 0), axis=1, dtype=np.float64)   # some tradable rows have NaN amount
P('tradable rows with NaN amount: %d of %d ; first date with one: %s' % (
    int((tr & np.isnan(d['amount_cny'])).sum()), int(tr.sum()),
    d['dates'][np.nonzero((tr & np.isnan(d['amount_cny'])).any(1))[0][0]]))
ca = np.concatenate([[0.0], np.cumsum(amt)])
x = np.full(T, np.nan)
x[60:] = amt[60:] / ((ca[60:T] - ca[:T - 60]) / 60)
S['abn_amount'] = x
per = periods(yrs)


def rk(v):
    r = np.empty(v.size)
    r[np.argsort(v)] = np.arange(v.size)
    return r


P('state | L | for ALL,P1,P2,P3,EX2015: Pearson / Spearman (n days; effective independent n is about n/L)')
for sn, s in S.items():
    for L in (1, 5, 20):
        y = fm[L]
        row = []
        for pn, pm in per.items():
            v = pm & np.isfinite(s) & np.isfinite(y)
            row.append('%.3f/%.3f (%d)' % (np.corrcoef(s[v], y[v])[0, 1], np.corrcoef(rk(s[v]), rk(y[v]))[0, 1], v.sum()))
        P('%s | %d | %s' % (sn, L, '  '.join(row)))
    y = fm[5]
    yr = []
    for yy in range(2014, 2023):
        v = (yrs == yy) & np.isfinite(s) & np.isfinite(y)
        yr.append(np.corrcoef(s[v], y[v])[0, 1])
    allv = np.isfinite(s) & np.isfinite(y)
    sg = np.sign(np.corrcoef(s[allv], y[allv])[0, 1])
    P('%s | L=5 by year 2014..2022: %s | years with the full-sample sign: %d of 9' % (
        sn, ' '.join('%.2f' % a for a in yr), int(sum(np.sign(a) == sg for a in yr))))
P('\nreference: daily autocorrelation of the equal-weight market return (lag 1): %.3f ; ex-2015: %.3f' % (
    np.corrcoef(m[2:], m[1:-1])[0, 1],
    np.corrcoef(m[2:][(yrs[2:] != 2015)], m[1:-1][(yrs[2:] != 2015)])[0, 1]))
finish()
