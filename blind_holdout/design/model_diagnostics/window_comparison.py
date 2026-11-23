# window_comparison.py -- run C: descriptive comparison on the three already-used windows (V, selection, hold-out). Not new evidence.
# Uses the frozen default-GBDT scores cached by return_attribution (cache_c3_scores.npz) and opens from the full panel.
# Returns here are SIMPLE open(t+1) -> open(t+1+H) on forward-filled opens (entry needs a valid open at t+1; limit-up at the open is not checked).
# Two conventions for series that end inside the data: 'stale' keeps the last open, 'zero' books -100%.
import sys
sys.path.insert(0, r'<BLIND_ROOT>\design\base_rates')
import common as C
from common import *
C.OUTDIR = C.ROOT + r'\design\model_diagnostics'
start('d0_c')
z = np.load(C.ROOT + r'\design\return_attribution\cache_c3_scores.npz', allow_pickle=False)
r0 = int(z['r0'])
cd, G, U, rc = z['dates'], z['G'], z['U'], z['rc']
z.close()
f = np.load(C.ROOT + r'\data_full\ashare_mainboard_full_20261009.npz', allow_pickle=False)
dates = f['dates'][r0:]
assert dates.shape == cd.shape and bool((dates == cd).all())
tr = f['tradable'][r0:]
oq = f['open_qfq'][r0:].astype(np.float32)
f.close()
T, N = G.shape
assert tr.shape == (T, N)
OQ = np.where(tr & (oq > 0), oq, np.nan).astype(np.float32)
del oq, tr
valid = np.isfinite(OQ)
lastv = T - 1 - np.argmax(valid[::-1], axis=0)
ended = valid.any(0) & (lastv < T - 60)
Off = ffill(OQ)
rows = np.arange(T)[:, None]
ent = np.zeros((T, N), bool)
ent[:T - 1] = valid[1:]
P('rows %d (%s..%s) ; names whose series ends more than 60 days before the last date: %d' % (T, dates[0], dates[-1], ended.sum()))


def urank(x):
    n = x.size
    r = np.empty(n)
    r[np.argsort(x, kind='stable')] = (np.arange(n) + 0.5) / n - 0.5
    return r


def fwd(H, zero):
    R = np.full((T, N), np.nan, np.float32)
    R[:T - 1 - H] = Off[1 + H:] / Off[1:T - H] - 1
    R[~ent] = np.nan
    if zero:
        R[ended[None, :] & (rows + 1 + H > lastv[None, :]) & ent & np.isfinite(R)] = -1.0
    return R


WIN = [('V 2020-01-03..2022-12-30', '2020-01-03', '2022-12-30'), ('SEL 2023-01-03..2024-06-28', '2023-01-03', '2024-06-28'),
       ('HOLD 2024-07-01..2026-10-09', '2024-07-01', '2026-10-09'), ('2026Q2', '2026-04-01', '2026-06-30')]
KEYS = ['top50', 'top10%', 'bot10%', 'bot50', 'cap decile mean', 'G top50 in cap decile', 'smallest 2% by cap', 'G bot50 in cap decile', 'cap decile ex G bot50']
P('all numbers: mean over days of the equal-weight relative return (minus the same-day pool mean), bp/day ; then rank IC all / inside top 10% ; share of G top-50 in the smallest cap decile ;'
  ' top-50 slots hit by an ended series %')
for H in (5, 10):
    for zero in (False, True):
        R = fwd(H, zero)
        for wn, a, b in WIN:
            acc = {k: [] for k in KEYS}
            ica, ict, shr, dead = [], [], [], []
            for t in np.nonzero((dates >= a) & (dates <= b))[0]:
                v = U[t] & np.isfinite(R[t]) & np.isfinite(G[t])
                n = int(v.sum())
                if n < 200:
                    continue
                x = R[t, v].astype(np.float64)
                rawx = x.copy()
                x -= x.mean()
                s = G[t, v]
                c = rc[t, v]
                o = np.argsort(-s, kind='stable')
                d1 = n // 10
                acc['top50'].append(x[o[:50]].mean())
                acc['top10%'].append(x[o[:d1]].mean())
                acc['bot10%'].append(x[o[n - d1:]].mean())
                acc['bot50'].append(x[o[n - 50:]].mean())
                dm = c < 20
                xd, sd_ = x[dm], s[dm]
                od = np.argsort(-sd_, kind='stable')
                acc['cap decile mean'].append(xd.mean())
                acc['G top50 in cap decile'].append(xd[od[:50]].mean())
                acc['G bot50 in cap decile'].append(xd[od[-50:]].mean())
                acc['cap decile ex G bot50'].append(xd[od[:-50]].mean())
                acc['smallest 2% by cap'].append(x[c <= 3].mean())
                ica.append(np.corrcoef(urank(s), urank(x))[0, 1])
                ict.append(np.corrcoef(urank(s[o[:d1]]), urank(x[o[:d1]]))[0, 1])
                shr.append(dm[o[:50]].mean())
                dead.append((rawx[o[:50]] == -1.0).mean())
            k_ = 1e4 / H
            P('H %2d | %-5s | %-27s | %s | IC %+.4f / %+.4f | %.2f | %.2f' % (
                H, 'zero' if zero else 'stale', wn, ' | '.join('%s %7.2f' % (k, np.mean(acc[k]) * k_) for k in KEYS), np.mean(ica), np.mean(ict), np.mean(shr),
                100 * np.mean(dead)))
        del R
finish()
