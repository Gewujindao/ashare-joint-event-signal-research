# size_only_account.py — naive rule (ii) of return_attribution (hold the 50 smallest by the float-cap proxy, at most 5 swaps a day, pool
# leavers sold first, X1/X2 excluded from buys) run as ONE continuous account from the first possible day (2015) to
# 2026-10-09, read year by year against benchmark B; plus gross equal-weighted quintile returns minus the pool, by year.
# Engine, costs and execution: design\gbm_account_backtest\account_core (unchanged). Rule (ii) = the call used in design\return_attribution\attribution.py
# (Q.run_account(..., lambda t: np.where(U[t], negcap[t], np.nan), use_x=True)); pool = frozen gbm_composite.py pool (pool_ranks.py).
import sys, os, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'gbm_account_backtest'))
import account_core as Q
from account_core import C, np, sqrt, P
C.OUTDIR = HERE
C.start('c4_run')
ROOT = r'<BLIND_ROOT>'
NPZ = ROOT + r'\data_full\ashare_mainboard_full_20261009.npz'
DATA_END = '2026-10-09'
K0 = 500000.0


def tick(msg):
    P('# [%5.1fs] %s' % (time.time() - C._T0, msg))


z = np.load(os.path.join(HERE, 'cache_c4_pool.npz'), allow_pickle=False)
r0, t0 = int(z['r0']), int(z['t0'])
U, ra, rc, dz = z['U'], z['ra'], z['rc'], z['dates']
z.close()

# ---- check against the return_attribution cache on the overlapping rows ----
z = np.load(os.path.join(HERE, '..', 'return_attribution', 'cache_c3_scores.npz'), allow_pickle=False)
k = int(np.searchsorted(dz, z['dates'][0]))
ok = bool((dz[k:] == z['dates']).all() and (U[k:] == z['U']).all() and (ra[k:] == z['ra']).all() and (rc[k:] == z['rc']).all())
P('pool and ranks vs cache_c3_scores.npz on %d overlapping rows (%s..%s): %s' % (dz.size - k, z['dates'][0], z['dates'][-1], 'IDENTICAL' if ok else 'DIFFERENT'))
z.close()
assert ok

# ---- float-cap proxy, raw values (same code as attribution.py main) ----
zz = np.load(NPZ, allow_pickle=False)
nrow = int((zz['dates'] <= DATA_END).sum())
s0 = r0 - 40
tr = zz['tradable'][s0:nrow]
am, tu = zz['amount_cny'][s0:nrow], zz['turnover_pct'][s0:nrow]
zz.close()
with np.errstate(invalid='ignore', divide='ignore'):
    xx = (np.where(tr & (am > 0), am, np.nan).astype(np.float32) / np.where(tr & (tu > 0), tu, np.nan).astype(np.float32) * 100)
del am, tu, tr
v = np.isfinite(xx)
cs = np.cumsum(np.where(v, xx, 0), axis=0, dtype=np.float64)
cn = np.cumsum(v, axis=0, dtype=np.int32)
del xx, v
W = 20
s = cs[W - 1:].copy()
s[1:] -= cs[:-W]
n_ = cn[W - 1:].copy()
n_[1:] -= cn[:-W]
del cs, cn
with np.errstate(invalid='ignore', divide='ignore'):
    fcap = (s / n_)
fcap[n_ < 10] = np.nan
fcap = fcap[40 - (W - 1):].astype(np.float32)
del s, n_
negcap = -fcap
del fcap
tick('fcap')

M = Q.market(NPZ, DATA_END, r0)
for kk in ('cq', 'cr', 'pc', 'tr', 'st', 'x1', 'x2'):
    del M[kk]
tick('market')
dts = M['dates']
assert bool((dts == dz).all())
Tn, N = negcap.shape
assert U.shape == (Tn, N)
M['U'] = U
labA = (ra // 40).astype(np.int16)
labC = (rc // 40).astype(np.int16)
M['qa'] = labA
bad = tot = 0
for t in range(t0 - r0, Tn):
    idx = np.nonzero(rc[t] != 255)[0]
    o = np.argsort(-negcap[t, idx], kind='stable')
    rr = np.empty(idx.size, np.int64)
    rr[o] = np.arange(idx.size)
    bad += int((rr * 200 // idx.size != rc[t, idx]).sum())
    tot += idx.size
P('float-cap proxy recomputed: 200-bucket rank differs from the frozen fcap rank on %d of %d stock-days' % (bad, tot))

a, b = t0 - r0 + 1, Tn
P('account: cash at the open of %s (signal day %s, pool %d) .. %s ; %d days' % (dts[a], dts[a - 1], int(U[a - 1].sum()), dts[b - 1], b - a))
B = Q.bench_B(M, a, b, K0, zero_term=True, use_x=True)
P('benchmark B: pool %d, bought %d, ended series held %d' % (B['npool'], B['nbought'], B['n_term']))
R = Q.run_account(M, a, b, K0, lambda t: np.where(U[t], negcap[t], np.nan), detail=True, use_x=True)
tick('accounts')
navS, navB = R['navz'] / K0, B['nav']
r = Q.readings(R['navz'], navB, K0)
c = R['c']
yrs = r['n'] / 252.0
P('WHOLE %s..%s n=%d | rule ann %.2f maxDD %.2f | B ann %.2f maxDD %.2f | ann diff %.2f (arith %.2f) t %.2f | NAV rule %.4f B %.4f'
  % (dts[a], dts[b - 1], r['n'], r['annS'], r['ddS'], r['annB'], r['ddB'], r['t1'], r['t1_arith'], r['t'], r['navS'], r['navB']))
P('rule rhythm: mean held %.1f | cash share %.3f | buys %d sells %d | cost %%/yr %.2f | ended series ever held %d | sells pending at end %d'
  % (R['nheld'].mean(), R['cashr'].mean(), c['nbuy'], c['nsell'], c['cost'] / R['navz'].mean() / yrs * 100, R['n_term'], R['pend_final']))


def dd_info(path, d):
    p = np.r_[1.0, path]
    dd_ = np.concatenate([np.array(['start']), d])
    pk = np.maximum.accumulate(p)
    x = 1 - p / pk
    i = int(x.argmax())
    ip = int(np.nonzero(p[:i + 1] == pk[i])[0][-1])
    rec = np.nonzero(p[i:] >= pk[i])[0]
    ir = i + int(rec[0]) if rec.size else -1
    return x[i] * 100, dd_[ip], dd_[i], (dd_[ir] if ir >= 0 else 'not recovered by ' + d[-1]), i - ip, (ir - i if ir >= 0 else -1)


for tag, pth in (('rule', navS), ('B', navB)):
    m, d0, d1, d2, n1, n2 = dd_info(pth, dts[a:b])
    P('max drawdown %-4s %.2f%% | peak %s -> trough %s (%d days) -> recovered %s (%d days after trough)' % (tag, m, d0, d1, n1, d2, n2))

yr = np.array([s_[:4] for s_ in dts[a:b]])
stuck = np.array([(M['last_tr'][hl] < a + j).sum() for j, (_, _, hl) in enumerate(R['log'])])
P('\n-- by calendar year (account, close to close, conservative valuation; ann. = calendar year return) --')
P('year days | rule %% | B %% | rule-B pts | maxDD rule %% | maxDD B %% | rule mean held | stuck ended slots (mean / year end) | cash share | daily diff t')
prevS = prevB = 1.0
for y in sorted(set(yr)):
    ix = np.nonzero(yr == y)[0]
    pS, pB = navS[ix], navB[ix]
    ys, yb = pS[-1] / prevS - 1, pB[-1] / prevB - 1
    ddS, ddB = Q.maxdd(np.r_[prevS, pS]) * 100, Q.maxdd(np.r_[prevB, pB]) * 100
    dd = r['d'][ix]
    P('%s %3d | %7.2f | %7.2f | %7.2f | %6.2f | %6.2f | %5.1f | %5.2f / %2d | %.3f | %5.2f' % (
        y, ix.size, ys * 100, yb * 100, (ys - yb) * 100, ddS, ddB, R['nheld'][ix].mean(), stuck[ix].mean(), stuck[ix[-1]],
        R['cashr'][ix].mean(), dd.mean() / dd.std(ddof=1) * sqrt(ix.size)))
    prevS, prevB = pS[-1], pB[-1]
for lo, hi in (('2015', '2015'), ('2016', '2018'), ('2017', '2018'), ('2019', '2021'), ('2022', '2023'), ('2024', '2026'), ('2020', '2026'), ('2015', '2019')):
    ix = np.nonzero((yr >= lo) & (yr <= hi))[0]
    sN = np.r_[1.0, navS]
    bN = np.r_[1.0, navB]
    gS, gB = sN[ix[-1] + 1] / sN[ix[0]] - 1, bN[ix[-1] + 1] / bN[ix[0]] - 1
    aS, aB = (1 + gS) ** (252.0 / ix.size) - 1, (1 + gB) ** (252.0 / ix.size) - 1
    P('group %s-%s %4d d | cum rule %7.2f%% B %7.2f%% | ann rule %6.2f%% B %6.2f%% diff %6.2f | maxDD rule %.2f B %.2f' % (
        lo, hi, ix.size, gS * 100, gB * 100, aS * 100, aB * 100, (aS - aB) * 100, Q.maxdd(sN[ix[0]:ix[-1] + 2]) * 100, Q.maxdd(bN[ix[0]:ix[-1] + 2]) * 100))

# ---- gross quintiles (no account): open(e)->open(e+1), pool and labels of the signal day e-1, equal weights ----
with np.errstate(invalid='ignore', divide='ignore'):
    Off = np.where(M['off0'] != 0, M['off0'], np.nan).astype(np.float32)
    oo = np.full(Off.shape, np.nan, np.float32)
    oo[:-1] = Off[1:] / Off[:-1] - 1
    del Off
ee = np.arange(a, b - 1)
G = np.ones((len(ee), 11))      # pool, cap q0..q4, amt q0..q4 daily gross return
for j, e in enumerate(ee):
    rr_ = oo[e]
    uu = U[e - 1] & np.isfinite(rr_)
    G[j, 0] = rr_[uu].mean(dtype=np.float64)
    for q in range(5):
        G[j, 1 + q] = rr_[uu & (labC[e - 1] == q)].mean(dtype=np.float64)
        G[j, 6 + q] = rr_[uu & (labA[e - 1] == q)].mean(dtype=np.float64)
yq = np.array([dts[e][:4] for e in ee])
P('\n-- gross equal-weighted quintile return minus pool return, by year, percentage points (compounded within the year; open-to-open) --')
P('year days | pool %% | cap q1(smallest) q2 q3 q4 q5 | amt60 q1(lowest) q2 q3 q4 q5')
for y in sorted(set(yq)):
    ix = yq == y
    g = np.prod(1 + G[ix], axis=0) - 1
    P('%s %3d | %7.2f | %s | %s' % (y, ix.sum(), g[0] * 100, ' '.join('%7.2f' % ((g[1 + q] - g[0]) * 100) for q in range(5)),
                                   ' '.join('%7.2f' % ((g[6 + q] - g[0]) * 100) for q in range(5))))
g = np.prod(1 + G, axis=0) ** (252.0 / len(ee)) - 1
P('whole (annualised) | %7.2f | %s | %s' % (g[0] * 100, ' '.join('%7.2f' % ((g[1 + q] - g[0]) * 100) for q in range(5)), ' '.join('%7.2f' % ((g[6 + q] - g[0]) * 100) for q in range(5))))
np.savez_compressed(os.path.join(HERE, 'record_c4.npz'), dates=dts[a:b], navS=navS, navB=navB, nheld=R['nheld'], stuck=stuck, cashr=R['cashr'],
                    qdates=dts[ee], qret=G)
C.finish()
