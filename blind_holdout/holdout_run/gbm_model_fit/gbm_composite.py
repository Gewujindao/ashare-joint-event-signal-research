# gbm_composite.py — frozen GBDT composite G, quota structure K=50 / H=10.
# usage: gbm_composite.py DATA_END MODE TAG PREV_RECORD SEL_START
#   MODE fit    : fit G on the fit segment F (pool start .. 2019-12-31), save the model and the daily record.
#   MODE extend : load the saved model, recompute everything on data <= DATA_END, check that the previously saved daily
#                 record is reproduced exactly (prefix invariance), then report the new days.
# Data rows after DATA_END are never read into any computation: every array is sliced right after loading.
# Feature, pool, label, model and quota code is the same as design\boosted_tree_measurements\training_measurements.py; rolling means are evaluated
# in column blocks (identical values, less memory).
import sys
sys.path.insert(0, r'<BLIND_ROOT>\design\base_rates')
import common as C
from common import np, sqrt, P, start, finish, ffill, exec_maps, next_true, pct_rank_rows, abn_turnover
import time, pickle, os
DATA_END, MODE, TAG, PREV, SEL_START = sys.argv[1:6]
assert MODE in ('fit', 'extend') and DATA_END <= '2024-06-28'
OUT = C.ROOT + r'\design\gbm_model_fit'
C.OUTDIR = OUT
start('c1_' + TAG)
K, H = 50, 10
COST = {'strict': 35.2, 'mine': 19.2}
DRAG = 0.5e-4


def tick(msg):
    P('# [%5.1fs] %s' % (time.time() - C._T0, msg))


def load(*names):
    z = np.load(C.NPZ, allow_pickle=False)
    dts = z['dates']
    n = int((dts <= DATA_END).sum())
    assert dts[n - 1] <= DATA_END and (n == dts.size or dts[n] > DATA_END)
    out = {'dates': dts[:n].copy()}
    for k in names:
        out[k] = np.ascontiguousarray(z[k][:n])
    z.close()
    return out


d = load('tradable', 'st', 'close_qfq')
dates = d['dates']
assert bool((dates <= DATA_END).all())
years = np.array([int(s[:4]) for s in dates])
quarters = np.array([(int(s[5:7]) - 1) // 3 + 1 for s in dates])
tr, st = d['tradable'], d['st']
T, N = tr.shape
iF = int((dates <= '2019-12-31').sum()) - 1
assert dates[iF] <= '2019-12-31' and dates[iF + 1] >= '2020-01-02'
P('data rows %d, last date %s ; mode %s ; tag %s' % (T, dates[-1], MODE, TAG))
has = ~np.isnan(d['close_qfq'])
first = has.argmax(axis=0)
age = np.cumsum(tr, axis=0, dtype=np.int32)
seas = tr & ((first == 0)[None, :] | (age > 750))
del age, has
Cff = ffill(np.where(tr, d['close_qfq'], np.nan).astype(np.float32))
del d
r1 = np.full((T, N), np.nan, np.float32)
r1[1:] = Cff[1:] / Cff[:-1] - 1


def lagret(k):
    out = np.full((T, N), np.nan, np.float32)
    out[k:] = Cff[k:] / Cff[:-k] - 1
    return out


def _rmean(x, W, minc):
    v = np.isfinite(x)
    cs = np.cumsum(np.where(v, x, 0), axis=0, dtype=np.float64)
    cn = np.cumsum(v, axis=0, dtype=np.int32)
    s = cs[W - 1:].copy()
    s[1:] -= cs[:-W]
    n = cn[W - 1:].copy()
    n[1:] -= cn[:-W]
    del cs, cn
    with np.errstate(invalid='ignore', divide='ignore'):
        m = s / n
    m[n < minc] = np.nan
    return m


def rmean(x, W, minc=None):
    minc = W if minc is None else minc
    out = np.full(x.shape, np.nan, np.float32)
    for a in range(0, N, 500):
        out[W - 1:, a:a + 500] = _rmean(x[:, a:a + 500], W, minc)
    return out


def rext(x, W, mx=True):
    out = np.full((T, N), -np.inf if mx else np.inf, np.float32)
    f = np.fmax if mx else np.fmin
    for j in range(W):
        f(out[j:], x[:T - j], out=out[j:])
    out[~np.isfinite(out)] = np.nan
    return out


NAMES = ['ret1', 'ret3', 'ret5', 'ovn5', 'intra5', 'ovn20', 'intra20', 'abn_turn', 'turn5_60', 'turn20', 'pv_corr20',
         'amihud20', 'amt60', 'logprice', 'fcap', 'dist_high60', 'dist_low60', 'clv5', 'dist_high250', 'streak', 'lu20', 'ld20',
         'max20', 'vol20', 'downshare20', 'hl20', 'ret20', 'ret60', 'ret120', 'mom250_20']
NF = 30
RK = np.full((NF, T, N), 255, np.uint8)
mom = np.full((T, N), np.nan, np.float32)
mom[250:] = Cff[230:T - 20] / Cff[:T - 250] - 1
U0 = seas & ~st & np.isfinite(mom)
del seas


def store(name, x):
    f = NAMES.index(name)
    fin = U0 & np.isfinite(x)
    for t in range(T):
        idx = np.nonzero(fin[t])[0]
        if idx.size < 200:
            continue
        o = np.argsort(x[t, idx], kind='stable')
        r = np.empty(idx.size, np.int64)
        r[o] = np.arange(idx.size)
        RK[f, t, idx] = r * 200 // idx.size


store('mom250_20', mom)
del mom
store('ret1', r1)
for k, nm in ((3, 'ret3'), (5, 'ret5'), (20, 'ret20'), (60, 'ret60'), (120, 'ret120')):
    store(nm, lagret(k))
store('max20', rext(r1, 20))
m1 = rmean(r1, 20)
m2 = rmean(r1 * r1, 20)
store('vol20', np.sqrt(np.maximum(m2 - m1 * m1, 0)))
del m1
neg = np.minimum(r1, 0)
with np.errstate(invalid='ignore', divide='ignore'):
    store('downshare20', rmean(neg * neg, 20) / m2)
del m2, neg
run = np.zeros(N, np.float32)
stk = np.zeros((T, N), np.float32)
for t in range(T):
    x = r1[t]
    run = np.where(x > 0, np.maximum(run, 0) + 1, np.where(x < 0, np.minimum(run, 0) - 1, 0)).astype(np.float32)
    stk[t] = run
store('streak', stk)
del stk
tick('stage1')

OQ = load('open_qfq')['open_qfq']
OQ = np.where(tr, OQ, np.nan).astype(np.float32)
Cprev = np.full((T, N), np.nan, np.float32)
Cprev[1:] = Cff[:-1]
with np.errstate(invalid='ignore', divide='ignore'):
    ovn = np.where(tr, OQ / Cprev - 1, np.where(np.isfinite(Cprev), 0, np.nan)).astype(np.float32)
del Cprev
store('ovn5', rmean(ovn, 5))
store('ovn20', rmean(ovn, 20))
del ovn
with np.errstate(invalid='ignore', divide='ignore'):
    intra = np.where(tr, Cff / OQ - 1, np.where(np.isfinite(Cff), 0, np.nan)).astype(np.float32)
store('intra5', rmean(intra, 5))
store('intra20', rmean(intra, 20))
del intra
tick('stage2')

raw = load('turnover_pct')['turnover_pct']
store('abn_turn', abn_turnover(raw, tr))
turn = np.where(tr & (raw > 0), raw, np.nan).astype(np.float32)
del raw
t20 = rmean(turn, 20, 10)
store('turn20', t20)
store('turn5_60', rmean(turn, 5, 3) / rmean(turn, 60, 30))
x = np.where(np.isfinite(turn), r1, np.nan).astype(np.float32)
mx_ = rmean(x, 20, 10)
cov = rmean(x * turn, 20, 10) - mx_ * t20
vx = rmean(x * x, 20, 10) - mx_ ** 2
del mx_
vy = rmean(turn * turn, 20, 10) - t20 ** 2
del t20
with np.errstate(invalid='ignore', divide='ignore'):
    pv = cov / np.sqrt(vx * vy)
del cov, vx, vy
pv[~np.isfinite(pv)] = np.nan
store('pv_corr20', pv)
del pv
raw = load('amount_cny')['amount_cny']
amt = np.where(tr & (raw > 0), raw, np.nan).astype(np.float32)
del raw
with np.errstate(invalid='ignore', divide='ignore'):
    store('amihud20', rmean(np.abs(x) / amt, 20, 10))
    del x
    store('amt60', rmean(amt, 60, 30))
    store('fcap', rmean(amt / turn * 100, 20, 10))
del turn, amt
tick('stage3')

d4 = load('open_raw', 'preclose_raw', 'close_raw')
cr, pc = d4['close_raw'], d4['preclose_raw']
store('logprice', np.where(tr, cr, np.nan).astype(np.float32))
lim = np.where(st, 0.05, 0.10).astype(np.float32)
base = tr & (pc > 0)
LU = (base & (cr >= pc * (1 + lim) - 0.005)).astype(np.float32)
store('lu20', rmean(LU, 20) * 20)
del LU
LD = (base & (cr <= pc * (1 - lim) + 0.005)).astype(np.float32)
store('ld20', rmean(LD, 20) * 20)
del LD, base, lim
sell_ok, buy_ok = exec_maps(dict(tradable=tr, st=st, open_raw=d4['open_raw'], preclose_raw=pc, open_qfq=OQ))
del d4, cr, pc
tick('stage4')

Hq = load('high_qfq')['high_qfq']
Hq = np.where(tr, Hq, np.nan).astype(np.float32)
store('dist_high60', Cff / rext(Hq, 60) - 1)
store('dist_high250', Cff / rext(Hq, 250) - 1)
Lq = load('low_qfq')['low_qfq']
Lq = np.where(tr, Lq, np.nan).astype(np.float32)
store('dist_low60', Cff / rext(Lq, 60, False) - 1)
Ct = np.where(tr, Cff, np.nan).astype(np.float32)
del Cff, r1
with np.errstate(invalid='ignore', divide='ignore'):
    clv = np.where(Hq > Lq, (Ct - Lq) / (Hq - Lq), np.where(tr, 0.5, np.nan)).astype(np.float32)
    del Ct
    store('clv5', rmean(clv, 5, 3))
    del clv
    store('hl20', rmean(Hq / Lq - 1, 20, 10))
del Hq, Lq
tick('stage5')

U = U0.copy()
for f in range(NF):
    U &= RK[f] != 255
del U0
npool = U.sum(1)
t0 = int(np.nonzero(npool >= 200)[0][0])
P('pool: first day %s (row %d); fit segment ends %s (row %d)' % (dates[t0], t0, dates[iF], iF))
MODEL = OUT + r'\g_model.pkl'
if MODE == 'fit':
    TF = iF + 1
    nsell = next_true(sell_ok[:TF])
    nbuy = next_true(buy_ok[:TF])
    ar = np.arange(TF)[:, None]
    b = nbuy[1:TF + 1].copy()
    okb = b < TF
    s_ = np.take_along_axis(nsell, np.minimum(np.where(okb, b, 0) + 10, TF), axis=0)
    ok = okb & (s_ < TF)
    OQF = OQ[:TF]
    R = np.take_along_axis(OQF, np.where(ok, s_, 0), axis=0) / np.take_along_axis(OQF, np.where(ok, b, 0), axis=0) - 1
    R[~ok] = np.nan
    R = R.astype(np.float32)
    del nsell, nbuy, b, okb, s_, ok
    gx, gy = [], []
    for t in range(t0, iF + 1):
        if (t - t0) % 5 != 0:
            continue
        v = U[t] & np.isfinite(R[t])
        if v.sum() < 200:
            continue
        xr = (R[t, v] - R[t, v].mean(dtype=np.float64)).astype(np.float32)
        n = xr.size
        y = np.empty(n)
        y[np.argsort(xr)] = (np.arange(n) + 0.5) / n - 0.5
        gx.append(RK[:, t, v].T.copy())
        gy.append(y.astype(np.float32))
    del R
    from sklearn.ensemble import HistGradientBoostingRegressor
    GX = np.concatenate(gx).astype(np.float32)
    GY = np.concatenate(gy)
    del gx, gy
    gb = HistGradientBoostingRegressor(random_state=0).fit(GX, GY)
    P('GBDT fit: rows %d, days sampled every 5th from %s, iterations %d, params %s' % (GX.shape[0], dates[t0], gb.n_iter_, gb.get_params()))
    del GX, GY
    with open(MODEL, 'wb') as fh:
        pickle.dump(gb, fh)
else:
    with open(MODEL, 'rb') as fh:
        gb = pickle.load(fh)
P('model file bytes %d' % os.path.getsize(MODEL))
G = np.full((T, N), np.nan, np.float32)
for a in range(t0, T, 50):
    tt, ii = np.nonzero(U[a:min(a + 50, T)])
    if tt.size:
        G[tt + a, ii] = gb.predict(RK[:, tt + a, ii].T.astype(np.float32))
PRG = pct_rank_rows(G, U & np.isfinite(G))
del G
qa = RK[NAMES.index('amt60')] // 40
del RK
tick('scores')
Off = ffill(OQ)
oo = np.full((T, N), np.nan, np.float32)
oo[:-1] = Off[1:] / Off[:-1] - 1
del Off, OQ

# ---- quota structure (same rule as training_measurements.quota), plus the stratum decomposition ----
held = np.zeros(N, bool)
x = np.full(T, np.nan)
buys = np.zeros(T)
bs = np.full(T, np.nan)
ms = np.full(T, np.nan)
sty = np.full(T, np.nan)
sel = np.full(T, np.nan)
csh = np.full(T, np.nan)
a0 = np.full(T, np.nan)
for t in range(t0, T - 1):
    e = t + 1
    sc = PRG[t]
    u = U[t]
    nq = (t + 1) * K // H - t * K // H
    ci = np.nonzero(u & ~held & buy_ok[e] & np.isfinite(sc))[0]
    ci = ci[np.argsort(-sc[ci], kind='stable')]
    hi_ = np.nonzero(held & sell_ok[e])[0]
    hs = np.where(np.isfinite(sc[hi_]), sc[hi_], -np.inf)
    ho = np.argsort(hs, kind='stable')
    hi_, hs = hi_[ho], hs[ho]
    fill = min(K - int(held.sum()), ci.size)
    nb = fill
    if fill:
        held[ci[:fill]] = True
    cand = ci[fill:fill + nq]
    m = min(cand.size, hi_.size)
    if m > 0:
        m = int((sc[cand[:m]] > hs[:m]).sum())
        if m > 0:
            held[hi_[:m]] = False
            held[cand[:m]] = True
            nb += m
    r = oo[e]
    fin = np.isfinite(r)
    hv = held & fin
    b = r[hv].sum(dtype=np.float64) / K
    uu = u & fin
    if not uu.any():
        continue
    mm = r[uu].mean(dtype=np.float64)
    x[e], buys[e], bs[e], ms[e] = b - mm, nb, b, mm
    qt = qa[t]
    bench = np.full(N, mm)
    for q in range(5):
        z = uu & (qt == q)
        if z.any():
            bench[z] = r[z].mean(dtype=np.float64)
    sty[e] = (bench[hv] - mm).sum() / K
    sel[e] = (r[hv].astype(np.float64) - bench[hv]).sum() / K
    csh[e] = -(1 - hv.sum() / K) * mm
    hq = held & u & (qt < 5)
    a0[e] = (qt[hq] == 0).mean() if hq.any() else np.nan
tick('quota')
assert np.nanmax(np.abs(x - sty - sel - csh)) < 1e-9


def stats(name, v):
    v = v & np.isfinite(x)
    n = int(v.sum())
    g = x[v].mean()
    te = x[v].std(ddof=1)
    to = buys[v].mean() / K
    out = []
    for cname, c in COST.items():
        net = x[v] - buys[v] / K * c * 1e-4 - DRAG
        out.append('%s net %%/yr %7.2f  t(n) %6.2f  t551(net/TE) %6.2f' % (cname, net.mean() * 25200, net.mean() / net.std(ddof=1) * sqrt(n),
                                                                          net.mean() / te * sqrt(551)))
    P('%-8s | %s..%s n=%d | gross bp/d %6.2f | TE %%/d %.3f (annual %% %.2f) | holding days %.1f | %s' % (
        name, dates[v][0], dates[v][-1], n, g * 1e4, te * 100, te * 100 * sqrt(252), 1 / to, ' | '.join(out)))
    P('%-8s | share of holdings in smallest amount quintile A0 %.3f | decomposition bp/d: style %6.2f  within-stratum %6.2f  cash %6.2f |'
      ' sd %%/d: style %.3f within %.3f | gross t(n): style %.2f within %.2f' % (
          name, np.nanmean(a0[v]), sty[v].mean() * 1e4, sel[v].mean() * 1e4, csh[v].mean() * 1e4, sty[v].std(ddof=1) * 100, sel[v].std(ddof=1) * 100,
          sty[v].mean() / sty[v].std(ddof=1) * sqrt(n), sel[v].mean() / sel[v].std(ddof=1) * sqrt(n)))


rows = np.arange(T)
fin = np.isfinite(x)
SEGS = {'F': (rows >= t0 + 2) & (rows <= iF), 'V20-22': (rows >= iF + 2) & (dates <= '2022-12-30'), 'ALL_OOS': rows >= iF + 2}
REC = OUT + '\\record_' + TAG + '.npz'
if MODE == 'extend':
    pr = np.load(OUT + '\\' + PREV, allow_pickle=False)
    k = np.searchsorted(dates, pr['dates'])
    assert bool((dates[k] == pr['dates']).all())
    dx = np.abs(x[k] - pr['x']).max()
    db = np.abs(buys[k] - pr['buys']).max()
    P('prefix check against %s: %d days %s..%s ; max |x diff| %.3e ; max |buys diff| %.1f' % (PREV, k.size, pr['dates'][0], pr['dates'][-1], dx, db))
    if not (dx == 0 and db == 0):
        P('PREFIX CHECK FAILED - nothing further is computed')
        finish()
        sys.exit(4)
    isrec = np.zeros(T, bool)
    isrec[k] = True
    SEGS['REC'] = isrec
    SEGS['NEW'] = (rows > k[-1]) & fin
    SEGS['NEWcal'] = (dates >= SEL_START) & fin
    SEGS['REC+NEW'] = isrec | SEGS['NEW']
P('\nSEGMENT TABLE (costs: strict 35.2bp, mine 19.2bp per round trip; drag 0.5bp/day; t(n) = ordinary t of the daily net series)')
for name, v in SEGS.items():
    if (v & fin).sum() > 30:
        stats(name, v)
P('\nBLOCKS of 60 return days, 2020-01-03 .. 2022-12-30 only (training rows): start | n | gross bp/d | TE %/d | A0 share | within-stratum bp/d')
tt = np.nonzero(SEGS['V20-22'] & fin)[0]
blk = []
for a in range(0, tt.size - 59, 60):
    ix = tt[a:a + 60]
    blk.append((x[ix].mean() * 1e4, x[ix].std(ddof=1) * 100, np.nanmean(a0[ix]), sel[ix].mean() * 1e4))
    P('%s | %d | %6.2f | %.3f | %.3f | %6.2f' % (dates[ix[0]], ix.size, *blk[-1]))
blk = np.array(blk)
P('sd across blocks: gross %.2f  TE %.3f  A0 %.3f  within %.2f  (blocks %d)' % (*blk.std(0, ddof=1), blk.shape[0]))
P('\nQUARTERS: quarter | n | net %/yr strict | net %/yr mine')
for y in range(2020, int(dates[-1][:4]) + 1):
    for q in range(1, 5):
        v = SEGS['ALL_OOS'] & fin & (years == y) & (quarters == q)
        if v.sum() < 5:
            continue
        P('%dQ%d | %d | %7.2f | %7.2f' % (y, q, v.sum(), (x[v] - buys[v] / K * 35.2e-4 - DRAG).mean() * 25200,
                                          (x[v] - buys[v] / K * 19.2e-4 - DRAG).mean() * 25200))
v = SEGS['ALL_OOS'] & fin
np.savez(REC, dates=dates[v], x=x[v], buys=buys[v], sty=sty[v], sel=sel[v], csh=csh[v], a0=a0[v])
P('record saved: %s (%d days %s..%s)' % (REC, v.sum(), dates[v][0], dates[v][-1]))
finish()
