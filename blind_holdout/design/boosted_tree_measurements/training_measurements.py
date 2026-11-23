# training_measurements.py — training-segment measurement. Training rows only: common.load slices to <= 2022-12-30 and asserts.
# Fit segment F = pool start .. 2019-12-31 (labels mature inside F); check segment V = 2020-01-02 .. 2022-12-30, read once.
import sys
sys.path.insert(0, r'<BLIND_ROOT>\design\base_rates')
import common as C
from common import *
C.OUTDIR = C.ROOT + r'\design\boosted_tree_measurements'
import time
start('c0_measure')


def tick(msg):
    P('# [%5.1fs] %s' % (time.time() - C._T0, msg))


d = load('tradable', 'st', 'close_qfq')
dates, years = d['dates'], d['years']
assert d['tradable'].shape[0] == 2192 and dates[-1] == '2022-12-30' and bool((dates <= C.TRAIN_END).all())
tr, st = d['tradable'], d['st']
T, N = tr.shape
iF = int((dates <= '2019-12-31').sum()) - 1
assert dates[iF] <= '2019-12-31' and dates[iF + 1] >= '2020-01-02'
has = ~np.isnan(d['close_qfq'])
first = has.argmax(axis=0)
age = np.cumsum(tr, axis=0, dtype=np.int32)
seas = tr & ((first == 0)[None, :] | (age > 750))
del age, has
Cff = ffill(np.where(tr, d['close_qfq'], np.nan).astype(np.float32))
del d['close_qfq']
r1 = np.full((T, N), np.nan, np.float32)
r1[1:] = Cff[1:] / Cff[:-1] - 1


def lagret(k):
    out = np.full((T, N), np.nan, np.float32)
    out[k:] = Cff[k:] / Cff[:-k] - 1
    return out


def rmean(x, W, minc=None):
    minc = W if minc is None else minc
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
    out = np.full(x.shape, np.nan, np.float32)
    out[W - 1:] = m
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
CLS = [0] * 3 + [1] * 4 + [2] * 4 + [3] * 4 + [4] * 4 + [5] * 3 + [6] * 4 + [7] * 4
CLSN = ['A_rev', 'B_ovn', 'C_vol', 'D_liq', 'E_pos', 'F_lim', 'G_vola', 'H_mom']
NF = 30
assert len(NAMES) == NF and len(CLS) == NF
RK = np.full((NF, T, N), 255, np.uint8)
LO = np.full((NF, T, 30), -1, np.int16)
HI = np.full((NF, T, 30), -1, np.int16)
mom = np.full((T, N), np.nan, np.float32)
mom[250:] = Cff[230:T - 20] / Cff[:T - 250] - 1
U0 = seas & ~st & np.isfinite(mom)


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
        LO[f, t] = idx[o[:30]]
        HI[f, t] = idx[o[::-1][:30]]


store('mom250_20', mom)
del mom
store('ret1', r1)
for k, nm in ((3, 'ret3'), (5, 'ret5'), (20, 'ret20'), (60, 'ret60'), (120, 'ret120')):
    store(nm, lagret(k))
store('max20', rext(r1, 20))
m1 = rmean(r1, 20)
m2 = rmean(r1 * r1, 20)
store('vol20', np.sqrt(np.maximum(m2 - m1 * m1, 0)))
neg = np.minimum(r1, 0)
with np.errstate(invalid='ignore', divide='ignore'):
    store('downshare20', rmean(neg * neg, 20) / m2)
del m1, m2, neg
run = np.zeros(N, np.float32)
stk = np.zeros((T, N), np.float32)
for t in range(T):
    x = r1[t]
    run = np.where(x > 0, np.maximum(run, 0) + 1, np.where(x < 0, np.minimum(run, 0) - 1, 0)).astype(np.float32)
    stk[t] = run
store('streak', stk)
del stk
tick('stage1 features')

d2 = load('open_qfq')
open_qfq = d2['open_qfq']
OQ = np.where(tr, open_qfq, np.nan).astype(np.float32)
Cprev = np.full((T, N), np.nan, np.float32)
Cprev[1:] = Cff[:-1]
with np.errstate(invalid='ignore', divide='ignore'):
    ovn = np.where(tr, OQ / Cprev - 1, np.where(np.isfinite(Cprev), 0, np.nan)).astype(np.float32)
    intra = np.where(tr, Cff / OQ - 1, np.where(np.isfinite(Cff), 0, np.nan)).astype(np.float32)
del Cprev
store('ovn5', rmean(ovn, 5))
store('ovn20', rmean(ovn, 20))
store('intra5', rmean(intra, 5))
store('intra20', rmean(intra, 20))
del ovn, intra
tick('stage2 features')

d3 = load('turnover_pct', 'amount_cny')
store('abn_turn', abn_turnover(d3['turnover_pct'], tr))
turn = np.where(tr & (d3['turnover_pct'] > 0), d3['turnover_pct'], np.nan).astype(np.float32)
amt = np.where(tr & (d3['amount_cny'] > 0), d3['amount_cny'], np.nan).astype(np.float32)
del d3
t20 = rmean(turn, 20, 10)
store('turn20', t20)
store('turn5_60', rmean(turn, 5, 3) / rmean(turn, 60, 30))
x = np.where(np.isfinite(turn), r1, np.nan).astype(np.float32)
mx_ = rmean(x, 20, 10)
cov = rmean(x * turn, 20, 10) - mx_ * t20
vx = rmean(x * x, 20, 10) - mx_ ** 2
vy = rmean(turn * turn, 20, 10) - t20 ** 2
with np.errstate(invalid='ignore', divide='ignore'):
    pv = cov / np.sqrt(vx * vy)
pv[~np.isfinite(pv)] = np.nan
store('pv_corr20', pv)
del pv, cov, vx, vy, mx_, t20
with np.errstate(invalid='ignore', divide='ignore'):
    store('amihud20', rmean(np.abs(x) / amt, 20, 10))
    amt60 = rmean(amt, 60, 30)
    store('amt60', amt60)
    store('fcap', rmean(amt / turn * 100, 20, 10))
del x, turn
tick('stage3 features')

d4 = load('open_raw', 'preclose_raw', 'close_raw')
cr, pc = d4['close_raw'], d4['preclose_raw']
store('logprice', np.where(tr, cr, np.nan).astype(np.float32))
lim = np.where(st, 0.05, 0.10).astype(np.float32)
base = tr & (pc > 0)
LU = (base & (cr >= pc * (1 + lim) - 0.005)).astype(np.float32)
LD = (base & (cr <= pc * (1 - lim) + 0.005)).astype(np.float32)
store('lu20', rmean(LU, 20) * 20)
store('ld20', rmean(LD, 20) * 20)
sell_ok, buy_ok = exec_maps(dict(tradable=tr, st=st, open_raw=d4['open_raw'], preclose_raw=pc, open_qfq=open_qfq))
nsell = next_true(sell_ok)
nbuy = next_true(buy_ok)
del d4, cr, pc, LU, LD, base, lim, open_qfq, d2
tick('stage4 features')

d5 = load('high_qfq', 'low_qfq')
Hq = np.where(tr, d5['high_qfq'], np.nan).astype(np.float32)
Lq = np.where(tr, d5['low_qfq'], np.nan).astype(np.float32)
del d5
store('dist_high60', Cff / rext(Hq, 60) - 1)
store('dist_low60', Cff / rext(Lq, 60, False) - 1)
store('dist_high250', Cff / rext(Hq, 250) - 1)
Ct = np.where(tr, Cff, np.nan).astype(np.float32)
with np.errstate(invalid='ignore', divide='ignore'):
    clv = np.where(Hq > Lq, (Ct - Lq) / (Hq - Lq), np.where(tr, 0.5, np.nan)).astype(np.float32)
    store('clv5', rmean(clv, 5, 3))
    store('hl20', rmean(Hq / Lq - 1, 20, 10))
del Hq, Lq, Ct, clv, r1, Cff
tick('stage5 features')

U = U0.copy()
del seas
for f in range(NF):
    U &= RK[f] != 255
npool = U.sum(1)
t0 = int(np.nonzero(npool >= 200)[0][0])
rows = np.arange(T)
SEG = {'F': (rows >= t0 + 2) & (rows <= iF), 'V': rows >= iF + 2}
P('pool: first day %s (row %d); F rows %d (%s..%s); V rows %d (%s..%s); mean names/day F %.0f, V %.0f' % (
    dates[t0], t0, SEG['F'].sum(), dates[t0 + 2], dates[iF], SEG['V'].sum(), dates[iF + 2], dates[-1],
    npool[SEG['F']].mean(), npool[SEG['V']].mean()))
Off = ffill(OQ)
oo = np.full((T, N), np.nan, np.float32)
oo[:-1] = Off[1:] / Off[:-1] - 1
del Off
ar = rows[:, None]


def real_ret(H, void=False):
    b = nbuy[1:T + 1].copy()
    okb = b < T
    if void:
        okb &= (b == ar + 1)
    s_ = np.take_along_axis(nsell, np.minimum(np.where(okb, b, 0) + H, T), axis=0)
    ok = okb & (s_ < T)
    ok[:iF + 1] &= s_[:iF + 1] <= iF      # fit-segment labels mature inside the fit segment
    R = np.take_along_axis(OQ, np.where(ok, s_, 0), axis=0) / np.take_along_axis(OQ, np.where(ok, b, 0), axis=0) - 1
    R[~ok] = np.nan
    return R.astype(np.float32)


def relative(R):
    X = np.full((T, N), np.nan, np.float32)
    for t in range(t0, T):
        v = U[t] & np.isfinite(R[t])
        if v.sum() < 200:
            continue
        X[t, v] = R[t, v] - R[t, v].mean(dtype=np.float64)
    return X


# ---------------- per-feature tables (H = 5, 20) ----------------
HS = (5, 20)
ICs = np.full((2, T, NF), np.nan)
QS = np.zeros((2, T, NF, 20), np.float32)
QC = np.zeros((2, T, NF, 20), np.float32)
off20 = (20 * np.arange(NF))[:, None]
XREL = {}
for hi, H in enumerate(HS):
    X = relative(real_ret(H))
    for t in range(t0, T):
        v = np.isfinite(X[t])
        n = int(v.sum())
        if n < 200:
            continue
        r = X[t, v].astype(np.float64)
        Bq = RK[:, t, v]
        B = Bq.astype(np.float64)
        rr = np.empty(n)
        rr[np.argsort(r)] = np.arange(n)
        rr -= rr.mean()
        Bc = B - B.mean(1, keepdims=True)
        with np.errstate(invalid='ignore', divide='ignore'):
            ICs[hi, t] = (Bc @ rr) / np.sqrt((Bc * Bc).sum(1) * (rr @ rr))
        k = (Bq // 10 + off20).ravel()
        QS[hi, t] = np.bincount(k, weights=np.tile(r, NF), minlength=600).reshape(NF, 20)
        QC[hi, t] = np.bincount(k, minlength=600).reshape(NF, 20)
    if H == 5:
        XREL[H] = X
del X
tick('per-feature stats')


def zmean(hi, f, lo, up, seg):
    s = QS[hi, :, f, lo:up].sum(1)
    c = QC[hi, :, f, lo:up].sum(1)
    ok = seg & (c > 0)
    return (s[ok] / c[ok]).mean() * 1e4 / HS[hi]


segF, segV = SEG['F'], SEG['V']
icF = np.array([[np.nanmean(ICs[hi, segF, f]) for f in range(NF)] for hi in range(2)])
icV = np.array([[np.nanmean(ICs[hi, segV, f]) for f in range(NF)] for hi in range(2)])
sg = np.sign(icF)
P('\nTABLE 1  feature | class | rank IC 5d F V | rank IC 20d F V | dir5 dir20 | long-side drift bp/day (direction = sign of F rank IC at that horizon):'
  ' 5d top10% F V ; 5d top30% F V ; 20d top10% F V ; 20d top30% F V')
LS = np.zeros((NF, 2, 2, 2))      # feature, horizon, zone(10/30), seg
for f in range(NF):
    out = []
    for hi in range(2):
        for zi, w in enumerate((2, 6)):
            lo, up = (20 - w, 20) if sg[hi, f] > 0 else (0, w)
            for si, seg in enumerate((segF, segV)):
                LS[f, hi, zi, si] = zmean(hi, f, lo, up, seg)
            out.append('%6.2f %6.2f' % (LS[f, hi, zi, 0], LS[f, hi, zi, 1]))
    P('%-12s | %-6s | %+.3f %+.3f | %+.3f %+.3f | %+d %+d | %s' % (NAMES[f], CLSN[CLS[f]], icF[0, f], icV[0, f], icF[1, f], icV[1, f],
                                                                   sg[0, f], sg[1, f], ' ; '.join(out)))
P('\nTABLE 1b  by class: mean over features of long-side drift bp/day: class | 5d top10% F V | 5d top30% F V | 20d top10% F V | 20d top30% F V |'
  ' features with top30% > 0 in both segments (5d, 20d) of n')
for c in range(8):
    fs = [f for f in range(NF) if CLS[f] == c]
    m = LS[fs].mean(0)
    P('%-6s | %6.2f %6.2f | %6.2f %6.2f | %6.2f %6.2f | %6.2f %6.2f | %d %d of %d' % (
        CLSN[c], m[0, 0, 0], m[0, 0, 1], m[0, 1, 0], m[0, 1, 1], m[1, 0, 0], m[1, 0, 1], m[1, 1, 0], m[1, 1, 1],
        sum((LS[f, 0, 1] > 0).all() for f in fs), sum((LS[f, 1, 1] > 0).all() for f in fs), len(fs)))
P('count: features with top30%% > 0 in both segments: 5d %d, 20d %d ; with top30%% > 2bp in both: 5d %d, 20d %d ; top10%% > 0 both: 5d %d, 20d %d' % (
    sum((LS[f, 0, 1] > 0).all() for f in range(NF)), sum((LS[f, 1, 1] > 0).all() for f in range(NF)),
    sum((LS[f, 0, 1] > 2).all() for f in range(NF)), sum((LS[f, 1, 1] > 2).all() for f in range(NF)),
    sum((LS[f, 0, 0] > 0).all() for f in range(NF)), sum((LS[f, 1, 0] > 0).all() for f in range(NF))))
P('\nTABLE 2  20 bins of 5% (low feature value -> high), realisable relative drift bp/day: feature | H | seg | 20 values')
for f in range(NF):
    for hi in range(2):
        for sn, seg in (('F', segF), ('V', segV)):
            P('%-12s | %2d | %s | %s' % (NAMES[f], HS[hi], sn, ' '.join('%6.1f' % zmean(hi, f, k, k + 1, seg) for k in range(20))))

del QS, QC, ICs
# ---------------- composites ----------------
sgnE = np.sign(icF[0] + icF[1])
E = np.zeros((T, N), np.float32)
for c in range(8):
    acc = np.zeros((T, N), np.float32)
    m = 0
    for f in range(NF):
        if CLS[f] != c:
            continue
        xx = RK[f].astype(np.float32)
        acc += xx if sgnE[f] > 0 else 199 - xx
        m += 1
    E += acc / m
del acc, xx
E /= 8
E[~U] = np.nan
X10 = relative(real_ret(10))
XtX = np.zeros((NF, NF))
Xty = np.zeros(NF)
gx, gy = [], []
for t in range(t0, iF + 1):
    v = np.isfinite(X10[t])
    n = int(v.sum())
    if n < 200:
        continue
    y = np.empty(n)
    y[np.argsort(X10[t, v])] = (np.arange(n) + 0.5) / n - 0.5
    Xm = RK[:, t, v].T.astype(np.float64) / 200 - 0.4975
    XtX += Xm.T @ Xm
    Xty += Xm.T @ y
    if (t - t0) % 5 == 0:
        gx.append(RK[:, t, v].T.copy())
        gy.append(y.astype(np.float32))
w = np.linalg.solve(XtX + 1e-8 * np.trace(XtX) / NF * np.eye(NF), Xty)
L = np.full((T, N), np.nan, np.float32)
for t in range(t0, T):
    v = U[t]
    if v.sum() < 200:
        continue
    L[t, v] = (RK[:, t, v].T.astype(np.float64) / 200 - 0.4975) @ w
P('\nlinear weights (per unit of rank in [0,1]; label = 10d realisable relative return rank): ' +
  ' '.join('%s %+.3f' % (NAMES[f], w[f]) for f in range(NF)))
tick('E and L composites')
from sklearn.ensemble import HistGradientBoostingRegressor
GX = np.concatenate(gx).astype(np.float32)
GY = np.concatenate(gy)
del gx, gy
gb = HistGradientBoostingRegressor(random_state=0).fit(GX, GY)
P('GBDT: rows %d (every 5th fit day), iterations %d' % (GX.shape[0], gb.n_iter_))
del GX, GY
G = np.full((T, N), np.nan, np.float32)
for a in range(t0, T, 50):
    tt, ii = np.nonzero(U[a:min(a + 50, T)])
    if tt.size:
        G[tt + a, ii] = gb.predict(RK[:, tt + a, ii].T.astype(np.float32))
tick('GBDT fit and predict')
COMP = {'E': E, 'L': L, 'G': G}
PR = {k: pct_rank_rows(v, U & np.isfinite(v)) for k, v in COMP.items()}
CI = {}
for cn, pr in PR.items():
    ci = np.full((T, 30), -1, np.int16)
    for t in range(t0, T):
        idx = np.nonzero(np.isfinite(pr[t]))[0]
        if idx.size < 200:
            continue
        ci[t] = idx[np.argsort(-pr[t, idx], kind='stable')[:30]]
    CI[cn] = ci
COMP = {k: None for k in COMP}
del E, L, G


def bp(x, seg):
    v = seg & np.isfinite(x)
    return x[v].mean() * 1e4


P('\nTABLE 3  composite frozen-hold realisable relative drift bp/day: comp | H | top10% F V | top30% F V | top50 names F V | top100 names F V')
XREL[10] = X10
for H in (1, 2, 5, 10, 20):
    X = XREL[H] if H in XREL else relative(real_ret(H))
    for cn, pr in PR.items():
        dz = np.full((4, T), np.nan)
        for t in range(t0, T):
            v = np.isfinite(X[t]) & np.isfinite(pr[t])
            n = int(v.sum())
            if n < 200:
                continue
            xv = X[t, v]
            o = np.argsort(-pr[t, v], kind='stable')
            dz[0, t] = xv[o[:n // 10]].mean()
            dz[1, t] = xv[o[:3 * n // 10]].mean()
            dz[2, t] = xv[o[:50]].mean()
            dz[3, t] = xv[o[:100]].mean()
        P('%s | %2d | %s' % (cn, H, ' | '.join('%6.2f %6.2f' % (bp(dz[z], segF) / H, bp(dz[z], segV) / H) for z in range(4))))
del X
tick('composite frozen-hold')

P('\nTABLE 4  composite rank autocorrelation (lag 1,5,10,20) and residence days in the top 10% (pure / enter>0.9 exit<0.8): comp | seg | ...')
for cn, pr in PR.items():
    Zp = pr > 0.9
    Zh = np.zeros((T, N), bool)
    ent = np.zeros(T)
    cur = np.zeros(N, bool)
    for t in range(T):
        nxt = (pr[t] > 0.9) | (cur & (pr[t] >= 0.8))
        ent[t] = (nxt & ~cur).sum()
        cur = nxt
        Zh[t] = cur
    entp = np.zeros(T)
    entp[1:] = (Zp[1:] & ~Zp[:-1]).sum(1)
    for sn, seg in SEG.items():
        acs = []
        tt = np.nonzero(seg)[0]
        for lg in (1, 5, 10, 20):
            vals = []
            for t in tt[::5]:
                if t + lg >= T or not seg[t + lg]:
                    continue
                v = np.isfinite(pr[t]) & np.isfinite(pr[t + lg])
                vals.append(np.corrcoef(pr[t, v], pr[t + lg, v])[0, 1])
            acs.append(np.mean(vals))
        P('%s | %s | %s | %.1f / %.1f' % (cn, sn, ' '.join('%.3f' % a for a in acs), Zp[seg].sum() / entp[seg].sum(),
                                          Zh[seg].sum() / ent[seg].sum()))
    del Zp, Zh


def quota(score, K, H):
    held = np.zeros(N, bool)
    x = np.full(T, np.nan)
    buys = np.zeros(T)
    bs = np.full(T, np.nan)
    ms = np.full(T, np.nan)
    for t in range(t0, T - 1):
        e = t + 1
        sc = score[t]
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
        b = r[held & np.isfinite(r)].sum(dtype=np.float64) / K
        mm = r[u & np.isfinite(r)].mean(dtype=np.float64)
        x[e], buys[e], bs[e], ms[e] = b - mm, nb, b, mm
    return x, buys, bs, ms


CELLS = [(K, H) for K in (50, 80) for H in (5, 10, 20)]
COST = {'strict': lambda K: 25.2 + 0.2 * K, 'mine': lambda K: 9.2 + 0.2 * K}
QR = {}
P('\nTABLE 5  quota structure Q(K,H): comp | K | H | seg | gross bp/d | realised holding days | TE %/d | beta | strict: net %/yr, t551 | mine: net %/yr, t551 |'
  ' V only: net %/yr by year 2020 2021 2022 (strict ; mine)')
for cn, sc in PR.items():
    for (K, H) in CELLS:
        x, buys, bs, ms = quota(sc, K, H)
        for sn, seg in SEG.items():
            v = seg & np.isfinite(x)
            g = x[v].mean()
            te = x[v].std(ddof=1)
            to = buys[v].mean() / K
            beta = np.cov(bs[v], ms[v])[0, 1] / ms[v].var(ddof=1)
            row = []
            for cname, cf in COST.items():
                net = g - to * cf(K) * 1e-4 - 0.5e-4
                QR[(cn, K, H, sn, cname)] = (net * 25200, net / te * sqrt(551), g * 1e4, te * 100, 1 / to)
                row.append('%6.2f %5.2f' % (net * 25200, net / te * sqrt(551)))
            yr = ''
            if sn == 'V':
                yr = ' | ' + ' ; '.join(' '.join('%6.1f' % ((x[v & (years == y)].mean() - buys[v & (years == y)].mean() / K * cf(K) * 1e-4 - 0.5e-4)
                                                             * 25200) for y in (2020, 2021, 2022)) for cf in COST.values())
            P('%s | %2d | %2d | %s | %6.2f | %5.1f | %.3f | %.3f | %s%s' % (cn, K, H, sn, g * 1e4, 1 / to, te * 100, beta, ' | '.join(row), yr))
tick('quota structure')
P('\nVERDICT INPUT (section 5 reading rule): cost | cell chosen on F by median-of-three t | F median net, t | V median net, t | V per composite (E L G) net ; t')
SRdiv = {}
for cname in COST:
    best = max(CELLS, key=lambda kh: np.median([QR[(cn, kh[0], kh[1], 'F', cname)][1] for cn in COMP]))
    K, H = best
    fn = np.median([QR[(cn, K, H, 'F', cname)][0] for cn in COMP])
    ft = np.median([QR[(cn, K, H, 'F', cname)][1] for cn in COMP])
    vn = [QR[(cn, K, H, 'V', cname)][0] for cn in COMP]
    vt = [QR[(cn, K, H, 'V', cname)][1] for cn in COMP]
    SRdiv[cname] = np.median(vt) / sqrt(551 / 252)
    P('%-6s | K=%d H=%d | %.2f %.2f | %.2f %.2f | %s ; %s' % (cname, K, H, fn, ft, np.median(vn), np.median(vt),
                                                              ' '.join('%.2f' % a for a in vn), ' '.join('%.2f' % a for a in vt)))

# ---------------- appended part: tails, strata, concentrated-bet arithmetic ----------------
sg5 = sg[0]
SIGS = [(NAMES[f], HI[f] if sg5[f] > 0 else LO[f]) for f in range(NF)] + [('COMP_' + cn, CI[cn]) for cn in COMP]
n1 = np.minimum(np.ceil(0.01 * npool), 30).astype(int)
col = np.arange(30)[None, :]
bfail = np.zeros((T, N), bool)
bfail[:-1] = ~buy_ok[1:]
TAIL = {}
P('\nTABLE 6  long-side tail bets (buy at open t+1 or void; sell first sellable open >= h days later); per-bet relative return (minus same-day pool mean):'
  ' signal | set | h | seg | mean bp | median bp | sd % | win%% (rel > 17.2bp) | bets/day | void %% (buy not executable)')
for h in (1, 2, 3, 5):
    Xv = relative(real_ret(h, void=True))
    for sname, idx in SIGS:
        ok = idx >= 0
        ix = np.where(ok, idx, 0)
        vals = Xv[ar, ix].astype(np.float64)
        inu = U[ar, ix] & ok
        vals[~inu] = np.nan
        bf = bfail[ar, ix] & inu
        for setn, msk in (('top20', col < 20), ('top1%', col < n1[:, None])):
            for sn, seg in SEG.items():
                sel = msk & seg[:, None]
                a = vals[sel]
                a = a[np.isfinite(a)]
                nd = int((inu & sel).any(1).sum())
                void = bf[sel].sum() / max(inu[sel].sum(), 1)
                if sname.startswith('COMP_') and setn == 'top20':
                    A = np.where(sel, vals, np.nan)
                    cnt = np.isfinite(A).sum(1)
                    dm = np.nansum(A, 1)[cnt > 0] / cnt[cnt > 0]
                    TAIL[(sname, h, sn)] = (a.mean(), a.std(ddof=1), dm.var(ddof=1), cnt[cnt > 0].mean())
                    if h == 3:
                        TAIL[(sname, 'vals3')] = (vals, inu, bf)
                P('%-13s | %-5s | %d | %s | %7.1f | %7.1f | %5.2f | %4.1f | %5.1f | %4.1f' % (
                    sname, setn, h, sn, a.mean() * 1e4, np.median(a) * 1e4, a.std(ddof=1) * 100, 100 * (a > 17.2e-4).mean(),
                    a.size / max(nd, 1), 100 * void))
del Xv
tick('tail bets')

P('\nTABLE 7  concentrated structure arithmetic (composite, top 20 per day as the bet pool): comp | h | seg | m_gross bp | s % | rho | then for k=3 and k=5,'
  ' cost 27.2 (strict) and 17.2 (mine): net %/yr, SR, years to t=2')
CONC = {}
for cn in COMP:
    for h in (1, 2, 3, 5):
        for sn in SEG:
            mg, s, vday, nbar = TAIL[('COMP_' + cn, h, sn)]
            rho = (vday - s * s / nbar) / (s * s * (1 - 1 / nbar))
            rr_ = max(rho, 0.0)
            out = []
            for k in (3, 5):
                for c in (27.2, 17.2):
                    m = mg - c * 1e-4
                    sr = m / s * sqrt(252 * k / h) / sqrt(1 + (k - 1) * rr_)
                    CONC[(cn, h, sn, k, c)] = (25200 * m / h, sr)
                    out.append('k=%d c=%.1f: %7.1f %5.2f %s' % (k, c, 25200 * m / h, sr, ('%.1f' % (4 / sr ** 2)) if sr > 0 else 'never'))
            P('%s | %d | %s | %6.1f | %5.2f | %+.3f | %s' % (cn, h, sn, mg * 1e4, s * 100, rho, ' ; '.join(out)))
P('\nVERDICT INPUT (section 9 reading rule): cost | h chosen on F by median SR (k=5) | F median SR | V median net %/yr, SR (k=5) | V per composite SR | SR concentrated / SR diversified cell')
for c, cname in ((27.2, 'strict'), (17.2, 'mine')):
    hb = max((1, 2, 3, 5), key=lambda h: np.median([CONC[(cn, h, 'F', 5, c)][1] for cn in COMP]))
    fs = np.median([CONC[(cn, hb, 'F', 5, c)][1] for cn in COMP])
    vs = [CONC[(cn, hb, 'V', 5, c)][1] for cn in COMP]
    vn = [CONC[(cn, hb, 'V', 5, c)][0] for cn in COMP]
    P('%-6s | h=%d | %.2f | %.1f %.2f | %s | %.2f (diversified SR %.2f)' % (cname, hb, fs, np.median(vn), np.median(vs), ' '.join('%.2f' % a for a in vs),
                                                                           np.median(vs) / SRdiv[cname] if SRdiv[cname] != 0 else np.nan, SRdiv[cname]))

P('\nTABLE 8  strata (quintile 0 = smallest) by 60-day average amount (A) and float-cap proxy (C)')
qa = RK[NAMES.index('amt60')] // 40
qc = RK[NAMES.index('fcap')] // 40
X5 = XREL[5]
P('8a pool-wide: stratum | median day amount (CNY 10k) | median 50k / day amount %% | share of name-days with 60d avg amount < 5m (50k > 1%%) | < 10m (100k > 1%%) | < 16.7m | buy-not-executable %%')
for nm, qq in (('A', qa), ('C', qc)):
    for q in range(5):
        z = U & (qq == q) & segV[:, None]
        am = amt[z]
        am = am[np.isfinite(am)]
        a60 = amt60[z]
        P('%s%d V | %8.0f | %.3f | %.3f | %.3f | %.3f | %.2f' % (nm, q, np.median(am) / 1e4, 100 * np.median(5e4 / am), (a60 < 5e6).mean(), (a60 < 1e7).mean(),
                                                               (a60 < 1.67e7).mean(), 100 * bfail[z].mean()))
P('8b composite top 10%% (frozen 5 days, realisable): comp | seg | stratum: share of zone, drift bp/day, buy-not-executable %% ... (A0..A4 then C0..C4)')
for cn, pr in PR.items():
    Z = pr > 0.9
    for sn, seg in SEG.items():
        out = []
        for nm, qq in (('A', qa), ('C', qc)):
            for q in range(5):
                zz = Z & (qq == q) & seg[:, None]
                xs = np.where(zz & np.isfinite(X5), X5, 0).sum(1)
                cs_ = (zz & np.isfinite(X5)).sum(1)
                okd = cs_ > 0
                out.append('%s%d %.2f %6.2f %.2f' % (nm, q, zz.sum() / (Z & seg[:, None]).sum(), (xs[okd] / cs_[okd]).mean() * 1e4 / 5, 100 * bfail[zz].mean()))
        P('%s | %s | %s' % (cn, sn, ' ; '.join(out)))
P('8c composite top 20 per day (h=3, void convention): comp | seg | stratum: share of bets, mean per-bet relative bp, void %% ... (A0..A4 then C0..C4)')
for cn in COMP:
    vals, inu, bf = TAIL[('COMP_' + cn, 'vals3')]
    ix = np.where(CI[cn] >= 0, CI[cn], 0)
    for sn, seg in SEG.items():
        out = []
        sel0 = inu & (col < 20) & seg[:, None]
        for nm, qq in (('A', qa), ('C', qc)):
            qv = qq[ar, ix]
            for q in range(5):
                sel = sel0 & (qv == q)
                a = vals[sel]
                a = a[np.isfinite(a)]
                out.append('%s%d %.2f %7.1f %.1f' % (nm, q, sel.sum() / max(sel0.sum(), 1), a.mean() * 1e4 if a.size else np.nan, 100 * bf[sel].sum() / max(sel.sum(), 1)))
        P('%s | %s | %s' % (cn, sn, ' ; '.join(out)))
finish()
