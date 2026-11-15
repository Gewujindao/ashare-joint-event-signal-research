# d0_b.py -- 丁零 run B: relations change over time (item c), row-wise early stopping (item a), raw-label tails (item b).
# Training rows only (<= 2022-12-30): the c0 feature prefix is executed verbatim (common.load slices and asserts).
ROOTD = r'<BLIND_ROOT>\design'
src = open(ROOTD + r'\stageC0\c0_measure.py', encoding='utf-8').read()
src = src[:src.index('# ---------------- per-feature tables')]
_a = "C.OUTDIR = C.ROOT + r'\\design\\stageC0'"
_b = "start('c0_measure')"
assert src.count(_a) == 1 and src.count(_b) == 1
src = src.replace(_a, "C.OUTDIR = C.ROOT + r'\\design\\stageD0'").replace(_b, "start('d0_b')")
exec(compile(src, 'c0_prefix', 'exec'))
import gc
from sklearn.ensemble import HistGradientBoostingRegressor as HGB

segF, segV = SEG['F'], SEG['V']
del LO, HI, oo, amt, amt60, U0, d, st, tr


def ret_sell(H):
    b = nbuy[1:T + 1].copy()
    okb = b < T
    s_ = np.take_along_axis(nsell, np.minimum(np.where(okb, b, 0) + H, T), axis=0)
    ok = okb & (s_ < T)
    R = np.take_along_axis(OQ, np.where(ok, s_, 0), axis=0) / np.take_along_axis(OQ, np.where(ok, b, 0), axis=0) - 1
    R[~ok] = np.nan
    return R.astype(np.float32), np.where(ok, s_, 10 ** 6).astype(np.int32)


R10, S10 = ret_sell(10)
XA10 = relative(R10)
del R10
R5, _s = ret_sell(5)
XA5 = relative(R5)
del R5, _s, OQ, nbuy, nsell, buy_ok, sell_ok
gc.collect()
tick('labels; peak MB %.0f' % C.peak_mb()[0])


def urank(x):
    n = x.size
    r = np.empty(n)
    r[np.argsort(x, kind='stable')] = (np.arange(n) + 0.5) / n - 0.5
    return r


# ------------------------------------------------------------------ (c1) per-feature rank IC by year
IC5 = np.full((T, NF), np.nan)
for t in range(t0, T):
    v = np.isfinite(XA5[t])
    n = int(v.sum())
    if n < 200:
        continue
    rr = urank(XA5[t, v])
    B = RK[:, t, v].astype(np.float64)
    Bc = B - B.mean(1, keepdims=True)
    IC5[t] = (Bc @ rr) / np.sqrt((Bc * Bc).sum(1) * (rr @ rr))
YRS = list(range(2015, 2023))
YM = np.zeros((NF, 8))
YT = np.zeros((NF, 8))
for j, yv in enumerate(YRS):
    x = IC5[years == yv]
    m = np.nanmean(x, 0)
    YM[:, j] = m
    YT[:, j] = m / (np.nanstd(x, 0, ddof=1) / np.sqrt(np.isfinite(x).sum(0) / 5.0))
icF, icV = np.nanmean(IC5[segF], 0), np.nanmean(IC5[segV], 0)
P('\nC1 rank IC (x100) of each feature with the 5d realisable relative return, by year 2015..2022 | F mean | V mean | years with t>2 / t<-2 (t uses days/5)')
for f in range(NF):
    P('%-12s | %s | %+.2f | %+.2f | %d / %d' % (NAMES[f], ' '.join('%+6.2f' % (100 * a) for a in YM[f]), 100 * icF[f], 100 * icV[f], (YT[f] > 2).sum(), (YT[f] < -2).sum()))
flip = np.sign(icF) != np.sign(icV)
sig2 = (YT > 2).any(1) & (YT < -2).any(1)
same = (np.sign(YM) == np.sign(YM[:, :1])).all(1)
half = flip | (np.abs(icV) < 0.5 * np.abs(icF))
P('C1 summary: sign(F mean) != sign(V mean): %d of 30 (%s) ; significant in both directions in different years: %d (%s) ; same sign in all 8 years: %d (%s) ;'
  ' lost at least half of |IC| or flipped from F to V: %d ; mean |IC| F %.4f ; mean of sign(F)*IC in V %.4f (ratio %.2f) ;'
  ' correlation of the 30-feature IC vector between consecutive years: %s' % (
      flip.sum(), ','.join(NAMES[i] for i in np.nonzero(flip)[0]), sig2.sum(), ','.join(NAMES[i] for i in np.nonzero(sig2)[0]), same.sum(),
      ','.join(NAMES[i] for i in np.nonzero(same)[0]), half.sum(), np.abs(icF).mean(), (np.sign(icF) * icV).mean(), (np.sign(icF) * icV).mean() / np.abs(icF).mean(),
      ' '.join('%.2f' % np.corrcoef(YM[:, j], YM[:, j + 1])[0, 1] for j in range(7))))

# ------------------------------------------------------------------ (c2) can a failing relation be seen in advance
A_ = np.where(np.isfinite(IC5), IC5, 0.0)
cA = np.vstack([np.zeros((1, NF)), np.cumsum(A_, 0)])
cN = np.vstack([np.zeros((1, NF)), np.cumsum(np.isfinite(IC5).astype(np.float64), 0)])


def mwin(a, b):
    return (cA[b] - cA[a]) / np.maximum(cN[b] - cN[a], 1)


t250, t60, ex, fw = [], [], [], []
for t in range(t0 + 260, T - 61, 60):
    e = t - 6
    t250.append(mwin(e - 250, e))
    t60.append(mwin(e - 60, e))
    ex.append(mwin(t0, e))
    fw.append(mwin(t + 1, t + 61))
t250, t60, ex, fw = map(np.array, (t250, t60, ex, fw))
cc = lambda a, b: np.corrcoef(a.ravel(), b.ravel())[0, 1]
P('\nC2 feature-level: %d dates x 30 features. corr(trailing 250d IC, next 60d IC) %.3f ; corr(trailing 60d, next 60d) %.3f ; corr(expanding mean, next 60d) %.3f ;'
  ' corr(trailing 250d minus expanding, next 60d minus expanding) %.3f ; corr(trailing 60d minus expanding, next 60d minus expanding) %.3f ;'
  ' sign agreement with next 60d: trailing 250d %.3f, trailing 60d %.3f, expanding %.3f' % (
      t250.shape[0], cc(t250, fw), cc(t60, fw), cc(ex, fw), cc(t250 - ex, fw - ex), cc(t60 - ex, fw - ex),
      (np.sign(t250) == np.sign(fw)).mean(), (np.sign(t60) == np.sign(fw)).mean(), (np.sign(ex) == np.sign(fw)).mean()))
tV = np.nonzero(segV)[0]
e = tV - 6
w250 = (cA[e] - cA[e - 250]) / np.maximum(cN[e] - cN[e - 250], 1)
w60 = (cA[e] - cA[e - 60]) / np.maximum(cN[e] - cN[e - 60], 1)
wex = (cA[e] - cA[t0]) / np.maximum(cN[e] - cN[t0], 1)
icv = np.where(np.isfinite(IC5[tV]), IC5[tV], 0.0)
SCH = {'fixed sign from F': np.sign(icF)[None, :] * np.ones_like(icv), 'sign of expanding mean': np.sign(wex), 'sign of trailing 250d': np.sign(w250),
       'sign of trailing 60d': np.sign(w60), 'expanding sign, switched off when trailing 250d disagrees': np.sign(wex) * (np.sign(wex) == np.sign(w250)),
       'weights = expanding mean IC': wex / np.abs(wex).mean(1, keepdims=True), 'weights = trailing 250d IC': w250 / np.abs(w250).mean(1, keepdims=True),
       'weights = F mean IC': (icF / np.abs(icF).mean())[None, :] * np.ones_like(icv)}
P('C2 directional IC on V (mean over 30 features of weight x daily IC, x100): scheme | V | 2020 | 2021 | 2022')
for k, w in SCH.items():
    dday = (w * icv).mean(1)
    P('C2 %-58s | %.3f | %s' % (k, 100 * dday.mean(), ' | '.join('%.3f' % (100 * dday[years[tV] == yv].mean()) for yv in (2020, 2021, 2022))))
del cA, cN, A_
tick('c: feature drift')


# ------------------------------------------------------------------ fits
def build(a, e, label='rank', step=5):
    gx, gy, gd = [], [], []
    for t in range(a, e + 1):
        if (t - t0) % step:
            continue
        v = np.isfinite(XA10[t]) & (S10[t] <= e)
        n = int(v.sum())
        if n < 200:
            continue
        gx.append(RK[:, t, v].T.copy())
        gy.append(urank(XA10[t, v]).astype(np.float32) if label == 'rank' else XA10[t, v].copy())
        gd.append(np.full(n, t, np.int32))
    return np.concatenate(gx).astype(np.float32), np.concatenate(gy), np.concatenate(gd)


def predict_rows(model, S, lo, hi):
    for a in range(lo, hi + 1, 50):
        b = min(a + 50, hi + 1)
        tt, ii = np.nonzero(U[a:b])
        if tt.size:
            S[tt + a, ii] = model.predict(RK[:, tt + a, ii].T.astype(np.float32))


def evalS(S, mrows):
    ic, a5, a10, d5 = [], [], [], []
    for t in np.nonzero(mrows)[0]:
        s = S[t]
        v = np.isfinite(XA10[t]) & np.isfinite(s)
        if v.sum() >= 200:
            ic.append(np.corrcoef(urank(s[v]), urank(XA10[t, v]))[0, 1])
            a10.append(XA10[t, v][np.argsort(-s[v], kind='stable')[:50]].mean())
        v = np.isfinite(XA5[t]) & np.isfinite(s)
        n = int(v.sum())
        if n >= 200:
            x = XA5[t, v][np.argsort(-s[v], kind='stable')]
            a5.append(x[:50].mean())
            d5.append(x[:n // 10].mean())
    return np.mean(ic), np.mean(a5) * 1e4 / 5, np.mean(a10) * 1e4 / 10, np.mean(d5) * 1e4 / 5, np.array(ic)


def staged_r2(model, X, y):
    out = []
    vy_ = ((y - y.mean()) ** 2).mean()
    for p in model.staged_predict(X):
        out.append(1 - ((p - y) ** 2).mean() / vy_)
    return np.array(out)


def yrow(yv, last=True):
    idx = np.nonzero(years == yv)[0]
    return int(idx[-1] if last else idx[0])


def el():
    return time.time() - C._T0


vx, vy = [], []
for t in range(iF + 2, T, 10):
    v = np.isfinite(XA10[t])
    if v.sum() < 200:
        continue
    vx.append(RK[:, t, v].T.copy())
    vy.append(urank(XA10[t, v]).astype(np.float32))
VX = np.concatenate(vx).astype(np.float32)
VY = np.concatenate(vy)
del vx, vy
its = [1, 5, 10, 20, 30, 50, 75, 100]
SC = {}
GX, GY, GD = build(t0, iF)
gb = HGB(random_state=0).fit(GX, GY)
vs = np.asarray(gb.validation_score_)
r2_in = staged_r2(gb, GX[::3], GY[::3])
r2_v = staged_r2(gb, VX, VY)
P('\nA3 ONCE (fit days every 5th in F, rows %d, iterations %d): R2 %% in-sample %.3f ; row-random validation %.3f ; V %.3f ; V by iteration %s ; best on V %d (%.3f)' % (
    GX.shape[0], gb.n_iter_, 100 * r2_in[-1], 100 * (1 - vs[-1] / vs[0]), 100 * r2_v[-1],
    ' '.join('%d:%.3f' % (i, 100 * r2_v[min(i, len(r2_v)) - 1]) for i in its), int(np.argmax(r2_v)) + 1, 100 * r2_v.max()))
P('A3 ONCE row-random validation R2 %% by iteration ' + ' '.join('%d:%.3f' % (i, 100 * (1 - vs[min(i, len(vs) - 1)] / vs[0])) for i in its))
S = np.full((T, N), np.nan, np.float32)
predict_rows(gb, S, iF + 2, T - 1)
SC['ONCE 2015-2019'] = S
del gb
tick('ONCE fit')

S = np.full((T, N), np.nan, np.float32)
done = []
for yv in (2020, 2021, 2022):
    if el() > 200:
        break
    a = max(yrow(yv - 3, False), t0)
    gx_, gy_, _ = build(a, yrow(yv - 1))
    m_ = HGB(random_state=0).fit(gx_, gy_)
    predict_rows(m_, S, max(yrow(yv, False), iF + 2), yrow(yv))
    done.append('%d: rows %d, iter %d' % (yv, gx_.shape[0], m_.n_iter_))
    del gx_, gy_, m_
SC['ROLL3 (previous 3 years, refit yearly)'] = S
P('ROLL3 fits: ' + ' ; '.join(done))
tick('ROLL3 fits')

S = np.full((T, N), np.nan, np.float32)
S[iF + 2:yrow(2020) + 1] = SC['ONCE 2015-2019'][iF + 2:yrow(2020) + 1]
done = []
for yv in (2021, 2022):
    if el() > 190:
        break
    gx_, gy_, _ = build(t0, yrow(yv - 1))
    m_ = HGB(random_state=0).fit(gx_, gy_)
    predict_rows(m_, S, yrow(yv, False), yrow(yv))
    done.append('%d: rows %d, iter %d' % (yv, gx_.shape[0], m_.n_iter_))
    del gx_, gy_, m_
SC['EXPAND (all years so far, refit yearly)'] = S
P('EXPAND fits: ' + ' ; '.join(done))
tick('EXPAND fits')

if el() < 215:
    ud = np.unique(GD)
    k80 = int(0.8 * ud.size)
    trm, hom = GD <= ud[k80 - 2], GD >= ud[k80]
    gbB = HGB(random_state=0, early_stopping=False).fit(GX[trm], GY[trm])
    r2h = staged_r2(gbB, GX[hom], GY[hom])
    r2vB = staged_r2(gbB, VX, VY)
    r2iB = staged_r2(gbB, GX[trm][::3], GY[trm][::3])
    nhat = int(np.argmax(r2h)) + 1
    P('\nA4 day-block validation: fit on the first 80%% of F fit days (%d rows, to %s, no early stopping), held out the last 20%% (%d rows, from %s)' % (
        trm.sum(), dates[ud[k80 - 2]], hom.sum(), dates[ud[k80]]))
    P('A4 in-sample R2 %% by iteration ' + ' '.join('%d:%.3f' % (i, 100 * r2iB[i - 1]) for i in its))
    P('A4 held-out-days R2 %% by iteration ' + ' '.join('%d:%.3f' % (i, 100 * r2h[i - 1]) for i in its) + ' ; best iteration %d (%.3f)' % (nhat, 100 * r2h.max()))
    P('A4 V R2 %% by iteration ' + ' '.join('%d:%.3f' % (i, 100 * r2vB[i - 1]) for i in its) + ' ; at the held-out-best iteration %.3f ; best on V %d (%.3f)' % (
        100 * r2vB[nhat - 1], int(np.argmax(r2vB)) + 1, 100 * r2vB.max()))
    del gbB
    tick('day-block fit')
del VX, VY

if el() < 225:
    _x, GYR, _d = build(t0, iF, label='raw')
    assert GYR.size == GY.size
    del _x, _d
    gbR = HGB(random_state=0).fit(GX, GYR)
    S = np.full((T, N), np.nan, np.float32)
    predict_rows(gbR, S, iF + 2, T - 1)
    SC['RAW label (10d relative return, squared loss), 2015-2019'] = S
    P('RAW fit: rows %d, iterations %d' % (GX.shape[0], gbR.n_iter_))
    del gbR, GYR
    tick('RAW fit')
del GX, GY

P('\nC3 models on V (labels: 10d relative return rank unless noted; fit days every 5th): model | span | rank IC 10d | top-50 drift bp/day H=5 | H=10 | top 10% H=5')
for k, S in SC.items():
    for nm, mr in (('V', segV), ('2020', segV & (years == 2020)), ('2021', segV & (years == 2021)), ('2022', segV & (years == 2022))):
        if el() > 275 or not np.isfinite(S[mr]).any():
            continue
        r = evalS(S, mr)
        P('C3 %-58s | %s | %+.4f | %6.2f | %6.2f | %6.2f' % (k, nm, r[0], r[1], r[2], r[3]))
        if nm == 'V' and k.startswith('ONCE'):
            icm = r[4]
            a_, b_ = [], []
            for t in range(71, icm.size - 20, 20):
                a_.append(icm[t - 71:t - 11].mean())
                b_.append(icm[t:t + 20].mean())
            P('C3 ONCE model-level: corr(trailing 60d realised IC known at t, next 20d IC) over %d non-overlapping points %.3f ; daily IC mean %.4f sd %.4f ; share of days with IC < 0: %.3f' % (
                len(a_), np.corrcoef(a_, b_)[0, 1], icm.mean(), icm.std(), (icm < 0).mean()))
finish()
