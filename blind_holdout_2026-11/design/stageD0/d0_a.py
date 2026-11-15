# d0_a.py -- 丁零 run A: what a default boosted tree assumes vs what the panel looks like (items a, b, d, e, f).
# Training rows only (<= 2022-12-30): the c0 feature prefix is executed verbatim (common.load slices and asserts).
ROOTD = r'<BLIND_ROOT>\design'
src = open(ROOTD + r'\stageC0\c0_measure.py', encoding='utf-8').read()
src = src[:src.index('# ---------------- per-feature tables')]
_a = "C.OUTDIR = C.ROOT + r'\\design\\stageC0'"
_b = "start('c0_measure')"
assert src.count(_a) == 1 and src.count(_b) == 1
src = src.replace(_a, "C.OUTDIR = C.ROOT + r'\\design\\stageD0'").replace(_b, "start('d0_a')")
exec(compile(src, 'c0_prefix', 'exec'))
import gc
from sklearn.ensemble import HistGradientBoostingRegressor as HGB

segF, segV = SEG['F'], SEG['V']
lastrow = T - 1 - np.argmax(tr[::-1], axis=0)
everlisted = tr.any(0)
del LO, HI, oo, amt, amt60, U0, d, st
X10 = relative(real_ret(10))
X5 = relative(real_ret(5))
del OQ, nbuy, nsell, buy_ok, sell_ok, tr
gc.collect()
tick('labels; peak MB %.0f' % C.peak_mb()[0])


def urank(x):
    n = x.size
    r = np.empty(n)
    r[np.argsort(x, kind='stable')] = (np.arange(n) + 0.5) / n - 0.5
    return r


def nw_factor(x, L=20):
    x = x - x.mean()
    v = (x * x).mean()
    s = 1.0
    for k in range(1, L + 1):
        s += 2 * (1 - k / (L + 1)) * (x[k:] * x[:-k]).mean() / v
    return max(s, 0.1)


def pct(x):
    out = np.full((T, N), np.nan, np.float32)
    for t in range(t0, T):
        idx = np.nonzero(U[t] & np.isfinite(x[t]))[0]
        if idx.size < 200:
            continue
        o = np.argsort(x[t, idx], kind='stable')
        r = np.empty(idx.size)
        r[o] = (np.arange(idx.size) + 0.5) / idx.size
        out[t, idx] = r
    return out


def daily_corr(S, X, rank=True):
    out = np.full(T, np.nan)
    for t in range(t0, T):
        v = np.isfinite(X[t]) & np.isfinite(S[t])
        if v.sum() < 200:
            continue
        a, b = S[t, v].astype(np.float64), X[t, v].astype(np.float64)
        if rank:
            a, b = urank(a), urank(b)
        out[t] = np.corrcoef(a, b)[0, 1]
    return out


def sm(x, seg):
    v = seg & np.isfinite(x)
    return x[v].mean()


# ------------------------------------------------------------------ (a) effective sample size
IC10 = np.full((T, NF), np.nan)
NN = np.zeros(T)
for t in range(t0, T):
    v = np.isfinite(X10[t])
    n = int(v.sum())
    if n < 200:
        continue
    rr = urank(X10[t, v])
    B = RK[:, t, v].astype(np.float64)
    Bc = B - B.mean(1, keepdims=True)
    IC10[t] = (Bc @ rr) / np.sqrt((Bc * Bc).sum(1) * (rr @ rr))
    NN[t] = n


def deff(ic, seg):
    ok = seg & np.isfinite(ic)
    x, nn = ic[ok], NN[ok]
    Td = x.size
    vd = x.var(ddof=1)
    fac = nw_factor(x)
    de_cs = vd * nn.mean()
    de = vd * fac / Td * nn.sum()
    return x.mean(), Td, nn.sum(), de_cs, fac, de, nn.sum() / de, x.mean() * sqrt(nn.sum()), x.mean() / sqrt(vd * fac / Td)


P('\nA1 design effect per feature on F (label = 10d realisable relative return rank): feature | mean IC | days | rows | deff cross-section (var_day*n) |'
  ' serial factor (NW 20 lags) | deff total | N_eff rows | naive t (IC*sqrt(rows)) | actual t')
DE = []
for f in range(NF):
    r = deff(IC10[:, f], segF)
    DE.append(r)
    P('%-12s | %+.4f | %d | %d | %7.1f | %5.2f | %8.1f | %7.0f | %7.1f | %6.2f' % ((NAMES[f],) + r))
DE = np.array(DE)
P('A1 summary F: rows %d ; deff cross-section median %.1f (min %.1f max %.1f) ; serial factor median %.2f ; deff total median %.0f (min %.0f max %.0f) ;'
  ' N_eff median %.0f (min %.0f max %.0f) ; N_eff/day median %.2f ; naive t / actual t median %.1f' % (
      DE[0, 2], np.median(DE[:, 3]), DE[:, 3].min(), DE[:, 3].max(), np.median(DE[:, 4]), np.median(DE[:, 5]), DE[:, 5].min(), DE[:, 5].max(),
      np.median(DE[:, 6]), DE[:, 6].min(), DE[:, 6].max(), np.median(DE[:, 6]) / DE[0, 1], np.median(np.abs(DE[:, 7] / DE[:, 8]))))
tick('a: design effect')

# default G, same rows / label / seed as c0
gx, gy, gd = [], [], []
for t in range(t0, iF + 1):
    v = np.isfinite(X10[t])
    n = int(v.sum())
    if n < 200:
        continue
    if (t - t0) % 5 == 0:
        y = np.empty(n)
        y[np.argsort(X10[t, v])] = (np.arange(n) + 0.5) / n - 0.5
        gx.append(RK[:, t, v].T.copy())
        gy.append(y.astype(np.float32))
        gd.append(np.full(n, t, np.int32))
GX = np.concatenate(gx).astype(np.float32)
GY = np.concatenate(gy)
GD = np.concatenate(gd)
del gx, gy, gd
vx, vy = [], []
for t in range(iF + 2, T, 5):
    v = np.isfinite(X10[t])
    if v.sum() < 200:
        continue
    vx.append(RK[:, t, v].T.copy())
    vy.append(urank(X10[t, v]).astype(np.float32))
VX = np.concatenate(vx).astype(np.float32)
VY = np.concatenate(vy)
del vx, vy


def staged_r2(model, X, y):
    out = []
    vy_ = ((y - y.mean()) ** 2).mean()
    for p in model.staged_predict(X):
        out.append(1 - ((p - y) ** 2).mean() / vy_)
    return np.array(out)


gb = HGB(random_state=0).fit(GX, GY)
P('\nA2 default GBDT: rows %d, iterations %d (max 100; early stopping on a random 10%% of ROWS)' % (GX.shape[0], gb.n_iter_))
tick('default fit; peak MB %.0f' % C.peak_mb()[0])
its = [1, 5, 10, 20, 30, 50, 75, 100]
sub = np.arange(0, GX.shape[0], 5)
r2_in = staged_r2(gb, GX[sub], GY[sub])
r2_v = staged_r2(gb, VX, VY)
vs = np.asarray(gb.validation_score_)
ts = np.asarray(gb.train_score_)
P('A2 R2 (rank label) of the default model, %% : in-sample F rows %.3f ; row-random validation (sklearn internal) %.3f ; internal train-score %.3f ; V rows %.3f' % (
    100 * r2_in[-1], 100 * (1 - vs[-1] / vs[0]), 100 * (1 - ts[-1] / ts[0]), 100 * r2_v[-1]))
P('A2 default model, V R2 %% by iteration ' + ' '.join('%d:%.3f' % (i, 100 * r2_v[min(i, len(r2_v)) - 1]) for i in its) + ' ; best iteration on V %d (%.3f)' % (
    int(np.argmax(r2_v)) + 1, 100 * r2_v.max()))
P('A2 default model, in-sample R2 %% by iteration ' + ' '.join('%d:%.3f' % (i, 100 * r2_in[min(i, len(r2_in)) - 1]) for i in its))
del VX, VY, GD
gc.collect()


def predict_all(model):
    G_ = np.full((T, N), np.nan, np.float32)
    for a in range(t0, T, 50):
        tt, ii = np.nonzero(U[a:min(a + 50, T)])
        if tt.size:
            G_[tt + a, ii] = model.predict(RK[:, tt + a, ii].T.astype(np.float32))
    return G_


G = predict_all(gb)
del gb
PRG = pct(G)

# ------------------------------------------------------------------ (b) signal-to-noise, tails
P('\nB1 predictable share: daily cross-sectional correlation of the default-model score with the realised relative return (mean over days) and its square')
for nm, X, H in (('5d', X5, 5), ('10d', X10, 10)):
    cp = daily_corr(G, X, rank=False)
    cr_ = daily_corr(G, X, rank=True)
    P('B1 %s | Pearson F %+.4f V %+.4f (R2 %% F %.3f V %.3f) | rank F %+.4f V %+.4f (R2 %% F %.3f V %.3f) | daily sd of rank IC V %.4f' % (
        nm, sm(cp, segF), sm(cp, segV), 100 * sm(cp, segF) ** 2, 100 * sm(cp, segV) ** 2, sm(cr_, segF), sm(cr_, segV),
        100 * sm(cr_, segF) ** 2, 100 * sm(cr_, segV) ** 2, np.nanstd(cr_[segV])))
    if H == 10:
        ICG10 = cr_
del G
mF = segF[:, None] & np.isfinite(X10)
for nm, X in (('5d', X5), ('10d', X10)):
    m_ = segF[:, None] & np.isfinite(X)
    y = X[m_].astype(np.float64)
    a2 = np.sort(y * y)[::-1]
    tot = a2.sum()
    P('B2 %s raw relative return, F rows %d: sd %.2f%% ; kurtosis %.1f ; share of sum of squares from the largest |y| 0.1%% of rows %.1f%%, 1%% %.1f%%, 5%% %.1f%% ;'
      ' |y| at 99%% quantile %.1f%%, 99.9%% %.1f%%' % (nm, y.size, 100 * y.std(), ((y - y.mean()) ** 4).mean() / y.var() ** 2, 100 * a2[:y.size // 1000].sum() / tot,
                                                    100 * a2[:y.size // 100].sum() / tot, 100 * a2[:y.size // 20].sum() / tot,
                                                    100 * sqrt(a2[y.size // 100]), 100 * sqrt(a2[y.size // 1000])))
y = X10[mF].astype(np.float64)
thr = np.quantile(np.abs(y), 0.99)
core = np.abs(y) <= thr


def best_split(S, n):
    cs, cn = np.cumsum(S)[:-1], np.cumsum(n)[:-1]
    St, nt = S.sum(), n.sum()
    with np.errstate(invalid='ignore', divide='ignore'):
        g = cs ** 2 / cn + (St - cs) ** 2 / (nt - cn) - St ** 2 / nt
    g[(cn < 20) | (nt - cn < 20)] = -1
    return int(np.argmax(g)) + 1, g.max(), g


P('\nB3 root split of a squared-loss tree on the RAW 10d relative return (F rows): feature | best split bin all rows | gain | best split bin without the 1%% largest |y| | gain |'
  ' gain of the all-rows split evaluated on core rows only / all-rows gain')
RS = []
for f in range(NF):
    b = RK[f][mF].astype(np.int64)
    Sa, na = np.bincount(b, weights=y, minlength=200), np.bincount(b, minlength=200).astype(np.float64)
    Sc, nc = np.bincount(b[core], weights=y[core], minlength=200), np.bincount(b[core], minlength=200).astype(np.float64)
    ka, ga, _ = best_split(Sa, na)
    kc, gc_, gcv = best_split(Sc, nc)
    RS.append((ka, ga, kc, gc_, gcv[ka - 1] / ga))
    P('%-12s | %3d | %.4f | %3d | %.4f | %.2f' % (NAMES[f], ka, ga, kc, gc_, gcv[ka - 1] / ga))
RS = np.array(RS)
oa, oc = np.argsort(-RS[:, 1]), np.argsort(-RS[:, 3])
ra, rc = np.empty(NF), np.empty(NF)
ra[oa] = np.arange(NF)
rc[oc] = np.arange(NF)
P('B3 summary: features whose best root split moves by >= 20 bins when the 1%% tail rows are removed: %d of 30 ; median (core gain at same split / all gain) %.2f ;'
  ' rank correlation of the 30 gains %.2f ; top-5 by gain all rows: %s ; core rows: %s ; total gain best feature all %.4f core %.4f ; tail threshold |y| > %.1f%%' % (
      (np.abs(RS[:, 0] - RS[:, 2]) >= 20).sum(), np.median(RS[:, 4]), np.corrcoef(ra, rc)[0, 1], ','.join(NAMES[i] for i in oa[:5]),
      ','.join(NAMES[i] for i in oc[:5]), RS[:, 1].max(), RS[:, 3].max(), 100 * thr))
del y, core, mF
gc.collect()
tick('b: snr and tails')

# ------------------------------------------------------------------ (d) extrapolation
cr = C.load('close_raw')['close_raw']
LOW2 = U & (cr < 2.0)
LOW15 = U & (cr < 1.5)
del cr
iR120, iR60, iLP = NAMES.index('ret120'), NAMES.index('ret60'), NAMES.index('logprice')
REG = [('R1 ret120 low 2.5% and price low 2.5%', U & (RK[iR120] <= 4) & (RK[iLP] <= 4)), ('R2 ret60 low 1%', U & (RK[iR60] <= 1)),
       ('R3 price low 1%', U & (RK[iLP] <= 1)), ('R4 close < 2 yuan', LOW2), ('R5 close < 1.5 yuan', LOW15)]
PX10 = pct(X10)


def topmask(PR, k=50):
    M = np.zeros((T, N), bool)
    for t in range(t0, T):
        idx = np.nonzero(np.isfinite(PR[t]))[0]
        if idx.size < 200:
            continue
        M[t, idx[np.argsort(-PR[t, idx], kind='stable')[:k]]] = True
    return M


TOP50 = topmask(PRG)
P('\nD1 extreme regions: region | seg | rows | %% of pool rows | rows per day | %% without a realisable 10d label | mean model percentile | mean realised percentile |'
  ' realised 10d relative mean bp, median bp | P(in model top 50) %% (pool base rate %%) | share of top-50 slots %%')
for name, Z in REG:
    for sn, seg in SEG.items():
        z = Z & seg[:, None]
        n = int(z.sum())
        pool = int((U & seg[:, None]).sum())
        if n == 0:
            P('%s | %s | 0 rows' % (name, sn))
            continue
        x = X10[z]
        x = x[np.isfinite(x)]
        P('%s | %s | %d | %.3f | %.1f | %.1f | %.3f | %.3f | %+.0f %+.0f | %.1f (%.1f) | %.1f' % (
            name, sn, n, 100 * n / pool, n / seg.sum(), 100 * (~np.isfinite(X10[z])).mean(), np.nanmean(PRG[z]), np.nanmean(PX10[z]) if x.size else np.nan,
            x.mean() * 1e4 if x.size else np.nan, np.median(x) * 1e4 if x.size else np.nan, 100 * TOP50[z].mean(), 100 * 50 * seg.sum() / pool,
            100 * (TOP50 & z).sum() / max((TOP50 & seg[:, None]).sum(), 1)))
ext = np.zeros((T, N), np.uint8)
for f in range(NF):
    ext += (((RK[f] <= 4) | (RK[f] >= 195)) & U).astype(np.uint8)
for sn, seg in SEG.items():
    a, b = ext[TOP50 & seg[:, None]], ext[U & seg[:, None]]
    P('D2 %s: number of the 30 features sitting in their outer 2.5%% tails: model top-50 mean %.2f (share with >= 5: %.1f%%) ; pool mean %.2f (share with >= 5: %.1f%%)' % (
        sn, a.mean(), 100 * (a >= 5).mean(), b.mean(), 100 * (b >= 5).mean()))
del ext
van = everlisted & (lastrow < T - 60)
for sn, cond in (('F', lastrow <= iF), ('V', lastrow > iF)):
    ids = np.nonzero(van & cond)[0]
    for (a, b) in ((60, 21), (20, 0)):
        vp, top, cnt, nolab, names = [], 0, 0, 0, 0
        for i in ids:
            rr_ = np.arange(max(lastrow[i] - a, 0), lastrow[i] - b + 1)
            rr_ = rr_[U[rr_, i]]
            if rr_.size == 0:
                continue
            names += 1
            vp.extend(PRG[rr_, i])
            top += TOP50[rr_, i].sum()
            cnt += rr_.size
            nolab += (~np.isfinite(X10[rr_, i])).sum()
        P('D3 names whose price series ends inside the data (last tradable row > 60 days before the end), series ending in %s: %d names, %d with pool rows in the window %d..%d days'
          ' before the end ; rows %d ; mean model percentile %.3f ; in model top-50 %.1f%% of rows ; without realisable 10d label %.1f%%' % (
              sn, ids.size, names, a, b, cnt, np.nanmean(vp) if cnt else np.nan, 100 * top / max(cnt, 1), 100 * nolab / max(cnt, 1)))
del PX10, LOW2, LOW15, REG
gc.collect()
tick('d: extrapolation')

# ------------------------------------------------------------------ (e) objective mismatch
KEYS = ['top20', 'top50', '51-100', '101-200', 'top10%', '10-30%', '30-70%', '70-90%', 'bot10%', 'bot50']


def bins_table(PR, X, H, seg):
    acc = {k: [] for k in KEYS}
    ict, icb, ica = [], [], []
    for t in np.nonzero(seg)[0]:
        v = np.isfinite(X[t]) & np.isfinite(PR[t])
        n = int(v.sum())
        if n < 200:
            continue
        xv = X[t, v]
        sv = PR[t, v]
        o = np.argsort(-sv, kind='stable')
        xs = xv[o]
        d1, d3, d7, d9 = n // 10, 3 * n // 10, 7 * n // 10, 9 * n // 10
        for k, sl in zip(KEYS, (slice(0, 20), slice(0, 50), slice(50, 100), slice(100, 200), slice(0, d1), slice(d1, d3), slice(d3, d7), slice(d7, d9),
                                slice(d9, n), slice(n - 50, n))):
            acc[k].append(xs[sl].mean())
        ica.append(np.corrcoef(urank(sv), urank(xv))[0, 1])
        ict.append(np.corrcoef(urank(sv[o[:d1]]), urank(xs[:d1]))[0, 1])
        icb.append(np.corrcoef(urank(sv[o[d9:]]), urank(xs[d9:]))[0, 1])
    return {k: np.mean(v_) * 1e4 / H for k, v_ in acc.items()}, np.mean(ica), np.mean(ict), np.mean(icb)


P('\nE1 default model, frozen-hold realisable relative drift bp/day by score zone: H | seg | ' + ' | '.join(KEYS) + ' | rank IC all | rank IC inside top 10% | inside bottom 10%')
BT = {}
for H, X in ((5, X5), (10, X10)):
    for sn, seg in SEG.items():
        bt, ia, it_, ib = bins_table(PRG, X, H, seg)
        BT[(H, sn)] = bt
        P('E1 %2d | %s | %s | %+.4f | %+.4f | %+.4f' % (H, sn, ' | '.join('%6.2f' % bt[k] for k in KEYS), ia, it_, ib))
SMraw = rmean(PRG, 5, 3)
PRS = pct(SMraw)
prev = np.full((T, N), np.nan, np.float32)
prev[1:] = SMraw[:-1]
INN = PRG - prev
del SMraw
TOP50S = topmask(PRS)
for sn, seg in SEG.items():
    tt = np.nonzero(seg)[0][:-1]
    keep = np.mean([(TOP50[t] & TOP50[t + 1]).sum() / 50 for t in tt])
    keeps = np.mean([(TOP50S[t] & TOP50S[t + 1]).sum() / 50 for t in tt])
    keep10 = np.mean([(TOP50[t] & TOP50[t + 10]).sum() / 50 for t in tt[:-10]])
    b5, _, _, _ = bins_table(PRS, X5, 5, seg)
    b10, _, _, _ = bins_table(PRS, X10, 10, seg)
    P('E2 %s: top-50 set kept next day: raw score %.3f, 5-day mean of score %.3f ; kept after 10 days (raw) %.3f ; top-50 drift bp/day raw H5 %.2f H10 %.2f ; smoothed H5 %.2f H10 %.2f' % (
        sn, keep, keeps, keep10, BT[(5, sn)]['top50'], BT[(10, sn)]['top50'], b5['top50'], b10['top50']))
for nm, S in (('score', PRG), ('yesterday 5d mean', prev), ('innovation = score - yesterday 5d mean', INN)):
    c5 = daily_corr(S, X5)
    P('E3 rank IC with 5d label, %s: F %+.4f V %+.4f' % (nm, sm(c5, segF), sm(c5, segV)))
del prev, INN, PRS, TOP50S
gc.collect()
tick('e: mismatch')

# ------------------------------------------------------------------ (f) style contamination
iFC, iAM = NAMES.index('fcap'), NAMES.index('amt60')
cellid = (RK[iAM] // 40).astype(np.uint8) * 5 + (RK[iFC] // 40).astype(np.uint8)
FC = np.where(U, RK[iFC], np.nan).astype(np.float32)
AM = np.where(U, RK[iAM], np.nan).astype(np.float32)


def wrank(x, c):
    n = x.size
    o = np.lexsort((x, c))
    cnt = np.bincount(c, minlength=25)
    st_ = np.cumsum(cnt) - cnt
    r = np.empty(n)
    co = c[o]
    r[o] = (np.arange(n) - st_[co] + 0.5) / cnt[co] - 0.5
    return r


def style_report(tag, PR):
    cf, ca = daily_corr(PR, FC), daily_corr(PR, AM)
    P('F1 %s: daily rank correlation of score with cap proxy: F %+.3f V %+.3f ; with 60d amount: F %+.3f V %+.3f' % (tag, sm(cf, segF), sm(cf, segV), sm(ca, segF), sm(ca, segV)))
    for H, X in ((5, X5), (10, X10)):
        for sn, seg in SEG.items():
            g, s_, w2, wic, q0c, q0a, sub_g, sub_dec, sub_sm = [], [], [], [], [], [], [], [], []
            for t in np.nonzero(seg)[0]:
                v = np.isfinite(X[t]) & np.isfinite(PR[t])
                n = int(v.sum())
                if n < 200:
                    continue
                x = X[t, v].astype(np.float64)
                c = cellid[t, v].astype(np.int64)
                sc = PR[t, v]
                cnt = np.bincount(c, minlength=25)
                cm = np.bincount(c, weights=x, minlength=25) / np.maximum(cnt, 1)
                o = np.argsort(-sc, kind='stable')[:50]
                g.append(x[o].mean())
                s_.append(cm[c[o]].mean())
                q0c.append((RK[iFC, t, v][o] < 40).mean())
                q0a.append((RK[iAM, t, v][o] < 40).mean())
                ws, wx = wrank(sc, c), wrank(x, c)
                wic.append(np.corrcoef(ws, wx)[0, 1])
                sel = ws >= (0.5 - 2.0 / np.maximum(cnt[c], 2))
                w2.append((x[sel] - cm[c[sel]]).mean())
                fc = RK[iFC, t, v]
                dm = fc < 20
                xd, sd_ = x[dm], sc[dm]
                sub_dec.append(xd.mean())
                sub_g.append(xd[np.argsort(-sd_, kind='stable')[:50]].mean())
                sub_sm.append(x[fc <= 3].mean())
            g, s_ = np.array(g), np.array(s_)
            w = g - s_
            k_ = 1e4 / H
            P('F2 %s | H %2d | %s | top-50: gross %.2f = style(25 cells) %.2f + within-cell %.2f bp/day (t of within %.2f) ; top-50 in smallest cap quintile %.2f, smallest amount quintile %.2f ;'
              ' within-cell rank IC %+.4f ; best 2 per cell (about 50 names, style-neutral): within %.2f bp/day (t %.2f) ;'
              ' smallest cap decile: decile mean %.2f, model top-50 inside the decile %.2f, smallest 2%% by cap (about 50 names) %.2f' % (
                  tag, H, sn, g.mean() * k_, s_.mean() * k_, w.mean() * k_, w.mean() / w.std(ddof=1) * sqrt(w.size / H), np.mean(q0c), np.mean(q0a), np.mean(wic),
                  np.mean(w2) * k_, np.mean(w2) / np.std(w2, ddof=1) * sqrt(len(w2) / H), np.mean(sub_dec) * k_, np.mean(sub_g) * k_, np.mean(sub_sm) * k_))


style_report('default G', PRG)
tick('f: default style report')
el = time.time() - C._T0
if el < 215:
    gyn = []
    for t in range(t0, iF + 1):
        v = np.isfinite(X10[t])
        if v.sum() < 200:
            continue
        if (t - t0) % 5 == 0:
            gyn.append(wrank(X10[t, v].astype(np.float64), cellid[t, v].astype(np.int64)).astype(np.float32))
    GYN = np.concatenate(gyn)
    assert GYN.size == GY.size
    del gyn, GY, PRG, TOP50
    gc.collect()
    gbN = HGB(random_state=0).fit(GX, GYN)
    P('\nF3 cell-neutral-label GBDT (label = rank inside the same amount-quintile x cap-quintile cell): rows %d, iterations %d' % (GX.shape[0], gbN.n_iter_))
    del GX, GYN
    GN = predict_all(gbN)
    del gbN
    PRN = pct(GN)
    del GN
    tick('f: neutral fit; peak MB %.0f' % C.peak_mb()[0])
    if time.time() - C._T0 < 262:
        style_report('cell-neutral label', PRN)
else:
    P('F3 skipped: elapsed %.0f s' % el)
finish()
