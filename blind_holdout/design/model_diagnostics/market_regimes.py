# market_regimes.py -- run D: the same quantities split by market regime inside the training data (<= 2022-12-30). Model-free where possible.
ROOTD = r'<BLIND_ROOT>\design'
src = open(ROOTD + r'\boosted_tree_measurements\training_measurements.py', encoding='utf-8').read()
src = src[:src.index('# ---------------- per-feature tables')]
_a = "C.OUTDIR = C.ROOT + r'\\design\\boosted_tree_measurements'"
_b = "start('c0_measure')"
assert src.count(_a) == 1 and src.count(_b) == 1
src = src.replace(_a, "C.OUTDIR = C.ROOT + r'\\design\\model_diagnostics'").replace(_b, "start('d0_d')")
exec(compile(src, 'c0_prefix', 'exec'))
import gc

lastrow = T - 1 - np.argmax(tr[::-1], axis=0)
everlisted = tr.any(0)
del LO, HI, oo, amt, amt60, U0, d, st, tr


def ret_sell(H):
    b = nbuy[1:T + 1].copy()
    okb = b < T
    s_ = np.take_along_axis(nsell, np.minimum(np.where(okb, b, 0) + H, T), axis=0)
    ok = okb & (s_ < T)
    R = np.take_along_axis(OQ, np.where(ok, s_, 0), axis=0) / np.take_along_axis(OQ, np.where(ok, b, 0), axis=0) - 1
    R[~ok] = np.nan
    return R.astype(np.float32)


X10 = relative(ret_sell(10))
X5 = relative(ret_sell(5))
del OQ, nbuy, nsell, buy_ok, sell_ok
cr = C.load('close_raw')['close_raw']
LOW15 = U & (cr < 1.5)
LOW2 = U & (cr < 2.0)
del cr
gc.collect()
tick('labels')


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


REGS = [('2015', 2015, 2015), ('2016-2018', 2016, 2018), ('2019-2021', 2019, 2021), ('2022', 2022, 2022), ('2015-2022', 2015, 2022)]


def rmask(a, b):
    return (years >= a) & (years <= b) & (rows >= t0 + 2)


IC10 = np.full((T, NF), np.nan)
IC5 = np.full((T, NF), np.nan)
NN = np.zeros(T)
for t in range(t0, T):
    for X, IC in ((X10, IC10), (X5, IC5)):
        v = np.isfinite(X[t])
        n = int(v.sum())
        if n < 200:
            continue
        rr = urank(X[t, v])
        B = RK[:, t, v].astype(np.float64)
        Bc = B - B.mean(1, keepdims=True)
        IC[t] = (Bc @ rr) / np.sqrt((Bc * Bc).sum(1) * (rr @ rr))
        if X is X10:
            NN[t] = n

P('\nRA effective sample size by regime (30 features, 10d rank label): regime | days | rows | deff cross-section median | serial factor median | deff total median | N_eff per day median |'
  ' names per day | effective independent names per day (n / deff cross-section)')
for nm, a, b in REGS:
    m = rmask(a, b)
    out = []
    for f in range(NF):
        ok = m & np.isfinite(IC10[:, f])
        x, nn = IC10[ok, f], NN[ok]
        vd = x.var(ddof=1)
        fac = nw_factor(x)
        out.append((vd * nn.mean(), fac, vd * nn.mean() * fac, nn.mean() / (vd * nn.mean() * fac), nn.mean(), x.size, nn.sum()))
    o = np.array(out)
    P('RA %-9s | %d | %d | %.1f | %.2f | %.0f | %.2f | %.0f | %.0f' % (nm, o[0, 5], o[0, 6], np.median(o[:, 0]), np.median(o[:, 1]), np.median(o[:, 2]), np.median(o[:, 3]),
                                                                 o[0, 4], o[0, 4] / np.median(o[:, 0])))

P('\nRB tails by regime (raw 10d relative return): regime | sd % | kurtosis | share of sum of squares from the largest 1% of rows %')
for nm, a, b in REGS:
    m = rmask(a, b)[:, None] & np.isfinite(X10)
    y = X10[m].astype(np.float64)
    a2 = np.sort(y * y)[::-1]
    P('RB %-9s | %.2f | %.1f | %.1f' % (nm, 100 * y.std(), ((y - y.mean()) ** 4).mean() / y.var() ** 2, 100 * a2[:y.size // 100].sum() / a2.sum()))

mall = rmask(2015, 2022)
icall = np.nanmean(IC5[mall], 0)
STAB = ['ret1', 'ret3', 'ret5', 'intra5', 'intra20', 'abn_turn', 'turn5_60', 'turn20', 'pv_corr20', 'dist_low60', 'clv5', 'streak', 'lu20', 'max20', 'vol20', 'downshare20',
        'hl20', 'ret20', 'ret60']
UNST = ['fcap', 'logprice', 'amihud20', 'amt60', 'dist_high60', 'dist_high250', 'mom250_20', 'ret120', 'ld20', 'ovn5', 'ovn20']
iS = [NAMES.index(k) for k in STAB]
iU = [NAMES.index(k) for k in UNST]
SHOW = ['fcap', 'amt60', 'logprice', 'dist_high250', 'mom250_20', 'ret120', 'ret20', 'max20', 'turn20', 'vol20', 'lu20']
P('\nRC relations by regime (rank IC x100 with the 5d label): regime | features whose sign differs from the 2015-2022 sign (names) | directional IC of the 19 heat features |'
  ' directional IC of the 11 size/price/position features | ' + ' '.join(SHOW))
for nm, a, b in REGS:
    ic = np.nanmean(IC5[rmask(a, b)], 0)
    fl = np.nonzero(np.sign(ic) != np.sign(icall))[0]
    P('RC %-9s | %d (%s) | %.2f | %.2f | %s' % (nm, fl.size, ','.join(NAMES[i] for i in fl), 100 * (np.sign(icall[iS]) * ic[iS]).mean(), 100 * (np.sign(icall[iU]) * ic[iU]).mean(),
                                               ' '.join('%+.2f' % (100 * ic[NAMES.index(k)]) for k in SHOW)))

iR120, iLP = NAMES.index('ret120'), NAMES.index('logprice')
REGN = [('ret120 low 2.5% and price low 2.5%', U & (RK[iR120] <= 4) & (RK[iLP] <= 4)), ('close < 2 yuan', LOW2), ('close < 1.5 yuan', LOW15)]
P('\nRD extreme regions by regime: region | regime | rows per day | % without a realisable 10d label | realised 10d relative mean bp, median bp')
for name, Z in REGN:
    for nm, a, b in REGS:
        m = rmask(a, b)
        z = Z & m[:, None]
        n = int(z.sum())
        x = X10[z]
        x = x[np.isfinite(x)]
        P('RD %s | %-9s | %.1f | %.1f | %+.0f %+.0f' % (name, nm, n / m.sum(), 100 * (~np.isfinite(X10[z])).mean() if n else np.nan, x.mean() * 1e4 if x.size else np.nan,
                                                       np.median(x) * 1e4 if x.size else np.nan))
van = everlisted & (lastrow < T - 60)
P('RD names whose price series ends inside the data: regime of the last tradable day | names | with pool rows in the last 20 days | rows | % of those rows without a realisable 10d label')
for nm, a, b in REGS:
    ids = np.nonzero(van & (years[lastrow] >= a) & (years[lastrow] <= b))[0]
    cnt, nol, names = 0, 0, 0
    for i in ids:
        rr_ = np.arange(max(lastrow[i] - 20, 0), lastrow[i] + 1)
        rr_ = rr_[U[rr_, i]]
        if rr_.size:
            names += 1
            cnt += rr_.size
            nol += (~np.isfinite(X10[rr_, i])).sum()
    P('RD ended %-9s | %d | %d | %d | %.1f' % (nm, ids.size, names, cnt, 100 * nol / max(cnt, 1)))
tick('a-d by regime')

# model-free heat composite: mean oriented rank of the 19 heat features (orientation = sign of the 2015-2022 IC; all 19 keep one sign in all 8 years)
HC = np.zeros((T, N), np.float32)
for k in iS:
    xx = RK[k].astype(np.float32)
    HC += xx if icall[k] > 0 else 199 - xx
HC /= len(iS)
HC[~U] = np.nan
del xx
iFC, iAM = NAMES.index('fcap'), NAMES.index('amt60')
KE = ['top10', 'bot10', 'top50', 'bot50', 'ic', 'ict', 'icb', 'style50', 'dec', 'dtop', 'dbot', 'dex', 'sm2', 'cfc', 'cam']
D_ = {k: np.full(T, np.nan) for k in KE}
for t in range(t0, T):
    v = np.isfinite(X5[t]) & np.isfinite(HC[t])
    n = int(v.sum())
    if n < 200:
        continue
    x = X5[t, v].astype(np.float64)
    s = HC[t, v]
    fc = RK[iFC, t, v]
    am = RK[iAM, t, v]
    o = np.argsort(-s, kind='stable')
    d1 = n // 10
    D_['top10'][t], D_['bot10'][t], D_['top50'][t], D_['bot50'][t] = x[o[:d1]].mean(), x[o[n - d1:]].mean(), x[o[:50]].mean(), x[o[n - 50:]].mean()
    us = urank(s)
    D_['ic'][t] = np.corrcoef(us, urank(x))[0, 1]
    D_['ict'][t] = np.corrcoef(urank(s[o[:d1]]), urank(x[o[:d1]]))[0, 1]
    D_['icb'][t] = np.corrcoef(urank(s[o[n - d1:]]), urank(x[o[n - d1:]]))[0, 1]
    c = (am // 40).astype(np.int64) * 5 + (fc // 40).astype(np.int64)
    cm = np.bincount(c, weights=x, minlength=25) / np.maximum(np.bincount(c, minlength=25), 1)
    D_['style50'][t] = cm[c[o[:50]]].mean()
    dm = fc < 20
    xd, sd_ = x[dm], s[dm]
    od = np.argsort(-sd_, kind='stable')
    D_['dec'][t], D_['dtop'][t], D_['dbot'][t], D_['dex'][t] = xd.mean(), xd[od[:50]].mean(), xd[od[-50:]].mean(), xd[od[:-50]].mean()
    D_['sm2'][t] = x[fc <= 3].mean()
    D_['cfc'][t] = np.corrcoef(us, urank(fc.astype(np.float64)))[0, 1]
    D_['cam'][t] = np.corrcoef(us, urank(am.astype(np.float64)))[0, 1]
P('\nRE heat composite (no fitting), 5d frozen hold, bp/day relative to the pool unless noted: regime | top 10% | bottom 10% | long share of the decile spread % | top 50 | bottom 50 |'
  ' rank IC all / inside top 10% / inside bottom 10% | top-50 = style (25 cells) + within | corr of composite with cap proxy, with 60d amount ||'
  ' smallest cap decile: decile mean | smallest 2% | relative to the decile mean: composite worst 50, composite best 50, gain from dropping the worst 50, smallest 2%')
YEARS8 = [(str(yv), yv, yv) for yv in range(2015, 2023)]
for nm, a, b in REGS + YEARS8:
    m = rmask(a, b)
    g = lambda k: np.nanmean(D_[k][m])
    k_ = 1e4 / 5
    P('RE %-9s | %6.2f | %6.2f | %3.0f | %6.2f | %6.2f | %+.3f / %+.3f / %+.3f | %.2f = %.2f + %.2f | %+.2f %+.2f || %6.2f | %6.2f | %6.2f %6.2f %6.2f %6.2f' % (
        nm, g('top10') * k_, g('bot10') * k_, 100 * g('top10') / (g('top10') - g('bot10')), g('top50') * k_, g('bot50') * k_, g('ic'), g('ict'), g('icb'),
        g('top50') * k_, g('style50') * k_, (g('top50') - g('style50')) * k_, g('cfc'), g('cam'), g('dec') * k_, g('sm2') * k_,
        (g('dbot') - g('dec')) * k_, (g('dtop') - g('dec')) * k_, (g('dex') - g('dec')) * k_, (g('sm2') - g('dec')) * k_))
finish()
