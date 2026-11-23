# rank_residence.py — persistence of the signal ranks and residence time in the cold zone (training rows only).
# pure zone: rank <= q. hysteresis state: enter at rank <= q, leave at rank > 2q or when the name drops out of the pool.
# Drift of the held set: names in the set at close t-1 earn open t -> open t+1, relative to the pool mean.
from cold_signal_common import *

start('m2_residence')
B = build()
d, T, N, U, sig, OQ = B['d'], B['T'], B['N'], B['U'], B['sig'], B['OQ']
yrs = d['years']
per = periods(yrs)
Off, oo = oo_returns(OQ)
KEEP = {}
P('TABLE A  rank autocorrelation (every 5th day, names in the pool on both days): signal | lag 1, 5, 20, 40')
for sn in SIGS:
    R = sig[sn]
    row = []
    for lag in (1, 5, 20, 40):
        vals = []
        for t in range(0, T - lag, 5):
            v = U[t] & U[t + lag]
            if v.sum() < 200:
                continue
            vals.append(np.corrcoef(R[t, v], R[t + lag, v])[0, 1])
        row.append(np.mean(vals))
    P('%s | %s' % (sn, ' '.join('%.3f' % v for v in row)))


def drift(x, mask):
    v = x[mask & np.isfinite(x)]
    return v.mean() * 1e4


P('\nTABLE B  signal | q | residence days pure / hysteresis | names in hysteresis set (mean, share of pool) | daily churn')
P('         then relative drift bp/day for ALL P1 P2 P3 EX2015: hysteresis set ; pure zone ; hysteresis net of 35.2bp/residence (ALL, EX2015)')
for sn in SIGS:
    R = np.where(U, sig[sn], np.nan)
    for q in (0.1, 0.2, 0.3):
        with np.errstate(invalid='ignore'):
            Zp = R <= q
            Zh = np.zeros((T, N), bool)
            cur = np.zeros(N, bool)
            ent = 0
            for t in range(T):
                r = R[t]
                nxt = (r <= q) | (cur & (r <= 2 * q))
                ent += int((nxt & ~cur).sum())
                cur = nxt
                Zh[t] = cur
        ent_p = int((Zp[1:] & ~Zp[:-1]).sum() + Zp[0].sum())
        res_p = Zp.sum() / ent_p
        res_h = Zh.sum() / ent
        bh, m, nh, sdh, _ = basket_series(Zh, U, oo)
        KEEP[(sn, q)] = (bh, m, nh, sdh, res_h)
        bp_, _, _, _, _ = basket_series(Zp, U, oo)
        xh, xp = bh - m, bp_ - m
        pool_n = U.sum(1)
        ok = pool_n > 0
        P('%s | %.1f | %5.1f / %5.1f | %6.0f %.3f | %.4f' % (sn, q, res_p, res_h, Zh.sum(1)[ok].mean(),
                                                           (Zh.sum(1)[ok] / pool_n[ok]).mean(), 1.0 / res_h))
        P('      hyst: %s ; pure: %s ; hyst net: %.2f %.2f' % (
            ' '.join('%5.2f' % drift(xh, per[p]) for p in per), ' '.join('%5.2f' % drift(xp, per[p]) for p in per),
            drift(xh, per['ALL']) - 35.2 / res_h, drift(xh, per['EX2015']) - 35.2 / res_h))
        P('      hyst by year: %s' % ' '.join('%5.2f' % drift(xh, yrs == y) for y in range(2014, 2023)))

# Added after M3/M4 were read: exposure and tracking error of the hysteresis set, same definitions as basket_tracking_error.py.
wins = {}
for w in WIN:
    mk = np.zeros(T, bool)
    mk[w:w + WLEN] = True
    wins['W%d_%s' % (w, d['dates'][w][:7])] = mk
P('\nTABLE C  hysteresis set versus the daily equal-weight pool: signal q | period | beta | TE% | sigma_e set% | n | sigma_F% | gross bp/d |'
  ' net bp/d and daily ratio for K=inf (27.2/res), K=50 (35.2/res), K=80 (41.2/res)')
for (sn, q), (bh, m, nh, sdh, res_h) in KEEP.items():
    x = bh - m
    fin = np.isfinite(x)
    for pn, pmk in [('ALL', per['ALL']), ('EX2015', per['EX2015'])] + list(wins.items()):
        v = pmk & fin
        beta = np.cov(bh[v], m[v])[0, 1] / m[v].var(ddof=1)
        te = x[v].std(ddof=1)
        sez = sqrt((sdh[v] ** 2).mean())
        nb = nh[v].mean()
        sF = sqrt(max(te ** 2 - sez ** 2 / nb, 0.0))
        g = x[v].mean()
        out = []
        for K, c in ((0, 27.2), (50, 35.2), (80, 41.2)):
            tk = sF if K == 0 else sqrt(sez ** 2 / K + sF ** 2)
            net = g - c * 1e-4 / res_h
            out.append('%5.2f %+.3f' % (net * 1e4, net / tk))
        P('%s %.1f | %-13s | %.3f | %.3f | %.3f | %4.0f | %.3f | %5.2f | %s' % (sn, q, pn, beta, te * 100, sez * 100, nb, sF * 100,
                                                                           g * 1e4, ' ; '.join(out)))
P('\nTABLE D  hysteresis set (daily equal weight, net of 35.2bp/residence) versus the buy-and-hold pool benchmark, four windows:')
P('signal q | window | annualised mean daily difference % | daily ratio | t | end NAV set / benchmark | MDD set / benchmark')
first_ok = int(np.nonzero(U.sum(1) >= 200)[0][0]) + 1


def mdd(nav):
    return float((1 - nav / np.maximum.accumulate(nav)).max())


for w in WIN:
    ts = max(w, first_ok)
    te_ = min(w + WLEN, T - 1)
    names = U[ts - 1]
    navb = np.nanmean(Off[ts:te_ + 1][:, names] / Off[ts][names][None, :], axis=1)
    rb = navb[1:] / navb[:-1] - 1
    for (sn, q), (bh, m, nh, sdh, res_h) in KEEP.items():
        r = np.nan_to_num(bh[ts:te_]) - 35.2e-4 / res_h
        df = r - rb
        nv = np.concatenate([[1.0], np.cumprod(1 + r)])
        P('%s %.1f | %s | %6.2f | %+.3f | %+.2f | %.3f / %.3f | %.3f / %.3f' % (
            sn, q, d['dates'][ts][:7], df.mean() * 25200, df.mean() / df.std(ddof=1), df.mean() / df.std(ddof=1) * sqrt(df.size),
            nv[-1], navb[-1], mdd(nv), mdd(navb)))
finish()
