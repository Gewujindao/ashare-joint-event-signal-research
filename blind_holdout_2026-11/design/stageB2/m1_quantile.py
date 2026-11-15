# m1_quantile.py — conditional relative-return curve by signal percentile (training rows only).
# Signals known at close t (higher = hotter): 20-day max daily return, 20-day vol, 4-rank composite.
# ideal = open(t+1) -> open(t+1+H), both tradable. real = first executable buy open b >= t+1 -> first executable
# sell open s >= b+H (both inside training). Relative = minus the same-day pool mean of the same return.
from b2common import *

start('m1_quantile')
B = build()
d, T, N, U, sig, OQ = B['d'], B['T'], B['N'], B['U'], B['sig'], B['OQ']
yrs = d['years']
per = periods(yrs)
sell_ok, buy_ok = exec_maps(d)
nsell = next_true(sell_ok)
nbuy = next_true(buy_ok)
del sell_ok, buy_ok
P('pool: mean names/day = %.0f ; days with pool = %d of %d' % (U.sum(1)[U.sum(1) > 0].mean(), (U.sum(1) > 0).sum(), T))
bins = {}
for s in SIGS:
    x = np.nan_to_num(sig[s], nan=0.0)
    bins[s] = np.where(U, np.clip(np.ceil(x * 100) - 1, 0, 99), 0).astype(np.int64)
HS = (5, 20, 40, 60)
res = {}
for H in HS:
    ideal = np.full((T, N), np.nan, np.float32)
    ideal[:T - 1 - H] = OQ[1 + H:] / OQ[1:T - H] - 1
    b = nbuy[1:T + 1]
    okb = b < T
    s_ = np.take_along_axis(nsell, np.minimum(np.where(okb, b, 0) + H, T), axis=0)
    ok = okb & (s_ < T)
    real = np.take_along_axis(OQ, np.where(ok, s_, 0), axis=0) / np.take_along_axis(OQ, np.where(ok, b, 0), axis=0) - 1
    real[~ok] = np.nan
    del s_, ok, okb
    for conv, R in (('ideal', ideal), ('real', real)):
        V = U & np.isfinite(R)
        for sn in SIGS:
            sums = np.zeros((T, 100))
            cnts = np.zeros((T, 100))
            for t in range(T):
                v = V[t]
                if v.sum() < 200:
                    continue
                r = R[t, v].astype(np.float64)
                r -= r.mean()
                bb = bins[sn][t, v]
                sums[t] = np.bincount(bb, weights=r, minlength=100)
                cnts[t] = np.bincount(bb, minlength=100)
            res[(sn, H, conv)] = (sums, cnts)
    del ideal, real, V


def zs(key, lo, hi):
    sums, cnts = res[key]
    s = sums[:, lo:hi].sum(1)
    c = cnts[:, lo:hi].sum(1)
    return np.where(c > 0, s / np.maximum(c, 1), np.nan)


def pm(x, mask, H):
    v = x[mask & np.isfinite(x)]
    return v.mean() * 1e4 / H if v.size else np.nan


ZONES = [('cold1', 0, 1), ('cold2', 0, 2), ('cold5', 0, 5), ('cold10', 0, 10), ('cold20', 0, 20), ('cold30', 0, 30),
         ('mid40', 30, 70), ('hot30', 70, 100), ('hot20', 80, 100), ('hot10', 90, 100), ('hot5', 95, 100),
         ('hot2', 98, 100), ('hot1', 99, 100)]
P('\nTABLE A  20 bins of 5% (cold -> hot), whole training, relative drift in bp/day: signal | H | convention | 20 values')
for sn in SIGS:
    for H in HS:
        for conv in ('ideal', 'real'):
            row = [pm(zs((sn, H, conv), 5 * k, 5 * k + 5), per['ALL'], H) for k in range(20)]
            P('%s | %2d | %-5s | %s' % (sn, H, conv, ' '.join('%6.1f' % v for v in row)))
P('\nTABLE B  zones, relative drift bp/day: signal | H | convention | zone | ALL P1_14-16 P2_17-19 P3_20-22 EX2015')
for sn in SIGS:
    for H in HS:
        for conv in ('ideal', 'real'):
            for zn, lo, hi in ZONES:
                x = zs((sn, H, conv), lo, hi)
                P('%s | %2d | %-5s | %-6s | %s' % (sn, H, conv, zn, ' '.join('%6.1f' % pm(x, per[p], H) for p in per)))
P('\nTABLE C  by year, real, relative drift bp/day: signal | H | zone | 2014..2022 | years>0')
for sn in SIGS:
    for H in (20, 40):
        for zn, lo, hi in ZONES:
            if zn not in ('cold10', 'cold20', 'cold30', 'hot10', 'hot20', 'hot30'):
                continue
            x = zs((sn, H, 'real'), lo, hi)
            row = [pm(x, yrs == y, H) for y in range(2014, 2023)]
            P('%s | %2d | %-6s | %s | %d' % (sn, H, zn, ' '.join('%6.1f' % v for v in row), sum(v > 0 for v in row)))
P('\nTABLE D  long-side share psi = cold/(cold-hot) and kappa = cold drift / symmetric reference (ASYM); real')
P('signal | H | q | for ALL P1 P2 P3 EX2015: cold bp/d, hot bp/d, psi ; kappa(ALL) kappa(EX2015)')
for sn in SIGS:
    for H in HS:
        for q in (10, 20, 30):
            xc = zs((sn, H, 'real'), 0, q)
            xh = zs((sn, H, 'real'), 100 - q, 100)
            row = []
            for p in per:
                c_, h_ = pm(xc, per[p], H), pm(xh, per[p], H)
                row.append('%5.1f %6.1f %5.2f' % (c_, h_, c_ / (c_ - h_) if c_ - h_ != 0 else np.nan))
            P('%s | %2d | %2d | %s ; %.2f %.2f' % (sn, H, q, ' ; '.join(row), pm(xc, per['ALL'], H) / ASYM[H],
                                                    pm(xc, per['EX2015'], H) / ASYM[H]))
P('\nTABLE E  share of pool-days where the buy at open t+1 is not executable (limit-up open or untradable), by zone, S3')
blk = nbuy[1:T + 1] != (np.arange(T)[:, None] + 1)
for zn, lo, hi in ZONES:
    z = U & (bins['S3_comp4'] >= lo) & (bins['S3_comp4'] < hi)
    P('%-6s  %.3f%%' % (zn, 100.0 * (blk & z).sum() / z.sum()))
finish()
