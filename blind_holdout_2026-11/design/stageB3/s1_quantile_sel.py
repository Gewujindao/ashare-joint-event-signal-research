# s1_quantile_sel.py — P1-P5. Corresponds to design\stageB2\m1_quantile.py (same definitions; only the signal-day
# range changes). Computes only what PREDICTIONS_selection.txt lists: frozen 20-day hold, executable fills,
# relative drift in bp/day for coldest 1%, coldest 10%, coldest 30%, the 10%-50% segment, hottest 10%; psi.
from b3common import *

start('s1_quantile_' + MODE)
B = build()
d, T, N, U, sig, OQ = B['d'], B['T'], B['N'], B['U'], B['sig'], B['OQ']
sell_ok, buy_ok = exec_maps(d)
nsell = next_true(sell_ok)
nbuy = next_true(buy_ok)
H = 20
ev = np.arange(T) >= WARM
P('mode=%s  signal days %s..%s  rows in window=%d  pool mean names/day=%.0f' % (MODE, d['dates'][WARM], d['dates'][-1], T - WARM, U[WARM:].sum(1).mean()))
b = nbuy[1:T + 1]
okb = b < T
s_ = np.take_along_axis(nsell, np.minimum(np.where(okb, b, 0) + H, T), axis=0)
ok = okb & (s_ < T)
real = np.take_along_axis(OQ, np.where(ok, s_, 0), axis=0) / np.take_along_axis(OQ, np.where(ok, b, 0), axis=0) - 1
real[~ok] = np.nan
V = U & np.isfinite(real)
ZONES = [('cold1', 0, 1), ('cold10', 0, 10), ('cold30', 0, 30), ('seg10_50', 10, 50), ('hot10', 90, 100)]
P('signal | matured signal days | ' + ' | '.join(z[0] for z in ZONES) + ' | psi = cold10/(cold10-hot10) | cold1 < seg10_50')
for sn in SIGS:
    x = np.nan_to_num(sig[sn], nan=0.0)
    bins = np.where(U, np.clip(np.ceil(x * 100) - 1, 0, 99), 0).astype(np.int64)
    sums = np.zeros((T, 100))
    cnts = np.zeros((T, 100))
    for t in range(WARM, T):
        v = V[t]
        if v.sum() < 200:
            continue
        r = real[t, v].astype(np.float64)
        r -= r.mean()
        bb = bins[t, v]
        sums[t] = np.bincount(bb, weights=r, minlength=100)
        cnts[t] = np.bincount(bb, minlength=100)
    out = {}
    nd = 0
    for zn, lo, hi in ZONES:
        s = sums[:, lo:hi].sum(1)
        c = cnts[:, lo:hi].sum(1)
        zs = np.where(c > 0, s / np.maximum(c, 1), np.nan)
        m = ev & np.isfinite(zs)
        nd = int(m.sum())
        out[zn] = zs[m].mean() * 1e4 / H
    psi = out['cold10'] / (out['cold10'] - out['hot10'])
    P('%s | %d | %s | %.3f | %s' % (sn, nd, ' | '.join('%+.2f' % out[z[0]] for z in ZONES), psi,
                                   'yes' if out['cold1'] < out['seg10_50'] else 'no'))
finish()
