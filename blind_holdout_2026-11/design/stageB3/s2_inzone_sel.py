# s2_inzone_sel.py — P6 (and P8). Corresponds to design\stageB2\m2_residence.py, hysteresis set of S1 with q = 0.3
# (enter at rank <= 0.3, leave at rank > 0.6 or when the name drops out of the pool), tables B and C of that script.
# Same definitions; only the date range changes. Cost per round trip for 50 names (10k per position, minimum
# commission): 40.2bp while the stamp duty is 0.1% (before 2023-08-28) and 35.2bp afterwards.
from b3common import *
from b2common import oo_returns, basket_series
C.OUTDIR = C.ROOT + r'\design\stageB3'      # b2common points the output folder at stageB2 on import; point it back

start('s2_inzone_' + MODE)
B = build()
d, T, N, U, sig, OQ = B['d'], B['T'], B['N'], B['U'], B['sig'], B['OQ']
dates = d['dates']
Off, oo = oo_returns(OQ)
q = 0.3
R = np.where(U, sig['S1_max20'], np.nan)
Zh = np.zeros((T, N), bool)
cur = np.zeros(N, bool)
ent = 0
with np.errstate(invalid='ignore'):
    for t in range(T):
        r = R[t]
        nxt = (r <= q) | (cur & (r <= 2 * q))
        if t >= WARM:
            ent += int((nxt & ~cur).sum())
        cur = nxt
        Zh[t] = cur
res_h = Zh[WARM:].sum() / ent
bh, m, nh, sdh, _ = basket_series(Zh, U, oo)
x = bh - m
v = (np.arange(T) >= WARM) & np.isfinite(x)
g = x[v].mean()
te = x[v].std(ddof=1)
sez = sqrt((sdh[v] ** 2).mean())
nb = nh[v].mean()
sF = sqrt(max(te ** 2 - sez ** 2 / nb, 0.0))
beta = np.cov(bh[v], m[v])[0, 1] / m[v].var(ddof=1)
c = np.where(dates < STAMP_SWITCH, 40.2e-4, 35.2e-4)[v].mean()
net = g - c / res_h
te50 = sqrt(sez ** 2 / 50 + sF ** 2)
ratio = net / te50
nd = int(v.sum())
P('mode=%s  return days %s..%s  n=%d' % (MODE, dates[np.nonzero(v)[0][0]], dates[np.nonzero(v)[0][-1]], nd))
P('residence days=%.1f  names in set=%.0f  beta=%.3f  TE vs pool=%.3f%%  sigma_e set=%.3f%%  sigma_F=%.3f%%' % (res_h, nb, beta, te * 100, sez * 100, sF * 100))
P('gross drift bp/day=%.2f  mean round-trip cost bp=%.1f  net bp/day (K=50)=%.2f  annualised net %%=%.2f  TE(50)=%.3f%%  daily ratio=%.4f  t=%.2f'
  % (g * 1e4, c * 1e4, net * 1e4, net * 252 * 100, te50 * 100, ratio, ratio * sqrt(nd)))
finish()
