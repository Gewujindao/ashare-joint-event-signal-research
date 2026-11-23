# basket_tracking_error.py — exposure and tracking error of the cold-zone equal-weight basket (training rows only).
# Basket = names with rank <= q at close t-1, equal weight, earning open t -> open t+1. m = pool equal-weight mean.
# sigma_F^2 = var(basket - m) - sigma_e_zone^2 / n ; TE(K)^2 = sigma_e_zone^2 / K + sigma_F^2.
from cold_signal_common import *

start('m3_basket')
B = build()
d, T, N, U, sig, OQ = B['d'], B['T'], B['N'], B['U'], B['sig'], B['OQ']
yrs = d['years']
per = periods(yrs)
Off, oo = oo_returns(OQ)
del Off
TH = 2 / sqrt(551)
wins = {}
for w in WIN:
    mk = np.zeros(T, bool)
    mk[w:w + WLEN] = True
    wins['W%d_%s' % (w, d['dates'][w][:7])] = mk
P('threshold daily mean/sd = %.4f' % TH)
P('columns per period: beta | sd(m)%% | TE%% | market part%% | sigma_e zone%% | sigma_e pool%% | n | sigma_F%% | gross bp/d |'
  ' TE(30) TE(50) TE(80)%% | ratio K=inf H20, H40 (c=27.2) | ratio K=50 H20, H40 (c=35.2)')
for sn in SIGS:
    for q in (0.1, 0.2, 0.3):
        with np.errstate(invalid='ignore'):
            Z = U & (sig[sn] <= q)
        b, m, n, sdz, sdu = basket_series(Z, U, oo)
        x = b - m
        fin = np.isfinite(x)
        P('\n%s q=%.1f' % (sn, q))
        store = {}
        for pn, pmk in list(per.items()) + list(wins.items()):
            v = pmk & fin
            bb, mm, xx = b[v], m[v], x[v]
            beta = np.cov(bb, mm)[0, 1] / mm.var(ddof=1)
            te = xx.std(ddof=1)
            sez = sqrt((sdz[v] ** 2).mean())
            seu = sqrt((sdu[v] ** 2).mean())
            nb = n[v].mean()
            sF = sqrt(max(te ** 2 - sez ** 2 / nb, 0.0))
            g = xx.mean()
            tek = [sqrt(sez ** 2 / k + sF ** 2) for k in (30, 50, 80)]
            r_inf = [(g - 27.2e-4 / h) / sF for h in (20, 40)]
            r_50 = [(g - 35.2e-4 / h) / tek[1] for h in (20, 40)]
            store[pn] = (g, sF, sez, beta)
            P('  %-11s %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | %4.0f | %.3f | %5.2f | %.3f %.3f %.3f | %+.3f %+.3f | %+.3f %+.3f'
              % (pn, beta, mm.std() * 100, te * 100, abs(beta - 1) * mm.std() * 100, sez * 100, seu * 100, nb, sF * 100,
                 g * 1e4, tek[0] * 100, tek[1] * 100, tek[2] * 100, r_inf[0], r_inf[1], r_50[0], r_50[1]))
        row = []
        for y in range(2014, 2023):
            v = (yrs == y) & fin
            row.append('%5.2f(%+.3f)' % (x[v].mean() * 1e4, (x[v].mean() - 27.2e-4 / 20) / x[v].std(ddof=1)))
        P('  by year gross bp/d (ratio K=inf H20 using yearly TE): ' + ' '.join(row))
        nwin = sum(1 for pn in wins if (store[pn][0] - 27.2e-4 / 20) / store[pn][1] < TH)
        P('  windows with K=inf H20 ratio < %.4f: %d of 4' % (TH, nwin))
finish()
