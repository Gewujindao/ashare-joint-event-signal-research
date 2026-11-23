# market_volatility_drawdown.py — reads: tradable, close_qfq, open_qfq (training rows only).
# Computes: equal-weight market daily return (close-to-close and open-to-open) sd/mean by period,
# median individual daily vol, pooled stock-day mean return, market R2, rolling 551-day market t (Zm)
# and max drawdown, then recomputes the stage-1 threshold table with measured values.
from common import *
from math import exp, pi

start('q1_base')
d = load('tradable', 'close_qfq', 'open_qfq')
tr, yrs = d['tradable'], d['years']
T, N = tr.shape
seas = seasoned(tr, ~np.isnan(d['close_qfq']))
C = np.where(tr, d['close_qfq'], np.nan).astype(np.float32)
O = np.where(tr, d['open_qfq'], np.nan).astype(np.float32)
Cff = ffill(C)
rcc = np.full((T, N), np.nan, np.float32)
rcc[1:] = Cff[1:] / Cff[:-1] - 1
U = seas & np.isfinite(rcc)
m = np.where(U, rcc, 0).sum(1, dtype=np.float64) / np.maximum(U.sum(1), 1)
roo = np.full((T, N), np.nan, np.float32)
roo[:-1] = O[1:] / O[:-1] - 1
U2 = seas & np.isfinite(roo)
moo = np.where(U2, roo, 0).sum(1, dtype=np.float64) / np.maximum(U2.sum(1), 1)
P('rows', T, 'cols', N, 'first', d['dates'][0], 'last', d['dates'][-1])
P('universe (seasoned tradable) per day: mean %.0f min %d max %d' % (U.sum(1)[1:].mean(), U.sum(1)[1:].min(), U.sum(1).max()))

res = {}
P('\n[period] sigma_m_cc%  ann%  mean_m_cc_bp | sigma_m_oo%  mean_m_oo_bp | median_indiv_vol%  pooled_mu_bp  median_R2_mkt')
for name, pm in periods(yrs).items():
    pm = pm.copy()
    pm[0] = False
    mc = m[pm]
    mo = moo[pm & (np.arange(T) < T - 1)]
    x = np.where(U & pm[:, None], rcc, np.nan)
    cnt = np.isfinite(x).sum(0)
    sd_i = np.nanstd(x[:, cnt >= 100], axis=0)
    mu = np.nanmean(x)
    # market R2 per stock
    M = np.isfinite(x)
    x0 = np.where(M, x, 0).astype(np.float64)
    n_i = M.sum(0).astype(np.float64)
    sel = n_i >= 250
    mm = (M * m[:, None]).sum(0)
    sx = x0.sum(0)
    sxx = (x0 ** 2).sum(0)
    smm = (M * (m ** 2)[:, None]).sum(0)
    sxm = (x0 * m[:, None]).sum(0)
    with np.errstate(all='ignore'):
        cov = sxm / n_i - sx / n_i * mm / n_i
        r2 = cov ** 2 / ((sxx / n_i - (sx / n_i) ** 2) * (smm / n_i - (mm / n_i) ** 2))
    r2m = float(np.nanmedian(r2[sel]))
    res[name] = (mc.std(ddof=1), float(np.median(sd_i)), float(mu))
    P('%-9s %.3f  %.1f  %.2f | %.3f  %.2f | %.3f  %.2f  %.3f' % (
        name, mc.std(ddof=1) * 100, mc.std(ddof=1) * sqrt(252) * 100, mc.mean() * 1e4,
        mo.std(ddof=1) * 100, mo.mean() * 1e4, np.median(sd_i) * 100, mu * 1e4, r2m))

P('\n[by year] sigma_m_cc%  mean_m_cc_bp')
for y in range(2014, 2023):
    pm = yrs == y
    pm[0] = False
    P(y, '%.3f  %.2f' % (m[pm].std(ddof=1) * 100, m[pm].mean() * 1e4))

W = 551
Z, DD, st_year = [], [], []
for s in range(1, T - W + 1):
    w = m[s:s + W]
    Z.append(sqrt(W) * w.mean() / w.std(ddof=1))
    nav = np.cumprod(1 + w)
    DD.append((1 - nav / np.maximum.accumulate(nav)).max())
    st_year.append(yrs[s])
Z, DD, st_year = np.array(Z), np.array(DD), np.array(st_year)
qs = [0, 5, 25, 50, 75, 95, 100]
P('\n[rolling 551-day windows] n=%d  quantiles min,5,25,50,75,95,max' % Z.size)
P('Zm all windows      ', np.round(np.percentile(Z, qs), 2), 'mean %.2f sd %.2f' % (Z.mean(), Z.std()))
P('Zm windows from 2016', np.round(np.percentile(Z[st_year >= 2016], qs), 2), 'mean %.2f' % Z[st_year >= 2016].mean())
P('maxDD all windows      ', np.round(np.percentile(DD, qs), 3))
P('maxDD windows from 2016', np.round(np.percentile(DD[st_year >= 2016], qs), 3))
for s in (1, 547, 1094, 1641):
    w = m[s:s + W]
    P('window start %s: Zm %.2f  sigma_m %.3f%%  maxDD %.3f' % (d['dates'][s], sqrt(W) * w.mean() / w.std(ddof=1), w.std(ddof=1) * 100, DD[s - 1]))


def phi(z):
    return exp(-z * z / 2) / sqrt(2 * pi)


P('\n[recomputed stage-1 thresholds, c = 27.2bp (holdout tax); k = 2/sqrt(551) = %.5f]' % (2 / sqrt(551)))
c = 0.00272
k = 2 / sqrt(551)
for name in ('ALL', 'EX2015', 'P3_20-22', 'P2_17-19', 'P1_14-16'):
    sm, si, mu = res[name]
    h = k * sm
    P('--- basis %s: sigma_m %.3f%%  indiv vol %.3f%%  mu %.2fbp  ->  h = %.2f bp/day ; A_min = %.3f x out-fraction per year' % (
        name, sm * 100, si * 100, mu * 1e4, h * 1e4, k * sqrt(252) * sm * sqrt(252)))
    P('  L: breakeven bp/day | target-1 bp/day | target-1 per spell %')
    for L in (1, 2, 3, 5, 10, 20):
        P('  L=%2d: %.1f | %.1f | %.2f' % (L, -c / L * 1e4, -(h + c / L) * 1e4, -(h * L + c) * 100))
    mub = max(mu, 0.0)
    for q, z in ((0.01, 2.3263), (0.05, 1.6449), (0.10, 1.2816), (0.20, 0.8416)):
        lam = phi(z) / q
        be = 2 * sqrt(mub * c) / (si * lam) if mub > 0 else c / (si * lam * 1.0)
        tg = 2 * sqrt((mub + h) * c) / (si * lam)
        tg5 = ((mub + h) * 5 + c) / (si * sqrt(5) * lam)
        be5 = (mub * 5 + c) / (si * sqrt(5) * lam)
        P('  q=%.2f: rho breakeven(optimal L) %.4f  target(optimal L) %.4f | L=5: breakeven %.4f target %.4f' % (q, be, tg, be5, tg5))
finish()
