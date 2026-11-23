# overheated_states.py — reads: tradable, st, turnover_pct, close_qfq, open_qfq, open_raw, preclose_raw (training rows).
# Overheated states: abnormal turnover (today / median of previous 60 days) x past 5-day return.
#   HOT_top q = top q of stocks per day by min(pct rank abnormal turnover, pct rank 5-day return) (cumulative sets)
#   AT_top q  = top q by abnormal turnover alone
# For L in 5,10,20: ideal open(t+1)->open(t+1+L) mean, the same minus the same-day universe mean, realised mean
# under the fill rules, clustered t. Basket series X_d = -(out share x basket open-to-open return) - cost:
# daily mean/sd gross and net of 27.2bp per round trip, to compare with 0.0852; years (of 9) with net ratio > 0.
from common import *

start('q2_hot')
d = load('tradable', 'st', 'turnover_pct', 'close_qfq', 'open_qfq', 'open_raw', 'preclose_raw')
tr, yrs = d['tradable'], d['years']
T, N = tr.shape
seas = seasoned(tr, ~np.isnan(d['close_qfq']))
Cff = ffill(np.where(tr, d['close_qfq'], np.nan).astype(np.float32))
states, U = hot_states(d, seas, Cff)
del d['turnover_pct'], d['close_qfq']
sell_ok, buy_ok = exec_maps(d)
nsell, nbuy = next_true(sell_ok), next_true(buy_ok)
OQ = np.where(tr, d['open_qfq'], np.nan).astype(np.float32)
oo = np.zeros((T, N), np.float32)
oo[:-1] = np.nan_to_num(OQ[1:] / OQ[:-1] - 1, nan=0.0, posinf=0.0, neginf=0.0)
Nu = np.maximum(seas.sum(1), 1).astype(np.float64)
per = periods(yrs)
Ls = (5, 10, 20)
# same-day universe mean of ideal forward return
umean = {}
for L in Ls:
    F = np.full((T, N), np.nan, np.float32)
    F[:T - 1 - L] = OQ[1 + L:] / OQ[1:T - L] - 1
    F[~U] = np.nan
    umean[L] = np.nanmean(F, axis=1)
    del F
P('universe per day mean %.0f' % U.sum(1)[100:].mean())
P('\nTABLE A  state | period | n | sell_blocked% | for L=5,10,20: ideal_mean% / ideal_minus_universe% / real_mean% (clustered t of real)')
for name, E in states.items():
    R = {L: spell_returns(E, L, OQ, nsell, nbuy) for L in Ls}
    for pn, pm in per.items():
        row = []
        for L in Ls:
            t, i, ideal, real, sblk, bblk, ok = R[L]
            sel = pm[t]
            n, rm, rmed, rt = cstat(real[sel], t[sel], T)
            row.append('%.2f/%.2f/%.2f (t %.1f)' % (np.nanmean(ideal[sel]) * 100,
                                                   np.nanmean((ideal - umean[L][t])[sel]) * 100, rm * 100, rt))
        P('%s | %s | %d | %.1f | %s' % (name, pn, int(pm[R[5][0]].sum()), R[5][4][pm[R[5][0]]].mean() * 100, '  '.join(row)))

P('\nTABLE B  basket series (ideal fills). state | L | period | mean out share% | X mean bp/day gross | ratio gross | ratio net(27.2bp) | ratio net(32.2bp) ; need >= +0.0852')
c1, c2 = 0.00272, 0.00322
for name, E in states.items():
    for L in Ls:
        out = np.zeros((T, N), bool)
        for j in range(1, L + 1):
            out[j:] |= E[:T - j]      # signal at t -> out on days t+1 .. t+L (open-to-open intervals)
        st_ = np.zeros((T, N), bool)
        st_[1:] = out[1:] & ~out[:-1]
        g = -(out * oo).sum(1, dtype=np.float64) / Nu
        ns = st_.sum(1) / Nu
        share = out.sum(1) / Nu
        for pn, pm in per.items():
            x0 = g[pm]
            x1 = (g - c1 * ns)[pm]
            x2 = (g - c2 * ns)[pm]
            P('%s | %d | %s | %.2f | %.3f | %.4f | %.4f | %.4f' % (
                name, L, pn, share[pm].mean() * 100, x0.mean() * 1e4, x0.mean() / x0.std(ddof=1),
                x1.mean() / x1.std(ddof=1), x2.mean() / x2.std(ddof=1)))
        x1 = g - c1 * ns
        pos = sum(1 for y in range(2014, 2023) if x1[yrs == y].mean() > 0)
        P('%s | %d | years with net(27.2bp) mean > 0: %d of 9 ; by year ratio: %s' % (
            name, L, pos, ' '.join('%.3f' % (x1[yrs == y].mean() / x1[yrs == y].std(ddof=1)) for y in range(2014, 2023))))
finish()
