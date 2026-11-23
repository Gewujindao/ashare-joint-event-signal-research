# limit_events.py — reads: tradable, st, open_raw, high_raw, low_raw, close_raw, preclose_raw, open_qfq (training rows).
# Limit-up / limit-down event table (limit thresholds 10%; ST uses 5%), ST and non-ST separately.
# For each event class and L in 1,2,3,5,10: ideal return open(t+1)->open(t+1+L) and realised return under the
# fill rules (sell at first executable open >= t+1, buy back at first executable open >= s+L),
# share of blocked sells / blocked buy-backs, median, date-clustered t. Reported for ALL, three sub-periods, ex-2015.
from common import *

start('q3_limits')
d = load('tradable', 'st', 'open_raw', 'high_raw', 'low_raw', 'close_raw', 'preclose_raw', 'open_qfq')
tr, st, yrs = d['tradable'], d['st'], d['years']
T, N = tr.shape
seas = seasoned(tr, ~np.isnan(d['close_raw']))
ev = limit_events(d, seas)
sell_ok, buy_ok = exec_maps(d)
nsell, nbuy = next_true(sell_ok), next_true(buy_ok)
OQ = np.where(tr, d['open_qfq'], np.nan).astype(np.float32)
per = periods(yrs)
Ls = (1, 2, 3, 5, 10)
P('unconditional: sell blocked share of seasoned tradable days %.4f ; buy blocked %.4f' % (
    1 - sell_ok[seas].mean(), 1 - buy_ok[seas].mean()))
# baseline: all seasoned tradable stock-days
P('\ncolumns: class | ST | period | n_events | sell_blocked% | buyback_blocked%(L=3) | for L=1,2,3,5,10: ideal_mean% / real_mean% | real_median%(L=3) | clustered t of real mean (L=3, L=5)')
classes = dict(ev)
classes['BASE_all_stock_days_every_5_days_take_1'] = seas & (np.arange(T) % 5 == 0)[:, None]   # subsample to stay under 1 GiB
for name, E0 in classes.items():
    for stname, smask in (('nonST', ~st), ('ST', st)):
        E = E0 & smask
        R = {L: spell_returns(E, L, OQ, nsell, nbuy) for L in Ls}
        for pn, pm in per.items():
            row = []
            t3 = R[3][0]
            sel3 = pm[t3]
            n = int(sel3.sum())
            if n < 30:
                P('%s | %s | %s | n=%d (too few)' % (name, stname, pn, n))
                continue
            sb = R[3][4][sel3].mean() * 100
            bb = R[3][5][sel3 & R[3][6]].mean() * 100
            ts = {}
            for L in Ls:
                t, i, ideal, real, sblk, bblk, ok = R[L]
                sel = pm[t]
                im = np.nanmean(ideal[sel]) * 100
                nn, rm, rmed, rt = cstat(real[sel], t[sel], T)
                row.append('%.2f/%.2f' % (im, rm * 100))
                ts[L] = (rmed, rt)
            P('%s | %s | %s | %d | %.1f | %.1f | %s | %.2f | %.1f, %.1f' % (
                name, stname, pn, n, sb, bb, '  '.join(row), ts[3][0] * 100, ts[3][1], ts[5][1]))

# basket series for the non-ST limit-up classes (same construction as Q2 table B; ideal fills)
P('\nTABLE B  basket series (ideal fills), nonST. class | L | period | out share% mean / sd over days | X mean bp/day gross | ratio gross | ratio net(27.2bp) ; need >= +0.0852')
oo = np.zeros((T, N), np.float32)
oo[:-1] = np.nan_to_num(OQ[1:] / OQ[:-1] - 1, nan=0.0, posinf=0.0, neginf=0.0)
Nu = np.maximum(seas.sum(1), 1).astype(np.float64)
Bk = {k: ev[k] & ~st for k in ('A1_first_board_non_oneprice', 'A2_second_board_non_oneprice', 'A3_third_board_plus_non_oneprice', 'A6_streak_break_prior_2plus_boards')}
Bk['A2_or_A3'] = Bk['A2_second_board_non_oneprice'] | Bk['A3_third_board_plus_non_oneprice']
c1 = 0.00272
for name, E in Bk.items():
    for L in (1, 3, 5):
        out = np.zeros((T, N), bool)
        for j in range(1, L + 1):
            out[j:] |= E[:T - j]
        st_ = np.zeros((T, N), bool)
        st_[1:] = out[1:] & ~out[:-1]
        g = -(out * oo).sum(1, dtype=np.float64) / Nu
        ns = st_.sum(1) / Nu
        share = out.sum(1) / Nu
        x1 = g - c1 * ns
        for pn, pm in per.items():
            P('%s | %d | %s | %.3f / %.3f | %.3f | %.4f | %.4f' % (
                name, L, pn, share[pm].mean() * 100, share[pm].std() * 100, g[pm].mean() * 1e4,
                g[pm].mean() / g[pm].std(ddof=1), x1[pm].mean() / x1[pm].std(ddof=1)))
        pos = sum(1 for y in range(2014, 2023) if x1[yrs == y].mean() > 0)
        P('%s | %d | years with net mean > 0: %d of 9 ; by year ratio: %s' % (
            name, L, pos, ' '.join('%.3f' % (x1[yrs == y].mean() / x1[yrs == y].std(ddof=1)) for y in range(2014, 2023))))
finish()
