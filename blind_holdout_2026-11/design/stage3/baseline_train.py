# baseline_train.py — "simple indicator baseline" on the training segment only (rows <= 2022-12-30).
# Reads: tradable, st, low_raw, close_raw, preclose_raw, close_qfq, open_raw, open_qfq.
# Rule (no variants, no tuning): a non-ST stock that closes limit-up for at least the second consecutive day
# with the board opened intraday is sold at the next open and bought back 3 trading days after the sell.
# Windows: whole training segment, 2016-2022 (the run without 2014-2015), four 551-day windows.
# Pool of a window: tradable on its first day and at least 750 tradable days before it (stocks whose record
# starts on the first data day count as satisfying this, because the file has no history before 2014).
# Costs use the historical stamp tax (0.1% throughout training). Capital 500k (main) and 20k (sensitivity).
from engine import *

start('baseline_train')
d = load('tradable', 'st', 'low_raw', 'close_raw', 'preclose_raw', 'close_qfq', 'open_raw', 'open_qfq')
tr, st, yrs, dates = d['tradable'], d['st'], d['years'], d['dates']
T, N = tr.shape
sig = signals(d)
board = sig['board']
sell_ok, buy_ok = fills(tr, st, d['open_raw'].astype(np.float64), d['preclose_raw'].astype(np.float64),
                        d['open_qfq'].astype(np.float64))
want = plan(board, sell_ok, buy_ok)
del sell_ok, buy_ok, d['low_raw'], d['close_raw']
first = (~np.isnan(d['close_qfq'])).argmax(axis=0)
cum_before = np.vstack([np.zeros((1, N), np.int32), np.cumsum(tr, axis=0, dtype=np.int32)[:-1]])
stamp = stamp_rates(dates)
i2016 = int(np.nonzero(yrs == 2016)[0][0])
windows = [('whole 2014-2022', 0, T - 1), ('2016-2022', i2016, T - 1)] + \
          [('win %s' % dates[s][:7], s, s + 550) for s in (0, 547, 1094, 1641)]
P('events (accepted board signals) by year: ' + '  '.join('%d:%d' % (y, board[yrs == y].sum()) for y in range(2014, 2023)))
P('share of days the virtual rule is out, all seasoned stock-days: %.4f' % (1 - want[sig['seas']].mean()))
P('\ncolumns: window | K | pool | T1: ann excess % / ratio / t / NW t | T2: median DD strat vs hold ; merged DD strat vs hold ; pass | '
  'T3: NAV fast / shift mean / shift 97.5% / z / pass ; swap z | NAV exact strat / hold | out share mean / daily sd | '
  'round trips per stock-year | extra cost %/yr')
rng = np.random.default_rng(20261109)
for wname, a, b in windows:
    pool = tr[a] & ((first == 0) | (cum_before[a] >= 750))
    sl = slice(a, b + 1)
    W = {k: d[k][sl][:, pool] for k in ('open_raw', 'preclose_raw', 'open_qfq', 'close_qfq')}
    trw, stw, ww = tr[sl][:, pool], st[sl][:, pool], want[sl][:, pool]
    yw = yrs[sl]
    years = (b - a + 1) / 243.5
    for K in (500000.0, 20000.0):
        eqH, posH, costH, nbH, nsH = simulate(trw, stw, W['open_raw'], W['preclose_raw'], W['open_qfq'], W['close_qfq'],
                                              np.ones(ww.shape, bool), K, stamp[sl])
        eqS, posS, costS, nbS, nsS = simulate(trw, stw, W['open_raw'], W['preclose_raw'], W['open_qfq'], W['close_qfq'],
                                              ww, K, stamp[sl])
        res, X = metrics(eqS, eqH, posS, K, W['open_qfq'], W['close_qfq'], trw, nbS, nsS,
                         SLIP + COMM + TRANSFER + STAMP_OLD, seeds=200, rng=rng)
        P('%s | %dk | %d | %.3f / %.4f / %.2f / %.2f | %.4f vs %.4f ; %.4f vs %.4f ; %s | %.5f / %.5f / %.5f / %.2f / %s ; %.2f | %.4f / %.4f | %.4f / %.4f | %.2f | %.3f' % (
            wname, K / 1000, int(pool.sum()), res['ann_excess_pct'], res['ratio'], res['t'], res['t_nw5'],
            res['dd_med_S'], res['dd_med_H'], res['dd_merged_S'], res['dd_merged_H'], 'PASS' if res['t2_pass'] else 'FAIL',
            res['nav_fast'], res['shift_mean'], res['shift_q975'], res['shift_z'], 'PASS' if res['t3_pass'] else 'FAIL',
            res['swap_z'], res['nav_S'], res['nav_H'], res['out_frac'], res['out_sd'],
            nsS.mean() / years, (costS - costH).mean() / K / years * 100))
        if wname.startswith('whole'):
            m, ratio, t, tnw = tstats(X[yw != 2015])
            P('   whole run, 2015 days removed from the daily series | %dk | T1: ann excess %.3f / ratio %.4f / t %.2f / NW t %.2f (n=%d)' % (
                K / 1000, 252 * m * 100, ratio, t, tnw, (yw != 2015).sum()))
            if K == 500000.0:
                P('   by year (500k, whole run): year | X mean bp/day | ratio | out share mean % / daily sd % | max daily out share %')
                osh = 1 - posS.mean(1)
                for y in range(2014, 2023):
                    x = X[yw == y]
                    o = osh[yw == y]
                    P('   %d | %.3f | %.4f | %.3f / %.3f | %.2f' % (y, x.mean() * 1e4, x.mean() / x.std(ddof=1), o.mean() * 100, o.std() * 100, o.max() * 100))
    del W, eqH, eqS, posS, posH
finish()
