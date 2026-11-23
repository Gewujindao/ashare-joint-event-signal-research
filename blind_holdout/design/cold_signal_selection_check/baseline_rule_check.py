# baseline_rule_check.py — P7. Corresponds to design\backtest_engine\baseline_train.py: the same engine (design\backtest_engine\engine.py)
# and the same baseline rule (non-ST stock, second or later consecutive limit-up close with the board opened
# intraday -> sell at the next open, buy back 3 trading days after the sell). One window, 500k per account,
# accounts start in cash on the window's first day, stamp duty by actual date (0.1% before 2023-08-28, then 0.05%).
# Only the quantities listed in P7 are computed (annualised excess, daily ratio, t, mean out-of-market share);
# the three lines for the daily difference are the ones inside engine.metrics.
import sys
sys.path.insert(0, r'<BLIND_ROOT>\design\backtest_engine')
from selection_check_common import *
import engine as E
C.OUTDIR = C.ROOT + r'\design\cold_signal_selection_check'

start('s3_baseline_' + MODE)
d = load_to('tradable', 'st', 'low_raw', 'close_raw', 'preclose_raw', 'close_qfq', 'open_raw', 'open_qfq')
tr, st, dates = d['tradable'], d['st'], d['dates']
T, N = tr.shape
sig = E.signals(d)
board = sig['board']
del d['low_raw'], d['close_raw']
sell_ok, buy_ok = E.fills(tr, st, d['open_raw'].astype(np.float64), d['preclose_raw'].astype(np.float64),
                          d['open_qfq'].astype(np.float64))
want = E.plan(board, sell_ok, buy_ok)
del sell_ok, buy_ok
first = (~np.isnan(d['close_qfq'])).argmax(axis=0)
cum_before = np.vstack([np.zeros((1, N), np.int32), np.cumsum(tr, axis=0, dtype=np.int32)[:-1]])
stamp = E.stamp_rates(dates)
a = int(np.nonzero(dates >= EV0)[0][0])
b = T - 1
if MODE == 'sel':
    assert dates[a] == '2023-01-03' and b - a + 1 == 359
pool = tr[a] & ((first == 0) | (cum_before[a] >= 750))
sl = slice(a, b + 1)
W = {k: d[k][sl][:, pool] for k in ('open_raw', 'preclose_raw', 'open_qfq', 'close_qfq')}
trw, stw, ww = tr[sl][:, pool], st[sl][:, pool], want[sl][:, pool]
K = 500000.0
eqH, posH, costH, nbH, nsH = E.simulate(trw, stw, W['open_raw'], W['preclose_raw'], W['open_qfq'], W['close_qfq'],
                                        np.ones(ww.shape, bool), K, stamp[sl])
eqS, posS, costS, nbS, nsS = E.simulate(trw, stw, W['open_raw'], W['preclose_raw'], W['open_qfq'], W['close_qfq'],
                                        ww, K, stamp[sl])
n = eqS.shape[1]
rS = eqS / np.vstack([np.full((1, n), K), eqS[:-1]]) - 1
rH = eqH / np.vstack([np.full((1, n), K), eqH[:-1]]) - 1
X = (rS - rH).mean(1)
mX, ratio, t, tnw = E.tstats(X)
P('mode=%s  window %s..%s  days=%d  pool=%d  board events in window (pool)=%d  stamp 0.1%% days=%d' % (
    MODE, dates[a], dates[b], b - a + 1, int(pool.sum()), int(board[sl][:, pool].sum()), int((stamp[sl] > 7e-4).sum())))
P('annualised excess %%=%.3f  daily ratio=%.4f  t=%.2f  mean out-of-market share %%=%.3f' % (
    252 * mX * 100, ratio, t, (1 - posS.mean()) * 100))
finish()
