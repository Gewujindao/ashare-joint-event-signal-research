# prefix_check.py — prefix-invariance check of the decision path inside the training segment.
# Reads: tradable, st, low_raw, close_raw, preclose_raw, close_qfq, open_raw, open_qfq (training rows only).
# The decisions (want) computed from the first 1500 rows must equal the first 1500 rows of the decisions
# computed from all 2192 training rows.
from engine import *

start('prefix_check')
d = load('tradable', 'st', 'low_raw', 'close_raw', 'preclose_raw', 'close_qfq', 'open_raw', 'open_qfq')


def decisions(n):
    dd = {k: d[k][:n] for k in ('tradable', 'st', 'low_raw', 'close_raw', 'preclose_raw', 'close_qfq', 'open_raw', 'open_qfq')}
    sig = signals(dd)
    so, bo = fills(dd['tradable'], dd['st'], dd['open_raw'].astype(np.float64), dd['preclose_raw'].astype(np.float64),
                   dd['open_qfq'].astype(np.float64))
    return plan(sig['board'], so, bo), sig['board']


w_full, b_full = decisions(2192)
for n in (700, 1500):
    w_cut, b_cut = decisions(n)
    P('rows %d: events identical %s ; decisions identical %s ; out decisions in prefix %d' % (
        n, bool((b_cut == b_full[:n]).all()), bool((w_cut == w_full[:n]).all()), int((~w_cut).sum())))
finish()
