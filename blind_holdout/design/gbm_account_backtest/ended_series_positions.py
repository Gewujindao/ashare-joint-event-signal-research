# ended_series_positions.py — report only: the positions of the main account whose price series ended while held.
# usage: ended_series_positions.py NPZ DATA_END CACHE      (reads the scratch cache written by account_v1.py main)
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import account_core as Q
from account_core import C, np, P
NPZ, DATA_END, CACHE = sys.argv[1:4]
C.start('c2_diag_dead')
z = np.load(CACHE, allow_pickle=False)
r0 = int(z['r0'])
M = Q.market(NPZ, DATA_END, r0)
for kk in ('G', 'U', 'qa'):
    M[kk] = z[kk]
M['PRG'] = Q.pct_rank_rows(M['G'], M['U'] & np.isfinite(M['G']))
dts = M['dates']
for nm, d0, d1 in (('V', '2020-01-03', '2022-12-30'), ('SEL', '2023-01-03', '2024-06-28')):
    a, b = int(np.searchsorted(dts, d0)), int(np.searchsorted(dts, d1, 'right'))
    R = Q.run_account(M, a, b, 500000.0, lambda t: M['PRG'][t], detail=True)
    P('\n%s: code | bought | last tradable | days held until last tradable | of these: ST days, out-of-pool days, sellable opens | value at last close / value at buy open | same per 10,000 CNY bought | amt60 quintile at buy' % nm)
    for i in R['dead_idx']:
        jb = [j for j, (es, eb, hl) in enumerate(R['log']) if i in eb][-1]
        eb_, el = a + jb, int(M['last_tr'][i])
        rows = slice(eb_ + 1, el + 1)
        val = M['cff0'][el, i] / M['o_q'][eb_, i]
        P('  %s | %s | %s | %d | %d, %d, %d | %.3f | %.0f | %d' % (
            M['codes'][i], dts[eb_], dts[el], el - eb_, int(M['st'][rows, i].sum()), int((~M['U'][eb_:el, i]).sum()), int(M['sell_ok'][rows, i].sum()),
            val, val * 10000.0, int(M['qa'][eb_ - 1, i])))
    Bm = Q.bench_B(M, a, b, 500000.0)
    Bz = Q.bench_B(M, a, b, 500000.0, zero_term=True)
    P('  benchmark B: accounts with an ended series %d of %d ; B final NAV main %.4f, ended=0 %.4f (%.2f%% of NAV)' % (
        Bm['n_term'], Bm['npool'], Bm['nav'][-1], Bz['nav'][-1], (1 - Bz['nav'][-1] / Bm['nav'][-1]) * 100))
    P('  strategy: %d ended positions out of %d buys ; carried value %.2f%% of final NAV ; slots frozen at window end %d of 50' % (
        R['n_term'], R['c']['nbuy'], R['dead_val'] / R['nav'][-1] * 100, R['n_term']))
C.finish()
