# c2b_run.py — 丙二补: version 2 (= frozen rule + buy exclusions X1, X2) with the conservative valuation as the main reading.
# usage: c2b_run.py NPZ DATA_END MODE TAG [CACHE]      MODE main | perturb | random   (same meaning as c2_run.py)
# Definitions: design\stageC2\丙二补_第二版与口径_运行前写定.txt. Version 1 is recomputed side by side in MODE main.
# MAIN reading = "conservative overlay": one set of trades (sized on last-close NAV), positions whose price series has
# ended are valued 0 from the day after their last tradable day (navz); benchmark B likewise.
import sys, os, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import c2_core as Q
from c2_core import C, np, sqrt, P, K
NPZ, DATA_END, MODE, TAG = sys.argv[1:5]
CACHE = sys.argv[5] if len(sys.argv) > 5 else None
assert MODE in ('main', 'perturb', 'random')
WINDOWS = [('V', '2020-01-03', '2022-12-30'), ('SEL', '2023-01-03', '2024-06-28'), ('HOLD', '2024-07-01', '2026-10-09')]
K0 = 500000.0
PACK = ('G', 'U', 'qa')


def tick(msg):
    P('# [%5.1fs] %s  peak_ws_MB=%.0f' % (time.time() - C._T0, msg, C.peak_mb()[0]))


if MODE == 'main' or not (CACHE and os.path.exists(CACHE)):
    S = Q.frozen_scores(NPZ, DATA_END, TAG)
    T = S['dates'].size
    xf, bf, msf = Q.setsim(S['PRG'], S['U'], S['buy_ok'], S['sell_ok'], S['oo'], S['t0'], T - 1)
    rec = np.load(os.path.join(HERE, '..', 'stageC1', 'record_SEL.npz'), allow_pickle=False)
    keep = rec['dates'] < S['dates'][-1]
    k = np.searchsorted(S['dates'], rec['dates'][keep])
    assert k.size > 0 and bool((S['dates'][k] == rec['dates'][keep]).all())
    dx, db = np.abs(xf[k] - rec['x'][keep]).max(), np.abs(bf[k] - rec['buys'][keep]).max()
    P('list check (frozen rule, no exclusions): decide() in set mode against record_SEL.npz, %d days %s..%s ; max |x diff| %.3e ; max |buys diff| %.1f'
      % (k.size, rec['dates'][keep][0], rec['dates'][keep][-1], dx, db))
    if not (dx == 0 and db == 0):
        P('LIST CHECK FAILED - no account reading is produced')
        C.finish()
        sys.exit(4)
    r0 = int(S['iF']) - 25
    pack = dict(G=S['G'][r0:].copy(), U=S['U'][r0:].copy(), qa=S['qa'][r0:].copy())
    bo, so = S['buy_ok'][r0:].copy(), S['sell_ok'][r0:].copy()
    del S, xf, bf, msf
    M = Q.market(NPZ, DATA_END, r0)
    assert bool((M['buy_ok'] == bo).all() and (M['sell_ok'] == so).all())
    del bo, so
    M.update(pack)
    if CACHE:
        np.savez_compressed(CACHE, r0=r0, dates=M['dates'], **pack)
        P('scratch cache written: %s (%.1f MB)' % (CACHE, os.path.getsize(CACHE) / 2 ** 20))
else:
    C.start('c2_' + TAG)
    z = np.load(CACHE, allow_pickle=False)
    r0 = int(z['r0'])
    M = Q.market(NPZ, DATA_END, r0)
    assert bool((M['dates'] == z['dates']).all())
    for kk in PACK:
        M[kk] = z[kk]
    z.close()
    P('scores read from scratch cache %s' % CACHE)
M['PRG'] = Q.pct_rank_rows(M['G'], M['U'] & np.isfinite(M['G']))
dts = M['dates']
N = M['G'].shape[1]
tick('scores and market ready; rows %s..%s' % (dts[0], dts[-1]))
wins = []
for nm, d0, d1 in WINDOWS:
    if d1 <= dts[-1]:
        a, b = int(np.searchsorted(dts, d0)), int(np.searchsorted(dts, d1, 'right'))
        assert dts[a] == d0 and dts[b - 1] == d1
        wins.append((nm, a, b))
base = lambda t: M['PRG'][t]


def line(tag, r):
    P('%-36s n=%d | annS %7.2f annB %7.2f | T1 excess %%/yr %7.2f (arith %7.2f) | T2 t %5.2f t551 %5.2f TE %%/d %.3f | '
      'T3 ddS %6.2f ddB %6.2f | NAV S %.4f B %.4f' % (tag, r['n'], r['annS'], r['annB'], r['t1'], r['t1_arith'], r['t'], r['t551'],
                                                    r['te'], r['ddS'], r['ddB'], r['navS'], r['navB']))


def describe(R, a, title):
    """Positions whose series ended while held; every descriptive column uses rows <= t (decision day) only,
    except the last-days path which ends at the last tradable day."""
    P(title)
    P('  code | decision day t | bought | last tradable | close_raw[t] | min close_raw rows t-20..t-1 | X1 (st in t-60..t) | X2 (close<1 in t-20..t) | '
      'ret5 %% | ret20 %% | limit-down closes rows t-9..t | amt60 quintile | score rank from top / pool | value last close / buy open | '
      'last 5 tradable closes (L = limit-down close)')
    for i in R['dead_idx']:
        lim_i = np.where(M['st'][:, i], 0.05, 0.10)
        jb = [j for j, (es, eb, hl) in enumerate(R['log']) if i in eb][-1]
        e = a + jb
        t = e - 1
        el = int(M['last_tr'][i])
        cr, pc, tr = M['cr'][:, i], M['pc'][:, i], M['tr'][:, i]
        ld = tr & (pc > 0) & (cr <= pc * (1 - lim_i) + 0.005)
        w = cr[t - 20:t]
        w = w[np.isfinite(w) & (w > 0)]
        npool = int(M['U'][t].sum())
        rk = npool - int(round(float(M['PRG'][t, i]) * npool)) + 1
        last = np.nonzero(tr[:el + 1])[0][-5:]
        P('  %s | %s | %s | %s | %.2f | %.2f | %d | %d | %6.1f | %6.1f | %d | %d | %d / %d | %.3f | %s' % (
            M['codes'][i], dts[t], dts[e], dts[el], cr[t], w.min() if w.size else np.nan, M['x1'][t, i], M['x2'][t, i],
            (M['cff0'][t, i] / M['cff0'][t - 5, i] - 1) * 100, (M['cff0'][t, i] / M['cff0'][t - 20, i] - 1) * 100, int(ld[t - 9:t + 1].sum()),
            int(M['qa'][t, i]), rk, npool, M['cff0'][el, i] / M['o_q'][e, i],
            ' '.join('%s:%.2f%s' % (dts[r][5:], cr[r], 'L' if ld[r] else '') for r in last)))


def main_window(nm, a, b):
    n = b - a
    yrs = n / 252.0
    P('\n==== WINDOW %s  %s..%s  %d days ====' % (nm, dts[a], dts[b - 1], n))
    sd = slice(a - 1, b - 1)                      # decision days
    U = M['U'][sd]
    tot = U.sum()
    P('exclusions on decision days: frozen-pool stock-days %d | X1 %.3f%% | X2 %.3f%% | X1 or X2 %.3f%% | mean pool size %.0f, mean excluded per day %.1f' % (
        tot, (U & M['x1'][sd]).sum() / tot * 100, (U & M['x2'][sd]).sum() / tot * 100, (U & M['xbuy'][sd]).sum() / tot * 100,
        tot / (n), (U & M['xbuy'][sd]).sum() / n))
    B1, B1z = Q.bench_B(M, a, b, K0), Q.bench_B(M, a, b, K0, zero_term=True)
    B2, B2z = Q.bench_B(M, a, b, K0, use_x=True), Q.bench_B(M, a, b, K0, zero_term=True, use_x=True)
    P('benchmark B: frozen pool of %s = %d stocks (series ended %d) | version-2 pool = %d stocks (series ended %d)' % (
        dts[a - 1], B1['npool'], B1['n_term'], B2['npool'], B2['n_term']))
    R1 = Q.run_account(M, a, b, K0, base, detail=True)
    R1s = Q.run_account(M, a, b, K0, base, zero_term=True)
    R2 = Q.run_account(M, a, b, K0, base, detail=True, use_x=True)
    R2s = Q.run_account(M, a, b, K0, base, zero_term=True, use_x=True)
    line('v1 last-close (round-1 main)', Q.readings(R1['nav'], B1['nav'], K0))
    line('v1 conservative, round-1 sizing', Q.readings(R1s['nav'], B1z['nav'], K0))
    line('v1 CONSERVATIVE (overlay)', Q.readings(R1['navz'], B1z['nav'], K0))
    rd = Q.readings(R2['navz'], B2z['nav'], K0)
    line('v2 MAIN = CONSERVATIVE (overlay)', rd)
    line('v2 last-close (report only)', Q.readings(R2['nav'], B2['nav'], K0))
    line('v2 conservative, round-1 sizing', Q.readings(R2s['nav'], B2z['nav'], K0))
    line('v2 conservative vs frozen-pool B', Q.readings(R2['navz'], B1z['nav'], K0))
    R02 = Q.run_account(M, a, b, K0, base, slip=2e-4, use_x=True)
    line('v2 slip 0.02% conservative (report)', Q.readings(R02['navz'], Q.bench_B(M, a, b, K0, slip=2e-4, zero_term=True, use_x=True)['nav'], K0))
    R20 = Q.run_account(M, a, b, 20000.0, base, use_x=True)
    line('v2 20k conservative (report)', Q.readings(R20['navz'], Q.bench_B(M, a, b, 20000.0, zero_term=True, use_x=True)['nav'], 20000.0))
    c = R2['c']
    mnav = R2['navz'].mean()
    turn = (c['buy_amt'] + c['sell_amt']) / 2 / mnav / yrs
    costr = c['cost'] / mnav / yrs * 100
    P('v2 account: mean holdings %.2f (min %d) | cash share mean %.4f | buys %d sells %d | turnover /yr %.2f | cost %%/yr %.2f | '
      'gross excess (arith net + cost) %.2f | cost / gross %.3f' % (R2['nheld'].mean(), R2['nheld'][1:].min(), R2['cashr'].mean(), c['nbuy'], c['nsell'],
                                                                   turn, costr, rd['t1_arith'] + costr, costr / (rd['t1_arith'] + costr)))
    P('v2 account events: buy blocked %d | one lot unaffordable %d | no cash for one lot %d | cut by cash %d | 1%% cap binding %d | '
      'pending-sell days %d | pending at end %d' % (c['fail_block'], c['fail_lot'], c['fail_cash'], c['cash_cut'], c['cap_bind'], c['pend_days'], R2['pend_final']))
    q = R2['qd'] / R2['qd'].sum()
    P('v2 holdings by amt60 quintile (lowest..highest | not in pool): %s' % ' '.join('%.3f' % v for v in q))
    P('series ended: v1 positions %d, carried %.2f%% of last-close NAV | v2 positions %d, carried %.2f%% of last-close NAV (%.0f CNY)' % (
        R1['n_term'], R1['dead_val'] / R1['nav'][-1] * 100, R2['n_term'], R2['dead_val'] / R2['nav'][-1] * 100, R2['dead_val']))
    describe(R1, a, 'VERSION 1 positions whose series ended:')
    describe(R2, a, 'VERSION 2 positions whose series ended:')
    yq = np.array([s[:4] + 'Q' + str((int(s[5:7]) - 1) // 3 + 1) for s in dts[a:b]])
    yy = np.array([s[:4] for s in dts[a:b]])
    sN, bN = np.r_[K0, R2['navz']], np.r_[1.0, B2z['nav']]
    for lab, key in (('YEAR', yy), ('QUARTER', yq)):
        for u in sorted(set(key)):
            ix = np.nonzero(key == u)[0]
            rs, rb = sN[ix[-1] + 1] / sN[ix[0]] - 1, bN[ix[-1] + 1] / bN[ix[0]] - 1
            P('  v2 main %s %s | days %3d | strategy %7.2f%% | B %7.2f%% | difference %7.2f pts' % (lab, u, ix.size, rs * 100, rb * 100, (rs - rb) * 100))
    hm = np.full((n, K), -1, np.int32)
    bm = np.full((n, K + 5), -1, np.int32)
    sm = np.full((n, 12), -1, np.int32)
    for j, (es, eb, hl) in enumerate(R2['log']):
        hm[j, :hl.size], bm[j, :len(eb)], sm[j, :len(es)] = hl, eb, es
    f = os.path.join(HERE, 'record_c2_%s_%s.npz' % (TAG, nm))
    np.savez_compressed(f, dates=dts[a:b], nav=R2['nav'], navB=B2['nav'], navz=R2['navz'], navBz=B2z['nav'], held=hm, buys=bm, sells=sm, codes=M['codes'])
    P('record saved: %s (%.2f MB)' % (f, os.path.getsize(f) / 2 ** 20))


if MODE == 'main':
    for nm, a, b in wins:
        main_window(nm, a, b)
        tick('window ' + nm)
elif MODE == 'perturb':
    for nm, a, b in wins:
        P('\n==== WINDOW %s target 5, version 2, conservative: 20 perturbed runs (start shift 0-4 x 4 seeds); annualised net excess over B, %%/yr ====' % nm)
        res = []
        for kshift in range(5):
            ak = a + kshift
            Bk = Q.bench_B(M, ak, b, K0, zero_term=True, use_x=True)
            r0_ = Q.readings(Q.run_account(M, ak, b, K0, base, use_x=True)['navz'], Bk['nav'], K0)
            row = []
            for s in range(4):
                rng = np.random.default_rng(2000 + kshift * 10 + s)
                Gw = M['G'][ak - 1:b].astype(np.float64)
                Uw = M['U'][ak - 1:b]
                sd_ = np.nanstd(np.where(Uw, Gw, np.nan), axis=1)
                PR = Q.pct_rank_rows(Gw + 1e-6 * sd_[:, None] * rng.standard_normal(Gw.shape), Uw & np.isfinite(Gw))
                rr = Q.readings(Q.run_account(M, ak, b, K0, lambda t, PR=PR, o=ak - 1: PR[t - o], use_x=True)['navz'], Bk['nav'], K0)
                row.append(rr['t1'])
                res.append(rr['t1'])
            P('  shift %d (start %s, %d days) | unperturbed %7.2f | perturbed %s' % (kshift, dts[ak], b - ak, r0_['t1'], ' '.join('%7.2f' % v for v in row)))
        res = np.array(res)
        P('  20 runs: min %.2f | mean %.2f | sd %.2f | sd/mean %.3f | target 5 (min > 0 and sd <= mean/3): %s' % (
            res.min(), res.mean(), res.std(ddof=1), res.std(ddof=1) / res.mean(), 'PASS' if res.min() > 0 and res.std(ddof=1) <= res.mean() / 3 else 'FAIL'))
        tick('perturb ' + nm)
else:
    for nm, a, b in wins:
        R = Q.run_account(M, a, b, K0, base, use_x=True)
        B = Q.bench_B(M, a, b, K0, zero_term=True, use_x=True)
        fin, fin_lc, hold = [], [], []
        for seed in range(1000, 1200):
            rng = np.random.default_rng(seed)
            Rr = Q.run_account(M, a, b, K0, lambda t: np.where(M['U'][t], rng.random(N), np.nan), use_x=True)
            fin.append(Rr['navz'][-1] / K0)
            fin_lc.append(Rr['nav'][-1] / K0)
            hold.append((Rr['nheld'].mean(), Rr['c']['nbuy'], Rr['n_term']))
        fin, fin_lc, hold = np.array(fin), np.array(fin_lc), np.array(hold)
        s = R['navz'][-1] / K0
        P('\n==== WINDOW %s target 4, version 2, conservative: random selection in the same buyable set, exact engine, 200 seeds ====' % nm)
        P('  final NAV: strategy %.4f | B %.4f | random mean %.4f sd %.4f min %.4f median %.4f q97.5 %.4f max %.4f | strategy above %d of 200 | z %.2f | target 4: %s'
          % (s, B['nav'][-1], fin.mean(), fin.std(ddof=1), fin.min(), np.median(fin), np.percentile(fin, 97.5), fin.max(), int((s > fin).sum()),
             (s - fin.mean()) / fin.std(ddof=1), 'PASS' if s > np.percentile(fin, 97.5) else 'FAIL'))
        P('  last-close valuation (report only): strategy %.4f | random mean %.4f q97.5 %.4f' % (R['nav'][-1] / K0, fin_lc.mean(), np.percentile(fin_lc, 97.5)))
        P('  rhythm: strategy mean holdings %.2f buys %d ended %d | random mean holdings %.2f buys %.0f ended per run %.2f' % (
            R['nheld'].mean(), R['c']['nbuy'], R['n_term'], hold[:, 0].mean(), hold[:, 1].mean(), hold[:, 2].mean()))
        tick('random ' + nm)
C.finish()
