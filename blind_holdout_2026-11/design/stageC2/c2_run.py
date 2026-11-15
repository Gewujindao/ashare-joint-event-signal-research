# c2_run.py — 丙二 readings. usage: c2_run.py NPZ DATA_END MODE TAG [CACHE]
#   MODE main    : frozen scores from scratch, list check against design\stageC1\record_SEL.npz, account + benchmark B,
#                  targets 1-3, decomposition against the set simulation, side readings, daily records for the prefix check.
#   MODE perturb : target 5 (20 perturbed runs per window).      MODE random : target 4 (200 random-selection seeds).
#   CACHE (optional): small scratch file with the window rows of the frozen scores, written by MODE main and read by
#                  the other modes to skip the 77 s feature pass. Without it every mode recomputes the scores.
#   Windows whose last day is beyond the loaded data are skipped (the hold-out window only exists in the evaluator's data).
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
PACK = ('G', 'U', 'qa', 'oo', 'xf', 'bf', 'msf')


def tick(msg):
    P('# [%5.1fs] %s  peak_ws_MB=%.0f' % (time.time() - C._T0, msg, C.peak_mb()[0]))


if MODE == 'main' or not (CACHE and os.path.exists(CACHE)):
    S = Q.frozen_scores(NPZ, DATA_END, TAG)
    T = S['dates'].size
    xf, bf, msf = Q.setsim(S['PRG'], S['U'], S['buy_ok'], S['sell_ok'], S['oo'], S['t0'], T - 1)
    rec = np.load(os.path.join(HERE, '..', 'stageC1', 'record_SEL.npz'), allow_pickle=False)
    keep = rec['dates'] < S['dates'][-1]      # all 1,085 recorded days unless the data is cut earlier (dry runs)
    k = np.searchsorted(S['dates'], rec['dates'][keep])
    assert k.size > 0 and bool((S['dates'][k] == rec['dates'][keep]).all())
    dx, db = np.abs(xf[k] - rec['x'][keep]).max(), np.abs(bf[k] - rec['buys'][keep]).max()
    P('list check: decide() in set mode against record_SEL.npz, %d days %s..%s ; max |x diff| %.3e ; max |buys diff| %.1f'
      % (k.size, rec['dates'][keep][0], rec['dates'][keep][-1], dx, db))
    if not (dx == 0 and db == 0):
        P('LIST CHECK FAILED - no account reading is produced')
        C.finish()
        sys.exit(4)
    r0 = int(S['iF']) - 25
    pack = dict(G=S['G'][r0:].copy(), U=S['U'][r0:].copy(), qa=S['qa'][r0:].copy(), oo=S['oo'][r0:].copy(),
                xf=xf[r0:], bf=bf[r0:], msf=msf[r0:])
    bo, so = S['buy_ok'][r0:].copy(), S['sell_ok'][r0:].copy()
    del S
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
    P('%-26s n=%d | annS %7.2f annB %7.2f | T1 excess %%/yr %7.2f (arith %7.2f) | T2 t %5.2f t551 %5.2f TE %%/d %.3f | '
      'T3 ddS %6.2f ddB %6.2f | NAV S %.4f B %.4f' % (tag, r['n'], r['annS'], r['annB'], r['t1'], r['t1_arith'], r['t'], r['t551'],
                                                    r['te'], r['ddS'], r['ddB'], r['navS'], r['navB']))


def o2o(v):
    return v[1:] / v[:-1] - 1


def main_window(nm, a, b):
    n = b - a
    yrs = n / 252.0
    P('\n==== WINDOW %s  %s..%s  %d days ====' % (nm, dts[a], dts[b - 1], n))
    B = Q.bench_B(M, a, b, K0)
    Bz = Q.bench_B(M, a, b, K0, zero_term=True)
    R = Q.run_account(M, a, b, K0, base, detail=True)
    Rz = Q.run_account(M, a, b, K0, base, zero_term=True)
    rd = Q.readings(R['nav'], B['nav'], K0)
    P('benchmark B: pool of %s = %d stocks, bought %d, series ended inside the window %d' % (dts[a - 1], B['npool'], B['nbought'], B['n_term']))
    line('MAIN 500k slip0.1%', rd)
    rz = Q.readings(Rz['nav'], Bz['nav'], K0)
    line('CONSERVATIVE (ended=0)', rz)
    line('  cons. strategy vs main B', Q.readings(Rz['nav'], B['nav'], K0))
    R02 = Q.run_account(M, a, b, K0, base, slip=2e-4)
    r02 = Q.readings(R02['nav'], Q.bench_B(M, a, b, K0, slip=2e-4)['nav'], K0)
    line('slip 0.02% (report only)', r02)
    R20 = Q.run_account(M, a, b, 20000.0, base)
    B20 = Q.bench_B(M, a, b, 20000.0)
    line('20k tier (report only)', Q.readings(R20['nav'], B20['nav'], 20000.0))
    P('20k tier: mean holdings %.1f, cash share %.3f, buys %d, lot-unaffordable %d, B bought %d of %d' % (
        R20['nheld'].mean(), R20['cashr'].mean(), R20['c']['nbuy'], R20['c']['fail_lot'], B20['nbought'], B20['npool']))
    c = R['c']
    mnav = R['nav'].mean()
    turn = (c['buy_amt'] + c['sell_amt']) / 2 / mnav / yrs
    costr = c['cost'] / mnav / yrs * 100
    P('account: mean holdings %.2f (min %d) | cash share mean %.4f | buys %d sells %d | turnover /yr %.2f | cost %%/yr %.2f | '
      'gross excess (arith net + cost) %.2f | cost / gross %.3f' % (R['nheld'].mean(), R['nheld'][1:].min(), R['cashr'].mean(), c['nbuy'], c['nsell'],
                                                                   turn, costr, rd['t1_arith'] + costr, costr / (rd['t1_arith'] + costr)))
    P('account events: buy blocked %d | one lot unaffordable %d | no cash for one lot %d | cut by cash %d | 1%% cap binding %d | '
      'pending-sell days %d | pending at end %d' % (c['fail_block'], c['fail_lot'], c['fail_cash'], c['cash_cut'], c['cap_bind'], c['pend_days'], R['pend_final']))
    q = R['qd'] / R['qd'].sum()
    P('holdings by amt60 quintile (lowest..highest | not in pool): %s' % ' '.join('%.3f' % v for v in q))
    P('series ended: positions %d (codes %s; last tradable %s) | value carried at window end %.0f CNY = %.3f%% of NAV | conservative NAV %.4f vs main %.4f'
      % (R['n_term'], ','.join(M['codes'][R['dead_idx']]), ','.join(dts[M['last_tr'][R['dead_idx']]]), R['dead_val'], R['dead_val'] / R['nav'][-1] * 100,
         Rz['nav'][-1] / K0, R['nav'][-1] / K0))
    # ---- decomposition against the set simulation (open-to-open days e = a..b-2, against the daily equal-weight pool)
    xs, bs_, ms_ = Q.setsim(M['PRG'], M['U'], M['buy_ok'], M['sell_ok'], M['oo'], a - 1, b - 1)
    sl = slice(a, b - 1)
    ms = ms_[sl]
    assert np.isfinite(ms).all() and np.nanmax(np.abs(ms - M['msf'][sl])) == 0
    ann = lambda v: v.mean() * 25200
    P('DECOMPOSITION (%d open-to-open days; %%/yr, arithmetic)' % ms.size)
    for cn, cc in (('strict 35.2bp', 35.2e-4), ('mine 19.2bp', 19.2e-4)):
        P('  set sim, continuous path from 2015 (= record): gross %.2f | net %s %.2f' % (ann(M['xf'][sl]), cn, ann(M['xf'][sl] - M['bf'][sl] / K * cc - 0.5e-4)))
        P('  set sim restarted empty at the window start  : gross %.2f | net %s %.2f | buys/day %.3f' % (ann(xs[sl]), cn, ann(xs[sl] - bs_[sl] / K * cc - 0.5e-4), bs_[sl].mean()))
    for sp, sn in ((1e-3, 'slip 0.1%'), (2e-4, 'slip 0.02%')):
        Ag = Q.run_account(M, a, b, K0, base, lots=False, nofee=True)
        A0 = Q.run_account(M, a, b, K0, base, slip=sp, lots=False, mincomm=0.0)
        A1 = Q.run_account(M, a, b, K0, base, slip=sp, lots=True, mincomm=0.0, frac_unaff=True)
        A2 = Q.run_account(M, a, b, K0, base, slip=sp, lots=True, mincomm=0.0)
        A3 = Q.run_account(M, a, b, K0, base, slip=sp)
        Bs = Q.bench_B(M, a, b, K0, slip=sp)
        r3 = Q.readings(A3['nav'], Bs['nav'], K0)
        P('  [%s] account vs pool: frictionless fractional %.2f | A0 +costs (no lots, no min commission) %.2f | A1 +whole lots %.2f | '
          'A2 +unaffordable buys dropped %.2f | A3 +min commission (= engine) %.2f' % (sn, ann(o2o(Ag['navo']) - ms), ann(o2o(A0['navo']) - ms),
                                                                                     ann(o2o(A1['navo']) - ms), ann(o2o(A2['navo']) - ms), ann(o2o(A3['navo']) - ms)))
        P('  [%s] engine vs B open-to-open %.2f | vs B close-to-close arithmetic (all %d days) %.2f | compounded = target 1 %.2f | '
          'pool minus B %.2f | cash share A0 %.4f A1 %.4f A2 %.4f A3 %.4f | holdings A1 %.2f A2 %.2f' % (
              sn, ann(o2o(A3['navo']) - o2o(Bs['navo'])), n, r3['t1_arith'], r3['t1'], ann(ms - o2o(Bs['navo'])),
              A0['cashr'].mean(), A1['cashr'].mean(), A2['cashr'].mean(), A3['cashr'].mean(), A1['nheld'].mean(), A2['nheld'].mean()))
    # ---- by year and by quarter (main reading)
    yq = np.array([s[:4] + 'Q' + str((int(s[5:7]) - 1) // 3 + 1) for s in dts[a:b]])
    yy = np.array([s[:4] for s in dts[a:b]])
    sN, bN = np.r_[K0, R['nav']], np.r_[1.0, B['nav']]
    for lab, key in (('YEAR', yy), ('QUARTER', yq)):
        for u in sorted(set(key)):
            ix = np.nonzero(key == u)[0]
            rs, rb = sN[ix[-1] + 1] / sN[ix[0]] - 1, bN[ix[-1] + 1] / bN[ix[0]] - 1
            P('  %s %s | days %3d | strategy %7.2f%% | B %7.2f%% | difference %7.2f pts | arith excess %%/yr %7.2f' % (
                lab, u, ix.size, rs * 100, rb * 100, (rs - rb) * 100, rd['d'][ix].mean() * 25200))
    # ---- daily record for the prefix check
    hm = np.full((n, K), -1, np.int32)
    bm = np.full((n, K + 5), -1, np.int32)
    sm = np.full((n, 12), -1, np.int32)
    for j, (es, eb, hl) in enumerate(R['log']):
        hm[j, :hl.size], bm[j, :len(eb)], sm[j, :len(es)] = hl, eb, es
    f = os.path.join(HERE, 'record_c2_%s_%s.npz' % (TAG, nm))
    np.savez_compressed(f, dates=dts[a:b], nav=R['nav'], navB=B['nav'], held=hm, buys=bm, sells=sm, codes=M['codes'])
    P('record saved: %s (%.2f MB)' % (f, os.path.getsize(f) / 2 ** 20))
    return B


if MODE == 'main':
    first = True
    for nm, a, b in wins:
        B = main_window(nm, a, b)
        tick('window ' + nm)
        if first:   # benchmark B against the stage-3 engine's hold accounts
            first = False
            sys.path.insert(0, os.path.join(HERE, '..', 'stage3'))
            import engine
            ix = B['idx']
            eq = engine.simulate(M['tr'][a:b][:, ix], M['st'][a:b][:, ix], M['o_raw'][a:b][:, ix], M['pc'][a:b][:, ix], M['o_q'][a:b][:, ix],
                                 M['cq'][a:b][:, ix], np.ones((b - a, ix.size), bool), K0, M['stamp'][a:b])[0]
            P('benchmark B vs design\\stage3\\engine.simulate hold accounts (%s, %d accounts): max |NAV diff| %.3e' % (nm, ix.size, np.abs(eq.mean(1) / K0 - B['nav']).max()))
            del eq
elif MODE == 'perturb':
    for nm, a, b in wins:
        P('\n==== WINDOW %s target 5: 20 perturbed runs (start shift 0-4 days x 4 seeds); annualised net excess over B, %%/yr ====' % nm)
        res = []
        for kshift in range(5):
            ak = a + kshift
            Bk = Q.bench_B(M, ak, b, K0)
            r0_ = Q.readings(Q.run_account(M, ak, b, K0, base)['nav'], Bk['nav'], K0)
            row = []
            for s in range(4):
                rng = np.random.default_rng(2000 + kshift * 10 + s)
                Gw = M['G'][ak - 1:b].astype(np.float64)
                Uw = M['U'][ak - 1:b]
                sd = np.nanstd(np.where(Uw, Gw, np.nan), axis=1)
                PR = Q.pct_rank_rows(Gw + 1e-6 * sd[:, None] * rng.standard_normal(Gw.shape), Uw & np.isfinite(Gw))
                rr = Q.readings(Q.run_account(M, ak, b, K0, lambda t, PR=PR, o=ak - 1: PR[t - o])['nav'], Bk['nav'], K0)
                row.append(rr['t1'])
                res.append(rr['t1'])
            P('  shift %d (start %s, %d days) | unperturbed %7.2f | perturbed %s' % (kshift, dts[ak], b - ak, r0_['t1'], ' '.join('%7.2f' % v for v in row)))
        res = np.array(res)
        P('  20 runs: min %.2f | mean %.2f | sd %.2f | sd/mean %.3f | target 5 (min > 0 and sd <= mean/3): %s' % (
            res.min(), res.mean(), res.std(ddof=1), res.std(ddof=1) / res.mean(), 'PASS' if res.min() > 0 and res.std(ddof=1) <= res.mean() / 3 else 'FAIL'))
        tick('perturb ' + nm)
else:
    for nm, a, b in wins:
        R = Q.run_account(M, a, b, K0, base)
        B = Q.bench_B(M, a, b, K0)
        fin, hold = [], []
        for seed in range(1000, 1200):
            rng = np.random.default_rng(seed)
            Rr = Q.run_account(M, a, b, K0, lambda t: np.where(M['U'][t], rng.random(N), np.nan))
            fin.append(Rr['nav'][-1] / K0)
            hold.append((Rr['nheld'].mean(), Rr['c']['nbuy']))
        fin, hold = np.array(fin), np.array(hold)
        s = R['nav'][-1] / K0
        P('\n==== WINDOW %s target 4: random selection, exact engine, 200 seeds ====' % nm)
        P('  final NAV: strategy %.4f | B %.4f | random mean %.4f sd %.4f min %.4f median %.4f q97.5 %.4f max %.4f | strategy rank: above %d of 200 | z %.2f | target 4: %s'
          % (s, B['nav'][-1], fin.mean(), fin.std(ddof=1), fin.min(), np.median(fin), np.percentile(fin, 97.5), fin.max(), int((s > fin).sum()),
             (s - fin.mean()) / fin.std(ddof=1), 'PASS' if s > np.percentile(fin, 97.5) else 'FAIL'))
        P('  rhythm: strategy mean holdings %.2f buys %d | random mean holdings %.2f buys %.0f' % (R['nheld'].mean(), R['c']['nbuy'], hold[:, 0].mean(), hold[:, 1].mean()))
        tick('random ' + nm)
C.finish()
