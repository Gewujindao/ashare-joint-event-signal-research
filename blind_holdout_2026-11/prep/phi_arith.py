# phi_arith.py - arithmetic of a fixed invested fraction phi, on the saved daily NAV records only (description, no backtest).
# Input: evaluator\stageC2\record_c2_full_{V,SEL,HOLD}.npz, written by c2b_run.py main_window():
#   navz  = version-2 strategy NAV in CNY, slippage 0.1% per side, conservative valuation (main reading)
#   navBz = benchmark B NAV as a ratio to capital, conservative valuation
# No market data is read. All three windows have already been looked at: nothing here is out-of-sample evidence.
# usage: python.exe -X utf8 prep\phi_arith.py      (writes prep\phi_arith_out.txt)
import os
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REC = os.path.join(HERE, '..', 'evaluator', 'stageC2', 'record_c2_full_%s.npz')
OUT = os.path.join(HERE, 'phi_arith_out.txt')
K0 = 500000.0
NSLOTS = 50
MINCOMM = 5.0
PHIS = np.array([1.0, 0.9, 0.8, 0.75, 0.7, 0.6, 0.5, 0.4, 0.3])
GRID = np.arange(1, 1001) / 1000.0
DD_ABS = 17.0
# 'v2 MAIN = CONSERVATIVE (overlay)' readings and 'v2 account' trade counts, copied from evaluator\stageC2\out_c2_full.txt
WINDOWS = {
    'V': dict(d0='2020-01-03', d1='2022-12-30', annS=29.73, annB=7.19, t1=22.54, t=2.90, ddS=19.15, ddB=26.26, trades=3680 + 3630),
    'SEL': dict(d0='2023-01-03', d1='2024-06-28', annS=5.61, annB=-9.74, t1=15.35, t=1.39, ddS=36.93, ddB=29.29, trades=1840 + 1790),
    'HOLD': dict(d0='2024-07-01', d1='2026-10-09', annS=48.27, annB=21.11, t1=27.17, t=2.33, ddS=29.45, ddB=23.09, trades=2800 + 2750),
}
LINES = []


def P(s=''):
    print(s, flush=True)
    LINES.append(s)


def readings(phi, r, b):
    """Daily rebalancing to the fraction phi (1-d array); cash earns nothing; the rebalancing trades cost nothing."""
    phi = np.atleast_1d(np.asarray(phi, float))
    n = r.size
    nav = np.cumprod(1 + phi[:, None] * r[None, :], axis=1)
    navB = np.cumprod(1 + b)
    d = phi[:, None] * r[None, :] - b[None, :]
    annS = nav[:, -1] ** (252.0 / n) - 1
    annB = navB[-1] ** (252.0 / n) - 1
    path = np.c_[np.ones(phi.size), nav]
    ddS = (1 - path / np.maximum.accumulate(path, axis=1)).max(axis=1)
    pB = np.r_[1.0, navB]
    ddB = (1 - pB / np.maximum.accumulate(pB)).max()
    sd = d.std(axis=1, ddof=1)
    rebal = (phi[:, None] * (1 - phi[:, None]) * np.abs(r[None, :]) / (1 + phi[:, None] * r[None, :])).sum(axis=1) * 252.0 / n
    return dict(annS=annS * 100, annB=annB * 100, t1=(annS - annB) * 100, t=d.mean(axis=1) / sd * np.sqrt(n),
                t551=d.mean(axis=1) / sd * np.sqrt(551), te=sd * np.sqrt(252) * 100, ddS=ddS * 100, ddB=ddB * 100, nav=nav, rebal=rebal)


def runs(mask):
    """Contiguous runs of True on GRID, as text."""
    e = np.diff(np.r_[0, mask.astype(int), 0])
    lo, hi = np.nonzero(e == 1)[0], np.nonzero(e == -1)[0] - 1
    return ' , '.join('[%.3f, %.3f]' % (GRID[a], GRID[c]) for a, c in zip(lo, hi)) if lo.size else 'empty'


def yn(x):
    return 'yes' if x else 'no '


def one(tag, g, i):
    return '%s phi %.3f | annS %6.2f | T1 %6.2f | t %5.2f (t551 %5.2f) | TE %5.2f | ddS %5.2f (ddB %5.2f)' % (
        tag, GRID[i], g['annS'][i], g['t1'][i], g['t'][i], g['t551'][i], g['te'][i], g['ddS'][i], g['ddB'])


P('phi_arith.py : fixed invested fraction phi, daily rebalanced, cash at zero interest, rebalancing cost ignored')
P('main reading = slippage 0.1% per side, conservative valuation (navz against navBz); all three windows are already-seen data')
joint_all = np.ones(GRID.size, bool)
joint551_all = np.ones(GRID.size, bool)
dd17_all = np.ones(GRID.size, bool)
store = {}
for nm, W in WINDOWS.items():
    z = np.load(REC % nm, allow_pickle=False)
    dts, S, B = z['dates'], z['navz'].astype(np.float64), z['navBz'].astype(np.float64)
    n = dts.size
    assert dts[0] == W['d0'] and dts[-1] == W['d1']
    P('\n==== WINDOW %s  %s..%s  %d days ====' % (nm, dts[0], dts[-1], n))
    P('file %s' % os.path.normpath(REC % nm))
    P('members: %s' % ' '.join('%s%s' % (k, z[k].shape) for k in z.files))
    P('strategy daily series = navz (CNY, start capital %.0f) ; benchmark B daily series = navBz (ratio to capital)' % K0)
    r = S / np.r_[K0, S[:-1]] - 1
    b = B / np.r_[1.0, B[:-1]] - 1
    one1 = readings(1.0, r, b)
    for k in ('annS', 'annB', 't1', 't', 'ddS', 'ddB'):
        assert abs(float(np.ravel(one1[k])[0]) - W[k]) < 0.0051, (nm, k, one1[k], W[k])
    P('phi = 1 reproduces out_c2_full.txt (annS %.2f annB %.2f T1 %.2f t %.2f ddS %.2f ddB %.2f)' % tuple(W[k] for k in ('annS', 'annB', 't1', 't', 'ddS', 'ddB')))
    cov = np.cov(r, b, ddof=1)
    sum_r2_yr = (r ** 2).sum() * 252.0 / n
    P('moments: mean r %.2f bp/d | mean b %.2f bp/d | sd r %.3f %%/d | sd b %.3f %%/d | corr %.3f | slope of r on b %.3f | '
      'phi that minimises the tracking error = cov/var(r) %.3f | sum r^2 per year %.4f' % (
          r.mean() * 1e4, b.mean() * 1e4, np.sqrt(cov[0, 0]) * 100, np.sqrt(cov[1, 1]) * 100, cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]),
          cov[0, 1] / cov[1, 1], cov[0, 1] / cov[0, 0], sum_r2_yr))

    q = readings(PHIS, r, b)
    lnN1 = np.log(S[-1] / K0)
    apx_ann = (np.exp((PHIS * lnN1 + PHIS * (1 - PHIS) / 2 * (r ** 2).sum()) * 252.0 / n) - 1) * 100
    apx_dd = (1 - (1 - q['ddS'][0] / 100) ** PHIS) * 100
    P('  phi | annS %  | annB %  | T1 pts  >5 | t      >=2 | t551  | TE %/yr | ddS %  | ddB %  | T3 ddS<=ddB | ddS<=17 | approx annS | approx ddS | ignored rebalancing, x NAV/yr one way')
    for i, ph in enumerate(PHIS):
        P(' %4.2f | %7.2f | %7.2f | %7.2f %s | %5.2f  %s | %5.2f | %7.2f | %6.2f | %6.2f |     %s     |   %s   | %11.2f | %10.2f | %6.2f' % (
            ph, q['annS'][i], q['annB'], q['t1'][i], yn(q['t1'][i] > 5), q['t'][i], yn(q['t'][i] >= 2), q['t551'][i], q['te'][i], q['ddS'][i], q['ddB'],
            yn(q['ddS'][i] <= q['ddB']), yn(q['ddS'][i] <= DD_ABS), apx_ann[i], apx_dd[i], q['rebal'][i]))

    g = readings(GRID, r, b)
    m1, m2, m3, m17 = g['t1'] > 5, g['t'] >= 2, g['ddS'] <= g['ddB'], g['ddS'] <= DD_ABS
    joint = m1 & m2 & m3
    m2f = g['t551'] >= 2
    joint_all &= joint
    joint551_all &= m1 & m2f & m3
    dd17_all &= m17
    store[nm] = g
    P('grid phi = 0.001 .. 1.000 (step 0.001):')
    P('  T1 (excess > 5 pts) holds on        : %s' % runs(m1))
    P('  T2 (t >= 2) holds on                : %s' % runs(m2))
    P('  report only, t scaled to 551 days: t551 >= 2 holds on %s ; T1 and t551 >= 2 and T3 hold on %s' % (runs(m2f), runs(m1 & m2f & m3)))
    P('  T3 (ddS <= ddB) holds on            : %s' % runs(m3))
    P('  T1 and T2 and T3 hold on            : %s' % runs(joint))
    P('  T1 and T2 hold on                   : %s' % runs(m1 & m2))
    P('  ddS <= 17%% holds on                 : %s' % runs(m17))
    P('  T1 and T2 and ddS <= 17%% hold on    : %s' % runs(m1 & m2 & m17))
    P('  ' + one('largest t on the grid       :', g, int(g['t'].argmax())))
    P('  ' + one('largest phi with T3         :', g, int(np.nonzero(m3)[0][-1])))
    if (m1 & m2).any():
        P('  ' + one('smallest phi with T1 and T2 :', g, int(np.nonzero(m1 & m2)[0][0])))
    P('  ' + one('largest phi with ddS <= 17% :', g, int(np.nonzero(m17)[0][-1])))

    # implementation B: every buy sized phi * NAV / 50, no extra rebalancing. The 5 CNY minimum commission is a fixed amount per trade.
    yrs = n / 252.0
    navS1 = S
    P('implementation B (each buy scaled by phi, no extra rebalancing): minimum commission %.0f CNY per trade, %d trades in the window' % (MINCOMM, W['trades']))
    P('  phi | position at start CNY | commission per side at start, bp | extra per round trip against phi=1: at start, bp | window mean, bp | extra drag on the account, pts/yr')
    for i, ph in enumerate(PHIS):
        navphi = q['nav'][i] * K0
        pos0 = ph * K0 / NSLOTS
        side0 = MINCOMM / pos0 * 1e4
        extra0 = 2 * MINCOMM / pos0 * 1e4 - 2 * MINCOMM / (K0 / NSLOTS) * 1e4
        extra_mean = (2 * MINCOMM * NSLOTS / (ph * navphi) - 2 * MINCOMM * NSLOTS / navS1).mean() * 1e4
        drag = MINCOMM * W['trades'] / yrs * (1 / navphi - ph / navS1).mean() * 100
        P(' %4.2f | %21.0f | %32.2f | %48.2f | %15.2f | %32.2f' % (ph, pos0, side0, extra0, extra_mean, drag))
    assert bool((MINCOMM > 1e-4 * S / NSLOTS).all())       # the minimum binds on every day of the phi = 1 record
    z.close()

P('\n==== ALL THREE WINDOWS ====')
P('T1 and T2 and T3 in V, SEL and HOLD together : %s' % runs(joint_all))
P('report only, with t551 in place of t        : %s' % runs(joint551_all))
P('ddS <= 17%% in V, SEL and HOLD together       : %s' % runs(dd17_all))
ic = int(np.nonzero(dd17_all)[0][-1])
for nm in WINDOWS:
    P('  ' + one('%-4s at the common' % nm, store[nm], ic))
for nm in WINDOWS:
    g = store[nm]
    P('  %-4s T1 and T2 at each listed phi: %s' % (nm, ' | '.join('%.2f: %6.2f, %4.2f' % (ph, g['t1'][int(round(ph * 1000)) - 1], g['t'][int(round(ph * 1000)) - 1]) for ph in PHIS)))
with open(OUT, 'w', encoding='utf-8') as fh:
    fh.write('\n'.join(LINES) + '\n')
