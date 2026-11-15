# test_engine.py — synthetic checks of engine.py against hand-computed values (no data file is read).
# Each expected number below was worked out by hand in the stage-3A report (section 2).
from engine import *

start('test_engine')
nan = np.nan


def mk(opens, closes, pres, tradable=None, st=None):
    T = len(opens)
    f = lambda a: np.array(a, dtype=np.float64).reshape(T, 1)
    tr = np.ones((T, 1), bool) if tradable is None else np.array(tradable, bool).reshape(T, 1)
    s = np.zeros((T, 1), bool) if st is None else np.array(st, bool).reshape(T, 1)
    return tr, s, f(opens), f(pres), f(opens), f(closes)


def run(case, want, K=500000.0):
    tr, st, o, pc, oq, cq = case
    w = np.array(want, bool).reshape(len(want), 1)
    stamp = np.full(len(want), STAMP_OLD)
    return simulate(tr, st, o, pc, oq, cq, w, K, stamp)


def check(name, got, exp, tol=0.01):
    ok = abs(got - exp) <= tol
    P('%-52s engine %.6f  hand %.6f  %s' % (name, got, exp, 'OK' if ok else 'MISMATCH'))
    return ok


allok = True
# A: buy at open 0, sell at open 2 (10.50), buy back at open 4 (10.00); old stamp tax 0.1%
A = mk([10.0, 10.0, 10.5, 10.3, 10.0, 10.1], [10.0, 10.2, 10.4, 10.1, 10.1, 10.2], [10.0, 10.0, 10.2, 10.4, 10.1, 10.1])
eq, pos, cost, nb, ns = run(A, [1, 1, 0, 0, 1, 1])
allok &= check('A day0 equity after first buy', eq[0, 0], 499446.05511)
allok &= check('A day1 equity', eq[1, 0], 509426.05511)
allok &= check('A day2 equity after sell at 10.50', eq[2, 0], 523291.1021945)
allok &= check('A day4 equity after buy-back at 10.00', eq[4, 0], 527931.6247745)
allok &= check('A day5 final equity', eq[5, 0], 533151.6247745)
allok &= check('A number of buys', nb[0], 2, 0) and check('A number of sells', ns[0], 1, 0)
eqh, posh, costh, nbh, nsh = run(A, [1, 1, 1, 1, 1, 1])
allok &= check('A hold account final equity', eqh[5, 0], 509426.05511)
# B: buy-back blocked by a limit-up open on day 4 (11.11 vs preclose 10.10), filled on day 5 at 11.50
B = mk([10.0, 10.0, 10.5, 10.3, 11.11, 11.5], [10.0, 10.2, 10.4, 10.1, 11.11, 11.6], [10.0, 10.0, 10.2, 10.4, 10.1, 11.11])
eq, pos, cost, nb, ns = run(B, [1, 1, 0, 0, 1, 1])
allok &= check('B day4 equity (buy blocked, still cash)', eq[4, 0], 523291.1021945)
allok &= check('B day4 holding flag (0 = not bought)', float(pos[4, 0]), 0, 0)
allok &= check('B day5 final equity (bought at 11.50)', eq[5, 0], 527251.5137635)
# C: sell blocked by a limit-down open on day 2 (9.18 vs preclose 10.20), filled on day 3 at 9.00
C = mk([10.0, 10.0, 9.18, 9.0, 9.0, 9.0], [10.0, 10.2, 9.18, 9.1, 9.0, 9.0], [10.0, 10.0, 10.2, 9.18, 9.1, 9.0])
eq, pos, cost, nb, ns = run(C, [1, 1, 0, 0, 0, 0])
allok &= check('C day2 equity (sell blocked, still holding)', eq[2, 0], 458528.05511)
allok &= check('C day3 equity (sold at 9.00)', eq[3, 0], 448598.952611)
allok &= check('C day5 final equity (stays in cash)', eq[5, 0], 448598.952611)
# D: suspended on day 3 (return 0, sell deferred to day 4 at 10.00)
D = mk([10.0, 10.0, 10.5, nan, 10.0, 10.1], [10.0, 10.2, 10.4, nan, 10.1, 10.2], [10.0, 10.0, 10.2, nan, 10.4, 10.1],
       tradable=[1, 1, 1, 0, 1, 1])
eq, pos, cost, nb, ns = run(D, [1, 1, 1, 0, 0, 0])
allok &= check('D day2 equity', eq[2, 0], 519406.05511)
allok &= check('D day3 equity (suspended: unchanged)', eq[3, 0], 519406.05511)
allok &= check('D day4 equity (sold at 10.00)', eq[4, 0], 498393.719)
# E: 20k account, minimum commission 5 CNY; E2: minimum commission forces one lot fewer
eq, pos, cost, nb, ns = run(A, [1, 1, 1, 1, 1, 1], K=20000.0)
allok &= check('E 20k day0 equity (1900 shares, fee 5)', eq[0, 0], 19975.80981)
eq, pos, cost, nb, ns = run(A, [1, 1, 1, 1, 1, 1], K=10014.0)
allok &= check('E2 10,014 CNY day0 equity (900 shares)', eq[0, 0], 9000 + 999.90991)
# F: fill rules with the ST 5% band
tr1 = np.ones((1, 1), bool)
f = lambda x: np.array([[x]], dtype=np.float64)
so, bo = fills(tr1, np.array([[True]]), f(10.5), f(10.0), f(10.5))
allok &= check('F ST: buy at +5% open blocked (0)', float(bo[0, 0]), 0, 0)
so, bo = fills(tr1, np.array([[False]]), f(10.5), f(10.0), f(10.5))
allok &= check('F non-ST: buy at +5% open allowed (1)', float(bo[0, 0]), 1, 0)
so, bo = fills(tr1, np.array([[True]]), f(9.5), f(10.0), f(9.5))
allok &= check('F ST: sell at -5% open blocked (0)', float(so[0, 0]), 0, 0)
# G: state machine. Event at close of day 1 -> out at opens 2,3,4, back at open 5
T = 8
board = np.zeros((T, 1), bool)
board[1, 0] = True
okm = np.ones((T, 1), bool)
w = plan(board, okm, okm)
allok &= check('G want sequence (binary 11000111 = 199)', int(''.join(str(int(x)) for x in w[:, 0]), 2), 199, 0)
sb = okm.copy()
sb[2, 0] = False                      # sell blocked on day 2 -> sold on day 3 -> back at open 6
w = plan(board, sb, okm)
allok &= check('G sell blocked on day 2 (11000011 = 195)', int(''.join(str(int(x)) for x in w[:, 0]), 2), 195, 0)
# H: metrics helpers
m, ratio, t, tnw = tstats(np.array([1.0, 2.0, 3.0, 4.0]))
allok &= check('H t of [1,2,3,4] = 2*2.5/1.290994', t, 3.872983, 1e-5)
allok &= check('H max drawdown of 100 -> 110 -> 99 -> 120', float(maxdd(np.array([[110.0], [99.0], [120.0]]), 100.0)[0]), 0.1, 1e-9)
# I: metrics on case A: target-1 series and the fast account model used by the controls
tr, st, o, pc, oq, cq = A
eqS, posS, costS, nbS, nsS = run(A, [1, 1, 0, 0, 1, 1])
res, X = metrics(eqS, eqh, posS, 500000.0, oq, cq, tr, nbS, nsS, SLIP + COMM + TRANSFER + STAMP_OLD, seeds=20)
allok &= check('I day2 return difference (523291.10/509426.06 - 518960+446.06 ...)', X[2],
               523291.1021945 / 509426.05511 - (49900 * 10.4 + 446.05511) / 509426.05511, 1e-9)
allok &= check('I fast-model NAV vs hand 1.071*(1-0.00111)^2*(1-0.00211)', res['nav_fast'],
               1.071 * (1 - 0.00111) ** 2 * (1 - 0.00211), 1e-9)
P('I exact engine NAV %.6f vs fast model %.6f (difference = idle cash from whole lots)' % (res['nav_S'], res['nav_fast']))
P('ALL CHECKS PASSED' if allok else 'SOME CHECKS FAILED')
finish()
