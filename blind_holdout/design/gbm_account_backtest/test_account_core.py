# test_account_core.py — synthetic checks of account_core against hand-computed values (no data file is read).
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import account_core as Q
from account_core import C, np, P

C.start('c2_test')
nan = np.nan
T, N = 6, 7   # stocks A..G = 0..6
A, B, Cc, D, E, F, G = range(7)
o_raw = np.array([
    # A      B      C      D      E     F      G
    [10.00, 33.00, 20.00, 10.00, 5.00, 600.0, 10.00],
    [10.50, 33.60, 20.00, 10.00, 5.00, 600.0, 10.00],
    [10.40, 30.60, 20.60, 11.00, 5.00, 600.0, 10.00],
    [10.40, 29.00, 20.90, 11.00, 5.00, 600.0, 10.00],
    [10.40, 29.00, 21.10, 11.00, 5.00, 600.0, 10.00],
    [10.40, 29.00, 21.00, 11.00, nan, 600.0, 10.00]])
c_raw = np.array([
    [10.20, 33.50, 20.00, 10.00, 5.00, 600.0, 10.00],
    [10.40, 34.00, 20.50, 10.00, 5.00, 600.0, 10.00],
    [10.40, 30.60, 20.80, 11.00, 5.00, 600.0, 10.00],
    [10.40, 29.00, 21.00, 11.00, 5.10, 600.0, 10.00],
    [10.40, 29.00, 21.20, 11.00, 5.00, 600.0, 10.10],
    [10.40, 29.00, 21.00, 11.00, nan, 600.0, 10.00]])
pc = np.vstack([o_raw[:1], c_raw[:-1]])
pc[0] = o_raw[0]
tr = np.ones((T, N), bool)
tr[5, E] = False                      # E's series ends after day 4
fac = np.ones(N)
fac[Cc] = 0.5                         # C: qfq = raw * 0.5 (checks the unit conversion)
o_q, c_q = o_raw * fac, c_raw * fac
st = np.zeros((T, N), bool)
sell_ok, buy_ok = C.exec_maps(dict(tradable=tr, st=st, open_raw=o_raw, preclose_raw=pc, open_qfq=np.where(tr, o_q, nan)))
adv = np.full((T, N), 1e12)
adv[3, E] = 1e6                       # 1% cap = 10,000 CNY on day 3
M = dict(o_raw=np.nan_to_num(o_raw), o_q=np.nan_to_num(o_q), off0=np.nan_to_num(Q.ffill(np.where(tr, o_q, nan))),
         cff0=np.nan_to_num(Q.ffill(np.where(tr, c_q, nan))), buy_ok=buy_ok, sell_ok=sell_ok, adv=adv,
         last_tr=np.array([5, 5, 5, 5, 4, 5, 5]), stamp=np.full(T, Q.STAMP_OLD),
         U=np.ones((T, N), bool), qa=np.zeros((T, N), np.uint8))
I = lambda *a: np.array(a, np.int64)
orders = {0: (I(), I(A, B)), 1: (I(A), I(Cc)), 2: (I(B), I(D)), 3: (I(), I(E)), 4: (I(), I(F, G))}
allok = True


def check(name, got, exp, tol=0.005):
    global allok
    ok = abs(got - exp) <= tol
    allok &= ok
    P('%-62s engine %.6f  hand %.6f  %s' % (name, got, exp, 'OK' if ok else 'MISMATCH'))


R = Q.run_account(M, 0, T, 100000.0, nslots=2, orders=orders)
check('1 day0 build: A 4900 sh, B 1500 sh, min commission twice', R['nav'][0], 101620.514015)
check('2 day1 sell A at 10.50 (old stamp) + buy C 2500 sh', R['nav'][1], 104926.5061245)
check('3 day2 sell B blocked (limit-down open) + buy D blocked', R['nav'][2], 100576.5061245)
check('4 day3 pending B sold at 29.00 + E capped at 1% ADV (1900 sh)', R['nav'][3], 98759.5199645)
check('5 day4 F: one lot unaffordable; G cut by cash to 3600 sh', R['nav'][4], 99388.1596045)
check('6 day5 E series ended: main keeps last close', R['nav'][5], 98528.1596045)
Rz = Q.run_account(M, 0, T, 100000.0, nslots=2, orders=orders, zero_term=True)
check('7 day5 E series ended: conservative marks it 0', Rz['nav'][5], 89028.1596045)
check('7b conservative equals main before the end of the series', Rz['nav'][4], 99388.1596045)
check('7c conservative overlay on the last-close account, day5 (stage C2 supplement main reading)', R['navz'][5], 89028.1596045)
check('7c conservative overlay, day4 (nothing ended yet)', R['navz'][4], 99388.1596045)
f_, s_, b_ = Q.decide(np.array([0.2, 0.5, 0.9, 0.7], np.float32), np.ones(4, bool), np.array([1, 1, 0, 0], bool), np.ones(4, bool), np.ones(4, bool), 2, 1,
                      xb=np.array([0, 0, 1, 0], bool))
check('12b decide (version 2): best candidate (2) excluded by X -> buy 3, sell 0', s_.tolist() == [0] and b_.tolist() == [3], 1, 0)
f_, s_, b_ = Q.decide(np.array([0.2, 0.5, 0.9, 0.7], np.float32), np.ones(4, bool), np.array([1, 1, 0, 0], bool), np.ones(4, bool), np.ones(4, bool), 2, 1,
                      xb=np.array([1, 0, 1, 1], bool))
check('12b decide (version 2): held stock flagged by X is not sold for that reason; no candidate left', s_.size == 0 and b_.size == 0, 1, 0)
c = R['c']
for nm, exp in (('nbuy', 5), ('nsell', 2), ('pend_days', 1), ('fail_block', 1), ('fail_lot', 1), ('cap_bind', 1), ('cash_cut', 1), ('fail_cash', 0)):
    check('8 counter ' + nm, c[nm], exp, 0)
check('8 positions that met a series end', R['n_term'], 1, 0)
check('9 NAV at the open of day1 before trading (4900*10.5+1500*33.6+cash)', R['navo'][1], 1390.514015 + 51450 + 50400)
# total cost identity: capital + mark-to-market gains - costs = NAV
R0 = Q.run_account(M, 0, 1, 100000.0, nslots=2, orders={0: (I(), I(A))}, lots=False, mincomm=0.0)
check('10 fractional mode (decomposition only): 4995.004995 sh of A', R0['nav'][0], 100943.550949)
# hold account of benchmark B: backtest_engine hand value (K = 500,000, open 10.00, close 10.00 -> 499,446.05511)
Mh = dict(M)
Mh['U'] = np.zeros((T, N), bool)
Mh['U'][0, G] = True
Bh = Q.bench_B(Mh, 1, 3, 500000.0)
check('11 benchmark hold account (49,900 sh of G) equity', Bh['nav'][0] * 500000.0, 499446.05511)
# decision rule, 2 slots, 1 swap per day
u = np.ones(4, bool)
ok4 = np.ones(4, bool)
held = np.array([1, 1, 0, 0], bool)
f_, s_, b_ = Q.decide(np.array([0.2, 0.5, 0.9, 0.7], np.float32), u, held, ok4, ok4, 2, 1)
check('12 decide: sell lowest held (0), buy best candidate (2)', s_.tolist() == [0] and b_.tolist() == [2] and f_.size == 0, 1, 0)
f_, s_, b_ = Q.decide(np.array([0.2, 0.5, 0.9, 0.7], np.float32), u, held, ok4, np.array([0, 1, 1, 1], bool), 2, 1)
check('12 decide: lowest held unsellable -> next lowest (1) is sold', s_.tolist() == [1] and b_.tolist() == [2], 1, 0)
f_, s_, b_ = Q.decide(np.array([0.2, 0.5, 0.1, 0.15], np.float32), u, held, ok4, ok4, 2, 1)
check('12 decide: candidate not higher -> no swap', s_.size == 0 and b_.size == 0, 1, 0)
f_, s_, b_ = Q.decide(np.array([nan, 0.5, 0.1, 0.15], np.float32), np.array([0, 1, 1, 1], bool), held, ok4, ok4, 2, 1)
check('12 decide: held stock out of pool is sold first', s_.tolist() == [0] and b_.tolist() == [3], 1, 0)
f_, s_, b_ = Q.decide(np.array([0.2, 0.5, 0.9, 0.7], np.float32), u, np.array([1, 0, 0, 0], bool), np.array([1, 1, 0, 1], bool), ok4, 2, 1)
check('12 decide: empty slot filled by best buyable (3), then swap 0 -> 1', f_.tolist() == [3] and s_.tolist() == [0] and b_.tolist() == [1], 1, 0)
check('13 max drawdown 100 -> 110 -> 99 -> 120', Q.maxdd(np.array([100.0, 110.0, 99.0, 120.0])), 0.1, 1e-12)
rd = Q.readings(np.array([101.0, 103.02]), np.array([1.0, 1.01]), 100.0)
check('13 readings: daily difference day0 = 1%, day1 = 2% - 1%', rd['d'][0] + rd['d'][1], 0.02, 1e-12)
P('ALL CHECKS PASSED' if allok else 'SOME CHECKS FAILED')
C.finish()
