# c3_scores.py — 丙三 step 2: frozen scores on the full panel (data_full), list check, score cache for the descriptive
# decomposition, and the reproduction of the evaluator's hold-out readings (version 2, conservative overlay).
# Nothing frozen is changed: the score code is the source text of design\stageC1\c1_run.py executed verbatim (same three
# textual substitutions as c2_core.frozen_scores, output prefix c3_), the account engine is design\stageC2\c2_core.py.
# Extra over c2_core.frozen_scores: the 0..199 rank arrays of amt60 and fcap are kept (strata), and the held lists of the
# set simulation (frozen rule, no exclusions) are recorded so that the 丙一 decomposition can be repeated.
# usage: c3_scores.py
import sys, os, hashlib, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'stageC2'))
import c2_core as Q
from c2_core import C, np, sqrt, P, K
from common import ffill, pct_rank_rows
ROOT = r'<BLIND_ROOT>'
NPZ = ROOT + r'\data_full\ashare_mainboard_full_20261009.npz'
DATA_END = '2026-10-09'
CACHE = os.path.join(HERE, 'cache_c3_scores.npz')
EV = ROOT + r'\evaluator\stageC2'
WINDOWS = [('V', '2020-01-03', '2022-12-30'), ('SEL', '2023-01-03', '2024-06-28'), ('HOLD', '2024-07-01', '2026-10-09')]
K0 = 500000.0


def tick(msg):
    P('# [%5.1fs] %s  peak_ws_MB=%.0f' % (time.time() - C._T0, msg, C.peak_mb()[0]))


src = open(Q.C1, 'rb').read()
assert hashlib.sha256(src).hexdigest()[:16].upper() == Q.C1_SHA16, 'c1_run.py differs from the frozen version'
txt = src.decode('utf-8-sig').replace('\r\n', '\n')
txt = txt[:txt.index('PRG = pct_rank_rows(G, U & np.isfinite(G))')]
for old, new in ((" and DATA_END <= '2024-06-28'", ''), ('C.OUTDIR = OUT', 'C.OUTDIR = HERE2'), ("start('c1_' + TAG)", "start('c3_' + TAG)")):
    assert txt.count(old) == 1, old
    txt = txt.replace(old, new)
C.NPZ = NPZ
argv = sys.argv
sys.argv = [Q.C1, DATA_END, 'extend', 'scores', 'none', 'none']
ns = {'__name__': 'c1_prefix', 'HERE2': HERE}
exec(compile(txt, Q.C1, 'exec'), ns)
sys.argv = argv
dates, G, U, t0, iF = ns['dates'], ns['G'], ns['U'], ns['t0'], ns['iF']
r0 = int(iF) - 25
ia, ic = ns['NAMES'].index('amt60'), ns['NAMES'].index('fcap')
ra, rc = ns['RK'][ia][r0:].copy(), ns['RK'][ic][r0:].copy()
qa = ns['RK'][ia] // 40
del ns['RK']
PRG = pct_rank_rows(G, U & np.isfinite(G))
Off = ffill(ns['OQ'])
oo = np.full(G.shape, np.nan, np.float32)
oo[:-1] = Off[1:] / Off[:-1] - 1
del Off
buy_ok, sell_ok = ns['buy_ok'], ns['sell_ok']
ns.clear()
T, N = G.shape
tick('frozen scores ready, rows %d, last %s' % (T, dates[-1]))

# set simulation of c1_run.py (frozen rule, no exclusions) with the held lists kept; same loop as c2_core.setsim
held = np.zeros(N, bool)
xf = np.full(T, np.nan)
bf = np.zeros(T)
sh = np.full((T - r0, K), -1, np.int32)
for t in range(t0, T - 1):
    e = t + 1
    u = U[t]
    f_, s_, c_ = Q.decide(PRG[t], u, held, buy_ok[e], sell_ok[e])
    held[f_] = True
    held[s_] = False
    held[c_] = True
    if e >= r0:
        hl = np.nonzero(held)[0]
        sh[e - r0, :hl.size] = hl
    r = oo[e]
    fin = np.isfinite(r)
    hv = held & fin
    b = r[hv].sum(dtype=np.float64) / K
    uu = u & fin
    if not uu.any():
        continue
    mm = r[uu].mean(dtype=np.float64)
    xf[e], bf[e] = b - mm, f_.size + c_.size
rec = np.load(os.path.join(HERE, '..', 'stageC1', 'record_SEL.npz'), allow_pickle=False)
keep = rec['dates'] < dates[-1]
k = np.searchsorted(dates, rec['dates'][keep])
assert k.size > 0 and bool((dates[k] == rec['dates'][keep]).all())
dx, db = np.abs(xf[k] - rec['x'][keep]).max(), np.abs(bf[k] - rec['buys'][keep]).max()
P('list check (frozen rule, no exclusions) against record_SEL.npz: %d days %s..%s ; max |x diff| %.3e ; max |buys diff| %.1f'
  % (k.size, rec['dates'][keep][0], rec['dates'][keep][-1], dx, db))
if not (dx == 0 and db == 0):
    P('LIST CHECK FAILED - stop')
    C.finish()
    sys.exit(4)
pack = dict(G=G[r0:].copy(), U=U[r0:].copy(), qa=qa[r0:].copy())
bo, so = buy_ok[r0:].copy(), sell_ok[r0:].copy()
del G, U, qa, PRG, oo, buy_ok, sell_ok, held
np.savez_compressed(CACHE, r0=r0, dates=dates[r0:], ra=ra, rc=rc, setheld=sh, **pack)
P('score cache written: %s (%.1f MB)' % (CACHE, os.path.getsize(CACHE) / 2 ** 20))
del ra, rc, sh
M = Q.market(NPZ, DATA_END, r0)
assert bool((M['buy_ok'] == bo).all() and (M['sell_ok'] == so).all())
del bo, so
M.update(pack)
M['PRG'] = pct_rank_rows(M['G'], M['U'] & np.isfinite(M['G']))
dts = M['dates']
tick('market ready; rows %s..%s' % (dts[0], dts[-1]))
base = lambda t: M['PRG'][t]
allok = True
for nm, d0, d1 in WINDOWS:
    a, b = int(np.searchsorted(dts, d0)), int(np.searchsorted(dts, d1, 'right'))
    assert dts[a] == d0 and dts[b - 1] == d1
    R2 = Q.run_account(M, a, b, K0, base, detail=True, use_x=True)
    B2z = Q.bench_B(M, a, b, K0, zero_term=True, use_x=True)
    r = Q.readings(R2['navz'], B2z['nav'], K0)
    P('WINDOW %-4s n=%d | v2 MAIN conservative: annS %.2f annB %.2f | T1 excess %%/yr %.2f (arith %.2f) | T2 t %.2f | TE %%/d %.4f | ddS %.2f ddB %.2f | NAV S %.4f B %.4f'
      % (nm, r['n'], r['annS'], r['annB'], r['t1'], r['t1_arith'], r['t'], r['te'], r['ddS'], r['ddB'], r['navS'], r['navB']))
    ev = np.load(os.path.join(EV, 'record_c2_full_%s.npz' % nm), allow_pickle=False)
    hm = np.full((b - a, K), -1, np.int32)
    for j, (es, eb, hl) in enumerate(R2['log']):
        hm[j, :hl.size] = hl
    same_d = bool((ev['dates'] == dts[a:b]).all())
    dn, dB = np.abs(ev['navz'] - R2['navz']).max(), np.abs(ev['navBz'] - B2z['nav']).max()
    dh = int((ev['held'] != hm).any(axis=1).sum())
    ok = same_d and dn == 0 and dB == 0 and dh == 0
    allok &= ok
    P('   against evaluator record_c2_full_%s.npz: dates same %s | navz max |diff| %.3e | navBz max |diff| %.3e | days with different held list %d | %s'
      % (nm, same_d, dn, dB, dh, 'IDENTICAL' if ok else 'DIFFERENT'))
    if nm == 'HOLD':
        tgt = (27.17, 2.33, 29.45)
        got = (round(r['t1'], 2), round(r['t'], 2), round(r['ddS'], 2))
        P('   hold-out readings vs evaluator report (+27.17, t 2.33, drawdown 29.45): got %s -> %s' % (got, 'MATCH' if got == tgt else 'MISMATCH'))
        allok &= got == tgt
    tick('window ' + nm)
P('REPRODUCTION ' + ('PASSED' if allok else 'FAILED'))
C.finish()
sys.exit(0 if allok else 6)
