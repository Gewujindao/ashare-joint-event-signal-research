# forward_tracking.py - forward tracking of the two frozen objects on the extended panel.
#   plan     = version 2 (frozen G + K=50, 5 swaps/day, X1/X2), exactly as account_v2.py main_window builds it
#   baseline = (ii) smallest-cap 50, 5 swaps/day, pool leavers sold first (design\return_attribution\attribution.py, line 263 and the fcap block)
# Both accounts start in cash on 2024-07-01 and run to DATA_END in ONE pass each. Rows <= 2026-10-09 must reproduce the saved
# hold-out records (prefix invariance); forward readings are printed only after that check passes.
# usage: forward_tracking.py NPZ DATA_END TAG SERIES_OUT [PREV_SERIES]
#   PREV_SERIES (later months): series file of the previous month; its rows must be reproduced too, and the new segment
#   is the days after its last date.
# Copies of the frozen files sit next to this script (forward\base_rates, gbm_model_fit, gbm_account_backtest; byte-identical to design\);
# g_model.pkl is read in place from design\gbm_model_fit by the frozen gbm_composite.py text.
import sys, os, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import account_core as Q
from account_core import C, np, sqrt, P, K
NPZ, DATA_END, TAG, SERIES_OUT = sys.argv[1:5]
PREV = sys.argv[5] if len(sys.argv) > 5 else None
ROOT = C.ROOT
REC_PLAN = ROOT + r'\holdout_run\gbm_account_backtest\record_c2_full_HOLD.npz'
REC_BASE = ROOT + r'\design\return_attribution\record_c3_main.npz'
W0, HOLD_END = '2024-07-01', '2026-10-09'
K0 = 500000.0
BAD5 = ['002084', '002155', '002445', '002906', '600984']     # extension build: no qfq anchor (old panel has no rows 08-24..10-09)


def tick(msg):
    P('# [%5.1fs] %s' % (time.time() - C._T0, msg))


# ---------------- frozen scores and market (same lines as account_v2.py) ----------------
S = Q.frozen_scores(NPZ, DATA_END, TAG)
T = S['dates'].size
assert S['dates'][-1] == DATA_END
xf, bf, msf = Q.setsim(S['PRG'], S['U'], S['buy_ok'], S['sell_ok'], S['oo'], S['t0'], T - 1)
rec = np.load(os.path.join(HERE, '..', 'gbm_model_fit', 'record_SEL.npz'), allow_pickle=False)
keep = rec['dates'] < S['dates'][-1]
k = np.searchsorted(S['dates'], rec['dates'][keep])
assert k.size > 0 and bool((S['dates'][k] == rec['dates'][keep]).all())
dx, db = np.abs(xf[k] - rec['x'][keep]).max(), np.abs(bf[k] - rec['buys'][keep]).max()
P('list check (frozen rule, no exclusions) against record_SEL.npz, %d days %s..%s ; max |x diff| %.3e ; max |buys diff| %.1f'
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
M['PRG'] = Q.pct_rank_rows(M['G'], M['U'] & np.isfinite(M['G']))
dts = M['dates']
Tn, N = M['G'].shape
U = M['U']
tick('scores and market ready; rows %s..%s' % (dts[0], dts[-1]))

# ---------------- float-cap proxy (verbatim from attribution.py MODE main) ----------------
zz = np.load(NPZ, allow_pickle=False)
nrow = int((zz['dates'] <= DATA_END).sum())
s0 = r0 - 40
tr = zz['tradable'][s0:nrow]
am, tu = zz['amount_cny'][s0:nrow], zz['turnover_pct'][s0:nrow]
zz.close()
with np.errstate(invalid='ignore', divide='ignore'):
    xx = (np.where(tr & (am > 0), am, np.nan).astype(np.float32) / np.where(tr & (tu > 0), tu, np.nan).astype(np.float32) * 100)
del am, tu, tr
v = np.isfinite(xx)
cs = np.cumsum(np.where(v, xx, 0), axis=0, dtype=np.float64)
cn = np.cumsum(v, axis=0, dtype=np.int32)
del xx, v
W = 20
s = cs[W - 1:].copy()
s[1:] -= cs[:-W]
n_ = cn[W - 1:].copy()
n_[1:] -= cn[:-W]
del cs, cn
with np.errstate(invalid='ignore', divide='ignore'):
    fcap = (s / n_)
fcap[n_ < 10] = np.nan
fcap = fcap[40 - (W - 1):].astype(np.float32)
del s, n_
assert fcap.shape == (Tn, N)
negcap = -fcap

# ---------------- one window 2024-07-01 .. DATA_END ----------------
a, b = int(np.searchsorted(dts, W0)), Tn
ih = int(np.searchsorted(dts, HOLD_END))
assert dts[a] == W0 and dts[ih] == HOLD_END and dts[b - 1] == DATA_END
nh = ih + 1 - a
n = b - a
plan_sc = lambda t: M['PRG'][t]
base_sc = lambda t: np.where(U[t], negcap[t], np.nan)
B2z = Q.bench_B(M, a, b, K0, zero_term=True, use_x=True)
B2 = Q.bench_B(M, a, b, K0, use_x=True)
B02z = Q.bench_B(M, a, b, K0, slip=2e-4, zero_term=True, use_x=True)
OBJ = {}
for nm, sc in (('plan', plan_sc), ('baseline', base_sc)):
    OBJ[nm] = dict(R=Q.run_account(M, a, b, K0, sc, detail=True, use_x=True),
                   Rp=Q.run_account(M, a, ih + 1, K0, sc, use_x=True),          # hold-out rows only: counters up to 2026-10-09
                   R02=Q.run_account(M, a, b, K0, sc, slip=2e-4, use_x=True))
    tick('accounts ' + nm)


def lists(R, m):
    hm = np.full((m, K), -1, np.int32)
    bm = np.full((m, K + 5), -1, np.int32)
    sm = np.full((m, 12), -1, np.int32)
    for j, (es, eb, hl) in enumerate(R['log'][:m]):
        hm[j, :hl.size], bm[j, :len(eb)], sm[j, :len(es)] = hl, eb, es
    return hm, bm, sm


def cmp(x, y):
    d = ~((x == y) | (np.isnan(x) & np.isnan(y)))
    return float(np.nanmax(np.abs(x - y))) if d.any() else 0.0, int(d.sum())


# ---------------- prefix invariance ----------------
P('\n==== PREFIX CHECK: rows %s..%s (%d days) of the extended run against the saved hold-out records ====' % (dts[a], dts[ih], nh))
Rp_, Rb_ = OBJ['plan']['R'], OBJ['baseline']['R']
rp = np.load(REC_PLAN, allow_pickle=False)
assert bool((rp['dates'] == dts[a:ih + 1]).all()) and bool((rp['codes'] == M['codes']).all())
hm, bm, sm = lists(Rp_, nh)
dl = [int((x != rp[k_]).any(axis=1).sum()) for x, k_ in ((hm, 'held'), (bm, 'buys'), (sm, 'sells'))]
c_nav, c_navz = cmp(Rp_['nav'][:nh], rp['nav']), cmp(Rp_['navz'][:nh], rp['navz'])
c_navB, c_navBz = cmp(B2['nav'][:nh], rp['navB']), cmp(B2z['nav'][:nh], rp['navBz'])
P('plan vs %s: days whose held / bought / sold list differs %d / %d / %d | nav max|diff| %.3e days %d | navz %.3e days %d | '
  'navB %.3e days %d' % (os.path.basename(REC_PLAN), dl[0], dl[1], dl[2], c_nav[0], c_nav[1], c_navz[0], c_navz[1], c_navB[0], c_navB[1]))
P('  report only (conservative benchmark): navBz max|diff| %.3e, days that differ %d' % c_navBz)
rb = np.load(REC_BASE, allow_pickle=False)
assert bool((rb['HOLD_dates'] == dts[a:ih + 1]).all())
c_b = cmp(Rb_['navz'][:nh], rb['HOLD_nav_ii'])
c_bp = cmp(Rp_['navz'][:nh], rb['HOLD_navz'])
c_bB = cmp(B2z['nav'][:nh], rb['HOLD_navB'])
P('baseline vs %s: navz (HOLD_nav_ii) max|diff| %.3e days %d | plan navz (HOLD_navz) %.3e days %d | report only: navBz (HOLD_navB) %.3e days %d'
  % (os.path.basename(REC_BASE), c_b[0], c_b[1], c_bp[0], c_bp[1], c_bB[0], c_bB[1]))
ok = sum(dl) == 0 and c_nav[1] == 0 and c_navz[1] == 0 and c_navB[1] == 0 and c_b[1] == 0 and c_bp[1] == 0
f0 = nh                                                     # first forward day (window index)
if PREV:
    zp = np.load(PREV, allow_pickle=False)
    m = zp['dates'].size
    assert bool((zp['dates'] == dts[a:a + m]).all())
    hmp, _, _ = lists(Rp_, m)
    hmb, _, _ = lists(Rb_, m)
    cc_ = [cmp(Rp_['navz'][:m], zp['plan_navz']), cmp(Rp_['nav'][:m], zp['plan_nav']), cmp(Rb_['navz'][:m], zp['base_navz']),
           cmp(Rb_['nav'][:m], zp['base_nav']), cmp(B2['nav'][:m], zp['B_nav'])]
    dlp = int((hmp != zp['plan_held']).any(axis=1).sum()) + int((hmb != zp['base_held']).any(axis=1).sum())
    P('previous series %s (%d days to %s): held-list days that differ %d | plan navz/nav, baseline navz/nav, B nav days that differ %s | report only: navBz %d'
      % (PREV, m, zp['dates'][-1], dlp, ' '.join(str(c__[1]) for c__ in cc_), cmp(B2z['nav'][:m], zp['B_navz'])[1]))
    ok = ok and dlp == 0 and all(c__[1] == 0 for c__ in cc_)
    f0 = m
if not ok:
    P('PREFIX CHECK FAILED - forward readings are not produced')
    C.finish()
    sys.exit(5)
P('PREFIX CHECK PASSED')

# ---------------- F6 daily-difference sd from the saved hold-out records ----------------
r_pl = rb['HOLD_navz'] / np.r_[K0, rb['HOLD_navz'][:-1]] - 1
r_ii = rb['HOLD_nav_ii'] / np.r_[K0, rb['HOLD_nav_ii'][:-1]] - 1
dd6 = r_pl - r_ii
sd6 = dd6.std(ddof=1)
P('hold-out records (record_c3_main.npz HOLD_navz, HOLD_nav_ii): plan minus baseline daily mean %.2f bp, sd %.4f%%, t %.2f (%d days)'
  % (dd6.mean() * 1e4, sd6 * 100, dd6.mean() / sd6 * sqrt(dd6.size), dd6.size))

# ---------------- forward segment readings ----------------
nF = n - f0
fw = range(f0, n)
P('\n==== FORWARD SEGMENT %s..%s  n = %d trading days (after %s) ====' % (dts[a + f0], dts[b - 1], nF, dts[a + f0 - 1]))


def seg(navS, navB):
    pS, pB = navS[f0 - 1:], navB[f0 - 1:]
    rS, rB = pS[1:] / pS[:-1] - 1, pB[1:] / pB[:-1] - 1
    d = rS - rB
    cS, cB = pS[-1] / pS[0] - 1, pB[-1] / pB[0] - 1
    return dict(cS=cS * 100, cB=cB * 100, ex=(cS - cB) * 100, dm=d.mean() * 100, dsd=d.std(ddof=1) * 100,
                ddS=Q.maxdd(pS) * 100, ddB=Q.maxdd(pB) * 100, rS=rS)


SEG = {}
for nm in ('plan', 'baseline'):
    R, Rp, R02 = OBJ[nm]['R'], OBJ[nm]['Rp'], OBJ[nm]['R02']
    g = seg(R['navz'], B2z['nav'])
    SEG[nm] = g
    nb_ = sum(len(R['log'][j][1]) for j in fw)
    ns_ = sum(len(R['log'][j][0]) for j in fw)
    assert nb_ == R['c']['nbuy'] - Rp['c']['nbuy'] and ns_ == R['c']['nsell'] - Rp['c']['nsell']
    cnt = np.zeros(6)
    ended = {}
    for j in fw:
        e = a + j
        hl = R['log'][j][2]
        qt, u = M['qa'][e - 1][hl], U[e - 1][hl]
        mk = u & (qt < 5)
        cnt[:5] += np.bincount(qt[mk], minlength=5)[:5]
        cnt[5] += hl.size - mk.sum()
        for i in hl[M['last_tr'][hl] < e]:
            ended.setdefault(int(i), dts[e])
    ev = {kk: R['c'][kk] - Rp['c'][kk] for kk in ('fail_block', 'fail_lot', 'fail_cash', 'pend_days')}
    P('%s MAIN (slip 0.1%%/side, conservative) | strategy %+.2f%% | benchmark B %+.2f%% | cumulative excess %+.2f pts | daily difference mean %.3f%% sd %.3f%% | '
      'maxDD strategy %.2f%% B %.2f%%' % (nm, g['cS'], g['cB'], g['ex'], g['dm'], g['dsd'], g['ddS'], g['ddB']))
    P('%s trades: buys %d sells %d | holdings share by amt60 quintile (lowest..highest | not in pool): %s | lowest-quintile share %.3f'
      % (nm, nb_, ns_, ' '.join('%.3f' % v_ for v_ in cnt / cnt.sum()), cnt[0] / cnt.sum()))
    P('%s series ended while held in the segment: %d%s' % (nm, len(ended), ''.join(
        ' | %s first counted %s last tradable %s' % (M['codes'][i], d_, dts[M['last_tr'][i]]) for i, d_ in ended.items())))
    P('%s events: buy blocked at the open %d | one lot unaffordable %d | no cash for one lot %d | pending-sell days %d | pending at end %d'
      % (nm, ev['fail_block'], ev['fail_lot'], ev['fail_cash'], ev['pend_days'], R['pend_final']))
    g2, gl = seg(R02['navz'], B02z['nav']), seg(R['nav'], B2['nav'])
    P('%s report only, slip 0.02%%/side conservative | strategy %+.2f%% | B %+.2f%% | excess %+.2f pts' % (nm, g2['cS'], g2['cB'], g2['ex']))
    P('%s report only, last-close valuation          | strategy %+.2f%% | B %+.2f%% | excess %+.2f pts' % (nm, gl['cS'], gl['cB'], gl['ex']))
dpb = SEG['plan']['rS'] - SEG['baseline']['rS']
cd = SEG['plan']['cS'] - SEG['baseline']['cS']
se6 = sd6 * 100 * sqrt(nF)
P('F6 plan minus baseline: cumulative difference %+.2f pts (sum of daily differences %+.2f) | SE = %.4f%% x sqrt(%d) = %.2f pts | range %+.2f to %+.2f'
  % (cd, dpb.sum() * 100, sd6 * 100, nF, se6, -1.96 * se6, 1.96 * se6))
se1 = 0.8201 * sqrt(nF)
P('F1 centre %+.2f pts, SE %.2f, range %+.2f to %+.2f' % (12 * nF / 252, se1, 12 * nF / 252 - 1.96 * se1, 12 * nF / 252 + 1.96 * se1))
ix5 = [int(np.nonzero(M['codes'] == c_)[0][0]) for c_ in BAD5]
inB = [c_ for c_, i in zip(BAD5, ix5) if i in set(B2z['idx'].tolist())]
held5 = {nm: sorted({str(M['codes'][i]) for j in range(n) for i in OBJ[nm]['R']['log'][j][2] if i in ix5}) for nm in OBJ}
P('extension codes without qfq anchor %s: held by plan %s | held by baseline %s | in benchmark B pool %s | in the pool on forward decision days: %d stock-days'
  % (','.join(BAD5), held5['plan'] or 'none', held5['baseline'] or 'none', inB or 'none', int(U[a + f0 - 1:b - 1][:, ix5].sum())))
if PREV:                                                    # all forward days joined
    rs_ = lambda x, k0: (x / np.r_[k0, x[:-1]] - 1)[nh:]
    for tg, d_ in (('plan minus B', rs_(Rp_['navz'], K0) - rs_(B2z['nav'], 1.0)), ('plan minus baseline', rs_(Rp_['navz'], K0) - rs_(Rb_['navz'], K0))):
        P('all forward days %s..%s (%d): %s daily mean %.3f%% sd %.3f%% t %.2f' % (dts[a + nh], dts[b - 1], d_.size, tg, d_.mean() * 100, d_.std(ddof=1) * 100,
                                                                              d_.mean() / d_.std(ddof=1) * sqrt(d_.size)))
hmP, _, _ = lists(Rp_, n)
hmB, _, _ = lists(Rb_, n)
np.savez_compressed(SERIES_OUT, dates=dts[a:b], hold_end=HOLD_END, plan_navz=Rp_['navz'], plan_nav=Rp_['nav'], base_navz=Rb_['navz'], base_nav=Rb_['nav'],
                    B_navz=B2z['nav'], B_nav=B2['nav'], plan_held=hmP, base_held=hmB, codes=M['codes'], f6_sd=sd6)
P('series saved: %s (%.2f MB)' % (SERIES_OUT, os.path.getsize(SERIES_OUT) / 2 ** 20))
C.finish()
