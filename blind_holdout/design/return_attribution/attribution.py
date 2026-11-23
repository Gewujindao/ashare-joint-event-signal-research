# attribution.py — descriptive split of the frozen plan's excess into "holding the smallest / least active stocks" and
# within-stratum selection. Nothing frozen is changed: scores come from the cache written by full_panel_scores.py (frozen gbm_composite.py
# text), the account engine is design\gbm_account_backtest\account_core.run_account (unchanged). Controls and naive baselines use the same
# engine; only the decision function handed to the engine differs (account_core.decide is swapped for a policy while they run).
# usage: attribution.py MODE     MODE = main | ctrlA | ctrlAC | inact
import sys, os, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'gbm_account_backtest'))
import account_core as Q
from account_core import C, np, sqrt, P, K
MODE = sys.argv[1]
assert MODE in ('main', 'ctrlA', 'ctrlAC', 'ctrlF', 'inact')
C.OUTDIR = HERE
WSEL = sys.argv[2].split(',') if len(sys.argv) > 2 else None
SUF = MODE + ('_' + ''.join(WSEL) if WSEL else '')
C.start('c3_' + SUF)
ROOT = r'<BLIND_ROOT>'
NPZ = ROOT + r'\data_full\ashare_mainboard_full_20261009.npz'
DATA_END = '2026-10-09'
CACHE = os.path.join(HERE, 'cache_c3_scores.npz')
WINDOWS = [('V', '2020-01-03', '2022-12-30'), ('SEL', '2023-01-03', '2024-06-28'), ('HOLD', '2024-07-01', '2026-10-09')]
K0 = 500000.0
NSEED = 200
EMPTY = Q.EMPTY
ORIG_DECIDE = Q.decide


def tick(msg):
    P('# [%5.1fs] %s' % (time.time() - C._T0, msg))


z = np.load(CACHE, allow_pickle=False)
r0 = int(z['r0'])
M = Q.market(NPZ, DATA_END, r0)
assert bool((M['dates'] == z['dates']).all())
for kk in ('G', 'U', 'qa'):
    M[kk] = z[kk]
ra, rc, setheld = z['ra'], z['rc'], z['setheld']
z.close()
M['PRG'] = Q.pct_rank_rows(M['G'], M['U'] & np.isfinite(M['G']))
dts = M['dates']
Tn, N = M['G'].shape
U = M['U']
SB = U & np.isfinite(M['G']) & ~M['xbuy']                 # pool, scored, not X1/X2 (signal-day row)
labA = (ra // 40).astype(np.int16)                          # amt60 quintile 0..4 (6 = no rank); same as gbm_composite qa
assert bool((labA == M['qa']).all())
labC5 = (rc // 40).astype(np.int16)                         # fcap quintile 0..4
labC6 = np.where(rc < 40, rc // 20, rc // 40 + 1).astype(np.int16)   # 0,1 = two halves of the smallest quintile; 2..5
labAC = np.where((labA < 5) & (labC5 < 5), labA * 5 + labC5, 99).astype(np.int16)
labC40 = np.where(rc < 200, rc // 5, 99).astype(np.int16)             # 40 cap buckets of 2.5%
labA40 = np.where(ra < 200, ra // 5, 99).astype(np.int16)             # 40 amt60 buckets of 2.5%
labF = np.where((ra < 200) & (rc < 200), (ra // 40).astype(np.int16) * 20 + rc // 10, 999).astype(np.int16)   # amt quintile x cap 5% bucket
labAC100 = np.where((ra < 200) & (rc < 200), (ra // 20).astype(np.int16) * 10 + rc // 20, 999).astype(np.int16)   # amt decile x cap decile
with np.errstate(invalid='ignore', divide='ignore'):
    Off = np.where(M['off0'] != 0, M['off0'], np.nan).astype(np.float32)
    oo = np.full(Off.shape, np.nan, np.float32)
    oo[:-1] = Off[1:] / Off[:-1] - 1                        # open(e) -> open(e+1), same float32 arithmetic as gbm_composite.py
    del Off
    Cf = np.where(M['cff0'] != 0, M['cff0'], np.nan)
    cc = np.full(Cf.shape, np.nan, np.float32)
    cc[1:] = Cf[1:] / Cf[:-1] - 1                           # close(e-1) -> close(e)
    del Cf
wins = []
for nm, d0, d1 in WINDOWS:
    a, b = int(np.searchsorted(dts, d0)), int(np.searchsorted(dts, d1, 'right'))
    assert dts[a] == d0 and dts[b - 1] == d1
    wins.append((nm, a, b))
base = lambda t: M['PRG'][t]
tick('ready; rows %s..%s' % (dts[0], dts[-1]))


def with_policy(pol, a, b, **kw):
    Q.decide = pol
    try:
        return Q.run_account(M, a, b, K0, sc_fn=lambda t: t, use_x=True, **kw)
    finally:
        Q.decide = ORIG_DECIDE


def decomp(lists, a, b, lab, nlab):
    """gbm_model_fit method (gbm_composite.py lines 325-344): lists[j] = holdings after the trade at the open of row a+j; return oo[e];
    pool and stratum labels of the signal day e-1; weights 1/K; style = stratum mean - pool mean; within = own - stratum mean."""
    n = b - a
    x, sty, sel, csh = (np.full(n, np.nan) for _ in range(4))
    sh = np.zeros(nlab + 1)
    for j in range(n):
        e = a + j
        t = e - 1
        r = oo[e]
        fin = np.isfinite(r)
        uu = U[t] & fin
        if not uu.any():
            continue
        mm = r[uu].mean(dtype=np.float64)
        hl = lists[j]
        hl = hl[hl >= 0]
        hv = hl[fin[hl]]
        lt = lab[t]
        bench = np.full(N, mm)
        for q in range(nlab):
            zq = uu & (lt == q)
            if zq.any():
                bench[zq] = r[zq].mean(dtype=np.float64)
        bsum = r[hv].sum(dtype=np.float64) / K
        x[j] = bsum - mm
        sty[j] = (bench[hv] - mm).sum() / K
        sel[j] = (r[hv].astype(np.float64) - bench[hv]).sum() / K
        csh[j] = -(1 - hv.size / K) * mm
        lq = lt[hl]
        inq = U[t][hl] & (lq < nlab)
        sh[:nlab] += np.bincount(lq[inq], minlength=nlab)[:nlab]
        sh[nlab] += hl.size - inq.sum()
    return dict(x=x, sty=sty, sel=sel, csh=csh, share=sh / sh.sum())


def dline(tag, D, v=None):
    v = np.isfinite(D['x']) if v is None else (v & np.isfinite(D['x']))
    n = int(v.sum())
    g, s, w, c = D['x'][v].mean(), D['sty'][v].mean(), D['sel'][v].mean(), D['csh'][v].mean()
    ts = s / D['sty'][v].std(ddof=1) * sqrt(n)
    tw = w / D['sel'][v].std(ddof=1) * sqrt(n)
    tg = g / D['x'][v].std(ddof=1) * sqrt(n)
    P('%-44s n=%3d | gross bp/d %6.2f (t %5.2f) | style %6.2f (%4.0f%%) sd %.3f%% t %5.2f | within %6.2f (%4.0f%%) sd %.3f%% t %5.2f | cash %5.2f'
      % (tag, n, g * 1e4, tg, s * 1e4, s / g * 100, D['sty'][v].std(ddof=1) * 100, ts, w * 1e4, w / g * 100, D['sel'][v].std(ddof=1) * 100, tw, c * 1e4))


def lists_of(R):
    return [hl for (_, _, hl) in R['log']]


def qkeys(a, b):
    return np.array([s[:4] + 'Q' + str((int(s[5:7]) - 1) // 3 + 1) for s in dts[a:b]])


def pret(nav, ix, K_=K0):
    sN = np.r_[K_, nav]
    return sN[ix[-1] + 1] / sN[ix[0]] - 1


# ---------------- policies handed to the unchanged engine ----------------
def pol_matched(rng, a, log, lab, nlab, lab_fb=None):
    """Position-by-position stand-in: when the plan buys stock p, the control buys a random stock of the same stratum
    (signal-day label; pool, scored, not X1/X2, buyable at the open, not held); when the plan sells p the control sells
    the stand-in. A stand-in whose buy did not execute (one lot unaffordable) is redrawn on the next day."""
    mp = {}

    def pol(t, u, held, bok, sok, nslots, nq, xb):
        ex_s, ex_b, hl = log[t + 1 - a]
        sells = []
        for p in ex_s:
            c = mp.pop(p, None)
            if c is not None and held[c]:
                sells.append(c)
        bs = SB[t] & bok & ~held
        lt = lab[t]
        buys = []
        for p in hl.tolist():
            c = mp.get(p)
            if c is not None and held[c]:
                continue
            q = lt[p]
            cand = np.nonzero(bs & (lt == q))[0] if q < nlab else EMPTY
            if cand.size == 0 and lab_fb is not None:
                cand = np.nonzero(bs & (lab_fb[t] == lab_fb[t, p]))[0]
            if cand.size == 0:
                cand = np.nonzero(bs)[0]
            c = int(cand[rng.integers(cand.size)])
            bs[c] = False
            mp[p] = c
            buys.append(c)
        return np.array(buys, np.int64), np.array(sells, np.int64), EMPTY
    return pol


def pol_inactive(rng):
    """Naive baseline (i): 50 names drawn uniformly from the lowest amt60 quintile, 5 random swaps a day."""
    def pol(t, u, held, bok, sok, nslots, nq, xb):
        cand = np.nonzero(SB[t] & bok & ~held & (labA[t] == 0))[0]
        fill = max(nslots - int(held.sum()), 0)
        hs = np.nonzero(held & sok)[0]
        m = max(min(nq, hs.size, cand.size - fill), 0)
        pick = rng.choice(cand, size=min(cand.size, fill + m), replace=False)
        return pick[:fill].astype(np.int64), rng.choice(hs, size=m, replace=False).astype(np.int64), pick[fill:fill + m].astype(np.int64)
    return pol


def pol_basket(a, negcap, nb=400, every=20):
    """Naive baseline (iii): the nb smallest by the float-cap proxy (pool, not X1/X2); membership reset every 20 days
    (leavers sold, entrants bought at NAV/nb); members not yet held are retried every day."""
    st = {'D': np.zeros(N, bool)}

    def pol(t, u, held, bok, sok, nslots, nq, xb):
        sells = EMPTY
        if (t + 1 - a) % every == 0:
            c = negcap[t]
            idx = np.nonzero(u & ~xb & np.isfinite(c))[0]
            D = np.zeros(N, bool)
            D[idx[np.argsort(-c[idx], kind='stable')[:nb]]] = True
            st['D'] = D
            sells = np.nonzero(held & ~D)[0]
        return np.nonzero(st['D'] & ~held & bok)[0], sells, EMPTY
    return pol


def acct_line(tag, R, B, extra=''):
    r = Q.readings(R['navz'], B['nav'], K0)
    c = R['c']
    yrs = r['n'] / 252.0
    P('%-40s ann %7.2f | maxDD %6.2f | vs B: excess %%/yr %7.2f (arith %7.2f) t %5.2f | NAV %.4f | mean held %6.1f cash %.3f buys %5d cost %%/yr %5.2f%s'
      % (tag, r['annS'], r['ddS'], r['t1'], r['t1_arith'], r['t'], r['navS'], R['nheld'].mean(), R['cashr'].mean(), c['nbuy'],
         c['cost'] / R['navz'].mean() / yrs * 100, extra))
    return r


if MODE == 'main':
    # ---- float-cap proxy, raw values (same formula as gbm_composite.py 'fcap': 20-day mean of amount / turnover% * 100, >= 10 valid) ----
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
    bad = tot = 0
    for t in range(wins[0][1] - 1, Tn):
        idx = np.nonzero(rc[t] != 255)[0]
        o = np.argsort(fcap[t, idx], kind='stable')
        rr = np.empty(idx.size, np.int64)
        rr[o] = np.arange(idx.size)
        bad += int((rr * 200 // idx.size != rc[t, idx]).sum())
        tot += idx.size
    P('float-cap proxy recomputed for ordering: 200-bucket rank differs from the frozen feature rank on %d of %d stock-days (%.4f%%)' % (bad, tot, bad / tot * 100))
    negcap = -fcap
    tick('fcap')

    RES = {}
    for nm, a, b in wins:
        n = b - a
        P('\n==================== WINDOW %s  %s..%s  %d days ====================' % (nm, dts[a], dts[b - 1], n))
        B = Q.bench_B(M, a, b, K0, zero_term=True, use_x=True)
        R = Q.run_account(M, a, b, K0, base, detail=True, use_x=True)
        P('-- accounts (same engine, 500k, slip 0.1%/side, conservative valuation; B = hold-all benchmark of the evaluation) --')
        rS = acct_line('PLAN v2 (frozen)', R, B)
        P('%-40s ann %7.2f | maxDD %6.2f' % ('benchmark B', rS['annB'], rS['ddB']))
        R_ii = Q.run_account(M, a, b, K0, lambda t: np.where(U[t], negcap[t], np.nan), detail=True, use_x=True)
        r_ii = acct_line('(ii) smallest-cap 50, 5 swaps/day', R_ii, B)
        R_iix = Q.run_account(M, a, b, K0, lambda t: negcap[t], detail=True, use_x=True)
        r_iix = acct_line('(ii-x) same, pool leavers kept', R_iix, B)
        R3 = {}
        for tag, kw in (('(iii-a) smallest 400, lots + min comm', dict()), ('(iii-b) smallest 400, lots, no min comm', dict(mincomm=0.0)),
                        ('(iii-c) smallest 400, no lots, no min c', dict(mincomm=0.0, lots=False))):
            R3[tag[:7]] = with_policy(pol_basket(a, negcap), a, b, nslots=400, detail=True, **kw)
            c3 = R3[tag[:7]]['c']
            R3[tag[:7] + 'r'] = acct_line(tag, R3[tag[:7]], B, ' | lot-unaffordable attempts %d' % c3['fail_lot'])
        d = lambda r_: r_['rS']
        P('daily return correlation with the plan: (ii) %.3f | (iii-a) %.3f | (iii-c) %.3f | B %.3f' % (
            np.corrcoef(d(rS), d(r_ii))[0, 1], np.corrcoef(d(rS), d(R3['(iii-a)r']))[0, 1], np.corrcoef(d(rS), d(R3['(iii-c)r']))[0, 1], np.corrcoef(rS['rS'], rS['rB'])[0, 1]))
        dd = rS['rS'] - r_ii['rS']
        P('plan minus (ii): daily mean %.2f bp, t %.2f | plan minus (iii-c): daily mean %.2f bp, t %.2f' % (
            dd.mean() * 1e4, dd.mean() / dd.std(ddof=1) * sqrt(n), (rS['rS'] - R3['(iii-c)r']['rS']).mean() * 1e4,
            (rS['rS'] - R3['(iii-c)r']['rS']).mean() / (rS['rS'] - R3['(iii-c)r']['rS']).std(ddof=1) * sqrt(n)))

        P('-- decomposition, stage C1 method (weights 1/50, open-to-open, pool = daily equal-weighted pool) --')
        LS = lists_of(R)
        Lset = [setheld[a + j] for j in range(n)]
        Ds = decomp(Lset, a, b, labA, 5)
        dline('set simulation v1 rule | amt60 quintiles', Ds)
        P('   set simulation: mean number of the 50 slots whose price series has already ended (stuck, return 0): %.2f (last day %d)' % (
            np.mean([(M['last_tr'][l_[l_ >= 0]] < a + j_).sum() for j_, l_ in enumerate(Lset)]), (M['last_tr'][Lset[-1][Lset[-1] >= 0]] < b - 1).sum()))
        if nm != 'HOLD':
            rec = np.load(os.path.join(HERE, '..', 'gbm_model_fit', 'record_SEL.npz'), allow_pickle=False)
            kk = np.searchsorted(dts[a:b], rec['dates'])
            okk = (kk < n)
            okk[okk] &= dts[a:b][kk[okk]] == rec['dates'][okk]
            P('   check against stage C1 record_SEL.npz on %d common days: max |style diff| %.2e, max |within diff| %.2e' % (
                okk.sum(), np.nanmax(np.abs(Ds['sty'][kk[okk]] - rec['sty'][okk])), np.nanmax(np.abs(Ds['sel'][kk[okk]] - rec['sel'][okk]))))
        DA = decomp(LS, a, b, labA, 5)
        DC = decomp(LS, a, b, labC6, 6)
        DC5 = decomp(LS, a, b, labC5, 5)
        DAC = decomp(LS, a, b, labAC, 25)
        dline('PLAN v2 account lists | (a) amt60 quintiles', DA)
        dline('PLAN v2 account lists | (b) cap 6 strata', DC)
        dline('PLAN v2 account lists | cap quintiles', DC5)
        dline('PLAN v2 account lists | amt x cap 25 cells', DAC)
        DC40, DA40, DF, D100 = decomp(LS, a, b, labC40, 40), decomp(LS, a, b, labA40, 40), decomp(LS, a, b, labF, 100), decomp(LS, a, b, labAC100, 100)
        dline('PLAN v2 | cap 40 buckets (2.5%)', DC40)
        dline('PLAN v2 | amt60 40 buckets (2.5%)', DA40)
        dline('PLAN v2 | amt quintile x cap 5% bucket', DF)
        dline('PLAN v2 | amt decile x cap decile', D100)
        P('   plan holdings share in the four smallest 2.5%% cap buckets: %s | in the four lowest 2.5%% amt60 buckets: %s' % (
            ' '.join('%.3f' % v_ for v_ in DC40['share'][:4]), ' '.join('%.3f' % v_ for v_ in DA40['share'][:4])))
        P('   plan holdings share by amt60 quintile (low..high | none): %s' % ' '.join('%.3f' % v_ for v_ in DA['share']))
        P('   plan holdings share by cap strata (smallest 10%% | next 10%% | q2 | q3 | q4 | q5 | none): %s' % ' '.join('%.3f' % v_ for v_ in DC['share']))
        L2 = lists_of(R_ii)
        D2A, D2C = decomp(L2, a, b, labA, 5), decomp(L2, a, b, labC6, 6)
        dline('(ii) smallest-cap 50 | amt60 quintiles', D2A)
        dline('(ii) smallest-cap 50 | cap 6 strata', D2C)
        P('   (ii) holdings share by amt60 quintile: %s | by cap strata: %s' % (' '.join('%.3f' % v_ for v_ in D2A['share']), ' '.join('%.3f' % v_ for v_ in D2C['share'])))
        yq = qkeys(a, b)
        if nm == 'HOLD':
            P('-- hold-out by quarter: n | gross bp/d | (a) amt60: style, within | (b) cap 6 strata: style, within | amt x cap: style, within --')
            for u_ in sorted(set(yq)):
                v_ = (yq == u_) & np.isfinite(DA['x'])
                if v_.sum() == 0:
                    continue
                P('  %s | %2d | %7.2f | %7.2f %7.2f | %7.2f %7.2f | %7.2f %7.2f' % (
                    u_, v_.sum(), DA['x'][v_].mean() * 1e4, DA['sty'][v_].mean() * 1e4, DA['sel'][v_].mean() * 1e4, DC['sty'][v_].mean() * 1e4,
                    DC['sel'][v_].mean() * 1e4, DAC['sty'][v_].mean() * 1e4, DAC['sel'][v_].mean() * 1e4))
            ex = (yq != '2026Q2')
            dline('PLAN (a) amt60, hold-out without 2026Q2', DA, ex)
            dline('PLAN (b) cap 6 strata, without 2026Q2', DC, ex)

            # ---------------- 2026Q2 ----------------
            P('\n-- 2026Q2 by month (account NAV, close to close, conservative valuation) --')
            mon = np.array([s_[:7] for s_ in dts[a:b]])
            rows = [('PLAN v2', R['navz'], K0), ('benchmark B', B['nav'], 1.0), ('(ii) smallest-cap 50', R_ii['navz'], K0),
                    ('(iii-a)', R3['(iii-a)']['navz'], K0), ('(iii-b)', R3['(iii-b)']['navz'], K0), ('(iii-c)', R3['(iii-c)']['navz'], K0)]
            ms = ['2026-04', '2026-05', '2026-06']
            ixq = np.nonzero(yq == '2026Q2')[0]
            for tag, nav, k_ in rows:
                P('  %-22s %s | quarter %7.2f%%' % (tag, ' '.join('%s %7.2f%%' % (m_, pret(nav, np.nonzero(mon == m_)[0], k_) * 100) for m_ in ms), pret(nav, ixq, k_) * 100))
            P('  cash share of NAV, mean over 2026Q2: plan %.3f | (ii) %.3f | (ii-x) %.3f | (iii-a) %.3f | (iii-c) %.3f ; (ii-x) quarter %.2f%%' % (
                R['cashr'][ixq].mean(), R_ii['cashr'][ixq].mean(), R_iix['cashr'][ixq].mean(), R3['(iii-a)']['cashr'][ixq].mean(), R3['(iii-c)']['cashr'][ixq].mean(), pret(R_iix['navz'], ixq) * 100))
            P('  by quarter, account NAV %: quarter | plan | (ii) | (iii-c) | B')
            for u_ in sorted(set(yq)):
                ix_ = np.nonzero(yq == u_)[0]
                P('    %s %7.2f %7.2f %7.2f %7.2f' % (u_, pret(R['navz'], ix_) * 100, pret(R_ii['navz'], ix_) * 100, pret(R3['(iii-c)']['navz'], ix_) * 100, pret(B['nav'], ix_, 1.0) * 100))
            for tag, D_ in (('PLAN | cap 40 buckets', DC40), ('PLAN | amt q x cap 5%', DF)):
                v_ = (yq == '2026Q2') & np.isfinite(D_['x'])
                P('  2026Q2 sums, pts: %-24s gross %7.2f | style %7.2f | within %7.2f' % (tag, D_['x'][v_].sum() * 100, D_['sty'][v_].sum() * 100, D_['sel'][v_].sum() * 100))
            P('-- 2026Q2 by month: equal-weighted close-to-close return of the pool by stratum (labels of the previous close) --')

            def ew_month(lab, q):
                out = []
                for m_ in ms + ['Q']:
                    ix = ixq if m_ == 'Q' else np.nonzero(mon == m_)[0]
                    g_ = 1.0
                    for j in ix:
                        e = a + j
                        r = cc[e]
                        zq = U[e - 1] & np.isfinite(r)
                        if q is not None:
                            zq &= lab[e - 1] == q
                        g_ *= 1 + r[zq].mean(dtype=np.float64)
                    out.append((g_ - 1) * 100)
                return out
            P('  %-26s %s' % ('whole pool', ' '.join('%7.2f%%' % v_ for v_ in ew_month(labA, None))))
            for q in range(5):
                P('  amt60 quintile %d %-9s %s' % (q, '(lowest)' if q == 0 else '', ' '.join('%7.2f%%' % v_ for v_ in ew_month(labA, q))))
            for q, nmq in enumerate(('cap smallest 10%', 'cap next 10%', 'cap quintile 2', 'cap quintile 3', 'cap quintile 4', 'cap quintile 5')):
                P('  %-26s %s' % (nmq, ' '.join('%7.2f%%' % v_ for v_ in ew_month(labC6, q))))
            P('  %-26s %s' % ('lowest amt x smallest cap', ' '.join('%7.2f%%' % v_ for v_ in ew_month(labAC, 0))))
            DAq, DCq = decomp(LS[ixq[0]:ixq[-1] + 1], a + ixq[0], a + ixq[-1] + 1, labA, 5), decomp(LS[ixq[0]:ixq[-1] + 1], a + ixq[0], a + ixq[-1] + 1, labC6, 6)
            D2q = decomp(L2[ixq[0]:ixq[-1] + 1], a + ixq[0], a + ixq[-1] + 1, labC6, 6)
            D2qa = decomp(L2[ixq[0]:ixq[-1] + 1], a + ixq[0], a + ixq[-1] + 1, labA, 5)
            P('-- 2026Q2 holdings share: plan by amt60 quintile %s | plan by cap strata %s' % (' '.join('%.3f' % v_ for v_ in DAq['share']), ' '.join('%.3f' % v_ for v_ in DCq['share'])))
            P('                          (ii) by amt60 quintile %s | (ii) by cap strata %s' % (' '.join('%.3f' % v_ for v_ in D2qa['share']), ' '.join('%.3f' % v_ for v_ in D2q['share'])))
            P('-- 2026Q2 sums of daily terms, percentage points (weights 1/50, open-to-open): gross vs pool = style + within (+cash) --')
            for tag, D_ in (('PLAN | amt60 quintiles', DA), ('PLAN | cap 6 strata', DC), ('PLAN | amt x cap 25 cells', DAC), ('(ii)  | amt60 quintiles', D2A), ('(ii)  | cap 6 strata', D2C)):
                for m_ in ms + ['Q']:
                    v_ = ((yq == '2026Q2') if m_ == 'Q' else (mon == m_)) & np.isfinite(D_['x'])
                    P('  %-26s %-7s gross %7.2f | style %7.2f | within %7.2f | cash %5.2f' % (tag, m_, D_['x'][v_].sum() * 100, D_['sty'][v_].sum() * 100, D_['sel'][v_].sum() * 100, D_['csh'][v_].sum() * 100))
            v_ = (yq == '2026Q2') & np.isfinite(DC['x'])
            P('  plan minus (ii), 2026Q2, cap 6 strata: gross %.2f = style %.2f + within %.2f + cash %.2f' % (
                (DC['x'][v_] - D2C['x'][v_]).sum() * 100, (DC['sty'][v_] - D2C['sty'][v_]).sum() * 100, (DC['sel'][v_] - D2C['sel'][v_]).sum() * 100, (DC['csh'][v_] - D2C['csh'][v_]).sum() * 100))
            P('  plan minus (ii), 2026Q2, amt60 quintiles: gross %.2f = style %.2f + within %.2f + cash %.2f' % (
                (DA['x'][v_] - D2A['x'][v_]).sum() * 100, (DA['sty'][v_] - D2A['sty'][v_]).sum() * 100, (DA['sel'][v_] - D2A['sel'][v_]).sum() * 100, (DA['csh'][v_] - D2A['csh'][v_]).sum() * 100))
        RES[nm] = dict(navz=R['navz'], navB=B['nav'], nav_ii=R_ii['navz'], nav_iiia=R3['(iii-a)']['navz'], nav_iiib=R3['(iii-b)']['navz'], nav_iiic=R3['(iii-c)']['navz'],
                       x=DA['x'], styA=DA['sty'], selA=DA['sel'], styC=DC['sty'], selC=DC['sel'], styAC=DAC['sty'], selAC=DAC['sel'])
        tick('window ' + nm)
    np.savez_compressed(os.path.join(HERE, 'record_c3_main.npz'), **{nm + '_' + k_: v_ for nm in RES for k_, v_ in RES[nm].items()},
                        **{nm + '_dates': dts[a:b] for nm, a, b in wins})
else:
    lab, nlab, fb = {'ctrlA': (labA, 5, None), 'ctrlAC': (labAC, 25, labA), 'ctrlF': (labF, 100, labA), 'inact': (None, 0, None)}[MODE]
    SAVE = {}
    for nm, a, b in wins[::-1]:
        if WSEL and nm not in WSEL:
            continue
        n = b - a
        B = Q.bench_B(M, a, b, K0, zero_term=True, use_x=True)
        R = Q.run_account(M, a, b, K0, base, detail=True, use_x=True)
        rS = Q.readings(R['navz'], B['nav'], K0)
        log = R['log']
        fin, ann, ddl, exc, tt, hold, qd = [], [], [], [], [], [], np.zeros(6)
        paths = np.empty((NSEED, n), np.float32)
        done = 0
        for seed in range(NSEED):
            if time.time() - C._T0 > 245:
                break
            rng = np.random.default_rng(3000 + seed)
            pol = pol_inactive(rng) if MODE == 'inact' else pol_matched(rng, a, log, lab, nlab, fb)
            Rr = with_policy(pol, a, b, detail=seed < 20)
            rr = Q.readings(Rr['navz'], B['nav'], K0)
            fin.append(rr['navS']); ann.append(rr['annS']); ddl.append(rr['ddS']); exc.append(rr['t1']); tt.append(rr['t'])
            hold.append((Rr['nheld'].mean(), Rr['c']['nbuy'], Rr['n_term'], Rr['c']['fail_lot'], Rr['c']['cost'] / Rr['navz'].mean() / (n / 252.0) * 100))
            if seed < 20:
                qd += Rr['qd']
            paths[seed] = Rr['navz'] / K0
            done += 1
        if done < 20:
            P('\n==== WINDOW %s | %s | only %d seeds done before the time limit - skipped ====' % (nm, MODE, done))
            continue
        fin, ann, ddl, exc, tt, hold = (np.array(v_) for v_ in (fin, ann, ddl, exc, tt, hold))
        paths = paths[:done]
        s = rS['navS']
        P('\n==== WINDOW %s  %s..%s  %d days | %s | seeds done %d ====' % (nm, dts[a], dts[b - 1], n, MODE, done))
        P('  plan: final NAV %.4f, ann %.2f, maxDD %.2f, excess over B %.2f %%/yr, t %.2f | B: final NAV %.4f, ann %.2f' % (s, rS['annS'], rS['ddS'], rS['t1'], rS['t'], rS['navB'], rS['annB']))
        P('  control final NAV: mean %.4f sd %.4f min %.4f median %.4f q97.5 %.4f max %.4f | plan above %d of %d | z %.2f | plan > q97.5: %s' % (
            fin.mean(), fin.std(ddof=1), fin.min(), np.median(fin), np.percentile(fin, 97.5), fin.max(), int((s > fin).sum()), done,
            (s - fin.mean()) / fin.std(ddof=1), 'YES' if s > np.percentile(fin, 97.5) else 'NO'))
        P('  control annualised %%: mean %.2f sd %.2f q2.5 %.2f q97.5 %.2f | maxDD %%: mean %.2f (min %.2f max %.2f) | excess over B %%/yr: mean %.2f | t vs B: mean %.2f, share >= 2: %.3f' % (
            ann.mean(), ann.std(ddof=1), np.percentile(ann, 2.5), np.percentile(ann, 97.5), ddl.mean(), ddl.min(), ddl.max(), exc.mean(), tt.mean(), (tt >= 2).mean()))
        P('  rhythm: plan mean holdings %.2f buys %d cost %%/yr %.2f | control mean holdings %.2f buys %.0f ended/run %.2f lot-unaffordable/run %.1f cost %%/yr %.2f' % (
            R['nheld'].mean(), R['c']['nbuy'], R['c']['cost'] / R['navz'].mean() / (n / 252.0) * 100, hold[:, 0].mean(), hold[:, 1].mean(), hold[:, 2].mean(), hold[:, 3].mean(), hold[:, 4].mean()))
        P('  holdings by amt60 quintile (low..high | none): plan %s | control (first 20 seeds) %s' % (
            ' '.join('%.3f' % v_ for v_ in R['qd'] / R['qd'].sum()), ' '.join('%.3f' % v_ for v_ in qd / qd.sum())))
        mpath = paths.mean(axis=0, dtype=np.float64)
        dmean = mpath / np.r_[1.0, mpath[:-1]] - 1
        dd_ = rS['rS'] - dmean
        P('  plan minus mean control path: daily mean %.2f bp, sd %.3f%%, t %.2f (the mean path has almost no selection noise of its own)' % (
            dd_.mean() * 1e4, dd_.std(ddof=1) * 100, dd_.mean() / dd_.std(ddof=1) * sqrt(n)))
        yq = qkeys(a, b)
        if nm == 'HOLD':
            P('  by quarter: plan %% | control mean %% | difference pts | plan above k of %d' % done)
            for u_ in sorted(set(yq)):
                ix = np.nonzero(yq == u_)[0]
                pS = pret(R['navz'], ix) * 100
                pc_ = np.array([pret(paths[i_].astype(np.float64), ix, 1.0) * 100 for i_ in range(done)])
                P('    %s %2d d | %7.2f | %7.2f | %7.2f | %d' % (u_, ix.size, pS, pc_.mean(), pS - pc_.mean(), int((pS > pc_).sum())))
            mon = np.array([s_[:7] for s_ in dts[a:b]])
            for m_ in ('2026-04', '2026-05', '2026-06'):
                ix = np.nonzero(mon == m_)[0]
                pS = pret(R['navz'], ix) * 100
                pc_ = np.array([pret(paths[i_].astype(np.float64), ix, 1.0) * 100 for i_ in range(done)])
                P('    %s %2d d | %7.2f | %7.2f | %7.2f | %d' % (m_, ix.size, pS, pc_.mean(), pS - pc_.mean(), int((pS > pc_).sum())))
        SAVE[nm + '_fin'] = fin
        SAVE[nm + '_ann'] = ann
        SAVE[nm + '_dd'] = ddl
        if nm == 'HOLD':
            SAVE['HOLD_paths'] = paths
        tick('window ' + nm)
    np.savez_compressed(os.path.join(HERE, 'record_c3_%s.npz' % SUF), **SAVE)
C.finish()
