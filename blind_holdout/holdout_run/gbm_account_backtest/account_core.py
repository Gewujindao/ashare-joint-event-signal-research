# account_core.py — trade-by-trade portfolio account for the frozen composite G (K=50, 5 swaps/day).
# Frozen parts are NOT re-implemented: the feature/pool/score code is the source text of design\gbm_model_fit\gbm_composite.py,
# executed verbatim up to (not including) the line "PRG = pct_rank_rows(G, ...)". The decision rule decide() is the
# quota logic of gbm_composite.py lines 303-324 with the held set supplied by the caller (the real account holdings).
import os, sys
from math import floor
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'base_rates'))
import common as C
from common import np, sqrt, P, ffill, pct_rank_rows
C.OUTDIR = HERE

K, H, NQ = 50, 10, 5
COMM, TRANSFER = 1e-4, 1e-5
STAMP_OLD, STAMP_NEW, STAMP_SWITCH = 1e-3, 5e-4, '2023-08-28'
C1 = os.path.normpath(os.path.join(HERE, '..', 'gbm_model_fit', 'gbm_composite.py'))
EMPTY = np.zeros(0, np.int64)


def frozen_scores(npz, data_end, tag):
    """Runs the frozen gbm_composite.py source up to the raw model scores G. Three textual substitutions only
    (output directory, output file prefix, the DATA_END assertion)."""
    src = open(C1, 'rb').read()
    txt = src.decode('utf-8-sig').replace('\r\n', '\n')
    txt = txt[:txt.index('PRG = pct_rank_rows(G, U & np.isfinite(G))')]
    for old, new in ((" and DATA_END <= '2024-06-28'", ''), ('C.OUTDIR = OUT', 'C.OUTDIR = HERE2'),
                     ("start('c1_' + TAG)", "start('c2_' + TAG)")):
        assert txt.count(old) == 1, old
        txt = txt.replace(old, new)
    C.NPZ = npz
    argv = sys.argv
    sys.argv = [C1, data_end, 'extend', tag, 'none', 'none']
    ns = {'__name__': 'c1_prefix', 'HERE2': HERE}
    exec(compile(txt, C1, 'exec'), ns)
    sys.argv = argv
    G, U = ns['G'], ns['U']
    qa = ns['RK'][ns['NAMES'].index('amt60')] // 40          # same line as gbm_composite.py
    del ns['RK']
    PRG = pct_rank_rows(G, U & np.isfinite(G))               # same line as gbm_composite.py
    Off = ffill(ns['OQ'])
    oo = np.full(G.shape, np.nan, np.float32)
    oo[:-1] = Off[1:] / Off[:-1] - 1                         # same lines as gbm_composite.py
    out = dict(dates=ns['dates'], G=G, U=U, qa=qa, PRG=PRG, oo=oo, buy_ok=ns['buy_ok'], sell_ok=ns['sell_ok'],
               t0=ns['t0'], iF=ns['iF'])
    ns.clear()
    return out


def decide(sc, u, held, bok, sok, nslots=K, nq=NQ, xb=None):
    """Decision at the close of t for the open of e = t+1 (gbm_composite.py lines 307-324, held = caller's holdings).
    Returns (fills, sells, swap-ins); sells[j] is paired with swap-ins[j].
    xb (version 2 only): stocks that must not be bought (X1 or X2); it removes candidates and nothing else."""
    ok = u & ~held & bok & np.isfinite(sc)
    if xb is not None:
        ok &= ~xb
    ci = np.nonzero(ok)[0]
    ci = ci[np.argsort(-sc[ci], kind='stable')]
    hi_ = np.nonzero(held & sok)[0]
    hs = np.where(np.isfinite(sc[hi_]), sc[hi_], -np.inf)
    ho = np.argsort(hs, kind='stable')
    hi_, hs = hi_[ho], hs[ho]
    fill = min(nslots - int(held.sum()), ci.size)
    fill = max(fill, 0)
    cand = ci[fill:fill + nq]
    m = min(cand.size, hi_.size)
    if m > 0:
        m = int((sc[cand[:m]] > hs[:m]).sum())
    return ci[:fill], hi_[:m], cand[:m]


def setsim(PRG, U, buy_ok, sell_ok, oo, ta, tb):
    """Set simulation of gbm_composite.py (every wanted trade is assumed filled), signal days ta..tb-1, starting empty."""
    T, N = PRG.shape
    held = np.zeros(N, bool)
    x = np.full(T, np.nan)
    buys = np.zeros(T)
    ms = np.full(T, np.nan)
    for t in range(ta, tb):
        e = t + 1
        u = U[t]
        f_, s_, c_ = decide(PRG[t], u, held, buy_ok[e], sell_ok[e])
        held[f_] = True
        held[s_] = False
        held[c_] = True
        r = oo[e]
        fin = np.isfinite(r)
        hv = held & fin
        b = r[hv].sum(dtype=np.float64) / K
        uu = u & fin
        if not uu.any():
            continue
        mm = r[uu].mean(dtype=np.float64)
        x[e], buys[e], ms[e] = b - mm, f_.size + c_.size, mm
    return x, buys, ms


def market(npz, data_end, r0):
    """Execution arrays for rows r0.. (row index j of every returned array = data row r0 + j)."""
    z = np.load(npz, allow_pickle=False)
    dts = z['dates']
    n = int((dts <= data_end).sum())
    g = lambda k: np.ascontiguousarray(z[k][:n])
    tr, st = g('tradable'), g('st')
    T, N = tr.shape
    OQ = np.where(tr, g('open_qfq'), np.nan).astype(np.float32)
    o, pc = g('open_raw'), g('preclose_raw')
    sell_ok, buy_ok = C.exec_maps(dict(tradable=tr, st=st, open_raw=o, preclose_raw=pc, open_qfq=OQ))
    M = {'sell_ok': sell_ok[r0:].copy(), 'buy_ok': buy_ok[r0:].copy()}
    del sell_ok, buy_ok
    M['o_raw'] = np.nan_to_num(o[r0:].astype(np.float64))
    M['pc'] = pc[r0:].copy()
    del o, pc
    M['o_q'] = np.nan_to_num(OQ[r0:].astype(np.float64))
    Off = ffill(OQ)
    M['off0'] = np.nan_to_num(Off[r0:].astype(np.float64))
    del Off, OQ
    cq = g('close_qfq')
    M['cq'] = cq[r0:].copy()
    Cff = ffill(np.where(tr, cq, np.nan).astype(np.float32))
    M['cff0'] = np.nan_to_num(Cff[r0:].astype(np.float64))
    del Cff, cq
    am = g('amount_cny')
    cs = np.cumsum(np.where(tr & (am > 0), am, 0).astype(np.float64), axis=0)
    del am
    M['adv'] = (cs[r0 - 1:T - 1] - cs[r0 - 21:T - 21]) / 20.0     # mean amount of the 20 market days before the row
    del cs
    # version 2 exclusions: X1 = st on any of rows t-60..t ; X2 = close_raw < 1 on any of rows t-20..t
    cr = g('close_raw')
    M['cr'] = cr[r0:].copy()

    def anyroll(flag, W):
        c = np.cumsum(flag, axis=0, dtype=np.int32)
        out = c.copy()
        out[W:] -= c[:-W]
        return out > 0
    M['x1'] = anyroll(st, 61)[r0:].copy()
    M['x2'] = anyroll((cr > 0) & (cr < 1.0), 21)[r0:].copy()
    M['xbuy'] = M['x1'] | M['x2']
    del cr
    M['last_tr'] = np.where(tr.any(0), T - 1 - tr[::-1].argmax(0), -1) - r0
    M['tr'], M['st'] = tr[r0:].copy(), st[r0:].copy()
    M['dates'] = dts[r0:n].copy()
    M['codes'] = z['codes'].copy()
    M['stamp'] = np.where(M['dates'] < STAMP_SWITCH, STAMP_OLD, STAMP_NEW)
    z.close()
    return M


def run_account(M, a, b, K0, sc_fn=None, slip=1e-3, lots=True, mincomm=5.0, frac_unaff=False, zero_term=False,
                nofee=False, nslots=K, nq=NQ, orders=None, detail=False, use_x=False):
    """One portfolio account on rows a..b-1 (cash at the open of row a). sc_fn(t) -> score row of signal day t.
    orders (tests only): {day j: (sell indices, buy indices)} replaces the decision rule.
    lots/mincomm/frac_unaff/nofee switch off parts of the frictions for the difference decomposition only."""
    oraw, oq, off0, cff0 = M['o_raw'], M['o_q'], M['off0'], M['cff0']
    bok, sok, adv, last_tr, stampv = M['buy_ok'], M['sell_ok'], M['adv'], M['last_tr'], M['stamp']
    comm, trf = (0.0, 0.0) if nofee else (COMM, TRANSFER)
    if nofee:
        slip, mincomm = 0.0, 0.0
    N = oraw.shape[1]
    n = b - a
    units = np.zeros(N)
    held = np.zeros(N, bool)
    pend = np.zeros(N, bool)
    everdead = np.zeros(N, bool)
    cash = float(K0)
    nav_prev = float(K0)
    nav = np.empty(n)
    navo = np.empty(n)
    navz = np.empty(n)
    nheld = np.zeros(n, np.int32)
    cashr = np.empty(n)
    c = dict(nbuy=0, nsell=0, pend_days=0, fail_block=0, fail_lot=0, fail_cash=0, cap_bind=0, cash_cut=0,
             buy_amt=0.0, sell_amt=0.0, cost=0.0)
    qd = np.zeros(6)
    log = []
    for j in range(n):
        e = a + j
        uv = units * (last_tr >= e) if zero_term else units
        navo[j] = cash + uv @ off0[e]
        if orders is not None:
            sells, buys = orders.get(j, (EMPTY, EMPTY))
        else:
            f_, s_, c_ = decide(sc_fn(e - 1), M['U'][e - 1], held, bok[e], sok[e], nslots, nq, M['xbuy'][e - 1] if use_x else None)
            sells, buys = s_, np.concatenate([f_, c_])
        if pend.any():
            sells = np.concatenate([np.nonzero(pend)[0], np.asarray(sells, np.int64)])
        st_e = 0.0 if nofee else stampv[e]
        ex_s, ex_b = [], []
        for i in sells:
            if not held[i]:
                continue
            if sok[e, i]:
                gross = units[i] * oq[e, i]
                amt = gross * (1 - slip)
                fee = max(mincomm, comm * amt) + trf * amt + st_e * amt
                cash += amt - fee
                c['cost'] += gross - (amt - fee)
                c['sell_amt'] += amt
                c['nsell'] += 1
                units[i] = 0.0
                held[i] = False
                pend[i] = False
                ex_s.append(int(i))
            else:
                pend[i] = True
                c['pend_days'] += 1
        target = nav_prev / nslots
        for i in buys:
            if held[i]:
                continue
            if not bok[e, i]:
                c['fail_block'] += 1
                continue
            px = oraw[e, i] * (1 + slip)
            lim = target
            cap = 0.01 * adv[e, i]
            if cap < lim:
                lim = cap
                c['cap_bind'] += 1
            if lots:
                nsh = floor(lim / px / 100.0) * 100.0
                if nsh <= 0:
                    if not frac_unaff:
                        c['fail_lot'] += 1
                        continue
                    nsh = min(lim, cash / (1 + comm + trf)) / px
                else:
                    cut = False
                    while nsh > 0 and nsh * px + max(mincomm, comm * nsh * px) + trf * nsh * px > cash:
                        nsh -= 100.0
                        cut = True
                    if nsh <= 0:
                        if not frac_unaff:
                            c['fail_cash'] += 1
                            continue
                        nsh = cash / (1 + comm + trf) / px
                    elif cut:
                        c['cash_cut'] += 1
            else:
                nsh = min(lim, cash / (1 + comm + trf)) / px
            if nsh <= 0:
                continue
            notional = nsh * px
            fee = max(mincomm, comm * notional) + trf * notional
            cash -= notional + fee
            units[i] = nsh * oraw[e, i] / oq[e, i]
            held[i] = True
            c['cost'] += notional + fee - nsh * oraw[e, i]
            c['buy_amt'] += notional
            c['nbuy'] += 1
            ex_b.append(int(i))
        uv = units * (last_tr >= e) if zero_term else units
        nav[j] = cash + uv @ cff0[e]
        nav_prev = nav[j]
        nheld[j] = held.sum()
        cashr[j] = cash / nav[j]
        everdead |= held & (last_tr < e)
        navz[j] = cash + (units * (last_tr >= e)) @ cff0[e]      # same trades, ended series marked 0 (valuation overlay)
        if detail:
            qt, u = M['qa'][e - 1], M['U'][e - 1]
            hq = held & u & (qt < 5)
            qd[:5] += np.bincount(qt[hq], minlength=5)[:5]
            qd[5] += held.sum() - hq.sum()
            log.append((ex_s, ex_b, np.nonzero(held)[0]))
    dead_val = float((units * everdead) @ cff0[b - 1])
    return dict(nav=nav, navo=navo, navz=navz, nheld=nheld, cashr=cashr, c=c, qd=qd, log=log, n_term=int(everdead.sum()),
                dead_val=dead_val, dead_idx=np.nonzero(everdead)[0], pend_final=int(pend.sum()))


def bench_B(M, a, b, K0, slip=1e-3, mincomm=5.0, zero_term=False, use_x=False):
    """Benchmark B: one hold account (capital K0, whole lots, buy at the first executable open of the window) for
    every stock of the frozen pool of row a-1; NAV = equal-weighted mean of account equity / K0.
    Buy rule copied from design\\backtest_engine\\engine.py simulate(). use_x (version 2): the pool excludes X1 or X2 stocks."""
    idx = np.nonzero(M['U'][a - 1] & ~M['xbuy'][a - 1] if use_x else M['U'][a - 1])[0]
    oraw, oq, off0, cff0 = M['o_raw'], M['o_q'], M['off0'], M['cff0']
    last = M['last_tr'][idx]
    n = b - a
    cash = np.full(idx.size, float(K0))
    units = np.zeros(idx.size)
    hold = np.zeros(idx.size, bool)
    nav = np.empty(n)
    navo = np.empty(n)
    for j in range(n):
        e = a + j
        uv = units * (last >= e) if zero_term else units
        navo[j] = (cash + uv * off0[e, idx]).mean() / K0
        if not hold.all():
            w = np.nonzero(~hold & M['buy_ok'][e, idx])[0]
            if w.size:
                ii = idx[w]
                px = oraw[e, ii] * (1 + slip)
                nn = np.floor(cash[w] / (px * (1 + COMM + TRANSFER)) / 100) * 100
                tot = nn * px + np.maximum(mincomm, COMM * nn * px) + TRANSFER * nn * px
                nn = np.maximum(np.where(tot > cash[w], nn - 100, nn), 0)
                tot = nn * px + np.where(nn > 0, np.maximum(mincomm, COMM * nn * px), 0.0) + TRANSFER * nn * px
                ok = nn > 0
                w, ii, nn, tot = w[ok], ii[ok], nn[ok], tot[ok]
                cash[w] -= tot
                units[w] = nn * oraw[e, ii] / oq[e, ii]
                hold[w] = True
        uv = units * (last >= e) if zero_term else units
        nav[j] = (cash + uv * cff0[e, idx]).mean() / K0
    return dict(nav=nav, navo=navo, npool=int(idx.size), nbought=int(hold.sum()), n_term=int((hold & (last < b - 1)).sum()), idx=idx)


def maxdd(path):
    return float((1 - path / np.maximum.accumulate(path)).max())


def readings(navS, navB, K0):
    """navS in CNY, navB as a ratio to capital. The three account targets (excess return, t, drawdown) on one window."""
    n = navS.size
    rS = navS / np.r_[K0, navS[:-1]] - 1
    rB = navB / np.r_[1.0, navB[:-1]] - 1
    d = rS - rB
    annS = (navS[-1] / K0) ** (252.0 / n) - 1
    annB = navB[-1] ** (252.0 / n) - 1
    sd = d.std(ddof=1)
    return dict(n=n, annS=annS * 100, annB=annB * 100, t1=(annS - annB) * 100, t1_arith=d.mean() * 25200,
                t=d.mean() / sd * sqrt(n), t551=d.mean() / sd * sqrt(551), te=sd * 100, ddS=maxdd(np.r_[K0, navS]) * 100,
                ddB=maxdd(np.r_[1.0, navB]) * 100, navS=navS[-1] / K0, navB=navB[-1], d=d, rS=rS, rB=rB)
