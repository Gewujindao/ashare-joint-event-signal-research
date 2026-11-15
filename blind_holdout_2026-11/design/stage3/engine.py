# engine.py — stage-3 strategy rule + backtest engine + target metrics (numpy only, CPU, one day per step
# vectorised over all stocks). Reuses the resource guard and helpers of design\stage2\common.py.
#
# Parts:
#   signals(d)            causal event matrices (row t uses rows <= t only)
#   plan(d, variant, th)  virtual state machine -> want[t] = desired holding at the open of day t
#                         (decided at the close of t-1 from data up to t-1)
#   simulate(...)         per-stock accounts under PROTOCOL sections 2-3, 6-2 and MANIFEST section 6
#   metrics(...)          the three targets, circular-shift control (200 seeds) and cross-stock swap control
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'stage2'))
import common
from common import *

common.OUTDIR = os.path.dirname(os.path.abspath(__file__))

SLIP = 0.001          # PROTOCOL section 3: 0.1% per side on the fill price
COMM, COMM_MIN = 1e-4, 5.0
TRANSFER = 1e-5
STAMP_OLD, STAMP_NEW, STAMP_SWITCH = 1e-3, 5e-4, '2023-08-28'

# baseline rule constant: out for 3 open-to-open intervals (stage 2, Q3 table, L = 3). No tuning, no variants.
L_BOARD = 3


def stamp_rates(dates):
    return np.where(dates < STAMP_SWITCH, STAMP_OLD, STAMP_NEW)


def signals(d):
    """Baseline event at the close of day t: non-ST, seasoned stock closes limit-up for at least the second
    consecutive day and the board was opened during the day (low below the limit).
    Needs tradable, st, low_raw, close_raw, preclose_raw, close_qfq. Row t depends on rows <= t only."""
    tr, st = d['tradable'], d['st']
    seas = seasoned(tr, ~np.isnan(d['close_qfq']))
    pc, c, l = d['preclose_raw'], d['close_raw'], d['low_raw']
    lim = np.where(st, 0.05, 0.10).astype(np.float32)
    up = pc * (1 + lim) - 0.005
    base = seas & (pc > 0)
    LU = base & (c >= up)
    LU1 = base & (l >= up)
    T, N = LU.shape
    streak = np.zeros((T, N), np.int16)
    run = np.zeros(N, np.int16)
    for t in range(T):
        run = np.where(LU[t], run + 1, 0).astype(np.int16)
        streak[t] = run
    board = base & ~st & LU & (streak >= 2) & ~LU1
    return {'board': board, 'seas': seas}


def plan(board, sell_ok, buy_ok, L=L_BOARD):
    """Virtual state machine. Returns want (T,N): desired holding at the open of day t (decided at the close
    of t-1). Event at close t -> sell at the first executable open >= t+1, buy back at the first executable
    open >= (sell open + L). A blocked sell stays pending; a blocked buy-back is retried every day."""
    T, N = sell_ok.shape
    Lnew = np.where(board, L, 0).astype(np.int8)
    vpos = np.ones(N, bool)
    pend = np.zeros(N, bool)
    Lcur = np.zeros(N, np.int32)
    reent = np.zeros(N, np.int32)
    want = np.ones((T, N), bool)
    for t in range(T):
        if t > 0:
            s = vpos & ~want[t] & sell_ok[t]
            vpos[s] = False
            pend[s] = False
            reent[s] = t + Lcur[s]
            b = ~vpos & want[t] & buy_ok[t]
            vpos[b] = True
        ln = Lnew[t].astype(np.int32)
        ev = ln > 0
        h = vpos & ev
        Lcur[h] = np.where(pend[h], np.maximum(Lcur[h], ln[h]), ln[h])
        pend[h] = True
        o = ~vpos & ev
        reent[o] = np.maximum(reent[o], t + 1 + ln[o])
        if t + 1 < T:
            want[t + 1] = np.where(vpos, ~pend, (t + 1) >= reent)
    return want


def labels(holding, ever_held, want_next):
    """Decision labels for the next open from the account state and the target."""
    out = np.full(holding.shape, 'HOLD', dtype='<U12')
    out[holding & ~want_next] = 'SELL_TO_CASH'
    out[~holding & want_next & ever_held] = 'RE_ENTER'
    out[~holding & want_next & ~ever_held] = 'BUY'
    return out


def fills(tr, st, o_raw, pc_raw, o_q):
    lim = np.where(st, 0.05, 0.10)
    valid = tr & (o_raw > 0) & (pc_raw > 0) & (o_q > 0)
    return valid & (o_raw > pc_raw * (1 - lim) + 0.005), valid & (o_raw < pc_raw * (1 + lim) - 0.005)


def simulate(tr, st, o_raw, pc_raw, o_q, c_q, want, K, stamp):
    """Accounts start in cash on day 0 of the window. want[t]: desired holding at the open of t.
    Position value follows the qfq ratio from the entry open; suspended days are valued at the last valid
    close (return 0). Returns equity (T,N), holding-after-open (T,N), cost paid, number of buys and sells."""
    T, N = tr.shape
    o_raw = o_raw.astype(np.float64)
    o_q = o_q.astype(np.float64)
    c_q = c_q.astype(np.float64)
    sell_ok, buy_ok = fills(tr, st, o_raw, pc_raw.astype(np.float64), o_q)
    cash = np.full(N, float(K))
    units = np.zeros(N)
    hold = np.zeros(N, bool)
    lastpx = np.full(N, np.nan)
    cost = np.zeros(N)
    nb = np.zeros(N, np.int32)
    ns = np.zeros(N, np.int32)
    eq = np.empty((T, N))
    pos = np.zeros((T, N), bool)
    for t in range(T):
        s = hold & ~want[t] & sell_ok[t]
        if s.any():
            gross = units[s] * o_q[t, s]
            amt = gross * (1 - SLIP)
            fee = np.maximum(COMM_MIN, COMM * amt) + TRANSFER * amt + stamp[t] * amt
            cash[s] += amt - fee
            cost[s] += gross - (amt - fee)
            units[s] = 0.0
            hold[s] = False
            ns[s] += 1
        b = ~hold & want[t] & buy_ok[t] & ~s
        if b.any():
            idx = np.nonzero(b)[0]
            px = o_raw[t, idx] * (1 + SLIP)
            n = np.floor(cash[idx] / (px * (1 + COMM + TRANSFER)) / 100) * 100
            tot = n * px + np.maximum(COMM_MIN, COMM * n * px) + TRANSFER * n * px
            n = np.maximum(np.where(tot > cash[idx], n - 100, n), 0)
            tot = n * px + np.where(n > 0, np.maximum(COMM_MIN, COMM * n * px), 0.0) + TRANSFER * n * px
            okb = n > 0
            idx, n, tot = idx[okb], n[okb], tot[okb]
            cash[idx] -= tot
            units[idx] = n * o_raw[t, idx] / o_q[t, idx]
            cost[idx] += tot - n * o_raw[t, idx]
            hold[idx] = True
            nb[idx] += 1
        v = tr[t] & (c_q[t] > 0)
        lastpx = np.where(v, c_q[t], lastpx)
        eq[t] = cash + np.where(hold, units * lastpx, 0.0)
        pos[t] = hold
    return eq, pos, cost, nb, ns


def maxdd(eq, K):
    path = np.concatenate([np.full((1,) + eq.shape[1:], float(K)), eq], axis=0)
    return (1 - path / np.maximum.accumulate(path, axis=0)).max(axis=0)


def tstats(x):
    n = x.size
    m, s = x.mean(), x.std(ddof=1)
    e = x - m
    v = (e * e).mean()
    for k in range(1, 6):
        v += 2 * (1 - k / 6) * (e[k:] * e[:-k]).sum() / n
    return m, m / s, sqrt(n) * m / s, m / sqrt(max(v, 1e-30) / n)


def metrics(eqS, eqH, posS, K, o_q, c_q, tr, nbS, nsS, cs_sell, seeds=200, rng=None, day_mask=None):
    """Targets 1-3 for one window. o_q, c_q, tr: window arrays of the pool. cs_sell: proportional sell cost
    used by the fast account model of the controls (buy 11.1bp)."""
    T, N = eqS.shape
    rS = eqS / np.vstack([np.full((1, N), float(K)), eqS[:-1]]) - 1
    rH = eqH / np.vstack([np.full((1, N), float(K)), eqH[:-1]]) - 1
    X = (rS - rH).mean(1)
    if day_mask is not None:
        X = X[day_mask]
    m, ratio, t, tnw = tstats(X)
    res = {'ann_excess_pct': 252 * m * 100, 'ratio': ratio, 't': t, 't_nw5': tnw, 'ndays': X.size}
    ddS, ddH = maxdd(eqS, K), maxdd(eqH, K)
    mS, mH = maxdd(eqS.sum(1), K * N), maxdd(eqH.sum(1), K * N)
    res.update(dd_med_S=float(np.median(ddS)), dd_med_H=float(np.median(ddH)), dd_merged_S=float(mS),
               dd_merged_H=float(mH), t2_pass=bool(np.median(ddS) <= np.median(ddH) and mS <= mH))
    res.update(nav_S=float(eqS[-1].mean() / K), nav_H=float(eqH[-1].mean() / K),
               out_frac=float(1 - posS.mean()), out_sd=float((1 - posS.mean(1)).std()))
    # fast account model on open-to-open intervals for the controls
    Off = ffill(np.where(tr & (o_q > 0), o_q, np.nan).astype(np.float64))
    Cff = ffill(np.where(tr & (c_q > 0), c_q, np.nan).astype(np.float64))
    g = np.zeros((T, N))
    with np.errstate(all='ignore'):
        g[:-1] = np.log(Off[1:] / Off[:-1])
        g[-1] = np.log(Cff[-1] / Off[-1])
    g = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
    out = (~posS).astype(np.float64)
    G = g.sum(0)
    lnc = nbS * np.log1p(-(SLIP + COMM + TRANSFER)) + nsS * np.log1p(-cs_sell)
    actual = np.exp(G - (out * g).sum(0) + lnc).mean()
    Sall = np.fft.irfft(np.conj(np.fft.rfft(out, axis=0)) * np.fft.rfft(g, axis=0), n=T, axis=0)
    ar = np.arange(N)
    rng = rng or np.random.default_rng(20261109)
    sh = np.array([np.exp(G - Sall[rng.integers(0, T, N), ar] + lnc).mean() for _ in range(seeds)])
    sw = []
    ob = ~posS
    for _ in range(seeds):
        p = rng.permutation(N)
        sw.append(np.exp(G - (ob[:, p] * g).sum(0) + lnc[p]).mean())
    sw = np.array(sw)
    res.update(nav_fast=float(actual), shift_mean=float(sh.mean()), shift_q975=float(np.percentile(sh, 97.5)),
               shift_z=float((actual - sh.mean()) / sh.std(ddof=1)), t3_pass=bool(actual > np.percentile(sh, 97.5)),
               swap_mean=float(sw.mean()), swap_q975=float(np.percentile(sw, 97.5)),
               swap_z=float((actual - sw.mean()) / sw.std(ddof=1)))
    return res, X
