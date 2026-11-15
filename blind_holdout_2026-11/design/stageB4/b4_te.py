# b4_te.py — stage B4: can the tracking error of the in-zone set (S1, enter <=30%, leave >60%) be removed by construction?
# Training rows only (common.load slices to <= 2022-12-30 and asserts). numpy only. No arrays written to disk.
# base = equal weight in zone; C1 = entropy-balanced on beta, size rank, liquidity rank; C2 = C1 + vol rank + 20d-return rank.
# C1B (four windows) = C1 with base measure = buy-and-hold benchmark weights and benchmark-weighted targets.
import sys
sys.path.insert(0, r'<BLIND_ROOT>\design\stageB2')
from b2common import *
C.OUTDIR = C.ROOT + r'\design\stageB4'
start('b4_te')

d0 = load('tradable', 'close_raw', 'volume_shares', 'turnover_pct')
assert d0['tradable'].shape[0] == 2192 and d0['dates'][-1] == '2022-12-30' and bool((d0['dates'] <= '2022-12-30').all())
with np.errstate(divide='ignore', invalid='ignore'):
    ok = d0['tradable'] & (d0['turnover_pct'] > 0) & (d0['volume_shares'] > 0) & (d0['close_raw'] > 0)
    mc_ = np.log(d0['close_raw'].astype(np.float64) * d0['volume_shares'] / (d0['turnover_pct'] / 100.0))
    lmc = ffill(np.where(ok, mc_, np.nan).astype(np.float32))      # log float market cap proxy
del d0, mc_, ok

B = build(extra=('amount_cny',))
d, T, N, U, sig, OQ = B['d'], B['T'], B['N'], B['U'], B['sig'], B['OQ']
assert T == 2192 and d['dates'][-1] == '2022-12-30'
tr = d['tradable']
yrs = d['years']
per = periods(yrs)
for k in ('open_raw', 'preclose_raw', 'open_qfq'):
    del d[k]
del sig['S3_comp4']
Cff = ffill(np.where(tr, d['close_qfq'], np.nan).astype(np.float32))
r1 = np.full((T, N), np.nan, np.float32)
r1[1:] = Cff[1:] / Cff[:-1] - 1
r1[~tr] = np.nan
ret20 = np.full((T, N), np.nan, np.float32)
ret20[20:] = Cff[20:] / Cff[:-20] - 1
del Cff, d['close_qfq']
val = np.isfinite(r1)
mcc = np.array([r1[t, val[t]].mean() if val[t].sum() >= 200 else 0.0 for t in range(T)])
A = np.cumsum(np.where(val, r1, 0).astype(np.float64) * mcc[:, None], axis=0)
Bm = np.cumsum(val * (mcc ** 2)[:, None], axis=0)
cnt = np.cumsum(val, axis=0, dtype=np.int32)
W = 250
beta = np.full((T, N), np.nan, np.float32)
with np.errstate(divide='ignore', invalid='ignore'):
    bb = A[:W] / Bm[:W]
    bb[cnt[:W] < 60] = np.nan
    beta[:W] = bb
    bb = (A[W:] - A[:-W]) / (Bm[W:] - Bm[:-W])
    bb[(cnt[W:] - cnt[:-W]) < 120] = np.nan
    beta[W:] = bb
del A, Bm, cnt, bb, r1, val
np.clip(beta, -0.5, 3.0, out=beta)
am = np.cumsum(np.where(tr, np.nan_to_num(d['amount_cny']), 0).astype(np.float64), axis=0)
cn = np.cumsum(tr, axis=0, dtype=np.int32)
liq = np.full((T, N), np.nan, np.float32)
x_ = am[60:] - am[:-60]
n_ = cn[60:] - cn[:-60]
liq[60:] = np.where(n_ >= 30, x_ / np.maximum(n_, 1), np.nan)
del am, cn, x_, n_, d['amount_cny']
U2 = U & np.isfinite(beta) & np.isfinite(lmc) & np.isfinite(liq) & np.isfinite(ret20)
rs = pct_rank_rows(lmc, U2)
rl = pct_rank_rows(liq, U2)
rr = pct_rank_rows(ret20, U2)
del lmc, liq, ret20
CH = [beta, rs, rl, sig['S2_vol20'], rr]
CHN = ('beta', 'size', 'liq', 'vol', 'ret20')

# zone state machine, identical to m2_residence.py (S1, q = 0.3)
R = np.where(U, sig['S1_max20'], np.nan)
Zh = np.zeros((T, N), bool)
cur = np.zeros(N, bool)
ent = 0
with np.errstate(invalid='ignore'):
    for t in range(T):
        r = R[t]
        nxt = (r <= 0.3) | (cur & (r <= 0.6))
        ent += int((nxt & ~cur).sum())
        cur = nxt
        Zh[t] = cur
RES = Zh.sum() / ent
del R
Off, oo = oo_returns(OQ)
P('pool U mean n=%.0f  U2 mean n=%.0f  zone residence (entries) = %.1f days' % (U.sum(1)[U.sum(1) > 0].mean(), U2.sum(1)[U2.sum(1) > 0].mean(), RES))


def balance(x, q, lam):
    # weights w ∝ q exp(x lam) with sum_i w_i x_i = 0 (x already centred on the target). Newton with backtracking on the dual.
    def F(l):
        a = x @ l
        mx = a.max()
        e = q * np.exp(a - mx)
        s = e.sum()
        return mx + np.log(s), e / s
    f, w = F(lam)
    err = 1.0
    for it in range(40):
        g = w @ x
        err = np.abs(g).max()
        if err < 1e-8:
            break
        H = (x * w[:, None]).T @ x - np.outer(g, g)
        try:
            step = np.linalg.solve(H + 1e-10 * np.eye(len(g)), g)
        except np.linalg.LinAlgError:
            break
        s = 1.0
        while s > 1e-4:
            f2, w2 = F(lam - s * step)
            if f2 <= f:
                break
            s *= 0.5
        lam = lam - s * step
        f, w = f2, w2
    return w, lam, err


KS = ('base', 'C1', 'C2')
ser = {k: np.full(T, np.nan) for k in KS}
sd2 = {k: np.full(T, np.nan) for k in KS}
nef = {k: np.full(T, np.nan) for k in KS}
tov = {k: np.full(T, np.nan) for k in KS}
prev = {k: None for k in KS}
m = np.full(T, np.nan)
contrib = np.full((T, 5), np.nan)
resid = np.full(T, np.nan)
dgs = np.full((T, 5), np.nan)
errs = {'C1': [], 'C2': []}
lam1 = np.zeros(3)
lam2 = np.zeros(5)
for t in range(1, T - 1):
    ra = oo[t]
    u = U2[t - 1] & np.isfinite(ra)
    z = Zh[t - 1] & u
    if z.sum() < 50 or u.sum() < 200:
        continue
    iu = np.nonzero(u)[0]
    iz = np.nonzero(z)[0]
    r = ra[iu].astype(np.float64)
    Xu = np.column_stack([ch[t - 1, iu] for ch in CH]).astype(np.float64)
    xm = Xu.mean(0)
    Xz = np.column_stack([ch[t - 1, iz] for ch in CH]).astype(np.float64)
    rz = ra[iz].astype(np.float64)
    m[t] = r.mean()
    D = np.column_stack([np.ones(iu.size), Xu - xm])
    coef = np.linalg.solve(D.T @ D, D.T @ r)
    dg = Xz.mean(0) - xm
    dgs[t] = dg
    contrib[t] = dg * coef[1:]
    resid[t] = rz.mean() - m[t] - contrib[t].sum()
    q = np.full(iz.size, 1.0 / iz.size)
    xd = Xz - xm
    w1, lam1, e1 = balance(xd[:, :3], q, lam1)
    w2, lam2, e2 = balance(xd, q, lam2)
    errs['C1'].append(e1)
    errs['C2'].append(e2)
    for k, w in (('base', q), ('C1', w1), ('C2', w2)):
        bw = w @ rz
        ser[k][t] = bw
        sd2[k][t] = w @ (rz - bw) ** 2
        nef[k][t] = 1.0 / (w @ w)
        full = np.zeros(N)
        full[iz] = w
        if prev[k] is not None:
            tov[k][t] = 0.5 * np.abs(full - prev[k]).sum()
        prev[k] = full
P('balance not converged (err > 1e-6): C1 %d of %d days, C2 %d of %d days; max err C1 %.2e C2 %.2e' % (
    sum(e > 1e-6 for e in errs['C1']), len(errs['C1']), sum(e > 1e-6 for e in errs['C2']), len(errs['C2']),
    max(errs['C1']), max(errs['C2'])))

wins = []
for w in WIN:
    mk = np.zeros(T, bool)
    mk[w:w + WLEN] = True
    wins.append(('W%d_%s' % (w, d['dates'][w][:7]), mk))
PER = [('ALL', per['ALL']), ('EX2015', per['EX2015']), ('P1_14-16', per['P1_14-16']), ('P2_17-19', per['P2_17-19']),
       ('P3_20-22', per['P3_20-22'])] + wins
x0 = ser['base'] - m
fin = np.isfinite(x0)

P('\nTABLE 1  variance decomposition of x = (equal-weight in-zone set) - (daily equal-weight pool)')
P('period | TE(x)%% | sd(m)%% | beta | market part (beta-1)^2 var(m) share | cov shares: %s resid | mean bp/d: %s resid | mean exposure gap: %s'
  % (' '.join(CHN), ' '.join(CHN), ' '.join(CHN)))
for pn, pmk in PER:
    v = pmk & fin
    xx = x0[v]
    vx = xx.var(ddof=1)
    bt = np.cov(ser['base'][v], m[v])[0, 1] / m[v].var(ddof=1)
    sh = [np.cov(contrib[v, j], xx)[0, 1] / vx for j in range(5)] + [np.cov(resid[v], xx)[0, 1] / vx]
    mn = [contrib[v, j].mean() * 1e4 for j in range(5)] + [resid[v].mean() * 1e4]
    P('%-13s | %.3f | %.3f | %.3f | %.3f | %s | %s | %s' % (pn, sqrt(vx) * 100, m[v].std(ddof=1) * 100, bt, (bt - 1) ** 2 * m[v].var(ddof=1) / vx,
                                                       ' '.join('%+.3f' % s for s in sh), ' '.join('%+.2f' % s for s in mn),
                                                       ' '.join('%+.3f' % s for s in dgs[v].mean(0))))
P('  sd of each piece %% (ALL): %s resid %.3f' % (' '.join('%.3f' % (contrib[fin, j].std(ddof=1) * 100) for j in range(5)), resid[fin].std(ddof=1) * 100))

CK = ((50, 35.2), (80, 41.2), (100, 45.2))


def line(tag, pn, b, ref, v, s2, ne, tv):
    x = b[v] - ref[v]
    g = x.mean()
    te = x.std(ddof=1)
    bt = np.cov(b[v], ref[v])[0, 1] / ref[v].var(ddof=1)
    se = sqrt(np.nanmean(s2[v]))
    nb = np.nanmean(ne[v])
    sF = sqrt(max(te ** 2 - se ** 2 / nb, 0.0))
    H = 1.0 / np.nanmean(tv[v])
    out = []
    for K, c in CK:
        tk = sqrt(se ** 2 / K + sF ** 2)
        for Hu in (H, RES):
            net = g - c * 1e-4 / Hu
            out.append('%5.2f %+.2f' % (net * 25200, net / tk * sqrt(551)))
    tinf = (g - 27.2e-4 / H) / sF * sqrt(551)
    P('%-4s %-13s | %5.2f | %.3f | %.3f | %.3f | %5.0f | %.3f | %5.1f | %s | %+.2f' % (tag, pn, g * 1e4, bt, te * 100, se * 100, nb, sF * 100, H,
                                                                                 ' ; '.join(out), tinf))


P('\nTABLE 2  versus the daily equal-weight pool. columns: gross bp/d | beta | TE% | sigma_e(w)% | n_eff | sigma_F% | eff. residence (1/weight turnover) |')
P('         for K=50, 80, 100: [annual net %%, expected t over 551 days] with cost over eff. residence , then with cost over entry residence %.1f | t K=inf' % RES)
for k in KS:
    for pn, pmk in PER:
        line(k, pn, ser[k], m, pmk & fin, sd2[k], nef[k], tov[k])
    P('')

# four windows against the buy-and-hold benchmark; C1B = benchmark-weight base measure
P('TABLE 3  versus the buy-and-hold benchmark B (bought equally on day 1 of the window), four windows')
P('window | TE(m-B)% | mean(m-B) bp/d | corr(x_base, m-B) ; then per construction the same columns as TABLE 2 with B as reference')
first_ok = int(np.nonzero(U2.sum(1) >= 200)[0][0]) + 1
for w in WIN:
    ts = max(w, first_ok)
    te_ = min(w + WLEN, T - 1)
    names = U[ts - 1] & np.isfinite(Off[ts])
    navb = np.nanmean(Off[ts:te_ + 1][:, names] / Off[ts][names][None, :], axis=1)
    rb = np.full(T, np.nan)
    rb[ts:te_] = navb[1:] / navb[:-1] - 1
    v = np.zeros(T, bool)
    v[ts:te_] = True
    v &= fin
    dmb = m[v] - rb[v]
    P('%s | %.3f | %+.2f | %+.3f' % (d['dates'][ts][:7], dmb.std(ddof=1) * 100, dmb.mean() * 1e4, np.corrcoef(x0[v], dmb)[0, 1]))
    sB = np.full(T, np.nan)
    s2B = np.full(T, np.nan)
    neB = np.full(T, np.nan)
    tvB = np.full(T, np.nan)
    pv = None
    lam = np.zeros(3)
    for t in range(ts, te_):
        if not v[t]:
            continue
        ra = oo[t]
        val_ = Off[t - 1] / Off[ts] if t > ts else np.ones(N)
        u = U2[t - 1] & np.isfinite(ra) & names & np.isfinite(val_)
        z = Zh[t - 1] & u
        iu = np.nonzero(u)[0]
        iz = np.nonzero(z)[0]
        qu = val_[iu].astype(np.float64)
        qu /= qu.sum()
        Xu = np.column_stack([ch[t - 1, iu] for ch in CH[:3]]).astype(np.float64)
        xmB = qu @ Xu
        qz = val_[iz].astype(np.float64)
        qz /= qz.sum()
        Xz = np.column_stack([ch[t - 1, iz] for ch in CH[:3]]).astype(np.float64)
        wB, lam, _ = balance(Xz - xmB, qz, lam)
        rz = ra[iz].astype(np.float64)
        bw = wB @ rz
        sB[t] = bw
        s2B[t] = wB @ (rz - bw) ** 2
        neB[t] = 1.0 / (wB @ wB)
        full = np.zeros(N)
        full[iz] = wB
        if pv is not None:
            tvB[t] = 0.5 * np.abs(full - pv).sum()
        pv = full
    for k in KS:
        line(k, d['dates'][ts][:7], ser[k], rb, v, sd2[k], nef[k], tov[k])
    line('C1B', d['dates'][ts][:7], sB, rb, v, s2B, neB, tvB)
    P('')
finish()
