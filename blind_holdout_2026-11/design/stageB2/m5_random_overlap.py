# m5_random_overlap.py — (A) width of the random-selection control; (B) path overlap of two baseline rules under
# the goal-5 perturbation (score * (1 + 1e-6 * noise), start shifted 0..4 days). Training rows only.
# (A) fast account model: K slots, each holds a random pool name for H days (staggered), 500k account costs with
#     minimum commission, 0.05% stamp duty, no lot rounding, no limit checks. 200 seeds, four 551-day windows.
# (B1) Qlib-style quota rule: hold K=50, each day sell the held names that fall in the bottom n=2 of (held + top-n
#      outsiders) and buy the same number of top outsiders. Score = minus the raw signal (cold is good).
# (B2) passive rule: hold every pool name outside the hottest 30%.
from b2common import *

start('m5_random_overlap')
B = build(keep_raw=True)
d, T, N, U, sig, OQ, raw = B['d'], B['T'], B['N'], B['U'], B['sig'], B['OQ'], B['raw']
dates = d['dates']
Off, oo = oo_returns(OQ)
first_ok = int(np.nonzero(U.sum(1) >= 200)[0][0]) + 1
rng = np.random.default_rng(20261109)
_c = {}


def cands(t):
    if t not in _c:
        _c[t] = np.nonzero(U[t])[0]
    return _c[t]


P('TABLE A  random control: window | sigma_e pool %/day | K | H | mean NAV | sd | q97.5 | (q97.5/mean-1) % | per day bp | formula 1.96*sigma_e*sqrt(L/K) % | ratio')
for w in WIN:
    ts = max(w, first_ok)
    te = min(w + WLEN, T - 1)
    L = te - ts
    v2 = []
    for t in range(ts, te):
        r = oo[t, U[t - 1]]
        r = r[np.isfinite(r)]
        v2.append(r.var())
    se = sqrt(np.mean(v2))
    for K in (30, 50, 80):
        comm = max(1e-4, 5.0 / (500000.0 / K))
        cb = 1.0 / (1.001 * (1 + comm + 1e-5))
        cs = 0.999 * (1 - comm - 1e-5 - 0.0005)
        for H in (20, 40):
            nav = np.zeros(200)
            for s in range(K):
                off = (s * H) // K
                bounds = [ts] + list(range(ts + off + 1, te, H)) + [te]
                v = np.ones(200)
                for a, b_ in zip(bounds[:-1], bounds[1:]):
                    c = cands(a - 1)
                    idx = c[rng.integers(0, c.size, 200)]
                    v *= Off[b_, idx] / Off[a, idx] * cb
                    if b_ != te:
                        v *= cs
                nav += v
            nav /= K
            mean, sd, q = nav.mean(), nav.std(), np.quantile(nav, 0.975)
            ex = q / mean - 1
            f = 1.96 * se * sqrt(L / K)
            P('%s | %.3f | %2d | %2d | %.4f | %.4f | %.4f | %6.2f | %5.2f | %6.2f | %.2f'
              % (dates[ts][:7], se * 100, K, H, mean, sd, q, ex * 100, ex / L * 1e4, f * 100, ex / f))


def pair_overlap(held, size):
    k = held.shape[0]
    c = held.sum(0).astype(np.float64)
    return ((c ** 2).sum() - k * size) / (k * (k - 1) * size)


def quota_rule(score, t0, t1, K=50, n=2, nruns=20):
    held = np.zeros((nruns, N), bool)
    started = np.zeros(nruns, bool)
    shifts = np.arange(nruns) % 5
    ov_all, ov_same, trades, days = [], [], 0, 0
    for t in range(t0, t1):
        u = U[t]
        base = score[t].astype(np.float64)
        for r in range(nruns):
            if t < t0 + shifts[r]:
                continue
            s = np.where(u, base * (1 + 1e-6 * rng.standard_normal(N)), -np.inf)
            h = held[r]
            if not started[r]:
                h[np.argpartition(-s, K)[:K]] = True
                started[r] = True
                continue
            sc = np.where(h, -np.inf, s)
            ci = np.argpartition(-sc, n)[:n]
            ci = ci[np.isfinite(sc[ci])]
            comb = np.concatenate([np.nonzero(h)[0], ci])
            worst = comb[np.argsort(s[comb], kind='stable')][:n]
            sell = worst[h[worst]]
            if sell.size:
                buy = ci[np.argsort(-s[ci])][:sell.size]
                h[sell] = False
                h[buy] = True
                if r == 0:
                    trades += sell.size
        if t >= t0 + 4:
            ov_all.append(pair_overlap(held, K))
            ov_same.append(pair_overlap(held[shifts == 0], K))
            days += 1
    return np.array(ov_all), np.array(ov_same), trades / max(days, 1)


P('\nTABLE B1  quota rule K=50 n=2: signal | window | overlap o, all 20 runs: mean / min / last 100 days | same-start runs only: mean / min | swaps per day (run 0) | implied holding days')
for sn in ('S3_comp4', 'S1_max20'):
    score = -raw[sn]
    for w in WIN:
        ts = max(w, first_ok)
        te = min(w + WLEN, T - 1)
        oa, os_, tpd = quota_rule(score, ts, te)
        P('%s | %s | %.3f / %.3f / %.3f | %.3f / %.3f | %.2f | %.1f' % (sn, dates[ts][:7], oa.mean(), oa.min(), oa[-100:].mean(),
                                                                    os_.mean(), os_.min(), tpd, 50 / max(tpd, 1e-9)))
P('\nTABLE B2  passive rule (hold all names outside the hottest 30%), S3, last window: overlap mean / min over 20 perturbed runs')
score = raw['S3_comp4']
ts, te = WIN[3], T - 1
ov = []
for t in range(ts, te):
    u = cands(t)
    base = score[t, u].astype(np.float64)
    k = int(0.7 * u.size)
    held = np.zeros((20, u.size), bool)
    for r in range(20):
        s = base * (1 + 1e-6 * rng.standard_normal(u.size))
        held[r, np.argpartition(s, k)[:k]] = True
    ov.append(pair_overlap(held, k))
ov = np.array(ov)
P('%.6f / %.6f' % (ov.mean(), ov.min()))
finish()
