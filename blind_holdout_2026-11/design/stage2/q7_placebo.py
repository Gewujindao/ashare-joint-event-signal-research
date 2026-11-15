# q7_placebo.py — reads: tradable, close_qfq (training rows only).
# Zero-information random in/out sequences (out fraction 2/10/20%, block length 3/5/10 days) on four 551-day
# training windows (starts 0, 547, 1094, 1641; adjacent windows overlap by 4 days because 2192 < 4*551).
# Computes: t1 and sd(X) of "strategy minus hold" vs the stage-1 formulas; sd of the circular-shift control
# (200 seeds, FFT cross-correlation) vs sigma_idio*sqrt(f*T/N); sd of the cross-stock swap control (200 seeds).
# Costs use the holdout tax (buy 11.1bp, sell 16.1bp). Returns are close-to-close; suspended days count 0.
from common import *

start('q7_placebo')
d = load('tradable', 'close_qfq')
tr = d['tradable']
seas = seasoned(tr, ~np.isnan(d['close_qfq']))
Cff = ffill(np.where(tr, d['close_qfq'], np.nan).astype(np.float32))
W = 551
rng = np.random.default_rng(20261109)
CB, CS = 11.1e-4, 16.1e-4
P('win f L | f_real Leff | Zm sig_m% | t1_nocost t1_cost pred_t1_cost | sdX_bp pred_bp | shift: sd_logNAV% pred% sd_NAV% q975-mean% per_outday_bp | swap: sd_NAV%')
for s in (0, 547, 1094, 1641):
    pool = seas[s]
    Pm = Cff[s:s + W][:, pool].astype(np.float64)
    n = Pm.shape[1]
    r = np.zeros((W, n))
    r[1:] = Pm[1:] / Pm[:-1] - 1
    lr = np.log1p(r)
    m = r.mean(1)
    sig_m = m.std(ddof=1)
    Zm = sqrt(W) * m.mean() / sig_m
    sig_idio = (r - m[:, None]).std()
    tot = lr.sum(0)
    Flr = np.fft.rfft(lr, axis=0)
    P('--- window %s..%s  n=%d  sig_idio %.3f%%' % (d['dates'][s], d['dates'][s + W - 1], n, sig_idio * 100))
    for f in (0.02, 0.10, 0.20):
        for L in (3, 5, 10):
            nb = W // L + 2
            blk = rng.random((nb, n)) < f
            rows = (np.arange(W)[:, None] + rng.integers(0, L, n)[None, :]) // L
            out = np.take_along_axis(blk, rows, axis=0)
            ex = np.zeros((W, n), bool)
            ex[1:] = out[1:] & ~out[:-1]
            en = np.zeros((W, n), bool)
            en[1:] = ~out[1:] & out[:-1]
            fr = out.mean()
            Leff = out.sum() / max(ex.sum() + out[0].sum(), 1)
            X0 = -(out * r).mean(1)
            X1 = X0 - (CS * ex.sum(1) + CB * en.sum(1)) / n
            t0 = sqrt(W) * X0.mean() / X0.std(ddof=1)
            t1 = sqrt(W) * X1.mean() / X1.std(ddof=1)
            pred = -Zm - sqrt(W) * ((CB + CS) / Leff) / sig_m
            lncost = ex.sum(0) * np.log1p(-CS) + en.sum(0) * np.log1p(-CB)
            Sall = np.fft.irfft(np.conj(np.fft.rfft(out.astype(np.float64), axis=0)) * Flr, n=W, axis=0)
            ar = np.arange(n)
            vl, va = [], []
            for seed in range(200):
                S = Sall[rng.integers(0, W, n), ar]
                ln = tot - S + lncost
                vl.append(ln.mean())
                va.append(np.exp(ln).mean())
            vl, va = np.array(vl), np.array(va)
            vs = []
            for seed in range(200):
                S = (out[:, rng.permutation(n)] * lr).sum(0)
                vs.append(np.exp(tot - S + lncost).mean())
            vs = np.array(vs)
            q975 = np.percentile(va, 97.5) - va.mean()
            P('%s %.2f %2d | %.3f %.1f | %.2f %.3f | %.2f %.2f %.2f | %.2f %.2f | %.3f %.3f %.3f %.3f %.2f | %.3f' % (
                d['dates'][s][:7], f, L, fr, Leff, Zm, sig_m * 100, t0, t1, pred,
                X0.std(ddof=1) * 1e4, fr * sig_m * 1e4,
                vl.std(ddof=1) * 100, sig_idio * sqrt(fr * W / n) * 100, va.std(ddof=1) * 100, q975 * 100,
                q975 / va.mean() / (fr * W) * 1e4, vs.std(ddof=1) * 100))
finish()
