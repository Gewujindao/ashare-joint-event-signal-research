# m4_drawdown.py — max drawdown of the cold-zone basket versus the buy-and-hold pool benchmark, four 551-day windows.
# Benchmark = equal money in every pool name on the window's first usable day, held (open-based NAV).
# Basket = daily equal-weight cold zone; net version subtracts 35.2bp/20 per day.
from b2common import *

start('m4_drawdown')
B = build()
d, T, N, U, sig, OQ = B['d'], B['T'], B['N'], B['U'], B['sig'], B['OQ']
dates = d['dates']
Off, oo = oo_returns(OQ)
first_ok = int(np.nonzero(U.sum(1) >= 200)[0][0]) + 1


def mdd(nav):
    peak = np.maximum.accumulate(nav)
    dd = 1 - nav / peak
    j = int(dd.argmax())
    i = int(nav[:j + 1].argmax())
    return dd[j], i, j


series = {}
for sn in SIGS:
    for q in (0.1, 0.2, 0.3):
        with np.errstate(invalid='ignore'):
            Z = U & (sig[sn] <= q)
        b, m, _, _, _ = basket_series(Z, U, oo)
        series[(sn, q)] = b
mser = m
COST = 35.2e-4 / 20
fails = {k: 0 for k in series}
for w in WIN:
    ts = max(w, first_ok)
    te = min(w + WLEN, T - 1)
    names = U[ts - 1]
    navb = np.nanmean(Off[ts:te + 1][:, names] / Off[ts][names][None, :], axis=1)
    db, i, j = mdd(navb)
    navm = np.concatenate([[1.0], np.cumprod(1 + np.nan_to_num(mser[ts:te]))])
    dm, _, _ = mdd(navm)
    rb = navb[1:] / navb[:-1] - 1
    P('\nwindow %s..%s  names=%d  benchmark(buy-hold) MDD=%.3f  peak %s trough %s (%.2f yr)  end NAV=%.3f | daily-EW pool MDD=%.3f end NAV=%.3f | TE(buy-hold vs daily-EW)=%.3f%%/day'
      % (dates[ts], dates[te], names.sum(), db, dates[ts + i], dates[ts + j], (j - i) / 252.0, navb[-1], dm, navm[-1],
         np.std(rb - np.nan_to_num(mser[ts:te])) * 100))
    P('  signal q | MDD gross | MDD net | minus benchmark (net) | basket rel. return in benchmark peak->trough (net) | end NAV net | pass')
    for (sn, q), b in series.items():
        r = np.nan_to_num(b[ts:te])
        ng = np.concatenate([[1.0], np.cumprod(1 + r)])
        nn = np.concatenate([[1.0], np.cumprod(1 + r - COST)])
        dg, _, _ = mdd(ng)
        dn, _, _ = mdd(nn)
        rel = nn[j] / nn[i] - navb[j] / navb[i]
        ok = dn <= db
        fails[(sn, q)] += (not ok)
        P('  %s %.1f | %.3f | %.3f | %+.3f | %+.3f | %.3f | %s' % (sn, q, dg, dn, dn - db, rel, nn[-1], 'yes' if ok else 'NO'))
P('\nwindows where the net basket MDD exceeds the benchmark MDD (of 4):')
for k, v in fails.items():
    P('  %s %.1f : %d' % (k[0], k[1], v))
finish()
