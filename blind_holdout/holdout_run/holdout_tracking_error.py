"""Report-only post-processing from the saved daily NAV record of the hold-out window (no market data is read):
annualised tracking error against benchmark B = sd(daily return difference, ddof=1) * sqrt(252)."""
import os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
z = np.load(os.path.join(HERE, 'gbm_account_backtest', 'record_c2_full_HOLD.npz'), allow_pickle=False)
K0 = 500000.0
n = z['dates'].size
print('window %s..%s, %d days' % (z['dates'][0], z['dates'][-1], n))
for lab, ks, kb in (('main = conservative (navz vs navBz)', 'navz', 'navBz'), ('last-close (nav vs navB)', 'nav', 'navB')):
    s, b = z[ks], z[kb]
    d = s / np.r_[K0, s[:-1]] - b / np.r_[1.0, b[:-1]]
    sd = d.std(ddof=1)
    annS, annB = (s[-1] / K0) ** (252.0 / n) - 1, b[-1] ** (252.0 / n) - 1
    t = d.mean() / sd * np.sqrt(n)
    print('%-38s | T1 %.2f (arith %.2f) | t %.2f | TE %%/d %.4f | TE annual %% = sd*sqrt(252) %.2f | prediction-file formula T1*sqrt(n/252)/t %.2f' % (
        lab, (annS - annB) * 100, d.mean() * 25200, t, sd * 100, sd * np.sqrt(252) * 100, (annS - annB) * 100 * np.sqrt(n / 252.0) / t))
