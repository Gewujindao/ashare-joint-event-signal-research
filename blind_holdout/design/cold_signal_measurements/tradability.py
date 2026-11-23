# tradability.py — liquidity bucket, executability and lot effects for cold-zone names (training rows only).
from cold_signal_common import *

start('m6_tradability')
B = build(extra=('amount_cny', 'close_raw'))
d, T, N, U, sig, OQ = B['d'], B['T'], B['N'], B['U'], B['sig'], B['OQ']
del OQ
tr = d['tradable']
yrs = d['years']
sell_ok, buy_ok = exec_maps(d)
amt = np.where(tr & np.isfinite(d['amount_cny']), d['amount_cny'], 0).astype(np.float64)
del d['amount_cny']
cs = np.cumsum(amt, axis=0)
del amt
cn = np.cumsum(tr, axis=0, dtype=np.int32)


def adv(W):
    a = np.full((T, N), np.nan, np.float32)
    a[W:] = (cs[W:] - cs[:-W]) / np.maximum(cn[W:] - cn[:-W], 1)
    return a


adv60 = adv(60)
adv20 = adv(20)
del cs, cn
bucket = np.nan_to_num(np.ceil(pct_rank_rows(adv60, U & np.isfinite(adv60)) * 5), nan=0).astype(np.int8)
lot = (100 * d['close_raw']).astype(np.float32)
PER = {'ALL': np.ones(T, bool), 'P3_20-22': yrs >= 2020}
P('columns: signal zone period | share in adv60 quintile 1(smallest)..5 | buy at t+1 blocked % (of which limit-up open %) | sell at t+1 blocked % |'
  ' pos 10k: unbuyable %, idle cash % | pos 4k: unbuyable %, idle cash % | 1%% of adv20 < 10k %, < 4k %')
for sn in SIGS:
    with np.errstate(invalid='ignore'):
        zones = {'cold10': U & (sig[sn] <= 0.1), 'cold20': U & (sig[sn] <= 0.2), 'cold30': U & (sig[sn] <= 0.3),
                 'hot10': U & (sig[sn] > 0.9), 'pool': U}
    for zn, Z in zones.items():
        for pn, pm_ in PER.items():
            Zp = Z & pm_[:, None]
            tot = Zp.sum()
            sh = [((bucket == k) & Zp).sum() / tot for k in range(1, 6)]
            Z1 = Zp[:-1]
            n1 = Z1.sum()
            bb = (Z1 & ~buy_ok[1:]).sum() / n1
            bl = (Z1 & ~buy_ok[1:] & tr[1:]).sum() / n1
            sb = (Z1 & ~sell_ok[1:]).sum() / n1
            lv = lot[Zp]
            out = []
            for pos in (10000.0, 4000.0):
                ok = lv <= pos
                idle = 1 - np.floor(pos / lv[ok]) * lv[ok] / pos
                out.append('%.2f %.2f' % (100 * (1 - ok.mean()), 100 * idle.mean()))
            a20 = adv20[Zp]
            P('%s %-6s %-8s | %s | %.3f (%.3f) | %.3f | %s | %s | %.3f %.3f' % (
                sn, zn, pn, ' '.join('%.3f' % v for v in sh), 100 * bb, 100 * bl, 100 * sb, out[0], out[1],
                100 * np.mean(0.01 * a20 < 10000), 100 * np.mean(0.01 * a20 < 4000)))
P('\nlot value (100 shares) percentiles among cold20 names of S3, 2020-2022: 10/50/90/99 = %s'
  % ' '.join('%.0f' % v for v in np.percentile(lot[(U & (np.nan_to_num(sig['S3_comp4'], nan=9) <= 0.2)) & (yrs >= 2020)[:, None]], [10, 50, 90, 99])))
finish()
