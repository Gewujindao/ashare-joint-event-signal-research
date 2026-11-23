# check_against_attribution.py — one-time check: rule (ii) on the return_attribution windows with this folder's pool/market (r0 = 225) must reproduce
# the return_attribution readings (V 24.98 / 25.61 / +17.79; SEL 3.36 / 36.54 / +13.10; HOLD 55.30 / 20.76 / +34.19).
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'gbm_account_backtest'))
import account_core as Q
from account_core import C, np, P
C.OUTDIR = HERE
C.start('c4_check')
NPZ = r'<BLIND_ROOT>\data_full\ashare_mainboard_full_20261009.npz'
z = np.load(os.path.join(HERE, 'cache_c4_pool.npz'), allow_pickle=False)
r0, U, rc = int(z['r0']), z['U'], z['rc']
z.close()
zz = np.load(NPZ, allow_pickle=False)
nrow = int((zz['dates'] <= '2026-10-09').sum())
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
s = cs[19:].copy(); s[1:] -= cs[:-20]
n_ = cn[19:].copy(); n_[1:] -= cn[:-20]
del cs, cn
with np.errstate(invalid='ignore', divide='ignore'):
    fcap = s / n_
fcap[n_ < 10] = np.nan
negcap = -fcap[21:].astype(np.float32)
del s, n_, fcap
M = Q.market(NPZ, '2026-10-09', r0)
M['U'] = U
dts = M['dates']
for nm, d0, d1 in (('V', '2020-01-03', '2022-12-30'), ('SEL', '2023-01-03', '2024-06-28'), ('HOLD', '2024-07-01', '2026-10-09')):
    a, b = int(np.searchsorted(dts, d0)), int(np.searchsorted(dts, d1, 'right'))
    B = Q.bench_B(M, a, b, 500000.0, zero_term=True, use_x=True)
    R = Q.run_account(M, a, b, 500000.0, lambda t: np.where(U[t], negcap[t], np.nan), use_x=True)
    r = Q.readings(R['navz'], B['nav'], 500000.0)
    P('%-4s rule ann %.2f maxDD %.2f | B ann %.2f maxDD %.2f | diff %.2f t %.2f' % (nm, r['annS'], r['ddS'], r['annB'], r['ddB'], r['t1'], r['t']))
C.finish()
