# c4_pool.py — 丙四 step 1: the frozen pool U and the amt60 / fcap 200-level ranks on ALL rows of the full panel.
# The source text of design\stageC1\c1_run.py is executed verbatim up to (not including) the model section
# ("MODEL = OUT"), with the same three textual substitutions as c3_scores.py / c2_core.frozen_scores.
# Nothing frozen is changed. Output: cache_c4_pool.npz (rows r0.., r0 = t0 - 25).
import sys, os, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'stageC2'))
import c2_core as Q
from c2_core import C, np, P
ROOT = r'<BLIND_ROOT>'
NPZ = ROOT + r'\data_full\ashare_mainboard_full_20261009.npz'
DATA_END = '2026-10-09'
C.OUTDIR = HERE

src = open(Q.C1, 'rb').read()
txt = src.decode('utf-8-sig').replace('\r\n', '\n')
txt = txt[:txt.index('MODEL = OUT')]
for old, new in ((" and DATA_END <= '2024-06-28'", ''), ('C.OUTDIR = OUT', 'C.OUTDIR = HERE2'), ("start('c1_' + TAG)", "start('c4_' + TAG)")):
    assert txt.count(old) == 1, old
    txt = txt.replace(old, new)
C.NPZ = NPZ
sys.argv = [Q.C1, DATA_END, 'extend', 'pool', 'none', 'none']
ns = {'__name__': 'c1_prefix', 'HERE2': HERE}
exec(compile(txt, Q.C1, 'exec'), ns)
dates, U, t0 = ns['dates'], ns['U'], int(ns['t0'])
r0 = t0 - 25
ra = ns['RK'][ns['NAMES'].index('amt60')][r0:].copy()
rc = ns['RK'][ns['NAMES'].index('fcap')][r0:].copy()
ns.clear()
P('t0 %d (%s); r0 %d (%s); rows %d; pool size on t0 %d' % (t0, dates[t0], r0, dates[r0], U.shape[0], int(U[t0].sum())))
np.savez_compressed(os.path.join(HERE, 'cache_c4_pool.npz'), r0=r0, t0=t0, dates=dates[r0:], U=U[r0:], ra=ra, rc=rc)
P('# peak_ws_MB=%.0f' % C.peak_mb()[0])
C.finish()
