# inspect_unanchored_codes.py - status flags only (no prices) of the 5 codes the extension build flagged NOT CLEAN.
import sys, numpy as np
ROOT = r'<BLIND_ROOT>'
BAD = ['002084', '002155', '002445', '002906', '600984']
z = np.load(ROOT + r'\data_full\ashare_mainboard_full_20261009.npz', allow_pickle=False)
d, c = z['dates'], z['codes']
tr = z['tradable']
cq = z['close_qfq']
cr = z['close_raw']
e = np.load(ROOT + r'\data_new\ashare_mainboard_ext_20261012_20261109.npz', allow_pickle=False)
etr, ecq, ecr = e['tradable'], e['close_qfq'], e['close_raw']
for b in BAD:
    j = int(np.nonzero(c == b)[0][0])
    t = tr[:, j]
    lt = np.nonzero(t)[0]
    fin = np.isfinite(cr[:, j])
    lf = np.nonzero(fin)[0]
    print('%s | data_full: tradable days %d, last tradable %s, last finite close_raw row %s, tradable days in 2026 %d, finite close_raw days after last tradable %d | '
          'ext: tradable %d of %d, finite close_raw %d, finite close_qfq %d' % (
              b, t.sum(), d[lt[-1]] if lt.size else '-', d[lf[-1]] if lf.size else '-', int(t[d >= '2026-01-01'].sum()),
              int(fin[lt[-1] + 1:].sum()) if lt.size else -1, etr[:, j].sum(), etr.shape[0], np.isfinite(ecr[:, j]).sum(), np.isfinite(ecq[:, j]).sum()))
