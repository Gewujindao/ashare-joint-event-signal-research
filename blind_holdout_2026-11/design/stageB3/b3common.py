# b3common.py — set-up for the one-time selection-segment check (PREDICTIONS_selection.txt, P1-P8).
# Corresponds to design\stageB2\b2common.py: same pool rule, same three signals, same helper functions.
# Only the date range differs. MODE 'dry' = try-out on training rows (signal days 2022-07-01..2022-12-30, data cut
# at 2022-12-30); MODE 'sel' = selection segment (signal days 2023-01-03..2024-06-28, data cut at 2024-06-28).
# Pool age and the forward-filled close are computed on the full history up to the cut, then everything is sliced
# to [first evaluation day - 400 rows, cut] so that memory stays far below 1 GiB.
import sys
sys.path.insert(0, r'<BLIND_ROOT>\design\stage2')
sys.path.insert(0, r'<BLIND_ROOT>\design\stageB2')
import common as C
from common import *
C.OUTDIR = C.ROOT + r'\design\stageB3'
MODE = sys.argv[1] if len(sys.argv) > 1 else 'dry'
assert MODE in ('dry', 'sel')
END = {'dry': '2022-12-30', 'sel': '2024-06-28'}[MODE]
EV0 = {'dry': '2022-07-01', 'sel': '2023-01-03'}[MODE]
NROWS = {'dry': 2192, 'sel': 2551}[MODE]
WARM = 400
SIGS = ('S1_max20', 'S2_vol20', 'S3_comp4')
STAMP_SWITCH = '2023-08-28'


def load_to(*names):
    z = np.load(C.NPZ, allow_pickle=False)
    dates = z['dates']
    n = int((dates <= END).sum())
    assert n == NROWS and dates[n - 1] == END and bool((dates[:n] <= END).all())
    d = {'dates': dates[:n].copy(), 'codes': z['codes'].copy()}
    for k in names:
        d[k] = np.ascontiguousarray(z[k][:n])
    z.close()
    d['years'] = np.array([int(s[:4]) for s in d['dates']])
    return d


def build():
    full = load_to('tradable', 'close_qfq')
    dates = full['dates']
    n = dates.size
    i0 = int(np.nonzero(dates >= EV0)[0][0])
    if MODE == 'sel':
        assert dates[i0] == '2023-01-03' and n - i0 == 359
    a0 = i0 - WARM
    trF = full['tradable']
    has = ~np.isnan(full['close_qfq'])
    first = has.argmax(axis=0)
    age = np.cumsum(trF, axis=0, dtype=np.int32)
    seas = (trF & ((first == 0)[None, :] | (age > 750)))[a0:].copy()
    del age, has
    Cff = ffill(np.where(trF, full['close_qfq'], np.nan).astype(np.float32))[a0:].copy()
    tr = trF[a0:].copy()
    del full, trF
    z = np.load(C.NPZ, allow_pickle=False)
    d = {'dates': dates[a0:n].copy(), 'tradable': tr}
    for k in ('st', 'open_qfq', 'open_raw', 'preclose_raw', 'turnover_pct'):
        d[k] = np.ascontiguousarray(z[k][a0:n])
    z.close()
    T, N = tr.shape
    assert T == WARM + (n - i0) and d['dates'][WARM] >= EV0 and d['dates'][-1] == END
    # ---- from here on identical to b2common.build ----
    r1 = np.full((T, N), np.nan, np.float32)
    r1[1:] = Cff[1:] / Cff[:-1] - 1
    ret20 = np.full((T, N), np.nan, np.float32)
    ret20[20:] = Cff[20:] / Cff[:-20] - 1
    abn = abn_turnover(d['turnover_pct'], tr)
    del d['turnover_pct']
    mx = np.full((T, N), -np.inf, np.float32)
    s1 = np.zeros((T, N), np.float32)
    s2 = np.zeros((T, N), np.float32)
    for j in range(20):
        sh = np.full((T, N), np.nan, np.float32)
        sh[j:] = r1[:T - j]
        mx = np.fmax(mx, sh)
        s1 += sh
        s2 += sh * sh
    vol20 = np.sqrt(np.maximum(s2 / 20 - (s1 / 20) ** 2, 0))
    mx[~np.isfinite(s1)] = np.nan
    del s1, s2, sh, r1, Cff
    U = seas & ~d['st'] & np.isfinite(mx) & np.isfinite(vol20) & np.isfinite(ret20) & np.isfinite(abn)
    rm = pct_rank_rows(mx, U)
    rv = pct_rank_rows(vol20, U)
    rr = pct_rank_rows(ret20, U)
    ra = pct_rank_rows(abn, U)
    comp = ((rm + rv + rr + ra) / 4).astype(np.float32)
    rc = pct_rank_rows(comp, U)
    U &= np.isfinite(rc)
    OQ = np.where(tr, d['open_qfq'], np.nan).astype(np.float32)
    return dict(d=d, T=T, N=N, U=U, sig={'S1_max20': rm, 'S2_vol20': rv, 'S3_comp4': rc}, OQ=OQ)
