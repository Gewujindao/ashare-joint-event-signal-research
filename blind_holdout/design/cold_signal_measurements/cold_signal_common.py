# cold_signal_common.py — shared set-up for cold_signal_measurements (training rows only; common.load slices to <= 2022-12-30 and asserts).
import sys
sys.path.insert(0, r'<BLIND_ROOT>\design\base_rates')
import common as C
from common import *
C.OUTDIR = C.ROOT + r'\design\cold_signal_measurements'
SIGS = ('S1_max20', 'S2_vol20', 'S3_comp4')
WIN = (0, 547, 1094, 1641)       # four 551-day windows
WLEN = 551
ASYM = {5: 12.80, 20: 8.23, 40: 6.14, 60: 5.02}   # symmetric-normal reference drift, bp/day (cold_signal_cost_check)


def build(extra=(), keep_raw=False):
    d = load('tradable', 'st', 'close_qfq', 'open_qfq', 'open_raw', 'preclose_raw', 'turnover_pct', *extra)
    assert d['tradable'].shape[0] == 2192 and d['dates'][-1] == '2022-12-30' and bool((d['dates'] <= '2022-12-30').all())
    tr = d['tradable']
    T, N = tr.shape
    has = ~np.isnan(d['close_qfq'])
    first = has.argmax(axis=0)
    age = np.cumsum(tr, axis=0, dtype=np.int32)
    seas = tr & ((first == 0)[None, :] | (age > 750))      # pool: tradable today, >= 750 tradable days before
    del age, has
    Cff = ffill(np.where(tr, d['close_qfq'], np.nan).astype(np.float32))
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
    del rr, ra, ret20, abn, vol20
    U &= np.isfinite(rc)
    OQ = np.where(tr, d['open_qfq'], np.nan).astype(np.float32)
    out = dict(d=d, T=T, N=N, U=U, sig={'S1_max20': rm, 'S2_vol20': rv, 'S3_comp4': rc}, OQ=OQ, seas=seas)
    if keep_raw:
        out['raw'] = {'S1_max20': mx, 'S3_comp4': comp}
    return out


def oo_returns(OQ):
    # Off = forward-filled open; oo[t] = return from open t to open t+1 (0 on suspended days)
    Off = ffill(OQ)
    oo = np.full(OQ.shape, np.nan, np.float32)
    oo[:-1] = Off[1:] / Off[:-1] - 1
    return Off, oo


def basket_series(Z, U, oo):
    # Z, U: membership at close of t-1 (row t-1). Returns b[t], m[t], n[t], sd_in_zone[t] for the open t -> open t+1 return.
    T = Z.shape[0]
    b = np.full(T, np.nan)
    m = np.full(T, np.nan)
    n = np.zeros(T)
    sdz = np.full(T, np.nan)
    sdu = np.full(T, np.nan)
    for t in range(1, T - 1):
        r = oo[t]
        u = U[t - 1] & np.isfinite(r)
        z = Z[t - 1] & u
        k = int(z.sum())
        if k < 5 or u.sum() < 200:
            continue
        ru = r[u].astype(np.float64)
        rz = r[z].astype(np.float64)
        m[t] = ru.mean()
        b[t] = rz.mean()
        n[t] = k
        sdz[t] = rz.std()
        sdu[t] = ru.std()
    return b, m, n, sdz, sdu
