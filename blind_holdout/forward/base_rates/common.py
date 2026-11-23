# common.py — shared helpers for the base_rates measurement scripts.
# Training segment only: every array is sliced to dates <= 2022-12-30 right after loading.
# Single-threaded numeric libraries.
import os
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_k] = '1'
import sys, time
from math import sqrt
import numpy as np

ROOT = r'<BLIND_ROOT>'
NPZ = ROOT + r'\data\ashare_mainboard_to_20240628.npz'
OUTDIR = ROOT + r'\design\base_rates'
TRAIN_END = '2022-12-30'
_T0 = time.time()

_OUT = None


def start(name):
    global _OUT
    _OUT = open(os.path.join(OUTDIR, 'out_' + name + '.txt'), 'w', encoding='utf-8')
    P('# %s' % name)


def P(*a):
    s = ' '.join(str(x) for x in a)
    print(s)
    _OUT.write(s + '\n')
    _OUT.flush()


def finish():
    P('# done  elapsed_s=%.1f' % (time.time() - _T0))
    _OUT.close()


def load(*names):
    z = np.load(NPZ, allow_pickle=False)
    dates = z['dates']
    n = int((dates <= TRAIN_END).sum())
    assert n == 2192 and dates[n - 1] == TRAIN_END
    d = {'dates': dates[:n].copy(), 'codes': z['codes'].copy()}
    for k in names:
        d[k] = np.ascontiguousarray(z[k][:n])   # slice to training immediately
    z.close()
    d['years'] = np.array([int(s[:4]) for s in d['dates']])
    return d


def periods(years):
    return {'ALL': np.ones(years.shape, bool), 'P1_14-16': years <= 2016,
            'P2_17-19': (years >= 2017) & (years <= 2019), 'P3_20-22': years >= 2020,
            'EX2015': years != 2015}


def seasoned(tradable, has_rec):
    # tradable today and (listed before the data start or >= 250 tradable days since first record)
    first = has_rec.argmax(axis=0)
    age = np.cumsum(tradable, axis=0)
    return tradable & ((first == 0)[None, :] | (age >= 250))


def ffill(a):
    idx = np.where(np.isnan(a), 0, np.arange(a.shape[0], dtype=np.int32)[:, None]).astype(np.int32)
    np.maximum.accumulate(idx, axis=0, out=idx)
    return np.take_along_axis(a, idx, axis=0)


def exec_maps(d):
    # fill rules at the open: no sell at the limit-down open, no buy at the limit-up open (limit 10%, ST 5%)
    tr, st = d['tradable'], d['st']
    o, pc, oq = d['open_raw'], d['preclose_raw'], d['open_qfq']
    lim = np.where(st, 0.05, 0.10).astype(np.float32)
    valid = tr & (o > 0) & (pc > 0) & (oq > 0)
    sell_ok = valid & (o > pc * (1 - lim) + 0.005)
    buy_ok = valid & (o < pc * (1 + lim) - 0.005)
    return sell_ok, buy_ok


def next_true(ok):
    T, N = ok.shape
    out = np.empty((T + 1, N), np.int32)
    cur = np.full(N, 10 ** 6, np.int32)
    out[T] = cur
    for t in range(T - 1, -1, -1):
        cur = np.where(ok[t], t, cur).astype(np.int32)
        out[t] = cur
    return out


def spell_returns(E, L, OQ, nsell, nbuy):
    # E: event at close of t. ideal = open(t+1) -> open(t+1+L), both tradable.
    # realised = first executable sell open s >= t+1 -> first executable buy open b >= s+L (within training).
    T, N = E.shape
    t, i = np.nonzero(E)
    keep = t + 1 + L <= T - 1
    t, i = t[keep], i[keep]
    ideal = OQ[t + 1 + L, i] / OQ[t + 1, i] - 1
    s = nsell[t + 1, i]
    ok = s < T
    b = nbuy[np.minimum(s + L, T), i]
    ok &= b < T
    real = np.full(t.shape, np.nan, np.float32)
    real[ok] = OQ[b[ok], i[ok]] / OQ[s[ok], i[ok]] - 1
    sell_blk = s != t + 1
    buy_blk = ok & (b != s + L)
    return t, i, ideal, real, sell_blk, buy_blk, ok


def cstat(x, t, T):
    # n, mean, median, date-clustered t of the pooled mean
    v = np.isfinite(x)
    x = x[v].astype(np.float64)
    t = t[v]
    n = x.size
    if n < 30:
        return n, np.nan, np.nan, np.nan
    m = x.mean()
    S = np.bincount(t, weights=x - m, minlength=T)
    se = sqrt((S ** 2).sum()) / n
    return n, m, float(np.median(x)), m / se


def limit_events(d, seas):
    st = d['st']
    pc, c, h, l = d['preclose_raw'], d['close_raw'], d['high_raw'], d['low_raw']
    lim = np.where(st, 0.05, 0.10).astype(np.float32)
    up = pc * (1 + lim) - 0.005
    dn = pc * (1 - lim) + 0.005
    base = seas & (pc > 0)
    LU = base & (c >= up)
    LU1 = base & (l >= up)
    LUfail = base & (h >= up) & (c < up)
    LD = base & (c <= dn)
    LD1 = base & (h <= dn)
    LDrec = base & (l <= dn) & (c > dn)
    T, N = LU.shape
    streak = np.zeros((T, N), np.int16)
    run = np.zeros(N, np.int16)
    for t in range(T):
        run = np.where(LU[t], run + 1, 0).astype(np.int16)
        streak[t] = run
    prev = np.zeros((T, N), np.int16)
    prev[1:] = streak[:-1]
    ev = {
        'A1_first_board_non_oneprice': LU & (streak == 1) & ~LU1,
        'A2_second_board_non_oneprice': LU & (streak == 2) & ~LU1,
        'A3_third_board_plus_non_oneprice': LU & (streak >= 3) & ~LU1,
        'A4_oneprice_limit_up': LU1,
        'A5_touched_limit_not_sealed': LUfail,
        'A6_streak_break_prior_2plus_boards': base & ~LU & (prev >= 2),
        'C1_limit_down_close_non_oneprice': LD & ~LD1,
        'C2_oneprice_limit_down': LD1,
        'C3_touched_limit_down_recovered': LDrec,
    }
    return ev


def pct_rank_rows(x, mask):
    T, N = x.shape
    out = np.full((T, N), np.nan, np.float32)
    for t in range(T):
        idx = np.nonzero(mask[t])[0]
        if idx.size < 50:
            continue
        r = np.empty(idx.size, np.float32)
        r[np.argsort(x[t, idx], kind='stable')] = np.arange(1, idx.size + 1, dtype=np.float32)
        out[t, idx] = r / idx.size
    return out


def abn_turnover(turn, tradable, W=60):
    # today's turnover / median of the previous 60 rows (tradable days only, >= 30 valid)
    T, N = turn.shape
    x = np.where(tradable, turn, np.nan).astype(np.float32)
    med = np.full((T, N), np.nan, np.float32)
    for a in range(W, T, 100):
        b = min(a + 100, T)
        win = np.lib.stride_tricks.sliding_window_view(x[a - W:b - 1], W, axis=0)
        s = np.sort(win, axis=2)
        cnt = (~np.isnan(s)).sum(axis=2)
        k = np.clip((cnt - 1) // 2, 0, W - 1)
        m = np.take_along_axis(s, k[:, :, None], axis=2)[:, :, 0]
        m[cnt < 30] = np.nan
        med[a:b] = m
    med[~(med > 0)] = np.nan
    return x / med


def hot_states(d, seas, Cff):
    # joint score = min(pct rank of abnormal turnover, pct rank of past 5-day return); top q per day
    at = abn_turnover(d['turnover_pct'], d['tradable'])
    ret5 = np.full(Cff.shape, np.nan, np.float32)
    ret5[5:] = Cff[5:] / Cff[:-5] - 1
    U = seas & np.isfinite(at) & np.isfinite(ret5)
    ra = pct_rank_rows(at, U)
    rr = pct_rank_rows(ret5, U)
    rj = pct_rank_rows(np.minimum(ra, rr), U)
    st = {}
    for q in (0.01, 0.02, 0.05, 0.10):
        st['HOT_top%g%%' % (q * 100)] = rj > 1 - q
        st['AT_top%g%%' % (q * 100)] = ra > 1 - q
    return st, U
