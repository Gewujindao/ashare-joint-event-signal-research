# common.py — shared helpers for the stage-2 measurement scripts.
# Training segment only: every array is sliced to dates <= 2022-12-30 right after loading.
# Resource guard: single thread, BelowNormal priority, 1 GiB job-object memory cap, 285 s watchdog.
import os
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_k] = '1'
import sys, time, ctypes, threading
from ctypes import wintypes
from math import sqrt
import numpy as np

ROOT = r'<BLIND_ROOT>'
NPZ = ROOT + r'\data\ashare_mainboard_to_20240628.npz'
OUTDIR = ROOT + r'\design\stage2'
TRAIN_END = '2022-12-30'
_T0 = time.time()

_k32 = ctypes.WinDLL('kernel32', use_last_error=True)
_k32.GetCurrentProcess.restype = wintypes.HANDLE
_k32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
_k32.SetPriorityClass(_k32.GetCurrentProcess(), 0x00004000)  # BELOW_NORMAL


class _MEMSTAT(ctypes.Structure):
    _fields_ = [('dwLength', wintypes.DWORD), ('dwMemoryLoad', wintypes.DWORD)] + \
               [(n, ctypes.c_ulonglong) for n in ('ullTotalPhys', 'ullAvailPhys', 'ullTotalPageFile',
                                                  'ullAvailPageFile', 'ullTotalVirtual', 'ullAvailVirtual',
                                                  'ullAvailExtendedVirtual')]


def avail_mb():
    s = _MEMSTAT()
    s.dwLength = ctypes.sizeof(s)
    _k32.GlobalMemoryStatusEx(ctypes.byref(s))
    return s.ullAvailPhys / 2 ** 20


class _BASIC(ctypes.Structure):
    _fields_ = [('PerProcessUserTimeLimit', ctypes.c_longlong), ('PerJobUserTimeLimit', ctypes.c_longlong),
                ('LimitFlags', wintypes.DWORD), ('MinimumWorkingSetSize', ctypes.c_size_t),
                ('MaximumWorkingSetSize', ctypes.c_size_t), ('ActiveProcessLimit', wintypes.DWORD),
                ('Affinity', ctypes.c_size_t), ('PriorityClass', wintypes.DWORD),
                ('SchedulingClass', wintypes.DWORD)]


class _IO(ctypes.Structure):
    _fields_ = [(n, ctypes.c_ulonglong) for n in ('a', 'b', 'c', 'd', 'e', 'f')]


class _EXT(ctypes.Structure):
    _fields_ = [('Basic', _BASIC), ('Io', _IO), ('ProcessMemoryLimit', ctypes.c_size_t),
                ('JobMemoryLimit', ctypes.c_size_t), ('PeakProcessMemoryUsed', ctypes.c_size_t),
                ('PeakJobMemoryUsed', ctypes.c_size_t)]


def _cap_memory():
    _k32.CreateJobObjectW.restype = wintypes.HANDLE
    _k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
    _k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    _k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    job = _k32.CreateJobObjectW(None, None)
    info = _EXT()
    info.Basic.LimitFlags = 0x100  # JOB_OBJECT_LIMIT_PROCESS_MEMORY
    info.ProcessMemoryLimit = 1 << 30
    a = _k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info))
    b = _k32.AssignProcessToJobObject(job, _k32.GetCurrentProcess())
    return bool(a and b)


class _PMC(ctypes.Structure):
    _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + \
               [(n, ctypes.c_size_t) for n in ('PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage',
                                               'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
                                               'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]


def peak_mb():
    c = _PMC()
    c.cb = ctypes.sizeof(c)
    _k32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
    _k32.K32GetProcessMemoryInfo(_k32.GetCurrentProcess(), ctypes.byref(c), c.cb)
    return c.PeakWorkingSetSize / 2 ** 20, c.PeakPagefileUsage / 2 ** 20


_AVAIL0 = avail_mb()
assert _AVAIL0 >= 2000, 'available memory below 2000 MB'
_CAPPED = _cap_memory()
_wd = threading.Timer(285, lambda: os._exit(3))
_wd.daemon = True
_wd.start()

_OUT = None


def start(name):
    global _OUT
    _OUT = open(os.path.join(OUTDIR, 'out_' + name + '.txt'), 'w', encoding='utf-8')
    P('# %s  avail_MB_at_start=%.0f  job_cap_1GiB=%s' % (name, _AVAIL0, _CAPPED))


def P(*a):
    s = ' '.join(str(x) for x in a)
    print(s)
    _OUT.write(s + '\n')
    _OUT.flush()


def finish():
    ws, pf = peak_mb()
    P('# done  elapsed_s=%.1f  peak_workingset_MB=%.0f  peak_commit_MB=%.0f' % (time.time() - _T0, ws, pf))
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
    # protocol fill rules at the open (MANIFEST section 6)
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
        'A1_首板非一字': LU & (streak == 1) & ~LU1,
        'A2_二连板非一字': LU & (streak == 2) & ~LU1,
        'A3_三连板以上非一字': LU & (streak >= 3) & ~LU1,
        'A4_一字涨停': LU1,
        'A5_触板未封': LUfail,
        'A6_断板_前2板以上': base & ~LU & (prev >= 2),
        'C1_跌停收盘非一字': LD & ~LD1,
        'C2_一字跌停': LD1,
        'C3_触跌停收回': LDrec,
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
