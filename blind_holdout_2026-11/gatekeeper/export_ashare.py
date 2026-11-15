"""Gatekeeper export: physically truncated, re-anchored copy of the A-share main-board panel.

Reads the source panel read-only, keeps dates <= CUT, drops codes with no tradable row up to CUT,
rescales the four qfq arrays per stock so that close_qfq == close_raw on the stock's last valid row
up to CUT (removes every adjustment factor that belongs to corporate actions after the cutoff),
and writes an .npz (zip of deflated .npy members, same format as numpy.savez_compressed) one array
at a time. Never writes inside the source project.
"""
import os
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
import ctypes
import json
import sys
import time
import zipfile

import numpy as np

SRC = r'<DATA_ROOT>\mainboard_live_20261009.npz'
DEST = r'<BLIND_ROOT>\data\ashare_mainboard_to_20240628.npz'
LOG = r'<BLIND_ROOT>\gatekeeper\export_log.json'
CUT = '2024-06-30'
FIELDS = ['open_raw', 'high_raw', 'low_raw', 'close_raw', 'preclose_raw', 'open_qfq', 'high_qfq', 'low_qfq',
          'close_qfq', 'volume_shares', 'amount_cny', 'turnover_pct']
QFQ = ['open_qfq', 'high_qfq', 'low_qfq', 'close_qfq']
MEM_ABORT_MB = 900.0


class PMC(ctypes.Structure):
    _fields_ = [('cb', ctypes.c_ulong), ('PageFaultCount', ctypes.c_ulong),
                ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
                ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t),
                ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t), ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t)]


K32 = ctypes.WinDLL('kernel32', use_last_error=True)
K32.GetCurrentProcess.restype = ctypes.c_void_p
K32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
K32.K32GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC), ctypes.c_ulong]


def peak_mb():
    pmc = PMC()
    pmc.cb = ctypes.sizeof(PMC)
    assert K32.K32GetProcessMemoryInfo(K32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
    return pmc.PeakWorkingSetSize / 2 ** 20, pmc.PeakPagefileUsage / 2 ** 20


def guard():
    ws, commit = peak_mb()
    if max(ws, commit) > MEM_ABORT_MB:
        print(f'ABORT: peak memory {ws:.0f}/{commit:.0f} MB above {MEM_ABORT_MB} MB', flush=True)
        sys.exit(3)


def put(zf, name, arr):
    arr = np.ascontiguousarray(arr)
    with zf.open(name + '.npy', 'w', force_zip64=True) as fh:
        np.lib.format.write_array(fh, arr, allow_pickle=False)
    return list(arr.shape), str(arr.dtype)


def main():
    t0 = time.monotonic()
    K32.SetPriorityClass(K32.GetCurrentProcess(), 0x00004000)  # BELOW_NORMAL_PRIORITY_CLASS
    print('pid', os.getpid(), flush=True)
    if os.path.exists(DEST):
        print('DEST exists, refusing to overwrite', flush=True)
        sys.exit(2)
    z = np.load(SRC, allow_pickle=False)
    dates, codes = z['dates'], z['codes']
    keep_t = dates.astype(str) <= CUT
    nT = int(keep_t.sum())
    assert keep_t[:nT].all() and not keep_t[nT:].any()  # dates sorted, prefix slice
    tr = z['tradable'][:nT]
    keep_c = tr.any(axis=0)
    nC = int(keep_c.sum())
    tr = np.ascontiguousarray(tr[:, keep_c])
    st = np.ascontiguousarray(z['st'][:nT][:, keep_c])

    # per-stock anchor: last row up to CUT where close_raw and close_qfq are both finite and positive
    cr = z['close_raw'][:nT][:, keep_c]
    cq = z['close_qfq'][:nT][:, keep_c]
    valid = np.isfinite(cr) & np.isfinite(cq) & (cr > 0) & (cq > 0)
    has = valid.any(axis=0)
    last = nT - 1 - np.argmax(valid[::-1], axis=0)
    cols = np.arange(nC)
    f = np.ones(nC, np.float64)
    f[has] = cr[last[has], cols[has]].astype(np.float64) / cq[last[has], cols[has]].astype(np.float64)
    del cr, cq, valid
    guard()

    tmp = DEST + '.tmp'
    members = {}
    with zipfile.ZipFile(tmp, 'w', compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        members['dates'] = put(zf, 'dates', dates[:nT])
        members['codes'] = put(zf, 'codes', codes[keep_c])
        members['tradable'] = put(zf, 'tradable', tr)
        members['st'] = put(zf, 'st', st)
        del tr, st
        for k in FIELDS:
            a = z[k][:nT][:, keep_c]
            assert a.dtype == np.float32
            if k in QFQ:
                a = (a.astype(np.float64) * f[None, :]).astype(np.float32)
            members[k] = put(zf, k, a)
            del a
            guard()
            print('wrote', k, round(time.monotonic() - t0, 1), 's', flush=True)
    z.close()
    os.replace(tmp, DEST)
    ws, commit = peak_mb()
    log = dict(source=os.path.basename(SRC), dest=DEST, cut=CUT, n_dates=nT, n_codes=nC,
               n_codes_source=int(len(codes)), first_date=str(dates[0]), last_date=str(dates[nT - 1]),
               codes_without_anchor=int((~has).sum()), members=members,
               bytes=os.path.getsize(DEST), seconds=round(time.monotonic() - t0, 1),
               peak_working_set_mb=round(ws, 1), peak_commit_mb=round(commit, 1))
    with open(LOG, 'w', encoding='utf-8') as fh:
        json.dump(log, fh, ensure_ascii=False, indent=1)
    print(json.dumps(log, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
