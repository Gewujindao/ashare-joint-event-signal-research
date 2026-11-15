"""Evaluator step 1 check: rows <= 2024-06-28 of the full panel must be bitwise identical to the truncated copy
for all 16 members. Prints only member names, shapes, dtypes and the count of differing bytes (no market values)."""
import os
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
import ctypes
import sys
import time

import numpy as np

TRUNC = r'<BLIND_ROOT>\data\ashare_mainboard_to_20240628.npz'
FULL = r'<BLIND_ROOT>\data_full\ashare_mainboard_full_20261009.npz'

K32 = ctypes.WinDLL('kernel32', use_last_error=True)
K32.GetCurrentProcess.restype = ctypes.c_void_p
K32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
K32.SetPriorityClass(K32.GetCurrentProcess(), 0x00004000)


class PMC(ctypes.Structure):
    _fields_ = [('cb', ctypes.c_ulong), ('PageFaultCount', ctypes.c_ulong),
                ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
                ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t),
                ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t), ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t)]


K32.K32GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC), ctypes.c_ulong]
t0 = time.monotonic()
s = np.load(TRUNC, allow_pickle=False)
l = np.load(FULL, allow_pickle=False)
assert sorted(s.files) == sorted(l.files) and len(s.files) == 16, (s.files, l.files)
nT = s['dates'].size
bad = 0
for k in s.files:
    a, b = s[k], l[k]
    if k == 'codes':
        same = a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()
        ndiff = 0 if same else -1
    else:
        b = np.ascontiguousarray(b[:nT])
        ok_meta = a.shape == b.shape and a.dtype == b.dtype
        if ok_meta:
            ndiff = int((np.frombuffer(a.tobytes(), np.uint8) != np.frombuffer(b.tobytes(), np.uint8)).sum())
        else:
            ndiff = -1
        same = ok_meta and ndiff == 0
    bad += not same
    print('%-14s trunc %s %s | full-prefix %s %s | differing bytes %d | %s' % (
        k, a.shape, a.dtype, b.shape, b.dtype, ndiff, 'IDENTICAL' if same else 'DIFFERENT'), flush=True)
    del a, b
print('full panel: dates %d (%s..%s), prefix rows %d (last prefix date %s), codes %d' % (
    l['dates'].size, l['dates'][0], l['dates'][-1], nT, l['dates'][nT - 1], l['codes'].size))
print('rows after the prefix: %d (first %s)' % (l['dates'].size - nT, l['dates'][nT]))
pmc = PMC()
pmc.cb = ctypes.sizeof(PMC)
K32.K32GetProcessMemoryInfo(K32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
print('seconds %.1f peak_working_set_mb %.0f peak_commit_mb %.0f' % (
    time.monotonic() - t0, pmc.PeakWorkingSetSize / 2 ** 20, pmc.PeakPagefileUsage / 2 ** 20))
print('STEP 1 CHECK ' + ('PASSED: all 16 members bitwise identical on rows <= 2024-06-28' if bad == 0 else 'FAILED: %d members differ' % bad))
sys.exit(0 if bad == 0 else 5)
