# concat_ext.py - append the panel extension (prep\build_panel_extension.py output) to data_full and check the prefix.
# usage: concat_ext.py OLD_FULL EXT OUT
#   OLD_FULL  data_full\ashare_mainboard_full_20261009.npz (anchored full panel; read only)
#   EXT       data_new\ashare_mainboard_ext_<first>_<last>.npz (new dates only, same anchor, same 16 members and code order)
#   OUT       data_full_ext\ashare_mainboard_full_<last>.npz
# Members are streamed one at a time into a deflated zip (same format as np.savez_compressed).
# Check after writing: every member's rows <= the old last date equal OLD_FULL bit for bit; the new rows equal EXT bit for bit.
# Prints counts only (rows with a record, tradable), never prices.
import os
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_v] = '1'
import sys, time, zipfile, ctypes
import numpy as np
from numpy.lib import format as npf

k32 = ctypes.WinDLL('kernel32')
k32.GetCurrentProcess.restype = ctypes.c_void_p
k32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
k32.SetPriorityClass(k32.GetCurrentProcess(), 0x00004000)
OLD, EXT, OUT = sys.argv[1:4]
t0 = time.monotonic()
zo, ze = np.load(OLD, allow_pickle=False), np.load(EXT, allow_pickle=False)
names = list(zo.files)
assert sorted(names) == sorted(ze.files) and len(names) == 16
od, ed = zo['dates'], ze['dates']
assert od.dtype == ed.dtype and ed[0] > od[-1] and bool((np.diff(ed.astype('U10').astype('datetime64[D]')).astype(int) > 0).all())
assert bool((zo['codes'] == ze['codes']).all())
os.makedirs(os.path.dirname(OUT), exist_ok=True)
assert not os.path.exists(OUT)
tmp = OUT + '.tmp'
with zipfile.ZipFile(tmp, 'w', compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
    for k in names:
        a = zo[k]
        v = a if k == 'codes' else np.concatenate([a, ze[k]], axis=0)
        assert k == 'codes' or v.dtype == a.dtype
        with zf.open(k + '.npy', 'w', force_zip64=True) as fh:
            npf.write_array(fh, np.ascontiguousarray(v), allow_pickle=False)
        del a, v
os.replace(tmp, OUT)
print('written %s | %.1f MB | %.1f s' % (OUT, os.path.getsize(OUT) / 2 ** 20, time.monotonic() - t0))

# ---- check ----
zn = np.load(OUT, allow_pickle=False)
n0 = od.size
bad = 0
for k in names:
    a, b = zo[k], zn[k]
    if k == 'codes':
        same_pre, same_new = bool((a == b).all()), True
    else:
        assert b.shape[0] == n0 + ed.size
        same_pre = a.shape == b[:n0].shape and a.tobytes() == b[:n0].tobytes()
        same_new = ze[k].tobytes() == b[n0:].tobytes()
    bad += (not same_pre) + (not same_new)
    print('  %-14s prefix rows <= %s identical to old full: %s | new rows identical to extension: %s' % (k, od[-1], same_pre, same_new))
    del a, b
d = zn['dates']
print('dates %d (%s..%s) = old %d + new %d (%s..%s) | codes %d' % (d.size, d[0], d[-1], n0, ed.size, ed[0], ed[-1], zn['codes'].size))
tr = zn['tradable'][n0:]
has = np.isfinite(zn['close_raw'][n0:])
for i in range(ed.size):
    print('  %s | codes with a record %d | tradable %d' % (ed[i], int(has[i].sum()), int(tr[i].sum())))
print('CHECK %s | %.1f s' % ('PASSED' if bad == 0 else 'FAILED (%d member checks)' % bad, time.monotonic() - t0))
sys.exit(0 if bad == 0 else 5)
