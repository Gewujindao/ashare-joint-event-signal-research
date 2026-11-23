"""Holdout run, step 1: full-length panel (all dates to 2026-10-09) for the same 3,329 codes as the truncated copy,
qfq arrays re-anchored with the same per-stock factor as the data-split export (same operation order and dtypes:
f = float64(close_raw[T]) / float64(close_qfq[T]), T = last row <= 2024-06-30 with both valid and positive;
a = (float32 -> float64 * f).astype(float32)). Mechanical only: no market value is printed or summarised.
"""
import os
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
import sys
import time
import zipfile

import numpy as np

SRC = r'<DATA_ROOT>\mainboard_live_20261009.npz'
TRUNC = r'<BLIND_ROOT>\data\ashare_mainboard_to_20240628.npz'
DEST = r'<BLIND_ROOT>\data_full\ashare_mainboard_full_20261009.npz'
CUT = '2024-06-30'
FIELDS = ['open_raw', 'high_raw', 'low_raw', 'close_raw', 'preclose_raw', 'open_qfq', 'high_qfq', 'low_qfq',
          'close_qfq', 'volume_shares', 'amount_cny', 'turnover_pct']
QFQ = ['open_qfq', 'high_qfq', 'low_qfq', 'close_qfq']


def put(zf, name, arr):
    arr = np.ascontiguousarray(arr)
    with zf.open(name + '.npy', 'w', force_zip64=True) as fh:
        np.lib.format.write_array(fh, arr, allow_pickle=False)
    return list(arr.shape), str(arr.dtype)


def main():
    t0 = time.monotonic()
    print('pid', os.getpid(), flush=True)
    assert os.path.isdir(os.path.dirname(DEST))
    if os.path.exists(DEST):
        print('DEST exists, refusing to overwrite', flush=True)
        sys.exit(2)
    t = np.load(TRUNC, allow_pickle=False)
    tcodes, tdates = t['codes'], t['dates']
    t.close()
    z = np.load(SRC, allow_pickle=False)
    dates, codes = z['dates'], z['codes']
    keep_t = dates.astype(str) <= CUT
    nT = int(keep_t.sum())
    assert keep_t[:nT].all() and not keep_t[nT:].any()
    assert nT == tdates.size and bool((dates[:nT] == tdates).all())
    assert bool((dates[1:] > dates[:-1]).all())
    keep_c = np.isin(codes, tcodes)
    nC = int(keep_c.sum())
    assert nC == tcodes.size == 3329 and bool((codes[keep_c] == tcodes).all())
    tr_all = z['tradable']
    # the data-split step's own column rule must select the same codes
    assert bool((tr_all[:nT].any(axis=0) == keep_c).all())
    tr = np.ascontiguousarray(tr_all[:, keep_c])
    del tr_all
    st = np.ascontiguousarray(z['st'][:, keep_c])

    # per-stock anchor, exactly as data_split\export_ashare.py (rows up to CUT only)
    cr = z['close_raw'][:nT][:, keep_c]
    cq = z['close_qfq'][:nT][:, keep_c]
    valid = np.isfinite(cr) & np.isfinite(cq) & (cr > 0) & (cq > 0)
    has = valid.any(axis=0)
    last = nT - 1 - np.argmax(valid[::-1], axis=0)
    cols = np.arange(nC)
    f = np.ones(nC, np.float64)
    f[has] = cr[last[has], cols[has]].astype(np.float64) / cq[last[has], cols[has]].astype(np.float64)
    del cr, cq, valid

    tmp = DEST + '.tmp'
    members = {}
    with zipfile.ZipFile(tmp, 'w', compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        members['dates'] = put(zf, 'dates', dates)
        members['codes'] = put(zf, 'codes', codes[keep_c])
        members['tradable'] = put(zf, 'tradable', tr)
        members['st'] = put(zf, 'st', st)
        del tr, st
        for k in FIELDS:
            a = z[k][:, keep_c]
            assert a.dtype == np.float32
            if k in QFQ:
                a = (a.astype(np.float64) * f[None, :]).astype(np.float32)
            members[k] = put(zf, k, a)
            del a
            print('wrote', k, round(time.monotonic() - t0, 1), 's', flush=True)
    z.close()
    os.replace(tmp, DEST)
    print('n_dates', int(dates.size), 'n_dates_prefix', nT, 'n_codes', nC, 'n_codes_source', int(codes.size),
          'first_date', str(dates[0]), 'last_date', str(dates[-1]), 'codes_without_anchor', int((~has).sum()), flush=True)
    print('members', members, flush=True)
    print('seconds', round(time.monotonic() - t0, 1), flush=True)


if __name__ == '__main__':
    main()
