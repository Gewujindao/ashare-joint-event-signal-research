r"""build_panel_extension.py - turn data_new\raw\<code>.csv (written by fetch_baostock_range.py) into a panel extension:
an npz with the same 16 members as data\ashare_mainboard_to_20240628.npz, holding ONLY the dates after the old panel's last date.

STATUS   Written offline on 2026-11-09. Trial run the same day on 30 fetched codes against the source panel (--old-panel mainboard_live_20261009.npz):
         CLEAN, 0 of 270 overlap cells differ in the 8 raw arrays, tradable and st; largest qfq ratio spread 7.5e-08. Not yet run on the full batch.
         No code change was needed. Reads local files only, no network.
INPUTS   --raw-dir    data_new\raw                       one CSV per code, 16 columns
         data_new\trade_dates.csv                        trading days of the fetched range (next to the raw folder)
         --panel      data\ashare_mainboard_to_20240628.npz   gives the code list and its order (3,329 codes)
         --old-panel  full-length panel up to 2026-10-09, SUPPLIED BY THE EVALUATOR SIDE. This script does not look for it.
                      Either the re-anchored full panel (output of evaluator\build_full.py) or the source panel works.
ANCHOR   Same rule as the truncated copy (MANIFEST.txt section 5), in two factors per stock:
         s = median over the overlap days of  old close_qfq / new close_qfq       (brings the new download onto the old panel's anchor)
         g = old close_raw[T] / old close_qfq[T], T = last row <= CUT (2024-06-30) with both valid and positive
                                                                                (the data-split step's factor; exactly 1 when the old panel is already re-anchored)
         new qfq arrays = float32( float64(csv value) * s * g ). Raw arrays are not touched.
CHECKS   on the overlap days (fetched days that are already in the old panel):
         - the 8 raw arrays, tradable and st must equal the old panel cell by cell (NaN equals NaN);
         - for the 4 qfq arrays, old / (new * s) must be 1 within --tol on every overlap day;
         every code that fails either check is listed in the report. The npz is written in any case; the report says whether it is clean.
         The convention for suspended days (tradable <- tradestatus == 1, prices kept as Baostock returns them) is not known in advance:
         these checks are what verifies it.
OUTPUT   data_new\ashare_mainboard_ext_<first new date>_<last new date>.npz and data_new\build_report.txt

usage (python = python -I -X utf8):
  python prep\build_panel_extension.py --old-panel <path given by the evaluator side>
"""
import os
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
import argparse
import csv
import ctypes
import sys
import time
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIELDS = ['open_raw', 'high_raw', 'low_raw', 'close_raw', 'preclose_raw', 'open_qfq', 'high_qfq', 'low_qfq',
          'close_qfq', 'volume_shares', 'amount_cny', 'turnover_pct']
QFQ = ['open_qfq', 'high_qfq', 'low_qfq', 'close_qfq']
RAW = [k for k in FIELDS if k not in QFQ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--old-panel', required=True, help='full-length panel up to 2026-10-09, supplied by the evaluator side')
    ap.add_argument('--raw-dir', default=os.path.join(ROOT, 'data_new', 'raw'))
    ap.add_argument('--panel', default=os.path.join(ROOT, 'data', 'ashare_mainboard_to_20240628.npz'), help='truncated copy: code list and order')
    ap.add_argument('--cut', default='2024-06-30', help='anchor cutoff of the truncated copy')
    ap.add_argument('--tol', type=float, default=1e-5, help='largest allowed relative spread of the qfq ratio over the overlap days')
    ap.add_argument('--min-overlap', type=int, default=5, help='least number of overlap days')
    a = ap.parse_args()
    t0 = time.monotonic()
    k32 = ctypes.WinDLL('kernel32', use_last_error=True)
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    k32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    k32.SetPriorityClass(k32.GetCurrentProcess(), 0x00004000)          # BELOW_NORMAL_PRIORITY_CLASS
    base = os.path.dirname(os.path.abspath(a.raw_dir))
    L = []

    def P(s=''):
        print(s, flush=True)
        L.append(s)

    t = np.load(a.panel, allow_pickle=False)
    codes = t['codes']
    t.close()
    N = codes.size
    with open(os.path.join(base, 'trade_dates.csv'), encoding='utf-8') as fh:
        cal = [ln.strip() for ln in fh if ln.strip()]
    z = np.load(a.old_panel, allow_pickle=False)
    od, ocodes = z['dates'], z['codes']
    sel = np.searchsorted(ocodes, codes)
    assert bool((ocodes[sel] == codes).all()), 'the old panel does not contain every code of the truncated copy'
    ov = [d for d in cal if d <= od[-1]]
    new = [d for d in cal if d > od[-1]]
    nov, D = len(ov), len(cal)
    assert nov >= a.min_overlap and len(new) > 0, 'need at least %d overlap days and one new day' % a.min_overlap
    k0 = int(np.searchsorted(od, ov[0]))
    assert [str(d) for d in od[k0:]] == ov, 'the overlap days are not exactly the last days of the old panel'
    missing = [str(c) for c in codes if not os.path.exists(os.path.join(a.raw_dir, str(c) + '.csv'))]
    assert not missing, 'the fetch is not complete: %d codes have no CSV, first %s' % (len(missing), missing[:5])
    P('build_panel_extension.py | old panel %s (%d dates to %s, %d codes) | codes %d | fetched days %d = overlap %d (%s..%s) + new %d (%s..%s)' % (
        os.path.basename(a.old_panel), od.size, od[-1], ocodes.size, N, D, nov, ov[0], ov[-1], len(new), new[0], new[-1]))

    # ---- read the CSVs: raw arrays as float32, qfq arrays as float64 until they are rescaled ----
    X = {k: np.full((D, N), np.nan, np.float64 if k in QFQ else np.float32) for k in FIELDS}
    tradable = np.zeros((D, N), bool)
    st = np.zeros((D, N), bool)
    has_row = np.zeros((D, N), bool)
    row_of = {d: i for i, d in enumerate(cal)}
    for j, c in enumerate(codes):
        with open(os.path.join(a.raw_dir, str(c) + '.csv'), encoding='utf-8', newline='') as fh:
            for rec in csv.DictReader(fh):
                i = row_of[rec['date']]
                has_row[i, j] = True
                for k in FIELDS:
                    if rec[k] != '':
                        X[k][i, j] = float(rec[k])
                tradable[i, j] = rec['tradestatus'] == '1'
                st[i, j] = rec['is_st'] == '1'
    P('rows read: %d | codes with no row at all (header-only files): %d | %.1f s' % (has_row.sum(), int((~has_row.any(0)).sum()), time.monotonic() - t0))
    P('per day: date | codes with a row | tradable')
    for i, d in enumerate(cal):
        P('  %s %s | %5d | %5d' % (d, 'overlap' if i < nov else 'new    ', has_row[i].sum(), tradable[i].sum()))

    # ---- check 1: raw arrays, tradable, st on the overlap days ----
    bad = {}
    P('overlap check, raw members: member | cells | cells that differ | of those, on rows the new data marks not tradable')
    for k in RAW + ['tradable', 'st']:
        old = z[k][k0:][:, sel]
        cur = {'tradable': tradable, 'st': st}[k][:nov] if k in ('tradable', 'st') else X[k][:nov]
        diff = (old != cur) if old.dtype == bool else ~((old == cur) | (np.isnan(old) & np.isnan(cur)))
        P('  %-14s | %d | %d | %d' % (k, diff.size, diff.sum(), (diff & ~tradable[:nov]).sum()))
        for j in np.nonzero(diff.any(axis=0))[0]:
            bad.setdefault(str(codes[j]), []).append('%s differs on %d days' % (k, diff[:, j].sum()))
        del old

    # ---- anchor: s from the overlap ratio of close_qfq, g from the data-split step's rule on the old panel ----
    ocq_full, ocr_full = z['close_qfq'][:, sel], z['close_raw'][:, sel]
    nT = int((od <= a.cut).sum())
    cr, cq = ocr_full[:nT], ocq_full[:nT]
    valid = np.isfinite(cr) & np.isfinite(cq) & (cr > 0) & (cq > 0)
    has = valid.any(axis=0)
    last = nT - 1 - np.argmax(valid[::-1], axis=0)
    cols = np.arange(N)
    g = np.ones(N, np.float64)
    g[has] = cr[last[has], cols[has]].astype(np.float64) / cq[last[has], cols[has]].astype(np.float64)
    del cr, cq, valid, ocr_full
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)             # codes without any overlap row give an all-NaN column: s stays NaN
        ratio = ocq_full[k0:].astype(np.float64) / X['close_qfq'][:nov]
        ratio[~(np.isfinite(ratio) & (ratio > 0))] = np.nan
        del ocq_full
        s = np.nanmedian(ratio, axis=0)
        spread = np.zeros(N)
        for k in QFQ:
            rk = z[k][k0:][:, sel].astype(np.float64) / (X[k][:nov] * s[None, :])
            rk[~np.isfinite(rk)] = np.nan
            spread = np.fmax(spread, np.nanmax(np.abs(rk - 1), axis=0))
    n_ratio = np.isfinite(ratio).sum(axis=0)
    no_anchor = np.isnan(s) & has_row[nov:].any(axis=0)
    for j in np.nonzero(spread > a.tol)[0]:
        bad.setdefault(str(codes[j]), []).append('qfq ratio spread %.2e over %d overlap days (scale %.8f)' % (spread[j], n_ratio[j], s[j]))
    for j in np.nonzero(no_anchor)[0]:
        bad.setdefault(str(codes[j]), []).append('has new rows but no overlap day with a valid close_qfq: qfq left as NaN')
    fin = np.isfinite(s)
    P('overlap check, qfq: codes with a scale %d | scale equal to 1 within 1e-6: %d | largest spread %.3e | codes above tol %.0e: %d | new rows but no anchor: %d' % (
        fin.sum(), (np.abs(s[fin] - 1) <= 1e-6).sum(), spread[fin].max(), a.tol, (spread > a.tol).sum(), no_anchor.sum()))
    P('data-split factor g on the old panel: codes with g != 1: %d (0 means the old panel is already on the truncated copy\'s anchor)' % (g != 1).sum())

    # ---- write the new dates only ----
    out = os.path.join(base, 'ashare_mainboard_ext_%s_%s.npz' % (new[0].replace('-', ''), new[-1].replace('-', '')))
    F = s * g
    arrays = {k: ((X[k][nov:] * F[None, :]) if k in QFQ else X[k][nov:]).astype(np.float32) for k in FIELDS}
    np.savez_compressed(out, dates=np.array(new, dtype=od.dtype), codes=codes, tradable=tradable[nov:], st=st[nov:], **arrays)
    z.close()
    P('written %s | %d dates x %d codes | %.2f MB | members: dates codes tradable st %s' % (out, len(new), N, os.path.getsize(out) / 2 ** 20, ' '.join(FIELDS)))
    P('RESULT: %s' % ('CLEAN - every overlap check passed' if not bad else 'NOT CLEAN - %d codes listed below; do not use the extension before they are resolved' % len(bad)))
    for c in sorted(bad):
        P('  %s : %s' % (c, ' ; '.join(bad[c])))
    P('%.1f s' % (time.monotonic() - t0))
    with open(os.path.join(base, 'build_report.txt'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(L) + '\n')
    return 0 if not bad else 4


if __name__ == '__main__':
    sys.exit(main())
