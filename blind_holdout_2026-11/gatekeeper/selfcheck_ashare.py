"""Gatekeeper self-check of the exported copy. Prints structure and consistency figures only (no prices)."""
import os
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
import json
import time

import numpy as np

SRC = r'<DATA_ROOT>\mainboard_live_20261009.npz'
EXP = r'<BLIND_ROOT>\data\ashare_mainboard_to_20240628.npz'
QFQ = ['open_qfq', 'high_qfq', 'low_qfq', 'close_qfq']
SAME = ['open_raw', 'high_raw', 'low_raw', 'close_raw', 'preclose_raw', 'volume_shares', 'amount_cny', 'turnover_pct']

t0 = time.monotonic()
out = {}
e = np.load(EXP, allow_pickle=False)
d = e['dates'].astype(str)
c = e['codes'].astype(str)
nT, nC = len(d), len(c)
out['keys'] = list(e.files)
out['n_keys'] = len(e.files)
out['dates'] = [d[0], d[-1], nT]
out['dates_sorted_unique'] = bool((d[1:] > d[:-1]).all())
out['max_date_le_20240628'] = bool(max(d.tolist()) <= '2024-06-28')
out['n_codes'] = nC
out['days_train_le_20221231'] = int((d <= '2022-12-31').sum())
out['days_select'] = int(((d > '2022-12-31') & (d <= '2024-06-30')).sum())
tr = e['tradable']
out['shape'] = list(tr.shape)
out['all_codes_have_tradable_row'] = bool(tr.any(axis=0).all())
out['tradable_rows'] = int(tr.sum())
out['tradable_rows_train'] = int(tr[d <= '2022-12-31'].sum())
out['dtypes'] = {k: [str(e[k].dtype), list(e[k].shape)] for k in ('dates', 'codes', 'tradable', 'st')}

# anchor check on every stock: close_qfq == close_raw on its last valid row
cr = e['close_raw']
cq = e['close_qfq']
out['dtypes'].update({k: [str(x.dtype), list(x.shape)] for k, x in (('close_raw', cr), ('close_qfq', cq))})
valid = np.isfinite(cr) & np.isfinite(cq) & (cr > 0) & (cq > 0)
has = valid.any(axis=0)
last = nT - 1 - np.argmax(valid[::-1], axis=0)
cols = np.arange(nC)
a = cr[last[has], cols[has]].astype(np.float64)
b = cq[last[has], cols[has]].astype(np.float64)
out['anchor'] = dict(stocks_with_anchor=int(has.sum()), stocks_without_anchor=int((~has).sum()),
                     exactly_equal=int((a == b).sum()), max_rel_diff=float(np.abs(b / a - 1).max()))

# against the source (rows up to the cutoff only)
s = np.load(SRC, allow_pickle=False)
sd = s['dates'].astype(str)
sc = s['codes'].astype(str)
out['dates_match_source_prefix'] = bool((sd[:nT] == d).all())
out['next_source_date_after_cut'] = bool(sd[nT] > '2024-06-30')
pos = {x: i for i, x in enumerate(sc)}
idx_all = np.array([pos[x] for x in c])
stt = s['tradable'][:nT]
kept = np.zeros(len(sc), bool)
kept[idx_all] = True
out['dropped_codes'] = int((~kept).sum())
out['dropped_codes_have_no_tradable_row'] = bool(not stt[:, ~kept].any())
out['tradable_equal_source'] = bool(np.array_equal(stt[:, idx_all], tr))
out['st_equal_source'] = bool(np.array_equal(s['st'][:nT][:, idx_all], e['st']))
del stt

# every stock: exported close_qfq / source close_qfq is one constant per stock (returns unchanged)
scq = s['close_qfq'][:nT][:, idx_all].astype(np.float64)
ecq = cq.astype(np.float64)
out['close_qfq_nan_pattern_equal'] = bool(np.array_equal(np.isfinite(scq), np.isfinite(ecq)))
with np.errstate(divide='ignore', invalid='ignore'):
    ratio = ecq / scq
ratio[~np.isfinite(ratio)] = np.nan
med = np.nanmedian(ratio[:, has], axis=0)
spread = np.nanmax(np.abs(ratio[:, has] / med - 1), axis=0)
out['close_qfq_factor_constant_all_stocks_max_rel_spread'] = float(np.nanmax(spread))
del scq, ecq, ratio

# 5 random stocks: adjacent-day qfq ratios equal to source; other fields bitwise equal
rng = np.random.default_rng(20261109)
pick = np.sort(rng.choice(nC, 5, replace=False))
src_idx = idx_all[pick]
five = dict(codes=[str(x) for x in c[pick]], qfq_adjacent_ratio_max_rel_diff={}, unchanged_fields_equal={})
for k in QFQ:
    x = s[k][:nT][:, src_idx].astype(np.float64)
    y = e[k][:, pick].astype(np.float64)
    assert np.array_equal(np.isfinite(x), np.isfinite(y))
    with np.errstate(divide='ignore', invalid='ignore'):
        rx, ry = x[1:] / x[:-1], y[1:] / y[:-1]
    m = np.isfinite(rx) & np.isfinite(ry)
    five['qfq_adjacent_ratio_max_rel_diff'][k] = float(np.abs(ry[m] / rx[m] - 1).max())
    five.setdefault('ratio_pairs_checked', {})[k] = int(m.sum())
for k in SAME:
    five['unchanged_fields_equal'][k] = bool(np.array_equal(s[k][:nT][:, src_idx], e[k][:, pick], equal_nan=True))
out['five_random_stocks'] = five
out['file_bytes'] = os.path.getsize(EXP)
out['seconds'] = round(time.monotonic() - t0, 1)
print(json.dumps(out, ensure_ascii=False, indent=1))
