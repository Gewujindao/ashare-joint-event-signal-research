"""Frozen same-event EAR / JOINT gross opportunity comparison.

Only the published quote component of the manual-reversal comparison and its
vector formulas are reused; its memberships, window arrays and economic
summaries are not read.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import re
import time

import numpy as np
import short_joint_h15_new_cohort_opportunity_source_20260908 as src
from compare_short_joint_h15_manual_reversal_opportunity_20260908 import vector_origin, stats, period_groups, ENTRY_CODES

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'output/research_once/short_joint_h15_ear_matched_opportunity_v1_20260908'
UP = ROOT / 'output/research_once/short_joint_h15_ear_matched_control_cohort_v1_20260908'
SCORES = ROOT / 'output/research_once/short_joint_h15_ear_score_cohort_v1_20260908'
CACHE = ROOT / 'output/research_once/short_joint_h15_manual_reversal_opportunity_v1_20260908'
ROUTES = ROOT / 'output/research_once/short_joint_h15_ear_local_input_inventory_v1_20260908'
DOC = ROOT / 'docs/research/short_joint_h15_ear_matched_opportunity_20260908.md'
LOG = ROOT / 'docs/research/continue_pool_c_20260907_2040_risk_log.md'
WORK = 'joint_h15_ear_matched_opportunity'
FIELDS = ('open', 'high', 'low', 'close')
ARMS = ('EAR_FULL', 'EAR_COMMON', 'JOINT_COMMON')
VIEWS = ('ALL_SIGNALS', 'NOMINAL_H15_NONOVERLAP')
CUT = {'TRAIN_COHORT_2021SEP_2022': '2022-12-30', 'SEEN_DEV_2023': '2023-12-29'}
CAP = 1024 ** 3


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def iter_rows(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8-sig') as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def dump(name, value):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def write_rows(name, rows):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'wt', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n')


def epoch(value):
    return int(datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp()) if value else -1


def phase(name, **details):
    row = dict(work_id=WORK, phase=name, at=datetime.now(timezone.utc).isoformat(), **details)
    dump('phase.json', row)
    with (OUT / 'progress.jsonl').open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(json.dumps(row, ensure_ascii=False), flush=True)


def gate():
    # This is the mandatory control-plane gate, not a data revalidation.
    for attempt in range(4):
        try:
            pool = (ROOT / 'docs/research/STRATEGY_RESEARCH_POOL.yaml').read_text(encoding='utf-8-sig')
            break
        except PermissionError:
            if attempt == 3:
                raise
            time.sleep(.1)
    if not re.search(r'^  mode:\s*"RUNNING"', pool.split('operating_policy:', 1)[0], re.M):
        raise RuntimeError('EXECUTION_CONTROL_NOT_RUNNING')
    budget = OUT / 'execution_budget.json'
    if budget.exists() and time.time() > read(budget)['deadline_epoch']:
        raise RuntimeError('65-minute execution limit reached')


def outbytes():
    return sum(path.stat().st_size for path in OUT.rglob('*') if path.is_file())


def load_data(directory='quote_arrays'):
    return {name: np.load(OUT / directory / (name + '.npy'), mmap_mode='r')
            for name in ('qfq', 'available', 'raw_open', 'guard')}


def prepare():
    gate()
    OUT.mkdir(parents=True, exist_ok=True)
    spec = read(UP / 'reader_spec.json')
    native = read(UP / 'membership_seal.json')
    registry = read(native['daily_registry'])['origins']
    full_calendar = read(spec['calendar_receipt'])[spec['calendar_field']]
    cal = [day for day in full_calendar if '2021-09-01' <= day <= '2023-12-29']
    field_names = ['event_id', 'security_key', 'stable_instrument_key', 'source_security_key', 'origin_session', 'role',
                   'scheduled_entry_session', 'nominal_h15_end_session', 'current_eligible', 'eligibility_available_at', 'selection_available_at', 'event_lineage']
    # Project identity-only columns immediately; existing score values are unused.
    events = [{key: row.get(key) for key in field_names} for row in iter_rows(native['full_risk_set'])]
    events.sort(key=lambda row: (row['origin_session'], row['event_id']))
    lookup = {row['event_id']: i for i, row in enumerate(events)}
    common_ids = {row['event_id'] for row in iter_rows(native['common_risk_set'])}
    key_axis = sorted({row['source_security_key'] for row in events})
    key_index = {key: i for i, key in enumerate(key_axis)}
    origin_index = {row['origin_session']: i for i, row in enumerate(registry)}
    n = len(events)
    selected = np.zeros((n, 3, 2), dtype=bool)
    for ai, arm in enumerate(ARMS):
        for vi, field in enumerate(('all_selected_events', 'nominalH15_retained_events')):
            for row in iter_rows(native['arm_paths'][arm][field]):
                selected[lookup[row['event_id']], ai, vi] = True
    common = np.array([row['event_id'] in common_ids for row in events], bool)
    current = np.array([1 if row['current_eligible'] is True else 0 if row['current_eligible'] is False else -1 for row in events], np.int8)
    kk = np.array([key_index[row['source_security_key']] for row in events], np.uint16)
    oi = np.array([origin_index[row['origin_session']] for row in events], np.uint16)
    requested = np.zeros((len(key_axis), len(cal)), bool)
    origins = []
    for i, row in enumerate(registry):
        indices = np.flatnonzero(oi == i)
        day = row['origin_session']
        ep_full = bisect_right(full_calendar, day)
        entry_day = full_calendar[ep_full] if ep_full < len(full_calendar) else None
        ep = bisect_left(cal, entry_day) if entry_day else len(cal)
        cuti = bisect_right(cal, CUT[row['role']]) - 1
        if len(indices):
            # Native rows supply the anchor; full calendar only covers zero-event days.
            entry_day = events[int(indices[0])]['scheduled_entry_session']
            ep = bisect_left(cal, entry_day) if entry_day else len(cal)
        for j in indices:
            if current[j] == 1 and ep <= cuti:
                requested[kk[j], ep:min(ep + 16, cuti + 1)] = True
        origins.append(dict(origin_index=i, decision_session=day, role=row['role'], scheduled_entry_session=entry_day,
            entry_index=ep, cutoff_index=cuti, offset=int(indices[0]) if len(indices) else (origins[-1]['offset'] + origins[-1]['count'] if origins else 0),
            count=len(indices), common_count=int(common[indices].sum()),
            source_prefix_complete=row.get('source_prefix_complete'), unknown_fresh_period_groups=row.get('unknown_fresh_period_groups'),
            missing_score_count=row.get('missing_score_count'),
            selected_counts={arm: {VIEWS[vi]: int(selected[indices, ai, vi].sum()) for vi in range(2)} for ai, arm in enumerate(ARMS)}))
    # The first selected, current-eligible event inside its role window; its security is projected first.
    candidates = [i for i, row in enumerate(events) if selected[i].any() and current[i] == 1
                  and origins[int(oi[i])]['entry_index'] <= origins[int(oi[i])]['cutoff_index']]
    bi = candidates[0]
    b = events[bi]
    first_selected_event = dict(event_index=bi, event_id=b['event_id'], source_security_key=b['source_security_key'])
    np.savez_compressed(OUT / 'sealed_event_index.npz', key_index=kk, origin_index=oi, common=common, selected=selected, current_eligible=current)
    np.savez_compressed(OUT / 'exact_request_mask.npz', requested=requested)
    write_rows('event_identity.jsonl.gz', [dict(event_index=i, **row) for i, row in enumerate(events)])
    write_rows('exact_projection_requests.jsonl.gz', (dict(source_security_key=key_axis[k], exact_dates=[cal[j] for j in np.flatnonzero(requested[k])],
                fields=['qfq open/high/low/close', 'raw open', 'same-source daily conservative entry status']) for k in range(len(key_axis))))
    dump('origin_registry.json', dict(work_id=WORK, keys=key_axis, calendar=cal, origins=origins, first_selected_event=first_selected_event))
    dump('execution_budget.json', dict(started_at=datetime.now(timezone.utc).isoformat(), deadline_epoch=time.time() + 65 * 60,
                                      minutes=65, new_output_bytes_cap=CAP, source_preparation_started_before_actual=True))
    dump('membership_request.json', dict(work_id=WORK, written_at=datetime.now(timezone.utc).isoformat(),
        source110=str(UP / 'membership_seal.json'), source102=str(SCORES / 'terminal.json'), event_count=n,
        common_riskset_count=int(common.sum()), origin_count=len(origins), exact_key_dates=int(requested.sum()),
        selected_counts={arm: {VIEWS[vi]: int(selected[:, ai, vi].sum()) for vi in range(2)} for ai, arm in enumerate(ARMS)},
        calendar_bounds=['2021-09-01', '2023-12-29'], role_cutoffs=CUT,
        request_rule='Each original current-eligible event scheduled e through e+15, clipped at its role cutoff; no re-anchoring',
        field_rule='OHLC per-field availability plus raw_open and published daily conservative guard',
        controls='Full original R_event or G; selected nominal overlap never deduplicates controls. Same-signal-date control views separately identified.',
        no_signal='All native565 daily origins retained. No zero-return/cash imputation.',
        actual_postentry_quotes_read=0, API=0, fit=0, predict=0, TEST=0))
    phase('MEMBERSHIP_REQUEST_WRITTEN', events=n, common_events=int(common.sum()), exact_key_dates=int(requested.sum()),
          selected_counts=read(OUT / 'membership_request.json')['selected_counts'])


def project():
    """Copy only requested cache intersections; project only exact uncached days."""
    gate()
    request = read(OUT / 'membership_request.json')
    meta = read(OUT / 'origin_registry.json')
    cache_meta = read(CACHE / 'origin_registry.json')
    # Only the keys/calendar axis is retained, never old memberships or outcomes.
    cache_keys = {key: i for i, key in enumerate(cache_meta['keys'])}
    cache_days = {day: i for i, day in enumerate(cache_meta['calendar'])}
    del cache_meta
    cache_lineage = {row['security_key']: row for row in read(CACHE / 'quote_projection_receipt.json')['lineage']}
    oldmask = np.load(CACHE / 'exact_request_mask.npz', allow_pickle=False)['requested']
    old = {name: np.load(CACHE / 'quote_arrays' / (name + '.npy'), mmap_mode='r')
           for name in ('qfq', 'available', 'raw_open', 'guard')}
    requested = np.load(OUT / 'exact_request_mask.npz', allow_pickle=False)['requested']
    routes = {row['security_key']: row for row in iter_rows(ROUTES / 'local_source_routes.jsonl')}
    keys, cal = meta['keys'], meta['calendar']
    shape = (len(keys), len(cal))
    directory = OUT / 'quote_arrays'
    directory.mkdir(exist_ok=True)
    data = {
        'qfq': np.lib.format.open_memmap(directory / 'qfq.npy', mode='w+', dtype=np.float64, shape=shape + (4,)),
        'available': np.lib.format.open_memmap(directory / 'available.npy', mode='w+', dtype=np.int64, shape=shape + (4,)),
        'raw_open': np.lib.format.open_memmap(directory / 'raw_open.npy', mode='w+', dtype=np.float64, shape=shape),
        'guard': np.lib.format.open_memmap(directory / 'guard.npy', mode='w+', dtype=np.uint8, shape=shape)}
    data['qfq'][:] = np.nan
    data['available'][:] = -1
    data['raw_open'][:] = np.nan
    data['guard'][:] = 0
    counts = Counter()
    lineage = []
    started = time.monotonic()
    first_key = meta['first_selected_event']['source_security_key']
    order = sorted(range(len(keys)), key=lambda i: (keys[i] != first_key, keys[i]))
    gap_path = OUT / 'exact_quote_gaps.jsonl.gz'
    with gzip.open(gap_path, 'wt', encoding='utf-8') as gap_handle:
        for seq, k in enumerate(order, 1):
            if seq % 40 == 1:
                gate()
            key = keys[k]
            js = np.flatnonzero(requested[k])
            oldk = cache_keys.get(key)
            copied = [int(j) for j in js if oldk is not None and cal[j] in cache_days and oldmask[oldk, cache_days[cal[j]]]]
            copied_set = set(copied)
            remaining = [int(j) for j in js if int(j) not in copied_set]
            if copied:
                oo = np.array([cache_days[cal[j]] for j in copied])
                nn = np.array(copied)
                for name in data:
                    data[name][k, nn] = old[name][oldk, oo]
                counts['reused_key_dates'] += len(copied)
            row = dict(security_key=key, reused_key_dates=len(copied), projected_key_dates=len(remaining),
                       cache_projection_file=str(CACHE / 'quote_projection_receipt.json') if copied else None,
                       cached_lineage=cache_lineage.get(key) if copied else None)
            route = routes.get(key)
            if remaining and route:
                result = src.project_ohlc(route, [cal[j] for j in remaining])
                for j in remaining:
                    bar = result['bars'].get(cal[j], {})
                    for fi, field in enumerate(FIELDS):
                        cell = bar.get('fields', {}).get(field, {})
                        if cell.get('value') is not None:
                            data['qfq'][k, j, fi] = cell['value']
                        data['available'][k, j, fi] = epoch(cell.get('available_at'))
                    raw = bar.get('raw_fields', {}).get('open')
                    if raw is not None:
                        data['raw_open'][k, j] = raw
                    data['guard'][k, j] = {'OK': 1, 'BLOCKED': 2}.get(bar.get('entry_guard', {}).get('status'), 0)
                for gap in result['gaps']:
                    counts['source_gap:' + gap['reason']] += 1
                    gap_handle.write(json.dumps(gap, ensure_ascii=False) + '\n')
                counts['projected_key_dates'] += len(remaining)
                row.update(projected_coordinate_id=result['coordinate_id'], source_csv=route['csv_path'], source_record=route['source_receipt'])
            elif remaining:
                counts['NO_REGISTERED_SOURCE_ROUTE_key_dates'] += len(remaining)
                for j in remaining:
                    gap_handle.write(json.dumps(dict(security_key=key, session=cal[j], reason='NO_REGISTERED_SOURCE_ROUTE')) + '\n')
            lineage.append(row)
            if seq % 200 == 0 or seq == len(order):
                for arr in data.values():
                    arr.flush()
                phase('EXACT_QUOTE_COMPONENT', securities_done=seq, securities_total=len(order), counts=dict(counts), elapsed_seconds=round(time.monotonic() - started, 1))
    for arr in data.values():
        arr.flush()
    dump('quote_projection_summary.json', dict(work_id=WORK, status='COMPLETE_EXACT_CACHE_AND_SOURCE_DIFFERENCE',
        axes=['security_index', 'session_index', 'field_index'], fields=list(FIELDS), lineage=lineage, counts=dict(counts),
        exact_requested_key_dates=request['exact_key_dates'], unknown_cache_cells_not_recovered=True,
        availability='Original per-field Unix seconds UTC; -1 unknown', role_cutoffs=CUT,
        guard_codes={'0': 'UNKNOWN', '1': 'DAILY_CONSERVATIVE_OK', '2': 'DAILY_CONSERVATIVE_BLOCKED'},
        guard_available_at='15:00 posthoc development proxy; not open-time information',
        source_read_scope='The local source parser reads only exact missing key/dates; other CSV rows mechanically date-key scanned',
        API=0, fit=0, predict=0, TEST=0, old_component_retests=0))
    if outbytes() > CAP:
        raise RuntimeError('1 GiB output limit exceeded')
    phase('EXACT_QUOTE_COMPONENT_COMPLETE', counts=dict(counts), bytes=outbytes())


def evaluate():
    gate()
    meta = read(OUT / 'origin_registry.json')
    event_index = np.load(OUT / 'sealed_event_index.npz', allow_pickle=False)
    n = len(event_index['key_index'])
    directory = OUT / 'event_windows'
    directory.mkdir(exist_ok=True)
    arrays = {
        'entry': np.lib.format.open_memmap(directory / 'entry.npy', mode='w+', dtype=np.uint8, shape=(n,)),
        'primary': np.lib.format.open_memmap(directory / 'primary.npy', mode='w+', dtype=np.float64, shape=(n, 3, 3)),
        'peak_day': np.lib.format.open_memmap(directory / 'peak_day.npy', mode='w+', dtype=np.uint8, shape=(n, 3, 2)),
        'gaps': np.lib.format.open_memmap(directory / 'gaps.npy', mode='w+', dtype=np.float64, shape=(n, 3, 2, 3))}
    data = load_data()
    for i, origin in enumerate(meta['origins']):
        if i % 100 == 0:
            gate()
        sl = slice(origin['offset'], origin['offset'] + origin['count'])
        if origin['count']:
            result = vector_origin(event_index['key_index'][sl], origin['entry_index'], origin['cutoff_index'],
                                   event_index['current_eligible'][sl], data, meta['calendar'])
            for name, array in arrays.items():
                array[sl] = result[name]
    for array in arrays.values():
        array.flush()
    dump('event_window_manifest.json', dict(work_id=WORK, status='COMPLETE_FIXED_GROSS_UNFUNDED_EVENT_WINDOWS',
        event_count=n, event_identity='event_identity.jsonl.gz', compact_membership='sealed_event_index.npz', origin_offsets='origin_registry.json',
        entry_codes=ENTRY_CODES, primary_axes=['event', 'H5_H10_H15', 'MFE_DAY1_TO_H_MFE_T1_DAY2_TO_H_MAE_DAY1_TO_H'],
        peak_day_axes=['event', 'H5_H10_H15', 'MFE_DAY1_TO_H_MFE_T1_DAY2_TO_H'],
        gap_axes=['event', 'H5_H10_H15', 'MFE_DAY1_TO_H_MFE_T1_DAY2_TO_H', 'peak_close_gap_next_open_gap_prior_session_adverse'],
        missing='NaN is unknown; peak_day=0 unavailable; no signal or missing quote never becomes zero return',
        entry='EXACT_SCHEDULED_OPEN_WITH_RAW_AND_CONSERVATIVE_DAILY_STATUS_PROXY_NO_RETRY',
        timing='day1 high not sellable under T+1; day15 high after cap open is diagnostic only',
        prior_adverse='Strictly earlier complete sessions only; peak-day low/high order unknown',
        limits='Gross A/B; no executable exit, net profit C, or funded wealth D', role_cutoffs=CUT,
        old_formula_retests=0, API=0, fit=0, predict=0, TEST=0))
    phase('FIXED_EVENT_WINDOWS_COMPLETE', events=n, bytes=outbytes())


METRICS = tuple(src.CHANNELS) + tuple(channel + '__' + field for channel in src.CHANNELS[:2]
    for field in ('peak_close_gap', 'next_open_gap', 'prior_session_adverse_before_peak'))
POPULATIONS = ARMS + ('R_EVENT', 'G_COMMON')
PAIRS = (('EAR_FULL', 'R_EVENT'), ('EAR_COMMON', 'G_COMMON'), ('JOINT_COMMON', 'G_COMMON'), ('EAR_COMMON', 'JOINT_COMMON'))


def summary_stat(values, original_n=None, thresholds=False):
    result = stats(values, thresholds)
    if thresholds and 'thresholds' not in result:
        result['thresholds'] = {name: dict(count=0, denominator=0, fraction=None) for name, _, _ in src.THRESHOLDS}
    if original_n is not None:
        result.update(original_n=int(original_n), missing_n=int(original_n - result['n']),
            observed_fraction=result['n'] / original_n if original_n else None,
            full_roster_mean=result['mean'] if original_n and result['n'] == original_n else None)
    return result


def summarize():
    gate()
    read(OUT / 'event_window_manifest.json')
    if (OUT / 'summary_started.json').exists():
        raise RuntimeError('SUMMARY_ALREADY_STARTED_NO_REPEAT')
    dump('summary_started.json', dict(at=datetime.now(timezone.utc).isoformat()))
    meta = read(OUT / 'origin_registry.json')
    origins = meta['origins']
    z = np.load(OUT / 'sealed_event_index.npz', allow_pickle=False)
    oi, selected, common = z['origin_index'], z['selected'], z['common']
    a = {name: np.load(OUT / 'event_windows' / (name + '.npy'), mmap_mode='r') for name in ('entry', 'primary', 'peak_day', 'gaps')}
    # Compact originals are the only economic source. No quotes are reread and no evaluation functions rerun.
    values = np.concatenate((a['primary'], a['gaps'].reshape(len(oi), 3, 6)), axis=2)
    origin_means = np.full((len(origins), 2, 5, 3, 9), np.nan)
    origin_medians = np.full_like(origin_means, np.nan)
    origin_original = np.zeros((len(origins), 2, 5), np.int32)
    origin_known = np.zeros((len(origins), 2, 5, 3, 9), np.int32)
    with gzip.open(OUT / 'per_origin_population_statistics.jsonl.gz', 'wt', encoding='utf-8') as handle:
        for i, origin in enumerate(origins):
            if i % 100 == 0:
                gate()
            sl = slice(origin['offset'], origin['offset'] + origin['count'])
            for vi, view in enumerate(VIEWS):
                masks = [selected[sl, ai, vi] for ai in range(3)] + [np.ones(origin['count'], bool), common[sl]]
                for pi, population in enumerate(POPULATIONS):
                    mask = masks[pi]
                    original = int(mask.sum())
                    origin_original[i, vi, pi] = original
                    for hi, horizon in enumerate(src.HORIZONS):
                        metric = {}
                        for mi, name in enumerate(METRICS):
                            stat = summary_stat(values[sl][mask, hi, mi], original, thresholds=mi < 3)
                            metric[name] = stat
                            if stat['mean'] is not None:
                                origin_means[i, vi, pi, hi, mi] = stat['mean']
                                origin_medians[i, vi, pi, hi, mi] = stat['median']
                            origin_known[i, vi, pi, hi, mi] = stat['n']
                        handle.write(json.dumps(dict(origin_session=origin['decision_session'], role=origin['role'], view=view,
                            population=population, horizon=horizon, original_n=original, metrics=metric), ensure_ascii=False, separators=(',', ':')) + '\n')
    np.savez_compressed(OUT / 'per_origin_statistics.npz', mean=origin_means, median=origin_medians, original_n=origin_original, known_n=origin_known)
    paired_rows = []
    with gzip.open(OUT / 'per_origin_paired_metrics.jsonl.gz', 'wt', encoding='utf-8') as handle:
        for i, origin in enumerate(origins):
            for vi, view in enumerate(VIEWS):
                for left, right in PAIRS:
                    li, ri = POPULATIONS.index(left), POPULATIONS.index(right)
                    for hi, horizon in enumerate(src.HORIZONS):
                        metric = {}
                        for mi, name in enumerate(METRICS):
                            l, r = origin_means[i, vi, li, hi, mi], origin_means[i, vi, ri, hi, mi]
                            known = bool(np.isfinite(l) and np.isfinite(r))
                            metric[name] = dict(left_mean=float(l) if np.isfinite(l) else None, right_mean=float(r) if np.isfinite(r) else None,
                                left_known_n=int(origin_known[i, vi, li, hi, mi]), right_known_n=int(origin_known[i, vi, ri, hi, mi]),
                                paired=known, left_minus_right=float(l - r) if known else None)
                        handle.write(json.dumps(dict(origin_session=origin['decision_session'], role=origin['role'], view=view,
                            left=left, right=right, horizon=horizon, left_original_n=int(origin_original[i, vi, li]),
                            right_original_n=int(origin_original[i, vi, ri]), metrics=metric), ensure_ascii=False, separators=(',', ':')) + '\n')
    groups = period_groups(origins)
    pooled_rows = []
    equal_rows = []
    censor = np.array([[r['entry_index'] + horizon - 1 > r['cutoff_index'] for horizon in src.HORIZONS] for r in origins])
    for gi, group in enumerate(groups):
        gate()
        ids = np.array(group['origin_indices'], int)
        active = np.zeros(len(origins), bool)
        active[ids] = True
        group_meta = {key: value for key, value in group.items() if key != 'origin_indices'}
        for vi, view in enumerate(VIEWS):
            base_mask = active[oi]
            populations = [(arm, base_mask & selected[:, ai, vi], None) for ai, arm in enumerate(ARMS)]
            populations += [('R_EVENT', base_mask, None), ('G_COMMON', base_mask & common, None)]
            for ai, arm in enumerate(ARMS):
                at_signal = active & (origin_original[:, vi, ai] > 0)
                mask = at_signal[oi] & (np.ones(len(oi), bool) if ai == 0 else common)
                populations.append((('R_EVENT' if ai == 0 else 'G_COMMON') + '_AT_' + arm + '_SIGNAL_ORIGINS', mask, arm))
            for population, mask, matched_arm in populations:
                indices = np.flatnonzero(mask)
                entry = a['entry'][indices]
                for hi, horizon in enumerate(src.HORIZONS):
                    completed = np.isfinite(a['primary'][indices, hi, :]).all(axis=1)
                    censored = censor[oi[indices], hi]
                    counts = dict(original_n=len(indices), observed_entry_n=int(np.sum(entry == 1)),
                        entry_status_counts={ENTRY_CODES[int(code)]: int(np.sum(entry == code)) for code in np.unique(entry)},
                        right_censored_window_n=int(censored.sum()), observed_entry_right_censored_window_n=int(np.sum((entry == 1) & censored)),
                        mature_observed_entry_incomplete_channels_n=int(np.sum((entry == 1) & ~censored & ~completed)),
                        complete_all_primary_channels_n=int(completed.sum()))
                    metrics = {name: summary_stat(values[indices, hi, mi], len(indices), thresholds=mi < 3) for mi, name in enumerate(METRICS)}
                    peak_timing = {}
                    for ch, channel in enumerate(src.CHANNELS[:2]):
                        days = a['peak_day'][indices, hi, ch]
                        peak_timing[channel] = dict(original_n=len(indices), known_n=int(np.sum(days > 0)),
                            first_peak_holding_day_counts={str(day): int(np.sum(days == day)) for day in range(1, horizon + 1)},
                            peak_after_day15_cap_open_diagnostic_n=int(np.sum(days == 15)))
                    pooled_rows.append(dict(**group_meta, view=view, population=population, horizon=horizon,
                        original_registered_origins=len(ids), origins_with_events=len(np.unique(oi[indices])), matched_arm=matched_arm,
                        counts=counts, metrics=metrics, peak_timing=peak_timing, aggregation='POOLED_EVENT_OWN_CHANNEL_OBSERVED'))
            for left, right in PAIRS:
                li, ri = POPULATIONS.index(left), POPULATIONS.index(right)
                left_original = origin_original[ids, vi, li]
                right_original = origin_original[ids, vi, ri]
                for hi, horizon in enumerate(src.HORIZONS):
                    metrics = {}
                    for mi, name in enumerate(METRICS):
                        lv = origin_means[ids, vi, li, hi, mi]
                        rv = origin_means[ids, vi, ri, hi, mi]
                        both = np.isfinite(lv) & np.isfinite(rv)
                        lm = origin_medians[ids, vi, li, hi, mi]
                        rm = origin_medians[ids, vi, ri, hi, mi]
                        metrics[name] = dict(paired_common_origin_n=int(both.sum()),
                            left_no_signal_or_empty_n=int(np.sum(left_original == 0)), right_no_signal_or_empty_n=int(np.sum(right_original == 0)),
                            left_signaled_but_metric_unknown_n=int(np.sum((left_original > 0) & ~np.isfinite(lv))),
                            right_nonempty_but_metric_unknown_n=int(np.sum((right_original > 0) & ~np.isfinite(rv))),
                            both_signaled_n=int(np.sum((left_original > 0) & (right_original > 0))),
                            left_known_right_unknown_or_empty_n=int(np.sum(np.isfinite(lv) & ~np.isfinite(rv))),
                            right_known_left_unknown_or_empty_n=int(np.sum(np.isfinite(rv) & ~np.isfinite(lv))),
                            left_mean_on_common_origins=summary_stat(lv[both]), right_mean_on_common_origins=summary_stat(rv[both]),
                            left_minus_right_mean=summary_stat(lv[both] - rv[both]),
                            left_minus_right_median=summary_stat(lm[both] - rm[both]))
                    equal_rows.append(dict(**group_meta, view=view, left=left, right=right, horizon=horizon,
                        original_registered_origins=len(ids), metrics=metrics,
                        aggregation='ONE_EQUAL_WEIGHT_PER_PAIRED_OBSERVABLE_ORIGIN; NO_ZERO_CASH_OR_MISSING_IMPUTATION'))
        if (gi + 1) % 10 == 0:
            phase('FIXED_PERIOD_AGGREGATION', period_groups_done=gi + 1, period_groups_total=len(groups))
    dump('summary.json', dict(work_id=WORK, status='COMPLETE_FIXED_SAME_EVENT_A_B_COMPARISON',
        pooled_rows=pooled_rows, equal_origin_rows=equal_rows, horizons=list(src.HORIZONS), metric_names=list(METRICS),
        role_cutoffs=CUT, primary_channels=list(src.CHANNELS),
        controls='R_EVENT and G_COMMON retain every native event. AT_* controls retain full riskset at the selected arm signal origins for that view.',
        cross_role_periods='Descriptive calendar diagnostics only; TRAIN and SEEN_DEV primary summaries remain separate',
        endpoint='2023-12-31 calendar month; final observed session2023-12-29; final anchored block incomplete',
        limits=['gross unfunded opportunity A/B only', 'not C netprofit or D wealth', 'day1 HIGH cannot be sold under T+1',
                'day15 postcap HIGH diagnostic only', 'historical vintage/current source proxy limits',
                'EAR_COMMON is explicit intersection; EAR_FULL never silently replaced', 'common G includes old JOINT feature/5000 affordability restriction',
                'no signal and missing fields remain unknown; observed channel conditional means',
                'shared origins and overlapping windows create dependence; no independent-event significance claim'],
        API=0, fit=0, predict=0, TEST=0, orders=0, parameter_changes=0))
    dump('summary_period_registry.json', dict(periods=groups, metrics=list(METRICS), populations=list(POPULATIONS),
        per_origin_array_axes=['origin', 'ALL_SIGNALS_NOMINAL_H15_NONOVERLAP', 'EAR_FULL_EAR_COMMON_JOINT_COMMON_R_EVENT_G_COMMON', 'H5_H10_H15', 'metric_names']))
    if outbytes() > CAP:
        raise RuntimeError('1 GiB output limit exceeded')
    phase('FIXED_SUMMARY_COMPLETE', pooled_rows=len(pooled_rows), equal_origin_rows=len(equal_rows), bytes=outbytes())


def pct(value):
    return 'unknown' if value is None else f'{100 * value:.3f}'


def finalize():
    """Publish the fixed aggregate report; no economic or source replay."""
    gate()
    summary = read(OUT / 'summary.json')
    request = read(OUT / 'membership_request.json')
    projection = read(OUT / 'quote_projection_summary.json')
    assessment = dict(work_id=WORK, decision='EAR_HIGHER_UPSIDE_DEEPER_DRAWDOWN', viable_claim='DESCRIPTIVE_COMPARISON',
        named_version='EAR_FULL_COMMON_JOINT_MATCHED_H15_OPPORTUNITY_V1',
        named_version_disposition='RETAIN_DESCRIPTIVE_ENTRY_EVIDENCE_ONLY',
        supported='EAR shows positive matched upside-opportunity increments at H5/H10/H15 in TRAIN and SEEN_DEV, alongside deeper MAE.',
        unresolved='Causal exit realization, fees, funding and the upside/downside tradeoff remain unresolved; observed maxima are not returns.',
        recent='2023Q4 EAR_COMMON-minus-JOINT_COMMON H15 T1 upside increment is negative and MAE is deeper; complete fixed periods remain disclosed.',
        decision_rule='No added numeric gate, no requirement of every-block/all-metric dominance, no pooled rescue or automatic STOP from mixed evidence.',
        next_action_owner='',
        next_action='Possible follow-up: an exit study on the original policy or on each arm\'s own selections.',
        family_scope='Only this named EAR entry implementation and explicit common-cohort comparison; no EAR/manual/ML family verdict')
    dump('retention_assessment.json', assessment)
    limitations = dict(work_id=WORK,
        research_layers={'A': 'Fixed future upside opportunity, not a sell instruction', 'B': 'Path downside/giveback diagnostics',
                         'C': 'NOT_COMPUTED executable net profit', 'D': 'NOT_COMPUTED funded account wealth'},
        observations=['Exact scheduled entry only; no later executable-open search.',
            'Published daily status at15:00 is a posthoc development entry guard, not09:30 information or real fill.',
            'Original per-field availability and role cutoff are retained; TRAIN future dates past2022Dec30 excluded.',
            'Day1 HIGH is not sellable under China T+1. Day2+ HIGH is an observational upper bound, not an achievable fill.',
            'Day15 HIGH occurs after a nominal cap open; all postcap day15 peak counts remain diagnostic only.',
            'MAE can occur after the peak. Prior-session adverse excludes the peak session; peak-day LOW/HIGH order is unknown.',
            'Peak-close and next-open gaps divide by entry open; negative gap is possible. Next-open channel may be unknown independently.',
            'Original signals and exact native nominal-overlap memberships are retained; source gaps/censoring never delete the signal.',
            'Mean/quantiles are own-channel observed conditional statistics. full_roster_mean is null when any original row is missing.',
            'No-signal or unavailable metric is not a zero-return cash event. Equal-origin comparisons only pair two observable origin means.',
            'R_EVENT and G_COMMON unconditioned populations are distinct from AT_ARM_SIGNAL_ORIGINS matched pooled controls.',
            'Common G inherits the old JOINT feature eligibility and origin-close5000 affordability coverage restriction; EAR_FULL does not.',
            'JOINT is the frozen original W score, not an H15 probability. No fitting, new scoring, threshold tuning or reselection occurred.',
            'Provider qfq/raw and current-file historical-vintage limitations persist; no production PIT or live-candidate claim.',
            'Event windows overlap and share announcement origins; no independent-event significance or family-wide claim.',
            'First anchored block2021Sep2022Aug and second2022Sep2023Aug are complete calendar windows; third is partial through2023Dec.',
            'Last complete12m windows2022/2023 and common24m2022Jan2023Dec are reported; cross-role blocks are descriptive only.',
            'This sample starts2021Sep1. The2021 year/2021Q3 periods are partial; no earlier performance is inferred.'],
        formal_upstream={'102': str(SCORES / 'terminal.json'), '110': str(UP / 'terminal.json'),
                         '105': str(ROOT / 'output/research_once/short_joint_h15_new_cohort_opportunity_source_v1_20260908/terminal.json'),
                         '107_component_only': str(CACHE / 'quote_component_publication.json')},
        effects={'API': 0, 'fit': 0, 'predict': 0, 'TEST': 0, 'orders': 0, '107_economic_reads': 0, 'old_fixture_retests': 0})
    dump('limitations.json', limitations)
    lines = ['# EAR same-event three-arm comparison of opportunity and downside at H5/H10/H15', '',
        'For the current named EAR entry version, observed upside opportunity is higher in same-day pairings within TRAIN and SEEN_DEV, while the drawdown is usually deeper.', '',
        'The common-pool comparison is especially weak in 2023Q4: at H15, EAR_COMMON versus JOINT_COMMON has a day2+ MFE mean increment of -0.284 percentage points and a MAE mean difference of -4.159 percentage points, on only 14 pairable origins. Full quarterly, monthly, fixed 12-month and recent-window results are all stored; annual or 24-month pooled results do not override this recent weakness.', '',
        '## Populations, dates and definitions', '',
        f"The full R_event of the upstream event set has {request['event_count']:,} events and the common G has {request['common_riskset_count']:,}; all {request['origin_count']} calendar origins keep their no-signal and input-gap records. ALL_SIGNALS counts for EAR_FULL / EAR_COMMON / JOINT_COMMON are 697 / 648 / 1086, and nominalH15 counts are 697 / 647 / 1081.", '',
        'EAR_FULL is compared with the full same-day R_event; EAR_COMMON and JOINT_COMMON are compared with the same G. EAR_COMMON uses the thresholds from the complete prior quarter in the original EAR scoring step, without retraining or changing thresholds on G. JOINT_COMMON is the top 4 by the frozen original W probability ranking, not an H15 probability; G additionally carries its old feature eligibility and the 5000-yuan origin-close affordability coverage. EAR_FULL is not silently replaced by this intersection.', '',
        'The entry anchor is strictly the open of the planned next trading day after the signal; the anchor is not changed because of later suspension, missing prices or price moves. The gross unfunded A/B uses the original provider qfq, raw open and the 15:00 conservative daily guard. This guard is an ex post development filter, not information known at 09:30 and not an actual fill.', '',
        'TRAIN_COHORT_2021SEP_2022 reads only through 2022-12-30; SEEN_DEV_2023 through 2023-12-29. The first 2021 year/Q3 is a partial window; the third anchored 12-month block is also incomplete. End-of-day source availability, unknowns and the segment cutoff all pass through unchanged.', '',
        f"Of the 330,994 exact code-dates, {projection['counts'].get('reused_key_dates', 0):,} directly reuse existing formal quote components, and {projection['counts'].get('projected_key_dates', 0)} unrequested dates are projected from local sources. One end-of-day entry status gap on a newly added date is kept as is. Unknowns already in the cache are not re-fetched from source; existing economic files are not read.", '',
        '## Main comparison: same day, equal weight per origin', '',
        'All units are percentage points. Δ is left arm minus right arm; a more negative MAE means a deeper drawdown. N is the number of origins at which both sides are observable for the channel, each origin carrying one weight; it is not a pooled per-stock mean difference. day2+ MFE remains an upper bound observed from future HIGH, not a realisable sell return.', '',
        '| Role | View | Left − right | H | N day1/day2+/MAE | Δday1 MFE | Δday2+ MFE | ΔMAE |',
        '|---|---|---|---:|---:|---:|---:|---:|']
    for row in summary['equal_origin_rows']:
        if row['dimension'] == 'role':
            m = row['metrics']
            names = src.CHANNELS
            ns = '/'.join(str(m[key]['paired_common_origin_n']) for key in names)
            deltas = [pct(m[key]['left_minus_right_mean']['mean']) for key in names]
            lines.append(f"| {row['period']} | {row['view']} | {row['left']} − {row['right']} | {row['horizon']} | {ns} | " + ' | '.join(deltas) + ' |')
    lines += ['', '## Pooled distributions of the three selection arms by role', '',
        'In each row n/N is the known count for the channel / the original selected count. Percentages are pre-cost opportunity or downside observations. Full controls, all quantiles, both views and all time windows are in summary.json and the full tables.', '',
        '| Role | View | Arm | H | Channel | n/N | mean% | median% | q10% | q90% | >0 | ≥3% | ≥5% | ≥10% |',
        '|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in summary['pooled_rows']:
        if row['dimension'] == 'role' and row['population'] in ARMS:
            for channel in src.CHANNELS:
                s = row['metrics'][channel]
                hits = [str(s['thresholds'][key]['count']) for key in ('gt0', 'ge3pct', 'ge5pct', 'ge10pct')]
                lines.append(f"| {row['period']} | {row['view']} | {row['population']} | {row['horizon']} | {channel} | {s['n']}/{s['original_n']} | " +
                             ' | '.join(pct(s[key]) for key in ('mean', 'median', 'q10', 'q90')) + ' | ' + ' | '.join(hits) + ' |')
    lines += ['', '## Fixed quarterly and 12/24-month comparisons', '',
        'H15 ALL_SIGNALS is listed below; H5/H10, the nominal view, months and all metrics are kept, and the full tables contain every fixed cell. 2021Q3 covers only September 2021, and BLOCK_3_PARTIAL only 2023, months 9 to 12; cross-role windows do not change the TRAIN/SEEN_DEV labels.', '',
        '| Period | Pair | N | Δday2+ MFE pp | ΔMAE pp |', '|---|---|---:|---:|---:|']
    for row in summary['equal_origin_rows']:
        if row['dimension'] in ('quarter', 'anchored_12m', 'recent_12m', 'recent_24m') and row['view'] == 'ALL_SIGNALS' and row['horizon'] == 15 and (row['left'] == 'EAR_FULL' or row['right'] == 'JOINT_COMMON'):
            m = row['metrics']
            lines.append(f"| {row['period']} | {row['left']} − {row['right']} | {m['MFE_T1_DAY2_TO_H']['paired_common_origin_n']} | {pct(m['MFE_T1_DAY2_TO_H']['left_minus_right_mean']['mean'])} | {pct(m['MAE_DAY1_TO_H']['left_minus_right_mean']['mean'])} |")
    lines += ['', '## Peaks, giveback, gaps and reusable outputs', '',
        'peak_timing in summary.json counts the first-peak holding day for each channel over both the original and the observable denominators; tied peaks take the earliest date. The secondary metrics keep the quantiles and missing counts of peak_close_gap, next_open_gap and prior_session_adverse_before_peak. Pre-peak adverse uses only the LOW of sessions fully completed before the peak and does not assume that the peak-day LOW precedes the HIGH. All gaps are divided by the entry open.', '',
        'The day1 HIGH cannot be sold under T+1; the H15 day15 HIGH comes after the horizon open and serves only as a post-cap diagnostic. The next-open gap may extend beyond the original H window but is subject to the same segment cutoff; a tail that cannot be observed is unknown.', '',
        'The pooled R_EVENT / G_COMMON tables keep the complete original riskset; the AT_*_SIGNAL_ORIGINS tables further restrict to dates on which the arm actually has a signal in that view, still without separately deduplicating control securities. EAR_COMMON-minus-JOINT_COMMON is paired only on dates where the metric is observable on both sides. Counts of origins with no signal, one-sided observability, or a selection with an unknown metric are all in equal_origin_rows and are not filled in as zero cash returns.', '',
        'The file index is publication_pointer.json; summary.json contains 2352 rows of pooled population/window summaries and 1176 rows of equal-origin paired summaries, covering 49 predefined role/year/quarter/month/12-month/24-month window groups. primary_all_periods.md and paired_all_periods.md give readable full tables of the main metrics. event_identity.jsonl.gz / sealed_event_index.npz / event_windows keep the exact events, membership, entry status and the values of the three windows; per_origin_statistics.npz and two per_origin JSONL files hold the pairing basis.', '', '']
    (OUT / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    with (OUT / 'primary_all_periods.md').open('w', encoding='utf-8') as handle:
        handle.write('# All fixed cells: pooled primary\n\nMean/quantiles are percentages; n/N is known/original per channel; thresholds are event counts.\n\n| Dimension | Period | View | Population | H | Metric | n/N | mean | median | q10 | q90 | >0 | ≥3 | ≥5 | ≥10 |\n|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n')
        for row in summary['pooled_rows']:
            for key in src.CHANNELS:
                s = row['metrics'][key]
                handle.write(f"| {row['dimension']} | {row['period']} | {row['view']} | {row['population']} | {row['horizon']} | {key} | {s['n']}/{s['original_n']} | " +
                    ' | '.join(pct(s[name]) for name in ('mean', 'median', 'q10', 'q90')) + ' | ' +
                    ' | '.join(str(s['thresholds'][name]['count']) for name in ('gt0', 'ge3pct', 'ge5pct', 'ge10pct')) + ' |\n')
    with (OUT / 'paired_all_periods.md').open('w', encoding='utf-8') as handle:
        handle.write('# All fixed cells: equal-origin pairs\n\nUnits are percentage points; each commonly observable origin has one weight. N excludes origins with no signal or with either side unknown; those counts are kept in the original JSON.\n\n| Dimension | Period | View | Pair | H | Metric | Pair N | Δmean | Δmedian | Δq10 | Δq90 |\n|---|---|---|---|---:|---|---:|---:|---:|---:|---:|\n')
        for row in summary['equal_origin_rows']:
            for key in src.CHANNELS:
                m = row['metrics'][key]
                s = m['left_minus_right_mean']
                handle.write(f"| {row['dimension']} | {row['period']} | {row['view']} | {row['left']}−{row['right']} | {row['horizon']} | {key} | {m['paired_common_origin_n']} | " +
                    ' | '.join(pct(s[name]) for name in ('mean', 'median', 'q10', 'q90')) + ' |\n')
    notes = '\n[Result: EAR matched opportunity comparison]\nThe fixed single-batch windows and summaries are complete. The same-day upside increment of the EAR selections comes with deeper MAE, and the 2023Q4 common-pool comparison is weak. 330981 cache key/dates + 13 exact new projections; 0 API/fit/predict calls. Report and all fixed cells: output/research_once/short_joint_h15_ear_matched_opportunity_v1_20260908/report.md and summary.json.\n'
    for path in (DOC, LOG):
        with path.open('a', encoding='utf-8') as handle:
            handle.write(notes)
    pointer = dict(work_id=WORK, workflow_state='COMPLETE', final_publication=True,
        terminal=str(OUT / 'terminal.json'), summary=str(OUT / 'summary.json'), report=str(OUT / 'report.md'),
        retention_assessment=str(OUT / 'retention_assessment.json'), limitations=str(OUT / 'limitations.json'),
        exact_membership=str(OUT / 'membership_request.json'), event_window_manifest=str(OUT / 'event_window_manifest.json'),
        per_origin_statistics=str(OUT / 'per_origin_statistics.npz'), per_origin_pair_rows=str(OUT / 'per_origin_paired_metrics.jsonl.gz'),
        quote_projection=str(OUT / 'quote_projection_summary.json'))
    dump('publication_pointer.json', pointer)
    size = outbytes()
    if size > CAP:
        raise RuntimeError('1 GiB output limit exceeded')
    terminal = dict(work_id=WORK, owner='', workflow_state='COMPLETE', final_publication=True,
        status='COMPLETE_FIXED_SAME_EVENT_EAR_JOINT_OPPORTUNITY_COMPARISON', decision='EAR_HIGHER_UPSIDE_DEEPER_DRAWDOWN', viable_claim='DESCRIPTIVE_COMPARISON',
        named_version_disposition='RETAIN_DESCRIPTIVE_ENTRY_EVIDENCE_ONLY',
        publication_pointer=str(OUT / 'publication_pointer.json'), event_count=request['event_count'], common_events=request['common_riskset_count'],
        selected_counts=request['selected_counts'], source_counts=projection['counts'], summary_pooled_rows=len(summary['pooled_rows']),
        summary_equal_origin_rows=len(summary['equal_origin_rows']), new_output_bytes_before_terminal=size,
        effects=limitations['effects'], economic_window_batches=1, economic_repeats=0, parameter_changes=0,
        next='Possible follow-up: an exit study on the selected events', at=datetime.now(timezone.utc).isoformat())
    dump('terminal.json', terminal)
    phase('FORMAL_COMPLETE', terminal=terminal['status'], decision=terminal['decision'], bytes=outbytes())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['prepare', 'project', 'evaluate', 'summarize', 'finalize'])
    args = parser.parse_args()
    globals()[args.phase]()


if __name__ == '__main__':
    main()
