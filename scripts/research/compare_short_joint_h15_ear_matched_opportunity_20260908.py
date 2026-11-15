"""111: frozen same-event EAR / JOINT gross opportunity comparison.

Only the published107 quote component and pure, already-proven vector formulas
are reused. No107 memberships, window arrays or economic summaries are read.
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
WORK = 'POOL-20260908-JOINT-H15-111-EAR-MATCHED-COHORT-FIXED-OPPORTUNITY-COMPARISON'
CHECK = 'CPC-20260908-EAR-MEMBERSHIP-CACHE-WINDOW-BIND-01'
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
        raise RuntimeError('WORK111_65MIN_CAP')


def outbytes():
    return sum(path.stat().st_size for path in OUT.rglob('*') if path.is_file())


def formal(path):
    value = read(path)
    if value.get('workflow_state') != 'COMPLETE' or not value.get('final_publication'):
        raise RuntimeError('FORMAL_PUBLICATION_REQUIRED:' + str(path))
    return value


def load_data(directory='quote_arrays'):
    return {name: np.load(OUT / directory / (name + '.npy'), mmap_mode='r')
            for name in ('qfq', 'available', 'raw_open', 'guard')}


def prepare():
    gate()
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'membership_request_seal.json').exists():
        raise RuntimeError('MEMBERSHIP_ALREADY_SEALED_NO_REBUILD')
    formal(UP / 'terminal.json')
    formal(SCORES / 'terminal.json')
    formal(CACHE / 'quote_component_publication.json')
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
    native_selected_identity = {}
    for ai, arm in enumerate(ARMS):
        for vi, field in enumerate(('all_selected_events', 'nominalH15_retained_events')):
            for row in iter_rows(native['arm_paths'][arm][field]):
                selected[lookup[row['event_id']], ai, vi] = True
                native_selected_identity.setdefault(row['event_id'], {key: row.get(key) for key in field_names})
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
    binding_candidates = [i for i, row in enumerate(events) if selected[i].any() and current[i] == 1
                          and origins[int(oi[i])]['entry_index'] <= origins[int(oi[i])]['cutoff_index']]
    bi = binding_candidates[0]
    b = events[bi]
    selected_source = native_selected_identity[b['event_id']]
    binding = dict(event_index=bi, event_id=b['event_id'], source_event_id=selected_source['event_id'], source_security_key=b['source_security_key'],
                   stable_instrument_key=b['stable_instrument_key'], scheduled_entry_session=b['scheduled_entry_session'], role=b['role'],
                   origin_index=int(oi[bi]), in_full_riskset=True,
                   native_selected_identity={key: selected_source[key] for key in ('event_id', 'source_security_key', 'stable_instrument_key', 'origin_session', 'scheduled_entry_session', 'role')},
                   arm_memberships={arm: {VIEWS[vi]: bool(selected[bi, ai, vi]) for vi in range(2)} for ai, arm in enumerate(ARMS)})
    np.savez_compressed(OUT / 'sealed_event_index.npz', key_index=kk, origin_index=oi, common=common, selected=selected, current_eligible=current)
    np.savez_compressed(OUT / 'exact_request_mask.npz', requested=requested)
    write_rows('event_identity.jsonl.gz', [dict(event_index=i, **row) for i, row in enumerate(events)])
    write_rows('exact_projection_requests.jsonl.gz', (dict(source_security_key=key_axis[k], exact_dates=[cal[j] for j in np.flatnonzero(requested[k])],
                fields=['qfq open/high/low/close', 'raw open', 'same-source daily conservative entry status']) for k in range(len(key_axis))))
    dump('origin_registry.json', dict(work_id=WORK, keys=key_axis, calendar=cal, origins=origins, binding=binding))
    dump('execution_budget.json', dict(started_at=datetime.now(timezone.utc).isoformat(), deadline_epoch=time.time() + 65 * 60,
                                      minutes=65, new_output_bytes_cap=CAP, source_preparation_started_before_actual=True))
    dump('membership_request_seal.json', dict(work_id=WORK, sealed_at=datetime.now(timezone.utc).isoformat(),
        source110=str(UP / 'membership_seal.json'), source102=str(SCORES / 'terminal.json'), event_count=n,
        common_riskset_count=int(common.sum()), origin_count=len(origins), exact_key_dates=int(requested.sum()),
        selected_counts={arm: {VIEWS[vi]: int(selected[:, ai, vi].sum()) for vi in range(2)} for ai, arm in enumerate(ARMS)},
        calendar_bounds=['2021-09-01', '2023-12-29'], role_cutoffs=CUT,
        request_rule='Each original current-eligible event scheduled e through e+15, clipped at its role cutoff; no re-anchoring',
        field_rule='OHLC per-field availability plus raw_open and published daily conservative guard',
        controls='Full original R_event or G; selected nominal overlap never deduplicates controls. Same-signal-date control views separately identified.',
        no_signal='All native565 daily origins retained. No zero-return/cash imputation.',
        binding=binding, actual_postentry_quotes_read=0, API=0, fit=0, predict=0, TEST=0))
    phase('MEMBERSHIP_REQUEST_SEALED', events=n, common_events=int(common.sum()), exact_key_dates=int(requested.sum()),
          selected_counts=read(OUT / 'membership_request_seal.json')['selected_counts'])


def project():
    """Copy only sealed cache intersections; project only exact uncached days."""
    gate()
    seal = read(OUT / 'membership_request_seal.json')
    meta = read(OUT / 'origin_registry.json')
    if (OUT / 'quote_projection_started.json').exists():
        raise RuntimeError('PROJECTION_ALREADY_STARTED_NO_SOURCE_REPEAT')
    formal(CACHE / 'quote_component_publication.json')
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
    dump('quote_projection_started.json', dict(at=datetime.now(timezone.utc).isoformat(), exact_seal='membership_request_seal.json'))
    counts = Counter()
    lineage = []
    started = time.monotonic()
    binding_key = meta['binding']['source_security_key']
    order = sorted(range(len(keys)), key=lambda i: (keys[i] != binding_key, keys[i]))
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
                       cache_receipt=str(CACHE / 'quote_projection_receipt.json') if copied else None,
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
                row.update(projected_coordinate_id=result['coordinate_id'], source_csv=route['csv_path'], source_receipt=route['source_receipt'])
            elif remaining:
                counts['NO_REGISTERED_SOURCE_ROUTE_key_dates'] += len(remaining)
                for j in remaining:
                    gap_handle.write(json.dumps(dict(security_key=key, session=cal[j], reason='NO_REGISTERED_SOURCE_ROUTE')) + '\n')
            lineage.append(row)
            if key == binding_key:
                bind_one(meta, requested, k, copied, remaining, data, old, oldk, cache_days, row)
            if seq % 200 == 0 or seq == len(order):
                for arr in data.values():
                    arr.flush()
                phase('EXACT_QUOTE_COMPONENT', securities_done=seq, securities_total=len(order), counts=dict(counts), elapsed_seconds=round(time.monotonic() - started, 1))
    for arr in data.values():
        arr.flush()
    dump('quote_projection_receipt.json', dict(work_id=WORK, status='COMPLETE_SEALED_EXACT_CACHE_AND_SOURCE_DIFFERENCE',
        axes=['security_index', 'session_index', 'field_index'], fields=list(FIELDS), lineage=lineage, counts=dict(counts),
        exact_requested_key_dates=seal['exact_key_dates'], unknown_cache_cells_not_recovered=True,
        availability='Original per-field Unix seconds UTC; -1 unknown', role_cutoffs=CUT,
        guard_codes={'0': 'UNKNOWN', '1': 'DAILY_CONSERVATIVE_OK', '2': 'DAILY_CONSERVATIVE_BLOCKED'},
        guard_available_at='15:00 posthoc development proxy; not open-time information',
        source_read_scope='105 parses only exact missing key/dates; other CSV rows mechanically date-key scanned',
        API=0, fit=0, predict=0, TEST=0, old_component_retests=0))
    if outbytes() > CAP:
        raise RuntimeError('WORK111_1GIB_OUTPUT_CAP')
    phase('EXACT_QUOTE_COMPONENT_COMPLETE', counts=dict(counts), bytes=outbytes())


def bind_one(meta, requested, k, copied, remaining, data, old, oldk, cache_days, lineage):
    """The one new110-to111 binding check; no replay of105/107 arithmetic."""
    b = meta['binding']
    origin = meta['origins'][b['origin_index']]
    ep = origin['entry_index']
    expected_dates = meta['calendar'][ep:min(ep + 16, origin['cutoff_index'] + 1)]
    reused = [j for j in copied if meta['calendar'][j] in expected_dates]
    checks = dict(
        stable_event_identity=b['event_id'] == b['source_event_id'],
        selected_to_control_identity=all(b[key] == b['native_selected_identity'][key] for key in ('event_id', 'source_security_key', 'stable_instrument_key', 'scheduled_entry_session', 'role')),
        native_origin_connection=origin['decision_session'] == b['native_selected_identity']['origin_session'],
        original_full_riskset_membership=b['in_full_riskset'],
        cache_security_axis=meta['keys'][k] == b['source_security_key'],
        exact_scheduled_entry=meta['calendar'][ep] == b['scheduled_entry_session'],
        origin_role_cutoff=meta['calendar'][origin['cutoff_index']] == CUT[b['role']],
        fixed_horizon_index_offsets=[H - 1 for H in src.HORIZONS] == [4, 9, 14],
        sealed_dates_cover_original_fixed_windows=all(requested[k, meta['calendar'].index(day)] for day in expected_dates),
        no_requested_day_after_role_cutoff=not expected_dates or max(expected_dates) <= CUT[b['role']],
        original_availability_preserved=all(np.array_equal(data['available'][k, j], old['available'][oldk, cache_days[meta['calendar'][j]]]) for j in reused),
        reused_key_bound_to_same_source=not copied or lineage['cached_lineage']['security_key'] == b['source_security_key'])
    ok = all(checks.values())
    result = dict(work_id=WORK, check_id=CHECK, conclusion='PASS' if ok else 'FAIL', checks=checks,
        event_id=b['event_id'], source_security_key=b['source_security_key'], role=b['role'], arm_memberships=b['arm_memberships'],
        original_dates=expected_dates, cache_reused_dates=[meta['calendar'][j] for j in reused],
        attribution='NONE' if ok else 'DOWNSTREAM', value='KEEP', old_math_fixture_retests=0, source_rereads=0)
    dump('membership_cache_window_binding_receipt.json', result)
    post = '\n[检验后]\n检验ID: ' + CHECK + '\n结论: ' + result['conclusion'] + '\n担忧是否发生: ' + ('否' if ok else '是') + '\n直接证据: membership_cache_window_binding_receipt.json；正常生产首个冻结 event 的身份、原 riskset、scheduled entry、角色截止和缓存 availability 透传。未重算105/107公式。\n故障归因: ' + result['attribution'] + '\n对决策的影响: ' + ('允许完成本固定一次111生产。' if ok else '阻断新连接，停止经济计算。') + '\n检验价值: KEEP\n复发条件: 110身份 ABI、111轴连接或角色日期边界变更；不因样本条数增加重验。\n'
    for path in (DOC, LOG):
        with path.open('a', encoding='utf-8') as handle:
            handle.write(post)
    if not ok:
        raise RuntimeError('NEW_MEMBERSHIP_CACHE_WINDOW_BINDING_FAIL')
    phase('MEMBERSHIP_CACHE_WINDOW_BINDING_PASS', event_id=b['event_id'])


def evaluate():
    gate()
    if read(OUT / 'membership_cache_window_binding_receipt.json')['conclusion'] != 'PASS':
        raise RuntimeError('NEW_BINDING_PASS_REQUIRED')
    read(OUT / 'quote_projection_receipt.json')
    if (OUT / 'evaluation_started.json').exists():
        raise RuntimeError('ECONOMIC_WINDOWS_ALREADY_STARTED_NO_REPEAT')
    dump('evaluation_started.json', dict(at=datetime.now(timezone.utc).isoformat(), source='quote_projection_receipt.json',
                                       formula_source='107 pure vector_origin; unchanged105 formulas and existing proof'))
    meta = read(OUT / 'origin_registry.json')
    sealed = np.load(OUT / 'sealed_event_index.npz', allow_pickle=False)
    n = len(sealed['key_index'])
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
            result = vector_origin(sealed['key_index'][sl], origin['entry_index'], origin['cutoff_index'],
                                   sealed['current_eligible'][sl], data, meta['calendar'])
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
        existing_105_107_proof_consumed=True, old_formula_retests=0, API=0, fit=0, predict=0, TEST=0))
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
    # Compact originals are the only economic source. No quotes or evaluators rerun.
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
        raise RuntimeError('WORK111_1GIB_OUTPUT_CAP')
    phase('FIXED_SUMMARY_COMPLETE', pooled_rows=len(pooled_rows), equal_origin_rows=len(equal_rows), bytes=outbytes())


def pct(value):
    return 'unknown' if value is None else f'{100 * value:.3f}'


def finalize():
    """Publish the fixed aggregate report; no economic or source replay."""
    gate()
    if (OUT / 'terminal.json').exists():
        raise RuntimeError('ALREADY_FORMALLY_PUBLISHED')
    summary = read(OUT / 'summary.json')
    seal = read(OUT / 'membership_request_seal.json')
    projection = read(OUT / 'quote_projection_receipt.json')
    binding = read(OUT / 'membership_cache_window_binding_receipt.json')
    assessment = dict(work_id=WORK, decision='TRADEOFF_UNRESOLVED', viable_claim='NO_VIABLE_CLAIM',
        named_version='EAR_FULL_COMMON_JOINT_MATCHED_H15_OPPORTUNITY_V1',
        named_version_disposition='RETAIN_DESCRIPTIVE_ENTRY_EVIDENCE_ONLY',
        supported='EAR shows positive matched upside-opportunity increments at H5/H10/H15 in TRAIN and SEEN_DEV, alongside deeper MAE.',
        unresolved='Causal exit realization, fees, funding and the upside/downside tradeoff remain unresolved; observed maxima are not returns.',
        recent='2023Q4 EAR_COMMON-minus-JOINT_COMMON H15 T1 upside increment is negative and MAE is deeper; complete fixed periods remain disclosed.',
        decision_rule='No added numeric gate, no requirement of every-block/all-metric dominance, no pooled rescue or automatic STOP from mixed evidence.',
        next_action_owner='/root',
        next_action='Root may decide whether the mechanism justifies a separately frozen original-policy C or own-selected exit study; no automatic READY or execution.',
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
    lines = ['# EAR 同事件三臂 H5/H10/H15 机会与下行比较', '',
        '当前命名 EAR entry 版本的观察上行机会在 TRAIN 与 SEEN_DEV 的同日配对中有增量，同时通常承受更深的下探。结论为 **TRADEOFF_UNRESOLVED / NO_VIABLE_CLAIM**：保留 entry 描述性证据，尚未证明净利润或资金策略；不因混合结果自动淘汰，也不把它当成可以交易的策略。', '',
        '2023Q4 的共同池比较尤其弱：H15 的 EAR_COMMON 相对 JOINT_COMMON，day2+ MFE 均值增量为 -0.284 个百分点，MAE 均值差为 -4.159 个百分点，仅14个可配对 origin。完整季度、月份、固定12个月与近期窗口均已落盘；年度或24个月 pooled 结果不能覆盖这一近期问题。', '',
        '本项不新增收益门、p值门或“每块全部指标支配对照”的规则。是否值得另行冻结原文政策 C 或自身选中池内的因果退出研究，由 root 结合机制决定；没有自行编译 READY。', '',
        '## 人群、日期和口径', '',
        f"110 的完整 R_event 为 {seal['event_count']:,} 个事件，共同 G 为 {seal['common_riskset_count']:,} 个；全部 {seal['origin_count']} 个日历 origin 保留无信号与输入缺口记录。EAR_FULL / EAR_COMMON / JOINT_COMMON 的 ALL_SIGNALS 数分别为697 / 648 / 1086，nominalH15 数为697 / 647 / 1081。", '',
        'EAR_FULL 对照完整同日 R_event；EAR_COMMON 和 JOINT_COMMON 对照同一 G。EAR_COMMON 使用102原来完整前季度的阈值，不在 G 上重新训练或改阈值。JOINT_COMMON 是冻结原 W 概率排序前4名，非H15概率；G额外包含其旧特征与5000元 origin-close affordability 适用范围。EAR_FULL不受该交集静默替换。', '',
        '入场锚点严格为信号后计划下一交易日 open，不因未来停牌、缺价或涨跌而换 anchor。gross unfunded A/B 使用原 provider qfq、raw open 和15:00 conservative daily guard。这个 guard 是事后开发观察过滤，不是09:30已知信息或实际成交。', '',
        'TRAIN_COHORT_2021SEP_2022 只读至2022-12-30；SEEN_DEV_2023 至2023-12-29。首个2021年/Q3为部分窗口；锚定第三个12个月块也不完整。source 的日末 availability、unknown 和角色 cutoff 均透传。', '',
        f"330,994个精确代码日期中，{projection['counts'].get('reused_key_dates', 0):,} 直接复用107正式报价组件，{projection['counts'].get('projected_key_dates', 0)} 个未请求日期通过105/100本地源投影。一个新增日期的日终 entry status 缺口原样保留。缓存原有 unknown 未重新回源救援；107经济文件未读。", '',
        '## 同日、等 origin 权重的主要比较', '',
        '单位均为百分点。Δ为左臂减右臂；MAE更负表示下探更深。这里的N是该channel双方均可观测的origin数，每个origin只占一个权重；并不是 pooled 个股均值差。day2+ MFE仍是未来HIGH观察上界，不是可实现卖出收益。', '',
        '| Role | View | 左 − 右 | H | N day1/day2+/MAE | Δday1 MFE | Δday2+ MFE | ΔMAE |',
        '|---|---|---|---:|---:|---:|---:|---:|']
    for row in summary['equal_origin_rows']:
        if row['dimension'] == 'role':
            m = row['metrics']
            names = src.CHANNELS
            ns = '/'.join(str(m[key]['paired_common_origin_n']) for key in names)
            deltas = [pct(m[key]['left_minus_right_mean']['mean']) for key in names]
            lines.append(f"| {row['period']} | {row['view']} | {row['left']} − {row['right']} | {row['horizon']} | {ns} | " + ' | '.join(deltas) + ' |')
    lines += ['', '## 三个入选臂的 pooled 完整角色分布', '',
        '每行n/N为channel已知数/原始入选数。百分数均为未计费机会或下行观察值；阈值计数不等于获利交易数。完整对照、所有量化分位数、两个view和时间窗见summary.json及完整表。', '',
        '| Role | View | Arm | H | Channel | n/N | mean% | median% | q10% | q90% | >0 | ≥3% | ≥5% | ≥10% |',
        '|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in summary['pooled_rows']:
        if row['dimension'] == 'role' and row['population'] in ARMS:
            for channel in src.CHANNELS:
                s = row['metrics'][channel]
                hits = [str(s['thresholds'][key]['count']) for key in ('gt0', 'ge3pct', 'ge5pct', 'ge10pct')]
                lines.append(f"| {row['period']} | {row['view']} | {row['population']} | {row['horizon']} | {channel} | {s['n']}/{s['original_n']} | " +
                             ' | '.join(pct(s[key]) for key in ('mean', 'median', 'q10', 'q90')) + ' | ' + ' | '.join(hits) + ' |')
    lines += ['', '## 固定季度及12/24个月比较', '',
        '以下列出H15 ALL_SIGNALS；H5/H10、nominal view、月份和全部指标没有删去，完整表包含所有固定格。2021Q3仅2021年9月，BLOCK_3_PARTIAL仅2023年9月至12月；cross-role窗口不改变TRAIN/SEEN_DEV标签。', '',
        '| Period | Pair | N | Δday2+ MFE pp | ΔMAE pp |', '|---|---|---:|---:|---:|']
    for row in summary['equal_origin_rows']:
        if row['dimension'] in ('quarter', 'anchored_12m', 'recent_12m', 'recent_24m') and row['view'] == 'ALL_SIGNALS' and row['horizon'] == 15 and (row['left'] == 'EAR_FULL' or row['right'] == 'JOINT_COMMON'):
            m = row['metrics']
            lines.append(f"| {row['period']} | {row['left']} − {row['right']} | {m['MFE_T1_DAY2_TO_H']['paired_common_origin_n']} | {pct(m['MFE_T1_DAY2_TO_H']['left_minus_right_mean']['mean'])} | {pct(m['MAE_DAY1_TO_H']['left_minus_right_mean']['mean'])} |")
    lines += ['', '## 峰值、回吐、缺口与可复用产物', '',
        'summary.json 的 peak_timing 按每个channel原始分母和可观测分母给首次峰值holding day计数；相等峰值取最早日期。secondary metric保留peak_close_gap、next_open_gap及prior_session_adverse_before_peak的各自分位数与缺失数。峰前adverse仅取峰值之前已完整结束session的LOW，不假定峰日LOW先于HIGH。所有gap均除以entry open。', '',
        'day1 HIGH在T+1下不可卖；H15 day15 HIGH在期限open之后，只作postcap诊断。next-open gap可能跨原H窗口，但受同一角色cutoff，末尾不可观测即unknown。给回吐或MAE不表示存在当时可见的退出方法。', '',
        'R_EVENT / G_COMMON的pooled表保持完整原riskset；AT_*_SIGNAL_ORIGINS表进一步限定该臂该view确实有信号的日期，仍不独立对control证券去重。EAR_COMMON-minus-JOINT_COMMON只在双方该指标可观测的日期配对。无信号、单边可观测、已选但metric unknown等origin数量均在equal_origin_rows内，未补成现金零收益。', '',
        '交付入口 publication_pointer.json；summary.json含2352行pooled人口/窗口汇总与1176行等origin配对汇总，覆盖49个预定角色/年/季度/月/12个月/24个月窗口组。primary_all_periods.md与paired_all_periods.md提供完整主指标可读表。event_identity.jsonl.gz / sealed_event_index.npz / event_windows保留精确事件、membership、entry status和三窗口值；per_origin_statistics.npz及两份per_origin JSONL保存配对基础。', '',
        '本次唯一新membership/cache/window连接检验PASS，POST已经立即落盘；复用105/107公式证明，没有重新跑旧fixture、费用、hash或经济试验。输出仅支持本命名版本的A/B证据；原源PIT、公告first-history范围、重叠窗口与共同origin依赖限制继续有效。', '']
    (OUT / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    with (OUT / 'primary_all_periods.md').open('w', encoding='utf-8') as handle:
        handle.write('# 全部固定格：pooled primary\n\nMean/quantiles为百分比；n/N为各channel已知/原始；阈值为事件计数。\n\n| Dimension | Period | View | Population | H | Metric | n/N | mean | median | q10 | q90 | >0 | ≥3 | ≥5 | ≥10 |\n|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n')
        for row in summary['pooled_rows']:
            for key in src.CHANNELS:
                s = row['metrics'][key]
                handle.write(f"| {row['dimension']} | {row['period']} | {row['view']} | {row['population']} | {row['horizon']} | {key} | {s['n']}/{s['original_n']} | " +
                    ' | '.join(pct(s[name]) for name in ('mean', 'median', 'q10', 'q90')) + ' | ' +
                    ' | '.join(str(s['thresholds'][name]['count']) for name in ('gt0', 'ge3pct', 'ge5pct', 'ge10pct')) + ' |\n')
    with (OUT / 'paired_all_periods.md').open('w', encoding='utf-8') as handle:
        handle.write('# 全部固定格：等origin配对\n\n单位百分点；一个共同可观测origin一个权重。N不含无信号或任一方unknown；其计数在原JSON中保留。\n\n| Dimension | Period | View | Pair | H | Metric | Pair N | Δmean | Δmedian | Δq10 | Δq90 |\n|---|---|---|---|---:|---|---:|---:|---:|---:|---:|\n')
        for row in summary['equal_origin_rows']:
            for key in src.CHANNELS:
                m = row['metrics'][key]
                s = m['left_minus_right_mean']
                handle.write(f"| {row['dimension']} | {row['period']} | {row['view']} | {row['left']}−{row['right']} | {row['horizon']} | {key} | {m['paired_common_origin_n']} | " +
                    ' | '.join(pct(s[name]) for name in ('mean', 'median', 'q10', 'q90')) + ' |\n')
    notes = '\n[111正式结果]\n固定单批窗口与汇总完成，TRADEOFF_UNRESOLVED / NO_VIABLE_CLAIM。保留命名EAR entry描述性证据；同日上行机会增量伴随更深MAE，2023Q4共同池比较弱。不增加数值门，不自动STOP或编译C；由root决定后继另冻。330981缓存key/dates+13精确新投影，0API/fit/predict/TEST/107经济读取；1新绑定PASS，旧证明不重验。报告及所有固定格见 output/research_once/short_joint_h15_ear_matched_opportunity_v1_20260908/report.md 和 summary.json。\n'
    for path in (DOC, LOG):
        with path.open('a', encoding='utf-8') as handle:
            handle.write(notes)
    pointer = dict(work_id=WORK, workflow_state='COMPLETE', final_publication=True,
        terminal=str(OUT / 'terminal.json'), summary=str(OUT / 'summary.json'), report=str(OUT / 'report.md'),
        retention_assessment=str(OUT / 'retention_assessment.json'), limitations=str(OUT / 'limitations.json'),
        exact_membership=str(OUT / 'membership_request_seal.json'), event_window_manifest=str(OUT / 'event_window_manifest.json'),
        per_origin_statistics=str(OUT / 'per_origin_statistics.npz'), per_origin_pair_rows=str(OUT / 'per_origin_paired_metrics.jsonl.gz'),
        source_receipt=str(OUT / 'quote_projection_receipt.json'), binding_receipt=str(OUT / 'membership_cache_window_binding_receipt.json'))
    dump('publication_pointer.json', pointer)
    size = outbytes()
    if size > CAP:
        raise RuntimeError('WORK111_1GIB_OUTPUT_CAP')
    terminal = dict(work_id=WORK, owner='/root/continue_pool_c', workflow_state='COMPLETE', final_publication=True,
        status='COMPLETE_FIXED_SAME_EVENT_EAR_JOINT_OPPORTUNITY_COMPARISON', decision='TRADEOFF_UNRESOLVED', viable_claim='NO_VIABLE_CLAIM',
        named_version_disposition='RETAIN_DESCRIPTIVE_ENTRY_EVIDENCE_ONLY', check_status=binding['conclusion'],
        publication_pointer=str(OUT / 'publication_pointer.json'), event_count=seal['event_count'], common_events=seal['common_riskset_count'],
        selected_counts=seal['selected_counts'], source_counts=projection['counts'], summary_pooled_rows=len(summary['pooled_rows']),
        summary_equal_origin_rows=len(summary['equal_origin_rows']), new_output_bytes_before_terminal=size,
        effects=limitations['effects'], economic_window_batches=1, economic_repeats=0, parameter_changes=0,
        next='Separate root mechanism decision, no automatic C/D/READY or family verdict', at=datetime.now(timezone.utc).isoformat())
    dump('terminal.json', terminal)
    phase('FORMAL_COMPLETE', terminal=terminal['status'], decision=terminal['decision'], bytes=outbytes())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['prepare', 'project', 'evaluate', 'summarize', 'finalize'])
    args = parser.parse_args()
    globals()[args.phase]()


if __name__ == '__main__':
    main()
