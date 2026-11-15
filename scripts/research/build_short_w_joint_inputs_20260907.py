"""Granted daily joint-event inputs. No fit, network, account, or old-file writes."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(ROOT / 'src')]
import build_short_h3_first_exit_inputs_20260907 as h3
import build_short_cmlf_selected_w_paths_20260907 as paths
from build_short_w5_buy_daily_inputs_20260907 import calendar_rows
from ashare_strategy.features.math12 import RANKED_FEATURE_COLUMNS
from ashare_strategy.features.m12_dated_monthly_input_assembly import common_source_code

WORK = 'POOL-20260907-SHORT-W-JOINT-DATA-CHAIN-01'
ABI = 'SHORT_W_JOINT_DAILY_INPUTS_V1_20260907'
POP = 'SHORT_W_JOINT_DATED_MAINBOARD_1X5000_V1_20260907'
OUT = ROOT / 'output/research_once/short_w_joint_inputs_v1_20260907'
OLD = ROOT / 'output/research_once/short_h3_first_exit_inputs_v1_20260907'
CAL16 = ROOT / 'output/research_once/short_h3_cal16_inputs_v1_20260907'
BIND = ROOT / 'docs/research/short_w_joint_exact_binding_20260907.json'
GRANT = ROOT / 'docs/research/short_w_joint_data_actual_grant_20260907.json'
DOC = ROOT / 'docs/research/short_w_joint_data_20260907.md'
POOL = ROOT / 'docs/research/STRATEGY_RESEARCH_POOL.yaml'
FIELDS = ('adjusted_open', 'adjusted_close', 'high', 'low', 'raw_open', 'raw_close')
STATUSES = ['OK', 'MISSING_PREVIOUS_EXACT_CLOSE_GUARD', 'MISSING_ADJUSTED_OPEN',
            'MISSING_RAW_EXECUTION_PROXY_FIELDS', 'UNKNOWN_TRADING_STATUS', 'SUSPENDED',
            'UNKNOWN_SIDE_EXECUTION', 'BUY_BLOCKED_DAILY_CONSERVATIVE',
            'SELL_BLOCKED_DAILY_CONSERVATIVE', 'UNKNOWN_SOURCE_EXECUTION',
            'SOURCE_EXECUTION_UNAVAILABLE', 'NO_EXACT_SESSION_QUOTE', 'UNKNOWN_SOURCE_CONTEXT']
PRICE_MODE = 'NATIVE_QFQ_HIGH_LOW_IF_PRESENT_ELSE_RAW_TIMES_LEGAL_DAILY_QFQ_CLOSE_FACTOR_PROXY'
read, dump, stamp = h3.read, h3.dump, h3.stamp


def post(check, conclusion, evidence, impact, attribution='NONE'):
    with DOC.open('a', encoding='utf-8') as f:
        f.write(f'\n[检验后]\n检验ID: {check}\n结论: {conclusion}\n担忧是否发生: '
                + ('否' if conclusion == 'PASS' else '无法判断')
                + f'\n直接证据: {evidence}\n故障归因: {attribution}\n对决策的影响: {impact}'
                + ('\n检验价值: REMOVE' if attribution == 'TEST_ITSELF' else '\n检验价值: KEEP')
                + '\n复发条件: 仅本 adapter 语义改变或未提交分区新增；已提交分区不重算。\n')


def emit(**x):
    print(json.dumps(x, ensure_ascii=False, default=str), flush=True)


def save_npz(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('wb') as f:
        np.savez_compressed(f, **data)
    temp.replace(path)


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def running():
    control = POOL.read_text(encoding='utf-8-sig').split('operating_policy:', 1)[0]
    return bool(re.search(r'^  mode:\s*[\"\']?RUNNING[\"\']?\s*$', control, re.M))


class CommittedPause(RuntimeError):
    pass


class Budget:
    def __init__(self, grant, seconds):
        self.grant, self.seconds, self.start = grant, seconds, time.monotonic()
        self.bytes = sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())

    def boundary(self):
        if not running():
            raise CommittedPause('PAUSED_REVIEW_HOLD_AT_COMMITTED_UNIT')
        if time.monotonic() - self.start > self.seconds:
            raise CommittedPause('PAUSED_INITIAL_PHASE_CAP_AT_COMMITTED_UNIT')
        if self.bytes > self.grant['max_output_bytes']:
            raise CommittedPause('PAUSED_OUTPUT_CAP_AT_COMMITTED_UNIT')

    def account(self, *files):
        self.bytes += sum(Path(p).stat().st_size for p in files)


def origin_arrays(frame, features):
    """Prediction population uses origin data only; no label argument exists."""
    x = frame[features].to_numpy(dtype=np.float64)
    raw = pd.to_numeric(frame.raw_close, errors='coerce').to_numpy(dtype=np.float64)
    q1, q2 = h3.quantities(raw, 1), h3.quantities(raw, 2)
    eligible = frame.model_universe_eligible.fillna(False).astype(bool).to_numpy()
    finite = np.isfinite(x).all(axis=1)
    take = eligible & finite & (q1 >= 100)
    return dict(x=x[take], stable_ids=frame.stable_instrument_key.astype(str).to_numpy(dtype=str)[take],
                stock_codes=frame.stock_code.astype(str).str.zfill(6).to_numpy(dtype=str)[take],
                raw_close=raw[take], qcap_1x=q1[take], qcap_2x=q2[take]), dict(
        original_rows=len(frame), eligible=int(eligible.sum()), finite_eligible=int((eligible & finite).sum()),
        origin_unaffordable=int((eligible & finite & (q1 < 100)).sum()), prediction_rows=int(take.sum()))


def proxy_projection(frame, typed, record, calendar):
    """Consume registered quote guard. Missing high/low use explicit daily factor."""
    f = frame.reindex(calendar)
    p = h3.project_quotes(frame, typed, record, calendar)
    factor = np.divide(p['adjusted_close'], p['raw_close'],
                       out=np.full(len(calendar), np.nan),
                       where=np.isfinite(p['raw_close']) & (p['raw_close'] > 0))
    factor[~np.isfinite(factor) | (factor <= 0)] = np.nan
    mode = {}
    for field in ('high', 'low'):
        reconstructed = h3.numeric_col(f, field + '_raw').to_numpy(dtype=float) * factor
        if typed:
            native = f.native_qfq_coordinate_status.eq('EXPLICIT_NATIVE2')
            available = pd.to_datetime(f.derived_event_available_at, errors='coerce', utc=True)
            clock = pd.to_datetime(pd.Series(calendar, index=f.index) + 'T15:00:00+08:00', utc=True)
            direct = h3.numeric_col(f, field+'_qfq_native').where(native, h3.numeric_col(f, field+'_qfq_derived'))
            direct = direct.where(native | available.le(clock)).to_numpy(dtype=float)
        else:
            direct = h3.numeric_col(f, field+'_qfq').to_numpy(dtype=float)
        ok = np.isfinite(direct) & (direct > 0)
        p[field] = np.where(ok, direct, reconstructed)
        p[field][~np.isfinite(p[field]) | (p[field] <= 0)] = np.nan
        mode[field] = dict(native_or_direct=int(ok.sum()), reconstructed=int((~ok & np.isfinite(p[field])).sum()),
                           unknown=int((~np.isfinite(p[field])).sum()))
    for side in ('entry', 'exit'):
        p[side + '_status'] = np.asarray([STATUSES.index(s) for s in p[side + '_status']], dtype=np.uint8)
    return p, dict(price_mode=PRICE_MODE, fields=mode,
        factor_source='LEGAL_AT_BAR_CLOSE_QFQ_CLOSE_DIVIDED_BY_SAME_DAY_RAW_CLOSE',
        rounding_limit='RECONSTRUCTED_HIGH_LOW_NOT_CERTIFIED_EQUAL_TO_VENDOR_ROUNDED_QFQ_HIGH_LOW',
        same_proxy_all_arms=True, missing_factor='UNKNOWN_NO_IMPUTATION')


def joint_labels(opening, highs, lows, *, L, W_end, cutoff):
    """All input columns are frozen roster order; unknown rows remain present."""
    n = len(opening); observed = highs.shape[0]
    states = np.full((n, 10), -1, dtype=np.int8)
    anchor_ok = np.isfinite(opening) & (opening > 0)
    active = anchor_ok.copy(); cumulative = np.zeros(n, dtype=np.int8)
    first_u = np.zeros(n, dtype=np.int16); first_d = np.zeros(n, dtype=np.int16)
    same_first = np.zeros(n, dtype=bool)
    for k in range(min(observed, L)):
        valid = np.isfinite(highs[k]) & (highs[k] > 0) & np.isfinite(lows[k]) & (lows[k] > 0)
        active &= valid
        u = active & (highs[k] >= 1.05 * opening)
        d = active & (lows[k] <= .95 * opening)
        same_first |= active & (cumulative == 0) & u & d
        first_u[(first_u == 0) & u] = k + 1
        first_d[(first_d == 0) & d] = k + 1
        cumulative |= u.astype(np.int8) | (d.astype(np.int8) << 1)
        states[active, k] = cumulative[active]
    boundary = bool(W_end and W_end <= cutoff and observed == L)
    complete = active & boundary
    mature = complete.copy(); known = complete.copy()
    first_u[~known] = -1; first_d[~known] = -1
    maxhigh, minlow = np.full(n, np.nan), np.full(n, np.nan)
    if observed:
        maxhigh[complete] = np.max(highs[:, complete], axis=0) / opening[complete] - 1
        minlow[complete] = np.min(lows[:, complete], axis=0) / opening[complete] - 1
    reasons = np.full(n, 'COMPLETE_BILATERAL_W', dtype='U48')
    reasons[~complete] = 'UNKNOWN_BILATERAL_PATH_OR_REFERENCE'
    if not boundary:
        reasons[:] = 'RIGHT_CENSORED_AT_ROLE_CUTOFF'
    available = np.where(complete, W_end + 'T15:00:00+08:00', '').astype('U40')
    return dict(states=states, known=known, mature=mature, complete=complete, available_at=available,
        scheduled_max_high_return=maxhigh, scheduled_min_low_return=minlow,
        first_up_day=first_u, first_down_day=first_d, same_bar_first_both_unknown_order=same_first,
        label_status=reasons)


def require_prediction(receipt, binding, origins):
    if receipt.get('status') != 'ALL_CAL_PREDICTIONS_COMMITTED_BEFORE_LABELS':
        raise PermissionError('CAL_PREDICTION_BARRIER_NOT_COMMITTED')
    pop = receipt.get('input_population_receipt_id', receipt.get('population_receipt_id'))
    if pop != binding['population_receipt_id'] or receipt.get('abi') != ABI:
        raise PermissionError('CAL_PREDICTION_POPULATION_OR_ABI_MISMATCH')
    expected = {r['origin_id'] for r in origins if r['role'] != 'TRAIN'}
    if set(receipt.get('completed_origin_ids', [])) != expected:
        raise PermissionError('CAL_PREDICTION_ORIGINS_NOT_COMPLETE')
    if set(receipt.get('arms', [])) != set(binding.get('prediction_arm_ids', [binding['method_id'], binding['control_id'], 'H3_W_FIRST_EXIT_NET_SPO_LINEAR_V1'])):
        raise PermissionError('CAL_THREE_ARMS_NOT_COMMITTED')
    if not receipt.get('selected_keys_path'):
        raise PermissionError('SEALED_SELECTED_KEYS_PATH_REQUIRED')


def authority(phase):
    b, g = read(BIND), read(GRANT)
    if g.get('authorized_by') != '/root' or g.get('work_id') != WORK or g.get('mode') != 'ACTUAL':
        raise PermissionError('EXACT_ROOT_ACTUAL_GRANT_REQUIRED')
    if b.get('abi') != ABI or b.get('population_receipt_id') != POP:
        raise PermissionError('FROZEN_BINDING_MISMATCH')
    if read(OUT / 'source_receipt.json').get('status') != 'PASS_SOURCE_NEW_STATE_ABI_PROXY_BARRIER':
        raise PermissionError('SOURCE_PROOF_REQUIRED')
    if not running():
        raise CommittedPause('PAUSED_REVIEW_HOLD_BEFORE_ACTUAL')
    if phase == 'cal':
        manifest = read(OUT / 'input_manifest.json')
        require_prediction(read(ROOT / g['cal_labels_require']), b, manifest['origins'])
    return b, g


def freeze_origins(binding, grant, budget):
    target = OUT / 'input_manifest.json'
    if target.exists() and read(target).get('status') == 'COMPLETE':
        return read(target)
    source = read(CAL16 / 'origin_freeze.json')
    official = read(CAL16 / 'scope_binding.json')['calendar']
    lookup = {r['origin_session']: r for r in source['origins']}
    schedule = [r for role, (start, end, cutoff) in binding['periods'].items()
                for r in calendar_rows(official, role, start, end, cutoff[:10])]
    if len(schedule) > grant['max_origins']:
        raise RuntimeError('ORIGIN_CAP_EXCEEDED')
    features = list(RANKED_FEATURE_COLUMNS)
    origins = []
    for r in schedule:
        budget.boundary()
        day, role = r['origin_session'], r['split']
        target_npz = OUT / 'origins' / f'{day}.npz'; receipt = target_npz.with_suffix('.json')
        if receipt.exists():
            item = read(receipt)
        else:
            upstream = lookup[day]
            frame = pd.read_pickle(upstream['features_path'])
            data, counts = origin_arrays(frame, features)
            save_npz(target_npz, data)
            item = dict(origin_id=day, role=role, scheduled_entry_session=r['scheduled_entry_session'],
                W_end=r['W_end'], T10_end=r['T10_end'], L=r['W_sessions'], origin_path=str(target_npz),
                n=len(data['stock_codes']), training_label_path=str(OUT / 'labels/TRAIN' / f'{day}.npz') if role=='TRAIN' else None,
                cutoff=binding['periods'][role][2], upstream_origin_path=upstream['features_path'],
                upstream_origin_receipt=str(Path(upstream['features_path']).with_suffix('.json')),
                counts=counts, abi=ABI, population_receipt_id=POP, future_labels_read=0,
                population_filter='ORIGIN_ELIGIBLE_FINITE_RANK_AND_RAW_CLOSE_QCAP_1X_GE100_ONLY',
                committed_at=stamp())
            dump(receipt, item); budget.account(target_npz, receipt)
        origins.append(item)
        if len(origins) % 100 == 0:
            emit(phase='ORIGIN_INPUTS', completed=len(origins), rows=sum(q['n'] for q in origins), seconds=time.monotonic()-budget.start)
    roles = {role: dict(D=sum(r['role']==role for r in origins), R=sum(r['n'] for r in origins if r['role']==role),
                         empty_origins=sum(r['role']==role and r['n']==0 for r in origins)) for role in binding['periods']}
    result = dict(status='COMPLETE', abi=ABI, population_receipt_id=POP, input_population_receipt_id=POP,
        work_id=WORK, feature_order=features, periods=binding['periods'], origins=origins,
        scope_counts=roles, D=len(origins), R=sum(r['n'] for r in origins), calendar=official,
        source_index=str(h3.REGISTRY), prediction_population_independent_of_future_labels=True,
        price_mode=PRICE_MODE, output_bytes_so_far=budget.bytes, committed_at=stamp())
    dump(target, result); budget.account(target)
    return result


class RoleQuotes:
    def __init__(self, role, manifest, binding, grant, budget):
        self.role, self.budget = role, budget
        self.cutoff = binding['periods'][role][2][:10]
        rows = [r for r in manifest['origins'] if r['role'] == role]
        self.calendar = [d for d in manifest['calendar'] if rows[0]['origin_id'] <= d <= self.cutoff]
        self.positions = {d:i for i,d in enumerate(self.calendar)}
        code_set = set()
        for r in rows:
            with np.load(r['origin_path'], allow_pickle=False) as z:
                code_set.update(common_source_code(str(c), [r['origin_id'], min(r['W_end'], self.cutoff)]) for c in z['stock_codes'])
        self.codes = sorted(code_set); self.indices = {c:i for i,c in enumerate(self.codes)}
        records = read(h3.REGISTRY)['records']
        if len(records) > grant['max_source_records']:
            raise RuntimeError('SOURCE_RECORD_CAP_EXCEEDED')
        cache_index = read((OLD if role == 'TRAIN' else CAL16) / 'source_cache_index.json')
        shape = (len(self.calendar), len(self.codes))
        self.data = {f:np.full(shape, np.nan) for f in FIELDS}
        self.data.update({s:np.full(shape, STATUSES.index('UNKNOWN_SOURCE_CONTEXT'), dtype=np.uint8) for s in ('entry_status','exit_status')})
        resident = sum(v.nbytes for v in self.data.values())
        if resident > grant['max_resident_quote_cache_bytes']:
            raise RuntimeError('RESIDENT_QUOTE_CACHE_CAP_EXCEEDED')
        self.lineage = {}
        for j, code in enumerate(self.codes):
            budget.boundary()
            key = h3.provider(code)
            dest = OUT / 'quote_projection' / role / f'{key}.npz'; commit = dest.with_suffix('.json')
            if commit.exists():
                lineage = read(commit); projected = load_npz(dest)
            else:
                record = records.get(key)
                contexts = cache_index.get(key, {})
                candidate = contexts.get('CAL2')
                chosen = None
                if record and candidate:
                    expected_path, typed = h3.select_daily_quote_context(record, self.cutoff)
                    if candidate.get('source_path') == str(expected_path):
                        chosen = candidate
                if chosen:
                    allowed_last = '2021-06-30' if role == 'TRAIN' else '2022-12-31'
                    if chosen['max_cutoff'] > allowed_last:
                        raise PermissionError('UNSEALED_OR_DEV_RAW_CACHE_FORBIDDEN')
                    frame = pd.read_pickle(chosen['path'])
                    if resident + int(frame.memory_usage(index=True, deep=True).sum()) > grant['max_resident_quote_cache_bytes']:
                        raise RuntimeError('RESIDENT_WITH_RAW_FRAME_CAP_EXCEEDED')
                    frame = frame.loc[self.calendar[0]:self.cutoff].copy()
                    projected, price_lineage = proxy_projection(frame, chosen['typed'], record, self.calendar)
                    lineage = dict(status='COMPLETE_LOCAL_ROLE_PROJECTION', source_key=key,
                        source_path=chosen['source_path'], reused_cache_path=chosen['path'],
                        reused_source_receipt=chosen, role=role, cutoff=self.cutoff,
                        projected_start=self.calendar[0], projected_end=self.calendar[-1], **price_lineage)
                    del frame
                else:
                    projected = {f:np.full(len(self.calendar), np.nan) for f in FIELDS}
                    projected.update({s:np.full(len(self.calendar), STATUSES.index('UNKNOWN_SOURCE_CONTEXT'), dtype=np.uint8) for s in ('entry_status','exit_status')})
                    lineage = dict(status='UNKNOWN_EXACT_SOURCE_CONTEXT', source_key=key, role=role,
                        cutoff=self.cutoff, price_mode=PRICE_MODE, reason='NO_MATCHING_BOUNDED_EXISTING_CONTEXT',
                        prediction_rows_retained=True)
                projected = {f:projected[f] for f in (*FIELDS, 'entry_status', 'exit_status')}
                save_npz(dest, projected); dump(commit, dict(lineage, committed_at=stamp(), projection_path=str(dest)))
                budget.account(dest, commit)
            for f in self.data:
                self.data[f][:, j] = projected[f]
            self.lineage[code] = lineage
            if (j+1) % 200 == 0:
                emit(phase='ROLE_QUOTES', role=role, completed=j+1, total=len(self.codes), resident_bytes=resident, seconds=time.monotonic()-budget.start)
        dump(OUT / 'quote_projection' / role / 'projection_receipt.json', dict(status='COMPLETE', role=role,
             source_count=len(self.codes), resident_bytes=resident, price_mode=PRICE_MODE,
             statuses=dict(Counter(r['status'] for r in self.lineage.values())), committed_at=stamp()))

    def indices_for(self, row, codes):
        return np.asarray([self.indices[common_source_code(str(c), [row['origin_id'], min(row['W_end'], self.cutoff)])] for c in codes], dtype=int)

    def labels(self, row, codes):
        js = self.indices_for(row, codes)
        start = self.positions.get(row['scheduled_entry_session'])
        days = [d for d in self.calendar if row['scheduled_entry_session'] <= d <= row['W_end']]
        ii = [self.positions[d] for d in days]
        opening = self.data['adjusted_open'][start, js] if start is not None else np.full(len(js), np.nan)
        return joint_labels(opening, self.data['high'][np.ix_(ii, js)], self.data['low'][np.ix_(ii, js)],
                            L=row['L'], W_end=row['W_end'], cutoff=self.cutoff)

    def event_rows(self, row, data, i):
        code = common_source_code(str(data['stock_codes'][i]), [row['origin_id'], min(row['W_end'], self.cutoff)])
        j = self.indices[code]; result = []
        for day in self.calendar:
            if row['origin_id'] <= day <= row['W_end']:
                t = self.positions[day]
                result.append(dict(session=day, raw_open=paths.number(self.data['raw_open'][t,j]),
                    adjusted_open=paths.number(self.data['adjusted_open'][t,j]), high=paths.number(self.data['high'][t,j]),
                    low=paths.number(self.data['low'][t,j]), close=paths.number(self.data['adjusted_close'][t,j]),
                    buy_status=STATUSES[int(self.data['entry_status'][t,j])], sell_status=STATUSES[int(self.data['exit_status'][t,j])],
                    origin_qcap_1x=int(data['qcap_1x'][i]), origin_qcap_2x=int(data['qcap_2x'][i]), origin_qcap_status='KNOWN'))
        return result, self.lineage[code]


def path_summary(bars, calendar, entry, W_end, cutoff):
    subset = [r for r in bars if entry <= r['session'] <= min(W_end, cutoff)]
    complete = W_end <= cutoff and all(paths.positive(r.get(f)) for r in subset for f in ('high','low'))
    functions = paths.upstream()
    value = paths.metrics(bars, calendar, entry, cutoff, functions[0], functions, fixed_end=W_end)
    value['bilateral_complete'] = complete and value.get('window_complete', False)
    if not value['bilateral_complete']:
        value['path_status'] = 'INCOMPLETE_BILATERAL_OR_CENSORED'
        value['observed_max_high_lower_bound'] = value.get('observed_max_high_lower_bound', value.get('max_high_return'))
        for f in ('max_high_return', 'min_low_return', 'max_close_return', 'first_hit_5pct_day'):
            value[f] = None
        value['window_complete'] = False
    value['day1_only_5pct_opportunity'] = bool(subset and paths.positive(value.get('entry_price'))
        and paths.positive(subset[0].get('high')) and subset[0]['high'] / value['entry_price'] - 1 >= .05
        and value['bilateral_complete'] and all(r['high'] / value['entry_price'] - 1 < .05 for r in subset[1:])) if value['bilateral_complete'] else None
    return value


def selected_event(key, row, data, i, quotes, calendar):
    bars, lineage = quotes.event_rows(row, data, i)
    scheduled, end, cutoff = row['scheduled_entry_session'], row['W_end'], quotes.cutoff
    nested = path_summary(bars, calendar, scheduled, end, cutoff)
    nested['entry_reference'] = 'SCHEDULED_OPEN_DIAGNOSTIC_NOT_FILL'
    _, _, affordable, _, _ = paths.upstream()
    entry = None; statuses = []
    for bar in bars:
        if bar['session'] < scheduled:
            continue
        state = bar['buy_status']
        q = min(bar['origin_qcap_1x'], affordable(5000., bar['raw_open'], 1))
        if state == 'OK' and q >= 100 and paths.positive(bar['adjusted_open']):
            entry = bar['session']; break
        statuses.append('OPEN_UNAFFORDABLE' if state == 'OK' and q < 100 else state)
    if entry:
        result = path_summary(bars, calendar, entry, end, cutoff)
        result.update(entry_status='REFERENCE_FILL_OBSERVED', entry_reference='OBSERVED_EXECUTABLE_OPEN_REFERENCE', observed_entry_session=entry)
    else:
        nofill = end <= cutoff and bool(statuses) and all(s in paths.KNOWN_BLOCKED for s in statuses)
        result = dict(path_status='NO_FILL_OBSERVED' if nofill else 'PENDING_UNEXECUTABLE',
            entry_status='NO_FILL_WITHIN_ORIGINAL_W' if nofill else 'UNKNOWN_OR_CENSORED_ENTRY',
            entry_reference='NO_OBSERVED_EXECUTABLE_ENTRY', observed_entry_session=None,
            max_high_return=None, min_low_return=None, max_close_return=None, bilateral_complete=False)
    result.update(key, scheduled_entry_session=scheduled, scheduled_W_end=end, W_end=end,
        T10_end=row['T10_end'], pending_entry_sessions=len(statuses), entry_search_statuses=statuses,
        scheduled_open_diagnostic=nested, source_binding=lineage, actual_account_fill=False,
        future_path_used_for_actions=False, same_bar_high_low_order='UNKNOWN')
    for stem in ('first_peak', 'first_hit_3pct', 'first_hit_5pct', 'first_hit_10pct'):
        d = result.get(stem + '_session')
        result[stem + '_day_since_scheduled'] = calendar.index(d)-calendar.index(scheduled)+1 if d else None
    return paths.clean(result)


def labels_phase(roles, manifest, binding, grant, budget, selected=None):
    all_labels = []
    selected = selected or []
    group = defaultdict(list)
    for key in selected:
        group[(key['role'], key['origin_id'])].append(key)
    for role in roles:
        quotes = RoleQuotes(role, manifest, binding, grant, budget)
        for row in (r for r in manifest['origins'] if r['role']==role):
            budget.boundary()
            day = row['origin_id']; dest = OUT / 'labels' / role / f'{day}.npz'; commit = dest.with_suffix('.json')
            with np.load(row['origin_path'], allow_pickle=False) as z:
                data = {k:z[k] for k in z.files}
            if commit.exists():
                item = read(commit)
            else:
                label = quotes.labels(row, data['stock_codes'])
                save_npz(dest, label)
                item = dict(origin_id=day, role=role, label_path=str(dest), n=row['n'], L=row['L'],
                    known_count=int(label['known'].sum()), complete_count=int(label['complete'].sum()),
                    mature_count=int(label['mature'].sum()), status_counts=dict(Counter(label['label_status'])),
                    original_prediction_rows_preserved=True, price_mode=PRICE_MODE, committed_at=stamp())
                dump(commit, item); budget.account(dest, commit)
            all_labels.append(item)
            keys = group[(role, day)]
            if keys:
                target = OUT / 'selected_events' / f'{day}.json'; receipt = target.with_suffix('.commit.json')
                if not receipt.exists():
                    ids = {str(s):i for i,s in enumerate(data['stable_ids'])}
                    events = []
                    for key in keys:
                        i = ids[key['stable_id']]
                        if str(data['stock_codes'][i]) != str(key['stock_code']).zfill(6):
                            raise ValueError('SEALED_SELECTED_IDENTITY_NOT_IN_ORIGIN')
                        events.append(selected_event(key, row, data, i, quotes, manifest['calendar']))
                    dump(target, events); dump(receipt, dict(status='COMPLETE', origin_id=day, path=str(target), events=len(events), committed_at=stamp()))
                    budget.account(target, receipt)
            if len(all_labels) % 50 == 0:
                emit(phase='LABELS', role=role, origins_completed=len(all_labels), rows=sum(r['n'] for r in all_labels), seconds=time.monotonic()-budget.start)
        del quotes
    label_role = 'TRAIN' if roles == ['TRAIN'] else 'CAL'
    dest = OUT / 'labels' / label_role / 'label_manifest.json'
    result = dict(status='COMPLETE', abi=ABI, population_receipt_id=POP, origins=all_labels,
        D=len(all_labels), R=sum(r['n'] for r in all_labels), known=sum(r['known_count'] for r in all_labels),
        complete=sum(r['complete_count'] for r in all_labels), mature=sum(r['mature_count'] for r in all_labels),
        scope_counts={role:dict(D=sum(r['role']==role for r in all_labels), R=sum(r['n'] for r in all_labels if r['role']==role),
                    mature=sum(r['mature_count'] for r in all_labels if r['role']==role)) for role in roles},
        price_mode=PRICE_MODE, rounding_limit='NOT_VENDOR_HIGH_LOW_EXACT_EQUIVALENCE',
        future_unknown_rows_preserved=True, committed_at=stamp())
    dump(dest, result); budget.account(dest)
    if selected:
        target = OUT / 'selected_w_paths.jsonl'; temp = target.with_suffix('.tmp'); count = 0
        with temp.open('w', encoding='utf-8') as f:
            for r in manifest['origins']:
                if group[(r['role'], r['origin_id'])]:
                    for event in read(OUT / 'selected_events' / f"{r['origin_id']}.json"):
                        f.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + '\n'); count += 1
        temp.replace(target)
        dump(OUT / 'selected_w_paths_receipt.json', dict(status='COMPLETE_ALL_SEALED_SELECTED_EVENTS', abi=ABI,
            population_receipt_id=POP, path=str(target), events=count, selected_keys_count=len(selected),
            primary_reference='OBSERVED_EXECUTABLE_OPEN_REFERENCE', delayed_entry_extends_W=False,
            scheduled_reference='NESTED_SEPARATE_DIAGNOSTIC', price_mode=PRICE_MODE,
            actual_account_fill=False, no_fit=True, committed_at=stamp()))
    return result


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('phase', choices=['train','cal']); ap.add_argument('--check-id', required=True)
    args = ap.parse_args()
    if args.check_id not in DOC.read_text(encoding='utf-8-sig'):
        raise PermissionError('ON_DISK_ACTUAL_PRE_REQUIRED')
    try:
        binding, grant = authority(args.phase)
        budget = Budget(grant, grant['actual_phase_initial_cap_seconds'])
        manifest = freeze_origins(binding, grant, budget)
        if args.phase == 'train':
            result = labels_phase(['TRAIN'], manifest, binding, grant, budget)
            state = 'COMPLETE_TRAIN_LABELS_AND_ALL_ORIGIN_INPUTS'
        else:
            pred = read(ROOT / grant['cal_labels_require'])
            selected = read(pred['selected_keys_path'])
            if isinstance(selected, dict):
                selected = selected['keys']
            result = labels_phase(['CAL1','CAL2'], manifest, binding, grant, budget, selected)
            state = 'COMPLETE_CAL_LABELS_AND_SELECTED_W_PATHS'
        receipt = dict(status=state, phase=args.phase, work_id=WORK, input_manifest=str(OUT/'input_manifest.json'),
            label_manifest=str(OUT/'labels'/('TRAIN' if args.phase=='train' else 'CAL')/'label_manifest.json'),
            label_D=result['D'], label_R=result['R'], known=result['known'], mature=result['mature'],
            seconds=time.monotonic()-budget.start, output_bytes=budget.bytes, price_mode=PRICE_MODE,
            fits=0, api=0, accounts=0, committed_at=stamp())
        dump(OUT / f'{args.phase}_completion_receipt.json', receipt)
        post(args.check_id, 'PASS', json.dumps(receipt, ensure_ascii=False), '允许消费本已提交输入；不形成经济或策略晋级结论。')
        emit(**receipt)
    except CommittedPause as e:
        result = dict(status=str(e), phase=args.phase, work_id=WORK, committed_partitions_reusable=True, at=stamp())
        dump(OUT / f'{args.phase}_progress.json', result)
        post(args.check_id, 'INCONCLUSIVE', str(e), '仅暂停于已提交分区，非经济失败。')
        emit(**result)
    except Exception as e:
        post(args.check_id, 'INCONCLUSIVE', repr(e), '保留已提交分区；按直接错误归因后最小修复，不把基础设施错当经济终态。', 'UNKNOWN')
        raise


if __name__ == '__main__':
    main()
