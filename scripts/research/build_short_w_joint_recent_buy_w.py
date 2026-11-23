"""Only new 2025+ own W paths; published earlier CAL events remain untouched."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import build_short_w_joint_inputs as paths

ROOT = Path(__file__).resolve().parents[2]
WORK = 'POOL-20260907-SHORT-JOINT-RECENT-BUY-W-DIAG-01'
METHOD = 'SHORT_W_JOINT_TRAIN_CENTERED_EVENT_DISTRIBUTION_BUY_V1'
OUT = ROOT / 'output/research_once/short_w_joint_recent_buy_w_v1_20260907'
GRANT = ROOT / 'docs/research/short_w_joint_recent_buy_w_grant_20260907.json'
START, END = '2025-01-01', '2026-08-21'
spec = importlib.util.spec_from_file_location('_joint_recent_selected_path_context_20260907',
                                            Path(__file__).with_name('build_short_w_joint_selected_exit_cal_20260907.py'))
context = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = context
spec.loader.exec_module(context)
# Isolated configuration of already-proven source-only selected context functions.
context.OUT, context.ROLES = OUT, {'RECENT': (START, END)}
read, save, jsonl, clean = context.read, context.save, context.jsonl, context.clean


def token(event): return (event['origin_id'], event['stable_id'])


def merge_events(old, new, expected):
    all_events = list(old) + list(new)
    tokens = [token(e) for e in all_events]
    if len(tokens) != len(set(tokens)) or set(tokens) != set(expected):
        raise ValueError('EXACT_ALL_SELECTED_KEYS_ONCE_REQUIRED')
    return sorted(all_events, key=token)


def recent_event(key, sources):
    class BoundQuotes:
        cutoff = END
        def event_rows(self, row, data, i): return sources.path_bars(key), sources.context(key)
    row = dict(scheduled_entry_session=key['scheduled_entry_session'], W_end=key['cap_session'], T10_end=key['T10_end'])
    identity = dict(episode_id=key['episode_id'], origin_id=key['origin_session'], stable_id=key['stable_instrument_key'],
                    stock_code=key['stock_code'], arm=METHOD, role='RECENT', split='RECENT', rank=key['rank'])
    return paths.selected_event(identity, row, None, 0, BoundQuotes(), sources.calendar)


def freeze(grant):
    destination = OUT / 'candidate_selection.json'
    if destination.exists(): return read(destination)
    buy_selection = read(ROOT / grant['buy_seal'])
    all_selected = read(buy_selection['selected_keys_path'])
    new_selected = [k for k in all_selected if k['arm'] == METHOD and START <= k['origin_id'] <= END]
    if len(new_selected) > grant['max_new_keys']: raise ValueError('NEW_KEY_CAP')
    selected_by_day = defaultdict(list)
    for key in new_selected: selected_by_day[key['origin_id']].append(key)
    daily = {r['origin_session']: r for r in buy_selection['daily']}
    keys = []
    for day, selected in sorted(selected_by_day.items()):
        selection = read(daily[day]['selection_receipt'])
        member = {r['stable_id']: r for r in selection['selected_JOINT']}
        for chosen in selected:
            row = member[chosen['stable_id']]
            keys.append(dict(episode_id='RECENT__' + day + '__' + chosen['stock_code'], origin_session=day,
                stock_code=chosen['stock_code'], source_code=chosen['stock_code'], split='RECENT',
                stable_instrument_key=chosen['stable_id'], rank=chosen['rank'], buy_model_id=METHOD,
                scheduled_entry_session=selection['scheduled_entry_session'], cap_session=selection['W_end'],
                T10_end=selection['T10_end'], original_L=selection['L'],
                origin_qcap_1x=int(row['qcap_1x']), origin_qcap_2x=int(row['qcap_2x']),
                origin_raw_close=row['raw_close'], selection_receipt=daily[day]['selection_receipt']))
    calendar_record = read(buy_selection['calendar_receipt'])
    jsonl(OUT / 'candidate_keys.jsonl', keys)
    result = dict(status='FROZEN_ONLY_NEW_RECENT_OWN_KEYS', work_id=WORK, buy_model_id=METHOD,
        new_key_count=len(keys), new_origin_count=len(selected_by_day), all_selected_key_count=len(all_selected),
        candidate_keys_path=str(OUT / 'candidate_keys.jsonl'), all_selected_keys_path=buy_selection['selected_keys_path'],
        buy_selection_file=str(ROOT / grant['buy_seal']), calendar=calendar_record['calendar'],
        calendar_file=buy_selection['calendar_receipt'], model_file=buy_selection['model_receipt'],
        center_file=buy_selection['center_receipt'], price_cutoff=grant['price_cutoff'],
        scheduled_tail_independent_of_quotes=True, at=context.stamp())
    save(destination, result)
    return result


def describe(values):
    values = np.asarray([float(v) for v in values if context.number(v) is not None], float)
    if not len(values): return dict(n=0, mean=None, median=None, q10=None, q25=None, q75=None, q90=None)
    return dict(n=len(values), mean=float(values.mean()), median=float(np.median(values)),
        q10=float(np.quantile(values, .1)), q25=float(np.quantile(values, .25)),
        q75=float(np.quantile(values, .75)), q90=float(np.quantile(values, .9)),
        minimum=float(values.min()), maximum=float(values.max()),
        positive=int((values > 0).sum()), negative=int((values < 0).sum()), zero=int((values == 0).sum()))


def lower_bound(event):
    return event.get('observed_lower_bound_max_high_return', event.get('observed_max_high_lower_bound'))


def summarize(events):
    complete = [e for e in events if e.get('bilateral_complete') and e.get('max_high_return') is not None]
    complete_tokens = {token(e) for e in complete}
    metric_names = ['max_high_return', 'max_close_return', 'min_low_return', 'first_peak_day',
        'first_peak_day_since_scheduled', 'min_low_before_peak_exclusive_return', 'first_low_session',
        'day2plus_high_ceiling', 'best_legal_open_return', 'first_hit_3pct_day', 'first_hit_5pct_day', 'first_hit_10pct_day',
        'hindsight_best_legal_open_net_slot_return_1x', 'hindsight_best_legal_open_net_slot_return_2x',
        'hindsight_high_net_slot_ceiling_1x', 'high_vs_best_legal_open_net_slot_gap_1x']
    metrics = {name: describe([e.get(name) for e in complete]) for name in metric_names if name != 'first_low_session'}
    date_means = defaultdict(list)
    for e in complete: date_means[e['origin_id']].append(e['max_high_return'])
    thresholds = {}
    for threshold in (.03, .05, .10):
        n = sum(e['max_high_return'] >= threshold for e in complete)
        known = sum((e.get('max_high_return') if e.get('max_high_return') is not None else lower_bound(e) or -np.inf) >= threshold for e in events)
        thresholds[str(int(threshold * 100))] = dict(complete_hit_count=n, complete_denominator=len(complete),
            fraction_of_complete=n / len(complete) if complete else None, all_selected_denominator=len(events),
            known_hit_count_all_keys=known, known_hit_fraction_all_keys=known / len(events) if events else None,
            unresolved_keys=len(events)-len(complete))
    return dict(all_selected_keys=len(events), origin_count=len({e['origin_id'] for e in events}),
        complete_paths=len(complete), incomplete_or_no_entry=len(events)-len(complete),
        complete_coverage=len(complete)/len(events) if events else None,
        path_status=dict(Counter(e.get('path_status', 'ABSENT') for e in events)),
        entry_status=dict(Counter(e.get('entry_status', 'ABSENT') for e in events)),
        thresholds=thresholds, metrics=metrics, date_equal_MFE=describe([np.mean(v) for v in date_means.values()]),
        partial_observed_MFE_lower_bounds=describe([lower_bound(e) for e in events if token(e) not in complete_tokens]),
        entry_day_only_5pct=sum(e.get('day1_only_5pct_opportunity') is True for e in complete),
        down_5pct_count=sum(e.get('min_low_return') is not None and e['min_low_return'] <= -.05 for e in complete),
        strict_before_peak_down_5pct_count=sum(e.get('min_low_before_peak_exclusive_return') is not None
                                             and e['min_low_before_peak_exclusive_return'] <= -.05 for e in complete),
        peak_low_chronology=dict(Counter(e.get('peak_low_chronology', 'UNKNOWN') for e in complete)),
        same_bar_high_low_order='UNKNOWN_NO_INTRADAY_ORDER_INFERENCE', actual_account_returns=False)


def dependencies(events, calendar):
    stocks = Counter(e['stable_id'] for e in events)
    per_stock = defaultdict(list)
    per_week = defaultdict(list)
    for e in events:
        per_stock[e['stable_id']].append(e)
        week = pd.Timestamp(e['origin_id']).isocalendar()
        per_week[f'{week.year}-W{week.week:02d}'].append(e)
    overlap_pairs, overlapped_events = 0, 0
    for values in per_stock.values():
        active = []
        for event in sorted(values, key=lambda e: e['origin_id']):
            start = event.get('scheduled_entry_session')
            if not start: continue
            active = [end for end in active if end and end >= start]
            overlap_pairs += len(active)
            overlapped_events += bool(active)
            active.append(event.get('W_end'))
    return dict(selected_events=len(events), unique_stocks=len(stocks),
        repeated_stocks=sum(n > 1 for n in stocks.values()), events_on_repeated_stocks=sum(n for n in stocks.values() if n > 1),
        top_repeated_stock_counts=[dict(stable_id=stock, events=count) for stock, count in stocks.most_common(15)],
        same_stock_overlapping_original_W_pairs=overlap_pairs, events_with_previous_same_stock_overlapping_W=overlapped_events,
        weekly_batches=len(per_week), weekly_batch_key_count_distribution=describe([len(v) for v in per_week.values()]),
        inferential_status='DESCRIPTIVE_CORRELATED_OVERLAPPING_EVENTS; NO_IID_STANDARD_ERRORS_OR_SIGNIFICANCE_CLAIM')


def reporting(events, frozen, prior):
    windows = [('FULL_FUTURE', '2023-01-01', END),
        *[(f'YEAR_{year}' if year < 2026 else 'YEAR_2026_PARTIAL', f'{year}-01-01', f'{year}-12-31' if year < 2026 else END) for year in range(2023,2027)],
        ('RECENT12_OLDER', '2024-08-01', '2025-07-31'), ('RECENT12_LATEST', '2025-08-01', '2026-07-31'),
        ('RECENT24', '2024-08-01', '2026-07-31'), ('AUG2026_PARTIAL', '2026-08-01', END)]
    reports = {name: dict(start=start, end=end, **summarize([e for e in events if start <= e['origin_id'] <= end]))
               for name,start,end in windows}
    monthly, weekly = {}, {}
    month_groups, week_groups = defaultdict(list), defaultdict(list)
    for event in events:
        month_groups[event['origin_id'][:7]].append(event)
        week = pd.Timestamp(event['origin_id']).isocalendar()
        week_groups[f'{week.year}-W{week.week:02d}'].append(event)
    monthly = {month: summarize(values) for month,values in sorted(month_groups.items())}
    weekly = {week: summarize(values) for week,values in sorted(week_groups.items())}
    flat = lambda name,row: dict(period=name, all_selected_keys=row['all_selected_keys'], complete_paths=row['complete_paths'],
        complete_coverage=row['complete_coverage'], MFE_mean=row['metrics']['max_high_return']['mean'],
        MFE_median=row['metrics']['max_high_return']['median'], MFE_q10=row['metrics']['max_high_return']['q10'],
        MFE_q90=row['metrics']['max_high_return']['q90'], max_close_median=row['metrics']['max_close_return']['median'],
        min_low_median=row['metrics']['min_low_return']['median'], peak_day_median=row['metrics']['first_peak_day']['median'],
        hit_5pct_count=row['thresholds']['5']['complete_hit_count'], hit_5pct_fraction=row['thresholds']['5']['fraction_of_complete'],
        all_selected_known_hit_5pct_fraction=row['thresholds']['5']['known_hit_fraction_all_keys'],
        entry_day_only_5pct=row['entry_day_only_5pct'], strict_before_peak_down_5pct_count=row['strict_before_peak_down_5pct_count'])
    for name, grouping in [('window_opportunity_summary', reports), ('monthly_opportunity_distribution',monthly), ('weekly_opportunity_distribution',weekly)]:
        pd.DataFrame([flat(key,value) for key,value in grouping.items()]).to_csv(OUT/(name+'.csv'),index=False)
    report = dict(status='COMPLETE_DESCRIPTIVE_SELECTED_W_OPPORTUNITY_REPORT', work_id=WORK, method_id=METHOD,
        windows=reports, monthly=monthly, weekly=weekly, dependency=dependencies(events,frozen['calendar']),
        monthly_rows_are='SELECTION_OPPORTUNITY_DISTRIBUTIONS_NOT_ACCOUNT_MONTHLY_RETURNS',
        prior_CAL_reuse=dict(manifest=str(ROOT/read(GRANT)['prior_CAL']), event_path=prior['selected_w_paths_path'],
                            keys=prior['candidate_count'], old_prices_read=0, old_events_recomputed=0),
        censoring_policy='2023/2024 original role-cutoff censoring retained unchanged; 2025+ uses 2026-08-21 cutoff; always show coverage',
        boundary='OBSERVED_EXECUTABLE_OPEN_REFERENCE_TO_ORIGINAL_SCHEDULED_W_END; DELAY_DOES_NOT_RESTART_W',
        Tplus1='Entry-day high is descriptive; legal-open and day2plus diagnostics are separate, no future argmax trade',
        promotion='NONE; positive MFE alone does not certify usable net opportunity or rescue failed Q exit',
        fits=0, new_sell_policies=0, accounts=0, price_API=0, at=context.stamp())
    save(OUT/'opportunity_report.json', report)
    percent = lambda v: 'NA' if v is None else f'{100*v:.2f}%'
    lines = ['# JOINT selected W opportunity', '',
        'These are observed-reference stock opportunity distributions. They are not realized trades or account returns.', '',
        '| Window | All keys | Complete | MFE mean | MFE median | >=5% / complete | Low median | Peak day median |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,row in reports.items():
        lines.append(f"| {name} | {row['all_selected_keys']} | {row['complete_paths']} | {percent(row['metrics']['max_high_return']['mean'])} | {percent(row['metrics']['max_high_return']['median'])} | {percent(row['thresholds']['5']['fraction_of_complete'])} | {percent(row['metrics']['min_low_return']['median'])} | {row['metrics']['first_peak_day']['median']} |")
    lines += ['', 'All original selected keys remain in the denominators. Incomplete/no-entry rows do not become zero returns or known misses. Partial observed maxima are lower bounds.',
        '', 'The original 2023/2024 year-end censoring is retained exactly; later 2025+ events use the August 21, 2026 price cutoff. This affects complete-path denominators.',
        '', 'Monthly and weekly CSV rows describe selection opportunities; their means must not be summed or compounded into account returns.',
        '', json.dumps(report['dependency'],ensure_ascii=False), '',
        'Full output: opportunity_report.json; window_opportunity_summary.csv; monthly_opportunity_distribution.csv; weekly_opportunity_distribution.csv.']
    (OUT/'opportunity_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return report


def actual():
    grant = read(GRANT)
    budget = context.Budget(grant)
    frozen = freeze(grant)
    keys = context.rows(frozen['candidate_keys_path'])
    sources = context.Sources(frozen['calendar'],keys,budget)
    new = []
    for i,key in enumerate(keys,1):
        budget.boundary()
        dest = OUT/'new_events'/(key['episode_id']+'.json')
        if dest.exists(): event = read(dest)
        else:
            event = recent_event(key,sources)
            save(dest,event)
            budget.account(dest)
        new.append(event)
        if i%100==0: print(json.dumps(dict(phase='ONLY_NEW_RECENT_W',events=i,total=len(keys),seconds=time.monotonic()-budget.start)),flush=True)
    jsonl(OUT/'new_selected_w_paths.jsonl',new)
    prior = read(ROOT/grant['prior_CAL'])
    old = context.rows(prior['selected_w_paths_path'])
    expected = [(k['origin_id'],k['stable_id']) for k in read(frozen['all_selected_keys_path']) if k['arm']==METHOD]
    events = merge_events(old,new,expected)
    jsonl(OUT/'all_selected_w_paths.jsonl',events)
    report = reporting(events,frozen,prior)
    byte_count = sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
    if byte_count > grant['max_output_bytes']: raise context.train.Pause('OUTPUT_CAP_BEFORE_TERMINAL')
    terminal = dict(status='COMPLETE_RECENT_AND_FULL_JOINT_BUY_W_DIAGNOSTIC',work_id=WORK,method_id=METHOD,
        new_keys=len(new),prior_reused_keys=len(old),all_keys=len(events),
        new_events_path=str(OUT/'new_selected_w_paths.jsonl'),all_events_path=str(OUT/'all_selected_w_paths.jsonl'),
        report=str(OUT/'opportunity_report.json'),calendar_file=frozen['calendar_file'],
        price_cutoff=grant['price_cutoff'],source_contexts=len(sources.first),
        source_context_files=[str(OUT/'source_contexts'/role/(code+'.json')) for role,code in sorted(sources.first)],
        window_coverage={key:dict(all_keys=row['all_selected_keys'],complete=row['complete_paths'],path_status=row['path_status']) for key,row in report['windows'].items()},
        output_bytes_before_terminal=byte_count,seconds=time.monotonic()-budget.start,
        fits=0,new_exit=0,accounts=0,price_API=0,prior_prices_read=0,prior_events_recomputed=0,
        automatic_economic_PASS=False,at=context.stamp())
    save(OUT/'terminal.json',terminal)
    return dict(terminal=terminal,windows={key:dict(mean=row['metrics']['max_high_return']['mean'],
        median=row['metrics']['max_high_return']['median'],hit5=row['thresholds']['5']['fraction_of_complete'],
        low_median=row['metrics']['min_low_return']['median'],peak_day=row['metrics']['first_peak_day']['median']) for key,row in report['windows'].items()})


def fixture():
    class SyntheticSources:
        calendar=['2026-08-20','2026-08-21','2026-08-24','2026-08-25','2026-08-26','2026-08-27',
                  '2026-08-28','2026-08-31','2026-09-01','2026-09-02','2026-09-03','2026-09-04']
        def path_bars(self,key):
            if key['scheduled_entry_session']>END:return []
            return [dict(session='2026-08-21',adjusted_open=10.,high=10.6,low=9.8,close=10.4,raw_open=10.,
                         buy_status='OK',sell_status='OK',origin_qcap_1x=300,origin_qcap_2x=300)]
        def context(self,key):return dict(source_receipt='SYNTHETIC_ONLY')
    base=dict(stock_code='000001',stable_instrument_key='SYNTH',rank=1)
    a=recent_event(dict(base,episode_id='A',origin_session='2026-08-20',scheduled_entry_session='2026-08-21',
                        cap_session='2026-09-03',T10_end='2026-09-03'),SyntheticSources())
    b=recent_event(dict(base,episode_id='B',origin_session='2026-08-21',scheduled_entry_session='2026-08-24',
                        cap_session='2026-09-04',T10_end='2026-09-04'),SyntheticSources())
    assert a['W_end']=='2026-09-03' and not a['bilateral_complete'] and a['max_high_return'] is None
    assert abs(lower_bound(a)-.06)<1e-12 and a['observed_entry_session']=='2026-08-21'
    assert b['W_end']=='2026-09-04' and b['path_status']=='PENDING_UNEXECUTABLE' and b['observed_entry_session'] is None
    old=dict(origin_id='2024-12-31',stable_id='OLD',payload='UNCHANGED_OLD_EVENT')
    combined=merge_events([old],[a,b],[token(old),token(a),token(b)])
    assert combined[0] is old and len(combined)==3
    rejected=False
    try:merge_events([old],[a],[token(old),token(a),token(b)])
    except ValueError:rejected=True
    assert rejected
    result=dict(status='PASS',work_id=WORK,new_role=context.ROLES,
        cutoff_tail_not_shortened=True,partial_observed_lower_bound_retained=True,noentry_tail_preserved=True,
        old_event_identity_reused=True,missing_immutable_key_rejected=True,
        actual_price_reads=0,actual_models_read=0,old_numeric_proofs_retested=False,at=context.stamp())
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=['fixture','actual'])
    args=parser.parse_args()
    print(json.dumps(fixture() if args.phase=='fixture' else actual(),ensure_ascii=False),flush=True)
