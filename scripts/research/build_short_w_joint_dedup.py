"""One date-only regrouping of published JOINT events; no fit or price replay."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT/'docs/research/short_w_joint_dedup_20260907.md'
SRC = ROOT/'output/research_once/short_w_joint_recent_buy_w_v1_20260907'
OUT = ROOT/'output/research_once/short_w_joint_dedup_v1_20260907'
WORK = 'joint_dedup'


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def post(check, result, occurred, evidence, attribution, impact, value='KEEP'):
    with DOC.open('a', encoding='utf-8') as f:
        f.write(f'\n[Check record]\nCheck ID: {check}\nOutcome: {result}\nIssue observed: {occurred}'
                f'\nDirect evidence: {evidence}\nFailure attribution: {attribution}\nImplication: {impact}'
                f'\nCheck disposition: {value}\nRe-run condition: a new grouping rule or a change in the actual input boundary is treated as a separate question; this result is not re-run.\n')


def membership(events):
    """Outcome fields are never used for inclusion or anchor replacement."""
    kept, dropped, unresolved, ledger = [], [], [], []
    anchors, unknown_stocks = {}, set()
    for e in sorted(events, key=lambda x: (x['origin_id'], x['stable_id'])):
        stock = e['stable_id']
        start, end = e.get('scheduled_entry_session'), e.get('W_end')
        anchor = anchors.get(stock)
        row = {k: e.get(k) for k in ('episode_id', 'origin_id', 'stable_id', 'stock_code',
                                    'scheduled_entry_session', 'W_end', 'path_status', 'entry_status')}
        if stock in unknown_stocks or not start or not end:
            unknown_stocks.add(stock)
            unresolved.append(e)
            row['decision'] = 'UNRESOLVED_DEDUP_CALENDAR'
        elif anchor is None or start > anchor['W_end']:
            anchors[stock] = e
            kept.append(e)
            row['decision'] = 'KEEP_FIRST_NONOVERLAP'
        else:
            dropped.append(e)
            row['decision'] = 'DROP_OVERLAPS_RETAINED_ORIGINAL_W'
            row['anchor_episode_id'] = anchor.get('episode_id')
            row['anchor_origin_id'] = anchor['origin_id']
            row['anchor_W_end'] = anchor['W_end']
        ledger.append(row)
    return kept, dropped, unresolved, ledger


def fixture():
    check = 'dedup_boundary_fixture'
    def e(key, stock, origin, start, end, **other):
        return dict(episode_id=key,stable_id=stock,origin_id=origin,
                    scheduled_entry_session=start,W_end=end,**other)
    events = [
        e('A1','A','2024-12-27','2024-12-30','2025-01-10',entry_status='PENDING',bilateral_complete=False),
        e('A2','A','2025-01-02','2025-01-03','2025-01-16',max_high_return=9.0),
        e('A3','A','2025-01-09','2025-01-10','2025-01-23'),
        e('A4','A','2025-01-10','2025-01-13','2025-01-24',max_high_return=-.4),
        e('B1','B','2025-01-02','2025-01-03','2025-01-16'),
        e('C1','C','2025-01-02','2025-01-03',None),
        e('C2','C','2025-01-30','2025-01-31','2025-02-13')]
    try:
        kept,dropped,unresolved,ledger = membership(list(reversed(events)))
        assert {x['episode_id'] for x in kept} == {'A1','A4','B1'}
        assert {x['episode_id'] for x in dropped} == {'A2','A3'}
        assert {x['episode_id'] for x in unresolved} == {'C1','C2'}
        assert all(x['anchor_episode_id']=='A1' for x in ledger if x['decision'].startswith('DROP'))
        OUT.mkdir(parents=True,exist_ok=True)
        write(OUT/'dedup_boundary_proof.json',dict(status='PASS',synthetic_only=True,ledger=ledger))
        post(check,'PASS','No','7 synthetic events: A1/A4/B1 retained; A2/A3 dropped, both anchored to A1; C1/C2 calendar-unresolved.',
             'NONE','The fixed dedup function can be used for a single saved event summary.')
    except Exception as exc:
        post(check,'FAIL','Yes',repr(exc),'TEST_ITSELF','The real comparison is not run until the new grouping function is fixed.','REMOVE')
        raise
    print('PASS synthetic dedup boundary; no economic data read.')


def pct(x): return 'NA' if x is None else f'{x*100:.2f}%'
def num(x): return 'NA' if x is None else f'{x:.2f}'


def actual():
    check = 'dedup_actual_comparison'
    if (OUT/'terminal.json').exists():
        print('ALREADY_COMPLETE_NO_RECOMPUTE')
        return
    if json.loads((OUT/'dedup_boundary_proof.json').read_text(encoding='utf-8'))['status'] != 'PASS':
        raise RuntimeError('NEW_DEDUP_BOUNDARY_PROOF_REQUIRED')
    started = time.monotonic()
    try:
        # Only this pure summarizer is called; the upstream actual/fixture/read-price routines are not called.
        from build_short_w_joint_recent_buy_w import summarize
        with (SRC/'all_selected_w_paths.jsonl').open(encoding='utf-8-sig') as f:
            events = [json.loads(line) for line in f if line.strip()]
        before = json.loads((SRC/'opportunity_report.json').read_text(encoding='utf-8-sig'))
        kept,dropped,unresolved,ledger = membership(events)
        windows = {}
        for name, old in before['windows'].items():
            start,end = old['start'],old['end']
            subset = [e for e in kept if start <= e['origin_id'] <= end]
            unknown = [e for e in unresolved if start <= e['origin_id'] <= end]
            new = summarize(subset)
            windows[name] = dict(start=start,end=end,before=old,after=new,
                unresolved_dedup_calendar=len(unknown),
                delta=dict(MFE_mean=new['metrics']['max_high_return']['mean']-old['metrics']['max_high_return']['mean'] if subset else None,
                    MFE_median=new['metrics']['max_high_return']['median']-old['metrics']['max_high_return']['median'] if subset else None,
                    hit5_fraction=new['thresholds']['5']['fraction_of_complete']-old['thresholds']['5']['fraction_of_complete'] if subset else None))
        for filename,values in [('membership_ledger.jsonl',ledger),
                                ('retained_event_keys.jsonl',[{k:e.get(k) for k in ('episode_id','origin_id','stable_id','scheduled_entry_session','W_end')} for e in kept])]:
            with (OUT/filename).open('w',encoding='utf-8') as f:
                for row in values: f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        report = dict(status='COMPLETE_DESCRIPTIVE_SAME_STOCK_NONOVERLAP_COMPARISON',work_id=WORK,
            source_events=str(SRC/'all_selected_w_paths.jsonl'),source_before_report=str(SRC/'opportunity_report.json'),
            method='GLOBAL_DATE_ONLY_GREEDY_FIRST_EVENT_LAST_RETAINED_ORIGINAL_W; EQUALITY_OVERLAPS; DROPPED_DOES_NOT_EXTEND',
            all_events=len(events),retained_events=len(kept),dropped_events=len(dropped),
            unresolved_calendar_events=len(unresolved),unique_stocks_before=len({e['stable_id'] for e in events}),
            unique_stocks_retained=len({e['stable_id'] for e in kept}),
            windows=windows,
            membership_status=dict(Counter(row['decision'] for row in ledger)),
            boundary='Daily state unchanged; no-entry/censored participate in date-only grouping; complete-path denominator separate; globally dedup before period slicing.',
            limitations=['Descriptive nonoverlap events are not IID: other stocks and market dates remain correlated.',
                         'No matched market baseline; no alpha or statistical significance claim.',
                         'Maximum high is an ex post opportunity ceiling, not realized or T+1-guaranteed income.',
                         'Prior 2023/2024 role-cutoff censoring remains unchanged; 2026 is partial through August 21.',
                         'Historical data were already seen; this is a causal descriptive development diagnostic, not untouched TEST.'],
            new_fits=0,raw_price_reads=0,price_API=0,new_exit_policies=0,accounts=0)
        write(OUT/'dedup_comparison.json',report)
        lines = ['# JOINT buy events: same-stock overlapping-window deduplication comparison','',
            'This report compares the opportunity distribution within the original two-week window after stock selection. The original daily selections and states are unchanged.','',
            f'Original {len(events)} events → {len(kept)} retained, {len(dropped)} overlapping events dropped, {len(unresolved)} calendar-unresolved. '
            f'Original stocks: {report["unique_stocks_before"]}; after retention: {report["unique_stocks_retained"]}.','',
            'For each stock the event with the earliest signal is retained first; a later event is retained only if its scheduled entry day falls after the end of the original W of the most recently retained event. Dropped events do not extend the window; deduplication runs over the full path before slicing by year. Unfilled events and events whose window has not ended also take part in the date-based deduplication.','',
            '| Window | Complete/all: original→dedup | Mean max gain: original→dedup | Median max gain: original→dedup | Reaching 5%: original→dedup | Count reaching 5% after dedup |',
            '|---|---:|---:|---:|---:|---:|']
        for name,w in windows.items():
            a,b=w['before'],w['after']; am,bm=a['metrics'],b['metrics']
            lines.append(f'| {name} ({w["start"]}—{w["end"]}) | {a["complete_paths"]}/{a["all_selected_keys"]} → {b["complete_paths"]}/{b["all_selected_keys"]} | '
                         f'{pct(am["max_high_return"]["mean"])} → {pct(bm["max_high_return"]["mean"])} | '
                         f'{pct(am["max_high_return"]["median"])} → {pct(bm["max_high_return"]["median"])} | '
                         f'{pct(a["thresholds"]["5"]["fraction_of_complete"])} → {pct(b["thresholds"]["5"]["fraction_of_complete"])} | '
                         f'{b["thresholds"]["5"]["complete_hit_count"]}/{b["complete_paths"]} |')
        lines += ['', 'The denominator of the gains and shares above is complete paths; unresolved events are not counted as having no gain. The JSON also lists the 3%/5%/10% hit counts, the complete-path shares, and the confirmed-hit lower bound and coverage over all retained events.','',
            '| Window | Median first-peak day: original→dedup | Mean pre-peak minimum return: original→dedup | Median pre-peak minimum return: original→dedup | Dedup pre-peak drop ≥5%/determinable | Median full-window minimum return: original→dedup |',
            '|---|---:|---:|---:|---:|---:|']
        for name,w in windows.items():
            a,b=w['before'],w['after']; am,bm=a['metrics'],b['metrics']
            ap,bp=am['min_low_before_peak_exclusive_return'],bm['min_low_before_peak_exclusive_return']
            lines.append(f'| {name} | {num(am["first_peak_day"]["median"])} → {num(bm["first_peak_day"]["median"])} | '
                         f'{pct(ap["mean"])} → {pct(bp["mean"])} | {pct(ap["median"])} → {pct(bp["median"])} | '
                         f'{b["strict_before_peak_down_5pct_count"]}/{bp["n"]} | '
                         f'{pct(am["min_low_return"]["median"])} → {pct(bm["min_low_return"]["median"])} |')
        lines += ['', '"Pre-peak" strictly excludes the day on which the high occurs and assumes no intraday order of high and low; when the peak falls on the entry day there may be no determinable pre-peak interval. The first-peak day follows the original upstream day index; a delayed entry does not restart the original window.',
            '', '## Measurement notes', '',
            '- Deduplication only removes repeated counting from overlapping W windows of the same stock; same-day co-movement across different stocks remains, so the retained count is not a count of independent samples.',
            '- The maximum gain is an ex post opportunity range; the high on the buy day may not be sellable under T+1. The day2plus and executable-open upper-bound statistics are kept in the JSON.',
            '- The original 2023/2024 year-end label censoring is kept as is; 2026 data end on 2026-08-21, so August 2026 is incomplete.',
            '', 'Per-event retain/drop decisions and anchors: membership_ledger.jsonl; full metrics: dedup_comparison.json.']
        (OUT/'dedup_comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        terminal = dict(status=report['status'],work_id=WORK,report=str(OUT/'dedup_comparison.json'),
            readable_report=str(OUT/'dedup_comparison.md'),all_events=len(events),retained_events=len(kept),
            dropped_events=len(dropped),unresolved_calendar_events=len(unresolved),seconds=time.monotonic()-started,
            new_fits=0,raw_price_reads=0,price_API=0,accounts=0,
            at=datetime.now().astimezone().isoformat(timespec='seconds'))
        write(OUT/'terminal.json',terminal)
        full=windows['FULL_FUTURE']
        evidence=json.dumps(dict(all_events=len(events),retained=len(kept),dropped=len(dropped),
            unresolved=len(unresolved),full_delta=full['delta']),ensure_ascii=False)
        post(check,'PASS','Yes: same-stock overlap causes repeated weighting; see the direct comparison for its size',evidence,'DOWNSTREAM',
             'The original daily event statistics are kept but cannot serve as a count of independent opportunities; descriptive conclusions are revised on the deduplicated distribution, and no strategy is promoted.')
        print(json.dumps(dict(terminal=terminal,windows={k:dict(before_complete=w['before']['complete_paths'],
            after_complete=w['after']['complete_paths'],after_all=w['after']['all_selected_keys'],
            mean=w['after']['metrics']['max_high_return']['mean'],median=w['after']['metrics']['max_high_return']['median'],
            hit5=w['after']['thresholds']['5'],peak_day=w['after']['metrics']['first_peak_day'],
            pre_peak_low=w['after']['metrics']['min_low_before_peak_exclusive_return'],
            pre_peak_down5=w['after']['strict_before_peak_down_5pct_count'],delta=w['delta']) for k,w in windows.items()}),ensure_ascii=False))
    except Exception as exc:
        post(check,'INCONCLUSIVE','Undetermined',repr(exc),'TEST_ITSELF',
             'After the exact interface is fixed, the same dedup question continues under a new check ID; a tooling failure is not treated as a strategy outcome.','REMOVE')
        raise


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=['fixture','actual'])
    args=parser.parse_args()
    fixture() if args.phase=='fixture' else actual()
