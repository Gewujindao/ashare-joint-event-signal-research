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
WORK = 'POOL-20260907-JOINT-DEDUP-01'


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def post(check, result, occurred, evidence, attribution, impact, value='KEEP'):
    with DOC.open('a', encoding='utf-8') as f:
        f.write(f'\n[检验后]\n检验ID: {check}\n结论: {result}\n担忧是否发生: {occurred}'
                f'\n直接证据: {evidence}\n故障归因: {attribution}\n对决策的影响: {impact}'
                f'\n检验价值: {value}\n复发条件: 仅新归组规则或实际输入边界改变时另立问题；本结果不重复执行。\n')


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
    check = 'JOINT-DEDUP-GREEDY-BOUNDARY-20260907-01'
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
        post(check,'PASS','否','7 个合成事件：保留 A1/A4/B1，删除 A2/A3 且锚点均 A1，C1/C2 为日历未决。',
             'NONE','允许既定去重函数用于一次保存事件汇总。')
    except Exception as exc:
        post(check,'FAIL','是',repr(exc),'TEST_ITSELF','不运行真实比较，先修复新增归组函数。','REMOVE')
        raise
    print('PASS synthetic dedup boundary; no economic data read.')


def pct(x): return 'NA' if x is None else f'{x*100:.2f}%'
def num(x): return 'NA' if x is None else f'{x:.2f}'


def actual():
    check = 'JOINT-DEDUP-ACTUAL-COMPARISON-20260907-01'
    if (OUT/'terminal.json').exists():
        print('ALREADY_COMPLETE_NO_RECOMPUTE')
        return
    if json.loads((OUT/'dedup_boundary_proof.json').read_text(encoding='utf-8'))['status'] != 'PASS':
        raise RuntimeError('NEW_DEDUP_BOUNDARY_PROOF_REQUIRED')
    started = time.monotonic()
    try:
        # Only this pure summarizer is called; the upstream actual/fixture/read-price routines are not called.
        from build_short_w_joint_recent_buy_w_20260907 import summarize
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
        lines = ['# JOINT 买入事件同股重叠窗口去重比较','',
            '本报告比较选股后原两周窗口内的机会分布，不是策略账户收益。原每日选股和状态保持不变。','',
            f'原始 {len(events)} 个事件 → 保留 {len(kept)}，删除 {len(dropped)} 个重叠事件，日历未决 {len(unresolved)}。'
            f'原始股票 {report["unique_stocks_before"]} 只，保留后 {report["unique_stocks_retained"]} 只。','',
            '每只股票按最早信号保留第一条，后续只有预定入场日在最近保留事件原 W 结束后才保留。删除事件不延长窗口；先全路径处理，再切年份。未成交与窗口未结束事件也参与日期去重。','',
            '| 窗口 | 完整/全部：原→去重 | 最大涨幅均值：原→去重 | 最大涨幅中位：原→去重 | 达到 5%：原→去重 | 去重后达到5%数量 |',
            '|---|---:|---:|---:|---:|---:|']
        for name,w in windows.items():
            a,b=w['before'],w['after']; am,bm=a['metrics'],b['metrics']
            lines.append(f'| {name} ({w["start"]}—{w["end"]}) | {a["complete_paths"]}/{a["all_selected_keys"]} → {b["complete_paths"]}/{b["all_selected_keys"]} | '
                         f'{pct(am["max_high_return"]["mean"])} → {pct(bm["max_high_return"]["mean"])} | '
                         f'{pct(am["max_high_return"]["median"])} → {pct(bm["max_high_return"]["median"])} | '
                         f'{pct(a["thresholds"]["5"]["fraction_of_complete"])} → {pct(b["thresholds"]["5"]["fraction_of_complete"])} | '
                         f'{b["thresholds"]["5"]["complete_hit_count"]}/{b["complete_paths"]} |')
        lines += ['', '以上涨幅与比例的分母为完整路径，不能把未决事件当成没有上涨。JSON 同时列出 3%/5%/10% 命中数、完整路径比例、全部保留事件的已确认命中下界及覆盖率。','',
            '| 窗口 | 首次峰值日中位：原→去重 | 峰前最低收益均值：原→去重 | 峰前最低收益中位：原→去重 | 去重峰前跌≥5%/可判定 | 全窗口最低收益中位：原→去重 |',
            '|---|---:|---:|---:|---:|---:|']
        for name,w in windows.items():
            a,b=w['before'],w['after']; am,bm=a['metrics'],b['metrics']
            ap,bp=am['min_low_before_peak_exclusive_return'],bm['min_low_before_peak_exclusive_return']
            lines.append(f'| {name} | {num(am["first_peak_day"]["median"])} → {num(bm["first_peak_day"]["median"])} | '
                         f'{pct(ap["mean"])} → {pct(bp["mean"])} | {pct(ap["median"])} → {pct(bp["median"])} | '
                         f'{b["strict_before_peak_down_5pct_count"]}/{bp["n"]} | '
                         f'{pct(am["min_low_return"]["median"])} → {pct(bm["min_low_return"]["median"])} |')
        lines += ['', '“峰前”严格排除最高价发生当日，不假设同一日高低价的先后；峰值在入场当天时可能没有可判定峰前区间。首次峰值日按原上游日序定义，延迟入场不重启原窗口。',
            '', '## 解释边界', '',
            '- 去重只处理同一股票重叠 W 的重复计数；不同股票同日共振仍在，不能把保留数称为独立样本数。',
            '- 最高涨幅是事后机会空间，买入当天高点可能无法按 T+1 卖出。原有 day2plus 和可执行 open 上界统计保留在 JSON，仍不构成可实施卖出策略。',
            '- 未重新训练、选择参数或计算价格标签，没有账户回测或交易订单。此比较也没有匹配市场基准，不能凭正涨幅认定买入 alpha。',
            '- 原 2023/2024 年末标签截断原样保留；2026 截至 8 月 21 日，其中 8 月尤其未成熟。',
            '', '逐事件保留/删除及锚点：membership_ledger.jsonl；完整指标：dedup_comparison.json。']
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
        post(check,'PASS','是：同股重叠重复加权存在；幅度见直接对比',evidence,'DOWNSTREAM',
             '原逐日事件统计保留，但不能充当独立机会数；依据去重后的分布修订描述性判断，不晋级策略。')
        print(json.dumps(dict(terminal=terminal,windows={k:dict(before_complete=w['before']['complete_paths'],
            after_complete=w['after']['complete_paths'],after_all=w['after']['all_selected_keys'],
            mean=w['after']['metrics']['max_high_return']['mean'],median=w['after']['metrics']['max_high_return']['median'],
            hit5=w['after']['thresholds']['5'],peak_day=w['after']['metrics']['first_peak_day'],
            pre_peak_low=w['after']['metrics']['min_low_before_peak_exclusive_return'],
            pre_peak_down5=w['after']['strict_before_peak_down_5pct_count'],delta=w['delta']) for k,w in windows.items()}),ensure_ascii=False))
    except Exception as exc:
        post(check,'INCONCLUSIVE','无法判断',repr(exc),'TEST_ITSELF',
             '修复精确接口后以新检验 ID 继续同一去重问题；不把工具失败作为策略终态。','REMOVE')
        raise


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=['fixture','actual'])
    args=parser.parse_args()
    fixture() if args.phase=='fixture' else actual()
