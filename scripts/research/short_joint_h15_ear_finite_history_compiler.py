"""Pure finite-history EAR compiler. No import-time I/O and no model fitting.

Inputs are explicit caller-bound metadata, query coverage, market calendar,
dated investment eligibility, and close-price receipts. This is a development
proxy, never a claim of globally first disclosure or historical provider PIT.
"""
from __future__ import annotations
from bisect import bisect_right
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from math import ceil, isfinite
from short_joint_h15_earnings_event_parser_20260908 import parse_title

ABI = 'JOINT_H15_EAR_FINITE_REQUIRED_HISTORY_COMPILER_V1'
IDENTITY = 'SOURCE_FIRST_WITHIN_REQUIRED_CLASSES_REGISTERED_HISTORY_H2021JAN01'
HISTORY_START = '2021-01-01'
REQUIRED_NODES = (1, 5, 6)
CN = timezone(timedelta(hours=8))


def stamp(day, time='15:00:00'):
    return datetime.fromisoformat(day + 'T' + time).replace(tzinfo=CN)


def moment(value):
    result = datetime.fromisoformat(value) if isinstance(value, str) else value
    if not isinstance(result, datetime) or result.tzinfo is None:
        raise ValueError('Timezone-aware availability/cutoff required')
    return result.astimezone(CN)


def iso_day(value):
    try:
        return date.fromisoformat(value).isoformat() if isinstance(value, str) and len(value) == 10 else None
    except ValueError:
        return None


def quarter_bounds(day):
    d = date.fromisoformat(day)
    month = 1 + 3 * ((d.month - 1) // 3)
    start = date(d.year, month, 1)
    next_start = date(d.year + 1, 1, 1) if month == 10 else date(d.year, month + 3, 1)
    return start.isoformat(), (next_start - timedelta(days=1)).isoformat()


def previous_quarter(day):
    start, _ = quarter_bounds(day)
    return quarter_bounds((date.fromisoformat(start) - timedelta(days=1)).isoformat())


def coverage_for_prefix(coverage, end):
    """Require all three registered query classes continuously from H to end.

    A fully returned whole-quarter query can document its within-quarter prefix.
    Its later announcement rows still cannot enter a decision before publication.
    Coverage is a retrospective source-quality receipt, not a historical PIT claim.
    """
    gaps, used = [], []
    current = HISTORY_START
    while current <= end:
        qstart, qend = quarter_bounds(current)
        needed_end = min(qend, end)
        for node in REQUIRED_NODES:
            candidates = [c for c in coverage
                          if c.get('s_node') == node and c.get('f_node') == 1
                          and (iso_day(c.get('start')) or '9999') <= current
                          and (iso_day(c.get('end')) or '') >= needed_end]
            good = [c for c in candidates
                    if c.get('pages_complete') is True
                    and c.get('count_compatible') is True
                    and c.get('explicit_cap_suspected') is False
                    and not c.get('date_or_identity_conflicts')]
            if good:
                used.append(good[0].get('query_id'))
            else:
                gaps.append(dict(s_node=node, start=current, end=needed_end,
                                 query_ids=[c.get('query_id') for c in candidates],
                                 reason='REQUIRED_CLASS_PREFIX_NOT_PROVEN_COMPLETE'))
        current = (date.fromisoformat(qend) + timedelta(days=1)).isoformat()
    return dict(complete=not gaps, gaps=gaps, query_ids=sorted(set(used)),
                hidden_provider_cap='UNKNOWN', historical_PIT=False)


def compile_events(rows, *, coverage, sessions, as_of):
    """Compile only publication-visible rows; preserve every source occurrence.

    Row ABI: provider, announcement_id, security_key, title_raw, notice_date;
    all other fields, including all native code/category associations, are kept.
    One provider ID can legitimately name several issuers. Conflict grouping is
    per provider/ID/issuer, while the original provider ID remains untouched.
    Later-received provider corrections are not historically vintaged here.
    """
    cutoff = moment(as_of)
    visible, future, unknown, groups = [], [], [], defaultdict(list)
    id_rows = defaultdict(list)
    for index, original in enumerate(rows):
        row = dict(original, occurrence_index=index)
        day = iso_day(row.get('notice_date'))
        if day is None:
            unknown.append(dict(row=row, reason='UNKNOWN_PUBLICATION_DATE')); continue
        row['available_at'] = stamp(day, '23:59:59').isoformat()
        if moment(row['available_at']) > cutoff:
            future.append(row); continue
        if day < HISTORY_START:
            unknown.append(dict(row=row, reason='OUTSIDE_REGISTERED_FINITE_HISTORY')); continue
        if any(not row.get(k) for k in ('provider', 'announcement_id', 'security_key')):
            unknown.append(dict(row=row, reason='MISSING_PROVIDER_ID_OR_ISSUER')); continue
        row['parsed_title'] = parse_title(row.get('title_raw'))
        visible.append(row)
        id_rows[(row['provider'], row['announcement_id'], row['security_key'])].append(row)
    conflicts, unresolved_issuer = [], defaultdict(list)
    for key, occurrences in id_rows.items():
        signatures = {(r.get('title_raw'), r['notice_date']) for r in occurrences}
        collision = len(signatures) != 1
        if collision:
            conflicts.append(dict(provider_id_issuer=list(key), occurrences=occurrences))
        for row in occurrences:
            p = row['parsed_title']
            if p['period_type_groupable']:
                groups[(key[0], key[2], p['fiscal_period']['key'])].append(
                    dict(row, id_conflict=collision))
            elif p['base_kind'] in ('EARNINGS_FORECAST', 'EARNINGS_FLASH', 'UNKNOWN'):
                # Explicit earnings rows whose period cannot be assigned could
                # precede the known period. Retain exact issuer/date uncertainty.
                unresolved_issuer[(key[0], key[2])].append(row)
                unknown.append(dict(row=row, reason='UNKNOWN_EARNINGS_PERIOD'))
            else:
                unknown.append(dict(row=row, reason='NON_EVENT_OR_UNASSIGNED_TITLE'))
    period_groups, byreaction = [], defaultdict(list)
    for (provider, security, period), members in sorted(groups.items()):
        first_day = min(r['notice_date'] for r in members)
        bundle = [r for r in members if r['notice_date'] == first_day]
        prefix = coverage_for_prefix(coverage, first_day)
        reasons = []
        if not prefix['complete']:
            reasons.append('INCOMPLETE_REQUIRED_CLASS_HISTORY_PREFIX')
        if any(r['id_conflict'] for r in bundle):
            reasons.append('FIRST_BUNDLE_PROVIDER_ID_CONFLICT')
        if not any(r['parsed_title']['initial_disclosure_candidate'] for r in bundle):
            reasons.append('FIRST_BUNDLE_ONLY_REVISION_OR_SUMMARY')
        ambiguous_prior = [r for r in unresolved_issuer[(provider, security)]
                           if r['notice_date'] <= first_day]
        if ambiguous_prior:
            reasons.append('PRIOR_EARNINGS_ROW_PERIOD_UNKNOWN')
        reaction_index = bisect_right(sessions, first_day)
        reaction = sessions[reaction_index] if reaction_index < len(sessions) else None
        score_day = sessions[reaction_index + 1] if reaction_index + 1 < len(sessions) else None
        if reaction is None or score_day is None or reaction_index < 2:
            reasons.append('MARKET_CALENDAR_WINDOW_UNAVAILABLE')
        group = dict(provider=provider, security_key=security, fiscal_period_key=period,
                     first_notice_date=first_day,
                     available_at=stamp(first_day, '23:59:59').isoformat(),
                     first_bundle_ids=sorted(set(r['announcement_id'] for r in bundle)),
                     first_bundle_occurrences=bundle, later_visible_occurrences=[r for r in members if r not in bundle],
                     same_date_order='UNORDERED',
                     global_earliest_complete=False, parser_earliest_complete=False,
                     finite_history_identity=IDENTITY, history_start=HISTORY_START,
                     coverage=prefix, ambiguous_prior_occurrences=ambiguous_prior,
                     status='UNKNOWN_OR_LEFT_CENSORED' if reasons else 'SCOPED_FIRST_EVENT',
                     reasons=reasons, reaction_session=reaction, score_origin=score_day,
                     historical_PIT=False)
        period_groups.append(group)
        if not reasons:
            byreaction[(security, reaction)].append(group)
    events = []
    for (security, reaction), members in sorted(byreaction.items()):
        ri = bisect_right(sessions, reaction) - 1
        origins = sessions[ri + 1]
        events.append(dict(event_id=f'{IDENTITY}|{security}|{reaction}', security_key=security,
                           reaction_session=reaction, score_origin=origins,
                           score_available_at=stamp(origins).isoformat(),
                           first_action_session=sessions[ri + 2] if ri + 2 < len(sessions) else None,
                           four_close_sessions=list(sessions[ri - 2:ri + 2]),
                           period_groups=members,
                           announcement_identities=sorted({(g['provider'], ident) for g in members for ident in g['first_bundle_ids']}),
                           finite_history_identity=IDENTITY, earliest_complete=False,
                           status='SCOPED_EVENT_AWAITING_PRICE_RECEIPTS'))
    return dict(abi=ABI, as_of=cutoff.isoformat(), events=events, period_groups=period_groups,
                visible_occurrences=visible, future_publication_quarantine=future,
                unknown_or_non_event_rows=unknown, conflicting_ids=conflicts,
                historical_PIT=False, actual_source_history_global_complete=False)


def score_event(event, *, stock_closes, benchmark_closes, as_of):
    """Four same-coordinate positive closes per leg; no forward fill/interpolation.

    Receipts keyed by session: close, available_at, coordinate_id, receipt_id.
    Available after that session's close is allowed only if <= scoring cutoff.
    The returned score's availability is the latest actual receipt clock, never
    silently backdated to s+1 if the input reports later availability.
    """
    cutoff = moment(as_of)
    result = dict(event_id=event['event_id'], security_key=event['security_key'],
                  reaction_session=event['reaction_session'], score_origin=event['score_origin'],
                  earliest_complete=False, status='UNKNOWN_SCORE', ear=None,
                  available_at=event['score_available_at'], reasons=[], receipts=[])
    if cutoff < moment(event['score_available_at']):
        result['reasons'] = ['S_PLUS_ONE_CLOSE_NOT_YET_AVAILABLE']; return result
    values, clocks = [], [moment(event['score_available_at'])]
    for leg, mapping in [('STOCK', stock_closes), ('CSI300_FEATURE_ONLY', benchmark_closes)]:
        coordinate, legvalues = set(), []
        for day in event['four_close_sessions']:
            r = mapping.get(day)
            if r is None:
                result['reasons'].append(f'{leg}:MISSING_CLOSE:{day}'); continue
            try:
                value, available = float(r['close']), moment(r['available_at'])
                valid = isfinite(value) and value > 0 and bool(r.get('coordinate_id')) and bool(r.get('receipt_id'))
            except (TypeError, ValueError, KeyError):
                result['reasons'].append(f'{leg}:INVALID_CLOSE_RECEIPT:{day}'); continue
            if not valid or available < stamp(day) or available > cutoff:
                result['reasons'].append(f'{leg}:INVALID_OR_UNAVAILABLE_CLOSE:{day}'); continue
            coordinate.add(r['coordinate_id']); legvalues.append(value); clocks.append(available)
            result['receipts'].append(dict(r, leg=leg, session=day))
        if len(coordinate) != 1 or len(legvalues) != 4:
            result['reasons'].append(f'{leg}:FOUR_CLOSE_COORDINATE_INCOMPLETE')
        values.append(legvalues)
    if result['reasons']:
        return result
    stock_return = values[0][-1] / values[0][0] - 1
    benchmark_return = values[1][-1] / values[1][0] - 1
    result.update(status='SCORED', ear=stock_return - benchmark_return,
                  stock_three_session_compound_return=stock_return,
                  benchmark_three_session_compound_return=benchmark_return,
                  available_at=max(clocks).isoformat())
    return result


def threshold_for(score, reference_scores):
    """Previous reaction quarter only, matured by that completed quarter's end.

    Reference ABI requires event_id, reaction_session, status, ear, available_at.
    Caller must supply the full registered reference universe; this function
    performs no fallback, minimum-N heuristic, or cross-quarter pooling.
    """
    start, end = previous_quarter(score['reaction_session'])
    qcut = stamp(end, '23:59:59')
    identities, conflicts = {}, set()
    for r in reference_scores:
        if r.get('status') != 'SCORED' or not start <= r.get('reaction_session', '') <= end:
            continue
        if moment(r['available_at']) > qcut:
            continue
        value = float(r['ear'])
        if not isfinite(value):
            continue
        key = r['event_id']
        if key in identities and identities[key] != value:
            conflicts.add(key)
        identities[key] = value
    values = sorted(v for k, v in identities.items() if k not in conflicts)
    result = dict(reference_start=start, reference_end=end, mature_cutoff=qcut.isoformat(),
                  n=len(values), source_event_ids=sorted(k for k in identities if k not in conflicts),
                  conflicting_event_ids=sorted(conflicts), method='ORDER_STAT_CEIL_0P8_N_MINUS1',
                  status='KNOWN_THRESHOLD' if values and not conflicts else 'UNKNOWN_THRESHOLD',
                  cutpoint=values[ceil(.8 * len(values)) - 1] if values and not conflicts else None)
    return result


def daily_buy_candidates(scores, *, reference_scores, origin, eligibility):
    """Fresh s+1 scores only, strict positive/previous-quarter80%, <=4 names.

    Eligibility mapping uses caller-bound dated investment identity receipts,
    not provider ann_type. No fill, sale, wealth, or learned-exit claim here.
    Requiring origin==score_origin prevents a late receipt creating a stale buy.
    """
    cutoff = stamp(origin)
    considered, qualified = [], []
    for score in scores:
        if score.get('score_origin') != origin:
            continue
        key = score['security_key']
        e = eligibility.get(key)
        row = dict(event_id=score['event_id'], security_key=key, origin=origin,
                   proposed_state='HOLD', reason=None, threshold=None)
        if score.get('status') != 'SCORED' or moment(score['available_at']) > cutoff:
            row.update(proposed_state='PENDING_UNEXECUTABLE', reason='SCORE_NOT_AVAILABLE_AT_ORIGIN_CLOSE')
        elif e is None or e.get('origin') != origin or not e.get('receipt_id') or moment(e['available_at']) > cutoff:
            row.update(proposed_state='PENDING_UNEXECUTABLE', reason='DATED_INVESTMENT_IDENTITY_UNKNOWN')
        elif e.get('eligible') is not True:
            row['reason'] = 'CURRENT_INVESTMENT_UNIVERSE_INELIGIBLE'
        else:
            threshold = threshold_for(score, reference_scores)
            row['threshold'] = threshold
            if threshold['status'] != 'KNOWN_THRESHOLD':
                row.update(proposed_state='PENDING_UNEXECUTABLE', reason='UNKNOWN_PRIOR_QUARTER_THRESHOLD')
            elif score['ear'] > max(0., threshold['cutpoint']):
                row.update(reason='EAR_STRICTLY_ABOVE_ZERO_AND_PRIOR_QUARTER_P80', ear=score['ear'])
                qualified.append(row)
            else:
                row['reason'] = 'EAR_DOES_NOT_PASS_FROZEN_THRESHOLD'
        considered.append(row)
    # One security/reaction event; reject conflicting duplicate score receipts.
    bysecurity = defaultdict(list)
    for row in qualified:
        bysecurity[row['security_key']].append(row)
    unique = []
    for key, items in bysecurity.items():
        signatures = {(r['event_id'], r['ear']) for r in items}
        if len(signatures) != 1:
            for r in items:
                r.update(proposed_state='PENDING_UNEXECUTABLE', reason='CONFLICTING_SAME_SECURITY_SCORE')
        else:
            unique.append(items[0])
            for r in items[1:]:
                r['reason'] = 'DUPLICATE_SCORE_RECEIPT_NO_ADDITIONAL_ORDER'
    selected = sorted(unique, key=lambda r: (-r['ear'], r['security_key']))[:4]
    for row in selected:
        row['proposed_state'] = 'BUY'
    return dict(abi=ABI, origin=origin, available_at=cutoff.isoformat(),
                selected=selected, considered=considered, unfilled_signal_slots=4-len(selected),
                action_clock='STRICT_SUBSEQUENT_EXECUTABLE_OPEN',
                scope='BUY_SIGNAL_ONLY_ENGINE_MUST_GENERATE_HOLD_SELL_PENDING_REENTRY',
                trade_orders_created=0, historical_PIT=False)
