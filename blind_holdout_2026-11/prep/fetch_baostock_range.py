r"""fetch_baostock_range.py - daily bars of Shanghai/Shenzhen main-board stocks for one date range, one small CSV per code.

STATUS       Written offline on 2026-11-09; verified online the same day with baostock 0.9.4 (one stock, then 30 stocks; the full batch has not been run).
             The calls marked "VERIFY BEFORE RUN" were all found correct as written: login() without arguments, query_history_k_data_plus,
             query_trade_dates, the 12 field names, adjustflag 3 / 2, dates as YYYY-MM-DD, volume in shares, amount in CNY, turn in percent,
             preclose = previous close, tradestatus '1'/'0', isST '0'/'1'. A suspended day has a row with prices and EMPTY volume, amount and turn.
             A stock that left the market has no rows after its last day. Measured: 0.49 s per request, 0 failures in 60 requests, 4.05 KB per code.
             Changes made during verification: --blind-after (dry run prints only a format mask for later rows), GuardedSocket (a closed
             connection becomes an error instead of an endless loop inside baostock), load_baostock().
DATA SOURCE  Baostock (www.baostock.com), python package `baostock`, anonymous login: bs.login() with no arguments.
             No account, password or key is used, asked for or stored.
CODES        the `codes` member of data\ashare_mainboard_to_20240628.npz (3,329 six-digit codes: 000/001/002/003 -> sz, 600/601/603/605 -> sh),
             plus an optional text file with one six-digit code per line (--extra-codes).
PER CODE     two requests for the whole range: adjustflag=3 (raw prices) and adjustflag=2 (forward-adjusted prices).
REQUESTS     2 x N bar requests + 1 trade-calendar request + 1 login per RELOGIN_EVERY codes.
             N = 3,329 -> 6,658 bar requests + 1 + 17 logins.
TIME         N x (2 x t_req + SLEEP_REQUEST + SLEEP_CODE), t_req = seconds one request takes.
             Defaults 0.35 s and 0.5 s; t_req 0.6 to 1.5 s -> 3,329 x (2.05 to 3.85 s) = 1.9 to 3.6 hours.
             (Old project records: about 28 codes per minute on a fresh session = 2.0 hours; a 27-day range for 3,180 codes took 2 h 35 min.)
SIZE         rows per code = trading days in the range (2026-09-21 .. 2026-11-09 is 30 days); about 150 bytes per row
             -> about 4.6 KB per code, about 15 MB in total (about 27 MB on disk with 4 KB clusters).
OUTPUT       <out>\<code>.csv with the 16 columns of CSV_HEADER, values written exactly as the strings Baostock returns.
             A code with no row in the range (delisted earlier) gets a header-only file.
             <out>\..\trade_dates.csv, fetch_log.txt, fetch_meta.json, fetch.lock (removed at exit).
RESUME       a code whose CSV already exists is skipped; rerun the same command after any stop. Start and end must not change between runs.
MAPPING TO THE 16 PANEL MEMBERS (MANIFEST.txt section 4)
             dates <- date ; codes <- code ; tradable <- tradestatus == 1 ; st <- is_st == 1 ;
             open_raw high_raw low_raw close_raw preclose_raw <- open high low close preclose with adjustflag=3 ;
             open_qfq high_qfq low_qfq close_qfq <- open high low close with adjustflag=2 ;
             volume_shares <- volume ; amount_cny <- amount ; turnover_pct <- turn.

usage (after approval; python = python -I -X utf8):
  python prep\fetch_baostock_range.py --start 2026-09-21 --end 2026-11-09 --dry-run-one
  python prep\fetch_baostock_range.py --start 2026-09-21 --end 2026-11-09
"""
import os
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
import argparse
import csv
import ctypes
import datetime
import json
import re
import socket
import sys
import time
import traceback
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# ######## VERIFY BEFORE RUN (with --dry-run-one) ########
# Field names of bs.query_history_k_data_plus for frequency 'd', written from memory of the Baostock documentation:
#   date, code, open, high, low, close, preclose, volume (shares), amount (CNY), turn (percent),
#   tradestatus (1 = trading, 0 = suspended), isST (1 = ST or *ST, 0 = not).
# adjustflag: '3' = no adjustment, '2' = forward adjustment (qfq), '1' = backward adjustment.
BAR_FIELDS = ['date', 'code', 'open', 'high', 'low', 'close', 'preclose', 'volume', 'amount', 'turn', 'tradestatus', 'isST']
ADJ_RAW, ADJ_QFQ = '3', '2'
CSV_HEADER = ['date', 'code', 'open_raw', 'high_raw', 'low_raw', 'close_raw', 'preclose_raw', 'open_qfq', 'high_qfq', 'low_qfq', 'close_qfq',
              'volume_shares', 'amount_cny', 'turnover_pct', 'tradestatus', 'is_st']
SOURCE = 'Baostock (www.baostock.com), python package baostock, anonymous login; licence: see the notice on the Baostock site (not checked offline)'


def bs_code(code):
    return ('sh.' if code[0] == '6' else 'sz.') + code


class GuardedSocket(socket.socket):
    # baostock 0.9.4 send_msg() loops on recv() until an end marker arrives; when the server closes the connection recv() returns b''
    # for ever and the loop never ends. Raising here makes send_msg() return None, which the query reports as error 10002007 -> retry.
    def recv(self, *args):
        r = super().recv(*args)
        if not r:
            raise ConnectionError('the server closed the connection')
        return r


def load_baostock(pylibs):
    sys.path.insert(0, pylibs)
    import baostock as bs                                  # verified 2026-11-09 with baostock 0.9.4: imports with the environment's pandas, nothing else needed
    import baostock.util.socketutil as su
    su.socket = types.SimpleNamespace(socket=GuardedSocket, AF_INET=socket.AF_INET, SOCK_STREAM=socket.SOCK_STREAM)
    return bs


def login(bs):
    # ######## VERIFY BEFORE RUN ######## anonymous login: no arguments; the result has error_code ('0' = ok) and error_msg
    lg = bs.login()
    if lg.error_code != '0':
        raise RuntimeError('login failed: %s %s' % (lg.error_code, lg.error_msg))


def relogin(bs):
    try:
        bs.logout()
    except Exception:
        pass
    time.sleep(2.0)
    login(bs)


def all_rows(rs):
    # ######## VERIFY BEFORE RUN ######## result object: error_code, error_msg, fields, next(), get_row_data()
    if rs.error_code != '0':
        raise RuntimeError('query failed: %s %s' % (rs.error_code, rs.error_msg))
    rows = []
    while rs.error_code == '0' and rs.next():
        rows.append(rs.get_row_data())
    if rs.error_code != '0':
        raise RuntimeError('query failed while reading rows: %s %s' % (rs.error_code, rs.error_msg))
    return rows


def query_bars(bs, code, start, end, adjustflag):
    # ######## VERIFY BEFORE RUN ######## parameter names start_date, end_date, frequency, adjustflag; fields as one comma-joined string
    rs = bs.query_history_k_data_plus(bs_code(code), ','.join(BAR_FIELDS), start_date=start, end_date=end, frequency='d', adjustflag=adjustflag)
    rows = all_rows(rs)
    assert list(rs.fields) == BAR_FIELDS, rs.fields
    return rows


def query_trade_dates(bs, start, end):
    # ######## VERIFY BEFORE RUN ######## fields calendar_date, is_trading_day ('1' = trading day)
    rs = bs.query_trade_dates(start_date=start, end_date=end)
    rows = all_rows(rs)
    assert list(rs.fields) == ['calendar_date', 'is_trading_day'], rs.fields
    return [d for d, f in rows if f == '1']


def fetch_one(bs, code, start, end, sleep_request):
    raw = query_bars(bs, code, start, end, ADJ_RAW)
    time.sleep(sleep_request)
    qfq = query_bars(bs, code, start, end, ADJ_QFQ)
    assert [x[0] for x in raw] == [x[0] for x in qfq], 'raw and qfq dates differ for %s' % code
    out = []
    for a, q in zip(raw, qfq):
        d, e = dict(zip(BAR_FIELDS, a)), dict(zip(BAR_FIELDS, q))
        out.append([d['date'], code, d['open'], d['high'], d['low'], d['close'], d['preclose'], e['open'], e['high'], e['low'], e['close'],
                    d['volume'], d['amount'], d['turn'], d['tradestatus'], d['isST']])
    return out


def fetch_with_retry(bs, code, a, log, need_login):
    for k in range(a.retries + 1):
        try:
            if need_login:
                relogin(bs)
            return fetch_one(bs, code, a.start, a.end, a.sleep_request)
        except (RuntimeError, OSError) as ex:
            log('  %s attempt %d of %d failed: %s' % (code, k + 1, a.retries + 1, ex))
            need_login = True
            if k < a.retries:
                time.sleep(a.retry_wait * (k + 1))
    return None


def dry_run_one(bs, a):
    """One stock, one range, nothing written. Prints what the full run would rely on."""
    login(bs)
    cal = query_trade_dates(bs, a.start, a.end)
    print('trade calendar %s..%s: %d trading days: %s' % (a.start, a.end, len(cal), ' '.join(cal)))
    raw = query_bars(bs, a.dry_code, a.start, a.end, ADJ_RAW)
    time.sleep(a.sleep_request)
    qfq = query_bars(bs, a.dry_code, a.start, a.end, ADJ_QFQ)
    bs.logout()
    print('code %s as %s | fields %s' % (a.dry_code, bs_code(a.dry_code), ','.join(BAR_FIELDS)))

    def mask(v):
        # rows after --blind-after are reserved for a forward test: show the format of a value (N = integer part, d = one decimal), never the value
        m = re.fullmatch(r'(-?)\d+(?:\.(\d+))?', v)
        return '<empty>' if v == '' else '<non-numeric>' if not m else m.group(1) + 'N' + ('' if m.group(2) is None else '.' + 'd' * len(m.group(2)))

    for lab, rows in (('adjustflag=3 (raw)', raw), ('adjustflag=2 (qfq)', qfq)):
        seen = [x for x in rows if x[0] <= a.blind_after]
        blind = [x for x in rows if x[0] > a.blind_after]
        print('%s: %d rows = %d up to %s (shown) + %d later (format only)' % (lab, len(rows), len(seen), a.blind_after, len(blind)))
        for x in seen:
            print('   ' + ','.join(x))
        for x in blind[:1] + blind[-2:]:
            print('   ' + ','.join(x[:2] + [mask(v) for v in x[2:]]))
    print('CHECK 1  dates of both requests equal the trading days of the calendar: %s' % ([x[0] for x in raw] == cal == [x[0] for x in qfq]))
    print('CHECK 2  volume is in shares: amount / volume must lie between low and high on trading rows')
    for x in raw:
        d = dict(zip(BAR_FIELDS, x))
        if d['tradestatus'] == '1' and float(d['volume']) > 0:
            vw = float(d['amount']) / float(d['volume'])
            ok = float(d['low']) - 0.01 <= vw <= float(d['high']) + 0.01
            if d['date'] <= a.blind_after:
                print('   %s low %s  amount/volume %.4f  high %s  %s' % (d['date'], d['low'], vw, d['high'], ok))
            else:
                print('   %s %s' % (d['date'], ok))
    print('CHECK 3  values seen: tradestatus %s | isST %s | date format %s' % (sorted({x[10] for x in raw}), sorted({x[11] for x in raw}), raw[0][0] if raw else '-'))
    print('CHECK 4  by eye: preclose is the reference price of the day (equal to the previous close except on ex-right days); turn is a percentage;')
    print('         qfq rows equal raw rows unless the stock has an ex-right day after the row date.')
    print('CSV lines the full run would write (header, first row):')
    print('   ' + ','.join(CSV_HEADER))
    for a_, q_ in list(zip(raw, qfq))[:1]:
        d, e = dict(zip(BAR_FIELDS, a_)), dict(zip(BAR_FIELDS, q_))
        print('   ' + ','.join([d['date'], a.dry_code, d['open'], d['high'], d['low'], d['close'], d['preclose'], e['open'], e['high'], e['low'], e['close'],
                                d['volume'], d['amount'], d['turn'], d['tradestatus'], d['isST']]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', required=True, help='first date YYYY-MM-DD (2026-09-21 leaves 9 overlap days with the old panel)')
    ap.add_argument('--end', required=True, help='last date YYYY-MM-DD; must be a day whose data Baostock has finished loading')
    ap.add_argument('--panel', default=os.path.join(ROOT, 'data', 'ashare_mainboard_to_20240628.npz'), help='npz whose codes member is the code list')
    ap.add_argument('--extra-codes', default=None, help='optional text file, one six-digit code per line')
    ap.add_argument('--out', default=os.path.join(ROOT, 'data_new', 'raw'))
    ap.add_argument('--pylibs', default=os.path.join(ROOT, 'pylibs'), help='folder where baostock was installed with pip --target')
    ap.add_argument('--sleep-request', type=float, default=0.35, help='seconds between the two requests of one code')
    ap.add_argument('--sleep-code', type=float, default=0.5, help='seconds between codes')
    ap.add_argument('--relogin-every', type=int, default=200, help='log out and in again after this many codes (a session slows down with use)')
    ap.add_argument('--retries', type=int, default=3, help='extra attempts per code after a failed one, each after a fresh login')
    ap.add_argument('--retry-wait', type=float, default=20.0, help='seconds before the first retry; grows linearly')
    ap.add_argument('--max-fail-streak', type=int, default=5, help='stop the run when this many codes in a row fail')
    ap.add_argument('--timeout', type=float, default=60.0, help='socket timeout in seconds')
    ap.add_argument('--dry-run-one', action='store_true', help='fetch one stock for the range, print the fields and checks, write nothing')
    ap.add_argument('--dry-code', default='600000')
    ap.add_argument('--blind-after', default='2026-10-09', help='--dry-run-one prints the values of rows up to this date only; later rows as a format mask')
    a = ap.parse_args()

    # Baostock loads the daily bars at 17:30 and the adjustment factors at 18:00 Beijing time (VERIFY: from the site documentation, from memory)
    now_bj = datetime.datetime.utcnow() + datetime.timedelta(hours=8)
    last_complete = (now_bj - datetime.timedelta(hours=18)).date()
    assert datetime.date.fromisoformat(a.start) <= datetime.date.fromisoformat(a.end) <= last_complete, \
        'end %s is later than the last day Baostock has finished loading (%s, Beijing time now %s)' % (a.end, last_complete, now_bj.strftime('%Y-%m-%d %H:%M'))

    k32 = ctypes.WinDLL('kernel32', use_last_error=True)
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    k32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    k32.SetPriorityClass(k32.GetCurrentProcess(), 0x00004000)          # BELOW_NORMAL_PRIORITY_CLASS
    socket.setdefaulttimeout(a.timeout)
    bs = load_baostock(a.pylibs)

    if a.dry_run_one:
        dry_run_one(bs, a)
        return 0

    z = np.load(a.panel, allow_pickle=False)
    codes = [str(c) for c in z['codes']]
    z.close()
    if a.extra_codes:
        with open(a.extra_codes, encoding='utf-8') as fh:
            codes += [ln.strip() for ln in fh if ln.strip()]
    assert len(set(codes)) == len(codes) and all(len(c) == 6 and c.isdigit() and c[0] in '06' for c in codes)

    base = os.path.dirname(os.path.abspath(a.out))
    os.makedirs(a.out, exist_ok=True)
    f_log, f_meta, f_lock = (os.path.join(base, n) for n in ('fetch_log.txt', 'fetch_meta.json', 'fetch.lock'))

    def log(msg):
        line = '%s %s' % (time.strftime('%Y-%m-%d %H:%M:%S'), msg)
        print(line, flush=True)
        with open(f_log, 'a', encoding='utf-8') as fh:
            fh.write(line + '\n')

    lock = os.open(f_lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)       # one Baostock process at a time: anonymous sessions disturb each other
    status = 'STOPPED'
    try:
        meta = dict(source=SOURCE, start=a.start, end=a.end, n_codes=len(codes), csv_header=CSV_HEADER, first_started=time.strftime('%Y-%m-%d %H:%M:%S'))
        if os.path.exists(f_meta):
            with open(f_meta, encoding='utf-8') as fh:
                meta = json.load(fh)
            assert meta['start'] == a.start and meta['end'] == a.end and meta['n_codes'] == len(codes), 'a resumed run must use the same range and code list'
        todo = [c for c in codes if not os.path.exists(os.path.join(a.out, c + '.csv'))]
        log('pid %d | range %s..%s | codes %d | already on disk %d | to fetch %d | sleep %.2f + %.2f s | relogin every %d' % (
            os.getpid(), a.start, a.end, len(codes), len(codes) - len(todo), len(todo), a.sleep_request, a.sleep_code, a.relogin_every))
        login(bs)
        meta['baostock_version'] = getattr(bs, '__version__', 'unknown')
        cal = query_trade_dates(bs, a.start, a.end)
        with open(os.path.join(base, 'trade_dates.csv'), 'w', encoding='utf-8', newline='') as fh:
            fh.write('\n'.join(cal) + '\n')
        meta['trade_dates'] = cal
        with open(f_meta, 'w', encoding='utf-8') as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=1)
        log('trade calendar: %d trading days %s..%s' % (len(cal), cal[0], cal[-1]))

        t0 = time.monotonic()
        streak, done, failed = 0, 0, []
        for j, code in enumerate(todo):
            t1 = time.monotonic()
            rows = fetch_with_retry(bs, code, a, log, need_login=(j > 0 and j % a.relogin_every == 0))
            if rows is None:
                failed.append(code)
                streak += 1
                log('FAILED %s after %d attempts (failures in a row: %d)' % (code, a.retries + 1, streak))
                if streak >= a.max_fail_streak:
                    log('STOPPED: %d codes in a row failed. Rerun the same command later; finished codes are kept.' % streak)
                    break
            else:
                streak = 0
                done += 1
                tmp = os.path.join(a.out, code + '.csv.tmp')
                with open(tmp, 'w', encoding='utf-8', newline='') as fh:
                    w = csv.writer(fh)
                    w.writerow(CSV_HEADER)
                    w.writerows(rows)
                os.replace(tmp, os.path.join(a.out, code + '.csv'))
                el = time.monotonic() - t0
                log('%5d/%d %s rows %2d of %d | %4.1f s | %5.1f codes/min | about %4.0f min left' % (
                    j + 1, len(todo), code, len(rows), len(cal), time.monotonic() - t1, (j + 1) / el * 60, (len(todo) - j - 1) * el / (j + 1) / 60))
            time.sleep(a.sleep_code)
        try:
            bs.logout()
        except Exception:
            pass

        left = [c for c in codes if not os.path.exists(os.path.join(a.out, c + '.csv'))]
        sizes = [os.path.getsize(os.path.join(a.out, c + '.csv')) for c in codes if os.path.exists(os.path.join(a.out, c + '.csv'))]
        status = 'COMPLETE' if not left else 'INCOMPLETE'
        meta.update(status=status, last_finished=time.strftime('%Y-%m-%d %H:%M:%S'), n_on_disk=len(sizes), bytes_on_disk=int(sum(sizes)),
                    n_missing=len(left), failed_this_run=failed)
        with open(f_meta, 'w', encoding='utf-8') as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=1)
        log('%s: fetched this run %d | on disk %d of %d | %.1f MB | missing %d | failed this run: %s | %.1f min' % (
            status, done, len(sizes), len(codes), sum(sizes) / 2 ** 20, len(left), ' '.join(failed) if failed else 'none', (time.monotonic() - t0) / 60))
    except BaseException:
        log('ABORTED with an exception:\n' + traceback.format_exc())
        raise
    finally:
        os.close(lock)
        os.remove(f_lock)
    return 0 if status == 'COMPLETE' else 3


if __name__ == '__main__':
    sys.exit(main())
