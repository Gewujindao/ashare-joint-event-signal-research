# account_prefix_check.py — prefix invariance of the daily lists and the account NAV.
# usage: account_prefix_check.py TAG_SHORT TAG_LONG [CUTOFF]
# Compares record_c2_<TAG>_<window>.npz of two runs of account_v1.py main (shorter data vs longer data) for every window
# present in both, on all days <= CUTOFF (default 2024-06-28): held lists, executed buys, executed sells (compared as
# stock codes, so a different column order or extra columns do not matter), strategy NAV and benchmark-B NAV.
# Exit code 0 = identical, 5 = any difference (the evaluation is void in that case).
import sys, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
TS, TL = sys.argv[1:3]
CUT = sys.argv[3] if len(sys.argv) > 3 else '2024-06-28'
bad = False
lists_bad = False
relmax = 0.0
nwin = 0
for w in ('V', 'SEL', 'HOLD'):
    fs, fl = (os.path.join(HERE, 'record_c2_%s_%s.npz' % (t, w)) for t in (TS, TL))
    if not (os.path.exists(fs) and os.path.exists(fl)):
        continue
    s, l = np.load(fs, allow_pickle=False), np.load(fl, allow_pickle=False)
    m = int(min((s['dates'] <= CUT).sum(), (l['dates'] <= CUT).sum()))
    if m == 0:
        continue
    nwin += 1
    same_dates = bool((s['dates'][:m] == l['dates'][:m]).all())
    out = [w, '%d days %s..%s' % (m, s['dates'][0], s['dates'][m - 1]), 'dates %s' % ('same' if same_dates else 'DIFFERENT')]
    ok = same_dates
    for key in ('held', 'buys', 'sells'):
        cs = np.where(s[key][:m] >= 0, s['codes'][np.maximum(s[key][:m], 0)], '')
        cl = np.where(l[key][:m] >= 0, l['codes'][np.maximum(l[key][:m], 0)], '')
        nd = int((cs != cl).any(axis=1).sum())
        ok &= nd == 0
        out.append('%s: days that differ %d' % (key, nd))
    lists_bad |= not ok
    for key in ('nav', 'navB'):
        dmax = float(np.abs(s[key][:m] - l[key][:m]).max())
        relmax = max(relmax, float(np.abs(s[key][:m] / l[key][:m] - 1).max()))
        ok &= dmax == 0
        out.append('%s max |diff| %.3e' % (key, dmax))
    print(' | '.join(out) + ' | ' + ('IDENTICAL' if ok else 'NOT IDENTICAL'))
    # report only (records of account_v2.py): conservative NAV depends on where the data ends (a stock suspended at the end of
    # the short file counts as ended there); differences are expected only for such stocks and do not void the evaluation.
    rep = []
    for key in ('navz', 'navBz'):
        if key in s.files and key in l.files:
            df = np.abs(s[key][:m] - l[key][:m])
            nz = np.nonzero(df > 0)[0]
            rep.append('%s max |diff| %.3e, days that differ %d%s' % (key, float(df.max()), nz.size, (' (first %s)' % s['dates'][nz[0]]) if nz.size else ''))
    if rep:
        print('   report only, conservative valuation: ' + ' | '.join(rep))
    bad |= not ok
if nwin == 0:
    print('no common window found')
    sys.exit(6)
if bad and not lists_bad:
    # every daily list is identical but a last-close NAV is not bitwise identical (e.g. qfq arrays re-anchored in float32).
    # This is NOT a pass: the case needs a separate decision.
    print('PREFIX CHECK: LISTS IDENTICAL, NAV NOT BITWISE IDENTICAL (max relative NAV difference %.3e) - report to root' % relmax)
    sys.exit(7)
print('PREFIX CHECK ' + ('FAILED' if bad else 'PASSED'))
sys.exit(5 if bad else 0)
