# q4_counts.py — reads: tradable, st, open_raw(not used for fills here), high_raw, low_raw, close_raw,
# preclose_raw, close_qfq, turnover_pct (training rows).
# Daily counts of each limit-event class (mean, sd, share of days with none, concentration by year),
# the 3-day occupancy share (what the out-fraction would be with L = 3) and its sd over time,
# and the overlap of the limit classes with the overheated states of Q2 (HOT_top5%, AT_top5%, HOT_top1%).
from common import *

start('q4_counts')
d = load('tradable', 'st', 'high_raw', 'low_raw', 'close_raw', 'preclose_raw', 'close_qfq', 'turnover_pct')
tr, st, yrs = d['tradable'], d['st'], d['years']
T, N = tr.shape
seas = seasoned(tr, ~np.isnan(d['close_qfq']))
ev = limit_events(d, seas)
Cff = ffill(np.where(tr, d['close_qfq'], np.nan).astype(np.float32))
hot, U = hot_states(d, seas, Cff)
Nu = np.maximum(seas.sum(1), 1)
per = periods(yrs)
allst = dict(ev)
for k in ('HOT_top1%', 'HOT_top5%', 'AT_top1%', 'AT_top5%'):
    allst[k] = hot[k]
P('class | events total | per day mean / sd | days with none % | top-3 years share % (years) | L=3 occupancy share%: mean / sd over days | per-day mean by ALL,P1,P2,P3,EX2015 | ST share %')
for name, E in allst.items():
    c = E.sum(1)
    by = np.array([c[yrs == y].sum() for y in range(2014, 2023)])
    top = np.argsort(by)[::-1][:3]
    occ = np.zeros((T, N), bool)
    for j in range(1, 4):
        occ[j:] |= E[:T - j]
    sh = occ.sum(1) / Nu
    P('%s | %d | %.1f / %.1f | %.1f | %.1f (%s) | %.3f / %.3f | %s | %.1f' % (
        name, c.sum(), c.mean(), c.std(), (c == 0).mean() * 100, by[top].sum() / max(by.sum(), 1) * 100,
        ','.join(str(2014 + i) for i in top), sh.mean() * 100, sh.std() * 100,
        ' '.join('%.1f' % c[pm].mean() for pm in per.values()), (E & st).sum() / max(E.sum(), 1) * 100))
P('\noverlap: limit class | share of its events inside HOT_top5% | inside AT_top5% | share of HOT_top1% events that are in this class')
h1 = hot['HOT_top1%']
for name, E in ev.items():
    n = max(E.sum(), 1)
    P('%s | %.1f | %.1f | %.1f' % (name, (E & hot['HOT_top5%']).sum() / n * 100, (E & hot['AT_top5%']).sum() / n * 100,
                                   (E & h1).sum() / max(h1.sum(), 1) * 100))
finish()
