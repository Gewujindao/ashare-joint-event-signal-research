# q8_lots.py — reads: tradable, close_raw (training rows only).
# Computes: distribution of the value of one lot (100 shares) on the last trading day of 2016, 2019, 2022
# (and 2014 for reference); share of stocks a 20k account cannot buy or can only hold coarsely.
from common import *

start('q8_lots')
d = load('tradable', 'close_raw')
tr, yrs = d['tradable'], d['years']
seas = seasoned(tr, ~np.isnan(d['close_raw']))
P('date  n  price quantiles(5,25,50,75,95,99)  share lot>20000  share lot>5000  share lot>2000 | 500k: share lot>1% of capital')
for y in (2014, 2016, 2019, 2022):
    t = np.nonzero(yrs == y)[0][-1]
    px = d['close_raw'][t][seas[t]]
    lot = 100 * px
    P(d['dates'][t], px.size, np.round(np.percentile(px, [5, 25, 50, 75, 95, 99]), 2),
      '%.4f  %.4f  %.4f | %.4f' % ((lot > 20000).mean(), (lot > 5000).mean(), (lot > 2000).mean(), (lot > 5000).mean()))
finish()
