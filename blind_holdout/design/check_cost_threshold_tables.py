# Stage-1 arithmetic check (numpy only, no data files, < 30 s)
import numpy as np
from math import sqrt, log, exp, pi

def phi(z):
    return exp(-z * z / 2) / sqrt(2 * pi)

# --- costs ---
buy = 1.001 * 1.00011
sell_new = 0.999 * (1 - 0.00061)
sell_old = 0.999 * (1 - 0.00111)
print("buy bp", (buy - 1) * 1e4, "sell_new bp", (1 - sell_new) * 1e4, "sell_old bp", (1 - sell_old) * 1e4)
print("rt_new bp", (1 - sell_new / buy) * 1e4, "log", -log(sell_new / buy) * 100)
print("rt_old bp", (1 - sell_old / buy) * 1e4, "log", -log(sell_old / buy) * 100)

# --- target-1 hurdle ---
T = 551
k = 2 / sqrt(T)
print("2/sqrt(551)", k, "annual IR", k * sqrt(252))
sig_m = 0.014
h = k * sig_m
c = 0.00272
print("h bp/day", h * 1e4)
for L in (1, 2, 3, 5, 10, 20):
    print("L", L, "out-day mean bp", -(h + c / L) * 1e4, "per spell %", -(h * L + c) * 100)

# --- IC tables ---
sig, mu = 0.027, 0.0004
zq = {0.01: 2.3263, 0.05: 1.6449, 0.10: 1.2816, 0.20: 0.8416}
for q, z in zq.items():
    lam = phi(z) / q
    be = 2 * sqrt(mu * c) / (sig * lam)
    tg = 2 * sqrt((mu + h) * c) / (sig * lam)
    print("q", q, "lam", round(lam, 3), "breakeven rho", round(be, 4), "target rho", round(tg, 4))
print("L* breakeven", c / mu, "L* target", c / (mu + h))
for L in (1, 5, 20):
    a = 0.798 * sig * sqrt(L)
    print("L", L, "hit BE", 0.5 + (c + mu * L) / (2 * a), "hit target", 0.5 + (c + (mu + h) * L) / (2 * a))

# --- Monte Carlo: null width of target 3 and t1 of a zero-information rule ---
rng = np.random.default_rng(0)
N, out_frac, Lsp = 1000, 0.2, 5
sig_i = 0.022
m = rng.normal(0.0004, sig_m, T)
eps = rng.normal(0, sig_i, (T, N)).astype(np.float32)
r = m[:, None] + eps
# random spells: each stock, blocks of length Lsp out with prob out_frac
nb = T // Lsp + 1
blocks = rng.random((nb, N)) < out_frac
out = np.repeat(blocks, Lsp, axis=0)[:T]
ntr = (np.diff(out.astype(np.int8), axis=0) == 1).sum(axis=0)  # number of exits
X = -(out * r).mean(axis=1)
cost_per_day = ntr.mean() * c / T
Xc = X - cost_per_day
Zm = sqrt(T) * m.mean() / m.std(ddof=1)
print("Zm", Zm, "t1 no-cost", sqrt(T) * X.mean() / X.std(ddof=1), "t1 with cost", sqrt(T) * Xc.mean() / Xc.std(ddof=1))
print("sd(X) bp", X.std(ddof=1) * 1e4, "predicted", out_frac * sig_m * 1e4)
vals = []
for s in range(200):
    offs = rng.integers(0, T, N)
    idx = (np.arange(T)[:, None] - offs[None, :]) % T
    o2 = np.take_along_axis(out, idx, axis=0)
    vals.append(-(o2 * r).sum(axis=0).mean())
vals = np.array(vals)
print("null sd of mean log-NAV diff %", vals.std(ddof=1) * 100,
      "predicted %", sig_i * sqrt(out_frac * T / N) * 100)
