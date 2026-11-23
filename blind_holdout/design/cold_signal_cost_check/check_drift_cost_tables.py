# Stage B1 arithmetic check: numpy only, reads no data, runs in a few seconds.
import numpy as np

SE = 2.33      # daily sd of stock relative return, % (training, ex-2015)
SM = 1.44      # daily sd of EW market, %
T = 551
YRS = T / 252.0
C0 = 27.2      # round trip, bp, trade >= 50k
IC = {1: 0.04, 5: 0.07, 10: 0.08, 20: 0.09, 40: 0.095, 60: 0.095}   # 40/60 are extrapolations
LAM = 1.755    # mean z of top decile
DRAG = 0.5     # bp/day: beta gap + idle cash


def a_sym(h):          # symmetric-normal gross drift of a top-decile name, bp/day, frozen signal held h days
    return IC[h] * SE * np.sqrt(h) * LAM * 100.0 / h


def cost_rt(k, v):     # round-trip cost in bp including minimum commission
    pos = v / k
    comm = max(1.0, 5.0 / pos * 1e4)
    return C0 + 2.0 * (comm - 1.0)


def te(k, se=SE, sf=0.25):   # daily tracking error, %
    return np.sqrt(se * se / k + sf * sf)


print("a_sym(H) bp/day and D(H) bp:")
for h in IC:
    print(f"  H={h:3d}  a_sym={a_sym(h):6.2f}  D={a_sym(h) * h:7.1f}  c/H(27.2)={C0 / h:6.2f}")

print("\nround-trip cost bp: 500k and 20k")
for k in (5, 10, 20, 30, 50, 80, 100):
    print(f"  500k K={k:3d} pos={500000 / k:8.0f} c={cost_rt(k, 5e5):6.1f}")
for k in (1, 2, 3, 4, 5, 8, 10):
    print(f"  20k  K={k:3d} pos={20000 / k:8.0f} c={cost_rt(k, 2e4):6.1f}")

print("\nTE(K) %/day: se=2.33 sf=0 | se=2.33 sf=0.25 | se=1.9 sf=0.25 | se=1.9 sf=0.40 ; goal-2 net need bp/d (third col) ; goal-4 gross need bp/d")
for k in (5, 10, 20, 30, 50, 80, 100):
    t0, t1, t2, t3 = te(k, SE, 0), te(k, SE, .25), te(k, 1.9, .25), te(k, 1.9, .40)
    g4 = 1.96 * SE * 100 / np.sqrt(T * k)
    print(f"  K={k:3d}  {t0:.3f} | {t1:.3f} | {t2:.3f} | {t3:.3f} ; need2={8.52 * t2:5.2f} ({8.52 * t2 * 2.52:5.1f}%/yr) ; need4={g4:5.2f}")

for v, ks, se in ((5e5, (10, 20, 30, 50, 80, 100), 1.9), (2e4, (2, 4, 5, 8, 10), 1.9)):
    print(f"\n=== account {v:.0f}: required kappa for goal1 / goal2 (se={se}, sf=0.25); '--' means > 1 ===")
    print("       " + "".join(f"   H={h:<3d}      " for h in (5, 10, 20, 40, 60)))
    for k in ks:
        row = []
        for h in (5, 10, 20, 40, 60):
            c = cost_rt(k, v)
            n1 = (5.0 / 2.52 + DRAG + c / h) / a_sym(h)
            n2 = (8.52 * te(k, se, .25) + DRAG + c / h) / a_sym(h)
            f = lambda x: f"{x:4.2f}" if x <= 1 else " -- "
            row.append(f"{f(n1)}/{f(n2)}")
        print(f"  K={k:3d}  " + "    ".join(row))
    for kap in (0.3, 0.5, 0.7, 1.0):
        print(f"  kappa={kap}: net %/yr and E[t]")
        for k in ks:
            row = []
            for h in (5, 10, 20, 40, 60):
                net = kap * a_sym(h) - cost_rt(k, v) / h - DRAG
                row.append(f"{net * 2.52:6.1f}/{np.sqrt(T) * net / (te(k, se, .25) * 100):5.2f}")
            print(f"    K={k:3d}  " + "  ".join(row))

print("\ngoal 5: sd of annualised excess across runs = se*sqrt(T)/YRS*sqrt((1-o)/K)")
for se in (2.33, 1.9):
    f = se * np.sqrt(T) / YRS
    print(f"  se={se}: factor={f:.1f}%")
    for mean in (5.0, 8.0):
        lim = (mean / 3.0 / f) ** 2
        print("    mean=%.0f%%: max (1-o)/K = %.5f -> " % (mean, lim) + ", ".join(f"K={k}: 1-o<={min(1, lim * k):.2f}" for k in (5, 10, 20, 30, 50, 100)))

# Monte Carlo: rule 50 (swap when predicted gap > cost) versus score noise
rng = np.random.default_rng(1)
N, K, DAYS = 2500, 20, 400
b5 = IC[5] * SE * np.sqrt(5) * 100       # bp of 5-day drift per z unit
theta = C0 / b5
print(f"\nrule-50 simulation: b(5d)={b5:.1f} bp per z, threshold={theta:.2f} z")
for rho in (0.80, 0.90, 0.97, 0.99):
    s = rng.standard_normal(N)
    held = np.zeros(N, bool)
    held[np.argsort(-s)[:K]] = True
    swaps = 0
    for d in range(DAYS):
        s = rho * s + np.sqrt(1 - rho * rho) * rng.standard_normal(N)
        while True:
            hi = np.where(held)[0]
            lo = np.where(~held)[0]
            w = hi[np.argmin(s[hi])]
            bst = lo[np.argmax(s[lo])]
            if s[bst] - s[w] > theta:
                held[w] = False
                held[bst] = True
                if d >= 50:
                    swaps += 1
            else:
                break
    per_day = swaps / (DAYS - 50)
    print(f"  score lag-1 autocorr {rho}: swaps/day={per_day:5.2f} of K={K}; implied holding={K / max(per_day, 1e-9):6.1f} days; cost={per_day / K * C0 * 2.52:5.1f} %/yr")

# Monte Carlo: goal-5 dispersion formula
o, k = 0.5, 20
common = rng.standard_normal((T, int(k * o))) * SE
vals = []
for r in range(2000):
    own = rng.standard_normal((T, k - int(k * o))) * SE
    vals.append((common.sum() + own.sum()) / k / YRS)
print(f"\ngoal-5 check K={k} o={o}: simulated sd={np.std(vals):.2f}%  formula={SE * np.sqrt(T) / YRS * np.sqrt((1 - o) / k):.2f}%")
