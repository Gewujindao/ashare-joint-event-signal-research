# Arithmetic check of the planning figures. Reads no data. Pure stdlib.
from math import sqrt
from statistics import NormalDist

nd = NormalDist()
f = sqrt(551)
gross, style, within = 8.89, 3.88, 5.01          # bp/day, combined 1,085 days (gbm_model_fit)
cost = {"root": 3.52 + 0.5, "mine": 1.92 + 0.5}  # bp/day at K=50, H=10
print("A. sqrt(551) = %.3f ; ratio for t=2: %.4f ; for 80%% (t=2.84): %.4f" % (f, 2 / f, 2.84 / f))
for name, c in cost.items():
    for te in (0.602, 0.680, 0.818):
        net = gross - c
        print("  now  %-4s TE %.3f : net %.2f bp/d = %.1f%%/yr, t551 = %.2f" % (name, te, net, net * 2.52, f * net / (te * 100)))
print("A1. needed (TE fixed): TE, cost, t -> net, gross, within, x_orth, portfolio gain")
for name, c in cost.items():
    for te in (0.602, 0.680, 0.818):
        for t in (2.0, 2.84):
            net = t / f * te * 100
            g = net + c
            w = g - style
            x = sqrt(max(0.0, w * w - within * within))
            print("  %-4s TE %.3f t %.2f : net %.2f gross %.2f within %.2f x %.2f gain %+.2f" % (name, te, t, net, g, w, x, w - within))
print("A2. needed TE at fixed gross 8.89")
for name, c in cost.items():
    for t in (2.0, 2.84):
        print("  %-4s t %.2f : TE <= %.3f %%/day" % (name, t, (gross - c) * f / t / 100))
print("A3. bucket-neutral route (style dropped): TE_w, cost, t -> within needed, x")
for name, c in cost.items():
    for tew in (0.538, 0.40, 0.33):
        for t in (2.0, 2.84):
            w = t / f * tew * 100 + c
            x = sqrt(max(0.0, w * w - within * within))
            print("  %-4s TEw %.3f t %.2f : within %.2f x %.2f" % (name, tew, t, w, x))
print("A4. IC needed. drift per unit IC at H=10, sigma_e=233bp")
for q in (0.02, 0.10, 0.30):
    z = nd.inv_cdf(1 - q)
    lam = nd.pdf(z) / q
    per = lam * 233 * sqrt(10) / 10
    print("  top %.0f%%: lam %.3f, drift/IC %.1f bp/d ; IC for x=3.17: kappa .5 %.3f, .7 %.3f ; x=6.70: %.3f, %.3f"
          % (q * 100, lam, per, 3.17 / per / .5, 3.17 / per / .7, 6.70 / per / .5, 6.70 / per / .7))
print("A5. what a given x buys (TE .68, root): x -> within, t551")
for x in (1.0, 1.5, 2.0, 2.5, 3.0, 3.17, 4.0, 5.0, 6.7):
    w = sqrt(within ** 2 + x ** 2)
    for te in (0.602, 0.680, 0.818):
        print("  x %.2f TE %.3f: within %.2f, t(root) %.2f, t(mine) %.2f" % (x, te, w, f * (w + style - cost["root"]) / te / 100, f * (w + style - cost["mine"]) / te / 100))
print("A6. power. paired account diff: SD = sqrt(2(1-o)) * 2.33%/sqrt(50)")
for o in (0.5, 0.7, 0.85):
    sd = sqrt(2 * (1 - o)) * 233 / sqrt(50)
    print("  o %.2f: SD %.1f bp/d ; t for +0.92bp over 1085 d %.2f ; over 727 d %.2f" % (o, sd, 0.92 / sd * sqrt(1085), 0.92 / sd * sqrt(727)))
for sd in (15.0, 20.0, 30.0):
    print("  decile-level x: SD %.0f bp/d -> SE 727 d %.2f, 358 d %.2f, 1085 d %.2f" % (sd, sd / sqrt(727), sd / sqrt(358), sd / sqrt(1085)))

print("B. volume: 3300 stocks x 3100 days")
sd_ = 3300 * 3100
for nm, bars, bpr in (("1min", 240, 8), ("5min", 48, 10), ("15min", 16, 12)):
    rows = sd_ * bars
    print("  %-5s rows %.3e ; raw 30B/row %.1f GB ; csv 70B/row %.1f GB ; parquet %dB/row %.1f GB ; x0.82 (avg 2700 stocks) %.1f GB ; per year %.2f GB"
          % (nm, rows, rows * 30 / 1e9, rows * 70 / 1e9, bpr, rows * bpr / 1e9, rows * bpr * 0.82 / 1e9, rows * bpr / 1e9 / 12.3))
print("  local 15min gz json: 49.5MB / (848*341*16 rows) = %.1f B/row" % (49.5e6 / (848 * 341 * 16)))
print("  daily features: %d stock-days ; 40 float32 = %.2f GB total, %.0f MB/yr uncompressed" % (sd_, sd_ * 40 * 4 / 1e9, 3300 * 250 * 40 * 4 / 1e6))
print("  NN tensors fp16 1min: %.1f GB ; 5min: %.1f GB" % (sd_ * 240 * 6 * 2 / 1e9, sd_ * 48 * 6 * 2 / 1e9))
print("  stock-years %.0f ; at 50 ms each %.0f min ; tushare 8000 rows/call: 1min %.0f calls, 5min %.0f calls" % (3300 * 12.3, 3300 * 12.3 * 0.05 / 60, sd_ * 240 / 8000, sd_ * 48 / 8000))

print("C. NN")
p = (6 * 32 * 5 + 32) + 2 * (32 * 32 * 5 + 32) + 3 * (32 * 32 + 32 * 32 + 64) + 33
print("  params %d" % p)
for nm, L in (("1min", 240), ("5min", 48)):
    act = L * 32 * 3 * 4 * 3 / 1e6          # MB per day-encoding, fp32, x3 training overhead
    samp = act * 20
    flop = L * (5 * 6 * 32 + 2 * 5 * 32 * 32) * 2 * 3
    print("  %s: %.3f MB/day-enc, %.2f MB/sample(20d) fp32 ; batch 10GB %d / 14GB %d ; AMP %d / %d ; full 2500 xsec %.1f GB fp32, %.1f AMP"
          % (nm, act, samp, 10000 / samp, 14000 / samp, 20000 / samp, 28000 / samp, 2500 * samp / 1000, 1250 * samp / 1000))
    n_enc = 2500 * 1940 / 5 * 20
    print("     FLOP/day-enc fwd+bwd %.1f M ; epoch (every 5th day) %.2e FLOP ; at 2 TFLOPS %.0f s, at 0.5 TFLOPS %.0f s ; all days x5"
          % (flop / 1e6, n_enc * flop, n_enc * flop / 2e12, n_enc * flop / 0.5e12))
print("  samples: 2500 x 1940 = %.2e stock-days ; independent 10-day blocks %d" % (2500 * 1940, 1940 // 10))
