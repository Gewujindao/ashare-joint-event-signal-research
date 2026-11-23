"""Draw figures/summary.png: holdout result of the GBM scheme.

All numbers are constants copied from the WINDOW HOLD sections of
blind_holdout/holdout_run/gbm_account_backtest/out_account_v2_full.txt, out_account_v2_full_perturb.txt
and out_account_v2_full_random.txt, and (size-only reference rule) of
blind_holdout/design/return_attribution/out_attribution_main.txt; the market data are not
distributed, so nothing is read from disk.
Run from the repository root: python figures/make_figures.py
"""
from pathlib import Path

import matplotlib.pyplot as plt

# Shared style
BLUE, GRAY, ACCENT = "#1f4e79", "#9aa5b1", "#d98c1f"
REF_EDGE = "#5f6b78"    # outlined, hatched bar: reference rule
INK, MUTED, GRID = "#1a1a1a", "#5f6b78", "#e9ecef"
FS_TITLE, FS_LABEL, FS_SMALL = 12, 9, 8
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": FS_SMALL,
    "axes.linewidth": 0.6,
    "axes.edgecolor": INK,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.color": INK,
    "ytick.color": INK,
})

# Holdout 2024-07-01 to 2026-10-09, conservative valuation, 0.1% slippage per side
N_DAYS = 551            # out_account_v2_full.txt ("551 days")
SCHEME_ANN = 48.27      # out_account_v2_full.txt ("annS   48.27")
BENCH_ANN = 21.11       # out_account_v2_full.txt ("annB   21.11")
EXCESS_PTS = 27.17      # out_account_v2_full.txt ("excess %/yr   27.17")
T_STAT = 2.33           # out_account_v2_full.txt ("t  2.33")
NAV_SCHEME = 2.3661     # out_account_v2_full_random.txt ("strategy 2.3661")
NAV_RANDOM_P975 = 1.5785  # out_account_v2_full_random.txt ("q97.5 1.5785")
N_RANDOM = 200          # out_account_v2_full_random.txt ("200 seeds")
PERTURB_MIN = 25.08     # out_account_v2_full_perturb.txt ("min 25.08")
PERTURB_MEAN = 26.37    # out_account_v2_full_perturb.txt ("mean 26.37")
PERTURB_SD = 1.03       # out_account_v2_full_perturb.txt ("sd 1.03")
N_PERTURB = 20          # out_account_v2_full_perturb.txt ("20 perturbed runs")
SIZE_ONLY_ANN = 55.30   # design/return_attribution/out_attribution_main.txt, line 72 ("(ii) smallest-cap 50, 5 swaps/day  ann   55.30")


def clean(ax, grid_axis):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    getattr(ax, f"{grid_axis}axis").grid(True, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=FS_SMALL)


def label(fig, x, y, letter, title):
    fig.text(x, y, letter, fontsize=FS_LABEL, fontweight="bold", color=INK,
             ha="left", va="center")
    fig.text(x + 0.022, y, title, fontsize=FS_LABEL, color=INK, ha="left", va="center")


def main():
    fig = plt.figure(figsize=(8, 4.5), dpi=200, facecolor="white")
    fig.text(0.02, 0.945, f"Holdout run: GBM scheme {SCHEME_ANN:.2f}% a year, benchmark B {BENCH_ANN:.2f}%, "
             f"size-only rule {SIZE_ONLY_ANN:.2f}%",
             fontsize=FS_TITLE, fontweight="bold", color=INK, ha="left", va="center")
    fig.text(0.02, 0.885, f"Holdout 2024-07-01 to 2026-10-09 ({N_DAYS} trading days); "
             "net of costs, 0.1% slippage per side, conservative valuation",
             fontsize=FS_LABEL, color=MUTED, ha="left", va="center")

    # a: annualised return bars (left, full height)
    ax = fig.add_axes([0.085, 0.10, 0.30, 0.64])
    vals = [SCHEME_ANN, BENCH_ANN]
    ax.bar([0, 1], vals, width=0.58, color=[BLUE, GRAY], zorder=2)
    ax.bar([2], [SIZE_ONLY_ANN], width=0.58, color="white", edgecolor=REF_EDGE,
           hatch="////", linewidth=0.9, zorder=2)
    for x, v in zip([0, 1, 2], vals + [SIZE_ONLY_ANN]):
        ax.text(x, v + 1.2, f"{v:.2f}%", ha="center", va="bottom",
                fontsize=FS_LABEL, color=INK, fontweight="bold")
    ax.set_xticks([0, 1, 2], ["GBM\nscheme", "Benchmark\nB", "Size-only\nrule"],
                  fontsize=FS_LABEL)
    ax.set_xlim(-0.55, 2.55)
    ax.set_ylim(0, 70)
    ax.set_yticks([0, 20, 40, 60])
    ax.set_ylabel("Annualized net return (%)", fontsize=FS_LABEL, color=INK)
    clean(ax, "y")
    ax.tick_params(axis="x", length=0)
    ax.annotate("", xy=(0.5, SCHEME_ANN), xytext=(0.5, BENCH_ANN),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=0.8, shrinkA=0, shrinkB=0))
    ax.text(0.57, (SCHEME_ANN + BENCH_ANN) / 2, f"+{EXCESS_PTS:.2f} pts\nt = {T_STAT:.2f}",
            ha="left", va="center", fontsize=FS_SMALL, color=INK, linespacing=1.4)

    # b: final NAV vs random selection
    bx = fig.add_axes([0.56, 0.535, 0.415, 0.165])
    names = ["GBM scheme", f"Random, 97.5th pct"]
    navs = [NAV_SCHEME, NAV_RANDOM_P975]
    bx.barh([1, 0], navs, height=0.58, color=[BLUE, GRAY], zorder=2)
    for y, v in zip([1, 0], navs):
        bx.text(v + 0.04, y, f"{v:.4f}", ha="left", va="center", fontsize=FS_SMALL,
                color=INK, fontweight="bold")
    bx.set_yticks([1, 0], names, fontsize=FS_SMALL)
    bx.set_ylim(-0.6, 1.6)
    bx.set_xlim(0, 3)
    bx.set_xticks([0, 1, 2, 3])
    bx.set_xlabel("Final NAV", fontsize=FS_LABEL, color=INK, labelpad=2)
    clean(bx, "x")
    bx.tick_params(axis="y", length=0)

    # c: excess return across perturbed reruns
    cx = fig.add_axes([0.56, 0.10, 0.415, 0.13])
    cx.errorbar([PERTURB_MEAN], [0], xerr=[PERTURB_SD], fmt="o", color=BLUE, ms=6,
                elinewidth=1.2, capsize=3, capthick=1.2, zorder=3)
    cx.plot([PERTURB_MIN], [0], marker="|", ms=12, mew=1.4, color=ACCENT, zorder=4)
    cx.annotate(f"Min {PERTURB_MIN:.2f}", (PERTURB_MIN, 0), xytext=(0, -10),
                textcoords="offset points", ha="center", va="top", fontsize=FS_SMALL,
                color=ACCENT)
    cx.annotate(f"Mean {PERTURB_MEAN:.2f} ± {PERTURB_SD:.2f} sd", (PERTURB_MEAN, 0),
                xytext=(10, 8), textcoords="offset points", ha="right", va="bottom",
                fontsize=FS_SMALL, color=INK)
    cx.set_yticks([])
    cx.spines["left"].set_visible(False)
    cx.set_ylim(-1, 1)
    cx.set_xlim(0, 30)
    cx.set_xticks([0, 10, 20, 30])
    cx.set_xlabel("Annualized net excess (pts)", fontsize=FS_LABEL, color=INK, labelpad=2)
    clean(cx, "x")

    # Panel letters and titles
    label(fig, 0.02, 0.795, "a", "Annualized return")
    label(fig, 0.435, 0.795, "b", f"Final NAV vs. {N_RANDOM} random-selection runs")
    label(fig, 0.435, 0.325, "c", f"Excess return across 5 start dates ({N_PERTURB} reruns)")

    out = Path(__file__).resolve().parent / "summary.png"
    fig.savefig(out, dpi=200, facecolor="white")


if __name__ == "__main__":
    main()
