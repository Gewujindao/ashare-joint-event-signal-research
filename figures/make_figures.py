"""Draw figures/summary.png: blind-holdout result of the frozen scheme.

All numbers are constants copied from
blind_holdout_2026-11/evaluator/留出段评测报告.txt (第二部分);
the market data are not distributed, so nothing is read from disk.
Run from the repository root: python figures/make_figures.py
"""
from pathlib import Path

import matplotlib.pyplot as plt

# Shared style
BLUE, GRAY, ACCENT = "#1f4e79", "#9aa5b1", "#d98c1f"
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
N_DAYS = 551            # 留出段评测报告.txt line 3 ("551 个交易日")
SCHEME_ANN = 48.27      # 留出段评测报告.txt line 31 ("策略年化 48.27%")
BENCH_ANN = 21.11       # 留出段评测报告.txt line 31 ("基准 B 年化 21.11%")
EXCESS_PTS = 27.17      # 留出段评测报告.txt line 31 ("相差 +27.17 个点")
T_STAT = 2.33           # 留出段评测报告.txt line 32 ("t 值 2.33")
NAV_SCHEME = 2.3661     # 留出段评测报告.txt line 34 ("策略期末净值 2.3661")
NAV_RANDOM_P975 = 1.5785  # 留出段评测报告.txt line 34 ("97.5% 分位 1.5785")
N_RANDOM = 200          # 留出段评测报告.txt line 34 ("200 个种子")
PERTURB_MIN = 25.08     # 留出段评测报告.txt line 36 ("最小 25.08")
PERTURB_MEAN = 26.37    # 留出段评测报告.txt line 36 ("均值 26.37")
PERTURB_SD = 1.03       # 留出段评测报告.txt line 36 ("标准差 1.03")
N_PERTURB = 20          # 留出段评测报告.txt line 36 ("20 次扰动运行")


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
    fig.text(0.02, 0.945, "In a single blind holdout run, the frozen scheme beat its "
             f"benchmark by +{EXCESS_PTS:.2f} pts a year",
             fontsize=FS_TITLE, fontweight="bold", color=INK, ha="left", va="center")
    fig.text(0.02, 0.885, f"Holdout 2024-07-01 to 2026-10-09 ({N_DAYS} trading days); "
             "net of costs, 0.1% slippage per side, conservative valuation",
             fontsize=FS_LABEL, color=MUTED, ha="left", va="center")

    # a: annualised return bars (left, full height)
    ax = fig.add_axes([0.085, 0.10, 0.30, 0.64])
    vals = [SCHEME_ANN, BENCH_ANN]
    ax.bar([0, 1], vals, width=0.58, color=[BLUE, GRAY], zorder=2)
    for x, v in zip([0, 1], vals):
        ax.text(x, v + 1.2, f"{v:.2f}%", ha="center", va="bottom",
                fontsize=FS_LABEL, color=INK, fontweight="bold")
    ax.set_xticks([0, 1], ["Frozen scheme", "Benchmark B"], fontsize=FS_LABEL)
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(0, 60)
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
    names = ["Frozen scheme", f"Random, 97.5th pct"]
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
    label(fig, 0.435, 0.325, "c", f"Excess return across {N_PERTURB} perturbed reruns")

    out = Path(__file__).resolve().parent / "summary.png"
    fig.savefig(out, dpi=200, facecolor="white")


if __name__ == "__main__":
    main()
