"""Figure 1: share of employment at high risk under each change to the labels, model or data.
Figure 2: that share against the number of 'automatable' labels. Figure 3: slope of 2012-2023
employment change on the score.

  python -m autoscores.figure
"""
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
INK, MUTED, GRID, SERIES, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#fcfcfb"
ROWS = [("^flip", "One label changed\n(70 refits)"),
        ("^rflip7", "7 random labels changed\n(100 refits)"),
        ("^walsh", "Simulated single experts\n(200 refits)"),
        ("^model_", "Other classifiers\n(6 refits)"),
        ("^onet2023", "O*NET 2023 data\n(1 refit)")]
NOTE = "Each dot is one refit; black lines span the middle 90%. Dashed line: original labels."


def style(ax, xlim, fmt, title):
    ax.set_facecolor(SURF)
    ax.set_xlim(*xlim)
    ticks = [t for t in ax.get_xticks() if xlim[0] <= t <= xlim[1]]
    ax.set_xticks(ticks)
    ax.set_xticklabels([fmt(t) for t in ticks], fontsize=8, color=MUTED)
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.6, len(ROWS) - 0.4)
    ax.set_title(title, fontsize=9.5, color=INK, loc="left")
    for s_ in ["top", "right", "left"]:
        ax.spines[s_].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", color=GRID)


def dots(ax, sh, col, ref, rng):
    y = np.arange(len(ROWS))[::-1]
    for yi, (pat, _) in zip(y, ROWS):
        v = sh.filter(regex=pat, axis=0)[col].dropna().values
        jit = rng.uniform(-0.18, 0.18, len(v)) if len(v) > 1 else np.zeros(1)
        ax.scatter(v, yi + jit, s=9, color=SERIES, alpha=0.35, linewidth=0, zorder=2)
        if len(v) > 1:
            lo, hi = np.quantile(v, [0.05, 0.95])
            ax.hlines(yi, lo, hi, color=INK, lw=1.5, zorder=3)
    ax.axvline(ref, color=MUTED, lw=1, ls=(0, (3, 3)), zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels([r[1] for r in ROWS], fontsize=8, color=INK)


def main():
    sh = pd.read_csv(OUT / "shares.csv", index_col=0)
    rng = np.random.default_rng(0)
    pct = lambda v: f"{100 * v:.0f}%"
    # Figure 1: high-risk share under each change
    fig, ax = plt.subplots(figsize=(5.6, 2.9), facecolor=SURF)
    dots(ax, sh, "high", sh.loc["base", "high"], rng)
    style(ax, (-0.015, 0.6), pct, "Share of 2012 employment with probability of 0.7 or more")
    fig.text(0.01, 0.01, NOTE, fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT / "figure1.png", dpi=200, facecolor=SURF)
    # Figure 3: slope of employment change on the score, without and with a routine-task control
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.9), sharey=True, facecolor=SURF)
    for ax, col, title in [(axes[0], "slope", "Scores only"), (axes[1], "slope_routine", "With routine-task index")]:
        dots(ax, sh, col, sh.loc["base", col], rng)
        ax.axvline(0, color=MUTED, lw=0.8, zorder=0)
        style(ax, (-0.62, 0.17), lambda v: f"{v:.1f}", title)
    axes[1].tick_params(axis="y", labelleft=False)
    fig.text(0.01, 0.01, NOTE + " One refit per panel lies off the right edge.",
             fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT / "figure3.png", dpi=200, facecolor=SURF)
    figure2(sh)
    print("wrote figures 1-3")


def figure2(sh):
    """High-risk share against the number of 'automatable' labels, one dot per simulated expert."""
    runs = pd.read_csv(OUT / "runs.csv", index_col=0)
    w = sh.filter(regex="^walsh", axis=0).join(runs.n_automatable)
    fig, ax = plt.subplots(figsize=(4.6, 3.1), facecolor=SURF)
    ax.set_facecolor(SURF)
    ax.axvspan(29, 31.1, color=GRID, zorder=0)
    ax.text(30.05, 0.66, "experts'\naverage", ha="center", va="top", fontsize=7.5, color=MUTED)
    ax.scatter(w.n_automatable, w.high, s=12, color=SERIES, alpha=0.5, linewidth=0, zorder=2)
    ax.scatter([37], [sh.loc["base", "high"]], s=40, color=INK, zorder=3)
    ax.annotate("original labels (37)", (37, sh.loc["base", "high"]), xytext=(42, 0.25), fontsize=7.5, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.set_xlim(5, 65)
    ax.set_ylim(-0.02, 0.68)
    ax.set_yticks([0, 0.2, 0.4, 0.6])
    ax.set_yticklabels(["0%", "20%", "40%", "60%"], fontsize=8, color=MUTED)
    ax.tick_params(axis="x", labelsize=8, labelcolor=MUTED, color=GRID)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Occupations labeled automatable (of 70)", fontsize=8.5, color=INK)
    ax.set_title("Share of 2012 employment with probability of 0.7 or more", fontsize=9, color=INK, loc="left")
    for s_ in ["top", "right", "left"]:
        ax.spines[s_].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis="y", color=GRID, lw=0.6, zorder=0)
    fig.tight_layout()
    fig.savefig(OUT / "figure2.png", dpi=200, facecolor=SURF)


if __name__ == "__main__":
    main()
