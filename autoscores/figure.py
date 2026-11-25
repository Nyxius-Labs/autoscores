"""Figure 1: share of employment at high risk (left) and the link between scores and 2012-2023
employment change (right), under each change to the labels, model or data.

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
        ("^sub60", "60 of the 70 labeled\n(200 refits)"),
        ("^rflip7", "7 random labels changed\n(100 refits)"),
        ("^down7", "7 'automatable' labels\nchanged to 'not' (100)"),
        ("^walsh", "Simulated single experts,\nWalsh 2018 (200 refits)"),
        ("^model_", "Other classifiers\n(6 refits)"),
        ("^onet2023", "O*NET 2023 data\n(1 refit)")]


def main():
    sh = pd.read_csv(OUT / "shares.csv", index_col=0)
    rng = np.random.default_rng(0)
    y = np.arange(len(ROWS))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.3), sharey=True, facecolor=SURF)
    panels = [(axes[0], "high", "Share of 2012 employment with\nprobability of 0.7 or more", sh.loc["base", "high"],
               (-0.015, 0.6), lambda v: f"{100 * v:.0f}%"),
              (axes[1], "slope", "Change in log employment 2012-2023\nper unit of probability", sh.loc["base", "slope"],
               (-0.62, 0.06), lambda v: f"{v:.2f}")]
    for ax, col, title, ref, xlim, fmt in panels:
        ax.set_facecolor(SURF)
        for yi, (pat, _) in zip(y, ROWS):
            v = sh.filter(regex=pat, axis=0)[col].dropna().values
            if len(v) == 0:
                continue
            jit = rng.uniform(-0.18, 0.18, len(v)) if len(v) > 1 else np.zeros(1)
            ax.scatter(v, yi + jit, s=10, color=SERIES, alpha=0.35, linewidth=0, zorder=2)
            if len(v) > 1:
                lo, hi = np.quantile(v, [0.05, 0.95])
                ax.hlines(yi, lo, hi, color=INK, lw=1.5, zorder=3)
        ax.axvline(ref, color=MUTED, lw=1, ls=(0, (3, 3)), zorder=0)
        ax.text(ref + 0.006, len(ROWS) - 0.1, "original labels", ha="left", va="center", fontsize=7.5, color=MUTED)
        if col == "slope":
            ax.axvline(0, color=GRID, lw=1, zorder=0)
        ax.set_xlim(*xlim)
        ticks = ax.get_xticks()
        ax.set_xticks(ticks)
        ax.set_xticklabels([fmt(t) for t in ticks], fontsize=8, color=MUTED)
        ax.set_xlim(*xlim)
        ax.set_ylim(-0.6, len(ROWS) + 0.3)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        for s in ["top", "right", "left"]:
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", color=GRID)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([r[1] for r in ROWS], fontsize=8.5, color=INK)
    fig.text(0.01, 0.01, "Each dot is one refit of the Frey-Osborne classifier; black lines span the middle 90%. "
             "Employment from the 2012 and 2023 ACS.", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / "figure1.png", dpi=200, facecolor=SURF)
    print("wrote", OUT / "figure1.png")
    figure2(sh)


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
    print("wrote", OUT / "figure2.png")


if __name__ == "__main__":
    main()
