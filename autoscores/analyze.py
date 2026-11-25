"""Summarize the refits: the share of employment at high risk, occupation-level intervals, and
whether the scores predict employment change after 2012. Writes results/summary.md,
results/shares.csv and results/table1.csv.

  python -m autoscores.analyze
"""
import pathlib

import numpy as np
import pandas as pd
import statsmodels.api as sm

from . import features, link

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
HI, LO = 0.7, 0.3
FAMILIES = {  # run-name prefix -> description
    "flip": "one label changed",
    "drop": "one label left out",
    "sub60": "a random 60 of the 70 labeled",
    "rflip3": "3 random labels changed",
    "rflip7": "7 random labels changed",
    "down7": "7 'automatable' labels changed to 'not'",
    "model_": "other classifiers",
    "onet2023": "O*NET 2023 inputs",
}


def weights(lk: pd.DataFrame) -> pd.Series:
    """ACS 2012 employment per FO occupation (an ACS code's employment split equally among the
    FO occupations assigned to it)."""
    e12 = link.emp(2012)
    n = lk.groupby("acs12").soc.transform("count")
    return pd.Series((lk.acs12.map(e12) / n).fillna(0).values, index=lk.soc.values)


def share(p, w, lo=HI, hi=1.01):
    m = (p >= lo) & (p < hi)
    return float((w * m).sum() / w.sum())


def group_means(v: pd.Series, lk: pd.DataFrame, w: pd.Series) -> pd.Series:
    d = pd.DataFrame({"v": v.values, "w": w.reindex(v.index).values,
                      "g": lk.set_index("soc").group.reindex(v.index).values}).dropna()
    d = d[d.w > 0]
    d["vw"] = d.v * d.w
    a = d.groupby("g")[["vw", "w"]].sum()
    return a.vw / a.w


def growth_fit(p, lk, ge, w, routine=None):
    """Employment-weighted regression of log employment change on the group's mean probability,
    optionally controlling for the group's routine-task index."""
    df = ge.join(group_means(p, lk, w).rename("p"), how="inner")
    cols = ["p"]
    if routine is not None:
        df = df.join(group_means(routine, lk, w).rename("routine"), how="inner")
        cols.append("routine")
    df["dlog"] = np.log(df.emp23 / df.emp12)
    df["dlog"] = df.dlog.clip(*df.dlog.quantile([0.01, 0.99]))
    r = sm.WLS(df.dlog, sm.add_constant(df[cols]), weights=df.emp12).fit(cov_type="HC1")
    return r.params["p"], r.bse["p"], len(df)


def main():
    probs = pd.read_csv(OUT / "probs.csv", index_col=0)
    probs.index.name = "soc"
    app = features.appendix().set_index("soc")
    rout = features.routine(probs.index)
    lk = link.build(probs.index)
    w = weights(lk).reindex(probs.index).fillna(0)
    ge = link.group_employment(lk)
    lk19 = link.build(probs.index, 2019)
    ge19 = link.group_employment(lk19, 2019)

    rows = []
    for c in probs.columns:
        r = dict(run=c, high=share(probs[c], w), low=share(probs[c], w, 0, LO),
                 mean=float((w * probs[c]).sum() / w.sum()))
        r["slope"], r["slope_se"], r["groups"] = growth_fit(probs[c], lk, ge, w)
        r["slope_routine"], r["slope_routine_se"], _ = growth_fit(probs[c], lk, ge, w, rout)
        r["slope19"], r["slope19_se"], r["groups19"] = growth_fit(probs[c], lk19, ge19, w)
        rows.append(r)
    sh = pd.DataFrame(rows).set_index("run")
    sh.to_csv(OUT / "shares.csv")

    def fam(prefix):
        return sh.filter(regex="^" + prefix, axis=0)

    def q(prefix, col="high", f="{:.0%}"):
        s = fam(prefix)[col].dropna()
        lo, hi = (s.min(), s.max()) if len(s) < 20 else s.quantile([0.05, 0.95])
        return f"{f.format(lo)} to {f.format(hi)}"

    base_hi = probs.base >= HI

    def moved(prefix):
        cols = probs.filter(regex="^" + prefix).columns
        return pd.Series([((probs[c] >= HI) != base_hi).mul(w).sum() / w.sum() for c in cols])

    def interval(prefix):
        b = probs.filter(regex="^" + prefix)
        lo, hi = b.quantile(0.05, axis=1), b.quantile(0.95, axis=1)
        return lo, hi, (lo < HI) & (hi >= HI)

    auc = pd.read_csv(OUT / "auc.csv", index_col=0).auc
    L = ["# Results", "",
         "Ranges are min to max for fewer than 20 refits, otherwise the middle 90%.", "",
         "## 1. Reproduction",
         f"- Refit vs published probabilities: Pearson r = {np.corrcoef(probs.base, probs.published)[0, 1]:.3f}; "
         f"same side of 0.7 for {((probs.base >= HI) == (probs.published >= HI)).mean():.0%} of occupations.",
         f"- Share of 2012 ACS employment at p >= 0.7: published {sh.loc['published', 'high']:.1%}, "
         f"refit {sh.loc['base', 'high']:.1%} (paper, BLS 2010 employment: 47%).",
         f"- Held-out AUC (train on 35 labels, test on 35, 100 splits): refit {auc['gp_ard']:.3f} "
         f"(paper: 0.894).", "",
         "## 2. High-risk share",
         "| change | refits | share p >= 0.7 | employment-weighted mean p |", "|---|---|---|---|",
         f"| none | 1 | {sh.loc['base', 'high']:.0%} | {sh.loc['base', 'mean']:.2f} |"]
    for k, d in FAMILIES.items():
        L.append(f"| {d} | {len(fam(k))} | {q(k)} | {q(k, 'mean', '{:.2f}')} |")
    m = moved("flip")
    L += ["",
          f"- One label changed: employment that changes side of 0.7, median {m.median():.1%}, max {m.max():.1%}.",
          f"- 7 random labels changed: median {moved('rflip7').median():.1%}; "
          f"7 'automatable' labels changed to 'not': median {moved('down7').median():.1%}.",
          "- Classifiers (high-risk share, held-out AUC): " + "; ".join(
              f"{k}: {sh.loc['model_' + k, 'high']:.0%}, {auc[k]:.2f}" for k in auc.index if k != "gp_ard") + ".",
          "", "## 3. Occupations"]
    for k in ["sub60", "rflip7", "down7"]:
        lo, hi, st = interval(k)
        L.append(f"- {FAMILIES[k]}: median 90% interval width {(hi - lo).median():.2f}; "
                 f"{(w * st).sum() / w.sum():.0%} of employment in occupations whose interval contains 0.7.")
    F = probs.filter(regex="^flip")
    labeled = app[app.label.notna()].reset_index()
    for soc in ["41-2031", "43-4051"]:
        k = F.loc[soc].idxmin()
        L.append(f"- {app.title[soc]}: published {probs.published[soc]:.2f}; lowest after one change "
                 f"{F.loc[soc, k]:.2f} (relabeling {labeled.title[int(k[4:])]}).")
    lo, hi, _ = interval("rflip7")
    exact = lk.set_index("soc").acs12 == pd.Series(probs.index.str.replace("-", ""), index=probs.index)
    e12 = link.emp(2012)
    t = pd.DataFrame({"title": app.title, "label": app.label, "published": probs.published,
                      "flip1_min": F.min(axis=1), "flip1_max": F.max(axis=1), "flip7_q05": lo, "flip7_q95": hi,
                      "emp2012_m": lk.set_index("soc").acs12.map(e12) / 1e6})[exact.values]
    t.sort_values("emp2012_m", ascending=False).head(10).round(3).to_csv(OUT / "table1.csv")

    def neg(col):
        return (sh[col].dropna() < 0).mean()

    n = sh.slope.notna().sum()
    L += ["", "## 4. Employment change",
          f"- Outcome: log employment change, weighted by 2012 employment; {int(sh.loc['base', 'groups'])} occupation "
          f"groups for 2012-2023, {int(sh.loc['base', 'groups19'])} for 2012-2019 (1% tails clipped).",
          "", "| scores | 2012-2023 | with routine control | 2012-2019 |", "|---|---|---|---|"]
    for r in ["published", "base"]:
        L.append(f"| {r} | {sh.loc[r, 'slope']:.3f} ({sh.loc[r, 'slope_se']:.3f}) | "
                 f"{sh.loc[r, 'slope_routine']:.3f} ({sh.loc[r, 'slope_routine_se']:.3f}) | "
                 f"{sh.loc[r, 'slope19']:.3f} ({sh.loc[r, 'slope19_se']:.3f}) |")
    for k in ["flip", "sub60", "rflip7", "down7", "model_", "onet2023"]:
        L.append(f"| {FAMILIES[k]} | {q(k, 'slope', '{:.3f}')} | {q(k, 'slope_routine', '{:.3f}')} | "
                 f"{q(k, 'slope19', '{:.3f}')} |")
    sp = lambda f: probs.filter(regex=f).corrwith(probs.base, method="spearman")
    L += ["",
          f"- Refits with a negative slope ({n} refits): 2012-2023 {neg('slope'):.0%}; with routine control "
          f"{neg('slope_routine'):.0%}; 2012-2019 {neg('slope19'):.0%}.",
          f"- Refits with slope more than 1.96 SE below zero, 2012-2023: "
          f"{((sh.slope + 1.96 * sh.slope_se) < 0).sum() / n:.0%}; with routine control: "
          f"{((sh.slope_routine + 1.96 * sh.slope_routine_se) < 0).sum() / n:.0%}.",
          f"- Rank correlation with the refit: classifiers {sp('^model_').min():.2f} to {sp('^model_').max():.2f}; "
          f"O*NET 2023 {probs.onet2023.corr(probs.base, method='spearman'):.2f}; "
          f"7 random labels changed (median) {sp('^rflip7').median():.2f}."]
    (OUT / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
