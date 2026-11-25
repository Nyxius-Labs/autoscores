"""Refit the Frey-Osborne classifier under changes to its labels, model and data, and record
each occupation's probability. Output: results/probs.csv (one column per refit) and
results/runs.csv (one row per refit).

  python -m autoscores.run
"""
import pathlib
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, RationalQuadratic
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from . import features, link

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
B = 200  # refits with a random 60 of the 70 labels


def model(kind: str, seed: int = 0):
    if kind == "gp_ard":  # Frey & Osborne: GP classifier, exponentiated-quadratic kernel
        return GaussianProcessClassifier(ConstantKernel(1.0) * RBF(np.ones(9)), n_restarts_optimizer=2,
                                         random_state=seed)
    if kind == "gp_iso":
        return GaussianProcessClassifier(ConstantKernel(1.0) * RBF(1.0), n_restarts_optimizer=2, random_state=seed)
    if kind == "gp_rq":
        return GaussianProcessClassifier(ConstantKernel(1.0) * RationalQuadratic(), n_restarts_optimizer=2,
                                         random_state=seed)
    if kind == "logit":
        return LogisticRegression(C=1.0, max_iter=1000)
    if kind == "forest":
        return RandomForestClassifier(500, min_samples_leaf=3, random_state=seed)
    if kind == "boost":
        return GradientBoostingClassifier(n_estimators=200, max_depth=2, learning_rate=0.05, random_state=seed)
    if kind == "svm":
        return SVC(probability=True, random_state=seed)
    raise ValueError(kind)


def fit(X, train_idx, y, kind="gp_ard", seed=0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return model(kind, seed).fit(X[train_idx], y).predict_proba(X)[:, 1]


def design(x: pd.DataFrame, socs) -> np.ndarray:
    return StandardScaler().fit_transform(x.reindex(socs).fillna(x.mean()))


def jobs(app, X17, X28):
    lab = np.flatnonzero(app.label.notna().values)
    y = app.label.values[lab].astype(int)
    rng = np.random.default_rng(2026)
    out = [("base", "gp_ard", X17, lab, y, 0)]
    for b in range(B):  # label only 60 of the 70 occupations (random subsets, no replacement)
        i = np.sort(rng.choice(len(lab), 60, replace=False))
        out.append((f"sub60_{b:03d}", "gp_ard", X17, lab[i], y[i], b))
    for k in (3, 7):  # experts disagree on k of the 70 labels (7 = 10%)
        for b in range(100):
            yf = y.copy()
            j = rng.choice(len(lab), k, replace=False)
            yf[j] = 1 - yf[j]
            out.append((f"rflip{k}_{b:03d}", "gp_ard", X17, lab, yf, b))
    ones = np.flatnonzero(y == 1)
    for b in range(100):  # 7 of the 37 'automatable' labels changed to 'not': Walsh's (2018) experts
        yf = y.copy()      # marked 29-31 of the 70 as automatable, against Frey & Osborne's 37
        yf[rng.choice(ones, 7, replace=False)] = 0
        out.append((f"down7_{b:03d}", "gp_ard", X17, lab, yf, b))
    for k in range(len(lab)):  # drop one label
        keep = np.delete(np.arange(len(lab)), k)
        out.append((f"drop{k:02d}", "gp_ard", X17, lab[keep], y[keep], 0))
    for k in range(len(lab)):  # flip one label
        yf = y.copy()
        yf[k] = 1 - yf[k]
        out.append((f"flip{k:02d}", "gp_ard", X17, lab, yf, 0))
    for kind in ["gp_iso", "gp_rq", "logit", "forest", "boost", "svm"]:
        out.append((f"model_{kind}", kind, X17, lab, y, 0))
    out.append(("onet2023", "gp_ard", X28, lab, y, 0))
    return out


def heldout_auc(X, lab, y, kind, seed):
    """Frey & Osborne's check: train on a random half of the 70 labels, score the other half."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    while True:
        i = rng.permutation(len(lab))
        tr, te = i[:35], i[35:]
        if 0 < y[tr].sum() < 35 and 0 < y[te].sum() < 35:
            break
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = model(kind, seed).fit(X[lab[tr]], y[tr]).predict_proba(X[lab[te]])[:, 1]
    return kind, roc_auc_score(y[te], p)


def main():
    OUT.mkdir(exist_ok=True)
    app, x17, x28 = features.build()
    socs = app.soc.values
    X17, X28 = design(x17, socs), design(x28, socs)
    js = jobs(app, X17, X28)
    print(len(js), "refits")
    res = Parallel(n_jobs=4, verbose=5)(delayed(fit)(X, tr, y, kind, seed) for _, kind, X, tr, y, seed in js)
    probs = pd.DataFrame(np.column_stack(res), index=socs, columns=[j[0] for j in js])
    probs.insert(0, "published", app.prob.values)
    probs.to_csv(OUT / "probs.csv")
    lab = np.flatnonzero(app.label.notna().values)
    y = app.label.values[lab].astype(int)
    kinds = ["gp_ard", "gp_iso", "gp_rq", "logit", "forest", "boost", "svm"]
    auc = Parallel(n_jobs=4)(delayed(heldout_auc)(X17, lab, y, k, s) for k in kinds for s in range(100))
    pd.DataFrame(auc, columns=["model", "auc"]).groupby("model").auc.mean().to_csv(OUT / "auc.csv")
    lk = link.build(socs)
    lk.to_csv(OUT / "link.csv", index=False)
    print("wrote", OUT / "probs.csv")


if __name__ == "__main__":
    main()
