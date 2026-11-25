"""Frey-Osborne inputs: the published appendix (702 occupations, 70 hand labels) and the nine
O*NET 'bottleneck' variables, aggregated from O*NET-SOC to six-digit 2010 SOC codes."""
import pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "raw"

# (file, element name, scale) as listed in Frey & Osborne (2017), Table I. "Level" scale except the
# work-context item, which has only a context (frequency) scale.
VARS = [
    ("Abilities", "Finger Dexterity", "LV"),
    ("Abilities", "Manual Dexterity", "LV"),
    ("Work Context", "Cramped Work Space, Awkward Positions", "CX"),
    ("Abilities", "Originality", "LV"),
    ("Knowledge", "Fine Arts", "LV"),
    ("Skills", "Social Perceptiveness", "LV"),
    ("Skills", "Negotiation", "LV"),
    ("Skills", "Persuasion", "LV"),
    ("Work Activities", "Assisting and Caring for Others", "LV"),
]
SHORT = ["finger", "manual", "cramped", "original", "finearts", "socialperc", "negot", "persuade", "caring"]


def appendix() -> pd.DataFrame:
    df = pd.read_csv(RAW / "fo_appendix.csv", dtype={"soc": str})
    df["label"] = df["label"].astype("Int64")
    return df


def onet(release: str) -> pd.DataFrame:
    """O*NET-SOC x 9 variables for release: '17' (July 2012, the
    first release on the SOC 2010 taxonomy, closest to Frey & Osborne's data) or '28' (2023)."""
    cols = {}
    for (fname, elem, scale), short in zip(VARS, SHORT):
        t = pd.read_csv(RAW / f"onet{release}" / f"{fname}.txt", sep="\t", dtype=str)
        t = t[(t["Element Name"] == elem) & (t["Scale ID"] == scale)]
        cols[short] = t.set_index("O*NET-SOC Code")["Data Value"].astype(float)
    return pd.DataFrame(cols).dropna()


# Frey & Osborne list BLS OES 2010 "hybrid" codes. These have no O*NET-SOC 2010 rows of the same
# code; we map them to the 2010 SOC code or prefix that covers the same work.
MANUAL = {"29-1111": "29-1141", "13-1078": "13-107", "15-1179": "15-1122", "15-1799": "15-1199",
          "25-3999": "25-30", "29-2037": "29-203", "29-2799": "29-2099", "29-9799": "29-9099",
          "31-9799": "31-9099", "39-4831": "39-4031", "47-4799": "47-4099", "49-9799": "49-9099"}


def to_soc(x: pd.DataFrame, socs, crosswalk: dict | None = None) -> pd.DataFrame:
    """Average O*NET-SOC rows into each six-digit SOC code. Codes that BLS reports only as
    aggregates (e.g. 29-1060, 25-1000) match on their non-zero prefix."""
    x = x.copy()
    if crosswalk:  # 2019 O*NET-SOC -> 2010 O*NET-SOC
        x.index = [crosswalk.get(c, c) for c in x.index]
        x = x.groupby(level=0).mean()
    six = pd.Series(x.index.str[:7], index=x.index)
    out = {}
    for soc in socs:
        code = MANUAL.get(soc, soc)
        hit = six[six == code].index
        if len(hit) == 0:
            prefix = code.rstrip("0")
            hit = six[six.str.startswith(prefix)].index
        if len(hit):
            out[soc] = x.loc[hit].mean()
    return pd.DataFrame(out).T


def crosswalk_2019_to_2010() -> dict:
    cw = pd.read_csv(RAW / "onet_2010_to_2019.csv", dtype=str)
    # many-to-one only; for 2019 codes that came from several 2010 codes keep the first
    return cw.drop_duplicates("O*NET-SOC 2019 Code").set_index("O*NET-SOC 2019 Code")["O*NET-SOC 2010 Code"].to_dict()


def build():
    app = appendix()
    x17 = to_soc(onet("17"), app.soc)
    x28 = to_soc(onet("28"), app.soc, crosswalk_2019_to_2010())
    return app, x17, x28


if __name__ == "__main__":
    app, x17, x28 = build()
    print(len(app), "occupations;", len(x17), "matched in O*NET 17.0;", len(x28), "in O*NET 28.0")
    print("missing 17.0:", sorted(set(app.soc) - set(x17.index)))
    print("missing 28.0:", sorted(set(app.soc) - set(x28.index)))


# Routine-task index in the style of Acemoglu & Autor (2011), from the same O*NET release.
ROUTINE = {
    "cognitive": [("Work Context", "Importance of Repeating Same Tasks", "CX", 1),
                  ("Work Context", "Importance of Being Exact or Accurate", "CX", 1),
                  ("Work Context", "Structured versus Unstructured Work", "CX", -1)],
    "manual": [("Work Context", "Pace Determined by Speed of Equipment", "CX", 1),
               ("Work Activities", "Controlling Machines and Processes", "IM", 1),
               ("Work Context", "Spend Time Making Repetitive Motions", "CX", 1)],
}


def routine(socs, release: str = "17") -> pd.Series:
    """Standardized routine cognitive + routine manual index by SOC code."""
    parts = {}
    for group, items in ROUTINE.items():
        cols = {}
        for fname, elem, scale, sign in items:
            t = pd.read_csv(RAW / f"onet{release}" / f"{fname}.txt", sep="\t", dtype=str)
            t = t[(t["Element Name"] == elem) & (t["Scale ID"] == scale)]
            v = t.set_index("O*NET-SOC Code")["Data Value"].astype(float)
            cols[elem] = sign * (v - v.mean()) / v.std()
        parts[group] = pd.DataFrame(cols).mean(axis=1)
    x = to_soc(pd.DataFrame(parts).dropna(), socs)
    z = (x - x.mean()) / x.std()
    return z.sum(axis=1).rename("routine")
