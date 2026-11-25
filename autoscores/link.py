"""Link Frey-Osborne occupations (2010 SOC) to ACS PUMS occupation codes in 2012 and 2023.

ACS PUMS codes are six characters, with 'X' standing for any digit in aggregated groups
(e.g. 119XXX = other managers). Each SOC code is assigned to the most specific ACS code that
matches it. For 2023 (SOC 2018 basis), 2010 codes are first mapped to 2018 codes with the O*NET
2010-to-2019 crosswalk. Occupation groups that can be followed from 2012 to 2023 are the connected
components of the resulting 2012-code <-> 2023-code graph.
"""
import pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = ROOT / "raw"


def _match(code6: str, acs_codes) -> str | None:
    """Most specific ACS code matching a six-digit SOC code (no dash). ACS groups are written
    either with 'X' wildcards (119XXX) or with trailing zeros (291020 = 29-102x)."""
    best, best_n = None, -1
    for a in acs_codes:
        if len(a) != 6:
            continue
        if a == code6:
            return a
        if "X" in a:
            ok = all(c == "X" or c == d for c, d in zip(a, code6))
            n = sum(c != "X" for c in a)
        else:
            stem = a.rstrip("0")
            ok = a.endswith("0") and code6.startswith(stem)
            n = len(stem)
        if ok and n > best_n:
            best, best_n = a, n
    if best is None:  # BLS aggregate codes such as 29-1060 or 25-1000
        p = code6.rstrip("0")
        for a in acs_codes:
            if a.startswith(p):
                return a
    return best


def emp(year: int) -> pd.Series:
    s = pd.read_csv(DATA / f"emp_acs{year}.csv", dtype={"SOCP": str}).set_index("SOCP").emp
    return s[~s.index.str.startswith("55")]  # drop military


def soc2010_to_2018() -> dict:
    cw = pd.read_csv(RAW / "onet_2010_to_2019.csv", dtype=str)
    m = {}
    for a, b in zip(cw["O*NET-SOC 2010 Code"].str[:7], cw["O*NET-SOC 2019 Code"].str[:7]):
        m.setdefault(a, set()).add(b)
    return m


def build(socs, later: int = 2023) -> pd.DataFrame:
    """One row per FO occupation: its ACS 2012 code and its comparable 2012-to-later group id."""
    e12, e23 = emp(2012), emp(later)
    cw = soc2010_to_2018()
    rows = []
    for s in socs:
        a12 = _match(s.replace("-", ""), e12.index)
        targets = cw.get(s) or set().union(*[v for k, v in cw.items() if k.startswith(s.rstrip("0"))] or [set()])
        a23 = {_match(t.replace("-", ""), e23.index) for t in targets} - {None}
        rows.append((s, a12, a23))
    # union-find over ACS codes
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for _, a12, a23 in rows:
        if a12 is None:
            continue
        for b in a23:
            parent[find(("12", a12))] = find(("23", b))
    out = pd.DataFrame([(s, a12, find(("12", a12)) if a12 else None) for s, a12, _ in rows],
                       columns=["soc", "acs12", "group"])
    return out


def group_employment(link: pd.DataFrame, later: int = 2023) -> pd.DataFrame:
    """Employment in 2012 and the later year for each comparable group (built with the same year)."""
    e12, e23 = emp(2012), emp(later)
    parent_of = {}
    for g in link.group.dropna().unique():
        parent_of[g] = g
    # collect member codes of each group
    l12 = link.dropna(subset=["group"]).drop_duplicates("acs12")
    g12 = l12.groupby("group").acs12.apply(lambda c: e12.reindex(c).sum())
    # 2023 members: recompute from the crosswalk
    cw = soc2010_to_2018()
    mem23 = {}
    for s, g in zip(link.soc, link.group):
        if g is None:
            continue
        targets = cw.get(s) or set().union(*[v for k, v in cw.items() if k.startswith(s.rstrip("0"))] or [set()])
        for t in targets:
            a = _match(t.replace("-", ""), e23.index)
            if a:
                mem23.setdefault(g, set()).add(a)
    # a 2023 code must belong to one group only; groups were built so this holds
    g23 = pd.Series({g: e23.reindex(sorted(c)).sum() for g, c in mem23.items()})
    return pd.DataFrame({"emp12": g12, "emp23": g23}).dropna()
