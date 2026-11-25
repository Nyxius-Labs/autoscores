"""Civilian employment by SOC code from the ACS 1-year PUMS (person weights)."""
import pathlib
import zipfile

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "raw"


def employment(year: int) -> pd.Series:
    z = zipfile.ZipFile(RAW / f"pums_{year}.zip")
    parts = []
    for name in z.namelist():
        if not name.endswith(".csv"):
            continue
        for chunk in pd.read_csv(z.open(name), usecols=["SOCP", "PWGTP", "ESR"],
                                 dtype={"SOCP": str}, chunksize=500_000):
            chunk = chunk[chunk.ESR.isin([1, 2]) & chunk.SOCP.notna()]
            parts.append(chunk.groupby("SOCP").PWGTP.sum())
    s = pd.concat(parts).groupby(level=0).sum()
    s.index = s.index.str.strip()
    return s


if __name__ == "__main__":
    for y in (2012, 2019, 2023):
        s = employment(y)
        s.rename("emp").to_csv(ROOT / "data" / f"emp_acs{y}.csv")
        print(y, len(s), f"{s.sum()/1e6:.1f}M")
