"""Download the public inputs into raw/ (about 1.3 GB, mostly the two ACS PUMS files).

  python scripts/get_data.py            # everything
  python scripts/get_data.py --no-pums  # skip PUMS; use the shipped data/emp_acs*.csv instead
"""
import argparse
import csv
import pathlib
import re
import urllib.request
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "raw"
ONET = "https://www.onetcenter.org/dl_files"
FILES = {
    "fo.pdf": "https://www.oxfordmartin.ox.ac.uk/downloads/academic/The_Future_of_Employment.pdf",
    "db_17_0.zip": f"{ONET}/db_17_0.zip",                 # O*NET 17.0, July 2012
    "db_28_0.zip": f"{ONET}/database/db_28_0_text.zip",   # O*NET 28.0, August 2023
    "onet_2010_to_2019.csv": "https://www.onetcenter.org/taxonomy/2019/walk/2010_to_2019_Crosswalk.csv?fmt=csv",
}
PUMS = {f"pums_{y}.zip": f"https://www2.census.gov/programs-surveys/acs/data/pums/{y}/1-Year/csv_pus.zip"
        for y in (2012, 2019, 2023)}
NEEDED = ["Abilities.txt", "Knowledge.txt", "Skills.txt", "Work Activities.txt", "Work Context.txt"]


def get(name, url):
    dest = RAW / name
    if dest.exists():
        return
    print("downloading", url)
    req = urllib.request.Request(url, headers={"User-Agent": "autoscores-replication/1.0"})
    with urllib.request.urlopen(req) as r, open(dest, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)


def unpack(zname, folder):
    out = RAW / folder
    out.mkdir(exist_ok=True)
    z = zipfile.ZipFile(RAW / zname)
    for n in z.namelist():
        if n.split("/")[-1] in NEEDED:
            (out / n.split("/")[-1]).write_bytes(z.read(n))


def parse_appendix():
    """Appendix table of Frey & Osborne: rank, probability, hand label (if any), SOC code, title."""
    import pypdf
    r = pypdf.PdfReader(RAW / "fo.pdf")
    text = "\n".join(r.pages[i].extract_text() for i in range(56, len(r.pages)))
    pat = re.compile(r"^(\d+)\.\s+([\d.]+)\s+(?:([01])\s+)?(\d\d-\d{4})\s+(.*)$")
    rows = []
    for ln in text.splitlines():
        ln = ln.strip()
        m = pat.match(ln)
        if m:
            rows.append(list(m.groups()))
        elif rows and ln and not re.match(r"^(Computerisable|Rank|\d+\.?$)", ln):
            rows[-1][4] += " " + ln
    assert len(rows) == 702 and sum(r[2] is not None for r in rows) == 70, (len(rows),)
    with open(RAW / "fo_appendix.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "prob", "label", "soc", "title"])
        w.writerows(rows)
    print("parsed 702 occupations, 70 labels")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-pums", action="store_true")
    a = ap.parse_args()
    RAW.mkdir(exist_ok=True)
    for k, v in FILES.items():
        get(k, v)
    if not a.no_pums:
        for k, v in PUMS.items():
            get(k, v)
    unpack("db_17_0.zip", "onet17")
    unpack("db_28_0.zip", "onet28")
    parse_appendix()


if __name__ == "__main__":
    main()
