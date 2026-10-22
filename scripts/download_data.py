"""Download the public datasets used by the capstone and regenerate derived files.

    python scripts/download_data.py            # fetch anything missing, rebuild the cleaned RT-IoT2022 csv
    python scripts/download_data.py --force    # re-download everything

Sources
  RT-IoT2022       UCI ML Repository, dataset 942 (CC BY 4.0)  -> data/RT_IOT2022
  Dry Bean         UCI ML Repository, dataset 602 (CC BY 4.0)  -> data/Dry_Bean_Dataset.csv (converted from the .xlsx in the archive)
  Hotel bookings   Antonio, Almeida & Nunes (2019), Data in Brief, via the TidyTuesday mirror
                   (rfordatascience/tidytuesday, 2020-02-11)    -> data/hotel_bookings.csv

data/RT_IOT2022_cleaned.csv is derived from the raw file with exactly the cleaning of the regression
notebook (drop `Unnamed: 0` and `bwd_URG_flag_count`, drop duplicate rows, reset index).
"""
import argparse
import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
SOURCES = {
    "rt_iot2022": "https://archive.ics.uci.edu/static/public/942/rt-iot2022.zip",
    "dry_bean": "https://archive.ics.uci.edu/static/public/602/dry+bean+dataset.zip",
    "hotel": "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-02-11/hotels.csv",
}


def fetch(url: str) -> bytes:
    print(f"downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "ml-capstone-download/1.0"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read()


def get_rt_iot(force=False):
    out = DATA / "RT_IOT2022"
    if out.exists() and not force:
        return out
    with zipfile.ZipFile(io.BytesIO(fetch(SOURCES["rt_iot2022"]))) as z:
        out.write_bytes(z.read("RT_IOT2022"))
    return out


def get_dry_bean(force=False):
    out = DATA / "Dry_Bean_Dataset.csv"
    if out.exists() and not force:
        return out
    with zipfile.ZipFile(io.BytesIO(fetch(SOURCES["dry_bean"]))) as z:
        xlsx = z.read("DryBeanDataset/Dry_Bean_Dataset.xlsx")
    pd.read_excel(io.BytesIO(xlsx)).to_csv(out, index=False)  # needs openpyxl
    return out


def get_hotel(force=False):
    out = DATA / "hotel_bookings.csv"
    if out.exists() and not force:
        return out
    out.write_bytes(fetch(SOURCES["hotel"]))
    return out


def clean_rt_iot(raw: Path, out: Path):
    """Same cleaning as regression/regression.ipynb."""
    df = pd.read_csv(raw).drop(columns=["Unnamed: 0", "bwd_URG_flag_count"])
    df = df.drop_duplicates().reset_index(drop=True)
    df.to_csv(out, index=False)
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="re-download even if the file exists")
    args = ap.parse_args()
    DATA.mkdir(exist_ok=True)
    raw = get_rt_iot(args.force)
    get_dry_bean(args.force)
    get_hotel(args.force)
    cleaned = DATA / "RT_IOT2022_cleaned.csv"
    if args.force or not cleaned.exists():
        df = clean_rt_iot(raw, cleaned)
        print(f"wrote {cleaned.name}: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print("data ready in", DATA)


if __name__ == "__main__":
    main()
