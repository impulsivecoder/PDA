import pandas as pd
from pathlib import Path


INPUT_PATH = Path("final_processed_data.csv")


df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")
df["date"] = pd.to_datetime(df["date"], format="mixed", dayfirst=True)

countries = sorted(df["country"].unique())

# Build risky date sets per country
risky_sets = {
    c: set(df.loc[(df["Y_class"] == 1) & (df["country"] == c), "date"])
    for c in countries
}


all_common = set.intersection(*risky_sets.values())

total_risky = {c: len(risky_sets[c]) for c in countries}

unique_risky = {
    c: len(risky_sets[c] - set.union(*[risky_sets[o] for o in countries if o != c]))
    for c in countries
}

common_count = len(all_common)


col_w   = 14
row_lbl = 26

header = f"{'':>{row_lbl}}" + "".join(f"{c:>{col_w}}" for c in countries)
sep    = "-" * (row_lbl + col_w * len(countries))

print(sep)
print(header)
print(sep)

rows = [
    ("Total Risky days",       total_risky),
    ("Spesific to that country",    unique_risky),
    (f"{len(countries)} countries combined", {c: common_count for c in countries}),
]

for label, data in rows:
    print(f"{label:>{row_lbl}}" + "".join(f"{data[c]:>{col_w}}" for c in countries))

print(sep)
