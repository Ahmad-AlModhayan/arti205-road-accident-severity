"""
Profile the DfT 2024 collision file and write the numbers you need for the report.

Get the file (one click, ~20 MB):
  https://www.kaggle.com/datasets/juharimat/road-safety-data-collisions-2024
Unzip and put dft-road-casualty-statistics-collision-2024.csv in this folder.

Optional second year — add dft-road-casualty-statistics-collision-2023.csv from:
  https://www.kaggle.com/datasets/ngducchilly/uk-road-accident-data-19792024
  (that one is the full 1979-2024 archive; filter collision_year == 2023)

Run:  python prepare_dft2024.py
Writes: UK_collisions_model_ready.csv  and  UK_profile.txt
"""

from pathlib import Path
import pandas as pd

FILES = sorted(Path(".").glob("dft-road-casualty-statistics-collision-*.csv"))
OUT = Path("UK_collisions_model_ready.csv")
PROFILE = Path("UK_profile.txt")

TARGET = "collision_severity"
SEV = {1: "Fatal", 2: "Serious", 3: "Slight"}

# Conditions known at the time of the crash. Excludes casualty counts and
# anything recorded as part of the outcome.
KEEP = [
    "collision_index", "collision_year", TARGET,
    "date", "day_of_week", "time",
    "first_road_class", "road_type", "speed_limit",
    "junction_detail", "junction_control",
    "second_road_class",
    "pedestrian_crossing",
    "light_conditions", "weather_conditions", "road_surface_conditions",
    "special_conditions_at_site", "carriageway_hazards",
    "urban_or_rural_area", "number_of_vehicles",
    "local_authority_ons_district",
]

# DfT "missing / unknown" codes. -1 means data missing in every column; 9 or 99
# means unknown in most columns, but NOT everywhere: number_of_vehicles is a real
# count (9 vehicles is valid), and the 2024 re-coded columns use 99 for unknown.
DEFAULT_MISSING = {-1, 9}
MISSING_CODES = {
    "number_of_vehicles": set(),
    "day_of_week": set(),
    "first_road_class": set(),
    "junction_detail": {-1, 99},
    "pedestrian_crossing": {-1, 99},
    "carriageway_hazards": {-1, 99},
    "speed_limit": {-1, 99},
    "light_conditions": {-1},          # 7 = darkness, lighting unknown: still darkness
    "urban_or_rural_area": {-1, 3},    # 3 = unallocated
}


def main() -> None:
    if not FILES:
        raise SystemExit(
            "No dft-road-casualty-statistics-collision-*.csv here.\n"
            "Download it from Kaggle first — see the docstring."
        )

    df = pd.concat([pd.read_csv(f, low_memory=False) for f in FILES], ignore_index=True)
    print(f"read {len(FILES)} file(s): {', '.join(f.name for f in FILES)}")
    print(f"{len(df):,} collisions, {df.shape[1]} columns")

    absent = [c for c in KEEP if c not in df.columns]
    if absent:
        print("\n!! not in the file, skipping:", absent)
        print("   available columns:", list(df.columns))
    cols = [c for c in KEEP if c in df.columns]
    df = df[cols]

    df[TARGET] = df[TARGET].map(SEV).fillna(df[TARGET].astype(str))
    df.to_csv(OUT, index=False)

    L = []
    a = L.append
    years = sorted(df["collision_year"].unique()) if "collision_year" in df else ["?"]
    a(f"UK DfT road casualty statistics — collisions {', '.join(map(str, years))}")
    a(f"source: UK Department for Transport, obtained via Kaggle")
    a(f"rows: {len(df):,}   columns kept: {df.shape[1]}   duplicates: {df.duplicated().sum()}")

    vc = df[TARGET].value_counts()
    a(f"\nTarget — {TARGET}")
    for k in ["Fatal", "Serious", "Slight"]:
        if k in vc:
            a(f"  {k:<8} {vc[k]:>8,}  {vc[k]/len(df)*100:6.2f}%")
    a(f"  imbalance ratio {vc.max()/vc.min():.0f} : 1")

    a("\nDfT 'missing/unknown' codes per column (-1, and 9 or 99 where they mean unknown) — treat as missing, not a category")
    for c in df.columns:
        if c in (TARGET, "collision_index", "date", "time"):
            continue
        codes = MISSING_CODES.get(c, DEFAULT_MISSING)
        n = df[c].isin(codes).sum() if pd.api.types.is_numeric_dtype(df[c]) else 0
        n += df[c].isna().sum()
        if n:
            a(f"  {c:<45} {n:>7,}  {n/len(df)*100:5.1f}%")

    a("\nColumns kept")
    for c in df.columns:
        kind = "numeric" if pd.api.types.is_numeric_dtype(df[c]) else "text"
        a(f"  {c:<45} {kind:<8} {df[c].nunique():>6} distinct")

    a("\nNote: junction_control is -1 for most collisions that are not at a junction,")
    a("so its -1 count is mostly 'not applicable', not truly missing.")
    a("Note: special_conditions_at_site is -1 for about 59% of rows because many police")
    a("forces now report through CRASH, which does not fill this field.")
    a("\nNote: most columns are integer CODES, not labels. The DfT publishes a lookup")
    a("table ('Road Safety Open Dataset Data Guide') that maps each code to its meaning.")
    a("Map them before you plot or run SHAP, or your figures will be unreadable.")

    PROFILE.write_text("\n".join(L), encoding="utf-8")
    print(f"\nwrote {OUT} and {PROFILE}\n")
    print("\n".join(L[:12]))


if __name__ == "__main__":
    main()
