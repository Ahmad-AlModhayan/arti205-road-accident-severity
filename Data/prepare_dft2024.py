"""
Profile the DfT 2024 collision and vehicle files and write the numbers you need for the report.

Get the files (~20 MB each):
  https://www.kaggle.com/datasets/juharimat/road-safety-data-collisions-2024
  https://data.dft.gov.uk/road-accidents-safety-data/dft-road-casualty-statistics-vehicle-2024.csv
Unzip and put dft-road-casualty-statistics-collision-2024.csv and
dft-road-casualty-statistics-vehicle-2024.csv in this folder.

Optional second year — add dft-road-casualty-statistics-collision-2023.csv from:
  https://www.kaggle.com/datasets/ngducchilly/uk-road-accident-data-19792024
  (that one is the full 1979-2024 archive; filter collision_year == 2023)

Run:  python prepare_dft2024.py
Writes: UK_collisions_model_ready.csv  and  UK_profile.txt
"""

from pathlib import Path
import pandas as pd

FILES = sorted(Path(".").glob("dft-road-casualty-statistics-collision-*.csv"))
VEH_FILES = sorted(Path(".").glob("dft-road-casualty-statistics-vehicle-*.csv"))
OUT = Path("UK_collisions_model_ready.csv")
PROFILE = Path("UK_profile.txt")

TARGET = "collision_severity"
SEV = {1: "Fatal", 2: "Serious", 3: "Slight"}

VEHICLE_PRIORITY = [
    "Heavy goods (7.5t+)",
    "Large / other motorcycle",
    "Moped / small motorcycle",
    "Pedal cycle",
    "Van / medium goods",
    "Bus / Coach",
    "Other (e-scooter / agri / mobility)",
    "Car / Taxi",
    "Unknown",
]
VEHICLE_RANK = {k: i for i, k in enumerate(VEHICLE_PRIORITY)}


def map_vehicle_group(code: int) -> str:
    if code == 1:
        return "Pedal cycle"
    if code in (2, 3):
        return "Moped / small motorcycle"
    if code in (4, 5, 23, 97):
        return "Large / other motorcycle"
    if code in (8, 9):
        return "Car / Taxi"
    if code in (10, 11):
        return "Bus / Coach"
    if code in (19, 20, 98):
        return "Van / medium goods"
    if code == 21:
        return "Heavy goods (7.5t+)"
    if code in (16, 17, 18, 22, 33, 90):
        return "Other (e-scooter / agri / mobility)"
    return "Unknown"


MANOEUVRE_MAP = {
    1: "Reversing", 2: "Parked", 3: "Waiting to go ahead", 4: "Slowing or stopping",
    5: "Moving off", 6: "U-turn", 7: "Turning left", 8: "Waiting to turn left",
    9: "Turning right", 10: "Waiting to turn right", 11: "Changing lane left",
    12: "Changing lane right", 13: "Overtaking moving", 14: "Overtaking stationary",
    15: "Overtaking nearside", 19: "Going ahead", 20: "Parking / manoeuvring",
    99: "Unknown", -1: "Unknown",
}
JLOC_MAP = {
    0: "Not at junction", 1: "Approaching junction", 2: "Cleared junction",
    3: "Leaving roundabout", 4: "Entering roundabout", 5: "Leaving main road",
    6: "Entering main road", 7: "Entering slip road", 8: "Mid junction",
    9: "Unknown", -1: "Unknown",
}
SEX_MAP = {1: "Male", 2: "Female", 3: "Unknown", -1: "Unknown"}
JOURNEY_MAP = {
    1: "Work journey", 2: "Commuting", 6: "Unknown", 7: "School / education",
    8: "Emergency response", 9: "Personal / leisure", -1: "Unknown",
}
TOW_MAP = {
    0: "No tow / articulation", 1: "Articulated vehicle", 2: "Double / multiple trailer",
    3: "Caravan", 4: "Single trailer", 5: "Other tow", 9: "Unknown", -1: "Unknown",
}
PROP_MAP = {
    1: "Petrol", 2: "Diesel", 3: "Electric", 5: "Gas", 6: "Gas / LPG",
    7: "Gas / bi-fuel", 8: "Hybrid electric", 12: "Electric diesel", -1: "Unknown",
}


def aggregate_vehicles(veh_files: list[Path]) -> pd.DataFrame:
    use = [
        "collision_index", "vehicle_reference", "vehicle_type", "age_of_driver",
        "vehicle_manoeuvre", "junction_location", "sex_of_driver",
        "journey_purpose_of_driver", "towing_and_articulation", "propulsion_code",
        "age_of_vehicle",
    ]
    veh = pd.concat(
        [pd.read_csv(f, usecols=use, low_memory=False) for f in veh_files],
        ignore_index=True,
    )
    veh["vehicle_type_grp"] = veh["vehicle_type"].map(map_vehicle_group)
    veh["_rank"] = veh["vehicle_type_grp"].map(VEHICLE_RANK)
    vt = (
        veh.sort_values("_rank")
        .drop_duplicates("collision_index")[["collision_index", "vehicle_type_grp"]]
        .rename(columns={"vehicle_type_grp": "vehicle_type"})
    )
    va = (
        veh[veh["age_of_driver"] > 0]
        .groupby("collision_index")["age_of_driver"]
        .mean()
        .rename("mean_driver_age")
    )
    v1 = veh.sort_values("vehicle_reference").drop_duplicates("collision_index").copy()
    v1["vehicle_manoeuvre"] = v1["vehicle_manoeuvre"].map(MANOEUVRE_MAP).fillna("Unknown")
    v1["junction_location"] = v1["junction_location"].map(JLOC_MAP).fillna("Unknown")
    v1["sex_of_driver"] = v1["sex_of_driver"].map(SEX_MAP).fillna("Unknown")
    v1["journey_purpose"] = v1["journey_purpose_of_driver"].map(JOURNEY_MAP).fillna("Unknown")
    v1["towing_and_articulation"] = v1["towing_and_articulation"].map(TOW_MAP).fillna("Unknown")
    v1["propulsion_type"] = v1["propulsion_code"].map(PROP_MAP).fillna("Unknown")
    v1["vehicle_age"] = (
        pd.cut(v1["age_of_vehicle"].where(v1["age_of_vehicle"] >= 0), bins=[-1, 3, 8, 14, 100],
               labels=["0-3 yrs", "4-8 yrs", "9-14 yrs", "15+ yrs"])
        .astype(str)
        .replace("nan", "Unknown")
    )
    agg = vt.merge(va, on="collision_index", how="left").merge(
        v1[["collision_index", "vehicle_manoeuvre", "junction_location", "sex_of_driver",
            "journey_purpose", "towing_and_articulation", "propulsion_type", "vehicle_age"]],
        on="collision_index", how="left",
    )
    agg["driver_age"] = (
        pd.cut(agg["mean_driver_age"], bins=[0, 24, 39, 59, 120], labels=["Under 25", "25-39", "40-59", "60+"])
        .astype(str)
        .replace("nan", "Unknown")
    )
    return agg[[
        "collision_index", "vehicle_type", "driver_age", "sex_of_driver",
        "vehicle_manoeuvre", "junction_location", "journey_purpose",
        "towing_and_articulation", "propulsion_type", "vehicle_age",
    ]]

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

    if VEH_FILES:
        veh_agg = aggregate_vehicles(VEH_FILES)
        df = df.merge(veh_agg, on="collision_index", how="left")
        for c in [
            "vehicle_type", "driver_age", "sex_of_driver", "vehicle_manoeuvre",
            "junction_location", "journey_purpose", "towing_and_articulation",
            "propulsion_type", "vehicle_age",
        ]:
            df[c] = df[c].fillna("Unknown")

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
