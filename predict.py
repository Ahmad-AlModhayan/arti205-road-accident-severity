"""
Predict road traffic collision severity (Slight, Serious, Fatal) using the trained model.

Usage:
  python predict.py --demo
  python predict.py --interactive
  python predict.py --json '{"speed_limit": 60, "urban_or_rural_area": "Rural", "vehicle_type": "Heavy goods (7.5t+)"}'
  python predict.py --input collisions.csv --output predictions.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path(__file__).parent / "Analysis" / "results" / "final_model.joblib"
DATA_PATH = Path(__file__).parent / "Data" / "UK_collisions_final.csv"

DEFAULT_RECORD = {
    "month": 11,
    "day_of_week": "Fri",
    "hour": 16,
    "first_road_class": "A",
    "road_type": "Single carriageway",
    "speed_limit": 30,
    "junction_detail": "T or staggered",
    "junction_control": "Give way / uncontrolled",
    "second_road_class": "Unclassified",
    "pedestrian_crossing": "None within 50 m",
    "light_conditions": "Daylight",
    "weather_conditions": "Fine",
    "road_surface_conditions": "Dry",
    "carriageway_hazards": "No hazard",
    "urban_or_rural_area": "Urban",
    "number_of_vehicles": 2,
    "vehicle_type": "Car / Taxi",
    "driver_age": "25-39",
    "sex_of_driver": "Male",
    "vehicle_manoeuvre": "Going ahead",
    "junction_location": "Approaching junction",
    "journey_purpose": "Personal / leisure",
    "towing_and_articulation": "No tow / articulation",
    "propulsion_type": "Petrol",
    "vehicle_age": "9-14 yrs",
}


def load_bundle(model_path: Path = MODEL_PATH) -> dict:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Saved model not found at {model_path}. Run Analysis/03_Model_Tuning_and_Evaluation.ipynb first."
        )
    return joblib.load(model_path)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "speed_x_vehicles" not in out.columns:
        out["speed_x_vehicles"] = out["speed_limit"] * out["number_of_vehicles"]
    if "speed_per_vehicle" not in out.columns:
        out["speed_per_vehicle"] = (out["speed_limit"] / out["number_of_vehicles"]).round(2)
    speed_band = pd.cut(
        out["speed_limit"], bins=[-1, 35, 55, 200], labels=["20-30mph", "40-50mph", "60-70mph"]
    ).astype(str)
    dark_flag = out["light_conditions"].eq("Daylight").map({True: "Daylight", False: "Dark"})
    if "speed_x_area" not in out.columns:
        out["speed_x_area"] = out["urban_or_rural_area"] + " | " + speed_band
    if "light_x_speed" not in out.columns:
        out["light_x_speed"] = dark_flag + " | " + speed_band
    if "vehicle_x_speed" not in out.columns:
        out["vehicle_x_speed"] = out["vehicle_type"] + " | " + speed_band
    if "vehicle_x_manoeuvre" not in out.columns:
        out["vehicle_x_manoeuvre"] = out["vehicle_type"] + " | " + out["vehicle_manoeuvre"]
    return out


def predict_records(df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    pipe, cols, classes = bundle["pipeline"], bundle["columns"], bundle["classes"]
    enriched = add_engineered_features(df)
    missing = [c for c in cols if c not in enriched.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X = enriched[cols]
    pred_idx = pipe.predict(X)
    probs = pipe.predict_proba(X)

    out = enriched.copy()
    out["predicted_severity"] = [classes[i] for i in pred_idx]
    for i, cls in enumerate(classes):
        out[f"prob_{cls}"] = probs[:, i]
    return out


def print_prediction(row: pd.Series, classes: list[str], title: str) -> None:
    print(f"\n--- {title} ---")
    print(f"Predicted severity: {row['predicted_severity']}")
    for cls in classes:
        print(f"  {cls:<8}: {row[f'prob_{cls}']:6.1%}")


def run_interactive(bundle: dict) -> None:
    cols, classes = bundle["columns"], bundle["classes"]
    ref = pd.read_csv(DATA_PATH) if DATA_PATH.exists() else None
    record = dict(DEFAULT_RECORD)

    print("Enter collision details (press Enter to keep the default shown in brackets):")
    for col in DEFAULT_RECORD:
        default_val = record[col]
        if ref is not None and col in ref.columns and ref[col].dtype == object:
            choices = sorted(ref[col].unique().tolist())
            print(f"\n{col} choices:")
            for idx, ch in enumerate(choices, 1):
                marker = " (default)" if ch == default_val else ""
                print(f"  {idx}. {ch}{marker}")
            raw = input(f"Select 1-{len(choices)} or type value [{default_val}]: ").strip()
            if raw.isdigit() and 1 <= int(raw) <= len(choices):
                record[col] = choices[int(raw) - 1]
            elif raw:
                record[col] = raw
        else:
            raw = input(f"{col} [{default_val}]: ").strip()
            if raw:
                record[col] = int(raw)

    res = predict_records(pd.DataFrame([record]), bundle)
    print_prediction(res.iloc[0], classes, "Interactive Prediction")


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict traffic collision severity.")
    parser.add_argument("--demo", action="store_true", help="Run two example predictions.")
    parser.add_argument("--interactive", action="store_true", help="Prompt for collision inputs.")
    parser.add_argument("--json", type=str, help="JSON dict of fields overriding defaults.")
    parser.add_argument("--input", type=Path, help="Input CSV file of collisions.")
    parser.add_argument("--output", type=Path, default=Path("predictions.csv"), help="Output CSV path.")
    args = parser.parse_args()

    bundle = load_bundle()
    classes = bundle["classes"]

    if args.interactive:
        run_interactive(bundle)
        return

    if args.input:
        df_in = pd.read_csv(args.input)
        res = predict_records(df_in, bundle)
        res.to_csv(args.output, index=False)
        print(f"Wrote {len(res):,} predictions to {args.output}")
        return

    if args.json:
        rec = {**DEFAULT_RECORD, **json.loads(args.json)}
        res = predict_records(pd.DataFrame([rec]), bundle)
        print_prediction(res.iloc[0], classes, "Custom JSON Prediction")
        return

    # Default / --demo
    low_risk = dict(DEFAULT_RECORD)
    high_risk = {
        **DEFAULT_RECORD,
        "hour": 2,
        "speed_limit": 60,
        "urban_or_rural_area": "Rural",
        "light_conditions": "Dark, no lighting",
        "junction_detail": "Not at junction",
        "junction_control": "Not at junction",
        "second_road_class": "Not at junction",
        "number_of_vehicles": 1,
        "vehicle_type": "Large / other motorcycle",
        "driver_age": "60+",
        "vehicle_manoeuvre": "Going ahead",
        "junction_location": "Not at junction",
    }
    res = predict_records(pd.DataFrame([low_risk, high_risk]), bundle)
    print_prediction(res.iloc[0], classes, "Example 1: Urban 30 mph Car Collision (Daylight)")
    print_prediction(res.iloc[1], classes, "Example 2: Rural 60 mph Motorcycle Collision (Dark, No Lighting, 60+)")


if __name__ == "__main__":
    main()
