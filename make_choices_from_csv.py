"""
Generate artifacts/choices.json by scanning your dataset CSV
and extracting unique values for each categorical column used by the model.

Usage:
  python make_choices_from_csv.py --csv "Road Accident Data.csv" --bundle artifacts/model.pkl --limit 200
"""
import argparse, json, joblib, pandas as pd, numpy as np, os
from collections import OrderedDict

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to dataset CSV")
    ap.add_argument("--bundle", default="artifacts/model.pkl", help="Model bundle path")
    ap.add_argument("--limit", type=int, default=200, help="Max unique values per column to include")
    args = ap.parse_args()

    bundle = joblib.load(args.bundle)
    X_cols   = bundle.get("X_cols", [])
    cat_cols = bundle.get("cat_cols", [])
    if not X_cols or not cat_cols:
        raise RuntimeError("Bundle must contain X_cols and cat_cols")

    print(f"Loading CSV: {args.csv}")
    df = pd.read_csv(args.csv)
    # ensure only columns in X_cols
    cols = [c for c in cat_cols if c in df.columns]
    out = OrderedDict()

    for c in cols:
        s = df[c].dropna().astype(str).str.strip()
        vals = s[s != ""].value_counts().index.tolist()
        if len(vals) > args.limit:
            vals = vals[:args.limit]  # top-N
        out[c] = vals
        print(f"{c}: {len(vals)} values")

    os.makedirs("artifacts", exist_ok=True)
    with open(os.path.join("artifacts", "choices.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("Wrote artifacts/choices.json")

if __name__ == "__main__":
    main()
