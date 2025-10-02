import os
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(page_title="Accident Severity Predictor", layout="wide")

st.title("🚦 Accident Severity Predictor")
st.caption("Uses internal model at **artifacts/model.pkl**. Dropdown options are populated from **artifacts/choices.json** if present, otherwise sensible defaults.")

# ---------- Paths ----------
DEFAULT_BUNDLE_PATH = os.path.join("artifacts", "model.pkl")
CHOICES_PATH = os.path.join("artifacts", "choices.json")

# ---------- Load model bundle ----------
if not os.path.exists(DEFAULT_BUNDLE_PATH):
    st.error("Model bundle not found at artifacts/model.pkl. Please place the saved model there.")
    st.stop()

@st.cache_resource
def _load_bundle(path: str):
    return joblib.load(path)

try:
    bundle = _load_bundle(DEFAULT_BUNDLE_PATH)
except Exception as e:
    st.error(f"Failed to load model bundle: {e}")
    st.stop()

def bget(key, default=None):
    return bundle.get(key, default) if isinstance(bundle, dict) else default

model       = bget("model")
preproc     = bget("preproc_tree")
X_cols      = bget("X_cols", [])
cat_cols    = list(bget("cat_cols", []))
num_cols    = list(bget("num_cols", []))
classes     = list(bget("classes_", []))
metadata    = bget("metadata", {})

if any(x is None for x in [model, preproc]) or not X_cols:
    st.error("Bundle missing required keys: 'model', 'preproc_tree', 'X_cols'.")
    st.stop()

# ---------- Patch: normalize OHE categories to strings ----------
def _patch_ohe_categories_to_str(preproc):
    try:
        cat_trf = preproc.named_transformers_.get("cat", None)
        if cat_trf is None:
            return None
        ohe = getattr(cat_trf, "named_steps", {}).get("onehot", None)
        if ohe is None and hasattr(cat_trf, "categories_"):
            ohe = cat_trf  # directly an OHE
        if ohe is None or not hasattr(ohe, "categories_"):
            return None
        new_cats = []
        for cats in ohe.categories_:
            cats_list = []
            for c in list(cats):
                if isinstance(c, str):
                    cats_list.append(c)
                elif c is None or (isinstance(c, float) and np.isnan(c)):
                    cats_list.append(np.nan)
                else:
                    cats_list.append(str(c))
            new_cats.append(np.array(cats_list, dtype=object))
        ohe.categories_ = new_cats
        return ohe
    except Exception:
        return None

ohe = _patch_ohe_categories_to_str(preproc)

# ---------- Load choices from file if available ----------
choices_file = {}
if os.path.exists(CHOICES_PATH):
    try:
        with open(CHOICES_PATH, "r", encoding="utf-8") as f:
            choices_file = json.load(f)
    except Exception as e:
        st.warning(f"Could not load choices.json: {e}")

# ---------- Curated defaults (aligned to your dataset) ----------
DEFAULTS = {
    "Day_name": ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"],
    "Time_of_Day": ["Morning","Afternoon","Evening","Night"],
    "Month": list(range(1,13)),
    "Season": ["Winter","Spring","Summer","Autumn"],
    "Urban_or_Rural_Area": ["Urban","Rural"],
    "Light_Conditions": ["Daylight","Dark_lit","Dark_unlit","Dark_none","Dark_unknown"],
    "Weather_Conditions": ["Fine_no_high","Rain_no_high","Fine_high_winds","Rain_high_winds","Fog_mist","Snow_no_high","Snow_high_winds","Other"],
    "Road_Surface_Conditions": ["Dry","Wet or damp","Frost or ice","Snow","Flood_3cm_plus"],
    "Road_Type": ["Single carriageway","Dual carriageway","Roundabout","One way street","Slip road"],
    "Junction_Detail": ["Not at junction","T or staggered junction","Crossroads","Roundabout","Private drive or entrance","Mini-roundabout","More than 4 arms (not roundabout)","Slip road","Other junction"],
    "Junction_Control": ["Give way or uncontrolled","Unknown","Auto traffic signal","Not at junction or within 20 metres","Stop sign","Authorised person"],
    # High-card columns: provide free-text unless choices.json supplies lists
    # "Local_Authority_(District)": [...],
    # "Police_Force": [...],
}

# Merge choices from: OHE categories_ -> choices.json -> DEFAULTS (in that precedence for each column)
def _merge_choices_for_col(col):
    merged = []
    # from OHE (if available)
    if ohe is not None and hasattr(ohe, "categories_") and col in cat_cols:
        try:
            idx = cat_cols.index(col)
            merged.extend([c for c in ohe.categories_[idx] if not (isinstance(c, float) and np.isnan(c))])
        except Exception:
            pass
    # from choices.json
    if col in choices_file and isinstance(choices_file[col], list):
        merged.extend(choices_file[col])
    # from curated defaults
    if col in DEFAULTS:
        merged.extend(DEFAULTS[col])
    # dedupe, stringify and sort
    merged = [str(x) for x in merged if str(x).strip() != ""]
    seen = set()
    unique_sorted = []
    for x in merged:
        if x not in seen:
            seen.add(x)
            unique_sorted.append(x)
    # keep Month unsorted numeric
    if col == "Month":
        return [int(x) for x in unique_sorted if str(x).isdigit() and 1 <= int(x) <= 12]
    return sorted(unique_sorted, key=lambda s: s.lower())

# Build choices dict for all categorical columns
cat_choices = {c: _merge_choices_for_col(c) for c in cat_cols}

with st.expander("ℹ️ Model Info", expanded=False):
    st.write("**Classes:**", classes if classes else "(not provided)")
    st.write("**Features (X_cols):**")
    st.code("\\n".join(map(str, X_cols)))
    st.write("**Choice sources:** OHE categories ➝ choices.json ➝ curated defaults")
    st.write("Loaded choices.json:", os.path.exists(CHOICES_PATH))
    try:
        st.json(metadata if isinstance(metadata, dict) else {"metadata": str(metadata)})
    except Exception:
        st.write(metadata)

st.subheader("🧍 Single Prediction")
st.caption("Dropdowns are populated from the dataset/encoder. Provide values and click Predict.")

# Optional: raw Speed_limit input -> derives High_Speed (>=60) without changing model
st.markdown("**Optional helper:** If you prefer, set a `Speed_limit` and we'll auto-toggle `High_Speed` (>= 60 mph).")

with st.form("single_pred_form"):
    speed_limit = st.number_input("Speed_limit (optional, will set High_Speed>=60 automatically)", min_value=0, max_value=80, value=30, step=10)

    cols = st.columns(3)
    values = {}
    for i, col_name in enumerate(X_cols):
        with cols[i % 3]:
            if col_name in cat_cols:
                options = ["(leave blank)"] + cat_choices.get(col_name, [])
                choice = st.selectbox(col_name, options, index=0, key=f"sel_{col_name}")
                values[col_name] = ("" if choice == "(leave blank)" else choice)
            else:
                if col_name in {"Month"}:
                    # Prefer dropdown for Month to avoid out-of-range
                    month_opts = _merge_choices_for_col("Month")
                    mchoice = st.selectbox("Month", month_opts, index=5 if 6 in month_opts else 0, key="month_sel")
                    values[col_name] = int(mchoice)
                elif col_name in {"High_Speed"}:
                    # Will be overridden by speed_limit if provided
                    values[col_name] = st.number_input(col_name, min_value=0, max_value=1, value=0, step=1)
                elif col_name in {"Hazard_Flag"}:
                    values[col_name] = st.number_input(col_name, min_value=0, max_value=1, value=0, step=1)
                else:
                    values[col_name] = st.number_input(col_name, value=0)

    submitted = st.form_submit_button("⚡ Predict")

# Utilities
def coerce_input_types(record: dict):
    out = {}
    for k, v in record.items():
        if v is None or v == "":
            out[k] = np.nan
            continue
        if k in num_cols or k in {"Month", "High_Speed", "Hazard_Flag"}:
            try:
                out[k] = int(v)
            except Exception:
                try:
                    out[k] = float(v)
                except Exception:
                    out[k] = v
        else:
            out[k] = v
    return out

def _stringify_cats(df: pd.DataFrame, cat_columns):
    for c in cat_columns:
        if c in df.columns:
            df[c] = df[c].astype("string")
    return df

def predict_df(df_raw: pd.DataFrame):
    # Ensure all expected columns are present
    missing = [c for c in X_cols if c not in df_raw.columns]
    for c in missing:
        df_raw[c] = np.nan
    df = df_raw[X_cols].copy()
    df = _stringify_cats(df, cat_cols)
    Xt = preproc.transform(df)
    y_pred = model.predict(Xt)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(Xt)
        proba_df = pd.DataFrame(y_proba, columns=classes) if classes else pd.DataFrame(y_proba)
    else:
        proba_df = None
    return y_pred, proba_df

if submitted:
    # Derive High_Speed from speed_limit if user changed it
    if speed_limit is not None and speed_limit >= 0:
        values["High_Speed"] = int(speed_limit >= 60)

    rec = coerce_input_types(values)
    df_in = pd.DataFrame([rec])
    try:
        with st.spinner("Predicting..."):
            y_pred, proba_df = predict_df(df_in)
        st.success(f"**Predicted Severity:** {y_pred[0]}")
        if proba_df is not None:
            proba_row = proba_df.iloc[0].copy()
            proba_tbl = (proba_row*100).round(2).sort_values(ascending=False).reset_index()
            proba_tbl.columns = ["Class", "Probability (%)"]

            st.write("**Class Probabilities:**")
            # Show as table
            st.dataframe(proba_tbl, use_container_width=True)
            # Show as bar chart
            st.bar_chart(proba_tbl.set_index("Class"))

        with st.expander("🔎 Input record used", expanded=False):
            st.dataframe(df_in)
    except Exception as e:
        import traceback
        err = "".join(traceback.format_exception_only(type(e), e)).strip()
        st.error("Prediction failed. Tip: use dropdown values for categoricals.")
        with st.expander("Error details"):
            st.code(err)

