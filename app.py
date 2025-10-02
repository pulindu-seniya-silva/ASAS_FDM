import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(page_title="Accident Severity Predictor", layout="wide")

st.title("🚦 Accident Severity Predictor")
st.caption("This app uses an internal trained model bundle located at **artifacts/model.pkl**. Users cannot upload models.")

# ---------- Load internal model bundle ----------
DEFAULT_BUNDLE_PATH = os.path.join("artifacts", "model.pkl")

if not os.path.exists(DEFAULT_BUNDLE_PATH):
    st.error("Model bundle not found at artifacts/model.pkl. Please place the saved model there.")
    st.stop()

try:
    bundle = joblib.load(DEFAULT_BUNDLE_PATH)
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

# ---------- Patch: normalize OHE categories to string to avoid np.isnan on non-numeric ----------
def _patch_ohe_categories_to_str(preproc):
    """
    Force OneHotEncoder categories_ arrays to string dtype so old sklearn paths
    that call np.isnan(categories_) won't crash on object/int types.
    """
    try:
        cat_trf = preproc.named_transformers_.get("cat", None)
        if cat_trf is None:
            return
        ohe = getattr(cat_trf, "named_steps", {}).get("onehot", None)
        if ohe is None and hasattr(cat_trf, "categories_"):
            ohe = cat_trf  # already an OHE

        if ohe is not None and hasattr(ohe, "categories_"):
            new_cats = []
            for cats in ohe.categories_:
                # Convert every category to plain Python str (keep NaN as-is)
                cats_list = []
                for c in list(cats):
                    if isinstance(c, str):
                        cats_list.append(c)
                    elif pd.isna(c):
                        cats_list.append(np.nan)
                    else:
                        cats_list.append(str(c))
                new_cats.append(np.array(cats_list, dtype=object))
            ohe.categories_ = new_cats
    except Exception:
        pass

_patch_ohe_categories_to_str(preproc)

with st.expander("ℹ️ Model Info", expanded=False):
    st.write("**Classes:**", classes if classes else "(not provided)")
    st.write("**Features (X_cols):**")
    st.code("\n".join(map(str, X_cols)))
    try:
        st.json(metadata if isinstance(metadata, dict) else {"metadata": str(metadata)})
    except Exception:
        st.write(metadata)

# ---------- Try to retrieve categorical choices from the encoder (if available) ----------
cat_choices = {}
try:
    cat_transformer = preproc.named_transformers_.get('cat')
    if hasattr(cat_transformer, 'named_steps'):
        ohe = cat_transformer.named_steps.get('onehot')
    else:
        ohe = cat_transformer  # in case it's directly an OHE

    if hasattr(ohe, 'categories_'):
        for col_name, cats in zip(cat_cols, ohe.categories_):
            try:
                # Keep only non-NaN and stringify
                choices = [str(c) for c in cats if not (isinstance(c, float) and np.isnan(c)) and str(c) != 'nan']
            except Exception:
                choices = []
            cat_choices[col_name] = choices
except Exception:
    pass  # fallback to free text

st.subheader("🧍 Single Prediction")
st.caption("Provide input values. For categorical fields, choose from the dropdown if available; otherwise type a value.")

# Build a form so we only predict when the user clicks the button
with st.form("single_pred_form"):
    cols = st.columns(3)
    values = {}
    for i, col_name in enumerate(X_cols):
        with cols[i % 3]:
            if col_name in cat_cols:
                if col_name in cat_choices and len(cat_choices[col_name]) > 0:
                    options = ["(leave blank)"] + cat_choices[col_name]
                    choice = st.selectbox(col_name, options, index=0)
                    values[col_name] = ("" if choice == "(leave blank)" else choice)
                else:
                    values[col_name] = st.text_input(col_name, value="")
            else:
                # numeric fields
                if col_name in {"Month"}:
                    values[col_name] = st.number_input(col_name, min_value=1, max_value=12, value=6, step=1)
                elif col_name in {"High_Speed", "Hazard_Flag"}:
                    values[col_name] = st.number_input(col_name, min_value=0, max_value=1, value=0, step=1)
                else:
                    values[col_name] = st.number_input(col_name, value=0)
    submitted = st.form_submit_button("⚡ Predict")

# Utilities
def coerce_input_types(record: dict):
    """Best-effort type coercion for known integer flags/fields and numerics."""
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
    """Ensure categorical columns are string dtype (preserve NaN)."""
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
    # Make sure categoricals are strings to avoid np.isnan paths
    df = _stringify_cats(df, cat_cols)
    # Transform and predict
    Xt = preproc.transform(df)
    y_pred = model.predict(Xt)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(Xt)
        proba_df = pd.DataFrame(y_proba, columns=classes) if classes else pd.DataFrame(y_proba)
    else:
        proba_df = None
    return y_pred, proba_df

if submitted:
    rec = coerce_input_types(values)
    df_in = pd.DataFrame([rec])
    try:
        y_pred, proba_df = predict_df(df_in)
        st.success(f"**Predicted Severity:** {y_pred[0]}")
        if proba_df is not None:
            st.write("**Class Probabilities:**")
            st.bar_chart(proba_df.T)
        with st.expander("🔎 Input record used", expanded=False):
            st.dataframe(df_in)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
