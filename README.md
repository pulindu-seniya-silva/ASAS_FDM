# 🚦 Accident Severity Predictor

A Streamlit web application that predicts **road accident severity** (Slight / Serious / Fatal) using a trained Random Forest model.  
The app loads the trained `model.pkl` and provides a user interface to input accident conditions and view predictions with probability charts.

---

## ⚡ Quick Start


clone the repository
pip install -r requirements.txt
streamlit run app.py

## 🖥️ Features

Dropdowns for accident conditions (day, time, road type, weather, etc.).

Optional Speed_limit field → automatically toggles High_Speed (>=60 → 1).

Prediction output:

Predicted severity class

Probability table

Probability bar chart

Optional probability donut chart

## 📂 Project Structure


├── app.py                  # Streamlit UI
├── artifacts/
│   ├── model.pkl           # Trained Random Forest bundle
│   ├── choices.json        # (Optional) Dropdown options from dataset
├── requirements.txt        # Python dependencies


## 🛠️ Requirements

Python 3.12

NumPy, pandas, SciPy, scikit-learn, joblib, imbalanced-learn

Streamlit

All versions are pinned in requirements.txt.

## 🖥️ Usage

Select input values (day, time, road type, weather, etc.) from dropdowns.

Optionally provide a Speed_limit to auto-toggle High_Speed (>=60 → 1).

## Click ⚡ Predict to see:

Predicted severity

Probability table

Probability bar chart

(Optional) Probability donut chart

## 📌 Notes

Make sure your artifacts/model.pkl matches the environment specified in requirements.txt.

If you want to regenerate dropdown choices from your dataset, use the make_choices_from_csv.py helper.
