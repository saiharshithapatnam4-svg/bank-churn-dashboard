# run_pipeline.py
import os
import pandas as pd
import joblib
import requests
from datetime import datetime

# Paths (relative to repo root)
DATA_IN = "data/Bank Customer Churn Prediction.csv"
PRED_OUT = "data/predictions.csv"
TOP20_OUT = "reports/top_20_high_risk_customers.csv"
MODEL_PATH = "models/gradient_boosting_churn.pkl"

def main():
    os.makedirs("data", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    # 1) Load data
    df = pd.read_csv(DATA_IN)

    # 2) Preprocess (must match training preprocessing)
    df_encoded = pd.get_dummies(df, columns=["country", "gender"], drop_first=True)
    X = df_encoded.drop(["customer_id", "churn"], axis=1, errors="ignore")

    # 3) Load model
    model = joblib.load(MODEL_PATH)

    # 4) Predict probabilities
    df["churn_probability"] = model.predict_proba(X)[:, 1]

    # 5) Save predictions CSV (Streamlit app will read this)
    df.to_csv(PRED_OUT, index=False)

    # 6) Save top-20
    top20 = df.sort_values("churn_probability", ascending=False).head(20)
    top20.to_csv(TOP20_OUT, index=False)
    print("Saved top-20 to", TOP20_OUT)

    # 7) Optional: POST top20 to CRM if CRM_URL provided in env
    CRM_URL = os.environ.get("CRM_URL")
    CRM_API_KEY = os.environ.get("CRM_API_KEY")
    if CRM_URL:
        payload = {"generated_at": datetime.utcnow().isoformat(), "top20": top20.to_dict(orient="records")}
        headers = {"Content-Type": "application/json"}
        if CRM_API_KEY:
            headers["Authorization"] = f"Bearer {CRM_API_KEY}"
        try:
            r = requests.post(CRM_URL, json=payload, headers=headers, timeout=30)
            print("CRM POST status:", r.status_code)
        except Exception as e:
            print("CRM POST failed:", e)

if __name__ == "__main__":
    main()
