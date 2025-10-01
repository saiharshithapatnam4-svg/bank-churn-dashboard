import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import joblib
import os

# --- Load dataset ---
DATA_PATH = os.path.join("data", "Bank Customer Churn Prediction.csv")
df = pd.read_csv(DATA_PATH)

# --- Load trained Gradient Boosting model ---
MODEL_PATH = os.path.join("models", "gradient_boosting_churn.pkl")
gb = joblib.load(MODEL_PATH)

# --- Encode categorical features ---
df_encoded = pd.get_dummies(df, columns=["country", "gender"], drop_first=True)
X = df_encoded.drop(["customer_id", "churn"], axis=1, errors="ignore")

# --- Predict churn probabilities ---
if "churn_probability" not in df_encoded.columns:
    df_encoded["churn_probability"] = gb.predict_proba(X)[:, 1]

st.title("📊 Bank Customer Churn Prediction Dashboard")

# --- Automatically find best threshold based on F1-score ---
y_true = df_encoded["churn"] if "churn" in df_encoded.columns else None
y_prob = df_encoded["churn_probability"]

best_f1 = 0
best_thresh_f1 = 0.5
best_thresh_recall = 0.5

if y_true is not None:
    for t in [i / 100 for i in range(30, 71)]:  # thresholds 0.3 to 0.7
        y_pred_thresh = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, y_pred_thresh)
        recall = recall_score(y_true, y_pred_thresh)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh_f1 = t
        # Find threshold where recall >= 0.7
        if recall >= 0.7:
            best_thresh_recall = t

    st.subheader("📌 Suggested Thresholds")
    st.write(f"Best F1-score Threshold: {best_thresh_f1:.2f} (F1-score: {best_f1:.2f})")
    st.write(f"Recall ≥70% Threshold: {best_thresh_recall:.2f} (Recall: {recall_score(y_true, (y_prob >= best_thresh_recall).astype(int)):.2f})")
else:
    best_thresh_f1 = best_thresh_recall = 0.5

# --- Threshold slider ---
st.subheader("Set Churn Probability Threshold")
threshold = st.slider(
    "Select threshold for high-risk customers",
    0.0,
    1.0,
    best_thresh_recall  # default to recall≥70% threshold
)

# --- Filter high-risk customers ---
high_risk_customers = df_encoded[df_encoded["churn_probability"] >= threshold]

# --- Display top high-risk customers ---
st.subheader(f"Top High-Risk Customers (Threshold = {threshold})")
top_risk = high_risk_customers.sort_values(
    "churn_probability", ascending=False
).head(20)
st.dataframe(
    top_risk[["credit_score", "age", "balance", "products_number", "churn_probability"]]
)

# --- Number of high-risk customers ---
st.write(
    f"Number of high-risk customers above threshold: {high_risk_customers.shape[0]}"
)

# --- Metrics at selected threshold ---
if y_true is not None:
    y_pred_thresh = (df_encoded["churn_probability"] >= threshold).astype(int)
    precision = precision_score(y_true, y_pred_thresh)
    recall = recall_score(y_true, y_pred_thresh)
    f1 = f1_score(y_true, y_pred_thresh)
    accuracy = accuracy_score(y_true, y_pred_thresh)

    st.subheader("Model Metrics at Selected Threshold")
    st.write(f"Accuracy: {accuracy:.2f}")
    st.write(f"Precision: {precision:.2f}")
    st.write(f"Recall: {recall:.2f}")
    st.write(f"F1-score: {f1:.2f}")

# --- Churn probability distribution ---
st.subheader("Churn Probability Distribution")
fig1, ax1 = plt.subplots()
sns.histplot(df_encoded["churn_probability"], bins=20, color="skyblue", ax=ax1)
ax1.axvline(threshold, color="red", linestyle="--", label="High-risk threshold")
ax1.legend()
st.pyplot(fig1)

# --- Feature importance ---
st.subheader("Feature Importance - What drives churn")
importances = gb.feature_importances_
features = X.columns
fig2, ax2 = plt.subplots()
ax2.barh(features, importances, color="lightgreen")
st.pyplot(fig2)

# --- Balance vs Churn Probability scatter plot ---
st.subheader("Balance vs Churn Probability")
fig3, ax3 = plt.subplots()
sns.scatterplot(
    x=df_encoded["balance"],
    y=df_encoded["churn_probability"],
    hue=df_encoded["churn_probability"],
    palette="coolwarm",
    size=df_encoded["products_number"],
    sizes=(20, 200),
    ax=ax3,
)
st.pyplot(fig3)

st.success("✅ Dashboard is ready!")
