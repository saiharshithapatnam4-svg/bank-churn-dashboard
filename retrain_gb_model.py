import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib

# 1️⃣ Load dataset
df = pd.read_csv(r"C:\Users\PatnamSaiHarshitha\Desktop\BankChurnPrediction\data\Bank Customer Churn Prediction.csv")


# 2️⃣ Encode categorical columns
df_encoded = pd.get_dummies(df, columns=["country", "gender"], drop_first=True)

# 3️⃣ Features and target
X = df_encoded.drop(["customer_id", "churn"], axis=1, errors="ignore")
y = df_encoded["churn"]

# 4️⃣ Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5️⃣ Train Gradient Boosting model
gb = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
gb.fit(X_train, y_train)

# 6️⃣ Evaluate model
y_pred = gb.predict(X_test)
y_prob = gb.predict_proba(X_test)[:,1]

print("Classification Report:")
print(classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))

# 7️⃣ Save the model (overwrite old one)
joblib.dump(gb, "models/gradient_boosting_churn.pkl")
print("✅ Model saved successfully!")
