# =========================
# Data Handling
# =========================
import pandas as pd
import numpy as np

# =========================
# Load Dataset
# =========================
df = pd.read_csv(r"D:\CODEC\Customer Churn Prediction.csv")

df.head()
df.info()
df.describe()
print(df.shape)

# =========================
# Exploratory Data Analysis (EDA)
# =========================
import matplotlib.pyplot as plt
import seaborn as sns

sns.countplot(x="Churn", data=df)
plt.title("Churn Distribution")
plt.show()

sns.countplot(x="Contract", hue="Churn", data=df)
plt.title("Churn by Contract Type")
plt.show()

sns.boxplot(x="Churn", y="tenure", data=df)
plt.title("Tenure vs Churn")
plt.show()

sns.boxplot(x="Churn", y="MonthlyCharges", data=df)
plt.title("Monthly Charges vs Churn")
plt.show()

# =========================
# Data Cleaning & Preprocessing
# =========================
#df.replace({"Yes": 1, "No": 0}, inplace=True)
#df = df.infer_objects(copy=False)


df.drop("customerID", axis=1, inplace=True)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)

df.replace({"Yes": 1, "No": 0}, inplace=True)
df = pd.get_dummies(df, drop_first=True)
print(df.shape)
# =========================
# Feature Selection
# =========================
X = df.drop("Churn", axis=1)
y = df["Churn"]

# =========================
# Train-Test Split
# =========================
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# Standard Scaler
# =========================
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================
# Logistic Regression Model
# =========================
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_scaled, y_train)

lr_pred = lr.predict(X_test_scaled)
lr_prob = lr.predict_proba(X_test_scaled)[:, 1]

# =========================
# Random Forest Model
# =========================
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)
rf_prob = rf.predict_proba(X_test)[:, 1]

# =========================
# XGBoost Model
# =========================
from xgboost import XGBClassifier

xgb = XGBClassifier(eval_metric="logloss")
xgb.fit(X_train, y_train)

xgb_pred = xgb.predict(X_test)
xgb_prob = xgb.predict_proba(X_test)[:, 1]

# =========================
# Model Evaluation
# =========================
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score

print("LOGISTIC REGRESSION")
print("Accuracy:", accuracy_score(y_test, lr_pred))
print("Recall:", recall_score(y_test, lr_pred))
print("ROC-AUC:", roc_auc_score(y_test, lr_prob))

print("\nRANDOM FOREST")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print("Recall:", recall_score(y_test, rf_pred))
print("ROC-AUC:", roc_auc_score(y_test, rf_prob))

print("\nXGBOOST")
print("Accuracy:", accuracy_score(y_test, xgb_pred))
print("Recall:", recall_score(y_test, xgb_pred))
print("ROC-AUC:", roc_auc_score(y_test, xgb_prob))
