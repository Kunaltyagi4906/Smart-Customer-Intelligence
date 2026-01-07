import pandas as pd
from pymongo import MongoClient
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# MongoDB connection
client = MongoClient("mongodb://localhost:27017/")
db = client["customer_intelligence"]

transactions = pd.DataFrame(list(db.transactions.find()))
users = pd.DataFrame(list(db.users.find()))

# Drop Mongo _id
transactions.drop(columns=["_id"], inplace=True)
users.drop(columns=["_id"], inplace=True)

# -----------------------------
# 1️⃣ Feature Engineering
# -----------------------------
transactions["InvoiceDate"] = pd.to_datetime(
    transactions["InvoiceDate"], errors="coerce", dayfirst=True
)

last_date = transactions["InvoiceDate"].max()

last_purchase = (
    transactions.groupby("CustomerID")["InvoiceDate"]
    .max()
    .reset_index()
)

last_purchase["days_since_last_purchase"] = (
    last_date - last_purchase["InvoiceDate"]
).dt.days

# Churn definition
last_purchase["churn"] = last_purchase["days_since_last_purchase"].apply(
    lambda x: 1 if x > 90 else 0
)
print("Users Columns:", users.columns)
print("Last Purchase Columns:", last_purchase.columns)


# Merge with users
data = users.merge(
    last_purchase[["CustomerID", "churn"]],
    left_on="user_id",    # users dataframe ka column
    right_on="CustomerID" # last_purchase ka column
)

# Optional: remove redundant CustomerID column after merge
data.drop(columns=["CustomerID"], inplace=True)

# -----------------------------
# 2️⃣ Model Data
# -----------------------------
X = data[["total_spent", "total_orders", "avg_order_value"]]
y = data["churn"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# -----------------------------
# 3️⃣ Train Model
# -----------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

# -----------------------------
# 4️⃣ Evaluation
# -----------------------------
y_pred = model.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
