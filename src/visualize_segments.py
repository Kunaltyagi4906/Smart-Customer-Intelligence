import pandas as pd
from pymongo import MongoClient
import matplotlib.pyplot as plt
import seaborn as sns

# MongoDB connection
client = MongoClient("mongodb://localhost:27017/")
db = client["customer_intelligence"]
users = pd.DataFrame(list(db.users.find()))

# Clean
users = users.drop(columns=["_id"])

# ------------------------------
# 1️⃣ Count of users per segment
# ------------------------------
plt.figure()
sns.countplot(x="segment", data=users)
plt.title("Customer Count per Segment")
plt.xlabel("Segment")
plt.ylabel("Number of Customers")
plt.show()

# ------------------------------
# 2️⃣ Total spent per segment
# ------------------------------
plt.figure()
sns.boxplot(x="segment", y="total_spent", data=users)
plt.title("Spending Distribution per Segment")
plt.xlabel("Segment")
plt.ylabel("Total Spent")
plt.show()

# ------------------------------
# 3️⃣ Orders per segment
# ------------------------------
plt.figure()
sns.boxplot(x="segment", y="total_orders", data=users)
plt.title("Order Frequency per Segment")
plt.xlabel("Segment")
plt.ylabel("Total Orders")
plt.show()

# ------------------------------
# 4️⃣ Avg Order Value
# ------------------------------
plt.figure()
sns.boxplot(x="segment", y="avg_order_value", data=users)
plt.title("Average Order Value per Segment")
plt.xlabel("Segment")
plt.ylabel("Avg Order Value")
plt.show()
