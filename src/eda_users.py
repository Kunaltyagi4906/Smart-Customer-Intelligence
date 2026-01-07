from pymongo import MongoClient
import pandas as pd
import matplotlib.pyplot as plt

# 1. MongoDB Connection
client = MongoClient("mongodb://localhost:27017/")
db = client["customer_intelligence"]

# 2. Load users collection
users = pd.DataFrame(list(db.users.find({}, {"_id": 0})))

print("Users Data (First 5 rows)")
print(users.head())

print("\nUsers Data Summary")
print(users.describe())

# 3. Check missing values
print("\nMissing Values")
print(users.isnull().sum())

# 4. Distribution of Total Spent
plt.figure()
plt.hist(users["total_spent"], bins=50)
plt.xlabel("Total Spent")
plt.ylabel("Number of Users")
plt.title("Customer Spending Distribution")
plt.show()

# 5. Top 10 High-Value Customers
top_users = users.sort_values("total_spent", ascending=False).head(10)

print("\nTop 10 High-Value Customers")
print(top_users[["user_id", "total_spent", "total_orders"]])

# 6. Orders vs Spending
plt.figure()
plt.scatter(users["total_orders"], users["total_spent"])
plt.xlabel("Total Orders")
plt.ylabel("Total Spent")
plt.title("Orders vs Spending")
plt.show()

# 7. Average Spend per Order
users["avg_order_value"] = users["total_spent"] / users["total_orders"]

plt.figure()
plt.hist(users["avg_order_value"], bins=40)
plt.xlabel("Average Order Value")
plt.ylabel("Number of Users")
plt.title("Average Order Value Distribution")
plt.show()

print("\nEDA Completed Successfully")
