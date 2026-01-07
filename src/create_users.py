import pandas as pd
from pymongo import MongoClient

# Connect MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["customer_intelligence"]

# Load transactions from MongoDB
transactions = pd.DataFrame(
    list(db.transactions.find({}, {"_id": 0}))
)

# Create total spend column
transactions["total_amount"] = (
    transactions["Quantity"] * transactions["UnitPrice"]
)

# Group by CustomerID
users = transactions.groupby("CustomerID").agg(
    first_purchase=("InvoiceDate", "min"),
    country=("Country", "first"),
    total_spent=("total_amount", "sum"),
    total_orders=("InvoiceNo", "nunique")
).reset_index()

# Rename column
users.rename(columns={"CustomerID": "user_id"}, inplace=True)

# Insert into MongoDB
db.users.delete_many({})
db.users.insert_many(users.to_dict("records"))

print("Users collection created successfully")
