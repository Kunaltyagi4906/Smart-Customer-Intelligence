import pandas as pd
from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client["customer_intelligence"]

df = pd.read_csv("data/online_retail.csv", encoding="latin1")

df.dropna(subset=["CustomerID"], inplace=True)
df = df[df["Quantity"] > 0]

records = df.to_dict("records")

db.transactions.insert_many(records)

print("Data inserted successfully - Data load complete")

