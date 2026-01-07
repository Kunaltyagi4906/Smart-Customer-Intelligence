from pymongo import MongoClient
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 1. MongoDB connection
client = MongoClient("mongodb://localhost:27017/")
db = client["customer_intelligence"]

# 2. Load users data
users = pd.DataFrame(list(db.users.find({}, {"_id": 0})))

# 3. Feature engineering
import numpy as np

users["log_total_spent"] = np.log1p(users["total_spent"])
users["log_total_orders"] = np.log1p(users["total_orders"])
users["log_avg_order_value"] = np.log1p(users["avg_order_value"])

features = users[
    ["log_total_spent", "log_total_orders", "log_avg_order_value"]
]


# 4. Scaling (VERY IMPORTANT for KMeans)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# 5. Elbow Method to find optimal K
inertia = []

for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(scaled_features)
    inertia.append(km.inertia_)

plt.figure()
plt.plot(range(1, 11), inertia)
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal K")
plt.show()

# 6. Train final KMeans (K = 3 is usually best)
kmeans = KMeans(n_clusters=3, random_state=42)
users["segment"] = kmeans.fit_predict(scaled_features)

# 7. Save segments back to MongoDB
db.users.delete_many({})
db.users.insert_many(users.to_dict("records"))

print("Customer segmentation completed successfully")

# 8. Segment analysis
segment_summary = users.groupby("segment").agg(
    avg_spent=("total_spent", "mean"),
    avg_orders=("total_orders", "mean"),
    customer_count=("user_id", "count")
)

print("\nSegment Summary")
print(segment_summary)
