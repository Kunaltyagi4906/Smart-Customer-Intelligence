import streamlit as st
import pandas as pd
from pymongo import MongoClient
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np

# -----------------------------
# 1️⃣ MongoDB Connection
# -----------------------------
client = MongoClient("mongodb://localhost:27017/")
db = client["customer_intelligence"]

users = pd.DataFrame(list(db.users.find()))
transactions = pd.DataFrame(list(db.transactions.find()))

# Drop _id
users = users.drop(columns=["_id"], errors="ignore")
transactions = transactions.drop(columns=["_id"], errors="ignore")

# -----------------------------
# 2️⃣ Feature Engineering
# -----------------------------
users["Average Order Value"] = users["total_spent"] / users["total_orders"]
users.rename(columns={"segment": "Customer Type",
                      "user_id": "Customer ID",
                      "total_spent": "Total Money Spent",
                      "total_orders": "Total Orders"}, inplace=True)

transactions["InvoiceDate"] = pd.to_datetime(transactions["InvoiceDate"], errors="coerce", dayfirst=True)
last_date = transactions["InvoiceDate"].max()
last_purchase = transactions.groupby("CustomerID")["InvoiceDate"].max().reset_index()
last_purchase["Days Since Last Purchase"] = (last_date - last_purchase["InvoiceDate"]).dt.days
last_purchase["At Risk of Leaving?"] = last_purchase["Days Since Last Purchase"].apply(lambda x: 1 if x > 90 else 0)

# Merge churn
data = users.merge(last_purchase[["CustomerID", "At Risk of Leaving?"]],
                   left_on="Customer ID", right_on="CustomerID", how="left")
data.drop(columns=["CustomerID"], inplace=True)
data["At Risk of Leaving?"] = data["At Risk of Leaving?"].fillna(0)

# -----------------------------
# 3️⃣ ML Model for Churn Prediction
# -----------------------------
X = data[["Total Money Spent", "Total Orders", "Average Order Value"]]
y = data["At Risk of Leaving?"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = LogisticRegression()
model.fit(X_scaled, y)

# -----------------------------
# 4️⃣ Streamlit SaaS Dashboard
# -----------------------------
st.set_page_config(page_title="Smart Customer Intelligence", layout="wide")
st.title("🚀 Smart Customer Intelligence Dashboard (SaaS Style)")

# Sidebar Filters
st.sidebar.header("🔹 Filter Customers")
segment_filter = st.sidebar.multiselect("Customer Type:", options=data["Customer Type"].unique(), default=data["Customer Type"].unique())
country_filter = st.sidebar.multiselect("Customer Country:", options=data["country"].unique(), default=data["country"].unique())

filtered_data = data[(data["Customer Type"].isin(segment_filter)) & (data["country"].isin(country_filter))]

# -----------------------------
# 5️⃣ Color-coded Cards per Customer Type
# -----------------------------
st.subheader("🌟 Customer Insights by Type")

customer_types = filtered_data["Customer Type"].unique()
# Assign default color if customer type not in dict
colors = {"1": "#FF6B6B", "2": "#4ECDC4", "3": "#FFD93D"}  # Red, Teal, Yellow
default_color = "#A9A9A9"  # Grey for unknown types

cols = st.columns(len(customer_types))
for idx, ctype in enumerate(customer_types):
    subset = filtered_data[filtered_data["Customer Type"]==ctype]
    total_customers = subset.shape[0]
    total_spent = subset["Total Money Spent"].sum()
    churn_count = subset["At Risk of Leaving?"].sum()
    
    # Use default color if type not in colors
    color = colors.get(str(ctype), default_color)
    
    # Mini sparkline (Total Spent Trend)
    sparkline = subset["Total Money Spent"].rolling(5, min_periods=1).mean().values
    fig, ax = plt.subplots(figsize=(2.5,0.8))
    ax.plot(sparkline, color="white")
    ax.set_facecolor(color)
    ax.axis("off")
    
    with cols[idx]:
        st.markdown(f"<div style='background-color:{color};padding:15px;border-radius:10px;color:white;text-align:center;'>"
                    f"<h3>Type {ctype}</h3>"
                    f"<p><b>Total Customers:</b> {total_customers}</p>"
                    f"<p><b>Total Spent:</b> ${int(total_spent)}</p>"
                    f"<p><b>At Risk:</b> {churn_count}</p></div>", unsafe_allow_html=True)
        st.pyplot(fig)
        plt.clf()

# -----------------------------
# 6️⃣ Top Premium Customers Table
# -----------------------------
st.subheader("🏆 Top Premium Customers")
vip_customers = filtered_data[(filtered_data["Customer Type"]==2) & (filtered_data["At Risk of Leaving?"]==0)]
vip_customers = vip_customers.sort_values(by="Total Money Spent", ascending=False).head(10)
st.dataframe(vip_customers[["Customer ID","country","Total Money Spent","Total Orders","Average Order Value","At Risk of Leaving?"]])

# -----------------------------
# 7️⃣ Churn Prediction for Any Customer
# -----------------------------
st.subheader("🔮 Check if a Customer is at Risk")
customer_id_input = st.text_input("Enter Customer ID:")
if customer_id_input:
    try:
        customer_row = filtered_data[filtered_data["Customer ID"]==int(customer_id_input)]
        if customer_row.empty:
            st.warning("Customer not found.")
        else:
            X_user = scaler.transform(customer_row[["Total Money Spent","Total Orders","Average Order Value"]])
            pred = model.predict(X_user)[0]
            prob = model.predict_proba(X_user)[0][1]
            st.write(f"**Churn Prediction:** {'Yes' if pred==1 else 'No'}")
            st.write(f"**Churn Probability:** {prob:.2f}")
    except:
        st.error("Enter a valid numeric Customer ID")
