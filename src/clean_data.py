import pandas as pd

df = pd.read_csv("data/online_retail.csv", encoding="latin1")

df.dropna(subset=["CustomerID"], inplace=True)
df = df[df["Quantity"] > 0]
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

print(df.shape)
