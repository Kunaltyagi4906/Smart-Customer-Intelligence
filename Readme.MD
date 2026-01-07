# 📊 Smart Customer Intelligence

> Transforming customer data into actionable business insights 🚀

## 🔍 Overview

**Smart Customer Intelligence** is an end-to-end **Data Science & Machine Learning** project designed to analyze customer behavior, segment users, and predict churn using real-world transactional data.

The project demonstrates how raw data can be converted into meaningful insights that help businesses:
- Identify high-value customers  
- Detect customers at risk of leaving  
- Make data-driven retention and marketing decisions  

---

## 🧠 Problem Statement

Many businesses collect large amounts of customer data but struggle to:
- Understand customer value  
- Predict churn in advance  
- Translate analytics into business actions  

This project solves these challenges by building a complete data pipeline — from data ingestion to visualization.

---

## 🏗️ Tech Stack

- **Programming Language:** Python  
- **Database:** MongoDB  
- **Data Analysis:** Pandas, NumPy  
- **Visualization:** Matplotlib, Seaborn  
- **Machine Learning:** Scikit-learn  
- **Dashboard:** Streamlit  
- **Version Control:** Git & GitHub  

---

## 🔄 Project Workflow

1. **Data Collection**
   - Customer and transaction datasets
   - Data stored and managed using MongoDB  

2. **Data Cleaning & Exploratory Data Analysis (EDA)**
   - Handling missing and inconsistent values  
   - Analyzing customer spending patterns  
   - Country-wise and behavior-based insights  

3. **Feature Engineering**
   - Total money spent  
   - Total number of orders  
   - Average order value  
   - Log-transformed features for modeling  

4. **Customer Segmentation**
   - K-Means clustering  
   - Customers categorized into:
     - Premium Customers  
     - Regular Customers  
     - Low-Value Customers  

5. **Churn Prediction**
   - Binary classification model (At Risk / Not at Risk)  
   - Model evaluation using:
     - Confusion Matrix  
     - Precision, Recall, F1-Score  

6. **Interactive Dashboard**
   - Business-friendly KPIs  
   - Filters by customer type and country  
   - Visual insights for non-technical users  

---

## 📈 Key Insights

- Premium customers contribute a major portion of total revenue  
- Customers with low purchase frequency show higher churn risk  
- Clear relationship between spending behavior and churn probability  

---

## 🎯 Business Impact

- Enables early churn detection  
- Helps retain high-value customers  
- Supports targeted marketing strategies  
- Improves revenue forecasting and customer lifetime value analysis  

---


---

## ▶️ How to Run Locally

1. Clone the repository:
```bash
git clone https://github.com/your-username/smart-customer-intelligence.git
cd smart-customer-intelligence
pip install -r requirements.txt
streamlit run src/dashboard.py

👨‍💻 Author

Kunal Tyagi
B.Tech Computer Science
Data Science | Machine Learning | AI Enthusiast

