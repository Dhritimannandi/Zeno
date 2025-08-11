# Brazilian E-Commerce Analysis Dashboard

## 📌 Project Overview
This project analyzes the **Olist Brazilian E-Commerce dataset** to uncover trends in sales, delivery performance, customer retention, and other business KPIs.  
It includes:
- A **Streamlit dashboard** for interactive exploration.
- A **Jupyter notebook** (`brazilian_ecommerce.ipynb`) for in-depth data analysis and modeling.

The dataset comes from a Brazilian marketplace and contains orders from 2016 to 2018.

---

## 🚀 Live Dashboard
You can explore the live dashboard here:  
🔗 **[Streamlit App](https://fskpbl6fj5cvvumt9sqmuq.streamlit.app/)**

---

## 📂 Repository Structure
Zeno/
│-- app.py # Streamlit dashboard application
│-- brazilian_ecommerce.ipynb # Jupyter notebook with full analysis & ML model
│-- README.md # Project documentation
│-- olist_customers_dataset.csv
│-- olist_geolocation_dataset.csv
│-- olist_order_items_dataset.csv
│-- olist_order_payments_dataset.csv
│-- olist_orders_dataset.csv
│-- olist_order_reviews_dataset.csv
│-- olist_products_dataset.csv
│-- olist_sellers_dataset.csv
│-- product_category_name_translation.csv

yaml
Copy
Edit

---

## 📊 Features in the Dashboard
1. **Overview Tab**
   - Orders and late delivery trends over time.
   - Top categories by late orders.
   
2. **Delivery Performance Tab**
   - Lead time and freight cost analysis.
   - Late rate vs. average delivery distance.

3. **Retention (RFM) Tab**
   - Customer segmentation based on **Recency**, **Frequency**, and **Monetary value**.
   - Estimated historical Lifetime Value (LTV) per segment.

4. **SQL Lab**
   - Run SQL queries directly on the filtered dataset inside the dashboard.

---

## 🧠 Machine Learning
In the notebook, we:
- Prepared features from order, product, customer, and seller datasets.
- Built a binary classification model to predict **high vs. low review scores**.
- Compared multiple models (Random Forest, Logistic Regression, Extra Trees).
- Evaluated using **accuracy, precision, recall, f1-score, ROC AUC**.

---

## ⚙️ Running Locally

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
(or manually install streamlit, pandas, numpy, scikit-learn, plotly, duckdb)

2️⃣ Run the Streamlit Dashboard
bash
Copy
Edit
streamlit run app.py
3️⃣ Open in Browser
Streamlit will start a local server, usually at:

arduino
Copy
Edit
http://localhost:8501
📊 Dataset Source
The dataset is publicly available on Kaggle:
Olist Brazilian E-Commerce Public Dataset

✨ Author
Dhritiman Nandi
