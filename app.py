import os
import duckdb
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Olist BI — Delivery & Retention", layout="wide")

# ----------------------------
# Sidebar: Config
# ----------------------------
st.sidebar.header("Setup")
data_dir = st.sidebar.text_input("Data folder", value=".", help="Folder containing the Olist CSVs")
min_orders_filter = st.sidebar.number_input("Min orders per category", 50, 10000, 200, 50)
st.sidebar.markdown("---")
st.sidebar.caption("Tip: Put the app in the same folder as the CSVs, or set the full path above.")

# Helper: path join & existence
def pj(ddir, name): 
    return os.path.join(ddir, name)

needed = [
    "olist_orders_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_products_dataset.csv",
    "olist_order_payments_dataset.csv",
    "olist_order_reviews_dataset.csv",
    "olist_customers_dataset.csv",
    "olist_sellers_dataset.csv",
    "olist_geolocation_dataset.csv",
    "product_category_name_translation.csv",
]
missing = [f for f in needed if not os.path.exists(pj(data_dir, f))]
if missing:
    st.error(f"Missing files in `{data_dir}`:\n\n" + "\n".join(missing))
    st.stop()

@st.cache_data(show_spinner=False)
def load_data(ddir):
    orders = pd.read_csv(pj(ddir, "olist_orders_dataset.csv"),
                         parse_dates=[
                             "order_purchase_timestamp",
                             "order_approved_at",
                             "order_delivered_carrier_date",
                             "order_delivered_customer_date",
                             "order_estimated_delivery_date"
                         ])
    items = pd.read_csv(pj(ddir, "olist_order_items_dataset.csv"))
    products = pd.read_csv(pj(ddir, "olist_products_dataset.csv"))
    payments = pd.read_csv(pj(ddir, "olist_order_payments_dataset.csv"))
    reviews = pd.read_csv(pj(ddir, "olist_order_reviews_dataset.csv"),
                          parse_dates=["review_creation_date","review_answer_timestamp"])
    customers = pd.read_csv(pj(ddir, "olist_customers_dataset.csv"))
    sellers = pd.read_csv(pj(ddir, "olist_sellers_dataset.csv"))
    geoloc = pd.read_csv(pj(ddir, "olist_geolocation_dataset.csv"))
    cat_map = pd.read_csv(pj(ddir, "product_category_name_translation.csv"))
    return orders, items, products, payments, reviews, customers, sellers, geoloc, cat_map

orders, items, products, payments, reviews, customers, sellers, geoloc, cat_map = load_data(data_dir)

# Map categories to English
cat_map = cat_map.rename(columns={
    "product_category_name": "category_pt",
    "product_category_name_english": "category_en"
})
products = products.merge(cat_map, how="left", left_on="product_category_name", right_on="category_pt")

# Most expensive item per order for primary category & seller
idx = items.groupby("order_id")["price"].idxmax()
top_item = items.loc[idx, ["order_id", "product_id", "seller_id", "price", "freight_value"]].merge(
    products[["product_id", "category_en", "product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm"]],
    on="product_id", how="left"
)

# First seller per order (by order_item_id) — source of seller_id for orders
first_seller = (
    items.sort_values("order_item_id")
         .drop_duplicates("order_id")[["order_id", "seller_id"]]
)

# Order-level payments
pay = payments.groupby("order_id", as_index=False).agg(
    total_payment=("payment_value", "sum"),
    n_payments=("payment_sequential", "count"),
    pay_types=("payment_type", lambda s: ", ".join(sorted(s.astype(str).unique())))
)

# ZIP → centroid
geo_centroids = geoloc.groupby("geolocation_zip_code_prefix", as_index=False).agg(
    lat=("geolocation_lat", "mean"),
    lon=("geolocation_lng", "mean")
)

cust_geo = customers.merge(
    geo_centroids, left_on="customer_zip_code_prefix",
    right_on="geolocation_zip_code_prefix", how="left"
).rename(columns={"lat": "cust_lat", "lon": "cust_lon"})

sell_geo = sellers.merge(
    geo_centroids, left_on="seller_zip_code_prefix",
    right_on="geolocation_zip_code_prefix", how="left"
).rename(columns={"lat": "sell_lat", "lon": "sell_lon"})

# Merge core order fact (bring seller_id from first_seller / items, NOT from orders)
orders_fact = (
    orders
    .merge(top_item, on="order_id", how="left")  # brings in seller_id from items
    .merge(pay, on="order_id", how="left")
    .merge(customers[["customer_id","customer_unique_id","customer_city","customer_state","customer_zip_code_prefix"]],
           on="customer_id", how="left")
    .merge(sellers[["seller_id","seller_zip_code_prefix"]], on="seller_id", how="left")
    .merge(cust_geo[["customer_id","cust_lat","cust_lon"]], on="customer_id", how="left")
    .merge(sell_geo[["seller_id","sell_lat","sell_lon"]], on="seller_id", how="left")
    .merge(reviews[["order_id","review_score"]], on="order_id", how="left")
)

# KPIs & engineered features
orders_fact["delivered_flag"] = orders_fact["order_delivered_customer_date"].notna()
orders_fact["late_flag"] = (
    orders_fact["delivered_flag"] &
    (orders_fact["order_delivered_customer_date"] > orders_fact["order_estimated_delivery_date"])
)
orders_fact["lateness_days"] = (
    (orders_fact["order_delivered_customer_date"] - orders_fact["order_estimated_delivery_date"])
    .dt.total_seconds() / 86400.0
)
orders_fact["late_days_only"] = orders_fact["lateness_days"].clip(lower=0)
orders_fact["lead_time_days"] = (
    (orders_fact["order_delivered_customer_date"] - orders_fact["order_purchase_timestamp"])
    .dt.total_seconds() / 86400.0
)
orders_fact["purchase_month"] = orders_fact["order_purchase_timestamp"].dt.to_period("M").astype(str)
orders_fact["purchase_quarter"] = orders_fact["order_purchase_timestamp"].dt.to_period("Q").astype(str)

# Haversine distance (km) customer ↔ seller
def haversine(lon1, lat1, lon2, lat2, R=6371.0):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))

orders_fact["distance_km"] = haversine(
    orders_fact["sell_lon"], orders_fact["sell_lat"],
    orders_fact["cust_lon"], orders_fact["cust_lat"]
)

# ----------------------------
# Global filters
# ----------------------------
st.sidebar.header("Filters")
date_min = orders_fact["order_purchase_timestamp"].min()
date_max = orders_fact["order_purchase_timestamp"].max()
start, end = st.sidebar.date_input("Purchase date range", value=[date_min.date(), date_max.date()])
state_sel = st.sidebar.multiselect("States", sorted(orders_fact["customer_state"].dropna().unique()))
cats_sel = st.sidebar.multiselect("Categories", sorted(orders_fact["category_en"].dropna().unique()))

mask = (
    (orders_fact["order_purchase_timestamp"] >= pd.to_datetime(start)) &
    (orders_fact["order_purchase_timestamp"] <= pd.to_datetime(end) + pd.Timedelta(days=1))
)
if state_sel:
    mask &= orders_fact["customer_state"].isin(state_sel)
if cats_sel:
    mask &= orders_fact["category_en"].isin(cats_sel)

df = orders_fact.loc[mask].copy()

# ----------------------------
# Header KPIs
# ----------------------------
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.metric("Orders", f"{df['order_id'].nunique():,}")
with kpi2:
    st.metric("Late rate", f"{100 * df['late_flag'].mean():.1f}%")
with kpi3:
    st.metric("Avg lead time (d)", f"{df['lead_time_days'].mean():.1f}")
with kpi4:
    st.metric("Avg distance (km)", f"{df['distance_km'].mean():.1f}")

# ----------------------------
# Tabs
# ----------------------------
tab_overview, tab_delivery, tab_retention, tab_sql = st.tabs([
    "Overview", "Delivery Performance", "Retention (RFM)", "SQL Lab"
])

with tab_overview:
    col1, col2 = st.columns(2)

    # Orders by month
    by_month = df.groupby("purchase_month", as_index=False).agg(
        orders=("order_id", "nunique"),
        late_rate=("late_flag", "mean")
    )
    fig1 = px.line(by_month, x="purchase_month", y="orders", title="Orders over time (month)")
    fig1.update_xaxes(type="category")
    col1.plotly_chart(fig1, use_container_width=True)

    # Late rate by month
    fig2 = px.line(by_month, x="purchase_month", y="late_rate", title="Late rate over time (month)")
    fig2.update_traces(mode="lines+markers")
    fig2.update_xaxes(type="category")
    fig2.update_yaxes(tickformat=".1%")
    col2.plotly_chart(fig2, use_container_width=True)

    # Top categories by late orders
    top_cat = (
        df.groupby("category_en", as_index=False)
          .agg(n_orders=("order_id", "count"),
               late_orders=("late_flag", "sum"),
               late_rate=("late_flag", "mean"))
          .query("n_orders >= @min_orders_filter")
          .sort_values("late_orders", ascending=False)
          .head(15)
    )
    fig3 = px.bar(top_cat, x="late_orders", y="category_en", orientation="h",
                  title="Top categories by late orders")
    fig3.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig3, use_container_width=True)

with tab_delivery:
    col1, col2 = st.columns(2)

    # Lead time distribution by status
    fig4 = px.box(df, x="late_flag", y="lead_time_days", points=False,
                  title="Lead time by status (0=On-time, 1=Late)")
    col1.plotly_chart(fig4, use_container_width=True)

    # Freight by status
    fig5 = px.box(df, x="late_flag", y="freight_value", points=False,
                  title="Freight value by status")
    col2.plotly_chart(fig5, use_container_width=True)

    # Late rate vs avg distance
    cat_summ = (
        df.groupby("category_en", as_index=False)
          .agg(n_orders=("order_id", "count"),
               late_rate=("late_flag", "mean"),
               avg_distance_km=("distance_km", "mean"),
               avg_lead_time=("lead_time_days", "mean"))
          .query("n_orders >= @min_orders_filter")
          .sort_values("late_rate", ascending=False)
    )
    fig6 = px.scatter(cat_summ, x="avg_distance_km", y="late_rate",
                      size="n_orders", color="category_en",
                      title="Late rate vs avg distance (bubble size = volume)",
                      hover_data=["avg_lead_time"])
    fig6.update_yaxes(tickformat=".1%")
    st.plotly_chart(fig6, use_container_width=True)

with tab_retention:
    st.caption("RFM on delivered orders; LTV here = historical revenue (sum of payments).")
    # Monetary per order from payments
    pay_ord = payments.groupby("order_id", as_index=False)["payment_value"].sum() \
                      .rename(columns={"payment_value": "order_revenue"})
    orders_paid = orders.merge(pay_ord, on="order_id", how="left")
    orders_paid["order_purchase_timestamp"] = pd.to_datetime(orders_paid["order_purchase_timestamp"])
    snap = orders_paid["order_purchase_timestamp"].max()

    rfm = (
        orders_paid.groupby("customer_id", as_index=False)
                   .agg(frequency=("order_id", "nunique"),
                        last_purchase=("order_purchase_timestamp", "max"),
                        monetary=("order_revenue", "sum"))
    )
    rfm["recency"] = (snap - rfm["last_purchase"]).dt.days
    rfm = rfm.replace([np.inf, -np.inf], np.nan).dropna(subset=["recency"]).fillna({"monetary": 0})

    # Scale + KMeans (quick)
    Xr = rfm[["recency", "frequency", "monetary"]].copy()
    Xr["monetary"] = np.log1p(Xr["monetary"])
    Xs = StandardScaler().fit_transform(Xr)
    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    rfm["cluster"] = km.fit_predict(Xs)

    cl = (
        rfm.groupby("cluster", as_index=False)
           .agg(n_customers=("customer_id", "nunique"),
                mean_recency=("recency", "mean"),
                mean_frequency=("frequency", "mean"),
                mean_monetary=("monetary", "mean"),
                avg_hist_LTV=("monetary", "mean"),
                total_hist_LTV=("monetary", "sum"))
           .sort_values("avg_hist_LTV", ascending=False)
    )

    c1, c2 = st.columns(2)
    c1.dataframe(cl, use_container_width=True)
    fig7 = px.bar(cl, x="cluster", y="avg_hist_LTV", title="Average historical LTV by RFM cluster")
    c2.plotly_chart(fig7, use_container_width=True)

with tab_sql:
    st.caption("Run SQL on the filtered *orders_fact* using DuckDB (read-only). Example: `SELECT customer_state, COUNT(*) FROM df GROUP BY 1 ORDER BY 2 DESC`")
    query = st.text_area(
        "SQL",
        value="SELECT purchase_quarter, COUNT(*) AS orders, "
              "AVG(CASE WHEN late_flag THEN 1 ELSE 0 END) AS late_rate "
              "FROM df GROUP BY 1 ORDER BY 1"
    )
    btn = st.button("Run")
    if btn and query.strip():
        try:
            # DuckDB can query Python DataFrames by variable name
            res = duckdb.query(query).to_df()
            st.dataframe(res, use_container_width=True)
            if {"purchase_quarter", "orders"}.issubset(res.columns):
                fig = px.line(res, x="purchase_quarter", y="orders", title="Orders by quarter (SQL result)")
                fig.update_xaxes(type="category")
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(str(e))
