# Warehouse Clustering and Forecasting

This project identifies how different types of products move through a warehouse and predicts their future volume using two forecasting models: **Prophet** and **SARIMAX**.  

The goal is to support warehouse teams with better storage planning, labour allocation, and inventory flow management.

---

## 📦 Project Overview

Warehouses often store thousands of product types, each behaving differently in terms of volume, weight, value, and picking frequency.  
This project answers two key questions:
1. Can we group products based on how they behave in the warehouse?
2. Can we forecast how much stock will move in the future for each group?

The answer to both is yes — using a combination of **clustering** and **time series forecasting**.

---

## 🧪 Methods Used

### 1. Clustering (Product Grouping)
Using product features like:
- Average volume
- Weight
- Unit price
- Pick frequency
- Storage cost per unit

Products were grouped into three clusters:
- **Cluster 0 – Low-Value Electronics**: Low value, slow-moving, low handling cost
- **Cluster 1 – High-Volume Essentials**: Fast-moving items with high storage efficiency
- **Cluster 2 – High-Value Chemicals**: Expensive, heavier items with steady demand

### 2. Forecasting (Volume Prediction)
Time series forecasting models were trained on monthly inbound volumes for each cluster using:
- **Prophet** (from Meta)
- **SARIMAX** (from statsmodels)

Both models were tuned and tested using real 2024 data.

---

## 📊 Key Results

- **Clustering** created meaningful operational categories with actionable storage advice.
- **Forecasting Accuracy**:
  - SARIMAX consistently outperformed Prophet in most clusters, especially for items with more irregular flows.
  - Prophet captured seasonality well but required tuning to match SARIMAX’s precision.
- **Visual Plots** confirmed where models performed well and where improvements are needed.

---

## ✅ Business Value

This project enables warehouses to move from reactive to **data-driven planning** by:

- Understanding stock behaviour at the cluster level
- Predicting inbound volume more accurately
- Improving use of space through better storage layout decisions
- Reducing overstocking or under-preparation
- Supporting staffing and resource planning based on expected volume

By combining clustering with accurate forecasting, warehouse operators can **increase efficiency, reduce cost**, and make informed decisions — with minimal tech overhead.

---

## 🧰 Technologies Used

- Python
- scikit-learn (clustering)
- Facebook Prophet
- statsmodels (SARIMAX)
- pandas, matplotlib, seaborn (EDA and visualisation)

---
