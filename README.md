# Warehouse Clustering and Forecasting

This project began after structured stakeholder feedback sessions, which we analysed using VADER sentiment analysis. The analysis highlighted concerns around:

- Storage misalignment (the joint most common category, with strongly negative sentiment)
- Picking delays
- Forecasting gaps (the most negative score, although based on a single comment)

These insights shaped the project scope and aligned it with operational priorities in a high-volume logistics environment, combining clustering and time-series forecasting to improve layout planning, resource allocation, and operational efficiency.

---

## 📦 Project Overview

Warehouses often store thousands of product types, each with different volume, weight, value, and picking frequency.
This project answers two key questions:
1. Can we group products based on how they behave in the warehouse?
2. Can we forecast how much stock will move in the future for each group?

The answer to both is yes — using a combination of **clustering** and **time series forecasting**.

---

## 🧪 Methods Used

### 1. Clustering (Product Grouping)
Using operational behaviour features only, such as:
- Average volume
- Weight
- Unit price
- Pick frequency
- Lead time
- Storage cost per unit

Product category and dates were deliberately **not** used as inputs, so the clusters reflect how products behave. K-Means found three clusters (best silhouette score at k = 3):
- **Fast-Moving Chemicals**: light, low-value items with the highest pick frequency and the shortest lead times
- **Slow-Moving Heavy Electronics**: the heaviest items, with the lowest pick frequency and the longest lead times
- **High-Value Pharmaceuticals**: the most valuable items, with the highest volumes and moderate pick frequency

Even without being told the category, the clusters match the product categories for 99.2% of products, confirming each category behaves distinctly.

### 2. Forecasting (Volume Prediction)
Time series forecasting models were trained on monthly inbound volumes for each cluster using:
- **Prophet** (from Meta), baseline and tuned
- **SARIMAX** (from statsmodels)

All models were trained on 2020–2023 and tested on the unseen 2024 data (synthetic).

---

## 📊 Key Results

- **Clustering** created meaningful operational categories with actionable storage advice.
- **Forecasting Accuracy** (MAPE on 2024, lower is better):

| Cluster | SARIMAX | Prophet | Prophet (tuned) | Best model |
|---|---|---|---|---|
| Fast-Moving Chemicals | 4.87% | 5.00% | **4.05%** | Prophet (tuned) |
| Slow-Moving Heavy Electronics | **4.61%** | 4.96% | 5.04% | SARIMAX |
| High-Value Pharmaceuticals | 8.42% | **4.53%** | 4.53% | Prophet |

  - No single model won everywhere: Prophet was most accurate for two of the three clusters.
  - Prophet handled the 2024 upturn in Pharmaceuticals far better than SARIMAX (4.5% vs 8.4%).
  - Every cluster can be forecast within about 5% using its best model, well inside the 15% target.
- **Visual Plots** confirmed where models performed well and where improvements are needed.
- **Evaluation Metrics**: MAE, RMSE, MAPE

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

## ✅ Recommendations

📅 Use the best model per cluster for monthly volume planning (Prophet for Chemicals and Pharmaceuticals, SARIMAX for Electronics), and review the choice each year.
🗂️ Redesign warehouse layout based on cluster profiles: fast movers near picking and packing, heavy slow movers in bulk storage, high-value items in secure zones.
🧑‍🤝‍🧑 Integrate forecasts into staffing and equipment scheduling.
📣 Maintain stakeholder engagement to refine KPIs.

---

## ⚠️ Limitations

- Synthetic data, with 60 months of history per cluster (48 for training)
- Results are based on a single 12-month test period
- External demand drivers not modelled

---

## 🚀 Future Work

- Real-time data integration
- Dynamic modelling
- Scalable deployment across larger warehouses

---

## 🔄 Data

- Synthetic data created for educational purposes, stored in the `data/` folder

---

## 📝 Version Note

An earlier version reported tuned Prophet errors of 1–2%. Those models had been trained on data that included the 2024 test year. All models are now trained on 2020–2023 only, and the results above come from the corrected notebook.

---

## 🧰 Technologies Used

- Python
- scikit-learn (clustering)
- Prophet
- statsmodels (SARIMAX)
- pandas, matplotlib, seaborn (EDA and visualisation)
- NLTK VADER (sentiment analysis)
