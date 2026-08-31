# 🛡️ RetailGuard: Enterprise Return Risk Intelligence & Loss Prevention

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/framework-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Model](https://img.shields.io/badge/ML%20Model-Tuned%20XGBoost-orange?logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![Optimization](https://img.shields.io/badge/Imbalance%20Handling-SMOTE-green)](https://imbalanced-learn.org/)
[![License](https://img.shields.io/badge/license-MIT-purple.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-Production%20Ready-brightgreen.svg)]()

> **RetailGuard** is an end-to-end Machine Learning intelligence platform designed to predict, evaluate, and mitigate retail product returns in real-time. By transforming historical transactional records into predictive risk scores, RetailGuard enables e-commerce merchants and warehouse operators to shift from **reactive reverse logistics** to **proactive loss prevention**.

---

## 📌 Executive Summary & Problem Context

Product returns represent one of the largest hidden profit drains in modern retail and e-commerce:
- **Financial Drag:** Retail returns cost merchants hundreds of billions annually in reverse shipping, warehouse restocking fees, inventory depreciation, and unsalvageable goods.
- **Logistical Bottlenecks:** Unplanned return volumes overload fulfillment centers and complicate inventory forecasting.
- **The "Needle in a Haystack" Problem:** In real-world transactional data, return rates are naturally sparse (~1.94% in this benchmark dataset). Standard machine learning algorithms fail on such extreme class imbalance, often predicting zero returns across the board.

**RetailGuard** solves this challenge by pairing **Synthetic Minority Over-sampling (SMOTE)** with a **hyperparameter-tuned XGBoost classifier** optimized specifically for the minority-class $F_1$-score. The platform delivers instant, interpretable risk assessments before an order leaves the fulfillment center.

---

## ✨ Key Capabilities & Business Value

```
┌────────────────────────┐      ┌─────────────────────────┐      ┌──────────────────────────┐
│  Real-Time Risk Score  │ ───► │ Root-Cause Attribution  │ ───► │  Proactive Intervention  │
│  Predict probability   │      │ 13 behavioral customer  │      │ Automated operational    │
│  of return per invoice │      │ & product risk features │      │ dispatch recommendations │
└────────────────────────┘      └─────────────────────────┘      └──────────────────────────┘
```

### 1. 🎯 Real-Time Return Risk Scoring
- Evaluates individual line items and full shopping carts instantaneously upon checkout.
- Outputs calibrated probability percentages and categorizes transactions into actionable tiers: **Low Risk (<25%)**, **Moderate Risk (25%–55%)**, and **High Risk (>55%)**.

### 2. 🔍 Root-Cause Behavioral Attribution
- Synthesizes **13 engineered features** across temporal dynamics, monetary spend, customer purchase history, and product category return volatility.
- Identifies whether the primary risk driver is customer propensity (e.g., serial returners), SKU-level volatility (e.g., sizing discrepancies), or basket anomaly (e.g., large volume multi-buys).

### 3. 🚨 Actionable Operational Recommendations
- **Low Risk:** Clears order for standard automated high-speed fulfillment.
- **Moderate Risk:** Triggers soft retention measures (e.g., inserting exchange/sizing guides, automated 48-hour delivery check-ins).
- **High Risk:** Flags order for pre-shipment quality checks, customer outreach to verify specifications, and proactive reverse-logistics buffer allocation.

### 4. 🖥️ Interactive Streamlit Decision Dashboard
- Accessible, non-technical web interface featuring interactive scenario presets, visual risk progress gauges, dynamic order value calculations, and deep-dive feature vector inspection.

---

## 🖥️ Application Interface & Demonstration

RetailGuard includes a full-featured web dashboard designed for operations managers, fraud analysts, and customer support leads.

```text
+-----------------------------------------------------------------------------------+
|  [Sidebar Controls]                  RetailGuard — Return Risk Intelligence       |
|  - Scenario Presets                  --------------------------------------       |
|  - Transaction Inputs                Risk Probability: [ 74.2% ] (HIGH RISK)      |
|  - Customer Profile                  Order Value at Stake: £297.50                |
|  - Product Profile                   --------------------------------------       |
|                                      Action: 🚨 Manual Quality & Specs Outreach   |
|  [ Predict Return Risk ]             [=========================>........] 74.2%   |
+-----------------------------------------------------------------------------------+
```

### Dashboard Preview

![RetailGuard Web Dashboard](https://raw.githubusercontent.com/Nightwing-007/RetailGuard/main/reports/figures/cm_xgboost.png)
*(Interactive web UI screenshot: select hypothetical transactions or scenario presets to preview risk scores and operational directives in real time).*

---

## 💼 Business ROI & Impact Matrix

| Dimension | Traditional Reactive Approach | RetailGuard Proactive System |
| :--- | :--- | :--- |
| **Reverse Logistics Cost** | Paid in full on all returned items | Reduced by intercepting high-risk orders pre-dispatch |
| **Return Inventory Velocity** | 14–30 day turnaround before restocking | Anticipated return volumes staged for rapid re-shelving |
| **Customer Retention** | Friction-heavy disputes post-return | Proactive sizing & specification support prevents returns |
| **Model Sensitivity** | Misses rare return events due to class imbalance | Optimized for minority-class $F_1$ (**+45.3% relative lift**) |

---

## 🚀 Quick Start for Non-Technical Stakeholders

To launch the local decision dashboard:

1. **Activate your environment and start the app:**
   ```bash
   streamlit run app.py
   ```
2. **Access the web application:** Open `http://localhost:8501` in your browser.
3. **Test Scenarios:** Use the **Quick Scenario Presets** dropdown in the sidebar to test:
   - 🟢 *Standard Loyal Customer Order*
   - 🟡 *Volatile SKU / High Value Purchase*
   - 🔴 *High-Risk Serial Returner Batch*

---

## 🛠️ Technical Architecture & Developer Documentation

For complete technical specifications, data pipeline execution, machine learning experiments, and development setup, please refer to the dedicated developer guide:

👉 **[Read the Full Developer & Architecture Guide (DEVELOPER.md)](DEVELOPER.md)**

The developer documentation covers:
- **End-to-End System Architecture** (Phase 1 through Phase 6)
- **FIFO Return Allocation Algorithm**
- **SMOTE Balancing & Feature Scaling**
- **XGBoost Hyperparameter Tuning & Cross-Validation Results**
- **Step-by-Step Script & Notebook Reproducibility Guide**

---

## 📄 License & Attribution

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details. Built with open retail transaction datasets, Scikit-Learn, XGBoost, and Streamlit.
