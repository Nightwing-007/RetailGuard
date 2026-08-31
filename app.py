"""
RetailGuard — Streamlit Web Application
End-to-end Return Risk Prediction & Proactive Loss Prevention System.
"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------
# Page Configuration & Modern Aesthetics
# ---------------------------------------------------------
st.set_page_config(
    page_title="RetailGuard — Return Risk Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern, high-end styling
st.markdown(
    """
    <style>
    /* Global style adjustments */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .hero-container {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.95) 0%, rgba(15, 23, 42, 0.98) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 28px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
    }
    
    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }
    
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        line-height: 1.6;
        margin-bottom: 0px;
    }
    
    .badge-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 8px;
        margin-top: 12px;
        background: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
    }
    
    .risk-card {
        padding: 24px;
        border-radius: 16px;
        margin-top: 16px;
        margin-bottom: 24px;
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
    }
    
    .risk-low {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(5, 150, 105, 0.05) 100%);
        border: 1px solid rgba(16, 185, 129, 0.35);
        border-left: 6px solid #10b981;
    }
    
    .risk-medium {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.12) 0%, rgba(217, 119, 6, 0.05) 100%);
        border: 1px solid rgba(245, 158, 11, 0.35);
        border-left: 6px solid #f59e0b;
    }
    
    .risk-high {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(220, 38, 38, 0.06) 100%);
        border: 1px solid rgba(239, 68, 68, 0.4);
        border-left: 6px solid #ef4444;
    }

    .recommendation-box {
        background: rgba(15, 23, 42, 0.7);
        border-radius: 12px;
        padding: 18px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        margin-top: 16px;
    }
    
    .info-card {
        background: rgba(30, 41, 59, 0.4);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.06);
        height: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Path Handling & Model Caching
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

FEATURE_COLUMNS = [
    "Quantity",
    "UnitPrice",
    "OrderValue",
    "Hour",
    "DayOfWeek",
    "Month",
    "Customer_Total_Orders",
    "Customer_Total_Items",
    "Customer_Average_Spend",
    "Customer_Return_Rate",
    "Product_Total_Sales",
    "Product_Total_Orders",
    "Product_Return_Rate",
]


@st.cache_resource(show_spinner="Loading trained artifacts...")
def load_ml_artifacts():
    """Load serialized model, scaler, and imputer."""
    model_path = DATA_DIR / "best_model.joblib"
    scaler_path = DATA_DIR / "scaler.joblib"
    imputer_path = DATA_DIR / "imputer.joblib"

    missing = []
    for name, p in [
        ("XGBoost Model", model_path),
        ("Feature Scaler", scaler_path),
        ("Imputer", imputer_path),
    ]:
        if not p.exists():
            missing.append(f"{name} ({p.name})")

    if missing:
        st.error(
            f"⚠️ Missing required pipeline artifacts in `data/`: {', '.join(missing)}. "
            "Please execute the pipeline scripts (`scripts/04_feature_engineering.py` & `notebooks/05_model_tuning_xgb.ipynb`) first."
        )
        st.stop()

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    imputer = joblib.load(imputer_path)
    return model, scaler, imputer


model, scaler, imputer = load_ml_artifacts()

# ---------------------------------------------------------
# Hero Banner
# ---------------------------------------------------------
st.markdown(
    """
    <div class="hero-container">
        <div class="hero-title">🛡️ RetailGuard</div>
        <div class="hero-subtitle">
            Enterprise Machine Learning system for real-time product return risk assessment and proactive reverse logistics management.
        </div>
        <div>
            <span class="badge-pill">⚡ Phase 5.5 Tuned XGBoost</span>
            <span class="badge-pill">📊 SMOTE Balanced</span>
            <span class="badge-pill">🎯 Minority F1 Optimized (0.2290 vs 0.1576 Baseline)</span>
            <span class="badge-pill">🔒 13-Feature Pipeline</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Sidebar: Presets & Hypothetical Transaction Inputs
# ---------------------------------------------------------
st.sidebar.title("🛒 Transaction Input")
st.sidebar.caption("Configure transaction, customer, and product attributes.")

# Preset Scenarios for Rapid Testing
SCENARIOS = {
    "Custom Configuration": None,
    "🟢 Standard Loyal Customer Order": {
        "qty": 6, "price": 2.55, "hour": 12, "day": "Wednesday", "month": "Aug",
        "cust_orders": 8, "cust_items": 95, "cust_spend": 320.0, "cust_ret_rate": 0.02,
        "prod_sales": 4500.0, "prod_orders": 210, "prod_ret_rate": 0.01,
    },
    "🟡 Volatile SKU / High Value Purchase": {
        "qty": 2, "price": 45.00, "hour": 18, "day": "Friday", "month": "Nov",
        "cust_orders": 3, "cust_items": 15, "cust_spend": 140.0, "cust_ret_rate": 0.15,
        "prod_sales": 8200.0, "prod_orders": 85, "prod_ret_rate": 0.12,
    },
    "🔴 High-Risk Serial Returner Batch": {
        "qty": 35, "price": 8.50, "hour": 21, "day": "Sunday", "month": "Dec",
        "cust_orders": 12, "cust_items": 340, "cust_spend": 850.0, "cust_ret_rate": 0.45,
        "prod_sales": 15000.0, "prod_orders": 320, "prod_ret_rate": 0.28,
    },
}

selected_scenario = st.sidebar.selectbox("⚡ Quick Scenario Presets", list(SCENARIOS.keys()), index=0)
defaults = SCENARIOS[selected_scenario]

# Group 1: Transaction Details
with st.sidebar.expander("📦 Transaction Details", expanded=True):
    quantity = st.number_input(
        "Quantity",
        min_value=1,
        max_value=10000,
        value=defaults["qty"] if defaults else 6,
        step=1,
        help="Number of units ordered in this transaction line item.",
    )
    unit_price = st.number_input(
        "Unit Price (£)",
        min_value=0.01,
        max_value=10000.0,
        value=defaults["price"] if defaults else 2.95,
        step=0.25,
        format="%.2f",
        help="Price per single unit in GBP (£).",
    )
    
    # Auto-calculated order value
    order_value = float(quantity) * float(unit_price)
    st.info(f"💰 **Computed Order Value:** `£{order_value:,.2f}`")

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        hour = st.slider(
            "Hour of Day",
            0,
            23,
            defaults["hour"] if defaults else 12,
            help="24-hour format (e.g. 14 for 2 PM).",
        )
        day_mapping = {
            "Monday": 0,
            "Tuesday": 1,
            "Wednesday": 2,
            "Thursday": 3,
            "Friday": 4,
            "Saturday": 5,
            "Sunday": 6,
        }
        day_default_idx = (
            list(day_mapping.keys()).index(defaults["day"]) if defaults else 2
        )
        day_name = st.selectbox("Day of Week", list(day_mapping.keys()), index=day_default_idx)
        day_of_week = day_mapping[day_name]

    with col_t2:
        month_mapping = {
            "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
            "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
        }
        month_default_idx = (
            list(month_mapping.keys()).index(defaults["month"]) if defaults else 7
        )
        month_name = st.selectbox("Month", list(month_mapping.keys()), index=month_default_idx)
        month = month_mapping[month_name]

# Group 2: Customer Profile
with st.sidebar.expander("👤 Customer Profile", expanded=True):
    cust_orders = st.number_input(
        "Customer Total Orders",
        min_value=1,
        max_value=2000,
        value=defaults["cust_orders"] if defaults else 5,
        step=1,
        help="Total number of completed orders by this customer.",
    )
    cust_items = st.number_input(
        "Customer Total Items",
        min_value=1,
        max_value=50000,
        value=defaults["cust_items"] if defaults else 60,
        step=5,
        help="Cumulative volume of units purchased by this customer historically.",
    )
    cust_avg_spend = st.number_input(
        "Customer Average Spend (£)",
        min_value=0.0,
        max_value=100000.0,
        value=defaults["cust_spend"] if defaults else 220.00,
        step=10.0,
        format="%.2f",
        help="Historical average spending per invoice.",
    )
    cust_return_rate = st.slider(
        "Customer Return Rate",
        min_value=0.00,
        max_value=1.00,
        value=defaults["cust_ret_rate"] if defaults else 0.05,
        step=0.01,
        format="%.2f",
        help="Proportion of past orders returned by this customer (0.00 = 0%, 1.00 = 100%).",
    )

# Group 3: Product Profile
with st.sidebar.expander("🏷️ Product Profile", expanded=True):
    prod_sales = st.number_input(
        "Product Total Sales (£)",
        min_value=0.0,
        max_value=500000.0,
        value=defaults["prod_sales"] if defaults else 3200.00,
        step=50.0,
        format="%.2f",
        help="Lifetime gross sales revenue generated by this product.",
    )
    prod_orders = st.number_input(
        "Product Total Orders",
        min_value=1,
        max_value=50000,
        value=defaults["prod_orders"] if defaults else 140,
        step=5,
        help="Total number of times this product has been ordered across all customers.",
    )
    prod_return_rate = st.slider(
        "Product Return Rate",
        min_value=0.00,
        max_value=1.00,
        value=defaults["prod_ret_rate"] if defaults else 0.03,
        step=0.01,
        format="%.2f",
        help="Historical return rate for this specific SKU / StockCode.",
    )

st.sidebar.markdown("---")
predict_clicked = st.sidebar.button(
    "🔍 Predict Return Risk",
    type="primary",
)

# ---------------------------------------------------------
# Inference Engine
# ---------------------------------------------------------
# Construct input dictionary matching exact Phase 4 feature column schema
input_dict = {
    "Quantity": [float(quantity)],
    "UnitPrice": [float(unit_price)],
    "OrderValue": [float(order_value)],
    "Hour": [int(hour)],
    "DayOfWeek": [int(day_of_week)],
    "Month": [int(month)],
    "Customer_Total_Orders": [float(cust_orders)],
    "Customer_Total_Items": [float(cust_items)],
    "Customer_Average_Spend": [float(cust_avg_spend)],
    "Customer_Return_Rate": [float(cust_return_rate)],
    "Product_Total_Sales": [float(prod_sales)],
    "Product_Total_Orders": [float(prod_orders)],
    "Product_Return_Rate": [float(prod_return_rate)],
}

input_df = pd.DataFrame(input_dict, columns=FEATURE_COLUMNS)

# Transform input through imputer & scaler
imputed_array = imputer.transform(input_df)
imputed_df = pd.DataFrame(imputed_array, columns=FEATURE_COLUMNS)
scaled_array = scaler.transform(imputed_df)
scaled_df = pd.DataFrame(scaled_array, columns=FEATURE_COLUMNS)

# Predict probability
probabilities = model.predict_proba(scaled_df)[0]
prob_keep = float(probabilities[0])
prob_return = float(probabilities[1])
risk_percentage = prob_return * 100

# ---------------------------------------------------------
# Risk Assessment Display
# ---------------------------------------------------------
st.subheader("🎯 Real-Time Return Risk Assessment")

# Determine Risk Level Tier
if risk_percentage < 25.0:
    risk_level = "LOW RISK"
    card_class = "risk-low"
    badge_color = "#10b981"
    risk_summary = "Transaction has low return likelihood. Customer and product indicators reflect strong retention patterns."
    rec_title = "✅ Recommended Action: Standard Fast-Track Dispatch"
    recommendations = [
        "Fulfill via standard automated logistics workflow.",
        "No special return paperwork or verification required.",
        "Customer profile demonstrates healthy retention history.",
    ]
elif risk_percentage < 55.0:
    risk_level = "MODERATE RISK"
    card_class = "risk-medium"
    badge_color = "#f59e0b"
    risk_summary = "Elevated return likelihood detected based on product category volatility or customer return propensity."
    rec_title = "⚠️ Recommended Action: Soft Retention & Automated Follow-Up"
    recommendations = [
        "Include proactive sizing / styling / exchange guide with package dispatch.",
        "Trigger automated post-delivery check-in email 48 hours after delivery.",
        "Ensure SKU stock buffer accommodates potential re-stocking turnaround.",
    ]
else:
    risk_level = "HIGH RISK"
    card_class = "risk-high"
    badge_color = "#ef4444"
    risk_summary = "High probability of return invoice. Elevated potential for logistical overhead and reverse freight cost."
    rec_title = "🚨 Recommended Action: Proactive Loss Prevention & Quality Check"
    recommendations = [
        "Flag order for manual warehouse pre-shipment quality & item condition inspection.",
        "Send proactive customer outreach / order confirmation to clarify specifications prior to courier handover.",
        "Stage reverse logistics return label and reserve local distribution center inventory.",
    ]

# Metrics Row
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    st.metric(
        label="Return Risk Probability",
        value=f"{risk_percentage:.1f}%",
        delta=f"{'+' if risk_percentage > 1.94 else ''}{risk_percentage - 1.94:.1f}% vs baseline avg",
        delta_color="inverse",
    )
with col_m2:
    st.metric(
        label="Risk Classification",
        value=risk_level,
    )
with col_m3:
    st.metric(
        label="Order Value at Stake",
        value=f"£{order_value:,.2f}",
    )
with col_m4:
    st.metric(
        label="Model Decision Threshold",
        value="Flagged (Return)" if prob_return >= 0.5 else "Cleared (Keep)",
    )

# Visual Progress Gauge
st.markdown("#### Return Risk Gauge")
st.progress(min(max(prob_return, 0.0), 1.0))

# Visual Result Card
st.markdown(
    f"""
    <div class="risk-card {card_class}">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <h3 style="margin: 0; color: {badge_color};">Risk Assessment: {risk_level} ({risk_percentage:.2f}%)</h3>
            <span style="font-weight: 700; color: #94a3b8; font-size: 0.9rem;">Model: Tuned XGBoost Classifier</span>
        </div>
        <p style="margin-top: 8px; font-size: 1.05rem; color: #e2e8f0;">{risk_summary}</p>
        
        <div class="recommendation-box">
            <h4 style="margin: 0 0 10px 0; color: #f8fafc;">{rec_title}</h4>
            <ul style="margin: 0; padding-left: 20px; color: #cbd5e1;">
                {''.join(f'<li>{r}</li>' for r in recommendations)}
            </ul>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Feature Inspection & Transaction Summary
# ---------------------------------------------------------
with st.expander("🔍 View Raw Feature Vector & Standardized Pipeline Data", expanded=False):
    tab1, tab2 = st.tabs(["Raw Feature Input", "Standardized / Scaled Inputs"])
    with tab1:
        st.dataframe(input_df)
    with tab2:
        st.dataframe(scaled_df.round(4))

# ---------------------------------------------------------
# Business Context & Operational Overview
# ---------------------------------------------------------
st.markdown("---")
st.subheader("💡 Business Context & Loss Prevention Architecture")

col_b1, col_b2 = st.columns([1.4, 1])

with col_b1:
    st.markdown(
        """
        <div class="info-card">
            <h4 style="margin-top:0; color: #38bdf8;">Why Machine Learning for Retail Returns?</h4>
            <ul>
                <li><strong>Severe Real-World Class Imbalance:</strong> In retail e-commerce, natural return rates are low (~1.94% in this dataset). Standard uncalibrated models often default to predicting 0% returns.</li>
                <li><strong>Synthetic Minority Oversampling (SMOTE):</strong> The training pipeline synthesized minority return instances to teach the classifier nuanced return patterns without degrading precision.</li>
                <li><strong>Optimized for F1-Score:</strong> The XGBoost model was hyperparameter-tuned specifically maximizing minority-class F1-Score (<strong>0.2290</strong> vs 0.1576 baseline, a <strong>+45.3% relative improvement</strong>).</li>
                <li><strong>Proactive vs. Reactive:</strong> Rather than absorbing reverse logistics costs post-fulfillment, operations teams can intercept high-risk shipments, reduce restocking friction, and improve customer satisfaction.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_b2:
    st.markdown(
        """
        <div class="info-card">
            <h4 style="margin-top:0; color: #818cf8;">Pipeline Architecture Snapshot</h4>
            <ol>
                <li><strong>Data Cleaning:</strong> ISO-8859-1 parser, CustomerID filtering, positive price validation.</li>
                <li><strong>FIFO Return Reconciliation:</strong> Pairs return cancellations (<code>C...</code>) to original purchase records.</li>
                <li><strong>Feature Engineering:</strong> Computes historical customer frequency, monetary spend, and SKU return rates.</li>
                <li><strong>Live Inference:</strong> Median Imputation &rarr; StandardScaler &rarr; Tuned XGBoost probability prediction.</li>
            </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Footer
st.markdown("---")
st.caption(
    "RetailGuard DS • Built with Streamlit, Scikit-Learn, & XGBoost • Production Deployment Interface"
)
