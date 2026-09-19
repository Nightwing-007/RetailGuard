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
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="RetailGuard — Return Risk Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Global CSS Injection — Premium Dark SaaS Theme
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        background-color: #070c18 !important;
        color: #e2e8f0;
    }

    /* Hide Streamlit chrome */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        max-width: 1280px;
    }

    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #0f172a; }
    ::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #475569; }

    /* SIDEBAR */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #080e1f 0%, #0d1528 100%) !important;
        border-right: 1px solid rgba(56, 189, 248, 0.12) !important;
    }
    [data-testid="stSidebar"] .block-container { padding-top: 1rem !important; }

    [data-testid="stSidebar"] [data-testid="stButton"] > button {
        width: 100% !important;
        background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 12px !important;
        font-size: 1rem !important;
        font-weight: 700 !important;
        padding: 14px 20px !important;
        letter-spacing: 0.03em;
        box-shadow: 0 4px 24px rgba(99, 102, 241, 0.45) !important;
        transition: all 0.25s ease !important;
        cursor: pointer !important;
    }
    [data-testid="stSidebar"] [data-testid="stButton"] > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 32px rgba(99, 102, 241, 0.65) !important;
    }

    [data-testid="stSidebar"] label {
        color: #94a3b8 !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
    }
    [data-testid="stSidebar"] .stSelectbox > div > div,
    [data-testid="stSidebar"] .stNumberInput > div > div > input {
        background: rgba(30, 41, 59, 0.7) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 8px !important;
        color: #f1f5f9 !important;
    }

    /* HERO */
    .hero-wrap {
        position: relative;
        background: linear-gradient(135deg, rgba(14,165,233,0.07) 0%, rgba(99,102,241,0.06) 50%, rgba(192,132,252,0.05) 100%);
        border: 1px solid rgba(99,102,241,0.2);
        border-radius: 20px;
        padding: 32px 36px;
        margin-bottom: 28px;
        overflow: hidden;
    }
    .hero-wrap::before {
        content: '';
        position: absolute;
        top: -60%; right: -10%;
        width: 420px; height: 420px;
        background: radial-gradient(circle, rgba(99,102,241,0.12) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero-live-dot {
        display: inline-block;
        width: 8px; height: 8px;
        background: #10b981;
        border-radius: 50%;
        margin-right: 6px;
        animation: pulse-dot 2s ease-in-out infinite;
        vertical-align: middle;
    }
    @keyframes pulse-dot {
        0%, 100% { box-shadow: 0 0 0 0 rgba(16,185,129,0.6); }
        50%       { box-shadow: 0 0 0 6px rgba(16,185,129,0); }
    }
    .hero-live-label {
        font-size: 0.7rem; font-weight: 700;
        letter-spacing: 0.1em; color: #10b981;
        text-transform: uppercase; vertical-align: middle;
    }
    .hero-title {
        font-size: 2.6rem; font-weight: 800;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.2; margin: 10px 0 8px 0;
    }
    .hero-subtitle {
        color: #94a3b8; font-size: 1rem; line-height: 1.65;
        max-width: 680px; margin-bottom: 16px;
    }
    .badge-pill {
        display: inline-block; padding: 4px 13px;
        border-radius: 9999px; font-size: 0.75rem; font-weight: 600;
        margin-right: 8px; margin-top: 4px;
        background: rgba(56,189,248,0.1); color: #7dd3fc;
        border: 1px solid rgba(56,189,248,0.25); letter-spacing: 0.02em;
    }
    .badge-pill.violet { background: rgba(129,140,248,0.1); color: #a5b4fc; border-color: rgba(129,140,248,0.25); }
    .badge-pill.green  { background: rgba(16,185,129,0.1);  color: #6ee7b7; border-color: rgba(16,185,129,0.25); }
    .badge-pill.purple { background: rgba(192,132,252,0.1); color: #d8b4fe; border-color: rgba(192,132,252,0.25); }

    /* METRIC STAT CARDS */
    .stat-card {
        background: rgba(15,23,42,0.8);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 16px; padding: 20px 22px;
        position: relative; overflow: hidden;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        height: 100%;
    }
    .stat-card:hover { transform: translateY(-3px); box-shadow: 0 12px 30px rgba(0,0,0,0.35); }
    .stat-card .accent-bar {
        position: absolute; top: 0; left: 0; right: 0;
        height: 3px; border-radius: 16px 16px 0 0;
    }
    .stat-label { font-size: 0.72rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em; color: #64748b; margin-bottom: 8px; }
    .stat-value { font-size: 1.7rem; font-weight: 800; line-height: 1; margin-bottom: 6px; }
    .stat-delta { font-size: 0.75rem; font-weight: 500; color: #64748b; }

    /* GAUGE */
    .gauge-wrap {
        background: rgba(15,23,42,0.8);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 16px; padding: 22px 26px; margin-bottom: 20px;
    }
    .gauge-track {
        position: relative; width: 100%; height: 14px;
        background: rgba(255,255,255,0.06); border-radius: 9999px;
        overflow: visible; margin: 14px 0 10px 0;
    }
    .gauge-fill {
        height: 100%; border-radius: 9999px;
        background: linear-gradient(90deg, #10b981 0%, #f59e0b 55%, #ef4444 100%);
        transition: width 0.7s cubic-bezier(0.34,1.56,0.64,1); position: relative;
    }
    .gauge-needle {
        position: absolute; top: 50%; right: -9px;
        transform: translateY(-50%);
        width: 18px; height: 18px;
        background: #f8fafc; border-radius: 50%;
        border: 3px solid #1e293b;
        box-shadow: 0 0 0 2px rgba(255,255,255,0.3), 0 4px 12px rgba(0,0,0,0.4);
    }
    .gauge-labels { display: flex; justify-content: space-between; font-size: 0.7rem; color: #475569; font-weight: 600; }
    .gauge-section-label { font-size: 0.7rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: #475569; margin-bottom: 2px; }
    .gauge-title { font-size: 1rem; font-weight: 700; color: #e2e8f0; }

    /* RISK RESULT CARD */
    .risk-card { border-radius: 18px; padding: 26px 30px; margin-bottom: 24px; position: relative; overflow: hidden; backdrop-filter: blur(12px); }
    .risk-low   { background: linear-gradient(135deg,rgba(16,185,129,0.09) 0%,rgba(5,150,105,0.04) 100%); border: 1px solid rgba(16,185,129,0.3); border-left: 5px solid #10b981; box-shadow: 0 8px 32px rgba(16,185,129,0.08); }
    .risk-medium{ background: linear-gradient(135deg,rgba(245,158,11,0.10) 0%,rgba(217,119,6,0.04) 100%); border: 1px solid rgba(245,158,11,0.3); border-left: 5px solid #f59e0b; box-shadow: 0 8px 32px rgba(245,158,11,0.08); }
    .risk-high  { background: linear-gradient(135deg,rgba(239,68,68,0.12) 0%,rgba(220,38,38,0.04) 100%); border: 1px solid rgba(239,68,68,0.35); border-left: 5px solid #ef4444; animation: borderPulse 2.8s ease-in-out infinite; }
    @keyframes borderPulse {
        0%,100% { box-shadow: 0 8px 32px rgba(239,68,68,0.10); }
        50%      { box-shadow: 0 8px 48px rgba(239,68,68,0.28); }
    }
    .risk-verdict-row { display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px; margin-bottom: 12px; }
    .risk-verdict-icon { font-size: 1.6rem; }
    .risk-verdict-title { font-size: 1.4rem; font-weight: 800; margin: 0; }
    .risk-summary-text  { font-size: 0.95rem; color: #cbd5e1; line-height: 1.65; margin: 4px 0 0 0; }
    .rec-box { background: rgba(7,12,24,0.55); border: 1px solid rgba(255,255,255,0.07); border-radius: 12px; padding: 18px 20px; }
    .rec-box h4 { margin: 0 0 12px 0; font-size: 0.88rem; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase; color: #94a3b8; }
    .rec-item { display: flex; align-items: flex-start; gap: 10px; padding: 7px 0; border-bottom: 1px solid rgba(255,255,255,0.04); font-size: 0.9rem; color: #cbd5e1; line-height: 1.55; }
    .rec-item:last-child { border-bottom: none; }
    .rec-icon { flex-shrink: 0; font-size: 1rem; margin-top: 1px; }
    .model-tag { font-size: 0.72rem; font-weight: 600; color: #475569; background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.07); border-radius: 6px; padding: 4px 10px; letter-spacing: 0.04em; }

    /* FEATURE TABLE */
    .feat-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; margin-top: 4px; }
    .feat-table th { background: rgba(15,23,42,0.9); color: #64748b; font-weight: 700; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; padding: 10px 16px; text-align: left; border-bottom: 1px solid rgba(255,255,255,0.07); }
    .feat-table td { padding: 9px 16px; border-bottom: 1px solid rgba(255,255,255,0.04); color: #cbd5e1; font-variant-numeric: tabular-nums; }
    .feat-table tr:hover td { background: rgba(56,189,248,0.04); }
    .feat-val-highlight { color: #38bdf8; font-weight: 600; }

    /* INFO CARDS */
    .info-card { background: rgba(15,23,42,0.7); border: 1px solid rgba(255,255,255,0.07); border-radius: 16px; padding: 24px 26px; height: 100%; backdrop-filter: blur(8px); }
    .info-card h4 { margin-top: 0; font-size: 1rem; font-weight: 700; margin-bottom: 14px; }
    .info-card li { color: #94a3b8; font-size: 0.88rem; line-height: 1.65; margin-bottom: 8px; }
    .info-card strong { color: #e2e8f0; }
    .info-card code { background: rgba(56,189,248,0.1); color: #38bdf8; padding: 1px 5px; border-radius: 4px; font-size: 0.82rem; }

    /* HEADINGS & DIVIDERS */
    .section-heading { font-size: 1.1rem; font-weight: 700; color: #e2e8f0; margin-bottom: 16px; display: flex; align-items: center; gap: 8px; }
    .section-heading .dot { width: 6px; height: 6px; background: #6366f1; border-radius: 50%; display: inline-block; }
    .divider { border: none; border-top: 1px solid rgba(255,255,255,0.06); margin: 28px 0; }

    /* FOOTER */
    .app-footer { text-align: center; color: #334155; font-size: 0.78rem; padding: 16px 0 8px 0; letter-spacing: 0.03em; }
    .app-footer span { color: #475569; }
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
    "Quantity", "UnitPrice", "OrderValue", "Hour", "DayOfWeek", "Month",
    "Customer_Total_Orders", "Customer_Total_Items", "Customer_Average_Spend",
    "Customer_Return_Rate", "Product_Total_Sales", "Product_Total_Orders", "Product_Return_Rate",
]


@st.cache_resource(show_spinner="Loading trained artifacts…")
def load_ml_artifacts():
    """Load serialized model, scaler, and imputer."""
    model_path   = DATA_DIR / "best_model.joblib"
    scaler_path  = DATA_DIR / "scaler.joblib"
    imputer_path = DATA_DIR / "imputer.joblib"
    missing = []
    for name, p in [("XGBoost Model", model_path), ("Feature Scaler", scaler_path), ("Imputer", imputer_path)]:
        if not p.exists():
            missing.append(f"{name} ({p.name})")
    if missing:
        st.error(
            f"⚠️ Missing required pipeline artifacts in `data/`: {', '.join(missing)}. "
            "Please execute the pipeline scripts first."
        )
        st.stop()
    return joblib.load(model_path), joblib.load(scaler_path), joblib.load(imputer_path)


model, scaler, imputer = load_ml_artifacts()

# ---------------------------------------------------------
# Hero Banner
# ---------------------------------------------------------
st.markdown(
    """
    <div class="hero-wrap">
        <div>
            <span class="hero-live-dot"></span>
            <span class="hero-live-label">Live Inference</span>
        </div>
        <div class="hero-title">🛡️ RetailGuard</div>
        <div class="hero-subtitle">
            Enterprise machine learning pipeline for real-time product return risk assessment
            and proactive reverse logistics management — powered by a Tuned XGBoost classifier.
        </div>
        <div>
            <span class="badge-pill">⚡ Phase 5.5 Tuned XGBoost</span>
            <span class="badge-pill violet">📊 SMOTE Balanced</span>
            <span class="badge-pill green">🎯 F1 +45.3% vs Baseline</span>
            <span class="badge-pill purple">🔒 13-Feature Pipeline</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
st.sidebar.markdown(
    """
    <div style="padding:12px 0 4px 0;">
        <div style="font-size:1.2rem;font-weight:800;color:#f8fafc;letter-spacing:-0.01em;">🛒 Transaction Input</div>
        <div style="font-size:0.78rem;color:#64748b;margin-top:3px;">Configure transaction, customer &amp; product attributes.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

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

with st.sidebar.expander("📦 Transaction Details", expanded=True):
    quantity   = st.number_input("Quantity", min_value=1, max_value=10000, value=defaults["qty"] if defaults else 6, step=1, help="Number of units ordered.")
    unit_price = st.number_input("Unit Price (£)", min_value=0.01, max_value=10000.0, value=defaults["price"] if defaults else 2.95, step=0.25, format="%.2f", help="Price per unit in GBP.")
    order_value = float(quantity) * float(unit_price)
    st.markdown(
        f"""<div style="background:rgba(14,165,233,0.08);border:1px solid rgba(14,165,233,0.2);border-radius:8px;padding:9px 14px;margin:4px 0 8px 0;">
            <span style="font-size:0.72rem;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:#38bdf8;">Computed Order Value</span><br>
            <span style="font-size:1.2rem;font-weight:800;color:#f0f9ff;">£{order_value:,.2f}</span>
        </div>""",
        unsafe_allow_html=True,
    )
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        hour = st.slider("Hour of Day", 0, 23, defaults["hour"] if defaults else 12, help="24-hour format.")
        day_mapping = {"Monday":0,"Tuesday":1,"Wednesday":2,"Thursday":3,"Friday":4,"Saturday":5,"Sunday":6}
        day_default_idx = list(day_mapping.keys()).index(defaults["day"]) if defaults else 2
        day_name    = st.selectbox("Day of Week", list(day_mapping.keys()), index=day_default_idx)
        day_of_week = day_mapping[day_name]
    with col_t2:
        month_mapping = {"Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,"Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12}
        month_default_idx = list(month_mapping.keys()).index(defaults["month"]) if defaults else 7
        month_name = st.selectbox("Month", list(month_mapping.keys()), index=month_default_idx)
        month      = month_mapping[month_name]

with st.sidebar.expander("👤 Customer Profile", expanded=True):
    cust_orders      = st.number_input("Customer Total Orders",    min_value=1,   max_value=2000,   value=defaults["cust_orders"]   if defaults else 5,      step=1,    help="Total completed orders.")
    cust_items       = st.number_input("Customer Total Items",     min_value=1,   max_value=50000,  value=defaults["cust_items"]    if defaults else 60,     step=5,    help="Cumulative units purchased.")
    cust_avg_spend   = st.number_input("Customer Avg Spend (£)",   min_value=0.0, max_value=100000.0,value=defaults["cust_spend"]   if defaults else 220.00, step=10.0, format="%.2f", help="Avg spend per invoice.")
    cust_return_rate = st.slider("Customer Return Rate", 0.00, 1.00, value=defaults["cust_ret_rate"] if defaults else 0.05, step=0.01, format="%.2f", help="Proportion of past orders returned.")

with st.sidebar.expander("🏷️ Product Profile", expanded=True):
    prod_sales       = st.number_input("Product Total Sales (£)",  min_value=0.0, max_value=500000.0,value=defaults["prod_sales"]   if defaults else 3200.00,step=50.0, format="%.2f", help="Lifetime gross sales.")
    prod_orders      = st.number_input("Product Total Orders",     min_value=1,   max_value=50000,  value=defaults["prod_orders"]   if defaults else 140,    step=5,    help="Total orders for this SKU.")
    prod_return_rate = st.slider("Product Return Rate", 0.00, 1.00, value=defaults["prod_ret_rate"] if defaults else 0.03, step=0.01, format="%.2f", help="Historical return rate for this SKU.")

st.sidebar.markdown("<hr style='border-color:rgba(255,255,255,0.07);margin:12px 0;'>", unsafe_allow_html=True)
predict_clicked = st.sidebar.button("🔍 Predict Return Risk", type="primary")

# ---------------------------------------------------------
# Inference Engine — UNCHANGED ML LOGIC
# ---------------------------------------------------------
input_dict = {
    "Quantity":               [float(quantity)],
    "UnitPrice":              [float(unit_price)],
    "OrderValue":             [float(order_value)],
    "Hour":                   [int(hour)],
    "DayOfWeek":              [int(day_of_week)],
    "Month":                  [int(month)],
    "Customer_Total_Orders":  [float(cust_orders)],
    "Customer_Total_Items":   [float(cust_items)],
    "Customer_Average_Spend": [float(cust_avg_spend)],
    "Customer_Return_Rate":   [float(cust_return_rate)],
    "Product_Total_Sales":    [float(prod_sales)],
    "Product_Total_Orders":   [float(prod_orders)],
    "Product_Return_Rate":    [float(prod_return_rate)],
}

input_df      = pd.DataFrame(input_dict, columns=FEATURE_COLUMNS)
imputed_array = imputer.transform(input_df)
imputed_df    = pd.DataFrame(imputed_array, columns=FEATURE_COLUMNS)
scaled_array  = scaler.transform(imputed_df)
scaled_df     = pd.DataFrame(scaled_array, columns=FEATURE_COLUMNS)

probabilities   = model.predict_proba(scaled_df)[0]
prob_keep       = float(probabilities[0])
prob_return     = float(probabilities[1])
risk_percentage = prob_return * 100

# ---------------------------------------------------------
# Risk Level Classification — UNCHANGED THRESHOLDS & TEXT
# ---------------------------------------------------------
if risk_percentage < 25.0:
    risk_level   = "LOW RISK";  card_class = "risk-low";    badge_color = "#10b981"; stat_accent = "#10b981"; verdict_icon = "✅"; rec_icon = "✅"
    risk_summary = "Transaction has low return likelihood. Customer and product indicators reflect strong retention patterns."
    rec_title    = "Recommended Action: Standard Fast-Track Dispatch"
    recommendations = [
        "Fulfill via standard automated logistics workflow.",
        "No special return paperwork or verification required.",
        "Customer profile demonstrates healthy retention history.",
    ]
elif risk_percentage < 55.0:
    risk_level   = "MODERATE RISK"; card_class = "risk-medium"; badge_color = "#f59e0b"; stat_accent = "#f59e0b"; verdict_icon = "⚠️"; rec_icon = "⚠️"
    risk_summary = "Elevated return likelihood detected based on product category volatility or customer return propensity."
    rec_title    = "Recommended Action: Soft Retention & Automated Follow-Up"
    recommendations = [
        "Include proactive sizing / styling / exchange guide with package dispatch.",
        "Trigger automated post-delivery check-in email 48 hours after delivery.",
        "Ensure SKU stock buffer accommodates potential re-stocking turnaround.",
    ]
else:
    risk_level   = "HIGH RISK"; card_class = "risk-high"; badge_color = "#ef4444"; stat_accent = "#ef4444"; verdict_icon = "🚨"; rec_icon = "🚨"
    risk_summary = "High probability of return invoice. Elevated potential for logistical overhead and reverse freight cost."
    rec_title    = "Recommended Action: Proactive Loss Prevention & Quality Check"
    recommendations = [
        "Flag order for manual warehouse pre-shipment quality & item condition inspection.",
        "Send proactive customer outreach / order confirmation to clarify specifications prior to courier handover.",
        "Stage reverse logistics return label and reserve local distribution center inventory.",
    ]

model_decision     = "Flagged (Return)" if prob_return >= 0.5 else "Cleared (Keep)"
delta_vs_baseline  = risk_percentage - 1.94

# ---------------------------------------------------------
# Section: Real-Time Risk Assessment
# ---------------------------------------------------------
st.markdown('<div class="section-heading"><span class="dot"></span> Real-Time Return Risk Assessment</div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""
    <div class="stat-card">
        <div class="accent-bar" style="background:{stat_accent};"></div>
        <div class="stat-label">Return Risk Probability</div>
        <div class="stat-value" style="color:{stat_accent};">{risk_percentage:.1f}%</div>
        <div class="stat-delta">{'▲' if delta_vs_baseline > 0 else '▼'} {abs(delta_vs_baseline):.1f}% vs 1.94% baseline avg</div>
    </div>""", unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="stat-card">
        <div class="accent-bar" style="background:{stat_accent};"></div>
        <div class="stat-label">Risk Classification</div>
        <div class="stat-value" style="color:{stat_accent};font-size:1.15rem;margin-top:6px;">{risk_level}</div>
        <div class="stat-delta">Tuned XGBoost Classifier</div>
    </div>""", unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="stat-card">
        <div class="accent-bar" style="background:#6366f1;"></div>
        <div class="stat-label">Order Value at Stake</div>
        <div class="stat-value" style="color:#818cf8;">£{order_value:,.2f}</div>
        <div class="stat-delta">{quantity} unit(s) @ £{unit_price:.2f} each</div>
    </div>""", unsafe_allow_html=True)

with c4:
    decision_color = "#ef4444" if prob_return >= 0.5 else "#10b981"
    st.markdown(f"""
    <div class="stat-card">
        <div class="accent-bar" style="background:{decision_color};"></div>
        <div class="stat-label">Model Decision (τ = 0.50)</div>
        <div class="stat-value" style="color:{decision_color};font-size:1.1rem;margin-top:6px;">{model_decision}</div>
        <div class="stat-delta">P(return) = {prob_return:.4f}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

# Animated Gauge
gauge_pct = min(max(risk_percentage, 0.0), 100.0)
st.markdown(f"""
<div class="gauge-wrap">
    <div style="display:flex;justify-content:space-between;align-items:center;">
        <div>
            <div class="gauge-section-label">Return Risk Gauge</div>
            <div class="gauge-title">Live probability score visualized across the risk spectrum</div>
        </div>
        <div style="font-size:2rem;font-weight:800;color:{stat_accent};">{gauge_pct:.1f}%</div>
    </div>
    <div class="gauge-track">
        <div class="gauge-fill" style="width:{gauge_pct}%;">
            <div class="gauge-needle"></div>
        </div>
    </div>
    <div class="gauge-labels">
        <span style="color:#10b981;">▐ LOW (0–25%)</span>
        <span style="color:#f59e0b;">▐ MODERATE (25–55%)</span>
        <span style="color:#ef4444;">▐ HIGH (55–100%)</span>
    </div>
</div>""", unsafe_allow_html=True)

# Risk Result Card
rec_items_html = "".join(
    f'<div class="rec-item"><span class="rec-icon">{rec_icon}</span><span>{r}</span></div>'
    for r in recommendations
)
st.markdown(f"""
<div class="risk-card {card_class}">
    <div class="risk-verdict-row">
        <div style="display:flex;align-items:center;gap:14px;">
            <span class="risk-verdict-icon">{verdict_icon}</span>
            <div>
                <div class="risk-verdict-title" style="color:{badge_color};">{risk_level} &nbsp;·&nbsp; {risk_percentage:.2f}%</div>
                <div class="risk-summary-text">{risk_summary}</div>
            </div>
        </div>
        <span class="model-tag">Tuned XGBoost Classifier</span>
    </div>
    <div class="rec-box">
        <h4>{rec_title}</h4>
        {rec_items_html}
    </div>
</div>""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Feature Inspection
# ---------------------------------------------------------
with st.expander("🔬 View Raw Feature Vector & Standardized Pipeline Data", expanded=False):
    tab1, tab2 = st.tabs(["📋 Raw Feature Input", "📐 Standardized / Scaled Inputs"])

    def _build_table(df):
        rows = ""
        for col in df.columns:
            val = df[col].iloc[0]
            fmt = f"{val:,.4f}" if isinstance(val, float) else str(val)
            rows += f'<tr><td>{col}</td><td class="feat-val-highlight">{fmt}</td></tr>'
        return f'<table class="feat-table"><thead><tr><th>Feature</th><th>Value</th></tr></thead><tbody>{rows}</tbody></table>'

    with tab1:
        st.markdown(_build_table(input_df), unsafe_allow_html=True)
    with tab2:
        st.markdown(_build_table(scaled_df.round(4)), unsafe_allow_html=True)

# ---------------------------------------------------------
# Business Context
# ---------------------------------------------------------
st.markdown("<hr class='divider'>", unsafe_allow_html=True)
st.markdown('<div class="section-heading"><span class="dot"></span> Business Context & Loss Prevention Architecture</div>', unsafe_allow_html=True)

col_b1, col_b2 = st.columns([1.4, 1])
with col_b1:
    st.markdown("""
    <div class="info-card">
        <h4 style="color:#38bdf8;">Why Machine Learning for Retail Returns?</h4>
        <ul>
            <li><strong>Severe Real-World Class Imbalance:</strong> In retail e-commerce, natural return rates are low (~1.94% in this dataset). Standard uncalibrated models often default to predicting 0% returns.</li>
            <li><strong>Synthetic Minority Oversampling (SMOTE):</strong> The training pipeline synthesized minority return instances to teach the classifier nuanced return patterns without degrading precision.</li>
            <li><strong>Optimized for F1-Score:</strong> The XGBoost model was tuned maximizing minority-class F1-Score (<strong>0.2290</strong> vs 0.1576 baseline — a <strong>+45.3% relative improvement</strong>).</li>
            <li><strong>Proactive vs. Reactive:</strong> Rather than absorbing reverse logistics costs post-fulfillment, operations teams can intercept high-risk shipments and improve customer satisfaction.</li>
        </ul>
    </div>""", unsafe_allow_html=True)

with col_b2:
    st.markdown("""
    <div class="info-card">
        <h4 style="color:#818cf8;">Pipeline Architecture Snapshot</h4>
        <ol>
            <li><strong>Data Cleaning:</strong> ISO-8859-1 parser, CustomerID filtering, positive price validation.</li>
            <li><strong>FIFO Return Reconciliation:</strong> Pairs return cancellations (<code>C...</code>) to original purchase records.</li>
            <li><strong>Feature Engineering:</strong> Computes historical customer frequency, monetary spend, and SKU return rates.</li>
            <li><strong>Live Inference:</strong> Median Imputation &rarr; StandardScaler &rarr; Tuned XGBoost probability prediction.</li>
        </ol>
    </div>""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("""
<hr class="divider">
<div class="app-footer">
    <span>🛡️ RetailGuard DS</span> &nbsp;·&nbsp;
    Built with <span>Streamlit</span>, <span>Scikit-Learn</span> &amp; <span>XGBoost</span>
    &nbsp;·&nbsp; Production Deployment Interface &nbsp;·&nbsp;
    <span>Phase 5.5 — Tuned Classifier</span>
</div>""", unsafe_allow_html=True)
