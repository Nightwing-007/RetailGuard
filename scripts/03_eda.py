"""
03_eda.py
Phase 3 EDA scripts: generate plots and summaries for slide deck (non-interactive run OK).
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams["figure.dpi"] = 120

DATA = Path(__file__).resolve().parent.parent / "data" / "preprocessed_retail.csv"
OUT_DIR = Path(__file__).resolve().parent.parent / "reports" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    df = pd.read_csv(DATA, parse_dates=["InvoiceDate"], low_memory=False)
    df["IsReturned"] = df["IsReturned"].astype(int)

    # Target distribution
    total = len(df)
    counts = df["IsReturned"].value_counts().sort_index()
    percent = (counts / total * 100).round(2)

    fig, ax = plt.subplots(figsize=(8,5))
    sns.barplot(x=counts.index.astype(str), y=counts.values, palette=["#4c78a8", "#f58518"], ax=ax)
    ax.set_title("Target distribution: IsReturned (counts)")
    ax.set_xlabel("IsReturned")
    ax.set_ylabel("Count")
    for i, v in enumerate(counts.values):
        ax.text(i, v + total*0.005, f"{v:,}\n({percent.values[i]}%)", ha="center", va="bottom", fontsize=12)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "isreturned_counts.png")
    plt.close(fig)

    # Temporal analysis: Month & Weekday
    df["Month"] = df["InvoiceDate"].dt.month
    df["Weekday"] = df["InvoiceDate"].dt.day_name()
    monthly = df.groupby("Month").agg(total=("IsReturned","size"), returns=("IsReturned","sum"))
    monthly["return_rate_pct"] = monthly["returns"] / monthly["total"] * 100

    fig, ax = plt.subplots(figsize=(12,5))
    sns.lineplot(x=monthly.index, y=monthly["return_rate_pct"], marker="o", linewidth=2.5, ax=ax, color="#f58518")
    ax.set_xticks(range(1,13))
    ax.set_xticklabels([pd.Timestamp(2000, m, 1).strftime("%b") for m in range(1,13)])
    ax.set_xlabel("Month")
    ax.set_ylabel("Return rate (%)")
    ax.set_title("Monthly return rate (%) — aggregated across years")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "monthly_return_rate.png")
    plt.close(fig)

    weekday_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    weekday = df.groupby("Weekday").agg(total=("IsReturned","size"), returns=("IsReturned","sum"))
    weekday["return_rate_pct"] = weekday["returns"] / weekday["total"] * 100
    weekday = weekday.reindex(weekday_order)

    fig, ax = plt.subplots(figsize=(10,5))
    sns.barplot(x=weekday.index, y=weekday["return_rate_pct"], palette="crest", ax=ax)
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Return rate (%)")
    ax.set_title("Return rate by day of week")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "weekday_return_rate.png")
    plt.close(fig)

    # Price buckets
    price = df["UnitPrice"].clip(lower=0.0001)
    bins = [0, 1, 5, 20, df["UnitPrice"].quantile(0.95), df["UnitPrice"].max()+1]
    labels = ["Very Low (<=1)","Low (1-5)","Medium (5-20)","High (20-95th pct)","Top 5%"]
    df["PriceBin"] = pd.cut(price, bins=bins, labels=labels, include_lowest=True)
    price_stats = df.groupby("PriceBin").agg(total=("IsReturned","size"), returns=("IsReturned","sum"))
    price_stats["return_rate_pct"] = price_stats["returns"] / price_stats["total"] * 100
    price_stats = price_stats.reindex(labels).fillna(0)

    fig, ax = plt.subplots(figsize=(12,5))
    sns.barplot(x=price_stats.index, y=price_stats["return_rate_pct"], palette="Spectral", ax=ax)
    ax.set_xlabel("UnitPrice bucket")
    ax.set_ylabel("Return rate (%)")
    ax.set_title("Return rate by UnitPrice bucket")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "price_bucket_return_rate.png")
    plt.close(fig)

    # Geographical: Top 10 countries
    if "Country" in df.columns:
        top_countries = df["Country"].value_counts().nlargest(10).index.tolist()
        geo = df[df["Country"].isin(top_countries)].groupby("Country").agg(total=("IsReturned","size"), returns=("IsReturned","sum"))
        geo["return_rate_pct"] = geo["returns"] / geo["total"] * 100
        geo = geo.sort_values("return_rate_pct", ascending=False)
        fig, ax = plt.subplots(figsize=(10,6))
        sns.barplot(x=geo["return_rate_pct"], y=geo.index, palette="mako", ax=ax)
        ax.set_xlabel("Return rate (%)")
        ax.set_ylabel("Country")
        ax.set_title("Return rate for Top 10 countries (by transaction volume)")
        plt.tight_layout()
        fig.savefig(OUT_DIR / "country_return_rate.png")
        plt.close(fig)

    print("EDA charts written to", OUT_DIR)

