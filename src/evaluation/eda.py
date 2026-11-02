"""Data understanding + EDA: quality tables (CSV/JSON) and figures (PNG). Run via `python main.py analyze`."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.loader import EXPECTED_COLUMNS

MONTH_ORDER = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
                "October", "November", "December"]
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False})


def _save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------------------------------ tables
def quality_tables(raw: pd.DataFrame, clean: pd.DataFrame, clean_report: dict, out: Path) -> dict:
    t = out / "tables"; t.mkdir(parents=True, exist_ok=True)
    num = raw.select_dtypes("number")

    overview = pd.DataFrame({
        "dtype": raw.dtypes.astype(str), "n_missing": raw.isna().sum(),
        "pct_missing": (raw.isna().mean() * 100).round(2), "n_unique": raw.nunique(),
    })
    overview.to_csv(t / "column_overview.csv")

    desc = num.describe(percentiles=[.01, .25, .5, .75, .99]).T
    desc["skew"] = num.skew()
    desc.round(3).to_csv(t / "numeric_summary.csv")

    cats = raw.select_dtypes(exclude="number").columns
    pd.DataFrame({"n_unique": raw[cats].nunique(),
                  "top_value": [raw[c].value_counts().index[0] for c in cats],
                  "top_share_pct": [round(raw[c].value_counts(normalize=True).iloc[0] * 100, 1) for c in cats],
                  }).to_csv(t / "categorical_summary.csv")

    guests = raw["Adults"] + raw["Children"].fillna(0) + raw["Babies"]
    nights = raw["StaysInWeekendNights"] + raw["StaysInWeekNights"]
    issues = {
        "rows": len(raw), "columns": raw.shape[1],
        "exact_duplicate_rows": int(raw.duplicated().sum()),
        "pct_duplicates": round(raw.duplicated().mean() * 100, 2),
        "cancel_rate_duplicates": round(float(raw[raw.duplicated(keep=False) & raw.duplicated()]["IsCanceled"].mean()), 3),
        "cancel_rate_all_rows": round(float(raw["IsCanceled"].mean()), 3),
        "missing_Children": int(raw["Children"].isna().sum()),
        "missing_Country": int(raw["Country"].isna().sum()),
        "NULL_Agent": int(raw["Agent"].isna().sum()), "NULL_Company": int(raw["Company"].isna().sum()),
        "zero_guest_rows": int((guests == 0).sum()), "zero_night_rows": int((nights == 0).sum()),
        "ADR_zero": int((raw["ADR"] == 0).sum()), "ADR_negative": int((raw["ADR"] < 0).sum()),
        "ADR_above_1000": int((raw["ADR"] > 1000).sum()), "ADR_max": float(raw["ADR"].max()),
        "babies_more_than_adults": int((raw["Babies"] > raw["Adults"]).sum()),
        "undefined_MarketSegment": int((raw["MarketSegment"] == "Undefined").sum()),
        "undefined_DistributionChannel": int((raw["DistributionChannel"] == "Undefined").sum()),
        "ReservationStatus_vs_IsCanceled": pd.crosstab(raw["ReservationStatus"], raw["IsCanceled"]).to_dict(),
        "cleaning": clean_report,
    }
    with open(out / "data_quality_report.json", "w") as f:
        json.dump(issues, f, indent=2, default=int)
    return issues


# ------------------------------------------------------------------------------------------ figures
def figures(raw: pd.DataFrame, df: pd.DataFrame, out: Path) -> list[str]:
    f = out / "figures"; f.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df["total_nights"] = df["StaysInWeekendNights"] + df["StaysInWeekNights"]
    made = []

    def done(name): made.append(name)

    # 1 missing values (raw)
    miss = raw.isna().sum()
    fig, ax = plt.subplots(figsize=(6, 3))
    miss[miss > 0].sort_values().plot.barh(ax=ax, color="#c0504d")
    ax.set_title("Missing values per column (raw H2; 'NULL' converted to NaN)"); ax.set_xscale("log")
    _save(fig, f / "01_missing_values.png"); done("01_missing_values.png")

    # 2 targets
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
    df["IsCanceled"].value_counts().sort_index().plot.bar(ax=ax[0], color=["#4f81bd", "#c0504d"], rot=0)
    ax[0].set_title(f"IsCanceled (cancel rate {df['IsCanceled'].mean():.1%})")
    ax[1].hist(df["ADR"], bins=80, color="#4f81bd"); ax[1].set_title("ADR (skewed right)"); ax[1].set_xlabel("ADR")
    ax[2].hist(np.log1p(df["ADR"]), bins=80, color="#9bbb59"); ax[2].set_title("log1p(ADR)"); ax[2].set_xlabel("log1p(ADR)")
    _save(fig, f / "02_targets.png"); done("02_targets.png")

    # 3 numeric distributions
    cols = ["LeadTime", "total_nights", "Adults", "PreviousCancellations", "DaysInWaitingList", "BookingChanges"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 6))
    for a, c in zip(axes.ravel(), cols):
        a.hist(df[c], bins=50, color="#4f81bd"); a.set_yscale("log"); a.set_title(f"{c} (skew={df[c].skew():.1f}, log y)")
    _save(fig, f / "03_numeric_distributions.png"); done("03_numeric_distributions.png")

    # 4 outliers
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    for a, (c, d) in zip(axes, [("ADR", raw), ("ADR (after cleaning)", df), ("LeadTime", df), ("total_nights", df)]):
        col = c.split(" ")[0]
        a.boxplot(d[col].dropna(), vert=True, flierprops=dict(markersize=2)); a.set_title(c)
    _save(fig, f / "04_outliers_boxplots.png"); done("04_outliers_boxplots.png")

    # 5 cancel rate by categoricals
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    for a, c in zip(axes.ravel(), ["MarketSegment", "DepositType", "CustomerType", "DistributionChannel", "Meal", "ArrivalDateYear"]):
        r = df.groupby(c)["IsCanceled"].agg(["mean", "size"]).sort_values("mean")
        r["mean"].plot.barh(ax=a, color="#c0504d"); a.set_title(f"Cancel rate by {c}"); a.set_xlim(0, 1)
        for i, (m, n) in enumerate(zip(r["mean"], r["size"])):
            a.text(min(m + 0.01, 0.8), i, f"n={n}", va="center", fontsize=7)
    _save(fig, f / "05_cancel_rate_by_category.png"); done("05_cancel_rate_by_category.png")

    # 6 seasonality
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    m = df.groupby("ArrivalDateMonth")[["ADR", "IsCanceled", "LeadTime"]].mean().reindex(MONTH_ORDER)
    m["ADR"].plot(ax=axes[0], marker="o"); axes[0].set_title("Mean ADR by arrival month")
    m["IsCanceled"].plot(ax=axes[1], marker="o", color="#c0504d"); axes[1].set_title("Cancel rate by arrival month")
    df.boxplot(column="ADR", by="MarketSegment", ax=axes[2], rot=45, flierprops=dict(markersize=2))
    axes[2].set_title("ADR by MarketSegment"); plt.suptitle("")
    for a in axes[:2]: a.tick_params(axis="x", rotation=45)
    _save(fig, f / "06_seasonality_and_price.png"); done("06_seasonality_and_price.png")

    # 7 correlation
    num = df.select_dtypes("number").drop(columns=["total_nights"]).copy()
    corr = num.corr()
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1); ax.grid(False)
    ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns, fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8); ax.set_title("Pearson correlation (numeric columns, cleaned data)")
    _save(fig, f / "07_correlation.png"); done("07_correlation.png")
    corr[["IsCanceled", "ADR"]].round(3).to_csv(out / "tables" / "correlation_with_targets.csv")

    # 8 leakage evidence
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.8))
    mism = (df["ReservedRoomType"] != df["AssignedRoomType"]).map({False: "Assigned == Reserved", True: "Assigned != Reserved"})
    df.groupby(mism)["IsCanceled"].mean().plot.bar(ax=axes[0], rot=15, color="#c0504d"); axes[0].set_title("Cancel rate by room mismatch")
    df.groupby(df["BookingChanges"].clip(upper=3))["IsCanceled"].mean().plot.bar(ax=axes[1], rot=0, color="#c0504d")
    axes[1].set_title("Cancel rate by BookingChanges (3 = 3+)")
    df.groupby(df["TotalOfSpecialRequests"].clip(upper=3))["IsCanceled"].mean().plot.bar(ax=axes[2], rot=0, color="#c0504d")
    axes[2].set_title("Cancel rate by special requests")
    pd.crosstab(df["ReservationStatus"], df["IsCanceled"]).plot.bar(ax=axes[3], rot=0, stacked=True)
    axes[3].set_title("ReservationStatus vs IsCanceled (1:1 -> direct leak)")
    for a in axes[:3]: a.set_ylim(0, 1)
    _save(fig, f / "08_leakage_evidence.png"); done("08_leakage_evidence.png")

    # 9 cardinality
    card = raw.select_dtypes(exclude="number").nunique().sort_values()
    fig, ax = plt.subplots(figsize=(6, 3.6)); card.plot.barh(ax=ax, color="#4f81bd")
    ax.set_xscale("log"); ax.set_title("Cardinality of categorical columns (log)")
    _save(fig, f / "09_cardinality.png"); done("09_cardinality.png")

    # 10 duplicates
    dup = raw.duplicated()
    fig, ax = plt.subplots(figsize=(5, 3.4))
    pd.DataFrame({"unique rows": [raw[~dup]["IsCanceled"].mean()], "duplicate rows": [raw[dup]["IsCanceled"].mean()]}).T.plot.bar(
        ax=ax, legend=False, color="#c0504d", rot=0)
    ax.set_title("Cancel rate: unique vs duplicated rows"); ax.set_ylim(0, 1)
    _save(fig, f / "10_duplicates_cancel_rate.png"); done("10_duplicates_cancel_rate.png")
    return made


def run_eda(raw: pd.DataFrame, clean: pd.DataFrame, clean_report: dict, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    issues = quality_tables(raw, clean, clean_report, out)
    issues["figures"] = figures(raw, clean, out)
    return issues
