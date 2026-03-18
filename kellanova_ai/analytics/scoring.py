"""
Opportunity Scoring Engine
Produces territory, rep and store level ranked opportunity summaries.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from config.settings import SEVERITY_WEIGHTS, EVENT_RADIUS_KM
from geopy.distance import geodesic


# ── Store-level opportunity summary ──────────────────────────────────────────
def score_stores(
    opportunities:    pd.DataFrame,
    hidden:           pd.DataFrame,
    stores:           pd.DataFrame,
    events:           pd.DataFrame,
) -> pd.DataFrame:
    """Return one row per store with total opportunity value and priority score."""
    opp_agg = (
        opportunities.groupby("store_id")
        .agg(
            total_opportunity_value=("opportunity_value", "sum"),
            total_priority_score   =("priority_score",   "sum"),
            issue_count            =("opportunity_id",   "count"),
            top_issue              =("issue_type", lambda x: x.value_counts().idxmax()),
        )
        .reset_index()
    )
    df = stores[["store_id","store_name","chain","city","region","rep_id","territory_id",
                 "latitude","longitude"]].merge(opp_agg, on="store_id", how="left")
    df = df.merge(hidden[["store_id","hidden_opportunity"]], on="store_id", how="left")
    df["total_opportunity_value"] = df["total_opportunity_value"].fillna(0)
    df["total_priority_score"]    = df["total_priority_score"].fillna(0)
    df["hidden_opportunity"]      = df["hidden_opportunity"].fillna(0)
    df["combined_opportunity"]    = df["total_opportunity_value"] + df["hidden_opportunity"] * 0.5

    # Event boost
    def event_factor(row):
        for _, ev in events.iterrows():
            dist = geodesic((row.latitude, row.longitude), (ev.latitude, ev.longitude)).km
            if dist <= EVENT_RADIUS_KM:
                return 1.0 + min(0.3, ev.expected_attendance / 100_000)
        return 1.0

    df["event_factor"] = df.apply(event_factor, axis=1)
    df["final_score"]  = (
        df["total_priority_score"] * df["event_factor"]
        + df["hidden_opportunity"] * 0.3
    ).round(2)
    df["store_rank"]   = df["final_score"].rank(ascending=False, method="min").astype(int)
    return df.sort_values("final_score", ascending=False).reset_index(drop=True)


# ── Rep-level summary ─────────────────────────────────────────────────────────
def score_reps(
    store_scores: pd.DataFrame,
    reps:         pd.DataFrame,
    visit_plan:   pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate store scores to rep level."""
    rep_agg = (
        store_scores.groupby("rep_id")
        .agg(
            stores_managed        =("store_id",              "count"),
            total_opportunity_value=("total_opportunity_value","sum"),
            hidden_opportunity    =("hidden_opportunity",    "sum"),
            high_priority_stores  =("final_score",           lambda x: (x > x.quantile(0.75)).sum()),
            avg_priority_score    =("final_score",           "mean"),
        )
        .reset_index().round(2)
    )
    return reps.merge(rep_agg, on="rep_id", how="left").fillna(0)


# ── Territory-level summary ────────────────────────────────────────────────────
def score_territories(
    store_scores:   pd.DataFrame,
    rep_scores:     pd.DataFrame,
    territories:    pd.DataFrame,
    opportunities:  pd.DataFrame,
    events:         pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate to territory level."""
    terr_agg = (
        store_scores.groupby("territory_id")
        .agg(
            total_stores           =("store_id",               "count"),
            stores_with_opportunity=("issue_count",            lambda x: (x > 0).sum()),
            total_opportunity_value=("total_opportunity_value","sum"),
            total_hidden_opp       =("hidden_opportunity",     "sum"),
            event_opp_stores       =("event_factor",           lambda x: (x > 1.0).sum()),
        )
        .reset_index().round(2)
    )

    # Promotion compliance %
    promo_comp = (
        opportunities[opportunities["issue_type"]=="promotion_execution"]
        .merge(store_scores[["store_id","territory_id"]], on="store_id")
        .groupby("territory_id")["store_id"].nunique()
        .rename("promo_non_compliant")
        .reset_index()
    )
    terr_agg = terr_agg.merge(promo_comp, on="territory_id", how="left").fillna(0)
    terr_agg["promo_compliance_pct"] = (
        (1 - terr_agg["promo_non_compliant"] / terr_agg["total_stores"].clip(lower=1)) * 100
    ).round(1)

    return territories.merge(terr_agg, on="territory_id", how="left").fillna(0)


# ── Opportunity type breakdown ────────────────────────────────────────────────
def opportunity_breakdown(
    opportunities: pd.DataFrame,
    level: str = "territory",
    level_id: str | None = None,
    store_scores: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return opportunity counts and value by issue type for a given level."""
    df = opportunities.copy()
    if level in ("territory", "rep") and store_scores is not None and level_id:
        col = "territory_id" if level == "territory" else "rep_id"
        valid_stores = store_scores[store_scores[col] == level_id]["store_id"].tolist()
        df = df[df["store_id"].isin(valid_stores)]
    elif level == "store" and level_id:
        df = df[df["store_id"] == int(level_id)]

    return (
        df.groupby("issue_type")
        .agg(count=("opportunity_id","count"), total_value=("opportunity_value","sum"))
        .reset_index()
        .sort_values("total_value", ascending=False)
        .round(2)
    )

