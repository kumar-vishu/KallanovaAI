"""
Event & Activity Opportunity Engine
Links local events to nearby stores and generates recommended actions.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from config.settings import EVENT_RADIUS_KM
from geopy.distance import geodesic

# Category-specific event recommendations
EVENT_SKU_RECOMMENDATIONS = {
    "sports": {
        "categories": ["Savoury Snacks", "Snack Bars", "On-the-Go"],
        "action":     "Increase snack and on-the-go range facings; build event dump bin",
        "uplift_est": 0.35,
    },
    "concert": {
        "categories": ["Savoury Snacks", "Snack Bars"],
        "action":     "Feature Pringles and snack bars at store entrance; add POS material",
        "uplift_est": 0.28,
    },
    "festival": {
        "categories": ["Savoury Snacks", "Cereal", "On-the-Go"],
        "action":     "Build festival display; stock extra Pringles multipacks and LCMs",
        "uplift_est": 0.25,
    },
}

DEFAULT_REC = {
    "categories": ["Savoury Snacks"],
    "action":     "Increase snack inventory ahead of event",
    "uplift_est": 0.20,
}


def get_event_store_matches(
    events:  pd.DataFrame,
    stores:  pd.DataFrame,
    radius_km: float = EVENT_RADIUS_KM,
) -> pd.DataFrame:
    """Cross-join events and stores, filter by radius, return enriched matches."""
    rows = []
    for _, ev in events.iterrows():
        for _, st in stores.iterrows():
            dist = geodesic(
                (ev.latitude, ev.longitude),
                (st.latitude, st.longitude)
            ).km
            if dist <= radius_km:
                rec = EVENT_SKU_RECOMMENDATIONS.get(ev.event_type, DEFAULT_REC)
                rows.append({
                    "event_id":           ev.event_id,
                    "event_name":         ev.event_name,
                    "event_type":         ev.event_type,
                    "event_date":         ev.event_date,
                    "expected_attendance":ev.expected_attendance,
                    "store_id":           st.store_id,
                    "store_name":         st.store_name,
                    "chain":              st.chain,
                    "city":               st.city,
                    "rep_id":             st.rep_id,
                    "territory_id":       st.territory_id,
                    "distance_km":        round(dist, 2),
                    "event_latitude":      ev.latitude,
                    "event_longitude":     ev.longitude,
                    "recommended_categories": ", ".join(rec["categories"]),
                    "recommended_action": rec["action"],
                    "est_uplift_pct":     round(rec["uplift_est"] * 100, 1),
                    "est_revenue_uplift": round(
                        ev.expected_attendance * 0.01 * 4.50 * rec["uplift_est"], 2
                    ),
                })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["event_date","distance_km"]).reset_index(drop=True)
    return df


def get_rep_event_opportunities(
    rep_id:          str,
    event_store_map: pd.DataFrame,
) -> pd.DataFrame:
    return event_store_map[event_store_map["rep_id"] == rep_id].reset_index(drop=True)


def get_territory_event_opportunities(
    territory_id:    str,
    event_store_map: pd.DataFrame,
) -> pd.DataFrame:
    return event_store_map[event_store_map["territory_id"] == territory_id].reset_index(drop=True)


def get_store_event_opportunities(
    store_id:        int,
    event_store_map: pd.DataFrame,
) -> pd.DataFrame:
    return event_store_map[event_store_map["store_id"] == store_id].reset_index(drop=True)


def event_opportunity_summary(event_store_map: pd.DataFrame) -> pd.DataFrame:
    """Aggregate event opportunities by event for territory overview."""
    if event_store_map.empty:
        return pd.DataFrame()
    return (
        event_store_map.groupby(["event_id","event_name","event_type","event_date","expected_attendance"])
        .agg(
            nearby_stores        =("store_id", "nunique"),
            total_revenue_uplift =("est_revenue_uplift", "sum"),
            territories_affected =("territory_id", "nunique"),
        )
        .reset_index()
        .sort_values("expected_attendance", ascending=False)
        .round(2)
    )

