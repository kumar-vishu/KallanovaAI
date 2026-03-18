"""
Store Visit Optimiser
Ranks stores for each rep using a greedy nearest-neighbour heuristic
weighted by opportunity score, event timing, and visit recency.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from geopy.distance import geodesic
from config.settings import EVENT_RADIUS_KM


def _haversine(lat1, lon1, lat2, lon2) -> float:
    """Fast haversine distance in km."""
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def optimise_visit_route(
    rep_id:      str,
    store_scores: pd.DataFrame,
    events:       pd.DataFrame,
    today:        pd.Timestamp | None = None,
    max_stores:   int = 10,
) -> pd.DataFrame:
    """
    Returns an ordered visit route for a rep, balancing:
      - opportunity score (higher = visit sooner)
      - distance (shorter hops = visit sooner)
      - proximity to upcoming events (within 7 days)
    """
    today = today or pd.Timestamp.today().normalize()

    rep_stores = store_scores[store_scores["rep_id"] == rep_id].copy()
    if rep_stores.empty:
        return pd.DataFrame()

    # Event proximity within 7 days
    def near_upcoming_event(row):
        for _, ev in events.iterrows():
            days_away = (pd.Timestamp(ev.event_date) - today).days
            if 0 <= days_away <= 7:
                dist = _haversine(row.latitude, row.longitude, ev.latitude, ev.longitude)
                if dist <= EVENT_RADIUS_KM:
                    return True
        return False

    rep_stores["has_event_this_week"] = rep_stores.apply(near_upcoming_event, axis=1)
    rep_stores["event_bonus"] = rep_stores["has_event_this_week"].astype(float) * 50

    # Greedy nearest-neighbour from store with highest priority
    rep_stores = rep_stores.reset_index(drop=True)
    rep_stores["composite_score"] = rep_stores["final_score"] + rep_stores["event_bonus"]

    visited  = []
    remaining = rep_stores.copy()

    # Start at highest score store
    current = remaining.loc[remaining["composite_score"].idxmax()]
    visited.append(current)
    remaining = remaining.drop(current.name)

    while len(visited) < min(max_stores, len(rep_stores)) and not remaining.empty:
        cur_lat, cur_lon = current.latitude, current.longitude
        remaining = remaining.copy()
        remaining["dist_km"] = remaining.apply(
            lambda r: _haversine(cur_lat, cur_lon, r.latitude, r.longitude), axis=1
        )
        # Score = opportunity / (distance + 5) to balance
        remaining["route_score"] = (
            remaining["composite_score"] / (remaining["dist_km"] + 5)
        )
        nxt = remaining.loc[remaining["route_score"].idxmax()]
        visited.append(nxt)
        remaining = remaining.drop(nxt.name)
        current = nxt

    route = pd.DataFrame(visited).reset_index(drop=True)
    route["visit_order"] = route.index + 1
    route["visit_date"]  = today + pd.to_timedelta(route.index.map(lambda i: i // 3), unit="D")

    cols = ["visit_order","store_id","store_name","chain","city",
            "final_score","total_opportunity_value","has_event_this_week",
            "visit_date","latitude","longitude"]
    return route[[c for c in cols if c in route.columns]]


def optimise_all_reps(
    store_scores: pd.DataFrame,
    reps:          pd.DataFrame,
    events:        pd.DataFrame,
    today:         pd.Timestamp | None = None,
) -> dict[str, pd.DataFrame]:
    """Return visit routes for all reps as a dict keyed by rep_id."""
    routes = {}
    for _, rep in reps.iterrows():
        routes[rep.rep_id] = optimise_visit_route(
            rep.rep_id, store_scores, events, today
        )
    return routes

