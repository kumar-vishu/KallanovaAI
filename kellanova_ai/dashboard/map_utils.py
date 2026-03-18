"""
GeoPandas / Folium map utilities for the Kellanova NZ dashboard.
"""
from __future__ import annotations
import folium
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

ISSUE_COLORS = {
    "out_of_stock":         "red",
    "distribution_gap":     "orange",
    "promotion_execution":  "purple",
    "shelf_compliance":     "blue",
    "event_opportunity":    "green",
    "hidden_opportunity":   "darkblue",
}

def _make_base_map(lat: float = -41.0, lon: float = 174.0, zoom: int = 5) -> folium.Map:
    return folium.Map(
        location=[lat, lon],
        zoom_start=zoom,
        tiles="CartoDB positron",
    )


def make_territory_map(
    stores:       pd.DataFrame,
    store_scores: pd.DataFrame,
    events:       pd.DataFrame,
    territory_id: str | None = None,
) -> folium.Map:
    m = _make_base_map()

    df = stores.merge(
        store_scores[["store_id","final_score","total_opportunity_value","top_issue"]],
        on="store_id", how="left"
    )
    if territory_id:
        df = df[df["territory_id"] == territory_id]

    for _, row in df.iterrows():
        color = ISSUE_COLORS.get(str(row.get("top_issue","")), "gray")
        opp   = row.get("total_opportunity_value", 0)
        radius = max(6, min(20, float(opp) / 80 + 6)) if opp > 0 else 6

        folium.CircleMarker(
            location=[row.latitude, row.longitude],
            radius=radius,
            color=color,
            fill=True,
            fill_opacity=0.7,
            tooltip=folium.Tooltip(
                f"<b>{row.store_name}</b><br>"
                f"Chain: {row.chain}<br>"
                f"Opportunity: ${opp:,.0f}<br>"
                f"Top Issue: {row.get('top_issue','—')}"
            ),
        ).add_to(m)

    # Event markers
    for _, ev in events.iterrows():
        folium.Marker(
            location=[ev.latitude, ev.longitude],
            icon=folium.Icon(color="green", icon="star", prefix="fa"),
            tooltip=f"🎯 {ev.event_name} ({ev.expected_attendance:,} attendees)",
        ).add_to(m)

    return m


def make_rep_route_map(
    route:  pd.DataFrame,
    events: pd.DataFrame,
) -> folium.Map:
    if route.empty:
        return _make_base_map()
    center_lat = route["latitude"].mean()
    center_lon = route["longitude"].mean()
    m = _make_base_map(center_lat, center_lon, zoom=8)

    # Route line
    coords = list(zip(route["latitude"], route["longitude"]))
    folium.PolyLine(coords, color="#D52B1E", weight=2.5, opacity=0.7).add_to(m)

    for _, row in route.iterrows():
        folium.Marker(
            location=[row.latitude, row.longitude],
            icon=folium.DivIcon(
                html=f'<div style="background:#D52B1E;color:white;border-radius:50%;'
                     f'width:24px;height:24px;text-align:center;line-height:24px;'
                     f'font-weight:bold;font-size:12px;">{int(row.visit_order)}</div>',
                icon_size=(24, 24),
                icon_anchor=(12, 12),
            ),
            tooltip=f"#{int(row.visit_order)} {row.store_name} — ${row.get('total_opportunity_value',0):,.0f}",
        ).add_to(m)

    for _, ev in events.iterrows():
        folium.Marker(
            location=[ev.latitude, ev.longitude],
            icon=folium.Icon(color="green", icon="star", prefix="fa"),
            tooltip=f"🎯 {ev.event_name}",
        ).add_to(m)

    return m


def make_store_map(
    store:  pd.Series,
    events: pd.DataFrame,
) -> folium.Map:
    m = _make_base_map(float(store.latitude), float(store.longitude), zoom=12)
    folium.Marker(
        location=[store.latitude, store.longitude],
        icon=folium.Icon(color="red", icon="shopping-cart", prefix="fa"),
        tooltip=f"<b>{store.store_name}</b>",
    ).add_to(m)

    for _, ev in events.iterrows():
        lat = ev.get("event_latitude", ev.get("latitude", None))
        lon = ev.get("event_longitude", ev.get("longitude", None))
        if lat is None or lon is None:
            continue
        folium.Marker(
            location=[lat, lon],
            icon=folium.Icon(color="green", icon="star", prefix="fa"),
            tooltip=f"🎯 {ev.event_name} ({ev.distance_km:.1f} km away)",
        ).add_to(m)

    return m

