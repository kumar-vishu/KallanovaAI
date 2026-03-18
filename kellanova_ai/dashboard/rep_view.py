"""
Sales Rep Dashboard View
"""
import streamlit as st
import plotly.express as px
import pandas as pd
from streamlit_folium import st_folium

from api.data_context import DataContext
from analytics.visit_planner import optimise_visit_route
from analytics.event_engine import get_rep_event_opportunities
from dashboard.map_utils import make_rep_route_map
from dashboard.utils import fmt_currency
from ai.review_generator import generate_rep_review


def render_rep(ctx: DataContext, rep_id: str, ollama_ok: bool):
    rep_row = ctx.rep_scores[ctx.rep_scores["rep_id"] == rep_id]
    if rep_row.empty:
        st.error(f"Rep {rep_id} not found")
        return
    rep = rep_row.iloc[0]

    terr_name = ""
    terr = ctx.territories[ctx.territories["territory_id"] == rep.territory_id]
    if not terr.empty:
        terr_name = terr.iloc[0]["territory_name"]

    st.markdown(f"## 👤 Rep Dashboard: {rep.rep_name}")
    st.caption(f"Territory: {terr_name}")

    # ── KPI Row ───────────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.metric("Stores Assigned",       int(rep.stores_managed))
    c2.metric("High Priority Today",   int(rep.high_priority_stores))
    c3.metric("Today's Opportunity",   fmt_currency(float(rep.total_opportunity_value)))

    st.markdown("---")

    # ── Priority stores + route map ───────────────────────────────────────────
    route = optimise_visit_route(rep_id, ctx.store_scores, ctx.events)
    col_route, col_map = st.columns([2, 3])

    with col_route:
        st.markdown("### 📋 Priority Store List")
        rep_stores = ctx.store_scores[ctx.store_scores["rep_id"] == rep_id].nlargest(10, "final_score")
        display_df = rep_stores[["store_name","city","total_opportunity_value","top_issue","event_factor"]].copy()
        display_df["total_opportunity_value"] = display_df["total_opportunity_value"].apply(lambda x: f"${x:,.0f}")
        display_df["event_factor"] = display_df["event_factor"].apply(lambda x: "🎯 Yes" if x > 1.0 else "—")
        display_df.insert(0, "Rank", range(1, len(display_df)+1))
        st.dataframe(
            display_df.rename(columns={
                "store_name":"Store","city":"City",
                "total_opportunity_value":"Opportunity","top_issue":"Issue","event_factor":"Event Nearby"
            }),
            use_container_width=True, hide_index=True
        )

    with col_map:
        st.markdown("### 🗺 Visit Route Map")
        ev_opps = get_rep_event_opportunities(rep_id, ctx.event_store_map)
        ev_nearby = ctx.events[ctx.events["event_id"].isin(ev_opps["event_id"].unique())]
        route_map = make_rep_route_map(route if not route.empty else pd.DataFrame(), ev_nearby)
        st_folium(route_map, width=600, height=380, returned_objects=[])

    st.markdown("---")
    col_events, col_tasks = st.columns(2)

    with col_events:
        st.markdown("### 🎯 Event Opportunities")
        ev_opps = get_rep_event_opportunities(rep_id, ctx.event_store_map)
        if not ev_opps.empty:
            show_ev = ev_opps[["event_name","event_date","store_name","distance_km",
                               "recommended_action","est_revenue_uplift"]].head(6)
            show_ev["est_revenue_uplift"] = show_ev["est_revenue_uplift"].apply(lambda x: f"${x:,.0f}")
            st.dataframe(show_ev.rename(columns={
                "event_name":"Event","event_date":"Date","store_name":"Store",
                "distance_km":"Km","recommended_action":"Action","est_revenue_uplift":"Est. Uplift"
            }), use_container_width=True, hide_index=True)
        else:
            st.info("No events near your stores this week.")

    with col_tasks:
        st.markdown("### ✅ Recommended Tasks")
        all_rep_opps = ctx.opportunities[
            ctx.opportunities["store_id"].isin(rep_stores["store_id"].tolist())
        ]
        task_counts = all_rep_opps.groupby("issue_type")["opportunity_value"].sum().nlargest(5)
        for issue_type, val in task_counts.items():
            rca = ctx.opportunities[ctx.opportunities["issue_type"]==issue_type]["rca_actions"].iloc[0] \
                  if "rca_actions" in ctx.opportunities.columns else ""
            action = rca.split(";")[0] if rca else f"Address {issue_type}"
            st.checkbox(f"**{issue_type.replace('_',' ').title()}** — {fmt_currency(val)}: {action}", value=False)

    # ── AI Review ─────────────────────────────────────────────────────────────
    st.markdown("### 🤖 AI Daily Briefing")
    if ollama_ok:
        if st.button("Generate AI Rep Briefing", type="primary"):
            with st.spinner("Generating …"):
                ev_list = ev_opps["event_name"].head(3).tolist() if not ev_opps.empty else []
                top = rep_stores.iloc[0] if not rep_stores.empty else pd.Series()
                review = generate_rep_review({
                    "rep_name":       rep.rep_name,
                    "territory_name": terr_name,
                    "stores_managed": int(rep.stores_managed),
                    "high_priority_stores": int(rep.high_priority_stores),
                    "total_opportunity_value": float(rep.total_opportunity_value),
                    "top_store_name": top.get("store_name","") if not top.empty else "",
                    "top_store_opp":  top.get("total_opportunity_value",0) if not top.empty else 0,
                    "key_issues":     ", ".join(task_counts.index.tolist()[:3]),
                    "events_this_week": ", ".join(ev_list) or "None",
                    "visit_route":    " → ".join(route["store_name"].tolist()[:5]) if not route.empty else "N/A",
                })
                st.markdown(review)
    else:
        st.info("Start Ollama to enable AI briefings: `ollama serve`")

