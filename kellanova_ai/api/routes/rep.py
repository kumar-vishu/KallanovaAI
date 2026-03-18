"""
Sales Rep dashboard API routes.
"""
from fastapi import APIRouter, HTTPException
import pandas as pd
from api.data_context import DataContext
from analytics.visit_planner import optimise_visit_route
from analytics.event_engine import get_rep_event_opportunities
from ai.review_generator import generate_rep_review

router = APIRouter(prefix="/rep-dashboard", tags=["Rep"])


@router.get("/{rep_id}")
def rep_dashboard(rep_id: str, include_ai_review: bool = False):
    ctx = DataContext.get()

    rep_row = ctx.rep_scores[ctx.rep_scores["rep_id"] == rep_id]
    if rep_row.empty:
        raise HTTPException(404, f"Rep {rep_id} not found")
    rep = rep_row.iloc[0].to_dict()

    # Territory name
    terr_name = ""
    terr = ctx.territories[ctx.territories["territory_id"] == rep.get("territory_id")]
    if not terr.empty:
        terr_name = terr.iloc[0]["territory_name"]

    # Priority stores
    rep_stores = ctx.store_scores[ctx.store_scores["rep_id"] == rep_id].nlargest(10, "final_score")
    priority_stores = rep_stores[[
        "store_id","store_name","chain","city","final_score",
        "total_opportunity_value","top_issue","hidden_opportunity"
    ]].to_dict("records")

    # Optimised visit route
    route = optimise_visit_route(rep_id, ctx.store_scores, ctx.events)
    visit_route = route.to_dict("records") if not route.empty else []

    # Event opportunities
    ev_opps = get_rep_event_opportunities(rep_id, ctx.event_store_map)
    events  = ev_opps[["event_name","event_type","event_date","store_name",
                        "distance_km","recommended_action","est_revenue_uplift"]].to_dict("records")

    # Recommended tasks (aggregate top issues across all rep stores)
    all_rep_opps = ctx.opportunities[
        ctx.opportunities["store_id"].isin(rep_stores["store_id"].tolist())
    ]
    task_counts = all_rep_opps.groupby("issue_type")["opportunity_value"].sum().nlargest(5)
    tasks = [
        {"task": ctx.root_cause_label(issue_type), "opportunity_value": round(float(val),2)}
        if hasattr(ctx, "root_cause_label")
        else {"issue_type": issue_type, "total_value": round(float(val),2)}
        for issue_type, val in task_counts.items()
    ]

    payload = {
        "rep_id":    rep_id,
        "rep_name":  rep.get("rep_name"),
        "territory": terr_name,
        "snapshot": {
            "stores_assigned":         int(rep.get("stores_managed", 0)),
            "high_priority_stores":    int(rep.get("high_priority_stores", 0)),
            "total_opportunity_value": round(float(rep.get("total_opportunity_value", 0)), 2),
        },
        "priority_stores":   priority_stores,
        "visit_route":       visit_route,
        "event_opportunities": events,
        "recommended_tasks": tasks,
    }

    if include_ai_review:
        top = priority_stores[0] if priority_stores else {}
        events_str = ", ".join([e["event_name"] for e in events[:3]]) or "None"
        route_str  = " → ".join([r.get("store_name","") for r in visit_route[:5]])
        payload["ai_review"] = generate_rep_review({
            **payload["snapshot"],
            "rep_name":      rep.get("rep_name"),
            "territory_name":terr_name,
            "top_store_name":top.get("store_name",""),
            "top_store_opp": top.get("total_opportunity_value", 0),
            "key_issues":    ", ".join([t.get("issue_type","") for t in tasks[:3]]),
            "events_this_week": events_str,
            "visit_route":   route_str,
        })

    return payload

