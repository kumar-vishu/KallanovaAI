"""
Root Cause Analysis Engine
Maps detected opportunity types to human-readable root cause explanations
and recommended actions.
"""
from __future__ import annotations
import pandas as pd

ROOT_CAUSE_MAP: dict[str, dict] = {
    "out_of_stock": {
        "category":    "Inventory",
        "explanation": "Product is out of stock, causing 100% revenue loss for this SKU. "
                       "This is typically caused by insufficient replenishment ordering, "
                       "delivery failures, or unexpected demand spikes.",
        "actions": [
            "Place emergency replenishment order immediately",
            "Agree increased standing order with store",
            "Check backroom for misplaced stock",
            "Review delivery schedule with supply chain",
        ],
        "severity": "Critical",
    },
    "distribution_gap": {
        "category":    "Distribution",
        "explanation": "Product is listed but recording zero or near-zero sales despite "
                       "stock being available. Likely a ranging or planogram issue — "
                       "product may not be on shelf or placed in wrong location.",
        "actions": [
            "Verify product is correctly ranged and on planogram",
            "Check product is at correct shelf location",
            "Discuss range extension or new listing with category manager",
            "Confirm EDI orders are set up correctly",
        ],
        "severity": "High",
    },
    "promotion_execution": {
        "category":    "Promotion",
        "explanation": "A promotion is scheduled or active but execution has failed. "
                       "Display material is missing, price tickets not applied, "
                       "or promotional stock not in position.",
        "actions": [
            "Install promotional display material immediately",
            "Apply correct price tickets and shelf talkers",
            "Move promotional stock to feature display location",
            "Confirm promotional compliance with store manager",
        ],
        "severity": "High",
    },
    "shelf_compliance": {
        "category":    "Shelf Execution",
        "explanation": "Actual shelf facings are significantly below planogram standard. "
                       "Reduced visibility lowers shopper pick-up rate and "
                       "results in below-expected sales for this SKU.",
        "actions": [
            "Rebuild section to planogram standard",
            "Negotiate additional facings with department manager",
            "Add clip strips or secondary placement",
            "Review range rationalisation risk with store",
        ],
        "severity": "Medium",
    },
    "event_opportunity": {
        "category":    "Event Demand",
        "explanation": "A high-attendance local event is occurring within range of this store. "
                       "Snack and beverage categories typically see 20–50% demand uplift "
                       "around major events. Current inventory and display may be insufficient.",
        "actions": [
            "Increase inventory of snack and savoury SKUs ahead of event",
            "Build event-themed display at store entrance",
            "Place Pringles dump bin near high-traffic area",
            "Brief store manager on expected demand increase",
        ],
        "severity": "Medium",
    },
    "hidden_opportunity": {
        "category":    "Demand",
        "explanation": "Store is significantly under-indexing vs predicted sales based on "
                       "its demographics, format, and regional peers. "
                       "There is untapped demand potential that is not being captured.",
        "actions": [
            "Conduct full range audit vs planogram",
            "Review display and feature ad compliance",
            "Compare performance vs similar stores in region",
            "Present growth opportunity data to store manager",
        ],
        "severity": "Medium",
    },
}

UNKNOWN_RCA = {
    "category":    "Unknown",
    "explanation": "No specific root cause has been identified. Further investigation required.",
    "actions":     ["Conduct store visit and audit", "Review sales data trend"],
    "severity":    "Low",
}


def get_root_cause(issue_type: str) -> dict:
    return ROOT_CAUSE_MAP.get(issue_type, UNKNOWN_RCA)


def enrich_opportunities_with_rca(opportunities: pd.DataFrame) -> pd.DataFrame:
    """Add root cause columns to the opportunities DataFrame."""
    df = opportunities.copy()
    df["rca_category"]    = df["issue_type"].map(lambda x: get_root_cause(x)["category"])
    df["rca_explanation"] = df["issue_type"].map(lambda x: get_root_cause(x)["explanation"])
    df["rca_severity"]    = df["issue_type"].map(lambda x: get_root_cause(x)["severity"])
    df["rca_actions"]     = df["issue_type"].map(lambda x: "; ".join(get_root_cause(x)["actions"]))
    return df


def store_root_cause_summary(store_id: int, opportunities: pd.DataFrame) -> list[dict]:
    """Return a list of root cause summaries for a single store."""
    store_opps = opportunities[opportunities["store_id"] == store_id]
    results = []
    for issue_type, grp in store_opps.groupby("issue_type"):
        rca = get_root_cause(issue_type)
        results.append({
            "issue_type":    issue_type,
            "sku_count":     len(grp),
            "total_opp":     round(grp["opportunity_value"].sum(), 2),
            "severity":      rca["severity"],
            "category":      rca["category"],
            "explanation":   rca["explanation"],
            "actions":       rca["actions"],
        })
    return sorted(results, key=lambda x: x["total_opp"], reverse=True)

