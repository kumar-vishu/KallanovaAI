"""
Generate derived / analytics tables from ref + transactional data:
  store_features, store_opportunities, hidden_opportunities,
  visit_plan, case_library
"""
import numpy as np
import pandas as pd
from config.settings import RANDOM_SEED, SEVERITY_WEIGHTS, EVENT_RADIUS_KM, WEEK_DATES
from geopy.distance import geodesic

rng = np.random.default_rng(RANDOM_SEED)


# ── 1. Store Features ─────────────────────────────────────────────────────────
def generate_store_features(
    pos_sales:  pd.DataFrame,
    execution:  pd.DataFrame,
    promotions: pd.DataFrame,
    events:     pd.DataFrame,
    stores:     pd.DataFrame,
) -> pd.DataFrame:

    # sales velocity = avg weekly units over last 4 weeks
    recent_weeks = sorted(pos_sales["week"].unique())[-4:]
    recent_sales = pos_sales[pos_sales["week"].isin(recent_weeks)]
    velocity = (
        recent_sales.groupby(["store_id", "sku_id"])["units_sold"]
        .mean().rename("sales_velocity").reset_index()
    )

    # sales trend = slope (last 8 weeks)
    trend_weeks = sorted(pos_sales["week"].unique())[-8:]
    trend_sales = pos_sales[pos_sales["week"].isin(trend_weeks)].copy()
    trend_sales["week_num"] = trend_sales["week"].apply(lambda d: trend_weeks.index(d))

    def calc_trend(grp):
        if len(grp) < 3:
            return 0.0
        x = grp["week_num"].values.astype(float)
        y = grp["units_sold"].values.astype(float)
        if x.std() == 0:
            return 0.0
        return float(np.polyfit(x, y, 1)[0])

    trend = (
        trend_sales.groupby(["store_id", "sku_id"])
        .apply(calc_trend, include_groups=False)
        .rename("sales_trend").reset_index()
    )

    # shelf compliance score
    compliance = execution[["store_id", "sku_id", "compliance_rate", "display_present", "stock_status"]].copy()
    compliance["shelf_compliance_score"] = compliance["compliance_rate"]
    compliance["out_of_stock_flag"]      = compliance["stock_status"] == "out_of_stock"

    # promo execution flag
    active_promos = set(
        zip(promotions["store_id"], promotions["sku_id"])
    )
    compliance["promo_execution_flag"] = compliance.apply(
        lambda r: (r.store_id, r.sku_id) in active_promos, axis=1
    )

    # distance to nearest event (km)
    store_loc = stores.set_index("store_id")[["latitude", "longitude"]]

    def nearest_event_dist(store_id):
        loc = store_loc.loc[store_id]
        dists = events.apply(
            lambda ev: geodesic((loc.latitude, loc.longitude), (ev.latitude, ev.longitude)).km,
            axis=1
        )
        return round(float(dists.min()), 2)

    dist_map = {sid: nearest_event_dist(sid) for sid in stores["store_id"]}
    compliance["distance_to_event"] = compliance["store_id"].map(dist_map)

    # merge all
    feats = velocity.merge(trend, on=["store_id", "sku_id"], how="left")
    feats = feats.merge(
        compliance[["store_id","sku_id","shelf_compliance_score",
                    "out_of_stock_flag","promo_execution_flag","distance_to_event"]],
        on=["store_id","sku_id"], how="left"
    )
    return feats.fillna(0)


# ── 2. Store Opportunities ────────────────────────────────────────────────────
def generate_store_opportunities(
    pos_sales:  pd.DataFrame,
    execution:  pd.DataFrame,
    events:     pd.DataFrame,
    stores:     pd.DataFrame,
) -> pd.DataFrame:
    rows   = []
    opp_id = 1
    today  = WEEK_DATES[-1].date()
    exec_idx = execution.set_index(["store_id","sku_id"])
    store_loc = stores.set_index("store_id")[["latitude","longitude","territory_id","rep_id"]]

    # Latest actuals per store-sku
    latest_week = pos_sales["week"].max()
    latest = pos_sales[pos_sales["week"]==latest_week].set_index(["store_id","sku_id"])

    for (store_id, sku_id), row in latest.iterrows():
        actual   = float(row["units_sold"])
        expected = float(row["expected_units"])
        gap      = expected - actual
        price    = float(row["price"])

        try:
            ex = exec_idx.loc[(store_id, sku_id)]
        except KeyError:
            continue

        issues = []
        if ex.stock_status == "out_of_stock":
            issues.append(("out_of_stock", "product_oos", 1.0))
        if ex.compliance_rate < 0.6:
            issues.append(("shelf_compliance", "low_shelf_facings", ex.compliance_rate))
        if not ex.display_present:
            issues.append(("promotion_execution", "missing_display", 0.85))
        if actual == 0 and ex.stock_status == "in_stock":
            issues.append(("distribution_gap", "no_sales_in_stock", 0.9))

        # Event proximity
        loc = store_loc.loc[store_id]
        for _, ev in events.iterrows():
            dist_km = geodesic((loc.latitude, loc.longitude), (ev.latitude, ev.longitude)).km
            if dist_km <= EVENT_RADIUS_KM:
                issues.append(("event_opportunity", f"near_{ev.event_name[:20]}", 0.7))
                break

        for issue_type, root_cause, severity in issues:
            opp_value    = round(max(0, gap) * price, 2)
            priority     = round(opp_value * SEVERITY_WEIGHTS.get(issue_type, 0.7), 2)
            rows.append({
                "opportunity_id":  opp_id,
                "store_id":        store_id,
                "sku_id":          sku_id,
                "issue_type":      issue_type,
                "root_cause":      root_cause,
                "opportunity_value": opp_value,
                "priority_score":  priority,
                "detected_date":   today,
            })
            opp_id += 1

    return pd.DataFrame(rows)


# ── 3. Hidden Opportunities ───────────────────────────────────────────────────
def generate_hidden_opportunities(pos_sales: pd.DataFrame, stores: pd.DataFrame) -> pd.DataFrame:
    store_weekly = (
        pos_sales.groupby(["store_id","week"])["revenue"].sum().reset_index()
    )
    store_avg = store_weekly.groupby("store_id")["revenue"].mean().reset_index()
    store_avg.columns = ["store_id","actual_sales"]
    store_avg = store_avg.merge(stores[["store_id","base_demand_multiplier","region"]], on="store_id")

    # Predicted from regional avg × format multiplier
    regional_avg = store_avg.groupby("region")["actual_sales"].transform("mean")
    store_avg["predicted_sales"]     = regional_avg * store_avg["base_demand_multiplier"] * 1.15
    store_avg["hidden_opportunity"]  = (store_avg["predicted_sales"] - store_avg["actual_sales"]).clip(lower=0)
    store_avg["model_version"]       = "v1.0-linear"
    return store_avg[["store_id","predicted_sales","actual_sales","hidden_opportunity","model_version"]].round(2)


# ── 4. Visit Plan ─────────────────────────────────────────────────────────────
def generate_visit_plan(
    opportunities: pd.DataFrame,
    stores:        pd.DataFrame,
    reps:          pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    vid  = 1
    today = pd.Timestamp(WEEK_DATES[-1].date())

    rep_store = stores[["store_id","rep_id","latitude","longitude"]].copy()
    opp_summary = (
        opportunities.groupby("store_id")["priority_score"].sum()
        .rename("total_priority").reset_index()
    )
    rep_store = rep_store.merge(opp_summary, on="store_id", how="left").fillna(0)

    for _, rep in reps.iterrows():
        rep_stores = rep_store[rep_store["rep_id"]==rep.rep_id].sort_values(
            "total_priority", ascending=False
        ).reset_index(drop=True)
        for rank, (_, st) in enumerate(rep_stores.iterrows(), 1):
            visit_date = today + pd.Timedelta(days=int(rank % 5))
            rows.append({
                "visit_id":       vid,
                "rep_id":         rep.rep_id,
                "store_id":       int(st.store_id),
                "visit_priority": rank,
                "visit_date":     visit_date.date(),
                "reason":         "high opportunity score" if st.total_priority > 50 else "routine visit",
                "total_priority": round(float(st.total_priority), 2),
            })
            vid += 1
    return pd.DataFrame(rows)


# ── 5. Case Library ───────────────────────────────────────────────────────────
CASE_TEMPLATES = [
    {"issue_type":"shelf_compliance",     "description":"Low facings for cereal category reducing visibility on shelf","action_taken":"Negotiated extra shelf space with store manager, relocated to eye level","result":"Sales increased 22% within 3 weeks"},
    {"issue_type":"promotion_execution",  "description":"Display promotion not executed at store - missing endcap","action_taken":"Installed endcap display with POS material and price ticket","result":"Promotional sales lift of 35% observed during activation period"},
    {"issue_type":"out_of_stock",         "description":"Pringles range OOS for 2 consecutive weeks","action_taken":"Emergency replenishment arranged, buffer stock agreed with store","result":"Recovered $1,800 weekly revenue, OOS resolved"},
    {"issue_type":"distribution_gap",     "description":"New Nutri-Grain Bar variant not listed in PAK'nSAVE","action_taken":"Presented range review data to category manager, secured listing","result":"New SKU generating $500/week incremental revenue"},
    {"issue_type":"event_opportunity",    "description":"Rugby match 8km away - no snack display or event stock","action_taken":"Built event display, increased Pringles facing, added dump bin","result":"Weekend sales +45%, sold through all event stock"},
    {"issue_type":"hidden_opportunity",   "description":"Store under-indexing vs regional peers by 40% on cereal","action_taken":"Full range audit, fixed planogram, added Special K promotional display","result":"Sales grew 28% over 6 weeks to match regional benchmark"},
    {"issue_type":"shelf_compliance",     "description":"Snack bar section underfaced vs planogram","action_taken":"Rebuilt section to planogram standard, added clip strip","result":"Category sales increased 18%"},
    {"issue_type":"promotion_execution",  "description":"Feature ad running but no price ticket in store","action_taken":"Installed price tickets and shelf talkers on day of visit","result":"Promo redemption rate improved from 12% to 41%"},
    {"issue_type":"out_of_stock",         "description":"Coco Pops OOS during school holidays peak period","action_taken":"Worked with store to increase order quantity and delivery frequency","result":"Prevented estimated $2,200 lost revenue"},
    {"issue_type":"event_opportunity",    "description":"Music festival nearby with 20K attendance - no activation","action_taken":"Placed Pringles dump bin at store entrance with festival-themed POS","result":"Weekend snack sales +60%, clear ROI vs display cost"},
]

def generate_case_library() -> pd.DataFrame:
    rows = []
    for i, case in enumerate(CASE_TEMPLATES, 1):
        rows.append({"case_id": f"CASE{i:03d}", **case, "embedding_vector": None})
    return pd.DataFrame(rows)


# ── Entry-point ───────────────────────────────────────────────────────────────
def generate_all_derived(ref: dict, trans: dict) -> dict:
    features = generate_store_features(
        trans["pos_sales"], trans["retail_execution"],
        trans["promotions"], trans["local_events"], ref["stores"]
    )
    opportunities = generate_store_opportunities(
        trans["pos_sales"], trans["retail_execution"],
        trans["local_events"], ref["stores"]
    )
    hidden = generate_hidden_opportunities(trans["pos_sales"], ref["stores"])
    visit_plan = generate_visit_plan(opportunities, ref["stores"], ref["sales_reps"])
    cases = generate_case_library()
    return {
        "store_features":      features,
        "store_opportunities": opportunities,
        "hidden_opportunities":hidden,
        "visit_plan":          visit_plan,
        "case_library":        cases,
    }

