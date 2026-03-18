"""
Generate causally consistent transactional tables:
  pos_sales, retail_execution, promotions, local_events
Causal chain:
  base_demand × seasonality × promo_lift × compliance_factor × stock_factor × event_boost
  = realized_units_sold
"""
import numpy as np
import pandas as pd
from config.settings import RANDOM_SEED, WEEK_DATES, EVENT_RADIUS_KM
from geopy.distance import geodesic

rng = np.random.default_rng(RANDOM_SEED)

# ── Helpers ───────────────────────────────────────────────────────────────────
def _seasonality_factor(week_date: pd.Timestamp, profile: str) -> float:
    """NZ-aware seasonality: summer = Dec-Feb, event = year-round variance."""
    month = week_date.month
    if profile == "summer":
        return 1.25 if month in (12, 1, 2) else (0.85 if month in (6, 7, 8) else 1.0)
    elif profile == "event":
        return float(rng.uniform(0.9, 1.1))
    return 1.0  # "even"


# ── Local Events ─────────────────────────────────────────────────────────────
NZ_EVENTS_RAW = [
    {"event_name":"ASB Classic Tennis Auckland",   "event_type":"sports",  "city_lat":-36.848,"city_lon":174.763,"date_offset": 2,"attendance":18000},
    {"event_name":"Auckland Marathon",             "event_type":"sports",  "city_lat":-36.848,"city_lon":174.763,"date_offset": 5,"attendance":15000},
    {"event_name":"Homegrown Festival Wellington", "event_type":"concert", "city_lat":-41.286,"city_lon":174.776,"date_offset": 3,"attendance":20000},
    {"event_name":"Wellington Sevens",             "event_type":"sports",  "city_lat":-41.286,"city_lon":174.776,"date_offset": 8,"attendance":35000},
    {"event_name":"Hamilton Garden Arts Festival", "event_type":"festival","city_lat":-37.787,"city_lon":175.279,"date_offset": 4,"attendance":12000},
    {"event_name":"NRL Warriors Home Game AKL",    "event_type":"sports",  "city_lat":-36.900,"city_lon":174.730,"date_offset": 6,"attendance":25000},
    {"event_name":"Tauranga Summer Festival",      "event_type":"festival","city_lat":-37.688,"city_lon":176.165,"date_offset": 7,"attendance":8000},
    {"event_name":"Christchurch Food & Wine Fest", "event_type":"festival","city_lat":-43.532,"city_lon":172.636,"date_offset": 9,"attendance":10000},
    {"event_name":"Queenstown Winter Festival",    "event_type":"festival","city_lat":-45.031,"city_lon":168.663,"date_offset":10,"attendance":45000},
    {"event_name":"Dunedin Highlanders Rugby",     "event_type":"sports",  "city_lat":-45.879,"city_lon":170.503,"date_offset":11,"attendance":20000},
    {"event_name":"ASB Bank NZ Open Golf",         "event_type":"sports",  "city_lat":-36.780,"city_lon":174.755,"date_offset":12,"attendance":22000},
    {"event_name":"Napier Art Deco Festival",      "event_type":"festival","city_lat":-39.493,"city_lon":176.912,"date_offset": 1,"attendance":50000},
    {"event_name":"Rotorua Marathon",              "event_type":"sports",  "city_lat":-38.137,"city_lon":176.250,"date_offset":13,"attendance":9000},
    {"event_name":"Palmerston North Rugby",        "event_type":"sports",  "city_lat":-40.352,"city_lon":175.608,"date_offset":14,"attendance":11000},
    {"event_name":"Nelson Jazz Festival",          "event_type":"concert", "city_lat":-41.271,"city_lon":173.284,"date_offset":15,"attendance":7000},
]

def generate_local_events() -> pd.DataFrame:
    rows = []
    base_date = WEEK_DATES[0]
    for i, ev in enumerate(NZ_EVENTS_RAW):
        event_date = base_date + pd.Timedelta(weeks=ev["date_offset"])
        rows.append({
            "event_id":           f"EV{i+1:03d}",
            "event_name":         ev["event_name"],
            "event_type":         ev["event_type"],
            "latitude":           ev["city_lat"] + float(rng.uniform(-0.01, 0.01)),
            "longitude":          ev["city_lon"] + float(rng.uniform(-0.01, 0.01)),
            "event_date":         event_date.date(),
            "expected_attendance":ev["attendance"],
        })
    return pd.DataFrame(rows)


# ── Retail Execution ──────────────────────────────────────────────────────────
def generate_retail_execution(stores: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    rows = []
    eid  = 1
    for _, store in stores.iterrows():
        for _, sku in products.iterrows():
            # Compliance drawn per store-SKU pair; some stores are chronic under-performers
            compliance_rate = float(rng.beta(6, 2))           # mostly compliant
            if rng.random() < 0.15:                            # 15% stores have issues
                compliance_rate = float(rng.beta(2, 5))
            exp_facings = int(rng.choice([4, 6, 8]))
            act_facings = max(0, int(round(exp_facings * compliance_rate + rng.normal(0, 0.5))))
            display     = bool(rng.random() < compliance_rate)
            stock_ok    = rng.random() < (0.95 if compliance_rate > 0.6 else 0.70)
            rows.append({
                "execution_id":    eid,
                "store_id":        store.store_id,
                "sku_id":          sku.sku_id,
                "audit_date":      WEEK_DATES[-1].date(),
                "facings_actual":  act_facings,
                "facings_expected":exp_facings,
                "display_present": display,
                "stock_status":    "in_stock" if stock_ok else "out_of_stock",
                "compliance_rate": round(compliance_rate, 3),
            })
            eid += 1
    return pd.DataFrame(rows)


# ── Promotions ────────────────────────────────────────────────────────────────
def generate_promotions(stores: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pid = 1
    promo_types = ["price_reduction", "display_promotion", "feature_ad", "multipack_deal"]
    for _, store in stores.iterrows():
        for _, sku in products.iterrows():
            # ~30% chance a promo exists during the 16-week window
            if rng.random() < 0.30:
                start_week_idx = int(rng.integers(0, len(WEEK_DATES) - 3))
                duration_weeks = int(rng.integers(2, 5))
                end_week_idx   = min(start_week_idx + duration_weeks, len(WEEK_DATES) - 1)
                promo_type     = str(rng.choice(promo_types))
                promo_price    = round(float(sku.base_price) * float(rng.uniform(0.75, 0.92)), 2)
                rows.append({
                    "promo_id":    pid,
                    "store_id":    store.store_id,
                    "sku_id":      sku.sku_id,
                    "promo_type":  promo_type,
                    "promo_start": WEEK_DATES[start_week_idx].date(),
                    "promo_end":   WEEK_DATES[end_week_idx].date(),
                    "promo_price": promo_price,
                })
                pid += 1
    return pd.DataFrame(rows)


# ── POS Sales ─────────────────────────────────────────────────────────────────
def generate_pos_sales(
    stores:     pd.DataFrame,
    products:   pd.DataFrame,
    execution:  pd.DataFrame,
    promotions: pd.DataFrame,
    events:     pd.DataFrame,
    demographics: pd.DataFrame,
) -> pd.DataFrame:

    exec_idx  = execution.set_index(["store_id", "sku_id"])
    promo_list= promotions.to_dict("records")
    demo_idx  = demographics.set_index("region")
    rows      = []
    sales_id  = 1

    for _, store in stores.iterrows():
        demo = demo_idx.loc[store.region] if store.region in demo_idx.index else demo_idx.iloc[0]
        income_factor = float(demo.median_income) / 65000.0     # normalised around NZ median

        for _, sku in products.iterrows():
            # Execution lookup
            try:
                ex = exec_idx.loc[(store.store_id, sku.sku_id)]
                comp_rate  = float(ex.compliance_rate)
                in_stock   = ex.stock_status == "in_stock"
                display    = bool(ex.display_present)
            except KeyError:
                comp_rate, in_stock, display = 0.8, True, False

            # Base weekly units for this store-SKU
            base_units = (
                float(sku.base_price) ** -0.6             # price elasticity
                * float(store.base_demand_multiplier)
                * income_factor
                * 12                                       # weekly gross
            )

            for week_date in WEEK_DATES:
                seas = _seasonality_factor(week_date, sku.seasonality)

                # Promo flag
                promo_active  = any(
                    p["store_id"] == store.store_id and p["sku_id"] == sku.sku_id
                    and pd.Timestamp(p["promo_start"]) <= week_date <= pd.Timestamp(p["promo_end"])
                    for p in promo_list
                )
                promo_mult = float(sku.promo_sensitivity) if promo_active else 1.0

                # Shelf compliance impact
                shelf_mult  = 0.60 + 0.40 * comp_rate
                display_mult= 1.20 if display else 1.0
                stock_mult  = 1.0  if in_stock else 0.0

                # Event proximity boost
                week_ts = week_date.date()
                event_mult = 1.0
                for _, ev in events.iterrows():
                    dist_km = geodesic(
                        (store.latitude, store.longitude),
                        (ev.latitude,   ev.longitude)
                    ).km
                    days_diff = abs((pd.Timestamp(ev.event_date) - week_date).days)
                    if dist_km <= EVENT_RADIUS_KM and days_diff <= 7:
                        boost = 1.0 + min(0.40, ev.expected_attendance / 100000)
                        event_mult = max(event_mult, boost)

                expected_units = base_units * seas * promo_mult * display_mult
                realized_units = expected_units * shelf_mult * stock_mult * event_mult
                noise          = float(rng.normal(1.0, 0.07))
                units_sold     = max(0, round(realized_units * noise))
                price_used     = float(sku.base_price)
                rows.append({
                    "sales_id":       sales_id,
                    "store_id":       store.store_id,
                    "sku_id":         sku.sku_id,
                    "week":           week_date.date(),
                    "units_sold":     units_sold,
                    "price":          round(price_used, 2),
                    "revenue":        round(units_sold * price_used, 2),
                    "promo_active":   promo_active,
                    "expected_units": round(expected_units, 2),
                })
                sales_id += 1

    return pd.DataFrame(rows)


# ── Entry-point ───────────────────────────────────────────────────────────────
def generate_all_transactional(ref: dict) -> dict:
    events    = generate_local_events()
    execution = generate_retail_execution(ref["stores"], ref["products"])
    promotions= generate_promotions(ref["stores"], ref["products"])
    pos_sales = generate_pos_sales(
        ref["stores"], ref["products"], execution, promotions, events, ref["demographics"]
    )
    return {
        "local_events":      events,
        "retail_execution":  execution,
        "promotions":        promotions,
        "pos_sales":         pos_sales,
    }

