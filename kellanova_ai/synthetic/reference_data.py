"""
Generate reference/master tables:
  territories, sales_reps, stores, products, demographics
"""
import random
import numpy as np
import pandas as pd
from config.settings import (
    TERRITORIES, NZ_CITIES, CHAINS, STORE_FORMATS, PRODUCTS, RANDOM_SEED, NUM_STORES
)

rng = np.random.default_rng(RANDOM_SEED)
random.seed(RANDOM_SEED)

# ── 1. Territories ───────────────────────────────────────────────────────────
def generate_territories() -> pd.DataFrame:
    return pd.DataFrame(TERRITORIES)


# ── 2. Sales Reps ────────────────────────────────────────────────────────────
REP_NAMES = [
    ("R01", "Sarah Mitchell",  "T01"),
    ("R02", "James Tane",      "T01"),
    ("R03", "Emma Wilson",     "T02"),
    ("R04", "Tama Ngata",      "T02"),
    ("R05", "Lucy Chen",       "T03"),
    ("R06", "Michael Brown",   "T03"),
    ("R07", "Hannah Scott",    "T04"),
    ("R08", "Ravi Patel",      "T04"),
    ("R09", "Aroha Williams",  "T05"),
    ("R10", "Daniel Sione",    "T05"),
]

def generate_reps() -> pd.DataFrame:
    rows = []
    for rep_id, rep_name, territory_id in REP_NAMES:
        rows.append({
            "rep_id":       rep_id,
            "rep_name":     rep_name,
            "territory_id": territory_id,
            "email":        rep_name.lower().replace(" ", ".") + "@kellanova.com",
        })
    return pd.DataFrame(rows)


# ── 3. Stores ────────────────────────────────────────────────────────────────
# Assign reps to territories
TERRITORY_REPS = {}
for rep_id, rep_name, territory_id in REP_NAMES:
    TERRITORY_REPS.setdefault(territory_id, []).append(rep_id)

STORE_NAMES_BY_CHAIN = {
    "Woolworths NZ": ["Woolworths {city}", "Woolworths {city} Metro"],
    "New World":     ["New World {city}", "New World {city} Central"],
    "PAK'nSAVE":     ["PAK'nSAVE {city}"],
    "FreshChoice":   ["FreshChoice {city}"],
    "Four Square":   ["Four Square {city}", "Four Square {city} Local"],
    "SuperValue":    ["SuperValue {city}"],
}

def generate_stores() -> pd.DataFrame:
    stores = []
    store_id = 1001
    city_pool = NZ_CITIES.copy()
    # Weight cities so Auckland gets ~14 stores, others proportionally
    weights = []
    for c in city_pool:
        if "Auckland" in c["city"]: weights.append(4)
        elif c["territory_id"] in ("T02","T03"): weights.append(3)
        else: weights.append(2)
    total_w = sum(weights)
    probs   = [w/total_w for w in weights]

    for i in range(NUM_STORES):
        city_idx = rng.choice(len(city_pool), p=probs)
        city     = city_pool[city_idx]
        chain    = rng.choice(CHAINS)
        name_tpl = rng.choice(STORE_NAMES_BY_CHAIN[chain])
        store_name = name_tpl.replace("{city}", city["city"])

        # Small jitter on lat/lon for distinct pins
        lat = city["lat"] + rng.uniform(-0.02, 0.02)
        lon = city["lon"] + rng.uniform(-0.02, 0.02)

        territory_id = city["territory_id"]
        rep_id       = rng.choice(TERRITORY_REPS[territory_id])

        # Demand multiplier drives base sales
        format_mul = 1.5 if STORE_FORMATS[chain] == "Supermarket" else 0.7
        base_demand_multiplier = float(rng.uniform(0.7, 1.4) * format_mul)

        stores.append({
            "store_id":               store_id,
            "store_name":             store_name,
            "chain":                  chain,
            "store_format":           STORE_FORMATS[chain],
            "city":                   city["city"],
            "region":                 city["region"],
            "latitude":               round(lat, 6),
            "longitude":              round(lon, 6),
            "rep_id":                 rep_id,
            "territory_id":           territory_id,
            "base_demand_multiplier": round(base_demand_multiplier, 3),
        })
        store_id += 1
    return pd.DataFrame(stores)


# ── 4. Products ──────────────────────────────────────────────────────────────
def generate_products() -> pd.DataFrame:
    return pd.DataFrame(PRODUCTS)


# ── 5. Demographics ──────────────────────────────────────────────────────────
DEMO_DATA = {
    "Auckland":          {"population_density": 3000, "median_income": 68000, "household_size": 2.8, "ethnicity_index": 1.3},
    "Waikato/BOP":       {"population_density": 800,  "median_income": 56000, "household_size": 2.9, "ethnicity_index": 1.1},
    "Wellington/Manawatu":{"population_density":1200, "median_income": 72000, "household_size": 2.6, "ethnicity_index": 1.2},
    "Canterbury/Nelson": {"population_density": 600,  "median_income": 62000, "household_size": 2.7, "ethnicity_index": 0.95},
    "Otago/Southland":   {"population_density": 200,  "median_income": 55000, "household_size": 2.5, "ethnicity_index": 0.85},
}

def generate_demographics() -> pd.DataFrame:
    rows = []
    for region, vals in DEMO_DATA.items():
        rows.append({"region": region, **vals})
    return pd.DataFrame(rows)


# ── Entry-point ───────────────────────────────────────────────────────────────
def generate_all_reference() -> dict[str, pd.DataFrame]:
    return {
        "territories": generate_territories(),
        "sales_reps":  generate_reps(),
        "stores":      generate_stores(),
        "products":    generate_products(),
        "demographics":generate_demographics(),
    }

