"""
Global configuration and constants for the Kellanova NZ Retail POC.
"""
from pathlib import Path
import os

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "data" / "models"
VECTOR_DIR = BASE_DIR / "data" / "vector"

for _d in [DATA_DIR, MODELS_DIR, VECTOR_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Synthetic data parameters ───────────────────────────────────────────────
NUM_STORES   = 50
NUM_SKUS     = 20
NUM_WEEKS    = 16          # 16 weeks of POS history
NUM_EVENTS   = 15
RANDOM_SEED  = 42

# Week end dates (Sundays)
import pandas as pd
WEEK_DATES = pd.date_range(end="2024-06-30", periods=NUM_WEEKS, freq="W")

# ── Territories ─────────────────────────────────────────────────────────────
TERRITORIES = [
    {"territory_id": "T01", "territory_name": "Auckland Metro",      "region": "Auckland",        "country": "NZ"},
    {"territory_id": "T02", "territory_name": "Upper North Island",  "region": "Waikato/BOP",     "country": "NZ"},
    {"territory_id": "T03", "territory_name": "Lower North Island",  "region": "Wellington/Manawatu","country": "NZ"},
    {"territory_id": "T04", "territory_name": "Upper South Island",  "region": "Canterbury/Nelson","country": "NZ"},
    {"territory_id": "T05", "territory_name": "Lower South Island",  "region": "Otago/Southland", "country": "NZ"},
]

# ── NZ Retail Chains ────────────────────────────────────────────────────────
CHAINS = ["Woolworths NZ", "New World", "PAK'nSAVE", "Four Square", "FreshChoice", "SuperValue"]

STORE_FORMATS = {
    "Woolworths NZ": "Supermarket",
    "New World":     "Supermarket",
    "PAK'nSAVE":     "Supermarket",
    "Four Square":   "Convenience",
    "FreshChoice":   "Supermarket",
    "SuperValue":    "Supermarket",
}

# ── NZ Cities with lat/lon and territory mapping ─────────────────────────────
NZ_CITIES = [
    {"city": "Auckland CBD",    "lat": -36.8485, "lon": 174.7633, "territory_id": "T01", "region": "Auckland"},
    {"city": "Auckland North",  "lat": -36.7889, "lon": 174.7553, "territory_id": "T01", "region": "Auckland"},
    {"city": "Auckland South",  "lat": -37.0000, "lon": 174.8800, "territory_id": "T01", "region": "Auckland"},
    {"city": "Auckland West",   "lat": -36.9000, "lon": 174.6300, "territory_id": "T01", "region": "Auckland"},
    {"city": "Hamilton",        "lat": -37.7870, "lon": 175.2793, "territory_id": "T02", "region": "Waikato/BOP"},
    {"city": "Tauranga",        "lat": -37.6878, "lon": 176.1651, "territory_id": "T02", "region": "Waikato/BOP"},
    {"city": "Rotorua",         "lat": -38.1368, "lon": 176.2497, "territory_id": "T02", "region": "Waikato/BOP"},
    {"city": "New Plymouth",    "lat": -39.0556, "lon": 174.0752, "territory_id": "T03", "region": "Wellington/Manawatu"},
    {"city": "Palmerston North","lat": -40.3523, "lon": 175.6082, "territory_id": "T03", "region": "Wellington/Manawatu"},
    {"city": "Napier",          "lat": -39.4928, "lon": 176.9120, "territory_id": "T03", "region": "Wellington/Manawatu"},
    {"city": "Wellington CBD",  "lat": -41.2865, "lon": 174.7762, "territory_id": "T03", "region": "Wellington/Manawatu"},
    {"city": "Wellington North","lat": -41.2200, "lon": 174.8000, "territory_id": "T03", "region": "Wellington/Manawatu"},
    {"city": "Nelson",          "lat": -41.2706, "lon": 173.2840, "territory_id": "T04", "region": "Canterbury/Nelson"},
    {"city": "Christchurch CBD","lat": -43.5321, "lon": 172.6362, "territory_id": "T04", "region": "Canterbury/Nelson"},
    {"city": "Christchurch South","lat":-43.5800,"lon": 172.6500, "territory_id": "T04", "region": "Canterbury/Nelson"},
    {"city": "Dunedin",         "lat": -45.8788, "lon": 170.5028, "territory_id": "T05", "region": "Otago/Southland"},
    {"city": "Queenstown",      "lat": -45.0312, "lon": 168.6626, "territory_id": "T05", "region": "Otago/Southland"},
    {"city": "Invercargill",    "lat": -46.4132, "lon": 168.3538, "territory_id": "T05", "region": "Otago/Southland"},
]

# ── Kellanova Products ───────────────────────────────────────────────────────
PRODUCTS = [
    # Cereals
    {"sku_id":"KEL001","product_name":"Kellogg's Corn Flakes 500g",    "brand":"Kellogg's","category":"Cereal",      "pack_size":"500g", "base_price":5.49,"promo_sensitivity":1.3,"seasonality":"even"},
    {"sku_id":"KEL002","product_name":"Kellogg's Coco Pops 500g",      "brand":"Kellogg's","category":"Cereal",      "pack_size":"500g", "base_price":5.99,"promo_sensitivity":1.5,"seasonality":"even"},
    {"sku_id":"KEL003","product_name":"Kellogg's Nutri-Grain 500g",    "brand":"Kellogg's","category":"Cereal",      "pack_size":"500g", "base_price":6.49,"promo_sensitivity":1.2,"seasonality":"summer"},
    {"sku_id":"KEL004","product_name":"Kellogg's Special K 400g",      "brand":"Kellogg's","category":"Cereal",      "pack_size":"400g", "base_price":5.99,"promo_sensitivity":1.4,"seasonality":"even"},
    {"sku_id":"KEL005","product_name":"Kellogg's Rice Bubbles 500g",   "brand":"Kellogg's","category":"Cereal",      "pack_size":"500g", "base_price":5.49,"promo_sensitivity":1.2,"seasonality":"even"},
    {"sku_id":"KEL006","product_name":"Kellogg's Sultana Bran 500g",   "brand":"Kellogg's","category":"Cereal",      "pack_size":"500g", "base_price":5.99,"promo_sensitivity":1.1,"seasonality":"even"},
    {"sku_id":"KEL007","product_name":"Kellogg's Froot Loops 500g",    "brand":"Kellogg's","category":"Cereal",      "pack_size":"500g", "base_price":6.49,"promo_sensitivity":1.6,"seasonality":"even"},
    {"sku_id":"KEL008","product_name":"Kellogg's Just Right 500g",     "brand":"Kellogg's","category":"Cereal",      "pack_size":"500g", "base_price":5.99,"promo_sensitivity":1.2,"seasonality":"even"},
    # Snack Bars
    {"sku_id":"KEL009","product_name":"LCMs Choc Bars 6pk",            "brand":"LCMs",     "category":"Snack Bars",  "pack_size":"6pk",  "base_price":4.99,"promo_sensitivity":1.7,"seasonality":"summer"},
    {"sku_id":"KEL010","product_name":"LCMs Original Bars 6pk",        "brand":"LCMs",     "category":"Snack Bars",  "pack_size":"6pk",  "base_price":4.49,"promo_sensitivity":1.6,"seasonality":"summer"},
    {"sku_id":"KEL011","product_name":"Nutri-Grain Iron Man Bar 6pk",  "brand":"Kellogg's","category":"Snack Bars",  "pack_size":"6pk",  "base_price":5.49,"promo_sensitivity":1.5,"seasonality":"summer"},
    {"sku_id":"KEL012","product_name":"Special K Bars Choc 6pk",       "brand":"Kellogg's","category":"Snack Bars",  "pack_size":"6pk",  "base_price":4.99,"promo_sensitivity":1.4,"seasonality":"even"},
    # Savoury Snacks
    {"sku_id":"KEL013","product_name":"Pringles Original 134g",        "brand":"Pringles", "category":"Savoury Snacks","pack_size":"134g","base_price":4.49,"promo_sensitivity":1.8,"seasonality":"event"},
    {"sku_id":"KEL014","product_name":"Pringles Sour Cream 134g",      "brand":"Pringles", "category":"Savoury Snacks","pack_size":"134g","base_price":4.49,"promo_sensitivity":1.8,"seasonality":"event"},
    {"sku_id":"KEL015","product_name":"Pringles BBQ 134g",             "brand":"Pringles", "category":"Savoury Snacks","pack_size":"134g","base_price":4.49,"promo_sensitivity":1.7,"seasonality":"event"},
    {"sku_id":"KEL016","product_name":"Pringles Cheese 134g",          "brand":"Pringles", "category":"Savoury Snacks","pack_size":"134g","base_price":4.49,"promo_sensitivity":1.7,"seasonality":"event"},
    {"sku_id":"KEL017","product_name":"Pringles Multipack 5pk",        "brand":"Pringles", "category":"Savoury Snacks","pack_size":"5pk", "base_price":6.99,"promo_sensitivity":1.9,"seasonality":"event"},
    # Beverages / On-the-go
    {"sku_id":"KEL018","product_name":"Nutri-Grain Cereal Pouch 70g",  "brand":"Kellogg's","category":"On-the-Go",   "pack_size":"70g",  "base_price":2.99,"promo_sensitivity":1.3,"seasonality":"summer"},
    {"sku_id":"KEL019","product_name":"Special K Protein Bar 40g",     "brand":"Kellogg's","category":"On-the-Go",   "pack_size":"40g",  "base_price":2.49,"promo_sensitivity":1.4,"seasonality":"even"},
    {"sku_id":"KEL020","product_name":"Kellogg's Muesli Clusters 450g","brand":"Kellogg's","category":"Cereal",      "pack_size":"450g", "base_price":6.99,"promo_sensitivity":1.2,"seasonality":"even"},
]

# ── Ollama settings ──────────────────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL",    "llama3.1:8b")

# ── Opportunity scoring weights ──────────────────────────────────────────────
SEVERITY_WEIGHTS = {
    "out_of_stock":           1.0,
    "distribution_gap":       0.9,
    "promotion_execution":    0.85,
    "shelf_compliance":       0.75,
    "event_opportunity":      0.7,
    "hidden_opportunity":     0.65,
}

# ── Event proximity threshold ────────────────────────────────────────────────
EVENT_RADIUS_KM = 15.0

