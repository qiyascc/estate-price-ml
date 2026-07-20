import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_DB = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "db.sqlite3")

DB_PATH = os.environ.get("ESTATEPRICEML_DB", _DEFAULT_DB)
TABLE = "listings_listing"
TAG_TABLE = "listings_locationtag"

CURRENCY_TO_AZN = {
    "AZN": 1.0,
    "": 1.0,
    "USD": 1.70,
    "$": 1.70,
    "EUR": 1.85,
    "€": 1.85,
    "MANAT": 1.0,
    "₼": 1.0,
    "RUB": 0.018,
    "GBP": 2.15,
}

TARGET = "price_azn"

NUMERIC_FEATURES = [
    "area",
    "rooms",
    "land_area",
    "floor_current",
    "floor_total",
    "floor_ratio",
    "log_area",
    "has_coords",
    "view_count",
    "latitude",
    "longitude",
]

CATEGORICAL_FEATURES = [
    "city",
    "district",
    "neighborhood",
    "property_type",
    "source_key",
]

REGION_FIELDS = ["district", "neighborhood", "city"]

SEGMENTS = {
    "satis": {"deal_type": "sale"},
    "kiraye": {"deal_type": "rent"},
    "sahibkar_satis": {"deal_type": "sale", "owner_type": "owner"},
    "vasitechi_satis": {"deal_type": "sale", "owner_type": "agent"},
}

MIN_PRICE = 50.0
MAX_PRICE = 100000000.0
MIN_AREA = 5.0
MAX_AREA = 100000.0
MIN_SEGMENT_ROWS = 50
