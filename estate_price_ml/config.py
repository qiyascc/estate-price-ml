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

CITY_CENTER = (40.3777, 49.8920)

TARGET = "price_azn"

KEYWORD_FLAGS = {
    "temirli": ["temir", "remont", "təmir", "yaxsi veziyyet", "yaxşı vəziyyət"],
    "mebelli": ["mebel", "mébel", "əşya", "esya", "furnished"],
    "yeni_tikili": ["yeni tikili", "novostroy", "yeni bina", "new building"],
    "kombili": ["kombi", "kombili"],
    "heyetli": ["həyət", "heyet", "bag ev", "bağ ev"],
    "senedli": ["kupça", "kupca", "çıxarış", "cixaris", "şəhadətnamə", "sehadetname"],
}

NUMERIC_FEATURES = [
    "area",
    "rooms",
    "area_per_room",
    "land_area",
    "floor_current",
    "floor_total",
    "floor_ratio",
    "log_area",
    "has_coords",
    "latitude",
    "longitude",
    "dist_center",
    "view_count",
    "age_days",
    "is_daily_rent",
] + list(KEYWORD_FLAGS.keys())

CATEGORICAL_FEATURES = [
    "city",
    "district",
    "neighborhood",
    "metro",
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

MODEL_PARAMS = {
    "loss": "absolute_error",
    "max_iter": 700,
    "learning_rate": 0.05,
    "max_leaf_nodes": 63,
    "min_samples_leaf": 40,
    "l2_regularization": 1.0,
    "max_bins": 255,
    "early_stopping": True,
    "validation_fraction": 0.1,
    "n_iter_no_change": 25,
    "random_state": 42,
}

MIN_PRICE = 50.0
MAX_PRICE = 100000000.0
MIN_AREA = 5.0
MAX_AREA = 100000.0

SALE_MIN_PRICE = 5000.0
RENT_MIN_PRICE = 30.0
SALE_PPM_MIN = 100.0
SALE_PPM_MAX = 30000.0
RENT_PPM_MIN = 0.5
RENT_PPM_MAX = 400.0

MIN_SEGMENT_ROWS = 200
CV_FOLDS = 4
