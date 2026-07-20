import pandas as pd

from estate_price_ml import features
from estate_price_ml.model import PriceModel

DEFAULTS = {
    "deal_type": "sale",
    "is_daily_rent": 0,
    "price": 1.0,
    "currency": "AZN",
    "rooms": 0,
    "area": 0.0,
    "land_area": 0.0,
    "property_type": "",
    "floor_current": 0,
    "floor_total": 0,
    "city": "",
    "district": "",
    "neighborhood": "",
    "latitude": None,
    "longitude": None,
    "metro": "",
    "view_count": 0,
    "published_at": None,
    "title": "",
    "source_key": "",
}


def to_frame(records):
    if isinstance(records, dict):
        records = [records]
    rows = []
    for record in records:
        row = dict(DEFAULTS)
        row.update(record)
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame = features.add_price_azn(frame)
    frame = features.add_features(frame)
    return frame


def estimate(model_path, records):
    model = PriceModel.load(model_path)
    frame = to_frame(records)
    values = model.predict(frame)
    return [round(float(value), 2) for value in values]
