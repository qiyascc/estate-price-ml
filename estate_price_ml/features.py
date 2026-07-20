import numpy as np
import pandas as pd

from estate_price_ml import config

_TEXT_COLUMNS = [
    "city",
    "district",
    "neighborhood",
    "property_type",
    "currency",
    "deal_type",
    "owner_type",
    "source_key",
]

_NUMERIC_COLUMNS = [
    "price",
    "rooms",
    "area",
    "land_area",
    "floor_current",
    "floor_total",
    "view_count",
    "latitude",
    "longitude",
]


def add_price_azn(df):
    out = df.copy()
    currency = out.get("currency")
    if currency is None:
        out["price_azn"] = out["price"].astype(float)
        return out
    rate = (
        currency.fillna("").astype(str).str.strip().str.upper()
        .map(config.CURRENCY_TO_AZN)
        .fillna(1.0)
    )
    out["price_azn"] = out["price"].astype(float) * rate.astype(float)
    return out


def clean(df):
    out = df.copy()
    for col in _NUMERIC_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in _TEXT_COLUMNS:
        if col in out.columns:
            out[col] = out[col].fillna("").astype(str).str.strip()
    out = out[out["price"].notna() & (out["price"] > 0)]
    out = add_price_azn(out)
    out = out[(out["price_azn"] >= config.MIN_PRICE) & (out["price_azn"] <= config.MAX_PRICE)]
    out = out[out["area"].notna() & (out["area"] >= config.MIN_AREA) & (out["area"] <= config.MAX_AREA)]
    return out.reset_index(drop=True)


def add_features(df):
    out = df.copy()
    if "price_azn" not in out.columns:
        out = add_price_azn(out)
    out["price_per_m2"] = out["price_azn"] / out["area"].replace(0, np.nan)
    total = out["floor_total"].replace(0, np.nan)
    out["floor_ratio"] = (out["floor_current"] / total).clip(lower=0, upper=1).fillna(0)
    out["has_coords"] = (out["latitude"].notna() & out["longitude"].notna()).astype(int)
    out["log_area"] = np.log1p(out["area"].clip(lower=1))
    out["rooms"] = out["rooms"].fillna(0)
    out["land_area"] = out["land_area"].fillna(0)
    out["view_count"] = out["view_count"].fillna(0)
    if "published_at" in out.columns:
        published = pd.to_datetime(out["published_at"], errors="coerce", utc=True)
        now = pd.Timestamp.now(tz="UTC")
        out["age_days"] = (now - published).dt.days.clip(lower=0).fillna(0)
    return out


def winsorize_target(df, lower=0.01, upper=0.99, by=None):
    out = df.copy()
    target = config.TARGET
    if by and by in out.columns:
        keep = pd.Series(False, index=out.index)
        for _, index in out.groupby(by).groups.items():
            values = out.loc[index, target]
            low, high = values.quantile([lower, upper])
            keep.loc[index] = (values >= low) & (values <= high)
        return out[keep].reset_index(drop=True)
    low, high = out[target].quantile([lower, upper])
    return out[(out[target] >= low) & (out[target] <= high)].reset_index(drop=True)
