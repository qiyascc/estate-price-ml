import re

import numpy as np
import pandas as pd

from estate_price_ml import config

_TEXT_COLUMNS = [
    "city",
    "district",
    "neighborhood",
    "metro",
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
    "is_daily_rent",
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
    for col in config.CATEGORICAL_FEATURES:
        if col in out.columns:
            out[col] = out[col].astype("category")
    return out.reset_index(drop=True)


def _haversine(lat, lon, center_lat, center_lon):
    radius = 6371.0
    lat1 = np.radians(lat.astype(float))
    lon1 = np.radians(lon.astype(float))
    lat2 = np.radians(center_lat)
    lon2 = np.radians(center_lon)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    inner = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return radius * 2 * np.arcsin(np.sqrt(inner.clip(0, 1)))


def add_keyword_flags(df):
    out = df.copy()
    if "title" in out.columns or "description" in out.columns:
        title = out.get("title", pd.Series("", index=out.index)).fillna("").astype(str)
        desc = out.get("description", pd.Series("", index=out.index)).fillna("").astype(str)
        text = (title + " " + desc).str.lower()
    else:
        text = pd.Series("", index=out.index)
    for flag, keywords in config.KEYWORD_FLAGS.items():
        pattern = "|".join(re.escape(word) for word in keywords)
        out[flag] = text.str.contains(pattern, regex=True, na=False).astype(int)
    return out


def add_features(df):
    out = df.copy()
    if "price_azn" not in out.columns:
        out = add_price_azn(out)
    if "metro" not in out.columns:
        out["metro"] = ""
    if "is_daily_rent" not in out.columns:
        out["is_daily_rent"] = 0
    out["is_daily_rent"] = pd.to_numeric(out["is_daily_rent"], errors="coerce").fillna(0).astype(int)
    out["price_per_m2"] = out["price_azn"] / out["area"].replace(0, np.nan)
    rooms = out["rooms"].fillna(0)
    out["area_per_room"] = out["area"] / rooms.replace(0, np.nan)
    out["area_per_room"] = out["area_per_room"].fillna(out["area"])
    total = out["floor_total"].replace(0, np.nan)
    out["floor_ratio"] = (out["floor_current"] / total).clip(lower=0, upper=1).fillna(0)
    out["has_coords"] = (out["latitude"].notna() & out["longitude"].notna()).astype(int)
    out["log_area"] = np.log1p(out["area"].clip(lower=1))
    out["rooms"] = rooms
    out["land_area"] = out["land_area"].fillna(0)
    out["view_count"] = out["view_count"].fillna(0)
    center_lat, center_lon = config.CITY_CENTER
    dist = _haversine(out["latitude"].fillna(center_lat), out["longitude"].fillna(center_lon), center_lat, center_lon)
    out["dist_center"] = dist.where(out["has_coords"] == 1, np.nan)
    if "published_at" in out.columns:
        published = pd.to_datetime(out["published_at"], errors="coerce", utc=True)
        now = pd.Timestamp.now(tz="UTC")
        out["age_days"] = (now - published).dt.days.clip(lower=0).fillna(0)
    else:
        out["age_days"] = 0
    out = add_keyword_flags(out)
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
