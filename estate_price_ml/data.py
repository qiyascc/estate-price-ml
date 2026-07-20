import sqlite3

import pandas as pd

from estate_price_ml import config

LISTING_COLUMNS = [
    "id",
    "source_key",
    "deal_type",
    "is_daily_rent",
    "price",
    "currency",
    "rooms",
    "area",
    "land_area",
    "property_type",
    "floor_current",
    "floor_total",
    "city",
    "district",
    "neighborhood",
    "latitude",
    "longitude",
    "owner_type",
    "view_count",
    "published_at",
    "title",
    "is_active",
]


def load_listings(db_path=None, csv_path=None, active_only=True):
    if csv_path:
        return pd.read_csv(csv_path)
    path = db_path or config.DB_PATH
    con = sqlite3.connect(path)
    try:
        cols = ", ".join(LISTING_COLUMNS)
        where = "WHERE is_active = 1" if active_only else ""
        query = "SELECT {} FROM {} {}".format(cols, config.TABLE, where)
        df = pd.read_sql_query(query, con)
    finally:
        con.close()
    return df


def load_region_tags(db_path=None, kinds=("metro",)):
    path = db_path or config.DB_PATH
    con = sqlite3.connect(path)
    try:
        placeholders = ", ".join("?" for _ in kinds)
        query = "SELECT listing_id, kind, name FROM {} WHERE kind IN ({})".format(
            config.TAG_TABLE, placeholders
        )
        df = pd.read_sql_query(query, con, params=list(kinds))
    finally:
        con.close()
    return df


def attach_metro(df, tags):
    out = df.copy()
    if tags is None or len(tags) == 0:
        out["metro"] = ""
        return out
    metro = tags[tags["kind"] == "metro"].drop_duplicates("listing_id")
    metro = metro.rename(columns={"name": "metro", "listing_id": "id"})[["id", "metro"]]
    out = out.merge(metro, on="id", how="left")
    out["metro"] = out["metro"].fillna("")
    return out


def load_with_metro(db_path=None, csv_path=None, active_only=True):
    df = load_listings(db_path=db_path, csv_path=csv_path, active_only=active_only)
    if csv_path:
        if "metro" not in df.columns:
            df["metro"] = ""
        return df
    tags = load_region_tags(db_path=db_path)
    return attach_metro(df, tags)
