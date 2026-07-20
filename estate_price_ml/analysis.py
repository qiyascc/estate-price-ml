import pandas as pd

from estate_price_ml import config


def market_summary(df):
    per_m2 = (df[config.TARGET] / df["area"]).replace([float("inf"), float("-inf")], pd.NA).dropna()
    return {
        "elan_sayi": int(len(df)),
        "orta_qiymet": float(df[config.TARGET].mean()),
        "medyan_qiymet": float(df[config.TARGET].median()),
        "medyan_kv_metr_qiymet": float(per_m2.median()) if len(per_m2) else 0.0,
        "min_qiymet": float(df[config.TARGET].min()),
        "max_qiymet": float(df[config.TARGET].max()),
    }


def region_summary(df, by="district", min_count=5):
    data = df.copy()
    data["kv_metr"] = data[config.TARGET] / data["area"]
    grouped = data.groupby(by).agg(
        elan_sayi=(config.TARGET, "size"),
        medyan_qiymet=(config.TARGET, "median"),
        orta_qiymet=(config.TARGET, "mean"),
        medyan_kv_metr=("kv_metr", "median"),
    )
    grouped = grouped[grouped["elan_sayi"] >= min_count]
    grouped = grouped.sort_values("medyan_kv_metr", ascending=False)
    return grouped.reset_index()


def owner_vs_agent(df):
    sale = df[df["deal_type"] == "sale"]
    result = {}
    for name in ("owner", "agent"):
        part = sale[sale["owner_type"] == name]
        if len(part):
            per_m2 = (part[config.TARGET] / part["area"]).median()
            result[name] = {
                "elan_sayi": int(len(part)),
                "medyan_qiymet": float(part[config.TARGET].median()),
                "medyan_kv_metr": float(per_m2),
            }
    if "owner" in result and "agent" in result:
        owner_m2 = result["owner"]["medyan_kv_metr"]
        agent_m2 = result["agent"]["medyan_kv_metr"]
        if owner_m2:
            result["ferq_faiz"] = float((agent_m2 - owner_m2) / owner_m2 * 100.0)
    return result


def rental_yield_by_region(df, by="district", min_count=5):
    sale = df[df["deal_type"] == "sale"].groupby(by)[config.TARGET].median()
    rent = df[df["deal_type"] == "rent"].groupby(by)[config.TARGET].median()
    both = pd.DataFrame({"satis_medyan": sale, "kiraye_medyan": rent}).dropna()
    both = both[both["satis_medyan"] > 0]
    both["illik_gelirlilik_faiz"] = both["kiraye_medyan"] * 12.0 / both["satis_medyan"] * 100.0
    both = both.sort_values("illik_gelirlilik_faiz", ascending=False)
    return both.reset_index()


def property_type_breakdown(df, min_count=5):
    grouped = df.groupby("property_type").agg(
        elan_sayi=(config.TARGET, "size"),
        medyan_qiymet=(config.TARGET, "median"),
    )
    grouped = grouped[grouped["elan_sayi"] >= min_count]
    return grouped.sort_values("elan_sayi", ascending=False).reset_index()
