import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def raw_df():
    rng = np.random.default_rng(7)
    n = 600
    districts = np.array(["Yasamal", "Nesimi", "Xetai", "Sabuncu", "Binequdi"])
    base_price = {"Yasamal": 2600, "Nesimi": 2900, "Xetai": 1800, "Sabuncu": 1200, "Binequdi": 1400}
    district = rng.choice(districts, n)
    area = rng.uniform(30, 220, n)
    rooms = rng.integers(1, 6, n)
    floor_total = rng.integers(5, 26, n)
    floor_current = np.minimum(rng.integers(1, 26, n), floor_total)
    per_m2 = np.array([base_price[d] for d in district]) * rng.uniform(0.8, 1.2, n)
    deal = rng.choice(["sale", "rent"], n, p=[0.7, 0.3])
    owner = rng.choice(["owner", "agent"], n, p=[0.45, 0.55])
    price = per_m2 * area
    price = np.where(deal == "rent", price * 0.006, price)
    return pd.DataFrame(
        {
            "id": np.arange(n),
            "source_key": rng.choice(["bina_az", "tap_az", "emlak_az"], n),
            "deal_type": deal,
            "is_daily_rent": 0,
            "price": np.round(price, 2),
            "currency": "AZN",
            "rooms": rooms,
            "area": np.round(area, 2),
            "land_area": np.nan,
            "property_type": rng.choice(["menzil", "heyet_evi", "ofis"], n),
            "floor_current": floor_current,
            "floor_total": floor_total,
            "city": "Baki",
            "district": district,
            "neighborhood": rng.choice(["", "Ganclik", "8 Noyabr"], n),
            "latitude": rng.uniform(40.30, 40.50, n),
            "longitude": rng.uniform(49.70, 49.95, n),
            "owner_type": owner,
            "view_count": rng.integers(0, 1500, n),
            "published_at": "2026-07-01T10:00:00Z",
            "title": rng.choice(["Temirli menzil", "Yeni tikili", "Heyet evi"], n),
            "description": rng.choice(
                ["kombili ve mebelli", "temir olunub kupca var", "genis heyet"], n
            ),
            "metro": rng.choice(["", "Nizami", "Genclik", "28 May"], n),
            "is_active": 1,
        }
    )

