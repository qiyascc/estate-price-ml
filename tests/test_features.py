from estate_price_ml import config, features


def test_clean_drops_bad_prices(raw_df):
    raw_df.loc[0, "price"] = 0
    raw_df.loc[1, "price"] = None
    out = features.clean(raw_df)
    assert (out["price"] > 0).all()
    assert config.TARGET in out.columns
    assert len(out) <= len(raw_df) - 2


def test_price_azn_converts_usd(raw_df):
    raw_df.loc[0, "currency"] = "USD"
    raw_df.loc[0, "price"] = 100.0
    out = features.add_price_azn(raw_df)
    assert out.loc[0, "price_azn"] > 100.0


def test_add_features_columns(raw_df):
    out = features.add_features(features.clean(raw_df))
    for column in ("price_per_m2", "floor_ratio", "has_coords", "log_area", "age_days"):
        assert column in out.columns
    assert (out["floor_ratio"] >= 0).all()
    assert (out["floor_ratio"] <= 1).all()


def test_winsorize_reduces_extremes(raw_df):
    cleaned = features.add_features(features.clean(raw_df))
    cleaned.loc[0, config.TARGET] = cleaned[config.TARGET].max() * 100
    out = features.winsorize_target(cleaned, by="deal_type")
    assert out[config.TARGET].max() < cleaned[config.TARGET].max()
