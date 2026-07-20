from estate_price_ml import analysis, features


def _frame(raw_df):
    return features.add_features(features.clean(raw_df))


def test_market_summary(raw_df):
    summary = analysis.market_summary(_frame(raw_df))
    assert summary["elan_sayi"] > 0
    assert summary["medyan_qiymet"] > 0
    assert summary["medyan_kv_metr_qiymet"] > 0


def test_region_summary(raw_df):
    result = analysis.region_summary(_frame(raw_df), by="district", min_count=1)
    assert "medyan_kv_metr" in result.columns
    assert len(result) > 0


def test_owner_vs_agent(raw_df):
    result = analysis.owner_vs_agent(_frame(raw_df))
    assert "owner" in result
    assert "agent" in result


def test_rental_yield(raw_df):
    result = analysis.rental_yield_by_region(_frame(raw_df), by="district", min_count=1)
    assert "illik_gelirlilik_faiz" in result.columns
