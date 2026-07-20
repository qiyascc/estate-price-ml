from estate_price_ml import features, segments
from estate_price_ml.estimate import estimate, to_frame
from estate_price_ml.model import PriceModel


def test_to_frame_fills_defaults():
    frame = to_frame({"district": "Yasamal", "rooms": 2, "area": 70})
    assert len(frame) == 1
    assert "area_per_room" in frame.columns
    assert "dist_center" in frame.columns


def test_estimate_positive(raw_df, tmp_path):
    frame = features.add_features(features.clean(raw_df))
    sale = segments.split_segment(frame, "satis")
    model = PriceModel().fit(sale)
    path = tmp_path / "satis.joblib"
    model.save(str(path))
    values = estimate(str(path), {"district": "Yasamal", "city": "Baki", "rooms": 3, "area": 100})
    assert len(values) == 1
    assert values[0] > 0


def test_estimate_many(raw_df, tmp_path):
    frame = features.add_features(features.clean(raw_df))
    sale = segments.split_segment(frame, "satis")
    model = PriceModel().fit(sale)
    path = tmp_path / "satis.joblib"
    model.save(str(path))
    records = [
        {"district": "Yasamal", "rooms": 2, "area": 60},
        {"district": "Xetai", "rooms": 4, "area": 150},
    ]
    values = estimate(str(path), records)
    assert len(values) == 2
    assert all(v > 0 for v in values)
