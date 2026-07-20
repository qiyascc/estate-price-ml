from estate_price_ml import features, segments
from estate_price_ml.model import PriceModel


def _sale_frame(raw_df):
    frame = features.add_features(features.clean(raw_df))
    return segments.split_segment(frame, "satis")


def test_fit_predict_positive(raw_df):
    sale = _sale_frame(raw_df)
    model = PriceModel().fit(sale)
    predictions = model.predict(sale)
    assert len(predictions) == len(sale)
    assert (predictions > 0).all()


def test_evaluate_returns_metrics(raw_df):
    sale = _sale_frame(raw_df)
    metrics = PriceModel().evaluate(sale)
    for key in ("mae", "rmse", "r2", "mape", "n_train", "n_test"):
        assert key in metrics
    assert metrics["mae"] >= 0


def test_save_and_load(raw_df, tmp_path):
    sale = _sale_frame(raw_df)
    model = PriceModel().fit(sale)
    path = tmp_path / "satis.joblib"
    model.save(str(path))
    loaded = PriceModel.load(str(path))
    assert len(loaded.predict(sale)) == len(sale)
