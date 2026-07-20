import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict, train_test_split

from estate_price_ml import config


def build_regressor():
    return HistGradientBoostingRegressor(categorical_features="from_dtype", **config.MODEL_PARAMS)


def mean_absolute_percentage(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true > 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def _score(y_test_log, pred_log):
    true = np.expm1(y_test_log)
    pred = np.expm1(pred_log)
    errors = np.abs(true - pred)
    return {
        "mae": float(mean_absolute_error(true, pred)),
        "rmse": float(mean_squared_error(true, pred) ** 0.5),
        "medyan_xeta": float(np.median(errors)),
        "r2": float(r2_score(y_test_log, pred_log)),
        "mape": mean_absolute_percentage(true, pred),
    }


class PriceModel:
    def __init__(self, numeric_features=None, categorical_features=None):
        self.numeric_features = list(numeric_features or config.NUMERIC_FEATURES)
        self.categorical_features = list(categorical_features or config.CATEGORICAL_FEATURES)
        self.regressor = build_regressor()
        self.metrics_ = {}

    def _columns(self, df):
        return [c for c in self.numeric_features + self.categorical_features if c in df.columns]

    def _prepare(self, df):
        frame = df[self._columns(df)].copy()
        for column in self.categorical_features:
            if column in frame.columns:
                frame[column] = frame[column].fillna("").astype(str).astype("category")
        for column in self.numeric_features:
            if column in frame.columns:
                frame[column] = frame[column].astype(float)
        return frame

    def _target(self, df):
        return np.log1p(df[config.TARGET].astype(float))

    def fit(self, df):
        x = self._prepare(df)
        y = self._target(df)
        self.regressor.fit(x, y)
        return self

    def predict(self, df):
        x = self._prepare(df)
        return np.expm1(self.regressor.predict(x))

    def evaluate(self, df, test_size=0.2, random_state=42):
        x = self._prepare(df)
        y = self._target(df)
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size, random_state=random_state
        )
        self.regressor.fit(x_train, y_train)
        pred_log = self.regressor.predict(x_test)
        self.metrics_ = {"n_train": int(len(x_train)), "n_test": int(len(x_test))}
        self.metrics_.update(_score(np.asarray(y_test), np.asarray(pred_log)))
        return self.metrics_

    def cross_validate(self, df, folds=None):
        x = self._prepare(df)
        y = self._target(df)
        splitter = KFold(n_splits=folds or config.CV_FOLDS, shuffle=True, random_state=42)
        pred_log = cross_val_predict(build_regressor(), x, y, cv=splitter)
        result = {"n": int(len(x)), "folds": splitter.get_n_splits()}
        result.update(_score(np.asarray(y), np.asarray(pred_log)))
        return result

    def save(self, path):
        joblib.dump(
            {
                "regressor": self.regressor,
                "numeric": self.numeric_features,
                "categorical": self.categorical_features,
                "metrics": self.metrics_,
            },
            path,
            compress=3,
        )

    @staticmethod
    def load(path):
        payload = joblib.load(path)
        model = PriceModel(payload["numeric"], payload["categorical"])
        model.regressor = payload["regressor"]
        model.metrics_ = payload.get("metrics", {})
        return model
