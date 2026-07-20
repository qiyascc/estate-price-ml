import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from estate_price_ml import config


def build_pipeline(numeric_features, categorical_features):
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", max_categories=80)),
        ]
    )
    pre = ColumnTransformer(
        [
            ("num", numeric, numeric_features),
            ("cat", categorical, categorical_features),
        ]
    )
    regressor = HistGradientBoostingRegressor(
        max_iter=400,
        learning_rate=0.06,
        l2_regularization=1.0,
        random_state=42,
    )
    return Pipeline([("pre", pre), ("reg", regressor)])


def mean_absolute_percentage(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true > 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


class PriceModel:
    def __init__(self, numeric_features=None, categorical_features=None):
        self.numeric_features = list(numeric_features or config.NUMERIC_FEATURES)
        self.categorical_features = list(categorical_features or config.CATEGORICAL_FEATURES)
        self.pipeline = build_pipeline(self.numeric_features, self.categorical_features)
        self.metrics_ = {}

    def _columns(self, df):
        return [c for c in self.numeric_features + self.categorical_features if c in df.columns]

    def _xy(self, df):
        x = df[self._columns(df)]
        y = np.log1p(df[config.TARGET].astype(float))
        return x, y

    def fit(self, df):
        x, y = self._xy(df)
        self.pipeline.fit(x, y)
        return self

    def predict(self, df):
        x = df[self._columns(df)]
        return np.expm1(self.pipeline.predict(x))

    def evaluate(self, df, test_size=0.2, random_state=42):
        x, y = self._xy(df)
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size, random_state=random_state
        )
        self.pipeline.fit(x_train, y_train)
        pred_log = self.pipeline.predict(x_test)
        pred = np.expm1(pred_log)
        true = np.expm1(y_test)
        self.metrics_ = {
            "n_train": int(len(x_train)),
            "n_test": int(len(x_test)),
            "mae": float(mean_absolute_error(true, pred)),
            "rmse": float(mean_squared_error(true, pred) ** 0.5),
            "r2": float(r2_score(y_test, pred_log)),
            "mape": mean_absolute_percentage(true, pred),
        }
        return self.metrics_

    def save(self, path):
        joblib.dump(
            {
                "pipeline": self.pipeline,
                "numeric": self.numeric_features,
                "categorical": self.categorical_features,
                "metrics": self.metrics_,
            },
            path,
        )

    @staticmethod
    def load(path):
        payload = joblib.load(path)
        model = PriceModel(payload["numeric"], payload["categorical"])
        model.pipeline = payload["pipeline"]
        model.metrics_ = payload.get("metrics", {})
        return model
