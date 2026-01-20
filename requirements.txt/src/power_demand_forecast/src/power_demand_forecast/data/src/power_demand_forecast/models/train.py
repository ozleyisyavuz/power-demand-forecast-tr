from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    eps = 1e-6
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100)


def main() -> None:
    data_path = Path("data/processed/demand.csv")
    if not data_path.exists():
        raise FileNotFoundError("Önce veri üret: python -m src.power_demand_forecast.data.make_dataset")

    df = pd.read_csv(data_path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp")

    features = ["hour", "dayofweek", "is_weekend", "temperature_c"]
    target = "demand_mw"

    X = df[features]
    y = df[target].values

    val_hours = 14 * 24
    X_train, X_val = X.iloc[:-val_hours], X.iloc[-val_hours:]
    y_train, y_val = y[:-val_hours], y[-val_hours:]

    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["dayofweek"]),
        ],
        remainder="passthrough",
    )

    model = HistGradientBoostingRegressor(random_state=42)
    pipe = Pipeline([("pre", pre), ("model", model)])

    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_val)

    val_mape = mape(y_val, pred)
    print(f"Validation MAPE: {val_mape:.2f}%")

    out_dir = Path("models")
    out_dir.mkdir(exist_ok=True)
    joblib.dump(pipe, out_dir / "model.joblib")
    print(f"Saved model: {out_dir / 'model.joblib'}")


if __name__ == "__main__":
    main()

