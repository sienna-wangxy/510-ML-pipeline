"""Reproducible, leakage-aware engagement-rate experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


NUMERIC = ["followers", "posts"]
CATEGORICAL = ["country", "category"]
TARGET = "60_day_eng_rate"
SEED = 42


def make_model() -> Pipeline:
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=3)),
        ]
    )
    preprocess = ColumnTransformer(
        [("numeric", numeric, NUMERIC), ("categorical", categorical, CATEGORICAL)]
    )
    return Pipeline([("preprocess", preprocess), ("regressor", Ridge(alpha=10.0))])


def score(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae_percentage_points": round(float(mean_absolute_error(y_true, y_pred)), 3),
        "rmse_percentage_points": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 3),
        "r2": round(float(r2_score(y_true, y_pred)), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/instagram_global_influencers_2026.csv"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    required = NUMERIC + CATEGORICAL + [TARGET]
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if data[required].isna().any().any():
        raise ValueError("Dataset has missing values in required columns; inspect before training")
    if data[TARGET].lt(0).any() or data[NUMERIC].lt(0).any().any():
        raise ValueError("Engagement rate, followers, and posts must be nonnegative")

    X = data[NUMERIC + CATEGORICAL]
    y = data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED
    )

    baseline = DummyRegressor(strategy="median").fit(X_train, y_train)
    model = make_model().fit(X_train, y_train)
    baseline_pred = baseline.predict(X_test)
    model_pred = model.predict(X_test)

    result = {
        "dataset_rows": len(data),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "random_seed": SEED,
        "target": TARGET,
        "features": NUMERIC + CATEGORICAL,
        "baseline_test": score(y_test, baseline_pred),
        "ridge_test": score(y_test, model_pred),
        "follower_engagement_spearman": round(float(data["followers"].corr(y, method="spearman")), 3),
    }

    quartiles = pd.qcut(data["followers"], q=4, labels=["Q1", "Q2", "Q3", "Q4"])
    quartile_medians = data.groupby(quartiles, observed=True)[TARGET].median()
    result["follower_quartile_median_engagement_pct"] = {
        key: round(float(value), 3) for key, value in quartile_medians.items()
    }

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].scatter(data["followers"] / 1e6, y, alpha=0.7, s=28)
    axes[0].set(xlabel="Followers (millions)", ylabel="60-day engagement rate (%)", title="Follower size vs. engagement")
    axes[1].scatter(y_test, model_pred, alpha=0.8, s=38)
    limits = [min(y_test.min(), model_pred.min()), max(y_test.max(), model_pred.max())]
    axes[1].plot(limits, limits, linestyle="--", color="gray", linewidth=1)
    axes[1].set(xlabel="Observed engagement rate (%)", ylabel="Predicted rate (%)", title="Held-out test accounts")
    fig.tight_layout()
    fig.savefig(args.output / "engagement_story.png", dpi=160)
    plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
