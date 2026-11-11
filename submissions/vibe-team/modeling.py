"""ElasticNet, веса обучающих строк и метрики в процентных пунктах SOH."""

from __future__ import annotations

from collections import Counter

import numpy as np
import polars as pl
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def regression_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    """Посчитать MAE, P95, смещение и максимальную абсолютную ошибку."""
    if truth.shape != prediction.shape or not np.isfinite([truth, prediction]).all():
        raise ValueError("Нужны согласованные конечные векторы истины и прогноза")
    error = 100 * (prediction - truth)
    absolute = np.abs(error)
    return {
        "n": len(truth),
        "mae_pp": float(absolute.mean()),
        "p95_abs_pp": float(np.quantile(absolute, 0.95, method="linear")),
        "bias_pp": float(error.mean()),
        "max_abs_pp": float(absolute.max()),
    }


def fit_pipeline(
    matrix: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    parameters: dict[str, object],
    settings: dict[str, object],
) -> Pipeline:
    """Обучить преобразования и ElasticNet только на переданных строках."""
    pipeline = Pipeline(
        [
            ("variance", VarianceThreshold(threshold=1e-12)),
            ("scale", StandardScaler()),
            (
                "elasticnet",
                ElasticNet(
                    alpha=float(parameters["alpha"]),
                    l1_ratio=float(parameters["l1_ratio"]),
                    max_iter=int(settings["max_iter"]),
                    tol=float(settings["tol"]),
                    precompute=True,
                    selection="cyclic",
                    random_state=int(settings["seed"]),
                ),
            ),
        ]
    )
    weighting = parameters["weighting"]
    if weighting == "none":
        pipeline.fit(matrix, target)
    elif weighting == "cell_sqrt":
        counts = Counter(groups)
        weights = np.asarray([1 / np.sqrt(counts[group]) for group in groups])
        weights /= weights.mean()
        pipeline.fit(
            matrix, target, scale__sample_weight=weights, elasticnet__sample_weight=weights
        )
    else:
        raise ValueError(f"Неизвестные веса: {weighting}")
    return pipeline


def prediction_report(
    frame: pl.DataFrame, raw_prediction: np.ndarray, bounds: list[float]
) -> tuple[pl.DataFrame, dict[str, object]]:
    """Сформировать метрики и строки отчёта, сохранив исходные ID."""
    if not np.isfinite(raw_prediction).all():
        raise ValueError("В сырых прогнозах обнаружены NaN/Inf")
    prediction = np.clip(raw_prediction, *bounds)
    truth = frame["soh"].to_numpy()
    rows = (
        frame.select("fragment_id", "cell_id", "soh")
        .with_columns(
            pl.Series("raw_pred", raw_prediction),
            pl.Series("soh_pred", prediction),
            pl.col("cell_id").str.extract(r"^([^_]+)").alias("source"),
        )
        .with_columns(
            pl.Series("signed_error_pp", 100 * (prediction - truth)),
            pl.Series("abs_error_pp", 100 * np.abs(prediction - truth)),
        )
    )
    report = {
        **regression_metrics(truth, prediction),
        "raw_metrics": regression_metrics(truth, raw_prediction),
        "n_clipped": int(np.count_nonzero(raw_prediction != prediction)),
        "by_source": _group_metrics(rows, "source"),
        "by_cell": _group_metrics(rows, "cell_id"),
    }
    deciles = np.minimum(
        np.searchsorted(np.quantile(truth, np.linspace(0, 1, 11))[1:-1], truth, side="right"), 9
    )
    report["by_soh_decile"] = _group_metrics(
        rows.with_columns(pl.Series("soh_decile", deciles)), "soh_decile"
    )
    return rows, report


def _group_metrics(frame: pl.DataFrame, key: str) -> list[dict[str, object]]:
    """Посчитать метрики отдельно по источникам, ячейкам или SOH-группам."""
    result = []
    for value, group in frame.group_by(key, maintain_order=True):
        result.append(
            {
                key: value[0],
                "soh_min": float(group["soh"].min()),
                "soh_max": float(group["soh"].max()),
                **regression_metrics(group["soh"].to_numpy(), group["soh_pred"].to_numpy()),
            }
        )
    return result
