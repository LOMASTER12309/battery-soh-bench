"""Подбор ElasticNet на train, отложенная оценка val и финальное обучение."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import itertools
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import polars as pl
import yaml
from features import FEATURE_VERSION, make_feature_matrix
from modeling import fit_pipeline, prediction_report, regression_metrics
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits

TEAM_DIR = Path(__file__).resolve().parent
RESEARCH_DIR = TEAM_DIR / "research"
ARTIFACT_DIR = TEAM_DIR / "artifacts"


def _write_json(path: Path, value: object) -> None:
    """Записать отчёт без NaN, сохранив русские пояснения в UTF-8."""
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def _check_labeled_data(train: pl.DataFrame, val: pl.DataFrame) -> None:
    """Проверить метки, идентификаторы и разделение физических ячеек."""
    for name, frame in (("train", train), ("val", val)):
        required = {"fragment_id", "cell_id", "soh"}
        if not required.issubset(frame.columns) or frame.is_empty():
            raise ValueError(f"Неполная размеченная выборка {name}")
        if frame["fragment_id"].null_count() or frame["fragment_id"].n_unique() != frame.height:
            raise ValueError(f"Неуникальные или пустые fragment_id в {name}")
        if frame["cell_id"].null_count() or not np.isfinite(frame["soh"].to_numpy()).all():
            raise ValueError(f"Пустые cell_id или неконечные метки в {name}")
    if set(train["cell_id"]) & set(val["cell_id"]):
        raise ValueError("Одна ячейка присутствует одновременно в train и val")
    if set(train["fragment_id"]) & set(val["fragment_id"]):
        raise ValueError("Одинаковые fragment_id в train и val")


def _cross_validate(
    matrix: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    parameters: dict[str, object],
    settings: dict[str, object],
) -> tuple[dict[str, object], np.ndarray, list[dict[str, object]]]:
    """Получить прогноз для каждой строки только от модели без её ячейки."""
    raw_oof = np.empty_like(target)
    fold_reports, iterations = [], []
    bounds = settings["prediction_bounds"]
    for index, (fit_indices, check_indices) in enumerate(folds):
        if set(groups[fit_indices]) & set(groups[check_indices]):
            raise ValueError("Утечка ячейки внутри CV")
        pipeline = fit_pipeline(
            matrix[fit_indices], target[fit_indices], groups[fit_indices], parameters, settings
        )
        raw = pipeline.predict(matrix[check_indices])
        if not np.isfinite(raw).all():
            raise ValueError("Неконечный прогноз CV")
        raw_oof[check_indices] = raw
        iterations.append(int(pipeline.named_steps["elasticnet"].n_iter_))
        fold_reports.append(
            {"fold": index, **regression_metrics(target[check_indices], np.clip(raw, *bounds))}
        )
    report = {
        **parameters,
        **regression_metrics(target, np.clip(raw_oof, *bounds)),
        "n_features": matrix.shape[1],
        "n_clipped": int(np.count_nonzero(raw_oof != np.clip(raw_oof, *bounds))),
        "worst_fold_mae_pp": max(item["mae_pp"] for item in fold_reports),
        "max_iterations": max(iterations),
        "converged": max(iterations) < int(settings["max_iter"]),
    }
    return report, raw_oof, fold_reports


def _source_holdouts(
    train: pl.DataFrame,
    matrix: np.ndarray,
    parameters: dict[str, object],
    settings: dict[str, object],
) -> list[dict[str, object]]:
    """Диагностика переноса выбранной модели между источниками внутри train."""
    sources = train["cell_id"].str.extract(r"^([^_]+)").to_numpy()
    target, groups = train["soh"].to_numpy(), train["cell_id"].to_numpy()
    reports = []
    for source in sorted(set(sources)):
        check = sources == source
        pipeline = fit_pipeline(
            matrix[~check], target[~check], groups[~check], parameters, settings
        )
        raw = pipeline.predict(matrix[check])
        if not np.isfinite(raw).all():
            raise ValueError("Неконечный прогноз при проверке переноса между источниками")
        prediction = np.clip(raw, *settings["prediction_bounds"])
        reports.append(
            {
                "source": source,
                "converged": pipeline.named_steps["elasticnet"].n_iter_ < settings["max_iter"],
                "n_clipped": int(np.count_nonzero(raw != prediction)),
                **regression_metrics(target[check], prediction),
            }
        )
    return reports


def run_training(config_path: Path) -> dict[str, object]:
    """Воспроизвести поиск, отчёты и модель без использования скрытых меток test."""
    settings = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if settings["final_fit"] != "train+val":
        raise ValueError("Этот протокол ожидает финальное обучение train+val")
    bounds = settings["prediction_bounds"]
    if not (0 < bounds[0] < bounds[1] <= 1.2):
        raise ValueError("Некорректный заранее заданный диапазон прогноза")
    data_dir = (config_path.parent / settings["data_dir"]).resolve()
    train_path, val_path = data_dir / "train.parquet", data_dir / "val.parquet"
    train, val = pl.read_parquet(train_path), pl.read_parquet(val_path)
    _check_labeled_data(train, val)
    RESEARCH_DIR.mkdir(exist_ok=True)
    ARTIFACT_DIR.mkdir(exist_ok=True)
    target, groups = train["soh"].to_numpy(), train["cell_id"].to_numpy()
    folds = list(GroupKFold(n_splits=int(settings["cv_splits"])).split(target, groups=groups))
    fold_ids = np.empty(train.height, dtype=int)
    for index, (_, indices) in enumerate(folds):
        fold_ids[indices] = index
    train.select("fragment_id", "cell_id").with_columns(pl.Series("fold", fold_ids)).write_csv(
        RESEARCH_DIR / "cv_folds.csv"
    )
    matrices = {
        feature_set: make_feature_matrix(train, feature_set)
        for feature_set in settings["feature_sets"]
    }
    combinations = list(
        itertools.product(
            settings["feature_sets"],
            settings["alphas"],
            settings["l1_ratios"],
            settings["weighting"],
        )
    )
    candidates, best = [], None
    best_raw, best_folds = None, None
    print(f"Searching {len(combinations)} candidates on {len(folds)} grouped folds", flush=True)
    for index, (feature_set, alpha, l1_ratio, weighting) in enumerate(combinations):
        parameters = {
            "feature_set": feature_set,
            "alpha": alpha,
            "l1_ratio": l1_ratio,
            "weighting": weighting,
        }
        report, raw_oof, fold_reports = _cross_validate(
            matrices[feature_set][0], target, groups, folds, parameters, settings
        )
        candidates.append(report)
        if report["converged"] and (best is None or report["mae_pp"] < best["mae_pp"]):
            best, best_raw, best_folds = report, raw_oof, fold_reports
        if (index + 1) % 12 == 0 or index + 1 == len(combinations):
            score = best["mae_pp"] if best else float("nan")
            print(f"Finished {index + 1}/{len(combinations)}; best OOF MAE={score:.4f}", flush=True)
    if best is None:
        raise RuntimeError("Ни один кандидат не сошёлся: увеличьте max_iter")
    pl.DataFrame(candidates).sort("mae_pp").write_csv(RESEARCH_DIR / "cv_results.csv")
    parameters = {key: best[key] for key in ("feature_set", "alpha", "l1_ratio", "weighting")}
    _write_json(TEAM_DIR / "selected_model.json", parameters)
    oof_rows, oof_report = prediction_report(train, best_raw, bounds)
    oof_rows.write_csv(RESEARCH_DIR / "oof_predictions.csv")
    feature_set = str(parameters["feature_set"])
    matrix, feature_names = matrices[feature_set]
    pipeline = fit_pipeline(matrix, target, groups, parameters, settings)
    if pipeline.named_steps["elasticnet"].n_iter_ >= int(settings["max_iter"]):
        raise RuntimeError("Модель на всём train не сошлась: оценку val не выполняем")
    # Только после окончательного выбора по train начинаем оценку val.
    val_matrix, val_names = make_feature_matrix(val, feature_set)
    if feature_names != val_names:
        raise ValueError("Разный порядок признаков train и val")
    validation_rows, validation = prediction_report(val, pipeline.predict(val_matrix), bounds)
    train_mean = float(target.mean())
    validation_rows.with_columns(pl.lit(train_mean).alias("constant_pred")).write_csv(
        RESEARCH_DIR / "validation_predictions.csv"
    )
    baselines = {
        "train_mean": regression_metrics(val["soh"].to_numpy(), np.full(val.height, train_mean)),
        "train_median": regression_metrics(
            val["soh"].to_numpy(), np.full(val.height, float(np.median(target)))
        ),
    }
    reference_parameters = {
        "feature_set": "baseline",
        "alpha": 0.01,
        "l1_ratio": 0.5,
        "weighting": "none",
    }
    reference = fit_pipeline(
        matrices["baseline"][0], target, groups, reference_parameters, settings
    )
    reference_val, _ = make_feature_matrix(val, "baseline")
    baselines["reference_elasticnet"] = regression_metrics(
        val["soh"].to_numpy(), np.clip(reference.predict(reference_val), *bounds)
    )
    coefficients = pipeline.named_steps["elasticnet"].coef_
    used_names = np.asarray(feature_names)[pipeline.named_steps["variance"].get_support()]
    pl.DataFrame({"feature": used_names.tolist(), "standardized_coefficient": coefficients}).sort(
        "standardized_coefficient", descending=True
    ).write_csv(RESEARCH_DIR / "coefficients.csv")
    report = {
        "protocol": {
            "selection_data": "train only",
            "selection_metric": "pooled OOF MAE in percentage points",
            "splitter": "GroupKFold(shuffle=False), groups=cell_id",
            "cv_splits": len(folds),
            "n_candidates": len(candidates),
            "n_nonconverged": sum(not item["converged"] for item in candidates),
            "validation_used_for_selection": False,
            "final_fit": "train+val",
            "prediction_bounds": bounds,
            "seed": int(settings["seed"]),
        },
        "selected_parameters": parameters,
        "n_features": len(feature_names),
        "n_nonzero_coefficients": int(np.count_nonzero(coefficients)),
        "train_rows": train.height,
        "train_cells": train["cell_id"].n_unique(),
        "val_rows": val.height,
        "val_cells": val["cell_id"].n_unique(),
        "train_mean_soh": train_mean,
        "cv": oof_report,
        "cv_folds": best_folds,
        "validation": validation,
        "baselines_on_val": baselines,
        "training_source_holdouts": _source_holdouts(train, matrix, parameters, settings),
        "data_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (train_path, val_path)
        },
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "polars", "scipy", "scikit-learn", "joblib", "PyYAML")
        },
        "limitations": [
            "Validation is dominated by CALCE and is not the official NCA/NMC or LFP test.",
            "Validation scores describe the train-only model, not the refitted final model.",
            "No test labels, chemistry labels, intervals or economy nomination are used.",
        ],
    }
    final_data = pl.concat([train, val])
    final_matrix, final_names = make_feature_matrix(final_data, feature_set)
    final_pipeline = fit_pipeline(
        final_matrix,
        final_data["soh"].to_numpy(),
        final_data["cell_id"].to_numpy(),
        parameters,
        settings,
    )
    if final_pipeline.named_steps["elasticnet"].n_iter_ >= int(settings["max_iter"]):
        raise RuntimeError("Финальная модель не сошлась")
    report["final_model"] = {
        "rows": final_data.height,
        "cells": final_data["cell_id"].n_unique(),
        "iterations": int(final_pipeline.named_steps["elasticnet"].n_iter_),
    }
    joblib.dump(
        {
            "pipeline": final_pipeline,
            "feature_set": feature_set,
            "feature_names": final_names,
            "prediction_bounds": bounds,
            "metadata": {
                "feature_version": FEATURE_VERSION,
                "parameters": parameters,
                "training_data_sha256": report["data_sha256"],
                "final_fit": "train+val",
                "versions": report["versions"],
            },
        },
        ARTIFACT_DIR / "model.joblib",
        compress=3,
    )
    _write_json(TEAM_DIR / "validation_metrics.json", report)
    print(
        json.dumps(
            {
                "selected": parameters,
                "cv_mae_pp": best["mae_pp"],
                "val": {
                    key: validation[key]
                    for key in ("mae_pp", "p95_abs_pp", "bias_pp", "max_abs_pp", "n_clipped")
                },
                "final_fit": report["final_model"],
            },
            indent=2,
        )
    )
    return report


def main() -> None:
    """Обработать CLI и ограничить численные библиотеки одним CPU-потоком."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=TEAM_DIR / "config.yaml")
    args = parser.parse_args()
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    with threadpool_limits(limits=1):
        run_training(args.config.resolve())


if __name__ == "__main__":
    main()
