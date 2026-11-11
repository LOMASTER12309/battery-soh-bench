"""Create a contest CSV from the saved ElasticNet pipeline and public fragments."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
from features import make_feature_matrix
from sklearn.pipeline import Pipeline

TEAM_DIR = Path(__file__).resolve().parent
REPO_DIR = TEAM_DIR.parents[1]
MAX_SUBMISSION_BYTES = 250_000


def load_model(path: Path) -> dict[str, Any]:
    """Load the fitted pipeline and validate its feature and prediction contracts."""
    bundle = joblib.load(path)
    if not isinstance(bundle, dict):
        raise ValueError("The model file must contain a model bundle dictionary")
    if not isinstance(bundle.get("pipeline"), Pipeline):
        raise ValueError("The model bundle must contain a fitted sklearn Pipeline")
    if not isinstance(bundle.get("feature_set"), str) or not bundle["feature_set"]:
        raise ValueError("The model bundle is missing its feature_set name")
    names = bundle.get("feature_names")
    if (
        not isinstance(names, list)
        or not names
        or not all(isinstance(name, str) and name for name in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError("feature_names must be a nonempty list of unique feature names")
    _prediction_bounds(bundle)
    return bundle


def _prediction_bounds(bundle: dict[str, Any]) -> tuple[float, float]:
    bounds = np.asarray(bundle.get("prediction_bounds"), dtype=float)
    if bounds.shape != (2,) or not np.isfinite(bounds).all():
        raise ValueError("prediction_bounds must contain two finite numbers")
    lower, upper = (float(value) for value in bounds)
    if not 0 < lower < upper <= 1.2:
        raise ValueError("Prediction bounds must satisfy 0 < lower < upper <= 1.2")
    return lower, upper


def predict_frame(frame: pl.DataFrame, bundle: dict[str, Any]) -> pl.DataFrame:
    """Predict finite SOH values while preserving the input fragment identifiers."""
    if frame.height == 0:
        raise ValueError("The input fragment table is empty")
    if "fragment_id" not in frame.columns or frame.schema["fragment_id"] != pl.String:
        raise ValueError("The input must contain a String column named fragment_id")
    identifiers = frame["fragment_id"]
    if identifiers.null_count() or (identifiers.str.strip_chars() == "").any():
        raise ValueError("fragment_id values must not be null or empty")
    if identifiers.n_unique() != frame.height:
        raise ValueError("fragment_id values must be unique")

    matrix, names = make_feature_matrix(frame, bundle["feature_set"])
    if names != bundle["feature_names"]:
        raise ValueError("Input feature names or ordering differ from the saved model")
    if matrix.shape != (frame.height, len(names)) or not np.isfinite(matrix).all():
        raise ValueError("Feature extraction must produce a finite matrix of the expected shape")
    raw = np.asarray(bundle["pipeline"].predict(matrix), dtype=float)
    if raw.shape != (frame.height,) or not np.isfinite(raw).all():
        raise ValueError("The model returned nonfinite predictions or an unexpected shape")

    prediction = np.clip(raw, *_prediction_bounds(bundle))
    return pl.DataFrame({"fragment_id": identifiers, "soh_pred": prediction})


def save_submission(frame: pl.DataFrame, output: Path) -> int:
    """Write a contest-sized CSV and return its size in bytes."""
    content = frame.write_csv(float_precision=10).encode("utf-8")
    if len(content) > MAX_SUBMISSION_BYTES:
        raise ValueError(f"The submission CSV must be at most {MAX_SUBMISSION_BYTES} bytes")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(content)
    return len(content)


def main(argv: Sequence[str] | None = None) -> int:
    """Load a saved model, predict public fragments, and write submission.csv."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=TEAM_DIR / "artifacts" / "model.joblib")
    parser.add_argument(
        "--input", type=Path, default=REPO_DIR / "data" / "fragments" / "test_unlabeled.parquet"
    )
    parser.add_argument("--output", type=Path, default=TEAM_DIR / "submission.csv")
    args = parser.parse_args(argv)
    try:
        if args.output.resolve() in {args.input.resolve(), args.model.resolve()}:
            raise ValueError("The output path must differ from both the input and model paths")
        bundle = load_model(args.model)
        result = predict_frame(pl.read_parquet(args.input), bundle)
        size = save_submission(result, args.output)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Saved {result.height} predictions ({size} bytes) to {args.output}")
    print(f"SOH range: {result['soh_pred'].min():.6f} .. {result['soh_pred'].max():.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
