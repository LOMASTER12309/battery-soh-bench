"""Детерминированные признаки одного зарядного окна без ID и истории ячейки."""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy.signal import savgol_filter

FEATURE_SETS = ("baseline", "shape", "compact", "hybrid")
CHANNELS = ("q_norm", "current", "temperature")
REQUIRED_COLUMNS = ("v_start", "v_end", *CHANNELS)
FEATURE_VERSION = 1


def _row_features(row: dict[str, object]) -> dict[str, dict[str, float]]:
    """Извлечь блоки масштаба, формы и условий из одного фрагмента."""
    start, end = float(row["v_start"]), float(row["v_end"])
    if not np.isfinite([start, end]).all() or not np.isclose(end - start, 0.2, atol=1e-6):
        raise ValueError("Ожидалось конечное окно напряжения шириной 0.2 В")
    arrays = [np.asarray(row[channel], dtype=np.float64) for channel in CHANNELS]
    if any(array.shape != (128,) or not np.isfinite(array).all() for array in arrays):
        raise ValueError("Каждый канал должен содержать 128 конечных чисел")
    charge, current, temperature = arrays
    voltage = np.linspace(start, end, 128)
    position = np.linspace(0.0, 1.0, 128)
    smooth = savgol_filter(charge, window_length=11, polyorder=3)
    derivative = np.gradient(smooth, voltage)
    span = float(smooth[-1] - smooth[0])
    shape = (smooth - smooth[0]) / max(abs(span), 1e-8)
    shape_derivative = np.gradient(shape, position)
    average_current, average_temperature = float(current.mean()), float(temperature.mean())
    baseline = {
        "v_start": start,
        "v_end": end,
        "v_width": end - start,
        "q_mean": float(charge.mean()),
        "q_std": float(charge.std()),
        "q_min": float(charge.min()),
        "q_max": float(charge.max()),
        "q_last": float(charge[-1]),
        "q_integral": float(np.trapezoid(charge, voltage)),
        "q_slope": float((charge[-1] - charge[0]) / (end - start)),
        "dqdv_mean": float(derivative.mean()),
        "dqdv_std": float(derivative.std()),
        "dqdv_min": float(derivative.min()),
        "dqdv_max": float(derivative.max()),
        "dqdv_mid": float(derivative[64]),
        "dqdv_step": (end - start) / 127,
        "current_mean": average_current,
        "current_abs_max": float(np.abs(current).max()),
        "current_std": float(current.std()),
        "temperature_mean": average_temperature,
        "temperature_std": float(temperature.std()),
        "temperature_min": float(temperature.min()),
        "temperature_max": float(temperature.max()),
    }
    context = {
        key: baseline[key]
        for key in (
            "v_start",
            "v_end",
            "current_mean",
            "current_std",
            "temperature_mean",
            "temperature_std",
        )
    }
    centered_voltage = (start + end) / 2 - 3.5
    scaled_temperature = (average_temperature - 25) / 20
    context.update(
        {
            "voltage_squared": centered_voltage**2,
            "current_log": float(np.log1p(abs(average_current))),
            "temperature_squared": scaled_temperature**2,
            "voltage_current": centered_voltage * average_current,
            "voltage_temperature": centered_voltage * scaled_temperature,
            "current_temperature": average_current * scaled_temperature,
        }
    )
    normalized = {
        f"shape_q_{index}": float(np.interp(fraction, position, shape))
        for index, fraction in enumerate(np.linspace(0.1, 0.9, 9))
    }
    normalized.update(
        {
            "shape_integral": float(np.trapezoid(shape, position)),
            "shape_std": float(shape.std()),
            "shape_derivative_std": float(shape_derivative.std()),
            "shape_derivative_q10": float(np.quantile(shape_derivative, 0.1)),
            "shape_derivative_median": float(np.median(shape_derivative)),
            "shape_derivative_q90": float(np.quantile(shape_derivative, 0.9)),
            "shape_residual_mean": float(np.mean(shape - position)),
            "shape_residual_std": float(np.std(shape - position)),
            "flat_window": float(abs(span) < 1e-8),
        }
    )
    shape_block = {**context, **normalized}
    compact = {
        **shape_block,
        "q_span": span,
        "q_span_log": float(np.log(max(abs(span), 1e-8))),
        "q_offset": float(smooth[0]),
        "q_span_voltage": span * centered_voltage,
        "q_span_current": span * average_current,
    }
    hybrid = {**baseline, **shape_block, **compact}
    return {"baseline": baseline, "shape": shape_block, "compact": compact, "hybrid": hybrid}


def make_feature_matrix(frame: pl.DataFrame, feature_set: str) -> tuple[np.ndarray, list[str]]:
    """Вернуть конечную матрицу признаков и фиксированный порядок их названий.

    Все операции используют только один фрагмент. Обучаемая стандартизация
    выполняется отдельно внутри sklearn Pipeline на обучающей части фолда.
    Дополнительные колонки, включая SOH и идентификаторы, не читаются.
    """
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Неизвестный набор признаков: {feature_set}")
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing or frame.is_empty():
        raise ValueError(f"Пустые данные или отсутствующие измерения: {sorted(missing)}")
    records = [
        _row_features(row)[feature_set]
        for row in frame.select(REQUIRED_COLUMNS).iter_rows(named=True)
    ]
    names = list(records[0])
    matrix = np.asarray([[record[name] for name in names] for record in records])
    if not np.isfinite(matrix).all():
        raise ValueError("Не удалось получить конечные признаки")
    return matrix, names
