"""Spatial analysis: IDW weights, Moran's I, and hotspot detection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from dallas_crime.pipeline.analyze.core import (
    FDR_ALPHA,
    SPATIAL_METRICS,
    SPATIAL_PRACTICAL_EFFECT_THRESHOLD,
    _bh_adjust_series,
)


def _inverse_distance_weights(coordinates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    deltas = coordinates[:, None, :] - coordinates[None, :, :]
    distances = np.linalg.norm(deltas, axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        weights = np.where(distances > 0, 1.0 / distances, 0.0)
    row_sums = weights.sum(axis=1, keepdims=True)
    weights = np.where(row_sums > 0, weights / row_sums, 0.0)
    return weights, distances


def _morans_i(values: np.ndarray, weights: np.ndarray) -> float | None:
    centered = values - values.mean()
    denominator = float(np.square(centered).sum())
    s0 = float(weights.sum())
    if denominator == 0 or s0 == 0:
        return None
    numerator = float(centered @ weights @ centered)
    return (len(values) / s0) * (numerator / denominator)


def _build_spatial_artifacts(model_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"zip", "centroid_latitude", "centroid_longitude"}
    if not required <= set(model_df.columns):
        diagnostics = pd.DataFrame(
            columns=[
                "metric",
                "zip_count",
                "morans_i",
                "permutation_p_value",
                "spatial_lag_correlation",
                "fdr_q_value",
                "passes_fdr_10",
                "practical_effect_value",
                "practical_effect_threshold",
                "passes_practical_effect",
                "interpretation_allowed",
            ]
        )
        hotspots = pd.DataFrame(
            columns=[
                "zip",
                "total_rate_per_1000",
                "crime_spatial_lag",
                "crime_spatial_quadrant",
                "nearest_neighbor_distance",
            ]
        )
        return diagnostics, hotspots

    base = model_df[["zip", "centroid_latitude", "centroid_longitude", *SPATIAL_METRICS]].copy()
    for column in ("centroid_latitude", "centroid_longitude", *SPATIAL_METRICS):
        if column in base.columns:
            base[column] = pd.to_numeric(base[column], errors="coerce")
    base = base.dropna(subset=["zip", "centroid_latitude", "centroid_longitude"]).reset_index(drop=True)
    if len(base) < 3:
        return _build_spatial_artifacts(pd.DataFrame())

    coordinates = base[["centroid_latitude", "centroid_longitude"]].to_numpy(dtype=float)
    full_weights, full_distances = _inverse_distance_weights(coordinates)
    rng = np.random.default_rng(20260311)
    diagnostic_rows: list[dict[str, object]] = []
    hotspot_rows: list[dict[str, object]] = []

    for metric in SPATIAL_METRICS:
        if metric not in base.columns:
            continue
        working = base[["zip", "centroid_latitude", "centroid_longitude", metric]].dropna().reset_index()
        if len(working) < 3:
            continue
        subset_index = working["index"].to_numpy(dtype=int)
        weights = full_weights[np.ix_(subset_index, subset_index)]
        distances = full_distances[np.ix_(subset_index, subset_index)]
        values = working[metric].to_numpy(dtype=float)
        morans_i = _morans_i(values, weights)
        spatial_lag = weights @ values
        lag_correlation = (
            float(np.corrcoef(values, spatial_lag)[0, 1]) if len(working) > 2 else np.nan
        )
        permutation_stats = [
            _morans_i(rng.permutation(values), weights) for _ in range(199) if morans_i is not None
        ]
        permutation_p_value = (
            (
                sum(abs(stat) >= abs(morans_i) for stat in permutation_stats if stat is not None) + 1
            )
            / (len(permutation_stats) + 1)
            if morans_i is not None and permutation_stats
            else np.nan
        )
        diagnostic_rows.append(
            {
                "metric": metric,
                "zip_count": int(len(working)),
                "morans_i": morans_i,
                "permutation_p_value": permutation_p_value,
                "spatial_lag_correlation": lag_correlation,
            }
        )

        if metric == "total_rate_per_1000":
            centered = values - values.mean()
            lag_centered = spatial_lag - spatial_lag.mean()
            for index, row in working.iterrows():
                if centered[index] >= 0 and lag_centered[index] >= 0:
                    quadrant = "high-high"
                elif centered[index] < 0 and lag_centered[index] < 0:
                    quadrant = "low-low"
                elif centered[index] >= 0:
                    quadrant = "high-low"
                else:
                    quadrant = "low-high"
                nearest_neighbor_distance = float(
                    np.min(distances[index, distances[index] > 0])
                ) if np.any(distances[index] > 0) else np.nan
                hotspot_rows.append(
                    {
                        "zip": row["zip"],
                        "total_rate_per_1000": float(row[metric]),
                        "crime_spatial_lag": float(spatial_lag[index]),
                        "crime_spatial_quadrant": quadrant,
                        "nearest_neighbor_distance": nearest_neighbor_distance,
                    }
                )

    diagnostics = pd.DataFrame.from_records(
        diagnostic_rows,
        columns=[
            "metric",
            "zip_count",
                "morans_i",
                "permutation_p_value",
                "spatial_lag_correlation",
                "fdr_q_value",
                "passes_fdr_10",
                "practical_effect_value",
                "practical_effect_threshold",
                "passes_practical_effect",
                "interpretation_allowed",
            ],
        )
    if not diagnostics.empty:
        diagnostics["fdr_q_value"] = _bh_adjust_series(diagnostics["permutation_p_value"])
        diagnostics["passes_fdr_10"] = (
            diagnostics["fdr_q_value"].notna() & (diagnostics["fdr_q_value"] <= FDR_ALPHA)
        ).astype(int)
        diagnostics["practical_effect_value"] = diagnostics["morans_i"].abs()
        diagnostics["practical_effect_threshold"] = SPATIAL_PRACTICAL_EFFECT_THRESHOLD
        diagnostics["passes_practical_effect"] = (
            diagnostics["practical_effect_value"] >= SPATIAL_PRACTICAL_EFFECT_THRESHOLD
        ).astype(int)
        diagnostics["interpretation_allowed"] = (
            (diagnostics["passes_fdr_10"] == 1) & (diagnostics["passes_practical_effect"] == 1)
        ).astype(int)
        diagnostics = diagnostics.sort_values("metric", ignore_index=True)

    hotspots = pd.DataFrame.from_records(
        hotspot_rows,
        columns=[
            "zip",
            "total_rate_per_1000",
            "crime_spatial_lag",
            "crime_spatial_quadrant",
            "nearest_neighbor_distance",
        ],
    )
    if not hotspots.empty:
        hotspots = hotspots.sort_values("zip", ignore_index=True)
    return diagnostics, hotspots
