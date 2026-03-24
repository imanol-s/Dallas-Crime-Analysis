"""Spatial analysis: IDW weights, Moran's I, hotspot detection, and spatial lag regression."""

from __future__ import annotations

import numpy as np
import pandas as pd

from dallas_crime.pipeline.analyze.core import (
    DEFAULT_CONTROLS,
    DEFAULT_PREDICTORS,
    FDR_ALPHA,
    SPATIAL_METRICS,
    SPATIAL_PRACTICAL_EFFECT_THRESHOLD,
    RegressionResult,
    _bh_adjust_series,
    _coerce_model_columns,
    _ensure_dependent_column,
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


def _build_spatial_weights(
    model_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame] | None:
    """Extract coordinate-valid rows and compute IDW weights.

    Returns (full_weights, full_distances, base_frame) where base_frame is model_df
    with NaN-coordinate rows dropped and the index reset.  Returns None if fewer than
    3 rows have valid coordinates.
    """
    required = {"zip", "centroid_latitude", "centroid_longitude"}
    if not required <= set(model_df.columns):
        return None

    base = model_df.copy()
    base["centroid_latitude"] = pd.to_numeric(base["centroid_latitude"], errors="coerce")
    base["centroid_longitude"] = pd.to_numeric(base["centroid_longitude"], errors="coerce")
    base = base.dropna(subset=["zip", "centroid_latitude", "centroid_longitude"]).reset_index(
        drop=True
    )

    if len(base) < 3:
        return None

    coordinates = base[["centroid_latitude", "centroid_longitude"]].to_numpy(dtype=float)
    full_weights, full_distances = _inverse_distance_weights(coordinates)
    return full_weights, full_distances, base


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

    weights_result = _build_spatial_weights(model_df)
    if weights_result is None:
        return _build_spatial_artifacts(pd.DataFrame())

    full_weights, full_distances, base = weights_result

    # Coerce spatial metric columns to numeric within the coordinate-valid base frame
    for column in SPATIAL_METRICS:
        if column in base.columns:
            base[column] = pd.to_numeric(base[column], errors="coerce")

    rng = np.random.default_rng(20260311)
    diagnostic_rows: list[dict[str, object]] = []
    hotspot_rows: list[dict[str, object]] = []

    for metric in SPATIAL_METRICS:
        if metric not in base.columns:
            continue
        working = (
            base[["zip", "centroid_latitude", "centroid_longitude", metric]].dropna().reset_index()
        )
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
            (sum(abs(stat) >= abs(morans_i) for stat in permutation_stats if stat is not None) + 1)
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
                nearest_neighbor_distance = (
                    float(np.min(distances[index, distances[index] > 0]))
                    if np.any(distances[index] > 0)
                    else np.nan
                )
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


def _fit_spatial_lag_model(
    model_df: pd.DataFrame,
    *,
    dependent: str = "log_home_value",
    predictors: tuple[str, ...] = DEFAULT_PREDICTORS,
    controls: tuple[str, ...] = DEFAULT_CONTROLS,
    model_label: str = "spatial_lag",
) -> RegressionResult | None:
    """Fit a spatial lag model via spreg.ML_Lag using IDW spatial weights.

    Returns a RegressionResult-compatible object, or None if spreg/libpysal are not
    installed, coordinates are unavailable, or the fit fails for any reason.
    """
    try:
        import libpysal
        import spreg
    except ImportError:
        print(
            "[analyze] NOTE (R4): spreg/libpysal not installed; spatial lag model skipped.",
            flush=True,
        )
        return None

    # Build spatial weights on the full model_df (all coordinate-valid rows)
    weights_result = _build_spatial_weights(model_df)
    if weights_result is None:
        return None

    full_weights, _full_distances, base = weights_result

    # Ensure dependent column exists
    try:
        base = _ensure_dependent_column(base, dependent)
    except KeyError:
        return None

    # Required regression columns
    reg_cols = [dependent, *predictors, *controls]
    if any(c not in base.columns for c in reg_cols):
        return None

    # Coerce numeric and get complete-case rows
    base = _coerce_model_columns(base, reg_cols)
    complete_positions = base[reg_cols].notna().all(axis=1)
    if complete_positions.sum() < len(reg_cols) + 1:
        return None

    # Align the spatial weight submatrix to the complete-case rows
    orig_indices = base.index[complete_positions].to_numpy(dtype=int)
    sub_weights = full_weights[np.ix_(orig_indices, orig_indices)]
    working = base.loc[complete_positions].reset_index(drop=True)

    # Convert dense IDW array to a libpysal W object
    try:
        w = libpysal.weights.full2W(sub_weights)
    except Exception:
        return None

    # Build y (n, 1) and X (n, k) arrays for spreg
    y = working[dependent].to_numpy(dtype=float).reshape(-1, 1)
    x_cols = list([*predictors, *controls])
    x = working[x_cols].to_numpy(dtype=float)

    # Fit ML spatial lag model
    try:
        model = spreg.ML_Lag(
            y,
            x,
            w=w,
            name_y=dependent,
            name_x=x_cols,
        )
    except Exception:
        return None

    # Map spreg output to coefficients DataFrame.
    # ML_Lag betas: [CONSTANT, x1, ..., xk, rho] — length k+2
    betas_raw = np.asarray(model.betas).flatten()
    std_errs_raw = (
        np.asarray(model.std_err).flatten()
        if model.std_err is not None
        else np.full(len(betas_raw), np.nan)
    )
    if model.z_stat is not None:
        z_vals = [float(z) for z, _p in model.z_stat]
        p_vals = [float(p) for _z, p in model.z_stat]
    else:
        z_vals = [np.nan] * len(betas_raw)
        p_vals = [np.nan] * len(betas_raw)

    # Variable names — spreg names include "CONSTANT"; normalise to "Intercept"
    raw_names: list[str] = list(getattr(model, "name_x", [])) or (
        ["CONSTANT"] + x_cols + [f"W_{dependent}"]
    )
    if len(raw_names) != len(betas_raw):
        raw_names = ["CONSTANT"] + x_cols + [f"W_{dependent}"]
    var_names = ["Intercept" if n == "CONSTANT" else n for n in raw_names]

    conf_low = (betas_raw - 1.96 * std_errs_raw).tolist()
    conf_high = (betas_raw + 1.96 * std_errs_raw).tolist()

    # OLS-compatible formula is stored so downstream functions that call smf.ols(formula=...)
    # can produce valid OLS diagnostics on the spatial model's working frame.
    ols_formula = f"{dependent} ~ {' + '.join(x_cols)}"
    display_formula = f"ML_Lag: {dependent} ~ {' + '.join(x_cols)} (+ spatial lag)"

    coefficients = pd.DataFrame(
        {
            "model_label": model_label,
            "dependent_variable": dependent,
            "formula": display_formula,
            "term": var_names,
            "estimate": betas_raw.tolist(),
            "std_error": std_errs_raw.tolist(),
            "t_value": z_vals,
            "p_value": p_vals,
            "conf_low": conf_low,
            "conf_high": conf_high,
        }
    )

    # Residuals
    resid_arr = np.asarray(model.u).flatten() if model.u is not None else np.zeros(len(working))
    fitted_arr = (
        np.asarray(model.predy).flatten()
        if model.predy is not None
        else np.full(len(working), np.nan)
    )

    residuals_df = (
        working[["zip"]].copy()
        if "zip" in working.columns
        else pd.DataFrame(index=range(len(working)))
    )
    residuals_df["model_label"] = model_label
    residuals_df["observed"] = y.flatten()
    residuals_df["fitted_value"] = fitted_arr
    residuals_df["residual"] = resid_arr
    residuals_df["absolute_residual"] = np.abs(resid_arr)

    pseudo_r2 = float(model.pr2) if hasattr(model, "pr2") and model.pr2 is not None else np.nan

    # Use the OLS-compatible formula in RegressionResult so that _build_validation_artifacts
    # and _build_influence_robustness_artifacts (which call smf.ols(formula=...)) can
    # produce valid OLS-based diagnostics on the spatial model's complete-case working frame.
    return RegressionResult(
        model_label=model_label,
        formula=ols_formula,
        dependent_variable=dependent,
        predictors=predictors,
        controls=controls,
        nobs=int(model.n),
        r_squared=pseudo_r2,
        adjusted_r_squared=pseudo_r2,
        coefficients=coefficients,
        model_frame=working,
        residuals=residuals_df,
    )
