"""Regression utilities and report generation."""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import os
from pathlib import Path
from typing import TYPE_CHECKING

os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".matplotlib").resolve()))

import matplotlib
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.seasonal import seasonal_decompose

if TYPE_CHECKING:
    from dallas_crime.config import Settings

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_PREDICTORS = ("violent_rate_per_1000", "property_rate_per_1000")
DEFAULT_CONTROLS = (
    "median_household_income",
    "poverty_rate",
    "owner_occupied_share",
    "median_gross_rent",
    "educational_attainment",
)
EXPANDED_CONTROL_CANDIDATES = (
    "population_acs",
    "median_rent",
    "annual_change_pct",
    "FHFA_annual_change_pct",
    "realtor_active_listing_count",
    "realtor_median_days_on_market",
    "realtor_pending_ratio",
    "realtor_hist_listing_price_12m_change",
)
SEGMENTATION_FEATURE_GROUPS = {
    "crime": ("total_rate_per_1000", "violent_rate_per_1000", "property_rate_per_1000"),
    "socioeconomic": (
        "median_household_income",
        "poverty_rate",
        "owner_occupied_share",
        "median_gross_rent",
    ),
    "market": ("home_value", "median_rent", "annual_change_pct", "FHFA_annual_change_pct"),
}
SPATIAL_METRICS = ("total_rate_per_1000", "home_value")
SEGMENTATION_NAMES = {
    "crime": ("lower_crime", "mid_crime", "higher_crime"),
    "socioeconomic": ("stressed", "balanced", "advantaged"),
    "market": ("value", "stable", "premium"),
}
FORECAST_MODELS = ("naive_last", "seasonal_naive_4q", "moving_average_4", "linear_trend")
FORECAST_MODEL_FALLBACK_ORDER = (
    "seasonal_naive_4q",
    "moving_average_4",
    "naive_last",
    "linear_trend",
)
SCENARIO_MULTIPLIERS = {
    "baseline": 1.0,
    "stabilization": 0.95,
    "adverse_momentum": 1.1,
    "seasonal_peak": 1.05,
    "systemic_shock": 1.18,
}
FORECAST_HISTORY_MIN_QUARTERS = 12
FORECAST_LIMITED_HISTORY_MIN_QUARTERS = 4
DRIFT_HISTORY_MIN_QUARTERS = 5
TEMPORAL_HOLDOUT_QUARTERS = 4
FDR_ALPHA = 0.10
REGRESSION_PRACTICAL_EFFECT_THRESHOLD_PCT = 1.0
FEATURE_PRACTICAL_CORRELATION_THRESHOLD = 0.10
SPATIAL_PRACTICAL_EFFECT_THRESHOLD = 0.03
CLUSTER_PRACTICAL_SILHOUETTE_THRESHOLD = 0.40
CLUSTER_STABILITY_ARI_THRESHOLD = 0.60
CLUSTER_MIN_SIZE_THRESHOLD = 5
SEGMENTATION_PREPROCESSING_MODES = ("raw", "winsor_5_95", "winsor_10_90")
SEGMENT_SCENARIO_COVERAGE_THRESHOLD = 0.75
SEGMENT_HIGH_INFLUENCE_SHARE_THRESHOLD = 0.20
INFLUENCE_P90_HOME_VALUE_DELTA_THRESHOLD = 15.0
FEATURE_SELECTION_CANDIDATES = (
    *DEFAULT_PREDICTORS,
    *DEFAULT_CONTROLS,
    *EXPANDED_CONTROL_CANDIDATES,
    "total_rate_per_1000",
    "median_rent",
    "annual_change_pct",
    "population",
    "population_acs",
    "rent_burden",
    "vacancy_proxy",
    "educational_attainment",
    "housing_tenure_mix",
    "source_completeness_overall_score",
)


@dataclass(slots=True)
class RegressionResult:
    """Compact regression payload for report generation."""

    model_label: str
    formula: str
    dependent_variable: str
    predictors: tuple[str, ...]
    controls: tuple[str, ...]
    nobs: int
    r_squared: float
    adjusted_r_squared: float
    coefficients: pd.DataFrame
    model_frame: pd.DataFrame
    residuals: pd.DataFrame

    def metrics_table(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "model_label": self.model_label,
                    "dependent_variable": self.dependent_variable,
                    "formula": self.formula,
                    "predictors": ", ".join(self.predictors),
                    "controls": ", ".join(self.controls),
                    "nobs": self.nobs,
                    "r_squared": self.r_squared,
                    "adjusted_r_squared": self.adjusted_r_squared,
                }
            ]
        )


def _coerce_model_columns(model_df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    frame = model_df.copy()
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _minimum_rows(required_columns: list[str]) -> int:
    return len(required_columns) + 1


def _ensure_dependent_column(model_df: pd.DataFrame, dependent: str) -> pd.DataFrame:
    frame = model_df.copy()
    if dependent in frame.columns:
        return frame

    if dependent == "log_home_value" and "home_value" in frame.columns:
        home_values = pd.to_numeric(frame["home_value"], errors="coerce")
        frame["log_home_value"] = np.where(home_values > 0, np.log(home_values), np.nan)
        return frame

    raise KeyError(f"model_df is missing required dependent variable: {dependent}")


def _select_expanded_controls(
    model_df: pd.DataFrame,
    *,
    dependent: str,
    predictors: tuple[str, ...],
    baseline_controls: tuple[str, ...],
) -> tuple[str, ...]:
    frame = _ensure_dependent_column(model_df, dependent)
    base_controls = list(baseline_controls)
    selected_extras: list[str] = []

    for candidate in EXPANDED_CONTROL_CANDIDATES:
        if candidate not in frame.columns:
            continue

        trial_controls = [*base_controls, *selected_extras, candidate]
        trial_columns = [dependent, *predictors, *trial_controls]
        trial_frame = _coerce_model_columns(frame, trial_columns).dropna(subset=trial_columns)
        if len(trial_frame) >= _minimum_rows(trial_columns):
            selected_extras.append(candidate)

    if not selected_extras:
        raise ValueError(
            "unable to build an expanded-controls model: none of the candidate expanded controls "
            f"({', '.join(EXPANDED_CONTROL_CANDIDATES)}) were available with enough complete rows"
        )

    return tuple([*base_controls, *selected_extras])


def run_zip_regression(
    model_df: pd.DataFrame,
    *,
    dependent: str = "log_home_value",
    predictors: tuple[str, ...] = DEFAULT_PREDICTORS,
    controls: tuple[str, ...] = DEFAULT_CONTROLS,
    model_label: str = "baseline",
) -> RegressionResult:
    """Fit a robust OLS model over ZIP-level housing outcomes."""

    frame = _ensure_dependent_column(model_df, dependent)

    required_columns = [dependent, *predictors, *controls]
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        missing_list = ", ".join(missing)
        raise KeyError(f"model_df is missing required regression columns: {missing_list}")

    frame = _coerce_model_columns(frame, required_columns)
    frame = frame.dropna(subset=required_columns).copy()

    minimum_rows = _minimum_rows(required_columns)
    if len(frame) < minimum_rows:
        raise ValueError(
            f"need at least {minimum_rows} complete rows to fit the requested model; got {len(frame)}"
        )

    formula = f"{dependent} ~ {' + '.join([*predictors, *controls])}"
    fitted = smf.ols(formula=formula, data=frame).fit(cov_type="HC3")

    coefficients = pd.DataFrame(
        {
            "model_label": model_label,
            "dependent_variable": dependent,
            "formula": formula,
            "term": fitted.params.index,
            "estimate": fitted.params.values,
            "std_error": fitted.bse.values,
            "t_value": fitted.tvalues.values,
            "p_value": fitted.pvalues.values,
            "conf_low": fitted.conf_int()[0].values,
            "conf_high": fitted.conf_int()[1].values,
        }
    )

    residual_columns = ["zip"] if "zip" in frame.columns else []
    residuals = frame[residual_columns].copy()
    residuals["model_label"] = model_label
    residuals["observed"] = frame[dependent].to_numpy()
    residuals["fitted_value"] = fitted.fittedvalues.to_numpy()
    residuals["residual"] = fitted.resid.to_numpy()
    residuals["absolute_residual"] = residuals["residual"].abs()

    return RegressionResult(
        model_label=model_label,
        formula=formula,
        dependent_variable=dependent,
        predictors=predictors,
        controls=controls,
        nobs=int(fitted.nobs),
        r_squared=float(fitted.rsquared),
        adjusted_r_squared=float(fitted.rsquared_adj),
        coefficients=coefficients,
        model_frame=frame,
        residuals=residuals,
    )


def _load_optional_analysis_inputs(settings: "Settings") -> dict[str, pd.DataFrame]:
    inputs: dict[str, pd.DataFrame] = {}
    for name in ("crime_history_panel", "housing_history_panel"):
        path = settings.processed_dir / f"{name}.csv"
        inputs[name] = pd.read_csv(path) if path.exists() else pd.DataFrame()
    return inputs


def _bh_adjust_series(values: pd.Series) -> pd.Series:
    adjusted = pd.Series(np.nan, index=values.index, dtype=float)
    numeric = pd.to_numeric(values, errors="coerce")
    mask = numeric.notna()
    if mask.any():
        adjusted.loc[mask] = multipletests(
            numeric.loc[mask].to_numpy(dtype=float),
            alpha=FDR_ALPHA,
            method="fdr_bh",
        )[1]
    return adjusted


def _safe_ratio(numerator: float | int, denominator: float | int) -> float:
    if pd.isna(denominator) or float(denominator) == 0.0:
        return np.nan
    return float(numerator) / float(denominator)


def _prepare_temporal_analysis_inputs(
    crime_history_panel: pd.DataFrame,
    *,
    modeled_zips: set[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], list[str]]:
    columns = [
        "zip",
        "observed_quarters",
        "expected_quarters",
        "overall_completeness_ratio",
        "trailing_contiguous_quarters",
        "forecast_gate_pass",
        "drift_gate_pass",
        "first_period_start",
        "latest_period_start",
    ]
    notes: list[str] = []
    required = {"zip", "period_start", "total_rate_per_1000"}
    if crime_history_panel.empty:
        notes.append("Crime history panel was not available for temporal credibility checks.")
        return pd.DataFrame(columns=columns), {}, notes
    if not required <= set(crime_history_panel.columns):
        notes.append(
            "Crime history panel was missing one or more required temporal columns "
            f"({', '.join(sorted(required))})."
        )
        return pd.DataFrame(columns=columns), {}, notes

    panel = crime_history_panel.copy()
    panel["zip"] = panel["zip"].astype("string")
    panel["period_start"] = pd.to_datetime(panel["period_start"], errors="coerce")
    panel["total_rate_per_1000"] = pd.to_numeric(panel["total_rate_per_1000"], errors="coerce")
    panel = panel.dropna(subset=["zip", "period_start", "total_rate_per_1000"]).copy()
    if modeled_zips is not None:
        panel = panel.loc[panel["zip"].isin(sorted(modeled_zips))].copy()
    if panel.empty:
        notes.append("No modeled ZIP had complete quarterly crime history rows after filtering.")
        return pd.DataFrame(columns=columns), {}, notes

    temporal_rows: list[dict[str, object]] = []
    trailing_series: dict[str, pd.DataFrame] = {}
    for zip_code, zip_frame in panel.groupby("zip", sort=True):
        ordered = (
            zip_frame.sort_values("period_start", kind="mergesort", ignore_index=True)
            .drop_duplicates(subset=["period_start"], keep="last")
            .reset_index(drop=True)
        )
        periods = ordered["period_start"].dt.to_period("Q")
        ordinals = periods.astype(int)
        if ordered.empty:
            continue

        diffs = ordinals.diff()
        trailing_count = 1
        for diff in diffs.iloc[1:][::-1]:
            if diff == 1:
                trailing_count += 1
            else:
                break
        trailing_frame = ordered.iloc[-trailing_count:].copy().reset_index(drop=True)
        observed_quarters = int(len(ordered))
        expected_quarters = int(ordinals.iloc[-1] - ordinals.iloc[0] + 1)
        temporal_rows.append(
            {
                "zip": str(zip_code),
                "observed_quarters": observed_quarters,
                "expected_quarters": expected_quarters,
                "overall_completeness_ratio": _safe_ratio(observed_quarters, expected_quarters),
                "trailing_contiguous_quarters": int(len(trailing_frame)),
                "forecast_gate_pass": int(len(trailing_frame) >= FORECAST_HISTORY_MIN_QUARTERS),
                "drift_gate_pass": int(len(trailing_frame) >= DRIFT_HISTORY_MIN_QUARTERS),
                "first_period_start": ordered["period_start"].iloc[0],
                "latest_period_start": ordered["period_start"].iloc[-1],
            }
        )
        trailing_series[str(zip_code)] = trailing_frame

    summary = pd.DataFrame.from_records(temporal_rows, columns=columns)
    if summary.empty:
        notes.append("No ZIP-level quarterly histories were usable after temporal filtering.")
        return summary, trailing_series, notes

    forecast_eligible = int(summary["forecast_gate_pass"].sum())
    drift_eligible = int(summary["drift_gate_pass"].sum())
    gated_out = int(len(summary) - forecast_eligible)
    notes.append(
        f"Forecast gating kept {forecast_eligible} modeled ZIPs with >= {FORECAST_HISTORY_MIN_QUARTERS} "
        f"trailing contiguous quarters and excluded {gated_out} ZIPs."
    )
    notes.append(
        f"Drift gating kept {drift_eligible} modeled ZIPs with >= {DRIFT_HISTORY_MIN_QUARTERS} "
        "trailing contiguous quarters."
    )
    return summary.sort_values("zip", ignore_index=True), trailing_series, notes


def _walk_forward_forecast_metrics(history: np.ndarray, *, model_name: str) -> dict[str, float] | None:
    errors: list[float] = []
    absolute_errors: list[float] = []
    apes: list[float] = []
    for index in range(1, len(history)):
        prediction = _predict_with_model(history[:index], model_name, horizon=1)
        actual = float(history[index])
        if pd.isna(prediction) or pd.isna(actual):
            continue
        error = float(actual - prediction)
        errors.append(error)
        absolute_errors.append(abs(error))
        if actual > 0:
            apes.append(abs(error) / actual * 100.0)

    if not errors:
        return None

    abs_errors = np.asarray(absolute_errors, dtype=float)
    signed_errors = np.asarray(errors, dtype=float)
    return {
        "evaluation_points": float(len(errors)),
        "mae": float(np.mean(abs_errors)),
        "rmse": float(np.sqrt(np.mean(np.square(signed_errors)))),
        "mape": float(np.mean(apes)) if apes else np.nan,
        "error_std": float(np.std(signed_errors, ddof=0)),
        "p75_absolute_error": float(np.quantile(abs_errors, 0.75)),
    }


def _forecast_interval_bounds(
    *,
    prediction: float,
    interval_level: int,
    error_scale: float,
    history: np.ndarray,
    horizon: int,
) -> tuple[float, float]:
    z_score = 1.2816 if interval_level == 80 else 1.96
    clean_history = pd.Series(history, dtype=float).dropna().to_numpy(dtype=float)
    history_scale = float(np.quantile(clean_history, 0.75)) if clean_history.size else abs(prediction)
    history_max = float(np.max(clean_history)) if clean_history.size else abs(prediction)
    upper_cap = max(history_max * 2.0, abs(prediction) * 20.0, prediction + 0.01)
    base_scale = max(error_scale, history_scale * 0.05, abs(prediction) * 0.05, 0.01)
    raw_width = float(z_score * base_scale * np.sqrt(horizon))
    capped_width = min(raw_width, upper_cap - prediction)
    floor_width = min(max(history_scale * 0.02, abs(prediction) * 0.02, 0.01), upper_cap - prediction)
    width = max(capped_width, floor_width)
    lower_bound = max(float(prediction) - width, 0.0)
    upper_bound = min(float(prediction) + width, upper_cap)
    if upper_bound <= lower_bound:
        upper_bound = lower_bound + 0.01
    return float(lower_bound), float(upper_bound)


def _predict_with_model(history: np.ndarray, model_name: str, *, horizon: int) -> float:
    if len(history) == 0:
        return np.nan

    if model_name == "naive_last":
        return float(history[-1])
    if model_name == "seasonal_naive_4q":
        if len(history) < 4:
            return np.nan
        seasonal_index = -4 + ((horizon - 1) % 4)
        return float(history[seasonal_index])
    if model_name == "moving_average_4":
        if len(history) < 4:
            return np.nan
        return float(np.mean(history[-4:]))
    if model_name == "linear_trend":
        if len(history) < 2:
            return np.nan
        slope, intercept = np.polyfit(np.arange(len(history), dtype=float), history, 1)
        return float(intercept + (slope * (len(history) + horizon - 1)))
    raise KeyError(f"unknown forecast model: {model_name}")


def _select_forecast_model(
    zip_metrics: pd.DataFrame,
    *,
    history: np.ndarray,
    allowed_models: tuple[str, ...] | None = None,
) -> str | None:
    eligible_models = tuple(
        model_name
        for model_name in (allowed_models or FORECAST_MODEL_FALLBACK_ORDER)
        if model_name in FORECAST_MODELS
    )
    if not zip_metrics.empty:
        ranked = zip_metrics.copy()
        if eligible_models:
            ranked = ranked.loc[
                ranked["model_name"].astype(str).isin(set(eligible_models))
            ].copy()
        if ranked.empty:
            ranked = zip_metrics.copy()
        ranked["fallback_rank"] = ranked["model_name"].map(FORECAST_MODEL_FALLBACK_ORDER.index)
        ranked = ranked.sort_values(
            ["rmse", "mae", "fallback_rank"],
            ascending=[True, True, True],
            na_position="last",
            kind="mergesort",
        )
        return str(ranked.iloc[0]["model_name"])

    for model_name in eligible_models or FORECAST_MODEL_FALLBACK_ORDER:
        prediction = _predict_with_model(history, model_name, horizon=1)
        if pd.notna(prediction):
            return model_name
    return None


def _build_trend_decomposition_artifacts(
    crime_history_panel: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    columns = [
        "scope",
        "zip",
        "period_start",
        "observed_total_rate_per_1000",
        "trend_total_rate_per_1000",
        "seasonal_total_rate_per_1000",
        "residual_total_rate_per_1000",
        "history_points",
    ]
    required = {"zip", "period_start", "total_rate_per_1000"}
    notes: list[str] = []
    if crime_history_panel.empty:
        notes.append("Crime history panel was not available, so decomposition artifacts are empty.")
        return pd.DataFrame(columns=columns), notes
    if not required <= set(crime_history_panel.columns):
        notes.append(
            "Crime history panel was missing one or more required columns "
            f"({', '.join(sorted(required))}); decomposition artifacts are empty."
        )
        return pd.DataFrame(columns=columns), notes

    panel = crime_history_panel.copy()
    panel["zip"] = panel["zip"].astype("string")
    panel["period_start"] = pd.to_datetime(panel["period_start"], errors="coerce")
    panel["total_rate_per_1000"] = pd.to_numeric(panel["total_rate_per_1000"], errors="coerce")
    panel = panel.dropna(subset=["zip", "period_start", "total_rate_per_1000"]).copy()
    if panel.empty:
        notes.append("Crime history panel had no complete quarterly total-rate rows to decompose.")
        return pd.DataFrame(columns=columns), notes

    series_frames: list[tuple[str, str | None, pd.Series]] = []
    metro_series = (
        panel.groupby("period_start")["total_rate_per_1000"]
        .mean()
        .sort_index(kind="mergesort")
    )
    series_frames.append(("metro", None, metro_series))
    for zip_code, zip_frame in panel.groupby("zip", sort=True):
        zip_series = (
            zip_frame.sort_values("period_start", kind="mergesort")
            .set_index("period_start")["total_rate_per_1000"]
        )
        series_frames.append(("zip", str(zip_code), zip_series))

    rows: list[dict[str, object]] = []
    for scope, zip_code, series in series_frames:
        history_points = int(series.notna().sum())
        label = "metro aggregate" if zip_code is None else f"ZIP {zip_code}"
        if history_points < 8:
            notes.append(f"{label}: fewer than 8 quarterly observations were available.")
            continue

        try:
            decomposition = seasonal_decompose(
                series,
                model="additive",
                period=4,
                extrapolate_trend="freq",
            )
        except ValueError as exc:
            notes.append(f"{label}: decomposition failed ({exc}).")
            continue

        for period_start, observed in series.items():
            rows.append(
                {
                    "scope": scope,
                    "zip": zip_code,
                    "period_start": period_start,
                    "observed_total_rate_per_1000": float(observed),
                    "trend_total_rate_per_1000": float(decomposition.trend.loc[period_start])
                    if pd.notna(decomposition.trend.loc[period_start])
                    else np.nan,
                    "seasonal_total_rate_per_1000": float(
                        decomposition.seasonal.loc[period_start]
                    )
                    if pd.notna(decomposition.seasonal.loc[period_start])
                    else np.nan,
                    "residual_total_rate_per_1000": float(decomposition.resid.loc[period_start])
                    if pd.notna(decomposition.resid.loc[period_start])
                    else np.nan,
                    "history_points": history_points,
                }
            )

    decomposition_df = pd.DataFrame.from_records(rows, columns=columns)
    if decomposition_df.empty and not notes:
        notes.append("No crime-rate series met the minimum history requirement for decomposition.")
    if decomposition_df.empty:
        return decomposition_df, notes
    return decomposition_df.sort_values(
        ["scope", "zip", "period_start"],
        kind="mergesort",
        ignore_index=True,
    ), notes


def _build_forecast_artifacts(
    temporal_summary: pd.DataFrame,
    temporal_series: dict[str, pd.DataFrame],
    *,
    temporal_holdout: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    metric_columns = [
        "model_name",
        "evaluation_points",
        "zip_count",
        "mae",
        "rmse",
        "mape",
        "selected_zip_count",
        "eligible_zip_count",
        "gated_out_zip_count",
        "minimum_history_quarters",
        "temporal_holdout_mape",
        "temporal_holdout_pass",
        "temporal_holdout_rank",
    ]
    forecast_columns = [
        "zip",
        "forecast_period_start",
        "horizon_quarters",
        "selected_model",
        "forecast_tier",
        "policy_eligible",
        "selection_rule",
        "history_points",
        "trailing_contiguous_quarters",
        "overall_completeness_ratio",
        "predicted_total_rate_per_1000",
    ]
    interval_columns = [
        "zip",
        "forecast_period_start",
        "horizon_quarters",
        "selected_model",
        "forecast_tier",
        "policy_eligible",
        "interval_level",
        "forecast_value",
        "lower_bound",
        "upper_bound",
        "interval_width",
        "upper_to_forecast_ratio",
    ]
    notes: list[str] = []
    if temporal_summary.empty or not temporal_series:
        notes.append("Temporal summary inputs were unavailable, so forecast artifacts are empty.")
        return (
            pd.DataFrame(columns=metric_columns),
            pd.DataFrame(columns=forecast_columns),
            pd.DataFrame(columns=interval_columns),
            notes,
        )

    eligible = temporal_summary.loc[
        pd.to_numeric(temporal_summary["forecast_gate_pass"], errors="coerce").fillna(0).astype(int)
        == 1
    ].copy()
    limited_history = temporal_summary.loc[
        pd.to_numeric(temporal_summary["trailing_contiguous_quarters"], errors="coerce")
        .fillna(0)
        .between(FORECAST_LIMITED_HISTORY_MIN_QUARTERS, FORECAST_HISTORY_MIN_QUARTERS - 1)
    ].copy()
    carry_forward = temporal_summary.loc[
        pd.to_numeric(temporal_summary["trailing_contiguous_quarters"], errors="coerce")
        .fillna(0)
        .between(1, FORECAST_LIMITED_HISTORY_MIN_QUARTERS - 1)
    ].copy()
    if eligible.empty:
        notes.append(
            "No modeled ZIP met the trailing-contiguous quarterly history gate for forecasting "
            f"({FORECAST_HISTORY_MIN_QUARTERS} quarters)."
        )
        return (
            pd.DataFrame(columns=metric_columns),
            pd.DataFrame(columns=forecast_columns),
            pd.DataFrame(columns=interval_columns),
            notes,
        )
    notes.append(
        f"Forecasts use only modeled ZIPs with >= {FORECAST_HISTORY_MIN_QUARTERS} trailing contiguous quarters "
        f"({len(eligible)} of {len(temporal_summary)} modeled ZIPs passed)."
    )
    if not limited_history.empty or not carry_forward.empty:
        notes.append(
            "Additional forecast-only coverage is emitted for lower-history ZIPs using explicit "
            "reduced-confidence tiers; those rows are excluded from interval calibration and scenarios."
        )
    holdout_lookup: dict[str, dict[str, float]] = {}
    allowed_models: tuple[str, ...] = ()
    if temporal_holdout is not None and not temporal_holdout.empty:
        candidate_holdout = temporal_holdout.loc[
            temporal_holdout["evaluation_scope"].astype(str) == "candidate_model"
        ].copy()
        if not candidate_holdout.empty:
            candidate_holdout["temporal_holdout_rank"] = candidate_holdout["mape"].rank(
                method="dense",
                na_option="bottom",
            )
            holdout_lookup = {
                str(row.model_name): {
                    "temporal_holdout_mape": float(row.mape) if pd.notna(row.mape) else np.nan,
                    "temporal_holdout_pass": int(row.mape_pass)
                    if pd.notna(row.mape_pass)
                    else 0,
                    "temporal_holdout_rank": float(row.temporal_holdout_rank)
                    if pd.notna(row.temporal_holdout_rank)
                    else np.nan,
                }
                for row in candidate_holdout.itertuples(index=False)
            }
            allowed_models = tuple(
                model_name
                for model_name in FORECAST_MODELS
                if model_name
                in set(
                    candidate_holdout.loc[
                        pd.to_numeric(candidate_holdout["mape_pass"], errors="coerce")
                        .fillna(0)
                        .astype(int)
                        == 1,
                        "model_name",
                    ]
                    .astype(str)
                    .tolist()
                )
            )
    notes.append(
        "Selected ZIP-level forecast families are chosen from the pre-holdout training window used "
        "in temporal holdout validation, then projected forward on the full available history."
    )
    if allowed_models:
        notes.append(
            "ZIP-level model selection is restricted to candidate forecast families that pass aggregate "
            f"temporal holdout ({', '.join(allowed_models)})."
        )
    notes.append(
        "Scenario artifacts remain limited to high-confidence forecasts; lower-history tiers stay "
        "forecast-only and non-policy-eligible."
    )

    per_zip_metric_rows: list[dict[str, object]] = []
    for row in eligible.itertuples(index=False):
        zip_code = str(row.zip)
        ordered = temporal_series.get(zip_code)
        if ordered is None or ordered.empty:
            continue
        values = ordered["total_rate_per_1000"].to_numpy(dtype=float)
        for model_name in FORECAST_MODELS:
            metrics = _walk_forward_forecast_metrics(values, model_name=model_name)
            if metrics is None:
                continue
            per_zip_metric_rows.append(
                {
                    "zip": zip_code,
                    "model_name": model_name,
                    **metrics,
                }
            )

    per_zip_metrics = pd.DataFrame(per_zip_metric_rows)
    selected_models: dict[str, str] = {}
    zip_scales: dict[tuple[str, str], float] = {}
    if not per_zip_metrics.empty:
        per_zip_metrics = per_zip_metrics.sort_values(
            ["zip", "rmse", "mae", "model_name"],
            kind="mergesort",
            ignore_index=True,
        )
        for row in per_zip_metrics.itertuples(index=False):
            zip_scales[(str(row.zip), str(row.model_name))] = max(
                float(row.p75_absolute_error) if pd.notna(row.p75_absolute_error) else 0.0,
                float(row.mae) * 0.75 if pd.notna(row.mae) else 0.0,
                0.01,
            )

    overall_metric_rows: list[dict[str, object]] = []
    overall_scales: dict[str, float] = {}
    for model_name in FORECAST_MODELS:
        model_metrics = per_zip_metrics.loc[per_zip_metrics["model_name"] == model_name].copy()
        if model_metrics.empty:
            overall_metric_rows.append(
                {
                    "model_name": model_name,
                    "evaluation_points": 0,
                    "zip_count": 0,
                    "mae": np.nan,
                    "rmse": np.nan,
                    "mape": np.nan,
                    "selected_zip_count": 0,
                    "eligible_zip_count": int(len(eligible)),
                    "gated_out_zip_count": int(len(temporal_summary) - len(eligible)),
                    "minimum_history_quarters": FORECAST_HISTORY_MIN_QUARTERS,
                    "temporal_holdout_mape": np.nan,
                    "temporal_holdout_pass": 0,
                    "temporal_holdout_rank": np.nan,
                }
            )
            overall_scales[model_name] = 0.01
            continue

        overall_metric_rows.append(
            {
                "model_name": model_name,
                "evaluation_points": int(model_metrics["evaluation_points"].sum()),
                "zip_count": int(model_metrics["zip"].nunique()),
                "mae": float(model_metrics["mae"].mean()),
                "rmse": float(model_metrics["rmse"].mean()),
                "mape": float(model_metrics["mape"].mean()) if model_metrics["mape"].notna().any() else np.nan,
                "selected_zip_count": 0,
                "eligible_zip_count": int(len(eligible)),
                "gated_out_zip_count": int(len(temporal_summary) - len(eligible)),
                "minimum_history_quarters": FORECAST_HISTORY_MIN_QUARTERS,
                "temporal_holdout_mape": holdout_lookup.get(model_name, {}).get(
                    "temporal_holdout_mape",
                    np.nan,
                ),
                "temporal_holdout_pass": holdout_lookup.get(model_name, {}).get(
                    "temporal_holdout_pass",
                    0,
                ),
                "temporal_holdout_rank": holdout_lookup.get(model_name, {}).get(
                    "temporal_holdout_rank",
                    np.nan,
                ),
            }
        )
        overall_scales[model_name] = max(
            float(model_metrics["p75_absolute_error"].mean())
            if model_metrics["p75_absolute_error"].notna().any()
            else 0.0,
            float(model_metrics["mae"].mean()) * 0.75 if model_metrics["mae"].notna().any() else 0.0,
            0.01,
        )

    forecast_rows: list[dict[str, object]] = []
    interval_rows: list[dict[str, object]] = []
    completeness_lookup = temporal_summary.set_index("zip")
    for zip_code in eligible["zip"].astype(str):
        ordered = temporal_series.get(zip_code)
        if ordered is None or ordered.empty:
            continue
        values = ordered["total_rate_per_1000"].to_numpy(dtype=float)
        zip_metrics = (
            per_zip_metrics.loc[per_zip_metrics["zip"] == zip_code].copy()
            if not per_zip_metrics.empty
            else pd.DataFrame()
        )
        selection_history = values[:-TEMPORAL_HOLDOUT_QUARTERS]
        selection_metric_rows: list[dict[str, object]] = []
        selection_scales: dict[str, float] = {}
        if len(selection_history) >= 1:
            for model_name in FORECAST_MODELS:
                metrics = _walk_forward_forecast_metrics(selection_history, model_name=model_name)
                if metrics is None:
                    continue
                selection_metric_rows.append(
                    {
                        "model_name": model_name,
                        "evaluation_points": metrics["evaluation_points"],
                        "mae": metrics["mae"],
                        "rmse": metrics["rmse"],
                        "mape": metrics["mape"],
                    }
                )
                selection_scales[model_name] = max(
                    float(metrics["p75_absolute_error"]),
                    float(metrics["mae"]) * 0.75,
                    0.01,
                )
        selection_metrics = pd.DataFrame(selection_metric_rows)
        selected_model = _select_forecast_model(
            selection_metrics if not selection_metrics.empty else zip_metrics,
            history=selection_history if len(selection_history) >= 1 else values,
            allowed_models=allowed_models or None,
        )
        if selected_model is None:
            notes.append(f"ZIP {zip_code}: no forecast model was applicable for the available history.")
            continue

        selected_models[zip_code] = selected_model
        error_scale = selection_scales.get(
            selected_model,
            zip_scales.get((zip_code, selected_model), overall_scales.get(selected_model, 0.01)),
        )
        last_period = pd.Timestamp(ordered["period_start"].iloc[-1]).to_period("Q")
        for horizon in range(1, 5):
            prediction = _predict_with_model(values, selected_model, horizon=horizon)
            if pd.isna(prediction):
                continue
            forecast_period = (last_period + horizon).start_time
            forecast_rows.append(
                {
                    "zip": zip_code,
                    "forecast_period_start": forecast_period,
                    "horizon_quarters": horizon,
                    "selected_model": selected_model,
                    "forecast_tier": "high_confidence",
                    "policy_eligible": 1,
                    "selection_rule": (
                        "holdout_pass_screened_zip_selection"
                        if allowed_models
                        else "zip_level_history_selection"
                    ),
                    "history_points": int(len(values)),
                    "trailing_contiguous_quarters": int(
                        completeness_lookup.loc[zip_code, "trailing_contiguous_quarters"]
                    ),
                    "overall_completeness_ratio": float(
                        completeness_lookup.loc[zip_code, "overall_completeness_ratio"]
                    ),
                    "predicted_total_rate_per_1000": float(prediction),
                }
            )
            for interval_level in (80, 95):
                lower_bound, upper_bound = _forecast_interval_bounds(
                    prediction=float(prediction),
                    interval_level=interval_level,
                    error_scale=error_scale,
                    history=values,
                    horizon=horizon,
                )
                interval_rows.append(
                    {
                        "zip": zip_code,
                        "forecast_period_start": forecast_period,
                        "horizon_quarters": horizon,
                        "selected_model": selected_model,
                        "forecast_tier": "high_confidence",
                        "policy_eligible": 1,
                        "interval_level": interval_level,
                        "forecast_value": float(prediction),
                        "lower_bound": lower_bound,
                        "upper_bound": upper_bound,
                        "interval_width": float(upper_bound - lower_bound),
                        "upper_to_forecast_ratio": (
                            float(upper_bound / prediction) if float(prediction) > 0 else np.nan
                        ),
                    }
                )

    for tier_name, tier_frame, tier_model, selection_rule in (
        (
            "limited_history",
            limited_history,
            "moving_average_4",
            "limited_history_fixed_moving_average_4",
        ),
        (
            "carry_forward_only",
            carry_forward,
            "naive_last",
            "carry_forward_last_observation",
        ),
    ):
        for row in tier_frame.itertuples(index=False):
            zip_code = str(row.zip)
            ordered = temporal_series.get(zip_code)
            if ordered is None or ordered.empty:
                continue
            values = ordered["total_rate_per_1000"].to_numpy(dtype=float)
            last_period = pd.Timestamp(ordered["period_start"].iloc[-1]).to_period("Q")
            for horizon in range(1, 5):
                prediction = _predict_with_model(values, tier_model, horizon=horizon)
                if pd.isna(prediction):
                    continue
                forecast_period = (last_period + horizon).start_time
                forecast_rows.append(
                    {
                        "zip": zip_code,
                        "forecast_period_start": forecast_period,
                        "horizon_quarters": horizon,
                        "selected_model": tier_model,
                        "forecast_tier": tier_name,
                        "policy_eligible": 0,
                        "selection_rule": selection_rule,
                        "history_points": int(len(values)),
                        "trailing_contiguous_quarters": int(row.trailing_contiguous_quarters),
                        "overall_completeness_ratio": float(row.overall_completeness_ratio),
                        "predicted_total_rate_per_1000": float(prediction),
                    }
                )

    metrics_df = pd.DataFrame.from_records(overall_metric_rows, columns=metric_columns)
    if selected_models:
        selected_counts = pd.Series(selected_models).value_counts()
        metrics_df["selected_zip_count"] = metrics_df["model_name"].map(selected_counts).fillna(0).astype(int)
    if metrics_df["rmse"].notna().any():
        metrics_df["rmse_rank"] = metrics_df["rmse"].rank(method="dense", na_option="bottom")
        metrics_df = metrics_df.sort_values(["rmse_rank", "model_name"], kind="mergesort").drop(
            columns=["rmse_rank"]
        )
    if not forecast_rows and not notes:
        notes.append("No ZIP had enough quarterly history to produce crime forecasts.")

    return (
        metrics_df.reset_index(drop=True),
        pd.DataFrame.from_records(forecast_rows, columns=forecast_columns),
        pd.DataFrame.from_records(interval_rows, columns=interval_columns),
        notes,
    )


def _build_temporal_holdout_artifacts(
    temporal_summary: pd.DataFrame,
    temporal_series: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    holdout_columns = [
        "evaluation_scope",
        "model_name",
        "selected_family",
        "eligible_zip_count",
        "zip_count",
        "holdout_quarters",
        "evaluation_points",
        "mae",
        "rmse",
        "mape",
        "mape_threshold",
        "mape_pass",
    ]
    calibration_columns = [
        "evaluation_scope",
        "model_name",
        "selected_family",
        "interval_level",
        "eligible_zip_count",
        "zip_count",
        "holdout_quarters",
        "evaluation_points",
        "empirical_coverage",
        "target_coverage",
        "coverage_gap",
        "mean_interval_width",
        "median_interval_width",
        "equal_bound_count",
        "equal_bound_share",
        "extreme_upper_ratio_count",
        "extreme_upper_ratio_share",
        "max_upper_to_forecast_ratio",
        "coverage_pass",
        "shape_pass",
        "calibration_pass",
    ]
    notes: list[str] = []
    if temporal_summary.empty or not temporal_series:
        notes.append("Temporal holdout inputs were unavailable, so holdout artifacts are empty.")
        return (
            pd.DataFrame(columns=holdout_columns),
            pd.DataFrame(columns=calibration_columns),
            notes,
        )

    eligible = temporal_summary.loc[
        pd.to_numeric(temporal_summary["forecast_gate_pass"], errors="coerce").fillna(0).astype(int)
        == 1
    ].copy()
    if eligible.empty:
        notes.append(
            "No modeled ZIP met the quarterly history gate required for temporal holdout evaluation."
        )
        return (
            pd.DataFrame(columns=holdout_columns),
            pd.DataFrame(columns=calibration_columns),
            notes,
        )

    holdout_stats: dict[tuple[str, str], dict[str, object]] = {}
    calibration_stats: dict[tuple[str, str, int], dict[str, object]] = {}

    def _update_holdout(scope: str, model_name: str, zip_code: str, error: float, actual: float) -> None:
        stats = holdout_stats.setdefault(
            (scope, model_name),
            {"zips": set(), "errors": [], "abs_errors": [], "apes": []},
        )
        stats["zips"].add(zip_code)
        stats["errors"].append(float(error))
        stats["abs_errors"].append(abs(float(error)))
        if actual > 0:
            stats["apes"].append(abs(float(error)) / float(actual) * 100.0)

    def _update_calibration(
        scope: str,
        model_name: str,
        zip_code: str,
        interval_level: int,
        *,
        prediction: float,
        actual: float,
        lower_bound: float,
        upper_bound: float,
    ) -> None:
        stats = calibration_stats.setdefault(
            (scope, model_name, interval_level),
            {
                "zips": set(),
                "count": 0,
                "hits": 0,
                "widths": [],
                "equal_bound_count": 0,
                "extreme_upper_ratio_count": 0,
                "max_upper_to_forecast_ratio": np.nan,
            },
        )
        ratio = float(upper_bound / prediction) if prediction > 0 else np.nan
        stats["zips"].add(zip_code)
        stats["count"] += 1
        stats["hits"] += int(lower_bound <= actual <= upper_bound)
        stats["widths"].append(float(upper_bound - lower_bound))
        stats["equal_bound_count"] += int(np.isclose(lower_bound, upper_bound))
        stats["extreme_upper_ratio_count"] += int(pd.notna(ratio) and ratio > 100.0)
        if pd.notna(ratio):
            if pd.isna(stats["max_upper_to_forecast_ratio"]):
                stats["max_upper_to_forecast_ratio"] = ratio
            else:
                stats["max_upper_to_forecast_ratio"] = max(
                    float(stats["max_upper_to_forecast_ratio"]),
                    ratio,
                )

    zip_evaluations: list[dict[str, object]] = []
    for zip_code in eligible["zip"].astype(str):
        ordered = temporal_series.get(zip_code)
        if ordered is None or ordered.empty:
            continue
        values = ordered["total_rate_per_1000"].to_numpy(dtype=float)
        if len(values) < TEMPORAL_HOLDOUT_QUARTERS + 1:
            continue
        training_history = values[:-TEMPORAL_HOLDOUT_QUARTERS]
        holdout_actuals = values[-TEMPORAL_HOLDOUT_QUARTERS:]
        training_metric_rows: list[dict[str, object]] = []
        training_scales: dict[str, float] = {}
        for model_name in FORECAST_MODELS:
            metrics = _walk_forward_forecast_metrics(training_history, model_name=model_name)
            if metrics is None:
                continue
            training_metric_rows.append(
                {
                    "model_name": model_name,
                    "evaluation_points": metrics["evaluation_points"],
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "mape": metrics["mape"],
                }
            )
            training_scales[model_name] = max(
                float(metrics["p75_absolute_error"]),
                float(metrics["mae"]) * 0.75,
                0.01,
            )
        training_metrics = pd.DataFrame(training_metric_rows)
        if training_metrics.empty:
            continue
        zip_evaluations.append(
            {
                "zip": zip_code,
                "training_history": training_history,
                "holdout_actuals": holdout_actuals,
                "training_metrics": training_metrics,
                "training_scales": training_scales,
            }
        )
        for model_name in FORECAST_MODELS:
            if model_name not in training_scales:
                continue
            for horizon, actual in enumerate(holdout_actuals, start=1):
                prediction = _predict_with_model(training_history, model_name, horizon=horizon)
                if pd.isna(prediction) or pd.isna(actual):
                    continue
                error = float(actual - prediction)
                _update_holdout("candidate_model", model_name, zip_code, error, float(actual))
                for interval_level in (80, 95):
                    lower_bound, upper_bound = _forecast_interval_bounds(
                        prediction=float(prediction),
                        interval_level=interval_level,
                        error_scale=training_scales[model_name],
                        history=training_history,
                        horizon=horizon,
                    )
                    _update_calibration(
                        "candidate_model",
                        model_name,
                        zip_code,
                        interval_level,
                        prediction=float(prediction),
                        actual=float(actual),
                        lower_bound=lower_bound,
                        upper_bound=upper_bound,
                    )

    mape_threshold = 20.0
    def _holdout_rows_for_scope(
        *,
        scope: str,
        selected_family: str | None = None,
    ) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for (stats_scope, model_name), stats in holdout_stats.items():
            if stats_scope != scope:
                continue
            abs_errors = np.asarray(stats["abs_errors"], dtype=float)
            errors = np.asarray(stats["errors"], dtype=float)
            apes = np.asarray(stats["apes"], dtype=float)
            mean_mape = float(np.mean(apes)) if apes.size else np.nan
            rows.append(
                {
                    "evaluation_scope": stats_scope,
                    "model_name": model_name,
                    "selected_family": (
                        model_name if stats_scope == "candidate_model" else selected_family
                    ),
                    "eligible_zip_count": int(len(eligible)),
                    "zip_count": int(len(stats["zips"])),
                    "holdout_quarters": TEMPORAL_HOLDOUT_QUARTERS,
                    "evaluation_points": int(len(abs_errors)),
                    "mae": float(np.mean(abs_errors)),
                    "rmse": float(np.sqrt(np.mean(np.square(errors)))),
                    "mape": mean_mape,
                    "mape_threshold": mape_threshold,
                    "mape_pass": int(pd.notna(mean_mape) and mean_mape <= mape_threshold),
                }
            )
        return rows

    candidate_holdout_rows = _holdout_rows_for_scope(scope="candidate_model")
    candidate_holdout_df = pd.DataFrame.from_records(candidate_holdout_rows, columns=holdout_columns)
    allowed_models = tuple(
        model_name
        for model_name in FORECAST_MODELS
        if model_name
        in set(
            candidate_holdout_df.loc[
                pd.to_numeric(candidate_holdout_df["mape_pass"], errors="coerce").fillna(0).astype(int)
                == 1,
                "model_name",
            ]
            .astype(str)
            .tolist()
        )
    )
    selected_family = (
        "holdout_pass_screened_zip_selection" if allowed_models else "zip_level_holdout_selected"
    )
    if allowed_models:
        notes.append(
            "Selected ZIP holdout and forward forecasts are restricted to candidate families that "
            f"pass aggregate temporal holdout ({', '.join(allowed_models)})."
        )
    else:
        notes.append(
            "No candidate family cleared aggregate temporal holdout, so ZIP-level selection falls back "
            "to the full model family set."
        )

    for evaluation in zip_evaluations:
        zip_code = str(evaluation["zip"])
        training_history = np.asarray(evaluation["training_history"], dtype=float)
        holdout_actuals = np.asarray(evaluation["holdout_actuals"], dtype=float)
        training_metrics = pd.DataFrame(evaluation["training_metrics"]).copy()
        training_scales = dict(evaluation["training_scales"])
        selected_model = _select_forecast_model(
            training_metrics,
            history=training_history,
            allowed_models=allowed_models or None,
        )
        if selected_model is None:
            continue
        for horizon, actual in enumerate(holdout_actuals, start=1):
            prediction = _predict_with_model(training_history, selected_model, horizon=horizon)
            if pd.isna(prediction) or pd.isna(actual):
                continue
            error = float(actual - prediction)
            _update_holdout(
                "selected_zip_model",
                "selected_zip_model",
                zip_code,
                error,
                float(actual),
            )
            for interval_level in (80, 95):
                lower_bound, upper_bound = _forecast_interval_bounds(
                    prediction=float(prediction),
                    interval_level=interval_level,
                    error_scale=training_scales[selected_model],
                    history=training_history,
                    horizon=horizon,
                )
                _update_calibration(
                    "selected_zip_model",
                    "selected_zip_model",
                    zip_code,
                    interval_level,
                    prediction=float(prediction),
                    actual=float(actual),
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                )

    holdout_rows = candidate_holdout_rows
    holdout_rows.extend(
        _holdout_rows_for_scope(
            scope="selected_zip_model",
            selected_family=selected_family,
        )
    )

    calibration_rows: list[dict[str, object]] = []
    for (scope, model_name, interval_level), stats in calibration_stats.items():
        target_coverage = interval_level / 100.0
        empirical_coverage = _safe_ratio(stats["hits"], stats["count"])
        coverage_gap = (
            abs(empirical_coverage - target_coverage) if pd.notna(empirical_coverage) else np.nan
        )
        equal_share = _safe_ratio(stats["equal_bound_count"], stats["count"])
        extreme_share = _safe_ratio(stats["extreme_upper_ratio_count"], stats["count"])
        coverage_pass = pd.notna(coverage_gap) and coverage_gap <= 0.10
        shape_pass = (
            stats["equal_bound_count"] == 0 and stats["extreme_upper_ratio_count"] == 0
        )
        calibration_rows.append(
            {
                "evaluation_scope": scope,
                "model_name": model_name,
                "selected_family": (
                    model_name if scope == "candidate_model" else selected_family
                ),
                "interval_level": interval_level,
                "eligible_zip_count": int(len(eligible)),
                "zip_count": int(len(stats["zips"])),
                "holdout_quarters": TEMPORAL_HOLDOUT_QUARTERS,
                "evaluation_points": int(stats["count"]),
                "empirical_coverage": empirical_coverage,
                "target_coverage": target_coverage,
                "coverage_gap": coverage_gap,
                "mean_interval_width": float(np.mean(stats["widths"])) if stats["widths"] else np.nan,
                "median_interval_width": float(np.median(stats["widths"])) if stats["widths"] else np.nan,
                "equal_bound_count": int(stats["equal_bound_count"]),
                "equal_bound_share": equal_share,
                "extreme_upper_ratio_count": int(stats["extreme_upper_ratio_count"]),
                "extreme_upper_ratio_share": extreme_share,
                "max_upper_to_forecast_ratio": stats["max_upper_to_forecast_ratio"],
                "coverage_pass": int(coverage_pass),
                "shape_pass": int(shape_pass),
                "calibration_pass": int(coverage_pass and shape_pass),
            }
        )

    if not holdout_rows and not notes:
        notes.append("Temporal holdout evaluation did not produce any eligible ZIP-model comparisons.")
    else:
        notes.append(
            f"Temporal holdout evaluated the last {TEMPORAL_HOLDOUT_QUARTERS} quarters for "
            f"{len(eligible)} modeled ZIPs that passed the forecast history gate."
        )

    holdout_df = pd.DataFrame.from_records(holdout_rows, columns=holdout_columns)
    if not holdout_df.empty:
        holdout_df = holdout_df.sort_values(
            ["evaluation_scope", "model_name"],
            kind="mergesort",
            ignore_index=True,
        )
    calibration_df = pd.DataFrame.from_records(calibration_rows, columns=calibration_columns)
    if not calibration_df.empty:
        calibration_df = calibration_df.sort_values(
            ["evaluation_scope", "model_name", "interval_level"],
            kind="mergesort",
            ignore_index=True,
        )
    return holdout_df, calibration_df, notes


def _build_interval_shape_artifacts(forecast_intervals: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "evaluation_scope",
        "model_name",
        "selected_family",
        "interval_level",
        "eligible_zip_count",
        "zip_count",
        "holdout_quarters",
        "evaluation_points",
        "empirical_coverage",
        "target_coverage",
        "coverage_gap",
        "mean_interval_width",
        "median_interval_width",
        "equal_bound_count",
        "equal_bound_share",
        "extreme_upper_ratio_count",
        "extreme_upper_ratio_share",
        "max_upper_to_forecast_ratio",
        "coverage_pass",
        "shape_pass",
        "calibration_pass",
    ]
    if forecast_intervals.empty:
        return pd.DataFrame(columns=columns)

    frame = forecast_intervals.copy()
    frame["interval_level"] = pd.to_numeric(frame["interval_level"], errors="coerce")
    frame["forecast_value"] = pd.to_numeric(frame["forecast_value"], errors="coerce")
    frame["lower_bound"] = pd.to_numeric(frame["lower_bound"], errors="coerce")
    frame["upper_bound"] = pd.to_numeric(frame["upper_bound"], errors="coerce")
    grouped_rows: list[dict[str, object]] = []
    for interval_level, level_frame in frame.groupby("interval_level", dropna=True):
        selected_families = sorted(level_frame["selected_model"].dropna().astype(str).unique().tolist())
        ratio = np.where(
            pd.to_numeric(level_frame["forecast_value"], errors="coerce") > 0,
            pd.to_numeric(level_frame["upper_bound"], errors="coerce")
            / pd.to_numeric(level_frame["forecast_value"], errors="coerce"),
            np.nan,
        )
        equal_count = int(
            np.isclose(
                pd.to_numeric(level_frame["lower_bound"], errors="coerce"),
                pd.to_numeric(level_frame["upper_bound"], errors="coerce"),
            ).sum()
        )
        extreme_count = int(np.sum(pd.Series(ratio).fillna(-np.inf) > 100.0))
        shape_pass = equal_count == 0 and extreme_count == 0
        grouped_rows.append(
            {
                "evaluation_scope": "forecast_output_shape",
                "model_name": "selected_zip_model",
                "selected_family": (
                    selected_families[0]
                    if len(selected_families) == 1
                    else "mixed_selected_models"
                ),
                "interval_level": int(interval_level),
                "eligible_zip_count": int(level_frame["zip"].astype(str).nunique()),
                "zip_count": int(level_frame["zip"].astype(str).nunique()),
                "holdout_quarters": np.nan,
                "evaluation_points": int(len(level_frame)),
                "empirical_coverage": np.nan,
                "target_coverage": np.nan,
                "coverage_gap": np.nan,
                "mean_interval_width": float(
                    (
                        pd.to_numeric(level_frame["upper_bound"], errors="coerce")
                        - pd.to_numeric(level_frame["lower_bound"], errors="coerce")
                    ).mean()
                ),
                "median_interval_width": float(
                    np.median(
                        pd.to_numeric(level_frame["upper_bound"], errors="coerce")
                        - pd.to_numeric(level_frame["lower_bound"], errors="coerce")
                    )
                ),
                "equal_bound_count": equal_count,
                "equal_bound_share": _safe_ratio(equal_count, len(level_frame)),
                "extreme_upper_ratio_count": extreme_count,
                "extreme_upper_ratio_share": _safe_ratio(extreme_count, len(level_frame)),
                "max_upper_to_forecast_ratio": float(pd.Series(ratio).max())
                if pd.Series(ratio).notna().any()
                else np.nan,
                "coverage_pass": 0,
                "shape_pass": int(shape_pass),
                "calibration_pass": int(shape_pass),
            }
        )

    diagnostics = pd.DataFrame.from_records(grouped_rows, columns=columns)
    if not diagnostics.empty:
        diagnostics = diagnostics.sort_values(
            ["evaluation_scope", "interval_level"],
            kind="mergesort",
            ignore_index=True,
        )
    return diagnostics


def _build_scenario_artifacts(crime_forecasts: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "zip",
        "forecast_period_start",
        "horizon_quarters",
        "selected_model",
        "scenario_name",
        "scenario_multiplier",
        "baseline_total_rate_per_1000",
        "scenario_total_rate_per_1000",
        "scenario_delta_rate_per_1000",
        "evidence_posture",
        "causal_interpretation_allowed",
    ]
    if crime_forecasts.empty:
        return pd.DataFrame(columns=columns)

    forecasts = crime_forecasts.copy()
    if "forecast_tier" in forecasts.columns:
        forecasts = forecasts.loc[
            forecasts["forecast_tier"].astype(str) == "high_confidence"
        ].copy()
    if forecasts.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    for row in forecasts.itertuples(index=False):
        forecast_period = pd.Timestamp(row.forecast_period_start)
        for scenario_name, base_multiplier in SCENARIO_MULTIPLIERS.items():
            multiplier = base_multiplier
            if scenario_name == "seasonal_peak" and forecast_period.quarter in (2, 3):
                multiplier = 1.08
            scenario_value = float(row.predicted_total_rate_per_1000) * multiplier
            rows.append(
                {
                    "zip": row.zip,
                    "forecast_period_start": forecast_period,
                    "horizon_quarters": int(row.horizon_quarters),
                    "selected_model": row.selected_model,
                    "scenario_name": scenario_name,
                    "scenario_multiplier": float(multiplier),
                    "baseline_total_rate_per_1000": float(row.predicted_total_rate_per_1000),
                    "scenario_total_rate_per_1000": float(scenario_value),
                    "scenario_delta_rate_per_1000": float(
                        scenario_value - float(row.predicted_total_rate_per_1000)
                    ),
                    "evidence_posture": "exploratory_non_causal",
                    "causal_interpretation_allowed": 0,
                }
            )

    return pd.DataFrame.from_records(rows, columns=columns).sort_values(
        ["zip", "scenario_name", "forecast_period_start"],
        kind="mergesort",
        ignore_index=True,
    )


def _build_benchmark_artifacts(
    model_df: pd.DataFrame,
    cluster_assignments: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "zip",
        "crime_cluster",
        "socioeconomic_cluster",
        "market_cluster",
        "home_value",
        "total_rate_per_1000",
        "violent_rate_per_1000",
        "property_rate_per_1000",
        "median_household_income",
        "poverty_rate",
        "home_value_percentile",
        "total_rate_per_1000_percentile",
        "median_household_income_percentile",
        "home_value_zscore",
        "total_rate_per_1000_zscore",
        "median_household_income_zscore",
        "total_rate_vs_metro_pct",
        "home_value_vs_metro_pct",
        "violent_rate_vs_cluster_pct",
        "property_rate_vs_cluster_pct",
        "is_top_quartile_home_value",
        "is_top_quartile_crime_rate",
        "is_bottom_quartile_income",
    ]
    if model_df.empty:
        return pd.DataFrame(columns=columns)

    frame = model_df.copy()
    numeric_columns = [
        "home_value",
        "total_rate_per_1000",
        "violent_rate_per_1000",
        "property_rate_per_1000",
        "median_household_income",
        "poverty_rate",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if not cluster_assignments.empty and "zip" in cluster_assignments.columns:
        frame = frame.merge(cluster_assignments, on="zip", how="left")
    else:
        for column in ("crime_cluster", "socioeconomic_cluster", "market_cluster"):
            frame[column] = pd.NA

    metro_means = frame[numeric_columns].mean()
    for column in ("home_value", "total_rate_per_1000", "median_household_income"):
        frame[f"{column}_percentile"] = frame[column].rank(method="average", pct=True)
        column_std = frame[column].std(ddof=0)
        if pd.notna(column_std) and column_std > 0:
            frame[f"{column}_zscore"] = (frame[column] - frame[column].mean()) / column_std
        else:
            frame[f"{column}_zscore"] = np.nan

    frame["total_rate_vs_metro_pct"] = np.where(
        metro_means["total_rate_per_1000"] > 0,
        ((frame["total_rate_per_1000"] / metro_means["total_rate_per_1000"]) - 1.0) * 100.0,
        np.nan,
    )
    frame["home_value_vs_metro_pct"] = np.where(
        metro_means["home_value"] > 0,
        ((frame["home_value"] / metro_means["home_value"]) - 1.0) * 100.0,
        np.nan,
    )

    if "crime_cluster" in frame.columns:
        cluster_means = frame.groupby("crime_cluster", dropna=False)[
            ["violent_rate_per_1000", "property_rate_per_1000"]
        ].mean()
        frame["violent_rate_vs_cluster_pct"] = frame.apply(
            lambda row: (
                ((row["violent_rate_per_1000"] / cluster_means.loc[row["crime_cluster"], "violent_rate_per_1000"]) - 1.0) * 100.0
                if pd.notna(row["crime_cluster"])
                and row["crime_cluster"] in cluster_means.index
                and pd.notna(cluster_means.loc[row["crime_cluster"], "violent_rate_per_1000"])
                and cluster_means.loc[row["crime_cluster"], "violent_rate_per_1000"] != 0
                else np.nan
            ),
            axis=1,
        )
        frame["property_rate_vs_cluster_pct"] = frame.apply(
            lambda row: (
                ((row["property_rate_per_1000"] / cluster_means.loc[row["crime_cluster"], "property_rate_per_1000"]) - 1.0) * 100.0
                if pd.notna(row["crime_cluster"])
                and row["crime_cluster"] in cluster_means.index
                and pd.notna(cluster_means.loc[row["crime_cluster"], "property_rate_per_1000"])
                and cluster_means.loc[row["crime_cluster"], "property_rate_per_1000"] != 0
                else np.nan
            ),
            axis=1,
        )
    else:
        frame["violent_rate_vs_cluster_pct"] = np.nan
        frame["property_rate_vs_cluster_pct"] = np.nan

    frame["is_top_quartile_home_value"] = (
        frame["home_value_percentile"] >= 0.75
    ).astype(int)
    frame["is_top_quartile_crime_rate"] = (
        frame["total_rate_per_1000_percentile"] >= 0.75
    ).astype(int)
    frame["is_bottom_quartile_income"] = (
        frame["median_household_income_percentile"] <= 0.25
    ).astype(int)

    return frame[columns].sort_values("zip", kind="mergesort", ignore_index=True)


def _build_drift_artifacts(
    temporal_summary: pd.DataFrame,
    temporal_series: dict[str, pd.DataFrame],
    *,
    data_cutoff: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    columns = [
        "domain",
        "entity_type",
        "entity_id",
        "metric",
        "latest_period_start",
        "baseline_periods",
        "latest_value",
        "baseline_mean",
        "baseline_std",
        "z_score",
        "relative_change_pct",
        "drift_flag",
        "trailing_contiguous_quarters",
        "quarter_completeness_ratio",
    ]
    notes: list[str] = []
    if temporal_summary.empty or not temporal_series:
        notes.append("Temporal summary inputs were unavailable, so drift diagnostics are empty.")
        return pd.DataFrame(columns=columns), notes

    eligible = temporal_summary.loc[
        pd.to_numeric(temporal_summary["drift_gate_pass"], errors="coerce").fillna(0).astype(int)
        == 1
    ].copy()
    if eligible.empty:
        notes.append(
            "No modeled ZIP met the trailing-contiguous quarterly history gate required for drift checks."
        )
        return pd.DataFrame(columns=columns), notes

    notes.append(
        f"Drift checks use only modeled ZIPs with >= {DRIFT_HISTORY_MIN_QUARTERS} trailing contiguous quarters "
        f"({len(eligible)} of {len(temporal_summary)} modeled ZIPs passed)."
    )
    rows: list[dict[str, object]] = []
    entities: list[tuple[str, str, pd.Series]] = []
    metro_panel = []
    completeness_lookup = eligible.set_index("zip")
    for zip_code in eligible["zip"].astype(str):
        ordered = temporal_series.get(zip_code)
        if ordered is None or ordered.empty:
            continue
        metro_panel.append(
            ordered[["period_start", "total_rate_per_1000"]].assign(zip=zip_code).copy()
        )
        zip_series = (
            ordered.sort_values("period_start", kind="mergesort")
            .set_index("period_start")["total_rate_per_1000"]
        )
        entities.append(("zip", zip_code, zip_series))
    if not metro_panel:
        notes.append("No eligible ZIP panels were available after drift gating.")
        return pd.DataFrame(columns=columns), notes

    metro_series = (
        pd.concat(metro_panel, ignore_index=True)
        .groupby("period_start")["total_rate_per_1000"]
        .mean()
        .sort_index(kind="mergesort")
    )
    entities.append(("portfolio", "metro", metro_series))

    _cutoff = (data_cutoff if data_cutoff is not None else pd.Timestamp.today()).normalize()
    for entity_type, entity_id, series in entities:
        if len(series) < DRIFT_HISTORY_MIN_QUARTERS:
            notes.append(
                f"{entity_type} {entity_id}: fewer than {DRIFT_HISTORY_MIN_QUARTERS} periods were available for drift checks."
            )
            continue
        baseline = series.iloc[-DRIFT_HISTORY_MIN_QUARTERS:-1]
        latest_value = float(series.iloc[-1])
        # DQA-D2: pro-rate latest value if the latest period is a partial quarter
        latest_period_ts = pd.Timestamp(series.index[-1])
        try:
            quarter_period = pd.Period(latest_period_ts, freq="Q")
            quarter_end = quarter_period.end_time.normalize()
            quarter_length = max(1, (quarter_end - latest_period_ts).days)
            days_elapsed = max(1, min((_cutoff - latest_period_ts).days + 1, quarter_length))
            quarter_completeness = days_elapsed / quarter_length
        except Exception:
            quarter_completeness = 1.0
        if 0.0 < quarter_completeness < 1.0:
            adjusted_latest = latest_value / quarter_completeness
        else:
            adjusted_latest = latest_value
        baseline_mean = float(baseline.mean())
        baseline_std = float(baseline.std(ddof=0))
        z_score = (
            (adjusted_latest - baseline_mean) / baseline_std
            if baseline_std > 0
            else np.nan
        )
        relative_change = (
            ((adjusted_latest / baseline_mean) - 1.0) * 100.0
            if baseline_mean > 0
            else np.nan
        )
        rows.append(
            {
                "domain": "crime",
                "entity_type": entity_type,
                "entity_id": entity_id,
                "metric": "total_rate_per_1000",
                "latest_period_start": series.index[-1],
                "baseline_periods": int(len(baseline)),
                "latest_value": latest_value,
                "baseline_mean": baseline_mean,
                "baseline_std": baseline_std,
                "z_score": z_score,
                "relative_change_pct": relative_change,
                "drift_flag": int(
                    (pd.notna(z_score) and abs(z_score) >= 1.5)
                    or (pd.notna(relative_change) and abs(relative_change) >= 25.0)
                ),
                "trailing_contiguous_quarters": (
                    int(completeness_lookup.loc[entity_id, "trailing_contiguous_quarters"])
                    if entity_type == "zip" and entity_id in completeness_lookup.index
                    else int(len(series))
                ),
                "quarter_completeness_ratio": round(quarter_completeness, 4),
            }
        )

    if not rows and not notes:
        notes.append("No series met the minimum history requirement for drift diagnostics.")
    notes.append("Housing drift is not yet modeled in this offline artifact set.")
    return pd.DataFrame.from_records(rows, columns=columns), notes


def _build_feature_selection_artifacts(
    model_df: pd.DataFrame,
    *,
    expanded_controls: tuple[str, ...],
) -> tuple[pd.DataFrame, list[str]]:
    columns = [
        "feature_name",
        "feature_group",
        "available_rows",
        "availability_ratio",
        "missing_ratio",
        "correlation_with_log_home_value",
        "univariate_r_squared",
        "univariate_p_value",
        "selected_in_current_models",
        "selection_score",
        "selection_rank",
        "recommended_for_future_models",
        "fdr_q_value",
        "passes_fdr_10",
        "practical_effect_value",
        "practical_effect_threshold",
        "passes_practical_effect",
        "interpretation_allowed",
    ]
    notes: list[str] = []

    frame = _ensure_dependent_column(model_df, "log_home_value")
    frame["log_home_value"] = pd.to_numeric(frame["log_home_value"], errors="coerce")
    total_rows = len(frame)
    if total_rows == 0:
        notes.append("Model dataset is empty; feature-selection metrics are unavailable.")
        return pd.DataFrame(columns=columns), notes

    existing_candidates = [
        feature
        for feature in dict.fromkeys(FEATURE_SELECTION_CANDIDATES)
        if feature in frame.columns
    ]
    if not existing_candidates:
        notes.append("No configured feature-selection candidates were present in model data.")
        return pd.DataFrame(columns=columns), notes

    selected_now = set(DEFAULT_PREDICTORS) | set(DEFAULT_CONTROLS) | set(expanded_controls)
    rows: list[dict[str, object]] = []
    for feature in existing_candidates:
        values = pd.to_numeric(frame[feature], errors="coerce")
        mask = values.notna() & frame["log_home_value"].notna()
        available_rows = int(mask.sum())
        availability_ratio = float(available_rows / total_rows) if total_rows else np.nan
        missing_ratio = float(1.0 - availability_ratio) if pd.notna(availability_ratio) else np.nan
        correlation = np.nan
        univariate_r_squared = np.nan
        univariate_p_value = np.nan
        if available_rows >= 3 and values.loc[mask].nunique(dropna=True) > 1:
            correlation = float(values.loc[mask].corr(frame.loc[mask, "log_home_value"]))
            try:
                fitted = smf.ols(
                    formula=f"log_home_value ~ {feature}",
                    data=frame.loc[mask, ["log_home_value", feature]],
                ).fit(cov_type="HC3")
                univariate_r_squared = float(fitted.rsquared)
                univariate_p_value = float(fitted.pvalues.get(feature, np.nan))
            except (TypeError, ValueError, np.linalg.LinAlgError):
                notes.append(
                    f"- `{feature}`: skipped univariate fit due numerical instability."
                )
        score = (
            abs(correlation) * availability_ratio
            if pd.notna(correlation) and pd.notna(availability_ratio)
            else 0.0
        )
        feature_group = "crime" if "rate" in feature or "crime" in feature else "controls"
        rows.append(
            {
                "feature_name": feature,
                "feature_group": feature_group,
                "available_rows": available_rows,
                "availability_ratio": availability_ratio,
                "missing_ratio": missing_ratio,
                "correlation_with_log_home_value": correlation,
                "univariate_r_squared": univariate_r_squared,
                "univariate_p_value": univariate_p_value,
                "selected_in_current_models": int(feature in selected_now),
                "selection_score": float(score),
                "selection_rank": np.nan,
                "recommended_for_future_models": 0,
                "fdr_q_value": np.nan,
                "passes_fdr_10": 0,
                "practical_effect_value": abs(float(correlation)) if pd.notna(correlation) else np.nan,
                "practical_effect_threshold": FEATURE_PRACTICAL_CORRELATION_THRESHOLD,
                "passes_practical_effect": 0,
                "interpretation_allowed": 0,
            }
        )

    selection_df = pd.DataFrame.from_records(rows, columns=columns).sort_values(
        ["selection_score", "availability_ratio", "feature_name"],
        ascending=[False, False, True],
        kind="mergesort",
        ignore_index=True,
    )
    selection_df["selection_rank"] = np.arange(1, len(selection_df) + 1)
    top_cut = min(10, len(selection_df))
    recommended_mask = (
        (selection_df["selection_rank"] <= top_cut)
        & (selection_df["availability_ratio"] >= 0.7)
    ) | (selection_df["selected_in_current_models"] == 1)
    selection_df["recommended_for_future_models"] = recommended_mask.astype(int)
    selection_df["fdr_q_value"] = _bh_adjust_series(selection_df["univariate_p_value"])
    selection_df["passes_fdr_10"] = (
        selection_df["fdr_q_value"].notna() & (selection_df["fdr_q_value"] <= FDR_ALPHA)
    ).astype(int)
    selection_df["passes_practical_effect"] = (
        pd.to_numeric(selection_df["practical_effect_value"], errors="coerce").fillna(0.0)
        >= FEATURE_PRACTICAL_CORRELATION_THRESHOLD
    ).astype(int)
    selection_df["interpretation_allowed"] = (
        (selection_df["passes_fdr_10"] == 1)
        & (selection_df["passes_practical_effect"] == 1)
        & (selection_df["availability_ratio"] >= 0.7)
    ).astype(int)
    if int(selection_df["interpretation_allowed"].sum()) == 0:
        notes.append(
            "No feature candidate cleared both the FDR-adjusted significance screen and the "
            "practical-correlation threshold."
        )
    return selection_df, notes


def _build_feature_power_retention_artifacts(
    feature_selection_metrics: pd.DataFrame,
    predictive_model_metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    columns = [
        "metric",
        "value",
        "threshold",
        "meets_threshold",
        "definition",
    ]
    notes: list[str] = []

    selection_frame = feature_selection_metrics.copy()
    predictive_frame = predictive_model_metrics.copy()

    candidate_count = int(len(selection_frame)) if not selection_frame.empty else 0
    if not selection_frame.empty:
        recommended_mask = (
            pd.to_numeric(selection_frame["recommended_for_future_models"], errors="coerce")
            .fillna(0)
            .astype(int)
            == 1
        )
    else:
        recommended_mask = pd.Series(dtype=bool)
    recommended_count = int(recommended_mask.sum()) if not selection_frame.empty else 0
    recommended_feature_ratio = (
        float(recommended_count / candidate_count) if candidate_count > 0 else np.nan
    )

    selection_scores = pd.to_numeric(selection_frame.get("selection_score"), errors="coerce")
    selection_score_total = float(selection_scores.sum()) if not selection_frame.empty else np.nan
    selection_score_recommended = (
        float(selection_scores.loc[recommended_mask].sum())
        if not selection_frame.empty
        else np.nan
    )
    selection_score_retention_ratio = (
        float(selection_score_recommended / selection_score_total)
        if pd.notna(selection_score_total) and selection_score_total > 0
        else np.nan
    )

    predictive_frame["r_squared"] = pd.to_numeric(
        predictive_frame.get("r_squared"),
        errors="coerce",
    )
    predictive_frame["selected_for_ensemble"] = (
        pd.to_numeric(predictive_frame.get("selected_for_ensemble"), errors="coerce")
        .fillna(0)
        .astype(int)
    )
    model_tier = pd.Series(predictive_frame.get("model_tier"), dtype="string").fillna("unknown")
    model_family_frame = predictive_frame.loc[model_tier != "ensemble"].copy()
    best_family_r_squared = (
        float(model_family_frame["r_squared"].max())
        if not model_family_frame.empty and model_family_frame["r_squared"].notna().any()
        else np.nan
    )
    selected_frame = model_family_frame.loc[model_family_frame["selected_for_ensemble"] == 1].copy()
    best_selected_r_squared = (
        float(selected_frame["r_squared"].max())
        if not selected_frame.empty and selected_frame["r_squared"].notna().any()
        else np.nan
    )
    predictive_r_squared_retention_ratio = (
        float(best_selected_r_squared / best_family_r_squared)
        if pd.notna(best_selected_r_squared)
        and pd.notna(best_family_r_squared)
        and best_family_r_squared > 0
        else np.nan
    )

    score_threshold = 0.90
    r_squared_threshold = 0.90
    score_retention_pass = (
        pd.notna(selection_score_retention_ratio)
        and selection_score_retention_ratio >= score_threshold
    )
    r_squared_retention_pass = (
        pd.notna(predictive_r_squared_retention_ratio)
        and predictive_r_squared_retention_ratio >= r_squared_threshold
    )
    checkpoint_pass = bool(score_retention_pass and r_squared_retention_pass)

    rows = [
        {
            "metric": "feature_candidate_count",
            "value": float(candidate_count),
            "threshold": np.nan,
            "meets_threshold": np.nan,
            "definition": "Total feature-selection candidates present in model data.",
        },
        {
            "metric": "recommended_feature_count",
            "value": float(recommended_count),
            "threshold": np.nan,
            "meets_threshold": np.nan,
            "definition": "Candidates flagged as recommended_for_future_models.",
        },
        {
            "metric": "recommended_feature_ratio",
            "value": recommended_feature_ratio,
            "threshold": np.nan,
            "meets_threshold": np.nan,
            "definition": "recommended_feature_count / feature_candidate_count.",
        },
        {
            "metric": "feature_selection_score_retention_ratio",
            "value": selection_score_retention_ratio,
            "threshold": score_threshold,
            "meets_threshold": float(int(score_retention_pass)),
            "definition": (
                "sum(selection_score for recommended features) / "
                "sum(selection_score for all candidates)"
            ),
        },
        {
            "metric": "predictive_r_squared_retention_ratio",
            "value": predictive_r_squared_retention_ratio,
            "threshold": r_squared_threshold,
            "meets_threshold": float(int(r_squared_retention_pass)),
            "definition": (
                "best selected non-ensemble model-family R-squared / "
                "best available non-ensemble model-family R-squared"
            ),
        },
        {
            "metric": "feature_power_checkpoint_pass",
            "value": float(int(checkpoint_pass)),
            "threshold": 1.0,
            "meets_threshold": float(int(checkpoint_pass)),
            "definition": "Pass requires both retention ratios to meet >= 0.90.",
        },
    ]
    if not checkpoint_pass:
        notes.append(
            "Feature-power checkpoint did not pass all thresholds in this run; inspect retention ratios."
        )
    return pd.DataFrame.from_records(rows, columns=columns), notes


def _build_predictive_model_family_artifacts(
    model_df: pd.DataFrame,
    *,
    baseline_result: RegressionResult,
    expanded_result: RegressionResult,
    cluster_assignments: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    metric_columns = [
        "model_label",
        "model_tier",
        "nobs",
        "r_squared",
        "adjusted_r_squared",
        "mae",
        "rmse",
        "loocv_mae",
        "loocv_rmse",
        "predicted_r_squared",
        "selection_rank",
        "selected_for_ensemble",
        "ensemble_weight",
    ]
    prediction_columns = [
        "zip",
        "model_label",
        "model_tier",
        "observed",
        "predicted",
        "residual",
        "absolute_residual",
    ]
    notes: list[str] = []

    family_results: list[RegressionResult] = [baseline_result, expanded_result]
    tier_map: dict[str, str] = {
        baseline_result.model_label: "baseline",
        expanded_result.model_label: "intermediate",
    }

    def _try_add_model(
        *,
        frame: pd.DataFrame,
        predictors: tuple[str, ...],
        controls: tuple[str, ...],
        model_label: str,
        model_tier: str,
    ) -> None:
        required = set(predictors) | set(controls)
        if not required <= set(frame.columns):
            missing = sorted(required - set(frame.columns))
            notes.append(f"- `{model_label}` skipped: missing columns {', '.join(missing)}.")
            return
        try:
            result = run_zip_regression(
                frame,
                predictors=predictors,
                controls=controls,
                model_label=model_label,
            )
        except (KeyError, ValueError) as exc:
            notes.append(f"- `{model_label}` skipped: {exc}")
            return
        family_results.append(result)
        tier_map[model_label] = model_tier

    intermediate_controls = tuple(
        dict.fromkeys(
            [
                *DEFAULT_CONTROLS,
                "median_rent",
                "annual_change_pct",
            ]
        )
    )
    _try_add_model(
        frame=model_df,
        predictors=DEFAULT_PREDICTORS,
        controls=tuple(control for control in intermediate_controls if control in model_df.columns),
        model_label="intermediate_market_augmented",
        model_tier="intermediate",
    )

    advanced_frame = model_df.copy()
    if {"violent_rate_per_1000", "property_rate_per_1000"} <= set(advanced_frame.columns):
        advanced_frame["violent_property_interaction"] = (
            pd.to_numeric(advanced_frame["violent_rate_per_1000"], errors="coerce")
            * pd.to_numeric(advanced_frame["property_rate_per_1000"], errors="coerce")
        )
    _try_add_model(
        frame=advanced_frame,
        predictors=tuple(
            predictor
            for predictor in (
                *DEFAULT_PREDICTORS,
                "violent_property_interaction",
            )
            if predictor in advanced_frame.columns
        ),
        controls=DEFAULT_CONTROLS,
        model_label="advanced_interaction_model",
        model_tier="advanced",
    )

    specialized_frame = model_df.copy()
    if not cluster_assignments.empty and {"zip", "crime_cluster"} <= set(cluster_assignments.columns):
        specialized_frame = specialized_frame.merge(
            cluster_assignments[["zip", "crime_cluster"]],
            on="zip",
            how="left",
        )
        specialized_frame["crime_cluster_code"] = (
            pd.Series(specialized_frame["crime_cluster"], dtype="string")
            .fillna("unknown")
            .factorize()[0]
            .astype(float)
        )
        specialized_predictor = ("crime_cluster_code",)
    else:
        specialized_frame["high_crime_flag"] = (
            pd.to_numeric(specialized_frame["total_rate_per_1000"], errors="coerce")
            >= pd.to_numeric(specialized_frame["total_rate_per_1000"], errors="coerce").median()
        ).astype(float)
        specialized_predictor = ("high_crime_flag",)
    _try_add_model(
        frame=specialized_frame,
        predictors=tuple([*DEFAULT_PREDICTORS, *specialized_predictor]),
        controls=DEFAULT_CONTROLS,
        model_label="specialized_segment_model",
        model_tier="specialized",
    )

    validation_df, validation_notes = _build_validation_artifacts(family_results)
    notes.extend(validation_notes)
    base_metrics_df = pd.concat(
        [result.metrics_table() for result in family_results],
        ignore_index=True,
    )
    family_metrics = base_metrics_df.merge(
        validation_df,
        on="model_label",
        how="left",
    )
    family_metrics["model_tier"] = family_metrics["model_label"].map(tier_map).fillna("custom")
    family_metrics["selection_rank"] = np.nan
    family_metrics["selected_for_ensemble"] = 0
    family_metrics["ensemble_weight"] = 0.0
    if family_metrics["loocv_rmse"].notna().any():
        family_metrics["selection_rank"] = family_metrics["loocv_rmse"].rank(
            method="dense",
            na_option="bottom",
        )

    selected_models = (
        family_metrics.loc[family_metrics["loocv_rmse"].notna()]
        .sort_values(["loocv_rmse", "mae", "model_label"], kind="mergesort")
        .head(2)["model_label"]
        .tolist()
    )
    selected_weights: dict[str, float] = {}
    if selected_models:
        inverse_errors = []
        for model_label in selected_models:
            loocv_rmse = float(
                family_metrics.loc[family_metrics["model_label"] == model_label, "loocv_rmse"].iloc[0]
            )
            inverse_errors.append(1.0 / max(loocv_rmse, 1e-6))
        total_inverse = float(np.sum(inverse_errors))
        if total_inverse > 0:
            selected_weights = {
                model_label: float(weight / total_inverse)
                for model_label, weight in zip(selected_models, inverse_errors, strict=True)
            }
            family_metrics.loc[
                family_metrics["model_label"].isin(selected_models),
                "selected_for_ensemble",
            ] = 1
            family_metrics["ensemble_weight"] = family_metrics["model_label"].map(
                selected_weights
            ).fillna(0.0)

    prediction_rows: list[dict[str, object]] = []
    model_prediction_map: dict[str, pd.DataFrame] = {}
    for result in family_results:
        tier = tier_map.get(result.model_label, "custom")
        residual_frame = result.residuals.copy()
        if "zip" not in residual_frame.columns:
            continue
        residual_frame["zip"] = residual_frame["zip"].astype("string")
        model_prediction_map[result.model_label] = residual_frame[
            ["zip", "observed", "fitted_value", "residual", "absolute_residual"]
        ].copy()
        for row in residual_frame.itertuples(index=False):
            prediction_rows.append(
                {
                    "zip": str(row.zip),
                    "model_label": result.model_label,
                    "model_tier": tier,
                    "observed": float(row.observed),
                    "predicted": float(row.fitted_value),
                    "residual": float(row.residual),
                    "absolute_residual": float(row.absolute_residual),
                }
            )

    if len(selected_weights) >= 2 and all(model in model_prediction_map for model in selected_weights):
        observed_frame = _ensure_dependent_column(model_df, "log_home_value")[["zip", "log_home_value"]].copy()
        observed_frame["zip"] = observed_frame["zip"].astype("string")
        observed_frame["log_home_value"] = pd.to_numeric(
            observed_frame["log_home_value"], errors="coerce"
        )
        observed_frame = observed_frame.dropna(subset=["zip", "log_home_value"]).drop_duplicates(
            subset=["zip"], keep="first"
        )
        combined = observed_frame.rename(columns={"log_home_value": "observed"}).copy()
        for model_label, weight in selected_weights.items():
            model_predictions = model_prediction_map[model_label][["zip", "fitted_value"]].rename(
                columns={"fitted_value": f"pred_{model_label}"}
            )
            combined = combined.merge(model_predictions, on="zip", how="inner")
            combined[f"weight_{model_label}"] = weight

        prediction_columns_for_models = [f"pred_{label}" for label in selected_weights]
        if not combined.empty and combined[prediction_columns_for_models].notna().all(axis=1).any():
            combined = combined.loc[
                combined[prediction_columns_for_models].notna().all(axis=1)
            ].copy()
            combined["predicted"] = 0.0
            for model_label, weight in selected_weights.items():
                combined["predicted"] += combined[f"pred_{model_label}"] * weight
            combined["residual"] = combined["observed"] - combined["predicted"]
            combined["absolute_residual"] = combined["residual"].abs()

            for row in combined.itertuples(index=False):
                prediction_rows.append(
                    {
                        "zip": str(row.zip),
                        "model_label": "ensemble_top2_inverse_rmse",
                        "model_tier": "ensemble",
                        "observed": float(row.observed),
                        "predicted": float(row.predicted),
                        "residual": float(row.residual),
                        "absolute_residual": float(row.absolute_residual),
                    }
                )

            rmse = float(np.sqrt(np.mean(np.square(combined["residual"].to_numpy(dtype=float)))))
            mae = float(np.mean(combined["absolute_residual"].to_numpy(dtype=float)))
            response = combined["observed"].to_numpy(dtype=float)
            sse = float(np.square(combined["residual"].to_numpy(dtype=float)).sum())
            sst = float(np.square(response - response.mean()).sum())
            r_squared = (1.0 - (sse / sst)) if sst > 0 else np.nan
            family_metrics = pd.concat(
                [
                    family_metrics,
                    pd.DataFrame(
                        [
                            {
                                "model_label": "ensemble_top2_inverse_rmse",
                                "dependent_variable": "log_home_value",
                                "formula": "weighted_average(top2 loocv_rmse models)",
                                "predictors": "; ".join(selected_weights.keys()),
                                "controls": "n/a",
                                "nobs": int(len(combined)),
                                "r_squared": r_squared,
                                "adjusted_r_squared": np.nan,
                                "mae": mae,
                                "rmse": rmse,
                                "loocv_mae": np.nan,
                                "loocv_rmse": np.nan,
                                "predicted_r_squared": np.nan,
                                "mean_leverage": np.nan,
                                "max_leverage": np.nan,
                                "max_cooks_distance": np.nan,
                                "model_tier": "ensemble",
                                "selection_rank": 1.0,
                                "selected_for_ensemble": 1,
                                "ensemble_weight": 1.0,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )

    family_metrics = family_metrics.sort_values(
        ["selection_rank", "model_label"],
        ascending=[True, True],
        na_position="last",
        kind="mergesort",
        ignore_index=True,
    )
    predictive_metrics = family_metrics[metric_columns].copy()
    predictions = pd.DataFrame.from_records(prediction_rows, columns=prediction_columns)
    return predictive_metrics, predictions, notes


def _build_comprehensive_validation_artifacts(
    *,
    regression_validation: pd.DataFrame,
    predictive_metrics: pd.DataFrame,
    forecast_metrics: pd.DataFrame,
    temporal_holdout: pd.DataFrame,
    interval_calibration: pd.DataFrame,
    drift_diagnostics: pd.DataFrame,
    influence_robustness: pd.DataFrame,
    cluster_stability: pd.DataFrame,
    statistical_guardrails: pd.DataFrame,
    scenario_impacts: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    columns = ["domain", "item", "metric", "value"]
    rows: list[dict[str, object]] = []
    notes: list[str] = []

    for row in regression_validation.itertuples(index=False):
        rows.append(
            {
                "domain": "regression",
                "item": row.model_label,
                "metric": "loocv_rmse",
                "value": float(row.loocv_rmse),
            }
        )
        rows.append(
            {
                "domain": "regression",
                "item": row.model_label,
                "metric": "predicted_r_squared",
                "value": float(row.predicted_r_squared),
            }
        )

    for row in predictive_metrics.itertuples(index=False):
        if pd.notna(row.rmse):
            rows.append(
                {
                    "domain": "predictive_family",
                    "item": row.model_label,
                    "metric": "rmse",
                    "value": float(row.rmse),
                }
            )
        if pd.notna(row.loocv_rmse):
            rows.append(
                {
                    "domain": "predictive_family",
                    "item": row.model_label,
                    "metric": "loocv_rmse",
                    "value": float(row.loocv_rmse),
                }
            )

    for row in forecast_metrics.itertuples(index=False):
        if pd.notna(row.rmse):
            rows.append(
                {
                    "domain": "forecast",
                    "item": row.model_name,
                    "metric": "rmse",
                    "value": float(row.rmse),
                }
            )
        if pd.notna(row.mae):
            rows.append(
                {
                    "domain": "forecast",
                    "item": row.model_name,
                    "metric": "mae",
                    "value": float(row.mae),
                }
            )

    for row in temporal_holdout.itertuples(index=False):
        if pd.notna(row.rmse):
            rows.append(
                {
                    "domain": "temporal_holdout",
                    "item": f"{row.evaluation_scope}:{row.model_name}",
                    "metric": "rmse",
                    "value": float(row.rmse),
                }
            )
        if pd.notna(row.mape):
            rows.append(
                {
                    "domain": "temporal_holdout",
                    "item": f"{row.evaluation_scope}:{row.model_name}",
                    "metric": "mape",
                    "value": float(row.mape),
                }
            )

    for row in interval_calibration.itertuples(index=False):
        if pd.notna(row.empirical_coverage):
            rows.append(
                {
                    "domain": "interval_calibration",
                    "item": f"{row.evaluation_scope}:{int(row.interval_level)}",
                    "metric": "empirical_coverage",
                    "value": float(row.empirical_coverage),
                }
            )
        rows.append(
            {
                "domain": "interval_calibration",
                "item": f"{row.evaluation_scope}:{int(row.interval_level)}",
                "metric": "calibration_pass",
                "value": float(row.calibration_pass),
            }
        )

    if not drift_diagnostics.empty:
        drift_flag_share = float(pd.to_numeric(drift_diagnostics["drift_flag"], errors="coerce").mean())
        rows.append(
            {
                "domain": "drift",
                "item": "all_entities",
                "metric": "drift_flag_share",
                "value": drift_flag_share,
            }
        )
    else:
        notes.append("Drift diagnostics were empty; drift coverage is partial for this run.")

    if not influence_robustness.empty:
        rows.append(
            {
                "domain": "influence_robustness",
                "item": "all_models",
                "metric": "robustness_pass_share",
                "value": float(
                    pd.to_numeric(influence_robustness["robustness_pass"], errors="coerce").mean()
                ),
            }
        )
    else:
        notes.append("Influence robustness diagnostics were empty for this run.")

    if not cluster_stability.empty:
        rows.append(
            {
                "domain": "cluster_stability",
                "item": "all_domains",
                "metric": "practical_utility_pass_share",
                "value": float(
                    pd.to_numeric(cluster_stability["practical_utility_pass"], errors="coerce").mean()
                ),
            }
        )
    else:
        notes.append("Cluster stability diagnostics were empty for this run.")

    if not statistical_guardrails.empty:
        rows.append(
            {
                "domain": "statistical_guardrails",
                "item": "all_tests",
                "metric": "interpretation_allowed_share",
                "value": float(
                    pd.to_numeric(statistical_guardrails["interpretation_allowed"], errors="coerce").mean()
                ),
            }
        )
    else:
        notes.append("Statistical guardrail coverage was empty for this run.")

    if not scenario_impacts.empty:
        worst_case = scenario_impacts.loc[
            scenario_impacts["scenario_name"].astype(str) == "systemic_shock",
            "scenario_delta_rate_per_1000",
        ]
        if not worst_case.empty:
            rows.append(
                {
                    "domain": "scenario",
                    "item": "systemic_shock",
                    "metric": "mean_delta_rate_per_1000",
                    "value": float(pd.to_numeric(worst_case, errors="coerce").mean()),
                }
            )
    else:
        notes.append("Scenario impacts were empty; stress-testing coverage is partial.")

    validation = pd.DataFrame.from_records(rows, columns=columns)
    if validation.empty and not notes:
        notes.append("No comprehensive validation rows were produced for this run.")
    if not validation.empty:
        validation = validation.sort_values(
            ["domain", "item", "metric"],
            kind="mergesort",
            ignore_index=True,
        )
    return validation, notes


def _build_policy_recommendations_artifacts(
    zip_benchmarks: pd.DataFrame,
    scenario_impacts: pd.DataFrame,
    *,
    cluster_stability: pd.DataFrame,
    influence_robustness: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    columns = [
        "segment_type",
        "segment_label",
        "zip_count",
        "avg_total_rate_per_1000",
        "avg_home_value",
        "avg_adverse_momentum_delta",
        "avg_systemic_shock_delta",
        "priority_tier",
        "recommended_actions",
        "scenario_eligible_zip_count",
        "scenario_coverage_share",
        "high_influence_zip_count",
        "high_influence_zip_share",
        "cluster_selected_k",
        "cluster_min_cluster_size",
        "cluster_min_cluster_size_threshold",
        "cluster_silhouette_score",
        "cluster_leave_one_feature_out_mean_ari",
        "cluster_practical_utility_pass",
        "segment_size_pass",
        "segment_guardrail_status",
        "segment_confidence_tier",
        "guardrail_flags",
        "evidence_posture",
        "causal_interpretation_allowed",
    ]
    notes: list[str] = []
    if zip_benchmarks.empty:
        notes.append("ZIP benchmark inputs were empty, so no policy recommendations were generated.")
        return pd.DataFrame(columns=columns), notes
    if scenario_impacts.empty:
        notes.append("Scenario impacts were empty, so policy recommendations are unavailable.")
        return pd.DataFrame(columns=columns), notes

    scenario_frame = scenario_impacts.copy()
    scenario_frame["zip"] = scenario_frame["zip"].astype("string")
    scenario_frame["scenario_name"] = scenario_frame["scenario_name"].astype("string")
    scenario_frame["scenario_delta_rate_per_1000"] = pd.to_numeric(
        scenario_frame["scenario_delta_rate_per_1000"], errors="coerce"
    )
    scenario_summary = (
        scenario_frame.pivot_table(
            index="zip",
            columns="scenario_name",
            values="scenario_delta_rate_per_1000",
            aggfunc="mean",
        )
        .rename_axis(None, axis=1)
        .reset_index()
    )
    benchmarks = zip_benchmarks.copy()
    benchmarks["zip"] = benchmarks["zip"].astype("string")
    benchmarks["total_rate_per_1000"] = pd.to_numeric(
        benchmarks["total_rate_per_1000"], errors="coerce"
    )
    benchmarks["home_value"] = pd.to_numeric(benchmarks["home_value"], errors="coerce")
    scenario_supported_zips = set(scenario_summary["zip"].astype("string").dropna().astype(str))
    flagged_influence_zips = set(
        pd.Series(influence_robustness.get("removed_zip"), dtype="string").dropna().astype(str)
    )
    cluster_lookup = (
        cluster_stability.copy().assign(domain=cluster_stability["domain"].astype("string"))
        if not cluster_stability.empty
        else pd.DataFrame(columns=["domain"])
    )
    merged = benchmarks.merge(scenario_summary, on="zip", how="left")
    if merged.empty:
        notes.append("No ZIP overlap was found between benchmarks and scenarios.")
        return pd.DataFrame(columns=columns), notes

    global_rate_median = float(merged["total_rate_per_1000"].median())
    global_shock_median = float(pd.to_numeric(merged.get("systemic_shock"), errors="coerce").median())
    rows: list[dict[str, object]] = []
    segment_to_domain = {
        "crime_cluster": "crime",
        "socioeconomic_cluster": "socioeconomic",
        "market_cluster": "market",
    }
    for segment_type, domain in segment_to_domain.items():
        if segment_type not in merged.columns:
            continue
        domain_cluster = (
            cluster_lookup.loc[cluster_lookup["domain"].astype(str) == domain].iloc[0]
            if not cluster_lookup.empty
            and (cluster_lookup["domain"].astype(str) == domain).any()
            else None
        )
        for segment_label, segment_frame in merged.groupby(segment_type, dropna=True):
            zip_count = int(len(segment_frame))
            avg_rate = float(segment_frame["total_rate_per_1000"].mean())
            avg_home_value = float(segment_frame["home_value"].mean())
            adverse_delta = float(pd.to_numeric(segment_frame.get("adverse_momentum"), errors="coerce").mean())
            shock_delta = float(pd.to_numeric(segment_frame.get("systemic_shock"), errors="coerce").mean())
            scenario_eligible_zip_count = int(
                segment_frame["zip"].astype("string").isin(scenario_supported_zips).sum()
            )
            scenario_coverage_share = _safe_ratio(scenario_eligible_zip_count, zip_count)
            high_influence_zip_count = int(
                segment_frame["zip"].astype("string").isin(flagged_influence_zips).sum()
            )
            high_influence_zip_share = _safe_ratio(high_influence_zip_count, zip_count)
            cluster_selected_k = (
                int(pd.to_numeric(pd.Series([domain_cluster.get("selected_k")]), errors="coerce").iloc[0])
                if domain_cluster is not None
                and pd.notna(pd.to_numeric(pd.Series([domain_cluster.get("selected_k")]), errors="coerce").iloc[0])
                else np.nan
            )
            cluster_min_cluster_size = (
                int(
                    pd.to_numeric(pd.Series([domain_cluster.get("min_cluster_size")]), errors="coerce").iloc[0]
                )
                if domain_cluster is not None
                and pd.notna(
                    pd.to_numeric(pd.Series([domain_cluster.get("min_cluster_size")]), errors="coerce").iloc[0]
                )
                else np.nan
            )
            cluster_min_cluster_size_threshold = (
                int(
                    pd.to_numeric(
                        pd.Series([domain_cluster.get("min_cluster_size_threshold")]),
                        errors="coerce",
                    ).iloc[0]
                )
                if domain_cluster is not None
                and pd.notna(
                    pd.to_numeric(
                        pd.Series([domain_cluster.get("min_cluster_size_threshold")]),
                        errors="coerce",
                    ).iloc[0]
                )
                else np.nan
            )
            cluster_silhouette_score = (
                float(
                    pd.to_numeric(pd.Series([domain_cluster.get("silhouette_score")]), errors="coerce").iloc[0]
                )
                if domain_cluster is not None
                else np.nan
            )
            cluster_leave_one_feature_out_mean_ari = (
                float(
                    pd.to_numeric(
                        pd.Series([domain_cluster.get("leave_one_feature_out_mean_ari")]),
                        errors="coerce",
                    ).iloc[0]
                )
                if domain_cluster is not None
                else np.nan
            )
            cluster_practical_utility_pass = (
                int(
                    pd.to_numeric(
                        pd.Series([domain_cluster.get("practical_utility_pass")]),
                        errors="coerce",
                    )
                    .fillna(0)
                    .iloc[0]
                )
                if domain_cluster is not None
                else 0
            )
            segment_size_pass = int(
                pd.notna(cluster_min_cluster_size_threshold) and zip_count >= cluster_min_cluster_size_threshold
            )
            guardrail_flags: list[str] = []
            if cluster_practical_utility_pass == 0:
                guardrail_flags.append("cluster_utility_fail")
            if scenario_coverage_share < SEGMENT_SCENARIO_COVERAGE_THRESHOLD:
                guardrail_flags.append("partial_scenario_support")
            if segment_size_pass == 0:
                guardrail_flags.append("small_segment")
            if high_influence_zip_share >= SEGMENT_HIGH_INFLUENCE_SHARE_THRESHOLD:
                guardrail_flags.append("high_influence_concentration")
            if (avg_rate >= global_rate_median) or (pd.notna(shock_delta) and shock_delta >= global_shock_median):
                priority = "high"
                actions = (
                    "Focused deterrence operations; hot-spot lighting and CPTED upgrades; "
                    "targeted rental stabilization support"
                )
            elif pd.notna(adverse_delta) and adverse_delta > 0:
                priority = "moderate"
                actions = (
                    "Problem-oriented patrol alignment; nuisance-property remediation; "
                    "school and transit corridor prevention programming"
                )
            else:
                priority = "watch"
                actions = (
                    "Maintain preventive services; monitor drift flags quarterly; "
                    "preserve affordability and neighborhood maintenance capacity"
                )
            if any(flag in guardrail_flags for flag in ("small_segment", "high_influence_concentration")):
                segment_guardrail_status = "blocked"
                segment_confidence_tier = "low"
            elif guardrail_flags:
                segment_guardrail_status = "caution"
                segment_confidence_tier = "guardrailed"
            else:
                segment_guardrail_status = "clear"
                segment_confidence_tier = "clear"
            if guardrail_flags:
                actions = f"{actions} Use only as exploratory segment framing while guardrail flags remain active."
            rows.append(
                {
                    "segment_type": segment_type,
                    "segment_label": str(segment_label),
                    "zip_count": zip_count,
                    "avg_total_rate_per_1000": avg_rate,
                    "avg_home_value": avg_home_value,
                    "avg_adverse_momentum_delta": adverse_delta,
                    "avg_systemic_shock_delta": shock_delta,
                    "priority_tier": priority,
                    "recommended_actions": actions,
                    "scenario_eligible_zip_count": scenario_eligible_zip_count,
                    "scenario_coverage_share": scenario_coverage_share,
                    "high_influence_zip_count": high_influence_zip_count,
                    "high_influence_zip_share": high_influence_zip_share,
                    "cluster_selected_k": cluster_selected_k,
                    "cluster_min_cluster_size": cluster_min_cluster_size,
                    "cluster_min_cluster_size_threshold": cluster_min_cluster_size_threshold,
                    "cluster_silhouette_score": cluster_silhouette_score,
                    "cluster_leave_one_feature_out_mean_ari": cluster_leave_one_feature_out_mean_ari,
                    "cluster_practical_utility_pass": cluster_practical_utility_pass,
                    "segment_size_pass": segment_size_pass,
                    "segment_guardrail_status": segment_guardrail_status,
                    "segment_confidence_tier": segment_confidence_tier,
                    "guardrail_flags": ";".join(guardrail_flags) if guardrail_flags else pd.NA,
                    "evidence_posture": "exploratory_non_causal",
                    "causal_interpretation_allowed": 0,
                }
            )

    policy_df = pd.DataFrame.from_records(rows, columns=columns)
    if policy_df.empty:
        notes.append("No segment groups with usable metrics were available for policy recommendations.")
        return policy_df, notes
    if (policy_df["segment_guardrail_status"].astype(str) != "clear").any():
        notes.append(
            "Segment rows now include scenario coverage, cluster-utility, segment-size, and high-influence "
            "guardrail fields so small or fragile segments are visibly downgraded in the CSV."
        )
    return policy_df.sort_values(
        ["priority_tier", "segment_type", "segment_label"],
        kind="mergesort",
        ignore_index=True,
    ), notes


def _write_trend_notes(
    decomposition_df: pd.DataFrame,
    notes: list[str],
    output_path: Path,
) -> None:
    lines = [
        "# Crime Trend Decomposition",
        "",
    ]
    if decomposition_df.empty:
        lines.extend(
            [
                "No decomposition rows were produced for this run.",
                "",
                *notes,
            ]
        )
    else:
        lines.extend(
            [
                f"Decomposition rows written: {len(decomposition_df)}.",
                "",
                "See `crime_trend_decomposition.csv` for quarterly observed, trend, seasonal, and residual values.",
            ]
        )
        if notes:
            lines.extend(["", "Additional notes:", "", *notes])
    output_path.write_text("\n".join(lines) + "\n")


def _write_forecast_notes(
    model_metrics_df: pd.DataFrame,
    forecasts_df: pd.DataFrame,
    temporal_holdout_df: pd.DataFrame,
    interval_calibration_df: pd.DataFrame,
    notes: list[str],
    output_path: Path,
) -> None:
    lines = [
        "# Crime Forecast Notes",
        "",
    ]
    if forecasts_df.empty:
        lines.extend(
            [
                "No crime forecasts were produced for this run.",
                "",
                *notes,
            ]
        )
    else:
        selected = (
            forecasts_df.loc[
                forecasts_df.get("forecast_tier", pd.Series("high_confidence", index=forecasts_df.index))
                .astype(str)
                == "high_confidence",
                ["zip", "selected_model"],
            ]
            .drop_duplicates()
            .sort_values(["selected_model", "zip"], kind="mergesort")
        )
        eligible_zip_count = (
            int(pd.to_numeric(model_metrics_df["eligible_zip_count"], errors="coerce").max())
            if not model_metrics_df.empty
            else int(
                forecasts_df.loc[
                    forecasts_df.get("policy_eligible", pd.Series(1, index=forecasts_df.index))
                    .astype(int)
                    == 1,
                    "zip",
                ]
                .astype(str)
                .nunique()
            )
        )
        gated_out_zip_count = (
            int(pd.to_numeric(model_metrics_df["gated_out_zip_count"], errors="coerce").max())
            if not model_metrics_df.empty
            else 0
        )
        modeled_zip_count = eligible_zip_count + gated_out_zip_count
        tier_counts = (
            forecasts_df[["zip", "forecast_tier"]]
            .drop_duplicates()
            .groupby("forecast_tier")["zip"]
            .nunique()
            if "forecast_tier" in forecasts_df.columns
            else pd.Series(dtype=int)
        )
        limited_history_count = int(tier_counts.get("limited_history", 0))
        carry_forward_count = int(tier_counts.get("carry_forward_only", 0))
        lines.append(f"Forecast rows written: {len(forecasts_df)}.")
        lines.append("")
        if not model_metrics_df.empty:
            lines.append("Model family comparison is recorded in `forecast_model_metrics.csv`.")
            lines.append("")
        if not temporal_holdout_df.empty:
            selected_holdout = temporal_holdout_df.loc[
                temporal_holdout_df["evaluation_scope"].astype(str) == "selected_zip_model"
            ]
            if not selected_holdout.empty:
                row = selected_holdout.iloc[0]
                lines.append(
                    "Temporal holdout results are recorded in `temporal_holdout_results.csv` "
                    f"(selected ZIP model MAPE={row['mape']:.2f}, pass={int(row['mape_pass'])})."
                )
                lines.append("")
        if not interval_calibration_df.empty:
            lines.append(
                "Interval calibration diagnostics are recorded in `forecast_interval_calibration.csv`."
            )
            lines.append("")
        lines.append("Coverage tiers:")
        lines.append("")
        lines.append(
            f"- High-confidence forecast/scenario coverage is {eligible_zip_count}/{modeled_zip_count} modeled ZIPs."
        )
        if limited_history_count > 0:
            lines.append(
                f"- Limited-history forecast-only coverage adds {limited_history_count} modeled ZIPs with {FORECAST_LIMITED_HISTORY_MIN_QUARTERS}-{FORECAST_HISTORY_MIN_QUARTERS - 1} trailing quarters."
            )
        if carry_forward_count > 0:
            lines.append(
                f"- Carry-forward forecast-only coverage adds {carry_forward_count} modeled ZIPs with 1-{FORECAST_LIMITED_HISTORY_MIN_QUARTERS - 1} trailing quarters."
            )
        remaining_unforecasted = max(
            modeled_zip_count - eligible_zip_count - limited_history_count - carry_forward_count,
            0,
        )
        if remaining_unforecasted > 0:
            lines.append(
                f"- The remaining {remaining_unforecasted} modeled ZIPs still lacked enough contiguous history for any forecast tier."
            )
        lines.append("- Scenario and policy artifacts remain limited to the high-confidence subset.")
        lines.append("")
        lines.append("Selected high-confidence ZIP-level models:")
        lines.append("")
        for row in selected.itertuples(index=False):
            lines.append(f"- ZIP {row.zip}: {row.selected_model}")
        if notes:
            lines.extend(["", "Additional notes:", "", *notes])
    output_path.write_text("\n".join(lines) + "\n")


def _write_scenario_notes(
    forecast_model_metrics: pd.DataFrame,
    crime_forecasts: pd.DataFrame,
    output_path: Path,
) -> None:
    eligible_zip_count = (
        int(pd.to_numeric(forecast_model_metrics["eligible_zip_count"], errors="coerce").max())
        if not forecast_model_metrics.empty
        else 0
    )
    gated_out_zip_count = (
        int(pd.to_numeric(forecast_model_metrics["gated_out_zip_count"], errors="coerce").max())
        if not forecast_model_metrics.empty
        else 0
    )
    modeled_zip_count = eligible_zip_count + gated_out_zip_count
    lower_tier_zip_count = (
        int(
            crime_forecasts.loc[
                crime_forecasts.get("policy_eligible", pd.Series(1, index=crime_forecasts.index))
                .astype(int)
                == 0,
                "zip",
            ]
            .astype(str)
            .nunique()
        )
        if not crime_forecasts.empty
        else 0
    )
    lines = [
        "# Scenario Notes",
        "",
        "Scenario impacts are deterministic multipliers applied to the ZIP-level crime forecast output.",
        "They are exploratory planning aids, not causal predictions or policy-effect estimates.",
        "Use them only with `policy_guardrails.md` and the temporal holdout / interval calibration diagnostics.",
        (
            f"Scenario rows are emitted only for the {eligible_zip_count}/{modeled_zip_count} modeled ZIPs "
            "that passed high-confidence forecast eligibility."
            if modeled_zip_count > 0
            else "Scenario rows are emitted only when forecast-eligible ZIPs are available."
        ),
        (
            f"An additional {lower_tier_zip_count} modeled ZIPs receive forecast-only lower-confidence tiers, but they are excluded from scenario rows."
            if lower_tier_zip_count > 0
            else "No lower-confidence forecast-only ZIP tiers were emitted in this run."
        ),
        "",
        "- `baseline`: uses the point forecast unchanged.",
        "- `stabilization`: applies a 5% improvement to forecasted crime rate.",
        "- `adverse_momentum`: applies a 10% worsening to forecasted crime rate.",
        "- `seasonal_peak`: applies a modest uplift, with a larger seasonal bump in Q2 and Q3.",
        "- `systemic_shock`: applies a severe system-wide stress multiplier.",
    ]
    output_path.write_text("\n".join(lines) + "\n")


def _write_benchmark_summary(benchmarks_df: pd.DataFrame, output_path: Path) -> None:
    lines = [
        "# Benchmark Summary",
        "",
    ]
    if benchmarks_df.empty:
        lines.append("No benchmark rows were produced for this run.")
        output_path.write_text("\n".join(lines) + "\n")
        return

    top_home_value = benchmarks_df.sort_values("home_value", ascending=False).head(3)
    top_crime = benchmarks_df.sort_values("total_rate_per_1000", ascending=False).head(3)
    lines.extend(
        [
            f"ZIP benchmark rows written: {len(benchmarks_df)}.",
            "",
            "## Highest Home Value ZIPs",
            "",
            "| zip | home_value | home_value_vs_metro_pct |",
            "| --- | ---: | ---: |",
        ]
    )
    for row in top_home_value.itertuples(index=False):
        lines.append(
            f"| {row.zip} | {row.home_value:.0f} | {row.home_value_vs_metro_pct:.2f} |"
        )

    lines.extend(
        [
            "",
            "## Highest Crime-Rate ZIPs",
            "",
            "| zip | total_rate_per_1000 | total_rate_vs_metro_pct |",
            "| --- | ---: | ---: |",
        ]
    )
    for row in top_crime.itertuples(index=False):
        lines.append(
            f"| {row.zip} | {row.total_rate_per_1000:.2f} | {row.total_rate_vs_metro_pct:.2f} |"
        )

    output_path.write_text("\n".join(lines) + "\n")


def _write_drift_notes(notes: list[str], output_path: Path) -> None:
    lines = [
        "# Model Drift Notes",
        "",
        "Drift diagnostics compare the latest available quarter against the trailing four-quarter baseline.",
        "",
    ]
    if notes:
        lines.extend(notes)
    else:
        lines.append("No additional drift caveats were recorded.")
    output_path.write_text("\n".join(lines) + "\n")


def _write_feature_selection_notes(
    selection_df: pd.DataFrame,
    notes: list[str],
    output_path: Path,
) -> None:
    lines = [
        "# Feature Selection Notes",
        "",
    ]
    if selection_df.empty:
        lines.append("Feature-selection metrics were not produced for this run.")
        if notes:
            lines.extend(["", *notes])
        output_path.write_text("\n".join(lines) + "\n")
        return

    recommended = selection_df.loc[
        selection_df["recommended_for_future_models"] == 1,
        "feature_name",
    ].astype(str)
    interpretable = selection_df.loc[
        pd.to_numeric(selection_df["interpretation_allowed"], errors="coerce").fillna(0).astype(int) == 1,
        "feature_name",
    ].astype(str)
    lines.append(f"Feature candidates evaluated: {len(selection_df)}.")
    lines.append("")
    lines.append(
        "Recommended features (by availability-weighted association score): "
        f"{', '.join(recommended.tolist()) if not recommended.empty else 'none'}."
    )
    lines.append("")
    lines.append(
        "Features clearing both FDR and practical-effect guardrails: "
        f"{', '.join(interpretable.tolist()) if not interpretable.empty else 'none'}."
    )
    if notes:
        lines.extend(["", "Additional notes:", "", *notes])
    output_path.write_text("\n".join(lines) + "\n")


def _write_feature_power_retention_notes(
    power_metrics: pd.DataFrame,
    notes: list[str],
    output_path: Path,
) -> None:
    lines = [
        "# Feature Power Retention Notes",
        "",
        "This artifact tracks the Q2 feature-power checkpoint with explicit retention thresholds.",
        "",
    ]
    if power_metrics.empty:
        lines.append("Feature-power retention metrics were not produced for this run.")
        if notes:
            lines.extend(["", *notes])
        output_path.write_text("\n".join(lines) + "\n")
        return

    metric_to_value = {
        str(row.metric): row.value for row in power_metrics.itertuples(index=False)
    }
    score_retention = pd.to_numeric(
        pd.Series([metric_to_value.get("feature_selection_score_retention_ratio")]),
        errors="coerce",
    ).iloc[0]
    r_squared_retention = pd.to_numeric(
        pd.Series([metric_to_value.get("predictive_r_squared_retention_ratio")]),
        errors="coerce",
    ).iloc[0]
    checkpoint_pass = pd.to_numeric(
        pd.Series([metric_to_value.get("feature_power_checkpoint_pass")]),
        errors="coerce",
    ).iloc[0]

    lines.append(
        f"- Feature-selection score retention ratio: {score_retention:.3f}"
        if pd.notna(score_retention)
        else "- Feature-selection score retention ratio: not estimable."
    )
    lines.append(
        f"- Predictive R-squared retention ratio: {r_squared_retention:.3f}"
        if pd.notna(r_squared_retention)
        else "- Predictive R-squared retention ratio: not estimable."
    )
    lines.append(
        f"- Checkpoint pass flag: {int(checkpoint_pass)}"
        if pd.notna(checkpoint_pass)
        else "- Checkpoint pass flag: not estimable."
    )
    lines.extend(
        [
            "",
            "Threshold rule: both retention ratios must be >= 0.90.",
        ]
    )
    if notes:
        lines.extend(["", "Additional notes:", "", *notes])
    output_path.write_text("\n".join(lines) + "\n")


def _write_model_family_notes(
    predictive_metrics: pd.DataFrame,
    notes: list[str],
    output_path: Path,
) -> None:
    lines = [
        "# Predictive Model Family Notes",
        "",
    ]
    if predictive_metrics.empty:
        lines.append("Predictive model-family metrics were not produced for this run.")
        if notes:
            lines.extend(["", *notes])
        output_path.write_text("\n".join(lines) + "\n")
        return

    ranked = predictive_metrics.sort_values(
        ["selection_rank", "rmse", "model_label"],
        ascending=[True, True, True],
        na_position="last",
        kind="mergesort",
    )
    lines.append(f"Model-family rows written: {len(predictive_metrics)}.")
    lines.append("")
    lines.append("Top-ranked models by selection rank:")
    lines.append("")
    for row in ranked.head(3).itertuples(index=False):
        lines.append(
            f"- `{row.model_label}` ({row.model_tier}) "
            f"rmse={row.rmse:.4f} loocv_rmse={row.loocv_rmse if pd.notna(row.loocv_rmse) else float('nan'):.4f}"
        )
    if notes:
        lines.extend(["", "Additional notes:", "", *notes])
    output_path.write_text("\n".join(lines) + "\n")


def _write_comprehensive_validation_notes(notes: list[str], output_path: Path) -> None:
    lines = [
        "# Comprehensive Validation Notes",
        "",
        "This artifact aggregates regression, predictive-family, forecast, temporal holdout, interval calibration, influence robustness, cluster stability, drift, scenario, and statistical guardrail checks.",
    ]
    if notes:
        lines.extend(["", *notes])
    output_path.write_text("\n".join(lines) + "\n")


def _write_policy_recommendations_notes(
    policy_df: pd.DataFrame,
    notes: list[str],
    output_path: Path,
) -> None:
    lines = [
        "# Policy Recommendations By Segment",
        "",
    ]
    if policy_df.empty:
        lines.append("No policy recommendations were generated for this run.")
        if notes:
            lines.extend(["", *notes])
        output_path.write_text("\n".join(lines) + "\n")
        return

    high_priority = policy_df.loc[policy_df["priority_tier"] == "high"]
    blocked_segments = policy_df.loc[
        policy_df.get("segment_guardrail_status", pd.Series("clear", index=policy_df.index)).astype(str)
        == "blocked"
    ]
    lines.append(f"Segment recommendations written: {len(policy_df)}.")
    lines.append("")
    lines.append(
        f"High-priority segments: {len(high_priority)} "
        f"({', '.join(high_priority['segment_label'].astype(str).tolist()) if not high_priority.empty else 'none'})."
    )
    lines.append(
        f"Blocked / low-confidence segments: {len(blocked_segments)} "
        f"({', '.join(blocked_segments['segment_label'].astype(str).tolist()) if not blocked_segments.empty else 'none'})."
    )
    lines.append("")
    lines.append(
        "See `policy_recommendations_by_segment.csv` for segment guardrail status, confidence tiers, "
        "scenario support, and high-influence concentrations."
    )
    lines.append("These segment recommendations remain exploratory and non-causal; see `policy_guardrails.md`.")
    if notes:
        lines.extend(["", "Additional notes:", "", *notes])
    output_path.write_text("\n".join(lines) + "\n")


def _write_policy_guardrails(
    *,
    model_df: pd.DataFrame,
    target_universe: pd.DataFrame,
    crime_forecasts: pd.DataFrame,
    temporal_holdout: pd.DataFrame,
    interval_calibration: pd.DataFrame,
    influence_summary: pd.DataFrame,
    cluster_stability: pd.DataFrame,
    statistical_guardrails: pd.DataFrame,
    output_path: Path,
) -> None:
    modeled_zip_count = int(pd.Series(model_df.get("zip"), dtype="string").dropna().nunique())
    target_zip_count = (
        int(pd.Series(target_universe.get("zip"), dtype="string").dropna().nunique())
        if not target_universe.empty and "zip" in target_universe.columns
        else modeled_zip_count
    )
    selected_holdout = temporal_holdout.loc[
        temporal_holdout["evaluation_scope"].astype(str) == "selected_zip_model"
    ]
    selected_calibration = interval_calibration.loc[
        interval_calibration["evaluation_scope"].astype(str) == "selected_zip_model"
    ]
    holdout_pass = (
        bool(
            pd.to_numeric(selected_holdout["mape_pass"], errors="coerce").fillna(0).eq(1).all()
        )
        if not selected_holdout.empty
        else False
    )
    calibration_pass = (
        bool(
            pd.to_numeric(selected_calibration["calibration_pass"], errors="coerce")
            .fillna(0)
            .eq(1)
            .all()
        )
        if not selected_calibration.empty
        else False
    )
    influence_pass = (
        bool(
            pd.to_numeric(influence_summary["influence_robustness_pass"], errors="coerce")
            .fillna(0)
            .eq(1)
            .all()
        )
        if not influence_summary.empty
        else False
    )
    cluster_pass = (
        bool(
            pd.to_numeric(cluster_stability["practical_utility_pass"], errors="coerce")
            .fillna(0)
            .eq(1)
            .all()
        )
        if not cluster_stability.empty
        else False
    )
    interpretation_share = (
        float(pd.to_numeric(statistical_guardrails["interpretation_allowed"], errors="coerce").mean())
        if not statistical_guardrails.empty
        else np.nan
    )
    forecast_high_confidence_count = (
        int(
            crime_forecasts.loc[
                crime_forecasts.get("policy_eligible", pd.Series(1, index=crime_forecasts.index))
                .astype(int)
                == 1,
                "zip",
            ]
            .astype(str)
            .nunique()
        )
        if not crime_forecasts.empty
        else 0
    )
    lower_confidence_forecast_count = (
        int(
            crime_forecasts.loc[
                crime_forecasts.get("policy_eligible", pd.Series(1, index=crime_forecasts.index))
                .astype(int)
                == 0,
                "zip",
            ]
            .astype(str)
            .nunique()
        )
        if not crime_forecasts.empty
        else 0
    )
    eligible_zip_count = (
        int(pd.to_numeric(selected_holdout["eligible_zip_count"], errors="coerce").max())
        if not selected_holdout.empty
        else forecast_high_confidence_count
    )
    gated_out_zip_count = max(modeled_zip_count - eligible_zip_count, 0)
    selected_mape = (
        float(pd.to_numeric(selected_holdout["mape"], errors="coerce").iloc[0])
        if not selected_holdout.empty
        else np.nan
    )
    selected_mape_threshold = (
        float(pd.to_numeric(selected_holdout["mape_threshold"], errors="coerce").iloc[0])
        if not selected_holdout.empty
        else np.nan
    )
    influence_fail_count = (
        int(
            pd.to_numeric(influence_summary["influence_robustness_pass"], errors="coerce")
            .fillna(0)
            .eq(0)
            .sum()
        )
        if not influence_summary.empty
        else 0
    )
    max_violent_shift = (
        float(pd.to_numeric(influence_summary["max_violent_effect_pct_change"], errors="coerce").max())
        if not influence_summary.empty
        else np.nan
    )
    max_property_shift = (
        float(pd.to_numeric(influence_summary["max_property_effect_pct_change"], errors="coerce").max())
        if not influence_summary.empty
        else np.nan
    )
    max_prediction_delta = (
        float(pd.to_numeric(influence_summary["max_p90_home_value_pct_delta"], errors="coerce").max())
        if not influence_summary.empty
        else np.nan
    )
    fit_warning_count = (
        int(
            pd.to_numeric(influence_summary["fit_improvement_warning_count"], errors="coerce")
            .fillna(0)
            .gt(0)
            .sum()
        )
        if not influence_summary.empty
        else 0
    )
    cluster_fail_count = (
        int(
            pd.to_numeric(cluster_stability["practical_utility_pass"], errors="coerce")
            .fillna(0)
            .eq(0)
            .sum()
        )
        if not cluster_stability.empty
        else 0
    )

    lines = [
        "# Policy Guardrails",
        "",
        "Scenario and segment outputs in this repo are exploratory, non-causal planning aids.",
        "They must not be described as policy-impact estimates, intervention effects, or decision-ready resource-allocation evidence.",
        "",
        f"- Modeled ZIP coverage in this run: {modeled_zip_count}/{target_zip_count}.",
        f"- High-confidence forecast/scenario coverage in this run: {eligible_zip_count}/{modeled_zip_count} modeled ZIPs.",
        (
            f"- Lower-confidence forecast-only coverage adds {lower_confidence_forecast_count} modeled ZIPs; those rows are excluded from scenario and policy calculations."
            if lower_confidence_forecast_count > 0
            else f"- No lower-confidence forecast-only ZIP tiers were emitted; {gated_out_zip_count} modeled ZIPs remain uncovered."
        ),
        (
            f"- Selected ZIP temporal holdout: MAPE={selected_mape:.3f} versus threshold {selected_mape_threshold:.1f}."
            if pd.notna(selected_mape) and pd.notna(selected_mape_threshold)
            else "- Selected ZIP temporal holdout: not estimable."
        ),
        f"- Temporal holdout pass status: {int(holdout_pass)}.",
        f"- Interval calibration pass status: {int(calibration_pass)}.",
        (
            f"- Influence robustness severity: {influence_fail_count}/{len(influence_summary)} regression specs fail on prediction-stability guardrails; max p90 home-value delta={max_prediction_delta:.3f}%, max violent-term shift={max_violent_shift:.3f}%, max property-term shift={max_property_shift:.3f}%."
            if not influence_summary.empty
            and pd.notna(max_prediction_delta)
            and pd.notna(max_violent_shift)
            and pd.notna(max_property_shift)
            else "- Influence robustness severity: not estimable."
        ),
        f"- Influence robustness pass status: {int(influence_pass)}.",
        f"- Influence fit-improvement warning count: {fit_warning_count}/{len(influence_summary)} regression specs.",
        (
            f"- Cluster practical-utility severity: {cluster_fail_count}/{len(cluster_stability)} domains fail."
            if not cluster_stability.empty
            else "- Cluster practical-utility severity: not estimable."
        ),
        f"- Cluster practical-utility pass status: {int(cluster_pass)}.",
        (
            f"- Share of statistical tests clearing FDR + practical-effect guardrails: {interpretation_share:.3f}."
            if pd.notna(interpretation_share)
            else "- Share of statistical tests clearing FDR + practical-effect guardrails: not estimable."
        ),
        "",
        "Regression-linked crime terms remain interpretation-blocked whenever statistical guardrails or practical-effect checks fail, even if influence robustness improves.",
        "Decision rule for this repo: keep scenario and policy artifacts in descriptive/exploratory mode unless the temporal holdout, interval calibration, influence robustness, and cluster utility checks all pass.",
    ]
    output_path.write_text("\n".join(lines) + "\n")


def run_analysis(settings: "Settings") -> dict[str, str]:
    """Run regression analysis from the processed model dataset and write report artifacts."""

    model_path = settings.processed_dir / "model_dataset.csv"
    if not model_path.exists():
        raise FileNotFoundError(f"Processed model dataset not found at {model_path}")

    model_df = pd.read_csv(model_path)
    target_universe_path = settings.processed_dir / "target_zip_universe.csv"
    target_universe = pd.read_csv(target_universe_path) if target_universe_path.exists() else pd.DataFrame()
    optional_inputs = _load_optional_analysis_inputs(settings)
    crime_history_panel = optional_inputs["crime_history_panel"]
    _housing_history_panel = optional_inputs["housing_history_panel"]
    modeled_zips = set(pd.Series(model_df.get("zip"), dtype="string").dropna().astype(str))
    temporal_summary, temporal_series, temporal_notes = _prepare_temporal_analysis_inputs(
        crime_history_panel,
        modeled_zips=modeled_zips,
    )

    baseline_result = run_zip_regression(
        model_df,
        predictors=DEFAULT_PREDICTORS,
        controls=DEFAULT_CONTROLS,
        model_label="baseline",
    )
    expanded_controls = _select_expanded_controls(
        model_df,
        dependent="log_home_value",
        predictors=DEFAULT_PREDICTORS,
        baseline_controls=DEFAULT_CONTROLS,
    )
    expanded_result = run_zip_regression(
        model_df,
        predictors=DEFAULT_PREDICTORS,
        controls=expanded_controls,
        model_label="expanded_controls",
    )
    results = [baseline_result, expanded_result]
    if expanded_result.nobs < baseline_result.nobs:
        _fhfa_drop = baseline_result.nobs - expanded_result.nobs
        print(
            f"[analyze] NOTE (DQA-D1): expanded_controls model dropped {_fhfa_drop} observation(s) "
            f"vs baseline (n={baseline_result.nobs} → n={expanded_result.nobs}). "
            "This is likely due to FHFA_annual_change_pct missingness in 10 ZIPs; "
            "3 of those are high-influence observations. "
            "See model_validation_notes.md for discussion.",
            flush=True,
        )

    coefficients_path = settings.reports_dir / "regression_coefficients.csv"
    metrics_path = settings.reports_dir / "regression_metrics.csv"
    sample_sizes_path = settings.reports_dir / "model_sample_sizes.csv"
    residuals_path = settings.reports_dir / "model_residuals.csv"
    residual_review_path = settings.reports_dir / "residual_review.md"
    vif_path = settings.reports_dir / "model_vif.csv"
    vif_notes_path = settings.reports_dir / "model_vif_notes.md"
    scatter_path = settings.reports_dir / "home_value_vs_total_crime.png"
    geography_path = settings.reports_dir / "crime_home_value_geography.png"
    cluster_assignments_path = settings.reports_dir / "cluster_assignments.csv"
    cluster_profiles_path = settings.reports_dir / "cluster_profiles.csv"
    spatial_diagnostics_path = settings.reports_dir / "spatial_diagnostics.csv"
    spatial_hotspots_path = settings.reports_dir / "spatial_hotspots.csv"
    validation_metrics_path = settings.reports_dir / "model_validation_metrics.csv"
    validation_notes_path = settings.reports_dir / "model_validation_notes.md"
    zip_comparison_path = settings.reports_dir / "top_bottom_zip_comparison.md"
    model_summary_table_path = settings.reports_dir / "model_summary_table.md"
    trend_decomposition_path = settings.reports_dir / "crime_trend_decomposition.csv"
    trend_notes_path = settings.reports_dir / "crime_trend_decomposition.md"
    feature_selection_metrics_path = settings.reports_dir / "feature_selection_metrics.csv"
    feature_selection_notes_path = settings.reports_dir / "feature_selection_notes.md"
    feature_power_retention_metrics_path = settings.reports_dir / "feature_power_retention_metrics.csv"
    feature_power_retention_notes_path = settings.reports_dir / "feature_power_retention_notes.md"
    forecast_model_metrics_path = settings.reports_dir / "forecast_model_metrics.csv"
    crime_forecasts_path = settings.reports_dir / "crime_forecasts.csv"
    forecast_intervals_path = settings.reports_dir / "forecast_confidence_intervals.csv"
    forecast_notes_path = settings.reports_dir / "forecast_notes.md"
    predictive_model_metrics_path = settings.reports_dir / "predictive_model_metrics.csv"
    predictive_model_predictions_path = settings.reports_dir / "predictive_model_predictions.csv"
    model_selection_notes_path = settings.reports_dir / "model_selection_notes.md"
    scenario_impacts_path = settings.reports_dir / "scenario_impacts.csv"
    scenario_notes_path = settings.reports_dir / "scenario_notes.md"
    zip_benchmarks_path = settings.reports_dir / "zip_benchmarks.csv"
    benchmark_summary_path = settings.reports_dir / "benchmark_summary.md"
    drift_diagnostics_path = settings.reports_dir / "model_drift_diagnostics.csv"
    drift_notes_path = settings.reports_dir / "model_drift_notes.md"
    temporal_holdout_results_path = settings.reports_dir / "temporal_holdout_results.csv"
    forecast_interval_calibration_path = settings.reports_dir / "forecast_interval_calibration.csv"
    influence_robustness_path = settings.reports_dir / "influence_robustness_diagnostics.csv"
    cluster_stability_path = settings.reports_dir / "cluster_stability_diagnostics.csv"
    statistical_guardrails_path = settings.reports_dir / "statistical_guardrails.csv"
    comprehensive_validation_metrics_path = (
        settings.reports_dir / "comprehensive_validation_metrics.csv"
    )
    comprehensive_validation_notes_path = (
        settings.reports_dir / "comprehensive_validation_notes.md"
    )
    policy_recommendations_path = settings.reports_dir / "policy_recommendations_by_segment.csv"
    policy_recommendations_notes_path = (
        settings.reports_dir / "policy_recommendations_by_segment.md"
    )
    policy_guardrails_path = settings.reports_dir / "policy_guardrails.md"
    report_path = settings.reports_dir / "summary.md"

    coefficients = _apply_regression_guardrails(
        pd.concat([result.coefficients for result in results], ignore_index=True)
    )
    metrics = pd.concat([result.metrics_table() for result in results], ignore_index=True)
    sample_sizes = metrics[["model_label", "nobs", "predictors", "controls", "formula"]].copy()
    residuals = (
        pd.concat([result.residuals for result in results], ignore_index=True)
        .sort_values(["model_label", "absolute_residual"], ascending=[True, False])
        .reset_index(drop=True)
    )
    vif_table, vif_notes = _build_vif_artifacts(results)
    cluster_assignments, cluster_profiles = _build_segmentation_artifacts(model_df)
    cluster_stability = _build_cluster_stability_artifacts(model_df)
    spatial_diagnostics, spatial_hotspots = _build_spatial_artifacts(model_df)
    feature_selection_metrics, feature_selection_notes = _build_feature_selection_artifacts(
        model_df,
        expanded_controls=expanded_controls,
    )
    predictive_model_metrics, predictive_model_predictions, model_selection_notes = (
        _build_predictive_model_family_artifacts(
            model_df,
            baseline_result=baseline_result,
            expanded_result=expanded_result,
            cluster_assignments=cluster_assignments,
        )
    )
    feature_power_retention_metrics, feature_power_retention_notes = (
        _build_feature_power_retention_artifacts(
            feature_selection_metrics,
            predictive_model_metrics,
        )
    )
    validation_metrics, validation_notes = _build_validation_artifacts(results)
    influence_robustness, influence_notes = _build_influence_robustness_artifacts(results)
    influence_summary = _summarize_influence_robustness(influence_robustness)
    residuals = _annotate_residuals_with_influence_flags(residuals, influence_robustness)
    if not influence_summary.empty:
        validation_metrics = validation_metrics.merge(
            influence_summary,
            on="model_label",
            how="left",
        )
        for row in influence_summary.itertuples(index=False):
            validation_notes.append(
                f"- `{row.model_label}`: {int(row.influence_flag_count)} high-influence ZIPs were flagged; "
                f"max violent-term shift={float(row.max_violent_effect_pct_change):.3f}%, "
                f"max property-term shift={float(row.max_property_effect_pct_change):.3f}%, "
                f"max p90 home-value prediction delta={float(row.max_p90_home_value_pct_delta):.3f}%, "
                f"fit_deterioration_pass={int(row.fit_deterioration_pass)}, "
                f"fit_improvement_warning_count={int(row.fit_improvement_warning_count)}, "
                f"pass={int(row.influence_robustness_pass)}."
            )
    else:
        for column in (
            "influence_flag_count",
            "max_violent_effect_pct_change",
            "max_property_effect_pct_change",
            "max_p90_home_value_pct_delta",
            "max_home_value_pct_delta",
            "fit_stability_pass",
            "fit_deterioration_pass",
            "fit_improvement_warning_count",
            "any_crime_term_sign_flip",
            "influence_robustness_pass",
        ):
            validation_metrics[column] = np.nan
    validation_notes.extend(influence_notes)
    if not influence_robustness.empty:
        validation_notes.append(
            "See `influence_robustness_diagnostics.csv` for leave-high-leverage / high-Cook's ZIP-out refits."
        )
    trend_decomposition, trend_notes = _build_trend_decomposition_artifacts(crime_history_panel)
    temporal_holdout_results, interval_calibration_holdout, holdout_notes = (
        _build_temporal_holdout_artifacts(temporal_summary, temporal_series)
    )
    forecast_model_metrics, crime_forecasts, forecast_intervals, forecast_notes = (
        _build_forecast_artifacts(
            temporal_summary,
            temporal_series,
            temporal_holdout=temporal_holdout_results,
        )
    )
    forecast_notes = [*temporal_notes, *holdout_notes, *forecast_notes]
    interval_calibration = pd.concat(
        [
            interval_calibration_holdout,
            _build_interval_shape_artifacts(forecast_intervals),
        ],
        ignore_index=True,
    )
    scenario_impacts = _build_scenario_artifacts(crime_forecasts)
    zip_benchmarks = _build_benchmark_artifacts(model_df, cluster_assignments)
    _drift_data_cutoff = pd.Timestamp.today().normalize()
    drift_diagnostics, drift_notes = _build_drift_artifacts(
        temporal_summary,
        temporal_series,
        data_cutoff=_drift_data_cutoff,
    )
    statistical_guardrails = _build_statistical_guardrails_artifacts(
        coefficients,
        feature_selection_metrics,
        spatial_diagnostics,
    )
    comprehensive_validation_metrics, comprehensive_validation_notes = (
        _build_comprehensive_validation_artifacts(
            regression_validation=validation_metrics,
            predictive_metrics=predictive_model_metrics,
            forecast_metrics=forecast_model_metrics,
            temporal_holdout=temporal_holdout_results,
            interval_calibration=interval_calibration,
            drift_diagnostics=drift_diagnostics,
            influence_robustness=influence_robustness,
            cluster_stability=cluster_stability,
            statistical_guardrails=statistical_guardrails,
            scenario_impacts=scenario_impacts,
        )
    )
    policy_recommendations, policy_notes = _build_policy_recommendations_artifacts(
        zip_benchmarks,
        scenario_impacts,
        cluster_stability=cluster_stability,
        influence_robustness=influence_robustness,
    )
    policy_notes.append("See `policy_guardrails.md` before using any scenario or segment output.")
    if not influence_summary.empty:
        if (
            pd.to_numeric(influence_summary["influence_robustness_pass"], errors="coerce")
            .fillna(0)
            .eq(1)
            .all()
        ):
            policy_notes.append(
                "Regression-linked crime terms remain interpretation-blocked on current data because "
                "the FDR/practical-effect guardrails still do not clear the crime terms, even though "
                "prediction-stability influence checks improved."
            )
        else:
            policy_notes.append(
                "Regression-linked crime terms remain interpretation-blocked on current data because "
                "influence robustness fails in at least one modeled specification."
            )

    coefficients.to_csv(coefficients_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    sample_sizes.to_csv(sample_sizes_path, index=False)
    residuals.to_csv(residuals_path, index=False)
    vif_table.to_csv(vif_path, index=False)
    cluster_assignments.to_csv(cluster_assignments_path, index=False)
    cluster_profiles.to_csv(cluster_profiles_path, index=False)
    cluster_stability.to_csv(cluster_stability_path, index=False)
    spatial_diagnostics.to_csv(spatial_diagnostics_path, index=False)
    spatial_hotspots.to_csv(spatial_hotspots_path, index=False)
    feature_selection_metrics.to_csv(feature_selection_metrics_path, index=False)
    feature_power_retention_metrics.to_csv(feature_power_retention_metrics_path, index=False)
    predictive_model_metrics.to_csv(predictive_model_metrics_path, index=False)
    predictive_model_predictions.to_csv(predictive_model_predictions_path, index=False)
    validation_metrics.to_csv(validation_metrics_path, index=False)
    trend_decomposition.to_csv(trend_decomposition_path, index=False)
    forecast_model_metrics.to_csv(forecast_model_metrics_path, index=False)
    crime_forecasts.to_csv(crime_forecasts_path, index=False)
    forecast_intervals.to_csv(forecast_intervals_path, index=False)
    temporal_holdout_results.to_csv(temporal_holdout_results_path, index=False)
    interval_calibration.to_csv(forecast_interval_calibration_path, index=False)
    scenario_impacts.to_csv(scenario_impacts_path, index=False)
    zip_benchmarks.to_csv(zip_benchmarks_path, index=False)
    drift_diagnostics.to_csv(drift_diagnostics_path, index=False)
    influence_robustness.to_csv(influence_robustness_path, index=False)
    statistical_guardrails.to_csv(statistical_guardrails_path, index=False)
    comprehensive_validation_metrics.to_csv(comprehensive_validation_metrics_path, index=False)
    policy_recommendations.to_csv(policy_recommendations_path, index=False)
    _write_residual_review(residuals, residual_review_path)
    _write_vif_notes(vif_notes, vif_notes_path)
    _write_validation_notes(validation_notes, validation_notes_path)
    _write_trend_notes(trend_decomposition, trend_notes, trend_notes_path)
    _write_feature_selection_notes(
        feature_selection_metrics,
        feature_selection_notes,
        feature_selection_notes_path,
    )
    _write_feature_power_retention_notes(
        feature_power_retention_metrics,
        feature_power_retention_notes,
        feature_power_retention_notes_path,
    )
    _write_model_family_notes(
        predictive_model_metrics,
        model_selection_notes,
        model_selection_notes_path,
    )
    _write_forecast_notes(
        forecast_model_metrics,
        crime_forecasts,
        temporal_holdout_results,
        interval_calibration,
        forecast_notes,
        forecast_notes_path,
    )
    _write_scenario_notes(forecast_model_metrics, crime_forecasts, scenario_notes_path)
    _write_benchmark_summary(zip_benchmarks, benchmark_summary_path)
    _write_drift_notes(drift_notes, drift_notes_path)
    _write_comprehensive_validation_notes(
        comprehensive_validation_notes,
        comprehensive_validation_notes_path,
    )
    _write_policy_recommendations_notes(
        policy_recommendations,
        policy_notes,
        policy_recommendations_notes_path,
    )
    _write_policy_guardrails(
        model_df=model_df,
        target_universe=target_universe,
        crime_forecasts=crime_forecasts,
        temporal_holdout=temporal_holdout_results,
        interval_calibration=interval_calibration,
        influence_summary=influence_summary,
        cluster_stability=cluster_stability,
        statistical_guardrails=statistical_guardrails,
        output_path=policy_guardrails_path,
    )
    _write_scatter_plot(model_df, scatter_path)
    _write_geography_plot(model_df, geography_path)
    _write_zip_comparison_table(model_df, zip_comparison_path)
    _write_model_summary_table(metrics, model_summary_table_path)
    _write_summary_report(
        model_df=model_df,
        results=results,
        coefficients=coefficients,
        crime_forecasts=crime_forecasts,
        temporal_holdout=temporal_holdout_results,
        influence_summary=influence_summary,
        cluster_stability=cluster_stability,
        settings=settings,
        output_path=report_path,
    )

    return {
        "coefficients": str(coefficients_path),
        "metrics": str(metrics_path),
        "sample_sizes": str(sample_sizes_path),
        "residuals": str(residuals_path),
        "residual_review": str(residual_review_path),
        "vif": str(vif_path),
        "vif_notes": str(vif_notes_path),
        "scatter_plot": str(scatter_path),
        "geography_plot": str(geography_path),
        "cluster_assignments": str(cluster_assignments_path),
        "cluster_profiles": str(cluster_profiles_path),
        "cluster_stability_diagnostics": str(cluster_stability_path),
        "spatial_diagnostics": str(spatial_diagnostics_path),
        "spatial_hotspots": str(spatial_hotspots_path),
        "validation_metrics": str(validation_metrics_path),
        "validation_notes": str(validation_notes_path),
        "zip_comparison": str(zip_comparison_path),
        "model_summary_table": str(model_summary_table_path),
        "trend_decomposition": str(trend_decomposition_path),
        "trend_notes": str(trend_notes_path),
        "feature_selection_metrics": str(feature_selection_metrics_path),
        "feature_selection_notes": str(feature_selection_notes_path),
        "feature_power_retention_metrics": str(feature_power_retention_metrics_path),
        "feature_power_retention_notes": str(feature_power_retention_notes_path),
        "predictive_model_metrics": str(predictive_model_metrics_path),
        "predictive_model_predictions": str(predictive_model_predictions_path),
        "model_selection_notes": str(model_selection_notes_path),
        "forecast_model_metrics": str(forecast_model_metrics_path),
        "crime_forecasts": str(crime_forecasts_path),
        "forecast_confidence_intervals": str(forecast_intervals_path),
        "temporal_holdout_results": str(temporal_holdout_results_path),
        "forecast_interval_calibration": str(forecast_interval_calibration_path),
        "forecast_notes": str(forecast_notes_path),
        "scenario_impacts": str(scenario_impacts_path),
        "scenario_notes": str(scenario_notes_path),
        "zip_benchmarks": str(zip_benchmarks_path),
        "benchmark_summary": str(benchmark_summary_path),
        "drift_diagnostics": str(drift_diagnostics_path),
        "drift_notes": str(drift_notes_path),
        "influence_robustness_diagnostics": str(influence_robustness_path),
        "statistical_guardrails": str(statistical_guardrails_path),
        "comprehensive_validation_metrics": str(comprehensive_validation_metrics_path),
        "comprehensive_validation_notes": str(comprehensive_validation_notes_path),
        "policy_recommendations": str(policy_recommendations_path),
        "policy_recommendations_notes": str(policy_recommendations_notes_path),
        "policy_guardrails": str(policy_guardrails_path),
        "summary": str(report_path),
    }


def _write_scatter_plot(model_df: pd.DataFrame, output_path: Path) -> None:
    frame = model_df.copy()
    frame["home_value"] = pd.to_numeric(frame["home_value"], errors="coerce")
    frame["total_rate_per_1000"] = pd.to_numeric(frame["total_rate_per_1000"], errors="coerce")
    frame = frame.dropna(subset=["home_value", "total_rate_per_1000"])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(frame["total_rate_per_1000"], frame["home_value"], alpha=0.8, color="#0b5cab")
    ax.set_title("Dallas ZIP Home Value vs Total Crime Rate")
    ax.set_xlabel("Total Crime Rate per 1,000 Residents")
    ax.set_ylabel("Home Value")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _write_geography_plot(model_df: pd.DataFrame, output_path: Path) -> None:
    required = {"centroid_latitude", "centroid_longitude", "total_rate_per_1000", "home_value"}
    if not required <= set(model_df.columns):
        fig, ax = plt.subplots(figsize=(8.5, 6))
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "Geography view unavailable\nmissing ZIP centroid coordinates in model dataset",
            ha="center",
            va="center",
        )
        fig.tight_layout()
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
        return

    frame = model_df.copy()
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=list(required))
    if frame.empty:
        fig, ax = plt.subplots(figsize=(8.5, 6))
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "Geography view unavailable\nno ZIP rows had complete coordinates and metrics",
            ha="center",
            va="center",
        )
        fig.tight_layout()
        fig.savefig(output_path, dpi=200)
        plt.close(fig)
        return

    home_values = frame["home_value"].clip(lower=1)
    size_scale = np.sqrt(home_values / home_values.median()) * 80

    fig, ax = plt.subplots(figsize=(8.5, 6))
    scatter = ax.scatter(
        frame["centroid_longitude"],
        frame["centroid_latitude"],
        c=frame["total_rate_per_1000"],
        s=size_scale,
        cmap="YlOrRd",
        alpha=0.8,
        edgecolors="#1f2933",
        linewidths=0.4,
    )
    ax.set_title("Dallas ZIP Crime and Housing Geography")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(alpha=0.2)
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Total Crime Rate per 1,000")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _standardize_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    standardized = frame[columns].apply(pd.to_numeric, errors="coerce")
    std = standardized.std(ddof=0).replace(0, 1.0).fillna(1.0)
    return (standardized - standardized.mean()) / std


def _prepare_segmentation_working_frame(
    model_df: pd.DataFrame,
    *,
    features: list[str],
    preprocessing_mode: str,
) -> pd.DataFrame:
    working = model_df[["zip", *features]].copy()
    for feature in features:
        working[feature] = pd.to_numeric(working[feature], errors="coerce")
    working = working.dropna(subset=features).reset_index(drop=True)
    if working.empty or preprocessing_mode == "raw":
        return working

    quantile_map = {
        "winsor_5_95": (0.05, 0.95),
        "winsor_10_90": (0.10, 0.90),
    }
    if preprocessing_mode not in quantile_map:
        raise ValueError(f"unsupported segmentation preprocessing mode: {preprocessing_mode}")
    lower_quantile, upper_quantile = quantile_map[preprocessing_mode]
    for feature in features:
        lower_bound = working[feature].quantile(lower_quantile)
        upper_bound = working[feature].quantile(upper_quantile)
        working[feature] = working[feature].clip(lower=lower_bound, upper=upper_bound)
    return working


def _iter_segmentation_feature_sets(available_features: list[str]) -> list[list[str]]:
    feature_sets: list[list[str]] = []
    minimum_size = 2 if len(available_features) >= 2 else len(available_features)
    for size in range(len(available_features), minimum_size - 1, -1):
        for subset in itertools.combinations(available_features, size):
            feature_sets.append(list(subset))
    return feature_sets


def _deterministic_kmeans(values: np.ndarray, *, k: int, max_iter: int = 30) -> tuple[np.ndarray, np.ndarray]:
    row_order = np.argsort(values.sum(axis=1), kind="mergesort")
    anchors = np.linspace(0, len(row_order) - 1, num=k, dtype=int)
    centroids = values[row_order[anchors]].copy()
    labels = np.zeros(len(values), dtype=int)

    for _ in range(max_iter):
        distances = np.linalg.norm(values[:, None, :] - centroids[None, :, :], axis=2)
        next_labels = distances.argmin(axis=1)
        if np.array_equal(labels, next_labels):
            break
        labels = next_labels
        for cluster_id in range(k):
            mask = labels == cluster_id
            if mask.any():
                centroids[cluster_id] = values[mask].mean(axis=0)

    return labels, centroids


def _silhouette_score(values: np.ndarray, labels: np.ndarray) -> float | None:
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2 or len(values) <= len(unique_labels):
        return None

    distances = np.linalg.norm(values[:, None, :] - values[None, :, :], axis=2)
    scores: list[float] = []
    for index, label in enumerate(labels):
        same_cluster = labels == label
        same_cluster[index] = False
        intra_cluster = float(distances[index, same_cluster].mean()) if same_cluster.any() else 0.0
        nearest_cluster = min(
            float(distances[index, labels == other_label].mean())
            for other_label in unique_labels
            if other_label != label
        )
        denominator = max(intra_cluster, nearest_cluster)
        if denominator == 0:
            scores.append(0.0)
        else:
            scores.append((nearest_cluster - intra_cluster) / denominator)
    return float(np.mean(scores))


def _cluster_name(domain: str, cluster_order: int, *, cluster_count: int) -> str:
    names = SEGMENTATION_NAMES.get(domain)
    if names:
        if cluster_count == 2 and len(names) >= 3:
            names = (names[0], names[-1])
        if cluster_order < len(names):
            return names[cluster_order]
    return f"{domain}_{cluster_order + 1}"


def _fit_segmentation_labels(
    working: pd.DataFrame,
    *,
    domain: str,
    features: list[str],
    k: int,
) -> tuple[np.ndarray, float | None]:
    standardized = _standardize_frame(working, features)
    values = standardized.to_numpy(dtype=float)
    k = max(2, min(k, len(working)))
    labels, centroids = _deterministic_kmeans(values, k=k)
    centroid_order = np.argsort(centroids.mean(axis=1), kind="mergesort")
    remap = {int(old): int(new) for new, old in enumerate(centroid_order)}
    mapped_labels = np.array([remap[int(label)] for label in labels], dtype=int)
    return mapped_labels, _silhouette_score(values, mapped_labels)


def _evaluate_segmentation_solution(
    working: pd.DataFrame,
    *,
    domain: str,
    features: list[str],
    k: int,
) -> dict[str, object]:
    labels, silhouette = _fit_segmentation_labels(
        working,
        domain=domain,
        features=features,
        k=k,
    )
    ari_values: list[float] = []
    if len(features) > 1:
        for feature_to_drop in features:
            reduced_features = [feature for feature in features if feature != feature_to_drop]
            reduced_labels, _ = _fit_segmentation_labels(
                working[["zip", *reduced_features]].copy(),
                domain=domain,
                features=reduced_features,
                k=k,
            )
            ari = _adjusted_rand_index(labels, reduced_labels)
            if ari is not None:
                ari_values.append(float(ari))

    cluster_sizes = pd.Series(labels).value_counts().sort_index()
    min_cluster_size = int(cluster_sizes.min()) if not cluster_sizes.empty else 0
    min_cluster_size_threshold = max(CLUSTER_MIN_SIZE_THRESHOLD, int(np.ceil(len(working) * 0.10)))
    max_cluster_share = float(cluster_sizes.max() / len(working)) if len(working) > 0 else np.nan
    mean_ari = float(np.mean(ari_values)) if ari_values else np.nan
    silhouette_pass = pd.notna(silhouette) and silhouette >= CLUSTER_PRACTICAL_SILHOUETTE_THRESHOLD
    stability_pass = pd.notna(mean_ari) and mean_ari >= CLUSTER_STABILITY_ARI_THRESHOLD
    size_pass = min_cluster_size >= min_cluster_size_threshold
    return {
        "labels": labels,
        "silhouette": silhouette,
        "cluster_sizes": cluster_sizes,
        "cluster_count": int(cluster_sizes.shape[0]),
        "min_cluster_size": min_cluster_size,
        "min_cluster_size_threshold": min_cluster_size_threshold,
        "max_cluster_share": max_cluster_share,
        "leave_one_feature_out_mean_ari": mean_ari,
        "silhouette_pass": int(silhouette_pass),
        "stability_pass": int(stability_pass),
        "size_pass": int(size_pass),
        "practical_utility_pass": int(silhouette_pass and stability_pass and size_pass),
        "selected_k": int(k),
    }


def _select_segmentation_solution(
    model_df: pd.DataFrame,
    *,
    domain: str,
    available_features: list[str],
) -> dict[str, object]:
    candidate_solutions: list[dict[str, object]] = []
    mode_priority = {
        mode: len(SEGMENTATION_PREPROCESSING_MODES) - index
        for index, mode in enumerate(SEGMENTATION_PREPROCESSING_MODES)
    }
    for features in _iter_segmentation_feature_sets(available_features):
        for preprocessing_mode in SEGMENTATION_PREPROCESSING_MODES:
            working = _prepare_segmentation_working_frame(
                model_df,
                features=features,
                preprocessing_mode=preprocessing_mode,
            )
            for k in (2, 3):
                if len(working) < k:
                    continue
                solution = _evaluate_segmentation_solution(
                    working,
                    domain=domain,
                    features=features,
                    k=k,
                )
                solution["working"] = working.copy()
                solution["selected_features"] = tuple(features)
                solution["preprocessing_mode"] = preprocessing_mode
                solution["feature_count"] = len(features)
                solution["mode_priority"] = mode_priority.get(preprocessing_mode, 0)
                candidate_solutions.append(solution)
    if not candidate_solutions:
        return {}
    candidate_solutions.sort(
        key=lambda solution: (
            int(solution["practical_utility_pass"]),
            int(solution["size_pass"]),
            int(solution["stability_pass"]),
            int(solution["silhouette_pass"]),
            int(solution["min_cluster_size"]),
            -float(solution["max_cluster_share"]),
            float(solution["leave_one_feature_out_mean_ari"])
            if pd.notna(solution["leave_one_feature_out_mean_ari"])
            else float("-inf"),
            float(solution["silhouette"])
            if pd.notna(solution["silhouette"])
            else float("-inf"),
            int(solution["feature_count"]),
            int(solution["mode_priority"]),
            -int(solution["selected_k"]),
        ),
        reverse=True,
    )
    return candidate_solutions[0]


def _comb2(value: int) -> float:
    return float(value * (value - 1) / 2)


def _adjusted_rand_index(labels_a: np.ndarray, labels_b: np.ndarray) -> float | None:
    if len(labels_a) != len(labels_b) or len(labels_a) < 2:
        return None
    contingency = pd.crosstab(pd.Series(labels_a), pd.Series(labels_b))
    total = int(contingency.to_numpy().sum())
    total_pairs = _comb2(total)
    if total_pairs == 0:
        return None
    index = float(sum(_comb2(int(value)) for value in contingency.to_numpy().ravel()))
    row_pairs = float(sum(_comb2(int(value)) for value in contingency.sum(axis=1)))
    col_pairs = float(sum(_comb2(int(value)) for value in contingency.sum(axis=0)))
    expected = (row_pairs * col_pairs) / total_pairs if total_pairs > 0 else 0.0
    max_index = 0.5 * (row_pairs + col_pairs)
    denominator = max_index - expected
    if denominator == 0:
        return 1.0
    return float((index - expected) / denominator)


def _build_segmentation_artifacts(model_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    assignments = model_df[["zip"]].copy() if "zip" in model_df.columns else pd.DataFrame()
    profile_rows: list[dict[str, object]] = []

    for domain, feature_group in SEGMENTATION_FEATURE_GROUPS.items():
        available = [column for column in feature_group if column in model_df.columns]
        cluster_column = f"{domain}_cluster"
        if cluster_column not in assignments.columns:
            assignments[cluster_column] = pd.Series(pd.NA, index=assignments.index, dtype="string")
        if not available:
            continue

        solution = _select_segmentation_solution(
            model_df,
            domain=domain,
            available_features=available,
        )
        if not solution:
            continue
        working = solution["working"]
        if len(working) < 3:
            continue
        mapped_labels = np.asarray(solution["labels"], dtype=int)
        silhouette = solution["silhouette"]
        cluster_count = int(solution["cluster_count"])
        selected_features = list(solution["selected_features"])

        domain_assignments = working[["zip"]].copy()
        domain_assignments[cluster_column] = pd.Series(
            [
                _cluster_name(domain, int(label), cluster_count=cluster_count)
                for label in mapped_labels
            ],
            dtype="string",
        )
        assignments = assignments.drop(columns=[cluster_column]).merge(
            domain_assignments,
            on="zip",
            how="left",
        )

        for cluster_id in sorted(np.unique(mapped_labels)):
            cluster_name = _cluster_name(domain, int(cluster_id), cluster_count=cluster_count)
            cluster_frame = working.loc[mapped_labels == cluster_id].copy()
            for feature in selected_features:
                profile_rows.append(
                    {
                        "domain": domain,
                        "cluster_label": cluster_name,
                        "zip_count": int(len(cluster_frame)),
                        "silhouette_score": silhouette,
                        "feature": feature,
                        "mean_value": float(cluster_frame[feature].mean()),
                    }
                )

    if assignments.empty:
        assignments = pd.DataFrame(
            columns=["zip", "crime_cluster", "socioeconomic_cluster", "market_cluster"]
        )
    assignments = assignments.sort_values("zip", ignore_index=True)

    profiles = pd.DataFrame.from_records(
        profile_rows,
        columns=["domain", "cluster_label", "zip_count", "silhouette_score", "feature", "mean_value"],
    )
    if not profiles.empty:
        profiles = profiles.sort_values(
            ["domain", "cluster_label", "feature"],
            ignore_index=True,
        )
    return assignments, profiles


def _build_cluster_stability_artifacts(model_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "domain",
        "zip_count",
        "feature_count",
        "selected_feature_set",
        "preprocessing_mode",
        "selected_k",
        "cluster_count",
        "min_cluster_size",
        "min_cluster_size_threshold",
        "max_cluster_share",
        "silhouette_score",
        "leave_one_feature_out_mean_ari",
        "silhouette_pass",
        "stability_pass",
        "size_pass",
        "practical_utility_pass",
    ]
    rows: list[dict[str, object]] = []

    for domain, feature_group in SEGMENTATION_FEATURE_GROUPS.items():
        available = [column for column in feature_group if column in model_df.columns]
        if not available:
            continue

        solution = _select_segmentation_solution(
            model_df,
            domain=domain,
            available_features=available,
        )
        if not solution:
            continue
        working = solution["working"]
        if len(working) < 3:
            continue
        rows.append(
            {
                "domain": domain,
                "zip_count": int(len(working)),
                "feature_count": int(solution["feature_count"]),
                "selected_feature_set": ", ".join(solution["selected_features"]),
                "preprocessing_mode": str(solution["preprocessing_mode"]),
                "selected_k": int(solution["selected_k"]),
                "cluster_count": int(solution["cluster_count"]),
                "min_cluster_size": int(solution["min_cluster_size"]),
                "min_cluster_size_threshold": int(solution["min_cluster_size_threshold"]),
                "max_cluster_share": float(solution["max_cluster_share"]),
                "silhouette_score": solution["silhouette"],
                "leave_one_feature_out_mean_ari": solution["leave_one_feature_out_mean_ari"],
                "silhouette_pass": int(solution["silhouette_pass"]),
                "stability_pass": int(solution["stability_pass"]),
                "size_pass": int(solution["size_pass"]),
                "practical_utility_pass": int(solution["practical_utility_pass"]),
            }
        )

    diagnostics = pd.DataFrame.from_records(rows, columns=columns)
    if not diagnostics.empty:
        diagnostics = diagnostics.sort_values("domain", kind="mergesort", ignore_index=True)
    return diagnostics


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


def _build_validation_artifacts(results: list[RegressionResult]) -> tuple[pd.DataFrame, list[str]]:
    rows: list[dict[str, object]] = []
    notes: list[str] = []

    for result in results:
        fitted = smf.ols(formula=result.formula, data=result.model_frame).fit()
        influence = fitted.get_influence()
        hat_diag = influence.hat_matrix_diag
        residuals = fitted.resid.to_numpy(dtype=float)
        response = result.model_frame[result.dependent_variable].to_numpy(dtype=float)
        mae = float(np.mean(np.abs(residuals)))
        rmse = float(np.sqrt(np.mean(np.square(residuals))))
        press_residuals = residuals / np.clip(1.0 - hat_diag, 1e-6, None)
        loocv_mae = float(np.mean(np.abs(press_residuals)))
        loocv_rmse = float(np.sqrt(np.mean(np.square(press_residuals))))
        total_sum_squares = float(np.square(response - response.mean()).sum())
        predicted_r_squared = (
            1.0 - float(np.square(press_residuals).sum()) / total_sum_squares
            if total_sum_squares > 0
            else np.nan
        )
        cooks_distance = influence.cooks_distance[0]
        rows.append(
            {
                "model_label": result.model_label,
                "mae": mae,
                "rmse": rmse,
                "loocv_mae": loocv_mae,
                "loocv_rmse": loocv_rmse,
                "predicted_r_squared": predicted_r_squared,
                "mean_leverage": float(np.mean(hat_diag)),
                "max_leverage": float(np.max(hat_diag)),
                "max_cooks_distance": float(np.max(cooks_distance)),
            }
        )
        if float(np.max(hat_diag)) > 0.5:
            notes.append(
                f"- `{result.model_label}`: at least one ZIP has leverage above 0.50; "
                "review `model_residuals.csv` alongside validation metrics."
            )

    validation = pd.DataFrame.from_records(
        rows,
        columns=[
            "model_label",
            "mae",
            "rmse",
            "loocv_mae",
            "loocv_rmse",
            "predicted_r_squared",
            "mean_leverage",
            "max_leverage",
            "max_cooks_distance",
        ],
    )
    if not validation.empty:
        validation = validation.sort_values("model_label", ignore_index=True)
    return validation, notes


def _build_influence_robustness_artifacts(
    results: list[RegressionResult],
) -> tuple[pd.DataFrame, list[str]]:
    columns = [
        "model_label",
        "removed_zip",
        "flag_reason",
        "leverage",
        "leverage_threshold",
        "cooks_distance",
        "cooks_threshold",
        "r_squared_after_exclusion",
        "r_squared_change",
        "violent_effect_pct_change",
        "property_effect_pct_change",
        "mean_home_value_pct_delta",
        "p90_home_value_pct_delta",
        "max_home_value_pct_delta",
        "crime_term_sign_flip",
        "fit_stability_pass",
        "fit_deterioration_pass",
        "fit_improvement_warning",
        "robustness_pass",
    ]
    rows: list[dict[str, object]] = []
    notes: list[str] = []

    for result in results:
        if "zip" not in result.model_frame.columns:
            notes.append(f"- `{result.model_label}`: ZIP column was unavailable for influence checks.")
            continue

        fitted = smf.ols(formula=result.formula, data=result.model_frame).fit()
        robust_fitted = smf.ols(formula=result.formula, data=result.model_frame).fit(cov_type="HC3")
        influence = fitted.get_influence()
        leverage = influence.hat_matrix_diag
        cooks_distance = influence.cooks_distance[0]
        leverage_threshold = max((2.0 * len(fitted.params)) / max(len(result.model_frame), 1), 0.20)
        cooks_threshold = 4.0 / max(len(result.model_frame), 1)
        influence_frame = pd.DataFrame(
            {
                "zip": pd.Series(result.model_frame["zip"], dtype="string"),
                "leverage": leverage,
                "cooks_distance": cooks_distance,
            }
        )
        flagged = influence_frame.loc[
            (influence_frame["leverage"] >= leverage_threshold)
            | (influence_frame["cooks_distance"] >= cooks_threshold)
        ].copy()
        if flagged.empty and not influence_frame.empty:
            flagged = influence_frame.sort_values(
                ["leverage", "cooks_distance"],
                ascending=[False, False],
                kind="mergesort",
            ).head(1)
            notes.append(
                f"- `{result.model_label}`: no ZIP exceeded the leverage/Cook's thresholds, so the highest-leverage ZIP was still tested."
            )

        original_terms = {
            term: float(result.coefficients.loc[result.coefficients["term"] == term, "estimate"].iloc[0])
            if (result.coefficients["term"] == term).any()
            else np.nan
            for term in DEFAULT_PREDICTORS
        }
        for flagged_row in flagged.itertuples(index=False):
            reduced_frame = result.model_frame.loc[
                pd.Series(result.model_frame["zip"], dtype="string") != str(flagged_row.zip)
            ].copy()
            try:
                refit = smf.ols(formula=result.formula, data=reduced_frame).fit(cov_type="HC3")
            except (TypeError, ValueError, np.linalg.LinAlgError) as exc:
                notes.append(
                    f"- `{result.model_label}`: failed to refit after excluding ZIP {flagged_row.zip} ({exc})."
                )
                continue

            def _term_change_pct(term: str) -> float:
                original = original_terms.get(term, np.nan)
                updated = float(refit.params.get(term, np.nan))
                if pd.isna(original) or pd.isna(updated):
                    return np.nan
                if original == 0:
                    return abs(updated - original) * 100.0
                return abs(((updated - original) / original) * 100.0)

            violent_change = _term_change_pct("violent_rate_per_1000")
            property_change = _term_change_pct("property_rate_per_1000")
            original_predictions = np.expm1(robust_fitted.predict(reduced_frame)) + 1.0
            refit_predictions = np.expm1(refit.predict(reduced_frame)) + 1.0
            prediction_pct_delta = np.abs(
                (refit_predictions - original_predictions)
                / np.clip(original_predictions, 1e-6, None)
                * 100.0
            )
            mean_prediction_delta = float(np.mean(prediction_pct_delta))
            p90_prediction_delta = float(np.percentile(prediction_pct_delta, 90))
            max_prediction_delta = float(np.max(prediction_pct_delta))
            sign_flip = False
            for term in DEFAULT_PREDICTORS:
                original = original_terms.get(term, np.nan)
                updated = float(refit.params.get(term, np.nan))
                if pd.notna(original) and pd.notna(updated) and np.sign(original) != np.sign(updated):
                    sign_flip = True
            r_squared_change = float(refit.rsquared - result.r_squared)
            fit_stability_pass = int(abs(r_squared_change) <= 0.05)
            fit_deterioration_pass = int(r_squared_change <= 0.05)
            fit_improvement_warning = int(r_squared_change > 0.05)
            rows.append(
                {
                    "model_label": result.model_label,
                    "removed_zip": str(flagged_row.zip),
                    "flag_reason": "+".join(
                        [
                            reason
                            for reason, passed in (
                                ("high_leverage", flagged_row.leverage >= leverage_threshold),
                                ("high_cooks_distance", flagged_row.cooks_distance >= cooks_threshold),
                            )
                            if passed
                        ]
                    )
                    or "highest_leverage",
                    "leverage": float(flagged_row.leverage),
                    "leverage_threshold": leverage_threshold,
                    "cooks_distance": float(flagged_row.cooks_distance),
                    "cooks_threshold": cooks_threshold,
                    "r_squared_after_exclusion": float(refit.rsquared),
                    "r_squared_change": r_squared_change,
                    "violent_effect_pct_change": violent_change,
                    "property_effect_pct_change": property_change,
                    "mean_home_value_pct_delta": mean_prediction_delta,
                    "p90_home_value_pct_delta": p90_prediction_delta,
                    "max_home_value_pct_delta": max_prediction_delta,
                    "crime_term_sign_flip": int(sign_flip),
                    "fit_stability_pass": fit_stability_pass,
                    "fit_deterioration_pass": fit_deterioration_pass,
                    "fit_improvement_warning": fit_improvement_warning,
                    "robustness_pass": int(
                        (not sign_flip)
                        and pd.notna(p90_prediction_delta)
                        and p90_prediction_delta <= INFLUENCE_P90_HOME_VALUE_DELTA_THRESHOLD
                    ),
                }
            )

        model_rows = [row for row in rows if row["model_label"] == result.model_label]
        if model_rows:
            fit_warning_count = sum(1 for row in model_rows if int(row["fit_improvement_warning"]) == 1)
            if fit_warning_count > 0:
                notes.append(
                    f"- `{result.model_label}`: {fit_warning_count} flagged ZIP leave-out refits exceeded "
                    "the 0.05 R-squared fit-improvement warning threshold even though prediction stability is tracked separately."
                )

    diagnostics = pd.DataFrame.from_records(rows, columns=columns)
    if not diagnostics.empty:
        diagnostics = diagnostics.sort_values(
            ["model_label", "leverage", "cooks_distance"],
            ascending=[True, False, False],
            kind="mergesort",
            ignore_index=True,
        )
    return diagnostics, notes


def _summarize_influence_robustness(diagnostics: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "model_label",
        "influence_flag_count",
        "max_violent_effect_pct_change",
        "max_property_effect_pct_change",
        "max_p90_home_value_pct_delta",
        "max_home_value_pct_delta",
        "fit_stability_pass",
        "fit_deterioration_pass",
        "fit_improvement_warning_count",
        "any_crime_term_sign_flip",
        "influence_robustness_pass",
    ]
    if diagnostics.empty:
        return pd.DataFrame(columns=columns)

    summary = (
        diagnostics.groupby("model_label", as_index=False)
        .agg(
            influence_flag_count=("removed_zip", "nunique"),
            max_violent_effect_pct_change=("violent_effect_pct_change", "max"),
            max_property_effect_pct_change=("property_effect_pct_change", "max"),
            max_p90_home_value_pct_delta=("p90_home_value_pct_delta", "max"),
            max_home_value_pct_delta=("max_home_value_pct_delta", "max"),
            fit_stability_pass=("fit_stability_pass", "min"),
            fit_deterioration_pass=("fit_deterioration_pass", "min"),
            fit_improvement_warning_count=("fit_improvement_warning", "sum"),
            any_crime_term_sign_flip=("crime_term_sign_flip", "max"),
            influence_robustness_pass=("robustness_pass", "min"),
        )
        .reset_index(drop=True)
    )
    return summary[columns]


def _annotate_residuals_with_influence_flags(
    residuals: pd.DataFrame,
    influence_robustness: pd.DataFrame,
) -> pd.DataFrame:
    frame = residuals.copy()
    for column in (
        "is_high_influence_flagged",
        "influence_flag_reason",
        "influence_leverage",
        "influence_cooks_distance",
    ):
        if column in frame.columns:
            frame = frame.drop(columns=[column])
    if frame.empty or influence_robustness.empty:
        frame["is_high_influence_flagged"] = 0
        frame["influence_flag_reason"] = pd.Series(pd.NA, index=frame.index, dtype="string")
        frame["influence_leverage"] = np.nan
        frame["influence_cooks_distance"] = np.nan
        return frame
    if "zip" not in frame.columns:
        frame["is_high_influence_flagged"] = 0
        frame["influence_flag_reason"] = pd.Series(pd.NA, index=frame.index, dtype="string")
        frame["influence_leverage"] = np.nan
        frame["influence_cooks_distance"] = np.nan
        return frame

    flagged = influence_robustness.copy()
    flagged["model_label"] = flagged["model_label"].astype("string")
    flagged["removed_zip"] = flagged["removed_zip"].astype("string")
    flagged = (
        flagged.sort_values(
            ["model_label", "leverage", "cooks_distance"],
            ascending=[True, False, False],
            kind="mergesort",
        )
        .drop_duplicates(subset=["model_label", "removed_zip"], keep="first")
        .rename(
            columns={
                "removed_zip": "zip",
                "flag_reason": "influence_flag_reason",
                "leverage": "influence_leverage",
                "cooks_distance": "influence_cooks_distance",
            }
        )
    )
    frame["model_label"] = frame["model_label"].astype("string")
    frame["zip"] = frame["zip"].astype("string")
    merged = frame.merge(
        flagged[
            [
                "model_label",
                "zip",
                "influence_flag_reason",
                "influence_leverage",
                "influence_cooks_distance",
            ]
        ],
        on=["model_label", "zip"],
        how="left",
    )
    merged["is_high_influence_flagged"] = (
        merged["influence_flag_reason"].notna().astype(int)
    )
    return merged


def _apply_regression_guardrails(coefficients: pd.DataFrame) -> pd.DataFrame:
    if coefficients.empty:
        return coefficients.copy()

    guarded = coefficients.copy()
    non_intercept = guarded["term"].astype(str) != "Intercept"
    guarded["fdr_q_value"] = np.nan
    guarded.loc[non_intercept, "fdr_q_value"] = _bh_adjust_series(
        guarded.loc[non_intercept, "p_value"]
    )
    guarded["practical_effect_value"] = np.where(
        non_intercept,
        np.abs(np.expm1(pd.to_numeric(guarded["estimate"], errors="coerce")) * 100.0),
        np.nan,
    )
    guarded["practical_effect_threshold"] = REGRESSION_PRACTICAL_EFFECT_THRESHOLD_PCT
    guarded["passes_fdr_10"] = (
        guarded["fdr_q_value"].notna() & (guarded["fdr_q_value"] <= FDR_ALPHA)
    ).astype(int)
    guarded["passes_practical_effect"] = (
        pd.to_numeric(guarded["practical_effect_value"], errors="coerce").fillna(0.0)
        >= REGRESSION_PRACTICAL_EFFECT_THRESHOLD_PCT
    ).astype(int)
    guarded["interpretation_allowed"] = (
        non_intercept
        & (guarded["passes_fdr_10"] == 1)
        & (guarded["passes_practical_effect"] == 1)
    ).astype(int)
    return guarded


def _build_statistical_guardrails_artifacts(
    coefficients: pd.DataFrame,
    feature_selection_metrics: pd.DataFrame,
    spatial_diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "domain",
        "item",
        "term",
        "raw_p_value",
        "fdr_q_value",
        "practical_effect_value",
        "practical_effect_threshold",
        "passes_fdr_10",
        "passes_practical_effect",
        "interpretation_allowed",
    ]
    rows: list[dict[str, object]] = []

    for row in coefficients.loc[coefficients["term"].astype(str) != "Intercept"].itertuples(index=False):
        rows.append(
            {
                "domain": "regression",
                "item": row.model_label,
                "term": row.term,
                "raw_p_value": float(row.p_value),
                "fdr_q_value": float(row.fdr_q_value) if pd.notna(row.fdr_q_value) else np.nan,
                "practical_effect_value": float(row.practical_effect_value)
                if pd.notna(row.practical_effect_value)
                else np.nan,
                "practical_effect_threshold": float(row.practical_effect_threshold)
                if pd.notna(row.practical_effect_threshold)
                else np.nan,
                "passes_fdr_10": int(row.passes_fdr_10),
                "passes_practical_effect": int(row.passes_practical_effect),
                "interpretation_allowed": int(row.interpretation_allowed),
            }
        )

    for row in feature_selection_metrics.itertuples(index=False):
        rows.append(
            {
                "domain": "feature_selection",
                "item": "feature_selection_metrics",
                "term": row.feature_name,
                "raw_p_value": float(row.univariate_p_value) if pd.notna(row.univariate_p_value) else np.nan,
                "fdr_q_value": float(row.fdr_q_value) if pd.notna(row.fdr_q_value) else np.nan,
                "practical_effect_value": float(row.practical_effect_value)
                if pd.notna(row.practical_effect_value)
                else np.nan,
                "practical_effect_threshold": float(row.practical_effect_threshold)
                if pd.notna(row.practical_effect_threshold)
                else np.nan,
                "passes_fdr_10": int(row.passes_fdr_10),
                "passes_practical_effect": int(row.passes_practical_effect),
                "interpretation_allowed": int(row.interpretation_allowed),
            }
        )

    for row in spatial_diagnostics.itertuples(index=False):
        rows.append(
            {
                "domain": "spatial",
                "item": "spatial_diagnostics",
                "term": row.metric,
                "raw_p_value": float(row.permutation_p_value) if pd.notna(row.permutation_p_value) else np.nan,
                "fdr_q_value": float(row.fdr_q_value) if pd.notna(row.fdr_q_value) else np.nan,
                "practical_effect_value": float(row.practical_effect_value)
                if pd.notna(row.practical_effect_value)
                else np.nan,
                "practical_effect_threshold": float(row.practical_effect_threshold)
                if pd.notna(row.practical_effect_threshold)
                else np.nan,
                "passes_fdr_10": int(row.passes_fdr_10),
                "passes_practical_effect": int(row.passes_practical_effect),
                "interpretation_allowed": int(row.interpretation_allowed),
            }
        )

    guardrails = pd.DataFrame.from_records(rows, columns=columns)
    if not guardrails.empty:
        guardrails = guardrails.sort_values(
            ["domain", "item", "term"],
            kind="mergesort",
            ignore_index=True,
        )
    return guardrails


def _build_vif_artifacts(results: list[RegressionResult]) -> tuple[pd.DataFrame, list[str]]:
    vif_rows: list[pd.DataFrame] = []
    notes: list[str] = []

    for result in results:
        columns = [*result.predictors, *result.controls]
        frame = _coerce_model_columns(result.model_frame, columns).dropna(subset=columns)
        if len(frame) <= len(columns):
            notes.append(
                f"- `{result.model_label}`: skipped VIF because sample size "
                f"({len(frame)}) is too small for {len(columns)} regressors."
            )
            continue

        usable_columns = [column for column in columns if frame[column].nunique(dropna=True) > 1]
        dropped_columns = sorted(set(columns) - set(usable_columns))
        if len(usable_columns) < 2:
            notes.append(
                f"- `{result.model_label}`: skipped VIF because fewer than two regressors had usable variance."
            )
            continue

        design = sm.add_constant(frame[usable_columns], has_constant="add")
        model_rows = []
        try:
            for index, column in enumerate(design.columns):
                if column == "const":
                    continue
                model_rows.append(
                    {
                        "model_label": result.model_label,
                        "term": column,
                        "vif": float(variance_inflation_factor(design.values, index)),
                    }
                )
        except (TypeError, ValueError, np.linalg.LinAlgError) as error:
            notes.append(
                f"- `{result.model_label}`: skipped VIF due numerical instability ({error})."
            )
            continue

        infinite_terms = [row["term"] for row in model_rows if not np.isfinite(row["vif"])]
        if infinite_terms:
            notes.append(
                f"- `{result.model_label}`: infinite VIF for "
                f"{', '.join(sorted(infinite_terms))}, indicating exact collinearity."
            )
        vif_rows.append(pd.DataFrame(model_rows))
        if dropped_columns:
            notes.append(
                f"- `{result.model_label}`: dropped zero-variance terms before VIF: "
                f"{', '.join(dropped_columns)}."
            )

    if not vif_rows:
        empty = pd.DataFrame(columns=["model_label", "term", "vif"])
        return empty, notes
    combined = pd.concat(vif_rows, ignore_index=True).sort_values(
        ["model_label", "vif"],
        ascending=[True, False],
    )
    return combined, notes


def _write_vif_notes(notes: list[str], output_path: Path) -> None:
    lines = [
        "# Multicollinearity Check Notes",
        "",
    ]
    if not notes:
        lines.extend(
            [
                "VIF was computed successfully for all modeled terms.",
                "",
                "See `model_vif.csv` for model-by-model values.",
            ]
        )
    else:
        lines.extend(
            [
                "VIF checks were run with the following caveats:",
                "",
                *notes,
                "",
                "See `model_vif.csv` for available model-by-model values.",
            ]
        )
    output_path.write_text("\n".join(lines) + "\n")


def _write_validation_notes(notes: list[str], output_path: Path) -> None:
    lines = [
        "# Model Validation Notes",
        "",
    ]
    if not notes:
        lines.extend(
            [
                "LOOCV-style PRESS diagnostics completed without additional caveats.",
                "",
                "See `model_validation_metrics.csv` for model-by-model values.",
            ]
        )
    else:
        lines.extend(
            [
                "Validation checks completed with the following caveats:",
                "",
                *notes,
                "",
                "See `model_validation_metrics.csv` for model-by-model values.",
            ]
        )
    output_path.write_text("\n".join(lines) + "\n")


def _write_residual_review(residuals_df: pd.DataFrame, output_path: Path) -> None:
    lines = [
        "# Residual Review",
        "",
        "Rows below show the largest absolute residuals for each model.",
    ]
    for model_label, model_frame in residuals_df.groupby("model_label"):
        lines.extend(
            [
                "",
                f"## {model_label}",
                "",
                "| zip | observed | fitted_value | residual | absolute_residual |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in model_frame.head(5).itertuples(index=False):
            zip_code = getattr(row, "zip", "n/a")
            lines.append(
                f"| {zip_code} | {row.observed:.3f} | {row.fitted_value:.3f} | "
                f"{row.residual:.3f} | {row.absolute_residual:.3f} |"
            )
        flagged = model_frame.loc[
            pd.to_numeric(model_frame.get("is_high_influence_flagged"), errors="coerce")
            .fillna(0)
            .astype(int)
            == 1
        ].copy()
        if not flagged.empty:
            lines.extend(
                [
                    "",
                    "High-influence ZIPs can have small residuals; use leverage/Cook's diagnostics alongside residual size.",
                    "",
                    "| zip | influence_flag_reason | influence_leverage | influence_cooks_distance | absolute_residual |",
                    "| --- | --- | ---: | ---: | ---: |",
                ]
            )
            flagged = flagged.sort_values(
                ["influence_leverage", "influence_cooks_distance", "absolute_residual"],
                ascending=[False, False, False],
                kind="mergesort",
            )
            for row in flagged.head(5).itertuples(index=False):
                lines.append(
                    f"| {getattr(row, 'zip', 'n/a')} | {getattr(row, 'influence_flag_reason', '')} | "
                    f"{float(getattr(row, 'influence_leverage', np.nan)):.3f} | "
                    f"{float(getattr(row, 'influence_cooks_distance', np.nan)):.3f} | "
                    f"{float(getattr(row, 'absolute_residual', np.nan)):.3f} |"
                )

    output_path.write_text("\n".join(lines) + "\n")


def _write_zip_comparison_table(model_df: pd.DataFrame, output_path: Path) -> None:
    frame = model_df.copy()
    frame["home_value"] = pd.to_numeric(frame["home_value"], errors="coerce")
    frame["total_rate_per_1000"] = pd.to_numeric(frame["total_rate_per_1000"], errors="coerce")
    frame = frame.dropna(subset=["home_value", "total_rate_per_1000"])

    top = frame.sort_values("home_value", ascending=False).head(5)
    bottom = frame.sort_values("home_value", ascending=True).head(5)

    lines = [
        "# ZIP Comparison Table",
        "",
        "## Highest Home Value ZIPs",
        "",
        "| zip | home_value | total_rate_per_1000 |",
        "| --- | ---: | ---: |",
    ]
    for row in top.itertuples(index=False):
        lines.append(f"| {row.zip} | {row.home_value:.0f} | {row.total_rate_per_1000:.2f} |")

    lines.extend(
        [
            "",
            "## Lowest Home Value ZIPs",
            "",
            "| zip | home_value | total_rate_per_1000 |",
            "| --- | ---: | ---: |",
        ]
    )
    for row in bottom.itertuples(index=False):
        lines.append(f"| {row.zip} | {row.home_value:.0f} | {row.total_rate_per_1000:.2f} |")

    output_path.write_text("\n".join(lines) + "\n")


def _write_model_summary_table(metrics_df: pd.DataFrame, output_path: Path) -> None:
    lines = [
        "# Model Summary Table",
        "",
        "| model_label | nobs | r_squared | adjusted_r_squared |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in metrics_df.itertuples(index=False):
        lines.append(
            f"| {row.model_label} | {int(row.nobs)} | {row.r_squared:.3f} | {row.adjusted_r_squared:.3f} |"
        )
    output_path.write_text("\n".join(lines) + "\n")


def _coefficient_for_term(
    coefficients: pd.DataFrame,
    *,
    model_label: str,
    term: str,
) -> pd.Series | None:
    match = coefficients[
        (coefficients["model_label"].astype(str) == model_label)
        & (coefficients["term"].astype(str) == term)
    ]
    if match.empty:
        return None
    return match.iloc[0]


def _effect_size_text(coefficient_row: pd.Series | None) -> str:
    if coefficient_row is None:
        return "not estimable from this specification"
    estimate = pd.to_numeric(pd.Series([coefficient_row.get("estimate")]), errors="coerce").iloc[0]
    if pd.isna(estimate):
        return "not estimable from this specification"
    percent_change = (np.exp(float(estimate)) - 1.0) * 100.0
    q_value = pd.to_numeric(pd.Series([coefficient_row.get("fdr_q_value")]), errors="coerce").iloc[0]
    interpretation_allowed = (
        pd.to_numeric(pd.Series([coefficient_row.get("interpretation_allowed")]), errors="coerce")
        .fillna(0)
        .astype(int)
        .iloc[0]
        == 1
    )
    if interpretation_allowed:
        return f"{percent_change:.2f}% change in expected home value (passes FDR/practical-effect guardrails)"
    q_text = f"{q_value:.3f}" if pd.notna(q_value) else "n/a"
    return (
        f"{percent_change:.2f}% change in expected home value, but this term remains exploratory "
        f"(FDR q={q_text}; below repo guardrails)"
    )


def _write_summary_report(
    model_df: pd.DataFrame,
    results: list[RegressionResult],
    coefficients: pd.DataFrame,
    crime_forecasts: pd.DataFrame,
    temporal_holdout: pd.DataFrame,
    influence_summary: pd.DataFrame,
    cluster_stability: pd.DataFrame,
    settings: "Settings",
    output_path: Path,
) -> None:
    numeric = model_df[["total_rate_per_1000", "home_value"]].apply(pd.to_numeric, errors="coerce")
    total_corr = numeric.corr().iloc[0, 1]
    baseline = next(result for result in results if result.model_label == "baseline")
    expanded = next(result for result in results if result.model_label == "expanded_controls")

    baseline_violent = _effect_size_text(
        _coefficient_for_term(
            coefficients,
            model_label="baseline",
            term="violent_rate_per_1000",
        )
    )
    baseline_property = _effect_size_text(
        _coefficient_for_term(
            coefficients,
            model_label="baseline",
            term="property_rate_per_1000",
        )
    )
    expanded_violent = _effect_size_text(
        _coefficient_for_term(
            coefficients,
            model_label="expanded_controls",
            term="violent_rate_per_1000",
        )
    )
    expanded_property = _effect_size_text(
        _coefficient_for_term(
            coefficients,
            model_label="expanded_controls",
            term="property_rate_per_1000",
        )
    )
    housing_sources = sorted(
        str(source)
        for source in model_df.get("source", pd.Series(dtype="string")).dropna().unique()
    )
    total_corr_text = f"{total_corr:.3f}" if pd.notna(total_corr) else "not estimable"
    crime_history_path = settings.processed_dir / "crime_history_panel.csv"
    history_panel_path = settings.processed_dir / "housing_history_panel.csv"
    crime_history_note = (
        "Quarterly crime history was available for decomposition, forecasting, scenarios, and drift diagnostics."
        if crime_history_path.exists()
        else "Quarterly crime history was not available for this run."
    )
    history_note = (
        "Recent context was supplemented with a 2000-2025 historical housing panel "
        "(Realtor monthly history plus FHFA ZIP5 annual HPI)."
        if history_panel_path.exists()
        else "Historical housing context was not available for this run."
    )
    selected_holdout = temporal_holdout.loc[
        temporal_holdout["evaluation_scope"].astype(str) == "selected_zip_model"
    ]
    forecast_eligible_zip_count = (
        int(pd.to_numeric(selected_holdout["eligible_zip_count"], errors="coerce").max())
        if not selected_holdout.empty
        else 0
    )
    lower_confidence_forecast_count = (
        int(
            crime_forecasts.loc[
                crime_forecasts.get("policy_eligible", pd.Series(1, index=crime_forecasts.index))
                .astype(int)
                == 0,
                "zip",
            ]
            .astype(str)
            .nunique()
        )
        if not crime_forecasts.empty
        else 0
    )
    selected_holdout_mape = (
        float(pd.to_numeric(selected_holdout["mape"], errors="coerce").iloc[0])
        if not selected_holdout.empty
        else np.nan
    )
    selected_holdout_threshold = (
        float(pd.to_numeric(selected_holdout["mape_threshold"], errors="coerce").iloc[0])
        if not selected_holdout.empty
        else np.nan
    )
    selected_holdout_pass = (
        int(pd.to_numeric(selected_holdout["mape_pass"], errors="coerce").iloc[0])
        if not selected_holdout.empty
        else 0
    )
    influence_fail_count = (
        int(
            pd.to_numeric(influence_summary["influence_robustness_pass"], errors="coerce")
            .fillna(0)
            .eq(0)
            .sum()
        )
        if not influence_summary.empty
        else 0
    )
    max_violent_shift = (
        float(pd.to_numeric(influence_summary["max_violent_effect_pct_change"], errors="coerce").max())
        if not influence_summary.empty
        else np.nan
    )
    max_property_shift = (
        float(pd.to_numeric(influence_summary["max_property_effect_pct_change"], errors="coerce").max())
        if not influence_summary.empty
        else np.nan
    )
    max_prediction_delta = (
        float(pd.to_numeric(influence_summary["max_p90_home_value_pct_delta"], errors="coerce").max())
        if not influence_summary.empty
        else np.nan
    )
    fit_warning_count = (
        int(
            pd.to_numeric(influence_summary["fit_improvement_warning_count"], errors="coerce")
            .fillna(0)
            .gt(0)
            .sum()
        )
        if not influence_summary.empty
        else 0
    )
    cluster_fail_count = (
        int(
            pd.to_numeric(cluster_stability["practical_utility_pass"], errors="coerce")
            .fillna(0)
            .eq(0)
            .sum()
        )
        if not cluster_stability.empty
        else 0
    )

    lines = [
        "# Dallas Crime and Housing Report",
        "",
        "This narrative summarizes the latest pipeline outputs in plain language for non-technical readers.",
        "",
        "## Methods",
        "",
        (
            "- Geography: ZIP-level Dallas-area records using the processed model dataset; "
            f"ZIPs with fewer than {settings.min_total_incidents_per_zip} incident(s) in the active crime window "
            "were excluded from the study universe."
        ),
        "- Inputs: Dallas OpenData crime incidents, ACS ZIP controls, and current housing price signals "
        "from Zillow, Realtor, Redfin, and Realtor bulk ZIP feeds.",
        f"- Current housing sources represented in this model run: {', '.join(housing_sources) if housing_sources else 'none recorded'}.",
        f"- Crime history context: {crime_history_note}",
        f"- Historical context: {history_note}",
        "- Strategy: two robust regression models were run from the same processed dataset:",
        "  - baseline model with core crime + household controls",
        f"  - expanded-controls model adding completeness-qualified controls: {', '.join(expanded.controls)}",
        "",
        "## Findings",
        "",
        f"- Baseline sample size: {baseline.nobs} ZIPs; expanded-controls sample size: {expanded.nobs} ZIPs.",
        f"- Overall correlation between total crime rate and home value: {total_corr_text}.",
        "- Estimated effect per +1 crime incident per 1,000 residents:",
        f"  - Baseline violent-rate term: {baseline_violent}.",
        f"  - Baseline property-rate term: {baseline_property}.",
        f"  - Expanded violent-rate term: {expanded_violent}.",
        f"  - Expanded property-rate term: {expanded_property}.",
        f"- Model fit stayed stable across specs (baseline R-squared: {baseline.r_squared:.3f}, "
        f"expanded R-squared: {expanded.r_squared:.3f}).",
        "",
        "## Limitations",
        "",
        "- This is observational ZIP-level analysis and should not be read as causal proof.",
        (
            f"- High-confidence forecast/scenario coverage is {forecast_eligible_zip_count}/{len(model_df)} modeled ZIPs; lower-confidence forecast-only tiers add {lower_confidence_forecast_count} more ZIPs."
            if len(model_df) > 0
            else "- Forecast/scenario coverage was not estimable for this run."
        ),
        (
            f"- Selected ZIP temporal holdout is MAPE={selected_holdout_mape:.3f} versus threshold {selected_holdout_threshold:.1f} (pass={selected_holdout_pass})."
            if pd.notna(selected_holdout_mape) and pd.notna(selected_holdout_threshold)
            else "- Selected ZIP temporal holdout was not estimable for this run."
        ),
        (
            f"- Influence robustness fails in {influence_fail_count}/{len(influence_summary)} regression specs on prediction-stability guardrails; max p90 home-value prediction delta={max_prediction_delta:.3f}%, max violent-term shift={max_violent_shift:.3f}%, max property-term shift={max_property_shift:.3f}%."
            if not influence_summary.empty
            and pd.notna(max_prediction_delta)
            and pd.notna(max_violent_shift)
            and pd.notna(max_property_shift)
            else "- Influence robustness severity was not estimable for this run."
        ),
        f"- Fit-improvement warnings remain in {fit_warning_count}/{len(influence_summary)} regression specs."
        if not influence_summary.empty
        else "- Fit-improvement warnings were not estimable for this run.",
        (
            f"- Cluster practical utility fails in {cluster_fail_count}/{len(cluster_stability)} segmentation domains on current data."
            if not cluster_stability.empty
            else "- Cluster practical utility was not estimable for this run."
        ),
        "- Housing measures combine multiple reputable sources with different definitions "
        "(typical value, median listing price, median sale price, HPI), so they are not perfectly interchangeable.",
        "- ACS and FHFA inputs can lag real-time market and demographic shifts.",
        "- Small sample sizes and correlated neighborhood factors can inflate uncertainty.",
        "- Results are sensitive to ZIP-universe filtering, available controls, and the current crime lookback window.",
        "",
        "## Visuals and Artifacts",
        "",
        "- `home_value_vs_total_crime.png` (scatter visual)",
        "- `crime_home_value_geography.png` (ZIP centroid geography-aware view)",
        "- `cluster_assignments.csv` and `cluster_profiles.csv` (deterministic ZIP segmentation outputs)",
        "- `spatial_diagnostics.csv` and `spatial_hotspots.csv` (Moran-style clustering and local spillover view)",
        "- `model_validation_metrics.csv` and `model_validation_notes.md` (PRESS/LOOCV-style validation diagnostics)",
        "- `feature_selection_metrics.csv` and `feature_selection_notes.md` (candidate feature ranking and coverage)",
        "- `feature_power_retention_metrics.csv` and `feature_power_retention_notes.md` (explicit retained-signal checkpoint metrics)",
        "- `predictive_model_metrics.csv`, `predictive_model_predictions.csv`, and `model_selection_notes.md` (broader model-family and ensemble selection diagnostics)",
        "- `crime_trend_decomposition.csv` and `crime_trend_decomposition.md` (quarterly decomposition view)",
        "- `forecast_model_metrics.csv`, `crime_forecasts.csv`, and `forecast_confidence_intervals.csv` (ZIP-level forecast model comparison and 12-month outlook)",
        "- `temporal_holdout_results.csv` and `forecast_interval_calibration.csv` (temporal holdout and interval credibility checks)",
        "- `scenario_impacts.csv` and `scenario_notes.md` (deterministic planning scenarios)",
        "- `zip_benchmarks.csv` and `benchmark_summary.md` (ZIP-vs-metro and ZIP-vs-cluster comparisons)",
        "- `model_drift_diagnostics.csv` and `model_drift_notes.md` (latest-quarter drift checks)",
        "- `influence_robustness_diagnostics.csv` and `cluster_stability_diagnostics.csv` (robustness and segmentation-utility checks)",
        "- `statistical_guardrails.csv` and `policy_guardrails.md` (multiple-testing, practical-effect, and non-causal interpretation guardrails)",
        "- `comprehensive_validation_metrics.csv` and `comprehensive_validation_notes.md` (cross-artifact validation rollup)",
        "- `policy_recommendations_by_segment.csv` and `policy_recommendations_by_segment.md` (segment-level action framing)",
        "- `top_bottom_zip_comparison.md` (additional ZIP comparison table)",
        "- `model_summary_table.md` (model-level metrics table)",
        "- `regression_coefficients.csv` and `regression_metrics.csv`",
        "- `model_sample_sizes.csv`, `model_residuals.csv`, `residual_review.md`",
        "- `model_vif.csv` and `model_vif_notes.md`",
        "",
        "## Rebuild Path",
        "",
        "- Full refresh from raw sources: `dallas-crime acquire && dallas-crime build && dallas-crime analyze`",
        "- Report refresh from existing raw inputs: `dallas-crime build && dallas-crime analyze`",
        "",
        "## References",
        "",
        "- Dallas OpenData Police Incidents dataset",
        "- U.S. Census ACS 5-year ZIP-level tables",
        "- Zillow, Realtor, and Redfin ZIP market pages captured via Firecrawl",
        "- Realtor ZIP market pages and Realtor ZIP inventory/history feeds",
        "- FHFA ZIP5 annual HPI",
    ]
    output_path.write_text("\n".join(lines) + "\n")
