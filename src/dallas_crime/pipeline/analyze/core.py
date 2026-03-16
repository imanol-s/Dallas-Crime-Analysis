"""Core regression, validation, influence, guardrails, and feature-selection artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor

if TYPE_CHECKING:
    from dallas_crime.config import Settings

DEFAULT_PREDICTORS = ("total_rate_per_1000",)
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
DRIFT_MIN_COMPLETENESS = 0.33
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

    # ── Completeness gate: no regression column may exceed 30% null ──
    _MAX_COLUMN_NULL_SHARE = 0.30
    total_rows = len(frame)
    if total_rows > 0:
        high_null_cols = []
        for col in required_columns:
            null_share = float(frame[col].isna().sum()) / total_rows
            if null_share > _MAX_COLUMN_NULL_SHARE:
                high_null_cols.append(f"{col} ({null_share:.0%})")
        if high_null_cols:
            import warnings

            warnings.warn(
                f"Regression '{model_label}': columns exceed 30% null before row-dropping: "
                f"{', '.join(high_null_cols)}. Coefficient estimates may be unreliable.",
                stacklevel=2,
            )

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
        "crime_term_effect_pct_change",
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

            crime_term_change = _term_change_pct("total_rate_per_1000")
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
                    "crime_term_effect_pct_change": crime_term_change,
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
        "max_crime_term_effect_pct_change",
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
            max_crime_term_effect_pct_change=("crime_term_effect_pct_change", "max"),
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
