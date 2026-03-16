# V2 Execution Tracker

Concise translation of [`v2_implementation_roadmap.md`](/Users/imanol/Documents/Projects/Dallas-Crime-Analysis/v2_implementation_roadmap.md) into the current package surface.

Detailed roadmap task state now lives in [`task.md`](/Users/imanol/Documents/Projects/Dallas-Crime-Analysis/task.md).
Use this file as the short package summary and `task.md` as the line-by-line execution ledger.

## Implemented In Package

- `acquire`: current Dallas crime, longer-horizon crime history (`crime_history_records.csv`),
  ACS ZCTA controls (including unemployment/vacancy/education/public-assistance/transit fields),
  multi-year ACS snapshots (`acs_zcta_snapshots.csv`), multi-source housing refresh, historical
  housing collection (`housing_market_history.csv`), and generated optional sidecars for
  economic, real-estate, law-enforcement, social-services, and infrastructure categories.
- `build`: ZIP-universe filtering, QA artifacts, quarterly `crime_history_panel.csv`, expanded `crime_history_features.csv`, processed `acs_snapshot_features.csv`, processed `source_completeness_scores.csv`, optional ZIP-sidecar joins, processed `housing_history_panel.csv`, additive `housing_history_features.csv`, additive interaction/aggregate feature generation (`interaction_features.csv`), and additive merge into `model_dataset.csv`.
- `analyze`: HC3 regression pipeline, residual/VIF diagnostics, geography plot, deterministic segmentation outputs, spatial diagnostics, model validation artifacts, crime trend decomposition, feature-selection artifacts, predictive model-family comparison/selection/ensemble artifacts, 12-month forecasts with confidence intervals, 5-scenario stress outputs, ZIP benchmarks, drift diagnostics, comprehensive validation rollups, and policy recommendations by segment.

## Current Tranche

- Q2 package-completion lift that is now implemented:
  - ACS snapshot features now flow from `acs_zcta_snapshots.csv` into `acs_snapshot_features.csv` and `model_dataset.csv`
  - crime history features now include lags, seasonal context, momentum, and acceleration
  - completeness scoring now ships as `source_completeness_scores.csv` plus an overall score in `model_dataset.csv`
  - interaction and aggregate features now ship in `interaction_features.csv` and are joined additively into `model_dataset.csv`
  - optional ZIP sidecars now land additively for economic, real-estate, law-enforcement, social-services, and infrastructure categories
- Q2/Q3 analysis lift now implemented inside `analyze`:
  - `crime_trend_decomposition.csv` / `.md`
  - `feature_selection_metrics.csv` / `feature_selection_notes.md`
  - `feature_power_retention_metrics.csv` / `feature_power_retention_notes.md`
  - `predictive_model_metrics.csv` / `predictive_model_predictions.csv` / `model_selection_notes.md`
  - `forecast_model_metrics.csv`
  - `crime_forecasts.csv`
  - `forecast_confidence_intervals.csv`
  - `temporal_holdout_results.csv`
  - `forecast_interval_calibration.csv`
  - `scenario_impacts.csv` / `scenario_notes.md`
  - `zip_benchmarks.csv` / `benchmark_summary.md`
  - `model_drift_diagnostics.csv` / `model_drift_notes.md`
  - `influence_robustness_diagnostics.csv`
  - `cluster_stability_diagnostics.csv`
  - `statistical_guardrails.csv`
  - `comprehensive_validation_metrics.csv` / `comprehensive_validation_notes.md`
  - `policy_recommendations_by_segment.csv` / `policy_recommendations_by_segment.md`
  - `policy_guardrails.md`
- Documentation alignment:
  - `task.md`
  - `AGENTS.md`
  - `docs/v2_technical_compendium.md`
  - `docs/architecture_decisions.md`
  - `docs/technical_documentation_index.md`
  - `docs/dashboard_specifications.md`
  - `RUNLOG.md`

## MVM Gate Fixes (2026-03-15)

- Gate 2 (VIF): resolved by collapsing violent_rate + property_rate into total_rate_per_1000 as the sole crime predictor (DEFAULT_PREDICTORS change).
- Influence robustness columns renamed from violent/property to crime_term to match the new predictor.
- Expanded model relabeled to sensitivity_check; reporting narrative aligned.

## Remaining Open States

All roadmap implementation rows in `task.md` are now either `completed` or explicitly externalized (`blocked` / `out_of_scope`), and the repo-native statistical credibility follow-up requested by the 2026-03-12 audit is now implemented. Forecast/policy readiness still depends on how the new diagnostics score on refreshed data.

- Data sufficiency posture (statistical audit, 2026-03-12):
  - repo-native `acquire` + `build` + `analyze` workflows are reproducible and sufficient for descriptive, exploratory, and package-internal monitoring use
  - `acquire` now materializes all five optional category sidecars (`economic`, `real_estate`, `law_enforcement`, `social_services`, `infrastructure`) and `build` merges them additively into `model_dataset.csv`
  - current outputs are sufficient for high-confidence forecasting claims only within the validated `52/71` high-confidence subset, and are still not sufficient for broad-coverage sensitivity claims or policy-impact claims
  - current live evidence after the implemented credibility follow-up:
    - modeled coverage is `71/98` target ZIPs
    - high-confidence forecast/scenario coverage is `52/71` modeled ZIPs after the `>=12` trailing-quarter gate, with `19/71` additional modeled ZIPs now surfaced as lower-confidence forecast-only tiers
    - selected-model temporal holdout now clears the repo threshold (`MAPE=19.9369%`, `pass=1`)
    - interval calibration and output shape now pass on current data (`80%` and `95%` selected-model intervals both `calibration_pass=1`; `0` equal-bound and `0` extreme-upper-ratio rows)
    - prediction-stability influence checks now pass for both regression specs (`influence_robustness_pass=1`), and the one remaining baseline R-squared anomaly is now surfaced as an additive fit-improvement warning rather than a robustness failure
    - cluster practical utility now passes in all three domains after auditable preprocessing/feature-set screening (`practical_utility_pass=1` for crime, market, and socioeconomic)
    - only `0.163` of guarded regression / feature-selection / spatial tests clear the repo interpretation guardrails
- Implemented statistical credibility follow-up now emitted by the package:
  - quarter-completeness gating before drift and forecast workflows
  - holdout-pass-screened ZIP-level forecast-family selection plus additive lower-confidence forecast-only tiers for modeled ZIPs that fail the high-confidence gate
  - leave-high-leverage-ZIP-out influence robustness checks
  - residual-facing high-influence ZIP flags in `model_residuals.csv` / `residual_review.md`
  - forecast interval calibration diagnostics and output-shape pass/fail reporting
  - true temporal holdout evaluation for policy-facing claims
  - cluster stability checks, adaptive `k=2/3` selection, auditable preprocessing/feature-set screening, and practical-utility thresholds
  - segment-level policy guardrail fields for scenario support, high-influence concentration, cluster utility, and confidence tiering
  - multiple-testing correction, practical-effect thresholds, and explicit non-causal policy guardrails
- Remaining decision posture:
  - current outputs remain descriptive/exploratory until coverage and interpretation guardrails improve on refreshed data
- External program items:
  - stakeholder approvals and non-package operating cadence (weekly monitoring, monthly QA, UAT/training)

## Out Of Current Package

- PostgreSQL/TimescaleDB persistence and query serving
- five operational dashboards and frontend/backend application scaffolding
- always-on monthly scheduling, alerting, and runtime drift monitoring
- commercial-data integrations that depend on procurement or licensing
- stakeholder training, UAT process, and rollout support

## Execution Rule

Build inside the current package in this order:

1. `acquire` historical/panel inputs.
2. `build` panel-safe features and QA.
3. `analyze` forecasting, scenarios, and benchmarking on top of those inputs.
4. Only then split into dashboard or service work.
