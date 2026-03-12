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
  - `scenario_impacts.csv` / `scenario_notes.md`
  - `zip_benchmarks.csv` / `benchmark_summary.md`
  - `model_drift_diagnostics.csv` / `model_drift_notes.md`
  - `comprehensive_validation_metrics.csv` / `comprehensive_validation_notes.md`
  - `policy_recommendations_by_segment.csv` / `policy_recommendations_by_segment.md`
- Documentation alignment:
  - `task.md`
  - `AGENTS.md`
  - `docs/v2_technical_compendium.md`
  - `docs/architecture_decisions.md`
  - `docs/technical_documentation_index.md`
  - `docs/dashboard_specifications.md`
  - `RUNLOG.md`

## Remaining Open States

All repo-native roadmap rows in `task.md` are now either `completed` or explicitly externalized (`blocked` / `out_of_scope`).

- Data sufficiency posture (latest refresh):
  - repo-native temporal forecasting/sensitivity/policy workflows are data-sufficient in-package (`acquire` + `build` + `analyze` complete; forecast/scenario/policy artifacts non-empty)
  - `acquire` now materializes all five optional category sidecars (`economic`, `real_estate`, `law_enforcement`, `social_services`, `infrastructure`) and `build` merges them additively into `model_dataset.csv`
  - coverage caveat: modeled crime history depth is uneven (`8/71` modeled ZIPs have fewer than `12` quarterly periods)
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
