# Methodology

## Study frame

- Unit of analysis: ZIP codes represented in the processed `model_dataset.csv`.
- Study-universe rule: ZIPs must survive the `crime_zip` and `housing_zip` intersection and must
  have at least `DCA_MIN_TOTAL_INCIDENTS_PER_ZIP` incidents in the active crime window.
- Modeled-universe rule: ZIPs must also have valid ACS controls; zero-population and sentinel ACS
  rows are excluded upstream rather than carried forward as missing model rows.
- Outcome: `log_home_value` (natural log of ZIP-level home value).
- Core exposure variables:
  - `violent_rate_per_1000`
  - `property_rate_per_1000`
- Baseline controls:
  - `median_household_income`
  - `poverty_rate`
  - `owner_occupied_share`
  - `median_gross_rent`
- Expanded controls: available additional controls from processed data
  (`population_acs`, `median_rent`, `annual_change_pct`, `realtor_active_listing_count`,
  `realtor_median_days_on_market`, `realtor_pending_ratio`,
  `realtor_hist_listing_price_12m_change`) when sample size supports estimation.

## Sources

- Crime:
  - Dallas OpenData Police Incidents active-window extract (`data/raw/crime_records.csv`)
  - Dallas OpenData Police Incidents longer-horizon panel extract
    (`data/raw/crime_history_records.csv`)
- ACS controls:
  - U.S. Census ACS 5-year ZCTA tables for the latest configured year (`data/raw/acs_zcta.csv`)
  - Multi-year ACS 5-year ZCTA snapshots (`data/raw/acs_zcta_snapshots.csv`)
  - Additional ACS-derived controls now include unemployment, vacancy, educational attainment,
    public-assistance share, and transit commute share
  - official bulk-table fallback for each requested year
- Current housing signal:
  - Zillow ZIP pages via Firecrawl search+scrape
  - Realtor ZIP market pages
  - Redfin ZIP housing-market pages
  - Realtor ZIP inventory CSV
- Historical housing context (`data/raw/housing_market_history.csv`,
  `data/processed/housing_history_panel.csv`):
  - Realtor ZIP monthly history (`2016-2025` in current source coverage)
  - FHFA ZIP5 annual HPI (`2000-2024` in current source coverage)
- Derived history feature table (`data/processed/housing_history_features.csv`):
  - ZIP-level coverage windows, latest observed history values, and simple full-period change metrics
    from Realtor monthly history and FHFA annual HPI.
- Derived crime history artifacts:
  - `data/processed/crime_history_panel.csv` aggregates the longer-horizon crime raw extract
    into quarterly ZIP-level counts and rates when that file is present, otherwise it falls back
    to the active-window incident extract
  - `data/processed/crime_history_features.csv` summarizes quarter coverage, latest period levels,
    lagged quarters, seasonal means, momentum, and acceleration terms
- Derived ACS snapshot and completeness artifacts:
  - `data/processed/acs_snapshot_features.csv` summarizes multi-year ACS controls by ZIP
    (latest levels, full-span changes, and trend slopes)
  - `data/processed/source_completeness_scores.csv` tracks per-category source coverage by ZIP
- Derived interaction artifact:
  - `data/processed/interaction_features.csv` adds additive interaction terms and aggregate stress indices,
    then left-joins them into `data/processed/model_dataset.csv`
- Optional category sidecars generated during `acquire`:
  - `data/raw/dfw_zip_economic_sidecar.csv`
  - `data/raw/dfw_zip_real_estate_sidecar.csv`
  - `data/raw/dfw_zip_law_enforcement_sidecar.csv`
  - `data/raw/dfw_zip_social_services_sidecar.csv`
  - `data/raw/dfw_zip_infrastructure_sidecar.csv`

## Model specifications

Two robust OLS specifications are estimated from the same processed dataset:

1. Baseline model (`baseline` label): crime rates + baseline controls.
2. Expanded controls model (`expanded_controls` label): baseline model plus selected additional
   controls that pass completeness checks.

Both models use HC3 robust standard errors.

## Diagnostics

Analysis exports:

- processed crime history artifacts (`crime_history_panel.csv`, `crime_history_features.csv`)
- sample size by model (`model_sample_sizes.csv`)
- geography-aware ZIP centroid view (`crime_home_value_geography.png`)
- deterministic segmentation outputs (`cluster_assignments.csv`, `cluster_profiles.csv`)
- spatial autocorrelation and hotspot views (`spatial_diagnostics.csv`, `spatial_hotspots.csv`)
- residual-level artifact (`model_residuals.csv`) and review summary (`residual_review.md`)
- multicollinearity check (`model_vif.csv`) with rationale notes (`model_vif_notes.md`) when terms
  are dropped or VIF is skipped
- PRESS/LOOCV-style validation diagnostics (`model_validation_metrics.csv`,
  `model_validation_notes.md`)
- trend decomposition outputs (`crime_trend_decomposition.csv`, `crime_trend_decomposition.md`)
- feature-selection diagnostics (`feature_selection_metrics.csv`, `feature_selection_notes.md`)
- feature-power checkpoint diagnostics (`feature_power_retention_metrics.csv`,
  `feature_power_retention_notes.md`)
- predictive model-family and selection artifacts (`predictive_model_metrics.csv`,
  `predictive_model_predictions.csv`, `model_selection_notes.md`)
- forecast and interval artifacts (`forecast_model_metrics.csv`, `crime_forecasts.csv`,
  `forecast_confidence_intervals.csv`, `forecast_notes.md`)
- stress/scenario outputs (`scenario_impacts.csv`, `scenario_notes.md`)
- benchmark outputs (`zip_benchmarks.csv`, `benchmark_summary.md`)
- drift diagnostics (`model_drift_diagnostics.csv`, `model_drift_notes.md`)
- comprehensive cross-artifact validation rollup (`comprehensive_validation_metrics.csv`,
  `comprehensive_validation_notes.md`)
- policy recommendations by segment (`policy_recommendations_by_segment.csv`,
  `policy_recommendations_by_segment.md`)

## Interpretation boundaries

- This is a ZIP-level observational model, not a causal design.
- Coefficients reflect association under included controls.
- Results depend on data completeness, source harmonization choices, and crime lookback choices.
- ACS snapshot history reflects rolling ACS 5-year estimates rather than point-in-time annual
  census counts.
- Current home-value signals are harmonized across multiple reputable but not identical measures
  (typical value, listing price, sale price, HPI).
