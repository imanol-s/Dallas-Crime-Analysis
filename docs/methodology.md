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

- Crime: Dallas OpenData Police Incidents.
- ACS controls: U.S. Census ACS 5-year ZCTA tables, with official bulk-table fallback.
- Current housing signal:
  - Zillow ZIP pages via Firecrawl search+scrape
  - Realtor ZIP market pages
  - Redfin ZIP housing-market pages
  - Realtor ZIP inventory CSV
- Historical housing context (`data/raw/housing_market_history.csv`,
  `data/processed/housing_history_panel.csv`):
  - Realtor ZIP monthly history (`2016-2025` in current source coverage)
  - FHFA ZIP5 annual HPI (`2000-2024` in current source coverage)

## Model specifications

Two robust OLS specifications are estimated from the same processed dataset:

1. Baseline model (`baseline` label): crime rates + baseline controls.
2. Expanded controls model (`expanded_controls` label): baseline model plus selected additional
   controls that pass completeness checks.

Both models use HC3 robust standard errors.

## Diagnostics

Analysis exports:

- sample size by model (`model_sample_sizes.csv`)
- geography-aware ZIP centroid view (`crime_home_value_geography.png`)
- residual-level artifact (`model_residuals.csv`) and review summary (`residual_review.md`)
- multicollinearity check (`model_vif.csv`) with rationale notes (`model_vif_notes.md`) when terms
  are dropped or VIF is skipped

## Interpretation boundaries

- This is a ZIP-level observational model, not a causal design.
- Coefficients reflect association under included controls.
- Results depend on data completeness, source harmonization choices, and crime lookback choices.
- Current home-value signals are harmonized across multiple reputable but not identical measures
  (typical value, listing price, sale price, HPI).
