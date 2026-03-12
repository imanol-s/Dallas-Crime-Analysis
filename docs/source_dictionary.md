# Source Dictionary

## Raw sources

## `data/raw/crime_records.csv`

- Origin: Dallas OpenData Police Incidents endpoint.
- Key fields used downstream:
  - `reported_at`: incident timestamp.
  - `offense_family`: normalized category (`violent`, `property`, `other`).
  - `zip`: incident ZIP code.

## `data/raw/acs_zcta.csv`

- Origin: U.S. Census ACS 5-year ZIP Code Tabulation Area data.
- Key fields used downstream:
  - `zip`
  - `population`
  - `median_household_income`
  - `poverty_rate`
  - `owner_occupied_share`
  - `median_gross_rent`

## `data/raw/housing_market.csv`

- Origin: Firecrawl-assisted extraction from a prioritized mix of public ZIP market pages,
  plus structured Realtor ZIP inventory feeds for broader coverage and recent-trend features.
- URL patterns:
  - Zillow: `https://www.zillow.com/home-values/`
  - Realtor.com: `https://www.realtor.com/local/market/texas/zipcode-{zip}`
  - Realtor ZIP current feed: `https://econdata.s3-us-west-2.amazonaws.com/Reports/Core/RDC_Inventory_Core_Metrics_Zip.csv`
  - Realtor ZIP history feed: `https://econdata.s3-us-west-2.amazonaws.com/Reports/Core/RDC_Inventory_Core_Metrics_Zip_History.csv`
  - Redfin: `https://www.redfin.com/zipcode/{zip}/housing-market`
- Key fields used downstream:
  - `zip`
  - `home_value`
  - `as_of_date`
  - `annual_change_pct`
  - `median_rent`
  - `source`
  - `source_url`
  - `metric_label`
  - `supplemental_sources`
  - `realtor_listing_price`
  - `realtor_active_listing_count`
  - `realtor_median_days_on_market`
  - `realtor_pending_ratio`
  - `realtor_hist_listing_price_12m_avg`
  - `realtor_hist_listing_price_12m_change`
  - `realtor_hist_active_listing_count_12m_avg`
  - `realtor_hist_median_days_on_market_12m_avg`
  - `realtor_hist_pending_ratio_12m_avg`

## `data/raw/housing_market_history.csv`

- Origin: combined historical housing panel built from:
  - Realtor ZIP monthly history for recent market observations
  - official FHFA ZIP5 annual HPI for year-2000 backfill
- Coverage target:
  - years `2000` through `2025`
- URL patterns:
  - Realtor ZIP history feed: `https://econdata.s3-us-west-2.amazonaws.com/Reports/Core/RDC_Inventory_Core_Metrics_Zip_History.csv`
  - FHFA ZIP5 annual HPI: `https://www.fhfa.gov/hpi/download/annual/hpi_at_zip5.xlsx`
- Key fields:
  - `zip`
  - `period_start`, `period_end`, `period_year`, `period_month`
  - `frequency`
  - `source`, `source_url`, `metric_label`
  - `price_signal_value`, `price_signal_unit`
  - Realtor monthly history columns where available
  - FHFA annual HPI columns where available

## Processed model dataset

## `data/processed/model_dataset.csv`

Key analysis columns:

- Geography and period:
  - `zip`, `period_start`, `period_end`, `centroid_latitude`, `centroid_longitude`
- Crime volume and rates:
  - `total_incidents`, `violent_incidents`, `property_incidents`, `other_incidents`
  - `total_rate_per_1000`, `violent_rate_per_1000`, `property_rate_per_1000`
- Housing:
  - `home_value`, `log_home_value`, `as_of_date`, `annual_change_pct`, `median_rent`
- ACS controls:
  - `population_acs`, `median_household_income`, `poverty_rate`,
    `owner_occupied_share`, `median_gross_rent`
- Source metadata:
  - `source`, `source_url`
  - `metric_label`, `supplemental_sources`

## `data/processed/housing_history_panel.csv`

Key analysis columns:

- `zip`
- `period_start`, `period_end`, `period_year`, `period_month`
- `frequency`
- `source`, `source_url`, `metric_label`
- `price_signal_value`, `price_signal_unit`
- `realtor_hist_listing_price`
- `realtor_hist_active_listing_count`
- `realtor_hist_median_days_on_market`
- `realtor_hist_pending_ratio`
- `fhfa_annual_change_pct`
- `fhfa_hpi`, `fhfa_hpi_1990_base`, `fhfa_hpi_2000_base`

## Notes

- `population` may appear from the crime-rate merge and `population_acs` from ACS controls.
- `centroid_latitude` and `centroid_longitude` are median incident coordinates by ZIP from the
  current crime acquisition window and are used only for geography-aware reporting.
- `home_value` is a harmonized housing price signal, not a single-source measure:
  Zillow contributes typical home value, Realtor.com contributes median home price,
  Realtor's structured ZIP feed contributes median listing price when page-level sources
  are unavailable, and Redfin contributes median sale price as the last fallback.
- All numeric fields are coerced to numeric during build and/or analysis steps.
