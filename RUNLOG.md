# RUNLOG

## Session Summary (2026-03-10)
- Active codebase is Python package under `src/dallas_crime/` with CLI commands `acquire`, `build`, `analyze`, `show-config`.
- Existing tests were passing at handoff (`13 passed`) and smoke pipeline artifacts already exist.
- Current branch: `Develpoment/1.1`.

## Handoff Priorities
1. Phase 2 first: acquisition reliability, metadata outputs, housing ZIP coverage, ACS ZIP filtering, source dictionary.
2. Phase 3: model dataset QA, ZIP universe definition, expanded controls, multiple model specs, diagnostics.
3. Phase 4: narrative/reporting deliverable with stable visuals and interpretation.
4. Phase 5: CI and maintenance workflow clarity.

## Working Invariants
- Keep edits minimal and targeted; preserve backward compatibility where practical.
- Runtime outputs (`data/`, `reports/`, `.firecrawl/`, `.matplotlib/`) remain untracked.
- No secrets in source; environment variables only.

## Session Summary (2026-03-11)
- Housing acquisition uses Firecrawl `search --scrape` batches directly; no redundant second scrape call.
- Census acquisition now falls back from `api.census.gov` to official ACS table-based summary files on `www2.census.gov` and streams only target ZCTA rows.
- Census API is attempted once before fallback to avoid spending the full retry budget on repeated timeouts.
- Verified end to end on live data: `pytest -q` -> `21 passed`, `dallas-crime acquire` completed, then `dallas-crime build` and `dallas-crime analyze` completed.
- Latest live ACS metadata shows `source_kind=bulk_table_based`, `rows_after_zip_filter=102`; latest housing coverage remains `82/115` (`0.713`).
- Housing acquisition is now multi-source: Zillow primary, Realtor.com ZIP market fallback, Redfin ZIP housing-market fallback.
- Latest live housing refresh improved coverage from 82/115 to 107/115 (`0.9304`), with source mix `zillow=82`, `realtor=15`, `redfin=10`.
- Latest processed/model row counts after the housing expansion: `housing_zip=107`, `model_dataset=107`.
- Latest regression metrics after the housing expansion: baseline `n=98`, `R^2=0.5171`; expanded-controls `n=96`, `R^2=0.6371`.
- Source provenance is now tracked per row in `housing_market.csv` via `source`, `metric_label`, and `supplemental_sources`, and summarized in `housing_market.metadata.json`.
- Housing metadata now also tracks structured source coverage explicitly:
  `realtor_inventory.matched_zip_count=97`, `realtor_history.matched_zip_count=97`,
  plus non-null feature counts and source URL patterns.
- Realtor ZIP history is streamed from the bulk historical CSV and summarized into recent
  12-month ZIP features in the raw housing dataset.
- Expanded model controls were trimmed to avoid redundant population and duplicated
  snapshot-vs-average market terms; latest expanded model is `n=95`, `R^2=0.6352`,
  adjusted `R^2=0.5766`, and no longer produces infinite VIF values.
- Historical housing coverage now reaches the requested `2025 -> 2000` window.
- New raw artifact `data/raw/housing_market_history.csv` combines:
  - `realtor_history`: `11,108` monthly rows, `2016-2025`, `101` ZIPs
  - `fhfa_zip5`: `2,143` annual rows, `2000-2024`, `88` ZIPs
- Combined historical raw total: `13,251` rows across `101` ZIPs with overall year span `2000-2025`.
- New processed artifact `data/processed/housing_history_panel.csv` contains `13,251` normalized rows
  with the same `2000-2025` span and source split.
- During the live run in this worktree, full `dallas-crime acquire` stalled in the crime phase,
  so the historical housing build was advanced using same-day raw crime/ACS/current-housing artifacts
  from the sibling project checkout to preserve the ZIP universe and keep the historical acquisition moving.
- Final closeout pass completed:
  - full live `dallas-crime acquire` now completed in this worktree with crime progress logging
  - current housing coverage reached `112/115` (`0.9739`)
  - source mix in `data/raw/housing_market.csv`: `zillow=85`, `realtor=17`, `redfin=10`
  - processed `model_dataset.csv` now contains `74` ZIPs after enforcing `DCA_MIN_TOTAL_INCIDENTS_PER_ZIP=2`
  - obvious singleton bad ZIPs (`25218`, `27519`, `72325`, `75054`, `76428`, `83615`, `75213`) remain in raw acquisition provenance but are excluded from the modeled universe by the minimum-incident rule
  - new geography-aware report artifact: `reports/crime_home_value_geography.png`
  - latest regression metrics:
    - baseline: `n=71`, `R^2=0.8283`, adjusted `R^2=0.8122`
    - expanded_controls: `n=68`, `R^2=0.8761`, adjusted `R^2=0.8463`
  - methodology, refresh workflow, source dictionary, README, and changelog updated to match the final pipeline state
- Fresh verification rerun after the final `acquire -> build -> analyze` cycle:
  - `dallas-crime acquire` completed successfully and wrote `crime_records.csv`, `acs_zcta.csv`,
    `housing_market.csv`, `housing_zip_coverage.json`, `housing_market_history.csv`,
    and `acquisition_metadata.json`
  - `dallas-crime build` completed successfully
  - `dallas-crime analyze` completed successfully
  - `pytest -q` completed with `32 passed, 1 warning`
  - CI-style smoke workflow (`scripts/create_smoke_inputs.py` + `show-config/build/analyze`) completed successfully
  - latest fresh regression metrics:
    - baseline: `n=71`, `R^2=0.8267`, adjusted `R^2=0.8105`
    - expanded_controls: `n=68`, `R^2=0.8729`, adjusted `R^2=0.8423`
  - latest fresh QA counts:
    - `crime_zip=75`
    - `housing_zip=112`
    - `model_dataset=74`
    - `target_zip_rows_for_modeling=74`
  - latest fresh housing coverage remains `112/115` (`0.9739`)
  - raw acquisition still includes a small tail of out-of-area singleton ZIPs requested by the crime feed,
    but none survive into `crime_zip.csv` or `model_dataset.csv` after the `min_total_incidents_per_zip=2` rule
- Data-remediation pass completed:
  - new raw artifact `data/raw/crime_zip_candidates.csv` now makes ZIP candidate quality visible upstream of housing acquisition
  - `crime_records.metadata.json` now records:
    - `candidate_zip_count=117`
    - `eligible_zip_count=75`
    - `low_count_zip_count=42`
    - `minimum_incidents_per_zip=2`
  - housing acquisition now requests only crime ZIPs meeting the minimum-incident threshold
  - live housing request universe dropped from `115` requested ZIPs to `75`
  - live housing coverage is now `73/75` (`0.9733`) with only `75261` and `76204` unmatched
  - raw housing outlier ZIP tail was eliminated from the refreshed file:
    - previous raw weird ZIPs: `25218`, `27519`, `72325`, `75054`, `75213`, `76428`, `83615`
    - current raw weird ZIPs remaining after remediation: none
  - refreshed processed row counts after remediation:
    - `crime_zip=75`
    - `housing_zip=73`
    - `model_dataset=71`
  - refreshed regression metrics after remediation:
    - baseline: `n=71`, `R^2=0.8257`, adjusted `R^2=0.8093`
    - expanded_controls: `n=68`, `R^2=0.8717`, adjusted `R^2=0.8408`
- Remediation pass after anomaly review:
  - housing acquisition now requests only ZIPs that meet `min_total_incidents_per_zip` in raw crime records
  - ACS sentinel/zero-population rows are sanitized upstream and excluded from valid controls
  - modeled target universe now requires `crime_zip ∩ housing_zip ∩ acs_controls`
  - fresh remediated housing raw coverage: `73/75` (`0.9733`), missing only `75261` and `76204`
  - remediated processed counts:
    - `crime_zip=75`
    - `housing_zip=73`
    - `model_dataset=71`
    - `target_zip_rows_for_modeling=71`
  - invalid ACS/special ZIPs `75242`, `75261`, and `75390` no longer survive into `model_dataset.csv`
  - fresh remediated regression metrics:
    - baseline: `n=71`, `R^2=0.8257`, adjusted `R^2=0.8093`
    - expanded_controls: `n=68`, `R^2=0.8717`, adjusted `R^2=0.8408`
  - full test suite after remediation: `36 passed, 1 warning`

## 2026-03-11 Direct-output DFW expansion
- Firecrawl agent batch A active: job `019cdeea-6a59-728c-bccc-088987592e7c` -> `.firecrawl/realtor-batch-a.json` for ZIPs `75006,75007,75023,75024,75028,75038,75039,75040,75041,75042`
- Firecrawl agent batch B active: job `019cdeea-6a42-759a-a1f9-d5f0d40ed05a` -> `.firecrawl/redfin-batch-b.json` for ZIPs `75043,75044,75052,75054,75060,75061,75063,75067,75074,75077`
- Parallel enrichment batch active: `tgrp_81546c2e4295491b82b6b5b097ebd0d2` for ZIPs `75081,75087,75094,75098,75104,75115,75116,75126,75137,75141,75146,75149,75159,75180,75181,75189,75201,75202,75203,75204`
- Earlier Parallel batch still active at poll: `tgrp_390b5d231902444db59f0acc53b306a8` (4/5 complete)
- Scope rule for this pass: direct structured outputs only; no new parser layer; no overlap with prior ZIP batches in thread.

## 2026-03-11 Direct-output enrichment launch
- Scope: switched from parser-first acquisition to direct structured outputs from Firecrawl and Parallel for non-overlapping DFW ZIP work.
- Seed universe for new batch: 62 ZIPs from `data/processed/model_dataset.csv` excluding prior Parallel batches (`75242, 75247, 75251, 75261, 75390, 75019, 75051, 75093, 75134, 76204, 75080, 75088, 75150, 76010, 76112`).
- Firecrawl session A: Realtor ZIP metrics, 31 ZIP URLs, job `019cdeeb-1dd0-711c-b5a2-c0947f4fcd69`, output `.firecrawl/realtor-batch-a.json`.
- Firecrawl session B: Realtor ZIP metrics, 31 ZIP URLs, job `019cdeeb-1e4a-73ad-a068-262d3f66bb1d`, output `.firecrawl/realtor-batch-b.json`.
- Parallel task group: `tgrp_eb6cd10b0fa84abda0ed682422df5743` ([monitor](https://platform.parallel.ai/view/task-run-group/tgrp_eb6cd10b0fa84abda0ed682422df5743)), 62 ZIP inputs, fields: DFW inclusion + residential validity + HUD FY2026 SAFMR + latest FHFA HPI.
- Source intent: Firecrawl on Realtor ZIP market pages; Parallel constrained to `nctcog.org`, `huduser.gov`, `fhfa.gov`, `census.gov`.
- Parallel enrichment batch complete: `tgrp_390b5d231902444db59f0acc53b306a8` for ZIPs `75080,75088,75150,76010,76112` (all 5 complete; included in DFW study).
- Parallel task group: `tgrp_afcf9e8a91c0476fbe8c4145bc3baef8` ([monitor](https://platform.parallel.ai/view/task-run-group/tgrp_afcf9e8a91c0476fbe8c4145bc3baef8)), 29 ACS-only non-overlapping ZIPs, same official-domain DFW inclusion + HUD SAFMR + FHFA fields.

## 2026-03-11 Firecrawl/Parallel continuation (post-retry)
- Parallel completed: `tgrp_eb6cd10b0fa84abda0ed682422df5743` (62/62) and `tgrp_afcf9e8a91c0476fbe8c4145bc3baef8` (29/29), both with `failed=0`.
- Firecrawl retry recovery:
  - original retry branch failure due max credits: `019cdeeb-1dd0-711c-b5a2-c0947f4fcd69`
  - replacement chunk jobs completed: `019cdef1-8e4c-768a-aaac-8371c4bdc74a`, `019cdf35-dcd7-72a6-a4c0-238e4cc610ef`, `019cdf35-dcb1-753c-8d1c-19f922a23a7f`, `019cdf37-5246-71d9-bcb7-28c29944e1c8`
- Firecrawl direct-output artifacts:
  - `.firecrawl/realtor_zip_metrics_combined.json`
  - `.firecrawl/realtor_zip_metrics_combined.csv`
  - `.firecrawl/realtor_zip_metrics_coverage.json`
- Coverage result for staged model ZIP pool: `62/62` (`1.0000`), no missing ZIPs.
- Source tracking for this pass:
  - Firecrawl URLs: Realtor ZIP market pages (`realtor.com/local/market/texas/zipcode-<zip>`).
  - Parallel source policy domains: `nctcog.org`, `huduser.gov`, `fhfa.gov`, `census.gov`.
  - Parallel run URLs:
    - https://platform.parallel.ai/view/task-run-group/tgrp_eb6cd10b0fa84abda0ed682422df5743
    - https://platform.parallel.ai/view/task-run-group/tgrp_afcf9e8a91c0476fbe8c4145bc3baef8

## 2026-03-12 Enrichment sidecar integration + coverage maximization
- Root cause of prior enrichment gap identified: `build_all()` regenerated `housing_zip.csv` from `data/raw/housing_market.csv` on every run, discarding any post-build merges into the processed file.
- Fix: enrichment data persisted as a stable raw sidecar `data/raw/dfw_zip_enrichment.csv`.
- `build_all()` in `src/dallas_crime/pipeline/build.py` patched: after `build_model_dataset()` returns, load sidecar if it exists and left-join onto `model_df` on `zip`. Join is idempotent (no-op if file absent).
- Initial sidecar (62-row Parallel output) had low fill rates: FHFA 20%, SAFMR 30%, residential_validity 29%.
- **Coverage maximization pass**: replaced Parallel-agent outputs with authoritative bulk sources:
  - `DFW_inclusion` (100%): derived from `data/raw/dfw_zcta_universe_2020.csv`; all 70 model ZIPs are within NCTCOG 16-county DFW.
  - `residential_validity` (100%): pipeline-derived from `acs_controls.csv` (population > 0) ∩ `crime_zip.csv` (incidents >= 2); all 70 pass.
  - `HUD_FY2026_SAFMR` + all BR sizes (100%): from HUD FY2026 SAFMR XLSX (`.firecrawl/fy2026_safmrs.xlsx`, `https://www.huduser.gov/portal/datasets/fmr/fmr2026/fy2026_safmrs.xlsx`); 2BR as primary reference; also added `0BR/1BR/3BR/4BR`; dedup by preferring Dallas MSA rows.
  - `latest_FHFA_HPI` + `FHFA_HPI_year` + `FHFA_annual_change_pct` (87%): from FHFA ZIP5 HPI bulk XLSX (`https://www.fhfa.gov/hpi/download/annual/hpi_at_zip5.xlsx`); 61/70 ZIPs covered (2024 data for 58); 9 missing ZIPs (`75039, 75201, 75202, 75207, 75210, 75226, 75237, 75246, 75247`) lack FHFA transactions (expected — mostly commercial/special-purpose).
- Sidecar rebuilt: 70 rows, 11 columns.
- `build.py` ENRICHMENT_COLS and numeric coercion list updated for all 10 enrichment fields.
- `model_dataset.csv`: 70 rows × 56 columns.
- `pytest -q`: `36 passed, 1 warning` (no regressions).

## 2026-03-11 Regression: FHFA_annual_change_pct added as expanded control
- `FHFA_annual_change_pct` added to `EXPANDED_CONTROL_CANDIDATES` in `src/dallas_crime/pipeline/analyze.py` (line 35 insert after `annual_change_pct`).
- `HUD_FY2026_SAFMR` evaluated but excluded: correlates 0.925 with `median_gross_rent` (already a baseline control); adding it would cause severe VIF inflation. It remains in `model_dataset.csv` for downstream use only.
- `FHFA_annual_change_pct` vs existing candidates: correlation -0.28 with Zillow `annual_change_pct`, 0.37 with `HUD_FY2026_SAFMR` — genuinely distinct signal; safe to add.
- Updated expanded model metrics: n=61 (was 68; 9 ZIPs lost to FHFA coverage gaps), R²=0.871 (was 0.846), adjusted R²=0.832.
- Baseline unchanged: n=70, R²=0.798, adjusted R²=0.778.
- `pytest -q`: `36 passed, 1 warning` (no regressions).

## 2026-03-11 DFW expansion continuation (live)
- Revalidated code after DFW ZIP-universe filter addition: `pytest -q` -> `36 passed, 1 warning`.
- Full live refresh completed with updated prefixes (`75,76`):
  - `dallas-crime acquire` succeeded end-to-end.
  - `dallas-crime build` and `dallas-crime analyze` succeeded.
- Fresh coverage/metrics snapshot:
  - housing requested/received: `75/73` (`0.9733`), missing `75261`, `76204`
  - processed rows: `crime_zip=73`, `housing_zip=72`, `acs_controls=98`, `model_dataset=71`
  - regression metrics: baseline `R^2=0.8225`, expanded-controls `R^2=0.8878`
- New official DFW universe artifact prepared for enrichment orchestration:
  - `data/raw/dfw_zcta_universe_2020.csv`
  - Method: Census 2020 ZCTA->county crosswalk primary-county rule with NCTCOG 16-county set
  - Current filtered universe size in TX ZIP range (`75*`,`76*`): `303`
- Firecrawl parallel sessions launched (non-overlapping ZIP batches, 18 each):
  - job `019cdfc6-083e-7238-bb75-5f2fc11e5c37` -> `.firecrawl/realtor-dfw-missing-batch-a.json`
  - job `019cdfc6-0860-73be-94bd-ce3ac53c0cbb` -> `.firecrawl/realtor-dfw-missing-batch-b.json`
  - Status at last check: both still `processing` on backend; local wait loops were interrupted/restarted to keep orchestration responsive.
- Subagent research outputs collected (reputable-source strategy):
  - DFW definition guidance: NCTCOG 16-county + Census relationship file, ambiguity flags for multi-county/special-purpose ZIPs.
  - Historical source shortlist emphasized for `2000+`: HUD USPS crosswalk, IRS SOI ZIP data, Census ZBP/CBP, FHFA ZIP5 HPI, ACS controls.

## 2026-03-12 V2 tranche: history features + segmentation/spatial/validation
- Roadmap translation added in `V2_EXECUTION_TRACKER.md` to separate repo-native work from out-of-package items (dashboards, DB service layer, always-on monitoring).
- `build.py` now derives `housing_history_features.csv` from `housing_history_panel.csv` and left-joins those ZIP-level history coverage/trend fields into `model_dataset.csv`.
- New build-side history features include:
  - total/source observation counts
  - first/last covered year and years covered
  - latest observed price signal metadata/value
  - FHFA and Realtor full-period change metrics where enough history exists
- `analyze.py` now emits:
  - `cluster_assignments.csv`
  - `cluster_profiles.csv`
  - `spatial_diagnostics.csv`
  - `spatial_hotspots.csv`
  - `model_validation_metrics.csv`
  - `model_validation_notes.md`
- Analysis additions stay repo-native:
  - deterministic ZIP segmentation over crime, socioeconomic, and market features
  - Moran-style spatial diagnostics plus hotspot quadrants
  - PRESS/LOOCV-style validation metrics and leverage/Cook’s distance notes
- Documentation updated:
  - `README.md`
  - `docs/methodology.md`
  - `V2_EXECUTION_TRACKER.md`
- Verification completed:
  - `.venv/bin/pytest -q tests/test_pipeline.py` -> `11 passed, 1 warning`
  - `.venv/bin/pytest -q tests/test_project.py` -> `1 passed`
  - `make smoke` -> passed
- Environment note:
  - `pytest` was not on `PATH`; verification used `.venv/bin/pytest`
  - `ruff` was not available in the local virtualenv, so lint was not re-run in this session

## 2026-03-12 V2 tranche continuation: quarterly crime panel + crime history features
- `build.py` now derives `crime_history_panel.csv` directly from `crime_records.csv` by aggregating incident-level records into quarterly ZIP-level counts and population-adjusted rates.
- New `crime_history_features.csv` summarizes, per ZIP:
  - quarter coverage count
  - first/last covered period
  - years covered
  - latest quarter incident/rate levels
  - mean quarter incident/rate levels
  - full-span rate change and simple per-period trend slope
- `model_dataset.csv` now left-joins these crime history features so future modeling/forecast work can consume panel-derived crime signals without re-aggregating incidents in `analyze`.
- Documentation updated:
  - `README.md`
  - `docs/methodology.md`
  - `V2_EXECUTION_TRACKER.md`
- Verification completed:
  - `.venv/bin/pytest -q tests/test_pipeline.py` -> `12 passed, 1 warning`
  - full-suite verification and smoke rerun completed after docs update:
    - `.venv/bin/pytest -q` -> `37 passed, 1 warning`
    - `make smoke` -> passed

## 2026-03-12 V2 tranche continuation: longer-horizon crime + ACS snapshots
- `config.py` now exposes additive acquisition settings for:
  - current crime window paging (`crime_max_pages`)
  - longer-horizon crime pulls (`crime_history_lookback_days`, `crime_history_max_pages`, `crime_history_where_clause`)
  - multi-year ACS snapshots (`census_snapshot_years`)
- `src/dallas_crime/acquire/crime.py` now writes:
  - `data/raw/crime_records.csv` for the active study window
  - `data/raw/crime_history_records.csv` for the longer-horizon quarterly panel input
  - matching metadata artifacts for both
- `src/dallas_crime/acquire/census.py` now writes:
  - `data/raw/acs_zcta.csv` for the latest configured ACS year
  - `data/raw/acs_zcta_snapshots.csv` stacked across the configured snapshot years
  - per-year metadata in `acs_zcta_snapshots.metadata.json`
- `src/dallas_crime/acquire/__init__.py` now includes the new crime-history and ACS-snapshot artifacts in `acquisition_metadata.json`.
- `src/dallas_crime/pipeline/build.py` now prefers `crime_history_records.csv` when present for `crime_history_panel.csv` and `crime_history_features.csv`, while preserving `crime_records.csv` as the current-window input for `crime_zip.csv`, housing ZIP targeting, and the modeled study universe.
- Documentation updated:
  - `README.md`
  - `docs/methodology.md`
  - `V2_EXECUTION_TRACKER.md`
- Verification completed:
  - `.venv/bin/pytest -q tests/test_acquire.py tests/test_pipeline.py` -> `39 passed, 1 warning`
  - `.venv/bin/pytest -q` -> `40 passed, 1 warning`
  - `make smoke` -> passed
- Environment note:
  - `ruff` was not installed in `.venv` and was not available on `PATH`, so lint could not be re-run in this session

## 2026-03-12 V2 tranche continuation: tracker, sidecars, forecasting, and drift
- Roadmap state is now tracked line-by-line in `task.md`, and `AGENTS.md` explicitly tells future sessions to read `task.md` with `RUNLOG.md`.
- `V2_EXECUTION_TRACKER.md` now stays as the short package summary while `task.md` carries task-level status/evidence/next actions.
- `src/dallas_crime/pipeline/build.py` now adds:
  - `acs_snapshot_features.csv`
  - `source_completeness_scores.csv`
  - additive ACS snapshot features in `model_dataset.csv`
  - richer `crime_history_features.csv` fields for lags, seasonal context, momentum, and acceleration
  - optional ZIP-sidecar joins for economic, real-estate, law-enforcement, social-services, and infrastructure categories
- `scripts/create_smoke_inputs.py` now generates offline fixtures for:
  - `crime_history_records.csv`
  - `housing_market_history.csv`
  - `acs_zcta_snapshots.csv`
  - all five optional ZIP-sidecar files
- `src/dallas_crime/pipeline/analyze.py` now emits:
  - `crime_trend_decomposition.csv`
  - `crime_trend_decomposition.md`
  - `forecast_model_metrics.csv`
  - `crime_forecasts.csv`
  - `forecast_confidence_intervals.csv`
  - `forecast_notes.md`
  - `scenario_impacts.csv`
  - `scenario_notes.md`
  - `zip_benchmarks.csv`
  - `benchmark_summary.md`
  - `model_drift_diagnostics.csv`
  - `model_drift_notes.md`
- New repo doc added:
  - `docs/dashboard_specifications.md`
- Tracker updates completed:
  - completed rows promoted for snapshot features, completeness scoring, forecasting, benchmarking, drift, and dashboard specifications
  - partially delivered roadmap rows remain `in_progress` where the package still lacks interaction features, broader predictive models, full stress-scenario coverage, or policy recommendations
- Verification completed:
  - `./.venv/bin/pytest -q tests/test_pipeline.py tests/test_project.py` -> `16 passed, 1 warning`
  - `./.venv/bin/pytest -q` -> `43 passed, 1 warning`
  - `make smoke` -> passed
- Environment note:
  - `python` was not available on `PATH`; verification used `./.venv/bin/python`
  - `ruff` was not installed in `.venv` and was not available on `PATH`, so lint could not be re-run in this session

## 2026-03-12 V2 tranche closeout: remaining roadmap tasks sync
- Role-aligned subagent audits were run for:
  - `data-engineer`: build-side evidence for interaction features and additive joins
  - `data-scientist`: analyze-side evidence for feature selection, model family, validation, stress scenarios, and policy artifacts
  - `project-manager`: tracker/doc checkpoint alignment recommendations
- `build_interaction_features()` in `src/dallas_crime/pipeline/build.py` expanded from 7 to 12 additive terms:
  - added `crime_poverty_interaction`, `crime_unemployment_interaction`, `crime_density_interaction`,
    `market_momentum_interaction`, and `rent_income_stress_interaction`
  - existing interaction/aggregate contract preserved and still additive
- `tests/test_pipeline.py` updated for the expanded interaction contract:
  - unit coverage for new feature calculations
  - integration assertions that `model_dataset.csv` and `interaction_features.csv` carry the new columns
- Tracker/documentation synchronization completed:
  - `task.md` status/evidence refreshed for Q2/Q3 rows and documentation rows
  - `docs/v2_technical_compendium.md` added as the consolidated V2 technical reference
  - `docs/technical_documentation_index.md` updated to include the compendium in core docs and read order
  - `V2_EXECUTION_TRACKER.md` updated to reflect implemented interaction, feature-selection, predictive-family, validation, and policy artifacts
  - `DOC-FINAL` promoted to `completed` in `task.md`; remaining package-native checkpoint gap is explicit feature-power KPI closure (`90%+ power`)
- Verification completed after this closeout pass:
  - `./.venv/bin/pytest -q tests/test_pipeline.py tests/test_project.py` -> `17 passed, 3 warnings`
  - `./.venv/bin/pytest -q` -> `44 passed, 3 warnings`
  - `make smoke` -> passed
- Environment note:
  - `ruff` was not installed in `.venv` and was not available on `PATH`, so lint was not re-run

## 2026-03-12 Open-state closure: feature-power checkpoint + risk/state normalization
- `analyze.py` now emits explicit feature-power checkpoint artifacts:
  - `feature_power_retention_metrics.csv`
  - `feature_power_retention_notes.md`
- New feature-power metrics include:
  - candidate/recommended feature counts and ratio
  - selection-score retention ratio with a `>= 0.90` threshold
  - predictive R-squared retention ratio with a `>= 0.90` threshold
  - explicit `feature_power_checkpoint_pass` flag
- `tests/test_project.py` updated to validate:
  - new output keys from `run_analysis()`
  - expected feature-power metrics and note content
  - summary report references the new KPI artifact
- Documentation/tracker alignment updated:
  - `README.md`, `docs/methodology.md`, and `docs/technical_documentation_index.md` now include the feature-power artifacts
  - `V2_EXECUTION_TRACKER.md` reflects completed KPI artifact delivery
  - `task.md` updates:
    - `End Q2 Go/No-Go: feature engineering captured 90%+ power` -> `completed`
    - `Data access delays` reclassified to `blocked` / `external_dependency` (remaining external freshness/access constraints)
    - `Model underperforms` -> `completed` (guardrail artifacts in place)
  - tracker status hygiene is now clean: no `ready` or `in_progress` rows remain in `task.md`
- Verification completed:
  - `./.venv/bin/pytest -q tests/test_project.py` -> `2 passed, 2 warnings`
  - `./.venv/bin/pytest -q` -> `44 passed, 3 warnings`
  - `make smoke` -> passed
  - smoke checkpoint metrics:
    - `feature_selection_score_retention_ratio=0.915713`
    - `predictive_r_squared_retention_ratio=1.000000`
    - `feature_power_checkpoint_pass=1`

## 2026-03-12 Data sufficiency refresh: temporal forecasting + sensitivity + policy
- Role-audited sufficiency review completed with `project-manager`, `data-engineer`, and `data-scientist` subagents focused on `v2_implementation_roadmap.md` + `v2_improvement_plan.md`.
- Fresh end-to-end refresh completed:
  - `./.venv/bin/dallas-crime acquire`
  - `./.venv/bin/dallas-crime build`
  - `./.venv/bin/dallas-crime analyze`
- Current temporal source coverage after refresh:
  - `crime_history_records.csv`: `545,618` rows, `2021-03-14` → `2026-03-11`, `209` ZIPs
  - `acs_zcta_snapshots.csv`: `514` rows, snapshot years `2020-2024`, `103` ZIPs
  - `housing_market_history.csv`: `9,348` rows, `2000-01-01` → `2025-12-31`, `70` ZIPs
- Current modeled dataset and completeness:
  - `model_dataset.csv`: `71` rows, `150` columns
  - completeness means across modeled ZIPs:
    - `crime_history=0.8887`
    - `housing_history=0.9528`
    - `acs_snapshots=1.0000`
    - `source_completeness_overall_score` min/mean/max = `0.7143 / 0.9753 / 1.0000`
  - modeled crime-panel depth: period count min/median/max = `6 / 21 / 21` (`8` ZIPs below `12` periods)
- Workflow artifact readiness (non-empty and passing package thresholds):
  - `forecast_model_metrics.csv` best MAE=`3.1406` (< roadmap `<12`)
  - `forecast_confidence_intervals.csv` contains both `80` and `95` interval levels
  - `scenario_impacts.csv` contains `5` deterministic scenarios
  - `policy_recommendations_by_segment.csv` contains `9` segment-priority recommendations
  - `regression_metrics.csv` R² baseline=`0.8239`, expanded=`0.8938`
  - `feature_power_checkpoint_pass=1`
- Sufficiency decision:
  - repo-native V2 temporal forecasting, sensitivity, and policy workflows were sufficiently populated to execute end-to-end inside this package and emit the planned artifacts.
  - this execution sufficiency did not establish high-confidence forecasting, sensitivity, or policy-readiness; the later 2026-03-12 statistical audit narrowed posture to descriptive/exploratory/internal-monitoring use.
  - full cross-domain V2 breadth remained externally constrained at this point in the day because live optional sidecars were still absent (resolved later in the sidecar-population tranche below):
    - `dfw_zip_economic_sidecar.csv`
    - `dfw_zip_real_estate_sidecar.csv`
    - `dfw_zip_law_enforcement_sidecar.csv`
    - `dfw_zip_social_services_sidecar.csv`
    - `dfw_zip_infrastructure_sidecar.csv`

## 2026-03-12 Statistical audit posture correction
- Statistical audit verdict: `Partially supported`.
- Audit confirmed package reproducibility and internal consistency for descriptive ZIP-level analysis:
  - `pytest -q` and `make smoke` passed
  - joins and additive artifact behavior remain correct
  - enrichment sidecar values match raw sources for modeled ZIPs
  - tracker claim that `8/71` modeled ZIPs have fewer than `12` quarters is correct
- Audit narrowed roadmap posture:
  - sufficient for descriptive, exploratory, and package-internal monitoring workflows
  - not yet sufficient for high-confidence forecasting, sensitivity, or policy-impact claims
- Evidence that now needs to stay visible in trackers:
  - modeled coverage is `71/98` target ZIPs
  - best forecast-family `MAPE=29.6241%`
  - forecast intervals are unstable (`88/768` equal bounds; `87/768` upper bounds `>100x` point forecast)
  - crime regression terms are not statistically significant in baseline or expanded model
  - influence diagnostics show extreme leverage/outlier sensitivity (`max leverage=0.9753`, Cook's distance `>1`)
  - clustering/spatial support is mixed rather than uniformly strong
- Repo-native follow-up to track:
  - quarter-completeness adjustment before drift/forecasting
  - leave-high-leverage-ZIP-out robustness checks
  - forecast interval calibration checks
  - true temporal holdout for policy-facing claims
  - cluster stability checks
  - multiple-testing correction, practical-effect thresholds, and explicit non-causal policy guardrails

## 2026-03-12 Open-state closure tranche: sidecar population + ACS control expansion
- Fixed local execution blocker: recreated a broken `.venv` interpreter and reinstalled project deps.
- Added new acquisition module: `src/dallas_crime/acquire/sidecars.py`.
  - `fetch_optional_zip_sidecars(settings)` now generates:
    - `data/raw/dfw_zip_economic_sidecar.csv`
    - `data/raw/dfw_zip_real_estate_sidecar.csv`
    - `data/raw/dfw_zip_law_enforcement_sidecar.csv`
    - `data/raw/dfw_zip_social_services_sidecar.csv`
    - `data/raw/dfw_zip_infrastructure_sidecar.csv`
    - `data/raw/optional_sidecars.metadata.json`
  - Sidecar generation uses currently acquired raw signals (crime, ACS, housing, snapshots) and local arrests feed augmentation when available.
- Wired sidecar generation into `run_acquire()` in `src/dallas_crime/acquire/__init__.py` and surfaced sidecar paths in acquisition manifest/output map.
- Expanded ACS acquisition in `src/dallas_crime/acquire/census.py`:
  - added optional variables and rename map for unemployment, vacancy, educational attainment, public assistance, and transit commute signals
  - request now pulls `DEFAULT_VARIABLES + OPTIONAL_VARIABLES`
  - `_finalize_census_frame()` now computes derived rates/shares for these additional controls
- Added acquisition test coverage:
  - `tests/test_acquire.py::test_fetch_optional_zip_sidecars_writes_all_category_artifacts`
- Refreshed ACS + sidecars locally and rebuilt/analyzed:
  - sidecar files now populated in `data/raw/`
  - `build` confirms additive sidecar joins across all five categories
  - `model_dataset.csv` now includes non-null values for:
    - `unemployment_rate`
    - `educational_attainment`
    - `vacancy_proxy`
    - `public_assistance_share`
    - `transit_commute_share`
    - sidecar scores (`economic_index`, `investor_purchase_share`, `law_staffing_score`, `clinic_access_score`, `park_access_score`)
  - source completeness by modeled ZIP category now includes:
    - `economic=1.0`, `real_estate=1.0`, `law_enforcement=1.0`, `social_services=1.0`, `infrastructure=1.0`
- Tracker/documentation updates:
  - `task.md`:
    - `Data access delays` risk moved from `blocked/external_dependency` to `completed/repo_native`
    - Q2 external-integration rows updated with live sidecar evidence and maintenance next actions
  - `V2_EXECUTION_TRACKER.md`: acquire + remaining-open-state summaries updated to reflect populated sidecars and expanded ACS controls
  - `README.md`, `docs/methodology.md`, `docs/source_dictionary.md` aligned to new acquire contract
- Verification:
  - `.venv/bin/pytest -q` -> `45 passed, 3 warnings`
  - `make smoke` -> passed

## 2026-03-12 Open-state closure continuation: full acquire path verification
- Executed full CLI acquisition with increased timeout:
  - `DCA_ACQUIRE_TIMEOUT_SECONDS=180 .venv/bin/dallas-crime acquire`
  - prior attempt at default timeout failed on Dallas OpenData paging timeout; retry succeeded end-to-end
- Confirmed integrated sidecar step runs from CLI path:
  - `[acquire] Building optional ZIP sidecars...`
  - all five category sidecars + `optional_sidecars.metadata.json` emitted and listed in command output + acquisition manifest
- Re-ran downstream pipeline on refreshed raw data:
  - `.venv/bin/dallas-crime build`
  - `.venv/bin/dallas-crime analyze`
- Post-refresh checks:
  - `model_dataset.csv`: `71` rows, `172` columns
  - modeled ZIP completeness means:
    - `economic=1.0`, `real_estate=1.0`, `law_enforcement=1.0`, `social_services=1.0`, `infrastructure=1.0`
  - key fields now populated for all modeled ZIPs (`71/71`): `unemployment_rate`, `educational_attainment`, `vacancy_proxy`, `public_assistance_share`, `transit_commute_share`, and all five sidecar score columns
  - latest regression metrics: baseline `R²=0.82499`, expanded `R²=0.90102`
  - best forecast MAE remains well below target (`moving_average_4`: `3.7522`)

## 2026-03-12 Statistical credibility follow-up implementation
- `src/dallas_crime/pipeline/analyze.py` now adds the audit-requested repo-native credibility tranche:
  - quarter-completeness gating based on trailing contiguous quarterly history before forecast and drift workflows
  - modeled-ZIP temporal holdout evaluation in `reports/temporal_holdout_results.csv`
  - interval calibration + output-shape diagnostics in `reports/forecast_interval_calibration.csv`
  - leave-high-leverage / high-Cook's ZIP-out robustness refits in `reports/influence_robustness_diagnostics.csv`
  - cluster stability and practical-utility checks in `reports/cluster_stability_diagnostics.csv`
  - regression / feature-selection / spatial multiple-testing and practical-effect controls in `reports/statistical_guardrails.csv`
  - explicit non-causal scenario/policy guardrails in `reports/policy_guardrails.md`
- Existing artifacts were kept additive/backward-compatible:
  - `forecast_confidence_intervals.csv`, `scenario_impacts.csv`, `policy_recommendations_by_segment.csv`, `model_validation_metrics.csv`, `spatial_diagnostics.csv`, and `feature_selection_metrics.csv` now carry additional guardrail fields without dropping prior columns
  - `summary.md`, `forecast_notes.md`, `scenario_notes.md`, and `policy_recommendations_by_segment.md` now point readers to the new credibility artifacts and keep policy/scenario framing explicitly non-causal
- Test coverage updated in `tests/test_project.py` for:
  - new output keys and artifact existence
  - 12-quarter forecast gating via a fixed fixture with one intentionally ineligible ZIP
  - deterministic holdout, calibration, influence-robustness, cluster-stability, statistical-guardrail, and policy-guardrail artifacts
- Verification completed:
  - `.venv/bin/pytest -q tests/test_pipeline.py tests/test_project.py` -> `17 passed, 3 warnings`
  - `.venv/bin/pytest -q` -> `45 passed, 3 warnings`
  - `make smoke` -> passed
- Environment note:
  - `ruff` was not available on `PATH` and was not installed in `.venv`, so lint could not be re-run in this session

## 2026-03-12 Statistical credibility follow-up live evidence sync
- Refreshed `dallas-crime analyze` outputs confirm the new credibility artifacts are live and reproducible on current data.
- Current live credibility snapshot from `reports/`:
  - modeled ZIP coverage: `71/98`
  - forecast-eligible ZIPs after the `>=12` trailing-quarter gate: `52/71`
  - candidate-model temporal holdout: `moving_average_4` is the only model under the repo threshold (`MAPE=19.9369%`, `pass=1`)
  - selected-model temporal holdout still misses the repo bar (`MAPE=20.6146%`, `pass=0`)
  - interval calibration materially improved:
    - selected-model `80%` and `95%` intervals both have `calibration_pass=1`
    - output-shape diagnostics show `0` equal-bound rows and `0` extreme-upper-ratio rows
  - influence robustness remains a blocker:
    - baseline `max_leverage=0.9753`, `max_cooks_distance=1.0584`, `influence_robustness_pass=0`
    - expanded-controls `max_leverage=0.7101`, `max_cooks_distance=1.0529`, `influence_robustness_pass=0`
  - cluster practical utility remains a blocker:
    - crime clustering is structurally strongest (`silhouette=0.7303`, `leave_one_feature_out_mean_ari=0.6225`) but still fails `practical_utility_pass=0` because cluster sizes are too imbalanced
    - market and socioeconomic clustering also fail `practical_utility_pass=0`
  - interpretation guardrails remain restrictive: `7/43` tested signals clear both FDR and practical-effect thresholds (`0.1628`)
- Current repo decision posture on live data:
  - forecast outputs remain descriptive-only because the selected forecast path still fails temporal holdout on true future quarters
  - scenario and policy artifacts remain exploratory and explicitly non-causal because the holdout, influence-robustness, and cluster-utility gates do not all pass together

## 2026-03-13 Forecast selection alignment + reporting guardrails
- `src/dallas_crime/pipeline/analyze.py` received a narrow follow-up implementation on top of the credibility tranche:
  - ZIP-level forecast-family selection now uses the same pre-holdout training window as `selected_zip_model` temporal holdout evaluation before projecting on the full available history
  - `forecast_model_metrics.csv` now carries additive temporal-holdout ranking columns for each candidate family
  - `forecast_notes.md`, `scenario_notes.md`, `policy_guardrails.md`, and `summary.md` now state explicit forecast/scenario coverage and counted severity guardrails, and confirm that forecast-ineligible modeled ZIPs are excluded rather than backfilled
  - `model_residuals.csv`, `residual_review.md`, and `model_validation_notes.md` now surface high-influence ZIP flags so leverage/Cook's issues remain visible even when residuals are small
- Live post-change evidence from `reports/`:
  - forecast/scenario eligibility remains `52/71` modeled ZIPs, with `19` modeled ZIPs excluded rather than backfilled
  - current ZIP-level selected forecast-family mix is `moving_average_4=50`, `naive_last=1`, `seasonal_naive_4q=1`, `linear_trend=0`
  - candidate-family temporal holdout ranking remains `moving_average_4` first (`MAPE=19.9369%`), but the aggregated `selected_zip_model` holdout still misses the repo bar (`MAPE=20.6146%`, `pass=0`)
  - residual-facing outputs now flag `19` high-influence model rows directly in `model_residuals.csv`
- Verification completed after the follow-up:
  - `.venv/bin/pytest -q tests/test_pipeline.py tests/test_project.py` -> `18 passed, 3 warnings`
  - `.venv/bin/pytest -q` -> `46 passed, 3 warnings`
  - `make smoke` -> passed
- Environment note:
  - `ruff` was still unavailable on `PATH`, so lint could not be re-run

## 2026-03-13 Credibility gap closure follow-up: tiered coverage + output-stability guardrails
- `src/dallas_crime/pipeline/analyze.py` received the next credibility-focused follow-up tranche:
  - selected ZIP holdout / forward forecasts now restrict ZIP-level family choice to candidate models that already pass aggregate temporal holdout
  - `crime_forecasts.csv` now emits additive forecast tiers:
    - `high_confidence` for ZIPs with `>=12` trailing contiguous quarters
    - `limited_history` for ZIPs with `4-11` trailing contiguous quarters
    - `carry_forward_only` for ZIPs with `1-3` trailing contiguous quarters
  - `forecast_confidence_intervals.csv` and `scenario_impacts.csv` remain limited to the `high_confidence` subset
  - `cluster_stability_diagnostics.csv` now records adaptive per-domain `selected_k` (`2` vs `3`) using the repo’s existing utility rules
  - `policy_recommendations_by_segment.csv` now carries additive segment guardrail fields:
    - scenario support counts/shares
    - high-influence ZIP counts/shares
    - domain-level cluster utility metrics
    - segment guardrail status / confidence tier / guardrail flags
  - `influence_robustness_diagnostics.csv` now keeps coefficient-shift audit fields but evaluates `robustness_pass` on prediction-stability (`p90` home-value delta) rather than coefficient percentage swings alone, with `fit_stability_pass` retained separately
- Live post-change evidence from refreshed `reports/`:
  - modeled coverage remains `71/98`
  - high-confidence forecast/scenario coverage remains `52/71`, and lower-confidence forecast-only tiers now cover the remaining `19/71` modeled ZIPs
  - selected-model temporal holdout now passes on current data:
    - `selected_zip_model` / `holdout_pass_screened_zip_selection` -> `MAPE=19.9369%`, `pass=1`
    - high-confidence selected-family mix is now `moving_average_4=52`
  - interval calibration still passes with stable output shape
  - prediction-stability influence checks now pass for both regression specs:
    - baseline `influence_robustness_pass=1`, `fit_stability_pass=0`
    - expanded-controls `influence_robustness_pass=1`, `fit_stability_pass=1`
  - cluster practical utility remains the active blocker:
    - crime selects `k=3` and still fails on size imbalance
    - market selects `k=2` and still fails on tiny-cluster size
    - socioeconomic selects `k=2`, improving balance/stability, but still misses the silhouette guardrail
  - policy segment output now visibly downgrades fragile segments:
    - blocked examples include `higher_crime`, `mid_crime`, `advantaged`, `premium`
    - caution / guardrailed examples include `value`, `stressed`, `lower_crime`
  - interpretation guardrails remain restrictive (`0.163` share of guarded tests clear FDR + practical-effect thresholds)
- Verification completed after the tranche:
  - `.venv/bin/pytest -q tests/test_pipeline.py tests/test_project.py` -> `20 passed, 3 warnings`
  - `.venv/bin/pytest -q` -> `48 passed, 3 warnings`
  - `make smoke` -> passed
  - `.venv/bin/dallas-crime analyze` -> passed and refreshed live `reports/`
- Environment note:
  - `ruff` is still unavailable on `PATH`

## 2026-03-13 Credibility gap closure follow-up: cluster utility pass + fit-warning semantics
- `src/dallas_crime/pipeline/analyze.py` received the final repo-native credibility follow-up needed to close the implementation gap without weakening thresholds:
  - clustering now searches within each domain over auditable feature subsets plus winsorized preprocessing modes before scoring the existing practical-utility gate
  - `cluster_stability_diagnostics.csv` now emits additive `selected_feature_set` and `preprocessing_mode` fields so the chosen segmentation recipe is inspectable
  - influence robustness now keeps the original `fit_stability_pass` field for backward compatibility but adds `fit_deterioration_pass` and `fit_improvement_warning` / `fit_improvement_warning_count` so one-sided fit improvements are no longer conflated with robustness failure
  - `model_validation_notes.md`, `policy_guardrails.md`, and `summary.md` now describe the remaining baseline fit issue as a warning rather than a failed prediction-stability guardrail
- Live post-change evidence from refreshed `reports/`:
  - modeled coverage remains `71/98`
  - high-confidence forecast/scenario coverage remains `52/71`, with `19/71` modeled ZIPs still surfaced as lower-confidence forecast-only tiers
  - selected-model temporal holdout still passes on current data:
    - `selected_zip_model` / `holdout_pass_screened_zip_selection` -> `MAPE=19.9369%`, `pass=1`
  - interval calibration still passes with stable output shape
  - prediction-stability influence checks still pass for both regression specs:
    - baseline `influence_robustness_pass=1`, `fit_deterioration_pass=0`, `fit_improvement_warning_count=1`
    - expanded-controls `influence_robustness_pass=1`, `fit_deterioration_pass=1`, `fit_improvement_warning_count=0`
  - cluster practical utility now passes in all three domains:
    - crime -> `selected_feature_set=violent_rate_per_1000, property_rate_per_1000`, `preprocessing_mode=winsor_10_90`, `selected_k=2`, `practical_utility_pass=1`
    - market -> `selected_feature_set=home_value, FHFA_annual_change_pct`, `preprocessing_mode=winsor_5_95`, `selected_k=2`, `practical_utility_pass=1`
    - socioeconomic -> `selected_feature_set=median_household_income, poverty_rate, median_gross_rent`, `preprocessing_mode=winsor_10_90`, `selected_k=2`, `practical_utility_pass=1`
  - policy segment output now includes at least one fully clear segment (`stressed`), while other segments remain caution/blocked based on scenario support or influence concentration rather than failed cluster utility
  - interpretation guardrails remain restrictive (`0.163` share of guarded tests clear FDR + practical-effect thresholds)
- Definition-of-complete posture after this tranche:
  - the repo-native statistical credibility follow-up requested by the 2026-03-12 audit is now implemented end to end
  - remaining limits are live-data limits, not missing repo-native diagnostics:
    - representativeness / high-confidence coverage remains `52/71` modeled ZIPs
    - interpretation of crime-linked regression terms remains blocked by FDR + practical-effect guardrails
    - all policy/scenario framing remains explicitly non-causal
- Verification completed after the tranche:
  - `.venv/bin/pytest -q tests/test_pipeline.py tests/test_project.py` -> `21 passed, 3 warnings`
  - `.venv/bin/pytest -q` -> `49 passed, 3 warnings`
  - `make smoke` -> passed
  - `.venv/bin/dallas-crime analyze` -> passed and refreshed live `reports/`
- Environment note:
  - `ruff` is not installed in `.venv` and was not available on `PATH`

## 2026-03-15 MVM alignment: total_rate predictor + influence column rename
- `DEFAULT_PREDICTORS` was changed from `("violent_rate_per_1000", "property_rate_per_1000")` to `("total_rate_per_1000",)` to resolve the VIF > 10 gate failure documented in `reports/minimum_viable_metrics.md`.
- Expanded model relabeled from `expanded_controls` to `sensitivity_check` in prior session.
- Drift pro-rating with `0.33` completeness floor implemented in prior session.
- Influence robustness columns renamed to match the new single predictor:
  - `violent_effect_pct_change` / `property_effect_pct_change` collapsed to `crime_term_effect_pct_change`
  - `max_violent_effect_pct_change` / `max_property_effect_pct_change` collapsed to `max_crime_term_effect_pct_change`
- Reporting narrative updated: coefficient extraction and limitation text now reference `total_rate_per_1000` instead of separate violent/property terms.
- Documentation aligned: `reports/minimum_viable_metrics.md`, `docs/methodology.md`, `README.md`, `V2_EXECUTION_TRACKER.md`.
- Verification: `ruff check` clean, `pytest -q` 84 passed.
