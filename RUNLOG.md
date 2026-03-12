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
