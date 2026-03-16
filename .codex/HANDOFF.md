# Handoff

## Current state

- Branch: `Develpoment/1.1`
- Foundation refresh is complete.
- The repo has been converted from a one-off RMarkdown project into a Python-first package with a CLI, acquisition layer, ZIP-level build pipeline, regression analysis, tests, and updated docs.
- Legacy R artifacts remain in the repo for reference only. The active implementation lives under `src/dallas_crime/`.

## Delivered in this phase

- Added a `uv`-managed Python project in `pyproject.toml`.
- Added CLI commands in `src/dallas_crime/cli.py`:
  - `dallas-crime acquire`
  - `dallas-crime build`
  - `dallas-crime analyze`
  - `dallas-crime show-config`
- Added acquisition modules for:
  - Dallas OpenData crime pulls
  - Census ACS ZCTA pulls
  - Firecrawl/Zillow housing discovery and parsing
- Added ZIP-level build and analysis modules for:
  - ZIP normalization
  - crime aggregation and rates per 1,000
  - ACS and housing merge into a model dataset
  - robust OLS regression
  - report artifact generation
- Added fixture-backed tests under `tests/`.
- Rewrote `README.md`, added `.env.example`, and added `.gitignore` coverage for runtime outputs and local caches.

## Verified

- Environment creation:
  - `uv venv`
  - `uv pip install -e ".[dev]"`
- Test suite:
  - `UV_CACHE_DIR=.uv-cache .venv/bin/pytest -q`
  - Result: `13 passed`
- Bounded live smoke run:
  - `DCA_CRIME_LOOKBACK_DAYS=14`
  - `DCA_MAX_HOUSING_ZIPS=10`
  - Commands run:
    - `dallas-crime acquire`
    - `dallas-crime build`
    - `dallas-crime analyze`
- Smoke outputs generated:
  - `data/raw/crime_records.csv`
  - `data/raw/acs_zcta.csv`
  - `data/raw/housing_market.csv`
  - `data/processed/crime_zip.csv`
  - `data/processed/housing_zip.csv`
  - `data/processed/acs_controls.csv`
  - `data/processed/model_dataset.csv`
  - `reports/regression_coefficients.csv`
  - `reports/regression_metrics.csv`
  - `reports/home_value_vs_total_crime.png`
  - `reports/summary.md`

## Operating notes

- Default crime source: `https://www.dallasopendata.com/resource/qv6i-rri7.json`
- Default ACS year: `2024`
- Firecrawl is used through the CLI, not MCP.
- Full live housing acquisition is slow because it currently searches Zillow per ZIP.
- Use `DCA_MAX_HOUSING_ZIPS` for bounded smoke runs.
- Runtime outputs under `data/`, `reports/`, `.firecrawl/`, `.matplotlib/`, `.venv/`, and `.uv-cache/` are intended to remain untracked.

## Remaining project plan

### Phase 2: Data hardening and acquisition coverage

Goal:
- Turn the current working acquisition layer into a reliable full-refresh data pipeline with measurable source coverage and failure handling.

Work:
- Add structured source metadata outputs for each raw dataset:
  - retrieval timestamp
  - source URL
  - query/filter parameters
  - row counts
- Add explicit logging/progress output to `acquire` so long Firecrawl runs are observable.
- Add retry/backoff and clearer error handling for Dallas OpenData, Census, and Firecrawl failures.
- Improve housing acquisition coverage:
  - track requested ZIPs vs returned ZIPs
  - write a coverage report
  - decide whether to keep Firecrawl search-per-ZIP or move to a faster discovery strategy
- Filter ACS rows to the relevant Dallas-area ZIPs before downstream joins.
- Add a source dictionary documenting every field used in the processed model dataset.

Acceptance criteria:
- A full live `dallas-crime acquire` run completes without manual edits.
- Acquisition writes a machine-readable metadata/coverage artifact for all sources.
- Housing output includes an explicit requested-vs-received ZIP coverage summary.
- Failures surface as actionable CLI errors instead of silent hangs.
- The raw acquisition step is reproducible from committed code and environment variables only.

### Phase 3: Model dataset quality and methodology

Goal:
- Make the analytical dataset defensible enough for a real project deliverable rather than a smoke-tested prototype.

Work:
- Define the exact target ZIP universe for the study.
- Add dataset QA checks:
  - duplicate ZIP detection
  - missingness report
  - impossible/negative value checks
  - outlier review
- Revisit the crime taxonomy and decide whether the current violent/property mapping is sufficient.
- Add additional control variables if needed:
  - rent burden
  - vacancy proxy
  - educational attainment
  - housing tenure mix
- Add at least two model specifications:
  - baseline model
  - expanded control model
- Add model diagnostics:
  - sample size by model
  - coefficient table export
  - residual review
  - multicollinearity check or documented rationale if skipped

Acceptance criteria:
- The processed model dataset has one row per ZIP with no duplicate keys.
- Required model columns have documented missingness and handling rules.
- At least two clearly named model specifications run from the same processed dataset.
- Analysis outputs include regression coefficients, model metrics, and a diagnostic summary.
- Method choices and limitations are documented in the repo.

### Phase 4: Reporting and presentation layer

Goal:
- Ship a presentation-quality deliverable that explains the refreshed project, not just the raw outputs.

Work:
- Create a narrative notebook or report that rebuilds from pipeline outputs.
- Add publication-quality visuals:
  - crime-rate vs home-value scatter
  - top/bottom ZIP comparison table
  - at least one map or geography-aware view
  - model summary table
- Add a methodology section:
  - data windows
  - geography choice
  - variable definitions
  - limitations
- Add a findings section focused on interpretation, not just coefficients.
- Decide whether to keep the deliverable as Markdown/notebook only or add a dashboard later.

Acceptance criteria:
- One documented command path regenerates the final report from raw or processed data.
- The report includes sources, methods, findings, and limitations.
- Visuals are saved as stable artifacts and referenced in the report.
- The report can be handed to a non-technical reader without needing code context.

### Phase 5: Productionization and maintenance

Goal:
- Make the project maintainable for repeat refreshes instead of a one-time rebuild.

Work:
- Add CI for:
  - dependency install
  - tests
  - import/CLI smoke checks
- Add a lightweight Makefile or task runner aliases if useful.
- Add versioned release notes or a changelog entry for major refreshes.
- Decide what to do with the legacy R files:
  - retain in repo with a legacy note
  - move to `legacy/`
  - archive elsewhere
- Document the standard refresh workflow for future runs.

Acceptance criteria:
- CI passes on the branch with the same commands used locally.
- A new collaborator can set up the environment and run the documented workflow from README plus `.env.example`.
- Legacy artifacts have an explicit policy and are no longer ambiguous project entrypoints.
- Future refresh steps are documented well enough that no reverse-engineering is needed.

## Recommended execution order

1. Finish Phase 2 first. The current biggest risk is not code correctness; it is live-data reliability and housing coverage.
2. Do Phase 3 next. The current model pipeline works, but the methodology still needs hardening before the conclusions should be treated as final.
3. Do Phase 4 after the dataset and model choices are stable. Otherwise the report will be rewritten repeatedly.
4. Finish with Phase 5 once the workflow and deliverables have settled.

## Immediate next action

- Start Phase 2 by adding acquisition metadata and coverage artifacts, then rerun a full live acquisition without the `DCA_MAX_HOUSING_ZIPS` cap to measure real ZIP coverage and runtime.
