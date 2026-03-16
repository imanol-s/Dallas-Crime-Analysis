# CLAUDE.md

Read `AGENTS.md` before making any changes — it is the authoritative guide for this repository.

## Parallel Code Review Pipeline

This project has three specialist agents mapped to its domain layers. When running a parallel review, use the Task tool to spawn all three concurrently.

### Agent: Data Engineer

**Scope:** `src/dallas_crime/acquire/` — acquisition layer (crime.py, census.py, housing.py, utils.py)
**Gate:** `ruff check src/dallas_crime/acquire/ tests/test_acquire.py` → `pytest -q tests/test_acquire.py`
**Fix loop:** up to 5 cycles. Report: status, files changed, any breaking changes to source contracts or `AcquisitionError` handling.

### Agent: Data Analyst

**Scope:** `src/dallas_crime/pipeline/build.py`, `src/dallas_crime/config.py` — transformations and settings
**Gate:** `ruff check src/dallas_crime/pipeline/build.py src/dallas_crime/config.py tests/test_pipeline.py` → `pytest -q tests/test_pipeline.py`
**Fix loop:** up to 5 cycles. Never remove existing `model_dataset.csv` columns — they are additive only.

### Agent: Data Scientist

**Scope:** `src/dallas_crime/pipeline/analyze/` — regressions, VIF, plots, reports (core.py, forecast.py, reporting.py, segmentation.py, spatial.py)
**Gate:** `ruff check src/dallas_crime/pipeline/analyze/ tests/test_project.py` → `pytest -q tests/test_project.py`
**Fix loop:** up to 5 cycles. HC3 robust standard errors (`cov_type="HC3"`) must remain on all regression fits.

### Orchestrator steps (after all 3 agents report green)

1. Run `make smoke` — must pass with no network access.
2. Run full `pytest -q` — must show `84 passed, 1 warning`.
3. Run `ruff check src/ tests/` — zero errors.
4. Commit with a summary of all agent changes.

## Project Type

This is a **data analytics / data science project**. Procedural and functional Python is the correct style. Do not over-engineer with OOP patterns, premature abstractions, or framework migrations. Verbosity in feature engineering is acceptable when it aids readability and auditability.

## Environment

- Always activate the project venv before running commands: use `uv run` or `.venv/bin/python`
- Lint: `uv tool run ruff check src/ tests/`
- Test: `uv run pytest -q`
- Smoke: `make smoke`

## Data Quality Principles

All pipeline outputs must satisfy these three principles. Verify on every data refresh.
Use the `/data-observability-gate` skill to classify and validate any data-layer change.

### Accuracy
Values must be factually correct, with no duplicates or wrong types.
**Trigger tier:** `full` for any schema or derived-field change.

| Check | Skill validation | Code guard |
|---|---|---|
| No duplicate ZIPs | `check_referential_integrity` (self-join) | `_dedupe_zip_rows()` in build.py |
| Numeric types only | `check_schema` (dtype conformance) | `pd.to_numeric(errors="coerce")` in build.py |
| Rates ≥ 0, pop > 0 | `check_statistical_bounds` (min: 0) | `np.where(pop > 0, rate, NaN)` in build.py |
| VIF < 10 | `check_statistical_bounds` (max: 10) | `variance_inflation_factor()` in core.py |

### Completeness
All required fields must be populated.
**Trigger tier:** `full` for join, filter, or null-handling changes.

| Check | Skill validation | Code guard |
|---|---|---|
| Cell-level ≤ 5% null | `check_null_thresholds` (overall) | `warnings.warn` in `build_model_dataset()` |
| Regression columns ≤ 30% null | `check_null_thresholds` (per-column) | `warnings.warn` in `run_zip_regression()` |
| ZIP-level join integrity | `check_referential_integrity` | `how="inner"` on crime+housing+ACS |
| Per-ZIP completeness scores | `check_schema` (column exists) | `build_source_completeness_scores()` |

### Timeliness
Data must be available when downstream processes need it.
**Trigger tier:** `full` for filter, aggregation, or temporal logic changes.

| Check | Skill validation | Code guard |
|---|---|---|
| Pro-rate partial quarters (floor 0.33) | `check_statistical_bounds` | `DRIFT_MIN_COMPLETENESS` in forecast.py |
| History depth ≥ 8 quarters for lag-4 | `check_null_thresholds` (lag cols) | `crime_history_sufficient_depth` flag in build.py |
| Forecast ≥ 12 quarters | `check_row_count` (eligible ZIPs) | `FORECAST_HISTORY_MIN_QUARTERS` in core.py |
| ACS snapshots ≥ 2 vintages | `check_null_thresholds` (trend cols) | `len(metric_frame) >= 2` in build.py |

### Validation contract: `build_model_dataset`

When modifying build.py data assembly, fill out a validation contract per the
`data-observability-gate` skill template. Key thresholds:

```
null_thresholds:
  zip: 0.00
  home_value: 0.05
  total_rate_per_1000: 0.05
  median_household_income: 0.10
  poverty_rate: 0.10
row_count:
  min: 50
  max: 100
statistical_bounds:
  home_value: {min: 50000, max: 3000000}       # Highland Park ZIPs reach ~$2M
  total_rate_per_1000: {min: 0, max: 2500}     # small-pop downtown ZIPs inflate rate
  population: {min: 1}
```
