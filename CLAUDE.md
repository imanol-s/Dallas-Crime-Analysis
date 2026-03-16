# CLAUDE.md

Authoritative guide for this repository. Read this before making any changes.

---

## Project Overview

A reproducible ZIP-level crime and housing analysis pipeline for the Dallas metro area.
Python package at `src/dallas_crime/`, exposed via the `dallas-crime` CLI.
Stages: `acquire` → `build` → `analyze`.

Runtime outputs (`data/`, `reports/`, `.firecrawl/`, `.matplotlib/`) are **untracked** — never commit them.

Roadmap execution log note: `task.md` is the line-by-line tracker; read it at session start with `RUNLOG.md`.

### Where Current Data Lives

| What | Location | Notes |
|------|----------|-------|
| Raw data | `data/raw/` (59 MB) | Crime, housing, ACS, sidecars |
| Processed datasets | `data/processed/` (2.8 MB) | `model_dataset.csv` is the primary input |
| Source code | `src/dallas_crime/` | acquire → pipeline → analyze |
| Analysis reports | `reports/` | Regression, forecasts, clusters, figures |
| Quick EDA visuals | `analysis_output/` | Exploratory plots |
| Tests | `tests/` | Unit + fixtures |
| Scripts | `scripts/` | dq_metrics, smoke fixture gen |

---

## Build / Test / Lint Commands

```bash
# Environment setup (run once)
uv venv && pip install -e ".[dev]"

# Full pipeline
make acquire       # fetch crime, ACS, housing raw data
make build         # transform raw → model_dataset.csv
make analyze       # run regressions → reports/

# Individual CLI commands (same as make targets)
.venv/bin/dallas-crime acquire
.venv/bin/dallas-crime build
.venv/bin/dallas-crime analyze
.venv/bin/dallas-crime show-config

# Tests
make test               # runs: pytest -q
pytest -q               # all tests
pytest -q tests/test_pipeline.py                     # single file
pytest -q tests/test_pipeline.py::test_normalize_zip_variants  # single test
pytest -q -k "keyword"  # filter by name

# Smoke test (no network, no API keys required)
make smoke
# or manually:
TMP=$(mktemp -d)
python scripts/create_smoke_inputs.py "$TMP"
.venv/bin/dallas-crime build --project-root "$TMP"
.venv/bin/dallas-crime analyze --project-root "$TMP"

# Lint (ruff — configured in pyproject.toml)
ruff check src/ tests/
ruff format src/ tests/
```

CI runs on every push/PR: `pytest -q` then `make smoke` (see `.github/workflows/ci.yml`).

---

## Project Structure

```
src/dallas_crime/
  cli.py              # Typer CLI entry point; thin dispatch only
  config.py           # Settings dataclass; no side effects
  acquire/
    crime.py          # Dallas OpenData crime fetch
    census.py         # ACS ZCTA fetch with bulk-table fallback
    housing.py        # Multi-source housing (Zillow → Realtor → Redfin)
    utils.py          # Shared acquisition helpers, AcquisitionError
  pipeline/
    build.py          # All transformations; build_all(settings) orchestrator
    analyze.py        # Regression, VIF, plots, reports; run_analysis(settings)
tests/
  test_pipeline.py    # 36 unit/integration tests for build + analyze
  test_project.py     # End-to-end run_analysis() artifact tests
  test_acquire.py     # Acquisition layer unit tests
scripts/
  create_smoke_inputs.py  # Generates minimal raw CSVs for offline smoke tests
data/raw/
  dfw_zip_enrichment.csv  # Stable enrichment sidecar; joined in build_all()
```

---

## Code Style

### Python version and imports

- **Requires Python ≥ 3.11**; target `py311` for ruff.
- Every module starts with `from __future__ import annotations`.
- Standard library imports first, then third-party (`numpy`, `pandas`, `statsmodels`, `matplotlib`), then intra-package.
- Use `TYPE_CHECKING` guard for imports that only serve type annotations to avoid circular imports:
  ```python
  from typing import TYPE_CHECKING
  if TYPE_CHECKING:
      from dallas_crime.config import Settings
  ```

### Formatting

- Line length: **100 characters** (ruff enforced).
- Ruff is the sole formatter and linter — no Black, no isort separately.
- Trailing commas in multi-line collections.

### Type annotations

- All public functions must have full parameter and return type annotations.
- Use `tuple[str, ...]` not `Tuple[str, ...]`; use `list[str]` not `List[str]` (PEP 585).
- `pd.DataFrame`, `pd.Series`, `Path` are preferred over `Any`.
- `@dataclass(slots=True)` for result/payload objects (see `RegressionResult`).

### Naming conventions

- Modules, variables, functions: `snake_case`.
- Private helpers: prefixed with `_` (e.g., `_coerce_model_columns`, `_write_scatter_plot`).
- Public-facing module-level constants: `UPPER_SNAKE_CASE` (e.g., `DEFAULT_CONTROLS`, `MISSING_STRINGS`).
- Immutable constant sets: use `frozenset` (e.g., `DEFAULT_VIOLENT_LABELS`).
- Tuple constants preferred over list for fixed sequences of control/predictor names.

### Keyword-only arguments

Multi-parameter public functions must use `*` to enforce keyword-only call sites:
```python
def run_zip_regression(
    model_df: pd.DataFrame,
    *,
    dependent: str = "log_home_value",
    predictors: tuple[str, ...] = DEFAULT_PREDICTORS,
    controls: tuple[str, ...] = DEFAULT_CONTROLS,
    model_label: str = "baseline",
) -> RegressionResult:
```

### Error handling

- Raise `KeyError` for missing required DataFrame columns; include the column name(s) in the message.
- Raise `ValueError` for invalid inputs or insufficient rows.
- Raise `FileNotFoundError` for missing required input files.
- Use `AcquisitionError` (from `dallas_crime.acquire.utils`) for acquisition-layer failures.
- Numeric coercion: always use `pd.to_numeric(..., errors="coerce")` and `pd.to_datetime(..., errors="coerce")` — never raise on bad data; NaN-out instead.
- In VIF/stats blocks, catch `(TypeError, ValueError, np.linalg.LinAlgError)` together and record a note rather than crashing.

### Pandas conventions

- After any dedup or sort, call `.reset_index(drop=True)`.
- Use `np.where(condition, a, b)` for vectorized conditional assignment instead of pandas `.where()`.
- Use `_safe_divide(numerator, denominator)` helper (defined in `build.py`) for ratio columns — avoids ZeroDivisionError.
- Always `.copy()` a frame before mutating it (avoids `SettingWithCopyWarning`).
- Keep I/O out of transformation functions: `build.py` helpers are pure transforms; `build_all()` does all file reads/writes.

### Matplotlib

- Always call `matplotlib.use("Agg")` before importing `pyplot` for headless safety.
- Always call `plt.close(fig)` after `fig.savefig(...)`.
- Use `os.environ.setdefault("MPLCONFIGDIR", ...)` to redirect the config dir.

---

## Architecture Rules

- **`cli.py` is thin.** No business logic; dispatches to `build_all(settings)` and `run_analysis(settings)` only.
- **`config.py` is pure.** `Settings` is a dataclass with no side effects; loaded once and passed down.
- **Transformations are I/O-free.** `build.py` helpers take DataFrames and return DataFrames; `build_all()` owns all file I/O.
- **Enrichment sidecar.** `data/raw/dfw_zip_enrichment.csv` is stable raw enrichment; `build_all()` left-joins it after `build_model_dataset()`. Idempotent — no-op if file absent. Never discard it.
- **Expanded controls are selected dynamically.** `_select_expanded_controls()` in `analyze.py` iterates `EXPANDED_CONTROL_CANDIDATES` and accepts a candidate only if it leaves ≥ (n_columns + 1) complete rows. Do not hard-code the expanded formula.
- **Regression uses HC3 robust standard errors.** Always `smf.ols(...).fit(cov_type="HC3")`.
- **No secrets in source.** API keys via environment variables only (see `.env.example`).

---

## Testing Guidelines

- Tests live in `tests/`; run with `pytest -q`.
- Use `tmp_path` (pytest fixture) for integration tests that write files.
- Use `unittest.mock.patch` to stub network calls in acquisition tests.
- Always assert DataFrame shape and specific column values — do not just assert "not empty".
- A single test can be run with: `pytest -q tests/<file>.py::<TestClass>::<test_method>` or `pytest -q tests/<file>.py::<test_function>`.
- The smoke script (`scripts/create_smoke_inputs.py`) generates offline-safe minimal inputs; CI depends on it. Do not break its column contracts.

---

## Key Invariants (do not break)

- `pytest -q` must pass (`36 passed, 1 warning`) before any commit.
- `make smoke` must pass with no network access.
- `data/raw/dfw_zip_enrichment.csv` must remain intact (70 rows, 11 enrichment columns).
- Regression model sample sizes: baseline n=70, expanded n=61 (current; will vary after re-acquire).
- `model_dataset.csv` columns are additive — never remove an existing column.
- Branch: `Develpoment/1.1` (note the typo — do not rename it).

---

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
