# TODOs

## ~~Comprehensive error-scenario test suite~~ ✅ DONE (2026-03-15)
**Priority:** Medium | **Effort:** ~3h | **Added:** 2026-03-15 | **Completed:** 2026-03-15

Added 18 error-scenario tests in `tests/test_error_scenarios.py` covering:
- HTTPError (404, 500) scenarios for crime and census acquire modules
- Malformed JSON payloads from Socrata API
- Non-array JSON responses
- Census payload with header-only (no data rows)
- subprocess.TimeoutExpired in Firecrawl
- run_with_retry exhaustion behavior
- Missing required regression columns
- Insufficient rows for regression
- Missing expanded-control candidates
- Empty/missing-column temporal inputs
- Empty forecast artifacts
- VIF with single-row model
- NaN dates in crime aggregation
- All-null home values in housing features
