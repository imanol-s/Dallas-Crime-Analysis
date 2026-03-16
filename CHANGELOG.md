# Changelog

## 2026-03-11

- Added official Census ACS bulk-table fallback when the API times out.
- Expanded current housing acquisition to use Zillow first, then Realtor and Redfin ZIP pages,
  plus Realtor ZIP bulk inventory/history feeds for structured features.
- Added a historical housing panel (`2000-2025`) from Realtor monthly ZIP history and FHFA ZIP5 HPI.
- Added ZIP-universe guardrails requiring a minimum incident count per ZIP before modeling.
- Added geography-aware reporting artifacts and refreshed methodology/refresh workflow docs.
