import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from urllib.error import URLError

import pandas as pd

from dallas_crime.acquire.census import (
    DEFAULT_VARIABLES,
    AcquisitionError,
    CensusRequest,
    build_census_url,
    fetch_census_zcta_snapshot_data,
    fetch_census_zcta_data,
    normalize_census_payload,
)
from dallas_crime.acquire.crime import (
    DallasCrimeSourceConfig,
    build_crime_url,
    fetch_crime_history_dataset,
    fetch_crime_dataset,
    fetch_crime_payload,
    normalize_crime_records,
)
from dallas_crime.acquire.housing import (
    FirecrawlScrapeRequest,
    FirecrawlSearchRequest,
    _load_local_realtor_history_summary,
    _merge_housing_records,
    build_firecrawl_scrape_command,
    fetch_fhfa_zip5_history,
    fetch_housing_history_dataset,
    build_firecrawl_search_command,
    fetch_realtor_zip_history,
    fetch_realtor_zip_history_panel,
    fetch_housing_dataset,
    normalize_firecrawl_documents,
    parse_firecrawl_search_results,
    run_firecrawl_command,
)
from dallas_crime.acquire.sidecars import fetch_optional_zip_sidecars
from dallas_crime.config import Settings


FIXTURES = Path(__file__).parent / "fixtures" / "acquire"


def load_fixture(name: str):
    return json.loads((FIXTURES / name).read_text())


class AcquireTests(unittest.TestCase):
    def test_build_crime_url_includes_paging_and_filters(self):
        config = DallasCrimeSourceConfig(
            dataset_url="https://www.dallasopendata.com/resource/crime.json",
            where_clause="date1 >= '2025-01-01T00:00:00'",
            app_token="secret-token",
        )

        url = build_crime_url(config, offset=1000)

        self.assertIn("crime.json?", url)
        self.assertIn("%24limit=50000", url)
        self.assertIn("%24offset=1000", url)
        self.assertIn("%24%24app_token=secret-token", url)
        self.assertIn("date1+%3E%3D+%272025-01-01T00%3A00%3A00%27", url)

    def test_normalize_crime_records_shapes_expected_columns(self):
        payload = load_fixture("crime_payload.json")

        frame = normalize_crime_records(payload)

        self.assertEqual(
            frame.to_dict(orient="records"),
            [
                {
                    "incident_id": "12345",
                    "reported_at": frame.iloc[0]["reported_at"],
                    "offense": "AGGRAVATED ASSAULT",
                    "offense_family": "violent",
                    "zip": "75201",
                    "latitude": 32.7811,
                    "longitude": -96.7989,
                },
                {
                    "incident_id": "67890",
                    "reported_at": frame.iloc[1]["reported_at"],
                    "offense": "BURGLARY OF HABITATION",
                    "offense_family": "property",
                    "zip": "75214",
                    "latitude": 32.8142,
                    "longitude": -96.7485,
                },
            ],
        )
        self.assertEqual(str(frame.iloc[0]["reported_at"]), "2025-01-15 03:30:00+00:00")

    def test_build_census_url_uses_requested_dataset(self):
        request = CensusRequest(
            year=2023,
            dataset="acs/acs5",
            variables=("B01003_001E", "B19013_001E"),
            api_key="api-key",
        )

        url = build_census_url(request)

        self.assertTrue(url.startswith("https://api.census.gov/data/2023/acs/acs5?"))
        self.assertIn("get=B01003_001E%2CB19013_001E", url)
        self.assertIn("for=zip+code+tabulation+area%3A%2A", url)
        self.assertIn("key=api-key", url)

    def test_normalize_census_payload_coerces_numeric_columns(self):
        payload = load_fixture("census_payload.json")

        frame = normalize_census_payload(
            payload,
            rename_map={
                "B01003_001E": "population",
                "B19013_001E": "median_household_income",
            },
        )

        self.assertEqual(
            frame.to_dict(orient="records"),
            [
                {"population": 12000, "median_household_income": 98500, "zip": "75201"},
                {"population": 34000, "median_household_income": 76500, "zip": "75214"},
            ],
        )

    def test_build_firecrawl_commands_match_cli_shape(self):
        search_command = build_firecrawl_search_command(
            FirecrawlSearchRequest(
                query="Dallas 75214 home values site:zillow.com",
                limit=3,
            )
        )
        scrape_command = build_firecrawl_scrape_command(
            FirecrawlScrapeRequest(
                urls=("https://www.zillow.com/home-values/75214/",),
            )
        )

        self.assertEqual(
            search_command[:3],
            ["firecrawl", "search", "Dallas 75214 home values site:zillow.com"],
        )
        self.assertIn("--scrape", search_command)
        self.assertIn("--json", search_command)
        self.assertEqual(
            scrape_command[:3],
            ["firecrawl", "scrape", "https://www.zillow.com/home-values/75214/"],
        )
        self.assertIn("--format", scrape_command)
        self.assertIn("--wait-for", scrape_command)

    def test_parse_firecrawl_search_results_normalizes_common_fields(self):
        payload = load_fixture("firecrawl_search.json")

        records = parse_firecrawl_search_results(payload)

        self.assertEqual(
            records,
            [
                {
                    "title": "75214 market trends",
                    "url": "https://www.zillow.com/home-values/75214/",
                    "description": "The average 75214 home value is $898,055.",
                }
            ],
        )

    def test_normalize_firecrawl_documents_extracts_housing_metrics(self):
        payload = load_fixture("firecrawl_scrape.json")

        frame = normalize_firecrawl_documents(payload)

        self.assertEqual(
            frame.to_dict(orient="records"),
            [
                {
                    "zip": "75214",
                    "source_url": "https://www.zillow.com/home-values/75214/",
                    "home_value": 898055.0,
                    "median_rent": 2450.0,
                    "annual_change_pct": 4.4,
                    "as_of_date": "2026-01-31",
                    "source": "zillow",
                    "metric_label": "typical_home_value",
                    "supplemental_sources": None,
                }
            ],
        )

    def test_normalize_firecrawl_documents_extracts_realtor_metrics(self):
        payload = {
            "markdown": """
## 75209, TX market summary
Key indicators as of January 2026
| Metric | Zipwide | 1Y Change | 3Y Change |
| --- | --- | --- | --- |
| Median home $ | $1,472,500 | 4.12% | 26.30% |
| Median rent | $1,919/mo | 7.57% | 2.26% |
""",
            "metadata": {"sourceURL": "https://www.realtor.com/local/market/texas/zipcode-75209"},
        }

        frame = normalize_firecrawl_documents(payload)

        self.assertEqual(
            frame.to_dict(orient="records"),
            [
                {
                    "zip": "75209",
                    "source_url": "https://www.realtor.com/local/market/texas/zipcode-75209",
                    "home_value": 1472500.0,
                    "median_rent": 1919.0,
                    "annual_change_pct": 4.12,
                    "as_of_date": "2026-01-01",
                    "source": "realtor",
                    "metric_label": "median_home_price",
                    "supplemental_sources": None,
                }
            ],
        )

    def test_normalize_firecrawl_documents_extracts_redfin_metrics(self):
        payload = {
            "markdown": """
# 75211, TX Housing Market
The 75211 housing market is somewhat competitive. The median sale price of a home in 75211 was $259K last month, down 4.6% since last year.
In January 2026, 75211 home prices were down 4.6% compared to last year, selling for a median price of $259K.
Median Sale Price
$259,000
-4.6% year-over-year
""",
            "metadata": {"sourceURL": "https://www.redfin.com/zipcode/75211/housing-market"},
        }

        frame = normalize_firecrawl_documents(payload)

        self.assertEqual(
            frame.to_dict(orient="records"),
            [
                {
                    "zip": "75211",
                    "source_url": "https://www.redfin.com/zipcode/75211/housing-market",
                    "home_value": 259000.0,
                    "median_rent": None,
                    "annual_change_pct": -4.6,
                    "as_of_date": "2026-01-01",
                    "source": "redfin",
                    "metric_label": "median_sale_price",
                    "supplemental_sources": None,
                }
            ],
        )

    def test_merge_housing_records_drops_source_url_zip_mismatches_before_priority(self):
        frame = _merge_housing_records(
            [
                {
                    "zip": "75054",
                    "source_url": "https://www.zillow.com/home-values/93000/knox-city-tx-79539/",
                    "home_value": 75054.0,
                    "as_of_date": "2026-01-31",
                    "source": "zillow",
                    "metric_label": "typical_home_value",
                    "supplemental_sources": None,
                },
                {
                    "zip": "75054",
                    "source_url": "https://www.realtor.com/local/market/texas/zipcode-75054",
                    "home_value": 420000.0,
                    "as_of_date": "2026-01-01",
                    "source": "realtor",
                    "metric_label": "median_home_price",
                    "supplemental_sources": None,
                },
            ]
        )

        self.assertEqual(frame.loc[0, "zip"], "75054")
        self.assertEqual(frame.loc[0, "source"], "realtor")
        self.assertEqual(frame.loc[0, "home_value"], 420000.0)

    def test_load_local_realtor_history_summary_reuses_existing_history_panel(self):
        with TemporaryDirectory() as tmp_dir:
            history_path = Path(tmp_dir) / "housing_market_history.csv"
            pd.DataFrame(
                {
                    "zip": ["75201", "75201", "75202"],
                    "source": ["realtor_history", "realtor_history", "fhfa_zip5"],
                    "period_start": ["2025-01-01", "2025-02-01", "2025-01-01"],
                    "realtor_hist_listing_price": [500000.0, 550000.0, None],
                    "realtor_hist_active_listing_count": [30.0, 35.0, None],
                    "realtor_hist_median_days_on_market": [40.0, 45.0, None],
                    "realtor_hist_pending_ratio": [0.20, 0.25, None],
                    "realtor_hist_quality_flag": [0.0, 1.0, None],
                }
            ).to_csv(history_path, index=False)

            frame = _load_local_realtor_history_summary(history_path, ["75201"])

        self.assertEqual(frame["zip"].tolist(), ["75201"])
        self.assertEqual(frame.loc[0, "realtor_hist_months_observed"], 2)
        self.assertAlmostEqual(frame.loc[0, "realtor_hist_listing_price_12m_avg"], 525000.0)

    def test_fetch_crime_payload_retries_on_transient_errors(self):
        config = DallasCrimeSourceConfig(dataset_url="https://example.com/resource.json")
        attempts = {"count": 0}

        class _Response:
            def read(self) -> bytes:
                return b'[{"incidentnum": "A-1"}]'

        def flaky_opener(_request):
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise URLError("temporary network issue")
            return _Response()

        payload = fetch_crime_payload(
            config,
            opener=flaky_opener,
            max_attempts=2,
            backoff_seconds=0,
        )

        self.assertEqual(attempts["count"], 2)
        self.assertEqual(payload, [{"incidentnum": "A-1"}])

    def test_fetch_crime_dataset_writes_metadata_artifact(self):
        payload = load_fixture("crime_payload.json")

        with TemporaryDirectory() as tmp_dir:
            settings = Settings.from_env(project_root=Path(tmp_dir))
            settings.ensure_directories()

            with patch("dallas_crime.acquire.crime.fetch_all_crime_records", return_value=payload):
                output = fetch_crime_dataset(settings)

            self.assertTrue(Path(output).exists())
            metadata_path = settings.raw_dir / "crime_records.metadata.json"
            candidate_path = settings.raw_dir / "crime_zip_candidates.csv"
            self.assertTrue(metadata_path.exists())
            self.assertTrue(candidate_path.exists())
            metadata = json.loads(metadata_path.read_text())
            self.assertEqual(metadata["row_counts"]["records_fetched"], len(payload))
            self.assertEqual(metadata["row_counts"]["records_written"], 2)
            self.assertEqual(metadata["query"]["where"], settings.resolved_crime_where_clause())
            self.assertEqual(metadata["zip_candidate_quality"]["minimum_incidents_per_zip"], 2)
            self.assertEqual(metadata["zip_candidate_quality"]["eligible_zip_count"], 0)
            candidates = pd.read_csv(candidate_path)
            self.assertEqual(candidates.loc[0, "zip"], 75201)
            self.assertEqual(candidates.loc[0, "candidate_quality"], "low_count")
            self.assertEqual(candidates.loc[1, "candidate_quality"], "low_count")

    def test_fetch_crime_history_dataset_writes_history_artifact(self):
        payload = load_fixture("crime_payload.json")

        with TemporaryDirectory() as tmp_dir:
            settings = Settings.from_env(project_root=Path(tmp_dir))
            settings.ensure_directories()

            with patch("dallas_crime.acquire.crime.fetch_all_crime_records", return_value=payload):
                output = fetch_crime_history_dataset(settings)

            self.assertTrue(Path(output).exists())
            self.assertEqual(Path(output).name, "crime_history_records.csv")
            metadata_path = settings.raw_dir / "crime_history_records.metadata.json"
            self.assertTrue(metadata_path.exists())
            metadata = json.loads(metadata_path.read_text())
            self.assertEqual(metadata["dataset"], "crime_history_records")
            self.assertEqual(
                metadata["query"]["where"],
                settings.resolved_crime_history_where_clause(),
            )
            self.assertEqual(
                metadata["query"]["lookback_days"],
                settings.crime_history_lookback_days,
            )
            self.assertEqual(metadata["query"]["max_pages"], settings.crime_history_max_pages)
            self.assertNotIn("zip_candidate_quality", metadata)

    def test_fetch_census_zcta_data_filters_to_crime_zip_universe(self):
        census_payload = [
            [*DEFAULT_VARIABLES, "zip code tabulation area"],
            ["12000", "98500", "4000", "2200", "1800", "1900", "11000", "1200", "75201"],
            ["34000", "76500", "10000", "5500", "4500", "1500", "30000", "4000", "75214"],
            ["99999", "55000", "20000", "10000", "10000", "1200", "19000", "2000", "79999"],
        ]

        with TemporaryDirectory() as tmp_dir:
            settings = Settings.from_env(project_root=Path(tmp_dir))
            settings.ensure_directories()
            settings.min_total_incidents_per_zip = 1
            pd.DataFrame({"zip": ["75201", "75201", "75214", "75214"]}).to_csv(
                settings.raw_dir / "crime_records.csv", index=False
            )

            with patch("dallas_crime.acquire.census.fetch_census_payload", return_value=census_payload):
                output = fetch_census_zcta_data(settings)

            frame = pd.read_csv(output, dtype={"zip": str})
            self.assertEqual(frame["zip"].tolist(), ["75201", "75214"])

            metadata = json.loads((settings.raw_dir / "acs_zcta.metadata.json").read_text())
            self.assertEqual(metadata["row_counts"]["rows_before_zip_filter"], 3)
            self.assertEqual(metadata["row_counts"]["rows_after_zip_filter"], 2)
            self.assertEqual(metadata["row_counts"]["crime_zip_universe_size"], 2)
            self.assertEqual(metadata["source_kind"], "api")
            self.assertFalse(metadata["fallback"]["activated"])

    def test_fetch_census_zcta_data_falls_back_to_bulk_tables(self):
        fallback_frame = pd.DataFrame(
            [
                {
                    "zip": "75201",
                    "population": 12000,
                    "median_household_income": 98500,
                    "occupied_housing_units": 4000,
                    "owner_occupied_units": 2200,
                    "renter_occupied_units": 1800,
                    "median_gross_rent": 1900,
                    "poverty_universe": 11000,
                    "poverty_count": 1200,
                },
                {
                    "zip": "75214",
                    "population": 34000,
                    "median_household_income": 76500,
                    "occupied_housing_units": 10000,
                    "owner_occupied_units": 5500,
                    "renter_occupied_units": 4500,
                    "median_gross_rent": 1500,
                    "poverty_universe": 30000,
                    "poverty_count": 4000,
                },
            ]
        )

        with TemporaryDirectory() as tmp_dir:
            settings = Settings.from_env(project_root=Path(tmp_dir))
            settings.ensure_directories()
            settings.min_total_incidents_per_zip = 1
            pd.DataFrame({"zip": ["75201", "75201", "75214", "75214"]}).to_csv(
                settings.raw_dir / "crime_records.csv", index=False
            )

            with (
                patch(
                    "dallas_crime.acquire.census.fetch_census_payload",
                    side_effect=AcquisitionError("api timeout"),
                ),
                patch(
                    "dallas_crime.acquire.census.fetch_census_bulk_dataset",
                    return_value=fallback_frame,
                ),
            ):
                output = fetch_census_zcta_data(settings)

            frame = pd.read_csv(output, dtype={"zip": str})
            self.assertEqual(frame["zip"].tolist(), ["75201", "75214"])
            self.assertAlmostEqual(frame.loc[0, "owner_occupied_share"], 0.55)
            self.assertAlmostEqual(frame.loc[1, "poverty_rate"], 4000 / 30000)

            metadata = json.loads((settings.raw_dir / "acs_zcta.metadata.json").read_text())
            self.assertEqual(metadata["source_kind"], "bulk_table_based")
            self.assertTrue(metadata["fallback"]["activated"])
            self.assertIn("api timeout", metadata["fallback"]["reason"])

    def test_fetch_census_zcta_snapshot_data_writes_multi_year_snapshot_artifact(self):
        payloads_by_year = {
            2022: [
                [*DEFAULT_VARIABLES, "zip code tabulation area"],
                ["12000", "98500", "4000", "2200", "1800", "1900", "11000", "1200", "75201"],
                ["34000", "76500", "10000", "5500", "4500", "1500", "30000", "4000", "75214"],
            ],
            2023: [
                [*DEFAULT_VARIABLES, "zip code tabulation area"],
                ["12500", "99500", "4200", "2300", "1900", "1950", "11200", "1300", "75201"],
                ["35000", "77500", "10100", "5550", "4550", "1525", "30100", "4050", "75214"],
            ],
        }

        with TemporaryDirectory() as tmp_dir:
            settings = Settings.from_env(project_root=Path(tmp_dir))
            settings.ensure_directories()
            settings.census_snapshot_years = (2022, 2023)
            pd.DataFrame({"zip": ["75201", "75201", "75214"]}).to_csv(
                settings.raw_dir / "crime_records.csv", index=False
            )

            def fake_fetch(request, **_kwargs):
                return payloads_by_year[request.year]

            with patch("dallas_crime.acquire.census.fetch_census_payload", side_effect=fake_fetch):
                output = fetch_census_zcta_snapshot_data(settings)

            frame = pd.read_csv(output, dtype={"zip": str})
            self.assertEqual(frame["snapshot_year"].tolist(), [2022, 2023, 2022, 2023])
            self.assertEqual(frame["zip"].tolist(), ["75201", "75201", "75214", "75214"])
            owner_share = frame.loc[
                (frame["zip"] == "75201") & (frame["snapshot_year"] == 2023),
                "owner_occupied_share",
            ].iloc[0]
            self.assertAlmostEqual(
                owner_share,
                2300 / 4200,
            )

            metadata = json.loads(
                (settings.raw_dir / "acs_zcta_snapshots.metadata.json").read_text()
            )
            self.assertEqual(metadata["snapshot_years"], [2022, 2023])
            self.assertEqual(len(metadata["year_results"]), 2)
            self.assertEqual(metadata["year_results"][0]["year"], 2022)
            self.assertEqual(metadata["year_results"][1]["rows_after_zip_filter"], 2)

    def test_fetch_housing_dataset_writes_coverage_and_metadata(self):
        firecrawl_payload = load_fixture("firecrawl_scrape.json")

        with TemporaryDirectory() as tmp_dir:
            settings = Settings.from_env(project_root=Path(tmp_dir))
            settings.ensure_directories()
            settings.min_total_incidents_per_zip = 1
            pd.DataFrame({"zip": ["75201", "75214"]}).to_csv(
                settings.raw_dir / "crime_records.csv", index=False
            )

            def fake_run(command, **_kwargs):
                if command[1] == "search":
                    query = command[2]
                    if "75214" in query:
                        return firecrawl_payload
                    return {"data": {"web": []}}
                if command[1] == "scrape":
                    return {"markdown": "", "metadata": {"sourceURL": command[2]}}
                raise AssertionError(f"unexpected command: {command}")

            with (
                patch("dallas_crime.acquire.housing.run_firecrawl_command", side_effect=fake_run),
                patch(
                    "dallas_crime.acquire.housing.fetch_realtor_zip_inventory",
                    return_value=pd.DataFrame(columns=["zip"]),
                ),
                patch(
                    "dallas_crime.acquire.housing.fetch_realtor_zip_history",
                    return_value=pd.DataFrame(columns=["zip"]),
                ),
                patch(
                    "dallas_crime.acquire.housing.fetch_housing_history_dataset",
                    return_value=str(settings.raw_dir / "housing_market_history.csv"),
                ),
            ):
                output = fetch_housing_dataset(settings)

            frame = pd.read_csv(output, dtype={"zip": str})
            self.assertEqual(frame["zip"].tolist(), ["75214"])

    def test_fetch_housing_dataset_filters_requested_zips_by_incident_threshold(self):
        realtor_75207 = {
            "markdown": """
## 75207, TX market summary
Key indicators as of January 2026
| Metric | Zipwide | 1Y Change | 3Y Change |
| --- | --- | --- | --- |
| Median home $ | $512,000 | 12.50% | 20.00% |
""",
            "metadata": {"sourceURL": "https://www.realtor.com/local/market/texas/zipcode-75207"},
        }

        with TemporaryDirectory() as tmp_dir:
            settings = Settings.from_env(project_root=Path(tmp_dir))
            settings.ensure_directories()
            pd.DataFrame({"zip": ["75207", "75207", "75209"]}).to_csv(
                settings.raw_dir / "crime_records.csv", index=False
            )

            def fake_run(command, **_kwargs):
                if command[1] == "search":
                    return {"data": {"web": []}}
                if command[1] == "scrape":
                    url = command[2]
                    if "zipcode-75207" in url:
                        return realtor_75207
                    return {"markdown": "", "metadata": {"sourceURL": url}}
                raise AssertionError(f"unexpected command: {command}")

            with (
                patch("dallas_crime.acquire.housing.run_firecrawl_command", side_effect=fake_run),
                patch(
                    "dallas_crime.acquire.housing.fetch_realtor_zip_inventory",
                    return_value=pd.DataFrame(columns=["zip"]),
                ),
                patch(
                    "dallas_crime.acquire.housing.fetch_realtor_zip_history",
                    return_value=pd.DataFrame(columns=["zip"]),
                ),
                patch(
                    "dallas_crime.acquire.housing.fetch_housing_history_dataset",
                    return_value=str(settings.raw_dir / "housing_market_history.csv"),
                ),
            ):
                fetch_housing_dataset(settings)

            coverage = json.loads((settings.raw_dir / "housing_zip_coverage.json").read_text())
            self.assertEqual(coverage["requested_zips"], ["75207"])
            frame = pd.read_csv(settings.raw_dir / "housing_market.csv", dtype={"zip": str})
            self.assertEqual(frame["zip"].tolist(), ["75207"])

            self.assertEqual(coverage["requested_zip_count"], 1)
            self.assertEqual(coverage["received_zip_count"], 1)
            self.assertEqual(coverage["coverage_rate"], 1.0)
            self.assertEqual(coverage["missing_zips"], [])

            metadata = json.loads((settings.raw_dir / "housing_market.metadata.json").read_text())
            self.assertEqual(metadata["row_counts"]["records_written"], 1)
            self.assertEqual(metadata["row_counts"]["requested_zip_count"], 1)

    def test_fetch_housing_dataset_uses_realtor_then_redfin_fallbacks(self):
        realtor_75207 = {
            "markdown": """
## 75207, TX market summary
Key indicators as of January 2026
| Metric | Zipwide | 1Y Change | 3Y Change |
| --- | --- | --- | --- |
| Rental properties | 86 | -64.53% | -19.10% |
| Median rent | $2,400/mo | 27.37% | 49.74% |
""",
            "metadata": {"sourceURL": "https://www.realtor.com/local/market/texas/zipcode-75207"},
        }
        realtor_75209 = {
            "markdown": """
## 75209, TX market summary
Key indicators as of January 2026
| Metric | Zipwide | 1Y Change | 3Y Change |
| --- | --- | --- | --- |
| Median home $ | $1,472,500 | 4.12% | 26.30% |
| Median rent | $1,919/mo | 7.57% | 2.26% |
""",
            "metadata": {"sourceURL": "https://www.realtor.com/local/market/texas/zipcode-75209"},
        }
        redfin_75207 = {
            "markdown": """
# 75207, TX Housing Market
The 75207 housing market is somewhat competitive. The median sale price of a home in 75207 was $512K last month, up 12.5% since last year.
In January 2026, 75207 home prices were up 12.5% compared to last year, selling for a median price of $512K.
""",
            "metadata": {"sourceURL": "https://www.redfin.com/zipcode/75207/housing-market"},
        }

        with TemporaryDirectory() as tmp_dir:
            settings = Settings.from_env(project_root=Path(tmp_dir))
            settings.ensure_directories()
            settings.min_total_incidents_per_zip = 1
            pd.DataFrame({"zip": ["75207", "75207", "75209", "75209"]}).to_csv(
                settings.raw_dir / "crime_records.csv", index=False
            )

            def fake_run(command, **_kwargs):
                if command[1] == "search":
                    return {"data": {"web": []}}
                if command[1] == "scrape":
                    url = command[2]
                    if "zipcode-75207" in url and "realtor.com" in url:
                        return realtor_75207
                    if "zipcode-75209" in url and "realtor.com" in url:
                        return realtor_75209
                    if "zipcode/75207" in url and "redfin.com" in url:
                        return redfin_75207
                    return {"markdown": "", "metadata": {"sourceURL": url}}
                raise AssertionError(f"unexpected command: {command}")

            with (
                patch("dallas_crime.acquire.housing.run_firecrawl_command", side_effect=fake_run),
                patch(
                    "dallas_crime.acquire.housing.fetch_realtor_zip_inventory",
                    return_value=pd.DataFrame(columns=["zip"]),
                ),
                patch(
                    "dallas_crime.acquire.housing.fetch_realtor_zip_history",
                    return_value=pd.DataFrame(columns=["zip"]),
                ),
                patch(
                    "dallas_crime.acquire.housing.fetch_housing_history_dataset",
                    return_value=str(settings.raw_dir / "housing_market_history.csv"),
                ),
            ):
                output = fetch_housing_dataset(settings)

            frame = pd.read_csv(output, dtype={"zip": str})
            frame = frame.sort_values("zip").reset_index(drop=True)
            self.assertEqual(frame["zip"].tolist(), ["75207", "75209"])
            self.assertEqual(frame.loc[0, "source"], "redfin")
            self.assertEqual(frame.loc[1, "source"], "realtor")
            self.assertEqual(frame.loc[0, "median_rent"], 2400.0)
            self.assertEqual(frame.loc[0, "supplemental_sources"], "realtor")

            metadata = json.loads((settings.raw_dir / "housing_market.metadata.json").read_text())
            source_counts = {item["name"]: item["records_written"] for item in metadata["sources"]}
            self.assertEqual(source_counts["zillow"], 0)
            self.assertEqual(source_counts["realtor"], 1)
            self.assertEqual(source_counts["redfin"], 1)

    def test_fetch_housing_dataset_uses_realtor_csv_for_remaining_zip(self):
        inventory = pd.DataFrame(
            [
                {
                    "zip": "75242",
                    "realtor_month_date_yyyymm": "202602",
                    "realtor_as_of_date": pd.Timestamp("2026-02-01"),
                    "realtor_listing_price": 425000.0,
                    "realtor_listing_price_yy": 0.083,
                    "realtor_active_listing_count": 18,
                    "realtor_median_days_on_market": 51.0,
                    "realtor_listing_price_per_square_foot": 240.0,
                    "realtor_pending_ratio": 0.32,
                    "realtor_quality_flag": 0,
                }
            ]
        )

        with TemporaryDirectory() as tmp_dir:
            settings = Settings.from_env(project_root=Path(tmp_dir))
            settings.ensure_directories()
            settings.min_total_incidents_per_zip = 1
            pd.DataFrame({"zip": ["75242", "75242"]}).to_csv(
                settings.raw_dir / "crime_records.csv", index=False
            )

            def fake_run(command, **_kwargs):
                if command[1] == "search":
                    return {"data": {"web": []}}
                if command[1] == "scrape":
                    return {"markdown": "", "metadata": {"sourceURL": command[2]}}
                raise AssertionError(f"unexpected command: {command}")

            with (
                patch("dallas_crime.acquire.housing.run_firecrawl_command", side_effect=fake_run),
                patch("dallas_crime.acquire.housing.fetch_realtor_zip_inventory", return_value=inventory),
                patch(
                    "dallas_crime.acquire.housing.fetch_realtor_zip_history",
                    return_value=pd.DataFrame(columns=["zip"]),
                ),
                patch(
                    "dallas_crime.acquire.housing.fetch_housing_history_dataset",
                    return_value=str(settings.raw_dir / "housing_market_history.csv"),
                ),
            ):
                output = fetch_housing_dataset(settings)

            frame = pd.read_csv(output, dtype={"zip": str})
            self.assertEqual(frame.loc[0, "source"], "realtor_csv")
            self.assertEqual(frame.loc[0, "home_value"], 425000.0)
            self.assertEqual(frame.loc[0, "realtor_active_listing_count"], 18)
            self.assertEqual(frame.loc[0, "realtor_pending_ratio"], 0.32)

    def test_fetch_realtor_zip_history_summarizes_recent_months(self):
        chunks = [
            pd.DataFrame(
                [
                    {
                        "month_date_yyyymm": "202501",
                        "postal_code": "75201",
                        "median_listing_price": 400000.0,
                        "active_listing_count": 20,
                        "median_days_on_market": 40.0,
                        "pending_ratio": 0.20,
                        "quality_flag": 0,
                    },
                    {
                        "month_date_yyyymm": "202502",
                        "postal_code": "75201",
                        "median_listing_price": 420000.0,
                        "active_listing_count": 22,
                        "median_days_on_market": 42.0,
                        "pending_ratio": 0.25,
                        "quality_flag": 0,
                    },
                ]
            ),
            pd.DataFrame(
                [
                    {
                        "month_date_yyyymm": "202512",
                        "postal_code": "75201",
                        "median_listing_price": 500000.0,
                        "active_listing_count": 30,
                        "median_days_on_market": 55.0,
                        "pending_ratio": 0.35,
                        "quality_flag": 1,
                    },
                    {
                        "month_date_yyyymm": "202602",
                        "postal_code": "75201",
                        "median_listing_price": 550000.0,
                        "active_listing_count": 35,
                        "median_days_on_market": 60.0,
                        "pending_ratio": 0.40,
                        "quality_flag": 0,
                    },
                    {
                        "month_date_yyyymm": "202602",
                        "postal_code": "99999",
                        "median_listing_price": 100000.0,
                        "active_listing_count": 1,
                        "median_days_on_market": 10.0,
                        "pending_ratio": 0.1,
                        "quality_flag": 0,
                    },
                ]
            ),
        ]

        with (
            patch("dallas_crime.acquire.housing._download_remote_tabular_source", return_value=Path("/tmp/realtor_history.csv")),
            patch("dallas_crime.acquire.housing.pd.read_csv", return_value=iter(chunks)),
        ):
            frame = fetch_realtor_zip_history(["75201"], chunksize=2)

        self.assertEqual(frame["zip"].tolist(), ["75201"])
        self.assertEqual(frame.loc[0, "realtor_hist_months_observed"], 2)
        self.assertAlmostEqual(frame.loc[0, "realtor_hist_listing_price_12m_avg"], (500000 + 550000) / 2)
        self.assertAlmostEqual(frame.loc[0, "realtor_hist_listing_price_12m_change"], (550000 / 500000) - 1)
        self.assertAlmostEqual(frame.loc[0, "realtor_hist_active_listing_count_12m_avg"], (30 + 35) / 2)
        self.assertEqual(frame.loc[0, "realtor_hist_quality_flag_12m_max"], 1)

    def test_fetch_realtor_zip_history_panel_filters_to_year_floor(self):
        chunks = [
            pd.DataFrame(
                [
                    {
                        "month_date_yyyymm": "199912",
                        "postal_code": "75201",
                        "median_listing_price": 190000.0,
                        "active_listing_count": 9,
                        "median_days_on_market": 70.0,
                        "pending_ratio": 0.10,
                        "quality_flag": 0,
                    },
                    {
                        "month_date_yyyymm": "200001",
                        "postal_code": "75201",
                        "median_listing_price": 200000.0,
                        "active_listing_count": 10,
                        "median_days_on_market": 60.0,
                        "pending_ratio": 0.15,
                        "quality_flag": 0,
                    },
                ]
            ),
            pd.DataFrame(
                [
                    {
                        "month_date_yyyymm": "200002",
                        "postal_code": "75201",
                        "median_listing_price": 205000.0,
                        "active_listing_count": 12,
                        "median_days_on_market": 55.0,
                        "pending_ratio": 0.20,
                        "quality_flag": 1,
                    }
                ]
            ),
        ]

        with (
            patch("dallas_crime.acquire.housing._download_remote_tabular_source", return_value=Path("/tmp/realtor_history.csv")),
            patch("dallas_crime.acquire.housing.pd.read_csv", return_value=iter(chunks)),
        ):
            frame = fetch_realtor_zip_history_panel(["75201"], year_floor=2000, chunksize=2)

        self.assertEqual(frame["zip"].tolist(), ["75201", "75201"])
        self.assertEqual(frame["period_year"].tolist(), [2000, 2000])
        self.assertEqual(frame["source"].unique().tolist(), ["realtor_history"])
        self.assertEqual(frame.loc[0, "price_signal_unit"], "usd_median_listing_price")

    def test_fetch_fhfa_zip5_history_filters_to_zip_and_year_floor(self):
        workbook_rows = pd.DataFrame(
            [
                ["75201", 1999, 4.0, 95.0, 97.0, 98.0],
                ["75201", 2000, 5.0, 100.0, 102.0, 100.0],
                ["75201", 2001, 6.0, 106.0, 108.0, 106.0],
                ["99999", 2000, 3.0, 101.0, 103.0, 101.0],
            ]
        )

        with (
            patch("dallas_crime.acquire.housing._download_remote_tabular_source", return_value=Path("/tmp/fhfa.xlsx")),
            patch("dallas_crime.acquire.housing.pd.read_excel", return_value=workbook_rows),
        ):
            frame = fetch_fhfa_zip5_history(["75201"], year_floor=2000)

        self.assertEqual(frame["zip"].tolist(), ["75201", "75201"])
        self.assertEqual(frame["period_year"].tolist(), [2000, 2001])
        self.assertEqual(frame["source"].unique().tolist(), ["fhfa_zip5"])
        self.assertEqual(frame.loc[0, "price_signal_unit"], "index_2000_base")

    def test_fetch_housing_history_dataset_writes_year_2000_panel(self):
        realtor_panel = pd.DataFrame(
            [
                {
                    "zip": "75201",
                    "period_start": "2025-01-01",
                    "period_end": "2025-01-31",
                    "period_year": 2025,
                    "period_month": 1,
                    "frequency": "monthly",
                    "source": "realtor_history",
                    "source_url": "https://example.com/realtor",
                    "metric_label": "median_listing_price",
                    "price_signal_value": 500000.0,
                    "price_signal_unit": "usd_median_listing_price",
                }
            ]
        )
        fhfa_panel = pd.DataFrame(
            [
                {
                    "zip": "75201",
                    "period_start": "2000-01-01",
                    "period_end": "2000-12-31",
                    "period_year": 2000,
                    "period_month": pd.NA,
                    "frequency": "annual",
                    "source": "fhfa_zip5",
                    "source_url": "https://example.com/fhfa",
                    "metric_label": "fhfa_hpi",
                    "price_signal_value": 100.0,
                    "price_signal_unit": "index_2000_base",
                }
            ]
        )

        with TemporaryDirectory() as tmp_dir:
            settings = Settings.from_env(project_root=Path(tmp_dir))
            settings.ensure_directories()

            with (
                patch("dallas_crime.acquire.housing.fetch_realtor_zip_history_panel", return_value=realtor_panel),
                patch("dallas_crime.acquire.housing.fetch_fhfa_zip5_history", return_value=fhfa_panel),
            ):
                output = fetch_housing_history_dataset(settings, zip_codes=["75201"], year_floor=2000)

            frame = pd.read_csv(output, dtype={"zip": str})
            metadata = json.loads((settings.raw_dir / "housing_market_history.metadata.json").read_text())
            self.assertEqual(len(frame), 2)
            self.assertEqual(metadata["min_year"], 2000)
            self.assertEqual(metadata["source_summary"]["fhfa_zip5"]["rows"], 1)
            self.assertEqual(metadata["source_summary"]["realtor_history"]["rows"], 1)

    def test_fetch_housing_dataset_anchors_output_zip_to_requested_zip(self):
        mismatched_payload = {
            "data": [
                {
                    "markdown": "Typical home value in ZIP 79999 is $450,000.",
                    "metadata": {"sourceURL": "https://www.zillow.com/home-values/75214/"},
                }
            ]
        }

        with TemporaryDirectory() as tmp_dir:
            settings = Settings.from_env(project_root=Path(tmp_dir))
            settings.ensure_directories()
            settings.min_total_incidents_per_zip = 1
            pd.DataFrame({"zip": ["75214", "75214"]}).to_csv(
                settings.raw_dir / "crime_records.csv", index=False
            )

            def fake_run(command, **_kwargs):
                if command[1] == "search":
                    return mismatched_payload
                raise AssertionError(f"unexpected command: {command}")

            with (
                patch("dallas_crime.acquire.housing.run_firecrawl_command", side_effect=fake_run),
                patch(
                    "dallas_crime.acquire.housing.fetch_realtor_zip_inventory",
                    return_value=pd.DataFrame(columns=["zip"]),
                ),
                patch(
                    "dallas_crime.acquire.housing.fetch_realtor_zip_history",
                    return_value=pd.DataFrame(columns=["zip"]),
                ),
                patch(
                    "dallas_crime.acquire.housing.fetch_housing_history_dataset",
                    return_value=str(settings.raw_dir / "housing_market_history.csv"),
                ),
            ):
                output = fetch_housing_dataset(settings)

            frame = pd.read_csv(output, dtype={"zip": str})
            self.assertEqual(frame["zip"].tolist(), ["75214"])

    def test_fetch_optional_zip_sidecars_writes_all_category_artifacts(self):
        with TemporaryDirectory() as tmp_dir:
            settings = Settings.from_env(project_root=Path(tmp_dir))
            settings.ensure_directories()

            pd.DataFrame(
                {
                    "zip": ["75201", "75214"],
                    "incident_id": ["a1", "a2"],
                    "reported_at": ["2026-03-01", "2026-03-02"],
                    "offense_family": ["violent", "property"],
                }
            ).to_csv(settings.raw_dir / "crime_records.csv", index=False)

            pd.DataFrame(
                {
                    "zip": ["75201", "75214"],
                    "population": [10000, 20000],
                    "median_household_income": [95000, 78000],
                    "occupied_housing_units": [4200, 8700],
                    "owner_occupied_units": [2000, 4200],
                    "renter_occupied_units": [2200, 4500],
                    "median_gross_rent": [1900, 1650],
                    "poverty_universe": [9800, 19500],
                    "poverty_count": [900, 2600],
                    "unemployment_rate": [0.045, 0.058],
                    "vacancy_proxy": [0.075, 0.09],
                    "educational_attainment": [0.62, 0.48],
                    "public_assistance_share": [0.05, 0.11],
                    "transit_commute_share": [0.14, 0.09],
                }
            ).to_csv(settings.raw_dir / "acs_zcta.csv", index=False)

            pd.DataFrame(
                {
                    "zip": ["75201", "75201", "75214", "75214"],
                    "snapshot_year": [2023, 2024, 2023, 2024],
                    "median_household_income": [93000, 95000, 76000, 78000],
                    "poverty_rate": [0.095, 0.092, 0.14, 0.133],
                }
            ).to_csv(settings.raw_dir / "acs_zcta_snapshots.csv", index=False)

            pd.DataFrame(
                {
                    "zip": ["75201", "75214"],
                    "home_value": [510000, 430000],
                    "annual_change_pct": [4.2, 2.1],
                    "median_rent": [2100, 1750],
                    "realtor_pending_ratio": [0.62, 0.48],
                    "realtor_median_days_on_market": [42, 56],
                }
            ).to_csv(settings.raw_dir / "housing_market.csv", index=False)

            pd.DataFrame(
                {
                    "zip": ["75201", "75214"],
                    "county_name": ["DALLAS COUNTY", "DALLAS COUNTY"],
                    "area_score": [1.0, 1.0],
                }
            ).to_csv(settings.raw_dir / "zcta_county_crosswalk_2020.csv", index=False)

            pd.DataFrame(
                {
                    "Arrest Year": [2025, 2025],
                    "Arrest Zipcode": ["75201", "75214"],
                    "Drug Related": ["Yes", "No"],
                }
            ).to_csv(settings.project_root / "Police_Arrests.csv", index=False)

            artifacts = fetch_optional_zip_sidecars(settings)

            economic = pd.read_csv(artifacts.economic, dtype={"zip": str})
            real_estate = pd.read_csv(artifacts.real_estate, dtype={"zip": str})
            law = pd.read_csv(artifacts.law_enforcement, dtype={"zip": str})
            social = pd.read_csv(artifacts.social_services, dtype={"zip": str})
            infra = pd.read_csv(artifacts.infrastructure, dtype={"zip": str})

            self.assertEqual(economic["zip"].tolist(), ["75201", "75214"])
            self.assertIn("economic_index", economic.columns)
            self.assertIn("investor_purchase_share", real_estate.columns)
            self.assertIn("law_staffing_score", law.columns)
            self.assertIn("clinic_access_score", social.columns)
            self.assertIn("park_access_score", infra.columns)

            metadata = json.loads(Path(artifacts.metadata).read_text())
            self.assertEqual(metadata["zip_universe_size"], 2)
            self.assertEqual(metadata["rows_by_category"]["economic"], 2)

    def test_run_firecrawl_command_retries_then_succeeds(self):
        failure = subprocess.CompletedProcess(
            args=["firecrawl", "search", "75214"],
            returncode=1,
            stdout="",
            stderr="temporary failure",
        )
        success = subprocess.CompletedProcess(
            args=["firecrawl", "search", "75214"],
            returncode=0,
            stdout='{"data": []}',
            stderr="",
        )

        with patch("dallas_crime.acquire.housing.subprocess.run", side_effect=[failure, success]):
            payload = run_firecrawl_command(
                ["firecrawl", "search", "75214"],
                max_attempts=2,
                backoff_seconds=0,
            )

        self.assertEqual(payload, {"data": []})


if __name__ == "__main__":
    unittest.main()
