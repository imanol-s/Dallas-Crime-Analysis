import json
from pathlib import Path
import unittest

from dallas_crime.acquire.census import CensusRequest, build_census_url, normalize_census_payload
from dallas_crime.acquire.crime import DallasCrimeSourceConfig, build_crime_url, normalize_crime_records
from dallas_crime.acquire.housing import (
    FirecrawlScrapeRequest,
    FirecrawlSearchRequest,
    build_firecrawl_scrape_command,
    build_firecrawl_search_command,
    normalize_firecrawl_documents,
    parse_firecrawl_search_results,
)


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
                    "source": "firecrawl",
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
