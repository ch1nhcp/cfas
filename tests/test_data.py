"""Mock-data integrity tests.

The retrieval tools look up policies/guidelines by FeedbackCategory value, so
every category string in the JSON files MUST be a valid enum member - a drift
here silently turns every lookup into not_found. These tests are the guard.

They also pin the grounding contract: every record carries a citable ID with
a stable prefix, IDs are unique, and cross-references between files resolve.
"""

import json
import re
from pathlib import Path

import pytest

from cfas.models import FeedbackCategory

DATA_DIR = Path(__file__).parent.parent / "data"

# Sample runs use this deliberately-absent ID to demonstrate the
# missing-record path; it must never be added to customers.json.
MISSING_CUSTOMER_ID = "CUST-404"

VALID_CATEGORIES = {c.value for c in FeedbackCategory}


def load(name: str) -> list[dict]:
    with open(DATA_DIR / name, encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list) and data, f"{name} must be a non-empty JSON array"
    return data


@pytest.fixture(scope="module")
def customers() -> list[dict]:
    return load("customers.json")


@pytest.fixture(scope="module")
def policies() -> list[dict]:
    return load("policies.json")


@pytest.fixture(scope="module")
def guidelines() -> list[dict]:
    return load("cs_guidelines.json")


class TestCustomers:
    def test_ids_unique_and_citable(self, customers):
        ids = [c["customer_id"] for c in customers]
        assert len(ids) == len(set(ids))
        assert all(re.fullmatch(r"CUST-\d{3}", i) for i in ids)

    def test_required_fields_present(self, customers):
        for c in customers:
            for field in ("customer_id", "name", "email", "tier",
                          "tenure_months", "past_tickets", "orders"):
                assert field in c, f"{c.get('customer_id')} missing {field}"

    def test_designated_missing_customer_stays_missing(self, customers):
        assert MISSING_CUSTOMER_ID not in {c["customer_id"] for c in customers}

    def test_past_ticket_categories_are_valid(self, customers):
        for c in customers:
            for t in c["past_tickets"]:
                assert t["category"] in VALID_CATEGORIES, (
                    f"{c['customer_id']} ticket {t['ticket_id']}: {t['category']}"
                )


class TestPolicies:
    def test_ids_unique_and_citable(self, policies):
        ids = [p["policy_id"] for p in policies]
        assert len(ids) == len(set(ids))
        assert all(re.fullmatch(r"POL-[A-Z]+-\d{3}", i) for i in ids)

    def test_required_fields_present(self, policies):
        for p in policies:
            for field in ("policy_id", "name", "applicable_categories",
                          "rules", "last_updated"):
                assert field in p, f"{p.get('policy_id')} missing {field}"
            assert p["rules"], f"{p['policy_id']} has no rules"

    def test_applicable_categories_match_taxonomy(self, policies):
        for p in policies:
            for cat in p["applicable_categories"]:
                assert cat in VALID_CATEGORIES, f"{p['policy_id']}: {cat}"


class TestGuidelines:
    def test_ids_unique_and_citable(self, guidelines):
        ids = [g["guideline_id"] for g in guidelines]
        assert len(ids) == len(set(ids))
        assert all(re.fullmatch(r"SOP-[A-Z]+-\d{3}", i) for i in ids)

    def test_required_fields_present(self, guidelines):
        for g in guidelines:
            for field in ("guideline_id", "category", "title", "required_steps"):
                assert field in g, f"{g.get('guideline_id')} missing {field}"
            assert g["required_steps"], f"{g['guideline_id']} has no steps"

    def test_categories_match_taxonomy(self, guidelines):
        for g in guidelines:
            assert g["category"] in VALID_CATEGORIES, (
                f"{g['guideline_id']}: {g['category']}"
            )

    def test_policy_references_in_steps_resolve(self, guidelines, policies):
        """A SOP step citing POL-XXX-NNN must point at a real policy - the
        agent will surface these IDs in reports, so they must be groundable."""
        known = {p["policy_id"] for p in policies}
        for g in guidelines:
            for step in g["required_steps"]:
                for ref in re.findall(r"POL-[A-Z]+-\d{3}", step):
                    assert ref in known, f"{g['guideline_id']} cites unknown {ref}"
