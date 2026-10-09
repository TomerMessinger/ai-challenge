"""Deterministic tests for the request-understanding models and finalization (no network)."""

import json
from datetime import date

import pytest
from pydantic import ValidationError

from pipeline.brief_understanding import (
    BriefExtraction,
    BriefUnderstanding,
    DietaryRequirements,
    finalize_understanding,
    normalize_for_match,
)
from pipeline.taxonomy import DIETARY_OPTIONS


def make_extraction(*, dates: dict | None = None, dietary: dict | None = None, **fields) -> BriefExtraction:
    """A valid extraction: the empty brief with the given fields filled in."""
    data = BriefExtraction.empty().model_dump()
    data["date_constraints"].update(dates or {})
    data["dietary_requirements"].update(dietary or {})
    data.update(fields)
    return BriefExtraction.model_validate(data)


def need(service_type=None, required=(), preferred=(), unmapped=(), is_required=True) -> dict:
    return {
        "service_type": service_type,
        "required_tags": list(required),
        "preference_tags": list(preferred),
        "unmapped_required_terms": list(unmapped),
        "required": is_required,
    }


# --- normalize_for_match -------------------------------------------------------

@pytest.mark.parametrize(
    ("value", "key"),
    [
        ("Málaga", "malaga"),
        ("MALAGA", "malaga"),
        ("  málaga\t", "malaga"),
        (" Sevilla ", "seville"),
        ("SEVILLA", "seville"),
        ("Seville", "seville"),
        ("Bilbao", "bilbao"),
        ("San   Sebastián", "san sebastian"),
    ],
)
def test_normalize_for_match(value, key):
    assert normalize_for_match(value) == key


def test_finalize_keeps_the_display_city():
    understanding = finalize_understanding(make_extraction(requested_city="  Málaga "))
    assert understanding.requested_city == "Málaga"
    assert normalize_for_match(understanding.requested_city) == "malaga"


def test_unsupported_city_is_preserved():
    understanding = finalize_understanding(make_extraction(requested_city="Bilbao"))
    assert understanding.requested_city == "Bilbao"
    assert normalize_for_match(understanding.requested_city) == "bilbao"


def test_blank_city_becomes_null():
    assert finalize_understanding(make_extraction(requested_city="   ")).requested_city is None


# --- Budgets -----------------------------------------------------------------

def test_per_person_budget_derives_total():
    understanding = finalize_understanding(make_extraction(party_size=10, budget_per_person_max=80))
    assert understanding.budget_per_person_max == 80
    assert understanding.budget_total_max == 800


def test_total_budget_derives_per_person_rounded_down_to_the_cent():
    understanding = finalize_understanding(make_extraction(party_size=45, budget_total_max=3000))
    assert understanding.budget_total_max == 3000
    assert understanding.budget_per_person_max == 66.66


def test_budgets_are_not_derived_without_party_size():
    understanding = finalize_understanding(make_extraction(budget_per_person_max=80))
    assert understanding.budget_total_max is None
    assert understanding.budget_per_person_max == 80


def test_consistent_explicit_budgets_are_kept_as_stated():
    understanding = finalize_understanding(
        make_extraction(party_size=45, budget_total_max=3000, budget_per_person_max=66.67)
    )
    assert (understanding.budget_total_max, understanding.budget_per_person_max) == (3000, 66.67)


def test_inconsistent_explicit_budgets_fail_validation():
    with pytest.raises(ValidationError, match="inconsistent"):
        finalize_understanding(make_extraction(party_size=10, budget_total_max=500, budget_per_person_max=80))


# --- Tags and service types -------------------------------------------------------

def test_required_tag_wins_over_duplicate_preference_and_tags_are_deduplicated():
    extraction = make_extraction(
        service_needs=[need("activity", required=["go_kart", "go_kart"], preferred=["fun", "go_kart", "fun"])],
        global_preference_tags=["premium", "fun", "premium"],
    )
    understanding = finalize_understanding(extraction)
    assert understanding.service_needs[0].required_tags == ["go_kart"]
    assert understanding.service_needs[0].preference_tags == ["fun"]
    assert understanding.global_preference_tags == ["fun", "premium"]  # vocabulary order


def test_understanding_rejects_a_tag_both_required_and_preferred():
    data = BriefExtraction.empty().model_dump()
    data["service_needs"] = [need("activity", required=["go_kart"], preferred=["go_kart"])]
    with pytest.raises(ValidationError, match="both required and preferred"):
        BriefUnderstanding.model_validate(data)


@pytest.mark.parametrize("model", [BriefExtraction, BriefUnderstanding])
@pytest.mark.parametrize(
    "fields",
    [
        {"global_preference_tags": ["team_building"]},
        {"service_needs": [need("activity", required=["karting"])]},
        {"service_needs": [need("activity", preferred=["Fun"])]},
    ],
)
def test_non_canonical_tag_is_rejected(model, fields):
    data = BriefExtraction.empty().model_dump() | fields
    with pytest.raises(ValidationError):
        model.model_validate(data)


# --- Unmapped required terms ---------------------------------------------------

def test_unmapped_terms_get_a_stable_deduplicated_form():
    extraction = make_extraction(
        service_needs=[need("activity", unmapped=[" Paintball ", "paintball", "PAINTBALL", "laser  tag", "  "])]
    )
    assert finalize_understanding(extraction).service_needs[0].unmapped_required_terms == ["laser tag", "paintball"]


def test_unmapped_terms_are_never_mapped_to_canonical_tags():
    extraction = make_extraction(service_needs=[need("activity", unmapped=["Go-Karting"])])
    [finalized] = finalize_understanding(extraction).service_needs
    assert finalized.unmapped_required_terms == ["go-karting"]
    assert finalized.required_tags == []


@pytest.mark.parametrize("terms", [[" paintball"], ["Paintball"], ["paintball", "paintball"], ["tag", "archery"], [""]])
def test_understanding_rejects_unnormalized_unmapped_terms(terms):
    data = BriefExtraction.empty().model_dump()
    data["service_needs"] = [need("activity", unmapped=terms)]
    with pytest.raises(ValidationError, match="unmapped_required_terms"):
        BriefUnderstanding.model_validate(data)


@pytest.mark.parametrize("model", [BriefExtraction, BriefUnderstanding])
def test_unknown_service_type_is_rejected(model):
    data = BriefExtraction.empty().model_dump() | {"service_needs": [need("restaurant")]}
    with pytest.raises(ValidationError):
        model.model_validate(data)


# --- Shape and ranges --------------------------------------------------------

@pytest.mark.parametrize("model", [BriefExtraction, BriefUnderstanding])
def test_unexpected_fields_are_rejected(model):
    data = BriefExtraction.empty().model_dump() | {"missing_information": []}
    with pytest.raises(ValidationError):
        model.model_validate(data)
    data = BriefExtraction.empty().model_dump()
    data["date_constraints"]["season"] = "spring"
    with pytest.raises(ValidationError):
        model.model_validate(data)


@pytest.mark.parametrize("model", [BriefExtraction, BriefUnderstanding])
def test_omitted_keys_are_rejected(model):
    data = BriefExtraction.empty().model_dump()
    del data["requested_city"]
    with pytest.raises(ValidationError):
        model.model_validate(data)
    data = BriefExtraction.empty().model_dump()
    del data["date_constraints"]["month"]
    with pytest.raises(ValidationError):
        model.model_validate(data)
    data = BriefExtraction.empty().model_dump()
    data["service_needs"] = [need("activity")]
    del data["service_needs"][0]["unmapped_required_terms"]
    with pytest.raises(ValidationError):
        model.model_validate(data)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        (None, "party_size", 0),
        (None, "party_size", -3),
        (None, "budget_total_max", -5),
        (None, "budget_per_person_max", 0),
        ("date_constraints", "month", 0),
        ("date_constraints", "month", 13),
        ("date_constraints", "day_of_month", 0),
        ("date_constraints", "day_of_month", 32),
        ("date_constraints", "weekday", "funday"),
        ("date_constraints", "period_of_month", "start"),
        ("date_constraints", "time_of_day", "noon"),
        ("dietary_requirements", "vegan_count", -1),
        ("dietary_requirements", "vegetarian_count", -1),
    ],
)
def test_out_of_range_values_are_rejected(section, key, value):
    data = BriefExtraction.empty().model_dump()
    (data[section] if section else data)[key] = value
    with pytest.raises(ValidationError):
        BriefExtraction.model_validate(data)


def test_dietary_count_cannot_exceed_party_size():
    with pytest.raises(ValidationError, match="exceeds party_size"):
        finalize_understanding(make_extraction(party_size=10, dietary={"vegan_count": 12}))


def test_dietary_fields_follow_the_taxonomy():
    assert [name.removesuffix("_count") for name in DietaryRequirements.model_fields] == list(DIETARY_OPTIONS)


# --- Dates -------------------------------------------------------------------

@pytest.mark.parametrize(
    ("dates", "year"),
    [
        ({"month": 3}, 2027),
        ({"month": 5, "period_of_month": "late"}, 2027),
        ({"month": 12}, 2027),
        ({"month": 2}, 2027),  # the reference month is still upcoming
        ({"month": 1}, 2028),  # already past on 2027-02-01
        ({"month": 1, "year": 2027}, 2027),  # an explicit year is never overwritten
        ({"weekday": "friday", "time_of_day": "afternoon"}, None),
        ({"day_of_month": 15}, None),
    ],
)
def test_year_is_derived_from_the_reference_date(dates, year):
    understanding = finalize_understanding(make_extraction(dates=dates))
    assert understanding.date_constraints.year == year
    for key, value in dates.items():
        assert getattr(understanding.date_constraints, key) == value


def test_passed_day_in_the_reference_month_rolls_to_next_year():
    reference = date(2027, 2, 15)
    passed = finalize_understanding(make_extraction(dates={"month": 2, "day_of_month": 10}), reference)
    upcoming = finalize_understanding(make_extraction(dates={"month": 2, "day_of_month": 20}), reference)
    assert passed.date_constraints.year == 2028
    assert upcoming.date_constraints.year == 2027


# --- Complete, stable JSON shape ------------------------------------------------

EMPTY_UNDERSTANDING = {
    "requested_city": None,
    "party_size": None,
    "budget_total_max": None,
    "budget_per_person_max": None,
    "date_constraints": {
        "year": None,
        "month": None,
        "weekday": None,
        "day_of_month": None,
        "period_of_month": None,
        "time_of_day": None,
    },
    "service_needs": [],
    "dietary_requirements": {"vegan_count": None, "vegetarian_count": None},
    "global_preference_tags": [],
}


def test_empty_brief_has_the_complete_stable_json_shape():
    understanding = finalize_understanding(BriefExtraction.empty())
    dumped = json.loads(understanding.model_dump_json())
    assert dumped == EMPTY_UNDERSTANDING
    assert list(dumped) == list(EMPTY_UNDERSTANDING)
    assert understanding == BriefUnderstanding.empty()
    assert BriefUnderstanding.model_validate_json(understanding.model_dump_json()) == understanding


def test_populated_understanding_keeps_the_complete_shape():
    understanding = finalize_understanding(
        make_extraction(
            requested_city="Madrid",
            party_size=45,
            budget_total_max=3000,
            dietary={"vegan_count": 6},
            service_needs=[need("food", required=["dinner"])],
        )
    )
    dumped = json.loads(understanding.model_dump_json())
    assert list(dumped) == list(EMPTY_UNDERSTANDING)
    assert list(dumped["date_constraints"]) == list(EMPTY_UNDERSTANDING["date_constraints"])
    assert dumped["dietary_requirements"] == {"vegan_count": 6, "vegetarian_count": None}
    assert dumped["service_needs"] == [need("food", required=["dinner"])]
    assert list(dumped["service_needs"][0]) == [
        "service_type",
        "required_tags",
        "preference_tags",
        "unmapped_required_terms",
        "required",
    ]
