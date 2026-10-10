"""Deterministic Stage C tests on small in-memory catalogs (no database, no network)."""

import sqlite3
from datetime import date
from decimal import Decimal

import pytest

from pipeline.brief_understanding import REFERENCE_DATE, BriefUnderstanding, DateConstraints
from pipeline.candidate_search import Rejection, is_definitely_past, search_candidates
from pipeline.catalog import Service

ACTIVITY_MONTHS = (4, 5, 6, 7, 8, 9, 10)


def service(
    service_id, *, service_type="activity", city="Testville", supplier="Test Supplier", price="10",
    capacity=(1, 100), tags=(), dietary=(), months=None,
) -> Service:
    return Service(
        id=service_id,
        supplier_name=supplier,
        city=city,
        service_name=f"Service {service_id}",
        service_type=service_type,
        description="A test service.",
        price_per_person=Decimal(price),
        min_capacity=capacity[0],
        max_capacity=capacity[1],
        duration_minutes=None,
        tags=tuple(tags),
        dietary_options=tuple(dietary),
        wheelchair_accessible=None,
        available_months=None if months is None else tuple(months),
        notes=None,
    )


def need(service_type=None, required=(), preferred=(), unmapped=(), is_required=True) -> dict:
    return {
        "service_type": service_type,
        "required_tags": list(required),
        "preference_tags": list(preferred),
        "unmapped_required_terms": list(unmapped),
        "required": is_required,
    }


def understanding(
    *needs, city="Testville", party_size=10, budget_per_person=None, budget_total=None, dates=None,
    dietary=None, global_preferences=(),
) -> BriefUnderstanding:
    data = BriefUnderstanding.empty().model_dump()
    data.update(
        requested_city=city,
        party_size=party_size,
        budget_per_person_max=budget_per_person,
        budget_total_max=budget_total,
        service_needs=list(needs),
        global_preference_tags=list(global_preferences),
    )
    data["date_constraints"].update(dates or {})
    data["dietary_requirements"].update(dietary or {})
    return BriefUnderstanding.model_validate(data)


def catalog(*services: Service):
    return lambda: {s.id: s for s in services}


def must_not_load():
    raise AssertionError("the catalog must not be loaded")


def run(u: BriefUnderstanding, *services: Service, **kwargs):
    return search_candidates(u, load=catalog(*services), **kwargs)


def assert_feasible(result, bundles):
    assert (result.status, result.bundles, result.missing_fields, result.rejections) == ("feasible", bundles, [], [])


def assert_unfulfillable(result, rejections):
    assert (result.status, result.bundles, result.missing_fields) == ("unfulfillable", [], [])
    assert result.rejections == rejections


# --- Result shape ------------------------------------------------------------------

def test_result_always_has_every_field_and_the_unchanged_understanding():
    u = understanding(need("activity"))
    result = run(u, service(1))
    assert set(result.model_dump()) == {"status", "understanding", "bundles", "missing_fields", "rejections"}
    assert result.understanding == u
    result.model_dump_json()  # serializable


# --- Readiness gate ----------------------------------------------------------------

@pytest.mark.parametrize(
    ("fields", "missing"),
    [
        ({"city": None}, ["requested_city"]),
        ({"party_size": None}, ["party_size"]),
        ({"city": None, "party_size": None}, ["requested_city", "party_size"]),
    ],
)
def test_missing_city_or_party_size_needs_information(fields, missing):
    u = understanding(need("activity"), **fields)
    result = search_candidates(u, load=must_not_load)
    assert (result.status, result.missing_fields, result.bundles, result.rejections) == (
        "needs_information", missing, [], []
    )
    assert result.understanding == u


def test_no_service_need_needs_information():
    result = search_candidates(understanding(), load=must_not_load)
    assert (result.status, result.missing_fields) == ("needs_information", ["required_service_need"])


def test_optional_needs_alone_do_not_make_a_brief_ready():
    u = understanding(need("activity", is_required=False), need("food", is_required=False))
    result = search_candidates(u, load=must_not_load)
    assert (result.status, result.missing_fields) == ("needs_information", ["required_service_need"])


def test_every_missing_readiness_field_is_reported_in_stable_order():
    u = understanding(need("activity", is_required=False), city=None, party_size=None)
    result = search_candidates(u, load=must_not_load)
    assert result.missing_fields == ["requested_city", "party_size", "required_service_need"]
    assert result.bundles == [] and result.rejections == []


def test_readiness_runs_before_hard_blockers():
    u = understanding(need("activity", unmapped=["paintball"]), city=None, dates={"year": 1940})
    result = search_candidates(u, load=must_not_load)
    assert (result.status, result.missing_fields, result.rejections) == ("needs_information", ["requested_city"], [])


# --- Past dates ---------------------------------------------------------------------

AS_OF = date(2031, 6, 15)  # arbitrary, so the logic is shown to be generic


@pytest.mark.parametrize(
    ("fields", "past"),
    [
        ({"year": 1940}, True),
        ({"year": 2030}, True),
        ({"year": 2030, "month": 12, "day_of_month": 31}, True),
        ({"year": 2032}, False),
        ({"year": 2032, "month": 1, "day_of_month": 1}, False),
        ({"year": 2031, "month": 5}, True),
        ({"year": 2031, "month": 7}, False),
        ({"year": 2031, "month": 6, "day_of_month": 14}, True),
        ({"year": 2031, "month": 6, "day_of_month": 15}, False),
        ({"year": 2031, "month": 6, "day_of_month": 16}, False),
        ({"year": 2031}, False),
        ({"year": 2031, "day_of_month": 1}, False),
        ({"year": 2031, "month": 6}, False),
        ({"month": 1}, False),
        ({"weekday": "friday"}, False),
        ({"period_of_month": "early"}, False),
        ({"time_of_day": "morning"}, False),
        ({}, False),
    ],
)
def test_is_definitely_past(fields, past):
    dates = DateConstraints(**{**dict.fromkeys(DateConstraints.model_fields), **fields})
    assert is_definitely_past(dates, AS_OF) is past


def test_past_date_is_unfulfillable_without_loading_the_catalog():
    result = search_candidates(understanding(need("activity"), dates={"year": 1940}), load=must_not_load)
    assert_unfulfillable(result, [Rejection(reason="past_date")])


def test_past_date_defaults_to_the_challenge_reference_date():
    assert REFERENCE_DATE == date(2027, 2, 1)
    yesterday = understanding(need("activity"), dates={"year": 2027, "month": 1, "day_of_month": 31})
    today = understanding(need("activity"), dates={"year": 2027, "month": 2, "day_of_month": 1})
    assert_unfulfillable(run(yesterday, service(1)), [Rejection(reason="past_date")])
    assert_feasible(run(today, service(1)), [[1]])


def test_an_explicit_as_of_date_replaces_the_reference_date():
    u = understanding(need("activity"), dates={"year": 2027, "month": 1})
    assert run(u, service(1)).status == "unfulfillable"
    assert_feasible(run(u, service(1), as_of_date=date(2026, 12, 1)), [[1]])


# --- Pre-search blockers -------------------------------------------------------------

def test_required_unmapped_term_is_unfulfillable_without_loading_the_catalog():
    u = understanding(need("activity", unmapped=["paintball"]))
    result = search_candidates(u, load=must_not_load)
    assert_unfulfillable(result, [Rejection(reason="unmapped_required_term", need_index=0)])


def test_past_date_and_unmapped_terms_are_reported_together():
    u = understanding(need("food"), need("activity", unmapped=["paintball"]), dates={"year": 1940})
    result = search_candidates(u, load=must_not_load)
    assert_unfulfillable(
        result, [Rejection(reason="past_date"), Rejection(reason="unmapped_required_term", need_index=1)]
    )


def test_optional_unmapped_term_is_ignored():
    u = understanding(need("activity"), need("activity", unmapped=["paintball"], is_required=False))
    assert_feasible(run(u, service(1)), [[1]])


def test_unsupported_city_is_unfulfillable():
    result = run(understanding(need("activity"), city="Bilbao"), service(1))
    assert_unfulfillable(result, [Rejection(reason="unsupported_city")])


def test_city_is_matched_on_its_normalized_form():
    assert_feasible(run(understanding(need("activity"), city="MALAGA"), service(1, city="Málaga")), [[1]])


def test_search_is_scoped_to_the_requested_city():
    services = (service(1, city="Madrid"), service(2, city="Testville"))
    assert_feasible(run(understanding(need("activity")), *services), [[2]])


# --- Service type and required tags -------------------------------------------------------

def test_service_type_mismatch_rejects():
    result = run(understanding(need("food")), service(1, service_type="activity"))
    assert_unfulfillable(result, [Rejection(reason="service_type", need_index=0, service_ids=[1])])


def test_null_service_type_allows_every_type():
    services = [service(i, service_type=kind) for i, kind in enumerate(("activity", "food", "transport", "venue"), 1)]
    assert_feasible(run(understanding(need()), *services), [[1], [2], [3], [4]])


def test_every_required_tag_must_be_present():
    services = (
        service(1, tags=["outdoor", "active", "fun"]),
        service(2, tags=["outdoor"]),
        service(3, tags=["active"]),
    )
    assert_feasible(run(understanding(need(required=["outdoor", "active"])), *services), [[1]])


def test_partial_required_tag_match_rejects():
    result = run(understanding(need(required=["outdoor", "active"])), service(1, tags=["outdoor", "fun"]))
    assert_unfulfillable(result, [Rejection(reason="required_tags", need_index=0, service_ids=[1])])


def test_no_required_tags_imposes_no_tag_filter():
    services = (service(1, tags=["indoor"]), service(2, tags=["outdoor"]), service(3))
    assert_feasible(run(understanding(need("activity")), *services), [[1], [2], [3]])


# --- Capacity ----------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("capacity", "fits"),
    [((10, 20), True), ((5, 10), True), ((11, 20), False), ((5, 9), False)],
    ids=["exactly-min", "exactly-max", "one-below-min", "one-above-max"],
)
def test_whole_party_must_fit_inclusive_capacity(capacity, fits):
    result = run(understanding(need("activity"), party_size=10), service(1, capacity=capacity))
    if fits:
        assert_feasible(result, [[1]])
    else:
        assert_unfulfillable(result, [Rejection(reason="capacity", need_index=0, service_ids=[1])])


# --- Dietary -----------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("dietary", "options", "fits"),
    [
        ({"vegan_count": 2}, ["vegan", "vegetarian"], True),
        ({"vegan_count": 2}, ["vegan"], True),
        ({"vegan_count": 2}, ["vegetarian"], False),
        ({"vegan_count": 2}, [], False),
        ({"vegetarian_count": 2}, ["vegetarian"], True),
        ({"vegetarian_count": 2}, ["vegan"], True),
        ({"vegetarian_count": 2}, [], False),
        ({"vegan_count": 1, "vegetarian_count": 1}, ["vegetarian"], False),
        ({"vegan_count": 0, "vegetarian_count": 0}, [], True),
        ({}, [], True),
    ],
)
def test_dietary_requirements_filter_food(dietary, options, fits):
    result = run(understanding(need("food"), dietary=dietary), service(1, service_type="food", dietary=options))
    if fits:
        assert_feasible(result, [[1]])
    else:
        assert_unfulfillable(result, [Rejection(reason="dietary", need_index=0, service_ids=[1])])


def test_dietary_requirements_do_not_reject_non_food_services():
    u = understanding(need(), dietary={"vegan_count": 2, "vegetarian_count": 3})
    assert_feasible(run(u, service(1, service_type="activity"), service(2, service_type="venue")), [[1], [2]])


# --- Season ------------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("month", "months", "fits"),
    [(3, ACTIVITY_MONTHS, False), (5, ACTIVITY_MONTHS, True), (3, None, True), (None, ACTIVITY_MONTHS, True)],
    ids=["unavailable-month", "available-month", "no-seasonal-restriction", "no-requested-month"],
)
def test_requested_month_must_be_an_available_month(month, months, fits):
    result = run(understanding(need("activity"), dates={"month": month}), service(1, months=months))
    if fits:
        assert_feasible(result, [[1]])
    else:
        assert_unfulfillable(result, [Rejection(reason="season", need_index=0, service_ids=[1])])


# --- Several needs -----------------------------------------------------------------------

def test_required_needs_are_searched_independently():
    services = (
        service(1, tags=["fun"]),
        service(2, tags=["fun", "active"]),
        service(3, service_type="food"),
        service(4, service_type="food", capacity=(50, 100)),
    )
    u = understanding(need("activity", required=["fun"]), need("food"))
    assert_feasible(run(u, *services), [[1, 3], [2, 3]])


def test_a_required_need_without_candidates_makes_the_brief_unfulfillable():
    services = (service(1), service(2, service_type="food", capacity=(50, 100)))
    result = run(understanding(need("activity"), need("food")), *services)
    # Only the empty need's rejections are reported: one per city service, with its first failing filter.
    assert_unfulfillable(
        result,
        [
            Rejection(reason="service_type", need_index=1, service_ids=[1]),
            Rejection(reason="capacity", need_index=1, service_ids=[2]),
        ],
    )


def test_optional_need_without_candidates_is_ignored():
    u = understanding(need("transport", is_required=False), need("activity"))
    assert_feasible(run(u, service(1)), [[1]])


def test_need_index_counts_optional_needs():
    u = understanding(need("activity", is_required=False), need("food"))
    result = run(u, service(1))
    assert_unfulfillable(result, [Rejection(reason="service_type", need_index=1, service_ids=[1])])


def test_one_service_cannot_fill_two_required_needs():
    result = run(understanding(need("activity"), need("activity")), service(1))
    assert_unfulfillable(result, [Rejection(reason="duplicate_service", service_ids=[1, 1])])


def test_bundles_that_repeat_a_service_are_dropped():
    services = (service(1, tags=["fun"]), service(2, tags=["fun", "active"]))
    u = understanding(need(required=["fun"]), need(required=["active"]))
    assert_feasible(run(u, *services), [[1, 2]])  # (2, 2) is rejected


def test_different_services_from_one_supplier_are_allowed():
    services = (service(1, supplier="Same Supplier"), service(2, service_type="food", supplier="Same Supplier"))
    assert_feasible(run(understanding(need("activity"), need("food")), *services), [[1, 2]])


# --- Budget ------------------------------------------------------------------------------

def test_budget_applies_to_the_bundle_not_to_each_candidate():
    services = (
        service(1, price="30"),
        service(2, service_type="food", price="15"),
        service(3, service_type="food", price="25"),
    )
    u = understanding(need("activity"), need("food"), budget_per_person=50)
    assert_feasible(run(u, *services), [[1, 2]])


def test_a_service_over_budget_is_rejected_as_a_bundle_not_as_a_candidate():
    result = run(understanding(need("activity"), budget_per_person=50), service(1, price="60"))
    assert_unfulfillable(result, [Rejection(reason="budget_per_person", service_ids=[1])])


@pytest.mark.parametrize(("price", "fits"), [("40.00", True), ("40.01", False)])
def test_per_person_ceiling(price, fits):
    result = run(understanding(need("activity"), budget_per_person=40), service(1, price=price))
    if fits:
        assert_feasible(result, [[1]])
    else:
        assert_unfulfillable(result, [Rejection(reason="budget_per_person", service_ids=[1])])


@pytest.mark.parametrize(("price", "fits"), [("40.00", True), ("40.01", False)])
def test_total_ceiling(price, fits):
    result = run(understanding(need("activity"), party_size=10, budget_total=400), service(1, price=price))
    if fits:
        assert_feasible(result, [[1]])
    else:
        assert_unfulfillable(result, [Rejection(reason="budget_total", service_ids=[1])])


@pytest.mark.parametrize(
    ("per_person", "total", "reason"),
    [(50, 500, None), (50, 499.99, "budget_total"), (49.99, 500, "budget_per_person")],
)
def test_both_ceilings_are_enforced(per_person, total, reason):
    u = understanding(need("activity"), party_size=10, budget_per_person=per_person, budget_total=total)
    result = run(u, service(1, price="50"))
    if reason is None:
        assert_feasible(result, [[1]])
    else:
        assert_unfulfillable(result, [Rejection(reason=reason, service_ids=[1])])


def test_bundle_prices_are_summed_exactly():
    services = (service(1, price="0.10"), service(2, service_type="food", price="0.20"))
    u = understanding(need("activity"), need("food"), budget_per_person=0.3)  # 0.1 + 0.2 > 0.3 in floats
    assert_feasible(run(u, *services), [[1, 2]])


# --- Stage boundary ------------------------------------------------------------------------

def test_preferences_never_change_the_result():
    services = (
        service(1, tags=["outdoor", "active", "competitive"]),
        service(2, tags=["outdoor", "relaxed"], price="80"),
        service(3, tags=["indoor", "premium"]),
    )
    plain = understanding(need(required=["outdoor"]), need("food", is_required=False), budget_per_person=100)
    preferring = understanding(
        need(required=["outdoor"], preferred=["competitive", "fun"]),
        need("food", preferred=["premium"], is_required=False),
        budget_per_person=100,
        global_preferences=["relaxed", "premium", "memorable"],
    )
    first, second = run(plain, *services), run(preferring, *services)
    assert_feasible(first, [[1], [2]])
    assert (second.status, second.bundles, second.rejections) == (first.status, first.bundles, first.rejections)


def test_bundles_are_ordered_by_service_ids():
    services = (
        service(5),
        service(4, service_type="food"),
        service(2, service_type="food"),
        service(1),
    )
    assert_feasible(run(understanding(need("activity"), need("food")), *services), [[1, 2], [1, 4], [5, 2], [5, 4]])


# --- Infrastructure failures -----------------------------------------------------------------

def test_catalog_errors_raise_instead_of_returning_unfulfillable():
    def broken():
        raise sqlite3.OperationalError("unable to open database file")

    with pytest.raises(sqlite3.OperationalError):
        search_candidates(understanding(need("activity")), load=broken)
