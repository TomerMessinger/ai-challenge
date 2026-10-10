"""Deterministic Stage E tests on small in-memory catalogs (no database, no network).

Each Stage C result is produced by the real search_candidates() unless a test
needs a state Stage C never produces.
"""

import json
from decimal import Decimal

import pytest

from pipeline.brief_understanding import BriefUnderstanding
from pipeline.candidate_search import CandidateSearchResult, Rejection, search_candidates
from pipeline.catalog import Service
from pipeline.proposal_builder import ProposalOutput, UnmetRequirement, WarningInfo, build_output
from pipeline.ranking import PreferenceRef, evaluate_preferences, select_winner

TOP_LEVEL_KEYS = {"status", "request_met", "understanding", "proposal", "issues", "error"}
ISSUE_KEYS = {"missing_information", "unmet_requirements", "warnings"}


def service(
    service_id, *, service_type="activity", price="10", capacity=(1, 100), tags=(), dietary=(), months=None,
    supplier="Test Supplier", name=None, description="A test service.",
) -> Service:
    return Service(
        id=service_id,
        supplier_name=supplier,
        city="Testville",
        service_name=name or f"Service {service_id}",
        service_type=service_type,
        description=description,
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


def must_not_load():
    raise AssertionError("the catalog must not be loaded")


def build(u: BriefUnderstanding, *services: Service) -> ProposalOutput:
    """Stage C on an in-memory catalog, then Stage E on its result."""
    catalog = {s.id: s for s in services}
    return build_output(search_candidates(u, load=lambda: catalog), load=lambda: catalog)


def as_json(output: ProposalOutput) -> dict:
    return json.loads(output.model_dump_json())


def warning_codes(output: ProposalOutput) -> list[str]:
    return [warning.code for warning in output.issues.warnings]


def unmet(code, need_index=None, reasons=(), terms=()) -> UnmetRequirement:
    return UnmetRequirement(code=code, need_index=need_index, reasons=list(reasons), terms=list(terms))


EXACT_DATE = {"year": 2027, "month": 3, "day_of_month": 12}


# --- Schema ---------------------------------------------------------------------------------

@pytest.mark.parametrize(
    "output",
    [
        build(understanding(need("activity")), service(1)),
        build(understanding(need("activity"), city=None), service(1)),
        build(understanding(need("activity"), city="Bilbao"), service(1)),
    ],
    ids=["fulfilled", "needs-information", "unfulfillable"],
)
def test_every_status_serializes_every_key(output):
    data = as_json(output)
    assert set(data) == TOP_LEVEL_KEYS
    assert set(data["issues"]) == ISSUE_KEYS


def test_fulfilled_output_shape():
    output = build(understanding(need("activity"), dates=EXACT_DATE), service(1))
    data = as_json(output)
    assert (data["status"], data["request_met"], data["error"]) == ("fulfilled", True, None)
    assert data["issues"] == {"missing_information": [], "unmet_requirements": [], "warnings": []}
    assert set(data["proposal"]) == {"currency", "services", "price_per_person", "total_price", "preference_match"}
    assert data["proposal"]["currency"] == "EUR"
    assert set(data["proposal"]["services"][0]) == {
        "service_id", "need_index", "supplier_name", "service_name", "service_type", "description",
        "quantity", "price_per_person", "subtotal",
    }
    assert set(data["proposal"]["preference_match"]) == {"score", "matched", "unmatched"}
    assert output.understanding == understanding(need("activity"), dates=EXACT_DATE)


# --- Fulfilled ------------------------------------------------------------------------------

def test_service_facts_come_from_the_catalog():
    kart = service(
        1, supplier="Karting Costa del Sol", name="Go-kart Grand Prix (2h)", description="Outdoor karting.",
        price="55", tags=["go_kart"],
    )
    [selected] = build(understanding(need("activity", required=["go_kart"])), kart).proposal.services
    assert (selected.service_id, selected.supplier_name, selected.service_name) == (
        1, "Karting Costa del Sol", "Go-kart Grand Prix (2h)"
    )
    assert (selected.service_type, selected.description, selected.price_per_person) == (
        "activity", "Outdoor karting.", Decimal("55")
    )


def test_need_index_is_the_original_service_need_index():
    u = understanding(
        need("transport", is_required=False),
        need("activity"),
        need("venue", is_required=False),
        need("food"),
    )
    output = build(u, service(1), service(2, service_type="food"))
    assert [(s.service_id, s.need_index) for s in output.proposal.services] == [(1, 1), (2, 3)]


def test_quantity_is_the_party_size_and_subtotal_is_price_times_quantity():
    output = build(understanding(need("activity"), party_size=7), service(1, price="29.99"))
    [selected] = output.proposal.services
    assert (selected.quantity, selected.price_per_person, selected.subtotal) == (7, Decimal("29.99"), Decimal("209.93"))


def test_proposal_totals_sum_the_line_items_and_agree_with_the_winner():
    u = understanding(need("activity"), need("food"), party_size=12)
    services = (service(1, price="55"), service(2, service_type="food", price="19.50"))
    proposal = build(u, *services).proposal
    assert proposal.price_per_person == sum(s.price_per_person for s in proposal.services) == Decimal("74.50")
    assert proposal.total_price == sum(s.subtotal for s in proposal.services) == Decimal("894.00")
    catalog = {s.id: s for s in services}
    winner = select_winner(search_candidates(u, load=lambda: catalog), catalog)
    assert (proposal.price_per_person, proposal.total_price) == (winner.price_per_person, winner.total_price)


def test_the_proposal_is_the_ranked_winner():
    u = understanding(need("activity", preferred=["competitive"]))
    output = build(u, service(1, price="20"), service(2, price="30", tags=["competitive"]))
    assert [s.service_id for s in output.proposal.services] == [2]


def test_preference_match_is_the_shared_evaluation():
    u = understanding(
        need("activity", preferred=["competitive", "memorable"]),
        need("food", preferred=["premium"]),
        global_preferences=["fun", "relaxed"],
    )
    services = (service(1, tags=["competitive", "fun"]), service(2, service_type="food", tags=["premium"]))
    catalog = {s.id: s for s in services}
    proposal = build(u, *services).proposal
    assert proposal.preference_match == evaluate_preferences([1, 2], u, catalog)
    assert proposal.preference_match.matched == [
        PreferenceRef(scope="service_need", tag="competitive", need_index=0),
        PreferenceRef(scope="service_need", tag="premium", need_index=1),
        PreferenceRef(scope="global", tag="fun", need_index=None),
    ]
    assert proposal.preference_match.unmatched == [
        PreferenceRef(scope="service_need", tag="memorable", need_index=0),
        PreferenceRef(scope="global", tag="relaxed", need_index=None),
    ]


def test_unmatched_soft_preferences_never_make_a_request_unfulfillable():
    output = build(understanding(need("activity", preferred=["premium"]), dates=EXACT_DATE), service(1))
    assert (output.status, output.request_met) == ("fulfilled", True)
    assert output.issues.unmet_requirements == []


# --- Warnings -------------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("dates", "codes"),
    [
        ({}, ["date_missing"]),
        ({"time_of_day": "evening"}, ["date_missing"]),
        ({"weekday": "friday"}, ["date_not_exact"]),
        ({"year": 2027}, ["date_not_exact"]),
        ({"year": 2027, "month": 3}, ["date_not_exact"]),
        ({"year": 2027, "month": 3, "weekday": "friday", "time_of_day": "afternoon"}, ["date_not_exact"]),
        ({"year": 2027, "month": 5, "period_of_month": "late"}, ["date_not_exact"]),
        ({"month": 3, "day_of_month": 12}, ["date_not_exact"]),
        (EXACT_DATE, []),
        ({**EXACT_DATE, "weekday": "friday", "time_of_day": "morning"}, []),
    ],
    ids=[
        "nothing", "time-of-day-only", "weekday-only", "year-only", "month", "friday-afternoon-in-march",
        "late-may", "day-without-year", "exact-date", "exact-date-with-extras",
    ],
)
def test_date_warnings(dates, codes):
    output = build(understanding(need("activity"), dates=dates), service(1))
    assert output.status == "fulfilled"
    assert warning_codes(output) == codes


def test_date_warning_messages():
    missing = build(understanding(need("activity")), service(1)).issues.warnings
    not_exact = build(understanding(need("activity"), dates={"month": 3}), service(1)).issues.warnings
    assert missing == [
        WarningInfo(code="date_missing", message="No date or time of year was specified in the request.")
    ]
    assert not_exact == [
        WarningInfo(code="date_not_exact", message="The request does not specify an exact calendar date.")
    ]


def test_an_unmatched_preference_warns_without_repeating_the_tags():
    output = build(understanding(need("activity", preferred=["premium"]), dates=EXACT_DATE), service(1))
    assert output.issues.warnings == [
        WarningInfo(
            code="soft_preferences_unmatched",
            message="The selected proposal does not match all requested soft preferences.",
        )
    ]


@pytest.mark.parametrize(
    ("preferred", "global_preferences"), [(["premium"], ["fun"]), ([], [])], ids=["all-matched", "none-requested"]
)
def test_no_preference_warning_when_nothing_is_unmatched(preferred, global_preferences):
    u = understanding(need("activity", preferred=preferred), dates=EXACT_DATE, global_preferences=global_preferences)
    output = build(u, service(1, tags=["premium", "fun"]))
    assert output.issues.warnings == []


def test_date_and_preference_warnings_coexist_in_order():
    output = build(understanding(need("activity"), global_preferences=["memorable"]), service(1))
    assert warning_codes(output) == ["date_missing", "soft_preferences_unmatched"]


@pytest.mark.parametrize(
    "output",
    [
        build(understanding(need("activity")), service(1)),
        build(understanding(need("activity"), dates={"month": 3}, global_preferences=["fun"]), service(1)),
        build(understanding(need("activity"), city=None), service(1)),
        build(understanding(need("activity"), need("activity")), service(1)),
    ],
    ids=["fulfilled", "fulfilled-with-warnings", "needs-information", "unfulfillable"],
)
def test_no_availability_warning_is_ever_emitted(output):
    assert "availability_not_verified" not in warning_codes(output)


# --- Needs information ----------------------------------------------------------------------

def test_needs_information_copies_the_missing_fields_and_nothing_else():
    u = understanding(need("activity", is_required=False), city=None, party_size=None)
    result = search_candidates(u, load=must_not_load)
    output = build_output(result, load=must_not_load)
    assert output == ProposalOutput(
        status="needs_information",
        request_met=None,
        understanding=u,
        proposal=None,
        issues={"missing_information": ["requested_city", "party_size", "required_service_need"]},
        error=None,
    )
    assert output.issues.missing_information == result.missing_fields
    assert output.issues.unmet_requirements == [] and output.issues.warnings == []


# --- Unfulfillable --------------------------------------------------------------------------

def assert_unfulfillable(output: ProposalOutput, requirements, warnings=()):
    assert (output.status, output.request_met, output.proposal, output.error) == ("unfulfillable", False, None, None)
    assert output.issues.missing_information == []
    assert output.issues.unmet_requirements == requirements
    assert warning_codes(output) == list(warnings)


def test_past_date():
    output = build(understanding(need("activity"), dates={"year": 2026, "month": 12}), service(1))
    assert_unfulfillable(output, [unmet("past_date", reasons=["past_date"])])


def test_unsupported_city_does_not_repeat_the_city():
    output = build(understanding(need("activity"), city="Bilbao"), service(1))
    assert_unfulfillable(output, [unmet("unsupported_city", reasons=["unsupported_city"])])
    assert output.understanding.requested_city == "Bilbao"


def test_unmapped_required_terms_are_reported_per_need_with_their_terms():
    u = understanding(
        need("food", is_required=False),
        need("activity", unmapped=["laser tag", "paintball"]),
        need("activity", unmapped=["archery"]),
    )
    assert_unfulfillable(
        build(u, service(1)),
        [
            unmet("unsupported_requirement", 1, ["unmapped_required_term"], ["laser tag", "paintball"]),
            unmet("unsupported_requirement", 2, ["unmapped_required_term"], ["archery"]),
        ],
    )


def test_past_date_and_unmapped_terms_are_reported_together_in_stage_c_order():
    u = understanding(need("activity", unmapped=["paintball"]), dates={"year": 1940})
    assert_unfulfillable(
        build(u, service(1)),
        [
            unmet("past_date", reasons=["past_date"]),
            unmet("unsupported_requirement", 0, ["unmapped_required_term"], ["paintball"]),
        ],
    )


def test_an_empty_need_reports_only_its_deepest_failing_filter():
    services = (
        service(1, service_type="activity"),  # service_type
        service(2, service_type="food", tags=["lunch"]),  # required_tags
        service(3, service_type="food", tags=["dinner"], capacity=(50, 100)),  # capacity
        service(4, service_type="food", tags=["dinner"]),  # dietary: no vegan option
        service(5, service_type="food", tags=["dinner"], dietary=["vegan"], months=[6, 7]),  # season
    )
    u = understanding(need("food", required=["dinner"]), dietary={"vegan_count": 2}, dates={"month": 3})
    for count, reason in [(1, "service_type"), (2, "required_tags"), (3, "capacity"), (4, "dietary"), (5, "season")]:
        output = build(u, *services[:count])
        assert_unfulfillable(output, [unmet("no_feasible_service", 0, [reason])])


def test_only_empty_needs_are_reported_with_their_original_index():
    u = understanding(need("venue", is_required=False), need("activity"), need("food", required=["dinner"]))
    services = (service(1), service(2, service_type="food", tags=["lunch"]))
    assert_unfulfillable(build(u, *services), [unmet("no_feasible_service", 2, ["required_tags"])])


def test_each_empty_need_is_reported_once_in_need_order():
    u = understanding(need("food"), need("transport"), party_size=40)
    services = (service(1), service(2, service_type="food", capacity=(1, 20)))
    assert_unfulfillable(
        build(u, *services),
        [unmet("no_feasible_service", 0, ["capacity"]), unmet("no_feasible_service", 1, ["service_type"])],
    )


def test_raw_service_rejections_are_not_exposed():
    services = [service(i, service_type="food") for i in range(1, 8)] + [service(8, capacity=(8, 20))]
    output = build(understanding(need("activity"), party_size=40), *services)
    assert_unfulfillable(output, [unmet("no_feasible_service", 0, ["capacity"])])
    text = output.model_dump_json()
    assert "service_ids" not in text and "rejections" not in text


def test_budget_per_person_bundle_failure():
    output = build(understanding(need("activity"), budget_per_person=50), service(1, price="60"))
    assert_unfulfillable(output, [unmet("no_feasible_bundle", reasons=["budget_per_person"])])


def test_budget_total_bundle_failure():
    output = build(understanding(need("activity"), party_size=10, budget_total=400), service(1, price="40.01"))
    assert_unfulfillable(output, [unmet("no_feasible_bundle", reasons=["budget_total"])])


def test_duplicate_service_bundle_failure_warns_about_service_reuse():
    output = build(understanding(need("activity"), need("activity")), service(1))
    assert_unfulfillable(
        output, [unmet("no_feasible_bundle", reasons=["duplicate_service"])], ["service_reuse_not_supported"]
    )
    assert output.issues.warnings == [
        WarningInfo(
            code="service_reuse_not_supported",
            message="V1 does not allow one catalog service to satisfy multiple separate required service needs.",
        )
    ]


def test_bundle_reasons_are_deduplicated_in_stage_c_check_order():
    # 50/person and 499.50 in total agree within tolerance; a 49.99/person bundle passes
    # the per-person ceiling but not the total one.
    u = understanding(need(), need(), party_size=10, budget_per_person=50, budget_total=499.5)
    services = (service(1, price="20"), service(2, price="29.99"), service(3, price="40"))
    assert_unfulfillable(
        build(u, *services),
        [unmet("no_feasible_bundle", reasons=["duplicate_service", "budget_per_person", "budget_total"])],
        ["service_reuse_not_supported"],
    )


def test_budget_failures_alone_do_not_warn_about_service_reuse():
    output = build(understanding(need("activity"), budget_per_person=5), service(1, price="6"), service(2, price="7"))
    assert_unfulfillable(output, [unmet("no_feasible_bundle", reasons=["budget_per_person"])], [])


def test_unfulfillable_output_never_loads_the_catalog():
    services = {1: service(1)}
    result = search_candidates(understanding(need("activity"), city="Bilbao"), load=lambda: services)
    assert build_output(result, load=must_not_load).status == "unfulfillable"


def unfulfillable(u: BriefUnderstanding, rejections) -> CandidateSearchResult:
    return CandidateSearchResult(
        status="unfulfillable", understanding=u, bundles=[], missing_fields=[], rejections=rejections
    )


@pytest.mark.parametrize(
    "rejections",
    [
        [],
        [Rejection(reason="past_date"), Rejection(reason="capacity", need_index=0, service_ids=[1])],
        [Rejection(reason="unsupported_city"), Rejection(reason="budget_total", service_ids=[1])],
        [Rejection(reason="capacity", service_ids=[1])],
        [Rejection(reason="unmapped_required_term", need_index=0)],
        [Rejection(reason="unmapped_required_term")],
    ],
    ids=[
        "no-rejections", "pre-search-and-service", "city-and-bundle", "service-without-need",
        "unmapped-without-terms", "unmapped-without-need",
    ],
)
def test_rejections_matching_no_terminal_case_raise(rejections):
    with pytest.raises(ValueError):
        build_output(unfulfillable(understanding(need("activity")), rejections), load=must_not_load)


# --- Money ----------------------------------------------------------------------------------

def test_money_stays_an_exact_decimal_internally():
    u = understanding(need("activity"), need("food"), party_size=3)
    proposal = build(u, service(1, price="0.10"), service(2, service_type="food", price="0.20")).proposal
    assert (proposal.price_per_person, proposal.total_price) == (Decimal("0.30"), Decimal("0.90"))  # not 0.3000…04
    dumped = proposal.model_dump()
    assert isinstance(dumped["total_price"], Decimal) and isinstance(dumped["services"][0]["subtotal"], Decimal)


def test_money_serializes_as_json_numbers_at_cent_precision():
    u = understanding(need("activity"), need("food"), party_size=7)
    output = build(u, service(1, price="0.10"), service(2, service_type="food", price="29.99"))
    proposal = as_json(output)["proposal"]
    assert [(s["price_per_person"], s["subtotal"]) for s in proposal["services"]] == [(0.1, 0.7), (29.99, 209.93)]
    assert (proposal["price_per_person"], proposal["total_price"]) == (30.09, 210.63)
    assert all(isinstance(proposal[key], float) for key in ("price_per_person", "total_price"))
    assert '"total_price":210.63' in output.model_dump_json()
