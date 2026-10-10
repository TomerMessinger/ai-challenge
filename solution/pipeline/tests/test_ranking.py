"""Deterministic Stage D tests on small in-memory catalogs (no database, no network)."""

from decimal import Decimal
from itertools import permutations

import pytest

from pipeline.brief_understanding import BriefUnderstanding
from pipeline.candidate_search import CandidateSearchResult
from pipeline.catalog import Service
from pipeline.ranking import RankedBundle, rank_bundles, select_winner


def service(service_id, *, price="10", tags=()) -> Service:
    return Service(
        id=service_id,
        supplier_name="Test Supplier",
        city="Testville",
        service_name=f"Service {service_id}",
        service_type="activity",
        description="A test service.",
        price_per_person=Decimal(price),
        min_capacity=1,
        max_capacity=100,
        duration_minutes=None,
        tags=tuple(tags),
        dietary_options=(),
        wheelchair_accessible=None,
        available_months=None,
        notes=None,
    )


def catalog(*services: Service) -> dict[int, Service]:
    return {s.id: s for s in services}


def need(preferred=(), is_required=True) -> dict:
    return {
        "service_type": None,
        "required_tags": [],
        "preference_tags": list(preferred),
        "unmapped_required_terms": [],
        "required": is_required,
    }


def result(*needs, bundles, party_size=10, global_preferences=(), status="feasible") -> CandidateSearchResult:
    data = BriefUnderstanding.empty().model_dump()
    data.update(
        requested_city="Testville",
        party_size=party_size,
        service_needs=list(needs),
        global_preference_tags=list(global_preferences),
    )
    return CandidateSearchResult(
        status=status,
        understanding=BriefUnderstanding.model_validate(data),
        bundles=bundles,
        missing_fields=[],
        rejections=[],
    )


def ranked(r: CandidateSearchResult, cat: dict[int, Service]) -> list[tuple[list[int], int, Decimal]]:
    return [(b.service_ids, b.preference_score, b.total_price) for b in rank_bundles(r, cat)]


# --- No preferences -------------------------------------------------------------------

def test_without_preferences_every_bundle_scores_zero_and_the_cheapest_wins():
    cat = catalog(service(1, price="30"), service(2, price="20"), service(3, price="25"))
    r = result(need(), bundles=[[1], [2], [3]])
    assert ranked(r, cat) == [([2], 0, Decimal("200")), ([3], 0, Decimal("250")), ([1], 0, Decimal("300"))]
    assert select_winner(r, cat).service_ids == [2]


# --- Need preferences ---------------------------------------------------------------------

def test_a_need_preference_is_scored_only_on_the_service_filling_that_need():
    # The same two services at the same price; only the need each one fills differs.
    cat = catalog(service(1, tags=["premium"]), service(2, tags=["competitive"]))
    r = result(need(preferred=["competitive"]), need(preferred=["premium"]), bundles=[[1, 2], [2, 1]])
    assert ranked(r, cat) == [([2, 1], 2, Decimal("200")), ([1, 2], 0, Decimal("200"))]


def test_need_preference_matches_add_up():
    cat = catalog(service(1, tags=["competitive", "fun", "outdoor"]), service(2, tags=["fun"]))
    r = result(need(preferred=["competitive", "fun", "memorable"]), bundles=[[1], [2]])
    assert [(b.service_ids, b.preference_score) for b in rank_bundles(r, cat)] == [([1], 2), ([2], 1)]


# --- Global preferences -------------------------------------------------------------------

@pytest.mark.parametrize(
    ("first_tags", "second_tags", "score"),
    [([], [], 0), (["memorable"], [], 1), ([], ["memorable"], 1), (["memorable"], ["memorable"], 1)],
    ids=["neither", "first-service", "second-service", "both-services-count-once"],
)
def test_a_global_preference_counts_once_if_any_selected_service_has_it(first_tags, second_tags, score):
    cat = catalog(service(1, tags=first_tags), service(2, tags=second_tags))
    r = result(need(), need(), bundles=[[1, 2]], global_preferences=["memorable"])
    assert select_winner(r, cat).preference_score == score


def test_need_and_global_preferences_add_up():
    cat = catalog(service(1, tags=["competitive", "memorable"]), service(2, tags=["premium", "memorable"]))
    r = result(
        need(preferred=["competitive"]),
        need(preferred=["relaxed"]),
        bundles=[[1, 2]],
        global_preferences=["memorable", "premium", "fun"],
    )
    assert select_winner(r, cat).preference_score == 1 + 2  # competitive; memorable and premium


# --- Ordering -----------------------------------------------------------------------------

def test_a_higher_score_beats_a_lower_price():
    cat = catalog(service(1, price="100", tags=["competitive"]), service(2, price="10"))
    r = result(need(preferred=["competitive"]), bundles=[[1], [2]])
    assert select_winner(r, cat).service_ids == [1]


def test_on_equal_scores_the_lower_total_price_wins():
    cat = catalog(service(1, price="30", tags=["competitive"]), service(2, price="29.99", tags=["competitive"]))
    r = result(need(preferred=["competitive"]), bundles=[[1], [2]])
    assert ranked(r, cat) == [([2], 1, Decimal("299.90")), ([1], 1, Decimal("300"))]


def test_on_equal_score_and_price_the_smaller_service_id_tuple_wins():
    cat = catalog(service(1, price="10"), service(2, price="15"), service(3, price="20"), service(4, price="15"))
    r = result(need(), need(), bundles=[[3, 1], [2, 4], [1, 3]])  # all 30 per person
    assert [b.service_ids for b in rank_bundles(r, cat)] == [[1, 3], [2, 4], [3, 1]]


def test_ranking_does_not_depend_on_the_input_bundle_order():
    cat = catalog(service(1), service(2, tags=["fun"]), service(3, price="5"), service(4))
    bundles = [[1], [2], [3], [4]]
    expected = rank_bundles(result(need(preferred=["fun"]), bundles=bundles), cat)
    assert [b.service_ids for b in expected] == [[2], [3], [1], [4]]  # score, then price, then IDs
    for order in permutations(bundles):
        assert rank_bundles(result(need(preferred=["fun"]), bundles=list(order)), cat) == expected


def test_optional_needs_are_skipped_when_aligning_bundles_with_needs():
    cat = catalog(service(1, tags=["relaxed"]), service(2, tags=["premium"]))
    r = result(
        need(preferred=["premium"], is_required=False),
        need(preferred=["relaxed"]),
        need(preferred=["fun"], is_required=False),
        need(preferred=["premium"]),
        bundles=[[1, 2]],
    )
    assert select_winner(r, cat).preference_score == 2


# --- Prices -------------------------------------------------------------------------------

def test_prices_are_recomputed_as_exact_decimals():
    cat = catalog(service(1, price="0.10"), service(2, price="0.20"))
    winner = select_winner(result(need(), need(), bundles=[[1, 2]], party_size=3), cat)
    assert winner == RankedBundle(
        service_ids=[1, 2], preference_score=0, price_per_person=Decimal("0.30"), total_price=Decimal("0.90")
    )
    assert isinstance(winner.price_per_person, Decimal) and isinstance(winner.total_price, Decimal)


# --- Invariants ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("status", "bundles", "party_size", "message"),
    [
        ("needs_information", [], 10, "got status 'needs_information'"),
        ("unfulfillable", [], 10, "got status 'unfulfillable'"),
        ("feasible", [], 10, "no bundles"),
        ("feasible", [[1, 2]], 10, "required needs"),
        ("feasible", [[9]], 10, "missing from the catalog"),
        ("feasible", [[1]], None, "no party_size"),
    ],
    ids=["needs-information", "unfulfillable", "no-bundles", "wrong-length", "unknown-id", "no-party-size"],
)
@pytest.mark.parametrize("rank", [rank_bundles, select_winner])
def test_invalid_internal_states_raise(rank, status, bundles, party_size, message):
    r = result(need(), bundles=bundles, party_size=party_size, status=status)
    with pytest.raises(ValueError, match=message):
        rank(r, catalog(service(1), service(2)))


def test_nothing_is_mutated():
    cat = catalog(service(1, price="20"), service(2, tags=["fun"]))
    r = result(need(preferred=["fun"]), bundles=[[1], [2]])
    result_before, catalog_before = r.model_copy(deep=True), dict(cat)
    winner = select_winner(r, cat)
    winner.service_ids.append(99)
    assert r == result_before and cat == catalog_before
