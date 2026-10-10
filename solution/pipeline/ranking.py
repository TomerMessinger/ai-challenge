"""Stage D: deterministic preference ranking and single-winner selection.

Every bundle reaching this stage has already passed Stage C's hard checks
(city, service type, required tags, capacity, diet, season, duplicates and
budgets), so none of them is re-checked here. Bundles are scored on the
client's soft preferences alone, as a plain count of matched tags:

  - need preferences: +1 for each preference tag of a required need found on
    the service that fills that need (bundle[i] fills the i-th required need;
    optional needs are skipped), never on another need's service;
  - global preferences: +1 for each global preference tag found on at least
    one selected service, however many of them carry it.

Bundles rank by higher score, then lower total price, then the smaller
service-ID tuple, so the order never depends on the order Stage C listed them
in. The winner is simply the best-ranked bundle. Prices are recomputed from
the catalog in exact Decimal arithmetic.
"""

from decimal import Decimal

from pydantic import BaseModel

from pipeline.brief_understanding import ServiceNeed
from pipeline.candidate_search import CandidateSearchResult
from pipeline.catalog import Service


class RankedBundle(BaseModel):
    service_ids: list[int]  # one service ID per required need, in required-need order
    preference_score: int
    price_per_person: Decimal  # EUR, VAT included
    total_price: Decimal  # price_per_person * party_size


def rank_bundles(result: CandidateSearchResult, catalog: dict[int, Service]) -> list[RankedBundle]:
    """Every bundle of a feasible search result, scored and ordered best first.

    Raises ValueError for a result that is not feasible, and for internal
    invariant failures: a feasible result without bundles or party size, a
    bundle that does not hold one service per required need, or an unknown
    service ID.
    """
    u = result.understanding
    if result.status != "feasible":
        raise ValueError(f"only a feasible search result can be ranked, got status {result.status!r}")
    if not result.bundles:
        raise ValueError("a feasible search result has no bundles")
    if u.party_size is None:
        raise ValueError("a feasible search result has no party_size")
    required = [need for need in u.service_needs if need.required]
    ranked = [_score(bundle, required, u.global_preference_tags, u.party_size, catalog) for bundle in result.bundles]
    return sorted(ranked, key=lambda r: (-r.preference_score, r.total_price, tuple(r.service_ids)))


def select_winner(result: CandidateSearchResult, catalog: dict[int, Service]) -> RankedBundle:
    """The best-ranked bundle: the ranking alone decides the winner."""
    return rank_bundles(result, catalog)[0]


def _score(
    bundle: list[int],
    required: list[ServiceNeed],
    global_tags: list[str],
    party_size: int,
    catalog: dict[int, Service],
) -> RankedBundle:
    if len(bundle) != len(required):
        raise ValueError(f"bundle {bundle} has {len(bundle)} services for {len(required)} required needs")
    if unknown := [service_id for service_id in bundle if service_id not in catalog]:
        raise ValueError(f"bundle {bundle} references services missing from the catalog: {unknown}")
    services = [catalog[service_id] for service_id in bundle]
    need_matches = sum(
        tag in service.tags for need, service in zip(required, services) for tag in need.preference_tags
    )
    global_matches = sum(any(tag in service.tags for service in services) for tag in global_tags)
    per_person = sum(service.price_per_person for service in services)
    return RankedBundle(
        service_ids=list(bundle),
        preference_score=need_matches + global_matches,
        price_per_person=per_person,
        total_price=per_person * party_size,
    )
