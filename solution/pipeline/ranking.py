"""Stage D: deterministic preference ranking and single-winner selection.

Every bundle reaching this stage has already passed Stage C's hard checks
(city, service type, required tags, capacity, diet, season, duplicates and
budgets), so none of them is re-checked here. Bundles are scored on the
client's soft preferences alone, as a plain count of matched preferences:

  - need preferences: +1 for each preference tag of a required need found on
    the service that fills that need (bundle[i] fills the i-th required need;
    optional needs are skipped), never on another need's service;
  - global preferences: +1 for each global preference tag found on at least
    one selected service, however many of them carry it.

evaluate_preferences() is the only implementation of these rules: ranking
takes its score from it, and the proposal output reports its matched and
unmatched preferences.

Bundles rank by higher score, then lower total price, then the smaller
service-ID tuple, so the order never depends on the order Stage C listed them
in. The winner is simply the best-ranked bundle. Prices are recomputed from
the catalog in exact Decimal arithmetic.
"""

from decimal import Decimal
from typing import Literal

from pydantic import BaseModel

from pipeline.brief_understanding import BriefUnderstanding
from pipeline.candidate_search import CandidateSearchResult
from pipeline.catalog import Service


class RankedBundle(BaseModel):
    service_ids: list[int]  # one service ID per required need, in required-need order
    preference_score: int
    price_per_person: Decimal  # EUR, VAT included
    total_price: Decimal  # price_per_person * party_size


class PreferenceRef(BaseModel):
    """One soft preference of the brief: a need's preference tag, or a global one."""

    scope: Literal["service_need", "global"]
    tag: str
    need_index: int | None  # position in understanding.service_needs; None for a global preference


class PreferenceEvaluation(BaseModel):
    score: int  # len(matched): one point per matched preference
    matched: list[PreferenceRef]
    unmatched: list[PreferenceRef]


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
    ranked = [_score(bundle, u, catalog) for bundle in result.bundles]
    return sorted(ranked, key=lambda r: (-r.preference_score, r.total_price, tuple(r.service_ids)))


def select_winner(result: CandidateSearchResult, catalog: dict[int, Service]) -> RankedBundle:
    """The best-ranked bundle: the ranking alone decides the winner."""
    return rank_bundles(result, catalog)[0]


def evaluate_preferences(
    service_ids: list[int], understanding: BriefUnderstanding, catalog: dict[int, Service]
) -> PreferenceEvaluation:
    """Which of the brief's soft preferences a bundle matches, and its score.

    service_ids holds one service per required need, in required-need order.
    Need preferences come first, in need order, then global preferences; both
    lists keep that order. Raises ValueError if the bundle does not hold one
    service per required need or references an unknown service ID.
    """
    required = [(index, need) for index, need in enumerate(understanding.service_needs) if need.required]
    if len(service_ids) != len(required):
        raise ValueError(
            f"bundle {service_ids} has {len(service_ids)} services for {len(required)} required needs"
        )
    if unknown := [service_id for service_id in service_ids if service_id not in catalog]:
        raise ValueError(f"bundle {service_ids} references services missing from the catalog: {unknown}")
    services = [catalog[service_id] for service_id in service_ids]
    checks = [
        (PreferenceRef(scope="service_need", tag=tag, need_index=index), tag in service.tags)
        for (index, need), service in zip(required, services)
        for tag in need.preference_tags
    ]
    checks += [
        (PreferenceRef(scope="global", tag=tag, need_index=None), any(tag in service.tags for service in services))
        for tag in understanding.global_preference_tags
    ]
    matched = [ref for ref, hit in checks if hit]
    unmatched = [ref for ref, hit in checks if not hit]
    return PreferenceEvaluation(score=len(matched), matched=matched, unmatched=unmatched)


def _score(bundle: list[int], u: BriefUnderstanding, catalog: dict[int, Service]) -> RankedBundle:
    score = evaluate_preferences(bundle, u, catalog).score
    per_person = sum(catalog[service_id].price_per_person for service_id in bundle)
    return RankedBundle(
        service_ids=list(bundle),
        preference_score=score,
        price_per_person=per_person,
        total_price=per_person * u.party_size,
    )
