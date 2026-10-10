"""Stage C: deterministic hard-feasibility search for candidate service bundles.

search_candidates() turns a BriefUnderstanding into every bundle of catalog
services that satisfies the brief's hard constraints, in five steps:

  1. Readiness gate: city, party size and at least one required need must be
     known. This is the only path to "needs_information", and it runs before
     the catalog is touched.
  2. Pre-search blockers: a date proven to be past, or a required need with
     unmapped terms, ends the search before SQLite is opened; an unsupported
     city ends it right after the catalog is loaded.
  3. Each required need is searched independently over the city's services,
     filtering on service type, required tags, capacity, diet and season.
  4. The candidate lists are combined into bundles (one service per required
     need, in need order); a service cannot fill two needs.
  5. Budgets are bundle-level ceilings, checked in exact Decimal arithmetic.

Optional needs and all preference tags are ignored here: ranking belongs to
Stage D. Rejections are identifiers only; amounts, terms and limits can be
looked up in the understanding and the catalog.
"""

from collections.abc import Callable
from datetime import date
from decimal import Decimal
from itertools import product
from typing import Literal

from pydantic import BaseModel

from pipeline.brief_understanding import (
    REFERENCE_DATE,
    BriefUnderstanding,
    DateConstraints,
    DietaryRequirements,
    ServiceNeed,
    normalize_for_match,
)
from pipeline.catalog import Service, load_catalog

Status = Literal["needs_information", "unfulfillable", "feasible"]
RejectionReason = Literal[
    "past_date",
    "unmapped_required_term",
    "unsupported_city",
    "service_type",
    "required_tags",
    "capacity",
    "dietary",
    "season",
    "duplicate_service",
    "budget_per_person",
    "budget_total",
]


class Rejection(BaseModel):
    """The first hard check that failed for the brief, a service or a bundle."""

    reason: RejectionReason
    need_index: int | None = None  # position in understanding.service_needs
    service_ids: list[int] = []  # [id] for a candidate service, the whole bundle for a bundle


class CandidateSearchResult(BaseModel):
    status: Status
    understanding: BriefUnderstanding
    bundles: list[list[int]]  # one service ID per required need, in required-need order
    missing_fields: list[str]
    rejections: list[Rejection]


def search_candidates(
    understanding: BriefUnderstanding,
    as_of_date: date = REFERENCE_DATE,
    load: Callable[[], dict[int, Service]] = load_catalog,
) -> CandidateSearchResult:
    """Return every hard-feasible bundle for the brief, or why there is none.

    as_of_date is "today" for the past-date check; it defaults to the
    challenge's reference date. load supplies the catalog and is only called
    once the brief is ready and not already impossible. Catalog errors raise.
    """
    u = understanding
    required = [(index, need) for index, need in enumerate(u.service_needs) if need.required]

    missing = [
        name
        for name, absent in (
            ("requested_city", u.requested_city is None),
            ("party_size", u.party_size is None),
            ("required_service_need", not required),
        )
        if absent
    ]
    if missing:
        return CandidateSearchResult(
            status="needs_information", understanding=u, bundles=[], missing_fields=missing, rejections=[]
        )

    blockers = [Rejection(reason="past_date")] if is_definitely_past(u.date_constraints, as_of_date) else []
    blockers += [
        Rejection(reason="unmapped_required_term", need_index=index)
        for index, need in required
        if need.unmapped_required_terms
    ]
    if blockers:
        return _unfulfillable(u, blockers)

    catalog = load()
    city = normalize_for_match(u.requested_city)
    city_services = [service for service in catalog.values() if normalize_for_match(service.city) == city]
    if not city_services:
        return _unfulfillable(u, [Rejection(reason="unsupported_city")])

    candidate_lists, need_rejections = [], []
    for index, need in required:
        candidates = []
        for service in city_services:
            if reason := _hard_failure(service, need, u):
                need_rejections.append(Rejection(reason=reason, need_index=index, service_ids=[service.id]))
            else:
                candidates.append(service.id)
        candidate_lists.append(candidates)
    if not all(candidate_lists):
        empty = {index for (index, _), candidates in zip(required, candidate_lists) if not candidates}
        return _unfulfillable(u, [r for r in need_rejections if r.need_index in empty])

    bundles, bundle_rejections = [], []
    for bundle in product(*candidate_lists):
        if reason := _bundle_failure(bundle, catalog, u):
            bundle_rejections.append(Rejection(reason=reason, service_ids=list(bundle)))
        else:
            bundles.append(list(bundle))
    if not bundles:
        return _unfulfillable(u, bundle_rejections)
    return CandidateSearchResult(
        status="feasible", understanding=u, bundles=sorted(bundles), missing_fields=[], rejections=[]
    )


def is_definitely_past(dates: DateConstraints, as_of_date: date) -> bool:
    """True only when the stated year, month and day prove the event is before as_of_date.

    Fields are compared from the most significant down; the first unknown one
    leaves the date undecided, so it is not past. Weekday, period of month and
    time of day never prove anything.
    """
    stated = (dates.year, dates.month, dates.day_of_month)
    today = (as_of_date.year, as_of_date.month, as_of_date.day)
    for value, current in zip(stated, today):
        if value is None or value != current:
            return value is not None and value < current
    return False


def _hard_failure(service: Service, need: ServiceNeed, u: BriefUnderstanding) -> RejectionReason | None:
    """The first hard filter the service fails for this need, in the documented order."""
    if need.service_type is not None and service.service_type != need.service_type:
        return "service_type"
    if not set(need.required_tags) <= set(service.tags):
        return "required_tags"
    if not service.min_capacity <= u.party_size <= service.max_capacity:
        return "capacity"
    if service.service_type == "food" and not _meets_diet(service, u.dietary_requirements):
        return "dietary"
    month = u.date_constraints.month
    if month is not None and service.available_months is not None and month not in service.available_months:
        return "season"
    return None


def _meets_diet(service: Service, diet: DietaryRequirements) -> bool:
    """Vegans need a vegan option; vegetarians are served by a vegetarian or a vegan one."""
    options = set(service.dietary_options)
    if (diet.vegan_count or 0) > 0 and "vegan" not in options:
        return False
    if (diet.vegetarian_count or 0) > 0 and not options & {"vegetarian", "vegan"}:
        return False
    return True


def _bundle_failure(
    bundle: tuple[int, ...], catalog: dict[int, Service], u: BriefUnderstanding
) -> RejectionReason | None:
    # Separate needs are separately purchased components, so one row cannot fill two of them.
    if len(set(bundle)) < len(bundle):
        return "duplicate_service"
    per_person = sum(catalog[service_id].price_per_person for service_id in bundle)
    if u.budget_per_person_max is not None and per_person > Decimal(str(u.budget_per_person_max)):
        return "budget_per_person"
    if u.budget_total_max is not None and per_person * u.party_size > Decimal(str(u.budget_total_max)):
        return "budget_total"
    return None


def _unfulfillable(u: BriefUnderstanding, rejections: list[Rejection]) -> CandidateSearchResult:
    return CandidateSearchResult(
        status="unfulfillable", understanding=u, bundles=[], missing_fields=[], rejections=rejections
    )
