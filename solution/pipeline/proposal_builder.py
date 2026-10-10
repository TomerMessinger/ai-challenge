"""Stage E: the final external JSON for one brief, built deterministically.

build_output() turns a Stage C search result into a ProposalOutput:

  - feasible: Stage D's winner, priced line by line from the catalog, with the
    preferences it matches and deterministic warnings ("fulfilled");
  - needs_information: Stage C's missing fields, copied as they are;
  - unfulfillable: Stage C's terminal rejections, summarized into one concise
    unmet requirement per cause (the search is never rerun).

error_output() is the shape returned when the pipeline itself fails. Every
top-level key and every issues list is always present.

Money stays an exact Decimal in Python and becomes a JSON number, rounded to
the cent, only when the output is serialized. Supplier and service facts
always come from the catalog, never from the language model.
"""

from collections.abc import Callable
from decimal import Decimal
from typing import Annotated, Literal

from pydantic import BaseModel, PlainSerializer

from pipeline.brief_understanding import BriefUnderstanding, DateConstraints
from pipeline.candidate_search import CandidateSearchResult, Rejection
from pipeline.catalog import Service, load_catalog
from pipeline.ranking import PreferenceEvaluation, evaluate_preferences, select_winner

CENT = Decimal("0.01")


def _json_amount(amount: Decimal) -> float:
    return float(amount.quantize(CENT))


# Exact Decimal for arithmetic; a JSON number at cent precision only at the serialization boundary.
Money = Annotated[Decimal, PlainSerializer(_json_amount, return_type=float, when_used="json")]

Status = Literal["fulfilled", "needs_information", "unfulfillable", "error"]

WARNING_MESSAGES = {
    "date_missing": "No date or time of year was specified in the request.",
    "date_not_exact": "The request does not specify an exact calendar date.",
    "soft_preferences_unmatched": "The selected proposal does not match all requested soft preferences.",
    "service_reuse_not_supported": (
        "V1 does not allow one catalog service to satisfy multiple separate required service needs."
    ),
}

# Stage C's checks in the order it runs them; later means the service got further.
_SERVICE_FILTERS = ("service_type", "required_tags", "capacity", "dietary", "season")
_BUNDLE_CHECKS = ("duplicate_service", "budget_per_person", "budget_total")


class SelectedService(BaseModel):
    service_id: int
    need_index: int  # position in understanding.service_needs
    supplier_name: str
    service_name: str
    service_type: str
    description: str
    quantity: int  # every price is per person, so one unit per attendee
    price_per_person: Money
    subtotal: Money  # price_per_person * quantity


class Proposal(BaseModel):
    currency: Literal["EUR"]
    services: list[SelectedService]
    price_per_person: Money
    total_price: Money
    preference_match: PreferenceEvaluation


class UnmetRequirement(BaseModel):
    code: str
    need_index: int | None  # position in understanding.service_needs, when one need is affected
    reasons: list[str]
    terms: list[str]


class WarningInfo(BaseModel):
    code: str
    message: str


class ErrorInfo(BaseModel):
    code: str
    message: str


class Issues(BaseModel):
    missing_information: list[str] = []
    unmet_requirements: list[UnmetRequirement] = []
    warnings: list[WarningInfo] = []


class ProposalOutput(BaseModel):
    status: Status
    request_met: bool | None  # null when the request could not be evaluated
    understanding: BriefUnderstanding | None
    proposal: Proposal | None
    issues: Issues
    error: ErrorInfo | None


def build_output(
    result: CandidateSearchResult, load: Callable[[], dict[int, Service]] = load_catalog
) -> ProposalOutput:
    """The external output for one Stage C result.

    load supplies the catalog and is only called for a feasible result.
    Raises ValueError on internal invariant failures, such as unfulfillable
    rejections that match none of Stage C's terminal cases.
    """
    u = result.understanding
    if result.status == "needs_information":
        return ProposalOutput(
            status="needs_information",
            request_met=None,
            understanding=u,
            proposal=None,
            issues=Issues(missing_information=list(result.missing_fields)),
            error=None,
        )
    if result.status == "unfulfillable":
        unmet, warnings = _summarize_rejections(u, result.rejections)
        return ProposalOutput(
            status="unfulfillable",
            request_met=False,
            understanding=u,
            proposal=None,
            issues=Issues(unmet_requirements=unmet, warnings=warnings),
            error=None,
        )
    return _fulfilled(result, load())


def error_output(code: str, message: str) -> ProposalOutput:
    return ProposalOutput(
        status="error",
        request_met=None,
        understanding=None,
        proposal=None,
        issues=Issues(),
        error=ErrorInfo(code=code, message=message),
    )


def _warning(code: str) -> WarningInfo:
    return WarningInfo(code=code, message=WARNING_MESSAGES[code])


# --- Fulfilled -------------------------------------------------------------------------------

def _fulfilled(result: CandidateSearchResult, catalog: dict[int, Service]) -> ProposalOutput:
    u = result.understanding
    winner = select_winner(result, catalog)
    evaluation = evaluate_preferences(winner.service_ids, u, catalog)
    required = [(index, need) for index, need in enumerate(u.service_needs) if need.required]
    services = [
        _selected_service(catalog[service_id], index, u.party_size)
        for service_id, (index, _) in zip(winner.service_ids, required, strict=True)
    ]
    per_person = sum(service.price_per_person for service in services)
    total = sum(service.subtotal for service in services)
    if (per_person, total) != (winner.price_per_person, winner.total_price):
        raise ValueError(
            f"line items ({per_person}/person, {total} total) disagree with the ranked winner "
            f"({winner.price_per_person}/person, {winner.total_price} total)"
        )
    warnings = _date_warnings(u.date_constraints)
    if evaluation.unmatched:
        warnings.append(_warning("soft_preferences_unmatched"))
    return ProposalOutput(
        status="fulfilled",
        request_met=True,
        understanding=u,
        proposal=Proposal(
            currency="EUR",
            services=services,
            price_per_person=per_person,
            total_price=total,
            preference_match=evaluation,
        ),
        issues=Issues(warnings=warnings),
        error=None,
    )


def _selected_service(service: Service, need_index: int, party_size: int) -> SelectedService:
    return SelectedService(
        service_id=service.id,
        need_index=need_index,
        supplier_name=service.supplier_name,
        service_name=service.service_name,
        service_type=service.service_type,
        description=service.description,
        quantity=party_size,
        price_per_person=service.price_per_person,
        subtotal=service.price_per_person * party_size,
    )


def _date_warnings(dates: DateConstraints) -> list[WarningInfo]:
    """date_missing without any date context, date_not_exact without a full year, month and day.

    time_of_day alone gives no date.
    """
    context = (dates.year, dates.month, dates.weekday, dates.day_of_month, dates.period_of_month)
    if all(value is None for value in context):
        return [_warning("date_missing")]
    if None in (dates.year, dates.month, dates.day_of_month):
        return [_warning("date_not_exact")]
    return []


# --- Unfulfillable ---------------------------------------------------------------------------

def _summarize_rejections(
    u: BriefUnderstanding, rejections: list[Rejection]
) -> tuple[list[UnmetRequirement], list[WarningInfo]]:
    """One unmet requirement per terminal cause, for each of Stage C's four terminal cases."""
    reasons = {rejection.reason for rejection in rejections}
    if reasons and reasons <= {"past_date", "unmapped_required_term"}:
        return [_pre_search_unmet(rejection, u) for rejection in rejections], []
    if reasons == {"unsupported_city"}:
        return [_unmet("unsupported_city", None, ["unsupported_city"], [])], []
    if reasons and reasons <= set(_SERVICE_FILTERS):
        return _empty_needs(rejections), []
    if reasons and reasons <= set(_BUNDLE_CHECKS):
        ordered = [reason for reason in _BUNDLE_CHECKS if reason in reasons]
        warnings = [_warning("service_reuse_not_supported")] if "duplicate_service" in reasons else []
        return [_unmet("no_feasible_bundle", None, ordered, [])], warnings
    raise ValueError(f"unfulfillable result with rejections that match no terminal case: {sorted(reasons)}")


def _pre_search_unmet(rejection: Rejection, u: BriefUnderstanding) -> UnmetRequirement:
    if rejection.reason == "past_date":
        return _unmet("past_date", None, ["past_date"], [])
    terms = [] if rejection.need_index is None else u.service_needs[rejection.need_index].unmapped_required_terms
    if not terms:
        raise ValueError(f"unmapped_required_term rejection without unmapped terms: {rejection}")
    return _unmet("unsupported_requirement", rejection.need_index, ["unmapped_required_term"], terms)


def _empty_needs(rejections: list[Rejection]) -> list[UnmetRequirement]:
    """Each empty need's deepest failing filter, as one no_feasible_service per need."""
    by_need: dict[int, list[str]] = {}
    for rejection in rejections:
        if rejection.need_index is None:
            raise ValueError(f"service rejection without a need_index: {rejection}")
        by_need.setdefault(rejection.need_index, []).append(rejection.reason)
    return [
        _unmet("no_feasible_service", index, [max(by_need[index], key=_SERVICE_FILTERS.index)], [])
        for index in sorted(by_need)
    ]


def _unmet(code: str, need_index: int | None, reasons: list[str], terms: list[str]) -> UnmetRequirement:
    return UnmetRequirement(code=code, need_index=need_index, reasons=reasons, terms=list(terms))
