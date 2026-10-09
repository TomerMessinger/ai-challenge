"""Typed request-understanding models and their deterministic finalization.

The language model fills a BriefExtraction: what the client's brief states,
with nothing derived. finalize_understanding() then applies only mechanical
steps (tag normalization, the second budget representation, the year of a
month) and returns a BriefUnderstanding, whose validators guarantee the
invariants later pipeline stages rely on.

Every field is required on input and always serialized, so an understanding
has a complete shape: unknown scalars are explicit nulls, empty collections
are explicit lists. Service types and tags come from pipeline/taxonomy.py.

A null or empty canonical field means the client did not specify anything
more precise. A concrete requirement the taxonomy cannot represent (say,
paintball) is kept verbatim in its need's unmapped_required_terms rather than
dropped or forced onto a loosely related tag.
"""

import unicodedata
from datetime import date
from decimal import ROUND_FLOOR, Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pipeline.taxonomy import SERVICE_TYPES, TAG_VOCABULARY

# "Today" for every brief in the challenge (README, Step 3).
REFERENCE_DATE = date(2027, 2, 1)

# Canonical tags in vocabulary order, which is also the order tag lists are sorted into.
CANONICAL_TAGS = tuple(tag for tags in TAG_VOCABULARY.values() for tag in tags)
_TAG_ORDER = {tag: position for position, tag in enumerate(CANONICAL_TAGS)}

# Literal types built at runtime from the taxonomy, so the schema can never drift from it.
ServiceType = Literal[SERVICE_TYPES]
Tag = Literal[CANONICAL_TAGS]
Weekday = Literal["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
PeriodOfMonth = Literal["early", "mid", "late"]
TimeOfDay = Literal["morning", "afternoon", "evening", "night"]

# Explicit per-person and total budgets agree if they differ by at most a cent
# per person or 1%, whichever is larger; anything more is a contradiction.
BUDGET_TOLERANCE = 0.01
_CENT = Decimal("0.01")

# Tiny, explicit aliases applied to whole match keys; nothing is stemmed or guessed.
_MATCH_ALIASES = {"sevilla": "seville"}


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DateConstraints(_Model):
    """When the event should happen, as far as the brief says."""

    year: int | None
    month: int | None = Field(ge=1, le=12)
    weekday: Weekday | None
    day_of_month: int | None = Field(ge=1, le=31)
    period_of_month: PeriodOfMonth | None
    time_of_day: TimeOfDay | None


class ServiceNeed(_Model):
    """One independently purchasable component the client asks for."""

    service_type: ServiceType | None
    required_tags: list[Tag]
    preference_tags: list[Tag]
    unmapped_required_terms: list[str]
    required: bool


class DietaryRequirements(_Model):
    """Attendee counts per diet, as stated in the brief."""

    vegan_count: int | None = Field(ge=0)
    vegetarian_count: int | None = Field(ge=0)


class _BriefFields(_Model):
    requested_city: str | None
    party_size: int | None = Field(gt=0)
    budget_total_max: float | None = Field(gt=0)
    budget_per_person_max: float | None = Field(gt=0)
    date_constraints: DateConstraints
    service_needs: list[ServiceNeed]
    dietary_requirements: DietaryRequirements
    global_preference_tags: list[Tag]

    @classmethod
    def empty(cls):
        """The brief that states nothing: every scalar null, every list empty."""
        return cls(
            requested_city=None,
            party_size=None,
            budget_total_max=None,
            budget_per_person_max=None,
            date_constraints=DateConstraints(**dict.fromkeys(DateConstraints.model_fields)),
            service_needs=[],
            dietary_requirements=DietaryRequirements(**dict.fromkeys(DietaryRequirements.model_fields)),
            global_preference_tags=[],
        )


class BriefExtraction(_BriefFields):
    """What the client's brief states, before anything is derived from it."""


class BriefUnderstanding(_BriefFields):
    """The validated, finalized understanding of one client brief."""

    @model_validator(mode="after")
    def _check_invariants(self) -> "BriefUnderstanding":
        city = self.requested_city
        if city is not None and (not city or city != " ".join(city.split())):
            raise ValueError(f"requested_city must be null or trimmed, non-blank text, got {city!r}")
        for need in self.service_needs:
            _check_unique("required_tags", need.required_tags)
            _check_unique("preference_tags", need.preference_tags)
            if overlap := set(need.required_tags) & set(need.preference_tags):
                raise ValueError(f"tags {sorted(overlap)} are both required and preferred in one service need")
            if need.unmapped_required_terms != _normalize_terms(need.unmapped_required_terms):
                raise ValueError(
                    "unmapped_required_terms must be trimmed, lowercase, non-blank, unique and sorted, "
                    f"got {need.unmapped_required_terms}"
                )
        _check_unique("global_preference_tags", self.global_preference_tags)
        if self.party_size is not None:
            self._check_budgets_agree()
            for diet, count in self.dietary_requirements:
                if count is not None and count > self.party_size:
                    raise ValueError(f"{diet} ({count}) exceeds party_size ({self.party_size})")
        return self

    def _check_budgets_agree(self) -> None:
        total, per_person = self.budget_total_max, self.budget_per_person_max
        if total is None or per_person is None:
            return
        implied = total / self.party_size
        if abs(per_person - implied) > max(0.01, BUDGET_TOLERANCE * implied):
            raise ValueError(
                f"budget_per_person_max ({per_person}) is inconsistent with budget_total_max ({total}) "
                f"for party_size {self.party_size} (implies {implied:.2f} per person)"
            )


def _check_unique(name: str, tags: list[str]) -> None:
    if len(set(tags)) != len(tags):
        raise ValueError(f"{name} must not repeat a tag, got {tags}")


def normalize_for_match(value: str) -> str:
    """Comparison key for free text such as a city: 'Málaga' and ' MALAGA ' both give 'malaga'.

    Removes accents, casefolds and collapses whitespace, then applies only the
    explicit aliases in _MATCH_ALIASES (so 'Sevilla' gives 'seville').
    """
    decomposed = unicodedata.normalize("NFKD", value)
    without_accents = "".join(char for char in decomposed if not unicodedata.combining(char))
    key = " ".join(without_accents.casefold().split())
    return _MATCH_ALIASES.get(key, key)


def finalize_understanding(extraction: BriefExtraction, reference_date: date = REFERENCE_DATE) -> BriefUnderstanding:
    """Apply the deterministic steps that follow extraction and validate the result.

    Nothing is re-interpreted here: tags are deduplicated and sorted (a tag that
    is both required and preferred stays required), unmapped terms get a stable
    form without being mapped to anything, the missing budget
    representation and the year of a month are derived, and explicitly stated
    values are never overwritten. requested_city keeps its display form; use
    normalize_for_match() to compare it.

    Raises pydantic.ValidationError when the brief contradicts itself, e.g.
    per-person and total budgets that disagree with the party size.
    """
    data = extraction.model_dump()
    if extraction.requested_city is not None:
        data["requested_city"] = " ".join(extraction.requested_city.split()) or None
    data["service_needs"] = [_normalize_need(need) for need in extraction.service_needs]
    data["global_preference_tags"] = _canonical(extraction.global_preference_tags)
    data["budget_total_max"], data["budget_per_person_max"] = _derive_budgets(
        extraction.budget_total_max, extraction.budget_per_person_max, extraction.party_size
    )
    data["date_constraints"]["year"] = _derive_year(extraction.date_constraints, reference_date)
    return BriefUnderstanding.model_validate(data)


def _normalize_need(need: ServiceNeed) -> dict:
    required = _canonical(need.required_tags)
    preferred = [tag for tag in _canonical(need.preference_tags) if tag not in required]
    return {
        **need.model_dump(),
        "required_tags": required,
        "preference_tags": preferred,
        "unmapped_required_terms": _normalize_terms(need.unmapped_required_terms),
    }


def _canonical(tags: list[str]) -> list[str]:
    return sorted(set(tags), key=_TAG_ORDER.__getitem__)


def _normalize_terms(terms: list[str]) -> list[str]:
    """Trim, collapse whitespace and lowercase free-text terms, then deduplicate and sort them.

    Purely mechanical: 'Paintball' and ' paintball ' become one 'paintball',
    but no term is stemmed, aliased or mapped to a canonical tag.
    """
    return sorted({normalized for term in terms if (normalized := " ".join(term.split()).lower())})


def _derive_budgets(
    total: float | None, per_person: float | None, party_size: int | None
) -> tuple[float | None, float | None]:
    """Fill in the budget representation the client did not state.

    Derived amounts are rounded down to the cent: a derived ceiling never
    exceeds what the client stated, and nothing is lost when it is compared
    with prices in whole cents.
    """
    if party_size is None:
        return total, per_person
    if total is None and per_person is not None:
        total = _floor_to_cent(Decimal(str(per_person)) * party_size)
    elif per_person is None and total is not None:
        per_person = _floor_to_cent(Decimal(str(total)) / party_size)
    return total, per_person


def _floor_to_cent(amount: Decimal) -> float:
    return float(amount.quantize(_CENT, rounding=ROUND_FLOOR))


def _derive_year(dates: DateConstraints, reference_date: date) -> int | None:
    """Year of the next occurrence of a month stated without one, counted from reference_date.

    The reference month itself still counts as upcoming unless an explicit
    day_of_month has already passed.
    """
    if dates.year is not None or dates.month is None:
        return dates.year
    passed = (dates.month, dates.day_of_month or reference_date.day) < (reference_date.month, reference_date.day)
    return reference_date.year + 1 if passed else reference_date.year
