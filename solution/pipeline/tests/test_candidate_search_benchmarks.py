"""Stage C on the real services.db for the challenge and local benchmark briefs.

Each understanding is built by hand to match what the parser extracts for the
brief (see test_brief_parser_live.py), so these tests never call OpenAI.
Dates are interpreted as of the challenge's reference date, 2027-02-01.
"""

from datetime import date

import pytest

from pipeline.brief_understanding import REFERENCE_DATE, BriefUnderstanding
from pipeline.candidate_search import Rejection, search_candidates

AS_OF = date(2027, 2, 1)
assert AS_OF == REFERENCE_DATE


def need(service_type=None, required=(), preferred=(), unmapped=()) -> dict:
    return {
        "service_type": service_type,
        "required_tags": list(required),
        "preference_tags": list(preferred),
        "unmapped_required_terms": list(unmapped),
        "required": True,
    }


def brief(*needs, city, party_size, per_person=None, total=None, dates=None, dietary=None) -> BriefUnderstanding:
    data = BriefUnderstanding.empty().model_dump()
    data.update(
        requested_city=city,
        party_size=party_size,
        budget_per_person_max=per_person,
        budget_total_max=total,
        service_needs=list(needs),
    )
    data["date_constraints"].update(dates or {})
    data["dietary_requirements"].update(dietary or {})
    return BriefUnderstanding.model_validate(data)


BRIEFS = {
    "B01": brief(
        need("activity", required=["go_kart"]),
        need("food"),
        city="Málaga", party_size=10, per_person=80, total=800,
        dates={"year": 2027, "month": 3, "weekday": "friday", "time_of_day": "afternoon"},
    ),
    "B02": brief(
        need("food", required=["dinner"]),
        city="Madrid", party_size=45, per_person=66.66, total=3000, dietary={"vegan_count": 6},
    ),
    "B03": brief(need(preferred=["fun"]), city=None, party_size=None, dates={"year": 2027, "month": 3}),
    "B04": brief(need("activity", required=["go_kart"]), city="Málaga", party_size=40),
    "B05": brief(need(preferred=["premium", "memorable"]), city="Seville", party_size=15, per_person=20, total=300),
    "B06": brief(
        need("activity"), city="Bilbao", party_size=12, dates={"year": 2027, "month": 5, "period_of_month": "late"}
    ),
    "X01": brief(
        need(required=["outdoor", "active"], preferred=["competitive"]),
        city="Barcelona", party_size=20, per_person=75, total=1500,
    ),
    "X02": brief(
        need(required=["indoor"], preferred=["relaxed"]), city="Málaga", party_size=12, per_person=40, total=480
    ),
    "X03": brief(need("activity", required=["padel"]), city="Madrid", party_size=20, per_person=35, total=700),
    "X04": brief(
        need("activity", required=["kayak"]),
        city="Seville", party_size=18, per_person=50, total=900, dates={"year": 2027, "month": 3},
    ),
    "PAINTBALL": brief(need("activity", unmapped=["paintball"]), city="Madrid", party_size=20),
}

EXPECTED = {
    "B01": ("feasible", [[1, 3]]),
    "B02": ("feasible", [[10]]),
    "B03": ("needs_information", []),
    "B04": ("unfulfillable", []),
    "B05": ("feasible", [[21], [23]]),
    "B06": ("unfulfillable", []),
    "X01": ("feasible", [[28], [29]]),
    "X02": ("feasible", [[3], [7]]),
    "X03": ("feasible", [[14]]),
    "X04": ("unfulfillable", []),
    "PAINTBALL": ("unfulfillable", []),
}


def search(brief_id: str):
    return search_candidates(BRIEFS[brief_id], as_of_date=AS_OF)


@pytest.mark.parametrize("brief_id", list(EXPECTED))
def test_status_and_bundles(brief_id):
    result = search(brief_id)
    assert (result.status, result.bundles) == EXPECTED[brief_id]
    assert result.understanding == BRIEFS[brief_id]


def test_b01_budget_is_applied_to_the_combined_bundles():
    without_budget = BRIEFS["B01"].model_copy(update={"budget_per_person_max": None, "budget_total_max": None})
    # Karting candidates [1] and food candidates [3, 4]: tapas (2) needs 15 people.
    assert search_candidates(without_budget, as_of_date=AS_OF).bundles == [[1, 3], [1, 4]]
    assert search("B01").rejections == []  # (1, 4) at €89/person is dropped, and a feasible result carries no rejections


def test_b03_needs_city_and_party_size_without_touching_the_catalog():
    def must_not_load():
        raise AssertionError("the catalog must not be loaded")

    result = search_candidates(BRIEFS["B03"], as_of_date=AS_OF, load=must_not_load)
    assert result.missing_fields == ["requested_city", "party_size"]
    assert result.rejections == []


def test_b04_reports_the_karting_capacity():
    assert Rejection(reason="capacity", need_index=0, service_ids=[1]) in search("B04").rejections


def test_b06_reports_the_unsupported_city():
    assert search("B06").rejections == [Rejection(reason="unsupported_city")]


def test_x04_reports_the_kayak_season():
    assert Rejection(reason="season", need_index=0, service_ids=[19]) in search("X04").rejections


def test_paintball_reports_the_unmapped_term():
    assert search("PAINTBALL").rejections == [Rejection(reason="unmapped_required_term", need_index=0)]
