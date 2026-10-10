"""Stage D on the real services.db for the challenge and local benchmark briefs.

The understandings are the hand-built ones Stage C's benchmarks run on, and
each Stage C result is built by hand from that stage's expected bundles, so
these tests never call OpenAI.
"""

from decimal import Decimal

import pytest
from test_candidate_search_benchmarks import AS_OF, BRIEFS

from pipeline.candidate_search import CandidateSearchResult, search_candidates
from pipeline.catalog import load_catalog
from pipeline.ranking import RankedBundle, evaluate_preferences, rank_bundles, select_winner

STAGE_C_BUNDLES = {
    "B01": [[1, 3]],
    "B02": [[10]],
    "B05": [[21], [23]],
    "X01": [[28], [29]],
    "X02": [[3], [7]],
    "X03": [[14]],
}

# Best first: (service_ids, preference_score, price_per_person, total_price).
RANKINGS = {
    "B01": [([1, 3], 0, "74", "740")],
    "B02": [([10], 0, "48", "2160")],
    "B05": [([21], 0, "14", "210"), ([23], 0, "18", "270")],  # neither is premium or memorable: cheaper wins
    "X01": [([28], 1, "69", "1380"), ([29], 0, "35", "700")],  # competitive beats a lower price
    "X02": [([7], 1, "32", "384"), ([3], 0, "19", "228")],  # relaxed beats a lower price
    "X03": [([14], 0, "29", "580")],
}

NOT_FEASIBLE = ["B03", "B04", "B06", "X04", "PAINTBALL"]


@pytest.fixture(scope="module")
def catalog():
    return load_catalog()


def stage_c_result(brief_id: str) -> CandidateSearchResult:
    return CandidateSearchResult(
        status="feasible",
        understanding=BRIEFS[brief_id],
        bundles=STAGE_C_BUNDLES[brief_id],
        missing_fields=[],
        rejections=[],
    )


@pytest.mark.parametrize("brief_id", list(RANKINGS))
def test_ranking_and_winner(brief_id, catalog):
    expected = [
        RankedBundle(
            service_ids=ids, preference_score=score, price_per_person=Decimal(per_person), total_price=Decimal(total)
        )
        for ids, score, per_person, total in RANKINGS[brief_id]
    ]
    result = stage_c_result(brief_id)
    assert rank_bundles(result, catalog) == expected
    assert select_winner(result, catalog) == expected[0]


@pytest.mark.parametrize("brief_id", list(RANKINGS))
def test_preference_evaluation_agrees_with_every_ranking_score(brief_id, catalog):
    for bundle in rank_bundles(stage_c_result(brief_id), catalog):
        evaluation = evaluate_preferences(bundle.service_ids, BRIEFS[brief_id], catalog)
        assert evaluation.score == len(evaluation.matched) == bundle.preference_score


@pytest.mark.parametrize("brief_id", list(RANKINGS))
def test_hand_built_results_match_stage_c(brief_id):
    assert search_candidates(BRIEFS[brief_id], as_of_date=AS_OF) == stage_c_result(brief_id)


@pytest.mark.parametrize("brief_id", NOT_FEASIBLE)
def test_results_that_are_not_feasible_never_enter_ranking(brief_id, catalog):
    result = search_candidates(BRIEFS[brief_id], as_of_date=AS_OF)
    assert result.status != "feasible"
    with pytest.raises(ValueError, match="only a feasible search result can be ranked"):
        select_winner(result, catalog)
