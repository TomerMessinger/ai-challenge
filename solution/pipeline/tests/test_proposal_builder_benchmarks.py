"""Stage E on the real services.db for the challenge and local benchmark briefs.

The understandings are the hand-built ones Stage C's benchmarks run on, so
these tests never call OpenAI.
"""

from decimal import Decimal

import pytest
from test_candidate_search_benchmarks import AS_OF, BRIEFS

from pipeline.candidate_search import search_candidates
from pipeline.catalog import load_catalog
from pipeline.proposal_builder import ProposalOutput, UnmetRequirement, build_output

# (service_id, need_index) pairs, price per person, total, warning codes.
FULFILLED = {
    "B01": ([(1, 0), (3, 1)], "74", "740", ["date_not_exact"]),
    "B02": ([(10, 0)], "48", "2160", ["date_missing"]),
    "B05": ([(21, 0)], "14", "210", ["date_missing", "soft_preferences_unmatched"]),
    "X01": ([(28, 0)], "69", "1380", ["date_missing"]),
    "X02": ([(7, 0)], "32", "384", ["date_missing"]),
    "X03": ([(14, 0)], "29", "580", ["date_missing"]),
}

# code, need_index, reasons, terms
UNFULFILLABLE = {
    "B04": ("no_feasible_service", 0, ["capacity"], []),
    "B06": ("unsupported_city", None, ["unsupported_city"], []),
    "X04": ("no_feasible_service", 0, ["season"], []),
    "PAINTBALL": ("unsupported_requirement", 0, ["unmapped_required_term"], ["paintball"]),
}


@pytest.fixture(scope="module")
def catalog():
    return load_catalog()


def output_for(brief_id: str) -> ProposalOutput:
    return build_output(search_candidates(BRIEFS[brief_id], as_of_date=AS_OF))


@pytest.mark.parametrize("brief_id", list(FULFILLED))
def test_fulfilled(brief_id, catalog):
    pairs, per_person, total, warnings = FULFILLED[brief_id]
    output = output_for(brief_id)
    assert (output.status, output.request_met, output.understanding) == ("fulfilled", True, BRIEFS[brief_id])
    proposal = output.proposal
    assert [(s.service_id, s.need_index) for s in proposal.services] == pairs
    assert (proposal.price_per_person, proposal.total_price) == (Decimal(per_person), Decimal(total))
    assert [w.code for w in output.issues.warnings] == warnings
    party_size = BRIEFS[brief_id].party_size
    for selected in proposal.services:
        row = catalog[selected.service_id]
        assert (selected.supplier_name, selected.service_name, selected.service_type, selected.description) == (
            row.supplier_name, row.service_name, row.service_type, row.description
        )
        assert selected.quantity == party_size
        assert selected.subtotal == row.price_per_person * party_size


def test_b05_reports_its_unmatched_need_preferences():
    match = output_for("B05").proposal.preference_match
    assert match.score == 0 and match.matched == []
    assert [(ref.scope, ref.tag, ref.need_index) for ref in match.unmatched] == [
        ("service_need", "premium", 0),
        ("service_need", "memorable", 0),
    ]


def test_b03_needs_information():
    output = output_for("B03")
    assert (output.status, output.request_met, output.proposal) == ("needs_information", None, None)
    assert output.issues.missing_information == ["requested_city", "party_size"]


@pytest.mark.parametrize("brief_id", list(UNFULFILLABLE))
def test_unfulfillable(brief_id):
    code, need_index, reasons, terms = UNFULFILLABLE[brief_id]
    output = output_for(brief_id)
    assert (output.status, output.request_met, output.proposal) == ("unfulfillable", False, None)
    assert output.issues.unmet_requirements == [
        UnmetRequirement(code=code, need_index=need_index, reasons=reasons, terms=terms)
    ]
    assert output.issues.warnings == []
    assert "service_ids" not in output.model_dump_json()
