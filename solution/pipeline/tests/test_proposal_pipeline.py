"""Deterministic tests for the raw-text entry point, its error boundary and the scripts using it.

A fake client stands in for OpenAI (both client and model are always passed,
so the real configuration is never read) and the catalog is the real
services.db unless a test breaks it on purpose.
"""

import json
import logging
import sqlite3
import tempfile
from pathlib import Path
from types import SimpleNamespace

import generate_outputs
import openai
import pytest
import run_proposal
from test_brief_parser import REFUSAL, fake_client, fake_response

from pipeline import brief_parser
from pipeline.brief_understanding import BriefExtraction
from pipeline.catalog import load_catalog
from pipeline.proposal_builder import ProposalOutput, error_output
from pipeline.proposal_pipeline import build_proposal

TOP_LEVEL_KEYS = {"status", "request_met", "understanding", "proposal", "issues", "error"}
EMPTY_ISSUES = {"missing_information": [], "unmet_requirements": [], "warnings": []}
OFFICIAL_BRIEFS = Path(__file__).resolve().parents[3] / "data" / "briefs.json"
SECRET = "sk-proj-SECRET-1234"


def b01_extraction() -> BriefExtraction:
    data = BriefExtraction.empty().model_dump()
    data.update(
        requested_city="Málaga",
        party_size=10,
        budget_per_person_max=80,
        service_needs=[
            {"service_type": "activity", "required_tags": ["go_kart"], "preference_tags": [],
             "unmapped_required_terms": [], "required": True},
            {"service_type": "food", "required_tags": [], "preference_tags": [],
             "unmapped_required_terms": [], "required": True},
        ],
    )
    data["date_constraints"].update(month=3, weekday="friday", time_of_day="afternoon")
    return BriefExtraction.model_validate(data)


def run(text="10 of us in Málaga, go-karting then food, max €80 each, a Friday afternoon in March", **kwargs):
    kwargs.setdefault("client", fake_client(fake_response(b01_extraction())))
    return build_proposal(text, model="test-model", **kwargs)


def must_not_load():
    raise AssertionError("the catalog must not be loaded")


def as_json(output: ProposalOutput) -> dict:
    data = json.loads(output.model_dump_json())
    assert set(data) == TOP_LEVEL_KEYS
    return data


def assert_error(output: ProposalOutput, code: str):
    data = as_json(output)
    assert data["status"] == "error" and data["error"]["code"] == code
    assert (data["request_met"], data["understanding"], data["proposal"], data["issues"]) == (
        None, None, None, EMPTY_ISSUES
    )
    return data


# --- Successful runs ------------------------------------------------------------------------

def test_a_brief_runs_end_to_end_into_a_fulfilled_proposal():
    client = fake_client(fake_response(b01_extraction()))
    output = run(client=client)
    data = as_json(output)
    assert (data["status"], data["request_met"], data["error"]) == ("fulfilled", True, None)
    assert [(s["service_id"], s["need_index"]) for s in data["proposal"]["services"]] == [(1, 0), (3, 1)]
    assert (data["proposal"]["price_per_person"], data["proposal"]["total_price"]) == (74.0, 740.0)
    assert data["understanding"]["date_constraints"]["year"] == 2027  # finalized as of 2027-02-01
    assert [w["code"] for w in data["issues"]["warnings"]] == ["date_not_exact"]
    assert len(client.responses.calls) == 1


def test_a_blank_brief_needs_information_without_the_model_or_the_catalog():
    output = run("   ", client=fake_client(AssertionError("the model must not be called")), load=must_not_load)
    data = as_json(output)
    assert (data["status"], data["request_met"], data["proposal"]) == ("needs_information", None, None)
    assert data["issues"]["missing_information"] == ["requested_city", "party_size", "required_service_need"]


def test_the_catalog_is_loaded_once():
    calls = []

    def counting_load():
        calls.append(1)
        return load_catalog()

    assert run(load=counting_load).status == "fulfilled"
    assert len(calls) == 1


# --- Error boundary -------------------------------------------------------------------------

@pytest.mark.parametrize(
    "result",
    [
        openai.OpenAIError(f"Incorrect API key provided: {SECRET}"),
        fake_response(None, output=[REFUSAL]),
        fake_response(None, status="incomplete", incomplete_details=SimpleNamespace(reason="max_output_tokens")),
    ],
    ids=["api-error", "refusal", "incomplete"],
)
def test_model_failures_are_request_interpretation_failures(result):
    output = run(client=fake_client(result), load=must_not_load)
    data = assert_error(output, "request_interpretation_failed")
    assert data["error"]["message"] == "The request could not be interpreted."
    text = output.model_dump_json()
    for leaked in (SECRET, "Incorrect API key", "I can't help with that", "max_output_tokens"):
        assert leaked not in text


def test_a_self_contradictory_brief_is_a_request_interpretation_failure():
    extraction = b01_extraction().model_copy(update={"budget_total_max": 3000})  # 80/person disagrees with 3000/10
    output = run(client=fake_client(fake_response(extraction)), load=must_not_load)
    data = assert_error(output, "request_interpretation_failed")
    assert "inconsistent" not in json.dumps(data)


def test_a_database_failure_is_an_internal_error():
    def broken():
        raise sqlite3.OperationalError(f"unable to open database file C:/private/{SECRET}.db")

    output = run(load=broken)
    data = assert_error(output, "internal_error")
    assert data["error"]["message"] == "The proposal could not be built because of an internal error."
    assert SECRET not in output.model_dump_json() and "database" not in output.model_dump_json()


def test_an_unexpected_client_failure_is_an_internal_error():
    output = run(client=fake_client(RuntimeError(f"boom {SECRET}")), load=must_not_load)
    assert_error(output, "internal_error")
    assert SECRET not in output.model_dump_json()


def test_a_missing_api_key_is_an_internal_error(monkeypatch):
    def no_key():
        raise brief_parser.ParserConfigurationError("OPENAI_API_KEY is not set")

    monkeypatch.setattr(brief_parser, "load_config", no_key)
    output = build_proposal("Team dinner for 20 in Madrid", load=must_not_load)  # no client: config is needed
    assert_error(output, "internal_error")
    assert "OPENAI_API_KEY" not in output.model_dump_json()


def test_failures_log_one_line_with_the_step_and_exception_type_only(caplog):
    def broken():
        raise sqlite3.OperationalError(f"unable to open {SECRET}")

    with caplog.at_level(logging.ERROR, logger="pipeline.proposal_pipeline"):
        run(load=broken)
        run(client=fake_client(openai.OpenAIError(SECRET)), load=must_not_load)
    assert [(r.levelno, r.getMessage(), r.exc_info) for r in caplog.records] == [
        (logging.ERROR, "proposal building failed: OperationalError", None),
        (logging.ERROR, "brief parsing failed: BriefParserError", None),
    ]
    assert SECRET not in caplog.text


def test_error_output_shape():
    data = assert_error(error_output("internal_error", "message"), "internal_error")
    assert data["error"] == {"code": "internal_error", "message": "message"}


# --- CLI and output generation ----------------------------------------------------------------

@pytest.fixture
def output_dir():
    # A plain temporary directory: pytest's tmp_path needs its shared base directory to be accessible.
    with tempfile.TemporaryDirectory() as directory:
        yield Path(directory)


def test_cli_prints_the_pipeline_output(monkeypatch, capsys):
    received = []

    def fake_build(text):
        received.append(text)
        return build_proposal("")

    monkeypatch.setattr(run_proposal, "build_proposal", fake_build)
    assert run_proposal.main(["45 people in Madrid"]) == 0
    assert received == ["45 people in Madrid"]
    printed = capsys.readouterr().out
    assert printed == build_proposal("").model_dump_json(indent=2) + "\n"
    assert set(json.loads(printed)) == TOP_LEVEL_KEYS


def test_cli_exits_1_on_an_error_output(monkeypatch, capsys):
    monkeypatch.setattr(run_proposal, "build_proposal", lambda text: error_output("internal_error", "message"))
    assert run_proposal.main(["anything"]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "error"


def test_generate_outputs_writes_one_valid_file_per_official_brief(monkeypatch, output_dir, capsys):
    briefs = json.loads(OFFICIAL_BRIEFS.read_text(encoding="utf-8"))["briefs"]
    received = []

    def fake_build(text):
        received.append(text)
        return build_proposal("")

    monkeypatch.setattr(generate_outputs, "build_proposal", fake_build)
    assert generate_outputs.main(output_dir=output_dir) == 0
    assert received == [brief["text"] for brief in briefs]
    assert sorted(path.name for path in output_dir.iterdir()) == [f"{brief['id']}.json" for brief in briefs]
    for brief in briefs:
        data = json.loads((output_dir / f"{brief['id']}.json").read_text(encoding="utf-8"))
        assert set(data) == TOP_LEVEL_KEYS
    assert "B01: needs_information" in capsys.readouterr().out


def test_generate_outputs_exits_1_if_any_output_is_an_error(monkeypatch, output_dir):
    monkeypatch.setattr(generate_outputs, "build_proposal", lambda text: error_output("internal_error", "message"))
    assert generate_outputs.main(output_dir=output_dir) == 1
