"""Deterministic tests for the parser's prompt, configuration and failure handling.

A fake client stands in for OpenAI, so nothing here touches the network.
"""

from pathlib import Path
from types import SimpleNamespace

import openai
import pytest
from pydantic import ValidationError

from pipeline.brief_parser import (
    DEFAULT_MODEL,
    BriefParserError,
    ParserConfigurationError,
    build_parser_instructions,
    load_config,
    parse_brief,
)
from pipeline.brief_understanding import BriefExtraction, BriefUnderstanding
from pipeline.taxonomy import ALL_TAGS, SERVICE_TYPES, TAG_DESCRIPTIONS

# Config tests point load_config at a file that does not exist, so the real .env is never read.
NO_ENV_FILE = Path(__file__).with_name("no-such-file.env")


class FakeResponses:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def fake_client(result) -> SimpleNamespace:
    return SimpleNamespace(responses=FakeResponses(result))


def fake_response(parsed, *, status="completed", output=(), incomplete_details=None) -> SimpleNamespace:
    return SimpleNamespace(
        output_parsed=parsed, status=status, output=list(output), incomplete_details=incomplete_details
    )


def test_instructions_inject_the_taxonomy_and_reference_date():
    instructions = build_parser_instructions()
    assert "today is 2027-02-01" in instructions
    for name in (*SERVICE_TYPES, *ALL_TAGS, *TAG_DESCRIPTIONS.values()):
        assert name in instructions
    assert "{" not in instructions and "}" not in instructions


@pytest.mark.parametrize("key", [None, "", "your_key_here"])
def test_missing_api_key_is_a_configuration_error(monkeypatch, key):
    if key is None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    else:
        monkeypatch.setenv("OPENAI_API_KEY", key)
    with pytest.raises(ParserConfigurationError, match="OPENAI_API_KEY"):
        load_config(env_file=NO_ENV_FILE)


def test_model_defaults_and_can_be_overridden(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-secret")
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    config = load_config(env_file=NO_ENV_FILE)
    assert config.model == DEFAULT_MODEL
    assert "sk-test-secret" not in repr(config)
    monkeypatch.setenv("OPENAI_MODEL", "some-other-model")
    assert load_config(env_file=NO_ENV_FILE).model == "some-other-model"


def test_parse_brief_sends_the_brief_and_finalizes_the_extraction():
    data = BriefExtraction.empty().model_dump()
    data.update(party_size=10, budget_per_person_max=80)
    data["date_constraints"]["month"] = 3
    client = fake_client(fake_response(BriefExtraction.model_validate(data)))

    understanding = parse_brief("10 of us, €80 each, in March", client=client, model="test-model")

    assert understanding.budget_total_max == 800
    assert understanding.date_constraints.year == 2027
    [call] = client.responses.calls
    assert call["model"] == "test-model"
    assert call["input"] == "10 of us, €80 each, in March"
    assert call["instructions"] == build_parser_instructions()  # the brief never enters the instructions
    assert call["text_format"] is BriefExtraction
    assert call["store"] is False


def _schema_error() -> ValidationError:
    try:
        BriefExtraction.model_validate({})
    except ValidationError as exc:
        return exc
    raise AssertionError("an empty object should not validate")


REFUSAL = SimpleNamespace(type="message", content=[SimpleNamespace(type="refusal", refusal="I can't help with that.")])


@pytest.mark.parametrize(
    "result",
    [
        openai.OpenAIError("connection reset"),
        _schema_error(),
        fake_response(None, output=[REFUSAL]),
        fake_response(None),
        fake_response(None, status="incomplete", incomplete_details=SimpleNamespace(reason="max_output_tokens")),
    ],
    ids=["api-error", "schema-mismatch", "refusal", "no-parsed-output", "incomplete"],
)
def test_model_failure_raises_instead_of_returning_an_empty_understanding(result):
    with pytest.raises(BriefParserError):
        parse_brief("Team dinner for 20 in Madrid", client=fake_client(result), model="test-model")


def test_api_failure_message_hides_sdk_text_but_keeps_the_cause():
    sdk_error = openai.OpenAIError("Incorrect API key provided: sk-proj-****abcd")
    with pytest.raises(BriefParserError) as raised:
        parse_brief("Team dinner for 20 in Madrid", client=fake_client(sdk_error), model="test-model")
    assert str(raised.value) == "OpenAI request failed"
    assert raised.value.__cause__ is sdk_error


def test_blank_brief_is_empty_without_calling_the_model():
    client = fake_client(AssertionError("the model must not be called"))
    assert parse_brief("  \n ", client=client, model="test-model") == BriefUnderstanding.empty()
    assert client.responses.calls == []
