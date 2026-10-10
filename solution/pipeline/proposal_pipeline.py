"""Raw client brief to final ProposalOutput: the pipeline's public entry point.

build_proposal() runs the stages in order (parse_brief, search_candidates,
build_output) and is the outer error boundary: whatever fails below it, it
returns a valid ProposalOutput. A model or interpretation failure becomes
request_interpretation_failed; anything else, including a missing API key or
a database error, becomes internal_error. Error messages are fixed text, so
no exception detail, provider response or secret reaches the output.

Each caught exception is logged as one stderr line naming the failed step and
the exception type only: exception messages can carry model output or brief
content (a refusal, a schema error's input values), so they are never logged.
"""

import logging
from collections.abc import Callable
from datetime import date
from functools import cache

import openai
from pydantic import ValidationError

from pipeline.brief_parser import BriefParserError, parse_brief
from pipeline.brief_understanding import REFERENCE_DATE
from pipeline.candidate_search import search_candidates
from pipeline.catalog import Service, load_catalog
from pipeline.proposal_builder import ProposalOutput, build_output, error_output

logger = logging.getLogger(__name__)

INTERPRETATION_FAILED_MESSAGE = "The request could not be interpreted."
INTERNAL_ERROR_MESSAGE = "The proposal could not be built because of an internal error."


def build_proposal(
    brief_text: str,
    *,
    as_of_date: date = REFERENCE_DATE,
    client: openai.OpenAI | None = None,
    model: str | None = None,
    load: Callable[[], dict[int, Service]] = load_catalog,
) -> ProposalOutput:
    """The proposal output for one raw brief; never raises an Exception.

    as_of_date is "today" for both the parser and the past-date check. client
    and model are passed to parse_brief (both default to the configured
    OpenAI settings); load supplies the catalog and is read at most once.
    """
    try:
        understanding = parse_brief(brief_text, reference_date=as_of_date, client=client, model=model)
    except (BriefParserError, ValidationError) as exc:  # ValidationError: a self-contradictory brief
        _log("brief parsing", exc)
        return error_output("request_interpretation_failed", INTERPRETATION_FAILED_MESSAGE)
    except Exception as exc:
        _log("brief parsing", exc)
        return error_output("internal_error", INTERNAL_ERROR_MESSAGE)
    try:
        load_once = cache(load)
        result = search_candidates(understanding, as_of_date, load=load_once)
        return build_output(result, load=load_once)
    except Exception as exc:
        _log("proposal building", exc)
        return error_output("internal_error", INTERNAL_ERROR_MESSAGE)


def _log(step: str, exc: Exception) -> None:
    logger.error("%s failed: %s", step, type(exc).__name__)
