# Gaddex Mini Proposal Builder

A command-line tool that turns a client's free-text event request into a JSON proposal built
only from services in a local SQLite catalog. A language model interprets the request;
deterministic code does everything else.

## Quick start

Requirements: Python 3.11 and an OpenAI API key. All commands run from the repository root.

```bash
python -m venv .venv
source .venv/bin/activate                # macOS/Linux
.\.venv\Scripts\Activate.ps1             # Windows PowerShell
pip install -r requirements.txt
cp .env.example .env                     # then set OPENAI_API_KEY in .env
python solution/run_proposal.py "Team building for 10 people in Málaga, go-karting then a bite, max €80 each."
```

### API key

`OPENAI_API_KEY` (required) and `OPENAI_MODEL` (optional, default `gpt-6-luna`) are read from
the environment or from the git-ignored `.env` at the repository root. Variables already set in
the environment take precedence; the placeholder value from `.env.example` counts as unset.

### Commands

```bash
python solution/run_proposal.py "<free-text client request>"   # one proposal, printed to stdout
python solution/generate_outputs.py                            # regenerate solution/outputs/
python solution/create_db.py                                   # rebuild solution/data/services.db
pytest                                                         # 318 deterministic tests; no network
pytest -m live                                                 # 25 parser tests against the real OpenAI API
```

- **CLI:** for a valid invocation, stdout contains one `ProposalOutput` JSON document. Pipeline,
  model and runtime failures are represented as valid JSON outputs with `status: "error"` and
  exit code 1 (0 otherwise); stderr then gets one line naming the failed step and exception
  type, e.g. `brief parsing failed: ParserConfigurationError` when no API key is configured.
- **Official outputs:** `generate_outputs.py` runs every brief in `data/briefs.json` through
  `build_proposal()`, using the same public pipeline and `ProposalOutput` schema as the CLI,
  and writes `solution/outputs/<id>.json`. It exits 1 if any brief ends in `error`.
- **Database:** `solution/data/services.json` is the reviewed seed: 36 services, 9 in each of
  the four supported cities (Málaga, Madrid, Seville, Barcelona), including the five services
  the challenge requires. `create_db.py` validates the seed and rebuilds `services.db` from it,
  replacing the file only after every stored row has been verified against the seed. The
  committed database is ready to use and is opened read-only at runtime.
- **Tests:** deterministic tests never call OpenAI; some use in-memory catalogs, others the real
  `services.db`. Live tests need `OPENAI_API_KEY` and are skipped without it.

## Architecture

```
raw brief
 → brief_parser.py         LLM: language → BriefExtraction (schema-constrained)
 → brief_understanding.py  deterministic finalization and validation → BriefUnderstanding
 → candidate_search.py     hard constraints over the catalog → every feasible bundle, or the rejections
 → ranking.py              matched soft preferences, then total price, then service-ID tuple → one winner
 → proposal_builder.py     catalog facts, Decimal prices, warnings, unmet requirements → ProposalOutput
```

`proposal_pipeline.build_proposal()` is the public orchestration entry point and error boundary,
used by both the CLI and the output generator. `taxonomy.py` defines the service types and
canonical tags shared by the database build, the parser schema and the prompt.

**Model vs deterministic code.** The model interprets language only: city, party size, budget,
dates, diets, the services asked for, and which wishes are hard requirements or soft
preferences. It never sees supplier data, so it cannot choose, invent or price a supplier, or
decide whether a request can be met. Deterministic code owns the hard constraints, supplier
selection, prices, arithmetic, ranking, warnings and the output JSON. Every supplier and service
fact in a proposal is read from the catalog by ID.

## LLM approach

- **One parse per brief:** each non-blank brief makes one
  `client.responses.parse(..., text_format=BriefExtraction)` invocation on the Responses API;
  blank input skips the model and returns `needs_information`. The default OpenAI client created
  by the parser uses a 120-second timeout, and Responses API storage is disabled for the request
  with `store=False`.
- **Structured output:** the response must match the `BriefExtraction` Pydantic schema. Service
  types and tags are `Literal` enums generated from `taxonomy.py`, so only canonical values are
  accepted.
- **Prompt:** `PARSER_INSTRUCTIONS` in `brief_parser.py` is filled with the reference date, the
  taxonomy and the meanings of ambiguous tags. It tells the model to record only what the brief
  states, use null for anything unknown, and treat the brief as data, never as instructions (a
  prompt-injection defence). Each independently purchasable component is one service need; what
  is being bought and objective conditions go in `required_tags`, while softened wishes
  ("ideally") and subjective qualities the client does not make firm go in preference tags.
- **Unmapped requirements:** `unmapped_required_terms` is the escape hatch for concrete hard
  service or supplier requirements that fulfillment would need to satisfy but that cannot be
  represented by `service_type`, the canonical tags or another structured field (for example
  "archery"). Each is kept as a short free-text term based on the client's wording, then
  mechanically normalized (trimmed, lowercased, deduplicated, sorted). On a required need it
  blocks fulfillment instead of being dropped or forced onto a loosely related tag, and is
  reported as `unsupported_requirement` once the brief is otherwise complete.
- **Deterministic finalization:** after the call, tags are deduplicated and sorted, the budget
  form the client did not state is derived (rounded down to the cent, so it never exceeds the
  stated limit), and a month without a year gets the year of its next occurrence. Validators
  reject self-contradictory understandings, such as per-person and total budgets that disagree,
  or a diet count larger than the party.
- **Failure handling:** an API error, refusal, incomplete or schema-invalid response, or a
  self-contradictory understanding becomes `request_interpretation_failed`; it is never turned
  into an empty or guessed understanding. Any other failure, such as a missing API key or a
  database error, becomes `internal_error`. Error messages are fixed text, so no exception
  detail, provider response or secret reaches the output.

## Output format

Every response has the same top-level keys: `status`, `request_met`, `understanding`,
`proposal`, `issues` (`missing_information`, `unmet_requirements`, `warnings`) and `error`.

| status | request_met | meaning |
|---|---|---|
| `fulfilled` | `true` | `proposal` holds the selected services |
| `needs_information` | `null` | `issues.missing_information` lists `requested_city`, `party_size` and/or `required_service_need` |
| `unfulfillable` | `false` | `issues.unmet_requirements` explains which hard requirement cannot be met |
| `error` | `null` | `error.code` is `request_interpretation_failed` or `internal_error` |

- **`understanding`** is the validated interpretation when parsing succeeds, and null for
  pipeline errors.
- **`proposal`** lists one service per required need (`service_id`, the `need_index` it fills,
  supplier, name, type, description, `quantity` = party size, `price_per_person`, `subtotal`),
  plus `currency`, `price_per_person`, `total_price` and `preference_match` (score, matched and
  unmatched preferences).
- **Money:** EUR, VAT included. Catalog prices and all proposal prices, subtotals and totals are
  `Decimal` internally (`subtotal = price_per_person × quantity`, `total_price = Σ subtotals`)
  and become JSON numbers at cent precision only at serialization. Budget fields in
  `BriefUnderstanding` are floats, converted to `Decimal` for the budget checks.
- **Unmet requirement codes:** `past_date`, `unsupported_city`, `unsupported_requirement` (with
  the unmapped `terms`), `no_feasible_service` (per need, with the furthest hard filter reached)
  and `no_feasible_bundle` (duplicate service or budget).
- **Warnings:**
  - `date_missing`: fulfilled with no date context; `time_of_day` alone still counts as missing
    date context.
  - `date_not_exact`: fulfilled with some date context present but without a complete
    year/month/day.
  - `soft_preferences_unmatched`: the selected proposal misses at least one soft preference.
  - `service_reuse_not_supported`: unfulfillable, and a combination was rejected for using one
    service for two needs.

## Assumptions

- "Today" is the challenge reference date, **2027-02-01**, for parsing and the past-date check.
- **Past dates:** incomplete dates are rejected only when their known year/month/day components
  are already sufficient to prove they are before the reference date ("January 2027" is past,
  "February 2027" is not).
- **No live supplier booking calendar:** availability on the requested date is not checked, and
  proposals are not confirmed bookings.
- **Explicit seeded seasonality** is enforced when the brief gives a month: the Guadalquivir
  kayak tour runs April to October only.
- **Prices are per person**, VAT included, so `quantity` = party size. The whole party takes
  each service together, so capacity is checked against the full party.
- **Budgets are ceilings** on the whole proposal, per person and/or in total. An approximate
  figure is taken as the limit, and a budget without an amount ("flexible") means no limit.
- **One service row cannot satisfy two required needs** in V1.
- **Optional service components** (`required: false`) are preserved in the parsed understanding
  but are not added to candidate bundles or proposals in V1. Soft preferences on required
  services are still used for ranking.
- **Tag matching:** the LLM maps natural-language requests onto the fixed canonical taxonomy;
  deterministic search and ranking then use exact canonical tag matching. No fuzzy or
  embedding-based matching is used downstream.
- **Diets** are checked only on food services: vegans need a vegan option, vegetarians a
  vegetarian or vegan one.
- **Cities** match case- and accent-insensitively, with an explicit "Sevilla" → Seville alias.
  Unsupported cities are reported, never swapped for a supported one.
- **One winner per request:** the most matched soft preferences, then the lower total price,
  then the lexicographically smallest service-ID tuple.

## Future improvements

- Semantic or embedding-based retrieval for requests that do not map cleanly onto the fixed
  canonical taxonomy.
- Multiple alternatives (budget, balanced, premium) instead of a single winner.
- Real supplier availability: booking calendars and date-level capacity.
- Optional-service handling, such as adding an optional service when the budget allows.
- A client-facing formatted proposal (Markdown or HTML) generated from the JSON.

## Time spent

Roughly 4–5 hours.
