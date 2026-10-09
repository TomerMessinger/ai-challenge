"""Create the Gaddex mock supplier database from its canonical JSON seed.

``data/services.json`` (next to this script) is the source of truth: 36
human-reviewable mock services, nine per city. ``data/services.db`` is a
derived artifact that is rebuilt from scratch on every run, so running the
script repeatedly always produces the same logical contents.

Steps:
  1. Load services.json and check every invariant: field types and rules,
     city/ID allocation, the controlled tag vocabulary and the five services
     the challenge requires.
  2. Build the database in a temporary file, inserting rows in ID order.
  3. Reopen that file read-only and check that every stored row matches the
     seed values.
  4. Atomically replace services.db with the verified file and print an audit.

Any violation stops the script with exit code 1 and leaves services.db as it was.

Prices are EUR per person, VAT included. List fields (tags, dietary_options,
available_months) are stored as JSON text; wheelchair_accessible as 1/0/NULL.

Usage: python solution/create_db.py
"""

import json
import math
import os
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path

from pipeline.taxonomy import ALL_TAGS, DIETARY_OPTIONS, SERVICE_TYPES, TAG_DESCRIPTIONS, TAG_VOCABULARY

SOLUTION_DIR = Path(__file__).resolve().parent
SEED_PATH = SOLUTION_DIR / "data" / "services.json"
DB_PATH = SOLUTION_DIR / "data" / "services.db"

CITIES = ("Málaga", "Madrid", "Seville", "Barcelona")
SERVICES_PER_CITY = 9
EXPECTED_TOTAL = len(CITIES) * SERVICES_PER_CITY
# Stable ID blocks: Málaga 1-9, Madrid 10-18, Seville 19-27, Barcelona 28-36.
CITY_ID_BLOCKS = {
    city: range(i * SERVICES_PER_CITY + 1, (i + 1) * SERVICES_PER_CITY + 1)
    for i, city in enumerate(CITIES)
}

FIELDS = (
    "id",
    "supplier_name",
    "city",
    "service_name",
    "service_type",
    "description",
    "price_per_person",
    "min_capacity",
    "max_capacity",
    "duration_minutes",
    "tags",
    "dietary_options",
    "wheelchair_accessible",
    "available_months",
    "notes",
)

TAG_PATTERN = re.compile(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*")

# The five services the challenge requires, with the structured facts implied
# by their notes. wheelchair_accessible is None wherever the challenge says
# nothing about access, so accessibility is never invented for these rows.
MANDATORY_SERVICES = (
    {
        "supplier_name": "Karting Costa del Sol",
        "city": "Málaga",
        "service_name": "Go-kart Grand Prix (2h)",
        "price_per_person": 55,
        "min_capacity": 8,
        "max_capacity": 20,
        "notes": "Outdoor",
        "service_type": "activity",
        "duration_minutes": 120,
        "wheelchair_accessible": None,
        "available_months": None,
        "required_tags": ("go_kart", "outdoor"),
    },
    {
        "supplier_name": "La Malagueña Catering",
        "city": "Málaga",
        "service_name": "Málaga-style tapas menu",
        "price_per_person": 22,
        "min_capacity": 15,
        "max_capacity": 80,
        "notes": "Minimum 15 people. Vegetarian and vegan options",
        "service_type": "food",
        "dietary_options": ["vegan", "vegetarian"],
        "wheelchair_accessible": None,
        "available_months": None,
        "required_tags": ("tapas",),
    },
    {
        "supplier_name": "Restaurante El Patio",
        "city": "Madrid",
        "service_name": "Group dinner in private room",
        "price_per_person": 48,
        "min_capacity": 10,
        "max_capacity": 60,
        "notes": "Vegan option",
        "service_type": "food",
        "dietary_options": ["vegan", "vegetarian"],  # "Vegan option"; a vegan menu also serves vegetarians
        "wheelchair_accessible": None,
        "available_months": None,
        "required_tags": ("dinner", "private_room"),
    },
    {
        "supplier_name": "The Locked House",
        "city": "Madrid",
        "service_name": "Team escape room (1h30)",
        "price_per_person": 24,
        "min_capacity": 12,
        "max_capacity": 30,
        "notes": "Not wheelchair accessible",
        "service_type": "activity",
        "duration_minutes": 90,
        "wheelchair_accessible": False,
        "available_months": None,
        "required_tags": ("escape_room",),
    },
    {
        "supplier_name": "Guadalquivir Kayak",
        "city": "Seville",
        "service_name": "River kayak tour (2h)",
        "price_per_person": 35,
        "min_capacity": 6,
        "max_capacity": 40,
        "notes": "April to October only",
        "service_type": "activity",
        "duration_minutes": 120,
        "wheelchair_accessible": None,
        "available_months": [4, 5, 6, 7, 8, 9, 10],
        "required_tags": ("kayak",),
    },
)


def _sql_list(values: tuple[str, ...]) -> str:
    return ", ".join(f"'{value}'" for value in values)


SCHEMA = f"""
CREATE TABLE services (
    id                    INTEGER PRIMARY KEY,
    supplier_name         TEXT    NOT NULL CHECK (trim(supplier_name) <> ''),
    city                  TEXT    NOT NULL CHECK (city IN ({_sql_list(CITIES)})),
    service_name          TEXT    NOT NULL CHECK (trim(service_name) <> ''),
    service_type          TEXT    NOT NULL CHECK (service_type IN ({_sql_list(SERVICE_TYPES)})),
    description           TEXT    NOT NULL CHECK (trim(description) <> ''),
    price_per_person      REAL    NOT NULL CHECK (price_per_person > 0),
    min_capacity          INTEGER NOT NULL CHECK (min_capacity >= 1),
    max_capacity          INTEGER NOT NULL CHECK (max_capacity >= min_capacity),
    duration_minutes      INTEGER CHECK (duration_minutes IS NULL OR duration_minutes > 0),
    tags                  TEXT    NOT NULL CHECK (json_type(tags) = 'array'),
    dietary_options       TEXT    NOT NULL CHECK (json_type(dietary_options) = 'array'),
    wheelchair_accessible INTEGER CHECK (wheelchair_accessible IS NULL OR wheelchair_accessible IN (0, 1)),
    available_months      TEXT    CHECK (available_months IS NULL OR json_type(available_months) = 'array'),
    notes                 TEXT
);
"""
INSERT_SQL = f"INSERT INTO services ({', '.join(FIELDS)}) VALUES ({', '.join('?' * len(FIELDS))})"


class BuildError(Exception):
    """An invariant violation that stops the build; details are shown as a list."""

    def __init__(self, message: str, details: list[str] | None = None):
        super().__init__(message)
        self.details = details or []


# --- Loading -----------------------------------------------------------------

def load_seed(path: Path) -> object:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise BuildError(f"{path.name} is not valid UTF-8: {exc}") from exc
    except OSError as exc:
        raise BuildError(f"cannot read {path}: {exc.strerror or exc}") from exc
    try:
        return json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise BuildError(f"{path.name} is not valid JSON: {exc.msg} (line {exc.lineno}, column {exc.colno})") from exc
    except ValueError as exc:
        raise BuildError(f"{path.name}: {exc}") from exc


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict:
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError(f"duplicate key {key!r} in a JSON object")
        obj[key] = value
    return obj


# --- Validation --------------------------------------------------------------

def validate_services(services: object) -> list[str]:
    """Return every invariant violation in the seed (an empty list means valid)."""
    if not isinstance(services, list):
        return [f"top level must be a JSON array of services, got {type(services).__name__}"]
    problems = []
    if len(services) != EXPECTED_TOTAL:
        problems.append(f"expected exactly {EXPECTED_TOTAL} services, found {len(services)}")
    row_problems = [p for position, service in enumerate(services) for p in _service_problems(position, service)]
    if row_problems:
        return problems + row_problems  # the cross-row checks below assume well-formed rows
    return problems + _dataset_problems(services) + _vocabulary_problems() + _mandatory_problems(services)


def _service_problems(position: int, service: object) -> list[str]:
    if not isinstance(service, dict):
        return [f"entry {position}: must be a JSON object, got {type(service).__name__}"]
    problems = []
    missing = [field for field in FIELDS if field not in service]
    unknown = sorted(set(service) - set(FIELDS))
    if missing:
        problems.append(f"missing field(s): {', '.join(missing)}")
    if unknown:
        problems.append(f"unknown field(s): {', '.join(unknown)}")
    for field, check in FIELD_CHECKS.items():
        if field in service:
            problems += [f"{field} {problem}" for problem in check(service[field])]
    if not missing:
        problems += _cross_field_problems(service)
    label = f"entry {position} (id={service.get('id')!r})"
    return [f"{label}: {problem}" for problem in problems]


def _cross_field_problems(service: dict) -> list[str]:
    problems = []
    low, high = service["min_capacity"], service["max_capacity"]
    if _is_int(low) and _is_int(high) and high < low:
        problems.append(f"max_capacity ({high}) must be >= min_capacity ({low})")
    dietary = service["dietary_options"]
    if isinstance(dietary, list) and "vegan" in dietary and "vegetarian" not in dietary:
        problems.append("dietary_options lists vegan without vegetarian; a vegan option also serves vegetarians")
    return problems


def _dataset_problems(services: list[dict]) -> list[str]:
    problems = []
    ids = [service["id"] for service in services]
    for duplicate in _duplicates(ids):
        problems.append(f"id {duplicate} is used by more than one service")
    expected_ids = set(range(1, EXPECTED_TOTAL + 1))
    if missing := sorted(expected_ids - set(ids)):
        problems.append(f"ids must be exactly 1-{EXPECTED_TOTAL}; missing {missing}")
    if unexpected := sorted(set(ids) - expected_ids):
        problems.append(f"ids must be exactly 1-{EXPECTED_TOTAL}; unexpected {unexpected}")
    per_city = Counter(service["city"] for service in services)
    for city in CITIES:
        if per_city[city] != SERVICES_PER_CITY:
            problems.append(f"{city} must have exactly {SERVICES_PER_CITY} services, found {per_city[city]}")
    for service in services:
        block = CITY_ID_BLOCKS[service["city"]]
        if service["id"] not in block:
            problems.append(
                f"id {service['id']} is in {service['city']}, whose stable ID block is {block.start}-{block.stop - 1}"
            )
    offerings = Counter((s["city"], s["supplier_name"], s["service_name"]) for s in services)
    for (city, supplier, name), count in offerings.items():
        if count > 1:
            problems.append(f"{supplier} / {name} ({city}) appears {count} times")
    return problems


def _vocabulary_problems() -> list[str]:
    listed = [tag for tags in TAG_VOCABULARY.values() for tag in tags]
    problems = [f"TAG_VOCABULARY lists {duplicate!r} more than once" for duplicate in _duplicates(listed)]
    for tag in sorted(set(TAG_DESCRIPTIONS) - ALL_TAGS):
        problems.append(f"TAG_DESCRIPTIONS describes {tag!r}, which is not in TAG_VOCABULARY")
    return problems


def _mandatory_problems(services: list[dict]) -> list[str]:
    problems = []
    for spec in MANDATORY_SERVICES:
        label = f"mandatory service {spec['supplier_name']} / {spec['service_name']}"
        matches = _find_mandatory(services, spec)
        if len(matches) != 1:
            problems.append(f"{label}: expected exactly one row with this supplier and service name, found {len(matches)}")
            continue
        row = matches[0]
        for field, expected in spec.items():
            if field == "required_tags":
                if missing := [tag for tag in expected if tag not in row["tags"]]:
                    problems.append(f"{label}: tags must include {missing}")
            elif not _same(row[field], expected):
                problems.append(f"{label}: {field} must be {expected!r}, got {row[field]!r}")
    return problems


def _same(actual: object, expected: object) -> bool:
    """Equality where lists compare as sets: list order carries no meaning here."""
    if isinstance(actual, list) and isinstance(expected, list):
        return set(actual) == set(expected)
    return actual == expected


def _find_mandatory(services: list[dict], spec: dict) -> list[dict]:
    return [
        service
        for service in services
        if service["supplier_name"] == spec["supplier_name"] and service["service_name"] == spec["service_name"]
    ]


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _duplicates(values: list) -> list:
    counts = Counter(value for value in values if isinstance(value, (str, int)))
    return sorted((value for value, count in counts.items() if count > 1), key=repr)


def _check_text(value: object, nullable: bool = False) -> list[str]:
    if value is None and nullable:
        return []
    if not isinstance(value, str):
        return [f"must be a string{' or null' if nullable else ''}, got {value!r}"]
    if not value.strip():
        return ["must not be blank" + (" (use null instead)" if nullable else "")]
    if value != value.strip():
        return ["must not have leading or trailing whitespace"]
    return []


def _check_notes(value: object) -> list[str]:
    return _check_text(value, nullable=True)


def _check_city(value: object) -> list[str]:
    return _check_text(value) or _check_one_of(value, CITIES)


def _check_service_type(value: object) -> list[str]:
    return _check_one_of(value, SERVICE_TYPES)


def _check_one_of(value: object, allowed: tuple[str, ...]) -> list[str]:
    return [] if value in allowed else [f"must be one of {', '.join(allowed)}; got {value!r}"]


def _check_positive_int(value: object) -> list[str]:
    return [] if _is_int(value) and value >= 1 else [f"must be an integer >= 1, got {value!r}"]


def _check_price(value: object) -> list[str]:
    if not (isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)):
        return [f"must be a number, got {value!r}"]
    if value <= 0:
        return [f"must be > 0, got {value!r}"]
    if round(value, 2) != value:
        return [f"must have at most 2 decimal places (EUR cents), got {value!r}"]
    return []


def _check_duration(value: object) -> list[str]:
    if value is None or (_is_int(value) and value > 0):
        return []
    return [f"must be a positive integer number of minutes or null, got {value!r}"]


def _check_tags(value: object) -> list[str]:
    if not isinstance(value, list):
        return [f"must be a JSON array, got {value!r}"]
    if not value:
        return ["must contain at least one tag"]
    problems = []
    for tag in value:
        if not isinstance(tag, str) or not TAG_PATTERN.fullmatch(tag):
            problems.append(f"contains {tag!r}, which is not a lowercase snake_case string")
        elif tag not in ALL_TAGS:
            problems.append(
                f"contains {tag!r}, which is not in TAG_VOCABULARY (reuse an existing tag or add it to pipeline/taxonomy.py deliberately)"
            )
    if duplicates := _duplicates(value):
        problems.append(f"must not repeat a tag: {duplicates}")
    return problems


def _check_dietary(value: object) -> list[str]:
    if not isinstance(value, list):
        return [f"must be a JSON array, got {value!r}"]
    if unsupported := [option for option in value if option not in DIETARY_OPTIONS]:
        return [f"contains unsupported value(s) {unsupported!r}; allowed: {', '.join(DIETARY_OPTIONS)}"]
    if len(set(value)) != len(value):
        return [f"must not contain duplicates, got {value!r}"]
    return []


def _check_wheelchair(value: object) -> list[str]:
    return [] if value is None or isinstance(value, bool) else [f"must be true, false or null, got {value!r}"]


def _check_months(value: object) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not value:
        return [f"must be null (no known seasonal restriction) or a non-empty list of months, got {value!r}"]
    if invalid := [month for month in value if not _is_int(month) or not 1 <= month <= 12]:
        return [f"contains invalid month(s) {invalid!r}; months are integers 1-12"]
    if len(set(value)) != len(value):
        return [f"must not contain duplicates, got {value!r}"]
    if len(value) == 12:
        return ["lists all 12 months; use null for 'no seasonal restriction'"]
    return []


FIELD_CHECKS = {
    "id": _check_positive_int,
    "supplier_name": _check_text,
    "city": _check_city,
    "service_name": _check_text,
    "service_type": _check_service_type,
    "description": _check_text,
    "price_per_person": _check_price,
    "min_capacity": _check_positive_int,
    "max_capacity": _check_positive_int,
    "duration_minutes": _check_duration,
    "tags": _check_tags,
    "dietary_options": _check_dietary,
    "wheelchair_accessible": _check_wheelchair,
    "available_months": _check_months,
    "notes": _check_notes,
}


# --- Database ----------------------------------------------------------------

def build_database(services: list[dict]) -> list[dict]:
    """Rebuild services.db from validated rows and return the rows read back from it."""
    tmp_path = DB_PATH.with_name(DB_PATH.name + ".tmp")
    try:
        _write_database(tmp_path, services)
        stored = read_services(tmp_path)
        if mismatches := persistence_mismatches(services, stored):
            raise BuildError("rows read back from SQLite differ from services.json; services.db was not modified", mismatches)
        try:
            os.replace(tmp_path, DB_PATH)
        except PermissionError as exc:
            raise BuildError(
                f"cannot replace {DB_PATH} ({exc.strerror}); close any program that has it open and re-run"
            ) from exc
    except sqlite3.Error as exc:
        raise BuildError(f"SQLite error while building the database: {exc}") from exc
    finally:
        _remove_database_files(tmp_path)
    return stored


def _write_database(path: Path, services: list[dict]) -> None:
    _remove_database_files(path)  # leftovers from an interrupted run
    conn = sqlite3.connect(path)
    try:
        conn.executescript(SCHEMA)
        conn.executemany(INSERT_SQL, [_to_db_row(s) for s in sorted(services, key=lambda s: s["id"])])
        conn.commit()
    finally:
        conn.close()


def _remove_database_files(path: Path) -> None:
    for candidate in (path, path.with_name(path.name + "-journal")):
        candidate.unlink(missing_ok=True)


def read_services(path: Path) -> list[dict]:
    """Read every row from a services database back into seed-shaped dicts."""
    conn = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
    try:
        rows = conn.execute(f"SELECT {', '.join(FIELDS)} FROM services ORDER BY id").fetchall()
    finally:
        conn.close()
    return [_from_db_row(row) for row in rows]


def persistence_mismatches(services: list[dict], stored: list[dict]) -> list[str]:
    """Compare the decoded stored rows with the seed, field by field."""
    expected = {service["id"]: service for service in services}
    actual = {service["id"]: service for service in stored}
    problems = []
    if len(stored) != len(services):
        problems.append(f"database has {len(stored)} rows, services.json has {len(services)}")
    for service_id in sorted(expected.keys() | actual.keys()):
        want, got = expected.get(service_id), actual.get(service_id)
        if want is None or got is None:
            problems.append(f"id {service_id} exists only in {'the database' if want is None else 'services.json'}")
            continue
        for field in FIELDS:
            if got[field] != want[field]:
                problems.append(f"id {service_id}: {field} stored as {got[field]!r}, expected {want[field]!r}")
    return problems


def _to_db_row(service: dict) -> tuple:
    wheelchair, months = service["wheelchair_accessible"], service["available_months"]
    values = {
        **service,
        "price_per_person": float(service["price_per_person"]),
        "tags": _json_text(service["tags"]),
        "dietary_options": _json_text(service["dietary_options"]),
        "wheelchair_accessible": None if wheelchair is None else int(wheelchair),
        "available_months": None if months is None else _json_text(months),
    }
    return tuple(values[field] for field in FIELDS)


def _from_db_row(row: tuple) -> dict:
    service = dict(zip(FIELDS, row))
    service["tags"] = json.loads(service["tags"])
    service["dietary_options"] = json.loads(service["dietary_options"])
    if service["wheelchair_accessible"] is not None:
        service["wheelchair_accessible"] = bool(service["wheelchair_accessible"])
    if service["available_months"] is not None:
        service["available_months"] = json.loads(service["available_months"])
    return service


def _json_text(value: list) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


# --- Audit -------------------------------------------------------------------

def print_audit(stored: list[dict]) -> None:
    by_city = Counter(service["city"] for service in stored)
    by_type = Counter(service["service_type"] for service in stored)
    tag_uses = Counter(tag for service in stored for tag in service["tags"])
    print(f"Validated {SEED_PATH.name}: {len(stored)} services, all invariants hold")
    print(f"Rebuilt {DB_PATH.name}: every row read back matches the seed")
    print("Rows by city: " + ", ".join(f"{city} {by_city[city]}" for city in CITIES))
    print("Rows by type: " + ", ".join(f"{kind} {by_type[kind]}" for kind in SERVICE_TYPES))
    print(f"Tag vocabulary ({len(ALL_TAGS)} tags, with number of services using each):")
    for group, tags in TAG_VOCABULARY.items():
        print(f"  {group + ':':<12}" + ", ".join(f"{tag} {tag_uses[tag]}" for tag in tags))
    print("Mandatory challenge services (all required facts verified):")
    for spec in MANDATORY_SERVICES:
        row = _find_mandatory(stored, spec)[0]
        print(f"  id {row['id']:>2}  {spec['supplier_name']} / {spec['service_name']}")


def main() -> int:
    try:
        services = load_seed(SEED_PATH)
        if problems := validate_services(services):
            raise BuildError(
                f"{SEED_PATH.name} failed validation with {len(problems)} problem(s); services.db was not modified",
                problems,
            )
        stored = build_database(services)
    except BuildError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        for detail in exc.details:
            print(f"  - {detail}", file=sys.stderr)
        return 1
    print_audit(stored)
    return 0


if __name__ == "__main__":
    sys.exit(main())
