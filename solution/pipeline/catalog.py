"""Read-only runtime access to the supplier catalog in data/services.db.

The database is the source of truth. It is opened in SQLite's read-only mode,
so loading never creates, rebuilds or mutates it (create_db.py owns that).
Prices become Decimal so budget comparisons are exact; any SQLite error
propagates to the caller.
"""

import json
import sqlite3
from dataclasses import dataclass, fields
from decimal import Decimal
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "services.db"


@dataclass(frozen=True)
class Service:
    """One row of the services table, with JSON list columns decoded."""

    id: int
    supplier_name: str
    city: str
    service_name: str
    service_type: str
    description: str
    price_per_person: Decimal  # EUR, VAT included
    min_capacity: int
    max_capacity: int
    duration_minutes: int | None
    tags: tuple[str, ...]
    dietary_options: tuple[str, ...]
    wheelchair_accessible: bool | None
    available_months: tuple[int, ...] | None  # None: no known seasonal restriction
    notes: str | None


_COLUMNS = tuple(field.name for field in fields(Service))


def load_catalog(path: Path = DB_PATH) -> dict[int, Service]:
    """Every service keyed by ID, in ascending ID order."""
    conn = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
    try:
        rows = conn.execute(f"SELECT {', '.join(_COLUMNS)} FROM services ORDER BY id").fetchall()
    finally:
        conn.close()
    services = [_to_service(dict(zip(_COLUMNS, row))) for row in rows]
    return {service.id: service for service in services}


def _to_service(row: dict) -> Service:
    wheelchair, months = row["wheelchair_accessible"], row["available_months"]
    return Service(
        **{
            **row,
            "price_per_person": Decimal(str(row["price_per_person"])),
            "tags": tuple(json.loads(row["tags"])),
            "dietary_options": tuple(json.loads(row["dietary_options"])),
            "wheelchair_accessible": None if wheelchair is None else bool(wheelchair),
            "available_months": None if months is None else tuple(json.loads(months)),
        }
    )
