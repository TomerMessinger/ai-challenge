"""Tests for the read-only catalog loader against the real services.db."""

import hashlib
import json
import sqlite3
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path

import pytest

from pipeline.catalog import DB_PATH, Service, load_catalog

SEED_PATH = DB_PATH.with_name("services.json")


def test_catalog_has_every_service_keyed_by_id_in_id_order():
    catalog = load_catalog()
    assert list(catalog) == list(range(1, 37))
    assert all(isinstance(service, Service) and service.id == service_id for service_id, service in catalog.items())


def test_columns_are_decoded_into_typed_fields():
    catalog = load_catalog()
    kart, kayak, escape_room = catalog[1], catalog[19], catalog[11]
    assert kart.price_per_person == Decimal("55") and isinstance(kart.price_per_person, Decimal)
    assert (kart.min_capacity, kart.max_capacity) == (8, 20)
    assert "go_kart" in kart.tags and isinstance(kart.tags, tuple)
    assert kart.available_months is None and kart.wheelchair_accessible is None
    assert kayak.available_months == (4, 5, 6, 7, 8, 9, 10)
    assert escape_room.wheelchair_accessible is False
    assert catalog[2].dietary_options == ("vegan", "vegetarian")


def test_every_row_matches_the_seed():
    seed = json.loads(SEED_PATH.read_text(encoding="utf-8"))
    catalog = load_catalog()
    for row in seed:
        loaded = asdict(catalog[row["id"]])
        expected = {
            **row,
            "price_per_person": Decimal(str(row["price_per_person"])),
            "tags": tuple(row["tags"]),
            "dietary_options": tuple(row["dietary_options"]),
            "available_months": None if row["available_months"] is None else tuple(row["available_months"]),
        }
        assert loaded == expected


def test_loading_leaves_the_database_untouched():
    before = hashlib.sha256(DB_PATH.read_bytes()).hexdigest()
    load_catalog()
    assert hashlib.sha256(DB_PATH.read_bytes()).hexdigest() == before


def test_missing_database_raises_and_is_not_created():
    path = Path(__file__).with_name("no-such-services.db")
    with pytest.raises(sqlite3.OperationalError):
        load_catalog(path)
    assert not path.exists()
