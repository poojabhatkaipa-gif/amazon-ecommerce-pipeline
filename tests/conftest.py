"""Shared pytest helpers: load the digit-prefixed scripts as modules."""

import importlib.util
import pickle
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"


def load_module(filename: str, name: str):
    """Import src/<filename> (module names starting with digits can't be imported normally)."""
    spec = importlib.util.spec_from_file_location(name, SRC / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def clean_mod():
    return load_module("02_clean.py", "clean_mod")


@pytest.fixture(scope="session")
def transform_mod():
    return load_module("03_transform.py", "transform_mod")


@pytest.fixture
def pickle_io(monkeypatch, clean_mod, transform_mod):
    """Swap Parquet I/O for pickle so end-to-end tests run without pyarrow.

    Real Parquet round-trips are covered by tests that call
    ``pytest.importorskip("pyarrow")``.
    """

    def _write(df, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(df, fh)

    def _read(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)

    monkeypatch.setattr(clean_mod, "write_parquet", _write)
    monkeypatch.setattr(transform_mod, "write_parquet", _write)
    monkeypatch.setattr(transform_mod, "read_parquet", _read)
    return _read
