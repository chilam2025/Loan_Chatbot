"""Test-wide configuration: mock model and a small reference-table fixture.

Set before any application module is imported, because ``abc_credit.api.main``
builds a default app at import time. Tests that need the real model artifact
skip themselves when it is absent.
"""

import os
from pathlib import Path

FIXTURE_TABLES = Path(__file__).parent / "fixtures" / "reference_tables.json"

os.environ.setdefault("ABC_MODEL_PROVIDER", "mock")
os.environ.setdefault("ABC_REFERENCE_TABLES_PATH", str(FIXTURE_TABLES))
os.environ.setdefault("ABC_LOG_LEVEL", "WARNING")
