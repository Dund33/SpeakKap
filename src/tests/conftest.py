from __future__ import annotations

import pytest

from src.api import app


@pytest.fixture()
def client():
    app.config["TESTING"] = True

    with app.test_client() as client:
        yield client