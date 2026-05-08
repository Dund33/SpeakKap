from __future__ import annotations

import pytest
import requests


BASE_URL = "http://127.0.0.1:5000"


@pytest.fixture()
def client():
    session = requests.Session()

    yield session

    session.close()