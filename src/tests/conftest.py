from __future__ import annotations
from pathlib import Path
import pytest
import requests
import numpy as np

BASE_URL = "http://127.0.0.1:5000"
SAMPLES_DIR = Path("data")


@pytest.fixture()
def client():
    session = requests.Session()

    yield session

    session.close()


@pytest.fixture(scope="session", autouse=True)
def data_partitions():
    joor_files = list(SAMPLES_DIR.glob("joor*.wav"))

    rng = np.random.default_rng()

    joor_register_files = list(rng.choice(joor_files, size=8, replace=False))

    joor_login_files = [f for f in joor_files if f not in joor_register_files]

    knur_files = list(SAMPLES_DIR.glob("knur*.wav"))

    return {
        "joor_register": joor_register_files,
        "joor_login": joor_login_files,
        "knur": knur_files,
    }