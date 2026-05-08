from __future__ import annotations

from pathlib import Path

SAMPLES_DIR = Path("data")


def test_register(client):

    files = [
        ("files", ("joor1.wav", open(SAMPLES_DIR / "joor1.wav", "rb"), "audio/wav")),
        ("files", ("joor2.wav", open(SAMPLES_DIR / "joor2.wav", "rb"), "audio/wav")),
    ]

    data = {
        "username": "pytest_user",
        "password": "secret123",
    }

    response = client.post(
        "http://127.0.0.1:5000/register",
        data=data,
        files=files,
    )

    assert response.status_code in [201, 409]

    json_data = response.json()

    assert json_data is not None


def test_identify(client):

    with open(SAMPLES_DIR / "joor3.wav", "rb") as audio:
        response = client.post(
            "http://127.0.0.1:5000/identify",
            files={
                "file": ("joor3.wav", audio, "audio/wav"),
            },
        )

    assert response.status_code in [200, 404]

    json_data = response.json()

    assert json_data is not None

    if response.status_code == 200:
        assert "login" in json_data
        assert "xor_hash" in json_data


def test_authenticate_success(client):

    with open(SAMPLES_DIR / "joor3.wav", "rb") as audio:
        response = client.post(
            "http://127.0.0.1:5000/authenticate",
            data={
                "login": "pytest_user",
                "password": "secret123",
                "threshold": 0.5,
            },
            files={
                "file": ("joor3.wav", audio, "audio/wav"),
            },
        )

    assert response.status_code in [200, 401, 404]

    json_data = response.json()

    assert json_data is not None


def test_authenticate_invalid_password(client):

    with open(SAMPLES_DIR / "joor3.wav", "rb") as audio:
        response = client.post(
            "http://127.0.0.1:5000/authenticate",
            data={
                "login": "pytest_user",
                "password": "wrong_password",
                "threshold": 0.5,
            },
            files={
                "file": ("joor3.wav", audio, "audio/wav"),
            },
        )

    assert response.status_code in [401, 404]

    json_data = response.json()

    assert json_data is not None