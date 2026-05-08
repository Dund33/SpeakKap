from __future__ import annotations

from pathlib import Path

SAMPLES_DIR = Path("samples")


def test_register(client):

    files = [
        (
            open(
                SAMPLES_DIR / "voice1.wav",
                "rb"
            ),
            "voice1.wav"
        ),

        (
            open(
                SAMPLES_DIR / "voice2.wav",
                "rb"
            ),
            "voice2.wav"
        ),
    ]

    data = {
        "username": "pytest_user",
        "password": "secret123",
        "files": files,
    }

    response = client.post(
        "/register",
        data=data,
        content_type="multipart/form-data",
    )

    assert response.status_code in [201, 409]

    json_data = response.get_json()

    assert json_data is not None


def test_identify(client):

    with open(
        SAMPLES_DIR / "test.wav",
        "rb"
    ) as audio:

        response = client.post(
            "/identify",
            data={
                "file": (
                    audio,
                    "test.wav"
                )
            },
            content_type="multipart/form-data",
        )

    assert response.status_code in [200, 404]

    json_data = response.get_json()

    assert json_data is not None

    if response.status_code == 200:
        assert "login" in json_data
        assert "xor_hash" in json_data


def test_authenticate_success(client):

    with open(
        SAMPLES_DIR / "test.wav",
        "rb"
    ) as audio:

        response = client.post(
            "/authenticate",
            data={
                "login": "pytest_user",
                "password": "secret123",
                "threshold": 0.5,
                "file": (
                    audio,
                    "test.wav"
                ),
            },
            content_type="multipart/form-data",
        )

    assert response.status_code in [200, 401, 404]

    json_data = response.get_json()

    assert json_data is not None


def test_authenticate_invalid_password(client):

    with open(
        SAMPLES_DIR / "test.wav",
        "rb"
    ) as audio:

        response = client.post(
            "/authenticate",
            data={
                "login": "pytest_user",
                "password": "wrong_password",
                "threshold": 0.5,
                "file": (
                    audio,
                    "test.wav"
                ),
            },
            content_type="multipart/form-data",
        )

    assert response.status_code in [401, 404]

    json_data = response.get_json()

    assert json_data is not None