from __future__ import annotations

from http.client import responses
from pathlib import Path

SAMPLES_DIR = Path("data")


def test_register(client, data_partitions):

    joor_files = data_partitions['joor_register']

    files = [
        ("files", (filepath.name, open(filepath, "rb"), "audio/wav")) for filepath in joor_files
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


def test_identify(client, data_partitions):

    joor_files = data_partitions['joor_login']

    responses = []

    for joor_file in joor_files:
        with open(joor_file, "rb") as audio:
            response = client.post(
                "http://127.0.0.1:5000/identify",
                files={
                    "file": (joor_file.name, audio, "audio/wav"),
                },
        )
        responses.append(response)

    assert all(response.status_code in [200, 404] for response in responses)

    assert all(map(lambda r: r.json() is not None, responses))

    for response in responses:
        json_data = response.json()
        if response.status_code == 200:
            assert "login" in json_data
            assert "xor_hash" in json_data


def test_authenticate_success(client, data_partitions):

    joor_files = data_partitions['joor_login']

    responses = []

    for joor_file in joor_files:
        with open(joor_file, "rb") as audio:
            response = client.post(
                "http://127.0.0.1:5000/authenticate",
                data={
                    "login": "pytest_user",
                    "password": "secret123",
                    "threshold": 0.5,
                },
                files={
                    "file": (joor_file.name, audio, "audio/wav"),
                },
            )
        responses.append(response)

    #All responses should be 200 OK
    assert all(response.status_code == 200 for response in responses)

    #All responses should contain valid JSON data
    assert all(map(lambda r: r.json() is not None, responses))


def test_authenticate_invalid_password(client, data_partitions):

    joor_files = data_partitions['joor_login']

    responses = []

    for joor_file in joor_files:
        with open(joor_file, "rb") as audio:
            response = client.post(
                "http://127.0.0.1:5000/authenticate",
                data={
                    "login": "pytest_user",
                    "password": "wrong_password",
                    "threshold": 0.5,
                },
                files={
                    "file": (joor_file.name, audio, "audio/wav"),
                },
            )
        responses.append(response)

    #All responses should be either 401 Unauthorized or 404 Not Found, since the password is wrong
    assert all(response.status_code in [401, 404] for response in responses)

    #All responses should contain valid JSON data, even if the authentication fails
    assert all(map(lambda r: r.json() is not None, responses))