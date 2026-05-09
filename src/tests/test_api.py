from __future__ import annotations

from pathlib import Path

SAMPLES_DIR = Path("data")

MAX_FRR = 0.15


def register_user(client, username: str, password: str, files: list[Path]) -> None:
    client.post("http://127.0.0.1:5000/clear")

    file_tuples = [
        ("files", (filepath.name, open(filepath, "rb"), "audio/wav"))
        for filepath in files
    ]

    data = {
        "username": username,
        "password": password,
    }

    client.post(
        "http://127.0.0.1:5000/register",
        data=data,
        files=file_tuples,
    )


def test_register(client, data_partitions):
    clear_response = client.post("http://127.0.0.1:5000/clear")
    assert clear_response.status_code == 200

    clear_json = clear_response.json()
    assert clear_json is not None
    assert clear_json["message"] == "database cleared"

    joor_files = data_partitions["joor_register"]

    files = [
        ("files", (filepath.name, open(filepath, "rb"), "audio/wav"))
        for filepath in joor_files
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

    assert response.status_code == 201

    json_data = response.json()
    assert json_data is not None
    assert json_data["message"] == "registered"
    assert json_data["login"] == "pytest_user"


def test_identify(client, data_partitions):

    joor_files = data_partitions["joor_login"]
    joor_register_files = data_partitions["joor_register"]

    register_user(client, "pytest_user", "secret123", joor_register_files)

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

    # Responses should be 200 OK
    auth_statuses = (response.status_code == 200 for response in responses)
    successes = sum(auth_statuses)
    assert successes / len(responses) >= (1 - MAX_FRR)

    # All responses should contain valid JSON data
    assert all(map(lambda r: r.json() is not None, responses))

    for response in responses:
        json_data = response.json()
        if response.status_code == 200:
            assert "combined_hash" in json_data


def test_authenticate_success(client, data_partitions):

    joor_files = data_partitions["joor_login"]

    responses = []

    for joor_file in joor_files:
        with open(joor_file, "rb") as audio:
            response = client.post(
                "http://127.0.0.1:5000/authenticate",
                data={
                    "login": "pytest_user",
                    "password": "secret123",
                },
                files={
                    "file": (joor_file.name, audio, "audio/wav"),
                },
            )
        responses.append(response)

    # All responses should be 200 OK
    auth_statuses = (response.status_code == 200 for response in responses)
    successes = sum(auth_statuses)
    assert successes / len(responses) >= (1 - MAX_FRR)

    # All responses should contain valid JSON data
    assert all(map(lambda r: r.json() is not None, responses))


def test_authenticate_invalid_password(client, data_partitions):

    joor_files = data_partitions["joor_login"]

    responses = []

    for joor_file in joor_files:
        with open(joor_file, "rb") as audio:
            response = client.post(
                "http://127.0.0.1:5000/authenticate",
                data={
                    "login": "pytest_user",
                    "password": "wrong_password",
                },
                files={
                    "file": (joor_file.name, audio, "audio/wav"),
                },
            )
        responses.append(response)

    # All responses should be either 401 Unauthorized or 404 Not Found, since the password is wrong
    assert all(response.status_code in [401, 404] for response in responses)

    # All responses should contain valid JSON data, even if the authentication fails
    assert all(map(lambda r: r.json() is not None, responses))


MAX_FAR = 0.05


def test_authenticate_wrong_speaker(client, data_partitions):
    knur_files = data_partitions["knur"]
    joor_register_files = data_partitions["joor_register"]

    clear_response = client.post("http://127.0.0.1:5000/clear")
    assert clear_response.status_code == 200

    register_user(client, "pytest_user", "secret123", joor_register_files)

    responses = []

    for knur_file in knur_files:
        with open(knur_file, "rb") as audio:
            response = client.post(
                "http://127.0.0.1:5000/authenticate",
                data={
                    "login": "pytest_user",
                    "password": "secret123",
                },
                files={
                    "file": (knur_file.name, audio, "audio/wav"),
                },
            )
        responses.append(response)

    accepted = sum(response.status_code == 200 for response in responses)
    far = accepted / len(responses)

    assert far <= MAX_FAR

    assert all(response.json() is not None for response in responses)
