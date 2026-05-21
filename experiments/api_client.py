from __future__ import annotations

from pathlib import Path

import requests


class SpeakKapClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def close(self) -> None:
        self.session.close()

    def clear(self) -> None:
        response = self.session.post(f"{self.base_url}/clear", timeout=60)
        response.raise_for_status()

    def register(self, username: str, password: str, files: list[Path]) -> None:
        handles = []
        try:
            multipart = []
            for path in files:
                handle = path.open("rb")
                handles.append(handle)
                multipart.append(("files", (path.name, handle, "audio/wav")))

            response = self.session.post(
                f"{self.base_url}/register",
                data={"username": username, "password": password},
                files=multipart,
                timeout=300,
            )
            response.raise_for_status()
        finally:
            for handle in handles:
                handle.close()

    def authenticate(self, login: str, password: str, file_path: Path) -> dict:
        with file_path.open("rb") as audio:
            response = self.session.post(
                f"{self.base_url}/authenticate",
                data={"login": login, "password": password},
                files={"file": (file_path.name, audio, "audio/wav")},
                timeout=120,
            )

        payload = response.json() if response.content else {}
        return {
            "accepted": response.status_code == 200,
            "status_code": response.status_code,
            "distance": payload.get("distance"),
            "threshold": payload.get("threshold"),
            "payload": payload,
        }
