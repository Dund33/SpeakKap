from __future__ import annotations

import hashlib
import os
import uuid

from flask import Flask, jsonify, request
from werkzeug.security import check_password_hash
from werkzeug.utils import secure_filename

from utils.config import Config

from src.services.RedisStoreService import RedisStoreService
from src.services.WeSpeakerService import WeSpeakerService
from src.utils.MathUtils import MathUtils

app = Flask(__name__)

UPLOAD_FOLDER = Config.UPLOAD_FOLDER

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

redis_store_service = RedisStoreService(
    redis_url=Config.REDIS_URL,
    embedding_dim=Config.EMBEDDING_DIM
)

redis_store_service.ensure_index()

wespeaker_service = WeSpeakerService(
    device=Config.WESPEAKER_DEVICE
)


def xor_hash(
    login: str,
    password_hash: str
) -> str:

    login_hash = hashlib.sha256(
        login.encode()
    ).digest()

    password_hash_digest = hashlib.sha256(
        password_hash.encode()
    ).digest()

    xor_bytes = bytes(
        a ^ b
        for a, b in zip(
            login_hash,
            password_hash_digest
        )
    )

    return xor_bytes.hex()


@app.post("/register")
def register():

    login = request.form.get(
        "username"
    )

    password = request.form.get(
        "password"
    )

    files = request.files.getlist(
        "files"
    )

    if not login or not password:
        return jsonify({
            "error":
            "username and password required"
        }), 400

    if redis_store_service.get_profile(
        login
    ):
        return jsonify({
            "error":
            "user already exists"
        }), 409

    if not files:
        return jsonify({
            "error":
            "at least one file required"
        }), 400

    embeddings = []

    for file in files:

        if not file or not file.filename:
            continue

        filename = secure_filename(
            file.filename
        )

        filepath = os.path.join(
            UPLOAD_FOLDER,
            f"{uuid.uuid4()}_{filename}"
        )

        file.save(filepath)

        embedding = (
            wespeaker_service.get_embedding(
                filepath
            )
        )

        embeddings.append(
            embedding
        )

    medoid = MathUtils.get_medoid(
        embeddings
    )

    redis_store_service.create_profile(
        login=login,
        password=password,
        embedding=medoid
    )

    return jsonify({
        "message":
        "registered",
        "login":
        login
    }), 201


@app.post("/identify")
def identify():

    file = request.files.get(
        "file"
    )

    if not file or not file.filename:
        return jsonify({
            "error":
            "audio file required"
        }), 400

    filename = secure_filename(
        file.filename
    )

    filepath = os.path.join(
        UPLOAD_FOLDER,
        f"{uuid.uuid4()}_{filename}"
    )

    file.save(filepath)

    embedding = (
        wespeaker_service.get_embedding(
            filepath
        )
    )

    result = (
        redis_store_service
        .find_similar_speakers(
            embedding,
            top_k=1
        )
    )

    if len(result.docs) == 0:
        return jsonify({
            "error":
            "no matching user"
        }), 404

    doc = result.docs[0]

    doc_id = (
        doc.id.decode()
        if isinstance(
            doc.id,
            (bytes, bytearray)
        )
        else str(doc.id)
    )

    login = doc_id.replace(
        "speaker:",
        ""
    )

    profile = (
        redis_store_service
        .get_profile(login)
    )

    return jsonify({
        "login":
        login,

        "xor_hash":
        xor_hash(
            login,
            profile.password_hash
        )
    })


@app.post("/authenticate")
def authenticate():

    login = request.form.get(
        "login"
    )

    password = request.form.get(
        "password"
    )

    file = request.files.get(
        "file"
    )

    threshold = float(
        request.form.get(
            "threshold",
            0.25
        )
    )

    if not login or not password:
        return jsonify({
            "error":
            "login and password required"
        }), 400

    if not file or not file.filename:
        return jsonify({
            "error":
            "audio file required"
        }), 400

    profile = (
        redis_store_service
        .get_profile(login)
    )

    if profile is None:
        return jsonify({
            "error":
            "user not found"
        }), 404

    if not check_password_hash(
        profile.password_hash,
        password
    ):
        return jsonify({
            "error":
            "invalid credentials"
        }), 401

    filename = secure_filename(
        file.filename
    )

    filepath = os.path.join(
        UPLOAD_FOLDER,
        f"{uuid.uuid4()}_{filename}"
    )

    file.save(filepath)

    current_embedding = (
        wespeaker_service.get_embedding(
            filepath
        )
    )

    distance = (
        MathUtils.cosine_distance(
            current_embedding,
            profile.embedding
        )
    )

    if distance > threshold:
        return jsonify({
            "error":
            "authentication failed",

            "distance":
            float(distance),

            "threshold":
            threshold
        }), 401

    return jsonify({
        "message":
        "authentication successful",

        "login":
        login,

        "distance":
        float(distance),

        "threshold":
        threshold
    })


if __name__ == "__main__":
    app.run(debug=True)