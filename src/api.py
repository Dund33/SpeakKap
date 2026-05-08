from flask import Flask, request, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import jwt
import datetime
from functools import wraps
import uuid
import os

app = Flask(__name__)

# =========================================
# KONFIGURACJA
# =========================================

app.config['SECRET_KEY'] = 'SUPER_SECRET_KEY'

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Prosta baza danych w pamięci
users = []


# =========================================
# MIDDLEWARE AUTORYZACJI
# =========================================

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):

        token = None

        auth_header = request.headers.get('Authorization')

        if auth_header and auth_header.startswith('Bearer '):
            token = auth_header.split(" ")[1]

        if not token:
            return jsonify({
                'error': 'Brak tokenu'
            }), 401

        try:
            data = jwt.decode(
                token,
                app.config['SECRET_KEY'],
                algorithms=["HS256"]
            )

            current_user = next(
                (u for u in users if u['id'] == data['user_id']),
                None
            )

            if current_user is None:
                return jsonify({
                    'error': 'Użytkownik nie istnieje'
                }), 401

        except jwt.ExpiredSignatureError:
            return jsonify({
                'error': 'Token wygasł'
            }), 401

        except jwt.InvalidTokenError:
            return jsonify({
                'error': 'Nieprawidłowy token'
            }), 401

        return f(current_user, *args, **kwargs)

    return decorated


# =========================================
# REGISTER
# Przyjmuje:
# - username
# - password
# - lista plików
# =========================================

@app.route('/register', methods=['POST'])
def register():

    username = request.form.get('username')
    password = request.form.get('password')

    if not username or not password:
        return jsonify({
            'error': 'Username i password są wymagane'
        }), 400

    existing_user = next(
        (u for u in users if u['username'] == username),
        None
    )

    if existing_user:
        return jsonify({
            'error': 'Użytkownik już istnieje'
        }), 409

    # Lista plików
    uploaded_files = request.files.getlist('files')

    saved_files = []

    for file in uploaded_files:

        if file.filename == '':
            continue

        filename = secure_filename(file.filename)

        unique_filename = f"{uuid.uuid4()}_{filename}"

        filepath = os.path.join(
            UPLOAD_FOLDER,
            unique_filename
        )

        file.save(filepath)

        saved_files.append(filepath)

    hashed_password = generate_password_hash(password)

    user = {
        'id': str(uuid.uuid4()),
        'username': username,
        'password': hashed_password,
        'files': saved_files
    }

    users.append(user)

    return jsonify({
        'message': 'Użytkownik zarejestrowany',
        'user_id': user['id'],
        'uploaded_files': saved_files
    }), 201


# =========================================
# IDENTIFY
# Przyjmuje:
# - username
# - password
# - jeden plik
# =========================================

@app.route('/identify', methods=['POST'])
def identify():

    username = request.form.get('username')
    password = request.form.get('password')

    uploaded_file = request.files.get('file')

    if uploaded_file is None:
        return jsonify({
            'error': 'Plik jest wymagany'
        }), 400

    user = next(
        (u for u in users if u['username'] == username),
        None
    )

    if not user:
        return jsonify({
            'error': 'Nieprawidłowy login lub hasło'
        }), 401

    if not check_password_hash(user['password'], password):
        return jsonify({
            'error': 'Nieprawidłowy login lub hasło'
        }), 401

    # Zapis pliku
    filename = secure_filename(uploaded_file.filename)

    unique_filename = f"{uuid.uuid4()}_{filename}"

    filepath = os.path.join(
        UPLOAD_FOLDER,
        unique_filename
    )

    uploaded_file.save(filepath)

    # JWT
    token = jwt.encode({
        'user_id': user['id'],
        'username': user['username'],
        'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=1)
    },
        app.config['SECRET_KEY'],
        algorithm="HS256"
    )

    return jsonify({
        'message': 'Logowanie poprawne',
        'token': token,
        'uploaded_file': filepath
    })


# =========================================
# AUTHENTICATE / AUTHORIZE
# Przyjmuje:
# - JWT
# - jeden plik
# =========================================

@app.route('/authenticate', methods=['POST'])
@token_required
def authenticate(current_user):

    uploaded_file = request.files.get('file')

    if uploaded_file is None:
        return jsonify({
            'error': 'Plik jest wymagany'
        }), 400

    filename = secure_filename(uploaded_file.filename)

    unique_filename = f"{uuid.uuid4()}_{filename}"

    filepath = os.path.join(
        UPLOAD_FOLDER,
        unique_filename
    )

    uploaded_file.save(filepath)

    return jsonify({
        'message': 'Autoryzacja poprawna',
        'user': {
            'id': current_user['id'],
            'username': current_user['username']
        },
        'uploaded_file': filepath
    })


# =========================================
# START
# =========================================

if __name__ == '__main__':
    app.run(debug=True)