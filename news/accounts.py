"""Учётные записи: пароль — scrypt с солью, вход — случайный токен в cookie, в базе — только хеш токена."""
import hashlib
import hmac
import re
import secrets
import sqlite3
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone

from news import db

SESSION_DAYS = 30
EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class AuthError(Exception):
    """Почему не вышло войти или зарегистрироваться — текст для пользователя."""


class Limiter:
    """Не больше count попыток за period секунд с одного адреса — от подбора паролей и потока регистраций."""

    def __init__(self, count: int, period: int):
        self.count, self.period = count, period
        self.attempts = defaultdict(deque)

    def allow(self, key: str) -> bool:
        now, attempts = time.monotonic(), self.attempts[key]
        while attempts and attempts[0] < now - self.period:
            attempts.popleft()
        if len(attempts) >= self.count:
            return False
        attempts.append(now)
        return True


def _scrypt(password: str, salt: bytes) -> str:
    return hashlib.scrypt(password.encode(), salt=salt, n=2 ** 14, r=8, p=1, dklen=32).hex()


def _key(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def register(email: str, password: str) -> int:
    email = email.strip().lower()
    if len(email) > 254 or not EMAIL.match(email):
        raise AuthError("Проверьте адрес почты")
    if not 8 <= len(password) <= 200:
        raise AuthError("Пароль — от 8 знаков")
    salt = secrets.token_bytes(16)
    try:
        with db.connect() as conn:
            return conn.execute("INSERT INTO users (email, password, created) VALUES (?, ?, ?)",
                                (email, f"scrypt${salt.hex()}${_scrypt(password, salt)}", db.now())).lastrowid
    except sqlite3.IntegrityError:
        raise AuthError("Эта почта уже зарегистрирована — войдите")


def check(email: str, password: str) -> int:
    with db.connect() as conn:
        row = conn.execute("SELECT id, password FROM users WHERE email = ?", (email.strip().lower(),)).fetchone()
    if row:
        _, salt, digest = row["password"].split("$")
        if hmac.compare_digest(_scrypt(password, bytes.fromhex(salt)), digest):
            return row["id"]
    raise AuthError("Неверная почта или пароль")


def open_session(user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    with db.connect() as conn:
        conn.execute("DELETE FROM sessions WHERE expires < ?", (db.now(),))
        conn.execute("INSERT INTO sessions (token, user_id, expires) VALUES (?, ?, ?)",
                     (_key(token), user_id, db.iso(datetime.now(timezone.utc) + timedelta(days=SESSION_DAYS))))
    return token


def user_by_session(token: str | None):
    if not token:
        return None
    with db.connect() as conn:
        return conn.execute("SELECT users.id, users.email FROM sessions JOIN users ON users.id = sessions.user_id "
                            "WHERE sessions.token = ? AND sessions.expires > ?", (_key(token), db.now())).fetchone()


def close_session(token: str | None):
    if token:
        with db.connect() as conn:
            conn.execute("DELETE FROM sessions WHERE token = ?", (_key(token),))
