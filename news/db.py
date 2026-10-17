"""SQLite: пользователи и сессии, темы, новости из лент, готовые сводки. Время — строкой ISO в UTC,
поэтому сравнивается как текст."""
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

from news import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    email TEXT NOT NULL UNIQUE,
    password TEXT NOT NULL,          -- scrypt$соль$хеш
    created TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    token TEXT PRIMARY KEY,          -- sha256 токена из cookie: по копии базы не войти
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    expires TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS topics (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    query TEXT NOT NULL,             -- как тему написал пользователь
    brief TEXT NOT NULL,             -- как её поняла модель: JSON {title, description, keywords}
    created TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS articles (
    id INTEGER PRIMARY KEY,
    url TEXT NOT NULL UNIQUE,
    source TEXT NOT NULL,
    title TEXT NOT NULL,
    lead TEXT NOT NULL DEFAULT '',
    published TEXT NOT NULL,
    added TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS articles_published ON articles(published);
CREATE TABLE IF NOT EXISTS digests (
    id INTEGER PRIMARY KEY,
    topic_id INTEGER NOT NULL REFERENCES topics(id) ON DELETE CASCADE,
    created TEXT NOT NULL,
    auto INTEGER NOT NULL,           -- 1 — по расписанию, 0 — по кнопке
    content TEXT NOT NULL            -- JSON: сводка, сюжеты, отброшенное, цифры шагов
);
CREATE INDEX IF NOT EXISTS digests_topic ON digests(topic_id, created);
"""


def iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat(timespec="seconds")


def now() -> str:
    return iso(datetime.now(timezone.utc))


@contextmanager
def connect():
    conn = sqlite3.connect(config.DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def init():
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    with connect() as conn:
        conn.execute("PRAGMA journal_mode = WAL")  # сборщик пишет, пока страница читает
        conn.executescript(SCHEMA)
