"""Сводка новостей по теме: регистрация, темы, ежедневные сводки. Сборщик лент и сводки по расписанию
работают фоном в этом же процессе — поэтому uvicorn с одним воркером."""
import asyncio
import json
import logging
import math
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from news import accounts, config, db, digest, feeds, llm

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx2").setLevel(logging.WARNING)  # иначе в журнале каждый запрос к лентам: 25 строк в 15 минут

PAGE = config.ROOT / "static" / "index.html"
COOKIE = "session"
logins = accounts.Limiter(10, 300)
signups = accounts.Limiter(5, 3600)


@asynccontextmanager
async def lifespan(app):
    db.init()
    tasks = [asyncio.create_task(feeds.run_forever()), asyncio.create_task(digest.scheduler())]
    yield
    for task in tasks:
        task.cancel()


app = FastAPI(title="Сводка", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)


class Credentials(BaseModel):
    email: str
    password: str


class NewTopic(BaseModel):
    query: str


class Brief(BaseModel):
    description: str
    keywords: list[str]


def current_user(request: Request):
    user = accounts.user_by_session(request.cookies.get(COOKIE))
    if not user:
        raise HTTPException(401, "Войдите в аккаунт")
    return user


def _own_topic(topic_id: int, user) -> dict:
    with db.connect() as conn:
        row = conn.execute("SELECT * FROM topics WHERE id = ? AND user_id = ?", (topic_id, user["id"])).fetchone()
    if not row:
        raise HTTPException(404, "Такой темы нет")
    return dict(row)


def _sign_in(response: Response, user_id: int):
    response.set_cookie(COOKIE, accounts.open_session(user_id), max_age=accounts.SESSION_DAYS * 86400,
                        httponly=True, samesite="lax", secure=config.COOKIE_SECURE)


@app.get("/", include_in_schema=False)
async def page():
    return FileResponse(PAGE)


@app.get("/api/status")
def status():
    """Сборщик и база — видно и без входа: живой ли сервис."""
    with db.connect() as conn:
        day = conn.execute("SELECT COUNT(*), COUNT(DISTINCT source) FROM articles WHERE published > ?",
                           (db.iso(datetime.now(timezone.utc) - config.WINDOW),)).fetchone()
    return {"collector": feeds.last, "articles": day[0], "sources": day[1], "feeds": list(config.FEEDS),
            "every": config.COLLECT_EVERY // 60, "digest_at": config.DIGEST_AT.strftime("%H:%M"),
            "model": config.LLM_MODEL, "topics_per_user": config.TOPICS_PER_USER}


@app.post("/api/register")
def register(body: Credentials, request: Request, response: Response):
    if not signups.allow(request.client.host):
        raise HTTPException(429, "Слишком много регистраций с этого адреса — попробуйте через час")
    try:
        user_id = accounts.register(body.email, body.password)
    except accounts.AuthError as e:
        raise HTTPException(400, str(e))
    _sign_in(response, user_id)
    return {"email": body.email.strip().lower()}


@app.post("/api/login")
def login(body: Credentials, request: Request, response: Response):
    if not logins.allow(request.client.host):
        raise HTTPException(429, "Слишком много попыток входа — подождите 5 минут")
    try:
        user_id = accounts.check(body.email, body.password)
    except accounts.AuthError as e:
        raise HTTPException(400, str(e))
    _sign_in(response, user_id)
    return {"email": body.email.strip().lower()}


@app.post("/api/logout")
def logout(request: Request, response: Response):
    accounts.close_session(request.cookies.get(COOKIE))
    response.delete_cookie(COOKIE)
    return {"ok": True}


@app.get("/api/me")
def me(user=Depends(current_user)):
    return {"email": user["email"]}


@app.get("/api/topics")
def topics(user=Depends(current_user)):
    with db.connect() as conn:
        rows = conn.execute("SELECT topics.*, MAX(digests.created) AS last FROM topics LEFT JOIN digests "
                            "ON digests.topic_id = topics.id WHERE user_id = ? GROUP BY topics.id ORDER BY topics.id",
                            (user["id"],)).fetchall()
    return [{"id": r["id"], "query": r["query"], "brief": json.loads(r["brief"]), "last": r["last"]} for r in rows]


@app.post("/api/topics")
async def add_topic(body: NewTopic, user=Depends(current_user)):
    """Новая тема: модель разбирает её — название, что относится к теме, ключи поиска."""
    query = " ".join(body.query.split())
    if not 2 <= len(query) <= 200:
        raise HTTPException(400, "Тема — от 2 до 200 знаков")
    with db.connect() as conn:
        mine = conn.execute("SELECT COUNT(*) FROM topics WHERE user_id = ?", (user["id"],)).fetchone()[0]
        total = conn.execute("SELECT COUNT(*) FROM topics").fetchone()[0]
    if mine >= config.TOPICS_PER_USER:
        raise HTTPException(400, f"Тем — не больше {config.TOPICS_PER_USER}: удалите ненужную")
    if total >= config.MAX_TOPICS:
        raise HTTPException(503, "Свободных мест нет: сервис работает в тестовом режиме")
    try:
        brief, usage = await digest.make_brief(query)
    except llm.LLMError as e:
        raise HTTPException(502, f"Модель не ответила: {e}")
    with db.connect() as conn:
        topic_id = conn.execute("INSERT INTO topics (user_id, query, brief, created) VALUES (?, ?, ?, ?)",
                                (user["id"], query, json.dumps(brief, ensure_ascii=False), db.now())).lastrowid
    return {"id": topic_id, "query": query, "brief": brief, "last": None, "usage": usage}


@app.put("/api/topics/{topic_id}")
def edit_topic(topic_id: int, body: Brief, user=Depends(current_user)):
    """Пользователь поправил, как модель поняла тему: описание для отбора и ключи поиска."""
    topic = _own_topic(topic_id, user)
    keywords = digest.clean_keywords(body.keywords)
    if not keywords:
        raise HTTPException(400, "Нужен хотя бы один ключ поиска")
    brief = {**json.loads(topic["brief"]), "description": body.description.strip()[:600], "keywords": keywords}
    with db.connect() as conn:
        conn.execute("UPDATE topics SET brief = ? WHERE id = ?", (json.dumps(brief, ensure_ascii=False), topic_id))
    return brief


@app.delete("/api/topics/{topic_id}")
def delete_topic(topic_id: int, user=Depends(current_user)):
    _own_topic(topic_id, user)
    with db.connect() as conn:
        conn.execute("DELETE FROM topics WHERE id = ?", (topic_id,))
    return {"ok": True}


@app.get("/api/topics/{topic_id}/digests")
def topic_digests(topic_id: int, user=Depends(current_user)):
    """Архив сводок по теме: дата и цифры шагов, без текста."""
    _own_topic(topic_id, user)
    with db.connect() as conn:
        rows = conn.execute("SELECT id, created, auto, json_extract(content, '$.stats') AS stats FROM digests "
                            "WHERE topic_id = ? ORDER BY created DESC", (topic_id,)).fetchall()
    return [{"id": r["id"], "created": r["created"], "auto": bool(r["auto"]), "stats": json.loads(r["stats"])}
            for r in rows]


@app.get("/api/digests/{digest_id}")
def get_digest(digest_id: int, user=Depends(current_user)):
    with db.connect() as conn:
        row = conn.execute("SELECT digests.* FROM digests JOIN topics ON topics.id = digests.topic_id "
                           "WHERE digests.id = ? AND topics.user_id = ?", (digest_id, user["id"])).fetchone()
    if not row:
        raise HTTPException(404, "Такой сводки нет")
    return {"id": row["id"], "created": row["created"], "auto": bool(row["auto"]), **json.loads(row["content"])}


@app.post("/api/topics/{topic_id}/run")
async def run_digest(topic_id: int, user=Depends(current_user)):
    """«Собрать сейчас»: сводка за последние сутки, шаги — потоком NDJSON."""
    topic = _own_topic(topic_id, user)
    if topic_id in digest.running:
        raise HTTPException(409, "Сводка по этой теме уже собирается")
    with db.connect() as conn:
        last = conn.execute("SELECT MAX(created) FROM digests WHERE topic_id = ?", (topic_id,)).fetchone()[0]
    wait = (datetime.fromisoformat(last) + config.MANUAL_EVERY - datetime.now(timezone.utc)).total_seconds() if last else 0
    if wait > 0:
        raise HTTPException(429, f"Новую сводку по теме можно собрать через {math.ceil(wait / 60)} мин")

    async def stream():
        async for event in digest.build(topic, auto=False):
            yield json.dumps(event, ensure_ascii=False) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


if __name__ == "__main__":
    uvicorn.run(app, host=config.HOST, port=config.PORT)
