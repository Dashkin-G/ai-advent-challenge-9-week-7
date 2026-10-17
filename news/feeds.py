"""Сборщик: каждые 15 минут обходит RSS-ленты изданий и складывает новые новости в базу. Модель здесь не
нужна — это просто чтение лент; она включается, когда собирается сводка."""
import asyncio
import html
import logging
import re
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from xml.etree import ElementTree as ET

import httpx2

from news import config, db

logger = logging.getLogger(__name__)

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; Svodka/1.0; +news digest)"}
LEAD = 400  # знаков лида хватает и для отбора, и для саммари в одну-две фразы
TAGS = re.compile(r"<[^>]+>")
SPACES = re.compile(r"\s+")
META = re.compile(r"<meta\s[^>]*>", re.I)
ATTR = re.compile(r"""([\w:-]+)\s*=\s*(?:"([^"]*)"|'([^']*)')""")

last = {}  # итог последнего обхода — для страницы: когда, сколько новых, какие ленты не ответили


def _client(timeout: int):
    # напрямую: прокси с зарубежным выходом часть русских сайтов не пускает (Известия отвечали 403)
    return httpx2.AsyncClient(timeout=timeout, follow_redirects=True, headers=HEADERS, trust_env=False)


def clean(text: str) -> str:
    return SPACES.sub(" ", html.unescape(TAGS.sub(" ", text or ""))).strip()


def _cut(text: str) -> str:
    return text if len(text) <= LEAD else text[:LEAD].rsplit(" ", 1)[0] + "…"


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _date(text: str):
    text = text.strip()
    try:
        moment = parsedate_to_datetime(text)
    except (TypeError, ValueError):
        try:
            moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
    # без пояса русские ленты пишут московское время
    return (moment if moment.tzinfo else moment.replace(tzinfo=config.MSK)).astimezone(timezone.utc)


def _url(link: str):
    """Ссылка без меток utm_*, иначе одна и та же новость задваивается."""
    parts = urlsplit(link.strip())
    if parts.scheme not in ("http", "https") or not parts.netloc:
        return None
    query = urlencode([(k, v) for k, v in parse_qsl(parts.query) if not k.startswith("utm_")])
    return urlunsplit((parts.scheme, parts.netloc, parts.path, query, ""))


def parse(source: str, xml: bytes, now: datetime) -> list[tuple]:
    """Новости ленты RSS или Atom — строками для таблицы articles."""
    rows = []
    for item in ET.fromstring(xml).iter():
        if _local(item.tag) not in ("item", "entry"):
            continue
        fields = {}
        for child in item:
            name = _local(child.tag)
            fields.setdefault(name, child.get("href", "") if name == "link" and not child.text else child.text or "")
        url, title = _url(fields.get("link", "")), clean(fields.get("title", ""))
        published = _date(fields.get("pubDate") or fields.get("published") or fields.get("updated")
                          or fields.get("date") or "")
        if not (url and title and published):
            continue
        lead = clean(fields.get("description") or fields.get("summary") or fields.get("full-text")
                     or fields.get("encoded") or "")
        # у МК время на час впереди: новость «из будущего» считаем вышедшей сейчас
        rows.append((url, source, title, _cut(lead), db.iso(min(published, now)), db.iso(now)))
    return rows


def _reason(error: Exception) -> str:
    if isinstance(error, httpx2.HTTPStatusError):
        return f"HTTP {error.response.status_code}"
    return f"{type(error).__name__} {error}".strip()[:120]


async def _fetch(client, source: str, url: str, now: datetime) -> list[tuple]:
    response = await client.get(url)
    response.raise_for_status()
    return parse(source, response.content, now)


async def collect() -> dict:
    """Один обход всех лент: новые новости — в базу, старше KEEP_DAYS — из базы."""
    now = datetime.now(timezone.utc)
    async with _client(20) as client:
        results = await asyncio.gather(*(_fetch(client, source, url, now) for source, url in config.FEEDS.items()),
                                       return_exceptions=True)
    rows, errors = [], {}
    for source, result in zip(config.FEEDS, results):
        if isinstance(result, Exception):
            errors[source] = _reason(result)
        else:
            rows += result
    with db.connect() as conn:
        before = conn.total_changes
        conn.executemany("INSERT OR IGNORE INTO articles (url, source, title, lead, published, added) "
                         "VALUES (?, ?, ?, ?, ?, ?)", rows)
        added = conn.total_changes - before
        conn.execute("DELETE FROM articles WHERE published < ?", (db.iso(now - timedelta(days=config.KEEP_DAYS)),))
    last.update(at=db.iso(now), feeds=len(config.FEEDS), ok=len(config.FEEDS) - len(errors), added=added,
                errors=errors)
    return last


async def run_forever():
    while True:
        try:
            result = await collect()
            logger.info("Обход лент: +%s новостей, ответили %s из %s", result["added"], result["ok"], result["feeds"])
            if result["errors"]:
                logger.warning("Не ответили: %s", result["errors"])
        except Exception:
            logger.exception("Обход лент не удался")
        await asyncio.sleep(config.COLLECT_EVERY)


def _description(page: str) -> str:
    found = {}
    for tag in META.findall(page):
        attrs = {m[1].lower(): m[2] or m[3] for m in ATTR.finditer(tag)}
        key = (attrs.get("property") or attrs.get("name") or "").lower()
        if key in ("og:description", "description") and attrs.get("content"):
            found.setdefault(key, attrs["content"])
    return clean(found.get("og:description") or found.get("description") or "")


async def fill_leads(articles: list[dict]) -> int:
    """У РИА, Ленты и Ведомостей в ленте только заголовок — лид берём из описания на странице новости
    (og:description). Только у кандидатов в сводку: обходить так все новости сайты не любят."""
    missing = [a for a in articles if len(a["lead"]) < 40]
    limit = asyncio.Semaphore(6)

    async def one(client, article):
        async with limit:
            try:
                async with client.stream("GET", article["url"]) as response:
                    if response.status_code != 200:
                        return
                    page = b""
                    async for chunk in response.aiter_bytes():
                        page += chunk
                        if len(page) > 200_000:  # описание — в <head>, дальше читать незачем
                            break
                    encoding = response.encoding or "utf-8"
            except httpx2.HTTPError:
                return
        try:
            lead = _description(page.decode(encoding, errors="replace"))
        except LookupError:
            return
        if len(lead) >= 40 and lead != article["title"]:
            article["lead"] = _cut(lead)

    if missing:
        async with _client(10) as client:
            await asyncio.gather(*(one(client, a) for a in missing))
    filled = [a for a in missing if len(a["lead"]) >= 40]
    with db.connect() as conn:
        conn.executemany("UPDATE articles SET lead = ? WHERE id = ?", [(a["lead"], a["id"]) for a in filled])
    return len(filled)
