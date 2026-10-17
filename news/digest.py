"""Сводка по теме за сутки — четыре шага, их же видно на странице:
① новости за сутки — из базы, которую копит сборщик лент;
② кандидаты — новости, где в заголовке или лиде есть все слова хотя бы одного ключа темы (ключи при создании
   темы предложила модель, пользователь может их поправить);
③ модель проверяет кандидатов: о теме ли новость, и коротко пишет почему;
④ модель склеивает одну новость из разных изданий в сюжет, пишет по 1–2 фразы на сюжет и абзац о главном.
По расписанию — каждый день в 20:00 по Москве, по кнопке «Собрать сейчас» — сразу."""
import asyncio
import json
import logging
import math
import re
import time
from collections import Counter
from datetime import datetime, timedelta, timezone

from news import config, db, feeds, llm

logger = logging.getLogger(__name__)

BATCH = 20  # новостей на один вызов проверки
STORY_LIMIT = 40  # новостей в сюжеты; остальные по теме — списком ссылок
running: set[int] = set()  # темы, по которым сводка собирается прямо сейчас

BRIEF = """Ты настраиваешь ежедневную сводку новостей российских СМИ по теме, которую назвал пользователь.
Ответь JSON-объектом {"title": "...", "description": "...", "keywords": ["...", "..."]}.
title — тема в 2–6 словах, с заглавной буквы.
description — 1–2 предложения: какие новости относятся к теме. Новость, где тема упомянута вскользь, не относится.
keywords — 4–12 ключей поиска по заголовкам и лидам. Ключ находит новость, в которой есть каждое его слово,
поэтому ключ должен указывать именно на тему:
- имя и фамилия, должность, название органа, закона, компании, события — то, без чего новость не о теме;
- фамилия человека — отдельным ключом: в заголовках часто нет имени, а однофамильцев потом отсеет проверка;
- слова — основой без окончания, чтобы находились все падежи: «набиуллин», «ключев ставк»; предлогов не пиши;
- общее слово отдельным ключом не годится: «санкции», «встреча», «заявление», «Россия», «США», «закон»
  есть в сотнях новостей за день. Общее слово — только в паре с точным: «дизел сша», «маркетплейс закон».
Примеры ключей: «Эльвира Набиуллина» → ["набиуллин", "глава цб", "председател банк росси"];
«закон о маркетплейсах» → ["маркетплейс", "платформенн экономик", "wildberries", "ozon"]."""

JUDGE = """Ты отбираешь новости в ежедневную сводку по теме.
Новость о теме, если тема в ней главная или новость прямо её касается. Упоминание вскользь, однофамилец,
случайное совпадение слова — не о теме.
Ответь JSON-объектом {"items": [{"id": 1, "relevant": true, "reason": "..."}]} — по элементу на каждую
новость списка; reason — до 10 слов: почему о теме или нет."""

STORIES = """Ты пишешь ежедневную сводку новостей по теме для занятого читателя. Ниже — новости о теме за сутки
из разных изданий; одно событие часто описано в нескольких изданиях.
1. Склей новости об одном событии в один сюжет. Каждая новость — ровно в одном сюжете.
2. У сюжета — title: заголовок до 10 слов, и summary: 1–2 предложения о том, что произошло, с фактами и цифрами
   из новостей.
3. Сюжеты — от главного к второстепенному: выше тот, о котором написало больше изданий.
4. overview — 2–3 предложения: главное по теме за сутки.
Пиши только то, что есть в новостях, без оценок и догадок; чужое заявление — с указанием, кто его сделал.
Ответь JSON-объектом {"overview": "...", "stories": [{"title": "...", "summary": "...", "ids": [1, 4]}]}."""


def _norm(text: str) -> str:
    return text.lower().replace("ё", "е")


def clean_keywords(words) -> list[str]:
    """Ключи поиска: нижний регистр, «е» вместо «ё», без повторов, не больше 25."""
    if isinstance(words, str):
        words = words.split(",")
    keywords = []
    for word in words if isinstance(words, list) else []:
        word = " ".join(_norm(str(word)).split())[:60]
        if word and word not in keywords:
            keywords.append(word)
    return keywords[:25]


async def make_brief(query: str) -> tuple[dict, dict]:
    """Как модель поняла тему: название, что к ней относится, ключи поиска."""
    answer, usage = await llm.ask(BRIEF, f"Тема: {query}", budget=400)
    brief = {"title": str(answer.get("title") or query).strip()[:80],
             "description": str(answer.get("description") or "").strip()[:600],
             "keywords": clean_keywords(answer.get("keywords"))}
    if not brief["keywords"]:
        raise llm.LLMError("модель не предложила слов для поиска — переформулируйте тему")
    return brief, usage


def _matcher(keywords: list[str]):
    """Ключ находит новость, если каждое его слово — начало какого-то слова текста: основа ловит все падежи."""
    compiled = [(kw, [re.compile(r"(?<!\w)" + re.escape(word)) for word in kw.split()]) for kw in keywords]
    return lambda text: [kw for kw, words in compiled if all(w.search(text) for w in words)]


def find_candidates(articles: list[dict], keywords: list[str]) -> list[dict]:
    match, found, seen = _matcher(keywords), [], set()
    for article in articles:
        title = _norm(article["title"])
        hits = match(f"{title} {_norm(article['lead'])}")
        # одна новость дважды в ленте одного издания (статья и видео) — проверяется один раз
        if hits and (article["source"], title) not in seen:
            seen.add((article["source"], title))
            found.append({**article, "hits": hits, "in_title": match(title)})
    # редкий ключ весит больше: «дизел» точнее указывает на тему, чем «сша» из сотен новостей; совпадение
    # в заголовке считается ещё раз — тема в новости главная, а не в хвосте лида
    count = Counter(kw for a in found for kw in a["hits"])
    weight = {kw: math.log(len(articles) / n) for kw, n in count.items()}
    for article in found:
        article["score"] = sum(weight[kw] for kw in article["hits"] + article["in_title"])
    found.sort(key=lambda a: (a["score"], a["published"]), reverse=True)
    return found


def _line(number: int, article: dict, when: bool = False) -> str:
    """Новость строкой для промпта: проверке хватает начала лида, сюжетам нужны весь лид и время."""
    moment = datetime.fromisoformat(article["published"]).astimezone(config.MSK).strftime(", %d.%m %H:%M")
    lead = article["lead"] if when else article["lead"][:200]
    return f"[{number}] {article['source']}{moment if when else ''}: {article['title']}" + (f" — {lead}" if lead else "")


def _link(article: dict) -> dict:
    return {key: article[key] for key in ("source", "title", "url", "published")}


def _numbers(values, size: int) -> list[int]:
    """Номера новостей из ответа модели: только целые из списка, без повторов."""
    numbers = []
    for value in values if isinstance(values, list) else [values]:
        try:
            number = int(value)
        except (TypeError, ValueError):
            continue
        if 1 <= number <= size and number not in numbers:
            numbers.append(number)
    return numbers


async def _judge(brief: dict, batch: list[dict]):
    news = "\n".join(_line(i, a) for i, a in enumerate(batch, 1))
    answer, usage = await llm.ask(JUDGE, f"Тема: {brief['title']}\nЧто относится к теме: {brief['description']}"
                                         f"\n\nНовости:\n{news}", budget=300)
    verdicts = {}
    for item in answer.get("items") or []:
        if isinstance(item, dict) and (number := _numbers(item.get("id"), len(batch))):
            verdicts[batch[number[0] - 1]["id"]] = {"relevant": item.get("relevant") in (True, "true", "да"),
                                                    "reason": str(item.get("reason") or "").strip()[:120]}
    return verdicts, usage, len(batch)


async def _stories(brief: dict, relevant: list[dict]):
    news = "\n".join(_line(i, a, when=True) for i, a in enumerate(relevant, 1))
    answer, usage = await llm.ask(STORIES, f"Тема: {brief['title']}\n\nНовости:\n{news}", budget=800)
    stories, used = [], set()
    for story in answer.get("stories") or []:
        if not isinstance(story, dict):
            continue
        numbers = [n for n in _numbers(story.get("ids"), len(relevant)) if n not in used]
        summary = str(story.get("summary") or "").strip()
        if numbers and summary:
            used.update(numbers)
            stories.append({"title": str(story.get("title") or "").strip(), "summary": summary,
                            "links": sorted((_link(relevant[n - 1]) for n in numbers), key=lambda l: l["published"])})
    more = [_link(a) for n, a in enumerate(relevant, 1) if n not in used]
    return str(answer.get("overview") or "").strip(), stories, more, usage


async def build(topic: dict, auto: bool):
    """Сводка по теме за последние сутки: события шагов для страницы, в конце — сохранённая сводка."""
    if topic["id"] in running:
        yield {"type": "error", "step": "articles", "message": "Сводка по этой теме уже собирается"}
        return
    running.add(topic["id"])
    try:
        async for event in _build(topic, auto):
            yield event
    finally:
        running.discard(topic["id"])


async def _build(topic: dict, auto: bool):
    started, tokens = time.perf_counter(), 0
    brief = json.loads(topic["brief"])
    end = datetime.now(timezone.utc)
    start = end - config.WINDOW

    with db.connect() as conn:
        articles = [dict(row) for row in conn.execute(
            "SELECT id, url, source, title, lead, published FROM articles WHERE published > ? "
            "ORDER BY published DESC", (db.iso(start),))]
    by_source = Counter(a["source"] for a in articles)
    yield {"type": "articles", "count": len(articles), "sources": len(by_source)}

    found = find_candidates(articles, brief["keywords"])
    checked = found[:config.CANDIDATES]
    await feeds.fill_leads(checked)
    yield {"type": "candidates", "count": len(found), "checked": len(checked)}

    verdicts, done = {}, 0
    tasks = [asyncio.create_task(_judge(brief, checked[i:i + BATCH])) for i in range(0, len(checked), BATCH)]
    try:
        for next_done in asyncio.as_completed(tasks):
            result, usage, size = await next_done
            verdicts.update(result)
            done, tokens = done + size, tokens + usage["tokens"]
            yield {"type": "judge", "done": done, "total": len(checked),
                   "relevant": sum(v["relevant"] for v in verdicts.values())}
    except llm.LLMError as e:
        yield {"type": "error", "step": "judge", "message": f"Модель не ответила: {e}"}
        return
    finally:
        for task in tasks:
            task.cancel()
    if not tasks:
        yield {"type": "judge", "done": 0, "total": 0, "relevant": 0}

    relevant = [a for a in checked if verdicts.get(a["id"], {}).get("relevant")]
    rejected = [{**_link(a), "reason": verdicts[a["id"]]["reason"] if a["id"] in verdicts
                 else "модель пропустила эту новость"} for a in checked if a not in relevant]
    overview, stories, more = "За сутки новостей по теме не нашлось.", [], []
    if relevant:
        yield {"type": "writing", "count": min(len(relevant), STORY_LIMIT)}
        try:
            overview, stories, more, usage = await _stories(brief, relevant[:STORY_LIMIT])
        except llm.LLMError as e:
            yield {"type": "error", "step": "stories", "message": f"Модель не ответила: {e}"}
            return
        tokens += usage["tokens"]
        more += [_link(a) for a in relevant[STORY_LIMIT:]]

    stats = {"articles": len(articles), "sources": len(by_source), "candidates": len(found), "checked": len(checked),
             "relevant": len(relevant), "stories": len(stories), "tokens": tokens,
             "seconds": round(time.perf_counter() - started, 1), "model": config.LLM_MODEL}
    content = {"title": brief["title"], "from": db.iso(start), "to": db.iso(end), "overview": overview,
               "stories": stories, "more": more, "rejected": rejected, "stats": stats,
               "keywords": dict(Counter(kw for a in found for kw in a["hits"]).most_common()),
               "by_source": dict(by_source.most_common())}
    created = db.now()
    with db.connect() as conn:
        digest_id = conn.execute("INSERT INTO digests (topic_id, created, auto, content) VALUES (?, ?, ?, ?)",
                                 (topic["id"], created, int(auto), json.dumps(content, ensure_ascii=False))).lastrowid
    logger.info("Сводка «%s»: %s новостей → %s по словам → %s о теме → %s сюжетов · %s токенов · %s с", brief["title"],
                stats["articles"], stats["candidates"], stats["relevant"], stats["stories"], tokens, stats["seconds"])
    yield {"type": "done", "digest": {"id": digest_id, "created": created, "auto": auto, **content}}


async def scheduler():
    """Каждый день после 20:00 по Москве — сводка по каждой теме, у которой сегодня после 20:00 сводки ещё нет.
    Сервер был выключен в 20:00 — соберёт, как только включится; не вышло — повтор через полчаса.
    Тему моложе 5 минут не трогает: первую сводку по ней собирает страница."""
    failed = {}
    while True:
        try:
            now = datetime.now(config.MSK)
            if now.time() >= config.DIGEST_AT:
                with db.connect() as conn:
                    due = [dict(row) for row in conn.execute(
                        "SELECT * FROM topics WHERE created < ? AND id NOT IN "
                        "(SELECT topic_id FROM digests WHERE created >= ?)",
                        (db.iso(now - timedelta(minutes=5)), db.iso(datetime.combine(now.date(), config.DIGEST_AT,
                                                                                     config.MSK))))]
                for topic in due:
                    if topic["id"] in running or time.monotonic() - failed.get(topic["id"], -1e9) < 1800:
                        continue
                    error = None
                    try:
                        async for event in build(topic, auto=True):
                            error = event.get("message") if event["type"] == "error" else error
                    except Exception as e:
                        logger.exception("Сводка по теме %s упала", topic["id"])
                        error = str(e)
                    if error:
                        failed[topic["id"]] = time.monotonic()
                        logger.warning("Сводка по теме %s не собрана: %s", topic["id"], error)
        except Exception:
            logger.exception("Сводки по расписанию прервались")
        await asyncio.sleep(60)
