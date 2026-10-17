"""RAG по документации проекта: README и docs/ → фрагменты по заголовкам → векторы bge-m3 → поиск."""
import asyncio
import hashlib
import json
import re
import time

import numpy as np

from app import config, llm

MAX = 900  # знаков во фрагменте: короче промпт — быстрее отвечает локальная модель
HEADING = re.compile(r"^(#{1,3})\s+(.+?)\s*$")
CACHE = config.DATA_DIR / "embeddings.json"

_index = {"fingerprint": None, "chunks": [], "matrix": None, "built": None}
_lock = asyncio.Lock()  # сборка при старте и первый /help не считают векторы дважды


def files():
    """Файлы документации проекта по порядку: README, потом docs/."""
    found = []
    for pattern in config.DOCS:
        found += sorted(config.PROJECT_DIR.glob(pattern))
    return found


def _sections(text):
    """(заголовки над разделом, текст раздела); строки внутри ``` заголовками не считаются."""
    trail, body, in_code = [], [], False
    for line in text.splitlines():
        if line.startswith("```"):
            in_code = not in_code
        heading = None if in_code else HEADING.match(line)
        if heading:
            if "\n".join(body).strip():
                yield trail, "\n".join(body).strip()
            trail = trail[:len(heading.group(1)) - 1] + [heading.group(2)]
            body = []
        else:
            body.append(line)
    if "\n".join(body).strip():
        yield trail, "\n".join(body).strip()


def _blocks(body):
    """Абзацы; блок кода и таблица — целиком."""
    found, current, in_code = [], [], False
    for line in body.splitlines():
        if line.startswith("```"):
            in_code = not in_code
        if line.strip() or in_code:
            current.append(line)
        elif current:
            found.append("\n".join(current))
            current = []
    if current:
        found.append("\n".join(current))
    return found


def _pieces(block):
    """Блок длиннее MAX — по строкам; у таблицы в каждой части повторяется шапка."""
    if len(block) <= MAX:
        return [block]
    lines = block.splitlines()
    head = lines[:2] if lines[0].startswith("|") else []
    pieces, current = [], []
    for line in lines[len(head):]:
        if current and len("\n".join(head + current + [line])) > MAX:
            pieces.append("\n".join(head + current))
            current = []
        current.append(line)
    pieces.append("\n".join(head + current))
    return pieces


def split(path):
    """Фрагменты одного файла: {"file", "title", "text"}; title — заголовки без названия документа."""
    file = path.relative_to(config.PROJECT_DIR).as_posix()
    for trail, body in _sections(path.read_text(encoding="utf-8")):
        title = " › ".join(trail[1:] or trail[:1])
        current = ""
        for block in _blocks(body):
            for piece in _pieces(block):
                # короткое начало (вступление к списку) не отрываем от продолжения
                if len(current) >= MAX // 3 and len(current) + len(piece) + 2 > MAX:
                    yield {"file": file, "title": title, "text": current}
                    current = ""
                current = f"{current}\n\n{piece}" if current else piece
        if current:
            yield {"file": file, "title": title, "text": current}


def _key(chunk):
    return hashlib.sha256(f"{config.EMBED_MODEL}|{chunk['file']}|{chunk['title']}|{chunk['text']}".encode()).hexdigest()


async def ensure_index():
    """Индекс актуален: документацию поменяли — пересчитываются векторы только новых фрагментов."""
    async with _lock:
        return await _build()


async def _build():
    paths = files()
    if not paths:
        raise FileNotFoundError(f"В {config.PROJECT_DIR} нет README.md и docs/ — проверьте PROJECT_DIR")
    fingerprint = hashlib.sha256(b"".join(p.as_posix().encode() + p.read_bytes() for p in paths)).hexdigest()
    if fingerprint == _index["fingerprint"]:
        return info()
    started = time.perf_counter()
    chunks = [chunk for path in paths for chunk in split(path)]
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    keys = [_key(chunk) for chunk in chunks]
    missing = [i for i, key in enumerate(keys) if key not in cache]
    if missing:
        vectors = await llm.embed([f"{chunks[i]['title']}\n\n{chunks[i]['text']}" for i in missing])
        cache.update({keys[i]: vector for i, vector in zip(missing, vectors)})
        config.DATA_DIR.mkdir(exist_ok=True)
        CACHE.write_text(json.dumps({key: cache[key] for key in keys}), encoding="utf-8")  # только живые фрагменты
    matrix = np.array([cache[key] for key in keys], dtype=np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    _index.update(fingerprint=fingerprint, chunks=chunks, matrix=matrix,
                  built={"embedded": len(missing), "seconds": round(time.perf_counter() - started, 2)})
    return info()


def info():
    chunks = _index["chunks"]
    return {"files": sorted({c["file"] for c in chunks}, key=lambda f: (f != "README.md", f)),
            "chunks": len(chunks), "model": config.EMBED_MODEL, "built": _index["built"]}


async def search(question, top=config.TOP):
    """Ближайшие по смыслу фрагменты документации — по косинусу векторов bge-m3."""
    await ensure_index()
    vector = np.array((await llm.embed([question]))[0], dtype=np.float32)
    scores = _index["matrix"] @ (vector / np.linalg.norm(vector))
    best = np.argsort(-scores)[:top]
    return [{**_index["chunks"][i], "score": round(float(scores[i]), 3)} for i in best]
