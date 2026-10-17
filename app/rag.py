"""RAG по проекту: документация — фрагменты по заголовкам, код Python — по функциям и классам;
векторы bge-m3, поиск по косинусу."""
import ast
import asyncio
import hashlib
import json
import re
import time

import numpy as np

from app import config, llm

MAX = 900  # знаков во фрагменте документации: короче промпт — быстрее отвечает локальная модель
MAX_CODE = 1500  # знаков во фрагменте кода: функция обычно помещается целиком
HEADING = re.compile(r"^(#{1,3})\s+(.+?)\s*$")


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


def split_docs(file, text):
    """Фрагменты Markdown: {"file", "title", "text", "kind"}; title — заголовки без названия документа."""
    for trail, body in _sections(text):
        title = " › ".join(trail[1:] or trail[:1])
        current = ""
        for block in _blocks(body):
            for piece in _pieces(block):
                # короткое начало (вступление к списку) не отрываем от продолжения
                if len(current) >= MAX // 3 and len(current) + len(piece) + 2 > MAX:
                    yield {"file": file, "title": title, "text": current, "kind": "docs"}
                    current = ""
                current = f"{current}\n\n{piece}" if current else piece
        if current:
            yield {"file": file, "title": title, "text": current, "kind": "docs"}


def _label(node):
    """Название верхнеуровневого узла Python для заголовка фрагмента."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return f"{node.name}()"
    if isinstance(node, ast.ClassDef):
        return f"class {node.name}"
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return "импорты"
    if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
        return ", ".join(ast.unparse(t) for t in getattr(node, "targets", None) or [node.target])
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
        return "описание модуля"
    return "код модуля"


def split_code(file, text):
    """Фрагменты Python: функция или класс целиком, импорты и константы между ними — вместе; комментарий
    над узлом — с ним. У фрагмента — номера строк: на них ссылается ревью."""
    lines = text.splitlines()
    try:
        nodes = ast.parse(text).body
    except SyntaxError:
        nodes = []
    groups, start = [], 1  # [названия, первая строка, последняя строка, отдельный ли узел]
    for node in nodes:
        whole = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        last = groups[-1] if groups else None
        if last and not whole and not last[3] and len("\n".join(lines[last[1] - 1:node.end_lineno])) <= MAX_CODE:
            last[0].append(_label(node))
            last[2] = node.end_lineno
        else:
            groups.append([[_label(node)], start, node.end_lineno, whole])
        start = node.end_lineno + 1
    if not groups and lines:
        groups.append([["код модуля"], 1, len(lines), False])
    for labels, first, end, _ in groups:
        while first < end and not lines[first - 1].strip():
            first += 1
        title = ", ".join(dict.fromkeys(labels))
        piece = first
        for n in range(first, end + 1):  # длинный фрагмент — по строкам
            if n > piece and len("\n".join(lines[piece - 1:n])) > MAX_CODE:
                yield {"file": file, "title": title, "text": "\n".join(lines[piece - 1:n - 1]),
                       "kind": "code", "lines": [piece, n - 1]}
                piece = n
        yield {"file": file, "title": title, "text": "\n".join(lines[piece - 1:end]), "kind": "code",
               "lines": [piece, end]}


def split(file, text):
    return split_code(file, text) if file.endswith(".py") else split_docs(file, text)


def source(chunk):
    """Заголовок фрагмента для промпта и ссылок: файл › раздел, у кода — ещё строки."""
    lines = f", строки {chunk['lines'][0]}–{chunk['lines'][1]}" if chunk.get("lines") else ""
    return f"{chunk['file']} › {chunk['title']}{lines}"


def _key(chunk):
    return hashlib.sha256(f"{config.EMBED_MODEL}|{chunk['file']}|{chunk['title']}|{chunk['text']}".encode()).hexdigest()


def _embedded(chunk):
    """Текст для вектора; у кода — с путём файла: запросы ревью тоже начинаются с пути."""
    if chunk["kind"] == "code":
        return f"{chunk['file']} › {chunk['title']}\n\n{chunk['text']}"
    return f"{chunk['title']}\n\n{chunk['text']}"


class Index:
    """Фрагменты и их векторы. Векторы кэшируются в data/<cache>.json: поменялись файлы —
    пересчитываются только новые фрагменты."""

    def __init__(self, cache):
        self.cache = config.DATA_DIR / f"{cache}.json"
        self.fingerprint, self.chunks, self.matrix, self.built = None, [], None, None

    async def build(self, texts):
        """texts — {путь: текст}."""
        fingerprint = hashlib.sha256(json.dumps(texts, sort_keys=True).encode()).hexdigest()
        if fingerprint == self.fingerprint:
            return
        started = time.perf_counter()
        chunks = [chunk for file, text in texts.items() for chunk in split(file, text)]
        cache = json.loads(self.cache.read_text(encoding="utf-8")) if self.cache.exists() else {}
        keys = [_key(chunk) for chunk in chunks]
        missing = [i for i, key in enumerate(keys) if key not in cache]
        if missing:
            vectors = await llm.embed([_embedded(chunks[i]) for i in missing])
            cache.update({keys[i]: vector for i, vector in zip(missing, vectors)})
            self.cache.parent.mkdir(parents=True, exist_ok=True)
            self.cache.write_text(json.dumps({key: cache[key] for key in keys}), encoding="utf-8")  # только живые
        matrix = None
        if keys:
            matrix = np.array([cache[key] for key in keys], dtype=np.float32)
            matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
        self.fingerprint, self.chunks, self.matrix = fingerprint, chunks, matrix
        self.built = {"embedded": len(missing), "seconds": round(time.perf_counter() - started, 2)}

    def info(self):
        return {"files": sorted({c["file"] for c in self.chunks}, key=lambda f: (f != "README.md", f)),
                "chunks": len(self.chunks), "model": config.EMBED_MODEL, "built": self.built}

    def nearest(self, vector, top, keep=lambda chunk: True):
        """Ближайшие по косинусу фрагменты, которые проходят фильтр keep, — с близостью. Можно передать
        несколько векторов запроса: у фрагмента берётся лучшая близость."""
        if self.matrix is None:
            return []
        queries = np.atleast_2d(np.array(vector, dtype=np.float32))
        scores = (self.matrix @ (queries / np.linalg.norm(queries, axis=1, keepdims=True)).T).max(axis=1)
        found = []
        for i in np.argsort(-scores):
            if len(found) == top:
                break
            if keep(self.chunks[i]):
                found.append({**self.chunks[i], "score": round(float(scores[i]), 3)})
        return found


docs = Index("embeddings")  # /help: README и docs/ из рабочей папки проекта
_lock = asyncio.Lock()  # сборка при старте и первый /help не считают векторы дважды


async def ensure_index():
    """Индекс документации актуален: поменяли файлы — пересчитываются векторы только новых фрагментов."""
    async with _lock:
        paths = files()
        if not paths:
            raise FileNotFoundError(f"В {config.PROJECT_DIR} нет README.md и docs/ — проверьте PROJECT_DIR")
        await docs.build({p.relative_to(config.PROJECT_DIR).as_posix(): p.read_text(encoding="utf-8") for p in paths})
        return docs.info()


def info():
    return docs.info()


async def search(question, top=config.TOP):
    """Ближайшие по смыслу фрагменты документации — по косинусу векторов bge-m3."""
    await ensure_index()
    return docs.nearest((await llm.embed([question]))[0], top)
