"""Ревью pull request: изменения — через MCP-сервер git, контекст — RAG по коду и документации основной
ветки, текст — локальная модель в роли двух ревьюеров: кода и архитектуры.
Запуск: python -m app.review --base main --head feature/x."""
import argparse
import ast
import asyncio
import fnmatch
import json
import logging
import os
import re
import sys
import time
from itertools import zip_longest
from pathlib import Path

from mcp import Client

from app import config, git_mcp, llm, rag
from app.assistant import reason

MAX_DIFF = 600  # строк diff в промпт: больше локальная модель читает минутами
QUERY = 1200  # знаков добавленных строк файла в поисковый запрос
TOP_DOCS, TOP_CODE = 3, 4  # разделов документации и фрагментов кода в промпт
SECTION = 1800  # раздел документации длиннее — в промпт идёт только найденный фрагмент
# промпт ревью в 4 раза длиннее, чем у /help; seed — один и тот же PR получает одно и то же ревью
OPTIONS = {"num_ctx": 12288, "num_predict": 1200, "seed": 7}
HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@ ?(.*)")

COMMON = """Тебе дают список изменённых файлов, {context} из основной ветки и diff: число в начале строки — номер строки в новой версии файла, + — строка добавлена, - — удалена.
Одна проблема — одно замечание, даже если она видна в нескольких строках. file и line — путь и номер строки из diff; problem — что не так и к чему приведёт, одно-два предложения; fix — как исправить; source — заголовок фрагмента из квадратных скобок, если замечание на него опирается, иначе пустая строка.
Не выдумывай: нечего сказать — пустой список. Два точных замечания лучше пяти сомнительных.
По-русски и коротко. Код — в `обратных кавычках`."""

CODE = """Ты — опытный ревьюер кода проекта {project}. Ищешь в pull request ошибки.
Пройди по каждой новой или изменённой функции в diff и проверь: что будет при пустом входе, нуле, None, слишком большом входе; где деление, индекс или ключ словаря могут упасть; какое исключение дойдёт до пользователя ошибкой 500; верна ли логика по сравнению с похожим кодом проекта. Прежде чем написать о баге, найди строку, где он случится, и убедись, что проверки там нет — ни в самой функции, ни раньше: в схеме входных данных или в коде, который её вызывает; в problem назови вход, при котором он случится.
Потом — каких тестов и проверок входных данных не хватает.
kind: bug — потенциальный баг (что-то упадёт или посчитается неверно), recommendation — рекомендация (тест, проверка, упрощение). summary — одно предложение: что добавляет или меняет pull request.
Архитектуру и документацию не оценивай — это делает другой ревьюер.
""" + COMMON

ARCHITECTURE = """Ты — архитектор проекта {project}. Проверяешь, согласуется ли pull request с устройством проекта и его документацией.
findings — для каждого изменённого модуля сверь с документацией его импорты и обязанности: не появилась ли зависимость, которой у модуля быть не должно; в том ли модуле код; нет ли дублирования того, что в проекте уже есть.
docs — отдельно о документации. По оглавлению найди разделы, которые описывают то, что меняет pull request: API, модули, поведение. Если pull request это меняет, а документация в нём не менялась, в files — файлы этих разделов, в why — какие разделы и что в них теперь не так, в source — заголовок фрагмента, где это видно, или пустая строка. Документация в порядке — files пустой. О документации пиши только в docs.
Ошибки в коде не ищи — это делает другой ревьюер.
""" + COMMON

FIELDS = {"file": {"type": "string"}, "line": {"type": "integer"}, "problem": {"type": "string"},
          "fix": {"type": "string"}, "source": {"type": "string"}}


def findings(top, **extra):
    """JSON Schema списка замечаний: Ollama не даст модели отойти от неё."""
    item = {"type": "object", "required": [*extra, *FIELDS], "properties": {**extra, **FIELDS}}
    return {"type": "array", "items": item, "maxItems": top}


CODE_SCHEMA = {"type": "object", "required": ["summary", "findings"], "properties": {
    "summary": {"type": "string"},
    "findings": findings(5, kind={"type": "string", "enum": ["bug", "recommendation"]})}}
ARCHITECTURE_SCHEMA = {"type": "object", "required": ["findings", "docs"], "properties": {
    "findings": findings(3),
    "docs": {"type": "object", "required": ["files", "why", "source"], "properties": {
        "files": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
        "why": {"type": "string"}, "source": {"type": "string"}}}}}

index = rag.Index("review-embeddings")  # документация и код основной ветки


def log(text):
    print(text, file=sys.stderr, flush=True)


def short(ref):
    return ref[:7] if re.fullmatch(r"[0-9a-f]{40}", ref) else ref


def number(x, digits=1):
    return f"{x:,.{digits}f}".replace(",", " ").replace(".", ",")


FILES, CHUNKS, TOKENS = ("файл", "файла", "файлов"), ("фрагмент", "фрагмента", "фрагментов"), ("токен", "токена", "токенов")


def count(n, words):
    """«1 файл», «3 файла», «12 файлов»."""
    one, few, many = words
    word = one if n % 10 == 1 and n % 100 != 11 else few if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14 else many
    return f"{number(n, 0)} {word}"


def matches(path, patterns):
    return any(fnmatch.fnmatch(path, p) for p in patterns)


async def collect(base, head):
    """① Изменения PR и файлы основной ветки для RAG — одна сессия с MCP-сервером git в этом же процессе."""
    calls = []
    async with Client(git_mcp.server) as client:
        async def call(tool, **args):
            calls.append(tool)
            result = await client.call_tool(tool, args)
            if result.is_error:
                raise RuntimeError("".join(getattr(part, "text", "") for part in result.content))
            return result.structured_content

        pr = await call("pr_files", base=base, head=head)
        diff = await call("pr_diff", base=base, head=head)
        listed = await call("list_files", ref=base)
        wanted = [f for f in listed["files"] if matches(f, config.DOCS + config.CODE)]
        texts = {f: (await call("read_file", path=f, ref=base))["text"] for f in wanted}
        # новые версии изменённых файлов кода — ради их импортов: в diff начала файла часто нет
        heads = {f["path"]: (await call("read_file", path=f["path"], ref=head))["text"] for f in pr["files"]
                 if f["status"] != "удалён" and matches(f["path"], config.CODE)}
    return pr, diff, texts, heads, calls


def imports(text):
    """Последняя строка начала модуля — описания и импортов; 0 — если файл начинается не с них."""
    try:
        nodes = ast.parse(text).body
    except SyntaxError:
        return 0
    end = 0
    for node in nodes:
        is_doc = isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
        if not (is_doc or isinstance(node, (ast.Import, ast.ImportFrom))):
            break
        end = node.end_lineno
    return end


def parse(diff):
    """Unified diff → {путь: {"lines": строки с номерами новой версии, "added": добавленные строки,
    "numbers": номера, которые видит модель}}. Содержимое удалённых файлов пропускается."""
    files, current, header, new = {}, None, True, 0
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            current, header = None, True
        elif header and line.startswith("+++ "):
            path = line[4:].rstrip("\t")
            current = None if path == "/dev/null" else files.setdefault(
                path[2:], {"lines": [], "added": [], "numbers": set()})
        elif line.startswith("@@") and (hunk := HUNK.match(line)):
            header, new = False, int(hunk.group(1))
            if current is not None:
                current["lines"].append(f"@@ {hunk.group(2)}".rstrip())
        elif header or current is None:
            continue
        elif line.startswith(("+", " ")):
            mark, text = line[0], line[1:]
            current["lines"].append(f"{new:>4} {mark} {text}")
            current["numbers"].add(new)
            if mark == "+":
                current["added"].append(text)
            new += 1
        elif line.startswith("-"):
            current["lines"].append(f"     - {line[1:]}")
    return files


def with_imports(parsed, heads):
    """У файла кода, чей diff не начинается с первой строки, в начало — его импорты: без них модель
    «не видит» импорт и пишет, что его нет."""
    for path, text in heads.items():
        item = parsed.get(path)
        if item and item["lines"] and 1 not in item["numbers"] and (end := imports(text)):
            top = text.splitlines()[:end]
            item["lines"][:0] = ["@@ начало файла"] + [f"{n:>4}   {line}" for n, line in enumerate(top, 1)]
            item["numbers"].update(range(1, end + 1))


def diff_text(files, parsed):
    """Diff для промпта: файл за файлом, пока не кончится запас в MAX_DIFF строк."""
    out, budget = [], MAX_DIFF
    for f in files:
        lines = parsed.get(f["path"], {}).get("lines", [])
        if f["status"] == "удалён" or not lines:
            out.append(f"### {f['path']} — {f['status'] if f['status'] == 'удалён' else 'без текстовых изменений'}")
            continue
        shown = lines[:budget]
        out += [f"### {f['path']}", *shown]
        if len(shown) < len(lines):
            out.append(f"… ещё {len(lines) - len(shown)} строк diff не поместились")
        budget -= len(shown)
    return "\n".join(out)


def pick(vectors, kind, top):
    """Лучший фрагмент по каждому запросу, потом второй… — без повторов. У документации заголовок
    фрагмента — раздел, и раздел, разрезанный на части, попадает в промпт целиком."""
    ranked = [index.nearest(vector, top, keep=lambda chunk: chunk["kind"] == kind) for vector in vectors]
    found = {}
    for hits in zip_longest(*ranked):
        for hit in filter(None, hits):
            if len(found) < top:
                found.setdefault(rag.source(hit), hit)
    if kind == "docs":
        for key, hit in found.items():
            whole = "\n\n".join(c["text"] for c in index.chunks if c["kind"] == "docs" and rag.source(c) == key)
            if len(whole) <= SECTION:
                found[key] = {**hit, "text": whole}
    return list(found.values())


def contents():
    """Оглавление документации: файл — его разделы по порядку. С ним ревьюер видит, где что описано,
    даже если поиск принёс не те разделы."""
    titles = {}
    for chunk in index.chunks:
        if chunk["kind"] == "docs":
            titles.setdefault(chunk["file"], {}).setdefault(chunk["title"])
    return "\n".join(f"{file}: {' · '.join(found)}" for file, found in titles.items())


def message(title, pr, hits, diff, what, task, docs=None):
    """Промпт ревьюера; docs — изменённые в PR файлы документации: ревьюеру архитектуры — с оглавлением."""
    listed = "\n".join(f"- {f['path']} ({f['status']}, +{f['added']} −{f['deleted']})" for f in pr["files"])
    note = "" if docs is None else (f"\nДокументация в pull request: {', '.join(docs) or 'не менялась'}\n\n"
                                    f"Оглавление документации проекта:\n{contents()}")
    fragments = "\n\n".join(f"[{rag.source(h)}]\n{h['text']}" for h in hits) or "(ничего не нашлось)"
    text = (f"Pull request: {title or '(без заголовка)'}\n"
            f"Изменённые файлы ({len(pr['files'])}, +{pr['added']} −{pr['deleted']}):\n{listed}{note}\n\n"
            f"{what} из основной ветки:\n\n{fragments}\n\nDiff:\n\n{diff}\n\nЗадание: {task}")
    # модель цитирует код дословно, а двойная кавычка внутри ответа-JSON закрывает строку, и замечание
    # обрывается на полуслове; для Python одинарные кавычки — то же самое
    return text.replace('"', "'")


async def ask(system, user, schema):
    """Один ревьюер: ответ модели строго по JSON Schema и счётчики."""
    answer, counters = "", {}
    prompt = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    async for event in llm.chat(prompt, options=OPTIONS, schema=schema):
        if event["type"] == "text":
            answer += event["text"]
        else:
            counters = event
    try:
        return json.loads(answer), counters
    except json.JSONDecodeError:
        raise llm.LLMError(f"Модель вернула не JSON — ответ оборвался на {count(counters.get('tokens', 0), TOKENS)}?")


def repeats_docs(finding, stale):
    """Замечание советует дописать документацию, которая уже отмечена устаревшей."""
    return "документац" in finding["fix"].lower() or any(s in finding["problem"] + finding["fix"] for s in stale)


async def review(base, head, title=""):
    """① изменения через MCP → ② RAG → ③ ревьюер кода → ④ ревьюер архитектуры. Возвращает Markdown."""
    started = time.perf_counter()
    pr, diff, texts, heads, calls = await collect(base, head)
    files = pr["files"]
    if not files:
        return f"### AI-ревью\n\nИзменений между `{short(base)}` и `{short(head)}` нет.\n"
    parsed = parse(diff["diff"])
    tools = ", ".join(dict.fromkeys(f"{t} ×{calls.count(t)}" if calls.count(t) > 1 else t for t in calls))
    log(f"① изменения — MCP git ({tools}): {count(len(files), FILES)}, +{pr['added']} −{pr['deleted']} · "
        f"{number(time.perf_counter() - started)} с")

    started = time.perf_counter()
    await index.build(texts)
    # на каждый изменённый файл три запроса: что о модуле сказано в документации; какая документация
    # ближе к новым строкам; какой код проекта ближе к ним (у фрагментов кода в векторе есть путь)
    changed = [(path, "\n".join(item["added"])[:QUERY]) for path, item in parsed.items() if item["added"]]
    queries = [q for path, added in changed for q in (f"Модуль {path}: за что отвечает и о чём знает", added,
                                                      f"{path}\n{added}")]
    vectors = await llm.embed(queries) if queries else []
    docs = pick([v for i, v in enumerate(vectors) if i % 3 < 2], "docs", TOP_DOCS)
    code = pick(vectors[2::3], "code", TOP_CODE)
    log(f"② RAG по коду и документации {short(base)}: {count(len(index.chunks), CHUNKS)}, "
        f"новых векторов {index.built['embedded']} · {number(time.perf_counter() - started)} с")
    for hit in code + docs:
        log(f"   {rag.source(hit)} · {number(hit['score'], 2)}")

    with_imports(parsed, heads)
    diff = diff_text(files, parsed)
    project = config.PROJECT_DIR.name
    # ревьюеру кода diff документации не нужен — только удлиняет промпт
    code_diff = diff_text([f for f in files if not matches(f["path"], config.DOCS)], parsed)
    found, counters = await ask(
        CODE.format(project=project, context="фрагменты кода проекта"),
        message(title, pr, code, code_diff, "Фрагменты кода проекта", "найди потенциальные баги и дай рекомендации."),
        CODE_SCHEMA)
    log(f"③ ревьюер кода — {config.MODEL}: {spent(counters)}")
    flagged, counters2 = await ask(
        ARCHITECTURE.format(project=project, context="разделы документации проекта"),
        message(title, pr, docs, diff, "Разделы документации проекта",
                "найди архитектурные проблемы и проверь, не устарела ли документация.",
                [f["path"] for f in files if matches(f["path"], config.DOCS)]),
        ARCHITECTURE_SCHEMA)
    log(f"④ ревьюер архитектуры — {config.MODEL}: {spent(counters2)}")

    # устаревшая документация — файлы, которые есть в основной ветке и не менялись в PR;
    # если она нашлась, замечание с советом дописать документацию — повтор
    stale = [f for f in dict.fromkeys(flagged["docs"]["files"]) if f in texts and f not in parsed]
    architecture = [f for f in flagged["findings"] if not (stale and repeats_docs(f, stale))]
    if stale:
        architecture.append({"file": stale[0], "line": 0,
                             "problem": flagged["docs"]["why"] or "Документация не обновлена под изменения.",
                             "fix": "обновить " + ", ".join(f"`{f}`" for f in stale) + ".",
                             "source": flagged["docs"]["source"]})
    sections = {"Потенциальные баги": [f for f in found["findings"] if f["kind"] == "bug"],
                "Архитектурные проблемы": architecture,
                "Рекомендации": [f for f in found["findings"] if f["kind"] == "recommendation"]}
    return render(found["summary"], sections, pr, parsed, texts, code + docs, tools, [counters, counters2], base)


def where(item, pr, parsed, texts):
    """Ссылка на место замечания. Номер строки — только если модель видела его в diff;
    на GitHub — ссылка на файл в коммите PR."""
    path, line = item["file"].strip("` ").split(":")[0], item["line"]
    if path not in parsed and path not in texts:
        return None
    line = line if line in parsed.get(path, {}).get("numbers", ()) else None
    label = f"{path}:{line}" if line else path
    repo = os.getenv("GITHUB_REPOSITORY")
    if not repo:
        return f"`{label}`"
    url = f"{os.getenv('GITHUB_SERVER_URL', 'https://github.com')}/{repo}/blob/{pr['head_sha']}/{path}"
    return f"[`{label}`]({url}{f'#L{line}' if line else ''})"


def spent(counters):
    return (f"промпт {count(counters['prompt'], TOKENS)} — {number(counters['reading'])} с, "
            f"ответ {count(counters['tokens'], TOKENS)} — {number(counters['writing'])} с")


def render(summary, sections, pr, parsed, texts, hits, tools, counters, base):
    out = [f"### AI-ревью · {config.MODEL}", "", f"**Что делает PR.** {summary}"]
    sources = [rag.source(h) for h in hits]
    for title, items in sections.items():
        out += ["", f"#### {title}", ""]
        if not items:
            out.append("Не найдено.")
        for n, item in enumerate(items, 1):
            place = where(item, pr, parsed, texts)
            # основание — только «файл › раздел» из фрагментов, которые модель правда видела в промпте
            source = item["source"].strip("[]` ")
            seen = "›" in source and any(s.startswith(source) or source.startswith(s) for s in sources)
            out.append(f"{n}. {place + ' — ' if place else ''}{item['problem']}  ")
            out.append(f"   **Исправить:** {item['fix']}" + (f"  \n   *Основание: {source}*" if seen else ""))
    fragments = " · ".join(f"`{rag.source(h)}` {number(h['score'], 2)}" for h in hits)
    model = "; ".join(f"{who} — {spent(c)}" for who, c in zip(("ревьюер кода", "ревьюер архитектуры"), counters))
    out += ["", "<details><summary>Как получено ревью</summary>", "",
            f"- **Изменения** — MCP-сервер git ({tools}): {count(len(pr['files']), FILES)}, "
            f"+{pr['added']} −{pr['deleted']}.",
            f"- **Контекст** — RAG по коду и документации `{short(base)}`: {count(len(index.chunks), CHUNKS)}, "
            f"в промпт — {len(hits)}: {fragments}.",
            f"- **Модель** — {config.MODEL} в Ollama: {model}.",
            "", "</details>"]
    return "\n".join(out) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Ревью pull request локальной моделью")
    parser.add_argument("--repo", default=str(config.PROJECT_DIR), help="папка git-репозитория проекта")
    parser.add_argument("--base", default="main", help="ветка или коммит, куда вливается PR")
    parser.add_argument("--head", default="HEAD", help="ветка или коммит с изменениями")
    parser.add_argument("--title", default="", help="заголовок PR")
    parser.add_argument("--out", help="файл для ревью в Markdown; без него — на экран")
    args = parser.parse_args()
    config.PROJECT_DIR = Path(args.repo).resolve()
    logging.getLogger("httpx2").setLevel(logging.WARNING)  # MCP SDK включает INFO — без строк о каждом запросе
    try:
        text = asyncio.run(review(args.base, args.head, args.title))
    except Exception as exc:  # причина словами — её видно в логе задания
        log(f"Ревью не получилось: {reason(exc)}")
        sys.exit(1)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
