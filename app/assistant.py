"""Команда /help: контекст проекта по MCP → поиск по документации → ответ локальной модели."""
import json
import time

from mcp import Client

from app import config, llm, rag

SYSTEM = """Ты — ассистент разработчика проекта ml-student-dropout. Отвечай по-русски, коротко: 2–5 предложений или список.
Опирайся только на «Контекст проекта» и «Фрагменты документации».
После утверждения из документации ставь ссылку на фрагмент — его заголовок в квадратных скобках, например [docs/model.md › Причины риска].
Про ветку, изменения и файлы отвечай по контексту проекта.
Если ответа нет ни там, ни там — так и скажи: «В документации этого нет».
Команды, пути, имена файлов и полей пиши в `обратных кавычках`."""

OVERVIEW = "Что это за проект, как он устроен и как его запустить?"
CHANGES = ("измен", "правк", "поменя", "diff", "дифф")  # вопрос про правки — нужен ещё и diff
GROUPS = {"изменён": "изменены", "новый": "новые", "удалён": "удалены", "добавлен": "добавлены",
          "переименован": "переименованы", "конфликт": "с конфликтом"}


def parse(text):
    """'/help вопрос' → вопрос; просто '/help' — вопрос о проекте в целом; без /help — None."""
    text = text.strip()
    if text != "/help" and not text.startswith("/help "):
        return None
    return text[len("/help"):].strip() or OVERVIEW


def server_info(client):
    return {"name": client.server_info.name if client.server_info else "?",
            "version": client.server_info.version if client.server_info else "",
            "protocol": client.protocol_version}


async def call(client, calls, name, args):
    """Вызов инструмента MCP: запись с временем и результатом — в calls; вернёт результат или None при ошибке."""
    started = time.perf_counter()
    result = await client.call_tool(name, args)
    text = "".join(getattr(part, "text", "") for part in result.content)
    calls.append({"tool": name, "args": args, "ok": not result.is_error,
                  "result": result.structured_content if result.structured_content is not None else text,
                  "ms": round((time.perf_counter() - started) * 1000)})
    return calls[-1]["result"] if calls[-1]["ok"] else None


async def call_tools(plan):
    """Одна сессия MCP: вызовы инструментов по очереди с временем и результатом."""
    calls = []
    async with Client(config.MCP_URL, read_timeout_seconds=30) as client:
        server = server_info(client)
        for name, args in plan:
            await call(client, calls, name, args)
    return server, calls


def context_text(calls):
    """Результаты MCP — коротким текстом для модели."""
    got = {c["tool"]: c["result"] for c in calls if c["ok"]}
    lines = [f"Проект: {config.PROJECT_DIR.name}"]
    if branch := got.get("git_branch"):
        lines.append(f"Текущая ветка: {branch['branch']}; ветки: {', '.join(branch['branches'])}; "
                     f"последний коммит: {branch['last_commit']}")
    if status := got.get("git_status"):
        groups = {}
        for c in status["changed"]:
            groups.setdefault(GROUPS.get(c["status"], c["status"]), []).append(c["path"])
        parts = [f"{name} ({len(paths)}): {', '.join(paths[:12])}{' …' if len(paths) > 12 else ''}"
                 for name, paths in groups.items()]
        lines.append("Незакоммиченных изменений нет" if status["clean"]
                     else f"Незакоммиченные изменения — {status['count']} файлов. " + "; ".join(parts))
    if files := got.get("list_files"):
        lines.append(f"Файлы проекта ({files['count']}): {', '.join(files['files'])}")
    if diff := got.get("git_diff"):
        lines.append(f"Diff ({diff['lines']} строк{', начало' if diff['truncated'] else ''}):\n{diff['diff'] or '(пусто)'}")
    return "\n".join(lines)


def messages(question, context, hits):
    # Неизменная часть — системный промпт и контекст проекта — идёт первой: Ollama кэширует начало промпта
    fragments = "\n\n".join(f"[{h['file']} › {h['title']}]\n{h['text']}" for h in hits)
    user = (f"Контекст проекта (git, получен через MCP):\n{context}\n\nВопрос: {question}\n\n"
            f"Фрагменты документации:\n\n{fragments}\n\nВопрос: {question}")
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


async def answer(question):
    """Шаги ответа событиями: mcp → sources → prompt → text… → done; ошибка — error с номером шага."""
    yield {"type": "question", "question": question}

    step = "mcp"
    try:
        started = time.perf_counter()
        plan = [("git_branch", {}), ("git_status", {}), ("list_files", {})]
        if any(word in question.lower() for word in CHANGES):
            plan.append(("git_diff", {}))
        server, calls = await call_tools(plan)
        context = context_text(calls)
        yield {"type": "mcp", "server": server, "calls": calls, "seconds": round(time.perf_counter() - started, 2)}

        step = "sources"
        started = time.perf_counter()
        hits = await rag.search(question)
        yield {"type": "sources", "hits": hits, "index": rag.info(), "seconds": round(time.perf_counter() - started, 2)}

        step = "answer"
        prompt = messages(question, context, hits)
        yield {"type": "prompt", "messages": prompt, "chars": sum(len(m["content"]) for m in prompt)}
        async for event in llm.chat(prompt):
            yield event
    except llm.LLMError as exc:
        yield {"type": "error", "step": step, "message": str(exc)}
    except Exception as exc:  # MCP, git, файлы — показываем словами, а не падением
        yield {"type": "error", "step": step, "message": reason(exc)}


def reason(exc):
    """Причина ошибки; клиент MCP бросает ExceptionGroup — достаём первую настоящую."""
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]
    return getattr(exc, "message", None) or str(exc) or type(exc).__name__


async def project():
    """Шапка страницы: проект и ветка — тоже через MCP."""
    try:
        _, calls = await call_tools([("git_branch", {})])
        return {"name": config.PROJECT_DIR.name, "branch": calls[0]["result"]["branch"] if calls[0]["ok"] else None}
    except Exception as exc:
        return {"name": config.PROJECT_DIR.name, "branch": None, "error": reason(exc)}


def dumps(event):
    return json.dumps(event, ensure_ascii=False) + "\n"
