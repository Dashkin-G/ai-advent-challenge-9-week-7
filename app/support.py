"""Ассистент поддержки: обращение и клиент из CRM по MCP → поиск по FAQ и документации → ответ клиенту."""
import asyncio
import json
import time

from mcp import Client

from app import assistant, config, llm, rag

SYSTEM = """Ты — ассистент поддержки сервиса «Риск отчисления студентов». Пишешь ответ клиенту от имени поддержки: по-русски, на «вы», начни с обращения по имени клиента из CRM; коротко — 3–6 предложений или список шагов.
Опирайся только на «Данные из CRM» и «Фрагменты базы знаний». Фрагменты найдены поиском и бывают не по теме: бери только те, что отвечают на вопрос клиента.
Если причина видна в технических данных обращения или в профиле клиента — назови её конкретно: поле, значение, текст ошибки, команду — и скажи, что сделать. Прошлые обращения упоминай, только если они о той же проблеме.
Учитывай, кто клиент и как он работает с сервисом: дашборд — простыми словами и кнопками страницы; API или Docker — полями, запросами и командами. Если сделать нужное клиент сам не может, а в профиле сказано, кто ведёт сервис, — подскажи, к кому обратиться.
После утверждения из базы знаний ставь ссылку на фрагмент — его заголовок в квадратных скобках, например [FAQ › Что значит ошибка 422?].
Не придумывай полей, команд, настроек, кнопок и эндпоинтов, которых нет во фрагментах и данных CRM. Если ответа нет — так и скажи и пообещай передать вопрос разработчикам.
Поля, команды и адреса пиши в `обратных кавычках`."""

NO_TICKET = "Обращения нет: вопрос о продукте без данных клиента — отвечай без обращения по имени."

index = rag.Index("support-embeddings")  # FAQ поддержки + README и docs/ проекта
_lock = asyncio.Lock()  # сборка при старте и первый вопрос не считают векторы дважды


async def ensure_index():
    """База знаний актуальна: поменяли FAQ или документацию — пересчитываются векторы только новых фрагментов."""
    async with _lock:
        texts = {"FAQ": config.FAQ.read_text(encoding="utf-8")}
        texts.update({p.relative_to(config.PROJECT_DIR).as_posix(): p.read_text(encoding="utf-8") for p in rag.files()})
        await index.build(texts)
        return index.info()


async def queue():
    """Очередь обращений для страницы — тоже через MCP."""
    started, calls = time.perf_counter(), []
    try:
        async with Client(config.CRM_MCP_URL, read_timeout_seconds=30) as client:
            server = assistant.server_info(client)
            found = await assistant.call(client, calls, "list_tickets", {})
    except Exception as exc:
        return {"error": assistant.reason(exc)}
    if found is None:
        return {"error": calls[0]["result"]}
    return {"server": server, "tickets": found["tickets"], "ms": round((time.perf_counter() - started) * 1000)}


async def crm(ticket_id):
    """Одна сессия MCP: обращение, затем его клиент — с временем и результатом каждого вызова."""
    calls = []
    async with Client(config.CRM_MCP_URL, read_timeout_seconds=30) as client:
        server = assistant.server_info(client)
        ticket = await assistant.call(client, calls, "get_ticket", {"ticket_id": ticket_id})
        user = await assistant.call(client, calls, "get_user", {"user_id": ticket["user_id"]}) if ticket else None
    return server, calls, ticket, user


def context_text(ticket, user):
    """Клиент и обращение — коротким текстом для модели."""
    lines = [f"Клиент: {user['name']} — {user['role']}, {user['org']}. "
             f"Работает с сервисом через {user['channel']}: {user['setup']}."]
    if user.get("notes"):
        lines.append(f"Заметки поддержки: {user['notes']}")
    if past := [t for t in user["tickets"] if t["id"] != ticket["id"]]:
        lines.append("Прошлые обращения: " + "; ".join(
            f"{t['id']} «{t['subject']}» — {t['status']}" + (f": {t['resolution']}" if t.get("resolution") else "")
            for t in past))
    lines.append(f"Обращение {ticket['id']} «{ticket['subject']}» — {ticket['status']}, "
                 f"приоритет {ticket['priority']}, {ticket['created']}.")
    lines.append(f"Сообщение клиента: {ticket['message']}")
    if ticket["technical"]:
        lines.append("Технические данные обращения:")
        lines += [f"- {k}: {v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)}"
                  for k, v in ticket["technical"].items()]
    return "\n".join(lines)


def queries(question, ticket):
    """Запросы к базе знаний: вопрос и тексты ошибок сервиса из обращения — каждый отдельно. Сообщение клиента
    в запрос не идёт: длинный текст размывает вектор, и нужный раздел FAQ уходит с первого места."""
    errors = [v["details"] for v in (ticket or {}).get("technical", {}).values()
              if isinstance(v, dict) and v.get("details")]
    return list(dict.fromkeys([question, *errors]))


def messages(question, context, hits):
    fragments = "\n\n".join(f"[{rag.source(h)}]\n{h['text']}" for h in hits)
    crm_part = f"Данные из CRM (получены через MCP):\n{context}" if context else NO_TICKET
    user = f"{crm_part}\n\nВопрос: {question}\n\nФрагменты базы знаний:\n\n{fragments}\n\nВопрос: {question}"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


async def answer(ticket_id, question):
    """Шаги ответа событиями: mcp (если выбрано обращение) → sources → prompt → text… → done;
    ошибка — error с шагом."""
    yield {"type": "question", "question": question, "ticket_id": ticket_id}
    step = "mcp"
    try:
        ticket = context = None
        if ticket_id:
            started = time.perf_counter()
            server, calls, ticket, user = await crm(ticket_id)
            yield {"type": "mcp", "server": server, "calls": calls, "seconds": round(time.perf_counter() - started, 2)}
            if not user:
                yield {"type": "error", "step": step, "message": next(c["result"] for c in calls if not c["ok"])}
                return
            context = context_text(ticket, user)

        step = "sources"
        started = time.perf_counter()
        await ensure_index()
        asked = queries(question, ticket)
        hits = index.nearest(await llm.embed(asked), config.TOP)
        yield {"type": "sources", "hits": hits, "queries": asked, "index": index.info(),
               "seconds": round(time.perf_counter() - started, 2)}

        step = "answer"
        prompt = messages(question, context, hits)
        yield {"type": "prompt", "messages": prompt, "chars": sum(len(m["content"]) for m in prompt)}
        async for event in llm.chat(prompt, options={"seed": 42}):  # то же обращение — тот же ответ
            yield event
    except llm.LLMError as exc:
        yield {"type": "error", "step": step, "message": str(exc)}
    except Exception as exc:  # MCP, файл CRM — показываем словами, а не падением
        yield {"type": "error", "step": step, "message": assistant.reason(exc)}
