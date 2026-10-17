"""Ассистент по файлам проекта: задача словами → модель сама вызывает инструменты MCP — ищет, читает, правит
и создаёт файлы → правки копятся в черновике → diff, который пользователь применяет или отменяет."""
import json
import time

from mcp import Client

from app import assistant, config, llm

SYSTEM = """Ты — ассистент разработчика проекта ml-student-dropout: сервис на FastAPI и scikit-learn — API в src/api, модель в src/ml, схемы в src/schemas, дашборд в static/, тесты в tests/, документация в README.md и docs/. Задачу выполняешь сам через инструменты: ищешь, читаешь, правишь и создаёшь файлы проекта. Пользователя ни о чём не спрашивай.
Правила:
- Начинай с search, а не с чтения файлов целиком. Ищи так, как написано в файлах: имена функций, классов, констант, адреса эндпоинтов, числа, слова из документации; несколько вариантов — в одном запросе через |.
- read_file — только строки вокруг найденного: start и end по номерам из search.
- edit_file: old — точная строка из файла без номера, new — она же после правки; каждое место — отдельной правкой. Новый файл — write_file. Правки идут в черновик, на диск их запишет пользователь.
- Меняй только то, что требует задача. Не выдумывай файлов, функций, чисел и фактов — бери только то, что видел в файлах и git; замеры (время, скорость, проценты) не пересчитывай.
Типовые задачи:
- «Где используется X» — search по всем вариантам имени (функция, класс, эндпоинт, русское название) и перечисли каждое место.
- «Обнови документацию по изменениям в коде» — git_diff; по строкам «-» и «+» пойми, что было и что стало. Документация пишет значения, а не имена констант: поменялось число или строка — search прежнего значения по всему проекту. Поправь в README.md и docs/ каждое место с прежним значением, другие места не трогай. Перед отчётом — search прежнего значения ещё раз: в документации его не осталось.
- «Составь CHANGELOG по истории» — git_log с oldest_first=true, затем write_file: «# Журнал изменений» и по строке на коммит в том же порядке — «- дата — что изменилось (хеш)», например «- дд.мм.гггг — Добавлена выгрузка отчёта в CSV (1a2b3c4)». Даты — как в git_log. Что изменилось — по сообщению и файлам коммита, по-русски своими словами; у общего сообщения вроде «init commit» — какие части сервиса появились, словами, без путей; термины вроде pull request и CI не переводи. Без других заголовков, списков файлов и чисел строк.
Когда задача выполнена, ответь без вызова инструментов: коротко по-русски, что нашёл (`файл:строка`) и что изменил — только правки, которые прошли."""

# Что получает модель; apply_draft и discard_draft зовёт только код — по кнопке пользователя
MODEL_TOOLS = {"files": ["list_files", "read_file", "search", "edit_file", "write_file"],
               "git": ["git_status", "git_diff", "git_log"]}
MAX_STEPS = 12  # ходов модели; дальше — просим подвести итог
OPTIONS = {"num_ctx": 16384, "num_predict": 1500, "seed": 7}  # разговор с файлами длинный; seed — та же задача, тот же путь
BLOCKS = {"text", "diff"}  # текст файла и diff модель получает как есть, а не строкой JSON
FINISH = "Шаги закончились. Не вызывай инструменты — коротко отчитайся: что нашёл и что изменил."
# 4B-модель пропускает проверку из промпта; после правок код один раз просит её сам — что искать, решает модель
CHECK = ("Перед отчётом проверь себя: search того, что было до правки, по всему проекту — не осталось ли его в "
         "документации. Осталось — поправь. Потом дай итоговый отчёт: что нашёл (`файл:строка`) и что изменил.")


def spec(tool):
    """Инструмент MCP → функция для Ollama; заголовки из JSON Schema модели не нужны — лишние токены."""
    def clean(node):
        if isinstance(node, dict):
            return {k: clean(v) for k, v in node.items() if not (k == "title" and isinstance(v, str))}
        return node
    return {"type": "function",
            "function": {"name": tool.name, "description": tool.description, "parameters": clean(tool.input_schema)}}


def for_model(result):
    """Результат инструмента — текстом: файл, совпадения и diff — строками как есть, список записей — по строке
    на запись. В JSON кавычки экранируются, и модель копирует \\" в правку, а по вложенному JSON 4B-модель
    теряет порядок записей."""
    if not isinstance(result, dict):
        return str(result)
    head, blocks = {}, []
    for key, value in result.items():
        if key in BLOCKS and isinstance(value, str):
            blocks.append(value)
        elif isinstance(value, list) and value and all(isinstance(v, str) for v in value):
            blocks.append("\n".join(value))
        elif isinstance(value, list) and value and all(isinstance(v, dict) for v in value):
            blocks.append("\n".join(" · ".join(f"{k}: {', '.join(map(str, v)) if isinstance(v, list) else v}"
                                               for k, v in item.items()) for item in value))
        else:
            head[key] = value
    return "\n".join([json.dumps(head, ensure_ascii=False), *blocks])


async def turn(messages, tools=None):
    """Один ход модели: текст потоком, в конце — {"type": "turn"} с текстом, вызовами и счётчиками."""
    text, calls, counters = "", [], {}
    async for event in llm.chat(messages, options=OPTIONS, tools=tools):
        if event["type"] == "text":
            text += event["text"]
            yield event
        elif event["type"] == "tool_calls":
            calls += event["calls"]
        else:
            counters = event
    yield {"type": "turn", "text": text, "calls": calls, "llm": counters}


async def run(goal):
    """Работа над задачей событиями: goal → context → (thinking → text… → step)… → [check → ещё шаги] → report →
    draft → done; ошибка — error."""
    yield {"type": "goal", "goal": goal}
    started, step = time.perf_counter(), 0
    try:
        async with Client(config.FILES_MCP_URL, read_timeout_seconds=60) as files, \
                Client(config.MCP_URL, read_timeout_seconds=60) as git:
            await files.call_tool("discard_draft", {})  # неприменённые правки прошлой задачи сюда не смешиваем
            servers = {"files": files, "git": git}

            # контекст зовёт код: ветка, изменения и список файлов — чтобы модель не гадала пути
            calls = []
            await assistant.call(git, calls, "git_branch", {})
            await assistant.call(git, calls, "git_status", {})
            await assistant.call(files, calls, "list_files", {})
            owners, specs, tools = {}, [], []
            for name, client in servers.items():
                for tool in (await client.list_tools()).tools:
                    if tool.name in MODEL_TOOLS[name]:
                        owners[tool.name] = name
                        specs.append(spec(tool))
                        tools.append({"name": tool.name, "server": name, "description": tool.description})
            messages = [{"role": "system", "content": SYSTEM},
                        {"role": "user", "content": f"{assistant.context_text(calls)}\n\nЗадача: {goal}"}]
            yield {"type": "context", "calls": calls, "tools": tools, "prompt": messages,
                   "servers": {name: assistant.server_info(client) for name, client in servers.items()},
                   "seconds": round(time.perf_counter() - started, 2)}
            # вызов → его запись: 4B-модель иногда повторяет тот же вызов. Ключ — с версией черновика: после
            # правки те же чтение и поиск дают новое, и проверить себя модели можно
            seen, version, checked = {}, 0, False
            while True:
                step += 1
                last = step > MAX_STEPS
                if last:
                    messages.append({"role": "user", "content": FINISH})
                yield {"type": "thinking", "step": step}
                async for event in turn(messages, None if last else specs):
                    if event["type"] == "text":
                        yield event
                    else:
                        reply = event
                messages.append({"role": "assistant", "content": reply["text"],
                                 **({"tool_calls": reply["calls"]} if reply["calls"] else {})})
                if not reply["calls"]:
                    edited = any(r["ok"] and r["tool"] == "edit_file" for r in seen.values())
                    if edited and not checked and not last:
                        checked = True
                        messages.append({"role": "user", "content": CHECK})
                        yield {"type": "check", "step": step, "text": reply["text"], "llm": reply["llm"]}
                        continue
                    yield {"type": "report", "step": step, "text": reply["text"], "llm": reply["llm"]}
                    break

                done = []
                for call in reply["calls"]:
                    name, args = call["function"]["name"], call["function"].get("arguments") or {}
                    key = json.dumps([name, args, version], sort_keys=True, ensure_ascii=False)
                    if name not in owners:
                        done.append({"tool": name, "args": args, "ok": False, "result": f"Инструмента {name} нет", "ms": 0})
                    elif key in seen:
                        before = seen[key]
                        done.append({"tool": name, "args": args, "ok": False, "ms": 0, "result":
                                     "Этот вызов уже был, его результат выше. Сделай другой шаг или закончи." if before["ok"]
                                     else f"Этот вызов уже был и не прошёл: {before['result']} Исправь аргументы или закончи."})
                    else:
                        await assistant.call(servers[owners[name]], done, name, args)
                        done[-1]["server"] = owners[name]
                        seen[key] = done[-1]
                        version += done[-1]["ok"] and name in ("edit_file", "write_file")
                    messages.append({"role": "tool", "tool_name": name, "content": for_model(done[-1]["result"])})
                yield {"type": "step", "step": step, "thought": reply["text"], "calls": done, "llm": reply["llm"]}

            draft = (await files.call_tool("draft_diff", {})).structured_content
            yield {"type": "draft", **draft}
        yield {"type": "done", "steps": step, "seconds": round(time.perf_counter() - started, 1)}
    except llm.LLMError as exc:
        yield {"type": "error", "step": step, "message": str(exc)}
    except Exception as exc:  # MCP, git, файлы — словами, а не падением
        yield {"type": "error", "step": step, "message": assistant.reason(exc)}


async def apply():
    """Записать черновик в проект — по кнопке пользователя; git status после записи показывает, что файлы
    правда изменились."""
    calls = []
    async with Client(config.FILES_MCP_URL, read_timeout_seconds=60) as files:
        written = await assistant.call(files, calls, "apply_draft", {})
    async with Client(config.MCP_URL, read_timeout_seconds=60) as git:
        status = await assistant.call(git, calls, "git_status", {})
    return {"written": written, "status": status, "calls": calls}


async def discard():
    async with Client(config.FILES_MCP_URL, read_timeout_seconds=60) as files:
        return (await files.call_tool("discard_draft", {})).structured_content
