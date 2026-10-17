"""Ollama по HTTP: векторы для поиска (bge-m3) и ответ потоком (qwen3.5:4b)."""
import json

import httpx2

from app import config


class LLMError(Exception):
    """Ошибка по-человечески: что случилось и что сделать."""


# Под ответ по фрагментам: факты, а не творчество (подобрано на неделе 6, день 29)
OPTIONS = {"temperature": 0.1, "top_p": 0.8, "presence_penalty": 0, "num_predict": 512, "num_ctx": 4096}
NOT_RUNNING = "Ollama не запущен — откройте его из меню «Пуск»"


def _client(read=None):
    return httpx2.AsyncClient(base_url=config.OLLAMA_URL, trust_env=False, timeout=httpx2.Timeout(10, read=read))


def _missing(model):
    return LLMError(f"Модели {model} нет — скачайте: ollama pull {model}")


async def status():
    """Запущен ли Ollama и скачаны ли обе модели."""
    try:
        async with _client(5) as client:
            version = (await client.get("/api/version")).json()["version"]
            names = {m["name"] for m in (await client.get("/api/tags")).json()["models"]}
    except httpx2.HTTPError:
        return {"running": False}
    return {"running": True, "version": version,
            "models": {model: model in names or f"{model}:latest" in names for model in (config.MODEL, config.EMBED_MODEL)}}


async def load():
    """Загрузить модель в память заранее — иначе первый ответ ждёт ещё 5–8 с."""
    try:
        async with _client(120) as client:
            await client.post("/api/generate", json={"model": config.MODEL})
    except httpx2.ConnectError:
        raise LLMError(NOT_RUNNING)


async def embed(texts):
    """Векторы bge-m3 для списка текстов."""
    try:
        async with _client(600) as client:
            response = await client.post("/api/embed", json={"model": config.EMBED_MODEL, "input": texts})
    except httpx2.ConnectError:
        raise LLMError(NOT_RUNNING)
    if response.status_code == 404:
        raise _missing(config.EMBED_MODEL)
    response.raise_for_status()
    return response.json()["embeddings"]


async def chat(messages, options=None, schema=None, tools=None):
    """Ответ потоком: {"type": "text"} кусочками, вызовы инструментов — {"type": "tool_calls"}, в конце
    {"type": "done"} со счётчиками. options дополняют OPTIONS; schema — JSON Schema, от которой Ollama не даст
    модели отойти; tools — инструменты, которые модель может вызвать сама."""
    body = {"model": config.MODEL, "messages": messages, "stream": True, "think": False,
            "options": {**OPTIONS, **(options or {})}}
    if schema:
        body["format"] = schema
    if tools:
        body["tools"] = tools
    try:
        async with _client() as client, client.stream("POST", "/api/chat", json=body) as response:
            if response.status_code == 404:
                raise _missing(config.MODEL)
            if response.status_code != 200:
                raise LLMError(f"Ollama ответил {response.status_code}: {(await response.aread()).decode()[:200]}")
            async for line in response.aiter_lines():
                if not line:
                    continue
                part = json.loads(line)
                if part.get("error"):
                    raise LLMError(part["error"])
                if text := part.get("message", {}).get("content"):
                    yield {"type": "text", "text": text}
                if calls := part.get("message", {}).get("tool_calls"):
                    yield {"type": "tool_calls", "calls": calls}
                if part.get("done"):
                    yield {"type": "done", **_counters(part)}
    except httpx2.ConnectError:
        raise LLMError(NOT_RUNNING)


def _counters(part):
    sec = lambda key: part.get(key, 0) / 1e9  # Ollama отдаёт наносекунды
    tokens = part.get("eval_count", 0)
    return {
        "tokens": tokens,
        "speed": round(tokens / sec("eval_duration"), 1) if part.get("eval_duration") else 0,
        "prompt": part.get("prompt_eval_count", 0),
        "reading": round(sec("prompt_eval_duration"), 2),
        "writing": round(sec("eval_duration"), 2),
        "load": round(sec("load_duration"), 2),
        "seconds": round(sec("total_duration"), 2),
    }
