"""Модель: Alibaba Model Studio (DashScope), OpenAI-совместимый режим. Каждый шаг сводки просит у модели
JSON-объект; qwen3.8 перед ответом думает всегда — сколько, задаёт бюджет размышлений."""
import asyncio
import json
import time

import httpx2

from news import config


class LLMError(Exception):
    """Ошибка модели по-человечески."""


async def ask(system: str, user: str, budget: int) -> tuple[dict, dict]:
    """JSON-ответ модели и расход: токены вместе с размышлениями и секунды. Сбой сети, 429, 5xx и ответ
    не JSON-объектом повторяются ещё дважды — вечерняя сводка не должна срываться из-за одного сбоя."""
    if not config.DASHSCOPE_API_KEY:
        raise LLMError("нет ключа модели DASHSCOPE_API_KEY")
    body = {"model": config.LLM_MODEL, "response_format": {"type": "json_object"}, "thinking_budget": budget,
            "seed": 7, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    headers = {"Authorization": f"Bearer {config.DASHSCOPE_API_KEY}"}
    error = None
    for attempt in range(3):
        if attempt:
            await asyncio.sleep(3 * attempt)
        start = time.perf_counter()
        try:
            async with httpx2.AsyncClient(timeout=httpx2.Timeout(180, connect=15)) as client:
                response = await client.post(f"{config.DASHSCOPE_BASE_URL}/chat/completions", json=body, headers=headers)
        except httpx2.HTTPError as e:
            error = LLMError(f"модель не отвечает ({type(e).__name__})")
            continue
        if response.status_code != 200:
            error = LLMError(f"модель ответила {response.status_code}: {_detail(response)}")
            if response.status_code == 429 or response.status_code >= 500:
                continue
            raise error
        data = response.json()
        answer = _object(data["choices"][0]["message"].get("content") or "")
        if answer is None:
            error = LLMError("модель ответила не JSON-объектом")
            body.pop("seed", None)  # с тем же seed повтор вернул бы тот же ответ
            continue
        return answer, {"tokens": data.get("usage", {}).get("total_tokens", 0),
                        "seconds": round(time.perf_counter() - start, 1)}
    raise error


def _object(text: str) -> dict | None:
    """JSON-объект из ответа: qwen3.8 иногда оборачивает его в ```json … ```, хотя response_format — json_object."""
    try:
        answer = json.loads(text[text.find("{"):text.rfind("}") + 1])
    except json.JSONDecodeError:
        return None
    return answer if isinstance(answer, dict) else None


def _detail(response) -> str:
    try:
        return response.json()["error"]["message"][:200]
    except Exception:
        return response.text[:200]
