"""Ассистент разработчика: страница, команда /help потоком шагов, MCP-сервер git на /mcp/git."""
import asyncio
import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from app import assistant, config, git_mcp, llm, rag

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PAGE = config.ROOT / "static" / "index.html"
mcp_app = git_mcp.server.streamable_http_app(streamable_http_path="/")


async def warm_up():
    """Векторы документации и модель — при старте, чтобы первый /help не ждал."""
    try:
        logger.info("Индекс документации: %s", await rag.ensure_index())
        await llm.load()
        logger.info("Модель %s загружена в память", config.MODEL)
    except Exception as exc:
        logger.warning("Подготовка не завершена: %s", exc)


@asynccontextmanager
async def lifespan(app):
    # у смонтированного приложения MCP свой lifespan не запускается — менеджер сессий поднимаем здесь
    async with git_mcp.server.session_manager.run():
        task = asyncio.create_task(warm_up())
        yield
        task.cancel()


app = FastAPI(title="Ассистент разработчика", lifespan=lifespan)
app.mount("/mcp/git", mcp_app)


class Command(BaseModel):
    text: str


@app.get("/", include_in_schema=False)
async def page():
    return FileResponse(PAGE)


@app.get("/api/status")
async def status():
    ollama, project = await asyncio.gather(llm.status(), assistant.project())
    return {"project": {**project, "path": str(config.PROJECT_DIR)}, "ollama": ollama,
            "model": config.MODEL, "embed_model": config.EMBED_MODEL, "docs": rag.info()}


@app.post("/api/help")
async def help_command(command: Command):
    question = assistant.parse(command.text)
    if question is None:
        raise HTTPException(status_code=400, detail="Команда ассистента: /help <вопрос о проекте>")

    async def stream():
        async for event in assistant.answer(question):
            yield assistant.dumps(event)

    return StreamingResponse(stream(), media_type="application/x-ndjson")


if __name__ == "__main__":
    uvicorn.run(app, host=config.HOST, port=config.PORT)
