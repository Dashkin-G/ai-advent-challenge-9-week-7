"""Ассистент разработчика и поддержки: страница /help, место оператора поддержки /support, работа с файлами
проекта /files, MCP-серверы git на /mcp/git, CRM на /mcp/crm и файлов на /mcp/files."""
import asyncio
import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from app import assistant, config, crm_mcp, files, files_mcp, git_mcp, llm, rag, support

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PAGE = config.ROOT / "static" / "index.html"
SUPPORT_PAGE = config.ROOT / "static" / "support.html"
FILES_PAGE = config.ROOT / "static" / "files.html"


async def warm_up():
    """Векторы документации и базы знаний, модель — при старте, чтобы первый вопрос не ждал."""
    try:
        logger.info("Индекс документации: %s", await rag.ensure_index())
        logger.info("База знаний поддержки: %s", await support.ensure_index())
        await llm.load()
        logger.info("Модель %s загружена в память", config.MODEL)
    except Exception as exc:
        logger.warning("Подготовка не завершена: %s", exc)


@asynccontextmanager
async def lifespan(app):
    # у смонтированных приложений MCP свой lifespan не запускается — менеджеры сессий поднимаем здесь
    async with git_mcp.server.session_manager.run(), crm_mcp.server.session_manager.run(), \
            files_mcp.server.session_manager.run():
        task = asyncio.create_task(warm_up())
        yield
        task.cancel()


app = FastAPI(title="Ассистент разработчика и поддержки", lifespan=lifespan)
app.mount("/mcp/git", git_mcp.server.streamable_http_app(streamable_http_path="/"))
app.mount("/mcp/crm", crm_mcp.server.streamable_http_app(streamable_http_path="/"))
app.mount("/mcp/files", files_mcp.server.streamable_http_app(streamable_http_path="/"))


class Command(BaseModel):
    text: str


class Question(BaseModel):
    question: str
    ticket_id: str | None = None


class Task(BaseModel):
    goal: str


@app.get("/", include_in_schema=False)
async def page():
    return FileResponse(PAGE)


@app.get("/support", include_in_schema=False)
async def support_page():
    return FileResponse(SUPPORT_PAGE)


@app.get("/files", include_in_schema=False)
async def files_page():
    return FileResponse(FILES_PAGE)


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


@app.get("/api/support/status")
async def support_status():
    ollama, crm = await asyncio.gather(llm.status(), support.queue())
    return {"ollama": ollama, "model": config.MODEL, "embed_model": config.EMBED_MODEL,
            "kb": support.index.info(), "crm": crm}


@app.post("/api/support/answer")
async def support_answer(body: Question):
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Напишите вопрос клиента")

    async def stream():
        async for event in support.answer(body.ticket_id, question):
            yield assistant.dumps(event)

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.post("/api/files/run")
async def files_run(task: Task):
    goal = task.goal.strip()
    if not goal:
        raise HTTPException(status_code=400, detail="Напишите задачу: что найти, обновить или создать")

    async def stream():
        async for event in files.run(goal):
            yield assistant.dumps(event)

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.post("/api/files/apply")
async def files_apply():
    return await files.apply()


@app.post("/api/files/discard")
async def files_discard():
    return await files.discard()


if __name__ == "__main__":
    uvicorn.run(app, host=config.HOST, port=config.PORT)
