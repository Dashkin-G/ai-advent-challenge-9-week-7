"""MCP-сервер git: ветка, изменения, файлы и diff проекта. Только чтение."""
import subprocess
from typing import Annotated, Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field

from app import config

server = MCPServer(
    name="git",
    version="1.0.0",
    instructions="Git-репозиторий проекта, о котором отвечает ассистент: текущая ветка, незакоммиченные "
                 "изменения, список файлов и diff. Инструменты только читают, ничего не меняют.",
)

STATUS = {"M": "изменён", "A": "добавлен", "D": "удалён", "R": "переименован", "?": "новый", "U": "конфликт"}


def git(*args):
    done = subprocess.run(["git", "-C", str(config.PROJECT_DIR), *args],
                          capture_output=True, text=True, encoding="utf-8", timeout=15)
    if done.returncode != 0:
        # ToolError доходит до клиента своим текстом; любое другое исключение SDK прячет за общей фразой
        raise ToolError(done.stderr.strip() or f"git {' '.join(args)} завершился с кодом {done.returncode}")
    return done.stdout


@server.tool()
def git_branch() -> dict[str, Any]:
    """Текущая ветка, все локальные ветки и последний коммит."""
    return {
        "branch": git("branch", "--show-current").strip() or "(detached HEAD)",
        "branches": git("branch", "--format=%(refname:short)").split(),
        # точная дата, а не «5 минут назад»: начало промпта не меняется, и Ollama берёт его из кэша
        "last_commit": git("log", "-1", "--format=%h %s · %cd", "--date=format:%d.%m.%Y %H:%M").strip(),
    }


@server.tool()
def git_status() -> dict[str, Any]:
    """Незакоммиченные изменения: какие файлы изменены, добавлены, удалены или новые."""
    changed = []
    for line in git("status", "--porcelain=v1", "--untracked-files=all").splitlines():
        code = line[:2].strip()
        changed.append({"path": line[3:].strip('"'), "status": STATUS.get(code[:1], code)})
    return {"clean": not changed, "count": len(changed), "changed": changed}


@server.tool()
def list_files(
    path: Annotated[str, Field(description="Папка внутри проекта, например src/ml; пусто — весь проект")] = "",
) -> dict[str, Any]:
    """Файлы проекта — отслеживаемые git и новые, без игнорируемых (.venv, data, models)."""
    found = git("ls-files", "--cached", "--others", "--exclude-standard", "--", path or ".").splitlines()
    return {"path": path or ".", "count": len(found), "files": found}


@server.tool()
def git_diff(
    path: Annotated[str, Field(description="Файл внутри проекта; пусто — все изменения")] = "",
    max_lines: Annotated[int, Field(ge=10, le=400, description="Сколько строк diff вернуть")] = 80,
) -> dict[str, Any]:
    """Незакоммиченные правки отслеживаемых файлов (git diff HEAD): что именно поменялось."""
    lines = git("diff", "HEAD", "--", path or ".").splitlines()
    return {"path": path or ".", "lines": len(lines), "truncated": len(lines) > max_lines,
            "diff": "\n".join(lines[:max_lines])}
