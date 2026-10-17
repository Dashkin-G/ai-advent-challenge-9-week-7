"""MCP-сервер git: ветка, изменения, файлы и diff проекта, история, изменения pull request. Только чтение."""
import subprocess
from typing import Annotated, Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field

from app import config

server = MCPServer(
    name="git",
    version="1.2.0",
    instructions="Git-репозиторий проекта, о котором отвечает ассистент: текущая ветка, незакоммиченные "
                 "изменения, список файлов, diff, история коммитов, файлы любого коммита и изменения pull request "
                 "между двумя ветками. Инструменты только читают, ничего не меняют.",
)

STATUS = {"M": "изменён", "A": "добавлен", "D": "удалён", "R": "переименован", "?": "новый", "U": "конфликт"}
Ref = Annotated[str, Field(description="Ветка или коммит, например main")]
# diff PR разбирается кодом: один формат при любых настройках git пользователя (цвет, префиксы, внешний diff)
PLAIN = ["--no-color", "--no-ext-diff", "--no-renames"]


def git(*args):
    # quotepath=false — пути не по-латински приходят как есть, а не восьмеричными кодами
    done = subprocess.run(["git", "-c", "core.quotepath=false", "-C", str(config.PROJECT_DIR), *args],
                          capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=15)
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
    ref: Annotated[str, Field(description="Ветка или коммит; пусто — рабочая папка с новыми файлами")] = "",
) -> dict[str, Any]:
    """Файлы проекта — отслеживаемые git и новые, без игнорируемых (.venv, data, models); или файлы коммита."""
    if ref:
        found = git("ls-tree", "-r", "--name-only", ref, "--", path or ".").splitlines()
    else:
        found = git("ls-files", "--cached", "--others", "--exclude-standard", "--", path or ".").splitlines()
    return {"path": path or ".", "ref": ref, "count": len(found), "files": found}


@server.tool()
def read_file(
    path: Annotated[str, Field(description="Файл внутри проекта, например src/ml/model.py")],
    ref: Ref = "HEAD",
) -> dict[str, Any]:
    """Текст файла в указанном коммите или ветке."""
    return {"path": path, "ref": ref, "text": git("show", f"{ref}:{path}")}


@server.tool()
def git_diff(
    path: Annotated[str, Field(description="Файл внутри проекта; пусто — все изменения")] = "",
    max_lines: Annotated[int, Field(ge=10, le=400, description="Сколько строк diff вернуть")] = 80,
) -> dict[str, Any]:
    """Незакоммиченные правки отслеживаемых файлов (git diff HEAD): что именно поменялось."""
    lines = git("diff", "HEAD", "--", path or ".").splitlines()
    return {"path": path or ".", "lines": len(lines), "truncated": len(lines) > max_lines,
            "diff": "\n".join(lines[:max_lines])}


@server.tool()
def git_log(
    n: Annotated[int, Field(ge=1, le=50, description="Сколько последних коммитов")] = 20,
    path: Annotated[str, Field(description="Только коммиты, которые меняли этот файл или папку; пусто — все")] = "",
    oldest_first: Annotated[bool, Field(description="Сначала старые коммиты — история по порядку")] = False,
) -> dict[str, Any]:
    """История текущей ветки без merge-коммитов, новые сверху: дата, хеш, сообщение, изменённые файлы и
    сколько строк добавлено и удалено."""
    commits = []
    out = git("log", f"-{n}", "--no-merges", *(["--reverse"] if oldest_first else []), *PLAIN, "--numstat",
              "--date=format:%d.%m.%Y", "--format=%x1e%h%x1f%ad%x1f%s", "--", path or ".")
    for record in out.split("\x1e")[1:]:
        head, *lines = record.strip("\n").split("\n")
        sha, date, subject = head.split("\x1f")
        stats = [line.split("\t", 2) for line in lines if line]
        commits.append({"date": date, "hash": sha, "subject": subject, "files": [file for _, _, file in stats],
                        # «-» вместо числа — двоичный файл
                        "added": sum(int(a) for a, _, _ in stats if a != "-"),
                        "deleted": sum(int(d) for _, d, _ in stats if d != "-")})
    return {"count": len(commits), "commits": commits}


@server.tool()
def pr_files(base: Ref, head: Ref) -> dict[str, Any]:
    """Изменённые файлы pull request — что сделано в head с момента ответвления от base: статус файла и
    сколько строк добавлено и удалено. Переименование показывается как удаление и добавление."""
    changes = f"{base}...{head}"
    counts = {}
    for line in git("diff", "--numstat", *PLAIN, changes).splitlines():
        added, deleted, path = line.split("\t", 2)
        counts[path] = (int(added), int(deleted)) if added != "-" else (0, 0)  # «-» — двоичный файл
    files = []
    for line in git("diff", "--name-status", *PLAIN, changes).splitlines():
        code, path = line.split("\t", 1)
        added, deleted = counts.get(path, (0, 0))
        files.append({"path": path, "status": STATUS.get(code[:1], code), "added": added, "deleted": deleted})
    return {
        "base": base, "head": head,
        "merge_base": git("merge-base", base, head).strip(),
        "head_sha": git("rev-parse", head).strip(),
        "files": files,
        "added": sum(f["added"] for f in files),
        "deleted": sum(f["deleted"] for f in files),
    }


@server.tool()
def pr_diff(
    base: Ref,
    head: Ref,
    path: Annotated[str, Field(description="Файл внутри проекта; пусто — все файлы")] = "",
    max_lines: Annotated[int, Field(ge=10, le=5000, description="Сколько строк diff вернуть")] = 2000,
) -> dict[str, Any]:
    """Diff pull request (git diff base...head): изменения head с момента ответвления от base,
    по 5 строк контекста вокруг правок."""
    lines = git("diff", "-U5", *PLAIN, "--src-prefix=a/", "--dst-prefix=b/", f"{base}...{head}", "--",
                path or ".").splitlines()
    return {"path": path or ".", "lines": len(lines), "truncated": len(lines) > max_lines,
            "diff": "\n".join(lines[:max_lines])}
