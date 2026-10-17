"""MCP-сервер files: чтение, поиск, правка и создание файлов проекта. Правки копятся в черновике: чтение и
поиск их уже видят, а на диск их записывает apply_draft — после того как человек посмотрел diff."""
import difflib
import re
from typing import Annotated, Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field

from app import config
from app.git_mcp import git

server = MCPServer(
    name="files",
    version="1.0.0",
    instructions="Файлы проекта, с которым работает ассистент: список, чтение с номерами строк, поиск по всем "
                 "файлам, правка и создание. Правки копятся в черновике и видны при чтении и поиске; на диск их "
                 "записывает apply_draft, отменяет discard_draft.",
)

READ_LINES = 80  # строк за один read_file: файл целиком раздувает промпт локальной модели
MATCHES = 60  # совпадений поиска в ответе
NUMBERED = re.compile(r"^\d+\| ?", re.M)  # так read_file нумерует строки
FilePath = Annotated[str, Field(description="Файл внутри проекта, например docs/api.md")]

draft = {}  # путь → текст файла с правками; на диске пока прежний


def _rel(path):
    """Путь относительно проекта; за пределы проекта и в .git — нельзя."""
    full = (config.PROJECT_DIR / path.strip().replace("\\", "/").lstrip("/")).resolve()
    if not full.is_relative_to(config.PROJECT_DIR) or full == config.PROJECT_DIR:
        raise ToolError(f"{path}: путь вне проекта")
    rel = full.relative_to(config.PROJECT_DIR).as_posix()
    if rel.split("/")[0] == ".git":
        raise ToolError(f"{path}: папка .git недоступна")
    return rel


def _disk(rel):
    """Текст файла на диске, переводы строк — \\n; None — файла нет."""
    file = config.PROJECT_DIR / rel
    if not file.is_file():
        return None
    try:
        return file.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raise ToolError(f"{rel}: не текстовый файл")


def _text(rel):
    text = draft[rel] if rel in draft else _disk(rel)
    if text is None:
        raise ToolError(f"{rel}: такого файла нет — посмотрите list_files или search")
    return text


def _files(path=""):
    """Файлы проекта без игнорируемых git и новые файлы черновика."""
    prefix = _rel(path) if path.strip(" ./") else ""  # «.» и «/» — весь проект
    found = set(git("ls-files", "--cached", "--others", "--exclude-standard", "--", prefix or ".").splitlines())
    found |= {rel for rel in draft if not prefix or rel == prefix or rel.startswith(prefix + "/")}
    return sorted(found)


def _plain(text):
    """Модель иногда копирует строки из read_file вместе с номерами — номера убираем."""
    text = text.replace("\r\n", "\n")
    if all(NUMBERED.match(line) for line in text.split("\n") if line):
        text = NUMBERED.sub("", text)
    return text


def _find(text, old):
    """Вхождения old: точные, а если их нет — без учёта пробелов и переводов строк: модель склеивает строки
    абзаца в одну."""
    if text.count(old):
        return [old] * text.count(old)
    return [m.group(0) for m in re.finditer(r"\s+".join(map(re.escape, old.split())), text)]


def _respace(found, new):
    """Переносы строк в замене — там же, где в файле, если слов столько же."""
    gaps, words = re.findall(r"\s+", found), new.split()
    if len(words) != len(gaps) + 1:
        return new.strip()
    return words[0] + "".join(gap + word for gap, word in zip(gaps, words[1:]))


def _diff(rel, old, new, context=3):
    return "\n".join(difflib.unified_diff(old.splitlines() if old is not None else [], new.splitlines(),
                                          "/dev/null" if old is None else f"a/{rel}", f"b/{rel}",
                                          n=context, lineterm=""))


def _counts(diff):
    lines = diff.splitlines()[2:]  # без заголовков --- и +++
    return {"added": sum(line.startswith("+") for line in lines),
            "removed": sum(line.startswith("-") for line in lines)}


@server.tool()
def list_files(
    path: Annotated[str, Field(description="Папка внутри проекта, например docs; пусто — весь проект")] = "",
) -> dict[str, Any]:
    """Файлы проекта: отслеживаемые git и новые, без игнорируемых (.venv, data, models), вместе с файлами,
    созданными в черновике."""
    files = _files(path)
    return {"path": path or ".", "count": len(files), "files": files}


@server.tool()
def read_file(
    path: FilePath,
    start: Annotated[int, Field(ge=1, description="Первая строка")] = 1,
    end: Annotated[int, Field(ge=0, description="Последняя строка; 0 — следующие 80 строк")] = 0,
) -> dict[str, Any]:
    """Текст файла с номерами строк, вместе с правками черновика. За раз — до 80 строк: читайте кусок
    вокруг строк, найденных search."""
    rel = _rel(path)
    lines = _text(rel).splitlines()
    if lines and start > len(lines):
        raise ToolError(f"В {rel} всего {len(lines)} строк")
    end = min(len(lines), end or start + READ_LINES - 1, start + READ_LINES - 1)
    return {"path": rel, "lines": len(lines), "from": start, "to": end, "draft": rel in draft,
            "text": "\n".join(f"{n}| {lines[n - 1]}" for n in range(start, end + 1))}


@server.tool()
def search(
    query: Annotated[str, Field(description="Что искать, без учёта регистра; несколько вариантов — через |")],
    path: Annotated[str, Field(description="Папка или файл; пусто — весь проект")] = "",
) -> dict[str, Any]:
    """Поиск по всем текстовым файлам проекта, вместе с правками черновика: файл, номер строки и сама строка."""
    words = [word.strip().lower() for word in query.split("|") if word.strip()]
    if not words:
        raise ToolError("Пустой запрос")
    matches, files = [], set()
    for rel in _files(path):
        try:
            text = _text(rel)
        except ToolError:  # двоичный файл
            continue
        for n, line in enumerate(text.splitlines(), 1):
            if any(word in line.lower() for word in words):
                matches.append(f"{rel}:{n}: {line.strip()[:200]}")
                files.add(rel)
    return {"query": query, "count": len(matches), "files": len(files), "truncated": len(matches) > MATCHES,
            "matches": matches[:MATCHES]}


@server.tool()
def edit_file(
    path: FilePath,
    old: Annotated[str, Field(description="Точный текст из файла, который заменить; встречается в файле один раз")],
    new: Annotated[str, Field(description="Текст на его место")],
) -> dict[str, Any]:
    """Заменить кусок файла. Правка идёт в черновик, на диск её записывает пользователь."""
    rel = _rel(path)
    text = _text(rel)
    old, new = _plain(old), _plain(new)
    found = _find(text, old) if old.strip() else []
    if not found:
        raise ToolError(f"В {rel} нет такого текста: прочитайте нужные строки read_file и скопируйте их точно, "
                        "без номеров строк")
    if len(found) > 1:
        raise ToolError(f"Этот текст встречается в {rel} {len(found)} раз(а) — добавьте соседние строки, "
                        "чтобы он был один")
    if found[0] != old:
        new = _respace(found[0], new)
    if found[0] == new:
        raise ToolError("new совпадает с old — правка ничего не меняет")
    changed = text.replace(found[0], new, 1)
    draft[rel] = changed
    diff = _diff(rel, text, changed, context=1)
    return {"path": rel, **_counts(diff), "diff": diff}


@server.tool()
def write_file(
    path: Annotated[str, Field(description="Путь нового файла, например CHANGELOG.md")],
    content: Annotated[str, Field(description="Полный текст файла")],
) -> dict[str, Any]:
    """Создать новый файл — в черновике. Существующий файл меняйте через edit_file."""
    rel = _rel(path)
    if rel in draft or (config.PROJECT_DIR / rel).exists():
        raise ToolError(f"{rel} уже есть — меняйте его через edit_file")
    draft[rel] = content.replace("\r\n", "\n").rstrip("\n") + "\n"
    return {"path": rel, "lines": draft[rel].count("\n"), "created": True}


@server.tool()
def draft_diff() -> dict[str, Any]:
    """Все правки черновика: unified diff каждого файла против диска и сколько строк добавлено и удалено."""
    files = []
    for rel in sorted(draft):
        old = _disk(rel)
        if draft[rel] == old:
            continue
        diff = _diff(rel, old, draft[rel])
        files.append({"path": rel, "status": "новый" if old is None else "изменён", **_counts(diff), "diff": diff})
    return {"files": files, "added": sum(f["added"] for f in files), "removed": sum(f["removed"] for f in files)}


@server.tool()
def apply_draft() -> dict[str, Any]:
    """Записать правки черновика в файлы проекта и очистить черновик."""
    written = []
    for rel, text in sorted(draft.items()):
        file = config.PROJECT_DIR / rel
        # переводы строк — как были у файла: с core.autocrlf в рабочей папке бывают и CRLF, и LF
        newline = "\r\n" if file.is_file() and b"\r\n" in file.read_bytes() else "\n"
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(text, encoding="utf-8", newline=newline)
        written.append(rel)
    draft.clear()
    return {"written": written, "count": len(written)}


@server.tool()
def discard_draft() -> dict[str, Any]:
    """Отменить все правки черновика."""
    count = len(draft)
    draft.clear()
    return {"discarded": count}
