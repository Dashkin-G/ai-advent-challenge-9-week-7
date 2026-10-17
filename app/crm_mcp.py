"""MCP-сервер CRM поддержки: клиенты сервиса и их обращения из JSON-файла. Только чтение."""
import json
from typing import Annotated, Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field

from app import config

server = MCPServer(
    name="crm",
    version="1.0.0",
    instructions="CRM поддержки сервиса «Риск отчисления студентов»: клиенты — сотрудники вузов — и их обращения "
                 "с техническими данными: запрос, ответ сервиса, лог. Инструменты только читают, ничего не меняют.",
)

CLOSED = {"решён"}


def load():
    # файл читается на каждый вызов: правки CRM видны без перезапуска
    return json.loads(config.CRM_FILE.read_text(encoding="utf-8"))


def find(items, item_id, what):
    for item in items:
        if item["id"] == item_id.strip().upper():
            return item
    raise ToolError(f"{what} {item_id} нет в CRM")


@server.tool()
def list_tickets(
    include_closed: Annotated[bool, Field(description="Показать и решённые обращения")] = False,
) -> dict[str, Any]:
    """Очередь обращений, новые сверху: номер, тема, сообщение клиента, статус, приоритет, дата и клиент."""
    crm = load()
    users = {u["id"]: u for u in crm["users"]}
    tickets = [{**{k: t[k] for k in ("id", "subject", "message", "status", "priority", "created")},
                "user": {k: users[t["user_id"]][k] for k in ("id", "name", "org")}}
               for t in crm["tickets"] if include_closed or t["status"] not in CLOSED]
    tickets.sort(key=lambda t: t["created"], reverse=True)
    return {"count": len(tickets), "tickets": tickets}


@server.tool()
def get_ticket(ticket_id: Annotated[str, Field(description="Номер обращения, например T-1041")]) -> dict[str, Any]:
    """Обращение целиком: сообщение клиента, статус, приоритет и технические данные — запрос, ответ сервиса, лог."""
    return find(load()["tickets"], ticket_id, "Обращения")


@server.tool()
def get_user(user_id: Annotated[str, Field(description="Номер клиента, например U-01")]) -> dict[str, Any]:
    """Клиент: вуз, роль, как работает с сервисом, заметки поддержки и все его обращения с решениями."""
    crm = load()
    user = find(crm["users"], user_id, "Клиента")
    tickets = [{k: t[k] for k in ("id", "subject", "status", "created", "resolution") if k in t}
               for t in crm["tickets"] if t["user_id"] == user["id"]]
    return {**user, "tickets": tickets}
