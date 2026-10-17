"""Настройки ассистента: какой проект он объясняет, где Ollama и какие модели."""
import os
from pathlib import Path

ROOT = Path(__file__).parent.parent

# Проект, о котором отвечает ассистент: его документация и git
PROJECT_DIR = Path(os.getenv("PROJECT_DIR", ROOT.parent / "ml-student-dropout")).resolve()
DOCS = ["README.md", "docs/*.md"]
CODE = ["*.py"]  # код проекта — в RAG ревью вместе с документацией

# Поддержка: база знаний и CRM — JSON с клиентами и их обращениями
FAQ = ROOT / "support" / "faq.md"
CRM_FILE = Path(os.getenv("CRM_FILE", ROOT / "support" / "crm.json"))

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
MODEL = os.getenv("MODEL", "qwen3.5:4b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")
TOP = int(os.getenv("TOP", "3"))  # фрагментов документации в промпт

HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8001"))  # 8000 обычно занят самим ML-проектом
MCP_URL = f"http://{HOST}:{PORT}/mcp/git/"
CRM_MCP_URL = f"http://{HOST}:{PORT}/mcp/crm/"
FILES_MCP_URL = f"http://{HOST}:{PORT}/mcp/files/"
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))

# Запросы к себе (MCP) и к Ollama — мимо системного прокси
os.environ["NO_PROXY"] = ",".join(filter(None, [os.getenv("NO_PROXY"), "127.0.0.1", "localhost"]))
