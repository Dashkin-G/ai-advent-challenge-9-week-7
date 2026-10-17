"""Настройки сводки: издания, расписание, модель, лимиты. Секреты — из окружения; локально — из .env в корне
репозитория, на сервере их задаёт служба (EnvironmentFile)."""
import os
from datetime import time, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).parent

_env = ROOT.parent / ".env"
if _env.exists():
    for line in _env.read_text(encoding="utf-8").splitlines():
        name, sep, value = line.partition("=")
        if sep and not name.lstrip().startswith("#"):
            os.environ.setdefault(name.strip(), value.strip().strip("\"'"))

DATA_DIR = Path(os.getenv("DATA_DIR", ROOT.parent / "data" / "news"))
DB_PATH = DATA_DIR / "news.db"

# RSS-ленты изданий — проверены 17.10.2026 напрямую, без прокси. ТАСС, РИА, RT и МК держат в ленте
# только 2–4 часа новостей, поэтому сборщик обходит ленты часто и копит новости в базе. Известий нет:
# их DDoS-Guard не пускает адреса дата-центров — с сервера даже главная страница отвечает 403.
FEEDS = {
    "ТАСС": "https://tass.ru/rss/v2.xml",
    "РИА Новости": "https://ria.ru/export/rss2/archive/index.xml",
    "Интерфакс": "https://www.interfax.ru/rss",
    "Коммерсантъ": "https://www.kommersant.ru/rss/news.xml",
    "РБК": "https://rssexport.rbc.ru/rbcnews/news/30/full.rss",
    "Ведомости": "https://www.vedomosti.ru/rss/news.xml",
    "Forbes": "https://www.forbes.ru/newrss.xml",
    "BFM": "https://www.bfm.ru/news.rss",
    "Лента.ру": "https://lenta.ru/rss/news",
    "Российская газета": "https://rg.ru/xml/index.xml",
    "RT": "https://russiatoday.ru/rss",
    "Газета.ру": "https://www.gazeta.ru/export/rss/first.xml",
    "Взгляд": "https://vz.ru/rss.xml",
    "МК": "https://www.mk.ru/rss/index.xml",
    "КП": "https://www.kp.ru/rss/allsections.xml",
    "АиФ": "https://aif.ru/rss/all.php",
    "Москва 24": "https://www.m24.ru/rss.xml",
    "Регнум": "https://regnum.ru/rss",
    "Парламентская газета": "https://www.pnp.ru/rss/index.xml",
    "РАПСИ": "https://rapsinews.ru/export/rss2/index.xml",
    "Право.ру": "https://pravo.ru/rss/",
    "Кремль": "http://kremlin.ru/events/all/feed",
    "Правительство": "http://government.ru/all/rss/",
    "Совет Федерации": "http://council.gov.ru/events/news/rss/",
}
COLLECT_EVERY = int(os.getenv("COLLECT_EVERY", "900"))  # секунд между обходами лент
KEEP_DAYS = 14  # новости старше удаляются; готовые сводки хранятся всегда

MSK = timezone(timedelta(hours=3), "МСК")  # без перехода на летнее время с 2014 года
DIGEST_AT = time(20, 0)  # по Москве: сводка за последние сутки по каждой теме
WINDOW = timedelta(hours=24)

# Модель: Alibaba Model Studio (DashScope), OpenAI-совместимый режим
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
DASHSCOPE_BASE_URL = os.getenv(
    "DASHSCOPE_BASE_URL", "https://ws-q1vxaj37wm4fa9q8.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3.7-plus")

# Лимиты: каждая сводка — платные вызовы модели
CANDIDATES = 80  # кандидатов на проверку моделью — самые совпадающие со словами темы
TOPICS_PER_USER = 3
MAX_TOPICS = int(os.getenv("MAX_TOPICS", "50"))  # на весь сервис
MANUAL_EVERY = timedelta(minutes=int(os.getenv("MANUAL_EVERY", "10")))  # «Собрать сейчас» по теме не чаще

HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8002"))  # 8000 — ML-проект, 8001 — ассистент разработчика
COOKIE_SECURE = os.getenv("COOKIE_SECURE") == "1"  # на сервере сайт открыт только по HTTPS
