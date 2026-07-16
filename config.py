import os
from pathlib import Path
from redis import Redis, ConnectionPool
from dotenv import load_dotenv
import structlog

load_dotenv()

logger = structlog.get_logger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
UPLOAD_TEMP_DIR = Path(os.getenv("UPLOAD_TEMP_DIR", "/tmp/audit_copilot_uploads"))
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "300"))
GEMINI_DEFAULT_MODEL = os.getenv("GEMINI_DEFAULT_MODEL", "gemini-2.5-flash")
AUDIT_CACHE_TTL = int(os.getenv("AUDIT_CACHE_TTL_HOURS", "168")) * 3600

UPLOAD_TEMP_DIR.mkdir(parents=True, exist_ok=True)
Path("/tmp/audit_copilot_decks").mkdir(parents=True, exist_ok=True)

redis_pool = ConnectionPool.from_url(
    REDIS_URL, max_connections=50, socket_keepalive=True
)
redis_client: Redis = Redis(connection_pool=redis_pool, decode_responses=True)

from rq import Queue
default_queue = Queue("audit_default", connection=redis_client, default_timeout=1800)
high_priority_queue = Queue("audit_high", connection=redis_client, default_timeout=900)
dead_letter_queue = Queue("audit_dead_letter", connection=redis_client)