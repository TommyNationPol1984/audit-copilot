import os
from pathlib import Path
from redis import Redis, ConnectionPool
from dotenv import load_dotenv
import structlog
import sys

load_dotenv()

logger = structlog.get_logger(__name__)

# Configuration values
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
UPLOAD_TEMP_DIR = Path(os.getenv("UPLOAD_TEMP_DIR", "/tmp/audit_copilot_uploads"))
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "300"))
GEMINI_DEFAULT_MODEL = os.getenv("GEMINI_DEFAULT_MODEL", "gemini-2.5-flash")
AUDIT_CACHE_TTL = int(os.getenv("AUDIT_CACHE_TTL_HOURS", "168")) * 3600
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Performance tuning
REDIS_MAX_CONNECTIONS = int(os.getenv("REDIS_MAX_CONNECTIONS", "50"))
WORKER_TIMEOUT_SECONDS = int(os.getenv("WORKER_TIMEOUT_SECONDS", "1800"))

# Create necessary directories
UPLOAD_TEMP_DIR.mkdir(parents=True, exist_ok=True)
Path("/tmp/audit_copilot_decks").mkdir(parents=True, exist_ok=True)

# Validate critical environment variables
def validate_environment():
    """Validate that all required environment variables are set."""
    required_vars = ["GEMINI_API_KEY"]
    missing = [var for var in required_vars if not os.getenv(var)]
    
    if missing:
        logger.error("missing_env_vars", vars=missing)
        if ENVIRONMENT == "production":
            sys.exit(1)
        else:
            logger.warning("missing_env_vars_in_development", will_fail_at_runtime=True)

validate_environment()

# Redis connection pool with optimizations
try:
    redis_pool = ConnectionPool.from_url(
        REDIS_URL,
        max_connections=REDIS_MAX_CONNECTIONS,
        socket_keepalive=True,
        socket_keepalive_options={
            1: 3,  # TCP_KEEPIDLE
            2: 3,  # TCP_KEEPINTVL
            3: 3,  # TCP_KEEPCNT
        },
        retry_on_timeout=True,
        decode_responses=True,
        health_check_interval=30  # Check connection health every 30 seconds
    )
    redis_client: Redis = Redis(connection_pool=redis_pool, decode_responses=True)
    
    # Verify Redis connectivity
    redis_client.ping()
    logger.info("redis_connected", max_connections=REDIS_MAX_CONNECTIONS)
except Exception as e:
    logger.error("redis_connection_failed", error=str(e))
    if ENVIRONMENT == "production":
        sys.exit(1)
    else:
        logger.warning("continuing_without_redis", will_fail_at_runtime=True)

# Job queue initialization
from rq import Queue, Scheduler

default_queue = Queue("audit_default", connection=redis_client, default_timeout=WORKER_TIMEOUT_SECONDS)
high_priority_queue = Queue("audit_high", connection=redis_client, default_timeout=WORKER_TIMEOUT_SECONDS // 2)
dead_letter_queue = Queue("audit_dead_letter", connection=redis_client)

# Scheduler for recurring jobs
scheduler = Scheduler(connection=redis_client)

logger.info(
    "config_loaded",
    environment=ENVIRONMENT,
    redis_url=REDIS_URL.split('@')[-1],  # Hide credentials in logs
    upload_dir=str(UPLOAD_TEMP_DIR),
    max_upload_size_mb=MAX_UPLOAD_SIZE_MB,
    worker_timeout_seconds=WORKER_TIMEOUT_SECONDS
)

