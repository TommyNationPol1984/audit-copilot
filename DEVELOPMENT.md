# Audit Copilot Development Guide

## Overview

Audit Copilot v5.0 is a production-grade FastAPI application for AI-powered design analysis. This guide covers local development setup, contributing, and deployment.

## Quick Start with Docker Compose

The easiest way to get started is using Docker Compose, which runs the entire stack locally:

```bash
# 1. Clone the repository
git clone <repo-url>
cd audit-copilot

# 2. Copy environment template
cp .env.example .env

# 3. Add your API keys to .env
# Edit .env and add:
#   - GEMINI_API_KEY
#   - ANTHROPIC_API_KEY

# 4. Start the stack
docker-compose up -d

# 5. Verify everything is running
curl http://localhost:8000/health
```

### Docker Compose Services

- **app**: Main FastAPI application on port 8000
- **postgres**: PostgreSQL database on port 5432
- **redis**: Redis cache & job queue on port 6379
- **worker**: RQ worker for async job processing
- **redis-commander**: Redis UI for debugging on port 8081

## Local Development (without Docker)

### Prerequisites

- Python 3.11+
- PostgreSQL 13+
- Redis 6+

### Setup

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up environment
cp .env.example .env
# Edit .env with your database and API credentials

# 4. Start services (in separate terminals)
# Terminal 1: Main app
python main.py

# Terminal 2: Worker process
rq worker audit_default audit_high --with-scheduler

# Terminal 3: Optional - Redis commander
redis-commander
```

## API Endpoints

### Health & Monitoring

- `GET /health` - Health check with dependency status
- `GET /metrics` - Prometheus metrics (for monitoring systems)
- `GET /docs` - Interactive API documentation (Swagger UI)
- `GET /redoc` - ReDoc API documentation

### Analysis Endpoints

- `POST /analyze/slide` - Analyze a single slide image
  - Parameters: `image_path` (string)
  - Returns: Design metrics and recommendations

- `POST /analyze/deck` - Queue entire PDF for analysis
  - Parameters: `pdf_path`, `guidelines`, `priority` (high/default)
  - Returns: Job ID for tracking

- `GET /analyze/status/{job_id}` - Get job status and results
  - Parameters: `job_id` (from /analyze/deck response)
  - Returns: Job status with progress and results

- `POST /analyze/batch` - Queue multiple PDFs
  - Parameters: `pdf_paths` (list), `guidelines` (string)
  - Returns: Batch job IDs

## Project Structure

```
audit-copilot/
├── main.py                  # FastAPI application
├── worker.py                # RQ job worker
├── config.py                # Configuration and connections
├── audit_logic.py           # Design audit logic
├── deck_analyzer.py         # PDF analysis
├── design_metrics.py        # Quantitative metrics
├── monitoring.py            # Prometheus metrics & middleware
├── cache.py                 # Redis caching utilities
├── error_handling.py        # Resilience patterns (circuit breaker, retry)
├── docker-compose.yml       # Local development stack
├── Dockerfile               # Container configuration
├── requirements.txt         # Python dependencies
├── .env.example             # Environment template
└── DEVELOPMENT.md          # This file
```

## Key Features

### 1. Performance
- **Response caching** with Redis
- **Request deduplication** for async jobs
- **Gzip compression** for API responses
- **Connection pooling** for database and Redis

### 2. Observability
- **Prometheus metrics** for all endpoints and jobs
- **Structured logging** with request IDs
- **Request tracing** across services
- **Job lifecycle tracking**

### 3. Reliability
- **Circuit breaker** pattern for API failures
- **Exponential backoff** retry logic
- **Graceful shutdown** for in-flight requests
- **Health checks** on all dependencies
- **Timeout management** for long operations

### 4. Scalability
- **Horizontal scaling** with RQ workers
- **Priority queue support** (high/default)
- **Multi-region ready** infrastructure
- **Job batching** for bulk analysis

### 5. Developer Experience
- **Docker Compose** for instant local setup
- **Auto-reload** on file changes
- **Comprehensive API docs** with Swagger UI
- **Request/response examples**
- **Environment validation**

## Testing

```bash
# Run health check
curl http://localhost:8000/health | jq

# Check metrics
curl http://localhost:8000/metrics | head -20

# Submit an analysis job
curl -X POST http://localhost:8000/analyze/deck \
  -H "Content-Type: application/json" \
  -d '{
    "pdf_path": "/tmp/example.pdf",
    "guidelines": "Use sans-serif fonts",
    "priority": "high"
  }'

# Check job status
curl http://localhost:8000/analyze/status/{job_id}
```

## Monitoring

### Metrics Export
Prometheus metrics are available at `/metrics`. Configure your monitoring system:

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'audit-copilot'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
```

### Log Aggregation
Logs use structured logging (structlog). Integrate with:
- ELK Stack (Elasticsearch, Logstash, Kibana)
- Splunk
- CloudWatch
- Datadog

Example log output:
```json
{
  "event": "request_completed",
  "request_id": "abc123",
  "method": "POST",
  "path": "/analyze/deck",
  "status": 200,
  "duration": 0.15
}
```

## Deployment

### Railway Deployment
```bash
# Automatic deployment on push to main
git push origin main
```

Configuration is in `railway.json` or via Railway dashboard.

### Environment Variables
Set these on the deployment platform:
- `GEMINI_API_KEY` - API key for Gemini
- `ANTHROPIC_API_KEY` - API key for Anthropic
- `REDIS_URL` - Redis connection string
- `DATABASE_URL` - PostgreSQL connection string

### Horizontal Scaling
To run multiple workers:

```bash
# Railway: Set numReplicas in deploy config
# Docker: Scale worker service
docker-compose up -d --scale worker=3
```

## Troubleshooting

### App won't start
```bash
# Check health endpoint
curl -v http://localhost:8000/health

# Check logs
docker-compose logs app

# Verify database connectivity
docker-compose exec postgres psql -U postgres -d audit_copilot -c "SELECT 1"

# Verify Redis connectivity
docker-compose exec redis redis-cli ping
```

### Jobs not processing
```bash
# Check worker logs
docker-compose logs worker

# Check queue depth
docker-compose exec redis redis-cli LLEN rq:queue:audit_default

# Connect to Redis Commander
# Visit http://localhost:8081
```

### High memory/CPU
```bash
# Check metrics
curl http://localhost:8000/metrics | grep -E "process_|http_requests"

# Check active connections
docker-compose exec redis redis-cli INFO stats | grep connected_clients
```

## Contributing

1. Create a feature branch
2. Make your changes
3. Add tests if applicable
4. Submit a pull request

## Support

For issues or questions:
1. Check `/health` and `/metrics` endpoints
2. Review logs: `docker-compose logs -f app`
3. Open an issue with:
   - Error message
   - Reproduction steps
   - Relevant logs
   - Environment details

