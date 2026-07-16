# Audit Copilot MEGA-ZORD v4.2 - Quick Start Guide

This guide will help you deploy the Audit Copilot backend quickly.

## 1. Prerequisites

- Docker & Docker Compose installed
- Gemini API Key (from Google AI Studio)

## 2. Quick Deploy (Recommended)

```bash
# 1. Unzip the package
unzip audit-copilot-backend-full.zip
cd audit-copilot-backend

# 2. Create environment file
cp .env.example .env

# 3. Edit .env and add your Gemini API key
nano .env   # or use any text editor

# 4. Start everything
docker compose up --build -d

# 5. Check if it's running
docker compose ps
```

The API will be available at: `http://localhost:8000`

## 3. Verify Installation

```bash
# Health check
curl http://localhost:8000/health
```

You should see a JSON response with `"status": "ok"`.

## 4. Key Features Available

- Strict bullet-by-bullet rationale generation (Grading style)
- Automatic font size detection + WCAG AAA contrast
- Quantitative design metrics per slide
- Deck-level batch analysis
- Movable / reorderable slides support
- Redis autosave & draft system

## 5. GitHub Actions CI/CD

A basic CI workflow is included at `.github/workflows/ci.yml`.

It automatically:
- Checks Python syntax on all Python files
- Builds the Docker image
- Tests that the container starts and responds to `/health`

This runs on every push and pull request to `main` or `master`.

## 6. Updating

```bash
docker compose up --build -d
```

---

**Note**: This system was built to meet strict Grading "Design Instructions" evaluation standards (bullet-by-bullet, 3 sentences per bullet, quantitative + qualitative combined).