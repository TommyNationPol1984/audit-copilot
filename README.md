# Audit Copilot MEGA-ZORD Backend v4.2

Production-ready backend with strict Grading "Design Instructions" compliance.

## Features

- **Strict bullet-by-bullet rationale generation** (exactly 3 sentences per bullet)
- **Automatic font size detection** + WCAG AAA contrast analysis
- **Quantitative design metrics** per slide (margins, alignment, whitespace, contrast)
- **Deck-level batch analysis** with overall scoring
- **Movable / reorderable slides** support
- **Redis autosave & draft system**
- Docker-ready deployment

## Quick Start

See [QUICKSTART.md](./QUICKSTART.md) for the fastest way to get running.

## Environment Setup

1. Copy `.env.example` → `.env`
2. Add your `GEMINI_API_KEY`
3. Run `docker compose up --build -d`

## GitHub Upload Note

This folder is ready to be uploaded to a new GitHub repository.  
Recommended files to commit:
- Everything except `.env` (use `.env.example` instead)
- The `.gitignore` is already included

## Documentation

- `QUICKSTART.md` — Fast deployment guide
- `EXECUTED_FEATURES_v4.2.md` — Full feature list
- `audit_logic.py` — Contains the strict Grading-compliant rationale engine
