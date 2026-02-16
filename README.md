# Trading Factory Starter (MVP)

This is a tiny starter repo meant to be extended by Codex into a full pipeline:
- Trading backend (FastAPI + risk gate)
- Data analysis automation (ETL + features + alerts)
- Strategy engine (signals + backtests)

## Local setup (Windows / WSL / macOS)

```bash
python -m venv .venv
# Windows PowerShell:
#   .venv\Scripts\Activate.ps1
# macOS/WSL:
#   source .venv/bin/activate
pip install -r requirements.txt
```

## Run API

```bash
python -m uvicorn apps.api.main:app --reload --port 8000
```

Then open:
- http://127.0.0.1:8000/health

## Run tests

```bash
python -m pytest -q
```

## Codex workflow (recommended)

1) Push this repo to GitHub.
2) Open `https://chatgpt.com/codex` and connect the repo.
3) Run small tasks (one PR per task) and always require passing tests.

