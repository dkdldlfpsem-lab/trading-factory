# AGENTS.md (Codex instructions)

## Non-negotiables
- Keep changes small and reviewable (prefer 1 feature per task).
- Add/adjust tests for every behavior change.
- Before you finish a task, run:
  - `python -m pytest -q`
- If tests fail, fix them. Do not leave the repo broken.

## Repo goals (MVP)
We are building a trading system with three parts:
1) Trading backend (FastAPI): signal ingest, order preview/execute (paper first), risk gate, audit log.
2) Data automation: ingest -> validate -> clean -> features -> signals -> reports.
3) Strategy engine: signal generation + backtests that reuse the same features.

## Style / safety
- Prefer high-confidence, boring code over clever code.
- No live broker integration by default. Implement `PaperBroker` first.
- Any risky action must have a kill-switch and a test.

## Project commands
- Tests: `python -m pytest -q`
- Run API: `python -m uvicorn apps.api.main:app --reload --port 8000`
