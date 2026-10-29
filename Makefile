.PHONY: ruff ruff-fix test test-fast run-backend run-frontend

ruff:
	uv run ruff format
	uv run ruff check --ignore FIX

ruff-fix:
	uv run ruff format
	uv run ruff check --fix

test:
	uv run pytest

test-fast:
	uv run pytest tests/test_standalone_independence.py -v

run-backend:
	uv run python chatbot/backend/main.py

run-frontend:
	cd chatbot/frontend && npm run dev
