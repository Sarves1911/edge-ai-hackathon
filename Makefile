.PHONY: setup setup-export test run-example

setup:
	uv sync --dev

setup-export:
	uv sync --dev --extra export

test:
	uv run pytest

run-example:
	uv run edge-detect run --source $(SOURCE)
