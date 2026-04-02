.PHONY: install test test-e2e lint format

install:
	pip install -e ".[all]"

test:
	pytest -v -m "not e2e"

test-e2e:
	pytest -v -m "e2e"

test-all:
	pytest -v

lint:
	ruff check spindle/ tests/

format:
	ruff format spindle/ tests/
