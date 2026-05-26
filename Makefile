PY ?= python3
VENV ?= .venv
BIN := $(VENV)/bin

.PHONY: help venv install lint format typecheck test test-unit test-integration bench bench-regress clean docker-build

help:
	@echo "make venv             create local virtualenv"
	@echo "make install          install package + dev deps into venv"
	@echo "make lint             ruff + black --check"
	@echo "make format           ruff --fix + black"
	@echo "make typecheck        mypy strict"
	@echo "make test             pytest unit (no integration)"
	@echo "make test-integration pytest integration (needs Docker)"
	@echo "make bench            run benchmark"
	@echo "make bench-regress    compare against baseline (30% gate)"

venv:
	$(PY) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip

install: venv
	$(BIN)/pip install -e ".[dev]"

lint:
	$(BIN)/ruff check src tests
	$(BIN)/black --check src tests

format:
	$(BIN)/ruff check --fix src tests
	$(BIN)/black src tests

typecheck:
	$(BIN)/mypy src

test: test-unit

test-unit:
	$(BIN)/pytest tests/unit -v --cov=convoengine --cov-report=term-missing

test-integration:
	$(BIN)/pytest tests/integration -v -m integration

bench:
	$(BIN)/python -m convoengine.bench.run --conversations 100 --turns 10

bench-regress:
	$(BIN)/python -m convoengine.bench.regress --gate 0.30

docker-build:
	docker build -t convoengine:dev .

clean:
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
