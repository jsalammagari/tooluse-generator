PYTHON   := .venv/bin/python
PIP      := .venv/bin/pip
PYTEST   := .venv/bin/pytest
RUFF     := .venv/bin/ruff
BLACK    := .venv/bin/black
MYPY     := .venv/bin/mypy
PRECOMMIT := .venv/bin/pre-commit

SRC      := src/tooluse_gen
TESTS    := tests

.PHONY: all install format lint test test-unit test-integration test-e2e test-cov clean pre-commit-install

# ── Default ───────────────────────────────────────────────────────────────────
all: format lint test

# ── Setup ─────────────────────────────────────────────────────────────────────
install:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

pre-commit-install:
	$(PRECOMMIT) install

# ── Formatting ────────────────────────────────────────────────────────────────
format:
	$(BLACK) $(SRC) $(TESTS)
	$(RUFF) format $(SRC) $(TESTS)
	$(RUFF) check --fix $(SRC) $(TESTS)

# ── Linting ───────────────────────────────────────────────────────────────────
lint:
	$(RUFF) check $(SRC) $(TESTS)
	$(MYPY) $(SRC)

# ── Testing ───────────────────────────────────────────────────────────────────
test:
	$(PYTEST) $(TESTS)

test-unit:
	$(PYTEST) $(TESTS)/unit -m unit

# Exit code 5 = no tests collected (suites are empty until implemented)
test-integration:
	$(PYTEST) $(TESTS)/integration -m integration; status=$$?; [ $$status -eq 5 ] && exit 0 || exit $$status

test-e2e:
	$(PYTEST) $(TESTS)/e2e -m e2e; status=$$?; [ $$status -eq 5 ] && exit 0 || exit $$status

test-cov:
	$(PYTEST) $(TESTS) --cov=$(SRC) --cov-report=term-missing --cov-report=html:htmlcov

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -not -path './.venv/*' | xargs rm -rf
	find . -type f -name "*.pyc" -not -path './.venv/*' -delete
	find . -type f -name "*.pyo" -not -path './.venv/*' -delete
	rm -rf build/ dist/ .eggs/ src/*.egg-info
	rm -rf .pytest_cache/ .mypy_cache/ .ruff_cache/
	rm -rf htmlcov/ .coverage
