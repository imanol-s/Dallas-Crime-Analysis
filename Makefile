PYTHON ?= .venv/bin/python
PIP ?= .venv/bin/pip
CLI ?= .venv/bin/dallas-crime
PYTEST ?= .venv/bin/pytest

.PHONY: setup test smoke acquire build analyze

setup:
	uv venv
	$(PIP) install -e ".[dev]"

test:
	$(PYTEST) -q

acquire:
	$(CLI) acquire

build:
	$(CLI) build

analyze:
	$(CLI) analyze

smoke:
	TMP_PROJECT="$$(mktemp -d)"; \
	$(PYTHON) scripts/create_smoke_inputs.py "$${TMP_PROJECT}"; \
	$(CLI) show-config --project-root "$${TMP_PROJECT}"; \
	$(CLI) build --project-root "$${TMP_PROJECT}"; \
	$(CLI) analyze --project-root "$${TMP_PROJECT}"
